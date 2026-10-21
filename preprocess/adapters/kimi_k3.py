"""Kimi-K3 / Moonshot KDA+MoE adapter (QuaRot maps + subgraph partners + MXFP4 unpack).

Module naming for Kimi-K3 QuaRot / Flex maps; used by preprocess only.
MXFP4 expert packs (``weight_packed`` + E8M0 ``weight_scale``) are dequantized
to BF16 ``.weight`` before QuaRot / Flex transforms.
"""

from __future__ import annotations

import gc
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from preprocess.adapters.base import ModelAdapter
from preprocess.transforms.base import module_layer_id

if TYPE_CHECKING:
    from preprocess.io import WeightStore
    from preprocess.transforms.base import PreprocessContext

_MXFP4_BLOCK_SIZE = 32
_E8M0_BIAS = 127
_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def unpack_fp4_from_uint8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 MXFP4 nibbles to float32 element values (no block scale)."""
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    lut = torch.tensor(_E2M1_LUT, device=packed.device, dtype=torch.float32)
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 1).to(torch.float32)
    abs_idx = (nibbles & 0x07).long()
    return sign * lut[abs_idx]


def e8m0_uint8_to_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 uint8 to float32 scales ``2^(x-127)``."""
    if scale_u8.dtype != torch.uint8:
        scale_u8 = scale_u8.to(torch.uint8)
    return torch.exp2(scale_u8.to(torch.float32) - float(_E8M0_BIAS))


def dequant_mxfp4_to_bf16(
    packed: torch.Tensor,
    scale_u8: torch.Tensor,
    block_size: int = _MXFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize CT MXFP4 ``[N, K/2]`` + scale ``[N, K/32]`` → BF16 ``[N, K]``."""
    if packed.ndim != 2:
        raise ValueError(f"Expected 2D weight_packed, got {tuple(packed.shape)}")
    values = unpack_fp4_from_uint8(packed)
    scale = e8m0_uint8_to_scale(scale_u8)
    while scale.ndim > 2 and scale.shape[-1] == 1:
        scale = scale.squeeze(-1)
    if scale.ndim != 2:
        raise ValueError(f"Expected 2D weight_scale, got {tuple(scale.shape)}")
    n, k = values.shape
    if scale.shape[0] != n:
        raise ValueError(f"scale rows {scale.shape[0]} != weight rows {n}")
    if scale.shape[1] * block_size != k:
        raise ValueError(
            f"K={k} != n_blocks({scale.shape[1]}) * block_size({block_size})"
        )
    scale_expanded = scale.repeat_interleave(block_size, dim=-1)
    return (values * scale_expanded).to(torch.bfloat16)


def _is_kda_layer(cfg: Dict[str, Any], layer_idx: int) -> bool:
    lac = cfg.get("linear_attn_config") or {}
    kda_layers = lac.get("kda_layers") or []
    return (layer_idx + 1) in kda_layers


def _is_dense_mlp(cfg: Dict[str, Any], layer_idx: int) -> bool:
    return layer_idx < int(cfg.get("first_k_dense_replace", 1))


def _use_attn_residuals(cfg: Dict[str, Any]) -> bool:
    return cfg.get("attn_res_block_size") is not None


def _kda_use_full_rank_gate(cfg: Dict[str, Any]) -> bool:
    lac = cfg.get("linear_attn_config") or {}
    return bool(lac.get("use_full_rank_gate", False))


def _layer_prefix(i: int) -> str:
    return f"language_model.model.layers.{i}"


def _attn_prefix(i: int) -> str:
    return f"{_layer_prefix(i)}.self_attn"


def _moe_prefix(i: int) -> str:
    return f"{_layer_prefix(i)}.block_sparse_moe"


def _input_layernorm_targets(cfg: Dict[str, Any], layer_idx: int) -> List[str]:
    attn = _attn_prefix(layer_idx)
    if _is_kda_layer(cfg, layer_idx):
        targets = [
            f"{attn}.q_proj",
            f"{attn}.k_proj",
            f"{attn}.v_proj",
            f"{attn}.f_a_proj",
            f"{attn}.b_proj",
        ]
        targets.append(f"{attn}.g_proj" if _kda_use_full_rank_gate(cfg) else f"{attn}.g_a_proj")
        return targets
    targets = [f"{attn}.q_a_proj", f"{attn}.kv_a_proj_with_mqa"]
    if cfg.get("mla_use_output_gate", False):
        targets.append(f"{attn}.g_proj")
    return targets


def _hidden_rot_right(cfg: Dict[str, Any], layer_idx: int) -> List[str]:
    names = list(_input_layernorm_targets(cfg, layer_idx))
    prefix = _layer_prefix(layer_idx)
    if _is_dense_mlp(cfg, layer_idx):
        names.extend([f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"])
    else:
        moe = _moe_prefix(layer_idx)
        names.extend(
            [
                f"{moe}.gate",
                f"{moe}.shared_experts.gate_proj",
                f"{moe}.shared_experts.up_proj",
                f"{moe}.routed_expert_down_proj",
            ]
        )
    if _use_attn_residuals(cfg):
        names.extend([f"{prefix}.self_attention_res_proj", f"{prefix}.mlp_res_proj"])
    return names


def _hidden_rot_left(cfg: Dict[str, Any], layer_idx: int) -> List[str]:
    names = [f"{_attn_prefix(layer_idx)}.o_proj"]
    prefix = _layer_prefix(layer_idx)
    if _is_dense_mlp(cfg, layer_idx):
        names.append(f"{prefix}.mlp.down_proj")
    else:
        moe = _moe_prefix(layer_idx)
        names.extend([f"{moe}.shared_experts.down_proj", f"{moe}.routed_expert_up_proj"])
    return names


class KimiK3Adapter(ModelAdapter):
    """Kimi-K3 HF weight names under ``language_model.model``."""

    name = "kimi_k3"

    def prepare_weights(self, weights: "WeightStore", ctx: "PreprocessContext") -> None:
        """Inline MXFP4 unpack is disabled by default.

        Full-model unpack must be done **streaming to disk** first
        (``unpack_mxfp4_checkpoint``); keeping thousands of BF16 experts in
        ``WeightStore`` cache OOMs (mmap Cannot allocate memory).

        Set recipe ``unpack_mxfp4_inline: true`` only for tiny smoke models.
        """
        if ctx.extras.get("skip_inline_unpack"):
            return
        if not ctx.recipe.get("unpack_mxfp4_inline", False):
            packed = sum(1 for n in weights.names() if n.endswith(".weight_packed"))
            if packed:
                raise RuntimeError(
                    f"Found {packed} MXFP4 weight_packed tensors but inline unpack is off. "
                    "Pipeline should call unpack_mxfp4_checkpoint() to a staging dir first. "
                    "Or set unpack_mxfp4_inline: true for tiny models only."
                )
            return
        n = self.unpack_mxfp4_weights(
            weights,
            layer_ids=ctx.layer_ids,
            device=ctx.device,
            rank=ctx.rank,
        )
        ctx.extras["unpacked_mxfp4"] = n

    def unpack_mxfp4_weights(
        self,
        weights: "WeightStore",
        *,
        layer_ids: Optional[Set[int]] = None,
        device: torch.device | str = "cpu",
        rank: int = 0,
        flush_every: int = 32,
    ) -> int:
        """In-memory unpack (smoke only). Prefer ``unpack_mxfp4_checkpoint``."""
        device = torch.device(device)
        packed_keys = [n for n in weights.names() if n.endswith(".weight_packed")]
        if not packed_keys:
            logger.info("kimi_k3 unpack: no weight_packed tensors (already float?)")
            return 0

        done = 0
        for packed_name in packed_keys:
            module = packed_name[: -len(".weight_packed")]
            lid = module_layer_id(module)
            if layer_ids is not None:
                if lid is None:
                    if rank != 0:
                        continue
                elif lid not in layer_ids:
                    continue

            scale_name = f"{module}.weight_scale"
            if not weights.has(scale_name):
                logger.warning("kimi_k3 unpack: missing {}; skip {}", scale_name, packed_name)
                continue

            shard = weights.weight_map[packed_name]
            packed = weights.get(packed_name, device="cpu")
            scale = weights.get(scale_name, device="cpu")
            # Drop mmap-backed cache entries immediately.
            weights._cache.pop(packed_name, None)
            weights._cache.pop(scale_name, None)
            try:
                w_bf16 = dequant_mxfp4_to_bf16(packed, scale).cpu()
            except ValueError as exc:
                logger.warning("kimi_k3 unpack failed for {}: {}", module, exc)
                continue
            del packed, scale

            weight_name = f"{module}.weight"
            weights.set(weight_name, w_bf16, shard=shard)
            weights.delete(packed_name)
            weights.delete(scale_name)
            done += 1
            if done <= 3 or done % 500 == 0:
                logger.info(
                    "kimi_k3 unpack #{} {} -> BF16 {}",
                    done,
                    packed_name,
                    tuple(w_bf16.shape),
                )
            del w_bf16
            if flush_every and done % flush_every == 0:
                gc.collect()

        logger.info(
            "kimi_k3 unpack: dequantized {} MXFP4 modules to BF16 (rank={}, layers={})",
            done,
            rank,
            sorted(layer_ids) if layer_ids is not None else "all",
        )
        return done

    @staticmethod
    def model_needs_mxfp4_unpack(model_dir: str | Path) -> bool:
        index = Path(model_dir) / "model.safetensors.index.json"
        if not index.is_file():
            return False
        wm = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
        return any(k.endswith(".weight_packed") for k in wm)

    @staticmethod
    def unpack_mxfp4_checkpoint(
        src_dir: str | Path,
        dst_dir: str | Path,
        *,
        batch_packs: int = 48,
    ) -> Path:
        """Stream MXFP4→BF16 to ``dst_dir`` one shard at a time (low memory).

        Large shards are split into multiple ``*.uXXX.safetensors`` parts so we
        never hold an entire unpacked shard in RAM. Safe for single-process use
        before multi-GPU transforms.
        """
        src = Path(src_dir).resolve()
        dst = Path(dst_dir).resolve()
        if src == dst:
            raise ValueError("unpack_mxfp4_checkpoint: src and dst must differ")
        marker = dst / "UNPACK_MXFP4_DONE.json"
        if marker.is_file() and (dst / "model.safetensors.index.json").is_file():
            logger.info("kimi_k3 unpack: reuse existing staging {}", dst)
            return dst

        index_path = src / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing index under {src}")
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: Dict[str, str] = dict(idx.get("weight_map") or {})
        by_shard: Dict[str, List[str]] = defaultdict(list)
        for name, shard in weight_map.items():
            by_shard[shard].append(name)

        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        # Copy non-weight sidecars.
        for item in src.iterdir():
            if item.name.startswith(".") or item.suffix == ".safetensors":
                continue
            if item.name == "model.safetensors.index.json":
                continue
            if item.is_file():
                shutil.copy2(item, dst / item.name)

        new_wm: Dict[str, str] = {}
        total_unpacked = 0
        shards = sorted(by_shard.keys())
        logger.info(
            "kimi_k3 streaming unpack: {} shards {} -> {} (batch_packs={})",
            len(shards),
            src,
            dst,
            batch_packs,
        )

        for si, shard in enumerate(shards):
            keys = by_shard[shard]
            packed_mods = {
                k[: -len(".weight_packed")] for k in keys if k.endswith(".weight_packed")
            }
            if not packed_mods:
                # Pure copy (attention / shared bf16 shards).
                shutil.copy2(src / shard, dst / shard)
                for k in keys:
                    new_wm[k] = shard
                logger.info("[{}/{}] copy {}", si + 1, len(shards), shard)
                continue

            part_idx = 0
            batch: Dict[str, torch.Tensor] = {}
            packs_in_batch = 0

            def flush_batch() -> None:
                nonlocal part_idx, packs_in_batch, batch
                if not batch:
                    return
                out_name = f"{Path(shard).stem}.u{part_idx:03d}.safetensors"
                save_file(batch, str(dst / out_name))
                for k in batch:
                    new_wm[k] = out_name
                logger.info(
                    "[{}/{}] wrote {} ({} tensors, {} packs)",
                    si + 1,
                    len(shards),
                    out_name,
                    len(batch),
                    packs_in_batch,
                )
                part_idx += 1
                packs_in_batch = 0
                batch = {}
                gc.collect()

            with safe_open(src / shard, framework="pt", device="cpu") as f:
                # Pass 1: non-pack tensors (and skip scales belonging to packs).
                for k in keys:
                    if k.endswith(".weight_packed"):
                        continue
                    if k.endswith(".weight_scale"):
                        mod = k[: -len(".weight_scale")]
                        if mod in packed_mods:
                            continue
                    batch[k] = f.get_tensor(k)
                    if len(batch) >= 256:
                        flush_batch()

                # Pass 2: packs → BF16 .weight in small batches.
                for k in keys:
                    if not k.endswith(".weight_packed"):
                        continue
                    mod = k[: -len(".weight_packed")]
                    scale_k = f"{mod}.weight_scale"
                    packed = f.get_tensor(k)
                    scale = f.get_tensor(scale_k)
                    w = dequant_mxfp4_to_bf16(packed, scale).contiguous()
                    del packed, scale
                    batch[f"{mod}.weight"] = w
                    packs_in_batch += 1
                    total_unpacked += 1
                    if packs_in_batch >= batch_packs:
                        flush_batch()
                flush_batch()

            logger.info(
                "[{}/{}] done {} (unpacked running total={})",
                si + 1,
                len(shards),
                shard,
                total_unpacked,
            )

        # Rewrite config without compression markers.
        cfg_path = src / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg.pop("compression_config", None)
            cfg.pop("quantization_config", None)
            if isinstance(cfg.get("text_config"), dict):
                cfg["text_config"] = dict(cfg["text_config"])
                cfg["text_config"].pop("compression_config", None)
                cfg["text_config"].pop("quantization_config", None)
            cfg["kimi_mxfp4_unpacked"] = True
            (dst / "config.json").write_text(
                json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

        # total_size from headers
        from preprocess.io import _safetensors_header_nbytes

        total = 0
        for sh in sorted(set(new_wm.values())):
            p = dst / sh
            if p.is_file():
                total += _safetensors_header_nbytes(p)
        out_index = {
            "metadata": {**(idx.get("metadata") or {}), "total_size": total},
            "weight_map": new_wm,
        }
        (dst / "model.safetensors.index.json").write_text(
            json.dumps(out_index, indent=2) + "\n", encoding="utf-8"
        )
        marker.write_text(
            json.dumps(
                {"status": "ok", "source": str(src), "unpacked_modules": total_unpacked},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "kimi_k3 streaming unpack finished: {} modules -> {} (total_size={:.2f} GiB)",
            total_unpacked,
            dst,
            total / 1024**3,
        )
        return dst

    def build_ln_fuse_map(
        self,
        cfg: Dict[str, Any],
        num_layers: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        if num_layers is None:
            num_layers = int(cfg["num_hidden_layers"])
        ln_map: Dict[str, List[str]] = {}
        for i in range(num_layers):
            prefix = _layer_prefix(i)
            attn = _attn_prefix(i)
            ln_map[f"{prefix}.input_layernorm"] = _input_layernorm_targets(cfg, i)
            if not _is_kda_layer(cfg, i):
                ln_map[f"{attn}.q_a_layernorm"] = [f"{attn}.q_b_proj"]
                ln_map[f"{attn}.kv_a_layernorm"] = [f"{attn}.kv_b_proj"]
            if _is_dense_mlp(cfg, i):
                ln_map[f"{prefix}.post_attention_layernorm"] = [
                    f"{prefix}.mlp.gate_proj",
                    f"{prefix}.mlp.up_proj",
                ]
            else:
                moe = _moe_prefix(i)
                ln_map[f"{prefix}.post_attention_layernorm"] = [
                    f"{moe}.gate",
                    f"{moe}.shared_experts.gate_proj",
                    f"{moe}.shared_experts.up_proj",
                    f"{moe}.routed_expert_down_proj",
                ]
                if cfg.get("latent_moe_use_norm", False):
                    ln_map[f"{moe}.routed_expert_norm"] = [f"{moe}.routed_expert_up_proj"]
            if _use_attn_residuals(cfg):
                ln_map[f"{prefix}.self_attention_res_norm"] = [f"{prefix}.self_attention_res_proj"]
                ln_map[f"{prefix}.mlp_res_norm"] = [f"{prefix}.mlp_res_proj"]
        ln_map["language_model.model.norm"] = ["language_model.lm_head"]
        if _use_attn_residuals(cfg):
            ln_map["language_model.model.output_attn_res_norm"] = [
                "language_model.model.output_attn_res_proj"
            ]
        return ln_map

    def build_rotate_pairs(
        self,
        cfg: Dict[str, Any],
        *,
        block_size: int = 32,
        num_layers: Optional[int] = None,
        num_experts: Optional[int] = None,
    ) -> Tuple[List[Tuple[Dict[str, str], Dict[str, str]]], Dict[str, Any]]:
        if num_layers is None:
            num_layers = int(cfg["num_hidden_layers"])
        if num_experts is None:
            num_experts = int(cfg.get("num_experts", 0) or 0)

        hidden = int(cfg["hidden_size"])
        q_lora = int(cfg.get("q_lora_rank") or 0)
        kv_lora = int(cfg.get("kv_lora_rank") or 0)
        qk_rope = int(cfg.get("qk_rope_head_dim") or 0)
        h_lat = cfg.get("routed_expert_hidden_size")

        meta: Dict[str, Any] = {
            "rot": {"size": hidden, "mode": "hadamard", "block_size": block_size},
            "rot_b_proj": {
                "size": q_lora,
                "mode": "block_hadamard_shifted",
                "block_size": block_size,
            },
            "rot_kv": {"size": kv_lora, "mode": "hadamard", "block_size": block_size},
            "qk_rope": qk_rope,
        }
        if h_lat is not None:
            meta["rot_latent"] = {
                "size": int(h_lat),
                "mode": "hadamard",
                "block_size": block_size,
            }

        pairs: List[Tuple[Dict[str, str], Dict[str, str]]] = []

        pre_left: Dict[str, str] = {"mm_projector.rot_proj": "rot"}
        pre_right: Dict[str, str] = {
            "language_model.model.embed_tokens": "rot",
            "language_model.lm_head": "rot",
        }
        if _use_attn_residuals(cfg):
            pre_right["language_model.model.output_attn_res_proj"] = "rot"
        pairs.append((pre_left, pre_right))

        left: Dict[str, str] = {}
        right: Dict[str, str] = {}
        for i in range(num_layers):
            for name in _hidden_rot_right(cfg, i):
                right[name] = "rot"
            for name in _hidden_rot_left(cfg, i):
                left[name] = "rot"
        pairs.append((left, right))

        if q_lora:
            lb: Dict[str, str] = {}
            rb: Dict[str, str] = {}
            for i in range(num_layers):
                if _is_kda_layer(cfg, i):
                    continue
                attn = _attn_prefix(i)
                lb[f"{attn}.q_a_proj"] = "rot_b_proj"
                rb[f"{attn}.q_b_proj"] = "rot_b_proj"
            pairs.append((lb, rb))

        if kv_lora:
            lk: Dict[str, str] = {}
            rk: Dict[str, str] = {}
            for i in range(num_layers):
                if _is_kda_layer(cfg, i):
                    continue
                attn = _attn_prefix(i)

                lk[f"{attn}.kv_a_proj_with_mqa"] = "rot_kv_block"
                rk[f"{attn}.kv_b_proj"] = "rot_kv"
            pairs.append((lk, rk))

        if h_lat is not None and num_experts > 0:
            ll: Dict[str, str] = {}
            rl: Dict[str, str] = {}
            for i in range(num_layers):
                if _is_dense_mlp(cfg, i):
                    continue
                moe = _moe_prefix(i)
                ll[f"{moe}.routed_expert_down_proj"] = "rot_latent"
                rl[f"{moe}.routed_expert_up_proj"] = "rot_latent"
                for e in range(num_experts):
                    expert = f"{moe}.experts.{e}"
                    rl[f"{expert}.w1"] = "rot_latent"
                    rl[f"{expert}.w3"] = "rot_latent"
                    ll[f"{expert}.w2"] = "rot_latent"
            pairs.append((ll, rl))

        return pairs, meta

    def resolve_up_down(self, up_module: str) -> Optional[str]:
        """Routed expert ``w3→w2``; dense/shared ``up_proj→down_proj``."""
        m = re.match(r"(.*\.experts\.\d+)\.w3$", up_module)
        if m:
            return f"{m.group(1)}.w2"
        m = re.match(r"(.*)\.up_proj$", up_module)
        if m:
            return f"{m.group(1)}.down_proj"
        return None
