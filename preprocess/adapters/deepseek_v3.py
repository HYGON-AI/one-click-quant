"""DeepSeek-V3 preprocess adapter (MLA MoE + FP8 block unpack).

Weight layout (HF ``DeepseekV3ForCausalLM``)::

    model.embed_tokens / model.norm / lm_head
    model.layers.{0..first_k_dense_replace-1}.*     # dense MLP
    model.layers.{k..}.*                              # MoE: experts.{i}.* + shared_experts
    model.layers.{num_hidden_layers}.*                # MTP / next-n (eh_proj, shared_head, …)

Official DeepSeek-V3 checkpoints ship **block FP8** (``float8_e4m3fn`` +
``weight_scale_inv``, block 128×128). Streaming unpack writes BF16 staging
before QuaRot / Flex (same role as Kimi MXFP4 unpack).
"""

from __future__ import annotations

import gc
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from preprocess.adapters.base import ModelAdapter
from preprocess.transforms.base import module_layer_id

if TYPE_CHECKING:
    from preprocess.io import WeightStore
    from preprocess.transforms.base import PreprocessContext

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
    torch.float8_e5m2,
    getattr(torch, "float8_e5m2fnuz", torch.float8_e5m2),
)
_FP8_GROUP_SIZE = 128


def dequantize_weight_from_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    group_size: int = _FP8_GROUP_SIZE,
) -> torch.Tensor:
    """DeepSeek-style block FP8 → BF16 (``W * scale_inv``, block ``group_size``)."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2D FP8 weight, got {tuple(weight.shape)}")
    g = int(group_size)
    d_out, d_in = weight.shape
    pad_out = math.ceil(d_out / g) * g - d_out
    pad_in = math.ceil(d_in / g) * g - d_in
    w = F.pad(weight.float(), (0, pad_in, 0, pad_out))
    s = scale_inv.float()
    d_out_pad, d_in_pad = w.shape
    expected = (d_out_pad // g, d_in_pad // g)
    if tuple(s.shape) != expected:
        # Some dumps store [n_blocks_out, n_blocks_in] already; accept close forms.
        if s.numel() != expected[0] * expected[1]:
            raise ValueError(
                f"FP8 scale shape {tuple(s.shape)} incompatible with weight "
                f"{tuple(weight.shape)} group={g} (expected blocks {expected})"
            )
        s = s.reshape(expected)
    w = w.view(d_out_pad // g, g, d_in_pad // g, g)
    s = s.view(d_out_pad // g, 1, d_in_pad // g, 1)
    out = (w * s).view(d_out_pad, d_in_pad)[:d_out, :d_in]
    return out.to(torch.bfloat16)


def _is_dense_mlp(cfg: Dict[str, Any], layer_idx: int) -> bool:
    return layer_idx < int(cfg.get("first_k_dense_replace", 0) or 0)


def _n_routed(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("n_routed_experts") or cfg.get("num_experts") or 0)


def _layer_prefix(i: int) -> str:
    return f"model.layers.{i}"


def _attn_prefix(i: int) -> str:
    return f"{_layer_prefix(i)}.self_attn"


def _mtp_layer_idx(cfg: Dict[str, Any]) -> Optional[int]:
    n_next = int(cfg.get("num_nextn_predict_layers") or 0)
    if n_next <= 0:
        return None
    # Official V3: main layers [0, num_hidden_layers), MTP at num_hidden_layers.
    return int(cfg["num_hidden_layers"])


def _decoder_layer_indices(cfg: Dict[str, Any]) -> List[int]:
    ids = list(range(int(cfg["num_hidden_layers"])))
    mtp = _mtp_layer_idx(cfg)
    if mtp is not None:
        ids.append(mtp)
    return ids


def _input_ln_targets(layer_idx: int) -> List[str]:
    attn = _attn_prefix(layer_idx)
    return [f"{attn}.q_a_proj", f"{attn}.kv_a_proj_with_mqa"]


def _post_attn_ln_targets(cfg: Dict[str, Any], layer_idx: int) -> List[str]:
    prefix = _layer_prefix(layer_idx)
    if _is_dense_mlp(cfg, layer_idx):
        return [f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"]
    n = _n_routed(cfg)
    names = [
        f"{prefix}.mlp.gate",
        f"{prefix}.mlp.shared_experts.gate_proj",
        f"{prefix}.mlp.shared_experts.up_proj",
    ]
    for e in range(n):
        names.append(f"{prefix}.mlp.experts.{e}.gate_proj")
        names.append(f"{prefix}.mlp.experts.{e}.up_proj")
    return names


def _fp8_group_size(cfg: Dict[str, Any]) -> int:
    qc = cfg.get("quantization_config") or {}
    block = qc.get("weight_block_size")
    if isinstance(block, (list, tuple)) and block:
        return int(block[0])
    return _FP8_GROUP_SIZE


class DeepSeekV3Adapter(ModelAdapter):
    """Adapter for DeepSeek-V3 (``model_type`` ``deepseek_v3``)."""

    name = "deepseek_v3"

    def norm_affine(self) -> str:
        return "rms"

    def prepare_weights(self, weights: "WeightStore", ctx: "PreprocessContext") -> None:
        if ctx.extras.get("skip_inline_unpack"):
            return
        if not ctx.recipe.get("unpack_fp8_inline", False):
            n_scale = sum(1 for n in weights.names() if n.endswith(".weight_scale_inv"))
            if n_scale:
                raise RuntimeError(
                    f"Found {n_scale} FP8 weight_scale_inv tensors but inline unpack is off. "
                    "Pipeline should call unpack_fp8_checkpoint() to a staging dir first. "
                    "Or set unpack_fp8_inline: true for tiny smoke models."
                )
            return
        self.unpack_fp8_weights(
            weights,
            cfg=ctx.text_config,
            layer_ids=ctx.layer_ids,
            rank=ctx.rank,
        )

    def unpack_fp8_weights(
        self,
        weights: "WeightStore",
        *,
        cfg: Dict[str, Any],
        layer_ids: Optional[Set[int]] = None,
        rank: int = 0,
        flush_every: int = 64,
    ) -> int:
        """In-memory FP8 unpack (smoke only). Prefer ``unpack_fp8_checkpoint``."""
        g = _fp8_group_size(cfg)
        scale_keys = [n for n in weights.names() if n.endswith(".weight_scale_inv")]
        done = 0
        for scale_name in scale_keys:
            weight_name = scale_name[: -len("_scale_inv")]
            if not weight_name.endswith(".weight"):
                continue
            module = weight_name[: -len(".weight")]
            lid = module_layer_id(module)
            if layer_ids is not None:
                if lid is None:
                    if rank != 0:
                        continue
                elif lid not in layer_ids:
                    continue
            if not weights.has(weight_name):
                logger.warning("deepseek_v3 unpack: missing {}; skip {}", weight_name, scale_name)
                continue
            shard = weights.weight_map.get(weight_name)
            w = weights.get(weight_name, device="cpu")
            s = weights.get(scale_name, device="cpu")
            weights._cache.pop(weight_name, None)
            weights._cache.pop(scale_name, None)
            if w.dtype not in _FP8_DTYPES:
                del w, s
                continue
            try:
                bf16 = dequantize_weight_from_fp8(w, s, group_size=g).cpu()
            except ValueError as exc:
                logger.warning("deepseek_v3 unpack failed for {}: {}", module, exc)
                continue
            del w, s
            weights.set(weight_name, bf16, shard=shard)
            weights.delete(scale_name)
            done += 1
            if done <= 3 or done % 1000 == 0:
                logger.info(
                    "deepseek_v3 unpack #{} {} -> BF16 {}",
                    done,
                    weight_name,
                    tuple(bf16.shape),
                )
            del bf16
            if flush_every and done % flush_every == 0:
                gc.collect()
        logger.info(
            "deepseek_v3 unpack: dequantized {} FP8 modules (rank={}, layers={})",
            done,
            rank,
            sorted(layer_ids) if layer_ids is not None else "all",
        )
        return done

    @staticmethod
    def model_needs_fp8_unpack(model_dir: str | Path) -> bool:
        index = Path(model_dir) / "model.safetensors.index.json"
        if not index.is_file():
            return False
        wm = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
        if any(k.endswith(".weight_scale_inv") for k in wm):
            return True
        cfg_path = Path(model_dir) / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            qc = cfg.get("quantization_config") or {}
            if str(qc.get("quant_method", "")).lower() == "fp8":
                return True
        return False

    @staticmethod
    def unpack_fp8_checkpoint(
        src_dir: str | Path,
        dst_dir: str | Path,
        *,
        batch_tensors: int = 64,
        group_size: Optional[int] = None,
    ) -> Path:
        """Stream block-FP8 → BF16 to ``dst_dir`` (low memory, before multi-GPU)."""
        src = Path(src_dir).resolve()
        dst = Path(dst_dir).resolve()
        if src == dst:
            raise ValueError("unpack_fp8_checkpoint: src and dst must differ")
        marker = dst / "UNPACK_FP8_DONE.json"
        if marker.is_file() and (dst / "model.safetensors.index.json").is_file():
            logger.info("deepseek_v3 unpack: reuse existing staging {}", dst)
            return dst

        index_path = src / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing index under {src}")
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: Dict[str, str] = dict(idx.get("weight_map") or {})
        cfg: Dict[str, Any] = {}
        cfg_path = src / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        g = int(group_size or _fp8_group_size(cfg))

        by_shard: Dict[str, List[str]] = defaultdict(list)
        for name, shard in weight_map.items():
            by_shard[shard].append(name)

        dst.mkdir(parents=True, exist_ok=True)
        # Resume: keep existing .uXXX / copied shards instead of wiping ~1T staging.
        existing_parts = list(dst.glob("*.safetensors"))
        resume = bool(existing_parts)
        if resume:
            logger.info(
                "deepseek_v3 unpack: resume staging {} ({} existing safetensors)",
                dst,
                len(existing_parts),
            )

        for item in src.iterdir():
            if item.name.startswith(".") or item.suffix == ".safetensors":
                continue
            if item.name == "model.safetensors.index.json":
                continue
            if item.is_file():
                target = dst / item.name
                if not target.is_file():
                    shutil.copy2(item, target)

        new_wm: Dict[str, str] = {}
        total_unpacked = 0
        missing_scales: List[str] = []
        raw_fp8_copied: List[str] = []
        shards = sorted(by_shard.keys())
        logger.info(
            "deepseek_v3 streaming FP8 unpack: {} shards {} -> {} (group={}, batch={}, resume={})",
            len(shards),
            src,
            dst,
            g,
            batch_tensors,
            resume,
        )

        def _collect_existing_parts(stem: str) -> List[Path]:
            parts = sorted(dst.glob(f"{stem}.u*.safetensors"))
            copied = dst / f"{stem}.safetensors"
            if copied.is_file():
                return [copied]
            return parts

        def _ingest_existing(paths: List[Path]) -> int:
            n_keys = 0
            for p in paths:
                with safe_open(p, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        new_wm[k] = p.name
                        n_keys += 1
            return n_keys

        for si, shard in enumerate(shards):
            keys = by_shard[shard]
            stem = Path(shard).stem
            existing = _collect_existing_parts(stem)
            if existing:
                n_keys = _ingest_existing(existing)
                # Treat a shard as done only when we recovered every non-scale key.
                expected = {
                    k
                    for k in keys
                    if not k.endswith(".weight_scale_inv")
                }
                have = expected & set(new_wm)
                if have == expected:
                    logger.info(
                        "[{}/{}] resume skip {} ({} files, {} tensors)",
                        si + 1,
                        len(shards),
                        shard,
                        len(existing),
                        n_keys,
                    )
                    continue
                # Partial shard: drop incomplete parts and re-unpack this shard.
                logger.warning(
                    "[{}/{}] incomplete {} ({}/{} keys); re-unpack",
                    si + 1,
                    len(shards),
                    shard,
                    len(have),
                    len(expected),
                )
                for p in existing:
                    p.unlink(missing_ok=True)
                    for k, s in list(new_wm.items()):
                        if s == p.name:
                            new_wm.pop(k, None)

            fp8_mods = {
                k[: -len(".weight")]
                for k in keys
                if k.endswith(".weight") and f"{k}_scale_inv" in weight_map
            }
            # Also detect scales present in this shard pointing at weights.
            for k in keys:
                if k.endswith(".weight_scale_inv"):
                    wname = k[: -len("_scale_inv")]
                    if wname.endswith(".weight"):
                        fp8_mods.add(wname[: -len(".weight")])

            if not fp8_mods:
                shutil.copy2(src / shard, dst / shard)
                for k in keys:
                    new_wm[k] = shard
                logger.info("[{}/{}] copy {}", si + 1, len(shards), shard)
                continue

            part_idx = 0
            batch: Dict[str, torch.Tensor] = {}
            unpacked_in_batch = 0

            def flush_batch() -> None:
                nonlocal part_idx, unpacked_in_batch, batch
                if not batch:
                    return
                out_name = f"{Path(shard).stem}.u{part_idx:03d}.safetensors"
                save_file(batch, str(dst / out_name))
                for k in batch:
                    new_wm[k] = out_name
                logger.info(
                    "[{}/{}] wrote {} ({} tensors, {} fp8→bf16)",
                    si + 1,
                    len(shards),
                    out_name,
                    len(batch),
                    unpacked_in_batch,
                )
                part_idx += 1
                unpacked_in_batch = 0
                batch = {}
                gc.collect()

            # Scales may live in another shard; open per-key as needed.
            with safe_open(src / shard, framework="pt", device="cpu") as f:
                local_keys = set(f.keys())
                for k in keys:
                    if k.endswith(".weight_scale_inv"):
                        continue
                    if k.endswith(".weight"):
                        mod = k[: -len(".weight")]
                        if mod in fp8_mods:
                            continue
                    if k not in local_keys:
                        continue
                    t = f.get_tensor(k)
                    # FP8 weight without a registered scale must not be written.
                    # Writing it would let a later resume treat the shard as complete
                    # and stamp deepseek_fp8_unpacked=True.
                    if k.endswith(".weight") and t.dtype in _FP8_DTYPES:
                        raw_fp8_copied.append(k)
                        raise RuntimeError(
                            "deepseek_v3 unpack: FP8 weight "
                            f"{k} has no weight_scale_inv in the index; "
                            "refusing to copy raw FP8 or mark unpack complete."
                        )
                    batch[k] = t
                    if len(batch) >= 256:
                        flush_batch()

                for mod in sorted(fp8_mods):
                    w_key = f"{mod}.weight"
                    s_key = f"{mod}.weight_scale_inv"
                    if w_key not in local_keys:
                        continue
                    if s_key not in weight_map:
                        missing_scales.append(mod)
                        raise RuntimeError(
                            "deepseek_v3 unpack: missing weight_scale_inv for "
                            f"{w_key}; refusing to copy raw FP8 or mark unpack complete."
                        )
                    w = f.get_tensor(w_key)
                    scale_shard = weight_map[s_key]
                    if scale_shard == shard and s_key in local_keys:
                        s = f.get_tensor(s_key)
                    else:
                        with safe_open(src / scale_shard, framework="pt", device="cpu") as fs:
                            s = fs.get_tensor(s_key)
                    if w.dtype not in _FP8_DTYPES:
                        batch[w_key] = w
                        continue
                    bf16 = dequantize_weight_from_fp8(w, s, group_size=g).contiguous()
                    del w, s
                    batch[w_key] = bf16
                    unpacked_in_batch += 1
                    total_unpacked += 1
                    if unpacked_in_batch >= batch_tensors:
                        flush_batch()
                flush_batch()

            logger.info(
                "[{}/{}] done {} (unpacked running total={})",
                si + 1,
                len(shards),
                shard,
                total_unpacked,
            )

        if missing_scales or raw_fp8_copied:
            examples = (missing_scales[:5] + raw_fp8_copied[:5])[:8]
            raise RuntimeError(
                "deepseek_v3 FP8 unpack incomplete: "
                f"missing_scale_modules={len(missing_scales)}, "
                f"raw_fp8_weights={len(raw_fp8_copied)} "
                f"(e.g. {examples}). Refusing to mark unpack complete."
            )

        if cfg_path.is_file():
            cfg_out = dict(cfg)
            cfg_out.pop("quantization_config", None)
            cfg_out["deepseek_fp8_unpacked"] = True
            (dst / "config.json").write_text(
                json.dumps(cfg_out, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

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
                {
                    "status": "ok",
                    "source": str(src),
                    "unpacked_modules": total_unpacked,
                    "group_size": g,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "deepseek_v3 streaming FP8 unpack finished: {} modules -> {} ({:.2f} GiB)",
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
        del num_layers
        ln_map: Dict[str, List[str]] = {}
        for i in _decoder_layer_indices(cfg):
            prefix = _layer_prefix(i)
            attn = _attn_prefix(i)
            ln_map[f"{prefix}.input_layernorm"] = _input_ln_targets(i)
            ln_map[f"{attn}.q_a_layernorm"] = [f"{attn}.q_b_proj"]
            ln_map[f"{attn}.kv_a_layernorm"] = [f"{attn}.kv_b_proj"]
            ln_map[f"{prefix}.post_attention_layernorm"] = _post_attn_ln_targets(cfg, i)
        ln_map["model.norm"] = ["lm_head"]
        mtp = _mtp_layer_idx(cfg)
        if mtp is not None:
            # enorm/hnorm feed concatenated eh_proj — not a single Norm→Linear fuse,
            # so eh_proj is also excluded from QuaRot (see build_rotate_pairs).
            ln_map[f"model.layers.{mtp}.shared_head.norm"] = [
                f"model.layers.{mtp}.shared_head.head"
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
        del num_layers
        if num_experts is None:
            num_experts = _n_routed(cfg)
        hidden = int(cfg["hidden_size"])
        q_lora = int(cfg.get("q_lora_rank") or 0)
        kv_lora = int(cfg.get("kv_lora_rank") or 0)
        qk_rope = int(cfg.get("qk_rope_head_dim") or 0)

        meta: Dict[str, Any] = {
            "rot": {"size": hidden, "mode": "hadamard", "block_size": block_size},
            "qk_rope": qk_rope,
        }
        if q_lora:
            meta["rot_b_proj"] = {
                "size": q_lora,
                "mode": "block_hadamard_shifted",
                "block_size": block_size,
            }
        if kv_lora:
            meta["rot_kv"] = {
                "size": kv_lora,
                "mode": "hadamard",
                "block_size": block_size,
            }

        pairs: List[Tuple[Dict[str, str], Dict[str, str]]] = []
        mtp = _mtp_layer_idx(cfg)

        pre_left: Dict[str, str] = {}
        pre_right: Dict[str, str] = {
            "model.embed_tokens": "rot",
            "lm_head": "rot",
        }
        if mtp is not None:
            pre_right[f"model.layers.{mtp}.embed_tokens"] = "rot"
            pre_right[f"model.layers.{mtp}.shared_head.head"] = "rot"
            # Do NOT rotate eh_proj: its inputs are enorm(embed) ‖ hnorm(hidden).
            # Those dual RMSNorm γ cannot be fused into a single Norm→Linear absorb
            # (see build_ln_fuse_map), and diagonal γ does not commute with Hadamard.
            # Rotating eh_proj alone would silently break MTP / speculative decoding.
        pairs.append((pre_left, pre_right))

        left: Dict[str, str] = {}
        right: Dict[str, str] = {}
        for i in _decoder_layer_indices(cfg):
            attn = _attn_prefix(i)
            prefix = _layer_prefix(i)
            right[f"{attn}.q_a_proj"] = "rot"
            right[f"{attn}.kv_a_proj_with_mqa"] = "rot"
            left[f"{attn}.o_proj"] = "rot"
            if _is_dense_mlp(cfg, i):
                right[f"{prefix}.mlp.gate_proj"] = "rot"
                right[f"{prefix}.mlp.up_proj"] = "rot"
                left[f"{prefix}.mlp.down_proj"] = "rot"
            else:
                right[f"{prefix}.mlp.gate"] = "rot"
                right[f"{prefix}.mlp.shared_experts.gate_proj"] = "rot"
                right[f"{prefix}.mlp.shared_experts.up_proj"] = "rot"
                left[f"{prefix}.mlp.shared_experts.down_proj"] = "rot"
                for e in range(num_experts):
                    right[f"{prefix}.mlp.experts.{e}.gate_proj"] = "rot"
                    right[f"{prefix}.mlp.experts.{e}.up_proj"] = "rot"
                    left[f"{prefix}.mlp.experts.{e}.down_proj"] = "rot"
        pairs.append((left, right))

        if q_lora:
            lb: Dict[str, str] = {}
            rb: Dict[str, str] = {}
            for i in _decoder_layer_indices(cfg):
                attn = _attn_prefix(i)
                lb[f"{attn}.q_a_proj"] = "rot_b_proj"
                rb[f"{attn}.q_b_proj"] = "rot_b_proj"
            pairs.append((lb, rb))

        if kv_lora:
            lk: Dict[str, str] = {}
            rk: Dict[str, str] = {}
            for i in _decoder_layer_indices(cfg):
                attn = _attn_prefix(i)
                lk[f"{attn}.kv_a_proj_with_mqa"] = "rot_kv_block"
                rk[f"{attn}.kv_b_proj"] = "rot_kv"
            pairs.append((lk, rk))

        return pairs, meta

    def resolve_up_down(self, up_module: str) -> Optional[str]:
        m = re.match(r"(.*\.mlp\.shared_experts)\.up_proj$", up_module)
        if m:
            return f"{m.group(1)}.down_proj"
        m = re.match(r"(.*\.mlp\.experts\.\d+)\.up_proj$", up_module)
        if m:
            return f"{m.group(1)}.down_proj"
        m = re.match(r"(.*\.mlp)\.up_proj$", up_module)
        if m:
            return f"{m.group(1)}.down_proj"
        return None
