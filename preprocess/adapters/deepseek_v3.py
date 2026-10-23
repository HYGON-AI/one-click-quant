"""DeepSeek-V3 preprocess adapter (MLA MoE structure glue).

Weight layout (HF ``DeepseekV3ForCausalLM``)::

    model.embed_tokens / model.norm / lm_head
    model.layers.{0..first_k_dense_replace-1}.*     # dense MLP
    model.layers.{k..}.*                              # MoE: experts.{i}.* + shared_experts
    model.layers.{num_hidden_layers}.*                # MTP / next-n (eh_proj, shared_head, …)

Official DeepSeek-V3 checkpoints ship **block FP8**. Staging unpack lives in
``preprocess.unpack.fp8`` (1:1 shards, optional multi-GPU); this module only
owns LN fuse / QuaRot pairs / UpDown resolution.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from preprocess.adapters.base import ModelAdapter
from preprocess.unpack.fp8 import unpack_fp8_weights_inline

if TYPE_CHECKING:
    from preprocess.io import WeightStore
    from preprocess.transforms.base import PreprocessContext


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
        unpack_fp8_weights_inline(
            weights,
            cfg=ctx.text_config,
            layer_ids=ctx.layer_ids,
            rank=ctx.rank,
        )

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
