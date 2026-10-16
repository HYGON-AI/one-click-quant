"""Qwen3.8-Flash-Next / qwen4_exp preprocess adapter.

Weight layout (HF ``Qwen4ExpForConditionalGeneration``)::

    model.language_model.layers.{i}....
    model.language_model.embed_tokens
    model.language_model.hyper_connection_mixer.*
    mtp.layers.0.*
    mtp.hyper_connection_mixer.*
    mtp.fc_embedding / mtp.fc_hidden / mtp.pre_fc_norm_*
    lm_head

Already BF16 (no MXFP4 unpack). Routed experts are fused 3D tensors
without a ``.weight`` suffix::

    ...mlp.experts.gate_up_proj   # [E, 2I, H]  gate=half 0, up=half 1
    ...mlp.experts.down_proj      # [E, H, I]

Hyper-connection (``hc_per_branch_norm``) uses Gemma grouped RMSNorm on
``hc_count * hidden`` (10240 = 4×2560): ``y = rms(x) * (1 + gamma)``,
gamma init 0. After norm, ``GatedResidual.mix`` builds gates via linears
**and** does ``gates * hyper_input_normed`` elementwise before the mean.

Implications for preprocess:
- Do **not** fuse ``hc_norm`` (fuse+γ=0 breaks the elementwise mix; mixer
  γ→0 collapsed logits in prior dumps).
- Do **not** QuaRot through the residual: the stream *is* the 4×H HC path
  with unfused diagonal ``(1+γ)``, which does not commute with Hadamard.
  ``build_rotate_pairs`` therefore returns empty maps (整段不转).
- Layer ``hc_norm`` Flex Smooth is off by default in the recipe (A/B);
  MTP ``pre_fc_norm_*`` may still be smoothed. Final mixers stay
  Smooth-excluded (shared ``lm_head``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from preprocess.adapters.base import ModelAdapter


def _tc(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer nested ``text_config`` (VL wrapper) when present."""
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def _layer_prefix(i: int) -> str:
    return f"model.language_model.layers.{i}"


def _is_full_attn_layer(cfg: Dict[str, Any], layer_idx: int) -> bool:
    """``full_attention_interval``: full self_attn when ``(i+1) % interval == 0``."""
    interval = int(cfg.get("full_attention_interval") or 4)
    if interval <= 0:
        return True
    return (layer_idx + 1) % interval == 0


def _attn_inputs(prefix: str, full: bool) -> List[str]:
    if full:
        attn = f"{prefix}.self_attn"
        return [
            f"{attn}.q_proj",
            f"{attn}.k_proj",
            f"{attn}.v_proj",
            f"{attn}.indexer.index_qk_proj",
        ]
    la = f"{prefix}.linear_attn"
    return [
        f"{la}.in_proj_qkv",
        f"{la}.in_proj_a",
        f"{la}.in_proj_b",
        f"{la}.in_proj_z",
    ]


def _mlp_inputs(prefix: str) -> List[str]:
    return [
        f"{prefix}.mlp.gate",
        f"{prefix}.mlp.shared_expert.gate_proj",
        f"{prefix}.mlp.shared_expert.up_proj",
        f"{prefix}.mlp.shared_expert_gate",
        f"{prefix}.mlp.experts.gate_up_proj",
    ]


class Qwen38FlashNextAdapter(ModelAdapter):
    """Adapter for Qwen3.8-Flash-Next (model_type ``qwen4_exp`` / ``qwen4_exp_text``)."""

    name = "qwen3_8_flash_next"

    def prepare_weights(self, weights, ctx) -> None:
        # Checkpoint is already BF16; nothing to unpack.
        return

    def norm_affine(self) -> str:
        return "gemma"

    def build_ln_fuse_map(
        self,
        cfg: Dict[str, Any],
        num_layers: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Only Norm→Linear paths with no elementwise reuse of the normed vector.

        All ``hc_norm`` entries are omitted: mix consumes ``hyper_input_normed``
        elementwise, so fuse+γ=0 is inequivalent (especially final mixers).
        ``pre_fc_norm_hidden`` is also omitted (shared ``fc_hidden``). Head-dim
        ``q_norm`` / ``k_norm`` are not included. SmoothQuant still covers
        layer ``hc_norm`` via ``resolve_norm_linear`` + partners.
        """
        del cfg, num_layers  # unused; map is architecture-fixed
        return {"mtp.pre_fc_norm_embedding": ["mtp.fc_embedding"]}

    def _hc_smooth_consumers(
        self,
        cfg: Dict[str, Any],
        num_layers: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Wide Linear consumers of each ``hc_norm`` for Smooth only (not QuaRot fuse)."""
        cfg = _tc(cfg)
        if num_layers is None:
            num_layers = int(cfg["num_hidden_layers"])
        out: Dict[str, List[str]] = {}

        def add_hc(module_prefix: str, *, combine: bool) -> None:
            consumers = [f"{module_prefix}.input_mix_weight_down"]
            if combine:
                consumers.append(f"{module_prefix}.block_inject_weight")
            out[f"{module_prefix}.hc_norm"] = consumers

        for i in range(num_layers):
            prefix = _layer_prefix(i)
            add_hc(f"{prefix}.attn_hyper_connection", combine=True)
            add_hc(f"{prefix}.mlp_hyper_connection", combine=True)
        add_hc("mtp.layers.0.attn_hyper_connection", combine=True)
        add_hc("mtp.layers.0.mlp_hyper_connection", combine=True)
        # Final mixers intentionally absent: no hidden partners (shared lm_head).
        return out

    def resolve_norm_linear(
        self,
        seed_module: str,
        cfg: Dict[str, Any],
        *,
        num_layers: Optional[int] = None,
    ):
        # Smooth groups differ from QuaRot fuse: hc_norm stays unfused but still
        # absorbs a tied Smooth scale into wide + hidden partners.
        if seed_module == "mtp.pre_fc_norm_hidden":
            # Shared fc_hidden cannot take a per-channel 10240 fold; partners only.
            return seed_module, []
        hc_map = self._hc_smooth_consumers(cfg, num_layers=num_layers)
        if seed_module in hc_map:
            return seed_module, list(hc_map[seed_module])
        for norm, consumers in hc_map.items():
            if seed_module in consumers:
                return norm, list(consumers)
        return super().resolve_norm_linear(seed_module, cfg, num_layers=num_layers)

    def hidden_smooth_partners(self, norm_module: str, cfg: Dict[str, Any]) -> List[str]:
        """Hidden-size linears that read the mix output of this hc_norm."""
        cfg = _tc(cfg)
        attn_suf = ".attn_hyper_connection.hc_norm"
        mlp_suf = ".mlp_hyper_connection.hc_norm"
        if norm_module.endswith(attn_suf):
            prefix = norm_module[: -len(attn_suf)]
            if prefix.startswith("mtp."):
                full = True
            else:
                m = re.search(r"layers\.(\d+)$", prefix)
                if m is None:
                    return []
                full = _is_full_attn_layer(cfg, int(m.group(1)))
            return _attn_inputs(prefix, full)
        if norm_module.endswith(mlp_suf):
            prefix = norm_module[: -len(mlp_suf)]
            return _mlp_inputs(prefix)
        if norm_module == "mtp.pre_fc_norm_hidden":
            return ["mtp.fc_hidden"]
        # Final mixers (no combine) write the vector that lm_head reads.
        # Shared lm_head cannot take a second Smooth scale; recipe also omits them.
        return []

    def build_rotate_pairs(
        self,
        cfg: Dict[str, Any],
        *,
        block_size: int = 32,
        num_layers: Optional[int] = None,
        num_experts: Optional[int] = None,
    ) -> Tuple[List[Tuple[Dict[str, str], Dict[str, str]]], Dict[str, Any]]:
        """Empty rotate maps: residual is HC 4×H with unfused Gemma ``hc_norm``.

        Hadamard does not commute with diagonal ``(1+γ)``. Rotating only
        non-HC linears still leaves the residual path inequivalent, so Qwen
        skips QuaRot rotation entirely (fuse_ln for ``pre_fc_norm_embedding``
        may still run when ``fuse_ln: true``).
        """
        del num_layers, num_experts
        cfg = _tc(cfg)
        hidden = int(cfg["hidden_size"])
        meta: Dict[str, Any] = {
            "rot": {"size": hidden, "mode": "hadamard", "block_size": block_size},
        }
        return [({}, {})], meta

    def resolve_up_down(self, up_module: str) -> Optional[str]:
        """``shared_expert.up_proj→down_proj``; fused ``experts.gate_up_proj→down_proj``."""
        m = re.match(r"(.*\.mlp\.shared_expert)\.up_proj$", up_module)
        if m:
            return f"{m.group(1)}.down_proj"
        m = re.match(r"(.*\.mlp\.experts)\.gate_up_proj$", up_module)
        if m:
            return f"{m.group(1)}.down_proj"
        m = re.match(r"(.*)\.up_proj$", up_module)
        if m:
            return f"{m.group(1)}.down_proj"
        return None
