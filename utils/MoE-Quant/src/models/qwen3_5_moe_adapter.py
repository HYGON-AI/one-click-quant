# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/Qwen3.8 MoE model adapter."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import re

import torch

from .base import ModelAdapter


class Qwen35MoeAdapter(ModelAdapter):
    """Adapter for Qwen3.5/Qwen3.8 MoE checkpoints.

    Handles the text-only layout (``model.layers.*``) and the multimodal wrapper
    one level deeper (``model.language_model.layers.*`` + ``model.visual.*``).
    Only the text backbone is quantized; vision tower and MTP head are copied
    verbatim by :meth:`save_extra_weights`.
    """

    name = "qwen3_5_moe"

    # Checkpoint-space (not in-memory) key prefixes:
    _TEXT_ONLY_PREFIX = "model."
    _MULTIMODAL_PREFIX = "model.language_model."

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        # hidden_size / num_experts / rope_parameters live only on text_config.
        text_config = getattr(config, "text_config", None)
        self.is_multimodal = text_config is not None
        self.text_config = text_config if text_config is not None else config

    @classmethod
    def matches(cls, config: Any) -> bool:
        model_type = getattr(config, "model_type", None)
        if model_type == "qwen3_5_moe_text":
            return True
        text_config = getattr(config, "text_config", None)
        return (
            model_type == "qwen3_5_moe"
            and getattr(text_config, "model_type", None) == "qwen3_5_moe_text"
        )

    # ---------- checkpoint naming ----------

    @property
    def _language_prefix(self) -> str:
        return self._MULTIMODAL_PREFIX if self.is_multimodal else self._TEXT_ONLY_PREFIX

    def get_layer_prefix(self, block_idx: int) -> str:
        return f"{self._language_prefix}layers.{block_idx}."

    def get_module_prefix(self, block_idx: int) -> str:
        """Always ``model.layers.N.``: ``build_empty_model`` instantiates the
        text-only ``Qwen3_5MoeForCausalLM`` regardless of the checkpoint layout.
        """
        return f"model.layers.{block_idx}."

    def embedding_weight_key(self) -> str:
        return f"{self._language_prefix}embed_tokens.weight"

    def get_final_tensor_keys(self) -> List[str]:
        return ["lm_head.weight", f"{self._language_prefix}norm.weight"]

    # ---------- fused (packed) routed-expert checkpoints ----------

    # transformers >= 5.x fuses experts into two 3D tensors
    # (`mlp.experts.gate_up_proj` / `down_proj`) while the unpacked MoE block
    # models one nn.Linear per expert; bridge both representations.
    _PACKED_EXPERT_RE = re.compile(
        r"^(?P<head>.+\.mlp\.experts)\.(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
    )

    def _packed_expert_key(self, model_key: str) -> Optional[str]:
        """Map a per-expert model key onto its fused checkpoint tensor."""
        match = self._PACKED_EXPERT_RE.match(model_key)
        if match is None:
            return None
        fused = (
            "gate_up_proj"
            if match.group("proj") in ("gate_proj", "up_proj")
            else "down_proj"
        )
        return f"{match.group('head')}.{fused}"

    def checkpoint_keys_for_model_key(
        self,
        model_key: str,
        weight_map: Dict[str, str],
    ) -> List[str]:
        """Per-expert keys resolve to themselves; fused ones to the packed tensor."""
        if model_key in weight_map:
            return [model_key]
        packed_key = self._packed_expert_key(model_key)
        if packed_key is not None and packed_key in weight_map:
            return [packed_key]
        return [model_key]

    def _unpack_expert_tensor(
        self,
        model_key: str,
        logical: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Slice one projection out of a fused expert tensor, or return None."""
        match = self._PACKED_EXPERT_RE.match(model_key)
        if match is None:
            return None
        packed_key = self._packed_expert_key(model_key)
        packed = logical.get(packed_key)
        if packed is None:
            return None
        if packed.ndim != 3:
            raise ValueError(
                f"Packed expert tensor {packed_key} must be rank-3, got shape "
                f"{tuple(packed.shape)}."
            )

        expert_idx = int(match.group("idx"))
        projection = match.group("proj")
        if expert_idx >= packed.shape[0]:
            raise ValueError(
                f"{model_key} references expert {expert_idx} but {packed_key} only "
                f"holds {packed.shape[0]} experts."
            )

        if projection == "down_proj":
            return packed[expert_idx].contiguous()

        # `gate, up = linear(x, gate_up_proj[e]).chunk(2, -1)`: rows are [gate | up].
        half = packed.shape[1] // 2
        rows = slice(None, half) if projection == "gate_proj" else slice(half, None)
        return packed[expert_idx, rows].contiguous()

    def materialize_state_dict(
        self,
        physical_state_dict: Dict[str, torch.Tensor],
        model_keys: Iterable[str],
        dtype: torch.dtype,
        expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Dict[str, torch.Tensor]:
        from .. import quant_utils

        logical = dict(physical_state_dict)
        # The shared FP8 dequantizer assumes 2D, so reject rank-3 fused experts.
        for key in logical:
            if key.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
                if logical[key].dtype in quant_utils.FP8_DTYPES:
                    raise NotImplementedError(
                        f"FP8 fused routed experts ({key}) are not supported yet: the "
                        "shared FP8 dequantizer assumes a 2D weight. Dequantize the "
                        "source checkpoint to BF16 first."
                    )
        if not quant_utils.can_dequantize_from_fp8(logical):
            raise RuntimeError(
                "A Qwen3.5 weight is stored in FP8 without its matching *_scale_inv "
                "tensor."
            )
        quant_utils.dequantize_state_dict(logical, dtype)

        result: Dict[str, torch.Tensor] = {}
        for key in model_keys:
            tensor = logical.get(key)
            if tensor is None:
                tensor = self._unpack_expert_tensor(key, logical)
            if tensor is None:
                continue
            expected = (expected_shapes or {}).get(key)
            if expected is not None and tuple(tensor.shape) != tuple(expected):
                raise ValueError(
                    f"Qwen3.5 tensor {key} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(expected)}."
                )
            result[key] = tensor
        return result

    # ---------- structural configuration ----------

    def prepare_config(self, config: Any, world_size: int) -> None:
        # The unpacked MoE block reads `ep_size` off the text config object.
        self.text_config.ep_size = world_size

    def validate_quantization_args(self, args: Any) -> None:
        if args.bits != 4:
            raise ValueError(
                "Qwen3.5/3.8 MoE currently supports only --bits 4 (W4A16); MTP FP8 handling "
                "is not validated for 8-bit."
            )

    def build_empty_model(
        self,
        config: Any,
        dtype: torch.dtype,
        attn_implementation: Optional[str] = None,
    ):
        # Text-only build; the vision tower is copied from the checkpoint later.
        return super().build_empty_model(self.text_config, dtype, attn_implementation)

    def prepare_model(self, model, config: Any, dtype) -> None:
        # Structural conversion lives in the Qwen-specific utility module.
        from . import qwen3_5_moe_utils

        qwen3_5_moe_utils.prepare_qwen3_5_moe_model(model, self.text_config, dtype)

    def default_ignore_rules(self) -> List[str]:
        """Structural rules for Qwen3.5-MoE: weights that never enter the quantized tree."""
        prefix = re.escape(self._language_prefix)
        return [
            "lm_head",
            r"re:^mtp\..*",
            r"re:.*visual.*",
            f"re:{prefix}embed_tokens.*",
            r"re:.*(?:input|post_attention)_layernorm.*",
            r"re:.*\.norm\.weight$",
            r"re:.*self_attn\.(?:k_norm|q_norm).*",
            r"re:.*linear_attn\.conv1d$",
            r"re:.*linear_attn\.A_log$",
            r"re:.*linear_attn\.dt_bias$",
            # SGLang builds this (1, hidden) sigmoid gate with quant_config=None,
            # so quantizing it leaves the model with an unloadable weight.
            r"re:.*mlp\.shared_expert_gate$",
        ]

    def legacy_ignore_rules(self) -> List[str]:
        """Historical ``--quantize_only_experts`` scope: experts only."""
        return [
            r"re:.*self_attn\.(q|k|v|o)_proj$",
            r"re:.*linear_attn\.in_proj_qkv$",
            r"re:.*linear_attn\.in_proj_z$",
            r"re:.*linear_attn\.in_proj_a$",
            r"re:.*linear_attn\.in_proj_b$",
            r"re:.*linear_attn\.out_proj$",
            r"re:.*mlp\.gate$",
            r"re:.*mlp\.shared_expert\.(gate|up|down)_proj$",
            r"re:.*mlp\.shared_expert_gate$",
        ]

    def count_extra_shards(self, weight_map: Dict[str, str]) -> int:
        from . import qwen3_5_moe_utils

        num_shards = qwen3_5_moe_utils.count_mtp_shards(weight_map)
        if self.is_multimodal:
            num_shards += qwen3_5_moe_utils.count_visual_shards(weight_map)
        return num_shards

    def save_extra_weights(
        self,
        weight_dir: str,
        weight_map: Dict[str, str],
        packed_model_path: str,
        next_shard_id: int,
        num_output_shards: int,
        safetensors_index: Dict[str, str],
    ) -> int:
        from . import qwen3_5_moe_utils

        next_shard_id = qwen3_5_moe_utils.save_mtp_weights(
            weight_dir,
            weight_map,
            packed_model_path,
            next_shard_id,
            num_output_shards,
            safetensors_index,
        )
        if self.is_multimodal:
            # The vision tower sits outside every block; carry it over explicitly.
            next_shard_id = qwen3_5_moe_utils.save_visual_weights(
                weight_dir,
                weight_map,
                packed_model_path,
                next_shard_id,
                num_output_shards,
                safetensors_index,
            )
        return next_shard_id

