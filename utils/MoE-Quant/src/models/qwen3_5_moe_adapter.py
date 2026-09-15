# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/Qwen3.8 MoE model adapter."""

from typing import Any, Dict

from .base import ModelAdapter


class Qwen35MoeAdapter(ModelAdapter):
    """Adapter for the text-only Qwen3.5-MoE structure used by Qwen3.8."""

    name = "qwen3_5_moe"

    @classmethod
    def matches(cls, config: Any) -> bool:
        return getattr(config, "model_type", None) == "qwen3_5_moe_text"

    def prepare_config(self, config: Any, world_size: int) -> None:
        config.ep_size = world_size

    def validate_quantization_args(self, args: Any) -> None:
        if args.bits != 4:
            raise ValueError(
                "Qwen3.5/3.8 MoE currently supports only --bits 4 (W4A16); MTP FP8 handling "
                "is not validated for 8-bit."
            )

    def prepare_model(self, model, config: Any, dtype) -> None:
        # Keep the structural conversion and forward compatibility code in the
        # existing Qwen-specific utility module.
        from . import qwen3_5_moe_utils

        qwen3_5_moe_utils.prepare_qwen3_5_moe_model(model, config, dtype)

    def get_quantization_ignore(self, quantize_only_experts: bool):
        ignored_modules = ["lm_head", r"re:^mtp\..*"]
        if quantize_only_experts:
            ignored_modules += [
                "model.embed_tokens",
                r"re:.*linear_attn\.conv1d$",
                r"re:.*linear_attn\.in_proj_a$",
                r"re:.*linear_attn\.in_proj_b$",
                r"re:.*linear_attn\.in_proj_qkv$",
                r"re:.*linear_attn\.in_proj_z$",
                r"re:.*linear_attn\.out_proj$",
                r"re:.*mlp\.gate$",
                r"re:.*mlp\.shared_expert\.(gate|up|down)_proj$",
                r"re:.*mlp\.shared_expert_gate$",
                r"re:.*self_attn\.(q|k|v|o)_proj$",
            ]
            rule_name = "qwen3_5_moe_experts_only"
        else:
            rule_name = "default"
        return rule_name, ignored_modules

    def count_extra_shards(self, weight_map: Dict[str, str]) -> int:
        from . import qwen3_5_moe_utils

        return qwen3_5_moe_utils.count_mtp_shards(weight_map)

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

        return qwen3_5_moe_utils.save_mtp_weights(
            weight_dir,
            weight_map,
            packed_model_path,
            next_shard_id,
            num_output_shards,
            safetensors_index,
        )
