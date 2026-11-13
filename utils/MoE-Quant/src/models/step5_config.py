"""Transformers configuration registration for the Step-5 checkpoint."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class Step5Config(PretrainedConfig):
    model_type = "step5"

    def __init__(self, **kwargs: Any) -> None:
        # Keep the original JSON fields available to the calibration model.
        # Explicit defaults are intentionally small; a real checkpoint always
        # supplies these values in text_config.
        self.hidden_size = kwargs.pop("hidden_size", 4096)
        self.intermediate_size = kwargs.pop("intermediate_size", 13824)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 92)
        self.vocab_size = kwargs.pop("vocab_size", 128896)
        self.rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-5)
        self.fp32_residual_connection = kwargs.pop("fp32_residual_connection", True)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 64)
        self.num_attention_groups = kwargs.pop("num_attention_groups", 4)
        self.head_dim = kwargs.pop("head_dim", 192)
        self.rope_theta = kwargs.pop("rope_theta", 10000.0)
        self.rope_scaling = kwargs.pop("rope_scaling", None)
        layer_types = kwargs.pop("layer_types", None)
        self.layer_types_with_mtp = (
            list(layer_types[: self.num_hidden_layers + kwargs.get("num_nextn_predict_layers", 0)])
            if layer_types
            else None
        )
        self.layer_types = (
            self.layer_types_with_mtp[: self.num_hidden_layers]
            if self.layer_types_with_mtp
            else None
        )
        self.partial_rotary_factors = kwargs.pop("partial_rotary_factors", None)
        self.sliding_window = kwargs.pop("sliding_window", None)
        self.attention_other_setting = kwargs.pop("attention_other_setting", None)
        self.use_head_wise_attn_gate = kwargs.pop("use_head_wise_attn_gate", False)
        self.moe_layer_list = list(kwargs.pop("moe_layer_list", []))
        self.moe_num_experts = kwargs.pop("moe_num_experts", 0)
        self.moe_intermediate_size = kwargs.pop("moe_intermediate_size", 0)
        self.share_expert_dim = kwargs.pop("share_expert_dim", self.moe_intermediate_size)
        self.moe_top_k = kwargs.pop("moe_top_k", kwargs.pop("num_experts_per_tok", 1))
        self.num_experts_per_tok = self.moe_top_k
        self.use_moe_router_bias = kwargs.pop("use_moe_router_bias", True)
        self.need_fp32_gate = kwargs.pop("need_fp32_gate", True)
        self.moe_router_scaling_factor = kwargs.pop("moe_router_scaling_factor", 1.0)
        self.norm_expert_weight = kwargs.pop("norm_expert_weight", True)
        self.swiglu_limits = kwargs.pop("swiglu_limits", None)
        self.swiglu_limits_shared = kwargs.pop("swiglu_limits_shared", None)
        self.num_nextn_predict_layers = kwargs.pop("num_nextn_predict_layers", 0)
        self.max_seq_len = kwargs.pop("max_seq_len", kwargs.get("max_position_embeddings", 4096))
        self.max_position_embeddings = kwargs.pop(
            "max_position_embeddings", self.max_seq_len
        )
        self.att_impl_type = kwargs.pop("att_impl_type", "GQA")
        self.use_rope_layers = kwargs.pop("use_rope_layers", None)
        self.yarn_only_types = kwargs.pop("yarn_only_types", None)
        self.sparse_config = kwargs.pop("sparse_config", None)
        self.moe_router_activation = kwargs.pop("moe_router_activation", "sigmoid")
        self.zero_centered = kwargs.pop("zero_centered", True)
        super().__init__(**kwargs)


class Step5VLConfig(PretrainedConfig):
    model_type = "step5_vl"
    architectures = ["Step5ForConditionalGeneration"]
    sub_configs = {"text_config": Step5Config}

    def __init__(
        self,
        text_config: dict[str, Any] | Step5Config | None = None,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if text_config is None:
            text_config = Step5Config()
        elif isinstance(text_config, dict):
            text_config = Step5Config(**text_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.hidden_size = text_config.hidden_size
        if kwargs.get("architectures") is None:
            kwargs["architectures"] = list(self.architectures)
        super().__init__(**kwargs)


def register_step5_configs() -> None:
    for model_type, config_class in (
        ("step5", Step5Config),
        ("step5_vl", Step5VLConfig),
    ):
        try:
            AutoConfig.register(model_type, config_class)
        except ValueError:
            # Importing this module more than once is harmless.
            pass


register_step5_configs()

