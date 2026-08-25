"""Model adapter contracts used by the quantization and packing pipelines."""

import re
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM


class ModelAdapter:
    """Small model-facing interface for MoE-Quant.

    The quantization engine owns GPTQ, calibration, distributed communication,
    and tensor serialization.  An adapter owns model structure and checkpoint
    conventions.  Methods intentionally have useful defaults for conventional
    Hugging Face causal language models.
    """

    name = "default"
    _routed_expert_pattern = re.compile(
        r"(?:^|\.)mlp\.experts\.\d+\.(?:down|gate|up)_proj(?:\.|$)"
    )

    def __init__(self, config: Optional[Any] = None):
        self.config = config

    @classmethod
    def matches(cls, config: Any) -> bool:
        """Return whether this adapter owns ``config``."""
        del config
        return False

    def prepare_config(self, config: Any, world_size: int) -> None:
        """Apply model-specific distributed configuration before construction."""
        del config, world_size

    def build_empty_model(
        self,
        config: Any,
        dtype: torch.dtype,
        attn_implementation: Optional[str] = None,
    ):
        """Construct the empty model used by the quantization pipeline."""
        kwargs = {
            "config": config,
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        return AutoModelForCausalLM.from_config(**kwargs).eval()

    def prepare_model(self, model: torch.nn.Module, config: Any, dtype: torch.dtype) -> None:
        """Patch/convert the model structure before weights are loaded."""
        del model, config, dtype

    def get_embedding_module(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.model.embed_tokens

    def embedding_weight_key(self) -> str:
        return "model.embed_tokens.weight"

    def embedding_state_keys(self) -> List[str]:
        return [self.embedding_weight_key(), self.embedding_weight_key() + "_scale_inv"]

    def get_transformer_layers(self, model: torch.nn.Module):
        return model.model.layers

    def get_layer_prefix(self, block_idx: int) -> str:
        return "model.layers.{}.".format(block_idx)

    def get_final_tensor_keys(self) -> List[str]:
        return ["lm_head.weight", "model.norm.weight"]

    def is_routed_expert(self, layer_name: str) -> bool:
        """Identify a routed expert Linear by its module name."""
        return self._routed_expert_pattern.search(layer_name) is not None

    def has_routed_experts(self, names: Iterable[str]) -> bool:
        return any(self.is_routed_expert(name) for name in names)

    def forward_block(
        self,
        block: torch.nn.Module,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder block and normalize its return value."""
        result = block(hidden_states, position_ids=position_ids)
        if isinstance(result, tuple):
            return result[0]
        return result

    def get_quantization_ignore(self, quantize_only_experts: bool):
        """Return ``(rule_name, compressed-tensors ignore list)``."""
        ignored_modules = ["lm_head"]
        if quantize_only_experts:
            ignored_modules += [
                r"re:.*self_attn.*",
                r"re:.*shared_experts.*",
                r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
            ]
            rule_name = "default_experts_only"
        else:
            rule_name = "default"
        return rule_name, ignored_modules

    def count_extra_shards(self, weight_map: Dict[str, str]) -> int:
        del weight_map
        return 0

    def save_extra_weights(
        self,
        weight_dir: str,
        weight_map: Dict[str, str],
        packed_model_path: str,
        next_shard_id: int,
        num_output_shards: int,
        safetensors_index: Dict[str, str],
    ) -> int:
        del weight_dir, weight_map, packed_model_path, num_output_shards, safetensors_index
        return next_shard_id
