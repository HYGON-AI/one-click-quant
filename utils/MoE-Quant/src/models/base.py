# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Model adapter contracts used by the quantization and packing pipelines."""

import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

    def checkpoint_keys_for_model_key(
        self,
        model_key: str,
        weight_map: Dict[str, str],
    ) -> List[str]:
        """Map one logical model state key to physical checkpoint tensor keys."""
        del weight_map
        return [model_key]

    def materialize_state_dict(
        self,
        physical_state_dict: Dict[str, torch.Tensor],
        model_keys: Iterable[str],
        dtype: torch.dtype,
        expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build a logical model state dict from checkpoint tensors."""
        from .. import quant_utils

        logical_state_dict = dict(physical_state_dict)
        if not quant_utils.can_dequantize_from_fp8(logical_state_dict):
            raise RuntimeError("An FP8 weight is missing its matching *_scale_inv tensor.")
        quant_utils.dequantize_state_dict(logical_state_dict, dtype)
        return {
            key: logical_state_dict[key]
            for key in model_keys
            if key in logical_state_dict
        }

    def get_block_index_from_layer_name(self, layer_name: str) -> int:
        """Extract a transformer block index from a fully-qualified layer name."""
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        if match is None:
            raise ValueError(f"Cannot determine transformer block index from {layer_name!r}.")
        return int(match.group(1))

    def get_tied_gptq_source(self, layer_name: str) -> Optional[str]:
        """Return the previously-created layer whose Hessian can be reused."""
        if not layer_name.endswith("up_proj"):
            return None
        parent_name, _ = layer_name.rsplit(".", 1)
        return f"{parent_name}.gate_proj"

    def validate_quantization_args(self, args: Any) -> None:
        """Validate model-specific calibration and GPTQ options."""
        del args

    def validate_packing_args(self, args: Any) -> None:
        """Validate model-specific packed-checkpoint options."""
        del args

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

    def build_packing_model(self, config: Any, dtype: torch.dtype):
        """Construct the structural model used while packing output shards."""
        return self.build_empty_model(config, dtype)

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

    def logical_block_keys(
        self,
        block: torch.nn.Module,
        block_idx: int,
    ) -> set[str]:
        prefix = self.get_layer_prefix(block_idx)
        return {f"{prefix}{key}" for key in block.state_dict()}

    def get_final_tensor_keys(self) -> List[str]:
        return ["lm_head.weight", "model.norm.weight"]

    def is_routed_expert(self, layer_name: str) -> bool:
        """Identify a routed expert Linear by its module name."""
        return self._routed_expert_pattern.search(layer_name) is not None

    def has_routed_experts(self, names: Iterable[str]) -> bool:
        return any(self.is_routed_expert(name) for name in names)

    def create_block_state(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Any:
        """Create per-calibration-sequence state carried across decoder blocks."""
        del hidden_states, position_ids
        return None

    def move_block_state(self, block_state: Any, device: Optional[str]) -> Any:
        """Move adapter-owned per-sequence state alongside hidden activations."""
        del device
        return block_state

    def forward_block(
        self,
        block: torch.nn.Module,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        block_state: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Run one decoder block and return hidden states plus carried state."""
        result = block(hidden_states, position_ids=position_ids)
        if isinstance(result, tuple):
            result = result[0]
        return result, block_state

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

    def set_quantization_config(self, config: Any, quantization_config: Dict[str, Any]) -> None:
        """Install output quantization metadata on the model config."""
        config.quantization_config = quantization_config

    def expected_quantized_layer_names(self, model: torch.nn.Module) -> Optional[set[str]]:
        """Return required GPTQ layer names when an adapter requires completeness."""
        del model
        return None

    def count_extra_shards(self, weight_map: Dict[str, str]) -> int:
        del weight_map
        return 0

    def copy_artifacts(self, source_dir: str, output_dir: str) -> None:
        """Copy model remote-code files needed to load the packed checkpoint."""
        modeling_files = sorted(
            name for name in os.listdir(source_dir)
            if name.startswith("modeling_") and name.endswith(".py")
        )
        if modeling_files:
            shutil.copy(os.path.join(source_dir, modeling_files[0]), output_dir)

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
