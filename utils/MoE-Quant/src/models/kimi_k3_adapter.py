"""Kimi-K3 model adapter."""

import os
import re
import shutil
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .base import ModelAdapter


class KimiK3Adapter(ModelAdapter):
    """Adapter for the multimodal Kimi-K3 checkpoint and text backbone."""

    name = "kimi_k3"
    _routed_expert_pattern = re.compile(
        r"(?:^|\.)block_sparse_moe\.experts\.\d+\.w[123](?:\.|$)"
    )
    _extra_prefixes = ("vision_tower.", "mm_projector.")
    _artifact_names = (
        "configuration.json",
        "configuration_kimi_k3.py",
        "encoding_k3.py",
        "generation_config.json",
        "kimi_k3_processor.py",
        "kimi_k3_vision_processing.py",
        "media_utils.py",
        "modeling_kimi_k3.py",
        "modeling_kimi_linear.py",
        "preprocessor_config.json",
        "tiktoken.model",
        "tokenization_kimi.py",
        "tokenizer_config.json",
    )

    @classmethod
    def matches(cls, config: Any) -> bool:
        model_type = str(getattr(config, "model_type", "")).lower()
        architectures = {
            str(value).lower()
            for value in getattr(config, "architectures", []) or []
        }
        return (
            model_type == "kimi_k3"
            or "kimik3forconditionalgeneration" in architectures
        )

    def prepare_config(self, config: Any, world_size: int) -> None:
        text_config = config.text_config
        for current_config in (config, text_config):
            if hasattr(current_config, "quantization_config"):
                delattr(current_config, "quantization_config")
        text_config.ep_size = world_size
        text_config.use_cache = False
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            print("[INFO] Kimi-K3 source routed expert format=MXFP4 (group_size=32)")
            print(
                "[INFO] Kimi-K3 output policy: routed experts=INT4 GPTQ, "
                "shared/attention/vision/projector=bfloat16, calibration=text-only"
            )

    def validate_quantization_args(self, args: Any) -> None:
        if not args.quantize_only_experts:
            raise ValueError(
                "Kimi-K3 currently requires --quantize_only_experts; its SGLang "
                "runtime keeps non-routed modules unquantized."
            )
        if not args.sym:
            raise ValueError("Kimi-K3 currently requires --sym for INT4 GPTQ output.")
        if args.dtype != "bfloat16":
            raise ValueError(
                "Kimi-K3 currently requires --dtype bfloat16 for calibration."
            )
        attn_implementation = getattr(
            args, "attn_implementation", "flash_attention_2"
        )
        if attn_implementation == "flash_attention_2":
            raise ValueError(
                "Kimi-K3 does not support Flash Attention 2 in its remote model code; "
                "use --attn_implementation eager or another backend supported by the model."
            )

    def validate_packing_args(self, args: Any) -> None:
        if not args.quantize_only_experts:
            raise ValueError(
                "Kimi-K3 currently requires an experts-only GPTQ result for packing."
            )
        if args.dtype != "bfloat16":
            raise ValueError("Kimi-K3 currently requires --dtype bfloat16 for packing.")
        if args.activation_bits != 16:
            raise ValueError(
                "Kimi-K3 currently supports only W4A16 packing "
                "(--activation-bits 16)."
            )

    def build_empty_model(
        self,
        config: Any,
        dtype: torch.dtype,
        attn_implementation: Optional[str] = None,
    ):
        from . import kimi_k3_utils

        kimi_k3_utils.patch_kimi_k3_for_ep(config)
        return super().build_empty_model(config, dtype, attn_implementation)

    def build_packing_model(self, config: Any, dtype: torch.dtype):
        from . import kimi_k3_utils

        kimi_k3_utils.set_packing_mode(True)
        try:
            return self.build_empty_model(
                config,
                dtype,
                attn_implementation="eager",
            )
        finally:
            kimi_k3_utils.set_packing_mode(False)

    def get_embedding_module(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.language_model.model.embed_tokens

    def embedding_weight_key(self) -> str:
        return "language_model.model.embed_tokens.weight"

    def get_transformer_layers(self, model: torch.nn.Module):
        return model.language_model.model.layers

    def get_layer_prefix(self, block_idx: int) -> str:
        return f"language_model.model.layers.{block_idx}."

    def get_final_tensor_keys(self) -> List[str]:
        keys = [
            "language_model.lm_head.weight",
            "language_model.model.norm.weight",
        ]
        text_config = getattr(self.config, "text_config", None)
        if getattr(text_config, "attn_res_block_size", None) is not None:
            keys.extend(
                [
                    "language_model.model.output_attn_res_norm.weight",
                    "language_model.model.output_attn_res_proj.weight",
                ]
            )
        return keys

    def checkpoint_keys_for_model_key(
        self,
        model_key: str,
        weight_map: Dict[str, str],
    ) -> List[str]:
        if model_key in weight_map:
            return [model_key]
        if model_key.endswith(".weight"):
            base = model_key.removesuffix(".weight")
            packed_key = f"{base}.weight_packed"
            scale_key = f"{base}.weight_scale"
            if packed_key in weight_map or scale_key in weight_map:
                missing = [
                    key for key in (packed_key, scale_key)
                    if key not in weight_map
                ]
                if missing:
                    raise KeyError(
                        f"Kimi MXFP4 weight {model_key} is missing checkpoint "
                        f"tensor(s): {', '.join(missing)}"
                    )
                return [packed_key, scale_key]
        return [model_key]

    def materialize_state_dict(
        self,
        physical_state_dict: Dict[str, torch.Tensor],
        model_keys: Iterable[str],
        dtype: torch.dtype,
        expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Dict[str, torch.Tensor]:
        from .. import quant_utils

        expected_shapes = expected_shapes or {}
        logical_state_dict = {}
        for model_key in model_keys:
            if model_key in physical_state_dict:
                tensor = physical_state_dict[model_key]
                expected_shape = expected_shapes.get(model_key)
                if model_key.endswith(".A_log"):
                    if expected_shape is None:
                        num_heads = self.config.text_config.linear_attn_config["num_heads"]
                        expected_shape = (num_heads,)
                    if tensor.ndim == 1 and tensor.numel() >= expected_shape[0]:
                        tensor = tensor[: expected_shape[0]].contiguous()
                if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
                    raise ValueError(
                        f"Checkpoint tensor {model_key} has shape {tuple(tensor.shape)}, "
                        f"expected {tuple(expected_shape)}."
                    )
                logical_state_dict[model_key] = tensor
                continue

            if not model_key.endswith(".weight"):
                continue
            base = model_key.removesuffix(".weight")
            packed_key = f"{base}.weight_packed"
            scale_key = f"{base}.weight_scale"
            packed = physical_state_dict.get(packed_key)
            scale = physical_state_dict.get(scale_key)
            if packed is None and scale is None:
                continue
            if packed is None or scale is None:
                raise KeyError(
                    f"Kimi MXFP4 logical weight {model_key} requires both "
                    f"{packed_key} and {scale_key}."
                )
            logical_state_dict[model_key] = quant_utils.dequantize_weight_from_mxfp4(
                packed,
                scale,
                dtype=dtype,
                expected_shape=expected_shapes.get(model_key),
            )
        return logical_state_dict

    def get_tied_gptq_source(self, layer_name: str) -> Optional[str]:
        if not layer_name.endswith(".w3"):
            return None
        return layer_name.removesuffix(".w3") + ".w1"

    def create_block_state(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del position_ids
        return {
            "block_residual": hidden_states.new_zeros(
                hidden_states.shape[0] * hidden_states.shape[1],
                0,
                hidden_states.shape[2],
            )
        }

    def move_block_state(self, block_state: Any, device: Optional[str]) -> Any:
        if block_state is None or device is None:
            return block_state
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in block_state.items()
        }

    def forward_block(
        self,
        block: torch.nn.Module,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        block_state: Any = None,
    ):
        from transformers.masking_utils import create_causal_mask

        cache_position = torch.arange(
            hidden_states.shape[1], device=hidden_states.device
        )
        attention_mask = None
        if not getattr(block, "is_linear_attn", False):
            attention_mask = create_causal_mask(
                config=block.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                past_key_values=None,
                position_ids=position_ids,
            )
        state = block_state or self.create_block_state(hidden_states, position_ids)
        result = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            block_residual=state["block_residual"],
        )
        if isinstance(result, tuple):
            hidden_states, block_residual = result
        else:
            hidden_states = result
            block_residual = state["block_residual"]
        return hidden_states, {"block_residual": block_residual}

    def prepare_model(self, model, config: Any, dtype) -> None:
        from . import kimi_k3_utils

        kimi_k3_utils.prepare_kimi_k3_model(model, config, dtype)

    def get_quantization_ignore(self, quantize_only_experts: bool):
        ignored_modules = [
            r"re:(?:.*\.)?lm_head$",
            r"re:^vision_tower\..*",
            r"re:^mm_projector\..*",
        ]
        if quantize_only_experts:
            ignored_modules.extend(
                [
                    r"re:.*model\.embed_tokens(?:\..*)?$",
                    r"re:.*\.self_attn\..*",
                    r"re:.*\.gate(?:\..*)?$",
                    r"re:.*\.shared_experts\..*",
                    r"re:.*\.routed_expert_(down|up)_proj(?:\..*)?$",
                    r"re:.*\.routed_expert_norm(?:\..*)?$",
                    r"re:.*\.mlp\.(gate|up|down)_proj(?:\..*)?$",
                    r"re:.*\.(self_attention|mlp)_res_(proj|norm)(?:\..*)?$",
                    r"re:.*\.output_attn_res_(proj|norm)(?:\..*)?$",
                ]
            )
            return "kimi_k3_experts_only", ignored_modules
        return "default", ignored_modules

    def set_quantization_config(self, config: Any, quantization_config: Dict[str, Any]) -> None:
        config.quantization_config = quantization_config
        config.text_config.quantization_config = quantization_config

    def logical_block_keys(
        self,
        block: torch.nn.Module,
        block_idx: int,
    ) -> set[str]:
        prefix = self.get_layer_prefix(block_idx)
        keys = {f"{prefix}{key}" for key in block.state_dict()}
        text_config = self.config.text_config
        if (
            block_idx >= text_config.first_k_dense_replace
            and block_idx % getattr(text_config, "moe_layer_freq", 1) == 0
        ):
            expert_prefix = f"{prefix}block_sparse_moe.experts"
            for expert_idx in range(text_config.num_experts):
                for projection in ("w1", "w2", "w3"):
                    keys.add(f"{expert_prefix}.{expert_idx}.{projection}.weight")
        return keys

    def expected_quantized_layer_names(self, model: torch.nn.Module) -> set[str]:
        del model
        text_config = self.config.text_config
        names = set()
        for layer_idx in range(text_config.num_hidden_layers):
            if (
                layer_idx < text_config.first_k_dense_replace
                or layer_idx % getattr(text_config, "moe_layer_freq", 1) != 0
            ):
                continue
            prefix = f"language_model.model.layers.{layer_idx}.block_sparse_moe.experts"
            for expert_idx in range(text_config.num_experts):
                for projection in ("w1", "w2", "w3"):
                    names.add(f"{prefix}.{expert_idx}.{projection}")
        return names

    def count_extra_shards(self, weight_map: Dict[str, str]) -> int:
        return len(
            {
                filename
                for key, filename in weight_map.items()
                if key.startswith(self._extra_prefixes)
            }
        )

    def save_extra_weights(
        self,
        weight_dir: str,
        weight_map: Dict[str, str],
        packed_model_path: str,
        next_shard_id: int,
        num_output_shards: int,
        safetensors_index: Dict[str, str],
    ) -> int:
        extra_files = sorted(
            {
                filename
                for key, filename in weight_map.items()
                if key.startswith(self._extra_prefixes)
            }
        )
        for filename in extra_files:
            tensors = {}
            with safe_open(
                os.path.join(weight_dir, filename), framework="pt", device="cpu"
            ) as handle:
                for key in handle.keys():
                    if key.startswith(self._extra_prefixes):
                        tensor = handle.get_tensor(key)
                        if tensor.is_floating_point():
                            tensor = tensor.to(torch.bfloat16)
                        tensors[key] = tensor
            if not tensors:
                continue
            shard_name = (
                f"model-{next_shard_id:05}-of-{num_output_shards:05}.safetensors"
            )
            save_file(tensors, os.path.join(packed_model_path, shard_name))
            for key in tensors:
                safetensors_index[key] = shard_name
            next_shard_id += 1
        return next_shard_id

    def copy_artifacts(self, source_dir: str, output_dir: str) -> None:
        for name in self._artifact_names:
            source_path = os.path.join(source_dir, name)
            if os.path.isfile(source_path):
                shutil.copy(source_path, output_dir)
