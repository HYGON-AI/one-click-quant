# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (glm5_next) model adapter for MoE-Quant."""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .base import ModelAdapter


class Glm5NextAdapter(ModelAdapter):
    """Adapter for GLM-5.3-Flash (``model_type="glm5_next"``).

    Scope: routed experts only, symmetric INT4 or INT8 GPTQ; channel-wise or a
    group_size that divides both routed-expert input dims (and is not 32, since
    vLLM's compressed_tensors_moe_wna16 branches to MXFP4 at ``group_size==32``).

    Non-expert paths kept in BF16 / their source dtypes:
      - MLA projections (``self_attn.{q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj}``)
      - DSA indexer (``self_attn.indexer.*``)
      - KDA linear-attention (``self_attn.{q,k,v,b,g_a,g_b,forget_gate.f_a,forget_gate.f_b,o}_proj``,
        ``self_attn.conv1d``, ``self_attn.forget_gate.{A_log,dt_bias}``)
      - Hyper-Connection parameters (``attn_hc.*`` / ``ffn_hc.*``)
      - Router (``mlp.gate.*``) and shared experts (``mlp.shared_experts.*``)
      - Dense MLP layers (indices where ``mlp_layer_types[i] != "sparse"``)
      - Norms, embedding, lm_head, vision tower, MTP layer 45.
    """

    name = "glm5_next"

    _routed_expert_pattern = re.compile(
        r"(?:^|\.)mlp\.experts\.\d+\.(?:down|gate|up)_proj(?:\.|$)"
    )

    # GLM-5.3-Flash checkpoints on disk use flat pre-nesting names
    # (e.g. ``hc_attn_fn`` / ``self_attn.A_log`` / three per-QKV ``*_conv1d``),
    # while the Transformers modeling code exposes nested modules
    # (``attn_hc.fn`` / ``self_attn.forget_gate.A_log`` / a single fused
    # ``self_attn.conv1d``). The upstream `conversion_mapping.py` `glm5_next`
    # entry translates on load; MoE-Quant needs the same translation so it can
    # find each modeling-side tensor in the safetensors index AND emit
    # checkpoint-side names on the packed output (vLLM's Glm5NextForConditional-
    # Generation expects the flat names since its modules bind them directly).
    _HC_MODEL_TO_CHECKPOINT: Tuple[Tuple[str, str], ...] = (
        (".attn_hc.fn", ".hc_attn_fn"),
        (".attn_hc.base", ".hc_attn_base"),
        (".attn_hc.scale", ".hc_attn_scale"),
        (".ffn_hc.fn", ".hc_ffn_fn"),
        (".ffn_hc.base", ".hc_ffn_base"),
        (".ffn_hc.scale", ".hc_ffn_scale"),
    )
    _FORGET_GATE_MODEL_INFIX = ".self_attn.forget_gate."
    _FORGET_GATE_CHECKPOINT_INFIX = ".self_attn."
    _FUSED_CONV1D_MODEL_SUFFIX = ".self_attn.conv1d.weight"
    _FUSED_CONV1D_MODEL_SUFFIX_TRIM = ".conv1d.weight"
    _CONV1D_PART_NAMES: Tuple[str, ...] = ("q", "k", "v")

    _artifact_names = (
        "chat_template.jinja",
        "configuration.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "video_preprocessor_config.json",
    )

    # ------------------------------------------------------------------ #
    # Adapter selection                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def matches(cls, config: Any) -> bool:
        model_type = str(getattr(config, "model_type", "")).lower()
        architectures = {
            str(value).lower()
            for value in getattr(config, "architectures", []) or []
        }
        return (
            model_type == "glm5_next"
            or "glm5nextforconditionalgeneration" in architectures
            or "glm5nextforcausallm" in architectures
        )

    # ------------------------------------------------------------------ #
    # Configuration + argument validation                                #
    # ------------------------------------------------------------------ #

    def _text_config(self) -> Any:
        text_config = getattr(self.config, "text_config", None)
        if text_config is None:
            raise AttributeError(
                "GLM-5.3-Flash config is missing text_config; expected a Glm5NextConfig."
            )
        return text_config

    def prepare_config(self, config: Any, world_size: int) -> None:
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            raise AttributeError(
                "GLM-5.3-Flash config is missing text_config; expected a Glm5NextConfig."
            )
        num_experts = int(getattr(text_config, "n_routed_experts", 0))
        if num_experts <= 0:
            raise ValueError(
                "GLM-5.3-Flash text_config must set n_routed_experts to a positive integer."
            )
        if num_experts % world_size != 0:
            raise ValueError(
                f"GLM-5.3-Flash n_routed_experts ({num_experts}) must be divisible by the "
                f"world size ({world_size})."
            )
        for target in (config, text_config):
            if hasattr(target, "quantization_config"):
                delattr(target, "quantization_config")
        text_config.ep_size = world_size
        text_config.use_cache = False
        config.use_cache = False

    def _validate_group_size(self, args: Any) -> None:
        if args.group_size is None:
            return
        if args.group_size <= 0:
            raise ValueError(
                f"GLM-5.3-Flash group_size must be positive, got {args.group_size}."
            )
        # vLLM's CompressedTensorsWNA16MoEMethod treats group_size==32 as MXFP4
        # (compressed_tensors_moe_wna16.py). Guard here so we don't emit a
        # checkpoint that the runtime silently reinterprets.
        if args.group_size == 32:
            raise ValueError(
                "GLM-5.3-Flash cannot use --group_size 32: vLLM's WNA16 MoE method "
                "reinterprets group_size==32 as an MXFP4 layout."
            )
        text_config = self._text_config()
        hidden = int(getattr(text_config, "hidden_size", 0))
        moe_intermediate = int(getattr(text_config, "moe_intermediate_size", 0))
        for dim_name, dim in (
            ("text_config.hidden_size", hidden),
            ("text_config.moe_intermediate_size", moe_intermediate),
        ):
            if dim <= 0 or dim % args.group_size != 0:
                raise ValueError(
                    f"GLM-5.3-Flash group_size ({args.group_size}) must divide "
                    f"{dim_name} ({dim})."
                )

    def validate_quantization_args(self, args: Any) -> None:
        if args.bits not in (4, 8):
            raise ValueError(
                "GLM-5.3-Flash currently supports --bits 4 (W4A16) or --bits 8 (W8A16)."
            )
        if not args.quantize_only_experts:
            raise ValueError(
                "GLM-5.3-Flash currently requires --quantize_only_experts; MLA, KDA, "
                "indexer, hyper-connection, router, and shared experts stay in BF16."
            )
        if not args.sym:
            raise ValueError(
                "GLM-5.3-Flash currently requires --sym (symmetric INT4/INT8 GPTQ)."
            )
        if args.dtype != "bfloat16":
            raise ValueError("GLM-5.3-Flash currently requires --dtype bfloat16.")
        attn_implementation = getattr(args, "attn_implementation", "flash_attention_2")
        if attn_implementation == "flash_attention_2":
            raise ValueError(
                "GLM-5.3-Flash disables Flash Attention 2 (Glm5NextPreTrainedModel."
                "_supports_flash_attn = False); use --attn_implementation sdpa or eager."
            )
        if args.tie_gptq_handles:
            raise ValueError(
                "--tie_gptq_handles is disabled for GLM-5.3-Flash; per-expert gate/up "
                "tying has not been validated for this recipe."
            )
        self._validate_group_size(args)

    def validate_packing_args(self, args: Any) -> None:
        if not args.quantize_only_experts:
            raise ValueError(
                "GLM-5.3-Flash currently requires an experts-only GPTQ result for packing."
            )
        if args.dtype != "bfloat16":
            raise ValueError("GLM-5.3-Flash currently requires --dtype bfloat16 for packing.")
        if args.activation_bits not in (16, 8):
            raise ValueError(
                "GLM-5.3-Flash currently supports --activation-bits 16 (W*A16) or "
                "8 (W8A8Int8 only)."
            )
        if args.activation_bits == 8 and args.bits != 8:
            raise ValueError(
                "GLM-5.3-Flash W8A8 packing requires --bits 8 GPTQ weights; W4A8 is not "
                "validated for this adapter."
            )
        self._validate_group_size(args)

    # ------------------------------------------------------------------ #
    # Model construction                                                 #
    # ------------------------------------------------------------------ #

    def _from_config(self, config: Any, dtype: torch.dtype, attn_implementation: Optional[str]):
        """Instantiate Glm5NextForConditionalGeneration under init_empty_weights."""
        try:
            from transformers import AutoModelForImageTextToText

            builder = AutoModelForImageTextToText.from_config
        except ImportError:
            from transformers.models.glm5_next.modeling_glm5_next import (
                Glm5NextForConditionalGeneration,
            )

            builder = Glm5NextForConditionalGeneration.from_config

        kwargs: Dict[str, Any] = {
            "config": config,
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        return builder(**kwargs).eval()

    def build_empty_model(
        self,
        config: Any,
        dtype: torch.dtype,
        attn_implementation: Optional[str] = None,
    ):
        return self._from_config(config, dtype, attn_implementation)

    def build_packing_model(self, config: Any, dtype: torch.dtype):
        # Packing runs on rank 0 only and does not need a valid attention kernel.
        return self._from_config(config, dtype, attn_implementation="eager")

    def prepare_model(self, model, config: Any, dtype: torch.dtype) -> None:
        from . import glm5_next_utils

        glm5_next_utils.prepare_glm5_next_model(model, config, dtype)

    # ------------------------------------------------------------------ #
    # Structural key overrides (multimodal → language_model prefix)      #
    # ------------------------------------------------------------------ #

    def get_embedding_module(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.model.language_model.embed_tokens

    def embedding_weight_key(self) -> str:
        return "model.language_model.embed_tokens.weight"

    def get_transformer_layers(self, model: torch.nn.Module):
        return model.model.language_model.layers

    def get_layer_prefix(self, block_idx: int) -> str:
        return f"model.language_model.layers.{block_idx}."

    def get_final_tensor_keys(self) -> List[str]:
        # Norm sits on Glm5NextTextModel; lm_head is bolted directly on the top
        # Glm5NextForConditionalGeneration. Both live in the main state dict.
        return ["lm_head.weight", "model.language_model.norm.weight"]

    # ------------------------------------------------------------------ #
    # Checkpoint tensor materialization                                  #
    # ------------------------------------------------------------------ #

    def _to_physical_keys(self, model_key: str) -> List[str]:
        """Modeling-side state-dict name → physical checkpoint tensor names.

        Any key already in checkpoint-side form is returned as-is (identity),
        which is what the packing pipeline relies on when it feeds this
        adapter its own ``logical_block_keys`` output back through the loader.
        """
        for suffix, physical_suffix in self._HC_MODEL_TO_CHECKPOINT:
            if model_key.endswith(suffix):
                return [model_key[: -len(suffix)] + physical_suffix]
        if self._FORGET_GATE_MODEL_INFIX in model_key:
            return [
                model_key.replace(
                    self._FORGET_GATE_MODEL_INFIX,
                    self._FORGET_GATE_CHECKPOINT_INFIX,
                )
            ]
        if model_key.endswith(self._FUSED_CONV1D_MODEL_SUFFIX):
            trimmed = model_key[: -len(self._FUSED_CONV1D_MODEL_SUFFIX_TRIM)]
            return [
                f"{trimmed}.{part}_conv1d.weight"
                for part in self._CONV1D_PART_NAMES
            ]
        return [model_key]

    def checkpoint_keys_for_model_key(
        self,
        model_key: str,
        weight_map: Dict[str, str],
    ) -> List[str]:
        del weight_map
        return self._to_physical_keys(model_key)

    def materialize_state_dict(
        self,
        physical_state_dict: Dict[str, torch.Tensor],
        model_keys: Iterable[str],
        dtype: torch.dtype,
        expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Dict[str, torch.Tensor]:
        from .. import quant_utils

        logical = dict(physical_state_dict)
        if not quant_utils.can_dequantize_from_fp8(logical):
            raise RuntimeError(
                "GLM-5.3-Flash FP8 weight is missing its matching *_scale_inv tensor."
            )
        quant_utils.dequantize_state_dict(logical, dtype)

        expected_shapes = expected_shapes or {}
        result: Dict[str, torch.Tensor] = {}
        for model_key in set(model_keys):
            tensor: Optional[torch.Tensor]
            if model_key.endswith(self._FUSED_CONV1D_MODEL_SUFFIX):
                trimmed = model_key[: -len(self._FUSED_CONV1D_MODEL_SUFFIX_TRIM)]
                parts: List[torch.Tensor] = []
                for part in self._CONV1D_PART_NAMES:
                    part_key = f"{trimmed}.{part}_conv1d.weight"
                    part_tensor = logical.get(part_key)
                    if part_tensor is None:
                        parts = []
                        break
                    parts.append(part_tensor)
                if not parts:
                    continue
                tensor = torch.cat(parts, dim=0)
            else:
                physical_key: Optional[str] = None
                for suffix, physical_suffix in self._HC_MODEL_TO_CHECKPOINT:
                    if model_key.endswith(suffix):
                        physical_key = (
                            model_key[: -len(suffix)] + physical_suffix
                        )
                        break
                if physical_key is None and self._FORGET_GATE_MODEL_INFIX in model_key:
                    physical_key = model_key.replace(
                        self._FORGET_GATE_MODEL_INFIX,
                        self._FORGET_GATE_CHECKPOINT_INFIX,
                    )
                if physical_key is None:
                    physical_key = model_key
                tensor = logical.get(physical_key)
                if tensor is None:
                    continue

            expected_shape = expected_shapes.get(model_key)
            if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
                raise ValueError(
                    f"GLM-5.3-Flash tensor {model_key} has shape "
                    f"{tuple(tensor.shape)}, expected {tuple(expected_shape)}."
                )
            result[model_key] = tensor
        return result

    # ------------------------------------------------------------------ #
    # Per-block calibration state                                        #
    # ------------------------------------------------------------------ #

    def create_block_state(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        del hidden_states, position_ids
        return {"prev_topk_indices": None}

    def move_block_state(self, block_state: Any, device: Optional[str]) -> Any:
        if block_state is None or device is None:
            return block_state
        moved = dict(block_state)
        topk = moved.get("prev_topk_indices")
        if isinstance(topk, torch.Tensor):
            moved["prev_topk_indices"] = topk.to(device)
        return moved

    def forward_block(
        self,
        block: torch.nn.Module,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        block_state: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        """One decoder layer forward.

        Glm5NextTextDecoderLayer expects hidden_states of shape
        ``[B, S, hc_mult, D]`` (hyper-connection multi-stream). The
        MoE-Quant embedding pass produces ``[B, S, D]``, so the first block sees
        3-D input and we expand to 4-D; subsequent blocks receive the layer's
        own 4-D output verbatim, preserving hc_mult state across depth.
        """
        text_config = self._text_config()
        hc_mult = int(getattr(text_config, "hc_mult", 4))
        if hidden_states.ndim == 3:
            hidden_stream = (
                hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()
            )
        elif hidden_states.ndim == 4:
            hidden_stream = hidden_states
        else:
            raise RuntimeError(
                f"Unexpected hidden_states ndim={hidden_states.ndim}; "
                "expected 3 or 4 for GLM-5.3-Flash."
            )

        # Glm5NextTextModel builds a bool attention mask (see modeling_glm5_next
        # `create_recurrent_attention_mask` path). Without padding in calibration
        # a simple all-ones bool mask matches the shape both KDA and DSA expect.
        attention_mask = torch.ones(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=torch.bool,
            device=hidden_states.device,
        )
        state = block_state or self.create_block_state(hidden_states, position_ids)
        result = block(
            hidden_stream,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=None,
            prev_topk_indices=state.get("prev_topk_indices"),
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "Glm5NextTextDecoderLayer must return (hidden_states, topk_indices); "
                f"got {type(result).__name__}."
            )
        next_hidden, topk_indices = result
        return next_hidden, {"prev_topk_indices": topk_indices}

    # ------------------------------------------------------------------ #
    # Quantization scope + coverage                                      #
    # ------------------------------------------------------------------ #

    def get_quantization_ignore(self, quantize_only_experts: bool):
        ignored_modules: List[str] = ["lm_head"]
        if quantize_only_experts:
            ignored_modules.extend(
                [
                    # embeddings and final norm
                    r"re:.*embed_tokens(?:\..*)?$",
                    r"re:.*\.norm(?:\..*)?$",
                    r"re:.*(?:input|post_attention)_layernorm(?:\..*)?$",
                    # MLA + KDA attention and DSA indexer (whole self_attn tree)
                    r"re:.*\.self_attn(?:\..*)?$",
                    # Hyper-Connection parameters (raw nn.Parameter, safe to be
                    # explicit so any Linear-shaped subclass never accidentally
                    # gets picked up).
                    r"re:.*\.(attn_hc|ffn_hc)(?:\..*)?$",
                    # Router (GateLinear in vLLM hard-codes quant_config=None).
                    r"re:.*\.mlp\.gate(?:\..*)?$",
                    # Shared experts (single non-routed MLP per sparse layer).
                    r"re:.*\.mlp\.shared_experts(?:\..*)?$",
                    # Dense MLP layers keep both the split and any fused form
                    # (compressed_tensors expands `mlp.experts.0.*` to probe
                    # the routed subtree, so this only affects dense blocks).
                    r"re:.*\.mlp\.gate_up_proj(?:\..*)?$",
                    r"re:.*\.mlp\.(gate|up|down)_proj(?:\..*)?$",
                    # MTP layer 45 (next-n predict head, always BF16). The
                    # `.*\.layers\.45` form covers both the multimodal path
                    # (`model.language_model.layers.45.*`) and the text-only
                    # path vLLM uses when building the MTP draft model, whose
                    # prefix drops `language_model.`
                    # (`model.layers.45.mtp_block.*`).
                    r"re:.*\.layers\.45(?:\..*)?$",
                    r"re:^model\.layers\.45(?:\..*)?$",
                    # Vision tower (BF16 in the source FP8 checkpoint).
                    r"re:.*visual(?:\..*)?$",
                ]
            )
            return "glm5_next_experts_only", ignored_modules
        return "default", ignored_modules

    def set_quantization_config(
        self, config: Any, quantization_config: Dict[str, Any]
    ) -> None:
        config.quantization_config = quantization_config
        # Some downstream tools read the text-level config; mirror Kimi-K3's
        # approach and expose the same block there.
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            text_config.quantization_config = quantization_config

    # ------------------------------------------------------------------ #
    # Packing coverage                                                   #
    # ------------------------------------------------------------------ #

    def logical_block_keys(
        self,
        block: torch.nn.Module,
        block_idx: int,
    ) -> set:
        """Enumerate the physical safetensor keys that make up one decoder block.

        The keys returned here drive both the pass-through loader (via
        ``checkpoint_keys_for_model_key`` inside pack.py) and the on-disk names
        of the packed shard, so this method must return **checkpoint-side**
        names — the flat ``hc_attn_fn`` layout the vLLM Glm5NextForConditional-
        Generation modules bind directly. Modeling-side names (``attn_hc.fn``,
        etc.) coming out of ``block.state_dict()`` are reverse-mapped through
        ``_to_physical_keys``, and the fused ``self_attn.conv1d.weight`` is
        expanded to its three per-QKV counterparts.
        """
        prefix = self.get_layer_prefix(block_idx)
        keys: set = set()
        for local_key in block.state_dict():
            model_key = f"{prefix}{local_key}"
            keys.update(self._to_physical_keys(model_key))
        text_config = self._text_config()
        layer_types = list(getattr(text_config, "mlp_layer_types", []) or [])
        if block_idx < len(layer_types) and layer_types[block_idx] == "sparse":
            num_experts = int(text_config.n_routed_experts)
            expert_prefix = f"{prefix}mlp.experts"
            for expert_idx in range(num_experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    keys.add(f"{expert_prefix}.{expert_idx}.{projection}.weight")
        return keys

    def expected_quantized_layer_names(self, model: torch.nn.Module) -> set:
        del model
        text_config = self._text_config()
        num_experts = int(text_config.n_routed_experts)
        layer_types = list(getattr(text_config, "mlp_layer_types", []) or [])
        names: set = set()
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type != "sparse":
                continue
            prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
            for expert_idx in range(num_experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    names.add(f"{prefix}.{expert_idx}.{projection}")
        return names

    def count_extra_shards(self, weight_map: Dict[str, str]) -> int:
        from . import glm5_next_utils

        return glm5_next_utils.count_extra_shards(weight_map)

    def save_extra_weights(
        self,
        weight_dir: str,
        weight_map: Dict[str, str],
        packed_model_path: str,
        next_shard_id: int,
        num_output_shards: int,
        safetensors_index: Dict[str, str],
    ) -> int:
        from . import glm5_next_utils

        return glm5_next_utils.save_extra_weights(
            weight_dir,
            weight_map,
            packed_model_path,
            next_shard_id,
            num_output_shards,
            safetensors_index,
        )

    def copy_artifacts(self, source_dir: str, output_dir: str) -> None:
        for name in self._artifact_names:
            source_path = os.path.join(source_dir, name)
            if os.path.isfile(source_path):
                shutil.copy(source_path, output_dir)
        for name in sorted(os.listdir(source_dir)):
            if (
                name.startswith(("modeling_", "configuration_", "processing_", "image_processing_", "video_processing_"))
                and name.endswith(".py")
            ):
                shutil.copy(os.path.join(source_dir, name), output_dir)
