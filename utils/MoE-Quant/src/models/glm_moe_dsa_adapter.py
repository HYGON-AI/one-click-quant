# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GLM-MoE-DSA model adapter for MoE-Quant."""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .base import ModelAdapter


class GlmMoeDsaAdapter(ModelAdapter):
    """Adapter for GLM-5.3 (``model_type="glm_moe_dsa"``).

    Supports symmetric INT4/INT8 GPTQ, including mixed W4A8 recipes selected
    with --ignore and --force_int8. The legacy --quantize_only_experts flag
    retains the routed-experts-only scope. Next-n layers bypass GPTQ and can
    be RTN INT8-quantized with --mtp-int8 while packing. --mtp-int4 uses INT4
    for routed MTP experts and INT8 for its other non-ignored Linears.
    """

    name = "glm_moe_dsa"
    supports_mtp_int8 = True
    supports_mtp_int4 = True

    _routed_expert_pattern = re.compile(
        r"(?:^|\.)mlp\.experts\.\d+\.(?:down|gate|up)_proj(?:\.|$)"
    )
    _artifact_names = (
        "chat_template.jinja",
        "configuration.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
    )

    @classmethod
    def matches(cls, config: Any) -> bool:
        model_type = str(getattr(config, "model_type", "")).lower()
        architectures = {
            str(value).lower()
            for value in getattr(config, "architectures", []) or []
        }
        return (
            model_type == "glm_moe_dsa"
            or "glmmoedsaforcausallm" in architectures
        )

    # ---------- configuration and validation ----------

    def prepare_config(self, config: Any, world_size: int) -> None:
        num_experts = int(getattr(config, "n_routed_experts", 0))
        if num_experts <= 0:
            raise ValueError("GLM config must set n_routed_experts to a positive integer.")
        if num_experts % world_size != 0:
            raise ValueError(
                f"GLM n_routed_experts ({num_experts}) must be divisible by the world size "
                f"({world_size})."
            )
        config.ep_size = world_size
        config.use_cache = False

    def _validate_group_size(self, args: Any) -> None:
        if args.group_size is None:
            return
        if args.group_size <= 0:
            raise ValueError(f"GLM group_size must be positive, got {args.group_size}.")
        cfg = self.config
        hidden = int(getattr(cfg, "hidden_size", 0))
        moe_intermediate = int(getattr(cfg, "moe_intermediate_size", 0))
        for dim_name, dim in (("hidden_size", hidden), ("moe_intermediate_size", moe_intermediate)):
            if dim <= 0 or dim % args.group_size != 0:
                raise ValueError(
                    f"GLM group_size ({args.group_size}) must divide {dim_name} ({dim})."
                )

    def validate_quantization_args(self, args: Any) -> None:
        if args.bits not in (4, 8):
            raise ValueError("GLM-5.3 currently supports --bits 4 (W4A16) or --bits 8 (W8A16).")
        if not args.quantize_only_experts and not getattr(args, "ignore", None):
            raise ValueError(
                "GLM-5.3 requires --quantize_only_experts (routed experts only) "
                "or an explicit --ignore list for a broader GPTQ scope."
            )
        if not args.sym:
            raise ValueError("GLM-5.3 currently requires --sym (symmetric INT4/INT8 GPTQ).")
        if args.dtype != "bfloat16":
            raise ValueError("GLM-5.3 currently requires --dtype bfloat16.")
        attn_implementation = getattr(args, "attn_implementation", "flash_attention_2")
        if attn_implementation == "flash_attention_2":
            raise ValueError(
                "GLM-5.3 (glm_moe_dsa) does not advertise Flash Attention 2 support; "
                "use --attn_implementation sdpa or eager."
            )
        if args.tie_gptq_handles:
            raise ValueError(
                "--tie_gptq_handles is disabled for GLM-5.3 in this release; gate/up tying "
                "has not been verified yet."
            )
        self._validate_group_size(args)

    def validate_packing_args(self, args: Any) -> None:
        if not args.quantize_only_experts and not getattr(args, "ignore", None):
            raise ValueError(
                "GLM-5.3 packing needs an experts-only GPTQ result or the "
                "--ignore list recorded in metadata.pt."
            )
        if args.dtype != "bfloat16":
            raise ValueError("GLM-5.3 currently requires --dtype bfloat16 for packing.")
        if args.activation_bits not in (16, 8):
            raise ValueError(
                "GLM-5.3 supports --activation-bits 16 (W*A16) or 8 (W4A8 / W8A8)."
            )
        mtp_int8 = getattr(args, "mtp_int8", False)
        mtp_int4 = getattr(args, "mtp_int4", False)
        if mtp_int8 and mtp_int4:
            raise ValueError("--mtp-int8 and --mtp-int4 are mutually exclusive.")
        if (args.activation_bits == 8 or mtp_int8 or mtp_int4) and args.group_size is not None:
            raise ValueError(
                "GLM-5.3 W4A8 / W8A8 requires channel-wise weights; omit --group_size."
            )
        if mtp_int8 or mtp_int4:
            if args.quantize_only_experts:
                raise ValueError("GLM-5.3 MTP RTN quantization requires the explicit --ignore recipe.")
            if not self._mtp_prefixes():
                raise ValueError("GLM-5.3 MTP RTN quantization requires num_nextn_predict_layers > 0.")
        self._validate_group_size(args)

    # ---------- model construction ----------

    def prepare_model(self, model, config: Any, dtype) -> None:
        from . import glm_moe_dsa_utils

        glm_moe_dsa_utils.prepare_glm_moe_dsa_model(model, config, dtype)

    # ---------- checkpoint handling ----------

    def checkpoint_keys_for_model_key(
        self,
        model_key: str,
        weight_map: Dict[str, str],
    ) -> List[str]:
        return [model_key]

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
                "GLM-5.3 FP8 weight is missing its matching *_scale_inv tensor."
            )
        quant_utils.dequantize_state_dict(logical, dtype)
        model_key_set = set(model_keys)
        result: Dict[str, torch.Tensor] = {}
        for key in model_key_set:
            if key not in logical:
                continue
            tensor = logical[key]
            expected_shape = (expected_shapes or {}).get(key)
            if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
                raise ValueError(
                    f"GLM checkpoint tensor {key} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(expected_shape)}."
                )
            result[key] = tensor
        return result

    # ---------- block iteration ----------

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
        from . import glm_moe_dsa_utils

        config = self.config
        attention_mask = glm_moe_dsa_utils.create_glm_causal_mask(
            config, hidden_states, position_ids
        )
        position_embeddings = glm_moe_dsa_utils.compute_position_embeddings(
            config, hidden_states, position_ids
        )
        state = block_state or self.create_block_state(hidden_states, position_ids)
        result = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
            prev_topk_indices=state.get("prev_topk_indices"),
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "GLM decoder layer must return (hidden_states, topk_indices); "
                f"got {type(result).__name__}."
            )
        next_hidden, topk_indices = result
        return next_hidden, {"prev_topk_indices": topk_indices}

    # ---------- quantization scope ----------

    def _mtp_prefixes(self) -> Tuple[str, ...]:
        start = int(self.config.num_hidden_layers)
        count = int(getattr(self.config, "num_nextn_predict_layers", 0))
        return tuple(f"model.layers.{idx}." for idx in range(start, start + count))

    def _mtp_ignore_rules(self) -> List[str]:
        return [f"re:^{re.escape(prefix[:-1])}(?:\\..*)?$" for prefix in self._mtp_prefixes()]

    def default_ignore_rules(self) -> List[str]:
        # Replayed into compressed-tensors config: structural rules must also
        # match sglang's MTP names (model.decoder.* / model.eh_proj, etc.).
        return [
            "lm_head",
            r"re:.*embed_tokens.*",
            r"re:^model\.norm$",
            r"re:.*(?:input_layernorm|post_attention_layernorm)$",
            r"re:.*mlp\.gate(?:\..*)?$",
            r"re:.*self_attn\.(?:q_a_layernorm|kv_a_layernorm)$",
            r"re:.*self_attn\.kv_b_proj$",
            r"re:.*self_attn\.indexer\.k_norm(?:\..*)?$",
            r"re:.*self_attn\.indexer\.(?:weights_proj|wk_weights_proj)$",
            r"re:.*\.(?:eh_proj|enorm|hnorm)$",
            r"re:.*shared_head\..*",
            *self._mtp_ignore_rules(),
        ]

    def mtp_int8_ignore(self, rules: Iterable[str]) -> List[str]:
        blanket_rules = self._mtp_ignore_rules()
        return [rule for rule in rules if rule not in blanket_rules]

    def mtp_int4_targets(self) -> List[str]:
        # sglang NextN registers the decoder under model.decoder.*, unlike
        # the checkpoint's model.layers.<num_hidden_layers>.* names.
        suffix = r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$"
        targets = [f"re:^{re.escape(prefix[:-1])}{suffix}" for prefix in self._mtp_prefixes()]
        if targets:
            targets.append(r"re:^model\.decoder" + suffix)
        return targets

    def legacy_ignore_rules(self) -> List[str]:
        return [
            "model.embed_tokens",
            r"re:.*\.self_attn(?:\..*)?$",
            r"re:.*\.mlp\.gate(?:\..*)?$",
            r"re:.*\.mlp\.shared_experts(?:\..*)?$",
            # sglang fuses dense-MLP gate+up into `gate_up_proj`; cover both spellings.
            r"re:.*\.mlp\.gate_up_proj(?:\..*)?$",
            r"re:.*\.mlp\.(gate|up|down)_proj(?:\..*)?$",
            r"re:.*_layernorm(?:\..*)?$",
            r"re:.*\.(input|post_attention)_layernorm(?:\..*)?$",
            r"re:^model\.norm(?:\..*)?$",
        ]

    # ---------- packing hooks ----------

    def logical_block_keys(
        self,
        block: torch.nn.Module,
        block_idx: int,
    ) -> set:
        prefix = self.get_layer_prefix(block_idx)
        keys = {f"{prefix}{key}" for key in block.state_dict()}
        cfg = self.config
        layer_types = list(getattr(cfg, "mlp_layer_types", []))
        if block_idx < len(layer_types) and layer_types[block_idx] == "sparse":
            num_experts = int(cfg.n_routed_experts)
            expert_prefix = f"{prefix}mlp.experts"
            for expert_idx in range(num_experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    keys.add(f"{expert_prefix}.{expert_idx}.{projection}.weight")
        return keys

    def expected_quantized_layer_names(self, model: torch.nn.Module) -> set:
        del model
        cfg = self.config
        num_experts = int(cfg.n_routed_experts)
        layer_types = list(getattr(cfg, "mlp_layer_types", []))
        names: set = set()
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type != "sparse" or layer_idx >= int(cfg.num_hidden_layers):
                continue
            prefix = f"model.layers.{layer_idx}.mlp.experts"
            for expert_idx in range(num_experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    names.add(f"{prefix}.{expert_idx}.{projection}")
        return names

    def count_extra_shards(self, weight_map: Dict[str, str]) -> int:
        from . import glm_moe_dsa_utils

        return glm_moe_dsa_utils.count_extra_shards(weight_map, self._mtp_prefixes())

    def save_extra_weights(
        self,
        weight_dir: str,
        weight_map: Dict[str, str],
        packed_model_path: str,
        next_shard_id: int,
        num_output_shards: int,
        safetensors_index: Dict[str, str],
        mtp_int8_ignore: Optional[List[str]] = None,
        mtp_pack_int8=None,
        mtp_pack_int4=None,
    ) -> int:
        from . import glm_moe_dsa_utils

        return glm_moe_dsa_utils.save_extra_weights(
            weight_dir,
            weight_map,
            packed_model_path,
            next_shard_id,
            num_output_shards,
            safetensors_index,
            mtp_prefixes=self._mtp_prefixes(),
            mtp_int8_ignore=mtp_int8_ignore,
            mtp_pack_int8=mtp_pack_int8,
            mtp_pack_int4=mtp_pack_int4,
            mtp_int4_targets=self.mtp_int4_targets(),
        )

    def copy_artifacts(self, source_dir: str, output_dir: str) -> None:
        for name in self._artifact_names:
            source_path = os.path.join(source_dir, name)
            if os.path.isfile(source_path):
                shutil.copy(source_path, output_dir)
        # Copy every modeling_*.py the source may ship (remote code), if any.
        for name in sorted(os.listdir(source_dir)):
            if name.startswith("modeling_") and name.endswith(".py"):
                shutil.copy(os.path.join(source_dir, name), output_dir)
            if name.startswith("configuration_") and name.endswith(".py"):
                shutil.copy(os.path.join(source_dir, name), output_dir)
