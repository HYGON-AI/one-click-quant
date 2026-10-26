"""MiMo-V2 source-dtype GPTQ recipe and source checkpoint decoding.

Only the text backbone participates in GPTQ.  The original MTP follows the
existing MiMo RTN converter, and unrelated submodels retain their source tensors.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Optional

import torch

from .base import ModelAdapter


@lru_cache(maxsize=1)
def get_reference_converter():
    """Use the same numerical decode, QKV layout and RTN contract as MiMo RTN."""
    path = Path(__file__).resolve().parents[4] / "examples/mimo/quantize_mimo_v26_flash_rl_int4_channel.py"
    name = "_moe_quant_mimo_reference_converter"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MiMo reference converter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class MiMoV2Adapter(ModelAdapter):
    """Fixed MXFP4->W4A8 / FP8->W8A8 / BF16,FP32->keep recipe."""

    name = "mimo_v2"
    supports_mtp_int8 = True
    load_only_required_tensors = True

    @classmethod
    def matches(cls, config: Any) -> bool:
        return str(getattr(config, "model_type", "")).lower() == "mimo_v2"

    def __init__(self, config=None):
        super().__init__(config)
        source = getattr(config, "_name_or_path", None)
        if not source:
            raise ValueError("MiMo requires a local source checkpoint in config._name_or_path.")
        self.source = Path(source).resolve()
        converter = get_reference_converter()
        # Read the original JSON, since the generic loader removes its native
        # quantization_config before creating a BF16 calibration model.
        self.layout, self.weight_map, self.meta = converter.load_source(self.source)
        converter.self_check_scope(self.layout, self.meta)
        self.layer_bits = {}
        for name, tensor in self.meta.items():
            match = re.match(r"^model\.layers\.(\d+)\.", name)
            if (not match or int(match[1]) >= int(self.layout.config["num_hidden_layers"])
                    or not name.endswith(".weight")):
                continue
            mode = self.layout.classify(name, tensor["dtype"])
            if mode == "keep":
                continue
            for output_name in converter.output_weight_names(name):
                self.layer_bits[output_name.removesuffix(".weight")] = 4 if mode == "w4a8" else 8
        self._manifest = {
            "version": 1,
            "source_tp": self.layout.source_tp,
            "source_config_sha256": _digest(self.layout.config),
            "source_tensor_metadata_sha256": _digest({
                name: {"dtype": tensor["dtype"], "shape": tensor["shape"]}
                for name, tensor in self.meta.items()
            }),
            "source_weight_map_sha256": _digest(self.weight_map),
            "source_quantization_config": self.layout.config.get("quantization_config", {}),
            "source_attention_projection_layout": self.layout.config.get("attention_projection_layout", "split"),
            "output_attention_projection_layout": "split",
            "backbone_algorithm": "gptq",
            "mtp_algorithm": "rtn",
            "layer_bits": dict(sorted(self.layer_bits.items())),
        }

    def prepare_config(self, config, world_size: int) -> None:
        count = int(self.layout.config["n_routed_experts"])
        if count <= 0 or count % world_size:
            raise ValueError(f"MiMo n_routed_experts={count} must be divisible by world_size={world_size}.")
        config.ep_size = world_size
        config.use_cache = False
        config.attention_projection_layout = "split"

    def build_empty_model(self, config, dtype, attn_implementation=None):
        from .mimo_v2_utils import build_model

        return build_model(config, dtype, attn_implementation or "eager")

    def forward_block(self, block, hidden_states, position_ids, block_state=None):
        from .mimo_v2_utils import forward_block

        return forward_block(block, hidden_states, position_ids, block_state)

    def _source_key(self, model_key: str) -> str:
        if model_key in self.weight_map:
            return model_key
        key = model_key.replace(".post_attention_layernorm.", ".pre_mlp_layernorm.")
        if key in self.weight_map:
            return key
        fused = re.sub(r"\.self_attn\.[qkv]_proj\.weight$", ".self_attn.qkv_proj.weight", key)
        return fused if fused in self.weight_map else key

    def checkpoint_keys_for_model_key(self, model_key: str, weight_map: dict) -> list[str]:
        key = self._source_key(model_key)
        result = [key]
        for suffix in ("_scale", "_scale_inv"):
            if key + suffix in weight_map:
                result.append(key + suffix)
        return result

    def materialize_state_dict(self, physical_state_dict, model_keys: Iterable[str], dtype,
                               expected_shapes=None):
        converter = get_reference_converter()
        result, decoded_qkv = {}, {}
        for model_key in sorted(model_keys):
            source_key = self._source_key(model_key)
            if source_key not in physical_state_dict:
                raise KeyError(f"Missing MiMo source weight {source_key} for {model_key}.")
            source = physical_state_dict[source_key]
            if source_key.endswith(".self_attn.qkv_proj.weight"):
                if source_key not in decoded_qkv:
                    decoded_qkv[source_key] = converter.split_qkv(
                        source_key, source, physical_state_dict.get(source_key + "_scale_inv"), self.layout
                    )
                part = re.search(r"\.self_attn\.([qkv])_proj\.weight$", model_key)
                if part is None:
                    raise ValueError(f"Calibration model must use canonical split QKV: {model_key}")
                tensor = decoded_qkv[source_key][part[1]]
            elif self.meta[source_key]["dtype"] == "U8":
                tensor = converter.decode_mxfp4(
                    source, physical_state_dict[source_key + "_scale"],
                    int(self.layout.config["quantization_config"].get("mxfp4_block_size", 32)),
                )
            elif self.meta[source_key]["dtype"] in converter.FP8_DTYPES:
                tensor = converter.decode_fp8(
                    source, physical_state_dict[source_key + "_scale_inv"],
                    tuple(self.layout.config["quantization_config"]["weight_block_size"]),
                )
            else:
                tensor = source
            shape = (expected_shapes or {}).get(model_key)
            if shape is not None and tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"MiMo shape mismatch for {model_key}: {tuple(tensor.shape)} != {tuple(shape)}")
            # Router correction biases are FP32 in the original checkpoint and
            # the calibration model.  Converting them to BF16 changes routing.
            preserve_fp32 = self.meta[source_key]["dtype"] == "F32"
            result[model_key] = tensor.to(dtype) if tensor.is_floating_point() and not preserve_fp32 else tensor
        return result

    def default_ignore_rules(self) -> list[str]:
        converter = get_reference_converter()
        ignored = {"lm_head", "model.embed_tokens", r"re:^model\.mtp\..*"}
        for name, tensor in self.meta.items():
            if name.endswith(".weight") and self.layout.classify(name, tensor["dtype"]) == "keep":
                for output_name in converter.output_weight_names(name):
                    ignored.update(converter.module_aliases(output_name.removesuffix(".weight")))
        return sorted(ignored)

    def resolve_ignore(self, ignore=None, quantize_only_experts=False, force_int8=None):
        rules = self.default_ignore_rules()
        if quantize_only_experts or force_int8 or (ignore and sorted(ignore) != rules):
            raise ValueError("MiMo scope and bit widths are fixed by source dtype; omit --ignore, --force_int8 and --quantize_only_experts.")
        return rules

    def resolve_layer_bits(self, layer_name, default_bits, force_int8=None):
        del default_bits
        if force_int8:
            raise ValueError("MiMo does not support overriding source-dtype bit widths.")
        if layer_name not in self.layer_bits:
            raise ValueError(f"MiMo source policy does not select {layer_name} for GPTQ.")
        return self.layer_bits[layer_name]

    def expected_quantized_layer_names(self, model=None) -> set[str]:
        del model
        return set(self.layer_bits)

    def validate_quantization_args(self, args) -> None:
        self.resolve_ignore(getattr(args, "ignore", None), getattr(args, "quantize_only_experts", False),
                            getattr(args, "force_int8", None))
        if args.bits != 4 or not args.sym or args.group_size is not None or args.dtype != "bfloat16":
            raise ValueError("MiMo requires --bits 4 --sym --dtype bfloat16 and channel weights (omit --group_size).")
        if getattr(args, "attn_implementation", None) != "eager":
            raise ValueError("MiMo calibration currently requires --attn_implementation eager.")
        if getattr(args, "resume", False):
            raise ValueError("MiMo --resume is unsupported: prior GPTQ weights must be replayed during calibration.")
        save_dir = getattr(args, "save_dir", None)
        if save_dir and Path(save_dir).exists() and any(Path(save_dir).iterdir()):
            raise ValueError(f"MiMo GPTQ save_dir must be empty: {save_dir}")

    def validate_packing_args(self, args) -> None:
        self.resolve_ignore(getattr(args, "ignore", None), getattr(args, "quantize_only_experts", False),
                            getattr(args, "force_int8", None))
        if args.bits != 4 or args.group_size is not None or args.dtype != "bfloat16" or args.activation_bits != 8:
            raise ValueError("MiMo packing requires the fixed mixed channel W4A8/W8A8 recipe and --dtype bfloat16 --activation-bits 8.")
        if not getattr(args, "mtp_int8", False) or getattr(args, "mtp_int4", False):
            raise ValueError("MiMo old MTP uses the original source-dtype RTN conversion; pass --mtp-int8 (omit --mtp-int4).")

    def quantization_metadata(self) -> dict:
        return {"mimo_source_manifest": copy.deepcopy(self._manifest)}

    def validate_quantization_metadata(self, metadata: dict) -> None:
        if metadata.get("mimo_source_manifest") != self._manifest:
            raise ValueError("MiMo GPTQ source manifest mismatch or missing: use the identical original checkpoint and recipe.")
        if metadata.get("complete") is not True:
            raise ValueError("MiMo GPTQ metadata is incomplete; wait for a successful full calibration run.")
        if metadata.get("sym") is not True or metadata.get("bits") != 4 or metadata.get("group_size") is not None:
            raise ValueError("MiMo GPTQ metadata must specify symmetric channel mixed INT4/INT8 quantization.")

    def pack_checkpoint(self, args, metadata) -> bool:
        from .mimo_v2_checkpoint import pack_checkpoint

        self.validate_packing_args(args)
        self.validate_quantization_metadata(metadata)
        pack_checkpoint(self, args)
        return True
