"""Step-5 GPTQ adapter with global-name expert parallelism."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from functools import lru_cache
import sys
from typing import Any

import torch

from .base import ModelAdapter
from .step5_config import Step5Config, Step5VLConfig
from .step5_utils import Step5CalibrationModel


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=1)
def get_reference_converter():
    path = Path(__file__).resolve().parents[4] / "examples/step/quantize_step5_fp8_channel_to_w4a8.py"
    name = "_moe_quant_step5_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Step5Adapter(ModelAdapter):
    name = "step5"
    load_only_required_tensors = True
    supports_mtp_int4 = True
    _routed_expert_pattern = re.compile(r"(?:^|\.)moe\.experts\.\d+\.(?:gate|up|down)_proj(?:\.|$)")

    @classmethod
    def matches(cls, config: Any) -> bool:
        return str(getattr(config, "model_type", "")).lower() == "step5_vl"

    def __init__(self, config=None):
        super().__init__(config)
        source = getattr(config, "_name_or_path", None)
        if not source:
            raise ValueError("Step5 requires a local source checkpoint.")
        self.source = Path(source).resolve()
        converter = get_reference_converter()
        raw = converter.load_config(self.source)
        self.raw_config = raw
        self.text = raw["text_config"]
        self.weight_map, self.meta = converter.read_metadata(self.source)
        self.plan, self.consumed_scales = converter.build_plan(raw, self.meta)
        self.moe_layers = set(self.text["moe_layer_list"])
        self.num_experts = int(self.text["moe_num_experts"])
        self.layer_bits = {item.output.removesuffix(".weight"): 4 for item in self.plan if item.int4}
        backbone = re.compile(
            r"^model\.layers\.(\d+)\."
            r"(?:self_attn\.(?:q|k|v|o|g)_proj|"
            r"(?:mlp|share_expert)\.(?:gate|up|down)_proj)\.weight$"
        )
        for name, (dtype, shape) in self.meta.items():
            match = backbone.fullmatch(name)
            if match and int(match[1]) < int(self.text["num_hidden_layers"]):
                if len(shape) != 2 or shape[1] % 8:
                    raise ValueError(f"Unsupported Step5 INT4 backbone shape: {name}: {shape}")
                self.layer_bits[name.removesuffix(".weight")] = 4
        if "lm_head.weight" not in self.meta:
            raise ValueError("Step5 checkpoint is missing lm_head.weight")
        if len(self.meta["lm_head.weight"][1]) != 2 or self.meta["lm_head.weight"][1][1] % 8:
            raise ValueError("Step5 lm_head must be a 2D INT4-compatible matrix")
        self.layer_bits["lm_head"] = 4
        self.mtp_names = set()
        start = int(self.text["num_hidden_layers"])
        end = start + int(self.text.get("num_nextn_predict_layers", 0))
        mtp = re.compile(
            r"^model\.layers\.(\d+)\.(?:"
            r"self_attn\.(?:q|k|v|o|g)_proj|"
            r"mlp\.(?:gate|up|down)_proj|"
            r"transformer\.shared_head\.output)\.weight$"
        )
        for name, (_, shape) in self.meta.items():
            match = mtp.fullmatch(name)
            if match and start <= int(match[1]) < end:
                if len(shape) != 2 or shape[1] % 8:
                    raise ValueError(f"Unsupported Step5 MTP INT4 shape: {name}: {shape}")
                self.mtp_names.add(name.removesuffix(".weight"))
        if end > start and not self.mtp_names:
            raise ValueError("Step5 MTP layers configured but no quantizable MTP weights found")
        self._manifest = {
            "version": 2,
            "source_config_sha256": _digest(raw),
            "source_weight_map_sha256": _digest(self.weight_map),
            "source_tensor_metadata_sha256": _digest(self.meta),
            "backbone_algorithm": "gptq",
            "mtp_algorithm": "rtn_int4",
            "quantized_layers": sorted(self.layer_bits),
            "mtp_layers": sorted(self.mtp_names),
        }

    def prepare_config(self, config, world_size: int) -> None:
        if self.num_experts % world_size:
            raise ValueError(f"Step5 experts={self.num_experts} must divide world_size={world_size}.")
        config.text_config.ep_size = world_size
        config.text_config.use_cache = False

    def build_empty_model(self, config, dtype, attn_implementation=None):
        del attn_implementation
        return Step5CalibrationModel(config.text_config, dtype=dtype).eval()

    def get_embedding_module(self, model):
        return model.model.embed_tokens

    def embedding_weight_key(self):
        return "model.embed_tokens.weight"

    def get_transformer_layers(self, model):
        return model.model.layers

    def get_layer_prefix(self, block_idx):
        return f"model.layers.{block_idx}."

    def get_final_tensor_keys(self):
        return ["lm_head.weight", "model.norm.weight"]

    def checkpoint_keys_for_model_key(self, model_key, weight_map):
        match = re.match(r"^(model\.layers\.(\d+)\.moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj))(\.weight)$", model_key)
        if match and model_key not in weight_map:
            bank = f"model.layers.{match[2]}.moe.{match[4]}.weight"
            result = [bank]
            scale = bank.removesuffix(".weight") + ".weight_scale"
            if scale in weight_map:
                result.append(scale)
            return result
        result = [model_key]
        if model_key + "_scale_inv" in weight_map:
            result.append(model_key + "_scale_inv")
        if model_key + "_scale" in weight_map:
            result.append(model_key + "_scale")
        return result

    @staticmethod
    def _decode(value, scale=None):
        if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz):
            if scale is None:
                raise KeyError("Step5 FP8 tensor is missing weight_scale.")
            return value.float() * scale.float()
        return value

    def materialize_state_dict(self, physical_state_dict, model_keys, dtype, expected_shapes=None):
        result = {}
        for key in model_keys:
            match = re.match(r"^model\.layers\.(\d+)\.moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", key)
            if match and key not in physical_state_dict:
                bank_key = f"model.layers.{match[1]}.moe.{match[3]}.weight"
                value = physical_state_dict[bank_key][int(match[2])]
                scale = physical_state_dict.get(bank_key.removesuffix(".weight") + ".weight_scale")
                if scale is not None:
                    value = self._decode(value, scale[int(match[2])])
                result[key] = value.to(dtype)
                continue
            if key not in physical_state_dict:
                raise KeyError(f"Missing Step5 source tensor {key}")
            value = physical_state_dict[key]
            if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz):
                value = self._decode(value, physical_state_dict.get(key.removesuffix(".weight") + ".weight_scale"))
            preserve = value.dtype == torch.float32
            if key.endswith(".moe.gate.weight"):
                value = value.float()
                preserve = True
            result[key] = value if preserve else (value.to(dtype) if value.is_floating_point() else value)
        if expected_shapes:
            for key, shape in expected_shapes.items():
                if key in result and tuple(result[key].shape) != tuple(shape):
                    raise ValueError(f"Step5 shape mismatch for {key}: {tuple(result[key].shape)} != {shape}")
        return result

    def default_ignore_rules(self):
        return list(self.raw_config["quantization_config"]["ignore"])

    def resolve_ignore(self, ignore=None, quantize_only_experts=False, force_int8=None):
        if quantize_only_experts:
            raise ValueError("Step5 full INT4 recipe does not support --quantize_only_experts.")
        if force_int8:
            raise ValueError("Step5 full INT4 recipe does not support --force_int8.")
        rules = [
            rule for rule in self.default_ignore_rules()
            if rule not in (
                r"re:.*lm_head$",
                r"re:.*model\.layers\.\d+\.self_attn\..*",
                r"re:.*model\.layers\.\d+\.mlp\..*",
                r"re:.*model\.layers\.\d+\.share_expert\..*",
            )
        ]
        # MTP bypasses GPTQ. Its checkpoint weights are handled during packing.
        if self.mtp_names:
            start = int(self.text["num_hidden_layers"])
            end = start + int(self.text.get("num_nextn_predict_layers", 0))
            rules.append(
                rf"re:^model\.layers\.(?:{'|'.join(map(str, range(start, end)))})\."
            )
        if ignore:
            if sorted(ignore) != sorted(rules):
                raise ValueError("Step5 full INT4 recipe uses a fixed ignore scope.")
        return rules

    def resolve_layer_bits(self, layer_name, default_bits, force_int8=None):
        del default_bits, force_int8
        if layer_name not in self.layer_bits:
            raise ValueError(f"Step5 policy does not select {layer_name}")
        return 4

    def expected_quantized_layer_names(self, model=None):
        del model
        return set(self.layer_bits)

    def validate_quantization_args(self, args):
        self.resolve_ignore(args.ignore, args.quantize_only_experts, args.force_int8)
        if args.bits != 4 or not args.sym or args.group_size is not None or args.dtype != "bfloat16":
            raise ValueError("Step5 GPTQ requires --bits 4 --sym --dtype bfloat16 and channel weights.")
        if args.attn_implementation not in ("eager", None):
            raise ValueError("Step5 calibration requires --attn-implementation eager.")
        if args.resume:
            raise ValueError("Step5 resume is unsupported.")
        sparse = self.text.get("sparse_config", {})
        dense_limit = int(sparse.get("topk", 0)) * int(sparse.get("region_block_size", 0))
        if sparse.get("enabled") and args.max_sequence_length > dense_limit:
            raise ValueError(
                f"Step5 calibration requires max_sequence_length <= {dense_limit}: "
                "within this range all DSA regions are selected and dense causal attention is equivalent."
            )
        if self.text.get("rope_scaling", {}).get("factor", 1.0) != 1.0:
            raise ValueError("Step5 calibration currently supports RoPE scaling factor=1 only.")
        for field in ("swiglu_limits", "swiglu_limits_shared"):
            if any(v not in (7, 7.0) for v in self.text.get(field, [])[:self.text["num_hidden_layers"]]):
                raise ValueError(f"Step5 calibration expects {field}=7 for this checkpoint.")
        if args.num_calibration_samples < 1:
            raise ValueError("Step5 calibration needs a positive sample count.")
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if args.num_calibration_samples % world:
            raise ValueError("Step5 calibration sample count must be divisible by the EP process count.")
        dataset = Path(args.dataset_name_or_path)
        self.calibration_metadata = {
            "dataset": str(dataset),
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest() if dataset.is_file() else None,
            "num_samples": args.num_calibration_samples,
            "max_sequence_length": args.max_sequence_length,
            "seed": args.seed,
            "rel_damp": args.rel_damp,
            "quantization_scale": args.quantization_scale,
            "quantization_order": args.quantization_order,
            "ep_size": world,
            "activation_fake_quant": False,
            "attention": "causal/full and sliding; all DSA regions within calibration length",
        }
        save_dir = getattr(args, "save_dir", None)
        if not save_dir or (Path(save_dir).exists() and any(Path(save_dir).iterdir())):
            raise ValueError(f"Step5 GPTQ requires an empty save_dir: {save_dir}")

    def validate_packing_args(self, args):
        if args.bits != 4 or args.group_size is not None or args.dtype != "bfloat16" or args.activation_bits != 8:
            raise ValueError("Step5 packing requires channel W4A8 with --dtype bfloat16.")
        if args.mtp_int8 or (self.mtp_names and not args.mtp_int4):
            raise ValueError("Step5 MTP requires --mtp-int4 (INT4 RTN) and does not support --mtp-int8.")

    def quantization_metadata(self):
        result = {"step5_source_manifest": copy.deepcopy(self._manifest)}
        if hasattr(self, "calibration_metadata"):
            result["step5_calibration"] = copy.deepcopy(self.calibration_metadata)
        return result

    def validate_quantization_metadata(self, metadata):
        if metadata.get("step5_source_manifest") != self._manifest:
            raise ValueError("Step5 GPTQ metadata does not match the source checkpoint.")
        if metadata.get("complete") is not True:
            raise ValueError("Step5 GPTQ metadata is incomplete.")
        if not metadata.get("sym") or metadata.get("bits") != 4 or metadata.get("group_size") is not None:
            raise ValueError("Step5 metadata must describe symmetric channel INT4.")

    def pack_checkpoint(self, args, metadata):
        from .step5_checkpoint import pack_checkpoint
        self.validate_packing_args(args)
        self.validate_quantization_metadata(metadata)
        pack_checkpoint(self, args, metadata)
        return True

