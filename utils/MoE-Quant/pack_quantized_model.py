import os
import gc
import json
import shutil
import argparse
from collections import defaultdict
from typing import Optional, Any

from tqdm import tqdm
import torch
from safetensors.torch import save_file
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from compressed_tensors.compressors import pack_to_int32

from src import quant_utils
from src import loading_utils
from src.models import qwen3_5_moe_utils


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the DeepSeek model",
    )
    parser.add_argument(
        "--quantized_model_path",
        type=str,
        required=True,
        help="Path to quantized model."
    )
    parser.add_argument(
        "--packed_model_path",
        type=str,
        required=True,
        help="Whether to save packed model."
    )
     # Misc params
    parser.add_argument(
        "--dtype",
        default="float16",
        type=str,
        choices=["float16", "bfloat16"],
        help="Torch dtype used."
    )
    parser.add_argument(
        "--activation-bits",
        type=int,
        choices=[16, 8],
        default=16,
        help="Activation precision declared in quantization_config; 16 keeps weight-only W4A16, 8 declares dynamic per-token INT8 activations.",
    )
    args = parser.parse_args()
    return args


def is_subset(set1: set, set2: set):
    return set1 <= set2


def pack_weight(
    weight: dict[torch.Tensor],
    bits: int,
    sym: bool,
    group_size: Optional[int] = None,
) -> dict[torch.Tensor]:
    compressed_data = {}
    qweight, scale, zero = weight['qweight'], weight['scale'], weight['zero']
    group_size = group_size or qweight.shape[-1]
    qweight_shifted = qweight.to(torch.int8) - zero.repeat_interleave(group_size, dim=-1).to(torch.int8)
    qweight_packed = pack_to_int32(qweight_shifted, bits)
    compressed_data = {
        "weight_packed": qweight_packed,
        "weight_shape": torch.tensor(qweight.shape),
        "weight_scale": scale
    }
    if not sym:
        compressed_data["weight_zero_point"] = weight['zero']
    return compressed_data


def prepare_quantization_config(args: argparse.Namespace, is_qwen3_5_moe: bool = False) -> dict[str, Any]:
    input_activations = None
    if args.activation_bits == 8:
        input_activations = {
            "dynamic": True,
            "group_size": None,
            "num_bits": 8,
            "observer": "minmax",
            "observer_kwargs": {},
            "strategy": "token",
            "symmetric": True,
            "type": "int",
        }

    ignored_modules = ["lm_head"]
    ignore_rule = "default"
    if args.quantize_only_experts:
        if is_qwen3_5_moe:
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
                "mtp.fc",
                r"re:^mtp\.layers\.0\.mlp\.gate$",
                r"re:^mtp\.layers\.0\.mlp\.shared_expert\.(gate|up|down)_proj$",
                r"re:^mtp\.layers\.0\.mlp\.shared_expert_gate$",
                r"re:^mtp\.layers\.0\.self_attn\.(q|k|v|o)_proj$",
                "mtp.pre_fc_norm_embedding",
                "mtp.pre_fc_norm_hidden",
            ]
            ignore_rule = "qwen3_5_moe_experts_only"
        else:
            ignored_modules += [
                r"re:.*self_attn.*",
                r"re:.*shared_experts.*",
                r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
            ]
            ignore_rule = "default_experts_only"
    print(f"[INFO] quantization_config ignore rule={ignore_rule}, count={len(ignored_modules)}")
    return {
        "config_groups": {
            "group_0": {
                "input_activations": input_activations,
                "output_activations": None,
                "targets": [
                    "Linear"
                ],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": args.group_size,
                    "num_bits": args.bits,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int"
                }
            }
        },
        "format": "pack-quantized",
        "ignore": ignored_modules,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }


def main():
    args = parse_args()

    dtype = getattr(torch, args.dtype)

    # Load DeepSeek model
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    is_qwen3_5_moe = getattr(config, "model_type", None) == "qwen3_5_moe_text"

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).eval()
        model.config.use_cache = False
        if is_qwen3_5_moe:
            qwen3_5_moe_utils.prepare_qwen3_5_moe_model(model, config, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Load quantization metadata
    metadata = torch.load(os.path.join(args.quantized_model_path, "metadata.pt"))
    args.bits = metadata["bits"]
    args.group_size = metadata["group_size"]
    args.quantize_only_experts = metadata["quantize_only_experts"]
    # Currently we do not support asymmetric quantization
    args.sym = True

    # Resolve input weights through the HuggingFace safetensors index instead of
    # assuming DeepSeek's model-xxxxx-of-000163 naming convention.
    weight_dir = args.model_name_or_path
    weight_map = loading_utils.load_safetensors_weight_map(weight_dir)
    num_mtp_shards = qwen3_5_moe_utils.count_mtp_shards(weight_map) if is_qwen3_5_moe else 0
    num_output_shards = len(model.model.layers) + 2 + num_mtp_shards
    current_output_shard_id = 1
    quantized_layer_names = defaultdict(list)
    for layer_name in sorted(os.listdir(args.quantized_model_path)):
        if os.path.isdir(os.path.join(args.quantized_model_path, layer_name)):
            block_idx = int(layer_name.split(".")[2])
            quantized_layer_names[block_idx].append(layer_name)
    safetensors_index = {}
    # Prepare directory to save packed weights
    os.makedirs(args.packed_model_path, exist_ok=True)

    loaded_shards = set()
    param_buffer = {}
    loaded = loading_utils.ensure_params_loaded(
        weight_dir,
        param_buffer,
        ["model.embed_tokens.weight"],
        weight_map,
        loaded_shards,
    )
    if loaded:
        print(f"Loaded embedding parameter from shards: {loaded}")

    # Save embeddings
    embedding_state_dict = {
        k: v
        for k, v in param_buffer.items()
        if k in ("model.embed_tokens.weight", "model.embed_tokens.weight_scale_inv")
    }
    if not quant_utils.can_dequantize_from_fp8(embedding_state_dict):
        raise RuntimeError(
            "The input embedding is stored as FP8 but model.embed_tokens.weight_scale_inv is missing."
        )
    quant_utils.dequantize_state_dict(embedding_state_dict, dtype)
    current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
    save_file(
        {"model.embed_tokens.weight": embedding_state_dict["model.embed_tokens.weight"]},
        os.path.join(args.packed_model_path, current_output_shard_path)
    )
    safetensors_index["model.embed_tokens.weight"] = current_output_shard_path
    param_buffer.pop("model.embed_tokens.weight", None)
    param_buffer.pop("model.embed_tokens.weight_scale_inv", None)

    # Process blocks
    for block_idx, block in tqdm(
        enumerate(model.model.layers),
        desc="Processing transformer blocks",
        total=len(model.model.layers)
    ):
        current_output_shard_id += 1
        prefix = f"model.layers.{block_idx}."
        block_keys_with_prefix = set(f"{prefix}{k}" for k in block.state_dict())

        loading_utils.ensure_params_loaded(
            weight_dir,
            param_buffer,
            block_keys_with_prefix,
            weight_map,
            loaded_shards,
        )

        block_state_dict = {k: param_buffer[k] for k in param_buffer if k.startswith(prefix)}
        quant_utils.dequantize_state_dict(block_state_dict, dtype)

        for layer_name in quantized_layer_names[block_idx]:
            weight_state_dict = torch.load(
                os.path.join(args.quantized_model_path, layer_name, "quantized_weight.pt"),
                weights_only=True,
                map_location="cpu"
            )
            packed_weight_state_dict = pack_weight(weight_state_dict, args.bits, args.sym, args.group_size)
            block_state_dict.pop(f"{layer_name}.weight")
            block_state_dict.pop(f"{layer_name}.weight_scale_inv", None)
            block_state_dict.update({f"{layer_name}.{k}": v for k, v in packed_weight_state_dict.items()})

        # Save block
        current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
        save_file(
            block_state_dict,
            os.path.join(args.packed_model_path, current_output_shard_path)
        )
        for k in block_state_dict:
            safetensors_index[k] = current_output_shard_path

        for k in block_keys_with_prefix:
            param_buffer.pop(k, None)

        del block_state_dict
        gc.collect()

    # Load final model tensors by name; their shard can be anywhere in the input index.
    loading_utils.ensure_params_loaded(
        weight_dir,
        param_buffer,
        ["lm_head.weight", "model.norm.weight"],
        weight_map,
        loaded_shards,
    )

    # Save lm head
    current_output_shard_id += 1
    current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
    save_file(
        {
            "lm_head.weight": param_buffer["lm_head.weight"],
            "model.norm.weight": param_buffer["model.norm.weight"]
        },
        os.path.join(args.packed_model_path, current_output_shard_path)
    )
    safetensors_index["lm_head.weight"] = current_output_shard_path
    safetensors_index["model.norm.weight"] = current_output_shard_path
    # Copy MTP weights verbatim (Qwen3.5-MoE only)
    if is_qwen3_5_moe:
        qwen3_5_moe_utils.save_mtp_weights(
            weight_dir,
            weight_map,
            args.packed_model_path,
            current_output_shard_id + 1,
            num_output_shards,
            safetensors_index,
        )
    # Save safetensors index
    with open(os.path.join(args.packed_model_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": safetensors_index}, f)
    # Add quantization metadata
    config.quantization_config = prepare_quantization_config(args, is_qwen3_5_moe)
    # Save configs
    config.save_pretrained(args.packed_model_path)
    model.generation_config.save_pretrained(args.packed_model_path)
    # Save tokenizer
    tokenizer.save_pretrained(args.packed_model_path)
    # Copy the modeling file shipped with the input model directory.
    modeling_files = sorted(
        name for name in os.listdir(args.model_name_or_path)
        if name.startswith("modeling_") and name.endswith(".py")
    )
    if modeling_files:
        shutil.copy(
            os.path.join(args.model_name_or_path, modeling_files[0]),
            args.packed_model_path,
        )


if __name__ == "__main__":
    main()