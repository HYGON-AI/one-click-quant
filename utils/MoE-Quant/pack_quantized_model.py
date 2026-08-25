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
from transformers import AutoConfig, AutoTokenizer
from compressed_tensors.compressors import pack_to_int32

from src import quant_utils
from src import loading_utils
from src.models import ModelAdapter, get_model_adapter


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


def prepare_quantization_config(
    args: argparse.Namespace,
    adapter: ModelAdapter,
) -> dict[str, Any]:
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

    ignore_rule, ignored_modules = adapter.get_quantization_ignore(args.quantize_only_experts)
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

    # Load model configuration and select its structural adapter.
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    adapter = get_model_adapter(config)
    print(f"[INFO] model adapter={adapter.name}")
    adapter.prepare_config(config, world_size=1)

    with init_empty_weights():
        model = adapter.build_empty_model(config, torch.bfloat16).eval()
        model.config.use_cache = False
        adapter.prepare_model(model, config, torch.bfloat16)

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
    num_extra_shards = adapter.count_extra_shards(weight_map)
    transformer_layers = adapter.get_transformer_layers(model)
    num_output_shards = len(transformer_layers) + 2 + num_extra_shards
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
        [adapter.embedding_weight_key()],
        weight_map,
        loaded_shards,
    )
    if loaded:
        print(f"Loaded embedding parameter from shards: {loaded}")

    # Save embeddings
    embedding_state_dict = {
        k: v
        for k, v in param_buffer.items()
        if k in adapter.embedding_state_keys()
    }
    if not quant_utils.can_dequantize_from_fp8(embedding_state_dict):
        raise RuntimeError(
            "The input embedding is stored as FP8 but model.embed_tokens.weight_scale_inv is missing."
        )
    quant_utils.dequantize_state_dict(embedding_state_dict, dtype)
    current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
    save_file(
        {adapter.embedding_weight_key(): embedding_state_dict[adapter.embedding_weight_key()]},
        os.path.join(args.packed_model_path, current_output_shard_path)
    )
    safetensors_index[adapter.embedding_weight_key()] = current_output_shard_path
    param_buffer.pop(adapter.embedding_weight_key(), None)
    param_buffer.pop(adapter.embedding_weight_key() + "_scale_inv", None)

    # Process blocks
    for block_idx, block in tqdm(
        enumerate(transformer_layers),
        desc="Processing transformer blocks",
        total=len(transformer_layers)
    ):
        current_output_shard_id += 1
        prefix = adapter.get_layer_prefix(block_idx)
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

    final_tensor_keys = adapter.get_final_tensor_keys()
    loading_utils.ensure_params_loaded(
        weight_dir,
        param_buffer,
        final_tensor_keys,
        weight_map,
        loaded_shards,
    )

    # Save final tensors
    current_output_shard_id += 1
    current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
    save_file(
        {key: param_buffer[key] for key in final_tensor_keys},
        os.path.join(args.packed_model_path, current_output_shard_path)
    )
    for key in final_tensor_keys:
        safetensors_index[key] = current_output_shard_path
    current_output_shard_id = adapter.save_extra_weights(
        weight_dir,
        weight_map,
        args.packed_model_path,
        current_output_shard_id + 1,
        num_output_shards,
        safetensors_index,
    )
    # Save safetensors index
    with open(os.path.join(args.packed_model_path, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {}, "weight_map": safetensors_index},
            f,
            indent=2,
        )
        f.write("\n")
    # Add quantization metadata
    config.quantization_config = prepare_quantization_config(args, adapter)
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