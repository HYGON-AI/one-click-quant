import os
import gc
import json
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
    if qweight.ndim != 2:
        raise ValueError(f"qweight must be 2D, got shape {tuple(qweight.shape)}.")
    effective_group_size = qweight.shape[-1] if group_size is None else group_size
    if effective_group_size <= 0:
        raise ValueError(f"group_size must be positive, got {effective_group_size}.")
    if qweight.shape[-1] % effective_group_size != 0:
        raise ValueError(
            f"group_size ({effective_group_size}) must divide the quantized "
            f"weight input dimension ({qweight.shape[-1]})."
        )
    expected_meta_shape = (
        qweight.shape[0],
        qweight.shape[-1] // effective_group_size,
    )
    if tuple(scale.shape) != expected_meta_shape or tuple(zero.shape) != expected_meta_shape:
        raise ValueError(
            "Quantization scale and zero-point shapes must both be "
            f"{expected_meta_shape}, got scale={tuple(scale.shape)} and "
            f"zero={tuple(zero.shape)}."
        )
    qweight_shifted = qweight.to(torch.int8) - zero.repeat_interleave(
        effective_group_size, dim=-1
    ).to(torch.int8)
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
    weight_strategy = "channel" if args.group_size is None else "group"
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
                    "strategy": weight_strategy,
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
        model = adapter.build_packing_model(config, torch.bfloat16).eval()
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
    adapter.validate_packing_args(args)

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
        layer_path = os.path.join(args.quantized_model_path, layer_name)
        if not os.path.isdir(layer_path):
            continue
        block_idx = adapter.get_block_index_from_layer_name(layer_name)
        quantized_layer_names[block_idx].append(layer_name)

    expected_quantized = adapter.expected_quantized_layer_names(model)
    if expected_quantized is not None:
        actual_quantized = {
            name for names in quantized_layer_names.values() for name in names
        }
        missing = sorted(expected_quantized - actual_quantized)
        unexpected = (
            sorted(actual_quantized - expected_quantized)
            if args.quantize_only_experts
            else []
        )
        if missing or unexpected:
            details = []
            if missing:
                details.append(
                    f"missing={len(missing)} ({', '.join(missing[:5])})"
                )
            if unexpected:
                details.append(
                    f"unexpected={len(unexpected)} ({', '.join(unexpected[:5])})"
                )
            raise ValueError(
                "Incomplete routed-expert GPTQ output: " + "; ".join(details)
            )
    safetensors_index = {}
    # Prepare directory to save packed weights
    os.makedirs(args.packed_model_path, exist_ok=True)

    loaded_shards = set()
    param_buffer = {}
    loaded = loading_utils.ensure_model_params_loaded(
        weight_dir,
        param_buffer,
        [adapter.embedding_weight_key()],
        weight_map,
        adapter,
        loaded_shards,
    )
    if loaded:
        print(f"Loaded embedding parameter from shards: {loaded}")

    # Save embeddings
    embedding_state_dict = loading_utils.materialize_model_state_dict(
        param_buffer,
        [adapter.embedding_weight_key()],
        weight_map,
        adapter,
        dtype,
        expected_shapes={
            adapter.embedding_weight_key(): tuple(
                adapter.get_embedding_module(model).weight.shape
            ),
        },
    )
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
        logical_block_keys = adapter.logical_block_keys(block, block_idx)
        quantized_weight_keys = {
            f"{layer_name}.weight"
            for layer_name in quantized_layer_names[block_idx]
        }
        source_model_keys = logical_block_keys - quantized_weight_keys

        loading_utils.ensure_model_params_loaded(
            weight_dir,
            param_buffer,
            source_model_keys,
            weight_map,
            adapter,
            loaded_shards,
        )
        expected_shapes = {
            f"{prefix}{key}": tuple(tensor.shape)
            for key, tensor in block.state_dict().items()
            if f"{prefix}{key}" in source_model_keys
        }
        block_state_dict = loading_utils.materialize_model_state_dict(
            param_buffer,
            source_model_keys,
            weight_map,
            adapter,
            dtype,
            expected_shapes=expected_shapes,
        )

        for layer_name in quantized_layer_names[block_idx]:
            weight_state_dict = torch.load(
                os.path.join(args.quantized_model_path, layer_name, "quantized_weight.pt"),
                weights_only=True,
                map_location="cpu"
            )
            packed_weight_state_dict = pack_weight(weight_state_dict, args.bits, args.sym, args.group_size)
            block_state_dict.pop(f"{layer_name}.weight", None)
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

        # Shard loading may have populated additional tensors sharing this block's
        # file (notably compact source MXFP4 experts). Evict the entire prefix.
        for key in list(param_buffer):
            if key.startswith(prefix):
                param_buffer.pop(key, None)

        del block_state_dict
        gc.collect()

    final_tensor_keys = adapter.get_final_tensor_keys()
    loading_utils.ensure_model_params_loaded(
        weight_dir,
        param_buffer,
        final_tensor_keys,
        weight_map,
        adapter,
        loaded_shards,
    )
    final_state_dict = loading_utils.materialize_model_state_dict(
        param_buffer,
        final_tensor_keys,
        weight_map,
        adapter,
        dtype,
    )

    # Save final tensors
    current_output_shard_id += 1
    current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
    save_file(
        final_state_dict,
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
    quantization_config = prepare_quantization_config(args, adapter)
    adapter.set_quantization_config(config, quantization_config)
    # Save configs
    config.save_pretrained(args.packed_model_path)
    model.generation_config.save_pretrained(args.packed_model_path)
    # Save tokenizer
    tokenizer.save_pretrained(args.packed_model_path)
    # Copy adapter-specific remote-code and multimodal companion assets.
    adapter.copy_artifacts(args.model_name_or_path, args.packed_model_path)


if __name__ == "__main__":
    main()