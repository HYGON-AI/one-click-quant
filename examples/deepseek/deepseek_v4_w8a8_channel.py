# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm


FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

FP4_BLOCK_SIZE = 32
FP8_BLOCK_SIZE = 128


def scale_name_for(weight_name: str) -> str:
    return ".".join(weight_name.split(".")[:-1] + ["scale"])


def cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cast a packed fp4 e2m1fn tensor to blockwise fp8 e4m3fn.
    """
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    assert in_dim % FP8_BLOCK_SIZE == 0 and out_dim % FP8_BLOCK_SIZE == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // FP4_BLOCK_SIZE

    x = unpack_e2m1fn_to_float(x)

    # max_fp4 (6.0) * MAX_OFFSET must fit in e4m3fn (max 448).
    max_offset_bits = 6

    b_out = out_dim // FP8_BLOCK_SIZE
    b_in = in_dim // FP8_BLOCK_SIZE
    x = x.view(b_out, FP8_BLOCK_SIZE, b_in, FP8_BLOCK_SIZE).transpose(1, 2)
    scale = scale.float().view(b_out, FP8_BLOCK_SIZE, b_in, -1).transpose(1, 2).flatten(2)
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**max_offset_bits)
    offset = scale / scale_max_offset_bits
    offset = offset.unflatten(-1, (FP8_BLOCK_SIZE, -1)).repeat_interleave(FP4_BLOCK_SIZE, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)


def unpack_e2m1fn_to_float(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8
    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    return torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)


def dequant_fp4_to_float(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    values = unpack_e2m1fn_to_float(x).float()
    expanded_scale = scale.float().repeat_interleave(FP4_BLOCK_SIZE, dim=1)
    return values * expanded_scale


def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(torch.float32)
    weight = (
        weight.unflatten(0, (-1, FP8_BLOCK_SIZE))
        .unflatten(-1, (-1, FP8_BLOCK_SIZE))
        .float()
        * scale[:, None, :, None].float()
    )
    return weight.flatten(2, 3).flatten(0, 1)


def quantize_int8_channelwise(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert tensor.ndim == 2
    qmax = 127.0
    abs_max = tensor.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = abs_max / qmax
    quantized = torch.round(tensor.float() / scale).clamp(-qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def quantize_fp8_channelwise(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert tensor.ndim == 2
    qmax = torch.finfo(torch.float8_e4m3fn).max
    abs_max = tensor.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = abs_max / qmax
    quantized = torch.clamp(tensor.float() / scale, -qmax, qmax)
    return quantized.to(torch.float8_e4m3fn), scale.to(torch.float32)


def cast_fp4_to_fp8(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight, scale = cast_e2m1fn_to_e4m3fn(x, scale)
    return weight, scale.to(torch.float32)


def cast_fp4_to_int8_channel(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_int8_channelwise(dequant_fp4_to_float(x, scale))


def cast_fp8_to_fp8_channel(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_fp8_channelwise(dequant_fp8_blockwise(x, scale))


def cast_fp4_to_fp8_channel(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight, scale = cast_fp4_to_fp8(x, scale)
    return cast_fp8_to_fp8_channel(weight, scale)


def cast_fp8_to_int8_channel(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_int8_channelwise(dequant_fp8_blockwise(x, scale))


def dequant_wo_a_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return dequant_fp8_blockwise(weight, scale).bfloat16()


def is_fp4_expert(name: str, tensor: torch.Tensor) -> bool:
    return tensor.dtype == torch.int8 and "experts" in name


def is_wo_a_weight(name: str) -> bool:
    return name.endswith("wo_a.weight")


def has_scale(name: str, state_dict: dict[str, torch.Tensor]) -> bool:
    return scale_name_for(name) in state_dict


def convert_one_file(input_path: str, output_path: str, output_format: str) -> None:
    state_dict = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name)

    new_state_dict = {}
    for name, tensor in state_dict.items():
        if name.endswith(".scale"):
            continue

        scale_name = scale_name_for(name)
        if is_fp4_expert(name, tensor):
            scale = state_dict[scale_name]
            if output_format == "fp8-channel":
                weight, q_scale = cast_fp4_to_fp8_channel(tensor, scale)
            else:
                weight, q_scale = cast_fp4_to_int8_channel(tensor, scale)
            new_state_dict[name] = weight
            new_state_dict[scale_name] = q_scale

        elif is_wo_a_weight(name):
            new_state_dict[name] = dequant_wo_a_to_bf16(tensor, state_dict[scale_name])

        elif has_scale(name, state_dict) and tensor.dtype == torch.float8_e4m3fn:
            scale = state_dict[scale_name]
            if output_format == "fp8-channel":
                weight, q_scale = cast_fp8_to_fp8_channel(tensor, scale)
                new_state_dict[name] = weight
                new_state_dict[scale_name] = q_scale
            else:
                weight, q_scale = cast_fp8_to_int8_channel(tensor, scale)
                new_state_dict[name] = weight
                new_state_dict[scale_name] = q_scale

        else:
            new_state_dict[name] = tensor

    save_file(new_state_dict, output_path)


def convert_model(input_dir: str, output_dir: str, output_format: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    for path in tqdm(files, desc="Converting"):
        fname = os.path.basename(path)
        convert_one_file(path, os.path.join(output_dir, fname), output_format)


def int8_compression_config() -> dict:
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "weights": {
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "ignore": [
            "re:.*attn.wo_a.*"
        ],
        "format": "int-quantized",
        "quant_method": "compressed-tensors",
    }
    
def fp8_compression_config() -> dict:
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "float",
                },
                "weights": {
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "float",
                },
            }
        },
        "ignore": [
            "re:.*attn.wo_a.*"
        ],
        "format": "float-quantized",
        "quant_method": "compressed-tensors",
    }


def update_index(output_dir: str) -> None:
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return

    with open(index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)

    model_index["weight_map"] = {
        name: fname
        for name, fname in model_index["weight_map"].items()
        if not (name.endswith(".scale") and is_wo_a_weight(name.removesuffix(".scale") + ".weight"))
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_metadata(input_dir: str, output_dir: str, output_format: str) -> None:
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "configuration.json",
        "model.safetensors.index.json",
    ]:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(output_dir, fname))

    update_index(output_dir)

    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if output_format == "fp8-channel":
        if "expert_dtype" in config:
            config["expert_dtype"] = "fp8"
        config.pop("quantization_config", None)
        config["compression_config"] = fp8_compression_config()
    else:
        if "expert_dtype" in config:
            config["expert_dtype"] = "int8"
        config.pop("quantization_config", None)
        config["compression_config"] = int8_compression_config()

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = ArgumentParser(description="Convert deepseek-v4 safetensors checkpoint to fp8-channel or int8-channel format.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to deepseek safetensors checkpoint directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output converted checkpoint directory.")
    parser.add_argument(
        "--output-format",
        choices=["fp8-channel", "int8-channel"],
        default="fp8-channel",
        help="Target format for main scaled weights.",
    )
    parser.add_argument("--num-threads", type=int, default=8, help="Torch CPU thread count.")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    convert_model(args.input_dir, args.output_dir, args.output_format)
    copy_metadata(args.input_dir, args.output_dir, args.output_format)
    print(f"Done. Converted {args.output_format} model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
