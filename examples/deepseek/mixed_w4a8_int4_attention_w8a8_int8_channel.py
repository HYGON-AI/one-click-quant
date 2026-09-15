# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import re
import shutil
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm


FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)
FP4_BLOCK_SIZE = 32
FP8_BLOCK_SIZE = 128


def scale_name_for(weight_name: str) -> str:
    return ".".join(weight_name.split(".")[:-1] + ["scale"])


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


def pack_signed_int4(q: torch.Tensor) -> torch.Tensor:
    assert q.dtype == torch.int8 and q.ndim == 2
    assert q.shape[1] % 2 == 0, "int4 packing requires an even K dimension"
    u = q.to(torch.uint8) & 0x0F
    high = u[:, 0::2]
    low = u[:, 1::2]
    return ((high << 4) | low).contiguous().to(torch.int8)


def quantize_int4_channelwise(
    tensor: torch.Tensor,
    scale_divisor: float = 16.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert tensor.ndim == 2
    qmax = 7.0
    abs_max = tensor.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    math_scale = abs_max / qmax
    q = torch.round(tensor.float() / math_scale).clamp(-8, 7).to(torch.int8)
    packed = pack_signed_int4(q)
    stored_scale = (math_scale / scale_divisor).to(torch.float32)
    return packed, stored_scale


def is_routed_expert_weight(name: str, tensor: torch.Tensor) -> bool:
    return tensor.dtype == torch.int8 and ".ffn.experts." in name and name.endswith(".weight")


def is_attention_weight(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or tensor.dtype != torch.float8_e4m3fn:
        return False
    return ".attn." in name


def is_shared_expert_weight(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or tensor.dtype != torch.float8_e4m3fn:
        return False
    return ".ffn.shared_experts." in name


def is_mtp_main_proj_weight(name: str, tensor: torch.Tensor) -> bool:
    return name == "mtp.0.main_proj.weight" and tensor.dtype == torch.float8_e4m3fn


def is_wo_a_weight(name: str) -> bool:
    return name.endswith("wo_a.weight")


def has_scale(name: str, state_dict: dict[str, torch.Tensor]) -> bool:
    return scale_name_for(name) in state_dict


def convert_one_file(
    input_path: str,
    output_path: str,
    scale_divisor: float,
    dequant_wo_a: bool,
    stats: dict[str, int],
) -> None:
    state_dict = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name)

    new_state_dict = {}
    for name, tensor in state_dict.items():
        if name.endswith(".scale"):
            weight_name = scale_name_for(name).removesuffix(".scale") + ".weight"
            weight = state_dict.get(weight_name)
            if weight is not None and (is_routed_expert_weight(weight_name, weight) or is_attention_weight(weight_name, weight) or is_shared_expert_weight(weight_name, weight) or is_mtp_main_proj_weight(weight_name, weight)):
                continue
            new_state_dict[name] = tensor
            stats["kept"] += 1
            continue

        scale_name = scale_name_for(name)
        if is_routed_expert_weight(name, tensor):
            scale = state_dict[scale_name]
            dequant = dequant_fp4_to_float(tensor, scale)
            q_weight, q_scale = quantize_int4_channelwise(dequant, scale_divisor=scale_divisor)
            new_state_dict[name] = q_weight
            new_state_dict[scale_name] = q_scale
            stats["routed_expert_int4"] += 1
        elif (is_attention_weight(name, tensor) or is_shared_expert_weight(name, tensor) or is_mtp_main_proj_weight(name, tensor)) and has_scale(name, state_dict):
            scale = state_dict[scale_name]
            if is_attention_weight(name, tensor) and dequant_wo_a and is_wo_a_weight(name):
                new_state_dict[name] = dequant_fp8_blockwise(tensor, scale).bfloat16()
                stats["attention_wo_a_bf16"] += 1
            else:
                dequant = dequant_fp8_blockwise(tensor, scale)
                q_weight, q_scale = quantize_int8_channelwise(dequant)
                new_state_dict[name] = q_weight
                new_state_dict[scale_name] = q_scale
                if is_shared_expert_weight(name, tensor):
                    stats["shared_expert_int8"] += 1
                elif is_mtp_main_proj_weight(name, tensor):
                    stats["mtp_main_proj_int8"] += 1
                else:
                    stats["attention_int8"] += 1
        else:
            new_state_dict[name] = tensor
            stats["kept"] += 1

    save_file(new_state_dict, output_path)


def convert_model(
    input_dir: str,
    output_dir: str,
    scale_divisor: float,
    dequant_wo_a: bool,
    limit_files: int | None,
) -> dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    if limit_files is not None:
        files = files[:limit_files]
    stats = {"routed_expert_int4": 0, "attention_int8": 0, "attention_wo_a_bf16": 0, "shared_expert_int8": 0, "mtp_main_proj_int8": 0, "kept": 0}
    for path in tqdm(files, desc="Converting"):
        fname = os.path.basename(path)
        convert_one_file(path, os.path.join(output_dir, fname), scale_divisor, dequant_wo_a, stats)
    return stats


def copy_metadata(input_dir: str, output_dir: str, dequant_wo_a: bool, limit_files: int | None, scale_divisor: float) -> None:
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "configuration.json",
        "model.safetensors.index.json",
        "README.md",
        "LICENSE",
    ]:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(output_dir, fname))

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            model_index = json.load(f)
        if limit_files is not None:
            kept_files = {os.path.basename(p) for p in sorted(glob(os.path.join(input_dir, "*.safetensors")))[:limit_files]}
            model_index["weight_map"] = {k: v for k, v in model_index["weight_map"].items() if v in kept_files}
        if dequant_wo_a:
            model_index["weight_map"] = {
                name: fname
                for name, fname in model_index["weight_map"].items()
                if not (name.endswith("wo_a.scale") and ".attn." in name)
            }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)

    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config.pop("quantization_config", None)
        config["quantization_config"] = {
            "activation_scheme": "dynamic",
            "quant_method": "slimquant_w4a8",
            "mixed_quantization": {
                "attention": "w8a8-int8-channel except wo_a bf16 by default",
                "shared_experts": "w8a8-int8-channel",
                "mtp_main_proj": "w8a8-int8-channel",
                "routed_experts": "w4a8-int4-channel",
                "int4_scale_divisor": scale_divisor,
            },
        }
        config["expert_dtype"] = "int4"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_summary(output_dir: str, stats: dict[str, int], args) -> None:
    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "routed_experts": "w4a8-int4-channel",
        "attention": "w8a8-int8-channel except wo_a bf16 by default",
        "shared_experts": "w8a8-int8-channel",
        "mtp_main_proj": "w8a8-int8-channel",
        "scale_divisor": args.scale_divisor,
        "dequant_wo_a": not args.quantize_wo_a,
        "limit_files": args.limit_files,
        "stats": stats,
    }
    with open(os.path.join(output_dir, "mixed_w4a8_int4_attention_w8a8_int8_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = ArgumentParser(
        description="Convert DSpark checkpoint: routed experts to w4a8 int4 channel, attention to w8a8 int8 channel."
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--scale-divisor", type=float, default=16.0, help="Stored int4 scale divisor used by slimquant kernels.")
    parser.add_argument("--quantize-wo-a", action="store_true", help="Quantize attention wo_a to int8-channel instead of keeping bf16.")
    parser.add_argument("--limit-files", type=int, default=None, help="Convert only the first N shard files for smoke testing.")
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must be different")
    torch.set_num_threads(args.num_threads)
    stats = convert_model(args.input_dir, args.output_dir, args.scale_divisor, not args.quantize_wo_a, args.limit_files)
    copy_metadata(args.input_dir, args.output_dir, not args.quantize_wo_a, args.limit_files, args.scale_divisor)
    write_summary(args.output_dir, stats, args)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print(f"Done. Mixed quantized model saved to {args.output_dir}")


if __name__ == "__main__":
    main()