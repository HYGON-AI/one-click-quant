"""
Convert Qwen3.8-Flash-Next BF16 fused expert weights to INT4 (slimquant_w4a8).

Input experts (HF fused MoE, no ``.weight`` suffix):
  model.language_model.layers.*.mlp.experts.gate_up_proj  # (E, 2*I, H) bf16
  model.language_model.layers.*.mlp.experts.down_proj     # (E, H, I) bf16

Output (same names + scale companion):
  ...gate_up_proj / ...down_proj              # (E, N, K//2) int8 nibble-packed
  ...gate_up_proj.weight_scale                # (E, N, 1) float32
  ...down_proj.weight_scale                   # (E, N, 1) float32

INT4 convention (matching slimquant_w4a8 / Kimi offline scripts):
  - per-channel symmetric along N (output) dim
  - scale = abs_max / 7.0 / 16.0
  - pack: (even << 4) | odd along K

MTP experts (``mtp.*``) are left BF16 by default.

Usage:
  python qwen38_flash_bf16_experts_to_int4.py
  python qwen38_flash_bf16_experts_to_int4.py --dry-run
  python qwen38_flash_bf16_experts_to_int4.py \\
      --input-path /model/Qwen/Qwen3.8-Flash-Next \\
      --output-path /module/Qwen3.8-Flash-Next-INT4
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentParser
from glob import glob
from multiprocessing import Manager
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

DEFAULT_INPUT_PATH = "/model/Qwen/Qwen3.8-Flash-Next"
DEFAULT_OUTPUT_PATH = "/module/Qwen3.8-Flash-Next-INT4"

# Language-model fused experts only (exclude mtp.* unless --include-mtp).
EXPERT_BF16_RE = re.compile(
    r"(?:^|.*\.)language_model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
MTP_EXPERT_BF16_RE = re.compile(
    r"^mtp\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"
)


def is_expert_bf16_target(name: str, include_mtp: bool) -> bool:
    if EXPERT_BF16_RE.match(name) is not None:
        return True
    if include_mtp and MTP_EXPERT_BF16_RE.match(name) is not None:
        return True
    return False


def expert_scale_name(weight_name: str) -> str:
    return f"{weight_name}.weight_scale"


# ============================================================
# INT4 quantization (matching slimquant_w4a8 convention)
# ============================================================

def weight_quant_int4_per_channel(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel symmetric INT4 with K-nibble packing.

    Accepts (N, K) or (E, N, K). Scale is along the N (channel) dim.
    """
    if tensor.dim() == 2:
        n, k = tensor.shape
        if k % 2 != 0:
            raise ValueError(f"INT4 packing requires even K, got {k}")
        abs_max = torch.abs(tensor).amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = (abs_max / 7.0) / 16.0
        quantized = torch.round(tensor / (scale * 16.0)).clamp(-8, 7).to(torch.int8)
        unsigned = quantized.to(torch.uint8)
        even = unsigned[..., ::2]
        odd = unsigned[..., 1::2]
        packed = ((even << 4) | (odd & 0x0F)).to(torch.int8)
        return packed.contiguous(), scale.to(torch.float32)

    if tensor.dim() == 3:
        e, n, k = tensor.shape
        if k % 2 != 0:
            raise ValueError(f"INT4 packing requires even K, got {k}")
        abs_max = torch.abs(tensor).amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (abs_max / 7.0) / 16.0  # (E, N, 1)
        quantized = torch.round(tensor / (scale * 16.0)).clamp(-8, 7).to(torch.int8)
        unsigned = quantized.to(torch.uint8)
        even = unsigned[..., ::2]
        odd = unsigned[..., 1::2]
        packed = ((even << 4) | (odd & 0x0F)).to(torch.int8)
        return packed.contiguous(), scale.to(torch.float32)

    raise ValueError(f"expected 2D or 3D weight, got shape {tuple(tensor.shape)}")


# ============================================================
# Asset copying
# ============================================================

def copy_model_assets(src_dir: Path, dst_dir: Path) -> None:
    allowed_suffixes = {".json", ".py", ".md", ".jinja", ".txt"}
    allowed_names = {"__init__.py", ".gitattributes"}
    for entry in src_dir.iterdir():
        if entry.is_dir() or entry.suffix == ".safetensors":
            continue
        if entry.suffix in allowed_suffixes or entry.name in allowed_names:
            shutil.copy2(entry, dst_dir / entry.name)


# ============================================================
# Shard worker
# ============================================================

def worker_process_shard(
    rank: int,
    world_size: int,
    safetensor_files: list,
    output_dir: str,
    shared_weight_map,
    shared_stats,
    dry_run: bool,
    include_mtp: bool,
):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    local_files = safetensor_files[rank::world_size]
    local_stats: dict[str, int] = {}

    for safetensor_file in tqdm(local_files, position=rank, desc=f"GPU {rank}"):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device="cpu")
        new_state_dict = {} if not dry_run else None

        if dry_run:
            for weight_name, weight in state_dict.items():
                if is_expert_bf16_target(weight_name, include_mtp):
                    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
                        local_stats["expert_bf16"] = local_stats.get("expert_bf16", 0) + 1
                    else:
                        local_stats["expert_unexpected_dtype"] = (
                            local_stats.get("expert_unexpected_dtype", 0) + 1
                        )
                else:
                    local_stats["kept_other"] = local_stats.get("kept_other", 0) + 1
            continue

        for weight_name, weight in state_dict.items():
            if is_expert_bf16_target(weight_name, include_mtp):
                if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise TypeError(
                        f"Expert {weight_name} has dtype {weight.dtype}, "
                        "expected floating BF16/FP16/FP32."
                    )
                if weight.dim() not in (2, 3):
                    raise ValueError(
                        f"Expert {weight_name} has shape {tuple(weight.shape)}, "
                        "expected 2D or fused 3D (E, N, K)."
                    )
                weight_gpu = weight.to(device=device, dtype=torch.float32)
                packed, scale = weight_quant_int4_per_channel(weight_gpu)
                scale_name = expert_scale_name(weight_name)
                new_state_dict[weight_name] = packed.cpu()
                new_state_dict[scale_name] = scale.cpu()
                shared_weight_map[weight_name] = file_name
                shared_weight_map[scale_name] = file_name
                local_stats["quantized_int4"] = local_stats.get("quantized_int4", 0) + 1
                del weight_gpu, packed, scale
                continue

            new_state_dict[weight_name] = weight
            shared_weight_map[weight_name] = file_name
            local_stats["kept_other"] = local_stats.get("kept_other", 0) + 1

        save_file(new_state_dict, os.path.join(output_dir, file_name))
        del state_dict, new_state_dict
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    shared_stats[rank] = local_stats


def merge_stats(stats_by_rank: dict) -> dict:
    merged: dict[str, int] = {}
    for stats in stats_by_rank.values():
        for key, value in stats.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def build_output_config(config: dict) -> dict:
    config.pop("quantization_config", None)
    if "text_config" in config and isinstance(config["text_config"], dict):
        config["text_config"].pop("quantization_config", None)
        config["text_config"].pop("compression_config", None)

    compression = {
        "quant_method": "slimquant_w4a8",
        "activation_scheme": "dynamic",
        "mixed_precision": {
            "experts": {
                "weight_bits": 4,
                "format": "pack-int4-nibble",
                "scope": "per_channel",
                "modules": [
                    "model.language_model.layers.*.mlp.experts.gate_up_proj",
                    "model.language_model.layers.*.mlp.experts.down_proj",
                ],
            }
        },
    }
    config["compression_config"] = compression
    if "text_config" in config and isinstance(config["text_config"], dict):
        config["text_config"]["compression_config"] = compression
    return config


def main(
    input_path: str,
    output_path: str | None,
    dry_run: bool,
    include_mtp: bool,
) -> None:
    src_dir = Path(input_path)
    if not src_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {src_dir}")

    config_path = src_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Can not find config.json under {src_dir}")

    safetensor_files = sorted(glob(str(src_dir / "model-*.safetensors")))
    if not safetensor_files:
        safetensor_files = sorted(glob(str(src_dir / "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"Can not find any *.safetensors under {src_dir}")

    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found")

    if dry_run:
        print("=" * 60)
        print("Dry run: Qwen3.8-Flash-Next BF16 experts → INT4")
        print(f"Input path: {src_dir}")
        print(f"Safetensor shards: {len(safetensor_files)}")
        print(f"Include MTP experts: {include_mtp}")
        print("=" * 60)

        dry_stats: dict[str, int] = {}
        for sf in tqdm(safetensor_files, desc="scan"):
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if is_expert_bf16_target(key, include_mtp):
                        dry_stats["expert_bf16"] = dry_stats.get("expert_bf16", 0) + 1
                    else:
                        dry_stats["kept_other"] = dry_stats.get("kept_other", 0) + 1

        print("\nWeight classification:")
        print(json.dumps(dry_stats, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"\nTotal entries: {sum(dry_stats.values())}")
        print("Fused experts will be quantized to per-channel INT4 (nibble-packed).")
        print("All other weights kept as-is.")
        return

    if not output_path:
        raise ValueError("--output-path is required unless --dry-run is set")

    dst_dir = Path(output_path)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_model_assets(src_dir, dst_dir)

    with open(dst_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    index_path = dst_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors.index.json under {dst_dir}")
    with open(index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)

    print("=" * 60)
    print("Phase 1: BF16 fused experts → INT4 (per-channel slimquant)")
    print(f"GPUs: {world_size}  |  Shards: {len(safetensor_files)}")
    print(f"Input:  {src_dir}")
    print(f"Output: {dst_dir}")
    print(f"Include MTP: {include_mtp}")
    print("=" * 60)

    manager = Manager()
    shared_weight_map = manager.dict()
    shared_stats = manager.dict()

    mp.spawn(
        worker_process_shard,
        args=(
            world_size,
            safetensor_files,
            str(dst_dir),
            shared_weight_map,
            shared_stats,
            False,
            include_mtp,
        ),
        nprocs=world_size,
        join=True,
    )

    stats = merge_stats(dict(shared_stats))
    weight_map = dict(shared_weight_map)

    print("\nConversion stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))

    if stats.get("quantized_int4", 0) <= 0:
        raise RuntimeError(
            "No expert weights were quantized. Check that input is BF16 "
            "Qwen3.8-Flash-Next with fused experts.gate_up_proj / down_proj."
        )

    print("\n" + "=" * 60)
    print("Phase 2: Updating index and config")
    print("=" * 60)

    total_size = sum(sf.stat().st_size for sf in dst_dir.glob("*.safetensors"))
    model_index["weight_map"] = weight_map
    model_index["metadata"] = model_index.get("metadata", {})
    model_index["metadata"]["total_size"] = total_size

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False)

    config = build_output_config(config)
    with open(dst_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)

    done_path = dst_dir / "QUANTIZATION_DONE.txt"
    with open(done_path, "w", encoding="utf-8") as f:
        f.write("qwen38_flash_bf16_experts_to_int4.py completed\n")
        f.write(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
        f.write("\n")

    print(f"Updated model index: {index_path}")
    print(f"Updated config: {dst_dir / 'config.json'}")
    print(f"Total weight_map entries: {len(weight_map)}")
    print(f"Wrote {done_path}")
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Qwen3.8-Flash-Next BF16 fused experts → slimquant INT4"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help=f"BF16 model directory (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"INT4 intermediate directory (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--include-mtp",
        action="store_true",
        help="Also quantize mtp.layers.*.mlp.experts.* (default: keep MTP BF16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze weights without writing files",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.dry_run, args.include_mtp)
