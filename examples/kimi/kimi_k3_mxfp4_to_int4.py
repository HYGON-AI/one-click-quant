"""
Convert Kimi-K3 MXFP4 expert weights to INT4 (per-channel symmetric, nibble-packed).

MXFP4 format (compressed-tensors mxfp4-pack-quantized):
  - Scale (E8M0 exponent): uint8 → scale_float = 2^(value - 127 - 2)
  - Data (FP4 E2M1): nibble-packed uint8, values in {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
  - group_size=32 along K dimension

Output INT4 format:
  - Per-channel symmetric INT4, nibble-pair packed into int8
  - scale = abs_max / 7.0 / 16.0 (matching slimquant_w4a8 convention)
  - weight: packed int8 (N, K//2), weight_scale: float32 (N, 1)

Usage:
  python kimi_k3_mxfp4_to_int4.py --input-path ./Kimi-K3 --output-path ./Kimi-K3-INT4
  python kimi_k3_mxfp4_to_int4.py --input-path ./Kimi-K3 --output-path ./Kimi-K3-INT4 --dry-run
"""

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


# ============================================================
# FP4 E2M1 lookup table
# ============================================================
# E2M1 encoding: 1 sign, 2 exponent, 1 mantissa
#   index 0-7: positive values {0, 0.5, 1, 1.5, 2, 3, 4, 6}
#   index 8-15: negative (sign bit set) {-0, -0.5, -1, -1.5, -2, -3, -4, -6}
_E2M1_FLOAT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# ============================================================
# MXFP4 dequantization
# ============================================================

def unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack nibble-paired uint8 into FP32 values using E2M1 lookup.

    Each uint8 byte contains two 4-bit nibbles:
      - low nibble (bits 0-3): first FP4 value
      - high nibble (bits 4-7): second FP4 value

    Args:
        packed: (N, K//2) uint8 tensor

    Returns:
        (N, K) float32 tensor
    """
    n, k_half = packed.shape
    k = k_half * 2

    packed_flat = packed.flatten().to(torch.int32)
    low = packed_flat & 0x0F
    high = (packed_flat >> 4) & 0x0F
    indices = torch.stack([low, high], dim=1).flatten()

    table = _E2M1_FLOAT.to(device=packed.device)
    return table[indices].reshape(n, k).to(torch.float32)


def decode_mxfp_scale(scale_uint8: torch.Tensor) -> torch.Tensor:
    """
    Decode E8M0 exponent scale to float.

    An E8M0 byte directly encodes the exponent of the dequantization scale:
    scale_float = 2^(value - 127).

    The FP4 E2M1 range adjustment is applied when the source MXFP4 scale is
    generated. It must not be applied again while decoding the stored scale.

    Args:
        scale_uint8: (N, K//32) uint8 tensor

    Returns:
        (N, K//32) float32 tensor
    """
    scale_exp = scale_uint8.to(torch.int32) - 127
    return (2.0 ** scale_exp.float()).to(torch.float32)


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MXFP4 weight to float32.

    Args:
        packed: (N, K//2) uint8  — nibble-packed FP4 E2M1 data
        scale:  (N, K//32) uint8 — E8M0 exponent per group of 32

    Returns:
        (N, K) float32 tensor
    """
    n, k_half = packed.shape
    k = k_half * 2

    values = unpack_fp4_e2m1(packed)           # (N, K) float32
    scale_f = decode_mxfp_scale(scale)          # (N, K//32) float32

    # Expand scale: repeat each scale value 32 times
    scale_expanded = scale_f.repeat_interleave(32, dim=1)  # (N, K) float32

    return values * scale_expanded


# ============================================================
# INT4 quantization (matching slimquant_w4a8 convention)
# ============================================================

def weight_quant_int4_per_channel(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel symmetric INT4 quantization with nibble-pair packing.

    Quantizes each row to signed INT4 [-8, 7], then converts to unsigned
    [0, 15] and packs consecutive values as nibble pairs into int8 bytes:
    (val_even << 4) | val_odd.

    scale = abs_max / 7.0 / 16.0
    The /16 compensates for INT4 → INT8 high 4-bit × 16 in the kernel.

    Args:
        tensor: (N, K) float32 tensor

    Returns:
        packed: (N, K//2) int8 tensor
        scale:  (N, 1) float32 tensor
    """
    n, k = tensor.shape
    if k % 2 != 0:
        raise ValueError(f"INT4 packing requires even K dimension, got {k}")

    qmax = 7.0
    abs_max = (
        torch.abs(tensor).max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    )  # (N, 1)

    scale = (abs_max / qmax) / 16.0  # /16 compensates INT4→INT8 high 4-bit ×16

    quantized = torch.round(tensor / (scale * 16.0))
    quantized = torch.clamp(quantized, -8, 7).to(torch.int8)

    # Convert to unsigned for nibble packing (ZP=0 convention)
    unsigned = quantized.to(torch.uint8)

    # Pack nibble pairs: (even << 4) | odd
    even = unsigned[..., ::2]
    odd = unsigned[..., 1::2]
    packed = ((even << 4) | (odd & 0x0F)).to(torch.int8)

    return packed.contiguous(), scale.to(torch.float32)


# ============================================================
# Weight classification
# ============================================================

EXPERT_PACKED_RE = re.compile(
    r".*block_sparse_moe\.experts\.\d+\.(w1|w2|w3)\.weight_packed$"
)
EXPERT_SCALE_RE = re.compile(
    r".*block_sparse_moe\.experts\.\d+\.(w1|w2|w3)\.weight_scale$"
)


def is_expert_packed(name: str) -> bool:
    return EXPERT_PACKED_RE.match(name) is not None


def is_expert_scale(name: str) -> bool:
    return EXPERT_SCALE_RE.match(name) is not None


def get_expert_base_name(packed_name: str) -> str:
    """Convert '...experts.N.w1.weight_packed' → '...experts.N.w1.weight'"""
    return packed_name.replace("weight_packed", "weight")


def get_expert_scale_name(packed_name: str) -> str:
    """Convert '...experts.N.w1.weight_packed' → '...experts.N.w1.weight_scale'"""
    return packed_name.replace("weight_packed", "weight_scale")


# ============================================================
# Asset copying
# ============================================================

def copy_model_assets(src_dir: Path, dst_dir: Path):
    """Copy all non-safetensor assets to the output directory."""
    allowed_suffixes = {".json", ".py", ".md", ".jinja", ".txt"}
    allowed_names = {"__init__.py", ".gitattributes"}
    for entry in src_dir.iterdir():
        if entry.is_dir():
            continue
        if entry.suffix == ".safetensors":
            continue
        if entry.suffix in allowed_suffixes or entry.name in allowed_names:
            shutil.copy2(entry, dst_dir / entry.name)


# ============================================================
# Phase 1: Parallel shard processing
# ============================================================

def worker_process_shard(
    rank: int,
    world_size: int,
    safetensor_files: list,
    output_dir: str,
    shared_weight_map,
    shared_stats,
    dry_run: bool,
):
    """Process a subset of safetensor shards on one GPU."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    local_files = safetensor_files[rank::world_size]
    local_stats = {}

    for safetensor_file in tqdm(local_files, position=rank, desc=f"GPU {rank}"):
        file_name = os.path.basename(safetensor_file)
        # Keep the complete shard in host memory.  A shard also contains weights
        # that do not need conversion, and loading all of them on the GPU makes
        # the transient FP32 MXFP4 decode peak unnecessarily large.
        state_dict = load_file(safetensor_file, device="cpu")
        new_state_dict = {} if not dry_run else None

        # First pass: collect expert packed/scale pairs
        expert_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        keys_to_remove = set()

        for weight_name, weight in state_dict.items():
            if is_expert_packed(weight_name):
                scale_name = weight_name.replace("weight_packed", "weight_scale")
                if scale_name in state_dict:
                    expert_pairs[weight_name] = (weight, state_dict[scale_name])
                    keys_to_remove.add(weight_name)
                    keys_to_remove.add(scale_name)

        if dry_run:
            for weight_name, weight in state_dict.items():
                if is_expert_packed(weight_name):
                    local_stats["mxfp4_expert"] = local_stats.get("mxfp4_expert", 0) + 1
                elif is_expert_scale(weight_name):
                    pass  # counted with packed
                else:
                    cat = "kept_bf16" if weight.dtype in (torch.bfloat16, torch.float16) else "kept_other"
                    local_stats[cat] = local_stats.get(cat, 0) + 1
            continue

        # Process expert weights: MXFP4 → INT4
        for packed_name, (packed, scale_uint8) in expert_pairs.items():
            # Move only the current compressed expert pair to the GPU.  Do not
            # retain GPU tensors after this iteration: loop variables otherwise
            # keep the last large expert alive until the next shard is loaded.
            packed_gpu = packed.to(device)
            scale_uint8_gpu = scale_uint8.to(device)

            # Dequantize MXFP4 → float32
            weight_fp32 = dequantize_mxfp4(packed_gpu, scale_uint8_gpu)

            # Quantize to INT4
            int4_packed, int4_scale = weight_quant_int4_per_channel(weight_fp32)

            # Save with new names
            base_name = get_expert_base_name(packed_name)
            scale_name = get_expert_scale_name(packed_name)

            new_state_dict[base_name] = int4_packed.cpu()
            new_state_dict[scale_name] = int4_scale.cpu()

            shared_weight_map[base_name] = file_name
            shared_weight_map[scale_name] = file_name

            local_stats["quantized_int4"] = local_stats.get("quantized_int4", 0) + 1

            del packed_gpu, scale_uint8_gpu, weight_fp32, int4_packed, int4_scale

        # Copy all non-expert weights as-is
        for weight_name, weight in state_dict.items():
            if weight_name in keys_to_remove:
                continue
            new_state_dict[weight_name] = weight
            shared_weight_map[weight_name] = file_name
            local_stats["kept_bf16"] = local_stats.get("kept_bf16", 0) + 1

        if not dry_run:
            save_file(new_state_dict, os.path.join(output_dir, file_name))

            # Free GPU memory
            del state_dict, new_state_dict, expert_pairs
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    shared_stats[rank] = local_stats


# ============================================================
# Config update
# ============================================================

def build_output_config(config: dict) -> dict:
    """Remove MXFP4 quantization config, add INT4 config."""
    config.pop("quantization_config", None)
    config.pop("compression_config", None)
    # Also clean up quantization_config nested inside text_config
    if "text_config" in config:
        config["text_config"].pop("quantization_config", None)
    config["compression_config"] = {
        "quant_method": "slimquant_w4a8",
    }
    return config


def merge_stats(stats_by_rank: dict) -> dict:
    merged = {}
    for stats in stats_by_rank.values():
        for key, value in stats.items():
            merged[key] = merged.get(key, 0) + value
    return merged


# ============================================================
# Main
# ============================================================

def main(input_path: str, output_path: str | None, dry_run: bool):
    src_dir = Path(input_path)
    if not src_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {src_dir}")

    config_path = src_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Can not find config.json under {src_dir}")

    safetensor_files = sorted(glob(str(src_dir / "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"Can not find any *.safetensors under {src_dir}")

    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found")

    # --- Dry run ---
    if dry_run:
        print("=" * 60)
        print("Dry run: analyzing weights")
        print(f"Input path: {src_dir}")
        print(f"Safetensor shards: {len(safetensor_files)}")
        print("=" * 60)

        device = "cuda:0"
        torch.cuda.set_device(device)
        dry_stats = {}
        for sf in tqdm(safetensor_files):
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if is_expert_packed(key):
                        dry_stats["mxfp4_expert_packed"] = dry_stats.get("mxfp4_expert_packed", 0) + 1
                    elif is_expert_scale(key):
                        dry_stats["mxfp4_expert_scale"] = dry_stats.get("mxfp4_expert_scale", 0) + 1
                    else:
                        dry_stats["kept_original"] = dry_stats.get("kept_original", 0) + 1

        print("\nWeight classification:")
        print(json.dumps(dry_stats, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"\nTotal entries: {sum(dry_stats.values())}")
        print("All expert packed + scale will be dequantized (MXFP4→BF16) "
              "and requantized to INT4.")
        print("All other weights will be kept as-is.")
        return

    # --- Production run ---
    if not output_path:
        raise ValueError("--output-path is required unless --dry-run is set")

    dst_dir = Path(output_path)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_model_assets(src_dir, dst_dir)

    # Load config and index
    with open(dst_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    index_path = dst_dir / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)

    # ================================================================
    # Phase 1: Parallel shard processing (MXFP4 → INT4)
    # ================================================================
    print("=" * 60)
    print("Phase 1: MXFP4 → INT4 conversion (per-channel symmetric)")
    print(f"GPUs: {world_size}  |  Shards: {len(safetensor_files)}")
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
            dry_run,
        ),
        nprocs=world_size,
        join=True,
    )

    stats = merge_stats(dict(shared_stats))
    weight_map = dict(shared_weight_map)

    print("\nConversion stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))

    # ================================================================
    # Phase 2: Update index and config
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Updating index and config")
    print("=" * 60)

    model_index["weight_map"] = weight_map
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False)

    config = build_output_config(config)
    with open(dst_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Updated model index: {index_path}")
    print(f"Updated config: {dst_dir / 'config.json'}")
    print(f"Total weight_map entries: {len(weight_map)}")
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert Kimi-K3 MXFP4 expert weights to INT4"
    )
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to the MXFP4 Kimi-K3 model directory")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output directory for INT4 model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze weights without writing files")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.dry_run)
