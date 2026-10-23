"""
Qwen3.8-Flash-Next-INT4 → INT4(experts) + INT8(W8 include layers).

Input: checkpoint from qwen38_flash_bf16_experts_to_int4.py
  - fused experts: packed INT4 + ``.weight_scale`` (kept as-is)
  - other Linear: BF16 ``.weight``

W8 include (aligned with Kimi-K3 w4a8-w8a8-slimquant intent):
  - self_attn.{q,k,v}_proj
  - mlp.shared_expert.{gate,up,down}_proj
  - linear_attn.{in_proj_qkv,in_proj_z}

Kept BF16 (Flash-Next / Kimi-style ignore):
  - o_proj / linear_attn.out_proj
  - mlp.gate / shared_expert_gate
  - hyper_connection*, indexer*, ple*, visual*, embed, lm_head, mtp*

Usage:
  python qwen38_flash_int4_to_int4_8.py
  python qwen38_flash_int4_to_int4_8.py --dry-run
  python qwen38_flash_int4_to_int4_8.py \\
      --input-path /module/Qwen3.8-Flash-Next-INT4 \\
      --output-path /module/Qwen3.8-Flash-Next-w4a8-slimquant
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

DEFAULT_INPUT_PATH = "/module/Qwen3.8-Flash-Next-INT4"
DEFAULT_OUTPUT_PATH = "/module/Qwen3.8-Flash-Next-w4a8-slimquant"

W8_MODULES = [
    "model.language_model.layers.*.self_attn.q_proj",
    "model.language_model.layers.*.self_attn.k_proj",
    "model.language_model.layers.*.self_attn.v_proj",
    "model.language_model.layers.*.mlp.shared_expert.gate_proj",
    "model.language_model.layers.*.mlp.shared_expert.up_proj",
    "model.language_model.layers.*.mlp.shared_expert.down_proj",
    "model.language_model.layers.*.linear_attn.in_proj_qkv",
    "model.language_model.layers.*.linear_attn.in_proj_z",
]

W4_MODULES = [
    "model.language_model.layers.*.mlp.experts.gate_up_proj",
    "model.language_model.layers.*.mlp.experts.down_proj",
]

IGNORE_PATTERNS = [
    r"re:.*self_attn\.o_proj(?:\.|$)",
    r"re:.*linear_attn\.out_proj(?:\.|$)",
    r"re:.*linear_attn\.in_proj_a(?:\.|$)",
    r"re:.*linear_attn\.in_proj_b(?:\.|$)",
    r"re:.*linear_attn\.conv1d(?:\.|$)",
    r"re:.*mlp\.gate(?:\.|$)",
    r"re:.*shared_expert_gate(?:\.|$)",
    r"re:.*hyper_connection.*",
    r"re:.*indexer.*",
    r"re:.*\.ple\..*",
    r"re:.*visual.*",
    r"re:.*lm_head(?:\.|$)",
    r"re:.*embed_tokens(?:\.|$)",
    r"re:^mtp\..*",
]

W8_WEIGHT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"model\.language_model\.layers\.\d+\.self_attn\."
        r"(q_proj|k_proj|v_proj)\.weight$"
    ),
    re.compile(
        r"model\.language_model\.layers\.\d+\.mlp\.shared_expert\."
        r"(gate_proj|up_proj|down_proj)\.weight$"
    ),
    re.compile(
        r"model\.language_model\.layers\.\d+\.linear_attn\."
        r"(in_proj_qkv|in_proj_z)\.weight$"
    ),
]

EXPERT_INT4_WEIGHT_RE = re.compile(
    r"(?:^|.*\.)language_model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
EXPERT_INT4_SCALE_RE = re.compile(
    r"(?:^|.*\.)language_model\.layers\.\d+\.mlp\.experts\."
    r"(gate_up_proj|down_proj)\.weight_scale$"
)


def is_w8_target_weight(name: str) -> bool:
    return any(p.match(name) is not None for p in W8_WEIGHT_PATTERNS)


def is_expert_int4_weight(name: str) -> bool:
    return EXPERT_INT4_WEIGHT_RE.match(name) is not None


def is_expert_int4_scale(name: str) -> bool:
    return EXPERT_INT4_SCALE_RE.match(name) is not None


def w8_scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"not a weight tensor: {weight_name}")
    return weight_name[: -len(".weight")] + ".weight_scale"


def weight_quant_int8_per_channel(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric INT8: scale = absmax / 127."""
    assert tensor.dim() == 2, f"expected 2D weight, got {tuple(tensor.shape)}"
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    scale = abs_max / qmax
    quantized = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.float32)


def copy_model_assets(src_dir: Path, dst_dir: Path) -> None:
    allowed_suffixes = {".json", ".py", ".md", ".jinja", ".txt"}
    allowed_names = {"__init__.py", ".gitattributes"}
    for entry in src_dir.iterdir():
        if entry.is_dir() or entry.suffix == ".safetensors":
            continue
        if entry.suffix in allowed_suffixes or entry.name in allowed_names:
            shutil.copy2(entry, dst_dir / entry.name)


def worker_process_shard(
    rank: int,
    world_size: int,
    safetensor_files: list,
    output_dir: str,
    shared_weight_map,
    shared_stats,
    dry_run: bool,
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
                if is_w8_target_weight(weight_name):
                    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
                        local_stats["w8_bf16_candidates"] = (
                            local_stats.get("w8_bf16_candidates", 0) + 1
                        )
                    else:
                        local_stats["w8_unexpected_dtype"] = (
                            local_stats.get("w8_unexpected_dtype", 0) + 1
                        )
                elif is_expert_int4_weight(weight_name):
                    local_stats["kept_expert_int4_weight"] = (
                        local_stats.get("kept_expert_int4_weight", 0) + 1
                    )
                elif is_expert_int4_scale(weight_name):
                    local_stats["kept_expert_int4_scale"] = (
                        local_stats.get("kept_expert_int4_scale", 0) + 1
                    )
                else:
                    local_stats["kept_other"] = local_stats.get("kept_other", 0) + 1
            continue

        for weight_name, weight in state_dict.items():
            if weight_name.endswith(".weight_scale") and is_w8_target_weight(
                weight_name.replace(".weight_scale", ".weight")
            ):
                local_stats["dropped_stale_w8_scale"] = (
                    local_stats.get("dropped_stale_w8_scale", 0) + 1
                )
                continue

            if is_w8_target_weight(weight_name):
                if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise TypeError(
                        f"W8 target {weight_name} has dtype {weight.dtype}, "
                        "expected floating BF16/FP16/FP32."
                    )
                weight_gpu = weight.to(device=device, dtype=torch.float32)
                q_int8, scale = weight_quant_int8_per_channel(weight_gpu)
                scale_name = w8_scale_name(weight_name)
                new_state_dict[weight_name] = q_int8.cpu()
                new_state_dict[scale_name] = scale.cpu()
                shared_weight_map[weight_name] = file_name
                shared_weight_map[scale_name] = file_name
                local_stats["quantized_w8"] = local_stats.get("quantized_w8", 0) + 1
                del weight_gpu, q_int8, scale
                continue

            new_state_dict[weight_name] = weight
            shared_weight_map[weight_name] = file_name
            if is_expert_int4_weight(weight_name):
                local_stats["kept_expert_int4_weight"] = (
                    local_stats.get("kept_expert_int4_weight", 0) + 1
                )
            elif is_expert_int4_scale(weight_name):
                local_stats["kept_expert_int4_scale"] = (
                    local_stats.get("kept_expert_int4_scale", 0) + 1
                )
            else:
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

    compression = {
        "quant_method": "slimquant_w4a8",
        "activation_scheme": "dynamic",
        "ignore": list(IGNORE_PATTERNS),
        "mixed_precision": {
            "experts": {
                "weight_bits": 4,
                "format": "pack-int4-nibble",
                "scope": "per_channel",
                "modules": list(W4_MODULES),
            },
            "w8a8_include": {
                "weight_bits": 8,
                "activation_bits": 8,
                "format": "int8-per-channel",
                "scope": "per_channel",
                "activation_scheme": "dynamic_per_token",
                "modules": list(W8_MODULES),
            },
        },
    }
    config["compression_config"] = compression
    if "text_config" in config and isinstance(config["text_config"], dict):
        config["text_config"]["compression_config"] = compression
    return config


def main(input_path: str, output_path: str | None, dry_run: bool) -> None:
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
        print("Dry run: Flash-Next-INT4 → W8 include layers")
        print(f"Input path: {src_dir}")
        print(f"Safetensor shards: {len(safetensor_files)}")
        print(f"GPUs: {world_size}")
        print("=" * 60)

        dry_stats: dict[str, int] = {}
        for sf in tqdm(safetensor_files, desc="scan"):
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if is_w8_target_weight(key):
                        dry_stats["w8_targets"] = dry_stats.get("w8_targets", 0) + 1
                    elif is_expert_int4_weight(key):
                        dry_stats["expert_int4_weight"] = (
                            dry_stats.get("expert_int4_weight", 0) + 1
                        )
                    elif is_expert_int4_scale(key):
                        dry_stats["expert_int4_scale"] = (
                            dry_stats.get("expert_int4_scale", 0) + 1
                        )
                    else:
                        dry_stats["other"] = dry_stats.get("other", 0) + 1

        print("\nWeight classification:")
        print(json.dumps(dry_stats, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"\nTotal entries: {sum(dry_stats.values())}")
        print("W8 targets → per-channel INT8; expert INT4 + others kept.")
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
    print("Phase 1: W8 include BF16 → INT8 (experts INT4 kept)")
    print(f"GPUs: {world_size}  |  Shards: {len(safetensor_files)}")
    print(f"Input:  {src_dir}")
    print(f"Output: {dst_dir}")
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
        ),
        nprocs=world_size,
        join=True,
    )

    stats = merge_stats(dict(shared_stats))
    weight_map = dict(shared_weight_map)

    print("\nConversion stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))

    if stats.get("quantized_w8", 0) <= 0:
        raise RuntimeError(
            "No W8 target weights were quantized. Check that input is "
            "Qwen3.8-Flash-Next-INT4 with BF16 tensors matching W8 include."
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
        f.write("qwen38_flash_int4_to_int4_8.py completed\n")
        f.write(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
        f.write("\n")

    print(f"Updated model index: {index_path}")
    print(f"Updated config: {dst_dir / 'config.json'}")
    print(f"Total weight_map entries: {len(weight_map)}")
    print(f"Wrote {done_path}")
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Qwen3.8-Flash-Next-INT4 → keep expert INT4 + W8A8 on include layers"
        )
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help=f"INT4 intermediate directory (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Final mixed INT4+INT8 directory (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze weights without writing files",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.dry_run)
