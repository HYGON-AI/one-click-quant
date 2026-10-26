"""
Quantize Qwen3.8-Flash-Next MTP block to match the main-net slimquant scheme.

Input: a Flash-Next checkpoint whose MTP is still BF16 (typically
``Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8``, where ``re:^mtp\\..*`` is ignored).

MTP targets (aligned with main-net mixed_precision):
  - INT4 pack-int4-nibble (per-channel):
      mtp.layers.*.mlp.experts.{gate_up_proj,down_proj}
  - INT8 per-channel (+ dynamic A8 at runtime):
      mtp.layers.*.self_attn.{q,k,v}_proj
      mtp.layers.*.mlp.shared_expert.{gate,up,down}_proj

Left BF16 (same spirit as main-net ignore):
  o_proj, mlp.gate, shared_expert_gate, hyper_connection*, indexer*,
  fc_*, norms, etc.

Existing main-net INT4 experts / W8 / ngram INT8 tensors are copied as-is.
``compression_config`` is updated: drop blanket ``re:^mtp\\..*``, append MTP
modules into ``experts`` / ``w8a8_include``, add fine-grained MTP ignore.

Usage:
  python qwen38_flash_next_mtp_to_w4a8.py --dry-run
  python qwen38_flash_next_mtp_to_w4a8.py
  python qwen38_flash_next_mtp_to_w4a8.py \\
      --input-path /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8 \\
      --output-path /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp
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

DEFAULT_INPUT_CANDIDATES = [
    "/module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8",
    "/public4/opendas/DL_DATA/llm-models-qua/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8",
]
DEFAULT_OUTPUT_PATH = "/module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp"

# Main-net module globs (kept when merging compression_config).
MAIN_W4_MODULES = [
    "model.language_model.layers.*.mlp.experts.gate_up_proj",
    "model.language_model.layers.*.mlp.experts.down_proj",
]
MAIN_W8_MODULES = [
    "model.language_model.layers.*.self_attn.q_proj",
    "model.language_model.layers.*.self_attn.k_proj",
    "model.language_model.layers.*.self_attn.v_proj",
    "model.language_model.layers.*.mlp.shared_expert.gate_proj",
    "model.language_model.layers.*.mlp.shared_expert.up_proj",
    "model.language_model.layers.*.mlp.shared_expert.down_proj",
    "model.language_model.layers.*.linear_attn.in_proj_qkv",
    "model.language_model.layers.*.linear_attn.in_proj_z",
]

MTP_W4_MODULES = [
    "mtp.layers.*.mlp.experts.gate_up_proj",
    "mtp.layers.*.mlp.experts.down_proj",
]
MTP_W8_MODULES = [
    "mtp.layers.*.self_attn.q_proj",
    "mtp.layers.*.self_attn.k_proj",
    "mtp.layers.*.self_attn.v_proj",
    "mtp.layers.*.mlp.shared_expert.gate_proj",
    "mtp.layers.*.mlp.shared_expert.up_proj",
    "mtp.layers.*.mlp.shared_expert.down_proj",
]

# Replace blanket ``re:^mtp\\..*`` with these (keep non-quant MTP layers BF16).
MTP_IGNORE_PATTERNS = [
    r"re:^mtp\.layers\.\d+\.self_attn\.o_proj(?:\.|$)",
    r"re:^mtp\.layers\.\d+\.mlp\.gate(?:\.|$)",
    r"re:^mtp\.layers\.\d+\.mlp\.shared_expert_gate(?:\.|$)",
    r"re:^mtp\.layers\.\d+\.self_attn\.indexer.*",
    r"re:^mtp\..*hyper_connection.*",
    r"re:^mtp\.fc_.*",
    r"re:^mtp\.pre_fc_norm_.*",
    r"re:^mtp\.layers\.\d+\.self_attn\.(q_norm|k_norm)(?:\.|$)",
]

MTP_EXPERT_RE = re.compile(
    r"^mtp\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
MTP_W8_WEIGHT_RE = re.compile(
    r"^mtp\.layers\.\d+\."
    r"(?:self_attn\.(q_proj|k_proj|v_proj)|"
    r"mlp\.shared_expert\.(gate_proj|up_proj|down_proj))"
    r"\.weight$"
)


def resolve_default_input() -> str:
    for path in DEFAULT_INPUT_CANDIDATES:
        if Path(path).is_dir() and (Path(path) / "config.json").is_file():
            return path
    return DEFAULT_INPUT_CANDIDATES[0]


def is_mtp_expert(name: str) -> bool:
    return MTP_EXPERT_RE.match(name) is not None


def is_mtp_w8_weight(name: str) -> bool:
    return MTP_W8_WEIGHT_RE.match(name) is not None


def expert_scale_name(weight_name: str) -> str:
    return f"{weight_name}.weight_scale"


def w8_scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"not a weight tensor: {weight_name}")
    return weight_name[: -len(".weight")] + ".weight_scale"


def weight_quant_int4_per_channel(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric INT4 with K-nibble packing (slimquant_w4a8)."""
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
        _e, _n, k = tensor.shape
        if k % 2 != 0:
            raise ValueError(f"INT4 packing requires even K, got {k}")
        abs_max = torch.abs(tensor).amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (abs_max / 7.0) / 16.0
        quantized = torch.round(tensor / (scale * 16.0)).clamp(-8, 7).to(torch.int8)
        unsigned = quantized.to(torch.uint8)
        even = unsigned[..., ::2]
        odd = unsigned[..., 1::2]
        packed = ((even << 4) | (odd & 0x0F)).to(torch.int8)
        return packed.contiguous(), scale.to(torch.float32)

    raise ValueError(f"expected 2D or 3D weight, got shape {tuple(tensor.shape)}")


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
                if is_mtp_expert(weight_name):
                    key = (
                        "mtp_expert_bf16"
                        if weight.dtype
                        in (torch.bfloat16, torch.float16, torch.float32)
                        else "mtp_expert_unexpected_dtype"
                    )
                    local_stats[key] = local_stats.get(key, 0) + 1
                elif is_mtp_w8_weight(weight_name):
                    key = (
                        "mtp_w8_bf16"
                        if weight.dtype
                        in (torch.bfloat16, torch.float16, torch.float32)
                        else "mtp_w8_unexpected_dtype"
                    )
                    local_stats[key] = local_stats.get(key, 0) + 1
                elif weight_name.startswith("mtp."):
                    local_stats["mtp_kept_bf16"] = (
                        local_stats.get("mtp_kept_bf16", 0) + 1
                    )
                else:
                    local_stats["copied_other"] = local_stats.get("copied_other", 0) + 1
            continue

        for weight_name, weight in state_dict.items():
            # Drop stale MTP W8 scales if re-running on a partially converted tree.
            if weight_name.endswith(".weight_scale") and is_mtp_w8_weight(
                weight_name.replace(".weight_scale", ".weight")
            ):
                local_stats["dropped_stale_mtp_w8_scale"] = (
                    local_stats.get("dropped_stale_mtp_w8_scale", 0) + 1
                )
                continue
            if weight_name.endswith(".weight_scale") and is_mtp_expert(
                weight_name[: -len(".weight_scale")]
            ):
                local_stats["dropped_stale_mtp_w4_scale"] = (
                    local_stats.get("dropped_stale_mtp_w4_scale", 0) + 1
                )
                continue

            if is_mtp_expert(weight_name):
                if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise TypeError(
                        f"MTP expert {weight_name} has dtype {weight.dtype}, "
                        "expected floating BF16/FP16/FP32."
                    )
                if weight.dim() not in (2, 3):
                    raise ValueError(
                        f"MTP expert {weight_name} has shape {tuple(weight.shape)}, "
                        "expected 2D or fused 3D (E, N, K)."
                    )
                weight_gpu = weight.to(device=device, dtype=torch.float32)
                packed, scale = weight_quant_int4_per_channel(weight_gpu)
                scale_name = expert_scale_name(weight_name)
                new_state_dict[weight_name] = packed.cpu()
                new_state_dict[scale_name] = scale.cpu()
                shared_weight_map[weight_name] = file_name
                shared_weight_map[scale_name] = file_name
                local_stats["quantized_mtp_int4"] = (
                    local_stats.get("quantized_mtp_int4", 0) + 1
                )
                del weight_gpu, packed, scale
                continue

            if is_mtp_w8_weight(weight_name):
                if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise TypeError(
                        f"MTP W8 target {weight_name} has dtype {weight.dtype}, "
                        "expected floating BF16/FP16/FP32."
                    )
                weight_gpu = weight.to(device=device, dtype=torch.float32)
                q_int8, scale = weight_quant_int8_per_channel(weight_gpu)
                scale_name = w8_scale_name(weight_name)
                new_state_dict[weight_name] = q_int8.cpu()
                new_state_dict[scale_name] = scale.cpu()
                shared_weight_map[weight_name] = file_name
                shared_weight_map[scale_name] = file_name
                local_stats["quantized_mtp_w8"] = (
                    local_stats.get("quantized_mtp_w8", 0) + 1
                )
                del weight_gpu, q_int8, scale
                continue

            new_state_dict[weight_name] = weight
            shared_weight_map[weight_name] = file_name
            if weight_name.startswith("mtp."):
                local_stats["mtp_kept_bf16"] = local_stats.get("mtp_kept_bf16", 0) + 1
            else:
                local_stats["copied_other"] = local_stats.get("copied_other", 0) + 1

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


def _uniq_extend(dst: list, extra: list) -> list:
    seen = set(dst)
    out = list(dst)
    for item in extra:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def build_output_config(config: dict) -> dict:
    """Merge MTP W4/W8 into existing compression_config; drop blanket mtp ignore."""
    existing = (
        config.get("compression_config")
        or (config.get("text_config") or {}).get("compression_config")
        or {}
    )
    mixed = dict(existing.get("mixed_precision") or {})

    experts = dict(mixed.get("experts") or {})
    experts.setdefault("weight_bits", 4)
    experts.setdefault("format", "pack-int4-nibble")
    experts.setdefault("scope", "per_channel")
    experts["modules"] = _uniq_extend(
        list(experts.get("modules") or MAIN_W4_MODULES), MTP_W4_MODULES
    )
    mixed["experts"] = experts

    w8 = dict(mixed.get("w8a8_include") or {})
    w8.setdefault("weight_bits", 8)
    w8.setdefault("activation_bits", 8)
    w8.setdefault("format", "int8-per-channel")
    w8.setdefault("scope", "per_channel")
    w8.setdefault("activation_scheme", "dynamic_per_token")
    w8["modules"] = _uniq_extend(
        list(w8.get("modules") or MAIN_W8_MODULES), MTP_W8_MODULES
    )
    mixed["w8a8_include"] = w8

    # Preserve ngram_embedding / other mixed blocks from input.
    for key, value in (existing.get("mixed_precision") or {}).items():
        if key not in mixed:
            mixed[key] = value

    ignore = [
        x
        for x in (existing.get("ignore") or [])
        if x not in ("re:^mtp\\..*", r"re:^mtp\..*")
    ]
    ignore = _uniq_extend(ignore, MTP_IGNORE_PATTERNS)

    compression = {
        "quant_method": existing.get("quant_method") or "slimquant_w4a8",
        "activation_scheme": existing.get("activation_scheme") or "dynamic",
        "ignore": ignore,
        "mixed_precision": mixed,
    }
    # Keep any extra top-level keys from existing compression_config.
    for key, value in existing.items():
        if key not in compression:
            compression[key] = value

    config.pop("quantization_config", None)
    config["compression_config"] = compression
    if "text_config" in config and isinstance(config["text_config"], dict):
        config["text_config"].pop("quantization_config", None)
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
        print("Dry run: Flash-Next MTP → INT4 experts + INT8 attn/shared")
        print(f"Input path: {src_dir}")
        print(f"Safetensor shards: {len(safetensor_files)}")
        print(f"GPUs: {world_size}")
        print("=" * 60)

        dry_stats: dict[str, int] = {}
        for sf in tqdm(safetensor_files, desc="scan"):
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if is_mtp_expert(key):
                        dry_stats["mtp_expert"] = dry_stats.get("mtp_expert", 0) + 1
                    elif is_mtp_w8_weight(key):
                        dry_stats["mtp_w8"] = dry_stats.get("mtp_w8", 0) + 1
                    elif key.startswith("mtp."):
                        dry_stats["mtp_other"] = dry_stats.get("mtp_other", 0) + 1
                    else:
                        dry_stats["other"] = dry_stats.get("other", 0) + 1

        print("\nWeight classification:")
        print(json.dumps(dry_stats, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"\nTotal entries: {sum(dry_stats.values())}")
        print("MTP experts → INT4 nibble; MTP q/k/v + shared_expert → INT8.")
        print("Other MTP / main-net tensors kept.")
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
    print("Phase 1: MTP experts→INT4, attn/shared→INT8 (main-net copied)")
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

    if stats.get("quantized_mtp_int4", 0) <= 0:
        raise RuntimeError(
            "No MTP expert weights were quantized. Check that input still has "
            "BF16 mtp.layers.*.mlp.experts.{gate_up_proj,down_proj}."
        )
    if stats.get("quantized_mtp_w8", 0) <= 0:
        raise RuntimeError(
            "No MTP W8 weights were quantized. Check that input still has "
            "BF16 mtp self_attn.q/k/v and shared_expert.* weights."
        )

    print("\n" + "=" * 60)
    print("Phase 2: Updating index and compression_config")
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
        f.write("qwen38_flash_next_mtp_to_w4a8.py completed\n")
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
            "Quantize Flash-Next MTP: experts INT4 + attn/shared INT8 "
            "(aligned with main-net slimquant_w4a8)"
        )
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=resolve_default_input(),
        help="Checkpoint with BF16 MTP (default: auto-detect ngram-int8 path)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output directory (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze MTP targets without writing files",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.dry_run)
