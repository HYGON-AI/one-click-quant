"""
Quantize Qwen3.8-Flash-Next PLE ``ngram_embedding`` shards to INT8.

Input (already mixed W4/W8 slimquant, ngram still BF16):
  /module/Qwen3.8-Flash-Next-w4a8-slimquant
  model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_N.weight
    shape ~(2.5M, 160) bf16 × 128 shards  (~102 GiB)

Output:
  INT8 packed table + scale(s). Other tensors copied as-is.

Scale modes:
  - per_tensor (default): one scalar
      ...ngram_embedding.weight_scale
    Matches the official FP8 PLE layout (sglang does
    ``embeddings * ngram_embedding.weight_scale``). Best for future runtime
    INT8 support that mirrors the FP8 path.
  - per_channel: per-row scale per shard
      ...ngram_embedding.shard_N.weight_scale  # (rows, 1)
    Higher fidelity; needs runtime to apply ``scale[ids]`` after gather.

NOTE: current sglang PLE path loads BF16 or FP8 tables, not INT8 yet.
This script only produces the checkpoint; serving still needs an INT8
(or dequant-on-load) path.

Usage:
  python qwen38_flash_next_ngram_to_int8.py --dry-run
  python qwen38_flash_next_ngram_to_int8.py
  python qwen38_flash_next_ngram_to_int8.py --scale-mode per_channel
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

DEFAULT_INPUT_PATH = "/module/Qwen3.8-Flash-Next-w4a8-slimquant"
DEFAULT_OUTPUT_PATH = "/module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8"

NGRAM_SHARD_WEIGHT_RE = re.compile(
    r".*\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$"
)
NGRAM_MODULE_PREFIX_RE = re.compile(
    r"^(?P<prefix>.*\.ple\.ple_embedding\.ngram_embedding)\.shard_\d+\.weight$"
)


def is_ngram_shard_weight(name: str) -> bool:
    return NGRAM_SHARD_WEIGHT_RE.match(name) is not None


def ngram_module_prefix(weight_name: str) -> str:
    m = NGRAM_MODULE_PREFIX_RE.match(weight_name)
    if m is None:
        raise ValueError(f"not an ngram shard weight: {weight_name}")
    return m.group("prefix")


def per_channel_scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(weight_name)
    return weight_name[: -len(".weight")] + ".weight_scale"


def per_tensor_scale_name(weight_name: str) -> str:
    return ngram_module_prefix(weight_name) + ".weight_scale"


def weight_quant_int8_per_channel(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric INT8: scale shape (N, 1)."""
    assert tensor.dim() == 2, f"expected 2D, got {tuple(tensor.shape)}"
    qmax = 127.0
    abs_max = torch.abs(tensor).amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = abs_max / qmax
    q = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return q.contiguous(), scale.to(torch.float32)


def weight_quant_int8_per_tensor(
    tensor: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Quantize with a precomputed scalar (or 0-dim / (1,) / (1,1)) scale."""
    assert tensor.dim() == 2, f"expected 2D, got {tuple(tensor.shape)}"
    qmax = 127.0
    q = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return q.contiguous()


def compute_absmax(tensor: torch.Tensor) -> torch.Tensor:
    return torch.abs(tensor).amax().clamp(min=1e-12)


def copy_model_assets(src_dir: Path, dst_dir: Path) -> None:
    allowed_suffixes = {".json", ".py", ".md", ".jinja", ".txt"}
    allowed_names = {"__init__.py", ".gitattributes"}
    for entry in src_dir.iterdir():
        if entry.is_dir() or entry.suffix == ".safetensors":
            continue
        if entry.suffix in allowed_suffixes or entry.name in allowed_names:
            shutil.copy2(entry, dst_dir / entry.name)


def scan_global_absmax(safetensor_files: list[str]) -> dict[str, float]:
    """Return {ngram_module_prefix: absmax} across all shards."""
    absmax_by_prefix: dict[str, float] = {}
    for sf in tqdm(safetensor_files, desc="scan absmax"):
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not is_ngram_shard_weight(key):
                    continue
                prefix = ngram_module_prefix(key)
                t = f.get_tensor(key)
                a = float(compute_absmax(t.float()).item())
                absmax_by_prefix[prefix] = max(absmax_by_prefix.get(prefix, 0.0), a)
    if not absmax_by_prefix:
        raise RuntimeError("No ngram_embedding.shard_*.weight found.")
    return absmax_by_prefix


def worker_process_shard(
    rank: int,
    world_size: int,
    safetensor_files: list,
    output_dir: str,
    shared_weight_map,
    shared_stats,
    dry_run: bool,
    scale_mode: str,
    global_scale_by_prefix: dict,
):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    local_files = safetensor_files[rank::world_size]
    local_stats: dict[str, int] = {}

    for safetensor_file in tqdm(local_files, position=rank, desc=f"GPU {rank}"):
        file_name = os.path.basename(safetensor_file)

        if dry_run:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if is_ngram_shard_weight(key):
                        local_stats["ngram_bf16"] = local_stats.get("ngram_bf16", 0) + 1
                    else:
                        local_stats["kept_other"] = local_stats.get("kept_other", 0) + 1
            continue

        # Fast path: no ngram in this shard → copy file
        has_ngram = False
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if is_ngram_shard_weight(key):
                    has_ngram = True
                    break

        if not has_ngram:
            shutil.copy2(safetensor_file, os.path.join(output_dir, file_name))
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    shared_weight_map[key] = file_name
                    local_stats["copied_other"] = local_stats.get("copied_other", 0) + 1
            continue

        state_dict = load_file(safetensor_file, device="cpu")
        new_state_dict: dict[str, torch.Tensor] = {}

        for weight_name, weight in state_dict.items():
            if not is_ngram_shard_weight(weight_name):
                new_state_dict[weight_name] = weight
                shared_weight_map[weight_name] = file_name
                local_stats["kept_other"] = local_stats.get("kept_other", 0) + 1
                continue

            if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise TypeError(
                    f"{weight_name}: expected floating dtype, got {weight.dtype}"
                )
            if weight.dim() != 2:
                raise ValueError(
                    f"{weight_name}: expected 2D embedding shard, "
                    f"got {tuple(weight.shape)}"
                )

            weight_gpu = weight.to(device=device, dtype=torch.float32)
            prefix = ngram_module_prefix(weight_name)

            if scale_mode == "per_channel":
                q_int8, scale = weight_quant_int8_per_channel(weight_gpu)
                scale_name = per_channel_scale_name(weight_name)
                new_state_dict[weight_name] = q_int8.cpu()
                new_state_dict[scale_name] = scale.cpu()
                shared_weight_map[weight_name] = file_name
                shared_weight_map[scale_name] = file_name
                local_stats["quantized_ngram_int8"] = (
                    local_stats.get("quantized_ngram_int8", 0) + 1
                )
                del weight_gpu, q_int8, scale
            else:
                # per_tensor: scalar scale written once in main() after workers
                absmax = float(global_scale_by_prefix[prefix])
                scale = torch.tensor(
                    [absmax / 127.0], dtype=torch.float32, device=device
                )
                q_int8 = weight_quant_int8_per_tensor(weight_gpu, scale)
                new_state_dict[weight_name] = q_int8.cpu()
                shared_weight_map[weight_name] = file_name
                local_stats["quantized_ngram_int8"] = (
                    local_stats.get("quantized_ngram_int8", 0) + 1
                )
                del weight_gpu, q_int8, scale

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


def build_output_config(config: dict, scale_mode: str) -> dict:
    compression = config.get("compression_config")
    if not isinstance(compression, dict):
        compression = {"quant_method": "slimquant_w4a8", "activation_scheme": "dynamic"}

    mixed = compression.setdefault("mixed_precision", {})
    mixed["ngram_embedding"] = {
        "weight_bits": 8,
        "format": "int8",
        "scope": "per_tensor" if scale_mode == "per_tensor" else "per_channel",
        "modules": [
            "model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_*",
        ],
        "scale_name": (
            "model.language_model.layers.*.ple.ple_embedding.ngram_embedding.weight_scale"
            if scale_mode == "per_tensor"
            else "model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_*.weight_scale"
        ),
    }
    # Remove ngram from ignore if present as a plain string list
    ignore = compression.get("ignore")
    if isinstance(ignore, list):
        compression["ignore"] = [
            p for p in ignore if "ple" not in p and "ngram" not in p
        ]

    config["compression_config"] = compression
    if "text_config" in config and isinstance(config["text_config"], dict):
        config["text_config"]["compression_config"] = compression
        # Hint for loaders that understand PLE dtype (FP8 uses float8_e4m3fn).
        config["text_config"]["ple_embedding_dtype"] = "int8"
    return config


def fill_weight_map_from_copied_files(
    src_index: dict, dst_dir: Path, weight_map: dict
) -> dict:
    """Ensure weight_map covers tensors in copied (non-rewritten) shards."""
    # weight_map already filled by workers for all keys they saw; just return it.
    return weight_map


def main(
    input_path: str,
    output_path: str | None,
    dry_run: bool,
    scale_mode: str,
) -> None:
    if scale_mode not in ("per_tensor", "per_channel"):
        raise ValueError(f"invalid --scale-mode: {scale_mode}")

    src_dir = Path(input_path)
    if not src_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {src_dir}")
    if not (src_dir / "config.json").exists():
        raise FileNotFoundError(f"Can not find config.json under {src_dir}")

    safetensor_files = sorted(glob(str(src_dir / "model-*.safetensors")))
    if not safetensor_files:
        safetensor_files = sorted(glob(str(src_dir / "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"Can not find any *.safetensors under {src_dir}")

    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found")

    # Count / absmax scan
    print("=" * 60)
    print("Qwen3.8-Flash-Next ngram_embedding → INT8")
    print(f"Input:  {src_dir}")
    print(f"Scale:  {scale_mode}")
    print(f"Shards: {len(safetensor_files)}  GPUs: {world_size}")
    print("=" * 60)

    ngram_count = 0
    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                if is_ngram_shard_weight(key):
                    ngram_count += 1
    print(f"Found {ngram_count} ngram_embedding shard weights")
    if ngram_count <= 0:
        raise RuntimeError("No ngram_embedding.shard_*.weight tensors found.")

    global_scale_by_prefix: dict[str, float] = {}
    if scale_mode == "per_tensor" and not dry_run:
        absmax_by_prefix = scan_global_absmax(safetensor_files)
        global_scale_by_prefix = absmax_by_prefix
        print("Global absmax by module:")
        for prefix, a in absmax_by_prefix.items():
            print(f"  {prefix}: absmax={a:.6e}  scale={a/127.0:.6e}")

    if dry_run:
        print("Dry run only — no files written.")
        print(
            "Quantizable: yes. Embedding table is a large lookup; INT8 is "
            "standard. Official FP8 uses a scalar weight_scale; prefer "
            "--scale-mode per_tensor for that layout."
        )
        print(
            "Runtime note: sglang currently loads PLE as BF16 or FP8, not INT8; "
            "checkpoint conversion alone does not enable serving."
        )
        return

    if not output_path:
        raise ValueError("--output-path is required unless --dry-run is set")

    dst_dir = Path(output_path)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_model_assets(src_dir, dst_dir)

    with open(dst_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    index_path = dst_dir / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)

    manager = Manager()
    shared_weight_map = manager.dict()
    shared_stats = manager.dict()
    # Manager dict needs plain types
    global_scale_proxy = manager.dict(global_scale_by_prefix)

    mp.spawn(
        worker_process_shard,
        args=(
            world_size,
            safetensor_files,
            str(dst_dir),
            shared_weight_map,
            shared_stats,
            False,
            scale_mode,
            dict(global_scale_proxy),
        ),
        nprocs=world_size,
        join=True,
    )

    stats = merge_stats(dict(shared_stats))
    weight_map = fill_weight_map_from_copied_files(
        model_index, dst_dir, dict(shared_weight_map)
    )

    print("\nConversion stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
    if stats.get("quantized_ngram_int8", 0) <= 0:
        raise RuntimeError("No ngram shards were quantized.")

    # Write per-tensor scalar scale(s) once (FP8-compatible key).
    if scale_mode == "per_tensor":
        scale_file = "ngram_embedding_scales.safetensors"
        scale_state: dict[str, torch.Tensor] = {}
        for prefix, absmax in global_scale_by_prefix.items():
            scale_name = prefix + ".weight_scale"
            scale_state[scale_name] = torch.tensor(
                [float(absmax) / 127.0], dtype=torch.float32
            )
            weight_map[scale_name] = scale_file
        save_file(scale_state, str(dst_dir / scale_file))
        stats["wrote_per_tensor_scale"] = len(scale_state)
        print(f"Wrote {len(scale_state)} per-tensor scale(s) → {scale_file}")

    total_size = sum(p.stat().st_size for p in dst_dir.glob("*.safetensors"))
    model_index["weight_map"] = weight_map
    model_index["metadata"] = model_index.get("metadata", {})
    model_index["metadata"]["total_size"] = total_size
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False)

    config = build_output_config(config, scale_mode)
    with open(dst_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)

    done_path = dst_dir / "QUANTIZATION_DONE.txt"
    with open(done_path, "w", encoding="utf-8") as f:
        f.write("qwen38_flash_next_ngram_to_int8.py completed\n")
        f.write(f"scale_mode={scale_mode}\n")
        f.write(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
        f.write("\n")

    print(f"Updated model index: {index_path}")
    print(f"Updated config: {dst_dir / 'config.json'}")
    print(f"Total weight_map entries: {len(weight_map)}")
    print(f"Wrote {done_path}")
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Quantize Flash-Next PLE ngram_embedding shards to INT8"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help=f"default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--scale-mode",
        type=str,
        default="per_tensor",
        choices=["per_tensor", "per_channel"],
        help="per_tensor: one scalar scale (FP8-like); per_channel: per-row scale",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.dry_run, args.scale_mode)
