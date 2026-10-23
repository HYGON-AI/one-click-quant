"""
GLM-5.3-Flash block-FP8 → mixed INT4 (routed experts) + INT8 (shared / attn QKV / dense MLP).

仿造:
  - kimi_k3_mxfp4_to_int4.py      (expert → per-channel packed INT4)
  - kimi-k3-int4-to-int4_8.py     (W8 include → per-channel INT8)

输入: HuggingFace block-FP8 checkpoint
  - ``*.weight`` (F8_E4M3) + ``*.weight_scale_inv`` (block 128×128)
  - 以及已是 BF16 的 Linear（如 linear-attention 的 q/k/v_proj）

输出:
  - routed experts ``mlp.experts.*.{gate,up,down}_proj`` → packed INT4 + ``.weight_scale``
  - shared experts / dense MLP(L0-2) / attn QKV(含 DSA q_a/q_b/kv_a) → INT8 + ``.weight_scale``
  - 去掉全部 ``weight_scale_inv``；其余张量原样拷贝
  - ``compression_config.quant_method = slimquant_w4a8``（与 Kimi 脚本一致）

默认路径:
  --input-path  /public/opendas/DL_DATA/llm-models/ZhipuAI/GLM-5.3-Flash
  --output-path /parastor/share/wufan/model/glm-5.3-flash-int4-int8

用法:
  python glm-5.3-flash-fp8-to-int4-int8.py --dry-run
  python glm-5.3-flash-fp8-to-int4-int8.py
  python glm-5.3-flash-fp8-to-int4-int8.py \\
      --input-path /path/to/GLM-5.3-Flash \\
      --output-path /path/to/out
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from argparse import ArgumentParser
from glob import glob
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

DEFAULT_INPUT = "/public/opendas/DL_DATA/llm-models/ZhipuAI/GLM-5.3-Flash"
DEFAULT_OUTPUT = "/parastor/share/wufan/model/glm-5.3-flash-int4-int8"
DEFAULT_BLOCK = 128

# ---------------------------------------------------------------------------
# Name patterns (GLM-5.3-Flash / Glm5Next)
# ---------------------------------------------------------------------------

# Routed MoE experts → INT4
EXPERT_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)

# Shared experts → INT8
SHARED_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.shared_experts\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)

# Dense MLP (first_k_dense_replace=3 → layers 0/1/2) → INT8
DENSE_MLP_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.[012]\.mlp\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)

# Attn QKV: linear-attn q/k/v + DSA MLA q_a/q_b/kv_a → INT8
# (对齐 Kimi best_practice：不含 o_proj / kv_b_proj)
ATTN_QKV_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa)\.weight$"
)


def is_expert_int4_target(name: str) -> bool:
    return EXPERT_WEIGHT_RE.match(name) is not None


def is_w8_target(name: str) -> bool:
    return (
        SHARED_WEIGHT_RE.match(name) is not None
        or DENSE_MLP_WEIGHT_RE.match(name) is not None
        or ATTN_QKV_WEIGHT_RE.match(name) is not None
    )


def scale_inv_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(weight_name)
    return weight_name[: -len(".weight")] + ".weight_scale_inv"


def out_scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(weight_name)
    return weight_name[: -len(".weight")] + ".weight_scale"


# ---------------------------------------------------------------------------
# Block FP8 → float
# ---------------------------------------------------------------------------

def dequantize_weight_from_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    group_size: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Block FP8 (e4m3) * scale_inv → float32 (same layout as DeepSeek/GLM dumps)."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2D FP8 weight, got {tuple(weight.shape)}")
    g = int(group_size)
    d_out, d_in = weight.shape
    pad_out = math.ceil(d_out / g) * g - d_out
    pad_in = math.ceil(d_in / g) * g - d_in
    w = F.pad(weight.float(), (0, pad_in, 0, pad_out))
    s = scale_inv.float()
    d_out_pad, d_in_pad = w.shape
    expected = (d_out_pad // g, d_in_pad // g)
    if tuple(s.shape) != expected:
        if s.numel() != expected[0] * expected[1]:
            raise ValueError(
                f"FP8 scale shape {tuple(s.shape)} incompatible with weight "
                f"{tuple(weight.shape)} group={g} (expected blocks {expected})"
            )
        s = s.reshape(expected)
    w = w.view(d_out_pad // g, g, d_in_pad // g, g)
    s = s.view(d_out_pad // g, 1, d_in_pad // g, 1)
    out = (w * s).view(d_out_pad, d_in_pad)[:d_out, :d_in]
    return out.contiguous()


def _read_block_size(config: dict) -> int:
    qc = config.get("quantization_config") or {}
    block = qc.get("weight_block_size")
    if isinstance(block, (list, tuple)) and block:
        return int(block[0])
    return DEFAULT_BLOCK


# ---------------------------------------------------------------------------
# INT4 / INT8 (slimquant convention, same as Kimi scripts)
# ---------------------------------------------------------------------------

def weight_quant_int4_per_channel(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric INT4, nibble-packed. scale = absmax/7/16."""
    n, k = tensor.shape
    if k % 2 != 0:
        raise ValueError(f"INT4 packing requires even K, got {k}")
    qmax = 7.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    scale = (abs_max / qmax) / 16.0
    quantized = torch.round(tensor / (scale * 16.0)).clamp(-8, 7).to(torch.int8)
    unsigned = quantized.to(torch.uint8)
    even = unsigned[..., ::2]
    odd = unsigned[..., 1::2]
    packed = ((even << 4) | (odd & 0x0F)).to(torch.int8)
    return packed.contiguous(), scale.to(torch.float32)


def weight_quant_int8_per_channel(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric INT8: scale = absmax / 127."""
    assert tensor.dim() == 2, f"expected 2D weight, got {tuple(tensor.shape)}"
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    scale = abs_max / qmax
    quantized = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

def copy_model_assets(src_dir: Path, dst_dir: Path) -> None:
    allowed_suffixes = {".json", ".py", ".md", ".jinja", ".txt"}
    allowed_names = {"__init__.py", ".gitattributes", "LICENSE", "chat_template.jinja"}
    for entry in src_dir.iterdir():
        if entry.is_dir():
            continue
        if entry.suffix == ".safetensors":
            continue
        if entry.suffix in allowed_suffixes or entry.name in allowed_names:
            shutil.copy2(entry, dst_dir / entry.name)


def merge_stats(stats_by_rank: dict) -> dict:
    merged: Dict[str, int] = {}
    for stats in stats_by_rank.values():
        for key, value in stats.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def build_output_config(config: dict) -> dict:
    config.pop("quantization_config", None)
    if isinstance(config.get("text_config"), dict):
        config["text_config"].pop("quantization_config", None)

    config["compression_config"] = {
        "quant_method": "slimquant_w4a8",
        "activation_scheme": "dynamic",
        "mixed_precision": {
            "experts": {
                "weight_bits": 4,
                "format": "pack-int4-nibble",
                "scope": "per_channel",
                "modules": [
                    "model.language_model.layers.*.mlp.experts.*.gate_proj",
                    "model.language_model.layers.*.mlp.experts.*.up_proj",
                    "model.language_model.layers.*.mlp.experts.*.down_proj",
                ],
            },
            "w8a8_include": {
                "weight_bits": 8,
                "activation_bits": 8,
                "format": "int8-per-channel",
                "scope": "per_channel",
                "activation_scheme": "dynamic_per_token",
                "modules": [
                    "model.language_model.layers.*.mlp.shared_experts.gate_proj",
                    "model.language_model.layers.*.mlp.shared_experts.up_proj",
                    "model.language_model.layers.*.mlp.shared_experts.down_proj",
                    "model.language_model.layers.[0-2].mlp.gate_proj",
                    "model.language_model.layers.[0-2].mlp.up_proj",
                    "model.language_model.layers.[0-2].mlp.down_proj",
                    "model.language_model.layers.*.self_attn.q_proj",
                    "model.language_model.layers.*.self_attn.k_proj",
                    "model.language_model.layers.*.self_attn.v_proj",
                    "model.language_model.layers.*.self_attn.q_a_proj",
                    "model.language_model.layers.*.self_attn.q_b_proj",
                    "model.language_model.layers.*.self_attn.kv_a_proj_with_mqa",
                ],
            },
        },
    }
    return config


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _load_scale(
    input_dir: str,
    weight_map: Dict[str, str],
    s_key: str,
    local_state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    if s_key in local_state:
        return local_state[s_key]
    shard = weight_map.get(s_key)
    if shard is None:
        raise KeyError(f"missing scale {s_key} in weight_map")
    with safe_open(os.path.join(input_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(s_key)


def _to_float_weight(
    weight: torch.Tensor,
    weight_name: str,
    local_state: Dict[str, torch.Tensor],
    input_dir: str,
    weight_map: Dict[str, str],
    group_size: int,
    device: str,
) -> torch.Tensor:
    """FP8(+scale_inv) or floating → float32 on ``device``."""
    s_key = scale_inv_name(weight_name)
    if s_key in weight_map:
        scale = _load_scale(input_dir, weight_map, s_key, local_state)
        w = weight.to(device=device, non_blocking=True)
        s = scale.to(device=device, non_blocking=True)
        return dequantize_weight_from_fp8(w, s, group_size=group_size)
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight.to(device=device, dtype=torch.float32, non_blocking=True)
    raise TypeError(
        f"{weight_name}: dtype={weight.dtype} and no weight_scale_inv; "
        "cannot convert to float for quantization"
    )


def worker_process_shard(
    rank: int,
    world_size: int,
    safetensor_files: list,
    input_dir: str,
    output_dir: str,
    weight_map: dict,
    group_size: int,
    shared_weight_map,
    shared_stats,
    dry_run: bool,
) -> None:
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    local_files = safetensor_files[rank::world_size]
    local_stats: Dict[str, int] = {}

    for safetensor_file in tqdm(local_files, position=rank, desc=f"GPU {rank}"):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device="cpu")

        if dry_run:
            for name, tensor in state_dict.items():
                if name.endswith(".weight_scale_inv"):
                    local_stats["fp8_scale_inv"] = local_stats.get("fp8_scale_inv", 0) + 1
                elif is_expert_int4_target(name):
                    local_stats["expert_int4_targets"] = (
                        local_stats.get("expert_int4_targets", 0) + 1
                    )
                elif is_w8_target(name):
                    local_stats["w8_targets"] = local_stats.get("w8_targets", 0) + 1
                else:
                    local_stats["kept_other"] = local_stats.get("kept_other", 0) + 1
            continue

        new_state: Dict[str, torch.Tensor] = {}

        for name, tensor in state_dict.items():
            # Drop FP8 scales from output
            if name.endswith(".weight_scale_inv"):
                local_stats["dropped_scale_inv"] = (
                    local_stats.get("dropped_scale_inv", 0) + 1
                )
                continue

            if is_expert_int4_target(name):
                fp = _to_float_weight(
                    tensor,
                    name,
                    state_dict,
                    input_dir,
                    weight_map,
                    group_size,
                    device,
                )
                packed, scale = weight_quant_int4_per_channel(fp)
                s_name = out_scale_name(name)
                new_state[name] = packed.cpu()
                new_state[s_name] = scale.cpu()
                shared_weight_map[name] = file_name
                shared_weight_map[s_name] = file_name
                local_stats["quantized_int4"] = local_stats.get("quantized_int4", 0) + 1
                del fp, packed, scale
                torch.cuda.empty_cache()
                continue

            if is_w8_target(name):
                fp = _to_float_weight(
                    tensor,
                    name,
                    state_dict,
                    input_dir,
                    weight_map,
                    group_size,
                    device,
                )
                q8, scale = weight_quant_int8_per_channel(fp)
                s_name = out_scale_name(name)
                new_state[name] = q8.cpu()
                new_state[s_name] = scale.cpu()
                shared_weight_map[name] = file_name
                shared_weight_map[s_name] = file_name
                local_stats["quantized_int8"] = local_stats.get("quantized_int8", 0) + 1
                del fp, q8, scale
                torch.cuda.empty_cache()
                continue

            # Keep as-is (BF16 norms, vision, o_proj, indexer, ...)
            # If this is an FP8 weight that we chose NOT to re-quantize, leave FP8?
            # Safer: dequant leftover FP8 linears to BF16 so output has no FP8+scale_inv.
            if name.endswith(".weight") and scale_inv_name(name) in weight_map:
                fp = _to_float_weight(
                    tensor,
                    name,
                    state_dict,
                    input_dir,
                    weight_map,
                    group_size,
                    device,
                )
                new_state[name] = fp.to(torch.bfloat16).cpu()
                shared_weight_map[name] = file_name
                local_stats["dequant_fp8_to_bf16"] = (
                    local_stats.get("dequant_fp8_to_bf16", 0) + 1
                )
                del fp
                torch.cuda.empty_cache()
                continue

            new_state[name] = tensor
            shared_weight_map[name] = file_name
            local_stats["kept_other"] = local_stats.get("kept_other", 0) + 1

        save_file(new_state, os.path.join(output_dir, file_name))
        del state_dict, new_state
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    shared_stats[rank] = local_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    input_path: str,
    output_path: Optional[str],
    dry_run: bool,
) -> None:
    src_dir = Path(input_path)
    if not src_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {src_dir}")

    config_path = src_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Can not find config.json under {src_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    group_size = _read_block_size(config)

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)
    weight_map: Dict[str, str] = dict(model_index["weight_map"])

    safetensor_files = sorted(glob(str(src_dir / "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"Can not find any *.safetensors under {src_dir}")

    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found")

    if dry_run:
        print("=" * 60)
        print("Dry run: GLM-5.3-Flash FP8 → INT4(experts) + INT8(shared/attn/dense)")
        print(f"Input: {src_dir}")
        print(f"Shards: {len(safetensor_files)}  GPUs: {world_size}  block={group_size}")
        print("=" * 60)
        dry_stats: Dict[str, int] = {}
        for sf in tqdm(safetensor_files, desc="scan"):
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith(".weight_scale_inv"):
                        dry_stats["fp8_scale_inv"] = dry_stats.get("fp8_scale_inv", 0) + 1
                    elif is_expert_int4_target(key):
                        dry_stats["expert_int4_targets"] = (
                            dry_stats.get("expert_int4_targets", 0) + 1
                        )
                    elif is_w8_target(key):
                        dry_stats["w8_targets"] = dry_stats.get("w8_targets", 0) + 1
                    else:
                        dry_stats["other"] = dry_stats.get("other", 0) + 1
        print(json.dumps(dry_stats, indent=2, ensure_ascii=False, sort_keys=True))
        print(
            "\nINT4: mlp.experts.*.{gate,up,down}_proj\n"
            "INT8: shared_experts + layers.[0-2].mlp + self_attn "
            "{q,k,v,q_a,q_b,kv_a_proj_with_mqa}\n"
            "Other FP8 weights (e.g. o_proj) → BF16; rest kept."
        )
        return

    if not output_path:
        raise ValueError("--output-path is required unless --dry-run is set")

    dst_dir = Path(output_path)
    if dst_dir.resolve() == src_dir.resolve():
        raise ValueError("--input-path and --output-path must differ")
    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_model_assets(src_dir, dst_dir)

    print("=" * 60)
    print("GLM-5.3-Flash: FP8 → INT4(experts) + INT8(shared/attn QKV/dense MLP)")
    print(f"GPUs: {world_size}  |  Shards: {len(safetensor_files)}  |  block={group_size}")
    print(f"Input:  {src_dir}")
    print(f"Output: {dst_dir}")
    print("=" * 60)

    manager = Manager()
    shared_weight_map = manager.dict()
    shared_stats = manager.dict()

    # mp.spawn uses spawn start method (CUDA-safe)
    mp.spawn(
        worker_process_shard,
        args=(
            world_size,
            safetensor_files,
            str(src_dir),
            str(dst_dir),
            weight_map,
            group_size,
            shared_weight_map,
            shared_stats,
            False,
        ),
        nprocs=world_size,
        join=True,
    )

    stats = merge_stats(dict(shared_stats))
    out_wm = dict(shared_weight_map)
    print("\nConversion stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))

    if stats.get("quantized_int4", 0) <= 0:
        raise RuntimeError("No expert INT4 weights quantized; check naming patterns.")
    if stats.get("quantized_int8", 0) <= 0:
        raise RuntimeError("No INT8 targets quantized; check naming patterns.")

    total_size = 0
    for sf in sorted(dst_dir.glob("*.safetensors")):
        total_size += sf.stat().st_size
    model_index["weight_map"] = out_wm
    model_index["metadata"] = model_index.get("metadata", {})
    model_index["metadata"]["total_size"] = total_size

    out_index = dst_dir / "model.safetensors.index.json"
    with open(out_index, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False)
        f.write("\n")

    out_cfg = build_output_config(config)
    with open(dst_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    done_path = dst_dir / "QUANTIZATION_DONE.txt"
    with open(done_path, "w", encoding="utf-8") as f:
        f.write("glm-5.3-flash-fp8-to-int4-int8.py completed\n")
        f.write(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
        f.write("\n")

    print(f"Updated index: {out_index}")
    print(f"Updated config: {dst_dir / 'config.json'}")
    print(f"weight_map entries: {len(out_wm)}")
    print(f"Wrote {done_path}")
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="GLM-5.3-Flash FP8 → INT4(routed experts) + INT8(shared/attn/dense)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT,
        help=f"HF model dir (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"output dir (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="classify weights without writing",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.dry_run)
