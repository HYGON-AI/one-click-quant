#!/usr/bin/env python3
"""
从 Qwen3.8-2.4T-A95B-FP8 中提取子集，构建 mini 模型用于推理/量化链路验证。

特点：
  - 适配 Qwen3.8 张量命名：model.layers.N.* / model.layers.N.mlp.experts.E.*
  - 同时支持 Qwen3.8 FP8 block wise：weight + weight_scale_inv → BF16
  - 同时支持 FP8_DYNAMIC / channel wise：weight + weight_scale → BF16
  - 支持按层、按 expert 子集裁剪，并把保留层重映射为连续的 model.layers.0..N-1
  - 输出默认是纯 BF16 checkpoint，因此会移除 quantization_config

用法：
  python3 build_mini_qwen3_8.py \
    --src /models/Qwen3.8-2.4T-A95B-FP8 \
    --dst /models/Qwen3.8-2.4T-A95B-FP8-L0-8-E8 \
    -n 8 -e 8

建议：
  Qwen3.8 每层有 512 个 experts；如果只是快速验证，建议加 -e 限制 expert 数，例如 -e 8。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

# torch / safetensors / tqdm 在实际读取权重时再导入，
# 这样 --dry-run 可在轻量环境中只验证 index 筛选逻辑。

_LAYER_REMAP: dict[int, int] = {}


def normalize_path(path: str | Path) -> Path:
    """支持 Linux path，也兼容 WSL 下的 C:\\foo\\bar 写法。"""
    s = str(path)
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s)


def remap_layer_name(name: str) -> str:
    """把 model.layers.<orig>. 重写成 model.layers.<new>."""
    match = re.match(r"(model\.layers\.)(\d+)(\..*)", name)
    if match is None:
        return name
    orig = int(match.group(2))
    if orig not in _LAYER_REMAP:
        return name
    return f"{match.group(1)}{_LAYER_REMAP[orig]}{match.group(3)}"


def should_keep_tensor(
    name: str,
    keep_layers: set[int],
    num_experts: int,
    include_mtp: bool,
) -> bool:
    """判断 Qwen3.8 tensor 是否应纳入 mini 模型。"""
    # 主模型全局权重。
    if name.startswith((
        "lm_head.",
        "model.embed_tokens.",
        "model.norm.",
    )):
        return True

    # MTP 权重：默认保留，避免 config 仍声明 MTP 时加载缺权重。
    if name.startswith("mtp."):
        if not include_mtp:
            return False
        if num_experts > 0:
            expert_match = re.match(r"(mtp\.layers\.\d+\.mlp\.experts\.)(\d+)(\..*)", name)
            if expert_match and int(expert_match.group(2)) >= num_experts:
                return False
        return True

    # 主模型 experts 过滤；num_experts <= 0 表示保留全部 experts。
    if num_experts > 0:
        expert_match = re.match(r"(model\.layers\.\d+\.mlp\.experts\.)(\d+)(\..*)", name)
        if expert_match and int(expert_match.group(2)) >= num_experts:
            return False

    # 主模型层过滤。
    layer_match = re.match(r"model\.layers\.(\d+)\.", name)
    if layer_match:
        return int(layer_match.group(1)) in keep_layers

    return False


def needs_dequant(weight: torch.Tensor) -> bool:
    """判断权重是否为 FP8 dtype。"""
    import torch

    fp8_dtypes = [torch.float8_e4m3fn]
    if hasattr(torch, "float8_e4m3fnuz"):
        fp8_dtypes.append(torch.float8_e4m3fnuz)
    return weight.dtype in tuple(fp8_dtypes)


def dequant_fp8_block(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
    dtype=None,
) -> torch.Tensor:
    """HF FP8 block 权重解量化：weight + weight_scale_inv → BF16。"""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    if weight.dim() != 2:
        raise ValueError(f"FP8 block weight must be 2D, got shape={tuple(weight.shape)}")

    rows, cols = weight.shape
    block_rows, block_cols = block_size
    padded_rows = math.ceil(rows / block_rows) * block_rows
    padded_cols = math.ceil(cols / block_cols) * block_cols

    if (padded_rows, padded_cols) != (rows, cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            dtype=weight.dtype,
            device=weight.device,
        )
        padded[:rows, :cols] = weight
        weight = padded

    num_row_blocks = padded_rows // block_rows
    num_col_blocks = padded_cols // block_cols
    expected_scale_shape = (num_row_blocks, num_col_blocks)
    if tuple(weight_scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"weight_scale_inv shape mismatch: expected={expected_scale_shape}, "
            f"got={tuple(weight_scale_inv.shape)}"
        )

    blocks = weight.reshape(
        num_row_blocks,
        block_rows,
        num_col_blocks,
        block_cols,
    ).transpose(1, 2)
    scale = weight_scale_inv.to(torch.float32).unsqueeze(-1).unsqueeze(-1)
    dequant = (blocks.to(torch.float32) * scale).to(dtype)
    dequant = dequant.transpose(1, 2).reshape(padded_rows, padded_cols)
    return dequant[:rows, :cols].contiguous()


def dequant_fp8_channel(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    dtype=None,
) -> torch.Tensor:
    """FP8_DYNAMIC/channel-wise 权重解量化：weight + weight_scale → BF16。"""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    scale = weight_scale.to(torch.float32)
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1)
    return (weight.to(torch.float32) * scale).to(dtype).contiguous()


def detect_fp8_format(src_format: str, src_weight_map: dict[str, str]) -> str:
    """把 fp8/auto 解析成 fp8-block 或 fp8-channel。"""
    if src_format in {"fp8-block", "fp8-channel", "bf16"}:
        return src_format
    if src_format not in {"fp8", "auto"}:
        raise ValueError(f"不支持的 src_format: {src_format}")

    names = src_weight_map.keys()
    if any(name.endswith(".weight_scale_inv") for name in names):
        return "fp8-block"
    if any(name.endswith(".weight_scale") for name in names):
        return "fp8-channel"
    return "bf16"


def qparam_name_for_weight(name: str, resolved_format: str) -> str | None:
    if resolved_format == "fp8-block":
        return name.replace(".weight", ".weight_scale_inv")
    if resolved_format == "fp8-channel":
        return name.replace(".weight", ".weight_scale")
    return None


def is_aux_quant_tensor(name: str, resolved_format: str) -> bool:
    if resolved_format == "fp8-block":
        return name.endswith(".weight_scale_inv")
    if resolved_format == "fp8-channel":
        return name.endswith(".weight_scale")
    return False


def maybe_slice_gate(name: str, tensor: torch.Tensor, num_experts: int) -> torch.Tensor:
    """按 expert 数裁剪 router/gate 的第一维。"""
    if num_experts <= 0:
        return tensor
    if re.match(r"(?:model\.layers\.\d+|mtp\.layers\.\d+)\.mlp\.gate\.weight$", name):
        if tensor.dim() >= 1 and tensor.shape[0] >= num_experts:
            return tensor[:num_experts].contiguous()
    return tensor


def filter_modules_to_not_convert(
    modules: list[str],
    keep_layers: set[int],
    num_experts: int,
    include_mtp: bool,
) -> list[str]:
    """如果 --keep-quant，裁剪 modules_to_not_convert 并重映射层号。"""
    filtered: list[str] = []
    for module in modules:
        # MTP 不按主层号重映射。
        if module.startswith("mtp."):
            if include_mtp:
                filtered.append(module)
            continue

        layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", module)
        if layer_match:
            orig = int(layer_match.group(1))
            if orig not in keep_layers:
                continue
            filtered.append(f"model.layers.{_LAYER_REMAP[orig]}.{layer_match.group(2)}")
            continue

        # 全局项：lm_head / model.embed_tokens 等。
        filtered.append(module)
    return filtered


def build_mini_config(
    src_dir: Path,
    keep_layers: list[int],
    num_experts: int,
    keep_quant: bool,
    include_mtp: bool,
) -> dict:
    with open(src_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    orig_layer_types = config.get("layer_types")
    if orig_layer_types is not None:
        config["layer_types"] = [orig_layer_types[i] for i in keep_layers]

    total_experts = int(config.get("num_experts", 0))
    actual_experts = num_experts if num_experts > 0 else total_experts

    config["num_hidden_layers"] = len(keep_layers)
    if actual_experts > 0:
        config["num_experts"] = actual_experts
        if "num_experts_per_tok" in config:
            config["num_experts_per_tok"] = min(int(config["num_experts_per_tok"]), actual_experts)

    if not include_mtp:
        config["mtp_num_hidden_layers"] = 0

    if keep_quant:
        qcfg = config.get("quantization_config")
        if isinstance(qcfg, dict) and isinstance(qcfg.get("modules_to_not_convert"), list):
            qcfg["modules_to_not_convert"] = filter_modules_to_not_convert(
                qcfg["modules_to_not_convert"],
                set(keep_layers),
                num_experts,
                include_mtp,
            )
    else:
        # 默认输出 BF16，不能保留源 FP8 quantization_config。
        config.pop("quantization_config", None)

    print(f"  keep_layers (源 0-index):    {keep_layers}")
    print(f"  remap (源 → mini):          {_LAYER_REMAP}")
    if orig_layer_types is not None:
        print(f"  layer_types:                {config['layer_types']}")
    print(f"  num_experts:                {config.get('num_experts')}")
    print(f"  num_experts_per_tok:        {config.get('num_experts_per_tok')}")
    print(f"  mtp_num_hidden_layers:      {config.get('mtp_num_hidden_layers')}")
    print(f"  keep_quant:                 {keep_quant}")
    return config


def resolve_keep_layers(
    num_layers: int | None,
    layer_range: str | None,
    layer_ids: str | None,
    src_num_layers: int,
) -> list[int]:
    provided = [x is not None for x in (num_layers, layer_range, layer_ids)]
    if sum(provided) > 1:
        raise ValueError("--num-layers / --layer-range / --layer-ids 三个参数互斥")

    if layer_ids is not None:
        ids = [int(x) for x in layer_ids.split(",") if x.strip()]
    elif layer_range is not None:
        start_s, end_s = layer_range.split(":", 1)
        ids = list(range(int(start_s), int(end_s)))
    else:
        n = num_layers if num_layers is not None else 4
        ids = list(range(n))

    if not ids:
        raise ValueError("keep_layers 不能为空")
    for idx in ids:
        if not (0 <= idx < src_num_layers):
            raise ValueError(f"层号 {idx} 越界；源模型共 {src_num_layers} 层")
    return sorted(ids)


def build_mini_model(
    src_dir: Path,
    dst_dir: Path,
    keep_layers: list[int],
    num_experts: int,
    src_format: str,
    keep_quant: bool,
    include_mtp: bool,
    dry_run: bool,
):
    global _LAYER_REMAP
    _LAYER_REMAP = {orig: new for new, orig in enumerate(keep_layers)}
    keep_layers_set = set(keep_layers)

    with open(src_dir / "config.json", encoding="utf-8") as f:
        src_config = json.load(f)
    total_experts = int(src_config.get("num_experts", 0))
    if num_experts > 0 and total_experts > 0 and num_experts > total_experts:
        raise ValueError(f"--num-experts={num_experts} 超过源模型 num_experts={total_experts}")

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"未找到 index: {index_path}")
    with open(index_path, encoding="utf-8") as f:
        src_index = json.load(f)
    src_weight_map: dict[str, str] = src_index.get("weight_map", {})
    resolved_src_format = detect_fp8_format(src_format, src_weight_map)

    print(f"源模型张量: {len(src_weight_map)}")
    print(f"源格式: {src_format} -> {resolved_src_format}")
    print(f"保留层 (源 0-index): {keep_layers}")

    files_to_tensors: dict[str, list[str]] = defaultdict(list)
    kept, skipped = 0, 0
    for name, fname in src_weight_map.items():
        if should_keep_tensor(name, keep_layers_set, num_experts, include_mtp):
            files_to_tensors[fname].append(name)
            kept += 1
        else:
            skipped += 1

    print(
        f"保留: {kept}  排除: {skipped}  "
        f"(layers={len(keep_layers)}, experts={num_experts}, include_mtp={include_mtp})"
    )
    if kept == 0:
        raise RuntimeError("未筛选到任何 tensor，请检查源模型命名或筛选参数")
    if dry_run:
        print("dry-run: 仅完成 index 筛选，不读取/写入 safetensors。")
        return

    from safetensors.torch import load_file, save_file
    from tqdm import tqdm

    dst_dir.mkdir(parents=True, exist_ok=True)

    # FP8 权重的 qparam 可能跨 shard，先预收集需要的 scale / scale_inv。
    scale_cache: dict[str, torch.Tensor] = {}
    scale_files = set()
    if resolved_src_format in {"fp8-block", "fp8-channel"} and not keep_quant:
        needed_scales = {
            scale_name
            for tensor_names in files_to_tensors.values()
            for name in tensor_names
            if name.endswith(".weight")
            for scale_name in [qparam_name_for_weight(name, resolved_src_format)]
            if scale_name is not None
        }
        for scale_name in needed_scales:
            fname = src_weight_map.get(scale_name)
            if fname is not None:
                scale_files.add(fname)
        qparam_desc = "weight_scale_inv" if resolved_src_format == "fp8-block" else "weight_scale"
        for fname in tqdm(sorted(scale_files), desc=f"预收集 {qparam_desc}"):
            fpath = src_dir / fname
            if not fpath.exists():
                print(f"  ⚠ 跳过不存在的 scale shard: {fpath}")
                continue
            local = load_file(str(fpath))
            for name in local.keys():
                if name in needed_scales:
                    scale_cache[name] = local[name]
    print(f"  预收集 qparam: {len(scale_cache)} 个")

    all_new: dict[str, torch.Tensor] = {}
    new_weight_map: dict[str, str] = {}
    total_size = 0

    for fname, tensor_names in tqdm(sorted(files_to_tensors.items()), desc="处理 safetensors"):
        fpath = src_dir / fname
        if not fpath.exists():
            print(f"  ⚠ 跳过不存在的文件: {fpath}")
            continue

        tensors = load_file(str(fpath))
        for name in tensor_names:
            tensor = tensors[name]

            # keep_quant 时直接透传，包括 weight_scale_inv。
            if keep_quant:
                key = remap_layer_name(name)
                tensor = maybe_slice_gate(name, tensor, num_experts)
                all_new[key] = tensor
                total_size += tensor.numel() * tensor.element_size()
                new_weight_map[key] = "model.safetensors"
                continue

            # FP8 block / FP8_DYNAMIC 权重 → BF16。只有存在对应 qparam 的 FP8 weight 会进入此分支。
            if resolved_src_format in {"fp8-block", "fp8-channel"} and name.endswith(".weight") and needs_dequant(tensor):
                qparam_name = qparam_name_for_weight(name, resolved_src_format)
                scale = tensors.get(qparam_name)
                if scale is None:
                    scale = scale_cache.get(qparam_name)
                if scale is None:
                    raise ValueError(f"缺少 {qparam_name}，无法解量化 {name}")
                if resolved_src_format == "fp8-block":
                    deq = dequant_fp8_block(tensor, scale)
                else:
                    deq = dequant_fp8_channel(tensor, scale)
                key = remap_layer_name(name)
                all_new[key] = deq
                total_size += deq.numel() * deq.element_size()
                new_weight_map[key] = "model.safetensors"
                continue

            # 解量化输出时不保存 qparam。
            if is_aux_quant_tensor(name, resolved_src_format):
                continue

            # BF16 / 非 FP8 普通 tensor 透传，router gate 按 expert 数裁剪。
            tensor = maybe_slice_gate(name, tensor, num_experts)
            key = remap_layer_name(name)
            all_new[key] = tensor
            total_size += tensor.numel() * tensor.element_size()
            new_weight_map[key] = "model.safetensors"

    print(f"\n保存 safetensors (~{total_size / 1e9:.1f} GB, {len(all_new)} tensors)...")
    save_file(all_new, dst_dir / "model.safetensors")
    with open(dst_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size, "num_tensors": len(new_weight_map)},
                "weight_map": new_weight_map,
            },
            f,
            indent=2,
        )

    config = build_mini_config(src_dir, keep_layers, num_experts, keep_quant, include_mtp)
    with open(dst_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 复制 tokenizer / generation / README 等非权重文件；config 和 index 使用新生成版本。
    skip_names = {"config.json", "model.safetensors.index.json"}
    for p in src_dir.iterdir():
        if p.name in skip_names or p.suffix == ".safetensors":
            continue
        if p.is_file():
            shutil.copy(p, dst_dir / p.name)

    print(f"\n✅ Mini 模型已构建: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="构建 Qwen3.8 mini 模型（支持 FP8 block/channel → BF16）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=str, default="/models/Qwen3.8-2.4T-A95B-FP8", help="源模型目录")
    parser.add_argument("--dst", type=str, default="/models/Qwen3.8-2.4T-A95B-FP8-Mini", help="输出目录")
    parser.add_argument("--src-format", choices=["auto", "fp8", "fp8-block", "fp8-channel", "bf16"], default="auto", help="源模型格式；auto 会自动识别 fp8-block / fp8-channel")
    parser.add_argument("-n", "--num-layers", type=int, default=None, help="保留前 N 层；默认 4")
    parser.add_argument("--layer-range", type=str, default=None, help="截取源层区间 START:END，例 40:44")
    parser.add_argument("--layer-ids", type=str, default=None, help="显式指定源层号，逗号分隔，例 0,3,7")
    parser.add_argument("-e", "--num-experts", type=int, default=0, help="每层保留 expert 数；默认 0 表示全部")
    parser.add_argument("--no-mtp", action="store_true", help="不保留 mtp.* 权重，并把 config.mtp_num_hidden_layers 置 0")
    parser.add_argument("--keep-quant", action="store_true", help="保留源量化 tensor 与 quantization_config；默认输出 BF16")
    parser.add_argument("--dry-run", action="store_true", help="只统计筛选结果，不读取/写入 safetensors")
    args = parser.parse_args()

    src_dir = normalize_path(args.src)
    dst_dir = normalize_path(args.dst)

    with open(src_dir / "config.json", encoding="utf-8") as f:
        src_config = json.load(f)
    src_num_layers = int(src_config["num_hidden_layers"])
    keep_layers = resolve_keep_layers(
        args.num_layers,
        args.layer_range,
        args.layer_ids,
        src_num_layers,
    )

    if args.num_experts == 0:
        total_experts = int(src_config.get("num_experts", 0))
        topk = int(src_config.get("num_experts_per_tok", 0))
        print(
            f"⚠ --num-experts=0 将保留全部 experts ({total_experts}); "
            "Qwen3.8 即使只截取少量层也会很大。"
        )
        print(
            "  加载 Qwen3.8 full-expert mini 时，Transformers 可能对 512 experts "
            "执行 torch.stack，单个投影会产生约 16GiB 临时张量并触发 GPU OOM。"
        )
        print(
            f"  若只是验证 block-wise 与 channel-wise 链路，建议使用 -e 8 "
            f"(num_experts_per_tok 会从 {topk} 自动裁剪到 <= 8)。"
        )

    build_mini_model(
        src_dir=src_dir,
        dst_dir=dst_dir,
        keep_layers=keep_layers,
        num_experts=args.num_experts,
        src_format=args.src_format,
        keep_quant=args.keep_quant,
        include_mtp=not args.no_mtp,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
