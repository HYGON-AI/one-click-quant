#!/usr/bin/env python3
"""
从 GLM-5.3 FP8 block / W4A16 / W8A16 checkpoint 中提取子集，构建 mini 模型。

默认输出纯 BF16 checkpoint，用于验证 GLM-5.3 的加载、推理和后续量化链路。
本脚本仅支持标准 GLM-5.3 的 model.layers.N.* 张量命名，不处理 Flash 版本。

用法：
  python build_mini_glm5_3.py \
    --src C:\\chl\\models\\GLM-5.3 \
    --dst C:\\chl\\models\\GLM-5.3-Mini \
    -n 4 -e 8

  python build_mini_glm5_3.py --src C:\\chl\\models\\GLM-5.3 --dry-run
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

_LAYER_REMAP: dict[int, int] = {}


def normalize_path(path: str | Path) -> Path:
    s = str(path)
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if match:
        windows_path = Path(f"{match.group(1)}:/{match.group(2).replace(chr(92), '/')}")
        if windows_path.exists() or os.name == "nt":
            return windows_path
        return Path(f"/mnt/{match.group(1).lower()}/{match.group(2).replace(chr(92), '/')}")
    return Path(s)


def remap_layer_name(name: str) -> str:
    match = re.match(r"(model\.layers\.)(\d+)(\..*)", name)
    if match is None:
        return name
    original = int(match.group(2))
    if original not in _LAYER_REMAP:
        return name
    return f"{match.group(1)}{_LAYER_REMAP[original]}{match.group(3)}"


def layer_index(name: str) -> int | None:
    match = re.match(r"model\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def should_keep_tensor(
    name: str,
    keep_layers: set[int],
    num_experts: int,
    include_nextn: bool,
    src_num_layers: int,
) -> bool:
    if name.startswith(("lm_head.", "model.embed_tokens.", "model.norm.")):
        return True

    current_layer = layer_index(name)
    if current_layer is None:
        return False

    if current_layer >= src_num_layers and not include_nextn:
        return False

    expert_match = re.match(r"model\.layers\.\d+\.mlp\.experts\.(\d+)\.", name)
    if expert_match and num_experts > 0 and int(expert_match.group(1)) >= num_experts:
        return False

    return current_layer in keep_layers


def needs_dequant(weight) -> bool:
    import torch

    fp8_dtypes = [torch.float8_e4m3fn]
    for dtype_name in ("float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
        if hasattr(torch, dtype_name):
            fp8_dtypes.append(getattr(torch, dtype_name))
    return weight.dtype in tuple(fp8_dtypes)


def dequant_fp8_block(weight, weight_scale_inv, block_size: tuple[int, int] = (128, 128), dtype=None):
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
        padded = torch.zeros((padded_rows, padded_cols), dtype=weight.dtype, device=weight.device)
        padded[:rows, :cols] = weight
        weight = padded

    row_blocks = padded_rows // block_rows
    col_blocks = padded_cols // block_cols
    expected_shape = (row_blocks, col_blocks)
    if tuple(weight_scale_inv.shape) != expected_shape:
        raise ValueError(
            f"weight_scale_inv shape mismatch: expected={expected_shape}, "
            f"got={tuple(weight_scale_inv.shape)}"
        )

    blocks = weight.reshape(row_blocks, block_rows, col_blocks, block_cols).transpose(1, 2)
    scale = weight_scale_inv.to(torch.float32).unsqueeze(-1).unsqueeze(-1)
    dequant = (blocks.to(torch.float32) * scale).to(dtype)
    return dequant.transpose(1, 2).reshape(padded_rows, padded_cols)[:rows, :cols].contiguous()


def dequant_fp8_channel(weight, weight_scale, dtype=None):
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    scale = weight_scale.to(torch.float32)
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1)
    return (weight.to(torch.float32) * scale).to(dtype).contiguous()


def unpack_int4_from_int32(packed, rows: int, cols: int):
    import torch

    if packed.dtype != torch.int32:
        packed = packed.to(torch.int32)
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    values = ((packed.reshape(-1).unsqueeze(-1) >> shifts) & 0xF).to(torch.int8) - 8
    return values.reshape(rows, -1)[:, :cols]


def unpack_int8_from_int32(packed, rows: int, cols: int):
    import torch

    if packed.dtype != torch.int32:
        packed = packed.to(torch.int32)
    shifts = torch.arange(0, 32, 8, device=packed.device, dtype=torch.int32)
    # compressed_tensors uint8b128：pack 前 +128 存为无符号 byte，还原走 -128（同 int4 的 -8）。
    values = ((packed.reshape(-1).unsqueeze(-1) >> shifts) & 0xFF).to(torch.int16) - 128
    return values.to(torch.int8).reshape(rows, -1)[:, :cols]


def dequant_w8a16_packed(name: str, tensors: dict):
    import torch

    base = name.removesuffix(".weight_packed")
    scale = tensors.get(f"{base}.weight_scale")
    shape = tensors.get(f"{base}.weight_shape")
    if scale is None or shape is None:
        raise ValueError(f"W8A16 tensor {name} 缺少 weight_scale 或 weight_shape")
    packed = tensors[name]
    if packed.dtype != torch.int32 or shape.numel() != 2:
        raise ValueError(f"非法 W8A16 tensor: {name}")

    rows, cols = int(shape[0].item()), int(shape[1].item())
    if packed.shape[0] != rows or scale.dim() != 2 or scale.shape[0] != rows:
        raise ValueError(f"W8A16 shape mismatch: {name}")
    groups = scale.shape[1]
    if cols % groups != 0:
        raise ValueError(f"in_features={cols} 不能被 groups={groups} 整除: {name}")
    group_size = cols // groups
    values = unpack_int8_from_int32(packed, rows, cols).to(torch.float32)
    output = values.reshape(rows, groups, group_size) * scale.to(torch.float32).unsqueeze(-1)
    return output.reshape(rows, cols).to(torch.bfloat16).contiguous()


def dequant_w4a16_packed(name: str, tensors: dict):
    import torch

    base = name.removesuffix(".weight_packed")
    scale = tensors.get(f"{base}.weight_scale")
    shape = tensors.get(f"{base}.weight_shape")
    if scale is None or shape is None:
        raise ValueError(f"W4A16 tensor {name} 缺少 weight_scale 或 weight_shape")
    packed = tensors[name]
    if packed.dtype != torch.int32 or shape.numel() != 2:
        raise ValueError(f"非法 W4A16 tensor: {name}")

    rows, cols = int(shape[0].item()), int(shape[1].item())
    if packed.shape[0] != rows or scale.dim() != 2 or scale.shape[0] != rows:
        raise ValueError(f"W4A16 shape mismatch: {name}")
    groups = scale.shape[1]
    if cols % groups != 0:
        raise ValueError(f"in_features={cols} 不能被 groups={groups} 整除: {name}")
    group_size = cols // groups
    values = unpack_int4_from_int32(packed, rows, cols).to(torch.float32)
    output = values.reshape(rows, groups, group_size) * scale.to(torch.float32).unsqueeze(-1)
    return output.reshape(rows, cols).to(torch.bfloat16).contiguous()


def is_aux_quant_tensor(name: str) -> bool:
    return name.endswith((
        ".weight_scale_inv",
        ".weight_scale",
        ".weight_shape",
        ".weight_packed",
        ".weight_zero_point",
    ))


def maybe_slice_gate(name: str, tensor, num_experts: int):
    if num_experts <= 0:
        return tensor
    if re.match(r"model\.layers\.\d+\.mlp\.gate\.weight$", name):
        if tensor.dim() >= 1 and tensor.shape[0] >= num_experts:
            return tensor[:num_experts].contiguous()
    if re.match(r"model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$", name):
        if tensor.dim() == 1 and tensor.shape[0] >= num_experts:
            return tensor[:num_experts].contiguous()
    return tensor


def detect_src_format(src_format: str, weight_map: dict[str, str], src_dir: Path) -> str:
    if src_format in {"fp8", "fp8-block", "w4a16", "w8a16", "w8a8", "bf16"}:
        return src_format
    if src_format not in {"auto", "fp8"}:
        raise ValueError(f"不支持的 src_format: {src_format}")

    packed_bases = {name.removesuffix(".weight_packed") for name in weight_map if name.endswith(".weight_packed")}
    for base in packed_bases:
        scale_name = f"{base}.weight_scale"
        shape_name = f"{base}.weight_shape"
        if scale_name not in weight_map or shape_name not in weight_map:
            continue
        shard_file = weight_map[base + ".weight_packed"]
        if shard_file != weight_map.get(scale_name) or shard_file != weight_map.get(shape_name):
            continue
        # W4/W8 packed 权重都使用 int32；根据 packed 元素数与原始 shape 判断每个
        # int32 中封装的是 8 个 int4 还是 4 个 int8，避免 W8 被误判为 W4。
        import torch
        from safetensors import safe_open
        with safe_open(str(src_dir / shard_file), framework="pt", device="cpu") as f:
            packed = f.get_tensor(base + ".weight_packed")
            shape = f.get_tensor(shape_name)
        if packed.dtype != torch.int32 or shape.numel() != 2:
            continue
        rows, cols = (int(v) for v in shape.reshape(-1)[:2].tolist())
        packed_values = packed.numel() * 4
        if packed_values == rows * cols:
            return "w8a16"
        if packed_values * 2 == rows * cols:
            return "w4a16"
        raise ValueError(
            f"无法识别 packed 权重位宽: {base}, packed={tuple(packed.shape)}, "
            f"shape=({rows}, {cols})"
        )
    # W8A8 int-quantized：weight(int8) + weight_scale，无 weight_packed / weight_scale_inv。
    # 找第一对同 shard 的 (.weight, .weight_scale)，看 weight 是否 int8，就能与 fp8-channel 区分
    # （fp8-channel 的 weight 是 fp8_e4m3fn，走 --src-format fp8 手动指定，本函数保持原状不自动识别）。
    scale_bases = sorted({n.removesuffix(".weight_scale") for n in weight_map if n.endswith(".weight_scale")})
    for base in scale_bases:
        wname = f"{base}.weight"
        scale_name = f"{base}.weight_scale"
        if wname not in weight_map or weight_map[wname] != weight_map[scale_name]:
            continue
        import torch
        from safetensors import safe_open
        with safe_open(str(src_dir / weight_map[wname]), framework="pt", device="cpu") as f:
            if f.get_tensor(wname).dtype == torch.int8:
                return "w8a8"
        # 第一对匹配的 weight 不是 int8（多半是 fp8_e4m3fn），不再遍历，交给下面兜底或手动 --src-format。
        break
    if any(name.endswith(".weight_scale_inv") for name in weight_map):
        return "fp8-block"
    return "bf16"


def remap_layer_lists(config: dict, keep_layers: list[int], src_num_layers: int) -> None:
    for key, value in list(config.items()):
        if not isinstance(value, list) or len(value) < src_num_layers:
            continue
        if key in {"eos_token_id", "pad_token_id"}:
            continue
        config[key] = [value[index] for index in keep_layers]


def filter_modules_to_not_convert(modules: list[str], keep_layers: set[int]) -> list[str]:
    filtered = []
    for module in modules:
        match = re.match(r"model\.layers\.(\d+)\.(.*)", module)
        if match:
            original = int(match.group(1))
            if original not in keep_layers:
                continue
            filtered.append(f"model.layers.{_LAYER_REMAP[original]}.{match.group(2)}")
        else:
            filtered.append(module)
    return filtered


def build_mini_config(src_dir: Path, keep_layers: list[int], num_experts: int, keep_quant: bool, include_nextn: bool) -> dict:
    with open(src_dir / "config.json", encoding="utf-8") as file:
        config = json.load(file)

    src_num_layers = int(config["num_hidden_layers"])
    remap_layer_lists(config, keep_layers, src_num_layers)
    total_experts = int(config.get("n_routed_experts", 0))
    actual_experts = num_experts if num_experts > 0 else total_experts

    config["num_hidden_layers"] = len(keep_layers)
    config["first_k_dense_replace"] = sum(
        1 for layer_index_value in keep_layers if layer_index_value < int(config.get("first_k_dense_replace", 0))
    )
    if actual_experts > 0:
        config["n_routed_experts"] = actual_experts
        if "num_experts_per_tok" in config:
            config["num_experts_per_tok"] = min(int(config["num_experts_per_tok"]), actual_experts)
    if "topk_group" in config and "n_group" in config:
        config["topk_group"] = min(int(config["topk_group"]), int(config["n_group"]))

    if not include_nextn:
        config["num_nextn_predict_layers"] = 0
    if keep_quant:
        qconfig = config.get("quantization_config")
        if isinstance(qconfig, dict) and isinstance(qconfig.get("modules_to_not_convert"), list):
            qconfig["modules_to_not_convert"] = filter_modules_to_not_convert(
                qconfig["modules_to_not_convert"], set(keep_layers)
            )
    else:
        config.pop("quantization_config", None)
    return config


def resolve_keep_layers(num_layers: int | None, layer_range: str | None, layer_ids: str | None, src_num_layers: int) -> list[int]:
    if sum(value is not None for value in (num_layers, layer_range, layer_ids)) > 1:
        raise ValueError("--num-layers / --layer-range / --layer-ids 三个参数互斥")
    if layer_ids is not None:
        ids = [int(value) for value in layer_ids.split(",") if value.strip()]
    elif layer_range is not None:
        start, end = layer_range.split(":", 1)
        ids = list(range(int(start), int(end)))
    else:
        ids = list(range(num_layers if num_layers is not None else 4))
    if not ids:
        raise ValueError("keep_layers 不能为空")
    if any(index < 0 or index >= src_num_layers for index in ids):
        raise ValueError(f"层号越界；源模型共 {src_num_layers} 层")
    return sorted(set(ids))


def build_mini_model(src_dir: Path, dst_dir: Path, keep_layers: list[int], num_experts: int, src_format: str, keep_quant: bool, include_nextn: bool, dry_run: bool):
    global _LAYER_REMAP
    _LAYER_REMAP = {original: new for new, original in enumerate(keep_layers)}

    with open(src_dir / "config.json", encoding="utf-8") as file:
        src_config = json.load(file)
    src_num_layers = int(src_config["num_hidden_layers"])
    total_experts = int(src_config.get("n_routed_experts", 0))
    if num_experts > 0 and total_experts > 0 and num_experts > total_experts:
        raise ValueError(f"--num-experts={num_experts} 超过源模型 n_routed_experts={total_experts}")

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"未找到 index: {index_path}")
    with open(index_path, encoding="utf-8") as file:
        weight_map = json.load(file).get("weight_map", {})
    resolved_format = detect_src_format(src_format, weight_map, src_dir)

    files_to_tensors: dict[str, list[str]] = defaultdict(list)
    kept = 0
    for name, filename in weight_map.items():
        if should_keep_tensor(name, set(keep_layers), num_experts, include_nextn, src_num_layers):
            files_to_tensors[filename].append(name)
            kept += 1
    skipped = len(weight_map) - kept
    print(f"源模型张量: {len(weight_map)}")
    print(f"源格式: {src_format} -> {resolved_format}")
    print(f"保留层: {keep_layers}; experts: {num_experts if num_experts > 0 else total_experts}")
    print(f"保留: {kept}  排除: {skipped}")
    if kept == 0:
        raise RuntimeError("未筛选到任何 tensor，请检查源模型命名或参数")
    if dry_run:
        print("dry-run: 仅完成 index 筛选，不读取/写入 safetensors。")
        return

    import torch
    from safetensors.torch import load_file, save_file
    from tqdm import tqdm

    dst_dir.mkdir(parents=True, exist_ok=True)
    needed_qparams = set()
    if not keep_quant:
        for names in files_to_tensors.values():
            for name in names:
                if name.endswith(".weight"):
                    for suffix in ("_scale_inv", "_scale"):
                        candidate = f"{name}{suffix}"
                        if candidate in weight_map:
                            needed_qparams.add(candidate)
                elif name.endswith(".weight_packed"):
                    base = name.removesuffix(".weight_packed")
                    for suffix in (".weight_scale", ".weight_shape", ".weight_zero_point"):
                        candidate = f"{base}{suffix}"
                        if candidate in weight_map:
                            needed_qparams.add(candidate)

    qparam_cache = {}
    for filename in tqdm(sorted({weight_map[name] for name in needed_qparams}), desc="预收集 qparam"):
        local = load_file(str(src_dir / filename))
        for name in local:
            if name in needed_qparams:
                qparam_cache[name] = local[name]

    all_new = {}
    new_weight_map = {}
    total_size = 0
    for filename, names in tqdm(sorted(files_to_tensors.items()), desc="处理 safetensors"):
        tensors = load_file(str(src_dir / filename))
        lookup = dict(qparam_cache)
        lookup.update(tensors)
        for name in names:
            tensor = tensors[name]
            if keep_quant:
                tensor = maybe_slice_gate(name, tensor, num_experts)
                key = remap_layer_name(name)
            elif resolved_format == "fp8-block" and name.endswith(".weight") and needs_dequant(tensor):
                scale = lookup.get(f"{name}_scale_inv")
                if scale is None:
                    raise ValueError(f"缺少 {name}_scale_inv，无法解量化 {name}")
                tensor = dequant_fp8_block(tensor, scale)
                key = remap_layer_name(name)
            elif resolved_format == "w4a16" and name.endswith(".weight_packed"):
                tensor = dequant_w4a16_packed(name, lookup)
                key = remap_layer_name(f"{name.removesuffix('.weight_packed')}.weight")
            elif resolved_format == "w8a16" and name.endswith(".weight_packed"):
                tensor = dequant_w8a16_packed(name, lookup)
                key = remap_layer_name(f"{name.removesuffix('.weight_packed')}.weight")
            elif resolved_format == "w8a8" and name.endswith(".weight") and tensor.dtype == torch.int8:
                # W8A8 int-quantized：weight 已是有符号 int8 (MoE-Quant 里 qweight - 128)。
                # int8 → f32 是 sign-preserving，dequant_fp8_channel 的 (weight.to(f32) * scale)
                # 逻辑对 int8 同样成立，直接复用。
                scale = lookup.get(f"{name}_scale")
                if scale is None:
                    raise ValueError(f"缺少 {name}_scale，无法解量化 W8A8 {name}")
                tensor = dequant_fp8_channel(tensor, scale)
                key = remap_layer_name(name)
            elif is_aux_quant_tensor(name):
                continue
            elif resolved_format == "fp8" and name.endswith(".weight") and needs_dequant(tensor):
                scale = lookup.get(f"{name}_scale")
                if scale is None:
                    raise ValueError(f"缺少 {name}_scale，无法解量化 {name}")
                tensor = dequant_fp8_channel(tensor, scale)
                key = remap_layer_name(name)
            else:
                tensor = maybe_slice_gate(name, tensor, num_experts)
                key = remap_layer_name(name)

            all_new[key] = tensor
            total_size += tensor.numel() * tensor.element_size()
            new_weight_map[key] = "model.safetensors"

    save_file(all_new, dst_dir / "model.safetensors")
    with open(dst_dir / "model.safetensors.index.json", "w", encoding="utf-8") as file:
        json.dump({"metadata": {"total_size": total_size, "num_tensors": len(new_weight_map)}, "weight_map": new_weight_map}, file, indent=2)
    with open(dst_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(build_mini_config(src_dir, keep_layers, num_experts, keep_quant, include_nextn), file, indent=2, ensure_ascii=False)

    skip_names = {"config.json", "model.safetensors.index.json"}
    for path in src_dir.iterdir():
        if path.name not in skip_names and path.suffix != ".safetensors" and path.is_file():
            shutil.copy(path, dst_dir / path.name)
    print(f"Mini 模型已构建: {dst_dir} ({total_size / 1e9:.1f} GB, {len(all_new)} tensors)")


def main():
    parser = argparse.ArgumentParser(description="构建标准 GLM-5.3 mini 模型（FP8 block/W4A16/W8A16/W8A8 -> BF16）")
    parser.add_argument("--src", type=str, default="/models/GLM-5.3", help="源模型目录")
    parser.add_argument("--dst", type=str, default="/models/GLM-5.3-Mini", help="输出目录")
    parser.add_argument("--src-format", choices=["auto", "fp8", "fp8-block", "w4a16", "w8a16", "w8a8", "bf16"], default="auto")
    parser.add_argument("-n", "--num-layers", type=int, default=None, help="保留前 N 层；默认 4")
    parser.add_argument("--layer-range", type=str, default=None, help="截取层区间 START:END")
    parser.add_argument("--layer-ids", type=str, default=None, help="显式指定源层号，逗号分隔")
    parser.add_argument("-e", "--num-experts", type=int, default=0, help="每层保留 expert 数；默认 0 表示全部")
    parser.add_argument("--include-nextn", action="store_true", help="保留源模型主干层之外的 nextn 权重")
    parser.add_argument("--keep-quant", action="store_true", help="保留源量化张量与 quantization_config")
    parser.add_argument("--dry-run", action="store_true", help="只统计筛选结果，不读取/写入 safetensors")
    args = parser.parse_args()

    src_dir = normalize_path(args.src)
    dst_dir = normalize_path(args.dst)
    with open(src_dir / "config.json", encoding="utf-8") as file:
        src_num_layers = int(json.load(file)["num_hidden_layers"])
    keep_layers = resolve_keep_layers(args.num_layers, args.layer_range, args.layer_ids, src_num_layers)
    build_mini_model(src_dir, dst_dir, keep_layers, args.num_experts, args.src_format, args.keep_quant, args.include_nextn, args.dry_run)


if __name__ == "__main__":
    main()
