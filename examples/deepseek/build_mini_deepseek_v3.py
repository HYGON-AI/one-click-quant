#!/usr/bin/env python3
"""
从 DeepSeek-V3 FP8 / W4A16 checkpoint 中提取子集，构建 mini 模型用于推理/量化链路验证。

特点：
  - 适配 DeepSeek-V3 张量命名：model.layers.N.* / model.layers.N.mlp.experts.E.*
  - 支持 DeepSeek-V3 FP8 block-wise / W4A16：weight + weight_scale_inv 或 weight_packed + weight_shape + weight_scale → BF16
  - 支持 MoE-Quant pack_quantized_model.py 产出的 compressed-tensors W4A16：
    weight_packed + weight_shape + weight_scale → BF16
  - 支持按层、按 expert 子集裁剪，并把保留层重映射为连续的 model.layers.0..N-1
  - 默认输出纯 BF16 checkpoint，因此会移除 quantization_config
  - 默认不保留 nextn predict 层（源 checkpoint 中的 model.layers.61.*），并把 num_nextn_predict_layers 置 0

用法：
  python3 build_mini_deepseek_v3.py \
    --src /models/DeepSeek-V3 \
    --dst /models/DeepSeek-V3-Mini \
    -n 4 -e 8

建议：
  DeepSeek-V3 前 3 层是 dense MLP，从第 3 层开始是 MoE。若要覆盖 MoE 结构，默认 -n 4 会保留层 0,1,2,3。
  如果只是快速验证，建议加 -e 8 或 -e 16，避免保留全部 256 experts 导致 mini 仍然很大。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path


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
    include_nextn: bool,
    src_num_layers: int,
) -> bool:
    """判断 DeepSeek-V3 tensor 是否应纳入 mini 模型。"""
    # 主模型全局权重。
    if name.startswith((
        "lm_head.",
        "model.embed_tokens.",
        "model.norm.",
    )):
        return True

    # 源 checkpoint 中 num_hidden_layers=61，但 index 还包含 model.layers.61.*，对应 nextn predict 相关权重。
    # 默认不保留，以输出标准 causal LM mini；如显式 --include-nextn，则原样保留。
    nextn_match = re.match(r"model\.layers\.(\d+)\.", name)
    if nextn_match and int(nextn_match.group(1)) >= src_num_layers:
        if not include_nextn:
            return False
        if num_experts > 0:
            expert_match = re.match(r"model\.layers\.\d+\.mlp\.experts\.(\d+)\.", name)
            if expert_match and int(expert_match.group(1)) >= num_experts:
                return False
        return True

    # Experts 过滤；num_experts <= 0 表示保留全部 experts。
    if num_experts > 0:
        expert_match = re.match(r"model\.layers\.\d+\.mlp\.experts\.(\d+)\.", name)
        if expert_match and int(expert_match.group(1)) >= num_experts:
            return False

    # 主模型层过滤。
    layer_match = re.match(r"model\.layers\.(\d+)\.", name)
    if layer_match:
        return int(layer_match.group(1)) in keep_layers

    return False


def needs_dequant(weight) -> bool:
    """判断权重是否为 FP8 dtype。"""
    import torch

    fp8_dtypes = [torch.float8_e4m3fn]
    if hasattr(torch, "float8_e4m3fnuz"):
        fp8_dtypes.append(torch.float8_e4m3fnuz)
    if hasattr(torch, "float8_e5m2"):
        fp8_dtypes.append(torch.float8_e5m2)
    if hasattr(torch, "float8_e5m2fnuz"):
        fp8_dtypes.append(torch.float8_e5m2fnuz)
    return weight.dtype in tuple(fp8_dtypes)


def dequant_fp8_block(weight, weight_scale_inv, block_size: tuple[int, int] = (128, 128), dtype=None):
    """DeepSeek/HF FP8 block 权重解量化：weight + weight_scale_inv → BF16。"""
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

    num_row_blocks = padded_rows // block_rows
    num_col_blocks = padded_cols // block_cols
    expected_scale_shape = (num_row_blocks, num_col_blocks)
    if tuple(weight_scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"weight_scale_inv shape mismatch: expected={expected_scale_shape}, "
            f"got={tuple(weight_scale_inv.shape)}"
        )

    blocks = weight.reshape(num_row_blocks, block_rows, num_col_blocks, block_cols).transpose(1, 2)
    scale = weight_scale_inv.to(torch.float32).unsqueeze(-1).unsqueeze(-1)
    dequant = (blocks.to(torch.float32) * scale).to(dtype)
    dequant = dequant.transpose(1, 2).reshape(padded_rows, padded_cols)
    return dequant[:rows, :cols].contiguous()


def unpack_int4_from_int32(packed, rows: int, cols: int):
    """compressed-tensors int32-packed INT4 → signed int8 (-8..7)。"""
    import torch

    if packed.dtype != torch.int32:
        packed = packed.to(torch.int32)
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    values = ((packed.reshape(-1).unsqueeze(-1) >> shifts) & 0xF).to(torch.int8) - 8
    return values.reshape(rows, -1)[:, :cols]


def dequant_w4a16_packed(name: str, tensors: dict):
    """MoE-Quant compressed-tensors W4A16：weight_packed + scale + shape → BF16。"""
    import torch

    base = name.removesuffix(".weight_packed")
    scale_name = f"{base}.weight_scale"
    shape_name = f"{base}.weight_shape"
    if scale_name not in tensors or shape_name not in tensors:
        raise ValueError(f"W4A16 tensor {name} 缺少 {scale_name} 或 {shape_name}")

    packed = tensors[name]
    scale = tensors[scale_name]
    shape = tensors[shape_name]
    if packed.dtype != torch.int32:
        raise ValueError(f"W4A16 weight_packed 必须为 int32，{name} 实际为 {packed.dtype}")
    if shape.numel() != 2:
        raise ValueError(f"W4A16 weight_shape 必须包含 [out_features, in_features]，{name} 实际为 {tuple(shape.shape)}")

    rows, cols = (int(shape[0].item()), int(shape[1].item()))
    if packed.shape[0] != rows:
        raise ValueError(f"W4A16 out_features 不一致：{name} packed={packed.shape[0]}, shape={rows}")
    if scale.dim() != 2 or scale.shape[0] != rows:
        raise ValueError(f"W4A16 weight_scale shape 非法：{name} 得到 {tuple(scale.shape)}")
    num_groups = scale.shape[1]
    if cols % num_groups != 0:
        raise ValueError(f"W4A16 in_features={cols} 不能被 scale groups={num_groups} 整除：{name}")

    group_size = cols // num_groups
    unpacked = unpack_int4_from_int32(packed, rows, cols).to(torch.float32)
    dequant = (unpacked.reshape(rows, num_groups, group_size) * scale.to(torch.float32).unsqueeze(-1))
    return dequant.reshape(rows, cols).to(torch.bfloat16).contiguous()


def detect_src_format(src_format: str, src_weight_map: dict[str, str]) -> str:
    """把 auto 解析成 w4a16 / fp8-block / bf16。"""
    if src_format in {"fp8-block", "w4a16", "bf16"}:
        return src_format
    if src_format not in {"auto", "fp8"}:
        raise ValueError(f"不支持的 src_format: {src_format}")

    # 优先识别 pack_quantized_model.py 的 W4A16：同一模块必须同时有
    # weight_packed、weight_scale、weight_shape，避免误把其他 packed 格式识别成 W4A16。
    packed_bases = {
        name.removesuffix(".weight_packed")
        for name in src_weight_map
        if name.endswith(".weight_packed")
    }
    if any(
        f"{base}.weight_scale" in src_weight_map
        and f"{base}.weight_shape" in src_weight_map
        for base in packed_bases
    ):
        return "w4a16"
    if any(name.endswith(".weight_scale_inv") for name in src_weight_map):
        return "fp8-block"
    return "bf16"


def qparam_name_for_weight(name: str, resolved_format: str) -> str | None:
    if resolved_format == "fp8-block":
        return name.replace(".weight", ".weight_scale_inv")
    return None


def is_aux_quant_tensor(name: str, resolved_format: str) -> bool:
    if resolved_format == "fp8-block":
        return name.endswith(".weight_scale_inv")
    if resolved_format == "w4a16":
        return name.endswith((".weight_packed", ".weight_scale", ".weight_shape", ".weight_zero_point"))
    return False


def maybe_slice_gate(name: str, tensor, num_experts: int):
    """按 expert 数裁剪 router/gate 的第一维。"""
    if num_experts <= 0:
        return tensor
    if re.match(r"model\.layers\.\d+\.mlp\.gate\.weight$", name):
        if tensor.dim() >= 1 and tensor.shape[0] >= num_experts:
            return tensor[:num_experts].contiguous()
    if re.match(r"model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$", name):
        if tensor.dim() == 1 and tensor.shape[0] >= num_experts:
            return tensor[:num_experts].contiguous()
    return tensor


def choose_group_count(num_experts: int, original_n_group: int) -> int:
    """为裁剪后的专家数选择能整除 n_routed_experts 且每组至少 2 个 experts 的 n_group。"""
    if num_experts <= 0:
        return original_n_group
    # DeepSeek-V3 noaux_tc gate 会在每个 group 内 topk(2)，所以每组至少需要 2 个 experts。
    max_groups = max(1, num_experts // 2)
    n_group = min(original_n_group, num_experts, max_groups)
    while n_group > 1 and num_experts % n_group != 0:
        n_group -= 1
    return max(1, n_group)


def build_mini_config(
    src_dir: Path,
    keep_layers: list[int],
    num_experts: int,
    keep_quant: bool,
    include_nextn: bool,
) -> dict:
    with open(src_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    total_experts = int(config.get("n_routed_experts", 0))
    actual_experts = num_experts if num_experts > 0 else total_experts

    config["num_hidden_layers"] = len(keep_layers)
    original_first_dense = int(config.get("first_k_dense_replace", 0))
    config["first_k_dense_replace"] = sum(1 for layer_idx in keep_layers if layer_idx < original_first_dense)
    if actual_experts > 0:
        config["n_routed_experts"] = actual_experts
        if "num_experts_per_tok" in config:
            config["num_experts_per_tok"] = min(int(config["num_experts_per_tok"]), actual_experts)
        original_n_group = int(config.get("n_group", 1))
        n_group = choose_group_count(actual_experts, original_n_group)
        config["n_group"] = n_group
        if "topk_group" in config:
            config["topk_group"] = min(int(config["topk_group"]), n_group)

    if not include_nextn:
        config["num_nextn_predict_layers"] = 0

    if not keep_quant:
        # 默认输出 BF16，不能保留源 FP8 quantization_config。
        config.pop("quantization_config", None)

    print(f"  keep_layers (源 0-index):    {keep_layers}")
    print(f"  remap (源 → mini):          {_LAYER_REMAP}")
    print(f"  n_routed_experts:           {config.get('n_routed_experts')}")
    print(f"  num_experts_per_tok:        {config.get('num_experts_per_tok')}")
    print(f"  n_group / topk_group:       {config.get('n_group')} / {config.get('topk_group')}")
    print(f"  first_k_dense_replace:      {config.get('first_k_dense_replace')}")
    print(f"  num_nextn_predict_layers:   {config.get('num_nextn_predict_layers')}")
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
            raise ValueError(f"层号 {idx} 越界；源模型主干共 {src_num_layers} 层")
    return sorted(ids)


def build_mini_model(
    src_dir: Path,
    dst_dir: Path,
    keep_layers: list[int],
    num_experts: int,
    src_format: str,
    keep_quant: bool,
    include_nextn: bool,
    dry_run: bool,
):
    global _LAYER_REMAP
    _LAYER_REMAP = {orig: new for new, orig in enumerate(keep_layers)}
    keep_layers_set = set(keep_layers)

    with open(src_dir / "config.json", encoding="utf-8") as f:
        src_config = json.load(f)
    src_num_layers = int(src_config["num_hidden_layers"])
    total_experts = int(src_config.get("n_routed_experts", 0))
    if num_experts > 0 and total_experts > 0 and num_experts > total_experts:
        raise ValueError(f"--num-experts={num_experts} 超过源模型 n_routed_experts={total_experts}")

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"未找到 index: {index_path}")
    with open(index_path, encoding="utf-8") as f:
        src_index = json.load(f)
    src_weight_map: dict[str, str] = src_index.get("weight_map", {})
    resolved_src_format = detect_src_format(src_format, src_weight_map)

    print(f"源模型张量: {len(src_weight_map)}")
    print(f"源格式: {src_format} -> {resolved_src_format}")
    print(f"保留层 (源 0-index): {keep_layers}")

    files_to_tensors: dict[str, list[str]] = defaultdict(list)
    kept, skipped = 0, 0
    for name, fname in src_weight_map.items():
        if should_keep_tensor(name, keep_layers_set, num_experts, include_nextn, src_num_layers):
            files_to_tensors[fname].append(name)
            kept += 1
        else:
            skipped += 1

    print(
        f"保留: {kept}  排除: {skipped}  "
        f"(layers={len(keep_layers)}, experts={num_experts}, include_nextn={include_nextn})"
    )
    if kept == 0:
        raise RuntimeError("未筛选到任何 tensor，请检查源模型命名或筛选参数")
    if dry_run:
        print("dry-run: 仅完成 index 筛选，不读取/写入 safetensors。")
        return

    from safetensors.torch import load_file, save_file
    from tqdm import tqdm

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 量化辅助张量可能跨 shard，先预收集需要的 qparam。
    qparam_cache = {}
    qparam_files = set()
    if resolved_src_format == "fp8-block" and not keep_quant:
        needed_qparams = {
            scale_name
            for tensor_names in files_to_tensors.values()
            for name in tensor_names
            if name.endswith(".weight")
            for scale_name in [qparam_name_for_weight(name, resolved_src_format)]
            if scale_name is not None
        }
    elif resolved_src_format == "w4a16" and not keep_quant:
        needed_qparams = {
            qparam_name
            for tensor_names in files_to_tensors.values()
            for name in tensor_names
            if name.endswith(".weight_packed")
            for base in [name.removesuffix(".weight_packed")]
            for qparam_name in (
                f"{base}.weight_scale",
                f"{base}.weight_shape",
                f"{base}.weight_zero_point",
            )
            if qparam_name in src_weight_map
        }
    else:
        needed_qparams = set()

    for qparam_name in needed_qparams:
        fname = src_weight_map.get(qparam_name)
        if fname is not None:
            qparam_files.add(fname)
    for fname in tqdm(sorted(qparam_files), desc="预收集 qparam"):
        fpath = src_dir / fname
        if not fpath.exists():
            print(f"  ⚠ 跳过不存在的 qparam shard: {fpath}")
            continue
        local = load_file(str(fpath))
        for name in local.keys():
            if name in needed_qparams:
                qparam_cache[name] = local[name]
    print(f"  预收集 qparam: {len(qparam_cache)} 个")

    all_new = {}
    new_weight_map: dict[str, str] = {}
    total_size = 0

    for fname, tensor_names in tqdm(sorted(files_to_tensors.items()), desc="处理 safetensors"):
        fpath = src_dir / fname
        if not fpath.exists():
            print(f"  ⚠ 跳过不存在的文件: {fpath}")
            continue

        tensors = load_file(str(fpath))
        tensors_for_lookup = dict(qparam_cache)
        tensors_for_lookup.update(tensors)
        for name in tensor_names:
            tensor = tensors[name]

    # keep_quant 时直接透传，包括 weight_scale_inv / weight_packed / weight_shape / weight_zero_point。
            if keep_quant:
                key = remap_layer_name(name)
                tensor = maybe_slice_gate(name, tensor, num_experts)
                all_new[key] = tensor
                total_size += tensor.numel() * tensor.element_size()
                new_weight_map[key] = "model.safetensors"
                continue

            # FP8 block 权重 → BF16。只有存在对应 qparam 的 FP8 weight 会进入此分支。
            if resolved_src_format == "fp8-block" and name.endswith(".weight") and needs_dequant(tensor):
                qparam_name = qparam_name_for_weight(name, resolved_src_format)
                scale = tensors_for_lookup.get(qparam_name)
                if scale is None:
                    raise ValueError(f"缺少 {qparam_name}，无法解量化 {name}")
                deq = dequant_fp8_block(tensor, scale)
                key = remap_layer_name(name)
                all_new[key] = deq
                total_size += deq.numel() * deq.element_size()
                new_weight_map[key] = "model.safetensors"
                continue

            # W4A16 packed 权重 → BF16。
            if resolved_src_format == "w4a16" and name.endswith(".weight_packed"):
                deq = dequant_w4a16_packed(name, tensors_for_lookup)
                key = remap_layer_name(f"{name.removesuffix('.weight_packed')}.weight")
                all_new[key] = deq
                total_size += deq.numel() * deq.element_size()
                new_weight_map[key] = "model.safetensors"
                continue

            # 解量化输出时不保存 qparam。
            if is_aux_quant_tensor(name, resolved_src_format):
                continue

            # BF16 / 非 FP8 / 非 W4A16 普通 tensor 透传，router gate 按 expert 数裁剪。
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

    config = build_mini_config(src_dir, keep_layers, num_experts, keep_quant, include_nextn)
    with open(dst_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 复制 tokenizer / remote code 等非权重文件；config 和 index 使用新生成版本。
    skip_names = {"config.json", "model.safetensors.index.json"}
    for p in src_dir.iterdir():
        if p.name in skip_names or p.suffix == ".safetensors":
            continue
        if p.is_file():
            shutil.copy(p, dst_dir / p.name)

    print(f"\n✅ Mini 模型已构建: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="构建 DeepSeek-V3 mini 模型（支持 FP8 block / W4A16 → BF16）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=str, default="/models/DeepSeek-V3", help="源模型目录")
    parser.add_argument("--dst", type=str, default="/models/DeepSeek-V3-Mini", help="输出目录")
    parser.add_argument(
        "--src-format",
        choices=["auto", "fp8", "fp8-block", "w4a16", "bf16"],
        default="auto",
        help="源模型格式；auto 会优先识别 w4a16，再识别 fp8-block / bf16",
    )
    parser.add_argument("-n", "--num-layers", type=int, default=None, help="保留前 N 个主干层；默认 4")
    parser.add_argument("--layer-range", type=str, default=None, help="截取源层区间 START:END，例 0:4")
    parser.add_argument("--layer-ids", type=str, default=None, help="显式指定源层号，逗号分隔，例 0,1,2,3")
    parser.add_argument("-e", "--num-experts", type=int, default=8, help="每个 MoE 层保留 expert 数；默认 8，<=0 表示全部")
    parser.add_argument("--include-nextn", action="store_true", help="保留源 checkpoint 中 model.layers.<num_hidden_layers>.* 的 nextn 权重；默认丢弃并将 num_nextn_predict_layers 置 0")
    parser.add_argument("--keep-quant", action="store_true", help="保留源量化张量与 quantization_config；默认输出 BF16")
    parser.add_argument("--dry-run", action="store_true", help="只统计筛选结果，不读取/写入 safetensors")
    args = parser.parse_args()

    src_dir = normalize_path(args.src)
    dst_dir = normalize_path(args.dst)

    with open(src_dir / "config.json", encoding="utf-8") as f:
        src_config = json.load(f)
    src_num_layers = int(src_config["num_hidden_layers"])
    keep_layers = resolve_keep_layers(args.num_layers, args.layer_range, args.layer_ids, src_num_layers)

    if args.num_experts <= 0:
        total_experts = int(src_config.get("n_routed_experts", 0))
        topk = int(src_config.get("num_experts_per_tok", 0))
        print(
            f"⚠ --num-experts<=0 将保留全部 routed experts ({total_experts}); "
            "DeepSeek-V3 即使只截取少量层也会很大。"
        )
        print(f"  若只是验证链路，建议使用默认 -e 8 或 -e 16 (num_experts_per_tok 会从 {topk} 自动裁剪到 <= expert 数)。")

    build_mini_model(
        src_dir=src_dir,
        dst_dir=dst_dir,
        keep_layers=keep_layers,
        num_experts=args.num_experts,
        src_format=args.src_format,
        keep_quant=args.keep_quant,
        include_nextn=args.include_nextn,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
