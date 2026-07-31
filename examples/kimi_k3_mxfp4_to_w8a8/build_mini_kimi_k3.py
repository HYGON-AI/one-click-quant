#!/usr/bin/env python3
"""
从 Kimi K3 量化输出（或原始 MXFP4）中提取子集，构建 mini 模型用于推理验证。

所用权重统一转为 BF16（FP8/MXFP4 先解量化），可直接用 from_pretrained 加载。

用法:
  # 默认: 4 层 4 experts（覆盖全部结构变体）
  python build_mini_kimi_k3.py

  # 自定义层数和专家数
  python build_mini_kimi_k3.py --num-layers 4 --num-experts 4

  # 从原始 MXFP4 构建
  python build_mini_kimi_k3.py --src-format mxfp4

层结构参考（0-indexed）:
  Layer 0: dense MLP + KDA
  Layer 1: MoE     + KDA   ← 第一个 MoE 层
  Layer 2: MoE     + KDA
  Layer 3: MoE     + MLA   ← 第一个 MLA 层
"""

import argparse
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file

# ─── MXFP4 解包工具（从 compressed-tensors 移植）───

def unpack_fp4_from_uint8(packed: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """NVFP4/MXFP4 packed uint8 → float32 (E2M1 编码)"""
    lut = torch.zeros(16, dtype=torch.float32)
    for i in range(16):
        sign = -1.0 if (i & 0x8) else 1.0
        exp = (i >> 1) & 0x3
        mant = i & 0x1
        if exp == 0:
            val = sign * mant * 0.5
        elif exp == 3:
            val = sign * (1.0 + mant * 0.5) * 2.0 if mant == 0 else float("nan")
        else:
            val = sign * (1.0 + mant * 0.5) * (2.0 ** (exp - 1.0))
        lut[i] = 0.0 if math.isnan(val) else val

    flat = packed.flatten()
    lo = flat & 0xF
    hi = (flat >> 4) & 0xF
    unpacked = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(-1)[: m * n]
    return unpacked.reshape(m, n)


def decompress_mx_scale(scale_uint8: torch.Tensor) -> torch.Tensor:
    """MX 格式 scale: uint8 → E8M0 → float32"""
    return 2.0 ** (scale_uint8.to(torch.float32) - 127.0)

# ─── 张量名匹配 ───

def should_keep_tensor(name: str, num_layers: int, num_experts: int) -> bool:
    """
    判断张量是否应纳入 mini 模型。
    保留: lm_head, embed_tokens, norm, output_attn_res_*, layers.0 .. layers.(num_layers-1)
    专家仅保留 0..num_experts-1
    """
    # 排除视觉模块
    if name.startswith(("vision_tower.", "mm_projector.")):
        return False

    # Expert 过滤（num_experts <= 0 表示保留全部）
    if num_experts > 0:
        expert_match = re.match(
            r"(language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.)(\d+)(\..*)",
            name,
        )
        if expert_match:
            expert_idx = int(expert_match.group(2))
            if expert_idx >= num_experts:
                return False

    # Layer 过滤: 仅保留 layer 0 .. num_layers-1
    layer_match = re.match(
        r"language_model\.model\.layers\.(\d+)\.",
        name,
    )
    if layer_match:
        layer_idx = int(layer_match.group(1))
        if layer_idx >= num_layers:
            return False

    # 全局张量: lm_head, embed_tokens, norm, output_attn_res_*
    if name.startswith((
        "language_model.lm_head.",
        "language_model.model.embed_tokens.",
        "language_model.model.norm.",
        "language_model.model.output_attn_res_",
    )):
        return True

    # 已经通过 layer filter 的层内张量
    if layer_match:
        return True

    return False


def needs_dequant(weight_tensor: torch.Tensor) -> bool:
    """判断权重是否为 FP8（需要解量化）"""
    return weight_tensor.dtype in (
        torch.float8_e4m3fn,
        getattr(torch, "float8_e4m3fnuz", None),
    )

# ─── 权重解量化 ───

def dequant_weight(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Per-channel FP8 → BF16"""
    scale = weight_scale.to(torch.float32)
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1)
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


def dequant_mxfp4_packed(
    name: str,
    tensors: dict[str, torch.Tensor],
    group_size: int = 32,
) -> torch.Tensor | None:
    """MXFP4 packed → BF16。返回 None 如果不是 packed 格式。"""
    base = name.replace(".weight_packed", "")
    scale_name = f"{base}.weight_scale"
    shape_name = f"{base}.weight_shape"

    if name not in tensors or scale_name not in tensors:
        return None

    packed = tensors[name].to(torch.int32)
    scale_uint8 = tensors[scale_name]

    if shape_name in tensors:
        shape = tensors[shape_name]
        m, k = int(shape[0].item()), int(shape[1].item())
    else:
        m = packed.shape[0]
        num_groups = scale_uint8.shape[1]
        k = num_groups * group_size

    unpacked = unpack_fp4_from_uint8(packed, m, k)
    scale_float = decompress_mx_scale(scale_uint8)

    num_groups = k // group_size
    w_fp32 = unpacked.reshape(m, num_groups, group_size).float()
    scale_expanded = scale_float.reshape(m, num_groups, 1)
    w_fp32 = (w_fp32 * scale_expanded).reshape(m, k)

    return w_fp32.to(torch.bfloat16)

# ─── Config 构建 ───

def build_mini_config(
    src_dir: Path, num_layers: int, num_experts: int
) -> dict:
    """生成 mini 模型的 config.json"""
    with open(src_dir / "config.json") as f:
        config = json.load(f)

    tc = config.get("text_config", config)

    # 读取原始 layer 类型信息
    orig_full_attn = set()
    orig_kda = set()
    lac = tc.get("linear_attn_config")
    if lac is not None:
        for idx in lac.get("full_attn_layers", []):
            orig_full_attn.add(idx)
        for idx in lac.get("kda_layers", []):
            orig_kda.add(idx)

    # 筛选到 num_layers 范围内
    mini_kda = sorted([i for i in orig_kda if i <= num_layers])
    mini_full = sorted([i for i in orig_full_attn if i <= num_layers])

    # 修正 config（num_experts <= 0 表示使用原始值）
    actual_experts = num_experts if num_experts > 0 else tc["num_experts"]
    tc["num_hidden_layers"] = num_layers
    tc["num_experts"] = actual_experts
    tc["num_experts_per_token"] = min(2, actual_experts)
    tc["first_k_dense_replace"] = 1  # layer 0 是 dense

    if lac is not None:
        lac["kda_layers"] = mini_kda
        lac["full_attn_layers"] = mini_full

    # 彻底移除量化配置，让模型以纯 BF16 加载
    # （需同时清理 text_config 和顶层，因为 FP8_DYNAMIC 配置可能在顶层）
    tc.pop("quantization_config", None)
    config.pop("quantization_config", None)

    # 保留 vision_config（modeling 代码需要它初始化 VisionTowerConfig），
    # 但移除顶层引用视觉占位符的字段以避免运行时未定义行为
    for key in ("image_placeholder", "media_placeholder_token_id"):
        config.pop(key, None)

    print(f"  kda_layers (1-indexed):  {mini_kda}")
    print(f"  full_attn_layers:       {mini_full}")
    print(f"  num_experts_per_token:  {tc['num_experts_per_token']}")

    return config

# ─── 主流程 ───

def build_mini_model(
    src_dir: Path,
    dst_dir: Path,
    src_format: str,
    num_layers: int,
    num_experts: int,
):
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 从源 config 读取真实 expert 数（避免硬编码）
    with open(src_dir / "config.json") as f:
        src_config = json.load(f)
    src_tc = src_config.get("text_config", src_config)
    total_experts = src_tc.get("num_experts", 0)

    # ── 1. 读取 index ──
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"未找到 index: {index_path}")
    with open(index_path) as f:
        src_index = json.load(f)
    src_weight_map: dict[str, str] = src_index.get("weight_map", {})
    print(f"源模型张量: {len(src_weight_map)}")

    # ── 2. 筛选 ──
    files_to_tensors: dict[str, list[str]] = defaultdict(list)
    kept, skipped = 0, 0
    # 找第一个 MoE 层的编号
    first_moe_layer = 1  # Kimi K3: layer 0 dense, layer 1+ MoE

    for name, fname in src_weight_map.items():
        if should_keep_tensor(name, num_layers, num_experts):
            files_to_tensors[fname].append(name)
            kept += 1
        else:
            skipped += 1

    print(f"保留: {kept}  排除: {skipped}  (layers={num_layers}, experts={num_experts})")

    # ── 3. 逐文件处理 ──
    all_new: dict[str, torch.Tensor] = {}
    new_weight_map: dict[str, str] = {}
    total_size = 0

    # Gate 切片追踪
    gate_jobs = []
    gate_weight_pat = re.compile(
        r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight$"
    )
    e_score_pat = re.compile(
        r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.gate\.e_score_correction_bias$"
    )

    # ── 预收集所有 weight_scale（跨 shard 依赖） ──
    # 量化模型的 weight 和 weight_scale 可能分散在不同 safetensors 文件中，
    # 需要先收集所有 scale，才能在处理 weight 时完成解量化。
    scale_cache: dict[str, torch.Tensor] = {}
    scale_files = set()
    for fname, tensor_names in files_to_tensors.items():
        for name in tensor_names:
            if name.endswith((".weight_scale", ".weight_shape")):
                scale_files.add(fname)
    for fname in sorted(scale_files):
        local = load_file(str(src_dir / fname))
        for name in list(local.keys()):
            if name.endswith((".weight_scale", ".weight_shape")) and name in files_to_tensors.get(fname, []):
                scale_cache[name] = local[name]
    print(f"  预收集 scale/shape: {len(scale_cache)} 个")

    for fname, tensor_names in tqdm(
        sorted(files_to_tensors.items()), desc="处理 safetensors"
    ):
        fpath = src_dir / fname
        if not fpath.exists():
            print(f"  ⚠ 跳过不存在的文件: {fpath}")
            continue

        tensors = load_file(str(fpath))

        for name in tensor_names:
            tensor = tensors[name]

            # ── Gate weight 切片（num_experts <= 0 跳过） ──
            gm = gate_weight_pat.match(name)
            if gm and num_experts > 0:
                tensor = tensor[:num_experts, :].contiguous()
                gate_jobs.append(f"gate.weight (layer {gm.group(1)})")

            # ── e_score_correction_bias 切片 ──
            em = e_score_pat.match(name)
            if em and num_experts > 0:
                tensor = tensor[:num_experts].contiguous()
                gate_jobs.append(f"e_score_correction_bias (layer {em.group(1)})")

            # ── MXFP4 packed → BF16 ──
            if name.endswith(".weight_packed") and src_format == "mxfp4":
                deq = dequant_mxfp4_packed(name, tensors)
                if deq is not None:
                    base = name.replace(".weight_packed", "")
                    key = f"{base}.weight"
                    all_new[key] = deq
                    total_size += deq.numel() * deq.element_size()
                    new_weight_map[key] = "model.safetensors"
                    continue

            # ── FP8 weight → BF16 ──
            if name.endswith(".weight") and needs_dequant(tensor):
                scale_name = name.replace(".weight", ".weight_scale")
                # 尝试当前文件，找不到则查跨 shard 缓存
                scale = tensors.get(scale_name)
                if scale is None:
                    scale = scale_cache.get(scale_name)
                if scale is not None:
                    deq = dequant_weight(tensor, scale)
                    all_new[name] = deq
                    total_size += deq.numel() * deq.element_size()
                    new_weight_map[name] = "model.safetensors"
                    continue

            # ── 跳过 weight_scale / weight_shape（已在解量化中消耗） ──
            if name.endswith(".weight_scale") or name.endswith(".weight_shape"):
                continue

            # ── 普通张量 ──
            all_new[name] = tensor
            total_size += tensor.numel() * tensor.element_size()
            new_weight_map[name] = "model.safetensors"

    if gate_jobs:
        print(f"Gate (截取 {num_experts}/{total_experts} 行): {len(gate_jobs)} 个张量")
    elif num_experts <= 0:
        print(f"Gate (保留全部 {total_experts} 行，无需切片)")
    else:
        print("  ⚠ 未找到 Gate 张量（可能模型中无 MoE 层或层号不匹配）")

    # ── 4. 保存 ──
    print(f"\n保存 safetensors (~{total_size / 1e9:.1f} GB, {len(all_new)} tensors)...")
    save_file(all_new, dst_dir / "model.safetensors")

    with open(dst_dir / "model.safetensors.index.json", "w") as f:
        json.dump({
            "metadata": {"total_size": total_size, "num_tensors": len(new_weight_map)},
            "weight_map": new_weight_map,
        }, f, indent=2)

    # ── 5. Config ──
    config = build_mini_config(src_dir, num_layers, num_experts)
    with open(dst_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── 6. 复制附属文件 ──
    for pyf in src_dir.glob("*.py"):
        shutil.copy(pyf, dst_dir / pyf.name)
    for fn in ["tokenizer_config.json", "preprocessor_config.json",
               "generation_config.json", "tiktoken.model"]:
        p = src_dir / fn
        if p.exists():
            shutil.copy(p, dst_dir / fn)

    print(f"\n✅ Mini 模型已构建: {dst_dir}")

# ─── CLI ───

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建 Kimi K3 mini 模型（用于推理验证）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认 4 层 4 experts（覆盖全部结构: dense+KDA, MoE+KDA, MoE+MLA）
  python build_mini_kimi_k3.py

  # 从原始 MXFP4 构建
  python build_mini_kimi_k3.py --src-format mxfp4
        """,
    )
    parser.add_argument("--src", type=Path,
                        default=Path("/mnt/c/chl/models/Kimi-K3-FP8-DYNAMIC"),
                        help="源模型目录")
    parser.add_argument("--dst", type=Path,
                        default=Path("/mnt/c/chl/models/Kimi-K3-Mini"),
                        help="输出目录")
    parser.add_argument("--src-format", choices=["fp8", "mxfp4"], default="fp8",
                        help="源模型格式")
    parser.add_argument("-n", "--num-layers", type=int, default=4,
                        help="保留前 N 层 (默认 4, 覆盖全部结构变体)")
    parser.add_argument("-e", "--num-experts", type=int, default=0,
                        help="每层保留的 expert 数 (默认 0 保留全部)")
    args = parser.parse_args()

    build_mini_model(args.src, args.dst, args.src_format,
                     args.num_layers, args.num_experts)
