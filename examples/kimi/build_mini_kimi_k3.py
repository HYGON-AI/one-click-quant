#!/usr/bin/env python3
"""
从 Kimi K3 量化输出（或原始 MXFP4）中提取子集，构建 mini 模型用于推理验证。

所用权重统一转为 BF16（量化格式先解量化），可直接用 from_pretrained 加载。

支持的源格式:
  fp8     - FP8_DYNAMIC（per-channel weight scale）
  mxfp4   - 原始 MXFP4（uint8 packed, group_size=32, E8M0 scale）
  int4    - INT4 uint8 packed（每 byte 2 个 INT4, +8 偏移）
  w4a16   - compressed-tensors W4A16（int32 packed, group_size=128, float scale）
  bf16    - 纯 BF16（权重已解量化，直接透传）

用法:
  # 默认: 4 层保留全部 experts（覆盖全部结构变体）
  python build_mini_kimi_k3.py

  # 自定义层数和专家数
  python build_mini_kimi_k3.py --num-layers 4 --num-experts 4

  # 从原始 MXFP4 构建
  python build_mini_kimi_k3.py --src-format mxfp4

  # 从 W4A16 量化模型构建
  python build_mini_kimi_k3.py --src /model/Kimi-K3-W4A16 --src-format w4a16

  # 从 BF16 模型构建（无需解量化）
  python build_mini_kimi_k3.py --src /model/Kimi-K3-BF16 --src-format bf16

层结构参考（0-indexed）:
  Layer 0: dense MLP + KDA
  Layer 1: MoE     + KDA   ← 第一个 MoE 层
  Layer 2: MoE     + KDA
  Layer 3: MoE     + MLA   ← 第一个 MLA 层
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file

# ─── MXFP4 解包工具（从 compressed-tensors 移植）───

# E2M1 (MXFP4) 8 个非负值的查表, 参考 compressed_tensors/compressors/nvfp4/helpers.py
# 4-bit 编码: [sign(1) | exp(2) | mantissa(1)], 低 3 位 (exp+mant) 索引这张表, bit3 决定 sign
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def unpack_fp4_from_uint8(packed: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """NVFP4/MXFP4 packed uint8 → float32 (E2M1 编码)

    每 uint8 打包 2 个 E2M1 值: 低 4 位 = 第 1 个, 高 4 位 = 第 2 个。
    每 4 位构成 [sign(1) | exp(2) | mantissa(1)]。
    """
    lut = _E2M1_LUT.to(packed.device)
    flat = packed.flatten().to(torch.int32)

    def decode(nibble: torch.Tensor) -> torch.Tensor:
        abs_idx = (nibble & 0x7).to(torch.long)
        sign = torch.where((nibble & 0x8).bool(), -1.0, 1.0)
        return lut[abs_idx] * sign

    lo = decode(flat & 0xF)
    hi = decode((flat >> 4) & 0xF)
    unpacked = torch.stack([lo, hi], dim=-1).reshape(-1)[: m * n]
    return unpacked.reshape(m, n)


def decompress_mx_scale(scale_uint8: torch.Tensor) -> torch.Tensor:
    """MX 格式 scale: uint8 → E8M0 → float32"""
    return 2.0 ** (scale_uint8.to(torch.float32) - 127.0)

# ─── INT4 解包 ───

def unpack_int4_from_uint8(packed: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """
    INT4 packed uint8 → int8 (-8~7)。
    每字节打包 2 个 INT4 值：偶数索引在低 4 位，奇数索引在高 4 位，
    compressed-tensors 存储时加 8 偏移 (packed_nibble = int4_val + 8)。
    """
    flat = packed.flatten()
    lo = (flat & 0xF).to(torch.int8) - 8
    hi = ((flat >> 4) & 0xF).to(torch.int8) - 8
    combined = torch.stack([lo, hi], dim=-1).reshape(-1)[: m * n]
    return combined.reshape(m, n)


def dequant_int4_packed(
    name: str,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """INT4 packed → BF16。返回 None 如果不是 packed 格式。"""
    base = name.replace(".weight_packed", "")
    scale_name = f"{base}.weight_scale"

    if name not in tensors or scale_name not in tensors:
        return None

    packed = tensors[name]
    scale = tensors[scale_name].to(torch.float32)

    m = packed.shape[0]        # out_features
    n = packed.shape[1] * 2    # in_features (每个 uint8 = 2 个 int4)

    unpacked = unpack_int4_from_uint8(packed, m, n)
    # per-channel: scale shape (out_features,) → (out_features, 1) 广播到 (m, n)
    w_fp32 = unpacked.float() * scale.view(-1, 1)
    return w_fp32.to(torch.bfloat16)


# ─── W4A16 解包（compressed-tensors int32 packed format）───

def unpack_int4_from_int32(packed: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """
    compressed-tensors W4A16 int32-packed INT4 → int8 (-8~7)。

    每个 int32 打包 8 个 INT4 值（32/4=8），采用 dense bit packing：
    - 元素 i 的 4-bit 值位于全局 bit 位置 [i*4, (i+1)*4)
    - 由于 4 整除 32，INT4 值不会跨越 int32 边界
    - 存储偏移 +8（即 signed int4 + 8 → unsigned nibble）
    """
    if packed.dtype != torch.int32:
        packed = packed.to(torch.int32)

    device = packed.device
    total = packed.shape[0] * packed.shape[1]
    flat = packed.reshape(-1)  # (total_ints,)

    # 每个 int32 → 8 个 INT4，位偏移 0,4,8,12,16,20,24,28
    shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
    vals = (flat.unsqueeze(-1) >> shifts.unsqueeze(0)) & 0xF  # (total_ints, 8)

    # 去偏移: unsigned nibble [0,15] → signed int8 [-8,7]
    vals = vals.to(torch.int8) - 8

    # 整形并截断到实际元素数
    vals = vals.reshape(m, -1)[:, :n]
    return vals


def dequant_w4a16_packed(
    name: str,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """
    compressed-tensors W4A16 packed → BF16。

    W4A16 checkpoint 格式:
      - weight_packed: int32, shape (out, ceil(in * 4 / 32))
      - weight_scale:  float (bf16/fp16), shape (out, in // 128)  per-group
      - weight_shape:  int64, shape (2,)  [out, in]

    返回 None 如果不是 W4A16 packed 格式。
    """
    base = name.replace(".weight_packed", "")
    scale_name = f"{base}.weight_scale"
    shape_name = f"{base}.weight_shape"

    if name not in tensors or scale_name not in tensors:
        return None

    packed = tensors[name]
    scale = tensors[scale_name].to(torch.float32)

    # packed dtype 必须是 int32（区分于 MXFP4 的 uint8 packed）
    if packed.dtype != torch.int32:
        return None

    # 获取原始形状 & group_size (从 scale.shape[1] 反推, 不硬编码)
    m = packed.shape[0]  # out_features
    num_groups = scale.shape[1]
    if shape_name not in tensors:
        raise ValueError(
            f"Missing {shape_name} in checkpoint for {name}. "
            f"compressed-tensors W4A16 checkpoint 必须包含 weight_shape 元数据 "
            f"(pack_quantized/base.py:92 强制保存); 无此元数据则无法确定原始 in_features"
        )
    shape = tensors[shape_name]
    n = int(shape[1].item())
    assert n % num_groups == 0, (
        f"weight_scale groups={num_groups} does not divide in_features={n} for {name}"
    )
    group_size = n // num_groups

    # 解包
    unpacked = unpack_int4_from_int32(packed, m, n)  # (m, n) int8

    # per-group dequant
    scale_f32 = scale.to(torch.float32)

    unpacked_float = unpacked.float()
    unpacked_reshaped = unpacked_float.reshape(m, num_groups, group_size)
    w_fp32 = (unpacked_reshaped * scale_f32.unsqueeze(-1)).reshape(m, n)

    return w_fp32.to(torch.bfloat16)


# ─── 张量名匹配 ───

# 全局层重映射表 (源层号 → mini 层号), 由 build_mini_model 初始化
_LAYER_REMAP: dict[int, int] = {}


def remap_layer_name(name: str) -> str:
    """
    把张量名里的 layers.<orig_idx>. 重写成 layers.<new_idx>.
    (KimiDecoderLayer 索引必须从 0 连续, 无法跳号)
    """
    match = re.match(r"(language_model\.model\.layers\.)(\d+)(\..*)", name)
    if match is None:
        return name
    orig = int(match.group(2))
    if orig not in _LAYER_REMAP:
        return name  # 未选中的层不重命名 (should_keep_tensor 会先把它过滤掉)
    new_idx = _LAYER_REMAP[orig]
    return f"{match.group(1)}{new_idx}{match.group(3)}"


def should_keep_tensor(name: str, keep_layers: set[int], num_experts: int) -> bool:
    """
    判断张量是否应纳入 mini 模型。
    保留: lm_head, embed_tokens, norm, output_attn_res_*,
          以及 keep_layers 集合中列出的层
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

    # Layer 过滤: 只保留 keep_layers 集合中的层
    layer_match = re.match(
        r"language_model\.model\.layers\.(\d+)\.",
        name,
    )
    if layer_match:
        layer_idx = int(layer_match.group(1))
        if layer_idx not in keep_layers:
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
    src_dir: Path, keep_layers: list[int], num_experts: int,
    keep_quant: bool = False,
) -> dict:
    """生成 mini 模型的 config.json

    :param keep_layers: 源模型中保留的层号列表 (按原始顺序), 会被重映射为 [0..N-1]
    :param keep_quant: True 时保留源 quantization_config（用于生成可复现量化 code path
                       的 mini checkpoint）；False 时移除，让模型以纯 BF16 加载。
    """
    with open(src_dir / "config.json") as f:
        config = json.load(f)

    tc = config.get("text_config", config)

    # 读取原始 layer 类型信息 (1-indexed 与源 config 一致)
    orig_full_attn = set()
    orig_kda = set()
    lac = tc.get("linear_attn_config")
    if lac is not None:
        for idx in lac.get("full_attn_layers", []):
            orig_full_attn.add(idx)
        for idx in lac.get("kda_layers", []):
            orig_kda.add(idx)

    # 源 config 的 kda_layers / full_attn_layers 是 1-indexed 的 (layer 1..N),
    # 张量名是 0-indexed 的 (layers.0..N-1). keep_layers 是 0-indexed.
    # 重映射时:
    #   orig 0-index i  →  mini 0-index remap[i]
    # 对应 1-indexed:  orig (i+1) → mini (remap[i]+1)
    remap = {orig: new for new, orig in enumerate(keep_layers)}
    mini_kda_1based = sorted(
        remap[orig - 1] + 1
        for orig in orig_kda
        if (orig - 1) in remap
    )
    mini_full_1based = sorted(
        remap[orig - 1] + 1
        for orig in orig_full_attn
        if (orig - 1) in remap
    )

    # 修正 config（num_experts <= 0 表示使用原始值）
    actual_experts = num_experts if num_experts > 0 else tc["num_experts"]
    num_layers = len(keep_layers)
    tc["num_hidden_layers"] = num_layers
    tc["num_experts"] = actual_experts
    tc["num_experts_per_token"] = min(2, actual_experts)
    # first_k_dense_replace 只在源第 0 层被保留时才是 dense; 否则 mini 全 MoE
    tc["first_k_dense_replace"] = 1 if 0 in remap else 0

    if lac is not None:
        lac["kda_layers"] = mini_kda_1based
        lac["full_attn_layers"] = mini_full_1based

    # 量化配置：keep_quant=True 保留，让 mini checkpoint 走 compressed-tensors
    # 加载路径（复现 pack-quantized 相关问题）；否则彻底移除，纯 BF16 加载。
    # 顶层与 text_config 都要处理，因为 FP8_DYNAMIC 配置可能在顶层。
    if not keep_quant:
        tc.pop("quantization_config", None)
        config.pop("quantization_config", None)

    # 保留 vision_config（modeling 代码需要它初始化 VisionTowerConfig），
    # 但移除顶层引用视觉占位符的字段以避免运行时未定义行为
    for key in ("image_placeholder", "media_placeholder_token_id"):
        config.pop(key, None)

    print(f"  keep_layers (源 0-index):      {keep_layers}")
    print(f"  remap (源 → mini, 0-index):    {remap}")
    print(f"  kda_layers (mini 1-indexed):   {mini_kda_1based}")
    print(f"  full_attn_layers (mini 1-idx): {mini_full_1based}")
    print(f"  first_k_dense_replace:         {tc['first_k_dense_replace']}")
    print(f"  num_experts_per_token:         {tc['num_experts_per_token']}")

    return config

# ─── 主流程 ───

def build_mini_model(
    src_dir: Path,
    dst_dir: Path,
    src_format: str,
    keep_layers: list[int],
    num_experts: int,
    keep_quant: bool = False,
):
    """
    :param keep_layers: 源模型中要保留的层号列表 (0-indexed, 按原始顺序)
                        例如 [0,1,2,3] 表示前 4 层, [40,41,42,43] 表示中间 4 层
    :param keep_quant:  True 时保留 packed 权重和源量化配置（复现 compressed-tensors
                        加载路径）；False (默认) 时解量化为 BF16。
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 初始化全局层号重映射: 源层号 → mini 层号 (0-based 连续)
    global _LAYER_REMAP
    _LAYER_REMAP = {orig: new for new, orig in enumerate(keep_layers)}
    keep_layers_set = set(keep_layers)

    # 从源 config 读取真实 expert 数（避免硬编码）
    with open(src_dir / "config.json") as f:
        src_config = json.load(f)
    src_tc = src_config.get("text_config", src_config)
    total_experts = src_tc.get("num_experts", 0)

    # 校验 num_experts 不超过源模型
    if num_experts > 0 and num_experts > total_experts:
        raise ValueError(
            f"--num-experts={num_experts} 超过源模型 total_experts={total_experts}; "
            f"若想保留全部请传 0 (或省略参数)"
        )

    # ── 1. 读取 index ──
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"未找到 index: {index_path}")
    with open(index_path) as f:
        src_index = json.load(f)
    src_weight_map: dict[str, str] = src_index.get("weight_map", {})
    print(f"源模型张量: {len(src_weight_map)}")
    print(f"保留层 (源 0-index): {keep_layers}")

    # ── 2. 筛选 ──
    files_to_tensors: dict[str, list[str]] = defaultdict(list)
    kept, skipped = 0, 0

    for name, fname in src_weight_map.items():
        if should_keep_tensor(name, keep_layers_set, num_experts):
            files_to_tensors[fname].append(name)
            kept += 1
        else:
            skipped += 1

    print(f"保留: {kept}  排除: {skipped}  "
          f"(layers={len(keep_layers)}, experts={num_experts})")

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
            if name.endswith(".weight_packed") and src_format == "mxfp4" and not keep_quant:
                deq = dequant_mxfp4_packed(name, tensors)
                if deq is not None:
                    base = name.replace(".weight_packed", "")
                    key = remap_layer_name(f"{base}.weight")
                    all_new[key] = deq
                    total_size += deq.numel() * deq.element_size()
                    new_weight_map[key] = "model.safetensors"
                    continue

            # ── INT4 uint8-packed → BF16 ──
            if name.endswith(".weight_packed") and src_format == "int4" and not keep_quant:
                deq = dequant_int4_packed(name, tensors)
                if deq is not None:
                    base = name.replace(".weight_packed", "")
                    key = remap_layer_name(f"{base}.weight")
                    all_new[key] = deq
                    total_size += deq.numel() * deq.element_size()
                    new_weight_map[key] = "model.safetensors"
                    continue

            # ── W4A16 int32-packed → BF16 ──
            if name.endswith(".weight_packed") and src_format == "w4a16" and not keep_quant:
                deq = dequant_w4a16_packed(name, tensors)
                if deq is not None:
                    base = name.replace(".weight_packed", "")
                    key = remap_layer_name(f"{base}.weight")
                    all_new[key] = deq
                    total_size += deq.numel() * deq.element_size()
                    new_weight_map[key] = "model.safetensors"
                    continue

            # ── FP8 weight → BF16 ──
            if name.endswith(".weight") and needs_dequant(tensor) and not keep_quant:
                scale_name = name.replace(".weight", ".weight_scale")
                # 尝试当前文件，找不到则查跨 shard 缓存
                scale = tensors.get(scale_name)
                if scale is None:
                    scale = scale_cache.get(scale_name)
                if scale is not None:
                    deq = dequant_weight(tensor, scale)
                    key = remap_layer_name(name)
                    all_new[key] = deq
                    total_size += deq.numel() * deq.element_size()
                    new_weight_map[key] = "model.safetensors"
                    continue

            # ── 跳过 weight_scale / weight_shape（已在解量化中消耗） ──
            # keep_quant 模式下这些辅助张量需要保留，让下面"普通张量"分支透传。
            if (name.endswith(".weight_scale") or name.endswith(".weight_shape")) and not keep_quant:
                continue

            # ── 普通张量 ──
            key = remap_layer_name(name)
            all_new[key] = tensor
            total_size += tensor.numel() * tensor.element_size()
            new_weight_map[key] = "model.safetensors"

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
    config = build_mini_config(src_dir, keep_layers, num_experts, keep_quant=keep_quant)
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


def resolve_keep_layers(
    num_layers: int | None,
    layer_range: str | None,
    layer_ids: str | None,
    src_num_layers: int,
) -> list[int]:
    """三种层选择方式互斥, 统一转成 0-index 层号列表 (按原始顺序)."""
    provided = [x is not None for x in (num_layers, layer_range, layer_ids)]
    if sum(provided) > 1:
        raise ValueError(
            "--num-layers / --layer-range / --layer-ids 三个参数互斥, 只能选其一"
        )

    if layer_ids is not None:
        ids = sorted(int(x) for x in layer_ids.split(",") if x.strip())
    elif layer_range is not None:
        parts = layer_range.split(":")
        if len(parts) != 2:
            raise ValueError(f"--layer-range 格式应为 START:END, 得到 {layer_range}")
        start, end = int(parts[0]), int(parts[1])
        if not (0 <= start < end <= src_num_layers):
            raise ValueError(
                f"--layer-range {start}:{end} 越界; 源模型共 {src_num_layers} 层"
            )
        ids = list(range(start, end))
    else:
        # 默认或显式 --num-layers
        n = num_layers if num_layers is not None else 4
        if not (0 < n <= src_num_layers):
            raise ValueError(f"--num-layers={n} 越界 (源共 {src_num_layers} 层)")
        ids = list(range(n))

    # 校验都在合法范围
    for idx in ids:
        if not (0 <= idx < src_num_layers):
            raise ValueError(f"层号 {idx} 越界 (源共 {src_num_layers} 层, 0-index)")
    if len(ids) == 0:
        raise ValueError("keep_layers 不能为空; 请传入至少 1 个层号")
    return ids


# ─── CLI ───

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建 Kimi K3 mini 模型（用于推理验证）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认前 4 层 (0..3, 覆盖 dense+KDA / MoE+KDA / MoE+MLA)
  python build_mini_kimi_k3.py

  # 从原始 MXFP4 构建
  python build_mini_kimi_k3.py --src-format mxfp4

  # 截取中间连续 4 层 (源 layer 40..43)
  python build_mini_kimi_k3.py --layer-range 40:44

  # 显式指定不连续层号 (源 layer 0,12,45,92, 一般用于挑覆盖 MLA/KDA 边界)
  python build_mini_kimi_k3.py --layer-ids 0,12,45,92
        """,
    )
    parser.add_argument("--src", type=Path,
                        default=Path("/mnt/c/chl/models/Kimi-K3-FP8-DYNAMIC"),
                        help="源模型目录")
    parser.add_argument("--dst", type=Path,
                        default=Path("/mnt/c/chl/models/Kimi-K3-Mini"),
                        help="输出目录")
    parser.add_argument("--src-format", choices=["fp8", "mxfp4", "int4", "w4a16", "bf16"], default="fp8",
                        help="源模型格式 (fp8 / mxfp4 / int4 / w4a16 / bf16)")
    parser.add_argument("-n", "--num-layers", type=int, default=None,
                        help="保留前 N 层 (0..N-1); 默认 4; 与 --layer-range/--layer-ids 互斥")
    parser.add_argument("--layer-range", type=str, default=None,
                        help="截取源层区间 START:END (半开区间, 0-index), 例: 40:44")
    parser.add_argument("--layer-ids", type=str, default=None,
                        help="显式指定源层号列表, 逗号分隔, 例: 0,12,45,92")
    parser.add_argument("-e", "--num-experts", type=int, default=0,
                        help="每层保留的 expert 数 (默认 0 保留全部)")
    parser.add_argument("--keep-quant", action="store_true",
                        help="保留源量化权重与 quantization_config（透传 weight_packed/"
                             "weight_scale/weight_shape），用于本地复现 compressed-tensors "
                             "加载路径。默认解量化为 BF16。仅对 fp8/mxfp4/int4/w4a16 有意义。")
    args = parser.parse_args()

    # 从源 config 读 num_hidden_layers 做范围校验
    with open(args.src / "config.json") as f:
        _src_cfg = json.load(f)
    _src_tc = _src_cfg.get("text_config", _src_cfg)
    _src_num_layers = int(_src_tc["num_hidden_layers"])

    keep_layers = resolve_keep_layers(
        args.num_layers, args.layer_range, args.layer_ids, _src_num_layers
    )

    build_mini_model(args.src, args.dst, args.src_format,
                     keep_layers, args.num_experts, keep_quant=args.keep_quant)
