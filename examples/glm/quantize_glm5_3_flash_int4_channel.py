# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3-Flash FP8 block -> 混合精度 INT4/INT8 W4A8+W8A8 转换脚本

背景与设计
==========
GLM-5.3-Flash 官方 FP8 (E4M3 blockwise 128x128) 模型 → 混合精度落盘布局:
  - layer 3-44 的 routed experts (mlp.experts.{i}.{gate,up,down}_proj)
    量化为 INT4 W4A8 channelwise + 沿 in_dim int32 pack (compressed-tensors
    pack-quantized), 共 288 × 3 × 42 = 36288 个 tensor
  - 其它所有原本 INT8 的目标 (layer 45 MTP experts / shared_experts /
    dense MLP layer 0-2 / MLA 4 投影 / linear-attn q/k/v/o / DSA indexer
    wq_b/wk) 保持 INT8 W8A8 channelwise (compressed-tensors int-quantized)
  - 所有 BF16/FP32 保留张量 (norm / conv1d / hc_ / A_log / dt_bias /
    router / vision / linear-attn 5 门控 / kv_b_proj / indexer 其它)
    原样透传

覆盖范围对齐 iter_0033200_fixclamp_HF_int4 (私有 w4a8_int 格式) 但字节格式
改用 compressed-tensors 标准, 可被 vllm.glm53flash 分支的
CompressedTensorsW4A8Int8MoEMethod + CompressedTensorsW8A8Int8MoEMethod
双 loader 原生识别加载.

关键公式来源
============
INT4 对称量化公式对齐 utils/MoE-Quant/src/quant_utils.py:94-116
(find_quantization_meta, symmetric=True):
    abs_max  = max(|w|) along in_dim
    scale    = abs_max / 7        (qmax_pos = (2^4 - 1 - 1) / 2 = 7)
    qzero    = 8                  ((2^4 - 1 + 1) / 2 = 8)
    q_uint4  = round(w / scale + 8).clamp(0, 15)   ∈ [0, 15]
    q_int    = q_uint4 - 8                          ∈ [-8, 7]
用 scale/7 (而非 scale/8) 保证 ±absmax 精确 round-trip; -8 是哨兵不使用.

Pack 布局对齐 compressed_tensors.compressors.pack_to_int32 (packed_dim=1):
    element i (int8, [-8,7]) → int32 lane 的 bits [i*4 : i*4+4]
    (函数内部会 +8 转成 uint4 [0,15] 再左移)
    输入必须 torch.int8, 输出 shape (out, in//8) int32

用法
====
  python quantize_glm5_3_flash_int4_channel.py \\
      --input-dir  /path/GLM-5.3-Flash \\
      --output-dir /path/GLM-5.3-Flash-CHANNEL-INT4-w4a8 \\
      --no-mla-to-bf16 --linear-attn-int8 --kda-int8-scope qkvo --indexer-int8

前置条件
========
  - torch >= 2.1 (含 torch.float8_e4m3fn dtype)
  - safetensors, tqdm
  - compressed_tensors (提供 pack_to_int32; 若不可用会 fallback 到内嵌等价实现)
  - 源 checkpoint 为 GLM-5.3-Flash 官方 FP8 发布版
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# pack_to_int32 依赖: 首选 compressed_tensors 官方实现 (loader 端解包契约).
# 若 import 失败降级到本地等价实现 (位序与官方一致, 但仅在无官方包时使用).
# --------------------------------------------------------------------------- #

try:
    from compressed_tensors.compressors import pack_to_int32 as _ct_pack_to_int32
    _HAVE_COMPRESSED_TENSORS_PACK = True
except ImportError:
    _HAVE_COMPRESSED_TENSORS_PACK = False

    def _ct_pack_to_int32(value: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Fallback for compressed_tensors.compressors.pack_to_int32(packed_dim=1).

        与官方等价: element i 放到 lane bits [i*num_bits : i*num_bits+num_bits],
        输入 int8 值域 [-2**(bits-1), 2**(bits-1)-1] 内部 +offset 转 uint 再打包.
        仅覆盖 num_bits <= 8, packed_dim=1 (最后一维), cols 可被 (32/num_bits) 整除的情形
        (足够本脚本 GLM-5.3-Flash W4 使用; in_dim 都是 128 的倍数).
        """
        if value.dtype is not torch.int8:
            raise ValueError("Tensor must be quantized to torch.int8 before packing")
        if not 1 <= num_bits <= 8:
            raise ValueError(f"num_bits must be in [1, 8], got {num_bits}")
        per_lane = 32 // num_bits
        if value.shape[-1] % per_lane != 0:
            raise ValueError(
                f"fallback pack_to_int32 requires last-dim divisible by {per_lane}, "
                f"got {value.shape}"
            )
        offset = 1 << (num_bits - 1)
        v = value.to(torch.int32) + offset   # → [0, 2**bits - 1]
        lanes = v.view(*value.shape[:-1], -1, per_lane)   # (..., cols/per_lane, per_lane)
        shifts = torch.arange(per_lane, device=v.device, dtype=torch.int32) * num_bits
        packed = (lanes << shifts).sum(dim=-1).to(torch.int32)
        return packed


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

INT8_QMAX = 127.0

INT4_BITS = 4
INT4_QMAX_POS = 7.0                       # (2^4 - 1 - 1) / 2 = 7 (对称上限)
INT4_QZERO = 8                            # (2^4 - 1 + 1) / 2 = 8 (中心化零点)
INT4_PACK_FACTOR = 32 // INT4_BITS        # 8 个 int4 pack 到 1 个 int32

# GLM-5.3-Flash 官方 FP8 blockwise tile size (config.json.weight_block_size)
FP8_BLOCK_SIZE = 128


# --------------------------------------------------------------------------- #
# scale 命名: 源 = .weight_scale_inv, 输出 = .weight_scale (compressed-tensors)
# --------------------------------------------------------------------------- #

def src_scale_name(weight_name: str) -> str:
    """源 checkpoint 中 FP8 权重对应的 scale 名 (GLM 用 _inv 后缀)"""
    assert weight_name.endswith(".weight")
    return weight_name[: -len(".weight")] + ".weight_scale_inv"


def out_scale_name(weight_name: str) -> str:
    """INT8 输出中 scale 的名字 (compressed-tensors int-quantized 期望 .weight_scale)"""
    assert weight_name.endswith(".weight")
    return weight_name[: -len(".weight")] + ".weight_scale"


def out_int4_packed_name(weight_name: str) -> str:
    """INT4 输出的 packed 权重名 (compressed-tensors pack-quantized 期望 .weight_packed)"""
    assert weight_name.endswith(".weight")
    return weight_name[: -len(".weight")] + ".weight_packed"


def out_int4_shape_name(weight_name: str) -> str:
    """INT4 输出的原始 shape 张量名 (供加载器解包用)"""
    assert weight_name.endswith(".weight")
    return weight_name[: -len(".weight")] + ".weight_shape"


def is_fp8_blockwise_weight(name: str, tensor: torch.Tensor, state_dict: dict) -> bool:
    """FP8 blockwise 权重判据: dtype 是 float8_e4m3fn 且存在配对的 _scale_inv"""
    if not name.endswith(".weight"):
        return False
    if tensor.dtype != torch.float8_e4m3fn:
        return False
    return src_scale_name(name) in state_dict


# --------------------------------------------------------------------------- #
# 各类被特殊处理的张量后缀 (与 INT8 脚本完全一致)
# --------------------------------------------------------------------------- #

# MLA 4 投影: vLLM/SGLang 加载器硬编码 BF16 假设的融合投影.
_MLA_BF16_SUFFIXES = (
    ".self_attn.q_a_proj.weight",
    ".self_attn.q_b_proj.weight",
    ".self_attn.kv_a_proj_with_mqa.weight",
    ".self_attn.o_proj.weight",
)


def is_mla_bf16_target(name: str) -> bool:
    return any(name.endswith(sfx) for sfx in _MLA_BF16_SUFFIXES)


# Linear-Attention 层 (KDA / GDN) 的 2D Linear 权重, 源 BF16.
_LINEAR_ATTN_INT8_QKVO_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",     # 只命中 linear-attn 层 (BF16); sparse-attn 层的
                                    # o_proj 是 FP8, 已由 FP8 分支处理.
)
_LINEAR_ATTN_INT8_GATE_SUFFIXES = (
    ".self_attn.b_proj.weight",
    ".self_attn.f_a_proj.weight",
    ".self_attn.g_a_proj.weight",
    ".self_attn.f_b_proj.weight",
    ".self_attn.g_b_proj.weight",
)


def is_linear_attn_int8_target(name: str, scope: str = "full") -> bool:
    if any(name.endswith(sfx) for sfx in _LINEAR_ATTN_INT8_QKVO_SUFFIXES):
        return True
    if scope == "full" and any(
        name.endswith(sfx) for sfx in _LINEAR_ATTN_INT8_GATE_SUFFIXES
    ):
        return True
    return False


# DSA (Deepseek Sparse Attention) 层 indexer 里的两个 2D Linear 权重.
_INDEXER_INT8_SUFFIXES = (
    ".self_attn.indexer.wq_b.weight",
    ".self_attn.indexer.wk.weight",
)


def is_indexer_int8_target(name: str) -> bool:
    return any(name.endswith(sfx) for sfx in _INDEXER_INT8_SUFFIXES)


# --------------------------------------------------------------------------- #
# INT4 路由: layer 3-44 的 routed experts (与 iter_0033200 覆盖范围对齐).
# layer 0-2 是 dense MLP, layer 45 是 MTP (MoE experts 保 INT8 —— 对齐
# iter_0033200 config 的 moe_int8_layers: [45]).
# 命名前缀兼容 "model.layers." 与 "model.language_model.layers." (GLM 特有).
# --------------------------------------------------------------------------- #

_ROUTED_EXPERT_INT4_LAYERS = tuple(range(3, 45))   # [3, 4, ..., 44]
_ROUTED_EXPERT_INT4_PATTERN = re.compile(
    r"(?:^|\.)layers\.(?:" + "|".join(str(i) for i in _ROUTED_EXPERT_INT4_LAYERS)
    + r")\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
)


def is_routed_expert_int4_target(name: str) -> bool:
    """layer 3-44 的 routed experts (gate/up/down_proj) — 唯一走 INT4 的类别."""
    return _ROUTED_EXPERT_INT4_PATTERN.search(name) is not None


# --------------------------------------------------------------------------- #
# FP8 blockwise 反量化 + 量化 (INT8 channelwise / INT4 channelwise packed)
# --------------------------------------------------------------------------- #

def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    反量化 FP8 blockwise (128x128) 权重为 float32.
      weight: FP8 E4M3FN,   shape (out_dim, in_dim)
      scale:  float32/bf16, shape (ceil(out_dim/128), ceil(in_dim/128))
    """
    assert weight.ndim == 2, f"expect 2D weight, got {weight.shape}"
    assert scale.ndim == 2, f"expect 2D scale, got {scale.shape}"

    scale = scale.float()
    w = weight.float()
    out_dim, in_dim = w.shape
    s_rows, s_cols = scale.shape

    if s_rows > 0 and s_cols > 0 and out_dim % s_rows == 0 and in_dim % s_cols == 0:
        row_block = out_dim // s_rows
        col_block = in_dim // s_cols
        w_blocks = w.unflatten(0, (s_rows, row_block)).unflatten(2, (s_cols, col_block))
        expanded = scale[:, None, :, None]
        return (w_blocks * expanded).flatten(0, 1).flatten(1, 2)

    row_block = FP8_BLOCK_SIZE
    col_block = FP8_BLOCK_SIZE
    out = torch.empty_like(w)
    for i in range(s_rows):
        r0, r1 = i * row_block, min((i + 1) * row_block, out_dim)
        for j in range(s_cols):
            c0, c1 = j * col_block, min((j + 1) * col_block, in_dim)
            out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * scale[i, j]
    return out


def quantize_int8_channelwise(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对称 INT8 channelwise (per-output-channel) 量化.
      tensor: float, shape (out_dim, in_dim)
    返回:
      quantized: int8,    shape (out_dim, in_dim), 值域 [-127, 127]
      scale:     float32, shape (out_dim, 1)
    """
    assert tensor.ndim == 2, f"expect 2D tensor, got {tensor.shape}"
    w = tensor.float()
    abs_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = abs_max / INT8_QMAX
    q = torch.round(w / scale).clamp(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q, scale.to(torch.float32)


def quantize_int4_channelwise_packed(
    tensor: torch.Tensor,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对称 INT4 channelwise (per-output-channel) 量化 + int32 pack.
      tensor: float, shape (out_dim, in_dim), in_dim 必须能被 8 整除
    返回:
      weight_packed: int32, shape (out_dim, in_dim // 8)  (每 int32 存 8 个 int4)
      weight_shape : int32, shape (2,)  值 [out_dim, in_dim]
      weight_scale : bf16,  shape (out_dim, 1)  per-out-channel scale

    量化公式对齐 utils/MoE-Quant/src/quant_utils.py:94-116:
      scale = abs_max / 7, qzero = 8, q_uint4 ∈ [0,15], 中心化后 q_int ∈ [-8,7]
    这样 ±absmax 精确 round-trip; -8 是哨兵 (代码 0 对应, 未使用).
    """
    assert tensor.ndim == 2, f"expect 2D, got {tensor.shape}"
    out_dim, in_dim = tensor.shape
    assert in_dim % INT4_PACK_FACTOR == 0, (
        f"in_dim {in_dim} must be divisible by {INT4_PACK_FACTOR} for int4 pack"
    )

    w = tensor.float()
    abs_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)            # (out, 1) fp32
    scale = abs_max / INT4_QMAX_POS                                         # (out, 1) fp32
    q_uint4 = torch.round(w / scale + INT4_QZERO).clamp(0, 15)              # (out, in) fp32

    # 借 int32 中转,避免 uint8→int8 的溢出/wrap:
    # (float 15).to(int8) 无问题, 但 (uint8 128).to(int8) 在部分平台行为不定.
    q_int = (q_uint4.to(torch.int32) - INT4_QZERO).to(torch.int8)           # (out, in) int8 ∈ [-8,7]

    weight_packed = _ct_pack_to_int32(q_int, INT4_BITS)                     # (out, in//8) int32
    weight_shape = torch.tensor([out_dim, in_dim], dtype=torch.int32)       # (2,)
    weight_scale = scale.to(scale_dtype).contiguous()                       # (out, 1) bf16

    return weight_packed.contiguous(), weight_shape, weight_scale


# --------------------------------------------------------------------------- #
# 单 shard 转换
# --------------------------------------------------------------------------- #

_STATS_KEYS = (
    "experts_int4_packed",         # layer 3-44 routed experts 量化为 INT4 的 tensor 数
    "fp8_weights_converted",       # FP8 -> INT8 (含 layer 45 experts / shared / dense MLP / MLA / linear-attn/indexer 中的 FP8 源)
    "mla_dequant_to_bf16",         # MLA 4 投影反量化到 BF16 透传的 tensor 数
    "linear_attn_int8_quantized",  # Linear-Attention 层 BF16 -> INT8 量化的 tensor 数
    "indexer_int8_quantized",      # DSA indexer wq_b/wk BF16 -> INT8 量化的 tensor 数
    "scales_dropped",              # 消费掉的源 _scale_inv 数
    "kept",                        # 透传张量数
    "skipped_shape_mismatch",      # weight/scale shape 不匹配, 原样透传
)


def _empty_stats() -> dict[str, int]:
    return {k: 0 for k in _STATS_KEYS}


def convert_one_file(
    input_path: str,
    output_path: str,
    stats: dict[str, int],
    device: torch.device | str = "cpu",
    mla_to_bf16: bool = True,
    linear_attn_int8: bool = False,
    kda_int8_scope: str = "full",
    indexer_int8: bool = False,
) -> None:
    """转换单个 safetensors 分片 (混合精度 INT4/INT8/BF16).

    分支顺序 (先窄后宽, 命中即出):
      1. `.weight_scale_inv` -> 丢弃 (稍后按 INT4/INT8 重新写)
      2. MLA 4 投影反量化到 BF16 (若 mla_to_bf16=True)
      3. 【INT4】routed experts layer 3-44 (FP8 -> INT4 W4A8 packed)
      4. 【INT8】FP8 blockwise -> INT8 channelwise (涵盖 layer 45 experts /
         shared_experts / dense MLP / MLA / 其它 FP8 源)
      5. 【INT8】Linear-attn BF16 -> INT8 (若 linear_attn_int8=True)
      6. 【INT8】DSA indexer BF16 -> INT8 (若 indexer_int8=True)
      7. 其它张量原样透传

    参数语义与 INT8 脚本 (quantize_glm5_3_flash_int8_channel-x.py) 一致.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name)

    new_state_dict: dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        # 1. 源 scale 单独处理
        if name.endswith(".weight_scale_inv"):
            stats["scales_dropped"] += 1
            continue

        # 2. MLA 4 投影反量化到 BF16 透传
        if (
            mla_to_bf16
            and is_mla_bf16_target(name)
            and is_fp8_blockwise_weight(name, tensor, state_dict)
        ):
            scale = state_dict[src_scale_name(name)]
            if scale.ndim != 2 or scale.shape[0] == 0 or scale.shape[1] == 0:
                print(f"[mla skip non-2D scale] {name}: w={tuple(tensor.shape)}, "
                      f"s={tuple(scale.shape)}")
                new_state_dict[name] = tensor
                new_state_dict[src_scale_name(name)] = scale
                stats["skipped_shape_mismatch"] += 1
                stats["scales_dropped"] -= 1
                continue
            w_gpu = tensor.to(device)
            s_gpu = scale.to(device)
            weight_bf16 = dequant_fp8_blockwise(w_gpu, s_gpu).bfloat16()
            new_state_dict[name] = weight_bf16.cpu().contiguous()
            stats["mla_dequant_to_bf16"] += 1
            del w_gpu, s_gpu, weight_bf16
            continue

        # 3. 【INT4】routed experts layer 3-44 (必须在 FP8→INT8 分支之前)
        if (
            is_routed_expert_int4_target(name)
            and is_fp8_blockwise_weight(name, tensor, state_dict)
        ):
            scale = state_dict[src_scale_name(name)]
            if scale.ndim != 2 or scale.shape[0] == 0 or scale.shape[1] == 0:
                print(f"[int4 skip non-2D scale] {name}: w={tuple(tensor.shape)}, "
                      f"s={tuple(scale.shape)}")
                new_state_dict[name] = tensor
                new_state_dict[src_scale_name(name)] = scale
                stats["skipped_shape_mismatch"] += 1
                stats["scales_dropped"] -= 1
                continue

            w_gpu = tensor.to(device)
            s_gpu = scale.to(device)
            weight_fp32 = dequant_fp8_blockwise(w_gpu, s_gpu)
            # pack_to_int32 在 CPU 上稳定; GPU 上也可用. 保持 device 一致即可.
            weight_packed, weight_shape, weight_scale = quantize_int4_channelwise_packed(
                weight_fp32, scale_dtype=torch.bfloat16
            )
            new_state_dict[out_int4_packed_name(name)] = weight_packed.cpu().contiguous()
            new_state_dict[out_int4_shape_name(name)] = weight_shape.cpu().contiguous()
            new_state_dict[out_scale_name(name)] = weight_scale.cpu().contiguous()
            stats["experts_int4_packed"] += 1
            del w_gpu, s_gpu, weight_fp32, weight_packed, weight_shape, weight_scale
            continue

        # 4. 【INT8】FP8 blockwise -> INT8 channelwise (兜底所有 FP8 源)
        if is_fp8_blockwise_weight(name, tensor, state_dict):
            scale = state_dict[src_scale_name(name)]
            out_dim, in_dim = tensor.shape
            s_rows, s_cols = scale.shape if scale.ndim == 2 else (0, 0)
            if scale.ndim != 2 or s_rows == 0 or s_cols == 0:
                print(f"[skip non-2D scale] {name}: w={tuple(tensor.shape)}, "
                      f"s={tuple(scale.shape)}")
                new_state_dict[name] = tensor
                new_state_dict[src_scale_name(name)] = scale
                stats["skipped_shape_mismatch"] += 1
                stats["scales_dropped"] -= 1
                continue

            w_gpu = tensor.to(device)
            s_gpu = scale.to(device)
            weight_fp32 = dequant_fp8_blockwise(w_gpu, s_gpu)
            weight_int8, new_scale = quantize_int8_channelwise(weight_fp32)
            new_state_dict[name] = weight_int8.cpu().contiguous()
            new_state_dict[out_scale_name(name)] = new_scale.cpu().contiguous()
            stats["fp8_weights_converted"] += 1
            del w_gpu, s_gpu, weight_fp32, weight_int8, new_scale
            continue

        # 5. 【INT8】Linear-Attention 层 BF16 -> INT8 channelwise
        if (
            linear_attn_int8
            and is_linear_attn_int8_target(name, scope=kda_int8_scope)
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
        ):
            w_gpu = tensor.to(device)
            weight_int8, new_scale = quantize_int8_channelwise(w_gpu)
            new_state_dict[name] = weight_int8.cpu().contiguous()
            new_state_dict[out_scale_name(name)] = new_scale.cpu().contiguous()
            stats["linear_attn_int8_quantized"] += 1
            del w_gpu, weight_int8, new_scale
            continue

        # 6. 【INT8】DSA indexer wq_b / wk BF16 -> INT8 channelwise
        if (
            indexer_int8
            and is_indexer_int8_target(name)
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
        ):
            w_gpu = tensor.to(device)
            weight_int8, new_scale = quantize_int8_channelwise(w_gpu)
            new_state_dict[name] = weight_int8.cpu().contiguous()
            new_state_dict[out_scale_name(name)] = new_scale.cpu().contiguous()
            stats["indexer_int8_quantized"] += 1
            del w_gpu, weight_int8, new_scale
            continue

        # 7. 其它张量原样透传
        new_state_dict[name] = tensor
        stats["kept"] += 1

    save_file(new_state_dict, output_path)


# --------------------------------------------------------------------------- #
# 设备解析 + 多 GPU worker (与 INT8 脚本同结构)
# --------------------------------------------------------------------------- #

def _resolve_devices(
    device_kind: str,
    gpu_ids: str | None,
    workers: int | None,
    num_files: int,
) -> tuple[list[torch.device], int]:
    cuda_available = torch.cuda.is_available()

    if device_kind == "cpu":
        return [torch.device("cpu")], 1
    if device_kind == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda specified but torch.cuda.is_available() == False")
    if device_kind == "auto" and not cuda_available:
        return [torch.device("cpu")], 1

    if gpu_ids:
        try:
            ids = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
        except ValueError as e:
            raise RuntimeError(f"invalid --gpu-ids {gpu_ids!r}: {e}") from e
    else:
        ids = list(range(torch.cuda.device_count()))
    if not ids:
        raise RuntimeError("no CUDA device selected (device_count == 0)")

    devices = [torch.device(f"cuda:{i}") for i in ids]
    if workers is None:
        effective = min(len(devices), max(1, num_files))
    else:
        effective = min(max(1, workers), max(1, num_files))
    return devices, effective


def _worker_convert_shard(
    args: tuple[str, str, int, int, bool, bool, str, bool],
) -> dict[str, int]:
    (input_path, output_path, gpu_id, num_threads,
     mla_to_bf16, linear_attn_int8, kda_int8_scope, indexer_int8) = args
    torch.set_num_threads(max(1, num_threads))
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    stats = _empty_stats()
    convert_one_file(
        input_path,
        output_path,
        stats,
        device,
        mla_to_bf16=mla_to_bf16,
        linear_attn_int8=linear_attn_int8,
        kda_int8_scope=kda_int8_scope,
        indexer_int8=indexer_int8,
    )
    return stats


def convert_model(
    input_dir: str,
    output_dir: str,
    limit_files: int | None,
    device_kind: str,
    gpu_ids: str | None,
    workers: int | None,
    num_threads: int,
    mla_to_bf16: bool = True,
    linear_attn_int8: bool = False,
    kda_int8_scope: str = "full",
    indexer_int8: bool = False,
) -> tuple[dict[str, int], list[torch.device], int]:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    if limit_files is not None:
        files = files[:limit_files]

    devices, effective_workers = _resolve_devices(device_kind, gpu_ids, workers, len(files))
    stats = _empty_stats()

    if effective_workers <= 1 or len(devices) == 1:
        device = devices[0]
        for path in tqdm(files, desc=f"Converting on {device}"):
            fname = os.path.basename(path)
            convert_one_file(
                path,
                os.path.join(output_dir, fname),
                stats,
                device,
                mla_to_bf16=mla_to_bf16,
                linear_attn_int8=linear_attn_int8,
                kda_int8_scope=kda_int8_scope,
                indexer_int8=indexer_int8,
            )
        return stats, devices, effective_workers

    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")

    tasks: list[tuple[str, str, int, int, bool, bool, str, bool]] = []
    for i, path in enumerate(files):
        gpu_id = devices[i % len(devices)].index
        fname = os.path.basename(path)
        tasks.append((
            path,
            os.path.join(output_dir, fname),
            int(gpu_id),
            num_threads,
            mla_to_bf16,
            linear_attn_int8,
            kda_int8_scope,
            indexer_int8,
        ))

    desc = f"Converting on {len(devices)} GPU(s) ({effective_workers} workers)"
    with ctx.Pool(processes=effective_workers) as pool:
        for per_stats in tqdm(
            pool.imap_unordered(_worker_convert_shard, tasks),
            total=len(tasks),
            desc=desc,
        ):
            for k, v in per_stats.items():
                stats[k] = stats.get(k, 0) + v

    return stats, devices, effective_workers


# --------------------------------------------------------------------------- #
# compressed-tensors config 生成 (双 config_groups: INT4 W4A8 + INT8 W8A8)
# --------------------------------------------------------------------------- #

def build_ignore_list(
    mla_to_bf16: bool = True,
    linear_attn_int8: bool = False,
    kda_int8_scope: str = "full",
    indexer_int8: bool = False,
) -> list[str]:
    """
    ignore 列表 — 覆盖所有 BF16/FP32 保留张量, 与 INT8 脚本完全一致.
    (routed experts / shared experts / dense MLP / MLA 4 投影 / linear-attn qkvo /
     indexer wq_b/wk 都不列入 ignore, 会分别命中 group_int4 或 group_int8.)
    """
    _LINEAR_ATTN_KEEP_ALWAYS = [
        r"re:.*self_attn\.A_log$",
        r"re:.*self_attn\.dt_bias$",
        r"re:.*self_attn\.k_conv1d$",
        r"re:.*self_attn\.q_conv1d$",
        r"re:.*self_attn\.v_conv1d$",
        r"re:.*self_attn\.o_norm$",
    ]
    _LINEAR_ATTN_PROJ_QKVO = [
        r"re:.*self_attn\.k_proj$",
        r"re:.*self_attn\.q_proj$",
        r"re:.*self_attn\.v_proj$",
        r"re:.*self_attn\.qkv_proj$",
        r"re:.*self_attn\.fused_qkvbfg_a_proj$",
    ]
    _LINEAR_ATTN_PROJ_GATE = [
        r"re:.*self_attn\.b_proj$",
        r"re:.*self_attn\.f_a_proj$",
        r"re:.*self_attn\.f_b_proj$",
        r"re:.*self_attn\.g_a_proj$",
        r"re:.*self_attn\.g_b_proj$",
    ]

    ignore = [
        "lm_head",
        r"re:.*embed_tokens.*",
        r"re:^model\.language_model\.norm$",
        r"re:^model\.norm$",
        r"re:.*layers\.\d+\.(input_layernorm|post_attention_layernorm)$",
        r"re:.*layers\.\d+\.hc_(attn|ffn)_(base|fn|scale)$",
        r"re:.*mlp\.gate$",
        r"re:.*mlp\.gate\..*",
    ]
    ignore.extend(_LINEAR_ATTN_KEEP_ALWAYS)

    if not linear_attn_int8:
        ignore.extend(_LINEAR_ATTN_PROJ_QKVO)
        ignore.extend(_LINEAR_ATTN_PROJ_GATE)
    elif kda_int8_scope == "qkvo":
        ignore.extend(_LINEAR_ATTN_PROJ_GATE)

    ignore.extend([
        r"re:.*self_attn\.q_a_layernorm$",
        r"re:.*self_attn\.kv_a_layernorm$",
        r"re:.*self_attn\.kv_b_proj$",
    ])

    if indexer_int8:
        ignore.extend([
            r"re:.*self_attn\.indexer\.k_norm(\..*)?$",
            r"re:.*self_attn\.indexer\.weights_proj$",
            r"re:.*self_attn\.indexer\.index_kpool_compress_ape$",
            r"re:.*self_attn\.indexer\.index_kpool_compress_gate$",
        ])
    else:
        ignore.extend([
            r"re:.*self_attn\.indexer\..*",
            r"re:.*self_attn\.indexer$",
        ])

    ignore.extend([
        r"re:.*layers\.\d+\.eh_proj$",
        r"re:.*layers\.\d+\.enorm$",
        r"re:.*layers\.\d+\.hnorm$",
        r"re:.*layers\.\d+\.shared_head\..*",
        r"re:.*\.visual\..*",
        r"re:^model\.visual\..*",
        r"re:^visual\..*",
    ])

    if mla_to_bf16:
        # MLA 4 投影反量化到 BF16 落盘: 显式列入 ignore.
        ignore.extend([
            r"re:.*self_attn\.q_a_proj$",
            r"re:.*self_attn\.q_b_proj$",
            r"re:.*self_attn\.kv_a_proj_with_mqa$",
        ])
        if linear_attn_int8:
            ignore.append(
                r"re:.*layers\.(3|7|11|15|19|23|27|31|35|39|43|45)\.self_attn\.o_proj$"
            )
        else:
            ignore.append(r"re:.*self_attn\.o_proj$")

    return ignore


# routed experts (layer 3-44) 的 targets 正则. 用非贪婪匹配以兼容
# "model.layers." / "model.language_model.layers." 两种前缀.
_ROUTED_EXPERT_INT4_TARGET_REGEX = (
    r"re:.*\.layers\.(?:" + "|".join(str(i) for i in _ROUTED_EXPERT_INT4_LAYERS)
    + r")\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$"
)


def _build_int4_activations() -> dict:
    """W4A8: 输入激活走 per-token dynamic INT8 (与 INT8 分支相同)."""
    return {
        "dynamic": True, "group_size": None, "num_bits": 8,
        "observer": "minmax", "observer_kwargs": {},
        "strategy": "token", "symmetric": True, "type": "int",
    }


def build_compression_config(
    mla_to_bf16: bool = True,
    linear_attn_int8: bool = False,
    kda_int8_scope: str = "full",
    indexer_int8: bool = False,
) -> dict:
    """生成 compressed-tensors 双 config_groups 配置:
      group_int4: routed experts layer 3-44 (W4A8, pack-quantized, channel)
      group_int8: Linear 兜底 (W8A8, int-quantized, channel)
    vLLM 侧 find_matched_target 会按最具体正则为每层分派到对应 group.
    """
    int8_input_activations = _build_int4_activations()   # 两组都用 A8 per-token dynamic

    group_int8 = {
        "targets": ["Linear"],
        "input_activations": int8_input_activations,
        "output_activations": None,
        "format": "int-quantized",   # per-group 覆盖: W8A8 用 int-quantized
        "weights": {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": -1,        # channel-wise (compressed-tensors 用 -1 表 channel)
            "num_bits": 8,
            "observer": "minmax",
            "observer_kwargs": {},
            "strategy": "channel",
            "symmetric": True,
            "type": "int",
        },
    }

    group_int4 = {
        "targets": [_ROUTED_EXPERT_INT4_TARGET_REGEX],
        "input_activations": int8_input_activations,
        "output_activations": None,
        "format": "pack-quantized",  # per-group 覆盖: W4 必须 pack-quantized
        "weights": {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": -1,
            "num_bits": 4,
            "observer": "minmax",
            "observer_kwargs": {},
            "strategy": "channel",
            "symmetric": True,
            "type": "int",
        },
    }

    return {
        "config_groups": {
            "group_int4": group_int4,
            "group_int8": group_int8,
        },
        # 顶层 format: 两组都显式覆盖了 per-group format, 顶层作为默认值.
        # 选 pack-quantized 是因为 (a) 双 loader 里 pack-quantized 的兼容面更广;
        # (b) 万一某组 format 被 loader 回退到顶层, W4 侧必须是 pack-quantized.
        "format": "pack-quantized",
        "ignore": build_ignore_list(
            mla_to_bf16=mla_to_bf16,
            linear_attn_int8=linear_attn_int8,
            kda_int8_scope=kda_int8_scope,
            indexer_int8=indexer_int8,
        ),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


# --------------------------------------------------------------------------- #
# index.json / config.json 更新
# --------------------------------------------------------------------------- #

def _rebuild_index(
    output_dir: str,
    input_index_path: str | None,
    limit_files: int | None,
) -> None:
    """按新分片实际 tensor keys 重建 weight_map + total_size."""
    output_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    model_index: dict = {}
    if os.path.exists(output_index_path):
        with open(output_index_path, "r", encoding="utf-8") as f:
            model_index = json.load(f)
    elif input_index_path and os.path.exists(input_index_path):
        with open(input_index_path, "r", encoding="utf-8") as f:
            model_index = json.load(f)

    files = sorted(glob(os.path.join(output_dir, "*.safetensors")))
    if limit_files is not None:
        files = files[:limit_files]

    new_weight_map: dict[str, str] = {}
    total_size = 0
    for path in files:
        fname = os.path.basename(path)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                new_weight_map[key] = fname
                t = f.get_tensor(key)
                total_size += t.numel() * t.element_size()

    model_index["weight_map"] = new_weight_map
    if "metadata" in model_index and isinstance(model_index["metadata"], dict):
        model_index["metadata"]["total_size"] = total_size
    else:
        model_index["metadata"] = {"total_size": total_size}

    with open(output_index_path, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_metadata(
    input_dir: str,
    output_dir: str,
    limit_files: int | None,
    mla_to_bf16: bool = True,
    linear_attn_int8: bool = False,
    kda_int8_scope: str = "full",
    indexer_int8: bool = False,
) -> None:
    """复制配置文件, 更新 quantization_config, 并重建 index."""
    for fname in [
        "config.json",
        "configuration.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "merges.txt",
        "vocab.json",
        "model.safetensors.index.json",
        "README.md",
        "LICENSE",
    ]:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(output_dir, fname))

    _rebuild_index(
        output_dir,
        os.path.join(input_dir, "model.safetensors.index.json"),
        limit_files,
    )

    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.pop("compression_config", None)
    config.pop("quantization_config", None)
    for k in list(config.keys()):
        v = config[k]
        if isinstance(v, dict) and "quantization_config" in v:
            v.pop("quantization_config", None)

    config["quantization_config"] = build_compression_config(
        mla_to_bf16=mla_to_bf16,
        linear_attn_int8=linear_attn_int8,
        kda_int8_scope=kda_int8_scope,
        indexer_int8=indexer_int8,
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_summary(
    output_dir: str,
    stats: dict[str, int],
    args,
    devices: list[torch.device],
    workers: int,
) -> None:
    if args.linear_attn_int8:
        if args.kda_int8_scope == "qkvo":
            linear_attn_desc = " / linear-attn q/k/v/o_proj (34 layers, qkvo-only, INT8)"
            kda_gate_kept = " / linear-attn b/f_a/g_a/f_b/g_b_proj (kept BF16 by --kda-int8-scope qkvo)"
        else:
            linear_attn_desc = " / linear-attn q/k/v/b/f_a/g_a/f_b/g_b/o_proj (34 layers, INT8)"
            kda_gate_kept = ""
    else:
        linear_attn_desc = ""
        kda_gate_kept = ""

    indexer_desc = " / indexer wq_b/wk (12 DSA layers, INT8)" if args.indexer_int8 else ""
    indexer_kept = (
        "indexer k_norm/weights_proj/index_kpool_compress_*"
        if args.indexer_int8 else "indexer"
    )

    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "limit_files": args.limit_files,
        "compute": {
            "device": args.device,
            "gpu_ids": args.gpu_ids,
            "workers_requested": args.workers,
            "workers_effective": workers,
            "devices_used": [str(d) for d in devices],
        },
        "layout": {
            "int4_targets": (
                "routed experts layer 3-44 (mlp.experts.*.gate/up/down_proj): "
                "INT4 W4A8 channelwise (pack-quantized, int32 packed, per-out-channel bf16 scale)"
            ),
            "int8_targets": (
                "routed experts layer 45 MTP (moe_int8_layers=[45] alignment) / "
                "shared_experts / dense MLP (layer 0-2)"
                + ("" if args.mla_to_bf16 else
                   " / MLA q_a/q_b/kv_a_proj_with_mqa/o_proj")
                + linear_attn_desc
                + indexer_desc
                + ": INT8 W8A8 channelwise (int-quantized, per-out-channel fp32 scale)"
            ),
            "kept_bf16": (
                f"embed / lm_head / norms / hyper-connection / "
                f"MoE router gate / linear-attn A_log/dt_bias/*_conv1d/o_norm / "
                f"MLA kv_b_proj + layernorms / {indexer_kept} / "
                f"MTP special tensors / vision encoder"
                + (" / MLA q_a/q_b/kv_a_proj_with_mqa/o_proj (dequantized "
                   "from FP8, aligned with vLLM GLM5-Next loader)"
                   if args.mla_to_bf16 else "")
                + ("" if args.linear_attn_int8 else
                   " / linear-attn 9 Linear projections")
                + kda_gate_kept
            ),
            "mla_to_bf16": args.mla_to_bf16,
            "linear_attn_int8": args.linear_attn_int8,
            "kda_int8_scope": args.kda_int8_scope,
            "indexer_int8": args.indexer_int8,
            "used_compressed_tensors_pack": _HAVE_COMPRESSED_TENSORS_PACK,
        },
        "stats": stats,
    }
    with open(
        os.path.join(output_dir, "int4_conversion_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = ArgumentParser(
        description=(
            "Convert GLM-5.3-Flash (FP8 E4M3 blockwise 128x128) to mixed-precision "
            "INT4/INT8 (compressed-tensors dual config_groups). Routed experts in "
            "layer 3-44 are quantized to W4A8 (pack-quantized, int32-packed); all "
            "other quantization targets (layer 45 MTP experts / shared_experts / "
            "dense MLP / MLA 4 projections / linear-attn qkvo / DSA indexer) go to "
            "W8A8 (int-quantized). Layout aligned 1:1 with iter_0033200 coverage."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"/mnt/c/chl/models/GLM-5.3-Flash",
        help="Path to GLM-5.3-Flash FP8 checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"/mnt/c/chl/models/GLM-5.3-Flash-CHANNEL-INT4-w4a8",
        help="Path to output mixed-precision INT4/INT8 checkpoint directory.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Torch CPU thread count (per worker in multi-GPU mode).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device: auto (GPU if available else CPU), cpu, or cuda.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device indices, e.g. '0,1,3'. Default: all detected GPUs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes across GPUs. Default: min(#GPUs, #shards).",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Convert only first N shard files (smoke testing).",
    )
    parser.add_argument(
        "--mla-to-bf16",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Dequantize MLA 4 projections (q_a_proj / q_b_proj / "
            "kv_a_proj_with_mqa / o_proj) from FP8 to BF16 instead of quantizing "
            "to INT8. Default: True — required for vLLM / SGLang GLM5-Next inference. "
            "Use --no-mla-to-bf16 to quantize these 4 projections to INT8 W8A8 "
            "(needs custom inference kernels; matches iter_0033200 layout)."
        ),
    )
    parser.add_argument(
        "--linear-attn-int8",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Also quantize the 34 Linear-Attention layers' Linear projections "
            "from BF16 to INT8 channelwise. Scope is controlled by "
            "--kda-int8-scope. Default: False."
        ),
    )
    parser.add_argument(
        "--kda-int8-scope",
        type=str,
        choices=["full", "qkvo"],
        default="full",
        help=(
            "When --linear-attn-int8 is enabled: 'full' covers all 9 KDA projections "
            "(q/k/v/o + b/f_a/g_a/f_b/g_b); 'qkvo' only q/k/v/o_proj (matches "
            "iter_0033200 layout). No effect when --linear-attn-int8 is False."
        ),
    )
    parser.add_argument(
        "--indexer-int8",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Also quantize the DSA layers' indexer.wq_b and indexer.wk from BF16 "
            "to INT8 channelwise (12 layers). Default: False."
        ),
    )
    args = parser.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must be different")

    if args.kda_int8_scope == "qkvo" and not args.linear_attn_int8:
        print(
            "[warn] --kda-int8-scope=qkvo has no effect without "
            "--linear-attn-int8; ignoring."
        )

    if not _HAVE_COMPRESSED_TENSORS_PACK:
        print(
            "[warn] `compressed_tensors` package not installed; falling back to "
            "in-repo pack_to_int32 implementation. Bit ordering has been aligned "
            "with the upstream implementation, but installing the official "
            "package (`pip install compressed-tensors`) is strongly recommended "
            "for production runs."
        )

    torch.set_num_threads(args.num_threads)

    print(f"Converting {args.input_dir} to mixed INT4/INT8 W4A8+W8A8 format...")
    print(f"  output_dir            : {args.output_dir}")
    print(f"  routed experts (INT4) : layer {_ROUTED_EXPERT_INT4_LAYERS[0]}-"
          f"{_ROUTED_EXPERT_INT4_LAYERS[-1]} mlp.experts.*.{{gate,up,down}}_proj")
    print(f"  mla_to_bf16           : {args.mla_to_bf16}")
    print(f"  linear_attn_int8      : {args.linear_attn_int8}")
    print(f"  kda_int8_scope        : {args.kda_int8_scope}")
    print(f"  indexer_int8          : {args.indexer_int8}")
    print(f"  device                : {args.device}"
          + (f" (cuda_available={torch.cuda.is_available()},"
             f" device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0})"))
    if args.gpu_ids:
        print(f"  gpu_ids               : {args.gpu_ids}")
    if args.workers is not None:
        print(f"  workers               : {args.workers}")

    stats, devices, workers = convert_model(
        args.input_dir,
        args.output_dir,
        args.limit_files,
        args.device,
        args.gpu_ids,
        args.workers,
        args.num_threads,
        mla_to_bf16=args.mla_to_bf16,
        linear_attn_int8=args.linear_attn_int8,
        kda_int8_scope=args.kda_int8_scope,
        indexer_int8=args.indexer_int8,
    )
    copy_metadata(
        args.input_dir,
        args.output_dir,
        args.limit_files,
        mla_to_bf16=args.mla_to_bf16,
        linear_attn_int8=args.linear_attn_int8,
        kda_int8_scope=args.kda_int8_scope,
        indexer_int8=args.indexer_int8,
    )
    write_summary(args.output_dir, stats, args, devices, workers)

    print()
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"\nDone! Mixed-precision INT4/INT8 model saved to {args.output_dir}")
    print(f"  devices used : {[str(d) for d in devices]}")
    print(f"  workers used : {workers}")


if __name__ == "__main__":
    main()
