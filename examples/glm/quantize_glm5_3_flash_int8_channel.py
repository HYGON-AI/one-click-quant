# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3-Flash FP8 block -> INT8 W8A8 channelwise 转换脚本

流程：
  1. 遍历源 checkpoint 分片，识别 FP8 E4M3FN blockwise 权重
     （判据：同 state_dict 中存在配对的 `<name>_scale_inv`）
  2. 反量化 FP8 block 权重为 FP32
  3. 重新量化为 INT8 channelwise (per-output-channel, symmetric)
  4. 更新 config.json 和 model.safetensors.index.json

原始模型格式（GLM-5.3-Flash 的 config.json.quantization_config）：
  - quant_method: "fp8"
  - fmt: "e4m3"
  - activation_scheme: "dynamic"
  - weight_block_size: [128, 128]
  - modules_to_not_convert: 大量白名单（norm/vision/linear-attn/indexer/...）

量化目标（与原 FP8 布局 1:1 对齐）：
  ✅ 转 INT8 W8A8：
      - dense MLP (layer 0-2): mlp.gate_proj / up_proj / down_proj
      - routed experts (layer 3-44 + MTP 45): mlp.experts.{0..287}.{gate,up,down}_proj
      - shared experts:                       mlp.shared_experts.{gate,up,down}_proj
      - (仅在 --no-mla-to-bf16 时) MLA 4 投影:
          self_attn.q_a_proj / q_b_proj / kv_a_proj_with_mqa / o_proj
      - (仅在 --linear-attn-int8 时) 34 层 Linear-Attention 的 9 类 2D Linear:
          self_attn.q_proj / k_proj / v_proj / b_proj /
          f_a_proj / g_a_proj / f_b_proj / g_b_proj / o_proj
          源本就是 BF16, 直接 BF16 -> INT8 channelwise 量化.
  🟦 保持 BF16 (透传, 不重量化):
      - embed_tokens / lm_head / model.norm
      - 所有 input_layernorm / post_attention_layernorm
      - hyper-connection: hc_attn_{base,fn,scale} / hc_ffn_{base,fn,scale}
      - MoE 路由: mlp.gate.weight / mlp.gate.e_score_correction_bias
      - Linear-Attention 层的 A_log / dt_bias / *_conv1d / o_norm (vLLM 硬约束)
      - (默认) Linear-Attention 层 9 类 Linear (--linear-attn-int8 可切成 INT8)
      - MLA 辅助: q_a_layernorm / kv_a_layernorm / kv_b_proj
      - Lightning Indexer 全部 (indexer.wq_b / wk / weights_proj / k_norm / index_kpool_*)
      - MTP 专属: eh_proj / enorm / hnorm / shared_head.norm
      - Vision encoder 整个塔 (visual.*)
      - (默认 --mla-to-bf16) MLA 4 投影: q_a_proj / q_b_proj /
        kv_a_proj_with_mqa / o_proj — 反量化到 BF16 落盘, 与 vLLM/SGLang
        GLM5-Next 加载器的硬编码 BF16 假设保持一致.

用法：
  # 单机 CPU
  python quantize_glm5_3_flash_int8_channel.py \\
      --input-dir  /mnt/c/chl/models/GLM-5.3-Flash \\
      --output-dir /mnt/c/chl/models/GLM-5.3-Flash-INT8-W8A8

前置条件：
  - torch >= 2.1 (含 torch.float8_e4m3fn dtype), safetensors, tqdm
  - 源 checkpoint 为 GLM-5.3-Flash 官方 FP8 发布版
    (每个可量化 .weight 均存在配对的 .weight_scale_inv)
"""

from __future__ import annotations

import json
import os
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

INT8_QMAX = 127.0

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
    """输出 checkpoint 中 scale 的名字 (compressed-tensors 期望 .weight_scale)"""
    assert weight_name.endswith(".weight")
    return weight_name[: -len(".weight")] + ".weight_scale"


def is_fp8_blockwise_weight(name: str, tensor: torch.Tensor, state_dict: dict) -> bool:
    """FP8 blockwise 权重的判据: dtype 是 float8_e4m3fn 且存在配对的 _scale_inv"""
    if not name.endswith(".weight"):
        return False
    if tensor.dtype != torch.float8_e4m3fn:
        return False
    return src_scale_name(name) in state_dict


# MLA 投影后缀: vLLM/SGLang 的 GLM5-Next 加载器硬编码把这 4 类 FP8 投影反量化
# 回 BF16 再喂给模型 (q_a_proj + kv_a_proj_with_mqa 融合到 fused_qkv_a_proj,
# 内部固定 BF16; q_b_proj / sparse-attn o_proj 也是 BF16 nn.Linear).
# 因此我们的 INT8 版本对这 4 类默认也反量化到 BF16, 保持与官方 FP8 版在
# vLLM 内存中的实际布局等价, 避免加载器抛 KeyError('...weight_scale').
_MLA_BF16_SUFFIXES = (
    ".self_attn.q_a_proj.weight",
    ".self_attn.q_b_proj.weight",
    ".self_attn.kv_a_proj_with_mqa.weight",
    ".self_attn.o_proj.weight",
)


def is_mla_bf16_target(name: str) -> bool:
    """判断是否属于 MLA 4 个投影 (加载时会被 vLLM 反量化回 BF16)."""
    return any(name.endswith(sfx) for sfx in _MLA_BF16_SUFFIXES)


# Linear-Attention 层 (KDA / GDN) 里 vLLM 侧带 quant_config 的 2D Linear 权重.
# 源 checkpoint 里全部是 BF16, 若启用 --linear-attn-int8 就地 BF16 -> INT8
# channelwise. 排除 A_log / dt_bias / *_conv1d / o_norm — 它们不是 nn.Linear
# 或 params_dtype 硬编码 FP32 (kda.py:228-266), 必须保留原精度.
# 说明: shard 4/5 (f_a/g_a) 在 in_proj_qkvbfg_a 里 replicated, MergedColumn
# 的 per-shard scale 拼接由 compressed-tensors 自动处理.
#
# 按 --kda-int8-scope 拆成两组:
#   QKVO: q/k/v/o_proj 4 类 (与 iter_0033200 对齐时只量化这 4 类)
#   GATE: b/f_a/g_a/f_b/g_b_proj 5 类 (scope="full" 时也量化, "qkvo" 时保 BF16)
_LINEAR_ATTN_INT8_QKVO_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",  # 只命中 linear-attn 层 (BF16); sparse-attn 的
                                 # o_proj 是 FP8, 已由上方 FP8 分支处理.
)
_LINEAR_ATTN_INT8_GATE_SUFFIXES = (
    ".self_attn.b_proj.weight",
    ".self_attn.f_a_proj.weight",
    ".self_attn.g_a_proj.weight",
    ".self_attn.f_b_proj.weight",
    ".self_attn.g_b_proj.weight",
)


def is_linear_attn_int8_target(name: str, scope: str = "full") -> bool:
    """判断是否是 Linear-Attention 层里应被 INT8 量化的 2D Linear 权重.

    scope:
      "full" — q/k/v/o + 5 门控 proj 都算 (与原 --linear-attn-int8 行为一致)
      "qkvo" — 只 q/k/v/o_proj, 5 门控 proj 保 BF16 (与 iter_0033200 对齐)
    """
    if any(name.endswith(sfx) for sfx in _LINEAR_ATTN_INT8_QKVO_SUFFIXES):
        return True
    if scope == "full" and any(
        name.endswith(sfx) for sfx in _LINEAR_ATTN_INT8_GATE_SUFFIXES
    ):
        return True
    return False


# DSA (Deepseek Sparse Attention) 层 indexer 里的两个 2D Linear 权重.
# 源 checkpoint 里是 BF16 (config.json.modules_to_not_convert 逐层列出),
# 若启用 --indexer-int8 就地 BF16 -> INT8 channelwise. 只覆盖 12 个 DSA 层
# (full_attn_layers=[3,7,11,15,19,23,27,31,35,39,43] + MTP 45).
# 注: wq_b 走 vLLM 的 ReplicatedLinear + quant_config, 可直接加载;
#     wk 需要自行补 vLLM patch (attention.py:259-266 wk_weights_proj 目前
#     硬编码 quant_config=None, model.py:1140-1167 _try_load_fp8_indexer_wk
#     只识 FP8 weight_scale_inv).
_INDEXER_INT8_SUFFIXES = (
    ".self_attn.indexer.wq_b.weight",
    ".self_attn.indexer.wk.weight",
)


def is_indexer_int8_target(name: str) -> bool:
    """判断是否是 DSA indexer 里应被 INT8 量化的 2D Linear 权重."""
    return any(name.endswith(sfx) for sfx in _INDEXER_INT8_SUFFIXES)


# --------------------------------------------------------------------------- #
# FP8 blockwise 反量化 + INT8 channelwise 重量化
# --------------------------------------------------------------------------- #

def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    反量化 FP8 blockwise (128x128) 权重为 float32.
      weight: FP8 E4M3FN,   shape (out_dim, in_dim)
      scale:  float32/bf16, shape (ceil(out_dim/128), ceil(in_dim/128))

    GLM-5.3-Flash 采用与 DeepSeek 相同的对称 blockwise 布局: 每个 128x128 tile
    共享一个 scale, 无需 unpack. 边界 tile 允许非 128 的余数 (DeepSeek 规范).
    """
    assert weight.ndim == 2, f"expect 2D weight, got {weight.shape}"
    assert scale.ndim == 2, f"expect 2D scale, got {scale.shape}"

    scale = scale.float()
    w = weight.float()
    out_dim, in_dim = w.shape
    s_rows, s_cols = scale.shape

    # 若能整除, 走高效路径 (与 DeepSeek 脚本相同)
    if s_rows > 0 and s_cols > 0 and out_dim % s_rows == 0 and in_dim % s_cols == 0:
        row_block = out_dim // s_rows
        col_block = in_dim // s_cols
        w_blocks = w.unflatten(0, (s_rows, row_block)).unflatten(2, (s_cols, col_block))
        expanded = scale[:, None, :, None]
        return (w_blocks * expanded).flatten(0, 1).flatten(1, 2)

    # 边界余数场景: 逐 tile 缩放 (推导 tile 大小 = 官方 block_size)
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


# --------------------------------------------------------------------------- #
# 单 shard 转换
# --------------------------------------------------------------------------- #

_STATS_KEYS = (
    "fp8_weights_converted",       # 反量化 + INT8 重量化的 FP8 权重数
    "mla_dequant_to_bf16",         # MLA 4 投影反量化到 BF16 透传的权重数
    "linear_attn_int8_quantized",  # Linear-Attention 层 BF16 -> INT8 量化的权重数
    "indexer_int8_quantized",      # DSA indexer wq_b/wk BF16 -> INT8 量化的权重数
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
    """转换单个 safetensors 分片.

    Args:
        mla_to_bf16: True (默认) 时, MLA 的 4 个 FP8 投影
            (q_a_proj / q_b_proj / kv_a_proj_with_mqa / o_proj) 反量化到
            BF16 落盘, 而不是重量化为 INT8. 这与 vLLM/SGLang GLM5-Next
            加载器硬编码假设 (这些投影必须是 BF16) 保持一致.
            设 False 会把这 4 类也走 INT8 W8A8 路径, 需要自行修改推理侧代码.
        linear_attn_int8: True 时, 34 层 Linear-Attention 的 KDA 投影
            (q/k/v/o + 可选 b/f_a/g_a/f_b/g_b) 从源 BF16 就地量化为 INT8
            channelwise. 默认 False 保持与原 FP8 布局一致.
        kda_int8_scope: "full" (默认) 时 linear_attn_int8=True 覆盖全部 9 类
            (q/k/v/o + b/f_a/g_a/f_b/g_b); "qkvo" 时只覆盖 q/k/v/o 4 类,
            5 门控 proj 保 BF16 (与 iter_0033200 对齐). linear_attn_int8=False
            时此参数无效.
        indexer_int8: True 时, 12 个 DSA 层 (full_attn_layers=[3,7,...,43] +
            MTP 45) 的 self_attn.indexer.wq_b/wk 从源 BF16 就地量化为 INT8
            channelwise. 默认 False 保持与原 FP8 布局一致. 注: wk 走 INT8
            在当前 vLLM glm53flash 分支需自行补 loader patch
            (attention.py:259-266 wk_weights_proj 目前硬编码 quant_config=None,
            model.py:1140-1167 _try_load_fp8_indexer_wk 只识 FP8 weight_scale_inv).
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name)

    new_state_dict: dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        # 源 scale 单独处理: 直接丢弃 (稍后按 INT8 重新写入 .weight_scale)
        if name.endswith(".weight_scale_inv"):
            stats["scales_dropped"] += 1
            continue

        # MLA 4 投影: 反量化到 BF16 透传 (不重量化为 INT8)
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

        # FP8 blockwise 权重: 反量化 -> INT8 channelwise
        if is_fp8_blockwise_weight(name, tensor, state_dict):
            scale = state_dict[src_scale_name(name)]

            # shape 校验: 若不匹配则原样透传 (保守 fallback)
            out_dim, in_dim = tensor.shape
            s_rows, s_cols = scale.shape if scale.ndim == 2 else (0, 0)
            if scale.ndim != 2 or s_rows == 0 or s_cols == 0:
                print(f"[skip non-2D scale] {name}: w={tuple(tensor.shape)}, "
                      f"s={tuple(scale.shape)}")
                new_state_dict[name] = tensor
                new_state_dict[src_scale_name(name)] = scale  # 保留原 scale
                stats["skipped_shape_mismatch"] += 1
                stats["scales_dropped"] -= 1  # 未真正丢弃
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

        # Linear-Attention 层 2D Linear (源 BF16) -> INT8 channelwise.
        # dtype/ndim 双重护栏: conv1d 后缀不在列表中已跳过; A_log/dt_bias 是
        # 非 .weight 后缀; 保险起见再校验一次.
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

        # DSA indexer 里的 wq_b / wk (源 BF16) -> INT8 channelwise.
        # 与 linear-attn 分支同型 (源即 BF16, 无 FP8 blockwise scale).
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

        # 其他张量: 原样透传 (BF16 权重 / norm / bias / gate / MTP 特殊张量 / vision / ...)
        new_state_dict[name] = tensor
        stats["kept"] += 1

    save_file(new_state_dict, output_path)


# --------------------------------------------------------------------------- #
# 设备解析 + 多 GPU worker (与 qwen 脚本同结构)
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
# compressed-tensors config 生成 (INT8 W8A8 channelwise)
# --------------------------------------------------------------------------- #

def build_ignore_list(
    mla_to_bf16: bool = True,
    linear_attn_int8: bool = False,
    kda_int8_scope: str = "full",
    indexer_int8: bool = False,
) -> list[str]:
    """
    ignore 列表 (与 GLM-5.3-Flash 原 FP8 modules_to_not_convert 语义一致).

    覆盖所有 BF16/FP32 保留张量, 与原 FP8 布局 1:1 对齐:
      - embed_tokens / lm_head / model.norm
      - 每层 input_layernorm / post_attention_layernorm
      - hyper-connection (hc_attn_* / hc_ffn_*)
      - MoE router gate (含 e_score_correction_bias)
      - Linear-Attention 层的 A_log / dt_bias / *_conv1d / o_norm (vLLM 硬约束
        必须保 BF16/FP32, 参见 kda.py:228-266). 其余 9 类 Linear 默认也保
        BF16, 在 linear_attn_int8=True 时切成 INT8 (从 ignore 中移出);
        kda_int8_scope="qkvo" 时 b/f_a/g_a/f_b/g_b 仍保 BF16.
      - MLA 辅助模块: q_a_layernorm / kv_a_layernorm / kv_b_proj
      - Lightning Indexer (self_attn.indexer.*): 默认整棵保 BF16;
        indexer_int8=True 时 wq_b/wk 切 INT8, 其余仍保 BF16.
      - MTP 专属模块 (eh_proj / enorm / hnorm / shared_head.norm)
      - Vision encoder 整个塔 (model.visual.*)
      - (mla_to_bf16=True 时) MLA 4 投影: q_a_proj / q_b_proj /
        kv_a_proj_with_mqa / o_proj — 与 vLLM GLM5-Next 加载器硬编码 BF16
        假设保持一致.

    routed / shared experts 与 dense MLP 的 gate/up/down_proj 不列入 ignore.
    """
    # Linear-Attention 层里"必须保 BF16/FP32" 的子模块 (vLLM 硬约束)
    _LINEAR_ATTN_KEEP_ALWAYS = [
        r"re:.*self_attn\.A_log$",
        r"re:.*self_attn\.dt_bias$",
        r"re:.*self_attn\.k_conv1d$",
        r"re:.*self_attn\.q_conv1d$",
        r"re:.*self_attn\.v_conv1d$",
        r"re:.*self_attn\.o_norm$",
    ]
    # Linear-Attention 层的 Linear 拆两组, 与 is_linear_attn_int8_target 的
    # QKVO / GATE 分组一一对应:
    #   QKVO (q/k/v/o + 融合前 qkv_proj/fused_qkvbfg_a_proj 白名单名):
    #     linear_attn_int8=True 时永远从 ignore 移出.
    #   GATE (b/f_a/f_b/g_a/g_b): kda_int8_scope="full" 时移出,
    #     "qkvo" 时保留在 ignore (对齐 iter_0033200).
    # 注: fused_qkvbfg_a_proj / qkv_proj 是官方 config.json 里的 "融合前"
    # 白名单名, 实际 checkpoint 张量分别叫 q/k/v/b/f_a/g_a_proj, 由 vLLM
    # 加载时合成; 它们随 QKVO 分组走.
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
        # 常规
        "lm_head",
        r"re:.*embed_tokens.*",
        r"re:^model\.language_model\.norm$",
        r"re:^model\.norm$",

        # 每层 layernorm
        r"re:.*layers\.\d+\.(input_layernorm|post_attention_layernorm)$",

        # hyper-connection
        r"re:.*layers\.\d+\.hc_(attn|ffn)_(base|fn|scale)$",

        # MoE router gate (含 gate.e_score_correction_bias)
        r"re:.*mlp\.gate$",
        r"re:.*mlp\.gate\..*",
    ]

    # Linear-Attention 层里 vLLM 硬约束必须 BF16/FP32 的子模块 — 总是 ignore
    ignore.extend(_LINEAR_ATTN_KEEP_ALWAYS)

    # Linear-Attention 层 Linear ignore 由 linear_attn_int8 + kda_int8_scope 决定
    if not linear_attn_int8:
        # 全部 9 proj 保 BF16 (含 QKVO + GATE)
        ignore.extend(_LINEAR_ATTN_PROJ_QKVO)
        ignore.extend(_LINEAR_ATTN_PROJ_GATE)
    elif kda_int8_scope == "qkvo":
        # QKVO 走 INT8, GATE 仍保 BF16 (对齐 iter_0033200)
        ignore.extend(_LINEAR_ATTN_PROJ_GATE)
    # kda_int8_scope=="full": QKVO + GATE 都走 INT8, 两组都不 ignore

    ignore.extend([
        # MLA 辅助 (仅 sparse-attn 层存在, 保 BF16)
        r"re:.*self_attn\.q_a_layernorm$",
        r"re:.*self_attn\.kv_a_layernorm$",
        r"re:.*self_attn\.kv_b_proj$",
    ])

    # Lightning Indexer: 默认整棵 BF16; indexer_int8=True 时只 ignore
    # wq_b/wk 之外的子模块, 让 wq_b/wk 走 INT8 (对齐 iter_0033200).
    # 注: vLLM 侧 wq_b 可直接加载 INT8; wk 需自行补 loader patch
    # (attention.py:259-266 wk_weights_proj 目前硬编码 quant_config=None).
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
        # MTP 专属 (layer 45)
        r"re:.*layers\.\d+\.eh_proj$",
        r"re:.*layers\.\d+\.enorm$",
        r"re:.*layers\.\d+\.hnorm$",
        r"re:.*layers\.\d+\.shared_head\..*",

        # Vision encoder 整个塔
        r"re:.*\.visual\..*",
        r"re:^model\.visual\..*",
        r"re:^visual\..*",
    ])

    if mla_to_bf16:
        # MLA 4 投影反量化到 BF16 落盘, 加载侧看到的是 BF16 nn.Linear,
        # 显式列入 ignore 避免 compressed-tensors 误认作 INT8 层.
        # 说明: linear-attn 层的 o_proj 本就 BF16 (无 weight_scale), 一并 ignore
        # 是幂等的; sparse-attn 层的 o_proj 在源 checkpoint 是 FP8 但会被
        # vLLM 反量化, 我们也做同样处理.
        # 注: 当 linear_attn_int8=True 时不要 ignore o_proj (linear-attn 的
        # o_proj 已量化为 INT8). MLA 的 sparse-attn o_proj 走反量化 BF16 分支
        # 落盘后, 仍需 ignore 才能被加载器识别为 BF16 层. 需精确到 layer 号.
        ignore.extend([
            r"re:.*self_attn\.q_a_proj$",
            r"re:.*self_attn\.q_b_proj$",
            r"re:.*self_attn\.kv_a_proj_with_mqa$",
        ])
        # o_proj 特殊处理: 若同时启用 linear_attn_int8, 用层号精确匹配
        # sparse-attn 层 (3,7,11,...,43,45) 而不通配所有 self_attn.o_proj.
        if linear_attn_int8:
            ignore.append(
                r"re:.*layers\.(3|7|11|15|19|23|27|31|35|39|43|45)\.self_attn\.o_proj$"
            )
        else:
            ignore.append(r"re:.*self_attn\.o_proj$")

    return ignore


def build_compression_config(
    mla_to_bf16: bool = True,
    linear_attn_int8: bool = False,
    kda_int8_scope: str = "full",
    indexer_int8: bool = False,
) -> dict:
    """生成 compressed-tensors int-quantized W8A8 channelwise 配置."""
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "output_activations": None,
                "weights": {
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "int-quantized",
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

    # 清理旧的 FP8 quantization_config / compression_config
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
            linear_attn_desc = " / linear-attn q/k/v/o_proj (34 layers, qkvo-only)"
            kda_gate_kept = " / linear-attn b/f_a/g_a/f_b/g_b_proj (kept BF16 by --kda-int8-scope qkvo)"
        else:
            linear_attn_desc = " / linear-attn q/k/v/b/f_a/g_a/f_b/g_b/o_proj (34 layers)"
            kda_gate_kept = ""
    else:
        linear_attn_desc = ""
        kda_gate_kept = ""

    indexer_desc = " / indexer wq_b/wk (12 DSA layers)" if args.indexer_int8 else ""
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
            "quantized": ("INT8 W8A8 channelwise (per-output-channel weights, "
                          "per-token dynamic INT8 activations)"),
            "targets": (
                "dense MLP / routed experts / shared experts"
                + ("" if args.mla_to_bf16 else
                   " / MLA q_a/q_b/kv_a_proj_with_mqa/o_proj")
                + linear_attn_desc
                + indexer_desc
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
        },
        "stats": stats,
    }
    with open(
        os.path.join(output_dir, "int8_conversion_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = ArgumentParser(
        description=(
            "Convert GLM-5.3-Flash (FP8 E4M3 blockwise 128x128) to INT8 W8A8 "
            "channelwise (compressed-tensors int-quantized). Quantization targets "
            "are aligned 1:1 with the source FP8 layout via weight_scale_inv pairing."
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
        default=r"/mnt/c/chl/models/GLM-5.3-Flash-INT8-W8A8",
        help="Path to output INT8 W8A8 checkpoint directory.",
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
        help="Comma-separated CUDA device indices, e.g. '0,1,3'. "
        "Default: all detected GPUs.",
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
            "kv_a_proj_with_mqa / o_proj) from FP8 to BF16 instead of "
            "requantizing to INT8. Default: True — required for vLLM / SGLang "
            "GLM5-Next inference, whose loader hardcodes these projections as "
            "BF16 (q_a_proj + kv_a_proj_with_mqa are fused into "
            "fused_qkv_a_proj which is BF16-only; q_b_proj and sparse-attn "
            "o_proj are excluded via modules_to_not_convert). Use "
            "--no-mla-to-bf16 to force INT8 W8A8 on these 4 projections "
            "(requires custom inference kernels)."
        ),
    )
    parser.add_argument(
        "--linear-attn-int8",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Also quantize the 34 Linear-Attention layers' 9 Linear "
            "projections (q/k/v/b/f_a/g_a/f_b/g_b/o_proj) from BF16 to INT8 "
            "channelwise. Skips A_log / dt_bias / q_conv1d / k_conv1d / "
            "v_conv1d / o_norm (kept BF16/FP32 by vLLM contract, see "
            "kda.py:228-266). Default: False (matches original FP8 layout). "
            "Enable for extra compression at some potential accuracy cost — "
            "verify with eval before deploying."
        ),
    )
    parser.add_argument(
        "--kda-int8-scope",
        type=str,
        choices=["full", "qkvo"],
        default="full",
        help=(
            "When --linear-attn-int8 is enabled, choose which KDA projections "
            "to quantize. 'full' (default) covers all 9 (q/k/v/o + "
            "b/f_a/g_a/f_b/g_b). 'qkvo' only covers q/k/v/o_proj (4 per "
            "layer), keeping the 5 gate projections in BF16 — matches the "
            "iter_0033200 quantization layout. No effect when "
            "--linear-attn-int8 is False."
        ),
    )
    parser.add_argument(
        "--indexer-int8",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Also quantize the DSA layers' indexer.wq_b and indexer.wk from "
            "BF16 to INT8 channelwise (12 layers: full_attn_layers=[3,7,11,"
            "15,19,23,27,31,35,39,43] + MTP 45, 2 projections each, 24 "
            "tensors total). Other indexer submodules (k_norm / weights_proj "
            "/ index_kpool_compress_*) remain BF16. Default: False. "
            "IMPORTANT: while indexer.wq_b (ReplicatedLinear + quant_config) "
            "loads out-of-the-box, indexer.wk INT8 needs a self-supplied "
            "vLLM loader patch on this glm53flash branch — attention.py:"
            "259-266 hardcodes quant_config=None for wk_weights_proj, and "
            "model.py:1140-1167 _try_load_fp8_indexer_wk only recognizes "
            "FP8 weight_scale_inv. Enable this only after you have that "
            "patch ready, or use it purely to reproduce iter_0033200's "
            "on-disk layout."
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

    torch.set_num_threads(args.num_threads)

    print(f"Converting {args.input_dir} to INT8 W8A8 channelwise format...")
    print(f"  output_dir       : {args.output_dir}")
    print(f"  mla_to_bf16      : {args.mla_to_bf16}")
    print(f"  linear_attn_int8 : {args.linear_attn_int8}")
    print(f"  kda_int8_scope   : {args.kda_int8_scope}")
    print(f"  indexer_int8     : {args.indexer_int8}")
    print(f"  device           : {args.device}"
          + (f" (cuda_available={torch.cuda.is_available()},"
             f" device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0})"))
    if args.gpu_ids:
        print(f"  gpu_ids          : {args.gpu_ids}")
    if args.workers is not None:
        print(f"  workers          : {args.workers}")

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
    print(f"\nDone! INT8 W8A8 model saved to {args.output_dir}")
    print(f"  devices used : {[str(d) for d in devices]}")
    print(f"  workers used : {workers}")


if __name__ == "__main__":
    main()
