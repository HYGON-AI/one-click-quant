# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash FP4/FP8 block -> INT4 W4A16 / W4A8 转换脚本

流程：
  1. 反量化原始 checkpoint 中的 FP4/FP8 block 权重为 FP32
  2. 主干 routed experts 重新量化为 INT4 (compressed-tensors pack-quantized, int32-lane)
  3. 其余量化层反量化为 BF16;engram.embed 原样透传 FP8
  4. 更新 config.json 和 index 文件

原始模型格式（见 config.json）:
  - quant_method: "fp8", weight_block_size: [32, 32], scale_fmt: "ue8m0"
  - expert_dtype: "fp4"  (routed experts 是 FP4 E2M1FN packed)
  - Attention + Shared Experts + engram.wkv + main_proj: FP8 E4M3FN blockwise 32x32

目标格式：
  - 主干 routed experts (layers.{L}.ffn.experts.{E}.w[123]) -> INT4 compressed-tensors
    pack-quantized: weight_packed (int32, (N, K/8)) + weight_scale (bf16) + weight_shape (int64)
  - MTP experts (mtp.{M}.ffn.experts.{E}.w[123]) -> BF16 (保持精度)
  - Attention MLA (wq_a/wq_b/wkv/wo_a/wo_b/indexer/compressor)
    + shared_experts.w[123] + engram.wkv + main_proj -> BF16
  - engram.embed 原样透传 FP8 (sglang 硬编码格式,保留裸 .scale 键名)
  - W4A16 与 W4A8 权重字节完全一致,只在 config.input_activations 声明差异

用法：
  # W4A16 per-channel (默认)
  python quantize_deepseek_v4_1_flash_int4_w4a16.py \\
      --input-dir /path/to/DeepSeek-V4.1-Flash \\
      --output-dir /path/to/DeepSeek-V4.1-Flash-W4A16

  # W4A16 per-group group_size=128
  python quantize_deepseek_v4_1_flash_int4_w4a16.py \\
      --mode w4a16 --group-size 128 \\
      --input-dir ... --output-dir ...

  # W4A8 per-channel (activation int8 dynamic per-token)
  python quantize_deepseek_v4_1_flash_int4_w4a16.py \\
      --mode w4a8 --input-dir ... --output-dir ...
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

# FP4 E2M1FN 查找表
FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

# DeepSeek-V4.1-Flash 所有量化层 block_size 都是 32x32
FP4_BLOCK_SIZE = 32
FP8_BLOCK_SIZE = 32
INT4_QMAX = 7

# 键名分类正则 (DeepSeek-V4.1-Flash 无 model. 前缀)
ROUTED_EXPERT_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
MTP_EXPERT_RE = re.compile(r"^mtp\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
ENGRAM_EMBED_RE = re.compile(r"^(layers|mtp)\.\d+\.engram\.embed\.")
ENGRAM_WKV_RE = re.compile(r"^(layers|mtp)\.\d+\.engram\.wkv\.")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #

def scale_name_for(weight_name: str) -> str:
    """源 checkpoint 中 scale 的名字 (.weight -> .scale)"""
    return ".".join(weight_name.split(".")[:-1] + ["scale"])


def has_scale(name: str, state_dict: dict[str, torch.Tensor]) -> bool:
    return scale_name_for(name) in state_dict


# --------------------------------------------------------------------------- #
# 反量化: FP4 / FP8 blockwise -> FP32
# --------------------------------------------------------------------------- #

# FP4_TABLE 是查表反量化用的常量,GPU 场景下按 device 缓存一份, 避免每次搬运
_FP4_TABLE_CACHE: dict[torch.device, torch.Tensor] = {}


def _fp4_table_on(device: torch.device) -> torch.Tensor:
    if device not in _FP4_TABLE_CACHE:
        _FP4_TABLE_CACHE[device] = FP4_TABLE.to(device)
    return _FP4_TABLE_CACHE[device]


def unpack_e2m1fn_to_float(x: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 E2M1FN packed tensor (int8) to float32. 每字节 2 个 FP4 (低4/高4)."""
    assert x.dtype == torch.int8
    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    table = _fp4_table_on(x.device)
    return torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(1)


def dequant_fp4_to_float(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP4 blockwise (x: (N, K/2), scale: (N, K/32)) -> FP32 (N, K)."""
    values = unpack_e2m1fn_to_float(x).float()
    expanded_scale = scale.float().repeat_interleave(FP4_BLOCK_SIZE, dim=1)
    return values * expanded_scale


def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    FP8 e4m3fn blockwise (weight (N,K), scale (N/b, K/b)) -> FP32 (N, K).
    根据 scale shape 推断 block size (V4.1-Flash 是 32x32),不整除走 fallback.
    """
    scale = scale.float()

    if weight.ndim != 2 or scale.ndim != 2:
        if weight.numel() % scale.numel() == 0:
            ratio = weight.numel() // scale.numel()
            flat_w = weight.float().view(-1, ratio) if ratio > 1 else weight.float().view(-1, 1)
            flat_s = scale.float().view(-1, 1)
            return (flat_w * flat_s).view_as(weight).float()
        return weight.float() * scale.float().mean()

    out_dim, in_dim = weight.shape
    s_rows, s_cols = scale.shape

    if s_rows == 0 or s_cols == 0 or out_dim % s_rows != 0 or in_dim % s_cols != 0:
        return weight.float() * scale.float().mean()

    row_block = out_dim // s_rows
    col_block = in_dim // s_cols

    weight_blocks = weight.unflatten(0, (s_rows, row_block)).unflatten(2, (s_cols, col_block))
    scale_expanded = scale[:, None, :, None]
    dequantized = weight_blocks.float() * scale_expanded
    return dequantized.flatten(0, 1).flatten(1, 2)


# --------------------------------------------------------------------------- #
# INT4 量化 + int32-lane packing (compressed-tensors pack-quantized 规范)
# --------------------------------------------------------------------------- #

def quantize_int4(
    tensor: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对称 INT4 量化, 值域 [-8, 7], scale = abs_max / 7.
      group_size = -1  -> per-channel (scale shape (N, 1))
      group_size >  0  -> per-group  (scale shape (N, K/g)),要求 K % g == 0
    返回 (q_int8 保存 [-8,7] 有符号值, scale float32).
    """
    assert tensor.ndim == 2, f"expect 2D tensor, got {tensor.shape}"
    N, K = tensor.shape
    w = tensor.float()

    if group_size == -1 or group_size is None:
        abs_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # (N, 1)
        scale = abs_max / INT4_QMAX
        q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
    else:
        assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
        wg = w.view(N, K // group_size, group_size)
        abs_max = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # (N, K/g, 1)
        scale_g = abs_max / INT4_QMAX
        q = torch.round(wg / scale_g).clamp(-8, 7).to(torch.int8)
        q = q.view(N, K)
        scale = scale_g.squeeze(-1)  # (N, K/g)

    return q, scale.to(torch.float32)


def pack_int4_to_int32(q: torch.Tensor) -> torch.Tensor:
    """
    Signed int4 (int8 存储, 值 [-8, 7]) -> compressed-tensors pack-quantized int32.
    沿最后一维 (input dim K) 每 8 个 int4 -> 1 个 int32. K 必须能被 8 整除.

    Bit 布局 (升序, 4-bit slot 连续, 与 compressed_tensors.pack_to_int32 位级等价):
      word = u0 | (u1<<4) | (u2<<8) | (u3<<12) | (u4<<16) | (u5<<20) | (u6<<24) | (u7<<28)
    其中 u_i = q_i + 8 in [0, 15] (signed -> unsigned).

    参考:
      compressed_tensors/compressors/pack_quantized/helpers.py:pack_to_int32
      utils/MoE-Quant/pack_quantized_model.py:pack_to_int32
    """
    assert q.dtype == torch.int8 and q.ndim == 2, f"expect 2D int8, got {q.shape}/{q.dtype}"
    N, K = q.shape
    assert K % 8 == 0, f"K={K} must be divisible by 8"

    # signed [-8, 7] -> unsigned [0, 15]; mask 高位, cast 到 int32
    u = ((q.to(torch.int32) + 8) & 0xF)  # (N, K)
    u = u.view(N, K // 8, 8)  # 每 8 个 nibble 装 1 个 int32
    shifts = torch.arange(8, dtype=torch.int32, device=q.device) * 4  # [0,4,8,...,28]
    packed = (u << shifts).sum(dim=-1).to(torch.int32)  # OR 等价于 SUM (bit slot 不相交)
    return packed.contiguous()


# --------------------------------------------------------------------------- #
# 键名分类
# --------------------------------------------------------------------------- #

def is_routed_expert_weight(name: str) -> bool:
    """主干 routed experts: layers.{L}.ffn.experts.{E}.w[123].weight"""
    return ROUTED_EXPERT_RE.match(name) is not None


def is_mtp_expert_weight(name: str) -> bool:
    """MTP experts: mtp.{M}.ffn.experts.{E}.w[123].weight (dspark)"""
    return MTP_EXPERT_RE.match(name) is not None


def is_engram_embed(name: str) -> bool:
    """engram.embed 是 sglang 硬编码 fp8+e8m0fnu blockwise hash table,原样透传"""
    return ENGRAM_EMBED_RE.match(name) is not None


def is_engram_wkv(name: str) -> bool:
    return ENGRAM_WKV_RE.match(name) is not None


def is_fp8_blockwise_to_bf16(
    name: str, tensor: torch.Tensor, state_dict: dict[str, torch.Tensor]
) -> bool:
    """
    需要 FP8->BF16 反量化的层: attention MLA (wq_a/wq_b/wkv/wo_a/wo_b/indexer/compressor)
    + shared_experts + engram.wkv + main_proj.
    判据: FP8 e4m3fn dtype + .weight 结尾 + 有 .scale 兄弟,且不是 engram.embed.
    """
    if is_engram_embed(name):
        return False
    if tensor.dtype != torch.float8_e4m3fn:
        return False
    if not name.endswith(".weight"):
        return False
    return has_scale(name, state_dict)


# --------------------------------------------------------------------------- #
# 主转换流程
# --------------------------------------------------------------------------- #

def convert_one_file(
    input_path: str,
    output_path: str,
    group_size: int,
    stats: dict[str, int],
    engram_wkv_mode: str = "bf16",
    device: torch.device | str = "cpu",
) -> None:
    """转换单个 safetensors 分片.

    device: 计算所在 device (cpu 或 cuda:N). CPU tensor 从 safetensors 读出后临时搬到
    device 上做 dequant/quantize/pack, 保存前搬回 CPU (save_file 要求 CPU tensor).
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    def _to_dev(t: torch.Tensor) -> torch.Tensor:
        return t if t.device == device else t.to(device, non_blocking=False)

    def _to_cpu(t: torch.Tensor) -> torch.Tensor:
        return t if t.device.type == "cpu" else t.cpu()

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name)

    new_state_dict: dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        # scale 键由 weight 分支处理; 未被处理的 scale 后续判断是否保留
        if name.endswith(".scale"):
            continue

        scale_name = scale_name_for(name)
        prefix = name[: -len(".weight")] if name.endswith(".weight") else name

        # 0. engram.embed: 原样透传 FP8 + 保留裸 .scale (sglang 硬编码)
        if is_engram_embed(name):
            new_state_dict[name] = tensor
            if scale_name in state_dict:
                new_state_dict[scale_name] = state_dict[scale_name]
            stats["engram_embed_passthrough"] += 1
            continue

        # 0a. engram.wkv: 走 FP8->BF16 (默认) 或原样透传
        if is_engram_wkv(name) and name.endswith(".weight"):
            if engram_wkv_mode == "blockwise":
                new_state_dict[name] = tensor
                if scale_name in state_dict:
                    new_state_dict[scale_name] = state_dict[scale_name]
                stats["engram_wkv_passthrough"] += 1
                continue
            if scale_name in state_dict:
                scale = _to_dev(state_dict[scale_name])
                new_state_dict[name] = _to_cpu(
                    dequant_fp8_blockwise(_to_dev(tensor), scale).bfloat16()
                )
            else:
                new_state_dict[name] = tensor.bfloat16() if tensor.dtype == torch.float8_e4m3fn else tensor
            stats["engram_wkv_bf16"] += 1
            continue

        # 1. 主干 routed experts: FP4 -> INT4 pack-quantized
        if is_routed_expert_weight(name):
            if scale_name not in state_dict:
                # 理论上不该发生; 退化为透传避免整份任务崩掉
                new_state_dict[name] = tensor
                stats["routed_expert_no_scale"] += 1
                continue
            scale = _to_dev(state_dict[scale_name])
            weight_fp32 = dequant_fp4_to_float(_to_dev(tensor), scale)
            q_int8, q_scale = quantize_int4(weight_fp32, group_size)
            packed = pack_int4_to_int32(q_int8)
            new_state_dict[f"{prefix}.weight_packed"] = _to_cpu(packed)
            new_state_dict[f"{prefix}.weight_scale"] = _to_cpu(q_scale.to(torch.bfloat16))
            new_state_dict[f"{prefix}.weight_shape"] = torch.tensor(
                list(weight_fp32.shape), dtype=torch.int64
            )
            stats["routed_expert_int4"] += 1
            continue

        # 2. MTP experts: FP4 -> BF16 (不做 INT4)
        if is_mtp_expert_weight(name):
            if scale_name in state_dict:
                scale = _to_dev(state_dict[scale_name])
                weight_fp32 = dequant_fp4_to_float(_to_dev(tensor), scale)
                new_state_dict[name] = _to_cpu(weight_fp32.bfloat16())
            else:
                new_state_dict[name] = tensor
            stats["mtp_expert_bf16"] += 1
            continue

        # 3. Attention / shared_experts / main_proj (FP8 blockwise) -> BF16
        if is_fp8_blockwise_to_bf16(name, tensor, state_dict):
            scale = _to_dev(state_dict[scale_name])
            new_state_dict[name] = _to_cpu(
                dequant_fp8_blockwise(_to_dev(tensor), scale).bfloat16()
            )
            stats["fp8_to_bf16"] += 1
            continue

        # 4. 其它 (BF16 权重, gate, norm, embed, head, vision, attn_sink, ...) 透传
        new_state_dict[name] = tensor
        stats["kept"] += 1

    save_file(new_state_dict, output_path)


# --------------------------------------------------------------------------- #
# 设备解析 + 多 GPU worker
# --------------------------------------------------------------------------- #

_STATS_KEYS = (
    "routed_expert_int4",
    "routed_expert_no_scale",
    "mtp_expert_bf16",
    "fp8_to_bf16",
    "engram_embed_passthrough",
    "engram_wkv_bf16",
    "engram_wkv_passthrough",
    "kept",
)


def _empty_stats() -> dict[str, int]:
    return {k: 0 for k in _STATS_KEYS}


def _resolve_devices(
    device_kind: str,
    gpu_ids: str | None,
    workers: int | None,
    num_files: int,
) -> tuple[list[torch.device], int]:
    """根据 --device / --gpu-ids / --workers 参数解析实际 device 列表与并行度.

    Returns:
        (devices, effective_workers): devices 长度 >=1; effective_workers 是最终并行进程数
        (1 表示主进程顺序执行, >1 表示多进程池).
    """
    cuda_available = torch.cuda.is_available()

    if device_kind == "cpu":
        return [torch.device("cpu")], 1
    if device_kind == "cuda" and not cuda_available:
        raise RuntimeError(
            "--device cuda specified but torch.cuda.is_available() == False"
        )
    if device_kind == "auto" and not cuda_available:
        return [torch.device("cpu")], 1

    # GPU 分支
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
    args: tuple[str, str, int, str, int, int],
) -> dict[str, int]:
    """spawn worker 入口: 绑定一张 GPU, 处理一个 shard, 返回本 shard 的 stats.
    必须是模块顶层函数以支持 multiprocessing.spawn 的 pickle.
    """
    (input_path, output_path, group_size, engram_wkv_mode, gpu_id, num_threads) = args
    torch.set_num_threads(max(1, num_threads))
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    stats = _empty_stats()
    convert_one_file(input_path, output_path, group_size, stats, engram_wkv_mode, device)
    return stats


def convert_model(
    input_dir: str,
    output_dir: str,
    group_size: int,
    limit_files: int | None,
    engram_wkv_mode: str,
    device_kind: str,
    gpu_ids: str | None,
    workers: int | None,
    num_threads: int,
) -> tuple[dict[str, int], list[torch.device], int]:
    """遍历所有 shard, 单卡/CPU 顺序执行或多卡并行执行. 返回 (stats, devices, workers)."""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    if limit_files is not None:
        files = files[:limit_files]

    devices, effective_workers = _resolve_devices(
        device_kind, gpu_ids, workers, len(files)
    )
    stats = _empty_stats()

    # 单进程路径: CPU 或单 GPU
    if effective_workers <= 1 or len(devices) == 1:
        device = devices[0]
        desc = f"Converting on {device}"
        for path in tqdm(files, desc=desc):
            fname = os.path.basename(path)
            convert_one_file(
                path,
                os.path.join(output_dir, fname),
                group_size,
                stats,
                engram_wkv_mode,
                device,
            )
        return stats, devices, effective_workers

    # 多 GPU 并行: 每个 worker 绑定一张 GPU, 通过 mp.Pool + spawn 上下文调度 shard
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")

    tasks: list[tuple[str, str, int, str, int, int]] = []
    for i, path in enumerate(files):
        gpu_id = devices[i % len(devices)].index
        fname = os.path.basename(path)
        tasks.append(
            (
                path,
                os.path.join(output_dir, fname),
                group_size,
                engram_wkv_mode,
                int(gpu_id),
                num_threads,
            )
        )

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
# compressed-tensors config 生成
# --------------------------------------------------------------------------- #

def _weights_arg(group_size: int) -> dict:
    if group_size == -1 or group_size is None:
        strategy = "channel"
        gs = -1
    else:
        strategy = "group"
        gs = int(group_size)
    return {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": gs,
        "num_bits": 4,
        "observer": "minmax",
        "observer_kwargs": {},
        "strategy": strategy,
        "symmetric": True,
        "type": "int",
    }


def _input_activations_arg(mode: str) -> dict | None:
    if mode == "w4a16":
        return None
    if mode == "w4a8":
        return {
            "actorder": None,
            "block_structure": None,
            "dynamic": True,
            "group_size": None,
            "num_bits": 8,
            "observer": "minmax",
            "observer_kwargs": {},
            "strategy": "token",
            "symmetric": True,
            "type": "int",
        }
    raise ValueError(f"unknown mode: {mode}")


def build_ignore_list(engram_wkv_mode: str) -> list[str]:
    """所有非主干-experts 的 Linear/embed 层都要 ignore, 避免 compressed-tensors 误配.

    重要: sglang/vLLM 用运行时 module 名 (带 `model.` 前缀, attention 是 fused `self_attn.wqkv_a`
    而非 checkpoint 里的 `attn.wq_a/wkv`) 去查 quant scheme, 与 checkpoint 键名不同.
    这里的正则统一采用 `re:.*` 前缀通配, 同时覆盖两种命名, 避免出现
    'Unable to find matching target for model.layers.0.self_attn.wqkv_a' 之类的报错.
    """
    ignore = [
        # MTP experts 保持 BF16 (不做 INT4).
        # sglang DSpark draft (models/deepseek_v4_dspark.py:829) 把 MTP 挂在
        # `stages.{N}.mlp.experts.*` 下, should_ignore_layer 用运行时 module
        # 名匹配, 因此必须包含 `stages.*` 前缀; 同时保留 `mtp.*` 以兼容 vLLM
        # 或其它直接按 checkpoint 键实例化的 loader.
        r"re:.*stages\.\d+\..*experts\.\d+\..*",
        r"re:.*mtp\.\d+\..*experts\.\d+\..*",
        # attention: sglang 内部 fused `self_attn.wqkv_a/wq_b/wo_a/wo_b/indexer/compressor`
        # + checkpoint 里的 `attn.wq_a/wq_b/wkv/wo_a/wo_b/indexer/compressor`
        r"re:.*self_attn.*",
        r"re:.*\.attn\..*",
        # shared experts (无论叫 ffn.shared_experts 还是 mlp.shared_experts)
        r"re:.*shared_experts.*",
        # router gate: 精确匹配 module 名以防误伤 experts 的 gate_proj (Flash 用 w1/w2/w3, 无风险,
        # 但同一份 config 想给其它 HF-style checkpoint 复用时,这条保险)
        r"re:.*\.ffn\.gate$",
        r"re:.*\.mlp\.gate$",
        # engram (embed 透传 FP8; wkv 反量化为 BF16 或原样)
        r"re:.*engram\..*",
        # MTP 特殊模块
        r"re:.*main_proj.*",
        r"re:.*markov_head.*",
        r"re:.*confidence_head.*",
        # 视觉 / aligner / 顶层
        r"re:.*vision\..*",
        r"re:.*aligner\..*",
        "lm_head",
        "head",
        "embed",
        "norm",
    ]
    return ignore


def build_compression_config(mode: str, group_size: int, engram_wkv_mode: str) -> dict:
    """生成 compressed-tensors pack-quantized 配置 (W4A16 或 W4A8).

    targets 必须匹配 **推理引擎实例化时的 module 名**, 而不是 checkpoint 键名:
      - sglang deepseek_v4 loader (models/deepseek_v4.py:4565-4579) 会把
        `ffn.` -> `mlp.`, `w1/w2/w3` -> `gate_proj/down_proj/up_proj`;
      - compressed_tensors.py:800-807 get_moe_scheme 拼的 unfused_names 是
        `<experts_prefix>.0.gate_proj/up_proj/down_proj` (HF-style);
      - vLLM 同样用 HF-style 命名.
    这里 targets 同时接受 HF 命名与原始 checkpoint 命名, 两端通用.

    MTP experts 会被 targets 匹配但先被 ignore 拦截, 顺序无所谓 -- sglang/vLLM 的
    get_scheme_dict 都是先查 should_ignore_layer, 再查 find_matched_target.
    """
    return {
        "config_groups": {
            "group_0": {
                "input_activations": _input_activations_arg(mode),
                "output_activations": None,
                "targets": [
                    r"re:.*(?:mlp|ffn)\.experts\.\d+\.(?:gate_proj|up_proj|down_proj|w[123])$",
                ],
                "weights": _weights_arg(group_size),
            }
        },
        "format": "pack-quantized",
        "ignore": build_ignore_list(engram_wkv_mode),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


# --------------------------------------------------------------------------- #
# index.json / config.json 更新
# --------------------------------------------------------------------------- #

def _rebuild_index(
    output_dir: str,
    limit_files: int | None,
) -> None:
    """
    重建 model.safetensors.index.json 的 weight_map:
      读取每个新 shard 的实际 tensor keys, 覆盖旧 weight_map.
    这样自动:
      - 加入主干 experts 新键 (weight_packed / weight_scale / weight_shape)
      - 剔除所有已反量化为 BF16 层的旧 .scale
      - 保留 engram.embed 的 .scale 透传
    """
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return

    with open(index_path, "r", encoding="utf-8") as f:
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

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_metadata(
    input_dir: str,
    output_dir: str,
    mode: str,
    group_size: int,
    engram_wkv_mode: str,
    limit_files: int | None,
) -> None:
    """复制配置文件, 更新 quantization_config, 并重建 index."""
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "configuration.json",
        "model.safetensors.index.json",
        "README.md",
        "LICENSE",
    ]:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(output_dir, fname))

    _rebuild_index(output_dir, limit_files)

    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 移除原始 fp8/fp4 声明,替换为 compressed-tensors
    config.pop("compression_config", None)
    config.pop("quantization_config", None)

    config["quantization_config"] = build_compression_config(mode, group_size, engram_wkv_mode)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_summary(
    output_dir: str,
    stats: dict[str, int],
    args,
    devices: list[torch.device],
    workers: int,
) -> None:
    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "mode": args.mode,
        "group_size": args.group_size,
        "engram_wkv_mode": args.engram_wkv_mode,
        "limit_files": args.limit_files,
        "compute": {
            "device": args.device,
            "gpu_ids": args.gpu_ids,
            "workers_requested": args.workers,
            "workers_effective": workers,
            "devices_used": [str(d) for d in devices],
        },
        "layout": {
            "routed_experts (layers.*.ffn.experts.*.w[123])": "INT4 pack-quantized (int32-lane)",
            "mtp_experts (mtp.*.ffn.experts.*.w[123])": "BF16",
            "attention_mla / shared_experts / main_proj": "BF16",
            "engram.embed": "FP8 e4m3fn + ue8m0 scale (passthrough)",
            "engram.wkv": "BF16 (from FP8)" if args.engram_wkv_mode == "bf16" else "FP8 blockwise (passthrough)",
        },
        "stats": stats,
    }
    with open(os.path.join(output_dir, "int4_conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- #
# 自检: pack_int4_to_int32 位级正确性
# --------------------------------------------------------------------------- #

def _self_check_pack() -> None:
    """确认 pack_int4_to_int32 的位布局符合 compressed-tensors 规范.

    输入   q = [-8, -1, 0, 1, 7, -8, 7, 0]
    +8 -> u = [ 0,  7, 8, 9,15,  0,15, 8]
    word (LSB first, 4-bit slot 升序):
      0 | (7<<4) | (8<<8) | (9<<12) | (15<<16) | (0<<20) | (15<<24) | (8<<28)
      = 0x8F0F9870 (unsigned 2400163952) = -1894803344 (signed int32)
    """
    q = torch.tensor([[-8, -1, 0, 1, 7, -8, 7, 0]], dtype=torch.int8)
    expected = -1894803344
    got = int(pack_int4_to_int32(q).item())
    assert got == expected, f"pack self-check failed: got={got}, expected={expected}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = ArgumentParser(
        description="Convert DeepSeek-V4.1-Flash to INT4 W4A16 / W4A8 (compressed-tensors pack-quantized).",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"/models/DeepSeek-V4.1-Flash",
        help="Path to DeepSeek-V4.1-Flash checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"/models/DeepSeek-V4.1-Flash-W4A16",
        help="Path to output converted checkpoint directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["w4a16", "w4a8"],
        default="w4a16",
        help="Output activation scheme (weight bytes identical, only config differs).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=-1,
        help="INT4 group size along input dim K. -1 = per-channel (default). "
        "Positive values (e.g. 64, 128) enable per-group quantization; K must be divisible.",
    )
    parser.add_argument(
        "--engram-wkv-mode",
        type=str,
        choices=["bf16", "blockwise"],
        default="bf16",
        help="engram.wkv handling: bf16 (dequant, default) or blockwise (passthrough FP8).",
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
        help="Compute device: auto (GPU if torch.cuda.is_available() else CPU), "
        "cpu (force CPU), cuda (require GPU, error if unavailable).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device indices to use, e.g. '0,1,3'. "
        "Default: all detected GPUs (torch.cuda.device_count()).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes across GPUs. "
        "Default: min(#GPUs, #shards). Ignored in CPU / single-GPU mode.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Convert only first N shard files (smoke testing).",
    )
    args = parser.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must be different")

    torch.set_num_threads(args.num_threads)
    _self_check_pack()

    print(f"Converting {args.input_dir} to INT4 {args.mode.upper()} format...")
    print(f"  output_dir      : {args.output_dir}")
    print(f"  mode            : {args.mode}")
    print(f"  group_size      : {args.group_size} ({'per-channel' if args.group_size == -1 else 'per-group'})")
    print(f"  engram_wkv_mode : {args.engram_wkv_mode}")
    print(f"  device          : {args.device}"
          + (f" (cuda_available={torch.cuda.is_available()},"
             f" device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0})"))
    if args.gpu_ids:
        print(f"  gpu_ids         : {args.gpu_ids}")
    if args.workers is not None:
        print(f"  workers         : {args.workers}")

    stats, devices, workers = convert_model(
        args.input_dir,
        args.output_dir,
        args.group_size,
        args.limit_files,
        args.engram_wkv_mode,
        args.device,
        args.gpu_ids,
        args.workers,
        args.num_threads,
    )
    copy_metadata(
        args.input_dir, args.output_dir, args.mode, args.group_size, args.engram_wkv_mode, args.limit_files
    )
    write_summary(args.output_dir, stats, args, devices, workers)

    print()
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"\nDone! INT4 {args.mode.upper()} model saved to {args.output_dir}")
    print(f"  devices used   : {[str(d) for d in devices]}")
    print(f"  workers used   : {workers}")


if __name__ == "__main__":
    main()
