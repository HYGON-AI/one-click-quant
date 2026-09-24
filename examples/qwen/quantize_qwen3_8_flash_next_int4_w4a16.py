# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8-Flash-Next BF16 -> INT4 W4A16 / W4A8 转换脚本

流程：
  1. 遍历源 checkpoint 分片，识别 routed experts (packed 3D nn.Parameter)
  2. 主干与 MTP 层的 routed experts:
       - mlp.experts.gate_up_proj  [E, 2*I, H] -> 拆 gate/up -> 逐 expert INT4 pack
       - mlp.experts.down_proj     [E, H, I]   -> 逐 expert INT4 pack
     输出为 compressed-tensors pack-quantized HF-style per-expert 2D 布局，与
     /mnt/c/chl/models/Qwen3.8-Flash-Next-FP8 的键名约定完全一致。
  3. 其余键 (linear_attn / self_attn / shared_expert / hyper_connection / gate
     / vision / norm / embed / lm_head / mtp non-expert) 原样透传（源全 BF16）。
  4. 更新 config.json (写入 compressed-tensors quantization_config) 与
     model.safetensors.index.json (自动重建 weight_map).

W4A16 与 W4A8 权重字节完全一致，只在 quantization_config.config_groups.
group_0.input_activations 上有差异。

用法：
  # W4A16 per-channel (默认)
  python quantize_qwen3_8_flash_next_int4_w4a16.py \\
      --input-dir /mnt/c/chl/models/Qwen3.8-Flash-Next \\
      --output-dir /mnt/c/chl/models/Qwen3.8-Flash-Next-W4A16

  # W4A16 per-group group_size=128
  python quantize_qwen3_8_flash_next_int4_w4a16.py \\
      --mode w4a16 --group-size 128 --input-dir ... --output-dir ...

  # W4A8 per-channel (activation int8 dynamic per-token)
  python quantize_qwen3_8_flash_next_int4_w4a16.py \\
      --mode w4a8 --input-dir ... --output-dir ...

  # gate_up_proj 拆分顺序 (默认 gate_first, 若与 FP8 参考自检不符改 up_first)
  python quantize_qwen3_8_flash_next_int4_w4a16.py \\
      --gate-up-order up_first --input-dir ... --output-dir ...

前置条件：
  - torch >= 2.1, safetensors, tqdm
  - 源 checkpoint 为原生 BF16 (无 quantization_config), packed 3D experts
    (mlp.experts.gate_up_proj / mlp.experts.down_proj 无 .weight 后缀)
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

INT4_QMAX = 7


# --------------------------------------------------------------------------- #
# 键名分类正则 (Qwen3.8-Flash-Next 主干与 MTP 使用相同 packed 3D 命名)
# --------------------------------------------------------------------------- #

ROUTED_GATE_UP_RE = re.compile(
    r"^(?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts\.gate_up_proj$"
)
ROUTED_DOWN_RE = re.compile(
    r"^(?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts\.down_proj$"
)


def is_routed_gate_up(name: str) -> bool:
    return ROUTED_GATE_UP_RE.match(name) is not None


def is_routed_down(name: str) -> bool:
    return ROUTED_DOWN_RE.match(name) is not None


# --------------------------------------------------------------------------- #
# INT4 量化 + int32-lane packing (compressed-tensors pack-quantized 规范)
# --------------------------------------------------------------------------- #

def quantize_int4(
    tensor: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对称 INT4 量化, 值域 [-8, 7], scale = abs_max / 7.
      group_size = -1  -> per-channel (scale shape (N, 1))
      group_size >  0  -> per-group  (scale shape (N, K/g)), 要求 K % g == 0
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

    u = ((q.to(torch.int32) + 8) & 0xF)  # (N, K)
    u = u.view(N, K // 8, 8)             # 每 8 个 nibble 装 1 个 int32
    shifts = torch.arange(8, dtype=torch.int32, device=q.device) * 4  # [0,4,...,28]
    packed = (u << shifts).sum(dim=-1).to(torch.int32)  # bit slot 不相交, SUM==OR
    return packed.contiguous()


# --------------------------------------------------------------------------- #
# packed 3D experts 拆分 + INT4 pack
# --------------------------------------------------------------------------- #

def _emit_int4_pack_kv(
    w2: torch.Tensor,
    base: str,
    group_size: int,
) -> list[tuple[str, torch.Tensor]]:
    """对一个 2D 权重做 INT4 量化 + pack, 返回 compressed-tensors 需要的 3 个键."""
    assert w2.ndim == 2, f"expect 2D, got {w2.shape}"
    q_int8, scale = quantize_int4(w2.float(), group_size)
    packed = pack_int4_to_int32(q_int8)
    return [
        (f"{base}.weight_packed", packed.cpu().contiguous()),
        (f"{base}.weight_scale", scale.to(torch.bfloat16).cpu().contiguous()),
        (f"{base}.weight_shape", torch.tensor(list(w2.shape), dtype=torch.int64)),
    ]


def _quant_pack_batched_3d(
    tensor_3d: torch.Tensor,
    group_size: int,
    chunk_experts: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    批量 INT4 量化 + pack 一个 [E, out, in] 张量.
    每次处理 chunk_experts 个 expert (默认 64) 以限制峰值显存:
      峰值 ~ chunk_experts * out * in * 4 bytes (fp32 中间) x ~4 份
    E=512, out=640, in=2560, chunk=64 时峰值 ~ 1.6 GB.
    整层结束后返回完整 3D 张量.

    返回:
      packed_3d: [E, out, in/8] int32
      scale_3d:  [E, out, 1 或 in/group] float32
    """
    assert tensor_3d.ndim == 3, f"expect 3D, got {tensor_3d.shape}"
    E, out, in_ = tensor_3d.shape
    packed_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    for start in range(0, E, chunk_experts):
        end = min(start + chunk_experts, E)
        w_chunk = tensor_3d[start:end].contiguous().view((end - start) * out, in_).float()
        q_int8, scale = quantize_int4(w_chunk, group_size)          # [(end-start)*out, ...]
        packed = pack_int4_to_int32(q_int8)                         # [(end-start)*out, in/8]
        packed_chunks.append(packed.view(end - start, out, in_ // 8).cpu().contiguous())
        scale_chunks.append(scale.view(end - start, out, -1).cpu().contiguous())
        # 主动清理避免中间张量堆积
        del w_chunk, q_int8, scale, packed
    packed_3d = torch.cat(packed_chunks, dim=0)
    scale_3d = torch.cat(scale_chunks, dim=0)
    return packed_3d, scale_3d


def _emit_per_expert_pack_kv_from_3d(
    packed_3d: torch.Tensor,
    scale_3d: torch.Tensor,
    tensor_shape_2d: tuple[int, int],
    experts_prefix: str,
    proj_name: str,
) -> list[tuple[str, torch.Tensor]]:
    """把批量量化好的 3D 输出拆成 E 个 per-expert 键."""
    E = packed_3d.shape[0]
    shape_tensor = torch.tensor(list(tensor_shape_2d), dtype=torch.int64)
    out: list[tuple[str, torch.Tensor]] = []
    for e in range(E):
        base = f"{experts_prefix}.{e}.{proj_name}"
        out.append((f"{base}.weight_packed", packed_3d[e].contiguous().cpu()))
        out.append((f"{base}.weight_scale",
                    scale_3d[e].to(torch.bfloat16).contiguous().cpu()))
        out.append((f"{base}.weight_shape", shape_tensor.clone()))
    return out


def split_and_quant_gate_up(
    tensor_3d: torch.Tensor,
    experts_prefix: str,
    group_size: int,
    gate_up_order: str,
) -> list[tuple[str, torch.Tensor]]:
    """
    tensor_3d: [E, 2*I, H]  (I = moe_intermediate_size, H = hidden_size)
    experts_prefix: e.g. "model.language_model.layers.0.mlp.experts"
    gate_up_order: "gate_first" (前 I 是 gate, 后 I 是 up) 或 "up_first"

    输出: 每个 expert 3 个键 x 2 (gate + up), 共 E*6 项.
    使用批量量化, 全 E 个 expert 一次矩阵化 quantize_int4 + pack_int4_to_int32.
    """
    assert tensor_3d.ndim == 3, f"expect 3D, got {tensor_3d.shape}"
    E, two_I, H = tensor_3d.shape
    assert two_I % 2 == 0, f"gate_up out_dim={two_I} not divisible by 2"
    I = two_I // 2

    if gate_up_order == "gate_first":
        gate_slice, up_slice = tensor_3d[:, :I, :], tensor_3d[:, I:, :]
    elif gate_up_order == "up_first":
        up_slice, gate_slice = tensor_3d[:, :I, :], tensor_3d[:, I:, :]
    else:
        raise ValueError(f"unknown gate_up_order: {gate_up_order}")

    out: list[tuple[str, torch.Tensor]] = []
    for name, w3 in (("gate_proj", gate_slice), ("up_proj", up_slice)):
        # w3: [E, I, H]  批量量化 + pack
        packed_3d, scale_3d = _quant_pack_batched_3d(w3, group_size)
        out.extend(
            _emit_per_expert_pack_kv_from_3d(
                packed_3d, scale_3d, (I, H), experts_prefix, name
            )
        )
    return out


def split_and_quant_down(
    tensor_3d: torch.Tensor,
    experts_prefix: str,
    group_size: int,
) -> list[tuple[str, torch.Tensor]]:
    """
    tensor_3d: [E, H, I]  (下投影)
    experts_prefix: e.g. "model.language_model.layers.0.mlp.experts"
    """
    assert tensor_3d.ndim == 3, f"expect 3D, got {tensor_3d.shape}"
    E, H, I = tensor_3d.shape
    packed_3d, scale_3d = _quant_pack_batched_3d(tensor_3d, group_size)
    return _emit_per_expert_pack_kv_from_3d(
        packed_3d, scale_3d, (H, I), experts_prefix, "down_proj"
    )


# --------------------------------------------------------------------------- #
# 主转换流程
# --------------------------------------------------------------------------- #

_STATS_KEYS = (
    "routed_gate_up_experts_quantized",   # E * 2 (gate+up) 计数
    "routed_down_experts_quantized",      # E 计数
    "gate_up_tensors_processed",          # packed 3D 张量数 (每层 1)
    "down_tensors_processed",             # packed 3D 张量数 (每层 1)
    "kept",                               # 透传键数
)


def _empty_stats() -> dict[str, int]:
    return {k: 0 for k in _STATS_KEYS}


def convert_one_file(
    input_path: str,
    output_path: str,
    group_size: int,
    gate_up_order: str,
    stats: dict[str, int],
    device: torch.device | str = "cpu",
) -> None:
    """转换单个 safetensors 分片."""
    device = torch.device(device) if not isinstance(device, torch.device) else device

    def _to_dev(t: torch.Tensor) -> torch.Tensor:
        return t if t.device == device else t.to(device, non_blocking=False)

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name)

    new_state_dict: dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        # 1. routed experts gate_up_proj (packed 3D) -> 拆 + INT4
        if is_routed_gate_up(name):
            experts_prefix = name[: -len(".gate_up_proj")]  # 去掉尾部
            kv = split_and_quant_gate_up(
                _to_dev(tensor), experts_prefix, group_size, gate_up_order
            )
            for k, v in kv:
                new_state_dict[k] = v
            stats["gate_up_tensors_processed"] += 1
            stats["routed_gate_up_experts_quantized"] += tensor.shape[0] * 2
            continue

        # 2. routed experts down_proj (packed 3D) -> 拆 + INT4
        if is_routed_down(name):
            experts_prefix = name[: -len(".down_proj")]
            kv = split_and_quant_down(_to_dev(tensor), experts_prefix, group_size)
            for k, v in kv:
                new_state_dict[k] = v
            stats["down_tensors_processed"] += 1
            stats["routed_down_experts_quantized"] += tensor.shape[0]
            continue

        # 3. 其它键 (linear_attn / self_attn / shared_expert / hc / gate / vision
        #           / norm / embed / lm_head / mtp non-expert) -> 透传
        new_state_dict[name] = tensor
        stats["kept"] += 1

    save_file(new_state_dict, output_path)


# --------------------------------------------------------------------------- #
# 设备解析 + 多 GPU worker
# --------------------------------------------------------------------------- #

def _resolve_devices(
    device_kind: str,
    gpu_ids: str | None,
    workers: int | None,
    num_files: int,
) -> tuple[list[torch.device], int]:
    """根据 --device / --gpu-ids / --workers 解析实际 device 列表与并行度."""
    cuda_available = torch.cuda.is_available()

    if device_kind == "cpu":
        return [torch.device("cpu")], 1
    if device_kind == "cuda" and not cuda_available:
        raise RuntimeError(
            "--device cuda specified but torch.cuda.is_available() == False"
        )
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
    args: tuple[str, str, int, str, int, int],
) -> dict[str, int]:
    """spawn worker 入口: 绑定一张 GPU, 处理一个 shard, 返回本 shard 的 stats."""
    (input_path, output_path, group_size, gate_up_order, gpu_id, num_threads) = args
    torch.set_num_threads(max(1, num_threads))
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    stats = _empty_stats()
    convert_one_file(input_path, output_path, group_size, gate_up_order, stats, device)
    return stats


def convert_model(
    input_dir: str,
    output_dir: str,
    group_size: int,
    gate_up_order: str,
    limit_files: int | None,
    device_kind: str,
    gpu_ids: str | None,
    workers: int | None,
    num_threads: int,
) -> tuple[dict[str, int], list[torch.device], int]:
    """遍历所有 shard 顺序或多卡并行转换, 返回 (stats, devices, workers)."""
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
                gate_up_order,
                stats,
                device,
            )
        return stats, devices, effective_workers

    # 多 GPU 并行
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
                gate_up_order,
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


def build_ignore_list() -> list[str]:
    """
    ignore 列表 (与 Qwen3.8-Flash-Next-FP8 参考的 modules_to_not_convert 语义一致).
    使用简洁 regex, 覆盖:
      - linear_attention 层内所有 Linear 子模块 + conv1d
      - full_attention 层 Q/K/V/O 与 indexer
      - MoE router gate 与 shared_expert
      - hyper-connection 相关 (attn / mlp / mixer)
      - vision encoder
      - PLE (防御性, 源 state_dict 未见但 FP8 config 有)
      - MTP 顶层与 MTP 层的非 experts 模块
      - embedding / lm_head / norm
    routed experts (mlp.experts.\\d+.{gate,up,down}_proj) 显式不列入 ignore.

    模式采用整棵子树 `.*` 宽匹配 (compressed-tensors ignore 匹配的是模块名,
    不带 .weight 后缀), 比 FP8 参考的展开 968 条列表等价且更简洁.
    """
    return [
        # 常规
        "lm_head",
        r"re:.*embed_tokens.*",
        r"re:.*\.norm$",

        # 语言模型层内 layernorm
        r"re:.*layers\.\d+\.(input_layernorm|post_attention_layernorm)$",

        # linear_attention 层: 整棵子树 (A_log/dt_bias/conv1d/in_proj_*/out_proj/norm)
        r"re:.*linear_attn\..*",

        # full_attention 层: 整棵 self_attn 子树 (Q/K/V/O/q_norm/k_norm/indexer/...)
        r"re:.*self_attn\..*",

        # MoE 相关: router gate (整棵, 含 gate.e_score_correction_bias 之类)
        r"re:.*mlp\.gate$",
        r"re:.*mlp\.gate\..*",
        # shared_expert 整棵子树 (gate_proj/up_proj/down_proj + 未来可能新增)
        r"re:.*mlp\.shared_expert\..*",
        r"re:.*mlp\.shared_expert_gate$",
        r"re:.*mlp\.shared_expert_gate\..*",

        # hyper-connection (attn 侧, mlp 侧, 顶层 mixer)
        r"re:.*(attn|mlp)_hyper_connection\..*",
        r"re:.*hyper_connection_mixer\..*",

        # PLE (防御性; 源 state_dict 无, FP8 config 内 layer 1 有)
        r"re:.*\.ple\..*",

        # MTP 顶层非-experts
        r"re:^mtp\.(fc_embedding|fc_hidden|pre_fc_norm_embedding|pre_fc_norm_hidden)$",
        r"re:^mtp\.(fc_embedding|fc_hidden|pre_fc_norm_embedding|pre_fc_norm_hidden)\..*",
        r"re:^mtp\.hyper_connection_mixer\..*",

        # MTP 层的非-experts (self_attn / shared_expert / hc / gate)
        # 注: negative lookahead 排除 mlp.experts.\d+, 允许其它 mtp.layers.* 全 ignore
        r"re:^mtp\.layers\.\d+\.(?!mlp\.experts\.).*",

        # Vision encoder 整个塔
        r"re:.*visual\..*",
        r"re:^visual\..*",
    ]


def build_compression_config(mode: str, group_size: int) -> dict:
    """生成 compressed-tensors pack-quantized 配置 (W4A16 或 W4A8).

    targets 只匹配已解 pack + 拆开的 HF-style per-expert 2D experts:
      (model.language_model|mtp).layers.{L}.mlp.experts.{E}.(gate_proj|up_proj|down_proj)
    """
    return {
        "config_groups": {
            "group_0": {
                "input_activations": _input_activations_arg(mode),
                "output_activations": None,
                "targets": [
                    r"re:^(?:model\.language_model|mtp)\.layers\.\d+"
                    r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
                ],
                "weights": _weights_arg(group_size),
            }
        },
        "format": "pack-quantized",
        "ignore": build_ignore_list(),
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
    """
    重建 model.safetensors.index.json 的 weight_map:
      读取每个新 shard 的实际 tensor keys, 覆盖旧 weight_map.
    自动:
      - 加入拆出的 experts 新键 (weight_packed / weight_scale / weight_shape)
      - 剔除源里的 packed 3D 键 (gate_up_proj / down_proj)
      - 保留所有透传键 (等一切 non-experts)
    """
    output_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    # 若源 index 不存在或未拷贝, 从零构建
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
    mode: str,
    group_size: int,
    limit_files: int | None,
) -> None:
    """复制配置文件, 更新 quantization_config, 并重建 index."""
    # 尽量覆盖 Qwen3.8-Flash-Next 的所有辅助文件
    for fname in [
        "config.json",
        "configuration.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
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

    # 若源有旧 quantization_config (Qwen3.8-Flash-Next 原生 BF16 通常没有) 一并清理
    config.pop("compression_config", None)
    config.pop("quantization_config", None)
    # 嵌套字段中若挂了旧 quantization_config 也清掉
    for k in list(config.keys()):
        v = config[k]
        if isinstance(v, dict) and "quantization_config" in v:
            v.pop("quantization_config", None)

    config["quantization_config"] = build_compression_config(mode, group_size)

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
        "gate_up_order": args.gate_up_order,
        "limit_files": args.limit_files,
        "compute": {
            "device": args.device,
            "gpu_ids": args.gpu_ids,
            "workers_requested": args.workers,
            "workers_effective": workers,
            "devices_used": [str(d) for d in devices],
        },
        "layout": {
            "routed_experts": "INT4 pack-quantized (int32-lane, HF-style per-expert 2D)",
            "shared_expert / self_attn / linear_attn / hc / vision / mtp non-experts": "BF16 (passthrough)",
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
        description=(
            "Convert Qwen3.8-Flash-Next (BF16 packed 3D experts) to INT4 W4A16 / W4A8 "
            "(compressed-tensors pack-quantized, HF-style per-expert 2D layout, "
            "aligned with Qwen3.8-Flash-Next-FP8)."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"/models/Qwen3.8-Flash-Next",
        help="Path to Qwen3.8-Flash-Next BF16 checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"/models/Qwen3.8-Flash-Next-W4A16",
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
        "--gate-up-order",
        type=str,
        choices=["gate_first", "up_first"],
        default="gate_first",
        help="Order of gate/up in packed gate_up_proj[E, 2*I, H] along the out dim. "
        "gate_first: [:I]=gate, [I:]=up. up_first: [:I]=up, [I:]=gate. "
        "Verify against Qwen3.8-Flash-Next-FP8 reference before large runs.",
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
    print(f"  group_size      : {args.group_size} "
          f"({'per-channel' if args.group_size == -1 else 'per-group'})")
    print(f"  gate_up_order   : {args.gate_up_order}")
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
        args.gate_up_order,
        args.limit_files,
        args.device,
        args.gpu_ids,
        args.workers,
        args.num_threads,
    )
    copy_metadata(
        args.input_dir, args.output_dir, args.mode, args.group_size, args.limit_files
    )
    write_summary(args.output_dir, stats, args, devices, workers)

    print()
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"\nDone! INT4 {args.mode.upper()} model saved to {args.output_dir}")
    print(f"  devices used   : {[str(d) for d in devices]}")
    print(f"  workers used   : {workers}")


if __name__ == "__main__":
    main()
