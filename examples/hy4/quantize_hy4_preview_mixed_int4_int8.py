# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Hy4-preview MXFP8 -> W4A8 INT4/INT8/BF16 混合量化脚本。

设计目标
========
本脚本面向 ``/mnt/c/chl/models/Hy4-preview-FP8`` 的官方 MXFP8
checkpoint，输出 compressed-tensors 可识别的混合精度权重：

  * 默认：主干 layer 1-77 的 routed experts，以及 MTP routed experts，INT4；
  * ``--routed-experts int8``：主干 layer 1-77 的 routed experts 从默认
    INT4 升级为 INT8（用于排查 per-channel INT4 的精度损失）；
  * ``--int8-mlp``：主干/MTP shared experts 和 layer 0 dense MLP 改为 INT8；
  * ``--int8-mla``：主干/MTP 的 q_a/kv_a、q_b、kv_b、o_proj 改为 INT8；
  * ``--int8-indexer-wq-b``：full indexer 的 wq_b 改为 INT8；
  * ``--int8-indexer-wk``：实验性地把 full indexer 的 wk 改为 INT8，默认关闭；
  * ``--mtp-experts int8``：MTP routed experts 从默认 INT4 升级为 INT8；
  * ``--int8-linear-gate``：实验性地把 MLA 输出门控 ``self_attn.linear_gate``
    （主干和 MTP 全部层）改为 INT8，默认关闭，不包含在 ``--int8-all`` 中。
    注意 vLLM 的 HPC 融合 gated-MLA 内核要求 gate 权重为 BF16，开启后会退回
    eager 的 linear + sigmoid + 乘法三步路径。

所有 INT8 权重均为对称 per-output-channel INT8，配 FP32
``weight_scale``，激活声明为动态 per-token INT8（W8A8）。INT4 使用
对称 per-output-channel 或 per-group 量化，权重按 8 个 int4 打包到一个
int32 lane；激活声明为动态 per-token INT8（W4A8）。

重要的 vLLM 约束
================
Hy4 的源 routed experts 不是 DeepSeek 风格的
``experts.{id}.w1.weight``，而是 fused 的三维张量：

  * ``mlp.experts.gate_up_proj``: [E, 2I, H]
  * ``mlp.experts.down_proj``:    [E, H, I]

本脚本会拆成 vLLM loader 可识别的
``experts.{id}.{gate,up,down}_proj.weight`` 形式。源 scale 分别是
``<fused>_scale``；输出 scale 是 ``<projection>.weight_scale``。

vLLM 的 Hy4 模型还把 q_a + kv_a 融合为 ``fused_qkv_a_proj``，把
indexer.wk + weights_proj 融合为 ``wk_weights_proj``。因此：

  * q_a 与 kv_a 必须同档位、同为 INT8 或同为 BF16；
  * ``indexer.weights_proj`` 没有源 FP8 scale，始终保留 BF16；
  * 当前 vLLM ``Indexer.wk_weights_proj`` 使用 ``quant_config=None``，
    默认将 FP8 wk 转为 BF16。``--int8-indexer-wk`` 只是实验性输出开关，
    必须配合人工修改 vLLM 的融合 loader/kernel，stock vLLM 不可直接加载。
  * 当前 vLLM 的 W4A8 INT4 MoE backend 仅实现 CPU；本脚本仍按
    compressed-tensors contract 生成 W4A8 权重，GPU kernel/loader 由人工补齐。

示例
====
只量化 routed experts（主干和 MTP 默认 INT4）：

  python quantize_hy4_preview_mixed_int4_int8.py \\
      --input-dir /models/Hy4-preview-FP8 \\
      --output-dir /models/Hy4-preview-W4A8

shared/dense MLP 也改为 INT8：

  python quantize_hy4_preview_mixed_int4_int8.py \\
      --int8-mlp --input-dir ... --output-dir ...

MLA、indexer.wq_b 一起改为 INT8：

  python quantize_hy4_preview_mixed_int4_int8.py \\
      --int8-all --input-dir ... --output-dir ...

MTP routed experts 改为 INT8：

  python quantize_hy4_preview_mixed_int4_int8.py \\
      --mtp-experts int8 --input-dir ... --output-dir ...

主干 + MTP routed experts 都改为 INT8：

  python quantize_hy4_preview_mixed_int4_int8.py \\
      --routed-experts int8 --mtp-experts int8 --input-dir ... --output-dir ...

实验性 indexer.wk INT8（需要人工 vLLM 补丁）：

  python quantize_hy4_preview_mixed_int4_int8.py \\
      --int8-indexer-wk --input-dir ... --output-dir ...

依赖
====
``torch >= 2.1``、``safetensors``、``tqdm``。本地没有权重时只能做
``python3 -I -m py_compile``；完整转换必须在有 safetensors 分片和目标
运行环境的 Docker 中执行。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

INT4_QMAX = 7.0
INT8_QMAX = 127.0
FULL_INDEXER_LAYERS = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77)

# Source checkpoint names.  Expert tensors are fused and do not end in .weight.
EXPERT_GATE_UP_RE = re.compile(
    r"^(model\.(?:layers\.\d+|mtp_layers\.\d+)\.mlp\.experts)\.gate_up_proj$"
)
EXPERT_DOWN_RE = re.compile(
    r"^(model\.(?:layers\.\d+|mtp_layers\.\d+)\.mlp\.experts)\.down_proj$"
)
SHARED_RE = re.compile(
    r"^model\.(?:layers\.\d+|mtp_layers\.\d+)\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
DENSE_MLP_RE = re.compile(
    r"^model\.layers\.0\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
MLA_RE = re.compile(
    r"^model\.(?:layers\.\d+|mtp_layers\.\d+)\.self_attn\."
    r"(?:q_a_proj|kv_a_proj_with_mqa|q_b_proj|kv_b_proj|o_proj)\.weight$"
)
INDEXER_WQ_B_RE = re.compile(
    r"^model\.(?:layers\.\d+|mtp_layers\.\d+)\.self_attn\.indexer\.wq_b\.weight$"
)
INDEXER_WK_RE = re.compile(
    r"^model\.(?:layers\.\d+|mtp_layers\.\d+)\.self_attn\.indexer\.wk\.weight$"
)
LINEAR_GATE_RE = re.compile(
    r"^model\.(?:layers\.\d+|mtp_layers\.\d+)\.self_attn\.linear_gate\.weight$"
)


def source_scale_name(name: str) -> str:
    """Return the source ModelOpt scale key for a weight key."""
    if name.endswith(".weight"):
        return name[: -len(".weight")] + ".weight_scale"
    return name + "_scale"


def output_scale_name(name: str) -> str:
    """Return compressed-tensors' channel scale key."""
    return name[: -len(".weight")] + ".weight_scale"


def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a block-scaled FP8 tensor using scale-grid ratios.

    ModelOpt Hy4 uses scale grids whose dimensions divide the two matrix
    dimensions. This implementation also supports a leading expert dimension.
    """
    if scale.dtype == torch.uint8 and hasattr(torch, "float8_e8m0fnu"):
        # ModelOpt may serialize UE8M0 scales as raw exponent bytes.
        scale = scale.view(torch.float8_e8m0fnu)
    if weight.dtype != torch.float8_e4m3fn:
        return weight.float()
    if scale.ndim != weight.ndim:
        if scale.numel() == 1:
            return weight.float() * scale.float().reshape(1)
        raise ValueError(f"weight/scale ndim mismatch: {weight.shape} vs {scale.shape}")
    if weight.ndim < 2:
        return weight.float() * scale.float()

    out = weight.float()
    # Scale grids are [E, rows, cols] for fused experts and [rows, cols]
    # for ordinary Linear weights. Expand each scale cell over its tile.
    for dim in range(weight.ndim - 2):
        if scale.shape[dim] != weight.shape[dim]:
            raise ValueError(f"expert dimension mismatch: {weight.shape} vs {scale.shape}")
    row_dim, col_dim = weight.ndim - 2, weight.ndim - 1
    if weight.shape[row_dim] % scale.shape[row_dim] or weight.shape[col_dim] % scale.shape[col_dim]:
        raise ValueError(f"non-divisible FP8 scale grid: {weight.shape} vs {scale.shape}")
    row_block = weight.shape[row_dim] // scale.shape[row_dim]
    col_block = weight.shape[col_dim] // scale.shape[col_dim]
    expanded = scale.float().repeat_interleave(row_block, dim=row_dim)
    expanded = expanded.repeat_interleave(col_block, dim=col_dim)
    return out * expanded


def quantize_int8_channelwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 per-output-channel quantization for a 2-D weight."""
    if weight.ndim != 2:
        raise ValueError(f"INT8 channelwise expects 2-D weight, got {weight.shape}")
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / INT8_QMAX
    q = torch.round(w / scale).clamp(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


def quantize_int4_channelwise(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric signed INT4, returning int8 codes and FP32/BF16 scales."""
    if weight.ndim != 2:
        raise ValueError(f"INT4 quantization expects 2-D weight, got {weight.shape}")
    n, k = weight.shape
    w = weight.float()
    if group_size in (-1, None):
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / INT4_QMAX
        q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
        return q.contiguous(), scale.float().contiguous()
    if k % group_size:
        raise ValueError(f"K={k} is not divisible by group_size={group_size}")
    wg = w.view(n, k // group_size, group_size)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT4_QMAX
    q = torch.round(wg / scale).clamp(-8, 7).to(torch.int8).view(n, k)
    return q.contiguous(), scale.squeeze(-1).float().contiguous()


def pack_int4_to_int32(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 codes in ascending nibble order."""
    if q.dtype != torch.int8 or q.ndim != 2 or q.shape[1] % 8:
        raise ValueError(f"invalid INT4 packing input: {q.shape}/{q.dtype}")
    u = (q.to(torch.int32) + 8) & 0xF
    u = u.view(q.shape[0], q.shape[1] // 8, 8)
    shifts = torch.arange(8, device=q.device, dtype=torch.int32) * 4
    return (u << shifts).sum(dim=-1).to(torch.int32).contiguous()


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().contiguous()


def _is_expert_fused(name: str) -> bool:
    return EXPERT_GATE_UP_RE.match(name) is not None or EXPERT_DOWN_RE.match(name) is not None


def _is_mtp(name: str) -> bool:
    return ".mtp_layers." in name


def _expert_projection_names(name: str, expert_id: int) -> list[tuple[str, torch.Tensor, str]]:
    """Map a fused source name to split gate/up/down output names."""
    if EXPERT_GATE_UP_RE.match(name):
        base = EXPERT_GATE_UP_RE.match(name).group(1)
        return [
            (f"{base}.{expert_id}.gate_proj.weight", None, "gate_proj"),
            (f"{base}.{expert_id}.up_proj.weight", None, "up_proj"),
        ]
    match = EXPERT_DOWN_RE.match(name)
    if match:
        return [(f"{match.group(1)}.{expert_id}.down_proj.weight", None, "down_proj")]
    raise ValueError(f"not a fused expert key: {name}")


def load_float_weight(
    tensor: torch.Tensor, scale: torch.Tensor | None, device: torch.device
) -> torch.Tensor:
    """Return an fp32 weight on ``device`` from either an FP8+scale or a BF16/FP16 tensor."""
    if tensor.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise ValueError("FP8 weight has no source scale")
        return dequant_fp8_blockwise(tensor.to(device), scale.to(device))
    if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return tensor.to(device).float()
    raise ValueError(f"unsupported weight dtype: {tensor.dtype}")


def detect_input_format(input_dir: str, weight_map: dict[str, str]) -> str:
    """Detect ``fp8`` or ``bf16`` from the first fused expert tensor's real dtype."""
    for name, shard in sorted(weight_map.items()):
        if _is_expert_fused(name):
            with safe_open(os.path.join(input_dir, shard), framework="pt", device="cpu") as f:
                dtype = f.get_slice(name).get_dtype()
            has_scale = source_scale_name(name) in weight_map
            if dtype == "F8_E4M3" and has_scale:
                return "fp8"
            if dtype in ("BF16", "F16") and not has_scale:
                return "bf16"
            raise ValueError(
                f"cannot detect input format: {name} dtype={dtype}, scale present={has_scale}"
            )
    raise ValueError("no fused expert tensor found in the index; cannot detect input format")


def _quantize_expert(
    name: str,
    tensor: torch.Tensor,
    scale: torch.Tensor | None,
    opts: dict,
    new: dict[str, torch.Tensor],
    stats: dict[str, int],
) -> None:
    """Split one fused [E,*,*] expert tensor and quantize each expert."""
    if tensor.ndim != 3 or (scale is not None and (scale.ndim != 3 or tensor.shape[0] != scale.shape[0])):
        raise ValueError(f"fused expert shape mismatch for {name}: {tensor.shape}/{None if scale is None else scale.shape}")
    is_mtp = _is_mtp(name)
    mode = opts["mtp_experts"] if is_mtp else opts["routed_experts"]
    is_gate_up = name.endswith("gate_up_proj")
    if is_gate_up and tensor.shape[1] % 2:
        raise ValueError(f"gate_up_proj output dimension must be even: {tensor.shape}")
    base = name.rsplit(".", 1)[0]
    device = opts["_device"]

    # Process one expert at a time so the GPU only ever holds a single 2-D
    # matrix (~100 MB FP32) instead of the whole [E, N, K] tensor (~25 GB).
    for expert_id in range(tensor.shape[0]):
        weight = load_float_weight(
            tensor[expert_id], None if scale is None else scale[expert_id], device
        )
        if is_gate_up:
            half = weight.shape[0] // 2
            pieces = [("gate_proj", weight[:half]), ("up_proj", weight[half:])]
        else:
            pieces = [("down_proj", weight)]
        for projection, q_weight in pieces:
            out_name = f"{base}.{expert_id}.{projection}.weight"
            if mode == "int8":
                q, q_scale = quantize_int8_channelwise(q_weight)
                new[out_name] = _to_cpu(q)
                new[output_scale_name(out_name)] = _to_cpu(q_scale)
                stats["mtp_expert_int8" if is_mtp else "routed_expert_int8"] += 1
            else:
                q, q_scale = quantize_int4_channelwise(q_weight, opts["group_size"])
                # compressed-tensors pack-quantized, same layout as the DeepSeek script.
                prefix = out_name[: -len(".weight")]
                new[f"{prefix}.weight_packed"] = _to_cpu(pack_int4_to_int32(q))
                new[f"{prefix}.weight_scale"] = _to_cpu(q_scale.to(torch.bfloat16))
                new[f"{prefix}.weight_shape"] = torch.tensor(list(q.shape), dtype=torch.int64)
                stats["mtp_expert_int4" if is_mtp else "routed_expert_int4"] += 1
        del weight, pieces


def _convert_one_shard(
    input_path: str,
    output_path: str,
    opts: dict,
    stats: dict[str, int],
    device: torch.device,
) -> None:
    state: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state[name] = f.get_tensor(name)

    out: dict[str, torch.Tensor] = {}
    consumed_scales: set[str] = set()
    # Only a tensor that is the source scale of an FP8 weight may be dropped.
    # A name-suffix test is wrong: Hy4 stores real fp32 parameters whose names
    # also end in "_scale" (iHC ``hc_pre.hc_scale`` and ``hc_head.hc_head_scale``).
    scale_names = {
        source_scale_name(n)
        for n, t in state.items()
        if t.dtype == torch.float8_e4m3fn and source_scale_name(n) in state
    }
    for name, tensor in state.items():
        if name in scale_names:
            continue
        src_scale = source_scale_name(name)

        if _is_expert_fused(name):
            # FP8 input carries a UE8M0 scale tensor; BF16 input has none.
            if tensor.dtype == torch.float8_e4m3fn and src_scale not in state:
                raise ValueError(f"missing fused expert scale: {name} -> {src_scale}")
            _quantize_expert(name, tensor, state.get(src_scale), {**opts, "_device": device}, out, stats)
            if src_scale in state:
                consumed_scales.add(src_scale)
            continue

        is_weight = name.endswith(".weight")
        is_fp8 = tensor.dtype == torch.float8_e4m3fn
        is_int8_target = (
            (opts["int8_mlp"] and (SHARED_RE.match(name) or DENSE_MLP_RE.match(name)))
            or (opts["int8_mla"] and MLA_RE.match(name))
            or (opts["int8_indexer_wq_b"] and INDEXER_WQ_B_RE.match(name))
            or (opts["int8_indexer_wk"] and INDEXER_WK_RE.match(name))
            or (opts["int8_linear_gate"] and LINEAR_GATE_RE.match(name))
        )
        if is_int8_target and is_weight and (is_fp8 or tensor.dtype in (torch.bfloat16, torch.float16)):
            if is_fp8 and src_scale not in state:
                raise ValueError(f"INT8 target has no source scale: {name}")
            fp = load_float_weight(tensor, state.get(src_scale), device)
            q, q_scale = quantize_int8_channelwise(fp)
            out[name] = _to_cpu(q)
            out[output_scale_name(name)] = _to_cpu(q_scale)
            if is_fp8:
                consumed_scales.add(src_scale)
            stats["int8_weights"] += 1
            if LINEAR_GATE_RE.match(name):
                stats["linear_gate_int8"] += 1
            continue

        if is_fp8 and is_weight:
            if INDEXER_WK_RE.match(name):
                if src_scale not in state:
                    raise ValueError(f"indexer.wk has no source scale: {name}")
                fp = dequant_fp8_blockwise(tensor.to(device), state[src_scale].to(device))
                out[name] = _to_cpu(fp.bfloat16())
                consumed_scales.add(src_scale)
                stats["indexer_wk_bf16"] += 1
                continue
            # Ordinary FP8 weights are BF16 when their corresponding INT8
            # switch is disabled. This keeps every source FP8 module covered
            # without silently preserving an unclassified FP8 tensor.
            if src_scale not in state:
                raise ValueError(f"FP8 weight has no source scale: {name}")
            fp = dequant_fp8_blockwise(tensor.to(device), state[src_scale].to(device))
            out[name] = _to_cpu(fp.bfloat16())
            consumed_scales.add(src_scale)
            stats["fp8_to_bf16"] += 1
            continue

        out[name] = tensor
        stats["kept"] += 1
        if name.endswith("_scale") or name.endswith(".weight_scale"):
            stats["kept_scale_like"] += 1

    # Self-check: every input tensor must be written, consumed as the scale of
    # a converted FP8 weight, or replaced by split experts. Anything else was
    # silently dropped (or is an FP8 tensor left without its scale).
    fused_names = {n for n in state if _is_expert_fused(n)}
    unaccounted = sorted(set(state) - set(out) - consumed_scales - fused_names)
    if unaccounted:
        raise ValueError(
            f"{os.path.basename(input_path)}: {len(unaccounted)} input tensor(s) "
            f"were dropped without being converted, e.g. {unaccounted[:5]}"
        )

    save_file(out, output_path)


def _int8_weights() -> dict:
    return {
        "actorder": None, "block_structure": None, "dynamic": False,
        "group_size": -1, "num_bits": 8, "observer": "minmax",
        "observer_kwargs": {}, "strategy": "channel", "symmetric": True, "type": "int",
    }


def _int8_input() -> dict:
    return {
        "actorder": None, "block_structure": None, "dynamic": True,
        "group_size": None, "num_bits": 8, "observer": None,
        "observer_kwargs": {}, "strategy": "token", "symmetric": True, "type": "int",
    }


def _int4_weights(group_size: int) -> dict:
    return {
        "actorder": None, "block_structure": None, "dynamic": False,
        "group_size": group_size, "num_bits": 4, "observer": "minmax",
        "observer_kwargs": {}, "strategy": "channel" if group_size in (-1, None) else "group",
        "symmetric": True, "type": "int",
    }


def _runtime_layer_regex(layer_ids: list[int], suffix: str) -> str:
    ids = "|".join(str(i) for i in layer_ids)
    return rf"re:.*(?<!mtp_)layers\.(?:{ids})\.{suffix}$"


def _mtp_layer_regex(draft_ids: list[int], ckpt_ids: list[int], suffix: str) -> str:
    """Match the MTP layer under every name an engine may use.

    vLLM builds the draft as ``model.layers.<num_hidden_layers + i>``, the
    checkpoint names it ``model.mtp_layers.<i>``, and sglang
    (hunyuan_v4_nextn.py) builds a single ``model.decoder``.
    """
    d = "|".join(str(i) for i in draft_ids)
    c = "|".join(str(i) for i in ckpt_ids)
    return rf"re:.*(?:(?<!mtp_)layers\.(?:{d})|mtp_layers\.(?:{c})|model\.decoder)\.{suffix}$"


_EXPERT_SUFFIX = r"mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)"


def build_compression_config(opts: dict, cfg: dict) -> dict:
    hidden = int(cfg["num_hidden_layers"])
    mtp_count = int(cfg.get("num_nextn_predict_layers", 0))
    backbone_experts = list(range(1, hidden))
    mtp_layers = list(range(hidden, hidden + max(mtp_count, 1)))
    mtp_ckpt_ids = list(range(max(mtp_count, 1)))
    groups: dict[str, dict] = {}

    int4_common = {
        "input_activations": _int8_input(), "output_activations": None,
        "weights": _int4_weights(opts["group_size"]), "format": "pack-quantized",
    }
    int8_expert_common = {
        "input_activations": _int8_input(), "output_activations": None,
        "weights": _int8_weights(), "format": "int-quantized",
    }
    backbone_target = _runtime_layer_regex(backbone_experts, _EXPERT_SUFFIX)
    mtp_target = _mtp_layer_regex(mtp_layers, mtp_ckpt_ids, _EXPERT_SUFFIX)
    if opts["routed_experts"] == "int4":
        groups["group_int4_routed_experts"] = {"targets": [backbone_target], **int4_common}
    else:
        groups["group_int8_routed_experts"] = {"targets": [backbone_target], **int8_expert_common}
    if opts["mtp_experts"] == "int4":
        groups["group_int4_mtp_experts"] = {"targets": [mtp_target], **int4_common}
    else:
        groups["group_int8_mtp_experts"] = {"targets": [mtp_target], **int8_expert_common}

    int8_common = {"input_activations": _int8_input(), "output_activations": None,
                   "weights": _int8_weights(), "format": "int-quantized"}
    # Targets list the engine-fused names (fused_qkv_a_proj, gate_up_proj,
    # wk_weights_proj) and the unfused checkpoint names, so the config resolves
    # the same way in engines that do not fuse.
    if opts["int8_mlp"]:
        groups["group_int8_mlp"] = {
            "targets": [r"re:.*shared_experts\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$",
                        r"re:.*layers\.0\.mlp\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$"],
            **int8_common,
        }
    if opts["int8_mla"]:
        groups["group_int8_mla"] = {
            "targets": [r"re:.*self_attn\.(?:fused_qkv_a_proj(?:_with_mqa)?|q_a_proj|kv_a_proj_with_mqa|q_b_proj|kv_b_proj|o_proj)$"],
            **int8_common,
        }
    if opts["int8_indexer_wq_b"]:
        groups["group_int8_indexer_wq_b"] = {
            "targets": [r"re:.*self_attn\.indexer\.wq_b$"],
            **int8_common,
        }
    if opts["int8_indexer_wk"]:
        groups["group_int8_indexer_wk_EXPERIMENTAL"] = {
            "targets": [r"re:.*self_attn\.indexer\.(?:wk_weights_proj|wk)$"],
            **int8_common,
        }
    if opts["int8_linear_gate"]:
        groups["group_int8_linear_gate"] = {
            "targets": [r"re:.*self_attn\.linear_gate$"],
            **int8_common,
        }

    # ``.*gate$`` would also match ``self_attn.linear_gate``, so exclude that
    # name from the router pattern when linear_gate is quantized; otherwise the
    # module stays ignored and vLLM would load INT8 weights as BF16.
    router_gate = (r"re:.*(?<!linear_)(?:router\.)?gate$" if opts["int8_linear_gate"]
                   else r"re:.*(?:router\.)?gate$")
    ignore = [
        router_gate, r"re:.*(?:router\.)?gate\..*",
        r"re:.*hc_.*", r"re:.*(?:input_layernorm|post_attention_layernorm)$",
        r"re:.*(?:q_a_layernorm|kv_a_layernorm|norm)$", r"re:.*learnable_sink_param$",
        r"re:.*indexer\.k_norm(?:\..*)?$", r"re:.*indexer\.weights_proj$",
        r"lm_head", r"re:.*embed_tokens.*",
        r"re:.*(?:eh_proj|enorm|hnorm|final_layernorm)$",
    ]
    if not opts["int8_indexer_wk"]:
        ignore.append(r"re:.*indexer\.wk_weights_proj$")
    if not opts["int8_indexer_wq_b"]:
        ignore.append(r"re:.*indexer\.wq_b$")
    if not opts["int8_mlp"]:
        ignore.extend([r"re:.*shared_experts\.(?:gate_up_proj|down_proj)$",
                       r"re:.*layers\.0\.mlp\.(?:gate_up_proj|down_proj)$"])
    if not opts["int8_mla"]:
        ignore.append(r"re:.*self_attn\.(?:fused_qkv_a_proj(?:_with_mqa)?|q_a_proj|kv_a_proj_with_mqa|q_b_proj|kv_b_proj|o_proj)$")
    if not opts["int8_indexer_wk"]:
        ignore.append(r"re:.*indexer\.wk$")
    if not opts["int8_linear_gate"]:
        ignore.append(r"re:.*linear_gate$")
    return {"config_groups": groups, "format": "pack-quantized", "ignore": ignore,
            "kv_cache_scheme": None, "quant_method": "compressed-tensors",
            "quantization_status": "compressed"}


def _self_check_mtp_config(opts: dict, cfg: dict) -> None:
    """Assert the config resolves each module to the dtype this script writes.

    A mismatch (config says INT8/INT4, weight is BF16, or the reverse) leaves
    weight_scale uninitialised, so the draft is silently wrong: MTP acceptance
    collapses while the main model still looks fine. Checked under both the
    vLLM draft names (``model.layers.<N + i>``) and checkpoint names
    (``model.mtp_layers.<i>``); backbone names must never reach an MTP group.
    """
    qc = build_compression_config(opts, cfg)
    hidden = int(cfg["num_hidden_layers"])
    n_mtp = max(int(cfg.get("num_nextn_predict_layers", 0)), 1)

    def ignored(name: str) -> bool:
        return any(re.match(p[3:], name) if p.startswith("re:") else p == name
                   for p in qc["ignore"])

    def groups_of(name: str) -> list[str]:
        return [g for g, v in qc["config_groups"].items()
                if any(re.match(t[3:], name) for t in v["targets"])]

    def expect(name: str, group: str | None) -> None:
        got = groups_of(name)
        if group is None:
            ok = ignored(name) or not got
        else:
            ok = not ignored(name) and got == [group]
        if not ok:
            raise AssertionError(
                f"config mismatch for {name}: expected {group}, "
                f"got groups={got}, ignored={ignored(name)}")

    mtp_exp = f"group_{opts['mtp_experts']}_mtp_experts"
    bb_exp = f"group_{opts['routed_experts']}_routed_experts"
    mtp_names = [f"model.layers.{hidden + i}" for i in range(n_mtp)] + \
                [f"model.mtp_layers.{i}" for i in range(n_mtp)] + ["model.decoder"]
    for base in mtp_names:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expect(f"{base}.mlp.experts.7.{proj}", mtp_exp)
        for mod in ("mlp.gate", "eh_proj", "enorm", "hnorm", "final_layernorm",
                    "self_attn.indexer.weights_proj"):
            expect(f"{base}.{mod}", None)
        expect(f"{base}.self_attn.linear_gate",
               "group_int8_linear_gate" if opts["int8_linear_gate"] else None)
        for mod in ("self_attn.fused_qkv_a_proj", "self_attn.fused_qkv_a_proj_with_mqa",
                    "self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa",
                    "self_attn.q_b_proj", "self_attn.kv_b_proj", "self_attn.o_proj"):
            expect(f"{base}.{mod}", "group_int8_mla" if opts["int8_mla"] else None)
        for mod in ("mlp.shared_experts.gate_up_proj", "mlp.shared_experts.gate_proj",
                    "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj"):
            expect(f"{base}.{mod}", "group_int8_mlp" if opts["int8_mlp"] else None)
        expect(f"{base}.self_attn.indexer.wq_b",
               "group_int8_indexer_wq_b" if opts["int8_indexer_wq_b"] else None)
    for layer in (1, 40, hidden - 1):
        expect(f"model.layers.{layer}.mlp.experts.0.down_proj", bb_exp)
    expect("model.layers.0.mlp.experts.0.down_proj", None)

    # The write side must agree with the config for the MTP tensors.
    probes = {
        "model.mtp_layers.0.self_attn.q_b_proj.weight": opts["int8_mla"],
        "model.mtp_layers.0.mlp.shared_experts.down_proj.weight": opts["int8_mlp"],
        "model.mtp_layers.0.self_attn.indexer.wq_b.weight": opts["int8_indexer_wq_b"],
        "model.mtp_layers.0.self_attn.linear_gate.weight": opts["int8_linear_gate"],
    }
    for name, quantized in probes.items():
        targeted = bool(
            (opts["int8_mlp"] and (SHARED_RE.match(name) or DENSE_MLP_RE.match(name)))
            or (opts["int8_mla"] and MLA_RE.match(name))
            or (opts["int8_indexer_wq_b"] and INDEXER_WQ_B_RE.match(name))
            or (opts["int8_linear_gate"] and LINEAR_GATE_RE.match(name)))
        if targeted != bool(quantized):
            raise AssertionError(f"write side disagrees with config for {name}")
    print("Self-check passed: MTP config matches written dtypes (vLLM and checkpoint names).")


def _rebuild_index(output_dir: str, model_index: dict) -> None:
    weight_map: dict[str, str] = {}
    total_size = 0
    for path in sorted(glob(os.path.join(output_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                weight_map[name] = os.path.basename(path)
                t = f.get_tensor(name)
                total_size += t.numel() * t.element_size()
    model_index["weight_map"] = weight_map
    model_index["metadata"] = {"total_size": total_size}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)


def _verify_conversion(
    input_weight_map: dict[str, str], output_weight_map: dict[str, str], shards: set[str]
) -> None:
    """Global self-check: compare input/output indexes over the converted shards.

    Input tensors absent from the output must be one of the three expected
    kinds below; anything else (e.g. ``hc_scale``) is an unintended drop.
    """
    fused_scales = {source_scale_name(n) for n in input_weight_map if _is_expert_fused(n)}
    expected = {"fused_expert_split": 0, "fused_expert_scale": 0, "fp8_weight_scale": 0}
    unexpected: list[str] = []
    for name, shard in input_weight_map.items():
        if shard not in shards or name in output_weight_map:
            continue
        if _is_expert_fused(name):
            expected["fused_expert_split"] += 1
        elif name in fused_scales:
            expected["fused_expert_scale"] += 1
        elif name.endswith(".weight_scale") and name[: -len("_scale")] in input_weight_map:
            expected["fp8_weight_scale"] += 1
        else:
            unexpected.append(name)
    print(f"Self-check: expected input-only tensors {expected}")
    if unexpected:
        raise RuntimeError(
            f"Self-check FAILED: {len(unexpected)} input tensor(s) are missing from the output "
            f"and are not FP8 scales or fused experts, e.g. {sorted(unexpected)[:10]}"
        )
    print("Self-check passed: no unintended tensor was dropped.")


def _empty_stats() -> dict[str, int]:
    return {k: 0 for k in ("routed_expert_int4", "routed_expert_int8", "mtp_expert_int4",
                           "mtp_expert_int8", "int8_weights", "linear_gate_int8", "indexer_wk_bf16", "fp8_to_bf16",
                           "kept", "kept_scale_like")}


def _resolve_devices(
    device_kind: str, gpu_ids: str | None, workers: int | None, num_files: int
) -> tuple[list[torch.device], int]:
    """Resolve --device/--gpu-ids/--workers into (devices, effective_workers).

    effective_workers == 1 means sequential execution in the main process.
    """
    cuda_ok = torch.cuda.is_available()
    if device_kind == "cpu" or (device_kind == "auto" and not cuda_ok):
        return [torch.device("cpu")], 1
    if not cuda_ok:
        raise RuntimeError("--device cuda specified but torch.cuda.is_available() is False")
    if gpu_ids:
        ids = [int(x) for x in gpu_ids.split(",") if x.strip()]
    else:
        ids = list(range(torch.cuda.device_count()))
    if not ids:
        raise RuntimeError("no CUDA device selected")
    devices = [torch.device(f"cuda:{i}") for i in ids]
    effective = len(devices) if workers is None else max(1, workers)
    return devices, min(effective, max(1, num_files))


def _worker_convert_shard(task: tuple) -> dict[str, int]:
    """spawn worker entry: bind one GPU, convert one shard, return its stats."""
    input_path, output_path, opts, gpu_id, num_threads = task
    torch.set_num_threads(max(1, num_threads))
    torch.cuda.set_device(gpu_id)
    stats = _empty_stats()
    _convert_one_shard(input_path, output_path, opts, stats, torch.device(f"cuda:{gpu_id}"))
    return stats


def main() -> None:
    parser = ArgumentParser(description="Convert Hy4-preview MXFP8 to mixed W4A8 INT4/INT8/BF16.",
                             formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", default="/models/Hy4-preview-FP8")
    parser.add_argument("--output-dir", default="/models/Hy4-preview-W4A8")
    parser.add_argument("--input-format", choices=["auto", "fp8", "bf16"], default="auto",
                        help="Source checkpoint format. auto detects it from the first fused expert tensor.")
    parser.add_argument("--group-size", type=int, default=-1, help="INT4 group size; -1 is per-channel.")
    parser.add_argument("--int8-mlp", action="store_true", help="INT8 shared experts and dense layer 0.")
    parser.add_argument("--int8-mla", action="store_true", help="INT8 MLA projections in backbone and MTP.")
    parser.add_argument("--int8-indexer-wq-b", action="store_true", help="INT8 full-indexer wq_b.")
    parser.add_argument("--int8-indexer-wk", action="store_true", help="EXPERIMENTAL: INT8 indexer wk; requires vLLM patch.")
    parser.add_argument("--int8-linear-gate", action="store_true",
                        help="EXPERIMENTAL: INT8 the MLA output gate (self_attn.linear_gate, all layers incl. MTP). "
                             "Not part of --int8-all. Disables the HPC fused gated-MLA kernel, which needs a BF16 gate.")
    parser.add_argument("--int8-all", action="store_true", help="Enable int8-mlp, int8-mla and int8-indexer-wq-b.")
    parser.add_argument("--routed-experts", choices=["int4", "int8"], default="int4",
                        help="Backbone (layer 1-77) routed experts quantization.")
    parser.add_argument("--mtp-experts", choices=["int4", "int8"], default="int4")
    parser.add_argument("--start-file", type=int, default=0,
                        help="Zero-based index in sorted shard list to start from.")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="auto: all visible GPUs if available, else CPU.")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated CUDA indices, e.g. '0,1,3'. Default: all visible GPUs.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes. Default: number of GPUs. "
                             "Each worker holds one whole shard in CPU RAM, so lower this if RAM is tight.")
    parser.add_argument("--num-threads", type=int, default=8, help="Torch CPU threads per worker.")
    args = parser.parse_args()
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must differ")
    if args.group_size != -1 and args.group_size <= 0:
        raise ValueError("--group-size must be -1 or positive")
    if args.int8_all:
        args.int8_mlp = args.int8_mla = args.int8_indexer_wq_b = True
    if args.int8_indexer_wk:
        print("WARNING: --int8-indexer-wk requires a patched vLLM wk_weights_proj loader/kernel.")

    with open(os.path.join(args.input_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    opts = {"group_size": args.group_size, "int8_mlp": args.int8_mlp,
            "int8_mla": args.int8_mla, "int8_indexer_wq_b": args.int8_indexer_wq_b,
            "int8_indexer_wk": args.int8_indexer_wk, "int8_linear_gate": args.int8_linear_gate,
            "routed_experts": args.routed_experts,
            "mtp_experts": args.mtp_experts}
    with open(os.path.join(args.input_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        detected = detect_input_format(args.input_dir, json.load(f)["weight_map"])
    if args.input_format != "auto" and args.input_format != detected:
        raise ValueError(f"--input-format {args.input_format} conflicts with detected {detected}")
    input_format = detected
    _self_check_mtp_config(opts, cfg)
    print(f"Input format: {input_format}")
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(args.input_dir, "*.safetensors")))
    if args.start_file < 0 or args.start_file >= len(files):
        raise ValueError(f"--start-file must be in [0, {len(files) - 1}]")
    files = files[args.start_file:]
    if args.limit_files is not None:
        files = files[: args.limit_files]
    torch.set_num_threads(args.num_threads)
    devices, n_workers = _resolve_devices(args.device, args.gpu_ids, args.workers, len(files))
    stats = _empty_stats()
    if n_workers <= 1 or len(devices) == 1:
        for path in tqdm(files, desc=f"Converting on {devices[0]}"):
            _convert_one_shard(path, os.path.join(args.output_dir, os.path.basename(path)),
                               opts, stats, devices[0])
    else:
        import torch.multiprocessing as mp

        tasks = [(p, os.path.join(args.output_dir, os.path.basename(p)), opts,
                  devices[i % len(devices)].index, args.num_threads)
                 for i, p in enumerate(files)]
        desc = f"Converting on {len(devices)} GPU(s) ({n_workers} workers)"
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            for per_stats in tqdm(pool.imap_unordered(_worker_convert_shard, tasks),
                                  total=len(tasks), desc=desc):
                for k, v in per_stats.items():
                    stats[k] += v

    for filename in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "configuration.json",
                     "README.md", "README_CN.md", "LICENSE", "chat_template.jinja"):
        src = os.path.join(args.input_dir, filename)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.output_dir, filename))
    with open(os.path.join(args.input_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        model_index = json.load(f)
    # _rebuild_index overwrites weight_map, so keep the input map for the self-check.
    input_weight_map = dict(model_index["weight_map"])
    _rebuild_index(args.output_dir, model_index)
    _verify_conversion(input_weight_map, model_index["weight_map"],
                       {os.path.basename(p) for p in files})
    if input_format == "bf16" and (stats["fp8_to_bf16"] or stats["indexer_wk_bf16"]):
        raise RuntimeError("BF16 input produced FP8 conversion stats; format detection is inconsistent")
    cfg["quantization_config"] = build_compression_config(opts, cfg)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, sort_keys=True)
    summary = {"input_dir": args.input_dir, "input_format": input_format,"output_dir": args.output_dir, "options": opts,
               "experimental_int8_indexer_wk": args.int8_indexer_wk,
               "stats": stats,
               "layout": {"routed_experts": f"backbone layer 1-77: {args.routed_experts}",
                          "mtp_experts": args.mtp_experts,
                          "int8": "W8A8 symmetric channelwise, int8 weight + fp32 weight_scale",
                          "int4": "W4A8 signed int4, compressed-tensors pack-quantized (weight_packed int32 + weight_scale bf16 + weight_shape int64)",
                          "indexer_wk": "BF16 unless --int8-indexer-wk and patched vLLM"}}
    with open(os.path.join(args.output_dir, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
