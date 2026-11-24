# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Hy4 BF16 -> W4A8 INT4 / INT8 / BF16 混合量化脚本 (compressed-tensors)。

输入为 Hy4（含 ``dfly_layers``，不含 ``mtp_layers``）。默认布局：

  * 主干 layer 1-77 的 routed experts -> INT4 对称（默认 per-channel，
    ``--group-size`` 可改 per-group），权重按 8 个 int4 打包到一个 int32 lane，
    激活声明为动态 per-token INT8（W4A8，pack-quantized）。
  * 其余模块（shared experts、layer 0 dense MLP、MLA、indexer、linear_gate、
    router gate、hc_*、norm、embed、lm_head、vit、dfly_layers）原样保留 BF16/FP32。

INT8 = 对称 per-output-channel，FP32 ``weight_scale`` [out, 1]，激活动态 per-token
INT8（W8A8，int-quantized）。可选开关（默认全部关闭，与 FP8 脚本一一对应）：

  * ``--int8-mla``：主干的 q_a/kv_a、q_b、kv_b、o_proj 改为 INT8。
    q_a 与 kv_a 在 vLLM 中融合为 fused_qkv_a_proj，所以同开同关。
  * ``--int8-mlp``：主干 shared experts 和 layer 0 dense MLP 改为 INT8。
  * ``--int8-o-proj``：只把主干 layer 1-77 的 self_attn.o_proj 改为 INT8
    （layer 0 保持 BF16）。已开 --int8-mla 时无需再开。
  * ``--int8-dfly``：dfly_layers.{0..N} 的 MLA（q_a/kv_a/q_b/kv_b/o_proj）、linear_gate、
    dense MLP（gate/up/down）以及 hidden_correction.{gate,up,down}_proj 改为 INT8。
    mtp_fusion、confidence_head 始终保持 BF16。与 --int8-mla/--int8-mlp 互相独立。
  * ``--int8-indexer-wq-b``：indexer.wq_b 改为 INT8。
  * ``--int8-indexer-wk``：实验性，indexer.wk 改为 INT8。stock vLLM 的
    wk_weights_proj 使用 quant_config=None，需要人工补丁才能加载。
  * ``--int8-linear-gate``：实验性，主干 layer 1-77 的 self_attn.linear_gate 改为 INT8
    （layer 0 保持 BF16）。HPC 融合 gated-MLA 内核要求 BF16 gate，开启后退回
    eager 的 linear + sigmoid + 乘法。

不提供 indexer.weights_proj（没有量化意义且与 wk 融合）和 router mlp.gate 的量化。

源 routed experts 是融合三维张量 ``mlp.experts.gate_up_proj`` [E, 2I, H] 和
``mlp.experts.down_proj`` [E, H, I]，本脚本拆成 vLLM loader 可识别的
``experts.{id}.{gate,up,down}_proj``：

  INT4: ``.weight_packed`` int32 [N, K/8]、``.weight_scale`` bf16、``.weight_shape`` int64
  INT8: ``.weight`` int8 [N, K]、``.weight_scale`` fp32 [N, 1]

已知限制：
  * stock vLLM 的 W4A8 INT4 MoE 后端目前只有 CPU 实现，GPU kernel/loader 需另行补齐。
  * sglang 的 normalize_hyv4_weight_scale 会把所有 ``.weight_scale`` 改名为
    ``.weight_scale_inv``，compressed-tensors 产物在 sglang 上可能加载失败。
  * vLLM 的 dfly ``_validate_checkpoint_quantization`` 与 sglang 的
    ``check_hyv4_dfly_quant_config`` 只放行 FP8/MXFP4 组，INT4/INT8 配置在 dfly 投机
    解码下会被拒绝，需要另行扩展校验。

示例：

  python quantize_hy4_mixed_int4_int8.py \\
      --input-dir /models/Hy4 --output-dir /models/Hy4-W4A8-INT4

  python quantize_hy4_mixed_int4_int8.py --int8-o-proj --int8-linear-gate --int8-dfly \\
      --input-dir ... --output-dir ...

依赖：``torch``、``safetensors``、``tqdm``。输入必须是 BF16/FP16 checkpoint。
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

INT4_QMAX = 7.0
INT8_QMAX = 127.0

EXPERT_GATE_UP_RE = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.gate_up_proj$")
EXPERT_DOWN_RE = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.down_proj$")
MLA_RE = re.compile(
    r"^model\.layers\.\d+\.self_attn\."
    r"(?:q_a_proj|kv_a_proj_with_mqa|q_b_proj|kv_b_proj|o_proj)\.weight$"
)
INDEXER_WQ_B_RE = re.compile(r"^model\.layers\.\d+\.self_attn\.indexer\.wq_b\.weight$")
INDEXER_WK_RE = re.compile(r"^model\.layers\.\d+\.self_attn\.indexer\.wk\.weight$")
# linear_gate / 单独 o_proj 只量化 layer 1+（layer 0 的 self_attn 保持 BF16，与官方 Hy4-FP8 一致）。
LINEAR_GATE_RE = re.compile(r"^model\.layers\.(?!0\.)\d+\.self_attn\.linear_gate\.weight$")
O_PROJ_RE = re.compile(r"^model\.layers\.(?!0\.)\d+\.self_attn\.o_proj\.weight$")
SHARED_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
DENSE_MLP_RE = re.compile(r"^model\.layers\.0\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$")
# dfly_layers.<N>.*：MLA、linear_gate、dense MLP；dfly_layers.hidden_correction.*：
# gate/up/down。mtp_fusion、confidence_head 不量化。
DFLY_MLA_RE = re.compile(
    r"^model\.dfly_layers\.\d+\.self_attn\."
    r"(?:q_a_proj|kv_a_proj_with_mqa|q_b_proj|kv_b_proj|o_proj|linear_gate)\.weight$"
)
DFLY_MLP_RE = re.compile(
    r"^model\.dfly_layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
DFLY_HC_RE = re.compile(
    r"^model\.dfly_layers\.hidden_correction\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# config 正则前缀：排除 dfly_layers，使主干开关不会误匹配 dfly（``.*`` 会吃掉 ``model.dfly_``）。
NOT_DFLY = r"^(?!.*dfly_layers)"


# --------------------------------------------------------------------------- #
# 量化器
# --------------------------------------------------------------------------- #

def output_scale_name(name: str) -> str:
    """``x.weight`` -> ``x.weight_scale``。"""
    return name[: -len(".weight")] + ".weight_scale"


def quantize_int8_channelwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 INT8 per-output-channel：返回 int8 [N,K] 和 fp32 scale [N,1]。"""
    if weight.ndim != 2:
        raise ValueError(f"INT8 channelwise expects a 2-D weight, got {tuple(weight.shape)}")
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / INT8_QMAX
    q = torch.round(w / scale).clamp(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


def quantize_int4(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 INT4：返回 int8 存储的 [-8,7] 码值和 fp32 scale。

    group_size == -1: per-channel，scale [N,1]；否则 per-group，scale [N,K/g]。
    """
    if weight.ndim != 2:
        raise ValueError(f"INT4 quantization expects a 2-D weight, got {tuple(weight.shape)}")
    n, k = weight.shape
    w = weight.float()
    if group_size == -1:
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
    """按升序 nibble 把 8 个有符号 int4 打进一个 int32（compressed-tensors pack-quantized）。"""
    if q.dtype != torch.int8 or q.ndim != 2 or q.shape[1] % 8:
        raise ValueError(f"invalid INT4 packing input: {tuple(q.shape)}/{q.dtype}")
    u = (q.to(torch.int32) + 8) & 0xF
    u = u.view(q.shape[0], q.shape[1] // 8, 8)
    shifts = torch.arange(8, device=q.device, dtype=torch.int32) * 4
    return (u << shifts).sum(dim=-1).to(torch.int32).contiguous()


def unpack_int32_to_int4(packed: torch.Tensor) -> torch.Tensor:
    """pack_int4_to_int32 的逆运算（仅用于自检）。"""
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    u = (packed.unsqueeze(-1) >> shifts) & 0xF
    return (u - 8).to(torch.int8).flatten(1)


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().contiguous()


def _is_expert_fused(name: str) -> bool:
    return EXPERT_GATE_UP_RE.match(name) is not None or EXPERT_DOWN_RE.match(name) is not None


def _is_int8_target(name: str, opts: dict) -> bool:
    """True if this non-expert weight is quantized to INT8 under ``opts``."""
    return bool(
        (opts["int8_mla"] and MLA_RE.match(name))
        or (opts["int8_indexer_wq_b"] and INDEXER_WQ_B_RE.match(name))
        or (opts["int8_indexer_wk"] and INDEXER_WK_RE.match(name))
        or (opts["int8_linear_gate"] and LINEAR_GATE_RE.match(name))
        or (opts["int8_mlp"] and (SHARED_RE.match(name) or DENSE_MLP_RE.match(name)))
        or (opts["int8_o_proj"] and O_PROJ_RE.match(name))
        or (opts["int8_dfly"] and (DFLY_MLA_RE.match(name) or DFLY_MLP_RE.match(name)
                                   or DFLY_HC_RE.match(name)))
    )


def _check_float(name: str, t: torch.Tensor) -> None:
    if t.dtype not in FLOAT_DTYPES:
        raise ValueError(f"{name}: expected a BF16/FP16 source tensor, got {t.dtype}")


# --------------------------------------------------------------------------- #
# 转换
# --------------------------------------------------------------------------- #

def _quantize_expert(
    name: str, tensor: torch.Tensor, opts: dict, device: torch.device,
    out: dict[str, torch.Tensor], stats: dict[str, int],
) -> None:
    """Split one fused [E,*,*] expert tensor and quantize each expert to INT4."""
    _check_float(name, tensor)
    if tensor.ndim != 3:
        raise ValueError(f"fused expert tensor must be 3-D: {name} {tuple(tensor.shape)}")
    is_gate_up = name.endswith("gate_up_proj")
    if is_gate_up and tensor.shape[1] % 2:
        raise ValueError(f"gate_up_proj output dim must be even: {tuple(tensor.shape)}")
    base = name.rsplit(".", 1)[0]
    # One expert at a time keeps GPU memory at a single 2-D matrix.
    for expert_id in range(tensor.shape[0]):
        weight = tensor[expert_id].to(device)
        if is_gate_up:
            half = weight.shape[0] // 2
            pieces = [("gate_proj", weight[:half]), ("up_proj", weight[half:])]
        else:
            pieces = [("down_proj", weight)]
        for projection, w in pieces:
            prefix = f"{base}.{expert_id}.{projection}"
            q, s = quantize_int4(w, opts["group_size"])
            out[f"{prefix}.weight_packed"] = _to_cpu(pack_int4_to_int32(q))
            out[f"{prefix}.weight_scale"] = _to_cpu(s.to(torch.bfloat16))
            out[f"{prefix}.weight_shape"] = torch.tensor(list(q.shape), dtype=torch.int64)
            stats["routed_expert_int4"] += 1
        del weight, pieces


def _convert_one_shard(
    input_path: str, output_path: str, opts: dict,
    stats: dict[str, int], device: torch.device,
) -> None:
    state: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state[name] = f.get_tensor(name)

    out: dict[str, torch.Tensor] = {}
    fused_names: set[str] = set()
    for name, tensor in state.items():
        if _is_expert_fused(name):
            _quantize_expert(name, tensor, opts, device, out, stats)
            fused_names.add(name)
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            raise ValueError(f"{name}: source is already FP8; this script expects BF16 input")
        if _is_int8_target(name, opts) and tensor.ndim == 2 and tensor.dtype in FLOAT_DTYPES:
            q, s = quantize_int8_channelwise(tensor.to(device))
            out[name] = _to_cpu(q)
            out[output_scale_name(name)] = _to_cpu(s)
            stats["dfly_int8" if ".dfly_layers." in name else "int8_weights"] += 1
            continue
        out[name] = tensor
        stats["kept"] += 1

    # Every input tensor must be written or replaced by split experts.
    unaccounted = sorted(set(state) - set(out) - fused_names)
    if unaccounted:
        raise ValueError(
            f"{os.path.basename(input_path)}: {len(unaccounted)} tensor(s) dropped, "
            f"e.g. {unaccounted[:5]}"
        )
    save_file(out, output_path)


# --------------------------------------------------------------------------- #
# compressed-tensors 配置
# --------------------------------------------------------------------------- #

def _int8_weights() -> dict:
    return {
        "actorder": None, "block_structure": None, "dynamic": False,
        "group_size": -1, "num_bits": 8, "observer": "minmax",
        "observer_kwargs": {}, "strategy": "channel", "symmetric": True, "type": "int",
    }


def _int4_weights(group_size: int) -> dict:
    return {
        "actorder": None, "block_structure": None, "dynamic": False,
        "group_size": group_size, "num_bits": 4, "observer": "minmax",
        "observer_kwargs": {}, "strategy": "channel" if group_size == -1 else "group",
        "symmetric": True, "type": "int",
    }


def _int8_input() -> dict:
    return {
        "actorder": None, "block_structure": None, "dynamic": True,
        "group_size": None, "num_bits": 8, "observer": None,
        "observer_kwargs": {}, "strategy": "token", "symmetric": True, "type": "int",
    }


_EXPERT_SUFFIX = r"mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)"
_MLA_SUFFIX = (r"self_attn\.(?:fused_qkv_a_proj(?:_with_mqa)?|q_a_proj|"
               r"kv_a_proj_with_mqa|q_b_proj|kv_b_proj|o_proj)")
_DFLY_MLP_SUFFIX = r"mlp\.(?:gate_up_proj|gate_proj|up_proj|down_proj)"
_MLA_NO_O_SUFFIX = (r"self_attn\.(?:fused_qkv_a_proj(?:_with_mqa)?|q_a_proj|"
                    r"kv_a_proj_with_mqa|q_b_proj|kv_b_proj)")
_O_PROJ_L1_SUFFIX = r"layers\.(?!0\.)\d+\.self_attn\.o_proj"
_LINEAR_GATE_L1_SUFFIX = r"layers\.(?!0\.)\d+\.self_attn\.linear_gate"


def build_compression_config(opts: dict, cfg: dict) -> dict:
    hidden = int(cfg["num_hidden_layers"])
    backbone = "|".join(str(i) for i in range(1, hidden))
    backbone_target = rf"re:{NOT_DFLY}.*layers\.(?:{backbone})\.{_EXPERT_SUFFIX}$"
    int8_common = {"input_activations": _int8_input(), "output_activations": None,
                   "weights": _int8_weights(), "format": "int-quantized"}
    int4_common = {"input_activations": _int8_input(), "output_activations": None,
                   "weights": _int4_weights(opts["group_size"]), "format": "pack-quantized"}

    groups: dict[str, dict] = {
        "group_int4_routed_experts": {"targets": [backbone_target], **int4_common},
    }
    if opts["int8_mla"]:
        groups["group_int8_mla"] = {"targets": [rf"re:{NOT_DFLY}.*{_MLA_SUFFIX}$"], **int8_common}
    elif opts["int8_o_proj"]:
        groups["group_int8_o_proj"] = {
            "targets": [rf"re:{NOT_DFLY}.*{_O_PROJ_L1_SUFFIX}$"], **int8_common}
    if opts["int8_indexer_wq_b"]:
        groups["group_int8_indexer_wq_b"] = {
            "targets": [r"re:.*self_attn\.indexer\.wq_b$"], **int8_common}
    if opts["int8_indexer_wk"]:
        groups["group_int8_indexer_wk_EXPERIMENTAL"] = {
            "targets": [r"re:.*self_attn\.indexer\.(?:wk_weights_proj|wk)$"], **int8_common}
    if opts["int8_linear_gate"]:
        groups["group_int8_linear_gate"] = {
            "targets": [rf"re:{NOT_DFLY}.*{_LINEAR_GATE_L1_SUFFIX}$"], **int8_common}
    if opts["int8_mlp"]:
        groups["group_int8_mlp"] = {
            "targets": [rf"re:{NOT_DFLY}.*shared_experts\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$",
                        rf"re:{NOT_DFLY}.*layers\.0\.mlp\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$"],
            **int8_common}
    if opts["int8_dfly"]:
        groups["group_int8_dfly"] = {
            "targets": [rf"re:.*dfly_layers\.\d+\.{_MLA_SUFFIX}$",
                        r"re:.*dfly_layers\.\d+\.self_attn\.linear_gate$",
                        rf"re:.*dfly_layers\.\d+\.{_DFLY_MLP_SUFFIX}$",
                        r"re:.*dfly_layers\.hidden_correction\."
                        r"(?:gate_up_proj|gate_proj|up_proj|down_proj)$"],
            **int8_common}

    # ``.*gate$`` would also match ``self_attn.linear_gate``; always exclude it from the
    # router rule. linear_gate is ignored explicitly below when it is not quantized.
    router_gate = r"re:.*(?<!linear_)(?:router\.)?gate$"
    ignore = [
        router_gate, r"re:.*(?:router\.)?gate\..*",
        r"re:.*hc_.*", r"re:.*(?:input_layernorm|post_attention_layernorm)$",
        r"re:.*(?:q_a_layernorm|kv_a_layernorm|norm)$", r"re:.*learnable_sink_param$",
        r"re:.*indexer\.k_norm(?:\..*)?$", r"re:.*indexer\.weights_proj$",
        # ViT: checkpoint name is ``vit.*``, vLLM runtime name is ``visual.*``
        # (hf_to_vllm_mapper). ViT ``self_attn.o_proj`` would otherwise match the MLA target.
        r"lm_head", r"re:.*embed_tokens.*", r"re:.*vit\..*", r"re:.*visual\..*",
        r"re:.*(?:eh_proj|enorm|hnorm|final_layernorm)$",
        # dfly 的 mtp_fusion / confidence_head / final_norm 始终保持 BF16。
        r"re:.*dfly_layers\.(?:mtp_fusion|confidence_head|final_norm)(?:\..*)?$",
    ]
    if not opts["int8_linear_gate"]:
        ignore.append(rf"re:{NOT_DFLY}.*linear_gate$")
    if not opts["int8_indexer_wk"]:
        ignore.extend([r"re:.*indexer\.wk_weights_proj$", r"re:.*indexer\.wk$"])
    if not opts["int8_mlp"]:
        ignore.extend([rf"re:{NOT_DFLY}.*shared_experts\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$",
                       rf"re:{NOT_DFLY}.*layers\.0\.mlp\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$"])
    if not opts["int8_mla"]:
        ignore.append(rf"re:{NOT_DFLY}.*{_MLA_NO_O_SUFFIX}$")
        # layer 0 的 o_proj 与官方一致保持 BF16；只开 --int8-o-proj 时显式 ignore。
        ignore.append(rf"re:{NOT_DFLY}.*layers\.0\.self_attn\.o_proj$")
        if not opts["int8_o_proj"]:
            ignore.append(rf"re:{NOT_DFLY}.*self_attn\.o_proj$")
    if opts["int8_linear_gate"]:
        ignore.append(rf"re:{NOT_DFLY}.*layers\.0\.self_attn\.linear_gate$")
    if not opts["int8_indexer_wq_b"]:
        ignore.append(r"re:.*indexer\.wq_b$")
    if not opts["int8_dfly"]:
        ignore.append(r"re:.*dfly_layers.*")
    return {"config_groups": groups, "format": "pack-quantized", "ignore": ignore,
            "kv_cache_scheme": None, "quant_method": "compressed-tensors",
            "quantization_status": "compressed"}


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #

def _self_check_numerics() -> None:
    """INT4 打包位布局、打包解包往返，以及 INT8 / INT4 往返误差上界。"""
    q = torch.tensor([[-8, -1, 0, 1, 7, -8, 7, 0]], dtype=torch.int8)
    # u = q + 8 = [0,7,8,9,15,0,15,8], LSB-first nibbles -> 0x8F0F9870 = -1894803344 (int32)
    got = int(pack_int4_to_int32(q).item())
    assert got == -1894803344, f"INT4 pack layout wrong: {got}"
    torch.manual_seed(0)
    rnd = torch.randint(-8, 8, (4, 64), dtype=torch.int8)
    assert torch.equal(unpack_int32_to_int4(pack_int4_to_int32(rnd)), rnd), "INT4 pack/unpack mismatch"

    w = torch.randn(8, 64) * 3.0
    q8, s8 = quantize_int8_channelwise(w)
    assert q8.dtype == torch.int8 and s8.dtype == torch.float32 and tuple(s8.shape) == (8, 1)
    assert bool(((q8.float() * s8 - w).abs() <= s8 / 2 + 1e-6).all()), "INT8 error exceeds scale/2"
    for gs in (-1, 16):
        q4, s4 = quantize_int4(w, gs)
        assert int(q4.min()) >= -8 and int(q4.max()) <= 7
        scale = s4 if gs == -1 else s4.repeat_interleave(gs, dim=1)
        assert bool(((q4.float() * scale - w).abs() <= scale / 2 + 1e-6).all()), "INT4 error exceeds scale/2"
    zq, zs = quantize_int8_channelwise(torch.zeros(2, 4))
    assert bool(torch.isfinite(zs).all()) and int(zq.abs().max()) == 0, "zero row not handled"


def _self_check_config(opts: dict, cfg: dict) -> None:
    """Assert the config resolves each module to the dtype this script writes.

    A mismatch (config says INT8/INT4 but the weight is BF16, or the reverse) leaves
    weight_scale uninitialised and silently breaks the model.
    """
    qc = build_compression_config(opts, cfg)
    hidden = int(cfg["num_hidden_layers"])

    def ignored(name: str) -> bool:
        return any(re.match(p[3:], name) if p.startswith("re:") else p == name
                   for p in qc["ignore"])

    def groups_of(name: str) -> list[str]:
        return [g for g, v in qc["config_groups"].items()
                if any(re.match(t[3:], name) for t in v["targets"])]

    def expect(name: str, group: str | None) -> None:
        got = groups_of(name)
        ok = (ignored(name) or not got) if group is None else (not ignored(name) and got == [group])
        if not ok:
            raise AssertionError(f"config mismatch for {name}: expected {group}, "
                                 f"got groups={got}, ignored={ignored(name)}")

    mla_group = "group_int8_mla" if opts["int8_mla"] else None
    mlp_group = "group_int8_mlp" if opts["int8_mlp"] else None
    wk_group = "group_int8_indexer_wk_EXPERIMENTAL" if opts["int8_indexer_wk"] else None
    dfly_group = "group_int8_dfly" if opts["int8_dfly"] else None
    o_group = mla_group or ("group_int8_o_proj" if opts["int8_o_proj"] else None)
    mla_mods = ("fused_qkv_a_proj", "fused_qkv_a_proj_with_mqa", "q_a_proj",
                "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj")

    for base in [f"model.layers.{i}" for i in (1, 40, hidden - 1)]:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expect(f"{base}.mlp.experts.7.{proj}", "group_int4_routed_experts")
            expect(f"{base}.mlp.shared_experts.{proj}", mlp_group)
        expect(f"{base}.mlp.gate", None)
        expect(f"{base}.self_attn.linear_gate",
               "group_int8_linear_gate" if opts["int8_linear_gate"] else None)
        expect(f"{base}.self_attn.indexer.weights_proj", None)
        for mod in ("self_attn.indexer.wk", "self_attn.indexer.wk_weights_proj"):
            expect(f"{base}.{mod}", wk_group)
        for mod in mla_mods:
            expect(f"{base}.self_attn.{mod}", o_group if mod == "o_proj" else mla_group)
        expect(f"{base}.self_attn.indexer.wq_b",
               "group_int8_indexer_wq_b" if opts["int8_indexer_wq_b"] else None)

    # layer 0 的 self_attn：只有 --int8-mla 会量化（含 o_proj）；linear_gate / 单独 o_proj 不动。
    expect("model.layers.0.self_attn.o_proj", mla_group)
    expect("model.layers.0.self_attn.linear_gate", None)

    # dfly: MLA / linear_gate / dense MLP / hidden_correction 跟 --int8-dfly；其余始终 BF16。
    for base in [f"model.dfly_layers.{i}" for i in (0, 4)]:
        for mod in mla_mods:
            expect(f"{base}.self_attn.{mod}", dfly_group)
        for proj in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
            expect(f"{base}.mlp.{proj}", dfly_group)
        expect(f"{base}.self_attn.linear_gate", dfly_group)
        expect(f"{base}.self_attn.learnable_sink_param", None)
        expect(f"{base}.input_layernorm", None)
    for proj in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
        expect(f"model.dfly_layers.hidden_correction.{proj}", dfly_group)
    for mod in ("mtp_fusion.eh_proj", "mtp_fusion.gate", "confidence_head.proj", "final_norm",
                "hidden_correction.embed_norm"):
        expect(f"model.dfly_layers.{mod}", None)

    # Vision tower (checkpoint ``vit.*``, vLLM runtime ``visual.*``) is never quantized.
    for vis in ("vit", "visual", "model.visual", "vit.vit"):
        expect(f"{vis}.layers.0.self_attn.o_proj", None)
    expect("model.layers.0.mlp.experts.0.down_proj", None)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        expect(f"model.layers.0.mlp.{proj}", mlp_group)

    # The write side must agree with the config.
    probes = {
        "model.layers.5.self_attn.q_b_proj.weight": opts["int8_mla"],
        "model.layers.5.self_attn.indexer.wq_b.weight": opts["int8_indexer_wq_b"],
        "model.layers.5.self_attn.linear_gate.weight": opts["int8_linear_gate"],
        "model.layers.5.self_attn.indexer.wk.weight": opts["int8_indexer_wk"],
        "model.layers.5.mlp.shared_experts.down_proj.weight": opts["int8_mlp"],
        "model.layers.0.mlp.gate_proj.weight": opts["int8_mlp"],
        "model.layers.5.self_attn.o_proj.weight": opts["int8_mla"] or opts["int8_o_proj"],
        "model.layers.0.self_attn.o_proj.weight": opts["int8_mla"],
        "model.layers.0.self_attn.linear_gate.weight": False,
        "model.dfly_layers.0.self_attn.q_b_proj.weight": opts["int8_dfly"],
        "model.dfly_layers.0.self_attn.o_proj.weight": opts["int8_dfly"],
        "model.dfly_layers.0.mlp.gate_proj.weight": opts["int8_dfly"],
        "model.dfly_layers.0.self_attn.linear_gate.weight": opts["int8_dfly"],
        "model.dfly_layers.hidden_correction.gate_proj.weight": opts["int8_dfly"],
        "model.dfly_layers.hidden_correction.down_proj.weight": opts["int8_dfly"],
        "model.dfly_layers.mtp_fusion.eh_proj.weight": False,
        "model.dfly_layers.confidence_head.proj.weight": False,
    }
    for name, quantized in probes.items():
        if bool(_is_int8_target(name, opts)) != bool(quantized):
            raise AssertionError(f"write side disagrees with config for {name}")
    print("Self-check passed: config matches written dtypes (backbone and dfly).")


# --------------------------------------------------------------------------- #
# index / 设备 / worker
# --------------------------------------------------------------------------- #

def _rebuild_index(output_dir: str, model_index: dict) -> None:
    """Rebuild weight_map/total_size by reading the safetensors headers."""
    weight_map: dict[str, str] = {}
    total_size = 0
    for path in sorted(glob(os.path.join(output_dir, "*.safetensors"))):
        with open(path, "rb") as sf:
            header = json.loads(sf.read(int.from_bytes(sf.read(8), "little")))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            weight_map[key] = os.path.basename(path)
            total_size += meta["data_offsets"][1] - meta["data_offsets"][0]
    model_index["weight_map"] = weight_map
    model_index["metadata"] = {"total_size": total_size}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)


def _verify_conversion(input_map: dict[str, str], output_map: dict[str, str],
                       shards: set[str]) -> None:
    """Input tensors absent from the output must be fused experts, nothing else."""
    unexpected = [n for n, s in input_map.items()
                  if s in shards and n not in output_map and not _is_expert_fused(n)]
    if unexpected:
        raise RuntimeError(f"Self-check FAILED: {len(unexpected)} tensor(s) missing from "
                           f"output, e.g. {sorted(unexpected)[:10]}")
    print("Self-check passed: no unintended tensor was dropped.")


def _empty_stats() -> dict[str, int]:
    return {k: 0 for k in ("routed_expert_int4", "int8_weights", "dfly_int8", "kept")}


def _resolve_devices(device_kind: str, gpu_ids: str | None, workers: int | None,
                     num_files: int) -> tuple[list[torch.device], int]:
    """Resolve --device/--gpu-ids/--workers into (devices, effective_workers)."""
    cuda_ok = torch.cuda.is_available()
    if device_kind == "cpu" or (device_kind == "auto" and not cuda_ok):
        return [torch.device("cpu")], 1
    if not cuda_ok:
        raise RuntimeError("--device cuda specified but torch.cuda.is_available() is False")
    ids = ([int(x) for x in gpu_ids.split(",") if x.strip()] if gpu_ids
           else list(range(torch.cuda.device_count())))
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
    parser = ArgumentParser(description="Convert Hy4 BF16 to W4A8 INT4 / INT8 / BF16 mixed precision.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", default="/models/Hy4",
                        help="BF16 Hy4 checkpoint directory.")
    parser.add_argument("--output-dir", default="/models/Hy4-W4A8-INT4",
                        help="Output checkpoint directory (must differ from --input-dir).")
    parser.add_argument("--group-size", type=int, default=-1,
                        help="INT4 group size along K; -1 is per-channel. Applies to every INT4 expert.")
    parser.add_argument("--int8-mlp", action="store_true",
                        help="INT8 backbone shared experts and layer 0 dense MLP.")
    parser.add_argument("--int8-mla", action="store_true",
                        help="INT8 q_a/kv_a/q_b/kv_b/o_proj in the backbone.")
    parser.add_argument("--int8-o-proj", action="store_true",
                        help="INT8 only self_attn.o_proj of backbone layers 1-77 (layer 0 stays BF16). "
                             "Not needed with --int8-mla, which already includes o_proj.")
    parser.add_argument("--int8-dfly", action="store_true",
                        help="INT8 dfly_layers.N MLA (q_a/kv_a/q_b/kv_b/o_proj), linear_gate and dense MLP "
                             "(gate/up/down), plus hidden_correction gate/up/down. "
                             "mtp_fusion and confidence_head always stay BF16.")
    parser.add_argument("--int8-indexer-wq-b", action="store_true", help="INT8 indexer.wq_b.")
    parser.add_argument("--int8-indexer-wk", action="store_true",
                        help="EXPERIMENTAL: INT8 indexer.wk. stock vLLM loads wk_weights_proj "
                             "with quant_config=None, so this needs a patched loader/kernel.")
    parser.add_argument("--int8-linear-gate", action="store_true",
                        help="EXPERIMENTAL: INT8 the backbone MLA output gate (self_attn.linear_gate) "
                             "of layers 1-77 (layer 0 stays BF16). The vLLM HPC fused gated-MLA kernel "
                             "needs a BF16 gate, so this falls back to the eager linear + sigmoid + mul path.")
    parser.add_argument("--start-file", type=int, default=0,
                        help="Zero-based index in sorted shard list to start from.")
    parser.add_argument("--limit-files", type=int, default=None,
                        help="Convert at most this many shards (smoke test).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="auto: all visible GPUs if available, else CPU.")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated CUDA indices, e.g. '0,1,3'. Default: all visible.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes (each holds a whole shard in RAM). "
                             "Default: number of GPUs.")
    parser.add_argument("--num-threads", type=int, default=8, help="Torch CPU threads per worker.")
    args = parser.parse_args()
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must differ")
    if args.group_size != -1 and args.group_size <= 0:
        raise ValueError("--group-size must be -1 or positive")

    with open(os.path.join(args.input_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if args.int8_indexer_wk:
        print("WARNING: --int8-indexer-wk requires a patched vLLM wk_weights_proj loader/kernel.")
    opts = {"group_size": args.group_size,
            "int8_mlp": args.int8_mlp, "int8_mla": args.int8_mla,
            "int8_indexer_wq_b": args.int8_indexer_wq_b, "int8_indexer_wk": args.int8_indexer_wk,
            "int8_linear_gate": args.int8_linear_gate, "int8_dfly": args.int8_dfly,
            "int8_o_proj": args.int8_o_proj}
    _self_check_numerics()
    _self_check_config(opts, cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(args.input_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {args.input_dir}")
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
            for per in tqdm(pool.imap_unordered(_worker_convert_shard, tasks),
                            total=len(tasks), desc=desc):
                for k, v in per.items():
                    stats[k] += v

    for filename in sorted(os.listdir(args.input_dir)):
        src = os.path.join(args.input_dir, filename)
        if (os.path.isfile(src) and not filename.endswith((".safetensors", ".bin"))
                and filename not in ("config.json", "model.safetensors.index.json")):
            shutil.copyfile(src, os.path.join(args.output_dir, filename))
    with open(os.path.join(args.input_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        model_index = json.load(f)
    input_map = dict(model_index["weight_map"])
    _rebuild_index(args.output_dir, model_index)
    _verify_conversion(input_map, model_index["weight_map"], {os.path.basename(p) for p in files})

    cfg["quantization_config"] = build_compression_config(opts, cfg)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, sort_keys=True)
    summary = {
        "input_dir": args.input_dir, "output_dir": args.output_dir,
        "options": opts, "stats": stats,
        "layout": {
            "routed_experts": "backbone layer 1-77: INT4 W4A8 (pack-quantized: weight_packed int32 + "
                              "weight_scale bf16 + weight_shape int64)",
            "others": "BF16/FP32 unchanged unless an --int8-* switch is set (INT8 W8A8 channelwise)",
        },
    }
    with open(os.path.join(args.output_dir, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
