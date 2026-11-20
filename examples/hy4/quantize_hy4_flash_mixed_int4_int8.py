# SPDX-License-Identifier: Apache-2.0
"""
Hy4-flash BF16 -> W4A8 INT4 / W8A8 INT8 / BF16 混合量化脚本 (compressed-tensors)。

输入 ``Hy4-flash-testing``(BF16,融合 3D routed experts,草稿为 ``model.dfly_layers.N``)。默认布局:

  * 主干 layer 1..L-1 的 routed experts -> INT4 对称 per-channel(``--group-size`` 可改 per-group),
    8 个 int4 打进一个 int32,激活动态 per-token INT8(W4A8, pack-quantized)。
    融合 3D ``mlp.experts.{gate_up_proj,down_proj}`` 拆成逐专家 2D
    ``experts.{id}.{gate,up,down}_proj.{weight_packed,weight_scale,weight_shape}``。
  * dfly_layers 的 ``self_attn.{q_b_proj,o_proj}`` 与 dense ``mlp.{gate,up,down}_proj``
    -> 默认 INT8 per-output-channel(FP32 ``weight_scale`` [out,1],动态 per-token INT8)。
    ``--dfly int4`` 改为 INT4,``--dfly bf16`` 保持 BF16。
  * 默认保持 BF16 的开关项:
      ``--int8-attn``         主干 layer 1..L-1 的 self_attn.{q_b_proj,o_proj} 改 INT8
      ``--int8-linear-gate``  主干与 dfly 的 self_attn.linear_gate 一起改 INT8
                              (HPC 融合 gated-MLA 内核要 BF16 gate,开启后可能退回 eager 路径)
      ``--int8-dense-mlp``    主干 layer 0 的 dense mlp.{gate,up,down}_proj 改 INT8 per-channel
      ``--int8-shared-experts`` 主干 layer 1..L-1 的 shared_experts.{gate,up,down}_proj 改 INT8 per-channel
  * 原样保留: layer 0 的 attn 与 hc_*(dense MLP 除非开 --int8-dense-mlp)、q_a/kv_a/kv_b、
    shared_experts(除非开 --int8-shared-experts)、mlp.gate、routed_expert_{up,down}_proj、
    hc_*、compressed_indexer、engram_layer、norm、embed、lm_head、vit.*、
    dfly 的 confidence_head/hidden_correction/mtp_fusion。

已知限制:
  * 输出目录会同时写两份 config:``config.json``(INT4 targets 为 ``re:`` 正则,默认)和
    ``config.humming.json``(INT4 targets 为字面量 ``Linear``,供 ``vllm --quantization humming``)。
    两份只有 ``group_int4.targets`` 不同。使用 humming 时把后者复制/改名为 ``config.json``。
    ``Linear`` 会命中所有未被 ignore 的线性层,同一份 config 只有一个 group 能用 ``Linear``,
    所以 humming 版要求 ``group_int8`` 里没有主干层目标,即不带任何 ``--int8-*`` 开关;
    否则那些 INT8 主干层会被按 INT4 schema 建参数,加载时报 ``RowvLLMParameter ... missing 'input_dim'``。
    ``--dfly int8/int4`` 只影响 dfly 层,不开 DFly 时不受影响。
  * stock vLLM 的 W4A8 INT4 MoE 后端可能只有 CPU 实现;hy_v4 的 dfly_config 校验只放行 FP8,
    INT 模型需要补丁后才能加载。
  * sglang 会把 ``.weight_scale`` 改名为 ``.weight_scale_inv``,且无 Flash 实现。

示例:
  python quantize_hy4_flash_mixed_int4_int8.py \\
      --input-dir /models/Hy4-flash-testing --output-dir /models/Hy4-flash-W4A8
  python quantize_hy4_flash_mixed_int4_int8.py --dfly int4 --int8-attn --int8-linear-gate ...

依赖: torch、safetensors、tqdm。输入必须是 BF16/FP16 checkpoint。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

INT4_QMAX = 7.0
INT8_QMAX = 127.0
FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
ATTN_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_b_proj|o_proj|linear_gate)\.weight$")
DFLY_RE = re.compile(
    r"^model\.dfly_layers\.(\d+)\.(self_attn\.(?:q_b_proj|o_proj|linear_gate)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
DENSE_RE = re.compile(r"^model\.layers\.0\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$")
SHARED_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)\.weight$")


def text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


# --------------------------------------------------------------------------- #
# 量化器
# --------------------------------------------------------------------------- #

def quantize_int8_channelwise(weight: torch.Tensor):
    if weight.ndim != 2:
        raise ValueError(f"INT8 expects a 2-D weight, got {tuple(weight.shape)}")
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / INT8_QMAX
    q = torch.round(w / scale).clamp(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q.contiguous(), scale.float().contiguous()


def quantize_int4(weight: torch.Tensor, group_size: int):
    """对称 INT4: 返回 int8 存储的 [-8,7] 码值与 fp32 scale。group_size=-1 为 per-channel [N,1]。"""
    if weight.ndim != 2:
        raise ValueError(f"INT4 expects a 2-D weight, got {tuple(weight.shape)}")
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
    if q.dtype != torch.int8 or q.ndim != 2 or q.shape[1] % 8:
        raise ValueError(f"invalid INT4 packing input: {tuple(q.shape)}/{q.dtype}")
    u = (q.to(torch.int32) + 8) & 0xF
    u = u.view(q.shape[0], q.shape[1] // 8, 8)
    shifts = torch.arange(8, device=q.device, dtype=torch.int32) * 4
    return (u << shifts).sum(dim=-1).to(torch.int32).contiguous()


def unpack_int32_to_int4(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    return (((packed.unsqueeze(-1) >> shifts) & 0xF) - 8).to(torch.int8).flatten(1)


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().contiguous()


# --------------------------------------------------------------------------- #
# 写入侧分类: 张量名 -> "int4" / "int8" / None(原样保留)
# --------------------------------------------------------------------------- #

def is_expert(name: str) -> bool:
    return EXPERT_RE.match(name) is not None


def weight_mode(name: str, opts: dict) -> str | None:
    m = EXPERT_RE.match(name)
    if m:
        return "int4" if int(m.group(1)) >= 1 else None
    m = ATTN_RE.match(name)
    if m:
        if int(m.group(1)) < 1:
            return None
        if m.group(2) == "linear_gate":
            return "int8" if opts["int8_linear_gate"] else None
        return "int8" if opts["int8_attn"] else None
    m = DFLY_RE.match(name)
    if m:
        if m.group(2).endswith("linear_gate"):
            return "int8" if opts["int8_linear_gate"] else None
        return None if opts["dfly"] == "bf16" else opts["dfly"]
    if DENSE_RE.match(name):
        return "int8" if opts["int8_dense_mlp"] else None
    m = SHARED_RE.match(name)
    if m and int(m.group(1)) >= 1:
        return "int8" if opts["int8_shared_experts"] else None
    return None


def _write_2d(out: dict, name: str, mode: str, w: torch.Tensor, opts: dict) -> None:
    base = name[: -len(".weight")]
    if mode == "int8":
        q, s = quantize_int8_channelwise(w)
        out[f"{base}.weight"] = _cpu(q)
        out[f"{base}.weight_scale"] = _cpu(s)
    else:
        q, s = quantize_int4(w, opts["group_size"])
        out[f"{base}.weight_packed"] = _cpu(pack_int4_to_int32(q))
        out[f"{base}.weight_scale"] = _cpu(s.to(torch.bfloat16))
        out[f"{base}.weight_shape"] = torch.tensor(list(q.shape), dtype=torch.int64)


def _quantize_fused_expert(name: str, t: torch.Tensor, opts: dict, device, out: dict, stats: dict) -> None:
    if t.ndim != 3:
        raise ValueError(f"{name}: fused expert must be 3-D, got {tuple(t.shape)}")
    is_gate_up = name.endswith("gate_up_proj")
    if is_gate_up and t.shape[1] % 2:
        raise ValueError(f"{name}: gate_up_proj output dim must be even")
    base = name.rsplit(".", 1)[0]
    for e in range(t.shape[0]):  # 一次一个专家,显存只占一个 2D 矩阵
        w = t[e].to(device)
        if is_gate_up:
            half = w.shape[0] // 2
            pieces = [("gate_proj", w[:half]), ("up_proj", w[half:])]
        else:
            pieces = [("down_proj", w)]
        for proj, pw in pieces:
            q, s = quantize_int4(pw, opts["group_size"])
            p = f"{base}.{e}.{proj}"
            out[f"{p}.weight_packed"] = _cpu(pack_int4_to_int32(q))
            out[f"{p}.weight_scale"] = _cpu(s.to(torch.bfloat16))
            out[f"{p}.weight_shape"] = torch.tensor(list(q.shape), dtype=torch.int64)
            stats["expert_int4"] += 1


def convert_one_shard(input_path: str, output_path: str, opts: dict, stats: dict, device) -> None:
    out: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if t.dtype == torch.float8_e4m3fn:
                raise ValueError(f"{name}: source is already FP8; expected BF16 input")
            mode = weight_mode(name, opts)
            if mode is None:
                out[name] = t
                stats["kept"] += 1
                continue
            if t.dtype not in FLOAT_DTYPES:
                raise ValueError(f"{name}: expected float source, got {t.dtype}")
            if is_expert(name):
                _quantize_fused_expert(name, t, opts, device, out, stats)
            else:
                if t.ndim != 2:
                    raise ValueError(f"{name}: expected 2-D, got {tuple(t.shape)}")
                _write_2d(out, name, mode, t.to(device), opts)
                stats[f"{mode}_weights"] += 1
    save_file(out, output_path, metadata={"format": "pt"})


# --------------------------------------------------------------------------- #
# compressed-tensors 配置
# --------------------------------------------------------------------------- #

def _w(num_bits: int, group_size: int) -> dict:
    return {"actorder": None, "block_structure": None, "dynamic": False,
            "group_size": None if group_size == -1 else group_size, "num_bits": num_bits,
            "observer": "minmax", "observer_kwargs": {},
            "strategy": "channel" if group_size == -1 else "group", "symmetric": True, "type": "int"}


def _act() -> dict:
    return {"actorder": None, "block_structure": None, "dynamic": True, "group_size": None,
            "num_bits": 8, "observer": None, "observer_kwargs": {}, "strategy": "token",
            "symmetric": True, "type": "int"}


def build_quantization_config(opts: dict, cfg: dict, humming: bool = False) -> dict:
    """humming=True 时 INT4 组的 targets 写成字面量 ``Linear``。

    vLLM 的 humming 后端(compressed_tensors_get_config)只识别字面量 ``Linear`` 目标,
    ``re:`` 正则匹配不到 MoE 专家,专家会退回未量化路径。代价:``Linear`` 会匹配所有未被
    ignore 的线性层,所以 ignore 必须覆盖其余层;且同一份 config 里只有一个 group 能用
    ``Linear``,INT8 的 dfly 层无法再单独表达,humming 版必须配 ``--dfly bf16`` 或不带 dfly 使用。
    """
    layers = int(text_cfg(cfg)["num_hidden_layers"])
    bb = "|".join(str(i) for i in range(1, layers))
    main = rf"(?<!dfly_)layers\.(?:{bb})"  # 主干层 1..L-1,不会匹配 dfly_layers.N
    # Flash 的专家参数在 routed_experts 下时,运行时名可能带 routed_experts 前缀
    expert = rf"re:.*{main}\.mlp\.experts\.(?:routed_experts\.)?\d+\.(?:gate_proj|up_proj|down_proj)$"
    dfly_attn = r"dfly_layers\.\d+\.self_attn\.(?:q_b_proj|o_proj)"
    dfly_mlp = r"dfly_layers\.\d+\.mlp\.(?:gate_up_proj|gate_proj|up_proj|down_proj)"
    int8_t, int4_t = [], [expert]
    if opts["int8_attn"]:
        int8_t.append(rf"re:.*{main}\.self_attn\.(?:q_b_proj|o_proj)$")
    if opts["int8_linear_gate"]:
        int8_t.append(rf"re:.*(?:{main}|dfly_layers\.\d+)\.self_attn\.linear_gate$")
    if opts["int8_dense_mlp"]:  # 主干 layer 0 的 dense MLP(vLLM 里 gate/up 会融合成 gate_up_proj)
        int8_t.append(r"re:.*(?<!dfly_)layers\.0\.mlp\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$")
    if opts["int8_shared_experts"]:
        int8_t.append(rf"re:.*{main}\.mlp\.shared_experts\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$")
    dfly_t = [rf"re:.*{dfly_attn}$", rf"re:.*{dfly_mlp}$"]
    if opts["dfly"] == "int8":
        int8_t += dfly_t
    elif opts["dfly"] == "int4":
        int4_t += dfly_t

    int8_common = {"input_activations": _act(), "output_activations": None,
                   "weights": _w(8, -1), "format": "int-quantized"}
    int4_common = {"input_activations": _act(), "output_activations": None,
                   "weights": _w(4, opts["group_size"]), "format": "pack-quantized"}
    groups = {"group_int4": {"targets": ["Linear"] if humming else int4_t, **int4_common}}
    if int8_t:
        groups["group_int8"] = {"targets": int8_t, **int8_common}

    ignore = [
        "lm_head", r"re:.*embed_tokens.*", r"re:.*vit\..*", r"re:.*visual\..*",
        r"re:.*mlp\.gate$", r"re:.*mlp\.gate\..*", r"re:.*hc_.*",
        r"re:.*engram_layer\..*", r"re:.*compressed_indexer\..*",
        r"re:.*self_attn\.indexer\..*",  # sglang 把 compressed_indexer 改名为 indexer
        r"re:.*(?:q_a_proj|kv_a_proj_with_mqa|kv_b_proj)$", r"re:.*routed_expert_(?:up|down)_proj$",
        r"re:.*dfly_layers\.(?:confidence_head|hidden_correction|mtp_fusion)\..*",
    ]
    # 主干 layer 0 不量化;不能误伤 dfly_layers.0。开 --int8-dense-mlp 时放行 layer 0 的 dense MLP。
    # (?:\.linear)? 不可省:sglang 的 should_ignore_layer 对融合层(gate_up_proj)会拆成分片查
    # ignore,查不中时还会用 "<shard>.linear" 再查一遍;若负向前瞻只锚 $,这个探测名会让整条
    # ignore 反而命中,layer 0 的 gate_up_proj 就被当成未量化层,weight_scale 无处可放。
    if opts["int8_dense_mlp"]:
        ignore.append(
            r"re:.*(?<!dfly_)layers\.0\."
            r"(?!mlp\.(?:gate_up_proj|gate_proj|up_proj|down_proj)(?:\.linear)?$).*"
        )
    else:
        ignore.append(r"re:.*(?<!dfly_)layers\.0\..*")
    if not opts["int8_shared_experts"]:
        ignore.append(r"re:.*shared_experts\..*")
    if not opts["int8_linear_gate"]:
        ignore.append(r"re:.*self_attn\.linear_gate$")
    if not opts["int8_attn"]:
        ignore.append(rf"re:.*{main}\.self_attn\.(?:q_b_proj|o_proj)$")
    if opts["dfly"] == "bf16":
        ignore += [rf"re:.*{dfly_attn}$", rf"re:.*{dfly_mlp}$"]
    return {"config_groups": groups, "format": "pack-quantized", "ignore": ignore,
            "kv_cache_scheme": None, "quant_method": "compressed-tensors",
            "quantization_status": "compressed"}


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #

def self_check_numerics() -> None:
    q = torch.tensor([[-8, -1, 0, 1, 7, -8, 7, 0]], dtype=torch.int8)
    got = int(pack_int4_to_int32(q).item())
    assert got == -1894803344, f"INT4 pack layout wrong: {got}"
    torch.manual_seed(0)
    rnd = torch.randint(-8, 8, (4, 64), dtype=torch.int8)
    assert torch.equal(unpack_int32_to_int4(pack_int4_to_int32(rnd)), rnd)
    w = torch.randn(8, 64) * 3.0
    q8, s8 = quantize_int8_channelwise(w)
    assert tuple(s8.shape) == (8, 1) and bool(((q8.float() * s8 - w).abs() <= s8 / 2 + 1e-6).all())
    for gs in (-1, 16):
        q4, s4 = quantize_int4(w, gs)
        sc = s4 if gs == -1 else s4.repeat_interleave(gs, dim=1)
        assert int(q4.min()) >= -8 and int(q4.max()) <= 7
        assert bool(((q4.float() * sc - w).abs() <= sc / 2 + 1e-6).all())


def self_check_config(opts: dict, cfg: dict, humming: bool = False) -> None:
    """对代表性运行时模块名,断言 config 的解析结果与写入侧 weight_mode 一致。"""
    qc = build_quantization_config(opts, cfg, humming)
    layers = int(text_cfg(cfg)["num_hidden_layers"])

    def ignored(n: str) -> bool:
        return any(re.match(p[3:], n) if p.startswith("re:") else p == n for p in qc["ignore"])

    def hit_target(t: str, n: str) -> bool:
        # 字面量 "Linear" 对任何线性层都命中(humming 只认这种写法)
        return t == "Linear" or (t.startswith("re:") and re.match(t[3:], n) is not None)

    def group_of(n: str):
        if ignored(n):
            return None
        hit = [g for g, v in qc["config_groups"].items() if any(hit_target(t, n) for t in v["targets"])]
        if len(hit) > 1:
            raise AssertionError(f"{n} matches several groups {hit}")
        return hit[0].removeprefix("group_") if hit else None

    cases = []
    # 检查点里主干 layer>=1 没有 mlp.{gate,up,down}_proj(只有 layer 0 是 dense MLP),这几个名字是
    # 正则版用来确认"不会误匹配"的假想名;字面量 Linear 对任何名字都命中,humming 模式下跳过它们。
    absent_in_moe_layers = ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj")
    for i in (0, 1, layers // 2, layers - 1):
        for m in ("self_attn.q_b_proj", "self_attn.o_proj", "self_attn.linear_gate",
                  "self_attn.q_a_proj", "self_attn.kv_b_proj", "mlp.shared_experts.down_proj",
                  "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
                  "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"):
            if humming and i >= 1 and m in absent_in_moe_layers:
                continue
            cases.append((f"model.layers.{i}.{m}", f"model.layers.{i}.{m}.weight"))
        for p in ("gate_proj", "up_proj", "down_proj"):
            cases.append((f"model.layers.{i}.mlp.experts.7.{p}", f"model.layers.{i}.mlp.experts.gate_up_proj"
                          if p != "down_proj" else f"model.layers.{i}.mlp.experts.down_proj"))
    for d in (0, 2, 4):  # dfly 第 0 层曾被 layers\.0 的 ignore 误伤,首中尾都查
        for m in ("self_attn.q_b_proj", "self_attn.o_proj", "self_attn.linear_gate", "self_attn.kv_b_proj",
                  "self_attn.q_a_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            cases.append((f"model.dfly_layers.{d}.{m}", f"model.dfly_layers.{d}.{m}.weight"))
    bad = []
    for module, tensor in cases:
        want = weight_mode(tensor, opts)
        if group_of(module) != want:
            bad.append((module, group_of(module), want))
    for m in ("hidden_correction.down_proj", "hidden_correction.gate_proj", "mtp_fusion.eh_proj",
              "confidence_head.proj", "mtp_fusion.gate"):
        n = f"model.dfly_layers.{m}"
        if group_of(n) is not None or weight_mode(f"{n}.weight", opts) is not None:
            bad.append((n, group_of(n), None))
    for n in ("lm_head", "vit.layers.0.self_attn.o_proj", "visual.layers.0.self_attn.o_proj"):
        if group_of(n) is not None:
            bad.append((n, group_of(n), None))
    # 复现 sglang should_ignore_layer 的融合层探测:分片名查不中 ignore 时会追加 ".linear"
    # 再查一次,被量化的融合层必须对这两种探测名都不被 ignore 命中。
    fused_probes = [("model.layers.0.mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj"), 0)]
    for i in (1, layers - 1):
        fused_probes.append((f"model.layers.{i}.mlp.shared_experts.gate_up_proj",
                             ("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"), i))
    for fused, shards, i in fused_probes:
        if group_of(fused) is None:
            continue
        for shard in shards:
            for probe in (f"model.layers.{i}.{shard}", f"model.layers.{i}.{shard}.linear"):
                if ignored(probe):
                    bad.append((probe, "ignored by sglang fused-shard probe", group_of(fused)))
    if bad:
        raise AssertionError(f"config/write-side mismatch: {bad[:10]}")
    print(f"Self-check passed: config 与写入的 dtype 一致(含 dfly 第 0/2/4 层){' [humming]' if humming else ''}。")


def rebuild_index(output_dir: str, model_index: dict) -> None:
    weight_map, total = {}, 0
    for path in sorted(glob(os.path.join(output_dir, "*.safetensors"))):
        with open(path, "rb") as sf:
            header = json.loads(sf.read(int.from_bytes(sf.read(8), "little")))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            weight_map[key] = os.path.basename(path)
            total += meta["data_offsets"][1] - meta["data_offsets"][0]
    model_index["weight_map"] = weight_map
    model_index["metadata"] = {"total_size": total}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)


def verify_conversion(input_map: dict, output_map: dict, shards: set, opts: dict) -> None:
    """逐个输入键核对输出: 保留键必须在;INT8 要 scale;INT4 要 packed/scale/shape;融合专家各层键数一致。"""
    out = set(output_map)
    errs = []
    expert_counts: dict[str, int] = {}
    for n, s in input_map.items():
        if s not in shards:
            continue
        mode = weight_mode(n, opts)
        if mode is None:
            if n not in out:
                errs.append(f"missing {n}")
        elif is_expert(n):
            pre = n.rsplit(".", 1)[0] + "."
            expert_counts[n] = sum(1 for k in out if k.startswith(pre) and k.endswith(".weight_packed")
                                   and k[len(pre):].split(".")[1] in
                                   (("gate_proj", "up_proj") if n.endswith("gate_up_proj") else ("down_proj",)))
        else:
            base = n[: -len(".weight")]
            need = [f"{base}.weight", f"{base}.weight_scale"] if mode == "int8" else \
                   [f"{base}.weight_packed", f"{base}.weight_scale", f"{base}.weight_shape"]
            errs += [f"missing {k}" for k in need if k not in out]
            if mode == "int4" and n in out:
                errs.append(f"INT4 weight should not keep original key {n}")
    if expert_counts:
        per_proj = {v // 2 if k.endswith("gate_up_proj") else v for k, v in expert_counts.items()}
        if len(per_proj) != 1:
            errs.append(f"experts per layer inconsistent: {sorted(per_proj)}")
    if errs:
        raise RuntimeError(f"Self-check FAILED ({len(errs)}): {errs[:10]}")
    print(f"Self-check passed: 输出键完整(融合专家层数 {len(expert_counts) // 2})。")


def empty_stats() -> dict:
    return {k: 0 for k in ("expert_int4", "int4_weights", "int8_weights", "kept")}


def resolve_devices(kind: str, gpu_ids, workers, n_files: int):
    cuda_ok = torch.cuda.is_available()
    if kind == "cpu" or (kind == "auto" and not cuda_ok):
        return [torch.device("cpu")], 1
    if not cuda_ok:
        raise RuntimeError("--device cuda specified but CUDA is unavailable")
    ids = [int(x) for x in gpu_ids.split(",") if x.strip()] if gpu_ids else list(range(torch.cuda.device_count()))
    if not ids:
        raise RuntimeError("no CUDA device selected")
    devices = [torch.device(f"cuda:{i}") for i in ids]
    eff = len(devices) if workers is None else max(1, workers)
    return devices, min(eff, max(1, n_files))


def worker_convert_shard(task: tuple) -> dict:
    input_path, output_path, opts, gpu_id, num_threads = task
    torch.set_num_threads(max(1, num_threads))
    torch.cuda.set_device(gpu_id)
    stats = empty_stats()
    convert_one_shard(input_path, output_path, opts, stats, torch.device(f"cuda:{gpu_id}"))
    return stats


def main() -> None:
    p = ArgumentParser(description="Convert Hy4-flash BF16 to W4A8 INT4 / INT8 / BF16 mixed precision.",
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", default="/models/Hy4-flash-testing")
    p.add_argument("--output-dir", default="/models/Hy4-flash-W4A8-INT4-INT8")
    p.add_argument("--dfly", choices=["int8", "int4", "bf16"], default="int8",
                   help="dfly 草稿层 self_attn.{q_b_proj,o_proj} 与 dense MLP 的精度")
    p.add_argument("--group-size", type=int, default=-1, help="INT4 分组大小,-1 为 per-channel")
    p.add_argument("--int8-attn", action="store_true", help="主干 layer 1..L-1 的 q_b_proj/o_proj 改 INT8")
    p.add_argument("--int8-linear-gate", action="store_true",
                   help="主干与 dfly 的 self_attn.linear_gate 一起改 INT8(HPC 融合内核要 BF16 gate)")
    p.add_argument("--int8-dense-mlp", action="store_true",
                   help="主干 layer 0 的 dense MLP {gate,up,down}_proj 改 INT8 per-channel")
    p.add_argument("--int8-shared-experts", action="store_true",
                   help="主干 layer 1..L-1 的 shared_experts {gate,up,down}_proj 改 INT8 per-channel")
    p.add_argument("--start-file", type=int, default=0)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gpu-ids", type=str, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--num-threads", type=int, default=8)
    args = p.parse_args()
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must differ")
    if args.group_size != -1 and args.group_size <= 0:
        raise ValueError("--group-size must be -1 or positive")

    with open(os.path.join(args.input_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    opts = {"dfly": args.dfly, "group_size": args.group_size,
            "int8_attn": args.int8_attn, "int8_linear_gate": args.int8_linear_gate,
            "int8_dense_mlp": args.int8_dense_mlp, "int8_shared_experts": args.int8_shared_experts}
    print("WARNING: vLLM hy_v4 dfly_config 的校验只放行 FP8,INT 模型需补丁后才能加载;"
          "W4A8 INT4 MoE 的 GPU 后端也需确认。")
    self_check_numerics()
    self_check_config(opts, cfg)
    # humming 版用字面量 Linear 目标,vLLM 的 compressed_tensors_get_config 只读带 Linear 的那个 group,
    # group_int8 的 re: 目标它完全看不到。主干层总是会被构建,所以 group_int8 里一旦有主干目标,
    # humming 版必然在加载时报 RowvLLMParameter ... missing 'input_dim';只含 dfly 目标时,
    # 不开 DFly 则 dfly 层不被构建,humming 版仍可用。
    int8_targets = build_quantization_config(opts, cfg)["config_groups"].get("group_int8", {}).get("targets", [])
    trunk_probe = ("model.layers.1.self_attn.q_b_proj", "model.layers.1.self_attn.o_proj",
                   "model.layers.1.self_attn.linear_gate", "model.layers.0.mlp.gate_up_proj",
                   "model.layers.1.mlp.shared_experts.gate_up_proj")
    trunk_int8 = [t for t in int8_targets
                  if any(re.match(t[3:], n) for n in trunk_probe if t.startswith("re:"))]
    if trunk_int8:
        print("WARNING: config.humming.json 不可用于本次选项 —— group_int8 含主干层目标"
              f"({len(trunk_int8)} 条,来自 --int8-* 开关)。humming 只识别字面量 Linear 目标,"
              "这些 INT8 主干层会被按 INT4 schema 建参数,加载时报 RowvLLMParameter missing 'input_dim'。"
              "humming 测试请去掉 --int8-* 开关,或改用 config.json(re: 正则版)。")
    elif int8_targets:
        print("NOTE: group_int8 只含 dfly 层目标。不开 DFly(启动不带 --speculative-config)时 "
              "config.humming.json 可用;开 DFly 时这些 INT8 层会被 Linear 按 INT4 schema 建参数而失败。")
    else:
        self_check_config(opts, cfg, humming=True)

    files = sorted(glob(os.path.join(args.input_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {args.input_dir}")
    if not 0 <= args.start_file < len(files):
        raise ValueError(f"--start-file must be in [0, {len(files) - 1}]")
    files = files[args.start_file:]
    if args.limit_files is not None:
        files = files[: args.limit_files]
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(args.num_threads)
    devices, n_workers = resolve_devices(args.device, args.gpu_ids, args.workers, len(files))
    stats = empty_stats()
    if n_workers <= 1 or len(devices) == 1:
        for path in tqdm(files, desc=f"Converting on {devices[0]}"):
            convert_one_shard(path, os.path.join(args.output_dir, os.path.basename(path)), opts, stats, devices[0])
    else:
        import torch.multiprocessing as mp

        tasks = [(f, os.path.join(args.output_dir, os.path.basename(f)), opts,
                  devices[i % len(devices)].index, args.num_threads) for i, f in enumerate(files)]
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            for per in tqdm(pool.imap_unordered(worker_convert_shard, tasks), total=len(tasks),
                            desc=f"Converting on {len(devices)} GPU(s) ({n_workers} workers)"):
                for k, v in per.items():
                    stats[k] += v

    for fn in sorted(os.listdir(args.input_dir)):
        src = os.path.join(args.input_dir, fn)
        if (os.path.isfile(src) and not fn.endswith((".safetensors", ".bin"))
                and fn not in ("config.json", "config.humming.json", "model.safetensors.index.json")):
            shutil.copyfile(src, os.path.join(args.output_dir, fn))
    with open(os.path.join(args.input_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        model_index = json.load(f)
    input_map = dict(model_index["weight_map"])
    rebuild_index(args.output_dir, model_index)
    verify_conversion(input_map, model_index["weight_map"], {os.path.basename(x) for x in files}, opts)

    # 同时写两份 config:config.json 用 re: 正则目标(默认),config.humming.json 用字面量 Linear(humming 后端)。
    # 两份只有 group_int4.targets 不同,其余(weights/input_activations/ignore/INT8 组)完全一致。
    for fname, use_humming in (("config.json", False), ("config.humming.json", True)):
        out_cfg = json.loads(json.dumps(cfg))
        (out_cfg["text_config"] if "text_config" in out_cfg else out_cfg)["quantization_config"] = \
            build_quantization_config(opts, cfg, humming=use_humming)
        with open(os.path.join(args.output_dir, fname), "w", encoding="utf-8") as f:
            json.dump(out_cfg, f, indent=2, ensure_ascii=False, sort_keys=True)
    summary = {"input_dir": args.input_dir, "output_dir": args.output_dir, "options": opts, "stats": stats}
    with open(os.path.join(args.output_dir, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
