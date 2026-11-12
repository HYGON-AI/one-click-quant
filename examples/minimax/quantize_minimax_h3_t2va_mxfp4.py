# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-H3 DiT (BF16, diffusers 布局) -> 混合精度 MXFP4 / FP8 / BF16 转换脚本
(compressed-tensors config_groups, 面向 vLLM-Omni)

只改写 DiT(默认 `transformer/`)的分片, 其余目录/文件默认以软链接放入输出目录。

层选择 (依据: 社区 NVFP4 版的 layer config + vllm-omni 的 DiT 定义; 质量尚未评测)
  始终保持原 dtype (BF16 / F32):
    transformer_blocks.{0,1,47,48,49}.*   首 2 块、末 3 块 (可用 --first-bf16 / --last-bf16 调整)
    attn.to_out.0, attn.norm_q/k, norm*, to_gate_compress
    proj_in/out, audio_proj_in/out, context_embedder, time_embedder.*, norm_out.*, token_refiner.*
  默认量化 (N in [first-bf16, num_layers-last-bf16)):
    transformer_blocks.N.ff.net.0.proj  (运行时 blocks.N.mlp.fc1)  -> MXFP4
    transformer_blocks.N.ff.net.2       (运行时 blocks.N.mlp.fc2)  -> MXFP4
  命令行开关 (默认关闭), 开启后对同一批 block 量化为 FP8 逐通道:
    --fp8-qkv    attn.to_q / to_k / to_v   (三者必须同进同出, 运行时融合为 attn.qkv_proj)
    --fp8-adaln  adaln_proj.linear         ([96768, 2688], 约占 39% 参数)

MXFP4 公式 (对齐 vLLM mxfp4_quantize, 与 compressed_tensors.generate_mx_scales 一致):
    amax      = max|w| (沿 K 每 32 个元素)
    amax'     = (bits(amax) + 0x200000) & 0x7F800000        # 尾数 >= 1.75 进位
    scale_exp = clamp(floor(log2(amax')) - 2, -127, 127)    # 2 = floor(log2(6.0))
    E8M0 存储 = scale_exp + 127 (uint8)
    code      = E2M1 round-to-nearest-even(w / 2^scale_exp), 饱和 +-6
    packed    = code_even | (code_odd << 4)

FP8 逐通道: scale = amax_row / 448, weight = (w / scale).to(float8_e4m3fn), weight_scale fp32 (N, 1)。
    激活在 config 中声明为动态 per-token (无 input_scale)。

落盘键名:
    MXFP4: <p>.weight_packed uint8 (N, K/2) ; <p>.weight_scale uint8 (N, K/32)
    FP8  : <p>.weight float8_e4m3fn (N, K)  ; <p>.weight_scale float32 (N, 1)
    其余 : 原样

注意: 行顺序保持不变。fc1 在 diffusers 中是 value-first 的融合 gate/up, 运行时 loader 按行
chunk/swap; MXFP4 的 group 沿 K 方向, 与行拆分互不影响。

!! 待验证 (见脚本末尾 NEXT STEPS):
   vllm-omni 的 MiniMax-H3 load_weights 目前对 diffusers 权重只接受 weight/bias/weight_scale,
   遇到 weight_packed 会抛 "unsupported Diffusers H3 weight"; 需要在引擎侧放行。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass, replace
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
MXFP4_GROUP = 32
MX_ELEM_OFFSET = 2  # floor(log2(6.0))
FP8_MAX = 448.0

# 两种 DiT 命名:
#   native    (FL2VA/transformer, MiniMaxH3DiTModel, 与运行时模块同名):
#             blocks.N.mlp.fc1 / mlp.fc2 / attn.qkv_proj(已融合的 grouped QKV) / adaln_proj.linear
#   diffusers (顶层 transformer/, MiniMaxH3Transformer3DModel):
#             transformer_blocks.N.ff.net.0.proj / ff.net.2 / attn.to_q|k|v / adaln_proj.linear
# 前缀锚定在行首, token_refiner.blocks.* 不会被匹配。
BLOCK_RE = re.compile(r"^(?:transformer_blocks|blocks)\.(\d+)\.(.+)\.weight$")
MXFP4_SUFFIXES = ("ff.net.0.proj", "ff.net.2", "mlp.fc1", "mlp.fc2")
FP8_QKV_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.qkv_proj")
FP8_ADALN_SUFFIXES = ("adaln_proj.linear",)

_STATS_KEYS = ("mxfp4", "fp8_qkv", "fp8_adaln", "kept")

# 激活声明: 只影响 config, 权重字节完全相同。
#   w4a16: input_activations=null  -> vLLM CompressedTensorsW4A4Mxfp4; 无 FP4 硬件(如 Hopper)
#          时走权重量化内核(Marlin/Humming), 激活保持 BF16。
#   w4a8 : 动态 FP8 per-token  -> vLLM CompressedTensorsW4A8Mxfp4; 若平台没有支持 FP8 激活的
#          MXFP4 内核, vLLM 只打 warning 并回退到 MXFP4 激活 (compressed_tensors_w4a8_mxfp4.py)。
#   w4a4 : 动态 MXFP4 group-32 激活 (同 compressed-tensors 预设 MXFP4) -> vLLM 同样分发到
#          CompressedTensorsW4A4Mxfp4 (分发只看 weight_quant); SM100+ 有 FlashInfer 时才是真 W4A4,
#          其余平台(如 Hopper)回退到 Marlin W4A16。
MODES = ("w4a16", "w4a8", "w4a4")


@dataclass(frozen=True)
class ConvOptions:
    num_layers: int = 50
    first_bf16: int = 2
    last_bf16: int = 3
    fp8_qkv: bool = False
    fp8_adaln: bool = False
    mode: str = "w4a16"
    # 原生命名 attn.qkv_proj 的 grouped QKV 布局 (来自 DiT config.json, 转换每个目录前读取)
    num_heads: int = 56
    head_dim: int = 128

    def block_range(self) -> range:
        return range(self.first_bf16, self.num_layers - self.last_bf16)


# --------------------------------------------------------------------------- #
# MXFP4 量化
# --------------------------------------------------------------------------- #

def compute_e8m0_exponent(amax: torch.Tensor) -> torch.Tensor:
    """amax fp32 -> 无偏 scale 指数 (int32), 钳位 [-127, 127]。amax==0 -> -127。"""
    bits = amax.float().contiguous().view(torch.int32)
    rounded = (bits + 0x200000) & 0x7F800000
    exp = (rounded >> 23) - 127
    return (exp - MX_ELEM_OFFSET).clamp(-127, 127)


def float_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """FP32 (已除以 scale) -> E2M1 4-bit 码 (uint8, 含符号位), RNE, 饱和到 6。"""
    a = x.abs().clamp(max=6.0)
    code = (
        (a > 0.25).to(torch.uint8)
        + (a >= 0.75).to(torch.uint8)
        + (a > 1.25).to(torch.uint8)
        + (a >= 1.75).to(torch.uint8)
        + (a > 2.5).to(torch.uint8)
        + (a >= 3.5).to(torch.uint8)
        + (a > 5.0).to(torch.uint8)
    )
    return code | ((x < 0).to(torch.uint8) << 3)


def quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) -> (weight_packed uint8 (N, K/2), weight_scale uint8 (N, K/32))。"""
    assert w.ndim == 2, f"expect 2D, got {tuple(w.shape)}"
    N, K = w.shape
    assert K % MXFP4_GROUP == 0, f"K={K} not divisible by {MXFP4_GROUP}"
    wg = w.float().view(N, K // MXFP4_GROUP, MXFP4_GROUP)
    exp = compute_e8m0_exponent(wg.abs().amax(dim=-1))
    inv_scale = torch.pow(2.0, (-exp).float()).unsqueeze(-1)
    codes = float_to_e2m1_codes(wg * inv_scale).view(N, K)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, (exp + 127).to(torch.uint8)


def dequant_mxfp4(packed: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    low, high = packed & 0x0F, (packed >> 4) & 0x0F
    table = FP4_TABLE.to(packed.device)
    vals = torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(1)
    scale = torch.pow(2.0, e8m0.float() - 127.0).repeat_interleave(MXFP4_GROUP, dim=1)
    return vals * scale


# --------------------------------------------------------------------------- #
# FP8 逐通道量化
# --------------------------------------------------------------------------- #

def reorder_grouped_qkv_rows(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """grouped [g0:(q,k,v), g1:(q,k,v), ...] -> [all q; all k; all v] (沿第 0 维)。

    与 vllm-omni minimax_h3_transformer._reorder_grouped_qkv_to_qkv(heads_per_group=1) 完全一致。
    引擎只对 qkv_proj.weight 做这个重排, weight_scale 原样加载, 所以逐通道 FP8 的 scale
    必须在这里按同一置换预先排好, 否则 q/k/v 各行会拿到别的行的 scale (画面异常)。
    """
    per_group = 3 * head_dim
    if x.shape[0] != num_heads * per_group:
        raise ValueError(f"qkv rows {x.shape[0]} != heads({num_heads}) * 3 * head_dim({head_dim})")
    g = x.reshape(num_heads, per_group, *x.shape[1:])
    q, k, v = torch.split(g, [head_dim, head_dim, head_dim], dim=1)
    rest = x.shape[1:]
    return torch.cat([q.reshape(num_heads * head_dim, *rest),
                      k.reshape(num_heads * head_dim, *rest),
                      v.reshape(num_heads * head_dim, *rest)], dim=0)


def quantize_fp8_channelwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) -> (float8_e4m3fn (N, K), fp32 scale (N, 1))。"""
    assert w.ndim == 2, f"expect 2D, got {tuple(w.shape)}"
    wf = w.float()
    amax = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


# --------------------------------------------------------------------------- #
# 键名分类
# --------------------------------------------------------------------------- #

def classify(name: str, opts: ConvOptions) -> str | None:
    """返回 'mxfp4' | 'fp8_qkv' | 'fp8_adaln' | None (透传)。"""
    m = BLOCK_RE.match(name)
    if m is None or int(m.group(1)) not in opts.block_range():
        return None
    suffix = m.group(2)
    if suffix in MXFP4_SUFFIXES:
        return "mxfp4"
    if opts.fp8_qkv and suffix in FP8_QKV_SUFFIXES:
        return "fp8_qkv"
    if opts.fp8_adaln and suffix in FP8_ADALN_SUFFIXES:
        return "fp8_adaln"
    return None


# --------------------------------------------------------------------------- #
# 分片转换
# --------------------------------------------------------------------------- #

def convert_one_file(input_path, output_path, opts, stats, device="cpu") -> None:
    device = torch.device(device)
    out: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            kind = classify(name, opts)
            if kind is None:
                out[name] = tensor
                stats["kept"] += 1
                continue
            prefix = name[: -len(".weight")]
            w = tensor.to(device)
            if kind == "mxfp4":
                packed, e8m0 = quantize_mxfp4(w)
                out[f"{prefix}.weight_packed"] = packed.cpu()
                out[f"{prefix}.weight_scale"] = e8m0.cpu()
            else:
                q, scale = quantize_fp8_channelwise(w)
                if kind == "fp8_qkv" and name.endswith(".attn.qkv_proj.weight"):
                    # 权重保持 checkpoint 的 grouped 行序 (引擎会自行重排), scale 预先重排
                    scale = reorder_grouped_qkv_rows(scale, opts.num_heads, opts.head_dim)
                out[name] = q.cpu()
                out[f"{prefix}.weight_scale"] = scale.cpu()
            stats[kind] += 1
    save_file(out, output_path, metadata={"format": "pt"})


def _empty_stats() -> dict[str, int]:
    return {k: 0 for k in _STATS_KEYS}


def _worker(args):
    inp, outp, opts, gpu_id = args
    torch.cuda.set_device(gpu_id)
    stats = _empty_stats()
    convert_one_file(inp, outp, opts, stats, torch.device(f"cuda:{gpu_id}"))
    return stats


def convert_transformer_dir(src, dst, opts, limit_files, device_kind, gpu_ids) -> dict[str, int]:
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob(os.path.join(src, "*.safetensors")))
    if limit_files is not None:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {src}")
    stats = _empty_stats()

    cuda = torch.cuda.is_available()
    use_gpu = device_kind == "cuda" or (device_kind == "auto" and cuda)
    if use_gpu and not cuda:
        raise RuntimeError("--device cuda specified but CUDA is unavailable")
    ids = ([int(x) for x in gpu_ids.split(",") if x.strip()] if gpu_ids
           else list(range(torch.cuda.device_count()))) if use_gpu else []

    if len(ids) > 1:
        import torch.multiprocessing as mp
        tasks = [(p, os.path.join(dst, os.path.basename(p)), opts, ids[i % len(ids)])
                 for i, p in enumerate(files)]
        with mp.get_context("spawn").Pool(processes=min(len(ids), len(files))) as pool:
            for per in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc=src):
                for k, v in per.items():
                    stats[k] += v
    else:
        device = torch.device(f"cuda:{ids[0]}") if ids else torch.device("cpu")
        for p in tqdm(files, desc=f"{src} on {device}"):
            convert_one_file(p, os.path.join(dst, os.path.basename(p)), opts, stats, device)
    return stats


# --------------------------------------------------------------------------- #
# config / index
# --------------------------------------------------------------------------- #

def _block_alt(r: range) -> str:
    return "(?:" + "|".join(str(i) for i in r) + ")"


def mxfp4_input_activations(mode: str) -> dict | None:
    """MXFP4 层的激活声明 (仅 config, 不影响权重字节)。"""
    if mode == "w4a8":
        return {"dynamic": True, "num_bits": 8, "type": "float", "strategy": "token",
                "symmetric": True}
    if mode == "w4a4":
        # 对齐 compressed_tensors.quantization.quant_scheme.MXFP4 的 input_activations
        return {"dynamic": True, "num_bits": 4, "type": "float", "strategy": "group",
                "group_size": 32, "symmetric": True, "scale_dtype": "torch.uint8",
                "zp_dtype": None}
    return None


def build_compression_config(opts: ConvOptions) -> dict:
    """compressed-tensors 配置。targets 匹配**运行时模块名** (blocks.N.mlp.fc1 等),
    同时列出 diffusers 名以便容错; 未被任何 target 命中的层按未量化处理。"""
    blk = _block_alt(opts.block_range())
    # 排除 token_refiner.blocks.* (其层号 0/1 本就不在范围内, 这里仍显式排除)
    head = r"re:^(?!.*token_refiner).*(?:^|\.)(?:blocks|transformer_blocks)\." + blk + r"\."

    fp8_weight_args = {
        "dynamic": False, "num_bits": 8, "type": "float", "strategy": "channel",
        "symmetric": True,
    }
    groups: dict[str, dict] = {
        "group_mxfp4": {
            "targets": [head + r"(?:mlp\.fc1|mlp\.fc2|ff\.net\.0\.proj|ff\.net\.2)$"],
            "weights": {
                "dynamic": False, "num_bits": 4, "type": "float", "strategy": "group",
                "group_size": 32, "symmetric": True, "scale_dtype": "torch.uint8",
                "zp_dtype": None,
            },
            "input_activations": mxfp4_input_activations(opts.mode),
            "format": "mxfp4-pack-quantized",
        }
    }
    fp8_targets = []
    if opts.fp8_qkv:
        fp8_targets.append(head + r"(?:attn\.qkv_proj|attn\.to_q|attn\.to_k|attn\.to_v)$")
    if opts.fp8_adaln:
        fp8_targets.append(head + r"adaln_proj\.linear$")
    if fp8_targets:
        groups["group_fp8"] = {
            "targets": fp8_targets,
            "weights": dict(fp8_weight_args),
            "input_activations": {
                "dynamic": True, "num_bits": 8, "type": "float", "strategy": "token",
                "symmetric": True,
            },
            "format": "float-quantized",
        }
    return {
        "config_groups": groups,
        "format": "mixed-precision",
        "ignore": [],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def rebuild_index(dst: str, limit_files) -> None:
    src_idx = os.path.join(dst, "diffusion_pytorch_model.safetensors.index.json")
    index = {}
    if os.path.exists(src_idx):
        with open(src_idx, encoding="utf-8") as f:
            index = json.load(f)
    files = sorted(glob(os.path.join(dst, "*.safetensors")))
    wm, total = {}, 0
    for p in files:
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                wm[k] = os.path.basename(p)
                t = f.get_tensor(k)
                total += t.numel() * t.element_size()
    index["weight_map"] = wm
    index["metadata"] = {**index.get("metadata", {}), "total_size": total}
    with open(src_idx, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_config(src: str, dst: str, opts: ConvOptions) -> None:
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    n = int(cfg.get("num_layers", opts.num_layers))
    if n != opts.num_layers:
        raise ValueError(f"config num_layers={n} != --num-layers {opts.num_layers}")
    cfg["quantization_config"] = build_compression_config(opts)
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, sort_keys=True)


def link_others(src: str, dst: str, quantized: set[str], mode: str, rel: str = "") -> None:
    """把未量化的条目放进输出目录; 含已量化子目录的父目录逐级展开。"""
    if mode == "skip":
        return
    for entry in sorted(os.listdir(src)):
        r = f"{rel}/{entry}" if rel else entry
        s, d = os.path.join(src, entry), os.path.join(dst, entry)
        if r in quantized:
            continue
        if any(q.startswith(r + "/") for q in quantized):
            os.makedirs(d, exist_ok=True)
            link_others(s, d, quantized, mode, r)
            continue
        if os.path.lexists(d):
            continue
        if mode == "symlink":
            os.symlink(os.path.abspath(s), d)
        elif os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copyfile(s, d)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #

def _self_check() -> None:
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.equal(float_to_e2m1_codes(vals), torch.arange(8, dtype=torch.uint8))
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 100.0])
    assert float_to_e2m1_codes(ties).tolist() == [0, 2, 2, 4, 4, 6, 6, 7]
    got = compute_e8m0_exponent(torch.tensor([6.0, 7.0, 1.74, 1.75, 0.0, 1e-45])).tolist()
    assert got == [0, 1, -2, -1, -127, -127], got

    torch.manual_seed(0)
    w = torch.randn(16, 128)
    packed, e8m0 = quantize_mxfp4(w)
    assert packed.dtype == torch.uint8 and packed.shape == (16, 64)
    assert e8m0.dtype == torch.uint8 and e8m0.shape == (16, 4)
    rec = dequant_mxfp4(packed, e8m0)
    assert ((rec - w) ** 2).mean() / (w ** 2).mean() < 0.05
    assert torch.equal(dequant_mxfp4(*quantize_mxfp4(rec)), rec), "MXFP4 grid must be a fixed point"

    q, s = quantize_fp8_channelwise(w)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (16, 1) and s.dtype == torch.float32
    rel = ((q.float() * s - w) ** 2).mean() / (w ** 2).mean()
    assert rel < 1e-2, rel

    # --- QKV scale 置换: 与引擎 _reorder_grouped_qkv_to_qkv(heads_per_group=1) 的逐字副本对照 ---
    def engine_reorder(weight, num_query_groups, heads_per_group, head_dim):
        per_group = (heads_per_group + 2) * head_dim
        rest = weight.shape[1:]
        grouped = weight.reshape(num_query_groups, per_group, *rest)
        q, k, v = torch.split(grouped, [heads_per_group * head_dim, head_dim, head_dim], dim=1)
        return torch.cat([q.reshape(num_query_groups * heads_per_group * head_dim, *rest),
                          k.reshape(num_query_groups * head_dim, *rest),
                          v.reshape(num_query_groups * head_dim, *rest)], dim=0)

    H, D, K = 4, 8, 64
    wq = torch.randn(H * 3 * D, K)
    q8, sc = quantize_fp8_channelwise(wq)
    true = engine_reorder(q8.float() * sc, H, 1, D)                      # 理想: 每行配自己的 scale
    w_eng = engine_reorder(q8.float(), H, 1, D)                          # 引擎重排后的权重
    assert torch.equal(reorder_grouped_qkv_rows(sc, H, D), engine_reorder(sc, H, 1, D))
    good = w_eng * reorder_grouped_qkv_rows(sc, H, D)                    # 修复后: scale 预排序
    bad = w_eng * sc                                                     # 旧产物: scale 未排序
    assert torch.equal(good, true), "预排序 scale 后应与理想反量化完全一致"
    assert not torch.equal(bad, true), "对照: 不排序 scale 必然错位"

    o = ConvOptions()
    assert list(o.block_range())[0] == 2 and list(o.block_range())[-1] == 46
    assert classify("transformer_blocks.2.ff.net.0.proj.weight", o) == "mxfp4"
    assert classify("transformer_blocks.46.ff.net.2.weight", o) == "mxfp4"
    for n in ("transformer_blocks.1.ff.net.2.weight", "transformer_blocks.47.ff.net.2.weight",
              "transformer_blocks.5.attn.to_out.0.weight", "transformer_blocks.5.attn.to_q.weight",
              "transformer_blocks.5.adaln_proj.linear.weight", "token_refiner.refiner_blocks.0.ff.net.2.weight",
              "transformer_blocks.5.ff.net.2.bias"):
        assert classify(n, o) is None, n
    o2 = ConvOptions(fp8_qkv=True, fp8_adaln=True)
    assert classify("transformer_blocks.5.attn.to_k.weight", o2) == "fp8_qkv"
    assert classify("transformer_blocks.5.adaln_proj.linear.weight", o2) == "fp8_adaln"

    # config 正则在运行时名与 diffusers 名上的命中, 以及 token_refiner / 边界块的排除
    def hit(cfg, name):
        return any(re.match(t[3:], name) for g in cfg["config_groups"].values() for t in g["targets"])

    c = build_compression_config(o2)
    for n in ("blocks.2.mlp.fc1", "transformer.blocks.46.mlp.fc2", "transformer_blocks.10.ff.net.0.proj",
              "blocks.20.attn.qkv_proj", "blocks.20.adaln_proj.linear"):
        assert hit(c, n), n
    for n in ("blocks.1.mlp.fc1", "blocks.47.mlp.fc1", "blocks.20.attn.out_proj",
              "token_refiner.blocks.0.mlp.fc1", "token_refiner.blocks.1.attn.qkv_proj",
              "final_layer.adaln_proj.linear", "time_embedder.proj_in"):
        assert not hit(c, n), n
    c0 = build_compression_config(o)
    assert "group_fp8" not in c0["config_groups"] and not hit(c0, "blocks.20.attn.qkv_proj")

    # 原生命名 (FL2VA/transformer)
    assert classify("blocks.2.mlp.fc1.weight", o) == "mxfp4"
    assert classify("blocks.46.mlp.fc2.weight", o) == "mxfp4"
    assert classify("blocks.47.mlp.fc1.weight", o) is None and classify("blocks.1.mlp.fc1.weight", o) is None
    assert classify("blocks.5.attn.qkv_proj.weight", o) is None and classify("blocks.5.attn.out_proj.weight", o) is None
    assert classify("token_refiner.blocks.0.mlp.fc1.weight", o) is None
    assert classify("blocks.5.attn.qkv_proj.weight", o2) == "fp8_qkv"
    assert classify("blocks.5.adaln_proj.linear.weight", o2) == "fp8_adaln"
    assert classify("blocks.5.adaln_proj.linear.bias", o2) is None

    # w4a16 / w4a8 仅差 input_activations
    a16 = build_compression_config(ConvOptions(mode="w4a16"))["config_groups"]["group_mxfp4"]
    a8 = build_compression_config(ConvOptions(mode="w4a8"))["config_groups"]["group_mxfp4"]
    assert a16["input_activations"] is None
    assert a8["input_activations"]["num_bits"] == 8 and a8["input_activations"]["dynamic"] is True
    assert a16["weights"] == a8["weights"] and a16["targets"] == a8["targets"]
    a4 = build_compression_config(ConvOptions(mode="w4a4"))["config_groups"]["group_mxfp4"]
    ia = a4["input_activations"]
    assert (ia["num_bits"], ia["type"], ia["strategy"], ia["group_size"], ia["dynamic"]) == (4, "float", "group", 32, True)
    assert a4["weights"] == a16["weights"] and a4["targets"] == a16["targets"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    p = ArgumentParser(
        description="MiniMax-H3 DiT BF16 -> mixed MXFP4 / FP8 / BF16 (compressed-tensors).",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, help="MiniMax-H3 根目录")
    p.add_argument("--output-dir", required=True, help="输出目录, 不能与 input-dir 相同")
    p.add_argument("--transformer-dirs", nargs="+", default=["FL2VA/transformer"],
                   help="相对 input-dir 的 DiT 目录。T2VA 与 FL2VA 共用 FL2VA/transformer (原生命名); "
                        "顶层 transformer/ 为 diffusers 命名 (modular 布局), 两种命名都支持")
    p.add_argument("--mode", choices=list(MODES), default="w4a16",
                   help="仅控制 MXFP4 层 (mlp.fc1/fc2) 的激活声明, 权重字节相同, 只有 config 不同: "
                        "w4a16=激活不量化; w4a8=动态 FP8 per-token 激活; w4a4=动态 MXFP4 group-32 激活。"
                        "不影响 FP8 层 (--fp8-qkv/--fp8-adaln 开启的层): 它们始终是 W8A8 "
                        "(权重 FP8 逐通道 + 动态 FP8 per-token 激活), 与本选项无关")
    p.add_argument("--num-layers", type=int, default=50, help="DiT block 总数, 需与 config.json 的 num_layers 一致")
    p.add_argument("--first-bf16", type=int, default=2, help="开头保持 BF16 的 block 数")
    p.add_argument("--last-bf16", type=int, default=3, help="末尾保持 BF16 的 block 数")
    p.add_argument("--fp8-qkv", action="store_true", default=False,
                   help="attn.to_q/k/v 量化为 FP8 逐通道 (默认关闭); 原生命名的 qkv_proj scale 会按引擎的行置换预排序")
    p.add_argument("--fp8-adaln", action="store_true", default=False,
                   help="adaln_proj.linear 量化为 FP8 逐通道 (默认关闭)")
    p.add_argument("--config-dirs", nargs="*", default=["transformer"],
                   help="额外写入 quantization_config 的目录 (相对 input-dir, 只改 config.json, 其余软链接)。"
                        "vllm-omni 从顶层 transformer/config.json 读取量化配置, 所以默认包含它; 传空则不写")
    p.add_argument("--others", choices=["symlink", "copy", "skip"], default="symlink",
                   help="非 DiT 的其余条目如何放入输出目录")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="计算设备: auto=有 GPU 用 GPU 否则 CPU; cpu=强制 CPU; cuda=必须用 GPU, 不可用则报错")
    p.add_argument("--gpu-ids", default=None,
                   help="逗号分隔的 CUDA 编号, 如 '0,1,3'; 默认用全部可见 GPU; 多于 1 张时按分片多进程并行")
    p.add_argument("--limit-files", type=int, default=None, help="每个 DiT 目录只转前 N 个分片 (冒烟)")
    p.add_argument("--num-threads", type=int, default=8, help="torch CPU 线程数")
    args = p.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must be different")
    torch.set_num_threads(args.num_threads)
    _self_check()

    opts = ConvOptions(args.num_layers, args.first_bf16, args.last_bf16, args.fp8_qkv, args.fp8_adaln, args.mode)
    br = opts.block_range()
    print(f"mode: {opts.mode}")
    print(f"MXFP4: blocks {br.start}..{br.stop - 1} ff.net.0.proj / ff.net.2 (expect {2 * len(br)} layers per DiT)")
    print(f"FP8 qkv  : {'ON' if opts.fp8_qkv else 'off'}   FP8 adaln: {'ON' if opts.fp8_adaln else 'off'}")
    if opts.fp8_qkv:
        print("note: --fp8-qkv 时, 原生命名 attn.qkv_proj 的逐通道 scale 已按引擎的 grouped->qkv 置换预排序"
              " (权重保持原行序); diffusers 命名 (to_q/k/v 分开) 无需置换。")

    os.makedirs(args.output_dir, exist_ok=True)
    total = _empty_stats()
    for sub in args.transformer_dirs:
        src, dst = os.path.join(args.input_dir, sub), os.path.join(args.output_dir, sub)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(os.path.join(src, "config.json"), encoding="utf-8") as fcfg:
            tcfg = json.load(fcfg)
        opts = replace(opts, num_heads=int(tcfg.get("num_attention_heads", opts.num_heads)),
                       head_dim=int(tcfg.get("attention_head_dim", opts.head_dim)))
        stats = convert_transformer_dir(src, dst, opts, args.limit_files, args.device, args.gpu_ids)
        write_config(src, dst, opts)
        rebuild_index(dst, args.limit_files)
        print(sub, json.dumps(stats, sort_keys=True))
        for k, v in stats.items():
            total[k] += v
        if args.limit_files is None:
            # qkv: native 每块 1 个融合层, diffusers 每块 3 个 (q/k/v)
            qkv_ok = (stats["fp8_qkv"] in (len(br), 3 * len(br))) if opts.fp8_qkv else stats["fp8_qkv"] == 0
            exp = {"mxfp4": 2 * len(br), "fp8_adaln": len(br) if opts.fp8_adaln else 0}
            bad = {k: (stats[k], e) for k, e in exp.items() if stats[k] != e}
            if bad or not qkv_ok:
                raise RuntimeError(
                    f"{sub}: 层数与预期不符 (实际, 预期): {bad}; fp8_qkv={stats['fp8_qkv']}, "
                    f"预期 {len(br)}(native) 或 {3 * len(br)}(diffusers)")

    # vllm-omni 的 MiniMax-H3 从 **顶层 transformer/config.json** 读取量化配置
    # (minimax_h3_transformer_subfolder -> "transformer"), 而权重来自 FL2VA/transformer。
    # 因此顶层目录需要带同一份 quantization_config; 其余文件软链接到原目录, 不改动。
    extra_done: list[str] = []
    for sub in args.config_dirs:
        if sub in args.transformer_dirs:
            continue
        src, dst = os.path.join(args.input_dir, sub), os.path.join(args.output_dir, sub)
        if not os.path.isfile(os.path.join(src, "config.json")):
            print(f"[warn] {sub}/config.json 不存在, 跳过写入量化配置")
            continue
        os.makedirs(dst, exist_ok=True)
        for entry in sorted(os.listdir(src)):
            if entry == "config.json":
                continue
            d = os.path.join(dst, entry)
            if not os.path.lexists(d):
                os.symlink(os.path.abspath(os.path.join(src, entry)), d)
        write_config(src, dst, opts)
        extra_done.append(sub)
        print(f"{sub}: 已写入 quantization_config (仅 config.json; 其余软链接, 权重不变)")

    link_others(args.input_dir, args.output_dir, set(args.transformer_dirs) | set(extra_done), args.others)
    with open(os.path.join(args.output_dir, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"options": opts.__dict__, "transformer_dirs": args.transformer_dirs, "stats": total},
                  f, indent=2, sort_keys=True)
    print(json.dumps(total, indent=2, sort_keys=True))
    print(f"Done -> {args.output_dir}")


if __name__ == "__main__":
    main()
