# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8-Flash-Next NVFP4 (ModelOpt 0.46, MIXED_PRECISION) -> MXFP4 (compressed-tensors) 转换脚本

只转换主干路由专家 `model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj` (NVFP4):
  源:  <p>.weight          uint8   (N, K/2)   E2M1 打包, 低 4 位 = 偶数下标元素
       <p>.weight_scale    fp8 e4m3 (N, K/16)
       <p>.weight_scale_2  fp32 标量
       <p>.input_scale     fp32 标量
  目标: <p>.weight_packed  uint8 (N, K/2)
       <p>.weight_scale    uint8 (N, K/32)    E8M0 = 无偏指数 + 127
  weight_scale_2 / input_scale 丢弃。

其余张量一律原样透传 (attention / linear_attn / router / shared_expert / hyper-connection /
lm_head / embed / visual 为 BF16; PLE n-gram embedding 为 FP8; MTP 专家为 FP8 块缩放)。

转换策略: NVFP4 反量化到 FP32 -> 按 32 一块重新量化为 MXFP4。这是**二次量化**: NVFP4 的值
(e2m1 * E4M3 块缩放 * 全局缩放) 一般不落在 MXFP4 (e2m1 * 2^k) 的网格上, 误差叠加在 NVFP4 自身
误差之上; 没有 BF16 原始权重, 无法避免。可用 --measure-error 量化这部分额外误差。

公式 (对齐 vLLM mxfp4_quantize / compressed_tensors.generate_mx_scales, 与
quantize_minimax_h3_t2va_mxfp4.py、quantize_qwen3_5_nvfp4_to_mxfp4.py 相同):
    amax      = max|v| (块内 32 个元素)
    amax'     = (bits(amax) + 0x200000) & 0x7F800000        # 尾数 >= 1.75 进位
    scale_exp = clamp(floor(log2(amax')) - 2, -127, 127)    # 2 = floor(log2(6.0))
    code      = E2M1 round-to-nearest-even(v / 2^scale_exp), 饱和 +-6

NVFP4 反量化与 Model-Optimizer nvfp4_tensor.py:389-440 一致:
    value = e2m1(nibble) * weight_scale * weight_scale_2    # nibble: 低位 = 偶数下标

config.json:
  - 移除 ModelOpt 的 quantization_config, 写 compressed-tensors (group_mxfp4, mxfp4-pack-quantized)
  - text_config.ple_embedding_dtype = "float8_e4m3fn": 换成 compressed-tensors 后 vLLM 会把无 scheme
    的 PLE 当成未量化 (BF16), 与 47.7 GiB 的 FP8 分片不符; 该字段让 PLE 无视 quant_config 直接走 FP8
    (vllm/models/qwen4_exp/common/ngram_embedding.py: from_quant_config 的 embedding_dtype 分支;
     调用处 nvidia/ngram_embedding.py 读的是 Qwen4ExpTextConfig, 故写在 text_config)
  - 不复制 hf_quant_config.json (它描述 NVFP4, 留着会误导)

!! 待云端验证: 开启 MTP 时 compressed-tensors 下 MTP FP8 块缩放专家 (weight_scale_inv) 能否加载。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from contextlib import ExitStack
from dataclasses import dataclass
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
NVFP4_GROUP = 16
MXFP4_GROUP = 32
MX_ELEM_OFFSET = 2  # floor(log2(6.0))
E2M1_MAX = 6.0
E4M3_MAX = 448.0

EXPERT_RE = re.compile(
    r"^(model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj)"
    r"\.(weight|weight_scale|weight_scale_2|input_scale)$"
)
SCALE_LEAVES = ("weight_scale", "weight_scale_2", "input_scale")

SOURCE_ONLY_FILES = {"hf_quant_config.json", "model.safetensors.index.json", "config.json"}
EXPECTED_EXPERT_MODULES = 48 * 512 * 3

_STATS_KEYS = ("nvfp4_to_mxfp4", "kept", "dropped_scales", "err_num", "err_den")


@dataclass(frozen=True)
class ConvOptions:
    mode: str = "w4a16"
    measure_error: bool = False


# --------------------------------------------------------------------------- #
# NVFP4 反量化
# --------------------------------------------------------------------------- #

_TABLE_CACHE: dict[torch.device, torch.Tensor] = {}


def _table_on(device: torch.device) -> torch.Tensor:
    if device not in _TABLE_CACHE:
        _TABLE_CACHE[device] = FP4_TABLE.to(device)
    return _TABLE_CACHE[device]


def unpack_fp4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """uint8 (N, K/2) -> 码值 int64 (N, K); 低 4 位为偶数下标元素。"""
    assert packed.dtype == torch.uint8, f"expect uint8, got {packed.dtype}"
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).flatten(1).long()


def dequant_nvfp4(packed, weight_scale, weight_scale_2) -> torch.Tensor:
    """NVFP4 -> FP32 (N, K): e2m1 * weight_scale(E4M3, block16) * weight_scale_2。"""
    values = _table_on(packed.device)[unpack_fp4_nibbles(packed)]
    N, K = values.shape
    assert weight_scale.shape == (N, K // NVFP4_GROUP), (
        f"weight_scale shape {tuple(weight_scale.shape)} != {(N, K // NVFP4_GROUP)}"
    )
    scale = weight_scale.float().repeat_interleave(NVFP4_GROUP, dim=1)
    return values * scale * weight_scale_2.float().reshape(())


# --------------------------------------------------------------------------- #
# MXFP4 量化 (对齐 vLLM)
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
    """FP32 (N, K) -> (weight_packed uint8 (N, K/2), weight_scale uint8 (N, K/32))。"""
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
    values = _table_on(packed.device)[unpack_fp4_nibbles(packed)]
    scale = torch.pow(2.0, e8m0.float() - 127.0).repeat_interleave(MXFP4_GROUP, dim=1)
    return values * scale


# --------------------------------------------------------------------------- #
# 全局索引: 7 个专家模块的张量跨分片, 必须按索引跨分片取 scale
# --------------------------------------------------------------------------- #

def load_plan(input_dir: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """返回 (weight_map, nvfp4_modules)。nvfp4_modules: 前缀 -> {leaf: 分片名}。"""
    with open(os.path.join(input_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        wm = json.load(f)["weight_map"]
    modules: dict[str, dict[str, str]] = {}
    for name, shard in wm.items():
        m = EXPERT_RE.match(name)
        if m:
            modules.setdefault(m.group(1), {})[m.group(2)] = shard
    bad = {p: sorted(set(("weight", *SCALE_LEAVES)) - set(v)) for p, v in modules.items()
           if set(v) != {"weight", *SCALE_LEAVES}}
    if bad:
        first = next(iter(bad.items()))
        raise ValueError(f"{len(bad)} 个专家模块缺少张量, 例: {first}")
    return wm, modules


class ShardReader:
    """按需打开分片并缓存句柄; 用于跨分片读取 scale。"""

    def __init__(self, input_dir: str, stack: ExitStack):
        self.input_dir, self.stack, self.handles = input_dir, stack, {}

    def get(self, shard: str, name: str) -> torch.Tensor:
        if shard not in self.handles:
            self.handles[shard] = self.stack.enter_context(
                safe_open(os.path.join(self.input_dir, shard), framework="pt", device="cpu")
            )
        return self.handles[shard].get_tensor(name)


# --------------------------------------------------------------------------- #
# 分片转换
# --------------------------------------------------------------------------- #

def convert_one_file(input_dir, shard, output_dir, opts, stats, device="cpu") -> None:
    device = torch.device(device)
    _, modules = load_plan(input_dir)
    out: dict[str, torch.Tensor] = {}
    with ExitStack() as stack:
        reader = ShardReader(input_dir, stack)
        f = reader.handles[shard] = stack.enter_context(
            safe_open(os.path.join(input_dir, shard), framework="pt", device="cpu")
        )
        for name in f.keys():
            m = EXPERT_RE.match(name)
            if m is None:
                out[name] = f.get_tensor(name)
                stats["kept"] += 1
                continue
            prefix, leaf = m.group(1), m.group(2)
            if leaf in SCALE_LEAVES:
                stats["dropped_scales"] += 1
                continue
            where = modules[prefix]
            w_packed = f.get_tensor(name)
            ws = reader.get(where["weight_scale"], f"{prefix}.weight_scale")
            ws2 = reader.get(where["weight_scale_2"], f"{prefix}.weight_scale_2")
            if w_packed.dtype != torch.uint8 or ws.dtype != torch.float8_e4m3fn or ws2.numel() != 1:
                raise TypeError(
                    f"{prefix}: 期望 U8 / F8_E4M3 / 标量, 实际 {w_packed.dtype} / {ws.dtype} / {tuple(ws2.shape)}"
                )
            ref = dequant_nvfp4(w_packed.to(device), ws.to(device), ws2.to(device))
            packed, e8m0 = quantize_mxfp4(ref)
            if opts.measure_error:
                rec = dequant_mxfp4(packed, e8m0)
                stats["err_num"] += float(((rec - ref) ** 2).sum())
                stats["err_den"] += float((ref**2).sum())
            out[f"{prefix}.weight_packed"] = packed.cpu()
            out[f"{prefix}.weight_scale"] = e8m0.cpu()
            stats["nvfp4_to_mxfp4"] += 1
    save_file(out, os.path.join(output_dir, shard), metadata={"format": "pt"})


def _empty_stats() -> dict[str, float]:
    return {k: 0 for k in _STATS_KEYS}


def _worker(args):
    input_dir, shard, output_dir, opts, gpu_id = args
    torch.cuda.set_device(gpu_id)
    stats = _empty_stats()
    convert_one_file(input_dir, shard, output_dir, opts, stats, torch.device(f"cuda:{gpu_id}"))
    return stats


def convert_model(input_dir, output_dir, opts, limit_files, device_kind, gpu_ids):
    os.makedirs(output_dir, exist_ok=True)
    wm, _ = load_plan(input_dir)
    shards = sorted(set(wm.values()))
    if limit_files is not None:
        shards = shards[:limit_files]
    stats = _empty_stats()

    cuda = torch.cuda.is_available()
    use_gpu = device_kind == "cuda" or (device_kind == "auto" and cuda)
    if use_gpu and not cuda:
        raise RuntimeError("--device cuda specified but CUDA is unavailable")
    ids = ([int(x) for x in gpu_ids.split(",") if x.strip()] if gpu_ids
           else list(range(torch.cuda.device_count()))) if use_gpu else []

    if len(ids) > 1:
        import torch.multiprocessing as mp
        tasks = [(input_dir, s, output_dir, opts, ids[i % len(ids)]) for i, s in enumerate(shards)]
        with mp.get_context("spawn").Pool(processes=min(len(ids), len(shards))) as pool:
            for per in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks)):
                for k, v in per.items():
                    stats[k] += v
    else:
        device = torch.device(f"cuda:{ids[0]}") if ids else torch.device("cpu")
        for s in tqdm(shards, desc=f"Converting on {device}"):
            convert_one_file(input_dir, s, output_dir, opts, stats, device)
    return stats, shards


# --------------------------------------------------------------------------- #
# config / index / 其余文件
# --------------------------------------------------------------------------- #

def mxfp4_input_activations(mode: str) -> dict | None:
    """MXFP4 层的激活声明 (仅 config, 不影响权重字节)。"""
    if mode == "w4a4":
        return {"dynamic": True, "num_bits": 4, "type": "float", "strategy": "group",
                "group_size": 32, "symmetric": True, "scale_dtype": "torch.uint8",
                "zp_dtype": None}
    return None


def build_compression_config(opts: ConvOptions) -> dict:
    return {
        "config_groups": {
            "group_mxfp4": {
                "targets": [
                    # 不锚定前缀: vLLM 匹配的是运行时模块名 (language_model.model.layers.N...),
                    # 与 checkpoint 名 (model.language_model.layers.N...) 不同; 排除 mtp (FP8 专家)
                    r"re:^(?!.*mtp).*\.mlp\.experts\.\d+\.(?:gate|up|down)_proj$"
                ],
                "weights": {
                    "dynamic": False, "num_bits": 4, "type": "float", "strategy": "group",
                    "group_size": 32, "symmetric": True, "scale_dtype": "torch.uint8",
                    "zp_dtype": None,
                },
                "input_activations": mxfp4_input_activations(opts.mode),
                "format": "mxfp4-pack-quantized",
            }
        },
        "format": "mixed-precision",
        "ignore": [
            "lm_head",
            r"re:.*embed_tokens$",
            r"re:.*\.mlp\.gate$",
            r"re:.*\.mlp\.shared_expert.*",
            r"re:.*\.linear_attn.*",
            r"re:.*\.self_attn.*",
            r"re:.*hyper_connection.*",
            r"re:.*\.ple\..*",
            r"re:.*visual.*",
            r"re:.*mtp.*",
        ],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def write_config(input_dir: str, output_dir: str, opts: ConvOptions) -> None:
    with open(os.path.join(input_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    src_qc = cfg.pop("quantization_config", None)
    if not src_qc or src_qc.get("quant_method") != "modelopt":
        raise ValueError(f"源 config 不是 ModelOpt 量化: {None if not src_qc else src_qc.get('quant_method')}")
    cfg["quantization_config"] = build_compression_config(opts)
    cfg["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, sort_keys=True)


def rebuild_index(input_dir: str, output_dir: str, shards: list[str]) -> None:
    """用 get_slice 读形状/dtype, 不加载数据。"""
    sizes = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8, "U8": 1, "F8_E4M3": 1, "F64": 8, "I32": 4}
    with open(os.path.join(input_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        index = json.load(f)
    wm, total = {}, 0
    for s in shards:
        with safe_open(os.path.join(output_dir, s), framework="pt", device="cpu") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                n = 1
                for d in sl.get_shape():
                    n *= d
                total += n * sizes[sl.get_dtype()]
                wm[k] = s
    index["weight_map"] = wm
    index["metadata"] = {**index.get("metadata", {}), "total_size": total}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_others(input_dir: str, output_dir: str) -> None:
    for entry in sorted(os.listdir(input_dir)):
        if entry in SOURCE_ONLY_FILES or entry.endswith(".safetensors"):
            continue
        s, d = os.path.join(input_dir, entry), os.path.join(output_dir, entry)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copyfile(s, d)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #

def _ref_quantize_nvfp4(w: torch.Tensor):
    """按 Model-Optimizer 公式 (nvfp4_tensor.py:216/227) 造一个 NVFP4 张量, 仅供自检。"""
    N, K = w.shape
    amax = w.abs().max()
    ws2 = amax / (E2M1_MAX * E4M3_MAX)
    blk = w.view(N, K // NVFP4_GROUP, NVFP4_GROUP)
    bs = (blk.abs().amax(-1) / (E2M1_MAX * ws2)).clamp(min=2**-9, max=E4M3_MAX).to(torch.float8_e4m3fn)
    scaled = blk / (bs.float().unsqueeze(-1) * ws2)
    codes = float_to_e2m1_codes(scaled).view(N, K)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), bs, ws2


def _self_check() -> None:
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.equal(float_to_e2m1_codes(vals), torch.arange(8, dtype=torch.uint8))
    assert torch.equal(float_to_e2m1_codes(-vals[1:]), torch.arange(1, 8, dtype=torch.uint8) | 8)
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 100.0])
    assert float_to_e2m1_codes(ties).tolist() == [0, 2, 2, 4, 4, 6, 6, 7]
    got = compute_e8m0_exponent(torch.tensor([6.0, 7.0, 1.74, 1.75, 0.0, 1e-45])).tolist()
    assert got == [0, 1, -2, -1, -127, -127], got

    torch.manual_seed(0)
    w = torch.randn(16, 128)
    packed, e8m0 = quantize_mxfp4(w)
    assert packed.shape == (16, 64) and e8m0.shape == (16, 4) and e8m0.dtype == torch.uint8
    rec = dequant_mxfp4(packed, e8m0)
    assert ((rec - w) ** 2).mean() / (w**2).mean() < 0.05
    assert torch.equal(dequant_mxfp4(*quantize_mxfp4(rec)), rec), "MXFP4 grid must be a fixed point"

    p = torch.tensor([[0x21, 0x98]], dtype=torch.uint8)
    assert unpack_fp4_nibbles(p).tolist() == [[1, 2, 8, 9]]
    packed32 = torch.zeros(1, 16, dtype=torch.uint8)
    packed32[0, :2] = p[0]
    ws = torch.full((1, 2), 1.5).to(torch.float8_e4m3fn)
    d = dequant_nvfp4(packed32, ws, torch.tensor(2.0))
    assert d[0, :4].tolist() == [1.5, 3.0, 0.0, -1.5], d[0, :4]

    # NVFP4 (按 ModelOpt 公式造) -> 反量化 ≈ 原权重 -> MXFP4: 两步误差都应在合理范围
    w = torch.randn(64, 256) * 0.02
    p4, bs, ws2 = _ref_quantize_nvfp4(w)
    deq = dequant_nvfp4(p4, bs, ws2)
    e1 = ((deq - w) ** 2).sum() / (w**2).sum()
    mx, ex = quantize_mxfp4(deq)
    e2 = ((dequant_mxfp4(mx, ex) - w) ** 2).sum() / (w**2).sum()
    assert e1 < 0.02 and e2 < 0.06 and e2 >= e1 * 0.9, (float(e1), float(e2))

    # 正则: 只命中主干专家; shared_expert / mtp / ple 不命中
    ok = "model.language_model.layers.7.mlp.experts.511.down_proj.weight_scale_2"
    assert EXPERT_RE.match(ok).group(2) == "weight_scale_2"
    for n in ("model.language_model.layers.7.mlp.shared_expert.down_proj.weight",
              "mtp.layers.0.mlp.experts.3.up_proj.weight",
              "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
              "model.language_model.layers.7.mlp.experts.3.up_proj.weight_scale_inv"):
        assert EXPERT_RE.match(n) is None, n
    c4 = build_compression_config(ConvOptions(mode="w4a4"))["config_groups"]["group_mxfp4"]
    c16 = build_compression_config(ConvOptions())["config_groups"]["group_mxfp4"]
    assert c16["input_activations"] is None and c4["input_activations"]["num_bits"] == 4
    assert c4["weights"] == c16["weights"] and c4["targets"] == c16["targets"]
    tgt = c16["targets"][0][3:]
    for n in ("model.language_model.layers.7.mlp.experts.3.up_proj",          # checkpoint 名
              "language_model.model.layers.0.mlp.experts.0.gate_proj"):       # vLLM 运行时名
        assert re.match(tgt, n), n
    for n in ("model.language_model.layers.7.mlp.shared_expert.up_proj",
              "language_model.model.layers.0.mlp.shared_expert.gate_proj",
              "mtp.layers.0.mlp.experts.3.up_proj",
              "language_model.mtp.layers.0.mlp.experts.0.gate_proj",
              "language_model.model.layers.3.self_attn.q_proj"):
        assert not re.match(tgt, n), n


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = ArgumentParser(
        description="Qwen3.8-Flash-Next NVFP4 (ModelOpt) -> MXFP4 (compressed-tensors), 仅转路由专家。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["w4a16", "w4a4"], default="w4a16",
                    help="只影响 config 里 MXFP4 层的 input_activations, 权重字节相同")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--gpu-ids", default=None, help="逗号分隔; 多于 1 张时按分片多进程并行")
    ap.add_argument("--limit-files", type=int, default=None, help="只转前 N 个分片 (冒烟, 不写完整 index)")
    ap.add_argument("--measure-error", action="store_true",
                    help="统计 MXFP4 相对 NVFP4 反量化值的额外相对 MSE (较慢)")
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must be different")
    torch.set_num_threads(args.num_threads)
    _self_check()

    opts = ConvOptions(mode=args.mode, measure_error=args.measure_error)
    _, modules = load_plan(args.input_dir)
    split = sum(len(set(v.values())) > 1 for v in modules.values())
    print(f"NVFP4 专家模块 {len(modules)} 个 (预期 {EXPECTED_EXPERT_MODULES}); 张量跨分片的模块 {split} 个")
    if len(modules) != EXPECTED_EXPERT_MODULES:
        print("[warn] 模块数与 Qwen3.8-Flash-Next 预期不符, 请确认输入")

    stats, shards = convert_model(args.input_dir, args.output_dir, opts, args.limit_files,
                                  args.device, args.gpu_ids)
    write_config(args.input_dir, args.output_dir, opts)
    copy_others(args.input_dir, args.output_dir)
    if args.limit_files is None:
        rebuild_index(args.input_dir, args.output_dir, shards)
        if stats["nvfp4_to_mxfp4"] != len(modules):
            raise RuntimeError(f"转换数 {stats['nvfp4_to_mxfp4']} != 模块数 {len(modules)}")
        if stats["dropped_scales"] != 3 * len(modules):
            raise RuntimeError(f"丢弃的 scale 数 {stats['dropped_scales']} != {3 * len(modules)}")
    summary = {"options": opts.__dict__, "shards": shards, "stats": stats}
    if opts.measure_error and stats["err_den"]:
        summary["rel_mse_mxfp4_vs_nvfp4_dequant"] = stats["err_num"] / stats["err_den"]
    with open(os.path.join(args.output_dir, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Done -> {args.output_dir}")


if __name__ == "__main__":
    main()
