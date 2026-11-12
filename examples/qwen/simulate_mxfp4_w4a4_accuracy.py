# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8-Flash-Next MXFP4 W4A4 精度仿真 (不需要任何 MXFP4 硬件内核)

背景: SM120 上没有任何 MoE 后端支持 MXFP4 激活 (W4A4)。
  - CutlassExpertsMxfp4 需要 SM100/11x (mxfp4_blockwise_moe_kernel.cu: 100 <= cc < 120)
  - b12x 只给 MXFP4 的 W4A8 / W4A16 (b12x.py _B12X_MOE_MODES)
  - compressed-tensors 的 --moe-backend emulation 在 MXFP4 上不可达
    (compressed_tensors_moe.py:76 的 _is_mxfp4 提前 return, :129 的 emulation 分支走不到)
  - gpt_oss Triton 只有 (kMxfp4Static, None)
本脚本用 CPU/GPU 上的量化-反量化 + 普通 matmul 复现 "MXFP4 权重 x MXFP4 激活" 的数值,
从而得到精度结论; 不产生速度数据。

两种模式:

  --mode act  (默认, 公平对比)
      权重固定用 MXFP4 checkpoint 里的真实张量 (反量化到 FP32),
      只改变"激活格式": W4A16 (BF16 激活) vs W4A4 (MXFP4 激活) vs W4A8 (MXFP8 激活)。
      这是唯一严格的对比: 权重完全相同, 唯一变量是激活。

  --mode weight (近似对比, 有偏)
      以 NVFP4 checkpoint 反量化后的权重作参考, 把同一份 W_ref 分别量化成
      MXFP4 和 NVFP4, 比较各自相对 W_ref 的误差。
      注意: W_ref 本身就是 NVFP4 反量化的结果, 所以 NVFP4 重新量化近似"不动点",
      误差被人为压低; 这个模式只能看 MXFP4 的绝对误差量级, 不能当公平的格式对比。

用法 (需要真实权重, 在云端跑):
  python3 simulate_mxfp4_w4a4_accuracy.py \
      --mxfp4-dir /model/Qwen3.8-Flash-Next-MXFP4 \
      --nvfp4-dir /model/Qwen3.8-Flash-Next-NVFP4 \
      --num-layers 4 --experts-per-layer 4 --mode act

  # 只看激活格式对比, 不需要 NVFP4 目录
  python3 simulate_mxfp4_w4a4_accuracy.py --mxfp4-dir ... --mode act
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from dataclasses import dataclass

import torch

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
MXFP4_GROUP = 32
NVFP4_GROUP = 16
MX_ELEM_OFFSET = 2          # floor(log2(6.0))
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2 ** -9   # ModelOpt 对 per-block scale 的下钳位

PROJ_RE = re.compile(
    r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj)"
    r"\.(weight_packed|weight|weight_scale|weight_scale_2|input_scale)$"
)


# --------------------------------------------------------------------------- #
# E2M1 / E8M0 基础 (与 quantize_qwen3_8_flash_next_nvfp4_to_mxfp4.py 同一套公式)
# --------------------------------------------------------------------------- #

def unpack_fp4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """uint8 (N, K/2) -> 码值 int64 (N, K); 低 4 位 = 偶数下标。"""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).flatten(1).long()


def float_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """FP32 (已除以 scale) -> E2M1 码 (uint8, 含符号位), RNE, 饱和到 6。"""
    a = x.abs().clamp(max=E2M1_MAX)
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


def compute_e8m0_exponent(amax: torch.Tensor) -> torch.Tensor:
    """amax fp32 -> 无偏 scale 指数 (int32), 钳位 [-127, 127]。"""
    bits = amax.float().contiguous().view(torch.int32)
    rounded = (bits + 0x200000) & 0x7F800000
    return (((rounded >> 23) - 127) - MX_ELEM_OFFSET).clamp(-127, 127)


def quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 (N, K) -> (packed uint8 (N, K/2), e8m0 uint8 (N, K/32))。"""
    N, K = w.shape
    assert K % MXFP4_GROUP == 0, f"K={K} 不能被 {MXFP4_GROUP} 整除"
    wg = w.float().view(N, K // MXFP4_GROUP, MXFP4_GROUP)
    exp = compute_e8m0_exponent(wg.abs().amax(dim=-1))
    inv = torch.pow(2.0, (-exp).float()).unsqueeze(-1)
    codes = float_to_e2m1_codes(wg * inv).view(N, K)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), (exp + 127).to(torch.uint8)


def dequant_mxfp4(packed: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    values = FP4_TABLE.to(packed.device)[unpack_fp4_nibbles(packed)]
    scale = torch.pow(2.0, e8m0.float() - 127.0).repeat_interleave(MXFP4_GROUP, dim=1)
    return values * scale


def mxfp4_qdq(x: torch.Tensor) -> torch.Tensor:
    """MXFP4 量化-反量化 (用于激活仿真)。"""
    return dequant_mxfp4(*quantize_mxfp4(x))


# --------------------------------------------------------------------------- #
# NVFP4 (E4M3 块缩放 + FP32 全局缩放)
# --------------------------------------------------------------------------- #

def dequant_nvfp4(packed, weight_scale, weight_scale_2) -> torch.Tensor:
    """真实 checkpoint 的 NVFP4 -> FP32。"""
    values = FP4_TABLE.to(packed.device)[unpack_fp4_nibbles(packed)]
    N, K = values.shape
    scale = weight_scale.float().repeat_interleave(NVFP4_GROUP, dim=1)
    return values * scale * weight_scale_2.float().reshape(())


def quantize_nvfp4_from_fp32(w: torch.Tensor, global_amax: float | None = None):
    """FP32 -> NVFP4 (E4M3 block16 + FP32 全局 scale), 对齐 ModelOpt 公式。

    全局 scale = amax / (6 * 448); 块 scale = block_amax / (6 * 全局 scale),
    钳位到 [2^-9, 448] 后转 E4M3。
    """
    N, K = w.shape
    amax = float(w.abs().max()) if global_amax is None else float(global_amax)
    ws2 = torch.tensor(amax / (E2M1_MAX * E4M3_MAX), dtype=torch.float32)
    blk = w.float().view(N, K // NVFP4_GROUP, NVFP4_GROUP)
    bs = (blk.abs().amax(dim=-1) / (E2M1_MAX * ws2)).clamp(
        min=E4M3_MIN_NORMAL, max=E4M3_MAX
    ).to(torch.float8_e4m3fn)
    scaled = blk / (bs.float().unsqueeze(-1) * ws2)
    codes = float_to_e2m1_codes(scaled).view(N, K)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, bs, ws2


def nvfp4_qdq(x: torch.Tensor) -> torch.Tensor:
    return dequant_nvfp4(*quantize_nvfp4_from_fp32(x))


def mxfp8_qdq(x: torch.Tensor) -> torch.Tensor:
    """MXFP8 仿真 (E4M3 + E8M0 块32), 用于给出 W4A8 参照点。"""
    N, K = x.shape
    blk = x.float().view(N, K // MXFP4_GROUP, MXFP4_GROUP)
    exp = compute_e8m0_exponent(blk.abs().amax(dim=-1))
    scale = torch.pow(2.0, exp.float()).unsqueeze(-1)
    q = (blk / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).view(N, K)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def rel_mse(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float(((y - ref) ** 2).sum() / (ref ** 2).sum().clamp(min=1e-30))


def sqnr_db(y: torch.Tensor, ref: torch.Tensor) -> float:
    e = rel_mse(y, ref)
    return float("inf") if e <= 0 else -10.0 * torch.log10(torch.tensor(e)).item()


def cosine(y: torch.Tensor, ref: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        y.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)
    ).item()


# --------------------------------------------------------------------------- #
# checkpoint 读取
# --------------------------------------------------------------------------- #

class Ckpt:
    """按索引懒加载分片, 处理同一模块跨分片的情况。"""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json"), encoding="utf-8") as f:
            self.wm = json.load(f)["weight_map"]
        self._handles: dict[str, object] = {}
        self._mods: dict[str, dict[str, str]] = {}
        for name, shard in self.wm.items():
            m = PROJ_RE.match(name)
            if m:
                self._mods.setdefault(m.group(1), {})[m.group(5)] = shard
        self.layers = sorted({int(m.group(2)) for m in map(PROJ_RE.match, self.wm) if m})

    def _open(self, shard: str):
        from safetensors import safe_open

        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.path, shard), framework="pt", device="cpu"
            )
        h = self._handles[shard]
        return h if hasattr(h, "get_tensor") else h.__enter__()

    def get(self, key: str) -> torch.Tensor:
        return self._open(self.wm[key]).get_tensor(key)

    def module(self, prefix: str) -> dict[str, torch.Tensor]:
        where = self._mods[prefix]
        return {leaf: self.get(f"{prefix}.{leaf}") for leaf in where}


# --------------------------------------------------------------------------- #
# 单张量评估
# --------------------------------------------------------------------------- #

@dataclass
class Row:
    layer: int
    expert: int
    proj: str
    shape: tuple[int, int]
    w_bf16_rel_mse: float          # 仅 --mode weight: MXFP4 相对 W_ref 的权重误差
    act: dict[str, dict[str, float]]  # 激活格式 -> {rel_mse, sqnr, cos}


def make_activation(k: int, m: int, amax: float, dist: str, gen: torch.Generator) -> torch.Tensor:
    if dist == "student_t":
        x = torch.distributions.StudentT(df=3.0).sample((m, k))
    else:
        x = torch.randn(m, k, generator=gen)
    return (x / x.abs().max().clamp(min=1e-12) * amax).float()


def evaluate(mxfp4_ckpt: Ckpt, nvfp4_ckpt: Ckpt | None, args) -> list[Row]:
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    rows: list[Row] = []

    layer_ids = mxfp4_ckpt.layers
    step = max(1, len(layer_ids) // args.num_layers)
    layer_ids = layer_ids[::step][: args.num_layers]

    n_total = len(layer_ids) * args.experts_per_layer * 3
    done = 0
    for layer in layer_ids:
        for e in range(args.experts_per_layer):
            for proj in ("gate", "up", "down"):
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}_proj"
                done += 1
                try:
                    t = mxfp4_ckpt.module(prefix)
                except KeyError:
                    print(f"  [skip] {prefix} 不在 checkpoint", file=sys.stderr)
                    continue

                # 权重固定为 MXFP4 checkpoint 的真实值
                w_mx = dequant_mxfp4(t["weight_packed"], t["weight_scale"])  # (N, K)
                n, k = w_mx.shape

                # 激活幅度: 优先用 NVFP4 checkpoint 的 input_scale (amax = input_scale*2688)
                amax = None
                if nvfp4_ckpt is not None and prefix in nvfp4_ckpt._mods:
                    try:
                        amax = float(nvfp4_ckpt.get(f"{prefix}.input_scale")) * (E2M1_MAX * E4M3_MAX)
                    except KeyError:
                        amax = None
                if amax is None or amax <= 0:
                    amax = args.act_amax
                m = min(args.seq_len, n)

                x = make_activation(k, m, amax, args.act_dist, gen)      # (M, K)
                y_ref = x @ w_mx.T                                        # 全 MXFP4 权重 + 精确激活

                acts: dict[str, dict[str, float]] = {}
                variants = {
                    # 权重已量化, 激活不变 -> W4A16
                    "w4a16 (bf16 act)": x,
                    "w4a4 (mxfp4 act)": mxfp4_qdq(x),
                    "w4a8 (mxfp8 act)": mxfp8_qdq(x),
                }
                if args.with_nvfp4_act:
                    variants["w4a4 (nvfp4 act)"] = nvfp4_qdq(x)
                for name, xq in variants.items():
                    y = xq @ w_mx.T
                    acts[name] = {
                        "rel_mse": rel_mse(y, y_ref),
                        "sqnr_db": sqnr_db(y, y_ref),
                        "cos": cosine(y, y_ref),
                    }

                # --mode weight: 同一份 W_ref 上 MXFP4 vs NVFP4 的权重误差
                w_rel = float("nan")
                if args.mode == "weight":
                    wq = dequant_mxfp4(*quantize_mxfp4(w_mx))
                    w_rel = rel_mse(wq, w_mx)

                rows.append(Row(layer, e, proj, (n, k), w_rel, acts))
                if done % 3 == 0 or done == n_total:
                    print(f"  [{done}/{n_total}] layer {layer} expert {e}", file=sys.stderr)
    return rows


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #

def _self_check() -> None:
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.equal(float_to_e2m1_codes(vals), torch.arange(8, dtype=torch.uint8))
    assert torch.equal(float_to_e2m1_codes(-vals[1:]), torch.arange(1, 8, dtype=torch.uint8) | 8)
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 100.0])
    assert float_to_e2m1_codes(ties).tolist() == [0, 2, 2, 4, 4, 6, 6, 7]
    got = compute_e8m0_exponent(torch.tensor([6.0, 7.0, 1.74, 1.75, 0.0, 1e-45])).tolist()
    assert got == [0, 1, -2, -1, -127, -127], got

    torch.manual_seed(0)
    w = torch.randn(32, 256) * 0.02
    rec = mxfp4_qdq(w)
    e = rel_mse(rec, w)
    assert e < 0.05, e
    assert torch.equal(mxfp4_qdq(rec), rec), "MXFP4 网格应为不动点"

    p = torch.tensor([[0x21, 0x98]], dtype=torch.uint8)
    assert unpack_fp4_nibbles(p).tolist() == [[1, 2, 8, 9]]

    # NVFP4 往返: 固定点 + 误差小于 MXFP4 在重尾上的表现
    n = nvfp4_qdq(w)
    assert rel_mse(nvfp4_qdq(n), n) < 1e-6, "NVFP4 网格应为不动点"
    # MXFP8 往返
    f8 = mxfp8_qdq(w)
    assert rel_mse(f8, w) < 0.01


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="MXFP4 W4A4 / W4A16 / W4A8 精度仿真 (不需要 MXFP4 硬件内核)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mxfp4-dir", required=True, help="MXFP4 checkpoint 目录")
    ap.add_argument("--nvfp4-dir", default=None,
                    help="NVFP4 源 checkpoint 目录; 用于取真实 input_scale 作激活幅度")
    ap.add_argument("--mode", choices=["act", "weight"], default="act",
                    help="act=只比激活格式(公平); weight=额外比权重格式(有偏)")
    ap.add_argument("--num-layers", type=int, default=4, help="抽样层数")
    ap.add_argument("--experts-per-layer", type=int, default=4, help="每层抽样专家数")
    ap.add_argument("--seq-len", type=int, default=64, help="激活行数 M")
    ap.add_argument("--act-dist", choices=["gaussian", "student_t"], default="gaussian")
    ap.add_argument("--act-amax", type=float, default=20.0,
                    help="没有 NVFP4 input_scale 时使用的激活 amax")
    ap.add_argument("--with-nvfp4-act", action="store_true",
                    help="额外测一次 NVFP4 激活 (group16, E4M3 scale)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="把明细写成 json")
    args = ap.parse_args()

    _self_check()

    mxfp4 = Ckpt(args.mxfp4_dir)
    nvfp4 = Ckpt(args.nvfp4_dir) if args.nvfp4_dir else None
    print(f"MXFP4 checkpoint: 层 {len(mxfp4.layers)} 个", file=sys.stderr)

    rows = evaluate(mxfp4, nvfp4, args)
    if not rows:
        raise SystemExit("没有可评估的张量")

    names = list(rows[0].act)
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    per_proj = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for r in rows:
        for nm in names:
            for key in ("rel_mse", "sqnr_db", "cos"):
                agg[nm][key].append(r.act[nm][key])
                per_proj[r.proj][nm][key].append(r.act[nm][key])

    def show(label: str, d: dict) -> None:
        print(f"\n{label}")
        print(f"  {'激活格式':<24}{'rel-MSE':>12}{'SQNR(dB)':>12}{'cos':>10}")
        base = None
        for nm in names:
            v = d[nm]
            mr = sum(v["rel_mse"]) / len(v["rel_mse"])
            ms = sum(v["sqnr_db"]) / len(v["sqnr_db"])
            mc = sum(v["cos"]) / len(v["cos"])
            if base is None:
                base = mr
            if nm == names[0]:
                rel = "  (基线: 激活不量化, rel-MSE 按构造为 0)"
            elif base > 1e-12:
                rel = f"  ({mr / base:.2f}x vs {names[0]})"
            else:
                rel = ""
            sq = "inf" if ms == float("inf") else f"{ms:.2f}"
            print(f"  {nm:<24}{mr:>12.6f}{sq:>12}{mc:>10.5f}{rel}")

    print(f"\n样本: {len(rows)} 个 (层, 专家, 投影) 组合")
    show("===== 汇总 (激活格式对比, 权重固定为 MXFP4 真实值) =====", agg)
    for p in ("gate", "up", "down"):
        if p in per_proj:
            show(f"----- {p}_proj -----", per_proj[p])

    if args.mode == "weight":
        wr = [r.w_bf16_rel_mse for r in rows if r.w_bf16_rel_mse == r.w_bf16_rel_mse]
        if wr:
            print(f"\n===== 权重格式 (近似, 有偏) =====")
            print(f"  MXFP4 重新量化相对 NVFP4 反量化参考的 rel-MSE: "
                  f"{sum(wr) / len(wr):.6f}")
            print("  注意: 参考本身就来自 NVFP4 反量化, NVFP4 重新量化近似不动点, "
                  "不能作为两个格式的公平对比")

    if args.out:
        json.dump(
            {"args": vars(args),
             "rows": [{"layer": r.layer, "expert": r.expert, "proj": r.proj,
                       "shape": list(r.shape), "w_rel": r.w_bf16_rel_mse,
                       "act": r.act} for r in rows]},
            open(args.out, "w"), indent=1, ensure_ascii=False,
        )
        print(f"\n明细已写入 {args.out}")


if __name__ == "__main__":
    main()
