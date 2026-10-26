"""
MSE check for MTP slimquant (aligned with qwen38_flash_next_mtp_to_w4a8.py).

Source: BF16 MTP checkpoint (typically ngram-int8, MTP still floating).
Destination: /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp

Checks:
  - INT4 experts: unpack nibble + scale*16 vs BF16
  - INT8 attn/shared: int8 * scale vs BF16

Usage (inside container):
  python3 qwen38_flash_next_mtp_to_w4a8-MSE.py
  python3 qwen38_flash_next_mtp_to_w4a8-MSE.py \\
      --src /public4/opendas/DL_DATA/llm-models-qua/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8 \\
      --dst /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_SRC_CANDIDATES = [
    Path("/module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8"),
    Path(
        "/public4/opendas/DL_DATA/llm-models-qua/"
        "Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8"
    ),
]
DEFAULT_DST = Path("/module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp")

MTP_EXPERT_RE = re.compile(
    r"^mtp\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
MTP_W8_WEIGHT_RE = re.compile(
    r"^mtp\.layers\.\d+\."
    r"(?:self_attn\.(q_proj|k_proj|v_proj)|"
    r"mlp\.shared_expert\.(gate_proj|up_proj|down_proj))"
    r"\.weight$"
)

# Health thresholds for per-channel symmetric quant (weight recon vs BF16).
# INT4 nibble typically lands ~0.10–0.20 rel L2; INT8 ~0.01–0.04.
# Flag only if clearly above the normal quant noise floor.
WARN_REL_L2_INT4 = 0.25
WARN_REL_L2_INT8 = 0.05


def resolve_default_src() -> Path:
    for p in DEFAULT_SRC_CANDIDATES:
        if p.is_dir() and (p / "config.json").is_file():
            return p
    return DEFAULT_SRC_CANDIDATES[-1]


def load(root: Path, wm: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(str(root / wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def unpack_int4_nibble(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.uint8)
    even = ((p >> 4) & 0x0F).to(torch.int32)
    odd = (p & 0x0F).to(torch.int32)
    even = torch.where(even >= 8, even - 16, even)
    odd = torch.where(odd >= 8, odd - 16, odd)
    codes = torch.stack([even, odd], dim=-1).reshape(*packed.shape[:-1], -1)
    return codes.to(torch.float32)


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    codes = unpack_int4_nibble(packed)
    return codes * (scale.float() * 16.0)


def mse_rel(ref: torch.Tensor, recon: torch.Tensor) -> tuple[float, float]:
    diff = ref - recon
    mse = torch.mean(diff ** 2).item()
    rel = (torch.norm(diff) / torch.norm(ref).clamp_min(1e-12)).item()
    return mse, rel


def eval_int4(
    src_root: Path,
    dst_root: Path,
    src_idx: dict[str, str],
    dst_idx: dict[str, str],
) -> list[tuple[str, float, float]]:
    names = sorted(n for n in dst_idx if MTP_EXPERT_RE.match(n))
    if not names:
        raise RuntimeError("No MTP INT4 expert weights in destination index.")

    print(f"\n=== MTP INT4 experts ({len(names)}) ===")
    print(f"src={src_root}")
    print(f"dst={dst_root}")
    print("-" * 80)

    rows: list[tuple[str, float, float]] = []
    for i, name in enumerate(names, 1):
        if name not in src_idx:
            raise KeyError(f"missing in source index: {name}")
        scale_name = f"{name}.weight_scale"
        if scale_name not in dst_idx:
            raise KeyError(f"missing scale in destination: {scale_name}")

        w_bf16 = load(src_root, src_idx, name).float()
        w_packed = load(dst_root, dst_idx, name)
        scale = load(dst_root, dst_idx, scale_name).float()

        if w_packed.dtype not in (torch.int8, torch.uint8):
            raise TypeError(f"{name}: expected packed int8, got {w_packed.dtype}")
        if w_bf16.shape[:-1] != w_packed.shape[:-1]:
            raise ValueError(
                f"{name}: shape mismatch bf16={tuple(w_bf16.shape)} "
                f"packed={tuple(w_packed.shape)}"
            )
        if w_bf16.shape[-1] != w_packed.shape[-1] * 2:
            raise ValueError(
                f"{name}: K mismatch bf16 K={w_bf16.shape[-1]} "
                f"packed K/2={w_packed.shape[-1]}"
            )

        recon = dequant_int4(w_packed, scale)
        mse, rel = mse_rel(w_bf16, recon)
        flag = "  WARN" if rel > WARN_REL_L2_INT4 else ""
        rows.append((name, mse, rel))
        print(
            f"[{i}/{len(names)}] {name}  mse={mse:.6e}  rel_l2={rel:.6e}{flag}"
        )
    return rows


def eval_int8(
    src_root: Path,
    dst_root: Path,
    src_idx: dict[str, str],
    dst_idx: dict[str, str],
) -> list[tuple[str, float, float]]:
    names = sorted(n for n in dst_idx if MTP_W8_WEIGHT_RE.match(n))
    if not names:
        raise RuntimeError("No MTP INT8 W8 weights in destination index.")

    print(f"\n=== MTP INT8 attn/shared ({len(names)}) ===")
    print(f"src={src_root}")
    print(f"dst={dst_root}")
    print("-" * 80)

    rows: list[tuple[str, float, float]] = []
    for i, name in enumerate(names, 1):
        if name not in src_idx:
            raise KeyError(f"missing in source index: {name}")
        scale_name = name.replace(".weight", ".weight_scale")
        if scale_name not in dst_idx:
            raise KeyError(f"missing scale in destination: {scale_name}")

        w_src = load(src_root, src_idx, name)
        if w_src.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(
                f"{name}: source expected floating, got {w_src.dtype}"
            )
        w_bf16 = w_src.float()
        w_q = load(dst_root, dst_idx, name)
        scale = load(dst_root, dst_idx, scale_name).float()

        if w_q.dtype != torch.int8:
            raise TypeError(f"{name}: expected int8 weight, got {w_q.dtype}")
        if w_bf16.shape != w_q.shape:
            raise ValueError(
                f"{name}: shape mismatch bf16={tuple(w_bf16.shape)} "
                f"int8={tuple(w_q.shape)}"
            )

        recon = w_q.float() * scale
        mse, rel = mse_rel(w_bf16, recon)
        flag = "  WARN" if rel > WARN_REL_L2_INT8 else ""
        rows.append((name, mse, rel))
        print(
            f"[{i}/{len(names)}] {name}  mse={mse:.6e}  rel_l2={rel:.6e}{flag}"
        )
    return rows


def summarize(tag: str, rows: list[tuple[str, float, float]], warn_rel: float) -> bool:
    mses = torch.tensor([r[1] for r in rows], dtype=torch.float64)
    rels = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    worst = max(rows, key=lambda x: x[2])
    ok = bool(rels.max().item() <= warn_rel)
    print("-" * 80)
    print(f"[{tag}] count={len(rows)}")
    print(f"[{tag}] mse  mean={mses.mean().item():.6e}  max={mses.max().item():.6e}")
    print(f"[{tag}] rel  mean={rels.mean().item():.6e}  max={rels.max().item():.6e}")
    print(
        f"[{tag}] worst_rel_l2: {worst[0]}  "
        f"mse={worst[1]:.6e}  rel_l2={worst[2]:.6e}"
    )
    print(
        f"[{tag}] threshold rel_l2<={warn_rel:.3f} → "
        f"{'PASS' if ok else 'WARN (above threshold)'}"
    )
    return ok


def main(src_root: Path, dst_root: Path) -> None:
    if not src_root.is_dir():
        raise FileNotFoundError(f"src not found: {src_root}")
    if not dst_root.is_dir():
        raise FileNotFoundError(f"dst not found: {dst_root}")

    src_idx = json.load(open(src_root / "model.safetensors.index.json"))["weight_map"]
    dst_idx = json.load(open(dst_root / "model.safetensors.index.json"))["weight_map"]

    print("=" * 80)
    print("MTP W4A8 reconstruction MSE")
    print(f"src (BF16 MTP): {src_root}")
    print(f"dst (quant MTP): {dst_root}")
    print("=" * 80)

    int4_rows = eval_int4(src_root, dst_root, src_idx, dst_idx)
    int8_rows = eval_int8(src_root, dst_root, src_idx, dst_idx)

    print("\n=== Summary ===")
    ok4 = summarize("INT4", int4_rows, WARN_REL_L2_INT4)
    ok8 = summarize("INT8", int8_rows, WARN_REL_L2_INT8)
    print("-" * 80)
    if ok4 and ok8:
        print("OVERALL: PASS (no precision red flags vs thresholds)")
    else:
        print("OVERALL: WARN — inspect layers marked above")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="MSE: BF16 MTP vs INT4/INT8 reconstruction (mtp-w4a8)"
    )
    parser.add_argument("--src", type=Path, default=resolve_default_src())
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = parser.parse_args()
    main(args.src, args.dst)
