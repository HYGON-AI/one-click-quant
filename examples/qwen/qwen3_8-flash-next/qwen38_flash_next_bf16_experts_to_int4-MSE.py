"""
Compare BF16 source vs INT4 reconstruction for fused expert weights
under /module/Qwen3.8-Flash-Next-INT4.

Matches packing in qwen38_flash_next_bf16_experts_to_int4.py:
  - per-channel scale along N; stored scale = abs_max / 7 / 16
  - K-nibble pack: (even << 4) | odd
  - recon = unpack_signed_int4(packed) * (scale * 16)

Usage (inside container):
  python3 qwen38_flash_next_bf16_experts_to_int4-MSE.py
  python3 qwen38_flash_next_bf16_experts_to_int4-MSE.py \\
      --src /model/Qwen/Qwen3.8-Flash-Next \\
      --dst /module/Qwen3.8-Flash-Next-INT4
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_SRC = Path("/model/Qwen/Qwen3.8-Flash-Next")
DEFAULT_DST = Path("/module/Qwen3.8-Flash-Next-INT4")

EXPERT_WEIGHT_RE = re.compile(
    r"(?:^|.*\.)language_model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"
)


def is_expert_weight(name: str) -> bool:
    return EXPERT_WEIGHT_RE.match(name) is not None


def expert_scale_name(weight_name: str) -> str:
    return f"{weight_name}.weight_scale"


def load(root: Path, wm: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(str(root / wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def unpack_int4_nibble(packed: torch.Tensor) -> torch.Tensor:
    """Unpack slimquant K-nibble int8 → signed INT4 codes (same shape as BF16 K)."""
    p = packed.to(torch.uint8)
    even = ((p >> 4) & 0x0F).to(torch.int32)
    odd = (p & 0x0F).to(torch.int32)
    even = torch.where(even >= 8, even - 16, even)
    odd = torch.where(odd >= 8, odd - 16, odd)
    # Interleave back to K: [e0,o0,e1,o1,...]
    codes = torch.stack([even, odd], dim=-1).reshape(*packed.shape[:-1], -1)
    return codes.to(torch.float32)


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """packed (..., K/2) int8 + scale (..., 1) → float (..., K)."""
    codes = unpack_int4_nibble(packed)
    # Stored scale already /16; multiply back for float reconstruction.
    return codes * (scale.float() * 16.0)


def main(src_root: Path, dst_root: Path) -> None:
    src_idx = json.load(open(src_root / "model.safetensors.index.json"))["weight_map"]
    dst_idx = json.load(open(dst_root / "model.safetensors.index.json"))["weight_map"]

    names = sorted(n for n in dst_idx if is_expert_weight(n))
    if not names:
        raise RuntimeError("No fused expert weights found in destination index.")

    print(f"Evaluating {len(names)} INT4 expert tensors")
    print(f"src={src_root}")
    print(f"dst={dst_root}")
    print("-" * 80)

    rows: list[tuple[str, float, float]] = []
    for i, name in enumerate(names, 1):
        if name not in src_idx:
            raise KeyError(f"missing in source index: {name}")
        scale_name = expert_scale_name(name)
        if scale_name not in dst_idx:
            raise KeyError(f"missing scale in destination index: {scale_name}")

        w_bf16 = load(src_root, src_idx, name).float()
        w_packed = load(dst_root, dst_idx, name)
        scale = load(dst_root, dst_idx, scale_name).float()

        if w_packed.dtype not in (torch.int8, torch.uint8):
            raise TypeError(
                f"{name}: expected packed int8, got {w_packed.dtype}"
            )
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
        mse = torch.mean((w_bf16 - recon) ** 2).item()
        rel = (torch.norm(w_bf16 - recon) / torch.norm(w_bf16).clamp_min(1e-12)).item()
        rows.append((name, mse, rel))
        print(f"[{i}/{len(names)}] {name}  mse={mse:.6e}  rel_l2={rel:.6e}")

    mses = torch.tensor([r[1] for r in rows], dtype=torch.float64)
    rels = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    worst = max(rows, key=lambda x: x[2])

    print("-" * 80)
    print(f"count={len(rows)}")
    print(f"mse  mean={mses.mean().item():.6e}  max={mses.max().item():.6e}")
    print(f"rel  mean={rels.mean().item():.6e}  max={rels.max().item():.6e}")
    print(f"worst_rel_l2: {worst[0]}  mse={worst[1]:.6e}  rel_l2={worst[2]:.6e}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="MSE: BF16 experts vs INT4 reconstruction (Flash-Next-INT4)"
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = parser.parse_args()
    main(args.src, args.dst)
