"""
Compare BF16 source vs INT8 reconstruction for all W8 include layers
under /module/Qwen3.8-Flash-Next-w4a8-slimquant.

W8 include (same as qwen38_flash_next_int4_to_int4_8.py):
  - self_attn.{q,k,v}_proj
  - mlp.shared_expert.{gate,up,down}_proj
  - linear_attn.{in_proj_qkv,in_proj_z}

Default source is the Stage-1 INT4 checkpoint (those layers still BF16 there).
You may also point --src at the original BF16 model.

Usage (inside container):
  python3 qwen38_flash_next_int4_to_int4_8-MSE.py
  python3 qwen38_flash_next_int4_to_int4_8-MSE.py \\
      --src /module/Qwen3.8-Flash-Next-INT4 \\
      --dst /module/Qwen3.8-Flash-Next-w4a8-slimquant
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_SRC = Path("/module/Qwen3.8-Flash-Next-INT4")
DEFAULT_DST = Path("/module/Qwen3.8-Flash-Next-w4a8-slimquant")

W8_WEIGHT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"model\.language_model\.layers\.\d+\.self_attn\."
        r"(q_proj|k_proj|v_proj)\.weight$"
    ),
    re.compile(
        r"model\.language_model\.layers\.\d+\.mlp\.shared_expert\."
        r"(gate_proj|up_proj|down_proj)\.weight$"
    ),
    re.compile(
        r"model\.language_model\.layers\.\d+\.linear_attn\."
        r"(in_proj_qkv|in_proj_z)\.weight$"
    ),
]


def is_w8_target_weight(name: str) -> bool:
    return any(p.match(name) is not None for p in W8_WEIGHT_PATTERNS)


def load(root: Path, wm: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(str(root / wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def main(src_root: Path, dst_root: Path) -> None:
    src_idx = json.load(open(src_root / "model.safetensors.index.json"))["weight_map"]
    dst_idx = json.load(open(dst_root / "model.safetensors.index.json"))["weight_map"]

    names = sorted(n for n in dst_idx if is_w8_target_weight(n))
    if not names:
        raise RuntimeError("No W8 target weights found in destination index.")

    print(f"Evaluating {len(names)} W8 layers")
    print(f"src={src_root}")
    print(f"dst={dst_root}")
    print("-" * 80)

    rows: list[tuple[str, float, float]] = []
    for i, name in enumerate(names, 1):
        if name not in src_idx:
            raise KeyError(f"missing in source index: {name}")
        scale_name = name.replace(".weight", ".weight_scale")
        if scale_name not in dst_idx:
            raise KeyError(f"missing scale in destination index: {scale_name}")

        w_bf16 = load(src_root, src_idx, name).float()
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
        description="MSE: BF16 vs INT8 reconstruction (Flash-Next-w4a8-slimquant W8)"
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = parser.parse_args()
    main(args.src, args.dst)
