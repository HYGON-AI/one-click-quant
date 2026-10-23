"""
Compare BF16 source vs INT8 reconstruction for PLE ngram_embedding shards.

Default:
  src = /module/Qwen3.8-Flash-Next-w4a8-slimquant          (BF16 ngram)
  dst = /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8

Auto-detects scale layout produced by qwen38_flash_next_ngram_to_int8.py:
  - per_tensor:  ...ngram_embedding.weight_scale          # scalar (1,)
  - per_channel: ...ngram_embedding.shard_N.weight_scale  # (rows, 1)

recon = int8.float() * scale

Usage:
  python3 qwen38_flash_next_ngram_to_int8-MSE.py
  python3 qwen38_flash_next_ngram_to_int8-MSE.py \\
      --src /module/Qwen3.8-Flash-Next-w4a8-slimquant \\
      --dst /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_SRC = Path("/module/Qwen3.8-Flash-Next-w4a8-slimquant")
DEFAULT_DST = Path("/module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8")

NGRAM_SHARD_WEIGHT_RE = re.compile(
    r".*\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$"
)
NGRAM_MODULE_PREFIX_RE = re.compile(
    r"^(?P<prefix>.*\.ple\.ple_embedding\.ngram_embedding)\.shard_\d+\.weight$"
)


def is_ngram_shard_weight(name: str) -> bool:
    return NGRAM_SHARD_WEIGHT_RE.match(name) is not None


def ngram_module_prefix(weight_name: str) -> str:
    m = NGRAM_MODULE_PREFIX_RE.match(weight_name)
    if m is None:
        raise ValueError(f"not an ngram shard weight: {weight_name}")
    return m.group("prefix")


def load(root: Path, wm: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(str(root / wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def resolve_scale(
    dst_idx: dict[str, str],
    weight_name: str,
    scale_cache: dict[str, torch.Tensor],
    dst_root: Path,
) -> tuple[torch.Tensor, str]:
    """Return (scale_tensor, mode) for one shard weight."""
    per_channel = weight_name[: -len(".weight")] + ".weight_scale"
    if per_channel in dst_idx:
        return load(dst_root, dst_idx, per_channel).float(), "per_channel"

    prefix = ngram_module_prefix(weight_name)
    per_tensor = prefix + ".weight_scale"
    if per_tensor not in dst_idx:
        raise KeyError(
            f"missing scale for {weight_name}: expected {per_channel} "
            f"or {per_tensor}"
        )
    if per_tensor not in scale_cache:
        scale_cache[per_tensor] = load(dst_root, dst_idx, per_tensor).float()
    return scale_cache[per_tensor], "per_tensor"


def main(src_root: Path, dst_root: Path) -> None:
    src_idx = json.load(open(src_root / "model.safetensors.index.json"))["weight_map"]
    dst_idx = json.load(open(dst_root / "model.safetensors.index.json"))["weight_map"]

    names = sorted(n for n in dst_idx if is_ngram_shard_weight(n))
    if not names:
        raise RuntimeError("No ngram_embedding.shard_*.weight found in destination.")

    print(f"Evaluating {len(names)} ngram_embedding shards")
    print(f"src={src_root}")
    print(f"dst={dst_root}")
    print("-" * 80)

    scale_cache: dict[str, torch.Tensor] = {}
    rows: list[tuple[str, float, float, str]] = []
    modes: set[str] = set()

    for i, name in enumerate(names, 1):
        if name not in src_idx:
            raise KeyError(f"missing in source index: {name}")

        w_bf16 = load(src_root, src_idx, name).float()
        w_q = load(dst_root, dst_idx, name)
        scale, mode = resolve_scale(dst_idx, name, scale_cache, dst_root)
        modes.add(mode)

        if w_q.dtype != torch.int8:
            raise TypeError(f"{name}: expected int8, got {w_q.dtype}")
        if w_bf16.shape != w_q.shape:
            raise ValueError(
                f"{name}: shape mismatch bf16={tuple(w_bf16.shape)} "
                f"int8={tuple(w_q.shape)}"
            )

        recon = w_q.float() * scale
        mse = torch.mean((w_bf16 - recon) ** 2).item()
        rel = (torch.norm(w_bf16 - recon) / torch.norm(w_bf16).clamp_min(1e-12)).item()
        rows.append((name, mse, rel, mode))
        print(
            f"[{i}/{len(names)}] {name}  mode={mode}  "
            f"mse={mse:.6e}  rel_l2={rel:.6e}"
        )
        del w_bf16, w_q, recon

    mses = torch.tensor([r[1] for r in rows], dtype=torch.float64)
    rels = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    worst = max(rows, key=lambda x: x[2])

    print("-" * 80)
    print(f"count={len(rows)}  scale_mode(s)={sorted(modes)}")
    print(f"mse  mean={mses.mean().item():.6e}  max={mses.max().item():.6e}")
    print(f"rel  mean={rels.mean().item():.6e}  max={rels.max().item():.6e}")
    print(
        f"worst_rel_l2: {worst[0]}  mode={worst[3]}  "
        f"mse={worst[1]:.6e}  rel_l2={worst[2]:.6e}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="MSE: BF16 vs INT8 ngram_embedding reconstruction"
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = parser.parse_args()
    main(args.src, args.dst)
