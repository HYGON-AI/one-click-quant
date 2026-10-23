"""Shared safetensors shard helpers for multi-process unpack pipelines."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Set

from safetensors import safe_open
from safetensors.torch import save_file

UNPACK_WM_RANK = ".unpack_wm_rank{}.json"


def write_safetensors(path: Path, tensors: Dict[str, Any]) -> None:
    """Write one safetensors file; allow empty (e.g. scale-only source shards)."""
    if tensors:
        save_file(tensors, str(path))
        return
    # Valid empty archive: 8-byte header length + 8-byte-aligned JSON "{}".
    meta = b"{}"
    pad = (8 - len(meta) % 8) % 8
    meta = meta + b" " * pad
    path.write_bytes(struct.pack("<Q", len(meta)) + meta)


def shard_keys_complete(dst_shard: Path, expected: Set[str]) -> bool:
    """Return True if ``dst_shard`` exists and its keys equal ``expected``.

    Empty ``expected`` only requires the file to exist (scale-only / empty shards).
    """
    if not expected:
        return dst_shard.is_file()
    if not dst_shard.is_file():
        return False
    try:
        with safe_open(dst_shard, framework="pt", device="cpu") as f:
            have = set(f.keys())
    except Exception:
        return False
    return have == expected


def cleanup_legacy_u_parts(dst: Path, stem: str) -> None:
    """Remove legacy ``{stem}.uXXX.safetensors`` split parts under ``dst``."""
    for p in dst.glob(f"{stem}.u*.safetensors"):
        p.unlink(missing_ok=True)


def clear_rank_manifests(dst: Path, *, max_ranks: int = 64) -> None:
    """Delete stale per-rank weight-map manifests under ``dst``."""
    for r in range(max_ranks):
        p = dst / UNPACK_WM_RANK.format(r)
        if p.is_file():
            p.unlink(missing_ok=True)


def write_rank_manifest(
    dst: Path,
    rank: int,
    *,
    weight_map: Dict[str, str],
    unpacked_modules: int,
    shards_done: Iterable[str],
) -> Path:
    """Write ``.unpack_wm_rank{rank}.json`` and return its path."""
    path = dst / UNPACK_WM_RANK.format(rank)
    path.write_text(
        json.dumps(
            {
                "rank": rank,
                "weight_map": weight_map,
                "unpacked_modules": unpacked_modules,
                "shards_done": list(shards_done),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_and_merge_rank_manifests(
    dst: Path,
    world_size: int,
    *,
    missing_label: str = "unpack",
) -> tuple[Dict[str, str], int]:
    """Load rank manifests, merge weight maps, delete manifests.

    Returns ``(merged_weight_map, total_unpacked_modules)``.
    """
    new_wm: Dict[str, str] = {}
    total_unpacked = 0
    for r in range(world_size):
        manifest = dst / UNPACK_WM_RANK.format(r)
        if not manifest.is_file():
            raise FileNotFoundError(
                f"{missing_label}: missing rank manifest {manifest}"
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        new_wm.update(payload.get("weight_map") or {})
        total_unpacked += int(payload.get("unpacked_modules") or 0)
        manifest.unlink(missing_ok=True)
    return new_wm, total_unpacked
