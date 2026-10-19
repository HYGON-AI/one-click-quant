"""Deprecated: use ``preprocess.adapters.kimi_k3`` / ``ADAPTER_REGISTRY``.

Kept as thin re-exports so older imports keep working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from preprocess.adapters.kimi_k3 import KimiK3Adapter

_adapter = KimiK3Adapter()


def build_ln_fuse_map(cfg: Dict[str, Any], num_layers: Optional[int] = None) -> Dict[str, List[str]]:
    return _adapter.build_ln_fuse_map(cfg, num_layers=num_layers)


def build_rotate_pairs(
    cfg: Dict[str, Any],
    *,
    block_size: int = 32,
    num_layers: Optional[int] = None,
    num_experts: Optional[int] = None,
) -> Tuple[List[Tuple[Dict[str, str], Dict[str, str]]], Dict[str, Any]]:
    return _adapter.build_rotate_pairs(
        cfg, block_size=block_size, num_layers=num_layers, num_experts=num_experts
    )


__all__ = ["build_ln_fuse_map", "build_rotate_pairs"]
