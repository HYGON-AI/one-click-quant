"""Optional activation stats for Flex Smooth / AWQ-SSZ (light calib).

Default Flex transforms use a weight-side act proxy when ``use_calib_acts`` is
false. For true Flex alignment, precompute per-module act tensors and pass them
via ``--act-stats path.pt`` (a ``dict[str, Tensor]`` saved with ``torch.save``),
and set ``use_calib_acts: true`` in the recipe.

Collection sample for Kimi-K3 (hooks + calib forward → acts.pt)::

    preprocess/collect_acts_kimi_k3.py

This module only **loads** that file; it does not run model forward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from loguru import logger


def load_act_stats(path: Optional[str | Path]) -> Dict[str, torch.Tensor]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"act-stats file not found: {p}")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"act-stats must be a dict[str, Tensor], got {type(obj)}")
    out: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"act-stats[{k!r}] is not a Tensor")
        out[str(k)] = v.float().cpu()
    logger.info("loaded act stats for {} modules from {}", len(out), p)
    return out
