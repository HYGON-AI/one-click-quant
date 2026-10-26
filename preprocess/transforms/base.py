"""Transform interface for preprocess pipeline."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

import torch

if TYPE_CHECKING:
    from preprocess.adapters.base import ModelAdapter
    from preprocess.io import WeightStore

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def module_layer_id(name: str) -> Optional[int]:
    """Return ``layers.{i}`` index from a module or tensor name, else None."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


@dataclass
class PreprocessContext:
    """Shared state across transforms."""

    model_dir: str
    config: Dict[str, Any]
    recipe: Dict[str, Any]
    device: torch.device
    adapter: Optional["ModelAdapter"] = None
    # Optional activation cache: module_name -> [N, K] float tensor
    act_stats: Dict[str, torch.Tensor] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    # Multi-GPU layer shard: None = own all modules (single process).
    layer_ids: Optional[Set[int]] = None
    rank: int = 0
    world_size: int = 1

    @property
    def text_config(self) -> Dict[str, Any]:
        tc = self.config.get("text_config")
        return tc if isinstance(tc, dict) else self.config

    def require_adapter(self) -> "ModelAdapter":
        if self.adapter is None:
            raise RuntimeError(
                "PreprocessContext.adapter is None; pipeline must load an adapter "
                "from recipe (`adapter: kimi_k3`, ...)."
            )
        return self.adapter

    def owns_module(self, name: str) -> bool:
        """Whether this rank should mutate ``name`` (module or tensor path).

        - ``layer_ids is None``: single-process, own everything.
        - Names under ``layers.{i}``: owned iff ``i in layer_ids``.
        - Non-layer names (embed, lm_head, ...): owned only by rank 0.
        """
        if self.layer_ids is None:
            return True
        lid = module_layer_id(name)
        if lid is None:
            return self.rank == 0
        return lid in self.layer_ids


class PreprocessTransform(ABC):
    """One preprocess step; mutates ``WeightStore`` in place."""

    name: str = "base"

    @abstractmethod
    def apply(self, weights: "WeightStore", ctx: PreprocessContext) -> None:
        raise NotImplementedError
