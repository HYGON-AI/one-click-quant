"""Per-model adapter interface (rotate / LN map / subgraph resolution).

Transforms are model-agnostic; architecture-specific names live here
(per-model architecture glue for preprocess).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class ModelAdapter(ABC):
    """Architecture glue for preprocess.

    Required for QuaRot (LN fuse + rotate pairs) and for resolving
    NormLinear / UpDown subgraphs when Flex transforms run.
    """

    name: str = "base"

    @abstractmethod
    def build_ln_fuse_map(
        self,
        cfg: Dict[str, Any],
        num_layers: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """``norm_module -> [linear_module, ...]`` (no ``.weight`` suffix).

        Also the source of truth for Flex Smooth NormLinear completeness:
        every listed Linear must be scaled together with the Norm.
        """

    @abstractmethod
    def build_rotate_pairs(
        self,
        cfg: Dict[str, Any],
        *,
        block_size: int = 32,
        num_layers: Optional[int] = None,
        num_experts: Optional[int] = None,
    ) -> Tuple[List[Tuple[Dict[str, str], Dict[str, str]]], Dict[str, Any]]:
        """Return ``([(left_map, right_map), ...], rot_meta)``.

        Each map is ``module_name -> rot_key``; ``rot_meta[rot_key]`` describes
        how to build that rotation matrix (size / mode / block_size).
        """

    def resolve_up_down(self, up_module: str) -> Optional[str]:
        """Map an UpDown seed (e.g. ``...w3`` / ``...up_proj``) to its down partner.

        Return ``None`` if ``up_module`` is not an up-projection for this model.
        """
        return None

    def resolve_norm_linear(
        self,
        seed_module: str,
        cfg: Dict[str, Any],
        *,
        num_layers: Optional[int] = None,
    ) -> Optional[Tuple[str, List[str]]]:
        """Resolve ``(norm, all_linear_consumers)`` for a Flex Smooth seed.

        Default: look up ``build_ln_fuse_map`` (seed may be a Norm or any consumer).
        Override only if LN-fuse map and Smooth groups must differ.
        """
        ln_map = self.build_ln_fuse_map(cfg, num_layers=num_layers)
        if seed_module in ln_map:
            return seed_module, list(ln_map[seed_module])
        for norm, consumers in ln_map.items():
            if seed_module in consumers:
                return norm, list(consumers)
        return None

    def prepare_weights(self, weights: Any, ctx: Any) -> None:
        """Optional hook before transforms (e.g. MXFP4→BF16 unpack).

        Default: no-op. ``KimiK3Adapter`` unpacks ``weight_packed`` experts.
        """
        return

    def norm_affine(self) -> str:
        """``rms`` (gamma init ones, Smooth does ``gamma /= s``) or ``gemma`` (``1+gamma``)."""
        return "rms"

    def hidden_smooth_partners(self, norm_module: str, cfg: Dict[str, Any]) -> List[str]:
        """Extra hidden-size linears that must share a tied hyper-connection Smooth scale.

        Used when a Gemma norm of size ``hc_count * hidden`` feeds both wide
        (10240-d) linears and the mean-mixed hidden consumers. Default: none.
        """
        return []
