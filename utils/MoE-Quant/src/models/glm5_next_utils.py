# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (glm5_next) structural utilities for MoE-Quant.

The upstream Glm5NextTextExperts stores every routed expert as a slice of two
3-D ``nn.Parameter`` tensors (``experts.gate_up_proj`` and ``experts.down_proj``).
MoE-Quant needs one ``nn.Linear`` per expert per projection so it can attach one
GPTQ Hessian per weight matrix; this module rebuilds the sparse block that way
and forwards the checkpoint-compatible per-expert layout.

Layer 45 in a GLM-5.3-Flash checkpoint holds the MTP (multi-token predict) head,
which the Transformers modeling code intentionally drops on load
(``_keys_to_ignore_on_load_unexpected = [r"layers\\.45\\."]``). The main
calibration + packing pipeline mirrors that: layer 45 tensors bypass GPTQ and
are copied through as bf16 in dedicated extra shards.
"""

from __future__ import annotations

import gc
import os
import re
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .. import quant_utils


_MTP_LAYER_PREFIX = "model.language_model.layers.45."
_VISION_PREFIX = "model.visual."


def _import_glm5_next_upstream():
    """Import the Transformers Glm5Next classes lazily."""
    from transformers.activations import ACT2FN
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextMLP,
        Glm5NextTextMoE,
        Glm5NextTextTopkRouter,
    )

    return {
        "ACT2FN": ACT2FN,
        "Glm5NextTextMLP": Glm5NextTextMLP,
        "Glm5NextTextMoE": Glm5NextTextMoE,
        "Glm5NextTextTopkRouter": Glm5NextTextTopkRouter,
    }


def _text_config(config: Any) -> Any:
    text = getattr(config, "text_config", None)
    if text is None:
        raise AttributeError(
            "GLM-5.3-Flash config is missing text_config; expected a multimodal Glm5NextConfig."
        )
    return text


def _num_experts(config: Any) -> int:
    text = _text_config(config)
    value = getattr(text, "n_routed_experts", None)
    if value is None:
        value = getattr(text, "num_local_experts", None)
    if value is None:
        raise AttributeError(
            "Glm5NextTextConfig has neither n_routed_experts nor num_local_experts."
        )
    return int(value)


def _ep_context(config: Any) -> Tuple[int, int, int, int]:
    """Return ``(ep_size, rank, start, end)`` for the current process."""
    text = _text_config(config)
    ep_size = int(getattr(text, "ep_size", 1) or 1)
    if ep_size > 1:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "GLM-5.3-Flash expert parallelism requires initialized torch.distributed."
            )
        if dist.get_world_size() != ep_size:
            raise RuntimeError(
                f"text_config.ep_size ({ep_size}) must equal distributed world size "
                f"({dist.get_world_size()})."
            )
        rank = dist.get_rank()
    else:
        rank = 0
    num_experts = _num_experts(config)
    if num_experts % ep_size != 0:
        raise ValueError(
            f"n_routed_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
        )
    per_rank = num_experts // ep_size
    if per_rank == 0:
        raise ValueError(
            f"ep_size ({ep_size}) exceeds n_routed_experts ({num_experts}); "
            "every rank must own at least one routed expert."
        )
    return ep_size, rank, rank * per_rank, (rank + 1) * per_rank


class Glm5NextExpertMLP(nn.Module):
    """One GLM-5.3-Flash routed expert with checkpoint-compatible names.

    The forward matches the upstream Glm5NextTextExperts SwiGLU path: SiLU on the
    (clamped) gate projection times the (clamped) up projection, then down_proj.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        swiglu_limit: float,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class Glm5NextSparseMoeBlock(nn.Module):
    """GLM-5.3-Flash sparse MoE with individually addressable routed experts.

    ``None`` entries in ``experts`` are intentional under EP: the ModuleList
    preserves global expert numbering so upstream router indices are valid, but
    the state_dict only contains the tensors this rank actually owns.
    """

    def __init__(self, config: Any, dtype: torch.dtype):
        super().__init__()
        upstream = _import_glm5_next_upstream()
        text_config = _text_config(config)
        self.config = text_config
        self.hidden_dim = int(text_config.hidden_size)
        self.intermediate_dim = int(text_config.moe_intermediate_size)
        self.num_experts = _num_experts(config)
        self.swiglu_limit = float(getattr(text_config, "swiglu_limit", 10.0))

        (
            self.ep_size,
            self.ep_rank,
            self.experts_start_idx,
            self.experts_end_idx,
        ) = _ep_context(config)
        self.experts_per_rank = self.experts_end_idx - self.experts_start_idx

        # Router: keep the upstream class (FP32 sigmoid + score correction) so
        # topk selection matches inference exactly.
        self.gate = upstream["Glm5NextTextTopkRouter"](text_config)
        # e_score_correction_bias is an nn.Buffer on the upstream router and is
        # already registered; leave it alone. The router weight is instantiated
        # empty by the upstream __init__; keep it as-is so from_config wiring
        # (`init_empty_weights`) leaves a materializable meta tensor.

        # Shared expert (single, non-routed). Upstream uses Glm5NextTextMLP with
        # intermediate_size = moe_intermediate_size * n_shared_experts.
        self.shared_experts = upstream["Glm5NextTextMLP"](
            text_config,
            intermediate_size=self.intermediate_dim * int(text_config.n_shared_experts),
        )
        self.shared_experts.to(dtype=dtype)

        self.experts = nn.ModuleList(
            [
                (
                    Glm5NextExpertMLP(
                        self.hidden_dim,
                        self.intermediate_dim,
                        self.swiglu_limit,
                        dtype,
                    )
                    if self.experts_start_idx <= expert_idx < self.experts_end_idx
                    else None
                )
                for expert_idx in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        flat_hidden = hidden_states.reshape(-1, self.hidden_dim)
        if self.ep_size == 1:
            routed = self._dispatch_local(flat_hidden, topk_indices, topk_weights)
        else:
            routed = self._dispatch_ep(flat_hidden, topk_indices, topk_weights)
        shared = self.shared_experts(residual)
        return routed.view(*orig_shape) + shared

    def _dispatch_local(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        mask = F.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = (mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=False).flatten()
        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor)
            expert = self.experts[expert_idx]
            if expert is None:
                raise RuntimeError(
                    f"Local dispatch selected non-local expert {expert_idx}."
                )
            topk_pos, token_idx = torch.where(mask[expert_idx])
            current = expert(hidden_states[token_idx])
            current = current * topk_weights[token_idx, topk_pos, None]
            output.index_add_(0, token_idx, current.to(output.dtype))
        return output

    @torch.no_grad()
    def _dispatch_ep(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """All-to-all token dispatch matching the global expert numbering."""
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "EP dispatch requires initialized torch.distributed."
            )

        tokens_per_expert = topk_indices.new_zeros((self.num_experts,))
        tokens_per_expert.scatter_add_(
            0,
            topk_indices.reshape(-1),
            torch.ones_like(topk_indices.reshape(-1), dtype=tokens_per_expert.dtype),
        )
        sorted_pair_indices = topk_indices.reshape(-1).argsort()
        sorted_tokens = hidden_states[sorted_pair_indices // topk_indices.shape[-1]]
        sorted_weights = topk_weights.reshape(-1)[sorted_pair_indices]
        sorted_experts = topk_indices.reshape(-1)[sorted_pair_indices]

        tokens_per_rank = tokens_per_expert.reshape(
            self.ep_size, self.experts_per_rank
        ).sum(dim=1)
        received_counts = tokens_per_expert.new_empty(self.ep_size)
        dist.all_to_all_single(received_counts, tokens_per_rank)
        input_splits = tokens_per_rank.cpu().tolist()
        output_splits = received_counts.cpu().tolist()
        received_tokens = sorted_tokens.new_empty(
            int(received_counts.sum().item()), sorted_tokens.shape[-1]
        )
        dist.all_to_all(
            list(received_tokens.split(output_splits)),
            list(sorted_tokens.split(input_splits)),
        )
        received_weights = sorted_weights.new_empty(int(received_counts.sum().item()))
        dist.all_to_all(
            list(received_weights.split(output_splits)),
            list(sorted_weights.split(input_splits)),
        )
        received_experts = sorted_experts.new_empty(int(received_counts.sum().item()))
        dist.all_to_all(
            list(received_experts.split(output_splits)),
            list(sorted_experts.split(input_splits)),
        )
        local_expert_ids = received_experts - self.experts_start_idx
        local_output = torch.zeros_like(received_tokens)
        for local_idx in range(self.experts_per_rank):
            positions = (local_expert_ids == local_idx).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            expert = self.experts[self.experts_start_idx + local_idx]
            if expert is None:
                raise RuntimeError(
                    f"Missing local expert {self.experts_start_idx + local_idx}."
                )
            local_output[positions] = (
                expert(received_tokens[positions])
                * received_weights[positions, None]
            ).to(local_output.dtype)

        returned = sorted_tokens.new_empty(sorted_tokens.shape)
        dist.all_to_all(
            list(returned.split(input_splits)),
            list(local_output.split(output_splits)),
        )
        unsorted = torch.empty_like(returned)
        unsorted[sorted_pair_indices] = returned
        return unsorted.reshape(*topk_indices.shape, -1).sum(dim=1).to(
            hidden_states.dtype
        )


def _is_packed_glm5_next_moe(mlp: nn.Module) -> bool:
    experts = getattr(mlp, "experts", None)
    if experts is None:
        return False
    return isinstance(getattr(experts, "gate_up_proj", None), nn.Parameter)


def _language_layers(model: nn.Module):
    """Return the text-model decoder ModuleList (top → Glm5NextModel → Glm5NextTextModel)."""
    return model.model.language_model.layers


def prepare_glm5_next_model(model: nn.Module, config: Any, dtype: torch.dtype) -> None:
    """Replace packed sparse MoE blocks with per-expert nn.Linear structure.

    Runs before any checkpoint tensors are loaded; must be called under
    ``init_empty_weights()``. Every EP rank keeps the global expert numbering
    intact (with None placeholders for non-local experts).
    """
    text_config = _text_config(config)
    layer_types = list(getattr(text_config, "mlp_layer_types", []) or [])

    for layer_idx, block in enumerate(_language_layers(model)):
        is_sparse = (
            layer_idx < len(layer_types) and layer_types[layer_idx] == "sparse"
        )
        if is_sparse and _is_packed_glm5_next_moe(block.mlp):
            block.mlp = Glm5NextSparseMoeBlock(config, dtype)

    # Post-conditions.
    for layer_idx, block in enumerate(_language_layers(model)):
        if layer_idx >= len(layer_types) or layer_types[layer_idx] != "sparse":
            continue
        if _is_packed_glm5_next_moe(block.mlp):
            raise AssertionError(
                f"Layer {layer_idx} still contains packed Glm5NextTextExperts parameters."
            )
        local_experts = [e for e in block.mlp.experts if e is not None]
        if not local_experts:
            raise AssertionError(
                f"Layer {layer_idx} has no local routed experts on this rank; check ep_size."
            )
        first = local_experts[0]
        for projection in ("gate_proj", "up_proj", "down_proj"):
            proj_module = getattr(first, projection, None)
            if not isinstance(proj_module, nn.Linear):
                raise AssertionError(
                    f"Layer {layer_idx} routed expert is missing {projection} nn.Linear."
                )


# --------------------------------------------------------------------------- #
# Extra-shard helpers: pass through MTP layer 45 and the vision tower as bf16
# (dequantizing FP8 if necessary), completely bypassing GPTQ.
# --------------------------------------------------------------------------- #


_PASSTHROUGH_PREFIXES = (_MTP_LAYER_PREFIX, _VISION_PREFIX)


def _shards_for_prefixes(weight_map: Dict[str, str]) -> Dict[str, set[str]]:
    """Group weight-map keys by shard filename for the passthrough prefixes."""
    grouped: Dict[str, set[str]] = {}
    for key, shard in weight_map.items():
        if not key.startswith(_PASSTHROUGH_PREFIXES):
            continue
        grouped.setdefault(shard, set()).add(key)
    return grouped


def count_extra_shards(weight_map: Dict[str, str]) -> int:
    """Count source shards touched by MTP-layer-45 or visual passthrough keys."""
    return len(_shards_for_prefixes(weight_map))


def _load_extra_tensors(
    weight_dir: str, weight_map: Dict[str, str]
) -> Iterable[Tuple[str, Dict[str, torch.Tensor]]]:
    for shard, keys in sorted(_shards_for_prefixes(weight_map).items()):
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(
            os.path.join(weight_dir, shard), framework="pt", device="cpu"
        ) as handle:
            for key in handle.keys():
                if key in keys:
                    tensors[key] = handle.get_tensor(key)
                elif key.endswith("_scale_inv"):
                    base = key[: -len("_scale_inv")]
                    if base in keys:
                        tensors[key] = handle.get_tensor(key)
        if tensors:
            yield shard, tensors


def save_extra_weights(
    weight_dir: str,
    weight_map: Dict[str, str],
    packed_model_path: str,
    next_shard_id: int,
    num_output_shards: int,
    safetensors_index: Dict[str, str],
) -> int:
    """Dequantize + copy MTP (layer 45) and visual tensors into their own shards."""
    for source_shard, tensors in _load_extra_tensors(weight_dir, weight_map):
        packed_keys = sorted(
            key for key in tensors if key.endswith(".weight_packed")
        )
        if packed_keys:
            raise ValueError(
                f"GLM-5.3-Flash extra shard {source_shard} already contains packed "
                f"tensor {packed_keys[0]}."
            )
        if not quant_utils.can_dequantize_from_fp8(tensors):
            raise RuntimeError(
                f"GLM-5.3-Flash extra shard {source_shard} has an FP8 tensor "
                "without matching weight_scale_inv."
            )
        quant_utils.dequantize_state_dict(tensors, torch.bfloat16)
        for key, tensor in list(tensors.items()):
            if (
                tensor.is_floating_point()
                and tensor.dtype not in (torch.float32, torch.bfloat16)
            ):
                tensors[key] = tensor.to(torch.bfloat16)
        output_name = (
            f"model-{next_shard_id:05}-of-{num_output_shards:05}.safetensors"
        )
        save_file(tensors, os.path.join(packed_model_path, output_name))
        for key in tensors:
            safetensors_index[key] = output_name
        next_shard_id += 1
        del tensors
        gc.collect()
    return next_shard_id


__all__ = [
    "Glm5NextExpertMLP",
    "Glm5NextSparseMoeBlock",
    "count_extra_shards",
    "prepare_glm5_next_model",
    "save_extra_weights",
]
