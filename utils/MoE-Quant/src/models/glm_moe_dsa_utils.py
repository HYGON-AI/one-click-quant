# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GLM-MoE-DSA structural and checkpoint utilities.

The upstream GLM implementation packs routed experts into 3-D parameters, while
MoE-Quant needs ordinary 2-D ``nn.Linear`` modules so it can collect one GPTQ
Hessian per expert projection.  This module keeps the GLM router/MLP semantics
but exposes the checkpoint's per-expert layout directly.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .. import quant_utils


_LAYER_78_PREFIX = "model.layers.78."


def _import_glm_upstream():
    """Import the Transformers GLM-MoE-DSA classes lazily.

    Kept out of module import so tests and packing utilities that only touch
    checkpoint tensors do not require the full modeling module to be present.
    """
    from transformers.activations import ACT2FN
    from transformers.masking_utils import create_causal_mask
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaMLP,
        GlmMoeDsaMoE,
        GlmMoeDsaTopkRouter,
    )

    return {
        "ACT2FN": ACT2FN,
        "create_causal_mask": create_causal_mask,
        "GlmMoeDsaMLP": GlmMoeDsaMLP,
        "GlmMoeDsaMoE": GlmMoeDsaMoE,
        "GlmMoeDsaTopkRouter": GlmMoeDsaTopkRouter,
    }


def _num_experts(config: Any) -> int:
    value = getattr(config, "n_routed_experts", None)
    if value is None:
        value = getattr(config, "num_local_experts", None)
    if value is None:
        raise AttributeError("GLM config has neither n_routed_experts nor num_local_experts.")
    return int(value)


def _ep_context(config: Any) -> tuple[int, int, int, int]:
    """Return ``(ep_size, rank, start, end)`` for the current process."""
    ep_size = int(getattr(config, "ep_size", 1) or 1)
    if ep_size > 1:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("GLM expert parallelism requires initialized torch.distributed.")
        if dist.get_world_size() != ep_size:
            raise RuntimeError(
                f"config.ep_size ({ep_size}) must equal distributed world size "
                f"({dist.get_world_size()})."
            )
        rank = dist.get_rank()
    else:
        rank = 0
    num_experts = _num_experts(config)
    if num_experts % ep_size != 0:
        raise ValueError(f"n_routed_experts ({num_experts}) must be divisible by ep_size ({ep_size}).")
    per_rank = num_experts // ep_size
    if per_rank == 0:
        raise ValueError(
            f"ep_size ({ep_size}) exceeds n_routed_experts ({num_experts}); "
            "every rank must own at least one routed expert."
        )
    return ep_size, rank, rank * per_rank, (rank + 1) * per_rank


class GlmMoeDsaExpertMLP(nn.Module):
    """One GLM routed expert with checkpoint-compatible projection names."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn, dtype: torch.dtype):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype)
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class GlmMoeDsaSparseMoeBlock(nn.Module):
    """GLM sparse MoE with individually addressable routed experts.

    ``None`` entries are intentional under EP: the ModuleList preserves global
    expert numbering while state_dict contains only this rank's expert tensors.
    """

    def __init__(self, config: Any, dtype: torch.dtype):
        super().__init__()
        upstream = _import_glm_upstream()
        self.config = config
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_dim = int(config.moe_intermediate_size)
        self.num_experts = _num_experts(config)
        self.act_fn = upstream["ACT2FN"][config.hidden_act]

        self.ep_size, self.ep_rank, self.experts_start_idx, self.experts_end_idx = _ep_context(config)
        self.experts_per_rank = self.experts_end_idx - self.experts_start_idx

        # The router computes logits and score correction in FP32, but the
        # checkpoint's router weight is a normal model-dtype tensor.
        self.gate = upstream["GlmMoeDsaTopkRouter"](config)
        self.gate.weight = nn.Parameter(
            torch.empty((self.num_experts, self.hidden_dim), dtype=dtype)
        )
        self.shared_experts = upstream["GlmMoeDsaMLP"](
            config=config,
            intermediate_size=self.intermediate_dim * int(config.n_shared_experts),
        )
        self.shared_experts.to(dtype=dtype)

        self.experts = nn.ModuleList(
            [
                (
                    GlmMoeDsaExpertMLP(
                        self.hidden_dim,
                        self.intermediate_dim,
                        self.act_fn,
                        dtype,
                    )
                    if self.experts_start_idx <= expert_idx < self.experts_end_idx
                    else None
                )
                for expert_idx in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        residual = hidden_states
        _, topk_weights, topk_indices = self.gate(hidden_states)
        flat_hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        if self.ep_size == 1:
            routed = self._dispatch_local(flat_hidden_states, topk_indices, topk_weights)
        else:
            routed = self._dispatch_ep(flat_hidden_states, topk_indices, topk_weights)
        shared = self.shared_experts(residual.reshape(-1, self.hidden_dim))
        return (routed + shared).reshape(original_shape)

    def _dispatch_local(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        # Routing indices are data-dependent; only visit experts hit by this
        # calibration batch so hooks collect the same inputs as real inference.
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=False).flatten()
        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor)
            expert = self.experts[expert_idx]
            if expert is None:
                raise RuntimeError(f"Local dispatch selected non-local expert {expert_idx}.")
            topk_pos, token_idx = torch.where(expert_mask[expert_idx])
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
            raise RuntimeError("EP dispatch requires initialized torch.distributed.")

        # Sort token/expert pairs by global expert ID.
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

        tokens_per_rank = tokens_per_expert.reshape(self.ep_size, self.experts_per_rank).sum(dim=1)
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

        # The receiving rank gets each sender's experts in rank-major order.
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
                raise RuntimeError(f"Missing local expert {self.experts_start_idx + local_idx}.")
            # Router weights are FP32 (GLM router runs in FP32); expert output is
            # BF16. Cast the weighted result back to the destination dtype before
            # index_put, matching _dispatch_local.
            local_output[positions] = (
                expert(received_tokens[positions]) * received_weights[positions, None]
            ).to(local_output.dtype)

        returned = sorted_tokens.new_empty(sorted_tokens.shape)
        dist.all_to_all(
            list(returned.split(input_splits)),
            list(local_output.split(output_splits)),
        )
        # Undo the original pair sort and aggregate top-k choices per token.
        unsorted = torch.empty_like(returned)
        unsorted[sorted_pair_indices] = returned
        return unsorted.reshape(*topk_indices.shape, -1).sum(dim=1).to(hidden_states.dtype)


def _is_packed_glm_moe(mlp: nn.Module) -> bool:
    experts = getattr(mlp, "experts", None)
    return experts is not None and hasattr(experts, "gate_up_proj")


def prepare_glm_moe_dsa_model(model: nn.Module, config: Any, dtype: torch.dtype) -> None:
    """Replace packed GLM sparse MLPs before any checkpoint tensors are loaded."""
    layer_types = list(getattr(config, "mlp_layer_types", []))
    for layer_idx, block in enumerate(model.model.layers):
        is_sparse = layer_idx < len(layer_types) and layer_types[layer_idx] == "sparse"
        if is_sparse and _is_packed_glm_moe(block.mlp):
            block.mlp = GlmMoeDsaSparseMoeBlock(config, dtype)

    for layer_idx, block in enumerate(model.model.layers):
        if layer_idx < len(layer_types) and layer_types[layer_idx] == "sparse":
            if _is_packed_glm_moe(block.mlp):
                raise AssertionError(f"Layer {layer_idx} still contains packed GLM expert parameters.")
            # Every EP rank must own at least one routed expert; walk the
            # ModuleList structurally so ranks with experts_start_idx > 0 pass.
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


def compute_position_embeddings(
    config: Any,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GLM's rotary tensors without moving the root model to the GPU."""
    rope_parameters = getattr(config, "rope_parameters", None) or {
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "rope_type": "default",
    }
    theta = float(rope_parameters["rope_theta"])
    dim = int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=hidden_states.device) / dim)
    )
    expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    positions = position_ids[:, None, :].to(device=hidden_states.device, dtype=torch.float32)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        freqs = (expanded @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)


def create_glm_causal_mask(
    config: Any,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    upstream = _import_glm_upstream()
    return upstream["create_causal_mask"](
        config=config,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )


def count_extra_shards(weight_map: Dict[str, str]) -> int:
    """Count source shards containing the ignored next-n layer 78."""
    return len({
        shard for key, shard in weight_map.items() if key.startswith(_LAYER_78_PREFIX)
    })


def _load_extra_tensors(weight_dir: str, weight_map: Dict[str, str]) -> Iterable[tuple[str, Dict[str, torch.Tensor]]]:
    extra_keys = {key for key in weight_map if key.startswith(_LAYER_78_PREFIX)}
    for shard in sorted({weight_map[key] for key in extra_keys}):
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(os.path.join(weight_dir, shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in extra_keys:
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
    """Copy/dequantize layer-78 next-n tensors into dedicated output shards."""
    for source_shard, tensors in _load_extra_tensors(weight_dir, weight_map):
        packed_keys = sorted(key for key in tensors if key.endswith(".weight_packed"))
        if packed_keys:
            raise ValueError(
                f"GLM extra shard {source_shard} already contains packed tensor {packed_keys[0]}."
            )
        if not quant_utils.can_dequantize_from_fp8(tensors):
            raise RuntimeError(
                f"GLM extra shard {source_shard} has an FP8 tensor without matching weight_scale_inv."
            )
        quant_utils.dequantize_state_dict(tensors, torch.bfloat16)
        for key, tensor in list(tensors.items()):
            if tensor.is_floating_point() and tensor.dtype not in (torch.float32, torch.bfloat16):
                tensors[key] = tensor.to(torch.bfloat16)
        output_name = f"model-{next_shard_id:05}-of-{num_output_shards:05}.safetensors"
        save_file(tensors, os.path.join(packed_model_path, output_name))
        for key in tensors:
            safetensors_index[key] = output_name
        next_shard_id += 1
        del tensors
        gc.collect()
    return next_shard_id


__all__ = [
    "GlmMoeDsaExpertMLP",
    "GlmMoeDsaSparseMoeBlock",
    "compute_position_embeddings",
    "create_glm_causal_mask",
    "prepare_glm_moe_dsa_model",
    "count_extra_shards",
    "save_extra_weights",
]
