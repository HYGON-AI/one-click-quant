# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Differentiable EP communication for the MiMo calibration model.

The shared MoE-Quant forward remains unchanged. This adapter is installed only
on the AutoQuantize model and retains its eval-mode routing and FP32 expert sum.
"""

from __future__ import annotations

import types

import torch
import torch.distributed as dist


def _world_size(group=None):
    return dist.get_world_size(group) if dist.is_initialized() else 1


class _GatherTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, flat, sizes, group):
        ctx.sizes, ctx.group = sizes, group
        ctx.rank = dist.get_rank(group)
        padded = flat.new_zeros(max(sizes), flat.shape[1])
        padded[: flat.shape[0]].copy_(flat)
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded, group=group)
        return torch.cat([part[:size] for part, size in zip(gathered, sizes)], dim=0)

    @staticmethod
    def backward(ctx, gradient):
        # All ranks consume the gathered inputs, so their input-gradient
        # contributions must be summed before returning the owner's slice.
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=ctx.group)
        start = sum(ctx.sizes[: ctx.rank])
        return gradient.narrow(0, start, ctx.sizes[ctx.rank]).contiguous(), None, None


def gather_variable_tokens(hidden_states, group=None):
    """Gather rank-local token rows while preserving their activation gradients."""
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if _world_size(group) == 1:
        return flat, (flat.shape[0],)
    shape = torch.tensor(flat.shape, dtype=torch.long, device=flat.device)
    shapes = [torch.empty_like(shape) for _ in range(_world_size(group))]
    dist.all_gather(shapes, shape, group=group)
    if any(int(item[1].item()) != flat.shape[1] for item in shapes):
        raise ValueError("MiMo EP ranks must have equal hidden dimensions")
    sizes = tuple(int(item[0].item()) for item in shapes)
    if not any(sizes):
        raise ValueError("MiMo EP requires at least one token across the process group")
    return _GatherTokens.apply(flat, sizes, group), sizes


class _SumExpertOutputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        output = value.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, gradient):
        # Downstream ranks select different token slices. Pool their gradients
        # so each expert owner receives the gradients for all of its tokens.
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=ctx.group)
        return gradient, None


def differentiable_all_reduce(value, group=None):
    """Sum expert contributions in both forward and backward."""
    return value if _world_size(group) == 1 else _SumExpertOutputs.apply(value, group)


def _differentiable_ep_forward(self, hidden_states):
    if self.ep_size == 1:
        return self._autoquantize_ep_original_forward(hidden_states)
    if self.training:
        raise ValueError("MiMo EP gradient scoring requires eval-mode routing")
    group = self._autoquantize_ep_group
    if _world_size(group) != self.ep_size or dist.get_rank(group) != self.ep_rank:
        raise ValueError("MiMo EP process group differs from the model expert partition")
    gathered, sizes = gather_variable_tokens(hidden_states, group)
    topk_indices, topk_weights = self.gate(gathered.unsqueeze(0))
    partial = self._autoquantize_ep_local_compute(gathered, topk_indices, topk_weights)
    # A rank with no routed tokens must still enter both backward collectives.
    # Do not rely on trainable weights: analysis freezes all source parameters.
    partial = partial + gathered.float().sum() * 0.0
    output = differentiable_all_reduce(partial, group)
    start = sum(sizes[: self.ep_rank])
    return output.narrow(0, start, sizes[self.ep_rank]).to(hidden_states.dtype).reshape(hidden_states.shape)


def install_ep_forward(model, group=None):
    """Install the differentiable forward on already partitioned MiMo MoE blocks.

    Call after constructing rank-local experts, before ModelOpt registers score hooks.
    """

    installed = 0
    for module in model.modules():
        if not all(hasattr(module, attr) for attr in (
            "ep_size", "ep_rank", "experts_start_idx", "experts_end_idx", "experts", "gate"
        )):
            continue
        if hasattr(module, "_autoquantize_ep_original_forward"):
            continue
        module._autoquantize_ep_original_forward = module.forward
        module._autoquantize_ep_group = group
        module._autoquantize_ep_local_compute = types.MethodType(_local_moe_float32, module)
        module.forward = types.MethodType(_differentiable_ep_forward, module)
        installed += 1
    return installed


def _local_moe_float32(self, hidden_states, topk_indices, topk_weights):
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert_id in range(self.experts_start_idx, self.experts_end_idx):
        tokens, slots = torch.where(topk_indices == expert_id)
        if tokens.numel():
            values = self.experts[expert_id](hidden_states[tokens]).float()
            output.index_add_(0, tokens, values * topk_weights[tokens, slots].float().unsqueeze(-1))
    return output
