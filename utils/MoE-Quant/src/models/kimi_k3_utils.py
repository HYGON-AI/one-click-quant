"""Structural and expert-parallel utilities for Kimi-K3 calibration."""

import inspect
import os
import sys
import types

import torch
import torch.distributed as dist
from torch import nn


_PATCHED_CLASSES = set()
_PACKING_MODE = False


def set_packing_mode(enabled: bool) -> None:
    global _PACKING_MODE
    _PACKING_MODE = enabled


def _patch_sparse_moe_class(sparse_moe_class) -> None:
    if sparse_moe_class in _PATCHED_CLASSES:
        return
    modeling = inspect.getmodule(sparse_moe_class)
    if modeling is None:
        raise RuntimeError("Cannot locate Kimi modeling module for EP patching.")
    mlp_class = modeling.KimiBlockSparseMLP
    gate_class = modeling.KimiMoEGate
    shared_class = modeling.KimiMLP
    rms_class = modeling.KimiRMSNorm

    def ep_init(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize
        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = (
            config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        )
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        ep_size, ep_rank, start, end = _expert_range(config.num_experts)
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_per_rank = end - start
        self.experts_start_idx = start
        self.experts_end_idx = end
        self.experts = nn.ModuleList(
            [
                mlp_class(
                    config,
                    hidden_size=self.moe_hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                )
                if not _PACKING_MODE and start <= index < end
                else None
                for index in range(config.num_experts)
            ]
        )
        self.gate = gate_class(config)
        if config.num_shared_experts is not None:
            self.shared_experts = shared_class(
                config=config,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
            )
        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size, self.moe_hidden_size, bias=False
            )
            self.routed_expert_up_proj = nn.Linear(
                self.moe_hidden_size, config.hidden_size, bias=False
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm = rms_class(
                    self.moe_hidden_size, eps=config.rms_norm_eps
                )

    sparse_moe_class.__init__ = ep_init
    sparse_moe_class.moe_infer = _moe_infer_local_experts
    sparse_moe_class.forward = _moe_forward_ep
    _PATCHED_CLASSES.add(sparse_moe_class)


def patch_kimi_k3_for_ep(config) -> None:
    """Import Kimi remote classes and patch MoE construction before from_config."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_path = getattr(config, "_name_or_path", None)
    if not model_path:
        raise ValueError("Kimi-K3 config does not contain its source model path.")
    model_class = get_class_from_dynamic_module(
        "modeling_kimi_k3.KimiK3ForConditionalGeneration",
        model_path,
    )
    modeling = inspect.getmodule(model_class)
    if modeling is None:
        raise RuntimeError("Cannot locate Kimi-K3 remote modeling module.")
    text_modeling = inspect.getmodule(modeling.KimiLinearForCausalLM)
    if text_modeling is None:
        raise RuntimeError("Cannot locate Kimi-K3 text modeling module.")
    _patch_sparse_moe_class(text_modeling.KimiSparseMoeBlock)


def _expert_range(num_experts: int) -> tuple[int, int, int, int]:
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
        return 1, 0, 0, num_experts
    world_size = dist.get_world_size()
    if num_experts % world_size != 0:
        raise ValueError(
            f"Kimi num_experts ({num_experts}) must be divisible by "
            f"the distributed world size ({world_size})."
        )
    rank = dist.get_rank()
    experts_per_rank = num_experts // world_size
    start = rank * experts_per_rank
    return world_size, rank, start, start + experts_per_rank


def _gather_variable_sequence(hidden_states: torch.Tensor):
    """Gather [batch, sequence, hidden] tensors with rank-varying sequence lengths."""
    world_size = dist.get_world_size()
    local_shape = torch.tensor(
        hidden_states.shape, dtype=torch.long, device=hidden_states.device
    )
    shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
    dist.all_gather(shapes, local_shape)
    shape_values = [tuple(int(value) for value in shape.tolist()) for shape in shapes]
    batch_sizes = {shape[0] for shape in shape_values}
    hidden_sizes = {shape[2] for shape in shape_values}
    if len(batch_sizes) != 1 or len(hidden_sizes) != 1:
        raise ValueError(
            "Kimi EP calibration requires equal batch and hidden dimensions across "
            f"ranks, got {shape_values}."
        )
    max_sequence_length = max(shape[1] for shape in shape_values)
    if hidden_states.shape[1] < max_sequence_length:
        padded = hidden_states.new_zeros(
            hidden_states.shape[0], max_sequence_length, hidden_states.shape[2]
        )
        padded[:, : hidden_states.shape[1]] = hidden_states
    else:
        padded = hidden_states
    gathered_padded = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered_padded, padded.contiguous())
    sequence_lengths = [shape[1] for shape in shape_values]
    gathered = [
        tensor[:, :sequence_length]
        for tensor, sequence_length in zip(gathered_padded, sequence_lengths)
    ]
    result = torch.cat(gathered, dim=1)
    return result, sequence_lengths


def _moe_infer_local_experts(self, x, topk_ids, topk_weight):
    output = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
    for expert_id in range(self.experts_start_idx, self.experts_end_idx):
        expert = self.experts[expert_id]
        if expert is None:
            continue
        token_idx, topk_idx = torch.where(topk_ids == expert_id)
        if token_idx.numel() == 0:
            continue
        expert_input = x[token_idx]
        expert_output = expert(expert_input)
        weighted_output = (
            expert_output.to(torch.float32)
            * topk_weight[token_idx, topk_idx].unsqueeze(-1).to(torch.float32)
        )
        output[token_idx] += weighted_output
    result = output.to(x.dtype)
    return result


def _moe_forward_ep(self, hidden_states):
    if self.training:
        raise NotImplementedError("Training mode is not supported in KimiSparseMoeBlock")

    use_ep = self.ep_size > 1
    if use_ep:
        local_sequence_length = hidden_states.shape[1]
        hidden_states, sequence_lengths = _gather_variable_sequence(hidden_states)
        start = sum(sequence_lengths[: self.ep_rank])
        end = start + local_sequence_length
    else:
        start, end = 0, hidden_states.shape[1]

    identity = hidden_states
    original_shape = hidden_states.shape
    topk_ids, topk_weight = self.gate(hidden_states)
    routed_input = hidden_states.reshape(-1, hidden_states.shape[-1])
    if self.use_latent_moe:
        routed_input = self.routed_expert_down_proj(routed_input)
    routed_output = self.moe_infer(routed_input, topk_ids, topk_weight)
    if use_ep:
        dist.all_reduce(routed_output)
    if self.use_latent_moe:
        if self.latent_moe_use_norm:
            routed_output = self.routed_expert_norm(routed_output)
        routed_output = self.routed_expert_up_proj(routed_output)
    output = routed_output.view(*original_shape)
    if self.config.num_shared_experts is not None:
        output = output + self.shared_experts(identity)
    result = output[:, start:end, :] if use_ep else output
    return result


def _prepare_sparse_moe_block(block) -> None:
    ep_size, ep_rank, start, end = _expert_range(block.num_experts)
    if not hasattr(block, "experts_start_idx"):
        experts = list(block.experts)
        block.experts = nn.ModuleList(
            [expert if start <= index < end else None for index, expert in enumerate(experts)]
        )
    block.ep_size = ep_size
    block.ep_rank = ep_rank
    block.experts_per_rank = end - start
    block.experts_start_idx = start
    block.experts_end_idx = end
    block.moe_infer = types.MethodType(_moe_infer_local_experts, block)
    block.forward = types.MethodType(_moe_forward_ep, block)


def prepare_kimi_k3_model(model, config, torch_dtype: torch.dtype) -> None:
    """Convert Kimi MoE blocks to rank-local experts while preserving key names."""
    del torch_dtype
    layers = model.language_model.model.layers
    local_range = None
    for layer in layers:
        sparse_moe = getattr(layer, "block_sparse_moe", None)
        if sparse_moe is None:
            continue
        _prepare_sparse_moe_block(sparse_moe)
        local_range = (
            sparse_moe.ep_rank,
            sparse_moe.ep_size,
            sparse_moe.experts_start_idx,
            sparse_moe.experts_end_idx,
        )
    if local_range is not None:
        rank, size, start, end = local_range
        if rank == 0:
            print(
                f"[INFO] Kimi-K3 expert parallelism size={size}, "
                f"experts_per_rank={end - start}"
            )
        print(f"[INFO] rank={rank} Kimi-K3 expert range=[{start}, {end})")
