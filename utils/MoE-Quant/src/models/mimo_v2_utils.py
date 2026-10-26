# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 text calibration with split QKV and rank-local routed experts.

The attention, routing and decoder implementations are the checkpoint's own
remote code. Only expert ownership/communication is changed. In particular,
the routed sum remains FP32 until *after* the EP all-reduce.
"""

import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys
import types

import torch
import torch.distributed as dist
from torch import nn


_PATCHED_CLASSES = set()


def load_modeling_module(config):
    """Import local checkpoint code without writing to Hugging Face's cache.

    The caller already opts into trusted checkpoint code. Keeping an isolated
    package for each source directory also makes the source's relative config
    import work when its path contains hyphens or dots.
    """
    source_dir = Path(getattr(config, "_name_or_path", "")).resolve()
    modeling_path = source_dir / "modeling_mimo_v2.py"
    if not modeling_path.is_file():
        raise FileNotFoundError(
            f"MiMo calibration requires local checkpoint code: {modeling_path}"
        )
    source_hash = hashlib.sha256(os.fsencode(source_dir)).hexdigest()[:16]
    package_name = f"_moe_quant_mimo_{source_hash}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    modeling = importlib.import_module(f"{package_name}.modeling_mimo_v2")
    _patch_mask_api(modeling)
    return modeling


def _patch_mask_api(modeling):
    """Bridge the checkpoint's mask calls across Transformers API versions."""
    if getattr(modeling, "_moe_quant_mask_api_patched", False):
        return

    def compatible_mask(function):
        parameters = inspect.signature(function).parameters

        def call(*args, **kwargs):
            if "inputs_embeds" in parameters and "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            if "cache_position" not in parameters:
                # No cache is used in calibration, so the removed argument is
                # just arange(sequence_length), inferred by the modern API.
                kwargs.pop("cache_position", None)
            return function(*args, **kwargs)

        return call

    modeling.create_causal_mask = compatible_mask(modeling.create_causal_mask)
    modeling.create_sliding_window_causal_mask = compatible_mask(
        modeling.create_sliding_window_causal_mask
    )
    modeling._moe_quant_mask_api_patched = True


def _expert_range(config):
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    ep_size = int(getattr(config, "ep_size", world_size))
    num_experts = int(config.n_routed_experts)
    if ep_size < 1 or num_experts % ep_size:
        raise ValueError(
            f"MiMo n_routed_experts={num_experts} must be divisible by ep_size={ep_size}."
        )
    if ep_size == 1:
        return 1, 0, 0, num_experts
    if not distributed or world_size != ep_size:
        raise ValueError(
            f"MiMo ep_size={ep_size} requires an initialized process group of "
            f"that size (current world_size={world_size})."
        )
    rank = dist.get_rank()
    experts_per_rank = num_experts // ep_size
    start = rank * experts_per_rank
    return ep_size, rank, start, start + experts_per_rank


def _gather_variable_tokens(hidden_states):
    """Gather flattened tokens, allowing different batch/sequence sizes per rank.

    No attention crosses rank boundaries: this gather runs inside the
    token-wise MoE, after each rank's own attention has finished.
    """
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    shape = torch.tensor(flat.shape, dtype=torch.long, device=flat.device)
    shapes = [torch.empty_like(shape) for _ in range(dist.get_world_size())]
    dist.all_gather(shapes, shape)
    sizes = [int(item[0].item()) for item in shapes]
    if any(int(item[1].item()) != flat.shape[1] for item in shapes):
        raise ValueError("MiMo EP ranks must have equal hidden dimensions.")
    max_tokens = max(sizes)
    padded = flat.new_zeros(max_tokens, flat.shape[1])
    padded[: flat.shape[0]] = flat
    gathered = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(gathered, padded.contiguous())
    return torch.cat([item[:size] for item, size in zip(gathered, sizes)], dim=0), sizes


def _local_moe_float32(self, hidden_states, topk_indices, topk_weights):
    output = torch.zeros(
        hidden_states.shape, dtype=torch.float32, device=hidden_states.device
    )
    for expert_id in range(self.experts_start_idx, self.experts_end_idx):
        token_indices, weight_indices = torch.where(topk_indices == expert_id)
        if token_indices.numel() == 0:
            continue
        expert_output = self.experts[expert_id](hidden_states[token_indices])
        weighted_output = (
            expert_output.float()
            * topk_weights[token_indices, weight_indices].float().unsqueeze(-1)
        )
        output.index_add_(0, token_indices, weighted_output)
    return output


def _moe_forward_ep(self, hidden_states):
    if self.ep_size == 1:
        # Retain the exact source implementation, including accumulation order.
        return self._moe_quant_original_forward(hidden_states)
    if self.training:
        raise ValueError("MiMo expert-parallel calibration supports inference only.")
    original_shape = hidden_states.shape
    gathered, sizes = _gather_variable_tokens(hidden_states)
    # The source gate flattens the first two dimensions; treating the gathered
    # tokens as one sequence has identical token-wise routing semantics.
    topk_indices, topk_weights = self.gate(gathered.unsqueeze(0))
    output = _local_moe_float32(self, gathered, topk_indices, topk_weights)
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    start = sum(sizes[: self.ep_rank])
    local_output = output[start : start + sizes[self.ep_rank]]
    return local_output.to(hidden_states.dtype).reshape(original_shape)


def _patch_moe_class(modeling):
    moe_class = modeling.MiMoV2MoE
    if moe_class in _PATCHED_CLASSES:
        return
    original_forward = moe_class.forward

    def ep_init(self, config):
        nn.Module.__init__(self)
        self.config = config
        ep_size, ep_rank, start, end = _expert_range(config)
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_start_idx = start
        self.experts_end_idx = end
        self.experts_per_rank = end - start
        # Keep global expert IDs in the module/state_dict names.
        self.experts = nn.ModuleList(
            [
                modeling.MiMoV2MLP(config, intermediate_size=config.moe_intermediate_size)
                if start <= expert_id < end
                else None
                for expert_id in range(config.n_routed_experts)
            ]
        )
        self.gate = modeling.MiMoV2MoEGate(config)

    moe_class.__init__ = ep_init
    moe_class._moe_quant_original_forward = original_forward
    moe_class.forward = _moe_forward_ep
    _PATCHED_CLASSES.add(moe_class)


def preserve_router_bias_dtype(model):
    """Source routing correction is FP32 even when calibration weights are BF16."""
    for module in model.modules():
        bias = getattr(module, "e_score_correction_bias", None)
        if isinstance(bias, nn.Parameter) and bias.dtype != torch.float32:
            module.e_score_correction_bias = nn.Parameter(
                bias.to(dtype=torch.float32), requires_grad=bias.requires_grad
            )


class MiMoCalibrationModel(nn.Module):
    """Text-only causal LM with the checkpoint's exact decoder implementation."""

    def __init__(self, config, modeling):
        super().__init__()
        from transformers import GenerationConfig

        self.config = config
        self.model = modeling.MiMoV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.generation_config = GenerationConfig.from_model_config(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        # Calibration normally invokes forward_block directly. This also makes
        # a materialized meta-built wrapper usable for a small smoke forward.
        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
        modeling = sys.modules[type(self.model).__module__]
        for name, is_swa in (("rotary_emb", False), ("swa_rotary_emb", True)):
            rope = getattr(self.model, name)
            if rope.inv_freq.is_meta:
                setattr(
                    self.model, name,
                    modeling.MiMoV2RotaryEmbedding(self.config, is_swa, device=device),
                )
        outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        return CausalLMOutputWithPast(
            logits=self.lm_head(outputs.last_hidden_state),
            past_key_values=outputs.past_key_values,
        )


def build_model(config, dtype, attn_implementation="eager"):
    """Build under accelerate.init_empty_weights or on CPU, without multimodal towers."""
    if attn_implementation not in (None, "eager"):
        raise ValueError(
            "MiMo GPTQ calibration currently requires --attn_implementation eager "
            "to preserve sliding masks and attention sink bias."
        )
    config._attn_implementation = "eager"
    config.attention_projection_layout = "split"
    config.use_cache = False
    modeling = load_modeling_module(config)
    _patch_moe_class(modeling)
    model = MiMoCalibrationModel(config, modeling).to(dtype=dtype).eval()
    # Module.to(dtype) also casts floating buffers. Source RoPE computes its
    # frequencies in FP32; rebuilding keeps that precision (and honors an
    # enclosing init_empty_weights(include_buffers=True) context).
    model.model.rotary_emb = modeling.MiMoV2RotaryEmbedding(config, is_swa=False)
    model.model.swa_rotary_emb = modeling.MiMoV2RotaryEmbedding(config, is_swa=True)
    preserve_router_bias_dtype(model)
    return model


def forward_block(block, hidden_states, position_ids, block_state=None):
    """Run source decoder attention with its full/SWA RoPE and causal mask.

    Construct RoPE on the activation device instead of moving the original
    meta buffers. Masks/position tensors are ephemeral to avoid retaining a
    quadratic attention mask for every calibration sample and every layer.
    """
    config = block.self_attn.config
    if config._attn_implementation != "eager":
        raise ValueError("MiMo calibration forward requires eager attention.")
    modeling = sys.modules[type(block).__module__]
    is_swa = block.attention_type == "sliding_window_attention"
    if is_swa and getattr(config, "sliding_window", None) is None:
        raise ValueError("MiMo SWA layers require config.sliding_window.")
    position_ids = position_ids.to(hidden_states.device)
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    mask_builder = (
        modeling.create_sliding_window_causal_mask if is_swa else modeling.create_causal_mask
    )
    attention_mask = mask_builder(
        config=config,
        input_embeds=hidden_states,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    rope = modeling.MiMoV2RotaryEmbedding(config, is_swa, device=hidden_states.device)
    position_embeddings = rope(hidden_states, position_ids)
    result = block(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        cache_position=cache_position,
        past_key_values=None,
        use_cache=False,
    )
    if isinstance(result, tuple):
        result = result[0]
    return result, block_state
