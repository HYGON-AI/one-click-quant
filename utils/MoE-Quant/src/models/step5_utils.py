"""A dependency-light Step-5 text model used during GPTQ calibration.

The runtime model has HCU-specific attention and MoE kernels.  Calibration only
needs the same tensor semantics, so this module implements the reference
PyTorch path and keeps the global expert names required by MoE-Quant EP.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class Step5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, zero_centered: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        self.eps = eps
        self.zero_centered = zero_centered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float()
        if self.zero_centered:
            weight = weight + 1.0
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * weight).to(x.dtype)


class Step5MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, limit: float = 7.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_x = x.to(self.gate_proj.weight.dtype)
        gate = F.silu(self.gate_proj(compute_x))
        up = self.up_proj(compute_x)
        if self.limit:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(-self.limit, self.limit)
        return self.down_proj((gate * up).to(self.down_proj.weight.dtype))


def _expert_range(config):
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    ep_size = int(getattr(config, "ep_size", world))
    rank = dist.get_rank() if ep_size > 1 and world > 1 else 0
    count = int(config.moe_num_experts)
    if ep_size < 1 or count % ep_size:
        raise ValueError(f"Step5 experts={count} must be divisible by ep_size={ep_size}.")
    if ep_size > 1 and (not dist.is_initialized() or dist.get_world_size() != ep_size):
        raise ValueError(f"Step5 ep_size={ep_size} requires a matching process group.")
    per_rank = count // ep_size
    return ep_size, rank, rank * per_rank, (rank + 1) * per_rank


class Step5MoE(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.ep_size, self.ep_rank, self.experts_start_idx, self.experts_end_idx = _expert_range(config)
        self.experts = nn.ModuleList(
            [
                Step5MLP(config.hidden_size, config.moe_intermediate_size)
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(config.moe_num_experts)
            ]
        )
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.router_bias = nn.Parameter(
            torch.empty(config.moe_num_experts, dtype=torch.float32), requires_grad=False
        )

    def _route(self, x: torch.Tensor):
        logits = F.linear(x.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        _, ids = torch.topk(scores + self.router_bias.float(), self.config.moe_top_k, dim=-1)
        weights = scores.gather(1, ids)
        if self.config.norm_expert_weight:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return ids, weights * float(self.config.moe_router_scaling_factor)

    def _local(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor):
        out = torch.zeros_like(x, dtype=torch.float32)
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            token_idx, slot_idx = torch.where(ids == expert_id)
            if token_idx.numel() == 0:
                continue
            value = self.experts[expert_id](x[token_idx]).float()
            out.index_add_(0, token_idx, value * weights[token_idx, slot_idx].float().unsqueeze(-1))
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.ep_size == 1:
            ids, weights = self._route(x)
            routed = self._local(x, ids, weights)
        else:
            shape = torch.tensor([x.shape[0], x.shape[1]], device=x.device, dtype=torch.long)
            shapes = [torch.empty_like(shape) for _ in range(self.ep_size)]
            dist.all_gather(shapes, shape)
            if any(int(s[1]) != x.shape[1] for s in shapes):
                raise ValueError("Step5 EP hidden dimensions must match across ranks.")
            sizes = [int(s[0]) for s in shapes]
            max_tokens = max(sizes)
            padded = torch.zeros(max_tokens, x.shape[1], device=x.device, dtype=x.dtype)
            padded[: x.shape[0]] = x
            gathered = [torch.empty_like(padded) for _ in range(self.ep_size)]
            dist.all_gather(gathered, padded)
            all_x = torch.cat([t[:n] for t, n in zip(gathered, sizes)], dim=0)
            ids, weights = self._route(all_x)
            routed_all = self._local(all_x, ids, weights)
            dist.all_reduce(routed_all, op=dist.ReduceOp.SUM)
            begin = sum(sizes[: self.ep_rank])
            routed = routed_all[begin : begin + x.shape[0]]
        return routed.to(hidden_states.dtype).reshape(original_shape)


def _rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


def _apply_rope(x, positions, theta, rotary_dim):
    if rotary_dim <= 0:
        return x
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim))
    freqs = torch.einsum("t,d->td", positions.float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    head = x[..., :rotary_dim]
    return torch.cat(
        (head * cos + _rotate_half(head) * sin, x[..., rotary_dim:]), dim=-1
    ).to(x.dtype)


class Step5Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_groups
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.q_norm = Step5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Step5RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        theta = self.config.rope_theta[self.layer_idx] if isinstance(self.config.rope_theta, list) else self.config.rope_theta
        factor = self.config.partial_rotary_factors[self.layer_idx] if self.config.partial_rotary_factors else 1.0
        rotary_dim = int(self.head_dim * factor)
        q = _apply_rope(q, positions.reshape(-1), theta, rotary_dim)
        k = _apply_rope(k, positions.reshape(-1), theta, rotary_dim)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        if self.config.layer_types and self.config.layer_types[self.layer_idx] == "sliding_attention":
            window = int(self.config.sliding_window)
            mask = mask & torch.ones(t, t, device=x.device, dtype=torch.bool).triu(-window + 1)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        attn = attn.transpose(1, 2)
        attn = attn * self.g_proj(x).sigmoid().unsqueeze(-1)
        return self.o_proj(attn.reshape(b, t, -1))


class Step5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Step5Attention(config, layer_idx)
        self.post_attention_layernorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        if layer_idx in set(config.moe_layer_list):
            self.moe = Step5MoE(config, layer_idx)
            self.share_expert = Step5MLP(config.hidden_size, config.share_expert_dim)
            self.mlp = None
        else:
            self.mlp = Step5MLP(config.hidden_size, config.intermediate_size)
            self.moe = None

    def forward(self, hidden_states, position_ids=None):
        residual = hidden_states.float()
        x = self.input_layernorm(residual).to(self.self_attn.q_proj.weight.dtype)
        x = self.self_attn(x, position_ids)
        residual = residual.float() + x.float()
        x = self.post_attention_layernorm(residual).to(self.self_attn.q_proj.weight.dtype)
        if self.moe is not None:
            x = self.moe(x).float() + self.share_expert(x).float()
        else:
            x = self.mlp(x)
        return residual + x.float()


class Step5TextModel(nn.Module):
    def __init__(self, config, dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Step5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.to(dtype=dtype)
        # Norms and router correction biases are intentionally kept precise.
        for module in self.modules():
            if isinstance(module, Step5RMSNorm):
                module.weight.data = module.weight.data.float()
            if isinstance(module, Step5MoE):
                module.router_bias.data = module.router_bias.data.float()
                module.gate.weight.data = module.gate.weight.data.float()
        self.embed_tokens.to(dtype=dtype)


class Step5CalibrationModel(nn.Module):
    def __init__(self, config, dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.model = Step5TextModel(config, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

