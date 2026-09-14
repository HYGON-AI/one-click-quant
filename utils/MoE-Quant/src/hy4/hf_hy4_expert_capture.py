"""Scoped observation of pinned Transformers HYV4Experts fused projections.

The observer must consume tensors synchronously without modifying them. No
expert-wide Hessian buffers are allocated here. The caller owns the budget.
"""
from contextlib import contextmanager
import types
import torch
import torch.nn.functional as F


def _forward(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(mask[expert_idx])
        routed = hidden_states[token_idx]
        # Preserve the official fused GEMM, clamp/SwiGLU, down GEMM, and
        # weighted index_add ordering. Observers see the actual operands.
        gate_up = F.linear(routed, self.gate_up_proj[expert_idx])
        current = self._apply_gate(gate_up)
        observer = self._hy4_projection_observer
        number = int(expert_idx)
        observer(number, 'gate', routed.detach())
        observer(number, 'up', routed.detach())
        observer(number, 'down', current.detach())
        current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, current.to(final.dtype))
    return final


@contextmanager
def capture_experts(experts, observer):
    from transformers.models.hy_v4.modeling_hy_v4 import HYV4Experts
    if type(experts) is not HYV4Experts:
        raise TypeError('Requires pinned native HYV4Experts')
    if not callable(observer) or hasattr(experts, '_hy4_projection_observer'):
        raise ValueError('Invalid or nested projection observer')
    original = experts.__dict__.get('forward')
    had_override = 'forward' in experts.__dict__
    experts._hy4_projection_observer = observer
    experts.forward = types.MethodType(_forward, experts)
    try:
        yield experts
    finally:
        if had_override:
            experts.forward = original
        else:
            del experts.forward
        del experts._hy4_projection_observer
