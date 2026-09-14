"""HF-compatible Hy4 MTP structure derived from the pinned native MTP.

No iHC inside the draft decoder. Numerical equivalence with the native GPU
backend must be checked before this path is used for formal calibration.
"""
import copy
from dataclasses import dataclass, replace
import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.models.hy_v4.modeling_hy_v4 import HYV4Attention, HYV4MoE, HYV4RMSNorm


class HFHy4MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if getattr(config, 'num_nextn_predict_layers', None) != 1:
            raise ValueError('This adapter requires the verified single MTP block')
        self.enorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2*config.hidden_size, config.hidden_size, bias=False)
        self.input_layernorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.final_layernorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        draft = copy.deepcopy(config)
        index = config.num_hidden_layers
        # Draft step zero runs its own indexer, matching native set_skip_topk(False).
        draft.indexer_types = list(config.indexer_types) + ['full']
        draft.layer_types = list(config.layer_types) + [config.layer_types[-1]]
        draft.mlp_layer_types = list(config.mlp_layer_types) + ['sparse']
        self.self_attn = HYV4Attention(draft, index)
        self.mlp = HYV4MoE(draft)

    def forward(self, next_embeddings, previous_hidden, positions, mask, rope):
        if next_embeddings.shape != previous_hidden.shape or next_embeddings.ndim != 3:
            raise ValueError('Aligned [document, tokens, hidden] operands required')
        hidden = self.eh_proj(torch.cat((self.enorm(next_embeddings),
                                        self.hnorm(previous_hidden)), dim=-1))
        attention, _, topk = self.self_attn(self.input_layernorm(hidden),
            attention_mask=mask, position_ids=positions, position_embeddings=rope,
            past_key_values=None, use_cache=False, prev_topk_indices=None)
        hidden = hidden + attention
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return self.final_layernorm(hidden), topk


def prepare_teacher_forcing(model, input_ids, previous_hidden):
    """Single unpadded document; caller supplies native target-head features.

    The native proposer keeps query positions at t, shifts token IDs to t+1,
    and zeros the embedding at position zero. This does not synthesize labels
    or establish equivalence of the caller's target-head feature extraction.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError('One unpadded document with at least two tokens required')
    if previous_hidden.ndim != 3 or previous_hidden.shape[:2] != input_ids.shape:
        raise ValueError('Target features must match every source token')
    positions = torch.arange(input_ids.shape[1]-1, device=input_ids.device).unsqueeze(0)
    embeddings = model.embed_tokens(input_ids[:, 1:])
    embeddings = torch.where((positions == 0).unsqueeze(-1), 0, embeddings)
    mask = create_causal_mask(config=model.config, inputs_embeds=embeddings,
        attention_mask=None, past_key_values=None, position_ids=positions,
        allow_is_causal_skip=False)
    rope = model.rotary_emb(embeddings, position_ids=positions)
    return embeddings, previous_hidden[:, :-1], positions, mask, rope


@dataclass(frozen=True)
class MTPState:
    embeddings: torch.Tensor
    positions: torch.Tensor
    mask: object
    rope: tuple
    done: bool = False


def prepare_boundary(model, input_ids, backbone_hidden):
    """Finalize real backbone iHC features, then align one MTP document."""
    if backbone_hidden.ndim != 4 or backbone_hidden.shape[2] != model.config.hc_mult:
        raise ValueError('Expected unfinalized backbone iHC output')
    modules = (model.hc_head, model.norm, model.embed_tokens)
    if any(t.is_meta for module in modules for t in module.state_dict().values()):
        raise ValueError('MTP handoff auxiliary source weights are not loaded')
    previous = model.norm(model.hc_head(backbone_hidden))
    embeddings, hidden, positions, mask, rope = prepare_teacher_forcing(model, input_ids, previous)
    return hidden, MTPState(embeddings, positions, mask, rope)


def move_mtp_state(state, device):
    return replace(state, embeddings=state.embeddings.to(device),
        positions=state.positions.to(device),
        mask=None if state.mask is None else state.mask.to(device),
        rope=tuple(t.to(device) for t in state.rope))


def forward_mtp(block, hidden, positions, state):
    if state.done or not torch.equal(positions, state.positions):
        raise ValueError('MTP state already propagated or positions mismatched')
    output, _ = block(state.embeddings, hidden, positions, state.mask, state.rope)
    return output, replace(state, done=True)
