"""Per-document, cache-free state for the pinned official HF Hy4 decoder."""
from dataclasses import dataclass, replace
import torch
from transformers.masking_utils import create_causal_mask


@dataclass(frozen=True)
class LayerState:
    positions: torch.Tensor
    mask: object
    rope: tuple
    topk: object = None
    next_layer: int = 0


def prepare(model, input_ids):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('Exactly one nonempty unpadded document required')
    embeddings = model.embed_tokens(input_ids)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    return prepare_embeddings(model.config, model.rotary_emb, embeddings, positions)


def prepare_embeddings(config, rotary, embeddings, positions):
    if embeddings.ndim != 3 or embeddings.shape[0] != 1 or positions.shape != embeddings.shape[:2]:
        raise ValueError('Document embedding/position shape mismatch')
    mask = create_causal_mask(config=config, inputs_embeds=embeddings,
        attention_mask=None, past_key_values=None, position_ids=positions,
        allow_is_causal_skip=False)
    rope = rotary(embeddings, position_ids=positions)
    hidden = embeddings.unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
    return hidden, LayerState(positions, mask, rope)


def move_state(state, device):
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if value is None:
            return None
        raise TypeError('Unsupported state value')
    return replace(state, positions=move(state.positions), mask=move(state.mask),
                   rope=move(state.rope), topk=move(state.topk))


def forward_block(block, hidden, state):
    if block.layer_idx != state.next_layer:
        raise ValueError('Out-of-order layer or wrong document state')
    output, topk = block(hidden, attention_mask=state.mask,
        position_embeddings=state.rope, position_ids=state.positions,
        prev_topk_indices=state.topk, past_key_values=None, use_cache=False)
    return output, replace(state, topk=topk, next_layer=state.next_layer + 1)
