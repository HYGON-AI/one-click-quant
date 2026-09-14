"""MTP transition hook for the original quant.py layer loop."""
import torch
from transformers.models.hy_v4.modeling_hy_v4 import HYV4RotaryEmbedding
from .hf_hy4_source_loader import SourceLoader
from .hf_hy4_mtp import MTPState


@torch.no_grad()
def prepare(adapter, model, calibration, model_path, device, offload_device):
    backbone = model.model
    if len(calibration.inputs) != len(calibration.dataset):
        raise ValueError('MTP document count mismatch')
    if any(isinstance(state, MTPState) or state.next_layer != backbone.config.num_hidden_layers
           for state in calibration.block_states):
        raise ValueError('MTP transition requires completed main states exactly once')
    loader = SourceLoader(model_path)
    auxiliaries = [('embed_tokens', backbone.embed_tokens), ('hc_head', backbone.hc_head),
                   ('norm', backbone.norm)]
    for name, module in auxiliaries:
        module.to_empty(device=device)
        loader.load(module, 'model.'+name+'.')
    with torch.device(device):
        backbone.rotary_emb = HYV4RotaryEmbedding(backbone.config)
    inputs, positions, states, dataset, kept_ids, skipped = [], [], [], [], [], []
    sample_ids = calibration.frozen_identity['sample_ids']
    if len(sample_ids) != len(calibration.dataset):
        raise ValueError('Frozen sample IDs missing at MTP transition')
    for index, (ids, hidden) in enumerate(zip(calibration.dataset, calibration.inputs)):
        if ids.shape[1] < 2:
            skipped.append(dict(id=sample_ids[index], reason='no_within_document_next_token'))
            continue
        value, state = adapter.prepare_mtp_document(model, ids.to(device), hidden.to(device))
        inputs.append(value.to(offload_device))
        positions.append(state.positions)
        states.append(adapter.move_block_state(state, offload_device))
        dataset.append(ids)
        kept_ids.append(sample_ids[index])
    for _, module in auxiliaries:
        module.to(device='meta')
    backbone.rotary_emb.to(device='meta')
    calibration.inputs, calibration.position_ids, calibration.block_states = inputs, positions, states
    calibration.dataset = dataset
    calibration.num_seq_per_rank = len(dataset)
    calibration.frozen_identity = dict(calibration.frozen_identity,
        sample_ids=kept_ids, mtp_skipped=skipped, mtp_transition=True)
    return dict(documents=len(dataset), skipped=skipped,
                valid_mtp_tokens=sum(value.shape[1] for value in inputs))
