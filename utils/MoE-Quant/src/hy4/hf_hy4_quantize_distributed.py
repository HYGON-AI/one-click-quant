"""Native-stage hook: weighted merge, disjoint GPTQ owners, candidate sync."""
import torch
import torch.distributed as dist
from .hf_hy4_distributed_hessian import merge_to_owner
from .hf_hy4_quantize_handle import quantize_handle, commit_candidate


def quantize_distributed(handles, save_artifact):
    """All ranks call once per layer with identically named native handles.

    save_artifact runs only on the unique owner and must commit atomically.
    Dequantized broadcast is solely sequential calibration, not inference.
    """
    names = sorted(handles)
    rank, world = dist.get_rank(), dist.get_world_size()
    signatures = [None]*world
    signature = [(name, tuple(handles[name].layer.weight.shape)) for name in names]
    dist.all_gather_object(signatures, signature)
    if any(other != signature for other in signatures):
        raise ValueError('Projection inventory differs between calibration ranks')
    if not names:
        return dict(projections=0, owned=0)
    device = handles[names[0]].layer.weight.device
    owned = {}
    counts = {}
    host_hessian_bytes = 0
    # Preserve local tied statistics until *all* gate/up reductions have read
    # them. Replacing gate.H too early would corrupt the up projection merge.
    for number, name in enumerate(names):
        handle = handles[name]
        hessian = handle.H
        if hessian is None and handle.tied_gptq_handle is not None:
            hessian = handle.tied_gptq_handle.H
        owner = number % world
        merged, count = merge_to_owner(hessian, handle.num_samples,
            handle.layer.in_features, owner, device, name)
        counts[name] = count
        if rank == owner:
            # Local statistics (~40 GiB at real Hy4 width) must remain intact
            # for tied gate/up reductions. Do not add another ~10 GiB of owner
            # results to the same GPU: stage them in available host memory and
            # upload only the current projection inside NarrowGPTQ.
            owned[name] = None if merged is None else merged.to('cpu')
            if owned[name] is not None:
                host_hessian_bytes += owned[name].numel()*owned[name].element_size()
        del merged
    # All local statistics have now been consumed; retain only owner matrices.
    for handle in handles.values():
        handle.H = None
        handle.tied_gptq_handle = None
    error = None
    try:
        for name, hessian in owned.items():
            artifact, candidate = quantize_handle(handles[name], hessian, counts[name])
            save_artifact(name, artifact)
            commit_candidate(handles[name], candidate)
            owned[name] = None
    except Exception as failure:
        error = type(failure).__name__+': '+str(failure)
    errors = [None]*world
    dist.all_gather_object(errors, error)
    if any(item is not None for item in errors):
        raise RuntimeError('GPTQ owner failed; do not propagate this layer: '+repr(errors))
    for number, name in enumerate(names):
        handle = handles[name]
        # Use an explicit contiguous communication buffer, even for future
        # projection layouts that do not have contiguous fused-bank views.
        value = handle.layer.weight.detach().contiguous()
        dist.broadcast(value, src=number % world)
        commit_candidate(handle, value)
    return dict(projections=len(names), owned=len(owned), token_counts=counts,
                owner_hessian_host_bytes=host_hessian_bytes)
