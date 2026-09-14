"""Hy4 fused/shared expert collection hook for native quant.py collect()."""
from contextlib import ExitStack
from .hf_hy4_gptq_bridge import ExpertGroup


def collect(block, prefix, args, forward_documents, max_hessian_bytes):
    """Call forward_documents once with hooks installed; return native handles.

    Each rank gathers its own document subset. The native engine must merge
    token-weighted statistics before quantizing, not apply its default AVG.
    """
    from transformers.models.hy_v4.modeling_hy_v4 import HYV4MoE
    from src.gptq import GPTQ
    if not isinstance(getattr(block, 'mlp', None), HYV4MoE):
        forward_documents()
        return {}
    mlp = block.mlp
    shared = mlp.shared_experts
    shared_bytes = 4*(shared.gate_proj.in_features**2 + shared.down_proj.in_features**2)
    group = ExpertGroup(mlp.experts, range(mlp.experts.num_experts),
        prefix+'mlp.experts', args, max_hessian_bytes-shared_bytes)
    handles = dict(group.handles)
    shared_handles = {}
    for name in ('gate', 'up', 'down'):
        handle = GPTQ(getattr(shared, name+'_proj'), group_size=None, sym=True,
            rel_damp=args.rel_damp, block_size=128,
            quantization_order='default', quantization_scale='absmax',
            is_distributed=False,
            tied_gptq_handle=shared_handles['gate'] if name=='up' else None)
        shared_handles[name] = handle
        handles[prefix+'mlp.shared_experts.'+name+'_proj'] = handle
    with ExitStack() as stack:
        stack.enter_context(group.capture())
        for name in ('gate','down'):
            handle = shared_handles[name]
            def update(module, inputs, output, handle=handle):
                if inputs[0].numel():
                    handle.update(inputs[0])
            hook = getattr(shared, name+'_proj').register_forward_hook(update)
            stack.callback(hook.remove)
        forward_documents()
    group.finish_collection()
    shared_handles['up'].num_samples = shared_handles['gate'].num_samples
    return handles
