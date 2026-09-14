"""Explicit logical projection inventory for the README-native Hy4 adapter."""
from .hf_hy4_gptq_bridge import ProjectionView


def projection_views(block, prefix):
    """Return only routed/shared gate/up/down; do not register duplicate banks."""
    from transformers.models.hy_v4.modeling_hy_v4 import HYV4MoE
    mlp = getattr(block, 'mlp', None)
    if not isinstance(mlp, HYV4MoE):
        return {}
    result = {}
    experts = mlp.experts
    split = experts.intermediate_dim
    for number in range(experts.num_experts):
        banks = dict(gate=experts.gate_up_proj[number, :split],
                     up=experts.gate_up_proj[number, split:],
                     down=experts.down_proj[number])
        for projection, storage in banks.items():
            result[f'{prefix}mlp.experts.{number}.{projection}_proj'] = ProjectionView(storage)
    for projection in ('gate','up','down'):
        result[f'{prefix}mlp.shared_experts.{projection}_proj'] = getattr(mlp.shared_experts, projection+'_proj')
    return result


def model_projection_names(model, mtp):
    """Inventory both model components without loading any source weights."""
    result = {}
    for number, block in enumerate(model.layers):
        result.update(projection_views(block, f'model.layers.{number}.'))
    result.update(projection_views(mtp, 'model.mtp_layers.0.'))
    return result
