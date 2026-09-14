"""Bounded fused-expert group to native MoE-Quant GPTQ handle bridge."""
import torch
from .hf_hy4_expert_capture import capture_experts


class ProjectionView(torch.nn.Linear):
    def __init__(self, storage):
        torch.nn.Module.__init__(self)
        self.in_features = storage.shape[1]
        self.out_features = storage.shape[0]
        self.register_parameter('bias', None)
        self.weight = torch.nn.Parameter(storage, requires_grad=False)
        # Not a registered parameter: the fused bank remains owned by the model.
        object.__setattr__(self, '_destination', storage)

    @torch.no_grad()
    def commit(self):
        if self.weight.shape != self._destination.shape:
            raise ValueError('Quantized projection shape changed')
        self._destination.copy_(self.weight)


class ExpertGroup:
    def __init__(self, experts, expert_ids, prefix, args, max_hessian_bytes):
        from src.gptq import GPTQ
        ids = tuple(expert_ids)
        if not ids or len(set(ids)) != len(ids) or any(type(i)!=int or not 0<=i<experts.num_experts for i in ids):
            raise ValueError('Unique in-range expert group required')
        # gate/up share H; down has its own. Workspace is separately budgeted.
        required = len(ids)*4*(experts.hidden_dim**2 + experts.intermediate_dim**2)
        if required > max_hessian_bytes:
            raise MemoryError('Expert group exceeds Hessian budget')
        self.experts = experts
        self.handles = {}
        self.by_projection = {}
        for expert in ids:
            split = experts.intermediate_dim
            banks = dict(gate=experts.gate_up_proj[expert, :split],
                         up=experts.gate_up_proj[expert, split:],
                         down=experts.down_proj[expert])
            gate_handle = None
            for projection, storage in banks.items():
                handle = GPTQ(ProjectionView(storage), group_size=None, sym=True,
                    rel_damp=args.rel_damp, block_size=128,
                    quantization_order='default', quantization_scale='absmax',
                    is_distributed=False,
                    tied_gptq_handle=gate_handle if projection=='up' else None)
                if projection == 'gate':
                    gate_handle = handle
                self.handles[f'{prefix}.{expert}.{projection}_proj'] = handle
                self.by_projection[(expert, projection)] = handle

    def observe(self, expert, projection, inputs):
        handle = self.by_projection.get((expert, projection))
        if handle is not None and projection != 'up' and inputs.shape[0]:
            handle.update(inputs)

    def capture(self):
        return capture_experts(self.experts, self.observe)

    def finish_collection(self):
        for (expert, projection), handle in self.by_projection.items():
            if projection == 'up':
                handle.num_samples = self.by_projection[(expert, 'gate')].num_samples

    def commit(self):
        for handle in self.handles.values():
            handle.layer.commit()
