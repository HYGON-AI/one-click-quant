"""Read-only packed bank view of verified logical MoE-Quant projections.

This is a layout bridge, not an RTN metadata conversion. The caller supplies
a verified CheckpointLoader; full-model completeness remains its responsibility.
No expert weights are dequantized and only the selected experts are assembled.
"""
import re
import torch


class PackedBankView:
    def __init__(self, loader, experts=256):
        self.loader = loader
        self.experts = experts

    def _projection(self, name, field):
        values, metadata = self.loader.projection(name)
        rows, columns = metadata['shape']
        packed, scale = values['packed'], values['scale']
        if packed.dtype != torch.uint8 or scale.dtype != torch.float32:
            raise ValueError('Incorrect packed projection dtype')
        if packed.shape != (rows, (columns + 1)//2) or scale.shape != (rows, 1):
            raise ValueError('Incorrect packed projection shape')
        # Existing resident kernels infer width from packed bytes. Do not hide
        # odd-width padding from them; a separate original-shape kernel is needed.
        if columns % 2:
            raise ValueError('Resident bank kernel requires even input width')
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid channel scale')
        if field == 'int4_packed':
            return packed
        if field == 'scale':
            return scale[:, 0]
        if field == 'input_scale':
            # Current MoE-Quant recipe has no input scaling. Do not silently
            # discard a future scaled format.
            if 'input_scale' in values or 'input_scale' in metadata:
                raise ValueError('Scaled recipe needs an explicit runtime adapter')
            return torch.ones(columns, dtype=torch.float32, device='cpu')
        raise ValueError('Unknown bank field')

    def tensor(self, name, selection=None):
        match = re.fullmatch(r'(.+\.mlp)\.(experts\.(gate_up_proj|down_proj)|shared_experts\.(gate_proj|up_proj|down_proj)\.weight)\.(int4_packed|scale|input_scale)', name)
        if match is None:
            raise ValueError('Unknown physical expert bank name')
        prefix, _, routed, shared, field = match.groups()
        with torch.device('cpu'):
            if shared:
                if selection is not None:
                    raise ValueError('Shared projection has no expert dimension')
                return self._projection(f'{prefix}.shared_experts.{shared}.weight', field)
            if isinstance(selection, int):
                indices = [selection]
            elif isinstance(selection, slice):
                start, stop, step = selection.indices(self.experts)
                indices = list(range(start, stop, step))
            else:
                raise ValueError('Explicit bounded expert selection required')
            if not indices or any(i < 0 or i >= self.experts for i in indices):
                raise ValueError('Invalid expert selection')
            result = []
            for expert in indices:
                base = f'{prefix}.experts.{expert}'
                if routed == 'gate_up_proj':
                    gate = self._projection(base+'.gate_proj.weight', field)
                    up = self._projection(base+'.up_proj.weight', field)
                    if field == 'input_scale':
                        if not torch.equal(gate, up):
                            raise ValueError('Fused gate/up input scales differ')
                        value = gate
                    else:
                        value = torch.cat((gate, up), dim=0)
                else:
                    value = self._projection(base+'.down_proj.weight', field)
                result.append(value)
            return result[0] if isinstance(selection, int) else torch.stack(result)
