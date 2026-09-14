"""Bounded source-to-final-storage loading for native HF Hy4 adapters."""
import json
import math
import re
from pathlib import Path
import torch
from safetensors import safe_open


def translate(name):
    name = name.replace('.hc_pre.hc_', '.hc_')
    name = re.sub(r'\.learnable_sink_param$', '.sinks', name)
    name = name.replace('.linear_gate', '.gate_proj')
    for src, dst in (('hc_attn_layer', 'attn_hc'), ('hc_mlp_layer', 'ffn_hc')):
        for leaf in ('fn', 'base', 'scale'):
            name = name.replace('.'+src+'.hc_'+leaf, '.'+dst+'.'+leaf)
    for leaf in ('fn', 'base', 'scale'):
        name = name.replace('.hc_head_'+leaf, '.hc_'+leaf)
    return name


class SourceLoader:
    def __init__(self, root, chunk_bytes=64 << 20):
        self.root = Path(root).resolve()
        self.index = json.loads((self.root/'model.safetensors.index.json').read_text())['weight_map']
        self.mapping = {}
        if chunk_bytes <= 0:
            raise ValueError('Positive chunk budget required')
        self.chunk_bytes = chunk_bytes
        self._headers = {}
        for source in self.index:
            target = translate(source)
            if target in self.mapping:
                raise ValueError('Source mapping collision: '+target)
            self.mapping[target] = source

    def align_dtypes(self, module, prefix):
        """Keep source FP32 auxiliaries; works on meta or allocated modules.

        Call before constructing projection views/hooks. No model tensor values
        are read here and no previously loaded values may be relied upon.
        """
        changes=[]
        for relative,destination in module.state_dict(keep_vars=True).items():
            target=prefix+relative
            if target not in self.mapping:
                raise ValueError('Missing source: '+target)
            source=self.mapping[target]
            path=(self.root/self.index[source]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Source escapes root')
            if path not in self._headers:
                with safe_open(path,framework='pt',device='cpu') as handle:
                    self._headers[path]={name:(handle.get_slice(name).get_dtype(),handle.get_slice(name).get_shape())
                                         for name in handle.keys()}
            label,shape=self._headers[path][source]
            dtype={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32}.get(label)
            if dtype is None or shape!=list(destination.shape):
                raise ValueError('Unsupported source dtype or shape: '+source)
            if destination.dtype==dtype:
                continue
            parent_name,_,leaf=relative.rpartition('.')
            parent=module.get_submodule(parent_name) if parent_name else module
            replacement=torch.empty_like(destination,dtype=dtype)
            if leaf in parent._parameters:
                parent._parameters[leaf]=torch.nn.Parameter(replacement,requires_grad=destination.requires_grad)
            elif leaf in parent._buffers:
                parent._buffers[leaf]=replacement
            else:
                raise ValueError('Source target is not a registered tensor: '+target)
            changes.append(dict(name=target,before=str(destination.dtype),after=str(dtype)))
        return changes

    @torch.no_grad()
    def load(self, module, prefix):
        dtype_changes=self.align_dtypes(module,prefix)
        targets = module.state_dict(keep_vars=True)
        if any(t.is_meta for t in targets.values()):
            raise ValueError('Allocate only the current module final storage before loading')
        loaded = []
        peak_slice_bytes = 0
        for relative, destination in targets.items():
            target = prefix + relative
            if target not in self.mapping:
                raise ValueError('Missing source: '+target)
            source = self.mapping[target]
            path = (self.root/self.index[source]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Source escapes root')
            with safe_open(path, framework='pt', device='cpu') as handle:
                view = handle.get_slice(source)
                shape = view.get_shape()
                if shape != list(destination.shape):
                    raise ValueError('Shape mismatch: '+source)
                # BF16/F16/F32 are the supported source floating precisions.
                size = {'BF16': 2, 'F16': 2, 'F32': 4}.get(view.get_dtype())
                if size is None or not destination.is_floating_point():
                    raise ValueError('Unexpected source/destination dtype: '+source)
                def copy_slice(selection, dest):
                    nonlocal peak_slice_bytes
                    value = view[selection]
                    count = value.numel()*value.element_size()
                    if count > self.chunk_bytes:
                        raise MemoryError('Source slice budget exceeded')
                    peak_slice_bytes = max(peak_slice_bytes, count)
                    dest.copy_(value)
                if len(shape) == 3:
                    row_bytes = shape[2]*size
                    rows = self.chunk_bytes//row_bytes
                    if rows < 1:
                        raise MemoryError('One expert row exceeds budget')
                    for expert in range(shape[0]):
                        for start in range(0, shape[1], rows):
                            stop = min(shape[1], start+rows)
                            copy_slice((expert, slice(start, stop), slice(None)), destination[expert, start:stop])
                elif len(shape) in (1, 2):
                    row_bytes = math.prod(shape[1:])*size
                    rows = self.chunk_bytes//row_bytes
                    if rows < 1:
                        raise MemoryError('One retained row exceeds budget')
                    for start in range(0, shape[0], rows):
                        stop = min(shape[0], start+rows)
                        copy_slice(slice(start, stop), destination[start:stop])
                else:
                    raise ValueError('Unsupported tensor rank: '+source)
            loaded.append(target)
        return dict(loaded=loaded, peak_source_slice_bytes=peak_slice_bytes,dtype_changes=dtype_changes)
