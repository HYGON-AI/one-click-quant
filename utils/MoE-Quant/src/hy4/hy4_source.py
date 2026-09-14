"""Bounded logical expert reads for main and MTP blocks from fused BF16."""
import json
import math
import os
import struct
from pathlib import Path
import torch
from safetensors import safe_open

class Hy4Source:
    def __init__(self, root, manifest):
        self.root=Path(root).resolve()
        self.index=json.loads((self.root/'model.safetensors.index.json').read_text())['weight_map']
        self.config=json.loads((self.root/'config.json').read_text())
        data=json.loads(Path(manifest).read_text())
        self.targets={r['name']:r for r in data['targets']}
        self.headers={}

    def _read(self, name, row_start=0, rows=None, expert=None):
        path=(self.root/self.index[name]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Source escapes root')
        if path not in self.headers:
            with open(path,'rb',buffering=0) as f:
                n=struct.unpack('<Q',f.read(8))[0]
                if not 0<n<=64<<20:
                    raise ValueError('Invalid tensor header')
                self.headers[path]=(8+n,json.loads(f.read(n)))
        base,header=self.headers[path]
        spec=header[name]
        dtype={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,
               'I64':torch.int64,'I32':torch.int32,'U8':torch.uint8,'I8':torch.int8}[spec['dtype']]
        shape=spec['shape']; width=torch.empty((),dtype=dtype).element_size()
        lo,hi=spec['data_offsets']
        if math.prod(shape)*width!=hi-lo:
            raise ValueError('Invalid tensor byte span')
        offset=base+lo
        if expert is not None:
            if len(shape)!=3 or not 0<=expert<shape[0]:
                raise ValueError('Invalid expert slice')
            offset+=expert*shape[1]*shape[2]*width
            shape=shape[1:]
        if rows is not None:
            if len(shape)!=2 or row_start<0 or row_start+rows>shape[0]:
                raise ValueError('Invalid row slice')
            offset+=row_start*shape[1]*width
            shape=[rows,shape[1]]
        size=math.prod(shape)*width
        if size>2<<30:
            raise MemoryError('Single retained tensor exceeds2GiB')
        value=torch.empty(shape,dtype=dtype,device='cpu')
        buffer=memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast('B')
        with open(path,'rb',buffering=0) as f:
            os.posix_fadvise(f.fileno(),offset,size,os.POSIX_FADV_RANDOM)
            f.seek(offset)
            done=0
            while done<size:
                count=f.readinto(buffer[done:done+(16<<20)])
                if not count:
                    raise EOFError(str(path))
                # Eviction hint is limited to bytes just read by this task;
                # never global drop_caches or whole-source cache invalidation.
                os.posix_fadvise(f.fileno(),offset+done,count,os.POSIX_FADV_DONTNEED)
                done+=count
        return value

    def tensor(self, name):
        """Read a retained tensor, never use this on fused expert tensors."""
        if name in {v['source'] for v in self.targets.values()}:
            raise ValueError('Use projection() for fused expert storage')
        return self._read(name)

    def projection(self, name):
        item=self.targets[name]
        result=self._read(item['source'],item['row_start'],item['rows'],item['expert'])
        if result.dtype!=torch.bfloat16 or list(result.shape)!=[item['rows'],item['columns']]:
            raise ValueError('Unexpected expert projection')
        return result

    def block_prefix(self, block):
        n=self.config['num_hidden_layers']
        if 0<=block<n:
            return f'model.layers.{block}'
        if n<=block<n+self.config['num_nextn_predict_layers']:
            return f'model.mtp_layers.{block-n}'
        raise ValueError('Unknown block')

    def iter_block(self, block):
        prefix=self.block_prefix(block)+'.'
        # Projected logical tensors are owned by only this block.
        for name in sorted(self.targets):
            if name.startswith(prefix):
                yield name,self.projection(name)
        physical_targets={v['source'] for v in self.targets.values()}
        for name in sorted(self.index):
            if not name.startswith(prefix) or name in physical_targets:
                continue
            yield name,self.tensor(name)

def build_manifest(root):
    """Inspect tensor headers without reading BF16 payloads."""
    import hashlib
    root = Path(root).resolve()
    raw = (root / 'model.safetensors.index.json').read_bytes()
    index = json.loads(raw)['weight_map']
    config = json.loads((root / 'config.json').read_text())
    if config['num_hidden_layers'] != 78 or config['num_nextn_predict_layers'] != 1:
        raise ValueError('Expected Hy4 78+1 layout')
    headers = {}
    for filename in sorted(set(index.values())):
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Shard escapes source')
        with path.open('rb') as stream:
            n = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < n <= 64 << 20:
                raise ValueError('Invalid header length')
            header = json.loads(stream.read(n))
        for name, spec in header.items():
            if name != '__metadata__':
                if name in headers or index.get(name) != filename:
                    raise ValueError('Duplicate/mismatched tensor inventory')
                headers[name] = spec
    if set(headers) != set(index):
        raise ValueError('Incomplete source header inventory')
    h, m, e = config['hidden_size'], config['moe_intermediate_size'], config['n_routed_experts']
    targets, used = [], set()
    prefixes = ['model.layers.' + str(i) for i in range(1, 78)] + ['model.mtp_layers.0']
    for prefix in prefixes:
        for part in ('gate', 'up', 'down'):
            key = prefix + '.mlp.experts.' + ('down_proj' if part == 'down' else 'gate_up_proj')
            shape = [e, h, m] if part == 'down' else [e, 2*m, h]
            if headers[key]['dtype'] != 'BF16' or headers[key]['shape'] != shape:
                raise ValueError('Unexpected expert bank: ' + key)
            used.add(key)
            for expert in range(e):
                targets.append(dict(name=f'{prefix}.mlp.experts.{expert}.{part}_proj.weight',
                    source=key, expert=expert, row_start=m if part == 'up' else 0,
                    rows=h if part == 'down' else m, columns=m if part == 'down' else h))
            key = prefix + '.mlp.shared_experts.' + part + '_proj.weight'
            shape = [h, m*config['n_shared_experts']] if part == 'down' else [m*config['n_shared_experts'], h]
            if headers[key]['dtype'] != 'BF16' or headers[key]['shape'] != shape:
                raise ValueError('Unexpected shared projection: ' + key)
            used.add(key)
            targets.append(dict(name=key, source=key, expert=None, row_start=0,
                                rows=shape[0], columns=shape[1]))
    return dict(format='hy4-target-manifest-v1', source_index_sha256=hashlib.sha256(raw).hexdigest(),
                targets=targets, preserved=sorted(set(index)-used), logical_projections=len(targets),
                physical_target_tensors=len(used),
                mtp_projections=sum('.mtp_layers.' in t['name'] for t in targets))


if __name__ == '__main__':
    import argparse
    from .checkpoint_writer import atomic_json
    parser = argparse.ArgumentParser(description='Create an explicit Hy4 target manifest from source headers')
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    output = Path(args.manifest)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing manifest')
    result = build_manifest(args.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps({k: result[k] for k in ('logical_projections', 'physical_target_tensors', 'mtp_projections')}))
