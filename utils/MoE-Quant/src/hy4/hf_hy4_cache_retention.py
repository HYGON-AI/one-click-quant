"""Retire only reconstructible document tensors after a collective commit."""
import json
from pathlib import Path
import torch.distributed as dist
from .checkpoint_writer import atomic_json,digest
from .hf_hy4_layer_commit import latest


def retire(output,current_layer,run_identity,source_root=None):
    if current_layer<2:
        return 0
    root=Path(output).resolve(); rank=dist.get_rank(); world=dist.get_world_size()
    error=None; retired_bytes=0
    try:
        if json.loads((root/'native-run.json').read_text())!=run_identity:
            raise ValueError('Cache retirement run identity mismatch')
        newest=latest(root,run_identity,world)
        if newest is None or newest['layer']!=current_layer:
            raise ValueError('Cannot retire before current layer collective commit')
        layer=current_layer-2
        layer_commit=json.loads((root/'commits'/f'layer-{layer:03d}.json').read_text())
        record=layer_commit['ranks'][rank]
        relative=f'layer{layer}/boundary/rank{rank}/commit.json'
        if record['boundary']!=relative or record['rank']!=rank:
            raise ValueError('Unexpected boundary ownership')
        boundary=root/f'layer{layer}'/'boundary'/f'rank{rank}'
        if boundary.resolve()!=boundary or not boundary.is_dir():
            raise ValueError('Boundary path aliases or escapes expected cache')
        if digest(boundary/'commit.json')!=record['boundary_sha256']:
            raise ValueError('Boundary commit changed before retirement')
        state=json.loads((boundary/'commit.json').read_text())
        if state.get('complete') is not True or [item['id'] for item in state['documents']]!=record['sample_ids']:
            raise ValueError('Boundary document set mismatch')
        for number,item in enumerate(state['documents']):
            if item['file']!=f'doc-{number:05d}.safetensors':
                raise ValueError('Unexpected retirement filename')
        folder=root/'retired-boundaries'; folder.mkdir(exist_ok=True)
        marker=folder/f'layer-{layer:03d}-rank-{rank}.json'
        provenance=dict(format='hy4-retired-boundary-v1',layer=layer,rank=rank,
            after_layer=current_layer,boundary_sha256=record['boundary_sha256'],
            documents=state['documents'])
        existed=marker.exists()
        if existed:
            old=json.loads(marker.read_text())
            if any(old.get(key)!=value for key,value in provenance.items()):
                raise ValueError('Retirement journal changed')
        else:
            if any(not (boundary/item['file']).is_file() for item in state['documents']):
                raise ValueError('Unjournaled missing cache tensor')
            atomic_json(marker,dict(provenance,complete=False))
        for item in state['documents']:
            path=boundary/item['file']
            if path.is_symlink():
                raise ValueError('Refusing to remove a symlink')
            if not path.exists():
                if existed:
                    continue
                raise ValueError('Cache tensor disappeared during retirement')
            if path.resolve().parent!=boundary or digest(path)!=item['sha256']:
                raise ValueError('Cache tensor changed before retirement')
            retired_bytes+=path.stat().st_size
            path.unlink() # Exact hash-verified task-created tensor; never recursive.
        atomic_json(marker,dict(provenance,complete=True))
        if rank==0 and source_root is not None:
            from .hf_hy4_source_cache import advise_old_layer
            advised=advise_old_layer(source_root,layer,run_identity['main_layers'])
            print(f'[Hy4 native] source_cache_advised_bytes={advised} old_layer={layer}; source files unchanged',flush=True)
    except Exception as failure:
        error=repr(failure)
    errors=[None]*world; dist.all_gather_object(errors,error)
    if any(errors):
        raise RuntimeError('Cache retirement failed; preserve committed progress: '+repr(errors))
    return retired_bytes
