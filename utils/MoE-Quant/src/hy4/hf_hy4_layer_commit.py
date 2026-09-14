"""Collective layer commit: checkpoint indexes plus per-rank boundaries."""
import json
from pathlib import Path
import torch.distributed as dist
from .checkpoint_writer import CheckpointWriter, atomic_json, digest
from .hf_hy4_boundary import save as save_boundary, load as load_boundary
from .hf_hy4_layer_state import LayerState
from .hf_hy4_mtp import MTPState


def commit(output, layer, calibration, run_identity):
    root=Path(output).resolve()
    rank,world=dist.get_rank(),dist.get_world_size()
    identity=dict(run_identity,layer=layer,rank=rank,world_size=world)
    boundary=root/f'layer{layer}'/'boundary'/f'rank{rank}'
    error=None
    record=None
    try:
        for state in calibration.block_states:
            if isinstance(state,LayerState):
                if state.next_layer!=layer+1:
                    raise ValueError('Boundary has not completed the committed main layer')
            elif isinstance(state,MTPState):
                if state.done is not True or layer!=run_identity['main_layers']:
                    raise ValueError('Wrong MTP layer boundary')
            else:
                raise ValueError('Unknown layer boundary state')
        checkpoint=root/f'layer{layer}'/f'rank{rank}'/'hy4-checkpoint.index.json'
        index=json.loads(checkpoint.read_text())
        if index.get('complete') is not True:
            raise ValueError('Quantized rank index is not complete')
        # Bind the checkpoint index into the activation state identity.
        identity['checkpoint_index_sha256']=digest(checkpoint)
        save_boundary(boundary,calibration,identity)
        record=dict(rank=rank,boundary=str((boundary/'commit.json').relative_to(root)),
            boundary_sha256=digest(boundary/'commit.json'),
            checkpoint=str(checkpoint.relative_to(root)),
            checkpoint_sha256=identity['checkpoint_index_sha256'],
            sample_ids=calibration.frozen_identity['sample_ids'])
    except Exception as failure:
        error=repr(failure)
    messages=[None]*world
    dist.all_gather_object(messages,dict(error=error,record=record))
    if any(item['error'] for item in messages):
        raise RuntimeError('Incomplete layer boundary; no global commit: '+repr(messages))
    error=None
    if rank==0:
        try:
            folder=root/'commits'; folder.mkdir(exist_ok=True)
            target=folder/f'layer-{layer:03d}.json'
            if target.exists():
                raise ValueError('Refusing overwrite of committed layer')
            previous=folder/f'layer-{layer-1:03d}.json'
            if layer>0 and not previous.is_file():
                raise ValueError('Previous global layer commit missing')
            atomic_json(target,dict(format='hy4-hf-layer-commit-v1',layer=layer,
                identity=run_identity,world_size=world,complete=True,
                previous_sha256=digest(previous) if layer>0 else None,
                ranks=[item['record'] for item in messages]))
        except Exception as failure:
            error=repr(failure)
    errors=[None]*world
    dist.all_gather_object(errors,error)
    if any(errors):
        raise RuntimeError('Global layer commit failed: '+repr(errors))


def latest(output, run_identity, world_size):
    root=Path(output).resolve()
    folder=root/'commits'
    if not folder.exists():
        return None
    commits=sorted(folder.glob('layer-*.json'))
    previous=None
    state=None
    for layer,path in enumerate(commits):
        if path.name!=f'layer-{layer:03d}.json':
            raise ValueError('Layer commit chain has a gap')
        current=json.loads(path.read_text())
        if (current.get('format')!='hy4-hf-layer-commit-v1'
                or current.get('identity')!=run_identity or current.get('layer')!=layer
                or current.get('complete') is not True or current.get('world_size')!=world_size
                or current.get('previous_sha256')!=previous):
            raise ValueError('Layer commit identity or chain mismatch')
        if [r['rank'] for r in current['ranks']]!=list(range(world_size)):
            raise ValueError('Missing or duplicate rank commits')
        previous=digest(path)
        state=current
    return state


def restore_rank(output, commit_state, rank, expected_ids):
    root=Path(output).resolve()
    record=commit_state['ranks'][rank]
    if record['rank']!=rank or record['sample_ids']!=list(expected_ids):
        raise ValueError('Recovered rank sample partition differs')
    layer=commit_state['layer']
    expected_boundary=f'layer{layer}/boundary/rank{rank}/commit.json'
    expected_checkpoint=f'layer{layer}/rank{rank}/hy4-checkpoint.index.json'
    if record['boundary']!=expected_boundary or record['checkpoint']!=expected_checkpoint:
        raise ValueError('Unexpected commit paths')
    for field in ('boundary','checkpoint'):
        path=(root/record[field]).resolve()
        if not path.is_relative_to(root) or digest(path)!=record[field+'_sha256']:
            raise ValueError('Recovered commit payload binding mismatch')
    checkpoint_path=root/record['checkpoint']
    index=json.loads(checkpoint_path.read_text())
    if index.get('complete') is not True or set(index['parameters'])!=set(index['expected']):
        raise ValueError('Recovered checkpoint is incomplete')
    # Opening an existing writer revalidates its fingerprint and every shard;
    # an unchanged index is not evidence that payload files are intact.
    CheckpointWriter(checkpoint_path.parent,index['identity'],index['expected'])
    identity=dict(commit_state['identity'],layer=layer,rank=rank,
        world_size=commit_state['world_size'],checkpoint_index_sha256=record['checkpoint_sha256'])
    return load_boundary((root/record['boundary']).parent,identity,expected_ids)
