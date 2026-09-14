"""Versioned document-local native-loop boundaries; no pickle deserialization."""
import json
import os
import uuid
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from .checkpoint_writer import atomic_json, digest
from .hf_hy4_layer_state import LayerState
from .hf_hy4_mtp import MTPState


def validate(values, metadata):
    hidden,tokens,positions=values['hidden'],values['input_ids'],values['positions']
    if positions.ndim!=2 or positions.shape[0]!=1 or positions.shape[1]<1:
        raise ValueError('Invalid document position shape')
    length=positions.shape[1]
    if positions.dtype!=torch.int64 or not torch.equal(positions.cpu(),torch.arange(length).unsqueeze(0)):
        raise ValueError('Document position sequence changed')
    if tokens.dtype!=torch.int64 or tokens.ndim!=2 or tokens.shape[0]!=1 or (tokens<0).any():
        raise ValueError('Invalid document token IDs')
    if hidden.shape[:2]!=(1,length) or not hidden.is_floating_point() or not torch.isfinite(hidden).all():
        raise ValueError('Invalid document hidden values')
    for name in ('rope_cos','rope_sin'):
        rope=values[name]
        if rope.ndim!=3 or rope.shape[:2]!=(1,length) or not torch.isfinite(rope).all():
            raise ValueError('Invalid document RoPE values')
    if values['rope_cos'].shape!=values['rope_sin'].shape:
        raise ValueError('RoPE shapes differ')
    if metadata['kind']=='main':
        if hidden.ndim!=4 or tokens.shape[1]!=length or type(metadata['next_layer'])!=int or metadata['next_layer']<1:
            raise ValueError('Invalid committed main-layer boundary')
        if 'topk' in values and (values['topk'].ndim!=3 or values['topk'].shape[:2]!=(1,length)
                                or values['topk'].dtype not in (torch.int32,torch.int64)):
            raise ValueError('Invalid document indexer state')
    elif metadata['kind']=='mtp':
        if hidden.ndim!=3 or tokens.shape[1]!=length+1 or metadata['done'] is not True:
            raise ValueError('Invalid committed MTP teacher-forcing boundary')
        embeddings=values['embeddings']
        if embeddings.shape!=hidden.shape or not torch.isfinite(embeddings).all():
            raise ValueError('Invalid MTP embeddings')
    else:
        raise ValueError('Unknown boundary state')


def _write(root, calibration, identity):
    root=Path(root)
    ids=calibration.frozen_identity['sample_ids']
    count=len(ids)
    if len(set(ids))!=count or any(len(values)!=count for values in
            (calibration.inputs,calibration.dataset,calibration.position_ids,calibration.block_states)):
        raise ValueError('Boundary document inventory mismatch')
    root.mkdir(parents=True,exist_ok=False)
    records=[]
    for i,(sample_id,hidden,tokens,positions,state) in enumerate(zip(ids,calibration.inputs,
            calibration.dataset,calibration.position_ids,calibration.block_states)):
        if not torch.equal(positions.cpu(),state.positions.cpu()):
            raise ValueError('Boundary positions differ from model state')
        values=dict(hidden=hidden,input_ids=tokens,positions=positions,
                    rope_cos=state.rope[0],rope_sin=state.rope[1])
        if state.mask is not None:
            values['mask']=state.mask
        if isinstance(state,LayerState):
            metadata=dict(kind='main',next_layer=state.next_layer)
            if state.topk is not None:
                values['topk']=state.topk
        elif isinstance(state,MTPState):
            metadata=dict(kind='mtp',done=state.done)
            values['embeddings']=state.embeddings
        else:
            raise TypeError('Unknown boundary state type')
        values={name:value.detach().cpu().contiguous().clone() for name,value in values.items()}
        validate(values,metadata)
        filename=f'doc-{i:05d}.safetensors'
        target=root/filename
        temporary=root/(filename+'.tmp')
        save_file(values,temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        checksum=digest(temporary)
        temporary.replace(target)
        records.append(dict(id=sample_id,file=filename,sha256=checksum,state=metadata,
                            keys=sorted(values)))
    state=dict(format='hy4-hf-boundary-v1',identity=identity,
               frozen_identity=calibration.frozen_identity,documents=records,complete=True)
    atomic_json(root/'commit.json',state)
    return state


def save(root, calibration, identity):
    """Publish a whole rank boundary, and verify identical completed retries."""
    root=Path(root)
    if root.is_symlink():
        raise ValueError('Boundary root is a symlink')
    if root.exists():
        recovered=load(root,identity,calibration.frozen_identity['sample_ids'])
        if recovered['frozen_identity']!=calibration.frozen_identity:
            raise ValueError('Recovered boundary data metadata changed')
        for name in ('inputs','dataset','position_ids'):
            left,right=recovered[name],getattr(calibration,name)
            if len(left)!=len(right) or any(not torch.equal(a.cpu(),b.cpu()) for a,b in zip(left,right)):
                raise ValueError('Recomputed boundary differs: '+name)
        for a,b in zip(recovered['block_states'],calibration.block_states):
            if type(a) is not type(b):
                raise ValueError('Recomputed boundary state kind differs')
            fields=('positions','mask','rope','topk','next_layer') if isinstance(a,LayerState) else ('positions','mask','rope','embeddings','done')
            def same(x,y):
                if isinstance(x,torch.Tensor):
                    return isinstance(y,torch.Tensor) and torch.equal(x.cpu(),y.cpu())
                if isinstance(x,tuple):
                    return isinstance(y,tuple) and len(x)==len(y) and all(same(u,v) for u,v in zip(x,y))
                return x==y
            if any(not same(getattr(a,key),getattr(b,key)) for key in fields):
                raise ValueError('Recomputed model state differs')
        return json.loads((root/'commit.json').read_text())
    temporary=root.parent/('.'+root.name+'.partial-'+uuid.uuid4().hex)
    state=_write(temporary,calibration,identity)
    # Failed attempts stay isolated for diagnosis. Only a complete directory
    # appears at the normal rank path; no deletion of partial/user files.
    temporary.rename(root)
    return state


def load(root, expected_identity, expected_ids):
    root=Path(root).resolve()
    commit=json.loads((root/'commit.json').read_text())
    if commit.get('format')!='hy4-hf-boundary-v1' or commit.get('complete') is not True:
        raise ValueError('Incomplete or unknown boundary format')
    if commit['identity']!=expected_identity:
        raise ValueError('Boundary source/recipe/partition identity changed')
    records=commit['documents']
    if [r['id'] for r in records]!=list(expected_ids) or len(set(expected_ids))!=len(expected_ids):
        raise ValueError('Boundary sample order/membership changed')
    if commit['frozen_identity']['sample_ids']!=list(expected_ids):
        raise ValueError('Boundary frozen sample metadata differs')
    inputs,dataset,positions,states=[],[],[],[]
    for i,record in enumerate(records):
        if record['file']!=f'doc-{i:05d}.safetensors':
            raise ValueError('Unexpected boundary filename')
        path=(root/record['file']).resolve()
        if not path.is_relative_to(root) or digest(path)!=record['sha256']:
            raise ValueError('Boundary file escaped root or checksum mismatch')
        values=load_file(path,device='cpu')
        if sorted(values)!=record['keys']:
            raise ValueError('Boundary tensor inventory changed')
        metadata=record['state']
        validate(values,metadata)
        rope=(values['rope_cos'],values['rope_sin'])
        if metadata['kind']=='main':
            state=LayerState(values['positions'],values.get('mask'),rope,
                             values.get('topk'),metadata['next_layer'])
        elif metadata['kind']=='mtp':
            state=MTPState(values['embeddings'],values['positions'],values.get('mask'),rope,metadata['done'])
        else:
            raise ValueError('Unknown serialized state kind')
        inputs.append(values['hidden']); dataset.append(values['input_ids'])
        positions.append(values['positions']); states.append(state)
    return dict(inputs=inputs,dataset=dataset,position_ids=positions,block_states=states,
                frozen_identity=commit['frozen_identity'],num_seq_per_rank=len(records))
