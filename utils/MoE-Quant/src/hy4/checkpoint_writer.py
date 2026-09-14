"""Bounded shard writer: verified recovery, atomic index, explicit completeness."""
import hashlib
import json
import os
from pathlib import Path
import torch
from safetensors.torch import save_file
from safetensors import safe_open

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
        # Checksumming multi-hundred-GiB Hessians must not retain all their
        # reclaimable pages inside this container's conservative memory guard.
        if hasattr(os,'posix_fadvise'):
            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return h.hexdigest()

def atomic_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(data,f,indent=2); f.flush(); os.fsync(f.fileno())
    temp.replace(path)

class CheckpointWriter:
    def __init__(self,root,identity,expected,max_shard_bytes=512*(1<<20)):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.expected=expected; self.limit=max_shard_bytes
        self.pending={}; self.records={}; self.bytes=0
        self.path=self.root/'hy4-checkpoint.index.json'
        contract=dict(format='hy4_w4a8_v1',identity=identity,expected=expected)
        fingerprint=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if self.state['fingerprint']!=fingerprint:
                raise ValueError('Source, recipe or expected parameters changed')
            self.verify()
        else:
            if list(self.root.iterdir()):
                raise ValueError('Nonempty output without a verified checkpoint index')
            self.state=dict(contract,fingerprint=fingerprint,shards=[],parameters={},complete=False,
                            accuracy='NOT_EVALUATED')
            atomic_json(self.path,self.state)

    def verify(self):
        all_keys=set()
        for shard in self.state['shards']:
            path=(self.root/shard['file']).resolve()
            if not path.is_relative_to(self.root.resolve()) or digest(path)!=shard['sha256']:
                raise ValueError('Checkpoint shard checksum mismatch')
            with safe_open(path,framework='pt',device='cpu') as f:
                if set(f.keys())!=set(shard['keys']):
                    raise ValueError('Checkpoint shard tensor set mismatch')
                if all_keys.intersection(f.keys()):
                    raise ValueError('Duplicate checkpoint tensor key')
                all_keys.update(f.keys())
        wanted={key for record in self.state['parameters'].values() for key in record['keys']}
        if wanted!=all_keys or not set(self.state['parameters'])<=set(self.expected):
            raise ValueError('Checkpoint index coverage mismatch')

    def contains(self,name):
        return name in self.state['parameters'] or name in self.records

    def add(self,name,tensors,metadata):
        if name not in self.expected or self.contains(name):
            raise ValueError('Unknown or duplicate source parameter')
        kind=self.expected[name]
        if kind=='quantized':
            if set(tensors)!={'packed','scale'} or tensors['packed'].dtype!=torch.uint8 or tensors['scale'].dtype!=torch.float32:
                raise ValueError('Expected packed INT4 and FP32 scales')
            rows,cols=metadata['shape']
            packed=tensors['packed']; scale=tensors['scale']
            if packed.shape!=(rows,(cols+1)//2) or scale.shape!=(rows,1):
                raise ValueError('Quantized shape mismatch')
            if ((packed&15)==8).any() or ((packed>>4)==8).any() or not torch.isfinite(scale).all() or (scale<=0).any():
                raise ValueError('Invalid narrow INT4 payload')
            if cols%2 and (packed[:,-1]>>4).any():
                raise ValueError('Nonzero packing padding')
            for field in ('source_sha256','recipe_sha256','algorithm','coverage'):
                if field not in metadata: raise ValueError('Missing projection evidence: '+field)
        elif kind=='retained':
            if set(tensors)!={'weight'}: raise ValueError('Retained tensor required')
        else: raise ValueError('Unknown parameter kind')
        size=sum(t.numel()*t.element_size() for t in tensors.values())
        if size>self.limit:
            raise MemoryError('Projection exceeds configured shard budget')
        if self.bytes+size>self.limit: self.flush()
        keys=[]
        for suffix,tensor in tensors.items():
            key=name+'.'+suffix
            keys.append(key); self.pending[key]=tensor.detach().cpu().contiguous().clone()
        self.records[name]=dict(kind=kind,keys=keys,metadata=metadata)
        self.bytes+=size

    def flush(self):
        if not self.pending: return
        filename=f'shard-{len(self.state["shards"]):05d}.safetensors'
        path=self.root/filename
        if path.exists():
            raise ValueError('Uncommitted shard exists; preserve and inspect before recovery')
        tmp=self.root/(filename+'.tmp')
        save_file(self.pending,str(tmp))
        with tmp.open('rb+') as f: os.fsync(f.fileno())
        with safe_open(tmp,framework='pt',device='cpu') as f:
            if set(f.keys())!=set(self.pending): raise ValueError('Shard write mismatch')
        checksum=digest(tmp); tmp.replace(path)
        self.state['shards'].append(dict(file=filename,sha256=checksum,keys=sorted(self.pending)))
        self.state['parameters'].update(self.records)
        atomic_json(self.path,self.state)
        self.pending={}; self.records={}; self.bytes=0

    def finish(self):
        self.flush(); self.verify()
        missing=set(self.expected)-set(self.state['parameters'])
        if missing: raise ValueError(f'Incomplete checkpoint: {len(missing)} missing parameters')
        self.state['complete']=True
        atomic_json(self.path,self.state)
