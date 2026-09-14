"""Read the versioned shards after identity, coverage and checksum validation."""
import json
from pathlib import Path
from safetensors import safe_open
from .checkpoint_writer import CheckpointWriter

class CheckpointLoader:
    def __init__(self,root,identity,expected):
        self.root=Path(root)
        if not (self.root/'hy4-checkpoint.index.json').is_file():
            raise ValueError('Checkpoint index missing')
        verified=CheckpointWriter(self.root,identity,expected)
        self.state=verified.state
        if not self.state['complete'] or set(self.state['parameters'])!=set(expected):
            raise ValueError('Incomplete checkpoint')
        self.key_to_file={key:shard['file'] for shard in self.state['shards'] for key in shard['keys']}

    def projection(self,name):
        record=self.state['parameters'][name]
        if record['kind']!='quantized': raise ValueError('Not a quantized projection')
        result={}
        for key in record['keys']:
            with safe_open(self.root/self.key_to_file[key],framework='pt',device='cpu') as f:
                result[key[len(name)+1:]]=f.get_tensor(key).clone()
        return result,record['metadata']

    def retained(self,name):
        record=self.state['parameters'][name]
        if record['kind']!='retained': raise ValueError('Not a retained parameter')
        key=record['keys'][0]
        with safe_open(self.root/self.key_to_file[key],framework='pt',device='cpu') as f:
            return f.get_tensor(key).clone()
