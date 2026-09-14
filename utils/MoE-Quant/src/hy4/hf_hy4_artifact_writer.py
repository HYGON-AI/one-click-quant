"""Native per-rank layer writer using the existing versioned packed format."""
import hashlib
import torch
from safetensors import safe_open
from .checkpoint_writer import CheckpointWriter


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


class ArtifactWriter:
    def __init__(self, root, identity, handles, rank, world_size, recipe_sha256):
        if not 0 <= rank < world_size:
            raise ValueError('Invalid writer rank')
        self.handles = handles
        self.recipe_sha256 = recipe_sha256
        self.names = {name for i,name in enumerate(sorted(handles)) if i%world_size==rank}
        self.writer = CheckpointWriter(root, dict(identity,rank=rank,world_size=world_size),
                                      {name+'.weight':'quantized' for name in self.names})

    def __call__(self, name, artifact):
        if name not in self.names:
            raise ValueError('Projection assigned to another writer')
        # Called before candidate commit: hash the actual original projection,
        # not the source index and not its already-quantized replacement.
        metadata = dict(shape=artifact['shape'], algorithm=artifact['algorithm'],
            source_sha256=tensor_sha256(self.handles[name].layer.weight),
            source_hash_scope='loaded_original_projection_tensor_bytes',
            recipe_sha256=self.recipe_sha256, coverage=artifact['seen'],
            fallback=artifact['fallback'], damping=artifact['damp'],
            unobserved_columns=artifact['unobserved_columns'], failures=artifact['failures'])
        tensors = {key:artifact[key] for key in ('packed','scale')}
        key = name+'.weight'
        if self.writer.contains(key):
            self._verify_existing(key,tensors,metadata)
            return
        self.writer.add(key,tensors,metadata)

    def _verify_existing(self, name, tensors, metadata):
        record = self.writer.state['parameters'].get(name)
        if record is None:
            raise ValueError('Duplicate uncommitted artifact')
        if record['metadata'] != metadata:
            raise ValueError('Recovered projection provenance differs')
        locations = {key:shard['file'] for shard in self.writer.state['shards'] for key in shard['keys']}
        for suffix, value in tensors.items():
            key = name+'.'+suffix
            with safe_open(self.writer.root/locations[key],framework='pt',device='cpu') as handle:
                if not torch.equal(handle.get_tensor(key), value.detach().cpu()):
                    raise ValueError('Recovered projection payload differs')

    def finish(self):
        self.writer.finish()
