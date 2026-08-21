import os

import torch
from safetensors import safe_open

def load_param_shard(weight_dir: str, weight_path: str) -> dict[str, torch.Tensor]:
    param_shard = {}
    with safe_open(os.path.join(weight_dir, weight_path), framework="pt", device="cpu") as f:
        param_shard_keys = f.keys()
        for k in param_shard_keys:
            param_shard[k] = f.get_tensor(k)
    return param_shard
