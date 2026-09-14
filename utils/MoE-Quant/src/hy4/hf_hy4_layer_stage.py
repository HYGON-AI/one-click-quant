"""Quantize/save stage called from native quant.py's existing layer loop."""
import hashlib
import json
from pathlib import Path
import torch.distributed as dist
from .checkpoint_writer import digest, atomic_json
from .hf_hy4_artifact_writer import ArtifactWriter
from .hf_hy4_quantize_distributed import quantize_distributed


def quantize_and_save(handles, args, calibration, block_idx, rank, world_size):
    if not args.save_dir or not calibration.frozen_identity:
        raise ValueError('Frozen calibration and explicit output directory required')
    source = Path(args.model_name_or_path).resolve()
    output = Path(args.save_dir).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Quantization output must be separate from the source tree')
    files = ['hf_hy4_collect.py','hf_hy4_quantize_handle.py','moe_gptq_adapter.py',
             'hf_hy4_quantize_distributed.py','hf_hy4_distributed_hessian.py',
             'hf_hy4_artifact_writer.py','hf_hy4_layer_stage.py']
    directory = Path(__file__).parent
    recipe = dict(format='hy4-readme-native-recipe-v1',bits=4,activation_bits=8,
        range=[-7,7],activation_range=[-127,127],scale_dtype='float32',
        granularity='output-channel',block_size=128,damping=[.01,.03,.1],
        scope='main_mtp_routed_shared',propagation='weight_only_float_simulation',
        world_size=world_size,owner_partition='sorted-projection-mod-world',
        calibration_manifest_sha256=calibration.frozen_identity['manifest_sha256'],
        implementation={name:digest(directory/name) for name in files})
    recipe_hash=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    identity=dict(format='hy4-readme-native-layer-v1',layer=block_idx,
        recipe_sha256=recipe_hash,
        source_index_sha256=digest(source/'model.safetensors.index.json'),
        source_config_sha256=digest(source/'config.json'),
        manifest_sha256=calibration.frozen_identity['manifest_sha256'])
    root=output/f'layer{block_idx}'/f'rank{rank}'
    writer=None
    error=None
    try:
        writer=ArtifactWriter(root,identity,handles,rank,world_size,recipe_hash)
    except Exception as failure:
        error=repr(failure)
    errors=[None]*world_size
    dist.all_gather_object(errors,error)
    if any(errors):
        raise RuntimeError('Layer writer setup failed: '+repr(errors))
    result=quantize_distributed(handles,writer)
    error=None
    try:
        writer.finish()
        atomic_json(root/'native-recipe.json',recipe)
    except Exception as failure:
        error=repr(failure)
    dist.all_gather_object(errors,error)
    if any(errors):
        raise RuntimeError('Layer shard commit failed; do not propagate: '+repr(errors))
    return result
