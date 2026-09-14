"""Native-loop run identity and collective selection of committed boundaries."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
import transformers
from .checkpoint_writer import atomic_json, digest
from .hf_hy4_layer_commit import latest, restore_rank


def collective_errors(error, stage):
    errors=[None]*dist.get_world_size()
    dist.all_gather_object(errors,error)
    if any(errors):
        raise RuntimeError(stage+': '+repr(errors))


def prepare(args, calibration, main_layers, entry_file):
    """Return (identity, next_layer). Restore only a globally committed layer."""
    rank,world=dist.get_rank(),dist.get_world_size()
    source=Path(args.model_name_or_path).resolve()
    if not args.save_dir:
        raise ValueError('Hy4 requires a new explicit --save_dir')
    output=Path(args.save_dir).resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output overlaps source model')
    local_error=None
    identity=None
    try:
        from src import gptq, gptq_loop, quant_utils, linalg_utils
        directory=Path(__file__).parent
        files=sorted(directory.glob('hf_hy4_*.py'))
        files += [directory/name for name in ('moe_gptq_adapter.py','checkpoint_writer.py',
                  'format_contract.py','calibration_subpartition.py')]
        implementation={path.name:digest(path) for path in files}
        implementation['native_quant.py']=digest(entry_file)
        for module in (gptq,gptq_loop,quant_utils,linalg_utils):
            implementation[module.__name__]=digest(module.__file__)
        weight_map=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
        shards=[]
        for filename in sorted(set(weight_map.values())):
            path=(source/filename).resolve()
            if not path.is_relative_to(source):
                raise ValueError('Source shard outside read-only model root')
            stat=path.stat()
            shards.append(dict(file=filename,size=stat.st_size,mtime_ns=stat.st_mtime_ns))
        semantic_args=('bits','group_size','sym','dtype','quantize_scope','include_mtp',
            'activation_bits','weight_range','num_calibration_samples','max_sequence_length',
            'seed','rel_damp','block_size','quantization_scale','quantization_order','attn_implementation')
        identity=dict(format='hy4-native-run-v1',implementation=implementation,
            arguments={name:getattr(args,name,None) for name in semantic_args},
            torch_version=torch.__version__,transformers_version=transformers.__version__,
            world_size=world,main_layers=main_layers,
            source_config_sha256=digest(source/'config.json'),
            source_index_sha256=digest(source/'model.safetensors.index.json'),
            source_shard_stat_inventory=shards,
            source_payload_verification='per-projection hash on export; stat binding on resume, not full source SHA verification',
            manifest_sha256=calibration.frozen_identity['manifest_sha256'],
            accuracy='NOT_EVALUATED',integer_runtime='NOT_EVALUATED')
    except Exception as failure:
        local_error=repr(failure)
    collective_errors(local_error,'Run identity preparation failed')
    identities=[None]*world
    dist.all_gather_object(identities,identity)
    if any(other!=identity for other in identities):
        raise ValueError('Rank source/code/recipe identities differ')
    partitions=[None]*world
    dist.all_gather_object(partitions,calibration.frozen_identity['sample_ids'])
    flat=[sample for part in partitions for sample in part]
    if len(set(flat))!=len(flat):
        raise ValueError('Calibration partitions overlap')
    identity['partitions']=partitions
    local_error=None
    if rank==0:
        try:
            manifest=output/'native-run.json'
            if manifest.exists():
                if not args.resume:
                    raise ValueError('Existing run requires explicit --resume')
                if json.loads(manifest.read_text())!=identity:
                    raise ValueError('Cannot resume changed source/data/code/recipe')
            else:
                if output.exists() and any(output.iterdir()):
                    raise ValueError('Nonempty output without a native run identity')
                output.mkdir(parents=True,exist_ok=True)
                atomic_json(manifest,identity)
        except Exception as failure:
            local_error=repr(failure)
    collective_errors(local_error,'Run manifest preparation failed')
    state=None
    local_error=None
    try:
        state=latest(output,identity,world)
        if state is not None:
            ids=calibration.frozen_identity['sample_ids']
            if state['layer']>=main_layers:
                ids=[name for name,tokens in zip(ids,calibration.dataset) if tokens.shape[1]>=2]
            restored=restore_rank(output,state,rank,ids)
            for field,value in restored.items():
                setattr(calibration,field,value)
    except Exception as failure:
        local_error=repr(failure)
    collective_errors(local_error,'Boundary restore failed')
    if state is not None and state['layer']>=2:
        from .hf_hy4_cache_retention import retire
        retire(output,state['layer'],identity,args.model_name_or_path)
    return identity,0 if state is None else state['layer']+1


def status(output,identity,next_layer,total_layers):
    """Layer conversion is distinct from a loadable final model or evaluation."""
    if dist.get_rank()==0:
        atomic_json(Path(output)/'native-status.json',dict(format='hy4-native-status-v1',
            next_layer=next_layer,total_layers=total_layers,
            layer_conversion_complete=next_layer==total_layers,
            checkpoint_packing='NOT_COMPLETED',integer_runtime='NOT_EVALUATED',
            accuracy='NOT_EVALUATED',manifest_sha256=identity['manifest_sha256']))
