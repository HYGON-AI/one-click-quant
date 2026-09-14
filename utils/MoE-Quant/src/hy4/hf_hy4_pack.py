"""Independent native-entry pack stage for the versioned Hy4 W4A8 runtime."""
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
from .checkpoint_writer import CheckpointWriter, atomic_json, digest
from .checkpoint_loader import CheckpointLoader
from .hf_hy4_layer_commit import latest
from .hy4_source import Hy4Source
from .sglang_full_inventory import full_inventory
from .hf_hy4_pack_provenance import validate_manifest,write as write_provenance


def append_rank(writer, path, record, expected, run_identity, layer, rank,commit_index=True):
    """Verify a committed rank and import all its shards as one index update."""
    path=Path(path).resolve()
    index=path/'hy4-checkpoint.index.json'
    if digest(index)!=record['checkpoint_sha256']:
        raise ValueError('Rank index differs from global layer commit')
    state=json.loads(index.read_text()); ci=state['identity']
    required=dict(layer=layer,rank=rank,world_size=run_identity['world_size'],
        source_index_sha256=run_identity['source_index_sha256'],
        source_config_sha256=run_identity['source_config_sha256'],
        manifest_sha256=run_identity['manifest_sha256'])
    if any(ci.get(key)!=value for key,value in required.items()):
        raise ValueError('Native rank source/data/partition mismatch')
    if ci.get('format')!='hy4-readme-native-layer-v1':
        raise ValueError('Not a native-loop layer checkpoint')
    recipe=json.loads((path/'native-recipe.json').read_text())
    recipe_hash=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    if recipe_hash!=ci.get('recipe_sha256'):
        raise ValueError('Native recipe differs from quantization index')
    for key,value in dict(bits=4,activation_bits=8,range=[-7,7],activation_range=[-127,127],
        scale_dtype='float32',granularity='output-channel',scope='main_mtp_routed_shared').items():
        if recipe.get(key)!=value:
            raise ValueError('Wrong W4A8 quantization contract: '+key)
    loader=CheckpointLoader(path,ci,expected)
    for name,item in loader.state['parameters'].items():
        metadata=item['metadata']
        if (item['kind']!='quantized' or metadata['recipe_sha256']!=recipe_hash
                or type(metadata['coverage'])!=int or metadata['coverage']<0
                or metadata['algorithm'] not in ('MoE-Quant-GPTQ','RTN')):
            raise ValueError('Invalid projection provenance: '+name)
    overlap=set(expected)&set(writer.state['parameters'])
    if overlap:
        if overlap!=set(expected):
            raise ValueError('Partially imported rank index')
        if any(writer.state['parameters'][name]!=loader.state['parameters'][name] for name in expected):
            raise ValueError('Previously imported rank changed')
        return recipe_hash,recipe
    imported=[]
    for shard in loader.state['shards']:
        filename=f'layer{layer:02d}-rank{rank}-'+shard['file']
        destination=writer.root/filename
        source=path/shard['file']
        if destination.exists():
            if digest(destination)!=shard['sha256']:
                raise ValueError('Uncommitted import has wrong payload')
        else:
            try:
                os.link(source,destination)
            except OSError as error:
                if error.errno!=errno.EXDEV:
                    raise
                if shutil.disk_usage(writer.root).free<300*(1<<30)+source.stat().st_size:
                    raise OSError('Insufficient reserve for cross-filesystem shard copy')
                temporary=destination.with_suffix(destination.suffix+'.tmp')
                shutil.copyfile(source,temporary)
                with temporary.open('rb') as stream:
                    os.fsync(stream.fileno())
                if digest(temporary)!=shard['sha256']:
                    raise ValueError('Imported shard copy failed verification')
                temporary.replace(destination)
        imported.append(dict(shard,file=filename))
    # Crashes before this commit leave only verifiable uncommitted links/copies.
    writer.state['shards'].extend(imported)
    writer.state['parameters'].update(loader.state['parameters'])
    if commit_index:
        atomic_json(writer.path,writer.state)
    return recipe_hash,recipe


def pack(args):
    source_root=Path(args.model_name_or_path).resolve()
    pipeline=Path(args.quantized_model_path).resolve()
    output=Path(args.packed_model_path).resolve()
    for protected in (source_root,pipeline):
        if output==protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Packed output must not overlap source or native candidates')
    if args.activation_bits!=8 or args.dtype!='bfloat16':
        raise ValueError('Hy4 custom pack requires --activation-bits 8 --dtype bfloat16')
    config=json.loads((source_root/'config.json').read_text())
    source_index=json.loads((source_root/'model.safetensors.index.json').read_text())
    expected,physical=full_inventory(config,source_index)
    run_path=pipeline/'native-run.json'
    run_identity=json.loads(run_path.read_text())
    if run_identity.get('format')!='hy4-native-run-v1':
        raise ValueError('Unknown native run format')
    world=run_identity['world_size']
    if world not in (1,8):
        raise ValueError('Unsupported native partition layout')
    for filename,key in (('config.json','source_config_sha256'),
                          ('model.safetensors.index.json','source_index_sha256')):
        if digest(source_root/filename)!=run_identity[key]:
            raise ValueError('Source identity changed before packing')
    committed=latest(pipeline,run_identity,world)
    total=config['num_hidden_layers']+config['num_nextn_predict_layers']
    next_layer=0 if committed is None else committed['layer']+1
    if args.preflight:
        return dict(ready=next_layer==total,committed_layers=next_layer,total_layers=total,
            quantized_projections=sum(kind=='quantized' for kind in expected.values()),
            retained_tensors=sum(kind=='retained' for kind in expected.values()),accuracy='NOT_EVALUATED')
    if next_layer!=total:
        raise ValueError(f'Incomplete native main/MTP conversion: {next_layer}/{total} layers')
    if not args.hy4_target_manifest:
        raise ValueError('Explicit --hy4-target-manifest required for bounded source reader')
    validate_manifest(args.hy4_calibration_manifest,run_identity)
    source=Hy4Source(source_root,args.hy4_target_manifest)
    if set(source.targets)!={name for name,kind in expected.items() if kind=='quantized'}:
        raise ValueError('Target manifest does not match independent full-model inventory')
    if {item['source'] for item in source.targets.values()}!=physical:
        raise ValueError('Target manifest physical banks differ from model inventory')
    commit_hashes={str(layer):digest(pipeline/'commits'/f'layer-{layer:03d}.json') for layer in range(total)}
    identity=dict(scope='FULL_MAIN_AND_MTP_EXPERTS',native_run_sha256=digest(run_path),
        native_layer_commits=commit_hashes,source_index_sha256=run_identity['source_index_sha256'],
        source_config_sha256=run_identity['source_config_sha256'],
        calibration_manifest_sha256=run_identity['manifest_sha256'],
        packing='native-loop-rank-shards-to-hy4_w4a8_v1',accuracy='NOT_EVALUATED',
        packer_implementation={name:digest(Path(__file__).parent/name) for name in
            ('hf_hy4_pack.py','hf_hy4_pack_provenance.py','checkpoint_writer.py','checkpoint_loader.py','hy4_source.py',
             'sglang_full_inventory.py','hf_hy4_layer_commit.py')})
    writer=CheckpointWriter(output,identity,expected,max_shard_bytes=2<<30)
    recipes={}
    for layer in range(total):
        prefix=source.block_prefix(layer)+'.'
        names=sorted(name for name,kind in expected.items() if kind=='quantized' and name.startswith(prefix))
        layer_commit=json.loads((pipeline/'commits'/f'layer-{layer:03d}.json').read_text())
        for rank,record in enumerate(layer_commit['ranks']):
            relative=f'layer{layer}/rank{rank}/hy4-checkpoint.index.json'
            if record['checkpoint']!=relative or record['rank']!=rank:
                raise ValueError('Unexpected committed rank path')
            wanted={name:'quantized' for number,name in enumerate(names) if number%world==rank}
            recipe_hash,recipe=append_rank(writer,pipeline/f'layer{layer}'/f'rank{rank}',record,wanted,run_identity,layer,rank,commit_index=False)
            if recipe_hash in recipes and recipes[recipe_hash]!=recipe:
                raise ValueError('Conflicting quantization recipes')
            recipes[recipe_hash]=recipe
        # All ranks of this layer enter the large global index atomically.
        # Avoid rewriting a 60k-parameter JSON index once per individual rank.
        atomic_json(writer.path,writer.state)
        print(f'PACK_IMPORTED layer={layer} projections={len(names)}',flush=True)
    for name,kind in sorted(expected.items()):
        if kind!='retained' or writer.contains(name):
            continue
        if shutil.disk_usage(output).free<302*(1<<30):
            raise OSError('Disk reserve reached')
        value=source.tensor(name) # Original source dtype, no blanket BF16 cast.
        checksum=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        writer.add(name,{'weight':value},dict(source_sha256=checksum,shape=list(value.shape),dtype=str(value.dtype)))
        del value
    writer.flush()
    assets=write_provenance(output,pipeline,run_identity,writer.state['parameters'],recipes,args.hy4_calibration_manifest)
    for name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                 'generation_config.json','chat_template.jinja','tokenizer.model',
                 'LICENSE','LICENSE.txt','NOTICE','README.md'):
        original=source_root/name
        if original.is_file():
            temporary=output/(name+'.tmp')
            shutil.copyfile(original,temporary); temporary.replace(output/name)
            assets[name]=digest(output/name)
    name='hy4-source-model.index.json'
    temporary=output/(name+'.tmp')
    shutil.copyfile(source_root/'model.safetensors.index.json',temporary)
    if digest(temporary)!=identity['source_index_sha256']:
        raise ValueError('Source index changed during asset copy')
    temporary.replace(output/name); assets[name]=identity['source_index_sha256']
    atomic_json(output/'hy4-assets.json',dict(files=assets,format='hy4_w4a8_v1',
        requires_custom_loader=True,main_and_mtp=True,accuracy='NOT_EVALUATED'))
    writer.finish()
    return dict(complete=True,format='hy4_w4a8_v1',accuracy='NOT_EVALUATED',integer_runtime='NOT_EVALUATED')
