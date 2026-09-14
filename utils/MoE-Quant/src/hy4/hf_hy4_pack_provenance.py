"""Self-contained recipe/data/coverage metadata for a packed candidate."""
from collections import Counter
import json
from pathlib import Path
import shutil
from .checkpoint_writer import atomic_json,digest


def validate_manifest(path,run):
    if not path or digest(path)!=run['manifest_sha256']:
        raise ValueError('Packing requires the exact frozen calibration manifest')
    data=json.loads(Path(path).read_text())
    if data.get('format')!='hy4-calibration-manifest-v1' or data.get('scope')!='calibration':
        raise ValueError('Wrong calibration manifest format/scope')
    ids=[sample['id'] for sample in data['samples']]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate calibration IDs')
    if 'partitions' in run:
        partitioned=[name for part in run['partitions'] for name in part]
        if len(partitioned)!=len(ids) or set(partitioned)!=set(ids):
            raise ValueError('Packed run and calibration sample inventories differ')
    return data


def write(output,pipeline,run,parameters,recipes,manifest_path):
    output=Path(output); pipeline=Path(pipeline)
    manifest=validate_manifest(manifest_path,run)
    if json.loads((pipeline/'native-run.json').read_text())!=run:
        raise ValueError('Native run identity changed during packing')
    layers={}; algorithms=Counter(); fallbacks=Counter(); hits=[]
    for name,record in parameters.items():
        if record['kind']!='quantized':
            continue
        metadata=record['metadata']
        if metadata['recipe_sha256'] not in recipes:
            raise ValueError('Missing packed projection recipe')
        prefix=name.split('.mlp.')[0]
        layer=layers.setdefault(prefix,dict(projections=0,algorithms=Counter(),fallbacks=Counter(),
            zero_coverage=0,below_256_tokens=0,minimum_coverage=None,maximum_coverage=0))
        hit=metadata['coverage']
        if type(hit)!=int or hit<0:
            raise ValueError('Invalid projection coverage')
        hits.append(hit); algorithms[metadata['algorithm']]+=1
        layer['projections']+=1; layer['algorithms'][metadata['algorithm']]+=1
        layer['zero_coverage']+=hit==0; layer['below_256_tokens']+=hit<256
        layer['minimum_coverage']=hit if layer['minimum_coverage'] is None else min(hit,layer['minimum_coverage'])
        layer['maximum_coverage']=max(hit,layer['maximum_coverage'])
        if metadata.get('fallback'):
            fallbacks[metadata['fallback']]+=1; layer['fallbacks'][metadata['fallback']]+=1
    coverage=dict(format='hy4-native-coverage-v1',projections=len(hits),algorithms=algorithms,
        fallbacks=fallbacks,zero_coverage=sum(hit==0 for hit in hits),
        below_256_tokens=sum(hit<256 for hit in hits),minimum_coverage=min(hits,default=None),
        calibration_documents=len(manifest['samples']),manifest_sha256=run['manifest_sha256'],
        layers=layers,accuracy='NOT_EVALUATED',integer_runtime='NOT_EVALUATED')
    files={}
    for source,name in ((pipeline/'native-run.json','hy4-native-run.json'),
                        (Path(manifest_path),'hy4-calibration-manifest.json')):
        temporary=output/(name+'.tmp')
        shutil.copyfile(source,temporary); temporary.replace(output/name)
        files[name]=digest(output/name)
    for name,value in (('hy4-quantization-recipes.json',dict(format='hy4-native-recipes-v1',recipes=recipes)),
                       ('hy4-coverage.json',coverage)):
        atomic_json(output/name,value); files[name]=digest(output/name)
    return files
