"""Frozen corpus token loader, preserving legacy balanced eight-rank split."""
import hashlib
import json
import unicodedata
from pathlib import Path
import torch
from .calibration_subpartition import split_samples


def sha(value):
    return hashlib.sha256(value).hexdigest()


def load(manifest_path, model_path, corpus_path, rank, world_size, count, max_length):
    if world_size not in (1,8) or not 0 <= rank < world_size:
        raise ValueError('Unsupported frozen calibration rank layout')
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    if manifest.get('format')!='hy4-calibration-manifest-v1' or manifest.get('scope')!='calibration':
        raise ValueError('Wrong calibration manifest')
    model = Path(model_path)
    for filename, field in (('config.json','source_config_sha256'),
                            ('model.safetensors.index.json','source_index_sha256'),
                            ('tokenizer.json','tokenizer_file_sha256')):
        if sha((model/filename).read_bytes()) != manifest[field]:
            raise ValueError('Frozen model/tokenizer identity changed: '+filename)
    samples = manifest['samples']
    if len(samples)!=count or len({s['id'] for s in samples})!=count:
        raise ValueError('Calibration count or unique sample identity mismatch')
    partitions = [samples] if world_size==1 else [split_samples(samples,r//2,r%2) for r in range(8)]
    flattened = [sample['id'] for part in partitions for sample in part]
    if len(flattened)!=count or len(set(flattened))!=count:
        raise ValueError('Frozen rank partitions omit or duplicate samples')
    corpus = {}
    for source in manifest['sources']:
        domain = source['domain']
        if domain not in ('zh','en','math','code') or domain in corpus:
            raise ValueError('Unexpected corpus domain')
        payload = (Path(corpus_path)/(domain+'.json')).read_bytes()
        if sha(payload)!=source['file_sha256']:
            raise ValueError('Frozen corpus changed: '+domain)
        corpus[domain] = json.loads(payload)['records']
    vocab_size = json.loads((model/'config.json').read_text())['vocab_size']
    tensors = []
    selected = partitions[rank]
    for sample in selected:
        record = corpus[sample['domain']][sample['source_record']]
        if sample['id'] != sample['domain']+':'+str(record['id']):
            raise ValueError('Corpus sample ID mismatch')
        ids = record['input_ids']
        normalized = ' '.join(unicodedata.normalize('NFKC',record['text']).split())
        if sha(normalized.encode())!=sample['normalized_text_sha256']:
            raise ValueError('Sample text changed')
        if (len(ids)!=sample['tokens'] or not 1 <= len(ids) <= max_length
                or any(type(t)!=int or not 0 <= t < vocab_size for t in ids)):
            raise ValueError('Invalid frozen token IDs; refusing truncation')
        if sha(json.dumps(ids,separators=(',',':')).encode())!=sample['input_ids_sha256']:
            raise ValueError('Frozen token sequence changed')
        tensors.append(torch.tensor(ids,dtype=torch.int64).unsqueeze(0))
    return tensors, dict(manifest_sha256=sha(raw), sample_ids=[s['id'] for s in selected],
        rank=rank, world_size=world_size, partition_counts=[len(p) for p in partitions],
        tokens=sum(t.numel() for t in tensors), retokenized=False)
