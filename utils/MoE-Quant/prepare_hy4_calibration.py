"""Freeze user-supplied, licensed four-domain JSONL texts for Hy4 calibration.

Input: zh/en/math/code.jsonl ({id, text}) and sources.json mapping each domain
to {dataset, revision, split, license}. No network downloads or evaluation data.
"""
import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--per-domain', type=int, default=32)
    p.add_argument('--max-length', type=int, default=2048)
    a = p.parse_args()
    model, source, out = map(Path, (a.model, a.input, a.output))
    if a.per_domain <= 0 or a.per_domain % 8 or a.max_length < 2:
        raise ValueError('per-domain must be a positive multiple of 8; length >= 2')
    if out.exists():
        raise ValueError('Use a fresh output directory; existing corpus is immutable')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    provenance = json.loads((source / 'sources.json').read_text(encoding='utf-8'))
    manifest = dict(format='hy4-calibration-manifest-v1', scope='calibration',
                    samples=[], sources=[], max_length=a.max_length,
                    selection='first unique documents in supplied order; explicit token truncation')
    for name, field in [('config.json', 'source_config_sha256'),
                        ('model.safetensors.index.json', 'source_index_sha256'),
                        ('tokenizer.json', 'tokenizer_file_sha256')]:
        manifest[field] = digest((model / name).read_bytes())
    corpus, seen = {}, set()
    for domain in ('zh', 'en', 'math', 'code'):
        meta = provenance[domain]
        if not all(meta.get(k) for k in ('dataset', 'revision', 'split', 'license')):
            raise ValueError('Source version, split and license required: ' + domain)
        records, identifiers = [], set()
        with (source / (domain + '.jsonl')).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                rid, text = str(row['id']), row['text']
                normalized = ' '.join(unicodedata.normalize('NFKC', text).split())
                h = digest(normalized.encode())
                if not normalized or h in seen or rid in identifiers:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)[:a.max_length]
                if len(ids) < 2:
                    continue
                n = len(records)
                records.append(dict(id=rid, text=text, input_ids=ids))
                manifest['samples'].append(dict(id=domain + ':' + rid, domain=domain,
                    source_record=n, rank=n % 4, tokens=len(ids),
                    normalized_text_sha256=h,
                    input_ids_sha256=digest(json.dumps(ids, separators=(',', ':')).encode())))
                identifiers.add(rid)
                seen.add(h)
                if len(records) == a.per_domain:
                    break
        if len(records) != a.per_domain:
            raise ValueError('Insufficient unique documents: ' + domain)
        payload = json.dumps(dict(records=records), ensure_ascii=False).encode()
        corpus[domain] = payload
        manifest['sources'].append(dict(meta, domain=domain, file_sha256=digest(payload)))
    out.mkdir(parents=True, exist_ok=False)
    for domain, payload in corpus.items():
        (out / (domain + '.json')).write_bytes(payload)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(documents=len(manifest['samples']),
                          tokens=sum(s['tokens'] for s in manifest['samples']),
                          manifest=str(out / 'manifest.json'))))


if __name__ == '__main__':
    main()
