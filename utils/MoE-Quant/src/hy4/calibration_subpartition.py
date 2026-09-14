"""Split legacy data ranks without changing the frozen manifest or sample IDs.

This is scheduling metadata only. Subpartitions must be recombined and verified
before producing a legacy rank commit; they cannot masquerade as full ranks.
"""
import hashlib
import json


def split_samples(samples, rank, part, parts=2):
    if parts not in (1, 2) or not 0 <= part < parts or not 0 <= rank < 4:
        raise ValueError('Invalid bounded calibration subpartition')
    ids = [sample['id'] for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate calibration sample IDs')
    if any(sample['rank'] not in range(4) for sample in samples):
        raise ValueError('Unexpected frozen manifest rank')
    parent = [sample for sample in samples if sample['rank'] == rank]
    # Balance each domain separately; preserve manifest ordering in each part.
    counters = {}
    selected = []
    for sample in parent:
        domain = sample['domain']
        index = counters.get(domain, 0)
        counters[domain] = index + 1
        if index % parts == part:
            selected.append(sample)
    return selected


def identity(samples, rank, part, parts=2):
    selected = split_samples(samples, rank, part, parts)
    ids = [sample['id'] for sample in selected]
    return dict(format='hy4-calibration-subpartition-v1', parent_rank=rank,
                part=part, parts=parts, samples=len(ids),
                ordered_ids_sha256=hashlib.sha256(json.dumps(ids,
                    ensure_ascii=False, separators=(',', ':')).encode()).hexdigest())


def verify_membership(samples, rank, committed_ids):
    """Reject omissions/duplicates even when set union appears complete."""
    if len(committed_ids) not in (1, 2):
        raise ValueError('Unexpected number of subpartition commits')
    actual = []
    for part, ids in enumerate(committed_ids):
        expected = [s['id'] for s in split_samples(samples, rank, part, len(committed_ids))]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise ValueError('Wrong subpartition membership')
        actual.extend(ids)
    parent = [s['id'] for s in samples if s['rank'] == rank]
    if len(actual) != len(set(actual)) or set(actual) != set(parent):
        raise ValueError('Incomplete or overlapping parent rank')
