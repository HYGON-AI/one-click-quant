"""Real GPTQ bank grouping vs independent per-expert integer projections."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from src.hy4.checkpoint_loader import CheckpointLoader
from src.hy4.checkpoint_writer import atomic_json
from src.hy4.runtime.sglang_bank_view import PackedBankView
from src.hy4.runtime.hy4_gptq_resident import Bank
from src.hy4.runtime.hy4_gptq_kernel import projection
from src.hy4.runtime.hy4_gptq_grouped import routed, integer_evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260911)
    root = Path(a.candidate)
    state = json.loads((root/'hy4-checkpoint.index.json').read_text())
    loader = CheckpointLoader(root, state['identity'], state['expected'])
    view = PackedBankView(loader)
    name = next(n for n in state['parameters'] if '.experts.0.gate_proj.weight' in n)
    prefix = name.split('.experts.0.')[0]
    gate = Bank(view, prefix+'.experts.gate_up_proj', 'cuda', slice(0, 2))
    down = Bank(view, prefix+'.experts.down_proj', 'cuda', slice(0, 2))
    x = torch.randn(4, gate.packed.shape[-1]*2, device='cuda', dtype=torch.bfloat16)
    x[0].zero_()
    x[3].copy_(x[1])
    ids = torch.tensor([[0,1,2,3,4,5,6,7], [1,3,4,5,6,7,8,9],
                        [2,3,4,5,6,7,8,9], [1,0,2,3,4,5,6,7]],
                       device='cuda', dtype=torch.int32)
    weights = torch.softmax(torch.randn(4, 8, device='cuda'), -1)
    reports = []
    for limit in (0., 7.):
        actual = routed(x, ids, weights, gate, down, 0, limit)
        ref = torch.zeros_like(actual)
        # Preserve top-k accumulation order, including routes owned elsewhere.
        for slot in range(8):
            for expert in (0, 1):
                mask = ids[:, slot] == expert
                if not mask.any():
                    continue
                gu = projection(x[mask], gate.packed[expert], gate.scale[expert])
                g, u = gu.chunk(2, -1)
                if limit:
                    g, u = g.clamp(max=limit), u.clamp(-limit, limit)
                mid = F.silu(g)*u
                y = projection(mid, down.packed[expert], down.scale[expert])
                ref[mask] += y.float()*weights[mask, slot, None]
        error = (actual-ref).abs().max().item()
        assert torch.equal(actual, ref), f'Grouped flow differs: {error}'
        assert not actual[2].count_nonzero(), 'Nonlocal-only row must contribute zero'
        reports.append(dict(limit=limit, exact=True, max_abs_error=error))
    assert integer_evidence()['accumulator_bits']==32
    atomic_json(Path(a.report), dict(status='PASS', scope='TWO_REAL_EXPERTS_FIXED_TOP8_LOCAL_CONTRIBUTION',
        integer_backend=integer_evidence(),
        cases=reports, zero_token=True, repeated_input=True, nonlocal_only_row=True,
        full_native_router='NOT_EVALUATED', full_model='NOT_EVALUATED'))
    print('REAL_GPTQ_GROUPED_FLOW_EXACT')


if __name__ == '__main__': main()
