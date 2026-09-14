"""Actual packed GPTQ projection vs exact integer reference on one free GPU."""
import argparse
import json
import re
from pathlib import Path
import torch
from src.hy4.checkpoint_loader import CheckpointLoader
from src.hy4.runtime.sglang_bank_view import PackedBankView
from src.hy4.runtime.hy4_gptq_kernel import quant, mm, unpack, COMPILED
from src.hy4.checkpoint_writer import atomic_json


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
    bank = PackedBankView(loader)
    first = next(n for n in state['parameters'] if '.experts.0.gate_proj.weight' in n)
    base = first.split('.experts.0.')[0]+'.experts.gate_up_proj'
    w = bank.tensor(base+'.int4_packed', 0)
    sw = bank.tensor(base+'.scale', 0)
    k = w.shape[-1]*2
    x = torch.randn(3, k, device='cuda', dtype=torch.bfloat16)
    x[0].zero_()
    qa, sa = quant(x, 127)
    reference = qa.cpu().to(torch.int32) @ unpack(w, k).to(torch.int32).T
    actual = mm(qa, w.cuda(), sa, sw.cuda(), raw=True)
    assert torch.equal(actual.cpu(), reference), 'Integer accumulation mismatch'
    out = mm(qa, w.cuda(), sa, sw.cuda())
    ref_scaled = (reference.cuda().float()*sa[:, None]*sw.cuda()[None, :]).bfloat16()
    assert torch.equal(out, ref_scaled), 'Scaling/output mismatch'
    ptx = '\n'.join(kernel.asm['ptx'] for kernel in COMPILED)
    instructions = sorted(set(re.findall(r'mma[^;\n]*s32[^;\n]*', ptx)))
    assert any('.s8.s8.' in line for line in instructions), 'Missing signed INT8 tensor dot evidence'
    atomic_json(Path(a.report), dict(status='PASS', scope='REAL_GPTQ_SINGLE_PROJECTION',
        candidate=str(root), projection=base, tokens=3, zero_token=True,
        integer_exact=True, scaled_bf16_exact=True, instructions=instructions,
        full_model='NOT_EVALUATED', accuracy='NOT_EVALUATED'))
    print('REAL_GPTQ_INT8_ACCUMULATION_AND_SCALE_EXACT')


if __name__ == '__main__': main()
