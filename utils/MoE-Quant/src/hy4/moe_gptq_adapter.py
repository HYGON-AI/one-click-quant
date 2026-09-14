"""Narrow-channel policy around the vendored MoE-Quant GPTQ loop.

Does not change the user tool tree or claim model/runtime integration.
"""
import torch
from src.gptq import GPTQ
from src.gptq_loop import gptq_loop
from .format_contract import narrow_grid, pack


def merge_hessians(parts):
    """Input H is MoE-Quant's 2*mean(X.T X); weight by routed tokens."""
    if not parts:
        raise ValueError('No rank statistics')
    shape = parts[0][0].shape
    total = sum(n for _, n in parts)
    result = torch.zeros_like(parts[0][0], dtype=torch.float32)
    for h, n in parts:
        if type(n) is not int or n < 0 or h.shape != shape or not torch.isfinite(h).all():
            raise ValueError('Invalid rank statistics')
        if n:
            result.add_(h.float(), alpha=n / total)
    return result, total


class NarrowGPTQ(GPTQ):
    """Use upstream collection and error-compensation loop, explicit policy."""
    def __init__(self, layer):
        super().__init__(layer, group_size=None, sym=True, rel_damp=.01,
                         block_size=128, is_distributed=False)

    def use_merged_hessian(self, hessian, count):
        """Consume verified merged H=2*sum(X.T X)/count, not rank averages."""
        width=self.layer.weight.shape[1]
        if type(count) is not int or count<0:
            raise ValueError('Invalid real token count')
        if count==0:
            if hessian is not None:
                raise ValueError('Zero coverage must not carry fabricated Hessian')
            self.H=None
        else:
            if hessian is None or hessian.shape!=(width,width) or hessian.dtype!=torch.float32 or not torch.isfinite(hessian).all():
                raise ValueError('Invalid merged Hessian')
            if not torch.allclose(hessian,hessian.T,atol=1e-5,rtol=1e-5):
                raise ValueError('Asymmetric merged Hessian')
            self.H=hessian.to(device=self.layer.weight.device).clone()
        self.num_samples=count

    @torch.no_grad()
    def export(self):
        weight = self.layer.weight.detach().float()
        if not torch.isfinite(weight).all():
            raise ValueError('Nonfinite weights')
        scale, zero, _ = narrow_grid(weight)
        maxq = torch.tensor(14., device=weight.device)
        fallback, selected_damp, failures = None, None, []
        missing = 0
        if self.num_samples == 0:
            fallback = 'zero_coverage'
        elif self.H is None or not torch.isfinite(self.H).all():
            raise ValueError('Corrupt collected Hessian; refusing silent fallback')
        else:
            h = self.H.detach().float().clone()
            unobserved = h.diag() == 0
            missing = int(unobserved.sum())
            # Preserve the original W columns; independent positive diagonal
            # is used for unobserved channels, never W[:,dead]=0.
            h[unobserved, :] = 0
            h[:, unobserved] = 0
            h[unobserved, unobserved] = 1
            for damp in (.01, .03, .1):
                candidate = h.clone()
                candidate.diagonal().add_(damp * h.diag().mean())
                try:
                    chol = torch.linalg.cholesky(candidate)
                    inverse = torch.cholesky_inverse(chol)
                    upper = torch.linalg.cholesky(inverse, upper=True)
                    upper = upper / upper.diag()[:, None]
                except torch.linalg.LinAlgError as error:
                    failures.append(dict(damp=damp, error=str(error)))
                    continue
                # This is the upstream MoE-Quant algorithm, not custom GPTQ.
                unsigned = gptq_loop(
                    weight=weight.T.contiguous(), hessian_inv=upper,
                    scale=scale.expand_as(weight).T.contiguous(),
                    qzero=zero.expand_as(weight).T.contiguous(), maxq=maxq,
                    dtype=torch.float32, gptq_block_size=128).T.contiguous()
                selected_damp = damp
                break
            else:
                fallback = 'cholesky_failed_all_dampings'
        if fallback:
            unsigned = (weight / scale + zero).round().clamp(0, 14)
        q = (unsigned.to(torch.int16) - 7).to(torch.int8)
        return dict(packed=pack(q), scale=scale, shape=list(weight.shape),
                    algorithm='RTN' if fallback else 'MoE-Quant-GPTQ',
                    fallback=fallback, failures=failures, damp=selected_damp,
                    seen=self.num_samples, unobserved_columns=missing)


if __name__ == '__main__':
    from .format_contract import unpack
    torch.manual_seed(42)
    torch.set_num_threads(2)
    a, b = torch.randn(3, 7), torch.randn(19, 7)
    merged, count = merge_hessians([(2*a.T@a/len(a),len(a)),(2*b.T@b/len(b),len(b))])
    x=torch.cat((a,b))
    assert count==22 and torch.allclose(merged,2*x.T@x/22,atol=1e-6)
    layer=torch.nn.Linear(7,3,bias=False)
    layer.weight.data[0].zero_()
    before=layer.weight.detach().clone()
    result=NarrowGPTQ(layer).export()
    assert result['fallback']=='zero_coverage' and result['scale'].dtype==torch.float32
    assert result['scale'][0].item()==1
    assert torch.equal(before,layer.weight)
    assert (unpack(result['packed'],7)[0]==0).all()
    print('CPU_HESSIAN_AND_FALLBACK_PASSED; GPU_GPTQ_NOT_TESTED')
