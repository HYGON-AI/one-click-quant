import torch
from src.hy4.moe_gptq_adapter import NarrowGPTQ

torch.manual_seed(13)
layer=torch.nn.Linear(7,3,bias=False)
x=torch.randn(11,7)
native=NarrowGPTQ(layer); native.update(x)
merged=NarrowGPTQ(layer); merged.use_merged_hessian(2*x.T@x/len(x),len(x))
assert native.num_samples==merged.num_samples
assert torch.allclose(native.H,merged.H,atol=1e-6)
merged.use_merged_hessian(None,0)
assert merged.export()['fallback']=='zero_coverage'
print('MERGED_HESSIAN_MATCHES_UPSTREAM_COLLECTOR_PASS')
