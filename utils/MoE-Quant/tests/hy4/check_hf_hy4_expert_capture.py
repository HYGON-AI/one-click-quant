"""Pinned official expert comparison; synthetic inputs, not model accuracy."""
import torch
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config
from transformers.models.hy_v4.modeling_hy_v4 import HYV4Experts
from src.hy4.hf_hy4_expert_capture import capture_experts

torch.manual_seed(20260911)
torch.set_num_threads(1)
config = HYV4Config(hidden_size=7, moe_intermediate_size=5, n_routed_experts=3,
                   swiglu_limit=7.0)
experts = HYV4Experts(config)
with torch.no_grad():
    experts.gate_up_proj.normal_()
    experts.down_proj.normal_()
for tokens in (0, 1, 9):
    x = torch.randn(tokens, 7)
    # Includes repeated routes, empty experts, and sentinel expert index.
    ids = torch.tensor([[0, 0, 3]]).expand(tokens, -1)
    weights = torch.rand(tokens, 3)
    captured = {}
    def observer(expert, projection, value):
        captured[(expert, projection)] = value.clone()
    with torch.no_grad():
        expected = experts(x, ids, weights)
        with capture_experts(experts, observer):
            actual = experts(x, ids, weights)
        assert torch.equal(actual, expected)
        assert torch.equal(experts(x, ids, weights), expected)
    if tokens:
        assert set(captured) == {(0, 'gate'), (0, 'up'), (0, 'down')}
        routed = x[torch.arange(tokens).repeat(2)]
        assert torch.equal(captured[(0, 'gate')], routed)
        assert torch.equal(captured[(0, 'up')], routed)
        assert torch.equal(captured[(0, 'down')], experts._apply_gate(
            torch.nn.functional.linear(routed, experts.gate_up_proj[0])))
    else:
        assert not captured
try:
    with capture_experts(experts, lambda *args: None):
        raise RuntimeError('test observer lifetime')
except RuntimeError:
    pass
assert '_hy4_projection_observer' not in experts.__dict__
assert 'forward' not in experts.__dict__
print('PASS: native expert output exact, routed/down inputs, empty/repeated routes, restoration; synthetic CPU only')
