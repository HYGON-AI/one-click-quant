"""Synthetic full/shared-indexer layer chain compared with official forward."""
import torch
from transformers import AutoModelForCausalLM
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config
from src.hy4.hf_hy4_layer_state import prepare, forward_block

torch.manual_seed(20260911)
torch.set_num_threads(1)
config = HYV4Config(vocab_size=32, hidden_size=16, intermediate_size=24,
    moe_intermediate_size=8, num_hidden_layers=3, num_attention_heads=2,
    num_key_value_heads=2, n_routed_experts=3, n_shared_experts=1,
    num_experts_per_tok=2, q_lora_rank=8, kv_lora_rank=4,
    qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
    index_topk=2, index_head_dim=4, index_n_heads=2, hc_mult=2,
    indexer_types=['full', 'full', 'shared'],
    pad_token_id=0, bos_token_id=1, eos_token_id=2,
    rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0})
model = AutoModelForCausalLM.from_config(config, attn_implementation='eager').eval().model
with torch.no_grad():
    docs = [torch.tensor([[1, 5, 6, 7, 8]]), torch.tensor([[1, 9, 10]])]
    expected = [model(input_ids=t, use_cache=False).last_hidden_state for t in docs]
    work = [prepare(model, t) for t in docs]
    # Layer-major ordering interleaves independent document states.
    for block in model.layers:
        work = [forward_block(block, hidden, state) for hidden, state in work]
    for target, (hidden, state) in zip(expected, work):
        actual = model.norm(model.hc_head(hidden))
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
        assert state.next_layer == 3
    assert work[0][1].positions.shape != work[1][1].positions.shape
    try:
        forward_block(model.layers[0], *work[0])
    except ValueError:
        pass
    else:
        raise AssertionError('Out-of-order state accepted')
print('PASS: layer-major full/shared chain equals official FP32 CPU forward; synthetic, not source-model validation')
