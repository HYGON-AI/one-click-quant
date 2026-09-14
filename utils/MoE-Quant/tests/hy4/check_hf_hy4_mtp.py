import torch
from transformers import AutoModelForCausalLM
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config
from src.hy4.hf_hy4_mtp import HFHy4MTP, prepare_teacher_forcing
torch.manual_seed(31)
torch.set_num_threads(1)
config = HYV4Config(vocab_size=32, hidden_size=16, intermediate_size=24,
    moe_intermediate_size=8, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=2, n_routed_experts=3, n_shared_experts=1,
    num_experts_per_tok=2, q_lora_rank=8, kv_lora_rank=4,
    qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
    index_topk=2, index_head_dim=4, index_n_heads=2, hc_mult=2,
    indexer_types=['full','full'], pad_token_id=0, bos_token_id=1, eos_token_id=2,
    rope_parameters={'rope_type':'default','rope_theta':10000.0},
    num_nextn_predict_layers=1)
main = AutoModelForCausalLM.from_config(config, attn_implementation='eager').eval().model
mtp = HFHy4MTP(config).eval()
with torch.no_grad():
    for p in mtp.parameters():
        p.normal_(std=.02)
    for ids in (torch.tensor([[1,4,5,6]]), torch.tensor([[1,9]])):
        previous = torch.randn(1, ids.shape[1], 16)
        args = prepare_teacher_forcing(main,ids,previous)
        assert torch.equal(args[1],previous[:,:-1])
        assert args[2].tolist()==[list(range(ids.shape[1]-1))]
        assert torch.count_nonzero(args[0][:,0])==0
        if ids.shape[1]>2:
            assert torch.equal(args[0][:,1:],main.embed_tokens(ids[:,2:]))
        out, topk = mtp(*args)
        assert out.shape==(1,ids.shape[1]-1,16) and torch.isfinite(out).all()
        repeated, _ = mtp(*args)
        assert torch.equal(out,repeated)
print('PASS: document-local shifted inputs, native position-zero masking, synthetic deterministic MTP forward; native GPU parity NOT_EVALUATED')
