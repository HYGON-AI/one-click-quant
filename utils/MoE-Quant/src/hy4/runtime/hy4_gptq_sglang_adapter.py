# Adapted from the preserved rtn_eval_tp8/adapter.py; new GPTQ path only.
"""Hy4 TP8 resident MoE-Quant GPTQ W4A8 with graph-safe GPU expert grouping."""
import logging
import json
import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .sglang_gptq_checkpoint import Checkpoint
from .lazy_checkpoint import LazyCheckpoint


def install(root, source_index_path=None):
    from sglang.srt.models import hunyuan_v4 as main
    from sglang.srt.models import hunyuan_v4_nextn as mtp
    from sglang.srt.models.deepseek_v2 import MoEGate
    from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_rank, tensor_model_parallel_all_reduce
    from .hy4_gptq_resident import Bank, bounds
    from .hy4_gptq_kernel import projection
    from .hy4_gptq_grouped import routed, integer_evidence

    checkpoint = LazyCheckpoint(root, source_index_path, Checkpoint)
    if getattr(main, '_gptq_mtp_installed', False):
        raise RuntimeError('Adapter already installed')

    class ResidentMoE(nn.Module):
        def __init__(self, config, layer_id, quant_config=None, prefix='', alt_stream=None, is_nextn=False):
            super().__init__()
            parallel = get_parallel()
            if parallel.tp_size != 8 or parallel.moe_ep_size != 1 or get_pp_group().world_size != 1:
                raise ValueError('Resident evaluation requires TP=8, native EP=PP=1')
            self.rank=get_tensor_model_parallel_rank()
            self.lo,self.hi=bounds(config.n_routed_experts,self.rank,8)
            if quant_config is not None or config.hidden_act != 'silu' or getattr(config, 'num_hash_layers', 0):
                raise ValueError('Unsupported mixed quantization/activation/hash routing')
            self.num_fused_shared_experts = 0
            self.prefix = 'model.mtp_layers.0.mlp' if is_nextn else prefix.lstrip('.')
            if self.prefix + '.experts.down_proj' not in checkpoint.targets:
                raise ValueError(f'Unknown expert block: {self.prefix}')
            self.gate = MoEGate(config, None, prefix=prefix + '.gate')
            self.topk = TopK(
                top_k=config.num_experts_per_tok, layer_id=layer_id,
                renormalize=config.norm_topk_prob, use_grouped_topk=True,
                num_expert_group=config.n_group, num_fused_shared_experts=0,
                topk_group=config.topk_group, scoring_func=config.scoring_func,
                correction_bias=self.gate.e_score_correction_bias, quant_config=None,
                routed_scaling_factor=config.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=True,
                output_format=TopKOutputFormat.STANDARD)
            self.limit = config.swiglu_limit
            self.calls = 0
            device=torch.device('cuda',torch.cuda.current_device())
            self.routed_gate_up=Bank(checkpoint,self.prefix+'.experts.gate_up_proj',device,slice(self.lo,self.hi))
            self.routed_down=Bank(checkpoint,self.prefix+'.experts.down_proj',device,slice(self.lo,self.hi))
            if self.rank==0:
                self.shared_gate=Bank(checkpoint,self.prefix+'.shared_experts.gate_proj.weight',device)
                self.shared_up=Bank(checkpoint,self.prefix+'.shared_experts.up_proj.weight',device)
                self.shared_down=Bank(checkpoint,self.prefix+'.shared_experts.down_proj.weight',device)
            logging.getLogger(__name__).info('MoE-Quant GPTQ resident rank=%d block=%s experts=[%d,%d)',self.rank,self.prefix,self.lo,self.hi)

        def project(self, x, name, expert=None):
            if expert is not None:
                bank=self.routed_gate_up if name=='experts.gate_up_proj' else self.routed_down
                packed,scale=bank.weights(expert-self.lo)
            else:
                bank={'shared_experts.gate_proj.weight':self.shared_gate,
                      'shared_experts.up_proj.weight':self.shared_up,
                      'shared_experts.down_proj.weight':self.shared_down}[name]
                packed,scale=bank.weights()
            return projection(x, packed, scale)

        def forward(self, hidden_states, forward_batch=None, **kwargs):
            if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
                raise ValueError('Expected BF16 [tokens, hidden] input')
            self.calls += 1
            if not hidden_states.shape[0]:
                return hidden_states
            selected = self.topk(hidden_states, self.gate(hidden_states))
            ids, weights = selected.topk_ids, selected.topk_weights
            result = routed(hidden_states, ids, weights, self.routed_gate_up,
                            self.routed_down, self.lo, self.limit)
            if self.rank==0:
                gate = self.project(hidden_states, 'shared_experts.gate_proj.weight')
                up = self.project(hidden_states, 'shared_experts.up_proj.weight')
                shared = self.project(F.silu(gate) * up, 'shared_experts.down_proj.weight')
                result.add_(shared.float())
            result=tensor_model_parallel_all_reduce(result)
            return result.to(hidden_states.dtype)

    # Patch only the two Hy4 module globals, preserving their isinstance branches.
    main.DeepseekV2MoE = ResidentMoE
    mtp.DeepseekV2MoE = ResidentMoE
    for cls, is_mtp in ((main.HYV4ForCausalLM, False), (mtp.HYV4ForCausalLMNextN, True)):
        original = cls.load_weights
        def load_weights(self, unused_weights, _original=original, _mtp=is_mtp):
            # Do not consume the default whole-bank iterator. Retained tensors use
            # native name remapping, MLA processing, iHC and NextN loading logic.
            class RejectMissingWeight(logging.Handler):
                def emit(self, record):
                    if 'not found in params_dict' in record.getMessage():
                        raise RuntimeError('Unmapped retained weight: ' + record.getMessage())
            logger = logging.getLogger('sglang.srt.models.deepseek_common.deepseek_weight_loader')
            guard = RejectMissingWeight()
            logger.addHandler(guard)
            try:
                return _original(self, checkpoint.retained(mtp=_mtp))
            finally:
                logger.removeHandler(guard)
        cls.load_weights = load_weights
        original_forward = cls.forward
        def forward(self, *args, _original=original_forward, _mtp=is_mtp, **kwargs):
            result = _original(self, *args, **kwargs)
            report_dir = os.environ.get('HY4_GPTQ_REPORT_DIR')
            if report_dir and not torch.cuda.is_current_stream_capturing():
                blocks = [m for m in self.modules() if isinstance(m, ResidentMoE)]
                payload = dict(kind='mtp' if _mtp else 'main', pid=os.getpid(),
                               checkpoint_index_sha256=checkpoint.index_sha256,
                               integer_backend=integer_evidence(),
                               weight_bits=4,activation_bits=8,resident=True,tp=8,rank=get_tensor_model_parallel_rank(),
                               counter_scope='eager_and_capture_only; replay reported by SGLang logs',
                               blocks=[dict(prefix=m.prefix, calls=m.calls) for m in blocks])
                path = Path(report_dir) / ('runtime-%s-%d.json' % (payload['kind'],os.getpid()))
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(payload,indent=2)); temp.replace(path)
            return result
        cls.forward = forward
    main._gptq_mtp_installed = True
    return checkpoint
