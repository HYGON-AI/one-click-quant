# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Scoped ModelOpt policies for manually aggregated MiMo EP sensitivity scores."""

from contextlib import contextmanager


@contextmanager
def local_modelopt_state(model):
    """Keep calibration/statistics rank-local; the EP caller aggregates them.

    Enter before ``auto_quantize`` converts the model, and retain the context
    through calibration and extraction of raw scores. ModelOpt's default DP
    process group would otherwise combine different experts at matching local
    traversal positions. Converted modules retain their explicit local state;
    only the temporary library function replacements are restored on exit.
    """
    from modelopt.torch.quantization.nn import QuantModule
    from modelopt.torch.quantization.plugins import huggingface
    from modelopt.torch.utils.distributed import ParallelState
    from .ep_comm import _differentiable_ep_forward

    from modelopt.torch.quantization.algorithms import AutoQuantizeGradientSearcher

    original_search = AutoQuantizeGradientSearcher.run_search_with_stats
    original_score_rules = AutoQuantizeGradientSearcher.score_module_rules
    original_initialize = QuantModule._initialize_parallel_state
    original_sync = huggingface.sync_moe_expert_amax
    original_sparse_forward = huggingface._QuantSparseSequentialMoe.forward

    def initialize_local(module):
        module.parallel_state = ParallelState(
            data_parallel_group=-1,
            tensor_parallel_group=-1,
            expert_model_parallel_group=-1,
        )

    def sync_local_experts(experts, sync_weight_amax=False):
        return original_sync(
            [expert for expert in experts if expert is not None],
            sync_weight_amax=sync_weight_amax,
        )

    def sparse_forward(module, *args, **kwargs):
        # DynamicModule.convert stashes instance.forward in _forward_pre_dm.
        # The generic sparse wrapper's super().forward would otherwise call
        # MoE-Quant's original non-differentiable EP implementation instead.
        if hasattr(module, "_autoquantize_ep_original_forward"):
            if hasattr(module, "_autoquantize_ep_forward"):
                return module._autoquantize_ep_forward(*args, **kwargs)
            return _differentiable_ep_forward(module, *args, **kwargs)
        return original_sparse_forward(module, *args, **kwargs)

    def deferred_search(searcher, max_weight_size, verbose=False):
        # Keep official scoring/conversion, and perform the sole real ILP after global aggregation.
        selected = {}
        for key, stats in searcher.candidate_stats.items():
            index = min(range(len(stats["costs"])), key=lambda i: stats["costs"][i])
            selected[key] = {"format": stats["formats"][index], "costs": stats["costs"][index],
                             "scores": stats["scores"][index]}
        return selected, True

    # Existing converted modules need the same policy as newly converted ones.
    for module in model.modules():
        if isinstance(module, QuantModule):
            initialize_local(module)

    AutoQuantizeGradientSearcher.run_search_with_stats = deferred_search
    AutoQuantizeGradientSearcher.score_module_rules = [
        *getattr(model, "_autoquantize_score_module_rules", ()), *original_score_rules]
    QuantModule._initialize_parallel_state = initialize_local
    huggingface.sync_moe_expert_amax = sync_local_experts
    huggingface._QuantSparseSequentialMoe.forward = sparse_forward
    try:
        yield
    finally:
        AutoQuantizeGradientSearcher.run_search_with_stats = original_search
        AutoQuantizeGradientSearcher.score_module_rules = original_score_rules
        huggingface._QuantSparseSequentialMoe.forward = original_sparse_forward
        huggingface.sync_moe_expert_amax = original_sync
        QuantModule._initialize_parallel_state = original_initialize
