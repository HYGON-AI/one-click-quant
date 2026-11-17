"""Read-only max calibration and per-channel statistics reuse across W4/W8.

Only installed for this tool's two max recipes. ModelOpt still owns candidate
configuration, finish_stats_collection, synchronization and state serialization.
"""

from collections import Counter
from contextlib import contextmanager
from unittest.mock import patch

import torch


@contextmanager
def reuse_max_calibration(model, *, enabled=True):
    from modelopt.torch.quantization import model_calib
    from modelopt.torch.quantization.calib import MaxCalibrator
    from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
    from modelopt.torch.quantization.utils import quantizer_attr_names
    from .streaming import _LeafWeights

    stats = Counter()
    if not enabled:
        yield stats
        return
    original = model_calib.weight_only_quantize
    cache = {}

    def readonly(module):
        hooks = [getattr(child, "_hf_hook", None) for child in module.modules()]
        found = False
        while hooks:
            hook = hooks.pop()
            if hook is None:
                continue
            hooks.extend(getattr(hook, "hooks", ()))
            mapping = getattr(hook, "weights_map", None)
            if mapping is not None:
                if not isinstance(mapping, _LeafWeights):
                    return False
                found = True
        return found

    def reusable(q):
        return (isinstance(q, TensorQuantizer) and not q._disabled
                and q.num_bits in (4, 8) and q.axis in (0, (0,))
                and type(q._calibrator) is MaxCalibrator
                and q._if_calib and not q._if_quant
                and not q._dynamic and not q.block_sizes and not q.unsigned
                and q.pre_quant_scale is None and not q.rotate_is_enabled
                and q.bias is None and q.backend is None
                and not q._use_constant_amax and q._constant_amax is None
                and not q._calibrator._track_amax)

    @torch.no_grad()
    def calibrate(root):
        if root is not model:
            return original(root)
        stats["weight_scans"] += 1
        named = dict(root.named_modules())
        for prefix, module in named.items():
            if not isinstance(module, QuantModule):
                continue
            # Inspect raw parameters: accessing the dynamic .weight getter can
            # itself quantize/collect stats, even on a meta parameter.
            standard = type(module).iter_weights_for_calibration is QuantModule.iter_weights_for_calibration
            pairs = []
            if standard:
                for name, parameter in module._parameters.items():
                    q = getattr(module, quantizer_attr_names(name).weight_quantizer, None)
                    if parameter is not None and q is not None:
                        pairs.append((name, parameter, q))
                if not pairs:
                    stats["empty_modules_skipped"] += 1
                    continue
                pending = []
                for name, parameter, q in pairs:
                    if isinstance(q, TensorQuantizer) and q._disabled:
                        stats["disabled_weights_skipped"] += 1
                        continue
                    key = (prefix, name, tuple(parameter.shape), parameter.dtype)
                    saved = cache.pop(key, None) if reusable(q) else None
                    if saved is not None:
                        value, device = saved
                        q._calibrator._calib_amax = value.to(device=device, copy=True)
                        stats["amax_reused"] += 1
                    else:
                        pending.append((name, q, key))
                if not pending:
                    continue
            writeback = not readonly(module)
            if not writeback:
                stats["writebacks_skipped"] += 1
            with model_calib.enable_weight_access_and_writeback(module, root, named, writeback=writeback):
                if standard:
                    for name, q, key in pending:
                        q(getattr(module, name))
                        stats["weights_calibrated"] += 1
                        if reusable(q):
                            amax = q._calibrator.compute_amax()
                            if amax is not None:
                                cache[key] = (amax.detach().to("cpu", copy=True), amax.device)
                else:
                    # Transposed/fused/plugin-specific views retain their own
                    # iterator. Never assume their channel axis or layout.
                    for weight, q in module.iter_weights_for_calibration():
                        q(weight)
                        stats["fallback_weights"] += 1

    try:
        with patch.object(model_calib, "weight_only_quantize", calibrate):
            yield stats
    finally:
        cache.clear()
