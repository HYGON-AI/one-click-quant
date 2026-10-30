# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Source offload, block residency and activation-only checkpointed scoring.

Reuses the verified MiMo execution mechanisms without importing examples.
"""

from collections.abc import MutableMapping
from contextlib import contextmanager
from functools import wraps

import torch
from torch import nn


class _LeafWeights(MutableMapping):
    """Accelerate mapping whose original parameters are immutable during max calibration."""

    def __init__(self, prefix, parameters, decoded):
        self.prefix, self.decoded = prefix, decoded
        self.parameters = {
            name: (tuple(value.shape), value.dtype) for name, value in parameters
        }

    def __getitem__(self, key):
        shape, dtype = self.parameters[key]
        name = f"{self.prefix}.{key}" if self.prefix else key
        return self.decoded.get(name, shape, dtype)

    def __contains__(self, key):
        return key in self.parameters

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)

    def __setitem__(self, key, value):
        if key not in self.parameters:
            raise KeyError(key)
        if tuple(value.shape) != self.parameters[key][0]:
            raise ValueError(f"Unexpected source-parameter writeback shape: {key}")
        # shortcut: max-only calibration never changes source weights; supporting
        # AWQ/SmoothQuant would require bounded persistence of their weight updates.

    def __delitem__(self, key):
        raise TypeError("Source checkpoint weights are immutable")


def attach_source_offload(model, decoded, device):
    """Attach recognized Accelerate hooks so ModelOpt can also read weights outside forward."""
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    for prefix, module in list(model.named_modules()):
        parameters = list(module.named_parameters(recurse=False))
        if parameters:
            add_hook_to_module(
                module,
                AlignDevicesHook(
                    execution_device=device,
                    offload=True,
                    weights_map=_LeafWeights(prefix, parameters, decoded),
                    offload_buffers=False,
                ),
            )
        # Keep small source buffers (notably FP32 RoPE) resident. ModelOpt's
        # subsequently created amax buffers must not enter the source mapping.
        for name, value in module.named_buffers(recurse=False):
            if value.is_meta:
                raise ValueError(f"Source buffer was unexpectedly built on meta: {prefix}.{name}")
            module._buffers[name] = value.to(device)
    return decoded


@contextmanager
def checkpointed_gradient_context(model):
    """Recompute one decoder block at a time while keeping inference-only routing."""
    from torch.utils.checkpoint import checkpoint

    def seed_activation_gradient(module, inputs, output):
        return output.detach().requires_grad_(True) if torch.is_grad_enabled() else output

    def checkpoint_forward(original):
        @wraps(original)
        def forward(*args, **kwargs):
            if not torch.is_grad_enabled():
                return original(*args, **kwargs)
            # Reentrant checkpoint runs the global forward without autograd,
            # so offloaded weights are retained only during this block's backward.
            # Kimi's cross-layer block_residual is a keyword tensor. It must be
            # a checkpoint input so backward reaches earlier residual blocks.
            keys = tuple(kwargs)
            def recompute(*values):
                return original(*values[:len(args)], **dict(zip(keys, values[len(args):])))
            return checkpoint(recompute, *args, *kwargs.values(), use_reentrant=True)
        return forward

    originals = []
    handle = model.get_input_embeddings().register_forward_hook(seed_activation_gradient)
    try:
        for block in model.model.layers:
            originals.append((block, block.forward))
            block.forward = checkpoint_forward(block.forward)
        yield
    finally:
        handle.remove()
        for block, original in originals:
            block.forward = original


def _is_streaming_model(model):
    return getattr(model, "_autoquantize_streaming", False)


def _activation_gradients_only(name, model):
    return False


def register_gradient_support(model):
    from modelopt.torch.quantization.algorithms import AutoQuantizeGradientSearcher

    model._autoquantize_streaming = True
    if not any(checker is _is_streaming_model for checker, _, _ in AutoQuantizeGradientSearcher._custom_support):
        AutoQuantizeGradientSearcher.register_custom_support(
            _is_streaming_model, checkpointed_gradient_context, _activation_gradients_only
        )


def _source_hook(module, decoded):
    """Find the existing source hook, including an Accelerate SequentialHook."""
    pending = [getattr(module, "_hf_hook", None)]
    while pending:
        hook = pending.pop()
        if hook is None:
            continue
        if getattr(getattr(hook, "weights_map", None), "decoded", None) is decoded:
            return hook
        pending.extend(getattr(hook, "hooks", ()))
    return None


class BlockResidency:
    """Reuse decoded parameters across all calls inside one decoder block."""

    def __init__(self, model, decoded, device, *, blocks=None):
        self.device = torch.device(device)
        self.decoded = decoded
        self._blocks = []
        self._entries = {}
        self._active = None
        self._depth = 0
        self._hook_states = []
        self._closed = False
        self.block_loads = {}
        for index, block in enumerate(model.model.layers if blocks is None else blocks):
            entries = []
            for module in block.modules():
                hook = _source_hook(module, decoded)
                if hook is not None:
                    entries.append((module, tuple(hook.weights_map.parameters)))
            self._entries[block] = entries
            original = block.forward

            def wrap(current_block, forward):
                @wraps(forward)
                def resident_forward(*args, **kwargs):
                    with self.materialize(current_block):
                        return forward(*args, **kwargs)
                return resident_forward

            wrapper = wrap(block, original)
            block.forward = wrapper
            self._blocks.append((block, original, wrapper))
            self.block_loads[index] = 0

    def _release(self, block):
        for module, names in self._entries[block]:
            for name in names:
                parameter = module._parameters[name]
                if parameter is not None and not parameter.is_meta:
                    module._parameters[name] = nn.Parameter(
                        torch.empty_like(parameter, device="meta"),
                        requires_grad=parameter.requires_grad,
                    )
        for hook, old_offload in reversed(self._hook_states):
            hook.offload = old_offload
        self._hook_states = []
        self._active = None
        self._depth = 0

    @contextmanager
    def materialize(self, block):
        if self._closed:
            raise RuntimeError("Block residency manager is closed")
        if block not in self._entries:
            raise ValueError("Block is not managed by this residency manager")
        if self._active is not None and self._active is not block:
            raise RuntimeError("Decoder blocks must execute sequentially")
        if self._active is block:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return

        self._active, self._depth = block, 1
        try:
            for module, names in self._entries[block]:
                hook = _source_hook(module, self.decoded)
                if hook is None:
                    raise RuntimeError("Source offload hook was removed from a managed module")
                self._hook_states.append((hook, hook.offload))
                # Use raw parameters rather than getattr(module, "weight").
                # ModelOpt's dynamic weight getter may otherwise fake-quantize
                # an unmaterialized meta tensor before its scoring hook runs.
                for name in names:
                    parameter = module._parameters[name]
                    value = hook.weights_map[name]
                    module._parameters[name] = nn.Parameter(
                        value.to(device=self.device, dtype=parameter.dtype),
                        requires_grad=parameter.requires_grad,
                    )
                hook.offload = False
            index = next(i for i, (item, _, _) in enumerate(self._blocks) if item is block)
            self.block_loads[index] += 1
            yield
        finally:
            self._release(block)

    def close(self):
        if self._closed:
            return
        if self._active is not None:
            self._release(self._active)
        for block, original, wrapper in self._blocks:
            if block.forward is wrapper:
                block.forward = original
        self._closed = True


def attach_block_residency(model, decoded, device):
    """Install block residency after attach_source_offload, before ModelOpt conversion."""
    return BlockResidency(model, decoded, device)
