# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Source offload, block residency and activation-only checkpointed scoring.

Reuses the verified MiMo execution mechanisms without importing examples.
"""

from collections.abc import MutableMapping
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
import os
import math
import time

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
        return self.materialize(key)

    def materialize(self, key, *, decoded=None, defer_validation=False):
        shape, dtype = self.parameters[key]
        name = f"{self.prefix}.{key}" if self.prefix else key
        loader = self.decoded if decoded is None else decoded
        return loader.get_on_device(name, shape, dtype, loader.runtime.device,
                                    defer_validation=defer_validation)

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
    residency = getattr(model, "_autoquantize_residency", None)
    if residency is not None:
        residency.scoring = True
    try:
        for block in model.model.layers:
            originals.append((block, block.forward))
            block.forward = checkpoint_forward(block.forward)
        yield
    finally:
        handle.remove()
        for block, original in originals:
            block.forward = original
        if residency is not None:
            residency.scoring = False
            residency.drain_prefetch()


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

    def __init__(self, model, decoded, device, *, blocks=None, prefetch=False,
                 prefetch_budget_bytes=16 * 1024**3, reserve_bytes=16 * 1024**3):
        self.device = torch.device(device)
        self.decoded = decoded
        self._blocks = []
        self._entries = {}
        self._active = None
        self._depth = 0
        self._hook_states = []
        self._closed = False
        self.block_loads = {}
        self.stats = Counter()
        self.scoring = False
        self.prefetch = prefetch and self.device.type == "cuda"
        self.prefetch_budget_bytes = prefetch_budget_bytes
        self.reserve_bytes = reserve_bytes
        self._pending = None
        self._executor = None
        self._worker_decoded = None
        self._copy_stream = None
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
        self._indices = {block: i for i, (block, _, _) in enumerate(self._blocks)}
        self._block_bytes = {
            block: sum(torch.empty((), dtype=hook.weights_map.parameters[name][1]).element_size()
                       * math.prod(hook.weights_map.parameters[name][0])
                       for module, names in entries
                       for hook in [_source_hook(module, decoded)] for name in names)
            for block, entries in self._entries.items()
        }

    def _prepare(self, block, decoded):
        values = []
        for module, names in self._entries[block]:
            hook = _source_hook(module, self.decoded)
            if hook is None:
                raise RuntimeError("Source offload hook was removed from a managed module")
            for name in names:
                value = hook.weights_map.materialize(name, decoded=decoded, defer_validation=True)
                values.append((module, name, value))
        decoded.check_pending()
        return values

    def _prepare_async(self, block):
        # The worker owns separate mmap handles, decoded cache and pinned buffers.
        with torch.cuda.device(self.device), torch.cuda.stream(self._copy_stream):
            values = self._prepare(block, self._worker_decoded)
            event = torch.cuda.Event()
            event.record()
            return values, event

    def _schedule_next(self, block):
        if not self.prefetch or not self.scoring or self._pending is not None:
            return
        index = self._indices[block] + (-1 if torch.is_grad_enabled() else 1)
        if not 0 <= index < len(self._blocks):
            return
        following = self._blocks[index][0]
        size = self._block_bytes[following]
        free, _ = torch.cuda.mem_get_info(self.device)
        if size > self.prefetch_budget_bytes or free < size + self.reserve_bytes:
            self.stats["prefetch_budget_skips"] += 1
            return
        if self._executor is None:
            self._worker_decoded = self.decoded.fork()
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autoquant-prefetch")
        self._pending = (following, self._executor.submit(self._prepare_async, following))
        self.stats["prefetch_submitted"] += 1

    def drain_prefetch(self):
        if self._pending is not None:
            _, future = self._pending
            self._pending = None
            values, event = future.result()
            event.synchronize()
            del values

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
        started = time.monotonic()
        try:
            values = None
            if self._pending is not None:
                pending_block, future = self._pending
                if pending_block is block:
                    self._pending = None
                    values, event = future.result()
                    stream = torch.cuda.current_stream(self.device)
                    stream.wait_event(event)
                    for _, _, value in values:
                        value.record_stream(stream)
                    self.stats["prefetch_hits"] += 1
                else:
                    self.drain_prefetch()
            if values is None:
                values = self._prepare(block, self.decoded)
            for module, name, value in values:
                parameter = module._parameters[name]
                module._parameters[name] = nn.Parameter(value, requires_grad=parameter.requires_grad)
            del values
            for module, names in self._entries[block]:
                hook = _source_hook(module, self.decoded)
                if hook is None:
                    raise RuntimeError("Source offload hook was removed from a managed module")
                self._hook_states.append((hook, hook.offload))
                hook.offload = False
            index = self._indices[block]
            self.block_loads[index] += 1
            self.stats["materialize_wait_seconds"] += time.monotonic() - started
            if os.environ.get("AUTOQUANTIZE_LOG_BLOCKS") == "1":
                print(f"[rank {os.environ.get('RANK', '0')}] block={index} "
                      f"load={self.block_loads[index]} materialized in {time.monotonic() - started:.1f}s",
                      flush=True)
            self._schedule_next(block)
            yield
        finally:
            self._release(block)

    def close(self):
        if self._closed:
            return
        try:
            self.drain_prefetch()
        finally:
            try:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                    self._worker_decoded.close()
                    self.decoded.stats.update(self._worker_decoded.stats)
                    self.decoded.decoder.counts.update(self._worker_decoded.decoder.counts)
                    self.decoded.decoder.consumed_scales.update(self._worker_decoded.decoder.consumed_scales)
            finally:
                if self._active is not None:
                    self._release(self._active)
                for block, original, wrapper in self._blocks:
                    if block.forward is wrapper:
                        block.forward = original
                self._closed = True


def attach_block_residency(model, decoded, device, **options):
    """Install block residency after attach_source_offload, before ModelOpt conversion."""
    return BlockResidency(model, decoded, device, **options)
