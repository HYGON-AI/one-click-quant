"""Bounded pinned host buffers for checkpoint transfers on the current stream."""

import torch


class PinnedStaging:
    """Two host buffers; an event protects each buffer from early reuse."""

    def __init__(self, device, chunk_bytes=32 * 1024**2):
        self.device = torch.device(device)
        self.chunk_bytes = chunk_bytes
        self.slots = [None, None]
        self.events = [None, None]
        self.cursor = 0
        self.bytes_copied = 0

    def copy(self, value):
        if value.device == self.device:
            return value
        if value.device.type != "cpu":
            raise ValueError("Staging expects CPU checkpoint tensors")
        value = value.contiguous()
        result = torch.empty_like(value, device=self.device)
        source, target = value.view(-1), result.view(-1)
        step = self.chunk_bytes // value.element_size()
        for offset in range(0, value.numel(), step):
            count = min(step, value.numel() - offset)
            slot = self.cursor % 2
            self.cursor += 1
            if self.events[slot] is not None:
                self.events[slot].synchronize()
            size = count * value.element_size()
            if self.slots[slot] is None or self.slots[slot].numel() < size:
                self.slots[slot] = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            staging = self.slots[slot][:size].view(value.dtype)
            staging.copy_(source[offset:offset + count])
            target[offset:offset + count].copy_(staging, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self.events[slot] = event
            self.bytes_copied += size
        return result

    def close(self):
        for event in self.events:
            if event is not None:
                event.synchronize()
        self.events = [None, None]
        self.slots = [None, None]
