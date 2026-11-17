"""Packed E2M1 + E8M0 decoding without full-size CPU/FP32 intermediates.

The integer exponent construction also handles FP32 subnormals explicitly;
exp2 on accelerator hardware may flush those values to zero. Low nibble first,
32 values per scale, matching loading.decode_mxfp4 and the lightop decoder.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode(packed, scales, output, invalid, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    byte = tl.load(packed + i // 2, i < N, other=0).to(tl.uint32)
    code = (byte >> ((i % 2) * 4)) & 15
    scale = tl.load(scales + i // 32, i < N, other=127).to(tl.int32)
    mag = code & 7
    sign = (code & 8) << 28
    base = tl.where(mag >= 2, 0x3f000000 + (mag << 22), tl.where(mag == 1, 0x3f000000, 0))
    exponent = (base >> 23).to(tl.int32) + scale - 127
    mantissa = base & 0x7fffff
    normal = (exponent.to(tl.uint32) << 23) | mantissa
    subnormal = (0x800000 | mantissa) >> tl.minimum(tl.maximum(1 - exponent, 0), 31)
    bits = tl.where(exponent <= 0, subnormal, tl.where(exponent >= 255, 0x7f800000, normal))
    bits = tl.where(mag == 0, 0, bits) | sign
    value = bits.to(tl.uint32).to(tl.float32, bitcast=True).to(output.dtype.element_ty)
    bad = (scale == 255) | (tl.abs(value.to(tl.float32)) == float("inf"))
    if tl.sum(((i < N) & bad).to(tl.int32), 0) > 0:
        tl.atomic_or(invalid, 1)
    tl.store(output + i, value, i < N)


def decode_mxfp4_gpu(weight, scale, dtype):
    """Return (decoded weight, device error flag); caller must check the flag."""
    if weight.device.type != "cuda" or scale.device != weight.device:
        raise ValueError("GPU MXFP4 requires packed weights and scales on the same accelerator")
    if weight.dtype != torch.uint8 or scale.dtype != torch.uint8 or weight.ndim < 2:
        raise ValueError("GPU MXFP4 requires U8 packed matrices and U8 E8M0 scales")
    shape = (*weight.shape[:-1], weight.shape[-1] * 2)
    if shape[-1] % 32 or tuple(scale.shape) != (*shape[:-1], shape[-1] // 32):
        raise ValueError("MXFP4 weight/scale shape mismatch")
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Unsupported MXFP4 analysis dtype")
    if not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("GPU MXFP4 inputs must be contiguous")
    output = torch.empty(shape, dtype=dtype, device=weight.device)
    invalid = torch.zeros((), dtype=torch.int32, device=weight.device)
    _decode[(triton.cdiv(output.numel(), 1024),)](weight, scale, output, invalid, output.numel(), 1024)
    return output, invalid
