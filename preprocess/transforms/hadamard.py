"""Hadamard / orthogonal rotation helpers for QuaRot."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch


class QuaRotMode(Enum):
    HADAMARD = "hadamard"
    BLOCK_HADAMARD_SHIFTED = "block_hadamard_shifted"
    ORTHOGONAL = "orthogonal"  # QR fallback when size is awkward


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _walsh_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if size == 1:
        return torch.tensor([[1.0]], dtype=dtype, device=device)
    half = _walsh_matrix(size // 2, dtype=dtype, device=device)
    top = torch.cat([half, half], dim=1)
    bot = torch.cat([half, -half], dim=1)
    return torch.cat([top, bot], dim=0)


def _random_orthogonal(size: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    a = torch.randn(size, size, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    return q.to(device=device, dtype=dtype)


def random_hadamard_matrix(size: int, dtype: torch.dtype, device: torch.device, seed: int = 1234) -> torch.Tensor:
    """Normalized Hadamard-like rotation.

    Power-of-two sizes use Walsh-Hadamard with random sign flips.
    Other sizes fall back to a seeded QR orthogonal matrix (still valid QuaRot).
    """
    torch.manual_seed(seed)
    if _is_pow2(size):
        had = _walsh_matrix(size, dtype=torch.float64, device=device)
        signs = torch.randint(0, 2, (size,), device=device) * 2 - 1
        rot = had * signs.to(had).unsqueeze(0)
        rot = rot / (size ** 0.5)
        return rot.to(dtype=dtype)
    return _random_orthogonal(size, dtype, device, seed)


def random_hadamard_matrix_block(
    in_channel: int,
    stride: int,
    eye_step: tuple = (-1,),
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    seed: int = 1234,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    h_stride = random_hadamard_matrix(stride, dtype, device, seed=seed)
    h = torch.zeros((in_channel, in_channel), device=device, dtype=dtype)
    num_blocks = in_channel // stride
    for i in range(num_blocks):
        s, e = i * stride, (i + 1) * stride
        if i in eye_step:
            h[s:e, s:e] = torch.eye(stride, device=device, dtype=dtype)
        else:
            h[s:e, s:e] = h_stride
    return h


def create_rot(
    mode: QuaRotMode,
    size: int,
    block_size: int = -1,
    rot_step: int = 1,
    eye_step: tuple = (-1,),
    seed: int = 1234,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    # Build on CPU for deterministic QR / smaller peak VRAM; caller may .to(cuda).
    build = torch.device("cpu")
    shift = 16

    if mode == QuaRotMode.HADAMARD:
        dim = size if block_size == -1 else block_size
        rot = random_hadamard_matrix(dim, dtype, build, seed=seed)
        if block_size != -1:
            rot = torch.block_diag(*[rot] * (size // block_size))
        return rot.to(device) if device is not None else rot

    if mode == QuaRotMode.BLOCK_HADAMARD_SHIFTED:
        bs = 32 if block_size == -1 else block_size
        rot = random_hadamard_matrix_block(size, bs, eye_step, dtype, build, seed=seed)
        identity = torch.eye(size, dtype=torch.float32)
        p = torch.cat((identity[:, -shift:], identity[:, :-shift]), dim=1).to(rot.device)
        if rot_step == 1:
            rot = rot @ p @ rot
        elif rot_step == 2:
            rot = rot @ p @ rot @ p.T @ rot.T
        elif rot_step == 3:
            rot = rot @ p @ rot @ p.T @ rot.T @ p @ rot @ p.T @ rot.T
        else:
            raise ValueError("rot_step must be 1, 2, or 3")
        rot = rot.to(dtype=dtype)
        return rot.to(device) if device is not None else rot

    if mode == QuaRotMode.ORTHOGONAL:
        rot = _random_orthogonal(size, dtype, build, seed)
        return rot.to(device) if device is not None else rot

    raise ValueError(f"Unknown QuaRotMode: {mode}")


def rotate_weight(
    weight: torch.Tensor,
    rot: torch.Tensor,
    right_rotate: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return rotated weight (does not mutate).

    ``weight`` is ``[out, in]`` or fused-expert ``[E, out, in]``.
    Right multiply hits the last dim; left multiply hits the out dim
    (``shape[-2]``), so a 3D down-proj rotates hidden channels rather than
    the expert axis. When the target dim is a multiple of ``rot`` (for
    example 4×hidden hyper-connection streams), ``rot`` is block-repeated.

    When ``device`` is set (e.g. ``cuda:N``), matmul runs there; result dtype
    matches ``weight`` and stays on ``device`` (caller / WeightStore may .cpu()).
    """
    if weight.dim() not in (2, 3):
        raise ValueError(f"rotate_weight expects 2D or 3D, got {tuple(weight.shape)}")
    dtype = weight.dtype
    dest = device if device is not None else weight.device
    w = weight.to(device=dest, dtype=torch.float32)
    r = rot.to(device=dest, dtype=torch.float32)
    dim_size = w.shape[-1] if right_rotate else w.shape[-2]
    if dim_size != r.shape[0]:
        if dim_size % r.shape[0] != 0:
            raise ValueError(f"rotate dim {r.shape[0]} must divide weight dim {dim_size}")
        r = torch.block_diag(*[r] * (dim_size // r.shape[0]))
    if right_rotate:
        out = torch.matmul(w, r)
    else:
        out = torch.matmul(r.transpose(0, 1), w)
    return out.to(dtype=dtype)


def fuse_ln_into_linear(ln_weight: torch.Tensor, linear_weight: torch.Tensor) -> torch.Tensor:
    """Absorb RMSNorm scale into Linear: W' = W * ln_weight along in_features."""
    if ln_weight.dim() != 1:
        raise ValueError("only 1D RMSNorm weight supported")
    if ln_weight.numel() != linear_weight.shape[-1]:
        raise ValueError("layernorm dim must match linear in_features")
    scale = ln_weight.float().to(linear_weight.device)
    return (linear_weight.float() * scale).to(linear_weight.dtype)


def fuse_gemma_norm_into_linear(ln_weight: torch.Tensor, linear_weight: torch.Tensor) -> torch.Tensor:
    """Absorb GemmaRMSNorm into Linear.

    Gemma stores a zero-init offset: ``y = rms(x) * (1 + gamma)``.
    Fold that effective scale into the consumer, then the caller sets gamma to 0.
    """
    if ln_weight.dim() != 1:
        raise ValueError("only 1D GemmaRMSNorm weight supported")
    if ln_weight.numel() != linear_weight.shape[-1]:
        raise ValueError(
            f"layernorm dim {ln_weight.numel()} must match linear in_features {linear_weight.shape[-1]}"
        )
    scale = (1.0 + ln_weight.float()).to(linear_weight.device)
    return (linear_weight.float() * scale).to(linear_weight.dtype)
