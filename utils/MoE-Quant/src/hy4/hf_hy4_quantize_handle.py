"""Narrow-grid policy hook around the tool's native GPTQ handle/loop."""
import torch
from .moe_gptq_adapter import NarrowGPTQ
from .format_contract import unpack


@torch.no_grad()
def quantize_handle(handle, hessian, token_count):
    """Return packed W4 artifact and BF16 weight-only calibration candidate.

    The caller must provide token-weighted merged statistics. The temporary
    dequantized weight is for sequential calibration, not integer inference.
    Input activation quantization belongs to the actual W4A8 runtime.
    """
    quantizer = NarrowGPTQ(handle.layer)
    quantizer.use_merged_hessian(hessian, token_count)
    artifact = quantizer.export()
    integer = unpack(artifact['packed'], artifact['shape'][1])
    if artifact['scale'].dtype != torch.float32:
        raise ValueError('Channel scales must remain FP32')
    if artifact['scale'].shape != (artifact['shape'][0], 1):
        raise ValueError('Expected one scale per output channel')
    if bool((integer < -7).any()) or bool((integer > 7).any()):
        raise ValueError('Narrow INT4 range violated')
    candidate = (integer.float()*artifact['scale']).to(handle.layer.weight.dtype)
    if not torch.isfinite(candidate).all():
        raise ValueError('Nonfinite calibration candidate')
    return artifact, candidate


@torch.no_grad()
def commit_candidate(handle, candidate):
    if candidate.shape != handle.layer.weight.shape or candidate.dtype != handle.layer.weight.dtype:
        raise ValueError('Calibration candidate shape/dtype mismatch')
    handle.layer.weight.copy_(candidate)
    if hasattr(handle.layer, 'commit'):
        handle.layer.commit()
