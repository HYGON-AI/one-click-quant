from typing import Optional

import torch


def weight_quant_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric INT8 quantization with absmax scaling.

    :param tensor: 2D weight tensor
    :returns: (int8_weights, float32_scale)
    """
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def weight_quant_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric FP8 (E4M3) quantization with absmax scaling.

    :param tensor: 2D weight tensor
    :returns: (fp8_weights, float32_scale)
    """
    assert tensor.dim() == 2
    qmax = torch.finfo(torch.float8_e4m3fn).max
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = tensor / scale
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.float8_e4m3fn), scale.to(torch.float32)


def weight_quant_int4(
    tensor: torch.Tensor, percentile: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel INT4 quantization with percentile-based clipping and
    packing of 2 int4 values into 1 int8.

    :param tensor: 2D weight tensor (typically already int8-quantized values)
    :param percentile: fraction of columns used for absmax (0.0–1.0)
    :returns: (packed_int4_weights, float32_scale, dequantized_float)
    """
    assert tensor.dim() == 2
    qmax = 7.0
    abs_value = torch.abs(tensor)
    sorted_matrix, _ = torch.sort(abs_value, dim=1)
    k = tensor.shape[1]
    index = int(k * percentile)
    index = min(index, k - 1)
    abs_max = sorted_matrix[:, index].reshape(-1, 1)
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -8, 7).to(torch.int8)
    dequant = quantized * scale
    return pack_int4_to_int8(quantized), scale.to(torch.float32), dequant


def weight_quantint4_search_k(
    tensor: torch.Tensor,
    k_min: float = 0.98,
    k_max: float = 1.0,
    steps: int = 25,
    metric: str = "mse",
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Grid search over percentile values to find the best clipping threshold.

    :param tensor: 2D weight tensor
    :param k_min: minimum percentile
    :param k_max: maximum percentile
    :param steps: number of search steps
    :param metric: "mse" or "l2"
    :returns: (best_packed_weights, best_scale, best_percentile)
    """
    assert tensor.dim() == 2
    percentiles = torch.linspace(k_min, k_max, steps, device=tensor.device)
    best_error: float = float("inf")
    best_k: Optional[float] = None
    best_quant: Optional[torch.Tensor] = None
    best_scale: Optional[torch.Tensor] = None

    for p in percentiles:
        p_val = float(p.item())

        q_int4, scale, dequant = weight_quant_int4(tensor, p_val)

        if metric == "mse":
            error = torch.mean((tensor - dequant) ** 2).item()
        elif metric == "l2":
            error = torch.norm(
                tensor.to(torch.float16) - dequant.to(torch.float16)
            ).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if error < best_error:
            best_error = error
            best_k = p_val
            best_quant = q_int4
            best_scale = scale

    assert best_quant is not None and best_scale is not None and best_k is not None
    return best_quant, best_scale, best_k


def pack_int4_to_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Pack two int4 values (stored in int8 range [-8,7]) into one int8 byte.

    Packing order: |w0|w1|w2|w3|w4|w5|... → |w0<<4|w1, w2<<4|w3, ...|

    :param tensor: int8 tensor with values in [-8, 7]
    :returns: packed int8 tensor with shape (n, k // 2)
    """
    n, k = tensor.shape
    assert k % 2 == 0, f"Column dimension must be even, got {k}"
    quantized_uint8 = tensor.to(torch.uint8)
    new_shape = (n, k // 2)
    a = quantized_uint8[..., ::2]
    b = quantized_uint8[..., 1::2]
    b_4bit = b & 0x0F
    packed = ((a & 0x0F) << 4) | b_4bit
    return packed.contiguous().to(torch.int8)


def unpack_int4_from_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Unpack packed int8 weights back to int4 values (one per int8, range [-8,7]).

    :param tensor: packed int8 tensor with shape (n, k // 2)
    :returns: int8 tensor with shape (n, k)
    """
    n, half_k = tensor.shape
    k = half_k * 2
    tensor_uint8 = tensor.to(torch.uint8)
    high = (tensor_uint8 >> 4).to(torch.int8)
    low = (tensor_uint8 & 0x0F).to(torch.int8)
    high = high - 16 * (high >= 8)
    low = low - 16 * (low >= 8)
    unpacked = torch.empty((n, k), dtype=torch.int8, device=tensor.device)
    unpacked[..., ::2] = high
    unpacked[..., 1::2] = low
    return unpacked


def split_fused_moe_experts(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Find fused MoE experts (with gate_up_proj/down_proj).
    Split them from 3D tensors into individual 2D expert tensors.

    Args:
        tensors: Dictionary of loaded tensors from safetensors file

    Returns:
        split_tensors: New dictionary with split expert weights
    """
    from loguru import logger

    split_tensors = {}

    params_to_split = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "down_proj": ["down_proj"],
    }

    for name, tensor in tensors.items():
        keys_to_split = [key for key in params_to_split if key in name]
        if len(keys_to_split) >= 2:
            raise ValueError(f"Found multiple keys matching {name}: {keys_to_split}")

        elif len(keys_to_split) == 1 and tensor.ndim == 3:
            unsplit_name = keys_to_split[0]
            split_names = params_to_split[unsplit_name]

            num_experts = tensor.shape[0]

            if tensor.shape[1] % len(split_names) != 0:
                raise ValueError(
                    f"{unsplit_name} expects a second dimension divisible by "
                    f"{len(split_names)} but got shape: {tensor.shape}"
                )

            intermediate_size = tensor.shape[1] // len(split_names)
            for expert_idx in range(num_experts):
                expert_tensor = tensor[expert_idx]
                split_layers = expert_tensor.split(intermediate_size, dim=0)
                for split_name, split_layer in zip(split_names, split_layers):
                    key = name.replace(
                        unsplit_name, f"{expert_idx}.{split_name}.weight"
                    )
                    split_tensors[key] = split_layer

            logger.info(f"Split {name} into {num_experts} experts")

        else:
            split_tensors[name] = tensor

    return split_tensors
