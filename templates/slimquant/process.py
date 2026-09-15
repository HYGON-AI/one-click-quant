# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationScheme
from loguru import logger
from safetensors.torch import save_file

from templates.slimquant._vendor import (
    Converter,
    InverseWeightMap,
    load_tensors_from_inverse_weight_map,
    match_quantizable_tensors,
    validate_weight_for_quantization,
)

from templates.slimquant.helpers import (
    split_fused_moe_experts,
    weight_quant_fp8,
    weight_quant_int8,
    weight_quantint4_search_k,
)


def process_file_slimquant(
    inverse_weight_map: InverseWeightMap,
    save_path: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: set[str] | None,
    device: str | torch.device,
    converter: Converter | None = None,
    quant_type: str = "int4",
    moe_pattern: str | None = None,
    k_min: float | None = None,
    k_max: float | None = None,
    k_steps: int | None = None,
    search_metric: str | None = None,
) -> tuple[int, dict[str, str]]:
    """Quantize tensors in a safetensors file.

    Supported quant_type values:
      - ``"int8"``   per-channel INT8 for all Linear weights
      - ``"fp8"``    per-channel FP8 (E4M3) for all Linear weights
      - ``"int4"``   per-channel INT8 for all Linear weights, plus INT4 with
        percentile grid-search and 2×int4→int8 packing for MoE expert layers
        (those matching ``moe_pattern``)

    :param inverse_weight_map: mapping of source file path -> tensor names
    :param save_path: output path for the quantized safetensors file
    :param scheme: quantization scheme
    :param ignore: set of module name patterns to skip
    :param device: device for quantize computation
    :param converter: optional Converter for tensor preprocessing
    :param quant_type: "int8", "fp8", or "int4"
    :param moe_pattern: substring identifying MoE expert layers (int4 mode only)
    :param k_min: minimum percentile for grid search
    :param k_max: maximum percentile for grid search
    :param k_steps: number of grid-search steps
    :param search_metric: "mse" or "l2"
    :returns: (total_bytes, weight_map)
    """
    tensors = load_tensors_from_inverse_weight_map(inverse_weight_map, device)

    tensors = split_fused_moe_experts(tensors)

    if converter is not None:
        converter.process(tensors)

    ignore_list = list(ignore) if ignore else []

    for module_name, name in match_quantizable_tensors(
        tensors, ignore_list, scheme.targets
    ):
        validate_weight_for_quantization(tensors[name], scheme, name)

        weight = tensors[name]

        original_shape: Optional[torch.Size] = None
        if weight.dim() == 3:
            E, N, K = weight.shape
            weight = weight.reshape(E * N, K)
            original_shape = weight.shape  # (E*N, K)

        scale_name = f"{name}_scale"

        if quant_type == "fp8":
            q, s = weight_quant_fp8(weight.to(torch.float32))

        elif quant_type == "int8":
            q, s = weight_quant_int8(weight)

        elif quant_type == "int4":
            int8_w, int8_s = weight_quant_int8(weight)

            if moe_pattern and moe_pattern in name:
                int4_w, int4_s, best_k = weight_quantint4_search_k(
                    int8_w,
                    k_min=k_min if k_min is not None else 0.97,
                    k_max=k_max if k_max is not None else 1.0,
                    steps=k_steps if k_steps is not None else 30,
                    metric=search_metric if search_metric is not None else "mse",
                )
                q = int4_w
                s = int4_s * int8_s / 16
                logger.debug(
                    f"[{name}] MoE INT4: best_percentile={best_k:.4f}"
                )
            else:
                q = int8_w
                s = int8_s
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")

        if original_shape is not None:
            E, N, K = tensors[name].shape
            out_cols = q.shape[-1]
            q = q.reshape(E, N, out_cols)
            s = s.reshape(E, N, 1)

        del tensors[name]
        tensors[name] = q
        tensors[scale_name] = s

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    save_file(tensors, save_path)
    total_size = sum(t.nbytes for t in tensors.values())
    weight_map = {key: os.path.basename(save_path) for key in tensors.keys()}
    return total_size, weight_map
