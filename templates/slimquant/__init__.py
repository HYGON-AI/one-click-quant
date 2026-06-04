import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from loguru import logger

from templates.slimquant._vendor import (
    Converter,
    build_inverse_weight_maps,
    exec_jobs,
    get_checkpoint_files,
    get_weight_map,
    gpu_if_available,
    update_safetensors_index,
    validate_safetensors_index,
)

from templates.slimquant.config import (
    update_slimquant_config,
)
from templates.slimquant.process import (
    process_file_slimquant,
)

# ---- built-in scheme definitions ----

_INT8_SCHEME = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    ),
)

_FP8_SCHEME = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    ),
)

_SCHEMES = {
    "int8": _INT8_SCHEME,
    "fp8": _FP8_SCHEME,
    "int4": _INT8_SCHEME,  # int4 reuses INT8 base + second-stage MoE INT4
}


def slimquant_ptq(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme | None = None,
    ignore: set[str] | None = None,
    max_workers: int = 1,
    device: Optional[torch.device | str] = None,
    converter: Converter | None = None,
    quant_type: str = "int4",
    moe_pattern: str = ".mlp.experts.",
    k_min: float = 0.97,
    k_max: float = 1.0,
    k_steps: int = 30,
    search_metric: str = "mse",
):
    """Quantize a model directly from safetensors files.

    Supports three quantization types — pass ``quant_type`` and the
    corresponding scheme is selected automatically:

    - ``"int8"``: per-channel INT8 for all Linear weights
    - ``"fp8"``:  per-channel FP8 (E4M3) for all Linear weights
    - ``"int4"``: per-channel INT8 for all Linear weights, plus INT4 with
      percentile grid-search and 2×int4→int8 packing for MoE expert layers

    :param model_stub: huggingface model hub or path to local weights files
    :param save_directory: directory to save quantized weights to
    :param scheme: optional override QuantizationScheme. If ``None``, the
        built-in scheme for ``quant_type`` is used.
    :param ignore: set of module names or ``"re:..."`` regex patterns to skip.
        Modules ending with ``"norm"`` are automatically ignored.
    :param max_workers: number of worker threads
    :param device: GPU device to accelerate quantization with
    :param converter: optional Converter for tensor preprocessing
    :param quant_type: "int8", "fp8", or "int4"
    :param moe_pattern: substring identifying MoE expert layers (int4 only)
    :param k_min: minimum percentile for grid search
    :param k_max: maximum percentile for grid search
    :param k_steps: number of search steps
    :param search_metric: "mse" or "l2"
    """
    # --- resolve scheme ---
    if scheme is None:
        if quant_type not in _SCHEMES:
            raise ValueError(
                f"Unknown quant_type: {quant_type}. "
                f"Expected one of {list(_SCHEMES.keys())}"
            )
        scheme = _SCHEMES[quant_type]

    # --- validate arguments ---
    model_files = get_checkpoint_files(model_stub)

    device = gpu_if_available(device)
    validate_safetensors_index(model_files, scheme)

    # --- copy non-safetensors files ---
    for file_path, resolved_path in model_files.items():
        if not file_path.endswith("safetensors"):
            save_path = Path(save_directory) / file_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying {file_path} -> {save_path}")
            shutil.copyfile(resolved_path, save_path)

    # --- build quantization jobs ---
    weight_map = get_weight_map(model_files)
    inverse_weight_maps = build_inverse_weight_maps(
        weight_map=weight_map,
        model_files=model_files,
        converters=[converter] if converter is not None else [],
    )

    jobs: list[tuple] = []
    for shard_name in model_files.keys():
        if not shard_name.endswith("safetensors"):
            continue
        if shard_name not in inverse_weight_maps:
            raise ValueError(
                f"Could not find inverse_weight_map for shard {shard_name}"
            )
        save_path = Path(save_directory) / shard_name
        job = (
            process_file_slimquant,
            inverse_weight_maps[shard_name],
            save_path,
            scheme,
            ignore,
            device,
            converter,
            quant_type,
        )
        if quant_type == "int4":
            job += (moe_pattern, k_min, k_max, k_steps, search_metric)
        jobs.append(job)

    # --- quantize ---
    total_size = 0
    weight_map_out: dict[str, str] = {}
    quantize_results = exec_jobs(jobs, max_workers, desc="Quantizing")
    for _total_size, _weight_map in quantize_results:
        total_size += _total_size
        weight_map_out.update(_weight_map)

    # --- update config and safetensors index ---
    update_slimquant_config(
        save_directory,
        scheme=scheme,
        ignore=ignore,
        quant_type=quant_type,
    )
    update_safetensors_index(save_directory, total_size, weight_map_out)

    logger.info(f"SlimQuant {quant_type} quantization complete: {save_directory}")
