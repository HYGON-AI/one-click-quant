"""
Vendor module — backports of compressed_tensors functions that may not be
available in all installed versions (e.g. v0.15.0.1 lacks some entrypoints).

Keep this file self-contained with minimal external dependencies.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

import torch
import tqdm
from huggingface_hub import list_repo_files
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

__all__ = [
    "Converter",
    "InverseWeightMap",
    "build_inverse_weight_maps",
    "exec_jobs",
    "get_checkpoint_files",
    "get_weight_map",
    "gpu_if_available",
    "load_tensors_from_inverse_weight_map",
    "match_quantizable_tensors",
    "update_safetensors_index",
    "validate_safetensors_index",
    "validate_weight_for_quantization",
]

InverseWeightMap = dict[str, list[str] | None]


# ---------------------------------------------------------------------------
# Converter protocol
# ---------------------------------------------------------------------------

class Converter(Protocol):
    """
    Converter interface, to modify safetensors files based on tensor name and
    pointer to torch.Tensor, and create the QuantizationConfig.
    """

    def process(self, tensors: dict[str, torch.Tensor]):
        """Operate on safetensors file in-place."""
        raise NotImplementedError()

    def validate(self, tensors: dict[str, torch.Tensor]):
        """Validation layer."""
        raise NotImplementedError()

    def get_dependencies(self, weight_name: str) -> set[str]:
        """Return dependencies of a given weight name."""
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Inverse weight map & parallel job execution
# ---------------------------------------------------------------------------

def build_inverse_weight_maps(
    weight_map: dict[str, str],
    model_files: dict[str, str],
    converters: list[Converter],
) -> dict[str, InverseWeightMap]:
    """
    For each output shard, precompute which tensors to load from which source
    files — including partner tensors from other shards required by converters.

    :param weight_map: tensor name -> shard filename (from safetensors.index.json)
    :param model_files: shard filename -> resolved absolute path
    :param converters: list of Converter instances that may declare tensor dependencies
    :return: {shard_filename: {resolved_file_path: [tensor_names_to_load]}}
    """
    def get_dependencies_recursive(
        name: str, convs: list[Converter], current: set[str]
    ) -> set[str]:
        for conv in convs:
            for dep in conv.get_dependencies(name):
                if dep not in current:
                    current.add(dep)
                    get_dependencies_recursive(dep, convs, current)
        return current

    weight_deps_dict: dict[str, set[str]] = {}
    for wn in weight_map:
        weight_deps_dict[wn] = get_dependencies_recursive(wn, converters, set())
        assert wn not in weight_deps_dict[wn], (
            f"{wn} found in own dependencies {weight_deps_dict[wn]}"
        )

    all_dependencies: set[str] = set().union(*weight_deps_dict.values())

    iwm: dict[str, InverseWeightMap] = defaultdict(lambda: defaultdict(list))
    for wn, shard_name in weight_map.items():
        if wn in all_dependencies:
            continue
        current = iwm[shard_name]
        for name_to_add in [wn, *weight_deps_dict[wn]]:
            if name_to_add not in weight_map:
                raise ValueError(
                    f"Dependency weight {name_to_add} not found in weight map"
                )
            resolved = model_files[weight_map[name_to_add]]
            current[resolved].append(name_to_add)

    return {k: dict(v) for k, v in iwm.items()}


def exec_jobs(
    jobs: list[tuple[Callable, ...]],
    max_workers: int = 1,
    desc: str = "Executing Jobs",
) -> list:
    """
    Execute jobs in parallel using ThreadPoolExecutor.

    :param jobs: list of tuples (callable, *args)
    :param max_workers: number of worker threads
    :param desc: description for progress bar
    :return: list of results
    """
    results = []
    if max_workers == 1:
        for job in tqdm.tqdm(jobs, desc=desc):
            results.append(job[0](*job[1:]))
        return results

    with ThreadPoolExecutor(max_workers) as pool:
        futures = [pool.submit(*job) for job in jobs]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(future.result())
    return results


# ---------------------------------------------------------------------------
# Safetensors file discovery & weight map utilities
# ---------------------------------------------------------------------------

def _is_weights_file(file_name: str) -> bool:
    """Check whether a filename corresponds to a model weights file."""
    return any(
        file_name.endswith(suffix)
        for suffix in [".bin", ".safetensors", ".pth", ".msgpack", ".pt"]
    )


def _walk_directory_files(root_dir: str, ignore: Iterable[str] | None = None) -> list[str]:
    """Return relative file paths from root_dir, skipping ignored prefixes."""
    ignore = ignore or []
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
            if not any(rel_path.startswith(i) for i in ignore):
                all_files.append(rel_path)
    return all_files


def get_checkpoint_files(model_stub: str | os.PathLike) -> dict[str, str]:
    """
    Given a local path or HuggingFace model stub, return a mapping from each
    file's relative path to its resolved local path.
    """
    if os.path.exists(model_stub):
        file_paths = _walk_directory_files(
            str(model_stub), ignore=[".cache", ".gitattributes"]
        )
    else:
        file_paths = list_repo_files(model_stub)
    return {fp: cached_file(model_stub, fp) for fp in file_paths}


def _find_safetensors_index_file(model_files: dict[str, str]) -> str | None:
    """Find safetensors index file from model_files dict."""
    for file_path, resolved_path in model_files.items():
        if file_path.endswith(SAFE_WEIGHTS_INDEX_NAME):
            return resolved_path
    return None


def get_weight_map(model_files: dict[str, str]) -> dict[str, str]:
    """
    Get weight map from model_files.
    If safetensors index.json is found, weight_map is pulled from there.
    Otherwise, it is created from the single safetensors weights file.

    :returns: {weight_name -> safetensor_file_name}
    """
    index_file = _find_safetensors_index_file(model_files)
    if index_file is not None:
        with open(index_file, "r") as f:
            return json.load(f)["weight_map"]

    if SAFE_WEIGHTS_NAME not in model_files:
        raise ValueError(
            f"File {SAFE_WEIGHTS_NAME} expected but not found in {list(model_files.keys())}"
        )

    with safe_open(model_files[SAFE_WEIGHTS_NAME], framework="pt") as f:
        return {tensor: SAFE_WEIGHTS_NAME for tensor in f.keys()}


def _find_safetensors_index_path(save_directory: str | os.PathLike) -> str | None:
    """Search save_directory for a safetensors weight index file."""
    for fn in os.listdir(save_directory):
        if fn.endswith("safetensors.index.json"):
            return os.path.join(save_directory, fn)
    return None


def update_safetensors_index(
    save_directory: str | os.PathLike,
    total_size: int,
    weight_map: dict[str, str],
):
    """Write (or overwrite) the safetensors weight index file in save_directory."""
    file_path = _find_safetensors_index_path(save_directory)
    if file_path is None:
        file_path = os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME)

    with open(file_path, "w") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
            f,
            indent=2,
            sort_keys=True,
        )


def load_tensors_from_inverse_weight_map(
    inverse_weight_map: InverseWeightMap,
    device: str | torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """
    Given an inverse_weight_map, load all listed tensor names from safetensors
    files onto the specified device.

    :param inverse_weight_map: {resolved_source_file_path -> [tensor_names]}
        If list is None or empty, all tensors from that file are loaded.
    :param device: tensors will be loaded onto this device.
    :returns: {tensor_name: torch.Tensor}
    """
    tensors: dict[str, torch.Tensor] = {}
    for source_file, tensor_names in inverse_weight_map.items():
        if tensor_names is None or len(tensor_names) == 0:
            continue

        with safe_open(source_file, framework="pt", device=str(device)) as f:
            for name in tensor_names:
                if name in f.keys():
                    tensors[name] = f.get_tensor(name)
                else:
                    raise KeyError(
                        f"Tensor '{name}' not found in {source_file}. "
                        f"Available keys: {list(f.keys())}"
                    )

    return tensors


# ---------------------------------------------------------------------------
# Tensor name matching (simplified from compressed_tensors.utils)
# ---------------------------------------------------------------------------

def _match_name(name: str, target: str) -> bool:
    """Return True if target begins with 're:' and regex matches, or exact match."""
    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    return target == name


def _is_quantization_param(param_name: str) -> bool:
    """Return True if param_name represents a quantization parameter suffix."""
    return any(
        param_name.endswith(suffix)
        for suffix in (
            "_scale",
            "_zero_point",
            "_scale_inv",
            "_zero_point_inv",
        )
    )


def match_quantizable_tensors(
    tensors: dict[str, torch.Tensor],
    ignore: list[str] | None,
    targets: list[str] | None,
) -> Iterator[tuple[str, str]]:
    """
    Yield (module_name, tensor_full_name) pairs whose tensor name:
    - ends with '.weight' (implying a linear/projection layer)
    - does not end with '.norm.weight' or similar
    - does not match any entry in `ignore` by name/regex

    Quantization params (_scale, _zero_point, etc.) are always skipped.

    ``targets`` is used when actual module objects are available (class-based
    matching), e.g. ``"Linear"`` matches all ``torch.nn.Linear`` modules.  When
    only tensor names are available (this context), we match any weight tensor
    that is not a norm — the per-scheme ``targets`` list is informational only
    here because we cannot check module class types from tensor names alone.
    """
    ignore = ignore or []

    for name in list(tensors.keys()):
        if _is_quantization_param(name):
            continue

        if "." not in name:
            continue

        module_name, param_name = name.rsplit(".", 1)

        is_linear_weight = (
            param_name == "weight"
            and not module_name.endswith("norm")
            and not module_name.endswith("layernorm")
        )
        if not is_linear_weight:
            continue

        if any(_match_name(module_name, i) for i in ignore):
            continue

        yield module_name, name


# ---------------------------------------------------------------------------
# GPU device helper (originally from llmcompressor.entrypoints.model_free.helpers)
# ---------------------------------------------------------------------------

def gpu_if_available(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.accelerator.is_available():
        return torch.device(torch.accelerator.current_accelerator().type, 0)
    logger.warning("No accelerator available! Quantizing on CPU instead")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Safetensors index validation (simplified from llmcompressor)
# ---------------------------------------------------------------------------

def _is_microscale_scheme(scheme) -> bool:
    """Return True if scheme uses microscale (group_size > 0)."""
    try:
        ws = scheme.weights
        if ws and getattr(ws, "group_size", None) is not None and ws.group_size > 0:
            return True
    except Exception:
        pass
    return False


def validate_safetensors_index(
    model_files: dict[str, str],
    scheme,  # QuantizationScheme
):
    index_file = _find_safetensors_index_file(model_files)
    if index_file is None:
        return
    if _is_microscale_scheme(scheme):
        with open(index_file, "r") as f:
            weight_map = json.load(f)["weight_map"]
        invert = defaultdict(list)
        for wn, shard in weight_map.items():
            invert[shard].append(wn)
        for shard in sorted(invert):
            logger.debug(
                f"validate_safetensors_index: shard={shard}, "
                f"tensor_count={len(invert[shard])}"
            )


# ---------------------------------------------------------------------------
# Weight validation (simplified from llmcompressor lifecycle)
# ---------------------------------------------------------------------------

def validate_weight_for_quantization(
    weight: torch.Tensor,
    scheme,  # QuantizationScheme
    tensor_name: str,
):
    if weight.ndim != 2:
        raise ValueError(
            f"Unable to quantize tensor `{tensor_name}`: expected 2D linear weight, "
            f"but got shape {tuple(weight.shape)}"
        )
