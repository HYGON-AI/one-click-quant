import json
import os
from typing import Iterable, Optional

import torch
from safetensors import safe_open


SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"


def load_param_shard(weight_dir: str, weight_path: str) -> dict[str, torch.Tensor]:
    param_shard = {}
    with safe_open(os.path.join(weight_dir, weight_path), framework="pt", device="cpu") as f:
        param_shard_keys = f.keys()
        for k in param_shard_keys:
            param_shard[k] = f.get_tensor(k)
    return param_shard


def find_safetensors_index_file(weight_dir: str) -> str:
    """Return the safetensors index file in a HuggingFace weight directory."""
    preferred_path = os.path.join(weight_dir, SAFETENSORS_INDEX_NAME)
    if os.path.isfile(preferred_path):
        return preferred_path

    candidates = sorted(
        name for name in os.listdir(weight_dir)
        if name.endswith(".safetensors.index.json")
    )
    if candidates:
        return os.path.join(weight_dir, candidates[0])

    raise FileNotFoundError(
        f"Cannot find a safetensors index file under {weight_dir}. "
        f"Expected {SAFETENSORS_INDEX_NAME} or *.safetensors.index.json."
    )


def _build_weight_map_from_safetensors_files(weight_dir: str) -> dict[str, str]:
    """Fallback for unindexed safetensors directories."""
    weight_map = {}
    safetensors_files = sorted(
        name for name in os.listdir(weight_dir)
        if name.endswith(".safetensors")
    )
    if not safetensors_files:
        raise FileNotFoundError(f"Cannot find any *.safetensors files under {weight_dir}.")

    for weight_path in safetensors_files:
        with safe_open(os.path.join(weight_dir, weight_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = weight_path
    return weight_map


def load_safetensors_weight_map(weight_dir: str, allow_unindexed: bool = True) -> dict[str, str]:
    """Load parameter-name -> shard-file mapping from a HF safetensors model directory."""
    try:
        index_path = find_safetensors_index_file(weight_dir)
    except FileNotFoundError:
        if allow_unindexed:
            return _build_weight_map_from_safetensors_files(weight_dir)
        raise

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index file: {index_path}")
    return weight_map


def _expand_fp8_scale_keys(required_keys: set[str], weight_map: dict[str, str]) -> set[str]:
    expanded_keys = set(required_keys)
    for key in required_keys:
        scale_key = f"{key}_scale_inv"
        if scale_key in weight_map:
            expanded_keys.add(scale_key)
    return expanded_keys


def expand_model_keys(
    model_keys: Iterable[str],
    weight_map: dict[str, str],
    adapter,
) -> set[str]:
    """Expand logical model keys into physical checkpoint tensor keys."""
    physical_keys = set()
    for key in model_keys:
        physical_keys.update(adapter.checkpoint_keys_for_model_key(key, weight_map))
    return physical_keys


def ensure_model_params_loaded(
    weight_dir: str,
    param_buffer: dict[str, torch.Tensor],
    model_keys: Iterable[str],
    weight_map: dict[str, str],
    adapter,
    loaded_shards: Optional[set[str]] = None,
) -> list[str]:
    """Load physical checkpoint tensors needed by logical model state keys."""
    physical_keys = expand_model_keys(model_keys, weight_map, adapter)
    return ensure_params_loaded(
        weight_dir,
        param_buffer,
        physical_keys,
        weight_map,
        loaded_shards,
        include_fp8_scales=True,
    )


def materialize_model_state_dict(
    param_buffer: dict[str, torch.Tensor],
    model_keys: Iterable[str],
    weight_map: dict[str, str],
    adapter,
    dtype: torch.dtype,
    expected_shapes: Optional[dict[str, tuple[int, ...]]] = None,
) -> dict[str, torch.Tensor]:
    """Materialize logical tensors from their resident physical checkpoint data."""
    model_key_set = set(model_keys)
    physical_keys = expand_model_keys(model_key_set, weight_map, adapter)
    physical_keys.update(
        f"{key}_scale_inv"
        for key in model_key_set
        if f"{key}_scale_inv" in param_buffer
    )
    physical_state_dict = {
        key: param_buffer[key]
        for key in physical_keys
        if key in param_buffer
    }
    return adapter.materialize_state_dict(
        physical_state_dict,
        model_key_set,
        dtype,
        expected_shapes=expected_shapes,
    )


def ensure_params_loaded(
    weight_dir: str,
    param_buffer: dict[str, torch.Tensor],
    required_keys: Iterable[str],
    weight_map: dict[str, str],
    loaded_shards: Optional[set[str]] = None,
    include_fp8_scales: bool = True,
) -> list[str]:
    """Load all shards needed for required_keys into param_buffer.

    Args:
        weight_dir: Directory containing safetensors shards.
        param_buffer: Mutable CPU parameter cache.
        required_keys: Fully-qualified parameter names required by the current step.
        weight_map: Mapping loaded from model.safetensors.index.json.
        loaded_shards: Optional set tracking already-loaded shard filenames.
        include_fp8_scales: Also load companion ``*_scale_inv`` tensors when present.

    Returns:
        List of shard filenames newly loaded by this call.
    """
    required_key_set = set(required_keys)
    if include_fp8_scales:
        required_key_set = _expand_fp8_scale_keys(required_key_set, weight_map)

    missing_from_index = sorted(key for key in required_key_set if key not in weight_map)
    if missing_from_index:
        preview = ", ".join(missing_from_index[:8])
        suffix = " ..." if len(missing_from_index) > 8 else ""
        raise KeyError(
            f"{len(missing_from_index)} required parameters are missing from safetensors index: "
            f"{preview}{suffix}"
        )

    loaded_shards = loaded_shards if loaded_shards is not None else set()
    shards_to_load = []
    seen = set()
    for key in sorted(required_key_set):
        shard = weight_map[key]
        # A shard may have been loaded for an earlier block and later evicted from
        # param_buffer. Check resident keys instead of treating loaded_shards as
        # a permanent history of all shards ever seen.
        if key in param_buffer or shard in seen:
            continue
        seen.add(shard)
        shards_to_load.append(shard)

    for shard in shards_to_load:
        param_buffer.update(load_param_shard(weight_dir, shard))
        loaded_shards.add(shard)

    return shards_to_load
