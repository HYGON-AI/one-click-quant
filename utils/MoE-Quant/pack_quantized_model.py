# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import os
import gc
import functools
import re
import json
import argparse
from collections import defaultdict
from typing import Optional, Any

from tqdm import tqdm
import torch
from safetensors.torch import save_file
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer
from compressed_tensors.compressors import pack_to_int32

from src import quant_utils
from src import loading_utils
from src.models import ModelAdapter, get_model_adapter


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the DeepSeek model",
    )
    parser.add_argument(
        "--quantized_model_path",
        type=str,
        required=True,
        help="Path to quantized model."
    )
    parser.add_argument(
        "--packed_model_path",
        type=str,
        required=True,
        help="Whether to save packed model."
    )
     # Misc params
    parser.add_argument(
        "--dtype",
        default="float16",
        type=str,
        choices=["float16", "bfloat16"],
        help="Torch dtype used."
    )
    parser.add_argument(
        "--activation-bits",
        type=int,
        choices=[16, 8],
        default=16,
        help="Activation precision declared in quantization_config; 16 keeps weight-only W*A16, 8 declares dynamic per-token INT8 activations (W8A8 / W4A8).",
    )
    mtp_group = parser.add_mutually_exclusive_group()
    mtp_group.add_argument(
        "--mtp-int8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "RTN-quantize the non-ignored Linears of the MTP layer (which bypasses "
            "GPTQ) to INT8 channel-wise instead of copying them as bf16. Only for "
            "adapters that support it (GLM-5.3 / GLM-5.3-Flash)."
        ),
    )
    mtp_group.add_argument(
        "--mtp-int4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "RTN-quantize MTP routed experts to channel-wise INT4, and other "
            "non-ignored MTP Linears to INT8. Supported by GLM-5.3 (non-Flash). "
            "Mutually exclusive with --mtp-int8."
        ),
    )
    args = parser.parse_args()
    return args


def is_subset(set1: set, set2: set):
    return set1 <= set2


def pack_weight(
    weight: dict[torch.Tensor],
    bits: int,
    sym: bool,
    group_size: Optional[int] = None,
    packing_format: str = "pack-quantized",
) -> dict[torch.Tensor]:
    """Convert (qweight, scale, zero) into on-disk compressed-tensors artifacts.

    packing_format:
      - "pack-quantized": pack qweight into int32 lanes (compressed-tensors
        WNA16 method; used for both W4A16 and W8A16).
      - "int-quantized": emit qweight as plain int8 (compressed-tensors
        W8A8Int8 method; only valid for bits == 8).
    """
    compressed_data = {}
    qweight, scale, zero = weight['qweight'], weight['scale'], weight['zero']
    if qweight.ndim != 2:
        raise ValueError(f"qweight must be 2D, got shape {tuple(qweight.shape)}.")
    effective_group_size = qweight.shape[-1] if group_size is None else group_size
    if effective_group_size <= 0:
        raise ValueError(f"group_size must be positive, got {effective_group_size}.")
    if qweight.shape[-1] % effective_group_size != 0:
        raise ValueError(
            f"group_size ({effective_group_size}) must divide the quantized "
            f"weight input dimension ({qweight.shape[-1]})."
        )
    expected_meta_shape = (
        qweight.shape[0],
        qweight.shape[-1] // effective_group_size,
    )
    if tuple(scale.shape) != expected_meta_shape or tuple(zero.shape) != expected_meta_shape:
        raise ValueError(
            "Quantization scale and zero-point shapes must both be "
            f"{expected_meta_shape}, got scale={tuple(scale.shape)} and "
            f"zero={tuple(zero.shape)}."
        )
    # 在 int32 里做减法，最后再 cast 回 int8：
    #   - qweight 是 uint8 ∈ [0, 2**bits - 1]；.to(int8) 对 b=8 值 128..255 会 wrap，行为依赖平台。
    #   - zero 是 bf16 标量常数（对称量化下 =(maxq+1)/2，b=4 时 8.0，b=8 时 128.0）；
    #     bf16(128.0).to(int8) 在现代 PyTorch/CUDA 会饱和到 127，导致 W8A16 每个权重被系统性偏移 +scale。
    #   在 int32 做减法可以避开这两处溢出；结果由对称量化保证 ∈ [-128, 127]，最后 .to(int8) 无损。
    qweight_shifted = (
        qweight.to(torch.int32)
        - zero.repeat_interleave(effective_group_size, dim=-1).to(torch.int32)
    ).to(torch.int8)

    if packing_format == "int-quantized":
        if bits != 8:
            raise ValueError(
                f"int-quantized format only supports 8-bit weights, got bits={bits}."
            )
        if not sym:
            raise ValueError("int-quantized format requires symmetric quantization.")
        compressed_data = {
            "weight": qweight_shifted.contiguous(),
            # W8A8 loaders allocate a float32 per-channel scale.
            "weight_scale": scale.to(torch.float32),
        }
        return compressed_data

    if packing_format != "pack-quantized":
        raise ValueError(
            f"Unknown packing_format {packing_format!r}; expected "
            "'pack-quantized' or 'int-quantized'."
        )
    qweight_packed = pack_to_int32(qweight_shifted, bits)
    compressed_data = {
        "weight_packed": qweight_packed,
        "weight_shape": torch.tensor(qweight.shape),
        "weight_scale": scale
    }
    if not sym:
        compressed_data["weight_zero_point"] = weight['zero']
    return compressed_data


def pack_int8_rtn(
    weight: torch.Tensor,
    packing_format: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """RTN channel-wise INT8 for layers that bypass GPTQ (``--mtp-int8``).

    Same formula as the reference quantize_glm5_3_flash_int4_channel.py
    (abs-max / 127, round(w / s)). It runs on the accelerator like the
    reference: FP8 sources produce many exact .5 ties, and CPU/GPU division
    differ by 1 ulp there.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = weight.to(device=device, dtype=torch.float32)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    if packing_format != "int-quantized":
        # WNA16 loaders keep the scale in the model dtype.
        scale = scale.to(dtype)
    qweight = torch.round(weight / scale.float()).clamp(-127, 127) + 128
    return pack_weight(
        {
            "qweight": qweight.to(torch.uint8).cpu(),
            "scale": scale.cpu(),
            "zero": torch.full_like(scale, 128.0).cpu(),
        },
        8, True, None, packing_format=packing_format,
    )


def _packing_format_for(args: argparse.Namespace, bits: int) -> str:
    """Choose the compressed-tensors on-disk format for one layer's bit width."""
    if args.activation_bits == 8 and bits == 8:
        # vLLM's CompressedTensorsW8A8Int8MoEMethod expects `int-quantized`.
        return "int-quantized"
    # WNA16 and W4A8 (HCUCompressedTensorsW4A8Int8DynamicMoE) both read int32-packed weights.
    return "pack-quantized"


def pack_int4_rtn(weight: torch.Tensor, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Channel-wise INT4 RTN with the stored scale rounded before quantizing.

    Matches quantize_glm5_3_int4_channel.py: absmax/7, bf16 scale, signed
    [-8, 7] codes, and eight offset-binary nibbles per int32 along input dim.
    """
    if weight.ndim != 2 or weight.shape[1] % 8 != 0:
        raise ValueError("MTP INT4 requires a 2D weight with input dimension divisible by 8.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = weight.to(device=device, dtype=torch.float32)
    scale = (weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7.0).to(dtype)
    qweight = torch.round(weight / scale.float()).clamp(-8, 7) + 8
    return pack_weight(
        {
            "qweight": qweight.to(torch.uint8).cpu(),
            "scale": scale.cpu(),
            "zero": torch.full_like(scale, 8.0).cpu(),
        },
        4, True, None, packing_format="pack-quantized",
    )


def _quantize_mtp(args: argparse.Namespace) -> bool:
    return getattr(args, "mtp_int8", False) or getattr(args, "mtp_int4", False)


def output_name_for(adapter: ModelAdapter, layer_name: str, weight_map: dict[str, str]) -> str:
    """Translate a modelling-side GPTQ directory name into its checkpoint name."""
    mapped = adapter.checkpoint_keys_for_model_key(layer_name, weight_map)
    return mapped[0] if len(mapped) == 1 else layer_name


_LEAF_ORDER = ("gate_proj", "up_proj", "down_proj")


def build_layer_targets(names: list[str]) -> list[str]:
    """Collapse module names into ``re:`` targets, one per (sub-path, layer set).

    e.g. ``...layers.3.mlp.experts.0.gate_proj`` .. ``layers.44.mlp.experts.287.down_proj``
    -> ``re:.*\\.layers\\.(?:3|...|44)\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$``
    """
    layers_by_stem: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    exact = []
    for name in names:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)$", name)
        if match is None:
            exact.append(name)
            continue
        parts = match.group(2).split(".")
        stem = r"\.".join(r"\d+" if p.isdigit() else re.escape(p) for p in parts[:-1])
        layers_by_stem[stem][parts[-1]].add(int(match.group(1)))

    targets = []
    for stem, leaves in layers_by_stem.items():
        leaves_by_layers: dict[tuple[int, ...], list[str]] = defaultdict(list)
        for leaf, layers in leaves.items():
            leaves_by_layers[tuple(sorted(layers))].append(leaf)
        for layers, leaf_names in leaves_by_layers.items():
            leaf_names.sort(key=lambda x: (_LEAF_ORDER.index(x) if x in _LEAF_ORDER else len(_LEAF_ORDER), x))
            escaped = [re.escape(leaf) for leaf in leaf_names]
            leaf_re = escaped[0] if len(escaped) == 1 else "(?:" + "|".join(escaped) + ")"
            stem_re = rf"{stem}\." if stem else ""
            layer_re = "|".join(str(i) for i in layers)
            targets.append(rf"re:.*\.layers\.(?:{layer_re})\.{stem_re}{leaf_re}$")
    return targets + sorted(exact)


def resolve_ignore_for_packing(args: argparse.Namespace, adapter: ModelAdapter) -> tuple[str, list[str]]:
    """Return ``(rule_name, ignore list)`` for ``quantization_config``.

    Replays the list recorded in ``metadata.pt`` so the packed ``ignore`` can
    never disagree with the on-disk ``quantized_weight.pt`` set; metadata
    without an ``ignore`` key falls back to the legacy flag-driven list.
    Either MTP quantization flag drops the adapter's blanket MTP rules.
    """
    recorded = getattr(args, "ignore", None)
    if recorded:
        rule_name, rules = "recorded", list(recorded)
    else:
        rule_name, rules = adapter.get_quantization_ignore(args.quantize_only_experts)
    if _quantize_mtp(args):
        rules = adapter.mtp_int8_ignore(rules)
    return rule_name, rules


def _config_group(
    args: argparse.Namespace,
    targets: list[str],
    bits: int,
    input_activations: Optional[dict[str, Any]],
    packing_format: Optional[str] = None,
) -> dict[str, Any]:
    weight_strategy = "channel" if args.group_size is None else "group"
    # Channel-wise is expressed on disk as `-1` (compressed-tensors canonical
    # sentinel). vLLM's CompressedTensorsWNA16MoEMethod reads the value
    # verbatim and asserts `== -1` for 8-bit, so `null` breaks W8A16 loading.
    canonical_group_size = args.group_size if args.group_size is not None else -1
    group = {
        "input_activations": input_activations,
        "output_activations": None,
        "targets": targets,
        "weights": {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": canonical_group_size,
            "num_bits": bits,
            "observer": "minmax",
            "observer_kwargs": {},
            "strategy": weight_strategy,
            "symmetric": True,
            "type": "int"
        }
    }
    if packing_format is not None:
        # Per-group override for mixed-precision checkpoints.
        group["format"] = packing_format
    return group


def prepare_quantization_config(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    layer_bits: dict[str, int],
) -> dict[str, Any]:
    """``layer_bits`` maps every packed GPTQ layer (checkpoint name) to its bit width."""
    input_activations = None
    if args.activation_bits == 8:
        input_activations = {
            "dynamic": True,
            "group_size": None,
            "num_bits": 8,
            "observer": "minmax",
            "observer_kwargs": {},
            "strategy": "token",
            "symmetric": True,
            "type": "int",
        }

    ignore_rule, ignored_modules = resolve_ignore_for_packing(args, adapter)
    bits_used = set(layer_bits.values()) | ({8} if _quantize_mtp(args) else set())
    if getattr(args, "mtp_int4", False):
        bits_used.add(4)
    if len(bits_used) <= 1:
        bits = bits_used.pop() if bits_used else args.bits
        packing_format = _packing_format_for(args, bits)
        config_groups = {
            "group_0": _config_group(args, ["Linear"], bits, input_activations),
        }
    else:
        # Mixed precision (--force_int8 / MTP RTN): the low-bit layers are
        # listed explicitly, every other non-ignored Linear falls back to INT8.
        low_bits = sorted(bits_used - {8})
        if len(low_bits) != 1:
            raise ValueError(f"Unsupported mixed bit widths {sorted(bits_used)}.")
        low_bits = low_bits[0]
        low_bit_names = sorted(name for name, bits in layer_bits.items() if bits == low_bits)
        low_bit_targets = build_layer_targets(low_bit_names)
        if getattr(args, "mtp_int4", False):
            # The MTP bypasses GPTQ and is absent from layer_bits. Include both
            # checkpoint and NextN runtime names in the INT4 scheme explicitly.
            low_bit_targets.extend(adapter.mtp_int4_targets())
        packing_format = "pack-quantized"
        config_groups = {
            f"group_int{low_bits}": _config_group(
                args, low_bit_targets, low_bits, input_activations,
                _packing_format_for(args, low_bits),
            ),
            "group_int8": _config_group(
                args, ["Linear"], 8, input_activations, _packing_format_for(args, 8),
            ),
        }
    print(f"[INFO] quantization_config ignore rule={ignore_rule}, count={len(ignored_modules)}")
    print(f"[INFO] quantization_config format={packing_format}, groups={list(config_groups)}")
    return {
        "config_groups": config_groups,
        "format": packing_format,
        "ignore": ignored_modules,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }


def main():
    args = parse_args()

    dtype = getattr(torch, args.dtype)

    # Load model configuration and select its structural adapter.
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    adapter = get_model_adapter(config)
    print(f"[INFO] model adapter={adapter.name}")
    if args.mtp_int4 and not adapter.supports_mtp_int4:
        raise ValueError(f"--mtp-int4 is not supported by the {adapter.name} adapter.")
    adapter.prepare_config(config, world_size=1)

    with init_empty_weights():
        model = adapter.build_packing_model(config, torch.bfloat16).eval()
        model.config.use_cache = False
        adapter.prepare_model(model, config, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Load quantization metadata
    metadata = torch.load(os.path.join(args.quantized_model_path, "metadata.pt"))
    args.bits = metadata["bits"]
    args.group_size = metadata["group_size"]
    args.quantize_only_experts = metadata.get("quantize_only_experts", False)
    args.ignore = metadata.get("ignore")
    args.force_int8 = metadata.get("force_int8") or []
    # Currently we do not support asymmetric quantization
    args.sym = True
    adapter.validate_packing_args(args)
    if args.mtp_int8 and not adapter.supports_mtp_int8:
        raise ValueError(f"--mtp-int8 is not supported by the {adapter.name} adapter.")
    print(
        f"[INFO] default bits={args.bits}, force_int8={args.force_int8}, "
        f"mtp_int8={args.mtp_int8}, mtp_int4={args.mtp_int4}"
    )

    # Resolve input weights through the HuggingFace safetensors index instead of
    # assuming DeepSeek's model-xxxxx-of-000163 naming convention.
    weight_dir = args.model_name_or_path
    weight_map = loading_utils.load_safetensors_weight_map(weight_dir)
    num_extra_shards = adapter.count_extra_shards(weight_map)
    transformer_layers = adapter.get_transformer_layers(model)
    num_output_shards = len(transformer_layers) + 2 + num_extra_shards
    current_output_shard_id = 1
    quantized_layer_names = defaultdict(list)
    for layer_name in sorted(os.listdir(args.quantized_model_path)):
        layer_path = os.path.join(args.quantized_model_path, layer_name)
        if not os.path.isdir(layer_path):
            continue
        block_idx = adapter.get_block_index_from_layer_name(layer_name)
        quantized_layer_names[block_idx].append(layer_name)

    expected_quantized = adapter.expected_quantized_layer_names(model)
    if expected_quantized is not None:
        actual_quantized = {
            name for names in quantized_layer_names.values() for name in names
        }
        missing = sorted(expected_quantized - actual_quantized)
        unexpected = (
            sorted(actual_quantized - expected_quantized)
            if args.quantize_only_experts
            else []
        )
        if missing or unexpected:
            details = []
            if missing:
                details.append(
                    f"missing={len(missing)} ({', '.join(missing[:5])})"
                )
            if unexpected:
                details.append(
                    f"unexpected={len(unexpected)} ({', '.join(unexpected[:5])})"
                )
            raise ValueError(
                "Incomplete routed-expert GPTQ output: " + "; ".join(details)
            )

    # Per-layer bit widths, keyed by modelling-side directory name; the
    # quantized_weight.pt "bits" entry is cross-checked while packing.
    layer_bits = {
        name: adapter.resolve_layer_bits(name, args.bits, args.force_int8)
        for names in quantized_layer_names.values()
        for name in names
    }
    for name, bits in layer_bits.items():
        if args.activation_bits == 8 and bits != 8 and not adapter.is_routed_expert(name):
            raise ValueError(
                f"{name} is {bits}-bit but W{bits}A8 is only supported for routed "
                "experts; add it to --force_int8 (or --ignore) when quantizing."
            )
    if args.group_size is not None and (_quantize_mtp(args) or (
        args.activation_bits == 8 and 8 in layer_bits.values()
    )):
        raise ValueError(
            "INT8 W8A8 layers and MTP RTN quantization are channel-wise only; quantize "
            "without --group_size."
        )
    for bits in sorted(set(layer_bits.values())):
        count = sum(b == bits for b in layer_bits.values())
        print(f"[INFO] {count} GPTQ layer(s) bits={bits} format={_packing_format_for(args, bits)}")
    safetensors_index = {}
    # Prepare directory to save packed weights
    os.makedirs(args.packed_model_path, exist_ok=True)

    loaded_shards = set()
    param_buffer = {}
    loaded = loading_utils.ensure_model_params_loaded(
        weight_dir,
        param_buffer,
        [adapter.embedding_weight_key()],
        weight_map,
        adapter,
        loaded_shards,
    )
    if loaded:
        print(f"Loaded embedding parameter from shards: {loaded}")

    # Save embeddings
    embedding_state_dict = loading_utils.materialize_model_state_dict(
        param_buffer,
        [adapter.embedding_weight_key()],
        weight_map,
        adapter,
        dtype,
        expected_shapes={
            adapter.embedding_weight_key(): tuple(
                adapter.get_embedding_module(model).weight.shape
            ),
        },
    )
    current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
    save_file(
        {adapter.embedding_weight_key(): embedding_state_dict[adapter.embedding_weight_key()]},
        os.path.join(args.packed_model_path, current_output_shard_path)
    )
    safetensors_index[adapter.embedding_weight_key()] = current_output_shard_path
    param_buffer.pop(adapter.embedding_weight_key(), None)
    param_buffer.pop(adapter.embedding_weight_key() + "_scale_inv", None)

    # Process blocks
    for block_idx, block in tqdm(
        enumerate(transformer_layers),
        desc="Processing transformer blocks",
        total=len(transformer_layers)
    ):
        current_output_shard_id += 1
        prefix = adapter.get_layer_prefix(block_idx)
        logical_block_keys = adapter.logical_block_keys(block, block_idx)
        # Directory names are modelling-side, `logical_block_keys` are
        # checkpoint-side; translate before matching (they differ on GLM-5.3-Flash).
        quantized_output_names: dict[str, str] = {
            layer_name: output_name_for(adapter, layer_name, weight_map)
            for layer_name in quantized_layer_names[block_idx]
        }
        quantized_weight_keys = {
            f"{output_name}.weight"
            for output_name in quantized_output_names.values()
        }
        source_model_keys = logical_block_keys - quantized_weight_keys

        loading_utils.ensure_model_params_loaded(
            weight_dir,
            param_buffer,
            source_model_keys,
            weight_map,
            adapter,
            loaded_shards,
        )
        expected_shapes = {
            f"{prefix}{key}": tuple(tensor.shape)
            for key, tensor in block.state_dict().items()
            if f"{prefix}{key}" in source_model_keys
        }
        block_state_dict = loading_utils.materialize_model_state_dict(
            param_buffer,
            source_model_keys,
            weight_map,
            adapter,
            dtype,
            expected_shapes=expected_shapes,
        )

        for layer_name in quantized_layer_names[block_idx]:
            weight_state_dict = torch.load(
                os.path.join(args.quantized_model_path, layer_name, "quantized_weight.pt"),
                weights_only=True,
                map_location="cpu"
            )
            bits = layer_bits[layer_name]
            saved_bits = weight_state_dict.get("bits")
            if saved_bits is not None and int(saved_bits) != bits:
                raise ValueError(
                    f"{layer_name} was quantized with bits={int(saved_bits)} but "
                    f"metadata.pt (bits={args.bits}, force_int8={args.force_int8}) "
                    f"resolves to bits={bits}."
                )
            packed_weight_state_dict = pack_weight(
                weight_state_dict,
                bits,
                args.sym,
                args.group_size,
                packing_format=_packing_format_for(args, bits),
            )
            output_name = quantized_output_names[layer_name]
            block_state_dict.pop(f"{output_name}.weight", None)
            block_state_dict.pop(f"{output_name}.weight_scale_inv", None)
            block_state_dict.update({f"{output_name}.{k}": v for k, v in packed_weight_state_dict.items()})

        # Save block
        current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
        save_file(
            block_state_dict,
            os.path.join(args.packed_model_path, current_output_shard_path)
        )
        for k in block_state_dict:
            safetensors_index[k] = current_output_shard_path

        # Shard loading may have populated additional tensors sharing this block's
        # file (notably compact source MXFP4 experts). Evict the entire prefix.
        for key in list(param_buffer):
            if key.startswith(prefix):
                param_buffer.pop(key, None)

        del block_state_dict
        gc.collect()

    final_tensor_keys = adapter.get_final_tensor_keys()
    loading_utils.ensure_model_params_loaded(
        weight_dir,
        param_buffer,
        final_tensor_keys,
        weight_map,
        adapter,
        loaded_shards,
    )
    final_state_dict = loading_utils.materialize_model_state_dict(
        param_buffer,
        final_tensor_keys,
        weight_map,
        adapter,
        dtype,
    )

    # Save final tensors
    current_output_shard_id += 1
    current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
    save_file(
        final_state_dict,
        os.path.join(args.packed_model_path, current_output_shard_path)
    )
    for key in final_tensor_keys:
        safetensors_index[key] = current_output_shard_path
    extra_kwargs = {}
    if _quantize_mtp(args):
        int8_format = _packing_format_for(args, 8)
        _, mtp_ignore = resolve_ignore_for_packing(args, adapter)
        extra_kwargs = {
            "mtp_int8_ignore": mtp_ignore,
            "mtp_pack_int8": functools.partial(
                pack_int8_rtn, packing_format=int8_format, dtype=dtype
            ),
        }
        if args.mtp_int4:
            extra_kwargs["mtp_pack_int4"] = functools.partial(pack_int4_rtn, dtype=dtype)
    current_output_shard_id = adapter.save_extra_weights(
        weight_dir,
        weight_map,
        args.packed_model_path,
        current_output_shard_id + 1,
        num_output_shards,
        safetensors_index,
        **extra_kwargs,
    )
    # Save safetensors index
    with open(os.path.join(args.packed_model_path, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {}, "weight_map": safetensors_index},
            f,
            indent=2,
        )
        f.write("\n")
    # Add quantization metadata
    quantization_config = prepare_quantization_config(
        args,
        adapter,
        {
            output_name_for(adapter, name, weight_map): bits
            for name, bits in layer_bits.items()
        },
    )
    adapter.set_quantization_config(config, quantization_config)
    # Save configs
    config.save_pretrained(args.packed_model_path)
    model.generation_config.save_pretrained(args.packed_model_path)
    # Save tokenizer
    tokenizer.save_pretrained(args.packed_model_path)
    # Copy adapter-specific remote-code and multimodal companion assets.
    adapter.copy_artifacts(args.model_name_or_path, args.packed_model_path)


if __name__ == "__main__":
    main()
