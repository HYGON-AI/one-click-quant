"""Bounded-memory MiMo packing: main GPTQ artifacts, original MTP RTN."""

from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .mimo_v2_adapter import get_reference_converter


def expected_output_metadata(adapter) -> dict:
    """Derive the complete packed tensor schema without reading tensor payloads."""
    converter = get_reference_converter()
    result = {}
    for name, tensor in adapter.meta.items():
        if name.endswith((".weight_scale", ".weight_scale_inv")):
            continue
        mode = adapter.layout.classify(name, tensor["dtype"])
        names = converter.output_weight_names(name)
        shapes = [list(tensor["shape"])]
        if len(names) == 3:
            shapes = [[rows * adapter.layout.source_tp, adapter.layout.config["hidden_size"]]
                      for rows in converter.qkv_sizes(name, adapter.layout)]
        elif mode == "w4a8":
            shapes[0][1] *= 2
        for output_name, shape in zip(names, shapes):
            module = output_name.removesuffix(".weight")
            if mode == "keep":
                result[output_name] = {"dtype": tensor["dtype"], "shape": shape}
            else:
                result[module + ".weight_scale"] = {"dtype": "F32", "shape": [shape[0], 1]}
                if mode == "w4a8":
                    result[module + ".weight_packed"] = {"dtype": "I32", "shape": [shape[0], shape[1] // 8]}
                    result[module + ".weight_shape"] = {"dtype": "I64", "shape": [2]}
                else:
                    result[output_name] = {"dtype": "I8", "shape": shape}
    return result


def _pack_gptq(path: Path, module: str, bits: int, shape: list[int]) -> dict:
    weight = torch.load(path, map_location="cpu", weights_only=True)
    if weight.get("bits") != bits:
        raise ValueError(f"MiMo GPTQ bits mismatch for {module}: {weight.get('bits')} != {bits}")
    try:
        codes, scale, zero = weight["qweight"], weight["scale"], weight["zero"]
    except KeyError as error:
        raise ValueError(f"Incomplete GPTQ artifact: {path}") from error
    if codes.dtype != torch.uint8 or list(codes.shape) != shape:
        raise ValueError(f"Invalid MiMo GPTQ codes for {module}: {codes.dtype}, {tuple(codes.shape)}; expected uint8 {shape}")
    expected_scale = [shape[0], 1]
    if list(scale.shape) != expected_scale or list(zero.shape) != expected_scale:
        raise ValueError(f"MiMo GPTQ scale/zero for {module} must have channel shape {expected_scale}.")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError(f"MiMo GPTQ scale for {module} must be finite and positive.")
    center = 2 ** (bits - 1)
    if not torch.isfinite(zero).all() or not (zero == center).all():
        raise ValueError(f"MiMo GPTQ zero point for {module} must equal {center}.")
    if bits == 4 and (codes > 15).any():
        raise ValueError(f"MiMo INT4 GPTQ code exceeds 15 for {module}.")
    # Do the subtraction before narrowing to INT8: uint8 code 255 and BF16
    # zero=128 must never wrap or saturate independently.
    signed = (codes.to(torch.int32) - zero.to(torch.int32)).to(torch.int8)
    output = {module + ".weight_scale": scale.to(torch.float32).contiguous()}
    if bits == 4:
        output[module + ".weight_packed"] = get_reference_converter().pack_int4(signed)
        output[module + ".weight_shape"] = torch.tensor(shape, dtype=torch.int64)
    else:
        output[module + ".weight"] = signed.contiguous()
    return output


def _validate_output_metadata(writer, expected: dict) -> None:
    converter = get_reference_converter()
    converter.validate_output(writer, set(expected))
    for filename in sorted(set(writer.weight_map.values())):
        for name, actual in converter.header(writer.root / filename).items():
            wanted = expected[name]
            if actual["dtype"] != wanted["dtype"] or actual["shape"] != wanted["shape"]:
                raise ValueError(f"MiMo packed schema mismatch for {name}: {actual} != {wanted}")


def pack_checkpoint(adapter, args) -> None:
    converter = get_reference_converter()
    source = adapter.source
    intermediate = Path(args.quantized_model_path).resolve()
    target = Path(args.packed_model_path).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("MiMo output must be separate from and outside the source directory.")
    if intermediate == target or intermediate in target.parents or target in intermediate.parents:
        raise ValueError("MiMo packed output and GPTQ intermediate directories must be separate.")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"MiMo packed output directory is not empty: {target}")
    shard_mb = int(getattr(args, "max_shard_size_mb", 4096))
    if shard_mb <= 0:
        raise ValueError("max_shard_size_mb must be positive.")
    device_name = getattr(args, "rtn_device", "auto")
    device = torch.device("cuda:0" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("MiMo MTP RTN requested CUDA, but CUDA/HIP is unavailable; use --rtn-device cpu.")
    expected_artifacts = set(adapter.layer_bits)
    actual_artifacts = {path.parent.name for path in intermediate.glob("*/quantized_weight.pt")}
    missing = sorted(expected_artifacts - actual_artifacts)
    unexpected = sorted(actual_artifacts - expected_artifacts)
    if missing or unexpected:
        raise ValueError(f"MiMo GPTQ artifacts do not match the fixed source scope: missing={missing[:5]}, unexpected={unexpected[:5]}")
    expected = expected_output_metadata(adapter)
    target.mkdir(parents=True, exist_ok=True)
    incomplete = target / ".incomplete"
    incomplete.write_text("MiMo GPTQ packing is incomplete; do not serve this directory.\n")
    reader = converter.TensorReader(source, adapter.weight_map)
    writer = converter.ShardWriter(target, shard_mb * 1024**2, partial=False)
    counts = Counter()
    try:
        with torch.inference_mode():
            # Source-file ordering preserves locality for unquantized tensors;
            # each GPTQ matrix is loaded once, with a bounded pending output shard.
            names = sorted(adapter.weight_map, key=lambda name: (adapter.weight_map[name], name))
            for name in tqdm(names, desc="Packing MiMo GPTQ / MTP RTN"):
                if name.endswith((".weight_scale", ".weight_scale_inv")):
                    continue
                output_names = converter.output_weight_names(name)
                modules = [item.removesuffix(".weight") for item in output_names]
                if any(module in adapter.layer_bits for module in modules):
                    if not all(module in adapter.layer_bits for module in modules):
                        raise ValueError(f"Mixed GPTQ/source fused scope: {name}")
                    for module in modules:
                        bits = adapter.layer_bits[module]
                        if bits == 4:
                            packed_shape = expected[module + ".weight_packed"]["shape"]
                            shape = [packed_shape[0], packed_shape[1] * 8]
                        else:
                            shape = expected[module + ".weight"]["shape"]
                        output = _pack_gptq(intermediate / module / "quantized_weight.pt", module, bits, shape)
                        for key, value in output.items():
                            writer.add(key, value)
                        del output
                        counts[f"gptq_int{bits}"] += 1
                else:
                    # Original source tensors feed the same RTN function used
                    # by the existing MiMo converter, including all old MTP QKV.
                    output, kind = converter.convert_tensor(name, reader, adapter.layout, device)
                    for key, value in output.items():
                        writer.add(key, value)
                    del output
                    counts[f"source_{kind}"] += 1
            writer.flush()
        _validate_output_metadata(writer, expected)
        converter.copy_artifacts(source, target)
        config = copy.deepcopy(adapter.layout.config)
        config.pop("quantization_config", None)
        config.pop("compression_config", None)
        config["quantization_config"] = converter.compression_config(adapter.layout, adapter.meta)
        config["attention_projection_layout"] = "split"
        config["dtype"] = "bfloat16"
        if "torch_dtype" in config:
            config["torch_dtype"] = "bfloat16"
        (target / "moe_quant_provenance.json").write_text(json.dumps({
            **adapter.quantization_metadata(),
            "counts": dict(counts),
        }, indent=2, ensure_ascii=False) + "\n")
        (target / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": writer.total_bytes},
            "weight_map": writer.weight_map,
        }, indent=2) + "\n")
        # Config is the last model artifact, so a failed run cannot look complete.
        (target / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        incomplete.unlink()
    finally:
        reader.close()
    print(f"MiMo packed: {dict(counts)}; tensors={len(writer.weight_map)}; payload={writer.total_bytes / 1e9:.3f} GB")
