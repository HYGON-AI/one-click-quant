"""Stream Step-5 GPTQ artifacts into the existing mixed channel W4A8/W8A8 layout."""

from collections import Counter
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import shutil
import struct

import torch

from .step5_adapter import get_reference_converter


def read_header(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return {k: v for k, v in json.loads(handle.read(size)).items() if k != "__metadata__"}


def expected_schema(adapter):
    result = {}
    for item in adapter.plan:
        dtype, source_shape = adapter.meta[item.source]
        shape = source_shape[1:] if item.expert is not None else source_shape
        module = item.output.removesuffix(".weight")
        bits = adapter.layer_bits.get(module, 8 if module in adapter.mtp_names else None)
        if bits == 4:
            result[module + ".weight_packed"] = ("I32", (shape[0], shape[1] // 8))
            result[module + ".weight_scale"] = ("F32", (shape[0], 1))
            result[module + ".weight_shape"] = ("I64", (2,))
        elif bits == 8:
            result[module + ".weight"] = ("I8", tuple(shape))
            result[module + ".weight_scale"] = ("F32", (shape[0], 1))
        else:
            if dtype in get_reference_converter().FP8:
                dtype = "BF16"
            result[item.output] = (dtype, tuple(shape))
    return result


def compression_config(adapter):
    converter = get_reference_converter()
    if adapter.quantize_only_experts:
        # Experts-only: nothing outside the routed expert banks is quantized, so
        # the converter's single W4A8 group already is the whole recipe and the
        # source ignore list is kept as-is.
        result = converter.compression_config(adapter.raw_config, adapter.plan)
        _verify_precision(result, adapter, converter, [result["config_groups"]["group_w4a8"]])
        return result

    result = copy.deepcopy(adapter.raw_config["quantization_config"])
    group4 = converter.compression_config(adapter.raw_config, adapter.plan)["config_groups"]["group_w4a8"]
    expert_names = {item.output.removesuffix(".weight") for item in adapter.plan if item.int4}
    int8_names = sorted(set(adapter.layer_bits) - expert_names)
    # Runtime fuses q/k/v and gate/up, and renames the MTP prediction head.
    runtime = [
        r"re:.*model\.layers\.\d+\.(?:mtp_block\.)?self_attn\.(?:qkv_proj|qkvg_proj)$",
        r"re:.*model\.layers\.\d+\.(?:mtp_block\.)?(?:mlp|share_expert)\.gate_up_proj$",
        r"re:.*model\.layers\.\d+\.moe\.share_expert\.(?:gate_up_proj|down_proj)$",
        r"re:.*model\.layers\.\d+\.shared_head\.head$",
    ]
    group8 = {
        "targets": int8_names + sorted(adapter.mtp_names) + runtime,
        "format": "int-quantized",
        "weights": dict(num_bits=8, type="int", strategy="channel", symmetric=True,
                        dynamic=False, group_size=None, observer="minmax"),
        "input_activations": dict(num_bits=8, type="int", strategy="token",
                                  symmetric=True, dynamic=True, group_size=None),
        "output_activations": None,
    }
    result["config_groups"] = {"group_int4": group4, "group_int8": group8}
    result.update(format="pack-quantized", quantization_status="compressed",
                  global_compression_ratio=None)
    # Exact targets avoid inadvertently quantizing routers, norms, ViT or eh_proj.
    result["ignore"] = [
        rule for rule in result.get("ignore", [])
        if rule not in (
            r"re:.*lm_head$",
            r"re:.*model\.layers\.\d+\.self_attn\..*",
            r"re:.*model\.layers\.\d+\.mlp\..*",
            r"re:.*model\.layers\.\d+\.share_expert\..*",
            r"re:.*model\.layers\.\d+\.transformer\..*",
        )
    ]
    _verify_precision(result, adapter, converter, [group4, group8])
    return result


def _verify_precision(result, adapter, converter, groups):
    """Every planned tensor must be selected by exactly the group matching its bits."""
    for item in adapter.plan:
        module = item.output.removesuffix(".weight")
        ignored = converter.matches(module, result.get("ignore", []))
        hits = [g for g in groups if converter.matches(module, g["targets"])]
        selected_bits = None if ignored else (hits[0]["weights"]["num_bits"] if hits else None)
        expected_bits = adapter.layer_bits.get(module, 8 if module in adapter.mtp_names else None)
        if selected_bits != expected_bits:
            raise ValueError(
                f"Step5 config targets disagree with output: {module} "
                f"(selected={selected_bits}, planned={expected_bits}, groups_matched={len(hits)})"
            )


def pack_artifact(path, module, shape, bits):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    codes, scales, zeros = (artifact[k] for k in ("qweight", "scale", "zero"))
    if artifact.get("bits") != bits or codes.dtype != torch.uint8 or tuple(codes.shape) != tuple(shape):
        raise ValueError(f"Invalid Step5 GPTQ codes/bits: {module}")
    if tuple(scales.shape) != (shape[0], 1) or tuple(zeros.shape) != (shape[0], 1):
        raise ValueError(f"Invalid Step5 channel metadata: {module}")
    center = 2 ** (bits - 1)
    if not torch.isfinite(scales).all() or not (scales > 0).all() or not (zeros == center).all():
        raise ValueError(f"Invalid Step5 scales/zero points: {module}")
    if bits == 4:
        if (codes > 15).any():
            raise ValueError(f"Step5 INT4 code exceeds 15: {module}")
        # GPTQ stores offset binary codes already; do not add the offset again.
        lanes = codes.to(torch.int32).reshape(shape[0], -1, 8)
        shifts = torch.arange(8, dtype=torch.int32) * 4
        packed = (lanes << shifts).sum(-1, dtype=torch.int32)
        return {
            module + ".weight_packed": packed,
            module + ".weight_scale": scales.float().contiguous(),
            module + ".weight_shape": torch.tensor(shape, dtype=torch.int64),
        }
    # Do the subtraction before narrowing to int8: uint8 code 255 and zero=128
    # must never wrap or saturate independently.
    signed = (codes.to(torch.int32) - zeros.to(torch.int32)).to(torch.int8)
    return {
        module + ".weight": signed.contiguous(),
        module + ".weight_scale": scales.float().contiguous(),
    }


def pack_rtn_int8(value, module, fp8_scale=None):
    """RTN channel-wise INT8 for an MTP Linear that bypasses GPTQ.

    Same abs-max/127 formula as pack_quantized_model.py's pack_int8_rtn,
    adapted to this file's streaming block-of-256 loop and FP8 dequant.
    """
    if value.ndim != 2:
        raise ValueError(f"Step5 MTP INT8 requires a 2D weight: {module}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    signed = torch.empty(value.shape, dtype=torch.int8)
    scale = torch.empty((value.shape[0], 1), dtype=torch.float32)
    for begin in range(0, value.shape[0], 256):
        end = min(begin + 256, value.shape[0])
        block = value[begin:end].to(device=device, dtype=torch.float32)
        if fp8_scale is not None:
            block = block * fp8_scale[begin:end].to(device=device, dtype=torch.float32).reshape(-1, 1)
        block_scale = block.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
        codes = torch.round(block / block_scale).clamp(-127, 127).to(torch.int8)
        signed[begin:end] = codes.cpu()
        scale[begin:end] = block_scale.float().cpu()
    return {
        module + ".weight": signed.contiguous(),
        module + ".weight_scale": scale,
    }


def validate_schema(root, weight_map, expected):
    if set(weight_map) != set(expected):
        raise ValueError(f"Step5 output key mismatch: missing={len(set(expected)-set(weight_map))}, "
                         f"extra={len(set(weight_map)-set(expected))}")
    for shard in sorted(set(weight_map.values())):
        header = read_header(root / shard)
        if set(header) != {key for key in weight_map if weight_map[key] == shard}:
            raise ValueError(f"Step5 output header/index keys differ: {shard}")
        for key, value in header.items():
            if weight_map.get(key) != shard:
                raise ValueError(f"Unindexed Step5 output tensor {key}")
            actual = value["dtype"], tuple(value["shape"])
            if actual != expected[key]:
                raise ValueError(f"Step5 output schema mismatch for {key}: {actual} != {expected[key]}")


def pack_checkpoint(adapter, args, metadata=None):
    converter = get_reference_converter()
    source, target = adapter.source, Path(args.packed_model_path).resolve()
    intermediate = Path(args.quantized_model_path).resolve()
    for other in (source, intermediate):
        if target == other or target in other.parents or other in target.parents:
            raise ValueError("Step5 packed output must be separate from source and GPTQ artifacts.")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Step5 packed output must be empty: {target}")
    if int(args.max_shard_size_mb) <= 0:
        raise ValueError("Step5 max shard size must be positive.")
    actual = {path.parent.name for path in intermediate.glob("*/quantized_weight.pt")}
    # RTN-packed modules are read from the source checkpoint, so a GPTQ artifact
    # for them (lm_head still goes through GPTQ) is ignored rather than required.
    wanted = set(adapter.layer_bits) - adapter.rtn_int8_names
    actual -= adapter.rtn_int8_names
    if actual != wanted:
        raise ValueError(f"Step5 GPTQ artifacts mismatch: missing={len(wanted-actual)}, extra={len(actual-wanted)}")
    target.mkdir(parents=True, exist_ok=True)
    incomplete = target / ".incomplete"
    incomplete.write_text("Step5 GPTQ export in progress\n")
    writer = converter.ShardWriter(target, int(args.max_shard_size_mb) * 1024**2)
    counts = Counter()
    from safetensors import safe_open
    with ExitStack() as stack, torch.inference_mode():
        handles = {
            name: stack.enter_context(safe_open(str(source / name), framework="pt", device="cpu"))
            for name in sorted(set(adapter.weight_map.values()))
        }

        def read(name, expert):
            handle = handles[adapter.weight_map[name]]
            return handle.get_tensor(name) if expert is None else handle.get_slice(name)[expert]

        for i, item in enumerate(adapter.plan, 1):
            module = item.output.removesuffix(".weight")
            if module in adapter.rtn_int8_names:
                value = read(item.source, item.expert)
                scale = None
                if adapter.meta[item.source][0] in converter.FP8:
                    scale = read(item.source.removesuffix(".weight") + ".weight_scale", item.expert)
                for key, tensor in pack_rtn_int8(value, module, fp8_scale=scale).items():
                    writer.add(key, tensor)
                counts["mtp_rtn_int8_matrices" if module in adapter.mtp_names
                       else "rtn_int8_matrices"] += 1
            elif module in adapter.layer_bits:
                bits = adapter.layer_bits[module]
                _, shape = adapter.meta[item.source]
                shape = shape[1:] if item.expert is not None else shape
                output = pack_artifact(intermediate / module / "quantized_weight.pt", module, shape, bits)
                for key, value in output.items():
                    writer.add(key, value)
                counts[f"gptq_int{bits}_matrices"] += 1
            else:
                value = read(item.source, item.expert)
                if adapter.meta[item.source][0] in converter.FP8:
                    scale = read(item.source.removesuffix(".weight") + ".weight_scale", item.expert)
                    value = (value.float() * scale.float().reshape(*value.shape[:-1], 1)).bfloat16()
                writer.add(item.output, value)
                counts["kept_tensors"] += 1
            if i % 1000 == 0 or i == len(adapter.plan):
                print(f"Step5 packed {i}/{len(adapter.plan)}", flush=True)
        writer.flush()
    validate_schema(target, writer.weight_map, expected_schema(adapter))
    for path in source.iterdir():
        if path.is_file() and (path.suffix == ".py" or path.name.startswith((
            "tokenizer", "special_tokens", "added_tokens", "chat_template",
            "preprocessor", "processor", "generation_config", "vocab", "merges", "LICENSE",
        ))):
            shutil.copy2(path, target / path.name)
    config = copy.deepcopy(adapter.raw_config)
    config["quantization_config"] = compression_config(adapter)
    (target / "moe_quant_provenance.json").write_text(json.dumps({
        **adapter.quantization_metadata(), "counts": dict(counts),
        "step5_calibration": (metadata or {}).get("step5_calibration"),
    }, ensure_ascii=False, indent=2) + "\n")
    (target / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": writer.total_bytes}, "weight_map": writer.weight_map,
    }, indent=2) + "\n")
    temp = target / "config.json.tmp"
    temp.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    temp.replace(target / "config.json")
    incomplete.unlink()
    print(f"Step5 GPTQ export complete: {target}; {dict(counts)}; {writer.total_bytes / 1e9:.3f} GB", flush=True)
