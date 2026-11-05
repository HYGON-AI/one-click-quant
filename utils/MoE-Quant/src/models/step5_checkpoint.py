"""Stream Step-5 GPTQ artifacts into the existing channel W4A8 layout."""

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
        if item.output.removesuffix(".weight") in adapter.layer_bits or item.output.removesuffix(".weight") in adapter.mtp_names:
            module = item.output.removesuffix(".weight")
            result[module + ".weight_packed"] = ("I32", (shape[0], shape[1] // 8))
            result[module + ".weight_scale"] = ("F32", (shape[0], 1))
            result[module + ".weight_shape"] = ("I64", (2,))
        else:
            if dtype in get_reference_converter().FP8:
                dtype = "BF16"
            result[item.output] = (dtype, tuple(shape))
    return result


def compression_config(adapter):
    converter = get_reference_converter()
    result = copy.deepcopy(adapter.raw_config["quantization_config"])
    group = converter.compression_config(adapter.raw_config, adapter.plan)["config_groups"]["group_w4a8"]
    expert_targets = list(group["targets"])
    expert_names = {item.output.removesuffix(".weight") for item in adapter.plan if item.int4}
    targets = sorted((set(adapter.layer_bits) - expert_names) | adapter.mtp_names)
    # Runtime fuses q/k/v and gate/up, and renames the MTP prediction head.
    runtime = [
        r"re:.*model\.layers\.\d+\.(?:mtp_block\.)?self_attn\.(?:qkv_proj|qkvg_proj)$",
        r"re:.*model\.layers\.\d+\.(?:mtp_block\.)?(?:mlp|share_expert)\.gate_up_proj$",
        r"re:.*model\.layers\.\d+\.moe\.share_expert\.(?:gate_up_proj|down_proj)$",
        r"re:.*model\.layers\.\d+\.shared_head\.head$",
    ]
    group["targets"] = expert_targets + targets + runtime
    result["config_groups"] = {"group_w4a8": group}
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
    for item in adapter.plan:
        module = item.output.removesuffix(".weight")
        selected = converter.matches(module, group["targets"]) and not converter.matches(module, result["ignore"])
        if selected != (module in adapter.layer_bits or module in adapter.mtp_names):
            raise ValueError(f"Step5 config targets disagree with output: {module}")
    return result


def pack_artifact(path, module, shape):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    codes, scales, zeros = (artifact[k] for k in ("qweight", "scale", "zero"))
    if artifact.get("bits") != 4 or codes.dtype != torch.uint8 or tuple(codes.shape) != tuple(shape):
        raise ValueError(f"Invalid Step5 GPTQ codes/bits: {module}")
    if tuple(scales.shape) != (shape[0], 1) or tuple(zeros.shape) != (shape[0], 1):
        raise ValueError(f"Invalid Step5 channel metadata: {module}")
    if not torch.isfinite(scales).all() or not (scales > 0).all() or not (zeros == 8).all():
        raise ValueError(f"Invalid Step5 scales/zero points: {module}")
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


def pack_rtn(value, module, fp8_scale=None):
    converter = get_reference_converter()
    if value.ndim != 2 or value.shape[1] % 8:
        raise ValueError(f"Step5 MTP INT4 requires [N, K] with K divisible by 8: {module}")
    packed = torch.empty((value.shape[0], value.shape[1] // 8), dtype=torch.int32)
    scale = torch.empty((value.shape[0], 1), dtype=torch.float32)
    for begin in range(0, value.shape[0], 256):
        end = min(begin + 256, value.shape[0])
        block = value[begin:end]
        if fp8_scale is not None:
            block = block.float() * fp8_scale[begin:end].float().reshape(-1, 1)
        codes, scales, _ = converter.quantize_int4(block)
        packed[begin:end] = codes.cpu()
        scale[begin:end] = scales.cpu()
    return {
        module + ".weight_packed": packed,
        module + ".weight_scale": scale,
        module + ".weight_shape": torch.tensor(value.shape, dtype=torch.int64),
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
    wanted = set(adapter.layer_bits)
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
            if module in adapter.layer_bits:
                _, shape = adapter.meta[item.source]
                shape = shape[1:] if item.expert is not None else shape
                output = pack_artifact(intermediate / module / "quantized_weight.pt", module, shape)
                for key, value in output.items():
                    writer.add(key, value)
                counts["gptq_int4_matrices"] += 1
            elif module in adapter.mtp_names:
                value = read(item.source, item.expert)
                scale = None
                if adapter.meta[item.source][0] in converter.FP8:
                    scale = read(item.source.removesuffix(".weight") + ".weight_scale", item.expert)
                for key, tensor in pack_rtn(value, module, fp8_scale=scale).items():
                    writer.add(key, tensor)
                counts["mtp_rtn_int4_matrices"] += 1
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
