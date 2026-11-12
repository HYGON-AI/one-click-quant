#!/usr/bin/env python3
"""Step 5 channel FP8 -> channel INT4 W4A8 (RTN, without calibration).

Preserves source ignore rules, including backbone experts in layers 88-90.
Reads one expert at a time; supports separate 2D experts and 3D expert banks.
Writes compressed-tensors offset-binary INT32 packing and FP32 channel scales.
This converts checkpoint files; Step 5 runtime/kernel support is still required.

python quantize_step5_fp8_channel_to_w4a8.py --input-dir MODEL --dry-run
python quantize_step5_fp8_channel_to_w4a8.py \
    --input-dir MODEL --output-dir OUTPUT --device cuda:0
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EXPERT = re.compile(
    r"^(?P<bank>(?:.*\.)?model\.layers\.(?P<layer>\d+)\.moe)"
    r"(?:\.experts)?\.(?:(?P<expert>\d+)\.)?"
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
FP8 = {"F8_E4M3", "F8_E5M2", "F8_E4M3FN", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}
FLOAT = {"BF16", "F16", "F32"}
SCALE_SUFFIXES = (".weight_scale", ".weight_scale_inv")


def matches(module: str, rules: list[str]) -> bool:
    return any(
        bool(re.match(rule[3:], module)) if rule.startswith("re:") else module == rule
        for rule in rules
    )


def load_config(root: Path) -> dict:
    config = json.loads((root / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if config.get("model_type") != "step5_vl":
        raise ValueError("Expected model_type=step5_vl")
    groups = list(quant.get("config_groups", {}).values())
    if quant.get("quant_method") != "compressed-tensors" or len(groups) != 1:
        raise ValueError("Expected one compressed-tensors channel FP8 source group")
    weights = groups[0].get("weights", {})
    if any(weights.get(k) != v for k, v in {
        "num_bits": 8, "type": "float", "strategy": "channel",
        "symmetric": True, "dynamic": False,
    }.items()):
        raise ValueError("Expected static symmetric per-channel FP8 source weights")
    if groups[0].get("targets") != ["Linear"]:
        raise ValueError("Expected source targets=['Linear']")
    rules = quant.get("ignore", [])
    if not isinstance(rules, list) or any(not isinstance(x, str) or not x for x in rules):
        raise ValueError("ignore must be a list of module names or re: expressions")
    for rule in rules:
        if rule.startswith("re:"):
            re.compile(rule[3:])
    text = config["text_config"]
    layers = text["moe_layer_list"]
    if (not layers or len(set(layers)) != len(layers)
            or any(type(i) is not int or not 0 <= i < text["num_hidden_layers"] for i in layers)):
        raise ValueError("Invalid backbone moe_layer_list")
    return config


def read_metadata(root: Path):
    """Read indexed shards only, avoiding duplicate ViT/MTP sidecars."""
    index = root / "model.safetensors.index.json"
    source_index = json.loads(index.read_text()) if index.exists() else None
    weight_map = source_index["weight_map"] if source_index else {}
    filenames = sorted(set(weight_map.values())) if source_index else [
        p.name for p in sorted(root.glob("*.safetensors"))
    ]
    meta = {}
    skipped = Counter()
    for filename in filenames:
        path = root / filename
        if path.resolve().parent != root.resolve():
            raise ValueError(f"Expected a top-level shard: {filename}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if source_index:
                    selected = weight_map.get(name)
                    if selected is None:
                        raise ValueError(f"Tensor absent from source index: {name} in {filename}")
                    if selected != filename:
                        skipped[filename] += 1
                        continue
                if name in meta:
                    raise ValueError(f"Duplicate tensor: {name}; provide an unambiguous index")
                tensor = handle.get_slice(name)
                meta[name] = (tensor.get_dtype(), tuple(tensor.get_shape()))
                if not source_index:
                    weight_map[name] = filename
    if set(meta) != set(weight_map):
        missing = sorted(set(weight_map) - set(meta))
        raise ValueError(f"Source index points to missing tensors: {missing[:10]}")
    if skipped:
        print(f"Skipped {skipped.total()} shadowed tensor copies in indexed shards: {dict(skipped)}")
    return weight_map, meta


@dataclass(frozen=True)
class Matrix:
    source: str
    output: str
    expert: int | None
    int4: bool


def build_plan(config: dict, meta: dict):
    text = config["text_config"]
    rules = config["quantization_config"].get("ignore", [])
    plan, consumed, banks = [], set(), {}
    count = text["moe_num_experts"]
    for name, (dtype, shape) in sorted(meta.items()):
        if name.endswith(SCALE_SUFFIXES):
            continue
        if name.endswith((".weight_packed", ".weight_shape", ".weight_zero_point")):
            raise ValueError(f"Already packed or unsupported source tensor: {name}")
        match = EXPERT.fullmatch(name)
        if match and text["num_hidden_layers"] <= int(match["layer"]) < (
            text["num_hidden_layers"] + text.get("num_nextn_predict_layers", 0)
        ):
            match = None  # Direct MTP banks, like transformer.* MTP weights, stay floating point.
        if match:
            layer = int(match["layer"])
            if layer not in text["moe_layer_list"]:
                raise ValueError(f"Unexpected routed expert layer: {name}")
            expected = ((text["hidden_size"], text["moe_intermediate_size"])
                        if match["proj"] == "down_proj" else
                        (text["moe_intermediate_size"], text["hidden_size"]))
            bank = match["bank"] + ".experts"
            if match["expert"] is None:
                if shape != (count, *expected):
                    raise ValueError(f"Expected 3D [E,N,K] expert bank: {name}, {shape}")
                ids = range(count)
            else:
                if shape != expected or not 0 <= int(match["expert"]) < count:
                    raise ValueError(f"Unexpected 2D expert shape/id: {name}, {shape}")
                ids = [int(match["expert"])]
            for expert in ids:
                output = f"{bank}.{expert}.{match['proj']}.weight"
                aliases = [name.removesuffix(".weight"), output.removesuffix(".weight"), bank]
                int4 = not any(matches(alias, rules) for alias in aliases)
                if int4 and (dtype not in FP8 or expected[1] % 8):
                    raise ValueError(f"INT4 target needs FP8 and K divisible by 8: {name}")
                group = banks.setdefault((layer, bank), {})
                key = (expert, match["proj"])
                if key in group:
                    raise ValueError(f"Duplicate expert projection: {output}")
                group[key] = int4
                plan.append(Matrix(name, output, expert if len(shape) == 3 else None, int4))
        else:
            # Only routed experts are quantized. Everything else keeps source precision,
            # with any retained FP8 decoded to BF16 so it can be excluded in the output.
            plan.append(Matrix(name, name, None, False))
        if dtype in FP8:
            scale = name.removesuffix(".weight") + ".weight_scale"
            if not name.endswith(".weight") or scale not in meta:
                raise ValueError(f"FP8 tensor has no per-channel .weight_scale: {name}")
            if len(shape) not in (2, 3):
                raise ValueError(f"Unsupported FP8 rank: {name}, {shape}")
            scale_dtype, scale_shape = meta[scale]
            if scale_dtype not in FLOAT or scale_shape not in (shape[:-1], (*shape[:-1], 1)):
                raise ValueError(f"Invalid channel scale shape/dtype: {scale}, {meta[scale]}")
            consumed.add(scale)
        elif dtype not in FLOAT and match:
            raise ValueError(f"Unsupported expert dtype: {name}, {dtype}")
    if set(k for k in meta if k.endswith(SCALE_SUFFIXES)) != consumed:
        raise ValueError("Unexpected/orphan source scales; block FP8 and stale scales are unsupported")
    expected_parts = {(e, p) for e in range(count) for p in ("gate_proj", "up_proj", "down_proj")}
    if Counter(layer for layer, _ in banks) != Counter({i: 1 for i in text["moe_layer_list"]}):
        raise ValueError("Checkpoint must contain exactly one expert bank for each configured MoE layer")
    for (layer, _), parts in banks.items():
        if set(parts) != expected_parts or len(set(parts.values())) != 1:
            raise ValueError(f"Incomplete or mixed-precision expert bank in layer {layer}")
    if len({p.output for p in plan}) != len(plan):
        raise ValueError("Output tensor names collide")
    if not any(p.int4 for p in plan):
        raise ValueError("No non-ignored FP8 routed experts found")
    return plan, consumed


def compression_config(config: dict, plan: list[Matrix]) -> dict:
    result = copy.deepcopy(config["quantization_config"])
    banks = sorted({p.output.split(".experts.")[0] + ".experts" for p in plan if p.int4})
    targets = [rf"re:^(?:.*\.)?{re.escape(bank)}(?:\.\d+\.(?:gate_proj|up_proj|down_proj))?$"
               for bank in banks]
    result.update(
        format="pack-quantized", quantization_status="compressed",
        global_compression_ratio=None,
        config_groups={"group_w4a8": {
            "targets": targets, "format": "pack-quantized",
            "weights": dict(num_bits=4, type="int", strategy="channel", symmetric=True,
                            dynamic=False, group_size=None, observer="minmax"),
            "input_activations": dict(num_bits=8, type="int", strategy="token",
                                      symmetric=True, dynamic=True, group_size=None),
            "output_activations": None,
        }},
    )
    # Keep the original ignore list verbatim. Explicit targets cover experts only.
    for item in plan:
        module = item.output.removesuffix(".weight")
        selected = matches(module, targets) and not matches(module, result.get("ignore", []))
        if selected != item.int4:
            raise ValueError(f"Output config does not match converted precision: {module}")
    return result


def quantize_int4(weight: torch.Tensor):
    """CT pack-quantized: eight q+8 nibbles per INT32, lowest column first."""
    weight = weight.float()
    if weight.ndim != 2 or weight.shape[1] % 8 or not torch.isfinite(weight).all():
        raise ValueError("INT4 needs a finite matrix with K divisible by 8")
    maximum = weight.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 7)
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    quant = torch.round(weight / scale).clamp(-8, 7).to(torch.int32)
    lanes = (quant + 8).reshape(weight.shape[0], -1, 8)
    shifts = torch.arange(8, device=weight.device, dtype=torch.int32) * 4
    packed = (lanes << shifts).sum(-1, dtype=torch.int32)
    return packed, scale, torch.tensor(weight.shape, dtype=torch.int64)


class ShardWriter:
    def __init__(self, root: Path, limit: int):
        self.root, self.limit = root, limit
        self.pending, self.weight_map = {}, {}
        self.pending_bytes = self.total_bytes = self.count = 0

    def add(self, name: str, tensor: torch.Tensor):
        if name in self.pending or name in self.weight_map:
            raise ValueError(f"Duplicate output tensor: {name}")
        size = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + size > self.limit:
            self.flush()
        self.pending[name] = tensor.detach().cpu().contiguous().clone()
        self.pending_bytes += size

    def flush(self):
        if not self.pending:
            return
        self.count += 1
        filename = f"model-{self.count:05d}.safetensors"
        temporary = self.root / (filename + ".tmp")
        save_file(self.pending, temporary, metadata={"format": "pt"})
        temporary.replace(self.root / filename)
        self.weight_map.update({name: filename for name in self.pending})
        self.total_bytes += self.pending_bytes
        self.pending, self.pending_bytes = {}, 0


def convert(root: Path, output: Path, config: dict, weight_map: dict,
            meta: dict, plan: list[Matrix], device: str, shard_bytes: int):
    writer = ShardWriter(output, shard_bytes)
    stats = Counter()
    with ExitStack() as stack:
        handles = {filename: stack.enter_context(safe_open(root / filename, framework="pt", device="cpu"))
                   for filename in sorted(set(weight_map.values()))}

        def read(name: str, expert: int | None):
            handle = handles[weight_map[name]]
            return (handle.get_tensor(name) if expert is None else handle.get_slice(name)[expert])

        for i, item in enumerate(plan, 1):
            value = read(item.source, item.expert)
            if meta[item.source][0] in FP8:
                scale = read(item.source.removesuffix(".weight") + ".weight_scale", item.expert)
                value = value.to(device).float()
                scale = scale.to(device).float().reshape(*value.shape[:-1], 1)
                if not torch.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError(f"Invalid FP8 channel scale: {item.source}")
                value = value * scale
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite decoded FP8 weight: {item.source}")
                if not item.int4:
                    value = value.bfloat16()
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Retained FP8 weight overflows BF16: {item.source}")
                    stats["retained_fp8_decoded_to_bf16"] += 1
            if item.int4:
                packed, scale, shape = quantize_int4(value.to(device))
                module = item.output.removesuffix(".weight")
                writer.add(module + ".weight_packed", packed)
                writer.add(module + ".weight_scale", scale)
                writer.add(module + ".weight_shape", shape)
                stats["int4_matrices"] += 1
            else:
                writer.add(item.output, value)
                stats["kept_tensors"] += 1
            if i % 1000 == 0 or i == len(plan):
                print(f"Converted {i}/{len(plan)} matrices/tensors", flush=True)
    writer.flush()
    for path in root.iterdir():
        if path.is_file() and (
            path.suffix == ".py" or path.name.startswith((
                "tokenizer", "special_tokens", "added_tokens", "chat_template",
                "preprocessor", "processor", "video_preprocessor", "generation_config",
                "vocab", "merges", "LICENSE",
            ))
        ):
            shutil.copy2(path, output / path.name)
    target_config = copy.deepcopy(config)
    target_config["quantization_config"] = compression_config(config, plan)
    index = {"metadata": {"total_size": writer.total_bytes}, "weight_map": writer.weight_map}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(target_config, indent=2, ensure_ascii=False) + "\n")
    print(f"Done: {dict(stats)}, payload={writer.total_bytes / 1e9:.3f} GB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N (also used by ROCm/HCU PyTorch)")
    parser.add_argument("--shard-size-gb", type=float, default=2)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Inspect configuration/headers without writing output")
    args = parser.parse_args()
    if args.shard_size_gb <= 0 or args.num_threads <= 0:
        parser.error("shard-size-gb and num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    config = load_config(args.input_dir)
    text = config["text_config"]
    rules = config["quantization_config"].get("ignore", [])
    candidates = [i for i in text["moe_layer_list"]
                  if not matches(f"model.layers.{i}.moe.experts", rules)]
    print(f"Configuration candidates for W4A8: MoE layers {candidates}")
    print("Source ignore rules (preserved):\n  " + "\n  ".join(rules))
    weight_map, meta = read_metadata(args.input_dir)
    if not meta:
        if args.dry_run:
            print("CONFIG ONLY: no safetensors found; tensor names, dtypes, scales and completeness are NOT validated.")
            return
        raise FileNotFoundError("No source safetensors found; config.json alone cannot be quantized")
    plan, _ = build_plan(config, meta)
    compression_config(config, plan)
    print(f"Validated plan: {dict(Counter('int4' if p.int4 else 'keep' for p in plan))}")
    if args.dry_run:
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for conversion")
    source, target = args.input_dir.resolve(), args.output_dir.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Input and output must be separate, non-nested directories")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    convert(source, target, config, weight_map, meta, plan, args.device, int(args.shard_size_gb * 1e9))


if __name__ == "__main__":
    main()
