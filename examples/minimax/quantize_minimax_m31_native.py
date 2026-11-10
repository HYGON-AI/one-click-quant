# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Export MiniMax M3.1 native NVFP4/MXFP8 weights without loading the model."""

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX = "model.safetensors.index.json"
CONFIG = "config.json"
QUANT_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_inv",
    ".weight_global_scale",
    ".input_global_scale",
)
NVFP4_SUFFIXES = (
    ".weight_packed",
    ".weight_scale",
    ".weight_global_scale",
    ".input_global_scale",
)
MXFP8_SUFFIXES = (".weight", ".weight_scale_inv")
CLIP_RATIOS = (
    0.98,
    0.96,
    0.94,
    0.92,
    0.90,
    0.875,
    0.85,
    0.825,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
    0.45,
    0.40,
)
MAIN_ARCH = "MiniMaxM3SparseForConditionalGeneration"
DRAFT_ARCH = "DSparkMiniMaxDraftModel"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def round_even(values):
    # Preserve the tie rule of m31-channel-int4-weight-mse-v1.
    lower = values.floor()
    fraction = values - lower
    tie = (fraction - 0.5).abs() <= 1e-9
    return lower + ((fraction > 0.5 + 1e-9) | (tie & (lower.int() & 1 != 0)))


def channel_int8(weight):
    weight = weight.double()
    scale = weight.abs().amax(-1, keepdim=True) / 127
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = round_even(weight / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.float()


def channel_int4(weight):
    # FP64 makes the near-tied clipping search stable across CPU and HCU.
    weight = weight.double()
    scale = weight.abs().amax(-1, keepdim=True) / 7
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    best_q = round_even(weight / scale).clamp(-7, 7)
    best_scale = scale
    best_error = (best_q * scale - weight).square().sum(-1, keepdim=True)

    def try_scale(candidate):
        nonlocal best_q, best_scale, best_error
        q = round_even(weight / candidate).clamp(-8, 7)
        error = (q * candidate - weight).square().sum(-1, keepdim=True)
        better = error < best_error * (1 - 1e-12)
        best_q = torch.where(better, q, best_q)
        best_scale = torch.where(better, candidate, best_scale)
        best_error = torch.where(better, error, best_error)

    for ratio in CLIP_RATIOS:
        try_scale(scale * ratio)
    for _ in range(2):
        candidate = (weight * best_q).sum(-1, keepdim=True)
        candidate = candidate / best_q.square().sum(-1, keepdim=True).clamp_min(1)
        try_scale(candidate.clamp_min(torch.finfo(torch.float64).tiny))

    # SlimQuant stores even K in the high nibble and a scale divided by 16.
    codes = best_q.to(torch.int8)
    packed = ((codes[:, ::2] & 15) << 4) | (codes[:, 1::2] & 15)
    return packed, best_scale.float() / 16


def channel_fp8(weight):
    scale = weight.abs().amax(-1, keepdim=True) / 448
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


class NativeDecoder:
    def __init__(self, device):
        self.device = torch.device(device)
        self.e2m1 = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.e4m3 = (
            torch.arange(256)
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
            .float()
            .to(self.device)
        )

    def decode(self, tensors, prefix, nvfp4):
        if nvfp4:
            packed = tensors.get_tensor(prefix + ".weight_packed").to(self.device)
            scales = tensors.get_tensor(prefix + ".weight_scale")
            global_scale = (
                tensors.get_tensor(prefix + ".weight_global_scale")
                .float()
                .to(self.device)
            )
            if (
                packed.dtype != torch.uint8
                or packed.ndim != 2
                or packed.shape[1] % 8
                or scales.dtype != torch.float8_e4m3fn
                or scales.shape != (packed.shape[0], packed.shape[1] // 8)
                or global_scale.numel() != 1
                or not torch.isfinite(global_scale).all()
                or not (global_scale > 0).all()
            ):
                raise ValueError(f"Invalid NVFP4 layout or scale for {prefix}")
            block = self.e4m3[scales.view(torch.uint8).to(self.device).long()]
            # Native NVFP4 stores even K in the low nibble.
            codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
            weight = self.e2m1[codes] * block.repeat_interleave(16, -1) / global_scale
        else:
            raw = tensors.get_tensor(prefix + ".weight")
            exponent = tensors.get_tensor(prefix + ".weight_scale_inv").to(self.device)
            if (
                raw.dtype != torch.float8_e4m3fn
                or raw.ndim != 2
                or raw.shape[1] % 32
                or exponent.dtype != torch.uint8
                or exponent.shape != (raw.shape[0], raw.shape[1] // 32)
                or (exponent == 255).any()
            ):
                raise ValueError(f"Invalid MXFP8 layout or scale for {prefix}")
            scale = torch.pow(2.0, exponent.float() - 127).repeat_interleave(32, -1)
            weight = self.e4m3[raw.view(torch.uint8).to(self.device).long()] * scale
        if not torch.isfinite(weight).all():
            raise ValueError(f"Non-finite decoded weight for {prefix}")
        return weight


def draft_prefixes():
    return {
        f"language_model.model.dspark.layers.{i}.decoder_layer.{name}"
        for i in range(5)
        for name in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }


def source_layout(source, recipe):
    config = read_json(source / CONFIG)
    quant = config.get("quantization_config", {})
    expected_arch = MAIN_ARCH if recipe == "w4a8" else DRAFT_ARCH
    if (
        config.get("architectures") != [expected_arch]
        or quant.get("quant_method") != "mxfp8"
        or quant.get("weight_block_size") != [1, 32]
    ):
        raise ValueError(
            f"Expected native MiniMax M3.1 {expected_arch} checkpoint: {source}"
        )
    index = read_json(source / INDEX)["weight_map"]
    shards = defaultdict(set)
    for key, filename in index.items():
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Invalid shard filename: {filename}")
        shards[filename].add(key)
    experts = {
        key.removesuffix(".weight_packed")
        for key in index
        if key.endswith(".weight_packed")
    }
    linears = {
        key.removesuffix(".weight_scale_inv")
        for key in index
        if key.endswith(".weight_scale_inv")
    }
    if recipe == "w4a8":
        if (
            len(experts) != 21888
            or len(linears) != 540
            or quant.get("moe_quant_algo") != "NVFP4"
            or quant.get("group_size") != 16
            or any(".block_sparse_moe.experts." not in name for name in experts)
        ):
            raise ValueError(
                "Unexpected MiniMax M3.1 main-model quantized matrix layout"
            )
    elif experts or linears != draft_prefixes():
        raise ValueError("Expected all 35 native MXFP8 DSpark projections")

    if experts & linears:
        raise ValueError("A matrix is indexed in both native quantization formats")
    per_shard = defaultdict(list)
    for prefix in sorted(experts | linears):
        nvfp4 = prefix in experts
        suffixes = NVFP4_SUFFIXES if nvfp4 else MXFP8_SUFFIXES
        filename = index.get(prefix + suffixes[0])
        if filename is None or any(
            index.get(prefix + suffix) != filename for suffix in suffixes
        ):
            raise ValueError(f"Missing or split native tensors for {prefix}")
        per_shard[filename].append((prefix, nvfp4))
    return config, index, shards, per_shard, experts, linears


def channel_config(kind, ignore):
    fmt = "float-quantized" if kind == "float" else "int-quantized"
    common = dict(
        type=kind,
        num_bits=8,
        symmetric=True,
        observer_kwargs={},
        actorder=None,
        block_structure=None,
        group_size=None,
        scale_dtype=None,
        zp_dtype=None,
    )
    return dict(
        quant_method="compressed-tensors",
        format=fmt,
        version="0.13.0",
        quantization_status="compressed",
        global_compression_ratio=None,
        kv_cache_scheme=None,
        sparsity_config={},
        transform_config={},
        config_groups={
            "group_0": dict(
                format=fmt,
                targets=["Linear"],
                weights=dict(
                    common, strategy="channel", dynamic=False, observer="minmax"
                ),
                input_activations=dict(
                    common, strategy="token", dynamic=True, observer=None
                ),
                output_activations=None,
            )
        },
        ignore=ignore,
    )


def output_config(config, recipe, experts, linears):
    native = config.pop("quantization_config")
    config.pop("compression_config", None)
    if recipe == "w4a8":
        config["compression_config"] = dict(
            quant_method="slimquant_w4a8",
            activation_scheme="dynamic",
            ignore=[
                r"re:.*lm_head(?:\.|$)",
                r"re:.*embed_tokens(?:\.|$)",
                r"re:.*(?:block_sparse_moe|mlp)\.gate(?:\.|$)",
                r"re:.*vision_tower.*",
                r"re:.*multi_modal_projector.*",
                r"re:.*patch_merge_mlp.*",
            ],
            mixed_precision=dict(
                experts=dict(
                    weight_bits=4,
                    format="pack-int4-nibble",
                    scope="per_channel",
                    modules=sorted(experts),
                ),
                w8a8_include=dict(
                    weight_bits=8,
                    activation_bits=8,
                    format="int8-per-channel",
                    scope="per_channel",
                    activation_scheme="dynamic_per_token",
                    modules=sorted(linears),
                ),
            ),
        )
    else:
        ignore = list(native["ignored_layers"])
        if recipe == "dspark-bf16":
            # The service can still select compressed-tensors while collecting
            # activations with all draft projections in BF16.
            ignore = sorted(set(ignore) | linears)
            ignore += [
                f"language_model.model.dspark.layers.{i}.decoder_layer.{name}"
                for i in range(5)
                for name in ("self_attn.qkv_proj", "mlp.gate_up_proj")
            ]
        config["quantization_config"] = channel_config(
            "float" if recipe == "dspark-fp8" else "int", ignore
        )
    return config


def verify_saved_shard(path, tensors):
    with safe_open(path, framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(tensors):
            raise ValueError(f"Saved tensor keys differ: {path}")
        for name, tensor in tensors.items():
            actual = saved.get_tensor(name)
            if actual.dtype != tensor.dtype or actual.shape != tensor.shape:
                raise ValueError(f"Saved tensor layout differs: {name}")
            actual = (
                actual.reshape(1, -1).view(torch.uint8)
                if tensor.ndim < 2
                else actual.view(torch.uint8)
            )
            expected = (
                tensor.reshape(1, -1).view(torch.uint8)
                if tensor.ndim < 2
                else tensor.view(torch.uint8)
            )
            if name.endswith(".weight_scale") or tensor.ndim < 2:
                equal = torch.equal(actual, expected)
            elif tensor.shape[0]:
                rows = [0, tensor.shape[0] // 2, tensor.shape[0] - 1]
                equal = torch.equal(actual[rows], expected[rows])
            else:
                continue
            if not equal:
                raise ValueError(f"Saved tensor differs: {name}")


def convert_shard(path, keys, matrices, decoder, recipe, scales, norms):
    output = {}
    consumed = set()
    with safe_open(path, framework="pt", device="cpu") as native:
        if set(native.keys()) != keys:
            raise ValueError(f"Source shard does not match index: {path}")
        for prefix, nvfp4 in matrices:
            weight = decoder.decode(native, prefix, nvfp4)
            name = prefix + ".weight"
            if name in scales:
                weight = weight * scales[name].to(decoder.device)
            if recipe == "dspark-bf16":
                converted = weight.to(torch.bfloat16)
                if not torch.equal(converted.float(), weight):
                    raise ValueError(
                        f"BF16 round-trip differs from native decode: {prefix}"
                    )
            else:
                quantize = (
                    channel_int4
                    if nvfp4
                    else channel_fp8
                    if recipe == "dspark-fp8"
                    else channel_int8
                )
                converted, scale = quantize(weight)
                if not torch.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(f"Invalid output scale: {prefix}")
                output[prefix + ".weight_scale"] = scale.cpu().contiguous()
            output[name] = converted.cpu().contiguous()
            consumed.update(
                prefix + suffix
                for suffix in (NVFP4_SUFFIXES if nvfp4 else MXFP8_SUFFIXES)
            )
        for name in sorted(keys - consumed):
            if name.endswith(QUANT_SUFFIXES):
                raise ValueError(f"Unexpected unconsumed quantization tensor: {name}")
            tensor = native.get_tensor(name)
            if name in norms:
                replacement = norms[name]
                if (
                    replacement.shape != tensor.shape
                    or replacement.dtype != tensor.dtype
                ):
                    raise ValueError(f"Smoothed norm layout differs: {name}")
                tensor = replacement
            output[name] = tensor.clone()
    return output


@torch.no_grad()
def export_checkpoint(
    source, output, recipe, device="cpu", *, scales=None, norms=None, calibration=None
):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the source checkpoint")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    config, index, shards, per_shard, experts, linears = source_layout(source, recipe)
    scales, norms = scales or {}, norms or {}
    if (
        set(scales) - {prefix + ".weight" for prefix in linears}
        or set(norms) - index.keys()
    ):
        raise ValueError("Smoothing references tensors outside the source index")
    decoder = NativeDecoder(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.lock")
    with lock.open("x"):
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{output.name}-", dir=output.parent
            ) as tmp:
                stage = Path(tmp) / "checkpoint"
                stage.mkdir()
                weight_map, payload = {}, 0
                for filename, keys in sorted(shards.items()):
                    tensors = convert_shard(
                        source / filename,
                        keys,
                        per_shard[filename],
                        decoder,
                        recipe,
                        scales,
                        norms,
                    )
                    if weight_map.keys() & tensors.keys():
                        raise ValueError(f"Duplicate output tensor in {filename}")
                    save_file(tensors, stage / filename, metadata={"format": "pt"})
                    verify_saved_shard(stage / filename, tensors)
                    payload += sum(
                        t.numel() * t.element_size() for t in tensors.values()
                    )
                    weight_map.update(dict.fromkeys(tensors, filename))
                    del tensors
                    print(f"Converted {filename}", flush=True)
                write_json(
                    stage / CONFIG, output_config(config, recipe, experts, linears)
                )
                write_json(
                    stage / INDEX,
                    dict(metadata=dict(total_size=payload), weight_map=weight_map),
                )
                for path in source.iterdir():
                    if (
                        path.is_file()
                        and not path.name.startswith(".")
                        and path.suffix != ".safetensors"
                        and path.name
                        not in {
                            CONFIG,
                            INDEX,
                            "quantization_report.json",
                            "VALIDATION.json",
                        }
                    ):
                        shutil.copy2(path, stage / path.name)
                report = dict(
                    recipe=recipe,
                    source=str(source),
                    tensors=len(weight_map),
                    nvfp4_matrices=len(experts),
                    mxfp8_matrices=len(linears),
                    tensor_bytes=payload,
                )
                if calibration is not None:
                    report["calibration"] = calibration
                write_json(stage / "quantization_report.json", report)
                if output.exists():
                    raise FileExistsError(f"Output already exists: {output}")
                stage.rename(output)
        finally:
            lock.unlink()
    print(f"Exported {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        choices=("w4a8", "dspark-fp8", "dspark-int8", "dspark-bf16"),
        required=True,
    )
    parser.add_argument(
        "--device", default="cpu", help="Quantization device, e.g. cpu or cuda:0"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    export_checkpoint(args.source, args.output, args.recipe, args.device)


if __name__ == "__main__":
    main()
