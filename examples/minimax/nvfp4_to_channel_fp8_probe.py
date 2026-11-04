# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Validate NVFP4 -> FP32 -> channel FP8 on selected checkpoint matrices.

This is a matrix conversion probe, not a complete checkpoint exporter.
The source checkpoint is read only. No accelerator or FP4 hardware is used.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


@torch.no_grad()
def unpack_e2m1(packed):
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("expected a 2D uint8 packed matrix")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1)
    codes = codes.reshape(packed.shape[0], packed.shape[1] * 2)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=packed.device
    )
    values = lut[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -values, values)


@torch.no_grad()
def dequantize_nvfp4(packed, block_scale, global_scale):
    """Decode compressed-tensors NVFP4 with group_size=16.

    In this checkpoint convention the stored global scale is an encoding
    multiplier. Effective dequantization scale is block_scale/global_scale.
    input_global_scale belongs to activation quantization, not weight decoding.
    """
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    if cols % 16 or tuple(block_scale.shape) != (rows, cols // 16):
        raise ValueError("expected one block scale per 16 input columns")
    gs = global_scale.float()
    if gs.numel() != 1 or not torch.isfinite(gs).all() or not (gs > 0).all():
        raise ValueError("expected one finite positive global scale")
    scale = block_scale.float() / gs
    if not torch.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("invalid block scales")
    values = unpack_e2m1(packed)
    return (values.reshape(rows, cols // 16, 16) * scale.unsqueeze(-1)).reshape(
        rows, cols
    )


@torch.no_grad()
def quantize_channel_fp8(weight):
    """Return E4M3FN weights and FP32 [out_features, 1] decode scales."""
    weight = weight.float()
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("expected a finite 2D matrix")
    amax = weight.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    quantized = (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scale


def relative_rmse(actual, reference):
    return float(
        (actual - reference).double().norm() / reference.double().norm().clamp_min(1e-30)
    )


def load_matrix(model, weight_map, prefix):
    result = {}
    for suffix in [
        "weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"
    ]:
        name = prefix + "." + suffix
        with safe_open(str(model / weight_map[name]), framework="pt", device="cpu") as f:
            result[suffix] = f.get_tensor(name)
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)

    # An independent installed implementation checks both decoding steps.
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    from compressed_tensors.quantization.lifecycle.forward import dequantize

    # Check all 256 byte patterns, including signed zero.
    all_bytes = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    own = unpack_e2m1(all_bytes)
    ref = unpack_fp4_from_uint8(all_bytes, 16, 32, dtype=torch.float32)
    assert torch.equal(own, ref) and torch.equal(torch.signbit(own), torch.signbit(ref))

    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    results = []
    for layer, expert in [(3, 0), (31, 63), (59, 127)]:
        for projection in ["w1", "w2", "w3"]:
            prefix = "language_model.model.layers.{}.block_sparse_moe.experts.{}.{}".format(
                layer, expert, projection
            )
            data = load_matrix(args.model, weight_map, prefix)
            packed = data["weight_packed"]
            rows, cols = packed.shape[0], packed.shape[1] * 2
            weight = dequantize_nvfp4(
                packed, data["weight_scale"], data["weight_global_scale"]
            )
            values_ref = unpack_fp4_from_uint8(packed, rows, cols, dtype=torch.float32)
            weight_ref = dequantize(
                values_ref,
                data["weight_scale"].float(),
                global_scale=data["weight_global_scale"],
                dtype=torch.float32,
            )
            assert torch.equal(weight, weight_ref), "weight decoding mismatch: " + prefix
            del values_ref, weight_ref
            q, scale = quantize_channel_fp8(weight)
            reconstructed = q.float() * scale
            x = torch.randn(8, cols)
            result = {
                "matrix": prefix,
                "shape": [rows, cols],
                "stored_weight_global_scale": float(data["weight_global_scale"]),
                "stored_input_global_scale": float(data["input_global_scale"]),
                "reference_dequantization_exact": True,
                "fp8_dtype": str(q.dtype),
                "fp8_scale_shape": list(scale.shape),
                "weight_relative_rmse": relative_rmse(reconstructed, weight),
                "random_linear_relative_rmse": relative_rmse(x @ reconstructed.t(), x @ weight.t()),
                "weight_absmax": float(weight.abs().max()),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            del data, packed, weight, q, scale, reconstructed, x

    report = {
        "torch": torch.__version__,
        "device": "cpu",
        "all_256_byte_patterns_match_reference": True,
        "source_unchanged": True,
        "scope": "nine full expert matrices; no activation quantization or model inference",
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    print("REPORT=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
