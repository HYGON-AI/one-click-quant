# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Export Smooth INT8 DSpark from native weights and saved calibration activations."""

# Reuse saved rank activations or collect them with a BF16 DSpark draft.
# Fit channel-wise smoothing scales on sampled activations and native weights.
# Fold them into RMSNorm and the corresponding weight columns before exporting
# per-channel INT8 with dynamic per-token INT8 activations. Folded norms remain
# BF16. The local quantization_report.json records the calibration inputs and
# selected fold settings.

import argparse
import hashlib
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from quantize_minimax_m31_native import (
    NativeDecoder,
    export_checkpoint,
    read_json,
    source_layout,
)

PREFIX = "language_model.model.dspark."
HIDDEN_NORM = PREFIX + "hidden_norm.weight"


def groups():
    qkv = []
    mlp = []
    for layer in range(5):
        prefix = f"{PREFIX}layers.{layer}.decoder_layer."
        for name in ("q_proj", "k_proj", "v_proj"):
            qkv.append(
                (
                    f"{prefix}self_attn.{name}",
                    prefix + "input_layernorm.weight",
                    f"layers.{layer}.self_attn.qkv_proj",
                    name != "q_proj",
                )
            )
        mlp.append(
            (
                f"gate_up_{layer}",
                [
                    (
                        f"{prefix}mlp.{name}",
                        prefix + "post_attention_layernorm.weight",
                        f"layers.{layer}.mlp.gate_up_proj",
                        False,
                    )
                    for name in ("gate_proj", "up_proj")
                ],
            )
        )
    return [("shared_qkv_and_context", qkv), *mlp]


def activation_stats(paths, expected):
    raw = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    if any(set(rank) != expected for rank in raw):
        raise ValueError(
            "Expected QKV, gate_up and hidden_norm activations on every rank"
        )
    stats = {}
    for name in sorted(expected):
        amax, samples = [], []
        for rank in raw:
            entry = rank[name]
            maxima, values = entry["amax"], entry["samples"]
            if (
                maxima.ndim != 1
                or values.ndim != 2
                or not values.shape[0]
                or values.shape[1] != maxima.numel()
                or not torch.isfinite(maxima).all()
                or not torch.isfinite(values).all()
                or (maxima < 0).any()
            ):
                raise ValueError(f"Invalid calibration activations for {name}")
            amax.append(maxima)
            samples.append(values[::4])
        rows = torch.cat(samples).float()
        indices = torch.linspace(0, len(rows) - 1, min(192, len(rows))).long()
        stats[name] = (torch.stack(amax).amax(0), rows[indices])
    return stats


def qdq(value):
    scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-20) / 127
    return (value / scale).round().clamp(-127, 127) * scale


def choose_scale(probes, norms, amax, wmax):
    def score(scale):
        # Gemma RMSNorm stores gamma - 1. Simulate the BF16 rounding of the fold.
        folded = {
            key: ((1 + norm.float()) / scale - 1).to(torch.bfloat16)
            for key, norm in norms.items()
        }
        errors, fold_errors = [], []
        for activation, weight, norm_key, target in probes:
            gamma = 1 + norms[norm_key].float()
            gamma_new = 1 + folded[norm_key].float()
            if ((gamma != 0) & (gamma_new == 0)).any():
                return float("inf"), float("inf"), folded
            ratio = torch.where(gamma != 0, gamma_new / gamma, torch.ones_like(gamma))
            adjusted = activation * ratio
            weighted = weight * scale
            prediction = qdq(adjusted) @ qdq(weighted).T
            denom = target.square().mean().clamp_min(1e-20)
            errors.append(float((prediction - target).square().mean() / denom))
            fold_errors.append(
                float(((adjusted @ weighted.T - target).square().mean() / denom).sqrt())
            )
        return sum(errors) / len(errors), max(fold_errors), folded

    identity = torch.ones_like(wmax)
    baseline, _, _ = score(identity)
    best = (baseline, 0.0, None, "identity", identity, norms)
    for step in range(1, 10):
        alpha = step / 10
        scale = amax.clamp_min(1e-5).pow(alpha) / wmax.clamp_min(1e-5).pow(1 - alpha)
        scale = (scale / scale.log().mean().exp()).clamp(0.25, 4.0)
        for mode, candidate in (
            ("continuous", scale),
            ("power2", 2 ** scale.log2().round()),
        ):
            mse, fold_error, folded = score(candidate)
            if mse < best[0] and fold_error < 0.005:
                best = (mse, fold_error, alpha, mode, candidate, folded)
    mse, fold_error, alpha, mode, scale, folded = best
    return (
        scale,
        folded,
        dict(
            alpha=alpha,
            mode=mode,
            baseline_relative_mse=baseline,
            selected_relative_mse=mse,
            fold_rrmse_max=fold_error,
        ),
    )


@torch.no_grad()
def smooth_parameters(source, stats, matrix_groups):
    _, index, shards, _, _, _ = source_layout(source, "dspark-int8")
    decoder = NativeDecoder("cpu")
    scales, folded_norms, reports = {}, {}, []
    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(
                safe_open(source / name, framework="pt", device="cpu")
            )
            for name in shards
        }

        def tensor(name):
            return readers[index[name]].get_tensor(name)

        for group, matrices in matrix_groups:
            wmax = amax = None
            probes, norms = [], {}
            for prefix, norm_key, activation_key, uses_context in matrices:
                weight = decoder.decode(
                    readers[index[prefix + ".weight"]], prefix, False
                )
                if not torch.equal(weight.to(torch.bfloat16).float(), weight):
                    raise ValueError(f"Calibration BF16 round-trip differs: {prefix}")
                maxima = weight.abs().amax(0)
                wmax = maxima if wmax is None else torch.maximum(wmax, maxima)
                rows = torch.linspace(0, weight.shape[0] - 1, 48).long().unique()
                sampled = weight[rows]
                uses = [(norm_key, activation_key)]
                if uses_context:
                    uses.append((HIDDEN_NORM, "hidden_norm"))
                for norm, activations in uses:
                    norms[norm] = tensor(norm)
                    act_max, sample = stats[activations]
                    if (
                        norms[norm].dtype != torch.bfloat16
                        or norms[norm].shape != maxima.shape
                        or act_max.shape != maxima.shape
                    ):
                        raise ValueError(f"Calibration channels do not match {prefix}")
                    amax = act_max if amax is None else torch.maximum(amax, act_max)
                    probes.append((sample, sampled, norm, sample @ sampled.T))
            scale, new_norms, report = choose_scale(probes, norms, amax, wmax)
            for prefix, _, _, _ in matrices:
                scales[prefix + ".weight"] = scale
            folded_norms.update(new_norms)
            reports.append(dict(group=group, **report))
    return scales, folded_norms, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Original native MXFP8 dspark directory",
    )
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Directory with activations_*.pt",
    )
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    manifest = read_json(args.calibration_manifest)
    # The original DP8 calibration manifest predates the dp_size field.
    rank_count = manifest.get("dp_size", 8)
    paths = sorted(args.activations.glob("activations_*.pt"))
    if not paths or len(paths) != rank_count:
        parser.error(f"Expected {rank_count} activation files, found {len(paths)}")
    torch.set_num_threads(args.threads)
    matrix_groups = groups()
    expected = {key for _, matrices in matrix_groups for _, _, key, _ in matrices} | {
        "hidden_norm"
    }
    scales, norms, reports = smooth_parameters(
        args.source, activation_stats(paths, expected), matrix_groups
    )
    export_checkpoint(
        args.source,
        args.output,
        "dspark-int8",
        scales=scales,
        norms=norms,
        calibration=dict(
            method="smooth",
            manifest=manifest,
            groups=reports,
            activation_files={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
        ),
    )


if __name__ == "__main__":
    main()
