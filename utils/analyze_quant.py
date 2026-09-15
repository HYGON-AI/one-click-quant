# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import Module

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ChannelOutlierReport:
    layer_name: str
    layer_type: str
    out_channels: int
    in_channels: int
    channel_max: list[float] = field(default_factory=list)
    outlier_ratio: float = 0.0
    max_abs: float = 0.0
    median_abs: float = 0.0
    mean_abs: float = 0.0
    max_to_median_ratio: float = 0.0
    outlier_channels: list[int] = field(default_factory=list)
    is_sensitive: bool = False


@dataclass
class DiagnosisReport:
    model_path: str
    total_layers: int = 0
    linear_layers: int = 0
    layer_reports: list[ChannelOutlierReport] = field(default_factory=list)
    outlier_heavy_layers: list[str] = field(default_factory=list)
    recommend_awq: bool = False
    recommend_smoothquant: bool = False
    recommend_ignore: list[str] = field(default_factory=list)
    is_moe: bool = False
    summary: str = ""


def diagnose_model(
    model: "Module",
    threshold_ratio: float = 3.0,
    outlier_ratio_threshold: float = 0.10,
    verbose: bool = True,
) -> DiagnosisReport:

    report = DiagnosisReport(model_path="<loaded model>")
    layer_reports = []
    linear_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        linear_count += 1
        weight = module.weight.data
        if not _is_cpu_reducible_dtype(weight.dtype):
            target_shape = (weight.shape[0], weight.shape[1])
            dequant = _dequantize_weight_tensor(weight, module, target_shape)
            if dequant is not None:
                weight = dequant
            else:
                weight = weight.float()

        channel_max = weight.abs().max(dim=1).values

        median = channel_max.median().item()
        if median < 1e-8:
            median = 1e-8

        max_abs_val = channel_max.max().item()
        mean_abs_val = channel_max.mean().item()

        outlier_mask = channel_max > threshold_ratio * median
        outlier_channels = outlier_mask.nonzero(as_tuple=True)[0].tolist()
        outlier_ratio = len(outlier_channels) / len(channel_max)

        max_to_median = max_abs_val / median

        layer_report = ChannelOutlierReport(
            layer_name=name,
            layer_type=type(module).__name__,
            out_channels=weight.shape[0],
            in_channels=weight.shape[1],
            channel_max=channel_max.tolist(),
            outlier_ratio=outlier_ratio,
            max_abs=max_abs_val,
            median_abs=median,
            mean_abs=mean_abs_val,
            max_to_median_ratio=max_to_median,
            outlier_channels=outlier_channels,
            is_sensitive=(outlier_ratio > outlier_ratio_threshold
                          or max_to_median > 10.0),
        )
        layer_reports.append(layer_report)

        if verbose and layer_report.is_sensitive:
            logger.info(
                "[SENSITIVE] %s: outlier_ratio=%.2f%%, max/median=%.1fx, outliers=%d/%d",
                name, outlier_ratio * 100, max_to_median,
                len(outlier_channels), weight.shape[0],
            )

    report.linear_layers = linear_count
    report.layer_reports = layer_reports

    sorted_reports = sorted(layer_reports, key=lambda r: r.outlier_ratio, reverse=True)
    cutoff = max(3, int(len(sorted_reports) * 0.25))
    report.outlier_heavy_layers = [r.layer_name for r in sorted_reports[:cutoff]
                                    if r.is_sensitive]

    num_sensitive = sum(1 for r in layer_reports if r.is_sensitive)

    meaningful = [
        r for r in layer_reports
        if r.layer_name not in ("", "lm_head")
        and "embed" not in r.layer_name
    ]
    high_outlier = [r for r in meaningful if r.max_to_median_ratio > 10.0]
    moderate_outlier = [r for r in meaningful if 5.0 < r.max_to_median_ratio <= 10.0]
    awq_score = len(high_outlier) + 0.3 * len(moderate_outlier)
    has_extreme = len(high_outlier) >= 1
    enough_layers = awq_score >= 5
    if has_extreme and enough_layers:
        report.recommend_awq = True
    if num_sensitive > linear_count * 0.3:
        report.recommend_smoothquant = True

    max_ignore = max(5, min(100, int(linear_count * 0.10)))
    report.recommend_ignore = [
        name for name in report.outlier_heavy_layers
        if name not in ("", "lm_head")
    ][:max_ignore]

    parts = [
        f"Analyzed {linear_count} Linear layers.",
        f"{num_sensitive} layers ({num_sensitive/max(1,linear_count):.1%}) are quantization-sensitive.",
    ]
    if report.recommend_awq:
        parts.append("Recommend AWQ preprocessing (significant weight outliers).")
    if report.recommend_smoothquant:
        parts.append("Recommend SmoothQuant preprocessing (widespread activation sensitivity).")
    if report.recommend_ignore:
        parts.append(f"Recommend adding {len(report.recommend_ignore)} layers to ignore list.")
    report.summary = " ".join(parts)

    if verbose:
        logger.info("Diagnosis: %s", report.summary)

    return report


def _get_tensor_by_path(obj, path: str):
    try:
        cur = obj
        for part in path.split("."):
            cur = getattr(cur, part)
        if isinstance(cur, torch.Tensor):
            return cur.detach()
    except AttributeError:
        return None
    return None


def _is_cpu_reducible_dtype(dtype) -> bool:
    if dtype.is_floating_point and str(dtype) not in ("torch.float8_e4m3fn", "torch.float8_e5m2"):
        return True
    return False


def _get_first_tensor_by_paths(obj, paths: tuple[str, ...]):
    for p in paths:
        t = _get_tensor_by_path(obj, p)
        if t is not None:
            return t
    return None


def _get_weight_tensor(module):

    if isinstance(module, nn.Linear) and hasattr(module, "weight"):
        return module.weight.detach()
    return _get_first_tensor_by_paths(
        module,
        (
            "weight",
            "linear.weight",
            "base_layer.weight",
            "orig_layer.weight",
            "wrapped.weight",
            "inner.weight",
            "quantized_weight",
            "qweight",
            "weight_q",
            "weight_int",
        ),
    )


def _broadcast_param(param, target_shape: tuple[int, int]):
    if param.numel() == 1:
        return param
    p = param
    if p.ndim == 1 and p.shape[0] == target_shape[0]:
        p = p.view(target_shape[0], 1)
    while p.ndim < 2:
        p = p.unsqueeze(-1)
    if p.shape[0] != target_shape[0]:
        return None
    if p.shape[1] not in (1, target_shape[1]):
        return None
    return p


def _get_scale_tensor(module):
    t = _get_first_tensor_by_paths(
        module,
        (
            "weight_scale",
            "weight_scale_inv",
            "weight_scales",
            "scales",
            "scale",
            "linear.weight_scale",
            "linear.weight_scale_inv",
            "linear.scales",
            "base_layer.weight_scale",
            "base_layer.weight_scale_inv",
            "base_layer.scales",
        ),
    )
    if t is not None:
        return t
    try:
        sd = module.state_dict()
    except Exception:
        return None
    for k in (
        "weight_scale",
        "weight_scale_inv",
        "weight_scales",
        "scales",
        "scale",
        "linear.weight_scale",
        "linear.weight_scale_inv",
        "linear.weight_scales",
        "linear.scales",
        "linear.scale",
    ):
        v = sd.get(k)
        if isinstance(v, torch.Tensor):
            return v.detach()
    return None


def _get_zero_point_tensor(module):
    t = _get_first_tensor_by_paths(
        module,
        (
            "weight_zero_point",
            "weight_zero_points",
            "weight_zeros",
            "zeros",
            "zero_point",
            "linear.weight_zero_point",
            "linear.zeros",
            "base_layer.weight_zero_point",
            "base_layer.zeros",
        ),
    )
    if t is not None:
        return t
    try:
        sd = module.state_dict()
    except Exception:
        return None
    for k in (
        "weight_zero_point",
        "weight_zero_points",
        "weight_zeros",
        "zeros",
        "zero_point",
        "linear.weight_zero_point",
        "linear.weight_zero_points",
        "linear.zeros",
        "linear.zero_point",
    ):
        v = sd.get(k)
        if isinstance(v, torch.Tensor):
            return v.detach()
    return None


_BLOCKWISE_CHECK_DIMS = (1, 1)


def _blockwise_scale_dequant(qweight, scale, target_shape: tuple[int, int]):
    out_dim, in_dim = int(target_shape[0]), int(target_shape[1])
    g0, g1 = int(scale.shape[0]), int(scale.shape[1])
    bs0 = (out_dim + g0 - 1) // g0
    bs1 = (in_dim + g1 - 1) // g1

    q = qweight.to(dtype=torch.float32)
    result = torch.empty(target_shape, dtype=torch.float32, device=q.device)
    for i in range(g0):
        r0 = i * bs0
        r1 = min(r0 + bs0, out_dim)
        for j in range(g1):
            c0 = j * bs1
            c1 = min(c0 + bs1, in_dim)
            result[r0:r1, c0:c1] = q[r0:r1, c0:c1] * scale[i, j]
    return result


def _apply_fp8_scale_dequant(qweight, scale, target_shape: tuple[int, int]):
    scale_bc = _broadcast_param(scale, target_shape)
    if scale_bc is not None:
        return qweight.to(dtype=torch.float32) * scale_bc
    _bd = _BLOCKWISE_CHECK_DIMS
    if scale.ndim == 2 and _bd[0] < scale.shape[0] < target_shape[0] and _bd[1] < scale.shape[1] < target_shape[1]:
        return _blockwise_scale_dequant(qweight, scale, target_shape)
    return None


def _dequantize_weight_tensor(qweight, module, target_shape: tuple[int, int]):
    scale = _get_scale_tensor(module)
    if scale is None:
        return None
    zp = _get_zero_point_tensor(module)
    scale = scale.detach().to(dtype=torch.float32)

    if zp is None:
        return _apply_fp8_scale_dequant(qweight, scale, target_shape)

    zp = zp.detach().to(dtype=torch.float32)
    zp_bc = _broadcast_param(zp, target_shape)
    if zp_bc is None:
        return None
    scale_bc = _broadcast_param(scale, target_shape)
    if scale_bc is None:
        return None
    q = qweight.to(dtype=torch.float32)
    return (q - zp_bc) * scale_bc


def correlate_error_with_outliers(weight, quantized_weight, channel_max) -> float:
    per_channel_mse = ((weight - quantized_weight) ** 2).mean(dim=1)
    mx = channel_max.mean()
    my = per_channel_mse.mean()
    cov = ((channel_max - mx) * (per_channel_mse - my)).mean()
    sx = channel_max.std()
    sy = per_channel_mse.std()
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    corr = (cov / (sx * sy)).item()
    return corr


def compute_per_channel_mse(
    base_model: "Module",
    quantized_model: "Module",
) -> list[tuple[str, float, float, list[float]]]:

    base_linears: dict[str, nn.Linear] = {}
    for name, mod in base_model.named_modules():
        if isinstance(mod, nn.Linear):
            base_linears[name] = mod

    _ptr_to_scale: dict[int, torch.Tensor] = {}
    for _name, _mod in quantized_model.named_modules():
        _w = _get_weight_tensor(_mod)
        if _w is not None:
            _s = _get_scale_tensor(_mod)
            if _s is not None:
                _ptr_to_scale[_w.data_ptr()] = _s
    for _name, _mod in base_model.named_modules():
        if isinstance(_mod, nn.Linear):
            _w = _mod.weight.detach()
            _s = _get_scale_tensor(_mod)
            if _s is not None:
                _ptr_to_scale[_w.data_ptr()] = _s

    results: list[tuple[str, float, float, list[float]]] = []
    skipped_count = 0
    recovered_count = 0
    unknown_types: dict[str, int] = {}

    with torch.no_grad():
        for name, mod in quantized_model.named_modules():
            if name not in base_linears:
                continue

            base_mod = base_linears[name]
            base_raw = base_mod.weight.detach()
            target_shape = (base_raw.shape[0], base_raw.shape[1])

            if not _is_cpu_reducible_dtype(base_raw.dtype):
                base_weight = _dequantize_weight_tensor(base_raw, base_mod, target_shape)
                if base_weight is None:
                    tied_scale = _ptr_to_scale.get(base_raw.data_ptr())
                    if tied_scale is not None:
                        tied_scale = tied_scale.to(dtype=torch.float32)
                        base_weight = _apply_fp8_scale_dequant(base_raw, tied_scale, target_shape)
                    if base_weight is None:
                        base_weight = base_raw.float()
            else:
                base_weight = base_raw.to(dtype=torch.float32)

            quant_tensor = _get_weight_tensor(mod)
            quant_weight = None
            if quant_tensor is not None:
                if tuple(quant_tensor.shape) != target_shape:
                    quant_weight = None
                else:
                    quant_weight = _dequantize_weight_tensor(quant_tensor, mod, target_shape)
                    if quant_weight is not None:
                        recovered_count += 1
                    elif quant_tensor.dtype.is_floating_point:
                        tied_scale = _ptr_to_scale.get(quant_tensor.data_ptr())
                        if tied_scale is not None:
                            tied_scale = tied_scale.to(dtype=torch.float32)
                            quant_weight = _apply_fp8_scale_dequant(quant_tensor, tied_scale, target_shape)
                            if quant_weight is not None:
                                recovered_count += 1
                        if quant_weight is None:
                            quant_weight = quant_tensor.to(dtype=torch.float32)

            if quant_weight is None:
                scale = _get_scale_tensor(mod)
                if scale is not None:
                    scale = scale.to(dtype=torch.float32)
                    scale = _broadcast_param(scale, target_shape)
                    if scale is not None:
                        zp = _get_zero_point_tensor(mod)
                        if zp is not None:
                            zp = zp.to(dtype=torch.float32)
                            zp = _broadcast_param(zp, target_shape)
                        if zp is None:
                            quant_weight = torch.round(base_weight / scale) * scale
                        else:
                            quant_weight = (torch.round(base_weight / scale + zp) - zp) * scale
                        recovered_count += 1

            if quant_weight is None:
                skipped_count += 1
                mod_type = type(mod).__name__
                unknown_types[mod_type] = unknown_types.get(mod_type, 0) + 1
                continue

            per_channel = ((base_weight - quant_weight) ** 2).mean(dim=1)

            channel_max = base_weight.abs().max(dim=1).values
            corr = correlate_error_with_outliers(base_weight, quant_weight, channel_max)

            mean_mse = per_channel.mean().item()
            results.append((name, mean_mse, corr, per_channel.detach().cpu().tolist()))

    total = len(results) + skipped_count
    logger.info(
        "Per-channel MSE: %d analyzed, %d skipped (%.1f%%), %d recovered from scale",
        len(results), skipped_count,
        skipped_count / max(1, total) * 100,
        recovered_count,
    )
    if skipped_count > total * 0.5:
        logger.warning(
            "Over 50%% of layers skipped in MSE analysis. Unknown module types: %s",
            dict(sorted(unknown_types.items(), key=lambda x: -x[1])[:10]),
        )

    return sorted(results, key=lambda x: x[1], reverse=True)


_DEFAULT_OUTLIER_THRESHOLD = 3.0
_DEFAULT_CORR_THRESHOLD = 0.7
_DEFAULT_IGNORE_MIN = 0.005
_DEFAULT_IGNORE_TOPK = 5


def add_analysis_args(parser) -> None:
    parser.add_argument(
        "--analysis", type=str, choices=["outlier", "mse", "fisher", "activation_sensi", "outlier_mse", "none"], default="none",
        help="Which analysis to run.\n"
             "  none:             - do not run analysis\n"
             "  outlier:          - outlier channel analysis (needs --model)\n"
             "  mse:              - per-channel MSE (needs --model and --save-dir)\n"
             "  fisher:           - Fisher diagonal sensitivity (needs --model)\n"
             "  activation_sensi: - activation-aware sensitivity (needs --model and --save-dir)\n"
             "                      recommended: --num-calibration-samples 128 --max-seq-length 512\n"
             "  outlier_mse:      - outlier + MSE",
    )
    parser.add_argument(
        "--analysis-top",
        type=int,
        default=20,
        help="Number of worst layers to show in analysis report.",
    )
    parser.add_argument(
        "--analysis-output",
        type=str,
        default=None,
        help="Save analysis report to file instead of stdout.",
    )




def draw_progress_bar(current, total, bar_length=100, prefix="Progress"):
    pct = 100 * current / total
    filled = int(bar_length * current // total)
    bar = '\u2588' * filled + '-' * (bar_length - filled)
    end = '\n' if current == total else ''
    print(f"\r{prefix}: |{bar}| {pct:.2f}%", end=end, flush=True)


def _run_fisher_analysis(model_path, num_samples=128, batch_size=1, seqlen=2048, seed=0):

    _WIKI_DIR = "datasets/EleutherAI___wikitext_document_level/wikitext-2-raw-v1/0.0.0/647234772b9554e208af6c826f23b99e3cac88c8"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="auto",
    )
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    testdata = Dataset.from_file(f"{_WIKI_DIR}/wikitext_document_level-test.arrow")
    testenc = tokenizer("\n\n".join(testdata['page']), return_tensors='pt')

    device = next(model.parameters()).device
    blocks = model.model.layers
    samples_in_dataset = testenc.input_ids.numel() // seqlen
    num_samples = max(1, min(num_samples, samples_in_dataset))
    num_batches = num_samples // batch_size
    #assert num_samples % batch_size == 0

    layer_registry = {}
    for block_idx, block in enumerate(blocks):
        for module_name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                param = module.weight
                param.requires_grad_(True)
                full_name = f"model.layers.{block_idx}.{module_name}"
                layer_registry[full_name] = (param, torch.zeros_like(param, device="cpu"))

    logger.info("Fisher: tracking %d linear modules across %d block layers", len(layer_registry), len(blocks))
    logger.info("Fisher: %d samples, batch_size %d, num_batches %d", num_samples, batch_size, num_batches)

    for k in range(num_batches):
        i_start = k * batch_size
        j = i_start + batch_size

        inputs = testenc.input_ids[:, (i_start * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j - i_start, seqlen)

        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        model.zero_grad()
        loss.backward()

        for full_name, (param, acc) in layer_registry.items():
            if param.grad is not None:
                acc.add_(param.grad.detach().pow(2).cpu())
                param.grad = None

        model.zero_grad()

        draw_progress_bar(k + 1, num_batches, prefix="Fisher")

    stats = []
    for name, (param, acc) in layer_registry.items():
        diag = acc / num_batches
        stats.append((name, diag.mean().item(), diag.max().item(), diag.numel()))
    stats.sort(key=lambda x: x[1], reverse=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


# ---------------------------------------------------------------------------
# Activation-aware sensitivity analysis
# ---------------------------------------------------------------------------


def _load_calibration_samples(dataset_path, tokenizer, num_samples, seqlen):
    """Load and tokenize calibration data from a dataset path.

    Supports chat-format (messages key) and plain-text format.
    Returns a tensor of shape [num_samples, seqlen].
    """
    ds = None
    try:
        _WIKI_DIR = "datasets/EleutherAI___wikitext_document_level/wikitext-2-raw-v1/0.0.0/647234772b9554e208af6c826f23b99e3cac88c8"
        ds = Dataset.from_file(f"{_WIKI_DIR}/wikitext_document_level-validation.arrow")
        logger.info("Loaded wikitext validation data from %s", _WIKI_DIR)
    except Exception:
        logger.info("Cannot load wikitext validation.arrow; trying --dataset %s", dataset_path)

    if ds is None:
        for split_name in ("validation", "test", "train_sft"):
            try:
                ds = load_dataset(dataset_path, split=split_name, trust_remote_code=True)
                break
            except Exception:
                continue

    if ds is None:
        logger.warning("Cannot load %s; falling back to wikitext-2", dataset_path)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    ds = ds.shuffle(seed=42)

    samples = []
    for example in ds:
        if "messages" in example:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        elif "text" in example:
            text = example["text"]
        elif "page" in example:
            text = example["page"]
        else:
            continue
        if not text or not isinstance(text, str):
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < seqlen:
            continue
        samples.append(torch.tensor(tokens[:seqlen], dtype=torch.long))
        if len(samples) >= num_samples:
            break

    if len(samples) < num_samples:
        logger.warning(
            f"Only {len(samples)} valid samples available (requested {num_samples}). "
            f"Try a different --dataset or reduce --num-calibration-samples."
        )
        if len(samples) == 0:
            raise RuntimeError("No valid calibration samples found.")
    return torch.stack(samples)


def _capture_one_batch(model, inputs):
    """Run a single forward pass and capture every nn.Linear output.

    Returns a dict mapping ``"layer_{idx}.{rel_path}"`` -> output tensor (CPU float32).
    """
    import torch.nn as nn

    captured: dict[str, torch.Tensor] = {}
    hooks = []

    for layer_idx, layer in enumerate(model.model.layers):
        for name, mod in layer.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            key = f"layer_{layer_idx}.{name}"

            def _make_hook(k):
                def _hook(module, args, output):
                    captured[k] = output.detach().float().cpu()
                return _hook

            hooks.append(mod.register_forward_hook(_make_hook(key)))

    with torch.no_grad():
        model(inputs)

    for h in hooks:
        h.remove()
    return captured


def _accumulate_mse(base_captured, quant_captured, accum):
    """Accumulate per-key MSE into accum dict (sum of squared diffs)."""
    for key, bv in base_captured.items():
        if key not in quant_captured:
            continue
        mse = (bv - quant_captured[key]).pow(2).mean().item()
        accum[key] = accum.get(key, 0.0) + mse


def _finalize_mse_results(accum, num_batches):
    """Convert accumulated MSE sums into sorted result list."""
    results = []
    for key in sorted(accum.keys()):
        layer_str, _, lin_name = key.partition(".")
        layer_idx = int(layer_str.split("_")[1])
        results.append({
            "layer": layer_idx,
            "part": lin_name,
            "sensitivity": accum[key] / num_batches,
        })
    results.sort(key=lambda x: x["sensitivity"], reverse=True)
    return results


def compute_activation_sensitivity(
    base_model, quant_model, tokenizer, dataset_path,
    num_samples=128, seqlen=512, batch_size=1,
):
    """Run forward through both models in batches, average per-Linear MSE."""
    import torch.nn as nn

    samples = _load_calibration_samples(
        dataset_path, tokenizer, num_samples=num_samples, seqlen=seqlen,
    )
    embed_device = next(base_model.parameters()).device

    total = samples.shape[0]
    num_batches = (total + batch_size - 1) // batch_size
    accum: dict[str, float] = {}

    for i in range(num_batches):
        batch = samples[i * batch_size:(i + 1) * batch_size].to(embed_device)

        base_cap = _capture_one_batch(base_model, batch)
        quant_cap = _capture_one_batch(quant_model, batch)
        _accumulate_mse(base_cap, quant_cap, accum)

    return _finalize_mse_results(accum, num_batches)


def _run_activation_sensi_analysis(base_path, quant_path, dataset_path,
                                    num_samples, seqlen, batch_size,
                                    device_map="auto"):
    """Load models one at a time, run forward, and compare per-Linear outputs.

    Models are loaded sequentially to keep peak GPU memory at ~1× model size.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)

    def _load(path, name):
        logger.info("Loading %s model for activation_sensi: %s (device_map=%s)", name, path, device_map)
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype="auto", device_map=device_map, trust_remote_code=True,
        )
        m.eval()
        return m

    logger.info("--- Stage 1/2: base model inference ---")
    bm = _load(base_path, "base")
    samples = _load_calibration_samples(dataset_path, tokenizer, num_samples=num_samples, seqlen=seqlen)
    total = samples.shape[0]
    num_batches = (total + batch_size - 1) // batch_size
    logger.info("  %d samples split into %d batch(es) (batch_size=%d)", total, num_batches, batch_size)
    base_all = []
    for i in range(num_batches):
        batch = samples[i*batch_size:(i+1)*batch_size].to(next(bm.parameters()).device)
        base_all.append(_capture_one_batch(bm, batch))
    del bm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("--- Stage 2/2: quant model inference ---")
    qm = _load(quant_path, "quant")
    accum: dict[str, float] = {}
    for i in range(num_batches):
        batch = samples[i*batch_size:(i+1)*batch_size].to(next(qm.parameters()).device)
        _accumulate_mse(base_all[i], _capture_one_batch(qm, batch), accum)
    del qm
    del base_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _finalize_mse_results(accum, num_batches)


def run_analysis(args, base_model_path: str | None = None, quantized_model_path: str | None = None) -> None:

    if args.analysis == "none":
        return

    run_fisher = args.analysis == "fisher"
    run_diagnosis = args.analysis in ("outlier_mse", "outlier")
    run_mse = args.analysis in ("outlier_mse", "mse")
    run_act_sensi = args.analysis == "activation_sensi"

    _base = base_model_path or args.model
    _quant = quantized_model_path or getattr(args, "save_dir", None)

    if not _base:
        logger.error("--model is required for analysis")
        sys.exit(1)
    if not Path(_base).exists():
        logger.error("model not found: %s", _base)
        sys.exit(1)
    if run_mse or run_act_sensi:
        if not _quant:
            logger.error("--save-dir is required for --analysis mse/outlier_mse/activation_sensi")
            sys.exit(1)
        if not Path(_quant).exists():
            logger.error("quantized model not found: %s", _quant)
            sys.exit(1)

    top = getattr(args, "analysis_top", 20)
    outlier_thr = _DEFAULT_OUTLIER_THRESHOLD
    corr_thr = _DEFAULT_CORR_THRESHOLD
    ignore_min = _DEFAULT_IGNORE_MIN
    ignore_k = _DEFAULT_IGNORE_TOPK

    diagnosis = None
    mse_results = None

    if not run_act_sensi:
        logger.info("Loading base model: %s", _base)
        base_model = AutoModelForCausalLM.from_pretrained(
            _base, torch_dtype="auto", device_map="cpu", trust_remote_code=True,
        )
    else:
        base_model = None

    quantized_model = None
    if run_mse and not run_act_sensi:
        logger.info("Loading quantized model: %s", _quant)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            _quant, torch_dtype="auto", device_map="cpu", trust_remote_code=True,
        )

    if run_diagnosis:
        logger.info("Running channel outlier diagnosis on base model...")
        diagnosis = diagnose_model(
            base_model,
            threshold_ratio=outlier_thr,
            verbose=False,
        )

    if run_mse and not run_act_sensi:
        logger.info("Computing per-channel MSE between base and quantized weights...")
        mse_results = compute_per_channel_mse(base_model, quantized_model)

    fisher_stats = None
    if run_fisher:
        logger.info("Computing Fisher diagonal sensitivity on base model...")
        fisher_stats = _run_fisher_analysis(_base)

    act_sensi_results = None
    if run_act_sensi:
        logger.info("Computing activation sensitivity on base+quantized models...")
        act_sensi_results = _run_activation_sensi_analysis(
            _base, _quant,
            dataset_path=getattr(args, "dataset", "./datasets/ultrachat_200k"),
            num_samples=args.num_calibration_samples,
            seqlen=args.max_seq_length,
            batch_size=getattr(args, "batch_size", 1),
            device_map=getattr(args, "device_map", "auto"),
        )

    lines = []
    lines.append("=" * 80)
    lines.append("QUANTIZATION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Base model:      {_base}")
    if _quant:
        lines.append(f"Quantized model: {_quant}")
    lines.append(f"Analysis:        {args.analysis}")
    if run_diagnosis:
        lines.append(f"Outlier threshold: {outlier_thr}x median")
    lines.append("")

    outlier_map = {}
    if diagnosis is not None:
        for r in diagnosis.layer_reports:
            outlier_map[r.layer_name] = r.outlier_ratio

    if diagnosis is not None:
        lines.append("-- Channel Outlier Analysis (base model) --")
        lines.append(f"  Total Linear layers: {diagnosis.linear_layers}")
        num_sensitive = sum(1 for r in diagnosis.layer_reports if r.is_sensitive)
        lines.append(f"  Outlier-sensitive layers: {num_sensitive} "
                     f"({num_sensitive / max(1, diagnosis.linear_layers):.1%})")
        lines.append(f"  Recommend AWQ: {diagnosis.recommend_awq}")
        lines.append(f"  Recommend SmoothQuant: {diagnosis.recommend_smoothquant}")
        if diagnosis.recommend_ignore:
            lines.append(f"  Recommend ignore: {diagnosis.recommend_ignore}")
        lines.append("")
        lines.append(f"  Top {top} Layers by Outlier Ratio:")
        lines.append(f"  {'Layer':<55s} {'Outlier%':>8s}  {'Max/Med':>8s}  {'MaxAbs':>10s}  {'OutCh':>6s}")
        lines.append(f"  {'-'*55}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}")
        sorted_by_outlier = sorted(diagnosis.layer_reports,
                                   key=lambda r: r.outlier_ratio, reverse=True)
        for r in sorted_by_outlier[:top]:
            lines.append(
                f"  {r.layer_name:<55s}  {r.outlier_ratio:8.2%}  {r.max_to_median_ratio:8.1f}  "
                f"{r.max_abs:10.4f}  {len(r.outlier_channels):>5d}/{r.out_channels}"
            )
        non_expert = [r for r in sorted_by_outlier if ".experts." not in r.layer_name]
        if non_expert and any(".experts." in r.layer_name for r in sorted_by_outlier[:10]):
            lines.append(f"  Non-expert outlier top-{min(top, len(non_expert))}:")
            for r in non_expert[:top]:
                lines.append(
                    f"  {r.layer_name:<55s}  {r.outlier_ratio:8.2%}  {r.max_to_median_ratio:8.1f}  "
                    f"{r.max_abs:10.4f}  {len(r.outlier_channels):>5d}/{r.out_channels}"
                )
        lines.append("")

    if mse_results is not None:
        total_mse_layers = len(mse_results)
        all_zero = total_mse_layers > 0 and all(m[1] == 0.0 for m in mse_results)
        lines.append(f"-- Top {top} Layers by Per-Channel MSE ({total_mse_layers} analyzed) --")
        if all_zero:
            lines.append("  WARNING: All MSE values are zero -- weight extraction likely failed.")
            lines.append("  Check if the quantized model uses a format that is not yet supported.")
            lines.append("")
        else:
            lines.append(f"  {'Layer':<55s} {'Mean MSE':>14s}  {'Corr':>7s}  {'Outlier%':>8s}")
            lines.append(f"  {'-'*55}  {'-'*14}  {'-'*7}  {'-'*8}")

            for name, mean_mse, corr, _ in mse_results[:top]:
                outlier_pct = outlier_map.get(name, 0.0)
                lines.append(
                    f"  {name:<55s}  {mean_mse:14.6e}  {corr:7.3f}  {outlier_pct:8.2%}"
                )

            lines.append("")

        high_corr_all = [(n, m, c) for n, m, c, _ in mse_results if c > corr_thr]
        top_mse_names = {n for n, _, _, _ in mse_results[:top]}
        high_corr_topmse = [(n, m, c) for n, m, c in high_corr_all if n in top_mse_names]

        lines.append(f"-- Outlier<->MSE Correlation (threshold > {corr_thr:.2f}) --")
        lines.append(f"  High-corr layers (all analyzed): {len(high_corr_all)}")
        lines.append(f"  High-corr layers (within top-{top} MSE): {len(high_corr_topmse)}")

        if high_corr_topmse:
            lines.append(f"  Top-{min(top, len(high_corr_topmse))} high-corr layers within top-MSE:")
            for name, mse_val, corr in sorted(high_corr_topmse, key=lambda x: x[1], reverse=True)[:top]:
                outlier_pct = outlier_map.get(name, 0.0)
                lines.append(
                    f"  {name:<55s}  corr={corr:.3f}  mse={mse_val:.6e}  outlier={outlier_pct:.2%}"
                )
        elif high_corr_all:
            lines.append(f"  Top-{min(top, len(high_corr_all))} high-corr layers overall:")
            for name, mse_val, corr in sorted(high_corr_all, key=lambda x: x[1], reverse=True)[:top]:
                outlier_pct = outlier_map.get(name, 0.0)
                lines.append(
                    f"  {name:<55s}  corr={corr:.3f}  mse={mse_val:.6e}  outlier={outlier_pct:.2%}"
                )
        else:
            lines.append("  (none)")
        lines.append("")

    if fisher_stats is not None:
        lines.append(f"-- Fisher Diagonal Sensitivity Ranking (top {top}) --")
        lines.append(f"  {'Rank':<5} {'Layer':<55} {'Mean':>14} {'Max':>14} {'Params':>10}")
        lines.append(f"  {'-'*5}  {'-'*55}  {'-'*14}  {'-'*14}  {'-'*10}")
        for i, (name, mean_val, max_val, n_params) in enumerate(fisher_stats[:top]):
            lines.append(
                f"  {i+1:<5}  {name:<55}  {mean_val:14.6e}  {max_val:14.6e}  {n_params:>10,}"
            )
        lines.append("")

    if act_sensi_results is not None:
        lines.append(f"-- Activation Sensitivity Ranking (top {top}) --")
        lines.append(f"  {'Rank':<5} {'Layer':<8} {'Linear':<20} {'MSE':>14}")
        lines.append(f"  {'-'*5}  {'-'*6}  {'-'*24}  {'-'*14}")
        for i, r in enumerate(act_sensi_results[:top]):
            lines.append(
                f"  {i+1:<5}  {r['layer']:<6}  {r['part']:<24}  {r['sensitivity']:>14.6e}"
            )
        lines.append("")

    lines.append("-- Recommendations --")
    if diagnosis is not None and mse_results is not None:
        high_corr = [(n, m, c) for n, m, c, _ in mse_results if c > corr_thr]
        if diagnosis.recommend_awq and high_corr:
            lines.append("  [OK] High outlier-MSE correlation confirmed -> AWQ is likely to help.")
        if diagnosis.recommend_smoothquant:
            lines.append("  [OK] Widespread sensitivity -> SmoothQuant is recommended.")
        if not diagnosis.recommend_awq and not diagnosis.recommend_smoothquant:
            lines.append("  No strong outlier/MSE correlation pattern detected.")
        scored = []
        for name, mean_mse, corr, _ in mse_results:
            outlier_ratio = outlier_map.get(name, 0.0)
            if outlier_ratio < ignore_min:
                continue
            score = mean_mse * (1.0 + 5.0 * outlier_ratio)
            scored.append((score, name, mean_mse, outlier_ratio, corr))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            lines.append(f"  Ignore candidates (score=mean_mse*(1+5*outlier), outlier>={ignore_min:.2%}):")
            for _, name, mean_mse, outlier_ratio, corr in scored[:ignore_k]:
                lines.append(
                    f"    {name}  mse={mean_mse:.6e}  outlier={outlier_ratio:.2%}  corr={corr:.3f}"
                )
        else:
            ignore_candidates = [n for n, _, _, _ in mse_results[:ignore_k]]
            lines.append(f"  Ignore candidates (top-{ignore_k} MSE layers): {ignore_candidates}")
    elif diagnosis is not None:
        if diagnosis.recommend_awq:
            lines.append("  [OK] Recommend AWQ preprocessing based on outlier analysis.")
        if diagnosis.recommend_smoothquant:
            lines.append("  [OK] Recommend SmoothQuant preprocessing based on outlier analysis.")
    elif mse_results is not None:
        ignore_candidates = [n for n, _, _, _ in mse_results[:ignore_k]]
        lines.append(f"  Ignore candidates (top-{ignore_k} MSE layers): {ignore_candidates}")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)

    output_file = getattr(args, "analysis_output", None)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Report saved to: %s", output_file)
    else:
        print(report)

    if base_model is not None:
        del base_model
    if quantized_model is not None:
        del quantized_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
