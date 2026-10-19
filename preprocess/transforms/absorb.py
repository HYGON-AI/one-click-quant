"""Shared smooth-scale absorb helpers (UpDown / NormLinear)."""

from __future__ import annotations

import fnmatch
import re
from typing import Iterable, List, Optional, Sequence

import torch
from loguru import logger

from preprocess.io import WeightStore

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _pattern_matches(name: str, pat: str) -> bool:
    """Match one include pattern against a module or tensor name.

    ``re:`` prefixes are Python regular expressions (``re.search``), matching
    ``match_include``. Other patterns are fnmatch globs: ``.`` is literal and
    ``*`` is a wildcard. A ``re:`` body is never passed through glob rewriting,
    so ``\\d``, ``+``, groups, and already-escaped ``\\.`` keep their meaning.
    """
    if pat.startswith("re:"):
        return re.search(pat[3:], name) is not None
    if fnmatch.fnmatch(name, pat):
        return True
    return re.fullmatch(pat.replace(".", r"\.").replace("*", r".*"), name) is not None


def match_include(name: str, patterns: Sequence[str]) -> bool:
    """Match HF-style include with ``*`` wildcards (also accepts ``re:`` prefixes)."""
    return any(_pattern_matches(name, pat) for pat in patterns)


def expand_module_globs(weights: WeightStore, patterns: Sequence[str], suffix: str = ".weight") -> List[str]:
    """Return tensor names whose module path matches any pattern.

    Matches both ``{mod}{suffix}`` (default ``.weight``) and bare fused keys
    (e.g. ``...experts.gate_up_proj``) when the pattern names that module.
    Patterns use the same rules as ``match_include``: globs, or ``re:`` regex.
    """
    out = []
    seen = set()
    for name in weights.names():
        mods = []
        if suffix and name.endswith(suffix):
            mods.append(name[: -len(suffix)])
        mods.append(name)
        for mod in mods:
            if any(_pattern_matches(mod, pat) for pat in patterns):
                if name not in seen:
                    out.append(name)
                    seen.add(name)
                break
    return out


def module_name_from_weight(wname: str) -> str:
    """Strip a trailing ``.weight``; leave bare fused keys unchanged.

    ``experts.gate_up_proj`` has no ``.weight`` suffix. Slicing a fixed
    ``len(".weight")`` off that name drops ``proj`` and breaks up-down lookup.
    """
    suffix = ".weight"
    if wname.endswith(suffix):
        return wname[: -len(suffix)]
    return wname


def require_float_weight(weights: WeightStore, key: str, context: str) -> torch.Tensor:
    """Load ``key`` and raise if it is not a floating Linear/Norm weight."""
    w = weights.get(key)
    if w.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            f"{context}: expected float16/bfloat16/float32 for {key}, got {w.dtype}. "
            "Dequantize MXFP4/INT packs to float before preprocess."
        )
    return w


@torch.no_grad()
def compute_weight_channel_max(weight: torch.Tensor, *, along: str = "in") -> torch.Tensor:
    """Per-channel absmax.

    ``along='in'`` → shape ``[K]`` (max over out rows).
    ``along='out'`` → shape ``[N]`` (max over in cols).
    """
    w = weight.float().abs()
    if along == "in":
        return w.amax(dim=0).clamp(min=1e-5)
    if along == "out":
        return w.amax(dim=1).clamp(min=1e-5)
    raise ValueError(f"along must be 'in' or 'out', got {along}")


def as_act_matrix(act: torch.Tensor) -> torch.Tensor:
    """Normalize act stats to ``[N, K]`` for Flex search matmul / ``amax(dim=0)``.

    ``collect_acts_*.py --reduce channel_max`` stores per-channel absmax as
    ``[K]``; raw buffers are ``[N, K]`` (or higher rank). Treating ``[K]`` as a
    sample axis would truncate channels when ``K > 2048`` and break scale length.
    """
    a = act.float()
    if a.dim() == 0:
        raise ValueError("act stats must be at least 1-D")
    if a.dim() == 1:
        return a.unsqueeze(0)
    if a.dim() > 2:
        return a.reshape(-1, a.shape[-1])
    return a


@torch.no_grad()
def make_scales_from_alpha_beta(
    act_max: torch.Tensor,
    weight_max: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """``s = act^alpha / weight^beta`` (Flex AWQ-SSZ / Flex Smooth form)."""
    a = act_max.float().reshape(-1).clamp(min=1e-5)
    w = weight_max.float().reshape(-1).clamp(min=1e-5)
    if a.numel() != w.numel():
        raise ValueError(f"act_max ({a.numel()}) and weight_max ({w.numel()}) length mismatch")
    return (a**alpha) / (w**beta)


def _scale_linear_in(weights: WeightStore, module: str, scales: torch.Tensor, factor: float = 1.0) -> None:
    """``W *= (factor * s)`` along in_features (last dim); bias unchanged."""
    key = weights.weight_key(module)
    if key is None:
        raise KeyError(f"missing linear weight for {module}")
    w = require_float_weight(weights, key, "absorb in-scale")
    s = scales.float().reshape(-1) * factor
    weights.set(key, scale_last_dim(w, s))


def _scale_linear_out(weights: WeightStore, module: str, scales: torch.Tensor, factor: float = 1.0) -> None:
    """``W *= (factor * s)`` along out_features; also scale bias if present."""
    key = weights.weight_key(module)
    if key is None:
        raise KeyError(f"missing linear weight for {module}")
    w = require_float_weight(weights, key, "absorb out-scale")
    s = (scales.float().to(w.device) * factor).clamp(min=1e-5)
    if s.numel() != w.shape[0]:
        raise ValueError(f"{module}: scale dim {s.numel()} != out_features {w.shape[0]}")
    weights.set(key, (w.float() * s.unsqueeze(1)).to(w.dtype))
    bias_key = f"{module}.bias"
    if weights.has(bias_key):
        b = weights.get(bias_key)
        if b.dtype in _FLOAT_DTYPES and b.numel() == s.numel():
            weights.set(bias_key, (b.float() * s).to(b.dtype))


@torch.no_grad()
def absorb_up_down(
    weights: WeightStore,
    up_module: str,
    down_module: str,
    scales: torch.Tensor,
) -> None:
    """Absorb intermediate-channel scales for an up→down pair (UpDown).

    ``scales`` length must equal ``up.out_features == down.in_features``.

    - ``W_up *= 1/s`` along **out**
    - ``W_down *= s`` along **in**
    - ``up.bias *= 1/s`` when present
    """
    up_key = weights.weight_key(up_module)
    down_key = weights.weight_key(down_module)
    if up_key is None or down_key is None:
        raise KeyError(f"up-down absorb missing weights: {up_module} / {down_module}")
    wu = require_float_weight(weights, up_key, "absorb_up_down")
    wd = require_float_weight(weights, down_key, "absorb_up_down")
    inter = wu.shape[0]
    if wd.shape[1] != inter:
        raise ValueError(
            f"up-down shape mismatch: {up_module} out={inter}, {down_module} in={wd.shape[1]}"
        )
    s = scales.float().reshape(-1).to(wu.device).clamp(min=1e-5)
    if s.numel() != inter:
        raise ValueError(f"up-down scales length {s.numel()} != intermediate {inter}")

    weights.set(up_key, (wu.float() / s.unsqueeze(1)).to(wu.dtype))
    bias_key = f"{up_module}.bias"
    if weights.has(bias_key):
        b = weights.get(bias_key)
        if b.dtype in _FLOAT_DTYPES and b.numel() == s.numel():
            weights.set(bias_key, (b.float() / s).to(b.dtype))
    weights.set(down_key, (wd.float() * s.unsqueeze(0)).to(wd.dtype))


@torch.no_grad()
def absorb_fused_gate_up_down(
    weights: WeightStore,
    up_module: str,
    down_module: str,
    scales: torch.Tensor,
) -> None:
    """Per-expert AWQ absorb for fused ``gate_up_proj`` / ``down_proj``.

    ``gate_up`` is ``[E, 2I, H]`` with gate = half 0 and up = half 1
    (sglang ``stacked_params_mapping``). ``down`` is ``[E, H, I]``.
    ``scales`` is ``[E, I]``. Only the up half is divided; the gate half
    stays put. ``down`` is multiplied on its last dim.
    """
    up_key = weights.weight_key(up_module)
    down_key = weights.weight_key(down_module)
    if up_key is None or down_key is None:
        raise KeyError(f"fused up-down absorb missing weights: {up_module} / {down_module}")
    wu = require_float_weight(weights, up_key, "absorb_fused_gate_up_down")
    wd = require_float_weight(weights, down_key, "absorb_fused_gate_up_down")
    if wu.dim() != 3 or wd.dim() != 3:
        raise ValueError(
            f"fused up-down expects 3D weights, got {up_module} {tuple(wu.shape)} "
            f"{down_module} {tuple(wd.shape)}"
        )
    n_exp, two_i, hidden = wu.shape
    if two_i % 2 != 0:
        raise ValueError(f"{up_module}: fused out dim {two_i} is not 2*intermediate")
    inter = two_i // 2
    if wd.shape[0] != n_exp or wd.shape[1] != hidden or wd.shape[2] != inter:
        raise ValueError(
            f"fused shape mismatch: {up_module} {tuple(wu.shape)} vs {down_module} {tuple(wd.shape)} "
            f"(expected down [E, H, I] = [{n_exp}, {hidden}, {inter}])"
        )
    s = scales.float().to(wu.device).clamp(min=1e-5)
    if tuple(s.shape) != (n_exp, inter):
        raise ValueError(f"fused scales {tuple(s.shape)} != [{n_exp}, {inter}]")
    wu2 = wu.float().clone()
    wu2[:, inter:, :] = wu2[:, inter:, :] / s.unsqueeze(-1)
    wd2 = wd.float() * s.unsqueeze(1)
    weights.set(up_key, wu2.to(dtype=wu.dtype))
    weights.set(down_key, wd2.to(dtype=wd.dtype))


def scale_last_dim(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``W *= s`` on the last dimension (2D in-features or 3D ``[E, *, K]``)."""
    s = scales.float().reshape(-1).to(weight.device).clamp(min=1e-5)
    if weight.shape[-1] != s.numel():
        raise ValueError(f"scale dim {s.numel()} != in_features {weight.shape[-1]}")
    view = [1] * (weight.dim() - 1) + [s.numel()]
    return (weight.float() * s.view(*view)).to(dtype=weight.dtype)


# Alias kept for call-site clarity.
absorb_up_down_clean = absorb_up_down


@torch.no_grad()
def absorb_norm_linear(
    weights: WeightStore,
    norm_module: str,
    linear_modules: Sequence[str],
    scales: torch.Tensor,
    *,
    norm_style: str = "rms",
) -> None:
    """Absorb SmoothQuant scales for a shared Norm → many Linears (NormLinear).

    ``scales`` length must equal Norm dim / each Linear ``in_features``.

    - every Linear: ``W *= s`` along **in**
    - RMSNorm gamma: ``gamma /= s`` (identity is ones)
    - GemmaRMSNorm gamma: ``gamma = (1+gamma)/s - 1`` (identity is zeros; forward uses ``1+gamma``)
    - Norm bias: ``bias /= s`` when present
    """
    if not linear_modules:
        raise ValueError("absorb_norm_linear requires at least one linear consumer")
    if norm_style not in ("rms", "gemma"):
        raise ValueError(f"norm_style must be rms or gemma, got {norm_style}")
    nk = weights.weight_key(norm_module)
    if nk is None:
        raise KeyError(f"missing norm weight for {norm_module}")
    nw = require_float_weight(weights, nk, "absorb_norm_linear")
    s = scales.float().reshape(-1).to(nw.device).clamp(min=1e-5)
    if nw.dim() != 1 or nw.numel() != s.numel():
        raise ValueError(f"{norm_module}: norm dim {tuple(nw.shape)} != scales {s.numel()}")

    for lin in linear_modules:
        key = weights.weight_key(lin)
        if key is None:
            raise KeyError(f"missing linear weight for {lin}")
        w = require_float_weight(weights, key, "absorb_norm_linear")
        weights.set(key, scale_last_dim(w, s))

    if norm_style == "gemma":
        new_gamma = (1.0 + nw.float()) / s - 1.0
    else:
        new_gamma = nw.float() / s
    weights.set(nk, new_gamma.to(nw.dtype))
    nb = f"{norm_module}.bias"
    if weights.has(nb):
        b = weights.get(nb)
        if b.dtype in _FLOAT_DTYPES and b.numel() == s.numel():
            weights.set(nb, (b.float() / s).to(b.dtype))


@torch.no_grad()
def absorb_gemma_tied_hc(
    weights: WeightStore,
    norm_module: str,
    wide_modules: Sequence[str],
    hidden_modules: Sequence[str],
    scales_h: torch.Tensor,
    hc: int,
) -> None:
    """Gemma SmoothQuant with one scale per hidden channel, repeated across hc branches.

    Layout of the  ``hc * hidden`` stream is branch-major (``view(..., hc, hidden)``).
    Wide linears (in = hc*hidden) get ``s.repeat(hc)``. Hidden linears (in = hidden),
    including 3D ``[E, *, H]``, get ``s``. Gamma update is Gemma: ``(1+γ)/s_wide - 1``.

    Repeating ``s`` across branches keeps the mix output ``mean_c(gate * normed)``
    equal to a single hidden-channel scale, which the block linears absorb.
    """
    if hc < 1:
        raise ValueError(f"hc must be positive, got {hc}")
    if not wide_modules and not hidden_modules:
        raise ValueError(f"{norm_module}: tied absorb has no consumers")
    nk = weights.weight_key(norm_module)
    if nk is None:
        raise KeyError(f"missing norm weight for {norm_module}")
    nw = require_float_weight(weights, nk, "absorb_gemma_tied_hc")
    s = scales_h.float().reshape(-1).clamp(min=1e-5)
    s_wide = s.repeat(hc)
    if nw.dim() != 1 or nw.numel() != s_wide.numel():
        raise ValueError(
            f"{norm_module}: norm dim {tuple(nw.shape)} != hc*hidden {s_wide.numel()}"
        )
    for lin in wide_modules:
        key = weights.weight_key(lin)
        if key is None:
            raise KeyError(f"missing wide linear {lin}")
        w = require_float_weight(weights, key, "absorb_gemma_tied_hc")
        weights.set(key, scale_last_dim(w, s_wide))
    for lin in hidden_modules:
        key = weights.weight_key(lin)
        if key is None:
            raise KeyError(f"missing hidden linear {lin}")
        w = require_float_weight(weights, key, "absorb_gemma_tied_hc")
        weights.set(key, scale_last_dim(w, s))
    new_gamma = (1.0 + nw.float()) / s_wide.to(nw.device) - 1.0
    weights.set(nk, new_gamma.to(nw.dtype))


# Back-compat name used by older call sites — prefer absorb_norm_linear / absorb_up_down_clean.
@torch.no_grad()
def absorb_scales_linear(
    weights: WeightStore,
    linear_module: str,
    scales: torch.Tensor,
    *,
    pred_modules: Optional[Iterable[str]] = None,
    pred_is_norm: bool = False,
) -> None:
    """Deprecated wrapper: NormLinear-style absorb for a single linear + optional Norm pred."""
    preds = list(pred_modules or [])
    if pred_is_norm and preds:
        absorb_norm_linear(weights, preds[0], [linear_module], scales)
        if len(preds) > 1:
            logger.warning(
                "absorb_scales_linear: extra norm preds {} ignored; use absorb_norm_linear",
                preds[1:],
            )
        return
    if not preds:
        raise ValueError(
            f"Refusing orphan absorb on {linear_module}: scaling W alone changes the network. "
            "Use absorb_up_down_clean or absorb_norm_linear with a full consumer list."
        )
    # pred is Linear (linear-linear): pred out /= s, this in *= s
    absorb_up_down(weights, preds[0], linear_module, scales)
