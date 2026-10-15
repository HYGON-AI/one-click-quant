"""Flex SmoothQuant-style alpha/beta search + NormLinear absorb."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger

from preprocess.io import WeightStore
from preprocess.transforms.absorb import (
    absorb_gemma_tied_hc,
    absorb_norm_linear,
    compute_weight_channel_max,
    expand_module_globs,
    make_scales_from_alpha_beta,
    module_name_from_weight,
    require_float_weight,
)
from preprocess.transforms.base import PreprocessContext, PreprocessTransform


@torch.no_grad()
def _quant_int8_sym(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    xmax = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    q = torch.round(x / (xmax / 127)).clamp(-127, 127)
    return q * (xmax / 127)


@torch.no_grad()
def search_alpha_beta(
    act: torch.Tensor,
    weight: torch.Tensor,
    search_step: float = 0.1,
) -> Tuple[float, float, float]:
    """Grid-search alpha with beta=1-alpha, then refine beta. Returns (alpha, beta, mse)."""
    w = weight.float()
    a = act.float()
    if a.dim() > 2:
        a = a.reshape(-1, a.shape[-1])
    if a.shape[0] > 2048:
        idx = torch.randperm(a.shape[0])[:2048]
        a = a[idx]

    fp = a @ w.T
    normal = (fp.pow(2).mean().sqrt()).clamp(min=1e-8)

    def eval_ab(alpha: float, beta: float) -> float:
        scale = make_scales_from_alpha_beta(
            a.abs().amax(dim=0),
            compute_weight_channel_max(w, along="in"),
            alpha,
            beta,
        )
        sa = a / scale
        sw = w * scale.unsqueeze(0)
        qout = _quant_int8_sym(sa) @ _quant_int8_sym(sw).T
        return ((qout - fp).pow(2).mean().sqrt() / normal).item()

    best_alpha, best_mse = 0.0, float("inf")
    for alpha in np.round(np.arange(0.0, 1.0 + search_step, search_step), 2):
        mse = eval_ab(float(alpha), float(1.0 - alpha))
        if mse <= best_mse:
            best_mse, best_alpha = mse, float(alpha)

    best_beta, best_mse2 = 0.0, best_mse
    for beta in np.round(np.arange(0.0, 1.0 + search_step, search_step), 2):
        mse = eval_ab(best_alpha, float(beta))
        if mse <= best_mse2:
            best_mse2, best_beta = mse, float(beta)
    return best_alpha, best_beta, best_mse2


def _default_act_proxy(weight: torch.Tensor) -> torch.Tensor:
    """Weight-only proxy activations: |W| column max as fake act max ``[1, K]``."""
    k = weight.shape[-1]
    cols = weight.float().abs().reshape(-1, k).amax(dim=0).clamp(min=1e-5)
    return cols.view(1, k)


def _select_search_consumers(present: List[str], max_n: int) -> List[str]:
    """Pick a small subset for alpha/beta search (prefer non-routed experts)."""
    if max_n <= 0 or len(present) <= max_n:
        return list(present)
    preferred = [p for p in present if ".experts." not in p]
    experts = [p for p in present if ".experts." in p]
    out = list(preferred)
    remain = max_n - len(out)
    if remain > 0 and experts:
        step = max(1, len(experts) // remain)
        out.extend(experts[::step][:remain])
    if not out:
        out = present[:max_n]
    return out[:max_n]


@torch.no_grad()
def _incremental_in_channel_max(
    weights: WeightStore, modules: List[str]
) -> torch.Tensor:
    """Per-in-channel absmax over many Linears without concatenating them."""
    acc: torch.Tensor | None = None
    for mod in modules:
        key = weights.weight_key(mod)
        w = require_float_weight(weights, key, "flex_smooth")
        m = compute_weight_channel_max(w, along="in").cpu()
        acc = m if acc is None else torch.maximum(acc, m)
        del w
    if acc is None:
        raise ValueError("flex_smooth: empty consumer list for channel max")
    return acc


def _hidden_rows(weight: torch.Tensor, hidden: int, hc: int, max_rows: int = 4096) -> torch.Tensor:
    """Rows whose last dim is ``hidden``, subsampling so expert stacks stay small.

    A ``hc * hidden`` weight is branch-major and is folded to ``[*, hidden]``.
    """
    last = weight.shape[-1]
    flat = weight.detach().float().reshape(-1, last)
    if last == hidden * hc:
        flat = flat.reshape(-1, hc, hidden).reshape(-1, hidden)
    elif last != hidden:
        raise ValueError(
            f"smooth row last dim {last} is neither hidden {hidden} nor hc*hidden {hidden * hc}"
        )
    n = flat.shape[0]
    if n > max_rows:
        step = n / float(max_rows)
        idx = (torch.arange(max_rows) * step).long().clamp(max=n - 1)
        flat = flat[idx]
    return flat


class FlexSmoothTransform(PreprocessTransform):
    name = "flex_smooth_quant"

    def apply(self, weights: WeightStore, ctx: PreprocessContext) -> None:
        adapter = ctx.require_adapter()
        cfg = ctx.recipe.get("flex_smooth_quant", {})
        includes: List[str] = list(cfg.get("include") or [])
        search_step = float(cfg.get("search_step", 0.1))
        use_calib = bool(cfg.get("use_calib_acts", False))
        # DeepSeek MoE post_attn_ln has 500+ expert consumers; never cat them all.
        max_search_consumers = int(cfg.get("max_search_consumers", 8))
        if not includes:
            logger.warning("flex_smooth_quant: empty include; skip")
            return

        targets = expand_module_globs(weights, includes)
        targets = [w for w in targets if ctx.owns_module(w)]
        if not targets:
            logger.warning("flex_smooth_quant: include matched 0 modules on this rank; skip")
            return
        logger.info(
            "flex_smooth_quant: {} listed seed modules (rank={}/{})",
            len(targets),
            ctx.rank,
            ctx.world_size,
        )
        if not use_calib or not ctx.act_stats:
            logger.warning(
                "flex_smooth_quant: no calib acts — using |W| column max as act proxy "
                "(set use_calib_acts + --act-stats for real activation-aware search)"
            )

        # Group by shared Norm via adapter LN map (completeness source of truth).
        groups: Dict[str, List[str]] = {}
        skipped = 0
        for wname in targets:
            mod = module_name_from_weight(wname)
            mapping = adapter.resolve_norm_linear(mod, ctx.text_config)
            if mapping is None:
                logger.warning(
                    "flex_smooth_quant: {} not in adapter NormLinear map; "
                    "skip orphan absorb (UpDown seeds belong in flex_awq_ssz)",
                    mod,
                )
                skipped += 1
                continue
            norm, siblings = mapping
            if not ctx.owns_module(norm):
                continue
            groups.setdefault(norm, siblings)

        done = 0
        norm_style = adapter.norm_affine()
        text_cfg = ctx.text_config
        hidden = int(text_cfg.get("hidden_size") or 0)
        hc = int(text_cfg.get("hc_count") or 1)
        for norm, siblings in groups.items():
            if weights.weight_key(norm) is None:
                logger.warning("flex_smooth_quant: norm {} missing; skip", norm)
                continue
            present = [s for s in siblings if weights.weight_key(s) is not None]
            partners = [
                p
                for p in adapter.hidden_smooth_partners(norm, text_cfg)
                if weights.weight_key(p) is not None and ctx.owns_module(p)
            ]
            if set(present) != set(siblings) and siblings:
                missing = sorted(set(siblings) - set(present))
                logger.warning(
                    "flex_smooth_quant: {} missing consumers {}; absorbing only {}",
                    norm,
                    missing,
                    present,
                )
            nw = require_float_weight(weights, weights.weight_key(norm), "flex_smooth")
            tied = (
                norm_style == "gemma"
                and hc > 1
                and hidden > 0
                and nw.numel() == hidden * hc
                and (present or partners)
            )
            if tied and not partners:
                logger.warning(
                    "flex_smooth_quant: {} is hc*hidden but has no hidden-size "
                    "partner to absorb the tied scale; skip",
                    norm,
                )
                skipped += 1
                continue
            if tied:
                self._smooth_gemma_tied(
                    weights,
                    ctx,
                    norm,
                    present,
                    partners,
                    hidden=hidden,
                    hc=hc,
                    search_step=search_step,
                    use_calib=use_calib,
                )
                done += 1
                continue
            if not present:
                logger.warning("flex_smooth_quant: no consumers present for {}; skip", norm)
                continue

            search_mods = _select_search_consumers(present, max_search_consumers)
            if len(search_mods) < len(present):
                logger.info(
                    "flex_smooth: {} has {} consumers; search on {} "
                    "(prefer gate/shared), absorb all",
                    norm,
                    len(present),
                    len(search_mods),
                )
            ws = [
                require_float_weight(weights, weights.weight_key(s), "flex_smooth")
                for s in search_mods
            ]
            in_dim = ws[0].shape[-1]
            if any(w.shape[-1] != in_dim or w.dim() != 2 for w in ws):
                raise ValueError(f"flex_smooth: in_features mismatch under {norm}")
            merged = torch.cat(ws, dim=0)
            del ws

            seed_mod = search_mods[0]
            if use_calib and seed_mod in ctx.act_stats:
                act = ctx.act_stats[seed_mod]
            elif use_calib and norm in ctx.act_stats:
                act = ctx.act_stats[norm]
            else:
                act = _default_act_proxy(merged)

            max_cuda_elems = 512 * 1024**2  # ~2 GiB bf16 / ~4 GiB fp32
            if (
                ctx.device.type == "cuda"
                and merged.numel() + act.numel() <= max_cuda_elems
            ):
                act_s = act.to(ctx.device)
                merged_s = merged.to(ctx.device)
            else:
                if ctx.device.type == "cuda":
                    logger.info(
                        "flex_smooth: {} search tensor still large "
                        "(numel={}); search on CPU",
                        norm,
                        merged.numel(),
                    )
                act_s, merged_s = act, merged
            alpha, beta, mse = search_alpha_beta(act_s, merged_s, search_step=search_step)
            del merged_s, act_s
            a_max = act.float().abs().amax(dim=0) if act.dim() > 1 else act.float().abs()
            # Channel max over ALL consumers without building a giant cat.
            w_max = (
                _incremental_in_channel_max(weights, present)
                if len(search_mods) < len(present)
                else compute_weight_channel_max(merged, along="in").cpu()
            )
            del merged
            scales = make_scales_from_alpha_beta(
                a_max.cpu(),
                w_max,
                alpha,
                beta,
            )
            absorb_norm_linear(weights, norm, present, scales, norm_style=norm_style)
            done += 1
            logger.info(
                "smooth {} consumers={} search_on={} alpha={:.2f} beta={:.2f} mse={:.4f}",
                norm,
                len(present),
                len(search_mods),
                alpha,
                beta,
                mse,
            )

        logger.info(
            "flex_smooth_quant absorbed {} norm groups (skipped_orphan={}, rank={})",
            done,
            skipped,
            ctx.rank,
        )

    def _smooth_gemma_tied(
        self,
        weights: WeightStore,
        ctx: PreprocessContext,
        norm: str,
        wide: List[str],
        hidden_mods: List[str],
        *,
        hidden: int,
        hc: int,
        search_step: float,
        use_calib: bool,
    ) -> None:
        """One hidden-channel scale, broadcast across hc branches."""
        rows = []
        for name in list(wide) + list(hidden_mods):
            key = weights.weight_key(name)
            w = require_float_weight(weights, key, "flex_smooth")
            if w.shape[-1] not in (hidden, hidden * hc):
                logger.warning(
                    "flex_smooth_quant: {} last dim {} skipped for tied scale of {}",
                    name,
                    w.shape[-1],
                    norm,
                )
                continue
            rows.append(_hidden_rows(w, hidden, hc))
        if not rows:
            logger.warning("flex_smooth_quant: tied group {} has no usable rows; skip", norm)
            return
        merged = torch.cat(rows, dim=0)
        if use_calib and norm in ctx.act_stats and ctx.act_stats[norm].shape[-1] == hidden:
            act = ctx.act_stats[norm]
        elif use_calib and hidden_mods and hidden_mods[0] in ctx.act_stats:
            act = ctx.act_stats[hidden_mods[0]]
            if act.shape[-1] != hidden:
                act = _default_act_proxy(merged)
        else:
            act = _default_act_proxy(merged)
        max_cuda_elems = 512 * 1024**2
        if (
            ctx.device.type == "cuda"
            and merged.numel() + act.numel() <= max_cuda_elems
        ):
            act_s = act.to(ctx.device)
            merged_s = merged.to(ctx.device)
        else:
            act_s, merged_s = act, merged
        alpha, beta, mse = search_alpha_beta(act_s, merged_s, search_step=search_step)
        a_max = act.float().abs().amax(dim=0) if act.dim() > 1 else act.float().abs()
        if a_max.numel() != hidden:
            a_max = merged.float().abs().amax(dim=0)
        scales = make_scales_from_alpha_beta(
            a_max.cpu(),
            compute_weight_channel_max(merged, along="in"),
            alpha,
            beta,
        )
        wide_ok = []
        hidden_ok = []
        for name in wide:
            key = weights.weight_key(name)
            w = weights.get(key)
            if w.shape[-1] == hidden * hc:
                wide_ok.append(name)
        for name in hidden_mods:
            key = weights.weight_key(name)
            w = weights.get(key)
            if w.shape[-1] == hidden:
                hidden_ok.append(name)
        absorb_gemma_tied_hc(weights, norm, wide_ok, hidden_ok, scales, hc)
        logger.info(
            "smooth gemma-tied {} wide={} hidden={} alpha={:.2f} beta={:.2f} mse={:.4f}",
            norm,
            len(wide_ok),
            len(hidden_ok),
            alpha,
            beta,
            mse,
        )
