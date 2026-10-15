"""Flex AWQ-SSZ scale search + absorb for up-down subgraphs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from loguru import logger

from preprocess.io import WeightStore
from preprocess.transforms.absorb import (
    absorb_fused_gate_up_down,
    absorb_up_down,
    compute_weight_channel_max,
    expand_module_globs,
    make_scales_from_alpha_beta,
    module_name_from_weight,
    require_float_weight,
)
from preprocess.transforms.base import PreprocessContext, PreprocessTransform
from preprocess.transforms.flex_smooth import _quant_int8_sym


@torch.no_grad()
def fake_quant_weight_ssz(weight: torch.Tensor, step: int = 10) -> torch.Tensor:
    """Lightweight SSZ-like per-channel symmetric INT4 fake-quant for search."""
    w = weight.float()
    absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    best = w
    best_mse = float("inf")
    for i in range(step):
        ratio = 0.5 + 0.5 * (i + 1) / step
        clip = absmax * ratio
        qmax = 7.0
        q = torch.round((w / clip).clamp(-1, 1) * qmax).clamp(-qmax, qmax)
        rec = q / qmax * clip
        mse = (rec - w).pow(2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best = rec
    return best


@torch.no_grad()
def search_alpha_awq_ssz(
    act: torch.Tensor,
    down_weight: torch.Tensor,
    search_step: float = 0.1,
    ssz_step: int = 10,
) -> Tuple[float, float]:
    """Search alpha (beta=0) on intermediate act × down weight (UpDown subgraph)."""
    w = down_weight.float()
    a = act.float()
    if a.dim() > 2:
        a = a.reshape(-1, a.shape[-1])
    if a.shape[-1] != w.shape[1]:
        raise ValueError(
            f"act last dim {a.shape[-1]} != down in_features {w.shape[1]} for AWQ search"
        )
    if a.shape[0] > 2048:
        a = a[torch.randperm(a.shape[0])[:2048]]

    fp = a @ w.T
    normal = fp.pow(2).mean().sqrt().clamp(min=1e-8)

    best_alpha, best_mse = 0.0, float("inf")
    w_max = compute_weight_channel_max(w, along="in")
    a_max = a.abs().amax(dim=0)
    for alpha in np.round(np.arange(0.0, 1.0 + search_step, search_step), 2):
        scale = make_scales_from_alpha_beta(a_max, w_max, float(alpha), 0.0)
        sa = a / scale
        sw = fake_quant_weight_ssz(w * scale.unsqueeze(0), step=ssz_step)
        qout = _quant_int8_sym(sa) @ sw.T
        mse = ((qout - fp).pow(2).mean().sqrt() / normal).item()
        if mse <= best_mse:
            best_mse, best_alpha = mse, float(alpha)
    return best_alpha, best_mse


def _intermediate_act_proxy(down_weight: torch.Tensor) -> torch.Tensor:
    """Weight-only proxy for intermediate activations: |W_down| column max as [1, K]."""
    k = down_weight.shape[1]
    return torch.ones(1, k, dtype=torch.float32) * compute_weight_channel_max(down_weight, along="in")


class FlexAwqSszTransform(PreprocessTransform):
    name = "flex_awq_ssz"

    def apply(self, weights: WeightStore, ctx: PreprocessContext) -> None:
        adapter = ctx.require_adapter()
        cfg = ctx.recipe.get("flex_awq_ssz", {})
        includes: List[str] = list(cfg.get("include") or [])
        search_step = float(cfg.get("search_step", 0.1))
        ssz_step = int(cfg.get("ssz_step", 10))
        use_calib = bool(cfg.get("use_calib_acts", False))
        if not includes:
            logger.warning("flex_awq_ssz: empty include; skip")
            return

        targets = expand_module_globs(weights, includes)
        targets = [w for w in targets if ctx.owns_module(w)]
        if not targets:
            logger.warning("flex_awq_ssz: include matched 0 modules on this rank; skip")
            return
        logger.info(
            "flex_awq_ssz: {} target up-projections (rank={}/{})",
            len(targets),
            ctx.rank,
            ctx.world_size,
        )
        if not use_calib or not ctx.act_stats:
            logger.warning(
                "flex_awq_ssz: no calib acts — using |W_down| column max as act proxy "
                "(set use_calib_acts + --act-stats for real activation-aware search)"
            )

        done = 0
        for wname in targets:
            up_mod = module_name_from_weight(wname)
            down_mod = adapter.resolve_up_down(up_mod)
            if down_mod is None:
                logger.warning(
                    "flex_awq_ssz: adapter {} has no up-down partner for {}; skip",
                    adapter.name,
                    up_mod,
                )
                continue
            down_key = weights.weight_key(down_mod)
            if down_key is None:
                logger.warning("flex_awq_ssz: missing down weight {}; skip", down_mod)
                continue

            wu = require_float_weight(weights, wname, "flex_awq_ssz")
            wd = require_float_weight(weights, down_key, "flex_awq_ssz")
            if wu.dim() == 3 and wd.dim() == 3:
                self._apply_fused_experts(
                    weights,
                    ctx,
                    up_mod,
                    down_mod,
                    wu,
                    wd,
                    search_step=search_step,
                    ssz_step=ssz_step,
                    use_calib=use_calib,
                )
                done += 1
                continue

            if use_calib and down_mod in ctx.act_stats:
                act = ctx.act_stats[down_mod]
            elif use_calib and up_mod in ctx.act_stats:
                act = ctx.act_stats[up_mod]
                if act.shape[-1] != wd.shape[1]:
                    logger.warning(
                        "flex_awq_ssz: act for {} last dim {} != down in {}; using proxy",
                        up_mod,
                        act.shape[-1],
                        wd.shape[1],
                    )
                    act = _intermediate_act_proxy(wd)
            else:
                act = _intermediate_act_proxy(wd)

            # Run search on ctx.device when CUDA is available for this rank.
            if ctx.device.type == "cuda":
                act_s = act.to(ctx.device)
                wd_s = wd.to(ctx.device)
            else:
                act_s, wd_s = act, wd
            alpha, mse = search_alpha_awq_ssz(
                act_s, wd_s, search_step=search_step, ssz_step=ssz_step
            )
            a_max = act.float().abs().amax(dim=0) if act.dim() > 1 else act.float().abs()
            scales = make_scales_from_alpha_beta(
                a_max.cpu(),
                compute_weight_channel_max(wd, along="in"),
                alpha,
                0.0,
            )
            absorb_up_down(weights, up_mod, down_mod, scales)
            done += 1
            if done <= 5 or done % 50 == 0:
                logger.info("awq_ssz {}→{} alpha={:.2f} mse={:.4f}", up_mod, down_mod, alpha, mse)
        logger.info("flex_awq_ssz absorbed {} (rank={})", done, ctx.rank)
        if done == 0:
            logger.warning("flex_awq_ssz: matched targets but absorbed 0 pairs")

    def _apply_fused_experts(
        self,
        weights: WeightStore,
        ctx: PreprocessContext,
        up_mod: str,
        down_mod: str,
        wu: torch.Tensor,
        wd: torch.Tensor,
        *,
        search_step: float,
        ssz_step: int,
        use_calib: bool,
    ) -> None:
        """Per-expert AWQ on fused gate_up (up = second half) and down."""
        n_exp, two_i, _hidden = wu.shape
        if two_i % 2 != 0:
            raise ValueError(f"{up_mod}: fused out dim {two_i} is not 2*intermediate")
        inter = two_i // 2
        if wd.shape[0] != n_exp or wd.shape[-1] != inter:
            raise ValueError(
                f"flex_awq_ssz: fused shape {up_mod} {tuple(wu.shape)} vs "
                f"{down_mod} {tuple(wd.shape)}"
            )
        act_all = None
        if use_calib and down_mod in ctx.act_stats:
            act_all = ctx.act_stats[down_mod]
        elif use_calib and up_mod in ctx.act_stats:
            cand = ctx.act_stats[up_mod]
            if cand.shape[-1] == inter:
                act_all = cand
        scales = torch.empty(n_exp, inter, dtype=torch.float32)
        alpha_sum = 0.0
        for e in range(n_exp):
            down_e = wd[e]
            if act_all is not None and act_all.shape[0] == n_exp and act_all.shape[-1] == inter:
                act = act_all[e]
            else:
                act = _intermediate_act_proxy(down_e)
            if ctx.device.type == "cuda":
                act_s = act.to(ctx.device)
                wd_s = down_e.to(ctx.device)
            else:
                act_s, wd_s = act, down_e
            alpha, _mse = search_alpha_awq_ssz(
                act_s, wd_s, search_step=search_step, ssz_step=ssz_step
            )
            a_max = act.float().abs().amax(dim=0) if act.dim() > 1 else act.float().abs()
            scales[e] = make_scales_from_alpha_beta(
                a_max.cpu(),
                compute_weight_channel_max(down_e, along="in").cpu(),
                alpha,
                0.0,
            )
            alpha_sum += alpha
        absorb_fused_gate_up_down(weights, up_mod, down_mod, scales)
        logger.info(
            "awq_ssz fused {}→{} experts={} mean_alpha={:.2f}",
            up_mod,
            down_mod,
            n_exp,
            alpha_sum / max(n_exp, 1),
        )
