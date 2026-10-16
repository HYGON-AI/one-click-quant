"""QuaRot preprocess: fuse LN into Linear, then absorb Hadamard rotations into weights."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from loguru import logger
from safetensors.torch import save_file

from preprocess.io import WeightStore
from preprocess.transforms.base import PreprocessContext, PreprocessTransform
from preprocess.transforms.hadamard import (
    QuaRotMode,
    create_rot,
    fuse_gemma_norm_into_linear,
    fuse_ln_into_linear,
    rotate_weight,
)


class QuaRotTransform(PreprocessTransform):
    name = "quarot"

    def apply(self, weights: WeightStore, ctx: PreprocessContext) -> None:
        adapter = ctx.require_adapter()
        cfg = ctx.text_config
        recipe = ctx.recipe.get("quarot", {})
        block_size = int(recipe.get("block_size", 32))
        seed = int(recipe.get("seed", 1234))
        fuse_ln = bool(recipe.get("fuse_ln", True))
        max_layers = recipe.get("max_layers")  # optional smoke limit
        max_experts = recipe.get("max_experts")

        num_layers = int(cfg["num_hidden_layers"])
        if max_layers is not None:
            num_layers = min(num_layers, int(max_layers))
        num_experts = int(cfg.get("num_experts") or 0)
        if max_experts is not None:
            num_experts = min(num_experts, int(max_experts))

        compute_device = (
            ctx.device
            if isinstance(ctx.device, torch.device) and ctx.device.type == "cuda"
            else torch.device("cpu")
        )
        if fuse_ln:
            self._fuse_layernorms(weights, adapter, cfg, num_layers, ctx)

        pairs, meta = adapter.build_rotate_pairs(
            cfg, block_size=block_size, num_layers=num_layers, num_experts=num_experts or None
        )
        rots = self._build_rot_tensors(meta, seed=seed, cfg=cfg)
        # Pin rotation matrices on the compute device once (DeepSeek: ~7168² fp32).
        rots_compute = {k: v.to(compute_device) for k, v in rots.items()}
        logger.info(
            "QuaRot rotate compute_device={} rank={}/{}",
            compute_device,
            ctx.rank,
            ctx.world_size,
        )

        applied = 0
        skipped = 0
        for left_map, right_map in pairs:
            for mod, rot_key in right_map.items():
                if not ctx.owns_module(mod):
                    continue
                n = self._apply_one(
                    weights, mod, rots_compute[rot_key], right=True, device=compute_device
                )
                applied += int(n)
                skipped += int(not n)
                if applied and applied % 500 == 0:
                    logger.info(
                        "QuaRot rotate progress applied={} rank={}", applied, ctx.rank
                    )
                    weights.maybe_spill()
                    if compute_device.type == "cuda":
                        torch.cuda.empty_cache()
            for mod, rot_key in left_map.items():
                if not ctx.owns_module(mod):
                    continue
                n = self._apply_one(
                    weights, mod, rots_compute[rot_key], right=False, device=compute_device
                )
                applied += int(n)
                skipped += int(not n)
                if applied and applied % 500 == 0:
                    logger.info(
                        "QuaRot rotate progress applied={} rank={}", applied, ctx.rank
                    )
                    weights.maybe_spill()
                    if compute_device.type == "cuda":
                        torch.cuda.empty_cache()

        # persist global rotation for reproducibility (rank 0 only)
        out_opt = Path(ctx.extras.get("save_dir", "")) / "optional"
        if ctx.extras.get("save_dir") and ctx.rank == 0 and "rot" in rots:
            out_opt.mkdir(parents=True, exist_ok=True)
            save_file({"global_rotation": rots["rot"].cpu()}, str(out_opt / "quarot.safetensors"))

        if compute_device.type == "cuda":
            del rots_compute
            torch.cuda.empty_cache()

        logger.info(
            "QuaRot done: rank={}/{} applied={} skipped_missing={} device={}",
            ctx.rank,
            ctx.world_size,
            applied,
            skipped,
            compute_device,
        )

    def _build_rot_tensors(self, meta: dict, seed: int, cfg: dict) -> Dict[str, torch.Tensor]:
        mode_map = {
            "hadamard": QuaRotMode.HADAMARD,
            "block_hadamard_shifted": QuaRotMode.BLOCK_HADAMARD_SHIFTED,
            "orthogonal": QuaRotMode.ORTHOGONAL,
        }
        rots: Dict[str, torch.Tensor] = {}
        for key, spec in meta.items():
            if key == "qk_rope":
                continue
            if not isinstance(spec, dict) or "size" not in spec:
                continue
            size = int(spec["size"])
            if size <= 0:
                continue
            mode = mode_map[spec.get("mode", "hadamard")]
            rots[key] = create_rot(
                mode,
                size=size,
                block_size=int(spec.get("block_size", -1)),
                seed=seed,
                dtype=torch.float32,
            )

        # kv_a stacked rotation: block_diag(rot_kv, I_rope)
        qk_rope = int(meta.get("qk_rope") or cfg.get("qk_rope_head_dim") or 0)
        if "rot_kv" in rots and qk_rope > 0:
            eye = torch.eye(qk_rope, dtype=torch.float32)
            rots["rot_kv_block"] = torch.block_diag(rots["rot_kv"], eye)
        elif "rot_kv" in rots:
            rots["rot_kv_block"] = rots["rot_kv"]
        return rots

    def _apply_one(
        self,
        weights: WeightStore,
        module: str,
        rot: torch.Tensor,
        right: bool,
        device: torch.device | None = None,
    ) -> bool:
        key = weights.weight_key(module)
        if key is None:
            return False
        w = weights.get(key, device=device if device is not None else "cpu")
        if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                f"QuaRot: expected float weight for {key}, got {w.dtype}. "
                "Dequantize MXFP4/INT packs before preprocess."
            )
        new_w = rotate_weight(w, rot, right_rotate=right, device=device)
        # WeightStore.set() moves to CPU; drop GPU copy explicitly.
        weights.set(key, new_w.detach())
        del new_w
        # Left rotation mixes output channels: bias must be rotated the same way.
        if not right:
            bias_key = f"{module}.bias"
            if weights.has(bias_key):
                b = weights.get(bias_key)
                if b.dtype in (torch.float16, torch.bfloat16, torch.float32) and b.dim() == 1:
                    dest = device if device is not None else b.device
                    r = rot.to(device=dest, dtype=torch.float32)
                    bf = b.to(device=dest, dtype=torch.float32)
                    if b.numel() == r.shape[0]:
                        weights.set(bias_key, (r.T @ bf).to(dtype=b.dtype).detach())
                    elif b.numel() % r.shape[0] == 0:
                        r_full = torch.block_diag(*[r] * (b.numel() // r.shape[0]))
                        weights.set(bias_key, (r_full.T @ bf).to(dtype=b.dtype).detach())
                    else:
                        logger.warning("QuaRot: skip bias rotate for {} (dim {})", bias_key, b.numel())
        return True

    def _fuse_layernorms(
        self,
        weights: WeightStore,
        adapter,
        cfg: dict,
        num_layers: int,
        ctx: PreprocessContext,
    ) -> None:
        ln_map = adapter.build_ln_fuse_map(cfg, num_layers)
        gemma = adapter.norm_affine() == "gemma"
        fuse_fn = fuse_gemma_norm_into_linear if gemma else fuse_ln_into_linear
        compute_device = (
            ctx.device
            if isinstance(ctx.device, torch.device) and ctx.device.type == "cuda"
            else torch.device("cpu")
        )
        logger.info(
            "QuaRot LN-fuse compute_device={} rank={}/{}",
            compute_device,
            ctx.rank,
            ctx.world_size,
        )
        fused = 0
        for ln_mod, linears in ln_map.items():
            if not ctx.owns_module(ln_mod):
                continue
            ln_key = weights.weight_key(ln_mod)
            if ln_key is None:
                continue
            ln_w = weights.get(ln_key, device=compute_device)
            if ln_w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise TypeError(
                    f"QuaRot LN-fuse: expected float norm {ln_key}, got {ln_w.dtype}"
                )
            fused_here = 0
            for lin in linears:
                lin_key = weights.weight_key(lin)
                if lin_key is None:
                    continue
                w = weights.get(lin_key, device=compute_device)
                if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise TypeError(
                        f"QuaRot LN-fuse: expected float linear {lin_key}, got {w.dtype}"
                    )
                try:
                    weights.set(lin_key, fuse_fn(ln_w, w))
                    fused += 1
                    fused_here += 1
                except ValueError as exc:
                    logger.warning("LN fuse skip {} -> {}: {}", ln_mod, lin, exc)
                if compute_device.type == "cuda" and fused_here and fused_here % 64 == 0:
                    torch.cuda.empty_cache()
            # Only clear gamma when at least one Linear absorbed it.
            # RMSNorm identity is ones; Gemma identity is zeros (forward uses 1+gamma).
            if fused_here > 0:
                if gemma:
                    weights.set(ln_key, torch.zeros_like(ln_w))
                else:
                    weights.set(ln_key, torch.ones_like(ln_w))
            else:
                logger.warning(
                    "QuaRot LN-fuse: {} had 0 successful Linear absorbs; keeping original gamma",
                    ln_mod,
                )
            # DeepSeek: thousands of expert linears; spill early so dirty cache
            # cannot accumulate across the whole LN-fuse pass.
            if fused and fused % 200 == 0:
                n = weights.maybe_spill()
                if n:
                    logger.info(
                        "QuaRot LN-fuse mid-spill tensors={} fused={} rank={}",
                        n,
                        fused,
                        ctx.rank,
                    )
        logger.info("QuaRot LN-fuse absorbed into {} linears (rank={})", fused, ctx.rank)
