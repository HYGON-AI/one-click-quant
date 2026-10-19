"""CLI entry for ``--alg preprocess``.

Usage::

    python main.py --alg preprocess \\
        --model /path/to/Kimi-K3 \\
        --save-dir /path/to/Kimi-K3-preprocessed \\
        --recipe preprocess/recipes/kimi_k3.yaml \\
        --transforms quarot,flex_awq_ssz,flex_smooth_quant \\
        --preprocess-gpus 8

Then quantize the float output with any method::

    python main.py --alg slimquant_ptq --model /path/to/Kimi-K3-preprocessed --scheme W4A8 ...
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `import preprocess` when launched via importlib from templates/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from preprocess.pipeline import run_preprocess, run_preprocess_multiprocess


def run(args) -> None:
    model = args.model
    if not model:
        raise SystemExit("--model is required for preprocess")
    save_dir = args.save_dir or f"{model.rstrip('/')}-preprocessed"
    recipe = getattr(args, "recipe", None)
    transforms = getattr(args, "transforms", None)
    tlist = None
    if transforms:
        tlist = [x.strip() for x in str(transforms).split(",") if x.strip()]
    device = getattr(args, "preprocess_device", None) or "cpu"
    act_stats_path = getattr(args, "act_stats", None)
    gpus = int(getattr(args, "preprocess_gpus", 1) or 1)

    logger.info("preprocess: {} -> {} (gpus={})", model, save_dir, gpus)
    if gpus > 1:
        run_preprocess_multiprocess(
            model,
            save_dir,
            gpus=gpus,
            recipe_path=recipe,
            transforms=tlist,
            act_stats_path=act_stats_path,
        )
    else:
        from preprocess.calib import load_act_stats

        act_stats = load_act_stats(act_stats_path)
        run_preprocess(
            model,
            save_dir,
            recipe_path=recipe,
            transforms=tlist,
            device=device,
            act_stats=act_stats,
        )
    logger.info("done. Feed --model {} to slimquant_ptq / other algs.", save_dir)


# When executed via _execute_template, main.py may call module-level expected names.
# Keep a simple main for direct python templates/preprocess_ptq.py
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--recipe", default=None)
    p.add_argument("--transforms", default=None)
    p.add_argument("--preprocess-device", default="cpu")
    p.add_argument("--preprocess-gpus", type=int, default=1)
    p.add_argument("--act-stats", default=None, help="Optional torch.save dict of module act tensors")
    ns = p.parse_args()
    run(ns)
