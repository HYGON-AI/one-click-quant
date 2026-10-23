"""Preprocess pipeline: run selected transforms, write float HF checkpoint."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
import torch.multiprocessing as mp
import yaml
from loguru import logger

from preprocess.adapters import get_adapter, resolve_adapter_name
from preprocess.io import WeightStore, load_hf_config
from preprocess.transforms import TRANSFORM_REGISTRY
from preprocess.transforms.base import PreprocessContext


def _maybe_unpack_to_bf16(
    model_dir: str,
    save_dir: str,
    recipe: Dict[str, Any],
    adapter_name: str,
    *,
    num_gpus: int = 1,
) -> str:
    """Stream-unpack packed/quantized HF weights to a BF16 staging dir if needed."""
    staging = str(Path(save_dir).resolve()) + ".bf16_staging"

    if adapter_name == "kimi_k3":
        if not recipe.get("unpack_mxfp4", True):
            return model_dir
        from preprocess.adapters.kimi_k3 import KimiK3Adapter

        if not KimiK3Adapter.model_needs_mxfp4_unpack(model_dir):
            logger.info("preprocess: no MXFP4 packs under {}; skip staging unpack", model_dir)
            return model_dir
        logger.info(
            "preprocess: streaming MXFP4→BF16 unpack {} -> {} (before multi-GPU)",
            model_dir,
            staging,
        )
        KimiK3Adapter.unpack_mxfp4_checkpoint(model_dir, staging)
        return staging

    if adapter_name == "deepseek_v3":
        if not recipe.get("unpack_fp8", True):
            return model_dir
        from preprocess.unpack.fp8 import model_needs_fp8_unpack, unpack_fp8_checkpoint

        if not model_needs_fp8_unpack(model_dir):
            logger.info("preprocess: no FP8 scales under {}; skip staging unpack", model_dir)
            return model_dir
        unpack_gpus = int(recipe.get("unpack_fp8_gpus") or num_gpus or 1)
        logger.info(
            "preprocess: block-FP8→BF16 unpack {} -> {} (gpus={}, 1:1 shards)",
            model_dir,
            staging,
            unpack_gpus,
        )
        unpack_fp8_checkpoint(model_dir, staging, num_gpus=unpack_gpus)
        return staging

    return model_dir


def _maybe_unpack_kimi_mxfp4(
    model_dir: str,
    save_dir: str,
    recipe: Dict[str, Any],
    adapter_name: str,
    *,
    num_gpus: int = 1,
) -> str:
    """Back-compat alias for ``_maybe_unpack_to_bf16``."""
    return _maybe_unpack_to_bf16(
        model_dir, save_dir, recipe, adapter_name, num_gpus=num_gpus
    )


def _default_recipe_path() -> Path:
    return Path(__file__).resolve().parent / "recipes" / "kimi_k3.yaml"


def load_recipe(path: Optional[str | os.PathLike] = None) -> Dict[str, Any]:
    p = Path(path) if path else _default_recipe_path()
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_transforms(names: Sequence[str] | None, recipe: Dict[str, Any]) -> List[str]:
    if names:
        return list(names)
    ordered = recipe.get("transforms") or ["quarot", "flex_awq_ssz", "flex_smooth_quant"]
    return list(ordered)


def layer_ids_for_rank(num_layers: int, rank: int, world_size: int) -> Set[int]:
    """Round-robin layer assignment: ``i % world_size == rank``."""
    return {i for i in range(num_layers) if i % world_size == rank}


def _write_preprocess_meta(
    save_dir: Path,
    config: Dict[str, Any],
    *,
    adapter_name: str,
    tnames: List[str],
    recipe_path: Optional[str | os.PathLike],
    model_dir: str,
    gpus: int = 1,
) -> None:
    out_cfg = dict(config)
    out_cfg.pop("compression_config", None)
    out_cfg.pop("quantization_config", None)
    if isinstance(out_cfg.get("text_config"), dict):
        out_cfg["text_config"] = dict(out_cfg["text_config"])
        out_cfg["text_config"].pop("compression_config", None)
        out_cfg["text_config"].pop("quantization_config", None)
    out_cfg["preprocess"] = {
        "adapter": adapter_name,
        "transforms": tnames,
        "recipe": str(recipe_path or _default_recipe_path()),
        "gpus": gpus,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps(out_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    marker = {
        "status": "ok",
        "transforms": tnames,
        "source": model_dir,
        "gpus": gpus,
    }
    (save_dir / "PREPROCESS_DONE.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )


def run_preprocess(
    model_dir: str | os.PathLike,
    save_dir: str | os.PathLike,
    *,
    recipe_path: Optional[str | os.PathLike] = None,
    transforms: Optional[Sequence[str]] = None,
    device: str | torch.device = "cpu",
    act_stats: Optional[Dict[str, torch.Tensor]] = None,
    layer_ids: Optional[Set[int]] = None,
    rank: int = 0,
    world_size: int = 1,
    finalize: bool = True,
) -> "Path | WeightStore":
    """Apply preprocess transforms and optionally write a float checkpoint.

    When ``finalize=False`` (multi-GPU workers), returns the mutated
    ``WeightStore`` without saving; caller dumps dirty tensors.
    """
    model_dir = str(Path(model_dir).resolve())
    save_dir = str(Path(save_dir).resolve())
    if model_dir == save_dir:
        raise ValueError(
            f"Refuse to overwrite model_dir in place: {model_dir}. "
            "Pass a different --save-dir."
        )
    recipe = load_recipe(recipe_path)
    tnames = resolve_transforms(transforms, recipe)
    device = torch.device(device)

    config = load_hf_config(model_dir)
    adapter_name = resolve_adapter_name(recipe, config)
    # Streaming unpack before opening WeightStore for transforms (avoids RAM OOM).
    if finalize and rank == 0 and world_size == 1:
        model_dir = _maybe_unpack_kimi_mxfp4(
            model_dir, save_dir, recipe, adapter_name, num_gpus=1
        )
        config = load_hf_config(model_dir)

    adapter = get_adapter(adapter_name)
    store = WeightStore(model_dir)
    # MoE (DeepSeek-V3) can dirty 10k+ expert BF16 tensors per rank;
    # spill dirty weights to disk so host OOM-killer does not SIGKILL workers.
    need_spill = (not finalize) or adapter_name in ("deepseek_v3",)
    if need_spill:
        spill_root = _spill_base(Path(save_dir)) / f"rank{rank}_spill"
        # Cap in-RAM dirty cache tightly: 8 ranks × 32GiB previously approached
        # host OOM on bw47 (~750GiB). Prefer smaller cache + more spill I/O.
        store.enable_spill(spill_root, max_cache_bytes=16 * 1024**3)
    ctx = PreprocessContext(
        model_dir=model_dir,
        config=config,
        recipe=recipe,
        device=device,
        adapter=adapter,
        act_stats=act_stats or {},
        extras={
            "save_dir": save_dir,
            "adapter": adapter_name,
            "skip_inline_unpack": True,
        },
        layer_ids=layer_ids,
        rank=rank,
        world_size=world_size,
    )

    logger.info(
        "preprocess start: model={} adapter={} transforms={} rank={}/{} layers={}",
        model_dir,
        adapter_name,
        tnames,
        rank,
        world_size,
        sorted(layer_ids) if layer_ids is not None else "all",
    )
    # Adapter-specific prep (Kimi-K3: MXFP4 → BF16) before QuaRot / Flex.
    adapter.prepare_weights(store, ctx)
    for name in tnames:
        if name not in TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown transform '{name}'. Available: {sorted(TRANSFORM_REGISTRY)}")
        cls = TRANSFORM_REGISTRY[name]
        logger.info("running transform: {} (rank={})", name, rank)
        cls().apply(store, ctx)

    if not finalize:
        return store

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    store.save(save_dir, copy_non_weights=True)
    _write_preprocess_meta(
        Path(save_dir),
        config,
        adapter_name=adapter_name,
        tnames=tnames,
        recipe_path=recipe_path,
        model_dir=model_dir,
        gpus=world_size,
    )
    if need_spill:
        shutil.rmtree(_spill_base(Path(save_dir), create=False), ignore_errors=True)
    logger.info("preprocess finished -> {}", save_dir)
    return Path(save_dir)


def _shard_dir(save_dir: Path) -> Path:
    return Path(save_dir) / ".preprocess_shards"


def _spill_base(save_dir: Path, *, create: bool = True) -> Path:
    """Directory for large dirty-spill chunks (may be off the model volume).

    Set ``PREPROCESS_SPILL_ROOT`` to a spacious filesystem (e.g. /public) when
    ``--save-dir`` sits on a small local NVMe that cannot hold staging+spill.

    Default spill is a **sibling** of ``save_dir`` (not under it). Multiprocess
    ``commit_inplace_and_move`` may ``rmtree(save_dir)`` before rewriting dirty
    shards; keeping merge/rank spill under ``save_dir/.preprocess_shards`` would
    delete live spill chunks and crash on reload.
    """
    env = os.environ.get("PREPROCESS_SPILL_ROOT", "").strip()
    if env:
        base = Path(env) / Path(save_dir).name
    else:
        sd = Path(save_dir).resolve()
        base = sd.parent / f".preprocess_spill_{sd.name}"
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def _worker_entry(
    rank: int,
    world_size: int,
    model_dir: str,
    save_dir: str,
    recipe_path: Optional[str],
    transforms: Optional[List[str]],
    act_stats_path: Optional[str],
) -> None:
    """One GPU worker: own layers, run transforms, dump dirty tensors."""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)

    from preprocess.calib import load_act_stats

    act_stats = load_act_stats(act_stats_path)
    cfg = load_hf_config(model_dir)
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    num_layers = int(tc.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)
    # DeepSeek-V3 MTP / next-n blocks sit at indices
    # [num_hidden_layers, num_hidden_layers + num_nextn).
    num_nextn = int(tc.get("num_nextn_predict_layers") or cfg.get("num_nextn_predict_layers") or 0)
    lids = layer_ids_for_rank(num_layers + max(num_nextn, 0), rank, world_size)

    store = run_preprocess(
        model_dir,
        save_dir,
        recipe_path=recipe_path,
        transforms=transforms,
        device=device,
        act_stats=act_stats,
        layer_ids=lids,
        rank=rank,
        world_size=world_size,
        finalize=False,
    )
    assert isinstance(store, WeightStore)
    part_dir = _shard_dir(Path(save_dir))
    part_dir.mkdir(parents=True, exist_ok=True)
    # Prefer spill manifest (chunked safetensors) over a single giant .pt —
    # DeepSeek-V3 can exceed host RAM if dirty is materialized at once.
    if store._spill_dir is not None:
        manifest = store.export_spill_manifest()
        part_path = part_dir / f"rank{rank}.json"
        part_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "rank {} dumped spill manifest: {} dirty / {} deleted / {} chunks -> {}",
            rank,
            manifest.get("dirty_count", 0),
            len(manifest.get("deleted") or []),
            len(manifest.get("spill_files") or []),
            part_path,
        )
    else:
        payload = {
            "dirty": store.export_dirty(),
            "deleted": store.export_deleted(),
        }
        part_path = part_dir / f"rank{rank}.pt"
        torch.save(payload, part_path)
        logger.info(
            "rank {} dumped {} dirty / {} deleted -> {}",
            rank,
            len(payload["dirty"]),
            len(payload["deleted"]),
            part_path,
        )


def run_preprocess_multiprocess(
    model_dir: str | os.PathLike,
    save_dir: str | os.PathLike,
    *,
    gpus: int,
    recipe_path: Optional[str | os.PathLike] = None,
    transforms: Optional[Sequence[str]] = None,
    act_stats_path: Optional[str | os.PathLike] = None,
) -> Path:
    """Layer-parallel preprocess across ``gpus`` CUDA devices (not MoE EP)."""
    if gpus < 1:
        raise ValueError(f"gpus must be >= 1, got {gpus}")
    model_dir = str(Path(model_dir).resolve())
    save_dir_p = Path(save_dir).resolve()
    save_dir = str(save_dir_p)
    if model_dir == save_dir:
        raise ValueError(
            f"Refuse to overwrite model_dir in place: {model_dir}. "
            "Pass a different --save-dir."
        )
    if gpus == 1:
        from preprocess.calib import load_act_stats

        return run_preprocess(
            model_dir,
            save_dir,
            recipe_path=recipe_path,
            transforms=transforms,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            act_stats=load_act_stats(act_stats_path),
            finalize=True,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("--preprocess-gpus > 1 requires CUDA")
    if torch.cuda.device_count() < gpus:
        raise RuntimeError(
            f"Requested --preprocess-gpus={gpus} but only "
            f"{torch.cuda.device_count()} CUDA device(s) visible"
        )

    save_dir_p.mkdir(parents=True, exist_ok=True)
    shard_dir = _shard_dir(save_dir_p)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    recipe = load_recipe(recipe_path)
    config = load_hf_config(model_dir)
    adapter_name = resolve_adapter_name(recipe, config)
    # Unpack once in the parent process before spawning GPU transform workers.
    # DeepSeek FP8 unpack itself can use the same GPU count (1:1 shard files).
    model_dir = _maybe_unpack_kimi_mxfp4(
        model_dir, save_dir, recipe, adapter_name, num_gpus=gpus
    )

    tlist = list(transforms) if transforms else None
    recipe_s = str(recipe_path) if recipe_path else None
    act_s = str(act_stats_path) if act_stats_path else None

    logger.info(
        "preprocess multiprocess: gpus={} model={} -> {} (layer-parallel, not EP)",
        gpus,
        model_dir,
        save_dir,
    )
    # spawn workers; each dumps rank{r}.pt then exits
    mp.spawn(
        _worker_entry,
        args=(gpus, model_dir, save_dir, recipe_s, tlist, act_s),
        nprocs=gpus,
        join=True,
    )

    # Parent merges dirty parts and writes the HF checkpoint.
    recipe = load_recipe(recipe_path)
    tnames = resolve_transforms(tlist, recipe)
    config = load_hf_config(model_dir)
    adapter_name = resolve_adapter_name(recipe, config)
    store = WeightStore(model_dir)
    # Parent also spills so merging 8 ranks of DeepSeek experts cannot OOM.
    store.enable_spill(_spill_base(save_dir_p) / "merge_spill", max_cache_bytes=16 * 1024**3)
    total_dirty = 0
    total_deleted = 0
    from safetensors import safe_open

    for r in range(gpus):
        part_json = shard_dir / f"rank{r}.json"
        part_pt = shard_dir / f"rank{r}.pt"
        if part_json.is_file():
            manifest = json.loads(part_json.read_text(encoding="utf-8"))
            deleted = list(manifest.get("deleted") or [])
            for name in deleted:
                store.delete(name)
                total_deleted += 1
            n_merged = 0
            for spill_path in manifest.get("spill_files") or []:
                sp = Path(spill_path)
                if not sp.is_file():
                    raise FileNotFoundError(f"missing spill chunk: {sp}")
                with safe_open(sp, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        store.set(k, f.get_tensor(k))
                        n_merged += 1
                        total_dirty += 1
            logger.info(
                "merged rank{}: {} tensors from {} chunks, {} deletes",
                r,
                n_merged,
                len(manifest.get("spill_files") or []),
                len(deleted),
            )
            continue

        if not part_pt.is_file():
            raise FileNotFoundError(f"missing preprocess shard: {part_json} or {part_pt}")
        payload = torch.load(part_pt, map_location="cpu", weights_only=False)
        # New format: {"dirty": {...}, "deleted": [...]}; legacy: flat tensor dict.
        if isinstance(payload, dict) and "dirty" in payload:
            dirty = payload["dirty"]
            deleted = list(payload.get("deleted") or [])
        elif isinstance(payload, dict):
            dirty = payload
            deleted = []
        else:
            raise TypeError(f"shard {part_pt} must be a dict")
        for name in deleted:
            store.delete(name)
            total_deleted += 1
        if not isinstance(dirty, dict):
            raise TypeError(f"shard {part_pt} dirty must be dict[str, Tensor]")
        for k, v in dirty.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"shard {part_pt} key {k!r} is not a Tensor")
            store.set(k, v)
            total_dirty += 1
        logger.info(
            "merged rank{}: {} tensors, {} deletes",
            r,
            len(dirty),
            len(deleted),
        )

    # Staging BF16 (~1.3 TiB) + full copy will not fit on a 3.5 TiB /module.
    # When model_dir is a *.bf16_staging tree, rewrite dirty shards in-place and
    # rename staging -> save_dir (one volume, ~1 shard of temp space).
    # PREPROCESS_INPLACE_SAVE alone must NOT rewrite a user original checkpoint.
    is_staging = str(model_dir).endswith(".bf16_staging")
    env_inplace = os.environ.get("PREPROCESS_INPLACE_SAVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if env_inplace and not is_staging:
        raise ValueError(
            "PREPROCESS_INPLACE_SAVE is set but model_dir is not a *.bf16_staging "
            f"tree ({model_dir!r}); refusing to rewrite/rename the original "
            "checkpoint. Unpack to staging first, or unset PREPROCESS_INPLACE_SAVE."
        )
    if is_staging:
        store.commit_inplace_and_move(save_dir)
        # Rescue spill may have been moved under staging then renamed into save_dir.
        shutil.rmtree(save_dir_p / ".merge_spill_live", ignore_errors=True)
    else:
        store.save(save_dir, copy_non_weights=True)
    _write_preprocess_meta(
        save_dir_p,
        config,
        adapter_name=adapter_name,
        tnames=tnames,
        recipe_path=recipe_path,
        model_dir=model_dir,
        gpus=gpus,
    )
    shutil.rmtree(shard_dir, ignore_errors=True)
    # Always drop spill tree for this save_dir (env root or sibling default).
    shutil.rmtree(_spill_base(save_dir_p, create=False), ignore_errors=True)
    logger.info(
        "preprocess multiprocess finished -> {} "
        "(merged {} dirty, {} deleted, gpus={})",
        save_dir,
        total_dirty,
        total_deleted,
        gpus,
    )
    return save_dir_p


def assert_float_checkpoint(model_dir: str | os.PathLike, sample: int = 8) -> None:
    """Raise if config looks quantized or sampled Linear weights are not float."""
    cfg = load_hf_config(model_dir)
    for key in ("compression_config", "quantization_config"):
        if key in cfg:
            raise TypeError(
                f"Expected float preprocess output, but config.json has '{key}'. "
                "Pass the preprocess --save-dir, not a quantized checkpoint."
            )
        tc = cfg.get("text_config")
        if isinstance(tc, dict) and key in tc:
            raise TypeError(
                f"Expected float preprocess output, but text_config has '{key}'."
            )

    store = WeightStore(model_dir)
    checked = 0
    for name in store.names():
        if not name.endswith(".weight"):
            continue
        if "embed" in name or "norm" in name:
            continue
        t = store.get(name)
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                f"Expected float weights in preprocessed ckpt, got {t.dtype} for {name}. "
                "Re-run preprocess; do not pass an already-quantized directory."
            )
        checked += 1
        if checked >= sample:
            break
    if checked == 0:
        raise RuntimeError(f"No .weight tensors found under {model_dir}")
