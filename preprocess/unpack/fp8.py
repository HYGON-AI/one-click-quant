"""DeepSeek-style block FP8 → BF16 checkpoint unpack (1:1 shards, optional multi-GPU)."""

from __future__ import annotations

import gc
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open

from preprocess.transforms.base import module_layer_id
from preprocess.unpack.shard_io import (
    cleanup_legacy_u_parts,
    clear_rank_manifests,
    load_and_merge_rank_manifests,
    shard_keys_complete,
    write_rank_manifest,
    write_safetensors,
)

if TYPE_CHECKING:
    from preprocess.io import WeightStore

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
    torch.float8_e5m2,
    getattr(torch, "float8_e5m2fnuz", torch.float8_e5m2),
)
_FP8_GROUP_SIZE = 128


def fp8_group_size(cfg: Dict[str, Any]) -> int:
    qc = cfg.get("quantization_config") or {}
    block = qc.get("weight_block_size")
    if isinstance(block, (list, tuple)) and block:
        return int(block[0])
    return _FP8_GROUP_SIZE


def dequantize_weight_from_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    group_size: int = _FP8_GROUP_SIZE,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """DeepSeek-style block FP8 → BF16 (``W * scale_inv``, block ``group_size``)."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2D FP8 weight, got {tuple(weight.shape)}")
    g = int(group_size)
    d_out, d_in = weight.shape
    pad_out = math.ceil(d_out / g) * g - d_out
    pad_in = math.ceil(d_in / g) * g - d_in
    if device is not None:
        weight = weight.to(device=device, non_blocking=True)
        scale_inv = scale_inv.to(device=device, non_blocking=True)
    w = F.pad(weight.float(), (0, pad_in, 0, pad_out))
    s = scale_inv.float()
    d_out_pad, d_in_pad = w.shape
    expected = (d_out_pad // g, d_in_pad // g)
    if tuple(s.shape) != expected:
        if s.numel() != expected[0] * expected[1]:
            raise ValueError(
                f"FP8 scale shape {tuple(s.shape)} incompatible with weight "
                f"{tuple(weight.shape)} group={g} (expected blocks {expected})"
            )
        s = s.reshape(expected)
    w = w.view(d_out_pad // g, g, d_in_pad // g, g)
    s = s.view(d_out_pad // g, 1, d_in_pad // g, 1)
    out = (w * s).view(d_out_pad, d_in_pad)[:d_out, :d_in]
    return out.to(torch.bfloat16)


def _expected_non_scale_keys(keys: List[str]) -> Set[str]:
    return {k for k in keys if not k.endswith(".weight_scale_inv")}


def _unpack_one_fp8_shard(
    *,
    src: Path,
    dst: Path,
    shard: str,
    keys: List[str],
    weight_map: Dict[str, str],
    group_size: int,
    device: torch.device,
) -> Tuple[Dict[str, str], int]:
    """Unpack one source shard → one same-named BF16 safetensors. Returns (wm, n)."""
    fp8_mods = {
        k[: -len(".weight")]
        for k in keys
        if k.endswith(".weight") and f"{k}_scale_inv" in weight_map
    }
    for k in keys:
        if k.endswith(".weight_scale_inv"):
            wname = k[: -len("_scale_inv")]
            if wname.endswith(".weight"):
                fp8_mods.add(wname[: -len(".weight")])

    out_path = dst / shard
    new_wm: Dict[str, str] = {}

    if not fp8_mods:
        shutil.copy2(src / shard, out_path)
        for k in keys:
            new_wm[k] = shard
        return new_wm, 0

    out: Dict[str, torch.Tensor] = {}
    unpacked = 0
    with safe_open(src / shard, framework="pt", device="cpu") as f:
        local_keys = set(f.keys())
        for k in keys:
            if k.endswith(".weight_scale_inv"):
                continue
            if k.endswith(".weight"):
                mod = k[: -len(".weight")]
                if mod in fp8_mods:
                    continue
            if k not in local_keys:
                continue
            t = f.get_tensor(k)
            if k.endswith(".weight") and t.dtype in _FP8_DTYPES:
                raise RuntimeError(
                    "fp8 unpack: FP8 weight "
                    f"{k} has no weight_scale_inv in the index; "
                    "refusing to copy raw FP8 or mark unpack complete."
                )
            out[k] = t

        for mod in sorted(fp8_mods):
            w_key = f"{mod}.weight"
            s_key = f"{mod}.weight_scale_inv"
            if w_key not in local_keys:
                continue
            if s_key not in weight_map:
                raise RuntimeError(
                    "fp8 unpack: missing weight_scale_inv for "
                    f"{w_key}; refusing to copy raw FP8 or mark unpack complete."
                )
            w = f.get_tensor(w_key)
            scale_shard = weight_map[s_key]
            if scale_shard == shard and s_key in local_keys:
                s = f.get_tensor(s_key)
            else:
                with safe_open(src / scale_shard, framework="pt", device="cpu") as fs:
                    s = fs.get_tensor(s_key)
            if w.dtype not in _FP8_DTYPES:
                out[w_key] = w
                continue
            bf16 = dequantize_weight_from_fp8(
                w, s, group_size=group_size, device=device
            )
            if bf16.device.type != "cpu":
                bf16 = bf16.cpu()
            bf16 = bf16.contiguous()
            del w, s
            out[w_key] = bf16
            unpacked += 1

    write_safetensors(out_path, out)
    for k in out:
        new_wm[k] = shard
    del out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return new_wm, unpacked


def _unpack_fp8_worker(
    rank: int,
    world_size: int,
    src_s: str,
    dst_s: str,
    shards: List[str],
    by_shard: Dict[str, List[str]],
    weight_map: Dict[str, str],
    group_size: int,
    use_cuda: bool,
) -> None:
    """mp.spawn entry: unpack shards where ``i % world_size == rank``."""
    src = Path(src_s)
    dst = Path(dst_s)
    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    new_wm: Dict[str, str] = {}
    total_unpacked = 0
    mine = [(i, sh) for i, sh in enumerate(shards) if i % world_size == rank]
    logger.info(
        "fp8 unpack rank {}/{}: {} shards on {} (group={})",
        rank,
        world_size,
        len(mine),
        device,
        group_size,
    )

    for si, shard in mine:
        keys = by_shard[shard]
        expected = _expected_non_scale_keys(keys)
        stem = Path(shard).stem
        cleanup_legacy_u_parts(dst, stem)
        out_path = dst / shard
        if shard_keys_complete(out_path, expected):
            with safe_open(out_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    new_wm[k] = shard
            logger.info(
                "[{}/{}] rank{} resume skip {}",
                si + 1,
                len(shards),
                rank,
                shard,
            )
            continue
        if out_path.is_file():
            out_path.unlink(missing_ok=True)

        wm, n = _unpack_one_fp8_shard(
            src=src,
            dst=dst,
            shard=shard,
            keys=keys,
            weight_map=weight_map,
            group_size=group_size,
            device=device,
        )
        new_wm.update(wm)
        total_unpacked += n
        logger.info(
            "[{}/{}] rank{} wrote {} ({} tensors, {} fp8→bf16, device={})",
            si + 1,
            len(shards),
            rank,
            shard,
            len(wm),
            n,
            device,
        )

    write_rank_manifest(
        dst,
        rank,
        weight_map=new_wm,
        unpacked_modules=total_unpacked,
        shards_done=[sh for _, sh in mine],
    )
    logger.info(
        "fp8 unpack rank {}/{} finished: {} modules",
        rank,
        world_size,
        total_unpacked,
    )


def model_needs_fp8_unpack(model_dir: str | Path) -> bool:
    """True if checkpoint index/config indicates block FP8 weights."""
    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    wm = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
    if any(k.endswith(".weight_scale_inv") for k in wm):
        return True
    cfg_path = Path(model_dir) / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        qc = cfg.get("quantization_config") or {}
        if str(qc.get("quant_method", "")).lower() == "fp8":
            return True
    return False


def unpack_fp8_weights_inline(
    weights: "WeightStore",
    *,
    cfg: Dict[str, Any],
    layer_ids: Optional[Set[int]] = None,
    rank: int = 0,
    flush_every: int = 64,
) -> int:
    """In-memory FP8 unpack (smoke only). Prefer ``unpack_fp8_checkpoint``."""
    g = fp8_group_size(cfg)
    scale_keys = [n for n in weights.names() if n.endswith(".weight_scale_inv")]
    done = 0
    for scale_name in scale_keys:
        weight_name = scale_name[: -len("_scale_inv")]
        if not weight_name.endswith(".weight"):
            continue
        module = weight_name[: -len(".weight")]
        lid = module_layer_id(module)
        if layer_ids is not None:
            if lid is None:
                if rank != 0:
                    continue
            elif lid not in layer_ids:
                continue
        if not weights.has(weight_name):
            logger.warning("fp8 unpack: missing {}; skip {}", weight_name, scale_name)
            continue
        shard = weights.weight_map.get(weight_name)
        w = weights.get(weight_name, device="cpu")
        s = weights.get(scale_name, device="cpu")
        weights._cache.pop(weight_name, None)
        weights._cache.pop(scale_name, None)
        if w.dtype not in _FP8_DTYPES:
            del w, s
            continue
        try:
            bf16 = dequantize_weight_from_fp8(w, s, group_size=g).cpu()
        except ValueError as exc:
            logger.warning("fp8 unpack failed for {}: {}", module, exc)
            continue
        del w, s
        weights.set(weight_name, bf16, shard=shard)
        weights.delete(scale_name)
        done += 1
        if done <= 3 or done % 1000 == 0:
            logger.info(
                "fp8 unpack #{} {} -> BF16 {}",
                done,
                weight_name,
                tuple(bf16.shape),
            )
        del bf16
        if flush_every and done % flush_every == 0:
            gc.collect()
    logger.info(
        "fp8 unpack: dequantized {} FP8 modules (rank={}, layers={})",
        done,
        rank,
        sorted(layer_ids) if layer_ids is not None else "all",
    )
    return done


def unpack_fp8_checkpoint(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    num_gpus: int = 1,
    group_size: Optional[int] = None,
) -> Path:
    """Stream block-FP8 → BF16 to ``dst_dir`` (1:1 shard names, optional multi-GPU).

    Each source ``*.safetensors`` becomes exactly one same-named file under
    ``dst_dir`` (no ``.uXXX`` splits). With ``num_gpus > 1``, shards are
    round-robin'd across CUDA devices; dequant runs on GPU then writes CPU.
    """
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()
    if src == dst:
        raise ValueError("unpack_fp8_checkpoint: src and dst must differ")
    marker = dst / "UNPACK_FP8_DONE.json"
    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing index under {src}")
    idx = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = dict(idx.get("weight_map") or {})
    cfg: Dict[str, Any] = {}
    cfg_path = src / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    g = int(group_size or fp8_group_size(cfg))

    by_shard: Dict[str, List[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)
    shards = sorted(by_shard.keys())

    if marker.is_file() and (dst / "model.safetensors.index.json").is_file():
        try:
            done = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            done = {}
        legacy_u = list(dst.glob("*.u*.safetensors"))
        one_to_one = bool(done.get("one_to_one_shards")) and all(
            (dst / sh).is_file() for sh in shards
        )
        if one_to_one and not legacy_u:
            logger.info("fp8 unpack: reuse existing staging {}", dst)
            return dst
        logger.warning(
            "fp8 unpack: staging {} is legacy/incomplete "
            "(one_to_one={}, legacy_u_parts={}); re-unpacking",
            dst,
            bool(done.get("one_to_one_shards")),
            len(legacy_u),
        )
        marker.unlink(missing_ok=True)
        for p in legacy_u:
            p.unlink(missing_ok=True)

    world_size = max(1, int(num_gpus or 1))
    use_cuda = False
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"unpack_fp8_checkpoint num_gpus={world_size} requires CUDA"
            )
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"Requested num_gpus={world_size} but only "
                f"{torch.cuda.device_count()} CUDA device(s) visible"
            )
        use_cuda = True
    elif torch.cuda.is_available():
        use_cuda = True
        world_size = 1

    dst.mkdir(parents=True, exist_ok=True)
    n_existing = len(list(dst.glob("*.safetensors")))
    if n_existing:
        logger.info(
            "fp8 unpack: resume staging {} ({} existing safetensors)",
            dst,
            n_existing,
        )

    for item in src.iterdir():
        if item.name.startswith(".") or item.suffix == ".safetensors":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if item.is_file():
            target = dst / item.name
            if not target.is_file():
                shutil.copy2(item, target)

    clear_rank_manifests(dst, max_ranks=max(world_size, 64))

    logger.info(
        "fp8 unpack: {} shards {} -> {} (group={}, gpus={}, cuda={})",
        len(shards),
        src,
        dst,
        g,
        world_size,
        use_cuda,
    )

    if world_size == 1:
        _unpack_fp8_worker(
            0,
            1,
            str(src),
            str(dst),
            shards,
            dict(by_shard),
            weight_map,
            g,
            use_cuda,
        )
    else:
        mp.spawn(
            _unpack_fp8_worker,
            args=(
                world_size,
                str(src),
                str(dst),
                shards,
                dict(by_shard),
                weight_map,
                g,
                use_cuda,
            ),
            nprocs=world_size,
            join=True,
        )

    new_wm, total_unpacked = load_and_merge_rank_manifests(
        dst, world_size, missing_label="fp8 unpack"
    )

    missing_files = [sh for sh in shards if not (dst / sh).is_file()]
    if missing_files:
        raise RuntimeError(
            "fp8 unpack incomplete: missing output shards "
            f"({len(missing_files)}), e.g. {missing_files[:5]}"
        )
    for sh in shards:
        expected = _expected_non_scale_keys(by_shard[sh])
        if not shard_keys_complete(dst / sh, expected):
            raise RuntimeError(f"fp8 unpack incomplete for shard {sh}")

    if cfg_path.is_file():
        cfg_out = dict(cfg)
        cfg_out.pop("quantization_config", None)
        cfg_out["deepseek_fp8_unpacked"] = True
        (dst / "config.json").write_text(
            json.dumps(cfg_out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    from preprocess.io import _safetensors_header_nbytes

    total = 0
    for sh in shards:
        p = dst / sh
        if p.is_file():
            total += _safetensors_header_nbytes(p)
    out_index = {
        "metadata": {
            **(idx.get("metadata") or {}),
            "total_size": total,
            "deepseek_fp8_unpack_1to1": True,
            "unpack_gpus": world_size,
        },
        "weight_map": new_wm,
    }
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(out_index, indent=2) + "\n", encoding="utf-8"
    )
    marker.write_text(
        json.dumps(
            {
                "status": "ok",
                "source": str(src),
                "unpacked_modules": total_unpacked,
                "group_size": g,
                "num_gpus": world_size,
                "shard_count": len(shards),
                "one_to_one_shards": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "fp8 unpack finished: {} modules, {} shards -> {} ({:.2f} GiB, gpus={})",
        total_unpacked,
        len(shards),
        dst,
        total / 1024**3,
        world_size,
    )
    return dst
