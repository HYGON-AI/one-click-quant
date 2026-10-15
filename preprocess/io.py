"""Lazy safetensors weight store for preprocess transforms."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

_NON_WEIGHT_EXTS = {".json", ".py", ".txt", ".jinja", ".model", ".tiktoken", ".md"}

_DTYPE_NBYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def _safetensors_header_nbytes(path: Path) -> int:
    """Sum tensor storage bytes from a safetensors file header (no full load)."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
    total = 0
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        dtype = info.get("dtype")
        shape = info.get("shape") or []
        n = 1
        for d in shape:
            n *= int(d)
        total += n * int(_DTYPE_NBYTES.get(dtype, 0))
    return total


class WeightStore:
    """Load/modify/save HF safetensors checkpoints without loading the whole model."""

    def __init__(self, model_dir: str | os.PathLike):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        single = self.model_dir / "model.safetensors"
        if index_path.is_file():
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map: Dict[str, str] = dict(idx.get("weight_map", {}))
            self.metadata = dict(idx.get("metadata", {}))
            self._single_file: Optional[str] = None
        elif single.is_file():
            with safe_open(single, framework="pt", device="cpu") as f:
                keys = list(f.keys())
            self.weight_map = {k: "model.safetensors" for k in keys}
            self.metadata = {}
            self._single_file = "model.safetensors"
        else:
            raise FileNotFoundError(f"No safetensors index/model under {model_dir}")

        self._cache: Dict[str, torch.Tensor] = {}
        self._dirty: Set[str] = set()
        self._deleted: Set[str] = set()
        self._rewrite_shards: Set[str] = set()
        # Optional disk spill for MoE-scale dirty sets (DeepSeek-V3 etc.).
        self._spill_dir: Optional[Path] = None
        self._max_cache_bytes: Optional[int] = None
        self._cache_bytes: int = 0
        self._spill_files: List[Path] = []
        self._spill_index: Dict[str, Path] = {}
        self._spill_refs: Dict[Path, int] = {}
        self._spill_seq: int = 0

    def enable_spill(
        self,
        spill_dir: str | os.PathLike,
        *,
        max_cache_bytes: int = 48 * 1024**3,
    ) -> None:
        """Cap in-RAM dirty/clean cache; overflow dirty tensors to safetensors chunks.

        DeepSeek-V3 QuaRot/AWQ can dirty tens of thousands of expert BF16 weights
        per rank; without spill, workers die with SIGKILL (~host OOM).
        Superseded spill chunks are unlinked when their refcount hits zero so
        fuse→rotate→AWQ rewrites do not multiply disk usage.

        Chunk filenames use a monotonic ``_spill_seq`` counter (never
        ``len(_spill_files)``): unlinking old chunks shrinks the live list, and
        reusing that length would overwrite a still-referenced file and corrupt
        ``_spill_index`` (missing-tensor SafetensorError on reload).
        """
        self._spill_dir = Path(spill_dir)
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        self._max_cache_bytes = int(max_cache_bytes)
        self._spill_seq = 0
        logger.info(
            "WeightStore spill enabled: dir={} max_cache={:.2f} GiB",
            self._spill_dir,
            self._max_cache_bytes / 1024**3,
        )

    def _release_spill_key(self, name: str) -> None:
        path = self._spill_index.pop(name, None)
        if path is None:
            return
        left = self._spill_refs.get(path, 1) - 1
        if left <= 0:
            self._spill_refs.pop(path, None)
            if path in self._spill_files:
                self._spill_files = [p for p in self._spill_files if p != path]
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("failed to unlink spill chunk {}: {}", path, exc)
        else:
            self._spill_refs[path] = left

    def names(self) -> List[str]:
        return list(self.weight_map.keys())

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def weight_key(self, module: str) -> Optional[str]:
        """Resolve module path to a weight tensor name if present.

        Tries ``{module}.weight`` first, then bare ``{module}`` (fused MoE
        packs such as ``...experts.gate_up_proj`` without a ``.weight`` suffix).
        """
        for cand in (f"{module}.weight", module):
            if cand and self.has(cand):
                return cand
        return None

    def _nbytes(self, t: torch.Tensor) -> int:
        return int(t.numel() * t.element_size())

    def _drop_cache(self, name: str) -> None:
        t = self._cache.pop(name, None)
        if t is not None:
            self._cache_bytes = max(0, self._cache_bytes - self._nbytes(t))

    def _put_cache(self, name: str, tensor: torch.Tensor) -> None:
        old = self._cache.get(name)
        if old is not None:
            self._cache_bytes = max(0, self._cache_bytes - self._nbytes(old))
        self._cache[name] = tensor
        self._cache_bytes += self._nbytes(tensor)

    def _load_spilled(self, name: str) -> torch.Tensor:
        path = self._spill_index[name]
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def _resolve_tensor(self, name: str) -> Optional[torch.Tensor]:
        """Prefer cache, then spill (dirty), else None (caller reads shard)."""
        if name in self._cache:
            return self._cache[name]
        if name in self._spill_index:
            return self._load_spilled(name)
        return None

    def maybe_spill(self, *, force: bool = False) -> int:
        """Write cached dirty tensors to a spill chunk; return #tensors spilled."""
        if self._spill_dir is None or self._max_cache_bytes is None:
            return 0
        if not force and self._cache_bytes < self._max_cache_bytes:
            # Still evict clean cached tensors if over budget.
            self._evict_clean()
            return 0
        payload: Dict[str, torch.Tensor] = {}
        for name in list(self._dirty):
            t = self._cache.get(name)
            if t is None:
                continue
            payload[name] = t.detach().cpu().contiguous()
            self._drop_cache(name)
        if not payload:
            self._evict_clean()
            return 0
        path = self._spill_dir / f"chunk_{self._spill_seq:05d}.safetensors"
        self._spill_seq += 1
        if path.exists():
            # Should not happen with monotonic seq; refuse overwrite rather than
            # silently corrupt tensors still mapped to this path.
            raise FileExistsError(f"spill chunk already exists: {path}")
        save_file(payload, str(path))
        self._spill_files.append(path)
        self._spill_refs[path] = 0
        for name in payload:
            self._release_spill_key(name)
            self._spill_index[name] = path
            self._spill_refs[path] = self._spill_refs.get(path, 0) + 1
        logger.info(
            "WeightStore spilled {} tensors -> {} (cache≈{:.2f} GiB, live_chunks={})",
            len(payload),
            path.name,
            self._cache_bytes / 1024**3,
            len(self._spill_files),
        )
        self._evict_clean()
        return len(payload)

    def _evict_clean(self) -> None:
        if self._max_cache_bytes is None:
            return
        while self._cache_bytes > self._max_cache_bytes and self._cache:
            # Prefer dropping non-dirty entries.
            victim = next((n for n in self._cache if n not in self._dirty), None)
            if victim is None:
                # All dirty and still over budget — force spill again if possible.
                if self._spill_dir is not None and any(n in self._cache for n in self._dirty):
                    # Avoid recursion loops: only spill when force path already ran.
                    break
                break
            self._drop_cache(victim)

    def get(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        if name in self._cache:
            t = self._cache[name]
            return t if str(t.device) == str(device) else t.to(device)
        # Dirty-but-spilled must not fall through to the original shard.
        if name in self._spill_index:
            t = self._load_spilled(name)
            return t.to(device) if str(device) != "cpu" else t
        shard = self.weight_map[name]
        path = self.model_dir / shard
        with safe_open(path, framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        self._put_cache(name, t)
        self.maybe_spill()
        return t.to(device) if str(device) != "cpu" else t

    def set(self, name: str, tensor: torch.Tensor, *, shard: Optional[str] = None) -> None:
        if name not in self.weight_map:
            if shard is None:
                shard = next(iter(set(self.weight_map.values()) or ["model.safetensors"]))
            self.weight_map[name] = shard
        cpu = tensor.detach().cpu().contiguous()
        self._put_cache(name, cpu)
        self._dirty.add(name)
        self._deleted.discard(name)
        self._rewrite_shards.add(self.weight_map[name])
        # New value supersedes any older spill entry (may unlink orphan chunks).
        self._release_spill_key(name)
        self.maybe_spill()

    def delete(self, name: str) -> None:
        """Drop a tensor from the checkpoint (e.g. MXFP4 ``weight_packed``)."""
        shard = self.weight_map.pop(name, None)
        if shard is not None:
            self._rewrite_shards.add(shard)
        self._drop_cache(name)
        self._release_spill_key(name)
        self._dirty.discard(name)
        self._deleted.add(name)

    def export_dirty(self) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of tensors marked dirty (for multi-process merge).

        Warning: materializes the full dirty set in RAM. Prefer
        ``export_spill_manifest`` for MoE-scale checkpoints.
        """
        out: Dict[str, torch.Tensor] = {}
        for name in sorted(self._dirty):
            t = self._resolve_tensor(name)
            if t is None:
                t = self.get(name, device="cpu")
            out[name] = t.detach().cpu().contiguous()
        return out

    def export_spill_manifest(self) -> dict:
        """Force-spill remaining dirty cache; return paths for parent merge."""
        self.maybe_spill(force=True)
        live = [p for p in self._spill_files if p.is_file()]
        self._spill_files = live
        return {
            "spill_files": [str(p) for p in live],
            "deleted": self.export_deleted(),
            "dirty_count": len(self._dirty),
        }

    def export_deleted(self) -> List[str]:
        return sorted(self._deleted)

    def matching(self, substr: str) -> List[str]:
        return [n for n in self.weight_map if substr in n]

    def save(self, output_dir: str | os.PathLike, copy_non_weights: bool = True) -> None:
        out = Path(output_dir).resolve()
        src_root = self.model_dir.resolve()
        if out == src_root:
            raise ValueError(
                f"Refuse to overwrite model_dir in place: {out}. "
                "Pass a different --save-dir."
            )
        out.mkdir(parents=True, exist_ok=True)

        if copy_non_weights:
            self._copy_sidecar_files(out)

        dirty_shards = set(self._rewrite_shards) | {
            self.weight_map[n] for n in self._dirty if n in self.weight_map
        }
        # Shards present in the output index (after deletes / new weights).
        out_shards = set(self.weight_map.values()) | dirty_shards

        if not dirty_shards:
            logger.warning("WeightStore.save: no dirty tensors; copying all shards as-is")

        for shard in sorted(out_shards):
            dest = out / shard
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = self.model_dir / shard
            if shard not in dirty_shards:
                if src.is_file():
                    # Same filesystem: hardlink avoids a second full copy of
                    # DeepSeek-scale BF16 staging (~1.3 TiB) onto /module.
                    if dest.exists():
                        dest.unlink()
                    try:
                        os.link(src, dest)
                    except OSError:
                        shutil.copy2(src, dest)
                continue
            keys = [k for k, s in self.weight_map.items() if s == shard]
            tensors: Dict[str, torch.Tensor] = {}
            if src.is_file():
                with safe_open(src, framework="pt", device="cpu") as f:
                    src_keys = set(f.keys())
                    for k in keys:
                        resolved = self._resolve_tensor(k) if k in self._dirty else None
                        if resolved is not None:
                            tensors[k] = resolved.cpu().contiguous()
                        elif k in src_keys:
                            tensors[k] = f.get_tensor(k)
            for k in keys:
                if k not in tensors:
                    resolved = self._resolve_tensor(k)
                    if resolved is not None:
                        tensors[k] = resolved.cpu().contiguous()
            save_file(tensors, str(dest))
            logger.info("rewrote {} ({} tensors, dirty)", dest.name, len(tensors))
            # Free rewritten dirty tensors from RAM after the shard is on disk.
            for k in keys:
                if k in self._dirty:
                    self._drop_cache(k)

        # total_size from written shard headers (cheap; no full tensor load).
        total = 0
        for shard in sorted(out_shards):
            dest = out / shard
            if dest.is_file():
                total += _safetensors_header_nbytes(dest)

        index = {
            "metadata": {**(self.metadata or {}), "total_size": total},
            "weight_map": self.weight_map,
        }
        (out / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("saved preprocess checkpoint to {} (total_size={})", out, total)

    def commit_inplace_and_move(self, output_dir: str | os.PathLike) -> Path:
        """Rewrite dirty shards inside ``model_dir``, then rename it to ``output_dir``.

        DeepSeek-scale BF16 staging (~1.3 TiB) cannot be duplicated on a 3.5 TiB
        ``/module`` volume together with spill; this path only needs ~1 shard of
        temporary space, then ``rename``s the staging directory into place.
        """
        out = Path(output_dir).resolve()
        src = self.model_dir.resolve()
        if out == src:
            raise ValueError(f"output_dir must differ from model_dir: {out}")
        if out.exists():
            shutil.rmtree(out)

        dirty_shards = set(self._rewrite_shards) | {
            self.weight_map[n] for n in self._dirty if n in self.weight_map
        }
        if not dirty_shards:
            logger.warning("commit_inplace_and_move: no dirty shards; renaming as-is")

        for shard in sorted(dirty_shards):
            keys = [k for k, s in self.weight_map.items() if s == shard]
            tensors: Dict[str, torch.Tensor] = {}
            src_path = src / shard
            if src_path.is_file():
                with safe_open(src_path, framework="pt", device="cpu") as f:
                    src_keys = set(f.keys())
                    for k in keys:
                        resolved = self._resolve_tensor(k) if k in self._dirty else None
                        if resolved is not None:
                            tensors[k] = resolved.cpu().contiguous()
                        elif k in src_keys:
                            tensors[k] = f.get_tensor(k)
            for k in keys:
                if k not in tensors:
                    resolved = self._resolve_tensor(k)
                    if resolved is not None:
                        tensors[k] = resolved.cpu().contiguous()
            tmp = src / f".tmp.{Path(shard).name}"
            save_file(tensors, str(tmp))
            tmp.replace(src_path)
            for k in keys:
                if k in self._dirty:
                    self._drop_cache(k)
                    self._release_spill_key(k)
            logger.info("inplace rewrote {} ({} tensors)", src_path.name, len(tensors))

        out_shards = set(self.weight_map.values())
        total = 0
        for shard in sorted(out_shards):
            p = src / shard
            if p.is_file():
                total += _safetensors_header_nbytes(p)
        index = {
            "metadata": {**(self.metadata or {}), "total_size": total},
            "weight_map": self.weight_map,
        }
        (src / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
        for marker in ("UNPACK_FP8_DONE.json", "UNPACK_MXFP4_DONE.json"):
            mp = src / marker
            if mp.is_file():
                mp.unlink()

        src.rename(out)
        self.model_dir = out
        logger.info(
            "commit_inplace_and_move: {} -> {} (total_size={})", src, out, total
        )
        return out

    def _copy_sidecar_files(self, out: Path) -> None:
        for item in self.model_dir.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_dir():
                if item.name in ("optional",):
                    shutil.copytree(item, out / item.name, dirs_exist_ok=True)
                continue
            if item.suffix == ".safetensors":
                continue
            if item.name == "model.safetensors.index.json":
                continue
            if item.suffix in _NON_WEIGHT_EXTS or item.name in (
                "tokenizer.model",
                "tiktoken.model",
            ):
                shutil.copy2(item, out / item.name)


def load_hf_config(model_dir: str | os.PathLike) -> dict:
    path = Path(model_dir) / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))
