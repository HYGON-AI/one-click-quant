import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union
from uuid import uuid4

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def _parse_size_to_bytes(value: str) -> int:
    s = value.strip().upper()
    if s.endswith("GB"):
        return int(float(s[:-2]) * (1024**3))
    if s.endswith("MB"):
        return int(float(s[:-2]) * (1024**2))
    if s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    return int(s)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _resolve_compute_device(device: str) -> torch.device:
    d = torch.device(device)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but --device is set to CUDA")
        if d.index is not None:
            torch.cuda.get_device_properties(d.index)
    return d


def _binary_op_to_bf16(
    left: torch.Tensor, right: torch.Tensor, op: str, compute_device: torch.device
) -> torch.Tensor:
    left_f = left.to(device=compute_device, dtype=torch.float32)
    right_f = right.to(device=compute_device, dtype=torch.float32)
    if op == "mul":
        out = left_f * right_f
    else:
        out = left_f / right_f
    return out.to(torch.bfloat16).to("cpu")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _copy_model_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.endswith(".safetensors"):
                continue
            if name == "model.safetensors.index.json":
                continue
            shutil.copy2(root_path / name, out_dir / name)


def _update_config_to_bf16(dst: Path) -> Tuple[Optional[Tuple[int, int]], dict[str, Any]]:
    config_path = dst / "config.json"
    config = _load_json(config_path)
    block_size = None
    qcfg = config.get("quantization_config")
    if isinstance(qcfg, dict):
        wbs = qcfg.get("weight_block_size")
        if (
            isinstance(wbs, list)
            and len(wbs) == 2
            and isinstance(wbs[0], int)
            and isinstance(wbs[1], int)
        ):
            block_size = (wbs[0], wbs[1])
        config.pop("quantization_config", None)
    config["torch_dtype"] = "bfloat16"
    _save_json(config_path, config)
    return block_size, config


def _resolve_shard_files(src: Path) -> list[Path]:
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        index = _load_json(index_path)
        weight_map = index.get("weight_map", {})
        files = sorted({src / v for v in weight_map.values()})
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"Missing shard file: {f}")
        return files

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {src}")
    return shards


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _cast_if_float_to_bf16(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point():
        return t.to(torch.bfloat16)
    return t


def _maybe_cast_non_weight_float_tensor(
    name: str,
    tensor: torch.Tensor,
    preserve_non_weight_float_dtype: bool,
    compute_device: torch.device,
) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor
    if preserve_non_weight_float_dtype and not name.endswith(".weight"):
        return tensor
    return tensor.to(device=compute_device, dtype=torch.bfloat16).to("cpu")


def _dequant_fp8_weight_to_bf16(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: Optional[Tuple[int, int]],
    scale_inv_op: str,
    scale_layout: str,
    compute_device: torch.device,
) -> torch.Tensor:
    if weight.ndim < 2:
        return _binary_op_to_bf16(weight, scale_inv, scale_inv_op, compute_device)

    out_dim = weight.shape[0]
    in_dim = int(weight.numel() // out_dim)
    w2d = weight.reshape(out_dim, in_dim)

    if scale_inv.numel() == 1:
        return _binary_op_to_bf16(
            w2d, scale_inv, scale_inv_op, compute_device
        ).reshape(weight.shape)

    if scale_inv.ndim == 1:
        if scale_inv.shape[0] == out_dim:
            return _binary_op_to_bf16(
                w2d, scale_inv.reshape(out_dim, 1), scale_inv_op, compute_device
            ).reshape(weight.shape)
        if scale_inv.shape[0] == in_dim:
            return _binary_op_to_bf16(
                w2d, scale_inv.reshape(1, in_dim), scale_inv_op, compute_device
            ).reshape(weight.shape)
        if block_size is not None:
            bs0, bs1 = block_size
            grid0 = _ceil_div(out_dim, bs0)
            grid1 = _ceil_div(in_dim, bs1)
            if scale_inv.shape[0] == grid0 * grid1:
                return _dequant_fp8_weight_to_bf16(
                    weight=weight,
                    scale_inv=scale_inv.reshape(grid0, grid1),
                    block_size=block_size,
                    scale_inv_op=scale_inv_op,
                    scale_layout=scale_layout,
                    compute_device=compute_device,
                )
            if scale_inv.shape[0] == grid0 and grid1 == 1:
                out_bf16 = torch.empty_like(w2d, dtype=torch.bfloat16, device=compute_device)
                scale_inv_f = scale_inv.to(device=compute_device, dtype=torch.float32)
                w2d_f = w2d.to(device=compute_device, dtype=torch.float32)
                for i in range(grid0):
                    r0 = i * bs0
                    r1 = min((i + 1) * bs0, out_dim)
                    s = scale_inv_f[i]
                    if scale_inv_op == "mul":
                        out_bf16[r0:r1, :] = (w2d_f[r0:r1, :] * s).to(torch.bfloat16)
                    else:
                        out_bf16[r0:r1, :] = (w2d_f[r0:r1, :] / s).to(torch.bfloat16)
                return out_bf16.to("cpu").reshape(weight.shape)
            if scale_inv.shape[0] == grid1 and grid0 == 1:
                out_bf16 = torch.empty_like(w2d, dtype=torch.bfloat16, device=compute_device)
                scale_inv_f = scale_inv.to(device=compute_device, dtype=torch.float32)
                w2d_f = w2d.to(device=compute_device, dtype=torch.float32)
                for j in range(grid1):
                    c0 = j * bs1
                    c1 = min((j + 1) * bs1, in_dim)
                    s = scale_inv_f[j]
                    if scale_inv_op == "mul":
                        out_bf16[:, c0:c1] = (w2d_f[:, c0:c1] * s).to(torch.bfloat16)
                    else:
                        out_bf16[:, c0:c1] = (w2d_f[:, c0:c1] / s).to(torch.bfloat16)
                return out_bf16.to("cpu").reshape(weight.shape)
        raise ValueError(
            f"Unsupported scale_inv shape {tuple(scale_inv.shape)} for weight shape "
            f"{tuple(weight.shape)}"
        )

    if scale_inv.ndim == 2:
        if scale_inv.shape == (out_dim, 1):
            return _binary_op_to_bf16(
                w2d, scale_inv.reshape(out_dim, 1), scale_inv_op, compute_device
            ).reshape(weight.shape)
        if scale_inv.shape == (1, in_dim):
            return _binary_op_to_bf16(
                w2d, scale_inv.reshape(1, in_dim), scale_inv_op, compute_device
            ).reshape(weight.shape)
        if scale_inv.shape == (out_dim, in_dim):
            return _binary_op_to_bf16(
                w2d, scale_inv, scale_inv_op, compute_device
            ).reshape(weight.shape)

        if block_size is None:
            raise ValueError(
                f"Got 2D scale_inv {tuple(scale_inv.shape)} but block_size is unknown"
            )

        bs0, bs1 = block_size
        grid0 = _ceil_div(out_dim, bs0)
        grid1 = _ceil_div(in_dim, bs1)

        if scale_inv.shape != (grid0, grid1):
            if (
                scale_inv.shape[0] >= grid0
                and scale_inv.shape[1] >= grid1
                and (scale_inv.shape[0] > grid0 or scale_inv.shape[1] > grid1)
            ):
                return _dequant_fp8_weight_to_bf16(
                    weight=weight,
                    scale_inv=scale_inv[:grid0, :grid1],
                    block_size=block_size,
                    scale_inv_op=scale_inv_op,
                    scale_layout=scale_layout,
                    compute_device=compute_device,
                )
            raise ValueError(
                f"Unsupported block scale_inv shape {tuple(scale_inv.shape)} for "
                f"weight shape {tuple(weight.shape)} with block_size={block_size}; "
                f"expected {(grid0, grid1)}"
            )

        if scale_layout == "in_out":
            scale_inv = scale_inv.transpose(0, 1).contiguous()
            if scale_inv.shape != (grid0, grid1):
                raise ValueError(
                    f"scale_layout=in_out produced scale_inv shape {tuple(scale_inv.shape)} "
                    f"for weight shape {tuple(weight.shape)} with block_size={block_size}; "
                    f"expected {(grid0, grid1)}"
                )

        out_bf16 = torch.empty_like(w2d, dtype=torch.bfloat16, device=compute_device)
        scale_inv_f = scale_inv.to(device=compute_device, dtype=torch.float32)
        w2d_f = w2d.to(device=compute_device, dtype=torch.float32)
        for i in range(grid0):
            r0 = i * bs0
            r1 = min((i + 1) * bs0, out_dim)
            for j in range(grid1):
                c0 = j * bs1
                c1 = min((j + 1) * bs1, in_dim)
                s = scale_inv_f[i, j]
                if scale_inv_op == "mul":
                    out_bf16[r0:r1, c0:c1] = (w2d_f[r0:r1, c0:c1] * s).to(torch.bfloat16)
                else:
                    out_bf16[r0:r1, c0:c1] = (w2d_f[r0:r1, c0:c1] / s).to(torch.bfloat16)
        return out_bf16.to("cpu").reshape(weight.shape)

    raise ValueError(
        f"Unsupported scale_inv ndim {scale_inv.ndim} for weight shape {tuple(weight.shape)}"
    )


@dataclass
class _ShardWriter:
    dst_dir: Path
    max_shard_bytes: int
    metadata: dict[str, str]
    shard_index: int = 0
    current_bytes: int = 0
    current_tensors: dict[str, torch.Tensor] = None
    shard_files: list[Path] = None
    weight_map: dict[str, str] = None

    def __post_init__(self) -> None:
        self.current_tensors = {}
        self.shard_files = []
        self.weight_map = {}

    def _new_shard_path(self) -> Path:
        return self.dst_dir / f"model_{self.shard_index:05d}.safetensors"

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor_bytes = _tensor_nbytes(tensor)
        if self.current_tensors and self.current_bytes + tensor_bytes > self.max_shard_bytes:
            self.flush()
        self.current_tensors[name] = tensor
        self.current_bytes += tensor_bytes

    def flush(self) -> None:
        if not self.current_tensors:
            return
        shard_path = self._new_shard_path()
        save_file(self.current_tensors, str(shard_path), metadata={**self.metadata, "format": "pt"})
        self.shard_files.append(shard_path)
        shard_filename = shard_path.name
        for k in self.current_tensors.keys():
            self.weight_map[k] = shard_filename
        self.shard_index += 1
        self.current_tensors = {}
        self.current_bytes = 0


def _build_output_index(dst: Path, shard_files: Iterable[Path], weight_map: dict[str, str]) -> None:
    total_size = 0
    for f in shard_files:
        total_size += f.stat().st_size
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    _save_json(dst / "model.safetensors.index.json", index)


def _process_one_input_shard(
    shard_path: str,
    dst_dir: str,
    block_size: Optional[Tuple[int, int]],
    preserve_non_weight_float_dtype: bool,
    scale_inv_crop_ok: bool,
    scale_inv_op: str,
    scale_layout: str,
    compute_device: str,
) -> tuple[int, dict[str, str]]:
    shard_path_p = Path(shard_path)
    dst_dir_p = Path(dst_dir)
    out_path = dst_dir_p / shard_path_p.name

    out_tensors: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}

    device = _resolve_compute_device(compute_device)

    with safe_open(str(shard_path_p), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        key_set = set(keys)
        for key in keys:
            if key.endswith(".weight_scale_inv"):
                continue

            if key.endswith(".weight"):
                scale_key = f"{key}_scale_inv"
                if scale_key in key_set:
                    w = f.get_tensor(key)
                    s_inv = f.get_tensor(scale_key)
                    out = _dequant_fp8_weight_to_bf16(
                        w,
                        s_inv,
                        block_size,
                        scale_inv_op=scale_inv_op,
                        scale_layout=scale_layout,
                        compute_device=device,
                    )
                    out_tensors[key] = out
                    weight_map[key] = out_path.name
                    continue

            t = f.get_tensor(key)
            out_tensors[key] = _maybe_cast_non_weight_float_tensor(
                key, t, preserve_non_weight_float_dtype, device
            )
            weight_map[key] = out_path.name

    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{uuid4().hex}")
    save_file(out_tensors, str(tmp_path), metadata={"format": "pt", "converted_from": str(shard_path_p)})
    os.replace(tmp_path, out_path)
    return int(out_path.stat().st_size), weight_map


def convert_fp8_to_bf16(
    src: Union[str, os.PathLike],
    dst: Union[str, os.PathLike],
    max_shard_size: str = "5GB",
    block_size_override: Optional[Tuple[int, int]] = None,
    preserve_non_weight_float_dtype: bool = False,
    workers: int = 1,
    executor: str = "process",
    preserve_shard_structure: bool = True,
    scale_inv_op: str = "mul",
    scale_layout: str = "out_in",
    device: str = "cpu",
) -> None:
    src_dir = Path(src)
    dst_dir = Path(dst)
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    _copy_model_files(src_dir, dst_dir)
    block_size_from_config, _ = _update_config_to_bf16(dst_dir)
    block_size = block_size_override or block_size_from_config
    compute_device = _resolve_compute_device(device)

    shard_files = _resolve_shard_files(src_dir)
    total_shards = len(shard_files)

    if preserve_shard_structure:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        executor = executor.lower().strip()
        if executor not in ("process", "thread"):
            raise ValueError("executor must be one of: process, thread")

        total_size = 0
        weight_map: dict[str, str] = {}
        shard_out_files: set[Path] = set()

        if compute_device.type == "cuda" and workers > 1:
            raise ValueError(
                "GPU mode currently requires --workers 1 to avoid GPU OOM/contention. "
                "Please rerun with --workers 1, or use --device cpu for multi-worker conversion."
            )

        if workers == 1:
            for idx, p in enumerate(shard_files):
                shard_size, shard_weight_map = _process_one_input_shard(
                    str(p),
                    str(dst_dir),
                    block_size,
                    preserve_non_weight_float_dtype,
                    True,
                    scale_inv_op,
                    scale_layout,
                    str(compute_device),
                )
                total_size += shard_size
                weight_map.update(shard_weight_map)
                shard_out_files.add(dst_dir / p.name)
                pct = (idx + 1) * 100 // total_shards
                logger.info("fp8_to_bf16 progress: %d/%d (%d%%)", idx + 1, total_shards, pct)
            _build_output_index(dst_dir, sorted(shard_out_files), weight_map)
            return

        exec_cls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
        with exec_cls(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _process_one_input_shard,
                    str(p),
                    str(dst_dir),
                    block_size,
                    preserve_non_weight_float_dtype,
                    True,
                    scale_inv_op,
                    scale_layout,
                    str(compute_device),
                )
                for p in shard_files
            ]
            for fut in as_completed(futures):
                shard_size, shard_weight_map = fut.result()
                total_size += shard_size
                weight_map.update(shard_weight_map)
                any_file = next(iter(shard_weight_map.values()))
                shard_out_files.add(dst_dir / any_file)
                done = len(shard_out_files)
                pct = done * 100 // total_shards
                logger.info("fp8_to_bf16 progress: %d/%d (%d%%)", done, total_shards, pct)

        _build_output_index(dst_dir, sorted(shard_out_files), weight_map)
        return

    max_shard_bytes = _parse_size_to_bytes(max_shard_size)
    metadata = {"converted_from": str(src_dir), "conversion_id": str(uuid4())}
    writer = _ShardWriter(dst_dir=dst_dir, max_shard_bytes=max_shard_bytes, metadata=metadata)

    for idx, shard_path in enumerate(shard_files):
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            key_set = set(keys)
            for key in keys:
                if key.endswith(".weight_scale_inv"):
                    continue

                if key.endswith(".weight"):
                    scale_key = f"{key}_scale_inv"
                    if scale_key in key_set:
                        w = f.get_tensor(key)
                        s_inv = f.get_tensor(scale_key)
                        out = _dequant_fp8_weight_to_bf16(
                            w,
                            s_inv,
                            block_size,
                            scale_inv_op=scale_inv_op,
                            scale_layout=scale_layout,
                            compute_device=compute_device,
                        )
                        writer.add(key, out)
                        continue

                t = f.get_tensor(key)
                writer.add(
                    key,
                    _maybe_cast_non_weight_float_tensor(
                        key, t, preserve_non_weight_float_dtype, compute_device
                    ),
                )

        pct = (idx + 1) * 100 // total_shards
        logger.info("fp8_to_bf16 progress: %d/%d (%d%%)", idx + 1, total_shards, pct)

    writer.flush()
    _build_output_index(dst_dir, writer.shard_files, writer.weight_map)


def _parse_block_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError("block_size must be like '128,128'")
    return int(parts[0]), int(parts[1])


def run(args: argparse.Namespace) -> None:
    logger.info("fp8_to_bf16 converting %s", args.model)
    src = args.model
    dst = args.save_dir or f"{src}-TO-BF16"
    convert_fp8_to_bf16(
        src=src,
        dst=dst,
        max_shard_size="5GB",
        block_size_override=None,
        preserve_non_weight_float_dtype=False,
        workers=1,
        executor="process",
        preserve_shard_structure=True,
        scale_inv_op="mul",
        scale_layout="out_in",
        device="cpu",
    )
    logger.info("fp8_to_bf16 done, output: %s", dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source model directory (FP8 checkpoint)")
    parser.add_argument("--dst", required=True, help="Destination directory (BF16 checkpoint)")
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Max output shard size, e.g. 2GB, 5GB, 800MB",
    )
    parser.add_argument(
        "--block-size",
        default=None,
        help="Override weight block size, e.g. 128,128. If omitted, read from config.json",
    )
    parser.add_argument(
        "--preserve-non-weight-float-dtype",
        action="store_true",
        help=(
            "Do not cast non-.weight floating tensors (e.g. constants/statistics) to BF16. "
            "Only weights are guaranteed to be BF16 in the output."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers to convert input shards. Effective only with --preserve-shard-structure.",
    )
    parser.add_argument(
        "--executor",
        default="process",
        choices=["process", "thread"],
        help="Parallelism backend used with --workers. process is recommended for CPU parallelism.",
    )
    parser.add_argument(
        "--preserve-shard-structure",
        action="store_true",
        help=(
            "Write one output .safetensors per input shard (fastest, parallelizable). "
            "When disabled, outputs are reshared by --max-shard-size (single-threaded)."
        ),
    )
    parser.add_argument(
        "--scale-inv-op",
        default="mul",
        choices=["mul", "div"],
        help="Dequant operation: W = fp8 * weight_scale_inv (mul) or W = fp8 / weight_scale_inv (div).",
    )
    parser.add_argument(
        "--scale-layout",
        default="out_in",
        choices=["out_in", "in_out"],
        help="Block scale layout: (out_blocks, in_blocks) or transposed (in_blocks, out_blocks).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device for dequantization, e.g. cpu, cuda, cuda:0. Output tensors are always saved on CPU.",
    )
    args = parser.parse_args()
    convert_fp8_to_bf16(
        src=args.src,
        dst=args.dst,
        max_shard_size=args.max_shard_size,
        block_size_override=_parse_block_size(args.block_size),
        preserve_non_weight_float_dtype=args.preserve_non_weight_float_dtype,
        workers=args.workers,
        executor=args.executor,
        preserve_shard_structure=args.preserve_shard_structure,
        scale_inv_op=args.scale_inv_op,
        scale_layout=args.scale_layout,
        device=args.device,
    )


if __name__ == "__main__":
    main()
