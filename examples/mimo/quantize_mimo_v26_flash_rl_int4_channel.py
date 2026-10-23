# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash-RL -> compressed-tensors channel INT4/INT8 + unchanged BF16.

参考 ../glm/quantize_glm5_3_int4_channel.py 的 RTN / pack-quantized 合约。
固定按源 dtype 转换：MXFP4 -> channel INT4 W4A8；FP8 -> channel INT8 W8A8；
BF16 / FP32 原样保留。旧 MTP 也遵循此规则，独立子模型原样复制。
两种整数权重均使用每输出通道 FP32 scale，动态 per-token INT8 激活。
FP4 / FP8 的解码仅是重新量化的中间步骤，不把非专家 FP8 最终保存为 BF16。

源 TP-interleaved fused QKV 按原 TP shard 分别解码，再输出 canonical split
q_proj/k_proj/v_proj；不能把源 FP8 scale 按整个拼接矩阵展开。
MXFP4 E2M1 + E8M0 是数值解码，不是把 U8 字节当作 INT4 整数。

python quantize_mimo_v26_flash_rl_int4_channel.py --dry-run
python quantize_mimo_v26_flash_rl_int4_channel.py \
    --input-dir /llm-models/XiaomiMiMo/MiMo-V2.6-Flash-RL \
    --output-dir /quant_models/MiMo-V2.6-Flash-RL-Channel-INT4-W4A8 --device cuda:0

--only-files / --max-tensors 仅生成局部冒烟权重，不写模型 config/index。
全量转换逐矩阵读取，输出按大小重新分片，不载入完整模型或完整源 shard。
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, OrderedDict
from dataclasses import dataclass
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


DEFAULT_INPUT = "/llm-models/XiaomiMiMo/MiMo-V2.6-Flash-RL"
DEFAULT_OUTPUT = "/quant_models/MiMo-V2.6-Flash-RL-Channel-INT4-W4A8"
EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
FP8_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E4M3FN", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}
TORCH_FP8_DTYPES = {getattr(torch, name) for name in (
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"
) if hasattr(torch, name)}
E2M1 = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)


@dataclass(frozen=True)
class Layout:
    config: dict
    source_tp: int
    moe_layers: tuple[int, ...]

    def classify(self, name: str, dtype: str | torch.dtype) -> str:
        match = EXPERT.fullmatch(name)
        if match:
            layer, expert = int(match[1]), int(match[2])
            if layer not in self.moe_layers or not 0 <= expert < self.config["n_routed_experts"]:
                raise ValueError(f"Unexpected routed expert: {name}")
        if dtype in {"U8", torch.uint8}:
            if not match or self.config.get("quantization_config", {}).get("store_dtype") != "mxfp4":
                raise ValueError(f"Unrecognized packed FP4 weight: {name}")
            return "w4a8"
        if dtype in FP8_DTYPES or dtype in TORCH_FP8_DTYPES:
            if not name.endswith(".weight"):
                raise ValueError(f"FP8 quantization requires a Linear weight: {name}")
            return "w8a8"
        return "keep"


def header(path: Path) -> dict:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        tensors = json.loads(stream.read(size))
    return {k: v for k, v in tensors.items() if k != "__metadata__"}


def load_source(input_dir: Path, source_tp: int | None = None):
    config = json.loads((input_dir / "config.json").read_text())
    if config.get("model_type") != "mimo_v2":
        raise ValueError("Expected model_type=mimo_v2")
    index = json.loads((input_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    meta = {}
    for filename in sorted(set(weight_map.values())):
        path = input_dir / filename
        if path.parent.resolve() != input_dir.resolve():
            raise ValueError(f"Expected top-level indexed shards, got {filename}")
        for name, tensor in header(path).items():
            if name in meta:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            if weight_map.get(name) != filename:
                raise ValueError(f"Index shard mismatch: {name}")
            meta[name] = tensor
    if set(meta) != set(weight_map):
        raise ValueError("Index and safetensors headers have different tensor names")
    actual_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in meta.values())
    if actual_bytes != index["metadata"]["total_size"]:
        raise ValueError("Source index total_size differs from actual tensor payloads")
    if any(name.endswith(".weight_packed") for name in meta):
        raise ValueError("Source is already packed INT4; expected MXFP4/FP8/BF16 source weights")
    pattern = config["moe_layer_freq"]
    if len(pattern) != config["num_hidden_layers"]:
        raise ValueError("moe_layer_freq does not match num_hidden_layers")
    tp = source_tp or int(index.get("metadata", {}).get("tp_size", 0))
    if config.get("attention_projection_layout", "split") == "fused_qkv" and not tp:
        raise ValueError("Fused QKV needs source index metadata.tp_size or --source-tp")
    layout = Layout(config, tp or 1, tuple(i for i, value in enumerate(pattern) if value))
    return layout, weight_map, meta


def runtime_module_name(module: str) -> str:
    """Mirror MiMoV2MTP.map_model_name_to_mtp_param_name, plus norm rename."""
    module = module.replace("pre_mlp_layernorm", "post_attention_layernorm")
    match = re.match(r"^model\.mtp\.layers\.\d+\.(.*)$", module)
    if match:
        suffix = match[1]
        prefix = "model." if suffix.split(".")[0] in {"eh_proj", "enorm", "hnorm", "final_layernorm"} else "model.mtp_block."
        return prefix + suffix
    return module


def module_aliases(module: str) -> set[str]:
    """Cover checkpoint names, loader names, and fused Linear construction prefixes."""
    aliases = {module, runtime_module_name(module)}
    for name in tuple(aliases):
        # MTP parameter names are mtp_block.*, but quant_config sees decoder.*.
        if name.startswith("model.mtp_block."):
            aliases.add(name.replace("model.mtp_block.", "model.decoder.", 1))
    for name in tuple(aliases):
        if re.search(r"\.self_attn\.[qkv]_proj$", name):
            aliases.add(re.sub(r"[qkv]_proj$", "qkv_proj", name))
        if re.search(r"\.mlp\.(?:gate|up)_proj$", name):
            aliases.add(re.sub(r"(?:gate|up)_proj$", "gate_up_proj", name))
    return aliases


def output_weight_names(name: str) -> list[str]:
    return ([name.replace("qkv_proj", part + "_proj") for part in ("q", "k", "v")]
            if name.endswith(".self_attn.qkv_proj.weight") else [name])


def compression_config(layout: Layout, meta: dict) -> dict:
    # Derive targets from actual source dtype, never a catch-all "Linear" rule.
    selected = {"w4a8": set(), "w8a8": set(), "keep": set()}
    modes = {}
    for name, tensor in meta.items():
        if not name.endswith(".weight") or len(tensor["shape"]) != 2:
            continue
        mode = layout.classify(name, tensor["dtype"])
        for output_name in output_weight_names(name):
            for alias in module_aliases(output_name.removesuffix(".weight")):
                if alias in modes and modes[alias] != mode:
                    raise ValueError(f"Fused/runtime module has conflicting source precisions: {alias}")
                modes[alias] = mode
                selected[mode].add(alias)
    groups = {}
    expert_names = [n for n in meta if EXPERT.fullmatch(n)]
    for mode, bits in (("w4a8", 4), ("w8a8", 8)):
        targets = sorted(selected[mode])
        # Compact the actual all-FP4 expert scope without broadening it to BF16 experts.
        if bits == 4 and expert_names and all(layout.classify(n, meta[n]["dtype"]) == mode for n in expert_names):
            ids = "|".join(map(str, layout.moe_layers))
            targets = [rf"re:^model\.layers\.(?:{ids})\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$"]
        if targets:
            groups["group_" + mode] = dict(
                targets=targets, format="pack-quantized" if bits == 4 else "int-quantized",
                weights=dict(num_bits=bits, type="int", strategy="channel", symmetric=True,
                             dynamic=False, group_size=None, actorder=None, block_structure=None,
                             observer="minmax", observer_kwargs={}),
                input_activations=dict(num_bits=8, type="int", strategy="token", symmetric=True,
                                       dynamic=True, group_size=None, observer="minmax", observer_kwargs={}),
                output_activations=None)
    return dict(config_groups=groups, format="pack-quantized", ignore=sorted(selected["keep"]),
                quant_method="compressed-tensors", quantization_status="compressed", kv_cache_scheme=None)


def _matches(module: str, target: str) -> bool:
    return bool(re.match(target[3:], module)) if target.startswith("re:") else module == target


def self_check_scope(layout: Layout, meta: dict) -> Counter:
    config = compression_config(layout, meta)
    stats = Counter()
    consumed_scales = set()
    expert_count = 0
    for name, tensor in meta.items():
        if name.endswith((".weight_scale", ".weight_scale_inv")):
            continue
        dtype, shape = tensor["dtype"], tensor["shape"]
        mode = layout.classify(name, dtype)
        if dtype in FP8_DTYPES:
            scale = name + "_scale_inv"
            if scale not in meta or len(shape) != 2:
                raise ValueError(f"FP8 weight needs a 2D matrix and block scale: {name}")
            bn, bk = layout.config["quantization_config"]["weight_block_size"]
            scale_rows = math.ceil(shape[0] / bn)
            if name.endswith(".self_attn.qkv_proj.weight"):
                rows = sum(qkv_sizes(name, layout))
                if shape != [rows * layout.source_tp, layout.config["hidden_size"]]:
                    raise ValueError(f"Invalid fused QKV shape: {name}")
                scale_rows = math.ceil(rows / bn) * layout.source_tp
            if (meta[scale]["dtype"] != "F32"
                    or meta[scale]["shape"] != [scale_rows, math.ceil(shape[1] / bk)]):
                raise ValueError(f"Invalid source FP8 block scale shape/dtype: {name}")
            consumed_scales.add(scale)
        if dtype == "U8":
            scale = name + "_scale"
            if not EXPERT.fullmatch(name) or scale not in meta:
                raise ValueError(f"Unrecognized packed U8 tensor: {name}")
            if layout.config.get("quantization_config", {}).get("store_dtype") != "mxfp4":
                raise ValueError("U8 expert decoding requires source store_dtype=mxfp4")
            n, half_k = shape
            block = int(layout.config["quantization_config"].get("mxfp4_block_size", 32))
            if meta[scale]["dtype"] != "U8" or meta[scale]["shape"] != [n, half_k * 2 // block]:
                raise ValueError(f"Invalid MXFP4 scale shape/dtype: {name}")
            shape = [n, half_k * 2]
            consumed_scales.add(scale)
        if EXPERT.fullmatch(name):
            expert_count += 1
            projection = EXPERT.fullmatch(name)[3]
            h, inter = layout.config["hidden_size"], layout.config["moe_intermediate_size"]
            expected = [h, inter] if projection == "down_proj" else [inter, h]
            if shape != expected:
                raise ValueError(f"Unexpected logical expert shape: {name}: {shape} != {expected}")
        if mode == "w4a8" and (len(shape) != 2 or shape[1] % 8):
            raise ValueError(f"INT4 requires a 2D matrix with K divisible by 8: {name}")
        if name.endswith(".weight") and len(shape) == 2:
            for output_name in output_weight_names(name):
                if name.endswith(".self_attn.qkv_proj.weight"):
                    qkv_sizes(name, layout)
                for module in module_aliases(output_name.removesuffix(".weight")):
                    ignored = module in config["ignore"]
                    found = [group for group in config["config_groups"].values()
                             if any(_matches(module, rule) for rule in group["targets"])]
                    expected = "keep" if ignored or not found else f"w{found[0]['weights']['num_bits']}a8"
                    if len(found) > 1 or expected != mode:
                        raise ValueError(f"Written scope/config mismatch: {name} -> {module}: {mode} != {expected}")
        stats[mode] += 1
    orphan_scales = {k for k in meta if k.endswith((".weight_scale", ".weight_scale_inv"))} - consumed_scales
    if orphan_scales:
        raise ValueError(f"Unconsumed source scales: {sorted(orphan_scales)[:3]}")
    expected_count = len(layout.moe_layers) * layout.config["n_routed_experts"] * 3
    if expert_count != expected_count:
        raise ValueError(f"Expert matrix count {expert_count} != expected {expected_count}")
    stats["source_scales_consumed"] = len(consumed_scales)
    return stats


def decode_mxfp4(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    if weight.dtype != torch.uint8 or scale.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError("MXFP4 requires U8 packed weight and U8 E8M0 scales")
    n, half_k = weight.shape
    if scale.shape != (n, half_k * 2 // block_size) or (half_k * 2) % block_size:
        raise ValueError("MXFP4 weight/scale shape mismatch")
    if (scale == 255).any():
        raise ValueError("Reserved/NaN E8M0 scale code 255")
    table = torch.tensor(E2M1, dtype=torch.float32, device=weight.device)
    low, high = weight & 15, weight >> 4
    values = torch.stack((table[low.long()], table[high.long()]), dim=-1).flatten(1)
    # ldexp handles E8M0 code 0 (2^-127) and large finite exponents explicitly.
    scales = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.int() - 127)
    result = values * scales.repeat_interleave(block_size, dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite decoded MXFP4 weights")
    return result


def decode_fp8(weight: torch.Tensor, scale: torch.Tensor, block_shape: tuple[int, int]) -> torch.Tensor:
    n, k = weight.shape
    bn, bk = block_shape
    if scale.shape != (math.ceil(n / bn), math.ceil(k / bk)):
        raise ValueError(f"FP8 block scale mismatch: weight={tuple(weight.shape)}, scale={tuple(scale.shape)}, block={block_shape}")
    return weight.float() * scale.float().repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:n, :k]


def qkv_sizes(name: str, layout: Layout) -> tuple[int, int, int]:
    cfg = layout.config
    match = re.match(r"^model\.layers\.(\d+)\.", name)
    swa = name.startswith("model.mtp.") or (match is not None and cfg["hybrid_layer_pattern"][int(match[1])] == 1)
    heads = cfg["swa_num_attention_heads"] if swa else cfg["num_attention_heads"]
    kv = cfg["swa_num_key_value_heads"] if swa else cfg["num_key_value_heads"]
    hd = cfg["swa_head_dim"] if swa else cfg["head_dim"]
    vd = cfg["swa_v_head_dim"] if swa else cfg["v_head_dim"]
    if heads % layout.source_tp or kv % layout.source_tp:
        raise ValueError("Source TP must evenly divide Q and KV heads; replicated-KV layouts are not supported")
    return heads // layout.source_tp * hd, kv // layout.source_tp * hd, kv // layout.source_tp * vd


def split_qkv(name: str, weight: torch.Tensor, scale: torch.Tensor | None, layout: Layout):
    """Decode each TP shard independently, then deinterleave into global Q/K/V."""
    sizes = qkv_sizes(name, layout)
    rows = sum(sizes)
    if weight.ndim != 2 or weight.shape != (rows * layout.source_tp, layout.config["hidden_size"]):
        raise ValueError(f"Invalid fused QKV shape: {name}: {tuple(weight.shape)}")
    blocks = tuple(layout.config.get("quantization_config", {}).get("weight_block_size", [128, 128]))
    scale_rows = math.ceil(rows / blocks[0])
    if scale is not None and scale.shape != (scale_rows * layout.source_tp, math.ceil(weight.shape[1] / blocks[1])):
        raise ValueError(f"Invalid TP-interleaved QKV scale: {name}: {tuple(scale.shape)}")
    components = [[], [], []]
    for rank in range(layout.source_tp):
        shard = weight[rank * rows:(rank + 1) * rows]
        if scale is not None:
            shard = decode_fp8(shard, scale[rank * scale_rows:(rank + 1) * scale_rows], blocks)
        else:
            shard = shard.float()
        for output, tensor in zip(components, shard.split(sizes, dim=0)):
            output.append(tensor)
    return dict(zip(("q", "k", "v"), (torch.cat(tensors, dim=0) for tensors in components)))


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """CT offset-binary, little-endian INT32 lanes, low nibble first."""
    if q.dtype != torch.int8 or q.ndim != 2 or q.shape[1] % 8:
        raise ValueError("pack_int4 expects int8 [N,K], K divisible by 8")
    lanes = (q.int() + 8).reshape(q.shape[0], -1, 8)
    shifts = torch.arange(8, device=q.device, dtype=torch.int32) * 4
    return (lanes << shifts).sum(-1, dtype=torch.int32).contiguous()


def quantize_int4(weight: torch.Tensor):
    weight = weight.float()
    if weight.ndim != 2 or weight.shape[1] % 8 or not torch.isfinite(weight).all():
        raise ValueError("INT4 needs a finite 2D matrix with K divisible by 8")
    maximum = weight.abs().amax(dim=1, keepdim=True)
    scale = maximum / 7.
    scale = torch.where(maximum == 0, torch.ones_like(scale), scale.clamp_min(torch.finfo(torch.float32).tiny))
    q = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
    return pack_int4(q), scale.contiguous(), torch.tensor(weight.shape, dtype=torch.int64)


def quantize_int8(weight: torch.Tensor):
    """Symmetric per-output-channel RTN; keep scales in FP32 on disk."""
    weight = weight.float()
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("INT8 needs a finite 2D matrix")
    maximum = weight.abs().amax(dim=1, keepdim=True)
    scale = maximum / 127.
    scale = torch.where(maximum == 0, torch.ones_like(scale), scale.clamp_min(torch.finfo(torch.float32).tiny))
    q = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


class TensorReader:
    """Bound open mmap headers; weight/scale partners may live in different shards."""
    def __init__(self, root: Path, weight_map: dict):
        self.root, self.weight_map = root, weight_map
        self.handles = OrderedDict()

    def get(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        if filename not in self.handles:
            if len(self.handles) >= 4:
                _, handle = self.handles.popitem(last=False)
                handle.__exit__(None, None, None)
            handle = safe_open(self.root / filename, framework="pt", device="cpu")
            handle.__enter__()
            self.handles[filename] = handle
        self.handles.move_to_end(filename)
        return self.handles[filename].get_tensor(name)

    def close(self):
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()


class ShardWriter:
    def __init__(self, root: Path, limit: int, partial: bool):
        self.root, self.limit = root, limit
        self.prefix = "partial" if partial else "model"
        self.pending, self.weight_map = {}, {}
        self.pending_bytes = self.total_bytes = self.count = 0

    def add(self, name: str, tensor: torch.Tensor):
        if name in self.pending or name in self.weight_map:
            raise ValueError(f"Duplicate output tensor: {name}")
        tensor = tensor.detach().to("cpu").contiguous().clone()
        size = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + size > self.limit:
            self.flush()
        self.pending[name] = tensor
        self.pending_bytes += size

    def flush(self):
        if not self.pending:
            return
        self.count += 1
        filename = f"{self.prefix}-{self.count:05d}.safetensors"
        temporary = self.root / (filename + ".tmp")
        save_file(self.pending, temporary, metadata={"format": "pt"})
        os.replace(temporary, self.root / filename)
        self.weight_map.update({name: filename for name in self.pending})
        self.total_bytes += self.pending_bytes
        self.pending, self.pending_bytes = {}, 0


def convert_tensor(name: str, reader: TensorReader, layout: Layout, device: torch.device):
    source = reader.get(name)
    mode = layout.classify(name, source.dtype)
    fp8_scale = name + "_scale_inv"
    if name.endswith(".self_attn.qkv_proj.weight"):
        scale = reader.get(fp8_scale).to(device) if fp8_scale in reader.weight_map else None
        matrices = [(name.replace("qkv_proj", part + "_proj"), value)
                    for part, value in split_qkv(name, source.to(device), scale, layout).items()]
    else:
        if source.dtype == torch.uint8:
            block = layout.config["quantization_config"].get("mxfp4_block_size", 32)
            weight = decode_mxfp4(source.to(device), reader.get(name + "_scale").to(device), block)
        elif fp8_scale in reader.weight_map:
            blocks = tuple(layout.config["quantization_config"]["weight_block_size"])
            weight = decode_fp8(source.to(device), reader.get(fp8_scale).to(device), blocks)
        elif mode != "keep":
            weight = source.to(device).float()
        else:
            return {name: source}, "kept"
        matrices = [(name, weight)]
    output = {}
    for output_name, weight in matrices:
        module = output_name.removesuffix(".weight")
        if mode == "keep":
            # Only an unquantized fused QKV reaches here; retain its original dtype.
            output[output_name] = weight.to(source.dtype)
        elif mode == "w8a8":
            quantized, scale = quantize_int8(weight)
            output.update({output_name: quantized, module + ".weight_scale": scale})
        else:
            packed, scale, shape = quantize_int4(weight)
            output.update({module + ".weight_packed": packed, module + ".weight_scale": scale,
                           module + ".weight_shape": shape})
    return output, mode


def copy_artifacts(source: Path, target: Path):
    # Independent submodels keep their own config/index/dtype; never use the main index for them.
    for path in source.iterdir():
        if path.name.startswith(".") or path.name in {"config.json", "model.safetensors.index.json"}:
            continue
        if path.is_file() and path.suffix not in {".safetensors", ".bin", ".pt", ".pth"}:
            shutil.copy2(path, target / path.name)
        elif path.is_dir() and path.name in {"assets", "audio_tokenizer", "dflash"}:
            shutil.copytree(path, target / path.name, ignore=shutil.ignore_patterns(".*", "__pycache__"))


def validate_output(writer: ShardWriter, expected_names: set[str]):
    if set(writer.weight_map) != expected_names:
        raise ValueError("Output tensor names differ from the planned conversion")
    found = {}
    total = 0
    for filename in sorted(set(writer.weight_map.values())):
        for name, tensor in header(writer.root / filename).items():
            if name in found or writer.weight_map.get(name) != filename:
                raise ValueError(f"Output shard/index mismatch: {name}")
            if name.endswith(".weight_packed") and tensor["dtype"] != "I32":
                raise ValueError(f"Wrong packed dtype: {name}")
            found[name] = tensor
            total += tensor["data_offsets"][1] - tensor["data_offsets"][0]
    if set(found) != expected_names or total != writer.total_bytes:
        raise ValueError("Output payload size/name validation failed")
    for name, tensor in found.items():
        if not name.endswith((".weight", ".weight_packed")) or tensor["dtype"] not in {"I8", "I32"}:
            continue
        module = name.rsplit(".", 1)[0]
        scale = found.get(module + ".weight_scale")
        if (scale is None or scale["dtype"] != "F32"
                or scale["shape"] != [tensor["shape"][0], 1]):
            raise ValueError(f"Missing/wrong channel scale for {name}")
        if name.endswith(".weight_packed"):
            shape = found.get(module + ".weight_shape")
            if shape is None or shape["dtype"] != "I64" or shape["shape"] != [2]:
                raise ValueError(f"Missing/wrong logical weight shape for {name}")


def plan_output(name: str, layout: Layout, dtype: str) -> list[str]:
    if name.endswith((".weight_scale", ".weight_scale_inv")):
        return []
    names = output_weight_names(name)
    mode = layout.classify(name, dtype)
    if mode == "keep":
        return names
    suffixes = (".weight", ".weight_scale") if mode == "w8a8" else (".weight_packed", ".weight_scale", ".weight_shape")
    return [n.removesuffix(".weight") + suffix for n in names for suffix in suffixes]


def convert(args):
    source, target = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if source == target or source in target.parents:
        raise ValueError("Output must be separate from and outside the source directory")
    layout, weight_map, meta = load_source(source, args.source_tp)
    stats = self_check_scope(layout, meta)
    print(f"Source TP={layout.source_tp}; MoE layers={len(layout.moe_layers)}; output QKV=split")
    print("Scope:", json.dumps(stats, sort_keys=True))
    print("Source dtype policy: MXFP4 -> channel INT4 W4A8; FP8 -> channel INT8 W8A8; BF16/FP32 unchanged (including old MTP)")
    if args.dry_run:
        return stats
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Output directory is not empty: {target}")
    files = {value.strip() for value in args.only_files.split(",") if value.strip()} if args.only_files else None
    if files and not files <= set(weight_map.values()):
        raise ValueError(f"--only-files contains unindexed shards: {sorted(files - set(weight_map.values()))}")
    names = [name for name in sorted(weight_map, key=lambda n: (weight_map[n], n))
             if not name.endswith((".weight_scale", ".weight_scale_inv"))
             and (files is None or weight_map[name] in files)]
    partial = bool(files or args.max_tensors is not None)
    if args.max_tensors is not None:
        names = names[:args.max_tensors]
    if not names:
        raise ValueError("No source tensors selected")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA/HIP is unavailable; use --device cpu or run with device access")
    torch.set_num_threads(args.num_threads)
    target.mkdir(parents=True, exist_ok=True)
    incomplete = target / ".incomplete"
    incomplete.write_text("Conversion is incomplete; do not serve this directory.\n")
    reader = TensorReader(source, weight_map)
    writer = ShardWriter(target, args.max_shard_size_mb * 1024**2, partial)
    converted = Counter()
    expected = set()
    try:
        with torch.inference_mode():
            for name in tqdm(names, desc=f"Converting on {device}"):
                output, kind = convert_tensor(name, reader, layout, device)
                for key, tensor in output.items():
                    writer.add(key, tensor)
                expected.update(plan_output(name, layout, meta[name]["dtype"]))
                converted[kind] += 1
                del output
        writer.flush()
        validate_output(writer, expected)
        if not partial:
            copy_artifacts(source, target)
            config = copy.deepcopy(layout.config)
            config.pop("quantization_config", None)
            config.pop("compression_config", None)
            config["quantization_config"] = compression_config(layout, meta)
            config["attention_projection_layout"] = "split"
            config["dtype"] = "bfloat16"
            if "torch_dtype" in config:
                config["torch_dtype"] = "bfloat16"
            (target / "model.safetensors.index.json").write_text(json.dumps(
                dict(metadata=dict(total_size=writer.total_bytes), weight_map=writer.weight_map), indent=2) + "\n")
            # Write config last: failed conversions must not look like a complete model.
            (target / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        incomplete.unlink()
    finally:
        reader.close()
    print(f"Done: {dict(converted)}; tensors={len(writer.weight_map)}; payload={writer.total_bytes / 1e9:.3f} GB")
    if partial:
        print("PARTIAL smoke output only: no model config/index or auxiliary models were written.")
    return converted


def main():
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0 (also HIP PyTorch)")
    parser.add_argument("--source-tp", type=int, help="Override the source fused-QKV TP layout; normally from source index")
    parser.add_argument("--max-shard-size-mb", type=int, default=2048)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--only-files", help="Comma-separated source shard names; smoke output only")
    parser.add_argument("--max-tensors", type=int, help="Maximum source tensors for smoke conversion; no config/index")
    parser.add_argument("--dry-run", action="store_true", help="Validate all headers and config scope; write nothing")
    args = parser.parse_args()
    if args.max_shard_size_mb <= 0 or args.num_threads <= 0 or (args.max_tensors is not None and args.max_tensors <= 0):
        parser.error("Shard size, thread count and --max-tensors must be positive")
    if args.source_tp is not None and args.source_tp <= 0:
        parser.error("--source-tp must be positive")
    convert(args)


if __name__ == "__main__":
    main()
