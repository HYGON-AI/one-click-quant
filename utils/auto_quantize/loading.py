"""Load floating, block FP8 and MXFP4 checkpoints into a floating analysis model.

Scale contracts follow examples/mimo/quant_mimo_original.py and the block-FP8
converters in examples/glm and examples/deepseek. No example runtime is imported.
"""

from collections import Counter, OrderedDict
import copy
import json
import math
from pathlib import Path
import re
import struct

import torch

from .model_load import get_model_loader


FP8_DTYPES = {getattr(torch, name) for name in (
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
) if hasattr(torch, name)}
FP8_HEADERS = {"F8_E4M3", "F8_E5M2", "F8_E4M3FN", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}
E2M1 = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)


class CheckpointReader:
    """Validate the shard index and read weight/scale partners independently."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        index = self.root / "model.safetensors.index.json"
        self.metadata = {}
        if index.is_file():
            data = json.loads(index.read_text())
            self.weight_map = data["weight_map"]
            self.metadata = data.get("metadata", {})
        elif (self.root / "model.safetensors").is_file():
            self.weight_map = None
        else:
            raise ValueError("Quantized decoding requires model.safetensors or its shard index")
        self.headers = {}
        filenames = sorted(set(self.weight_map.values())) if self.weight_map else ["model.safetensors"]
        for filename in filenames:
            path = self.root / filename
            if path.resolve().parent != self.root:
                raise ValueError(f"Shard must be inside the model directory: {filename}")
            with path.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                if size > path.stat().st_size - 8:
                    raise ValueError(f"Invalid safetensors header: {filename}")
                header = json.loads(stream.read(size))
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                if name in self.headers:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                if self.weight_map is not None and self.weight_map.get(name) != filename:
                    raise ValueError(f"Index/shard mismatch: {name}")
                self.headers[name] = spec
        if self.weight_map is None:
            self.weight_map = dict.fromkeys(self.headers, "model.safetensors")
        if set(self.headers) != set(self.weight_map):
            raise ValueError("Checkpoint index and shard headers have different tensor names")
        self.handles = OrderedDict()

    def get(self, name):
        from safetensors import safe_open

        filename = self.weight_map[name]
        if filename not in self.handles:
            if len(self.handles) >= 4:
                _, handle = self.handles.popitem(last=False)
                handle.__exit__(None, None, None)
            self.handles[filename] = safe_open(self.root / filename, framework="pt", device="cpu")
        self.handles.move_to_end(filename)
        return self.handles[filename].get_tensor(name)

    def close(self):
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()


def _scale_values(scale, *, exponent_bytes=False):
    if scale.dtype == torch.uint8:
        if not exponent_bytes:
            raise ValueError("U8 scale requires an explicit E8M0/UE8M0 encoding")
        if (scale == 255).any():
            raise ValueError("Reserved/NaN E8M0 scale code 255")
        values = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.int() - 127)
    elif scale.is_floating_point():
        values = scale.float()
    else:
        raise ValueError(f"Unsupported scale dtype: {scale.dtype}")
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Scales must be finite and positive")
    return values


def decode_mxfp4(weight, scale, block_size=32):
    """Low nibble first, E2M1 values, E8M0 per-K-group multipliers."""
    if weight.dtype != torch.uint8 or weight.ndim < 2 or block_size != 32:
        raise ValueError("MXFP4 requires U8 packed matrices and a 32-element block")
    k = weight.shape[-1] * 2
    if k % block_size or tuple(scale.shape) != (*weight.shape[:-1], k // block_size):
        raise ValueError("MXFP4 weight/scale shape mismatch")
    if scale.dtype not in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)):
        raise ValueError("MXFP4 requires E8M0 scales (U8 exponent bytes or float8_e8m0fnu)")
    table = torch.tensor(E2M1, dtype=torch.float32, device=weight.device)
    values = torch.stack((table[(weight & 15).long()], table[(weight >> 4).long()]), -1).flatten(-2)
    return values * _scale_values(scale, exponent_bytes=True).repeat_interleave(block_size, -1)


def decode_block_fp8(weight, scale, block_shape, *, exponent_bytes=False):
    """FP8 * block multiplier; `_scale_inv` is a multiplier in this source contract."""
    if weight.dtype not in FP8_DTYPES or weight.ndim < 2:
        raise ValueError("Block FP8 requires a floating FP8 matrix")
    if (not isinstance(block_shape, (list, tuple)) or len(block_shape) != 2
            or any(type(v) is not int or v <= 0 for v in block_shape)):
        raise ValueError("Block FP8 requires weight_block_size=[positive N block, positive K block]")
    bn, bk = block_shape
    n, k = weight.shape[-2:]
    if tuple(scale.shape) != (*weight.shape[:-2], math.ceil(n / bn), math.ceil(k / bk)):
        raise ValueError(f"FP8 block scale shape mismatch: weight={tuple(weight.shape)}, "
                         f"scale={tuple(scale.shape)}, block={block_shape}")
    values = _scale_values(scale, exponent_bytes=exponent_bytes)
    return weight.float() * values.repeat_interleave(bn, -2).repeat_interleave(bk, -1)[..., :n, :k]


class SourceDecoder:
    """Select numeric and layout decoding from source metadata and actual tensors."""

    def __init__(self, reader, source_config):
        self.reader = reader
        self.config = source_config
        self.model_loader = get_model_loader(source_config.get("model_type"))
        self.quant = (source_config.get("quantization_config")
                      or source_config.get("text_config", {}).get("quantization_config") or {})
        method = self.quant.get("quant_method")
        self.packed_mxfp4 = method == "compressed-tensors" and self.quant.get("format") == "mxfp4-pack-quantized"
        if self.packed_mxfp4:
            groups = self.quant.get("config_groups", {})
            if not groups or any(
                group.get("format") != "mxfp4-pack-quantized"
                or group.get("weights", {}).get("num_bits") != 4
                or group["weights"].get("group_size") != 32
                or group["weights"].get("type") != "float"
                for group in groups.values()
            ):
                raise ValueError("compressed-tensors MXFP4 requires 4-bit float weights with group_size=32")
        if method not in (None, "", "fp8", "mxfp4") and not self.packed_mxfp4:
            raise ValueError(f"Unsupported source quant_method={method!r}; supported: fp8, mxfp4, floating")
        if self.quant.get("store_dtype") not in (None, "fp8", "mxfp4"):
            raise ValueError(f"Unsupported source store_dtype={self.quant['store_dtype']!r}")
        self.consumed_scales = set()
        self.counts = Counter()

    def partner_names(self, name):
        if name.endswith(".weight_packed") and self.packed_mxfp4:
            return [key for key in (name.removesuffix("_packed") + "_scale",) if key in self.reader.headers]
        names = [name + "_scale_inv", name + "_scale"]
        if self.quant.get("scale_fmt") in ("ue8m0", "e8m0"):
            names.append(name.rsplit(".", 1)[0] + ".scale")
        return [key for key in names if key in self.reader.headers]

    def _partner(self, name):
        found = self.partner_names(name)
        if len(found) != 1:
            raise ValueError(f"{name}: expected one scale partner, found {found}")
        self.consumed_scales.add(found[0])
        return self.reader.get(found[0])

    def _decode(self, name, weight, scale=None):
        if weight.dtype in FP8_DTYPES:
            scale = self._partner(name) if scale is None else scale
            value = decode_block_fp8(weight, scale, self.quant.get("weight_block_size"),
                                     exponent_bytes=self.quant.get("scale_fmt") in ("e8m0", "ue8m0"))
            self.counts["block_fp8"] += 1
        elif weight.dtype == torch.uint8:
            if self.quant.get("store_dtype", self.quant.get("quant_method")) != "mxfp4" and not self.packed_mxfp4:
                raise ValueError(f"{name}: packed U8 weight requires store_dtype=mxfp4")
            value = decode_mxfp4(weight, self._partner(name), self.quant.get("mxfp4_block_size", 32))
            self.counts["mxfp4"] += 1
        else:
            if not weight.is_floating_point():
                raise ValueError(f"{name}: unsupported weight encoding {weight.dtype}")
            value = weight
            self.counts["floating"] += 1
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}: non-finite decoded weights")
        return value

    def decode(self, name):
        weight = self.reader.get(name)
        if self.model_loader is not None:
            return self.model_loader.decode_source_tensor(self, name, weight)
        return {name: self._decode(name, weight)}

    def validate_scales(self, ignored=(), scale_names=()):
        scales = {name for name in self.reader.headers if name.endswith((".weight_scale", ".weight_scale_inv"))}
        orphan = (scales | set(scale_names)) - self.consumed_scales - set(ignored)
        if orphan:
            raise ValueError(f"Unconsumed source weight scales: {sorted(orphan)[:5]}")


class DecodedWeights:
    """One CPU matrix (or fused QKV) cache, never a full decoded checkpoint."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.reader = runtime.reader
        self.decoder = SourceDecoder(self.reader, runtime.source_config)
        self.cached_key = None
        self.values = {}

    def get(self, name, shape, dtype):
        key = self.runtime.source_key(name)
        if key != self.cached_key:
            self.clear()
            self.values = self.decoder.decode(key)
            self.cached_key = key
        output_name = name if name in self.values else key
        value = self.values[output_name]
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"Source decoded shape mismatch: {name}: {tuple(value.shape)} != {shape}")
        value = value.to(dtype=dtype).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite source value after conversion: {name}")
        return value

    def clear(self):
        self.cached_key = None
        self.values.clear()


def load_model(model_path, *, device="cuda:0", dtype=torch.bfloat16, trust_remote_code=False):
    """Return (floating model, untouched source config, loading audit metadata).

    The full decoded model must fit on one device. Reading by tensor bounds decode
    temporaries; it does not implement EP, TP or lazy/offloaded analysis execution.
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModelForCausalLM
    from .scope import matches

    path = Path(model_path)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Analysis dtype must be float16, bfloat16 or float32")
    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code, local_files_only=True)
    source_config = copy.deepcopy(config.to_dict())
    model_loader = get_model_loader(config.model_type)
    quant = source_config.get("quantization_config") or {}
    has_safetensors = (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()
    reader = CheckpointReader(path) if has_safetensors else None
    has_low_precision = reader is not None and any(
        spec["dtype"] in FP8_HEADERS or (spec["dtype"] in ("U8", "I8", "I32") and len(spec["shape"]) >= 2)
        for spec in reader.headers.values())
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    if hasattr(config, "_experts_implementation"):
        config._experts_implementation = "eager"
    if not quant.get("quant_method") and not has_low_precision:
        if reader:
            reader.close()
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, torch_dtype=dtype, attn_implementation="eager",
            trust_remote_code=trust_remote_code, local_files_only=True,
        ).to(device).eval()
        return model, source_config, {"loader": "transformers-floating", "source_quant_method": None}
    if reader is None:
        raise ValueError("Quantized decoding requires safetensors checkpoint files")
    ignored = set()
    if model_loader is not None and hasattr(model_loader, "prepare_config"):
        ignored = model_loader.prepare_config(config, reader)
    try:
        decoder = SourceDecoder(reader, source_config)
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=dtype, attn_implementation="eager", trust_remote_code=trust_remote_code,
            )
        model.tie_weights()
        expected = dict(model.named_parameters(remove_duplicate=False))
        groups = {}
        for name, value in expected.items():
            groups.setdefault(id(value), []).append(name)
        expected.update(model.named_buffers(remove_duplicate=False))
        loaded = set()
        keep_fp32 = [*(getattr(model, "_keep_in_fp32_modules", None) or []),
                     *(getattr(model, "_keep_in_fp32_modules_strict", None) or [])]
        scale_names = {n for n in reader.headers if n.endswith((".weight_scale", ".weight_scale_inv"))}
        for name, spec in reader.headers.items():
            if spec["dtype"] in FP8_HEADERS or (spec["dtype"] == "U8" and len(spec["shape"]) >= 2):
                scale_names.update(decoder.partner_names(name))
        if decoder.quant.get("scale_fmt") in ("ue8m0", "e8m0"):
            scale_names |= {n for n in reader.headers if n.endswith(".scale") and n not in expected}
        for name in reader.headers:
            if name in scale_names or name in ignored:
                continue
            if model_loader is not None and hasattr(model_loader, "map_source_name"):
                target_name, is_fused = model_loader.map_source_name(name, source_config)
            else:
                target_name, is_fused = name, False
            if target_name not in expected and not is_fused:
                if any(re.fullmatch(rule, name) for rule in (getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])):
                    ignored.add(name)
                    continue
                raise ValueError(f"Checkpoint tensor has no model parameter/buffer: {name}")
            spec = reader.headers[name]
            is_matrix = len(spec["shape"]) >= 2
            values = decoder.decode(name) if spec["dtype"] in FP8_HEADERS or is_fused or is_matrix else {name: reader.get(name)}
            for output_name, value in values.items():
                output_name = target_name if output_name == name else output_name
                if output_name not in expected or tuple(value.shape) != tuple(expected[output_name].shape):
                    raise ValueError(f"Decoded shape/name mismatch: {name} -> {output_name}: {tuple(value.shape)}")
                if output_name in loaded:
                    raise ValueError(f"Multiple checkpoint tensors map to {output_name}")
                target_dtype = expected[output_name].dtype
                if ((value.dtype == torch.float32 and output_name.endswith(".e_score_correction_bias"))
                        or any(matches(output_name, p) or matches(output_name.rpartition(".")[0], p) for p in keep_fp32)):
                    target_dtype = torch.float32
                value = value.to(dtype=target_dtype)
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite value after conversion to {target_dtype}: {output_name}")
                set_module_tensor_to_device(model, output_name, device, value=value, dtype=target_dtype, clear_cache=False)
                loaded.add(output_name)
        # A tied checkpoint may serialize only one alias; preserve both identity and cost semantics.
        for names in groups.values():
            present = [n for n in names if n in loaded]
            if present:
                value = model.get_parameter(present[0])
                for name in present[1:]:
                    if not torch.equal(value, model.get_parameter(name)):
                        raise ValueError(f"Conflicting values for tied parameters: {names}")
                for name in set(names) - {present[0]}:
                    parent, _, attr = name.rpartition(".")
                    setattr(model.get_submodule(parent), attr, value)
                    loaded.add(name)
        missing = [n for n, t in expected.items() if n not in loaded and t.is_meta]
        if missing:
            raise ValueError(f"Missing materialized checkpoint parameters: {missing[:10]}")
        decoder.validate_scales(ignored, scale_names)
        if model_loader is not None and hasattr(model_loader, "finish_model"):
            model_loader.finish_model(model)
        for name, buffer in model.named_buffers():
            if buffer.is_meta:
                raise ValueError(f"Unmaterialized model buffer: {name}")
            set_module_tensor_to_device(model, name, device, value=buffer, clear_cache=False)
        audit = {"loader": "decoded-safetensors", "source_quant_method": quant.get("quant_method"),
                 "decoded_tensors": dict(decoder.counts), "consumed_scales": len(decoder.consumed_scales),
                 "source_tensors_skipped": sorted(ignored), "weight_block_size": quant.get("weight_block_size"),
                 "scale_semantics": "decoded_weight = stored_value * scale; MXFP4 low nibble first"}
        return model.eval(), source_config, audit
    finally:
        reader.close()

