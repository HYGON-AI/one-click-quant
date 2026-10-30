"""MiMo text runtime: rank-local experts, source decoding and one-block residency."""

from contextlib import ExitStack
import copy
import importlib.util
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

from ..api import _inventory
from ..ep_comm import install_ep_forward
from ..ep_modelopt import local_modelopt_state
from ..loading import CheckpointReader, SourceDecoder, DecodedWeights, FP8_DTYPES
from ..scope import ModelAdapter, discover
from ..streaming import attach_source_offload, attach_block_residency, register_gradient_support


EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


class MiMoScopeAdapter(ModelAdapter):
    grouping_rules = (r"^(model\.layers\.\d+\.mlp\.experts)\.\d+\.(gate_proj|up_proj|down_proj)$",)

    def group_name(self, names):
        matches = [EXPERT.match(name) for name in names]
        if all(matches) and len({m[1] for m in matches}) == 1:
            return f"model.layers.{matches[0][1]}.mlp.experts"
        return super().group_name(names)


def _mimo_helpers():
    # Reuse shared model construction only. The MoE-Quant adapter depends on an example converter.
    name = "_one_click_autoquant_mimo_helpers"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "MoE-Quant/src/models/mimo_v2_utils.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Missing shared MiMo construction helpers: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class MiMoRuntime:
    """Explicit execution adapter; no imports from examples or full GPU weight loading."""

    loader_name = "mimo-ep-streaming"

    def __init__(self, path, *, device, dtype, rank, world_size, trust_remote_code):
        import json

        self.path = Path(path).resolve()
        self.source_config = json.loads((self.path / "config.json").read_text())
        if self.source_config.get("model_type") != "mimo_v2":
            raise ValueError("EP runtime currently supports model_type=mimo_v2; other architectures need a runtime adapter")
        if not trust_remote_code:
            raise ValueError("MiMo streaming execution requires --trust-remote-code")
        self.rank, self.world_size = rank, world_size
        self.device, self.dtype = torch.device(device), dtype
        count = int(self.source_config["n_routed_experts"])
        if count <= 0 or count % world_size:
            raise ValueError(f"MiMo experts={count} must be divisible by EP world size={world_size}")
        self.experts_per_rank = count // world_size
        self.adapter = MiMoScopeAdapter()
        self.reader = CheckpointReader(self.path)
        SourceDecoder(self.reader, self.source_config)  # Validate the numeric source contract before model execution.
        helpers = _mimo_helpers()
        self.modeling = helpers.load_modeling_module(SimpleNamespace(_name_or_path=str(self.path)))
        self.helpers = helpers
        self.model = self.build_empty(world_size)
        self.model._autoquantize_execution_device = str(self.device)
        install_ep_forward(self.model)
        self.decoded = DecodedWeights(self)
        self.resources = ExitStack()
        self.residency = None
        self.block_loads = {}
        self.validate_parameters(self.model)

    def build_empty(self, ep_size):
        from accelerate import init_empty_weights

        data = copy.deepcopy(self.source_config)
        data.pop("quantization_config", None)
        config = self.modeling.MiMoV2Config(**data)
        config._name_or_path = str(self.path)
        config.ep_size = ep_size
        config.use_cache = False
        with init_empty_weights(include_buffers=False):
            model = self.helpers.build_model(config, self.dtype, "eager").eval().requires_grad_(False)
        return model

    def source_key(self, name):
        if name in self.reader.headers:
            return name
        renamed = name.replace(".post_attention_layernorm.", ".pre_mlp_layernorm.")
        if renamed in self.reader.headers:
            return renamed
        fused = re.sub(r"\.self_attn\.[qkv]_proj\.weight$", ".self_attn.qkv_proj.weight", name)
        if fused in self.reader.headers and self.source_config.get("attention_projection_layout") == "fused_qkv":
            return fused
        raise ValueError(f"Model parameter has no source checkpoint tensor: {name}")

    def validate_parameters(self, model):
        for name, parameter in model.named_parameters():
            key = self.source_key(name)
            spec = self.reader.headers[key]
            logical_shape = list(spec["shape"])
            if spec["dtype"] == "U8":
                logical_shape[-1] *= 2
            if key == name or ".qkv_proj.weight" not in key:
                if list(parameter.shape) != logical_shape:
                    raise ValueError(f"Source/model shape differs: {name}")
            if spec["dtype"] == "F32":
                parent, _, attr = name.rpartition(".")
                model.get_submodule(parent)._parameters[attr] = torch.nn.Parameter(
                    torch.empty(parameter.shape, dtype=torch.float32, device="meta"), requires_grad=False)

    def global_inventory(self, *, exclude, include, grouping_rules):
        model = self.build_empty(1)
        self.validate_parameters(model)
        scope = discover(model, self.source_config, self.adapter, exclude, grouping_rules, include, allow_meta=True)
        weights, modules, groups, total, minimum = _inventory(model, scope, self.adapter)
        return {"modules": modules, "groups": groups, "weight_parameters": weights,
                "excluded_modules": scope.excluded, "non_linear_modules": scope.non_linear,
                "unmatched_exclusion_rules": scope.unmatched_rules, "grouping_rules": scope.rules,
                "weight_elements": total, "minimum_effective_bits": minimum}

    def owners(self, name):
        match = EXPERT.match(name)
        return {int(match[2]) // self.experts_per_rank} if match else set(range(self.world_size))

    def verify_scope(self, scope, groups):
        signature = [(key, g["status"], g["allowed_bits"]) for key, g in groups.items()]
        if dist.is_initialized():
            signatures = [None] * self.world_size
            dist.all_gather_object(signatures, signature)
            if any(item != signatures[0] for item in signatures):
                raise ValueError("EP ranks have inconsistent group order, exclusions or candidates")

    def forward(self, model, batch):
        inputs = {"input_ids": batch} if isinstance(batch, torch.Tensor) else dict(batch)
        inputs.pop("labels", None)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        return model(**inputs, use_cache=False).logits

    def __enter__(self):
        attach_source_offload(self.model, self.decoded, self.device)
        self.residency = attach_block_residency(self.model, self.decoded, self.device)
        self.resources.callback(self.residency.close)
        self.resources.enter_context(local_modelopt_state(self.model))
        register_gradient_support(self.model)
        return self

    def __exit__(self, *exception):
        self.block_loads = dict(self.residency.block_loads)
        self.resources.__exit__(*exception)
        self.decoded.clear()
        self.reader.close()


def prepare_config(config, reader):
    config.attention_projection_layout = "split"
    # The CLI calibrates the text backbone; these source towers and MTP are not executed.
    config.vision_config = None
    config.audio_config = None
    return {n for n in reader.headers if n.startswith(("visual.", "audio_encoder.", "speech_embeddings.", "model.mtp."))}


def map_source_name(name, source_config):
    target_name = name.replace(".pre_mlp_layernorm.", ".post_attention_layernorm.")
    is_fused = (source_config.get("attention_projection_layout") == "fused_qkv"
                and re.fullmatch(r"model\.layers\.\d+\.self_attn\.qkv_proj\.weight", name))
    return target_name, bool(is_fused)


def decode_source_tensor(decoder, name, weight):
    # MiMo stores [Q0,K0,V0,Q1,K1,V1,...] with block boundaries restarting per TP shard.
    if (decoder.config.get("model_type") == "mimo_v2"
            and decoder.config.get("attention_projection_layout") == "fused_qkv"
            and re.fullmatch(r"model\.layers\.\d+\.self_attn\.qkv_proj\.weight", name)):
        tp = int(decoder.reader.metadata.get("tp_size", 0))
        if tp <= 0:
            raise ValueError(f"{name}: fused MiMo QKV requires index metadata.tp_size")
        layer = int(name.split(".")[2])
        swa = decoder.config["hybrid_layer_pattern"][layer] == 1
        prefix = "swa_" if swa else ""
        heads = decoder.config[prefix + "num_attention_heads"]
        kv_heads = decoder.config[prefix + "num_key_value_heads"]
        hd = decoder.config[prefix + "head_dim"]
        vd = decoder.config[prefix + "v_head_dim"]
        if heads % tp or kv_heads % tp:
            raise ValueError("Source TP must divide MiMo Q and KV heads")
        sizes = (heads // tp * hd, kv_heads // tp * hd, kv_heads // tp * vd)
        rows = sum(sizes)
        if tuple(weight.shape) != (rows * tp, decoder.config["hidden_size"]):
            raise ValueError(f"{name}: fused MiMo QKV shape mismatch")
        scale = decoder._partner(name) if weight.dtype in FP8_DTYPES else None
        scale_rows = 0
        if scale is not None:
            bn, bk = decoder.quant["weight_block_size"]
            scale_rows = math.ceil(rows / bn)
            if tuple(scale.shape) != (scale_rows * tp, math.ceil(weight.shape[1] / bk)):
                raise ValueError(f"{name}: TP-interleaved FP8 scale shape mismatch")
        components = [[], [], []]
        for rank in range(tp):
            part = weight[rank * rows:(rank + 1) * rows]
            s = scale[rank * scale_rows:(rank + 1) * scale_rows] if scale is not None else None
            for output, value in zip(components, decoder._decode(name, part, s).split(sizes)):
                output.append(value)
        decoder.counts["tp_qkv_mappings"] += 1
        return {name.replace("qkv_proj", p + "_proj"): torch.cat(values)
                for p, values in zip(("q", "k", "v"), components)}
    return {name: decoder._decode(name, weight)}


def finish_model(model):
    """Source remote code uses the older Transformers attention-mask signature."""
    import inspect
    import sys

    module = sys.modules[type(model).__module__]
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original = getattr(module, name, None)
        if original is None or getattr(original, "_autoquant_compatible", False):
            continue
        parameters = inspect.signature(original).parameters

        def compatible(*args, _function=original, _parameters=parameters, **kwargs):
            if "inputs_embeds" in _parameters and "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            if "cache_position" not in _parameters:
                kwargs.pop("cache_position", None)
            return _function(*args, **kwargs)

        compatible._autoquant_compatible = True
        setattr(module, name, compatible)
