"""Kimi-K3 text calibration with rank-local experts and decoder-block streaming."""

from contextlib import ExitStack
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import types
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from ..ep_comm import gather_variable_tokens, differentiable_all_reduce, _local_moe_float32
from ..loading import CheckpointReader, SourceDecoder, DecodedWeights
from .mimo import MiMoRuntime
from ..scope import ModelAdapter
from ..streaming import BlockResidency


EXPERT = re.compile(r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.")


def _kimi_modeling(path):
    """Import trusted local code with version bridges, without writing a HF module cache."""
    import transformers.utils.generic as generic

    if torch.version.hip:
        # FLA 0.6 selects bf16x3 from the Triton version alone, but the DCU
        # backend does not support it. Use its full-precision fallback before
        # any KDA backward kernel is compiled; keep the installed FLA unchanged.
        import triton.language as tl
        intra = importlib.import_module("fla.ops.kda.chunk_intra")
        if getattr(getattr(intra, "SAFE_GATE_DOT_PRECISION", None), "value", None) == "bf16x3":
            intra.SAFE_GATE_DOT_PRECISION = tl.constexpr("ieee")

    package_name = "_autoquant_kimi_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(path)]
        # K3.1 releases ship configuration/tokenizer code but no HF text model.
        # Reuse the licensed text implementation; configuration still comes from
        # the checkpoint, and all parameters are validated against its headers.
        if not (path / "modeling_kimi_linear.py").is_file():
            package.__path__.append(str(Path(__file__).resolve().parents[3]
                                       / "examples/kimi/modeling"))
        sys.modules[package_name] = package
    if hasattr(generic, "OutputRecorder"):
        recorder = generic.OutputRecorder
    else:
        from transformers.utils.output_capturing import OutputRecorder
        recorder = OutputRecorder
    with patch.object(generic, "OutputRecorder", recorder, create=True):
        modeling = importlib.import_module(f"{package_name}.modeling_kimi_linear")
    if not getattr(modeling, "_autoquant_mask_patched", False):
        original = modeling.create_causal_mask
        parameters = inspect.signature(original).parameters
        def compatible_mask(**kwargs):
            if "inputs_embeds" in parameters and "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            if "cache_position" not in parameters:
                kwargs.pop("cache_position", None)
            return original(**kwargs)
        modeling.create_causal_mask = compatible_mask
        modeling._autoquant_mask_patched = True
    return modeling


class KimiScopeAdapter(ModelAdapter):
    grouping_rules = (r"^(language_model\.model\.layers\.\d+\.block_sparse_moe\.experts)\.\d+\.w[123]$",)
    # Remote attention residuals access these weights functionally, bypassing Linear.forward.
    exclude_patterns = ("*.self_attention_res_proj", "*.mlp_res_proj", "*.output_attn_res_proj",
                        "*.self_attn.indexer.*")

    def group_name(self, names):
        matches = [EXPERT.match(name) for name in names]
        if all(matches) and len({m[1] for m in matches}) == 1:
            return f"language_model.model.layers.{matches[0][1]}.block_sparse_moe.experts"
        return super().group_name(names)


def _kimi_ep_forward(self, hidden_states):
    if self.training:
        raise ValueError("Kimi EP gradient scoring requires eval-mode routing")
    if self.ep_size > 1 and (not dist.is_initialized() or dist.get_world_size() != self.ep_size
                             or dist.get_rank() != self.ep_rank):
        raise ValueError("Kimi EP process group differs from the model expert partition")
    indices, weights = self.gate(hidden_states)
    routed = hidden_states.reshape(-1, hidden_states.shape[-1])
    if self.use_latent_moe:
        routed = self.routed_expert_down_proj(routed)
    width = routed.shape[-1]
    # Shared projections see only local tokens; their sensitivity is summed once across ranks.
    packed, sizes = gather_variable_tokens(torch.cat((routed.float(), weights.float()), dim=-1))
    indices, _ = gather_variable_tokens(indices)
    partial = _local_moe_float32(self, packed[:, :width].to(routed.dtype), indices, packed[:, width:])
    # Empty expert owners must still participate in both backward collectives.
    partial = partial + packed.sum() * 0.0
    output = differentiable_all_reduce(partial)
    output = output.narrow(0, sum(sizes[:self.ep_rank]), sizes[self.ep_rank]).to(hidden_states.dtype)
    if self.use_latent_moe:
        if self.latent_moe_use_norm:
            output = self.routed_expert_norm(output)
        output = self.routed_expert_up_proj(output)
    output = output.reshape(hidden_states.shape)
    if self.config.num_shared_experts is not None:
        output = output + self.shared_experts(hidden_states)
    return output


class _KimiTextModel(nn.Module):
    """Preserve checkpoint key prefixes while omitting the unused vision tower."""

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
        self.config = language_model.config

    @property
    def model(self):
        return self.language_model.model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)


class _KimiShortContextIndexer(nn.Module):
    """Inventory DSA weights when every causal key fits in the top-k budget.

    The indexer cannot change attention in this regime. Longer sequences are
    rejected by KimiRuntime.forward rather than silently using dense attention.
    """

    def __init__(self, config):
        super().__init__()
        dsa = config.dsa_config
        if dsa.get("index_rope_dim", 0) != 0 or dsa.get("index_k_norm_type") != "layernorm":
            raise ValueError("K3.1 short-context adapter requires NoPE/LayerNorm DSA")
        dim = dsa["index_head_dim"]
        self.wq_b = nn.Linear(config.q_lora_rank, dsa["index_num_heads"] * dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, dim, bias=False)
        self.k_norm = nn.LayerNorm(dim)
        self.weights_proj = nn.Linear(config.hidden_size, dsa["index_num_heads"], bias=False)


class KimiRuntime(MiMoRuntime):
    """Reuse source offload, checkpoint scoring and global inventory from the MiMo runtime."""

    loader_name = "kimi-k3-ep-streaming"

    def __init__(self, path, *, device, dtype, rank, world_size, trust_remote_code):
        self.path = Path(path).resolve()
        self.source_config = json.loads((self.path / "config.json").read_text())
        if self.source_config.get("model_type") not in ("kimi_k3", "kimi_k31"):
            raise ValueError("Kimi streaming execution requires model_type=kimi_k3 or kimi_k31")
        if not trust_remote_code:
            raise ValueError("Kimi streaming execution requires --trust-remote-code")
        self.rank, self.world_size = rank, world_size
        self.device, self.dtype = torch.device(device), dtype
        count = int(self.source_config["text_config"]["num_experts"])
        if count <= 0 or world_size <= 0 or count % world_size:
            raise ValueError(f"Kimi experts={count} must be divisible by EP world size={world_size}")
        self.experts_per_rank = count // world_size
        self.adapter = KimiScopeAdapter()
        self.reader = CheckpointReader(self.path)
        SourceDecoder(self.reader, self.source_config)
        self.modeling = _kimi_modeling(self.path)
        self.dsa_config = self.source_config["text_config"].get("dsa_config")
        if self.dsa_config:
            self.loader_name = "kimi-k31-ep-streaming-short-context"
        self.model = self.build_empty(world_size)
        self.model._autoquantize_execution_device = str(self.device)
        # ModelOpt's Mixtral w1/w2/w3 rule picks ModuleList, whose forward never executes.
        self.model._autoquantize_score_module_rules = (
            r"^(.*?\.block_sparse_moe)\.experts\.\d+\.w[123]$",)
        self.decoded = DecodedWeights(self)
        self.resources = ExitStack()
        self.residency = None
        self.block_loads = {}
        self.validate_parameters(self.model)

    def build_empty(self, ep_size):
        from accelerate import init_empty_weights

        data = copy.deepcopy(self.source_config["text_config"])
        data.pop("quantization_config", None)
        config = self.modeling.KimiLinearConfig(**data)
        config._name_or_path = str(self.path)
        config.use_cache = False
        config._attn_implementation = "eager"
        count = config.num_experts
        start = self.rank * (count // ep_size) if ep_size > 1 else 0
        end = start + count // ep_size
        original_mlp = self.modeling.KimiBlockSparseMLP
        expert_index = 0

        def local_expert(*args, **kwargs):
            nonlocal expert_index
            index = expert_index % count
            expert_index += 1
            return original_mlp(*args, **kwargs) if start <= index < end else None

        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(self.dtype)
            with init_empty_weights(include_buffers=False), patch.object(self.modeling, "KimiBlockSparseMLP", local_expert):
                model = _KimiTextModel(self.modeling.KimiLinearForCausalLM(config)).eval().requires_grad_(False)
                if self.dsa_config:
                    for layer in self.dsa_config["dsa_layers"]:
                        attention = model.model.layers[layer - 1].self_attn
                        if not isinstance(attention, self.modeling.KimiMLAAttention):
                            raise ValueError(f"DSA layer {layer} is not an MLA layer")
                        if not hasattr(attention, "indexer"):
                            attention.indexer = _KimiShortContextIndexer(config).requires_grad_(False)
        finally:
            torch.set_default_dtype(original_dtype)
        # The remote model forces FlashAttention even when eager was requested.
        config._attn_implementation = "eager"
        model.model._use_flash_attention_2 = False
        for module in model.modules():
            if isinstance(module, self.modeling.KimiSparseMoeBlock):
                module.ep_size, module.ep_rank = ep_size, self.rank if ep_size > 1 else 0
                module.experts_start_idx, module.experts_end_idx = start, end
                module.experts_per_rank = end - start
                module._autoquantize_ep_original_forward = module.forward
                module._autoquantize_ep_forward = types.MethodType(_kimi_ep_forward, module)
                module.forward = module._autoquantize_ep_forward
        return model

    def forward(self, model, batch):
        if self.dsa_config:
            tokens = batch if isinstance(batch, torch.Tensor) else batch["input_ids"]
            if tokens.shape[-1] > self.dsa_config["index_topk"]:
                raise ValueError("K3.1 analysis requires sequence_length <= DSA index_topk; "
                                 "long-context sparse attention is not implemented")
        return super().forward(model, batch)

    def source_key(self, name):
        if name in self.reader.headers:
            return name
        packed = name.removesuffix(".weight") + ".weight_packed"
        if name.endswith(".weight") and packed in self.reader.headers:
            return packed
        raise ValueError(f"Model parameter has no source checkpoint tensor: {name}")

    def validate_parameters(self, model):
        consumed = set()
        decoder = SourceDecoder(self.reader, self.source_config)
        for name, parameter in model.named_parameters():
            source_key = self.source_key(name)
            consumed.add(source_key)
            consumed.update(decoder.partner_names(source_key))
            spec = self.reader.headers[source_key]
            shape = list(spec["shape"])
            if spec["dtype"] == "U8":
                shape[-1] *= 2
            if name.endswith(".self_attn.A_log") and shape == [1 << (parameter.numel() - 1).bit_length()]:
                shape = list(parameter.shape)
            if shape != list(parameter.shape):
                raise ValueError(f"Source/model shape differs: {name}: {spec['shape']} != {list(parameter.shape)}")
            if spec["dtype"] == "F32":
                parent, _, attr = name.rpartition(".")
                model.get_submodule(parent)._parameters[attr] = nn.Parameter(
                    torch.empty(parameter.shape, dtype=torch.float32, device="meta"), requires_grad=False)
        # Validate both directions: shape checks alone would silently omit new
        # K3.1 branches, e.g. its DSA indexer. Other ranks' experts are intentional.
        expert_ids = {int(m[2]) for name in consumed if (m := EXPERT.match(name))}
        unexpected = []
        for name in self.reader.headers:
            if not name.startswith("language_model.") or name in consumed:
                continue
            match = EXPERT.match(name)
            if (match and int(match[2]) < self.source_config["text_config"]["num_experts"]
                    and int(match[2]) not in expert_ids):
                continue
            unexpected.append(name)
        if unexpected:
            raise ValueError(f"Source text tensors absent from the Kimi model: {unexpected[:10]}")

    def loading_audit(self):
        code = Path(self.modeling.__file__)
        result = {"text_modeling_file": str(code),
                  "text_modeling_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                  "baseline": "source checkpoint decoded to analysis dtype, including MXFP4 experts"}
        if self.dsa_config:
            result.update(dsa_execution="all causal keys fit within index_topk; dense attention is equivalent",
                          maximum_calibration_sequence_length=self.dsa_config["index_topk"],
                          dsa_indexer="inventoried, excluded from quantization search")
        return result

    def owners(self, name):
        match = EXPERT.match(name)
        return {int(match[2]) // self.experts_per_rank} if match else set(range(self.world_size))

    def __enter__(self):
        super().__enter__()
        backbone = self.model.model
        if backbone.use_attn_residuals:
            # The final attention residual also reads projection/norm weights directly.
            original = backbone._apply_output_attn_res
            boundary = nn.Module()
            boundary.proj = backbone.output_attn_res_proj
            boundary.norm = backbone.output_attn_res_norm
            boundary.forward = original
            residency = BlockResidency(self.model, self.decoded, self.device, blocks=[boundary])
            self.resources.callback(residency.close)
            self.resources.callback(setattr, backbone, "_apply_output_attn_res", original)
            backbone._apply_output_attn_res = boundary.forward
        return self


def decode_source_tensor(decoder, name, weight):
    if decoder.config.get("model_type") in ("kimi_k3", "kimi_k31") and name.endswith(".self_attn.A_log"):
        heads = decoder.config["text_config"]["linear_attn_config"]["num_heads"]
        if tuple(weight.shape) == (1 << (heads - 1).bit_length(),) and weight.numel() > heads:
            # Kimi-K3 stores this vector padded to a power of two; only real heads are used.
            if torch.count_nonzero(weight[heads:]):
                raise ValueError(f"{name}: nonzero padding in KDA head parameters")
            weight = weight[:heads]
    return {name: decoder._decode(name, weight)}
