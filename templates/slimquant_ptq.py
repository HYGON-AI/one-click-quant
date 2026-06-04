"""
SlimQuant PTQ template -- data-free per-channel quantization supporting INT4/INT8/FP8.

Usage via main.py:
    python main.py --model <model_path> --alg slimquant_ptq --scheme <scheme> [--save-dir <dir>] [--ignore <patterns>]

Supported --scheme values and their quant_type mapping:
    W4A8 / W4A8_INT4   -> int4 (W4A8 per-channel INT8 + INT4 MoE with grid-search)
    W8A8 / W8A8_INT8   -> int8 (W8A8 per-channel INT8)
    FP8_DYNAMIC / FP8  -> fp8 (FP8 per-channel E4M3)

--ignore:
    Comma-separated module names or regex patterns to skip during quantization.
    Each pattern can be a literal module name (e.g. "lm_head") or a regex prefix
    (e.g. "re:.*mlp.gate$"). User-provided patterns are APPENDED to the
    auto-detected defaults below (not replaced).

    Default ignore lists auto-selected by architecture + (hidden_size, num_hidden_layers) signature:

    ┌──────────────────────────────────────────┬──────────────┬──────────────────────────────────────────────┐
    │ Architecture                             │ Signature    │ Default ignore patterns                      │
    ├──────────────────────────────────────────┼──────────────┼──────────────────────────────────────────────┤
    │ Qwen3MoeForCausalLM                      │ (2048, 48)   │ lm_head, re:.*mlp.gate$, re:.*embed_tokens.* │
    ├──────────────────────────────────────────┼──────────────┼──────────────────────────────────────────────┤
    │ Qwen3_5MoeForConditionalGeneration       │ (2048, 40)   │ lm_head, re:.*mlp.gate$,                     │
    │                                          │              │ re:.*mlp.shared_expert_gate.*, re:.*norm.*,  │
    │                                          │              │ re:.*embed_tokens.*, re:.*visual.*,          │
    │                                          │              │ re:.*conv1d.*                                │
    ├──────────────────────────────────────────┼──────────────┼──────────────────────────────────────────────┤
    │ DeepseekV32ForCausalLM                   │ (7168, 61)   │ lm_head, re:.*embed_tokens.*,                │
    │                                          │              │ re:.*mlp.gate$, re:.*weights_proj.*          │
    ├──────────────────────────────────────────┼──────────────┼──────────────────────────────────────────────┤
    │ GlmMoeDsaForCausalLM                     │ (6144, 78)   │ re:.*norm.weight.*, re:.*embed_tokens.*,     │
    │                                          │              │ re:.*input_layernorm.*,                      │
    │                                          │              │ re:.*post_attention_layernorm.*,             │
    │                                          │              │ re:.*mlp.gate$*, re:.*self_attn.k_norm.*,    │
    │                                          │              │ re:.*self_attn.q_norm.*, re:.*lm_head.*,     │
    │                                          │              │ re:.*weights_proj.*                          │
    ├──────────────────────────────────────────┼──────────────┼──────────────────────────────────────────────┤
    │ (unmatched architecture / signature)     │ fallback     │ lm_head, re:.*mlp.gate$, re:.*embed_tokens.* │
    └──────────────────────────────────────────┴──────────────┴──────────────────────────────────────────────┘

Examples:
    # W4A8 quantize DeepSeek-V3.2, auto-detected ignore + extra o_proj/q_proj
    python main.py --model DeepSeek-V3.2-bf16/ --alg slimquant_ptq --scheme W4A8 \\
        --save-dir ./output-w4a8 --ignore "re:.*self_attn.o_proj.*,re:.*self_attn.q_proj.*"

    # INT8 quantize with default ignores only
    python main.py --model DeepSeek-V3.2-bf16/ --alg slimquant_ptq --scheme W8A8

    # FP8 quantize, auto-generate output dir name
    python main.py --model Qwen3-30B-A3B/ --alg slimquant_ptq --scheme FP8_DYNAMIC
"""

import argparse
import json
import os

#from llmcompressor.entrypoints.slimquant import slimquant_ptq
from templates.slimquant import slimquant_ptq

from utils.logging_config import get_logger

logger = get_logger(__name__)

_WILDCARD_SIGNATURE = (-1, -1)

_ARCH_IGNORE: dict[str, dict[tuple[int, int], list[str]]] = {
    "Qwen3MoeForCausalLM": {
        # Qwen3-30B-A3B
        (2048, 48): [
            "lm_head",
            "re:.*mlp.gate$",
            "re:.*embed_tokens.*",
        ],
    },
    "Qwen3_5MoeForConditionalGeneration": {
        # Qwen3.5-35B-A3B
        (2048, 40): [
            "lm_head",
            "re:.*mlp.gate$",
            "re:.*mlp.shared_expert_gate.*",
            "re:.*norm.*",
            "re:.*embed_tokens.*",
            "re:.*visual.*",
            "re:.*conv1d.*",
        ],
    },
    "DeepseekV32ForCausalLM": {
        # DeepSeek-V3.2
        (7168, 61): [
            "lm_head",
            "re:.*embed_tokens.*",
            "re:.*mlp.gate$",
            "re:.*weights_proj.*",
        ],
    },
    "GlmMoeDsaForCausalLM": {
        # GLM-5
        (6144, 78): [
            "re:.*norm.weight.*",
            "re:.*embed_tokens.*",
            "re:.*input_layernorm.*",
            "re:.*post_attention_layernorm.*",
            "re:.*mlp.gate$*",
            "re:.*self_attn.k_norm.*",
            "re:.*self_attn.q_norm.*",
            "re:.*lm_head.*",
            "re:.*weights_proj.*",
        ],
    },
}

_DEFAULT_IGNORE = [
    "lm_head",
    "re:.*mlp.gate$",
    "re:.*embed_tokens.*",
]

# Map --scheme values to slimquant quant_type
_SCHEME_TO_QUANT_TYPE: dict[str, str] = {
    "W4A8": "int4",
    "W4A8_INT4": "int4",
    "W8A8": "int8",
    "W8A8_INT8": "int8",
    "FP8_DYNAMIC": "fp8",
    "FP8": "fp8",
}


def _get_signature(config: dict) -> tuple[int, int]:
    def _get(key: str):
        if key in config:
            return config[key]
        tc = config.get("text_config", {})
        if key in tc:
            return tc[key]
        raise KeyError(f"'{key}' not found in config (top-level or text_config)")

    return (_get("hidden_size"), _get("num_hidden_layers"))


def run(args: argparse.Namespace) -> None:
    quant_type = _SCHEME_TO_QUANT_TYPE.get(args.scheme, "int4")
    logger.info("slimquant_ptq converting %s with scheme=%s -> quant_type=%s",
                args.model, args.scheme, quant_type)

    config_path = os.path.join(args.model, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    arch = config["architectures"][0]
    signature = _get_signature(config)

    arch_ignores = _ARCH_IGNORE.get(arch, {})
    ignore_entry = arch_ignores.get(signature) or arch_ignores.get(_WILDCARD_SIGNATURE)
    ignore = set(ignore_entry) if ignore_entry else set(_DEFAULT_IGNORE)

    if args.ignore:
        user_ignore = [x.strip() for x in args.ignore.split(",") if x.strip()]
        ignore.update(user_ignore)

    save_dir = args.save_dir
    if not save_dir:
        save_dir = f"{args.model}-slimquant-{quant_type}"
        logger.warning("save_dir not specified, using default: %s", save_dir)

    logger.info("architecture=%s, signature=%s, ignore=%s", arch, signature, ignore)

    device = args.device_map if args.device_map != "auto" else "cuda:0"

    kwargs = dict(
        model_stub=args.model,
        save_directory=save_dir,
        ignore=ignore,
        quant_type=quant_type,
        device=device,
    )

    if quant_type == "int4":
        kwargs.update(
            moe_pattern=getattr(args, "moe_pattern", ".mlp.experts."),
            k_min=getattr(args, "k_min", 0.97),
            k_max=getattr(args, "k_max", 1.0),
            k_steps=getattr(args, "k_steps", 30),
            search_metric=getattr(args, "search_metric", "mse"),
        )

    slimquant_ptq(**kwargs)

    logger.info("slimquant_ptq done, output saved to %s", save_dir)
