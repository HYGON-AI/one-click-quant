# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os

from llmcompressor import model_free_ptq

from utils.logging_config import get_logger

logger = get_logger(__name__)

_WILDCARD_SIGNATURE = (-1, -1)

_ARCH_IGNORE: dict[str, dict[tuple[int, int], list[str]]] = {
    "Qwen3MoeForCausalLM": {
        # Qwen3-30B-A3B
        (2048, 48): [
            "lm_head",
            "re:.*mlp.gate$",
            "re:.*embed_tokens.*"
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
    #"re:.*model.layers.0.*",
    #"re:.*self_attn.o_proj*",
    "re:visual.*",
    "re:model.visual.*",
    "re:.*conv1d.*",
    "re:.*mtp.pre_fc_norm_embedding$",
    "re:.*mtp.pre_fc_norm_hidden$",
    "re:.*embed_tokens$",
    "re:.*shared_expert_gate$",
    #"re:^(?!.*mlp.experts).*",
]


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
    logger.info("model_free_ptq converting %s with scheme %s", args.model, args.scheme)

    config_path = os.path.join(args.model, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    arch = config["architectures"][0]
    signature = _get_signature(config)

    arch_ignores = _ARCH_IGNORE.get(arch, {})
    ignore_entry = arch_ignores.get(signature) or arch_ignores.get(_WILDCARD_SIGNATURE)
    ignore = list(ignore_entry) if ignore_entry else list(_DEFAULT_IGNORE)

    if args.ignore:
        ignore = [x.strip() for x in args.ignore.split(",") if x.strip()]

    save_dir = args.save_dir
    if not save_dir:
        save_dir = f"{args.model}-{args.scheme}"
        logger.warning("save_dir not specified, using default: %s", save_dir)

    logger.info("architecture=%s, signature=%s, ignore=%s", arch, signature, ignore)

    model_free_ptq(
        model_stub=args.model,
        save_directory=save_dir,
        scheme=args.scheme,
        ignore=ignore,
        max_workers=getattr(args, "max_workers", 15),
        device=args.device_map if args.device_map != 'auto' else "cuda:0",
    )

    logger.info("model_free_ptq done, output saved to %s", save_dir)
