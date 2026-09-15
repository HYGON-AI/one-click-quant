# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
import json
import os
import sys

import yaml

from utils.analyze_quant import add_analysis_args, run_analysis
from utils.lm_eval import run_lm_eval
from utils.logging_config import get_logger

logger = get_logger(__name__)

_WILDCARD_SIGNATURE = (-1, -1)


def _load_quant_templates(config_path: str | None = None) -> dict[str, dict[tuple[int, int], dict[str, dict[str, str]]]]:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "templates", "quant_templates.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    result: dict[str, dict[tuple[int, int], dict[str, dict[str, str]]]] = {}
    for arch, entries in raw.items():
        result[arch] = {}
        for entry in entries:
            sig = (entry["hidden_size"], entry["num_hidden_layers"])
            result[arch][sig] = entry["schemes"]
    return result


quant_templates = _load_quant_templates()


class _HelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=_HelpFormatter,
    )
    p.add_argument(
        "--model",
        type=str,
        help="Path to the local LLM model directory containing config.json",
    )
    p.add_argument(
        "--scheme",
        type=str,
        default="FP8_DYNAMIC",
        help=(
            "Quantization scheme. Options:\n"
            "  FP8_DYNAMIC  - BF16 to per-channel FP8\n"
            "  W8A8         - BF16 to per-channel INT8\n"
            "  FP8_TO_BF16  - Block FP8 to BF16\n"
            "  DeepseekV3_FP8_TO_INT8  - Block FP8 to per-channel INT8 for DeepSeek V3 series models\n"
            "  DeepseekV3_FP8_TO_FP8  - Block FP8 to per-channel FP8 for DeepSeek V3 series models\n"
            "  Kimi_W4_TO_INT8  - Group INT4 to per-channel INT8 for Kimi-K2.5\n"
            "  Kimi_W4_TO_FP8  - Group INT4 to per-channel FP8 for Kimi-K2.5\n"
            "  Qwen35_MOE_BF16_TO_INT8  - BF16 to per-channel INT8 for Qwen3.5 series models"
        ),
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="./datasets/ultrachat_200k",
        help="Path to the calibration dataset directory",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save the quantized model",
    )
    p.add_argument(
        "--print-model",
        action="store_true",
        help="Print model structure after loading",
    )
    p.add_argument(
        "--ignore",
        type=str,
        default=None,
        help="Comma-separated ignore patterns passed to the quantization modifier.\n"
        "The code has built-in default ignore patterns; explicitly specifying patterns here "
        "will override the defaults (i.e., defaults no longer apply).\n"
        "Example: 'lm_head,re:.*mlp.gate$,re:.*model.layers.0.*,re:.*self_attn.o_proj.*'",
    )
    p.add_argument(
        "--pipeline",
        type=str,
        default="independent",
        help="Calibration pipeline. Options: 'basic', 'datafree', "
        "'sequential', 'independent'.\n"
        "- basic:     basic calibration pipeline\n"
        "- datafree:  skip calibration data (weight-only quantization)\n"
        "- sequential: process layers one-by-one, lower VRAM on large models\n"
        "- independent: calibrate each module independently",
    )
    p.add_argument(
        "--sequential-targets",
        type=str,
        default=None,
        help="Optional comma-separated list of sequential layer targets.\n"
        "Used with pipeline=sequential. Set to 'Linear' to process one\n"
        "Linear layer at a time, which significantly reduces GPU memory\n"
        "usage on very large models. If not specified, defaults to the\n"
        "no_split_modules defined by the HF model config.",
    )
    p.add_argument(
        "--offload-hessians",
        action="store_true",
        help="Offload Hessian matrices to disk to save GPU memory",
    )
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map strategy for model loading.\n"
        "Supported values: 'auto', 'cpu', 'cuda:0', 'auto_offload'.\n"
        "'auto_offload' loads weights on CPU/disk first, then\n"
        "onloads to GPU on demand during calibration.\n"
        "Distributed mode is auto-detected when launched with\n"
        "  torchrun --nproc_per_node=N main.py ...",
    )
    p.add_argument(
        "--alg",
        type=str,
        default=None,
        help="Quantization algorithm (forced when not None, overrides scheme inference).\n"
        "  ptq:      - PTQ quantization\n"
        "  gptq:     - GPTQ quantization\n"
        "  awq:      - AWQ quantization\n"
        "  gptq_awq: - GPTQ + AWQ quantization\n"
        "  smoothquant_gptq: - SmoothQuant GPTQ quantization\n"
        "  smoothquant_awq: - SmoothQuant AWQ quantization\n"
        "  smoothquant_gptq_awq: - SmoothQuant GPTQ + AWQ quantization\n"
        "  model_free_ptq: - Data-free model-free PTQ\n"
        "  slimquant_ptq:  - Data-free SlimQuant PTQ (W4A8)",
    )
    p.add_argument(
        "--needs-data",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override whether calibration data is required.\n"
        "Set 'true' to force loading a dataset, 'false' to skip.\n"
        "When not set, the default is determined by --scheme.",
    )
    p.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples used for quantization",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration data",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for calibration",
    )
    p.add_argument(
        "--patch-for-llmc",
        action="store_true",
        default=False,
        help="Apply runtime monkey-patches for llmc.\n"
        "Currently skips from_accelerate during save_pretrained to avoid CUDA OOM.",
    )
    p.add_argument(
        "--lm-eval",
        type=str,
        default=None,
        help="Run lm_eval with the given model_args.\n"
        "Example: 'pretrained=/models/Qwen3-4B,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.5'",
    )
    add_analysis_args(p)
    return p.parse_args()


def _load_model_config(model_id: str) -> dict:
    config_path = os.path.join(model_id, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_architecture(config: dict) -> str:
    return config["architectures"][0]


def _get_model_signature(config: dict) -> tuple[int, int]:
    def _get(key: str):
        if key in config:
            return config[key]
        tc = config.get("text_config", {})
        if key in tc:
            return tc[key]
        raise KeyError(f"'{key}' not found in config (top-level or text_config)")

    return (_get("hidden_size"), _get("num_hidden_layers"))


def _lookup_template(architecture: str, signature: tuple[int, int], scheme: str) -> tuple[str | None, dict[str, str]]:
    def _resolve(entry: dict[str, str] | None) -> tuple[str | None, dict[str, str]]:
        if entry is None:
            return None, {}
        template = entry.get("template")
        extra = {k: v for k, v in entry.items() if k != "template"}
        return template, extra

    sub = quant_templates.get(architecture)
    if sub is not None:
        sub2 = sub.get(signature) or sub.get(_WILDCARD_SIGNATURE)
        if sub2 is not None:
            template, extra = _resolve(sub2.get(scheme))
            if template is not None or extra:
                return template, extra

    # Fallback to "*" wildcard architecture: matches any unsupported model arch,
    # e.g. FP8_TO_BF16 scheme applies to all models not covered by specific entries.
    wild = quant_templates.get("*", {})
    sub2 = wild.get(_WILDCARD_SIGNATURE)
    if sub2 is not None:
        return _resolve(sub2.get(scheme))

    return None, {}


def _execute_template(template_path: str, args: argparse.Namespace) -> None:
    modname = os.path.splitext(os.path.basename(template_path))[0]
    spec = importlib.util.spec_from_file_location(modname, template_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    module.run(args)


def _execute_generic_quant(args: argparse.Namespace) -> None:
    from llmc_oneshot import run as generic_run
    generic_run(args)


def main() -> None:
    args = _parse_args()
    logger.info("args: %s", args)

    if args.lm_eval:
        run_lm_eval(args.lm_eval)
        return

    if args.analysis != "none":
        run_analysis(args)
        return

    config = _load_model_config(args.model)
    architecture = _get_architecture(config)
    signature = _get_model_signature(config)
    logger.info("architecture=%s, signature=%s, scheme=%s", architecture, signature, args.scheme)

    template_path, extra_kwargs = _lookup_template(architecture, signature, args.scheme)
    if args.alg not in ("model_free_ptq", "slimquant_ptq"):
        for key, value in extra_kwargs.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    if args.alg == "model_free_ptq":
        logger.info("running model_free_ptq via template")
        _execute_template("templates/model_free_ptq.py", args)
    elif args.alg == "slimquant_ptq":
        logger.info("running slimquant_ptq via template")
        _execute_template("templates/slimquant_ptq.py", args)
    elif template_path is not None:
        logger.info("matched template: %s", template_path)
        _execute_template(template_path, args)
    else:
        logger.info("running generic quant via llmc_oneshot")
        _execute_generic_quant(args)


if __name__ == "__main__":
    main()
