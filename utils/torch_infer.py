# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Generic PyTorch inference script for testing model generations.

Use this when vLLM cannot load a quantized model — it falls back to PyTorch's
native dequant→matmul path via transformers AutoModelForCausalLM.

Usage::

    python -m utils.torch_infer --model ./Qwen3-4B-W4A8-channelwise

    python -m utils.torch_infer --model ./Qwen3-4B-W4A8-channelwise \\
        --prompt "你好，请介绍一下你自己" --max-tokens 200
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)


def _human_bytes(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TiB"


def _detect_device(prefer: Optional[str] = None):
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_model_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_model_info(config: dict) -> None:
    qconfig = config.get("quantization_config") or config.get("compression_config")

    logger.info("=" * 60)
    logger.info("Model Info")
    logger.info("=" * 60)

    hf_config = config.copy()
    for key in ("quantization_config", "compression_config"):
        hf_config.pop(key, None)

    arch = hf_config.get("architectures", ["unknown"])[0]
    logger.info("  Architecture:      %s", arch)
    logger.info("  Hidden size:       %s", hf_config.get("hidden_size", "?"))
    logger.info("  Num layers:        %s", hf_config.get("num_hidden_layers", "?"))
    logger.info("  Num attention:     %s", hf_config.get("num_attention_heads", "?"))
    logger.info("  Num kv heads:      %s", hf_config.get("num_key_value_heads", "?"))
    logger.info("  Vocabulary size:   %s", hf_config.get("vocab_size", "?"))

    logger.info("-" * 60)
    if qconfig:
        quant_method = qconfig.get("quant_method", "?")
        quant_status = qconfig.get("quantization_status", "?")
        logger.info("  Quant method:      %s", quant_method)
        logger.info("  Quant status:      %s", quant_status)

        groups = qconfig.get("config_groups", {})
        for gname, gcfg in groups.items():
            w = gcfg.get("weights", {})
            a = gcfg.get("input_activations", {})
            logger.info(
                "  [%s] W%sA%s  w_strategy=%s a_strategy=%s w_sym=%s a_sym=%s",
                gname,
                w.get("num_bits", "?"),
                a.get("num_bits", "?"),
                w.get("strategy", "?"),
                a.get("strategy", "?"),
                w.get("symmetric", "?"),
                a.get("symmetric", "?"),
            )
        logger.info("  Ignore:            %s", qconfig.get("ignore", []))
    else:
        logger.info("  Quantization:      <none — running BF16/FP16 original>")

    logger.info("=" * 60)


def _gpu_memory_report(prefix: str = "") -> None:
    import torch

    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    logger.info(
        "%sGPU memory: allocated=%s, reserved=%s",
        prefix + ": " if prefix else "",
        _human_bytes(allocated),
        _human_bytes(reserved),
    )


def _is_quantized_weight(weight) -> bool:
    import torch

    if not hasattr(weight, "dtype"):
        return False
    return not weight.dtype.is_floating_point


def _dequantize_compressed_model(model) -> int:
    """Fix dtype mismatch: convert int8/int4 quantized weights to bfloat16/float16.

    After ``dispatch_model`` wraps linear layers with ``CompressedLinear``,
    ``weight`` may still be int8, causing ``RuntimeError: c10::BFloat16 != signed char``
    when the forward pass tries ``F.linear(bf16_input, int8_weight)``.

    This function finds all modules with non-floating-point weights, reads the
    per-channel scale from ``weight_scale``, and replaces the weight tensor with
    its float32 → bf16 dequantized version.
    """
    import torch

    target_dtype = None
    count = 0

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        try:
            weight = module.weight
        except Exception:
            continue
        if not _is_quantized_weight(weight):
            continue

        if target_dtype is None:
            target_dtype = next(
                (p.dtype for p in model.parameters() if p.dtype.is_floating_point),
                torch.bfloat16,
            )

        scale = None
        for attr in ("weight_scale", "_weight_scale"):
            try:
                s = getattr(module, attr, None)
                if s is not None and hasattr(s, "dtype"):
                    scale = s
                    break
            except Exception:
                continue

        if scale is None:
            logger.info(
                "  [%s] quantized (dtype=%s) but no weight_scale — skipping",
                name,
                weight.dtype,
            )
            continue

        scale_f32 = scale.detach().to(torch.float32)
        if scale_f32.dim() == 1:
            scale_f32 = scale_f32.unsqueeze(1)

        weight_f32 = weight.detach().to(torch.float32)
        dequant = (weight_f32 * scale_f32).to(dtype=target_dtype).to(weight.device)

        new_param = torch.nn.Parameter(dequant, requires_grad=False)
        try:
            module.weight = new_param
        except Exception:
            module.weight.data.copy_(dequant)

        count += 1

    if count:
        logger.info(
            "Dequantized %d compressed module(s) to %s", count, target_dtype
        )
    return count


def load_model(
    model_dir: str,
    device,
    torch_dtype: str = "auto",
    no_dispatch: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_dir).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    config = _read_model_config(model_path)
    _print_model_info(config)

    logger.info("Loading tokenizer from %s ...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s (dtype=%s) ...", model_path, torch_dtype)
    _gpu_memory_report("before load")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map=str(device) if device.type == "cuda" else None,
        trust_remote_code=True,
    )

    _gpu_memory_report("after load")

    if not no_dispatch:
        try:
            from compressed_tensors.offload import dispatch_model

            model = dispatch_model(model)
            logger.info("Model dispatched (compressed_tensors hooks applied).")
        except Exception:
            logger.info("dispatch_model not applied (model may not be compressed).")
    else:
        logger.info("Skipping dispatch_model (--no-dispatch).")

    dequant_count = _dequantize_compressed_model(model)

    model.eval()
    return model, tokenizer


def run_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.95,
    do_sample: bool = False,
) -> tuple[str, float, int]:
    import torch

    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_token_count = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated_ids = outputs[0][input_token_count:]
    new_tokens = generated_ids.shape[0]
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else float("inf")

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, tok_per_sec, new_tokens


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic PyTorch inference for model validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick smoke-test with default prompts
    python -m utils.torch_infer --model ./Qwen3-4B-w4a8-channelwise

    # Custom prompt
    python -m utils.torch_infer --model ./Qwen3-4B-w4a8-channelwise \\
        --prompt "写一首关于春天的诗"

    # Batch (comma-separated prompts)
    python -m utils.torch_infer --model ./Qwen3-4B-w4a8-channelwise \\
        --prompt "你好,1+1等于几,Python是什么"
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model directory (HuggingFace format).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Prompt(s). Use commas for multiple prompts, e.g.: "
            "'你好,1+1等于几,Python是什么'. "
            "Defaults to a built-in Chinese+English test set."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature (0.0 = greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. 'cuda:0', 'cpu'). Auto-detected by default.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Torch dtype for model weights (e.g. 'auto', 'bfloat16', 'float16').",
    )
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="Skip compressed-tensors dispatch_model (use when dispatch causes errors).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    import torch
    device = _detect_device(args.device)
    logger.info("Using device: %s", device)

    model, tokenizer = load_model(
        args.model, device=device, torch_dtype=args.dtype, no_dispatch=args.no_dispatch
    )

    prompts: list[str]
    if args.prompt:
        prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    else:
        prompts = [
            "你好，请用一句话介绍一下你自己。",
            "What is the capital of France?",
            "请用Python写一个快速排序算法。",
        ]

    total_tokens = 0
    total_time = 0.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Running %d prompt(s)  (max_new_tokens=%d)", len(prompts), args.max_tokens)
    logger.info("=" * 60)

    for i, prompt in enumerate(prompts):
        logger.info("")
        logger.info("--- Prompt %d/%d ---", i + 1, len(prompts))
        logger.info("Input:  %s", prompt[:120] + ("..." if len(prompt) > 120 else ""))

        generated, tok_per_sec, new_tokens = run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0.0,
        )

        total_tokens += new_tokens
        total_time += (new_tokens / tok_per_sec) if tok_per_sec > 0 else 0.0

        logger.info("Output: %s", generated[:200] + ("..." if len(generated) > 200 else ""))
        logger.info("Speed:  %.1f tokens/sec  (%d new tokens)", tok_per_sec, new_tokens)

    _gpu_memory_report("peak")

    avg_speed = total_tokens / total_time if total_time > 0 else 0.0
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("  Prompts:        %d", len(prompts))
    logger.info("  Total tokens:   %d", total_tokens)
    logger.info("  Avg speed:      %.1f tokens/sec", avg_speed)
    logger.info("=" * 60)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    _gpu_memory_report("after cleanup")


if __name__ == "__main__":
    main()
