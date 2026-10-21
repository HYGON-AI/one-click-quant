#!/usr/bin/env python3
"""Sample: collect Kimi-K3 Linear input activations into acts.pt for Flex preprocess.

Output format (consumed by ``preprocess.calib.load_act_stats`` / ``--act-stats``)::

    dict[str, Tensor]  # key = module name (no ".weight"), value float on CPU

Default hooks align with ``preprocess/recipes/kimi_k3.yaml`` Flex targets:
  - experts.*.w2          (flex_awq_ssz prefers down-proj inputs)
  - shared_experts.up_proj / layers.0.mlp.up_proj / routed_expert_up_proj

This is a **single-process sample** (no EP / sync_act_stats / multimodal calibImages).
For full-scale Kimi-K3, extend this script yourself as needed.

Example::

    python3 preprocess/collect_acts_kimi_k3.py \\
      --model /path/Kimi-K3-Mini \\
      --out /path/acts.pt \\
      --device cuda:0 \\
      --max-layers 2 --max-experts 8
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

_DEFAULT_PROMPTS = [
    "Explain what a mixture-of-experts transformer is in one paragraph.",
    "Summarize the benefits of activation-aware weight quantization.",
    "Write a short Python function that computes channel-wise absmax.",
]

# Suffixes / name fragments defaulted to recipe Flex seeds (w2 for AWQ down input).
_DEFAULT_NAME_SUFFIXES = (
    ".w2",
    ".shared_experts.up_proj",
    ".mlp.up_proj",
    ".routed_expert_up_proj",
)


def _parse_layer_idx(name: str) -> Optional[int]:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _parse_expert_idx(name: str) -> Optional[int]:
    m = re.search(r"\.experts\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def should_hook_module(
    name: str,
    suffixes: Sequence[str],
    max_layers: Optional[int],
    max_experts: Optional[int],
) -> bool:
    if not any(name.endswith(sfx) for sfx in suffixes):
        return False
    layer = _parse_layer_idx(name)
    if max_layers is not None and layer is not None and layer >= max_layers:
        return False
    expert = _parse_expert_idx(name)
    if max_experts is not None and expert is not None and expert >= max_experts:
        return False
    return True


def load_calib_texts(path: Optional[Path], max_samples: int) -> List[str]:
    if path is None:
        return _DEFAULT_PROMPTS[:max_samples]
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"calib file is empty: {path}")
    return lines[:max_samples]


def install_hooks(
    model: nn.Module,
    suffixes: Sequence[str],
    max_layers: Optional[int],
    max_experts: Optional[int],
) -> Tuple[Dict[str, List[torch.Tensor]], List]:
    bufs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    handles = []

    def make_hook(mod_name: str):
        def hook(_module, inp, _out):
            if not inp or not isinstance(inp, tuple) or inp[0] is None:
                return
            x = inp[0]
            if not torch.is_tensor(x) or x.numel() == 0:
                return
            flat = x.detach().float().reshape(-1, x.shape[-1]).cpu()
            bufs[mod_name].append(flat)

        return hook

    hooked: Set[str] = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not should_hook_module(name, suffixes, max_layers, max_experts):
            continue
        handles.append(mod.register_forward_hook(make_hook(name)))
        hooked.add(name)

    return bufs, handles


def reduce_buffers(
    bufs: Dict[str, List[torch.Tensor]],
    reduce: str,
    max_tokens_per_mod: int,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, chunks in bufs.items():
        if not chunks:
            continue
        cat = torch.cat(chunks, dim=0)
        if cat.shape[0] > max_tokens_per_mod:
            cat = cat[:max_tokens_per_mod]
        if reduce == "channel_max":
            out[name] = cat.abs().amax(dim=0).float().cpu()
        elif reduce == "raw":
            out[name] = cat.float().cpu()
        else:
            raise ValueError(f"unknown --reduce {reduce!r}")
    return out


def run_collect(
    model_path: Path,
    out_path: Path,
    device: str,
    calib_file: Optional[Path],
    max_samples: int,
    max_seq_len: int,
    max_layers: Optional[int],
    max_experts: Optional[int],
    max_tokens_per_mod: int,
    reduce: str,
    suffixes: Sequence[str],
) -> Dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading model: {model_path}")
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
        model = model.to("cpu")
        exec_device = torch.device("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=device,
            **load_kwargs,
        )
        exec_device = next(model.parameters()).device

    model.eval()
    bufs, handles = install_hooks(model, suffixes, max_layers, max_experts)
    print(f"installed hooks on {len(handles)} Linear modules")

    texts = load_calib_texts(calib_file, max_samples)
    print(f"running {len(texts)} calib sample(s) on {exec_device}")
    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
            enc = {k: v.to(exec_device) for k, v in enc.items()}
            model(**enc)
            print(f"  [{i + 1}/{len(texts)}] tokens={enc['input_ids'].numel()}")

    for h in handles:
        h.remove()

    acts = reduce_buffers(bufs, reduce=reduce, max_tokens_per_mod=max_tokens_per_mod)
    if not acts:
        raise RuntimeError(
            "no activations collected — check --max-layers/--max-experts/suffixes "
            "or whether MoE experts were routed for these prompts"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(acts, out_path)
    print(f"saved {len(acts)} modules -> {out_path} (reduce={reduce})")
    return acts


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect Kimi-K3 Linear input activations into acts.pt (sample, no EP)."
    )
    p.add_argument("--model", type=Path, required=True, help="HF model dir (mini / float preferred)")
    p.add_argument("--out", type=Path, required=True, help="output acts.pt path")
    p.add_argument("--device", type=str, default="cuda:0", help="cpu | cuda:N | device_map string")
    p.add_argument(
        "--calib-file",
        type=Path,
        default=None,
        help="optional text file: one prompt per line (# comments ok)",
    )
    p.add_argument("--max-samples", type=int, default=3, help="max calib prompts")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="only hook layers.<i> with i < max_layers (smoke)",
    )
    p.add_argument(
        "--max-experts",
        type=int,
        default=None,
        help="only hook experts.<i> with i < max_experts (smoke)",
    )
    p.add_argument(
        "--max-tokens-per-mod",
        type=int,
        default=4096,
        help="cap tokens kept per module before reduce",
    )
    p.add_argument(
        "--reduce",
        choices=("channel_max", "raw"),
        default="channel_max",
        help="channel_max: store [K] absmax; raw: store [N,K] (larger)",
    )
    p.add_argument(
        "--suffixes",
        type=str,
        default=",".join(_DEFAULT_NAME_SUFFIXES),
        help="comma-separated name suffixes to hook",
    )
    return p


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_argparser().parse_args(list(argv) if argv is not None else None)
    suffixes = tuple(s.strip() for s in args.suffixes.split(",") if s.strip())
    if not suffixes:
        raise SystemExit("--suffixes resolved to empty list")
    run_collect(
        model_path=args.model,
        out_path=args.out,
        device=args.device,
        calib_file=args.calib_file,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        max_layers=args.max_layers,
        max_experts=args.max_experts,
        max_tokens_per_mod=args.max_tokens_per_mod,
        reduce=args.reduce,
        suffixes=suffixes,
    )


if __name__ == "__main__":
    main()
