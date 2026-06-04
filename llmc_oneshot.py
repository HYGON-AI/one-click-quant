import argparse
import contextlib
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import torch
from compressed_tensors.offload import init_dist, load_offloaded_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

from utils.logging_config import get_logger

logger = get_logger(__name__)

class _RecipeEntry(NamedTuple):
    modifier: type | str
    needs_data: bool


_SCHEME_RECIPE: dict[str, _RecipeEntry] = {
    "FP8_DYNAMIC": _RecipeEntry(QuantizationModifier, False),
    "FP8_BLOCK":   _RecipeEntry(QuantizationModifier, False),
    "FP8":         _RecipeEntry(QuantizationModifier, False),
    "NVFP4A16":    _RecipeEntry(QuantizationModifier, False),
    "MXFP4A16":    _RecipeEntry(QuantizationModifier, False),
    "MXFP8A16":    _RecipeEntry(QuantizationModifier, False),
    "NVFP4":       _RecipeEntry(QuantizationModifier, True),
    "MXFP4":       _RecipeEntry(QuantizationModifier, True),
    "MXFP8":       _RecipeEntry(QuantizationModifier, True),
    "W8A16":       _RecipeEntry(GPTQModifier,         True),
    "W4A16":       _RecipeEntry(GPTQModifier,         True),
    "W4A16_ASYM":  _RecipeEntry(AWQModifier,          True),
    "W8A8":        _RecipeEntry("smoothquant_gptq",   True),
    "W4A8":        _RecipeEntry("smoothquant_gptq",   True),
    "W4AFP8":      _RecipeEntry("smoothquant_gptq",   True),
}


def _build_recipe(
    scheme: str,
    ignore: list[str] | None,
    offload_hessians: bool,
    alg: str | None = None,
):
    if alg is not None:
        return _build_recipe_from_alg(scheme, ignore, offload_hessians, alg)

    entry = _SCHEME_RECIPE.get(scheme)
    if entry is None:
        raise ValueError(
            f"Unsupported scheme: {scheme}. "
            f"Supported schemes: {sorted(_SCHEME_RECIPE.keys())}"
        )

    mod_cls = entry.modifier

    if mod_cls == "smoothquant_gptq":
        modifier = GPTQModifier(
            targets="Linear",
            scheme=scheme,
            ignore=ignore,
            offload_hessians=offload_hessians,
        )
        return [SmoothQuantModifier(smoothing_strength=0.8), modifier]
    elif mod_cls is QuantizationModifier:
        return QuantizationModifier(
            targets="Linear",
            scheme=scheme,
            ignore=ignore,
        )
    elif mod_cls is GPTQModifier:
        return GPTQModifier(
            targets="Linear",
            scheme=scheme,
            ignore=ignore,
            offload_hessians=offload_hessians,
        )
    elif mod_cls is AWQModifier:
        return AWQModifier(
            targets="Linear",
            scheme=scheme,
            ignore=ignore,
        )
    else:
        raise RuntimeError(f"Unexpected recipe entry for scheme {scheme}: {mod_cls}")


def _build_recipe_from_alg(
    scheme: str,
    ignore: list[str] | None,
    offload_hessians: bool,
    alg: str,
):
    if alg == "ptq":
        return QuantizationModifier(
            targets="Linear", scheme=scheme, ignore=ignore,
        )
    elif alg == "gptq":
        return GPTQModifier(
            targets="Linear", scheme=scheme, ignore=ignore,
            offload_hessians=offload_hessians,
        )
    elif alg == "awq":
        return AWQModifier(
            targets="Linear", scheme=scheme, ignore=ignore,
        )
    elif alg == "gptq_awq":
        return [
            GPTQModifier(
                targets="Linear", scheme=scheme, ignore=ignore,
                offload_hessians=offload_hessians,
            ),
            AWQModifier(
                targets="Linear", scheme=scheme, ignore=ignore,
            ),
        ]
    elif alg == "smoothquant_gptq":
        modifier = GPTQModifier(
            targets="Linear", scheme=scheme, ignore=ignore,
            offload_hessians=offload_hessians,
        )
        return [SmoothQuantModifier(smoothing_strength=0.8), modifier]
    elif alg == "smoothquant_awq":
        modifier = AWQModifier(
            targets="Linear", scheme=scheme, ignore=ignore,
        )
        return [SmoothQuantModifier(smoothing_strength=0.8), modifier]
    elif alg == "smoothquant_gptq_awq":
        return [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(
                targets="Linear", scheme=scheme, ignore=ignore,
                offload_hessians=offload_hessians,
            ),
            AWQModifier(
                targets="Linear", scheme=scheme, ignore=ignore,
            ),
        ]
    else:
        raise ValueError(f"Unsupported alg: {alg}")


def _needs_calibration_data(scheme: str, override: str | None = None) -> bool:
    if override is not None:
        return override == "true"
    entry = _SCHEME_RECIPE.get(scheme)
    return entry is not None and entry.needs_data


def _should_init_dist() -> bool:
    return any(
        key in os.environ
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        )
    )


def _load_and_preprocess_dataset(
    dataset_id: str,
    tokenizer,
    num_calibration_samples: int,
    max_sequence_length: int,
    use_distributed: bool,
):
    if use_distributed:
        if num_calibration_samples < torch.distributed.get_world_size():
            raise ValueError("--num-calibration-samples must be >= world_size")
        splits = get_rank_partition("train_sft", num_calibration_samples)
    else:
        splits = f"train_sft[:{num_calibration_samples}]"

    if dataset_id == "./datasets/ultrachat_200k" and not os.path.exists(dataset_id):
        logger.error("Default dataset directory \"./datasets/ultrachat_200k\" does not exist. "
                     "Please download \"HuggingFaceH4/ultrachat_200k\" to the \"datasets\" directory, "
                     "or specify a dataset via --dataset.")
        sys.exit(1)

    ds = load_dataset(dataset_id, split=splits)
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds


def _copy_non_weight_files(src_root: Path, dst_root: Path) -> None:
    if not src_root.exists():
        return
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix == ".safetensors":
            continue
        if src.name == "model.safetensors.index.json":
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run(args: argparse.Namespace) -> None:
    logger.info("running generic quant via llmc_oneshot, scheme=%s", args.scheme)

    scheme = str(args.scheme)

    ignore = None
    if args.ignore:
        ignore = [x.strip() for x in args.ignore.split(",") if x.strip()]
    else:
        ignore = [
            "lm_head",
            "re:.*mlp.gate$",
            #"re:.*model.layers.0.*",
            #"re:.*self_attn.o_proj*",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*conv1d.*",
            "re:.*embed_tokens$",
            "re:.*shared_expert_gate$",
            #"re:^(?!.*mlp.experts).*",
        ]
    logger.info("ignore=%s", ignore)

    use_distributed = _should_init_dist()
    is_auto_offload = args.device_map == "auto_offload"

    if use_distributed:
        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")
        os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
        init_dist()
        rank = torch.distributed.get_rank()
        load_ctx = load_offloaded_model()
    elif is_auto_offload:
        rank = 0
        load_ctx = load_offloaded_model()
    else:
        rank = 0
        load_ctx = contextlib.nullcontext()

    with load_ctx:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype="auto",
            device_map=args.device_map,
            trust_remote_code=True,
        )

    if args.print_model:
        if rank == 0:
            print(model)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    num_calibration_samples = int(args.num_calibration_samples)
    max_sequence_length = int(args.max_seq_length)
    needs_data = _needs_calibration_data(scheme, override=args.needs_data)
    logger.info("needs_data=%s", needs_data)

    if needs_data:
        ds = _load_and_preprocess_dataset(
            dataset_id=args.dataset,
            tokenizer=tokenizer,
            num_calibration_samples=num_calibration_samples,
            max_sequence_length=max_sequence_length,
            use_distributed=use_distributed,
        )
    else:
        ds = None

    recipe = _build_recipe(
        scheme=scheme,
        ignore=ignore,
        offload_hessians=bool(args.offload_hessians),
        alg=args.alg,
    )

    sequential_targets = None
    if args.sequential_targets:
        sequential_targets = [
            x.strip() for x in args.sequential_targets.split(",") if x.strip()
        ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
        trust_remote_code_model=True,
        pipeline=args.pipeline,
        sequential_targets=sequential_targets,
        batch_size=int(args.batch_size),
    )

    if use_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        if rank != 0:
            return
        torch.distributed.init_process_group(
            backend="gloo",
            world_size=1,
            rank=0,
            timeout=timedelta(seconds=3600),
        )

    save_dir = args.save_dir
    if not save_dir:
        save_dir = f"{args.model}-{args.scheme}"
        if rank == 0:
            logger.warning("save_dir not specified, using default: %s", save_dir)

    if args.patch_for_llmc:
        import llmcompressor.transformers.compression.compressed_tensors_utils as ctu
        ctu.from_accelerate = lambda model: None
    model.requires_grad_(False)

    try:
        model.save_pretrained(save_dir, save_compressed=True)
    except Exception as e:
        logger.exception("model.save_pretrained exception: %s", e)
    tokenizer.save_pretrained(save_dir)
    _copy_non_weight_files(Path(args.model), Path(save_dir))

    logger.info("generic quant done, output saved to %s", save_dir)

    if use_distributed:
        torch.distributed.destroy_process_group()
