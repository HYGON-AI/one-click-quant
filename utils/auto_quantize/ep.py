"""Distributed execution and globally validated precision-plan aggregation."""

import copy
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .api import analyze_model, _calibration_metadata, validate_plan
from .model_load import get_ep_runtime


def merge_ep_plans(plans, inventory, *, effective_bits, owners):
    """Sum token-local raw scores, count shared weights once, solve once globally."""
    from modelopt.torch.quantization.algorithms import AutoQuantizeGradientSearcher

    groups = copy.deepcopy(inventory["groups"])
    expected = inventory["modules"]
    seen_groups = {key: set() for key in groups}
    for group in groups.values():
        group["forward_calls"] = dict.fromkeys(group["module_names"], 0)
        if group["status"] == "searched":
            group["scores"] = {"4": 0., "8": 0.}
    for rank, plan in enumerate(plans):
        if (not plan.get("_rank_local_scores") or plan["method"] != "modelopt_gradient"
                or plan["modelopt_version"] != plans[0]["modelopt_version"]
                or plan["model"]["config_sha256"] != plans[0]["model"]["config_sha256"]):
            raise ValueError("EP workers have incompatible scoring methods, versions or source configs")
        local_names = {name for name in expected if rank in owners(name)}
        if set(plan["modules"]) != local_names:
            raise ValueError(f"EP rank {rank} source coverage/expert ownership differs from the global inventory")
        for name, module in plan["modules"].items():
            if (module["parameters"] != expected[name]["parameters"]
                    or module["status"] != expected[name]["status"]):
                raise ValueError(f"EP parameter metadata/status mismatch: {name}")
        local_weights = {p["cost_key"] for module in plan["modules"].values() for p in module["parameters"].values()}
        if set(plan["weight_parameters"]) != local_weights or any(
                plan["weight_parameters"][name] != inventory["weight_parameters"][name] for name in local_weights):
            raise ValueError(f"EP rank {rank} weight cost inventory differs from the source")
        for key, local in plan["groups"].items():
            if key not in groups or rank in seen_groups[key]:
                raise ValueError(f"Unknown or duplicate EP group: {key}")
            global_group = groups[key]
            if (set(local["module_names"]) != set(global_group["module_names"]) & local_names
                    or local["status"] != global_group["status"]
                    or local["allowed_bits"] != global_group["allowed_bits"]):
                raise ValueError(f"EP fusion group differs from global inventory: {key}")
            count = sum(plan["weight_parameters"][name]["numel"] for name in local["weight_parameters"])
            if count != local["weight_elements"]:
                raise ValueError(f"EP local weight cost mismatch: {key}")
            seen_groups[key].add(rank)
            for name, calls in local["forward_calls"].items():
                global_group["forward_calls"][name] += calls
            if local["status"] == "searched":
                for bit in ("4", "8"):
                    value = float(local["scores"][bit])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"Non-finite/negative raw EP sensitivity: {key}")
                    global_group["scores"][bit] += value
    table = {}
    uncertain = []
    for key, group in groups.items():
        expected_ranks = set().union(*(owners(name) for name in group["module_names"]))
        if seen_groups[key] != expected_ranks:
            raise ValueError(f"EP ranks do not cover global group: {key}")
        if group["status"] == "excluded":
            group["selected_bits"] = None
            continue
        # Sum first: sum(min(local S4, local S8)) is not the global monotonic envelope.
        group["scores"]["8"] = min(group["scores"].values())
        group["unobserved_modules"] = [name for name, count in group["forward_calls"].items() if not count]
        group["all_scores_zero"] = all(value == 0 for value in group["scores"].values())
        group["individual_expert_coverage"] = "per-module forward observation"
        if len(group["unobserved_modules"]) == len(group["module_names"]):
            raise ValueError(f"Entire global scoring group was never executed: {key}")
        if group["unobserved_modules"] or group["all_scores_zero"]:
            uncertain.append(key)
        allowed = group["allowed_bits"]
        table[key] = {"formats": allowed, "scores": [group["scores"][str(bit)] for bit in allowed],
                      "costs": [group["weight_elements"] * bit / 16 for bit in allowed]}
    total = inventory["weight_elements"]
    solver = AutoQuantizeGradientSearcher()
    solver.candidate_stats = table
    selected, satisfied = solver.run_search_with_stats(total * effective_bits / 16)
    if not satisfied:
        raise ValueError("Global EP precision search did not satisfy the requested budget")
    bits = {}
    for key, selection in selected.items():
        groups[key]["selected_bits"] = int(selection["format"])
        bits.update(dict.fromkeys(groups[key]["module_names"], int(selection["format"])))
    actual = sum(g["weight_elements"] * g["selected_bits"] for g in groups.values()
                 if g["status"] == "searched") / total
    result = copy.deepcopy(plans[0])
    result.pop("_rank_local_scores")
    for key in ("modules", "weight_parameters", "excluded_modules", "non_linear_modules", "unmatched_exclusion_rules"):
        result[key] = copy.deepcopy(inventory[key])
    result.update(module_bits=dict(sorted(bits.items())), groups=groups,
                  effective_bits_target=effective_bits, actual_effective_bits=actual)
    result["policy"]["grouping_rules"] = inventory["grouping_rules"]
    result["cost"].update(weight_elements=total, weight_payload_bytes=total * actual / 8)
    result["coverage"].update(complete=True, uncertain_groups=uncertain)
    return validate_plan(result)


def _broadcast_result(result, world_size):
    value = [result]
    if world_size > 1:
        dist.broadcast_object_list(value, src=0)
    if value[0].get("error"):
        raise ValueError(value[0]["error"])
    return value[0]


def _bound_glibc_heap():
    """Return freed decode temporaries to the OS instead of retaining them per malloc arena.

    Weight calibration decodes every local expert matrix on CPU. glibc's dynamic mmap
    threshold places these multi-MB temporaries on arena heaps that are never trimmed,
    so Kimi EP4 anonymous RSS grew ~12 MB per module without any live tensor growth.
    """
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return  # Not glibc; its allocator has different retention behavior.
    libc.mallopt(-3, int(os.environ.get("MALLOC_MMAP_THRESHOLD_", 1 << 20)))  # M_MMAP_THRESHOLD
    libc.mallopt(-1, int(os.environ.get("MALLOC_TRIM_THRESHOLD_", 64 << 20)))  # M_TRIM_THRESHOLD


def analyze_ep(args, policy):
    """Torchrun entry. Runtime adapters control execution; only rank zero returns a plan."""
    from .calibration import load_calibration
    from transformers import AutoTokenizer

    _bound_glibc_heap()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cpu = args.device == "cpu"
    device = torch.device("cpu" if cpu else f"cuda:{local_rank}")
    if not cpu:
        if not torch.cuda.is_available():
            raise ValueError("EP requires accessible CUDA/HIP devices; use --device cpu for tiny tests or dry-run")
        torch.cuda.set_device(device)
    torch.set_num_threads(args.num_threads)
    owns_group = world_size > 1 and not dist.is_initialized()
    if owns_group:
        dist.init_process_group("gloo" if cpu else "nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank() if dist.is_initialized() else 0
    started = time.perf_counter()
    runtime = None
    try:
        model_type = json.loads((Path(args.model) / "config.json").read_text()).get("model_type")
        runtime_class = get_ep_runtime(model_type)
        runtime = runtime_class(args.model, device=device, dtype=getattr(torch, args.dtype), rank=rank,
                              world_size=world_size, trust_remote_code=args.trust_remote_code)
        exclude = [*policy.get("exclude", []), *args.exclude]
        include = [*policy.get("include", []), *args.include]
        grouping_rules = policy.get("grouping_rules", [])
        fingerprint = hashlib.sha256(json.dumps({"config": runtime.source_config,
            "headers": runtime.reader.headers, "weight_map": runtime.reader.weight_map}, sort_keys=True).encode()).hexdigest()
        if world_size > 1:
            fingerprints = [None] * world_size
            dist.all_gather_object(fingerprints, fingerprint)
            if len(set(fingerprints)) != 1:
                raise ValueError("EP ranks opened different source checkpoints")
        setup = None
        inventory = None
        if rank == 0:
            try:
                inventory = runtime.global_inventory(exclude=exclude, include=include, grouping_rules=grouping_rules)
                if args.effective_bits + 1e-9 < inventory["minimum_effective_bits"]:
                    raise ValueError(f"Global budget infeasible; minimum={inventory['minimum_effective_bits']:.6f}")
                # Propagate exclusions through the full fusion group, including experts on other ranks.
                setup = {"error": None, "excluded_modules": list(inventory["excluded_modules"])}
                if not args.dry_run:
                    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, local_files_only=True)
                    batches = load_calibration(args.calibration_data, tokenizer, samples=args.calibration_samples,
                        sequence_length=args.calibration_seq_length, split=args.dataset_split, text_field=args.text_field)
                    if len(batches) < world_size or len(batches) % world_size:
                        raise ValueError("Actual calibration sequence count must be a positive multiple of EP world size")
                    setup["batches"] = batches
            except Exception as error:
                setup = {"error": str(error)}
        setup = _broadcast_result(setup, world_size)
        block_bytes = [sum(p.numel() * p.element_size() for p in block.parameters()) for block in runtime.model.model.layers]
        print(f"[EP rank {rank}] device={device}; experts/layer={runtime.experts_per_rank}; "
              f"largest local source block={max(block_bytes) / 1024**3:.3f} GiB (excludes scoring temporaries)", flush=True)
        if args.dry_run:
            if rank == 0:
                print(f"DRY RUN OK: EP{world_size}; {len(inventory['modules'])} global targets; "
                      f"searched unique weights={inventory['weight_elements']:,}; no source weights decoded or outputs written")
            return None
        batches = setup["batches"]
        with runtime:
            local = analyze_model(runtime.model, batches[rank::world_size], effective_bits=args.effective_bits,
                exclude=[*exclude, *setup["excluded_modules"]], include=include, grouping_rules=grouping_rules,
                adapter=runtime.adapter, source_config=runtime.source_config, model_id=str(Path(args.model).resolve()),
                forward_step=runtime.forward, seed=args.seed, verbose=args.verbose, _runtime=runtime)
        local["_block_loads"] = runtime.block_loads
        received = [None] * world_size if rank == 0 else None
        if world_size > 1:
            dist.gather_object(local, received, dst=0)
        else:
            received = [local]
        outcome = None
        plan = None
        if rank == 0:
            try:
                plan = merge_ep_plans(received, inventory, effective_bits=args.effective_bits, owners=runtime.owners)
                plan.pop("_block_loads", None)
                plan["calibration"] = {**_calibration_metadata(batches), "seed": args.seed,
                    "data": args.calibration_data, "requested_samples": args.calibration_samples,
                    "max_sequence_length": args.calibration_seq_length}
                plan["model"]["source_fingerprint"] = fingerprint
                quant = (runtime.source_config.get("quantization_config")
                         or runtime.source_config.get("text_config", {}).get("quantization_config") or {})
                plan["model"]["source_loading"] = {"loader": runtime.loader_name,
                    "source_quant_method": quant.get("quant_method")}
                plan["execution"] = {"strategy": "expert_parallel", "world_size": world_size,
                    "experts_per_rank": runtime.experts_per_rank, "block_residency": True,
                    "block_loads_by_rank": [item["_block_loads"] for item in received],
                    "gradient_loss_normalization": "sum_of_independent_sequence_mean_cross_entropies",
                    "elapsed_seconds": time.perf_counter() - started}
                outcome = {"error": None}
            except Exception as error:
                outcome = {"error": str(error)}
        _broadcast_result(outcome, world_size)
        return plan
    finally:
        if runtime is not None:
            runtime.decoded.clear()
            runtime.reader.close()
        if owns_group:
            dist.destroy_process_group()
        gc.collect()
