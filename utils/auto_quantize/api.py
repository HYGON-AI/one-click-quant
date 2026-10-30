"""Score loaded PyTorch models and export backend-independent precision decisions."""

from contextlib import contextmanager
import functools
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import tempfile
import threading

import torch
from torch.nn import functional as F

from .scope import ModelAdapter, discover


_SEARCH_LOCK = threading.RLock()
SCHEMA = "one-click-quant.auto_quantize"


def channel_candidate(bits):
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer*",
             "cfg": {"num_bits": bits, "axis": 0, "narrow_range": bits == 8}},
            {"quantizer_name": "*input_quantizer",
             "cfg": {"num_bits": 8, "block_sizes": {-1: None}, "type": "dynamic", "narrow_range": True}},
        ],
        "algorithm": "max",
    }


def _disabled_weight_modules(model, active_backends):
    """Only matrix owners can become search dimensions; containers need no rules.

    ModelOpt scans disabled patterns for every candidate module. Passing every
    container/activation makes that scan quadratic and thrashes Python 3.10's
    256-entry fnmatch cache on large expert models.
    """
    return sorted(name for name, module in model.named_modules()
                  if name not in active_backends
                  and any(parameter.ndim >= 2 for parameter in module.parameters(recurse=False)))


def _input_device(model):
    if hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()
        if embedding is not None:
            return embedding.weight.device
    return next(model.parameters()).device


def causal_forward(model, batch):
    device = _input_device(model)
    inputs = {"input_ids": batch} if isinstance(batch, torch.Tensor) else dict(batch)
    inputs.pop("labels", None)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs["use_cache"] = False
    return model(**inputs).logits


def causal_loss(logits, batch):
    labels = batch if isinstance(batch, torch.Tensor) else batch["labels"] if "labels" in batch else batch["input_ids"]
    labels = labels[:, 1:].to(logits.device).clone()
    if isinstance(batch, dict) and "attention_mask" in batch:
        labels.masked_fill_(~batch["attention_mask"][:, 1:].to(logits.device).bool(), -100)
    if not (labels != -100).any():
        raise ValueError("Calibration batch has no valid next-token targets")
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels.reshape(-1))


@contextmanager
def _runtime_groups(scope):
    from modelopt.torch.quantization.algorithms import AutoQuantizeGradientSearcher

    original = AutoQuantizeGradientSearcher.quant_grouping_rules
    # *args supports ModelOpt versions calling a rule with (model, name) or (name,).
    AutoQuantizeGradientSearcher.quant_grouping_rules = [
        lambda *args: scope.backend_groups.get(args[-1], args[-1])
    ]
    try:
        yield
    finally:
        AutoQuantizeGradientSearcher.quant_grouping_rules = original


@contextmanager
def _unbounded_fnmatch_cache():
    # ModelOpt matches every quant module against every disabled pattern. Kimi EP4 has
    # 1463 patterns, beyond Python 3.10's 256-entry cache, so each match recompiled a regex.
    import fnmatch

    original = fnmatch._compile_pattern
    fnmatch._compile_pattern = functools.lru_cache(maxsize=None)(getattr(original, "__wrapped__", original))
    try:
        yield
    finally:
        fnmatch._compile_pattern = original


@contextmanager
def _cpu_scoring(model):
    # Optional accelerator plugins can report devices even when the model is CPU-only.
    original = torch.cuda.is_available
    cpu_only = all(t.device.type == "cpu" for t in model.parameters())
    if hasattr(model, "_autoquantize_execution_device"):
        cpu_only = torch.device(model._autoquantize_execution_device).type == "cpu"
    if cpu_only:
        torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = original


def _calibration_metadata(batches):
    digest = hashlib.sha256()
    token_count = 0

    def visit(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(key.encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())

    for batch in batches:
        visit(batch)
        if isinstance(batch, torch.Tensor) and batch.dtype in (torch.int32, torch.int64) and batch.ndim == 2:
            token_count += batch.numel()
        elif isinstance(batch, dict) and "input_ids" in batch:
            token_count += int(batch["attention_mask"].sum()) if "attention_mask" in batch else batch["input_ids"].numel()
    return {"batches": len(batches), "input_tokens": token_count, "batch_sha256": digest.hexdigest()}


def _inventory(model, scope, adapter):
    """Serialize source shapes and de-duplicate weights without materializing tensors."""
    searched = set(scope.targets) - set(scope.excluded)
    parameter_names = list(model.named_parameters(remove_duplicate=False))
    canonical_parameters = {id(t): n for n, t in reversed(parameter_names)}
    aliases = {}
    for name, tensor in parameter_names:
        aliases.setdefault(id(tensor), []).append(name)
    weights, modules = {}, {}
    for name, target in scope.targets.items():
        info = target.metadata()
        info["status"] = "excluded" if name in scope.excluded else "searched"
        info["exclusion_reasons"] = scope.excluded.get(name, [])
        for param_name, tensor in target.parameters.items():
            key = canonical_parameters[id(tensor)]
            info["parameters"][param_name]["cost_key"] = key
            weights[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "numel": tensor.numel(), "aliases": aliases[id(tensor)]}
        modules[name] = info
    groups = {}
    for key, names in scope.groups.items():
        params = sorted({p["cost_key"] for n in names for p in modules[n]["parameters"].values()})
        group = {"module_names": names, "module_count": len(names), "weight_parameters": params,
                 "weight_elements": sum(weights[n]["numel"] for n in params)}
        group["status"] = "searched" if names[0] in searched else "excluded"
        allowed = {4, 8}
        for name in names:
            choices = set(adapter.allowed_bits(scope.targets[name]))
            if not choices or not choices <= {4, 8}:
                raise ValueError(f"Adapter must return non-empty INT4/INT8 choices: {name}")
            allowed &= choices
        if not allowed:
            raise ValueError(f"Backend candidate restrictions conflict within fusion group: {key}")
        group["allowed_bits"] = sorted(allowed)
        groups[key] = group
    total = sum(g["weight_elements"] for g in groups.values() if g["status"] == "searched")
    if not total:
        raise ValueError("All discovered weight targets are excluded")
    minimum = sum(g["weight_elements"] * min(g["allowed_bits"]) for g in groups.values()
                  if g["status"] == "searched") / total
    return weights, modules, groups, total, minimum


def analyze_model(model, data_loader, *, effective_bits=4.2, exclude=(), include=(),
                  grouping_rules=(), adapter=None, source_config=None, model_id=None,
                  forward_step=None, loss_func=None, calibration=None, seed=42,
                  output_path=None, verbose=False, _runtime=None):
    """Return a precision plan; ModelOpt converts this analysis model in place.

    data_loader must be repeatable or fit in memory. Default inputs/loss are for
    causal text models; other tasks can supply forward_step and loss_func.
    Excluded weights retain their source dtype and are absent from module_bits.
    """
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.algorithms import AutoQuantizeGradientSearcher

    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"Plan already exists: {output_path}")
    if not 4 <= effective_bits <= 8 or not math.isfinite(effective_bits):
        raise ValueError("INT4/INT8 effective_bits must be finite and between 4 and 8")
    if torch.distributed.is_initialized() and _runtime is None:
        raise ValueError("Generic analysis is single-process; distributed scoring needs a runtime adapter")
    if _runtime is not None and (output_path is not None or model is not _runtime.model):
        raise ValueError("Streaming workers return rank-local statistics; only the global owner saves a plan")
    if set(calibration or {}) & {"batches", "input_tokens", "batch_sha256", "seed"}:
        raise ValueError("Calibration metadata must not override measured counts, input digest or scoring seed")
    exclude = [exclude] if isinstance(exclude, str) else list(exclude)
    include = [include] if isinstance(include, str) else list(include)
    grouping_rules = [grouping_rules] if isinstance(grouping_rules, str) else list(grouping_rules)
    if any(not isinstance(p, str) for p in exclude + include):
        raise ValueError("include/exclude patterns must be strings")
    batches = list(data_loader)
    if not batches:
        raise ValueError("Calibration data is empty")
    config = source_config if source_config is not None else getattr(model, "config", {})
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    config = json.loads(json.dumps(config, default=lambda value: value.to_dict()
                                  if hasattr(value, "to_dict") else str(value)))
    with _SEARCH_LOCK:
        adapter = adapter or ModelAdapter()
        scope = discover(model, config, adapter, exclude, grouping_rules, include, allow_meta=_runtime is not None)
        searched = set(scope.targets) - set(scope.excluded)
        if not searched:
            raise ValueError("All discovered weight targets are excluded")
        weights, modules, groups, total, minimum = _inventory(model, scope, adapter)
        if _runtime is None and effective_bits + 1e-9 < minimum:
            raise ValueError(f"Budget is infeasible with packing/fusion constraints; minimum effective_bits={minimum:.6f}")
        if _runtime is not None:
            _runtime.verify_scope(scope, groups)
        active_backends = {scope.targets[n].backend_name for n in searched}
        disabled = _disabled_weight_modules(model, active_backends)
        print(f"AutoQuantize: {len(searched)} searched targets, {len(groups)} groups, "
              f"{len(disabled)} excluded matrix modules; initializing ModelOpt", flush=True)
        observed = dict.fromkeys(scope.backend_targets, 0)

        def hook(name):
            def count(*_):
                observed[name] += 1
            return count

        handles = [model.get_submodule(n).register_forward_hook(hook(n)) for n in active_backends]
        formats = [channel_candidate(4), channel_candidate(8)]
        if _runtime is not None:
            for recipe in formats:
                recipe["algorithm"] = {"method": "max", "skip_forward_without_activation_calib": True}
        kwargs = {}
        if "module_search_spaces" in inspect.signature(mtq.auto_quantize).parameters:
            kwargs["module_search_spaces"] = [{"module_name_patterns": ["*"],
                                               "quantization_formats": formats, "allow_no_quant": False}]
        random.seed(seed)
        torch.manual_seed(seed)
        model.eval()
        try:
            with _runtime_groups(scope), _cpu_scoring(model), _unbounded_fnmatch_cache():
                _, state = mtq.auto_quantize(
                    model, constraints={"effective_bits": 8 if _runtime is not None else effective_bits,
                                        "cost": {"excluded_module_name_patterns": disabled}},
                    quantization_formats=formats, disabled_layers=disabled,
                    data_loader=batches, forward_step=forward_step or causal_forward,
                    loss_func=loss_func or causal_loss, num_calib_steps=len(batches),
                    num_score_steps=len(batches), method="gradient", verbose=verbose, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        table, covered = {}, set()
        if _runtime is not None:
            from modelopt.torch.opt.utils import named_hparams
            hparams = dict(named_hparams(model, unique=True))
        for hparam_name, stats in state["candidate_stats"].items():
            backends = stats["module_names"]
            names = {n for owner in backends for n in scope.backend_targets.get(owner, [])}
            if not names & searched:
                if not names and any(float(c) > 0 for c in stats["costs"]):
                    raise ValueError(f"ModelOpt searched unrecognized modules: {backends}")
                continue
            key = scope.backend_groups[next(owner for owner in backends if owner in active_backends)]
            if names != set(groups[key]["module_names"]) or names & covered or not names <= searched:
                raise ValueError(f"ModelOpt fusion/coverage differs from discovered scope: {backends}")
            source_scores = ([hparams[hparam_name].get_score(recipe) for recipe in stats["formats"]]
                             if _runtime is not None else stats["scores"])
            scores = {int(fmt.num_bits): float(score) for fmt, score in zip(stats["formats"], source_scores)
                      if int(fmt.num_bits) in (4, 8)}
            if set(scores) != {4, 8} or any(not math.isfinite(s) or s < 0 for s in scores.values()):
                raise ValueError(f"Missing/non-finite candidate sensitivity scores: {key}")
            choices = groups[key]["allowed_bits"]
            table[key] = {"formats": choices, "scores": [scores[b] for b in choices],
                          "costs": [groups[key]["weight_elements"] * b / 16 for b in choices]}
            groups[key]["scores"] = {str(b): scores[b] for b in (4, 8)}
            covered.update(names)
        if covered != searched:
            raise ValueError(f"Missing ModelOpt decisions for targets: {sorted(searched - covered)[:20]}")
        if _runtime is None:
            solver = AutoQuantizeGradientSearcher()
            solver.candidate_stats = table
            selected, satisfied = solver.run_search_with_stats(total * effective_bits / 16, verbose=verbose)
        else:
            selected = {key: {"format": value["formats"][0], "costs": value["costs"][0]}
                        for key, value in table.items()}
            satisfied = True
        actual = sum(v["costs"] for v in selected.values()) / total * 16
        if not satisfied or (_runtime is None and actual > effective_bits + 1e-7):
            raise ValueError("INT4/INT8 search did not satisfy the weight budget")
        module_bits, uncertain = {}, []
        for key, group in groups.items():
            names = group["module_names"]
            group["forward_calls"] = {n: observed[scope.targets[n].backend_name] for n in names}
            if group["status"] == "excluded":
                group["selected_bits"] = None
                continue
            group["selected_bits"] = int(selected[key]["format"])
            group["unobserved_modules"] = [n for n in names if not group["forward_calls"][n]]
            group["all_scores_zero"] = all(v == 0 for v in group["scores"].values())
            group["individual_expert_coverage"] = (
                "not_measured" if any(scope.targets[n].kind == "fused_expert_projection" for n in names)
                else "per-module forward observation")
            if (group["unobserved_modules"] or group["all_scores_zero"]
                    or group["individual_expert_coverage"] == "not_measured"):
                uncertain.append(key)
            if _runtime is None and len(group["unobserved_modules"]) == len(names):
                raise ValueError(f"Entire target group was never executed: {key}; check calibration coverage "
                                 "and whether fused kernels bypass the modules (use an eager/ModelOpt adapter)")
            module_bits.update(dict.fromkeys(names, group["selected_bits"]))
        metadata = _calibration_metadata(batches)
        metadata.update(calibration or {})
        metadata["seed"] = seed
        identity = {"model_id": model_id or getattr(getattr(model, "config", None), "_name_or_path", None),
                    "model_class": type(model).__name__, "model_type": config.get("model_type"),
                    "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
        plan = {
            "schema": SCHEMA, "schema_version": 1,
            "method": "modelopt_gradient", "modelopt_version": importlib.metadata.version("nvidia-modelopt"),
            "model": identity, "module_bits": dict(sorted(module_bits.items())), "modules": modules,
            "groups": groups, "weight_parameters": weights,
            "excluded_modules": scope.excluded, "non_linear_modules": scope.non_linear,
            "unmatched_exclusion_rules": scope.unmatched_rules,
            "policy": {"candidate_bits": [4, 8], "weight_granularity": "channel",
                       "activation_bits": 8, "activation_granularity": "dynamic_per_token",
                       "calibration_algorithm": "max_rtn", "grouping_rules": scope.rules},
            "effective_bits_target": effective_bits, "actual_effective_bits": actual,
            "cost": {"scope": "searched_unique_weight_parameters_only", "weight_elements": total,
                     "weight_payload_bytes": actual * total / 8,
                     "excludes": ["scales", "bias", "excluded weights", "packing padding", "runtime buffers"]},
            "calibration": metadata,
            "coverage": {"complete": True, "uncertain_groups": uncertain,
                         "forward_calls_include": "calibration, candidate replay and checkpoint recomputation",
                         "fused_experts": "parent execution does not prove coverage of every individual expert"},
        }
        if _runtime is not None:
            plan["_rank_local_scores"] = True
            return plan
        validate_plan(plan)
        if output_path is not None:
            save_plan(plan, output_path)
        return plan


def validate_plan(plan, model=None, *, adapter=None):
    """Validate consumer-facing coverage, fusion decisions, shapes and weight cost."""
    if plan.get("schema") != SCHEMA or plan.get("schema_version") != 1:
        raise ValueError("Not a supported generic AutoQuantize plan")
    if not math.isfinite(plan["effective_bits_target"]) or not 4 <= plan["effective_bits_target"] <= 8:
        raise ValueError("Invalid plan effective-bits budget")
    modules, bits, groups = plan["modules"], plan["module_bits"], plan["groups"]
    if any(info["status"] not in ("searched", "excluded") for info in modules.values()):
        raise ValueError("Unknown target status")
    for parameter in plan["weight_parameters"].values():
        if parameter["numel"] != math.prod(parameter["shape"]):
            raise ValueError("Parameter element count differs from its shape")
    expected = {n for n, info in modules.items() if info["status"] == "searched"}
    if set(bits) != expected or any(type(b) is not int or b not in (4, 8) for b in bits.values()):
        raise ValueError("module_bits must cover every searched target exactly with INT4/INT8")
    if set(plan["excluded_modules"]) != set(modules) - expected:
        raise ValueError("Every non-searched weight target needs an explicit exclusion")
    seen, parameters, weighted_bits = set(), set(), 0
    for group in groups.values():
        names = group["module_names"]
        if group["status"] not in ("searched", "excluded") or not names or seen & set(names):
            raise ValueError("Duplicate or empty fusion group")
        seen.update(names)
        if group["status"] == "searched":
            bit = group["selected_bits"]
            if set(group["scores"]) != {"4", "8"} or any(not math.isfinite(s) or s < 0 for s in group["scores"].values()):
                raise ValueError("Invalid candidate sensitivity scores")
            if any(bits.get(n) != bit for n in names) or bit not in group["allowed_bits"]:
                raise ValueError("Inconsistent or unsupported precision in fusion group")
            if parameters & set(group["weight_parameters"]):
                raise ValueError("Shared weight parameters were charged more than once")
            parameters.update(group["weight_parameters"])
            count = sum(plan["weight_parameters"][n]["numel"] for n in group["weight_parameters"])
            if count != group["weight_elements"]:
                raise ValueError("Fusion group weight cost differs from parameter inventory")
            weighted_bits += count * bit
        elif any(not plan["excluded_modules"].get(n) for n in names):
            raise ValueError("Excluded group lacks exclusion reasons")
    if seen != set(modules):
        raise ValueError("Fusion groups do not cover the complete weight inventory")
    count = sum(plan["weight_parameters"][n]["numel"] for n in parameters)
    if not count or not plan["coverage"]["complete"]:
        raise ValueError("Plan has no searched weights or incomplete coverage")
    actual = weighted_bits / count
    if (count != plan["cost"]["weight_elements"] or not math.isclose(actual, plan["actual_effective_bits"], abs_tol=1e-8)
            or not math.isclose(weighted_bits / 8, plan["cost"]["weight_payload_bytes"], abs_tol=1e-5)
            or actual > plan["effective_bits_target"] + 1e-7):
        raise ValueError("Plan weight cost/budget metadata is inconsistent")
    if model is not None:
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if model_type is not None and model_type != plan["model"]["model_type"]:
            raise ValueError("Plan/model architecture type mismatch")
        for name, info in modules.items():
            for param_name, param in info["parameters"].items():
                try:
                    tensor = model.get_parameter(param_name)
                except AttributeError as error:
                    raise ValueError(f"Plan parameter not present in model: {param_name}") from error
                if list(tensor.shape) != param["shape"]:
                    raise ValueError(f"Plan/model shape mismatch: {param_name}")
        # Catch a new Linear/expert branch even when all old parameter names still exist.
        known_owners = {info["backend_name"] for info in modules.values()}
        adapter = adapter or ModelAdapter()
        for name, module in model.named_modules(remove_duplicate=False):
            discovered = adapter.targets_for_module(name, module) if name else None
            if discovered and not all(target.name in modules for target in discovered):
                raise ValueError(f"Plan does not cover model weight operations: {name}")
            matrix = any(t.ndim >= 2 for t in module.parameters(recurse=False))
            if name and matrix and name not in known_owners and name not in plan["non_linear_modules"]:
                raise ValueError(f"Plan does not inventory matrix module: {name}")
    return plan


def save_plan(plan, path):
    validate_plan(plan)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Plan already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(plan, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.link(temporary, path)  # Exclusive creation; never replace an existing result.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_plan(path, model=None, *, adapter=None):
    return validate_plan(json.loads(Path(path).read_text()), model, adapter=adapter)
