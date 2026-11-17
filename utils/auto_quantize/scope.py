"""Discover weight targets and propagate exclusions through runtime fusion groups."""

from dataclasses import dataclass, field
import fnmatch
import inspect
import re

import torch
from torch import nn


EXCLUSION_KEYS = {
    "ignore", "ignored_layers", "modules_to_not_convert", "modules_to_exclude",
    "exclude_modules", "skip_modules", "llm_int8_skip_modules", "keep_in_fp32_modules",
}
DEFAULT_EXCLUDE = (
    "vision*", "*.vision*", "visual*", "*.visual*", "audio*", "*.audio*",
    "speech*", "*.speech*", "mtp*", "*.mtp*",
)


@dataclass
class WeightTarget:
    """One consumer-visible weight operation; backend_name identifies its ModelOpt module."""

    name: str
    backend_name: str
    parameters: dict[str, torch.Tensor]
    kind: str = "linear"
    channel_axis: int = 0
    input_axis: int = -1

    def metadata(self):
        return {
            "backend_name": self.backend_name, "kind": self.kind,
            "channel_axis": self.channel_axis, "input_axis": self.input_axis,
            "parameters": {n: {"shape": list(t.shape), "dtype": str(t.dtype)}
                           for n, t in self.parameters.items()},
        }


class ModelAdapter:
    """Override discovery for custom modules already supported by ModelOpt quantization."""

    grouping_rules = ()
    exclude_patterns = ()

    def group_name(self, names):
        return min(names)

    def allowed_bits(self, target):
        """Override for backend-specific tile, packing and operator restrictions."""
        return (4, 8) if all(t.shape[target.input_axis] % 8 == 0 for t in target.parameters.values()) else (8,)

    def aliases_for_target(self, name):
        aliases = {name}
        if re.search(r"\.(q_proj|k_proj|v_proj)$", name):
            aliases.add(re.sub(r"(q_proj|k_proj|v_proj)$", "qkv_proj", name))
        if re.search(r"\.(gate_proj|up_proj)$", name):
            aliases.add(re.sub(r"(gate_proj|up_proj)$", "gate_up_proj", name))
        return aliases

    def targets_for_module(self, name, module):
        if isinstance(module, nn.Linear):
            return [WeightTarget(name, name, {name + ".weight": module.weight})]
        params = dict(module.named_parameters(recurse=False))
        # Standard HF fused experts: per-expert channel quantization of [E,N,K].
        first = "gate_up_proj" if "gate_up_proj" in params else "up_proj"
        if (first in params and "down_proj" in params
                and params[first].ndim == params["down_proj"].ndim == 3
                and params[first].shape[0] == params["down_proj"].shape[0]):
            return [WeightTarget(f"{name}.{key}", name, {f"{name}.{key}": params[key]},
                                 "fused_expert_projection", channel_axis=1) for key in (first, "down_proj")]
        return None


def matches(name, pattern, module=None):
    if pattern.startswith("re:"):
        return re.match(pattern[3:], name) is not None
    if any(c in pattern for c in "*?["):
        return fnmatch.fnmatchcase(name, pattern)
    return (name == pattern or name.startswith(pattern + ".")
            or name.endswith("." + pattern)
            or (module is not None and pattern in {c.__name__ for c in type(module).__mro__}))


def exclusion_rules(config):
    """Read named exclusion fields, including nested text/quantization configs."""
    rules = []

    def visit(value, path):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            location = f"{path}.{key}"
            if key in EXCLUSION_KEYS and item is not None:
                if isinstance(item, str):
                    item = [item]
                if not isinstance(item, (list, tuple, set)) or any(not isinstance(x, str) for x in item):
                    raise ValueError(f"Expected string exclusion patterns at {location}")
                rules.extend((pattern, location) for pattern in item)
            elif isinstance(item, dict):
                visit(item, location)

    visit(config, "config")
    return rules


def _group_key(rule, model, name):
    if isinstance(rule, str):
        pattern = re.compile(rule)
        if not pattern.groups:
            raise ValueError(f"Grouping regex needs a capture group: {rule}")
        match = pattern.match(name)
        return match.group(1) if match else None
    signature = inspect.signature(rule)
    try:
        signature.bind(model, name)
    except TypeError:
        return rule(name)
    return rule(model, name)


@dataclass
class Scope:
    targets: dict
    groups: dict
    excluded: dict
    non_linear: dict
    backend_targets: dict
    backend_groups: dict
    rules: list
    unmatched_rules: list = field(default_factory=list)


def discover(model, config, adapter=None, exclude=(), grouping_rules=(), include=(), *, allow_meta=False):
    from modelopt.torch.quantization.algorithms import AutoQuantizeGradientSearcher

    adapter = adapter or ModelAdapter()
    rules = (exclusion_rules(config) + [(p, "policy.exclude") for p in exclude]
             + [(p, "default.text_scope") for p in DEFAULT_EXCLUDE]
             + [(p, "adapter.exclude") for p in adapter.exclude_patterns])
    for attr in ("_keep_in_fp32_modules", "_keep_in_fp32_modules_strict"):
        rules.extend((p, f"model.{attr}") for p in (getattr(model, attr, None) or []))
    grouping = [*grouping_rules, *adapter.grouping_rules,
                *AutoQuantizeGradientSearcher.quant_grouping_rules]
    # Validate rules even if this model has no matching module.
    for rule in grouping:
        if isinstance(rule, str) and not re.compile(rule).groups:
            raise ValueError(f"Grouping regex needs a capture group: {rule}")
    for pattern, _ in rules:
        if pattern.startswith("re:"):
            re.compile(pattern[3:])
    modules = dict(model.named_modules(remove_duplicate=False))
    canonical = {}
    targets, excluded, non_linear, backend_targets = {}, {}, {}, {}
    matched = set()
    for name, module in modules.items():
        if not name:
            if any(t.ndim >= 2 for t in module.parameters(recurse=False)):
                raise ValueError("Root-level matrix parameters need an explicit named-module adapter/wrapper")
            continue
        owner = canonical.setdefault(id(module), name)
        found = adapter.targets_for_module(name, module)
        reasons = []
        for index, (pattern, origin) in enumerate(rules):
            if matches(name, pattern, module):
                reasons.append(f"{origin}: {pattern}")
                matched.add(index)
        if include and not any(matches(name, p, module) for p in include):
            reasons.append("policy.include: outside requested scope")
        parent_name, _, child = name.rpartition(".")
        parent = modules.get(parent_name, model)
        if child in ("gate", "router") and hasattr(parent, "experts"):
            reasons.append("structural_router: selects experts")
        if found is None:
            matrix = {n: t for n, t in module.named_parameters(recurse=False) if t.ndim >= 2}
            non_linear_op = isinstance(module, (nn.Embedding, nn.EmbeddingBag, nn.Conv1d,
                                                nn.Conv2d, nn.Conv3d, nn.Bilinear))
            if matrix and not non_linear_op and not reasons:
                raise ValueError(f"Unsupported matrix module {name} ({type(module).__name__}); "
                                 "provide a ModelAdapter/ModelOpt plugin or explicitly exclude it")
            if list(module.parameters(recurse=False)):
                non_linear[name] = {"type": type(module).__name__,
                                    "reasons": reasons or ["not a quantizable Linear weight"]}
            continue
        for target in found:
            target.backend_name = owner
            if target.name in targets or not target.parameters:
                raise ValueError(f"Duplicate or empty target: {target.name}")
            for param_name, tensor in target.parameters.items():
                if (tensor.is_meta and not allow_meta) or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError(f"{param_name} must be a materialized FP16/BF16/FP32 tensor; "
                                     "decode packed/FP8 checkpoints before analysis")
            targets[target.name] = target
            backend_targets.setdefault(owner, []).append(target.name)
            own = []
            aliases = set(adapter.aliases_for_target(target.name))
            aliases |= set(target.parameters) | {alias + ".weight" for alias in aliases}
            for index, (pattern, origin) in enumerate(rules):
                if any(matches(alias, pattern, module) for alias in aliases):
                    own.append(f"{origin}: {pattern}")
                    matched.add(index)
            if reasons or own:
                excluded[target.name] = sorted(set(reasons + own))
    if not targets:
        raise ValueError("No quantizable Linear/expert weights found")

    # Union native fusion groups, fused expert projections, module aliases and tied weights.
    parent = {n: n for n in targets}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def join(a, b):
        parent[find(b)] = find(a)

    by_key, by_parameter = {}, {}
    for owner, names in backend_targets.items():
        key = owner
        for rule in grouping:
            result = _group_key(rule, model, owner)
            if result is not None:
                key = result
                break
        for name in names:
            if key in by_key:
                join(by_key[key], name)
            by_key[key] = name
            for tensor in targets[name].parameters.values():
                if id(tensor) in by_parameter:
                    join(by_parameter[id(tensor)], name)
                by_parameter[id(tensor)] = name
    groups = {}
    for name in targets:
        groups.setdefault(find(name), []).append(name)
    named_groups = {}
    for names in groups.values():
        key = adapter.group_name(names)
        if key in named_groups:
            raise ValueError(f"Adapter produced duplicate group name: {key}")
        named_groups[key] = sorted(names)
    groups = named_groups
    backend_groups = {}
    for key, names in groups.items():
        reasons = sorted({r for n in names for r in excluded.get(n, [])})
        if reasons:
            for n in names:
                excluded.setdefault(n, []).extend([f"fusion_group {key}: {r}" for r in reasons
                                                  if r not in excluded.get(n, [])])
        for n in names:
            backend_groups[targets[n].backend_name] = key
    return Scope(targets, groups, excluded, non_linear, backend_targets, backend_groups,
                 [r if isinstance(r, str) else f"{r.__module__}.{r.__qualname__}" for r in grouping],
                 [{"pattern": p, "origin": o} for i, (p, o) in enumerate(rules)
                  if i not in matched and not o.startswith("default.")])
