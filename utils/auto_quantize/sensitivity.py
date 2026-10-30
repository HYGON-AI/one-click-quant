"""Human-readable candidate sensitivity rankings, without rerunning ModelOpt."""

import math
import json
from pathlib import Path


def sensitivity_path(plan_path):
    return Path(plan_path).with_suffix(".sensitivity.txt")


def _group_label(key, names):
    if len(names) == 1:
        return names[0]
    parts = [name.split(".") for name in names]
    common = []
    for values in zip(*parts):
        if len(set(values)) != 1:
            break
        common.append(values[0])
    return ".".join(common) or key


def sensitivity_ranking(plan):
    """One row per fusion group, sorted by the stored INT4 score, descending.

    Accepts the generic plan and historical MiMo v2 plans. Group scores cannot be
    split into individual member scores. Missing legacy parameter counts remain unknown.
    """
    rows = []
    seen = set()
    for key, group in plan["groups"].items():
        if group.get("status") == "excluded" or group.get("selected_bits") is None:
            continue
        names = list(group["module_names"])
        if not names or seen.intersection(names):
            raise ValueError(f"Empty or overlapping sensitivity group: {key}")
        seen.update(names)
        bit = group["selected_bits"]
        if bit not in (4, 8) or any(plan["module_bits"].get(name) != bit for name in names):
            raise ValueError(f"Sensitivity group disagrees with module_bits: {key}")
        scores = group.get("scores", {})
        if not {"4", "8"} <= scores.keys():
            raise ValueError(f"Missing INT4/INT8 sensitivity scores: {key}")
        s4, s8 = float(scores["4"]), float(scores["8"])
        if any(not math.isfinite(s) or s < 0 for s in (s4, s8)):
            raise ValueError(f"Invalid sensitivity scores: {key}")
        count = group.get("weight_elements")
        if count is not None and (type(count) is not int or count <= 0):
            raise ValueError(f"Invalid weight element count: {key}")
        flags = []
        if group.get("unobserved_modules"):
            flags.append(f"unobserved={len(group['unobserved_modules'])}")
        if s4 == s8 == 0:
            flags.append("zero_scores")
        if group.get("individual_expert_coverage") == "not_measured":
            flags.append("expert_coverage_unknown")
        allowed = group.get("allowed_bits", [4, 8])
        gain = s4 - s8
        rows.append({"group_key": key, "group": _group_label(key, names), "module_names": names,
                     "module_count": len(names), "int4_score": s4, "int8_score": s8,
                     "upgrade_gain": gain, "selected_bits": bit, "weight_elements": count,
                     "allowed_bits": allowed, "coverage_flags": flags})
    if seen != set(plan["module_bits"]):
        raise ValueError("Sensitivity groups do not cover every assigned module")
    rows.sort(key=lambda row: (-row["int4_score"], row["group_key"]))
    return [dict(rank=i, **row) for i, row in enumerate(rows, 1)]


def _table(rows):
    headers = ["Rank", "Group", "W4A8 score", "W8A8 score", "S4-S8", "Bit", "Params", "Members"]
    headers += ["Coverage / constraints"]
    data = []
    for i, row in enumerate(rows, 1):
        flags = list(row["coverage_flags"])
        if set(row["allowed_bits"]) != {4, 8}:
            flags.append("allowed=" + "/".join(map(str, row["allowed_bits"])))
        values = [str(i), row["group"], f"{row['int4_score']:.6e}", f"{row['int8_score']:.6e}",
                  f"{row['upgrade_gain']:.6e}", str(row["selected_bits"]),
                  f"{row['weight_elements']:,}" if row["weight_elements"] is not None else "unknown",
                  str(row["module_count"])]
        values += [", ".join(flags) or "-"]
        data.append(values)
    widths = [max(len(header), *(len(row[i]) for row in data)) for i, header in enumerate(headers)]
    return "\n".join([
        "  ".join(value.ljust(width) for value, width in zip(headers, widths)),
        "  ".join("-" * width for width in widths),
        *("  ".join(value.ljust(width) for value, width in zip(row, widths)) for row in data),
    ])


def format_sensitivity_report(plan, *, top_k=None):
    """Full report, or a compact top-k preview using the same ranking."""
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    rows = sensitivity_ranking(plan)
    if not rows:
        raise ValueError("Plan has no sensitivity scores")
    model = plan.get("model", {})
    model_id = model.get("model_id") if isinstance(model, dict) else model
    method = plan.get("method", "unknown")
    lines = ["=" * 100, "AUTOQUANTIZE SENSITIVITY REPORT", "=" * 100,
             f"Base model: {model_id or plan.get('source_model') or 'not recorded'}",
             f"Analysis: {method}", f"Ranked fusion groups: {len(rows)}; assigned modules: {len(plan['module_bits'])}",
             f"Effective bits: target={plan.get('effective_bits_target', 'unknown')}, "
             f"actual={plan.get('actual_effective_bits', 'unknown')}",
             "", "-- INT4 Quantization Sensitivity (W4A8 score descending) --",
             _table(rows[:top_k] if top_k else rows)]
    if top_k is not None:
        return "\n".join(lines) + "\n"
    lines += ["", "Interpretation:",
              "- W4A8/W8A8 scores are accumulated candidate sensitivity proxies, not measured loss deltas.",
              "- Ranking uses the stored candidate scores; ModelOpt may apply a monotonic envelope to INT8 scores.",
              "- S4-S8 is the decrease in sensitivity score when changing W4A8 to W8A8, not measured loss or accuracy improvement.",
              "- Global ILP minimizes the total candidate score subject to effective_bits and fusion constraints.",
              "- A selected INT8 group need not be rank 1 in the S4 sensitivity ranking.",
              "- One row is one decision group. Members share a bit; no individual member score is inferred.",
              "- Zero scores or unobserved experts do not establish low sensitivity. Unknown legacy parameter counts stay unknown.",
              "- Scores depend on calibration data and loss normalization; compare within the same analysis run."]
    if method == "modelopt_gradient":
        lines += ["- Gradient metric: sum(clamp(dL/dY * (Y_candidate - Y_float), -1e10, 1e10)^2).",
                  "- dL/dY comes from floating-baseline backprop; local candidate replay uses no_grad.",
                  "- Weight Fisher Mean/Max are not computed by this metric."]
    lines += ["", "Group membership (same order as the sensitivity ranking):"]
    for row in rows:
        lines += [f"[{row['rank']}] {row['group_key']}", *(f"  {name}" for name in row["module_names"])]
    return "\n".join(lines) + "\n"


def save_sensitivity_report(plan, path):
    """Write a separate report exclusively; do not alter or rescore the plan."""
    report = format_sensitivity_report(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(report)
    return path


def save_sensitivity_json(plan, path):
    """Save machine-readable ranking rows independently of the precision plan."""
    payload = {"method": plan.get("method"), "model": plan.get("model"),
               "sort_by": "int4_score_descending", "ranking_unit": "fusion_group",
               "ranking": sensitivity_ranking(plan)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path
