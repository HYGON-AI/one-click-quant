"""Human-readable candidate sensitivity rankings, without rerunning ModelOpt."""

import math
import json
from pathlib import Path


REPORT_TOP_K = 10
FIELD_DESCRIPTIONS = {
    "rank": "按 W4A8 敏感度分数降序排列的位置，不代表最终升级 INT8 的顺序。",
    "group_key": "完整计划 groups 中的键；组成员见 plan.groups[group_key].module_names。",
    "group": "统一选择位宽的决策组；一行不是一个独立专家的分数。",
    "int4_score": "W4A8 的累计梯度敏感度代理值；越大表示当前校准数据下扰动越敏感，不是实测 loss/准确率下降。",
    "int8_score": "W8A8 的对应代理值；存储值可能经过单调包络处理。",
    "upgrade_gain": "int4_score - int8_score；升级 W8A8 的代理收益，不是实际精度提升。",
    "selected_bits": "全局预算和分组约束下最终选择的权重位宽；激活仍为动态 per-token INT8。",
    "weight_elements": "该组计入预算的唯一权重元素数，不是字节数，也不是整个模型的参数数。",
    "module_count": "组内目标模块/投影数；Kimi 的一个专家含 w1/w2/w3，不能直接当作专家数。",
    "allowed_bits": "该组允许的候选位宽；后端、融合或打包约束可能限制选择。",
    "coverage_flags": "unobserved=N 表示 N 个目标模块未观测到前向；zero_scores/覆盖未知不等于可以安全 INT4。",
}


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
    for row in rows:
        flags = list(row["coverage_flags"])
        if set(row["allowed_bits"]) != {4, 8}:
            flags.append("allowed=" + "/".join(map(str, row["allowed_bits"])))
        values = [str(row["rank"]), row["group"], f"{row['int4_score']:.6e}", f"{row['int8_score']:.6e}",
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


def format_sensitivity_report(plan, *, top_k=REPORT_TOP_K):
    """At most ten decision groups plus field definitions; full data stays in the plan."""
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    top_k = min(top_k or REPORT_TOP_K, REPORT_TOP_K)
    rows = sensitivity_ranking(plan)
    if not rows:
        raise ValueError("Plan has no sensitivity scores")
    model = plan.get("model", {})
    model_id = model.get("model_id") if isinstance(model, dict) else model
    method = plan.get("method", "unknown")
    lines = ["=" * 100, "AUTOQUANTIZE SENSITIVITY REPORT", "=" * 100,
             f"Base model: {model_id or plan.get('source_model') or 'not recorded'}",
             f"Analysis: {method}", f"Ranked fusion groups: {len(rows)}; assigned modules: {len(plan['module_bits'])}",
             f"Showing top {min(top_k, len(rows))} of {len(rows)} groups; full decisions/members remain in plan JSON.",
             f"Effective bits: target={plan.get('effective_bits_target', 'unknown')}, "
             f"actual={plan.get('actual_effective_bits', 'unknown')}",
             "", "-- INT4 Quantization Sensitivity (W4A8 score descending) --",
             _table(rows[:top_k])]
    lines += ["", "字段说明 / Field definitions:"]
    for label, key in (("Rank", "rank"), ("Group", "group"), ("W4A8 score", "int4_score"),
                       ("W8A8 score", "int8_score"), ("S4-S8", "upgrade_gain"),
                       ("Bit", "selected_bits"), ("Params", "weight_elements"),
                       ("Members", "module_count"), ("Coverage / constraints", "coverage_flags"),
                       ("allowed", "allowed_bits")):
        lines.append(f"- {label}: {FIELD_DESCRIPTIONS[key]}")
    lines += ["", "Interpretation:",
              "- W4A8/W8A8 scores are accumulated candidate sensitivity proxies, not measured loss deltas.",
              "- Ranking uses the stored candidate scores; ModelOpt may apply a monotonic envelope to INT8 scores.",
              "- S4-S8 is the decrease in sensitivity score when changing W4A8 to W8A8, not measured loss or accuracy improvement.",
              "- Global ILP uses ALL decision groups, including groups omitted from this top-10 report.",
              "- It minimizes the total candidate score subject to effective_bits and fusion constraints.",
              "- Effective bits = sum(unique_weight_elements * selected_bits) / sum(unique_weight_elements).",
              "- This budget excludes scales, bias, excluded weights, packing padding and runtime buffers.",
              "- A selected INT8 group need not be rank 1 in the S4 sensitivity ranking.",
              "- One row is one decision group. Members share a bit; no individual member score is inferred.",
              "- Zero scores or unobserved experts do not establish low sensitivity. Unknown legacy parameter counts stay unknown.",
              "- Scores depend on calibration data and loss normalization; compare within the same analysis run."]
    if method == "modelopt_gradient":
        lines += ["- Gradient metric: sum(clamp(dL/dY * (Y_candidate - Y_float), -1e10, 1e10)^2).",
                  "- Y_float uses the source checkpoint decoded to the analysis dtype, not a recovered pre-quantization model.",
                  "- dL/dY comes from floating-baseline backprop; local candidate replay uses no_grad.",
                  "- Weight Fisher Mean/Max are not computed by this metric."]
    lines += ["", "Plan fields: module_bits = assignments; excluded_modules = exclusions; policy = quantization rules.",
              "Full group membership and scores: plan.groups[group_key]. No member lists are expanded here."]
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
    """Save ten compact ranking rows; retain full membership only in the precision plan."""
    rows = sensitivity_ranking(plan)
    if not rows:
        raise ValueError("Plan has no sensitivity scores")
    payload = {"method": plan.get("method"), "model": plan.get("model"),
               "sort_by": "int4_score_descending", "ranking_unit": "fusion_group",
               "top_k": REPORT_TOP_K, "total_groups": len(rows),
               "reported_groups": min(REPORT_TOP_K, len(rows)), "truncated": len(rows) > REPORT_TOP_K,
               "effective_bits_target": plan.get("effective_bits_target"),
               "actual_effective_bits": plan.get("actual_effective_bits"),
               "field_descriptions": FIELD_DESCRIPTIONS,
               "ranking": [{k: v for k, v in row.items() if k != "module_names"}
                           for row in rows[:REPORT_TOP_K]]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path
