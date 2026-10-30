"""python -m utils.auto_quantize --model MODEL --calibration-data DATA --output PLAN.json"""

from argparse import ArgumentParser
import hashlib
import json
import math
import os
from pathlib import Path
import sys


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "utils.auto_quantize"


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local HF floating, block FP8 or MXFP4/FP8 checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New JSON precision-plan file")
    parser.add_argument("--calibration-data", required=True, help="txt/json/jsonl, saved HF dataset, or dataset ID")
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--calibration-seq-length", type=int, default=512)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-field")
    parser.add_argument("--effective-bits", type=float, default=4.2)
    parser.add_argument("--exclude", action="append", default=[], help="Keep matching modules floating; repeat for multiple patterns")
    parser.add_argument("--include", action="append", default=[], help="Search only matching modules; exclusions still win; default: all")
    parser.add_argument("--policy", type=Path, help="Reusable JSON containing include/exclude/grouping_rules; merged with CLI rules")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--parallel", choices=("single", "ep"), default="single", help="EP block-streaming runtime (MiMo/Kimi-K3); launch using torchrun")
    parser.add_argument("--num-threads", type=int, default=1, help="CPU threads per EP worker")
    parser.add_argument("--dry-run", action="store_true", help="EP metadata/topology preflight; do not decode source weights")
    args = parser.parse_args(argv)
    if args.num_threads < 1 or not math.isfinite(args.effective_bits) or not 4 <= args.effective_bits <= 8:
        parser.error("num-threads must be positive and effective-bits must be finite between 4 and 8")
    if args.dry_run and args.parallel != "ep":
        parser.error("--dry-run requires --parallel ep")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and args.parallel != "ep":
        parser.error("torchrun multi-process execution requires --parallel ep")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    from .sensitivity import sensitivity_path, format_sensitivity_report, save_sensitivity_report, save_sensitivity_json
    report_path = sensitivity_path(args.output)
    ranking_path = args.output.with_suffix(".sensitivity.json")
    if report_path.exists():
        parser.error(f"Sensitivity report already exists: {report_path}")
    if ranking_path.exists():
        parser.error(f"Sensitivity JSON already exists: {ranking_path}")
    if not (args.model / "config.json").is_file():
        parser.error("--model must be a local HuggingFace checkpoint directory")

    import torch
    from transformers import AutoTokenizer
    from .api import analyze_model, save_plan
    from .calibration import load_calibration
    from .loading import load_model

    policy = json.loads(args.policy.read_text()) if args.policy else {}
    if set(policy) - {"exclude", "include", "grouping_rules"}:
        parser.error("Policy supports only exclude, include and grouping_rules")
    for key, value in policy.items():
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            parser.error(f"Policy {key} must be a list of strings")
    if args.parallel == "ep":
        from .ep import analyze_ep
        plan = analyze_ep(args, policy)
        if plan is None:
            return
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, local_files_only=True)
        batches = load_calibration(args.calibration_data, tokenizer, samples=args.calibration_samples,
                                   sequence_length=args.calibration_seq_length, split=args.dataset_split,
                                   text_field=args.text_field)
        model, source_config, loading = load_model(
            args.model, device=args.device, dtype=getattr(torch, args.dtype), trust_remote_code=args.trust_remote_code,
        )
        print("Source loading:", json.dumps(loading, ensure_ascii=False))
        plan = analyze_model(
            model, batches, effective_bits=args.effective_bits,
            exclude=[*policy.get("exclude", []), *args.exclude],
            include=[*policy.get("include", []), *args.include],
            grouping_rules=policy.get("grouping_rules", []), source_config=source_config,
            model_id=str(args.model.resolve()), seed=args.seed, verbose=args.verbose,
            calibration={"data": args.calibration_data, "requested_samples": args.calibration_samples,
                         "max_sequence_length": args.calibration_seq_length},
        )
        plan["model"]["source_loading"] = loading
    manifest = [{"file": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                for p in sorted(args.model.iterdir())
                if p.is_file() and p.suffix in (".safetensors", ".bin", ".json")]
    plan["model"]["checkpoint_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    plan["model"]["identity_scope"] = "config, structure and file metadata; not a weight-payload hash"
    save_plan(plan, args.output)
    save_sensitivity_report(plan, report_path)
    save_sensitivity_json(plan, ranking_path)
    print(f"Saved {args.output}: {len(plan['module_bits'])} assigned targets, "
          f"{len(plan['excluded_modules'])} excluded, effective_bits={plan['actual_effective_bits']:.6f}")
    print(format_sensitivity_report(plan, top_k=10))
    print(f"Full sensitivity report: {report_path}")
    print(f"Sensitivity ranking JSON: {ranking_path}")
    if plan["coverage"]["uncertain_groups"]:
        print("Calibration coverage needs review:", plan["coverage"]["uncertain_groups"])
    if plan["unmatched_exclusion_rules"]:
        print("Exclusion patterns with no matching loaded module:", plan["unmatched_exclusion_rules"])


if __name__ == "__main__":
    main()
