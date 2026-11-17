#!/usr/bin/env python3
"""Wait for four idle DCUs, then run K3.1 Pro streaming W4A8/W8A8 analysis."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def devices(smi):
    completed = subprocess.run(
        [smi, "--showmeminfo", "vram", "--showuse", "--json"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    data = json.loads(completed.stdout)
    result = []
    for name, values in data.items():
        if not name.startswith("card") or not name[4:].isdigit():
            continue
        total = float(values["vram Total Memory (MiB)"])
        used = float(values["vram Total Used Memory (MiB)"])
        usage = float(values.get("HCU use (%)", values.get("GPU use (%)", "nan")))
        result.append({"id": int(name[4:]), "free_gib": (total - used) / 1024,
                       "used_gib": used / 1024, "utilization": usage})
    if not result:
        raise RuntimeError("No valid device metrics returned by rocm-smi")
    return sorted(result, key=lambda item: item["id"])


def eligible(cards, args):
    return [c["id"] for c in cards if c["free_gib"] >= args.min_free_gib
            and c["used_gib"] <= args.max_used_gib and c["utilization"] <= 5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/DL_DATA/llm-models/moonshotai/Kimi-K3.1-Pro"))
    parser.add_argument("--output", type=Path, default=Path("/my_code/modes/Kimi-K3.1-Pro-AutoQuantize-W4A8/autoquantize_plan.json"))
    parser.add_argument("--log-dir", type=Path, default=Path("/my_code/logs/auto_quantize_kimi_k3.1_pro"))
    parser.add_argument("--calibration-data", type=Path, default=Path("/my_code/logs/auto_quantize_kimi_k3.1_pro/calibration_open_platypus.jsonl"))
    parser.add_argument("--smi", default="/opt/dtk/bin/rocm-smi")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--min-free-gib", type=float, default=56)
    parser.add_argument("--max-used-gib", type=float, default=4)
    parser.add_argument("--check-once", action="store_true", help="Only print metrics; never launch analysis")
    args = parser.parse_args()
    if args.interval <= 0 or args.stable_checks < 1:
        parser.error("interval and stable-checks must be positive")
    if args.check_once:
        cards = devices(args.smi)
        print(json.dumps({"devices": cards, "eligible": eligible(cards, args)}, indent=2))
        return 0
    if not (args.model / "config.json").is_file() or not args.calibration_data.is_file():
        parser.error("Model config and prepared local calibration data must exist")
    outputs = [args.output, args.output.with_suffix(".sensitivity.json"),
               args.output.with_suffix(".sensitivity.txt")]
    if any(p.exists() for p in outputs):
        parser.error("Analysis output already exists; use a new output path")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.log_dir / "watcher.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another watcher is already running for this log directory")
    (args.log_dir / "watcher.pid").write_text(str(os.getpid()) + "\n")
    state = {"pid": os.getpid(), "output": str(args.output), "world_size": 4}
    child = None

    def update(status, **values):
        state.update(status=status, updated_at=datetime.now().astimezone().isoformat(), **values)
        temporary = args.log_dir / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.log_dir / "status.json")
        print(json.dumps(state), flush=True)

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        update("stopped", signal=signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    repo = Path(__file__).resolve().parents[2]
    selected, stable = [], 0
    while True:
        try:
            cards = devices(args.smi)
            available = eligible(cards, args)
            if len(available) < 4:
                selected, stable = [], 0
            elif selected and set(selected) <= set(available):
                stable += 1
            else:
                selected, stable = available[:4], 1
            update("waiting", devices=cards, selected_devices=selected, stable_checks=stable)
            if stable >= args.stable_checks:
                break
        except (subprocess.SubprocessError, ValueError, KeyError, OSError, RuntimeError) as error:
            selected, stable = [], 0
            update("waiting", detection_error=str(error), stable_checks=0)
        time.sleep(args.interval)
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        env[key] = ",".join(map(str, selected))
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MALLOC_ARENA_MAX="2",
               MALLOC_MMAP_THRESHOLD_="1048576", MALLOC_TRIM_THRESHOLD_="67108864",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TZ="Asia/Shanghai",
               AUTOQUANTIZE_LOG_BLOCKS="1")
    launch = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4"]
    commands = [
        ("smoke", launch + [str(repo / "utils/auto_quantize/tests/kimi_k31_smoke.py"),
                            "--source", str(args.model), "--work-dir", str(args.log_dir / "smoke_fixture")]),
        ("analysis", launch + ["-m", "utils.auto_quantize", "--parallel", "ep", "--trust-remote-code",
                               "--model", str(args.model), "--output", str(args.output),
                               "--calibration-data", str(args.calibration_data),
                               "--calibration-samples", "8", "--calibration-seq-length", "512",
                               "--effective-bits", "4.2", "--num-threads", "1"]),
    ]
    for phase, command in commands:
        logfile = args.log_dir / (phase + ".log")
        update("starting_" + phase, command=command, log=str(logfile))
        with logfile.open("a", buffering=1) as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            update("running_" + phase, child_pid=child.pid)
            code = child.wait()
        if code:
            update("failed_" + phase, exit_code=code)
            return code
    if not all(p.is_file() and p.stat().st_size for p in outputs):
        update("failed_validation", error="Process succeeded but expected output files are missing")
        return 1
    update("completed", exit_code=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
