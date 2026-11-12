#!/usr/bin/env bash
# Run INSIDE vllm0281_step-5_zyq. Wait for cards 0-7, serve Step5, then run
# HumanEval, GSM8K, and Math500.
# Defaults: poll every 5s; ALL eight cards must have used/total VRAM < 10%.
# Example (host):
# docker exec vllm0281_step-5_zyq bash \
#   /my_code/Quantitative_Code/one-click-quant/examples/step/wait_serve_step5_w4a8_bw1000_vllm_humaneval.sh
# Optional: EVAL_LIMIT=8 for a smoke evaluation; unset for the full datasets.
# The detached server remains running after evaluation or interruption.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd -- "$LOG_DIR" && pwd)"
RUN_DIR="$(mktemp -d "${LOG_DIR}/wait_step5_eval_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX")"
exec > >(tee -a "${RUN_DIR}/workflow.log") 2>&1

log() { printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%F %T')" "$*"; }
fail() { log "ERROR: $*"; exit 1; }

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMI_BIN="${SMI_BIN:-/opt/hyhal/bin/hy-smi}"
EVALSCOPE_BIN="${EVALSCOPE_BIN:-evalscope}"
SERVER_SCRIPT="${SERVER_SCRIPT:-${SCRIPT_DIR}/serve_step5_w4a8_bw1000_vllm.sh}"
export PORT="${PORT:-18000}"
export VRAM_THRESHOLD_PERCENT="${VRAM_THRESHOLD_PERCENT:-10}"
export POLL_INTERVAL="${POLL_INTERVAL:-5}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-7200}"
export MODEL_READY_TIMEOUT="${MODEL_READY_TIMEOUT:-0}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-7200}"
export EVAL_LIMIT="${EVAL_LIMIT:-}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MODEL_DIR="${MODEL_DIR:-/quant_models/step-5-preview-channel-w4a8-GPTQ}"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOG_DIR
export PYTHONUNBUFFERED=1

if (( $# == 0 )); then
  SERVER_ARGS=(--language-model-only)
else
  SERVER_ARGS=("$@")
fi

log "Workflow log: ${RUN_DIR}/workflow.log"
for executable in "$PYTHON_BIN" "$SMI_BIN" "$EVALSCOPE_BIN" flock setsid nohup timeout; do
  command -v "$executable" >/dev/null 2>&1 || fail "Executable not found: $executable (run inside the container)"
done
[[ -f "$SERVER_SCRIPT" ]] || fail "Server script not found: $SERVER_SCRIPT"

# A single workflow per port. Close fd 9 in detached children so they do not
# retain this lock after the evaluation process exits.
exec 9>"${LOG_DIR}/.wait_step5_eval_${PORT}.lock"
flock -n 9 || fail "Another wait/evaluation workflow holds port ${PORT}'s lock"

# Keep the probe in the run directory for reproducibility; it uses no GPU runtime.
cat > "${RUN_DIR}/probe.py" <<'PY'
import json
import math
import os
import socket
import sys
import urllib.request
from decimal import Decimal


def positive(name, integer=False):
    value = int(os.environ[name]) if integer else float(os.environ[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def main():
    action = sys.argv[1]
    if action == "config":
        port = positive("PORT", True)
        if port > 65535:
            raise ValueError("PORT must be <= 65535")
        if positive("VRAM_THRESHOLD_PERCENT") > 100:
            raise ValueError("VRAM_THRESHOLD_PERCENT must be <= 100")
        for name in ("POLL_INTERVAL", "EVAL_TIMEOUT"):
            positive(name)
        for name in ("STARTUP_TIMEOUT", "EVAL_BATCH_SIZE", "EVAL_MAX_TOKENS"):
            positive(name, True)
        model_ready_timeout = int(os.environ["MODEL_READY_TIMEOUT"])
        if model_ready_timeout < 0:
            raise ValueError("MODEL_READY_TIMEOUT must be >= 0")
        if os.environ["EVAL_LIMIT"]:
            positive("EVAL_LIMIT", True)
        print(json.dumps({
            "temperature": 0,
            "max_tokens": int(os.environ["EVAL_MAX_TOKENS"]),
            "stream": True,
            "timeout": float(os.environ["EVAL_TIMEOUT"]),
        }))
    elif action == "model_ready":
        from pathlib import Path
        from safetensors import safe_open

        root = Path(os.environ["MODEL_DIR"])
        config_path = root / "config.json"
        index_path = root / "model.safetensors.index.json"
        if not root.is_dir():
            print(f"model directory does not exist: {root}")
            return 1
        if not config_path.is_file() or not index_path.is_file():
            print("waiting for config.json and model.safetensors.index.json")
            return 1
        if any(root.rglob("*.tmp")):
            print("temporary checkpoint files are still present")
            return 1

        try:
            with config_path.open(encoding="utf-8") as f:
                config = json.load(f)
            with index_path.open(encoding="utf-8") as f:
                index = json.load(f)
        except Exception as exc:
            print(f"checkpoint metadata is not readable yet: {exc}")
            return 1

        quant = config.get("quantization_config", {})
        groups = quant.get("config_groups", {})
        group = groups.get("group_w4a8", {})
        weights = group.get("weights", {})
        if not (
            quant.get("quant_method") == "compressed-tensors"
            and quant.get("quantization_status") == "compressed"
            and quant.get("format") == "pack-quantized"
            and weights.get("num_bits") == 4
            and weights.get("type") == "int"
            and weights.get("strategy") == "channel"
        ):
            print("config.json is present but is not a completed compressed-tensors W4A8 config")
            return 1

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            print("model index has no weight_map yet")
            return 1
        shard_names = sorted(set(weight_map.values()))
        if not all(isinstance(name, str) and name.endswith(".safetensors") for name in shard_names):
            print("model index contains invalid shard names")
            return 1
        missing = [name for name in shard_names if not (root / name).is_file()]
        if missing:
            print(f"waiting for {len(missing)} indexed shard(s), first missing: {missing[0]}")
            return 1

        packed = sum(name.endswith(".weight_packed") for name in weight_map)
        scales = sum(name.endswith(".weight_scale") for name in weight_map)
        shapes = sum(name.endswith(".weight_shape") for name in weight_map)
        if not (packed and packed == scales == shapes):
            print(
                "W4A8 tensors are incomplete in the index: "
                f"packed={packed}, scale={scales}, shape={shapes}"
            )
            return 1

        try:
            for name in shard_names:
                path = root / name
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if not list(handle.keys()):
                        raise ValueError(f"empty shard: {name}")
        except Exception as exc:
            print(f"checkpoint shard is incomplete or unreadable: {exc}")
            return 1

        print(
            f"completed W4A8 checkpoint: {root} "
            f"({len(shard_names)} shards, {packed} packed matrices)"
        )
        return 0
    elif action == "metrics":
        with open(sys.argv[2]) as f:
            data = json.load(f)
        threshold = Decimal(os.environ["VRAM_THRESHOLD_PERCENT"])
        all_idle = True
        rows = []
        for index in range(8):
            card = data[f"card{index}"]
            total = Decimal(str(card["vram Total Memory (MiB)"]))
            used = Decimal(str(card["vram Total Used Memory (MiB)"]))
            if not total.is_finite() or not used.is_finite() or not 0 <= used <= total or total <= 0:
                raise ValueError(f"Invalid VRAM metrics on card{index}")
            percent = used * 100 / total
            all_idle = all_idle and percent < threshold
            rows.append(f"card{index}={percent:.2f}% ({used}/{total} MiB)")
        print(" | ".join(rows), flush=True)
        return 0 if all_idle else 1
    elif action == "port_free":
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", int(os.environ["PORT"])))
    elif action == "ready":
        base = f"http://127.0.0.1:{os.environ['PORT']}"
        # Never send local readiness requests through an HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(base + "/health", timeout=3) as response:
            if response.status != 200:
                return 1
        with opener.open(base + "/v1/models", timeout=3) as response:
            models = json.load(response)
        return 0 if any(m.get("id") == "step5" for m in models.get("data", [])) else 1
    else:
        raise ValueError(f"Unknown action: {action}")
    return 0


try:
    sys.exit(main())
except Exception as exc:
    print(f"Probe error: {exc}", file=sys.stderr)
    sys.exit(2)
PY

GENERATION_CONFIG="$("$PYTHON_BIN" "${RUN_DIR}/probe.py" config)"
printf '%s\n' "$GENERATION_CONFIG" > "${RUN_DIR}/generation_config.json"

# Use the local endpoint even when the container has proxy variables set.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
log "Wait for completed model checkpoint: ${MODEL_DIR}"
model_ready_start=$SECONDS
while true; do
  if model_status="$("$PYTHON_BIN" "${RUN_DIR}/probe.py" model_ready 2>&1)"; then
    log "$model_status"
    break
  fi
  log "Model is not ready: ${model_status}"
  if (( MODEL_READY_TIMEOUT > 0 && SECONDS - model_ready_start >= MODEL_READY_TIMEOUT )); then
    fail "Model readiness timed out after ${MODEL_READY_TIMEOUT}s: ${MODEL_DIR}"
  fi
  sleep "$POLL_INTERVAL"
done

log "Wait for cards 0-7: each VRAM < ${VRAM_THRESHOLD_PERCENT}%; poll ${POLL_INTERVAL}s"
while true; do
  if timeout 15 "$SMI_BIN" --showmeminfo vram --json > "${RUN_DIR}/hy_smi_latest.json" 2>"${RUN_DIR}/hy_smi_latest.stderr"; then
    if metrics="$("$PYTHON_BIN" "${RUN_DIR}/probe.py" metrics "${RUN_DIR}/hy_smi_latest.json" 2>&1)"; then
      log "$metrics"
      if "$PYTHON_BIN" "${RUN_DIR}/probe.py" port_free > /dev/null 2>&1; then
        break
      fi
      log "Cards idle, but port ${PORT} is occupied; continue waiting"
    else
      log "$metrics"
    fi
  else
    log "hy-smi failed/timed out; retry without starting server"
    cat "${RUN_DIR}/hy_smi_latest.stderr"
  fi
  sleep "$POLL_INTERVAL"
done

log "All eight cards below threshold. Starting server with:"
log "KV_CACHE_DTYPE=${KV_CACHE_DTYPE} MAX_NUM_SEQS=${MAX_NUM_SEQS} MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
printf 'Server command:'
printf ' %q' env \
  "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}" \
  "MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
  "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}" \
  bash "$SERVER_SCRIPT" "${SERVER_ARGS[@]}"
printf '\n'
nohup setsid env \
  "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}" \
  "MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
  "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}" \
  bash "$SERVER_SCRIPT" "${SERVER_ARGS[@]}" \
  > "${RUN_DIR}/server_console.log" 2>&1 < /dev/null 9>&- &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "${RUN_DIR}/server.pid"
log "Server launcher PID=${SERVER_PID}; console: ${RUN_DIR}/server_console.log"
# nohup + setsid keep the server alive if the workflow is interrupted.
# Existing serve script writes its own timestamped log directly under LOG_DIR.
trap 'log "Interrupted; detached server (if started) remains running"; exit 130' INT
trap 'log "Terminated; detached server (if started) remains running"; exit 143' TERM

startup_begin=$SECONDS
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    server_status=0
    wait "$SERVER_PID" || server_status=$?
    tail -n 60 "${RUN_DIR}/server_console.log"
    fail "Server exited before readiness (status ${server_status}); evaluation skipped"
  fi
  if "$PYTHON_BIN" "${RUN_DIR}/probe.py" ready > /dev/null 2>&1; then
    break
  fi
  if (( SECONDS - startup_begin >= STARTUP_TIMEOUT )); then
    fail "Readiness timed out after ${STARTUP_TIMEOUT}s; inspect ${RUN_DIR}/server_console.log (server remains running)"
  fi
  sleep "$POLL_INTERVAL"
done
log "Server ready: http://127.0.0.1:${PORT}/v1 ; model=step5"

eval_status=0
for dataset_spec in \
  "humaneval:HumanEval" \
  "gsm8k:GSM8K" \
  "math_500:Math500"; do
  dataset="${dataset_spec%%:*}"
  dataset_label="${dataset_spec#*:}"
  dataset_dir="${RUN_DIR}/evalscope_${dataset}"
  eval_log="${RUN_DIR}/evalscope_${dataset}.log"
  eval_args=(
    eval
    --model step5
    --api-url "http://127.0.0.1:${PORT}/v1"
    --api-key "${API_KEY:-EMPTY}"
    --eval-type openai_api
    --datasets "$dataset"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --generation-config "$GENERATION_CONFIG"
    --repeats 1
    --seed 42
    --work-dir "$dataset_dir"
    --no-timestamp
  )
  if [[ -n "$EVAL_LIMIT" ]]; then
    eval_args+=(--limit "$EVAL_LIMIT")
  fi
  log "Starting ${dataset_label}; limit=${EVAL_LIMIT:-full}, batch=${EVAL_BATCH_SIZE}, max_tokens=${EVAL_MAX_TOKENS}"
  printf 'Eval command:'
  printf ' %q' "$EVALSCOPE_BIN" "${eval_args[@]}"
  printf '\n'
  set +e
  "$EVALSCOPE_BIN" "${eval_args[@]}" 9>&- 2>&1 | tee "$eval_log"
  current_status=${PIPESTATUS[0]}
  set -e
  log "${dataset_label} EvalScope exit status=${current_status}; reports/results: ${dataset_dir}"
  if (( current_status != 0 )); then
    eval_status=$current_status
    log "${dataset_label} failed; continuing with the remaining datasets"
  fi
done

log "Server remains running; launcher PID=${SERVER_PID}; all workflow logs: ${RUN_DIR}"
exit "$eval_status"
