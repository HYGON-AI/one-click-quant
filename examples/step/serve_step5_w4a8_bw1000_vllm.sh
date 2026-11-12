#!/usr/bin/env bash
# Run inside vllm0281_step-5_zyq using its installed vLLM and HCU plugin.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/serve_step5_w4a8_bw1000_vllm_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
printf 'Log file: %s\n' "$LOG_FILE"

MODEL_DIR="${MODEL_DIR:-/quant_models/step-5-preview-channel-w4a8-GPTQ}"
VLLM_BIN="${VLLM_BIN:-vllm}"
if [[ ! -f "${MODEL_DIR}/config.json" || ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  printf 'Incomplete checkpoint: %s (config.json and model.safetensors.index.json required)\n' "$MODEL_DIR" >&2
  exit 1
fi
if ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
  printf 'vLLM executable not found: %s\n' "$VLLM_BIN" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TRITON_JSON_DIR="${TRITON_JSON_DIR:-/my_code/src/vllm-plugin-das/tuning/w8a8}"

W4A8_TUNING_JSON="${TRITON_JSON_DIR}/MOE_W4A8INT8_E=44_N1=3072_N2=4096_K=1536_TOPK8_gfx936_80cu.json"
if [[ ! -f "$W4A8_TUNING_JSON" ]]; then
  printf 'W4A8 tuning JSON not found: %s\n' "$W4A8_TUNING_JSON" >&2
  printf 'Run /tmp/tune_step5_w4a8.py first, then start the server again.\n' >&2
  exit 1
fi
python3 - "$W4A8_TUNING_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
required = {str(i) for i in range(1, 17)} | {"256", "1024"}
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
shape = data.get("3072_4096_1536", {})
missing = sorted(required - set(shape), key=lambda x: int(x))
if missing:
    raise SystemExit(
        "Incomplete W4A8 tuning JSON; missing keys: "
        + ", ".join(missing)
        + ". Finish /tmp/tune_step5_w4a8.py before starting the server."
    )
PY
printf 'W4A8 tuning directory: %s\n' "$TRITON_JSON_DIR"
printf 'W4A8 tuning JSON: %s\n' "$W4A8_TUNING_JSON"

# DP=1 and TP=8 with expert parallel enabled gives EP=8.
# MTP is disabled by leaving speculative decoding unconfigured.
# Preserve compressed-tensors so each config group keeps its own scheme.
# The HCU plugin converts channel INT4 expert weights to LightOp W4A8 in memory.
args=(
  --host 0.0.0.0
  --port "${PORT:-18000}"
  --served-model-name step5
  --tensor-parallel-size 8
  --pipeline-parallel-size 1
  --data-parallel-size 1
  --enable-expert-parallel
  --data-parallel-backend mp
  --distributed-executor-backend mp
  --disable-custom-all-reduce
  --quantization compressed-tensors
  --dtype bfloat16
  --max-model-len "${CONTEXT_LENGTH:-8192}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}"
  --enforce-eager
  --reasoning-parser "${REASONING_PARSER:-step3p5}"
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER:-step3p5}"
)

printf 'Command:'
printf ' %q' "$VLLM_BIN" serve "$MODEL_DIR" "${args[@]}" "$@"
printf '\n'

# Keep Bash as the parent so worker cleanup cannot terminate the log's tee.
"$VLLM_BIN" serve "$MODEL_DIR" "${args[@]}" "$@"
