#!/usr/bin/env bash
# Run inside the dedicated image. No native --quantization flag is needed.
set -euo pipefail
: "${HY4_GPTQ_ROOT:?Set HY4_GPTQ_ROOT to the read-only packed checkpoint}"
: "${HY4_GPTQ_REPORT_DIR:?Set HY4_GPTQ_REPORT_DIR to a separate writable directory}"
mkdir -p "$HY4_GPTQ_REPORT_DIR"
exec python3 -m sglang.launch_server \
  --model-path "$HY4_GPTQ_ROOT" --tp-size 8 --dtype bfloat16 \
  --host 127.0.0.1 --port "${PORT:-31108}" \
  --context-length "${CONTEXT_LENGTH:-32768}" \
  --max-total-tokens "${CONTEXT_LENGTH:-32768}" \
  --max-running-requests 8 --chunked-prefill-size 512 \
  --mem-fraction-static 0.80 --skip-server-warmup \
  --watchdog-timeout 1800 --random-seed 20260910 \
  --cuda-graph-config '{"decode":{"backend":"full","bs":[1,2,4,8]},"prefill":{"backend":"disabled"}}' "$@"
