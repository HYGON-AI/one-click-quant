#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/moe_quant_${TS}.log"

echo "Log file: ${LOG_FILE}"

export MODEL_PATH="${MODEL_PATH:-/models/GLM-5.3}"
echo "model: ${MODEL_PATH}"

#  --no-enable-prefix-caching \
#  --max-num-seqs 1 \
#  --no-enable-chunked-prefill \
#  --gpu_memory_utilization 0.95 \

PYTHONUNBUFFERED=1 HF_DATASETS_OFFLINE=1 \
  vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --enforce-eager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --max-model-len 32768 \
  --cpu-offload-gb 48 \
  --generation-config vllm \
  2>&1 | tee "${LOG_FILE}"

exit ${PIPESTATUS[0]}

