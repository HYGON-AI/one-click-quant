#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# GLM-5.3 (non-Flash): same precision scope as quantize_glm5_3_int4_channel.py.
# Backbone routed experts: INT4 GPTQ. Other selected Linears: INT8 GPTQ.
# MTP: INT8 RTN by default; --mtp-int4 selects INT4 RTN for routed experts.
# Quantized groups declare dynamic per-token INT8 activations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if (( $# > 1 )); then
  echo "Usage: bash $0 [--mtp-int8 | --mtp-int4]" >&2
  exit 2
fi
MTP_OPTION="${1:---mtp-int8}"
case "${MTP_OPTION}" in
  --mtp-int8) OUTPUT_SUFFIX="" ;;
  --mtp-int4) OUTPUT_SUFFIX="-MTP-INT4" ;;
  *) echo "Usage: bash $0 [--mtp-int8 | --mtp-int4]" >&2; exit 2 ;;
esac

MODEL_PATH="${MODEL_PATH:-/llm-models/GLM-5.3-BF16}"
DATASET="${DATASET:-${PWD}/OpenThoughts-114k}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-/quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-intermediate}"
PACKED_MODEL_PATH="${PACKED_MODEL_PATH:-/quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ${OUTPUT_SUFFIX}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_CALIBRATION_SAMPLES="${NUM_CALIBRATION_SAMPLES:-512}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG_DIR="${LOG_DIR:-${PWD}/logs}"

if [[ -f "${PACKED_MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "Output model already exists: ${PACKED_MODEL_PATH}; choose a new PACKED_MODEL_PATH." >&2
  exit 1
fi

# Structural defaults retain router/norms/embeddings/kv_b/indexer weights_proj
# and the MTP-specific eh_proj/enorm/hnorm/shared_head in BF16.
IGNORE='re:.*\.self_attn\.kv_b_proj$'
FORCE_INT8='re:.*\.mlp\.(?:gate|up|down)_proj$,re:.*\.mlp\.shared_experts\.(?:gate|up|down)_proj$,re:.*\.self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)$,re:.*\.self_attn\.indexer\.(?:wq_b|wk)$'

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/moe_quant_glm5_3_w4a8${OUTPUT_SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: ${LOG_FILE}"

# No --group_size: the reference uses per-output-channel scales.
# No --quantize_only_experts: dense/shared/attention/indexer also need INT8.
PYTHONUNBUFFERED=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}" \
  torchrun --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" quant.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name_or_path "${DATASET}" \
  --num_calibration_samples "${NUM_CALIBRATION_SAMPLES}" \
  --max_sequence_length "${MAX_SEQUENCE_LENGTH}" \
  --bits 4 \
  --rel_damp 0.1 \
  --sym \
  --ignore "${IGNORE}" \
  --force_int8 "${FORCE_INT8}" \
  --attn_implementation sdpa \
  --dtype bfloat16 \
  --save_dir "${SAVE_MODEL_PATH}" \
  2>&1 | tee "${LOG_FILE}"

PYTHONUNBUFFERED=1 python pack_quantized_model.py \
  --model_name_or_path "${MODEL_PATH}" \
  --quantized_model_path "${SAVE_MODEL_PATH}" \
  --packed_model_path "${PACKED_MODEL_PATH}" \
  --dtype bfloat16 \
  --activation-bits 8 \
  "${MTP_OPTION}" \
  2>&1 | tee -a "${LOG_FILE}"
