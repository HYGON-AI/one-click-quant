#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/moe_quant_${TS}.log"

echo "Log file: ${LOG_FILE}"

export MODEL_PATH="${MODEL_PATH:-/models/GLM-5.3}"
export QUANT_BIT="${QUANT_BIT:-4}"
export SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-/models/GLM-5.3-QUANT}"
echo "model: ${MODEL_PATH}"
echo "bits: ${QUANT_BIT}"
echo "save model: ${SAVE_MODEL_PATH}"

#export WANDB_API_KEY="xxx"
#export WANDB_PROJECT=moe-quant-deepseek-v3
#export WANDB_NAME=ds-v3-w4g128-openplatypus-512x4096
#  --offload_activations \
#  --tie_gptq_handles \
#  --attn_implementation eager \
#  --dataset_name_or_path open-platypus \
#  --dataset_name_or_path open-thoughts \

# PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=32 \
#  --quantization_scale mse \
#  --group_size 128 \

#CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=32 \
PYTHONUNBUFFERED=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=32 \
  torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name_or_path open-thoughts \
  --num_calibration_samples 512 \
  --max_sequence_length 4096 \
  --bits "${QUANT_BIT}" \
  --rel_damp 0.1 \
  --sym \
  --quantize_only_experts \
  --attn_implementation sdpa \
  --dtype bfloat16 \
  --save_dir "${SAVE_MODEL_PATH}" \
  2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"

