#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/moe_quant_${TS}.log"

echo "Log file: ${LOG_FILE}"

#export WANDB_API_KEY="xxx"
#export WANDB_PROJECT=moe-quant-deepseek-v3
#export WANDB_NAME=ds-v3-w4g128-openplatypus-512x4096
#  --offload_activations \
#  --tie_gptq_handles \

PYTHONUNBUFFERED=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=32 \
  CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nnodes=1 --nproc-per-node=4 --master_port 29501 quant.py \
  --model_name_or_path /models/Kimi-K3 \
  --dataset_name_or_path open-platypus \
  --num_calibration_samples 64 \
  --max_sequence_length 2048 \
  --bits 4 \
  --group_size 128 \
  --rel_damp 0.1 \
  --sym \
  --quantize_only_experts \
  --attn_implementation eager \
  --dtype bfloat16 \
  --save_dir /models/Kimi-K3-L0-2-W4A16-no-packed \
  2>&1 | tee "${LOG_FILE}"

exit ${PIPESTATUS[0]}

