#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# GLM-5.3-Flash GPTQ W4A8 (mixed INT4/INT8), scope aligned 1:1 with
# GLM-5.3-Flash-CHANNEL-INT4-w4a8-v3-same-as-iter_0033200
# (one-click-quant/examples/glm/quantize_glm5_3_flash_int4_channel.py
#  --no-mla-to-bf16 --linear-attn-int8 --kda-int8-scope qkvo --indexer-int8):
#   INT4 GPTQ : layer 3-44 routed experts
#   INT8 GPTQ : dense MLP 0-2 / shared_experts / MLA q_a,q_b,kv_a,o / KDA q,k,v,o / indexer wq_b,wk
#   INT8 RTN  : MTP layer 45 (bypasses GPTQ; --mtp-int8 at packing time)
#   BF16      : kv_b_proj / KDA b,f_a,f_b,g_a,g_b / router / indexer weights_proj / norms / vision
set -euo pipefail

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/moe_quant_glm5_flash_w4a8_${TS}.log"
echo "Log file: ${LOG_FILE}"

MODEL_PATH="${MODEL_PATH:-/llm-models/GLM-5.3-Flash}"
DATASET="${DATASET:-./OpenThoughts-114k}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-/llm-models/GLM-5.3-Flash-GPTQ-W4A8-gptq}"
PACKED_MODEL_PATH="${PACKED_MODEL_PATH:-/llm-models/GLM-5.3-Flash-GPTQ-W4A8}"

# Kept BF16 on top of the adapter defaults. f_a/f_b live under
# self_attn.forget_gate.* in the modelling code and self_attn.* at runtime.
IGNORE='re:.*\.self_attn\.kv_b_proj$,re:.*\.self_attn\.(?:forget_gate\.)?f_(?:a|b)_proj$,re:.*\.self_attn\.(?:b|g_a|g_b)_proj$'
# Everything else except routed experts is GPTQ INT8 (no commas inside a pattern).
FORCE_INT8='re:.*\.mlp\.(?:gate|up|down)_proj$,re:.*\.mlp\.shared_experts\.(?:gate|up|down)_proj$,re:.*\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa)$,re:.*\.self_attn\.indexer\.(?:wq_b|wk)$'

PYTHONUNBUFFERED=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=32 \
  torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name_or_path "${DATASET}" \
  --num_calibration_samples 512 \
  --max_sequence_length 4096 \
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
  --mtp-int8 \
  2>&1 | tee -a "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
