#!/usr/bin/env bash
# MiMo-V2.6-Flash-RL: source dtype policy, backbone GPTQ, old MTP RTN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if (( $# > 1 )); then
  echo "Usage: bash $0 [--preflight | --pack-only]" >&2
  exit 2
fi
MODE="${1:---run}"
case "$MODE" in
  --run|--preflight|--pack-only) ;;
  *) echo "Usage: bash $0 [--preflight | --pack-only]" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/llm-models/XiaomiMiMo/MiMo-V2.6-Flash-RL}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-/quant_models/models/MiMo-V2.6-Flash-RL-Channel-INT4-W4A8}"
DATASET="${DATASET:-open-platypus}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-/quant_models/models/MiMo-V2.6-Flash-RL-Channel-INT4-W4A8-GPTQ-intermediate}"
PACKED_MODEL_PATH="${PACKED_MODEL_PATH:-/quant_models/models/MiMo-V2.6-Flash-RL-Channel-INT4-W4A8-GPTQ}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_CALIBRATION_SAMPLES="${NUM_CALIBRATION_SAMPLES:-512}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
MASTER_PORT="${MASTER_PORT:-29503}"
LOG_DIR="${LOG_DIR:-/quant_models/logs/quant_logs}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/quant_models/.cache/moe-quant-hf-modules}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/quant_models/.cache/moe-quant-datasets}"

CHECK_ARGS=(--source-model "$MODEL_PATH")
if [[ -f "$REFERENCE_MODEL_PATH/model.safetensors.index.json" ]]; then
  CHECK_ARGS+=(--reference-model "$REFERENCE_MODEL_PATH")
fi
"$PYTHON" tests/check_mimo_v2.py "${CHECK_ARGS[@]}"
if [[ "$MODE" == --preflight ]]; then
  exit 0
fi

for VALUE in "$NPROC_PER_NODE" "$NUM_CALIBRATION_SAMPLES" "$MAX_SEQUENCE_LENGTH"; do
  if [[ ! "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Process count, calibration sample count and sequence length must be positive integers.' >&2
    exit 2
  fi
done
if (( NUM_CALIBRATION_SAMPLES % NPROC_PER_NODE != 0 )); then
  echo 'NUM_CALIBRATION_SAMPLES must be divisible by NPROC_PER_NODE.' >&2
  exit 2
fi
if [[ -d "$PACKED_MODEL_PATH" ]] && [[ -n "$(find "$PACKED_MODEL_PATH" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Output directory must be empty: $PACKED_MODEL_PATH" >&2
  exit 1
fi
if [[ "$MODE" == --run && -d "$SAVE_MODEL_PATH" ]] && [[ -n "$(find "$SAVE_MODEL_PATH" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "GPTQ directory must be empty for a new run: $SAVE_MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$HF_MODULES_CACHE" "$HF_DATASETS_CACHE"
LOG_FILE="$LOG_DIR/moe_quant_mimo_v26_flash_rl_w4a8_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
echo "Dataset: $DATASET; processes: $NPROC_PER_NODE; calibration: $NUM_CALIBRATION_SAMPLES x $MAX_SEQUENCE_LENGTH"

if [[ "$MODE" == --run ]]; then
  # Adapter derives mixed INT4/INT8 and keep scope from the source headers.
  # No group_size, quantize_only_experts, ignore or force_int8 overrides.
  "$PYTHON" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" quant.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name_or_path "$DATASET" \
    --num_calibration_samples "$NUM_CALIBRATION_SAMPLES" \
    --max_sequence_length "$MAX_SEQUENCE_LENGTH" \
    --bits 4 --sym --rel_damp 0.1 \
    --tie_gptq_handles --offload_activations \
    --attn_implementation eager --dtype bfloat16 \
    --save_dir "$SAVE_MODEL_PATH" \
    2>&1 | tee "$LOG_FILE"
fi

"$PYTHON" pack_quantized_model.py \
  --model_name_or_path "$MODEL_PATH" \
  --quantized_model_path "$SAVE_MODEL_PATH" \
  --packed_model_path "$PACKED_MODEL_PATH" \
  --dtype bfloat16 --activation-bits 8 --mtp-int8 \
  2>&1 | tee -a "$LOG_FILE"

"$PYTHON" tests/check_mimo_v2.py "${CHECK_ARGS[@]}" --packed-model "$PACKED_MODEL_PATH" \
  2>&1 | tee -a "$LOG_FILE"
