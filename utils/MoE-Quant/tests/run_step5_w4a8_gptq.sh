#!/usr/bin/env bash
# Run inside a container with source/calibration mounts; never overwrite existing model artifacts.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

MODEL_PATH="${MODEL_PATH:-/DL_DATA/step-5-preview-0913-fp8-channel}"
DATASET="${DATASET:-/quant_models/step5-gptq-calibration/sharegpt_code_general_512_seed0.jsonl}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-/quant_models/step-5-preview-mixed-w4a8-w8a8-GPTQ-intermediate}"
PACKED_MODEL_PATH="${PACKED_MODEL_PATH:-/quant_models/step-5-preview-mixed-w4a8-w8a8-GPTQ}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_CALIBRATION_SAMPLES="${NUM_CALIBRATION_SAMPLES:-512}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
MODE="${1:---run}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-2,5,6,7}"
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/quant_models/.cache/step5-gptq-datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/quant_models/.cache/step5-gptq-modules}"
export LOG_DIR="${LOG_DIR:-/my_code/Quantitative_Code/one-click-quant/logs/step5-gptq}"
mkdir -p "$LOG_DIR" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"
if [[ "$MODE" != --preflight ]]; then
    exec 8>"$LOG_DIR/.quantization.lock"
    flock -n 8 || { echo "Another Step5 GPTQ workflow is already running"; exit 1; }
fi
LOG_FILE="$LOG_DIR/gptq_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
printf 'Workflow PID=%s log=%s\n' "$$" "$LOG_FILE"
printf '%s\n' "$$" > "$LOG_DIR/workflow.pid"
printf '%s\n' "$LOG_FILE" > "$LOG_DIR/latest_log"

case "$MODE" in --run|--pack-only|--preflight) ;; *) exit 2 ;; esac
python tests/check_step5.py --source "$MODEL_PATH"
if [[ "$MODE" == --preflight ]]; then exit 0; fi
[[ -f "$DATASET" ]] || { echo "Missing local calibration data: $DATASET"; exit 1; }
if (( NUM_CALIBRATION_SAMPLES % NPROC_PER_NODE != 0 || 352 % NPROC_PER_NODE != 0 )); then
    echo "Sample/expert counts must be divisible by NPROC_PER_NODE"; exit 1
fi
if [[ -d "$PACKED_MODEL_PATH" ]] && [[ -n "$(ls -A "$PACKED_MODEL_PATH")" ]]; then
    echo "Output directory must be empty: $PACKED_MODEL_PATH"; exit 1
fi
if [[ "$MODE" == --run ]]; then
    if [[ -d "$SAVE_MODEL_PATH" ]] && [[ -n "$(ls -A "$SAVE_MODEL_PATH")" ]]; then
        echo "GPTQ intermediate directory must be empty: $SAVE_MODEL_PATH"; exit 1
    fi
    echo "STAGE=quantize"
    printf 'EP=%s cards=%s calibration=%s x max %s\n' "$NPROC_PER_NODE" "$HIP_VISIBLE_DEVICES" "$NUM_CALIBRATION_SAMPLES" "$MAX_SEQUENCE_LENGTH"
    python -m torch.distributed.run \
      --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-29512}" quant.py \
      --model_name_or_path "$MODEL_PATH" --dataset_name_or_path "$DATASET" \
      --num_calibration_samples "$NUM_CALIBRATION_SAMPLES" --max_sequence_length "$MAX_SEQUENCE_LENGTH" \
      --bits 4 --sym --rel_damp 0.1 --tie_gptq_handles --offload_activations \
      --attn_implementation eager --dtype bfloat16 --log_error \
      --save_dir "$SAVE_MODEL_PATH"
fi
echo "STAGE=pack"
python pack_quantized_model.py --model_name_or_path "$MODEL_PATH" \
  --quantized_model_path "$SAVE_MODEL_PATH" --packed_model_path "$PACKED_MODEL_PATH" \
  --dtype bfloat16 --activation-bits 8 --mtp-int8 --max-shard-size-mb 2048
python tests/check_step5.py --source "$MODEL_PATH" --packed "$PACKED_MODEL_PATH"
echo "STAGE=model_ready"

# The Step5 vLLM runtime must be taught to load non-expert INT8 (and the
# shared/dense MLP INT8 path) before serving.
if [[ "${SERVE_AFTER_PACK:-0}" != 1 ]]; then
    echo "Skipping serve/evaluation (set SERVE_AFTER_PACK=1 only after runtime INT8 support is verified)."
    exit 0
fi
# The existing per-port lock prevents duplicate server/evaluation workflows.
STEP_SCRIPTS="$(cd ../../examples/step && pwd)"
SERVER_LOG_DIR="$STEP_SCRIPTS/log"
mkdir -p "$SERVER_LOG_DIR"
PORT="${PORT:-18000}"
if flock -n "$SERVER_LOG_DIR/.wait_step5_eval_${PORT}.lock" true; then
    echo "Starting automatic serve/evaluation waiter on port $PORT"
    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MODEL_DIR="$PACKED_MODEL_PATH" \
      LOG_DIR="$SERVER_LOG_DIR" PORT="$PORT" \
      bash "$STEP_SCRIPTS/wait_serve_step5_w4a8_bw1000_vllm_humaneval.sh"
else
    echo "Existing automatic serve/evaluation waiter owns port $PORT; it will detect the completed GPTQ checkpoint."
fi
