#!/usr/bin/env bash
# Run inside a container with source/calibration mounts; never overwrite existing model artifacts.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

MODEL_PATH="${MODEL_PATH:-/DL_DATA/step-5-preview-0913-fp8-channel}"
DATASET="${DATASET:-/quant_models/step5-gptq-calibration/sharegpt_code_general_512_seed0.jsonl}"
# RECIPE selects one of the two fixed Step5 scopes:
#   experts -> routed experts only (channel INT4 W4A8), everything else BF16
#   mixed   -> routed experts INT4 W4A8 + attention/dense MLP/shared MLP
#              INT8 W8A8 + lm_head/MTP INT8 RTN
RECIPE="${RECIPE:-mixed}"
case "$RECIPE" in
    experts)
        SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-/quant_models/step-5-preview-channel-w4a8-GPTQ-intermediate}"
        PACKED_MODEL_PATH="${PACKED_MODEL_PATH:-/quant_models/step-5-preview-channel-w4a8-GPTQ}"
        SCOPE_ARGS=(--quantize_only_experts)
        CHECK_SCOPE_ARGS=(--quantize-only-experts)
        MTP_ARGS=()
        ;;
    mixed)
        SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-/quant_models/step-5-preview-int4-int8-GPTQ-intermediate}"
        PACKED_MODEL_PATH="${PACKED_MODEL_PATH:-/quant_models/step-5-preview-int4-int8-GPTQ}"
        SCOPE_ARGS=()
        CHECK_SCOPE_ARGS=()
        MTP_ARGS=(--mtp-int8)
        ;;
    *)
        echo "Unknown RECIPE=$RECIPE; expected 'mixed' or 'experts'" >&2
        exit 2
        ;;
esac
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_CALIBRATION_SAMPLES="${NUM_CALIBRATION_SAMPLES:-512}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
MODE="${1---run}"
case "$MODE" in
    --run|--pack-only|--preflight) ;;
    *)
        printf 'Unknown MODE: %s\nUsage: %s [--run|--pack-only|--preflight]\n' "$MODE" "$0" >&2
        exit 2
        ;;
esac
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

python tests/check_step5.py --source "$MODEL_PATH" "${CHECK_SCOPE_ARGS[@]}"
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
    printf 'RECIPE=%s EP=%s cards=%s calibration=%s x max %s\n' "$RECIPE" "$NPROC_PER_NODE" "$HIP_VISIBLE_DEVICES" "$NUM_CALIBRATION_SAMPLES" "$MAX_SEQUENCE_LENGTH"
    python -m torch.distributed.run \
      --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-29512}" quant.py \
      --model_name_or_path "$MODEL_PATH" --dataset_name_or_path "$DATASET" \
      --num_calibration_samples "$NUM_CALIBRATION_SAMPLES" --max_sequence_length "$MAX_SEQUENCE_LENGTH" \
      --bits 4 --sym --rel_damp 0.1 --tie_gptq_handles --offload_activations \
      --attn_implementation eager --dtype bfloat16 --log_error \
      "${SCOPE_ARGS[@]}" \
      --save_dir "$SAVE_MODEL_PATH"
fi
echo "STAGE=pack"
python pack_quantized_model.py --model_name_or_path "$MODEL_PATH" \
  --quantized_model_path "$SAVE_MODEL_PATH" --packed_model_path "$PACKED_MODEL_PATH" \
  --dtype bfloat16 --activation-bits 8 "${MTP_ARGS[@]}" --max-shard-size-mb 2048
python tests/check_step5.py --source "$MODEL_PATH" --packed "$PACKED_MODEL_PATH" "${CHECK_SCOPE_ARGS[@]}"
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
