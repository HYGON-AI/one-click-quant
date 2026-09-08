### GLM-5.3-Channel-INT4-w4a16

```bash
  cd one-click-quant/utils/MoE-Quant
  torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name_or_path open-thoughts \
  --num_calibration_samples 512 \
  --max_sequence_length 4096 \
  --bits 4 \
  --rel_damp 0.1 \
  --sym \
  --quantize_only_experts \
  --attn_implementation sdpa \
  --dtype bfloat16 \
  --save_dir "${SAVE_MODEL_PATH}"
```

