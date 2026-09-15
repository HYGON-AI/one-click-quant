### GLM-5.3-Channel-INT4-w4a16 / GLM-5.3-Flash-Channel-INT4-w4a16

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
  python3 pack_quantized_model.py \
    --model_name_or_path "${MODEL_PATH}" \
    --quantized_model_path "${SAVE_MODEL_PATH}" \
    --packed_model_path "${SAVE_MODEL_PATH}-packed" \
    --dtype bfloat16 \
    --activation-bits 16
```

### GLM-5.3-Channel-INT8-w8a8

```bash
  cd one-click-quant
  python3 examples/glm/quantize_glm5_3_w8a8_channel.py --model-id /models/GLM-5.3 --save-dir /models/GLM-5.3-Channel-INT8-w8a8 --quantize-shared-experts
```

### GLM-5.3-Flash-Channel-INT8-w8a8

```bash
  cd one-click-quant/utils/MoE-Quant
  torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name_or_path open-thoughts \
    --num_calibration_samples 512 \
    --max_sequence_length 4096 \
    --bits 8 \
    --rel_damp 0.1 \
    --sym \
    --quantize_only_experts \
    --attn_implementation sdpa \
    --dtype bfloat16 \
    --save_dir "${SAVE_MODEL_PATH}"
  python3 pack_quantized_model.py \
    --model_name_or_path "${MODEL_PATH}" \
    --quantized_model_path "${SAVE_MODEL_PATH}" \
    --packed_model_path "${SAVE_MODEL_PATH}-packed" \
    --dtype bfloat16 \
    --activation-bits 8
```

