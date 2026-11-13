### GLM-5.3-Channel-FP8-w8a8

```bash
  cd one-click-quant
  python3 examples/glm/quantize_glm5_3_fp8_dynamic.py \
  --model-path /models/GLM-5.3 \
  --save-dir /models/GLM-5.3-Channel-FP8-w8a8 \
  --quantize-shared-experts --quantize-first-3-layers --quantize-indexer
```

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
  
#  OR
  
  python quantize_glm5_3_flash_int8_channel.py \
    --input-dir /models/GLM-5.3-Flash \
    --output-dir /models/GLM-5.3-Flash-Channel-INT8-w8a8 \
    --no-mla-to-bf16 --linear-attn-int8 \
    --kda-int8-scope qkvo --indexer-int8
```

### GLM-5.3-Flash-Channel-INT4-w4a8

```bash
  cd one-click-quant
  python3 examples/glm/quantize_glm5_3_flash_int4_channel.py \
    --input-dir /models/GLM-5.3-Flash \
    --output-dir /models/GLM-5.3-Flash-Channel-INT4-w4a8 \
    --no-mla-to-bf16 --linear-attn-int8 \
    --kda-int8-scope qkvo --indexer-int8

# OR

  cd one-click-quant
  python3 examples/glm/quantize_glm5_3_flash_int4_channel.py \
    --input-dir /models/GLM-5.3-Flash \
    --output-dir /models/GLM-5.3-Flash-Channel-INT4-w4a8 \
    --no-mla-to-bf16 --linear-attn-int8 \
    --kda-int8-scope qkvo --indexer-int8 --mtp-int4
```
### GLM-5.3-Channel-INT4-w4a8
```
cd /quant_models/one-click-quant/examples/glm
python quantize_glm5_3_int4_channel.py \
  --input-dir  /llm-models/GLM-5.3 \
  --output-dir /quant_models/GLM-5.3-CHANNEL-INT4-w4a8
```
若 MTP routed experts 也需要 INT4，将运行命令改为：

```bash
cd /quant_models/one-click-quant/examples/glm
python quantize_glm5_3_int4_channel.py \
  --mtp-int4 \
  --input-dir  /llm-models/GLM-5.3 \
  --output-dir /quant_models/GLM-5.3-CHANNEL-INT4-w4a8
```

### GLM-5.3 / GLM-5.3-Flash 混合精度 GPTQ W4A8

在仓库根目录运行：

```bash
cd utils/MoE-Quant
bash tests/run_glm5_3_w4a8.sh            # GLM-5.3，MTP INT8 RTN
bash tests/run_glm5_3_w4a8.sh --mtp-int4 # GLM-5.3，MTP 路由专家 INT4 RTN
bash tests/run_glm5_flash_w4a8.sh       # GLM-5.3-Flash，MTP INT8 RTN
```

GLM-5.3 的 3–77 层路由专家用 channel INT4 GPTQ，普通/共享 MLP、MLA 的 q_a/q_b/kv_a/o
和 indexer 的 wq_b/wk 用 channel INT8 GPTQ。router、kv_b_proj、indexer 的 weights_proj、
norm、embedding、lm_head 和 MTP 专用投影/头保留 BF16。

GLM-5.3 的第 78 层和 GLM-5.3-Flash 的第 45 层为 MTP 层，不参与 GPTQ。
打包时 `--mtp-int8` 和 `--mtp-int4` 互斥；均不指定则保留 BF16。`--mtp-int4` 仅用于普通
GLM-5.3，MTP 路由专家用 INT4 RTN，其余符合量化范围的 MTP Linear 用 INT8 RTN。
Flash 的具体范围见 [运行脚本](../../utils/MoE-Quant/tests/run_glm5_flash_w4a8.sh)。

普通 GLM-5.3 的两种 MTP 配方共用 GPTQ 中间结果。已有中间结果时，可以直接重新打包：

```bash
python pack_quantized_model.py \
  --model_name_or_path /llm-models/GLM-5.3-BF16 \
  --quantized_model_path /quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-intermediate \
  --packed_model_path /quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-MTP-INT4 \
  --dtype bfloat16 --activation-bits 8 --mtp-int4
```

默认模型、校准数据和输出目录见 [GLM-5.3 脚本](../../utils/MoE-Quant/tests/run_glm5_3_w4a8.sh)。
