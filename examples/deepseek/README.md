### DeepSeek-V4-Pro-0813-Channel-FP8-W8A8

```bash
cd one-click-quant
python3 /one-click-quant/examples/deepseek/deepseek_v4_w8a8_channel.py \
  --input-dir /llm_models/data/DeepSeek-V4-Pro-0813 \
  --output-dir /quant_models/DeepSeek-V4-Pro-0813-Channel-FP8-W8A8 \
  --output-format fp8-channel \
  --num-threads 32
```

### DeepSeek-V4-Pro-0813-Channel-INT4-w4a8

```bash
cd one-click-quant
python3 examples/deepseek/mixed_w4a8_int4_attention_w8a8_int8_channel.py \
  --input-dir /llm_models/DeepSeek-V4-Pro-0813 \
  --output-dir /quant_models/DeepSeek-V4-Pro-0813-Channel-INT4-w4a8 \
  --scale-divisor 16 \
  --num-threads 8
```

### DeepSeek-V4.1-Flash-Channel-FP8-w8a8

```bash
cd one-click-quant
python3 examples/deepseek/quantize_deepseek_v4_1_flash_fp8_channel.py \
  --input-dir /llm_models/DeepSeek-V4.1-Flash \
  --output-dir /quant_models/DeepSeek-V4.1-Flash-Channel-FP8-w8a8 \
  --num-threads 32
```

