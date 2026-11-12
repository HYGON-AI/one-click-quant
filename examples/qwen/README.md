### Qwen3.5-397B-A17B-CHANNEL-FP8

```bash
cd /one_click_quant
python3 templates/qwen3_5_bf16_to_channel.py --input-path /models/Qwen3.5-397B-A17B --output-path /models/Qwen3.5-397B-A17B-CHANNEL-FP8 --quant-type fp8
```

### Qwen3.5-397B-A17B-Channel-INT8-w8a8

```bash
cd /one_click_quant
python3 templates/qwen3_5_bf16_to_channel.py \
  --input-path /llm_models/qwen3.5/Qwen3.5-397B-A17B \
  --output-path /quant_models/Qwen3.5-397B-A17B-Channel-INT8-w8a8 \
  --quant-type int8
```

### Qwen3.8-Flash-Next-MXFP4

```bash
cd /one_click_quant
python3 examples/qwen/quantize_qwen3_8_flash_next_nvfp4_to_mxfp4.py \
  --input-dir /model/Qwen3.8-Flash-Next-NVFP4 \
  --output-dir /model/Qwen3.8-Flash-Next-MXFP4 \
  --mode w4a4
```
