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