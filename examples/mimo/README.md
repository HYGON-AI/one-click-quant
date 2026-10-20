### MiMo-V2.6-Flash-RL-channel-INT4-w4a8
```bash
cd one-click-quant
python examples/mimo/quantize_mimo_v26_flash_rl_int4_channel.py \
  --input-dir /llm-models/XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --output-dir /quant_models/MiMo-V2.6-Flash-RL-Channel-INT4-W4A8 \
  --device cuda:0
```
