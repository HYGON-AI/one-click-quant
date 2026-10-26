### MiMo-V2.6-Flash-RL-channel-INT4-w4a8
提供以下两种量化方案
#### 方案一：直接转换（RTN）
```bash
python examples/mimo/quantize_mimo_v26_flash_rl_int4_channel.py \
  --input-dir /llm-models/XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --output-dir /quant_models/MiMo-V2.6-Flash-RL-Channel-INT4-W4A8 \
  --device cuda:0
```

#### 方案二：使用 MoE-Quant（GPTQ）

使用校准数据集量化，需先切换到moe-quant-v2分支，然后运行：

```bash
git checkout moe-quant-v2
cd utils/MoE-Quant

# 只需修改输入、输出模型目录
INPUT_DIR=/llm-models/XiaomiMiMo/MiMo-V2.6-Flash-RL
OUTPUT_DIR=/quant_models/MiMo-V2.6-Flash-RL-Channel-INT4-W4A8-GPTQ

MODEL_PATH="$INPUT_DIR" \
PACKED_MODEL_PATH="$OUTPUT_DIR" \
SAVE_MODEL_PATH="${OUTPUT_DIR}-intermediate" \
bash tests/run_mimo_v26_flash_rl_w4a8.sh
```
