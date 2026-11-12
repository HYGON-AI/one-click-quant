## MoE-Quant
---
这是一个基于[IST-DASLab/MoE-Quant](https://github.com/IST-DASLab/MoE-Quant)改造的量化工具，目标是在单机环境下可以对大权重模型(如5T级别BF16)做数据集校准的int4量化转换。

### 特性
- 支持GPTQ量化算法。
- 支持逐层量化，节省显存占用。
- 量化流程与模型适配进行抽象分离。
- 模型适配时支持EP并行，减少单卡显存占用，提高计算性能。
- 量化输出策略支持group和channel。
- 支持W4A16和W4A8输出配置。

### 安装依赖
```shell
pip install flash-attn --no-build-isolation
pip install causal-conv1d --no-build-isolation
pip install flash-linear-attention --no-build-isolation
```

### 模型量化命令
```shell
torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
  --model_name_or_path $MODEL_PATH \
  --dataset_name_or_path open-platypus \
  --num_calibration_samples 512 \
  --max_sequence_length 4096 \
  --bits 4 \
  --group_size 128 \
  --rel_damp 0.1 \
  --sym \
  --quantize_only_experts \
  --dtype bfloat16 \
  --save_dir $QUANTIZED_MODEL_PATH
```

### 量化范围控制（`--ignore`）

`--quantize_only_experts` 是历史开关（只量化路由专家）。需要量化 attention /
linear_attn 投影等其它层时，改用 `--ignore`：

```shell
torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
  --model_name_or_path $MODEL_PATH \
  --dataset_name_or_path open-platypus \
  --num_calibration_samples 512 \
  --max_sequence_length 4096 \
  --bits 4 \
  --group_size 128 \
  --rel_damp 0.1 \
  --sym \
  --ignore "re:.*self_attn\.(q|k|v|o)_proj$" \
  --dtype bfloat16 \
  --save_dir $QUANTIZED_MODEL_PATH
```

量化范围 = **全部 `nn.Linear` − 适配器结构默认项 − `--ignore` 追加项**。
规则语法与 compressed-tensors 一致：`re:<regex>` 走 `re.match`（从头匹配），
裸字符串为精确等值。注意 vLLM 会融合 `q/k/v_proj` 与 `gate/up_proj`，
这三者必须同进同退，否则加载时报 `vLLM requires all to use the same scheme`。

> **W4A8 说明**：DCU 上 sglang 的 `HCUCompressedTensorsW4A8Int8DynamicMoE` 可加载
> compressed-tensors INT4（pack-quantized, channel）路由专家 + 动态 per-token INT8 激活，
> 其它 Linear 需为 INT8 W8A8（int-quantized）。用 `--force_int8` 指定 GPTQ 量化为 INT8 的层，
> 打包时加 `--activation-bits 8`，会生成 `group_int4` + `group_int8` 两个 config_groups。

### 混合精度（`--force_int8`）

`--force_int8` 与 `--ignore` 语法相同，匹配到的层按 INT8 做 GPTQ，其余未 ignore 的层用 `--bits`；
`--ignore` 优先级高于 `--force_int8`。GLM-5.3 的 MTP 层 78、GLM-5.3-Flash 的 MTP 层 45 不参与 GPTQ，打包时加
`--mtp-int8` 以 RTN INT8 量化（与参考脚本逐字节一致）。
普通 GLM-5.3 还支持打包时使用 `--mtp-int4`：MTP routed experts 为 INT4 RTN，
其它未 ignore 的 MTP Linear 为 INT8 RTN。两个 MTP 参数互斥；均不指定时 MTP 保持 BF16。

与 `GLM-5.3-Flash-CHANNEL-INT4-w4a8-v3-same-as-iter_0033200` 量化范围一致的完整命令见
[tests/run_glm5_flash_w4a8.sh](./tests/run_glm5_flash_w4a8.sh)。

### GLM-5.3 BF16 → 混合 INT4/INT8 W4A8

主干 layers.3–77 的 routed experts 为 channel INT4，dense/shared MLP、MLA
q_a/q_b/kv_a/o、indexer wq_b/wk 为 channel INT8；MTP layers.78 的相应 Linear 为 INT8。
router、kv_b_proj、indexer weights_proj、norm、embedding、lm_head 和 MTP 专用投影/头保持 BF16。

#### 默认：MTP 用 INT8 → `GLM-5.3-CHANNEL-INT4-w4a8-GPTQ`

```bash
cd /my_code/Quantitative_Code/one-click-quant/utils/MoE-Quant
bash tests/run_glm5_3_w4a8.sh
```

输出：

- 中间结果 `/quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-intermediate`
- 打包产物 `/quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ`

默认读取 `/llm-models/GLM-5.3-BF16` 和本地 `OpenThoughts-114k`，8 卡、512 条校准数据、
最大长度 4096。可用 `MODEL_PATH`、`DATASET`、`SAVE_MODEL_PATH`、`PACKED_MODEL_PATH`、
`NPROC_PER_NODE`、`NUM_CALIBRATION_SAMPLES`、`MAX_SEQUENCE_LENGTH`、`MASTER_PORT` 覆盖。

#### MTP 的 routed experts 也用 INT4 → `GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-MTP-INT4`

```bash
cd /my_code/Quantitative_Code/one-click-quant/utils/MoE-Quant
bash tests/run_glm5_3_w4a8.sh --mtp-int4
```

输出打包产物 `/quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-MTP-INT4`。
主干校准和中间结果与默认方案相同，因此已有中间结果时只需重新打包：

```bash
python pack_quantized_model.py \
  --model_name_or_path /llm-models/GLM-5.3-BF16 \
  --quantized_model_path /quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-intermediate \
  --packed_model_path /quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-MTP-INT4 \
  --dtype bfloat16 --activation-bits 8 --mtp-int4
```

当前 `--mtp-int4` 仅支持普通 GLM-5.3（`glm_moe_dsa`），Flash 仍使用 `--mtp-int8`。

### 模型打包命令
```shell
python pack_quantized_model.py \
    --model_name_or_path $MODEL_PATH \
    --quantized_model_path $QUANTIZED_MODEL_PATH \
    --packed_model_path $QUANTIZED_MODEL_PATH-packed \
    --dtype bfloat16
```

### Step5 混合精度 W4A8 / W8A8

Step5 有两套固定配方，由 `tests/run_step5_w4a8_gptq.sh` 的 `RECIPE` 变量选择。
量化范围在 adapter 内按配方写死，不接受 `--force_int8`。

| 配方 | `RECIPE` | 路由专家 | attention / 普通 MLP / 共享 MLP | `lm_head` | MTP |
|---|---|---|---|---|---|
| 仅专家 | `experts` | INT4 W4A8 GPTQ | BF16 | BF16 | BF16 |
| 混合精度 | `mixed` | INT4 W4A8 GPTQ | INT8 W8A8 GPTQ | INT8 RTN | INT8 RTN |

两套配方下 router、norm、embedding、`eh_proj`、视觉模块和 DSA 索引器都保持
源精度，layers 88–90 的路由专家也保持 BF16。

#### 配方一：仅量化路由专家 → `step-5-preview-channel-w4a8-GPTQ`

```bash
cd /my_code/Quantitative_Code/one-click-quant/utils/MoE-Quant

HIP_VISIBLE_DEVICES=2,5,6,7 \
NPROC_PER_NODE=4 \
MODEL_PATH=/DL_DATA/step-5-preview-0913-fp8-channel \
DATASET=/quant_models/step5-gptq-calibration/sharegpt_code_general_512_seed0.jsonl \
NUM_CALIBRATION_SAMPLES=512 \
MAX_SEQUENCE_LENGTH=4096 \
RECIPE=experts \
bash tests/run_step5_w4a8_gptq.sh --run
```

输出：

- 中间结果 `/quant_models/step-5-preview-channel-w4a8-GPTQ-intermediate`
- 打包产物 `/quant_models/step-5-preview-channel-w4a8-GPTQ`

#### 配方二：INT4 专家 + INT8 主干 → `step-5-preview-int4-int8-GPTQ`

```bash
cd /my_code/Quantitative_Code/one-click-quant/utils/MoE-Quant

HIP_VISIBLE_DEVICES=2,5,6,7 \
NPROC_PER_NODE=4 \
MODEL_PATH=/DL_DATA/step-5-preview-0913-fp8-channel \
DATASET=/quant_models/step5-gptq-calibration/sharegpt_code_general_512_seed0.jsonl \
NUM_CALIBRATION_SAMPLES=512 \
MAX_SEQUENCE_LENGTH=4096 \
RECIPE=mixed \
bash tests/run_step5_w4a8_gptq.sh --run
```

输出：

- 中间结果 `/quant_models/step-5-preview-int4-int8-GPTQ-intermediate`
- 打包产物 `/quant_models/step-5-preview-int4-int8-GPTQ`

### 已支持模型
- DeepSeek-V3 (DeepseekV3ForCausalLM)
- Qwen3.8-2.4T-A95B-FP8 (Qwen3_5MoeForCausalLM)
- Qwen3.5-35B-A3B / Qwen3.5-397B-A17B (Qwen3_5MoeForConditionalGeneration，多模态包装类)
- Kimi-K3 (KimiK3ForConditionalGeneration)
- GLM-5.3 (GlmMoeDsaForCausalLM)
- GLM-5.3-Flash (Glm5NextForConditionalGeneration)
- MiMo-V2.6-Flash-RL (MiMoV2ForCausalLM)：源 MXFP4→channel INT4 GPTQ、FP8→channel INT8 GPTQ、BF16/FP32 保留；旧 MTP 沿用 RTN。

MiMo 的完整适配、启动命令和验收步骤见 [MiMo-V2.6-Flash-RL-GPTQ适配说明.md](./MiMo-V2.6-Flash-RL-GPTQ适配说明.md)。
先运行 `bash tests/run_mimo_v26_flash_rl_w4a8.sh --preflight`，正式量化运行
`bash tests/run_mimo_v26_flash_rl_w4a8.sh`。默认四卡，输出使用独立 GPTQ 目录。

> Qwen3.5 多模态包装类只量化文本主干；视觉塔 (`model.visual.*`) 与 MTP 头 (`mtp.*`)
> 以 BF16 原样搬运到打包产物，不参与 GPTQ，也不会被丢弃。

详见 [MoE-Quant-ignore改造说明.md](./MoE-Quant-ignore改造说明.md)。

