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
`--ignore` 优先级高于 `--force_int8`。

### 模型打包命令
```shell
python pack_quantized_model.py \
    --model_name_or_path $MODEL_PATH \
    --quantized_model_path $QUANTIZED_MODEL_PATH \
    --packed_model_path $QUANTIZED_MODEL_PATH-packed \
    --dtype bfloat16
```

### 模型示例

- [Step5 GPTQ](../../examples/step/README.md)
- [GLM-5.3 / GLM-5.3-Flash GPTQ](../../examples/glm/README.md#glm-53--glm-53-flash-混合精度-gptq-w4a8)

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

