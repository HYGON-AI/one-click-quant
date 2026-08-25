## MoE-Quant
---
这是一个基于[MoE-Quant](https://github.com/IST-DASLab/MoE-Quant)改造的量化工具，目标是在单机(如八卡H20)环境下可以对大权重模型(如5T级别BF16)做数据集校准的int4量化转换。

### 特性
- 支持GPTQ量化算法。
- 支持逐层量化，节省显存占用。
- 量化流程与模型适配进行抽象隔离。
- 模型适配时支持EP并行，提高计算性能和量化速度。
- 量化输出策略支持group和channel。
- 支持W4A16和W4A8输出配置。

### 安装依赖
```shell
pip install -v flash-attn --no-build-isolation
pip install -v causal-conv1d --no-build-isolation
pip install -v flash-linear-attention --no-build-isolation
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
> 数据集下载和离线使用，可能需要环境变量HF_ENDPOINT=https://hf-mirror.com或HF_DATASETS_OFFLINE=1

### 模型打包命令
```shell
python pack_quantized_model.py \
    --model_name_or_path $MODEL_PATH \
    --quantized_model_path $QUANTIZED_MODEL_PATH \
    --packed_model_path $QUANTIZED_MODEL_PATH-packed \
    --dtype bfloat16
```
