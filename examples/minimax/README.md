### QMiniMax-M2.5-W8A8-INT8

```bash
vllm v0.19.0 or
http://10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.12-ubuntu22.04-dtk26041-py3.10-scalex40-sf_b020-20260810-0214
pip install -i https://mirrors.aliyun.com/pypi/simple llmcompressor==0.11.0
pip install -i https://mirrors.aliyun.com/pypi/simple numpy==1.25.0
python3.10 -m pip install -U "huggingface-hub>=1.5.0,<2.0" "kernels>=0.12.0,<0.13"
python3.10 -m pip install "huggingface-hub>=0.34.0,<1.0"
python3 main.py --model /models/MiniMax-M2.5 --scheme FP8_TO_BF16 --save-dir /models/MiniMax-M2.5-BF16
python main.py --model /models/MiniMax-M2.5-BF16 --scheme W8A8 --alg ptq --needs-data false --ignore 'lm_head,re:.*moe.gate+,re:.*moe.e_score_correction_bias+' --save-dir /models/MiniMax-M2.5-W8A8
```

### MiniMax-M3.1 NVFP4 → channel FP8 转换验证

`nvfp4_to_channel_fp8_probe.py` 用于验证原生混合精度 checkpoint 中 **NVFP4
专家矩阵**转为逐输出通道 FP8 的数值路径。它是 CPU 上的矩阵级 probe，
不是完整 checkpoint 导出器，不需要 GPU、FP4 硬件或校准数据集。

#### 环境和运行

需要 `torch`（支持 `torch.float8_e4m3fn`）、`safetensors` 和
`compressed-tensors`。参考解码使用后者的 NVFP4 内部接口；已通过合成 checkpoint 验证的
环境为 PyTorch 2.11.0、safetensors 0.8.0、compressed-tensors 0.17.0。

在仓库根目录运行：

```bash
python examples/minimax/nvfp4_to_channel_fp8_probe.py \
    --model /path/to/MiniMax-M3.1 \
    --threads 4 \
    --output /path/to/reports/minimax-m3.1-nvfp4-channel-fp8.json
```

`--model` 必须包含 `model.safetensors.index.json` 及其引用的权重分片。
`--output` 可省略；指定时请先创建其父目录，并使用模型目录之外的新报告路径，
避免覆盖已有文件。输出是 JSON 报告，不是量化权重；每个矩阵的结果和最终
`REPORT=...` 汇总也会打印到标准输出。

#### 输入约定和转换过程

固定抽取 `(layer, expert) = (3, 0), (31, 63), (59, 127)`，每组处理
`w1/w2/w3`，共 9 个完整矩阵。键名前缀为：

```text
language_model.model.layers.<layer>.block_sparse_moe.experts.<expert>.<projection>
```

每个前缀要求以下张量，其中 `N` 为输出通道数、`K` 为输入通道数：

| 后缀 | 形状 / 用途 |
| --- | --- |
| `weight_packed` | uint8 `[N, K/2]`，每字节两个 E2M1 值，低 4 位在前 |
| `weight_scale` | `[N, K/16]`，每行每 16 个输入列一个 block scale |
| `weight_global_scale` | 单元素、有限且大于 0，权重编码倍率 |
| `input_global_scale` | 激活量化参数，仅读取并记录，不参与权重解码 |

转换公式：

```text
W[i,j] = E2M1[i,j] * weight_scale[i,j//16] / weight_global_scale
s[i]   = max(abs(W[i,:])) / 448          # 全零行使用 s[i] = 1
Q[i,j] = E4M3FN(clamp(W[i,j] / s[i], -448, 448))
W_hat  = Q.float() * s
```

此 checkpoint 约定下 global scale 是编码倍率，因此反量化时使用除法。
最终 `Q` 为 `[N, K]` 的 `torch.float8_e4m3fn`，`s` 为 FP32 `[N, 1]`
解码 scale。FP32 中间值只是源 NVFP4 权重的重建，不能恢复原先量化丢失的信息。

#### 验证和报告解读

- 枚举全部 256 种字节，与 `compressed_tensors` 解包结果逐值比较，包含正负零符号位。
- 将每个矩阵的 NVFP4 反量化结果与 `compressed_tensors.dequantize` 做精确相等检查。
- `weight_relative_rmse`：`||W_hat - W|| / ||W||`，衡量新增 FP8 重量化误差。
- `random_linear_relative_rmse`：`||X @ W_hat.T - X @ W.T|| / ||X @ W.T||`，
  使用固定随机种子 2026 和 `[8, K]` 的 FP32 正态随机输入。

解码不一致会触发断言；误差指标只报告数值，没有预设通过阈值。
请用普通 `python` 运行，不要使用会禁用断言的 `python -O`。

#### 与完整 W8A8 转换的边界

脚本不处理 checkpoint 中原生 FP8 矩阵，不量化激活 A8，不执行真实 FP8 GEMM，
不保存 `Q/s` 或更新模型配置，也不验证全模型推理精度和性能。
不能将随机 FP32 线性层的误差直接解释为完整 W8A8 的精度结果。
若要导出可加载的 channel FP8 W8A8 模型，还需完成全模型遍历、原生 FP8
权重处理、激活量化策略、权重及配置导出，并验证目标推理框架的实际加载和执行。
