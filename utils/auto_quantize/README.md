# AutoQuantize 精度分析工具

## 简介

one-click-quant 中的混合精度分析工具。基于 NVIDIA ModelOpt，输入模型和校准数据，自动选择哪些目标模块使用 **INT4**、哪些使用 **INT8**，输出 `autoquantize_plan.json`，供后续 RTN/GPTQ 量化使用。

自动保留源配置的 `ignore/ignored_layers` 等排除规则，Q/K/V、gate/up 等按融合组统一决策。**这里只生成分配计划，不导出量化模型。**

支持浮点模型（BF16/FP16/FP32），以及已接入源编码的 Block FP8、MXFP4/FP8/BF16 混合模型。模型架构需被 Transformers 和 ModelOpt 支持；量化源权重需为 safetensors 格式。
默认单设备模式需要容纳完整解码模型及评分开销。MiMo、Kimi-K3/K3.1 可通过本工具的 `--parallel ep` 使用多卡专家分片、逐块加载和反向重算，无需加载完整浮点模型到单卡。

## 使用方法

在已安装适配当前 NVIDIA/DCU 的 PyTorch 环境中安装依赖，保留厂商适配的 PyTorch：

```bash
python -m pip install "nvidia-modelopt[hf]==0.47.0"
```

从 **one-click-quant 根目录**运行，替换模型目录和输出路径：

```bash
python -m utils.auto_quantize \
  --model /path/to/model \
  --calibration-data open-platypus \
  --calibration-samples 8 \
  --calibration-seq-length 512 \
  --effective-bits 4.2 \
  --device cuda:0 \
  --output /path/to/analysis/autoquantize_plan.json
```

完整 MiMo 使用多卡 EP，例如以下两卡命令（替换成空闲卡）：

```bash
CUDA_VISIBLE_DEVICES=4,5 HIP_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m utils.auto_quantize \
  --parallel ep --trust-remote-code \
  --model /llm-models-1/XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --calibration-data open-platypus \
  --calibration-samples 8 --calibration-seq-length 512 \
  --effective-bits 4.2 --device cuda:0 \
  --output /quant_models/models/MiMo-AutoQuantize-EP/autoquantize_plan.json
```

EP 支持 MiMo 和 Kimi-K3/K3.1，未实现通用 TP；专家数需能被进程数整除，实际校准序列块数需为进程数的正整数倍。八卡使用 `--nproc_per_node=8` 和八个可见设备。增加 `--dry-run` 可只检查元数据与分片，不解码大权重、不输出计划。

Kimi-K3 使用相同命令，将 `--model` 改为 `/llm_models_2/Kimi-K3`，并选择新的输出路径。运行环境需有源模型依赖的 `fla-core`，且 KDA 算子支持反向传播。该路径只构造文本模型，按 decoder block 加载本 rank 的专家和共享权重，在块内复用，前向结束释放，反向重算时重新加载；支持源目录的嵌套 MXFP4 配置及 `weight_packed`/`weight_scale`。函数式 attention residual 投影保持浮点，不参与搜索。

模型使用本地 HuggingFace 目录，包含配置、权重和 tokenizer。需要模型自定义代码时，增加 `--trust-remote-code`。DCU 同样使用 `cuda:0` 设备名。

`open-platypus` 首次使用需要下载或已有缓存，也可将 `--calibration-data` 替换成本地 `.txt`、`.json`、`.jsonl` 文件。JSONL 每行一条文本，例如：

```json
{"text": "请解释二分查找，并写一个 Python 实现。"}
```

## 参数与结果

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model` | 必填 | 本地模型目录 |
| `--output` | 必填 | 输出 JSON 路径，不能覆盖已有文件 |
| `--calibration-data` | 必填 | 校准数据集或本地文本路径 |
| `--calibration-samples` | `8` | 最多使用的校准序列块数 |
| `--calibration-seq-length` | `512` | 每个序列块的最大 token 数 |
| `--calibration-batch-size` | `1` | 每个 rank 每次评分合并的序列数；不改变总 samples |
| `--effective-bits` | `4.2` | 搜索权重的平均位宽上限，范围 4～8 |
| `--device` | `cuda:0` | 分析设备；EP 模式按 LOCAL_RANK 选择对应可见卡 |
| `--parallel` | `single` | `ep` 启用 MiMo/Kimi 专家分片与逐块加载 |
| `--no-reuse-amax` | 不设置 | 设置后使用原 ModelOpt 候选权重扫描，供局部对照 |
| `--mxfp4-decode` | `auto` | EP 下有 CUDA/HIP 和 Triton 时在 GPU 解码；可显式选 `cpu`/`gpu` |
| `--prefetch-blocks` | `0` | 设为 `1`，在独立 stream 预取下一个 block，支持反向重计算 |
| `--prefetch-budget-gib` | `16` | 每个 rank 允许额外预取的解码块大小上限 |
| `--prefetch-reserve-gib` | `16` | 预取后为评分预留的空闲显存；预算不足跳过预取 |
| `--dry-run` | 关闭 | EP 元数据与分片预检，不执行评分 |
| `--dtype` | `bfloat16` | 分析工作精度，可选 BF16/FP16/FP32 对应的 `bfloat16/float16/float32` |
| `--exclude` | 无额外排除 | 额外保留浮点的模块，例如 `--exclude lm_head` |
| `--include` | 不限制范围 | 只搜索匹配的模块，例如 `--include 'model.layers.0.*'` |
| `--policy` | 不使用 | 复用包含 include/exclude/分组规则的 JSON 配置文件 |

通常不需要设置 `exclude/include/policy`。排除规则优先，include 不会重新启用源配置已排除的层。

`effective-bits` 是**位宽预算**，不是以 GB/s 表示的传输带宽。如何选择及计算见下文；它也不同于以 GiB 表示的预取显存预算。

输出json主要查看：

| 字段 | 含义 |
|---|---|
| `module_bits` | 每个搜索目标的位宽，`4` 为 INT4、`8` 为 INT8 |
| `excluded_modules` | 不应用本次搜索策略的目标及原因；分析时使用源权重解码后的工作精度 |
| `policy` | 候选位宽、per-channel 权重、动态 per-token A8、校准方法及分组规则 |
| `model`、`schema_version` | 模型标识、配置/源结构指纹及计划格式版本；用于匹配正确模型 |
| `effective_bits_target` / `actual_effective_bits` | 预算上限 / 参数量加权后的实际平均位宽 |
| `groups` | 全部决策组的成员、候选分数、选定位宽、覆盖情况和参数成本 |
| `cost.weight_elements` / `cost.weight_payload_bytes` | 搜索范围内去重后的权重元素数 / 估算的纯权重载荷字节数 |
| `calibration` | 实际 token 数、输入摘要、请求样本数、序列长度、数据源及 seed |
| `coverage` | 计划结构完整性及不确定组；`complete=true` 不代表所有专家都被校准数据触发 |
| `execution` | EP 执行策略、分析耗时、逐块加载次数和优化统计；单设备计划不一定有此字段 |
| `modules` 中的 shape/dtype | 分析模型的结构和工作精度，供校验及成本计算；不是最终 INT4/INT8 的存储 dtype |

命令还会输出 `autoquantize_plan.sensitivity.json` 和可阅读的 `.sensitivity.txt`。**两份敏感度报告和终端均只显示前 10 个决策组**，不足 10 个则全部显示，按 W4A8 分数降序排列。TXT 包含字段说明；JSON 包含 `field_descriptions`、`total_groups`、`reported_groups`、`truncated`。报告不展开全量 `module_names`，可用 `group_key` 到主计划 `groups` 中查询。

**主计划的 `module_bits`、`groups` 和全部候选分数保持完整**，全局求解仍使用所有组。不要拿 top-10 报告代替主计划进行量化。报告裁剪不需要重新跑模型。[报告实现](sensitivity.py)、[主计划生成及校验](api.py)。

后续由支持此计划格式的 RTN/GPTQ 脚本读取 JSON，执行量化并评测精度。**当前通用计划不能直接交给旧 MiMo 专用 RTN/GPTQ 脚本。**

完整参数使用 `python -m utils.auto_quantize --help` 查看。

## 结果示例与解读

以下是历史 Kimi-K3.1-Pro 分析结果按新报告格式展示的表格部分：EP4、8 条 512-token 序列、4096 个实际 token、预算 4.2。不是优化版全模型重新运行的结果。层号沿用模型模块名，**从 0 开始**。

```text
AUTOQUANTIZE SENSITIVITY REPORT
Base model: /DL_DATA/llm-models/moonshotai/Kimi-K3.1-Pro
Analysis: modelopt_gradient
Ranked fusion groups: 92; assigned modules: 247296
Showing top 10 of 92 groups; full decisions/members remain in plan JSON.
Effective bits: target=4.2, actual=4.173913043478261

Rank  Group                                                    W4A8 score    W8A8 score    S4-S8         Bit  Params          Members  Coverage / constraints
1     language_model.model.layers.47.block_sparse_moe.experts  2.010889e-05  9.209695e-07  1.918792e-05  8    29,595,009,024  2688     unobserved=129
2     language_model.model.layers.75.block_sparse_moe.experts  1.970567e-05  3.595002e-07  1.934617e-05  8    29,595,009,024  2688     unobserved=111
3     language_model.model.layers.72.block_sparse_moe.experts  1.714225e-05  4.714016e-07  1.667085e-05  8    29,595,009,024  2688     unobserved=126
4     language_model.model.layers.49.block_sparse_moe.experts  1.703852e-05  5.439160e-07  1.649461e-05  8    29,595,009,024  2688     unobserved=354
5     language_model.model.layers.74.block_sparse_moe.experts  1.620423e-05  3.258286e-07  1.587840e-05  4    29,595,009,024  2688     unobserved=219
6     language_model.model.layers.51.block_sparse_moe.experts  1.533788e-05  3.821094e-07  1.495577e-05  4    29,595,009,024  2688     unobserved=225
7     language_model.model.layers.46.block_sparse_moe.experts  1.524254e-05  7.328371e-07  1.450970e-05  4    29,595,009,024  2688     unobserved=156
8     language_model.model.layers.52.block_sparse_moe.experts  1.512770e-05  8.415508e-07  1.428615e-05  4    29,595,009,024  2688     unobserved=342
9     language_model.model.layers.41.block_sparse_moe.experts  1.497706e-05  5.603221e-07  1.441673e-05  4    29,595,009,024  2688     unobserved=108
10    language_model.model.layers.76.block_sparse_moe.experts  1.424347e-05  2.874605e-07  1.395601e-05  4    29,595,009,024  2688     unobserved=294
```

实际 TXT 在表格后继续列出字段说明。阅读顺序是：先看覆盖是否充分，再看 W4 分数与升级收益，最后结合预算理解 `Bit`。

| 表格列 / JSON 字段 | 怎么解读 |
|---|---|
| `Rank` / `rank` | 按 W4A8 分数排的名次；不是按升级收益或收益/成本排序 |
| `Group` / `group` | 一次统一位宽决策的范围；此例每行是某层全部 routed experts，不是单个专家 |
| `W4A8 score` / `int4_score` | 当前数据下 W4A8 的累计敏感度代理值；越大表示扰动更敏感，不是 loss 增量或准确率损失 |
| `W8A8 score` / `int8_score` | 同一组 W8A8 的代理值；EP 汇总后存储值取 `min(S4, S8_raw)`，不要把包络处理误认为实测 W8 总能提升精度 |
| `S4-S8` / `upgrade_gain` | 升级 W8A8 的代理收益。第 2 行收益比第 1 行略大，但 W4 分数更小，所以排第 2 |
| `Bit` / `selected_bits` | 最终权重位宽。第 5 行仍是 W4，主要受全局预算限制，不表示这层不敏感 |
| `Params` / `weight_elements` | 该组计入预算的唯一权重元素数；不同大小的组升级到 W8，成本不同 |
| `Members` / `module_count` | 目标模块数。此例 `2688 = 896 个专家 × 3 个投影(w1/w2/w3)` |
| `Coverage / constraints` | `unobserved=129` 指 129 个目标模块未被观测到前向，不是 129 个专家；`zero_scores` 不证明低敏感度；`allowed=8` 等表示候选受约束 |

梯度评分的含义可写成：

```text
S(g, b) = Σ clamp((∂L/∂Y_g) × (Y_g,b − Y_g,float), −1e10, 1e10)²
```

对观测到的输出元素、评分调用和样本累计；`g` 是评分组，`b` 是候选。它近似衡量局部量化扰动对损失的敏感度，不是重新完整评测量化模型得到的损失。分数受数据、样本量、序列长度和 loss 归一化影响，不能直接跨不同设置比较绝对大小。全局求解使用所有组的候选分数及参数成本，既不是只看前 10 名，也不是简单把 W4 排名最高的若干行升到 W8。[评分调用与提取](api.py)、[EP 汇总与单调包络](ep.py)。

此例中的 `Y_float` 基线来自**源 MXFP4 checkpoint 解码到分析 dtype**，不是训练时未量化的原始模型；分数描述的是在这个源模型上进一步采用 W4A8/W8A8 的扰动。[源解码](loading.py)、[Kimi 加载审计](model_load/kimi.py)。

主计划的执行字段摘录如下；这是示意摘录，不是可直接交给 `load_plan()` 的完整文件：

```json
{
  "module_bits": {
    "language_model.model.layers.47.block_sparse_moe.experts.0.w2": 8,
    "language_model.model.layers.74.block_sparse_moe.experts.0.w2": 4
  },
  "policy": {
    "candidate_bits": [4, 8],
    "weight_granularity": "channel",
    "activation_bits": 8,
    "activation_granularity": "dynamic_per_token",
    "calibration_algorithm": "max_rtn"
  },
  "effective_bits_target": 4.2,
  "actual_effective_bits": 4.173913043478261
}
```

后续量化依据 `module_bits`、`excluded_modules`、`policy` 执行，并检查模型身份。计划未提供量化后的权重或完整 scale，仍需要真正量化及独立精度评测。当前 `load_plan()` 还会检查形状、分组和成本，因此不能直接删除主计划中的结构字段。[校验逻辑](api.py)。

## 位宽预算怎么设置

令 `N_g` 为搜索组 `g` 的唯一权重元素数，`b_g` 为该组最终位宽。共享权重只计一次，排除项不进入分母：

```text
实际平均位宽 B_actual = Σ(N_g × b_g) / ΣN_g ≤ B_target
纯权重载荷字节数      = Σ(N_g × b_g) / 8
```

当候选只有 W4/W8 时，令 `p8` 为搜索权重元素中选为 W8 的比例：

```text
B_actual = 4 × (1 − p8) + 8 × p8 = 4 + 4p8
p8 ≤ (B_target − 4) / 4
```

| `--effective-bits` | W8 权重元素比例上限 | 含义 |
|---|---|---|
| `4.0` | 0% | 全部搜索权重 W4；若存在必须 W8 的组则不可行 |
| `4.2` | 5% | 小额 W8 预算，可作为第一轮分析起点 |
| `4.5` | 12.5% | 为更多敏感权重保留 W8 |
| `5.0` | 25% | 更大的 W8 预算 |
| `8.0` | 100% | 允许所有搜索权重 W8，不代表必须用满预算 |

比例针对**权重元素数**，不是模块数、层数或 MoE 激活参数数。此 Kimi 结果有 92 个等大小搜索组，4.2 的预算最多升级 `floor(92 × 5%) = 4` 组：

```text
B_actual = (88 × 4 + 4 × 8) / 92 = 4.173913043478261
```

因此层 47、49、72、75 使用 W8；预算没用满是整组选择的离散结果。组大小不等时必须使用加权公式，不能套用“4 层”的结论。[组成本及全局求解](ep.py)、[Kimi 分组规则](model_load/kimi.py)。

预算不包含 scale、bias、排除权重、打包 padding、激活/KV 缓存及运行临时空间。它不是最终模型文件大小或显存上限；分析仍使用解码后的 BF16/FP16/FP32 源权重和假量化，降低预算不会把整个分析模型变成 INT4 常驻。应结合目标量化文件大小及独立精度评测选择预算。[成本字段](api.py)、[源权重加载](loading.py)。

## 校准数据与 sample 设置依据

`open-platypus` 是 `garage-bAInd/Open-Platypus` 的别名。该公开数据集以英文逻辑推理为目标，包含数学、科学、代码等来源的指令及回答。[官方数据卡](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)

本次使用它作为工程基线：现有工具能直接读取 `instruction/input/output` 并拼接成文本，数据可固定为本地 JSONL，便于重现旧结果和排除下载时间。这个选择不意味着它是 Kimi 的最优校准集。中文、代码或特定业务模型应按实际输入分布准备有代表性的独立校准文本；不要用最终评测集来挑选量化方案。[读取和拼接实现](calibration.py)。

当前候选是权重 max + 动态 per-token A8：权重 amax 来自权重本身，动态激活也不需要固定的全数据集 amax。**校准文本主要决定梯度敏感度和 MoE 路由覆盖**，因此“权重统计可复用”并不意味着可以取消数据或任意减少样本。[候选配置与评分](api.py)、[amax 复用](max_calibration.py)。

`--calibration-samples` 指最多生成多少个 token 序列块，不是原始 JSONL 行数。一条长文本可以贡献多个块；末块允许短于指定长度，只有至少 2 个 token 的块会被采用。当前按数据源顺序取前若干有效块，**没有自动随机抽样**；`--seed` 不会把这一步变成随机选样。需要均衡抽样时，应先准备并固定本地 JSONL。[load_calibration](calibration.py)。

| 设置 | 本次选择的理由 | 限制与调整方法 |
|---|---|---|
| `samples=8` | 控制首次大模型分析成本，并与旧基线保持一致；EP4 每个 rank 分到 2 条 | 不是统计收敛保证。可按 8→16→32 增加代表性序列，检查 W8 选择、排名及覆盖是否稳定，再做独立精度评测 |
| `seq_length=512` | 控制激活、评分输出差值和反向临时空间；位于当前 K3.1 短上下文适配范围内 | 不代表长上下文质量已经验证。当前 K3.1 runtime 要求长度不超过配置的 DSA `index_topk` |
| `batch_size=2` | EP4、8 条序列时，每个 rank 的两条一起评分，减少块加载次数 | 总样本数仍为 8，总 token 数不减少；需要更多评分显存，并保持逐序列平均 CE 之和 |
