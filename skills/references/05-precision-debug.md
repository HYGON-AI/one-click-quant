# 阶段5 精度排查

> **权威来源**：仓库根 `docs/precision-tuning.md`（调优策略与指标含义）。
> 分析工具的参数与输出解读在 `20-analysis.md`；本文只讲**排查顺序与取舍规则**。

## 从快到慢的排查梯子

校准 + SmoothQuant/GPTQ 很耗时，先用短周期验证：

1. `--alg model_free_ptq`（README 标注为 Data-free：免校准、不实例化 HF 模型）—— 最快的基线
2. `--alg ptq`
3. 默认 scheme 路径 / SmoothQuant+GPTQ 系列
4. 用 `--analysis` 定位最差的层，按推荐做 `--ignore` / 换 AWQ，重新量化并对比（**这是主要手段**）
5. 以上都解释不了时，才去查 `llmcompressor` / `compressed-tensors` 版本（本仓库未 pin 版本）；
   升 upstream main 属**未经验证的回退手段**，换完要重跑 04 的契约校验，不能直接当精度结论

> 不要跳过第 4 步直接换依赖版本：版本问题表现为"所有层同时变差"，
> 而层敏感度问题表现为"少数层特别差"，两者用 `--analysis` 的输出很容易区分。

```bash
python3 main.py --model <真实模型目录> --scheme W8A8 --alg model_free_ptq
python3 main.py --model <真实模型目录> --scheme W8A8 --alg ptq
```

## 测 PPL

```bash
python3 main.py --lm-eval "pretrained=<量化产物真实目录>,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.6"
```

与 BF16 / 上一版 scheme 对比。做 ignore 的 A/B 时必须保持评测参数不变。

## 定位：用分析，不用猜

```bash
python3 main.py --model <真实模型目录> --save-dir <量化产物目录> --analysis mse
python3 main.py --model <真实模型目录> --save-dir <量化产物目录> \
  --analysis activation_sensi --num-calibration-samples 128 --max-seq-length 512
```

五种模式的选择、输出列的含义、**必须站在仓库根执行**的原因，见 `20-analysis.md`。
把最差的层加进 `--ignore`，重新量化后再测 PPL。

**ignore 注意**：CLI 的 `--ignore` 在 **llmcompressor / `model_free_ptq` 路径下会整体替换默认值**
（仍需要的 `lm_head` / gate / embed 必须自己带上）；只有 `slimquant_ptq` 是并集。
三条路径的对照表见 `03-quant-runbook.md` 第 5 节。

## 回退候选速查表（**起点，不是结论**）

按层类型给出的默认取向，用来在分析没跑通、或要在候选里排序时决定先试谁：

| 类别 | 默认处理 | 原因 |
|---|---|---|
| `mlp.down_proj` | 优先回退候选 | FFN 输出聚合层，误差易累积 |
| `self_attn.o_proj` | 结构化回退候选 | attention 输出聚合，与残差直接相加 |
| MoE `gate` / `router` | 默认排除量化 | 路由误判代价高（错一个 expert 全层偏移） |
| MoE `experts` | 低比特主落点 | 参数量占比大，收益最高 |
| `shared_experts` | 通常保持高精度 | 全局复用，误差被所有 token 放大 |
| MLA 低秩投影（`kv_b_proj` / `q_a_proj` / `q_b_proj`） | 条件性排除 | 误差沿 attention 累积，且低秩维窄 |
| `lm_head` / `embed_tokens` | 默认 ignore | 直接影响输出分布 |
| 所有 `*norm` / `*layernorm` | 不量化 | 量化框架内建跳过，不占 ignore 名额 |

使用纪律：

- 先按此表排序，再**逐层 A/B**（一次只动一层），否则无法归因
- 与 `--analysis` 结论冲突时**以分析为准**；此表只用于分析不可用时的占位
- 引用此表的结论必须标注「经验占位，弱于分析结果」
- `slimquant_ptq` 的 `_ARCH_IGNORE` 已有各架构默认表，先看它再决定是否要额外 ignore
  （见 `03-quant-runbook.md` 第 3.1 节）

## 预处理 A/B

如果用了 QuaRot/Flex：

- 对比**有**与**无**预处理（`--from-preprocessed`）。
- 用 `--transforms` 逐个开关 transform。
- 需要真实激活时可加 `--act-stats` + recipe 中**对应 transform 块下**的 `use_calib_acts: true`
  （`flex_awq_ssz:` / `flex_smooth_quant:` 里面；写在顶层不生效，`check_recipe.py` 会直接报错）。

## OOM / 卡住（不是精度问题，但常被混淆）

- 校准阶段：`--pipeline sequential`、`--sequential-targets Linear`、`--offload-hessians`、调小 `--batch-size` / `--max-seq-length`、`--device-map cpu`。
- 保存阶段 OOM：`--patch-for-llmc`。
- >100B 多卡卡住：`nproc_per_node` 用 2 或 4。
