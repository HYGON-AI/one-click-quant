# 敏感层分析（`--analysis`）

> **权威来源**：`utils/analyze_quant.py`（`add_analysis_args` 与 `run_analysis`）、仓库根 `docs/precision-tuning.md`。
> 本文只讲**工具怎么用、输出怎么读**。取舍策略与回退候选见 `05-precision-debug.md`。

## 1. 五种模式

| `--analysis` | 需要 | 用途 | 主要输出列 |
|---|---|---|---|
| `outlier` | `--model` | 离群通道多的层 → AWQ 或 ignore | `Outlier%` / `Max/Med` / `MaxAbs` / `OutCh` |
| `mse` | `--model` + `--save-dir`（量化产物） | 量化后权重的**静态**误差 | `Mean MSE` / `Corr` / `Outlier%` |
| `fisher` | `--model` | 基于梯度的敏感度 | `Rank` / `Mean` / `Max` / `Params` |
| `activation_sensi` | 两者 + 校准样本 | **动态** MSE（真跑激活） | `MSE` |
| `outlier_mse` | 两者 | 离群度与 MSE 的联合排序 | 前两组列 |

缺 `--model` 或（`mse` / `outlier_mse` / `activation_sensi` 时）缺 `--save-dir` 会直接 `sys.exit(1)`，不降级。

```bash
python3 main.py --model <真实模型目录> --analysis outlier --analysis-top 20
python3 main.py --model <真实模型目录> --save-dir <量化产物目录> --analysis mse
python3 main.py --model <真实模型目录> --analysis fisher --analysis-top 20

# 动态模式：参数按建议给，否则慢且显存高
python3 main.py --model <真实模型目录> --save-dir <量化产物目录> \
  --analysis activation_sensi --num-calibration-samples 128 --max-seq-length 512
```

## 2. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--analysis-top N` | `20` | 报告里列出最差的 N 层 |
| `--analysis-output <文件>` | stdout | 报告写文件（长报告建议用，便于 diff 两次分析） |
| `--num-calibration-samples` | `512` | 仅 `activation_sensi` 用；建议压到 `128` |
| `--max-seq-length` | `2048` | 仅 `activation_sensi` 用；建议压到 `512` |
| `--dataset` | `./datasets/ultrachat_200k` | 仅 `activation_sensi` 在取不到内置 wikitext 时回退使用；`fisher` 不读它（见下） |

**`fisher` 不吃这三个参数**：`_run_fisher_analysis` 用的是写死的 `num_samples=128` / `batch_size=1` /
`seqlen=2048`，命令行传 `--num-calibration-samples` / `--max-seq-length` / `--batch-size` **对它无效**。
想改只能改代码，或减少层（用小模型做 fisher）。

**显存/内存取向不同**（排查"为什么跑不动"时先分清）：

- `outlier` / `mse` / `outlier_mse`：模型以 `device_map="cpu"` 加载 → **不占显存，吃内存**（`torch_dtype="auto"`）
- `fisher`：`run_analysis` 先按 CPU 加载一份基座，`_run_fisher_analysis` 再以 `device_map="auto"` +
  `torch_dtype=float32` **加载第二份** → **内存与显存同时翻倍**，是最重的一种
- `activation_sensi`：按 `--device-map` 加载 → 可用 `cpu` / `auto_offload` 省显存

## 3. 数据来源与硬编码路径坑（**先看这条**）

`analyze_quant.py` 有两处**写死的相对路径**取 wikitext 数据集，且两处行为**不同**：

| 位置 | 用于 | 取不到时 |
|---|---|---|
| `_run_fisher_analysis` 的 `_WIKI_DIR`（读 `-test.arrow`） | `--analysis fisher` | **无回退**，直接抛异常（`Dataset.from_file` 未包 try） |
| `_load_calibration_samples` 的 `_WIKI_DIR`（读 `-validation.arrow`） | `activation_sensi` | `except Exception` 后**静默回退**，**不报错** |

`_load_calibration_samples` 的回退是**三级**，不是二级：

1. 内置相对路径的 wikitext arrow
2. `--dataset`（默认 `./datasets/ultrachat_200k`），依次试 `validation` / `test` / `train_sft` 三个 split
3. `load_dataset("wikitext", "wikitext-2-raw-v1")` —— **这一步要联网**，且**未包 try**，
   离线环境下会直接抛网络/连接错误（报错信息里不出现 `--dataset`，很容易误判成环境坏了）

后果三条：

1. **必须在仓库根目录执行**。换到别的工作目录，相对路径解析到别处 → `fisher` 直接失败
2. `activation_sensi` 失败得**无声**：同一份模型在两个不同工作目录下跑，会分别用内置 wikitext 与
   `--dataset` 两批完全不同的样本，**两份排序不可比且看不出区别**
3. 要复现/对比两次分析：固定工作目录，或两次都显式给同一个 `--dataset`；`fisher` 只能靠固定工作目录

推论：

- 报告里必须注明「用的是内置 wikitext 还是 `--dataset`」，否则排序结论没有可比性
- 看到 `load_dataset` 相关的连接报错，说明前两级都没取到，先补数据集而不是查网络
- 这两级都不达标时，`_load_calibration_samples` 还会在样本不足时 `warning`、
  样本为 0 时抛 `RuntimeError: No valid calibration samples found.` —— 看到它说明前三级**全部**失败

（登记为 `experiences.md` E-05）

## 4. 怎么读输出

- `Max/Med`（max 绝对值 / 中位数）是离群度的核心指标：**>10 属重度离群**，5~10 属中度
- `Mean MSE` / `Corr`：`Corr` 是离群度与 MSE 的相关性，相关性高说明"离群大的层确实误差大"，
  此时按 outlier 排序即可；相关性低说明两者分离，要两个都看
- `outlier` 模式的汇总里会给三条建议：`Recommend AWQ`（离群显著）、
  `Recommend SmoothQuant`（敏感层占比高）、`Recommend adding N layers to ignore list`
- **推荐清单上限是动态的**：`max(5, min(100, Linear 层数 × 10%))`，大模型上会给出上百个候选，
  不可能全塞进 `--ignore` —— 必须自己按 A/B 结果裁剪

> 分析输出只是**层排序候选**，不是精度结论。任何取舍都必须用 `--ignore` 做 A/B 并重测 PPL
> （`--ignore` 的替换/并集语义见 `03-quant-runbook.md` 第 5 节）。

## 5. 工具不可用时的降级规则

仅在**工具本身跑不起来**时（依赖缺失、脚本抛异常）才可降级；路径/schema/模型加载类错误**必须立即失败**，
不得用经验掩盖。降级时：

1. 候选层取**前 2~4 层 + 后 2~4 层**的 `mlp.down_proj`、`self_attn.o_proj`、
   MLA 的低秩投影（`kv_b_proj` / `q_a_proj` / `q_b_proj`）—— 误差在这几类层最易累积
2. 每加一层都要单独 A/B（一次只动一层），否则无法归因
3. 结论里**必须标注「经验占位，弱于分析结果」**，不能与分析结论并列呈现
4. 路径错 / 配置 schema 不符 / 模型加载失败 → 直接报错停止，规则同 `SKILL.md` 的硬门禁
