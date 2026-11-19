# one-click-quant Skill

Agent 入口是 [`SKILL.md`](SKILL.md)。本文给人看：能力边界、已适配路径与可复现测试步骤。

## Summary

新增 / 完善 `skills/`：给 Agent 提供 one-click-quant 的使用边界、决策顺序与可判定门禁。
Skill 不新增量化算法，只文档化仓库**已有**能力；参数与原理仍以 README / docs / `--help` 为准。

## 能力边界

### Skill 能做什么

- **任意模型入口**（`references/00-new-model-playbook.md`）：先判是否已适配
  - 已适配 → 给出已适配方案菜单供选择；客户自提方案 → 评估是否可行（不可行则给替代）
  - 未适配 → 取证 → 可行列表 → 是否 preprocess → 主推荐（确认后再执行/适配）
- 引导 Agent 走完整闭环：环境 → 模型准备 → 选路径量化 → 验收 → 精度排查
- 按模型族指向正确入口（通用 scheme / 定制 scheme / preprocess / slimquant / MoE-Quant / examples）
- 用 `check_recipe.py` / `check_artifact.py` 做静态门禁；约束假路径、相对路径、红线

### Skill / 本仓库明确不保证什么

- 不保证「任意 HF 模型开箱即用」；未适配先走 `00` 出方案，再按需加 adapter / scheme / examples
- 「已适配」只表示有专用入口，不表示客户自提的任意组合都合法（需经 `00` 阶段 U 评估）
- `00` 的推荐/评估是**工程可行结论**，不等于精度承诺；精度对比见 `04` §4.2（展示 ΔPPL / ΔNLL，**无阈值门禁**）
- 不做集群调度、账号下载、vLLM 运维压测
- 非 bug 不改 absorb / pipeline / transform / quant_templates 核心逻辑
- Skill 文档与命令以当前仓库代码为准；下表「适配」= **代码里有专用入口**，不等于本次 PR 对全量表做了端到端重跑

## 明确支持的模型与路径（复现以代码为准）

> 判据：有对应 recipe/adapter、`quant_templates.yaml` 条目、`_ARCH_IGNORE`、MoE-Quant registry 或 `examples/<family>/` 脚本。
> 通用路径（任意架构可试、无专用适配）：`FP8_DYNAMIC` / `W8A8` / `FP8_TO_BF16` / `--alg model_free_ptq`（能否跑通取决于 transformers 是否支持该结构）。

### 1) `main.py` 定制 scheme（`templates/quant_templates.yaml`）

| 模型 / 架构 | scheme | 说明 |
|-------------|--------|------|
| DeepSeek-V3 系（`DeepseekV3ForCausalLM`） | `DeepseekV3_FP8_TO_INT8` / `DeepseekV3_FP8_TO_FP8` | block FP8 → 逐通道 |
| Qwen3.5 MoE（`Qwen3_5MoeForConditionalGeneration`） | `Qwen35_MOE_BF16_TO_INT8` | BF16 → 逐通道 INT8 |
| Kimi-K2.5（`KimiK25ForConditionalGeneration`） | `Kimi_W4_TO_INT8` / `Kimi_W4_TO_FP8` | group INT4 → 逐通道 |
| Qwen3-1.7B（`Qwen3ForCausalLM`，2048×28） | `W8A8`（专用 template） | YAML 精确签名 |
| Qwen2.5-0.5B（`Qwen2ForCausalLM`，896×24） | `FP8_DYNAMIC`（专用 ignore） | YAML 精确签名 |
| MiniMax-M2.5/M2.7（`MiniMaxM2ForCausalLM`，3072×62） | `FP8_DYNAMIC`（专用 ignore） | YAML 精确签名 |
| 任意架构 | `FP8_TO_BF16` | `"*"` 通配 |

### 2) Preprocess（`preprocess/adapters` + `recipes`）

| 模型 | recipe / adapter | 备注 |
|------|------------------|------|
| DeepSeek-V3 | `deepseek_v3.yaml` / `DeepSeekV3Adapter` | 含 block FP8 unpack |
| Qwen3.8-Flash-Next | `qwen3_8_flash_next.yaml` / `Qwen38FlashNextAdapter` | 别名含 qwen4_exp 等 |
| Qwen2-MoE（如 Qwen2-57B-A14B） | `qwen2_moe.yaml` / `Qwen2MoeAdapter` | BF16；QuaRot + Flex AWQ/Smooth |

### 3) SlimQuant W4A8（`--alg slimquant_ptq`，`_ARCH_IGNORE` 有专用表）

| 架构签名（示例） | 典型模型 |
|------------------|----------|
| `Qwen3MoeForCausalLM` (2048, 48) | Qwen3-30B-A3B 类 |
| `Qwen3_5MoeForConditionalGeneration` (2048, 40) | Qwen3.5-35B-A3B 类 |
| `DeepseekV32ForCausalLM` (7168, 61) | DeepSeek-V3.2 类 |
| `GlmMoeDsaForCausalLM` (6144, 78) | GLM-5 类 |
| 其它 MoE | 可跑，走 `_DEFAULT_IGNORE`；expert 命名须匹配 `.mlp.experts.` 才会 INT4 |

### 4) MoE-Quant（`utils/MoE-Quant` registry）

| 模型 | 位宽约束（摘要） |
|------|------------------|
| DeepSeek-V3 | 仅 bits=4 → W4A16 |
| Qwen3.5 / 3.8 MoE | 仅 bits=4 → W4A16 |
| Qwen2-MoE | bits=4→W4A16 或 bits=8→W8A16/W8A8 |
| Kimi-K3 | 仅 bits=4 → W4A16 |
| GLM-5.3 / GLM-5.3-Flash | bits=4→W4A16 或 bits=8→W8A16/W8A8 |
| （无 W4A8） | `--bits 4 --activation-bits 8` 会在 pack 阶段硬失败；要 W4A8 走 `slimquant_ptq` |

### 5) `examples/` 模型族脚本（离线转换 / 混合精度，非统一 CLI）

| 目录 | 代表能力 |
|------|----------|
| `examples/deepseek/` | V3 mini；V4.1 Flash FP8/INT8/INT4/mixed；mixed W4A8 |
| `examples/qwen/` | Qwen3.8 mini；Flash-Next INT4/W4A16/NVFP4；多阶段见 `qwen3_8-flash-next/` |
| `examples/glm/` | GLM-5.3 / Flash 的 FP8、INT4、INT8、W8A8 |
| `examples/kimi/` | Kimi-K3 mini；MXFP4→INT4；FP8_DYNAMIC |
| `examples/mimo/` / `minimax/` / `hy4/` | 各型号专用转换脚本（以目录 README/`--help` 为准） |

### 6) build_mini（减层冒烟）

| 脚本 | 目标 |
|------|------|
| `examples/qwen/build_mini_qwen3_8.py` | Qwen3.8 MoE |
| `examples/deepseek/build_mini_deepseek_v3.py` | DeepSeek-V3 |
| `examples/glm/build_mini_glm5_3.py` | GLM-5.3 |
| `examples/kimi/build_mini_kimi_k3.py` | Kimi-K3 |

## 模型入口（先判适配 / 只出方案）

向 agent 描述示例：

> 读取 `<模型绝对路径>`，按 `00-new-model-playbook`：先判断是否已适配；若已适配，列出已适配方案供我选择；若我另行提出方案，请评估是否可行。若未适配，按新模型流程给出可行方案、是否 preprocess 与主推荐。先不要执行量化。

## 测试demo

如果不了解模型，也不知道怎么量化，可以给agent输入模型路径，让它制定合适的量化方案。
1. /public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-VL-72B-Instruct，这个模型在one-click-quant中是否已经适配，若已适配，请给出量化方案。若未适配，请推荐合适的量化方案。

如果明确知道怎么量化，可以借鉴下面的例子：
1. 明确使用镜像或有可用的容器，或已在容器环境中，就无需重新创建容器\
用镜像10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-20261102-0242创建容器，用 one-click-quant对/public/opendas/DL_DATA/llm-models/Qwen3.8-Flash-Next做已适配的 preprocess（recipe: qwen3_8_flash_next）。量化完的权重保存到当前工程目录，按 preprocess 标准验收。

2. 对preprocess后的权重做slimquant_ptq W4A8\
用 one-click-quant对 .../outputs/Qwen3.8-Flash-Next-preprocessed 做 slimquant_ptq W4A8量化（加 --from-preprocessed 与 Flash-Next 必要 ignore：re:.*conv1d.*、re:.*norm.*、re:.*visual.*），产物写到工程 outputs/，按 slimquant_w4a8 契约验收。

3. 量化为compressed-tensors 格式的 W4A8\
用one-click-quant将/public4/opendas/DL_DATA/llm-models/deepseek-ai/DeepSeek-V4.1-Flash量化为 compressed-tensors 格式的 W4A8，量化完的权重放在当前outputs目录下，用compressed-tensors格式验收。

4. 对模型做敏感度分析\
用one-click-quant的--analysis outlier功能对/public/opendas/DL_DATA/llm-models/Qwen3.8-Flash-Next做敏感度分析，并给出敏感度分析报告.

5. 为preprocess适配新的模型\
用one-click-quant中的preprocess对/public/opendas/DL_DATA/llm-models/Qwen/Qwen2-57B-A14B-Instruct做预处理，目前还未适配该模型的预处理功能，可以先适配，再运行，预处理后的权重保存到/module目录下.

6. 对未适配的新模型做w8a8 GPTQ量化\
用one-click-quant的Moe-Quant对/public/opendas/DL_DATA/llm-models/Qwen/Qwen2-57B-A14B-Instruct做w8a8 GPTQ量化，量化完的模型放在目录/module下。

7. 量化完做精度验证\
用one-click-quant的Moe-Quant对/public/opendas/DL_DATA/llm-models/Qwen/Qwen2-57B-A14B-Instruct做w8a8 GPTQ量化，量化完的模型放在目录/module下。并对量化后的模型做精度验证，给出精度报告。

## 目录结构

```text
skills/                             # 本 skill 根（上库后通常为仓库根下 skills/）
├── SKILL.md                        # 路由 + 门禁四件套（硬门禁 / 红线 / 检查清单 / 通过标准）
├── README.md                       # 本说明
├── references/
│   ├── 00-new-model-playbook.md    # 模型入口：判适配 → 已适配菜单/自提评估 → 未适配推荐
│   ├── 01-env-docker.md            # 阶段1 环境 / 镜像 / 依赖 / 数据集
│   ├── 02-model-prep.md            # 阶段2 权重取证 / mini / 解包
│   ├── 03-quant-runbook.md         # 阶段3 选 scheme/alg、跑量化
│   ├── 04-verify-eval.md           # 阶段4 产物契约 / dtype / MSE / PPL(ΔPPL/ΔNLL) / torch_infer
│   ├── 05-precision-debug.md       # 阶段5 精度排查阶梯
│   ├── 10-ext-preprocess.md        # 扩展：preprocess adapter + recipe
│   ├── 11-ext-scheme.md            # 扩展：scheme / quant_templates
│   ├── 12-ext-slimquant.md         # 扩展：slimquant 内部
│   ├── 13-ext-moe-quant.md         # 扩展：MoE-Quant adapter
│   ├── 20-analysis.md              # 知识：--analysis 五种模式
│   ├── 21-hybrid-quant.md          # 知识：混合精度多阶段
│   └── experiences.md              # 横切经验条目（E-xx，不占编号）
└── scripts/
    ├── check_recipe.py             # 静态校验 preprocess recipe
    ├── check_artifact.py           # 静态校验量化产物（不适用于 preprocess 浮点目录）
    └── report_ppl_delta.py         # 由两侧 word_perplexity 打印 ΔPPL / ΔNLL（无阈值）
```

### 编号约定

| 前缀 | 含义 | 范围 |
|------|------|------|
| `0x` | 生命周期阶段 | `00` 新模型方案；`01` → `05` 从零到可用 |
| `1x` | 扩展点 | `10` → `13`（改能力） |
| `2x` | 知识层 | `20` → `21`（查细节） |

编号按分组跳号，不是漏文件。`experiences.md` 不占编号。

## 内容分工（docs 权威，skill 写增量）

1. **仓库 docs / README / `--help`**：参数全集、算法原理、安装细节——以它们为准。
2. **本 skill**：决策顺序、门禁、已知坑、可判定命令；与 docs 冲突时先信 docs/`--help`，再把差异登记到 `experiences.md`。
3. **不负责**：集群调度、HF 账号、vLLM 运维压测、非 bug 改框架源码——见 `SKILL.md`「不负责 / 去向」。

## 怎么用（给 Agent / 给维护者）

**Agent**：按 `SKILL.md` 的 `description` 自动命中；先走执行链 `01→05`，扩展进 `1x`，细节查 `2x`。

**维护者改文档时**：

1. 改流程/门禁 → 改对应 `references/0x|1x|2x`，必要时同步 `SKILL.md` 路由与硬门禁。
2. 发现新边界结论 → 只追加 `experiences.md`（`E-xx` 不改号；证伪则新开条目并标注「取代 E-xx」）。
3. 示例命令保持通用：不写死个人路径、卡号、镜像 tag、包管理器；占位符用 `<...>` / 环境变量。测试步骤一节可保留已验证的真实路径作复现范例。
4. `check_artifact` 仅用于**量化产物**；preprocess 浮点验收见 `10-ext-preprocess.md`。

## 门禁四件套（定义在 SKILL.md）

| 名称 | 作用 |
|------|------|
| 硬门禁 | 不满足即停（路径真实、相对路径例外、recipe/artifact 脚本、wikitext） |
| 红线 | 禁止行为（抄 recipe、改核心逻辑、伪造输出、混用两套 adapter 等） |
| 检查清单 | 开跑前自检 |
| 通过 / 失败标准 | 量化与 preprocess 分开判定 |

## 相关仓库文档

- 仓库根 [`README.md`](../README.md)
- [`docs/`](../docs/)（preprocess、精度调优等）
