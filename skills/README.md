# one-click-quant Skill

本目录是 **one-click-quant** 仓库的 Agent Skill：指导 Agent 完成模型准备、量化执行、preprocess 接入、评测验收与精度排查，不替代仓库 `docs/` / `README.md` 的权威说明。

Agent 入口是 [`SKILL.md`](SKILL.md)；人类维护与上库说明看本文。

## 目录结构

```text
one-click-quant/                    # 本 skill 根（上库后通常为仓库根下 skills/）
├── SKILL.md                        # 路由 + 门禁四件套（硬门禁 / 红线 / 检查清单 / 通过标准）
├── README.md                       # 本说明
├── references/
│   ├── 01-env-docker.md            # 阶段1 环境 / 镜像 / 依赖 / 数据集
│   ├── 02-model-prep.md            # 阶段2 权重取证 / mini / 解包
│   ├── 03-quant-runbook.md         # 阶段3 选 scheme/alg、跑量化
│   ├── 04-verify-eval.md           # 阶段4 产物契约 / dtype / MSE / PPL / torch_infer
│   ├── 05-precision-debug.md       # 阶段5 精度排查阶梯
│   ├── 10-ext-preprocess.md        # 扩展：preprocess adapter + recipe
│   ├── 11-ext-scheme.md            # 扩展：scheme / quant_templates
│   ├── 12-ext-slimquant.md         # 扩展：slimquant 内部
│   ├── 13-ext-moe-quant.md         # 扩展：MoE-Quant adapter
│   ├── 20-analysis.md              # 知识：--analysis 五种模式
│   ├── 21-hybrid-quant.md          # 知识：多阶段混合精度
│   └── experiences.md              # 横切经验条目（E-xx，不占编号）
└── scripts/
    ├── check_recipe.py             # 静态校验 preprocess recipe
    └── check_artifact.py           # 静态校验量化产物（不适用于 preprocess 浮点目录）
```

### 编号约定

| 前缀 | 含义 | 范围 |
|------|------|------|
| `0x` | 生命周期阶段 | `01` → `05`（从零到可用） |
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
3. 示例命令保持通用：不写死个人路径、卡号、镜像 tag、包管理器；占位符用 `<...>` / 环境变量。
4. `check_artifact` 仅用于**量化产物**；preprocess 浮点验收见 `10-ext-preprocess.md`。

## 门禁四件套（定义在 SKILL.md）

| 名称 | 作用 |
|------|------|
| 硬门禁 | 不满足即停（路径真实、相对路径例外、recipe/artifact 脚本、wikitext） |
| 红线 | 禁止行为（抄 recipe、改核心逻辑、伪造输出、混用两套 adapter 等） |
| 检查清单 | 开跑前自检 |
| 通过 / 失败标准 | 量化与 preprocess 分开判定 |

## 相关仓库文档

- 仓库根 [`README.md`](../../../README.md)
- [`docs/`](../../../docs/)（preprocess、精度调优等）
