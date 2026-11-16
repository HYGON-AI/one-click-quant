---
name: one-click-quant
description: >-
  one-click-quant 仓库的一键量化使用指南，覆盖模型准备、量化执行、preprocess 预处理
  （QuaRot/Flex）、scheme 与模板扩展、评测验收、精度调优排查、混合精度多阶段。
  当涉及 main.py、--scheme、--alg、slimquant_ptq、model_free_ptq、preprocess recipe、
  adapter、quant_templates.yaml、MoE-Quant、pack_quantized、build_mini、model_xray、
  torch_infer、check_artifact、check_recipe、mixed_precision、hybrid、PPL/lm_eval、
  HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES 或精度调优时使用。
---

# one-click-quant

本仓库一键量化的操作手册。先判断属于「执行 / 扩展 / 知识」哪一类，再进对应文档。
参数不清楚时读 `python3 main.py --help`（但 `--scheme` 的 help 不是全集，见 `03-quant-runbook.md` 第 1 节）。

> 编号约定：`0x` = 生命周期阶段（01→05），`1x` = 扩展点（10→13），`2x` = 知识层（20→21）；
> 编号按分组跳，不是漏号。`experiences.md` 是不占编号的横切表。

## 执行（从零到可用）

1. 环境 / 镜像 / 依赖 / 数据集 → [01-env-docker.md](references/01-env-docker.md)
2. 新模型、判权重格式、建 mini、反量化 → [02-model-prep.md](references/02-model-prep.md)
3. 跑量化、选 scheme/alg → [03-quant-runbook.md](references/03-quant-runbook.md)
4. 跑完验收（产物契约 / dtype / MSE / PPL / 推理）→ [04-verify-eval.md](references/04-verify-eval.md)
5. 精度不达标 → [05-precision-debug.md](references/05-precision-debug.md)（分析工具见 `20-analysis.md`）

## 扩展（改能力）

- 新模型接 preprocess（adapter + recipe）→ [10-ext-preprocess.md](references/10-ext-preprocess.md)
- 新增或挂上 `--scheme` / 模板 → [11-ext-scheme.md](references/11-ext-scheme.md)
- 改 slimquant 内部（数值、packing、新 `quant_type`）→ [12-ext-slimquant.md](references/12-ext-slimquant.md)
- 接新模型到 `utils/MoE-Quant/`（adapter + 位宽约束）→ [13-ext-moe-quant.md](references/13-ext-moe-quant.md)

## 知识（查细节）

- 五种 `--analysis` 模式、输出怎么读、硬编码路径坑 → [20-analysis.md](references/20-analysis.md)
- 多阶段混合精度（通用约定 + Flash-Next 范本）→ [21-hybrid-quant.md](references/21-hybrid-quant.md)
- 历史结论、边界条件、踩过的坑 → [experiences.md](references/experiences.md)

## 不负责 / 去向

| 本 skill 负责 | 不负责（去向） |
|---|---|
| 选 scheme/alg，给出可执行命令 | 修改框架源码（非 bug 不改）→ 提 issue 或改上游 |
| preprocess adapter + recipe 接入 | 集群调度、HF 账号、模型下载 → 用户环境 |
| 精度排查路径与 `--ignore` 候选 | 评测平台搭建、vLLM 部署运维 → 部署侧 |
| 产物契约 / dtype 校验、推理 fallback | 推理性能调优、服务化压测 → 性能侧 |

## 硬门禁（不满足即停）

- [ ] `--model` 指向的源目录真实存在且 `config.json` 可解析；用户没给就先问，**不得**写 `/models/xxx`
      - **必须相对路径**：`--alg slimquant_ptq` / `--alg model_free_ptq`（见 `03-quant-runbook.md` 3.0）
      - **建议先试相对路径**：其它走 `cached_file` 的路径若报 `does not appear to have a file named`，
        不要先怀疑权重损坏，先按 3.0 建软链再跑
- [ ] 改 preprocess 前：`python3 ./skills/scripts/check_recipe.py <recipe>` 退出码为 0
      （`<recipe>` = `preprocess/recipes/` 下本模型对应文件，不要写死某一份）
- [ ] 声称「**量化**完成」前（`main.py` PTQ / slimquant / MoE-Quant 打包 / examples 量化脚本产物）：
      `python3 ./skills/scripts/check_artifact.py <产物目录>` 退出码为 0
      （逐条判据见 `04-verify-eval.md` 第 1、2 节）。**不适用于** `--alg preprocess` 浮点产物
      （那条用 `10-ext-preprocess.md` 验收：两个量化键都不存在）
- [ ] `--lm-eval` 前：`HF_DATASETS_CACHE`（默认 `./datasets`）下存在 `EleutherAI___wikitext_document_level`

## 红线（贯穿全流程）

- 不得为非本模型复用其它模型的 preprocess recipe
- 不得为绕过报错修改 `absorb` / `pipeline` / `transform` / `quant_templates` 核心逻辑
- 不得伪造命令输出，或以「应该能跑」代替 exit code / 产物判定
- 不得在命令失败后换未文档化命令续跑以规避
- 不得混用两套 adapter 注册表（`preprocess/adapters/` 与 `utils/MoE-Quant/src/models/registry.py`）
- 运行前提三条（Python ≥3.10 / 目录可写 / 站在仓库根）任一不满足即停，见 `01-env-docker.md`

## 检查清单（执行前自检）

- [ ] 模型路径真实存在、`config.json` 可解析，层数与 expert 数已确认
- [ ] scheme 与 `--alg` 组合在该模型上被支持（不属于「静默降级」情形）
- [ ] 产物目录磁盘空间充足（产物 ≈ 源权重规模 + 少量 scale）

## 通过 / 失败标准

- **通过（量化）** = exit code 0 + 产物契约齐全 + dtype 与 scheme 预期一致 + 给出基线与产物**两个**指标
- **通过（preprocess）** = exit code 0 + 浮点权重 + 顶层/`text_config` 均无量化配置键（见 `10`）
- **失败** = 任一项不满足：报「命令 + 退出码 + 错误摘要」后停止，不进入下一阶段

## 快速校验

```bash
# <recipe> 换成 preprocess/recipes/ 下实际文件，例如 deepseek_v3.yaml
python3 ./skills/scripts/check_recipe.py preprocess/recipes/<recipe>
# 仅量化产物；preprocess 浮点目录不要跑这条
python3 ./skills/scripts/check_artifact.py <产物目录> --expect-quant-method <期望值>
python3 tools/model_xray.py --input_dir <模型目录> --output_dir <输出目录>
```
`model_xray` 没有退出码语义（出错也是 0），要读它写出的 JSON；`--output_dir` 必须显式给。
`<期望值>` 按方案填（`slimquant_w4a8` / `compressed-tensors`），对照 `04-verify-eval.md` 1.1 的表。
