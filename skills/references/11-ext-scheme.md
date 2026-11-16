# 扩展：新增量化 scheme / 模板

> **权威来源**：`templates/quant_templates.yaml` 的注释 + 加载它的 `main.py` 逻辑（`--scheme` 的 help 不是全集）。
> YAML 头部注释与代码冲突时**以代码为准**。本文只写选哪一档、怎么挂、怎么验。

三档接入方式，优先选最轻的一档：

1. **只加 YAML ignore** —— 模型已能被 llmcompressor 正常量化，只需在 `templates/quant_templates.yaml` 补 ignore。
2. **自定义模板脚本** —— 非通用路径；代码放 `templates/`，在 YAML 里用 `template:` 挂上。
3. **examples/** —— 独立脚本 + README；适合先试验，跑通后再提升到 templates。

## `quant_templates.yaml` 匹配规则

文件：`templates/quant_templates.yaml`，加载逻辑在 `main.py` 的 `_load_quant_templates` / `_lookup_template`。

- 顶层 key = HF `architectures` 名（如 `DeepseekV3ForCausalLM`），或 `"*"` 表示任意架构。
- 每个架构下是一个列表，条目按 `(hidden_size, num_hidden_layers)` 匹配。

**通配只有一种**：`hidden_size: -1` 且 `num_hidden_layers: -1` 这一对整体作为通配。
**混合写法（如 `hidden_size: -1, num_hidden_layers: 28`）永远不会命中** —— 代码只查
`(签名)` 和 `(-1, -1)` 两个 key，不做逐字段通配。YAML 里现有条目也都是「精确签名」或「双 -1」。

**列表不是"从上到下逐个尝试"**：加载时 `result[arch][(hs, nhl)] = schemes` 是**字典赋值**，
同一个架构下写两条签名相同的条目，**后面的会静默覆盖前面的**。

真正生效的查找顺序是**两级**，不是一级逐条：

1. 先按 `architectures` 名取该架构的字典，在**字典里二选一**：有精确签名条目就用它，否则用 `(-1,-1)` 条目；
2. 第 1 步取到的条目里若没有请求的 scheme，**不会**回头再试同架构的 `(-1,-1)`；
   于是继续查 `"*"` 架构的 `(-1,-1)` 条目；再没有就落到通用 `llmc_oneshot` 路径
   —— 量化仍会跑，但 YAML 里配的 `ignore` 等覆盖**不会生效**。

由此两个坑：

- **YAML 文件头自己的注释是错的**：它写 "list one or more entries matched first to last"，与代码不符。
  以代码为准，不要照注释推理。
- **精确签名会屏蔽同架构的 `(-1,-1)`**：给一个已有精确签名的架构加 scheme，要加进它那条精确条目里；
  另起一个 `(-1,-1)` 条目，只要精确条目命中，新条目就永远不会被读到。

### 条目字段与 `ignore` 优先级

- `template`：`templates/` 下的 Python 脚本路径（可选）
- `ignore`：逗号分隔的层名 / 正则（可选）
- `quant_type`：给共用模板的类型提示（可选）
- `ignore` 优先级：**CLI `--ignore` > YAML `ignore` > llmc 内置默认**（仅对走 `llmc_oneshot` 的通用路径成立，见下）。

### 一个容易踩的坑：YAML 覆盖对不走通用路径的 alg 无效

`main.py` 应用 YAML 额外字段时有白名单排除：

```python
if args.alg not in ("model_free_ptq", "slimquant_ptq", "preprocess"):
    for key, value in extra_kwargs.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
```

也就是说：**`--alg model_free_ptq` / `--alg slimquant_ptq` / `--alg preprocess` 时，
YAML 里的额外字段（`ignore`、`quant_type` 等）完全不会生效**。给这几个 alg 配 ignore 只能走 `--ignore` CLI 参数
（`slimquant_ptq` 内部还有自己的 `_ARCH_IGNORE` 表，见 `03-quant-runbook.md` 第 3.1 节）。

示例写法：

```yaml
MyArchForCausalLM:
  - hidden_size: -1
    num_hidden_layers: -1
    schemes:
      MyScheme_NAME:
        template: "templates/my_scheme.py"
        quant_type: "int8"
      FP8_DYNAMIC:
        ignore: "lm_head,re:.*mlp.gate$"
```

新增公开 scheme 名后，如果需要用户可见，同时更新 `main.py` 中 `--scheme` 的 help 文本。

## 模板脚本约定

- 放在 `templates/` 下；尽量复用已有脚本（`fp8_to_bf16.py`、`w8a8_block_cast_channel.py`、`w4_dequant_channel.py`、`slimquant_ptq.py` 等）。
- 新脚本的参数风格与同类模板保持一致（argparse / 由 oneshot 路径调用）。
- 本仓库自有代码的文件头：

```text
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
```

## 只是换算法、不换配方

若只是算法不同，直接复用已有 scheme + `--alg`，不必新建 scheme：

```bash
python3 main.py --model <真实模型目录> --scheme W8A8 --alg model_free_ptq
python3 main.py --model <真实模型目录> --alg slimquant_ptq --scheme W4A8
```

## 提 PR 前检查清单

```text
- [ ] 可跑的完整命令已记录（README 或 examples/<model>/README.md）
- [ ] quant_templates.yaml 有条目，或 templates/*.py 已挂上
- [ ] 如果是新的公开 scheme 名，--scheme help 已更新
- [ ] 有 mini / build_mini 模型时跑过冒烟
```

参见 README 的「提交PR」章节，以及 DeepSeek / Kimi / Qwen3.5 MoE 的既有条目。
