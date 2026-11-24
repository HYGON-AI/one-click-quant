# 经验条目（追加制）

> **权威来源**：本表是 `references/` 各文档的**边界结论沉淀**，不重复正文流程。
> 与 reference 冲突时，reference 的流程说明优先；本表只回答「这条结论在什么条件下成立」。

## 追加规范

- 条目连续编号 `[E-序号]`，只追加、不改号；结论被证伪时新增一条并标注「取代 E-xx」。
- **来源**三选一：`实测` / `用户反馈` / `代码实证`。
- **验证状态**三选一：`已回归`（在本仓库当前代码上复核过）/ `待验证` / `已上流 docs`。
- 经验 ≠ 常见错误：错误表管「出错怎么修」，本表沉淀「正确认知 / 边界结论」。
- 不含日期：时间戳对检索复用无意义。

| 条目 | 主题 | 适用条件 | 结论要点 | 来源 | 验证状态 |
|---|---|---|---|---|---|
| E-01 | 镜像引用 | 起容器 | 支持 `registry/image:tag` 与本机 image ID；ID 或 `sha256:` 跳过 `docker pull`。ID 只在本机有效，不可移植 | 用户反馈 | 待验证 |
| E-02 | 模型路径 | 任何 `--model` | 不扫盘、不编造。用户未给真实目录时问一次，或留 `$MODEL` 模板占位。**必须相对路径**：`slimquant_ptq` / `model_free_ptq`（`cached_file`；软链 glob 须对 `$SRC` 展开，见 `03` 3.0）。**其它 alg**：默认可绝对路径；一旦报 `does not appear to have a file named`，优先按 3.0 建相对软链再跑，不要先怀疑权重损坏 | 实测 | 已回归 |
| E-03 | `--ignore` 语义 | 需要改忽略层 | 三条路径行为不同：llmcompressor 与 `model_free_ptq` 是**整表替换**（`lm_head`/gate/embed 必须自带）；`slimquant_ptq` 是**并集**。默认清单看当次 log 的 `ignore=[...]` | 代码实证 | 已回归 |
| E-04 | 容器问题 | 容器内失败 | 无通用解：按报错分类定位（镜像、GPU、shm、挂载、网络、缺依赖）。禁无脑复跑固定 `docker run` | 用户反馈 | 待验证 |
| E-05 | analysis 数据集 | 跑 `--analysis` | 内置按**相对路径** `datasets/EleutherAI___wikitext_document_level/...` 取数据，两种模式行为不同：`fisher` **无回退**（直接抛异常）且固定 128 样本 / seqlen 2048、不吃 CLI 的样本数参数；`activation_sensi` 三级回退（内置 → `--dataset` → 联网 `load_dataset("wikitext")`，第三级未包 try）且**静默**，看不出换了数据。要复现须固定工作目录或固定 `--dataset` | 代码实证 | 已回归 |
| E-06 | W4A8 语义 | 执行 `slimquant_ptq` | `int4` = 非 MoE 逐通道 INT8 + MoE routed experts INT4 二阶段，**不是**普通 W4A8。`moe_pattern` 等 int4 参数在模板内用 `getattr` 兜底默认值（`.mlp.experts.`、`k=0.97~1.0`），**未暴露为 CLI**；层名不含 `.mlp.experts.` 会**静默降级成全 INT8**。输入是 3D fused `gate_up_proj` / `down_proj` 时，`split_fused_moe_experts` 在匹配**之前**无条件执行（不区分是否 MoE），故**产物必定是 per-expert 2D**；fused 3D 只出现在源侧或其它工具的产物里 | 代码实证 | 已回归 |
| E-07 | PPL 数据集 | 跑 `--lm-eval` | 需要离线 `wikitext`（`EleutherAI___wikitext_document_level`），校准用的 ultrachat 不能替代。缺缓存直接退出，重试无用 | 代码实证 | 已回归 |
| E-08 | 两套 adapter | 接新模型 | `preprocess/adapters/` 与 `utils/MoE-Quant/src/models/registry.py` 完全独立；后者匹配**顺序敏感**，更具体的 adapter 必须在前 | 代码实证 | 已回归 |
| E-09 | 减层验证 | 建 mini 冒烟 | 「减层」≠「取前 N 层」：必须覆盖该模型每种结构单元（MLA / MoE / shared_experts / MTP / linear_head.norm / linear_attn），否则冒烟假通过 | 代码实证 | 已回归 |
| E-10 | 产物判定 | 判断量化是否成功 | 不以日志、也不以 `QUANTIZATION_DONE.txt` 判定。看三条：`config.json` 的 `quant_method` 与方案一致、权重 dtype 与 scheme 一致、分片齐全。单分片产物没有 `index.json` 属正常；`quantization_config` / `compression_config` 键名本身不是判据 | 代码实证 | 已回归 |
| E-11 | `ignore` 匹配规则 | 写 `--ignore` 或改 `compression_config.ignore` | `slimquant_ptq` 路径（自研 `_match_name`）：带 `re:` → `re.match` **前缀**匹配（整名匹配得自己加 `$`）；不带 `re:` → **整名相等**，不是子串包含；匹配对象是模块名（去掉 `.weight`）。因此 `embed_tokens` 匹配不到 `model.embed_tokens`，默认表才写 `re:.*embed_tokens.*`。llmcompressor / `model_free_ptq` 走上游 `compressed_tensors/utils/match.py`，语义不保证逐字一致，以产物 dtype 为准 | 代码实证 | 已回归 |
| E-12 | 写盘的 ignore 被清洗 | 从产物 `config.json` 抄 ignore | `templates/slimquant/config.py::_clean_ignore_patterns` 写盘时把 `re:.*mlp.gate$` 洗成 `mlp.gate`、`re:.*embed_tokens.*` 洗成 `embed_tokens`（去 `re:`、去前后 `.*`、去 `$`）。抄回去是**裸名字→整名相等**，匹配不到带前缀的模块名，等于 ignore 失效且不报错 | 代码实证 | 已回归 |
| E-13 | MoE-Quant 没有 W4A8 | 选 `utils/MoE-Quant/` 的打包档位 | `pack_quantized_model.py` 的 `--activation-bits` 只有两档：`16` → `pack-quantized`（W4A16 / W8A16）、`8` → `int-quantized`（**W8A8，`pack_weight` 硬性要求 bits==8**）。`--bits 4 --activation-bits 8` 直接 `ValueError: int-quantized format only supports 8-bit weights`。要 W4A8 只能走 `slimquant_ptq` | 代码实证 | 已回归 |
| E-14 | ② 的 CLI 参数被 ① 覆盖 | 打包 | `pack_quantized_model.py` 从 `quantized_model_path/metadata.pt` 读回 `bits` / `group_size` / `quantize_only_experts` 并**覆盖**同名 CLI，且强制 `sym = True`。改档位必须重跑 ①，只改 ② 的命令无效（也不会报错） | 代码实证 | 已回归 |
| E-15 | build_mini 残留压缩配置 | 用 mini 做冒烟/交付 | 四个 `build_mini_*` 默认解量化时只 `pop("quantization_config")`，**不处理 `compression_config`**。源模型是 `compression_config` 风格产物（Kimi / GLM / qwen38 多阶段）时，输出目录的 `config.json` 会残留"声称已量化"的配置，下游据此反量化纯浮点权重 → 交付前必须手工删掉 | 代码实证 | 已回归 |
| E-16 | model_xray 的两个副作用 | 权重取证 | ① `--output_dir` 省略时 JSON 写回**模型目录**，而这些 `.metadata.json` 会让 `slimquant_ptq` 的 `os.walk`+`cached_file` 报 `OSError`；② 它逐个 `get_tensor` 取值（不是只读 header），且**不返回非零退出码**，判据只能是 JSON 内容 | 代码实证 | 已回归 |
