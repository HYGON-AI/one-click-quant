# 第三方组件与来源清单（THIRD_PARTY_NOTICES）

本文件登记 one-click-quant 仓库中随源码分发的第三方组件、第三方源码与数据资产。
「本地路径」为仓库内相对路径；「固定版本」用于保证可复现与可追溯。

本仓库整体以 **Apache License 2.0** 发布，完整许可证文本见 [LICENSE.txt](LICENSE.txt)。

---

## 1. 第三方源码（随仓库分发）

| 编号 | 本地路径 | 来源项目 | 上游仓库 / 页面 | 固定版本 | Copyright | 许可证 | HYGON 修改 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TP-1 | `examples/kimi/modeling/modeling_kimi_k3.py` | Kimi-K3 modeling（上游文件 `modeling_kimi_k3.py`；文件头声明部分派生自 llava `llava/modeling_llava.py`） | https://huggingface.co/moonshotai/Kimi-K3 | revision `f831ab66814297da540d832a5235f8e904f29d06`（2026-09-02）；第二来源 ModelScope `moonshotai/Kimi-K3` @ `master`（同哈希，已实测比对一致）；上游原文件 SHA256 `b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2`；本地文件 SHA256 `d2d635063b7a0dd1ec273fac351ca63fee6e8a9db926f8d6351acad9df4e049b` | Copyright 2025-2026 The Moonshot AI Team and HuggingFace Inc. team；上游 `LICENSE` 归属 Copyright (c) 2026 Moonshot AI | 原文件头声明：llava 派生部分为 Apache-2.0；其余部分为 Kimi K3 License（**许可证全文**：[`LICENSES/Kimi-K3-License.txt`](LICENSES/Kimi-K3-License.txt)） | 与官方原版逐行 diff：**本地独有 17 行 / 官方独有 1 行** —— ① 新增 `import transformers`、`from packaging import version`；② `tie_weights()` 改为按 transformers 版本分发 `missing_keys` / `recompute_mapping`；③ 文件头追加变更声明（Modified by Hygon …, 2026，8 行）。原始版权与许可证声明**逐字节保留、未被改写** |
| TP-2 | `examples/kimi/modeling/modeling_kimi_linear.py` | Kimi-Linear modeling（上游文件 `modeling_kimi_linear.py`；文件头声明 MoE/MHA 部分派生自 DeepSeek-V3 `modeling_deepseek.py`） | https://huggingface.co/moonshotai/Kimi-K3 | revision `f831ab66814297da540d832a5235f8e904f29d06`（2026-09-02）；第二来源 ModelScope `moonshotai/Kimi-K3` @ `master`（同哈希，已实测比对一致）；上游原文件 SHA256 `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`；本地文件 SHA256 `2e01cf00c012ba749d5721073c618888be59e7f7522fc114157bc8ce405f6b8d` | Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team；上游 `LICENSE` 归属 Copyright (c) 2026 Moonshot AI | 原文件头声明：DeepSeek-V3 派生部分为 Apache-2.0；其余部分为 Kimi K3 License（**许可证全文**：[`LICENSES/Kimi-K3-License.txt`](LICENSES/Kimi-K3-License.txt)） | 与官方原版逐行 diff：**本地独有 43 行 / 官方独有 12 行** —— ① `OutputRecorder` 导入兼容 transformers ≥5.2（ImportError 时回退 `utils.output_capturing`）；② `normal_()` 前增加 dtype 保护（仅 float32/float16/bfloat16，跳过 float8 等量化权重）；③ 默认 attention 由强制 `flash_attention_2` 改为 `eager`（避免内网/无网环境 hub kernel 动态下载）；④ `create_causal_mask()` 沿用 `inputs_embeds` 签名；⑤ 文件头追加变更声明（14 行）。原始版权与许可证声明**逐字节保留、未被改写** |
| TP-3 | `templates/slimquant/_vendor.py` | compressed-tensors（函数级参考实现，自包含改写，**非逐行复制**） | https://github.com/vllm-project/compressed-tensors | 参考 tag `0.16.0`（commit `9c3dd050d29b8b2489527889c78a77b2391d7909`，比对后最接近版本）；本地文件 SHA256 `79c33af2619c3e4275abe4cb1279755e8dc9f1571c41457004c0edb0048dd7c2`；对应上游源文件见第 1.1 节 | Copyright the vLLM project | Apache-2.0 | 自包含重组，仅保留量化流程实际使用的函数：18 个顶层定义中 9 个可与上游对应（最高 `get_weight_map` 89.3%、`update_safetensors_index` 75.4%、`exec_jobs` 72.2%），另 9 个为 HYGON 自研（上游无同名实现）；已在文件头标注来源与修改 |

> TP-1 / TP-2 属于第三方原文件，按规则**保留原始文件头，不追加本仓库版权头**，仅在本清单登记来源。
> TP-3 在保留第三方来源标注的前提下，追加本仓库版权与 SPDX 标识。

### 1.1 来源锚点与溯源证据（2026-09-20 复核）

**TP-1 / TP-2 — Kimi-K3 modeling**

- 上游锚点：`https://huggingface.co/moonshotai/Kimi-K3`，revision `f831ab66814297da540d832a5235f8e904f29d06`（lastModified 2026-09-02），获取方式 `GET https://huggingface.co/api/models/moonshotai/Kimi-K3` 的 `sha` 字段。
- 原文件路径：`modeling_kimi_k3.py`、`modeling_kimi_linear.py`（与本地文件同名）。
- 上游文件 SHA256：`b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2`、`9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`。
- 第二来源互证：ModelScope `https://www.modelscope.cn/models/moonshotai/Kimi-K3` @ `master` 的同名文件 SHA256 与上述值**完全一致**（该仓库仅有 `master` 一个修订，`v1.0.0` / `main` / `release` 等候选均返回 404）；同时 HF 全历史 8 个提交中这两个文件**从未被更新过**。即上游原件哈希有两个独立来源互证。
- 比对结论：本地文件与两个官方来源**均不相同**，确认本地为**修改版本**。与官方原版的 diff：`modeling_kimi_k3.py` 本地独有 17 行 / 官方独有 1 行；`modeling_kimi_linear.py` 本地独有 43 行 / 官方独有 12 行（计数含文件头追加的修改声明）。

**TP-3 — compressed-tensors**

- 上游锚点：`https://github.com/vllm-project/compressed-tensors`，参考 tag `0.16.0`（commit `9c3dd050d29b8b2489527889c78a77b2391d7909`）。
- 对应上游源文件：
  - `src/compressed_tensors/utils/safetensors_load.py` — `get_weight_map`、`update_safetensors_index`、`get_checkpoint_files`、`load_tensors_from_inverse_weight_map`、`_walk_directory_files`
  - `src/compressed_tensors/utils/match.py` — `match_quantizable_tensors`
  - `src/compressed_tensors/entrypoints/convert/converters/base.py` — `build_inverse_weight_maps`、`Converter`
  - `src/compressed_tensors/entrypoints/convert/convert_checkpoint.py` — `exec_jobs`
- 比对结论：`_vendor.py` 共 18 个顶层定义，**9 个可与上游对应**（相似度最高 `get_weight_map` 89.3%），**另 9 个上游无同名实现**（`_is_weights_file`、`_find_safetensors_index_file`、`_find_safetensors_index_path`、`_match_name`、`_is_quantization_param`、`gpu_if_available`、`_is_microscale_scheme`、`validate_safetensors_index`、`validate_weight_for_quantization`），属 HYGON 自研辅助函数。
- 版本选择依据：各 tag 的平均相似度为 `0.16.0` 54.4%、`0.17.1` 54.3%、`0.18.0` 54.3%、`main` 52.5%，均无逐行一致项，故 `0.16.0` 仅作为**最接近的固定参考锚点**；其中 `converters/base.py` 与 `convert_checkpoint.py` 在 `0.16.0`–`0.18.0` 之间未发生变更，该锚点对这两个文件同样成立。

**复核方法（可复现）**

1. 取上游锚点：HF `GET /api/models/moonshotai/Kimi-K3` 取 `sha`；GitHub `GET /repos/vllm-project/compressed-tensors/tags` 取 tag→commit。
2. 下载上游原文件（HF `resolve/<revision>/<path>`；GitHub raw 或 `git show <tag>:<path>`）。
3. 用 AST 抽取双方顶层 `def`/`class` 源码片段，逐名做序列相似度比对；对上游存在的函数再 `git grep -l "def <name>"` 全仓确认位置。
4. 本地文件哈希：`sha256sum <本地路径>`，与上表登记值核对。

---

## 2. 第三方数据资产

| 编号 | 本地路径 | 上游来源 | 固定版本 | 许可证 | 用途与说明 |
| --- | --- | --- | --- | --- | --- |
| TP-D1 | `datasets/EleutherAI___wikitext_document_level/wikitext-2-raw-v1/0.0.0/647234772b9554e208af6c826f23b99e3cac88c8/` | HuggingFace 数据集 `EleutherAI/wikitext_document_level`（配置 `wikitext-2-raw-v1`） | revision `647234772b9554e208af6c826f23b99e3cac88c8`（与本地目录名一致，已核对） | 上游数据集卡片声明 **`cc-by-sa-3.0`**（署名 + 相同方式共享型，**不在自动准入的宽松许可证集合内**；内容派生自 Wikipedia）；本地 `dataset_info.json` 的 `license` 字段为空字符串 | 用于 PPL / 精度诊断；`utils/analyze_quant.py` 直接引用该本地目录 |

TP-D1 各文件校验值（用于固定版本核对）：

| 文件 | SHA256 |
| --- | --- |
| `wikitext_document_level-train.arrow` | `5a61b4eceaf6d85585251938ceaf76a6726e852bd190f87a719c2f61753bc9c0` |
| `wikitext_document_level-validation.arrow` | `6d868d420c1c3bad4d06aaa557bb1761d5096e5aee8078d58341a23f9180682f` |
| `wikitext_document_level-test.arrow` | `1811cc93f559e94692aec5a92d6daff8c837896ff241d61621c4d2327594ab8a` |
| `dataset_info.json` | `15b72c412e6192a029fcb0fac4d8b84ab0a972a0ea51878cf203ded0fcd48ead` |

---


## 3. 运行时依赖

以下组件通过包管理器安装、不随本仓库源码分发，因此本清单不重复登记其许可证文本，使用前请遵守各自许可证：

`torch`、`transformers`、`datasets`、`safetensors`、`triton`、`tqdm`、`numpy`、
`huggingface_hub`、`loguru`、`compressed-tensors`、`llmcompressor`、`lm_eval`、`vllm`。

新增依赖时，仅允许引入 MIT、BSD-2-Clause、BSD-3-Clause、Apache-2.0、0BSD 等宽松许可证组件。
