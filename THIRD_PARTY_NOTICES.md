# 第三方组件与来源清单（THIRD_PARTY_NOTICES）

本文件登记 one-click-quant 仓库中随源码分发的第三方组件、第三方源码与数据资产。
「本地路径」为仓库内相对路径；「固定版本」用于保证可复现与可追溯。

本仓库整体以 **Apache License 2.0** 发布，完整许可证文本见 [LICENSE.txt](LICENSE.txt)。

---

## 1. 第三方源码（随仓库分发）

| 编号 | 本地路径 | 来源项目 | 上游仓库 / 页面 | 固定版本 | Copyright | 许可证 | HYGON 修改 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TP-1 | `examples/kimi/modeling/modeling_kimi_k3.py` | Kimi-K3 modeling（上游文件 `modeling_kimi_k3.py`；文件头声明部分派生自 llava `llava/modeling_llava.py`） | https://huggingface.co/moonshotai/Kimi-K3 | revision `f831ab66814297da540d832a5235f8e904f29d06`（2026-09-02）；上游文件 SHA256 `b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2`；本地文件 SHA256 `bd93b7a1cbbb32868c5134b32bfa1ae08909b643fe4e1f8361cb1599e9dc0888` | Copyright 2025-2026 The Moonshot AI Team and HuggingFace Inc. team；上游 `LICENSE` 归属 Copyright (c) 2026 Moonshot AI | 原文件头声明：llava 派生部分为 Apache-2.0；其余部分为 Kimi K3 License（**许可证全文**：[`LICENSES/Kimi-K3-License.txt`](LICENSES/Kimi-K3-License.txt)；处置与义务履行方式见第 1.2 节） | 与上游该 revision 逐行 diff 共 **10 行**差异：`tie_weights()` 按 transformers 版本分发参数（≥5.2 传入 `missing_keys` / `recompute_mapping`），相应新增 `import transformers` 与 `from packaging import version`；原始版权与许可证声明保持原样 |
| TP-2 | `examples/kimi/modeling/modeling_kimi_linear.py` | Kimi-Linear modeling（上游文件 `modeling_kimi_linear.py`；文件头声明 MoE/MHA 部分派生自 DeepSeek-V3 `modeling_deepseek.py`） | https://huggingface.co/moonshotai/Kimi-K3 | revision `f831ab66814297da540d832a5235f8e904f29d06`（2026-09-02）；上游文件 SHA256 `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`；本地文件 SHA256 `1259804dc92b9b77768ff9298b9a329392cff87574a951480eaa28ae0613d0be` | Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team；上游 `LICENSE` 归属 Copyright (c) 2026 Moonshot AI | 原文件头声明：DeepSeek-V3 派生部分为 Apache-2.0；其余部分为 Kimi K3 License（**许可证全文**：[`LICENSES/Kimi-K3-License.txt`](LICENSES/Kimi-K3-License.txt)；处置与义务履行方式见第 1.2 节） | 与上游该 revision 逐行 diff 共 **41 行**差异：修复 `OutputRecorder` 导入以兼容 transformers ≥5.2（迁移至 `utils.output_capturing`）；`normal_()` 前增加 dtype 保护（float8 等跳过）；默认 attention 实现由强制 `flash_attention_2` 改为 eager，避免 hub kernel 动态下载 |
| TP-3 | `templates/slimquant/_vendor.py` | compressed-tensors（函数级参考实现，自包含改写，**非逐行复制**） | https://github.com/vllm-project/compressed-tensors | 参考 tag `0.16.0`（commit `9c3dd050d29b8b2489527889c78a77b2391d7909`，比对后最接近版本）；本地文件 SHA256 `79c33af2619c3e4275abe4cb1279755e8dc9f1571c41457004c0edb0048dd7c2`；对应上游源文件见第 1.1 节 | Copyright the vLLM project | Apache-2.0 | 自包含重组，仅保留量化流程实际使用的函数：18 个顶层定义中 9 个可与上游对应（最高 `get_weight_map` 89.3%、`update_safetensors_index` 75.4%、`exec_jobs` 72.2%），另 9 个为 HYGON 自研（上游无同名实现）；已在文件头标注来源与修改 |

> TP-1 / TP-2 属于第三方原文件，按规则**保留原始文件头，不追加本仓库版权头**，仅在本清单登记来源。
> TP-3 在保留第三方来源标注的前提下，追加本仓库版权与 SPDX 标识。

### 1.1 来源锚点与溯源证据（2026-09-20 复核）

**TP-1 / TP-2 — Kimi-K3 modeling**

- 上游锚点：`https://huggingface.co/moonshotai/Kimi-K3`，revision `f831ab66814297da540d832a5235f8e904f29d06`（lastModified 2026-09-02），获取方式 `GET https://huggingface.co/api/models/moonshotai/Kimi-K3` 的 `sha` 字段。
- 原文件路径：`modeling_kimi_k3.py`、`modeling_kimi_linear.py`（与本地文件同名）。
- 上游文件 SHA256：`b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2`、`9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`。
- 比对结论：以该 revision 的原文件与本地文件逐行 `diff`，差异分别为 10 行与 41 行，且差异内容全部为 HYGON 适配（见上表「HYGON 修改」列），确认本地文件是**该 revision 的修改版本**。

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

### 1.2 许可证决定记录（TP-1 / TP-2，处置方案 (a)：保留文件分发）

**处置决定**：采纳方案 (a)——`examples/kimi/modeling/modeling_kimi_k3.py` 与 `examples/kimi/modeling/modeling_kimi_linear.py` **保留在仓库中随源码分发**，上游条件 1–5 按下表逐条履行。
**生效条件**：本页为研发侧执行记录；**书面批准结论须由法务/合规在末尾签署栏签署后生效**。

| 上游条款 | 义务 | 本仓库履行方式 |
| --- | --- | --- |
| 条件 1 | 所有副本或实质部分须包含版权声明与许可声明 | ① 两个文件保留原始文件头（Moonshot AI / HuggingFace Inc. / DeepSeek-AI 版权与许可说明）；② 仓库内随源码分发许可证全文 `LICENSES/Kimi-K3-License.txt`（上游 revision `f831ab66814297da540d832a5235f8e904f29d06` 的**逐字节副本**，SHA256 `20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887`）；③ 本清单 TP-1/TP-2 登记并链接该文件；④ `README.md`「许可证与第三方组件」明示适用范围 |
| 条件 2 | 以 “Model as a Service” 形式对外提供且许可方及关联方连续 12 个月合计营收 > 2000 万美元时，需先与 Moonshot AI 单独签约 | 本仓库为离线量化工具链，**不提供模型推理/微调的 MaaS 服务**；若 HYGON 或其关联方后续开展 MaaS 业务并触发营收门槛，须在启用前完成单独签约——该事实判断由法务/业务在签署时确认并登记责任人 |
| 条件 3 | 用于月活 > 1 亿或月营收 > 2000 万美元的商业产品时，须在界面显著展示 “Kimi K3” | 本仓库自身无用户界面，不触发；该义务随衍生分发传递给下游使用者，已在 README 与本节作提示；若 HYGON 自研产品触发门槛，须在其 UI 显著位置展示 “Kimi K3” |
| 条件 4（豁免） | 内部使用、Moonshot AI 官方/认证渠道不受条件 2、3 约束 | 已记录，供内部使用场景引用 |
| 条件 5 | AS IS 免责 | 原样保留于许可证全文 |

**其他约束**

- 不得将这两个文件纳入本仓库 Apache-2.0 授权范围，也不得在文件头中统一改写为 Apache-2.0。
- 不得删除、替换或改写两个文件中的原始版权与许可声明。
- 回退方案：若法务最终判定不可发布（方案 (b)），从开源分支移除这两个文件，并同步清理第 1 节、`README.md`、`LICENSES/` 及 `examples/kimi/` 中的引用。

**签署栏（待法务 / 合规填写）**

| 项目 | 内容 |
| --- | --- |
| 批准结论 | ☐ 同意方案 (a)（保留文件分发）　☐ 不同意（改用方案 (b)） |
| 签署人 / 部门 |  |
| 签署日期 |  |
| 核对依据 | `LICENSES/Kimi-K3-License.txt` @ SHA256 `20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887`（上游 revision `f831ab66…`） |

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

## 3. 待法务 / 合规确认项

1. **Kimi K3 License（处置方案已选定 (a)，待法务书面签署）**：TP-1、TP-2 的文件头声明「其余部分遵循 Kimi K3 License」，该许可证不在自动准入的宽松许可证集合（MIT、BSD-2-Clause、BSD-3-Clause、Apache-2.0、0BSD）内。**处置决定：采纳方案 (a)，两个文件保留在仓库中随源码分发；条件 1–5 的履行方式已逐条登记在第 1.2 节。** 上游 `LICENSE` 正文要点（Copyright (c) 2026 Moonshot AI）：
   - 授权范围：允许 use / copy / modify / merge / publish / distribute / sublicense / sell，且允许创建衍生工作。
   - 条件 1：所有副本或实质部分必须保留版权与许可声明。
   - 条件 2：若以「Model as a Service」（使第三方对输入/参数/训练数据具有实质控制权的 API 形态）形式对外提供，且许可方及其关联主体连续 12 个月合计营收 > 2000 万美元，需先与 Moonshot AI 单独签约。
   - 条件 3：若用于月活 > 1 亿 或 月营收 > 2000 万美元的商业产品或服务，须在用户界面显著展示 "Kimi K3"。
   - 条件 4：条件 2、3 **不适用于**内部使用（不向第三方提供软件/输出/底层能力）与通过 Moonshot AI 官方产品、认证推理渠道的使用。
   - 条件 5：AS IS 免责。
   - **待法务书面签署**：在第 1.2 节签署栏填写「同意方案 (a)」后本条转为已闭环；若签署为「不同意」，则改用方案 (b)：从开源分支移除这两个文件，仅在内部仓库或运行时从上游模型仓库获取。
   - ❗ 注意：本仓库整体 Apache-2.0 与 TP-1/TP-2 的 Kimi K3 License **不是同一许可证**，不得把这两个文件的文件头统一改写为 Apache-2.0。
2. **compressed-tensors 溯源（已核实，2026-09-20）**：上游 Copyright 归属主体已确认为 `Copyright the vLLM project`，许可证为 Apache-2.0（属自动准入集合）；已补充固定参考 tag `0.16.0`（commit `9c3dd050d29b8b2489527889c78a77b2391d7909`）及对应的 4 个上游源文件路径（见第 1.1 节）。
   - 唯一保留说明：`_vendor.py` 属自包含改写，与任一上游 tag 均非逐行一致（平均相似度 54.4%），因此登记形式为「参考实现 + 相似度证据」而非「复制来源」；若合规口要求必须给出唯一复制来源，请以本条作为待复核项保留。
3. **wikitext 数据资产（上游许可证已定位，待决策）**：上游 `EleutherAI/wikitext_document_level` 在固定 revision `647234772b9554e208af6c826f23b99e3cac88c8` 的数据集卡片声明许可证为 **`cc-by-sa-3.0`**（署名 + 相同方式共享型，**不在自动准入的宽松许可证集合内**），本地 `dataset_info.json` 的 `license` 字段为空字符串。
   - 影响：若随 Apache-2.0 仓库对外分发，需履行署名（attribution）与「相同方式共享」（share-alike）义务，与仓库整体的宽松许可策略不一致。
   - 建议（推荐 b）：（a）法务书面确认可在 Apache-2.0 仓库中分发，并记录署名/SA 义务的履行方式；（b）改为**运行时下载**（与 `README.md` 中 `modelscope download --dataset HuggingFaceH4/ultrachat_200k` 的做法一致），并将该目录从 Git 移除；（c）仅保留在内部仓库使用。
   - 若选（b），需同步给 `utils/analyze_quant.py` 的 `_WIKI_DIR` 增加「本地不存在则下载」的兜底逻辑。
   - **若按与 TP-1/TP-2 相同的口径（保留文件分发，方案 (a)）**，义务履行方式为：① 在本清单第 2 节登记来源、固定 revision、上游许可证与各文件 SHA256（履行署名义务）；② 明确该数据集仍为 `cc-by-sa-3.0`，**不纳入本仓库 Apache-2.0 授权**（履行相同方式共享义务）；③ 本仓库未对数据内容做实质性修改，仅以 arrow 分片形式原样分发；④ 批准结论与 TP-1/TP-2 一并在第 1.2 节签署栏签署。
4. TP-D1 同时是质量清单中的 `GIT.LARGE_BLOB_REVIEW` 项（train 分片 10.4 MiB）。若结论为保留，需确认无需 Git LFS；若结论为移除，该核对项同时关闭。

---

## 4. 运行时依赖

以下组件通过包管理器安装、不随本仓库源码分发，因此本清单不重复登记其许可证文本，使用前请遵守各自许可证：

`torch`、`transformers`、`datasets`、`safetensors`、`triton`、`tqdm`、`numpy`、
`huggingface_hub`、`loguru`、`compressed-tensors`、`llmcompressor`、`lm_eval`、`vllm`。

新增依赖时，仅允许引入 MIT、BSD-2-Clause、BSD-3-Clause、Apache-2.0、0BSD 等宽松许可证组件。
