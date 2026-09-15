# 第三方组件与来源清单（THIRD_PARTY_NOTICES）

本文件登记 one-click-quant 仓库中随源码分发的第三方组件、第三方源码与数据资产。
「本地路径」为仓库内相对路径；「固定版本」用于保证可复现与可追溯。

本仓库整体以 **Apache License 2.0** 发布，完整许可证文本见 [LICENSE.txt](LICENSE.txt)。

---

## 1. 第三方源码（随仓库分发）

| 编号 | 本地路径 | 来源项目 | 上游仓库 / 页面 | 固定版本 | Copyright | 许可证 | HYGON 修改 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TP-1 | `examples/kimi/modeling/modeling_kimi_k3.py` | Kimi-K3 modeling（派生自 `llava/modeling_llava.py`） | HuggingFace `moonshotai/Kimi-K3` | 文件 SHA256 `bd93b7a1cbbb32868c5134b32bfa1ae08909b643fe4e1f8361cb1599e9dc0888` | Copyright 2025-2026 The Moonshot AI Team and HuggingFace Inc. team | 原文件头声明：llava 派生部分为 Apache-2.0；其余部分为 Kimi K3 License（见第 3 节待确认项） | 为适配 mini 模型构建/推理流程所做的接口调整；原始版权声明与许可证声明保持原样 |
| TP-2 | `examples/kimi/modeling/modeling_kimi_linear.py` | Kimi-Linear modeling（派生自 DeepSeek-V3 `modeling_deepseek.py`） | HuggingFace `moonshotai/Kimi-K3` | 文件 SHA256 `1259804dc92b9b77768ff9298b9a329392cff87574a951480eaa28ae0613d0be` | Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team | 原文件头声明：DeepSeek-V3 派生部分为 Apache-2.0；其余部分为 Kimi K3 License（见第 3 节待确认项） | 同上 |
| TP-3 | `templates/slimquant/_vendor.py` | compressed-tensors（函数级 backport） | https://github.com/vllm-project/compressed-tensors | 对齐上游 `compressed_tensors` v0.15.0.1 的接口；文件 SHA256 `79c33af2619c3e4275abe4cb1279755e8dc9f1571c41457004c0edb0048dd7c2` | Copyright the vLLM project | Apache-2.0 | 以「自包含 backport」方式重组，仅保留量化流程实际使用的函数；已在文件头标注来源与修改 |

> TP-1 / TP-2 属于第三方原文件，按规则**保留原始文件头，不追加本仓库版权头**，仅在本清单登记来源。
> TP-3 在保留第三方来源标注的前提下，追加本仓库版权与 SPDX 标识。

---

## 2. 第三方数据资产

| 编号 | 本地路径 | 上游来源 | 固定版本 | 许可证 | 用途与说明 |
| --- | --- | --- | --- | --- | --- |
| TP-D1 | `datasets/EleutherAI___wikitext_document_level/wikitext-2-raw-v1/0.0.0/647234772b9554e208af6c826f23b99e3cac88c8/` | HuggingFace 数据集 `EleutherAI/wikitext_document_level`（配置 `wikitext-2-raw-v1`） | revision `647234772b9554e208af6c826f23b99e3cac88c8` | 上游 WikiText-2 由 Wikipedia 内容派生，通常声明为 CC BY-SA；本地 `dataset_info.json` 的 `license` 字段为空，**结论待确认**（见第 3 节） | 用于 PPL / 精度诊断；`utils/analyze_quant.py` 直接引用该本地目录 |

TP-D1 各文件校验值（用于固定版本核对）：

| 文件 | SHA256 |
| --- | --- |
| `wikitext_document_level-train.arrow` | `5a61b4eceaf6d85585251938ceaf76a6726e852bd190f87a719c2f61753bc9c0` |
| `wikitext_document_level-validation.arrow` | `6d868d420c1c3bad4d06aaa557bb1761d5096e5aee8078d58341a23f9180682f` |
| `wikitext_document_level-test.arrow` | `1811cc93f559e94692aec5a92d6daff8c837896ff241d61621c4d2327594ab8a` |
| `dataset_info.json` | `15b72c412e6192a029fcb0fac4d8b84ab0a972a0ea51878cf203ded0fcd48ead` |

---

## 3. 待法务 / 合规确认项

1. **Kimi K3 License**：TP-1、TP-2 的文件头声明「其余部分遵循 Kimi K3 License」，该许可证不在自动准入的宽松许可证集合（MIT、BSD-2-Clause、BSD-3-Clause、Apache-2.0、0BSD）内。
   - 需确认这两个文件是否可以随 Apache-2.0 仓库对外发布。
   - 若不可，应从开源分支移除，仅在内部仓库或运行时从上游模型仓库获取。
2. **compressed-tensors 溯源**：TP-3 需确认上游 Copyright 归属主体与对应的固定 commit / tag。
3. **wikitext 数据资产**：TP-D1 需给出明确结论——保留在 Git 中分发（需先确认许可证与署名/相同方式共享义务），或改为运行时下载（与 `datasets/ultrachat_200k` 的做法保持一致，见 README「环境准备」），或从 Git 移除。
4. TP-D1 同时是质量清单中的 `GIT.LARGE_BLOB_REVIEW` 项（train 分片 10.4 MiB）。若结论为保留，需确认无需 Git LFS；若结论为移除，该核对项同时关闭。

---

## 4. 运行时依赖

以下组件通过包管理器安装、不随本仓库源码分发，因此本清单不重复登记其许可证文本，使用前请遵守各自许可证：

`torch`、`transformers`、`datasets`、`safetensors`、`triton`、`tqdm`、`numpy`、
`huggingface_hub`、`loguru`、`compressed-tensors`、`llmcompressor`、`lm_eval`、`vllm`。

新增依赖时，仅允许引入 MIT、BSD-2-Clause、BSD-3-Clause、Apache-2.0、0BSD 等宽松许可证组件。
