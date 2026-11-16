# 阶段1 环境与数据集

> **权威来源**：仓库根 `README.md`「环境准备」「下载量化校准数据集」「常见故障排查」三节。
> 镜像与依赖按用户实际环境调整，本文只写**怎么选、怎么判、缺什么先停**。

## Docker

**先判当前是否已在容器内**，再决定要不要起新容器。已在容器里时**不要**再 `docker run`。

```bash
# 任一命中即视为已在容器内（覆盖 Docker / containerd / k8s 常见形态）
if [ -f /.dockerenv ] \
   || grep -qaE 'docker|containerd|kubepods|libpod' /proc/1/cgroup 2>/dev/null \
   || [ -f /run/.containerenv ]; then
  echo "已在容器内 —— 跳过 docker pull/run，直接做依赖与运行前提自检"
else
  echo "宿主机环境 —— 需要用户提供镜像后再起容器"
fi
```

- **已在容器内** → 跳到「Python 依赖」与文末「自检」；不要要 `IMAGE`，不要 `docker run`。
- **不在容器内** → 再向用户确认镜像是下面哪一种，再 pull/run：

  - **registry/image:tag** —— 远程镜像，需要 `docker pull`（如 README 中的 `vllm/vllm-openai:<tag>`）
  - **image ID**（12~64 位十六进制，如 `1a2b3c4d5e6f`）—— 仅本机已有，**跳过** `docker pull`

```bash
# 仅「不在容器内」时执行。示例：
#   IMAGE="vllm/vllm-openai:<tag>"       # 远程 tag → 需要 pull（tag 以用户/README 为准）
#   IMAGE="1a2b3c4d5e6f"                 # 本机 ID → 不要 pull
IMAGE="<用户提供的镜像 tag 或 ID>"
NAME="${NAME:-test_quant}"

# 只对远程引用 pull；本机 ID / digest 跳过。
if [[ "$IMAGE" =~ ^[0-9a-fA-F]{12,64}$ || "$IMAGE" == sha256:* ]]; then
  echo "本机镜像 ID/digest —— 跳过 pull"
else
  docker pull "$IMAGE"
fi

# 下面参数全是占位：GPU 暴露方式、shm、特权、包管理器随加速栈与发行版变化。
# 先问用户实际 docker run 习惯；没有约定时再给最小可改模板，禁止原样假定 NVIDIA + 640g。
docker run <GPU与设备参数> -itd --entrypoint /bin/bash \
  --network=host --shm-size=<按机器调，大模型通常要较大> \
  --name "$NAME" "$IMAGE"
docker exec -it "$NAME" bash
# 缺 git 时按发行版安装：apt / yum / dnf / apk …（不要写死 apt）
```

常见加速栈差异（选一，不要叠用）：

| 栈 | 典型设备参数 | 可见设备环境变量 |
|---|---|---|
| NVIDIA Docker | `--gpus all` | `CUDA_VISIBLE_DEVICES` |
| ROCm / DCU 等 | 按镜像文档挂 `/dev/kfd`、`/dev/dri` 等 | `HIP_VISIBLE_DEVICES` 或厂商文档指定变量 |
| 仅 CPU 冒烟 | 不挂 GPU | 无 |

Agent 检查清单：

1. **先**做容器内/外判定；已在容器内则跳过本段后续，进入依赖与自检。
2. 仅不在容器内时才要 `IMAGE`（tag 或 ID）以及可选的容器名 `NAME`。
3. 形如镜像 ID（短/长十六进制、不含 `/`）时跳过 `docker pull`，只 `docker run`。
4. 形如 `registry/...:tag` 时先确认完整字符串，再 `pull` + `run`。
5. 两者都存在时优先用 tag —— ID 只在本机有效，不可移植。
6. **禁止**默认抄 `--gpus all` / `--shm-size=640g` / `--device /dev/mem` / `apt`；
   按用户加速栈与机器规格填上表，shm 只在大模型时加大。

## Python 依赖

```bash
pip install llmcompressor
pip install 'lm_eval[vllm,api]'
```

本仓库没有 pin 依赖版本，`llmcompressor` / `compressed-tensors` 混装出问题时的第一判据是版本本身：

```bash
pip show llmcompressor compressed-tensors | grep -E "Name|Version"
```

换版本属于**上游回退手段，不是本仓库保证过的路径**：换完必须重跑 `04-verify-eval.md` 的产物契约与
dtype 校验，确认产物形态没变，再谈精度。需要装 main 时：

```bash
pip uninstall llmcompressor compressed-tensors -y
pip install git+https://github.com/vllm-project/llm-compressor.git   # 离线环境先 clone 再 pip install -e .
```

## 代码与校准数据

```bash
git clone https://github.com/HYGON-AI/one-click-quant.git
cd one-click-quant
```

校准集（ultrachat）任选一种渠道落到本地目录即可，**不要绑死某一下载工具**：

```bash
# 已有目录：软链或 --dataset 直指
ln -s <已有 ultrachat 目录> datasets/ultrachat_200k
# 或：python3 main.py ... --dataset <真实目录>

# ModelScope（国内常见）
modelscope download --dataset HuggingFaceH4/ultrachat_200k --local_dir datasets/ultrachat_200k

# Hugging Face Hub（需网络 / 镜像）
huggingface-cli download HuggingFaceH4/ultrachat_200k --local-dir datasets/ultrachat_200k
```

CLI 默认校准路径：`./datasets/ultrachat_200k`。PPL 用的 wikitext **不是**这份，见 `04-verify-eval.md`。

## 运行前提（三条硬性要求）

跑 `main.py` / `--alg preprocess` / `build_mini_*` 前必须同时满足这三条。
任一不满足都会在**启动阶段**直接失败，与量化方案本身无关。

> 这三条是**唯一的通用前提**。容器内其它失败必须按报错分类定位（镜像、驱动、shm、挂载、网络、缺依赖），
> 不存在「固定的 `docker run` 复跑一遍就能好」的通用解。

### 1. Python ≥ 3.10

判定依据：`main.py` 使用 `str | None`、`dict[str, ...]` 等注解且**没有** `from __future__` 兜底 ——
这类注解在函数定义时就会求值，3.10 之前不支持 `X | Y` 写法。

典型症状（版本越低越早失败）：

```text
# Python 3.6 及更早：连 from __future__ import annotations 都不认
SyntaxError: future feature annotations is not defined

# Python 3.7 ~ 3.9：能过 future import，倒在注解求值上
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

真实暴露点可能是任意一个模块（`main.py` 的导入链很长），**先看 `python3 --version`，不要逐个改语法**。

处理：换用 ≥3.10 的解释器（一般就是进容器跑）。**不要改代码去迁就旧解释器。**

### 2. 工作目录可写

判定依据：`utils/logging_config.py` 在 **import 阶段**就 `os.makedirs(<repo>/logs)` 并打开日志文件。
仓库目录只读时连 `python3 main.py --help` 都起不来。

典型症状：

```text
OSError: [Errno 30] Read-only file system: <repo>/logs/quant_*.log
```

检查：

```bash
test -w /path/to/one-click-quant && echo RW || echo RO
```

处理：若为 RO，说明该路径是只读挂载。容器里同一份代码常同时存在**多个挂载点、权限不同**，
换到可写的那条路径再跑；不要假设宿主机路径在容器里也可写。

### 3. 站在仓库根目录执行

判定依据：`main.py` 用**相对路径**加载模板（`templates/<name>.py`）。

典型症状：

```text
FileNotFoundError: <cwd>/templates/slimquant_ptq.py
```

处理：`cd` 到仓库根再执行，不要把 `main.py` 的绝对路径丢给别的工作目录。

## 自检

在目标运行环境（仓库根目录）执行，四项全过再开始量化：

```bash
python3 --version                                  # 期望 ≥3.10
test -w "$(pwd)" && echo "目录可写"
ls templates/slimquant_ptq.py                      # 确认站在仓库根
python3 main.py --help
```

## 许可证提醒

- 本仓库自有代码：Apache-2.0 文件头（见 README）。
- Kimi 的 modeling 文件与部分数据集使用**不同**许可证，不要删除上游声明。见 `THIRD_PARTY_NOTICES.md`。
