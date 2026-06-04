# 目录
- [简介](#简介)
- [环境准备](#环境准备)
- [通用量化方法](#通用量化方法)
  - [BF16转W8A8 FP8](#bf16转w8a8-fp8)
  - [BF16转W8A8 INT8](#bf16转w8a8-int8)
  - [BLOCK FP8转BF16](#block-fp8转bf16)
- [定制化模型量化方法](#定制化模型量化方法)
- [强制使用某个量化算法](#强制使用某个量化算法)
- [slimquant(w4a8)量化方法](#slimquant-w4a8量化方法)
- [PPL评估](#ppl评估)
- [精度调优](#精度调优)
  - [离群通道分析](#离群通道分析)
  - [权重静态均方误差分析](#权重静态均方误差分析)
  - [激活动态均方误差分析](#激活动态均方误差分析)
  - [Fisher对角敏感层分析](#fisher对角敏感层分析)
- [提交PR](#提交pr)
- [常见故障排查 (Trouble shooting)](#常见故障排查-trouble-shooting)

# 简介

基于llmcompressor开发的一键量化工具。

# 环境准备

### docker

以nv环境为例

```bash
docker pull vllm/vllm-openai:v0.19.0
docker run --gpus all -itd --entrypoint /bin/bash --privileged=true --network=host --device /dev/mem --shm-size=640g --name test_quant vllm/vllm-openai:v0.19.0
docker exec -it test_quant bash
apt update
apt install git
```

### 工具

```bash
pip install llmcompressor
pip install lm_eval[vllm,api]
```

### 代码和数据

```bash
# 下载one_click_quant代码工程
git clone https://developer.sourcefind.cn/codes/OpenDAS/one_click_quant.git
cd one_click_quant
# 下载量化校准数据集，并软链接到当前目录的datasets/ultrachat_200k
modelscope download --dataset HuggingFaceH4/ultrachat_200k
ln -s /root/.cache/modelscope/hub/datasets/HuggingFaceH4/ultrachat_200k datasets/ultrachat_200k
```

> 如果已经有下载好的数据集，可以直接软链接，或者测试时通过\-\-dataset指定数据集文件夹路径。
> 
> 

### 查看帮助

```bash
python3 main.py --help
```

> 建议先熟悉\-\-help输出的命令参数和描述。
> 
> 

# 通用量化方法

### BF16转W8A8 FP8

```bash
python3 main.py --model /models/Qwen2.5-0.5B-Instruct --scheme FP8_DYNAMIC
```

> 量化输出模型默认保存在输入模型同目录下，可通过\-\-save\-dir进行修改。
> 
> 

### BF16转W8A8 INT8

```bash
python3 main.py --model /models/Qwen2.5-0.5B-Instruct --scheme W8A8
```

### BLOCK FP8转BF16

```bash
python3 main.py --model /models/Qwen3-1.7B-FP8 --scheme FP8_TO_BF16
```

更多的量化方法可查看[quant_scheme](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py)

# 定制化模型量化方法

通过\-\-scheme扩展自定义命令，把之前可以正常运行的量化代码整合成模板代码。

```bash
options:
  --scheme SCHEME       Quantization scheme. Options:
                          FP8_DYNAMIC  - BF16 to per-channel FP8
                          W8A8         - BF16 to per-channel INT8
                          FP8_TO_BF16  - Block FP8 to BF16
                          DeepseekV3_FP8_TO_INT8  - Block FP8 to per-channel INT8 for DeepSeek V3 series models
                          DeepseekV3_FP8_TO_FP8  - Block FP8 to per-channel FP8 for DeepSeek V3 series models
                          Qwen35_MOE_BF16_TO_INT8  - BF16 to per-channel INT8 for Qwen3.5 series models (default: FP8_DYNAMIC)
```

如

```bash
python3 main.py --model /models/DeepSeek-R1 --scheme DeepseekV3_FP8_TO_INT8
```

# 强制使用某个量化算法

一键量化内部有一套默认方案，但可以使用\-\-alg强制使用某种量化算法，使用\-\-needs\-data决定是否使用数据集校准。

这里特别推荐尝试model\_free\_ptq模式，不依赖transformers库加载模型，不使用数据集校准，速度快，量化后精度不一定差。

```bash
options:
  --alg ALG             Quantization algorithm (forced when not None, overrides scheme inference).
                          ptq:      - PTQ quantization
                          gptq:     - GPTQ quantization
                          awq:      - AWQ quantization
                          gptq_awq: - GPTQ + AWQ quantization
                          smoothquant_gptq: - SmoothQuant GPTQ quantization
                          smoothquant_awq: - SmoothQuant AWQ quantization
                          smoothquant_gptq_awq: - SmoothQuant GPTQ + AWQ quantization
                          model_free_ptq: - Data-free model-free PTQ
                          slimquant_ptq:  - Data-free SlimQuant PTQ (W4A8) (default: None)
  --needs-data {true,false}
                        Override whether calibration data is required.
                        Set 'true' to force loading a dataset, 'false' to skip.
                        When not set, the default is determined by --scheme. (default: None)
```

> 截止到260529，llmcompressor支持的最高transformers版本为4\.57\.6，此版本还不支持Qwen3\.5系列模型，如果要对Qwen3\.5进行量化，需要在命令行添加\-\-alg model\_free\_ptq。
> 
> 

# slimquant (w4a8)量化方法

`slimquant_ptq` 是免校准以及免transformer加载模型，尤其针对新模型或自定义模型等非transformer支持模型量化。直接操作 safetensors 权重文件的数据。支持三种量化模式：**INT4**（W4A8，MoE 层做chananel\_wise量化）、**INT8**（W8A8）、**FP8**（逐通道 E4M3）。

与 `model_free_ptq` 的区别：**INT4** 模式下，MoE expert 层额外进行 INT4 \+ 2×int4→int8 打包，非 MoE 层仅做 INT8。

slimquant主要用来支持INT4量化，INT8/FP8量化建议使用上面章节的方法\(底层实现一样，更通用\)。

```bash
#文件位置
examples\one_click_quant\main.py

# W4A8 量化 (int4, 默认)
python main.py --model DeepSeek-V3.2-bf16/ --alg slimquant_ptq --scheme W4A8
# INT8 量化
python main.py --model DeepSeek-V3.2-bf16/ --alg slimquant_ptq --scheme W8A8
# FP8 量化
python main.py --model DeepSeek-V3.2-bf16/ --alg slimquant_ptq --scheme FP8_DYNAMIC
# 指定输出目录 + 自定义 ignore
python main.py --model DeepSeek-V3.2-bf16/ --alg slimquant_ptq --scheme W4A8 \
    --save-dir ./output-w4a8 --ignore "lm_head, re:.*mlp.gate$, re:.*embed_tokens.*"
```

当前已支持ignore的模型，启动无需加\-\-ignore参数，未匹配的模型会采用默认ignore参数""lm\_head, re:\.\*mlp\.gate$, re:\.\*embed\_tokens\.\*""

# PPL评估

```bash
python3 main.py --lm-eval "pretrained=/models/Qwen2.5-0.5B-Instruct-FP8_DYNAMIC,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.6"

| Tasks  |Version|Filter|n-shot|    Metric     |   | Value |   |Stderr|
|--------|------:|------|-----:|---------------|---|------:|---|------|
|wikitext|      2|none  |     0|bits_per_byte  |↓  | 0.7851|±  |   N/A|
|        |       |none  |     0|byte_perplexity|↓  | 1.7232|±  |   N/A|
|        |       |none  |     0|word_perplexity|↓  |18.3558|±  |   N/A|
```

- 如果评估过程中遇到"CUDA out of memory"，可以尝试降低参数gpu\_memory\_utilization的值。

- 或者使用vllm server启动推理服务，再使用lm\_eval进行PPL评估。

    - lm\_eval运行参考：HF\_DATASETS\_OFFLINE=1 HF\_DATASETS\_CACHE=\./datasets lm\_eval \-\-model local\-completions \-\-model\_args model=/models/Qwen2\.5\-0\.5B\-Instruct\-FP8\_DYNAMIC,base\_url=http://localhost:8000/v1/completions,add\_bos\_token=true \-\-tasks wikitext \-\-batch\_size 1

# 精度调优

当PPL精度不符合期望时，需要做敏感层分析，优化ignore参数。

目前的分析功能还处于初级阶段，仅供参考。

```bash
options:
  --analysis {outlier,mse,fisher,activation_sensi,outlier_mse,none}
                        Which analysis to run.
                          none:             - do not run analysis
                          outlier:          - outlier channel analysis (needs --model)
                          mse:              - per-channel MSE (needs --model and --save-dir)
                          fisher:           - Fisher diagonal sensitivity (needs --model)
                          activation_sensi: - activation-aware sensitivity (needs --model and --save-dir)
                                              recommended: --num-calibration-samples 128 --max-seq-length 512
                          outlier_mse:      - outlier + MSE (default: none)
```

### 离群通道分析

分析权重的离群通道

```bash
# python3 main.py --model /models/Qwen3-4B  --analysis outlier --analysis-top 10
================================================================================
QUANTIZATION ERROR ANALYSIS REPORT
================================================================================
Base model:      /models/Qwen3-4B
Analysis:        outlier
Outlier threshold: 3.0x median

-- Channel Outlier Analysis (base model) --
  Total Linear layers: 253
  Outlier-sensitive layers: 10 (4.0%)
  Recommend AWQ: True
  Recommend SmoothQuant: False
  Recommend ignore: ['model.layers.2.mlp.up_proj', 'model.layers.6.mlp.down_proj', 'model.layers.6.self_attn.q_proj', 'model.layers.7.mlp.down_proj', 'model.layers.16.mlp.down_proj']

  Top 10 Layers by Outlier Ratio:
  Layer                                                   Outlier%   Max/Med      MaxAbs   OutCh
  -------------------------------------------------------  --------  --------  ----------  ------
  model.layers.2.mlp.up_proj                                 22.53%      16.4      0.5273   2192/9728
  model.layers.9.self_attn.k_proj                             2.93%       8.7      0.7812     30/1024
  model.layers.12.self_attn.k_proj                            2.34%       5.5      0.4922     24/1024
  model.layers.11.self_attn.k_proj                            2.15%       4.1      0.3750     22/1024
  model.layers.2.self_attn.k_proj                             2.05%       4.7      0.4160     21/1024
  model.layers.19.self_attn.k_proj                            2.05%       5.3      0.4141     21/1024
  model.layers.4.self_attn.k_proj                             1.95%       8.5      0.7422     20/1024
  model.layers.15.self_attn.k_proj                            1.86%       4.2      0.3594     19/1024
  model.layers.22.self_attn.k_proj                            1.86%      10.0      0.7344     19/1024
  model.layers.13.self_attn.k_proj                            1.66%       7.4      0.6523     17/1024

-- Recommendations --
  [OK] Recommend AWQ preprocessing based on outlier analysis.
================================================================================
```

主要关注“Outlier%”列数据，值越大表示这个算子离群越严重，可以尝试放到ignore中重新量化测试，对比精度是否有改善。

### 权重静态均方误差分析

计算量化前后每层权重的静态MSE

```bash
# python3 main.py --model /models/Qwen3-4B --save-dir /models/Qwen3-4B-W8A8 --analysis mse --analysis-top 10
================================================================================
QUANTIZATION ERROR ANALYSIS REPORT
================================================================================
Base model:      /models/Qwen3-4B
Quantized model: /models/Qwen3-4B-W8A8
Analysis:        mse
Outlier threshold: 3.0x median

-- Top 10 Layers by Per-Channel MSE (253 analyzed) --
  Layer                                                       Mean MSE     Corr  Outlier%
  -------------------------------------------------------  ------------  -------  --------
  model.layers.4.mlp.gate_proj                                 0.014219    0.498     0.00%
  model.layers.3.mlp.gate_proj                                 0.013686    0.565     0.03%
  model.layers.2.mlp.gate_proj                                 0.010513    0.848     0.12%
  model.layers.1.mlp.gate_proj                                 0.008415    0.837     0.02%
  model.layers.9.mlp.gate_proj                                 0.008399    0.589     0.13%
  model.layers.7.mlp.gate_proj                                 0.007401    0.710     0.20%
  model.layers.10.mlp.gate_proj                                0.006786    0.432     0.16%
  model.layers.6.mlp.gate_proj                                 0.006268    0.591     0.12%
  model.layers.11.mlp.gate_proj                                0.006115    0.359     0.22%
  model.layers.5.mlp.gate_proj                                 0.005927    0.789     0.09%

-- Outlier<->MSE Correlation (threshold > 0.70) --
  High-corr layers (all analyzed): 175
  High-corr layers (within top-10 MSE): 4
  Top-4 high-corr layers within top-MSE:
  model.layers.2.mlp.gate_proj                             corr=0.848  mse=0.010513  outlier=0.12%
  model.layers.1.mlp.gate_proj                             corr=0.837  mse=0.008415  outlier=0.02%
  model.layers.7.mlp.gate_proj                             corr=0.710  mse=0.007401  outlier=0.20%
  model.layers.5.mlp.gate_proj                             corr=0.789  mse=0.005927  outlier=0.09%

-- Recommendations --
  [OK] High outlier-MSE correlation confirmed -> AWQ is likely to help.
  Ignore candidates (score=mean_mse*(1+5*outlier), outlier>=0.50%):
    model.layers.2.mlp.up_proj  mse=0.005911  outlier=22.53%  corr=0.924
    model.layers.9.self_attn.k_proj  mse=0.001562  outlier=2.93%  corr=0.849
    model.layers.15.self_attn.k_proj  mse=0.001564  outlier=1.86%  corr=0.774
    model.layers.12.self_attn.k_proj  mse=0.001402  outlier=2.34%  corr=0.835
    model.layers.11.self_attn.k_proj  mse=0.001250  outlier=2.15%  corr=0.817
================================================================================
```

主要关注“Mean MSE”列数据，值越大表示这个算子量化前后误差越大，可以尝试放到ignore中重新量化测试，对比精度是否有改善。

### 激活动态均方误差分析

使用wikitext validation数据集作为激活数据，对量化前后的模型进行推理，计算每层的动态MSE

```bash
# python3 main.py --model /models/Qwen2.5-0.5B --save-dir /model/Qwen2.5-0.5B-W8A8 --analysis activation_sensi --num-calibration-samples 128 --max-seq-length 512
================================================================================
QUANTIZATION ANALYSIS REPORT
================================================================================
Base model:      /models/Qwen2.5-0.5B
Quantized model: /model/Qwen2.5-0.5B-W8A8
Analysis:        activation_sensi

-- Activation Sensitivity Ranking (top 10) --
  Rank  Layer    Linear                          MSE
  -----  ------  ------------------------  --------------
  1      21      mlp.down_proj               3.542094e-01
  2      3       mlp.down_proj               8.370512e-02
  3      23      mlp.down_proj               5.294336e-02
  4      21      self_attn.v_proj            3.714093e-02
  5      23      self_attn.v_proj            3.635331e-02
  6      22      self_attn.v_proj            3.088628e-02
  7      11      self_attn.k_proj            2.261675e-02
  8      16      self_attn.k_proj            2.225596e-02
  9      23      mlp.gate_proj               2.194914e-02
  10     23      mlp.up_proj                 1.847716e-02
```

主要关注“MSE”列数据，值越大表示这个算子量化前后误差越大，可以尝试放到ignore中重新量化测试，对比精度是否有改善。

### Fisher对角敏感层分析

通过forward\-\>loss\.backward计算grad，对一阶梯度的平方做Fisher对角求和，做敏感层分析。

```bash
# python3 main.py --model /models/Qwen2.5-0.5B-Instruct --analysis fisher
================================================================================
QUANTIZATION ANALYSIS REPORT
================================================================================
Base model:      /models/Qwen2.5-0.5B-Instruct
Analysis:        fisher

-- Fisher Diagonal Sensitivity Ranking (top 10) --
  Rank  Layer                                                             Mean            Max     Params
  -----  -------------------------------------------------------  --------------  --------------  ----------
  1      model.layers.4.self_attn.v_proj                            7.299922e-05    1.724683e-01     114,688
  2      model.layers.3.self_attn.v_proj                            6.914881e-05    6.970391e-01     114,688
  3      model.layers.0.self_attn.v_proj                            4.058762e-05    4.188551e-02     114,688
  4      model.layers.5.self_attn.v_proj                            3.853580e-05    1.132717e-01     114,688
  5      model.layers.8.self_attn.v_proj                            3.256991e-05    1.225300e-01     114,688
  6      model.layers.6.self_attn.v_proj                            3.240306e-05    1.033256e-01     114,688
  7      model.layers.2.self_attn.v_proj                            2.902400e-05    7.308751e-02     114,688
  8      model.layers.1.self_attn.v_proj                            2.054099e-05    1.369999e-02     114,688
  9      model.layers.16.self_attn.v_proj                           2.002842e-05    1.245235e-01     114,688
  10     model.layers.7.self_attn.v_proj                            1.825656e-05    4.843161e-02     114,688
```

主要关注“Mean”列数据，值越大越可疑，可以尝试放到ignore中重新量化测试，对比精度是否有改善。

### 调优步骤建议

当量化后的模型遇到精度问题后，典型如W8A8量化，默认方案为"SmoothQuant \+ GPTQ \+ 数据集校准"，耗时长，测试一次时间代价大。根据以往的量化经验，是否使用数据集校准和复杂量化方案，与量化精度高低没有必然关系，因此，可以先从耗时短的简单方案快速验证，再逐步切换到耗时长的复杂量化方案，建议步骤如下：

- 先使用最简单的model\_free\_ptq模式，不依赖transformers库加载模型，不使用数据集校准，速度非常快，参考命令：

python3 main\.py \-\-model /models/xxx \-\-scheme W8A8 \-\-alg model\_free\_ptq

- 如果model\_free\_ptq量化后模型精度不符合期望，再尝试ptq模式，速度也很快，参考命令：

python3 main\.py \-\-model /models/xxx \-\-scheme W8A8 \-\-alg ptq

- 还可以尝试更新llmc为github上最新main版本，之前遇到过Qwen3\-30B\-A3B的W8A8量化，在release版本llmcompressor\-0\.10\.2\+compressed\-tensors\-0\.14\.0精度异常，安装最新main版本后正常，安装命令参考如下：

```bash
pip uninstall llmcompressor compressed-tensors -y
# 如果docker内网络可以访问github，执行
pip install git+https://github.com/vllm-project/llm-compressor.git
# 或者，如果docker内网络不能访问github，想办法把llm-compressor下载到本地，进入文件夹执行
pip install -i https://mirrors.aliyun.com/pypi/simple -e .
```

- 如果尝试如上方案后精度还不符合期望，需要继续分析模型特性，通过设置ignore解决问题。

# 提交PR

当对某个模型或某一系列模型量化好之后，欢迎通过以下任一方式提交PR，一起完善这个工具。

修改方法：

1. 如果是调用llmc成功量化某个或某一系列模型，参考templates\\quant\_templates\.yaml中第一个例子，添加一个ignore即可。

2. 如果不是调用llmc实现的模型量化，参考其他例子，把自己的python代码集成到文件夹templates即可。

# 常见故障排查 (Trouble shooting)

- 量化过程中遇到"CUDA out of memory"，如何解决？

    - 参考python main\.py \-\-help，可调以下参数，节省GPU显存占用。

        - 带数据集校准场景

            - \-\-pipeline：默认independent按照模块校准，设置为sequential逐层校准，更省显存，但性能也更低。

            - \-\-sequential\-targets：配合\-\-pipeline sequential，默认为按层校准，此参数可设置为Linear，颗粒度比层更小，更省显存，但性能也更低。

            - \-\-offload\-hessians：默认为false，如果改为true，hessians计算过程中会把计算结果offload到cpu，更省显存，但性能也更低。

        - \-\-device\-map默认为auto，会把权重平均加载到多卡显存上，可设置为cpu，把权重加载到cpu上，更省显存，但性能也更低。

        - 其他\-\-batch\-size、\-\-max\-seq\-length越小越省显存。

    - 如果在最后一步model\.save\_pretrained\(save\_dir, save\_compressed=True\)时触发了“CUDA out of memory”，可尝试命令行添加\-\-patch\-for\-llmc。

- 量化速度太慢怎么办？

    - 可以尝试多卡并行计算，启动方式为torchrun \-\-nproc\_per\_node=N main\.py \.\.\.

    - 此功能为llmcompressor新特性，还不稳定，大于100B的模型量化，中途可能会遇到hang住问题，现象是所有GPU核都100%，量化进度卡着不更新。

    - 当模型权重大于100B时，建议N为2或4，设置为8可能会触发系统资源不足的相关错误，如shmem\-rss、vm\.max\_map\_count、cpu ram oom等。

