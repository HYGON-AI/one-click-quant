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
- [MoE-Quant(单机大权重gptq量化)](/utils/MoE-Quant/README.md)
- [PPL评估](#ppl评估)
- [精度调优](/docs/precision-tuning.md)
- [提交PR](#提交pr)
- [常见故障排查](#常见故障排查)

# 简介

基于开源社区组件(llmcompressor、MoE-Quant等)开发的一键量化工具。

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
git clone https://github.com/HYGON-AI/one-click-quant.git
cd one_click_quant
# 下载量化校准数据集
modelscope download --dataset HuggingFaceH4/ultrachat_200k --local_dir datasets/ultrachat_200k
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

# 提交PR

当对某个模型或某一系列模型量化好之后，欢迎通过以下任一方式提交PR，一起完善这个工具。

修改方法：

1. 如果是调用llmc成功量化某个或某一系列模型，参考templates\\quant\_templates\.yaml中第一个例子，添加一个ignore即可。

2. 如果不是调用llmc实现的模型量化，参考其他例子，把自己的python代码集成到文件夹templates即可。

# 常见故障排查

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

