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

通过forward\->loss\.backward计算grad，对一阶梯度的平方做Fisher对角求和，做敏感层分析。

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

当量化后的模型遇到精度问题后，典型如W8A8量化，默认方案为"SmoothQuant + GPTQ + 数据集校准"，耗时长，测试一次时间代价大。根据以往的量化经验，是否使用数据集校准和复杂量化方案，与量化精度高低没有必然关系，因此，可以先从耗时短的简单方案快速验证，再逐步切换到耗时长的复杂量化方案，建议步骤如下：

- 先使用最简单的model_free_ptq模式，不依赖transformers库加载模型，不使用数据集校准，速度非常快，参考命令：

python3 main.py --model /models/xxx --scheme W8A8 --alg model_free_ptq

- 如果model_free_ptq量化后模型精度不符合期望，再尝试ptq模式，速度也很快，参考命令：

python3 main.py --model /models/xxx --scheme W8A8 --alg ptq

- 还可以尝试更新llmc为github上最新main版本，之前遇到过Qwen3-30B-A3B的W8A8量化，在release版本llmcompressor-0.10.2+compressed-tensors-0.14.0精度异常，安装最新main版本后正常，安装命令参考如下：

```bash
pip uninstall llmcompressor compressed-tensors -y
# 如果docker内网络可以访问github，执行
pip install git+https://github.com/vllm-project/llm-compressor.git
# 或者，如果docker内网络不能访问github，想办法把llm-compressor下载到本地，进入文件夹执行
pip install -i https://mirrors.aliyun.com/pypi/simple -e .
```

- 如果尝试如上方案后精度还不符合期望，需要继续分析模型特性，通过设置ignore解决问题。
