### QMiniMax-M2.5-W8A8-INT8

```bash
vllm v0.19.0 or
http://10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.12-ubuntu22.04-dtk26041-py3.10-scalex40-sf_b020-20260810-0214
pip install -i https://mirrors.aliyun.com/pypi/simple llmcompressor==0.11.0
pip install -i https://mirrors.aliyun.com/pypi/simple numpy==1.25.0
python3.10 -m pip install -U "huggingface-hub>=1.5.0,<2.0" "kernels>=0.12.0,<0.13"
python3.10 -m pip install "huggingface-hub>=0.34.0,<1.0"
python3 main.py --model /models/MiniMax-M2.5 --scheme FP8_TO_BF16 --save-dir /models/MiniMax-M2.5-BF16
python main.py --model /models/MiniMax-M2.5-BF16 --scheme W8A8 --alg ptq --needs-data false --ignore 'lm_head,re:.*moe.gate+,re:.*moe.e_score_correction_bias+' --save-dir /models/MiniMax-M2.5-W8A8
```

