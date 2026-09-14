# Build context: utils/MoE-Quant. BASE_IMAGE must contain the verified Hy4
# SGLang runtime and CUDA/PyTorch dependencies; see README.hy4.md.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY src/hy4 /opt/moe-quant/src/hy4
COPY inference/hy4/sitecustomize.py /opt/hy4-bootstrap/sitecustomize.py
COPY inference/hy4/serve.sh /opt/hy4-bootstrap/serve.sh
# Do not inherit the historical /opt/hy4-gptq sitecustomize/import directory.
ENV PYTHONPATH=/opt/hy4-bootstrap:/opt/moe-quant:/opt/hy4-sglang/python
RUN python3 -m compileall -q /opt/moe-quant/src/hy4
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
