# SPDX-License-Identifier: Apache-2.0
"""
Hy4-flash BF16 -> FP8 per-channel (W8A8) 量化脚本。

输入 ``Hy4-flash-testing``(BF16),输出 per-output-channel 的 FP8 checkpoint。
**量化范围**与 ``Hy4-flash-FP8-testing`` 逐层对齐(依据两份 checkpoint 的分片头对比):

  F8_E4M3(新增 scale):
    * ``model.layers.{1..L-1}.mlp.experts.{gate_up_proj,down_proj}``    融合 3D [E,out,in]
    * ``model.layers.{1..L-1}.self_attn.{q_b_proj,o_proj,linear_gate}.weight``
    * ``model.dfly_layers.{i}.self_attn.{q_b_proj,o_proj,linear_gate}.weight``
    * ``model.dfly_layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight``
  其余张量原样保留(layer 0 全部、q_a/kv_a/kv_b、shared_experts、norm、embed、lm_head、
  F32 的 hc_*/compressed_indexer/learnable_sink_param/e_score_correction_bias、
  I32 的 engram lookup_table、dfly 的 confidence_head/hidden_correction/mtp_fusion、vit.*)。

与 FP8-testing 的**差异(有意为之)**:
  * FP8-testing 是 128x128 分块量化,scale 为 BF16 ``*_scale_inv`` [out/128,in/128];
    本脚本是 per-output-channel,scale 为 FP32:
      - 2D 权重: ``x.weight_scale``    [out, 1]
      - 融合专家: ``experts.gate_up_proj_scale`` [E, 2I, 1],``experts.down_proj_scale`` [E, H, 1]
  * 激活声明为动态 per-token FP8(compressed-tensors float-quantized)。
  * 专家保持融合 3D,不拆成逐专家 2D。

已知冲突: vLLM 的 hy_v4_flash ``validate_runtime`` 对 compressed-tensors 会调用
``_validate_block_fp8_config``,只接受 128x128 权重块 + group-128 动态激活;
本脚本产出的 per-channel 模型在未打补丁的 vLLM 里会被拒绝加载。

示例:
  python quantize_hy4_flash_fp8_channel.py \\
      --input-dir /models/Hy4-flash-testing --output-dir /models/Hy4-flash-FP8-channel

依赖: torch、safetensors、tqdm。输入必须是 BF16/FP16 checkpoint。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
EXPERT_CHUNK = 32  # 融合专家每次上 GPU 的专家数,控制显存

EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
ATTN_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.(?:q_b_proj|o_proj|linear_gate)\.weight$"
)
DFLY_RE = re.compile(
    r"^model\.dfly_layers\.(\d+)\.(?:self_attn\.(?:q_b_proj|o_proj|linear_gate)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


def text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def is_expert(name: str) -> bool:
    return EXPERT_RE.match(name) is not None


def is_2d_target(name: str) -> bool:
    """layer 0 不量化;主干只量化 attention 的三个投影,dfly 层量化 attention 与 dense MLP。"""
    m = ATTN_RE.match(name)
    if m:
        return int(m.group(1)) >= 1
    return DFLY_RE.match(name) is not None


def is_fp8_target(name: str) -> bool:
    m = EXPERT_RE.match(name)
    if m:
        return int(m.group(1)) >= 1
    return is_2d_target(name)


def scale_name(name: str) -> str:
    """2D: x.weight -> x.weight_scale;融合专家: experts.gate_up_proj -> experts.gate_up_proj_scale。"""
    return name[: -len(".weight")] + ".weight_scale" if name.endswith(".weight") else name + "_scale"


def quantize_channelwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 FP8 E4M3 per-output-channel 量化,沿最后一维取 amax。

    2D [out,in] -> scale [out,1];3D [E,out,in] -> scale [E,out,1]。
    """
    if weight.ndim not in (2, 3):
        raise ValueError(f"expect 2-D or 3-D weight, got {tuple(weight.shape)}")
    w = weight.float()
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX
    q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.float().contiguous()


def quantize_fused_expert(tensor: torch.Tensor, device: torch.device):
    """融合 3D 专家分块上 GPU 量化,结果留在 CPU 并按专家维拼回。"""
    qs, ss = [], []
    for start in range(0, tensor.shape[0], EXPERT_CHUNK):
        q, s = quantize_channelwise(tensor[start : start + EXPERT_CHUNK].to(device))
        qs.append(q.cpu())
        ss.append(s.cpu())
    return torch.cat(qs).contiguous(), torch.cat(ss).contiguous()


def convert_one_shard(input_path: str, output_path: str, stats: dict, device: torch.device) -> None:
    out: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if t.dtype == torch.float8_e4m3fn:
                raise ValueError(f"{name}: source is already FP8; expected BF16 input")
            if is_fp8_target(name):
                if t.dtype not in FLOAT_DTYPES:
                    raise ValueError(f"{name}: expected float source, got {t.dtype}")
                if is_expert(name):
                    if t.ndim != 3:
                        raise ValueError(f"{name}: fused expert must be 3-D, got {tuple(t.shape)}")
                    q, s = quantize_fused_expert(t, device)
                    stats["fused_expert_fp8"] += 1
                else:
                    if t.ndim != 2:
                        raise ValueError(f"{name}: expected 2-D, got {tuple(t.shape)}")
                    q, s = quantize_channelwise(t.to(device))
                    q, s = q.cpu(), s.cpu()
                    stats["fp8_weights"] += 1
                out[name] = q
                out[scale_name(name)] = s
            else:
                out[name] = t
                stats["kept"] += 1
    save_file(out, output_path, metadata={"format": "pt"})


def fp8_weights_cfg() -> dict:
    return {"actorder": None, "block_structure": None, "dynamic": False, "group_size": None,
            "num_bits": 8, "observer": "minmax", "observer_kwargs": {}, "strategy": "channel",
            "symmetric": True, "type": "float"}


def fp8_input_cfg() -> dict:
    return {"actorder": None, "block_structure": None, "dynamic": True, "group_size": None,
            "num_bits": 8, "observer": None, "observer_kwargs": {}, "strategy": "token",
            "symmetric": True, "type": "float"}


def build_quantization_config(cfg: dict) -> dict:
    t = text_cfg(cfg)
    layers = int(t["num_hidden_layers"])
    backbone = "|".join(str(i) for i in range(1, layers))
    # vLLM 的 MoE 方法会用 "<experts prefix>.<id>.<proj>" 解析 scheme,Flash 的参数在 routed_experts 下
    expert = rf"re:.*layers\.(?:{backbone})\.mlp\.experts\.(?:routed_experts\.)?\d+\.(?:gate_proj|up_proj|down_proj)$"
    attn = rf"re:.*layers\.(?:{backbone})\.self_attn\.(?:q_b_proj|o_proj|linear_gate)$"
    dfly_attn = r"re:.*dfly_layers\.\d+\.self_attn\.(?:q_b_proj|o_proj|linear_gate)$"
    dfly_mlp = r"re:.*dfly_layers\.\d+\.mlp\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$"
    common = {"input_activations": fp8_input_cfg(), "output_activations": None,
              "weights": fp8_weights_cfg(), "format": "float-quantized"}
    ignore = [
        "lm_head", r"re:.*embed_tokens.*", r"re:.*vit\..*", r"re:.*visual\..*",
        r"re:.*(?<!dfly_)layers\.0\..*",  # 主干 layer 0 整层不量化;不能误伤 dfly_layers.0
        r"re:.*mlp\.gate$", r"re:.*mlp\.gate\..*", r"re:.*hc_.*",
        r"re:.*shared_experts\..*", r"re:.*engram_layer\..*", r"re:.*compressed_indexer\..*",
        r"re:.*self_attn\.indexer\..*",  # sglang 把 compressed_indexer 改名为 indexer,按运行时名匹配
        r"re:.*(?:q_a_proj|kv_a_proj_with_mqa|kv_b_proj)$", r"re:.*routed_expert_(?:up|down)_proj$",
        r"re:.*dfly_layers\.(?:confidence_head|hidden_correction|mtp_fusion)\..*",
        r"re:.*dfly_layers\.\d+\.self_attn\.(?:q_a_proj|kv_a_proj_with_mqa|kv_b_proj)$",
    ]
    return {"config_groups": {"group_fp8": {"targets": [expert, attn, dfly_attn, dfly_mlp], **common}},
            "format": "float-quantized", "ignore": ignore, "kv_cache_scheme": None,
            "quant_method": "compressed-tensors", "quantization_status": "compressed"}


def self_check_config(cfg: dict) -> None:
    """用代表性模块名检查 config 的 targets/ignore 与本脚本写入的 dtype 一致。"""
    qc = build_quantization_config(cfg)
    layers = int(text_cfg(cfg)["num_hidden_layers"])
    targets = qc["config_groups"]["group_fp8"]["targets"]

    def ignored(n: str) -> bool:
        return any(re.match(p[3:], n) if p.startswith("re:") else p == n for p in qc["ignore"])

    def quantized(n: str) -> bool:
        return not ignored(n) and any(re.match(t[3:], n) for t in targets)

    expect_true = [f"model.layers.{i}.self_attn.{m}" for i in (1, layers - 1) for m in ("q_b_proj", "o_proj", "linear_gate")]
    expect_true += [f"model.layers.{i}.mlp.experts.7.{p}" for i in (1, layers - 1) for p in ("gate_proj", "up_proj", "down_proj")]
    # dfly 的第 0 层必须被量化(曾被 layers\.0 的 ignore 误伤),同时检查首尾层
    for d in (0, 2, 4):
        expect_true += [f"model.dfly_layers.{d}.self_attn.{m}" for m in ("q_b_proj", "o_proj", "linear_gate")]
        expect_true += [f"model.dfly_layers.{d}.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")]
    expect_false = [f"model.layers.0.self_attn.{m}" for m in ("q_b_proj", "o_proj", "linear_gate")]
    expect_false += ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.shared_experts.down_proj",
                     "model.layers.1.self_attn.q_a_proj", "model.layers.1.self_attn.kv_b_proj",
                     "model.dfly_layers.2.self_attn.kv_b_proj", "model.dfly_layers.hidden_correction.down_proj",
                     "vit.layers.0.self_attn.o_proj", "lm_head"]
    bad = [n for n in expect_true if not quantized(n)] + [n for n in expect_false if quantized(n)]
    if bad:
        raise AssertionError(f"config/targets mismatch: {bad}")
    # 写入侧与配置侧对同一批张量名的判断必须一致
    probes = {"model.layers.1.self_attn.q_b_proj.weight": True, "model.layers.0.self_attn.q_b_proj.weight": False,
              "model.layers.1.mlp.experts.gate_up_proj": True, "model.layers.0.mlp.gate_proj.weight": False,
              "model.dfly_layers.0.mlp.down_proj.weight": True, "model.dfly_layers.0.self_attn.kv_b_proj.weight": False,
              "model.layers.1.mlp.shared_experts.up_proj.weight": False}
    for n, want in probes.items():
        if is_fp8_target(n) != want:
            raise AssertionError(f"write side disagrees for {n}")
    print("Self-check passed: config 与写入的 dtype 一致。")


def rebuild_index(output_dir: str, model_index: dict) -> None:
    weight_map, total = {}, 0
    for path in sorted(glob(os.path.join(output_dir, "*.safetensors"))):
        with open(path, "rb") as sf:
            header = json.loads(sf.read(int.from_bytes(sf.read(8), "little")))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            weight_map[key] = os.path.basename(path)
            total += meta["data_offsets"][1] - meta["data_offsets"][0]
    model_index["weight_map"] = weight_map
    model_index["metadata"] = {"total_size": total}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)


def verify_conversion(input_map: dict, output_map: dict, shards: set[str], full_run: bool) -> None:
    """输入键必须全部保留;新增键必须恰好是被量化张量的 scale。"""
    scoped = {n for n, s in input_map.items() if s in shards}
    missing = sorted(scoped - set(output_map))
    if missing:
        raise RuntimeError(f"Self-check FAILED: {len(missing)} tensor(s) missing, e.g. {missing[:5]}")
    expected = {scale_name(n) for n in scoped if is_fp8_target(n)}
    extra = set(output_map) - scoped
    if extra != expected:
        raise RuntimeError(f"Self-check FAILED: scale keys differ, unexpected={sorted(extra - expected)[:5]}, "
                           f"missing={sorted(expected - extra)[:5]}")
    print(f"Self-check passed: 保留全部 {len(scoped)} 个输入键,新增 {len(extra)} 个 scale。")
    if full_run:
        print("提示: Hy4-flash-FP8-testing 新增 235 个 scale_inv(主干 41x5 加 dfly 5x6);"
              f"本次新增 {len(extra)} 个,数量应一致。")


def empty_stats() -> dict[str, int]:
    return {k: 0 for k in ("fused_expert_fp8", "fp8_weights", "kept")}


def resolve_devices(kind: str, gpu_ids: str | None, workers: int | None, n_files: int):
    cuda_ok = torch.cuda.is_available()
    if kind == "cpu" or (kind == "auto" and not cuda_ok):
        return [torch.device("cpu")], 1
    if not cuda_ok:
        raise RuntimeError("--device cuda specified but CUDA is unavailable")
    ids = [int(x) for x in gpu_ids.split(",") if x.strip()] if gpu_ids else list(range(torch.cuda.device_count()))
    if not ids:
        raise RuntimeError("no CUDA device selected")
    devices = [torch.device(f"cuda:{i}") for i in ids]
    eff = len(devices) if workers is None else max(1, workers)
    return devices, min(eff, max(1, n_files))


def worker_convert_shard(task: tuple) -> dict[str, int]:
    input_path, output_path, gpu_id, num_threads = task
    torch.set_num_threads(max(1, num_threads))
    torch.cuda.set_device(gpu_id)
    stats = empty_stats()
    convert_one_shard(input_path, output_path, stats, torch.device(f"cuda:{gpu_id}"))
    return stats


def main() -> None:
    p = ArgumentParser(description="Convert Hy4-flash BF16 to FP8 per-channel (W8A8).",
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", default="/models/Hy4-flash-testing")
    p.add_argument("--output-dir", default="/models/Hy4-flash-FP8-channel")
    p.add_argument("--start-file", type=int, default=0)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gpu-ids", type=str, default=None, help="e.g. '0,1,3'; 默认所有可见 GPU")
    p.add_argument("--workers", type=int, default=None, help="并行进程数(每个进程整片读入内存)")
    p.add_argument("--num-threads", type=int, default=8)
    args = p.parse_args()
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must differ")

    with open(os.path.join(args.input_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    print("WARNING: vLLM hy_v4_flash 的 validate_runtime 只接受 128x128 块 FP8,"
          "per-channel 输出在未打补丁的 vLLM 中会被拒绝加载。")
    self_check_config(cfg)

    files = sorted(glob(os.path.join(args.input_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {args.input_dir}")
    if not 0 <= args.start_file < len(files):
        raise ValueError(f"--start-file must be in [0, {len(files) - 1}]")
    full_run = args.start_file == 0 and args.limit_files is None
    files = files[args.start_file:]
    if args.limit_files is not None:
        files = files[: args.limit_files]
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(args.num_threads)
    devices, n_workers = resolve_devices(args.device, args.gpu_ids, args.workers, len(files))
    stats = empty_stats()
    if n_workers <= 1 or len(devices) == 1:
        for path in tqdm(files, desc=f"Converting on {devices[0]}"):
            convert_one_shard(path, os.path.join(args.output_dir, os.path.basename(path)), stats, devices[0])
    else:
        import torch.multiprocessing as mp

        tasks = [(f, os.path.join(args.output_dir, os.path.basename(f)), devices[i % len(devices)].index,
                  args.num_threads) for i, f in enumerate(files)]
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            for per in tqdm(pool.imap_unordered(worker_convert_shard, tasks), total=len(tasks),
                            desc=f"Converting on {len(devices)} GPU(s) ({n_workers} workers)"):
                for k, v in per.items():
                    stats[k] += v

    for fn in sorted(os.listdir(args.input_dir)):
        src = os.path.join(args.input_dir, fn)
        if (os.path.isfile(src) and not fn.endswith((".safetensors", ".bin"))
                and fn not in ("config.json", "model.safetensors.index.json")):
            shutil.copyfile(src, os.path.join(args.output_dir, fn))
    with open(os.path.join(args.input_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        model_index = json.load(f)
    input_map = dict(model_index["weight_map"])
    rebuild_index(args.output_dir, model_index)
    verify_conversion(input_map, model_index["weight_map"], {os.path.basename(x) for x in files}, full_run)

    # 与 FP8-testing 一致: quantization_config 放在 text_config 下(无 text_config 时放顶层)
    qc = build_quantization_config(cfg)
    (cfg["text_config"] if "text_config" in cfg else cfg)["quantization_config"] = qc
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, sort_keys=True)
    summary = {"input_dir": args.input_dir, "output_dir": args.output_dir, "stats": stats,
               "layout": {"quantized": "F8_E4M3 per-output-channel, FP32 scale, dynamic per-token FP8 activations",
                          "experts": "fused 3-D, scale key = <name>_scale [E,out,1]",
                          "others": "unchanged (aligned with Hy4-flash-FP8-testing quantization scope)"}}
    with open(os.path.join(args.output_dir, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
