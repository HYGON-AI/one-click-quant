#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""只读校验**量化产物**目录，不加载张量、不依赖 torch。

适用范围：main.py PTQ / slimquant / MoE-Quant 打包 / examples 量化脚本产物。
不适用：--alg preprocess 浮点目录（无量化配置块，必然 FAIL；见 10-ext-preprocess.md）、
        仅改 config 的中间手术步、未写完的半成品目录。

检查项：
  1. config.json 存在且可解析
  2. quantization_config 或 compression_config 至少有一个（顶层或 text_config 内）
  3. 权重文件存在；有 index 时 weight_map 指向的分片必须全部存在；
     多分片却缺 index 视为错误（下游无法确定 weight_map）
  4. 打印 ignore 列表（便于发现 --ignore 误覆盖）
  5. 抽样 safetensors header（只读前 8 字节 + JSON header），确认权重 dtype 不是全 bf16

用法：
  python3 ./skills/scripts/check_artifact.py <产物目录>
  python3 ./skills/scripts/check_artifact.py <产物目录> --expect-quant-method slimquant_w4a8

输出以 OK / FAIL 开头；退出码 0 = 通过，1 = 不通过，2 = 用法错误。
仅用 stdlib，兼容 Python >= 3.6（仓库主体需要 >= 3.10，本脚本不需要）。
"""

import argparse
import json
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple

QUANT_CONFIG_KEYS = ("quantization_config", "compression_config")

# safetensors 的 dtype 字符串 -> 是否属于"量化后权重"
QUANT_DTYPES = frozenset(
    ["I8", "U8", "I32", "F8_E4M3", "F8_E5M2", "F8_E4M3FN", "F8_E5M2FNUZ"]
)
FLOAT_DTYPES = frozenset(["BF16", "F16", "F32", "F64"])

# 名字里带这些后缀的是量化辅助张量，不算"权重"
AUX_SUFFIXES = (
    "_scale_inv",
    "_scale",
    "_zero_point",
    "weight_shape",
    "weight_global_scale",
    "weight_scale_inv",
)


def _is_aux(name):
    # type: (str) -> bool
    return name.endswith(AUX_SUFFIXES)


def read_header(path):
    # type: (str) -> Dict[str, dict]
    """只读 safetensors header（不 mmap、不加载任何张量）。"""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("文件过短，不是合法 safetensors")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > (1 << 30):  # 防御性上限 1GiB
            raise ValueError("header 长度异常: {}".format(n))
        blob = f.read(n)
    return json.loads(blob.decode("utf-8"))


def _quant_config(cfg):
    # type: (dict) -> Tuple[Optional[str], Optional[str], Optional[dict]]
    """返回 (来源描述, 配置键, 配置内容)；顶层优先，其次 text_config。"""
    holders = [("顶层", cfg), ("text_config", cfg.get("text_config"))]
    for scope, holder in holders:
        if not isinstance(holder, dict):
            continue
        for key in QUANT_CONFIG_KEYS:
            value = holder.get(key)
            if isinstance(value, dict):
                return scope, key, value
    return None, None, None


def check(directory, expect_quant_method, sample_shards):
    # type: (str, Optional[str], int) -> List[str]
    errors = []  # type: List[str]
    notes = []  # type: List[str]
    d = os.path.abspath(directory)

    # ---- 1. config.json ----
    cfg_path = os.path.join(d, "config.json")
    if not os.path.isfile(cfg_path):
        return ["缺少 config.json"]
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:  # noqa: BLE001
        return ["config.json 无法解析: {}".format(e)]
    if not isinstance(cfg, dict):
        return ["config.json 根节点不是对象"]

    # ---- 2. 量化配置块 ----
    scope, qkey, qcfg = _quant_config(cfg)
    if qcfg is None:
        errors.append(
            "config.json 缺量化配置块（{} 都没有，也已查 text_config）".format(
                " / ".join(QUANT_CONFIG_KEYS)
            )
        )
        return errors

    quant_method = qcfg.get("quant_method")
    qformat = qcfg.get("format")
    notes.append(
        "quant_config: {}.{}  quant_method={}  format={}".format(
            scope, qkey, quant_method, qformat
        )
    )

    if expect_quant_method and quant_method != expect_quant_method:
        errors.append(
            "quant_method={!r}，期望 {!r}".format(quant_method, expect_quant_method)
        )

    # ---- 4. ignore 列表 ----
    ignore = qcfg.get("ignore")
    if ignore is None:
        ignore = qcfg.get("modules_to_not_convert")
    if ignore:
        notes.append(
            "ignore[{}]={}".format(len(ignore), json.dumps(ignore, ensure_ascii=False))
        )
        joined = " ".join(str(x) for x in ignore)
        if "re:" not in joined:
            notes.append(
                "提示: ignore 全为裸名字，是整名相等匹配（不是子串包含）。"
                "要复用为 --ignore 时：若源路径是 slimquant，写盘可能已清洗掉 re: 包装，"
                "需自己补回 re:.*…$；其它路径（如 compressed-tensors）本来就可能是裸名"
                "（见 03-quant-runbook.md 第 5 节）"
            )
    else:
        notes.append("ignore: <空 —— 确认这是预期，全量化时才会为空>")

    mixed = qcfg.get("mixed_precision")
    if isinstance(mixed, dict) and mixed:
        notes.append("mixed_precision 分组: {}".format(", ".join(sorted(mixed))))

    # ---- 3. 权重与 index ----
    idx_path = os.path.join(d, "model.safetensors.index.json")
    shards = []  # type: List[str]
    if os.path.isfile(idx_path):
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                wm = json.load(f)["weight_map"]
        except Exception as e:  # noqa: BLE001
            errors.append("model.safetensors.index.json 无法解析: {}".format(e))
            return errors
        if not isinstance(wm, dict) or not wm:
            errors.append("index.weight_map 为空")
        else:
            names = sorted(set(wm.values()))
            missing = [
                s for s in names if not os.path.isfile(os.path.join(d, s))
            ]
            notes.append(
                "layout: sharded  tensors={}  shards={}".format(len(wm), len(names))
            )
            if missing:
                errors.append("index 指向的分片缺失: {}".format(", ".join(missing[:5])))
            shards = names
    else:
        cands = sorted(
            fn for fn in os.listdir(d) if fn.endswith(".safetensors")
        )
        if not cands:
            errors.append("没有 model.safetensors.index.json，也没有任何 *.safetensors")
        elif len(cands) > 1:
            # 多分片但缺索引：下游无法确定 weight_map，属实质错误
            errors.append(
                "有 {} 个 *.safetensors 但缺 model.safetensors.index.json：{}".format(
                    len(cands), ", ".join(cands[:5])
                )
            )
            shards = cands
        else:
            notes.append("layout: single   files={}".format(cands))
            shards = cands
            if "model.safetensors" not in cands:
                notes.append(
                    "WARN: 单分片但文件名不是 model.safetensors —— "
                    "slimquant_ptq 读取这种输入会 ValueError"
                )

    # ---- 5. 抽样 header 看 dtype ----
    picks = shards[: max(0, sample_shards)]
    n_quant = 0
    n_float = 0
    seen_dtypes = {}  # type: Dict[str, int]
    has_scale_inv = False
    scanned = 0
    for shard in picks:
        path = os.path.join(d, shard)
        if not os.path.isfile(path):
            continue
        try:
            header = read_header(path)
        except Exception as e:  # noqa: BLE001
            errors.append("{} header 无法解析: {}".format(shard, e))
            continue
        scanned += 1
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = (meta or {}).get("dtype")
            if not dtype:
                continue
            if name.endswith("_scale_inv"):
                has_scale_inv = True
            if _is_aux(name):
                continue
            seen_dtypes[dtype] = seen_dtypes.get(dtype, 0) + 1
            if dtype in QUANT_DTYPES:
                n_quant += 1
            elif dtype in FLOAT_DTYPES:
                n_float += 1

    total = n_quant + n_float
    if not picks:
        notes.append("提示: --sample-shards 0，跳过 dtype 抽样检查")
    elif scanned == 0:
        errors.append("没有可读的 safetensors 分片，无法校验 dtype")
    elif total == 0:
        errors.append("抽样分片里没有找到权重张量")
    else:
        ratio = float(n_quant) / total
        notes.append(
            "dtype 抽样 {} 分片 / {} 个权重张量: 量化 {} ({:.1%}), 浮点 {}  {}".format(
                scanned, total, n_quant, ratio, n_float, seen_dtypes
            )
        )
        if n_quant == 0:
            errors.append("权重 dtype 全是浮点（没有 int8/fp8/packed），量化可能没生效")
        elif ratio < 0.01:
            notes.append("WARN: 量化张量占比 <1%，确认 ignore/targets 是否过宽")

    if has_scale_inv:
        notes.append(
            "WARN: 存在 *_scale_inv（block FP8 痕迹）—— "
            "若产物应是逐通道量化，说明源权重没解包干净"
        )
    if picks and len(shards) > len(picks):
        notes.append(
            "提示: 只抽样了前 {} 个分片（--sample-shards 可调）".format(len(picks))
        )

    for line in notes:
        print("  " + line)
    return errors


def main():
    # type: () -> int
    p = argparse.ArgumentParser(description="只读校验量化产物目录")
    p.add_argument("artifact", help="量化产物目录")
    p.add_argument(
        "--expect-quant-method",
        default=None,
        help="期望的 quant_method，如 slimquant_w4a8 / compressed-tensors",
    )
    p.add_argument(
        "--sample-shards",
        type=int,
        default=2,
        help="抽样多少个 safetensors 分片读 header（默认 2，0 表示跳过 dtype 检查）",
    )
    args = p.parse_args()

    if not os.path.isdir(args.artifact):
        print("ERROR: 不是目录: {}".format(args.artifact))
        return 2

    errors = check(args.artifact, args.expect_quant_method, args.sample_shards)
    if errors:
        print("FAIL {}".format(args.artifact))
        for e in errors:
            print("  - {}".format(e))
        return 1

    print("OK {}".format(args.artifact))
    return 0


if __name__ == "__main__":
    sys.exit(main())
