#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""校验 preprocess recipe YAML：transforms 名是否合法、adapter 是否已注册、include 是否齐备。

本脚本自身只用 stdlib + PyYAML，兼容 Python >= 3.8；但仓库主体 `main.py` 需要 Python >= 3.10。
必须在仓库根目录执行：

  python3 ./skills/scripts/check_recipe.py preprocess/recipes/<recipe>
  # <recipe> 例：deepseek_v3.yaml / qwen3_8_flash_next.yaml（以 ls preprocess/recipes/ 为准）

输出以 OK / FAIL 开头，便于脚本判定（退出码 0 = 通过）。
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

KNOWN_TRANSFORMS = ("quarot", "flex_awq_ssz", "flex_smooth_quant")
TRANSFORMS_NEEDING_INCLUDE = ("flex_awq_ssz", "flex_smooth_quant")


def _repo_root() -> Path:
    # 兼容 ./skills/scripts/ 与本地开发时的深层目录；向上找含 main.py 的仓库根
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "main.py").is_file() and (p / "preprocess").is_dir():
            return p
    raise SystemExit("找不到仓库根（需含 main.py 与 preprocess/）")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("需要 PyYAML: {}".format(e))
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(
            "{}: 根节点必须是映射，实际为 {}".format(path, type(data).__name__)
        )
    return data


def _adapter_registry() -> Optional[Dict[str, Any]]:
    root = _repo_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    try:
        from preprocess.adapters import ADAPTER_REGISTRY  # type: ignore

        return dict(ADAPTER_REGISTRY)
    except Exception as e:  # noqa: BLE001
        print(
            "WARN: 无法导入 ADAPTER_REGISTRY（{}）；跳过 adapter 注册检查".format(e)
        )
        return None


def check_recipe(path: Path) -> List[str]:
    errors = []  # type: List[str]
    recipe = _load_yaml(path)

    adapter = recipe.get("adapter") or recipe.get("model")
    if not adapter:
        errors.append("缺少 `adapter:`（或旧字段 `model:`）")
    else:
        registry = _adapter_registry()
        if registry is not None:
            key = str(adapter).strip()
            if key not in registry and key.replace("-", "_") not in registry:
                known = ", ".join(sorted(set(registry)))
                errors.append(
                    "adapter '{}' 未在 ADAPTER_REGISTRY 注册（已注册：{}）".format(
                        adapter, known
                    )
                )

    transforms = recipe.get("transforms")
    if transforms is None:
        errors.append("缺少 `transforms:` 列表")
        transforms = []
    elif not isinstance(transforms, list) or not transforms:
        errors.append("`transforms:` 必须是非空列表")
        transforms = []
    else:
        for name in transforms:
            if name not in KNOWN_TRANSFORMS:
                errors.append(
                    "未知 transform '{}'（可用：{}）".format(
                        name, ", ".join(KNOWN_TRANSFORMS)
                    )
                )

    for name in transforms:
        if name not in TRANSFORMS_NEEDING_INCLUDE:
            continue
        block = recipe.get(name)
        if not isinstance(block, dict):
            errors.append("`{}:` 配置块缺失或不是映射".format(name))
            continue
        include = block.get("include")
        if not include:
            errors.append("`{}.include` 缺失或为空".format(name))
        elif not isinstance(include, list):
            errors.append("`{}.include` 必须是列表".format(name))

    if "quarot" in transforms:
        block = recipe.get("quarot")
        if block is not None and not isinstance(block, dict):
            errors.append("存在 `quarot:` 时其值必须是映射")

    if recipe.get("use_calib_acts") is True:
        errors.append(
            "顶层 use_calib_acts 不生效；请写在各 transform 下"
            "（flex_awq_ssz / flex_smooth_quant）"
        )

    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe", type=Path, help="recipe YAML 路径")
    args = p.parse_args()
    path = args.recipe
    if not path.is_file():
        print("ERROR: 文件不存在: {}".format(path))
        return 2

    errors = check_recipe(path)
    if errors:
        print("FAIL {}".format(path))
        for e in errors:
            print("  - {}".format(e))
        return 1

    print("OK {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
