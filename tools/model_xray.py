# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import json
import argparse
from pathlib import Path
from safetensors import safe_open

def extract_single_file(safetensors_path, output_json_path):
    metadata = {}
    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                metadata[key] = {
                    "dtype": str(tensor.dtype).replace('torch.', ''),
                    "shape": list(tensor.shape)
                }
        with open(output_json_path, "w") as jf:
            json.dump(metadata, jf, indent=2)
        print(f"✅ 已处理: {safetensors_path.name} -> {output_json_path.name}")
        return True
    except Exception as e:
        print(f"❌ 处理 {safetensors_path} 时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="批量提取 safetensors 分片的元数据")
    parser.add_argument("--input_dir", type=str, help="包含 .safetensors 文件的目录")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="元数据 JSON 输出目录（默认为输入目录）")
    parser.add_argument("--suffix", type=str, default=".metadata.json",
                        help="生成的 JSON 文件后缀（默认 .metadata.json）")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        print("❌ 输入路径不是目录")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_path
    output_dir.mkdir(parents=True, exist_ok=True)

    safetensors_files = list(input_path.glob("*.safetensors"))
    if not safetensors_files:
        print("⚠️ 未找到任何 .safetensors 文件")
        return

    print(f"📁 找到 {len(safetensors_files)} 个分片文件")
    for sf in safetensors_files:
        json_name = sf.stem + args.suffix
        json_path = output_dir / json_name
        extract_single_file(sf, json_path)

if __name__ == "__main__":
    main()