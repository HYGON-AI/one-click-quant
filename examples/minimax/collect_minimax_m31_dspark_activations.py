# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Collect calibration activations from an isolated MiniMax M3.1 server."""

# Install aiohttp, pyarrow and transformers in the client environment.
# Start an isolated SGLang server with the W4A8 target, the exported BF16 draft
# and --speculative-algorithm DSPARK. Pass the BF16 draft via
# --speculative-draft-model-path and set --cuda-graph-backend-prefill disabled
# and --cuda-graph-backend-decode disabled. Use --url if the server listens
# somewhere other than http://127.0.0.1:11010.
#
# In the launcher, prepend the absolute dspark_calibration directory to
# PYTHONPATH after assigning the runtime paths. Set M31_CALIB_OUTPUT to the
# client's --output directory. The server and client must share this directory.
#
# The reference server used DP8/EP8, context length 163840, chunk size 4096,
# max total tokens 524288 and static memory fraction 0.90. The client routes
# 128 WikiText training pages across DP ranks and saves the activation files
# and calibration_manifest.json for the Smooth INT8 exporter.

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import aiohttp
import pyarrow as pa
import torch
from transformers import AutoTokenizer


def load_inputs(dataset, tokenizer_path):
    with pa.memory_map(str(dataset), "r") as stream:
        pages = pa.ipc.open_stream(stream).read_all().column("page")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    inputs = []
    for page in pages:
        ids = tokenizer.encode(page.as_py(), add_special_tokens=False)
        if len(ids) >= 256:
            inputs.append(ids[:2048])
        if len(inputs) == 128:
            return inputs
    raise ValueError(
        "WikiText training data must supply 128 pages with at least 256 tokens each"
    )


async def collect(url, inputs, dp_size, output):
    limit = asyncio.Semaphore(16)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=900)
    ) as session:

        async def generate(index, ids, tokens):
            body = {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "ignore_eos": True,
                },
                "routed_dp_rank": index % dp_size,
            }
            async with (
                limit,
                session.post(url.rstrip("/") + "/generate", json=body) as response,
            ):
                response.raise_for_status()
                metadata = (await response.json())["meta_info"]
                if metadata["completion_tokens"] != tokens:
                    raise ValueError(
                        f"Unexpected completion length on request {index}: {metadata}"
                    )

        await asyncio.gather(
            *(generate(index, ids, 256) for index, ids in enumerate(inputs))
        )
        (output / "dump.trigger").touch(exist_ok=False)
        await asyncio.gather(
            *(generate(rank, inputs[rank][:256], 16) for rank in range(dp_size))
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Local WikiText train Arrow stream with a page column",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Trusted local MiniMax M3.1 checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Shared activation directory used by M31_CALIB_OUTPUT",
    )
    parser.add_argument("--url", default="http://127.0.0.1:11010")
    parser.add_argument("--dp-size", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.dp_size <= 128:
        parser.error("--dp-size must be in [1, 128]")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("--output must be empty before collecting")
    inputs = load_inputs(args.dataset, args.tokenizer)
    manifest = dict(
        dataset=str(args.dataset.resolve()),
        split="train",
        count=len(inputs),
        max_input_tokens=2048,
        output_tokens=256,
        dp_size=args.dp_size,
        input_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
        evaluation_gsm_used=False,
    )
    (args.output / "collect.enabled").touch(exist_ok=False)
    asyncio.run(collect(args.url, inputs, args.dp_size, args.output))
    paths = sorted(args.output.glob("activations_*.pt"))
    if len(paths) != args.dp_size:
        raise ValueError(f"Expected {args.dp_size} activation files, got {len(paths)}")
    for path in paths:
        entries = torch.load(path, map_location="cpu", weights_only=True)
        if len(entries) != 11 or any(
            entry["rows"] <= 0
            or not torch.isfinite(entry["amax"]).all()
            or not torch.isfinite(entry["samples"]).all()
            for entry in entries.values()
        ):
            raise ValueError(f"Incomplete or invalid calibration: {path}")
    (args.output / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Collected {len(paths)} rank files in {args.output}")


if __name__ == "__main__":
    main()
