"""Compare decode+H2D on one real expert matrix, without loading a model."""

import argparse
import json
from pathlib import Path
import time

import torch

from utils.auto_quantize.loading import CheckpointReader, SourceDecoder, decode_mxfp4
from utils.auto_quantize.mxfp4_gpu import decode_mxfp4_gpu
from utils.auto_quantize.staging import PinnedStaging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reader = CheckpointReader(args.model)
    decoder = SourceDecoder(reader, json.loads((args.model / "config.json").read_text()))
    name = next(n for n, spec in sorted(reader.headers.items())
                if ".experts." in n and ".w2." in n and spec["dtype"] == "U8")
    packed = reader.get(name)
    scale = decoder._partner(name)
    staging = PinnedStaging("cuda:0")

    def cpu():
        return decode_mxfp4(packed, scale).to(torch.bfloat16).cuda()

    def gpu():
        value, invalid = decode_mxfp4_gpu(staging.copy(packed), staging.copy(scale), torch.bfloat16)
        if invalid.item():
            raise ValueError("Invalid decoded source weight")
        return value

    expected = cpu()
    actual = gpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    results = {}
    for label, run in (("cpu_decode_then_bf16_h2d", cpu), ("packed_h2d_then_gpu_decode", gpu)):
        elapsed = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = run()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
            del value
        results[label] = {"seconds": elapsed, "mean_seconds": sum(elapsed) / len(elapsed)}
    results.update(source=name, logical_shape=list(actual.shape), bit_exact=True,
                   scope="one matrix, warm file cache, decode and transfer only")
    print(json.dumps(results, indent=2))
    staging.close()
    reader.close()


if __name__ == "__main__":
    main()
