#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""根据两侧 word_perplexity 打印 ΔPPL / ΔNLL，不跑评测、不设阈值。

定义（与 skills/references/04-verify-eval.md §4.2 一致）：
  PPL   = lm_eval wikitext 的 word_perplexity
  ΔPPL  = PPL_q - PPL_b
  ΔNLL  = ln(PPL_q) - ln(PPL_b)   # nats

用法：
  python3 ./skills/scripts/report_ppl_delta.py --baseline-ppl 18.35 --quant-ppl 19.10

退出码：0 = 打印成功；2 = 用法 / 输入非法（非正数等）。
不对 Δ 做 PASS/FAIL。
"""

from __future__ import print_function

import argparse
import math
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Print ΔPPL and ΔNLL from baseline/quant word_perplexity (no thresholds)."
    )
    p.add_argument(
        "--baseline-ppl",
        type=float,
        required=True,
        help="Baseline word_perplexity (PPL_b)",
    )
    p.add_argument(
        "--quant-ppl",
        type=float,
        required=True,
        help="Quantized model word_perplexity (PPL_q)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ppl_b = args.baseline_ppl
    ppl_q = args.quant_ppl

    if not (math.isfinite(ppl_b) and math.isfinite(ppl_q)):
        print("FAIL: PPL values must be finite", file=sys.stderr)
        return 2
    if ppl_b <= 0 or ppl_q <= 0:
        print("FAIL: PPL values must be > 0 (got PPL_b=%s PPL_q=%s)" % (ppl_b, ppl_q),
              file=sys.stderr)
        return 2

    delta_ppl = ppl_q - ppl_b
    delta_nll = math.log(ppl_q) - math.log(ppl_b)

    print("PPL_b   = %.6g" % ppl_b)
    print("PPL_q   = %.6g" % ppl_q)
    print("ΔPPL    = %.6g" % delta_ppl)
    print("ΔNLL    = %.6g   # nats; ln(PPL_q)-ln(PPL_b)" % delta_nll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
