"""Compare pie's and vLLM's per-layer activation dumps.

Produce the two dumps first:

    PIE_ACT_DUMP_DIR=/tmp/act/pie sdk/python-server/.venv/bin/python \
        benches/pie_bench.py latency --model M --driver cuda_native ... \
        --pretokenized-prompts
    /root/.venv/vllm/bin/python benches/act_dump_vllm.py --model M \
        --out /tmp/act/vllm

then:

    python benches/act_parity.py /tmp/act/pie /tmp/act/vllm

The tags are emitted in forward order, so the *first* row with a large error is
the sub-block to debug. Cosine similarity is the headline number because the
shrunken checkpoints have wildly different activation scales per layer, which
makes a raw absolute difference hard to read.

A uniform error that appears at the *first* GEMM and then neither grows nor
shrinks is not a bug -- it is a different arithmetic contract. vLLM runs
DeepSeek-V4's MXFP8 weights as w8a8, quantizing the activation to FP8 as well;
pie dequantizes the weight and keeps the activation in bf16. Against an fp64
reference built from the checkpoint, pie's `wq_b` output is rel_l2 0.0017
(bf16 rounding) and vLLM's is 0.0202, at every layer. So dsv4pro-mini emitting
different tokens from vLLM is vLLM's quantization noise, not pie drift, and
chasing it as a kernel bug is wasted effort. Build the fp64 reference before
concluding anything:

    W = w.to(f64) * scale.to(f32).to(f64).repeat_interleave(bn, 0)
                                          .repeat_interleave(bk, 1)
    ref = x @ W.T

(The scales are `float8_e8m0fnu` -- pure exponent -- over 128x128 blocks.)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def _order_key(tag: str) -> tuple:
    m = re.match(r"L(\d+)_(.*)", tag)
    if m:
        sub = {
            "norm_x": 0,
            "q_a": 1,
            "kv_c": 2,
            "q_b": 3,
            "kv_b": 4,
            "kv": 5,
            "attn_v": 6,
            "attn_out": 7,
            "attn_irope": 8,
            "wo_a_in": 9,
            "wo_a_out": 9.5,
            "o_proj": 9.7,
            "res_attn": 9.8,
            "ffn_in": 9.85,
            "ffn_out": 9.9,
            "post_attn": 10,
            "out": 11,
        }.get(m.group(2), 12)
        return (1, int(m.group(1)), sub)
    order = {
        "embed": 0,
        "hc_expand": 1,
        "hc_head": 2,
        "final_norm": 3,
        "logits": 4,
    }
    return (0 if tag in ("embed", "hc_expand") else 2, order.get(tag, 9), 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pie_dir")
    ap.add_argument("vllm_dir")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--all", action="store_true",
                    help="print every tag instead of stopping at the first divergence")
    ap.add_argument(
        "--device",
        type=int,
        default=0,
        help="Which CUDA device of the pie dump to read. At tp>1 every rank "
             "writes into the same directory and distinguishes itself with a "
             "`d<device>_` filename prefix; device 0 holds the full "
             "post-all-reduce residual stream, which is what a TP=1 reference "
             "engine like vLLM can be compared against.",
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=0,
        help="Compare only the last N token rows (0 = all). Useful when the "
             "two engines batch a different number of prompt positions.",
    )
    args = ap.parse_args()

    pie_dir, vllm_dir = Path(args.pie_dir), Path(args.vllm_dir)
    # pie tags its files with the writing device so that the ranks of a tp>1
    # run do not overwrite each other; a TP=1 dump from an older build has no
    # such prefix, so fall back to the bare form rather than reporting an
    # empty intersection.
    for prefix in (f"d{args.device}_s{args.step}_", f"s{args.step}_"):
        pie = {f.name[len(prefix):-4]: f
               for f in pie_dir.glob(f"{prefix}*.npy")}
        if pie:
            break
    # The reference is usually vLLM, whose dumps are plain `s0_<tag>.npy`, but a
    # TP=1 pie run is just as valid a reference and carries the device prefix.
    for prefix in ("s0_", "d0_s0_"):
        vllm = {f.name[len(prefix):-4]: f for f in vllm_dir.glob(f"{prefix}*.npy")}
        if vllm:
            break

    common = sorted(set(pie) & set(vllm), key=_order_key)
    if not common:
        print(f"no common tags. pie={sorted(pie)} vllm={sorted(vllm)}")
        return
    only_pie = sorted(set(pie) - set(vllm), key=_order_key)
    only_vllm = sorted(set(vllm) - set(pie), key=_order_key)
    if only_pie:
        print(f"[pie only]  {only_pie}")
    if only_vllm:
        print(f"[vllm only] {only_vllm}")

    print(f"{'tag':<18} {'shape':>18} {'cos':>10} {'rel_l2':>10} {'max_abs':>10}")
    for tag in common:
        a = np.load(pie[tag])
        b = np.load(vllm[tag])
        if a.shape != b.shape:
            if a.shape[1:] != b.shape[1:]:
                print(f"{tag:<18} {str(a.shape):>18} SHAPE MISMATCH vs {b.shape}")
                continue
            n = min(a.shape[0], b.shape[0])
            # pie pads the batch at the *end* (graph-shape rounding), so drop
            # its trailing rows; but for `logits` pie only materialises the
            # last row per request, so keep vLLM's trailing rows.
            a = a[:n] if a.shape[0] > b.shape[0] else a
            b = b[-n:] if b.shape[0] > n else b[:n]
            a = a[:n]
        if args.rows:
            a, b = a[-args.rows:], b[-args.rows:]
        af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        denom = np.linalg.norm(af) * np.linalg.norm(bf)
        cos = float(af @ bf / denom) if denom > 0 else float("nan")
        rel = float(np.linalg.norm(af - bf) / (np.linalg.norm(bf) + 1e-30))
        mx = float(np.max(np.abs(af - bf))) if af.size else 0.0
        flag = "  <-- FIRST DIVERGENCE" if cos < args.threshold else ""
        print(f"{tag:<18} {str(a.shape):>18} {cos:10.6f} {rel:10.5f} "
              f"{mx:10.4f}{flag}")
        if cos < args.threshold and not args.all:
            break


if __name__ == "__main__":
    main()
