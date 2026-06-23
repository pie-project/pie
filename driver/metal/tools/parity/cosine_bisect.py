#!/usr/bin/env python3
"""cosine_bisect — localize the FIRST kernel/layer where a raw-Metal port
diverges from the sealed MLX-path golden.

The qwen3.6 decode graph (driver/metal/src/model/qwen36.cpp) dumps per-kernel
intermediates as `<layer>.<kernel>.npy` (layer-less tags `embed/final_norm/
logits.npy`) when `PIE_METAL_GOLDEN_DIR` is set. delta/beta emit identically
named raw-Metal dumps from the port. This tool walks both sets in *execution
order*, computes per-tensor cosine similarity, and reports the FIRST tensor
whose cosine drops below the gate (default 0.99999) — so a divergence reads as
"gdn_core layer 7", not just "logits wrong".

Usage:
  cosine_bisect.py --golden <dir> --candidate <dir> [--threshold 0.99999] [-v]

Exit code: 0 = all tensors pass the gate (port is bit-faithful to the gate);
           1 = a divergence was localized (or a tensor is missing/misshaped).
"""
import argparse
import os
import sys

import numpy as np

# Per-kernel execution rank WITHIN a decoder layer. Both layer types share the
# same residual-stream order; absent tags are simply skipped. Ranks need only be
# monotonic in dispatch order (gaps are fine). Mirrors qwen36.cpp + delta's
# decode_abi.hpp `Kernel` enum / DAG (mac-raw-metal-decode-dag).
INTRA_LAYER_ORDER = [
    "attn_norm",                                   # rms (both layer types)
    # ── GDN / linear-attention sublayer ──
    "gdn_in_qkv", "gdn_in_z", "gdn_in_a", "gdn_in_b",
    "gdn_core",                                    # beta op output (folds gated_rms)
    "gdn_out",
    # ── full-attention sublayer ──
    "q_proj", "k_proj", "v_proj",
    "q_norm", "k_norm",
    "rope_q", "rope_k",
    "sdpa", "attn_gated", "o_proj",
    # ── shared residual + MLP ──
    "attn_resid",
    "ffn_norm", "gate_proj", "up_proj", "swiglu", "down_proj",
    "layer_out",
]
_INTRA_RANK = {name: i for i, name in enumerate(INTRA_LAYER_ORDER)}

# Layer-less tags pinned to the start / end of the global sequence.
HEAD_TAGS = {"embed": -1}
TAIL_TAGS = {"final_norm": 10**6, "logits": 10**6 + 1}


def parse_tag(fname):
    """`<layer>.<kernel>.npy` -> (layer:int, kernel:str); `<kernel>.npy` -> (None, kernel)."""
    base = fname[:-4] if fname.endswith(".npy") else fname
    head, _, tail = base.partition(".")
    if tail and head.lstrip("-").isdigit():
        return int(head), tail
    return None, base


def exec_key(layer, kernel):
    """Global ordering key so files sort into true execution order."""
    if layer is None:
        if kernel in HEAD_TAGS:
            return (HEAD_TAGS[kernel], 0)
        if kernel in TAIL_TAGS:
            return (TAIL_TAGS[kernel], 0)
        return (10**6 + 100, kernel)  # unknown layer-less tag: after the tail
    return (layer, _INTRA_RANK.get(kernel, len(INTRA_LAYER_ORDER) + hash(kernel) % 997))


def load_dir(d):
    out = {}
    for fn in os.listdir(d):
        if not fn.endswith(".npy"):
            continue
        layer, kernel = parse_tag(fn)
        out[(layer, kernel)] = os.path.join(d, fn)
    return out


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def fmt(layer, kernel):
    return kernel if layer is None else f"{kernel} layer {layer}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", required=True, help="sealed MLX-path dump dir")
    ap.add_argument("--candidate", required=True, help="raw-Metal port dump dir")
    ap.add_argument("--threshold", type=float, default=0.99999,
                    help="cosine gate (default 0.99999)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every tensor, not just the divergence")
    args = ap.parse_args()

    golden = load_dir(args.golden)
    cand = load_dir(args.candidate)
    if not golden:
        print(f"error: no .npy golden tensors in {args.golden}", file=sys.stderr)
        return 2

    keys = sorted(golden, key=lambda k: exec_key(*k))
    first_bad = None
    rows = []
    for layer, kernel in keys:
        g = np.load(golden[(layer, kernel)])
        if (layer, kernel) not in cand:
            rows.append((layer, kernel, None, "MISSING in candidate"))
            if first_bad is None:
                first_bad = (layer, kernel, "missing")
            continue
        c = np.load(cand[(layer, kernel)])
        if g.shape != c.shape:
            rows.append((layer, kernel, None,
                         f"SHAPE {g.shape} vs {c.shape}"))
            if first_bad is None:
                first_bad = (layer, kernel, "shape")
            continue
        cos = cosine(g, c)
        mae = float(np.max(np.abs(g.astype(np.float64) - c.astype(np.float64))))
        note = "" if cos >= args.threshold else "  <-- BELOW GATE"
        rows.append((layer, kernel, cos, f"maxabs={mae:.3e}{note}"))
        if cos < args.threshold and first_bad is None:
            first_bad = (layer, kernel, cos)

    if args.verbose or first_bad is not None:
        print(f"{'tensor':<28} {'cosine':>12}  detail")
        print("-" * 70)
        for layer, kernel, cos, detail in rows:
            cs = "      -     " if cos is None else f"{cos:>12.8f}"
            print(f"{fmt(layer, kernel):<28} {cs}  {detail}")
        print("-" * 70)

    if first_bad is None:
        print(f"PASS: all {len(rows)} golden tensors >= cosine {args.threshold}")
        return 0

    layer, kernel, why = first_bad
    if why == "missing":
        print(f"DIVERGENCE: {fmt(layer, kernel)} is MISSING in candidate")
    elif why == "shape":
        print(f"DIVERGENCE: {fmt(layer, kernel)} shape mismatch")
    else:
        print(f"DIVERGENCE: first kernel below gate {args.threshold} = "
              f"{fmt(layer, kernel)}  (cosine {why:.8f})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
