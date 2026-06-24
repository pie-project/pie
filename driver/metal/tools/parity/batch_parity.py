#!/usr/bin/env python3
"""batch_parity — per-sequence parity gate for the M>1 (multi-batch) decode path.

The throughput bench runs M sequences concurrently. Correctness for batch=M
reduces to: **every sequence in the batch must produce bit-identical-to-gate
output as it would have produced ALONE (M=1)** — i.e. batching introduces no
cross-sequence contamination, the per-seq paged-KV indexing is correct, and
ragged (mixed-length) prompts in one batch don't perturb each other.

So the gate is: for each sequence i, compare the batched run's seq-i taps against
that prompt's sealed M=1 golden (the same per-kernel cosine_bisect walk), AND
check the seq-i logits argmax matches the golden argmax exactly.

Two candidate layouts (the M>1 executor picks one; this gate supports both):

  --layout rowslice   batched dump writes one file per tap, shaped [M, ...]
                      (the natural decode-step buffer: M seqs x 1 token each).
                      seq i = row i of dim 0.
  --layout subdir     batched dump writes <batched>/seq<i>/<layer>.<kernel>.npy
                      (one standard single-seq dump dir per sequence).

Golden mapping: --golden i=<dir> ...  (dir = a sealed M=1 dump for prompt i,
exactly what cosine_bisect.py gates against today).

Exit 0 = every sequence passes its gate; 1 = at least one seq diverged.
"""
import argparse
import os
import sys

import numpy as np

# Reuse the single-seq walk machinery verbatim so the per-kernel ordering,
# arch auto-select, skip-tags and reshape-on-size-match stay identical.
import cosine_bisect as cb


def load_golden_arrays(d):
    """(layer,kernel) -> ndarray for a sealed single-seq golden dir."""
    paths = cb.load_dir(d)
    return {k: np.load(v) for k, v in paths.items()}


def load_candidate_subdir(batched, i):
    d = os.path.join(batched, f"seq{i}")
    if not os.path.isdir(d):
        return None
    return {k: np.load(v) for k, v in cb.load_dir(d).items()}


def load_qo_indptr(batched, explicit):
    """qo_indptr [R+1]: seq i owns activation rows [qo_indptr[i], qo_indptr[i+1])."""
    path = explicit or os.path.join(batched, "qo_indptr.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.int64).ravel()


def load_candidate_rowslice(batched, i, qo, golden, full_span):
    """Slice seq i's qo-span out of every [N, ...] batched tap (beta's contract).

    seq i owns rows [qo[i], qo[i+1]). The sealed M=1 goldens are single-position
    (the decode/decision row), so by default we compare the seq's LAST span row
    (the token that feeds the next decode) against the golden. With full_span and
    a multi-row golden of matching span length, compare the whole span (catches
    mid-prompt contamination in ragged prefill).
    """
    lo, hi = int(qo[i]), int(qo[i + 1])
    span_len = hi - lo
    out = {}
    for k, v in cb.load_dir(batched).items():
        if k == (None, "qo_indptr"):
            continue
        a = np.load(v)
        if not a.shape or a.shape[0] < hi:        # unbatched/global tap (shared const) — as-is
            out[k] = a
            continue
        span = a[lo:hi]                            # [span_len, ...]
        g = golden.get(k)
        want_full = (full_span and g is not None
                     and g.shape and g.shape[0] == span_len and g.ndim >= 2)
        out[k] = span if want_full else span[-1]   # decision row by default
    return out



def gate_one_seq(golden, cand, threshold, skip):
    """Run the cosine_bisect per-kernel walk on in-memory arrays.

    Returns (passed, first_bad, rows). first_bad = (layer,kernel,why) or None.
    """
    cb.select_order(golden)
    keys = sorted(golden, key=lambda k: cb.exec_key(*k))
    first_bad = None
    rows = []
    for layer, kernel in keys:
        if kernel in skip:
            continue
        g = golden[(layer, kernel)]
        if (layer, kernel) not in cand:
            rows.append((layer, kernel, None, "MISSING in candidate"))
            if first_bad is None:
                first_bad = (layer, kernel, "missing")
            continue
        c = cand[(layer, kernel)]
        if g.shape != c.shape and g.size != c.size:
            rows.append((layer, kernel, None, f"SIZE {g.size} vs {c.size}"))
            if first_bad is None:
                first_bad = (layer, kernel, "shape")
            continue
        cos = cb.cosine(g, c)
        note = "" if cos >= threshold else "  <-- BELOW GATE"
        rows.append((layer, kernel, cos, note.strip()))
        if cos < threshold and first_bad is None:
            first_bad = (layer, kernel, cos)
    return (first_bad is None), first_bad, rows


def argmax_of(arrays, prefer=("logits_softcap", "logits")):
    for tag in prefer:
        for (layer, kernel), a in arrays.items():
            if kernel == tag:
                return int(np.asarray(a).astype(np.float64).ravel().argmax())
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batched", required=True, help="M>1 batched dump dir")
    ap.add_argument("--golden", nargs="+", required=True,
                    help="per-seq goldens: i=<dir> (i = batch slot index)")
    ap.add_argument("--layout", choices=["rowslice", "subdir"], default="rowslice")
    ap.add_argument("--qo-indptr", default="",
                    help="qo_indptr .npy [R+1] (rowslice layout); default <batched>/qo_indptr.npy")
    ap.add_argument("--full-span", action="store_true",
                    help="compare each seq's full qo-span vs a full-span golden "
                         "(default: last/decision row only, matching single-position goldens)")
    ap.add_argument("--threshold", type=float, default=0.999,
                    help="cosine gate (default 0.999, the operational decision gate)")
    ap.add_argument("--skip", default="",
                    help="comma-separated kernel tags to exclude (e.g. q_norm,k_norm)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    gmap = {}
    for spec in args.golden:
        idx, _, path = spec.partition("=")
        if not path:
            print(f"error: bad --golden spec {spec!r} (want i=<dir>)", file=sys.stderr)
            return 2
        gmap[int(idx)] = path
    m = len(gmap)

    qo = None
    if args.layout == "rowslice":
        qo = load_qo_indptr(args.batched, args.qo_indptr)
        if qo is None:
            print(f"error: rowslice layout needs qo_indptr (looked for "
                  f"{args.qo_indptr or os.path.join(args.batched, 'qo_indptr.npy')})",
                  file=sys.stderr)
            return 2
        if len(qo) != m + 1:
            print(f"error: qo_indptr has {len(qo)} entries, expected R+1={m+1}",
                  file=sys.stderr)
            return 2

    print(f"M={m} batch-parity gate  layout={args.layout}  threshold={args.threshold}"
          + (f"  qo_indptr={list(map(int, qo))}" if qo is not None else "")
          + (f"  skipping {sorted(skip)}" if skip else ""))
    print("=" * 72)

    all_pass = True
    for i in sorted(gmap):
        golden = load_golden_arrays(gmap[i])
        if args.layout == "subdir":
            cand = load_candidate_subdir(args.batched, i)
        else:
            cand = load_candidate_rowslice(args.batched, i, qo, golden, args.full_span)
        if not cand:
            print(f"seq{i}: ✗ NO CANDIDATE TAPS (layout={args.layout})")
            all_pass = False
            continue

        passed, first_bad, rows = gate_one_seq(golden, cand, args.threshold, skip)
        g_arg = argmax_of(golden)
        c_arg = argmax_of(cand)
        arg_ok = (g_arg is not None and g_arg == c_arg)
        worst = min((c for (_l, _k, c, _d) in rows if c is not None), default=None)

        tag = "✓ PASS" if (passed and arg_ok) else "✗ FAIL"
        wtxt = f"worst-cos={worst:.8f}" if worst is not None else "worst-cos=n/a"
        print(f"seq{i}: {tag}  argmax={c_arg} (golden {g_arg}) "
              f"{'✓' if arg_ok else '✗ARGMAX'}  {wtxt}  ({len(rows)} taps)")
        if not passed:
            l, k, why = first_bad
            detail = (f"cosine {why:.8f}" if isinstance(why, float) else why)
            print(f"        first divergence: {cb.fmt(l, k)}  ({detail})")
        if args.verbose:
            for l, k, c, d in rows:
                cs = "      -     " if c is None else f"{c:>12.8f}"
                print(f"          {cb.fmt(l, k):<26} {cs}  {d}")
        all_pass = all_pass and passed and arg_ok

    print("=" * 72)
    if all_pass:
        print(f"BATCH PARITY PASS: all {m} sequences match their M=1 golden "
              f"(>= cosine {args.threshold}, argmax exact)")
        return 0
    print(f"BATCH PARITY FAIL: at least one sequence diverged from its M=1 golden")
    return 1


if __name__ == "__main__":
    sys.exit(main())
