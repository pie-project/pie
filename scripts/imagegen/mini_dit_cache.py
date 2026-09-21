#!/usr/bin/env python3
"""
mini_dit_cache.py -- STEP CACHING ON THE DEVICE, checked without a golden.

The design is `.wiki/imagegen/conditional-fire.md`; this is its §9 gate.  The
`mini-dit-parity` inferlet's `--cache` mode runs the whole Euler schedule on
the device -- one fire per step, three lanes, nothing per step across the host
-- and the image lane's epilogue carries the step-cache decision and the
reuse:

    skip_now = skip.take()                     # decided by the LAST fire
    v_use    = select(skip_now, cache, velocity())
    x        = x + dt(k) * v_use
    cache.put(v_use)
    skip.put(and(k >= 1, rel(v_use, cache) < threshold))   # for the NEXT fire

WHAT IS BEING CHECKED, AND WHY IT NEEDS NO REFERENCE
----------------------------------------------------
Three claims, none of which can hold by accident.

  1. AT THRESHOLD 0 NOTHING IS EVER SKIPPED AND THE TRAJECTORY IS BIT-FOR-BIT
     THE UNSKIPPED ONE.  `rel < 0` is false for every finite non-negative
     `rel`, so `select` takes the fresh velocity at every fire and the
     arithmetic on `x` is `euler_step` on the same operands as the plain loop.
     The plain loop is `latent::seed_or_step` itself -- the shipped body, not
     a rearrangement -- so this is the claim that proves the cache path is
     inert when it says it is inert.  Bit for bit; a "close" answer here would
     mean the select is not a select.

  2. AT A LARGE THRESHOLD EVERYTHING AFTER THE FIRST STEP IS SKIPPED AND THE
     TRAJECTORY IS THE FIRST STEP HELD.  `rel` is finite, so the writer fires
     from `k >= 1` and fire 2 onward reuses fire 1's velocity forever.  The
     host recovers each step's used velocity from the trajectory itself,
     `v_k = (x_k - x_{k-1}) / dt_k`, and they must all be one vector.  This is
     what proves the cache holds what was USED and not what the trunk
     answered: a cache that refreshed from the plane would drift.

  3. THE SKIP COUNT IS WHAT THE GUEST ASKED FOR.  The epilogue keeps its own
     running count on the device.  The host re-derives the rule from the
     reported metric series -- |{k : k >= 2 and rel[k-1] < threshold}| -- and
     the two must agree at EVERY threshold, not only the two extremes.  That
     is what proves the counter, the decision and the reuse are keyed on one
     word rather than three that happen to agree.

WHAT IT DOES NOT CHECK
----------------------
It does not check a speedup, because there is none yet: the trunk still runs
on a skipped fire and its answer is discarded by the guest's own `select`.
Making the trunk not run is the engine half of the design (§6.3 -- a
conditional node reading a per-lane byte plane), and it cannot change a
number, because the number was already discarded.  Any wall-clock difference
this gate could report is the cost of the metric arithmetic, which is the
opposite of the point.

    python mini_dit_cache.py all --out /tmp/mini-dit-cache --config ~/.pie/config.toml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mini_dit_parity as P

REPO = P.REPO

HUGE = 1e9
VELOCITY_REL_TOL = 1e-5

def run_cache(args, where: str, threshold: float | None) -> list[dict]:
    """Fire the cached loop once per batch element; answer the documents."""
    paths = P.numbered(args.out, "case", True)
    if not paths:
        raise SystemExit(f"{args.out}: no case JSON; run `case` first")
    pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")
    if not os.path.exists(pie):
        raise SystemExit(
            f"{pie}: no pie binary. Build one with "
            f"`cargo build -p pie --features cuda`, or pass --pie."
        )
    binary = P.wasm(args.inferlet)
    manifest = os.path.join(args.inferlet, "Pie.toml")
    os.makedirs(where, exist_ok=True)
    docs = []
    for b, case in enumerate(paths):
        out = os.path.join(where, f"pie_{b}.json")
        cmd = [pie]
        if args.config:
            cmd += ["--config", args.config]
        cmd += ["run", "--path", binary, "--manifest", manifest, "--"]
        text = open(case).read()
        n = 8
        step = -(-len(text) // n)
        for i in range(n):
            cmd += [f"--case_{i}", text[i * step:(i + 1) * step]]
        cmd += ["--cache", "true",
                "--cache_steps", str(args.steps),
                "--cache_seed", str(args.seed)]
        if threshold is not None:
            cmd += ["--cache_threshold", repr(float(threshold))]
        label = "baseline" if threshold is None else f"thr={threshold:g}"
        print(f"[cache] batch {b} {label}")
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        with open(out, "w") as f:
            f.write(done.stdout)
        with open(out[:-5] + ".stderr", "w") as f:
            f.write(done.stderr)
        if done.returncode != 0:
            sys.stderr.write(done.stdout)
            sys.stderr.write(done.stderr)
            raise SystemExit(f"pie run failed ({done.returncode}) for {label}")
        docs.append(P.document(out))
    return docs

def trajectory(doc: dict) -> np.ndarray:
    """`[fires, rows * feats]` — the latent after every fire."""
    return np.asarray(doc.get("cache_x", []), dtype=np.float32)

def series(doc: dict, key: str) -> np.ndarray:
    return np.asarray(doc.get(key, []), dtype=np.float64)

def claim_one(base: dict, zero: dict, at: int) -> int:
    """Threshold 0: nothing skipped, and the trajectory IS the plain loop's."""
    xb, xz = trajectory(base), trajectory(zero)
    if xb.size == 0 or xb.shape != xz.shape:
        print(f"[cache] batch {at}: FAIL claim 1 — the two runs traced "
              f"{xb.shape} and {xz.shape}")
        return 1
    did = series(zero, "cache_did")
    count = series(zero, "cache_count")
    bad = 0
    if np.array_equal(xb, xz):
        print(f"[cache] batch {at}: PASS claim 1 trajectory — threshold 0 is "
              f"BIT-FOR-BIT the unskipped loop over {xb.shape[0]} fires")
    else:
        off = float(np.abs(xb - xz).max())
        print(f"[cache] batch {at}: FAIL claim 1 trajectory — threshold 0 "
              f"differs from the plain loop by max-abs {off:.3g}")
        bad += 1
    skipped = int(did.sum())
    if skipped == 0 and (count.size == 0 or count[-1] == 0):
        print(f"[cache] batch {at}: PASS claim 1 count — threshold 0 skipped nothing")
    else:
        print(f"[cache] batch {at}: FAIL claim 1 count — threshold 0 skipped "
              f"{skipped} fire(s), device count {count[-1] if count.size else 'n/a'}")
        bad += 1
    return bad

def claim_two(big: dict, steps: int, at: int) -> int:
    """A large threshold: everything after the first step is the first step held."""
    x = trajectory(big)
    dts = series(big, "cache_dts")
    did = series(big, "cache_did")
    count = series(big, "cache_count")
    if x.shape[0] < 3:
        print(f"[cache] batch {at}: FAIL claim 2 — {x.shape[0]} fires is too few to hold")
        return 1
    bad = 0
    want = steps - 1
    if int(did.sum()) == want and count.size and int(count[-1]) == want:
        print(f"[cache] batch {at}: PASS claim 2 count — {want} of {steps} steps "
              f"skipped, which is every step after the first")
    else:
        print(f"[cache] batch {at}: FAIL claim 2 count — {int(did.sum())} skipped "
              f"and the device counted {count[-1] if count.size else 'n/a'}; "
              f"every-after-the-first is {want}")
        bad += 1
    used = []
    for k in range(1, x.shape[0]):
        dt = float(dts[k]) if k < dts.size else 0.0
        if dt == 0.0:
            print(f"[cache] batch {at}: FAIL claim 2 — fire {k} integrated dt 0")
            return bad + 1
        used.append((x[k] - x[k - 1]) / dt)
    first = used[0]
    scale = max(float(np.linalg.norm(first)), 1e-30)
    worst = max(float(np.linalg.norm(v - first)) / scale for v in used[1:])
    if worst <= VELOCITY_REL_TOL:
        print(f"[cache] batch {at}: PASS claim 2 held — every step after the "
              f"first integrated the FIRST step's velocity, worst rel {worst:.3g}")
    else:
        print(f"[cache] batch {at}: FAIL claim 2 held — a later step's velocity "
              f"differs from the first's by rel {worst:.3g}")
        bad += 1
    return bad

def claim_three(doc: dict, threshold: float, at: int, label: str) -> int:
    """The device's skip count is the rule applied to the reported metric."""
    rel = series(doc, "cache_rel")
    did = series(doc, "cache_did")
    count = series(doc, "cache_count")
    if rel.size == 0 or rel.size != did.size:
        print(f"[cache] batch {at} {label}: FAIL claim 3 — the probe reported "
              f"{rel.size} metrics and {did.size} decisions")
        return 1
    want = np.zeros_like(did)
    for k in range(1, did.size):
        want[k] = 1.0 if (k - 1 >= 1 and rel[k - 1] < threshold) else 0.0
    bad = 0
    if np.array_equal(want, did):
        print(f"[cache] batch {at} {label}: PASS claim 3 rule — the {int(did.sum())} "
              f"skips are exactly the fires the reported metric asked for")
    else:
        where = [int(i) for i in np.nonzero(want != did)[0]]
        print(f"[cache] batch {at} {label}: FAIL claim 3 rule — the device and the "
              f"rule disagree at fire(s) {where}")
        bad += 1
    if count.size and int(count[-1]) == int(did.sum()):
        print(f"[cache] batch {at} {label}: PASS claim 3 count — the DEVICE counted "
              f"{int(count[-1])}, which is what it skipped")
    else:
        print(f"[cache] batch {at} {label}: FAIL claim 3 count — the device counted "
              f"{count[-1] if count.size else 'n/a'} and skipped {int(did.sum())}")
        bad += 1
    return bad

def gate(args) -> int:
    case_args = argparse.Namespace(**vars(args))
    case_args.euler = True
    P.cases(case_args)

    base = run_cache(args, os.path.join(args.out, "baseline"), None)
    zero = run_cache(args, os.path.join(args.out, "thr0"), 0.0)
    big = run_cache(args, os.path.join(args.out, "thrbig"), HUGE)

    decidable = [series(doc, "cache_rel")[1:-1] for doc in zero]
    pool = np.concatenate([d for d in decidable if d.size]) if decidable else np.zeros(0)
    middle = float(np.median(pool)) if pool.size else 0.5
    mid = run_cache(args, os.path.join(args.out, "thrmid"), middle)

    bad = 0
    for at, (b, z, g, m) in enumerate(zip(base, zero, big, mid)):
        bad += claim_one(b, z, at)
        bad += claim_two(g, args.steps, at)
        bad += claim_three(z, 0.0, at, "thr=0")
        bad += claim_three(g, HUGE, at, "thr=huge")
        bad += claim_three(m, middle, at, f"thr={middle:.4g}")
        skips = int(series(m, "cache_did").sum())
        span = "all" if skips == args.steps - 1 else ("none" if skips == 0 else "partial")
        print(f"[cache] batch {at}: middle threshold {middle:.4g} skipped "
              f"{skips}/{args.steps - 1} ({span})")
        rel = series(z, "cache_rel")
        print(f"[cache] batch {at}: unskipped metric series "
              f"{[round(float(v), 5) for v in rel]}")

    print(f"[cache] {'PASS' if bad == 0 else 'FAIL'} {bad} claim(s) failed")
    return 1 if bad else 0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["case", "gate", "all"])
    ap.add_argument("--out", default="/tmp/mini-dit-cache")
    ap.add_argument("--golden", default=P.DEFAULT_GOLDEN)
    ap.add_argument("--policy", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--pie", default=None)
    ap.add_argument("--inferlet",
                    default=os.path.join(REPO, "examples/mini-dit-parity"))
    ap.add_argument("--steps", type=int, default=6,
                    help="Euler steps; fires are steps + 1")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.command == "case":
        args.euler = True
        P.cases(args)
        return 0
    return gate(args)

if __name__ == "__main__":
    raise SystemExit(main())
