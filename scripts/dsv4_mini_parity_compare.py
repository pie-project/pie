"""Read two logits dumps of the `mini-l5-e16` probe battery against each other.

    python3 scripts/dsv4_mini_parity_compare.py DIR --a pie --b ref [--probes probes.json]

`DIR/NAME.<tag>.tf.f32` ([len, vocab]) and `DIR/NAME.<tag>.gen.f32` ([steps, vocab])
for both tags. Per probe and arm: argmax agreement, where the first flip is, the
logit gap at that flip in each dump (a near-tie flip is the bf16 floor; a flip won by
several logits is a fault), max |Δlogit|, and the mean KL(b || a) over positions.
"""

import argparse
import json
import os

import numpy as np


def load(dir_, name, tag, arm, vocab):
    path = os.path.join(dir_, f"{name}.{tag}.{arm}.f32")
    if not os.path.exists(path):
        return None
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(-1, vocab)


def logsoftmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def compare(a, b, label):
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    aa, ba = a.argmax(-1), b.argmax(-1)
    agree = aa == ba
    la, lb = logsoftmax(a), logsoftmax(b)
    kl = (np.exp(lb) * (lb - la)).sum(-1)
    maxd = np.abs(a - b).max(-1)
    # b's own margin between its top two, where the two dumps disagree
    top2 = np.sort(b, axis=-1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    flips = np.nonzero(~agree)[0]
    line = f"    {label:<4} rows={n:3d} agree={agree.sum():3d}/{n:<3d} maxΔ={maxd.max():7.3f} meanKL={kl.mean():.4f} maxKL={kl.max():.4f}"
    if len(flips):
        f = flips[0]
        line += f"  first flip @{f} (b top {ba[f]} margin {margin[f]:.3f}; a chose {aa[f]}, gap in b {b[f, ba[f]] - b[f, aa[f]]:.3f})"
        big = flips[margin[flips] > 1.0]
        line += f"  flips won by >1 logit in b: {len(big)}"
    print(line)
    return agree.sum(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--a", default="pie")
    ap.add_argument("--b", default="ref")
    ap.add_argument("--probes", default=None)
    args = ap.parse_args()
    probes_path = args.probes or os.path.join(args.dir, "probes.json")
    probes = json.load(open(probes_path))["probes"]
    tot = [0, 0]
    for p in probes:
        name = p["name"]
        meta_b = os.path.join(args.dir, f"{name}.{args.b}.json")
        meta_a = os.path.join(args.dir, f"{name}.{args.a}.json")
        if not (os.path.exists(meta_a) and os.path.exists(meta_b)):
            print(f"  {name}: missing a dump ({args.a}: {os.path.exists(meta_a)}, {args.b}: {os.path.exists(meta_b)})")
            continue
        ma, mb = json.load(open(meta_a)), json.load(open(meta_b))
        vocab = mb["vocab"]
        print(f"  {name} ({len(p['ids'])} tokens)  gen {args.a}={ma['gen'][:10]}  gen {args.b}={mb['gen'][:10]}")
        for arm in ("tf", "gen"):
            a, b = load(args.dir, name, args.a, arm, vocab), load(args.dir, name, args.b, arm, vocab)
            if a is None or b is None:
                continue
            g, n = compare(a, b, arm)
            if arm == "tf":
                tot[0] += g
                tot[1] += n
    print(f"teacher-forced argmax agreement over the battery: {tot[0]}/{tot[1]}")


if __name__ == "__main__":
    main()
