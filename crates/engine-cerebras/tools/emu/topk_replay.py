#!/usr/bin/env python3
"""topk_replay.py <rlog.json (tf_gptoss.py --rlog)>: every recorded router decision's logits through a port of
ctl_cy_gptoss_moe.csl's top 4 (pairwise-maximum rounds, the first block and the first word holding the maximum,
that word out) against the emulation's route (a stable descending argsort): selection and order."""
import json, sys
import numpy as np
r = json.load(open(sys.argv[1]))
L = np.array(r["logits"], np.uint32).view(np.float32)
routes = r["routes"]
NEG = np.float32(-3.4e38)
def kernel_topk(g, tk=4):
    h = g.copy(); sel = []
    n = len(h)
    for _ in range(tk):
        t1 = np.maximum(h[0::2], h[1::2]); t2 = np.maximum(t1[0::2], t1[1::2]); t3 = np.maximum(t2[0::2], t2[1::2])
        m = t3[0]
        for b in range(1, n // 8):
            if t3[b] > m: m = t3[b]
        b = 0
        while t3[b] != m: b += 1
        i = b * 8
        while h[i] != m: i += 1
        sel.append(i); h[i] = NEG
    return sel
def kernel_topk_tree(g, tk=4):
    """the tree built once, the leftmost leaf holding the maximum down the tree, its path redone a round"""
    h = g.copy(); sel = []
    n = len(h)
    t1 = np.maximum(h[0::2], h[1::2]); t2 = np.maximum(t1[0::2], t1[1::2]); t3 = np.maximum(t2[0::2], t2[1::2])
    fm = lambda a, b: b if b > a else a
    for _ in range(tk):
        m = t3[0]
        for b in range(1, n // 8):
            if t3[b] > m: m = t3[b]
        b = 0
        while t3[b] != m: b += 1
        j2 = 2 * b + (0 if t2[2 * b] == m else 1)
        j1 = 2 * j2 + (0 if t1[2 * j2] == m else 1)
        i = 2 * j1 + (0 if h[2 * j1] == m else 1)
        sel.append(i); h[i] = NEG
        t1[j1] = fm(h[2 * j1], h[2 * j1 + 1]); t2[j2] = fm(t1[2 * j2], t1[2 * j2 + 1]); t3[b] = fm(t2[2 * b], t2[2 * b + 1])
    return sel
bad_tree = [k for k, (g, want) in enumerate(zip(L, routes)) if kernel_topk_tree(g) != list(want)]
# synthetic ties: logits rounded hard, so equal values (and +-0) are common
rng = np.random.default_rng(0)
syn = [np.float32(np.round(rng.normal(size=len(L[0])) * 2) / 2) * np.float32(rng.choice([1, -1])) for _ in range(20000)]
syn += [np.where(rng.random(len(L[0])) < 0.5, np.float32(0.0), np.float32(-0.0)).astype(np.float32) for _ in range(200)]
syn_bad = sum(1 for g in syn if kernel_topk_tree(np.asarray(g, np.float32)) != kernel_topk(np.asarray(g, np.float32)))
bad = [k for k, (g, want) in enumerate(zip(L, routes)) if kernel_topk(g) != list(want)]
ties = sum(1 for g in L if np.sort(g)[::-1][3] == np.sort(g)[::-1][4])
print(json.dumps({"decisions": len(routes), "differ": len(bad), "first": bad[:5], "exact_4th_5th_ties_in_logits": ties,
                  "tree_differ": len(bad_tree), "tree_vs_rebuild_synthetic_ties": [syn_bad, len(syn)]}))
