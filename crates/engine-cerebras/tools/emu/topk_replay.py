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
bad = [k for k, (g, want) in enumerate(zip(L, routes)) if kernel_topk(g) != list(want)]
ties = sum(1 for g in L if np.sort(g)[::-1][3] == np.sort(g)[::-1][4])
print(json.dumps({"decisions": len(routes), "differ": len(bad), "first": bad[:5], "exact_4th_5th_ties_in_logits": ties}))
