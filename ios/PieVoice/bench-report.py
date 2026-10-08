#!/usr/bin/env python3
"""Turn PIEBENCH jsonl into the table that goes on the project page."""
import json, sys, statistics
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by_model, current = defaultdict(list), None
for r in rows:
    if r.get("event") == "start":
        current = r["model"]
        by_model[current] = {"device": r.get("device"), "turns": [], "warmup": None, "peak": None}
    elif current is None:
        continue
    elif r.get("event") == "warmup":
        by_model[current]["warmup"] = r["seconds"]
    elif r.get("event") == "turn":
        by_model[current]["turns"].append(r)
    elif r.get("event") == "done":
        by_model[current]["peak"] = r.get("peak_footprint_mb")
    elif r.get("event") == "did_not_fit":
        # The process was killed mid-run — on a device that means the model
        # exceeded the memory the OS was willing to give the app.
        if current:
            by_model[current]["did_not_fit"] = r

print(f"{'model':<22}{'warmup':>8}{'TTFT t1':>9}{'decode':>9}{'prefill t1':>11}{'reuse t2+':>10}{'peak MB':>9}")
print("-" * 78)
for model, d in by_model.items():
    t = d["turns"]
    if d.get("did_not_fit"):
        n = d["did_not_fit"].get("turns_completed", 0)
        peak = f"{d['peak']:.0f}" if d.get("peak") else "?"
        print(f"{model:<22}  DID NOT FIT — killed after {n} turn(s), peak {peak} MB")
        continue
    if not t:
        print(f"{model:<22}  (no turns — failed to load)")
        continue
    ttft = t[0]["ttft_s"]
    decode = statistics.median(x["decode_tps"] for x in t)
    prefill1 = t[0]["prefill"]
    reuse = statistics.median(x["reused"] for x in t[1:]) if len(t) > 1 else 0
    print(f"{model:<22}{d['warmup']:>7.1f}s{ttft:>8.2f}s{decode:>8.1f}/s"
          f"{prefill1:>11}{reuse:>10.0f}{d['peak']:>9.0f}")
    if len(t) > 1:
        pre = ", ".join(str(x["prefill"]) for x in t)
        print(f"{'':<22}per-turn prefill: {pre}")
