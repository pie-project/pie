#!/usr/bin/env python3
"""classify_tf.py <hf golden.json> <tf32.json> <mismatches: a run report (run_gptoss.py --report json, or
tf_gptoss.py's json with a mode)> [mode]: the pass criteria for gpt-oss runs against the HF bf16 golden.
Primary: every teacher-forced mismatch at token i is classified as
  bf16_top1_tie   the golden's own top-1 margin at i is 0 (HF picked between equal bf16 logits)
  router_tie_step an exact 4th / 5th router tie in some layer of HF's step that produced token i
  router_tie_prior no tie at that step, but one at an earlier step (it changed HF's cached K / V)
  none            no tie anywhere before: a real disagreement
Secondary: at the same positions, an f32 teacher-forced argmax and whether ours equals it: hf_gptoss_tf32.py's json, or
tf_gptoss.py's json:ref (the numpy f32 reference; low memory: experts decoded on demand)."""
import json, sys
g = json.load(open(sys.argv[1]))
f32 = None
if sys.argv[2] != "-":
    f32 = json.load(open(sys.argv[2].split(":")[0]))
    if ":" in sys.argv[2]:                     # tf_gptoss.py's json and its f32 mode (ref): the low-memory reference
        f32 = f32[sys.argv[2].split(":")[1]]
r = json.load(open(sys.argv[3]))
mode = sys.argv[4] if len(sys.argv) > 4 else None
if mode:
    mis = [{"i": i} for i in r[mode]["mismatches"]]
else:
    mis = [m for rep in r["reps"] for m in rep["mismatches"]]
gaps = g["router_gaps"]                        # [step][layer]: step 0 = the prompt's last position
tie_step = [any(v == 0.0 for v in s) for s in gaps]
out = []
for m in mis:
    i = m["i"]
    if g["margins"][i] == 0.0:
        c = "bf16_top1_tie"
    elif tie_step[i]:
        c = "router_tie_step"
    elif any(tie_step[:i]):
        c = "router_tie_prior"
    else:
        c = "none"
    row = {"i": i, "class": c, "golden_margin": g["margins"][i], "got": m.get("got")}
    if f32:
        row["f32_argmax"] = f32["argmax"][i]; row["f32_margin"] = f32["margins"][i]
        row["ours_eq_f32"] = (m.get("got") == f32["argmax"][i]) if m.get("got") is not None else None
    out.append(row)
summ = {"mismatches": len(out), "by_class": {c: sum(1 for o in out if o["class"] == c) for c in ("bf16_top1_tie", "router_tie_step", "router_tie_prior", "none")},
        "golden_bf16_top1_ties": sum(1 for v in g["margins"] if v == 0.0), "golden_router_exact_ties": g["router_exact_ties"],
        "golden_steps_with_router_tie": sum(tie_step)}
if f32:
    summ["f32_vs_golden_disagreements"] = [i for i, (a, b) in enumerate(zip(f32["argmax"], g["tokens"])) if a != b]
print(json.dumps({"summary": summ, "rows": out}, indent=1))
