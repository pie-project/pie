#!/usr/bin/env python3
"""hf_gptoss_golden.py <model> <out.json> <name> <n_tokens> (--ids a,b,c | --chat "text") [--reps 2]

The gpt-oss golden from HF transformers as released: MXFP4 experts dequantized by HF's own path
(Mxfp4Config(dequantize=True): FP4 LUT then ldexp, to bf16), the whole model in bf16, eager
attention, eager experts, greedy decode with the KV cache on CPU.

Writes prompt_ids, tokens, margins (top-1 minus top-2 of the f32 copy of the bf16 logits), the
first step's top ids, and per (step, layer) the router's 4th-vs-5th logit gap (the logits HF's
torch.topk sees, bf16) and exact ties at the 4th place. --reps N reruns the decode and reports
whether the logits are bit-identical (HF nondeterminism check: thread-count dependent reductions)."""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("name"); ap.add_argument("n", type=int)
ap.add_argument("--ids", default=None); ap.add_argument("--chat", default=None)
ap.add_argument("--reps", type=int, default=1); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
try:
    tok = AutoTokenizer.from_pretrained(a.model)
except Exception:
    tok = None
if a.ids:
    ids = [int(v) for v in a.ids.split(",")]
else:
    ids = tok.apply_chat_template([{"role": "user", "content": a.chat}], add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, quantization_config=Mxfp4Config(dequantize=True),
                                         attn_implementation="eager", experts_implementation="eager")
m.eval()
print(f"loaded {time.time() - t0:.0f}s experts dtype {m.model.layers[0].mlp.experts.gate_up_proj.dtype}", flush=True)
gaps = []
def hook(mod, inp, out):
    lg = out[0].float()[-1]          # the last position's logits (bf16 values)
    s, _ = torch.sort(lg, descending=True)
    gaps[-1].append([float(s[3] - s[4]), bool(s[3] == s[4])])
for l in m.model.layers:
    l.mlp.router.register_forward_hook(hook)

def run():
    gaps.clear()
    x = torch.tensor([ids])
    past = None; toks = []; marg = []; lgs = []; first = None
    with torch.no_grad():
        for s in range(a.n):
            gaps.append([])
            o = m(input_ids=x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            lg = o.logits[0, -1].float()
            lgs.append(lg.clone())
            v, i = torch.topk(lg, 8)
            if first is None: first = i.tolist()
            toks.append(int(i[0])); marg.append(float(v[0] - v[1]))
            x = torch.tensor([[int(i[0])]])
            if s % 32 == 0: print(a.name, s, int(i[0]), f"{marg[-1]:.4f}", f"{time.time() - t0:.0f}s", flush=True)
    return toks, marg, first, torch.stack(lgs), [g[:] for g in gaps]

toks, marg, first, lg0, g0 = run()
rep = []
for r in range(1, a.reps):
    t2, _, _, lg2, _ = run()
    rep.append({"tokens_equal": t2 == toks, "logits_bit_equal": bool(torch.equal(lg2, lg0)),
                "max_logit_diff": float((lg2 - lg0).abs().max())})
# the prompt step's gaps cover every prompt position's last row only; decode steps one row each
dec = [g for s in g0 for g in s]
res = {"name": a.name, "golden": "HF transformers bf16, MXFP4 dequant (Mxfp4Config dequantize), eager",
       "prompt_ids": ids, "tokens": toks, "margins": marg, "first_logits_top": first,
       "router_gaps": [[g[0] for g in s] for s in g0], "router_exact_ties": sum(int(g[1]) for g in dec),
       "router_gap_min": min(g[0] for g in dec), "reps": rep, "text": tok.decode(toks) if tok else ""}
json.dump(res, open(a.out, "w"))
print(json.dumps({k: res[k] for k in ("router_exact_ties", "router_gap_min", "reps")}), "min margin", min(marg), flush=True)
