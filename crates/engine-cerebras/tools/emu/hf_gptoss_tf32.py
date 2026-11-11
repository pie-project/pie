#!/usr/bin/env python3
"""hf_gptoss_tf32.py <model> <golden.json> <out.json>: the secondary reference for gpt-oss runs: HF transformers in
float32 (MXFP4 experts through HF's dequant, cast to f32), eager attention and experts, one teacher-forced
forward over the golden's prompt + tokens; per generated position: the argmax, the top-1 margin and the
golden token's gap (its logit minus the best other)."""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config
torch.set_num_threads(8)
model, gpath, out = sys.argv[1:4]
g = json.load(open(gpath))
ids = g["prompt_ids"] + g["tokens"]
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, quantization_config=Mxfp4Config(dequantize=True),
                                         attn_implementation="eager", experts_implementation="eager")
m.eval()
with torch.no_grad():
    lg = m(input_ids=torch.tensor([ids[:-1]])).logits[0].float()
NP = len(g["prompt_ids"])
res = {"argmax": [], "margins": [], "golden_gap": []}
for i, t in enumerate(g["tokens"]):
    row = lg[NP - 1 + i]
    v, ix = torch.topk(row, 2)
    res["argmax"].append(int(ix[0])); res["margins"].append(float(v[0] - v[1]))
    other = torch.cat([row[:t], row[t + 1:]]).max()
    res["golden_gap"].append(float(row[t] - other))
res["disagree"] = [i for i, (a, b) in enumerate(zip(res["argmax"], g["tokens"])) if a != b]
res["experts_dtype"] = str(m.model.layers[0].mlp.experts.gate_up_proj.dtype)
json.dump(res, open(out, "w"))
print(json.dumps({"disagree": res["disagree"][:20], "n": len(res["disagree"]), "s": round(time.time() - t0)}))
