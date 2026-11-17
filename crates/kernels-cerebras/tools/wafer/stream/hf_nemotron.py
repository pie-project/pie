"""hf_nemotron.py <nem slice> <layers> <old golden.json> <out.json>: the nemotron slice's golden as the release
computes it: HF transformers' native NemotronH with each Mamba-2 mixer's dt limit the release's
((0, inf): config mamba_dt_limit's default; transformers 5.23's port floors dt at time_step_min, 0.001).
Teacher-forced on the old golden's tokens (its prompt and continuation) and greedy from the prompt;
each step's top-5, and the last layer's output at every position (`hidden`)."""
import json, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
torch.set_num_threads(8)
src, layers, old, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
g = json.load(open(old)); prompt = g["prompt"]
cfg = AutoConfig.from_pretrained(src); cfg.num_hidden_layers = layers
if getattr(cfg, "layer_types", None): cfg.layer_types = cfg.layer_types[:layers]
m = AutoModelForCausalLM.from_pretrained(src, config=cfg, torch_dtype=torch.float32).eval()
for l in m.model.layers:
    if hasattr(l.mixer, "time_step_limit"):
        l.mixer.time_step_limit = (0.0, float("inf"))
seq = list(prompt); steps = []
with torch.no_grad():
    for _ in range(len(g["tokens"]) - len(prompt)):
        lg = m(torch.tensor([seq]), use_cache=False).logits[0, -1]
        top = torch.topk(lg, 5); steps.append([[int(i), float(v)] for i, v in zip(top.indices, top.values)])
        seq.append(int(top.indices[0]))
    # the old golden's sequence teacher-forced: each step's top-5 against its tokens
    lg = m(torch.tensor([g["tokens"]]), use_cache=False).logits[0]
    tf = [[[int(i), float(v)] for i, v in zip(torch.topk(lg[s], 5).indices, torch.topk(lg[s], 5).values)] for s in range(len(prompt) - 1, len(g["tokens"]) - 1)]
    got = {}
    h = m.model.layers[-1].register_forward_hook(lambda _m, _i, o: got.__setitem__("v", o[0] if isinstance(o, tuple) else o))
    m(torch.tensor([seq]), use_cache=False); hid = got["v"][0].float()
    m(torch.tensor([g["tokens"]]), use_cache=False); h.remove(); tfh = got["v"][0].float()
r = {"model": src, "layers": layers, "prompt": prompt, "tokens": seq, "top5": steps, "dt_limit": [0.0, "inf"],
     "teacher_forced": {"tokens": g["tokens"], "top5": tf,
     "hidden": [[float(x) for x in row] for row in tfh]},
     "value": "model.layers.%d" % (layers - 1), "hidden": [[float(x) for x in row] for row in hid]}
json.dump(r, open(out, "w"))
print("greedy", seq[len(prompt):])
print("old tokens' argmax", [s[0][0] for s in tf])
print("old tokens' top2 gap", [round(s[0][1] - s[1][1], 3) for s in tf])
