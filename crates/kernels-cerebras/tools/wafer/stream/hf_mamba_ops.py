"""hf_mamba_ops.py <nem1> <golden.json> <out.pt>: HF f32's per-op values of the one Mamba layer at every position
(in_proj's output, the conv's pre-activation output, the norm's input (the scan) and gate, its output,
out_proj's, the layer's)."""
import json, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
torch.set_num_threads(8)
src, gold, out = sys.argv[1:4]
g = json.load(open(gold)); seq = g["tokens"]
cfg = AutoConfig.from_pretrained(src); cfg.num_hidden_layers = 1
if getattr(cfg, "layer_types", None): cfg.layer_types = cfg.layer_types[:1]
m = AutoModelForCausalLM.from_pretrained(src, config=cfg, torch_dtype=torch.float32).eval()
mx = m.model.layers[0].mixer
print(type(mx).__name__, mx.time_step_limit if hasattr(mx, "time_step_limit") else None, mx.norm.variance_epsilon, mx.norm.group_size)
got = {}
def keep(name, which="out"):
    def f(_m, i, o):
        got[name] = (o[0] if isinstance(o, tuple) else o) if which == "out" else i
    return f
hs = [mx.in_proj.register_forward_hook(keep("inproj")),
      mx.norm.register_forward_hook(keep("norm_in", "in")), mx.norm.register_forward_hook(keep("y")),
      mx.out_proj.register_forward_hook(keep("mix")), m.model.layers[0].register_forward_hook(keep("hid"))]
with torch.no_grad():
    lg = m(torch.tensor([seq]), use_cache=False).logits[0]
r = {"inproj": got["inproj"][0], "scan": got["norm_in"][0][0], "gate": got["norm_in"][1][0],
     "y": got["y"][0], "mix": got["mix"][0], "hid": got["hid"][0], "logits": lg}
torch.save({k: v.detach().double() for k, v in r.items()}, out)
print({k: tuple(v.shape) for k, v in r.items()})
