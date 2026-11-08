#!/usr/bin/env python3
"""mk_tinyhf.py <outdir> <kind> : a random tiny Qwen3.5 as an HF checkpoint (bf16 safetensors + config),
its HF f32 greedy golden (prompt 6 + 14 tokens, margins) and HF's per-head-row max logits without the
final rms scale (golden.json "rowmax": what the chip's res rows hold), for prep_q35.py + the simulator.
kind t4 : the 4B / 9B column mapping (v heads = 2 k heads, half the rows hold GDN state, Qh 4, odd Hc)
kind t08: the 0.8B mapping (v heads = k heads, every row holds state, Dc 2 with Qh 2)"""
import json, os, sys
import numpy as np
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
torch.set_num_threads(8)
out, kind = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
K = dict(
    t4=dict(hidden_size=48, intermediate_size=64, num_hidden_layers=16, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=8, linear_value_head_dim=16, vocab_size=144),
    t08=dict(hidden_size=24, intermediate_size=24, num_hidden_layers=24, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
             linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=16, vocab_size=80),
)[kind]
cfg = Qwen3_5TextConfig(**K, linear_conv_kernel_dim=4, tie_word_embeddings=True, rms_norm_eps=1e-6,
                        rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25),
                        layer_types=["full_attention" if (l % 4) == 3 else "linear_attention" for l in range(K["num_hidden_layers"])])
torch.manual_seed(int(os.environ.get("SEED", "5")))
m = Qwen3_5ForCausalLM(cfg).float().eval()
WS = float(os.environ.get("WSC", "0.6"))
with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith("norm.weight") and "linear_attn.norm" not in n:
            p.copy_(torch.randn_like(p) * 0.1)              # (1 + w) norms
        elif "linear_attn.norm.weight" in n:
            p.copy_(1 + torch.randn_like(p) * 0.1)
        elif "conv1d" in n:
            p.copy_(torch.randn_like(p) * 0.5)
        elif n.endswith("A_log") or n.endswith("dt_bias"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif "embed_tokens" in n:
            p.copy_(torch.randn_like(p) * float(os.environ.get("ESC", "1")))
        elif p.dim() == 2:
            p.copy_(torch.randn_like(p) * WS / np.sqrt(p.shape[1]))
    for p in m.parameters():                                # the checkpoint is bf16: the reference uses the same values
        p.copy_(p.to(torch.bfloat16).float())
m.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
m.float()
cfgj = json.load(open(os.path.join(out, "config.json")))
cfgj["tie_word_embeddings"] = True
json.dump(cfgj, open(os.path.join(out, "config.json"), "w"), indent=1)
rng = np.random.default_rng(1)
prompt = rng.integers(0, K["vocab_size"], 6).tolist()
NG = int(os.environ.get("NGEN", "14"))
ids = list(prompt)
with torch.no_grad():
    for _ in range(NG):
        lg = m(torch.tensor([ids]), use_cache=False).logits[0, -1]
        ids.append(int(lg.argmax()))
    cap = {}
    h = m.model.norm.register_forward_hook(lambda mod, inp, o: cap.__setitem__("x", inp[0][0].float()))
    lg = m(torch.tensor([ids]), use_cache=False).logits[0]
    h.remove()
toks = ids[len(prompt):]
pred = lg[len(prompt) - 1:len(prompt) - 1 + NG]
t2 = pred.topk(2, -1).values
margins = (t2[:, 0] - t2[:, 1]).tolist()
assert pred.argmax(-1).tolist() == toks
x = cap["x"][len(prompt) - 1:len(prompt) - 1 + NG]
rs = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
un = pred / rs                                              # W (x (1 + w)), the head's res rows
VR = 8
V = K["vocab_size"]
rowmax = un.reshape(NG, V // VR, VR).max(-1).values.tolist()
json.dump({"prompt_ids": prompt, "tokens": toks, "margins": margins, "rowmax": rowmax}, open(os.path.join(out, "golden.json"), "w"))
print(kind, "tokens", toks, "min margin", round(min(margins), 4))
