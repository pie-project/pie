#!/usr/bin/env python3
"""dense/mk_tiny_llama.py <outdir>: a random tiny Llama 3 as an HF checkpoint (bf16 safetensors + config) and its HF f32
greedy golden (prompt 6 + 14 tokens, margins), for the tiny cut (baseline/landed.py, wafer-stage) + the simulator. Dense attention with no q / k
norm, eps 1e-5 and Llama 3's scaled rotation at a short original length (16), so the stretched waves are exercised;
the shape fixture baseline/shapes/tiny_llama.json states the same model, its FREQS what this rotation turns at."""
import json, os, sys
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
torch.set_num_threads(8)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cfg = LlamaConfig(hidden_size=32, intermediate_size=32, num_hidden_layers=12, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=16, vocab_size=80, rms_norm_eps=1e-5, tie_word_embeddings=False, max_position_embeddings=128,
                  rope_parameters=dict(rope_type="llama3", rope_theta=500000.0, factor=8.0, low_freq_factor=1.0,
                                       high_freq_factor=4.0, original_max_position_embeddings=16))
torch.manual_seed(int(os.environ.get("SEED", "5")))
m = LlamaForCausalLM(cfg).float().eval()
WS = float(os.environ.get("WSC", "0.6"))
with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith("norm.weight"):
            p.copy_(1 + torch.randn_like(p) * 0.1)
        elif "embed_tokens" in n:
            p.copy_(torch.randn_like(p))
        elif p.dim() == 2:
            p.copy_(torch.randn_like(p) * WS / np.sqrt(p.shape[1]))
    for p in m.parameters():                                # the checkpoint is bf16: the reference uses the same values
        p.copy_(p.to(torch.bfloat16).float())
bufs = {n: b.clone() for n, b in m.named_buffers()}        # (the rotation's f32 frequencies: not the bf16 cast's)
m.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
m.float()
with torch.no_grad():
    for n, b in m.named_buffers():
        b.copy_(bufs[n])
rng = np.random.default_rng(1)
prompt = rng.integers(0, cfg.vocab_size, 6).tolist()
NG = int(os.environ.get("NGEN", "14"))
ids = list(prompt)
with torch.no_grad():
    for _ in range(NG):
        ids.append(int(m(torch.tensor([ids]), use_cache=False).logits[0, -1].argmax()))
    lg = m(torch.tensor([ids]), use_cache=False).logits[0]
toks = ids[len(prompt):]
pred = lg[len(prompt) - 1:len(prompt) - 1 + NG]
t2 = pred.topk(2, -1).values
margins = (t2[:, 0] - t2[:, 1]).tolist()
assert pred.argmax(-1).tolist() == toks
json.dump({"prompt_ids": prompt, "tokens": toks, "margins": margins}, open(os.path.join(out, "golden.json"), "w"))
inv = m.model.rotary_emb.inv_freq.float().tolist()
print("tokens", toks, "min margin", round(min(margins), 4))
print("FREQS", json.dumps(inv))
