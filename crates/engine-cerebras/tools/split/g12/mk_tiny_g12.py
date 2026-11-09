#!/usr/bin/env python3
"""mk_tiny_g12.py <out dir> [seed] [ntok]: a random tiny Gemma 4 (text, HF Gemma4ForCausalLM) in the 12B mapping's
shape for the tiny role-split simulation (split_geo_g12.tiny_geo: units 8 x 8, A rows 0-2 Hc 2, W rows 3-7 Hc 3, K 3,
4 units): hidden 20 (21 slots: one padded dim), 12 layers (every 3rd global), 4 q heads, 2 kv heads sliding (head 14:
16 slots), one global kv head (head 28: 32 slots, k = v, proportional RoPE 25%), MLP 30 (32 slots), no PLE, vocab
512, window 8 (a ring of 9 masked), tied, softcap 30.
Saves bf16 safetensors + config, and golden.json: HF f32 greedy (prompt + ntok tokens) with top-2 margins."""
import json, os, sys
import numpy as np
import torch
from transformers import Gemma4TextConfig, Gemma4ForCausalLM

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 28
torch.manual_seed(seed)
torch.set_num_threads(8)
os.makedirs(out, exist_ok=True)
NL = 12
cfg = Gemma4TextConfig(
    hidden_size=20, num_hidden_layers=NL, intermediate_size=30, num_attention_heads=4, num_key_value_heads=2,
    num_global_key_value_heads=1, head_dim=14, global_head_dim=28, sliding_window=int(os.environ.get("TWIN", 8)),
    num_kv_shared_layers=0, hidden_size_per_layer_input=0,
    vocab_size=512, vocab_size_per_layer_input=512, use_double_wide_mlp=False, attention_k_eq_v=True,
    layer_types=["full_attention" if l % 3 == 2 else "sliding_attention" for l in range(NL)],
    rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                     "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    final_logit_softcapping=30.0, tie_word_embeddings=True, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh",
    max_position_embeddings=4096, enable_moe_block=False, pad_token_id=0, bos_token_id=2, eos_token_id=1)
m = Gemma4ForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if "norm" in n:
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif "embed_tokens_per_layer" in n:
            p.copy_(0.3 * torch.randn_like(p))
        elif "embed_tokens" in n:
            p.copy_(0.6 * torch.randn_like(p))
        else:
            fan = p.shape[-1]
            p.copy_(torch.randn_like(p) / np.sqrt(fan))
    for l in m.model.layers:
        l.layer_scalar.copy_(0.6 + 0.8 * torch.rand(1))
m = m.to(torch.bfloat16)
m.save_pretrained(out, safe_serialization=True)
m32 = Gemma4ForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
g = torch.Generator().manual_seed(seed + 7)
prompt = [2] + torch.randint(3, 512, (5,), generator=g).tolist()
ids = list(prompt)
toks, margins = [], []
with torch.no_grad():
    for i in range(ntok):
        lo = m32(torch.tensor([ids])).logits[0, -1].float()
        top = torch.topk(lo, 2)
        t = int(top.indices[0])
        toks.append(t); margins.append(float(top.values[0] - top.values[1]))
        ids.append(t)
json.dump(dict(prompt_ids=prompt, tokens=toks, margins=margins, min_margin=min(margins)), open(os.path.join(out, "golden.json"), "w"))
print("tiny", out, "tokens", toks, "min margin", round(min(margins), 4))
