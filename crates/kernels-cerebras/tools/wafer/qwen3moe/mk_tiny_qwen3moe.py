#!/usr/bin/env python3
"""mk_tiny_qwen3moe.py <dir> [seed]: a random tiny Qwen3 MoE checkpoint in the release's format (bf16, the real
config's mechanisms: per-head q / k RMSNorms, full attention every layer, top-8 of 16 experts, SiLU's GLU, no
biases or sinks) for the tiny MoE geometry (`wafer-compile plan` pinned plan `tiny-gptoss` with MX=16,SWR=16,HVC=40): hidden 32
(4 rows x Hc 8), 4 q / 2 kv heads of 8 (Dc 4), expert intermediate 32 (16 MoE columns x Ic 2), 4 layers (2 units
x K 2), vocab 100. Then its golden: HF transformers greedy, f32 compute over the bf16 weights."""
import json, os, sys
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

d = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
os.makedirs(d, exist_ok=True)
torch.set_num_threads(8)
rng = np.random.default_rng(seed)
H, L, NH, NKV, HD, E, TK, I, V = 32, 4, 4, 2, 8, 16, 8, 32, 100
cfg = Qwen3MoeConfig(vocab_size=V, hidden_size=H, intermediate_size=4 * I, moe_intermediate_size=I, num_hidden_layers=L,
                     num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD, num_experts=E, num_experts_per_tok=TK,
                     norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[], rms_norm_eps=1e-6, rope_theta=1e6,
                     max_position_embeddings=4096, tie_word_embeddings=False, attention_bias=False, use_sliding_window=False,
                     torch_dtype="bfloat16", eos_token_id=99, bos_token_id=98)
cfg.architectures = ["Qwen3MoeForCausalLM"]
cfg.save_pretrained(d)
T = {}
def bf(a):
    a = np.asarray(a, np.float32).copy()
    a[np.abs(a) < 2.0 ** -13] = 0.0          # every bf16 value an f16 normal
    return torch.from_numpy(a).to(torch.bfloat16)
T["model.embed_tokens.weight"] = bf(rng.normal(0, 1.0, (V, H)))
T["lm_head.weight"] = bf(rng.normal(0, 0.3, (V, H)))
T["model.norm.weight"] = bf(rng.uniform(0.5, 1.5, H))
for l in range(L):
    p = f"model.layers.{l}."
    T[p + "input_layernorm.weight"] = bf(rng.uniform(0.5, 1.5, H))
    T[p + "post_attention_layernorm.weight"] = bf(rng.uniform(0.5, 1.5, H))
    for n, o in (("q", NH * HD), ("k", NKV * HD), ("v", NKV * HD)):
        T[p + f"self_attn.{n}_proj.weight"] = bf(rng.normal(0, 0.3, (o, H)))
    T[p + "self_attn.q_norm.weight"] = bf(rng.uniform(0.5, 1.5, HD))
    T[p + "self_attn.k_norm.weight"] = bf(rng.uniform(0.5, 1.5, HD))
    T[p + "self_attn.o_proj.weight"] = bf(rng.normal(0, 0.15, (H, NH * HD)))
    T[p + "mlp.gate.weight"] = bf(rng.normal(0, 0.5, (E, H)))
    for e in range(E):
        q = p + f"mlp.experts.{e}."
        T[q + "gate_proj.weight"] = bf(rng.normal(0, 0.3, (I, H)))
        T[q + "up_proj.weight"] = bf(rng.normal(0, 0.3, (I, H)))
        T[q + "down_proj.weight"] = bf(rng.normal(0, 0.15, (H, I)))
save_file(T, f"{d}/model.safetensors", metadata={"format": "pt"})

m = Qwen3MoeForCausalLM.from_pretrained(d, torch_dtype=torch.float32).eval()
prompt = [3, 17, 42, 7, 11]
got, margins, routers = [], [], []
with torch.no_grad():
    o = m(torch.tensor([prompt]), use_cache=True)
    past, logits = o.past_key_values, o.logits[0, -1]
    first = torch.topk(logits, 8).indices.tolist()
    for i in range(14):
        top = torch.topk(logits, 2)
        t = int(top.indices[0]); got.append(t); margins.append(float(top.values[0] - top.values[1]))
        o = m(torch.tensor([[t]]), past_key_values=past, use_cache=True)
        past, logits = o.past_key_values, o.logits[0, -1]
json.dump({"name": f"q3moe-s{seed}", "golden": "HF transformers 4.51 Qwen3MoeForCausalLM, f32 compute on the bf16 weights",
           "prompt_ids": prompt, "tokens": got, "margins": margins, "first_logits_top": first, "reps": [], "text": ""},
          open(f"{d}/golden.json", "w"))
print("ok", d, "tokens", got, "min margin", min(margins))
