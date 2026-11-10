#!/usr/bin/env python3
"""mk_tiny_gptoss.py <dir> [seed] [wsc]: a random tiny gpt-oss checkpoint in the release's format (MXFP4 experts:
blocks + E8M0 scales, bf16 elsewhere, the real config's mechanisms: sinks, alternating sliding / full layers,
YaRN, biases, clamped SwiGLU, top-4 of 8 experts) for the tiny gpt-oss geometry (`wafer-compile --geometry gptoss:tiny`): hidden 32 (4 rows x
Hc 8), 4 q / 2 kv heads of 8 (Dc 4), intermediate 32 (8 MoE columns x Ic 4), 4 layers (2 units x K 2), vocab
100, window 4. Then its golden: HF transformers greedy, bf16 compute on HF's MXFP4 dequant (hf_gptoss_golden.py)."""
import json, os, sys
import numpy as np
import torch
from safetensors.torch import save_file

d = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
wsc = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(seed)
H, L, NH, NKV, HD, E, TK, I, V, WIN = 32, 4, 4, 2, 8, 8, 4, 32, 100, 4
cfg = {"architectures": ["GptOssForCausalLM"], "attention_bias": True, "attention_dropout": 0.0, "eos_token_id": 99,
       "experts_per_token": TK, "head_dim": HD, "hidden_act": "silu", "hidden_size": H, "initial_context_length": 4096,
       "initializer_range": 0.02, "intermediate_size": I,
       "layer_types": ["sliding_attention", "full_attention"] * (L // 2), "max_position_embeddings": 131072,
       "model_type": "gpt_oss", "num_attention_heads": NH, "num_experts_per_tok": TK, "num_hidden_layers": L,
       "num_key_value_heads": NKV, "num_local_experts": E, "output_router_logits": False, "pad_token_id": 98,
       "quantization_config": {"modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"],
                               "quant_method": "mxfp4"},
       "rms_norm_eps": 1e-05, "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0, "original_max_position_embeddings": 4096,
                                               "rope_type": "yarn", "truncate": False},
       "rope_theta": 150000, "router_aux_loss_coef": 0.9, "sliding_window": WIN, "swiglu_limit": 7.0, "tie_word_embeddings": False,
       "transformers_version": "4.55.0.dev0", "use_cache": True, "vocab_size": V}
json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
T = {}
def bf(a):
    a = np.asarray(a, np.float32).copy()
    a[np.abs(a) < 2.0 ** -13] = 0.0          # (as the release's attention weights: every bf16 value an f16 normal)
    return torch.from_numpy(a).to(torch.bfloat16)
T["model.embed_tokens.weight"] = bf(rng.normal(0, 1.0, (V, H)))
T["lm_head.weight"] = bf(rng.normal(0, 0.3, (V, H)))
T["model.norm.weight"] = bf(rng.uniform(0.5, 1.5, H))
for l in range(L):
    p = f"model.layers.{l}."
    T[p + "input_layernorm.weight"] = bf(rng.uniform(0.5, 1.5, H))
    T[p + "post_attention_layernorm.weight"] = bf(rng.uniform(0.5, 1.5, H))
    for n, o in (("q", NH * HD), ("k", NKV * HD), ("v", NKV * HD)):
        T[p + f"self_attn.{n}_proj.weight"] = bf(rng.normal(0, wsc * 0.3, (o, H)))
        T[p + f"self_attn.{n}_proj.bias"] = bf(rng.normal(0, 0.1, o))
    T[p + "self_attn.o_proj.weight"] = bf(rng.normal(0, wsc * 0.15, (H, NH * HD)))
    T[p + "self_attn.o_proj.bias"] = bf(rng.normal(0, 0.05, H))
    T[p + "self_attn.sinks"] = bf(rng.normal(0, 1.0, NH))
    T[p + "mlp.router.weight"] = bf(rng.normal(0, 0.5, (E, H)))
    T[p + "mlp.router.bias"] = bf(rng.normal(0, 0.1, E))
    T[p + "mlp.experts.gate_up_proj_bias"] = bf(rng.normal(0, 0.1, (E, 2 * I)))
    T[p + "mlp.experts.down_proj_bias"] = bf(rng.normal(0, 0.05, (E, H)))
    # MXFP4: [E, out, in / 32, 16] code bytes (low nibble first), [E, out, in / 32] E8M0 scales
    T[p + "mlp.experts.gate_up_proj_blocks"] = torch.from_numpy(rng.integers(0, 256, (E, 2 * I, H // 32, 16), dtype=np.uint8))
    T[p + "mlp.experts.gate_up_proj_scales"] = torch.from_numpy(rng.integers(120, 125, (E, 2 * I, H // 32), dtype=np.uint8))
    T[p + "mlp.experts.down_proj_blocks"] = torch.from_numpy(rng.integers(0, 256, (E, H, I // 32, 16), dtype=np.uint8))
    T[p + "mlp.experts.down_proj_scales"] = torch.from_numpy(rng.integers(118, 124, (E, H, I // 32), dtype=np.uint8))
save_file(T, f"{d}/model.safetensors", metadata={"format": "pt"})
# a byte-level tokenizer is not needed: the golden takes ids
print("ok", d)
