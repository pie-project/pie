"""The qwen4 parity fixture: a seeded random micro model in the reference
implementation, its checkpoint written the way `qwen_4::import` reads one,
and its logits written the way the engine gate compares them.

Numbers mirror `Model::flash_micro` field for field; the indexer budget is
set above every sequence this fixture runs so QSA selection is the full
causal mask and the logits are exact (`qwen_4::model::Mixer::Attn`'s doc)."""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers.models.qwen4_exp import Qwen4ExpTextConfig, Qwen4ExpForCausalLM

OUT = os.path.join(os.path.dirname(__file__), "fixture")

config = Qwen4ExpTextConfig(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=4,
    layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    partial_rotary_factor=0.25,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [4, 2, 2],
        "mrope_interleaved": True,
    },
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    norm_topk_prob=True,
    hc_count=4,
    hc_lowrank=16,
    ple_layer_ids=[3],
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=1000,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=128,
    seed=1234,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=4096,
    indexer_compress_ratio=4,
    output_gate_type="sigmoid",
    rms_norm_eps=1e-6,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    eos_token_id=3,
    bos_token_id=3,
    pad_token_id=None,
    max_position_embeddings=4096,
    use_cache=True,
)

torch.manual_seed(7)
model = Qwen4ExpForCausalLM(config).to(torch.bfloat16).eval()

# ── the checkpoint, in the spelling the import reads ─────────────────────
os.makedirs(OUT, exist_ok=True)
state = model.state_dict()
renamed = {}
for name, tensor in state.items():
    if name == "lm_head.weight":
        renamed[name] = tensor.contiguous()
    elif name.endswith("ple_embedding.ngram_embedding.weight"):
        # `save_pretrained` restores the sharded layout the checkpoint is
        # bound to (`split_ngram_parts`); a bare `state_dict` holds the
        # merged runtime table, so the split is redone here.
        stem = "model.language_model." + name[len("model."):]
        stem = stem[: -len(".weight")]
        parts = config.split_ngram_parts
        rows = tensor.shape[0] // parts
        for i in range(parts):
            renamed[f"{stem}.shard_{i}.weight"] = (
                tensor[i * rows : (i + 1) * rows].contiguous()
            )
    elif name.startswith("model."):
        renamed["model.language_model." + name[len("model."):]] = tensor.contiguous()
    else:
        raise SystemExit(f"unplaced tensor {name}")
save_file(renamed, os.path.join(OUT, "model.safetensors"))
index = {
    "metadata": {"total_size": sum(t.numel() * t.element_size() for t in renamed.values())},
    "weight_map": {k: "model.safetensors" for k in renamed},
}
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=1)
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump({"model_type": "qwen4_exp_text", **config.to_diff_dict()}, f, indent=1, default=str)

# ── the prompts: an eos mid-stream exercises the hasher's segmentation ───
PROMPTS = {
    "plain": [11, 250, 42, 7, 199, 23, 88, 154, 9, 61, 77, 130],
    "eos_split": [10, 42, 3, 77, 5, 3, 200, 201, 202, 15],
    "short": [128, 64],
}

out = {}
for name, tokens in PROMPTS.items():
    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
    out[f"{name}/tokens"] = np.array(tokens, dtype=np.int32)
    out[f"{name}/logits"] = logits.numpy()

    # And the greedy continuation, decoded step by step through the cache —
    # the decode arm's own path — with every step's logits kept, so the gate
    # can teacher-force this path and compare rows instead of argmaxes (a
    # near-tie flipped by bf16 accumulation order would cascade a token
    # comparison into noise while every row still agrees).
    with torch.no_grad():
        past = None
        cur = ids
        produced = []
        rows = []
        for _ in range(16):
            res = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = res.past_key_values
            row = res.logits[0, -1].float()
            rows.append(row.numpy())
            nxt = int(row.argmax())
            produced.append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.long)
    out[f"{name}/greedy"] = np.array(produced, dtype=np.int32)
    out[f"{name}/step_logits"] = np.stack(rows)

# And the same facts as JSON, which is what the Rust gate parses: the last
# position's logits (the only row a fire hands back per lane) and the greedy
# continuation.
doc = {}
for name, tokens in PROMPTS.items():
    doc[name] = {
        "tokens": tokens,
        "all_logits": [[float(v) for v in row] for row in out[f"{name}/logits"]],
        "last_logits": [float(v) for v in out[f"{name}/logits"][-1]],
        "greedy": [int(v) for v in out[f"{name}/greedy"]],
        "step_logits": [[float(v) for v in row] for row in out[f"{name}/step_logits"]],
    }
with open(os.path.join(OUT, "reference.json"), "w") as f:
    json.dump(doc, f)
print("fixture written to", OUT)
for name in PROMPTS:
    print(name, "greedy:", out[f"{name}/greedy"].tolist())
