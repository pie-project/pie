#!/usr/bin/env python3
"""Writes a random Qwen3.5-MoE snapshot at the `qwen3-a3b-micro` row's dimensions.

    python3 scripts/tiny_qwen3_moe.py <out-dir> --tokenizer <a Qwen3.5 snapshot dir>

Every layer is gated attention and a routed MLP of 32 experts (top 4) beside a
gated shared expert, in the Transformers spelling. The weights are small random
bf16 values, so the model serves any prompt and says nothing in particular. Its
vocabulary is 2048 tokens, so the tokenizer beside it is Qwen3.5's cut to them: its
first ids and the merges among them, its added tokens last. Then:

    pie model import <out-dir> --deployment qwen3-a3b-micro-bf16-kv-bf16
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

HIDDEN, LAYERS = 512, 4
Q_HEADS, KV_HEADS, HEAD_DIM = 8, 2, 64
EXPERTS, TOP_K, INTER, SHARED_INTER = 32, 4, 128, 128
VOCAB = 2048
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! How are you today? I am fine, thank you.",
    "Expert parallelism sends each token to the rank that holds its experts.",
    "0123456789 + - * / = ( ) [ ] { } < > : ; , . ! ? ' \" \n\t",
]


def bf16(values):
    """`values` (float32) as bf16 bytes, rounded to nearest even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()


def count(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def build():
    """(name, shape, value scale or None for ones) in write order."""
    p = "model.language_model."
    fan = 1.0 / (HIDDEN ** 0.5)
    out = [
        (p + "embed_tokens.weight", [VOCAB, HIDDEN], 0.02),
        (p + "norm.weight", [HIDDEN], None),
    ]
    for l in range(LAYERS):
        n = lambda leaf: f"{p}layers.{l}.{leaf}"
        out += [
            (n("input_layernorm.weight"), [HIDDEN], None),
            (n("post_attention_layernorm.weight"), [HIDDEN], None),
            (n("self_attn.q_proj.weight"), [2 * Q_HEADS * HEAD_DIM, HIDDEN], fan),
            (n("self_attn.k_proj.weight"), [KV_HEADS * HEAD_DIM, HIDDEN], fan),
            (n("self_attn.v_proj.weight"), [KV_HEADS * HEAD_DIM, HIDDEN], fan),
            (n("self_attn.o_proj.weight"), [HIDDEN, Q_HEADS * HEAD_DIM], fan),
            (n("self_attn.q_norm.weight"), [HEAD_DIM], None),
            (n("self_attn.k_norm.weight"), [HEAD_DIM], None),
            (n("mlp.gate.weight"), [EXPERTS, HIDDEN], fan),
            (n("mlp.experts.gate_up_proj"), [EXPERTS, 2 * INTER, HIDDEN], fan),
            (n("mlp.experts.down_proj"), [EXPERTS, HIDDEN, INTER], 1.0 / (INTER ** 0.5)),
            (n("mlp.shared_expert.gate_proj.weight"), [SHARED_INTER, HIDDEN], fan),
            (n("mlp.shared_expert.up_proj.weight"), [SHARED_INTER, HIDDEN], fan),
            (n("mlp.shared_expert.down_proj.weight"), [HIDDEN, SHARED_INTER], 1.0 / (SHARED_INTER ** 0.5)),
            (n("mlp.shared_expert_gate.weight"), [1, HIDDEN], fan),
        ]
    return out


def write_safetensors(path, tensors, rng):
    header, at = {}, 0
    for name, shape, _ in tensors:
        size = 2 * count(shape)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [at, at + size]}
        at += size
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * ((8 - len(head) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for _, shape, scale in tensors:
            if scale is None:
                f.write(bf16(np.ones(count(shape))))
            else:
                f.write(bf16(rng.uniform(-scale, scale, count(shape))))


def write_tokenizer(out, source):
    """Qwen's tokenizer cut to the micro's vocabulary: the BPE's first ids and
    the merges among them, Qwen's added tokens (ChatML's, `<think>`, ...) at
    the last ids."""
    with open(os.path.join(source, "tokenizer.json")) as f:
        tok = json.load(f)
    added = sorted(tok["added_tokens"], key=lambda token: token["id"])
    keep = VOCAB - len(added)
    vocab = {piece: id for piece, id in tok["model"]["vocab"].items() if id < keep}
    merges = []
    for merge in tok["model"]["merges"]:
        a, b = merge.split(" ", 1) if isinstance(merge, str) else merge
        if a in vocab and b in vocab and a + b in vocab:
            merges.append(merge)
    tok["model"]["vocab"] = vocab
    tok["model"]["merges"] = merges
    tok["added_tokens"] = [dict(token, id=keep + at) for at, token in enumerate(added)]
    with open(os.path.join(out, "tokenizer.json"), "w") as f:
        json.dump(tok, f)
    template = os.path.join(source, "chat_template.jinja")
    with open(template) as f:
        chat_template = f.read()
    with open(os.path.join(out, "chat_template.jinja"), "w") as f:
        f.write(chat_template)
    with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
        json.dump({
            "tokenizer_class": "Qwen2Tokenizer",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "added_tokens_decoder": {
                str(token["id"]): {"content": token["content"], "special": token["special"]}
                for token in tok["added_tokens"]
            },
        }, f, indent=1)
    return len(vocab) + len(added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokenizer", required=True,
                    help="a Qwen3.5 snapshot directory holding tokenizer.json and chat_template.jinja")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    write_safetensors(os.path.join(args.out, "model.safetensors"), build(), rng)
    vocab = write_tokenizer(args.out, args.tokenizer)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "vocab_size": VOCAB,
                "num_attention_heads": Q_HEADS, "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
                "num_experts": EXPERTS, "num_experts_per_tok": TOP_K,
                "moe_intermediate_size": INTER, "shared_expert_intermediate_size": SHARED_INTER,
                "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000000.0},
                "tie_word_embeddings": True,
                "note": "random weights at a miniature width; a test fixture, not a model",
            },
        }, f, indent=1)
    size = os.path.getsize(os.path.join(args.out, "model.safetensors")) >> 20
    print(f"wrote {args.out}: {size} MiB of weights, {vocab} tokens")


if __name__ == "__main__":
    sys.exit(main())
