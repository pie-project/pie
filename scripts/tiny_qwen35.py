#!/usr/bin/env python3
"""Writes a random Qwen3.5 snapshot the `qwen35-tiny` row imports.

    python3 scripts/tiny_qwen35.py <out-dir> [--tokenizer <dir-with-tokenizer.json>]

The tensors follow the Transformers spelling `crates/models/src/qwen_3/import.rs`
reads, at the dimensions `Model::tiny` declares (crates/models/src/qwen_3/model.rs).
The weights are small random bf16 values, so the model serves any prompt and says
nothing in particular; the tokenizer beside it is the real Qwen3.5 one, fetched from
Hugging Face when no `--tokenizer` directory is given. Then:

    pie model import <out-dir> --sku qwen35-tiny
"""
import argparse
import json
import os
import random
import struct
import sys
import urllib.request

HIDDEN, LAYERS, ATTN_EVERY = 256, 4, 4
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 64
K_HEADS, V_HEADS, K_DIM, V_DIM, CONV = 4, 8, 128, 128, 4
INTER, VOCAB = 512, 248_320
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
HF = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/"


def bf16(values):
    out = bytearray(len(values) * 2)
    for i, v in enumerate(values):
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
        struct.pack_into("<H", out, 2 * i, bits >> 16)
    return bytes(out)


def tensor(shape, scale, rng, dtype="BF16"):
    n = 1
    for d in shape:
        n *= d
    if dtype == "F32":
        return struct.pack(f"<{n}f", *[rng.uniform(-scale, scale) for _ in range(n)])
    return bf16([rng.uniform(-scale, scale) for _ in range(n)])


def ones(shape, dtype="BF16"):
    n = 1
    for d in shape:
        n *= d
    return struct.pack(f"<{n}f", *([1.0] * n)) if dtype == "F32" else bf16([1.0] * n)


def build(rng):
    """(name, shape, dtype, bytes) in write order."""
    p = "model.language_model."
    out = []
    fan = 1.0 / (HIDDEN ** 0.5)
    out.append((p + "embed_tokens.weight", [VOCAB, HIDDEN], "BF16", None))
    out.append((p + "norm.weight", [HIDDEN], "BF16", ones([HIDDEN])))
    qkv = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM
    for l in range(LAYERS):
        n = lambda leaf: f"{p}layers.{l}.{leaf}"
        out.append((n("input_layernorm.weight"), [HIDDEN], "BF16", ones([HIDDEN])))
        out.append((n("post_attention_layernorm.weight"), [HIDDEN], "BF16", ones([HIDDEN])))
        if (l + 1) % ATTN_EVERY == 0:
            out.append((n("self_attn.q_proj.weight"), [2 * Q_HEADS * HEAD_DIM, HIDDEN], "BF16", tensor([2 * Q_HEADS * HEAD_DIM, HIDDEN], fan, rng)))
            out.append((n("self_attn.k_proj.weight"), [KV_HEADS * HEAD_DIM, HIDDEN], "BF16", tensor([KV_HEADS * HEAD_DIM, HIDDEN], fan, rng)))
            out.append((n("self_attn.v_proj.weight"), [KV_HEADS * HEAD_DIM, HIDDEN], "BF16", tensor([KV_HEADS * HEAD_DIM, HIDDEN], fan, rng)))
            out.append((n("self_attn.o_proj.weight"), [HIDDEN, Q_HEADS * HEAD_DIM], "BF16", tensor([HIDDEN, Q_HEADS * HEAD_DIM], fan, rng)))
            out.append((n("self_attn.q_norm.weight"), [HEAD_DIM], "BF16", ones([HEAD_DIM])))
            out.append((n("self_attn.k_norm.weight"), [HEAD_DIM], "BF16", ones([HEAD_DIM])))
        else:
            out.append((n("linear_attn.in_proj_qkv.weight"), [qkv, HIDDEN], "BF16", tensor([qkv, HIDDEN], fan, rng)))
            out.append((n("linear_attn.in_proj_z.weight"), [V_HEADS * V_DIM, HIDDEN], "BF16", tensor([V_HEADS * V_DIM, HIDDEN], fan, rng)))
            out.append((n("linear_attn.in_proj_b.weight"), [V_HEADS, HIDDEN], "BF16", tensor([V_HEADS, HIDDEN], fan, rng)))
            out.append((n("linear_attn.in_proj_a.weight"), [V_HEADS, HIDDEN], "BF16", tensor([V_HEADS, HIDDEN], fan, rng)))
            out.append((n("linear_attn.conv1d.weight"), [qkv, 1, CONV], "BF16", tensor([qkv, 1, CONV], 0.5, rng)))
            out.append((n("linear_attn.dt_bias"), [V_HEADS], "BF16", tensor([V_HEADS], 0.5, rng)))
            out.append((n("linear_attn.A_log"), [V_HEADS], "F32", tensor([V_HEADS], 1.0, rng, "F32")))
            out.append((n("linear_attn.norm.weight"), [V_DIM], "BF16", ones([V_DIM])))
            out.append((n("linear_attn.out_proj.weight"), [HIDDEN, V_HEADS * V_DIM], "BF16", tensor([HIDDEN, V_HEADS * V_DIM], fan, rng)))
        out.append((n("mlp.gate_proj.weight"), [INTER, HIDDEN], "BF16", tensor([INTER, HIDDEN], fan, rng)))
        out.append((n("mlp.up_proj.weight"), [INTER, HIDDEN], "BF16", tensor([INTER, HIDDEN], fan, rng)))
        out.append((n("mlp.down_proj.weight"), [HIDDEN, INTER], "BF16", tensor([HIDDEN, INTER], fan, rng)))
    return out


def write_safetensors(path, tensors, rng):
    header = {}
    at = 0
    sizes = []
    for name, shape, dtype, data in tensors:
        n = 1
        for d in shape:
            n *= d
        size = n * (4 if dtype == "F32" else 2)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + size]}
        sizes.append(size)
        at += size
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * ((8 - len(head) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for (name, shape, dtype, data), size in zip(tensors, sizes):
            if data is None:
                scale = 0.02
                for _ in range(shape[0]):
                    f.write(bf16([rng.uniform(-scale, scale) for _ in range(shape[1])]))
            else:
                f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokenizer", help="directory holding tokenizer.json (else fetched from Hugging Face)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    write_safetensors(os.path.join(args.out, "model.safetensors"), build(rng), rng)
    for name in TOKENIZER_FILES:
        dst = os.path.join(args.out, name)
        if args.tokenizer:
            src = os.path.join(args.tokenizer, name)
            if os.path.exists(src):
                with open(src, "rb") as i, open(dst, "wb") as o:
                    o.write(i.read())
        else:
            urllib.request.urlretrieve(HF + name, dst)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "vocab_size": VOCAB,
                "num_attention_heads": Q_HEADS, "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
                "intermediate_size": INTER, "tie_word_embeddings": True,
                "note": "random weights at a miniature width; a test fixture, not a model",
            },
        }, f, indent=1)
    print(f"wrote {args.out}: {os.path.getsize(os.path.join(args.out, 'model.safetensors')) >> 20} MiB of weights")


if __name__ == "__main__":
    sys.exit(main())
