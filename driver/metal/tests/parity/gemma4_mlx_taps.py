#!/usr/bin/env python3
"""Dump mlx-lm's per-kernel gemma4 activations under the names the Metal driver's
gemma4 tap hook writes, so cosine_bisect.py --family gemma4 can diff them.

mlx-lm is the oracle: it is the implementation that demonstrably produces
coherent text on this checkpoint. This re-runs its `gemma4_text.py` forward pass
inline rather than calling it, because the taps are the intermediates a single
`model(ids)` call does not expose.

The names mirror `tests/gemma4_forward_test.cpp`'s `tap_for`. The driver folds
several taps into one buffer (its in-place kinds share a value), so it emits
fewer names than this does; cosine_bisect skips whatever only one side has.

Usage: gemma4_mlx_taps.py <model_path> <comma_token_ids> <out_dir>
"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from taps_common import open_taps


def main():
    top, ids, dump, out_dir = open_taps(sys.argv)
    lm = top.language_model
    m = lm.model
    cfg = m.config

    x = mx.array([ids])
    n = len(ids)

    # ── embedding, and the per-layer-embedding stream ──
    h = m.embed_tokens(x) * m.embed_scale
    dump("embed", h)

    # The PLE stream is optional. E2B/E4B carry it; the 26B mixture sets
    # `hidden_size_per_layer_input` to 0 and has none of these modules, so
    # reaching for them is an AttributeError rather than an empty tensor.
    has_ple = bool(getattr(m, "hidden_size_per_layer_input", 0))
    ple = None
    if has_ple:
        ple_tok = m.embed_tokens_per_layer(x) * m.embed_tokens_per_layer_scale
        dump("ple_tok", ple_tok)
        ple_tok = mx.unflatten(
            ple_tok, -1, (cfg.num_hidden_layers, m.hidden_size_per_layer_input))

        ple_proj = m.per_layer_model_projection(h) * m.per_layer_projection_scale
        dump("ple_proj", ple_proj)
        ple_proj = mx.unflatten(
            ple_proj, -1, (cfg.num_hidden_layers, m.hidden_size_per_layer_input))
        ple_proj = m.per_layer_projection_norm(ple_proj)
        dump("ple_projnorm", ple_proj)

        ple = (ple_proj + ple_tok) * m.per_layer_input_scale
        dump("ple", ple)

    # ── the stack ──
    # Position 0 of an empty cache: the causal mask is a no-op for one token, and
    # a real prompt gets the full triangular one.
    mask = mx.triu(mx.full((n, n), -mx.inf, dtype=h.dtype), k=1) if n > 1 else None
    kvs = [None] * len(m.layers)

    for il, layer in enumerate(m.layers):
        at = layer.self_attn
        B, S, hd = 1, n, at.head_dim

        cur = layer.input_layernorm(h)
        dump(f"{il}.attn_norm", cur)

        q = at.q_proj(cur).reshape(B, S, at.n_heads, hd)
        dump(f"{il}.q_proj", q)
        q = at.q_norm(q)
        dump(f"{il}.q_norm", q)

        if at.has_kv:
            k = at.k_proj(cur).reshape(B, S, at.n_kv_heads, hd)
            dump(f"{il}.k_proj", k)
            # K-eq-V: the full-attention layers of the 26B mixture ship no
            # `v_proj` and take V from the k projection BEFORE k-norm and rope.
            v = k if at.use_k_eq_v else at.v_proj(cur).reshape(B, S, at.n_kv_heads, hd)
            if not at.use_k_eq_v:
                dump(f"{il}.v_proj", v)
            k = at.k_norm(k)
            dump(f"{il}.k_norm", k)
            k = at.rope(k.transpose(0, 2, 1, 3))
            dump(f"{il}.rope_k", k.transpose(0, 2, 1, 3))
            v = at.v_norm(v)
            dump(f"{il}.v_norm", v)
            v = v.transpose(0, 2, 1, 3)
            kvs[il] = (k, v)
        else:
            k, v = kvs[m.previous_kvs[il]]

        q = at.rope(q.transpose(0, 2, 1, 3))
        dump(f"{il}.rope_q", q.transpose(0, 2, 1, 3))

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=at.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, S, -1)
        dump(f"{il}.sdpa", o)
        o = at.o_proj(o)
        dump(f"{il}.o_proj", o)

        o = layer.post_attention_layernorm(o)
        dump(f"{il}.attn_postnorm", o)
        h = h + o
        dump(f"{il}.attn_resid", h)

        resid = h
        cur = layer.pre_feedforward_layernorm(h)
        dump(f"{il}.ffn_norm", cur)
        gate = layer.mlp.gate_proj(cur)
        up = layer.mlp.up_proj(cur)
        dump(f"{il}.gate_proj", gate)
        dump(f"{il}.up_proj", up)
        act = nn.gelu_approx(gate) * up
        dump(f"{il}.geglu", act)
        d = layer.mlp.down_proj(act)
        dump(f"{il}.down_proj", d)

        # The mixture sits BESIDE the dense FFN rather than replacing it: both
        # branches read the post-attention residual, each closes with its own
        # norm, and the sum goes through the layer's shared closing norm.
        if getattr(layer, "enable_moe", False):
            d = layer.post_feedforward_layernorm_1(d)
            dump(f"{il}.g4_dense_br", d)

            r = layer.router
            rn = mx.fast.rms_norm(h, r.scale * r._root_size, r.eps)
            dump(f"{il}.g4_router_n", rn)
            scores = r.proj(rn)
            dump(f"{il}.g4_router", scores)
            if os.environ.get("PIE_TIEBREAK"):
                # The router's scores are bf16 over 128 experts, so exact ties
                # at the top-k boundary are common -- and mlx's `argpartition`
                # breaks them by an order nothing defines. When the point is to
                # compare ARITHMETIC, ties are noise: this takes the lowest
                # index, which is what the driver's top-k does.
                k = r.config.top_k_experts
                idx = mx.argsort(-scores.astype(mx.float32), axis=-1)[..., :k]
                w = mx.softmax(mx.take_along_axis(
                    scores.astype(mx.float32), idx, axis=-1), axis=-1)
                w = w * r.per_expert_scale[idx]
            else:
                idx, w = r(h)

            h2 = layer.pre_feedforward_layernorm_2(h)
            dump(f"{il}.g4_moe_n", h2)
            h2 = layer.experts(h2, idx, w)
            dump(f"{il}.g4_moe_out", h2)
            h2 = layer.post_feedforward_layernorm_2(h2)
            dump(f"{il}.g4_moe_br", h2)

            d = d + h2
            dump(f"{il}.g4_branches", d)

        d = layer.post_feedforward_layernorm(d)
        dump(f"{il}.ffn_postnorm", d)
        h = resid + d
        dump(f"{il}.ffn_resid", h)

        if not has_ple:
            h = h * layer.layer_scalar
            dump(f"{il}.layer_out", h)
            continue

        g = layer.per_layer_input_gate(h)
        dump(f"{il}.ple_gate", g)
        g = nn.gelu_approx(g) * ple[:, :, il, :]
        dump(f"{il}.ple_act", g)
        g = layer.per_layer_projection(g)
        dump(f"{il}.ple_back", g)
        g = layer.post_per_layer_input_norm(g)
        dump(f"{il}.ple_norm", g)
        h = h + g
        dump(f"{il}.ple_resid", h)

        h = h * layer.layer_scalar
        dump(f"{il}.layer_out", h)

    h = m.norm(h)
    dump("final_norm", h)
    logits = m.embed_tokens.as_linear(h)
    dump("logits_raw", logits)
    cap = lm.final_logit_softcapping
    dump("logits", mx.tanh(logits / cap) * cap if cap else logits)
    print("wrote gemma4 taps ->", out_dir)


if __name__ == "__main__":
    main()
