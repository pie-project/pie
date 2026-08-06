#!/usr/bin/env python3
"""Dump mlx-lm's per-kernel activations under the same names the Metal driver's
PIE_METAL_GOLDEN_DIR hook writes, so cosine_bisect.py can diff them.

mlx-lm is the oracle here: it is the implementation that demonstrably produces
coherent text on these checkpoints. driver/metal/tests/mlx is a second reference,
but it has drifted from the family it models, so parity should be judged against
mlx-lm.

Usage: qwen3_5_mlx_taps.py <model_path> <comma_token_ids> <out_dir>
"""
import sys

import mlx.core as mx
import mlx.nn as nn
from taps_common import open_taps


def main():
    model, ids, dump, out_dir = open_taps(sys.argv)
    lm = model.language_model.model
    head = getattr(model.language_model, "lm_head", None)

    # The three projections that end a residual branch -- `gdn_out`, `o_proj`
    # and `down_proj` -- are published here with their residual ALREADY ADDED,
    # because that is what the driver publishes under those names. It fuses the
    # add into the projection's epilogue (`affine_qmv_fast_residual`,
    # `MetalExecutor::Impl::fuse_residual_`), so the value that reaches the tap
    # is the sum and there is no separate dispatch to tap instead. Dumping the
    # bare projection here reads as a real divergence -- cosine 0.81 and 0.47 on
    # Qwen3.5-0.8B -- that is entirely the residual the reference left out.
    x = mx.array([ids])
    h = lm.embed_tokens(x)
    dump("embed", h)

    # Full causal mask over the prompt; the linear layers take no mask for a
    # single unpadded request.
    n = len(ids)
    mask = mx.triu(mx.full((n, n), -mx.inf, dtype=h.dtype), k=1)

    for il, layer in enumerate(lm.layers):
        cur = layer.input_layernorm(h)
        dump(f"{il}.attn_norm", cur)
        if layer.is_linear:
            la = layer.linear_attn
            B, S = 1, n
            qkv = la.in_proj_qkv(cur)
            z = la.in_proj_z(cur)
            b = la.in_proj_b(cur)
            a = la.in_proj_a(cur)
            dump(f"{il}.gdn_in_qkv", qkv)
            dump(f"{il}.gdn_in_z", z)
            dump(f"{il}.gdn_in_a", a)
            dump(f"{il}.gdn_in_b", b)

            conv_state = mx.zeros(
                (B, la.conv_kernel_size - 1, la.conv_dim), dtype=cur.dtype)
            conv_out = nn.silu(la.conv1d(mx.concatenate([conv_state, qkv], axis=1)))
            q, k, v = [
                t.reshape(B, S, hh, d)
                for t, hh, d in zip(
                    mx.split(conv_out, [la.key_dim, 2 * la.key_dim], -1),
                    [la.num_k_heads, la.num_k_heads, la.num_v_heads],
                    [la.head_k_dim, la.head_k_dim, la.head_v_dim],
                )
            ]
            inv = k.shape[-1] ** -0.5
            q = (inv ** 2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv * mx.fast.rms_norm(k, None, 1e-6)
            from mlx_lm.models.gated_delta import gated_delta_update
            out, _ = gated_delta_update(
                q, k, v, a, b, la.A_log, la.dt_bias, None, None, use_kernel=True)
            out = la.norm(out, z.reshape(B, S, la.num_v_heads, la.head_v_dim))
            dump(f"{il}.gdn_core", out)
            r = la.out_proj(out.reshape(B, S, -1))
            dump(f"{il}.gdn_out", h + r)
        else:
            at = layer.self_attn
            B, S = 1, n
            qg = at.q_proj(cur)
            dump(f"{il}.q_proj", qg)
            queries, gate = mx.split(
                qg.reshape(B, S, at.num_attention_heads, -1), 2, axis=-1)
            gate = gate.reshape(B, S, -1)
            keys, values = at.k_proj(cur), at.v_proj(cur)
            dump(f"{il}.k_proj", keys)
            dump(f"{il}.v_proj", values)
            queries = at.q_norm(queries)
            keys_n = at.k_norm(keys.reshape(B, S, at.num_key_value_heads, -1))
            dump(f"{il}.q_norm", queries)
            dump(f"{il}.k_norm", keys_n)
            queries = at.rope(queries.transpose(0, 2, 1, 3))
            keys_r = at.rope(keys_n.transpose(0, 2, 1, 3))
            dump(f"{il}.rope_q", queries.transpose(0, 2, 1, 3))
            dump(f"{il}.rope_k", keys_r.transpose(0, 2, 1, 3))
            vals = values.reshape(B, S, at.num_key_value_heads, -1).transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(
                queries, keys_r, vals, scale=at.scale, mask=mask)
            o = o.transpose(0, 2, 1, 3).reshape(B, S, -1)
            dump(f"{il}.sdpa", o)
            o = o * mx.sigmoid(gate)
            dump(f"{il}.attn_gated", o)
            r = at.o_proj(o)
            dump(f"{il}.o_proj", h + r)

        h = h + r
        dump(f"{il}.attn_resid", h)
        normed = layer.post_attention_layernorm(h)
        dump(f"{il}.ffn_norm", normed)
        g = layer.mlp.gate_proj(normed)
        u = layer.mlp.up_proj(normed)
        dump(f"{il}.gate_proj", g)
        dump(f"{il}.up_proj", u)
        act = nn.silu(g) * u
        dump(f"{il}.swiglu", act)
        d = layer.mlp.down_proj(act)
        dump(f"{il}.down_proj", h + d)
        h = h + d
        dump(f"{il}.layer_out", h)

    h = lm.norm(h)
    dump("final_norm", h)
    logits = head(h) if head is not None else lm.embed_tokens.as_linear(h)
    dump("logits", logits)
    print("wrote taps ->", out_dir)


if __name__ == "__main__":
    main()
