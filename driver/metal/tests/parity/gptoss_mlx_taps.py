#!/usr/bin/env python3
"""Dump mlx-lm's per-kernel gpt-oss activations under the names the Metal
driver's gpt-oss tap hook writes, so cosine_bisect.py --family gptoss can diff
them.

mlx-lm is the oracle: it is the implementation that demonstrably produces
coherent text on this checkpoint. This re-runs its `gpt_oss.py` forward pass
inline, because the taps are the intermediates a single `model(ids)` call does
not expose.

The MoE taps are the k-wide stacks the driver actually computes -- one row per
SELECTED expert, in the order the router chose them -- not the dense
`num_experts`-wide tensors mlx's SwitchGLU hides internally.

Usage: gptoss_mlx_taps.py <model_path> <comma_token_ids> <out_dir>
"""
import sys

import mlx.core as mx
from taps_common import open_taps
from mlx_lm.models.gpt_oss import mlx_topk, swiglu


def main():
    top, ids, dump, out_dir = open_taps(sys.argv)
    m = top.model
    n = len(ids)

    x = mx.array([ids])
    h = m.embed_tokens(x)
    dump("embed", h)

    # A plain causal mask is right for both layer types while the prompt is
    # shorter than the 128-token window; keep prompts short or this lies.
    mask = mx.triu(mx.full((n, n), -mx.inf, dtype=h.dtype), k=1) if n > 1 else None

    for il, layer in enumerate(m.layers):
        at = layer.self_attn
        B, L, D = 1, n, at.head_dim

        cur = layer.input_layernorm(h)
        dump(f"{il}.attn_norm", cur)

        q = at.q_proj(cur).reshape(B, L, -1, D).swapaxes(1, 2)
        k = at.k_proj(cur).reshape(B, L, -1, D).swapaxes(1, 2)
        v = at.v_proj(cur).reshape(B, L, -1, D).swapaxes(1, 2)
        dump(f"{il}.q_proj", q.swapaxes(1, 2))
        dump(f"{il}.k_proj", k.swapaxes(1, 2))
        dump(f"{il}.v_proj", v.swapaxes(1, 2))

        q = at.rope(q)
        k = at.rope(k)
        dump(f"{il}.rope_q", q.swapaxes(1, 2))
        dump(f"{il}.rope_k", k.swapaxes(1, 2))

        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=at.sm_scale, mask=mask, sinks=at.sinks
        )
        o = o.swapaxes(1, 2).reshape(B, L, -1)
        dump(f"{il}.sdpa", o)
        o = at.o_proj(o)
        dump(f"{il}.o_proj", o)
        h = h + o
        dump(f"{il}.attn_resid", h)

        cur = layer.post_attention_layernorm(h)
        dump(f"{il}.ffn_norm", cur)

        # The router, and the k experts it chose.
        g = layer.mlp.router(cur)
        dump(f"{il}.router", g)
        vals, idx = mlx_topk(g, k=layer.mlp.num_experts_per_tok, axis=-1)
        # `mlx_topk` is argpartition, so its k are unordered. The driver's
        # `router_topk` emits them in descending order (k rounds of simd_max),
        # which is a defined order rather than an implementation detail -- so the
        # reference is sorted to match, and the per-expert taps line up row for
        # row. The weighted sum is order-invariant either way.
        order = mx.argsort(-vals, axis=-1)
        idx = mx.take_along_axis(idx, order, axis=-1)
        vals = mx.take_along_axis(vals, order, axis=-1)
        w = mx.softmax(vals, axis=-1, precise=True)

        # The k-wide stacks, in the router's order, so they line up with the
        # driver's `[k, width]` buffers row for row. `SwitchGLU` expands the
        # token axis before each SwitchLinear and squeezes after the last, so
        # this mirrors that rather than calling the sub-layers directly.
        experts = layer.mlp.experts
        xe = mx.expand_dims(cur, (-2, -3))
        gp = experts.gate_proj(xe, idx)
        up = experts.up_proj(xe, idx)
        dump(f"{il}.expert_gate", gp.reshape(B, L, -1))
        dump(f"{il}.expert_up", up.reshape(B, L, -1))
        act = swiglu(up, gp)
        dump(f"{il}.expert_act", act.reshape(B, L, -1))
        down = experts.down_proj(act, idx).squeeze(-2)
        dump(f"{il}.expert_out", down.reshape(B, L, -1))
        moe = (down * mx.expand_dims(w, axis=-1)).sum(axis=-2)
        dump(f"{il}.moe", moe)
        h = h + moe
        dump(f"{il}.layer_out", h)

    h = m.norm(h)
    dump("final_norm", h)
    dump("logits", top.lm_head(h))
    print("wrote gpt-oss taps ->", out_dir)


if __name__ == "__main__":
    main()
