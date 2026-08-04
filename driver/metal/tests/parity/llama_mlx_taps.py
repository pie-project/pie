#!/usr/bin/env python3
"""Dump mlx-lm's per-kernel llama/qwen activations under the names LlamaEngine's
tap hook writes, so cosine_bisect.py can diff them.

mlx-lm is the oracle: it is the implementation that demonstrably produces
coherent text on these checkpoints. This re-runs its forward pass inline,
because the taps are the intermediates a single `model(ids)` call does not
expose.

Covers the whole family the engine does -- llama/mistral/qwen2 (dense, no
qk-norm), qwen3 (dense, qk-norm) and qwen3_moe (routed) -- by reading the
modules off the loaded model rather than by branching on `model_type`: a layer
either has `q_norm` or it does not, and its `mlp` either has `switch_mlp` or
it does not.

The routed taps are the k-wide stacks the driver actually computes, one row per
SELECTED expert in the order the router chose it, not the `num_experts`-wide
tensors mlx's SwitchGLU hides internally.

Note the names the driver does NOT publish, and why: q_proj/k_proj and
q_norm/k_norm are computed in place and share a buffer with `rope_q`/`rope_k`,
so only that colour's final writer is named. cosine_bisect skips whatever only
one side has, so dumping them here is free and useful for a manual look.

Usage: llama_mlx_taps.py <model_path> <comma_token_ids> <out_dir>
"""
import sys

import mlx.core as mx
from taps_common import open_taps


def main():
    top, ids, dump, out_dir = open_taps(sys.argv)
    m = top.model
    n = len(ids)

    x = mx.array([ids])
    h = m.embed_tokens(x)
    dump("embed", h)

    mask = mx.triu(mx.full((n, n), -mx.inf, dtype=h.dtype), k=1) if n > 1 else None

    for li, layer in enumerate(m.layers):
        p = f"{li}."
        a = layer.self_attn
        hd = a.head_dim if hasattr(a, "head_dim") else m.args.head_dim

        y = layer.input_layernorm(h)
        dump(p + "attn_norm", y)
        q, k, v = a.q_proj(y), a.k_proj(y), a.v_proj(y)
        dump(p + "q_proj", q)
        dump(p + "k_proj", k)
        dump(p + "v_proj", v)

        nq = q.shape[-1] // hd
        nkv = k.shape[-1] // hd
        q = q.reshape(1, n, nq, hd).transpose(0, 2, 1, 3)
        k = k.reshape(1, n, nkv, hd).transpose(0, 2, 1, 3)
        v = v.reshape(1, n, nkv, hd).transpose(0, 2, 1, 3)
        # Qwen3 RMS-normalizes each head before the rotation; llama does not.
        if getattr(a, "q_norm", None) is not None:
            q = a.q_norm(q)
            k = a.k_norm(k)
            dump(p + "q_norm", q.transpose(0, 2, 1, 3))
            dump(p + "k_norm", k.transpose(0, 2, 1, 3))
        q = a.rope(q)
        k = a.rope(k)
        dump(p + "rope_q", q.transpose(0, 2, 1, 3))
        dump(p + "rope_k", k.transpose(0, 2, 1, 3))

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(1, n, -1)
        dump(p + "sdpa", o)
        o = a.o_proj(o)
        dump(p + "o_proj", o)
        h = h + o
        dump(p + "attn_out", h)

        y = layer.post_attention_layernorm(h)
        dump(p + "ffn_norm", y)
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            logits = mlp.gate(y)
            dump(p + "router", logits)
            k_top = mlp.top_k
            # mlx softmaxes the SELECTED logits, which is `norm_topk_prob: true`
            # -- the only variant the driver accepts.
            idx = mx.stop_gradient(mx.argpartition(-logits, kth=k_top - 1, axis=-1)[..., :k_top])
            chosen = mx.take_along_axis(logits, idx, axis=-1)
            w = mx.softmax(chosen.astype(mx.float32), axis=-1).astype(h.dtype)
            sm = mlp.switch_mlp
            # `SwitchGLU` expects the two broadcast axes the gather-matmul
            # indexes with, and squeezes one back off the result.
            xe = mx.expand_dims(y, (-2, -3))
            g = sm.gate_proj(xe, idx).squeeze(-2)
            u = sm.up_proj(xe, idx).squeeze(-2)
            dump(p + "expert_gate", g)
            dump(p + "expert_up", u)
            act = mx.sigmoid(g) * g * u
            dump(p + "expert_act", act)
            d = sm.down_proj(mx.expand_dims(act, -2), idx).squeeze(-2)
            dump(p + "expert_out", d)
            f = (d * mx.expand_dims(w, axis=-1)).sum(axis=-2)
            dump(p + "moe", f)
        else:
            g = mlp.gate_proj(y)
            u = mlp.up_proj(y)
            dump(p + "gate_proj", g)
            dump(p + "up_proj", u)
            act = mx.sigmoid(g) * g * u
            dump(p + "ffn_act", act)
            f = mlp.down_proj(act)
            dump(p + "ffn_out", f)
        h = h + f
        dump(p + "layer_out", h)

    h = m.norm(h)
    dump("final_norm", h)
    dump("logits", top.lm_head(h) if getattr(top, "lm_head", None) is not None
         else m.embed_tokens.as_linear(h))


if __name__ == "__main__":
    main()
