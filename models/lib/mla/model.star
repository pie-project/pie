# The weights of multi-head latent attention (MLA): queries through a
# low-rank `q_a`/`q_b` pair, keys and values cached as one latent row (plus
# its rope part) and expanded per head through `kv_b`.
#
# `d` states its shape: `heads`, `q_lora_rank`, `kv_lora_rank`,
# `qk_nope_head_dim`, `qk_rope_head_dim` and `v_head_dim`.

def attention(n, d, hidden, weights, norms, eps, kv, theta = None, gated = False, **extra):
    """The latent attention named by `n`, its projections in `weights` and
    its norms in `norms`, caching under `kv`. Its rope part is rotated at
    `theta`, or left as it is with none; `gated` adds a sigmoid gate on its
    output. `extra` rides along (an indexer, say)."""
    qk_head_dim = d.qk_nope_head_dim + d.qk_rope_head_dim
    v_width = d.heads * d.v_head_dim
    return struct(
        heads = d.heads,
        kv_lora_rank = d.kv_lora_rank,
        qk_nope_head_dim = d.qk_nope_head_dim,
        qk_rope_head_dim = d.qk_rope_head_dim,
        v_head_dim = d.v_head_dim,
        theta = theta,
        sm_scale = f32(1.0 / f32(sqrt(qk_head_dim))),
        q_a_proj = weight(n("q_a_proj"), [d.q_lora_rank, hidden], weights),
        q_a_norm = weight(n("q_a_norm"), [d.q_lora_rank], norms),
        q_a_norm_eps = eps,
        q_b_proj = weight(n("q_b_proj"), [d.heads * qk_head_dim, d.q_lora_rank], weights).columns(),
        kv_a_proj = weight(n("kv_a_proj"), [d.kv_lora_rank + d.qk_rope_head_dim, hidden], weights),
        kv_a_norm = weight(n("kv_a_norm"), [d.kv_lora_rank], norms),
        kv_a_norm_eps = eps,
        kv_b_proj = weight(
            n("kv_b_proj"),
            [d.heads * (d.qk_nope_head_dim + d.v_head_dim), d.kv_lora_rank],
            weights,
        ).columns(),
        gate = weight(n("o_gate"), [v_width, hidden], weights).columns() if gated else None,
        o_proj = weight(n("o_proj"), [hidden, v_width], weights).rows(),
        kv = kv,
        **extra
    )
