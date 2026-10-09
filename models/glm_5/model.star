# The weights of GLM-5: three dense layers, then routed experts with a shared
# one; each layer's attention latent (MLA) with an indexer of its own.

load("//lib/adapters/model.star", "banks")

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

GLM5 = struct(
    hidden = 4096,
    layers = 46,
    dense_layers = 3,
    heads = 96,
    q_lora_rank = 1536,
    kv_lora_rank = 512,
    qk_nope_head_dim = 128,
    qk_rope_head_dim = 64,
    v_head_dim = 128,
    theta = 10000.0,
    index_heads = 64,
    index_head_dim = 128,
    index_top_k = 2048,
    dense_inter = 10944,
    experts = 128,
    top_k = 8,
    moe_inter = 1408,
    shared_inter = 1408,
    norm_weights = True,
    scaling = 2.5,
    vocab = 151552,
    norm_eps = 1e-5,
)

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    d = GLM5
    w = deploy.weights[0]
    hidden = d.hidden
    index_width = d.index_heads * d.index_head_dim

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        norm = lambda s, width: weight(n(s), [width], w)
        lora_a, lora_b = banks("layer.{}".format(l), hidden, compute(w))
        attn = attention(
            n,
            d,
            hidden,
            weights = w,
            norms = w,
            eps = d.norm_eps,
            kv = "kv.{}".format(l),
            theta = d.theta,
            indexer = struct(
                heads = d.index_heads,
                head_dim = d.index_head_dim,
                top_k = d.index_top_k,
                rope_dim = d.qk_rope_head_dim,
                theta = d.theta,
                q_proj = weight(n("index_q_proj"), [index_width, d.q_lora_rank], w),
                k_proj = weight(n("index_k_proj"), [d.index_head_dim, hidden], w),
                weights_proj = weight(n("index_weights"), [d.index_heads, d.q_lora_rank], w),
                k_norm = norm("index_k_norm", d.index_head_dim),
                k_norm_eps = d.norm_eps,
                k_norm_bias = weight(n("index_k_norm_bias"), [d.index_head_dim], w),
                keys = "index.{}".format(l),
            ),
        )
        if l < d.dense_layers:
            iw = d.dense_inter
            mlp = struct(
                routed = False,
                gate_up = weight(n("gate_up"), [2 * iw, hidden], w).packed([iw, iw]),
                down = weight(n("down"), [hidden, iw], w).rows(),
                inter = iw,
            )
        else:
            iw = d.moe_inter
            sw = d.shared_inter
            mlp = struct(
                routed = True,
                router = weight(n("router"), [d.experts, hidden], w),
                gate_up = weight(n("experts_gate_up"), [d.experts, 2 * iw, hidden], w).bank([iw, iw]),
                down = weight(n("experts_down"), [d.experts, hidden, iw], w).rows(),
                shared = struct(
                    gate_up = weight(n("shared_gate_up"), [2 * sw, hidden], w).packed([sw, sw]),
                    down = weight(n("shared_down"), [hidden, sw], w).rows(),
                    inter = sw,
                ),
                experts = d.experts,
                top_k = d.top_k,
                inter = iw,
                norm_weights = d.norm_weights,
                scaling = d.scaling,
            )
        return struct(
            attn = attn,
            attn_norm = norm("attn_norm", hidden),
            attn_norm_eps = d.norm_eps,
            mlp_norm = norm("mlp_norm", hidden),
            mlp_norm_eps = d.norm_eps,
            mlp = mlp,
            lora_a = lora_a,
            lora_b = lora_b,
        )

    return struct(
        hidden = hidden,
        vocab = d.vocab,
        heads = d.heads,
        kv_lora_rank = d.kv_lora_rank,
        kv = deploy.kv,
        embed = weight("embed", [d.vocab, hidden], w),
        head = weight("lm_head", [d.vocab, hidden], w).packed([d.vocab]),
        layers = [layer(l) for l in range(d.layers)],
        final_norm = weight("final_norm", [hidden], w),
        final_norm_eps = d.norm_eps,
    )
