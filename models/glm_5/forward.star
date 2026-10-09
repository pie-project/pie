# The forward of GLM-5: latent attention over the keys the indexer picks,
# then a dense or routed SwiGLU MLP.

load("//lib/mla/forward.star", "attention", "cache", "plans")

def caches(m, c):
    kv = c.kv_space(m.kv)
    index = c.kv_space(m.kv)
    for w in m.layers:
        a = w.attn
        cache(c, kv, a)
        c.kv(index, a.indexer.keys, [a.indexer.head_dim], a.indexer.head_dim)

def forward(m, inputs):
    plan = plans(inputs, m.heads, m.kv_lora_rank)
    y = ops.layout.embed(inputs.tokens(), m.embed, m.vocab)
    routes = inputs.adapter_routes()

    def block(l, w, y):
        x = ops.elemwise.rmsnorm(y, w.attn_norm, w.attn_norm_eps)
        o = attention(x, inputs, plan, w.attn, select = lambda q_a: index_select(x, q_a, inputs, w.attn.indexer))
        adapted = fact.has(fact.Adapter)
        o = ops.linear.lora_correct(x.on(adapted), w.lora_a, w.lora_b, routes, o.on(adapted))
        y = ops.elemwise.residual_add(o, y)

        x = ops.elemwise.rmsnorm(y, w.mlp_norm, w.mlp_norm_eps)
        return ops.elemwise.residual_add(mlp(x, w.mlp), y)

    y = inputs.fold_layers(m.layers, y, block)
    x = ops.elemwise.rmsnorm(y, m.final_norm, m.final_norm_eps)
    x = ops.layout.gather_rows(x, inputs.readout_rows())
    return ops.linear.lm_head(x, m.head)

def mlp(x, f):
    if not f.routed:
        return ops.linear.matmul(ops.linear.mlp_swiglu(ops.linear.matmul(x, f.gate_up), f.inter), f.down)
    routes, weights = ops.linear.moe_topk_sigmoid(
        ops.linear.matmul(x, f.router),
        f.experts,
        f.top_k,
        f.norm_weights,
        f.scaling,
    )
    act = ops.linear.mlp_swiglu(ops.linear.matmul(x, f.shared.gate_up), f.shared.inter)
    shared = ops.linear.matmul(act, f.shared.down)
    packed = ops.linear.moe_matmul_select(x, f.gate_up, routes, f.top_k)
    act = ops.linear.mlp_swiglu(packed, f.inter)
    routed = ops.linear.moe_weighted_sum(
        ops.linear.moe_matmul_select(act, f.down, routes, f.top_k),
        weights,
    )
    return ops.elemwise.residual_add(shared, routed)

def index_select(x, q_a, inputs, ix):
    keys = inputs.kv(ix.keys)
    positions = inputs.positions()
    write_page = inputs.write_page(ix.keys)
    write_offset = inputs.write_offset(ix.keys)
    k = ops.attn.index_layernorm_rope(
        ops.linear.matmul(x, ix.k_proj),
        positions,
        ix.k_norm,
        ix.k_norm_eps,
        ix.k_norm_bias,
        ix.rope_dim,
        ix.theta,
    )
    ops.attn.index_kv_append(k, keys, write_page, write_offset)
    q = ops.attn.index_rope(
        ops.linear.matmul(q_a, ix.q_proj),
        positions,
        ix.heads,
        ix.head_dim,
        ix.rope_dim,
        ix.theta,
    )
    weights = ops.linear.matmul(q_a, ix.weights_proj)
    return ops.attn.index_topk(q, weights, keys, ix.heads, ix.head_dim, ix.top_k, 1)
