# The forward of GLM-5: latent attention over the keys the indexer picks,
# then a dense or routed SwiGLU MLP.

def caches(m, c):
    kv = c.kv_space(m.kv)
    index = c.kv_space(m.kv)
    for w in m.layers:
        a = w.attn
        c.kv(kv, a.kv, [m.kv_lora_rank, a.qk_rope_head_dim], a.qk_rope_head_dim)
        c.kv(index, a.indexer.keys, [a.indexer.head_dim], a.indexer.head_dim)

def forward(m, inputs):
    one = fact.single_token()
    input_d, input_p = inputs.on(one), inputs.on(~one)
    plan = [
        ops.attn.mla_plan(input_d, m.heads, m.kv_lora_rank),
        ops.attn.mla_plan(input_p, m.heads, m.kv_lora_rank),
    ]
    y = ops.layout.embed(inputs.tokens(), m.embed, m.vocab)
    routes = inputs.adapter_routes()

    def block(l, w, y):
        x = ops.elemwise.rmsnorm(y, w.attn_norm, w.attn_norm_eps)
        o = latent_attention(x, inputs, plan, m, w.attn)
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
    shared = None
    if f.shared != None:
        act = ops.linear.mlp_swiglu(ops.linear.matmul(x, f.shared.gate_up), f.shared.inter)
        shared = ops.linear.matmul(act, f.shared.down)
    packed = ops.linear.moe_matmul_select(x, f.gate_up, routes, f.top_k)
    act = ops.linear.mlp_swiglu(packed, f.inter)
    routed = ops.linear.moe_weighted_sum(
        ops.linear.moe_matmul_select(act, f.down, routes, f.top_k),
        weights,
    )
    if shared == None:
        return routed
    return ops.elemwise.residual_add(shared, routed)

def latent_attention(x, inputs, plan, m, a):
    pages = inputs.kv(a.kv)
    positions = inputs.positions()
    write_page = inputs.write_page(a.kv)
    write_offset = inputs.write_offset(a.kv)

    q_a = ops.linear.matmul(x, a.q_a_proj)
    q_a = ops.elemwise.rmsnorm(q_a, a.q_a_norm, a.q_a_norm_eps)
    q_b = ops.linear.matmul(q_a, a.q_b_proj)
    kv_a = ops.linear.matmul(x, a.kv_a_proj)
    seam.at(seam.ATTN_QV, [q_b, kv_a])

    selection = index_select(x, q_a, inputs, a.indexer)

    kv_c, k_pe = ops.attn.mla_latents_rope(
        kv_a,
        positions,
        a.kv_a_norm,
        a.kv_a_norm_eps,
        m.kv_lora_rank,
        a.qk_rope_head_dim,
        a.theta,
    )
    ops.attn.mla_kv_append(kv_c, k_pe, pages, write_page, write_offset)

    q_nope, q_pe = ops.attn.mla_split_q_b(q_b, m.heads, a.qk_nope_head_dim, a.qk_rope_head_dim)
    q_pe = ops.elemwise.rope_partial_q(q_pe, positions, a.qk_rope_head_dim, a.qk_rope_head_dim, a.theta)

    q = ops.attn.mla_absorb_q(q_nope, a.kv_b_proj, m.heads, m.kv_lora_rank, a.qk_nope_head_dim, a.v_head_dim)
    seam.at(seam.ATTN_Q, [q])

    one = fact.single_token()
    scored = merge([
        ops.attn.mla_decode_selected(
            q.on(one), plan[0], q_pe.on(one), selection.on(one), pages,
            m.heads, m.kv_lora_rank, a.sm_scale,
        ),
        ops.attn.mla_prefill_selected(
            q.on(~one), plan[1], q_pe.on(~one), selection.on(~one), pages,
            m.heads, m.kv_lora_rank, a.sm_scale,
        ),
    ])

    v = ops.attn.mla_absorb_out(scored, a.kv_b_proj, m.heads, m.kv_lora_rank, a.qk_nope_head_dim, a.v_head_dim)
    seam.at(seam.ATTN_OUT, [v])
    return ops.linear.matmul(v, a.o_proj)

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
