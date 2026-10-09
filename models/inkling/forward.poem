# The forward of Inkling: relative-biased attention, short convolutions
# around it and the MLP, a dense or sink-routed MLP.

LOCAL = 0
GLOBAL = 1

def caches(m, c):
    kv = c.kv_space(m.kv)
    taps = m.conv_width
    for w in m.layers:
        plane = w.kv_heads * m.head_dim
        c.kv(kv, w.kv, [plane, plane], m.head_dim, heads = True)
        c.state(w.k_state, [taps, plane], dtype.bf16, split = 1)
        c.state(w.v_state, [taps, plane], dtype.bf16, split = 1)
        c.state(w.attn_state, [taps, m.hidden], dtype.bf16)
        c.state(w.mlp_state, [taps, m.hidden], dtype.bf16)

def kv_heads_of(m, reading):
    for w in m.layers:
        if w.reading == reading:
            return w.kv_heads
    return 0

def forward(m, inputs):
    d = m.head_dim
    one = fact.single_token()
    input_d, input_p = inputs.on(one), inputs.on(~one)
    geometry = [(kv_heads_of(m, LOCAL), m.window), (kv_heads_of(m, GLOBAL), None)]
    plan_d = [
        ops.attn.plan_decode(input_d, m.heads, kv_heads, d, win) if kv_heads > 0 else None
        for kv_heads, win in geometry
    ]
    plan_p = [
        ops.attn.plan_prefill(input_p, m.heads, kv_heads, d, win) if kv_heads > 0 else None
        for kv_heads, win in geometry
    ]

    y = ops.elemwise.rmsnorm(ops.layout.embed(inputs.tokens(), m.embed, m.vocab), m.embed_norm, m.norm_eps)
    routes = inputs.adapter_routes()

    def block(l, w, y):
        reading = w.reading
        win = geometry[reading][1]
        x = ops.elemwise.rmsnorm(y, w.attn_norm, m.norm_eps)
        pages = inputs.kv(w.kv)

        q = ops.linear.matmul(x, w.q_proj)
        k = conv(ops.linear.matmul(x, w.k_proj), w.k_conv, w.k_state, inputs, m)
        v = conv(ops.linear.matmul(x, w.v_proj), w.v_conv, w.v_state, inputs, m)
        r = ops.linear.matmul(x, w.r_proj)
        q = ops.elemwise.rmsnorm_per_head(q, w.q_norm, d, m.norm_eps)
        k = ops.elemwise.rmsnorm_per_head(k, w.k_norm, d, m.norm_eps)
        ops.attn.kv_append(k, v, pages, inputs.write_page(w.kv), inputs.write_offset(w.kv))
        seam.at(seam.ATTN_Q, [q])

        bias = ops.linear.rel_bias(r, w.rel_proj, m.heads, m.d_rel, w.extent)
        log_scaling = m.log_scaling if reading == GLOBAL else None
        a = merge([
            ops.attn.decode_rel(
                q.on(one), plan_d[reading], pages, bias.on(one), win, d, w.extent,
                m.sm_scale, log_scaling,
            ),
            ops.attn.prefill_rel(
                q.on(~one), plan_p[reading], pages, bias.on(~one), win, d, w.kv_heads,
                w.extent, m.sm_scale, log_scaling,
            ),
        ])
        seam.at(seam.ATTN_OUT, [a])
        o = ops.linear.matmul(a, w.o_proj)
        adapted = fact.has(fact.Adapter)
        o = ops.linear.lora_correct(x.on(adapted), w.lora_a, w.lora_b, routes, o.on(adapted))
        o = conv(o, w.attn_conv, w.attn_state, inputs, m)
        y = ops.elemwise.residual_add(o, y)

        x = ops.elemwise.rmsnorm(y, w.mlp_norm, m.norm_eps)
        f = conv(mlp(x, w.mlp), w.mlp_conv, w.mlp_state, inputs, m)
        return ops.elemwise.residual_add(f, y)

    y = inputs.fold_layers(m.layers, y, block)
    x = ops.elemwise.rmsnorm(y, m.final_norm, m.norm_eps) * m.head_scale
    x = ops.layout.gather_rows(x, inputs.readout_rows())
    return ops.linear.lm_head(x, m.unembed)

def mlp(x, f):
    if not f.routed:
        act = ops.linear.mlp_swiglu(ops.linear.matmul(x, f.gate_up), f.inter)
        return ops.elemwise.scale(f.scale, ops.linear.matmul(act, f.down))
    fan = f.top_k + f.sink
    routes, weights = ops.linear.moe_topk_sigmoid_sink(
        ops.linear.matmul(x, f.router),
        f.bias,
        f.scale,
        f.experts,
        f.top_k,
        f.sink,
        f.scaling,
    )
    hidden = ops.linear.mlp_swiglu(ops.linear.moe_matmul_select(x, f.gate_up, routes, fan), f.inter)
    return ops.linear.moe_weighted_sum(ops.linear.moe_matmul_select(hidden, f.down, routes, fan), weights)

def conv(v, weight, state, inputs, m):
    slab = inputs.state(state)
    one = fact.single_token()
    return merge([
        ops.attn.short_conv(v.on(one), weight, slab, m.conv_width),
        ops.attn.short_conv_chunked(v.on(~one), weight, slab, m.conv_width),
    ])
