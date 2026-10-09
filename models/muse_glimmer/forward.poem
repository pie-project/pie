# The forward of Muse Glimmer: sandwich-normed gated attention over a sliding
# window, every fourth layer over the whole context, then a SwiGLU MLP.

WINDOWED = 0
FULL = 1
CLASSES = [fact.has(fact.Mask), fact.scores(), fact.single_token()]

def caches(m, c):
    kv = c.kv_space(m.kv)
    plane = m.kv_heads * m.head_dim
    for w in m.layers:
        c.kv(kv, w.kv, [plane, plane], m.head_dim, heads = True)

def forward(m, inputs):
    d = m.head_dim
    windows = [m.window, None]
    ([input_m, input_s, input_d], input_p) = inputs.partition(CLASSES)

    def plans(input, plan):
        return [plan(input, m.q_heads, m.kv_heads, d, win) for win in windows]

    plan_m = plans(input_m, ops.attn.plan_prefill)
    plan_s = plans(input_s, ops.attn.plan_prefill)
    plan_d = plans(input_d, ops.attn.plan_decode)
    plan_p = plans(input_p, ops.attn.plan_prefill)
    mask = inputs.mask()
    positions = inputs.positions()

    y = ops.elemwise.rmsnorm_no_scale(
        ops.layout.embed(inputs.tokens(), m.embed, m.vocab),
        m.hidden,
        m.norm_eps,
    )
    routes = inputs.adapter_routes()

    def block(l, w, y):
        reading = FULL if w.full else WINDOWED
        win = windows[reading]
        normed = ops.elemwise.rmsnorm_plus_one(y, w.attn_norm, w.attn_norm_eps)
        pages = inputs.kv(w.kv)

        q, k, v = ops.layout.split_qkv(ops.linear.matmul(normed, w.qkv), m.q_heads * d, m.kv_heads * d)
        q = ops.elemwise.rmsnorm_no_scale(q, d, m.norm_eps)
        k = ops.elemwise.rmsnorm_no_scale(k, d, m.norm_eps)
        if not w.full:
            q, k = ops.elemwise.rope_full(q, k, positions, d, m.theta, False)
        ops.attn.kv_append(k, v, pages, inputs.write_page(w.kv), inputs.write_offset(w.kv))
        seam.at(seam.ATTN_Q, [q])

        ([mq, sq, dq], p) = q.partition(CLASSES)
        if w.full:
            so, lse = ops.attn.prefill_lse(sq, plan_s[reading], pages, win, d, m.kv_heads, m.sm_scale)
            seam.at(seam.SCORES, [lse])
        else:
            so = ops.attn.prefill(sq, plan_s[reading], pages, win, d, m.kv_heads, m.sm_scale)
        a = merge([
            ops.attn.masked(mq, plan_m[reading], mask, pages, win, d, m.kv_heads, True, m.sm_scale),
            so,
            ops.attn.decode(dq, plan_d[reading], pages, win, d, m.sm_scale),
            ops.attn.prefill(p, plan_p[reading], pages, win, d, m.kv_heads, m.sm_scale),
        ])
        seam.at(seam.ATTN_OUT, [a])

        gate = ops.linear.matmul(normed, w.gate)
        o = ops.linear.matmul(ops.elemwise.gate_sigmoid_mul(a, gate), w.o_proj)
        adapted = fact.has(fact.Adapter)
        o = ops.linear.lora_correct(normed.on(adapted), w.lora_a, w.lora_b, routes, o.on(adapted))

        y = ops.elemwise.residual_add(
            ops.elemwise.rmsnorm_plus_one(o, w.post_attn_norm, w.post_attn_norm_eps),
            y,
        )
        mlp_in = ops.elemwise.rmsnorm_plus_one(y, w.pre_ffw_norm, w.pre_ffw_norm_eps)
        act = ops.linear.mlp_swiglu(ops.linear.matmul(mlp_in, w.gate_up), w.inter)
        f = ops.linear.matmul(act, w.down)
        return ops.elemwise.residual_add(
            ops.elemwise.rmsnorm_plus_one(f, w.post_ffw_norm, w.post_ffw_norm_eps),
            y,
        )

    y = inputs.fold_layers(m.layers, y, block)
    x = ops.elemwise.rmsnorm(y, m.final_norm, m.final_norm_eps) * m.output_multiplier
    x = ops.layout.gather_rows(x, inputs.readout_rows())
    return ops.attn.logit_softcap(ops.linear.lm_head(x, m.lm_head), m.softcap)
