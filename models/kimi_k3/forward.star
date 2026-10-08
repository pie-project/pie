# The forward of Kimi-K3: each layer's mixer (MLA or KDA) and MLP read a
# blend of the residual stream's closed AttnRes blocks.

def caches(m, c):
    kv = c.kv_space(m.kv)
    for w in m.layers:
        a = w.mixer
        if a.mla:
            c.kv(kv, a.kv, [m.kv_lora_rank, a.qk_rope_head_dim], a.qk_rope_head_dim)
        else:
            width = a.heads * a.head_dim
            c.state(a.conv_state, [a.conv_kernel, 3 * width], dtype.bf16, split = 1)

            # the KDA recurrence keeps its state in f32 on every backend
            c.state(a.delta_state, [a.heads, a.head_dim, a.head_dim], dtype.f32, split = 0)

def opens_block(m, l):
    return m.res_block > 0 and l % m.res_block == 0

def forward(m, inputs):
    one = fact.single_token()
    input_d, input_p = inputs.on(one), inputs.on(~one)
    plan = [
        ops.attn.mla_plan(input_d, m.mla_heads, m.kv_lora_rank),
        ops.attn.mla_plan(input_p, m.mla_heads, m.kv_lora_rank),
    ]
    y = ops.layout.embed(inputs.tokens(), m.embed, m.vocab)
    every = m.attn_res == "every"
    routes = inputs.adapter_routes()

    def block(l, w, carried):
        y, blocks = carried

        # The sublayer input: under `every`, a blend of the closed blocks and
        # the running prefix sum `y`; under `at_block_start`, `y` itself,
        # re-blended where a block opens.
        h = y
        b = w.res_blend
        if b != None:
            if every:
                if blocks:
                    h = ops.elemwise.res_blend(y, blocks, b.norm, b.norm_eps, b.proj)
            else:
                y = ops.elemwise.res_blend(y, blocks, b.norm, b.norm_eps, b.proj)
                blocks = blocks + [y]
                h = y

        # Where a block opens the prefix sum is banked and restarts from this
        # layer's attention output.
        fresh = False
        if every and opens_block(m, l):
            blocks = blocks + [y]
            fresh = True

        x = ops.elemwise.rmsnorm(h, w.mixer_norm, w.mixer_norm_eps)
        if w.mixer.mla:
            o = mla_mixer(x, inputs, plan, m, w.mixer)
        else:
            o = kda_mixer(x, inputs, w.mixer)
        adapted = fact.has(fact.Adapter)
        o = ops.linear.lora_correct(x.on(adapted), w.lora_a, w.lora_b, routes, o.on(adapted))
        y = o if fresh else ops.elemwise.residual_add(o, y)

        r = w.mlp_res
        h = ops.elemwise.res_blend(y, blocks, r.norm, r.norm_eps, r.proj) if r != None and blocks else y
        x = ops.elemwise.rmsnorm(h, w.mlp_norm, w.mlp_norm_eps)
        y = ops.elemwise.residual_add(mlp(x, w.mlp), y)
        return (y, blocks)

    y, blocks = inputs.fold_layers(m.layers, (y, []), block)
    r = m.output_res
    if r != None and blocks:
        y = ops.elemwise.res_blend(y, blocks, r.norm, r.norm_eps, r.proj)
    x = ops.elemwise.rmsnorm(y, m.final_norm, m.final_norm_eps)
    x = ops.layout.gather_rows(x, inputs.readout_rows())
    return ops.linear.lm_head(x, m.head)

def mlp(x, f):
    if not f.routed:
        act = ops.linear.mlp_situ(ops.linear.matmul(x, f.gate_up), f.inter, f.beta, f.up_cap)
        return ops.linear.matmul(act, f.down)
    logits = ops.linear.matmul(x, f.router)
    if f.bias != None:
        routes, weights = ops.linear.moe_topk_sigmoid_biased(
            logits,
            f.bias,
            f.experts,
            f.top_k,
            f.renorm,
            f.routed_scaling,
        )
    else:
        routes, weights = ops.linear.moe_topk_sigmoid(logits, f.experts, f.top_k, f.renorm, f.routed_scaling)
    lat = f.latent
    z = ops.linear.matmul(x, lat.down) if lat != None else x
    hidden = ops.linear.moe_matmul_select_quant(z, f.gate_up, routes, f.top_k)
    act = ops.linear.mlp_situ(hidden, f.inter, f.beta, f.up_cap)
    routed = ops.linear.moe_weighted_sum(
        ops.linear.moe_matmul_select_quant(act, f.down, routes, f.top_k),
        weights,
    )
    if lat != None:
        r = ops.elemwise.rmsnorm(routed, lat.norm, lat.norm_eps) if lat.norm != None else routed
        routed = ops.linear.matmul(r, lat.up)
    s = f.shared
    if s == None:
        return routed
    act = ops.linear.mlp_situ(ops.linear.matmul(x, s.gate_up), s.inter, f.beta, f.up_cap)
    return ops.elemwise.residual_add(ops.linear.matmul(act, s.down), routed)

def mla_mixer(x, inputs, plan, m, a):
    pages = inputs.kv(a.kv)
    write_page = inputs.write_page(a.kv)
    write_offset = inputs.write_offset(a.kv)
    q_a = ops.linear.matmul(x, a.q_a_proj)
    q_a = ops.elemwise.rmsnorm(q_a, a.q_a_norm, a.q_a_norm_eps)
    kv_c, k_pe = ops.attn.mla_latents(
        ops.linear.matmul(x, a.kv_a_proj),
        a.kv_a_norm,
        a.kv_a_norm_eps,
        m.kv_lora_rank,
    )
    q_nope, q_pe = ops.attn.mla_split_q_b(
        ops.linear.matmul(q_a, a.q_b_proj),
        m.mla_heads,
        a.qk_nope_head_dim,
        a.qk_rope_head_dim,
    )
    ops.attn.mla_kv_append(kv_c, k_pe, pages, write_page, write_offset)

    q = ops.attn.mla_absorb_q(q_nope, a.kv_b_proj, m.mla_heads, m.kv_lora_rank, a.qk_nope_head_dim, a.v_head_dim)
    seam.at(seam.ATTN_Q, [q])

    one = fact.single_token()
    latent = merge([
        ops.attn.mla_decode(q.on(one), plan[0], q_pe.on(one), pages, m.mla_heads, m.kv_lora_rank, a.sm_scale),
        ops.attn.mla_prefill(q.on(~one), plan[1], q_pe.on(~one), pages, m.mla_heads, m.kv_lora_rank, a.sm_scale),
    ])
    o = ops.attn.mla_absorb_out(latent, a.kv_b_proj, m.mla_heads, m.kv_lora_rank, a.qk_nope_head_dim, a.v_head_dim)
    if a.gate != None:
        o = ops.elemwise.gate_sigmoid_mul(o, ops.linear.matmul(x, a.gate))
    seam.at(seam.ATTN_OUT, [o])
    return ops.linear.matmul(o, a.o_proj)

def kda_mixer(x, inputs, k):
    conv = inputs.state(k.conv_state)
    delta = inputs.state(k.delta_state)
    qkv = ops.linear.matmul(x, k.qkv)
    f = ops.linear.matmul(ops.linear.matmul(x, k.f_a), k.f_b)
    b = ops.linear.matmul(x, k.b)
    seam.at(seam.RECURRENT, [qkv])

    one = fact.single_token()
    step = ops.attn.ssm_kda_step(
        ops.attn.ssm_causal_conv1d(qkv.on(one), k.conv, conv, k.conv_kernel),
        f.on(one),
        b.on(one),
        k.dt_bias,
        k.a_log,
        delta,
        k.heads,
        k.head_dim,
        k.norm_eps,
        k.gate_floor,
    )
    chunked = ops.attn.ssm_kda_chunked(
        ops.attn.ssm_causal_conv1d_chunked(qkv.on(~one), k.conv, conv, k.conv_kernel),
        f.on(~one),
        b.on(~one),
        k.dt_bias,
        k.a_log,
        delta,
        k.heads,
        k.head_dim,
        k.norm_eps,
        k.gate_floor,
    )
    core = merge([step, chunked])

    g = ops.linear.matmul(x, k.gate)
    o = ops.elemwise.rmsnorm_gated_by(core, g, k.o_norm, k.heads, k.o_norm_eps)
    return ops.linear.matmul(o, k.o_proj)
