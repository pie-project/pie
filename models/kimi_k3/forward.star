# The forward of Kimi-K3: each layer's mixer (MLA or KDA) and MLP read a
# blend of the residual stream's closed AttnRes blocks.

load("//lib/kda/forward.star", kda = "mixer", kda_caches = "caches")
load("//lib/mla/forward.star", "attend", "cache", "plans")

def caches(m, c):
    kv = c.kv_space(m.kv)
    for w in m.layers:
        a = w.mixer
        if a.mla:
            cache(c, kv, a)
        else:
            kda_caches(c, a)

def opens_block(m, l):
    return l % m.res_block == 0

def forward(m, inputs):
    plan = plans(inputs, m.mla_heads, m.kv_lora_rank)
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
            o = mla_mixer(x, inputs, plan, w.mixer)
        else:
            o = kda(x, inputs, w.mixer)
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
    act = ops.linear.mlp_situ(ops.linear.matmul(x, s.gate_up), s.inter, f.beta, f.up_cap)
    return ops.elemwise.residual_add(ops.linear.matmul(act, s.down), routed)

def mla_mixer(x, inputs, plan, a):
    """Latent attention over every cached key, its rope part unrotated and
    no ATTN_QV seam marked."""
    pages = inputs.kv(a.kv)
    write_page = inputs.write_page(a.kv)
    write_offset = inputs.write_offset(a.kv)
    q_a = ops.linear.matmul(x, a.q_a_proj)
    q_a = ops.elemwise.rmsnorm(q_a, a.q_a_norm, a.q_a_norm_eps)
    kv_c, k_pe = ops.attn.mla_latents(
        ops.linear.matmul(x, a.kv_a_proj),
        a.kv_a_norm,
        a.kv_a_norm_eps,
        a.kv_lora_rank,
    )
    q_nope, q_pe = ops.attn.mla_split_q_b(
        ops.linear.matmul(q_a, a.q_b_proj),
        a.heads,
        a.qk_nope_head_dim,
        a.qk_rope_head_dim,
    )
    ops.attn.mla_kv_append(kv_c, k_pe, pages, write_page, write_offset)
    return attend(x, q_nope, q_pe, pages, plan, a)
