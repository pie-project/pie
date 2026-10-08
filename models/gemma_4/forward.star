# The forward of Gemma 4: sliding and global attention, a GeGLU MLP (with
# routed experts beside it on 26B-A4B), per-layer embeddings on E4B, a vision
# tower's rows scattered into the embedding, a block-diffusion model's
# self-conditioning, and each drafter's proposals.

load("//lib/dflash/forward.star", "arm", "block_rows", "plant_readout", "tap", dflash_caches = "caches")

SLIDING = 0
GLOBAL = 1
FRAC_1_SQRT_2 = 0.70710677

def caches(m, c):
    kv = c.kv_space(m.kv)
    sliding = c.kv_space(m.kv, m.sliding.window)
    for w in m.layers:
        if not w.attn.banks.shared:
            if w.attn.reading == SLIDING:
                space, head_dim, kv_heads = sliding, m.sliding.head_dim, m.sliding.kv_heads
            else:
                space, head_dim, kv_heads = kv, m.glob.head_dim, m.glob.kv_heads
            plane = kv_heads * head_dim
            c.kv(space, w.attn.kv, [plane, plane], head_dim, heads = True)
    if m.draft != None:
        plane = m.glob.kv_heads * m.glob.head_dim
        c.kv(kv, m.draft.attn.kv, [plane, plane], m.glob.head_dim, heads = True)
    if m.dflash != None:
        dflash_caches(m.dflash, c, kv)

def geometry(m, reading):
    if reading == SLIDING:
        return m.sliding.head_dim, m.sliding.kv_heads, m.sliding.window
    return m.glob.head_dim, m.glob.kv_heads, None

def forward(m, inputs):
    classes = [fact.has(fact.Mask), fact.scores(), fact.single_token()]
    dr = m.dflash
    if dr != None:
        _ = inputs.on(block_rows(dr))
        trunk_inputs = inputs.on(~block_rows(dr))
    else:
        trunk_inputs = inputs
    ([input_m, input_s, input_d], input_p) = trunk_inputs.partition(classes)
    positions = trunk_inputs.positions()

    def plans(input, plan):
        return [
            plan(input, m.q_heads, m.sliding.kv_heads, m.sliding.head_dim, m.sliding.window),
            plan(input, m.q_heads, m.glob.kv_heads, m.glob.head_dim, None),
        ]

    plan_m = plans(input_m, ops.attn.plan_prefill)
    plan_d = plans(input_d, ops.attn.plan_decode)
    plan_p = plans(input_p, ops.attn.plan_prefill)
    plan_s = plans(input_s, ops.attn.plan_prefill)
    mask = inputs.mask()

    towered = tower(inputs, m.tower) if m.tower != None else None

    ids = inputs.tokens()
    root = f32(sqrt(m.hidden))
    y = ops.layout.embed(ids, m.embed, m.vocab) * root

    if towered != None:
        imaged = y.on(fact.has(fact.Media))
        y = ops.layout.scatter_live_rows(towered, inputs.patch_routes(), imaged).everywhere()

    if m.self_cond != None:
        sc = m.self_cond
        den, enc = y.on(fact.bidirectional()), y.on(~fact.bidirectional())
        input_den = inputs.on(fact.bidirectional())
        soft = ops.layout.embed_weighted(
            input_den.self_cond_rows(sc.taps),
            input_den.self_cond_weights(sc.taps),
            m.embed,
            m.vocab,
        ) * root
        normed = ops.elemwise.rmsnorm(soft, sc.pre_norm, sc.norm_eps)
        act = ops.linear.mlp_geglu_tanh_packed(ops.linear.matmul(normed, sc.gate_up), sc.inter)
        signal = ops.linear.matmul(act, sc.down)
        den = ops.elemwise.rmsnorm_no_scale(ops.elemwise.residual_add(den, signal), m.hidden, sc.norm_eps)
        y = merge([den, enc])

    h_block = None
    if dr != None:
        h_block = y.on(block_rows(dr))
        y = y.on(~block_rows(dr))

    relay = None
    if m.ple != None:
        ple = m.ple
        proj = ops.linear.matmul(y, ple.model_proj) * f32(1.0 / root)
        relay = ops.elemwise.rmsnorm_per_head(proj, ple.model_norm, ple.dim, ple.model_norm_eps)

    routes = inputs.adapter_routes()

    def block(l, w, carried):
        y, tapped = carried
        normed = ops.elemwise.rmsnorm(y, w.attn_norm, w.attn_norm_eps)
        at = w.attn
        d, kv_heads, win = geometry(m, at.reading)
        pages = inputs.kv(at.kv)
        if at.banks.shared:
            q = q_only(normed, positions, m, at, d, at.banks.q_proj)
        else:
            q = qkv_unfused(normed, positions, inputs, m, at, d, kv_heads, at.banks.qkv,
                            at.banks.k_norm, at.banks.k_norm_eps, pages)
        seam.at(seam.ATTN_Q, [q])

        r = at.reading
        ([mq, sq, dq], p) = q.partition(classes)
        if r == SLIDING:
            so = ops.attn.prefill(sq, plan_s[r], pages, win, d, kv_heads, at.sm_scale)
        else:
            so, lse = ops.attn.prefill_lse(sq, plan_s[r], pages, win, d, kv_heads, at.sm_scale)
            seam.at(seam.SCORES, [lse])
        a = merge([
            ops.attn.masked(mq, plan_m[r], mask, pages, win, d, kv_heads, True, at.sm_scale),
            so,
            ops.attn.decode(dq, plan_d[r], pages, win, d, at.sm_scale),
            ops.attn.prefill(p, plan_p[r], pages, win, d, kv_heads, at.sm_scale),
        ])
        seam.at(seam.ATTN_OUT, [a])
        o = ops.linear.matmul(a, w.o_proj)
        adapted = fact.has(fact.Adapter)
        o = ops.linear.lora_correct(normed.on(adapted), w.lora_a, w.lora_b, routes, o.on(adapted))

        y = ops.elemwise.residual_add(ops.elemwise.rmsnorm(o, w.post_attn_norm, w.post_attn_norm_eps), y)
        mlp_in = ops.elemwise.rmsnorm(y, w.pre_ffw_norm, w.pre_ffw_norm_eps)
        act = ops.linear.mlp_geglu_tanh_packed(ops.linear.matmul(mlp_in, w.gate_up), w.inter)
        f = ops.linear.matmul(act, w.down)
        if w.moe != None:
            x = w.moe
            h1 = ops.elemwise.rmsnorm(f, x.post_ffw_norm_1, x.post_ffw_norm_1_eps)
            experts, weights = ops.linear.moe_topk_softmax_scaled(
                ops.linear.matmul(ops.elemwise.rmsnorm(y, x.router_norm, x.router_norm_eps), x.router),
                x.per_expert_scale,
                x.experts,
                x.top_k,
            )
            moe_in = ops.elemwise.rmsnorm(y, x.pre_ffw_norm_2, x.pre_ffw_norm_2_eps)

            def select(act, bank):
                if bank.dtype in [dtype.bf16, dtype.f16, dtype.f32]:
                    return ops.linear.moe_matmul_select(act, bank, experts, x.top_k)
                return ops.linear.moe_matmul_select_quant(act, bank, experts, x.top_k)

            hidden = ops.linear.mlp_geglu_tanh_packed(select(moe_in, x.gate_up), x.inter)
            routed = ops.linear.moe_weighted_sum(select(hidden, x.down), weights)
            h2 = ops.elemwise.rmsnorm(routed, x.post_ffw_norm_2, x.post_ffw_norm_2_eps)
            f = ops.elemwise.residual_add(h1, h2)
        y = ops.elemwise.residual_add(ops.elemwise.rmsnorm(f, w.post_ffw_norm, w.post_ffw_norm_eps), y)

        if relay != None:
            ple = m.ple
            lp = ple.per_layer[l]
            table = ops.layout.embed(ids, lp.table, m.vocab) * f32(sqrt(ple.dim))
            mixed = ops.elemwise.residual_add(table, ops.layout.select(relay, l, ple.dim)) * FRAC_1_SQRT_2
            gated = ops.linear.mlp_geglu_tanh(ops.linear.matmul(y, lp.gate), mixed)
            out = ops.linear.matmul(gated, lp.proj)
            out = ops.elemwise.rmsnorm(out, lp.norm, lp.norm_eps)
            y = ops.elemwise.scale(lp.scalar, ops.elemwise.residual_add(out, y))

        if w.scalar != None:
            y = ops.elemwise.scale(w.scalar, y)
        if dr != None:
            tapped = tap(dr, l, y, tapped)
        return (y, tapped)

    y, tapped = inputs.fold_layers(m.layers, (y, None), block)

    x = ops.elemwise.rmsnorm(y, m.final_norm, m.final_norm_eps)
    hb = None
    if dr != None:
        hb = arm(dr, inputs, tapped, h_block, mask, block_rows(dr))
        x = merge([hb, x])
    x = ops.layout.gather_rows(x, inputs.readout_rows())
    logits = ops.linear.lm_head(x, m.embed)
    if m.softcap != None:
        logits = ops.attn.logit_softcap(logits, m.softcap)
    if dr != None:
        plant_readout(dr, logits, inputs, hb, block_rows(dr))

    if m.draft != None:
        eagle(m, inputs, x, logits, root)
    if m.assistant != None:
        assist(m, inputs, positions, x, logits, root)
    return logits

def eagle(m, inputs, x, logits, root):
    a = m.draft
    input_draft = inputs.on(fact.drafts())
    plan_draft = ops.attn.plan_prefill(input_draft, m.q_heads, m.glob.kv_heads, m.glob.head_dim, None)
    dx = x.on(fact.drafts())
    dlogits = logits.on(fact.drafts())
    chosen = ops.layout.argmax([dlogits])
    e = ops.layout.embed(chosen, m.embed, m.vocab) * root
    dy = ops.elemwise.residual_add(ops.linear.matmul(e, a.fc_embed), ops.linear.matmul(dx, a.fc_hidden))

    normed = ops.elemwise.rmsnorm(dy, a.attn_norm, a.norm_eps)
    o = draft_attn(normed, inputs, m, plan_draft, a)
    dy = ops.elemwise.residual_add(ops.elemwise.rmsnorm(o, a.post_attn_norm, a.norm_eps), dy)
    mlp_in = ops.elemwise.rmsnorm(dy, a.pre_ffw_norm, a.norm_eps)
    act = ops.linear.mlp_geglu_tanh_packed(ops.linear.matmul(mlp_in, a.gate_up), a.inter)
    f = ops.linear.matmul(act, a.down)
    dy = ops.elemwise.residual_add(ops.elemwise.rmsnorm(f, a.post_ffw_norm, a.norm_eps), dy)

    draft = ops.linear.lm_head(ops.elemwise.rmsnorm(dy, m.final_norm, m.final_norm_eps), m.embed)
    if m.softcap != None:
        draft = ops.attn.logit_softcap(draft, m.softcap)
    seam.at(seam.MTP, [draft])
    seam.at(seam.MTP_DRAFTS, [ops.layout.argmax([draft])])

def assist(m, inputs, positions, x, logits, root):
    a = m.assistant
    input_draft = inputs.on(fact.drafts())
    dpos = positions.on(fact.drafts())
    plans = [
        ops.attn.plan_prefill(input_draft, m.q_heads, m.sliding.kv_heads, m.sliding.head_dim, m.sliding.window),
        ops.attn.plan_prefill(input_draft, m.q_heads, m.glob.kv_heads, m.glob.head_dim, None),
    ]
    dx = x.on(fact.drafts())
    dlogits = logits.on(fact.drafts())
    token = ops.layout.argmax([dlogits])
    hidden = dx
    chain = []
    for step in range(a.depth):
        e = ops.layout.embed(token, m.embed, m.vocab) * root
        y = ops.elemwise.residual_add(ops.linear.matmul(e, a.pre_embed), ops.linear.matmul(hidden, a.pre_hidden))
        for w in a.layers:
            at = w.attn
            d, kv_heads, win = geometry(m, at.reading)
            pages = inputs.kv(at.kv)
            normed = ops.elemwise.rmsnorm(y, w.attn_norm, a.norm_eps)
            q = q_only(normed, dpos, m, at, d, at.banks.q_proj)
            o = ops.attn.prefill(q, plans[at.reading], pages, win, d, kv_heads, at.sm_scale)
            o = ops.linear.matmul(o, w.o_proj)
            y = ops.elemwise.residual_add(ops.elemwise.rmsnorm(o, w.post_attn_norm, a.norm_eps), y)
            mlp_in = ops.elemwise.rmsnorm(y, w.pre_ffw_norm, a.norm_eps)
            act = ops.linear.mlp_geglu_tanh_packed(ops.linear.matmul(mlp_in, w.gate_up), w.inter)
            f = ops.linear.matmul(act, w.down)
            y = ops.elemwise.residual_add(ops.elemwise.rmsnorm(f, w.post_ffw_norm, a.norm_eps), y)
            y = ops.elemwise.scale(w.scalar, y)
        read = ops.elemwise.rmsnorm(y, a.norm, a.norm_eps)
        draft = ops.linear.lm_head(read, a.embed)
        if step == 0:
            seam.at(seam.MTP, [draft])
        token = ops.layout.argmax([draft])
        hidden = ops.linear.matmul(read, a.post)
        chain.append(draft)
    seam.at(seam.MTP_DRAFTS, [ops.layout.argmax(chain)])

def draft_attn(x, inputs, m, plan, a):
    at = a.attn
    d = m.glob.head_dim
    pages = inputs.kv(at.kv)
    q = qkv_unfused(x, inputs.positions(), inputs, m, at, d, m.glob.kv_heads, at.banks.qkv,
                    at.banks.k_norm, at.banks.k_norm_eps, pages)
    o = ops.attn.prefill(q, plan, pages, None, d, m.glob.kv_heads, at.sm_scale)
    return ops.linear.matmul(o, a.o_proj)

def clipped(x, c):
    if c.clip == None:
        return ops.linear.matmul(x, c.bank)
    k = c.clip
    held = ops.elemwise.clamp_learned(ops.elemwise.copy(x), k.in_lo, k.in_hi)
    return ops.elemwise.clamp_learned(ops.linear.matmul(held, c.bank), k.out_lo, k.out_hi)

def tower(inputs, t):
    d = t.head_dim
    x = inputs.patches(t.patch_width)
    segments = inputs.patch_segments()
    grid = inputs.patch_positions()

    y = ops.linear.matmul(x, t.patch_embed)
    pos = ops.layout.embed_weighted(inputs.patch_embed_rows(2), inputs.patch_embed_weights(2), t.pos_embed, t.positions)
    y = ops.elemwise.residual_add(pos, y)

    for b in t.blocks:
        n = ops.elemwise.rmsnorm(y, b.attn_norm, t.norm_eps)
        q = ops.elemwise.rmsnorm_per_head(clipped(n, b.q), b.q_norm, d, t.norm_eps)
        k = ops.elemwise.rmsnorm_per_head(clipped(n, b.k), b.k_norm, d, t.norm_eps)
        v = ops.elemwise.rmsnorm_no_scale(clipped(n, b.v), d, t.norm_eps)
        q, k = ops.elemwise.rope_mrope(q, k, grid, [0, d // 4, d // 4], "split", d, d, t.theta)
        o = ops.attn.dense(q, k, v, segments, d, t.sm_scale)
        y = ops.elemwise.residual_add(ops.elemwise.rmsnorm(clipped(o, b.o), b.post_attn_norm, t.norm_eps), y)

        n = ops.elemwise.rmsnorm(y, b.pre_ffw_norm, t.norm_eps)
        act = ops.linear.mlp_geglu_tanh(clipped(n, b.gate), clipped(n, b.up))
        y = ops.elemwise.residual_add(ops.elemwise.rmsnorm(clipped(act, b.down), b.post_ffw_norm, t.norm_eps), y)

    pooled = ops.layout.pool_rows(y, t.pool) * f32(sqrt(t.hidden))
    if t.std != None:
        pooled = ops.elemwise.standardize(pooled, t.std.bias, t.std.scale)
    return ops.linear.matmul(pooled, t.projection)

def qkv_unfused(x, pos, inputs, m, at, d, kv_heads, qkv, k_norm, k_norm_eps, pages):
    write_page = inputs.write_page(at.kv)
    write_offset = inputs.write_offset(at.kv)
    q, k, v = ops.layout.split_qkv(ops.linear.matmul(x, qkv), m.q_heads * d, kv_heads * d)
    v = ops.elemwise.rmsnorm_no_scale(v, d, at.q_norm_eps)
    q = ops.elemwise.rmsnorm_per_head(q, at.q_norm, d, at.q_norm_eps)
    k = ops.elemwise.rmsnorm_per_head(k, k_norm, d, k_norm_eps)
    if at.reading == GLOBAL:
        q, k = ops.elemwise.rope_partial(q, k, pos, m.glob.rotary_dim, d, m.glob.theta)
    else:
        q, k = ops.elemwise.rope_full(q, k, pos, d, m.sliding.theta, False)
    ops.attn.kv_append(k, v, pages, write_page, write_offset)
    return q

def q_only(x, pos, m, at, d, q_proj):
    q = ops.elemwise.rmsnorm_per_head(ops.linear.matmul(x, q_proj), at.q_norm, d, at.q_norm_eps)
    if at.reading == GLOBAL:
        return ops.elemwise.rope_partial_q(q, pos, m.glob.rotary_dim, d, m.glob.theta)
    return ops.elemwise.rope_partial_q(q, pos, d, d, m.sliding.theta)
