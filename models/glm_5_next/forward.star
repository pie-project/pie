# The forward of GLM-5.3-Flash: hyper-connected streams gated into each
# sublayer and folded back, the MoE router hinted by the next layer's; the
# vision tower's patches scattered into the embedded rows; and the MTP head
# drafting from the trunk's last hidden rows.

load("//lib/hyper/forward.star", "gate", "predict_route", "summed")
load("//lib/kda/forward.star", kda = "mixer", kda_caches = "caches")
load("//lib/mla/forward.star", "attention", "boundaries", "cache", "plans", "whole")

def caches(m, c):
    kv = c.kv_space(m.kv)
    for w in m.layers:
        a = w.mixer
        if w.mixer_kind == "mla":
            index = c.kv_space(m.kv)
            cache(c, kv, a)
            c.kv(index, a.indexer.keys, [a.indexer.head_dim], a.indexer.head_dim)
        else:
            kda_caches(c, a)
    if m.mtp != None:
        a = m.mtp.attn
        index = c.kv_space(m.kv)
        cache(c, kv, a)
        c.kv(index, a.indexer.keys, [a.indexer.head_dim], a.indexer.head_dim)

def forward(m, inputs):
    hy = m.hyper
    plan = plans(inputs, m.heads, m.kv_lora_rank)
    positions = inputs.positions()
    towered = tower(inputs, m.tower) if m.tower != None else None
    narrow = ops.layout.embed(inputs.tokens(), m.embed, m.vocab)
    if towered != None:
        imaged = narrow.on(fact.has(fact.Media))
        narrow = ops.layout.scatter_live_rows(towered, inputs.patch_routes(), imaged).everywhere()
    streams = ops.elemwise.hc_expand(narrow, hy.streams)
    routes = inputs.adapter_routes()

    def block(l, w, streams):
        x, post_mix, comb_mix = gate(streams, w.attn_mix, hy)
        x = ops.elemwise.rmsnorm(x, w.mixer_norm, w.mixer_norm_eps)
        if w.mixer_kind == "mla":
            o = mla_mixer(x, inputs, plan, positions, m.act, w.mixer)
        else:
            o = kda(x, inputs, w.mixer)
        adapted = fact.has(fact.Adapter)
        o = ops.linear.lora_correct(x.on(adapted), w.lora_a, w.lora_b, routes, o.on(adapted))
        streams = ops.elemwise.hc_fold(o, streams, post_mix, comb_mix)

        x, post_mix, comb_mix = gate(streams, w.mlp_mix, hy)
        x = ops.elemwise.rmsnorm(x, w.mlp_norm, w.mlp_norm_eps)
        following = m.layers[l + 1] if l + 1 < len(m.layers) else None
        hint = predict_next(streams, following, hy)
        f = mlp(x, w.mlp, hint)
        return ops.elemwise.hc_fold(f, streams, post_mix, comb_mix)

    streams = inputs.fold_layers(m.layers, streams, block)

    y = summed(streams, m.hidden, hy.streams)
    y = ops.layout.gather_rows(y, inputs.readout_rows())
    x = ops.elemwise.rmsnorm(y, m.final_norm, m.final_norm_eps)
    logits = ops.linear.lm_head(x, m.head)

    if m.mtp != None:
        draft(m, inputs, y, positions, logits)
    return logits

def draft(m, inputs, y, positions, logits):
    """The MTP head's drafted token, from the trunk's last hidden rows `y`
    and the token its logits pick."""
    mtp = m.mtp
    drafted = inputs.on(fact.drafts())
    plan = whole(drafted, m.heads, m.kv_lora_rank)
    hidden = y.on(fact.drafts())
    positions = positions.on(fact.drafts())
    token = ops.layout.argmax([logits.on(fact.drafts())])

    e = ops.layout.embed(token, m.embed, m.vocab)
    e = ops.elemwise.rmsnorm(e, mtp.enorm, mtp.norm_eps)
    h = ops.elemwise.rmsnorm(hidden, mtp.hnorm, mtp.norm_eps)
    fused = ops.elemwise.residual_add(
        ops.linear.matmul(e, mtp.e_proj),
        ops.linear.matmul(h, mtp.h_proj),
    )
    x = ops.elemwise.rmsnorm(fused, mtp.mixer_norm, mtp.mixer_norm_eps)
    o = mla_mixer(x, drafted, plan, positions, m.act, mtp.attn)
    r = ops.elemwise.residual_add(o, fused)
    x = ops.elemwise.rmsnorm(r, mtp.mlp_norm, mtp.mlp_norm_eps)
    r = ops.elemwise.residual_add(mlp(x, mtp.mlp, None), r)
    proposal = ops.linear.lm_head(ops.elemwise.rmsnorm(r, mtp.norm, mtp.norm_eps), m.head)
    seam.at(seam.MTP, [proposal])
    ops.layout.argmax([proposal])
    seam.at(seam.MTP_DRAFTS, [ops.layout.argmax([proposal])])

def predict_next(streams, following, hy):
    if following == None or not following.mlp.routed:
        return None
    f = following.mlp
    return predict_route(
        streams,
        hy,
        following.mlp_mix,
        following.mlp_norm,
        following.mlp_norm_eps,
        f.router,
        f.bias,
        f.experts,
    )

def mlp(x, f, hint):
    if not f.routed:
        return ops.linear.matmul(
            ops.linear.mlp_swiglu_clamp(ops.linear.matmul(x, f.gate_up), f.inter, f.limit),
            f.down,
        )
    routes, weights = ops.linear.moe_topk_sigmoid_biased_hinted(
        ops.linear.matmul(x, f.router),
        f.bias,
        f.experts,
        f.top_k,
        f.renorm,
        f.scaling,
        hint,
    )
    packed = ops.linear.moe_matmul_select_quant(x, f.gate_up, routes, f.top_k)
    act = ops.linear.mlp_swiglu_clamp(packed, f.inter, f.limit)
    routed = ops.linear.moe_weighted_sum(
        ops.linear.moe_matmul_select_quant(act, f.down, routes, f.top_k),
        weights,
    )
    s = f.shared
    if s == None:
        return routed
    act = ops.linear.mlp_swiglu_clamp(ops.linear.matmul(x, s.gate_up), s.inter, f.limit)
    return ops.elemwise.residual_add(ops.linear.matmul(act, s.down), routed)

def mla_mixer(x, inputs, plan, positions, act, a):
    """Latent attention over the keys the indexer picks: its rows split into
    decode and prefill arms, or read whole (the draft head's)."""
    split = plan.decode != None
    select = lambda q_a: index_select(x, q_a, inputs, positions, act, a.indexer, split)
    return attention(x, inputs, plan, a, positions = positions, select = select)

def index_select(x, q_a, inputs, positions, act, ix, split):
    keys = inputs.kv(ix.keys)
    write_page = inputs.write_page(ix.keys)
    write_offset = inputs.write_offset(ix.keys)
    row_valid = inputs.row_valid()

    state_kv = ops.attn.index_layernorm_rope(
        ops.linear.matmul(x, ix.wk),
        positions,
        ix.k_norm,
        ix.k_norm_eps,
        ix.k_norm_bias,
        ix.rope_dim,
        ix.theta,
    )
    state_score = ops.linear.matmul(x, ix.kpool_gate)
    ops.attn.pool_state_write(state_kv, state_score, keys, write_page, write_offset, ix.head_dim, ix.kpool)

    bpos, breq, _ = boundaries(positions, row_valid, ix.kpool, split)
    k = ops.attn.pool_gather(bpos, breq, keys, ix.kpool_ape, ix.head_dim, ix.kpool, act)
    ops.attn.pool_kv_append(k, bpos, breq, keys, write_page, write_offset)

    q = ops.attn.index_rope(
        ops.linear.matmul(q_a, ix.wq_b),
        positions,
        ix.heads,
        ix.head_dim,
        ix.rope_dim,
        ix.theta,
    )
    weights = ops.linear.matmul(x, ix.weights_proj)
    return ops.attn.index_topk(q, weights, keys, ix.heads, ix.head_dim, ix.top_k, ix.kpool)

def tower(inputs, t):
    d = t.head_dim
    x = inputs.patches(t.patch_width)
    segments = inputs.patch_segments()
    grid = inputs.patch_positions()
    y = ops.elemwise.add_bias(t.patch_embed_bias, ops.linear.matmul(x, t.patch_embed))
    for b in t.blocks:
        n = ops.elemwise.rmsnorm(y, b.norm1, t.norm_eps)
        q, k, v = ops.layout.split_qkv(
            ops.elemwise.add_bias(b.qkv_bias, ops.linear.matmul(n, b.qkv)),
            t.hidden,
            t.hidden,
        )
        q = ops.elemwise.rmsnorm_per_head(q, b.q_norm, d, t.norm_eps)
        k = ops.elemwise.rmsnorm_per_head(k, b.k_norm, d, t.norm_eps)
        q, k = ops.elemwise.rope_mrope(q, k, grid, [0, d // 4, d // 4], "blocked", d, d, t.theta)
        o = ops.attn.dense(q, k, v, segments, d, t.sm_scale)
        y = ops.elemwise.residual_add(
            ops.elemwise.add_bias(b.proj_bias, ops.linear.matmul(o, b.proj)),
            y,
        )
        n = ops.elemwise.rmsnorm(y, b.norm2, t.norm_eps)
        h = ops.elemwise.add_bias(b.gate_up_bias, ops.linear.matmul(n, b.gate_up))
        a = ops.linear.mlp_swiglu_clamp(h, t.inter, t.limit)
        y = ops.elemwise.residual_add(
            ops.elemwise.add_bias(b.down_bias, ops.linear.matmul(a, b.down)),
            y,
        )
    y = ops.elemwise.rmsnorm(y, t.post_norm, t.norm_eps)
    folded = ops.layout.merge_rows(y, t.merge)
    y = ops.elemwise.add_bias(t.downsample_bias, ops.linear.matmul(folded, t.downsample))
    mg = t.merger
    p = ops.linear.matmul(y, mg.proj)
    n = ops.elemwise.layernorm(p, mg.norm, mg.norm_bias, t.norm_eps)
    g = ops.linear.mlp_gelu_tanh(n)
    h = ops.linear.matmul(g, mg.gate_up)
    a = ops.linear.mlp_swiglu_clamp(h, t.merger_inter, t.limit)
    return ops.linear.matmul(a, mg.down)
