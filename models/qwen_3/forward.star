# The forward of Qwen 3.5 / 3.6 / 3.8: gated-delta mixers walking their
# recurrence, gated attention every few layers, a dense or routed MLP; a
# vision tower's rows scattered into the embedding; the drafters'
# proposals. Ternary-Bonsai's weights are stored rotated, so its forward turns
# each rotated projection's input by the same Hadamard and signs; the KV-
# rotated geometries turn their attention's q, k and v by a Hadamard.

load("//lib/dflash/forward.star", "arm", "block_rows", "plant_readout", "tap", dflash_caches = "caches")

def draft_attn(x, inputs, plan, a, head_dim, kv_heads, rope, append = True):
    """A draft head's gated attention over `plan`, rotating q and k by
    `rope(q, k)`; `append` caches this step's k and v."""
    d = head_dim
    pages = inputs.kv(a.kv)
    write_page = inputs.write_page(a.kv)
    write_offset = inputs.write_offset(a.kv)
    q, gate = ops.layout.split_q_gate(ops.linear.matmul(x, a.qg_proj), d)
    k = ops.linear.matmul(x, a.k_proj)
    v = ops.linear.matmul(x, a.v_proj)
    q = ops.elemwise.rmsnorm_per_head_plus_one(q, a.q_norm, d, a.q_norm_eps)
    k = ops.elemwise.rmsnorm_per_head_plus_one(k, a.k_norm, d, a.k_norm_eps)
    q, k = rope(q, k)
    if append:
        ops.attn.kv_append(k, v, pages, write_page, write_offset)
    o = ops.attn.prefill(q, plan, pages, None, d, kv_heads, a.sm_scale)
    return ops.linear.matmul(ops.elemwise.gate_sigmoid_mul(o, gate), a.o_proj)

def moe(x, f):
    """The routed MLP: `f.top_k` of `f.experts`, plus the gated shared expert."""
    experts, weights = ops.linear.moe_topk_softmax(ops.linear.matmul(x, f.router), f.experts, f.top_k)

    def select(act, bank):
        if bank.dtype in [dtype.bf16, dtype.f16, dtype.f32]:
            return ops.linear.moe_matmul_select(act, bank, experts, f.top_k)
        return ops.linear.moe_matmul_select_quant(act, bank, experts, f.top_k)

    hidden = ops.linear.mlp_swiglu(select(x, f.gate_up), f.inter)
    routed = ops.linear.moe_weighted_sum(select(hidden, f.down), weights)
    shared = ops.linear.matmul(
        ops.linear.mlp_swiglu(ops.linear.matmul(x, f.shared_gate_up), f.shared_inter),
        f.shared_down,
    )
    return ops.linear.moe_sigmoid_gate_add(routed, shared, ops.linear.matmul(x, f.shared_gate))

MROPE_SECTIONS = [11, 11, 10]

def positions(inputs, vision):
    """The positions a trunk's attention rotates by: multimodal with a tower."""
    return inputs.mrope_positions() if vision else inputs.positions()

def rope(q, k, positions, vision, rotary_dim, head_dim, theta):
    if not vision:
        return ops.elemwise.rope_partial(q, k, positions, rotary_dim, head_dim, theta)
    return ops.elemwise.rope_mrope(q, k, positions, MROPE_SECTIONS, "interleaved", rotary_dim, head_dim, theta)

def tower(inputs, t):
    """The tower's merged rows for the request's patches."""
    d = t.head_dim
    x = inputs.patches(t.patch_width)
    segments = inputs.patch_segments()
    grid = inputs.patch_positions()

    y = ops.elemwise.add_bias(t.patch_embed_bias, ops.linear.matmul(x, t.patch_embed))
    ids = inputs.patch_embed_rows(t.taps)
    if t.taps == 1:
        pos = ops.layout.embed(ids, t.pos_embed, t.positions)
    else:
        pos = ops.layout.embed_weighted(ids, inputs.patch_embed_weights(t.taps), t.pos_embed, t.positions)
    y = ops.elemwise.residual_add(pos, y)

    for b in t.blocks:
        n = ops.elemwise.layernorm(y, b.norm1, b.norm1_bias, t.norm_eps)
        q, k, v = ops.layout.split_qkv(ops.elemwise.add_bias(b.qkv_bias, ops.linear.matmul(n, b.qkv)), t.hidden, t.hidden)
        q, k = ops.elemwise.rope_mrope(q, k, grid, [0, d // 4, d // 4], "blocked", d, d, t.theta)
        o = ops.attn.dense(q, k, v, segments, d, t.sm_scale)
        y = ops.elemwise.residual_add(ops.elemwise.add_bias(b.proj_bias, ops.linear.matmul(o, b.proj)), y)

        n = ops.elemwise.layernorm(y, b.norm2, b.norm2_bias, t.norm_eps)
        h = ops.elemwise.add_bias(b.fc1_bias, ops.linear.matmul(n, b.fc1))
        a = ops.linear.mlp_gelu_tanh(h)
        y = ops.elemwise.residual_add(ops.elemwise.add_bias(b.fc2_bias, ops.linear.matmul(a, b.fc2)), y)

    mg = t.merger
    n = ops.elemwise.layernorm(y, mg.norm, mg.norm_bias, t.norm_eps)
    folded = ops.layout.merge_rows(n, t.merge)
    h = ops.elemwise.add_bias(mg.fc1_bias, ops.linear.matmul(folded, mg.fc1))
    a = ops.linear.mlp_gelu_tanh(h)
    return ops.elemwise.add_bias(mg.fc2_bias, ops.linear.matmul(a, mg.fc2))

def scatter(y, towered, inputs):
    """`y` with the tower's rows in place of its media tokens' embeddings."""
    imaged = y.on(fact.has(fact.Media))
    return ops.layout.scatter_live_rows(towered, inputs.patch_routes(), imaged).everywhere()

BONSAI_BLOCK = 1024

def rot_copy(x, signs):
    """`H·(S·x)` on a copy of `x`, which a sibling projection still reads."""
    return ops.elemwise.hadamard_signed(ops.elemwise.copy(x), BONSAI_BLOCK, signs)

def rot_take(x, signs):
    """`H·(S·x)` over `x` itself, dead after its rotated matmul."""
    return ops.elemwise.hadamard_signed(x, BONSAI_BLOCK, signs)

def caches(m, c):
    kv = c.kv_space(m.kv)
    plane = m.kv_heads * m.head_dim
    for w in m.layers:
        if w.attn != None:
            c.kv(kv, w.attn.kv, [plane, plane], m.head_dim, heads = True)
        else:
            g = w.gdn
            c.state(g.conv_state, [g.conv_kernel, g.qkv_width], dtype.bf16, split = 1)
            c.state(g.delta_state, [g.v_heads, g.k_dim, g.v_dim], dtype.bf16, split = 0)
    if m.mtp != None:
        c.kv(kv, m.mtp.attn.kv, [plane, plane], m.head_dim, heads = True)
    if m.dflash != None:
        dflash_caches(m.dflash, c, kv)

def forward(m, inputs):
    dr = m.dflash
    trunk_inputs = inputs.on(~block_rows(dr)) if dr != None else inputs
    ([input_m, input_s, input_d], input_p) = trunk_inputs.partition([fact.has(fact.Mask), fact.scores(), fact.single_token()])
    plans = struct(
        m = ops.attn.plan_prefill(input_m, m.q_heads, m.kv_heads, m.head_dim, None),
        d = ops.attn.plan_decode(input_d, m.q_heads, m.kv_heads, m.head_dim, None),
        p = ops.attn.plan_prefill(input_p, m.q_heads, m.kv_heads, m.head_dim, None),
        s = ops.attn.plan_prefill(input_s, m.q_heads, m.kv_heads, m.head_dim, None),
    )
    mask = inputs.mask()

    towered = tower(inputs, m.tower) if m.tower != None else None

    ids = inputs.tokens()
    y = ops.layout.embed(ids, m.embed, m.vocab)

    # Bonsai's token embedding is stored rotated (`H·S·row`), and the residual
    # stream wants `S·(H·row)`: three plain-or-signed Hadamards make that
    # inverse order out of the one op.
    if m.bonsai != None:
        hy = ops.elemwise.hadamard_plain(y, BONSAI_BLOCK)
        hshy = ops.elemwise.hadamard_signed(hy, BONSAI_BLOCK, m.bonsai.hidden)
        y = ops.elemwise.hadamard_plain(hshy, BONSAI_BLOCK)

    if towered != None:
        y = scatter(y, towered, inputs)

    h_block = None
    if dr != None:
        h_block = y.on(block_rows(dr))
        y = y.on(~block_rows(dr))

    routes = inputs.adapter_routes()

    def block(l, w, carried):
        y, fused = carried
        x = ops.elemwise.rmsnorm_plus_one(y, w.mixer_norm, w.mixer_norm_eps)
        if w.attn != None:
            o = attn_mixer(x, inputs, m, plans, mask, w.attn)
        else:
            o = gdn_mixer(x, inputs, w.gdn, m.bonsai)
        adapted = fact.has(fact.Adapter)
        o = ops.linear.lora_correct(x.on(adapted), w.lora_a, w.lora_b, routes, o.on(adapted))
        y = ops.elemwise.residual_add(o, y)

        x = ops.elemwise.rmsnorm_plus_one(y, w.mlp_norm, w.mlp_norm_eps)
        y = ops.elemwise.residual_add(mlp(x, w.mlp, m.bonsai), y)
        if dr != None:
            fused = tap(dr, l, y, fused)
        return (y, fused)

    y, fused = inputs.fold_layers(m.layers, (y, None), block)

    x = ops.elemwise.rmsnorm_plus_one(y, m.final_norm, m.final_norm_eps)
    head = m.embed if m.head == None else m.head
    hb = None
    if dr != None:
        hb = arm(dr, inputs, fused, h_block, mask, block_rows(dr))
        x = merge([hb, x])
    x = ops.layout.gather_rows(x, inputs.readout_rows())
    # Bonsai's output head is stored rotated: its input turns with it.
    head_in = rot_copy(x, m.bonsai.hidden) if m.bonsai != None else x
    logits = ops.linear.lm_head(head_in, head)
    if dr != None:
        plant_readout(dr, logits, inputs, hb, block_rows(dr))
    if m.mtp != None:
        draft(m, inputs, x, logits, head)
    return logits

def mlp(x, f, bonsai):
    if not f.routed:
        # Bonsai: the gate and up projections share the rotated residual;
        # down rotates the swiglu intermediate.
        gate_in = rot_copy(x, bonsai.hidden) if bonsai != None else x
        h = ops.linear.mlp_swiglu(ops.linear.matmul(gate_in, f.gate_up), f.inter)
        down_in = rot_take(h, bonsai.ffn_down) if bonsai != None else h
        return ops.linear.matmul(down_in, f.down)
    return moe(x, f)

def draft(m, inputs, x, logits, head):
    mtp = m.mtp
    plan = ops.attn.plan_prefill(inputs.on(fact.drafts()), m.q_heads, m.kv_heads, m.head_dim, None)
    hidden = x.on(fact.drafts())
    chosen = ops.layout.argmax([logits.on(fact.drafts())])
    e = ops.layout.embed(chosen, m.embed, m.vocab)
    if mtp.pre_fc != None:
        pre = mtp.pre_fc
        e = ops.elemwise.rmsnorm_plus_one(e, pre.embedding, pre.eps)
        hidden = ops.elemwise.rmsnorm_plus_one(hidden, pre.hidden, pre.eps)
    dy = ops.elemwise.residual_add(ops.linear.matmul(e, mtp.fc_embed), ops.linear.matmul(hidden, mtp.fc_hidden))

    nx = ops.elemwise.rmsnorm_plus_one(dy, mtp.mixer_norm, mtp.mixer_norm_eps)
    a = mtp.attn
    o = draft_attn(nx, inputs, plan, a, m.head_dim, m.kv_heads, lambda q, k: rotate(q, k, inputs, m, a))
    dy = ops.elemwise.residual_add(o, dy)

    nx = ops.elemwise.rmsnorm_plus_one(dy, mtp.mlp_norm, mtp.mlp_norm_eps)
    f = ops.linear.matmul(ops.linear.mlp_swiglu(ops.linear.matmul(nx, mtp.mlp.gate_up), mtp.mlp.inter), mtp.mlp.down)
    dy = ops.elemwise.residual_add(f, dy)

    read = ops.elemwise.rmsnorm_plus_one(dy, mtp.norm, mtp.norm_eps) if mtp.norm != None else dy
    proposal = ops.linear.lm_head(read, head)
    seam.at(seam.MTP, [proposal])
    seam.at(seam.MTP_DRAFTS, [ops.layout.argmax([proposal])])

def rotate(q, k, inputs, m, a):
    vision = m.tower != None
    return rope(q, k, positions(inputs, vision), vision, a.rotary_dim, m.head_dim, a.theta)

def attn_mixer(x, inputs, m, plans, mask, a):
    pages = inputs.kv(a.kv)
    write_page = inputs.write_page(a.kv)
    write_offset = inputs.write_offset(a.kv)
    d = m.head_dim
    # Bonsai: q (with its gate), k and v all read one rotated copy of the
    # residual; the LoRA correction still reads it unrotated.
    qkv_in = rot_copy(x, m.bonsai.hidden) if m.bonsai != None else x
    q, gate = ops.layout.split_q_gate(ops.linear.matmul(qkv_in, a.qg_proj), d)
    k = ops.linear.matmul(qkv_in, a.k_proj)
    v = ops.linear.matmul(qkv_in, a.v_proj)
    seam.at(seam.ATTN_QV, [q, v])
    q = ops.elemwise.rmsnorm_per_head_plus_one(q, a.q_norm, d, a.q_norm_eps)
    k = ops.elemwise.rmsnorm_per_head_plus_one(k, a.k_norm, d, a.k_norm_eps)
    q, k = rotate(q, k, inputs, m, a)
    # KV rotation: K and V are cached turned by a Hadamard, Q turned to keep
    # the scores, and the output turned back once more.
    if m.rotate_kv:
        q = ops.elemwise.hadamard_plain(q, d)
        k = ops.elemwise.hadamard_plain(k, d)
        v = ops.elemwise.hadamard_plain(v, d)
    ops.attn.kv_append(k, v, pages, write_page, write_offset)
    seam.at(seam.ATTN_Q, [q])

    def scored(q):
        o, lse = ops.attn.prefill_lse(q, plans.s, pages, None, d, m.kv_heads, a.sm_scale)
        seam.at(seam.SCORES, [lse])
        return o

    o = switch(q, [
        (fact.has(fact.Mask), lambda q: ops.attn.masked(q, plans.m, mask, pages, None, d, m.kv_heads, True, a.sm_scale)),
        (fact.scores(), scored),
        (fact.single_token(), lambda q: ops.attn.decode(q, plans.d, pages, None, d, a.sm_scale)),
    ], otherwise = lambda q: ops.attn.prefill(q, plans.p, pages, None, d, m.kv_heads, a.sm_scale))
    seam.at(seam.ATTN_OUT, [o])
    if m.rotate_kv:
        o = ops.elemwise.hadamard_plain(o, d)
    # Bonsai: the gated output rotates by the 6144-wide signs, as `ssm_out`'s.
    gated = ops.elemwise.gate_sigmoid_mul(o, gate)
    o_in = rot_take(gated, m.bonsai.ssm) if m.bonsai != None else gated
    return ops.linear.matmul(o_in, a.o_proj)

def gdn_mixer(x, inputs, g, bonsai):
    conv_state = inputs.state(g.conv_state)
    delta_state = inputs.state(g.delta_state)
    # Bonsai: in_qkvz reads the rotated residual; in_ba the unrotated one.
    qkvz_in = rot_copy(x, bonsai.hidden) if bonsai != None else x
    qkvz = ops.linear.matmul(qkvz_in, g.in_qkvz)
    ba = ops.linear.matmul(x, g.in_ba)
    seam.at(seam.RECURRENT, [qkvz])
    width = g.qkv_width

    def step(qkvz):
        qkv, z = ops.layout.split_rows(qkvz, width)
        qkv = ops.attn.ssm_causal_conv1d(qkv, g.conv, conv_state, g.conv_kernel)
        gates = ops.attn.ssm_gdn_prep(ba.on(fact.single_token()), g.dt_bias, g.a_log)
        core = ops.attn.ssm_gated_delta(qkv, z, gates, delta_state, g.k_heads, g.v_heads, g.k_dim, g.v_dim)
        return (core, z)

    def chunked(qkvz):
        qkv, z = ops.layout.split_rows(qkvz, width)
        qkv = ops.attn.ssm_causal_conv1d_chunked(qkv, g.conv, conv_state, g.conv_kernel)
        gates = ops.attn.ssm_gdn_prep(ba.on(~fact.single_token()), g.dt_bias, g.a_log)
        core = ops.attn.ssm_gated_delta_chunked(qkv, z, gates, delta_state, g.k_heads, g.v_heads, g.k_dim, g.v_dim)
        return (core, z)

    o, z = switch(qkvz, [(fact.single_token(), step)], otherwise = chunked)
    o = ops.elemwise.rmsnorm_gated(o, z, g.norm, g.v_dim, g.norm_eps, "silu")
    o_in = rot_take(o, bonsai.ssm) if bonsai != None else o
    return ops.linear.matmul(o_in, g.out_proj)
