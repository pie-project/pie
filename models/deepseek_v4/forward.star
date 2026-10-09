# The forward of DeepSeek-V4: each layer reads its sub-block inputs out of
# the residual streams, attends over a sliding window and, where it pools, a
# compressed (and on V4.1 sparsely selected) pool, and folds its outputs back;
# V4.1's Engram writes n-gram memories into the streams before a layer.

load("//lib/hyper/forward.star", "collapse", "gate", "mixes", "predict_route", "summed")
load("//lib/mla/forward.star", "boundaries")

def caches(m, c):
    kv = c.kv_space(m.kv)
    for w in m.layers:
        at = w.attn
        c.kv(kv, at.kv, [at.kv_down.shape[0]], m.head_dim, heads = at.kv_down.cut_axis != None)
        if at.pool != None and at.pool.owner:
            pool = c.kv_space(m.kv)
            c.kv(pool, at.pool.entries, [m.head_dim], m.head_dim)
        if at.indexer != None and at.indexer.owns_keys:
            index = c.kv_space(m.kv)
            c.kv(index, at.indexer.keys, [at.indexer.head_dim], at.indexer.head_dim)
        if w.engram != None:
            c.state(w.engram.ids_state, [w.engram.ngram - 1], dtype.i32)
    if m.mtp != None:
        at = m.mtp.block.attn
        c.kv(kv, at.kv, [at.kv_down.shape[0]], m.head_dim, heads = at.kv_down.cut_axis != None)

def kv_heads(m):
    if not m.layers:
        return m.heads
    row = m.layers[0].attn.kv_down.shape[0]
    if m.head_dim == 0 or row % m.head_dim != 0:
        fail("the cached row is {} wide and the head width is {}".format(row, m.head_dim))
    return row // m.head_dim

def forward(m, inputs):
    hy = m.hyper
    positions = inputs.positions()
    kvh = kv_heads(m)
    plan_p = ops.attn.plan_prefill(inputs, m.heads, kvh, m.head_dim, m.window)
    ids = inputs.tokens()
    streams = ops.elemwise.hc_expand(ops.layout.embed(ids, m.embed, m.vocab), hy.streams)

    adapter_routes = inputs.adapter_routes()
    carry = {"mix": None, "selection": None}

    def block(l, w, streams):
        if w.engram != None:
            streams = engram(streams, ids, inputs, m, w.engram)
        nxt = m.layers[l + 1] if l + 1 < len(m.layers) else None
        return layer(m, inputs, plan_p, positions, adapter_routes, w, nxt, streams, ids, False, carry)

    streams = inputs.fold_layers(m.layers, streams, block)

    if m.hc_head != None:
        y = collapse(streams, m.hc_head, hy)
    elif hy.single_pass:
        y = premix(streams, carry["mix"], m)
    else:
        y = summed(streams, m.hidden, hy.streams)
    x = ops.elemwise.rmsnorm(y, m.final_norm, m.final_norm_eps)
    x = ops.layout.gather_rows(x, inputs.readout_rows())
    logits = ops.linear.lm_head(x, m.head if m.head != None else m.embed)

    if m.mtp != None and m.head != None:
        mtp = m.mtp
        input_mtp = inputs.on(fact.drafts())
        plan_mtp = ops.attn.plan_prefill(input_mtp, m.heads, kvh, m.head_dim, m.window)
        dstreams = ops.layout.gather_rows(streams, inputs.readout_rows()).on(fact.drafts())
        dpos = positions.on(fact.drafts())
        dlogits = logits.on(fact.drafts())

        token = ops.layout.argmax([dlogits])
        hidden = dstreams
        chain = []
        draft_carry = {"mix": None, "selection": None}
        for step in range(mtp.depth):
            e = ops.layout.embed(token, m.embed, m.vocab)
            e = ops.elemwise.rmsnorm(e, mtp.enorm, mtp.norm_eps)
            e = ops.elemwise.hc_expand(ops.linear.matmul(e, mtp.e_proj), hy.streams)
            h = ops.elemwise.rmsnorm_per_head(hidden, mtp.hnorm, m.hidden, mtp.norm_eps)
            routes = ops.linear.group_routes(h, hy.streams)
            h = ops.linear.matmul_grouped(h, mtp.h_proj, routes, hy.streams)
            fused = ops.elemwise.residual_add(e, h)

            out = layer(m, input_mtp, plan_mtp, dpos, adapter_routes, mtp.block, None, fused, token,
                        step > 0, draft_carry)
            dy = collapse(out, mtp.hc_head, hy)
            read = ops.elemwise.rmsnorm(dy, mtp.norm, mtp.norm_eps)
            draft = ops.linear.lm_head(read, m.head)
            if step == 0:
                seam.at(seam.MTP, [draft])
            token = ops.layout.argmax([draft])
            hidden = out
            chain.append(draft)
        seam.at(seam.MTP_DRAFTS, [ops.layout.argmax(chain)])

    return logits

def rope(x, pos, rope_dim, head_dim, theta, scaling, inverse = False):
    if scaling != None:
        scaling = yarn(
            factor = scaling.factor,
            beta_fast = scaling.beta_fast,
            beta_slow = scaling.beta_slow,
            original_max_position = scaling.original_max_position,
        )
    return ops.elemwise.rope_partial_last_yarn(x, pos, rope_dim, head_dim, theta, True, inverse, scaling)

def layer(m, inputs, plan_p, positions, adapter_routes, w, nxt, streams, ids, chain, carry):
    hy = m.hyper
    kvh = kv_heads(m)
    pos = positions
    at = w.attn
    pages = inputs.kv(at.kv)
    write_page = inputs.write_page(at.kv)
    write_offset = inputs.write_offset(at.kv)

    x, post_mix, comb_mix = sublayer_input(streams, w.attn_mix, m, carry)
    if w.attn_norm != None:
        x = ops.elemwise.rmsnorm(x, w.attn_norm, hy.norm_eps)

    q_a = ops.linear.matmul(x, at.q_down)
    q_a = ops.elemwise.rmsnorm(q_a, at.q_norm, at.q_norm_eps)
    q = ops.linear.matmul(q_a, at.q_up)
    q = ops.elemwise.rmsnorm_no_scale(q, m.head_dim, at.q_norm_eps)
    q = rope(q, pos, at.rope_dim, m.head_dim, at.theta, at.yarn)
    seam.at(seam.ATTN_Q, [q])

    plane = ops.linear.matmul(x, at.kv_down)
    plane = ops.elemwise.rmsnorm(plane, at.kv_norm, at.kv_norm_eps)
    plane = rope(plane, pos, at.rope_dim, m.head_dim, at.theta, at.yarn)
    if not chain:
        ops.attn.kv_append_shared(plane, pages, write_page, write_offset)

    o, lse = ops.attn.prefill_lse(q, plan_p, pages, m.window, m.head_dim, kvh, at.sm_scale)

    p = at.pool
    if p != None:
        entries = inputs.kv(p.entries)
        row_valid = inputs.row_valid()
        request_of_token = inputs.request_of_token()
        bpos, breq, brope = boundaries(pos, row_valid, p.ratio)

        if p.owner:
            latent = compress(m, x, p, pages, write_page, write_offset, bpos, breq, chain)
            ix = at.indexer
            if ix != None and ix.wk != None and ix.k_norm != None and ix.owns_keys and not chain:
                # CSA2: the index keys are a projection of the compressed
                # latent, taken before its rotation.
                k = ops.elemwise.rmsnorm(ops.linear.matmul(latent, ix.wk), ix.k_norm, hy.norm_eps)
                k = rope(k, brope, ix.rope_dim, ix.head_dim, ix.theta, ix.yarn)
                ops.attn.pool_kv_append(k, bpos, breq, inputs.kv(ix.keys), inputs.write_page(ix.keys), inputs.write_offset(ix.keys))
            pooled = rope(latent, brope, at.rope_dim, m.head_dim, at.theta, at.yarn)
            if not chain:
                ops.attn.pool_kv_append(pooled, bpos, breq, entries, inputs.write_page(p.entries), inputs.write_offset(p.entries))

        selection = None
        if at.selection == "own":
            ix = at.indexer
            if ix.compressor != None:
                s = indexer(x, q_a, ix, pos, bpos, breq, brope, inputs.kv(ix.keys),
                            inputs.write_page(ix.keys), inputs.write_offset(ix.keys), m.act, chain)
            else:
                s = rank(x, q_a, ix, pos, inputs.kv(ix.keys), p.ratio)
            carry["selection"] = (s, ix.top_k)
            selection = (s, ix.top_k)
        elif at.selection == "shared":
            selection = carry["selection"]
            if selection == None:
                fail("a CSA2 Reuse Mode layer follows a layer that ranked")
        if selection != None:
            po, plse = ops.attn.pool_lse_selected(q, pos, request_of_token, selection[0], entries,
                                                  p.ratio, selection[1], m.heads, m.head_dim, at.sm_scale)
        else:
            po, plse = ops.attn.pool_lse(q, pos, request_of_token, entries, p.ratio, m.heads, m.head_dim, at.sm_scale)
        o, lse = ops.attn.merge_lse(o, lse, po, plse, m.heads, m.head_dim)

    o = ops.attn.sink(o, lse, at.sink, m.head_dim)
    o = rope(o, pos, at.rope_dim, m.head_dim, at.theta, at.yarn, inverse = True)
    seam.at(seam.ATTN_OUT, [o])

    if at.o_groups > 1:
        routes = ops.linear.group_routes(o, at.o_groups)
        o = ops.linear.matmul_grouped(o, at.o_down, routes, at.o_groups)
    else:
        o = ops.linear.matmul(o, at.o_down)
    o = ops.linear.matmul(o, at.o_up)
    adapted = fact.has(fact.Adapter)
    o = ops.linear.lora_correct(x.on(adapted), w.lora_a, w.lora_b, adapter_routes, o.on(adapted))
    streams = ops.elemwise.hc_fold(o, streams, post_mix, comb_mix)

    x, post_mix, comb_mix = sublayer_input(streams, w.mlp_mix, m, carry)
    if w.mlp_norm != None:
        x = ops.elemwise.rmsnorm(x, w.mlp_norm, hy.norm_eps)
    f = mlp(x, ids, w.mlp, streams, nxt, hy)
    return ops.elemwise.hc_fold(f, streams, post_mix, comb_mix)

def compress(m, x, p, pages, write_page, write_offset, bpos, breq, chain):
    c = p.compressor
    if c == None:
        return ops.attn.pool_gather(bpos, breq, pages, None, m.head_dim, p.ratio, m.act)
    if c.wgate != None:
        if not chain:
            state_kv = ops.linear.matmul(x, c.wkv)
            state_score = ops.linear.matmul(x, c.wgate)
            ops.attn.pool_state_write(state_kv, state_score, pages, write_page, write_offset, m.head_dim, p.ratio)
        pooled = ops.attn.pool_gather(bpos, breq, pages, c.ape, m.head_dim, p.ratio, m.act)
        return ops.elemwise.rmsnorm(pooled, c.norm, c.norm_eps)
    kv = ops.linear.matmul(x, c.wkv)
    return ops.elemwise.rmsnorm(kv, c.norm, c.norm_eps)

def sublayer_input(streams, mix, m, carry):
    hy = m.hyper
    if not hy.single_pass:
        return gate(streams, mix, hy)
    normed = ops.elemwise.hc_rmsnorm_f32(streams, hy.norm_eps)
    mixed = mixes(normed, mix, hy)
    _, post_mix, comb_mix = ops.elemwise.hc_gates(mixed, streams, mix.scale, mix.base, hy.streams,
                                                  hy.gate_eps, hy.alpha, hy.sinkhorn)
    x = premix(streams, carry["mix"], m)
    carry["mix"] = struct(mixes = mixed, scale = mix.scale, base = mix.base)
    return x, post_mix, comb_mix

def premix(streams, prev, m):
    hy = m.hyper
    if prev != None:
        x, _, _ = ops.elemwise.hc_gates(prev.mixes, streams, prev.scale, prev.base, hy.streams,
                                        hy.gate_eps, hy.alpha, hy.sinkhorn)
        return x
    return ops.layout.split_rows(streams, m.hidden)[0]

def engram(streams, ids, inputs, m, e):
    hy = m.hyper
    state = inputs.state(e.ids_state)
    one = fact.single_token()
    ids_d, ids_p = ids.on(one), ids.on(~one)
    grams = merge([
        ops.attn.ple_ngram_ids(ids_d, state, e.pad, e.mults, e.primes, e.offsets, e.heads_per_ngram, m.token_map),
        ops.attn.ple_ngram_ids_chunked(ids_p, state, e.pad, e.mults, e.primes, e.offsets, e.heads_per_ngram, m.token_map),
    ])
    rows = 0
    for p in e.primes:
        rows += p
    fetched = ops.layout.embed_concat(grams, e.table, rows)
    kv = ops.linear.matmul(fetched, e.wkv)
    key, value = ops.layout.split_rows(kv, hy.streams * m.hidden)
    key = ops.elemwise.rmsnorm_grouped_plus_one(key, e.k_weight, m.hidden, e.eps)
    query = ops.elemwise.rmsnorm_grouped_plus_one(streams, e.q_weight, m.hidden, e.eps)
    gated = ops.elemwise.ple_gate(key, query, value, hy.streams)
    return ops.elemwise.residual_add(gated, streams)

def predict_next(streams, nxt, hy):
    if nxt == None:
        return None
    f = nxt.mlp
    if f.kind != "moe_flash" or f.gate.kind != "bias":
        return None
    if hy.single_pass:
        # The next layer's input mix is this layer's own prediction, which is
        # not settled here: the hint would score the wrong row.
        return None
    return predict_route(streams, hy, nxt.mlp_mix, nxt.mlp_norm, hy.norm_eps, f.router, f.gate.bias, f.experts)

def mlp(x, ids, f, streams, nxt, hy):
    if f.kind == "dense":
        return ops.linear.matmul(ops.linear.mlp_swiglu_clamp(ops.linear.matmul(x, f.gate_up), f.inter, f.limit), f.down)
    if f.kind == "routed":
        routes, weights = ops.linear.moe_topk_sqrt_softplus(
            ops.linear.matmul(x, f.router), f.bias, f.experts, f.top_k, f.renorm, f.scaling)
        hidden = ops.linear.moe_matmul_select(x, f.gate_up, routes, f.top_k)
        act = ops.linear.mlp_swiglu_clamp(hidden, f.inter, f.limit)
        return ops.linear.moe_weighted_sum(ops.linear.moe_matmul_select(act, f.down, routes, f.top_k), weights)
    if f.gate.kind == "bias":
        hint = predict_next(streams, nxt, hy)
        routes, weights = ops.linear.moe_topk_sqrt_softplus_hinted(
            ops.linear.matmul(x, f.router), f.gate.bias, f.experts, f.top_k, f.renorm, f.scaling, hint)
    else:
        vocab = f.gate.tid2eid.shape[0]
        routes, weights = ops.linear.moe_hash_route(
            ids, f.gate.tid2eid, ops.linear.matmul(x, f.router), vocab, f.experts, f.top_k, f.renorm, f.scaling)
    shared = ops.linear.matmul(
        ops.linear.mlp_swiglu_clamp(ops.linear.matmul(x, f.shared_gate_up), f.shared_inter, f.limit),
        f.shared_down,
    )

    def select(act, bank):
        if bank.dtype in [dtype.bf16, dtype.f16, dtype.f32]:
            return ops.linear.moe_matmul_select(act, bank, routes, f.top_k)
        return ops.linear.moe_matmul_select_quant(act, bank, routes, f.top_k)

    if f.gate_up.fused != None:
        act = ops.linear.mlp_swiglu_clamp(select(x, f.gate_up.fused), f.inter, f.limit)
    else:
        act = ops.linear.mlp_swiglu_clamp_split(select(x, f.gate_up.gate), select(x, f.gate_up.up), f.limit)
    routed = ops.linear.moe_weighted_sum(select(act, f.down), weights)
    return ops.elemwise.residual_add(shared, routed)

def indexer(x, q_a, ix, positions, bpos, breq, brope, keys, write_page, write_offset, act, chain):
    c = ix.compressor
    ratio = c.ape.shape[0]
    if not chain:
        state_kv = ops.linear.matmul(x, c.wkv)
        state_score = ops.linear.matmul(x, c.wgate)
        ops.attn.pool_state_write(state_kv, state_score, keys, write_page, write_offset, ix.head_dim, ratio)
    k = ops.attn.pool_gather(bpos, breq, keys, c.ape, ix.head_dim, ratio, act)
    k = ops.elemwise.rmsnorm(k, c.norm, c.norm_eps)
    k = rope(k, brope, ix.rope_dim, ix.head_dim, ix.theta, ix.yarn)
    if not chain:
        ops.attn.pool_kv_append(k, bpos, breq, keys, write_page, write_offset)
    q = ops.linear.matmul(q_a, ix.wq_b)
    q = rope(q, positions, ix.rope_dim, ix.head_dim, ix.theta, ix.yarn)
    weights = ops.linear.matmul(x, ix.weights_proj)
    return ops.attn.index_topk(q, weights, keys, ix.heads, ix.head_dim, ix.top_k, ratio)

def rank(x, q_a, ix, positions, keys, ratio):
    q = ops.linear.matmul(q_a, ix.wq_b)
    q = rope(q, positions, ix.rope_dim, ix.head_dim, ix.theta, ix.yarn)
    weights = ops.linear.matmul(x, ix.weights_proj)
    return ops.attn.index_topk(q, weights, keys, ix.heads, ix.head_dim, ix.top_k, ratio)
