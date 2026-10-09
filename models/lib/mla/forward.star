# The forward of multi-head latent attention: the queries absorbed into the
# latent space, scored against the cached latent rows (all of them, or the
# ones an indexer selects), and the result expanded back out per head.

def cache(c, space, a):
    """`a`'s latent rows in the kv `space`."""
    c.kv(space, a.kv, [a.kv_lora_rank, a.qk_rope_head_dim], a.qk_rope_head_dim)

def plans(inputs, heads, kv_lora_rank):
    """The decode and prefill arms' plans over `inputs`."""
    one = fact.single_token()
    decode, prefill = inputs.on(one), inputs.on(~one)
    return struct(
        decode = ops.attn.mla_plan(decode, heads, kv_lora_rank),
        prefill = ops.attn.mla_plan(prefill, heads, kv_lora_rank),
    )

def whole(inputs, heads, kv_lora_rank):
    """A plan that reads every row of `inputs` through the prefill arm."""
    return struct(decode = None, prefill = ops.attn.mla_plan(inputs, heads, kv_lora_rank))

def attention(x, inputs, plan, a, positions = None, select = None):
    """Latent attention of the rows `x`. `select(q_a)`, if given, picks the
    keys each query reads; `positions` default to the rows' own."""
    pages = inputs.kv(a.kv)
    if positions == None:
        positions = inputs.positions()
    write_page = inputs.write_page(a.kv)
    write_offset = inputs.write_offset(a.kv)

    q_a = ops.linear.matmul(x, a.q_a_proj)
    q_a = ops.elemwise.rmsnorm(q_a, a.q_a_norm, a.q_a_norm_eps)
    q_b = ops.linear.matmul(q_a, a.q_b_proj)
    kv_a = ops.linear.matmul(x, a.kv_a_proj)
    seam.at(seam.ATTN_QV, [q_b, kv_a])

    selection = select(q_a) if select != None else None

    if a.theta != None:
        kv_c, k_pe = ops.attn.mla_latents_rope(
            kv_a,
            positions,
            a.kv_a_norm,
            a.kv_a_norm_eps,
            a.kv_lora_rank,
            a.qk_rope_head_dim,
            a.theta,
        )
    else:
        kv_c, k_pe = ops.attn.mla_latents(kv_a, a.kv_a_norm, a.kv_a_norm_eps, a.kv_lora_rank)
    ops.attn.mla_kv_append(kv_c, k_pe, pages, write_page, write_offset)

    q_nope, q_pe = ops.attn.mla_split_q_b(q_b, a.heads, a.qk_nope_head_dim, a.qk_rope_head_dim)
    if a.theta != None:
        q_pe = ops.elemwise.rope_partial_q(q_pe, positions, a.qk_rope_head_dim, a.qk_rope_head_dim, a.theta)
    return attend(x, q_nope, q_pe, pages, plan, a, selection)

def attend(x, q_nope, q_pe, pages, plan, a, selection = None):
    """The split queries `q_nope`/`q_pe` of the rows `x` scored against the
    latent `pages` (or the `selection` of them), through `a`'s output
    projection. A plan with no decode arm reads its rows whole."""
    q = ops.attn.mla_absorb_q(q_nope, a.kv_b_proj, a.heads, a.kv_lora_rank, a.qk_nope_head_dim, a.v_head_dim)
    seam.at(seam.ATTN_Q, [q])

    if plan.decode == None:
        scored = arm(False, q, plan.prefill, q_pe, selection, pages, a)
    else:
        one = fact.single_token()
        on = lambda rows, holds: rows.on(holds) if rows != None else None
        scored = merge([
            arm(True, q.on(one), plan.decode, q_pe.on(one), on(selection, one), pages, a),
            arm(False, q.on(~one), plan.prefill, q_pe.on(~one), on(selection, ~one), pages, a),
        ])

    o = ops.attn.mla_absorb_out(scored, a.kv_b_proj, a.heads, a.kv_lora_rank, a.qk_nope_head_dim, a.v_head_dim)
    if a.gate != None:
        o = ops.elemwise.gate_sigmoid_mul(o, ops.linear.matmul(x, a.gate))
    seam.at(seam.ATTN_OUT, [o])
    return ops.linear.matmul(o, a.o_proj)

def arm(decode, q, plan, q_pe, selection, pages, a):
    if selection == None:
        score = ops.attn.mla_decode if decode else ops.attn.mla_prefill
        return score(q, plan, q_pe, pages, a.heads, a.kv_lora_rank, a.sm_scale)
    score = ops.attn.mla_decode_selected if decode else ops.attn.mla_prefill_selected
    return score(q, plan, q_pe, selection, pages, a.heads, a.kv_lora_rank, a.sm_scale)

def boundaries(positions, row_valid, ratio, split = True):
    """Where the rows close entries of a key pool that compresses `ratio`
    rows into one: each boundary's position, request and rope position.
    With `split` the decode and prefill arms find theirs apart."""
    if not split:
        return ops.attn.pool_boundary_prefill(positions, row_valid, ratio)
    one = fact.single_token()
    dpos, dreq, drope = ops.attn.pool_boundary_decode(positions.on(one), row_valid, ratio)
    ppos, preq, prope = ops.attn.pool_boundary_prefill(positions.on(~one), row_valid, ratio)
    return merge([dpos, ppos]), merge([dreq, preq]), merge([drope, prope])
