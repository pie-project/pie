# The forward of a DFlash block drafter: it reads the trunk's tapped residual
# stream into its blocks' kv, then drafts a masked block of rows through
# them, its proposals read out of the trunk's own head.

def caches(d, c, space):
    for b in d.blocks:
        a = b.attn
        plane = a.kv_heads * a.head_dim
        c.kv(space, a.kv, [plane, plane], a.head_dim, heads = True)

def reads_a_mask(d):
    """Whether a block attends through the lane's mask: one with no window
    attends every row the mask shows it, both ways."""
    for b in d.blocks:
        if b.window == None:
            return True
    return False

def block_rows(d):
    """The rows of a drafted block. A drafter whose attention reads the mask
    drafts only rows that carry one; the runtime stamps the whole extent on a
    block lane its inferlet sent without."""
    if reads_a_mask(d):
        return fact.block_draft() & fact.has(fact.Mask)
    return fact.block_draft()

def tap(d, layer, y, fused):
    """`fused`, with the trunk's residual `y` after `layer` added if the
    drafter taps it."""
    if layer not in d.taps:
        return fused
    part = ops.linear.matmul(y, d.fc[d.taps.index(layer)])
    if fused == None:
        return part
    return ops.elemwise.residual_add(part, fused)

def biased(x, bias):
    if bias == None:
        return x
    return ops.elemwise.add_bias(bias, x)

def conv_prepare(x, c):
    if c == None:
        return x, None
    coeff = ops.linear.matmul(x, c.proj)
    return ops.attn.block_dyn_conv(x, coeff, c.base, 0, c.taps, c.group), coeff

def conv_finish(y, c, coeff):
    if c == None or coeff == None:
        return y
    return ops.attn.block_dyn_conv(y, coeff, c.base, 1, c.taps, c.group)

def arm(d, inputs, fused, h_block, mask, block_draft):
    """The drafted block's hidden rows, from the trunk's `fused` taps and
    the block's embedded `h_block`."""
    h_ctx = ops.elemwise.rmsnorm_plus_one(fused, d.hidden_norm, d.hidden_norm_eps)
    ctx_positions = inputs.positions().on(~block_draft)
    for b in d.blocks:
        a = b.attn
        hd = a.head_dim
        k = biased(ops.linear.matmul(h_ctx, a.k_proj), a.k_bias)
        v = biased(ops.linear.matmul(h_ctx, a.v_proj), a.v_bias)
        k = ops.elemwise.rmsnorm_per_head_plus_one(k, a.k_norm, hd, a.k_norm_eps)
        k = ops.elemwise.rope_partial_q(k, ctx_positions, a.rotary_dim, hd, a.theta)
        ops.attn.kv_append(k, v, inputs.kv(a.kv), inputs.write_page(a.kv), inputs.write_offset(a.kv))

    input_block = inputs.on(block_draft)
    block_positions = inputs.positions().on(block_draft)
    h = h_block
    for b in d.blocks:
        a = b.attn
        hd = a.head_dim
        plan = ops.attn.plan_prefill(input_block, a.q_heads, a.kv_heads, hd, b.window)
        x = ops.elemwise.rmsnorm_plus_one(h, b.mixer_norm, b.mixer_norm_eps)
        x, attn_coeff = conv_prepare(x, b.attn_conv)
        q = biased(ops.linear.matmul(x, a.q_proj), a.q_bias)
        k = biased(ops.linear.matmul(x, a.k_proj), a.k_bias)
        v = biased(ops.linear.matmul(x, a.v_proj), a.v_bias)
        q = ops.elemwise.rmsnorm_per_head_plus_one(q, a.q_norm, hd, a.q_norm_eps)
        k = ops.elemwise.rmsnorm_per_head_plus_one(k, a.k_norm, hd, a.k_norm_eps)
        q, k = ops.elemwise.rope_partial(q, k, block_positions, a.rotary_dim, hd, a.theta)
        ops.attn.kv_append(k, v, inputs.kv(a.kv), inputs.write_page(a.kv), inputs.write_offset(a.kv))
        if b.window != None:
            o = ops.attn.prefill(q, plan, inputs.kv(a.kv), b.window, hd, a.kv_heads, a.sm_scale)
        else:
            o = ops.attn.masked(q, plan, mask, inputs.kv(a.kv), None, hd, a.kv_heads, False, a.sm_scale)
        o = biased(ops.linear.matmul(o, a.o_proj), a.o_bias)
        o = conv_finish(o, b.attn_conv, attn_coeff)
        h = ops.elemwise.residual_add(o, h)

        x = ops.elemwise.rmsnorm_plus_one(h, b.mlp_norm, b.mlp_norm_eps)
        x, mlp_coeff = conv_prepare(x, b.mlp_conv)
        f = ops.linear.matmul(ops.linear.mlp_swiglu(ops.linear.matmul(x, b.mlp.gate_up), b.mlp.inter), b.mlp.down)
        f = conv_finish(f, b.mlp_conv, mlp_coeff)
        h = ops.elemwise.residual_add(f, h)
    return ops.elemwise.rmsnorm_plus_one(h, d.norm, d.norm_eps)

def plant_readout(d, logits, inputs, hb, block_draft):
    """States the drafter on the trace and plants its proposals: the block's
    logits, and the tokens drafted from them."""
    block_drafter(
        logits,
        rows = d.block,
        mask_token = d.mask_token,
        bidirectional = reads_a_mask(d),
        proposals_from = d.proposals_from,
    )
    dlogits = logits.on(block_draft)
    seam.at(seam.MTP, [dlogits])
    sel = d.selector
    if sel != None and hb != None:
        unary, cand = ops.layout.topk(dlogits, sel.top_k)
        hp = ops.linear.matmul(hb, sel.hidden_projection) if sel.hidden_projection != None else None
        toks = inputs.tokens().on(block_draft)
        picks = ops.attn.selector_walk(cand, unary, hp, toks, sel.pred, sel.succ, d.proposals_from)
    else:
        picks = ops.layout.argmax([dlogits])
    seam.at(seam.MTP_DRAFTS, [picks])
