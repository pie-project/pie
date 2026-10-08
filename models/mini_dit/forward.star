# The forward of mini-dit: a single-stream block over text and image, a
# double-stream block attending both jointly, and a cross-attention block of
# the image over the context; each modulated by the timestep.

HIDDEN = 256
HEAD_DIM = 64
INTER = 512
PATCH_FEATURES = 64
TEXT_WIDTH = 256
CONTEXT_WIDTH = 512
TIMESTEP_DIM = 256
ROPE_DIMS = [16, 24, 24, 0]
ROPE_THETA = 10000.0
ROPE_AXES = 3
LN_EPS = 1e-6
RMS_EPS = 1e-6
SM_SCALE = 0.125

def caches(m, c):
    pass

def tapped(m, stem, tail, v):
    """`v`, read out as the velocity, if it is the intermediate `m` taps."""
    key = stem + tail if stem == "" or stem.endswith("_") else stem + "." + tail
    if m.tap == key:
        seam.at(seam.VELOCITY, [v])
        return struct(tapped = v)
    return None

def linear(w, x):
    return ops.elemwise.add_bias(w.bias, ops.linear.matmul(x, w.w))

def adaln6(m):
    msa, rest = ops.layout.split_rows(m, 2 * HIDDEN)
    gate_a, rest = ops.layout.split_rows(rest, HIDDEN)
    mmlp, gate_m = ops.layout.split_rows(rest, 2 * HIDDEN)
    return (msa, gate_a, mmlp, gate_m)

def heads(model, stem, x, attn, m, lanes, positions):
    h = ops.elemwise.modulate(ops.elemwise.layernorm_no_scale(x, LN_EPS), m, lanes, "scale_shift")
    t = tapped(model, stem, "norm1_out", h)
    if t:
        return t
    q, k, v = ops.layout.split_qkv(linear(attn.qkv, h), HIDDEN, HIDDEN)
    hp = "self_" if stem == "b2" else ""
    for tail, value in [("q_raw", q), ("k_raw", k), ("v", v)]:
        t = tapped(model, stem, hp + tail, value)
        if t:
            return t

    def turn(x, gain, tail):
        n = ops.elemwise.rmsnorm_per_head(x, gain, HEAD_DIM, RMS_EPS)
        t = tapped(model, stem, hp + tail + "_qknorm", n)
        if t:
            return t
        r = ops.elemwise.rope_axes(n, positions, ROPE_DIMS, [ROPE_THETA] * 4, "interleaved", HEAD_DIM, HEAD_DIM)
        t = tapped(model, stem, hp + tail + "_rope", r)
        if t:
            return t
        return r

    q = turn(q, attn.q_norm, "q")
    if type(q) == "struct":
        return q
    k = turn(k, attn.k_norm, "k")
    if type(k) == "struct":
        return k
    return (q, k, v)

def joint_attention(q, k, v, perm, csr):
    o = ops.attn.ragged(
        ops.layout.pack_rows(q, perm),
        ops.layout.pack_rows(k, perm),
        ops.layout.pack_rows(v, perm),
        csr,
        csr,
        HEAD_DIM,
        SM_SCALE,
        group_block_diagonal(),
    )
    return ops.layout.unpack_rows(o, perm)

def attn_sublayer(model, stem, x, attn, m, gate, lanes, positions, perm, csr):
    qkv = heads(model, stem, x, attn, m, lanes, positions)
    if type(qkv) == "struct":
        return qkv
    q, k, v = qkv
    o = joint_attention(q, k, v, perm, csr)
    heads_key, out_key = ("self_attn_heads", "self_attn_out") if stem == "b2" else ("attn_heads", "attn_out")
    t = tapped(model, stem, heads_key, o)
    if t:
        return t
    o = linear(attn.out, o)
    t = tapped(model, stem, out_key, o)
    if t:
        return t
    return ops.elemwise.gated_residual_add(x, gate, o, lanes)

def mlp_sublayer(model, stem, x, mlp, m, gate, lanes):
    h = ops.elemwise.modulate(ops.elemwise.layernorm_no_scale(x, LN_EPS), m, lanes, "scale_shift")
    t = tapped(model, stem, "norm3_out" if stem == "b2" else "norm2_out", h)
    if t:
        return t
    h = ops.linear.mlp_swiglu(linear(mlp.gate_up, h), INTER)
    y = linear(mlp.down, h)
    t = tapped(model, stem, "mlp_out", y)
    if t:
        return t
    return ops.elemwise.gated_residual_add(x, gate, y, lanes)

def context_kv(c, cross):
    k, v = ops.layout.split_rows(linear(cross.kv, c), HIDDEN)
    return (ops.elemwise.rmsnorm_per_head(k, cross.k_norm, HEAD_DIM, RMS_EPS), v)

def forward(m, inputs):
    ctx, joint = inputs.on(fact.stream("context")), inputs.on(~fact.stream("context"))
    txt_in, img_in = joint.on(fact.stream("text")), joint.on(~fact.stream("text"))

    lanes = inputs.request_of_token()
    positions = inputs.axis_positions(0, ROPE_AXES)

    t = inputs.lane_vector(0, 1)
    temb = ops.elemwise.silu(ops.elemwise.sinusoid(t, TIMESTEP_DIM, 10000.0, False, 1.0))

    joint_perm = joint.row_permutation()
    joint_csr = joint.group_indptr()
    img_perm = img_in.row_permutation()
    img_csr = img_in.group_indptr()

    txt = txt_in.context(0, TEXT_WIDTH)
    patches = img_in.latents(0, PATCH_FEATURES, dtype.bf16)
    ctx_rows = ctx.context(1, CONTEXT_WIDTH)

    # Each point a tap may read out at, in turn; the first one tapped ends
    # the forward there.
    def at(key, v):
        return tapped(m, "", key, v)

    r = at("in.text", ops.elemwise.layernorm_no_scale(txt, LN_EPS))
    if r:
        return r.tapped
    img = linear(m.x_embed, patches)
    r = at("x_embed", img)
    if r:
        return r.tapped

    x = merge([txt, img])
    r = at("b0.in", x)
    if r:
        return r.tapped
    msa, gate_a, mmlp, gate_m = adaln6(linear(m.single.ada, temb))
    x = attn_sublayer(m, "b0", x, m.single.attn, msa, gate_a, lanes, positions, joint_perm, joint_csr)
    if type(x) == "struct":
        return x.tapped
    r = at("b0.x_after_attn", x)
    if r:
        return r.tapped
    x = mlp_sublayer(m, "b0", x, m.single.mlp, mmlp, gate_m, lanes)
    if type(x) == "struct":
        return x.tapped
    r = at("b0.out", x)
    if r:
        return r.tapped

    txt, img = x.on(fact.stream("text")), x.on(~fact.stream("text"))
    txt_mod = adaln6(linear(m.double.txt.ada, temb))
    img_mod = adaln6(linear(m.double.img.ada, temb))
    tqkv = heads(m, "b1.txt_", txt, m.double.txt.attn, txt_mod[0], lanes, positions)
    if type(tqkv) == "struct":
        return tqkv.tapped
    iqkv = heads(m, "b1.img_", img, m.double.img.attn, img_mod[0], lanes, positions)
    if type(iqkv) == "struct":
        return iqkv.tapped
    o = joint_attention(
        merge([tqkv[0], iqkv[0]]),
        merge([tqkv[1], iqkv[1]]),
        merge([tqkv[2], iqkv[2]]),
        joint_perm,
        joint_csr,
    )
    r = at("b1.joint_attn_heads", o)
    if r:
        return r.tapped
    o_txt, o_img = o.on(fact.stream("text")), o.on(~fact.stream("text"))
    ta = linear(m.double.txt.attn.out, o_txt)
    r = at("b1.txt_attn_out", ta)
    if r:
        return r.tapped
    ia = linear(m.double.img.attn.out, o_img)
    r = at("b1.img_attn_out", ia)
    if r:
        return r.tapped
    txt = ops.elemwise.gated_residual_add(txt, txt_mod[1], ta, lanes)
    r = at("b1.txt_after_attn", txt)
    if r:
        return r.tapped
    img = ops.elemwise.gated_residual_add(img, img_mod[1], ia, lanes)
    r = at("b1.img_after_attn", img)
    if r:
        return r.tapped
    txt = mlp_sublayer(m, "b1.txt_", txt, m.double.txt.mlp, txt_mod[2], txt_mod[3], lanes)
    if type(txt) == "struct":
        return txt.tapped
    r = at("b1.out_txt", txt)
    if r:
        return r.tapped
    x = mlp_sublayer(m, "b1.img_", img, m.double.img.mlp, img_mod[2], img_mod[3], lanes)
    if type(x) == "struct":
        return x.tapped
    r = at("b1.out_img", x)
    if r:
        return r.tapped

    mod2 = ops.elemwise.add_bias(m.cross.mod_table, linear(m.cross.ada, temb))
    msa, gate_a, mffn, gate_f = adaln6(mod2)
    x = attn_sublayer(m, "b2", x, m.cross.self_attn, msa, gate_a, lanes, positions, img_perm, img_csr)
    if type(x) == "struct":
        return x.tapped
    r = at("b2.x_after_self", x)
    if r:
        return r.tapped

    hc = ops.elemwise.layernorm(x, m.cross.norm, m.cross.norm_bias, LN_EPS)
    r = at("b2.cross_norm_out", hc)
    if r:
        return r.tapped
    cq = ops.elemwise.rmsnorm_per_head(linear(m.cross.cross.q, hc), m.cross.cross.q_norm, HEAD_DIM, RMS_EPS)
    r = at("b2.cross_q", cq)
    if r:
        return r.tapped
    ck, cv = context_kv(ctx_rows, m.cross.cross)
    ctx_perm = ctx.row_permutation()
    ctx_csr = ctx.group_indptr()
    ca = ops.attn.ragged(
        ops.layout.pack_rows(cq, img_perm),
        ops.layout.pack_rows(ck, ctx_perm),
        ops.layout.pack_rows(cv, ctx_perm),
        img_csr,
        ctx_csr,
        HEAD_DIM,
        SM_SCALE,
        group_block_diagonal(),
    )
    ca = ops.layout.unpack_rows(ca, img_perm)
    r = at("b2.cross_attn_heads", ca)
    if r:
        return r.tapped
    ca = linear(m.cross.cross.out, ca)
    r = at("b2.cross_attn_out", ca)
    if r:
        return r.tapped
    x = ops.elemwise.residual_add(ca, x)
    r = at("b2.x_after_cross", x)
    if r:
        return r.tapped
    x = mlp_sublayer(m, "b2", x, m.cross.mlp, mffn, gate_f, lanes)
    if type(x) == "struct":
        return x.tapped
    r = at("b2.out", x)
    if r:
        return r.tapped

    scale_shift = linear(m.final_ada, temb)
    hf = ops.elemwise.modulate(ops.elemwise.layernorm_no_scale(x, LN_EPS), scale_shift, lanes, "scale_shift")
    r = at("final.norm_out", hf)
    if r:
        return r.tapped
    velocity = linear(m.final_proj, hf)
    seam.at(seam.VELOCITY, [velocity])
    return velocity
