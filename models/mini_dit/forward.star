# The forward of mini-dit: a single-stream block over text and image, a
# double-stream block attending both jointly, and a cross-attention block of
# the image over the context; each modulated by the timestep. A tap ends the
# forward at the first intermediate it names, read out as the velocity.

def linear(w, x):
    """`x` through the projection `w`, plus its bias if it has one."""
    y = ops.linear.matmul(x, w.w)
    if w.bias == None:
        return y
    return ops.elemwise.add_bias(w.bias, y)

def chunks(x, widths):
    """`x`'s rows cut into consecutive pieces of `widths`, the last one
    whatever remains."""
    out = []
    for width in widths[:-1]:
        head, x = ops.layout.split_rows(x, width)
        out.append(head)
    out.append(x)
    return out

def norm_modulate(x, scale_shift, lanes, eps):
    """`x` layer-normed without a gain, then scaled and shifted per lane."""
    return ops.elemwise.modulate(ops.elemwise.layernorm_no_scale(x, eps), scale_shift, lanes, "scale_shift")

def attend(q, k, v, rows, over, head_dim, sm_scale, mask):
    """`q`'s rows attending `k` and `v`'s, each side grouped by its own
    `perm` and `csr`."""
    o = ops.attn.ragged(
        ops.layout.pack_rows(q, rows.perm),
        ops.layout.pack_rows(k, over.perm),
        ops.layout.pack_rows(v, over.perm),
        rows.csr,
        over.csr,
        head_dim,
        sm_scale,
        mask,
    )
    return ops.layout.unpack_rows(o, rows.perm)

TIMESTEP_DIM = 256
ROPE_DIMS = [16, 24, 24, 0]
ROPE_THETA = 10000.0
LN_EPS = 1e-6
RMS_EPS = 1e-6

def caches(m, c):
    pass

def probe(at, stem):
    """`at` over the intermediates of the block (or block side) `stem`."""
    sep = "" if stem.endswith("_") else "."
    return lambda tail, v: at(stem + sep + tail, v)

def adaln(mods, m):
    return chunks(mods, [2 * m.hidden, m.hidden, 2 * m.hidden, m.hidden])

def heads(m, at, stem, x, attn, mods, lanes, positions):
    """`x`'s q, k and v, turned; None if a tap ends the forward on the way."""
    tap = probe(at, stem)
    h = norm_modulate(x, mods, lanes, LN_EPS)
    if tap("norm1_out", h):
        return None
    q, k, v = ops.layout.split_qkv(linear(attn.qkv, h), m.hidden, m.hidden)
    hp = "self_" if stem == "b2" else ""
    for tail, value in [("q_raw", q), ("k_raw", k), ("v", v)]:
        if tap(hp + tail, value):
            return None
    turned = []
    for tail, value, gain in [("q", q, attn.q_norm), ("k", k, attn.k_norm)]:
        n = ops.elemwise.rmsnorm_per_head(value, gain, m.head_dim, RMS_EPS)
        if tap(hp + tail + "_qknorm", n):
            return None
        r = ops.elemwise.rope_axes(n, positions, ROPE_DIMS, [ROPE_THETA] * 4, "interleaved", m.head_dim, m.head_dim)
        if tap(hp + tail + "_rope", r):
            return None
        turned.append(r)
    return (turned[0], turned[1], v)

def joint_attention(m, q, k, v, rows):
    return attend(q, k, v, rows, rows, m.head_dim, m.sm_scale, group_block_diagonal())

def attn_sublayer(m, at, stem, x, attn, mods, gate, lanes, positions, rows):
    qkv = heads(m, at, stem, x, attn, mods, lanes, positions)
    if qkv == None:
        return None
    o = joint_attention(m, qkv[0], qkv[1], qkv[2], rows)
    tap = probe(at, stem)
    prefix = "self_" if stem == "b2" else ""
    if tap(prefix + "attn_heads", o):
        return None
    o = linear(attn.out, o)
    if tap(prefix + "attn_out", o):
        return None
    return ops.elemwise.gated_residual_add(x, gate, o, lanes)

def mlp_sublayer(m, at, stem, x, mlp, mods, gate, lanes):
    tap = probe(at, stem)
    h = norm_modulate(x, mods, lanes, LN_EPS)
    if tap("norm3_out" if stem == "b2" else "norm2_out", h):
        return None
    y = linear(mlp.down, ops.linear.mlp_swiglu(linear(mlp.gate_up, h), m.inter))
    if tap("mlp_out", y):
        return None
    return ops.elemwise.gated_residual_add(x, gate, y, lanes)

def forward(m, inputs):
    ctx, joint = inputs.on(fact.stream("context")), inputs.on(~fact.stream("context"))
    txt_in, img_in = joint.on(fact.stream("text")), joint.on(~fact.stream("text"))

    lanes = inputs.request_of_token()
    positions = inputs.axis_positions(0, m.rope_axes)

    t = inputs.lane_vector(0, 1)
    temb = ops.elemwise.silu(ops.elemwise.sinusoid(t, TIMESTEP_DIM, 10000.0, False, 1.0))

    joint_rows = struct(perm = joint.row_permutation(), csr = joint.group_indptr())
    img_rows = struct(perm = img_in.row_permutation(), csr = img_in.group_indptr())

    txt = txt_in.context(0, m.text_width)
    patches = img_in.latents(0, m.patch_features, dtype.bf16)
    ctx_rows = ctx.context(1, m.context_width)

    hit = []

    def at(key, v):
        """Whether `v` is the intermediate `m` taps; read out if so."""
        if m.tap != key:
            return False
        seam.at(seam.VELOCITY, [v])
        hit.append(v)
        return True

    if at("in.text", ops.elemwise.layernorm_no_scale(txt, LN_EPS)):
        return hit[0]
    img = linear(m.x_embed, patches)
    if at("x_embed", img):
        return img

    x = merge([txt, img])
    if at("b0.in", x):
        return x
    msa, gate_a, mmlp, gate_m = adaln(linear(m.single.ada, temb), m)
    x = attn_sublayer(m, at, "b0", x, m.single.attn, msa, gate_a, lanes, positions, joint_rows)
    if x == None:
        return hit[0]
    if at("b0.x_after_attn", x):
        return x
    x = mlp_sublayer(m, at, "b0", x, m.single.mlp, mmlp, gate_m, lanes)
    if x == None:
        return hit[0]
    if at("b0.out", x):
        return x

    txt, img = x.on(fact.stream("text")), x.on(~fact.stream("text"))
    txt_mod = adaln(linear(m.double.txt.ada, temb), m)
    img_mod = adaln(linear(m.double.img.ada, temb), m)
    tqkv = heads(m, at, "b1.txt_", txt, m.double.txt.attn, txt_mod[0], lanes, positions)
    if tqkv == None:
        return hit[0]
    iqkv = heads(m, at, "b1.img_", img, m.double.img.attn, img_mod[0], lanes, positions)
    if iqkv == None:
        return hit[0]
    o = joint_attention(m, merge([tqkv[0], iqkv[0]]), merge([tqkv[1], iqkv[1]]), merge([tqkv[2], iqkv[2]]), joint_rows)
    if at("b1.joint_attn_heads", o):
        return o
    o_txt, o_img = o.on(fact.stream("text")), o.on(~fact.stream("text"))
    ta = linear(m.double.txt.attn.out, o_txt)
    if at("b1.txt_attn_out", ta):
        return ta
    ia = linear(m.double.img.attn.out, o_img)
    if at("b1.img_attn_out", ia):
        return ia
    txt = ops.elemwise.gated_residual_add(txt, txt_mod[1], ta, lanes)
    if at("b1.txt_after_attn", txt):
        return txt
    img = ops.elemwise.gated_residual_add(img, img_mod[1], ia, lanes)
    if at("b1.img_after_attn", img):
        return img
    txt = mlp_sublayer(m, at, "b1.txt_", txt, m.double.txt.mlp, txt_mod[2], txt_mod[3], lanes)
    if txt == None:
        return hit[0]
    if at("b1.out_txt", txt):
        return txt
    x = mlp_sublayer(m, at, "b1.img_", img, m.double.img.mlp, img_mod[2], img_mod[3], lanes)
    if x == None:
        return hit[0]
    if at("b1.out_img", x):
        return x

    mods = adaln(ops.elemwise.add_bias(m.cross.mod_table, linear(m.cross.ada, temb)), m)
    msa, gate_a, mffn, gate_f = mods
    x = attn_sublayer(m, at, "b2", x, m.cross.self_attn, msa, gate_a, lanes, positions, img_rows)
    if x == None:
        return hit[0]
    if at("b2.x_after_self", x):
        return x

    hc = ops.elemwise.layernorm(x, m.cross.norm, m.cross.norm_bias, LN_EPS)
    if at("b2.cross_norm_out", hc):
        return hc
    cross = m.cross.cross
    cq = ops.elemwise.rmsnorm_per_head(linear(cross.q, hc), cross.q_norm, m.head_dim, RMS_EPS)
    if at("b2.cross_q", cq):
        return cq
    ck, cv = ops.layout.split_rows(linear(cross.kv, ctx_rows), m.hidden)
    ck = ops.elemwise.rmsnorm_per_head(ck, cross.k_norm, m.head_dim, RMS_EPS)
    ctx_group = struct(perm = ctx.row_permutation(), csr = ctx.group_indptr())
    ca = attend(cq, ck, cv, img_rows, ctx_group, m.head_dim, m.sm_scale, group_block_diagonal())
    if at("b2.cross_attn_heads", ca):
        return ca
    ca = linear(cross.out, ca)
    if at("b2.cross_attn_out", ca):
        return ca
    x = ops.elemwise.residual_add(ca, x)
    if at("b2.x_after_cross", x):
        return x
    x = mlp_sublayer(m, at, "b2", x, m.cross.mlp, mffn, gate_f, lanes)
    if x == None:
        return hit[0]
    if at("b2.out", x):
        return x

    hf = norm_modulate(x, linear(m.final_ada, temb), lanes, LN_EPS)
    if at("final.norm_out", hf):
        return hf
    velocity = linear(m.final_proj, hf)
    seam.at(seam.VELOCITY, [velocity])
    return velocity
