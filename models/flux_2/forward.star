# The forward of FLUX.2: the text encoder's three tapped layers projected
# into the context, the denoiser over text, image and reference rows, and
# the VAE's decode and encode, each its own reading.

IN_CHANNELS = 128
PACK = 2
HEAD_DIM = 128
ROPE_DIMS = [32, 32, 32, 32]
ROPE_THETA = 2000.0
ROPE_AXES = 4
T_FREQ_DIM = 256
T_MAX_PERIOD = 10000.0
GUIDANCE_SCALE = 1000.0
NORM_EPS = 1e-6
SM_SCALE = 0.08838835
TE_TAPS = [9, 18, 27]
GN_GROUPS = 32
GN_EPS = 1e-6
RGB = 3

def caches(m, c):
    te = m.te
    if te == None:
        return
    kv = c.kv_space(m.kv)
    plane = te.kv_heads * te.head_dim
    for layer in te.layers:
        c.kv(kv, layer.kv, [plane, plane], te.head_dim, heads = True)

def forward(m, inputs):
    if m.te != None:
        inputs.reading("text", lambda rows: text_encode(rows, m.te, m.te_tap))
    velocity = inputs.reading("denoise", lambda rows: denoise(rows, m))
    if m.vae != None:
        inputs.reading("vae.decode", lambda rows: decode(rows, m.vae))
        inputs.reading("vae.encode", lambda rows: encode(rows, m.vae))
    return velocity

def text_encode(arm, te, tap_at):
    plan = ops.attn.plan_prefill(arm, te.q_heads, te.kv_heads, te.head_dim, None)
    ids = arm.tokens()
    positions = arm.positions()
    y = ops.layout.embed(ids, te.embed, te.vocab)
    if tap_at == 0:
        seam.at(seam.HIDDEN, [y])
        return

    def layer(l, w, carried):
        y, ctx, done = carried
        if done:
            return carried
        pages = arm.kv(w.kv)
        x = ops.elemwise.rmsnorm(y, w.attn_norm, te.eps)
        q = ops.linear.matmul(x, w.q)
        k = ops.linear.matmul(x, w.k)
        v = ops.linear.matmul(x, w.v)
        q = ops.elemwise.rmsnorm_per_head(q, w.q_norm, te.head_dim, te.eps)
        k = ops.elemwise.rmsnorm_per_head(k, w.k_norm, te.head_dim, te.eps)
        q, k = ops.elemwise.rope_full(q, k, positions, te.head_dim, te.theta, False)
        ops.attn.kv_append(k, v, pages, arm.write_page(w.kv), arm.write_offset(w.kv))
        o = ops.attn.prefill(q, plan, pages, None, te.head_dim, te.kv_heads, te.sm_scale)
        y = ops.elemwise.residual_add(ops.linear.matmul(o, w.o), y)

        x = ops.elemwise.rmsnorm(y, w.mlp_norm, te.eps)
        f = ops.linear.matmul(ops.linear.mlp_swiglu(ops.linear.matmul(x, w.gate_up), te.inter), w.down)
        y = ops.elemwise.residual_add(f, y)

        if tap_at == l + 1:
            seam.at(seam.HIDDEN, [y])
            return (y, ctx, True)
        if l + 1 in TE_TAPS:
            tap = TE_TAPS.index(l + 1)
            part = ops.linear.matmul(y, te.context_embed[tap])
            total = part if ctx == None else ops.elemwise.residual_add(part, ctx)
            if tap + 1 == len(TE_TAPS):
                seam.at(seam.HIDDEN, [total])
            ctx = total
        return (y, ctx, False)

    arm.fold_layers(te.layers, (y, None, False), layer)

def adaln6(mods, dim):
    a_ss, rest = ops.layout.split_rows(mods, 2 * dim)
    a_gate, rest = ops.layout.split_rows(rest, dim)
    m_ss, m_gate = ops.layout.split_rows(rest, 2 * dim)
    return struct(
        attn = struct(scale_shift = a_ss, gate = a_gate),
        mlp = struct(scale_shift = m_ss, gate = m_gate),
    )

def adaln3(mods, dim):
    scale_shift, gate = ops.layout.split_rows(mods, 2 * dim)
    return struct(scale_shift = scale_shift, gate = gate)

def embed(e, x):
    return ops.linear.matmul(ops.elemwise.silu(ops.linear.matmul(x, e.linear_1)), e.linear_2)

def norm_modulate(x, scale_shift, lanes):
    return ops.elemwise.modulate(ops.elemwise.layernorm_no_scale(x, NORM_EPS), scale_shift, lanes, "scale_shift")

def turn(x, gain, positions):
    return ops.elemwise.rope_axes(
        ops.elemwise.rmsnorm_per_head(x, gain, HEAD_DIM, NORM_EPS),
        positions,
        ROPE_DIMS,
        [ROPE_THETA] * 4,
        "interleaved",
        HEAD_DIM,
        HEAD_DIM,
    )

def heads(x, attn, mods, dim, lanes, positions):
    h = norm_modulate(x, mods.scale_shift, lanes)
    q, k, v = ops.layout.split_qkv(ops.linear.matmul(h, attn.qkv), dim, dim)
    return (turn(q, attn.q_norm, positions), turn(k, attn.k_norm, positions), v)

def joint_attention(q, k, v, j):
    o = ops.attn.ragged(
        ops.layout.pack_rows(q, j.perm),
        ops.layout.pack_rows(k, j.perm),
        ops.layout.pack_rows(v, j.perm),
        j.csr,
        j.csr,
        HEAD_DIM,
        SM_SCALE,
        group_block_diagonal(),
    )
    return ops.layout.unpack_rows(o, j.perm)

def ff_sublayer(x, ff, mods, inter, lanes):
    h = norm_modulate(x, mods.scale_shift, lanes)
    h = ops.linear.mlp_swiglu(ops.linear.matmul(h, ff.linear_in), inter)
    return ops.elemwise.gated_residual_add(x, mods.gate, ops.linear.matmul(h, ff.linear_out), lanes)

def denoise(arm, m):
    d, dit = m.dims, m.dit
    dim = d.dim
    text, image = fact.stream("text"), fact.stream("image")
    txt_in, img_in = arm.on(text), arm.on(~text)

    lanes = arm.request_of_token()
    positions = arm.axis_positions(0, ROPE_AXES)
    joint = struct(perm = arm.row_permutation(), csr = arm.group_indptr())

    t = arm.lane_vector(0, 1)
    temb = embed(dit.t_embed, ops.elemwise.sinusoid(t, T_FREQ_DIM, T_MAX_PERIOD, True, 1.0))
    if dit.g_embed != None:
        g = arm.lane_vector(1, 1)
        gemb = embed(dit.g_embed, ops.elemwise.sinusoid(g, T_FREQ_DIM, T_MAX_PERIOD, True, GUIDANCE_SCALE))
        temb = ops.elemwise.add(temb, gemb)
    stemb = ops.elemwise.silu(temb)
    mod_txt = ops.linear.matmul(stemb, dit.mod_txt).on(text)
    mod_img = ops.linear.matmul(stemb, dit.mod_img).on(~text)
    mod_txt = adaln6(mod_txt, dim)
    mod_img = adaln6(mod_img, dim)
    mod_single = adaln3(ops.linear.matmul(stemb, dit.mod_single), dim)
    mod_out = ops.linear.matmul(stemb, dit.norm_out)
    lanes_txt, lanes_img = lanes.on(text), lanes.on(~text)
    pos_txt, pos_img = positions.on(text), positions.on(~text)

    if dit.context_embed != None:
        txt = ops.linear.matmul(txt_in.context(0, d.context_in), dit.context_embed)
    else:
        c = txt_in.context(0, dim)
        perm = txt_in.row_permutation()
        txt = ops.layout.unpack_rows(ops.layout.pack_rows(c, perm), perm)
    img = ops.linear.matmul(img_in.latents(0, IN_CHANNELS, dtype.bf16), dit.x_embed)

    def double(l, block, carried):
        txt, img = carried
        tq, tk, tv = heads(txt, block.txt.attn, mod_txt.attn, dim, lanes_txt, pos_txt)
        iq, ik, iv = heads(img, block.img.attn, mod_img.attn, dim, lanes_img, pos_img)
        o = joint_attention(merge([tq, iq]), merge([tk, ik]), merge([tv, iv]), joint)
        o_txt, o_img = o.on(text), o.on(~text)
        txt = ops.elemwise.gated_residual_add(
            txt,
            mod_txt.attn.gate,
            ops.linear.matmul(o_txt, block.txt.attn.out),
            lanes_txt,
        )
        img = ops.elemwise.gated_residual_add(
            img,
            mod_img.attn.gate,
            ops.linear.matmul(o_img, block.img.attn.out),
            lanes_img,
        )
        txt = ff_sublayer(txt, block.txt.ff, mod_txt.mlp, d.inter, lanes_txt)
        img = ff_sublayer(img, block.img.ff, mod_img.mlp, d.inter, lanes_img)
        return (txt, img)

    txt, img = arm.fold_layers(dit.double, (txt, img), double)

    def single(l, block, x):
        h = norm_modulate(x, mod_single.scale_shift, lanes)
        proj = ops.linear.matmul(h, block.in_proj)
        qkv, mlp = ops.layout.split_rows(proj, 3 * dim)
        q, k, v = ops.layout.split_qkv(qkv, dim, dim)
        o = joint_attention(turn(q, block.q_norm, positions), turn(k, block.k_norm, positions), v, joint)
        a = ops.linear.matmul(o, block.out_attn)
        f = ops.linear.matmul(ops.linear.mlp_swiglu(mlp, d.inter), block.out_mlp)
        return ops.elemwise.gated_residual_add(x, mod_single.gate, ops.elemwise.residual_add(a, f), lanes)

    x = arm.fold_layers(dit.single, merge([txt, img]), single)

    img_all = x.on(~text)
    target, _ = img_all.on(image), img_all.on(~image)
    _, mod_out = mod_out.on(text), mod_out.on(~text)
    mod_out = mod_out.on(image)
    lanes_target = lanes_img.on(image)
    h = norm_modulate(target, mod_out, lanes_target)
    velocity = ops.linear.matmul(h, dit.proj_out)
    seam.at(seam.VELOCITY, [velocity])
    return velocity

def conv3(x, grid, c):
    shape = conv([1, 1], [1, 1], [0, 0]) if c.taps == 1 else conv([3, 3], [1, 1], [1, 1])
    return ops.spatial.conv3d(x, grid, c.w, c.bias, shape, None)[0]

def group_norm(x, grid, n, silu):
    return ops.spatial.group_norm(x, grid, GN_GROUPS, n.weight, n.bias, GN_EPS, silu)

def resnet(x, grid, r):
    h = group_norm(x, grid, r.norm1, True)
    h = conv3(h, grid, r.conv1)
    h = group_norm(h, grid, r.norm2, True)
    h = conv3(h, grid, r.conv2)
    skip = conv3(x, grid, r.shortcut) if r.shortcut != None else x
    return ops.elemwise.add(skip, h)

def biased(p, x):
    return ops.elemwise.add_bias(p.bias, ops.linear.matmul(x, p.w))

def attention(x, grid, a):
    h = group_norm(x, grid, a.norm, False)
    q = biased(a.q, h)
    k = biased(a.k, h)
    v = biased(a.v, h)
    o = ops.spatial.attention(q, k, v, grid, f32(1.0 / f32(sqrt(a.width))))
    return ops.elemwise.add(x, biased(a.out, o))

def mid(x, grid, m):
    h = resnet(x, grid, m.res0)
    h = attention(h, grid, m.attn)
    return resnet(h, grid, m.res1)

def decode(arm, vae):
    d = vae.decoder
    g0 = arm.grid()
    z = arm.voxels(0, IN_CHANNELS, dtype.bf16)
    z, g0 = ops.spatial.upsample_nearest(z, g0, [1, 1, 1], False)
    z = ops.elemwise.standardize(z, vae.bn_zero, vae.bn_scale)
    z = ops.elemwise.add_bias(vae.bn_mean, z)
    z, grid = ops.spatial.pixel_shuffle(z, g0, [1, PACK, PACK])
    z = conv3(z, grid, vae.post_quant_conv)
    h = conv3(z, grid, d.conv_in)
    h = mid(h, grid, d.mid)
    for block in d.up:
        for res in block.resnets:
            h = resnet(h, grid, res)
        if block.upsample != None:
            up, grid = ops.spatial.upsample_nearest(h, grid, [1, 2, 2], False)
            h = conv3(up, grid, block.upsample)
    h = group_norm(h, grid, d.norm_out, True)
    y = conv3(h, grid, d.conv_out)
    seam.at(seam.PIXELS, [y, grid])
    return y

def encode(arm, vae):
    e = vae.encoder
    grid = arm.grid()
    x = arm.voxels(1, RGB, dtype.bf16)
    h = conv3(x, grid, e.conv_in)
    for block in e.down:
        for res in block.resnets:
            h = resnet(h, grid, res)
        if block.downsample != None:
            c = block.downsample
            h, grid = ops.spatial.conv3d(h, grid, c.w, c.bias, conv([3, 3], [2, 2], [0, 0], pad_back = [0, 1, 1]), None)
    h = mid(h, grid, e.mid)
    h = group_norm(h, grid, e.norm_out, True)
    h = conv3(h, grid, e.conv_out)
    mean = conv3(h, grid, vae.quant_conv)
    packed, grid = ops.spatial.pixel_unshuffle(mean, grid, [1, PACK, PACK])
    z = ops.elemwise.standardize(packed, vae.bn_mean, vae.bn_rscale)
    seam.at(seam.PIXELS, [z, grid])
    return z
