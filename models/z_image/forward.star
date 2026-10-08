# The forward of Z-Image: the text encoder's last hidden rows, the context
# refiner over the caption, the denoiser over the image and the refined
# context jointly, and the VAE's decode and encode, each its own reading.

CHANNELS = 16
PATCH_FEATURES = 64
T_FREQ_DIM = 256
T_MAX_PERIOD = 10000.0
NORM_EPS = 1e-5
FINAL_LN_EPS = 1e-6
ROPE_AXES = 3
ROPE_THETA = 256.0
SCALING_FACTOR = 0.3611
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
        inputs.reading("text", lambda rows: text_encode(rows, m.te))
    inputs.reading("refine", lambda rows: refine(rows, m))
    velocity = inputs.reading("denoise", lambda rows: denoise(rows, m))
    if m.vae != None:
        inputs.reading("vae.decode", lambda rows: decode(rows, m.vae))
        inputs.reading("vae.encode", lambda rows: encode(rows, m.vae))
    return velocity

def text_encode(arm, te):
    plan = ops.attn.plan_prefill(arm, te.q_heads, te.kv_heads, te.head_dim, None)
    ids = arm.tokens()
    positions = arm.positions()
    y = ops.layout.embed(ids, te.embed, te.vocab)
    last = len(te.layers) - 1

    def layer(l, w, y):
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
        if l == last:
            seam.at(seam.HIDDEN, [y])
        return y

    arm.fold_layers(te.layers, y, layer)

def linear(w, x):
    return ops.elemwise.add_bias(w.bias, ops.linear.matmul(x, w.w))

def pad_rows(x, flag, bank):
    return ops.elemwise.modulate(x, ops.linear.matmul(flag, bank), None, "scale_shift")

def refine(arm, m):
    d, dit = m.dims, m.dit
    c = arm.context(0, d.cap_width)
    c = ops.elemwise.rmsnorm(c, dit.cap_norm, NORM_EPS)
    c = linear(dit.cap_embed, c)
    c = pad_rows(c, arm.latents(0, 1, dtype.bf16), dit.cap_pad_mod)
    geom = struct(
        positions = arm.axis_positions(0, ROPE_AXES),
        perm = arm.row_permutation(),
        csr = arm.lane_indptr(),
        mask = unmasked(),
    )
    last = len(dit.context_refiner) - 1

    def layer(l, b, c):
        c = run_block(c, b, None, m, geom, "")[0]
        if l == last:
            seam.at(seam.HIDDEN, [c])
        return c

    arm.fold_layers(dit.context_refiner, c, layer)

def adaln4(mods, dim):
    scale_msa, rest = ops.layout.split_rows(mods, dim)
    gate_msa, rest = ops.layout.split_rows(rest, dim)
    scale_mlp, gate_mlp = ops.layout.split_rows(rest, dim)
    return struct(
        scale_msa = scale_msa,
        gate_msa = ops.elemwise.tanh(gate_msa),
        scale_mlp = scale_mlp,
        gate_mlp = ops.elemwise.tanh(gate_mlp),
    )

def seam_tapped(image, context):
    width = image.width()
    if width < context.width():
        context = ops.layout.split_rows(context, width)[0]
    both = merge([image, context])
    seam.at(seam.VELOCITY, [both])
    return both

def tapped(value):
    """The forward's end, read out at a tap."""
    return struct(tapped = value)

def denoise(arm, m):
    d, dit = m.dims, m.dit
    image = fact.stream("image")
    img, ctx = arm.on(image), arm.on(~image)

    lanes = arm.request_of_token()
    positions = arm.axis_positions(0, ROPE_AXES)
    joint = struct(
        positions = positions,
        perm = arm.row_permutation(),
        csr = arm.group_indptr(),
        mask = group_block_diagonal(),
    )

    t = arm.lane_vector(0, 1)
    u = ops.elemwise.mul_scalar(-0.5, ops.elemwise.add(t, t))
    u = ops.elemwise.add_bias(dit.t_flip, u)
    temb = ops.elemwise.sinusoid(u, T_FREQ_DIM, T_MAX_PERIOD, True, 1.0)
    temb = linear(dit.t_mlp1, ops.elemwise.silu(linear(dit.t_mlp0, temb)))

    img_lanes = lanes.on(image)
    img_positions = positions.on(image)
    temb_img = temb.on(image)
    own = struct(
        positions = img_positions,
        perm = img.row_permutation(),
        csr = img.lane_indptr(),
        mask = unmasked(),
    )
    x = img.latents(1, PATCH_FEATURES, dtype.bf16)
    tap = m.tap if m.tap != None else ""
    c_early = ctx.latents(2, d.dim, dtype.bf16) if tap != "" else None
    if tap == "latents":
        return seam_tapped(x, c_early)
    x = linear(dit.x_embed, x)
    if tap == "x_linear":
        return seam_tapped(x, c_early)
    flag = img.latents(0, 1, dtype.bf16)
    x = pad_rows(x, flag, dit.x_pad_mod)
    if tap == "x_embed":
        return seam_tapped(x, c_early)

    def refiner(l, b, x):
        if type(x) == "struct":
            return x
        mods = adaln4(linear(b.ada, temb_img), d.dim)
        if l == 0:
            if tap == "normed0":
                return tapped(seam_tapped(ops.elemwise.rmsnorm(x, b.attn_norm1, NORM_EPS), c_early))
            if tap == "scaled0":
                h = ops.elemwise.rmsnorm(x, b.attn_norm1, NORM_EPS)
                return tapped(seam_tapped(ops.elemwise.modulate(h, mods.scale_msa, img_lanes, "scale"), c_early))
        want = tap if l == 0 else ""
        x, hit = run_block(x, b, (mods, img_lanes), m, own, want)
        if hit != None:
            return tapped(seam_tapped(hit, c_early))
        if tap == "refiner{}".format(l):
            return tapped(seam_tapped(x, c_early))
        return x

    x = img.fold_layers(dit.noise_refiner, x, refiner)
    if type(x) == "struct":
        return x.tapped

    if tap == "x_refined":
        return seam_tapped(x, c_early)
    c = c_early if c_early != None else ctx.latents(2, d.dim, dtype.bf16)

    def layer(l, b, u):
        if type(u) == "struct":
            return u
        mods = adaln4(linear(b.ada, temb), d.dim)
        u = run_block(u, b, (mods, lanes), m, joint, "")[0]
        if tap == "layer{}".format(l):
            both = merge([u.on(image), u.on(~image)])
            seam.at(seam.VELOCITY, [both])
            return tapped(both)
        return u

    u = arm.fold_layers(dit.layers, merge([x, c]), layer)
    if type(u) == "struct":
        return u.tapped

    scale = linear(dit.final_ada, ops.elemwise.silu(temb))
    scale_img = scale.on(image)
    ui = u.on(image)
    h = ops.elemwise.modulate(
        ops.elemwise.layernorm_no_scale(ui, FINAL_LN_EPS),
        scale_img,
        img_lanes,
        "scale",
    )
    if tap == "final_norm":
        return seam_tapped(h, c_early)
    v = linear(dit.final_linear, h)
    if tap == "final_linear":
        return seam_tapped(v, c_early)
    velocity = ops.elemwise.mul_scalar(-1.0, v)
    seam.at(seam.VELOCITY, [velocity])
    return velocity

def run_block(x, b, mods, m, g, tap):
    """The block's output, and the value at `tap` if it names one of its
    first block's intermediates."""
    d = m.dims
    dim, hd = d.dim, d.head_dim
    hit = [None]

    def at(name, v):
        if tap == name:
            hit[0] = v

    h = ops.elemwise.rmsnorm(x, b.attn_norm1, NORM_EPS)
    if mods != None:
        h = ops.elemwise.modulate(h, mods[0].scale_msa, mods[1], "scale")
    q, k, v = ops.layout.split_qkv(ops.linear.matmul(h, b.attn.qkv), dim, dim)

    def turn(x, gain):
        return ops.elemwise.rope_axes(
            ops.elemwise.rmsnorm_per_head(x, gain, hd, NORM_EPS),
            g.positions,
            d.rope_dims,
            [ROPE_THETA] * 4,
            "interleaved",
            hd,
            hd,
        )

    q = turn(q, b.attn.q_norm)
    k = turn(k, b.attn.k_norm)
    at("b0.q", q)
    at("b0.k", k)
    at("b0.v", v)
    o = ops.attn.ragged(
        ops.layout.pack_rows(q, g.perm),
        ops.layout.pack_rows(k, g.perm),
        ops.layout.pack_rows(v, g.perm),
        g.csr,
        g.csr,
        hd,
        m.sm_scale,
        g.mask,
    )
    o = ops.layout.unpack_rows(o, g.perm)
    at("b0.attn", o)
    o = ops.linear.matmul(o, b.attn.out)
    at("b0.out", o)
    o = ops.elemwise.rmsnorm(o, b.attn_norm2, NORM_EPS)
    at("b0.norm2", o)
    if mods != None:
        x = ops.elemwise.gated_residual_add(x, mods[0].gate_msa, o, mods[1])
    else:
        x = ops.elemwise.residual_add(o, x)
    at("b0.res1", x)

    h = ops.elemwise.rmsnorm(x, b.ffn_norm1, NORM_EPS)
    if mods != None:
        h = ops.elemwise.modulate(h, mods[0].scale_mlp, mods[1], "scale")
    f = ops.linear.matmul(ops.linear.mlp_swiglu(ops.linear.matmul(h, b.mlp.gate_up), d.inter), b.mlp.down)
    at("b0.ffn", f)
    f = ops.elemwise.rmsnorm(f, b.ffn_norm2, NORM_EPS)
    at("b0.ffn_norm2", f)
    if mods != None:
        out = ops.elemwise.gated_residual_add(x, mods[0].gate_mlp, f, mods[1])
    else:
        out = ops.elemwise.residual_add(f, x)
    at("b0.res2", out)
    return (out, hit[0])

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

def attention(x, grid, a):
    h = group_norm(x, grid, a.norm, False)
    q = linear(a.q, h)
    k = linear(a.k, h)
    v = linear(a.v, h)
    o = ops.spatial.attention(q, k, v, grid, f32(1.0 / f32(sqrt(a.width))))
    return ops.elemwise.add(x, linear(a.out, o))

def mid(x, grid, m):
    h = resnet(x, grid, m.res0)
    h = attention(h, grid, m.attn)
    return resnet(h, grid, m.res1)

def decode(arm, vae):
    d = vae.decoder
    grid = arm.grid()
    z = arm.voxels(0, CHANNELS, dtype.bf16)
    z = ops.elemwise.add(z, z)
    z = ops.elemwise.add_bias(vae.shift, ops.elemwise.mul_scalar(f32(0.5 * f32(1.0 / f32(SCALING_FACTOR))), z))
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
    mean = conv3(h, grid, e.conv_out)
    seam.at(seam.PIXELS, [mean, grid])
    return mean
