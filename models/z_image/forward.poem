# The forward of Z-Image: the text encoder's last hidden rows, the context
# refiner over the caption, the denoiser over the image and the refined
# context jointly, and the VAE's decode and encode, each its own reading.

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

GN_GROUPS = 32

GN_EPS = 1e-6

RGB = 3

def convolve(x, grid, c):
    shape = conv([1, 1], [1, 1], [0, 0]) if c.taps == 1 else conv([3, 3], [1, 1], [1, 1])
    return ops.spatial.conv3d(x, grid, c.w, c.bias, shape, None)[0]

def group_norm(x, grid, n, silu):
    return ops.spatial.group_norm(x, grid, GN_GROUPS, n.weight, n.bias, GN_EPS, silu)

def resnet(x, grid, r):
    h = group_norm(x, grid, r.norm1, True)
    h = convolve(h, grid, r.conv1)
    h = group_norm(h, grid, r.norm2, True)
    h = convolve(h, grid, r.conv2)
    skip = convolve(x, grid, r.shortcut) if r.shortcut != None else x
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

def vae_decode(vae, z, grid):
    """The pixels the decoder makes of the latent `z`, read out."""
    d = vae.decoder
    h = convolve(z, grid, d.conv_in)
    h = mid(h, grid, d.mid)
    for block in d.up:
        for res in block.resnets:
            h = resnet(h, grid, res)
        if block.upsample != None:
            up, grid = ops.spatial.upsample_nearest(h, grid, [1, 2, 2], False)
            h = convolve(up, grid, block.upsample)
    h = group_norm(h, grid, d.norm_out, True)
    y = convolve(h, grid, d.conv_out)
    seam.at(seam.PIXELS, [y, grid])
    return y

def vae_encode(vae, arm):
    """The encoder's output over `arm`'s pixels, and its grid."""
    e = vae.encoder
    grid = arm.grid()
    x = arm.voxels(1, RGB, dtype.bf16)
    h = convolve(x, grid, e.conv_in)
    for block in e.down:
        for res in block.resnets:
            h = resnet(h, grid, res)
        if block.downsample != None:
            c = block.downsample
            h, grid = ops.spatial.conv3d(h, grid, c.w, c.bias, conv([3, 3], [2, 2], [0, 0], pad_back = [0, 1, 1]), None)
    h = mid(h, grid, e.mid)
    h = group_norm(h, grid, e.norm_out, True)
    return convolve(h, grid, e.conv_out), grid

def qwen3_caches(te, c, kv):
    space = c.kv_space(kv)
    plane = te.kv_heads * te.head_dim
    for layer in te.layers:
        c.kv(space, layer.kv, [plane, plane], te.head_dim, heads = True)

def qwen3_encode(arm, te, tap = None):
    """The encoder over `arm`'s tokens. `tap(l, y)` sees each layer's output
    rows and reads the encoder out; without one, the last layer's rows are
    its readout."""
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
        if tap != None:
            tap(l, y)
        elif l == last:
            seam.at(seam.HIDDEN, [y])
        return y

    arm.fold_layers(te.layers, y, layer)

T_MAX_PERIOD = 10000.0
NORM_EPS = 1e-5
FINAL_LN_EPS = 1e-6
ROPE_THETA = 256.0
SCALING_FACTOR = 0.3611

def caches(m, c):
    if m.te != None:
        qwen3_caches(m.te, c, m.kv)

def forward(m, inputs):
    if m.te != None:
        inputs.reading("text", lambda rows: qwen3_encode(rows, m.te))
    inputs.reading("refine", lambda rows: refine(rows, m))
    velocity = inputs.reading("denoise", lambda rows: denoise(rows, m))
    if m.vae != None:
        inputs.reading("vae.decode", lambda rows: decode(rows, m))
        inputs.reading("vae.encode", lambda rows: encode(rows, m))
    return velocity

def pad_rows(x, flag, bank):
    return ops.elemwise.modulate(x, ops.linear.matmul(flag, bank), None, "scale_shift")

def refine(arm, m):
    d, dit = m.dims, m.dit
    c = arm.context(0, d.cap_width)
    c = ops.elemwise.rmsnorm(c, dit.cap_norm, NORM_EPS)
    c = linear(dit.cap_embed, c)
    c = pad_rows(c, arm.latents(0, 1, dtype.bf16), dit.cap_pad_mod)
    geom = struct(
        positions = arm.axis_positions(0, m.rope_axes),
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

def adaln(mods, dim):
    scale_msa, gate_msa, scale_mlp, gate_mlp = chunks(mods, [dim, dim, dim, dim])
    return struct(
        scale_msa = scale_msa,
        gate_msa = ops.elemwise.tanh(gate_msa),
        scale_mlp = scale_mlp,
        gate_mlp = ops.elemwise.tanh(gate_mlp),
    )

def seam_tapped(image, context):
    """`image` and `context`, cut to `image`'s width, read out as the
    velocity at a tap."""
    width = image.width()
    if width < context.width():
        context = ops.layout.split_rows(context, width)[0]
    both = merge([image, context])
    seam.at(seam.VELOCITY, [both])
    return both

def denoise(arm, m):
    d, dit = m.dims, m.dit
    image = fact.stream("image")
    img, ctx = arm.on(image), arm.on(~image)

    lanes = arm.request_of_token()
    positions = arm.axis_positions(0, m.rope_axes)
    joint = struct(
        positions = positions,
        perm = arm.row_permutation(),
        csr = arm.group_indptr(),
        mask = group_block_diagonal(),
    )

    # 1000 - t. An in-place op cannot write a runtime input, so `t + t`
    # makes the value the scale owns, the scale halving it back.
    t = arm.lane_vector(0, 1)
    u = ops.elemwise.add_bias(dit.t_flip, ops.elemwise.mul_scalar(-0.5, ops.elemwise.add(t, t)))
    temb = ops.elemwise.sinusoid(u, m.t_freq_dim, T_MAX_PERIOD, True, 1.0)
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
    x = img.latents(1, m.patch_features, dtype.bf16)

    # A tap reads the context in first, and ends the forward at its
    # intermediate; a fold carries `struct(tapped = ...)` past its rest.
    tap = m.tap
    context = ctx.latents(2, d.dim, dtype.bf16) if tap else None
    stop = lambda v: struct(tapped = seam_tapped(v, context))
    if tap == "latents":
        return seam_tapped(x, context)
    x = linear(dit.x_embed, x)
    if tap == "x_linear":
        return seam_tapped(x, context)
    x = pad_rows(x, img.latents(0, 1, dtype.bf16), dit.x_pad_mod)
    if tap == "x_embed":
        return seam_tapped(x, context)

    def refiner(l, b, x):
        if type(x) == "struct":
            return x
        mods = adaln(linear(b.ada, temb_img), d.dim)
        if l == 0 and tap in ["normed0", "scaled0"]:
            h = ops.elemwise.rmsnorm(x, b.attn_norm1, NORM_EPS)
            if tap == "scaled0":
                h = ops.elemwise.modulate(h, mods.scale_msa, img_lanes, "scale")
            return stop(h)
        x, hit = run_block(x, b, (mods, img_lanes), m, own, tap if l == 0 else "")
        if hit != None:
            return stop(hit)
        if tap == "refiner{}".format(l):
            return stop(x)
        return x

    x = img.fold_layers(dit.noise_refiner, x, refiner)
    if type(x) == "struct":
        return x.tapped
    if tap == "x_refined":
        return seam_tapped(x, context)
    c = context if context != None else ctx.latents(2, d.dim, dtype.bf16)

    def layer(l, b, u):
        if type(u) == "struct":
            return u
        mods = adaln(linear(b.ada, temb), d.dim)
        u = run_block(u, b, (mods, lanes), m, joint, "")[0]
        if tap == "layer{}".format(l):
            return struct(tapped = seam_tapped(u.on(image), u.on(~image)))
        return u

    u = arm.fold_layers(dit.layers, merge([x, c]), layer)
    if type(u) == "struct":
        return u.tapped

    scale = linear(dit.final_ada, ops.elemwise.silu(temb)).on(image)
    u = u.on(image)
    h = ops.elemwise.modulate(ops.elemwise.layernorm_no_scale(u, FINAL_LN_EPS), scale, img_lanes, "scale")
    if tap == "final_norm":
        return seam_tapped(h, context)
    v = linear(dit.final_linear, h)
    if tap == "final_linear":
        return seam_tapped(v, context)
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
    o = attend(q, k, v, g, g, hd, m.sm_scale, g.mask)
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

def decode(arm, m):
    grid = arm.grid()
    z = arm.voxels(0, m.channels, dtype.bf16)
    # z / s + shift, `z + z` owned (as above) and its halving folded into 1 / s.
    scale = f32(0.5 * f32(1.0 / f32(SCALING_FACTOR)))
    z = ops.elemwise.add_bias(m.vae.latent.shift, ops.elemwise.mul_scalar(scale, ops.elemwise.add(z, z)))
    return vae_decode(m.vae, z, grid)

def encode(arm, m):
    mean, grid = vae_encode(m.vae, arm)
    seam.at(seam.PIXELS, [mean, grid])
    return mean
