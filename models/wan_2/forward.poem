# The forward of Wan 2.2: the text encoder's reading, the denoiser's, and
# the VAE's decode and encode, each in a head (first chunk) and a rest form.

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

NORM_EPS = 1e-6
ROPE_THETA = 10000.0
T_MAX_PERIOD = 10000.0
TE_EPS = 1e-6
TE_MAX_DISTANCE = 128.0
VAE_EPS = 1e-12

def slab(c):
    return [c.front * c.plane, c.c_in]

def decoder_cached(v):
    out = [v.conv_in]
    for r in [v.mid_res0, v.mid_res1]:
        out += [r.conv1, r.conv2]
    for b in v.up:
        for r in b.resnets:
            out += [r.conv1, r.conv2]
        if b.upsampler != None and b.upsampler.time_conv != None:
            out.append(b.upsampler.time_conv)
    out.append(v.conv_out)
    return [c for c in out if c.cache != None]

def encoder_cached(e):
    out = [e.conv_in]
    for b in e.down:
        for r in b.resnets:
            out += [r.conv1, r.conv2]
        if b.downsampler != None and b.downsampler.time_conv != None:
            out.append(b.downsampler.time_conv)
    for r in [e.mid_res0, e.mid_res1]:
        out += [r.conv1, r.conv2]
    out.append(e.conv_out)
    return [c for c in out if c.cache != None]

def caches(m, c):
    if m.vae != None:
        for cv in decoder_cached(m.vae) + encoder_cached(m.vae.enc):
            c.state(cv.cache, slab(cv), dtype.bf16)

def forward(m, inputs):
    if m.te != None:
        inputs.reading("text", lambda text: text_encode(text, m.te))
    velocity = inputs.reading("denoise", lambda rows: denoise(rows, m))
    if m.vae != None:
        inputs.reading("vae.decode.head", lambda rows: vae_decode(rows, m.vae, True))
        inputs.reading("vae.decode", lambda rows: vae_decode(rows, m.vae, False))
        inputs.reading("vae.encode.head", lambda rows: vae_encode(rows, m.vae, True))
        inputs.reading("vae.encode", lambda rows: vae_encode(rows, m.vae, False))
    return velocity

def text_encode(arm, te):
    ids = arm.tokens()
    rows = struct(perm = arm.row_permutation(), csr = arm.lane_indptr())
    y = ops.layout.embed(ids, te.embed, te.vocab)

    def layer(l, w, y):
        table = ops.elemwise.relative_bucket_bias(arm, w.rel_bias, te.max_tokens, te.buckets, TE_MAX_DISTANCE, True)
        x = ops.elemwise.rmsnorm(y, w.attn_norm, TE_EPS)
        q = ops.linear.matmul(x, w.q)
        k = ops.linear.matmul(x, w.k)
        v = ops.linear.matmul(x, w.v)
        o = attend(q, k, v, rows, rows, te.head_dim, 1.0, ops.attn.relative_bias(table, te.max_tokens))
        y = ops.elemwise.residual_add(ops.linear.matmul(o, w.o), y)

        x = ops.elemwise.rmsnorm(y, w.ffn_norm, TE_EPS)
        gate = ops.linear.matmul(x, w.wi_0)
        up = ops.linear.matmul(x, w.wi_1)
        act = ops.linear.mlp_geglu_tanh(gate, up)
        return ops.elemwise.residual_add(ops.linear.matmul(act, w.wo), y)

    y = arm.fold_layers(te.layers, y, layer)
    out = ops.elemwise.rmsnorm(y, te.final_norm, TE_EPS)
    seam.at(seam.HIDDEN, [out])

def block(x, c, b, e, d, vg, cg):
    dim = d.dim
    hd = d.head_dim
    attn_ss, attn_gate, ffn_ss, ffn_gate = chunks(e, [2 * dim, dim, 2 * dim, dim])

    h = norm_modulate(x, attn_ss, vg.lanes, NORM_EPS)
    q, k, v = ops.layout.split_qkv(linear(b.self_attn.qkv, h), dim, dim)

    def turn(x, gain):
        return ops.elemwise.rope_axes(
            ops.elemwise.rmsnorm(x, gain, NORM_EPS),
            vg.positions,
            d.rope_dims,
            [ROPE_THETA] * 4,
            "interleaved",
            hd,
            hd,
        )

    o = ops.attn.ragged(
        ops.layout.pack_rows(turn(q, b.self_attn.norm_q), vg.perm),
        ops.layout.pack_rows(turn(k, b.self_attn.norm_k), vg.perm),
        ops.layout.pack_rows(v, vg.perm),
        vg.csr,
        vg.csr,
        hd,
        d.sm_scale,
        group_block_diagonal(),
    )
    o = ops.layout.unpack_rows(o, vg.perm)
    x = ops.elemwise.gated_residual_add(x, attn_gate, linear(b.self_attn.out, o), vg.lanes)

    hc = ops.elemwise.layernorm(x, b.norm2, b.norm2_bias, NORM_EPS)
    cq = ops.elemwise.rmsnorm(linear(b.cross.q, hc), b.cross.norm_q, NORM_EPS)
    ck, cv = ops.layout.split_rows(linear(b.cross.kv, c), dim)
    ck = ops.elemwise.rmsnorm(ck, b.cross.norm_k, NORM_EPS)
    ca = attend(cq, ck, cv, vg, cg, hd, d.sm_scale, group_block_diagonal())
    x = ops.elemwise.residual_add(linear(b.cross.out, ca), x)

    hf = norm_modulate(x, ffn_ss, vg.lanes, NORM_EPS)
    f = linear(b.ffn.down, ops.elemwise.gelu(linear(b.ffn.up, hf), True))
    return ops.elemwise.gated_residual_add(x, ffn_gate, f, vg.lanes)

def denoise(arm, m):
    d = m.dims
    dit = m.dit
    ctx, vid = arm.on(fact.stream("context")), arm.on(~fact.stream("context"))
    vg = struct(
        lanes = vid.request_of_token(),
        positions = vid.axis_positions(0, m.rope_axes),
        perm = vid.row_permutation(),
        csr = vid.group_indptr(),
    )
    cg = struct(perm = ctx.row_permutation(), csr = ctx.group_indptr())

    t = vid.lane_vector(0, 1)
    h_t = ops.elemwise.silu(linear(
        dit.time_embed.linear_1,
        ops.elemwise.sinusoid(t, d.freq_dim, T_MAX_PERIOD, True, 1.0),
    ))
    temb = linear(dit.time_embed.linear_2, h_t)
    proj = linear(dit.time_proj, ops.elemwise.silu(temb))
    head_mod = ops.elemwise.add_bias(dit.head_table, linear(dit.head_proj, h_t))

    c = ctx.context(0, d.text_dim)
    c = linear(dit.text_embed.linear_2, ops.elemwise.gelu(linear(dit.text_embed.linear_1, c), True))

    x = linear(dit.patch_embed, vid.latents(0, d.patch, dtype.bf16))

    def step(l, b, x):
        e = ops.elemwise.add_bias(b.table, ops.elemwise.copy(proj))
        return block(x, c, b, e, d, vg, cg)

    x = arm.fold_layers(dit.blocks, x, step)
    h = norm_modulate(x, head_mod, vg.lanes, NORM_EPS)
    velocity = linear(dit.proj_out, h)
    seam.at(seam.VELOCITY, [velocity])
    return velocity

def vconv(x, g, c, arm):
    if c.cache != None:
        shape = conv(c.k, [1, 1, 1], [0, c.k[1] // 2, c.k[2] // 2], causal = "zero")
        cache = arm.state(c.cache)
    else:
        shape = conv(c.k, [1, 1, 1], [0, c.k[1] // 2, c.k[2] // 2])
        cache = None
    return ops.spatial.conv3d(x, g, c.w, c.bias, shape, cache)

def norm_silu(x, gain):
    return ops.elemwise.silu(ops.elemwise.rmsnorm(x, gain, VAE_EPS))

def resnet(x, g, r, arm):
    h = norm_silu(x, r.norm1)
    h, _ = vconv(h, g, r.conv1, arm)
    h = norm_silu(h, r.norm2)
    h, _ = vconv(h, g, r.conv2, arm)
    skip = vconv(x, g, r.shortcut, arm)[0] if r.shortcut != None else x
    return ops.elemwise.add(skip, h)

def mid_attention(x, g, a):
    width = a.proj.w.shape[0]
    h = ops.elemwise.rmsnorm(x, a.norm, VAE_EPS)
    q, k, v = ops.layout.split_qkv(linear(a.qkv, h), width, width)
    o = ops.spatial.attention_over(q, k, v, g, segment(frames = 1), f32(1.0 / f32(sqrt(width))))
    return ops.elemwise.add(x, linear(a.proj, o))

def vae_decode(arm, vae, first):
    g0 = arm.grid()
    z = ops.elemwise.copy(arm.voxels(0, vae.z, dtype.bf16))
    z = ops.elemwise.standardize(z, vae.denorm_bias, vae.denorm_scale)
    z, g = vconv(z, g0, vae.post_quant, arm)
    x, _ = vconv(z, g, vae.conv_in, arm)

    x = resnet(x, g, vae.mid_res0, arm)
    x = mid_attention(x, g, vae.mid_attn)
    x = resnet(x, g, vae.mid_res1, arm)

    for up in vae.up:
        x_in, g_in = x, g
        for r in up.resnets:
            x = resnet(x, g, r, arm)
        u = up.upsampler
        if u != None:
            if u.time_conv != None and not first:
                y, gy = vconv(x, g, u.time_conv, arm)
                x, g = ops.spatial.pixel_shuffle(y, gy, [2, 1, 1])
            y, gy = ops.spatial.upsample_nearest(x, g, [1, 2, 2], False)
            x, g = vconv(y, gy, u.resample, arm)
        if up.shortcut == "nearest222":
            factor = [1, 2, 2] if first else [2, 2, 2]
            s = ops.spatial.upsample_nearest(x_in, g_in, factor, False)[0]
            x = ops.elemwise.add(x, s)
        elif up.shortcut == "shuffle_h":
            s, gs = ops.spatial.pixel_shuffle(x_in, g_in, [1, 2, 1])
            s = ops.spatial.upsample_nearest(s, gs, [1, 1, 2], False)[0]
            x = ops.elemwise.add(x, s)

    x = norm_silu(x, vae.norm_out)
    y, gy = vconv(x, g, vae.conv_out, arm)
    pixels, gp = ops.spatial.pixel_shuffle(y, gy, [1, vae.patch, vae.patch])
    pixels = ops.elemwise.clamp(pixels, -1.0, 1.0)
    seam.at(seam.PIXELS, [pixels, gp])
    return pixels

def downsample2d(x, g, c):
    shape = conv([3, 3], [2, 2], [0, 0], pad_back = [0, 1, 1])
    return ops.spatial.conv3d(x, g, c.w, c.bias, shape, None)

def downsample_time(x, g, c, arm):
    shape = conv3d(k = c.k, stride = [2, 1, 1], pad = [c.front, 0, 0], pad_back = [0, 0, 0], causal_t = True)
    return ops.spatial.conv3d(x, g, c.w, c.bias, shape, arm.state(c.cache))

def vae_encode(arm, vae, first):
    e = vae.enc
    g0 = arm.grid()
    px = arm.voxels(1, vae.rgb, dtype.bf16)
    x, g = ops.spatial.pixel_unshuffle(px, g0, [1, vae.patch, vae.patch])
    x, g = vconv(x, g, e.conv_in, arm)

    for b in e.down:
        x_in, g_in = x, g
        for r in b.resnets:
            x = resnet(x, g, r, arm)
        dn = b.downsampler
        if dn != None:
            x, g = downsample2d(x, g, dn.resample)
            tc = dn.time_conv
            if tc != None:
                if first:
                    x = ops.spatial.store_frames(x, g, arm.state(tc.cache), tc.front)
                else:
                    x, g = downsample_time(x, g, tc, arm)
        s, _ = ops.spatial.avg_down(x_in, g_in, b.shortcut.factor, b.shortcut.group)
        x = ops.elemwise.add(x, s)

    x = resnet(x, g, e.mid_res0, arm)
    x = mid_attention(x, g, e.mid_attn)
    x = resnet(x, g, e.mid_res1, arm)

    x = norm_silu(x, e.norm_out)
    h, gh = vconv(x, g, e.conv_out, arm)
    z, gz = vconv(h, gh, e.quant, arm)
    z = ops.elemwise.standardize(z, e.norm_bias, e.norm_scale)
    seam.at(seam.PIXELS, [z, gz])
    return z
