# The forward of FLUX.2: the text encoder's three tapped layers projected
# into the context, the denoiser over text, image and reference rows, and
# the VAE's decode and encode, each its own reading.

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

ROPE_DIMS = [32, 32, 32, 32]
ROPE_THETA = 2000.0
T_MAX_PERIOD = 10000.0
GUIDANCE_SCALE = 1000.0
NORM_EPS = 1e-6

def caches(m, c):
    if m.te != None:
        qwen3_caches(m.te, c, m.kv)

def forward(m, inputs):
    if m.te != None:
        inputs.reading("text", lambda rows: text(rows, m))
    velocity = inputs.reading("denoise", lambda rows: denoise(rows, m))
    if m.vae != None:
        inputs.reading("vae.decode", lambda rows: decode(rows, m))
        inputs.reading("vae.encode", lambda rows: encode(rows, m))
    return velocity

def text(arm, m):
    """The context: the encoder's tapped layers, each projected and summed."""
    total = [None]

    def tap(l, y):
        if l + 1 not in m.te_taps:
            return
        i = m.te_taps.index(l + 1)
        part = ops.linear.matmul(y, m.te_context[i])
        total[0] = part if total[0] == None else ops.elemwise.residual_add(part, total[0])
        if i + 1 == len(m.te_taps):
            seam.at(seam.HIDDEN, [total[0]])

    qwen3_encode(arm, m.te, tap)

def adaln_double(mods, dim):
    a_ss, a_gate, m_ss, m_gate = chunks(mods, [2 * dim, dim, 2 * dim, dim])
    return struct(
        attn = struct(scale_shift = a_ss, gate = a_gate),
        mlp = struct(scale_shift = m_ss, gate = m_gate),
    )

def adaln_single(mods, dim):
    scale_shift, gate = chunks(mods, [2 * dim, dim])
    return struct(scale_shift = scale_shift, gate = gate)

def embed(e, x):
    return ops.linear.matmul(ops.elemwise.silu(ops.linear.matmul(x, e.linear_1)), e.linear_2)

def turn(x, gain, positions, hd):
    return ops.elemwise.rope_axes(
        ops.elemwise.rmsnorm_per_head(x, gain, hd, NORM_EPS),
        positions,
        ROPE_DIMS,
        [ROPE_THETA] * 4,
        "interleaved",
        hd,
        hd,
    )

def denoise(arm, m):
    d, dit = m.dims, m.dit
    dim, hd = d.dim, m.head_dim
    text, image = fact.stream("text"), fact.stream("image")
    txt_in, img_in = arm.on(text), arm.on(~text)

    lanes = arm.request_of_token()
    positions = arm.axis_positions(0, m.rope_axes)
    joint = struct(perm = arm.row_permutation(), csr = arm.group_indptr())

    def attention(q, k, v):
        return attend(q, k, v, joint, joint, hd, m.sm_scale, group_block_diagonal())

    def heads(x, attn, mods, lanes, positions):
        h = norm_modulate(x, mods.scale_shift, lanes, NORM_EPS)
        q, k, v = ops.layout.split_qkv(ops.linear.matmul(h, attn.qkv), dim, dim)
        return (turn(q, attn.q_norm, positions, hd), turn(k, attn.k_norm, positions, hd), v)

    def ff_sublayer(x, ff, mods, lanes):
        h = norm_modulate(x, mods.scale_shift, lanes, NORM_EPS)
        h = ops.linear.mlp_swiglu(ops.linear.matmul(h, ff.linear_in), d.inter)
        return ops.elemwise.gated_residual_add(x, mods.gate, ops.linear.matmul(h, ff.linear_out), lanes)

    t = arm.lane_vector(0, 1)
    temb = embed(dit.t_embed, ops.elemwise.sinusoid(t, m.t_freq_dim, T_MAX_PERIOD, True, 1.0))
    if dit.g_embed != None:
        g = arm.lane_vector(1, 1)
        gemb = embed(dit.g_embed, ops.elemwise.sinusoid(g, m.t_freq_dim, T_MAX_PERIOD, True, GUIDANCE_SCALE))
        temb = ops.elemwise.add(temb, gemb)
    stemb = ops.elemwise.silu(temb)
    mod_txt = ops.linear.matmul(stemb, dit.mod_txt).on(text)
    mod_img = ops.linear.matmul(stemb, dit.mod_img).on(~text)
    mod_txt = adaln_double(mod_txt, dim)
    mod_img = adaln_double(mod_img, dim)
    mod_single = adaln_single(ops.linear.matmul(stemb, dit.mod_single), dim)
    mod_out = ops.linear.matmul(stemb, dit.norm_out)
    lanes_txt, lanes_img = lanes.on(text), lanes.on(~text)
    pos_txt, pos_img = positions.on(text), positions.on(~text)

    if dit.context_embed != None:
        txt = ops.linear.matmul(txt_in.context(0, d.context_in), dit.context_embed)
    else:
        c = txt_in.context(0, dim)
        perm = txt_in.row_permutation()
        txt = ops.layout.unpack_rows(ops.layout.pack_rows(c, perm), perm)
    img = ops.linear.matmul(img_in.latents(0, m.in_channels, dtype.bf16), dit.x_embed)

    def double(l, block, carried):
        txt, img = carried
        tq, tk, tv = heads(txt, block.txt.attn, mod_txt.attn, lanes_txt, pos_txt)
        iq, ik, iv = heads(img, block.img.attn, mod_img.attn, lanes_img, pos_img)
        o = attention(merge([tq, iq]), merge([tk, ik]), merge([tv, iv]))
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
        txt = ff_sublayer(txt, block.txt.ff, mod_txt.mlp, lanes_txt)
        img = ff_sublayer(img, block.img.ff, mod_img.mlp, lanes_img)
        return (txt, img)

    txt, img = arm.fold_layers(dit.double, (txt, img), double)

    def single(l, block, x):
        h = norm_modulate(x, mod_single.scale_shift, lanes, NORM_EPS)
        proj = ops.linear.matmul(h, block.in_proj)
        qkv, mlp = ops.layout.split_rows(proj, 3 * dim)
        q, k, v = ops.layout.split_qkv(qkv, dim, dim)
        o = attention(turn(q, block.q_norm, positions, hd), turn(k, block.k_norm, positions, hd), v)
        a = ops.linear.matmul(o, block.out_attn)
        f = ops.linear.matmul(ops.linear.mlp_swiglu(mlp, d.inter), block.out_mlp)
        return ops.elemwise.gated_residual_add(x, mod_single.gate, ops.elemwise.residual_add(a, f), lanes)

    x = arm.fold_layers(dit.single, merge([txt, img]), single)

    img_all = x.on(~text)
    target, _ = img_all.on(image), img_all.on(~image)
    _, mod_out = mod_out.on(text), mod_out.on(~text)
    mod_out = mod_out.on(image)
    lanes_target = lanes_img.on(image)
    h = norm_modulate(target, mod_out, lanes_target, NORM_EPS)
    velocity = ops.linear.matmul(h, dit.proj_out)
    seam.at(seam.VELOCITY, [velocity])
    return velocity

def decode(arm, m):
    lat = m.vae.latent
    grid = arm.grid()
    z = arm.voxels(0, m.in_channels, dtype.bf16)
    # The standardize below folds in place, and may not write the input:
    # a unit upsample is the decode's fresh copy of it.
    z, grid = ops.spatial.upsample_nearest(z, grid, [1, 1, 1], False)
    z = ops.elemwise.standardize(z, lat.bn_zero, lat.bn_scale)
    z = ops.elemwise.add_bias(lat.bn_mean, z)
    z, grid = ops.spatial.pixel_shuffle(z, grid, [1, m.pack, m.pack])
    z = convolve(z, grid, lat.post_quant_conv)
    return vae_decode(m.vae, z, grid)

def encode(arm, m):
    lat = m.vae.latent
    h, grid = vae_encode(m.vae, arm)
    mean = convolve(h, grid, lat.quant_conv)
    packed, grid = ops.spatial.pixel_unshuffle(mean, grid, [1, m.pack, m.pack])
    z = ops.elemwise.standardize(packed, lat.bn_mean, lat.bn_rscale)
    seam.at(seam.PIXELS, [z, grid])
    return z
