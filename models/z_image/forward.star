# The forward of Z-Image: the text encoder's last hidden rows, the context
# refiner over the caption, the denoiser over the image and the refined
# context jointly, and the VAE's decode and encode, each its own reading.

load("//lib/diffusion/forward.star", "attend", "chunks", "linear")
load("//lib/flux_vae/forward.star", vae_decode = "decode", vae_encode = "encode")
load("//lib/qwen3_text/forward.star", qwen3_caches = "caches", qwen3_encode = "encode")

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
    z = ops.elemwise.add(z, z)
    z = ops.elemwise.add_bias(m.vae.latent.shift, ops.elemwise.mul_scalar(f32(0.5 * f32(1.0 / f32(SCALING_FACTOR))), z))
    return vae_decode(m.vae, z, grid)

def encode(arm, m):
    mean, grid = vae_encode(m.vae, arm)
    seam.at(seam.PIXELS, [mean, grid])
    return mean
