# The forward of LTX-2.5: the denoiser over paired video and audio, the text
# connectors' refining of the prompt for each stream, and the VAE decode.

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

T_MAX_PERIOD = 10000.0
NORM_EPS = 1e-6
ROPE_THETA = 10000.0
GATE_SCALE = 2.0
VAE_EPS = 1e-8

def caches(m, c):
    pass

def forward(m, inputs):
    velocity = inputs.reading("denoise", lambda rows: denoise(rows, m))
    text_in = m.dims.text_in
    video, audio = m.connectors
    inputs.reading("refine.video", lambda rows: refine(rows, video, text_in))
    inputs.reading("refine.audio", lambda rows: refine(rows, audio, text_in))
    if m.vae != None:
        inputs.reading("vae.decode", lambda rows: vae_decode(rows, m.vae))
    return velocity

def rms(x):
    return ops.elemwise.rmsnorm_no_scale(x, x.width(), NORM_EPS)

def modulate(x, scale_shift, lanes):
    return ops.elemwise.modulate(x, scale_shift, lanes, "scale_shift")

def norm_modulate(x, scale_shift, lanes):
    return modulate(rms(x), scale_shift, lanes)

def adaln_block(e, dim):
    msa_ss, msa_gate, mlp_ss, mlp_gate, q_ss, q_gate = chunks(e, [2 * dim, dim, 2 * dim, dim, 2 * dim, dim])
    return struct(msa_ss = msa_ss, msa_gate = msa_gate, mlp_ss = mlp_ss, mlp_gate = mlp_gate, q_ss = q_ss, q_gate = q_gate)

def adaln_av(e, dim):
    a2v_ss, v2a_ss = chunks(e, [2 * dim, 2 * dim])
    return struct(a2v_ss = a2v_ss, v2a_ss = v2a_ss)

def adaln(head, sin):
    h = ops.elemwise.silu(linear(head.embed.linear_1, sin))
    emb = linear(head.embed.linear_2, h)
    return (linear(head.proj, ops.elemwise.silu(emb)), h)

def sinusoid(t, m):
    return ops.elemwise.sinusoid(t, m.t_freq_dim, T_MAX_PERIOD, True, 1.0)

def stream_mods(heads, t, m):
    sin = sinusoid(t, m)
    proj9, hidden = adaln(heads.adaln, sin)
    av_ss, _ = adaln(heads.av_ss, sin)
    av_gate, _ = adaln(heads.av_gate, sin)
    head = ops.elemwise.add_bias(heads.head_table, linear(heads.head_proj, hidden))
    return struct(proj9 = proj9, av_ss = av_ss, av_gate = av_gate, head = head)

def qk_norm(x, gain):
    return ops.elemwise.rmsnorm(x, gain, NORM_EPS)

def turn(x, positions, dims, head_dim):
    return ops.elemwise.rope_axes(x, positions, dims, [ROPE_THETA] * 4, "split_ladder", head_dim, head_dim)

def gate_out(o, h, a):
    logits = linear(a.gate, h)
    gated = ops.elemwise.gate_sigmoid_mul_heads(o, logits, a.head_dim, GATE_SCALE)
    return linear(a.out, gated)

def self_attention(h, a, dims, g):
    q, k, v = ops.layout.split_qkv(linear(a.qkv, h), a.inner, a.inner)
    q = turn(qk_norm(q, a.q_norm), g.positions, dims, a.head_dim)
    k = turn(qk_norm(k, a.k_norm), g.positions, dims, a.head_dim)
    o = attend(q, k, v, g, g, a.head_dim, a.sm_scale, group_block_diagonal())
    return gate_out(o, h, a)

def cross_attention(h, ctx, a, rope, qg, kg):
    q = qk_norm(linear(a.qkv, h), a.q_norm)
    k, v = ops.layout.split_rows(linear(a.kv, ctx), a.inner)
    k = qk_norm(k, a.k_norm)
    if rope != None:
        q_pos, k_pos, dims = rope
        q = turn(q, q_pos, dims, a.head_dim)
        k = turn(k, k_pos, dims, a.head_dim)
    o = attend(q, k, v, qg, kg, a.head_dim, a.sm_scale, group_block_diagonal())
    return gate_out(o, h, a)

def ff_sublayer(x, ff, ss, gate, lanes):
    h = norm_modulate(x, ss, lanes)
    f = linear(ff.down, ops.elemwise.gelu(linear(ff.up, h), True))
    return ops.elemwise.gated_residual_add(x, gate, f, lanes)

def table_add(table, v):
    return ops.elemwise.add_bias(table, ops.elemwise.copy(v))

def block_mods(side, s, prompt, dim):
    m = adaln_block(table_add(side.table, s.proj9), dim)
    av = adaln_av(table_add(side.av_ss_table, s.av_ss), dim)
    av_gate = table_add(side.av_gate_table, s.av_gate)
    prompt_ss = table_add(side.prompt_table, prompt)
    return struct(m = m, av = av, av_gate = av_gate, prompt_ss = prompt_ss)

def block(xv, xa, b, d, mv, ma, ctx, actx, vg, ag, v_time, cg, acg):
    hv = norm_modulate(xv, mv.m.msa_ss, vg.lanes)
    ov = self_attention(hv, b.video.self_attn, d.rope_dims, vg)
    xv = ops.elemwise.gated_residual_add(xv, mv.m.msa_gate, ov, vg.lanes)

    ha = norm_modulate(xa, ma.m.msa_ss, ag.lanes)
    oa = self_attention(ha, b.audio.self_attn, d.audio_rope_dims, ag)
    xa = ops.elemwise.gated_residual_add(xa, ma.m.msa_gate, oa, ag.lanes)

    hv = norm_modulate(xv, mv.m.q_ss, vg.lanes)
    c = modulate(ctx, mv.prompt_ss, cg.lanes)
    ov = cross_attention(hv, c, b.video.cross, None, vg, cg)
    xv = ops.elemwise.gated_residual_add(xv, mv.m.q_gate, ov, vg.lanes)

    ha = norm_modulate(xa, ma.m.q_ss, ag.lanes)
    ac = modulate(actx, ma.prompt_ss, acg.lanes)
    oa = cross_attention(ha, ac, b.audio.cross, None, ag, acg)
    xa = ops.elemwise.gated_residual_add(xa, ma.m.q_gate, oa, ag.lanes)

    nv = rms(xv)
    na = rms(xa)
    av_dims = d.audio_rope_dims

    q_in = modulate(nv, mv.av.a2v_ss, vg.lanes)
    kv_in = modulate(na, ma.av.a2v_ss, ag.lanes)
    o = cross_attention(q_in, kv_in, b.a2v, (v_time, ag.positions, av_dims), vg, ag)
    xv = ops.elemwise.gated_residual_add(xv, mv.av_gate, o, vg.lanes)

    q_in = modulate(na, ma.av.v2a_ss, ag.lanes)
    kv_in = modulate(nv, mv.av.v2a_ss, vg.lanes)
    o = cross_attention(q_in, kv_in, b.v2a, (ag.positions, v_time, av_dims), ag, vg)
    xa = ops.elemwise.gated_residual_add(xa, ma.av_gate, o, ag.lanes)

    xv = ff_sublayer(xv, b.video.ffn, mv.m.mlp_ss, mv.m.mlp_gate, vg.lanes)
    xa = ff_sublayer(xa, b.audio.ffn, ma.m.mlp_ss, ma.m.mlp_gate, ag.lanes)
    return (xv, xa)

def denoise(arm, m):
    d = m.dims
    dit = m.dit
    ctx, rest = arm.on(fact.stream("context")), arm.on(~fact.stream("context"))
    actx, media = rest.on(fact.stream("reference")), rest.on(~fact.stream("reference"))
    vid, aud = media.on(fact.stream("video")), media.on(~fact.stream("video"))

    vg = struct(
        lanes = vid.request_of_token(),
        positions = vid.axis_positions(0, m.rope_axes),
        perm = vid.row_permutation(),
        csr = vid.group_indptr(),
    )
    ag = struct(
        lanes = aud.request_of_token(),
        positions = aud.axis_positions(1, 1),
        perm = aud.row_permutation(),
        csr = aud.group_indptr(),
    )
    cg = struct(lanes = ctx.request_of_token(), perm = ctx.row_permutation(), csr = ctx.group_indptr())
    acg = struct(lanes = actx.request_of_token(), perm = actx.row_permutation(), csr = actx.group_indptr())
    v_time, _ = ops.layout.split_rows(vg.positions, 1)

    vt = vid.lane_vector(0, 1)
    at = aud.lane_vector(0, 1)
    mods_v = stream_mods(dit.video, vt, m)
    mods_a = stream_mods(dit.audio, at, m)
    prompt_v, _ = adaln(dit.prompt, sinusoid(ctx.lane_vector(0, 1), m))
    prompt_a, _ = adaln(dit.audio_prompt, sinusoid(actx.lane_vector(0, 1), m))

    xv = linear(dit.video.patchify, vid.latents(0, d.channels, dtype.bf16))
    xa = linear(dit.audio.patchify, aud.latents(0, d.channels, dtype.bf16))
    ctx_rows = ctx.context(0, d.cross_dim)
    actx_rows = actx.context(1, d.audio_cross_dim)

    def step(l, b, carried):
        xv, xa = carried
        mv = block_mods(b.video, mods_v, prompt_v, d.dim)
        ma = block_mods(b.audio, mods_a, prompt_a, d.audio_dim)
        return block(xv, xa, b, d, mv, ma, ctx_rows, actx_rows, vg, ag, v_time, cg, acg)

    xv, xa = arm.fold_layers(dit.blocks, (xv, xa), step)

    def head(x, s, mods, g):
        h = ops.elemwise.modulate(ops.elemwise.layernorm_no_scale(x, NORM_EPS), mods.head, g.lanes, "scale_shift")
        return linear(s.proj_out, h)

    vv = head(xv, dit.video, mods_v, vg)
    va = head(xa, dit.audio, mods_a, ag)
    velocity = merge([vv, va])
    seam.at(seam.VELOCITY, [velocity])
    return velocity

def refine(arm, conn, text_in):
    g = struct(
        lanes = arm.request_of_token(),
        positions = arm.axis_positions(1, 1),
        perm = arm.row_permutation(),
        csr = arm.lane_indptr(),
    )
    x = arm.latents(1, text_in, dtype.bf16)
    h = ops.elemwise.mul_scalar(conn.rescale, ops.linear.matmul(x, conn.aggregate.w))
    h = ops.elemwise.add_bias(conn.aggregate.bias, h)

    def step(l, b, h):
        n = rms(h)
        o = self_attention(n, b.attn, conn.rope_dims, g)
        h = ops.elemwise.residual_add(o, h)
        n = rms(h)
        f = linear(b.ffn.down, ops.elemwise.gelu(linear(b.ffn.up, n), True))
        return ops.elemwise.residual_add(f, h)

    h = arm.fold_layers(conn.blocks, h, step)
    out = rms(h)
    seam.at(seam.HIDDEN, [out])

def vconv(x, g, c):
    shape = conv3d(k = [3, 3, 3], stride = [1, 1, 1], pad = [1, 1, 1], pad_back = [1, 1, 1], time_pad = "replicate")
    return ops.spatial.conv3d(x, g, c.w, c.bias, shape, None)

def norm_silu(x):
    return ops.elemwise.silu(ops.elemwise.rmsnorm_no_scale(x, x.width(), VAE_EPS))

def resnet(x, g, r):
    h, _ = vconv(norm_silu(x), g, r.conv1)
    h, _ = vconv(norm_silu(h), g, r.conv2)
    return ops.elemwise.add(x, h)

def vae_decode(arm, vae):
    g = arm.grid()
    z = ops.elemwise.copy(arm.voxels(0, vae.z, dtype.bf16))
    z = ops.elemwise.standardize(z, vae.zero, vae.latents_std)
    z = ops.elemwise.add_bias(vae.latents_mean, z)

    x, _ = vconv(z, g, vae.conv_in)
    for r in vae.mid:
        x = resnet(x, g, r)
    for up in vae.up:
        y, gy = vconv(x, g, up.upsampler)
        x, g = ops.spatial.pixel_shuffle_trimming(y, gy, up.stride, up.stride[0] - 1)
        for r in up.resnets:
            x = resnet(x, g, r)

    x = norm_silu(x)
    y, gy = vconv(x, g, vae.conv_out)
    pixels, gp = ops.spatial.pixel_shuffle(y, gy, [1, vae.patch, vae.patch])
    seam.at(seam.PIXELS, [pixels, gp])
    return pixels
