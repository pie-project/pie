# The forward of MiniMax H3: the text encoder's `text` reading, the caption
# refiner's `refine` reading, and the `denoise` reading, which attends text,
# video, audio and reference rows jointly, each stream modulated by its own
# timestep through its modality's adaLN.

NORM_EPS = 1e-5
T_MAX_PERIOD = 10000.0
ROPE_THETA = 10000.0
ROPE_AXES = 3
ADALN_SLICES = 6
TIMESTEP_SLOTS = 4
VIDEO_FEATURES = 96
AUDIO_CHANNELS = 32

# The modality whose adaLN modulates a stream, and its timestep's slot.
MODALITY = {"video": 0, "reference": 0, "image": 0, "audio": 2, "text": 1, "context": 1}
SLOT = {"video": 0, "text": 0, "context": 0, "reference": 1, "image": 1, "audio": 2}

def caches(m, c):
    if m.te == None:
        return
    kv = c.kv_space(m.kv)
    plane = m.te.kv_heads * m.te.head_dim
    for layer in m.te.layers:
        c.kv(kv, layer.kv, [plane, plane], m.te.head_dim, heads = True)

def forward(m, inputs):
    if m.te != None:
        inputs.reading("text", lambda text: text_encode(text, m.te))
    inputs.reading("refine", lambda rows: refine(rows, m))
    return inputs.reading("denoise", lambda rows: denoise(rows, m))

def text_encode(arm, te):
    plan = ops.attn.plan_prefill(arm, te.q_heads, te.kv_heads, te.head_dim, None)
    ids = arm.tokens()
    positions = arm.positions()
    y = ops.layout.embed(ids, te.embed, te.vocab)
    last = len(te.layers) - 1

    def block(l, w, y):
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
        o = ops.linear.matmul(o, w.o)
        y = ops.elemwise.residual_add(o, y)

        x = ops.elemwise.rmsnorm(y, w.mlp_norm, te.eps)
        f = ops.linear.matmul(ops.linear.mlp_swiglu(ops.linear.matmul(x, w.gate_up), te.inter), w.down)
        y = ops.elemwise.residual_add(f, y)
        if l == last:
            seam.at(seam.HIDDEN, [y])
        return y

    arm.fold_layers(te.layers, y, block)

def linear(w, x):
    return ops.elemwise.add_bias(w.bias, ops.linear.matmul(x, w.w))

def turn(x, gain, positions, m):
    d = m.dims
    return ops.elemwise.rope_axes(
        ops.elemwise.rmsnorm_per_head(x, gain, d.head_dim, NORM_EPS),
        positions,
        m.rope_dims,
        [ROPE_THETA] * 4,
        "neox",
        m.rotary_dim,
        d.head_dim,
    )

def attention(x, a, m, g, positions):
    d = m.dims
    inner = d.heads * d.head_dim
    q, k, v = ops.layout.split_qkv(ops.linear.matmul(x, a.qkv), inner, inner)
    if positions != None:
        q, k = turn(q, a.q_norm, positions, m), turn(k, a.k_norm, positions, m)
    else:
        q = ops.elemwise.rmsnorm_per_head(q, a.q_norm, d.head_dim, NORM_EPS)
        k = ops.elemwise.rmsnorm_per_head(k, a.k_norm, d.head_dim, NORM_EPS)
    o = ops.attn.ragged(
        ops.layout.pack_rows(q, g.perm),
        ops.layout.pack_rows(k, g.perm),
        ops.layout.pack_rows(v, g.perm),
        g.csr,
        g.csr,
        d.head_dim,
        m.sm_scale,
        g.mask,
    )
    return ops.linear.matmul(ops.layout.unpack_rows(o, g.perm), a.out)

def mlp(x, f, d):
    return ops.linear.matmul(ops.linear.mlp_swiglu(ops.linear.matmul(x, f.fc1), d.inter), f.fc2)

def refine(arm, m):
    d = m.dims
    x = linear(m.dit.condition, arm.context(0, d.text_dim))
    geom = struct(perm = arm.row_permutation(), csr = arm.lane_indptr(), mask = unmasked())
    last = len(m.dit.refine) - 1

    def block(l, b, x):
        h = ops.elemwise.rmsnorm(x, b.norm1, NORM_EPS)
        x = ops.elemwise.residual_add(attention(h, b.attn, m, geom, None), x)
        h = ops.elemwise.rmsnorm(x, b.norm2, NORM_EPS)
        x = ops.elemwise.residual_add(mlp(h, b.mlp, d), x)
        if l == last:
            y = ops.elemwise.rmsnorm(x, m.dit.refine_norm, NORM_EPS)
            seam.at(seam.HIDDEN, [y])
        return x

    arm.fold_layers(m.dit.refine, x, block)

def adaln6(mod, dim):
    attn, rest = ops.layout.split_rows(mod, 2 * dim)
    attn_gate, rest = ops.layout.split_rows(rest, dim)
    mlp_, mlp_gate = ops.layout.split_rows(rest, 2 * dim)
    return struct(attn = attn, attn_gate = attn_gate, mlp = mlp_, mlp_gate = mlp_gate)

def column(v, slot, total):
    tail = v if slot == 0 else ops.layout.split_rows(v, slot)[1]
    if slot + 1 == total:
        return tail
    return ops.layout.split_rows(tail, 1)[0]

def denoise(arm, m):
    d = m.dims
    dit = m.dit
    text, rest = arm.on(fact.stream("text")), arm.on(~fact.stream("text"))
    video, rest = rest.on(fact.stream("video")), rest.on(~fact.stream("video"))
    audio, reference = rest.on(fact.stream("audio")), rest.on(~fact.stream("audio"))

    lanes = arm.request_of_token()
    positions = arm.axis_positions(0, ROPE_AXES)
    joint = struct(perm = arm.row_permutation(), csr = arm.group_indptr(), mask = group_block_diagonal())

    def side(rows, stream):
        t = column(rows.lane_vector(0, TIMESTEP_SLOTS), SLOT[stream], TIMESTEP_SLOTS)
        e = ops.elemwise.sinusoid(t, d.t_freq, T_MAX_PERIOD, True, 1.0)
        e = linear(dit.t_out, ops.elemwise.silu(linear(dit.t_in, e)))
        return struct(arm = rows, stream = stream, stemb = ops.elemwise.silu(e))

    sides = [
        side(text, "text"),
        side(video, "video"),
        side(audio, "audio"),
        side(reference, "reference"),
    ]

    ctx = sides[0].arm.latents(3, d.dim, dtype.bf16)
    ctx_perm = sides[0].arm.row_permutation()
    text_rows = ops.layout.unpack_rows(ops.layout.pack_rows(ctx, ctx_perm), ctx_perm)
    video_rows = linear(dit.video_patch, sides[1].arm.latents(0, VIDEO_FEATURES, dtype.bf16))
    audio_rows = linear(dit.audio_patch, sides[2].arm.latents(2, AUDIO_CHANNELS, dtype.bf16))
    reference_rows = linear(dit.video_patch, sides[3].arm.latents(1, VIDEO_FEATURES, dtype.bf16))
    x = merge([text_rows, video_rows, audio_rows, reference_rows])

    def block(l, b, x):
        mods = adaln6(merge([linear(b.adaln[MODALITY[s.stream]], s.stemb) for s in sides]), d.dim)
        h = ops.elemwise.modulate(ops.elemwise.rmsnorm(x, b.norm1, NORM_EPS), mods.attn, lanes, "scale_shift")
        x = ops.elemwise.gated_residual_add(x, mods.attn_gate, attention(h, b.attn, m, joint, positions), lanes)
        h = ops.elemwise.modulate(ops.elemwise.rmsnorm(x, b.norm2, NORM_EPS), mods.mlp, lanes, "scale_shift")
        return ops.elemwise.gated_residual_add(x, mods.mlp_gate, mlp(h, b.mlp, d), lanes)

    x = arm.fold_layers(dit.blocks, x, block)

    final_mod = merge([linear(dit.final_adaln, s.stemb) for s in sides])
    h = ops.elemwise.modulate(ops.elemwise.rmsnorm(x, dit.final_norm, NORM_EPS), final_mod, lanes, "scale_shift")
    rest = h.on(~fact.stream("text"))
    h_video, rest = rest.on(fact.stream("video")), rest.on(~fact.stream("video"))
    h_audio = rest.on(fact.stream("audio"))
    velocity = linear(dit.video_out, h_video)
    seam.at(seam.VELOCITY, [velocity])
    audio_velocity = linear(dit.audio_out, h_audio)
    seam.at(seam.HIDDEN, [audio_velocity])
    return velocity
