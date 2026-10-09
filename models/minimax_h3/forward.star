# The forward of MiniMax H3: the text encoder's `text` reading, the caption
# refiner's `refine` reading, and the `denoise` reading, which attends text,
# video, audio and reference rows jointly, each stream modulated by its own
# timestep through its modality's adaLN.

load("//lib/diffusion/forward.star", "attend", "chunks", "linear")
load("//lib/qwen3_text/forward.star", qwen3_caches = "caches", qwen3_encode = "encode")

NORM_EPS = 1e-5
T_MAX_PERIOD = 10000.0
ROPE_THETA = 10000.0

# The modality whose adaLN modulates a stream, and its timestep's slot.
MODALITY = {"video": 0, "reference": 0, "image": 0, "audio": 2, "text": 1, "context": 1}
SLOT = {"video": 0, "text": 0, "context": 0, "reference": 1, "image": 1, "audio": 2}

def caches(m, c):
    if m.te != None:
        qwen3_caches(m.te, c, m.kv)

def forward(m, inputs):
    if m.te != None:
        inputs.reading("text", lambda rows: qwen3_encode(rows, m.te))
    inputs.reading("refine", lambda rows: refine(rows, m))
    return inputs.reading("denoise", lambda rows: denoise(rows, m))

def attention(x, a, m, g, positions):
    """Self-attention over `g`'s rows; with `positions`, its q and k turned."""
    hd = m.dims.head_dim
    inner = m.dims.heads * hd

    def norm(x, gain):
        x = ops.elemwise.rmsnorm_per_head(x, gain, hd, NORM_EPS)
        if positions == None:
            return x
        return ops.elemwise.rope_axes(x, positions, m.rope_dims, [ROPE_THETA] * 4, "neox", m.rotary_dim, hd)

    q, k, v = ops.layout.split_qkv(ops.linear.matmul(x, a.qkv), inner, inner)
    q = norm(q, a.q_norm)
    k = norm(k, a.k_norm)
    o = attend(q, k, v, g, g, hd, m.sm_scale, g.mask)
    return ops.linear.matmul(o, a.out)

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
    positions = arm.axis_positions(0, m.rope_axes)
    joint = struct(perm = arm.row_permutation(), csr = arm.group_indptr(), mask = group_block_diagonal())

    def side(rows, stream):
        t = column(rows.lane_vector(0, m.timestep_slots), SLOT[stream], m.timestep_slots)
        e = ops.elemwise.sinusoid(t, d.t_freq, T_MAX_PERIOD, True, 1.0)
        e = linear(dit.t_out, ops.elemwise.silu(linear(dit.t_in, e)))
        return struct(stream = stream, stemb = ops.elemwise.silu(e))

    sides = [
        side(text, "text"),
        side(video, "video"),
        side(audio, "audio"),
        side(reference, "reference"),
    ]

    ctx = text.latents(3, d.dim, dtype.bf16)
    ctx_perm = text.row_permutation()
    text_rows = ops.layout.unpack_rows(ops.layout.pack_rows(ctx, ctx_perm), ctx_perm)
    video_rows = linear(dit.video_patch, video.latents(0, m.video_features, dtype.bf16))
    audio_rows = linear(dit.audio_patch, audio.latents(2, m.audio_channels, dtype.bf16))
    reference_rows = linear(dit.video_patch, reference.latents(1, m.video_features, dtype.bf16))
    x = merge([text_rows, video_rows, audio_rows, reference_rows])

    def block(l, b, x):
        mods = merge([linear(b.adaln[MODALITY[s.stream]], s.stemb) for s in sides])
        attn_ss, attn_gate, mlp_ss, mlp_gate = chunks(mods, [2 * d.dim, d.dim, 2 * d.dim, d.dim])
        h = ops.elemwise.modulate(ops.elemwise.rmsnorm(x, b.norm1, NORM_EPS), attn_ss, lanes, "scale_shift")
        x = ops.elemwise.gated_residual_add(x, attn_gate, attention(h, b.attn, m, joint, positions), lanes)
        h = ops.elemwise.modulate(ops.elemwise.rmsnorm(x, b.norm2, NORM_EPS), mlp_ss, lanes, "scale_shift")
        return ops.elemwise.gated_residual_add(x, mlp_gate, mlp(h, b.mlp, d), lanes)

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
