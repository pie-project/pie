# The forward of HunyuanImage 3. One trunk carries the text rows it encodes
# and the canvas rows it denoises; the `image.in` and `image.out` readings
# run its U-Net ends over a latent clip and the trunk's rows.

NORM_EPS = 1e-5
ROPE_THETA = 10000.0
ROPE_AXES = 2
T_FREQ_DIM = 256
T_MAX_PERIOD = 10000.0
GN_GROUPS = 32
GN_EPS = 1e-5
LATENT_CHANNELS = 32

def caches(m, c):
    kv = c.kv_space(m.kv)
    plane = m.kv_width
    for w in m.layers:
        c.kv(kv, w.kv, [plane, plane], m.dims.head_dim, heads = True)

def on_canvas():
    """The canvas rows: the denoise reading's, with the mask the canvas
    attends through. A denoise lane sent without one reads as text."""
    return fact.reading("denoise") & fact.has(fact.Mask)

def forward(m, inputs):
    inputs.reading("image.in", lambda rows: image_in(rows, m))
    inputs.reading("image.out", lambda rows: image_out(rows, m))
    trunk_rows = inputs.on(~fact.reading("image.in") & ~fact.reading("image.out"))
    canvas_rows = trunk_rows.on(on_canvas())
    return trunk(trunk_rows, canvas_rows, m)

def linear(w, x):
    return ops.elemwise.add_bias(w.bias, ops.linear.matmul(x, w.w))

def embedder(e, t_freq):
    return linear(e.mlp_out, ops.elemwise.gelu(linear(e.mlp_in, t_freq), True))

def trunk(all, den, m):
    d = m.dims
    hd = d.head_dim
    sm = m.sm_scale
    classes = [on_canvas(), fact.single_token()]
    ([canvas_in, ar_decode], ar_prefill) = all.partition(classes)

    plan_den = ops.attn.plan_prefill(canvas_in, d.q_heads, d.kv_heads, hd, None)
    plan_dec = ops.attn.plan_decode(ar_decode, d.q_heads, d.kv_heads, hd, None)
    plan_pre = ops.attn.plan_prefill(ar_prefill, d.q_heads, d.kv_heads, hd, None)
    mask = canvas_in.mask()

    positions = all.axis_positions(0, ROPE_AXES)
    y = ops.layout.embed(all.tokens(), m.embed, d.vocab)
    encoded = y.on(~on_canvas())
    y = merge([canvas_rows(den, m), encoded])
    rope_dims = [hd // 2, hd // 2, 0, 0]

    def block(l, w, y):
        n = ops.elemwise.rmsnorm(y, w.attn_norm, NORM_EPS)
        q, k, v = ops.layout.split_qkv(ops.linear.matmul(n, w.qkv), m.q_width, m.kv_width)
        turn = lambda x: ops.elemwise.rope_axes(x, positions, rope_dims, [ROPE_THETA] * 4, "split", hd, hd)
        q = ops.elemwise.rmsnorm_per_head(turn(q), w.q_norm, hd, NORM_EPS)
        k = ops.elemwise.rmsnorm_per_head(turn(k), w.k_norm, hd, NORM_EPS)

        pages = all.kv(w.kv)
        ops.attn.kv_append(k, v, pages, all.write_page(w.kv), all.write_offset(w.kv))

        ([dq, aq], pq) = q.partition(classes)
        a = merge([
            ops.attn.masked(dq, plan_den, mask, pages, None, hd, d.kv_heads, False, sm),
            ops.attn.decode(aq, plan_dec, pages, None, hd, sm),
            ops.attn.prefill(pq, plan_pre, pages, None, hd, d.kv_heads, sm),
        ])
        o = ops.linear.matmul(a, w.o_proj)
        if l == 0:
            y = ops.elemwise.add(o, y)
        else:
            y = ops.elemwise.residual_add(o, y)

        n = ops.elemwise.rmsnorm(y, w.mlp_norm, NORM_EPS)
        return ops.elemwise.residual_add(moe(n, w, m), y)

    y = all.fold_layers(m.layers, y, block)

    canvas, text = y.on(on_canvas()), y.on(~on_canvas())
    seam.at(seam.HIDDEN, [canvas])
    text_in = all.on(~on_canvas())
    x = ops.elemwise.rmsnorm(text, m.final_norm, NORM_EPS)
    x = ops.layout.gather_rows(x, text_in.readout_rows())
    logits = ops.linear.lm_head(x, m.head)
    seam.at(seam.OUT, [logits])
    return canvas

def moe(x, w, m):
    d = m.dims
    routes, weights = ops.linear.moe_topk_softmax(ops.linear.matmul(x, w.router), d.experts, d.top_k)

    def select(act, bank):
        if bank.dtype in [dtype.bf16, dtype.f16, dtype.f32]:
            return ops.linear.moe_matmul_select(act, bank, routes, d.top_k)
        return ops.linear.moe_matmul_select_quant(act, bank, routes, d.top_k)

    act = ops.linear.mlp_swiglu(select(x, w.experts_gate_up), d.moe_inter)
    routed = ops.linear.moe_weighted_sum(select(act, w.experts_down), weights)
    shared = ops.linear.matmul(
        ops.linear.mlp_swiglu(ops.linear.matmul(x, w.shared_gate_up), d.shared_inter),
        w.shared_down,
    )
    return ops.elemwise.residual_add(shared, routed)

def canvas_rows(arm, m):
    d = m.dims
    u = arm.latents(0, d.hidden, dtype.bf16)
    flag = arm.latents(1, 1, dtype.bf16)
    lanes = arm.request_of_token()

    t = arm.lane_vector(0, 1)
    freqs = ops.elemwise.sinusoid(t, T_FREQ_DIM, T_MAX_PERIOD, True, 1.0)
    doubled = embedder(m.timestep_emb, freqs)

    spread = ops.linear.matmul(flag, m.ones)
    pos, neg = ops.layout.split_rows(spread, d.hidden)
    zero = ops.elemwise.add(pos, neg)
    special = ops.elemwise.modulate(zero, doubled, lanes, "scale_shift")

    kept = ops.elemwise.modulate(u, neg, None, "scale")
    return ops.elemwise.add(kept, ops.elemwise.mul(special, pos))

def conv_(x, g, c):
    shape = conv(c.k, [1, 1, 1], [0, c.k[1] // 2, c.k[2] // 2])
    return ops.spatial.conv3d(x, g, c.w, c.bias, shape, None)

def resblock(x, g, r, temb):
    h = ops.spatial.group_norm(x, g, GN_GROUPS, r.norm_in.weight, r.norm_in.bias, GN_EPS, True)
    h, g1 = conv_(h, g, r.conv_in)
    mod = linear(r.emb, ops.elemwise.silu(temb))
    h = ops.spatial.group_norm(h, g1, GN_GROUPS, r.norm_out.weight, r.norm_out.bias, GN_EPS, False)
    h = ops.elemwise.modulate(h, mod, None, "scale_shift")
    h = ops.elemwise.silu(h)
    h, _ = conv_(h, g1, r.conv_out)
    skip = conv_(x, g, r.skip)[0] if r.skip != None else x
    return ops.elemwise.add(skip, h)

def image_in(arm, m):
    g = arm.grid()
    clip = arm.voxels(0, LATENT_CHANNELS + T_FREQ_DIM, dtype.bf16)
    z, freqs = ops.layout.split_rows(clip, LATENT_CHANNELS)
    temb = embedder(m.time_embed, freqs)
    h, g1 = conv_(z, g, m.patch_embed.conv_in)
    x = resblock(h, g1, m.patch_embed.res, temb)
    seam.at(seam.PIXELS, [x, g1])

def image_out(arm, m):
    g = arm.grid()
    clip = arm.voxels(1, m.dims.hidden + T_FREQ_DIM, dtype.bf16)
    rows, freqs = ops.layout.split_rows(clip, m.dims.hidden)
    temb = embedder(m.time_embed_2, freqs)
    x = resblock(rows, g, m.final_layer.res, temb)
    h = ops.spatial.group_norm(x, g, GN_GROUPS, m.final_layer.norm_out.weight, m.final_layer.norm_out.bias, GN_EPS, True)
    v, gv = conv_(h, g, m.final_layer.conv_out)
    seam.at(seam.PIXELS, [v, gv])
