# How a Z-Image checkpoint is laid out: a diffusers pipeline, its components
# under `dit.`, `te.` and `vae.`; or, for a model of the transformer alone,
# a bare transformer state_dict.

def biased(reads, w, stem):
    """The projection `w` from `stem.weight`, and its bias (if any) from
    `stem.bias`."""
    reads.read(w.w, stem + ".weight")
    if w.bias != None:
        reads.read(w.bias, stem + ".bias")

def conv(reads, c, stem):
    """The convolution `c` from `stem`, its kernel transmuted to the taps-major
    plane the layout holds."""
    name = stem + ".weight"
    reads.read_expr(c.w, src(name).transmute(c.w.shape, stored(name)))
    reads.read(c.bias, stem + ".bias")

def conv_head(reads, c, stem, rows_stored):
    """The convolution `c`: the first of the `rows_stored` output channels
    the checkpoint holds at `stem`."""
    kernel = stem + ".weight"
    natural = list(c.w.shape)
    natural[0] = rows_stored
    rows = c.c_out
    reads.read_expr(c.w, src(kernel).transmute(natural, stored(kernel)).slice(0, 0, rows))
    bias = stem + ".bias"
    held = stored(bias)
    want = encoding(c.bias.dtype)
    head = c.bias.name + ".head"
    reads.push(tensor(head, src(bias).slice(0, 0, rows), held, shape = c.bias.shape, internal = True))
    reads.push(tensor(c.bias.name, out(head) if want == held else out(head).cast(want), want, shape = c.bias.shape))

def group_norm(reads, n, stem):
    reads.read(n.weight, stem + ".weight")
    reads.read(n.bias, stem + ".bias")

def resnet(reads, r, stem):
    group_norm(reads, r.norm1, stem + ".norm1")
    conv(reads, r.conv1, stem + ".conv1")
    group_norm(reads, r.norm2, stem + ".norm2")
    conv(reads, r.conv2, stem + ".conv2")
    if r.shortcut != None:
        conv(reads, r.shortcut, stem + ".conv_shortcut")

def mid(reads, m, stem):
    resnet(reads, m.res0, stem + ".resnets.0")
    a = m.attn
    n = lambda s: stem + ".attentions.0." + s
    group_norm(reads, a.norm, n("group_norm"))
    biased(reads, a.q, n("to_q"))
    biased(reads, a.k, n("to_k"))
    biased(reads, a.v, n("to_v"))
    biased(reads, a.out, n("to_out.0"))
    resnet(reads, m.res1, stem + ".resnets.1")

def vae_read(reads, v, encoder_out_stored = None):
    """The decoder, then the encoder; with `encoder_out_stored`, the
    encoder's output convolution is the head of the one stored."""
    at = lambda tail: "vae." + tail
    d = v.decoder
    conv(reads, d.conv_in, at("decoder.conv_in"))
    mid(reads, d.mid, at("decoder.mid_block"))
    for i, up in enumerate(d.up):
        for r, block in enumerate(up.resnets):
            resnet(reads, block, at("decoder.up_blocks.{}.resnets.{}".format(i, r)))
        if up.upsample != None:
            conv(reads, up.upsample, at("decoder.up_blocks.{}.upsamplers.0.conv".format(i)))
    group_norm(reads, d.norm_out, at("decoder.conv_norm_out"))
    conv(reads, d.conv_out, at("decoder.conv_out"))

    e = v.encoder
    conv(reads, e.conv_in, at("encoder.conv_in"))
    for i, down in enumerate(e.down):
        for r, block in enumerate(down.resnets):
            resnet(reads, block, at("encoder.down_blocks.{}.resnets.{}".format(i, r)))
        if down.downsample != None:
            conv(reads, down.downsample, at("encoder.down_blocks.{}.downsamplers.0.conv".format(i)))
    mid(reads, e.mid, at("encoder.mid_block"))
    group_norm(reads, e.norm_out, at("encoder.conv_norm_out"))
    if encoder_out_stored == None:
        conv(reads, e.conv_out, at("encoder.conv_out"))
    else:
        conv_head(reads, e.conv_out, at("encoder.conv_out"), encoder_out_stored)

def te_read(reads, te, prefix):
    reads.read(te.embed, prefix + "embed_tokens.weight")
    for l, w in enumerate(te.layers):
        n = lambda s: "{}layers.{}.{}".format(prefix, l, s)
        reads.read(w.attn_norm, n("input_layernorm.weight"))
        reads.read(w.q, n("self_attn.q_proj.weight"))
        reads.read(w.k, n("self_attn.k_proj.weight"))
        reads.read(w.v, n("self_attn.v_proj.weight"))
        reads.read(w.o, n("self_attn.o_proj.weight"))
        reads.read(w.q_norm, n("self_attn.q_norm.weight"))
        reads.read(w.k_norm, n("self_attn.k_norm.weight"))
        reads.read(w.mlp_norm, n("post_attention_layernorm.weight"))
        reads.read_concat(w.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
        reads.read(w.down, n("mlp.down_proj.weight"))

def product(xs):
    out = 1
    for x in xs:
        out *= x
    return out

DIFFUSERS = "a diffusers pipeline (`dit.`/`te.` prefixes)"
BARE = "a bare transformer state_dict"
T_FLIP = 1000.0
SHIFT_FACTOR = 0.1159

def formats(m):
    pipeline = lambda checkpoint: has_prefix("dit.")
    out = [format(DIFFUSERS, recognizes = pipeline, read = lambda reads: read(m, reads, "dit."))]
    if m.te == None:
        out.append(format(
            BARE,
            recognizes = lambda checkpoint: not pipeline(checkpoint),
            read = lambda reads: read(m, reads, ""),
        ))
    return out

def read(m, reads, prefix):
    dit(reads, m.dit, prefix)
    if m.te != None:
        te_read(reads, m.te, "te.model.")
    if m.vae != None:
        constant_of(reads, m.vae.latent.shift, "vae.decoder.conv_in.bias", SHIFT_FACTOR)
        vae_read(reads, m.vae, 2 * m.vae.encoder.conv_out.c_out)

def raw_of(w, name, what):
    held = stored(name)
    if held.raw == None:
        fail("`{}`: `{}` is stored {}; {}".format(w.name, name, held, what))
    return held.raw

def pad_table(reads, w, token):
    dt = raw_of(w, token, "a pad token is a raw float row")
    dim = w.shape[0] // 2
    reads.read_expr(w, concat(0, [
        constant(w.name, [-1.0] * dim, [dim, 1], dt),
        src(token).transmute([dim, 1], raw(dt)),
    ]))

def constant_of(reads, w, seed, value):
    held = stored(seed)
    dt = raw_of(w, seed, "a constant is stated in a raw dtype")
    expr = constant(w.name, [value] * product(w.shape), w.shape, dt)
    want = encoding(w.dtype)
    if want != held:
        expr = expr.cast(want)
    reads.push(tensor(w.name, expr, want, shape = w.shape))

def dit(reads, m, prefix):
    at = lambda tail: prefix + tail
    biased(reads, m.x_embed, at("all_x_embedder.2-1"))
    pad_table(reads, m.x_pad_mod, at("x_pad_token"))
    reads.read(m.cap_norm, at("cap_embedder.0.weight"))
    biased(reads, m.cap_embed, at("cap_embedder.1"))
    pad_table(reads, m.cap_pad_mod, at("cap_pad_token"))
    biased(reads, m.t_mlp0, at("t_embedder.mlp.0"))
    biased(reads, m.t_mlp1, at("t_embedder.mlp.2"))
    constant_of(reads, m.t_flip, at("t_embedder.mlp.0.bias"), T_FLIP)
    for stem, blocks in [
        ("noise_refiner", m.noise_refiner),
        ("context_refiner", m.context_refiner),
        ("layers", m.layers),
    ]:
        for i, b in enumerate(blocks):
            transformer_block(reads, b, at("{}.{}".format(stem, i)))
    biased(reads, m.final_ada, at("all_final_layer.2-1.adaLN_modulation.1"))
    biased(reads, m.final_linear, at("all_final_layer.2-1.linear"))

def transformer_block(reads, b, stem):
    n = lambda s: stem + "." + s
    if b.ada != None:
        biased(reads, b.ada, n("adaLN_modulation.0"))
    reads.read(b.attn_norm1, n("attention_norm1.weight"))
    reads.read(b.attn_norm2, n("attention_norm2.weight"))
    reads.read(b.ffn_norm1, n("ffn_norm1.weight"))
    reads.read(b.ffn_norm2, n("ffn_norm2.weight"))
    reads.read_concat(b.attn.qkv, [
        n("attention.to_q.weight"),
        n("attention.to_k.weight"),
        n("attention.to_v.weight"),
    ])
    reads.read(b.attn.q_norm, n("attention.norm_q.weight"))
    reads.read(b.attn.k_norm, n("attention.norm_k.weight"))
    reads.read(b.attn.out, n("attention.to_out.0.weight"))
    reads.read_concat(b.mlp.gate_up, [n("feed_forward.w1.weight"), n("feed_forward.w3.weight")])
    reads.read(b.mlp.down, n("feed_forward.w2.weight"))
