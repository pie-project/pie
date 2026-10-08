# How a Z-Image checkpoint is laid out: a diffusers pipeline, its components
# under `dit.`, `te.` and `vae.`; or, for a model of the transformer alone,
# a bare transformer state_dict.

DIFFUSERS = "a diffusers pipeline (`dit.`/`te.` prefixes)"
BARE = "a bare transformer state_dict"
T_FLIP = 1000.0
SHIFT_FACTOR = 0.1159

def formats(m):
    pipeline = lambda checkpoint: checkpoint.has_prefix("dit.")
    out = [format(DIFFUSERS, recognizes = pipeline, read = lambda reads: read(m, reads, DIFFUSERS))]
    if m.te == None:
        out.append(format(
            BARE,
            recognizes = lambda checkpoint: not pipeline(checkpoint),
            read = lambda reads: read(m, reads, BARE),
        ))
    return out

def read(m, reads, layout):
    dit(reads, m.dit, layout)
    if m.te != None:
        text_encoder(reads, m.te, layout)
    if m.vae != None:
        vae(reads, m.vae, layout)

def product(xs):
    out = 1
    for x in xs:
        out *= x
    return out

def raw_of(w, name, what):
    held = stored(name)
    if held.raw == None:
        fail("`{}`: `{}` is stored {}; {}".format(w.name, name, held, what))
    return held.raw

def biased(reads, w, stem):
    reads.read(w.w, stem + ".weight")
    reads.read(w.bias, stem + ".bias")

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

def dit(reads, m, layout):
    at = lambda tail: "dit." + tail if layout == DIFFUSERS else tail
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

def text_encoder(reads, te, layout):
    if layout != DIFFUSERS:
        fail("`te.embed_tokens.weight`: {} carries no text encoder, and this row declares one".format(layout))
    at = lambda tail: "te.model." + tail
    reads.read(te.embed, at("embed_tokens.weight"))
    for l, w in enumerate(te.layers):
        n = lambda s: at("layers.{}.{}".format(l, s))
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

def vae(reads, v, layout):
    if layout != DIFFUSERS:
        fail("`vae.decoder.conv_in.bias`: {} carries no VAE, and this row declares one".format(layout))
    at = lambda tail: "vae." + tail
    constant_of(reads, v.shift, at("decoder.conv_in.bias"), SHIFT_FACTOR)

    d = v.decoder
    conv(reads, d.conv_in, at("decoder.conv_in"))
    mid_block(reads, d.mid, at("decoder.mid_block"))
    for i, block in enumerate(d.up):
        for r, res in enumerate(block.resnets):
            resnet(reads, res, at("decoder.up_blocks.{}.resnets.{}".format(i, r)))
        if block.upsample != None:
            conv(reads, block.upsample, at("decoder.up_blocks.{}.upsamplers.0.conv".format(i)))
    norm(reads, d.norm_out, at("decoder.conv_norm_out"))
    conv(reads, d.conv_out, at("decoder.conv_out"))

    e = v.encoder
    conv(reads, e.conv_in, at("encoder.conv_in"))
    for i, block in enumerate(e.down):
        for r, res in enumerate(block.resnets):
            resnet(reads, res, at("encoder.down_blocks.{}.resnets.{}".format(i, r)))
        if block.downsample != None:
            conv(reads, block.downsample, at("encoder.down_blocks.{}.downsamplers.0.conv".format(i)))
    mid_block(reads, e.mid, at("encoder.mid_block"))
    norm(reads, e.norm_out, at("encoder.conv_norm_out"))
    conv_head(reads, e.conv_out, at("encoder.conv_out"), v.encoder_out_stored)

def mid_block(reads, m, stem):
    resnet(reads, m.res0, stem + ".resnets.0")
    a = m.attn
    n = lambda s: stem + ".attentions.0." + s
    norm(reads, a.norm, n("group_norm"))
    biased(reads, a.q, n("to_q"))
    biased(reads, a.k, n("to_k"))
    biased(reads, a.v, n("to_v"))
    biased(reads, a.out, n("to_out.0"))
    resnet(reads, m.res1, stem + ".resnets.1")

def resnet(reads, r, stem):
    norm(reads, r.norm1, stem + ".norm1")
    conv(reads, r.conv1, stem + ".conv1")
    norm(reads, r.norm2, stem + ".norm2")
    conv(reads, r.conv2, stem + ".conv2")
    if r.shortcut != None:
        conv(reads, r.shortcut, stem + ".conv_shortcut")

def norm(reads, n, stem):
    reads.read(n.weight, stem + ".weight")
    reads.read(n.bias, stem + ".bias")

def conv(reads, c, stem):
    kernel = stem + ".weight"
    dt = raw_of(c.w, kernel, "a conv kernel is a raw plane")
    reads.read_expr(c.w, src(kernel).transmute(c.w.shape, raw(dt)))
    reads.read(c.bias, stem + ".bias")

def conv_head(reads, c, stem, rows_stored):
    kernel = stem + ".weight"
    dt = raw_of(c.w, kernel, "a conv kernel is a raw plane")
    natural = list(c.w.shape)
    natural[0] = rows_stored
    rows = c.c_out
    reads.read_expr(c.w, src(kernel).transmute(natural, raw(dt)).slice(0, 0, rows))
    bias = stem + ".bias"
    held = stored(bias)
    want = encoding(c.bias.dtype)
    head = c.bias.name + ".head"
    reads.push(tensor(head, src(bias).slice(0, 0, rows), held, shape = c.bias.shape, internal = True))
    reads.push(tensor(c.bias.name, out(head) if want == held else out(head).cast(want), want, shape = c.bias.shape))
