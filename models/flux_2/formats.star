# How a FLUX.2 checkpoint is laid out: a diffusers pipeline, its components
# under `dit.`, `te.` and `vae.`; or, for a model of the transformer alone,
# a bare transformer state_dict.

DIFFUSERS = "a diffusers pipeline (`dit.`/`te.`/`vae.` prefixes)"
BARE = "a bare transformer state_dict"
TE_HIDDEN = 2560
BN_EPS = 1e-4

def formats(m):
    pipeline = lambda checkpoint: checkpoint.has_prefix("dit.")
    out = [format(DIFFUSERS, recognizes = pipeline, read = lambda reads: read(m, reads, DIFFUSERS))]
    if m.te == None and m.vae == None:
        out.append(format(
            BARE,
            recognizes = lambda checkpoint: not pipeline(checkpoint),
            read = lambda reads: read(m, reads, BARE),
        ))
    return out

def read(m, reads, layout):
    dit(reads, m.dit, m.dims.dim, layout)
    if m.te != None:
        text_encoder(reads, m.te, layout)
    if m.vae != None:
        vae(reads, m.vae, layout)

def component(layout, prefix, tail):
    if layout != DIFFUSERS:
        fail("`{}{}`: {} carries no `{}` component, and this row declares one".format(prefix, tail, layout, prefix))
    return prefix + tail

def dit(reads, m, dim, layout):
    at = lambda tail: "dit." + tail if layout == DIFFUSERS else tail
    w = lambda tail: at(tail + ".weight")
    reads.read(m.x_embed, w("x_embedder"))
    if m.context_embed != None:
        reads.read(m.context_embed, w("context_embedder"))
    reads.read(m.t_embed.linear_1, w("time_guidance_embed.timestep_embedder.linear_1"))
    reads.read(m.t_embed.linear_2, w("time_guidance_embed.timestep_embedder.linear_2"))
    if m.g_embed != None:
        reads.read(m.g_embed.linear_1, w("time_guidance_embed.guidance_embedder.linear_1"))
        reads.read(m.g_embed.linear_2, w("time_guidance_embed.guidance_embedder.linear_2"))

    reordered(reads, m.mod_img, w("double_stream_modulation_img.linear"), 6, dim)
    reordered(reads, m.mod_txt, w("double_stream_modulation_txt.linear"), 6, dim)
    reordered(reads, m.mod_single, w("single_stream_modulation.linear"), 3, dim)

    for i, block in enumerate(m.double):
        stem = at("transformer_blocks.{}".format(i))
        attn(reads, block.img.attn, stem, ["to_q", "to_k", "to_v"], ["norm_q", "norm_k"], "to_out.0")
        swiglu(reads, block.img.ff, stem + ".ff")
        attn(
            reads,
            block.txt.attn,
            stem,
            ["add_q_proj", "add_k_proj", "add_v_proj"],
            ["norm_added_q", "norm_added_k"],
            "to_add_out",
        )
        swiglu(reads, block.txt.ff, stem + ".ff_context")

    for i, block in enumerate(m.single):
        stem = at("single_transformer_blocks.{}.attn".format(i))
        reads.read(block.in_proj, stem + ".to_qkv_mlp_proj.weight")
        reads.read(block.q_norm, stem + ".norm_q.weight")
        reads.read(block.k_norm, stem + ".norm_k.weight")
        out = stem + ".to_out.weight"
        column_block(reads, block.out_attn, out, 0, dim)
        column_block(reads, block.out_mlp, out, dim, block.out_mlp.shape[1])

    reads.read(m.norm_out, w("norm_out.linear"))
    reads.read(m.proj_out, w("proj_out"))

def attn(reads, a, stem, qkv, norms, out):
    n = lambda s: "{}.attn.{}.weight".format(stem, s)
    reads.read_concat(a.qkv, [n(s) for s in qkv])
    reads.read(a.q_norm, n(norms[0]))
    reads.read(a.k_norm, n(norms[1]))
    reads.read(a.out, n(out))

def swiglu(reads, ff, stem):
    reads.read(ff.linear_in, stem + ".linear_in.weight")
    reads.read(ff.linear_out, stem + ".linear_out.weight")

def reordered(reads, w, name, slices, dim):
    order = {3: [1, 0, 2], 6: [1, 0, 2, 4, 3, 5]}[slices]
    reads.read_expr(w, concat(0, [src(name).slice(0, i * dim, dim) for i in order]))

def text_encoder(reads, te, layout):
    at = lambda tail: component(layout, "te.model.", tail)
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
    embedder = "dit.context_embedder.weight" if layout == DIFFUSERS else "context_embedder.weight"
    for i, w in enumerate(te.context_embed):
        column_block(reads, w, embedder, TE_HIDDEN * i, TE_HIDDEN)

def column_block(reads, w, name, start, length):
    held = stored(name)
    want = encoding(w.dtype)
    sliced = src(name).slice(1, start, length)
    if held == want:
        reads.read_expr(w, sliced)
        return
    staged = w.name + ".read"
    reads.push(tensor(staged, sliced, held, shape = w.shape, internal = True))
    reads.push(tensor(w.name, out(staged).cast(want), want, shape = w.shape))

def vae(reads, v, layout):
    at = lambda tail: component(layout, "vae.", tail)
    batch_norm(reads, v, at("bn.running_mean"), at("bn.running_var"))
    conv(reads, v.post_quant_conv, at("post_quant_conv"))
    conv_head(reads, v.quant_conv, at("quant_conv"), 64)

    d = v.decoder
    conv(reads, d.conv_in, at("decoder.conv_in"))
    mid_block(reads, d.mid, at("decoder.mid_block"))
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
    mid_block(reads, e.mid, at("encoder.mid_block"))
    group_norm(reads, e.norm_out, at("encoder.conv_norm_out"))
    conv(reads, e.conv_out, at("encoder.conv_out"))

def conv(reads, c, stem):
    name = stem + ".weight"
    reads.read_expr(c.w, src(name).transmute(c.w.shape, stored(name)))
    reads.read(c.bias, stem + ".bias")

def conv_head(reads, c, stem, rows_stored):
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

def mid_block(reads, m, stem):
    resnet(reads, m.res0, stem + ".resnets.0")
    a = m.attn
    n = lambda s: stem + ".attentions.0." + s
    group_norm(reads, a.norm, n("group_norm"))
    for p, leaf in [(a.q, "to_q"), (a.k, "to_k"), (a.v, "to_v"), (a.out, "to_out.0")]:
        reads.read(p.w, n(leaf) + ".weight")
        reads.read(p.bias, n(leaf) + ".bias")
    resnet(reads, m.res1, stem + ".resnets.1")

def batch_norm(reads, v, mean, var):
    held = stored(var)
    shape = v.bn_scale.shape
    var_eps = v.bn_scale.name + ".var_eps"
    reads.push(tensor(var_eps, src(var).bias(BN_EPS), held, shape = shape, internal = True))
    for plane, op in [(v.bn_scale, "sqrt"), (v.bn_rscale, "rsqrt")]:
        root = plane.name + ".of_var"
        reads.push(tensor(root, out(var_eps).unary(op), held, shape = shape, internal = True))
        want = encoding(plane.dtype)
        reads.push(tensor(plane.name, out(root) if want == held else out(root).cast(want), want, shape = shape))
    reads.read(v.bn_mean, mean)
    want = encoding(v.bn_zero.dtype)
    if want.raw == None:
        fail("`{}`: a zero plane is stated raw, not {}".format(v.bn_zero.name, want))
    shape = v.bn_zero.shape
    reads.push(tensor(v.bn_zero.name, fill(0.0, shape, raw(want.raw)), want, shape = shape))
