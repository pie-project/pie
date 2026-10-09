# How a Wan 2.2 checkpoint is laid out: a diffusers pipeline, its
# components under `dit.`, `te.` and `vae.`; or, for a denoiser alone, a bare
# transformer state_dict.

load("//lib/diffusion/formats.star", "adaln_order", "biased", "packed", "reordered")
load("//lib/reads/formats.star", "product")

MOD_SLICES = 6
HEAD_SLICES = 2

VAE_LATENTS_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
]
VAE_LATENTS_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]

DIFFUSERS = "a diffusers pipeline (`dit.`/`te.`/`vae.` prefixes)"
BARE = "a bare transformer state_dict"

def formats(m):
    pipeline = lambda checkpoint: checkpoint.has_prefix("dit.")
    out = [format(DIFFUSERS, recognizes = pipeline, read = lambda reads: read(m, reads, "dit."))]
    if m.te == None and m.vae == None:
        out.append(format(
            BARE,
            recognizes = lambda checkpoint: not pipeline(checkpoint),
            read = lambda reads: read(m, reads, ""),
        ))
    return out

def read(m, reads, prefix):
    dit(reads, m.dit, m.dims, prefix)
    if m.te != None:
        text_encoder(reads, m.te)
    if m.vae != None:
        vae(reads, m.vae)

def transmuted(reads, w, name):
    held = stored(name)
    reads.read_over(w, name, lambda e: e.transmute(w.shape, held))

def table(reads, w, name, slices):
    held = stored(name)
    reads.read_over(w, name, lambda e: reordered(e, adaln_order(slices), 1, 1).transmute(w.shape, held))

def head_rows(c_out):
    rows = []
    for c in range(c_out):
        for ph in range(2):
            for pw in range(2):
                rows.append((ph * 2 + pw) * c_out + c)
    return rows

def time_conv_rows(c):
    rows = []
    for ch in range(c):
        rows += [ch, c + ch]
    return rows

def conv_out_rows(v):
    p = v.patch
    rows = []
    for c in range(v.rgb):
        for ph in range(p):
            for pw in range(p):
                rows.append(c * p * p + pw * p + ph)
    return rows

def dit(reads, m, d, prefix):
    at = lambda tail: prefix + tail
    dim = d.dim

    transmuted(reads, m.patch_embed.w, at("patch_embedding.weight"))
    reads.read(m.patch_embed.bias, at("patch_embedding.bias"))

    cond = lambda s: at("condition_embedder." + s)
    biased(reads, m.text_embed.linear_1, cond("text_embedder.linear_1"))
    biased(reads, m.text_embed.linear_2, cond("text_embedder.linear_2"))
    biased(reads, m.time_embed.linear_1, cond("time_embedder.linear_1"))
    biased(reads, m.time_embed.linear_2, cond("time_embedder.linear_2"))
    l2 = cond("time_embedder.linear_2")
    doubled = lambda e: concat(0, [e, e])
    reads.read_over(m.head_proj.w, l2 + ".weight", doubled)
    reads.read_over(m.head_proj.bias, l2 + ".bias", doubled)
    proj = cond("time_proj")
    swap = lambda e: reordered(e, adaln_order(MOD_SLICES), dim)
    reads.read_over(m.time_proj.w, proj + ".weight", swap)
    reads.read_over(m.time_proj.bias, proj + ".bias", swap)

    for i, b in enumerate(m.blocks):
        transformer_block(reads, b, at("blocks.{}".format(i)))

    table(reads, m.head_table, at("scale_shift_table"), HEAD_SLICES)
    rows = head_rows(d.channels)
    reads.read_over(m.proj_out.w, at("proj_out.weight"), lambda e: e.gather(0, rows))
    reads.read_over(m.proj_out.bias, at("proj_out.bias"), lambda e: e.gather(0, rows))

def transformer_block(reads, b, stem):
    n = lambda s: stem + "." + s
    table(reads, b.table, n("scale_shift_table"), MOD_SLICES)

    a = b.self_attn
    packed(reads, a.qkv, [n("attn1.to_q"), n("attn1.to_k"), n("attn1.to_v")])
    reads.read(a.norm_q, n("attn1.norm_q.weight"))
    reads.read(a.norm_k, n("attn1.norm_k.weight"))
    biased(reads, a.out, n("attn1.to_out.0"))

    reads.read(b.norm2, n("norm2.weight"))
    reads.read(b.norm2_bias, n("norm2.bias"))

    c = b.cross
    biased(reads, c.q, n("attn2.to_q"))
    packed(reads, c.kv, [n("attn2.to_k"), n("attn2.to_v")])
    reads.read(c.norm_q, n("attn2.norm_q.weight"))
    reads.read(c.norm_k, n("attn2.norm_k.weight"))
    biased(reads, c.out, n("attn2.to_out.0"))

    biased(reads, b.ffn.up, n("ffn.net.0.proj"))
    biased(reads, b.ffn.down, n("ffn.net.2"))

def text_encoder(reads, te):
    at = lambda tail: "te." + tail
    reads.read(te.embed, at("shared.weight"))
    for l, w in enumerate(te.layers):
        attn = lambda s: at("encoder.block.{}.layer.0.{}".format(l, s))
        ffn = lambda s: at("encoder.block.{}.layer.1.{}".format(l, s))
        reads.read(w.attn_norm, attn("layer_norm.weight"))
        reads.read(w.q, attn("SelfAttention.q.weight"))
        reads.read(w.k, attn("SelfAttention.k.weight"))
        reads.read(w.v, attn("SelfAttention.v.weight"))
        reads.read(w.o, attn("SelfAttention.o.weight"))
        reads.read(w.rel_bias, attn("SelfAttention.relative_attention_bias.weight"))
        reads.read(w.ffn_norm, ffn("layer_norm.weight"))
        reads.read(w.wi_0, ffn("DenseReluDense.wi_0.weight"))
        reads.read(w.wi_1, ffn("DenseReluDense.wi_1.weight"))
        reads.read(w.wo, ffn("DenseReluDense.wo.weight"))
    reads.read(te.final_norm, at("encoder.final_layer_norm.weight"))

def row_of(reads, w, seed, value):
    """`w`, a row the model states (`value(i)` its i-th value), stored as
    the raw dtype of the checkpoint's `seed`."""
    held = stored(seed)
    if held.raw == None:
        fail("`{}`: `{}` is stored {}; a stated row wants a raw dtype".format(w.name, seed, held))
    n = product(w.shape)
    values = [value(i) for i in range(n)]
    want = encoding(w.dtype)
    row = constant(w.name, values, [n], held.raw)
    if want != held:
        row = row.cast(want)
    reads.push(tensor(w.name, row, want, shape = w.shape))

def conv(reads, c, stem, rows = None):
    name = stem + ".weight"
    held = stored(name)
    if rows != None:
        reads.read_over(c.w, name, lambda e: e.gather(0, rows).transmute(c.w.shape, held))
        reads.read_over(c.bias, stem + ".bias", lambda e: e.gather(0, rows))
    else:
        reads.read_over(c.w, name, lambda e: e.transmute(c.w.shape, held))
        reads.read(c.bias, stem + ".bias")

def resnet(reads, r, stem):
    transmuted(reads, r.norm1, stem + ".norm1.gamma")
    conv(reads, r.conv1, stem + ".conv1")
    transmuted(reads, r.norm2, stem + ".norm2.gamma")
    conv(reads, r.conv2, stem + ".conv2")
    if r.shortcut != None:
        conv(reads, r.shortcut, stem + ".conv_shortcut")

def mid_attention(reads, a, attn):
    transmuted(reads, a.norm, attn + ".norm.gamma")
    transmuted(reads, a.qkv.w, attn + ".to_qkv.weight")
    reads.read(a.qkv.bias, attn + ".to_qkv.bias")
    transmuted(reads, a.proj.w, attn + ".proj.weight")
    reads.read(a.proj.bias, attn + ".proj.bias")

def vae(reads, v):
    at = lambda tail: "vae." + tail
    std = lambda i: f32(VAE_LATENTS_STD[i])
    mean = lambda i: f32(VAE_LATENTS_MEAN[i])
    row_of(reads, v.denorm_scale, at("post_quant_conv.bias"), std)
    row_of(reads, v.denorm_bias, at("post_quant_conv.bias"), lambda i: f32(-mean(i) / std(i)))

    conv(reads, v.post_quant, at("post_quant_conv"))
    conv(reads, v.conv_in, at("decoder.conv_in"))

    resnet(reads, v.mid_res0, at("decoder.mid_block.resnets.0"))
    mid_attention(reads, v.mid_attn, at("decoder.mid_block.attentions.0"))
    resnet(reads, v.mid_res1, at("decoder.mid_block.resnets.1"))

    for i, up in enumerate(v.up):
        for r, b in enumerate(up.resnets):
            resnet(reads, b, at("decoder.up_blocks.{}.resnets.{}".format(i, r)))
        u = up.upsampler
        if u != None:
            stem = at("decoder.up_blocks.{}.upsampler".format(i))
            if u.time_conv != None:
                conv(reads, u.time_conv, stem + ".time_conv", time_conv_rows(u.time_conv.c_in))
            conv(reads, u.resample, stem + ".resample.1")

    transmuted(reads, v.norm_out, at("decoder.norm_out.gamma"))
    conv(reads, v.conv_out, at("decoder.conv_out"), conv_out_rows(v))
    encoder(reads, v)

def encoder(reads, v):
    e = v.enc
    at = lambda tail: "vae." + tail
    std = lambda i: f32(VAE_LATENTS_STD[i])
    row_of(reads, e.norm_bias, at("quant_conv.bias"), lambda i: f32(VAE_LATENTS_MEAN[i]))
    row_of(reads, e.norm_scale, at("quant_conv.bias"), lambda i: f32(1.0 / std(i)))

    name = at("encoder.conv_in.weight")
    held = stored(name)
    cols = conv_out_rows(v)
    reads.read_over(e.conv_in.w, name, lambda x: x.gather(1, cols).transmute(e.conv_in.w.shape, held))
    reads.read(e.conv_in.bias, at("encoder.conv_in.bias"))

    for i, b in enumerate(e.down):
        for r, res in enumerate(b.resnets):
            resnet(reads, res, at("encoder.down_blocks.{}.resnets.{}".format(i, r)))
        dn = b.downsampler
        if dn != None:
            stem = at("encoder.down_blocks.{}.downsampler".format(i))
            conv(reads, dn.resample, stem + ".resample.1")
            if dn.time_conv != None:
                conv(reads, dn.time_conv, stem + ".time_conv")

    resnet(reads, e.mid_res0, at("encoder.mid_block.resnets.0"))
    mid_attention(reads, e.mid_attn, at("encoder.mid_block.attentions.0"))
    resnet(reads, e.mid_res1, at("encoder.mid_block.resnets.1"))

    transmuted(reads, e.norm_out, at("encoder.norm_out.gamma"))
    conv(reads, e.conv_out, at("encoder.conv_out"))
    name = at("quant_conv.weight")
    held = stored(name)
    reads.read_over(e.quant.w, name, lambda x: x.slice(0, 0, v.z).transmute(e.quant.w.shape, held))
    reads.read_over(e.quant.bias, at("quant_conv.bias"), lambda x: x.slice(0, 0, v.z))
