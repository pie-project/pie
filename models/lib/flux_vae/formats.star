# The Flux VAE as diffusers names it, under `vae.`.

load("//lib/diffusion/formats.star", "biased")

def conv(reads, c, stem):
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

def read(reads, v, encoder_out_stored = None):
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
