# How an LTX-2.5 checkpoint is laid out: a diffusers pipeline, the
# transformer under `dit.`, the text connectors under `connectors.` and the
# VAE under `vae.`.

load("//lib/diffusion/formats.star", "adaln_order", "biased", "packed", "reordered")

AV_SS_SLICES = 4
AV_GATE_SLICES = 1
VAE_RGB = 3

def formats(m):
    return [format("diffusers", read = lambda reads: read(m, reads))]

def read(m, reads):
    dit(reads, m.dit, m.dims)
    connector(reads, m.connectors[0], "video")
    connector(reads, m.connectors[1], "audio")
    if m.vae != None:
        vae(reads, m.vae)

def vae(reads, v):
    at = lambda tail: "vae." + tail
    reads.read(v.latents_mean, at("latents_mean"))
    reads.read(v.latents_std, at("latents_std"))
    zero_row(reads, v.zero)

    vae_conv(reads, v.conv_in, at("decoder.conv_in"))
    for r, res in enumerate(v.mid):
        vae_resnet(reads, res, at("decoder.mid_block.resnets.{}".format(r)))
    for i, up in enumerate(v.up):
        stem = at("decoder.up_blocks.{}".format(i))
        vae_conv(reads, up.upsampler, stem + ".upsamplers.0.conv")
        for r, res in enumerate(up.resnets):
            vae_resnet(reads, res, "{}.resnets.{}".format(stem, r))
    vae_conv(reads, v.conv_out, at("decoder.conv_out"), conv_out_rows(v.patch))

def vae_resnet(reads, r, stem):
    vae_conv(reads, r.conv1, stem + ".conv1")
    vae_conv(reads, r.conv2, stem + ".conv2")

def vae_conv(reads, c, stem, rows = None):
    name = stem + ".conv.weight"
    held = stored(name)
    bias = stem + ".conv.bias"
    if rows != None:
        reads.read_over(c.w, name, lambda e: e.gather(0, rows).transmute(c.w.shape, held))
        reads.read_over(c.bias, bias, lambda e: e.gather(0, rows))
    else:
        reads.read_over(c.w, name, lambda e: e.transmute(c.w.shape, held))
        reads.read(c.bias, bias)

def conv_out_rows(p):
    rows = []
    for c in range(VAE_RGB):
        for ph in range(p):
            for pw in range(p):
                rows.append(c * p * p + pw * p + ph)
    return rows

def zero_row(reads, w):
    want = encoding(w.dtype)
    if want.raw == None:
        fail("`{}`: declared {}; a stated zero row wants a raw dtype".format(w.name, want))
    reads.push(tensor(w.name, fill(0.0, w.shape, raw(want.raw)), want, shape = w.shape))

def dit(reads, m, d):
    at = lambda tail: "dit." + tail
    stream(reads, m.video, at("proj_in"), at("time_embed"), "", d.dim)
    stream(reads, m.audio, at("audio_proj_in"), at("audio_time_embed"), "audio_", d.audio_dim)
    adaln(reads, m.prompt, at("prompt_adaln"), d.dim, adaln_order(2))
    adaln(reads, m.audio_prompt, at("audio_prompt_adaln"), d.audio_dim, adaln_order(2))
    for i, b in enumerate(m.blocks):
        transformer_block(reads, b, at("transformer_blocks.{}".format(i)), d)

def stream(reads, s, proj_in, time_embed, prefix, dim):
    video = prefix == ""
    biased(reads, s.patchify, proj_in)
    adaln(reads, s.adaln, time_embed, dim, adaln_order(9))
    l2 = time_embed + ".emb.timestep_embedder.linear_2"
    rows = list(range(dim)) + list(range(dim))
    twice = lambda tail: src(l2 + "." + tail).gather(0, rows)
    reads.read_expr(s.head_proj.w, twice("weight"))
    reads.read_expr(s.head_proj.bias, twice("bias"))
    table(reads, s.head_table, "dit.{}scale_shift_table".format(prefix), adaln_order(2))
    av = "dit.av_cross_attn_{}".format("video" if video else "audio")
    adaln(reads, s.av_ss, av + "_scale_shift", dim, [0, 1, 2, 3])
    adaln(reads, s.av_gate, av + ("_a2v_gate" if video else "_v2a_gate"), dim, [0])
    biased(reads, s.proj_out, "dit.{}proj_out".format(prefix))

def adaln(reads, head, stem, dim, order):
    emb = stem + ".emb.timestep_embedder"
    biased(reads, head.embed.linear_1, emb + ".linear_1")
    biased(reads, head.embed.linear_2, emb + ".linear_2")
    proj = stem + ".linear"
    reads.read_expr(head.proj.w, ordered(proj + ".weight", order, dim))
    reads.read_expr(head.proj.bias, ordered(proj + ".bias", order, dim))

def transformer_block(reads, b, stem, d):
    side(reads, b.video, stem, "", d.dim)
    side(reads, b.audio, stem, "audio_", d.audio_dim)
    attention(reads, b.a2v, stem + ".audio_to_video_attn")
    attention(reads, b.v2a, stem + ".video_to_audio_attn")

def side(reads, s, stem, prefix, dim):
    table(reads, s.table, "{}.{}scale_shift_table".format(stem, prefix), adaln_order(9))
    av = "{}.{}_a2v_cross_attn_scale_shift_table".format(stem, "video" if prefix == "" else "audio")
    banded(reads, s.av_ss_table, av, 0, AV_SS_SLICES)
    banded(reads, s.av_gate_table, av, AV_SS_SLICES, AV_GATE_SLICES)
    table(reads, s.prompt_table, "{}.{}prompt_scale_shift_table".format(stem, prefix), adaln_order(2))
    attention(reads, s.self_attn, "{}.{}attn1".format(stem, prefix))
    attention(reads, s.cross, "{}.{}attn2".format(stem, prefix))
    feed_forward(reads, s.ffn, "{}.{}ff".format(stem, prefix))

def attention(reads, a, stem):
    if a.kv == None:
        packed(reads, a.qkv, [stem + ".to_q", stem + ".to_k", stem + ".to_v"])
    else:
        biased(reads, a.qkv, stem + ".to_q")
        packed(reads, a.kv, [stem + ".to_k", stem + ".to_v"])
    reads.read(a.q_norm, stem + ".norm_q.weight")
    reads.read(a.k_norm, stem + ".norm_k.weight")
    biased(reads, a.gate, stem + ".to_gate_logits")
    biased(reads, a.out, stem + ".to_out.0")

def feed_forward(reads, ff, stem):
    biased(reads, ff.up, stem + ".net.0.proj")
    biased(reads, ff.down, stem + ".net.2")

def connector(reads, conn, stem):
    biased(reads, conn.aggregate, "connectors.{}_text_proj_in".format(stem))
    for l, b in enumerate(conn.blocks):
        at = "connectors.{}_connector.transformer_blocks.{}".format(stem, l)
        attention(reads, b.attn, at + ".attn1")
        feed_forward(reads, b.ffn, at + ".ff")

def table(reads, w, name, order):
    held = stored(name)
    reads.read_expr(w, ordered(name, order, 1).transmute(w.shape, held))

def banded(reads, w, name, start, length):
    held = stored(name)
    reads.read_expr(w, src(name).slice(0, start, length).transmute(w.shape, held))

def ordered(name, order, width):
    """`name`'s `width`-wide slices in `order`; as stored if that is theirs."""
    if order == list(range(len(order))):
        return src(name)
    return reordered(src(name), order, width)
