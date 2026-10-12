# The weights of LTX-2.5: paired video and audio streams, each with its own
# self- and cross-attention and their attention across to each other; the
# text connectors that refine the prompt for each; and the video VAE decoder.

def linear(name, out, in_, banks, bias = True):
    """A projection from `in_` to `out` and, unless `bias` is off, its bias."""
    return struct(
        w = weight(name, [out, in_], banks),
        bias = weight(name + ".bias", [out], compute(banks)) if bias else None,
    )

def packed_linear(name, seams, in_, banks):
    """A biased projection whose output is the `seams`-wide pieces end to end."""
    out = 0
    for s in seams:
        out += s
    return struct(
        w = weight(name, [out, in_], banks).packed(seams),
        bias = weight(name + ".bias", [out], compute(banks)).packed(seams),
    )

def embedder(prefix, in_, dim, banks):
    """A two-layer timestep (or caption) embedder from `in_` to `dim`."""
    return struct(
        linear_1 = linear(prefix + ".1", dim, in_, banks),
        linear_2 = linear(prefix + ".2", dim, dim, banks),
    )

PATCH_T = 1
PATCH_H = 1
PATCH_W = 1
VAE_SPATIAL_COMPRESSION = 32
VAE_TEMPORAL_COMPRESSION = 8
VAE_Z = 128
VAE_RGB = 3
VAE_PATCH = 4
VAE_DECODER_DIMS = [1024, 512, 512, 256, 128]
VAE_MID_RESNETS = 2
VAE_UP_RESNETS = [2, 4, 6, 4]
VAE_UP_STRIDES = [[2, 2, 2], [2, 2, 2], [2, 1, 1], [1, 2, 2]]
T_FREQ_DIM = 256
ROPE_AXES = 3
MOD_SLICES = 9
AV_SS_SLICES = 4
AV_GATE_SLICES = 1
PROMPT_SLICES = 2
HEAD_SLICES = 2
TRAIN_STEPS = 1000
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
TEXT_LAYERS = 49
TEXT_LEN = 1024
CONN_FF_MULT = 4
HEAD_DIM = 128
AUDIO_HEAD_DIM = 64
CHANNELS = 128
FF_MULT = 4

def dims(layers, heads, audio_heads, cross_dim, audio_cross_dim, caption, conn_layers):
    dim = heads * HEAD_DIM
    audio_dim = audio_heads * AUDIO_HEAD_DIM
    f = dim // (2 * ROPE_AXES)
    return struct(
        layers = layers,
        heads = heads,
        head_dim = HEAD_DIM,
        audio_heads = audio_heads,
        audio_head_dim = AUDIO_HEAD_DIM,
        channels = CHANNELS,
        cross_dim = cross_dim,
        audio_cross_dim = audio_cross_dim,
        caption = caption,
        conn_layers = conn_layers,
        dim = dim,
        audio_dim = audio_dim,
        text_in = caption * TEXT_LAYERS,
        rope_dims = [2 * f, 2 * f, 2 * f, 0],
        audio_rope_dims = [audio_dim, 0, 0, 0],
    )

DIMS = {
    "ltx25": dims(
        layers = 48,
        heads = 32,
        audio_heads = 32,
        cross_dim = 4096,
        audio_cross_dim = 2048,
        caption = 3840,
        conn_layers = 8,
    ),
    "ltx25-mini": dims(
        layers = 2,
        heads = 2,
        audio_heads = 2,
        cross_dim = 256,
        audio_cross_dim = 128,
        caption = 16,
        conn_layers = 1,
    ),
}

def gain(name, width, banks):
    return weight(name, [width], compute(banks))

def attention(prefix, dim, heads, head_dim, banks, ctx = None):
    """Self-attention, or (given the `ctx` width it attends) cross-attention,
    gated per head."""
    inner = heads * head_dim
    if ctx == None:
        qkv = packed_linear(prefix + ".qkv", [inner, inner, inner], dim, banks)
        kv = None
    else:
        qkv = linear(prefix + ".q", inner, dim, banks)
        kv = packed_linear(prefix + ".kv", [inner, inner], ctx, banks)
    return struct(
        qkv = qkv,
        kv = kv,
        q_norm = gain(prefix + ".q_norm", inner, banks),
        k_norm = gain(prefix + ".k_norm", inner, banks),
        gate = linear(prefix + ".gate", heads, dim, banks),
        out = linear(prefix + ".out", dim, inner, banks),
        heads = heads,
        head_dim = head_dim,
        inner = inner,
        sm_scale = f32(1.0 / f32(sqrt(head_dim))),
    )

def ffn(prefix, dim, mult, bias, banks):
    inner = dim * mult
    return struct(
        up = linear(prefix + ".up", inner, dim, banks, bias = bias),
        down = linear(prefix + ".down", dim, inner, banks, bias = bias),
    )

def block(prefix, d, banks):
    table = lambda name, slices, width: weight(name, [slices * width], dtype.f32)

    def side(stem, width, heads, head_dim, ctx, bias):
        return struct(
            table = table(stem + ".table", MOD_SLICES, width),
            av_ss_table = table(stem + ".av_ss_table", AV_SS_SLICES, width),
            av_gate_table = table(stem + ".av_gate_table", AV_GATE_SLICES, width),
            prompt_table = table(stem + ".prompt_table", PROMPT_SLICES, width),
            self_attn = attention(stem + ".self", width, heads, head_dim, banks),
            cross = attention(stem + ".cross", width, heads, head_dim, banks, ctx),
            ffn = ffn(stem + ".ffn", width, FF_MULT, bias, banks),
        )

    return struct(
        video = side(prefix + ".video", d.dim, d.heads, d.head_dim, d.cross_dim, False),
        audio = side(prefix + ".audio", d.audio_dim, d.audio_heads, d.audio_head_dim, d.audio_cross_dim, True),
        a2v = attention(prefix + ".a2v", d.dim, d.audio_heads, d.audio_head_dim, banks, d.audio_dim),
        v2a = attention(prefix + ".v2a", d.audio_dim, d.audio_heads, d.audio_head_dim, banks, d.dim),
    )

def adaln(prefix, dim, slices, banks):
    return struct(
        embed = embedder(prefix + ".emb", T_FREQ_DIM, dim, banks),
        proj = linear(prefix + ".proj", slices * dim, dim, banks),
        slices = slices,
    )

def stream(prefix, channels, dim, banks):
    return struct(
        patchify = linear(prefix + ".patchify", dim, channels, banks),
        adaln = adaln(prefix + ".adaln", dim, MOD_SLICES, banks),
        av_ss = adaln(prefix + ".av_ss", dim, AV_SS_SLICES, banks),
        av_gate = adaln(prefix + ".av_gate", dim, AV_GATE_SLICES, banks),
        head_proj = linear(prefix + ".head_proj", HEAD_SLICES * dim, dim, banks),
        head_table = weight(prefix + ".head_table", [HEAD_SLICES * dim], dtype.f32),
        proj_out = linear(prefix + ".proj_out", channels, dim, banks),
    )

def connector(prefix, text_in, dim, heads, layers, caption, banks):
    head_dim = dim // heads
    return struct(
        aggregate = linear(prefix + ".aggregate", dim, text_in, banks),
        blocks = [
            struct(
                attn = attention("{}.block.{}.attn".format(prefix, l), dim, heads, head_dim, banks),
                ffn = ffn("{}.block.{}.ffn".format(prefix, l), dim, CONN_FF_MULT, True, banks),
            )
            for l in range(layers)
        ],
        dim = dim,
        heads = heads,
        head_dim = head_dim,
        rope_dims = [dim, 0, 0, 0],
        rescale = f32(sqrt(dim / caption)),
    )

def vae_conv(name, c_out, c_in, banks):
    taps = 27
    return struct(
        w = weight(name, [c_out, c_in * taps], banks).conv_taps_major(c_in, taps),
        bias = weight(name + ".bias", [c_out], dtype.f32),
        c_in = c_in,
        c_out = c_out,
    )

def vae_resnet(prefix, c, banks):
    return struct(
        conv1 = vae_conv(prefix + ".conv1", c, c, banks),
        conv2 = vae_conv(prefix + ".conv2", c, c, banks),
    )

def vae(banks):
    dims = VAE_DECODER_DIMS
    top = dims[0]
    up = []
    for i in range(4):
        c_in, c_out = dims[i], dims[i + 1]
        stride = VAE_UP_STRIDES[i]
        prefix = "vae.up.{}".format(i)
        up.append(struct(
            upsampler = vae_conv(prefix + ".upsampler", c_out * stride[0] * stride[1] * stride[2], c_in, banks),
            stride = stride,
            resnets = [vae_resnet("{}.res.{}".format(prefix, r), c_out, banks) for r in range(VAE_UP_RESNETS[i])],
        ))
    dense = compute(banks)
    row = lambda name: weight(name, [VAE_Z], dense)
    return struct(
        z = VAE_Z,
        patch = VAE_PATCH,
        latents_mean = row("vae.latents_mean"),
        latents_std = row("vae.latents_std"),
        zero = row("vae.zero"),
        conv_in = vae_conv("vae.conv_in", top, VAE_Z, banks),
        mid = [vae_resnet("vae.mid.res.{}".format(r), top, banks) for r in range(VAE_MID_RESNETS)],
        up = up,
        conv_out = vae_conv("vae.conv_out", VAE_RGB * VAE_PATCH * VAE_PATCH, dims[4], banks),
    )

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    banks = deploy.weights[0]
    d = DIMS[id]
    return struct(
        dims = d,
        rope_axes = ROPE_AXES,
        t_freq_dim = T_FREQ_DIM,
        dit = struct(
            video = stream("dit.video", d.channels, d.dim, banks),
            audio = stream("dit.audio", d.channels, d.audio_dim, banks),
            prompt = adaln("dit.prompt", d.dim, PROMPT_SLICES, banks),
            audio_prompt = adaln("dit.audio_prompt", d.audio_dim, PROMPT_SLICES, banks),
            blocks = [block("dit.block.{}".format(i), d, banks) for i in range(d.layers)],
        ),
        connectors = (
            connector("connectors.video", d.text_in, d.cross_dim, d.heads, d.conn_layers, d.caption, banks),
            connector("connectors.audio", d.text_in, d.audio_cross_dim, d.audio_heads, d.conn_layers, d.caption, banks),
        ),
        vae = vae(dtype.bf16) if id == "ltx25" else None,
    )

def generative(m):
    d = m.dims
    refine_ports = [
        port("text", "latents", d.text_in, ["text"], at = 1),
        port("text_positions", "axis_positions", 1, ["text"], at = 1),
    ]
    readings = [
        reading(
            "denoise",
            index = 0,
            streams = ["video", "audio", "context", "reference"],
            ports = [
                port("latents", "latents", d.channels, ["video", "audio"], at = 0),
                port("context", "context", d.cross_dim, ["context"], at = 0),
                port("audio_context", "context", d.audio_cross_dim, ["reference"], at = 1),
                port("timestep", "lane_vector", 1, [], at = 0),
                port("positions", "axis_positions", ROPE_AXES, ["video"], at = 0),
                port("audio_positions", "axis_positions", 1, ["audio"], at = 1),
            ],
            readout = "velocity",
            readout_width = d.channels,
        ),
        reading(
            "refine.video",
            index = 1,
            streams = ["text"],
            ports = refine_ports,
            readout = "hidden",
            readout_width = d.cross_dim,
        ),
        reading(
            "refine.audio",
            index = 2,
            streams = ["text"],
            ports = refine_ports,
            readout = "hidden",
            readout_width = d.audio_cross_dim,
        ),
    ]
    if m.vae != None:
        readings.append(reading(
            "vae.decode",
            index = 3,
            streams = ["video"],
            ports = [port("latent", "voxels", VAE_Z, ["video"], at = 0)],
            readout = "pixels",
            readout_width = VAE_RGB,
        ))
    return generation(
        readings = readings,
        latent = latent_space(
            channels = d.channels,
            patch_t = PATCH_T,
            patch_h = PATCH_H,
            patch_w = PATCH_W,
            spatial_compression = VAE_SPATIAL_COMPRESSION,
            temporal_compression = VAE_TEMPORAL_COMPRESSION,
        ),
        schedule = schedule("flow", shift = 1.0, train_steps = TRAIN_STEPS, pinned_sigmas = DISTILLED_SIGMAS),
        max_rows = 32768 + 2 * TEXT_LEN if d.layers == 48 else 4096,
    )
