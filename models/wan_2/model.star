# The weights of Wan 2.2: a diffusion transformer over video latents with
# cross-attention to text, its UMT5-XXL text encoder, and its causal 3D VAE.

PATCH_T = 1
PATCH_H = 2
PATCH_W = 2
PATCH_VOL = PATCH_T * PATCH_H * PATCH_W
VAE_SPATIAL_COMPRESSION = 16
VAE_TEMPORAL_COMPRESSION = 4
ROPE_AXES = 3
MOD_SLICES = 6
HEAD_SLICES = 2
TRAIN_STEPS = 1000
SHIFT_TI2V = 5.0
CONTEXT_LEN = 512

TE_HIDDEN = 4096
TE_VOCAB = 256384
TE_HEADS = 64
TE_HEAD_DIM = 64
TE_INTER = 10240
TE_LAYERS = 24
TE_BUCKETS = 32

VAE_Z = 48
VAE_PIX_CHANNELS = 12
VAE_RGB = 3
VAE_DECODER_DIMS = [1024, 1024, 1024, 512, 256]
VAE_RESNETS = 3
VAE_TEMPORAL_UP = [True, True, False, False]
VAE_ENCODER_DIMS = [160, 160, 320, 640, 640]
VAE_ENC_RESNETS = 2
VAE_TEMPORAL_DOWN = [False, True, True, False]
VAE_MAX_LATENT_PLANE = 44 * 80

def dims(dim, heads, head_dim, ffn, layers, in_channels, out_channels, text_dim, freq_dim):
    hw = 2 * (head_dim // 6)
    return struct(
        dim = dim,
        heads = heads,
        head_dim = head_dim,
        ffn = ffn,
        layers = layers,
        in_channels = in_channels,
        out_channels = out_channels,
        text_dim = text_dim,
        freq_dim = freq_dim,
        rope_dims = [head_dim - 2 * hw, hw, hw, 0],
        sm_scale = f32(1.0 / f32(sqrt(head_dim))),
        patch_in = in_channels * PATCH_VOL,
        patch_out = out_channels * PATCH_VOL,
    )

DIMS = {
    "wan22-ti2v-5b": dims(3072, 24, 128, 14336, 30, 48, 48, 4096, 256),
    "wan22-mini-d128": dims(256, 2, 128, 512, 2, 16, 16, 64, 256),
    "wan22-mini-nano": dims(48, 2, 24, 128, 2, 16, 16, 64, 32),
}

def linear(name, out, in_, banks):
    return struct(
        w = weight(name, [out, in_], banks),
        bias = weight(name + ".bias", [out], compute(banks)),
    )

def packed_linear(name, seams, in_, banks):
    out = sum(seams)
    return struct(
        w = weight(name, [out, in_], banks).packed(seams),
        bias = weight(name + ".bias", [out], compute(banks)).packed(seams),
    )

def sum(xs):
    out = 0
    for x in xs:
        out += x
    return out

def block(prefix, d, banks):
    dense = compute(banks)
    dim = d.dim
    n = lambda s: prefix + "." + s
    gain = lambda s: weight(n(s), [dim], dense)
    return struct(
        table = weight(n("table"), [MOD_SLICES * dim], dtype.f32),
        self_attn = struct(
            qkv = packed_linear(n("self.qkv"), [dim, dim, dim], dim, banks),
            norm_q = gain("self.norm_q"),
            norm_k = gain("self.norm_k"),
            out = linear(n("self.out"), dim, dim, banks),
        ),
        norm2 = gain("norm2"),
        norm2_bias = gain("norm2.bias"),
        cross = struct(
            q = linear(n("cross.q"), dim, dim, banks),
            kv = packed_linear(n("cross.kv"), [dim, dim], dim, banks),
            norm_q = gain("cross.norm_q"),
            norm_k = gain("cross.norm_k"),
            out = linear(n("cross.out"), dim, dim, banks),
        ),
        ffn = struct(
            up = linear(n("ffn.up"), d.ffn, dim, banks),
            down = linear(n("ffn.down"), dim, d.ffn, banks),
        ),
    )

def embedder(prefix, in_, dim, banks):
    return struct(
        linear_1 = linear(prefix + ".1", dim, in_, banks),
        linear_2 = linear(prefix + ".2", dim, dim, banks),
    )

def text_encoder(banks):
    dense = compute(banks)
    inner = TE_HEADS * TE_HEAD_DIM

    def layer(l):
        n = lambda s: "te.layer.{}.{}".format(l, s)
        return struct(
            attn_norm = weight(n("attn_norm"), [TE_HIDDEN], dense),
            q = weight(n("q"), [inner, TE_HIDDEN], banks),
            k = weight(n("k"), [inner, TE_HIDDEN], banks),
            v = weight(n("v"), [inner, TE_HIDDEN], banks),
            o = weight(n("o"), [TE_HIDDEN, inner], banks),
            rel_bias = weight(n("rel_bias"), [TE_BUCKETS, TE_HEADS], dense),
            ffn_norm = weight(n("ffn_norm"), [TE_HIDDEN], dense),
            wi_0 = weight(n("wi_0"), [TE_INTER, TE_HIDDEN], banks),
            wi_1 = weight(n("wi_1"), [TE_INTER, TE_HIDDEN], banks),
            wo = weight(n("wo"), [TE_HIDDEN, TE_INTER], banks),
        )

    return struct(
        embed = weight("te.embed", [TE_VOCAB, TE_HIDDEN], banks),
        layers = [layer(l) for l in range(TE_LAYERS)],
        final_norm = weight("te.final_norm", [TE_HIDDEN], dense),
    )

def conv(name, c_out, c_in, k, plane, banks, front = None):
    taps = k[0] * k[1] * k[2]
    return struct(
        w = weight(name, [c_out, c_in * taps], banks).conv_taps_major(c_in, taps),
        bias = weight(name + ".bias", [c_out], dtype.f32),
        c_in = c_in,
        c_out = c_out,
        k = k,
        front = front if front != None else max(k[0] - 1, 0),
        cache = name + ".frames" if k[0] > 1 else None,
        plane = plane,
    )

def resnet(prefix, c_in, c_out, plane, banks):
    dense = compute(banks)
    return struct(
        norm1 = weight(prefix + ".norm1", [c_in], dense),
        conv1 = conv(prefix + ".conv1", c_out, c_in, [3, 3, 3], plane, banks),
        norm2 = weight(prefix + ".norm2", [c_out], dense),
        conv2 = conv(prefix + ".conv2", c_out, c_out, [3, 3, 3], plane, banks),
        shortcut = conv(prefix + ".shortcut", c_out, c_in, [1, 1, 1], plane, banks) if c_in != c_out else None,
    )

def mid_attention(prefix, top, banks):
    return struct(
        norm = weight(prefix + ".norm", [top], compute(banks)),
        qkv = linear(prefix + ".qkv", 3 * top, top, banks),
        proj = linear(prefix + ".proj", top, top, banks),
    )

def vae_encoder(banks):
    dense = compute(banks)
    dims = VAE_ENCODER_DIMS
    p0 = VAE_MAX_LATENT_PLANE
    plane = 64 * p0
    down = []
    for i in range(4):
        c_in, c_out = dims[i], dims[i + 1]
        prefix = "vae.enc.down.{}".format(i)
        resnets = [
            resnet("{}.res.{}".format(prefix, r), c_in if r == 0 else c_out, c_out, plane, banks)
            for r in range(VAE_ENC_RESNETS)
        ]
        down_flag = i != 3
        temporal = VAE_TEMPORAL_DOWN[i]
        downsampler = None
        if down_flag:
            downsampler = struct(
                resample = conv(prefix + ".resample", c_out, c_out, [1, 3, 3], plane, banks),
                time_conv = conv(prefix + ".time_conv", c_out, c_out, [3, 1, 1], plane // 4, banks, front = 1) if temporal else None,
            )
        factor_t = 2 if temporal else 1
        factor_s = 2 if down_flag else 1
        volume = factor_t * factor_s * factor_s
        if c_in * volume % c_out != 0:
            fail("an AvgDown3D's widened channels must fold into whole groups")
        down.append(struct(
            resnets = resnets,
            downsampler = downsampler,
            shortcut = struct(factor = [factor_t, factor_s, factor_s], group = c_in * volume // c_out),
        ))
        if down_flag:
            plane //= 4
    top = dims[4]
    return struct(
        conv_in = conv("vae.enc.conv_in", dims[0], VAE_PIX_CHANNELS, [3, 3, 3], 64 * p0, banks),
        down = down,
        mid_res0 = resnet("vae.enc.mid.res.0", top, top, plane, banks),
        mid_attn = mid_attention("vae.enc.mid.attn", top, banks),
        mid_res1 = resnet("vae.enc.mid.res.1", top, top, plane, banks),
        norm_out = weight("vae.enc.norm_out", [top], dense),
        conv_out = conv("vae.enc.conv_out", 2 * VAE_Z, top, [3, 3, 3], plane, banks),
        quant = conv("vae.enc.quant", VAE_Z, 2 * VAE_Z, [1, 1, 1], plane, banks),
        norm_bias = weight("vae.enc.norm_bias", [VAE_Z], dense),
        norm_scale = weight("vae.enc.norm_scale", [VAE_Z], dense),
    )

def vae(banks):
    dense = compute(banks)
    dims = VAE_DECODER_DIMS
    top = dims[0]
    p0 = VAE_MAX_LATENT_PLANE
    plane = p0
    up = []
    for i in range(4):
        c_in, c_out = dims[i], dims[i + 1]
        prefix = "vae.up.{}".format(i)
        resnets = [
            resnet("{}.res.{}".format(prefix, r), c_in if r == 0 else c_out, c_out, plane, banks)
            for r in range(VAE_RESNETS)
        ]
        up_flag = i != 3
        upsampler = None
        shortcut = None
        if up_flag:
            upsampler = struct(
                time_conv = conv(prefix + ".time_conv", 2 * c_out, c_out, [3, 1, 1], plane, banks) if VAE_TEMPORAL_UP[i] else None,
                resample = conv(prefix + ".resample", c_out, c_out, [1, 3, 3], 4 * plane, banks),
            )
            if VAE_TEMPORAL_UP[i]:
                if c_in != c_out:
                    fail("a (2, 2, 2) DupUp3D keeps the width")
                shortcut = "nearest222"
            else:
                if c_in != 2 * c_out:
                    fail("a (1, 2, 2) DupUp3D halves the width")
                shortcut = "shuffle_h"
        up.append(struct(resnets = resnets, upsampler = upsampler, shortcut = shortcut))
        if up_flag:
            plane *= 4
    last = dims[4]
    return struct(
        denorm_bias = weight("vae.denorm_bias", [VAE_Z], dense),
        denorm_scale = weight("vae.denorm_scale", [VAE_Z], dense),
        post_quant = conv("vae.post_quant", VAE_Z, VAE_Z, [1, 1, 1], p0, banks),
        conv_in = conv("vae.conv_in", top, VAE_Z, [3, 3, 3], p0, banks),
        mid_res0 = resnet("vae.mid.res.0", top, top, p0, banks),
        mid_attn = mid_attention("vae.mid.attn", top, banks),
        mid_res1 = resnet("vae.mid.res.1", top, top, p0, banks),
        up = up,
        norm_out = weight("vae.norm_out", [last], dense),
        conv_out = conv("vae.conv_out", VAE_PIX_CHANNELS, last, [3, 3, 3], plane, banks),
        enc = vae_encoder(banks),
    )

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    banks = deploy.weights[0]
    d = DIMS[id]
    whole = id == "wan22-ti2v-5b"
    if d.heads * d.head_dim != d.dim:
        fail("plain MHA: heads × head_dim is the width")
    dim = d.dim
    return struct(
        dims = d,
        dit = struct(
            patch_embed = linear("dit.patch_embed", dim, d.patch_in, banks),
            text_embed = embedder("dit.text_embed", d.text_dim, dim, banks),
            time_embed = embedder("dit.time_embed", d.freq_dim, dim, banks),
            time_proj = linear("dit.time_proj", MOD_SLICES * dim, dim, banks),
            head_proj = linear("dit.head_proj", HEAD_SLICES * dim, dim, banks),
            blocks = [block("dit.block.{}".format(i), d, banks) for i in range(d.layers)],
            head_table = weight("dit.head_table", [HEAD_SLICES * dim], dtype.f32),
            proj_out = linear("dit.proj_out", d.patch_out, dim, banks),
        ),
        te = text_encoder(banks) if whole else None,
        vae = vae(dtype.bf16) if whole else None,
        shift = SHIFT_TI2V,
    )

def generative(m):
    d = m.dims
    index = 0
    readings = []
    if m.te != None:
        readings.append(reading(
            "text",
            index = index,
            takes_tokens = True,
            streams = ["text"],
            readout = "hidden",
            readout_width = TE_HIDDEN,
        ))
        index += 1
    readings.append(reading(
        "denoise",
        index = index,
        streams = ["video", "context"],
        ports = [
            port("latents", "latents", d.patch_in, ["video"]),
            port("context", "context", d.text_dim, ["context"], rows = CONTEXT_LEN if m.te != None else None),
            port("timestep", "lane_vector", 1, ["video"]),
            port("positions", "axis_positions", ROPE_AXES, ["video"]),
        ],
        positions = positions(
            axes = ["time", "height", "width"],
            text_axis = 0,
            text_origin = 0,
            image_follows_text = False,
        ),
        readout = "velocity",
        readout_width = d.patch_out,
    ))
    index += 1
    if m.vae != None:
        for name in ["vae.decode.head", "vae.decode"]:
            readings.append(reading(
                name,
                index = index,
                streams = ["video"],
                ports = [port("latent", "voxels", VAE_Z, ["video"])],
                readout = "pixels",
                readout_width = VAE_RGB,
            ))
            index += 1
        for name in ["vae.encode.head", "vae.encode"]:
            readings.append(reading(
                name,
                index = index,
                streams = ["video"],
                ports = [port("pixels", "voxels", VAE_RGB, ["video"], at = 1)],
                readout = "pixels",
                readout_width = VAE_Z,
            ))
            index += 1
    return generation(
        readings = readings,
        latent = latent_space(
            channels = d.in_channels,
            patch_t = PATCH_T,
            patch_h = PATCH_H,
            patch_w = PATCH_W,
            spatial_compression = VAE_SPATIAL_COMPRESSION,
            temporal_compression = VAE_TEMPORAL_COMPRESSION,
        ),
        schedule = schedule("flow", shift = m.shift, train_steps = TRAIN_STEPS),
        max_rows = 32768 + CONTEXT_LEN if m.te != None else 4096,
    )
