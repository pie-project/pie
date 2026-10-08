# The weights of FLUX.2: double-stream blocks over text and image, then
# single-stream blocks over both; a Qwen3 4B text encoder whose three tapped
# layers project into the context; and a VAE with a batch-normed latent.
# `PIE_FLUX2_TE_TAP=<n>` reads the encoder's residual stream after block `n`
# out instead of its projected taps.

IN_CHANNELS = 128
VAE_CHANNELS = 32
PACK = 2
TOKEN_COMPRESSION = 8 * PACK
HEAD_DIM = 128
ROPE_AXES = 4
REFERENCE_TIME_STRIDE = 10
T_FREQ_DIM = 256
DOUBLE_MOD_SLICES = 6
SINGLE_MOD_SLICES = 3
MLP_RATIO = 3
TRAIN_STEPS = 1000

TE = struct(
    hidden = 2560,
    vocab = 151936,
    q_heads = 32,
    kv_heads = 8,
    head_dim = 128,
    inter = 9728,
    theta = 1000000.0,
    eps = 1e-6,
    layers = 27,
    max_tokens = 512,
)

DIMS = {
    "flux2-klein-4b": struct(
        dim = 3072,
        heads = 24,
        inter = 3072 * MLP_RATIO,
        context_in = 3 * TE.hidden,
        double_blocks = 5,
        single_blocks = 20,
        guidance_embeds = False,
    ),
    "flux2-mini": struct(
        dim = 256,
        heads = 2,
        inter = 256 * MLP_RATIO,
        context_in = 192,
        double_blocks = 2,
        single_blocks = 2,
        guidance_embeds = True,
    ),
}

VAE_BLOCKS = [128, 256, 512, 512]
VAE_LAYERS_PER_BLOCK = 2
RGB = 3
TAPS3 = 9

def attn(prefix, d, banks):
    dense = compute(banks)
    dim = d.dim
    return struct(
        qkv = weight(prefix + ".qkv", [3 * dim, dim], banks).packed([dim, dim, dim]),
        q_norm = weight(prefix + ".q_norm", [HEAD_DIM], dense),
        k_norm = weight(prefix + ".k_norm", [HEAD_DIM], dense),
        out = weight(prefix + ".out", [dim, dim], banks),
    )

def swiglu(prefix, d, banks):
    return struct(
        linear_in = weight(prefix + ".in", [2 * d.inter, d.dim], banks).packed([d.inter, d.inter]),
        linear_out = weight(prefix + ".out", [d.dim, d.inter], banks),
    )

def side(prefix, d, banks):
    return struct(attn = attn(prefix + ".attn", d, banks), ff = swiglu(prefix + ".ff", d, banks))

def single(prefix, d, banks):
    dense = compute(banks)
    dim, inter = d.dim, d.inter
    return struct(
        in_proj = weight(prefix + ".in", [3 * dim + 2 * inter, dim], banks).packed([dim, dim, dim, inter, inter]),
        q_norm = weight(prefix + ".q_norm", [HEAD_DIM], dense),
        k_norm = weight(prefix + ".k_norm", [HEAD_DIM], dense),
        out_attn = weight(prefix + ".out_attn", [dim, dim], banks),
        out_mlp = weight(prefix + ".out_mlp", [dim, inter], banks),
    )

def embedder(prefix, d, banks):
    return struct(
        linear_1 = weight(prefix + ".1", [d.dim, T_FREQ_DIM], banks),
        linear_2 = weight(prefix + ".2", [d.dim, d.dim], banks),
    )

def text_encoder(d, banks):
    dense = compute(banks)
    hidden, hd, inter = TE.hidden, TE.head_dim, TE.inter

    def layer(l):
        n = lambda s: "te.layer.{}.{}".format(l, s)
        return struct(
            attn_norm = weight(n("attn_norm"), [hidden], dense),
            q = weight(n("q"), [TE.q_heads * hd, hidden], banks),
            k = weight(n("k"), [TE.kv_heads * hd, hidden], banks),
            v = weight(n("v"), [TE.kv_heads * hd, hidden], banks),
            o = weight(n("o"), [hidden, TE.q_heads * hd], banks),
            q_norm = weight(n("q_norm"), [hd], dense),
            k_norm = weight(n("k_norm"), [hd], dense),
            mlp_norm = weight(n("mlp_norm"), [hidden], dense),
            gate_up = weight(n("gate_up"), [2 * inter, hidden], banks).packed([inter, inter]),
            down = weight(n("down"), [hidden, inter], banks),
            kv = "te.kv.{}".format(l),
        )

    return struct(
        hidden = hidden,
        vocab = TE.vocab,
        q_heads = TE.q_heads,
        kv_heads = TE.kv_heads,
        head_dim = hd,
        inter = inter,
        theta = TE.theta,
        eps = TE.eps,
        sm_scale = f32(1.0 / f32(sqrt(hd))),
        embed = weight("te.embed", [TE.vocab, hidden], banks),
        layers = [layer(l) for l in range(TE.layers)],
        context_embed = [weight("dit.context_embed.{}".format(i), [d.dim, hidden], banks) for i in range(3)],
    )

def conv_w(name, c_out, c_in, taps, banks):
    return struct(
        w = weight(name, [c_out, c_in * taps], banks).conv_taps_major(c_in, taps),
        bias = weight(name + ".bias", [c_out], dtype.f32),
        c_in = c_in,
        c_out = c_out,
        taps = taps,
    )

def vae_norm(name, c):
    return struct(
        weight = weight(name + ".weight", [c], dtype.f32),
        bias = weight(name + ".bias", [c], dtype.f32),
    )

def res_block(name, c_in, c_out, banks):
    return struct(
        norm1 = vae_norm(name + ".norm1", c_in),
        conv1 = conv_w(name + ".conv1", c_out, c_in, TAPS3, banks),
        norm2 = vae_norm(name + ".norm2", c_out),
        conv2 = conv_w(name + ".conv2", c_out, c_out, TAPS3, banks),
        shortcut = conv_w(name + ".shortcut", c_out, c_in, 1, banks) if c_in != c_out else None,
    )

def proj(name, c, banks, dense):
    return struct(w = weight(name, [c, c], banks), bias = weight(name + ".bias", [c], dense))

def mid(name, c, banks, dense):
    return struct(
        res0 = res_block(name + ".res0", c, c, banks),
        attn = struct(
            norm = vae_norm(name + ".attn.norm", c),
            q = proj(name + ".attn.q", c, banks, dense),
            k = proj(name + ".attn.k", c, banks, dense),
            v = proj(name + ".attn.v", c, banks, dense),
            out = proj(name + ".attn.out", c, banks, dense),
            width = c,
        ),
        res1 = res_block(name + ".res1", c, c, banks),
    )

def vae(banks):
    dense = compute(banks)
    top = VAE_BLOCKS[-1]
    up = []
    c_prev = top
    for i, c in enumerate(reversed(VAE_BLOCKS)):
        name = "vae.dec.up{}".format(i)
        resnets = []
        for r in range(VAE_LAYERS_PER_BLOCK + 1):
            resnets.append(res_block("{}.res{}".format(name, r), c_prev, c, banks))
            c_prev = c
        last = i + 1 == len(VAE_BLOCKS)
        up.append(struct(
            resnets = resnets,
            upsample = conv_w(name + ".upsample", c, c, TAPS3, banks) if not last else None,
        ))
    down = []
    c_prev = VAE_BLOCKS[0]
    for i, c in enumerate(VAE_BLOCKS):
        name = "vae.enc.down{}".format(i)
        resnets = []
        for r in range(VAE_LAYERS_PER_BLOCK):
            resnets.append(res_block("{}.res{}".format(name, r), c_prev, c, banks))
            c_prev = c
        last = i + 1 == len(VAE_BLOCKS)
        down.append(struct(
            resnets = resnets,
            downsample = conv_w(name + ".downsample", c, c, TAPS3, banks) if not last else None,
        ))
    bn = lambda tail: weight("vae.bn." + tail, [IN_CHANNELS], dense)
    return struct(
        bn_zero = bn("zero"),
        bn_scale = bn("scale"),
        bn_rscale = bn("rscale"),
        bn_mean = bn("mean"),
        quant_conv = conv_w("vae.quant", VAE_CHANNELS, 2 * VAE_CHANNELS, 1, banks),
        post_quant_conv = conv_w("vae.post_quant", VAE_CHANNELS, VAE_CHANNELS, 1, banks),
        decoder = struct(
            conv_in = conv_w("vae.dec.conv_in", top, VAE_CHANNELS, TAPS3, banks),
            mid = mid("vae.dec.mid", top, banks, dense),
            up = up,
            norm_out = vae_norm("vae.dec.norm_out", VAE_BLOCKS[0]),
            conv_out = conv_w("vae.dec.conv_out", RGB, VAE_BLOCKS[0], TAPS3, banks),
        ),
        encoder = struct(
            conv_in = conv_w("vae.enc.conv_in", VAE_BLOCKS[0], RGB, TAPS3, banks),
            down = down,
            mid = mid("vae.enc.mid", top, banks, dense),
            norm_out = vae_norm("vae.enc.norm_out", top),
            conv_out = conv_w("vae.enc.conv_out", 2 * VAE_CHANNELS, top, TAPS3, banks),
        ),
    )

def te_tap():
    key = env("PIE_FLUX2_TE_TAP")
    if key == None or not key.isdigit():
        return None
    return int(key)

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    banks = deploy.weights[0]
    d = DIMS[id]
    if d.heads * HEAD_DIM != d.dim:
        fail("plain MHA over 128-wide heads: heads × 128 is the width")
    dim = d.dim
    klein = id == "flux2-klein-4b"
    te = text_encoder(d, banks) if klein else None
    return struct(
        dims = d,
        kv = dtype.bf16,
        dit = struct(
            x_embed = weight("dit.x_embed", [dim, IN_CHANNELS], banks),
            context_embed = weight("dit.context_embed", [dim, d.context_in], banks) if te == None else None,
            t_embed = embedder("dit.t_embed", d, banks),
            g_embed = embedder("dit.g_embed", d, banks) if d.guidance_embeds else None,
            mod_img = weight("dit.mod_img", [DOUBLE_MOD_SLICES * dim, dim], banks),
            mod_txt = weight("dit.mod_txt", [DOUBLE_MOD_SLICES * dim, dim], banks),
            mod_single = weight("dit.mod_single", [SINGLE_MOD_SLICES * dim, dim], banks),
            double = [
                struct(
                    img = side("dit.double.{}.img".format(i), d, banks),
                    txt = side("dit.double.{}.txt".format(i), d, banks),
                )
                for i in range(d.double_blocks)
            ],
            single = [single("dit.single.{}".format(i), d, banks) for i in range(d.single_blocks)],
            norm_out = weight("dit.norm_out", [2 * dim, dim], banks),
            proj_out = weight("dit.proj_out", [IN_CHANNELS, dim], banks),
        ),
        te = te,
        vae = vae(dtype.bf16) if klein else None,
        te_tap = te_tap(),
    )

A1 = 8.73809524e-5
B1 = 1.89833333
A2 = 0.00016927
B2 = 0.45666666

def empirical_mu(image_rows, steps):
    """The schedule's shift, as an f32, for `image_rows` and `steps`."""
    rows = float(image_rows)
    if image_rows > 4300:
        return f32(A2 * rows + B2)
    m_200 = A2 * rows + B2
    m_10 = A1 * rows + B1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return f32(a * float(steps) + b)

def sigmas(image_rows, steps):
    steps = max(steps, 1)
    shift = exp(empirical_mu(image_rows, steps))
    n = float(steps)
    out = []
    for i in range(steps):
        sigma = 1.0 - float(i) * (1.0 - 1.0 / n) / max(n - 1.0, 1.0)
        out.append(f32(shift / (shift + 1.0 / sigma - 1.0)))
    return out

def generative(m):
    d = m.dims
    next = [0]

    def take():
        next[0] += 1
        return next[0] - 1

    text = take() if m.te != None else None
    denoise = take()
    vae_decode = take() if m.vae != None else None
    vae_encode = take() if m.vae != None else None
    readings = []
    if text != None:
        readings.append(reading(
            "text",
            index = text,
            has_kv = True,
            takes_tokens = True,
            streams = ["text"],
            readout = "hidden",
            readout_width = TE.hidden if m.te_tap != None else d.dim,
        ))
    image_side = ["image", "reference"]
    every = ["text", "image", "reference"]
    ports = [
        port("latents", "latents", IN_CHANNELS, image_side),
        port("context", "context", d.dim if m.te != None else d.context_in, ["text"]),
        port("timestep", "lane_vector", 1, every),
    ]
    if d.guidance_embeds:
        ports.append(port("guidance", "lane_vector", 1, every))
    ports.append(port("positions", "axis_positions", ROPE_AXES, every))
    readings.append(reading(
        "denoise",
        index = denoise,
        streams = every,
        ports = ports,
        positions = positions(
            axes = ["time", "height", "width", "index"],
            text_axis = 3,
            text_origin = 0,
            image_follows_text = False,
            reference_stride = REFERENCE_TIME_STRIDE,
        ),
        readout = "velocity",
        readout_width = IN_CHANNELS,
    ))
    if vae_decode != None and vae_encode != None:
        readings.append(reading(
            "vae.decode",
            index = vae_decode,
            streams = ["image"],
            ports = [port("latent", "voxels", IN_CHANNELS, ["image"])],
            readout = "pixels",
            readout_width = RGB,
        ))
        readings.append(reading(
            "vae.encode",
            index = vae_encode,
            streams = ["image"],
            ports = [port("pixels", "voxels", RGB, ["image"], at = 1)],
            readout = "pixels",
            readout_width = IN_CHANNELS,
        ))
    return generation(
        readings = readings,
        latent = latent_space(
            channels = IN_CHANNELS,
            patch_t = 1,
            patch_h = 1,
            patch_w = 1,
            spatial_compression = TOKEN_COMPRESSION,
            temporal_compression = 1,
        ),
        schedule = schedule(
            "flow",
            shift = expf(empirical_mu(4096, 4)),
            train_steps = TRAIN_STEPS,
            pinned_sigmas = sigmas(4096, 4),
        ),
        max_rows = 5 * 4096 + TE.max_tokens if m.te != None else 4096,
    )
