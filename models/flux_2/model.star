# The weights of FLUX.2: double-stream blocks over text and image, then
# single-stream blocks over both; a Qwen3 4B text encoder whose three tapped
# layers project into the context; and a VAE with a batch-normed latent.

load("//lib/flux_vae/model.star", flux_vae = "vae", vae_conv = "conv")
load("//lib/qwen3_text/model.star", "QWEN3_4B", "encoder")

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
RGB = 3
TE_TAPS = [9, 18, 27]
TE_MAX_TOKENS = 512

DIMS = {
    "flux2-klein-4b": struct(
        dim = 3072,
        heads = 24,
        inter = 3072 * MLP_RATIO,
        context_in = len(TE_TAPS) * QWEN3_4B.hidden,
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

def vae_latent():
    bn = lambda tail: weight("vae.bn." + tail, [IN_CHANNELS], compute(dtype.bf16))
    return struct(
        bn_zero = bn("zero"),
        bn_scale = bn("scale"),
        bn_rscale = bn("rscale"),
        bn_mean = bn("mean"),
        quant_conv = vae_conv("vae.quant", VAE_CHANNELS, 2 * VAE_CHANNELS, 1),
        post_quant_conv = vae_conv("vae.post_quant", VAE_CHANNELS, VAE_CHANNELS, 1),
    )

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    banks = deploy.weights[0]
    d = DIMS[id]
    if d.heads * HEAD_DIM != d.dim:
        fail("plain MHA over 128-wide heads: heads × 128 is the width")
    dim = d.dim
    klein = id == "flux2-klein-4b"
    te = encoder(QWEN3_4B, TE_TAPS[-1], banks) if klein else None
    te_context = [
        weight("dit.context_embed.{}".format(i), [dim, QWEN3_4B.hidden], banks)
        for i in range(len(TE_TAPS))
    ] if klein else None
    return struct(
        dims = d,
        kv = dtype.bf16,
        in_channels = IN_CHANNELS,
        pack = PACK,
        head_dim = HEAD_DIM,
        # 1/sqrt(128) as the reference rounds it, an ulp above the f32 quotient.
        sm_scale = 0.08838835,
        rope_axes = ROPE_AXES,
        t_freq_dim = T_FREQ_DIM,
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
        te_taps = TE_TAPS,
        te_context = te_context,
        vae = flux_vae(VAE_CHANNELS, 2 * VAE_CHANNELS, vae_latent) if klein else None,
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
            readout_width = d.dim,
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
        max_rows = 5 * 4096 + TE_MAX_TOKENS if m.te != None else 4096,
    )
