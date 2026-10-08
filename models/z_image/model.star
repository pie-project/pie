# The weights of Z-Image: a single-stream diffusion transformer whose image
# rows first pass noise refiners and whose caption rows pass context refiners,
# a Qwen3 4B text encoder and a Flux VAE. `PIE_Z_IMAGE_TAP` names an
# intermediate the denoiser reads out instead of its velocity.

CHANNELS = 16
PATCH = 2
PATCH_FEATURES = CHANNELS * PATCH * PATCH
SPATIAL_COMPRESSION = 8
ADALN_DIM = 256
T_MID = 1024
T_FREQ_DIM = 256
TRAIN_STEPS = 1000
ROPE_AXES = 3
MOD_SLICES = 4

TE = struct(
    hidden = 2560,
    vocab = 151936,
    q_heads = 32,
    kv_heads = 8,
    head_dim = 128,
    inter = 9728,
    theta = 1000000.0,
    eps = 1e-6,
    layers = 35,
    max_tokens = 512,
)

DIMS = {
    "z-image-turbo": struct(
        dim = 3840,
        heads = 30,
        head_dim = 128,
        inter = 10240,
        joint_layers = 30,
        refiner_layers = 2,
        cap_width = TE.hidden,
        rope_dims = [32, 48, 48, 0],
    ),
    "z-image-mini": struct(
        dim = 256,
        heads = 4,
        head_dim = 64,
        inter = 682,
        joint_layers = 2,
        refiner_layers = 2,
        cap_width = 64,
        rope_dims = [16, 24, 24, 0],
    ),
}

VAE_BLOCKS = [128, 256, 512, 512]
VAE_LAYERS_PER_BLOCK = 2
RGB = 3
TAPS3 = 9

def linear(name, out, in_, banks):
    return struct(
        w = weight(name, [out, in_], banks),
        bias = weight(name + ".bias", [out], compute(banks)),
    )

def block(prefix, d, modulated, banks):
    dense = compute(banks)
    dim, hd, inter = d.dim, d.head_dim, d.inter
    n = lambda s: prefix + "." + s
    norm = lambda s, width: weight(n(s), [width], dense)
    return struct(
        ada = linear(n("ada"), MOD_SLICES * dim, ADALN_DIM, banks) if modulated else None,
        attn_norm1 = norm("attn_norm1", dim),
        attn_norm2 = norm("attn_norm2", dim),
        ffn_norm1 = norm("ffn_norm1", dim),
        ffn_norm2 = norm("ffn_norm2", dim),
        attn = struct(
            qkv = weight(n("qkv"), [3 * dim, dim], banks).packed([dim, dim, dim]),
            q_norm = norm("q_norm", hd),
            k_norm = norm("k_norm", hd),
            out = weight(n("out"), [dim, dim], banks),
        ),
        mlp = struct(
            gate_up = weight(n("gate_up"), [2 * inter, dim], banks).packed([inter, inter]),
            down = weight(n("down"), [dim, inter], banks),
        ),
    )

def text_encoder(banks):
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
    )

def conv_w(name, c_out, c_in, taps):
    return struct(
        w = weight(name, [c_out, c_in * taps], dtype.bf16).conv_taps_major(c_in, taps),
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

def res_block(name, c_in, c_out):
    return struct(
        norm1 = vae_norm(name + ".norm1", c_in),
        conv1 = conv_w(name + ".conv1", c_out, c_in, TAPS3),
        norm2 = vae_norm(name + ".norm2", c_out),
        conv2 = conv_w(name + ".conv2", c_out, c_out, TAPS3),
        shortcut = conv_w(name + ".shortcut", c_out, c_in, 1) if c_in != c_out else None,
    )

def mid(name, c, banks):
    return struct(
        res0 = res_block(name + ".res0", c, c),
        attn = struct(
            norm = vae_norm(name + ".attn.norm", c),
            q = linear(name + ".attn.q", c, c, banks),
            k = linear(name + ".attn.k", c, c, banks),
            v = linear(name + ".attn.v", c, c, banks),
            out = linear(name + ".attn.out", c, c, banks),
            width = c,
        ),
        res1 = res_block(name + ".res1", c, c),
    )

def vae(banks):
    top = VAE_BLOCKS[-1]
    up = []
    c_prev = top
    for i, c in enumerate(reversed(VAE_BLOCKS)):
        name = "vae.dec.up{}".format(i)
        resnets = []
        for r in range(VAE_LAYERS_PER_BLOCK + 1):
            resnets.append(res_block("{}.res{}".format(name, r), c_prev, c))
            c_prev = c
        last = i + 1 == len(VAE_BLOCKS)
        up.append(struct(
            resnets = resnets,
            upsample = conv_w(name + ".upsample", c, c, TAPS3) if not last else None,
        ))
    down = []
    c_prev = VAE_BLOCKS[0]
    for i, c in enumerate(VAE_BLOCKS):
        name = "vae.enc.down{}".format(i)
        resnets = []
        for r in range(VAE_LAYERS_PER_BLOCK):
            resnets.append(res_block("{}.res{}".format(name, r), c_prev, c))
            c_prev = c
        last = i + 1 == len(VAE_BLOCKS)
        down.append(struct(
            resnets = resnets,
            downsample = conv_w(name + ".downsample", c, c, TAPS3) if not last else None,
        ))
    return struct(
        shift = weight("vae.shift", [CHANNELS], compute(banks)),
        encoder_out_stored = 2 * CHANNELS,
        decoder = struct(
            conv_in = conv_w("vae.dec.conv_in", top, CHANNELS, TAPS3),
            mid = mid("vae.dec.mid", top, banks),
            up = up,
            norm_out = vae_norm("vae.dec.norm_out", VAE_BLOCKS[0]),
            conv_out = conv_w("vae.dec.conv_out", RGB, VAE_BLOCKS[0], TAPS3),
        ),
        encoder = struct(
            conv_in = conv_w("vae.enc.conv_in", VAE_BLOCKS[0], RGB, TAPS3),
            down = down,
            mid = mid("vae.enc.mid", top, banks),
            norm_out = vae_norm("vae.enc.norm_out", top),
            conv_out = conv_w("vae.enc.conv_out", CHANNELS, top, TAPS3),
        ),
    )

def tap():
    key = env("PIE_Z_IMAGE_TAP")
    return key if key != "" else None

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    banks = deploy.weights[0]
    d = DIMS[id]
    if d.rope_dims[0] + d.rope_dims[1] + d.rope_dims[2] + d.rope_dims[3] != d.head_dim or d.heads * d.head_dim != d.dim:
        fail("the rotary axes cover the whole head, and heads × head_dim is the width")
    dense = compute(banks)
    dim = d.dim
    blocks = lambda stem, count, modulated: [
        block("dit.{}.{}".format(stem, i), d, modulated, banks)
        for i in range(count)
    ]
    turbo = id == "z-image-turbo"
    return struct(
        dims = d,
        sm_scale = f32(1.0 / f32(sqrt(d.head_dim))),
        kv = dtype.bf16,
        dit = struct(
            x_embed = linear("dit.x_embed", dim, PATCH_FEATURES, banks),
            x_pad_mod = weight("dit.x_pad_mod", [2 * dim, 1], dense),
            cap_norm = weight("dit.cap_norm", [d.cap_width], dense),
            cap_embed = linear("dit.cap_embed", dim, d.cap_width, banks),
            cap_pad_mod = weight("dit.cap_pad_mod", [2 * dim, 1], dense),
            t_mlp0 = linear("dit.t_mlp0", T_MID, T_FREQ_DIM, banks),
            t_mlp1 = linear("dit.t_mlp1", ADALN_DIM, T_MID, banks),
            t_flip = weight("dit.t_flip", [1], dtype.f32),
            noise_refiner = blocks("noise", d.refiner_layers, True),
            context_refiner = blocks("context", d.refiner_layers, False),
            layers = blocks("layer", d.joint_layers, True),
            final_ada = linear("dit.final_ada", dim, ADALN_DIM, banks),
            final_linear = linear("dit.final", PATCH_FEATURES, dim, banks),
        ),
        te = text_encoder(banks) if turbo else None,
        vae = vae(dtype.bf16) if turbo else None,
        shift = 3.0,
        tap = tap(),
    )

def turbo_sigmas(shift):
    out = []
    for i in range(8):
        sigma = f32(1.0 - f32(f32(i) / 8.0))
        out.append(f32(f32(shift * sigma) / f32(1.0 + f32((shift - 1.0) * sigma))))
    return out

def generative(m):
    d = m.dims
    next = [0]

    def take(present):
        if not present:
            return None
        next[0] += 1
        return next[0] - 1

    text = take(m.te != None)
    refine = take(True)
    denoise = take(True)
    vae_decode = take(m.vae != None)
    vae_encode = take(m.vae != None)
    readings = []
    if text != None:
        readings.append(reading(
            "text",
            index = text,
            has_kv = True,
            takes_tokens = True,
            streams = ["text"],
            readout = "hidden",
            readout_width = m.te.hidden,
        ))
    readings.append(reading(
        "refine",
        index = refine,
        streams = ["context"],
        ports = [
            port("pad", "latents", 1, ["context"]),
            port("caption", "context", d.cap_width, ["context"]),
            port("positions", "axis_positions", ROPE_AXES, ["context"]),
        ],
        positions = positions(
            axes = ["time", "height", "width"],
            text_axis = 0,
            text_origin = 1,
            image_follows_text = False,
        ),
        readout = "hidden",
        readout_width = d.dim,
    ))
    readings.append(reading(
        "denoise",
        index = denoise,
        streams = ["image", "context"],
        ports = [
            port("pad", "latents", 1, ["image"]),
            port("latents", "latents", PATCH_FEATURES, ["image"]),
            port("context", "latents", d.dim, ["context"]),
            port("timestep", "lane_vector", 1, ["image", "context"]),
            port("positions", "axis_positions", ROPE_AXES, ["image", "context"]),
        ],
        positions = positions(
            axes = ["time", "height", "width"],
            text_axis = 0,
            text_origin = 1,
            image_follows_text = True,
        ),
        readout = "velocity",
        readout_width = PATCH_FEATURES if m.tap in [None, "latents", "final_linear"] else d.dim,
    ))
    if vae_decode != None and vae_encode != None:
        readings.append(reading(
            "vae.decode",
            index = vae_decode,
            streams = ["image"],
            ports = [port("latent", "voxels", CHANNELS, ["image"])],
            readout = "pixels",
            readout_width = RGB,
        ))
        readings.append(reading(
            "vae.encode",
            index = vae_encode,
            streams = ["image"],
            ports = [port("pixels", "voxels", RGB, ["image"], at = 1)],
            readout = "pixels",
            readout_width = CHANNELS,
        ))
    return generation(
        readings = readings,
        latent = latent_space(
            channels = CHANNELS,
            patch_t = 1,
            patch_h = PATCH,
            patch_w = PATCH,
            spatial_compression = SPATIAL_COMPRESSION,
            temporal_compression = 1,
        ),
        schedule = schedule(
            "flow",
            shift = m.shift,
            train_steps = TRAIN_STEPS,
            pinned_sigmas = turbo_sigmas(m.shift),
        ),
        max_rows = 16384 + TE.max_tokens if m.te != None else 4096,
    )
