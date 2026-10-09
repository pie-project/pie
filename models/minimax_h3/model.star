# The weights of MiniMax H3: the denoiser's patch projections, timestep
# embedder, caption refiner, blocks (each with an adaLN projection per
# modality) and output heads; and the text encoder the full model carries.

load("//lib/diffusion/model.star", "linear")
load("//lib/qwen3_text/model.star", "encoder")

LATENT_CHANNELS = 24
AUDIO_CHANNELS = 32
PATCH_T = 1
PATCH_H = 2
PATCH_W = 2
VIDEO_FEATURES = LATENT_CHANNELS * PATCH_T * PATCH_H * PATCH_W
SPATIAL_COMPRESSION = 16
TEMPORAL_COMPRESSION = 4
ROPE_AXES = 3
ADALN_SLICES = 6
MODALITIES = 3
FINAL_SLICES = 2
TIMESTEP_SLOTS = 4
TRAIN_STEPS = 1
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
STEPS = 50

TE = struct(
    hidden = 5120,
    vocab = 151936,
    q_heads = 64,
    kv_heads = 8,
    head_dim = 128,
    inter = 25600,
    theta = 5000000.0,
    eps = 1e-6,
)
TE_LAYERS = 50

DIMS = {
    "minimax-h3-fl2va": struct(
        dim = 5376,
        heads = 56,
        head_dim = 128,
        inter = 14336,
        blocks = 50,
        refiners = 2,
        text_dim = TE.hidden,
        t_freq = 256,
        t_hidden = 5376,
        t_dim = 2688,
        rope_freqs = 16,
    ),
    "minimax-h3-mini": struct(
        dim = 128,
        heads = 2,
        head_dim = 64,
        inter = 256,
        blocks = 2,
        refiners = 1,
        text_dim = 64,
        t_freq = 32,
        t_hidden = 128,
        t_dim = 64,
        rope_freqs = 8,
    ),
}

def attn(prefix, d, banks):
    dense = compute(banks)
    inner = d.heads * d.head_dim
    return struct(
        qkv = weight(prefix + ".qkv", [3 * inner, d.dim], banks).packed([inner, inner, inner]),
        q_norm = weight(prefix + ".q_norm", [d.head_dim], dense),
        k_norm = weight(prefix + ".k_norm", [d.head_dim], dense),
        out = weight(prefix + ".out", [d.dim, inner], banks),
    )

def mlp(prefix, d, banks):
    return struct(
        fc1 = weight(prefix + ".fc1", [2 * d.inter, d.dim], banks).packed([d.inter, d.inter]),
        fc2 = weight(prefix + ".fc2", [d.dim, d.inter], banks),
    )

def block(prefix, d, banks, modulated):
    """A refiner block, or (`modulated`) a denoiser block with an adaLN
    projection per modality."""
    dense = compute(banks)
    return struct(
        norm1 = weight(prefix + ".norm1", [d.dim], dense),
        norm2 = weight(prefix + ".norm2", [d.dim], dense),
        attn = attn(prefix + ".attn", d, banks),
        mlp = mlp(prefix + ".mlp", d, banks),
        adaln = [
            linear("{}.adaln.{}".format(prefix, m), ADALN_SLICES * d.dim, d.t_dim, banks)
            for m in range(MODALITIES)
        ] if modulated else None,
    )

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    banks = deploy.weights[0]
    d = DIMS[id]
    rotary = 6 * d.rope_freqs
    if rotary % 2 != 0 or rotary > d.head_dim:
        fail("the three rotary axes cover {} of a {}-wide head".format(rotary, d.head_dim))
    dense = compute(banks)
    dim = d.dim
    return struct(
        kv = dtype.bf16,
        dims = d,
        sm_scale = f32(1.0 / f32(sqrt(d.head_dim))),
        rope_dims = [2 * d.rope_freqs, 2 * d.rope_freqs, 2 * d.rope_freqs, 0],
        rotary_dim = rotary,
        rope_axes = ROPE_AXES,
        timestep_slots = TIMESTEP_SLOTS,
        video_features = VIDEO_FEATURES,
        audio_channels = AUDIO_CHANNELS,
        dit = struct(
            video_patch = linear("dit.video_patch", dim, VIDEO_FEATURES, banks),
            audio_patch = linear("dit.audio_patch", dim, AUDIO_CHANNELS, banks),
            condition = linear("dit.condition", dim, d.text_dim, banks),
            t_in = linear("dit.t_in", d.t_hidden, d.t_freq, banks),
            t_out = linear("dit.t_out", d.t_dim, d.t_hidden, banks),
            refine = [block("dit.refine.{}".format(i), d, banks, False) for i in range(d.refiners)],
            refine_norm = weight("dit.refine_norm", [dim], dense),
            blocks = [block("dit.block.{}".format(i), d, banks, True) for i in range(d.blocks)],
            final_norm = weight("dit.final_norm", [dim], dense),
            final_adaln = linear("dit.final_adaln", FINAL_SLICES * dim, d.t_dim, banks),
            video_out = linear("dit.video_out", VIDEO_FEATURES, dim, banks),
            audio_out = linear("dit.audio_out", AUDIO_CHANNELS, dim, banks),
        ),
        te = encoder(TE, TE_LAYERS, banks, sharded = True) if id == "minimax-h3-fl2va" else None,
    )

def shifted_sigmas(shift, steps):
    n = max(steps, 2)
    out = []
    for i in range(n - 1):
        base = 1.0 - float(i) / float(n - 1)
        out.append(shift * base / (1.0 + (shift - 1.0) * base))
    return out

def generative(m):
    d = m.dims
    readings = []
    index = 0
    if m.te != None:
        readings.append(reading(
            "text",
            index = index,
            has_kv = True,
            takes_tokens = True,
            streams = ["text"],
            readout = "hidden",
            readout_width = m.te.hidden,
        ))
        index += 1
    readings.append(reading(
        "refine",
        index = index,
        streams = ["text"],
        ports = [port("caption", "context", d.text_dim, ["text"], at = 0)],
        readout = "hidden",
        readout_width = d.dim,
    ))
    every = ["text", "video", "audio", "reference"]
    readings.append(reading(
        "denoise",
        index = index + 1,
        streams = every,
        ports = [
            port("latents", "latents", VIDEO_FEATURES, ["video"], at = 0),
            port("reference", "latents", VIDEO_FEATURES, ["reference"], at = 1),
            port("audio", "latents", AUDIO_CHANNELS, ["audio"], at = 2),
            port("context", "latents", d.dim, ["text"], at = 3),
            port("timestep", "lane_vector", TIMESTEP_SLOTS, every, at = 0),
            port("positions", "axis_positions", ROPE_AXES, every, at = 0),
        ],
        readout = "velocity",
        readout_width = VIDEO_FEATURES,
    ))
    return generation(
        readings = readings,
        latent = latent_space(
            channels = LATENT_CHANNELS,
            patch_t = PATCH_T,
            patch_h = PATCH_H,
            patch_w = PATCH_W,
            spatial_compression = SPATIAL_COMPRESSION,
            temporal_compression = TEMPORAL_COMPRESSION,
        ),
        schedule = schedule(
            "flow",
            shift = VIDEO_SHIFT,
            train_steps = TRAIN_STEPS,
            pinned_sigmas = shifted_sigmas(VIDEO_SHIFT, STEPS),
            stream_shifts = [("video", VIDEO_SHIFT), ("audio", AUDIO_SHIFT), ("reference", 1.0)],
        ),
        max_rows = 131072 if m.te != None else 4096,
    )
