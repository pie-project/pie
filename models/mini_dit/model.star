# The weights of mini-dit. `PIE_MINI_DIT_TAP` names an intermediate the
# forward reads out instead of its velocity, for parity against the
# reference.

HIDDEN = 256
HEADS = 4
HEAD_DIM = 64
INTER = 512
CHANNELS = 16
PATCH = 2
PATCH_FEATURES = CHANNELS * PATCH * PATCH
TEXT_WIDTH = 256
CONTEXT_WIDTH = 512
MOD_SLICES = 6
ROPE_AXES = 3

def linear(name, out, in_, banks):
    return struct(
        w = weight(name, [out, in_], banks),
        bias = weight(name + ".bias", [out], compute(banks)),
    )

def columns(l):
    return struct(w = l.w.columns(), bias = l.bias.columns())

def rows(l):
    return struct(w = l.w.rows(), bias = l.bias)

def packed(l, seams):
    return struct(w = l.w.packed(seams), bias = l.bias.packed(seams))

def self_attn(prefix, banks):
    dense = compute(banks)
    return struct(
        qkv = packed(linear(prefix + ".qkv", 3 * HIDDEN, HIDDEN, banks), [HIDDEN] * 3),
        q_norm = weight(prefix + ".q_norm", [HEAD_DIM], dense),
        k_norm = weight(prefix + ".k_norm", [HEAD_DIM], dense),
        out = rows(linear(prefix + ".o", HIDDEN, HIDDEN, banks)),
    )

def cross_attn(prefix, banks):
    dense = compute(banks)
    return struct(
        q = columns(linear(prefix + ".q", HIDDEN, HIDDEN, banks)),
        kv = packed(linear(prefix + ".kv", 2 * HIDDEN, CONTEXT_WIDTH, banks), [HIDDEN] * 2),
        q_norm = weight(prefix + ".q_norm", [HEAD_DIM], dense),
        k_norm = weight(prefix + ".k_norm", [HEAD_DIM], dense),
        out = rows(linear(prefix + ".o", HIDDEN, HIDDEN, banks)),
    )

def swiglu(prefix, banks):
    name = prefix + ".gate_up"
    seams = [INTER, INTER]
    return struct(
        gate_up = struct(
            w = weight(name, [2 * INTER, HIDDEN], banks).packed(seams),
            bias = weight(name + ".bias", [2 * INTER], compute(banks)).packed(seams),
        ),
        down = rows(linear(prefix + ".down", HIDDEN, INTER, banks)),
    )

def side(prefix, banks):
    return struct(
        ada = linear(prefix + ".ada", MOD_SLICES * HIDDEN, HIDDEN, banks),
        attn = self_attn(prefix + ".attn", banks),
        mlp = swiglu(prefix + ".mlp", banks),
    )

def tap():
    key = env("PIE_MINI_DIT_TAP")
    return key if key != "" else None

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    banks = deploy.weights[0]
    dense = compute(banks)
    return struct(
        x_embed = linear("x_embed", HIDDEN, PATCH_FEATURES, banks),
        single = side("single", banks),
        double = struct(img = side("double.img", banks), txt = side("double.txt", banks)),
        cross = struct(
            mod_table = weight("cross.mod_table", [MOD_SLICES * HIDDEN], dense),
            ada = linear("cross.ada", MOD_SLICES * HIDDEN, HIDDEN, banks),
            self_attn = self_attn("cross.self", banks),
            norm = weight("cross.norm", [HIDDEN], dense),
            norm_bias = weight("cross.norm.bias", [HIDDEN], dense),
            cross = cross_attn("cross.x", banks),
            mlp = swiglu("cross.mlp", banks),
        ),
        final_ada = linear("final.ada", 2 * HIDDEN, HIDDEN, banks),
        final_proj = linear("final.proj", PATCH_FEATURES, HIDDEN, banks),
        tap = tap(),
    )

def generative(m):
    p = lambda name, kind, width, streams: port(name, kind, width, streams)
    return generation(
        readings = [reading(
            "denoise",
            index = 0,
            streams = ["text", "image", "context"],
            ports = [
                p("latents", "latents", PATCH_FEATURES, ["image"]),
                p("text", "context", TEXT_WIDTH, ["text"]),
                p("context", "context", CONTEXT_WIDTH, ["context"]),
                p("timestep", "lane_vector", 1, ["text", "image"]),
                p("positions", "axis_positions", ROPE_AXES, ["text", "image"]),
            ],
            positions = positions(
                axes = ["time", "height", "width"],
                text_axis = 0,
                text_origin = 0,
                image_follows_text = False,
            ),
            readout = "velocity",
            readout_width = PATCH_FEATURES if m.tap in [None, "final.tokens"] else HIDDEN,
        )],
        latent = latent_space(
            channels = CHANNELS,
            patch_t = 1,
            patch_h = PATCH,
            patch_w = PATCH,
            spatial_compression = 1,
            temporal_compression = 1,
        ),
        schedule = schedule("flow", shift = 1.0, train_steps = 1000, pinned_sigmas = [1.0, 0.75, 0.5, 0.25]),
        max_rows = 4096,
    )
