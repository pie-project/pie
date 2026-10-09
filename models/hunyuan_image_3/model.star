# The weights of HunyuanImage 3: the trunk's layers (GQA attention with a 2-D
# rope, routed and shared experts), the timestep embedders, and the U-Net
# blocks that take a latent clip into the trunk and its rows back out.

TRAIN_STEPS = 1000
FLOW_SHIFT = 3.0
ROPE_AXES = 2
T_FREQ_DIM = 256
GN_GROUPS = 32
PATCH = 1
LATENT_CHANNELS = 32
SPATIAL_COMPRESSION = 16
CONV3 = [1, 3, 3]
CONV1 = [1, 1, 1]

DIMS = {
    "hunyuanimage3-80b-a13b": struct(
        hidden = 4096,
        layers = 32,
        q_heads = 32,
        kv_heads = 8,
        head_dim = 128,
        vocab = 133120,
        experts = 64,
        top_k = 8,
        moe_inter = 3072,
        shared_inter = 3072,
        head_hidden = 1024,
    ),
    "hunyuanimage3-mini": struct(
        hidden = 256,
        layers = 2,
        q_heads = 4,
        kv_heads = 2,
        head_dim = 64,
        vocab = 133120,
        experts = 8,
        top_k = 2,
        moe_inter = 256,
        shared_inter = 256,
        head_hidden = 64,
    ),
}

def linear(name, out, in_, banks):
    return struct(
        w = weight(name, [out, in_], banks),
        bias = weight(name + ".bias", [out], compute(banks)),
    )

def conv(name, c_out, c_in, k, banks):
    taps = k[0] * k[1] * k[2]
    return struct(
        w = weight(name, [c_out, c_in * taps], banks).conv_taps_major(c_in, taps),
        bias = weight(name + ".bias", [c_out], dtype.f32),
        k = k,
    )

def group_norm(name, c):
    return struct(
        weight = weight(name, [c], dtype.f32),
        bias = weight(name + ".bias", [c], dtype.f32),
    )

def resblock(prefix, c_in, c_out, emb, banks):
    n = lambda s: prefix + "." + s
    return struct(
        norm_in = group_norm(n("norm_in"), c_in),
        conv_in = conv(n("conv_in"), c_out, c_in, CONV3, banks),
        emb = linear(n("emb"), 2 * c_out, emb, banks),
        norm_out = group_norm(n("norm_out"), c_out),
        conv_out = conv(n("conv_out"), c_out, c_out, CONV3, banks),
        skip = conv(n("skip"), c_out, c_in, CONV1, banks) if c_in != c_out else None,
    )

def embedder(prefix, hidden, out, banks):
    return struct(
        mlp_in = linear(prefix + ".in", hidden, T_FREQ_DIM, banks),
        mlp_out = linear(prefix + ".out", out, hidden, banks),
    )

def layout(id, deploy):
    d = DIMS[id]
    if id == "hunyuanimage3-mini":
        if len(deploy.weights) != 1:
            fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
        banks = deploy.weights[0]
        experts = banks
    else:
        if len(deploy.weights) != 2 or deploy.weights[0] != dtype.bf16:
            fail("{} does not ship {}".format(id, deploy.weights))
        banks, experts = deploy.weights
    if d.head_dim % 4 != 0:
        fail("the 2-D rope splits a head into two even blocks")
    dense = compute(banks)
    hidden = d.hidden
    hd = d.head_dim
    q_w = d.q_heads * hd
    kv_w = d.kv_heads * hd
    iw = d.moe_inter
    sw = d.shared_inter

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        return struct(
            attn_norm = weight(n("attn_norm"), [hidden], dense),
            qkv = weight(n("qkv"), [q_w + 2 * kv_w, hidden], banks).packed([q_w, kv_w, kv_w], heads = [d.q_heads, d.kv_heads, d.kv_heads]),
            q_norm = weight(n("q_norm"), [hd], dense),
            k_norm = weight(n("k_norm"), [hd], dense),
            o_proj = weight(n("o_proj"), [hidden, q_w], banks).rows(),
            kv = "kv.{}".format(l),
            mlp_norm = weight(n("mlp_norm"), [hidden], dense),
            router = weight(n("router"), [d.experts, hidden], dense),
            experts_gate_up = weight(n("experts_gate_up"), [d.experts, 2 * iw, hidden], experts).bank([iw, iw]),
            experts_down = weight(n("experts_down"), [d.experts, hidden, iw], experts).rows(),
            shared_gate_up = weight(n("shared_gate_up"), [2 * sw, hidden], banks).packed([sw, sw]),
            shared_down = weight(n("shared_down"), [hidden, sw], banks).rows(),
        )

    hw = d.head_hidden
    return struct(
        dims = d,
        kv = deploy.kv,
        q_width = q_w,
        kv_width = kv_w,
        sm_scale = f32(1.0 / f32(sqrt(hd))),
        embed = weight("wte", [d.vocab, hidden], banks),
        head = weight("lm_head", [d.vocab, hidden], banks),
        final_norm = weight("ln_f", [hidden], dense),
        layers = [layer(l) for l in range(d.layers)],
        timestep_emb = struct(
            mlp_in = linear("timestep_emb.in", hidden, T_FREQ_DIM, banks),
            mlp_out = linear("timestep_emb.out", 2 * hidden, hidden, banks),
        ),
        time_embed = embedder("time_embed", hidden, hidden, banks),
        time_embed_2 = embedder("time_embed_2", hidden, hidden, banks),
        patch_embed = struct(
            conv_in = conv("patch_embed.conv", hw, LATENT_CHANNELS, CONV3, banks),
            res = resblock("patch_embed.res", hw, hidden, hidden, banks),
        ),
        final_layer = struct(
            res = resblock("final_layer.res", hidden, hw, hidden, banks),
            norm_out = group_norm("final_layer.norm_out", hw),
            conv_out = conv("final_layer.conv", LATENT_CHANNELS, hw, CONV3, banks),
        ),
        ones = weight("special.ones", [2 * hidden, 1], banks),
    )

def generative(m):
    d = m.dims
    p = lambda name, kind, width, at = None: port(name, kind, width, [], at = at)
    return generation(
        readings = [
            reading(
                "encode",
                index = 0,
                has_kv = True,
                takes_tokens = True,
                streams = ["text"],
                ports = [p("positions", "axis_positions", ROPE_AXES)],
                readout = "logits",
                readout_width = d.vocab,
            ),
            reading(
                "denoise",
                index = 1,
                has_kv = True,
                takes_tokens = True,
                streams = ["image"],
                ports = [
                    p("latents", "latents", d.hidden),
                    p("special", "latents", 1),
                    p("timestep", "lane_vector", 1),
                    p("positions", "axis_positions", ROPE_AXES),
                ],
                readout = "hidden",
                readout_width = d.hidden,
            ),
            reading(
                "image.in",
                index = 2,
                streams = ["image"],
                ports = [p("latent", "voxels", LATENT_CHANNELS + T_FREQ_DIM, at = 0)],
                readout = "pixels",
                readout_width = d.hidden,
            ),
            reading(
                "image.out",
                index = 3,
                streams = ["image"],
                ports = [p("rows", "voxels", d.hidden + T_FREQ_DIM, at = 1)],
                readout = "pixels",
                readout_width = LATENT_CHANNELS,
            ),
        ],
        latent = latent_space(
            channels = LATENT_CHANNELS,
            patch_t = 1,
            patch_h = PATCH,
            patch_w = PATCH,
            spatial_compression = SPATIAL_COMPRESSION,
            temporal_compression = 1,
        ),
        schedule = schedule("flow", shift = FLOW_SHIFT, train_steps = TRAIN_STEPS),
        max_rows = 8192 if d.layers > 4 else 1024,
    )

def diffusion(m):
    side = 64 if m.dims.layers > 4 else 8
    return canvas(canvas = side * side, hidden = m.dims.hidden, self_cond_taps = 0)
