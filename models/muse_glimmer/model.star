# Muse Glimmer: a dense decoder of sliding-window layers with a full-attention
# layer every fourth, gated attention and sandwich norms.

load("//lib/adapters/model.star", "banks")

def b30(layers = 52):
    return struct(
        hidden = 6656,
        layers = layers,
        full_every = 4,
        q_heads = 32,
        kv_heads = 2,
        head_dim = 128,
        intermediate = 19968,
        vocab = 202048,
        window = 2048,
        theta = 500000.0,
        qk_scale = 3.87,
        softcap = 20.0,
        output_multiplier = 0.19611613,
        norm_eps = 1e-5,
        post_norm_eps = 1e-8,
    )

DIMS = {
    "muse-glimmer-30b": b30(),
    "muse-glimmer-30b-mini-l8": b30(layers = 8),
}

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    d = DIMS[id]
    w = deploy.weights[0]
    dense = compute(w)
    proj = dtype.u4g64tiled if w == dtype.u4g64 else w
    hidden = d.hidden
    hd = d.head_dim
    q_w = d.q_heads * hd
    kv_w = d.kv_heads * hd
    iw = d.intermediate

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        norm = lambda s: weight(n(s), [hidden], dense)
        lora_a, lora_b = banks("layer.{}".format(l), hidden, dense)
        return struct(
            full = l % d.full_every == d.full_every - 1,
            qkv = weight(n("qkv"), [q_w + 2 * kv_w, hidden], proj).packed([q_w, kv_w, kv_w], heads = [d.q_heads, d.kv_heads, d.kv_heads]),
            gate = weight(n("gate"), [q_w, hidden], proj).columns(),
            o_proj = weight(n("o_proj"), [hidden, q_w], proj).rows(),
            kv = "kv.{}".format(l),
            attn_norm = norm("attn_norm"),
            attn_norm_eps = d.norm_eps,
            post_attn_norm = norm("post_attn_norm"),
            post_attn_norm_eps = d.post_norm_eps,
            pre_ffw_norm = norm("pre_ffw_norm"),
            pre_ffw_norm_eps = d.norm_eps,
            post_ffw_norm = norm("post_ffw_norm"),
            post_ffw_norm_eps = d.post_norm_eps,
            gate_up = weight(n("gate_up"), [2 * iw, hidden], proj).packed([iw, iw]),
            inter = iw,
            down = weight(n("down"), [hidden, iw], proj).rows(),
            lora_a = lora_a,
            lora_b = lora_b,
        )

    lm_head = weight("lm_head", [d.vocab, hidden], proj)
    if env("PIE_NO_VOCAB_SHARD") == None:
        lm_head = lm_head.packed([d.vocab])
    return struct(
        hidden = hidden,
        vocab = d.vocab,
        q_heads = d.q_heads,
        kv_heads = d.kv_heads,
        head_dim = hd,
        window = d.window,
        theta = d.theta,
        sm_scale = f32(f32(d.qk_scale) / f32(sqrt(hd))),
        norm_eps = d.norm_eps,
        kv = deploy.kv,
        softcap = d.softcap,
        output_multiplier = d.output_multiplier,
        embed = weight("embed", [d.vocab, hidden], w),
        lm_head = lm_head,
        layers = [layer(l) for l in range(d.layers)],
        final_norm = weight("final_norm", [hidden], dense),
        final_norm_eps = d.norm_eps,
    )
