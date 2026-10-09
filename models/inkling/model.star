# The weights of Inkling: attention with a relative bias over a local window,
# every sixth layer over a longer reach; short convolutions on its keys,
# values and both residual branches; two dense layers, then routed experts
# with sink experts always taken.

load("//lib/adapters/model.star", "banks")

def dims(layers = 66, experts = 256):
    return struct(
        hidden = 6144,
        layers = layers,
        vocab = 201024,
        head_rows = 200058,
        heads = 64,
        head_dim = 128,
        local_kv_heads = 16,
        global_kv_heads = 8,
        d_rel = 16,
        window = 512,
        global_extent = 1024,
        conv_width = 4,
        global_every = 6,
        dense_layers = 2,
        dense_inter = 24576,
        experts = experts,
        top_k = min(6, experts),
        sink = 2,
        moe_inter = 3072,
        route_scale = 8.0,
        mup = 24.0,
        norm_eps = 1e-6,
        log_floor = 128000,
        log_alpha = 0.1,
    )

DIMS = {
    "inkling": dims(),
    "inkling-mini-l7-e8": dims(layers = 7, experts = 8),
}

LOCAL = 0
GLOBAL = 1

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    d = DIMS[id]
    w = deploy.weights[0]
    dense = compute(w)
    hidden = d.hidden
    hd = d.head_dim
    q_w = d.heads * hd
    r_w = d.heads * d.d_rel
    kw = d.conv_width

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        vec = lambda s, length: weight(n(s), [length], dense)
        lora_a, lora_b = banks("layer.{}".format(l), hidden, dense)
        if (l + 1) % d.global_every == 0:
            reading, kv_heads, extent = GLOBAL, d.global_kv_heads, d.global_extent
        else:
            reading, kv_heads, extent = LOCAL, d.local_kv_heads, d.window
        kv_w = kv_heads * hd
        conv = lambda s, channels: weight(n(s), [channels, kw], dense).columns()
        if l < d.dense_layers:
            iw = d.dense_inter
            mlp = struct(
                routed = False,
                gate_up = weight(n("gate_up"), [2 * iw, hidden], w).packed([iw, iw]),
                inter = iw,
                down = weight(n("down"), [hidden, iw], w).rows(),
                scale = vec("mlp_scale", 1),
            )
        else:
            bank = d.experts + d.sink
            mi = d.moe_inter
            mlp = struct(
                routed = True,
                router = weight(n("router"), [bank, hidden], w),
                bias = weight(n("router_bias"), [d.experts], dtype.f32),
                scale = weight(n("router_scale"), [1], dtype.f32),
                gate_up = weight(n("experts_gate_up"), [bank, 2 * mi, hidden], w).bank([mi, mi]),
                down = weight(n("experts_down"), [bank, hidden, mi], w).rows(),
                experts = d.experts,
                top_k = d.top_k,
                sink = d.sink,
                inter = mi,
                scaling = d.route_scale,
            )
        return struct(
            reading = reading,
            kv_heads = kv_heads,
            extent = extent,
            attn_norm = vec("attn_norm", hidden),
            q_proj = weight(n("q_proj"), [q_w, hidden], w).columns(),
            k_proj = weight(n("k_proj"), [kv_w, hidden], w).columns(heads = kv_heads),
            v_proj = weight(n("v_proj"), [kv_w, hidden], w).columns(heads = kv_heads),
            r_proj = weight(n("r_proj"), [r_w, hidden], w).columns(),
            o_proj = weight(n("o_proj"), [hidden, q_w], w).rows(),
            q_norm = vec("q_norm", hd),
            k_norm = vec("k_norm", hd),
            rel_proj = weight(n("rel_proj"), [d.d_rel, extent], dense),
            k_conv = conv("k_conv", kv_w),
            v_conv = conv("v_conv", kv_w),
            attn_conv = weight(n("attn_conv"), [hidden, kw], dense),
            mlp_conv = weight(n("mlp_conv"), [hidden, kw], dense),
            kv = "kv.{}".format(l),
            k_state = "conv.{}.k".format(l),
            v_state = "conv.{}.v".format(l),
            attn_state = "conv.{}.attn".format(l),
            mlp_state = "conv.{}.mlp".format(l),
            mlp_norm = vec("mlp_norm", hidden),
            mlp = mlp,
            lora_a = lora_a,
            lora_b = lora_b,
        )

    return struct(
        hidden = hidden,
        vocab = d.vocab,
        head_rows = d.head_rows,
        heads = d.heads,
        head_dim = hd,
        d_rel = d.d_rel,
        window = d.window,
        conv_width = kw,
        sm_scale = f32(1.0 / f32(hd)),
        norm_eps = d.norm_eps,
        head_scale = f32(1.0 / f32(d.mup)),
        log_scaling = (d.log_floor, d.log_alpha),
        kv = deploy.kv,
        embed = weight("embed", [d.vocab, hidden], w),
        embed_norm = weight("embed_norm", [hidden], dense),
        layers = [layer(l) for l in range(d.layers)],
        final_norm = weight("final_norm", [hidden], dense),
        unembed = weight("unembed", [d.head_rows, hidden], w),
    )
