# The weights of GLM-5.3-Flash: KDA linear attention with a DSA layer (latent
# attention over the keys a pooling indexer picks) every fourth, three dense
# layers and then routed experts, the residual stream carried as four
# hyper-connected streams; an optional vision tower and MTP draft head.

load("//lib/adapters/model.star", "banks")
load("//lib/hyper/model.star", "hyper", "mix")
load("//lib/kda/model.star", kda = "mixer")
load("//lib/mla/model.star", "attention")

def flash(layers = 45, experts = 288):
    return struct(
        hidden = 4096,
        layers = layers,
        dense_layers = 3,
        full_attn_every = 4,
        mla = struct(
            heads = 64,
            q_lora_rank = 1536,
            kv_lora_rank = 512,
            qk_nope_head_dim = 256,
            qk_rope_head_dim = 0,
            v_head_dim = 256,
        ),
        kda = struct(heads = 64, head_dim = 128, f_rank = 128, conv_kernel = 4),
        index_heads = 32,
        index_head_dim = 128,
        index_top_k = 2048,
        index_kpool = 4,
        streams = 4,
        gate_eps = 1e-6,
        alpha = 2.0,
        sinkhorn = 20,
        dense_inter = 12288,
        moe = struct(
            experts = experts,
            top_k = min(8, experts),
            inter = 2048,
            shared_inter = 2048,
            renorm = True,
            scaling = 2.5,
        ),
        swiglu_limit = 10.0,
        theta = 10000.0,
        vocab = 154880,
        norm_eps = 1e-5,
    )

DIMS = {
    "glm53-flash": flash(),
    # The first eight layers at full width with the first 32 routed experts:
    # three dense KDA, DSA + MoE, three KDA + MoE, DSA.
    "glm53-flash-mini": flash(layers = 8, experts = 32),
}

def layout(id, deploy):
    vision = "vision" in deploy.parts
    if deploy.drafter == "mtp":
        if len(deploy.weights) != 3:
            fail("{} with its draft head stores its weights, experts and draft experts at a dtype each, not {}".format(id, deploy.weights))
        weights, experts, draft = deploy.weights
    else:
        if len(deploy.weights) != 2:
            fail("{} stores its weights and its experts at a dtype each, not {}".format(id, deploy.weights))
        weights, experts = deploy.weights
        draft = None
    d = DIMS[id]
    dense = compute(weights)
    hidden = d.hidden
    streams = d.streams
    a = d.mla
    k = d.kda
    index_width = d.index_heads * d.index_head_dim

    def mla_at(prefix, kv, keys):
        n = lambda s: "{}.{}".format(prefix, s)
        return attention(
            n,
            a,
            hidden,
            weights = weights,
            norms = dense,
            eps = d.norm_eps,
            kv = kv,
            indexer = struct(
                heads = d.index_heads,
                head_dim = d.index_head_dim,
                top_k = d.index_top_k,
                kpool = d.index_kpool,
                rope_dim = a.qk_rope_head_dim,
                theta = d.theta,
                wq_b = weight(n("index_q_proj"), [index_width, a.q_lora_rank], weights),
                wk = weight(n("index_k_proj"), [d.index_head_dim, hidden], weights),
                weights_proj = weight(n("index_weights"), [d.index_heads, hidden], weights),
                k_norm = weight(n("index_k_norm"), [d.index_head_dim], dense),
                k_norm_bias = weight(n("index_k_norm_bias"), [d.index_head_dim], dense),
                k_norm_eps = d.norm_eps,
                kpool_ape = weight(n("index_kpool_ape"), [d.index_kpool, d.index_head_dim], dtype.f32),
                kpool_gate = weight(n("index_kpool_gate"), [d.index_head_dim, hidden], dense),
                keys = keys,
            ),
        )

    def routed_at(prefix, banks_dtype):
        n = lambda s: "{}.{}".format(prefix, s)
        m = d.moe
        iw = m.inter
        sw = m.shared_inter
        return struct(
            routed = True,
            router = weight(n("router"), [m.experts, hidden], dense),
            bias = weight(n("router_bias"), [m.experts], dtype.f32),
            gate_up = weight(n("experts_gate_up"), [m.experts, 2 * iw, hidden], banks_dtype).bank([iw, iw]),
            down = weight(n("experts_down"), [m.experts, hidden, iw], banks_dtype).rows(),
            shared = struct(
                gate_up = weight(n("shared_gate_up"), [2 * sw, hidden], weights).packed([sw, sw]),
                down = weight(n("shared_down"), [hidden, sw], weights).rows(),
                inter = sw,
            ),
            experts = m.experts,
            top_k = m.top_k,
            inter = iw,
            limit = d.swiglu_limit,
            renorm = m.renorm,
            scaling = m.scaling,
        )

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        norm = lambda s, width: weight(n(s), [width], dense)
        lora_a, lora_b = banks("layer.{}".format(l), hidden, dense)
        if (l + 1) % d.full_attn_every == 0:
            mixer = mla_at("layer.{}".format(l), "kv.{}".format(l), "index.{}".format(l))
            mixer_kind = "mla"
        else:
            mixer_kind = "kda"
            mixer = kda(
                n,
                l,
                k,
                hidden,
                weights = weights,
                conv = dense,
                eps = d.norm_eps,
                gate_floor = -5.0,
                gate_rank = k.f_rank,
            )
        if l < d.dense_layers:
            iw = d.dense_inter
            mlp = struct(
                routed = False,
                gate_up = weight(n("gate_up"), [2 * iw, hidden], weights).packed([iw, iw]),
                down = weight(n("down"), [hidden, iw], weights).rows(),
                inter = iw,
                limit = d.swiglu_limit,
            )
        else:
            mlp = routed_at("layer.{}".format(l), experts)
        return struct(
            attn_mix = mix(n, "attn_hc", streams, hidden),
            mixer_norm = norm("mixer_norm", hidden),
            mixer_norm_eps = d.norm_eps,
            mixer_kind = mixer_kind,
            mixer = mixer,
            mlp_mix = mix(n, "ffn_hc", streams, hidden),
            mlp_norm = norm("mlp_norm", hidden),
            mlp_norm_eps = d.norm_eps,
            mlp = mlp,
            lora_a = lora_a,
            lora_b = lora_b,
        )

    mtp = None
    if draft != None:
        mtp = struct(
            enorm = weight("mtp.enorm", [hidden], dense),
            hnorm = weight("mtp.hnorm", [hidden], dense),
            e_proj = weight("mtp.e_proj", [hidden, hidden], dtype.bf16),
            h_proj = weight("mtp.h_proj", [hidden, hidden], dtype.bf16),
            mixer_norm = weight("mtp.mixer_norm", [hidden], dense),
            mixer_norm_eps = d.norm_eps,
            attn = mla_at("mtp", "kv.mtp", "index.mtp"),
            mlp_norm = weight("mtp.mlp_norm", [hidden], dense),
            mlp_norm_eps = d.norm_eps,
            mlp = routed_at("mtp", draft),
            norm = weight("mtp.norm", [hidden], dense),
            norm_eps = d.norm_eps,
        )

    return struct(
        hidden = hidden,
        vocab = d.vocab,
        act = dense,
        heads = a.heads,
        kv_lora_rank = a.kv_lora_rank,
        kv = deploy.kv,
        hyper = hyper(streams, d.norm_eps, d.gate_eps, d.alpha, d.sinkhorn),
        embed = weight("embed", [d.vocab, hidden], dtype.u4g64),
        head = weight("lm_head", [d.vocab, hidden], dtype.u4g64),
        layers = [layer(l) for l in range(d.layers)],
        final_norm = weight("final_norm", [hidden], dense),
        final_norm_eps = d.norm_eps,
        mtp = mtp,
        tower = tower(d) if vision else None,
    )

def tower(d):
    hidden, heads, depth, inter, merger_inter = 1024, 16, 24, 4096, 10240
    patch, temporal, merge = 14, 2, 2
    out = d.hidden
    head_dim = hidden // heads
    bf = dtype.bf16
    v = lambda s: "vision." + s

    def block(l):
        n = lambda s: v("blocks.{}.{}".format(l, s))
        return struct(
            norm1 = weight(n("norm1"), [hidden], bf),
            qkv = weight(n("qkv"), [3 * hidden, hidden], bf).packed([hidden] * 3),
            qkv_bias = weight(n("qkv_bias"), [3 * hidden], bf).packed([hidden] * 3),
            q_norm = weight(n("q_norm"), [head_dim], bf),
            k_norm = weight(n("k_norm"), [head_dim], bf),
            proj = weight(n("proj"), [hidden, hidden], bf),
            proj_bias = weight(n("proj_bias"), [hidden], bf),
            norm2 = weight(n("norm2"), [hidden], bf),
            gate_up = weight(n("gate_up"), [2 * inter, hidden], bf).packed([inter, inter]),
            gate_up_bias = weight(n("gate_up_bias"), [2 * inter], bf).packed([inter, inter]),
            down = weight(n("down"), [hidden, inter], bf),
            down_bias = weight(n("down_bias"), [hidden], bf),
        )

    return struct(
        hidden = hidden,
        heads = heads,
        head_dim = head_dim,
        merge = merge,
        patch_width = 3 * temporal * patch * patch,
        inter = inter,
        merger_inter = merger_inter,
        limit = d.swiglu_limit,
        theta = 10000.0,
        norm_eps = d.norm_eps,
        sm_scale = f32(1.0 / f32(sqrt(head_dim))),
        patch_embed = weight(v("patch_embed"), [hidden, 3 * temporal * patch * patch], bf),
        patch_embed_bias = weight(v("patch_embed_bias"), [hidden], bf),
        blocks = [block(l) for l in range(depth)],
        post_norm = weight(v("post_norm"), [hidden], bf),
        downsample = weight(v("downsample"), [out, merge * merge * hidden], bf),
        downsample_bias = weight(v("downsample_bias"), [out], bf),
        merger = struct(
            proj = weight(v("merger_proj"), [out, out], bf),
            norm = weight(v("merger_norm"), [out], bf),
            norm_bias = weight(v("merger_norm_bias"), [out], bf),
            gate_up = weight(v("merger_gate_up"), [2 * merger_inter, out], bf).packed([merger_inter, merger_inter]),
            down = weight(v("merger_down"), [out, merger_inter], bf),
        ),
    )
