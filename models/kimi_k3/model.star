# The weights of Kimi-K3: KDA linear attention with an MLA layer every
# fourth, one dense layer and then routed experts (in the released model a
# latent MoE), the residual stream carried as AttnRes blocks.

ADAPTERS = struct(slots = 8, rank = 16)

# Where AttnRes blends the residual stream with its closed blocks.
AT_BLOCK_START = "at_block_start"
EVERY = "every"

def fixture():
    return struct(
        hidden = 2048,
        layers = 8,
        dense_layers = 1,
        full_attn_every = 4,
        res_block = 4,
        attn_res = AT_BLOCK_START,
        gate_floor = 0.0,
        mla = struct(
            heads = 16,
            q_lora_rank = 768,
            kv_lora_rank = 256,
            qk_nope_head_dim = 128,
            qk_rope_head_dim = 64,
            v_head_dim = 128,
            output_gate = True,
        ),
        kda = struct(heads = 16, head_dim = 128, f_rank = 128, conv_kernel = 4, norm_eps = 1e-5),
        moe = struct(
            experts = 64,
            top_k = 6,
            routed_scaling = 2.0,
            inter = 1024,
            shared_inter = 1024,
            latent = None,
            latent_norm = False,
            bias = False,
            renorm = False,
        ),
        dense_inter = 5632,
        situ_beta = 1.0,
        situ_cap = None,
        vocab = 163840,
        norm_eps = 1e-5,
    )

def released(layers, experts, res_block):
    """`moonshotai/Kimi-K3` at `layers` layers and `experts` routed experts,
    AttnRes blocks of `res_block`."""
    return struct(
        hidden = 7168,
        layers = layers,
        dense_layers = 1,
        full_attn_every = 4,
        res_block = res_block,
        attn_res = EVERY,
        gate_floor = -5.0,
        mla = struct(
            heads = 96,
            q_lora_rank = 1536,
            kv_lora_rank = 512,
            qk_nope_head_dim = 128,
            qk_rope_head_dim = 64,
            v_head_dim = 128,
            output_gate = True,
        ),
        kda = struct(heads = 96, head_dim = 128, f_rank = 128, conv_kernel = 4, norm_eps = 1e-5),
        moe = struct(
            experts = experts,
            top_k = min(16, experts),
            routed_scaling = 1.0,
            inter = 3072,
            shared_inter = 2 * 3072,
            latent = 3584,
            latent_norm = True,
            bias = True,
            renorm = True,
        ),
        dense_inter = 33792,
        situ_beta = 4.0,
        situ_cap = 25.0,
        vocab = 163840,
        norm_eps = 1e-5,
    )

DIMS = {
    "kimik3-mini": released(8, 32, 4),
    "kimik3": fixture(),
}

def closes_a_block(l, every):
    return every > 0 and (l + 1) % every == 0

def banks(prefix, hidden, dense):
    return (
        weight(prefix + ".lora_a", [ADAPTERS.slots, ADAPTERS.rank, hidden], dense).registered(),
        weight(prefix + ".lora_b", [ADAPTERS.slots, hidden, ADAPTERS.rank], dense).registered(),
    )

def layout(id, deploy):
    if len(deploy.weights) != 2 or deploy.weights[0] != dtype.bf16:
        fail("{} stores its weights at bf16 and its experts at a dtype of their own, not {}".format(
            id,
            deploy.weights,
        ))
    d = DIMS[id]
    weights, experts = deploy.weights
    hidden = d.hidden
    moe_in = d.moe.latent if d.moe.latent != None else hidden
    a = d.mla
    k = d.kda
    qk_head_dim = a.qk_nope_head_dim + a.qk_rope_head_dim
    v_width = a.heads * a.v_head_dim
    kda_width = k.heads * k.head_dim

    def blend_at(l):
        # Under `every` the first layer has no closed block to blend with, so
        # its blend planes are never read.
        if d.attn_res == AT_BLOCK_START:
            return l > 0 and closes_a_block(l - 1, d.res_block)
        return l > 0

    def blend(n, norm_name, proj_name):
        return struct(
            norm = weight(n(norm_name), [hidden], weights),
            norm_eps = d.norm_eps,
            proj = weight(n(proj_name), [1, hidden], weights),
        )

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        norm = lambda s, width: weight(n(s), [width], weights)
        if closes_a_block(l, d.full_attn_every):
            mixer = struct(
                mla = True,
                qk_nope_head_dim = a.qk_nope_head_dim,
                qk_rope_head_dim = a.qk_rope_head_dim,
                v_head_dim = a.v_head_dim,
                sm_scale = f32(1.0 / f32(sqrt(qk_head_dim))),
                q_a_proj = weight(n("q_a_proj"), [a.q_lora_rank, hidden], weights),
                q_a_norm = norm("q_a_norm", a.q_lora_rank),
                q_a_norm_eps = d.norm_eps,
                q_b_proj = weight(n("q_b_proj"), [a.heads * qk_head_dim, a.q_lora_rank], weights).columns(),
                kv_a_proj = weight(n("kv_a_proj"), [a.kv_lora_rank + a.qk_rope_head_dim, hidden], weights),
                kv_a_norm = norm("kv_a_norm", a.kv_lora_rank),
                kv_a_norm_eps = d.norm_eps,
                kv_b_proj = weight(
                    n("kv_b_proj"),
                    [a.heads * (a.qk_nope_head_dim + a.v_head_dim), a.kv_lora_rank],
                    weights,
                ).columns(),
                gate = weight(n("o_gate"), [v_width, hidden], weights).columns() if a.output_gate else None,
                o_proj = weight(n("o_proj"), [hidden, v_width], weights).rows(),
                kv = "kv.{}".format(l),
            )
        else:
            mixer = struct(
                mla = False,
                heads = k.heads,
                head_dim = k.head_dim,
                conv_kernel = k.conv_kernel,
                norm_eps = k.norm_eps,
                gate_floor = d.gate_floor,
                qkv = weight(n("kda_qkv"), [3 * kda_width, hidden], weights).packed([kda_width] * 3),
                conv = weight(n("kda_conv"), [3 * kda_width, k.conv_kernel], weights).packed([kda_width] * 3),
                f_a = weight(n("kda_f_a"), [k.f_rank, hidden], weights),
                f_b = weight(n("kda_f_b"), [kda_width, k.f_rank], weights).columns(),
                b = weight(n("kda_b"), [k.heads, hidden], weights).columns(),
                dt_bias = weight(n("kda_dt_bias"), [k.heads, k.head_dim], dtype.f32).columns(),
                a_log = weight(n("kda_a_log"), [k.heads], dtype.f32).columns(),
                gate = weight(n("kda_gate"), [kda_width, hidden], weights).columns(),
                o_norm = weight(n("kda_o_norm"), [k.head_dim], dtype.f32),
                o_norm_eps = k.norm_eps,
                o_proj = weight(n("kda_o_proj"), [hidden, kda_width], weights).rows(),
                conv_state = "conv.{}".format(l),
                delta_state = "delta.{}".format(l),
            )
        if l >= d.dense_layers:
            m = d.moe
            inter = m.inter
            sw = m.shared_inter
            mlp = struct(
                routed = True,
                router = weight(n("router"), [m.experts, hidden], weights),
                bias = weight(n("router_bias"), [m.experts], dtype.f32) if m.bias else None,
                gate_up = weight(n("experts_gate_up"), [m.experts, 2 * inter, moe_in], experts).bank([inter, inter]),
                down = weight(n("experts_down"), [m.experts, moe_in, inter], experts).rows(),
                latent = struct(
                    width = m.latent,
                    down = weight(n("latent_down"), [m.latent, hidden], weights),
                    norm = weight(n("latent_norm"), [m.latent], weights) if m.latent_norm else None,
                    norm_eps = d.norm_eps,
                    up = weight(n("latent_up"), [hidden, m.latent], weights),
                ) if m.latent != None else None,
                shared = struct(
                    gate_up = weight(n("shared_gate_up"), [2 * sw, hidden], weights).packed([sw, sw]),
                    down = weight(n("shared_down"), [hidden, sw], weights).rows(),
                    inter = sw,
                ) if sw > 0 else None,
                experts = m.experts,
                top_k = m.top_k,
                renorm = m.renorm,
                routed_scaling = m.routed_scaling,
                inter = inter,
                beta = d.situ_beta,
                up_cap = d.situ_cap,
            )
        else:
            inter = d.dense_inter
            mlp = struct(
                routed = False,
                gate_up = weight(n("gate_up"), [2 * inter, hidden], weights).packed([inter, inter]),
                down = weight(n("down"), [hidden, inter], weights).rows(),
                inter = inter,
                beta = d.situ_beta,
                up_cap = d.situ_cap,
            )
        lora_a, lora_b = banks("layer.{}".format(l), hidden, compute(weights))
        return struct(
            res_blend = blend(n, "res_norm", "res_proj") if blend_at(l) else None,
            mlp_res = blend(n, "mlp_res_norm", "mlp_res_proj") if d.attn_res == EVERY else None,
            mixer = mixer,
            mixer_norm = norm("mixer_norm", hidden),
            mixer_norm_eps = d.norm_eps,
            mlp_norm = norm("mlp_norm", hidden),
            mlp_norm_eps = d.norm_eps,
            mlp = mlp,
            lora_a = lora_a,
            lora_b = lora_b,
        )

    head = weight("lm_head", [d.vocab, hidden], weights)
    if env("PIE_NO_VOCAB_SHARD") == None:
        head = head.packed([d.vocab])
    return struct(
        hidden = hidden,
        vocab = d.vocab,
        res_block = d.res_block,
        mla_heads = a.heads,
        kv_lora_rank = a.kv_lora_rank,
        kv = deploy.kv,
        embed = weight("embed", [d.vocab, hidden], weights),
        head = head,
        layers = [layer(l) for l in range(d.layers)],
        final_norm = weight("final_norm", [hidden], weights),
        final_norm_eps = d.norm_eps,
        attn_res = d.attn_res,
        output_res = struct(
            norm = weight("output_res_norm", [hidden], weights),
            norm_eps = d.norm_eps,
            proj = weight("output_res_proj", [1, hidden], weights),
        ) if d.attn_res == EVERY else None,
    )
