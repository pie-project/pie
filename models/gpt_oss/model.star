# The weights of gpt-oss: attention with sinks, every other layer over a
# sliding window, YaRN rope, and routed experts with biases, their banks
# stored as the second dtype of a deployment's weights.

load("//lib/adapters/model.star", "banks")
load("//lib/dflash/model.star", "declare", "head")

def dims(layers = 24, experts = 32):
    return struct(
        hidden = 2880,
        layers = layers,
        q_heads = 64,
        kv_heads = 8,
        head_dim = 64,
        theta = 150000.0,
        yarn_factor = 32.0,
        yarn_beta_fast = 32.0,
        yarn_beta_slow = 1.0,
        yarn_attention_factor = 1.3465736,
        yarn_original_max_position = 4096,
        window = 128,
        experts = experts,
        top_k = 4,
        inter = 2880,
        swiglu_limit = 7.0,
        swiglu_alpha = 1.702,
        vocab = 201088,
        norm_eps = 1e-5,
    )

DIMS = {
    "gptoss-20b": dims(),
    "gptoss-20b-mini": dims(layers = 5, experts = 16),
    "gptoss-120b": dims(layers = 36, experts = 128),
}

DFLASH_20B = head(
    taps = [1, 6, 11, 16, 21],
    windows = [None] * 8,
    q_heads = 64,
    kv_heads = 8,
    head_dim = 64,
    inter = 7680,
    theta = 150000.0,
    block = 8,
    mask_token = 200000,
    attn_bias = True,
)

WINDOWED = 0
FULL = 1

def layout(id, deploy):
    if len(deploy.weights) != 2:
        fail("{} stores its weights and its experts at a dtype each, not {}".format(id, deploy.weights))
    if deploy.drafter != None and deploy.drafter != "dflash":
        fail("{} drafts with dflash or nothing, not {}".format(id, deploy.drafter))
    d = DIMS[id]
    weights, experts = deploy.weights
    dense = compute(weights)
    router = dtype.u8g64 if weights == dtype.u4g64 else weights
    hidden = d.hidden
    hd = d.head_dim
    q_w = d.q_heads * hd
    kv_w = d.kv_heads * hd
    iw = d.inter

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        norm = lambda s, cols: weight(n(s), [cols], dense)
        lora_a, lora_b = banks("layer.{}".format(l), hidden, dense)
        return struct(
            attn = struct(
                reading = WINDOWED if l % 2 == 0 else FULL,
                sm_scale = f32(1.0 / f32(sqrt(hd))),
                theta = d.theta,
                factor = d.yarn_factor,
                beta_fast = d.yarn_beta_fast,
                beta_slow = d.yarn_beta_slow,
                attention_factor = d.yarn_attention_factor,
                original_max_position = d.yarn_original_max_position,
                q_proj = weight(n("q_proj"), [q_w, hidden], weights).columns(),
                q_bias = weight(n("q_bias"), [q_w], dense).columns(),
                k_proj = weight(n("k_proj"), [kv_w, hidden], weights).columns(),
                k_bias = weight(n("k_bias"), [kv_w], dense).columns(),
                v_proj = weight(n("v_proj"), [kv_w, hidden], weights).columns(),
                v_bias = weight(n("v_bias"), [kv_w], dense).columns(),
                o_proj = weight(n("o_proj"), [hidden, q_w], weights).rows(),
                o_bias = weight(n("o_bias"), [hidden], dense),
                sinks = weight(n("attn_sinks"), [d.q_heads], dense).columns(),
                kv = "kv.{}".format(l),
            ),
            attn_norm = norm("attn_norm", hidden),
            attn_norm_eps = d.norm_eps,
            mlp_norm = norm("mlp_norm", hidden),
            mlp_norm_eps = d.norm_eps,
            mlp = struct(
                experts = d.experts,
                top_k = d.top_k,
                inter = iw,
                swiglu_limit = d.swiglu_limit,
                swiglu_alpha = d.swiglu_alpha,
                router = weight(n("router"), [d.experts, hidden], router),
                router_bias = weight(n("router_bias"), [d.experts], dense),
                gate_up = weight(n("expert_gate_up_bank"), [d.experts, 2 * iw, hidden], experts).bank([iw, iw]),
                gate_up_bias = weight(n("expert_gate_up_bias"), [d.experts, 2 * iw], dense).bank([iw, iw]),
                down = weight(n("expert_down_bank"), [d.experts, hidden, iw], experts).rows(),
                down_bias = weight(n("expert_down_bias"), [d.experts, hidden], dense),
            ),
            lora_a = lora_a,
            lora_b = lora_b,
        )

    head_ = weight("lm_head", [d.vocab, hidden], weights)
    if env("PIE_NO_VOCAB_SHARD") == None:
        head_ = head_.packed([d.vocab])
    return struct(
        hidden = hidden,
        vocab = d.vocab,
        q_heads = d.q_heads,
        kv_heads = d.kv_heads,
        head_dim = hd,
        window = d.window,
        kv = deploy.kv,
        embed = weight("embed", [d.vocab, hidden], weights),
        head = head_,
        layers = [layer(l) for l in range(d.layers)],
        final_norm = weight("final_norm", [hidden], dense),
        final_norm_eps = d.norm_eps,
        dflash = declare(DFLASH_20B, "aux", hidden, d.vocab, d.norm_eps, weights, dense)
            if deploy.drafter == "dflash" else None,
    )
