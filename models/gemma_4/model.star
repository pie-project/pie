# The weights of Gemma 4: sliding-window layers with a global one every
# sixth, attention-only layers that borrow an earlier layer's kv in E4B's
# shared tail, per-layer embeddings (E4B), routed experts beside the dense
# MLP (26B-A4B), a vision tower, and the drafters each pairs with.

load("//lib/adapters/model.star", "banks")
load("//lib/dflash/model.star", dflash_head = "head", dflash_declare = "declare")

SLIDING = 0
GLOBAL = 1

SELF_COND_TAPS = 64
CANVAS = 256

ASSISTANT_HIDDEN = 1024
ASSISTANT_INTER = 8192
ASSISTANT_READINGS = [SLIDING, SLIDING, SLIDING, GLOBAL]

def trunk(hidden, layers, q_heads, kv_heads, global_kv_heads, inter, window, shared_tail = None, ple_dim = None, moe = None):
    return struct(
        hidden = hidden, layers = layers, q_heads = q_heads, kv_heads = kv_heads,
        global_kv_heads = global_kv_heads, inter = inter, window = window,
        shared_tail = shared_tail, ple_dim = ple_dim, moe = moe,
        full_every = 6, head_dim = 256, global_head_dim = 512, global_rotary_dim = 128,
        theta_local = 10000.0, theta_global = 1000000.0, sm_scale = 1.0, vocab = 262144,
        softcap = 30.0, norm_eps = 1e-6,
    )

def e4b(layers):
    owned = 42 - 18
    return trunk(hidden = 2560, layers = layers, q_heads = 8, kv_heads = 2, global_kv_heads = 2, inter = 10240,
                 window = 512, shared_tail = layers - owned if layers > owned else None, ple_dim = 256)

A4B = trunk(hidden = 2816, layers = 30, q_heads = 16, kv_heads = 8, global_kv_heads = 2, inter = 2112,
            window = 1024, moe = struct(experts = 128, top_k = 8, inter = 704))

TRUNKS = {
    "gemma4-26b-a4b": A4B,
    "gemma4-31b": trunk(hidden = 5376, layers = 60, q_heads = 32, kv_heads = 16, global_kv_heads = 4,
                        inter = 21504, window = 1024),
    "gemma4-e4b": e4b(42),
    "gemma4-e4b-mini-l1": e4b(1),
    "gemma4-e4b-mini-l6": e4b(6),
    "gemma4-e4b-mini-l24": e4b(24),
    "gemma4-e4b-mini-l30": e4b(30),
    "gemma4-e4b-mini-l36": e4b(36),
    "diffusiongemma-26b-a4b": A4B,
}

def vision_tower(depth, hidden, heads, inter, clipped, standardize):
    return struct(depth = depth, hidden = hidden, heads = heads, inter = inter, clipped = clipped,
                  standardize = standardize, patch_width = 3 * 16 * 16, pool = 3, positions = 10240,
                  theta = 100.0, norm_eps = 1e-6, sm_scale = 1.0)

WIDE_TOWER = vision_tower(depth = 27, hidden = 1152, heads = 16, inter = 4304, clipped = False, standardize = True)

TOWERS = {
    "gemma4-26b-a4b": WIDE_TOWER,
    "gemma4-31b": WIDE_TOWER,
    "gemma4-e4b": vision_tower(depth = 16, hidden = 768, heads = 12, inter = 3072, clipped = True, standardize = False),
}

DFLASH_26B_A4B = dflash_head(
    taps = [1, 6, 11, 17, 22, 27],
    windows = [2048, 2048, 2048, 2048, None],
    q_heads = 32,
    kv_heads = 8,
    head_dim = 128,
    inter = 5632,
    theta = 1000000.0,
    block = 16,
    mask_token = 4,
)

def layout(id, deploy):
    """`id`'s weights as `deploy` stores them: the trunk's at its first
    dtype, the experts' at the second if it has one, DiffusionGemma's
    self-conditioning block's at the third."""
    d = TRUNKS[id]
    weights = deploy.weights
    drafter = deploy.drafter
    vision = "vision" in deploy.parts
    self_cond = id == "diffusiongemma-26b-a4b"
    if self_cond:
        if len(weights) not in ([3] if "selfcond" in deploy.parts else [1, 2]):
            fail("{} does not ship {}".format(id, deploy))
    elif len(weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, weights))
    if vision and drafter != None:
        fail("{} does not ship {}".format(id, deploy))
    w = weights[0]
    xw = weights[1] if len(weights) > 1 else w
    sw = weights[2] if len(weights) > 2 else w
    dense = compute(w)
    gate = dtype.u8g64 if w == dtype.u4g64 else w
    proj = dtype.u4g64tiled if w == dtype.u4g64 else w
    hidden = d.hidden
    full_at = lambda l: l % d.full_every == d.full_every - 1
    shared_at = lambda l: d.shared_tail != None and l >= d.layers - d.shared_tail

    def owner(l):
        if not shared_at(l):
            return l
        for s in range(l - 1, -1, -1):
            if not shared_at(s) and full_at(s) == full_at(l):
                return s
        fail("layer {} borrows its kv cache and none of the layers before the shared tail is of its kind".format(l))

    sliding = struct(head_dim = d.head_dim, kv_heads = d.kv_heads, window = d.window, theta = d.theta_local)
    glob = struct(head_dim = d.global_head_dim, kv_heads = d.global_kv_heads,
                  rotary_dim = d.global_rotary_dim, theta = d.theta_global)

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        norm = lambda s, length: weight(n(s), [length], dense)
        lora_a, lora_b = banks("layer.{}".format(l), hidden, dense)
        reading = GLOBAL if full_at(l) else SLIDING
        head_dim, row_heads = (sliding.head_dim, sliding.kv_heads) if reading == SLIDING else (glob.head_dim, glob.kv_heads)
        q_w = d.q_heads * head_dim
        kv_w = row_heads * head_dim
        iw = d.inter
        if shared_at(l):
            attn_banks = struct(shared = True, q_proj = weight(n("q_proj"), [q_w, hidden], proj).columns())
        else:
            attn_banks = struct(
                shared = False,
                qkv = weight(n("qkv"), [q_w + 2 * kv_w, hidden], proj).packed([q_w, kv_w, kv_w], heads = [d.q_heads, row_heads, row_heads]),
                k_norm = norm("k_norm", head_dim),
                k_norm_eps = d.norm_eps,
            )
        moe = None
        if d.moe != None:
            mi = d.moe.inter
            moe = struct(
                router_norm = norm("router_norm", hidden),
                router_norm_eps = d.norm_eps,
                router = weight(n("router"), [d.moe.experts, hidden], gate),
                per_expert_scale = weight(n("per_expert_scale"), [d.moe.experts], dense).columns(),
                pre_ffw_norm_2 = norm("pre_ffw_norm_2", hidden),
                pre_ffw_norm_2_eps = d.norm_eps,
                post_ffw_norm_1 = norm("post_ffw_norm_1", hidden),
                post_ffw_norm_1_eps = d.norm_eps,
                post_ffw_norm_2 = norm("post_ffw_norm_2", hidden),
                post_ffw_norm_2_eps = d.norm_eps,
                gate_up = weight(n("experts_gate_up"), [d.moe.experts, 2 * mi, hidden], xw).bank([mi, mi]),
                down = weight(n("experts_down"), [d.moe.experts, hidden, mi], xw).rows(),
                experts = d.moe.experts,
                top_k = d.moe.top_k,
                inter = mi,
            )
        return struct(
            attn = struct(
                sm_scale = d.sm_scale,
                q_norm = norm("q_norm", head_dim),
                q_norm_eps = d.norm_eps,
                kv = "kv.{}".format(owner(l)),
                banks = attn_banks,
                reading = reading,
            ),
            o_proj = weight(n("o_proj"), [hidden, q_w], proj).rows(),
            attn_norm = norm("attn_norm", hidden),
            attn_norm_eps = d.norm_eps,
            post_attn_norm = norm("post_attn_norm", hidden),
            post_attn_norm_eps = d.norm_eps,
            pre_ffw_norm = norm("pre_ffw_norm", hidden),
            pre_ffw_norm_eps = d.norm_eps,
            post_ffw_norm = norm("post_ffw_norm", hidden),
            post_ffw_norm_eps = d.norm_eps,
            gate_up = weight(n("gate_up"), [2 * iw, hidden], proj).packed([iw, iw]),
            inter = iw,
            down = weight(n("down"), [hidden, iw], proj).rows(),
            scalar = weight(n("scalar"), [1], dense) if d.ple_dim == None else None,
            lora_a = lora_a,
            lora_b = lora_b,
            moe = moe,
        )

    tower = None
    if vision:
        t = TOWERS[id]
        th = t.hidden
        ti = t.inter
        head_dim = t.hidden // t.heads
        tn = lambda s: "vision." + s
        bank = lambda s, dims: weight(tn(s), dims, dense)
        vec1 = lambda s, length: weight(tn(s), [length], dense)

        def clip(s, dims):
            return struct(
                bank = bank(s, dims),
                clip = struct(
                    in_lo = vec1(s + "_in_lo", 1),
                    in_hi = vec1(s + "_in_hi", 1),
                    out_lo = vec1(s + "_out_lo", 1),
                    out_hi = vec1(s + "_out_hi", 1),
                ) if t.clipped else None,
            )

        def block(l):
            b = lambda s: "block.{}.{}".format(l, s)
            return struct(
                attn_norm = vec1(b("attn_norm"), th),
                post_attn_norm = vec1(b("post_attn_norm"), th),
                pre_ffw_norm = vec1(b("pre_ffw_norm"), th),
                post_ffw_norm = vec1(b("post_ffw_norm"), th),
                q = clip(b("q"), [th, th]),
                k = clip(b("k"), [th, th]),
                v = clip(b("v"), [th, th]),
                o = clip(b("o"), [th, th]),
                q_norm = vec1(b("q_norm"), head_dim),
                k_norm = vec1(b("k_norm"), head_dim),
                gate = clip(b("gate"), [ti, th]),
                up = clip(b("up"), [ti, th]),
                down = clip(b("down"), [th, ti]),
            )

        tower = struct(
            hidden = t.hidden,
            heads = t.heads,
            head_dim = head_dim,
            pool = t.pool,
            patch_width = t.patch_width,
            positions = 2 * t.positions,
            theta = t.theta,
            norm_eps = t.norm_eps,
            sm_scale = t.sm_scale,
            patch_embed = bank("patch_embed", [th, t.patch_width]),
            pos_embed = bank("pos_embed", [2 * t.positions, th]),
            blocks = [block(l) for l in range(t.depth)],
            projection = weight(tn("projection"), [hidden, th], w),
            std = struct(bias = vec1("std_bias", th), scale = vec1("std_scale", th)) if t.standardize else None,
        )

    # E4B's EAGLE head.
    draft = None
    if drafter == "eagle":
        hd = glob.head_dim
        q_w = d.q_heads * hd
        kv_w = glob.kv_heads * hd
        iw = d.inter
        an = lambda s: "aux." + s
        anorm = lambda s, length: weight(an(s), [length], dense)
        draft = struct(
            fc_embed = weight(an("fc_embed"), [hidden, hidden], w),
            fc_hidden = weight(an("fc_hidden"), [hidden, hidden], w),
            attn_norm = anorm("attn_norm", hidden),
            post_attn_norm = anorm("post_attn_norm", hidden),
            pre_ffw_norm = anorm("pre_ffw_norm", hidden),
            post_ffw_norm = anorm("post_ffw_norm", hidden),
            attn = struct(
                sm_scale = d.sm_scale,
                q_norm = anorm("q_norm", hd),
                q_norm_eps = d.norm_eps,
                kv = "kv.mtp",
                banks = struct(
                    shared = False,
                    qkv = weight(an("qkv"), [q_w + 2 * kv_w, hidden], w).packed([q_w, kv_w, kv_w], heads = [d.q_heads, glob.kv_heads, glob.kv_heads]),
                    k_norm = anorm("k_norm", hd),
                    k_norm_eps = d.norm_eps,
                ),
                reading = GLOBAL,
            ),
            o_proj = weight(an("o_proj"), [hidden, q_w], w).rows(),
            gate_up = weight(an("gate_up"), [2 * iw, hidden], w).packed([iw, iw]),
            inter = iw,
            down = weight(an("down"), [hidden, iw], w).rows(),
            norm_eps = d.norm_eps,
        )

    # The MTP assistant of 26B-A4B and 31B.
    assistant = None
    if drafter == "mtp":
        def last(want_full):
            for l in range(d.layers - 1, -1, -1):
                if not shared_at(l) and full_at(l) == want_full:
                    return owner(l)
            fail("the trunk has a layer of each reading")

        ah = ASSISTANT_HIDDEN
        iw = ASSISTANT_INTER

        def assistant_layer(l, reading):
            n = lambda s: "aux.layer.{}.{}".format(l, s)
            norm = lambda s, length: weight(n(s), [length], dense)
            hd = sliding.head_dim if reading == SLIDING else glob.head_dim
            q_w = d.q_heads * hd
            return struct(
                attn = struct(
                    sm_scale = d.sm_scale,
                    q_norm = norm("q_norm", hd),
                    q_norm_eps = d.norm_eps,
                    kv = "kv.{}".format(last(reading == GLOBAL)),
                    banks = struct(shared = True, q_proj = weight(n("q_proj.weight"), [q_w, ah], w)),
                    reading = reading,
                ),
                o_proj = weight(n("o_proj.weight"), [ah, q_w], w),
                attn_norm = norm("attn_norm", ah),
                post_attn_norm = norm("post_attn_norm", ah),
                pre_ffw_norm = norm("pre_ffw_norm", ah),
                post_ffw_norm = norm("post_ffw_norm", ah),
                gate_up = weight(n("gate_up.weight"), [2 * iw, ah], w).packed([iw, iw]),
                inter = iw,
                down = weight(n("down.weight"), [ah, iw], w),
                scalar = norm("scalar", 1),
            )

        assistant = struct(
            pre_embed = weight("aux.pre_embed.weight", [ah, hidden], w),
            pre_hidden = weight("aux.pre_hidden.weight", [ah, hidden], w),
            post = weight("aux.post.weight", [hidden, ah], w),
            embed = weight("aux.embed.weight", [d.vocab, ah], w),
            norm = weight("aux.final_norm", [ah], dense),
            norm_eps = d.norm_eps,
            layers = [assistant_layer(l, r) for l, r in enumerate(ASSISTANT_READINGS)],
        )

    embed = weight("embed", [d.vocab, hidden], dense if self_cond else w)
    if not self_cond:
        embed = embed.packed([d.vocab])

    ple = None
    if d.ple_dim != None:
        p = d.ple_dim
        ple = struct(
            dim = p,
            model_proj = weight("ple.model_proj", [d.layers * p, hidden], w),
            model_norm = weight("ple.model_norm", [p], dense),
            model_norm_eps = d.norm_eps,
            per_layer = [struct(
                table = weight("layer.{}.ple_table".format(l), [d.vocab, p], w),
                gate = weight("layer.{}.ple_gate".format(l), [p, hidden], w),
                proj = weight("layer.{}.ple_proj".format(l), [hidden, p], w),
                norm = weight("layer.{}.ple_norm".format(l), [hidden], dense),
                norm_eps = d.norm_eps,
                scalar = weight("layer.{}.ple_scalar".format(l), [1], dense),
            ) for l in range(d.layers)],
        )

    self_cond_block = None
    if self_cond:
        iw = d.inter
        self_cond_block = struct(
            taps = SELF_COND_TAPS,
            pre_norm = weight("self_cond.pre_norm", [hidden], dense),
            norm_eps = d.norm_eps,
            gate_up = weight("self_cond.gate_up", [2 * iw, hidden], sw).packed([iw, iw]),
            inter = iw,
            down = weight("self_cond.down", [hidden, iw], sw).rows(),
        )

    return struct(
        hidden = hidden,
        vocab = d.vocab,
        q_heads = d.q_heads,
        sliding = sliding,
        glob = glob,
        tower = tower,
        kv = deploy.kv,
        softcap = d.softcap,
        embed = embed,
        ple = ple,
        layers = [layer(l) for l in range(d.layers)],
        final_norm = weight("final_norm", [hidden], dense),
        final_norm_eps = d.norm_eps,
        draft = draft,
        assistant = assistant,
        self_cond = self_cond_block,
        dflash = dflash_declare(DFLASH_26B_A4B, "aux", hidden, d.vocab, d.norm_eps, w, dense) if drafter == "dflash" else None,
    )

def diffusion(m):
    if m.self_cond == None:
        return None
    return canvas(canvas = CANVAS, hidden = m.hidden, self_cond_taps = SELF_COND_TAPS)
