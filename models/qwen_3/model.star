# The weights of Qwen 3.5 / 3.6 / 3.8: gated-delta (GDN) mixers with a gated
# attention every few layers, dense or routed MLPs, a vision tower, an MTP or
# EAGLE draft head or a DFlash-family block drafter; Ternary-Bonsai's
# rotation signs; and the small geometries the engines' tests serve.

load("//lib/dflash/model.star", dflash_declare = "declare", dflash_head = "head", "markov", "selector")

ADAPTERS = struct(slots = 8, rank = 16)

# The Bonsai online-Hadamard rotation: a sign diagonal per rotated input width.
BONSAI_WIDTHS = struct(hidden = 5120, ssm = 6144, ffn_down = 17408)

def sign_name(width):
    """The fork's on-device name of a width's sign diagonal."""
    return "prism.hadamard.signs.{}".format(width)

SLIDING = 2048

HEADS = {
    "qwen36-27b-dflash": dflash_head(
        taps = [1, 16, 31, 46, 61], windows = [SLIDING] * 4 + [None], q_heads = 32, kv_heads = 8,
        head_dim = 128, inter = 17408, theta = 10000000.0, block = 16, mask_token = 248070,
    ),
    "qwen38-27b-dflash2": dflash_head(
        taps = [5, 19, 33, 47, 61], windows = [SLIDING] * 5, q_heads = 32, kv_heads = 8,
        head_dim = 128, inter = 17408, theta = 10000000.0, block = 8, mask_token = 248070,
        conv = struct(taps = 2, group = 16), readout = selector(rank = 256, top_k = 16),
    ),
    "qwen38-27b-dspark": dflash_head(
        taps = [1, 16, 31, 46, 61], windows = [None] * 5, q_heads = 32, kv_heads = 8,
        head_dim = 128, inter = 17408, theta = 10000000.0, block = 15, mask_token = 248200,
        proposals_from = 0, readout = markov(rank = 256, top_k = 16),
    ),
    "qwen36-35b-a3b-dflash": dflash_head(
        taps = [1, 6, 11, 16, 22, 27, 32, 37], windows = [4096] * 5 + [None], q_heads = 32,
        kv_heads = 8, head_dim = 128, inter = 6144, theta = 10000000.0, block = 16, mask_token = 248077,
    ),
    "qwen35-d9b-dflash": dflash_head(
        taps = [1, 5, 9, 13, 17, 21, 25, 29], windows = [4096] * 5 + [None], q_heads = 32,
        kv_heads = 8, head_dim = 128, inter = 12288, theta = 10000000.0, block = 16, mask_token = 248077,
    ),
}

def tower_qwen35():
    return struct(depth = 12, hidden = 768, heads = 12, inter = 3072, patch_width = 1536, merge = 2,
                  positions = 2304, out_hidden = 1024, theta = 10000.0, norm_eps = 1e-6, taps = 4)

def tower_qwen36():
    return struct(depth = 27, hidden = 1152, heads = 16, inter = 4304, patch_width = 1536, merge = 2,
                  positions = 2304, out_hidden = 5120, theta = 10000.0, norm_eps = 1e-6, taps = 4)

def dims(**d):
    base = dict(theta = 10000000.0, conv_kernel = 4, norm_eps = 1e-6, tower = None, draft = None,
                dflash_head = None, rotate_kv = False)
    base.update(d)
    return base

def dense(inter):
    return struct(routed = False, inter = inter)

def routed(experts, top_k, inter, shared_inter):
    return struct(routed = True, experts = experts, top_k = top_k, inter = inter, shared_inter = shared_inter)

def a3b(**over):
    d = dims(hidden = 2048, layers = 40, attn_every = 4, q_heads = 16, kv_heads = 2, head_dim = 256,
             rotary_dim = 64, k_heads = 16, v_heads = 32, k_dim = 128, v_dim = 128,
             mlp = routed(256, 8, 512, 512), vocab = 248320, tied = False)
    d.update(over)
    return d

def d27b(**over):
    d = dims(hidden = 5120, layers = 64, attn_every = 4, q_heads = 24, kv_heads = 4, head_dim = 256,
             rotary_dim = 64, k_heads = 16, v_heads = 48, k_dim = 128, v_dim = 128,
             mlp = dense(17408), vocab = 248320, tied = False)
    d.update(over)
    return d

def d0_8b(**over):
    d = dims(hidden = 1024, layers = 24, attn_every = 4, q_heads = 8, kv_heads = 2, head_dim = 256,
             rotary_dim = 64, k_heads = 16, v_heads = 16, k_dim = 128, v_dim = 128,
             mlp = dense(3584), vocab = 248320, tied = True)
    d.update(over)
    return d

def micro_text(head_dim, rotate_kv):
    return dims(hidden = 128, layers = 2, attn_every = 1, q_heads = 4, kv_heads = 2, head_dim = head_dim,
                rotary_dim = 16, k_heads = 4, v_heads = 4, k_dim = head_dim, v_dim = head_dim,
                mlp = dense(256), vocab = 256, tied = True, rotate_kv = rotate_kv)

FIXED = {
    "qwen35-d2b": dims(hidden = 2048, layers = 24, attn_every = 4, q_heads = 8, kv_heads = 2,
                       head_dim = 256, rotary_dim = 64, k_heads = 16, v_heads = 16, k_dim = 128,
                       v_dim = 128, mlp = dense(6144), vocab = 248320, tied = True),
    "qwen35-d3b": dims(hidden = 2048, layers = 24, attn_every = 4, q_heads = 16, kv_heads = 2,
                       head_dim = 256, rotary_dim = 64, k_heads = 16, v_heads = 32, k_dim = 128,
                       v_dim = 128, mlp = dense(8192), vocab = 151936, tied = True),
    "qwen35-d4b": dims(hidden = 2560, layers = 32, attn_every = 4, q_heads = 16, kv_heads = 4,
                       head_dim = 256, rotary_dim = 64, k_heads = 16, v_heads = 32, k_dim = 128,
                       v_dim = 128, mlp = dense(9216), vocab = 248320, tied = True),
    "qwen35-a3b": a3b(),
    "qwen35-tiny": dims(hidden = 256, layers = 4, attn_every = 4, q_heads = 4, kv_heads = 2,
                        head_dim = 64, rotary_dim = 32, k_heads = 4, v_heads = 8, k_dim = 128,
                        v_dim = 128, mlp = dense(512), vocab = 248320, tied = True),
    "qwen36-35b-a3b-mini": a3b(layers = 5, mlp = routed(16, 8, 512, 512)),
    "qwen36-35b-a3b-mini64": a3b(layers = 5, mlp = routed(64, 8, 512, 512)),
    "qwen3-a3b-micro": dims(hidden = 512, layers = 4, attn_every = 1, q_heads = 8, kv_heads = 2,
                            head_dim = 64, rotary_dim = 64, k_heads = 8, v_heads = 8, k_dim = 64,
                            v_dim = 64, mlp = routed(32, 4, 128, 128), vocab = 2048, tied = True),
    "qwen3-a3b-uncached-bank": dims(hidden = 2048, layers = 1, attn_every = 1, q_heads = 8,
                                    kv_heads = 2, head_dim = 64, rotary_dim = 64, k_heads = 8,
                                    v_heads = 8, k_dim = 64, v_dim = 64,
                                    mlp = routed(32, 4, 512, 512), vocab = 2048, tied = True),
    "qwen3-micro-text": micro_text(64, False),
    "qwen3-micro-text-rotated": micro_text(64, True),
    "qwen3-micro-text-hd128": micro_text(128, False),
    "qwen3-micro-text-hd128-rotated": micro_text(128, True),
    "qwen3-micro-text-hd256": micro_text(256, False),
    "qwen3-micro-text-hd256-rotated": micro_text(256, True),
    "qwen36-27b-bonsai": d27b(),
}

def drafted(id, deploy):
    """The dims `id` builds for `deploy`'s parts and drafter."""
    vision = "vision" in deploy.parts
    drafter = deploy.drafter

    def refuse():
        fail("{} does not ship {}".format(id, deploy))

    if id in ["qwen36-27b", "qwen38-27b"]:
        tower = tower_qwen36() if vision else None
        if drafter == None:
            return d27b(tower = tower)
        if drafter == "mtp":
            return d27b(tower = tower, draft = "mtp")
        if vision:
            refuse()
        if id == "qwen36-27b" and drafter == "dflash":
            return d27b(draft = "dflash", dflash_head = HEADS["qwen36-27b-dflash"])
        if id == "qwen38-27b" and drafter == "dflash2":
            return d27b(draft = "dflash2", dflash_head = HEADS["qwen38-27b-dflash2"])
        if id == "qwen38-27b" and drafter == "dspark":
            return d27b(draft = "dspark", dflash_head = HEADS["qwen38-27b-dspark"])
        refuse()
    if id == "qwen35-d0.8b":
        return d0_8b(tower = tower_qwen35() if vision else None, draft = drafter)
    if vision:
        refuse()
    if id == "qwen35-d9b":
        d = dims(hidden = 4096, layers = 32, attn_every = 4, q_heads = 16, kv_heads = 4, head_dim = 256,
                 rotary_dim = 64, k_heads = 16, v_heads = 32, k_dim = 128, v_dim = 128,
                 mlp = dense(12288), vocab = 248320, tied = False)
        if drafter == "dflash":
            d.update(draft = "dflash", dflash_head = HEADS["qwen35-d9b-dflash"])
        elif drafter != None:
            refuse()
        return d
    if id == "qwen36-35b-a3b":
        if drafter == None:
            return a3b()
        if drafter == "mtp":
            return a3b(draft = "mtp")
        if drafter == "dflash":
            return a3b(draft = "dflash", dflash_head = HEADS["qwen36-35b-a3b-dflash"])
        refuse()
    if drafter != None:
        refuse()
    return dict(FIXED[id])

def banks(prefix, hidden, dense):
    return (
        weight(prefix + ".lora_a", [ADAPTERS.slots, ADAPTERS.rank, hidden], dense).registered(),
        weight(prefix + ".lora_b", [ADAPTERS.slots, hidden, ADAPTERS.rank], dense).registered(),
    )

def gated_attn(w, d, prefix, kv):
    n = lambda s: "{}.{}".format(prefix, s)
    dense = compute(w)
    hd = d.head_dim
    return struct(
        rotary_dim = d.rotary_dim,
        theta = d.theta,
        sm_scale = f32(1.0 / f32(sqrt(hd))),
        qg_proj = weight(n("qg_proj"), [2 * d.q_heads * hd, d.hidden], w).columns(),
        k_proj = weight(n("k_proj"), [d.kv_heads * hd, d.hidden], w).columns(),
        v_proj = weight(n("v_proj"), [d.kv_heads * hd, d.hidden], w).columns(),
        o_proj = weight(n("o_proj"), [d.hidden, d.q_heads * hd], w).rows(),
        q_norm = weight(n("q_norm"), [hd], dense),
        q_norm_eps = d.norm_eps,
        k_norm = weight(n("k_norm"), [hd], dense),
        k_norm_eps = d.norm_eps,
        kv = kv,
    )

def dense_mlp(w, hidden, inter, prefix):
    n = lambda s: "{}.{}".format(prefix, s)
    return struct(
        routed = False,
        gate_up = weight(n("gate_up"), [2 * inter, hidden], w).packed([inter, inter]),
        down = weight(n("down"), [hidden, inter], w).rows(),
        inter = inter,
    )

def qkv_width(k_heads, v_heads, k_dim, v_dim):
    return 2 * k_heads * k_dim + v_heads * v_dim

def layout(id, deploy):
    if len(deploy.weights) != 1:
        fail("{} stores its weights at one dtype, not {}".format(id, deploy.weights))
    w = deploy.weights[0]
    d = struct(**drafted(id, deploy))
    bonsai = id == "qwen36-27b-bonsai"
    dense = compute(w)
    gate = dtype.u8g64 if w == dtype.u4g64 else w
    proj = dtype.u4g64tiled if w == dtype.u4g64 else w
    hidden = d.hidden
    attn_at = lambda l: l % d.attn_every == d.attn_every - 1

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        norm = lambda s, dim: weight(n(s), [dim], dense)
        lora_a, lora_b = banks("layer.{}".format(l), hidden, dense)
        if attn_at(l):
            mixer = struct(attn = True, a = gated_attn(proj, d, "layer.{}".format(l), "kv.{}".format(l)))
        else:
            k_w = d.k_heads * d.k_dim
            v_w = d.v_heads * d.v_dim
            qkv = qkv_width(d.k_heads, d.v_heads, d.k_dim, d.v_dim)
            # Ternary-Bonsai stores the GDN beta/alpha projections in bf16,
            # not ternary: they are served as stored.
            ba = dense if bonsai else proj
            mixer = struct(attn = False, g = struct(
                k_heads = d.k_heads,
                v_heads = d.v_heads,
                k_dim = d.k_dim,
                v_dim = d.v_dim,
                conv_kernel = d.conv_kernel,
                in_qkvz = weight(n("in_qkvz"), [qkv + v_w, hidden], proj).packed([k_w, k_w, v_w, v_w]),
                in_ba = weight(n("in_ba"), [2 * d.v_heads, hidden], ba).packed([d.v_heads, d.v_heads]),
                conv = weight(n("conv"), [qkv, d.conv_kernel], dense).packed([k_w, k_w, v_w]),
                dt_bias = weight(n("dt_bias"), [d.v_heads], dense).columns(),
                a_log = weight(n("a_log"), [d.v_heads], dtype.f32).columns(),
                norm = weight(n("gdn_norm"), [d.v_dim], dtype.f32),
                norm_eps = d.norm_eps,
                out_proj = weight(n("out_proj"), [hidden, v_w], proj).rows(),
                conv_state = "conv.{}".format(l),
                delta_state = "delta.{}".format(l),
            ))
        if d.mlp.routed:
            m = d.mlp
            mlp = struct(
                routed = True,
                router = weight(n("router"), [m.experts, hidden], gate),
                gate_up = weight(n("experts_gate_up"), [m.experts, 2 * m.inter, hidden], w).bank([m.inter, m.inter]),
                down = weight(n("experts_down"), [m.experts, hidden, m.inter], w).rows(),
                shared_gate_up = weight(n("shared_gate_up"), [2 * m.shared_inter, hidden], proj).packed([m.shared_inter, m.shared_inter]),
                shared_down = weight(n("shared_down"), [hidden, m.shared_inter], proj).rows(),
                shared_gate = weight(n("shared_gate"), [1, hidden], gate),
                experts = m.experts,
                top_k = m.top_k,
                inter = m.inter,
                shared_inter = m.shared_inter,
            )
        else:
            mlp = dense_mlp(proj, hidden, d.mlp.inter, "layer.{}".format(l))
        return struct(
            mixer = mixer,
            mixer_norm = norm("mixer_norm", hidden),
            mixer_norm_eps = d.norm_eps,
            mlp_norm = norm("mlp_norm", hidden),
            mlp_norm_eps = d.norm_eps,
            mlp = mlp,
            lora_a = lora_a,
            lora_b = lora_b,
        )

    tower = None
    if d.tower != None:
        t = d.tower
        if t.out_hidden != hidden:
            fail("a tower's out_hidden_size is the trunk's width")
        th = t.hidden
        ti = t.inter
        merged = t.merge * t.merge * th
        head_dim = t.hidden // t.heads
        tn = lambda s: "visual." + s
        plane = lambda s, dims: weight(tn(s), dims, dense)
        vec1 = lambda s, length: weight(tn(s), [length], dense)

        def block(l):
            b = lambda s: "block.{}.{}".format(l, s)
            return struct(
                norm1 = vec1(b("norm1"), th),
                norm1_bias = vec1(b("norm1_bias"), th),
                qkv = plane(b("qkv"), [3 * th, th]),
                qkv_bias = vec1(b("qkv_bias"), 3 * th),
                proj = plane(b("proj"), [th, th]),
                proj_bias = vec1(b("proj_bias"), th),
                norm2 = vec1(b("norm2"), th),
                norm2_bias = vec1(b("norm2_bias"), th),
                fc1 = plane(b("fc1"), [ti, th]),
                fc1_bias = vec1(b("fc1_bias"), ti),
                fc2 = plane(b("fc2"), [th, ti]),
                fc2_bias = vec1(b("fc2_bias"), th),
            )

        tower = struct(
            hidden = t.hidden,
            heads = t.heads,
            head_dim = head_dim,
            merge = t.merge,
            patch_width = t.patch_width,
            taps = t.taps,
            positions = t.positions,
            theta = t.theta,
            norm_eps = t.norm_eps,
            sm_scale = f32(1.0 / f32(sqrt(head_dim))),
            patch_embed = plane("patch_embed", [th, t.patch_width]),
            patch_embed_bias = vec1("patch_embed_bias", th),
            pos_embed = plane("pos_embed", [t.positions, th]),
            blocks = [block(l) for l in range(t.depth)],
            merger = struct(
                norm = vec1("merger_norm", th),
                norm_bias = vec1("merger_norm_bias", th),
                fc1 = plane("merger_fc1", [merged, merged]),
                fc1_bias = vec1("merger_fc1_bias", merged),
                fc2 = plane("merger_fc2", [hidden, merged]),
                fc2_bias = vec1("merger_fc2_bias", hidden),
            ),
        )

    blocks = ["dflash", "dflash2", "dspark"]
    mtp = None
    if d.draft != None and d.draft not in blocks:
        inter = d.mlp.inter
        p = "mtp" if d.draft == "mtp" else "aux"
        pn = lambda s: "{}.{}".format(p, s)
        mtp = struct(
            recipe = d.draft,
            prefix = p,
            pre_fc = struct(
                embedding = weight(pn("pre_fc_norm_embedding"), [hidden], dense),
                hidden = weight(pn("pre_fc_norm_hidden"), [hidden], dense),
                eps = d.norm_eps,
            ) if d.draft == "mtp" else None,
            fc_embed = weight(pn("fc_embed"), [hidden, hidden], w),
            fc_hidden = weight(pn("fc_hidden"), [hidden, hidden], w),
            mixer_norm = weight(pn("mixer_norm"), [hidden], dense),
            mixer_norm_eps = d.norm_eps,
            attn = gated_attn(w, d, p, "kv.mtp"),
            mlp_norm = weight(pn("mlp_norm"), [hidden], dense),
            mlp_norm_eps = d.norm_eps,
            mlp = dense_mlp(w, hidden, inter, p),
            norm = weight(pn("norm"), [hidden], dense) if d.draft == "mtp" else None,
            norm_eps = d.norm_eps,
        )
    dflash = None
    if d.draft in blocks:
        dflash = dflash_declare(d.dflash_head, "aux", hidden, d.vocab, d.norm_eps, w, dense)

    # Ternary-Bonsai's token embedding is gathered whole, which no ternary
    # kernel does: it is served decoded.
    embed = weight("embed", [d.vocab, hidden], dense if bonsai else w)
    head = None
    if not d.tied:
        head = weight("lm_head", [d.vocab, hidden], w)
        if env("PIE_NO_VOCAB_SHARD") == None:
            head = head.columns()
    signs = None
    if bonsai:
        sign = lambda width: weight(sign_name(width), [1, width], dense)
        signs = struct(
            hidden = sign(BONSAI_WIDTHS.hidden),
            ssm = sign(BONSAI_WIDTHS.ssm),
            ffn_down = sign(BONSAI_WIDTHS.ffn_down),
        )
    return struct(
        hidden = hidden,
        vocab = d.vocab,
        q_heads = d.q_heads,
        kv_heads = d.kv_heads,
        head_dim = d.head_dim,
        rotate_kv = d.rotate_kv,
        kv = deploy.kv,
        embed = embed,
        head = head,
        layers = [layer(l) for l in range(d.layers)],
        final_norm = weight("final_norm", [hidden], dense),
        final_norm_eps = d.norm_eps,
        tower = tower,
        mtp = mtp,
        dflash = dflash,
        bonsai = signs,
    )
