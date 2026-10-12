# The weights of a DFlash block drafter: a few transformer blocks that draft
# a block of tokens at once from taps of the trunk's residual stream.
#
# A head states the drafter's shape: `taps` (the trunk layers it reads),
# `windows` (one per block, `None` attending the whole masked block both
# ways), its attention and MLP widths, the `block` it drafts and the
# `mask_token` that fills it, an optional dynamic `conv` around each block's
# branches, and its `readout`: "argmax", or a `selector` or `markov` walk
# over the top candidates.

def head(
        taps,
        windows,
        q_heads,
        kv_heads,
        head_dim,
        inter,
        theta,
        block,
        mask_token,
        proposals_from = 1,
        conv = None,
        readout = "argmax",
        attn_bias = False):
    return struct(
        taps = taps,
        windows = windows,
        q_heads = q_heads,
        kv_heads = kv_heads,
        head_dim = head_dim,
        inter = inter,
        theta = theta,
        block = block,
        mask_token = mask_token,
        proposals_from = proposals_from,
        conv = conv,
        readout = readout,
        attn_bias = attn_bias,
    )

def selector(rank, top_k):
    return struct(kind = "selector", rank = rank, top_k = top_k)

def markov(rank, top_k):
    return struct(kind = "markov", rank = rank, top_k = top_k)

def declare(head, prefix, hidden, vocab, norm_eps, w, dense):
    """The drafter `head` describes, its weights under `prefix`, over a
    trunk of `hidden` and `vocab` whose norms take `norm_eps`, its banks
    stored as `w` and its vectors as `dense`."""
    n = lambda s: "{}.{}".format(prefix, s)
    dq, dkv, hd = head.q_heads, head.kv_heads, head.head_dim
    inter = head.inter

    def conv(l, which):
        c = head.conv
        if c == None:
            return None
        return struct(
            base = weight("{}.layers.{}.{}.base_kernel".format(prefix, l, which), [2 * c.taps, hidden], dense),
            proj = weight(
                "{}.layers.{}.{}.kernel_projection".format(prefix, l, which),
                [2 * c.taps * (hidden // c.group), hidden],
                w,
            ).columns(),
            taps = c.taps,
            group = c.group,
        )

    def block(l, window):
        b = lambda s: "{}.layers.{}.{}".format(prefix, l, s)
        bias = lambda s, width: weight(b(s), [width], dense) if head.attn_bias else None
        return struct(
            mixer_norm = weight(b("mixer_norm"), [hidden], dense),
            mixer_norm_eps = norm_eps,
            attn = struct(
                q_heads = dq,
                kv_heads = dkv,
                head_dim = hd,
                rotary_dim = hd,
                theta = head.theta,
                sm_scale = f32(1.0 / f32(sqrt(hd))),
                q_proj = weight(b("q_proj"), [dq * hd, hidden], w).columns(),
                k_proj = weight(b("k_proj"), [dkv * hd, hidden], w).columns(heads = dkv),
                v_proj = weight(b("v_proj"), [dkv * hd, hidden], w).columns(heads = dkv),
                o_proj = weight(b("o_proj"), [hidden, dq * hd], w).rows(),
                q_norm = weight(b("q_norm"), [hd], dense),
                q_norm_eps = norm_eps,
                k_norm = weight(b("k_norm"), [hd], dense),
                k_norm_eps = norm_eps,
                q_bias = bias("q_bias", dq * hd),
                k_bias = bias("k_bias", dkv * hd),
                v_bias = bias("v_bias", dkv * hd),
                o_bias = bias("o_bias", hidden),
                kv = "kv.dflash.{}".format(l),
            ),
            mlp_norm = weight(b("mlp_norm"), [hidden], dense),
            mlp_norm_eps = norm_eps,
            mlp = struct(
                gate_up = weight(b("gate_up"), [2 * inter, hidden], w).packed([inter, inter]),
                down = weight(b("down"), [hidden, inter], w).rows(),
                inter = inter,
            ),
            window = window,
            attn_conv = conv(l, "attention_conv"),
            mlp_conv = conv(l, "mlp_conv"),
        )

    codebook = lambda s, rank: weight(n(s), [vocab, rank], dense)
    readout = head.readout
    if readout == "argmax":
        chooser = None
    elif readout.kind == "selector":
        chooser = struct(
            kind = "selector",
            hidden_projection = weight(
                n("candidate_selector.hidden_projection"),
                [readout.rank, hidden],
                w,
            ).columns(),
            pred = codebook("candidate_selector.predecessor_codebook", readout.rank),
            succ = codebook("candidate_selector.successor_codebook", readout.rank),
            top_k = readout.top_k,
        )
    else:
        chooser = struct(
            kind = "markov",
            hidden_projection = None,
            pred = codebook("markov_w1", readout.rank),
            succ = codebook("markov_w2", readout.rank),
            top_k = readout.top_k,
        )
    return struct(
        taps = head.taps,
        fc = [weight(n("fc_tap{}".format(i)), [hidden, hidden], w) for i in range(len(head.taps))],
        hidden_norm = weight(n("hidden_norm"), [hidden], dense),
        hidden_norm_eps = norm_eps,
        blocks = [block(l, window) for l, window in enumerate(head.windows)],
        norm = weight(n("norm"), [hidden], dense),
        norm_eps = norm_eps,
        block = head.block,
        mask_token = head.mask_token,
        selector = chooser,
        proposals_from = head.proposals_from,
    )
