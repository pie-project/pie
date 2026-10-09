# A Qwen3 dense decoder read as a text encoder, under `te.`: its token
# embedding and its layers, each attending through its own kv cache.

QWEN3_4B = struct(
    hidden = 2560,
    vocab = 151936,
    q_heads = 32,
    kv_heads = 8,
    head_dim = 128,
    inter = 9728,
    theta = 1000000.0,
    eps = 1e-6,
)

def encoder(c, layers, banks, sharded = False):
    """The encoder of config `c` through its first `layers` layers; with
    `sharded`, its attention and mlp split across ranks."""
    dense = compute(banks)
    hidden, hd, inter = c.hidden, c.head_dim, c.inter

    def columns(w, heads = None):
        if not sharded:
            return w
        return w.columns() if heads == None else w.columns(heads = heads)

    def rows(w):
        return w.rows() if sharded else w

    def layer(l):
        n = lambda s: "te.layer.{}.{}".format(l, s)
        return struct(
            attn_norm = weight(n("attn_norm"), [hidden], dense),
            q = columns(weight(n("q"), [c.q_heads * hd, hidden], banks)),
            k = columns(weight(n("k"), [c.kv_heads * hd, hidden], banks), c.kv_heads),
            v = columns(weight(n("v"), [c.kv_heads * hd, hidden], banks), c.kv_heads),
            o = rows(weight(n("o"), [hidden, c.q_heads * hd], banks)),
            q_norm = weight(n("q_norm"), [hd], dense),
            k_norm = weight(n("k_norm"), [hd], dense),
            mlp_norm = weight(n("mlp_norm"), [hidden], dense),
            gate_up = weight(n("gate_up"), [2 * inter, hidden], banks).packed([inter, inter]),
            down = rows(weight(n("down"), [hidden, inter], banks)),
            kv = "te.kv.{}".format(l),
        )

    return struct(
        hidden = hidden,
        vocab = c.vocab,
        q_heads = c.q_heads,
        kv_heads = c.kv_heads,
        head_dim = hd,
        inter = inter,
        theta = c.theta,
        eps = c.eps,
        sm_scale = f32(1.0 / f32(sqrt(hd))),
        embed = weight("te.embed", [c.vocab, hidden], banks),
        layers = [layer(l) for l in range(layers)],
    )
