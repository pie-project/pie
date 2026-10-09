# The weights of Qwen 3.8 Flash Next. Its residual stream is `streams`
# hyper-connected copies, mixed in and injected back at every site; three of
# every four layers are gated delta-nets and the fourth full attention with a
# block-sparse indexer; every layer routes to experts beside a shared one; an
# n-gram embedding (PLE) enriches the stream at one layer.

def gated_attn(n, d, w, kv):
    """Gated attention over `d`'s heads, its projections at `w`, caching in `kv`."""
    dense = compute(w)
    hd = d.head_dim
    return struct(
        rotary_dim = d.rotary_dim,
        theta = d.theta,
        sm_scale = f32(1.0 / f32(sqrt(hd))),
        qg_proj = weight(n("qg_proj"), [2 * d.q_heads * hd, d.hidden], w).columns(),
        k_proj = weight(n("k_proj"), [d.kv_heads * hd, d.hidden], w).columns(heads = d.kv_heads),
        v_proj = weight(n("v_proj"), [d.kv_heads * hd, d.hidden], w).columns(heads = d.kv_heads),
        o_proj = weight(n("o_proj"), [d.hidden, d.q_heads * hd], w).rows(),
        q_norm = weight(n("q_norm"), [hd], dense),
        q_norm_eps = d.norm_eps,
        k_norm = weight(n("k_norm"), [hd], dense),
        k_norm_eps = d.norm_eps,
        kv = kv,
    )

def gdn(n, d, proj, ba, dense, conv_state, delta_state):
    """A GDN mixer: its projections at `proj`, beta/alpha's at `ba`, its
    convolution and gate parameters at `dense`."""
    k_w = d.k_heads * d.k_dim
    v_w = d.v_heads * d.v_dim
    qkv = 2 * k_w + v_w
    return struct(
        k_heads = d.k_heads,
        v_heads = d.v_heads,
        k_dim = d.k_dim,
        v_dim = d.v_dim,
        qkv_width = qkv,
        conv_kernel = d.conv_kernel,
        in_qkvz = weight(n("in_qkvz"), [qkv + v_w, d.hidden], proj).packed([k_w, k_w, v_w, v_w]),
        in_ba = weight(n("in_ba"), [2 * d.v_heads, d.hidden], ba).packed([d.v_heads, d.v_heads]),
        conv = weight(n("conv"), [qkv, d.conv_kernel], dense).packed([k_w, k_w, v_w]),
        dt_bias = weight(n("dt_bias"), [d.v_heads], dense).columns(),
        a_log = weight(n("a_log"), [d.v_heads], dtype.f32).columns(),
        norm = weight(n("gdn_norm"), [d.v_dim], dtype.f32),
        norm_eps = d.norm_eps,
        out_proj = weight(n("out_proj"), [d.hidden, v_w], proj).rows(),
        conv_state = conv_state,
        delta_state = delta_state,
    )

def routed(n, hidden, m, experts, proj, gate):
    """`m.experts` experts at `experts`, `m.top_k` routed to by a router at
    `gate`, beside a shared expert at `proj` whose gate is at `gate`."""
    return struct(
        routed = True,
        router = weight(n("router"), [m.experts, hidden], gate),
        gate_up = weight(n("experts_gate_up"), [m.experts, 2 * m.inter, hidden], experts).bank([m.inter, m.inter]),
        down = weight(n("experts_down"), [m.experts, hidden, m.inter], experts).rows(),
        shared_gate_up = weight(n("shared_gate_up"), [2 * m.shared_inter, hidden], proj).packed([m.shared_inter, m.shared_inter]),
        shared_down = weight(n("shared_down"), [hidden, m.shared_inter], proj).rows(),
        shared_gate = weight(n("shared_gate"), [1, hidden], gate),
        experts = m.experts,
        top_k = m.top_k,
        inter = m.inter,
        shared_inter = m.shared_inter,
    )

LARGE = struct(depth = 27, hidden = 1152, heads = 16, inter = 4304)

PATCH_WIDTH = 1536

MERGE = 2

POSITIONS = 2304

def tower(size, out, dt):
    """`size`'s tower, merging into rows `out` wide, its weights at `dt`."""
    th = size.hidden
    ti = size.inter
    merged = MERGE * MERGE * th
    head_dim = th // size.heads
    plane = lambda s, dims: weight("visual." + s, dims, dt)
    vec = lambda s, length: weight("visual." + s, [length], dt)

    def block(l):
        b = lambda s: "block.{}.{}".format(l, s)
        return struct(
            norm1 = vec(b("norm1"), th),
            norm1_bias = vec(b("norm1_bias"), th),
            qkv = plane(b("qkv"), [3 * th, th]),
            qkv_bias = vec(b("qkv_bias"), 3 * th),
            proj = plane(b("proj"), [th, th]),
            proj_bias = vec(b("proj_bias"), th),
            norm2 = vec(b("norm2"), th),
            norm2_bias = vec(b("norm2_bias"), th),
            fc1 = plane(b("fc1"), [ti, th]),
            fc1_bias = vec(b("fc1_bias"), ti),
            fc2 = plane(b("fc2"), [th, ti]),
            fc2_bias = vec(b("fc2_bias"), th),
        )

    return struct(
        hidden = th,
        heads = size.heads,
        head_dim = head_dim,
        merge = MERGE,
        patch_width = PATCH_WIDTH,
        taps = 4,
        positions = POSITIONS,
        theta = 10000.0,
        norm_eps = 1e-6,
        sm_scale = f32(1.0 / f32(sqrt(head_dim))),
        patch_embed = plane("patch_embed", [th, PATCH_WIDTH]),
        patch_embed_bias = vec("patch_embed_bias", th),
        pos_embed = plane("pos_embed", [POSITIONS, th]),
        blocks = [block(l) for l in range(size.depth)],
        merger = struct(
            norm = vec("merger_norm", th),
            norm_bias = vec("merger_norm_bias", th),
            fc1 = plane("merger_fc1", [merged, merged]),
            fc1_bias = vec("merger_fc1_bias", merged),
            fc2 = plane("merger_fc2", [out, merged]),
            fc2_bias = vec("merger_fc2_bias", out),
        ),
    )

def ple_dims(layer, heads_per_ngram, base_vocab, split_parts):
    return struct(
        layer = layer,
        heads_per_ngram = heads_per_ngram,
        ngram = 3,
        base_vocab = base_vocab,
        divisible_by = 128,
        split_parts = split_parts,
        seed = 1234,
        conv_kernel = 4,
    )

def dims(
        hidden = 2560,
        layers = 48,
        attn_every = 4,
        q_heads = 24,
        kv_heads = 2,
        head_dim = 256,
        rotary_dim = 64,
        k_heads = 16,
        v_heads = 48,
        k_dim = 128,
        v_dim = 128,
        lowrank = 320,
        experts = 512,
        top_k = 10,
        inter = 640,
        ple = ple_dims(1, 8, 20000000, 128),
        indexer = struct(heads = 4, head_dim = 128, budget = 2048, ratio = 4),
        vocab = 248320,
        eos = 248044):
    return struct(
        hidden = hidden,
        layers = layers,
        attn_every = attn_every,
        q_heads = q_heads,
        kv_heads = kv_heads,
        head_dim = head_dim,
        rotary_dim = rotary_dim,
        theta = 10000000.0,
        k_heads = k_heads,
        v_heads = v_heads,
        k_dim = k_dim,
        v_dim = v_dim,
        conv_kernel = 4,
        streams = 4,
        lowrank = lowrank,
        experts = experts,
        top_k = top_k,
        inter = inter,
        shared_inter = inter,
        ple = ple,
        indexer = indexer,
        vocab = vocab,
        eos = eos,
        norm_eps = 1e-6,
    )

FLASH = dims()

MINI = dims(layers = 4, experts = 16, ple = ple_dims(1, 8, 1250000, 8))

MICRO = dims(
    hidden = 64,
    layers = 4,
    attn_every = 2,
    q_heads = 4,
    kv_heads = 2,
    head_dim = 64,
    rotary_dim = 16,
    k_heads = 2,
    v_heads = 4,
    k_dim = 16,
    v_dim = 16,
    lowrank = 16,
    experts = 8,
    top_k = 2,
    inter = 32,
    ple = ple_dims(2, 2, 1000, 128),
    indexer = struct(heads = 2, head_dim = 16, budget = 32, ratio = 4),
    vocab = 256,
    eos = 3,
)

DRAFT_DEPTH = 2

# The PLE's hashing constants, as the checkpoint's own code derives them:
# multipliers from SplitMix64 over the seed, and a run of primes from the
# base vocabulary up, one per head.

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15

def splitmix64(v):
    v = (v + GAMMA) & MASK
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & MASK
    return v ^ (v >> 31)

def is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, int(sqrt(v)) + 2, 2):
        if d * d > v:
            break
        if v % d == 0:
            return False
    return True

def hash_constants(p, vocab):
    half_bound = max((((1 << 63) - 1) // max(vocab, 1)) // 2, 1)
    mults = [
        2 * (splitmix64((p.seed + GAMMA * (i + 1)) & MASK) % half_bound) + 1
        for i in range(p.ngram)
    ]
    heads = (p.ngram - 1) * p.heads_per_ngram
    primes = []
    offsets = []
    total = 0
    prime = p.base_vocab - 1
    for _ in range(heads):
        prime += 1
        for _ in range(100000):
            if is_prime(prime):
                break
            prime += 1
        primes.append(prime)
        offsets.append(total)
        total += prime
    return struct(mults = mults, primes = primes, offsets = offsets, total = total)

HASHES = {
    "flash": hash_constants(FLASH.ple, FLASH.vocab),
    "mini": hash_constants(MINI.ple, MINI.vocab),
    "micro": hash_constants(MICRO.ple, MICRO.vocab),
}

def mix_of(w):
    proj = dtype.u8g64 if w == dtype.u4g64 else compute(w)
    return struct(
        embed = proj,
        head = proj,
        proj = proj,
        inject = compute(w),
        gdn_ba = compute(w),
        experts = w,
        table = dtype.u4g32 if w == dtype.u4g64 else w,
    )

MIXED_2BIT = struct(
    embed = dtype.bf16,
    head = dtype.u4g64,
    proj = dtype.u4g64,
    inject = dtype.u4g64,
    gdn_ba = dtype.u4g64,
    experts = dtype.u2g128,
    table = dtype.u4g32,
)

def layout(id, deploy):
    vision = "vision" in deploy.parts
    mtp = deploy.drafter == "mtp"
    mixed = deploy.weights == [dtype.u4g64, dtype.u2g128]
    refused = "{} does not ship {}".format(id, deploy)
    if id == "qwen38-flash-next-micro":
        if len(deploy.weights) != 1:
            fail(refused)
        return build(mix_of(deploy.weights[0]), deploy.kv, MICRO, HASHES["micro"], False, False)
    if id == "qwen38-flash-next-mini":
        if not mixed:
            fail(refused)
        return build(MIXED_2BIT, deploy.kv, MINI, HASHES["mini"], False, False)
    if len(deploy.weights) == 1 and not vision and deploy.drafter == None:
        return build(mix_of(deploy.weights[0]), deploy.kv, FLASH, HASHES["flash"], False, False)
    if mixed and (mtp or deploy.drafter == None):
        return build(MIXED_2BIT, deploy.kv, FLASH, HASHES["flash"], mtp, vision)
    fail(refused)

def build(mix, kv, d, hashes, draft, vision):
    dense = compute(mix.proj)
    proj = mix.proj
    hidden = d.hidden
    sh = d.streams * hidden

    def residual(prefix, inject):
        return struct(
            norm = weight(prefix + ".norm", [sh], dense),
            down = weight(prefix + ".down", [d.lowrank, sh], proj),
            up = weight(prefix + ".up", [sh, d.lowrank], proj),
            inject = weight(prefix + ".inject", [d.streams, sh], mix.inject) if inject else None,
            eps = d.norm_eps,
        )

    def block(n, attn, indexer, delta, experts):
        return struct(
            attn = attn,
            indexer = indexer,
            gdn = delta,
            attn_res = residual(n("attn_res"), True),
            mlp_res = residual(n("mlp_res"), True),
            mlp = routed(n, hidden, d, experts, proj, dense),
        )

    def index(n, keys):
        ix = d.indexer
        q_w = ix.heads * ix.head_dim
        return struct(
            heads = ix.heads,
            head_dim = ix.head_dim,
            ratio = ix.ratio,
            top_k = ix.budget // ix.ratio,
            norm_eps = d.norm_eps,
            qk_proj = weight(n("index_qk_proj"), [q_w + ix.head_dim, hidden], proj).packed([q_w, ix.head_dim]),
            q_norm = weight(n("index_q_norm"), [ix.head_dim], dense),
            k_norm = weight(n("index_k_norm"), [ix.head_dim], dense),
            keys = keys,
        )

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        if l % d.attn_every == d.attn_every - 1:
            indexer = index(n, "index.{}".format(l))
            return block(n, gated_attn(n, d, proj, "kv.{}".format(l)), indexer, None, mix.experts)
        delta = gdn(n, d, proj, mix.gdn_ba, dense, "conv.{}".format(l), "delta.{}".format(l))
        return block(n, None, None, delta, mix.experts)

    visual = tower(LARGE, hidden, dense) if vision else None

    mtp = None
    if draft:
        n = lambda s: "mtp.layer." + s
        mtp = struct(
            norm_embed = weight("mtp.norm_embed", [hidden], dense),
            norm_hidden = weight("mtp.norm_hidden", [sh], dense),
            fc_embed = weight("mtp.fc_embed", [hidden, hidden], dense),
            fc_hidden = weight("mtp.fc_hidden", [sh, hidden], dense),
            block = block(n, gated_attn(n, d, proj, "kv.mtp"), None, None, proj),
            mixer = residual("mtp.mixer", False),
            eps = d.norm_eps,
            depth = DRAFT_DEPTH,
        )

    p = d.ple
    padded = (hashes.total + p.divisible_by - 1) // p.divisible_by * p.divisible_by
    heads = (p.ngram - 1) * p.heads_per_ngram
    shards = p.split_parts
    ple = struct(
        layer = p.layer,
        eos = d.eos,
        heads_per_ngram = p.heads_per_ngram,
        mults = hashes.mults,
        primes = hashes.primes,
        offsets = hashes.offsets,
        padded_vocab = padded,
        shards = shards,
        table = weight("ple.table", [padded, hidden // heads], mix.table).packed([padded // shards] * shards),
        key_proj = weight("ple.key_proj", [sh, hidden], proj),
        value_proj = weight("ple.value_proj", [hidden, hidden], proj),
        norm_key = weight("ple.norm_key", [sh], dense),
        norm_query = weight("ple.norm_query", [sh], dense),
        norm_conv = weight("ple.norm_conv", [sh], dense),
        conv = weight("ple.conv", [sh, p.conv_kernel], dense),
        conv_kernel = p.conv_kernel,
        dilation = p.ngram,
        eps = d.norm_eps,
        ids_state = "ple.ids",
        conv_state = "ple.conv",
    )

    return struct(
        hidden = hidden,
        vocab = d.vocab,
        q_heads = d.q_heads,
        kv_heads = d.kv_heads,
        head_dim = d.head_dim,
        streams = d.streams,
        kv = kv,
        embed = weight("embed", [d.vocab, hidden], mix.embed),
        head = weight("lm_head", [d.vocab, hidden], mix.head),
        layers = [layer(l) for l in range(d.layers)],
        mixer = residual("mixer", False),
        ple = ple,
        mtp = mtp,
        tower = visual,
    )
