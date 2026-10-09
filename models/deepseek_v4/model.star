# The weights of DeepSeek-V4: multi-latent attention over a sliding window
# and a pool of compressed entries, residual streams mixed by manifold
# hyper-connections, DeepSeekMoE experts; V4.1-Flash adds Compressed Sparse
# Attention 2 (cross-layer KV and index reuse), Single-Pass mHC and Engram.

load("//lib/adapters/model.star", "banks")
load("//lib/hyper/model.star", "head", "hyper", "mix")

DRAFT_EXPERTS = 256

def routed(gate, up, down, split, gate_at = []):
    """The dtypes a layer's routed banks are stored at: `gate`, `up` and
    `down`, the gate overridden at `dtype` on the layers `gate_at` lists as
    `(layer, dtype)`; with `split` the gate and up banks apart."""
    return struct(gate = gate, up = up, down = down, split = split, gate_at = gate_at)

def uniform(w):
    return routed(w, w, w, False)

def split_of(w):
    return routed(w, w, w, True)

DQ_2BIT = routed(dtype.u2g32, dtype.u2g64, dtype.u2g64, True, [(4, dtype.u2g64)])
DQ_2BIT_FULL = routed(dtype.u2g32, dtype.u2g64, dtype.u2g64, True, [(42, dtype.u2g64)])

def banks_at(r, layer):
    """`r` as stored on `layer`."""
    gate = r.gate
    for at, d in r.gate_at:
        if at == layer:
            gate = d
    return routed(gate, r.up, r.down, r.split)

# ---------------------------------------------------------------------------
# The V4 base.
# ---------------------------------------------------------------------------

BASE = struct(
    hidden = 2048, layers = 6, dense_layers = 1, pool = [1, 2, 4, None, None, None],
    heads = 16, head_dim = 128, q_lora = 768, o_lora = 512, rope_dim = 64, theta = 10000.0,
    window = 2048, streams = 4, gate_eps = 1e-6, alpha = 2.0, sinkhorn = 20, dense_inter = 5632,
    experts = 64, top_k = 6, moe_inter = 1024, renorm = False, scaling = 2.5,
    swiglu_limit = 7.0, vocab = 129280, norm_eps = 1e-5,
)

def base(w, kv, d):
    hidden = d.hidden
    streams = d.streams
    q_w = d.heads * d.head_dim

    def layer(l):
        n = lambda s: "layer.{}.{}".format(l, s)
        norm = lambda s, dim: weight(n(s), [dim], w)
        lora_a, lora_b = banks("layer.{}".format(l), hidden, compute(w))
        ratio = d.pool[l]
        if l < d.dense_layers:
            di = d.dense_inter
            mlp = struct(
                kind = "dense",
                gate_up = weight(n("gate_up"), [2 * di, hidden], w).packed([di, di]),
                down = weight(n("down"), [hidden, di], w).rows(),
                inter = di,
                limit = d.swiglu_limit,
            )
        else:
            mi = d.moe_inter
            mlp = struct(
                kind = "routed",
                router = weight(n("router"), [d.experts, hidden], w),
                bias = weight(n("router_bias"), [d.experts], w),
                gate_up = weight(n("experts_gate_up"), [d.experts, 2 * mi, hidden], w).bank([mi, mi]),
                down = weight(n("experts_down"), [d.experts, hidden, mi], w).rows(),
                experts = d.experts,
                top_k = d.top_k,
                inter = mi,
                limit = d.swiglu_limit,
                renorm = d.renorm,
                scaling = d.scaling,
            )
        return struct(
            attn_mix = mix(n, "attn_mix", streams, hidden, dynamic = False),
            attn_norm = None,
            mlp_norm = None,
            attn = struct(
                rope_dim = d.rope_dim,
                theta = d.theta,
                yarn = None,
                sm_scale = f32(1.0 / f32(sqrt(d.head_dim))),
                q_down = weight(n("q_down"), [d.q_lora, hidden], w),
                q_norm = norm("q_norm", d.q_lora),
                q_norm_eps = d.norm_eps,
                q_up = weight(n("q_up"), [q_w, d.q_lora], w).columns(),
                kv_down = weight(n("kv_down"), [q_w, hidden], w).columns(),
                kv_norm = weight(n("kv_norm"), [q_w], w).columns(),
                kv_norm_eps = d.norm_eps,
                o_down = weight(n("o_down"), [d.o_lora, q_w], w).rows(),
                o_up = weight(n("o_up"), [hidden, d.o_lora], w),
                o_groups = 1,
                sink = weight(n("attn_sink"), [d.heads], w).columns(),
                kv = "kv.{}".format(l),
                pool = struct(ratio = ratio, entries = "pool.{}".format(l), compressor = None, owner = True)
                    if ratio != None else None,
                indexer = None,
                selection = "none",
            ),
            mlp_mix = mix(n, "mlp_mix", streams, hidden, dynamic = False),
            mlp = mlp,
            lora_a = lora_a,
            lora_b = lora_b,
            engram = None,
        )

    return struct(
        hidden = hidden,
        vocab = d.vocab,
        heads = d.heads,
        head_dim = d.head_dim,
        window = d.window,
        kv = kv,
        hyper = hyper(streams, d.norm_eps, d.gate_eps, d.alpha, d.sinkhorn),
        embed = weight("embed", [d.vocab, hidden], w),
        head = None,
        hc_head = None,
        layers = [layer(l) for l in range(d.layers)],
        final_norm = weight("final_norm", [hidden], w),
        final_norm_eps = d.norm_eps,
        mtp = None,
        token_map = None,
    )

# ---------------------------------------------------------------------------
# V4-Flash.
# ---------------------------------------------------------------------------

def flash_ratios():
    return [None, None] + [4 if l % 2 == 0 else 128 for l in range(2, 43)]

FLASH_MICRO_RATIOS = [None, None, 4, 128, 4]

YARN = struct(factor = 16.0, beta_fast = 32.0, beta_slow = 1.0, original_max_position = 65536)

def flash_dims(layers, pool, hash, experts = 256, draft = False):
    return struct(
        hidden = 4096, layers = layers, pool = pool, num_hash_layers = hash, heads = 64,
        head_dim = 512, q_lora = 1024, kv_latent = 512, o_groups = 8, o_lora = 1024, rope_dim = 64,
        theta = 10000.0, compress_theta = 160000.0, draft = draft, window = 128,
        index_heads = 64, index_head_dim = 128, index_top_k = 512, index_window = 128,
        streams = 4, gate_eps = 1e-6, alpha = 2.0, sinkhorn = 20, experts = experts, top_k = 6,
        moe_inter = 2048, shared_inter = 2048, renorm = True, scaling = 1.5,
        swiglu_limit = 10.0, vocab = 129280, norm_eps = 1e-6,
    )

def moe_flash(n, d, experts, gate, banks, weights, dense):
    """A DeepSeekMoE block of `experts` routed by `gate`, its routed banks
    stored as `banks` says, its shared expert in `weights` and its router in
    `dense`."""
    mi = d.moe_inter
    si = d.shared_inter
    hidden = d.hidden
    if banks.split:
        half = lambda what, dt: weight(n(what), [experts, mi, hidden], dt).bank([mi])
        gate_up = struct(fused = None, gate = half("experts_gate", banks.gate), up = half("experts_up", banks.up))
    else:
        gate_up = struct(
            fused = weight(n("experts_gate_up"), [experts, 2 * mi, hidden], banks.gate).bank([mi, mi]),
            gate = None,
            up = None,
        )
    return struct(
        kind = "moe_flash",
        router = weight(n("gate"), [experts, hidden], dense),
        gate = gate,
        gate_up = gate_up,
        down = weight(n("experts_down"), [experts, hidden, mi], banks.down).rows(),
        shared_gate_up = weight(n("shared_gate_up"), [2 * si, hidden], weights).packed([si, si]),
        shared_down = weight(n("shared_down"), [hidden, si], weights).rows(),
        experts = experts,
        top_k = d.top_k,
        inter = mi,
        shared_inter = si,
        limit = d.swiglu_limit,
        renorm = d.renorm,
        scaling = d.scaling,
    )

def flash(w, r, kv, d):
    dense = compute(w)
    hidden = d.hidden
    streams = d.streams
    q_w = d.heads * d.head_dim
    kv_latent = d.kv_latent
    o_out = d.o_groups * d.o_lora
    idx_w = d.index_heads * d.index_head_dim

    def compressor(prefix, ratio, entries, norm_w):
        return struct(
            wkv = weight(prefix + ".wkv", [entries, hidden], w),
            wgate = weight(prefix + ".wgate", [entries, hidden], w),
            ape = weight(prefix + ".ape", [ratio, entries], dtype.f32),
            norm = weight(prefix + ".norm", [norm_w], dense),
            norm_eps = d.norm_eps,
        )

    def layer_at(prefix, key, ratio, hash, experts, routed_banks, weights, sdense):
        """The layer named `prefix`, its caches named by `key`."""
        n = lambda s: "{}.{}".format(prefix, s)
        norm = lambda s, dim: weight(n(s), [dim], sdense)
        lora_a, lora_b = banks(prefix, hidden, sdense)
        has_indexer = ratio == 4
        pool = None
        if ratio != None:
            entries = 2 * kv_latent if has_indexer else kv_latent
            pool = struct(
                ratio = ratio,
                entries = "pool." + key,
                compressor = compressor(n("compressor"), ratio, entries, kv_latent),
                owner = True,
            )
        indexer = None
        if has_indexer:
            indexer = struct(
                heads = d.index_heads,
                head_dim = d.index_head_dim,
                top_k = d.index_top_k,
                rope_dim = d.rope_dim,
                theta = d.compress_theta,
                yarn = YARN,
                window = d.index_window,
                wq_b = weight(n("indexer.wq_b"), [idx_w, d.q_lora], weights),
                weights_proj = weight(n("indexer.weights_proj"), [d.index_heads, hidden], weights),
                compressor = compressor(n("indexer.compressor"), ratio, 2 * d.index_head_dim, d.index_head_dim),
                wk = None,
                k_norm = None,
                keys = "index." + key,
                owns_keys = True,
            )
        if hash:
            gate = struct(kind = "hash", tid2eid = weight(n("gate.tid2eid"), [d.vocab, d.top_k], dtype.i64))
        else:
            gate = struct(kind = "bias", bias = weight(n("gate.bias"), [experts], dtype.f32))
        return struct(
            attn_mix = mix(n, "attn_mix", streams, hidden),
            attn_norm = norm("attn_norm", hidden),
            mlp_norm = norm("ffn_norm", hidden),
            attn = struct(
                rope_dim = d.rope_dim,
                theta = d.compress_theta if ratio != None else d.theta,
                yarn = YARN if ratio != None else None,
                sm_scale = f32(1.0 / f32(sqrt(d.head_dim))),
                q_down = weight(n("q_down"), [d.q_lora, hidden], weights),
                q_norm = norm("q_norm", d.q_lora),
                q_norm_eps = d.norm_eps,
                q_up = weight(n("q_up"), [q_w, d.q_lora], weights).columns(),
                kv_down = weight(n("kv_down"), [kv_latent, hidden], weights),
                kv_norm = norm("kv_norm", kv_latent),
                kv_norm_eps = d.norm_eps,
                o_down = weight(n("o_down"), [o_out, hidden], weights).columns(),
                o_up = weight(n("o_up"), [hidden, o_out], weights).rows(),
                o_groups = d.o_groups,
                sink = weight(n("attn_sink"), [d.heads], sdense).columns(),
                kv = "kv." + key,
                pool = pool,
                indexer = indexer,
                selection = "own" if has_indexer else "none",
            ),
            mlp_mix = mix(n, "mlp_mix", streams, hidden),
            mlp = moe_flash(n, d, experts, gate, routed_banks, weights, sdense),
            lora_a = lora_a,
            lora_b = lora_b,
            engram = None,
        )

    layers = [
        layer_at(
            "layer.{}".format(l),
            key = str(l),
            ratio = d.pool[l],
            hash = l < d.num_hash_layers,
            experts = d.experts,
            routed_banks = banks_at(r, l),
            weights = w,
            sdense = dense,
        )
        for l in range(d.layers)
    ]
    mtp = None
    if d.draft:
        bf = dtype.bf16
        mtp = struct(
            enorm = weight("mtp.enorm", [hidden], bf),
            hnorm = weight("mtp.hnorm", [hidden], bf),
            e_proj = weight("mtp.e_proj", [hidden, hidden], bf),
            h_proj = weight("mtp.h_proj", [streams * hidden, hidden], bf),
            block = layer_at(
                "mtp.decoder",
                key = "mtp",
                ratio = None,
                hash = False,
                experts = DRAFT_EXPERTS,
                routed_banks = split_of(dtype.mxfp4),
                weights = bf,
                sdense = bf,
            ),
            hc_head = head("mtp.hc_head", streams, hidden),
            norm = weight("mtp.norm", [hidden], bf),
            norm_eps = d.norm_eps,
        )
    return struct(
        hidden = hidden,
        vocab = d.vocab,
        heads = d.heads,
        head_dim = d.head_dim,
        window = d.window,
        kv = kv,
        hyper = hyper(streams, d.norm_eps, d.gate_eps, d.alpha, d.sinkhorn),
        embed = weight("embed", [d.vocab, hidden], w),
        head = weight("lm_head", [d.vocab, hidden], w),
        hc_head = head("hc_head", streams, hidden),
        layers = layers,
        final_norm = weight("final_norm", [hidden], dense),
        final_norm_eps = d.norm_eps,
        mtp = mtp,
        token_map = None,
    )

# ---------------------------------------------------------------------------
# V4.1-Flash: Causal Encoder-Decoder over CSA2 with cross-layer KV and index
# reuse, Single-Pass mHC, and Engram conditional memory.
# ---------------------------------------------------------------------------

V41_PLAN = struct(
    layers = 40,
    ratios = [0, 0] + [2] * 18 + [1] * 20,
    kv_sources = [2, 8, 14, 20],
    index_sources = [2, 8, 14, 20, 24, 28, 32, 36],
    engram_layers = [1, 14],
)

# The miniature: source layers 0,1,2,3,20,21,24,25 of the release, so every
# layer kind is present once, with 16 of the 384 experts and an Engram table
# hashed at base 20 000 instead of 16 000 000.
V41_MINI_PLAN = struct(
    layers = 8,
    ratios = [0, 0, 2, 2, 1, 1, 1, 1],
    kv_sources = [2, 4],
    index_sources = [2, 4, 6],
    engram_layers = [1],
)

def dims41(plan, experts = 384, base_vocab = 16000000):
    return struct(
        hidden = 5120, plan = plan, heads = 64, head_dim = 512, q_lora = 1280, kv_latent = 512,
        o_groups = 8, o_lora = 1024, rope_dim = 64, theta = 10000.0, compress_theta = 160000.0,
        window = 128, index_heads = 32, index_head_dim = 128, index_top_k = 512, streams = 4,
        gate_eps = 1e-6, alpha = 2.0, sinkhorn = 20, experts = experts, top_k = 6,
        moe_inter = 2304, shared_inter = 2304, renorm = True, scaling = 1.5, swiglu_limit = 10.0,
        vocab = 129280, norm_eps = 1e-20,
        engram = struct(ngram = 4, heads = 8, head_dim = 256, base_vocab = base_vocab,
                        compressed_vocab = 99092, pad = 2),
    )

def csa_of(plan, layer):
    if plan.ratios[layer] == 0:
        return "swa"
    if layer in plan.kv_sources:
        return "full"
    if layer in plan.index_sources:
        return "reindex"
    return "reuse"

I64_MAX = 9223372036854775807

def engram_hash_constants(e, engram_layers, which):
    """Engram's hash geometry for the `which`-th module: every (module,
    n-gram order, head) owns a distinct prime-sized bucket range, the primes
    drawn in module order, each order restarting at `base_vocab - 1` and never
    reusing a prime; the multipliers are the module's own
    `numpy.random.default_rng(10007 * layer)` draws, made odd."""
    seen = []
    primes = []
    offsets = []
    for module in range(len(engram_layers)):
        total = 0
        for _ in range(e.ngram - 1):
            current = e.base_vocab - 1
            for _ in range(e.heads):
                current = prime_after(current)
                for _ in range(1000000):
                    if current not in seen:
                        break
                    current = prime_after(current)
                seen.append(current)
                if module == which:
                    primes.append(current)
                    offsets.append(total)
                    total += current
    layer = engram_layers[which]
    bound = max(I64_MAX // max(e.compressed_vocab, 1) // 2, 1)
    mults = [v * 2 + 1 for v in numpy_integers(10007 * layer, bound, e.ngram)]
    return mults, primes, offsets

def flash41(w, r, kv, d):
    plan = d.plan
    dense = compute(w)
    hidden = d.hidden
    streams = d.streams
    q_w = d.heads * d.head_dim
    kv_latent = d.kv_latent
    o_out = d.o_groups * d.o_lora
    o_in = q_w // d.o_groups
    idx_w = d.index_heads * d.index_head_dim

    def layer(l):
        prefix = "layer.{}".format(l)
        n = lambda s: "{}.{}".format(prefix, s)
        norm = lambda s, dim: weight(n(s), [dim], dense)
        lora_a, lora_b = banks(prefix, hidden, dense)
        ratio = plan.ratios[l]
        csa = csa_of(plan, l)
        kv_of = None
        for s in plan.kv_sources:
            if s <= l and (kv_of == None or s > kv_of):
                kv_of = s
        owner = kv_of == l
        pool = None
        if ratio > 0:
            if kv_of == None:
                fail("layer {} compresses at ratio {} but no KV source precedes it".format(l, ratio))
            pool = struct(
                ratio = ratio,
                entries = "pool.{}".format(kv_of),
                compressor = struct(
                    wkv = weight(n("compressor.wkv"), [kv_latent, hidden], dense),
                    wgate = weight(n("compressor.wgate"), [kv_latent, hidden], dense) if ratio > 1 else None,
                    ape = None,
                    norm = norm("compressor.norm", kv_latent),
                    norm_eps = d.norm_eps,
                ) if owner else None,
                owner = owner,
            )
        indexer = None
        if csa in ["full", "reindex"]:
            indexer = struct(
                heads = d.index_heads,
                head_dim = d.index_head_dim,
                top_k = d.index_top_k,
                rope_dim = d.rope_dim,
                theta = d.compress_theta,
                yarn = YARN,
                window = d.window,
                wq_b = weight(n("indexer.wq_b"), [idx_w, d.q_lora], w),
                weights_proj = weight(n("indexer.weights_proj"), [d.index_heads, hidden], dense),
                compressor = None,
                wk = weight(n("indexer.wk"), [d.index_head_dim, kv_latent], dense) if owner else None,
                k_norm = norm("indexer.k_norm", d.index_head_dim) if owner else None,
                keys = "index.{}".format(kv_of if kv_of != None else l),
                owns_keys = owner,
            )
        selection = {"swa": "none", "full": "own", "reindex": "own", "reuse": "shared"}[csa]

        engram = None
        if l in plan.engram_layers:
            which = plan.engram_layers.index(l)
            e = d.engram
            mults, primes, offsets = engram_hash_constants(e, plan.engram_layers, which)
            rows = 0
            for p in primes:
                rows += p
            cols = (e.ngram - 1) * e.heads
            engram = struct(
                table = weight(n("engram.embed"), [rows, e.head_dim], dense),
                wkv = weight(n("engram.wkv"), [(streams + 1) * hidden, cols * e.head_dim], w)
                    .packed([streams * hidden, hidden]),
                q_weight = weight(n("engram.q_weight"), [streams, hidden], dense),
                k_weight = weight(n("engram.k_weight"), [streams, hidden], dense),
                mults = mults,
                primes = primes,
                offsets = offsets,
                ngram = e.ngram,
                heads_per_ngram = e.heads,
                head_dim = e.head_dim,
                pad = e.pad,
                eps = d.norm_eps,
                ids_state = "engram.{}".format(l),
            )

        return struct(
            attn_mix = mix(n, "attn_mix", streams, hidden),
            attn_norm = norm("attn_norm", hidden),
            mlp_norm = norm("ffn_norm", hidden),
            attn = struct(
                rope_dim = d.rope_dim,
                theta = d.compress_theta if ratio > 0 else d.theta,
                yarn = YARN if ratio > 0 else None,
                sm_scale = f32(1.0 / f32(sqrt(d.head_dim))),
                q_down = weight(n("q_down"), [d.q_lora, hidden], w),
                q_norm = norm("q_norm", d.q_lora),
                q_norm_eps = d.norm_eps,
                q_up = weight(n("q_up"), [q_w, d.q_lora], w).columns(),
                kv_down = weight(n("kv_down"), [kv_latent, hidden], w),
                kv_norm = norm("kv_norm", kv_latent),
                kv_norm_eps = d.norm_eps,
                o_down = weight(n("o_down"), [o_out, o_in], w),
                o_up = weight(n("o_up"), [hidden, o_out], w).rows(),
                o_groups = d.o_groups,
                sink = weight(n("attn_sink"), [d.heads], dense).columns(),
                kv = "kv.{}".format(l),
                pool = pool,
                indexer = indexer,
                selection = selection,
            ),
            mlp_mix = mix(n, "mlp_mix", streams, hidden),
            mlp = moe_flash(
                n,
                d,
                d.experts,
                gate = struct(kind = "bias", bias = weight(n("gate.bias"), [d.experts], dtype.f32)),
                banks = banks_at(r, l),
                weights = w,
                dense = dense,
            ),
            lora_a = lora_a,
            lora_b = lora_b,
            engram = engram,
        )

    return struct(
        hidden = hidden,
        vocab = d.vocab,
        heads = d.heads,
        head_dim = d.head_dim,
        window = d.window,
        kv = kv,
        hyper = hyper(streams, d.norm_eps, d.gate_eps, d.alpha, d.sinkhorn, single_pass = True),
        embed = weight("embed", [d.vocab, hidden], dense),
        head = weight("lm_head", [d.vocab, hidden], dense),
        hc_head = None,
        layers = [layer(l) for l in range(plan.layers)],
        final_norm = weight("final_norm", [hidden], dense),
        final_norm_eps = d.norm_eps,
        mtp = None,
        token_map = weight("engram.token_map", [d.vocab], dtype.i64) if plan.engram_layers else None,
    )

def layout(id, deploy):
    ws = deploy.weights
    bf = dtype.bf16
    u4 = dtype.u4g64
    mx = dtype.mxfp4
    kv = deploy.kv
    mtp = deploy.drafter == "mtp"
    if id == "dsv41-flash" and ws == [u4]:
        return flash41(u4, split_of(u4), kv, dims41(V41_PLAN))
    if id == "dsv41-flash" and ws == [bf, mx]:
        return flash41(bf, split_of(mx), kv, dims41(V41_PLAN))
    if id == "dsv41-flash-mini" and ws == [bf, mx]:
        return flash41(bf, split_of(mx), kv, dims41(V41_MINI_PLAN, experts = 16, base_vocab = 20000))
    if id == "dsv4-flash":
        if ws == [bf] and not mtp:
            return flash(bf, uniform(bf), kv, flash_dims(43, flash_ratios(), 3))
        if ws == [u4, dtype.u2g64] and not mtp:
            return flash(u4, DQ_2BIT_FULL, kv, flash_dims(43, flash_ratios(), 3))
        if ws == [u4, dtype.u2g64, mx] and mtp:
            return flash(u4, DQ_2BIT_FULL, kv, flash_dims(43, flash_ratios(), 3, draft = True))
    if id == "dsv4-flash-mini":
        if ws == [bf] and not mtp:
            return flash(bf, uniform(bf), kv, flash_dims(5, FLASH_MICRO_RATIOS, 3, experts = 16))
        if ws == [u4, dtype.u2g64] and not mtp:
            return flash(u4, DQ_2BIT, kv, flash_dims(5, FLASH_MICRO_RATIOS, 3, experts = 16))
        if ws == [u4, dtype.u2g64, mx] and mtp:
            return flash(u4, DQ_2BIT, kv, flash_dims(5, FLASH_MICRO_RATIOS, 3, experts = 16, draft = True))
        if ws == [bf, mx] and not mtp:
            return flash(bf, split_of(mx), kv, flash_dims(5, FLASH_MICRO_RATIOS, 3, experts = 16))
    if id == "dsv4-base" and ws == [bf] and not mtp:
        return base(bf, kv, BASE)
    fail("{} does not ship {}".format(id, deploy))
