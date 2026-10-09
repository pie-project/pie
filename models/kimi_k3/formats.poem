# How a Kimi-K3 checkpoint is laid out: transformers' names under
# `language_model.`, or llama.cpp's GGUF names.

def product(xs):
    out = 1
    for x in xs:
        out *= x
    return out

def squeezed(name):
    """A depthwise convolution bank stored `[channels, 1, kernel]` or
    `[channels, kernel, 1]`, read as `[channels, kernel]`."""
    held = shape(name)
    if len(held) == 3 and held[1] == 1:
        channels, kernel = held[0], held[2]
    elif len(held) == 3 and held[2] == 1:
        channels, kernel = held[0], held[1]
    else:
        fail("`{}`: a depthwise convolution bank is stored [channels, 1, kernel] or [channels, kernel, 1] and this one is stored {}".format(name, held))
    return src(name).transmute([channels, kernel], stored(name))

def kda_qkv(at):
    """The names of the q, k and v projections the packed `qkv` reads."""
    return [at("self_attn.{}_proj.weight".format(p)) for p in ["q", "k", "v"]]

def kda_conv(k, at):
    """The packed `conv` bank, read from the q, k and v banks."""
    return concat(k.conv.cut_axis, [squeezed(at("self_attn.{}_conv1d.weight".format(p))) for p in ["q", "k", "v"]])

def kda_gate(k, at):
    """The output gate's projections, each beside its name."""
    if len(k.gate) == 1:
        return [(k.gate[0], at("self_attn.g_proj.weight"))]
    return [(k.gate[0], at("self_attn.g_a_proj.weight")), (k.gate[1], at("self_attn.g_b_proj.weight"))]

def named(a, at):
    """`a`'s weights, each beside the name `at` gives its transformers leaf,
    in the order a checkpoint lists them."""
    out = [
        (a.q_a_proj, at("self_attn.q_a_proj.weight")),
        (a.q_a_norm, at("self_attn.q_a_layernorm.weight")),
        (a.q_b_proj, at("self_attn.q_b_proj.weight")),
        (a.kv_a_proj, at("self_attn.kv_a_proj_with_mqa.weight")),
        (a.kv_a_norm, at("self_attn.kv_a_layernorm.weight")),
        (a.kv_b_proj, at("self_attn.kv_b_proj.weight")),
    ]
    if a.gate != None:
        out.append((a.gate, at("self_attn.g_proj.weight")))
    out.append((a.o_proj, at("self_attn.o_proj.weight")))
    return out

GGUF_EMBED = "token_embd.weight"

def formats(m):
    return [
        format(
            "huggingface",
            recognizes = lambda c: not has(GGUF_EMBED),
            read = lambda reads: huggingface(m, reads),
        ),
        format("gguf", recognizes = lambda c: has(GGUF_EMBED), read = lambda reads: gguf(m, reads)),
    ]

def at(l, leaf):
    return "language_model.model.layers.{}.{}".format(l, leaf)

def blk(l, leaf):
    return "blk.{}.{}".format(l, leaf)

def lifted(w):
    """`w`'s shape, its cut axis left for the stack to fill."""
    dims = list(w.shape)
    dims[w.cut_axis] = -1
    return dims

def huggingface(m, reads):
    reads.read(m.embed, "language_model.model.embed_tokens.weight")
    reads.read(m.final_norm, "language_model.model.norm.weight")
    reads.read(m.head, "language_model.lm_head.weight")
    for l, w in enumerate(m.layers):
        n = lambda leaf: at(l, leaf)
        reads.read(w.mixer_norm, n("input_layernorm.weight"))
        reads.read(w.mlp_norm, n("post_attention_layernorm.weight"))
        if w.res_blend != None:
            reads.read(w.res_blend.norm, n("self_attention_res_norm.weight"))
            reads.read(w.res_blend.proj, n("self_attention_res_proj.weight"))
        if w.mlp_res != None:
            reads.read(w.mlp_res.norm, n("mlp_res_norm.weight"))
            reads.read(w.mlp_res.proj, n("mlp_res_proj.weight"))
        if w.mixer.mla:
            for a, name in named(w.mixer, n):
                reads.read(a, name)
        else:
            kda(reads, n, w.mixer)
        f = w.mlp
        if not f.routed:
            reads.read_concat(f.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
            reads.read(f.down, n("mlp.down_proj.weight"))
            continue
        reads.read(f.router, n("block_sparse_moe.gate.weight"))
        if f.bias != None:
            reads.read(f.bias, n("block_sparse_moe.gate.e_score_correction_bias"))
        lat = f.latent
        if lat != None:
            reads.read(lat.down, n("block_sparse_moe.routed_expert_down_proj.weight"))
            if lat.norm != None:
                reads.read(lat.norm, n("block_sparse_moe.routed_expert_norm.weight"))
            reads.read(lat.up, n("block_sparse_moe.routed_expert_up_proj.weight"))

        # The released checkpoint stores every expert leg as
        # compressed-tensors MXFP4 (`w1.weight_packed` beside
        # `w1.weight_scale`); the fixture stores bf16 `w1.weight`.
        def leg(e, what):
            stem = n("block_sparse_moe.experts.{}.{}".format(e, what))
            return stem + ".weight_packed" if has(stem + ".weight_packed") else stem + ".weight"

        gate_up = []
        for e in range(f.experts):
            gate_up += [leg(e, "w1"), leg(e, "w3")]
        expert_bank(reads, f.gate_up, gate_up)
        expert_bank(reads, f.down, [leg(e, "w2") for e in range(f.experts)])
        # one shared expert (`shared_expert.`) or several folded into one
        # wider MLP (`shared_experts.`)
        s = f.shared
        many = has(n("block_sparse_moe.shared_experts.gate_proj.weight"))
        stem = "block_sparse_moe.shared_experts" if many else "block_sparse_moe.shared_expert"
        reads.read_concat(s.gate_up, [n(stem + ".gate_proj.weight"), n(stem + ".up_proj.weight")])
        reads.read(s.down, n(stem + ".down_proj.weight"))
    r = m.output_res
    if r != None:
        reads.read(r.norm, "language_model.model.output_attn_res_norm.weight")
        reads.read(r.proj, "language_model.model.output_attn_res_proj.weight")

def kda(reads, n, k):
    reads.read_concat(k.qkv, kda_qkv(n))
    reads.read_expr(k.conv, kda_conv(k, n))
    reads.read(k.f_a, n("self_attn.f_a_proj.weight"))
    reads.read(k.f_b, n("self_attn.f_b_proj.weight"))
    reads.read(k.b, n("self_attn.b_proj.weight"))

    # stored flat `[heads * head_dim]`; read as the `[heads, head_dim]` plane
    reads.read_expr(k.dt_bias, src(n("self_attn.dt_bias")).transmute(lifted(k.dt_bias), raw(dtype.f32)))

    # The released Kimi-K3 stores `A_log` as `[128]` against 96 heads; the
    # reference's KDA gate loads one entry per head, so the first `heads`
    # entries are the decays and the tail is never read.
    a_log = n("self_attn.A_log")
    if has(a_log) and product(shape(a_log)) > k.heads:
        reads.read_expr(k.a_log, src(a_log).slice(0, 0, k.heads))
    else:
        reads.read(k.a_log, a_log)
    for w, name in kda_gate(k, n):
        reads.read(w, name)
    reads.read(k.o_norm, n("self_attn.o_norm.weight"))
    reads.read(k.o_proj, n("self_attn.o_proj.weight"))

def expert_bank(reads, w, names):
    if w.dtype == dtype.mxfp4 and names[0].endswith(".weight_packed"):
        packed_bank(reads, w, names)
        return
    read = logical(stored(names[0]))
    reads.read_expr(w, concat(0, [src(n) for n in names]).transmute(lifted(w), raw(read)))

def packed_bank(reads, w, names):
    """A routed bank from compressed-tensors MXFP4 legs: each
    `*.weight_packed` is a `[rows, cols/2]` u8 plane of e2m1 nibble pairs with
    a `[rows, cols/32]` u8 e8m0 `*.weight_scale` beside it, stacked on the
    leading axis and re-read as the bank's `[experts, ...]` rectangle."""
    rows, cols = w.shape[1], w.shape[2]
    legs = len(names)
    per_leg = rows * w.shape[0] // legs
    codes = []
    scales = []
    for part in names:
        stem = part[:-len(".weight_packed")] if part.endswith(".weight_packed") else part
        scale = stem + ".weight_scale"
        if not has(scale):
            fail("`{}`: `{}` holds MXFP4 codes whose exponents are stored beside it as `{}`, and the checkpoint holds none".format(w.name, part, scale))
        codes.append(src(part).transmute([1, per_leg, cols], encoding(dtype.mxfp4)))
        scales.append(src(scale).transmute([1, per_leg, cols // 32], encoding(dtype.e8m0)))
    counted = scales_shape(w)
    codes = concat(0, codes)
    scales = concat(0, scales)

    # Two legs per expert (gate, up) stack to `[2E, inter, cols]`, which is
    # the declared `[E, 2*inter, cols]` rectangle byte for byte; one leg per
    # expert already is the declared shape.
    if [legs, per_leg, cols] != list(w.shape):
        codes = codes.transmute(w.shape, encoding(dtype.mxfp4))
        scales = scales.transmute(counted, encoding(dtype.e8m0))
    reads.extend([
        tensor(w.name, codes, encoding(dtype.mxfp4)),
        tensor(scales_name(w.name), scales, encoding(dtype.e8m0), shape = counted, scaling = w),
    ])

def gguf(m, reads):
    reads.read(m.embed, GGUF_EMBED)
    reads.read(m.final_norm, "output_norm.weight")
    reads.read(m.head, "output.weight")
    for l, w in enumerate(m.layers):
        reads.read(w.mixer_norm, blk(l, "attn_norm.weight"))
        reads.read(w.mlp_norm, blk(l, "ffn_norm.weight"))
        if w.res_blend != None:
            reads.read(w.res_blend.norm, blk(l, "attn_res_norm.weight"))
            reads.read(w.res_blend.proj, blk(l, "attn_res_proj.weight"))
        a = w.mixer
        if a.mla:
            reads.read(a.q_a_proj, blk(l, "attn_q_a.weight"))
            reads.read(a.q_a_norm, blk(l, "attn_q_a_norm.weight"))
            reads.read(a.q_b_proj, blk(l, "attn_q_b.weight"))
            reads.read(a.kv_a_proj, blk(l, "attn_kv_a_mqa.weight"))
            reads.read(a.kv_a_norm, blk(l, "attn_kv_a_norm.weight"))
            reads.read(a.kv_b_proj, blk(l, "attn_kv_b.weight"))
            reads.read(a.gate, blk(l, "attn_gate.weight"))
            reads.read(a.o_proj, blk(l, "attn_output.weight"))
        else:
            reads.read(a.qkv, blk(l, "ssm_in.weight"))
            reads.read(a.conv, blk(l, "ssm_conv1d.weight"))
            reads.read(a.f_a, blk(l, "ssm_f_a.weight"))
            reads.read(a.f_b, blk(l, "ssm_f_b.weight"))
            reads.read(a.b, blk(l, "ssm_beta.weight"))
            reads.read(a.dt_bias, blk(l, "ssm_dt.bias"))
            reads.read(a.a_log, blk(l, "ssm_a"))
            reads.read(a.gate[0], blk(l, "ssm_gate.weight"))
            reads.read(a.o_norm, blk(l, "ssm_norm.weight"))
            reads.read(a.o_proj, blk(l, "ssm_out.weight"))
        f = w.mlp
        if not f.routed:
            reads.read_concat(f.gate_up, [blk(l, "ffn_gate.weight"), blk(l, "ffn_up.weight")])
            reads.read(f.down, blk(l, "ffn_down.weight"))
            continue
        reads.read(f.router, blk(l, "ffn_gate_inp.weight"))
        reads.read_concat(f.gate_up, [blk(l, "ffn_gate_exps.weight"), blk(l, "ffn_up_exps.weight")])
        reads.read(f.down, blk(l, "ffn_down_exps.weight"))
        reads.read_concat(f.shared.gate_up, [blk(l, "ffn_gate_shexp.weight"), blk(l, "ffn_up_shexp.weight")])
        reads.read(f.shared.down, blk(l, "ffn_down_shexp.weight"))
