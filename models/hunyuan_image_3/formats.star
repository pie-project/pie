# How a HunyuanImage 3 checkpoint is laid out: transformers' names. Its fused
# qkv interleaves each kv group's query, key and value heads and its rope
# channels are interleaved, so both are gathered into the trunk's order; its
# gate-and-up projections hold up before gate.

load("//lib/diffusion/formats.star", "biased", "conv")

def formats(m):
    return [format("huggingface", read = lambda reads: read(m, reads))]

def rope_channels(head_dim):
    q = head_dim // 4
    h = head_dim // 2
    return ([2 * j for j in range(q)] + [2 * j + h for j in range(q)] +
            [2 * j + 1 for j in range(q)] + [2 * j + 1 + h for j in range(q)])

def head_permutation(heads, head_dim):
    channels = rope_channels(head_dim)
    return [h * head_dim + c for h in range(heads) for c in channels]

def qkv_rows(kv_heads, groups, head_dim, q_perm, k_perm):
    d = head_dim
    stride = (groups + 2) * d
    rows = []
    for q in q_perm:
        head, c = q // d, q % d
        g, j = head // groups, head % groups
        rows.append(g * stride + j * d + c)
    for k in k_perm:
        head, c = k // d, k % d
        rows.append(head * stride + groups * d + c)
    for g in range(kv_heads):
        for c in range(d):
            rows.append(g * stride + (groups + 1) * d + c)
    return rows

def swap_halves(e, inter):
    return concat(0, [e.slice(0, inter, inter), e.slice(0, 0, inter)])

def doubled(e):
    return concat(0, [e, e])

def signs(reads, w, seed):
    """The sign table `w`: a column of ones over one of minus ones, stored as
    the raw dtype `seed` is stored in."""
    held = stored(seed)
    raw_dtype = held.raw
    if raw_dtype == None:
        fail("`{}`: `{}` is stored {}; a constant is stated in a raw dtype".format(w.name, seed, held))
    half = w.shape[0] // 2
    joined = constant(w.name, [1.0] * half + [-1.0] * half, [2 * half, 1], raw_dtype)
    want = encoding(w.dtype)
    expr = joined if want == held else joined.cast(want)
    reads.push(tensor(w.name, expr, want, shape = w.shape))

def embedder(reads, e, prefix):
    biased(reads, e.mlp_in, prefix + ".mlp.0")
    biased(reads, e.mlp_out, prefix + ".mlp.2")

def group_norm(reads, g, name):
    reads.read(g.weight, name + ".weight")
    reads.read(g.bias, name + ".bias")

def resblock(reads, r, prefix):
    group_norm(reads, r.norm_in, prefix + ".in_layers.0")
    conv(reads, r.conv_in, prefix + ".in_layers.2")
    biased(reads, r.emb, prefix + ".emb_layers.1")
    group_norm(reads, r.norm_out, prefix + ".out_layers.0")
    conv(reads, r.conv_out, prefix + ".out_layers.3")
    if r.skip != None:
        conv(reads, r.skip, prefix + ".skip_connection")

def read(m, reads):
    d = m.dims
    reads.read(m.embed, "model.wte.weight")
    reads.read(m.head, "lm_head.weight")
    reads.read(m.final_norm, "model.ln_f.weight")
    signs(reads, m.ones, "model.ln_f.weight")

    hd = d.head_dim
    groups = d.q_heads // d.kv_heads
    q_perm = head_permutation(d.q_heads, hd)
    k_perm = head_permutation(d.kv_heads, hd)
    rope_perm = rope_channels(hd)
    qkv = qkv_rows(d.kv_heads, groups, hd, q_perm, k_perm)

    for l, w in enumerate(m.layers):
        at = lambda tail: "model.layers.{}.{}".format(l, tail)
        reads.read(w.attn_norm, at("input_layernorm.weight"))
        reads.read(w.mlp_norm, at("post_attention_layernorm.weight"))
        reads.read_expr(w.qkv, src(at("self_attn.qkv_proj.weight")).gather(0, qkv))
        reads.read_expr(w.q_norm, src(at("self_attn.query_layernorm.weight")).gather(0, rope_perm))
        reads.read_expr(w.k_norm, src(at("self_attn.key_layernorm.weight")).gather(0, rope_perm))
        reads.read(w.o_proj, at("self_attn.o_proj.weight"))
        reads.read(w.router, at("mlp.gate.wg.weight"))

        inter = d.moe_inter
        hidden = d.hidden
        reads.read_expr(w.shared_gate_up, swap_halves(src(at("mlp.shared_mlp.gate_and_up_proj.weight")), d.shared_inter))
        reads.read(w.shared_down, at("mlp.shared_mlp.down_proj.weight"))

        held = stored(at("mlp.experts.0.down_proj.weight"))
        expert = lambda e, leaf: src(at("mlp.experts.{}.{}.weight".format(e, leaf)))
        reads.read_expr(w.experts_gate_up, concat(0, [
            swap_halves(expert(e, "gate_and_up_proj"), inter).transmute([1, 2 * inter, hidden], held)
            for e in range(d.experts)
        ]))
        reads.read_expr(w.experts_down, concat(0, [
            expert(e, "down_proj").transmute([1, hidden, inter], held)
            for e in range(d.experts)
        ]))

    biased(reads, m.timestep_emb.mlp_in, "timestep_emb.mlp.0")
    reads.read_expr(m.timestep_emb.mlp_out.w, doubled(src("timestep_emb.mlp.2.weight")))
    reads.read_expr(m.timestep_emb.mlp_out.bias, doubled(src("timestep_emb.mlp.2.bias")))
    embedder(reads, m.time_embed, "time_embed")
    embedder(reads, m.time_embed_2, "time_embed_2")

    conv(reads, m.patch_embed.conv_in, "patch_embed.model.0")
    resblock(reads, m.patch_embed.res, "patch_embed.model.1")
    resblock(reads, m.final_layer.res, "final_layer.model.0")
    group_norm(reads, m.final_layer.norm_out, "final_layer.model.1.0")
    conv(reads, m.final_layer.conv_out, "final_layer.model.1.2")
