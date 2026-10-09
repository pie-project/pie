# How a Qwen 3.8 Flash Next checkpoint is laid out: transformers' names, or
# mlx_lm's, with the hyper-connections, the n-gram embedding and the draft
# head beside the trunk.

def product(xs):
    out = 1
    for x in xs:
        out *= x
    return out

def flattened(name, want, broadcast = False):
    """The tensor `name` as stored, its elements read as `want`; with
    `broadcast`, a single stored value is read as any shape."""
    held = shape(name)
    if (not broadcast or product(held) > 1) and product(held) != product(want):
        fail("`{}` is stored {} ({} elements) and the plan reads it as {} ({} elements)".format(
            name, held, product(held), want, product(want)))
    return src(name).transmute(want, stored(name))

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

LAYOUTS = [
    struct(
        name = "transformers",
        trunk = "model.language_model.",
        head = "lm_head.weight",
        tower = "model.visual.",
        mlx = False,
    ),
    struct(
        name = "mlx_lm",
        trunk = "language_model.model.",
        head = "language_model.lm_head.weight",
        tower = "vision_tower.",
        mlx = True,
    ),
]

def safetensors_format(layout, read, states = []):
    """`layout`'s format, recognized by its embedding: `read(reads, layout)`."""
    return format(
        layout.name,
        recognizes = lambda checkpoint: has(layout.trunk + "embed_tokens.weight"),
        read = lambda reads: read(reads, layout),
        states = states,
    )

def norm_of(layout):
    """`norm(name)`: a norm as the layout stores it, read as the forward's."""
    if layout.mlx:
        return lambda name: src(name).bias(-1.0)
    return lambda name: src(name)

def read_attn(reads, a, n, norm):
    reads.read(a.qg_proj, n("self_attn.q_proj.weight"))
    reads.read(a.k_proj, n("self_attn.k_proj.weight"))
    reads.read(a.v_proj, n("self_attn.v_proj.weight"))
    reads.read(a.o_proj, n("self_attn.o_proj.weight"))
    reads.read_expr(a.q_norm, norm(n("self_attn.q_norm.weight")))
    reads.read_expr(a.k_norm, norm(n("self_attn.k_norm.weight")))

def read_gdn(reads, g, n):
    reads.read_concat(g.in_qkvz, [n("linear_attn.in_proj_qkv.weight"), n("linear_attn.in_proj_z.weight")])
    reads.read_concat(g.in_ba, [n("linear_attn.in_proj_b.weight"), n("linear_attn.in_proj_a.weight")])
    reads.read_expr(g.conv, squeezed(n("linear_attn.conv1d.weight")))
    reads.read(g.dt_bias, n("linear_attn.dt_bias"))
    reads.read(g.a_log, n("linear_attn.A_log"))
    reads.read(g.norm, n("linear_attn.norm.weight"))
    reads.read(g.out_proj, n("linear_attn.out_proj.weight"))

def read_routed(reads, f, n, layout):
    reads.read(f.router, n("mlp.gate.weight"))
    if layout.mlx:
        reads.read_concat(f.gate_up, [n("mlp.switch_mlp.gate_proj.weight"), n("mlp.switch_mlp.up_proj.weight")])
        reads.read(f.down, n("mlp.switch_mlp.down_proj.weight"))
    else:
        reads.read(f.gate_up, n("mlp.experts.gate_up_proj"))
        reads.read(f.down, n("mlp.experts.down_proj"))
    reads.read_concat(f.shared_gate_up, [n("mlp.shared_expert.gate_proj.weight"), n("mlp.shared_expert.up_proj.weight")])
    reads.read(f.shared_down, n("mlp.shared_expert.down_proj.weight"))
    reads.read(f.shared_gate, n("mlp.shared_expert_gate.weight"))

CHANNELS = 3

def read_tower(reads, t, prefix, channels_last):
    v = lambda s: prefix + s
    flat = flattened(v("patch_embed.proj.weight"), t.patch_embed.shape, broadcast = True)
    if channels_last:
        per = t.patch_embed.shape[1] // CHANNELS
        flat = flat.gather(1, [j * CHANNELS + c for c in range(CHANNELS) for j in range(per)])
    reads.read_expr(t.patch_embed, flat)
    reads.read(t.patch_embed_bias, v("patch_embed.proj.bias"))
    reads.read(t.pos_embed, v("pos_embed.weight"))
    for l, blk in enumerate(t.blocks):
        n = lambda s: v("blocks.{}.{}".format(l, s))
        for w, name in [
            (blk.norm1, "norm1.weight"),
            (blk.norm1_bias, "norm1.bias"),
            (blk.qkv, "attn.qkv.weight"),
            (blk.qkv_bias, "attn.qkv.bias"),
            (blk.proj, "attn.proj.weight"),
            (blk.proj_bias, "attn.proj.bias"),
            (blk.norm2, "norm2.weight"),
            (blk.norm2_bias, "norm2.bias"),
            (blk.fc1, "mlp.linear_fc1.weight"),
            (blk.fc1_bias, "mlp.linear_fc1.bias"),
            (blk.fc2, "mlp.linear_fc2.weight"),
            (blk.fc2_bias, "mlp.linear_fc2.bias"),
        ]:
            reads.read(w, n(name))
    mg = t.merger
    for w, name in [
        (mg.norm, "merger.norm.weight"),
        (mg.norm_bias, "merger.norm.bias"),
        (mg.fc1, "merger.linear_fc1.weight"),
        (mg.fc1_bias, "merger.linear_fc1.bias"),
        (mg.fc2, "merger.linear_fc2.weight"),
        (mg.fc2_bias, "merger.linear_fc2.bias"),
    ]:
        reads.read(w, v(name))

def formats(m):
    return [safetensors_format(layout, lambda reads, layout: read(m, reads, layout)) for layout in LAYOUTS]

def read(m, reads, layout):
    norm = norm_of(layout)
    trunk = lambda tail: layout.trunk + tail

    reads.read(m.embed, trunk("embed_tokens.weight"))
    reads.read(m.head, layout.head)

    for l, w in enumerate(m.layers):
        read_layer(reads, w, layout, norm, lambda tail: trunk("layers.{}.{}".format(l, tail)))

    mtp = m.mtp
    if mtp != None:
        n = lambda tail: "mtp." + tail
        reads.read(mtp.norm_embed, n("pre_fc_norm_embedding.weight"))
        reads.read(mtp.norm_hidden, n("pre_fc_norm_hidden.weight"))
        reads.read(mtp.fc_embed, n("fc_embedding.weight"))
        reads.read_expr(mtp.fc_hidden, concat(0, [src(n("fc_hidden.weight")) for _ in range(m.streams)]))
        read_layer(reads, mtp.block, layout, norm, lambda tail: "mtp.layers.0." + tail)
        read_mixer(reads, mtp.mixer, norm, n("hyper_connection_mixer"))

    if m.tower != None:
        read_tower(reads, m.tower, layout.tower, layout.mlx)

    p = m.ple
    n = lambda tail: trunk("layers.{}.ple.{}".format(p.layer, tail))
    reads.read_concat(p.table, [
        n("ple_embedding.ngram_embedding.shard_{}.weight".format(i))
        for i in range(p.shards)
    ])
    reads.read(p.key_proj, n("key_proj.weight"))
    reads.read(p.value_proj, n("value_proj.weight"))
    reads.read_expr(p.norm_key, norm(n("norm_key.weight")))
    reads.read_expr(p.norm_query, norm(n("norm_query.weight")))
    reads.read_expr(p.norm_conv, norm(n("norm_conv.weight")))
    reads.read_expr(p.conv, squeezed(n("conv1d.weight")))

    read_mixer(reads, m.mixer, norm, trunk("hyper_connection_mixer"))

def read_mixer(reads, res, norm, site):
    """A hyper-connection's mix-in, and its injection if it has one."""
    reads.read_expr(res.norm, norm(site + ".hc_norm.weight"))
    reads.read(res.down, site + ".input_mix_weight_down.weight")
    reads.read(res.up, site + ".input_mix_weight_up.weight")
    if res.inject != None:
        reads.read(res.inject, site + ".block_inject_weight.weight")

def read_layer(reads, w, layout, norm, n):
    if w.attn != None:
        read_attn(reads, w.attn, n, norm)
        ix = w.indexer
        if ix != None:
            reads.read(ix.qk_proj, n("self_attn.indexer.index_qk_proj.weight"))
            reads.read_expr(ix.q_norm, norm(n("self_attn.indexer.q_layernorm.weight")))
            reads.read_expr(ix.k_norm, norm(n("self_attn.indexer.k_layernorm.weight")))
    else:
        read_gdn(reads, w.gdn, n)
    read_mixer(reads, w.attn_res, norm, n("attn_hyper_connection"))
    read_mixer(reads, w.mlp_res, norm, n("mlp_hyper_connection"))
    read_routed(reads, w.mlp, n, layout)
