# How a Qwen 3.8 Flash Next checkpoint is laid out: transformers' names, or
# mlx_lm's, which nest the trunk otherwise, split the experts' gate and up,
# fold one into the norms and lay the tower's patch embedding out channel
# last.

load("//lib/reads/formats.star", "flattened", "squeezed")

TRANSFORMERS = "transformers"
MLX = "mlx_lm"
CHANNELS = 3

def embed_of(layout):
    return "model.language_model.embed_tokens.weight" if layout == TRANSFORMERS else "language_model.model.embed_tokens.weight"

def trunk(layout, tail):
    return ("model.language_model." if layout == TRANSFORMERS else "language_model.model.") + tail

def lm_head_of(layout):
    return "lm_head.weight" if layout == TRANSFORMERS else "language_model.lm_head.weight"

def tower_of(layout, leaf):
    return ("model.visual." if layout == TRANSFORMERS else "vision_tower.") + leaf

def formats(m):
    return [
        format(
            TRANSFORMERS,
            recognizes = lambda checkpoint: has(embed_of(TRANSFORMERS)),
            read = lambda reads: read(m, reads, TRANSFORMERS),
        ),
        format(
            MLX,
            recognizes = lambda checkpoint: has(embed_of(MLX)),
            read = lambda reads: read(m, reads, MLX),
        ),
    ]

def read(m, reads, layout):
    folds = layout == MLX

    def norm(w, name):
        reads.read_expr(w, src(name).bias(-1.0) if folds else src(name))

    reads.read(m.embed, embed_of(layout))
    reads.read(m.head, lm_head_of(layout))

    for l, w in enumerate(m.layers):
        layer_reads(reads, w, layout, norm, lambda tail: trunk(layout, "layers.{}.{}".format(l, tail)))

    mtp = m.mtp
    if mtp != None:
        n = lambda tail: "mtp." + tail
        reads.read(mtp.norm_embed, n("pre_fc_norm_embedding.weight"))
        reads.read(mtp.norm_hidden, n("pre_fc_norm_hidden.weight"))
        reads.read(mtp.fc_embed, n("fc_embedding.weight"))
        reads.read_expr(mtp.fc_hidden, concat(0, [src(n("fc_hidden.weight")) for _ in range(m.streams)]))
        layer_reads(reads, mtp.block, layout, norm, lambda tail: "mtp.layers.0." + tail)
        norm(mtp.mixer.norm, n("hyper_connection_mixer.hc_norm.weight"))
        reads.read(mtp.mixer.down, n("hyper_connection_mixer.input_mix_weight_down.weight"))
        reads.read(mtp.mixer.up, n("hyper_connection_mixer.input_mix_weight_up.weight"))

    t = m.tower
    if t != None:
        v = lambda leaf: tower_of(layout, leaf)
        want = t.patch_embed.shape
        flat = flattened(v("patch_embed.proj.weight"), want, broadcast = True)
        if layout == MLX:
            per = want[1] // CHANNELS
            flat = flat.gather(1, [j * CHANNELS + c for c in range(CHANNELS) for j in range(per)])
        reads.read_expr(t.patch_embed, flat)
        reads.read(t.patch_embed_bias, v("patch_embed.proj.bias"))
        reads.read(t.pos_embed, v("pos_embed.weight"))
        for l, blk in enumerate(t.blocks):
            n = lambda s: v("blocks.{}.{}".format(l, s))
            for weight_, name in [
                (blk.norm1, n("norm1.weight")),
                (blk.norm1_bias, n("norm1.bias")),
                (blk.qkv, n("attn.qkv.weight")),
                (blk.qkv_bias, n("attn.qkv.bias")),
                (blk.proj, n("attn.proj.weight")),
                (blk.proj_bias, n("attn.proj.bias")),
                (blk.norm2, n("norm2.weight")),
                (blk.norm2_bias, n("norm2.bias")),
                (blk.fc1, n("mlp.linear_fc1.weight")),
                (blk.fc1_bias, n("mlp.linear_fc1.bias")),
                (blk.fc2, n("mlp.linear_fc2.weight")),
                (blk.fc2_bias, n("mlp.linear_fc2.bias")),
            ]:
                reads.read(weight_, name)
        mg = t.merger
        for weight_, name in [
            (mg.norm, v("merger.norm.weight")),
            (mg.norm_bias, v("merger.norm.bias")),
            (mg.fc1, v("merger.linear_fc1.weight")),
            (mg.fc1_bias, v("merger.linear_fc1.bias")),
            (mg.fc2, v("merger.linear_fc2.weight")),
            (mg.fc2_bias, v("merger.linear_fc2.bias")),
        ]:
            reads.read(weight_, name)

    p = m.ple
    if p != None:
        n = lambda tail: trunk(layout, "layers.{}.ple.{}".format(p.layer, tail))
        reads.read_concat(p.table, [
            n("ple_embedding.ngram_embedding.shard_{}.weight".format(i))
            for i in range(p.shards)
        ])
        reads.read(p.key_proj, n("key_proj.weight"))
        reads.read(p.value_proj, n("value_proj.weight"))
        norm(p.norm_key, n("norm_key.weight"))
        norm(p.norm_query, n("norm_query.weight"))
        norm(p.norm_conv, n("norm_conv.weight"))
        reads.read_expr(p.conv, squeezed(n("conv1d.weight")))

    norm(m.mixer.norm, trunk(layout, "hyper_connection_mixer.hc_norm.weight"))
    reads.read(m.mixer.down, trunk(layout, "hyper_connection_mixer.input_mix_weight_down.weight"))
    reads.read(m.mixer.up, trunk(layout, "hyper_connection_mixer.input_mix_weight_up.weight"))

def layer_reads(reads, w, layout, norm, n):
    x = w.mixer
    if x.kind == "attn":
        a = x.attn
        reads.read(a.qg_proj, n("self_attn.q_proj.weight"))
        reads.read(a.k_proj, n("self_attn.k_proj.weight"))
        reads.read(a.v_proj, n("self_attn.v_proj.weight"))
        reads.read(a.o_proj, n("self_attn.o_proj.weight"))
        norm(a.q_norm, n("self_attn.q_norm.weight"))
        norm(a.k_norm, n("self_attn.k_norm.weight"))
        ix = x.indexer
        if ix != None:
            reads.read(ix.qk_proj, n("self_attn.indexer.index_qk_proj.weight"))
            norm(ix.q_norm, n("self_attn.indexer.q_layernorm.weight"))
            norm(ix.k_norm, n("self_attn.indexer.k_layernorm.weight"))
    else:
        reads.read_concat(x.in_qkvz, [n("linear_attn.in_proj_qkv.weight"), n("linear_attn.in_proj_z.weight")])
        reads.read_concat(x.in_ba, [n("linear_attn.in_proj_b.weight"), n("linear_attn.in_proj_a.weight")])
        reads.read_expr(x.conv, squeezed(n("linear_attn.conv1d.weight")))
        reads.read(x.dt_bias, n("linear_attn.dt_bias"))
        reads.read(x.a_log, n("linear_attn.A_log"))
        reads.read(x.norm, n("linear_attn.norm.weight"))
        reads.read(x.out_proj, n("linear_attn.out_proj.weight"))

    for res, site in [(w.attn_res, "attn_hyper_connection"), (w.mlp_res, "mlp_hyper_connection")]:
        norm(res.norm, n(site + ".hc_norm.weight"))
        reads.read(res.down, n(site + ".input_mix_weight_down.weight"))
        reads.read(res.up, n(site + ".input_mix_weight_up.weight"))
        if res.inject != None:
            reads.read(res.inject, n(site + ".block_inject_weight.weight"))

    f = w.mlp
    reads.read(f.router, n("mlp.gate.weight"))
    if layout == TRANSFORMERS:
        reads.read(f.gate_up, n("mlp.experts.gate_up_proj"))
        reads.read(f.down, n("mlp.experts.down_proj"))
    else:
        reads.read_concat(f.gate_up, [n("mlp.switch_mlp.gate_proj.weight"), n("mlp.switch_mlp.up_proj.weight")])
        reads.read(f.down, n("mlp.switch_mlp.down_proj.weight"))
    reads.read_concat(f.shared_gate_up, [n("mlp.shared_expert.gate_proj.weight"), n("mlp.shared_expert.up_proj.weight")])
    reads.read(f.shared_down, n("mlp.shared_expert.down_proj.weight"))
    reads.read(f.shared_gate, n("mlp.shared_expert_gate.weight"))
