# How a Qwen 3.8 Flash Next checkpoint is laid out: transformers' names, or
# mlx_lm's, with the hyper-connections, the n-gram embedding and the draft
# head beside the trunk.

load("//lib/qwen_gdn/formats.star", "LAYOUTS", "norm_of", "read_attn", "read_gdn", "read_routed", "safetensors_format")
load("//lib/qwen_vision/formats.star", "read_tower")
load("//lib/reads/formats.star", "squeezed")

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
