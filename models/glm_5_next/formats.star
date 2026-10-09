# How a GLM-5.3-Flash checkpoint is laid out: mlx's names; or an artifact
# already imported, its draft head and vision tower overlaid under `aux.`
# (`pie model import --aux`), the trunk read as the artifact holds it.

load("//lib/kda/formats.star", kda_conv = "conv", kda_gate = "gate", kda_qkv = "qkv")
load("//lib/mla/formats.star", "named")
load("//lib/reads/formats.star", "flattened")

HEAD = "model.language_model.layers.45."
VISUAL = "model.visual."

def formats(m):
    return [
        format(
            "an artifact with an `--aux` overlay",
            recognizes = lambda c: overlaid(m),
            read = lambda reads: own_with_aux(m, reads),
        ),
        format("mlx", recognizes = lambda c: not overlaid(m), read = lambda reads: land(m, reads, False, "")),
    ]

def overlaid(m):
    return (m.mtp != None and has("aux." + HEAD + "enorm.weight")) or (
        m.tower != None and has("aux." + VISUAL + "post_layernorm.weight")
    )

def own_with_aux(m, reads):
    if m.mtp == None and m.tower == None:
        fail("this row declares neither a draft head nor a tower, so there is no overlay to land on an artifact")
    land(m, reads, True, "aux.")

# A read of the source's tensor, or, from an artifact (`own`), the weight as
# the artifact holds it.

def read(reads, own, w, name):
    if own:
        reads.read_own(w)
    else:
        reads.read(w, name)

def read_concat(reads, own, w, names):
    if own:
        reads.read_own(w)
    else:
        reads.read_concat(w, names)

def read_stack(reads, own, w, rows):
    if own:
        reads.read_own(w)
    else:
        reads.read_stack(w, rows)

def read_expr(reads, own, w, expr):
    if own:
        reads.read_own(w)
    else:
        reads.read_expr(w, expr())

def land(m, reads, own, new):
    read(reads, own, m.embed, "model.language_model.embed_tokens.weight")
    read(reads, own, m.final_norm, "model.language_model.norm.weight")
    read(reads, own, m.head, "lm_head.weight")
    for l, w in enumerate(m.layers):
        n = lambda s: "model.language_model.layers.{}.{}".format(l, s)
        read(reads, own, w.attn_mix.scale, n("hc_attn_scale"))
        read(reads, own, w.attn_mix.base, n("hc_attn_base"))
        read(reads, own, w.attn_mix.dynamic, n("hc_attn_fn"))
        read(reads, own, w.mlp_mix.scale, n("hc_ffn_scale"))
        read(reads, own, w.mlp_mix.base, n("hc_ffn_base"))
        read(reads, own, w.mlp_mix.dynamic, n("hc_ffn_fn"))
        read(reads, own, w.mixer_norm, n("input_layernorm.weight"))
        read(reads, own, w.mlp_norm, n("post_attention_layernorm.weight"))
        if w.mixer_kind == "mla":
            mla(reads, own, n, w.mixer)
        else:
            kda(reads, own, n, w.mixer)
        if w.mlp.routed:
            moe(reads, own, n, w.mlp)
        else:
            read_concat(reads, own, w.mlp.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
            read(reads, own, w.mlp.down, n("mlp.down_proj.weight"))

    held = lambda name: own and has(name)
    if m.tower != None:
        tower(reads, held("vision.post_norm"), lambda s: new + VISUAL + s, m.tower)
    mtp = m.mtp
    if mtp == None:
        return
    own = held("mtp.enorm")
    n = lambda s: new + HEAD + s
    read(reads, own, mtp.enorm, n("enorm.weight"))
    read(reads, own, mtp.hnorm, n("hnorm.weight"))
    read_expr(reads, own, mtp.e_proj, lambda: src(n("eh_proj.weight")).slice(1, 0, m.hidden))
    read_expr(reads, own, mtp.h_proj, lambda: src(n("eh_proj.weight")).slice(1, m.hidden, m.hidden))
    read(reads, own, mtp.mixer_norm, n("input_layernorm.weight"))
    read(reads, own, mtp.mlp_norm, n("post_attention_layernorm.weight"))
    mla(reads, own, n, mtp.attn)
    moe(reads, own, n, mtp.mlp)
    read(reads, own, mtp.norm, n("shared_head.norm.weight"))

def tower(reads, own, v, t):
    read_expr(reads, own, t.patch_embed, lambda: flattened(v("patch_embed.proj.weight"), [t.hidden, t.patch_width], broadcast = True))
    read(reads, own, t.patch_embed_bias, v("patch_embed.proj.bias"))
    for l, blk in enumerate(t.blocks):
        n = lambda s: v("blocks.{}.{}".format(l, s))
        for w, name in [
            (blk.norm1, n("norm1.weight")),
            (blk.qkv, n("attn.qkv.weight")),
            (blk.qkv_bias, n("attn.qkv.bias")),
            (blk.q_norm, n("attn.q_norm.weight")),
            (blk.k_norm, n("attn.k_norm.weight")),
            (blk.proj, n("attn.proj.weight")),
            (blk.proj_bias, n("attn.proj.bias")),
            (blk.norm2, n("norm2.weight")),
            (blk.down, n("mlp.down_proj.weight")),
            (blk.down_bias, n("mlp.down_proj.bias")),
        ]:
            read(reads, own, w, name)
        read_concat(reads, own, blk.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
        read_concat(reads, own, blk.gate_up_bias, [n("mlp.gate_proj.bias"), n("mlp.up_proj.bias")])
    read(reads, own, t.post_norm, v("post_layernorm.weight"))

    def downsample():
        c, k = t.hidden, t.merge
        out = t.downsample.shape[0]
        flat = flattened(v("downsample.weight"), [out, c * k * k], broadcast = True)
        return flat.gather(1, [ch * k * k + kk for kk in range(k * k) for ch in range(c)])

    read_expr(reads, own, t.downsample, downsample)
    read(reads, own, t.downsample_bias, v("downsample.bias"))
    mg = t.merger
    read(reads, own, mg.proj, v("merger.proj.weight"))
    read(reads, own, mg.norm, v("merger.post_projection_norm.weight"))
    read(reads, own, mg.norm_bias, v("merger.post_projection_norm.bias"))
    read_concat(reads, own, mg.gate_up, [v("merger.gate_proj.weight"), v("merger.up_proj.weight")])
    read(reads, own, mg.down, v("merger.down_proj.weight"))

def moe(reads, own, n, f):
    read(reads, own, f.router, n("mlp.gate.weight"))
    read(reads, own, f.bias, n("mlp.gate.e_score_correction_bias"))
    read_stack(reads, own, f.gate_up, [
        [n("mlp.experts.{}.gate_proj.weight".format(e)), n("mlp.experts.{}.up_proj.weight".format(e))]
        for e in range(f.experts)
    ])
    read_stack(reads, own, f.down, [[n("mlp.experts.{}.down_proj.weight".format(e))] for e in range(f.experts)])
    read_concat(reads, own, f.shared.gate_up, [
        n("mlp.shared_experts.gate_proj.weight"),
        n("mlp.shared_experts.up_proj.weight"),
    ])
    read(reads, own, f.shared.down, n("mlp.shared_experts.down_proj.weight"))

def mla(reads, own, n, a):
    for w, name in named(a, n):
        read(reads, own, w, name)
    ix = a.indexer
    read(reads, own, ix.wq_b, n("self_attn.indexer.wq_b.weight"))
    read(reads, own, ix.wk, n("self_attn.indexer.wk.weight"))
    read(reads, own, ix.weights_proj, n("self_attn.indexer.weights_proj.weight"))
    read(reads, own, ix.k_norm, n("self_attn.indexer.k_norm.weight"))
    read(reads, own, ix.k_norm_bias, n("self_attn.indexer.k_norm.bias"))
    read(reads, own, ix.kpool_ape, n("self_attn.indexer.index_kpool_compress_ape"))
    read(reads, own, ix.kpool_gate, n("self_attn.indexer.index_kpool_compress_gate"))

def kda(reads, own, n, k):
    read_concat(reads, own, k.qkv, kda_qkv(n))
    read_expr(reads, own, k.conv, lambda: kda_conv(k, n))
    read(reads, own, k.f_a, n("self_attn.f_a_proj.weight"))
    read(reads, own, k.f_b, n("self_attn.f_b_proj.weight"))
    for w, name in kda_gate(k, n):
        read(reads, own, w, name)
    read(reads, own, k.b, n("self_attn.b_proj.weight"))
    read_expr(
        reads,
        own,
        k.dt_bias,
        lambda: src(n("self_attn.dt_bias")).transmute([k.heads, k.head_dim], encoding(dtype.f32)),
    )
    read(reads, own, k.a_log, n("self_attn.A_log"))
    read(reads, own, k.o_norm, n("self_attn.o_norm.weight"))
    read(reads, own, k.o_proj, n("self_attn.o_proj.weight"))
