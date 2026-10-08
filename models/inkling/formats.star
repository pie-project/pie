# How an Inkling checkpoint is laid out: the language model under
# `model.llm.`, its MLPs' gate and up projections interleaved row by row.

load("//lib/reads/formats.star", "flattened")

TRUNK = "model.llm."

def formats(m):
    return [format("huggingface", read = lambda reads: huggingface(m, reads))]

def huggingface(m, reads):
    at = lambda leaf: TRUNK + leaf
    reads.read(m.embed, at("embed.weight"))
    reads.read(m.embed_norm, at("embed_norm.weight"))
    reads.read(m.final_norm, at("norm.weight"))
    reads.read_expr(m.unembed, src(at("unembed.weight")).slice(0, 0, m.head_rows))
    for l, w in enumerate(m.layers):
        n = lambda leaf: "{}layers.{}.{}".format(TRUNK, l, leaf)
        reads.read(w.attn_norm, n("attn_norm.weight"))
        reads.read(w.mlp_norm, n("mlp_norm.weight"))
        reads.read(w.q_proj, n("attn.wq_du.weight"))
        reads.read(w.k_proj, n("attn.wk_dv.weight"))
        reads.read(w.v_proj, n("attn.wv_dv.weight"))
        reads.read(w.r_proj, n("attn.wr_du.weight"))
        reads.read(w.o_proj, n("attn.wo_ud.weight"))
        reads.read(w.q_norm, n("attn.q_norm.weight"))
        reads.read(w.k_norm, n("attn.k_norm.weight"))
        reads.read(w.rel_proj, n("attn.rel_logits_proj.proj"))
        for weight, leaf in [
            (w.k_conv, "attn.k_sconv.weight"),
            (w.v_conv, "attn.v_sconv.weight"),
            (w.attn_conv, "attn_sconv.weight"),
            (w.mlp_conv, "mlp_sconv.weight"),
        ]:
            reads.read_expr(weight, flattened(n(leaf), weight.shape))
        f = w.mlp
        if not f.routed:
            w13 = src(n("mlp.w13_dn.weight"))
            reads.read_expr(f.gate_up, concat(0, [
                w13.stride(0, 0, f.inter, 2),
                w13.stride(0, 1, f.inter, 2),
            ]))
            reads.read(f.down, n("mlp.w2_md.weight"))
            reads.read(f.scale, n("mlp.global_scale"))
            continue
        reads.read(f.router, n("mlp.gate.weight"))
        reads.read(f.bias, n("mlp.gate.bias"))
        reads.read(f.scale, n("mlp.gate.global_scale"))
        combed = lambda name: concat(1, [
            src(name).stride(1, 0, f.inter, 2),
            src(name).stride(1, 1, f.inter, 2),
        ])
        reads.read_expr(f.gate_up, concat(0, [
            combed(n("mlp.experts.w13_weight")),
            combed(n("mlp.shared_experts.shared_w13_weight")),
        ]))
        reads.read_expr(f.down, concat(0, [
            src(n("mlp.experts.w2_weight")),
            src(n("mlp.shared_experts.shared_w2_weight")),
        ]))
