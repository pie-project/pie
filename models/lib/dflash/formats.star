# How a DFlash drafter is read from a checkpoint it was imported into: under
# `aux.`, as the drafter's own release spells it. `norm(name)` reads one of
# its norms as the trunk's checkpoint states norms.

load("//lib/reads/formats.star", "flattened")

def bind_aux(d, reads, norm):
    reads.read_expr(d.hidden_norm, norm("aux.hidden_norm.weight"))
    span = d.fc[0].shape[1]
    for i, bank in enumerate(d.fc):
        reads.read_expr(bank, src("aux.fc.weight").slice(1, span * i, span))
    for l, block in enumerate(d.blocks):
        n = lambda s: "aux.layers.{}.{}".format(l, s)
        a = block.attn
        reads.read_expr(block.mixer_norm, norm(n("input_layernorm.weight")))
        reads.read(a.q_proj, n("self_attn.q_proj.weight"))
        reads.read(a.k_proj, n("self_attn.k_proj.weight"))
        reads.read(a.v_proj, n("self_attn.v_proj.weight"))
        reads.read(a.o_proj, n("self_attn.o_proj.weight"))
        for bias, leaf in [(a.q_bias, "q_proj"), (a.k_bias, "k_proj"), (a.v_bias, "v_proj"), (a.o_bias, "o_proj")]:
            if bias != None:
                reads.read(bias, n("self_attn.{}.bias".format(leaf)))
        reads.read_expr(a.q_norm, norm(n("self_attn.q_norm.weight")))
        reads.read_expr(a.k_norm, norm(n("self_attn.k_norm.weight")))
        reads.read_expr(block.mlp_norm, norm(n("post_attention_layernorm.weight")))
        for c, which in [(block.attn_conv, "attention_conv"), (block.mlp_conv, "mlp_conv")]:
            if c != None:
                reads.read_expr(c.base, flattened(n("{}.base_kernel".format(which)), c.base.shape))
                reads.read(c.proj, n("{}.kernel_projection.weight".format(which)))
        reads.read_concat(block.mlp.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
        reads.read(block.mlp.down, n("mlp.down_proj.weight"))
    reads.read_expr(d.norm, norm("aux.norm.weight"))
    sel = d.selector
    if sel == None:
        return
    if sel.kind == "selector":
        if sel.hidden_projection != None:
            reads.read(sel.hidden_projection, "aux.candidate_selector.hidden_projection.weight")
        reads.read(sel.pred, "aux.candidate_selector.predecessor_codebook")
        reads.read(sel.succ, "aux.candidate_selector.successor_codebook")
    else:
        reads.read(sel.pred, "aux.markov_head.markov_w1.weight")
        reads.read(sel.succ, "aux.markov_head.markov_w2.weight")
