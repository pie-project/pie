# How a mini-dit checkpoint is laid out: the reference implementation's
# names, its adaLN vectors' slices in another order.

HIDDEN = 256
MOD_SLICES = 6

def formats(m):
    return [format("reference", read = lambda reads: read(m, reads))]

def biased(reads, w, stem):
    reads.read(w.w, stem + ".weight")
    reads.read(w.bias, stem + ".bias")

def self_attn(reads, a, stem):
    biased(reads, a.qkv, stem + ".qkv")
    reads.read(a.q_norm, stem + ".norm_q")
    reads.read(a.k_norm, stem + ".norm_k")
    biased(reads, a.out, stem + ".out")

def swiglu(reads, m, stem):
    reads.read_concat(m.gate_up.w, [stem + ".gate_proj.weight", stem + ".up_proj.weight"])
    reads.read_concat(m.gate_up.bias, [stem + ".gate_proj.bias", stem + ".up_proj.bias"])
    biased(reads, m.down, stem + ".down_proj")

def slices_reordered(name, slices, width, axis):
    order = {2: [1, 0], 6: [1, 0, 2, 4, 3, 5]}[slices]
    return concat(axis, [src(name).slice(axis, i * width, width) for i in order])

def reordered(reads, w, stem, slices):
    reads.read_expr(w.w, slices_reordered(stem + ".weight", slices, HIDDEN, 0))
    reads.read_expr(w.bias, slices_reordered(stem + ".bias", slices, HIDDEN, 0))

def read(m, reads):
    biased(reads, m.x_embed, "x_embedder")
    biased(reads, m.final_proj, "final_proj")
    reordered(reads, m.final_ada, "final_adaLN", 2)

    reordered(reads, m.single.ada, "blocks.0.adaLN", MOD_SLICES)
    self_attn(reads, m.single.attn, "blocks.0.attn")
    swiglu(reads, m.single.mlp, "blocks.0.mlp")

    for side, stem in [(m.double.img, "blocks.1.img"), (m.double.txt, "blocks.1.txt")]:
        reordered(reads, side.ada, stem + "_adaLN", MOD_SLICES)
        self_attn(reads, side.attn, stem + "_attn")
        swiglu(reads, side.mlp, stem + "_mlp")

    cross = m.cross
    reads.read_expr(
        cross.mod_table,
        slices_reordered("blocks.2.mod_table", MOD_SLICES, 1, 0)
            .transmute(cross.mod_table.shape, stored("blocks.2.mod_table")),
    )
    reordered(reads, cross.ada, "blocks.2.adaLN", MOD_SLICES)
    self_attn(reads, cross.self_attn, "blocks.2.self_attn")
    reads.read(cross.norm, "blocks.2.norm_cross.weight")
    reads.read(cross.norm_bias, "blocks.2.norm_cross.bias")
    biased(reads, cross.cross.q, "blocks.2.cross_attn.q")
    biased(reads, cross.cross.kv, "blocks.2.cross_attn.kv")
    reads.read(cross.cross.q_norm, "blocks.2.cross_attn.norm_q")
    reads.read(cross.cross.k_norm, "blocks.2.cross_attn.norm_k")
    biased(reads, cross.cross.out, "blocks.2.cross_attn.out")
    swiglu(reads, cross.mlp, "blocks.2.mlp")
