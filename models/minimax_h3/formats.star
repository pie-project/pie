# How a MiniMax H3 checkpoint is laid out: a pipeline's components under
# `dit.` and `te.`, or (for the miniature, which carries no text encoder) a
# bare transformer state_dict. The fused qkv interleaves q, k and v head by
# head, and the adaLN projections' slices come in another order.

load("//lib/diffusion/formats.star", "adaln_order", "biased", "reordered")
load("//lib/qwen3_text/formats.star", te_read = "read")

ADALN_SLICES = 6
PIPELINE = "a MiniMax H3 partition (`dit.`/`te.` prefixes)"
BARE = "a bare transformer state_dict"

def formats(m):
    out = [format(
        PIPELINE,
        recognizes = lambda checkpoint: has_prefix("dit."),
        read = lambda reads: read(m, reads, "dit."),
    )]
    if m.te == None:
        out.append(format(
            BARE,
            recognizes = lambda checkpoint: not has_prefix("dit."),
            read = lambda reads: read(m, reads, ""),
        ))
    return out

def modulation(reads, w, stem, order, width):
    rows = lambda e: reordered(e, order, width)
    reads.read_over(w.w, stem + ".weight", rows)
    reads.read_over(w.bias, stem + ".bias", rows)

def attention(reads, m, a, stem):
    d = m.dims
    name = stem + ".qkv_proj.weight"
    raw_dtype = dtype_of(stored(name))
    if raw_dtype == None:
        fail("`{}`: `{}` is stored {}; a fused qkv bank is a raw plane".format(a.qkv.name, name, stored(name)))
    heads = d.heads
    group = d.head_dim * d.dim
    want = a.qkv.shape

    def interleaved(e):
        grouped = e.transmute([3 * heads, group], raw(raw_dtype))
        return concat(0, [grouped.stride(0, which, heads, 3) for which in range(3)]).transmute(want, raw(raw_dtype))

    reads.read_over(a.qkv, name, interleaved)
    reads.read(a.q_norm, stem + ".q_norm.weight")
    reads.read(a.k_norm, stem + ".k_norm.weight")
    reads.read(a.out, stem + ".out_proj.weight")

def block(reads, m, b, stem):
    reads.read(b.norm1, stem + ".norm1.weight")
    reads.read(b.norm2, stem + ".norm2.weight")
    attention(reads, m, b.attn, stem + ".attn")
    reads.read(b.mlp.fc1, stem + ".mlp.fc1.weight")
    reads.read(b.mlp.fc2, stem + ".mlp.fc2.weight")

def read(m, reads, prefix):
    d = m.dims
    dit = m.dit
    at = lambda tail: prefix + tail

    biased(reads, dit.video_patch, at("video_patch_proj"))
    biased(reads, dit.audio_patch, at("audio_patch_proj"))
    biased(reads, dit.condition, at("condition_proj"))
    biased(reads, dit.t_in, at("time_embedder.proj_in"))
    biased(reads, dit.t_out, at("time_embedder.proj_out"))

    for i, b in enumerate(dit.refine):
        block(reads, m, b, at("token_refiner.blocks.{}".format(i)))
    reads.read(dit.refine_norm, at("token_refiner.final_norm.weight"))

    for i, b in enumerate(dit.blocks):
        stem = at("blocks.{}".format(i))
        block(reads, m, b, stem)
        for k, w in enumerate(b.adaln):
            base = k * ADALN_SLICES
            order = [base + s for s in adaln_order(ADALN_SLICES)]
            modulation(reads, w, stem + ".adaln_proj.linear", order, d.dim)

    reads.read(dit.final_norm, at("final_layer.norm.weight"))
    modulation(reads, dit.final_adaln, at("final_layer.adaln_proj.linear"), adaln_order(2), d.dim)
    biased(reads, dit.video_out, at("final_layer.video_out"))
    biased(reads, dit.audio_out, at("final_layer.audio_out"))

    if m.te != None:
        te_read(reads, m.te, "te.model.language_model.")
