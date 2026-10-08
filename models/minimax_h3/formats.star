# How a MiniMax H3 checkpoint is laid out: a pipeline's components under
# `dit.` and `te.`, or (for the miniature, which carries no text encoder) a
# bare transformer state_dict. The fused qkv interleaves q, k and v head by
# head, and the adaLN projections' slices come in another order.

ADALN_SLICES = 6
PIPELINE = "a MiniMax H3 partition (`dit.`/`te.` prefixes)"
BARE = "a bare transformer state_dict"

def formats(m):
    out = [format(
        PIPELINE,
        recognizes = lambda checkpoint: has_prefix("dit."),
        read = lambda reads: read(m, reads, "dit.", "te.model.language_model."),
    )]
    if m.te == None:
        out.append(format(
            BARE,
            recognizes = lambda checkpoint: not has_prefix("dit."),
            read = lambda reads: read(m, reads, "", None),
        ))
    return out

def biased(reads, w, stem):
    reads.read(w.w, stem + ".weight")
    reads.read(w.bias, stem + ".bias")

def pairs(reads, w, stem, order, block):
    rows = lambda e: concat(0, [e.slice(0, s * block, block) for s in order])
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

def feedforward(reads, f, stem):
    reads.read(f.fc1, stem + ".fc1.weight")
    reads.read(f.fc2, stem + ".fc2.weight")

def read(m, reads, dit_prefix, te_prefix):
    d = m.dims
    dit = m.dit
    at = lambda tail: dit_prefix + tail

    biased(reads, dit.video_patch, at("video_patch_proj"))
    biased(reads, dit.audio_patch, at("audio_patch_proj"))
    biased(reads, dit.condition, at("condition_proj"))
    biased(reads, dit.t_in, at("time_embedder.proj_in"))
    biased(reads, dit.t_out, at("time_embedder.proj_out"))

    for i, block in enumerate(dit.refine):
        stem = at("token_refiner.blocks.{}".format(i))
        reads.read(block.norm1, stem + ".norm1.weight")
        reads.read(block.norm2, stem + ".norm2.weight")
        attention(reads, m, block.attn, stem + ".attn")
        feedforward(reads, block.mlp, stem + ".mlp")
    reads.read(dit.refine_norm, at("token_refiner.final_norm.weight"))

    for i, block in enumerate(dit.blocks):
        stem = at("blocks.{}".format(i))
        reads.read(block.norm1, stem + ".norm1.weight")
        reads.read(block.norm2, stem + ".norm2.weight")
        attention(reads, m, block.attn, stem + ".attn")
        feedforward(reads, block.mlp, stem + ".mlp")
        bank = stem + ".adaln_proj.linear"
        for k, w in enumerate(block.adaln):
            base = k * ADALN_SLICES
            pairs(reads, w, bank, [base + 1, base, base + 2, base + 4, base + 3, base + 5], d.dim)

    reads.read(dit.final_norm, at("final_layer.norm.weight"))
    pairs(reads, dit.final_adaln, at("final_layer.adaln_proj.linear"), [1, 0], d.dim)
    biased(reads, dit.video_out, at("final_layer.video_out"))
    biased(reads, dit.audio_out, at("final_layer.audio_out"))

    te = m.te
    if te == None:
        return
    if te_prefix == None:
        fail("`te.embed_tokens.weight`: {} carries no text encoder, and this row declares one".format(BARE))
    reads.read(te.embed, te_prefix + "embed_tokens.weight")
    for l, w in enumerate(te.layers):
        n = lambda s: "{}layers.{}.{}".format(te_prefix, l, s)
        reads.read(w.attn_norm, n("input_layernorm.weight"))
        reads.read(w.q, n("self_attn.q_proj.weight"))
        reads.read(w.k, n("self_attn.k_proj.weight"))
        reads.read(w.v, n("self_attn.v_proj.weight"))
        reads.read(w.o, n("self_attn.o_proj.weight"))
        reads.read(w.q_norm, n("self_attn.q_norm.weight"))
        reads.read(w.k_norm, n("self_attn.k_norm.weight"))
        reads.read(w.mlp_norm, n("post_attention_layernorm.weight"))
        reads.read_concat(w.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
        reads.read(w.down, n("mlp.down_proj.weight"))
