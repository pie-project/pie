# How a Z-Image checkpoint is laid out: a diffusers pipeline, its components
# under `dit.`, `te.` and `vae.`; or, for a model of the transformer alone,
# a bare transformer state_dict.

load("//lib/diffusion/formats.star", "biased")
load("//lib/flux_vae/formats.star", vae_read = "read")
load("//lib/qwen3_text/formats.star", te_read = "read")
load("//lib/reads/formats.star", "product")

DIFFUSERS = "a diffusers pipeline (`dit.`/`te.` prefixes)"
BARE = "a bare transformer state_dict"
T_FLIP = 1000.0
SHIFT_FACTOR = 0.1159

def formats(m):
    pipeline = lambda checkpoint: checkpoint.has_prefix("dit.")
    out = [format(DIFFUSERS, recognizes = pipeline, read = lambda reads: read(m, reads, "dit."))]
    if m.te == None:
        out.append(format(
            BARE,
            recognizes = lambda checkpoint: not pipeline(checkpoint),
            read = lambda reads: read(m, reads, ""),
        ))
    return out

def read(m, reads, prefix):
    dit(reads, m.dit, prefix)
    if m.te != None:
        te_read(reads, m.te, "te.model.")
    if m.vae != None:
        constant_of(reads, m.vae.latent.shift, "vae.decoder.conv_in.bias", SHIFT_FACTOR)
        vae_read(reads, m.vae, 2 * m.vae.encoder.conv_out.c_out)

def raw_of(w, name, what):
    held = stored(name)
    if held.raw == None:
        fail("`{}`: `{}` is stored {}; {}".format(w.name, name, held, what))
    return held.raw

def pad_table(reads, w, token):
    dt = raw_of(w, token, "a pad token is a raw float row")
    dim = w.shape[0] // 2
    reads.read_expr(w, concat(0, [
        constant(w.name, [-1.0] * dim, [dim, 1], dt),
        src(token).transmute([dim, 1], raw(dt)),
    ]))

def constant_of(reads, w, seed, value):
    held = stored(seed)
    dt = raw_of(w, seed, "a constant is stated in a raw dtype")
    expr = constant(w.name, [value] * product(w.shape), w.shape, dt)
    want = encoding(w.dtype)
    if want != held:
        expr = expr.cast(want)
    reads.push(tensor(w.name, expr, want, shape = w.shape))

def dit(reads, m, prefix):
    at = lambda tail: prefix + tail
    biased(reads, m.x_embed, at("all_x_embedder.2-1"))
    pad_table(reads, m.x_pad_mod, at("x_pad_token"))
    reads.read(m.cap_norm, at("cap_embedder.0.weight"))
    biased(reads, m.cap_embed, at("cap_embedder.1"))
    pad_table(reads, m.cap_pad_mod, at("cap_pad_token"))
    biased(reads, m.t_mlp0, at("t_embedder.mlp.0"))
    biased(reads, m.t_mlp1, at("t_embedder.mlp.2"))
    constant_of(reads, m.t_flip, at("t_embedder.mlp.0.bias"), T_FLIP)
    for stem, blocks in [
        ("noise_refiner", m.noise_refiner),
        ("context_refiner", m.context_refiner),
        ("layers", m.layers),
    ]:
        for i, b in enumerate(blocks):
            transformer_block(reads, b, at("{}.{}".format(stem, i)))
    biased(reads, m.final_ada, at("all_final_layer.2-1.adaLN_modulation.1"))
    biased(reads, m.final_linear, at("all_final_layer.2-1.linear"))

def transformer_block(reads, b, stem):
    n = lambda s: stem + "." + s
    if b.ada != None:
        biased(reads, b.ada, n("adaLN_modulation.0"))
    reads.read(b.attn_norm1, n("attention_norm1.weight"))
    reads.read(b.attn_norm2, n("attention_norm2.weight"))
    reads.read(b.ffn_norm1, n("ffn_norm1.weight"))
    reads.read(b.ffn_norm2, n("ffn_norm2.weight"))
    reads.read_concat(b.attn.qkv, [
        n("attention.to_q.weight"),
        n("attention.to_k.weight"),
        n("attention.to_v.weight"),
    ])
    reads.read(b.attn.q_norm, n("attention.norm_q.weight"))
    reads.read(b.attn.k_norm, n("attention.norm_k.weight"))
    reads.read(b.attn.out, n("attention.to_out.0.weight"))
    reads.read_concat(b.mlp.gate_up, [n("feed_forward.w1.weight"), n("feed_forward.w3.weight")])
    reads.read(b.mlp.down, n("feed_forward.w2.weight"))
