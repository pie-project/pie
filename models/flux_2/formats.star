# How a FLUX.2 checkpoint is laid out: a diffusers pipeline, its components
# under `dit.`, `te.` and `vae.`; or, for a model of the transformer alone,
# a bare transformer state_dict.

load("//lib/diffusion/formats.star", "adaln_order", "conv", "reordered")
load("//lib/flux_vae/formats.star", "conv_head", vae_read = "read")
load("//lib/qwen3_text/formats.star", te_read = "read")

DIFFUSERS = "a diffusers pipeline (`dit.`/`te.`/`vae.` prefixes)"
BARE = "a bare transformer state_dict"
BN_EPS = 1e-4

def formats(m):
    pipeline = lambda checkpoint: checkpoint.has_prefix("dit.")
    out = [format(DIFFUSERS, recognizes = pipeline, read = lambda reads: read(m, reads, "dit."))]
    if m.te == None and m.vae == None:
        out.append(format(
            BARE,
            recognizes = lambda checkpoint: not pipeline(checkpoint),
            read = lambda reads: read(m, reads, ""),
        ))
    return out

def read(m, reads, prefix):
    dit(reads, m.dit, m.dims.dim, prefix)
    if m.te != None:
        te_read(reads, m.te, "te.model.")
        hidden = m.te.hidden
        for i, w in enumerate(m.te_context):
            column_block(reads, w, "dit.context_embedder.weight", hidden * i, hidden)
    if m.vae != None:
        vae(reads, m.vae)

def dit(reads, m, dim, prefix):
    w = lambda tail: prefix + tail + ".weight"
    reads.read(m.x_embed, w("x_embedder"))
    if m.context_embed != None:
        reads.read(m.context_embed, w("context_embedder"))
    reads.read(m.t_embed.linear_1, w("time_guidance_embed.timestep_embedder.linear_1"))
    reads.read(m.t_embed.linear_2, w("time_guidance_embed.timestep_embedder.linear_2"))
    if m.g_embed != None:
        reads.read(m.g_embed.linear_1, w("time_guidance_embed.guidance_embedder.linear_1"))
        reads.read(m.g_embed.linear_2, w("time_guidance_embed.guidance_embedder.linear_2"))

    modulation(reads, m.mod_img, w("double_stream_modulation_img.linear"), 6, dim)
    modulation(reads, m.mod_txt, w("double_stream_modulation_txt.linear"), 6, dim)
    modulation(reads, m.mod_single, w("single_stream_modulation.linear"), 3, dim)

    for i, block in enumerate(m.double):
        stem = prefix + "transformer_blocks.{}".format(i)
        attn(reads, block.img.attn, stem, ["to_q", "to_k", "to_v"], ["norm_q", "norm_k"], "to_out.0")
        swiglu(reads, block.img.ff, stem + ".ff")
        attn(
            reads,
            block.txt.attn,
            stem,
            ["add_q_proj", "add_k_proj", "add_v_proj"],
            ["norm_added_q", "norm_added_k"],
            "to_add_out",
        )
        swiglu(reads, block.txt.ff, stem + ".ff_context")

    for i, block in enumerate(m.single):
        stem = prefix + "single_transformer_blocks.{}.attn".format(i)
        reads.read(block.in_proj, stem + ".to_qkv_mlp_proj.weight")
        reads.read(block.q_norm, stem + ".norm_q.weight")
        reads.read(block.k_norm, stem + ".norm_k.weight")
        out = stem + ".to_out.weight"
        column_block(reads, block.out_attn, out, 0, dim)
        column_block(reads, block.out_mlp, out, dim, block.out_mlp.shape[1])

    reads.read(m.norm_out, w("norm_out.linear"))
    reads.read(m.proj_out, w("proj_out"))

def attn(reads, a, stem, qkv, norms, out):
    n = lambda s: "{}.attn.{}.weight".format(stem, s)
    reads.read_concat(a.qkv, [n(s) for s in qkv])
    reads.read(a.q_norm, n(norms[0]))
    reads.read(a.k_norm, n(norms[1]))
    reads.read(a.out, n(out))

def swiglu(reads, ff, stem):
    reads.read(ff.linear_in, stem + ".linear_in.weight")
    reads.read(ff.linear_out, stem + ".linear_out.weight")

def modulation(reads, w, name, slices, dim):
    reads.read_expr(w, reordered(src(name), adaln_order(slices), dim))

def column_block(reads, w, name, start, length):
    held = stored(name)
    want = encoding(w.dtype)
    sliced = src(name).slice(1, start, length)
    if held == want:
        reads.read_expr(w, sliced)
        return
    staged = w.name + ".read"
    reads.push(tensor(staged, sliced, held, shape = w.shape, internal = True))
    reads.push(tensor(w.name, out(staged).cast(want), want, shape = w.shape))

def vae(reads, v):
    lat = v.latent
    batch_norm(reads, lat, "vae.bn.running_mean", "vae.bn.running_var")
    conv(reads, lat.post_quant_conv, "vae.post_quant_conv")
    conv_head(reads, lat.quant_conv, "vae.quant_conv", lat.quant_conv.c_in)
    vae_read(reads, v)

def batch_norm(reads, v, mean, var):
    held = stored(var)
    shape = v.bn_scale.shape
    var_eps = v.bn_scale.name + ".var_eps"
    reads.push(tensor(var_eps, src(var).bias(BN_EPS), held, shape = shape, internal = True))
    for plane, op in [(v.bn_scale, "sqrt"), (v.bn_rscale, "rsqrt")]:
        root = plane.name + ".of_var"
        reads.push(tensor(root, out(var_eps).unary(op), held, shape = shape, internal = True))
        want = encoding(plane.dtype)
        reads.push(tensor(plane.name, out(root) if want == held else out(root).cast(want), want, shape = shape))
    reads.read(v.bn_mean, mean)
    want = encoding(v.bn_zero.dtype)
    if want.raw == None:
        fail("`{}`: a zero plane is stated raw, not {}".format(v.bn_zero.name, want))
    shape = v.bn_zero.shape
    reads.push(tensor(v.bn_zero.name, fill(0.0, shape, raw(want.raw)), want, shape = shape))
