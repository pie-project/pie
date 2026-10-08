# How a DeepSeek-V4 checkpoint is laid out. V4.1-Flash: its per-expert
# release layout, or MLX's stacked one. V4-Flash and the base: an artifact
# with an `--aux` overlay, the flash mlx spelling, transformers' deepseek-v3
# one, or GGUF's; the flash spellings share their names, so a reading is what
# tells them apart.

AFFINE = [dtype.u4g64, dtype.u8g64, dtype.u4g32, dtype.u2g32, dtype.u2g64, dtype.u2g128]

def formats(m):
    if m.hyper.single_pass:
        return [
            format(
                "a DeepSeek-V4.1 checkpoint",
                recognizes = lambda checkpoint: has("embed.weight"),
                read = lambda reads: v41(m, reads, False),
            ),
            format(
                "a DeepSeek-V4.1 MLX checkpoint",
                recognizes = lambda checkpoint: has("model.layers.0.ffn.switch_mlp.gate_proj.weight"),
                read = lambda reads: v41_mlx(m, reads),
            ),
        ]
    return [
        format("an artifact with an `--aux` overlay", read = lambda reads: own_with_aux(m, reads)),
        format("flash mlx", read = lambda reads: flash_mlx(m, reads)),
        format("huggingface", read = lambda reads: huggingface(m, reads)),
        format("gguf", recognizes = lambda checkpoint: has("token_embd.weight"), read = lambda reads: gguf(m, reads)),
    ]

# ---------------------------------------------------------------------------
# The flash mlx spelling.
# ---------------------------------------------------------------------------

def one(w, name):
    return ("one", w, 0, [name])

def joined(w, axis, names):
    return ("concat", w, axis, names)

def layer_reads(w, n, reads):
    for mix, tag in [(w.attn_mix, "attn_hc"), (w.mlp_mix, "ffn_hc")]:
        reads.append(one(mix.scale, n(tag + ".scale")))
        reads.append(one(mix.base, n(tag + ".base")))
        if mix.dynamic != None:
            reads.append(one(mix.dynamic, n(tag + ".fn")))
    if w.attn_norm != None:
        reads.append(one(w.attn_norm, n("attn_norm.weight")))
    if w.mlp_norm != None:
        reads.append(one(w.mlp_norm, n("ffn_norm.weight")))
    at = w.attn
    reads.append(one(at.q_down, n("attn.wq_a.weight")))
    reads.append(one(at.q_norm, n("attn.q_norm.weight")))
    reads.append(one(at.q_up, n("attn.wq_b.weight")))
    reads.append(one(at.kv_down, n("attn.wkv.weight")))
    reads.append(one(at.kv_norm, n("attn.kv_norm.weight")))
    reads.append(one(at.o_down, n("attn.wo_a.weight")))
    reads.append(one(at.o_up, n("attn.wo_b.weight")))
    reads.append(one(at.sink, n("attn.attn_sink")))
    if at.pool != None and at.pool.compressor != None:
        c = at.pool.compressor
        reads.append(one(c.wkv, n("attn.compressor.wkv.weight")))
        if c.wgate != None:
            reads.append(one(c.wgate, n("attn.compressor.wgate.weight")))
        if c.ape != None:
            reads.append(one(c.ape, n("attn.compressor.ape")))
        reads.append(one(c.norm, n("attn.compressor.norm.weight")))
    ix = at.indexer
    if ix != None:
        reads.append(one(ix.wq_b, n("attn.indexer.wq_b.weight")))
        reads.append(one(ix.weights_proj, n("attn.indexer.weights_proj.weight")))
        if ix.compressor != None:
            c = ix.compressor
            reads.append(one(c.wkv, n("attn.indexer.compressor.wkv.weight")))
            if c.wgate != None:
                reads.append(one(c.wgate, n("attn.indexer.compressor.wgate.weight")))
            if c.ape != None:
                reads.append(one(c.ape, n("attn.indexer.compressor.ape")))
            reads.append(one(c.norm, n("attn.indexer.compressor.norm.weight")))
    f = w.mlp
    if f.kind == "moe_flash":
        if f.gate.kind == "hash":
            reads.append(one(f.router, n("ffn.gate.weight")))
            reads.append(one(f.gate.tid2eid, n("ffn.gate.tid2eid")))
        else:
            reads.append(one(f.router, n("ffn.gate.weight")))
            reads.append(one(f.gate.bias, n("ffn.gate.e_score_correction_bias")))
        if f.gate_up.fused != None:
            reads.append(joined(f.gate_up.fused, 1, [
                n("ffn.switch_mlp.gate_proj.weight"),
                n("ffn.switch_mlp.up_proj.weight"),
            ]))
        else:
            reads.append(one(f.gate_up.gate, n("ffn.switch_mlp.gate_proj.weight")))
            reads.append(one(f.gate_up.up, n("ffn.switch_mlp.up_proj.weight")))
        reads.append(one(f.down, n("ffn.switch_mlp.down_proj.weight")))
        reads.append(joined(f.shared_gate_up, 0, [
            n("ffn.shared_experts.gate_proj.weight"),
            n("ffn.shared_experts.up_proj.weight"),
        ]))
        reads.append(one(f.shared_down, n("ffn.shared_experts.down_proj.weight")))

def mlx_reads(m):
    reads = [one(m.embed, "model.embed_tokens.weight")]
    if m.head != None:
        reads.append(one(m.head, "lm_head.weight"))
    reads.append(one(m.final_norm, "model.norm.weight"))
    if m.hc_head != None:
        reads.append(one(m.hc_head.base, "model.hc_head.base"))
        reads.append(one(m.hc_head.dynamic, "model.hc_head.fn"))
        reads.append(one(m.hc_head.scale, "model.hc_head.scale"))
    for l, w in enumerate(m.layers):
        layer_reads(w, lambda s: "model.layers.{}.{}".format(l, s), reads)
    if m.mtp != None:
        mtp = m.mtp
        reads.append(one(mtp.enorm, "aux.enorm.weight"))
        reads.append(one(mtp.hnorm, "aux.hnorm.weight"))
        reads.append(one(mtp.e_proj, "aux.e_proj.weight"))
        reads.append(joined(mtp.h_proj, 0, ["aux.h_proj.weight"] * m.hyper.streams))
        layer_reads(mtp.block, lambda s: "aux.decoder." + s, reads)
        reads.append(one(mtp.hc_head.base, "aux.hc_head.base"))
        reads.append(one(mtp.hc_head.dynamic, "aux.hc_head.fn"))
        reads.append(one(mtp.hc_head.scale, "aux.hc_head.scale"))
        reads.append(one(mtp.norm, "aux.norm.weight"))
    return reads

def concat_expr(m, w, axis, names):
    parts = [
        src(name).transmute([-1, -1, m.hidden], encoding(w.dtype)) if len(w.shape) == 3 else src(name)
        for name in names
    ]
    return concat(axis, parts)

def flash_mlx(m, reads):
    for kind, w, axis, names in mlx_reads(m):
        if kind == "one":
            reads.read(w, names[0])
        elif w.dtype in AFFINE:
            reads.read_concat(w, names)
        else:
            reads.read_expr(w, concat_expr(m, w, axis, names))

def own_with_aux(m, reads):
    if m.mtp == None:
        fail("this row declares no draft head, so there is no overlay to land on an artifact")
    is_aux = lambda name: name.startswith("aux.")
    for kind, w, axis, names in mlx_reads(m):
        if kind == "one" and is_aux(names[0]):
            reads.read(w, names[0])
        elif kind == "concat" and all([is_aux(n) for n in names]) and w.dtype in AFFINE:
            reads.read_concat(w, names)
        elif kind == "concat" and all([is_aux(n) for n in names]):
            reads.read_expr(w, concat_expr(m, w, axis, names))
        else:
            reads.read_own(w)

# ---------------------------------------------------------------------------
# V4.1-Flash.
# ---------------------------------------------------------------------------

def v41_mlx(m, reads):
    if not has("model.layers.0.ffn.switch_mlp.gate_proj.weight"):
        fail("no stacked MLX V4.1 expert bank in model.layers")
    v41(m, reads, True)

def read_bank(reads, w, parts):
    """A routed bank `[experts, rows, cols]` from one plane per expert, at
    whatever representation the file stores them: bf16 rows are stacked,
    MLX affine trios by the affine reader, and MXFP4 codes (`w.weight` nibble
    pairs beside `w.scale` e8m0 exponents) with their exponents."""
    rows, cols = w.shape[1], w.shape[2]
    if w.dtype == dtype.mxfp4:
        codes = []
        scales = []
        for part in parts:
            stem = part[:-len(".weight")] if part.endswith(".weight") else part
            scale = stem + ".scale"
            if not has(scale):
                fail("`{}`: `{}` is read as MXFP4 codes, whose exponents are stored beside it as `{}`, and the checkpoint holds none".format(w.name, part, scale))
            codes.append(src(part).transmute([1, rows, cols], encoding(dtype.mxfp4)))
            scales.append(src(scale).transmute([1, rows, cols // 32], encoding(dtype.e8m0)))
        reads.extend([
            tensor(w.name, concat(0, codes), encoding(dtype.mxfp4)),
            tensor(scales_name(w.name), concat(0, scales), encoding(dtype.e8m0), shape = scales_shape(w), scaling = w),
        ])
    elif w.dtype in AFFINE:
        reads.read_stack(w, [[part] for part in parts])
    else:
        reads.read_expr(w, concat(0, [src(part).transmute([1, rows, cols], encoding(w.dtype)) for part in parts]))

def v41(m, reads, mlx):
    reads.read(m.embed, "model.embed_tokens.weight" if mlx else "embed.weight")
    if m.head != None:
        reads.read(m.head, "lm_head.weight" if mlx else "head.weight")
    reads.read(m.final_norm, "model.norm.weight" if mlx else "norm.weight")
    if m.token_map != None:
        if not has("engram.token_map"):
            fail("`{}`: Engram hashes tokenizer-compressed ids, and this checkpoint carries no `engram.token_map`; write it beside the weights with `scripts/bench/engram_token_map.py` first".format(m.token_map.name))
        reads.read(m.token_map, "engram.token_map")

    for l, w in enumerate(m.layers):
        n = lambda s: ("model.layers.{}.{}" if mlx else "layers.{}.{}").format(l, s)
        at = w.attn
        for mix, tag in [(w.attn_mix, "attn"), (w.mlp_mix, "ffn")]:
            name = lambda plane: n("{}_hc.{}".format(tag, plane)) if mlx else n("hc_{}_{}".format(tag, plane))
            reads.read(mix.scale, name("scale"))
            reads.read(mix.base, name("base"))
            if mix.dynamic != None:
                reads.read(mix.dynamic, name("fn"))
        if w.attn_norm != None:
            reads.read(w.attn_norm, n("attn_norm.weight"))
        if w.mlp_norm != None:
            reads.read(w.mlp_norm, n("ffn_norm.weight"))

        reads.read(at.q_down, n("attn.wq_a.weight"))
        reads.read(at.q_norm, n("attn.q_norm.weight"))
        reads.read(at.q_up, n("attn.wq_b.weight"))
        reads.read(at.kv_down, n("attn.wkv.weight"))
        reads.read(at.kv_norm, n("attn.kv_norm.weight"))
        reads.read(at.o_down, n("attn.wo_a.weight"))
        reads.read(at.o_up, n("attn.wo_b.weight"))
        reads.read(at.sink, n("attn.attn_sink"))

        if at.pool != None and at.pool.compressor != None:
            c = at.pool.compressor
            reads.read(c.wkv, n("attn.compressor.wkv.weight"))
            if c.wgate != None:
                reads.read(c.wgate, n("attn.compressor.wgate.weight"))
            reads.read(c.norm, n("attn.compressor.norm.weight"))
        ix = at.indexer
        if ix != None:
            reads.read(ix.wq_b, n("attn.indexer.wq_b.weight"))
            reads.read(ix.weights_proj, n("attn.indexer.weights_proj.weight"))
            if ix.wk != None:
                reads.read(ix.wk, n("attn.indexer.wk.weight"))
            if ix.k_norm != None:
                reads.read(ix.k_norm, n("attn.indexer.k_norm.weight"))

        f = w.mlp
        if f.kind != "moe_flash":
            fail("every V4.1 layer is a DeepSeekMoE block")
        reads.read(f.router, n("ffn.gate.weight"))
        if f.gate.kind != "bias":
            fail("V4.1 routes by bias-corrected scores on every layer")
        reads.read(f.gate.bias, n("ffn.gate.e_score_correction_bias" if mlx else "ffn.gate.bias"))
        expert = lambda e, leg: n("ffn.experts.{}.{}.weight".format(e, leg))
        if f.gate_up.fused != None:
            fail("V4.1 keeps its routed gate and up legs apart")
        if mlx:
            reads.read(f.gate_up.gate, n("ffn.switch_mlp.gate_proj.weight"))
            reads.read(f.gate_up.up, n("ffn.switch_mlp.up_proj.weight"))
        else:
            read_bank(reads, f.gate_up.gate, [expert(e, "w1") for e in range(f.experts)])
            read_bank(reads, f.gate_up.up, [expert(e, "w3") for e in range(f.experts)])
        if mlx:
            reads.read(f.down, n("ffn.switch_mlp.down_proj.weight"))
        else:
            read_bank(reads, f.down, [expert(e, "w2") for e in range(f.experts)])
        reads.read_concat(f.shared_gate_up, [
            n("ffn.shared_experts.gate_proj.weight" if mlx else "ffn.shared_experts.w1.weight"),
            n("ffn.shared_experts.up_proj.weight" if mlx else "ffn.shared_experts.w3.weight"),
        ])
        reads.read(f.shared_down, n("ffn.shared_experts.down_proj.weight" if mlx else "ffn.shared_experts.w2.weight"))

        if w.engram != None:
            e = w.engram
            reads.read(e.table, n("engram.embed.weight"))
            reads.read(e.wkv, n("engram.wkv.weight"))
            # The gate's normalisations carry `weight + 1`.
            reads.read_over(e.q_weight, n("engram.q_weight"), lambda x: x.bias(-1.0))
            reads.read_over(e.k_weight, n("engram.k_weight"), lambda x: x.bias(-1.0))

# ---------------------------------------------------------------------------
# transformers' deepseek-v3 spelling, and GGUF's.
# ---------------------------------------------------------------------------

def huggingface(m, reads):
    if m.mtp != None:
        fail("this SKU declares a draft head, which only the flash mlx reading (with an `--aux` overlay) lands")
    reads.read(m.embed, "model.embed_tokens.weight")
    reads.read(m.final_norm, "model.norm.weight")
    for l, w in enumerate(m.layers):
        n = lambda s: "model.layers.{}.{}".format(l, s)
        at = w.attn
        reads.read(w.attn_mix.scale, n("hc_attn_scale"))
        reads.read(w.attn_mix.base, n("hc_attn_base"))
        reads.read(w.mlp_mix.scale, n("hc_mlp_scale"))
        reads.read(w.mlp_mix.base, n("hc_mlp_base"))
        reads.read(at.q_down, n("self_attn.q_a_proj.weight"))
        reads.read(at.q_norm, n("self_attn.q_a_layernorm.weight"))
        reads.read(at.q_up, n("self_attn.q_b_proj.weight"))
        reads.read(at.kv_down, n("self_attn.kv_a_proj_with_mqa.weight"))
        reads.read(at.kv_norm, n("self_attn.kv_a_layernorm.weight"))
        reads.read(at.o_down, n("self_attn.o_a_proj.weight"))
        reads.read(at.o_up, n("self_attn.o_b_proj.weight"))
        reads.read(at.sink, n("self_attn.sinks"))
        f = w.mlp
        if f.kind == "dense":
            reads.read_concat(f.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
            reads.read(f.down, n("mlp.down_proj.weight"))
        elif f.kind == "routed":
            reads.read(f.router, n("mlp.gate.weight"))
            reads.read(f.bias, n("mlp.gate.e_score_correction_bias"))
            inter = f.gate_up.shape[1] // 2
            hidden = f.gate_up.shape[2]
            leg = lambda e, half: src(n("mlp.experts.{}.{}.weight".format(e, half))).transmute(
                [1, inter, hidden], encoding(f.gate_up.dtype))
            reads.read_expr(f.gate_up, concat(0, [
                concat(1, [leg(e, "gate_proj"), leg(e, "up_proj")]) for e in range(f.experts)
            ]))
            reads.read_expr(f.down, concat(0, [
                src(n("mlp.experts.{}.down_proj.weight".format(e))).transmute(
                    [1, f.down.shape[1], f.down.shape[2]], encoding(f.down.dtype))
                for e in range(f.experts)
            ]))
        else:
            fail("a flash SKU cannot read the deepseek-v3 huggingface layout; its artifact is the mlx one (`model.hc_head.base`)")

def gguf(m, reads):
    if m.mtp != None:
        fail("this SKU declares a draft head and no gguf spelling of one is settled")
    reads.read(m.embed, "token_embd.weight")
    reads.read(m.final_norm, "output_norm.weight")
    for l, w in enumerate(m.layers):
        n = lambda s: "blk.{}.{}".format(l, s)
        at = w.attn
        reads.read(w.attn_mix.scale, n("hc_attn_scale.weight"))
        reads.read(w.attn_mix.base, n("hc_attn_base.weight"))
        reads.read(w.mlp_mix.scale, n("hc_mlp_scale.weight"))
        reads.read(w.mlp_mix.base, n("hc_mlp_base.weight"))
        reads.read(at.q_down, n("attn_q_a.weight"))
        reads.read(at.q_norm, n("attn_q_a_norm.weight"))
        reads.read(at.q_up, n("attn_q_b.weight"))
        reads.read(at.kv_down, n("attn_kv_a_mqa.weight"))
        reads.read(at.kv_norm, n("attn_kv_a_norm.weight"))
        reads.read(at.o_down, n("attn_o_a.weight"))
        reads.read(at.o_up, n("attn_o_b.weight"))
        reads.read(at.sink, n("attn_sinks"))
        f = w.mlp
        if f.kind == "dense":
            reads.read_concat(f.gate_up, [n("ffn_gate.weight"), n("ffn_up.weight")])
            reads.read(f.down, n("ffn_down.weight"))
        elif f.kind == "routed":
            reads.read(f.router, n("ffn_gate_inp.weight"))
            reads.read(f.bias, n("exp_probs_b.bias"))
            reads.read_concat(f.gate_up, [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")])
            reads.read(f.down, n("ffn_down_exps.weight"))
        else:
            fail("a flash SKU has no gguf layout")
