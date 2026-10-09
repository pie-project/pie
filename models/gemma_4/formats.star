# How a Gemma 4 checkpoint is laid out: transformers', mlx_lm's and GGUF's
# spellings, and DiffusionGemma's own.

load("//lib/dflash/formats.star", "bind_aux")
load("//lib/reads/formats.star", "flattened")

LAYOUTS = {
    "transformers": struct(
        trunk = "model.language_model.",
        vision = "model.vision_tower.",
        embed_vision = "model.embed_vision.embedding_projection.weight",
    ),
    "mlx_lm": struct(
        trunk = "language_model.model.",
        vision = "vision_tower.",
        embed_vision = "embed_vision.embedding_projection.weight",
    ),
    "diffusion": struct(
        trunk = "model.decoder.",
        vision = "model.encoder.vision_tower.",
        embed_vision = "model.encoder.embed_vision.embedding_projection.weight",
    ),
}

def formats(m):
    if m.self_cond != None:
        return [format("diffusion", read = lambda reads: safetensors(m, reads, LAYOUTS["diffusion"]))]
    arch = text_attribute("general.architecture") or ""
    return [
        safetensors_format(m, "transformers"),
        safetensors_format(m, "mlx_lm"),
        format(
            "gguf",
            recognizes = lambda checkpoint: text_attribute("general.architecture") != None,
            read = lambda reads: gguf(m, reads),
            states = attributed(m, arch),
        ),
    ]

def safetensors_format(m, name):
    layout = LAYOUTS[name]
    return format(
        name,
        recognizes = lambda checkpoint: has(layout.trunk + "embed_tokens.weight"),
        read = lambda reads: safetensors(m, reads, layout),
        states = configured(m, "text_config."),
    )

def configured(m, at):
    """The shape a transformers configuration states of this model, its text
    model's keys under `at`."""
    return [
        config(at + "hidden_size", m.hidden),
        config(at + "vocab_size", m.vocab),
        config(at + "num_attention_heads", m.q_heads),
        config(at + "num_key_value_heads", m.sliding.kv_heads),
        config(at + "head_dim", m.sliding.head_dim),
        config(at + "global_head_dim", m.glob.head_dim),
        config(at + "num_hidden_layers", len(m.layers), or_deeper = True),
        config(at + "rms_norm_eps", m.final_norm_eps),
        config(at + "sliding_window", m.sliding.window),
        config(at + "rope_parameters.sliding_attention.rope_theta", m.sliding.theta),
        config(at + "rope_parameters.full_attention.rope_theta", m.glob.theta),
    ]

def attributed(m, arch):
    """The shape a GGUF's metadata states of this model, under `arch`."""
    return [
        attribute(arch + ".embedding_length", m.hidden),
        attribute(arch + ".block_count", len(m.layers), or_deeper = True),
        attribute(arch + ".attention.head_count", m.q_heads),
        attribute(arch + ".attention.sliding_window", m.sliding.window),
    ]

def safetensors(m, reads, layout):
    at = lambda leaf: layout.trunk + leaf
    reads.read(m.embed, at("embed_tokens.weight"))
    reads.read(m.final_norm, at("norm.weight"))

    for l, w in enumerate(m.layers):
        n = lambda leaf: "{}layers.{}.{}".format(layout.trunk, l, leaf)
        reads.read(w.attn_norm, n("input_layernorm.weight"))
        reads.read(w.post_attn_norm, n("post_attention_layernorm.weight"))
        reads.read(w.pre_ffw_norm, n("pre_feedforward_layernorm.weight"))
        reads.read(w.post_ffw_norm, n("post_feedforward_layernorm.weight"))
        reads.read(w.attn.q_norm, n("self_attn.q_norm.weight"))
        banks = w.attn.banks
        if not banks.shared:
            reads.read(banks.k_norm, n("self_attn.k_norm.weight"))
            k = n("self_attn.k_proj.weight")
            v = n("self_attn.v_proj.weight")
            reads.read_concat(banks.qkv, [n("self_attn.q_proj.weight"), k, v if has(v) else k])
        else:
            reads.read(banks.q_proj, n("self_attn.q_proj.weight"))
        reads.read(w.o_proj, n("self_attn.o_proj.weight"))
        reads.read_concat(w.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
        reads.read(w.down, n("mlp.down_proj.weight"))

        if w.moe != None:
            x = w.moe
            reads.read_expr(x.router_norm, src(n("router.scale")).scale(powf32(m.hidden, -0.5)))
            reads.read(x.router, n("router.proj.weight"))
            reads.read(x.per_expert_scale, n("router.per_expert_scale"))
            reads.read(x.pre_ffw_norm_2, n("pre_feedforward_layernorm_2.weight"))
            reads.read(x.post_ffw_norm_1, n("post_feedforward_layernorm_1.weight"))
            reads.read(x.post_ffw_norm_2, n("post_feedforward_layernorm_2.weight"))
            fused = n("experts.gate_up_proj")
            fused_quantized = n("experts.gate_up_proj.weight")
            if has(fused):
                reads.read(x.gate_up, fused)
                reads.read(x.down, n("experts.down_proj"))
            elif has(fused_quantized):
                reads.read(x.gate_up, fused_quantized)
                reads.read(x.down, n("experts.down_proj.weight"))
            else:
                reads.read_concat(x.gate_up, [
                    n("experts.switch_glu.gate_proj.weight"),
                    n("experts.switch_glu.up_proj.weight"),
                ])
                reads.read(x.down, n("experts.switch_glu.down_proj.weight"))

        if w.scalar != None:
            reads.read(w.scalar, n("layer_scalar"))

    if m.ple != None:
        ple = m.ple
        rows = ple.model_proj.shape[0]
        name = at("per_layer_model_projection.weight")
        if has(name) and shape(name)[0] != rows:
            reads.read_expr(ple.model_proj, src(name).slice(0, 0, rows))
        else:
            reads.read(ple.model_proj, name)
        reads.read(ple.model_norm, at("per_layer_projection_norm.weight"))
        for l, p in enumerate(ple.per_layer):
            n = lambda leaf: "{}layers.{}.{}".format(layout.trunk, l, leaf)
            reads.read_expr(p.table, src(at("embed_tokens_per_layer.weight")).slice(1, l * ple.dim, ple.dim))
            reads.read(p.gate, n("per_layer_input_gate.weight"))
            reads.read(p.proj, n("per_layer_projection.weight"))
            reads.read(p.norm, n("post_per_layer_input_norm.weight"))
            reads.read(p.scalar, n("layer_scalar"))

    if m.tower != None:
        t = m.tower
        v = lambda s: layout.vision + s
        reads.read(t.patch_embed, v("patch_embedder.input_proj.weight"))
        reads.read_expr(t.pos_embed, flattened(v("patch_embedder.position_embedding_table"), t.pos_embed.shape, broadcast = True))
        reads.read(t.projection, layout.embed_vision)
        if t.std != None:
            reads.read(t.std.bias, v("std_bias"))
            reads.read(t.std.scale, v("std_scale"))
        for l, blk in enumerate(t.blocks):
            n = lambda s: v("encoder.layers.{}.{}".format(l, s))
            for weight_, name in [
                (blk.attn_norm, n("input_layernorm.weight")),
                (blk.post_attn_norm, n("post_attention_layernorm.weight")),
                (blk.pre_ffw_norm, n("pre_feedforward_layernorm.weight")),
                (blk.post_ffw_norm, n("post_feedforward_layernorm.weight")),
                (blk.q_norm, n("self_attn.q_norm.weight")),
                (blk.k_norm, n("self_attn.k_norm.weight")),
            ]:
                reads.read(weight_, name)
            for c, stem in [
                (blk.q, n("self_attn.q_proj")),
                (blk.k, n("self_attn.k_proj")),
                (blk.v, n("self_attn.v_proj")),
                (blk.o, n("self_attn.o_proj")),
                (blk.gate, n("mlp.gate_proj")),
                (blk.up, n("mlp.up_proj")),
                (blk.down, n("mlp.down_proj")),
            ]:
                reads.read(c.bank, stem + ".linear.weight")
                if c.clip != None:
                    k = c.clip
                    for weight_, suffix in [
                        (k.in_lo, "input_min"),
                        (k.in_hi, "input_max"),
                        (k.out_lo, "output_min"),
                        (k.out_hi, "output_max"),
                    ]:
                        reads.read_expr(weight_, flattened(stem + "." + suffix, weight_.shape, broadcast = True))

    if m.self_cond != None:
        sc = m.self_cond
        reads.read(sc.pre_norm, at("self_conditioning.pre_norm.weight"))
        reads.read_concat(sc.gate_up, [
            at("self_conditioning.gate_proj.weight"),
            at("self_conditioning.up_proj.weight"),
        ])
        reads.read(sc.down, at("self_conditioning.down_proj.weight"))

    if m.draft != None:
        a = m.draft
        half = a.fc_embed.shape[1]
        reads.read_expr(a.fc_embed, src("aux.fc.weight").slice(1, 0, half))
        reads.read_expr(a.fc_hidden, src("aux.fc.weight").slice(1, half, half))
        n = lambda s: "aux.layers.0." + s
        for weight_, name in [
            (a.attn_norm, n("input_layernorm.weight")),
            (a.post_attn_norm, n("post_attention_layernorm.weight")),
            (a.pre_ffw_norm, n("pre_feedforward_layernorm.weight")),
            (a.post_ffw_norm, n("post_feedforward_layernorm.weight")),
            (a.attn.q_norm, n("self_attn.q_norm.weight")),
            (a.o_proj, n("self_attn.o_proj.weight")),
        ]:
            reads.read(weight_, name)
        reads.read_concat(a.attn.banks.qkv, [
            n("self_attn.q_proj.weight"),
            n("self_attn.k_proj.weight"),
            n("self_attn.v_proj.weight"),
        ])
        reads.read(a.attn.banks.k_norm, n("self_attn.k_norm.weight"))
        reads.read_concat(a.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
        reads.read(a.down, n("mlp.down_proj.weight"))

    if m.assistant != None:
        a = m.assistant
        if has("aux.pre_projection_embed.weight"):
            reads.read(a.pre_embed, "aux.pre_projection_embed.weight")
            reads.read(a.pre_hidden, "aux.pre_projection_hidden.weight")
        else:
            th = a.pre_embed.shape[1]
            reads.read_expr(a.pre_embed, src("aux.pre_projection.weight").slice(1, 0, th))
            reads.read_expr(a.pre_hidden, src("aux.pre_projection.weight").slice(1, th, th))
        reads.read(a.post, "aux.post_projection.weight")
        reads.read(a.embed, "aux.model.embed_tokens.weight")
        reads.read(a.norm, "aux.model.norm.weight")
        for l, w in enumerate(a.layers):
            n = lambda s: "aux.model.layers.{}.{}".format(l, s)
            for weight_, name in [
                (w.attn_norm, n("input_layernorm.weight")),
                (w.post_attn_norm, n("post_attention_layernorm.weight")),
                (w.pre_ffw_norm, n("pre_feedforward_layernorm.weight")),
                (w.post_ffw_norm, n("post_feedforward_layernorm.weight")),
                (w.attn.q_norm, n("self_attn.q_norm.weight")),
                (w.attn.banks.q_proj, n("self_attn.q_proj.weight")),
                (w.o_proj, n("self_attn.o_proj.weight")),
                (w.scalar, n("layer_scalar")),
                (w.down, n("mlp.down_proj.weight")),
            ]:
                reads.read(weight_, name)
            reads.read_concat(w.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])

    if m.dflash != None:
        bind_aux(m.dflash, reads, lambda name: src(name).bias(-1.0))

def gguf(m, reads):
    if m.self_cond != None:
        fail("this deployment is a block-diffusion text and no GGUF spelling of its self-conditioning block is settled; import it from the safetensors checkpoint")
    if m.draft != None or m.assistant != None:
        fail("this deployment declares an aux draft head and no GGUF spelling of one is settled; import it from the safetensors artifact")
    if m.tower != None:
        fail("this deployment declares a vision tower and no GGUF spelling of one is settled; import it from the safetensors checkpoint")
    for w in m.layers:
        if w.moe != None:
            fail("this deployment declares a routed feedforward branch and no GGUF spelling of gemma 4's `experts.switch_glu.*` or `router.*` is settled; import it from the safetensors checkpoint")
    reads.read(m.embed, "token_embd.weight")
    reads.read(m.final_norm, "output_norm.weight")
    for l, w in enumerate(m.layers):
        blk = lambda s: "blk.{}.{}".format(l, s)
        reads.read(w.attn_norm, blk("attn_norm.weight"))
        reads.read(w.post_attn_norm, blk("post_attention_norm.weight"))
        reads.read(w.pre_ffw_norm, blk("ffn_norm.weight"))
        reads.read(w.post_ffw_norm, blk("post_ffw_norm.weight"))
        reads.read(w.attn.q_norm, blk("attn_q_norm.weight"))
        banks = w.attn.banks
        if not banks.shared:
            reads.read(banks.k_norm, blk("attn_k_norm.weight"))
            reads.read_concat(banks.qkv, [blk("attn_q.weight"), blk("attn_k.weight"), blk("attn_v.weight")])
        else:
            reads.read(banks.q_proj, blk("attn_q.weight"))
        reads.read(w.o_proj, blk("attn_output.weight"))
        reads.read_concat(w.gate_up, [blk("ffn_gate.weight"), blk("ffn_up.weight")])
        reads.read(w.down, blk("ffn_down.weight"))
        if w.scalar != None:
            reads.read(w.scalar, blk("layer_scalar"))
    if m.ple != None:
        ple = m.ple
        reads.read(ple.model_proj, "per_layer_model_proj.weight")
        reads.read(ple.model_norm, "per_layer_proj_norm.weight")
        for l, p in enumerate(ple.per_layer):
            reads.read_expr(p.table, src("per_layer_token_embd.weight").slice(1, l * ple.dim, ple.dim))
            reads.read(p.gate, "blk.{}.inp_gate.weight".format(l))
            reads.read(p.proj, "blk.{}.proj.weight".format(l))
            reads.read(p.norm, "blk.{}.post_norm.weight".format(l))
            reads.read(p.scalar, "blk.{}.layer_scalar".format(l))
