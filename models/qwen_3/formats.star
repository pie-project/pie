# How a Qwen 3.5 / 3.6 / 3.8 checkpoint is laid out: transformers', mlx_lm's
# (its norms stored less one) and GGUF's spellings. Ternary-Bonsai's GGUF
# also states its rotation's sign diagonals in its metadata, which its
# contract carries as constants.

load("//lib/dflash/formats.star", "bind_aux")

# The Bonsai GGUF's rotation metadata.

def product(xs):
    out = 1
    for x in xs:
        out *= x
    return out

def flattened(name, want, broadcast = False):
    """The tensor `name` as stored, its elements read as `want`; with
    `broadcast`, a single stored value is read as any shape."""
    held = shape(name)
    if (not broadcast or product(held) > 1) and product(held) != product(want):
        fail("`{}` is stored {} ({} elements) and the plan reads it as {} ({} elements)".format(
            name, held, product(held), want, product(want)))
    return src(name).transmute(want, stored(name))

def squeezed(name):
    """A depthwise convolution bank stored `[channels, 1, kernel]` or
    `[channels, kernel, 1]`, read as `[channels, kernel]`."""
    held = shape(name)
    if len(held) == 3 and held[1] == 1:
        channels, kernel = held[0], held[2]
    elif len(held) == 3 and held[2] == 1:
        channels, kernel = held[0], held[1]
    else:
        fail("`{}`: a depthwise convolution bank is stored [channels, 1, kernel] or [channels, kernel, 1] and this one is stored {}".format(name, held))
    return src(name).transmute([channels, kernel], stored(name))

LAYOUTS = [
    struct(
        name = "transformers",
        trunk = "model.language_model.",
        head = "lm_head.weight",
        tower = "model.visual.",
        mlx = False,
    ),
    struct(
        name = "mlx_lm",
        trunk = "language_model.model.",
        head = "language_model.lm_head.weight",
        tower = "vision_tower.",
        mlx = True,
    ),
]

def safetensors_format(layout, read, states = []):
    """`layout`'s format, recognized by its embedding: `read(reads, layout)`."""
    return format(
        layout.name,
        recognizes = lambda checkpoint: has(layout.trunk + "embed_tokens.weight"),
        read = lambda reads: read(reads, layout),
        states = states,
    )

def norm_of(layout):
    """`norm(name)`: a norm as the layout stores it, read as the forward's."""
    if layout.mlx:
        return lambda name: src(name).bias(-1.0)
    return lambda name: src(name)

def read_attn(reads, a, n, norm):
    reads.read(a.qg_proj, n("self_attn.q_proj.weight"))
    reads.read(a.k_proj, n("self_attn.k_proj.weight"))
    reads.read(a.v_proj, n("self_attn.v_proj.weight"))
    reads.read(a.o_proj, n("self_attn.o_proj.weight"))
    reads.read_expr(a.q_norm, norm(n("self_attn.q_norm.weight")))
    reads.read_expr(a.k_norm, norm(n("self_attn.k_norm.weight")))

def read_gdn(reads, g, n):
    reads.read_concat(g.in_qkvz, [n("linear_attn.in_proj_qkv.weight"), n("linear_attn.in_proj_z.weight")])
    reads.read_concat(g.in_ba, [n("linear_attn.in_proj_b.weight"), n("linear_attn.in_proj_a.weight")])
    reads.read_expr(g.conv, squeezed(n("linear_attn.conv1d.weight")))
    reads.read(g.dt_bias, n("linear_attn.dt_bias"))
    reads.read(g.a_log, n("linear_attn.A_log"))
    reads.read(g.norm, n("linear_attn.norm.weight"))
    reads.read(g.out_proj, n("linear_attn.out_proj.weight"))

def read_routed(reads, f, n, layout):
    reads.read(f.router, n("mlp.gate.weight"))
    if layout.mlx:
        reads.read_concat(f.gate_up, [n("mlp.switch_mlp.gate_proj.weight"), n("mlp.switch_mlp.up_proj.weight")])
        reads.read(f.down, n("mlp.switch_mlp.down_proj.weight"))
    else:
        reads.read(f.gate_up, n("mlp.experts.gate_up_proj"))
        reads.read(f.down, n("mlp.experts.down_proj"))
    reads.read_concat(f.shared_gate_up, [n("mlp.shared_expert.gate_proj.weight"), n("mlp.shared_expert.up_proj.weight")])
    reads.read(f.shared_down, n("mlp.shared_expert.down_proj.weight"))
    reads.read(f.shared_gate, n("mlp.shared_expert_gate.weight"))

CHANNELS = 3

def read_tower(reads, t, prefix, channels_last):
    v = lambda s: prefix + s
    flat = flattened(v("patch_embed.proj.weight"), t.patch_embed.shape, broadcast = True)
    if channels_last:
        per = t.patch_embed.shape[1] // CHANNELS
        flat = flat.gather(1, [j * CHANNELS + c for c in range(CHANNELS) for j in range(per)])
    reads.read_expr(t.patch_embed, flat)
    reads.read(t.patch_embed_bias, v("patch_embed.proj.bias"))
    reads.read(t.pos_embed, v("pos_embed.weight"))
    for l, blk in enumerate(t.blocks):
        n = lambda s: v("blocks.{}.{}".format(l, s))
        for w, name in [
            (blk.norm1, "norm1.weight"),
            (blk.norm1_bias, "norm1.bias"),
            (blk.qkv, "attn.qkv.weight"),
            (blk.qkv_bias, "attn.qkv.bias"),
            (blk.proj, "attn.proj.weight"),
            (blk.proj_bias, "attn.proj.bias"),
            (blk.norm2, "norm2.weight"),
            (blk.norm2_bias, "norm2.bias"),
            (blk.fc1, "mlp.linear_fc1.weight"),
            (blk.fc1_bias, "mlp.linear_fc1.bias"),
            (blk.fc2, "mlp.linear_fc2.weight"),
            (blk.fc2_bias, "mlp.linear_fc2.bias"),
        ]:
            reads.read(w, n(name))
    mg = t.merger
    for w, name in [
        (mg.norm, "merger.norm.weight"),
        (mg.norm_bias, "merger.norm.bias"),
        (mg.fc1, "merger.linear_fc1.weight"),
        (mg.fc1_bias, "merger.linear_fc1.bias"),
        (mg.fc2, "merger.linear_fc2.weight"),
        (mg.fc2_bias, "merger.linear_fc2.bias"),
    ]:
        reads.read(w, v(name))

BLOCK_SIZE_KEY = "prism.hadamard.block_size"
SIGN_MODE_KEY = "prism.hadamard.sign_mode"
SIGN_WIDTHS_KEY = "prism.hadamard.sign_widths"
SIGN_VALUES_KEY = "prism.hadamard.sign_values"
GDN_V_GROUPED_KEY = "prism.hadamard.gdn_v_grouped"

def formats(m):
    if m.bonsai != None:
        return [format(
            "gguf",
            recognizes = lambda checkpoint: attribute_value(SIGN_VALUES_KEY) != None,
            read = lambda reads: gguf(m, reads),
        )]
    arch = text_attribute("general.architecture") or ""
    states = configured(m, "text_config.")
    return [
        safetensors_format(layout, lambda reads, layout: safetensors(m, reads, layout), states)
        for layout in LAYOUTS
    ] + [
        format(
            "gguf",
            recognizes = lambda checkpoint: text_attribute("general.architecture") != None,
            read = lambda reads: gguf(m, reads),
            states = attributed(m, arch),
        ),
    ]

def theta(m):
    for w in m.layers:
        if w.attn != None:
            return w.attn.theta
    return None

def configured(m, at):
    """The shape a transformers configuration states of this model, its text
    model's keys under `at`."""
    states = [
        config(at + "hidden_size", m.hidden),
        config(at + "vocab_size", m.vocab),
        config(at + "num_attention_heads", m.q_heads),
        config(at + "num_key_value_heads", m.kv_heads),
        config(at + "head_dim", m.head_dim),
        config(at + "num_hidden_layers", len(m.layers), or_deeper = True),
        config(at + "rms_norm_eps", m.final_norm_eps),
    ]
    if theta(m) != None:
        states.append(config(at + "rope_parameters.rope_theta", theta(m)))
    return states

def attributed(m, arch):
    """The shape a GGUF's metadata states of this model, under `arch`."""
    states = [
        attribute(arch + ".embedding_length", m.hidden),
        attribute(arch + ".block_count", len(m.layers), or_deeper = True),
        attribute(arch + ".attention.head_count", m.q_heads),
        attribute(arch + ".attention.head_count_kv", m.kv_heads),
    ]
    if theta(m) != None:
        states.append(attribute(arch + ".rope.freq_base", theta(m)))
    return states

def spelled(names):
    """The first of `names` the checkpoint holds, or `None`."""
    for name in names:
        if has(name):
            return name
    return None

def held(names):
    """The first of `names` the checkpoint holds, which it must."""
    name = spelled(names)
    if name == None:
        # The checkpoint's refusal names every spelling.
        shape("` or `".join(names))
    return name

def safetensors(m, reads, layout):
    norm = norm_of(layout)
    reads.read(m.embed, layout.trunk + "embed_tokens.weight")
    reads.read_expr(m.final_norm, norm(layout.trunk + "norm.weight"))
    if m.head != None:
        reads.read(m.head, layout.head)

    for l, w in enumerate(m.layers):
        n = lambda s: "{}layers.{}.{}".format(layout.trunk, l, s)
        reads.read_expr(w.mixer_norm, norm(n("input_layernorm.weight")))
        reads.read_expr(w.mlp_norm, norm(n("post_attention_layernorm.weight")))
        if w.attn != None:
            read_attn(reads, w.attn, n, norm)
        else:
            read_gdn(reads, w.gdn, n)
        if w.mlp.routed:
            read_routed(reads, w.mlp, n, layout)
        else:
            reads.read_concat(w.mlp.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
            reads.read(w.mlp.down, n("mlp.down_proj.weight"))

    if m.tower != None:
        read_tower(reads, m.tower, layout.tower, layout.mlx)

    if m.mtp != None:
        mtp = m.mtp
        p = "aux" if has("aux.fc.weight") or has("aux.fc_embed.weight") else mtp.prefix
        n = lambda s: "{}.layers.0.{}".format(p, s)
        if mtp.pre_fc != None:
            reads.read(mtp.pre_fc.embedding, p + ".pre_fc_norm_embedding.weight")
            reads.read(mtp.pre_fc.hidden, p + ".pre_fc_norm_hidden.weight")
        if has(p + ".fc_embed.weight"):
            reads.read(mtp.fc_embed, p + ".fc_embed.weight")
            reads.read(mtp.fc_hidden, p + ".fc_hidden.weight")
        else:
            half = mtp.fc_embed.shape[1]
            reads.read_expr(mtp.fc_embed, src(p + ".fc.weight").slice(1, 0, half))
            reads.read_expr(mtp.fc_hidden, src(p + ".fc.weight").slice(1, half, half))
        a = mtp.attn
        reads.read(mtp.mixer_norm, n("input_layernorm.weight"))
        reads.read(a.qg_proj, n("self_attn.q_proj.weight"))
        reads.read(a.k_proj, n("self_attn.k_proj.weight"))
        reads.read(a.v_proj, n("self_attn.v_proj.weight"))
        reads.read(a.o_proj, n("self_attn.o_proj.weight"))
        reads.read(a.q_norm, n("self_attn.q_norm.weight"))
        reads.read(a.k_norm, n("self_attn.k_norm.weight"))
        reads.read(mtp.mlp_norm, n("post_attention_layernorm.weight"))
        reads.read_concat(mtp.mlp.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
        reads.read(mtp.mlp.down, n("mlp.down_proj.weight"))
        if mtp.norm != None:
            reads.read(mtp.norm, p + ".norm.weight")

    if m.dflash != None:
        bind_aux(m.dflash, reads, norm)

def v_head_reorder_rows(prefix, heads, width, k_heads, rep):
    """The rows that reorder a GDN tensor's v-heads from a GGUF's tiled
    ("v-grouped") layout into the block layout the scan pairs by: a v-head at
    block position `p` lives at tiled index `(p % rep) * k_heads + p // rep`;
    `prefix` leading rows (the fused q and k) pass through."""
    idx = list(range(prefix))
    for p in range(heads):
        base = prefix + ((p % rep) * k_heads + p // rep) * width
        idx.extend(range(base, base + width))
    return idx

def gguf(m, reads):
    if m.tower != None:
        fail("this deployment declares a vision tower and no GGUF spelling of one is settled; import it from the safetensors checkpoint")
    if m.mtp != None:
        fail("this deployment declares an MTP draft head and no GGUF spelling of one is settled; import it from the safetensors checkpoint")
    minus_one = lambda e: e.bias(-1.0)
    # Ternary-Bonsai's GGUF stores the GDN v-heads tiled; the reads reorder
    # every v-head-indexed tensor to the block order the scan pairs by.
    v_grouped = attribute_value(GDN_V_GROUPED_KEY) == True

    reads.read(m.embed, "token_embd.weight")
    reads.read_over(m.final_norm, "output_norm.weight", minus_one)
    if m.head != None:
        reads.read(m.head, held(["output.weight", "token_embd.weight"]))

    for l, w in enumerate(m.layers):
        n = lambda s: "blk.{}.{}".format(l, s)
        reads.read_over(w.mixer_norm, n("attn_norm.weight"), minus_one)
        reads.read_over(w.mlp_norm, held([n("ffn_norm.weight"), n("post_attention_norm.weight")]), minus_one)
        if w.attn != None:
            a = w.attn
            reads.read(a.qg_proj, n("attn_q.weight"))
            reads.read(a.k_proj, n("attn_k.weight"))
            reads.read(a.v_proj, n("attn_v.weight"))
            reads.read(a.o_proj, n("attn_output.weight"))
            reads.read_over(a.q_norm, n("attn_q_norm.weight"), minus_one)
            reads.read_over(a.k_norm, n("attn_k_norm.weight"), minus_one)
        else:
            g = w.gdn
            k_w = 2 * g.k_heads * g.k_dim
            heads = g.v_heads
            width = g.v_dim
            rep = heads // g.k_heads
            sigma = lambda prefix, width: v_head_reorder_rows(prefix, heads, width, g.k_heads, rep)

            fused = spelled([n("ssm_in.weight")])
            if fused != None and v_grouped:
                z_base = k_w + heads * width
                reads.read_expr(g.in_qkvz, src(fused).gather(0, sigma(k_w, width) + [z_base + i for i in sigma(0, width)]))
            elif fused != None:
                reads.read(g.in_qkvz, fused)
            elif v_grouped:
                qkv = src(n("attn_qkv.weight")).gather(0, sigma(k_w, width))
                gate = src(n("attn_gate.weight")).gather(0, sigma(0, width))
                reads.read_expr(g.in_qkvz, concat(0, [qkv, gate]))
            else:
                reads.read_concat(g.in_qkvz, [n("attn_qkv.weight"), n("attn_gate.weight")])

            # `in_ba` is `[beta, alpha]`, beta first.
            legs = [n("ssm_beta.weight"), n("ssm_alpha.weight")]
            fused = spelled([n("ssm_beta_alpha.weight")])
            if fused != None and v_grouped:
                idx = sigma(0, 1)
                reads.read_expr(g.in_ba, src(fused).gather(0, idx + [len(idx) + i for i in sigma(0, 1)]))
            elif fused != None:
                reads.read(g.in_ba, fused)
            elif v_grouped:
                reads.read_expr(g.in_ba, concat(0, [src(leg).gather(0, sigma(0, 1)) for leg in legs]))
            else:
                reads.read_concat(g.in_ba, legs)
            if v_grouped:
                reads.read_expr(g.conv, src(n("ssm_conv1d.weight")).gather(0, sigma(k_w, width)))
                reads.read_expr(g.dt_bias, src(n("ssm_dt.bias")).gather(0, sigma(0, 1)))
            else:
                reads.read(g.conv, n("ssm_conv1d.weight"))
                reads.read(g.dt_bias, n("ssm_dt.bias"))
            logarithm = spelled([n("ssm_a_log")])
            if logarithm != None and v_grouped:
                reads.read_expr(g.a_log, src(logarithm).gather(0, sigma(0, 1)))
            elif logarithm != None:
                reads.read(g.a_log, logarithm)
            elif has(n("ssm_a")):
                a = src(n("ssm_a"))
                if v_grouped:
                    a = a.gather(0, sigma(0, 1))
                reads.read_expr(g.a_log, a.unary("neg_ln"))
            else:
                held([n("ssm_a_log")])
            reads.read(g.norm, n("ssm_norm.weight"))
            reads.read(g.out_proj, n("ssm_out.weight"))

        f = w.mlp
        if not f.routed:
            reads.read_concat(f.gate_up, [n("ffn_gate.weight"), n("ffn_up.weight")])
            reads.read(f.down, n("ffn_down.weight"))
        else:
            reads.read(f.router, n("ffn_gate_inp.weight"))
            reads.read_concat(f.gate_up, [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")])
            reads.read(f.down, n("ffn_down_exps.weight"))
            reads.read_concat(f.shared_gate_up, [n("ffn_gate_shexp.weight"), n("ffn_up_shexp.weight")])
            reads.read(f.shared_down, n("ffn_down_shexp.weight"))
            reads.read(f.shared_gate, n("ffn_gate_inp_shexp.weight"))

    if m.bonsai != None:
        signs = decode_signs()
        for sign in [m.bonsai.hidden, m.bonsai.ssm, m.bonsai.ffn_down]:
            width = sign.shape[1]
            if width not in signs:
                fail("the Bonsai GGUF states no sign diagonal {} wide".format(width))
            reads.read_expr(sign, constant(sign.name, signs[width], sign.shape, sign.dtype))

def decode_signs():
    """The sign diagonals a Ternary-Bonsai GGUF's metadata states, keyed by
    width: `sign_values` cut into runs of `sign_widths`, each a whole number
    of 1024-wide Hadamard blocks, every value exactly +1 or -1."""
    mode = attribute_value(SIGN_MODE_KEY)
    if mode == "identity":
        return {}
    if mode not in [None, "explicit"]:
        fail("{} `{}` is not identity|explicit".format(SIGN_MODE_KEY, mode))
    block = attribute_value(BLOCK_SIZE_KEY)
    if block != 1024:
        fail("{} is {}, and the Bonsai rotation is 1024 wide".format(BLOCK_SIZE_KEY, block))
    widths = attribute_value(SIGN_WIDTHS_KEY)
    values = attribute_value(SIGN_VALUES_KEY)
    if type(widths) != "list" or type(values) != "list":
        fail("the Bonsai GGUF states no {} and {}".format(SIGN_WIDTHS_KEY, SIGN_VALUES_KEY))
    out = {}
    at = 0
    for width in widths:
        if width <= 0 or width % block != 0 or at + width > len(values):
            fail("prism.hadamard: invalid sign width {}".format(width))
        run = values[at:at + width]
        for k, v in enumerate(run):
            if v != 1 and v != -1:
                fail("prism.hadamard: sign {} of width {} is {}, not +/-1".format(k, width, v))
        out[width] = [float(v) for v in run]
        at += width
    if at != len(values):
        fail("prism.hadamard.sign_values length mismatch: widths consume {} of {}".format(at, len(values)))
    return out
