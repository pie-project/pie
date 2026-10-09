# The forward of the Qwen 3.5-family vision tower, and the multimodal rotary
# positions a trunk serving it rotates its attention by.

MROPE_SECTIONS = [11, 11, 10]

def positions(inputs, vision):
    """The positions a trunk's attention rotates by: multimodal with a tower."""
    return inputs.mrope_positions() if vision else inputs.positions()

def rope(q, k, positions, vision, rotary_dim, head_dim, theta):
    if not vision:
        return ops.elemwise.rope_partial(q, k, positions, rotary_dim, head_dim, theta)
    return ops.elemwise.rope_mrope(q, k, positions, MROPE_SECTIONS, "interleaved", rotary_dim, head_dim, theta)

def tower(inputs, t):
    """The tower's merged rows for the request's patches."""
    d = t.head_dim
    x = inputs.patches(t.patch_width)
    segments = inputs.patch_segments()
    grid = inputs.patch_positions()

    y = ops.elemwise.add_bias(t.patch_embed_bias, ops.linear.matmul(x, t.patch_embed))
    ids = inputs.patch_embed_rows(t.taps)
    if t.taps == 1:
        pos = ops.layout.embed(ids, t.pos_embed, t.positions)
    else:
        pos = ops.layout.embed_weighted(ids, inputs.patch_embed_weights(t.taps), t.pos_embed, t.positions)
    y = ops.elemwise.residual_add(pos, y)

    for b in t.blocks:
        n = ops.elemwise.layernorm(y, b.norm1, b.norm1_bias, t.norm_eps)
        q, k, v = ops.layout.split_qkv(ops.elemwise.add_bias(b.qkv_bias, ops.linear.matmul(n, b.qkv)), t.hidden, t.hidden)
        q, k = ops.elemwise.rope_mrope(q, k, grid, [0, d // 4, d // 4], "blocked", d, d, t.theta)
        o = ops.attn.dense(q, k, v, segments, d, t.sm_scale)
        y = ops.elemwise.residual_add(ops.elemwise.add_bias(b.proj_bias, ops.linear.matmul(o, b.proj)), y)

        n = ops.elemwise.layernorm(y, b.norm2, b.norm2_bias, t.norm_eps)
        h = ops.elemwise.add_bias(b.fc1_bias, ops.linear.matmul(n, b.fc1))
        a = ops.linear.mlp_gelu_tanh(h)
        y = ops.elemwise.residual_add(ops.elemwise.add_bias(b.fc2_bias, ops.linear.matmul(a, b.fc2)), y)

    mg = t.merger
    n = ops.elemwise.layernorm(y, mg.norm, mg.norm_bias, t.norm_eps)
    folded = ops.layout.merge_rows(n, t.merge)
    h = ops.elemwise.add_bias(mg.fc1_bias, ops.linear.matmul(folded, mg.fc1))
    a = ops.linear.mlp_gelu_tanh(h)
    return ops.elemwise.add_bias(mg.fc2_bias, ops.linear.matmul(a, mg.fc2))

def scatter(y, towered, inputs):
    """`y` with the tower's rows in place of its media tokens' embeddings."""
    imaged = y.on(fact.has(fact.Media))
    return ops.layout.scatter_live_rows(towered, inputs.patch_routes(), imaged).everywhere()
