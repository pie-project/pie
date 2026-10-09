# The forward of a KDA mixer: one recurrent step per decoded row, the
# chunked recurrence over a prefill's rows.

def caches(c, k):
    """`k`'s convolution window and its f32 delta-rule state."""
    width = k.heads * k.head_dim
    c.state(k.conv_state, [k.conv_kernel, 3 * width], dtype.bf16, split = 1)
    c.state(k.delta_state, [k.heads, k.head_dim, k.head_dim], dtype.f32, split = 0)

def mixer(x, inputs, k):
    conv = inputs.state(k.conv_state)
    delta = inputs.state(k.delta_state)
    qkv = ops.linear.matmul(x, k.qkv)
    f = ops.linear.matmul(ops.linear.matmul(x, k.f_a), k.f_b)
    b = ops.linear.matmul(x, k.b)
    seam.at(seam.RECURRENT, [qkv])

    one = fact.single_token()
    step = ops.attn.ssm_kda_step(
        ops.attn.ssm_causal_conv1d(qkv.on(one), k.conv, conv, k.conv_kernel),
        f.on(one),
        b.on(one),
        k.dt_bias,
        k.a_log,
        delta,
        k.heads,
        k.head_dim,
        k.norm_eps,
        k.gate_floor,
    )
    chunked = ops.attn.ssm_kda_chunked(
        ops.attn.ssm_causal_conv1d_chunked(qkv.on(~one), k.conv, conv, k.conv_kernel),
        f.on(~one),
        b.on(~one),
        k.dt_bias,
        k.a_log,
        delta,
        k.heads,
        k.head_dim,
        k.norm_eps,
        k.gate_floor,
    )
    core = merge([step, chunked])

    g = x
    for proj in k.gate:
        g = ops.linear.matmul(g, proj)
    o = ops.elemwise.rmsnorm_gated_by(core, g, k.o_norm, k.heads, k.o_norm_eps)
    return ops.linear.matmul(o, k.o_proj)
