# The weights of a KDA (Kimi delta attention) mixer: a packed q/k/v
# projection through a short causal convolution into a gated delta-rule
# recurrence, its output normed under a gate and projected back.
#
# `k` states its shape: `heads`, `head_dim`, `f_rank` (the decay
# projection's rank) and `conv_kernel`.

def mixer(n, l, k, hidden, weights, conv, eps, gate_floor, gate_rank = None, **extra):
    """The KDA mixer of layer `l` named by `n`, its projections in `weights`
    and its convolution bank in `conv`. Its output gate is one projection,
    or a pair through `gate_rank`. `extra` rides along."""
    width = k.heads * k.head_dim
    if gate_rank == None:
        gate = [weight(n("kda_gate"), [width, hidden], weights).columns()]
    else:
        gate = [
            weight(n("kda_g_a"), [gate_rank, hidden], weights),
            weight(n("kda_g_b"), [width, gate_rank], weights).columns(),
        ]
    return struct(
        heads = k.heads,
        head_dim = k.head_dim,
        conv_kernel = k.conv_kernel,
        norm_eps = eps,
        gate_floor = gate_floor,
        qkv = weight(n("kda_qkv"), [3 * width, hidden], weights).packed([width] * 3),
        conv = weight(n("kda_conv"), [3 * width, k.conv_kernel], conv).packed([width] * 3),
        f_a = weight(n("kda_f_a"), [k.f_rank, hidden], weights),
        f_b = weight(n("kda_f_b"), [width, k.f_rank], weights).columns(),
        gate = gate,
        b = weight(n("kda_b"), [k.heads, hidden], weights).columns(),
        dt_bias = weight(n("kda_dt_bias"), [k.heads, k.head_dim], dtype.f32).columns(),
        a_log = weight(n("kda_a_log"), [k.heads], dtype.f32).columns(),
        o_norm = weight(n("kda_o_norm"), [k.head_dim], dtype.f32),
        o_norm_eps = eps,
        o_proj = weight(n("kda_o_proj"), [hidden, width], weights).rows(),
        conv_state = "conv.{}".format(l),
        delta_state = "delta.{}".format(l),
        **extra
    )
