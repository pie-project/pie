# The forward of manifold hyper-connections: each sublayer's input gated out
# of the streams and its output folded back, and the streams collapsed
# into one row at the end.

# How many experts the next layer's router is predicted to take.
PREDICT_K = 16

def mixes(normed, mix, hy):
    """`mix`'s per-row mixes of the `normed` streams: its dynamic
    projection, or with none the leading rows of the streams."""
    if mix.dynamic != None:
        return ops.elemwise.hc_project(normed, mix.dynamic, hy.streams)
    head, _ = ops.layout.split_rows(normed, mix.base.shape[0])
    return head

def gate(streams, mix, hy):
    """The sublayer input `mix` reads out of `streams`, and the post and
    combine mixes its output is folded back with."""
    normed = ops.elemwise.hc_rmsnorm_f32(streams, hy.norm_eps)
    return ops.elemwise.hc_gates(
        mixes(normed, mix, hy),
        streams,
        mix.scale,
        mix.base,
        hy.streams,
        hy.gate_eps,
        hy.alpha,
        hy.sinkhorn,
    )

def collapse(streams, head, hy):
    """The streams collapsed into one row through the mix `head`."""
    normed = ops.elemwise.hc_rmsnorm_f32(streams, hy.norm_eps)
    mixed = ops.elemwise.hc_project(normed, head.dynamic, hy.streams)
    return ops.elemwise.hc_collapse(mixed, streams, head.scale, head.base, hy.streams, hy.gate_eps)

def summed(streams, hidden, count):
    """The `count` streams, each `hidden` wide, added into one row."""
    y, rest = ops.layout.split_rows(streams, hidden)
    for _ in range(1, count - 1):
        stream, rest = ops.layout.split_rows(rest, hidden)
        y = ops.elemwise.residual_add(stream, y)
    return ops.elemwise.residual_add(rest, y)

def predict_route(streams, hy, mix, norm, norm_eps, router, bias, experts):
    """The experts a following MLP's router is predicted to take: the
    streams through its `mix` and `norm`, scored by its `router`."""
    x, _, _ = gate(streams, mix, hy)
    x = ops.elemwise.rmsnorm(x, norm, norm_eps)
    logits = ops.linear.matmul(x, router)
    return ops.linear.moe_predict_route(logits, bias, experts, PREDICT_K)
