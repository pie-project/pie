# The weights of manifold hyper-connections: the residual stream carried as
# `streams` parallel streams, each sublayer reading a mix of them and
# writing its output back through mixes of its own.

def hyper(streams, norm_eps, gate_eps, alpha, sinkhorn, single_pass = False):
    """The streams' shape and the constants their gates are computed with;
    under `single_pass` each sublayer's input mix is the one the previous
    sublayer predicted."""
    return struct(
        streams = streams,
        norm_eps = norm_eps,
        gate_eps = gate_eps,
        alpha = alpha,
        sinkhorn = sinkhorn,
        single_pass = single_pass,
    )

def mix(n, name, streams, hidden, dynamic = True):
    """A sublayer's mix `name` over `streams` streams `hidden` wide: a
    static `base` and `scale`, and with `dynamic` a projection of the
    normed streams added to it."""
    width = 2 * streams + streams * streams
    return struct(
        scale = weight(n(name + "_scale"), [3], dtype.f32),
        base = weight(n(name + "_base"), [width], dtype.f32),
        dynamic = weight(n(name + "_fn"), [width, streams * hidden], dtype.f32) if dynamic else None,
    )

def head(prefix, streams, hidden):
    """The mix `prefix` that collapses the streams into one row."""
    return struct(
        base = weight(prefix + ".base", [streams], dtype.f32),
        dynamic = weight(prefix + ".fn", [streams, streams * hidden], dtype.f32),
        scale = weight(prefix + ".scale", [1], dtype.f32),
    )
