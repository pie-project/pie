# Weights the diffusion packages lay out alike.

def linear(name, out, in_, banks, bias = True):
    """A projection from `in_` to `out` and, unless `bias` is off, its bias."""
    return struct(
        w = weight(name, [out, in_], banks),
        bias = weight(name + ".bias", [out], compute(banks)) if bias else None,
    )

def packed_linear(name, seams, in_, banks):
    """A biased projection whose output is the `seams`-wide pieces end to end."""
    out = 0
    for s in seams:
        out += s
    return struct(
        w = weight(name, [out, in_], banks).packed(seams),
        bias = weight(name + ".bias", [out], compute(banks)).packed(seams),
    )

def embedder(prefix, in_, dim, banks):
    """A two-layer timestep (or caption) embedder from `in_` to `dim`."""
    return struct(
        linear_1 = linear(prefix + ".1", dim, in_, banks),
        linear_2 = linear(prefix + ".2", dim, dim, banks),
    )
