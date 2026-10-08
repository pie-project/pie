# Reads a format makes of a tensor stored in another shape than the one a
# weight declares.

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
