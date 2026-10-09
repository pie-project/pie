# Reads the diffusion packages make alike.

def biased(reads, w, stem):
    """The projection `w` from `stem.weight`, and its bias (if any) from
    `stem.bias`."""
    reads.read(w.w, stem + ".weight")
    if w.bias != None:
        reads.read(w.bias, stem + ".bias")

def packed(reads, w, stems):
    """A packed biased projection, its pieces from each of `stems`."""
    reads.read_concat(w.w, [s + ".weight" for s in stems])
    reads.read_concat(w.bias, [s + ".bias" for s in stems])

def conv(reads, c, stem):
    """The convolution `c` from `stem`, its kernel transmuted to the taps-major
    plane the layout holds."""
    name = stem + ".weight"
    reads.read_expr(c.w, src(name).transmute(c.w.shape, stored(name)))
    reads.read(c.bias, stem + ".bias")

def adaln_order(slices):
    """The order a layout takes an adaLN projection's `slices` slices in
    from a checkpoint's: each (shift, scale) pair as (scale, shift), each
    gate where it stands."""
    return {
        2: [1, 0],
        3: [1, 0, 2],
        6: [1, 0, 2, 4, 3, 5],
        9: [1, 0, 2, 4, 3, 5, 7, 6, 8],
    }[slices]

def reordered(e, order, width, axis = 0):
    """`e`'s `width`-wide slices along `axis`, taken in `order`."""
    return concat(axis, [e.slice(axis, i * width, width) for i in order])
