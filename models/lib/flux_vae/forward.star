# The Flux VAE's decoder and encoder.

load("//lib/diffusion/forward.star", "linear")

GN_GROUPS = 32
GN_EPS = 1e-6
RGB = 3

def convolve(x, grid, c):
    shape = conv([1, 1], [1, 1], [0, 0]) if c.taps == 1 else conv([3, 3], [1, 1], [1, 1])
    return ops.spatial.conv3d(x, grid, c.w, c.bias, shape, None)[0]

def group_norm(x, grid, n, silu):
    return ops.spatial.group_norm(x, grid, GN_GROUPS, n.weight, n.bias, GN_EPS, silu)

def resnet(x, grid, r):
    h = group_norm(x, grid, r.norm1, True)
    h = convolve(h, grid, r.conv1)
    h = group_norm(h, grid, r.norm2, True)
    h = convolve(h, grid, r.conv2)
    skip = convolve(x, grid, r.shortcut) if r.shortcut != None else x
    return ops.elemwise.add(skip, h)

def attention(x, grid, a):
    h = group_norm(x, grid, a.norm, False)
    q = linear(a.q, h)
    k = linear(a.k, h)
    v = linear(a.v, h)
    o = ops.spatial.attention(q, k, v, grid, f32(1.0 / f32(sqrt(a.width))))
    return ops.elemwise.add(x, linear(a.out, o))

def mid(x, grid, m):
    h = resnet(x, grid, m.res0)
    h = attention(h, grid, m.attn)
    return resnet(h, grid, m.res1)

def decode(vae, z, grid):
    """The pixels the decoder makes of the latent `z`, read out."""
    d = vae.decoder
    h = convolve(z, grid, d.conv_in)
    h = mid(h, grid, d.mid)
    for block in d.up:
        for res in block.resnets:
            h = resnet(h, grid, res)
        if block.upsample != None:
            up, grid = ops.spatial.upsample_nearest(h, grid, [1, 2, 2], False)
            h = convolve(up, grid, block.upsample)
    h = group_norm(h, grid, d.norm_out, True)
    y = convolve(h, grid, d.conv_out)
    seam.at(seam.PIXELS, [y, grid])
    return y

def encode(vae, arm):
    """The encoder's output over `arm`'s pixels, and its grid."""
    e = vae.encoder
    grid = arm.grid()
    x = arm.voxels(1, RGB, dtype.bf16)
    h = convolve(x, grid, e.conv_in)
    for block in e.down:
        for res in block.resnets:
            h = resnet(h, grid, res)
        if block.downsample != None:
            c = block.downsample
            h, grid = ops.spatial.conv3d(h, grid, c.w, c.bias, conv([3, 3], [2, 2], [0, 0], pad_back = [0, 1, 1]), None)
    h = mid(h, grid, e.mid)
    h = group_norm(h, grid, e.norm_out, True)
    return convolve(h, grid, e.conv_out), grid
