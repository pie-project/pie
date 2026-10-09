# The Flux AutoencoderKL, under `vae.`: an encoder down and a decoder up
# through four resolutions, a self-attending mid block in each, its weights
# at bf16.

load("//lib/diffusion/model.star", "linear")

BLOCKS = [128, 256, 512, 512]
LAYERS_PER_BLOCK = 2
RGB = 3
TAPS3 = 9

def conv(name, c_out, c_in, taps):
    return struct(
        w = weight(name, [c_out, c_in * taps], dtype.bf16).conv_taps_major(c_in, taps),
        bias = weight(name + ".bias", [c_out], dtype.f32),
        c_in = c_in,
        c_out = c_out,
        taps = taps,
    )

def group_norm(name, c):
    return struct(
        weight = weight(name + ".weight", [c], dtype.f32),
        bias = weight(name + ".bias", [c], dtype.f32),
    )

def resnet(name, c_in, c_out):
    return struct(
        norm1 = group_norm(name + ".norm1", c_in),
        conv1 = conv(name + ".conv1", c_out, c_in, TAPS3),
        norm2 = group_norm(name + ".norm2", c_out),
        conv2 = conv(name + ".conv2", c_out, c_out, TAPS3),
        shortcut = conv(name + ".shortcut", c_out, c_in, 1) if c_in != c_out else None,
    )

def mid(name, c):
    return struct(
        res0 = resnet(name + ".res0", c, c),
        attn = struct(
            norm = group_norm(name + ".attn.norm", c),
            q = linear(name + ".attn.q", c, c, dtype.bf16),
            k = linear(name + ".attn.k", c, c, dtype.bf16),
            v = linear(name + ".attn.v", c, c, dtype.bf16),
            out = linear(name + ".attn.out", c, c, dtype.bf16),
            width = c,
        ),
        res1 = resnet(name + ".res1", c, c),
    )

def vae(channels, encoder_out, latent):
    """The VAE whose decoder takes `channels` and whose encoder gives
    `encoder_out`; `latent()` lays out what carries a latent to and from
    them."""
    top = BLOCKS[-1]
    up = []
    c_prev = top
    for i, c in enumerate(reversed(BLOCKS)):
        name = "vae.dec.up{}".format(i)
        resnets = []
        for r in range(LAYERS_PER_BLOCK + 1):
            resnets.append(resnet("{}.res{}".format(name, r), c_prev, c))
            c_prev = c
        last = i + 1 == len(BLOCKS)
        up.append(struct(
            resnets = resnets,
            upsample = conv(name + ".upsample", c, c, TAPS3) if not last else None,
        ))
    down = []
    c_prev = BLOCKS[0]
    for i, c in enumerate(BLOCKS):
        name = "vae.enc.down{}".format(i)
        resnets = []
        for r in range(LAYERS_PER_BLOCK):
            resnets.append(resnet("{}.res{}".format(name, r), c_prev, c))
            c_prev = c
        last = i + 1 == len(BLOCKS)
        down.append(struct(
            resnets = resnets,
            downsample = conv(name + ".downsample", c, c, TAPS3) if not last else None,
        ))
    return struct(
        latent = latent(),
        decoder = struct(
            conv_in = conv("vae.dec.conv_in", top, channels, TAPS3),
            mid = mid("vae.dec.mid", top),
            up = up,
            norm_out = group_norm("vae.dec.norm_out", BLOCKS[0]),
            conv_out = conv("vae.dec.conv_out", RGB, BLOCKS[0], TAPS3),
        ),
        encoder = struct(
            conv_in = conv("vae.enc.conv_in", BLOCKS[0], RGB, TAPS3),
            down = down,
            mid = mid("vae.enc.mid", top),
            norm_out = group_norm("vae.enc.norm_out", top),
            conv_out = conv("vae.enc.conv_out", encoder_out, top, TAPS3),
        ),
    )
