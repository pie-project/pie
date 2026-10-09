# How a KDA mixer is named in a transformers checkpoint: q, k and v each a
# projection and a depthwise convolution bank of their own.

load("//lib/reads/formats.star", "squeezed")

def qkv(at):
    """The names of the q, k and v projections the packed `qkv` reads."""
    return [at("self_attn.{}_proj.weight".format(p)) for p in ["q", "k", "v"]]

def conv(k, at):
    """The packed `conv` bank, read from the q, k and v banks."""
    return concat(k.conv.cut_axis, [squeezed(at("self_attn.{}_conv1d.weight".format(p))) for p in ["q", "k", "v"]])

def gate(k, at):
    """The output gate's projections, each beside its name."""
    if len(k.gate) == 1:
        return [(k.gate[0], at("self_attn.g_proj.weight"))]
    return [(k.gate[0], at("self_attn.g_a_proj.weight")), (k.gate[1], at("self_attn.g_b_proj.weight"))]
