# The weights of the Qwen 3.5-family vision tower: a patch embedding with a
# learned position grid, `depth` pre-norm ViT blocks, and a merger that folds
# each `merge`×`merge` patch square into one row of the trunk's width.

SMALL = struct(depth = 12, hidden = 768, heads = 12, inter = 3072)
LARGE = struct(depth = 27, hidden = 1152, heads = 16, inter = 4304)

PATCH_WIDTH = 1536
MERGE = 2
POSITIONS = 2304

def tower(size, out, dt):
    """`size`'s tower, merging into rows `out` wide, its weights at `dt`."""
    th = size.hidden
    ti = size.inter
    merged = MERGE * MERGE * th
    head_dim = th // size.heads
    plane = lambda s, dims: weight("visual." + s, dims, dt)
    vec = lambda s, length: weight("visual." + s, [length], dt)

    def block(l):
        b = lambda s: "block.{}.{}".format(l, s)
        return struct(
            norm1 = vec(b("norm1"), th),
            norm1_bias = vec(b("norm1_bias"), th),
            qkv = plane(b("qkv"), [3 * th, th]),
            qkv_bias = vec(b("qkv_bias"), 3 * th),
            proj = plane(b("proj"), [th, th]),
            proj_bias = vec(b("proj_bias"), th),
            norm2 = vec(b("norm2"), th),
            norm2_bias = vec(b("norm2_bias"), th),
            fc1 = plane(b("fc1"), [ti, th]),
            fc1_bias = vec(b("fc1_bias"), ti),
            fc2 = plane(b("fc2"), [th, ti]),
            fc2_bias = vec(b("fc2_bias"), th),
        )

    return struct(
        hidden = th,
        heads = size.heads,
        head_dim = head_dim,
        merge = MERGE,
        patch_width = PATCH_WIDTH,
        taps = 4,
        positions = POSITIONS,
        theta = 10000.0,
        norm_eps = 1e-6,
        sm_scale = f32(1.0 / f32(sqrt(head_dim))),
        patch_embed = plane("patch_embed", [th, PATCH_WIDTH]),
        patch_embed_bias = vec("patch_embed_bias", th),
        pos_embed = plane("pos_embed", [POSITIONS, th]),
        blocks = [block(l) for l in range(size.depth)],
        merger = struct(
            norm = vec("merger_norm", th),
            norm_bias = vec("merger_norm_bias", th),
            fc1 = plane("merger_fc1", [merged, merged]),
            fc1_bias = vec("merger_fc1_bias", merged),
            fc2 = plane("merger_fc2", [out, merged]),
            fc2_bias = vec("merger_fc2_bias", out),
        ),
    )
