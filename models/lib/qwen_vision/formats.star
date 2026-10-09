# How a Qwen 3.5-family checkpoint spells its vision tower, under `prefix`;
# mlx_lm stores the patch embedding channels-last.

load("//lib/reads/formats.star", "flattened")

CHANNELS = 3

def read_tower(reads, t, prefix, channels_last):
    v = lambda s: prefix + s
    flat = flattened(v("patch_embed.proj.weight"), t.patch_embed.shape, broadcast = True)
    if channels_last:
        per = t.patch_embed.shape[1] // CHANNELS
        flat = flat.gather(1, [j * CHANNELS + c for c in range(CHANNELS) for j in range(per)])
    reads.read_expr(t.patch_embed, flat)
    reads.read(t.patch_embed_bias, v("patch_embed.proj.bias"))
    reads.read(t.pos_embed, v("pos_embed.weight"))
    for l, blk in enumerate(t.blocks):
        n = lambda s: v("blocks.{}.{}".format(l, s))
        for w, name in [
            (blk.norm1, "norm1.weight"),
            (blk.norm1_bias, "norm1.bias"),
            (blk.qkv, "attn.qkv.weight"),
            (blk.qkv_bias, "attn.qkv.bias"),
            (blk.proj, "attn.proj.weight"),
            (blk.proj_bias, "attn.proj.bias"),
            (blk.norm2, "norm2.weight"),
            (blk.norm2_bias, "norm2.bias"),
            (blk.fc1, "mlp.linear_fc1.weight"),
            (blk.fc1_bias, "mlp.linear_fc1.bias"),
            (blk.fc2, "mlp.linear_fc2.weight"),
            (blk.fc2_bias, "mlp.linear_fc2.bias"),
        ]:
            reads.read(w, n(name))
    mg = t.merger
    for w, name in [
        (mg.norm, "merger.norm.weight"),
        (mg.norm_bias, "merger.norm.bias"),
        (mg.fc1, "merger.linear_fc1.weight"),
        (mg.fc1_bias, "merger.linear_fc1.bias"),
        (mg.fc2, "merger.linear_fc2.weight"),
        (mg.fc2_bias, "merger.linear_fc2.bias"),
    ]:
        reads.read(w, v(name))
