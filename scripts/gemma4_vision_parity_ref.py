#!/usr/bin/env python3
"""Dump the gemma-4 vision tower's weights and an HF reference, for the parity harnesses.

`gemma4_vision_full_parity{,_bf16}.cu` and `gemma4_vision_patch_parity.cu` have
named this script since they were written, and it did not exist -- so those
harnesses have never been runnable by anyone who did not already have the
dumps, and the vision tower's ~30 naive kernels could be read but not verified.

What comes out
--------------
    weights/<harness name>.npy    659 files, fp32 (the harness converts to bf16
                                  itself, and reads the clip bounds as f32[0])
    input_pixel_values_f32.npy    [N, 768] raw patch pixels, N = 9 * OUTL
    input_position_ids.npy        [N, 2] float32 patch (x, y)
    projected.npy                 HF output, bf16 rounded back to f32
    projected_f32.npy             HF output, fp32
    manifest.json                 shapes, grid, model id, revision

and with `--real-image`, the three the harness's second mode opens:

    realimg_pixel_values_f32.npy  [2520, 768] the PROCESSOR's own buffer
    realimg_position_ids.npy      [2520, 2] f32, trailing rows (-1, -1)
    realimg_projected_f32.npy     [OUTL, 2560] fp32 only

The two modes differ in what they exercise, which is why both exist. The
synthetic one is a fixed 60x42 grid with no padding and every patch valid,
and it carries the staged checkpoints, so a bad cosine says which layer. The
real one takes whatever geometry an image's aspect ratio gives -- 57x42 for
640x480, so 2394 valid patches of a padded 2520, and 266 pooled tokens rather
than 280 -- which is the padding strip and the pooling group index, and
nothing else checks those.

Name mapping, measured against the checkpoint rather than assumed
-----------------------------------------------------------------
    harness                                   HF
    vision.                                <- model.vision_tower.
    embed.                                 <- model.embed_vision.
    <proj>.linear.weight                      identical
    <proj>.{input,output}_{min,max}           identical, and they are scalar
                                              bf16 BUFFERS, not parameters

Two things this gets right that are easy to get wrong
-----------------------------------------------------
1. `input_pixel_values_f32.npy` holds RAW pixels. Both sides scale them:
   HF in `Gemma4VisionPatchEmbedder.forward` (`2 * (pixel_values - 0.5)`) and
   the driver in `k_scale` (gemma4_vision_forward.cu:48). Dumping HF's
   already-scaled activations would make the driver scale twice, and the
   harness would only report a poor cosine -- with no way to tell a bad kernel
   from a bad input.

2. `input_position_ids.npy` is float32, because the harness reads it as
   `const float*`. HF wants a LongTensor; the conversion happens here.

Usage
-----
    pip install torch transformers safetensors pillow
    python scripts/gemma4_vision_parity_ref.py --out /tmp/gemma4_vision_parity \\
                                               --real-image

    export PATH=/usr/local/cuda/bin:$PATH
    nvcc -O2 -arch=sm_89 -std=c++20 --extended-lambda --expt-relaxed-constexpr \\
         -I crates/driver-cuda/csrc/src -I crates/kernels-cuda/csrc/src \\
         crates/driver-cuda/csrc/tests/gemma4_vision_full_parity_bf16.cu -o /tmp/g4v
    /tmp/g4v /tmp/gemma4_vision_parity          # synthetic  0.99978
    /tmp/g4v /tmp/gemma4_vision_parity real     # real       0.99980

That nvcc line no longer runs
-----------------------------
The two cosines above are what it printed, which is why the command is kept:
it says what the numbers were measured with. Every path in it is deleted now.
`crates/driver-cuda/csrc` went at `b58db6c16` and took
`gemma4_vision_full_parity_bf16.cu` with it, and `crates/kernels-cuda/csrc/src`
was the ARCHIVE crate's host header tree, deleted with that whole crate at
`85c6c674b`. `gemma4_vision_forward.cu:48`, cited above for `k_scale`, is in
the first of those.

The second `-I` deserves the warning, because the name was reused.
`crates/kernels-cuda/csrc/src` EXISTS again -- it is the JIT crate's device
text, `.cuh` compiled by NVRTC at run time, with no host header in it. nvcc
would accept the flag and find none of what this command asked it for, which
is a worse failure than a missing directory.

The dump below is unaffected: it reads a checkpoint and writes `.npy`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# The synthetic case the harness hardcodes: OUTL = 280 pooled tokens, and a
# 3x3 pooling kernel, so N = 280 * 9 = 2520 patches. The grid has to be
# divisible by 3 on both axes for the harness's `grp` math (x/3 + gx*(y/3)).
GRID_W, GRID_H = 60, 42          # 2520 patches -> 20 x 14 = 280 pooled
PATCH_DIM = 768                  # 16 * 16 * 3
SEED = 0

LAYER_NORMS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
]
# Clipped linears: the harness's `clip()` reads `.linear.weight` plus four
# scalar bounds for each of these.
LAYER_CLIPPED = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
CLIP_BOUNDS = ["input_min", "input_max", "output_min", "output_max"]

GLOBALS = {
    "vision.patch_embedder.input_proj.weight":
        "model.vision_tower.patch_embedder.input_proj.weight",
    "vision.patch_embedder.position_embedding_table":
        "model.vision_tower.patch_embedder.position_embedding_table",
    "embed.embedding_projection.weight":
        "model.embed_vision.embedding_projection.weight",
}


def build_name_map(n_layers: int) -> dict[str, str]:
    """harness name -> HF name, for every file the harness opens."""
    m = dict(GLOBALS)
    for i in range(n_layers):
        h = f"vision.encoder.layers.{i}."
        f = f"model.vision_tower.encoder.layers.{i}."
        for w in LAYER_NORMS:
            m[h + w] = f + w
        for proj in LAYER_CLIPPED:
            m[f"{h}{proj}.linear.weight"] = f"{f}{proj}.linear.weight"
            for b in CLIP_BOUNDS:
                m[f"{h}{proj}.{b}"] = f"{f}{proj}.{b}"
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/tmp/gemma4_vision_parity")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--revision", default=None,
                    help="pin the checkpoint revision; recorded in manifest.json")
    ap.add_argument("--no-reference", action="store_true",
                    help="dump weights and inputs only, skip the HF forward")
    ap.add_argument("--real-image", metavar="WxH", nargs="?", const="640x480",
                    default=None,
                    help="also dump the harness's `real` mode: a deterministic "
                         "image of this size through the REAL processor, so the "
                         "patch count is variable and padded rather than the "
                         "synthetic grid's fixed 2520")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import AutoConfig, Gemma4ForConditionalGeneration
    except ImportError as e:
        print(f"needs torch + transformers: {e}", file=sys.stderr)
        print("  pip install torch transformers safetensors", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out)
    (out / "weights").mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model}" + (f" @ {args.revision}" if args.revision else ""))
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16)
    model.eval()

    cfg = AutoConfig.from_pretrained(args.model, revision=args.revision)
    vcfg = getattr(cfg, "vision_config", cfg)
    n_layers = getattr(vcfg, "num_hidden_layers", 16)

    # Parameters and buffers together: the clip bounds are BUFFERS (scalar
    # bf16), so a named_parameters()-only sweep silently drops 448 of the 659
    # files and the harness dies on the first `scal()`.
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))

    name_map = build_name_map(n_layers)
    missing = {h: f for h, f in name_map.items() if f not in tensors}
    if missing:
        print(f"\n{len(missing)} of {len(name_map)} names not found, e.g.:",
              file=sys.stderr)
        for h, f in list(missing.items())[:5]:
            print(f"   {h}\n       expected HF name: {f}", file=sys.stderr)
        vis = [n for n in tensors if "vision" in n or "embed_vision" in n]
        print(f"\nnames this checkpoint does have ({len(vis)} vision-ish, "
              f"first 20):", file=sys.stderr)
        for n in vis[:20]:
            print(f"   {n}", file=sys.stderr)
        print("\nThe harness opens files by name, so a rename upstream breaks "
              "it here rather than silently.", file=sys.stderr)
        return 3

    total = 0
    for harness_name, hf_name in name_map.items():
        t = tensors[hf_name].detach().float().contiguous().cpu().numpy()
        p = out / "weights" / f"{harness_name}.npy"
        np.save(p, t)
        total += t.nbytes
    print(f"wrote {len(name_map)} weights, {total/1e6:.1f} MB -> {out}/weights")

    # ── synthetic input ──────────────────────────────────────────────────
    n_patches = GRID_W * GRID_H
    gen = np.random.default_rng(SEED)
    # Raw pixels in [0, 1]. NOT pre-scaled: both HF and the driver apply
    # 2*(x-0.5) themselves (see the module docstring).
    pixels = gen.random((n_patches, PATCH_DIM), dtype=np.float32)
    pos = np.empty((n_patches, 2), dtype=np.float32)
    for i in range(n_patches):
        pos[i, 0] = i % GRID_W       # x
        pos[i, 1] = i // GRID_W      # y
    np.save(out / "input_pixel_values_f32.npy", pixels)
    np.save(out / "input_position_ids.npy", pos)
    out_len = (GRID_W // 3) * (GRID_H // 3)
    print(f"synthetic input: {GRID_W}x{GRID_H} = {n_patches} patches "
          f"-> {out_len} pooled tokens")

    if args.no_reference:
        print("skipping the HF forward (--no-reference)")
        return 0

    # The harness checks three staged intermediates in synthetic mode
    # (gemma4_vision_full_parity_bf16.cu:122/125/130), so the reference has to
    # carry them too or it dies on `open .../layer0_f32.npy`.
    #   layer0             [N, 768]     encoder layer 0 output
    #   layer_last         [N, 768]     last encoder layer, BEFORE pooling
    #   pooled_last_hidden [OUTL, 768]  after pooling -- which is what the
    #                                   tower returns as last_hidden_state
    taps: dict[str, "torch.Tensor"] = {}

    def tap(name):
        def hook(_m, _i, o):
            taps[name] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook

    layers = model.model.vision_tower.encoder.layers
    handles = [layers[0].register_forward_hook(tap("layer0")),
               layers[-1].register_forward_hook(tap("layer_last"))]

    print("running the HF vision tower for the reference")
    with torch.no_grad():
        pix = torch.from_numpy(pixels).unsqueeze(0).to(torch.bfloat16)
        pid = torch.from_numpy(pos).unsqueeze(0).long()
        feats = model.model.get_image_features(
            pixel_values=pix, image_position_ids=pid)
        # pooler_output comes back ALREADY unbatched, [OUTL, text_hidden].
        # Indexing [0] here took a single row and produced a 1-D (1536,).
        projected = feats.pooler_output.float()
        taps["pooled_last_hidden"] = feats.last_hidden_state.detach()
    for h in handles:
        h.remove()

    for tag in ("layer0", "layer_last", "pooled_last_hidden"):
        if tag not in taps:
            print(f"\nNOTE: no {tag} captured -- the hook points moved. The "
                  f"harness opens {tag}_f32.npy and will fail there.",
                  file=sys.stderr)
            continue
        t = taps[tag].float().squeeze(0).contiguous().cpu().numpy()
        np.save(out / f"{tag}_f32.npy", t)
        print(f"  ckpt {tag:20} {t.shape}")

    np.save(out / "projected_f32.npy", projected.numpy())
    np.save(out / "projected.npy",
            projected.to(torch.bfloat16).float().numpy())
    print(f"projected {tuple(projected.shape)}  "
          f"rms={projected.pow(2).mean().sqrt().item():.3f}")

    (out / "manifest.json").write_text(json.dumps({
        "model": args.model,
        "revision": args.revision or "(default)",
        "grid": [GRID_W, GRID_H],
        "n_patches": n_patches,
        "output_length": out_len,
        "patch_dim": PATCH_DIM,
        "seed": SEED,
        "n_weights": len(name_map),
        "projected_shape": list(projected.shape),
    }, indent=2) + "\n")

    # E4B, not E2B: the harness hardcodes TXT=2560, which is E4B's text
    # hidden size (E2B's is 1536). The vision tower is identical between them
    # (hidden 768, 16 layers) -- it is the projection target that differs.
    if projected.shape[-1] != 2560:
        print(f"\nNOTE: projected width is {projected.shape[-1]}, and the "
              f"harness hardcodes TXT=2560. That constant is E4B's text hidden "
              f"size; pass --model google/gemma-4-E4B-it.", file=sys.stderr)
    if projected.shape[0] != out_len:
        print(f"\nNOTE: HF pooled to {projected.shape[0]} tokens, the grid "
              f"implies {out_len}, and the harness hardcodes OUTL=280 for the "
              f"synthetic case. Reconcile before trusting a comparison.",
              file=sys.stderr)

    if args.real_image:
        return dump_real_image(args, model, out, np, torch)
    return 0


def dump_real_image(args, model, out, np, torch) -> int:
    """The harness's `real` mode: the processor's own output, padding and all.

    `gemma4_vision_full_parity_bf16.cu <dir> real` opens three files that no
    script produced, so that path had never run. The difference from the
    synthetic mode is the point: the processor emits a FIXED 2520-patch buffer
    with the trailing rows marked (-1, -1), and the real grid is whatever the
    image's aspect ratio gives -- 57x42 for 640x480, so 2394 valid patches and
    266 pooled tokens, neither of which is the synthetic 2520/280.

    The image is generated rather than loaded so this reproduces anywhere. It
    is only a carrier for the processor's geometry; what is being checked is
    the padding and the pooling, not anything about the picture.
    """
    from PIL import Image
    from transformers import AutoProcessor

    w, _, h = args.real_image.partition("x")
    w, h = int(w), int(h or w)

    proc = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    gen = np.random.default_rng(SEED)
    img = Image.fromarray((gen.random((h, w, 3)) * 255).astype(np.uint8))
    # The image token has to be in the text or `validate_inputs` rejects the
    # call -- the processor counts them against the image list.
    enc = proc(text="<|image|>", images=[img], return_tensors="pt")

    pix = enc["pixel_values"][0]          # [2520, 768] f32
    pid = enc["image_position_ids"][0]    # [2520, 2]   i64, (-1,-1) = padding

    pos = pid.numpy()
    valid = pos[:, 0] >= 0
    n_valid = int(valid.sum())
    # The harness reads the first N rows of BOTH arrays after counting the
    # valid ones (gemma4_vision_full_parity_bf16.cu:91-97). That is only right
    # if the padding is trailing, so check rather than assume -- a processor
    # that interleaved padding would otherwise be caught as a bad cosine.
    if not (valid[:n_valid].all() and not valid[n_valid:].any()):
        print(f"\n{n_valid} valid patches but they are NOT the leading rows. "
              f"The harness slices [0, N) of both arrays, so it would read "
              f"padding as data. Compact them here first, or teach the harness "
              f"to gather.", file=sys.stderr)
        return 4
    if n_valid % 9:
        print(f"\n{n_valid} valid patches is not a multiple of 9, and the "
              f"harness computes OUTL = N/9 for the 3x3 pooling.",
              file=sys.stderr)
        return 4

    np.save(out / "realimg_pixel_values_f32.npy", pix.float().numpy())
    # f32, not the i64 the processor returns: the harness casts the buffer to
    # `const float*` to test `>= 0` and to build the pooling group index.
    np.save(out / "realimg_position_ids.npy", pos.astype(np.float32))

    gx = int(pos[valid, 0].max()) + 1
    gy = int(pos[valid, 1].max()) + 1
    print(f"real image {w}x{h}: grid {gx}x{gy}, {n_valid} valid of "
          f"{pos.shape[0]} patches -> {n_valid // 9} pooled tokens")

    with torch.no_grad():
        feats = model.model.get_image_features(
            pixel_values=pix.unsqueeze(0).to(torch.bfloat16),
            image_position_ids=pid.unsqueeze(0))
        projected = feats.pooler_output.float()

    # fp32 only. The harness compares bf16 output against this at a 6%
    # threshold precisely because it is the unrounded reference; there is no
    # `realimg_projected.npy` and it does not open one.
    np.save(out / "realimg_projected_f32.npy", projected.numpy())
    print(f"realimg_projected_f32 {tuple(projected.shape)}  "
          f"rms={projected.pow(2).mean().sqrt().item():.3f}")

    if projected.shape[0] != n_valid // 9:
        print(f"\nNOTE: HF pooled to {projected.shape[0]} tokens and the "
              f"harness will size its output buffer at {n_valid // 9}. They "
              f"have to agree or the comparison runs off the end.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
