#!/usr/bin/env python3
"""Dump the Gemma-4 audio tower's weights and one reference forward.

`gemma4_audio_full_parity.cu` has read `/tmp/gemma4_audio_parity` since it was
written and no script produced it, so the harness has never run. This is that
script, built the same way `gemma4_vision_parity_ref.py` was: the harness opens
files BY NAME, so the name map here is derived from the names it opens rather
than from a guess about the checkpoint.

What comes out
--------------
    weights/<harness name>.npy   f32. The harness does `as_f32` then
                                 `upload_bf`, and bf16 -> f32 -> bf16 is
                                 lossless, so f32 costs only disk.
    input_features_f32.npy       [n_frames, 128] synthetic log-mel
    projected.npy                [N, TXT] the answer, ROUNDED TO BF16
    projected_f32.npy            the same in f32, for reading the gap
    sscp_out.npy layer{0..11}.npy encoder_out.npy
                                 the harness's checkpoint taps, so a bad
                                 cosine at the end says WHERE

Why the reference is rounded
----------------------------
The driver computes in bf16. Comparing its output against an f32 reference
charges it for arithmetic it never claimed to do -- at bf16's 8-bit
significand the quantum at magnitude m is about m/256, and a "failure" of
that size is the format, not the kernel. `projected.npy` is the metric;
`projected_f32.npy` is there to tell a real error from a rounding one.

The name map
------------
Pure prefix substitution, which is worth stating because it means a rename
upstream breaks LOUDLY here (the script reports the missing names) rather
than as a bad cosine in the harness:

    audio.<x>                          -> model.audio_tower.<x>
    embed.embedding_projection.weight  -> model.embed_audio.embedding_projection.weight

Usage
-----
    python scripts/gemma4_audio_parity_ref.py --out /tmp/gemma4_audio_parity

    export PATH=/usr/local/cuda/bin:$PATH
    nvcc -O2 -arch=sm_89 -std=c++20 --extended-lambda --expt-relaxed-constexpr \\
         -I crates/driver-cuda/csrc/src -I crates/kernels-cuda/csrc/src \\
         crates/driver-cuda/csrc/tests/gemma4_audio_full_parity.cu \\
         crates/driver-cuda/csrc/src/model/gemma4/gemma4_audio_forward.cu \\
         -o /tmp/gap
    /tmp/gap /tmp/gemma4_audio_parity

That nvcc line no longer runs
-----------------------------
It is kept as the record of what the harness was compiled against, not as an
instruction. Both translation units it names and both include roots are gone:
`crates/driver-cuda/csrc` was deleted at `b58db6c16`, which took
`gemma4_audio_full_parity.cu` and `gemma4_audio_forward.cu` together, and
`crates/kernels-cuda/csrc/src` was the ARCHIVE crate's host header tree --
the `.hpp` that declared the `.cu` launchers -- deleted with the crate itself
at `85c6c674b`.

Flag the second `-I` in particular, because the name has been reused since.
`crates/kernels-cuda/csrc/src` EXISTS again and holds the JIT crate's device
text: `.cuh` that NVRTC compiles at run time, no host headers at all. nvcc
would take the flag happily and find nothing behind it that this command
wanted, which reads as a source error rather than a stale path.

The dump below is unaffected. It reads a checkpoint and writes `.npy`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# 188 log-mel frames -> conv(conv(188)) = 47 audio tokens. Long enough that the
# 12 conformer layers see a real sequence (the depthwise conv has K=5 and the
# attention is relative-position), short enough to stay a quick check.
N_FRAMES = 188
N_MEL = 128
N_LAYERS = 12
SEED = 0

# Clipped linears: the harness's `clip()` opens `.linear.weight` plus four
# scalar bounds for each of these.
CLIP_BOUNDS = ["input_min", "input_max", "output_min", "output_max"]
LAYER_CLIPPED = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.post",
    "lconv1d.linear_start", "lconv1d.linear_end",
    "feed_forward1.ffw_layer_1", "feed_forward1.ffw_layer_2",
    "feed_forward2.ffw_layer_1", "feed_forward2.ffw_layer_2",
]
LAYER_PLAIN = [
    "norm_pre_attn.weight", "norm_post_attn.weight", "norm_out.weight",
    "self_attn.relative_k_proj.weight", "self_attn.per_dim_scale",
    "lconv1d.pre_layer_norm.weight", "lconv1d.conv_norm.weight",
    "lconv1d.depthwise_conv1d.weight",
    "feed_forward1.pre_layer_norm.weight", "feed_forward1.post_layer_norm.weight",
    "feed_forward2.pre_layer_norm.weight", "feed_forward2.post_layer_norm.weight",
]
GLOBALS_PLAIN = [
    "subsample_conv_projection.layer0.conv.weight",
    "subsample_conv_projection.layer0.norm.weight",
    "subsample_conv_projection.layer1.conv.weight",
    "subsample_conv_projection.layer1.norm.weight",
    "subsample_conv_projection.input_proj_linear.weight",
    "output_proj.weight",
    "output_proj.bias",
]


def build_name_map() -> dict[str, str]:
    """harness name -> HF name, for every file the harness opens."""
    m: dict[str, str] = {}
    for w in GLOBALS_PLAIN:
        m[f"audio.{w}"] = f"model.audio_tower.{w}"
    m["embed.embedding_projection.weight"] = \
        "model.embed_audio.embedding_projection.weight"
    for i in range(N_LAYERS):
        h, f = f"audio.layers.{i}.", f"model.audio_tower.layers.{i}."
        for w in LAYER_PLAIN:
            m[h + w] = f + w
        for proj in LAYER_CLIPPED:
            m[f"{h}{proj}.linear.weight"] = f"{f}{proj}.linear.weight"
            for b in CLIP_BOUNDS:
                m[f"{h}{proj}.{b}"] = f"{f}{proj}.{b}"
    return m


def subsampled_len(n_frames: int) -> int:
    """Mirrors `gemma4_audio_subsampled_len` in the driver's header."""
    conv = lambda n: (n + 2 * 1 - 3) // 2 + 1
    return conv(conv(n_frames))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/tmp/gemma4_audio_parity")
    ap.add_argument("--model", default="google/gemma-4-E4B-it",
                    help="E4B projects to 2560, which is the harness's TXT; "
                         "E2B projects to 1536 and will not match it")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--no-reference", action="store_true",
                    help="weights only; skip the HF forward")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import Gemma4ForConditionalGeneration
    except ImportError as e:
        print(f"needs torch + transformers: {e}", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out)
    (out / "weights").mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model}")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16)
    model.eval()

    # Buffers as well as parameters: the clip bounds are registered buffers on
    # the vision side and the audio side is built the same way. Reading only
    # `named_parameters()` silently drops them and the harness dies on its
    # first `scal_bf`.
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))

    name_map = build_name_map()
    missing = {h: f for h, f in name_map.items() if f not in tensors}
    if missing:
        print(f"\n{len(missing)} of {len(name_map)} names not found, e.g.:",
              file=sys.stderr)
        for h, f in list(missing.items())[:5]:
            print(f"   {h}\n       expected HF name: {f}", file=sys.stderr)
        au = [n for n in tensors if "audio" in n]
        print(f"\nnames this checkpoint does have ({len(au)} audio-ish, "
              f"first 20):", file=sys.stderr)
        for n in au[:20]:
            print(f"   {n}", file=sys.stderr)
        return 3

    total = 0
    for harness_name, hf_name in name_map.items():
        t = tensors[hf_name].detach().float().contiguous().cpu().numpy()
        np.save(out / "weights" / f"{harness_name}.npy", t)
        total += t.nbytes
    print(f"wrote {len(name_map)} weights, {total/1e6:.1f} MB -> {out}/weights")

    # ── synthetic input ──────────────────────────────────────────────────
    gen = np.random.default_rng(SEED)
    # Log-mel energies, not raw audio: real ones sit roughly in [-12, +2] with
    # most mass below zero, so a plain standard normal would exercise a range
    # the encoder never sees. Shifted and scaled to that band.
    feats = (gen.standard_normal((args.frames, N_MEL)) * 2.0 - 5.0
             ).astype(np.float32)
    np.save(out / "input_features_f32.npy", feats)
    out_len = subsampled_len(args.frames)
    print(f"synthetic log-mel: [{args.frames}, {N_MEL}] -> {out_len} audio tokens")

    if args.no_reference:
        print("skipping the HF forward (--no-reference)")
        return 0

    # ── reference forward, with the harness's checkpoints tapped ─────────
    taps: dict[str, "np.ndarray"] = {}

    def tap(name: str):
        def hook(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            taps[name] = t.detach().float().cpu().numpy().reshape(-1)
        return hook

    tower = model.model.audio_tower
    handles = [tower.subsample_conv_projection.register_forward_hook(tap("sscp_out"))]
    for i, layer in enumerate(tower.layers):
        handles.append(layer.register_forward_hook(tap(f"layer{i}")))

    with torch.no_grad():
        x = torch.from_numpy(feats).to(torch.bfloat16).unsqueeze(0)  # [1, T, mel]
        # A `Gemma4AudioModelOutput`, not the `tuple[Tensor, BoolTensor]` the
        # signature advertises: tuple-unpacking it drops the None fields, so
        # with no attention_mask passed it yields ONE value and the unpack
        # raises. Read the field.
        enc = tower(input_features=x).last_hidden_state
        # `output_proj` (1024 -> 1536) is applied inside the tower, so this is
        # already the harness's `encoder_out`, not the raw conformer output.
        enc2 = enc.reshape(-1, enc.shape[-1])
        # The WHOLE embedder, not `.embedding_projection`. A
        # `Gemma4MultimodalEmbedder` is a parameterless RMSNorm
        # (`embedding_pre_projection_norm`, `with_scale=False`) followed by the
        # linear, and calling only the linear skips the norm.
        #
        # Worth the comment because of how that failure presented: the harness
        # PASSED. Its criterion is cosine > 0.99 and an unnormalized input to a
        # linear is very nearly a pure rescale, so direction survived
        # (cosine 0.99681) while rel_rms_err went from 0.75% at `encoder_out`
        # to 87.9% at `projected`. The magnitude column is the one that said
        # something was wrong; the verdict did not.
        projected = model.model.embed_audio(enc2).float()

    for h in handles:
        h.remove()

    if enc2.shape[0] != out_len:
        print(f"\nNOTE: HF produced {enc2.shape[0]} audio tokens, the harness "
              f"computes {out_len} from `gemma4_audio_subsampled_len`. They "
              f"have to agree or the comparison is off by a row.",
              file=sys.stderr)

    np.save(out / "encoder_out.npy",
            enc2.detach().to(torch.bfloat16).float().cpu().numpy().reshape(-1))
    for name, arr in taps.items():
        np.save(out / f"{name}.npy",
                arr.astype(np.float32).astype(np.dtype("float32")))
    np.save(out / "projected_f32.npy", projected.cpu().numpy())
    np.save(out / "projected.npy",
            projected.to(torch.bfloat16).float().cpu().numpy())

    print(f"encoder_out {tuple(enc2.shape)}  projected {tuple(projected.shape)}")
    print(f"captured taps: {', '.join(sorted(taps))}")
    if len(taps) != N_LAYERS + 1:
        print(f"\nNOTE: {len(taps)} taps captured, expected {N_LAYERS + 1}. "
              f"The hook points moved; the harness will simply skip the ones "
              f"whose .npy is absent, which reads as a passing stage.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
