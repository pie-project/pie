#!/usr/bin/env python3
"""Mimi neural-codec DECODER parity reference (codes → waveform).

Runs the *real* transformers `MimiModel` (loaded from the local `kyutai/mimi`
checkpoint) on a fixed synthetic set of RVQ codes `[1, 32, T]`, exercising the
full DECODE path the CUDA `mimi_decoder` forward (AUDIO_OUTPUT.md) must
reproduce:

    codes [B, 32, T]
      → quantizer.decode               (SplitRVQ: 1 semantic + 31 acoustic;
                                         per-codebook nn.functional.embedding of
                                         a 2048×256 codebook, summed; then a
                                         1×1 conv `output_proj` 256→512)
      → upsample                       (ConvTranspose1d k4 s2 groups=512 → ×2 frames)
      → decoder_transformer            (8-layer Mimi transformer:
                                         LayerNorm(+bias) → RoPE attn (sliding
                                         window 250) + layer_scale → LayerNorm →
                                         GELU MLP + layer_scale)
      → decoder (SEANet)               (Conv1d k7 512→1024, then 4× [ELU →
                                         ConvTranspose1d (ratios 8,6,5,4) →
                                         ResnetBlock], ELU, Conv1d k3 →1) → 24kHz

This is *input-agnostic* and *standalone*: feed identical synthetic codes to
both torch and the CUDA decoder and compare. So a synthetic-but-valid set of
codes is sufficient and avoids any dependency on the CSM model (validated
end-to-end separately in csm_generate_ref.py).

Outputs → OUT_DIR as .npy + manifest.json (shape/dtype/mean/std/first values),
plus ALL Mimi DECODER weights (`decoder.*`, `decoder_transformer.*`,
`upsample.*`, `quantizer.*`) to weights/ so the CUDA Mimi decoder is
parity-testable in isolation (codes → waveform).

Usage: python3 mimi_decoder_parity_ref.py [checkpoint_dir] [out_dir]

Mirrors scripts/gemma4_vision_parity_ref.py / gemma4_audio_parity_ref.py
conventions. Parity metric (per MULTIMODAL.md §2.2): bf16-vs-bf16 rel_rms +
cosine; this script dumps fp32 references too for a tight standalone check.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

# ── Locate the checkpoint (guard with a clear message if absent) ─────────────
_DEFAULT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--kyutai--mimi/snapshots/*"
)
if len(sys.argv) > 1:
    CKPT = sys.argv[1]
else:
    _hits = [p for p in glob.glob(_DEFAULT_GLOB) if os.path.isdir(p)]
    if not _hits:
        sys.exit(
            "kyutai/mimi checkpoint not found. Expected a snapshot under\n"
            f"  {_DEFAULT_GLOB}\n"
            "Download it first:\n"
            "  huggingface-cli download kyutai/mimi\n"
            "or pass the checkpoint dir as argv[1]."
        )
    CKPT = _hits[0]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mimi_decoder_parity"
os.makedirs(OUT, exist_ok=True)

from transformers import MimiModel  # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

model = MimiModel.from_pretrained(CKPT).to(dev).eval()
cfg = model.config
print(
    f"mimi: hidden={cfg.hidden_size} codebook_size={cfg.codebook_size} "
    f"codebook_dim={cfg.codebook_dim} vq_hidden={cfg.vector_quantization_hidden_dimension} "
    f"num_quantizers={cfg.num_quantizers} semantic={cfg.num_semantic_quantizers} "
    f"num_filters={cfg.num_filters} ratios={list(cfg.upsampling_ratios)} "
    f"frame_rate={cfg.frame_rate} encodec_frame_rate={cfg.encodec_frame_rate} "
    f"sr={cfg.sampling_rate} dec_xf_layers={cfg.num_hidden_layers} "
    f"dec_xf_heads={cfg.num_attention_heads} sliding_window={cfg.sliding_window}"
)

# ── Fixed synthetic RVQ codes [1, 32, T] (random in [0, codebook_size)) ──────
NUM_CB = cfg.num_quantizers          # 32
T = 25                                # 25 frames @ 12.5 Hz ≈ 2.0 s of audio
g = torch.Generator().manual_seed(1234)
codes = torch.randint(0, cfg.codebook_size, (1, NUM_CB, T), generator=g).to(dev)
print(f"codes: {tuple(codes.shape)} in [0,{cfg.codebook_size}); "
      f"≈ {T / cfg.frame_rate:.2f}s @ {cfg.sampling_rate}Hz")

caps = {}
caps["input_codes"] = codes

# Tap the RVQ dequantize directly: `quantizer.decode` is NOT called via a
# module __call__ during MimiModel.decode (it's a plain method), so a forward
# hook on it never fires. Capture its output (post-output_proj, [B, 512, T])
# explicitly so the CUDA RVQ-sum + output_proj stage has a parity target.
with torch.no_grad():
    caps["dequantized_embeddings"] = model.quantizer.decode(codes).detach()

# ── Hooks: dequantized embeddings + a couple of SEANet decoder intermediates ─
def _unwrap(out):
    # Conv/ELU modules return a tensor; the decoder_transformer returns a
    # BaseModelOutputWithPast (take last_hidden_state); some return tuples.
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, tuple):
        return out[0]
    return out


def _mk_hook(name):
    def hook(_m, _inp, out):
        caps[name] = _unwrap(out).detach()
    return hook


# upsample output (input to the decoder_transformer).
model.upsample.register_forward_hook(_mk_hook("upsampled_embeddings"))
# decoder_transformer output, transposed back to [B, 512, T'] downstream.
model.decoder_transformer.register_forward_hook(_mk_hook("decoder_transformer_out"))
# SEANet decoder: first conv (layer 0) and first transposed conv (layer 2).
model.decoder.layers[0].register_forward_hook(_mk_hook("seanet_conv0"))
model.decoder.layers[2].register_forward_hook(_mk_hook("seanet_convtr0"))
# Last SEANet conv before output (final 64→1 conv).
model.decoder.layers[-1].register_forward_hook(_mk_hook("seanet_conv_last"))

with torch.no_grad():
    # MimiModel.decode(audio_codes) → MimiDecoderOutput(audio_values, ...).
    dec_out = model.decode(codes)
    waveform = dec_out.audio_values if hasattr(dec_out, "audio_values") else dec_out[0]

caps["output_waveform"] = waveform.detach()
n_samples = int(waveform.shape[-1])
print(f"waveform: {tuple(waveform.shape)} ({n_samples} samples, "
      f"≈ {n_samples / cfg.sampling_rate:.3f}s)")

# ── Dump bf16 reference activations + manifest ────────────────────────────────
manifest = {
    "checkpoint": CKPT,
    "config": {
        "hidden_size": cfg.hidden_size,
        "codebook_size": cfg.codebook_size,
        "codebook_dim": cfg.codebook_dim,
        "vector_quantization_hidden_dimension": cfg.vector_quantization_hidden_dimension,
        "num_quantizers": cfg.num_quantizers,
        "num_semantic_quantizers": cfg.num_semantic_quantizers,
        "num_filters": cfg.num_filters,
        "upsampling_ratios": list(cfg.upsampling_ratios),
        "kernel_size": cfg.kernel_size,
        "last_kernel_size": cfg.last_kernel_size,
        "residual_kernel_size": cfg.residual_kernel_size,
        "dilation_growth_rate": cfg.dilation_growth_rate,
        "compress": cfg.compress,
        "num_residual_layers": cfg.num_residual_layers,
        "use_causal_conv": cfg.use_causal_conv,
        "pad_mode": cfg.pad_mode,
        "trim_right_ratio": cfg.trim_right_ratio,
        "upsample_groups": cfg.upsample_groups,
        "use_conv_shortcut": cfg.use_conv_shortcut,
        "frame_rate": cfg.frame_rate,
        "encodec_frame_rate": cfg.encodec_frame_rate,
        "sampling_rate": cfg.sampling_rate,
        "layer_scale_initial_scale": cfg.layer_scale_initial_scale,
        # decoder_transformer dims.
        "dec_xf_num_hidden_layers": cfg.num_hidden_layers,
        "dec_xf_num_attention_heads": cfg.num_attention_heads,
        "dec_xf_num_key_value_heads": cfg.num_key_value_heads,
        "dec_xf_head_dim": cfg.head_dim,
        "dec_xf_intermediate_size": cfg.intermediate_size,
        "dec_xf_hidden_act": cfg.hidden_act,
        "dec_xf_rope_theta": getattr(cfg, "rope_theta", 10000.0),
        "dec_xf_sliding_window": cfg.sliding_window,
        "dec_xf_norm_eps": getattr(cfg, "norm_eps", 1e-5),
        "n_frames": T,
        "n_samples": n_samples,
    },
    "tensors": {},
}
for name, t in caps.items():
    arr = t.float().cpu().numpy() if t.dtype.is_floating_point else t.cpu().numpy()
    np.save(os.path.join(OUT, name + ".npy"), arr)
    flat = arr.reshape(-1).astype(np.float64)
    manifest["tensors"][name] = {
        "shape": list(arr.shape), "dtype": str(t.dtype),
        "mean": float(flat.mean()) if flat.size else 0.0,
        "std": float(flat.std()) if flat.size else 0.0,
        "first8": [float(x) for x in flat[:8]],
    }
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# ── fp32 reference run for a tight standalone CUDA check ──────────────────────
model32 = model.float()
caps32 = {}
with torch.no_grad():
    caps32["dequantized_embeddings"] = model32.quantizer.decode(codes).detach()
model32.upsample.register_forward_hook(
    lambda _m, _i, o: caps32.__setitem__("upsampled_embeddings", _unwrap(o).detach()))
model32.decoder_transformer.register_forward_hook(
    lambda _m, _i, o: caps32.__setitem__("decoder_transformer_out", _unwrap(o).detach()))
with torch.no_grad():
    dec32 = model32.decode(codes)
    wave32 = dec32.audio_values if hasattr(dec32, "audio_values") else dec32[0]
    caps32["output_waveform"] = wave32.detach()
for name, t in caps32.items():
    np.save(os.path.join(OUT, name + "_f32.npy"), t.cpu().numpy())

# ── Dump ALL decoder-side weights (the ones the CUDA decoder consumes) ───────
# Decode path touches: quantizer.* (RVQ codebooks + in/out projs),
# upsample.*, decoder_transformer.*, decoder.* (SEANet). We also derive each
# codebook's `embed = embed_sum / cluster_usage.clamp(min=eps)[:,None]` (the
# property MimiEuclideanCodebook.embed) so the CUDA side can embed directly
# without recomputing the division.
wdir = os.path.join(OUT, "weights")
os.makedirs(wdir, exist_ok=True)
DECODE_PREFIXES = ("decoder.", "decoder_transformer.", "upsample.", "quantizer.")
sd = model32.state_dict()
n_w = 0
for name, t in sd.items():
    if name.startswith(DECODE_PREFIXES):
        np.save(os.path.join(wdir, name + ".npy"), t.float().cpu().numpy())
        n_w += 1

# Materialize the resolved per-codebook embedding tables (codebook_size×dim).
EPS = 1e-5
n_emb = 0
for rvq in ("semantic_residual_vector_quantizer",
            "acoustic_residual_vector_quantizer"):
    q = getattr(model32.quantizer, rvq)
    for li, layer in enumerate(q.layers):
        cb = layer.codebook
        embed = (cb.embed_sum / cb.cluster_usage.clamp(min=EPS)[:, None]).float()
        np.save(os.path.join(wdir, f"quantizer.{rvq}.layers.{li}.codebook.embed.npy"),
                embed.cpu().numpy())
        n_emb += 1
print(f"  dumped fp32 refs ({list(caps32)}) + {n_w} decoder weights "
      f"+ {n_emb} resolved codebook embeds → weights/")

# Missing-weight sanity: every decode-path tensor name must have been dumped.
expected = [k for k in sd if k.startswith(DECODE_PREFIXES)]
missing = [k for k in expected
           if not os.path.exists(os.path.join(wdir, k + ".npy"))]
assert not missing, f"missing decoder weights: {missing[:8]}"
print(f"  missing decoder weights: {len(missing)}  (expected 0)")

print(f"\nwrote {len(caps)} tensors → {OUT}")
for n, m in manifest["tensors"].items():
    print(f"  {n:26s} {str(m['shape']):22s} mean={m['mean']:+.4f} std={m['std']:.4f}")
