#!/usr/bin/env python3
"""Gemma-4 audio-tower parity reference.

Runs the *real* transformers `Gemma4AudioModel` + the `Gemma4AudioFeatureExtractor`
+ the shared `Gemma4MultimodalEmbedder` (loaded from the local `google/gemma-4-E4B`
checkpoint) on a deterministic synthetic waveform, and dumps the intermediate
activations the CUDA `gemma4_audio` encoder (MULTIMODAL.md Phase 5, audio) must
reproduce.

The audio tower (`audio_config`, model_type `gemma4_audio`) is a USM/Conformer
encoder: a SSCP subsampling conv stack (2× Conv2d stride-2 → 4× time downsample,
freq 128→32) + an `input_proj_linear` → hidden 1024, then 12 Conformer blocks
(½·FFN → chunked-local logit-capped MHSA with relative-position bias → light
depthwise-conv module with GLU → ½·FFN → RMSNorm, residual_weight 0.5), then
`output_proj` (1024→1536, +bias), then the shared embedder (parameterless
RMSNorm → projection 1536→2560).

Encoder parity is input-agnostic: we feed identical `input_features` (log-mel
`[n_frames, 128]`) to both torch and the CUDA driver, so a synthetic-but-valid
waveform is sufficient. We dump the full chain so each CUDA stage is checkable.

Outputs → OUT_DIR as .npy + manifest.json (shape/dtype/mean/std/first values).
Usage: python3 gemma4_audio_parity_ref.py [checkpoint_dir] [out_dir]

Mirrors scripts/gemma4_vision_parity_ref.py conventions.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

# ── Locate the checkpoint (guard with a clear message if absent) ─────────────
_DEFAULT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/*"
)
if len(sys.argv) > 1:
    CKPT = sys.argv[1]
else:
    _hits = glob.glob(_DEFAULT_GLOB)
    if not _hits:
        sys.exit(
            "gemma-4-E4B checkpoint not found. Expected a snapshot under\n"
            f"  {_DEFAULT_GLOB}\n"
            "Download it first:\n"
            "  huggingface-cli download google/gemma-4-E4B\n"
            "or pass the checkpoint dir as argv[1]."
        )
    CKPT = _hits[0]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gemma4_audio_parity"
os.makedirs(OUT, exist_ok=True)

from transformers import AutoConfig
from transformers.models.gemma4.feature_extraction_gemma4 import (
    Gemma4AudioFeatureExtractor,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel,
    Gemma4MultimodalEmbedder,
)
from safetensors import safe_open

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

cfg = AutoConfig.from_pretrained(CKPT)
acfg, tcfg = cfg.audio_config, cfg.text_config
print(f"audio: hidden={acfg.hidden_size} layers={acfg.num_hidden_layers} "
      f"heads={acfg.num_attention_heads} conv_k={acfg.conv_kernel_size} "
      f"sscp={acfg.subsampling_conv_channels} chunk={acfg.attention_chunk_size} "
      f"ctx_left={acfg.attention_context_left} ctx_right={acfg.attention_context_right} "
      f"logit_cap={acfg.attention_logit_cap} out_proj={acfg.output_proj_dims} "
      f"residual_w={acfg.residual_weight} act={acfg.hidden_act}")
print(f"tokens: audio_token_id={cfg.audio_token_id} boa={cfg.boa_token_id} "
      f"eoa={cfg.eoa_token_id} text_hidden={tcfg.hidden_size}")

# ── Build the audio tower + embedder and load their weights ──────────────────
audio = Gemma4AudioModel(acfg).to(dev, dtype).eval()
embed = Gemma4MultimodalEmbedder(acfg, tcfg).to(dev, dtype).eval()

sf = os.path.join(CKPT, "model.safetensors")
at_sd, ea_sd = {}, {}
AT_PREFIX, EA_PREFIX = "model.audio_tower.", "model.embed_audio."
with safe_open(sf, framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.startswith(AT_PREFIX):
            at_sd[k[len(AT_PREFIX):]] = f.get_tensor(k).to(dtype)
        elif k.startswith(EA_PREFIX):
            ea_sd[k[len(EA_PREFIX):]] = f.get_tensor(k).to(dtype)
ma = audio.load_state_dict(at_sd, strict=False)
me = embed.load_state_dict(ea_sd, strict=False)
print(f"audio load: {len(at_sd)} tensors, missing={len(ma.missing_keys)} "
      f"unexpected={len(ma.unexpected_keys)}")
print(f"embed load: {len(ea_sd)} tensors, missing={len(me.missing_keys)} "
      f"unexpected={len(me.unexpected_keys)}")
# Non-buffer missing keys are a real problem; clip-range buffers are fine.
real_missing = [k for k in ma.missing_keys if k.endswith(".weight") or k.endswith(".bias")]
assert not real_missing, f"missing audio weights: {real_missing[:8]}"
assert not me.missing_keys, f"missing embed weights: {me.missing_keys[:8]}"

# ── Deterministic synthetic waveform: ~2s sine chirp at 16kHz ────────────────
SR = acfg_sr = 16000
DUR_S = 2.0
n_samples = int(SR * DUR_S)
t = np.arange(n_samples, dtype=np.float64) / SR
# Linear chirp 200 Hz → 2000 Hz, amplitude 0.5 (deterministic, no noise).
f0, f1 = 200.0, 2000.0
inst_freq = f0 + (f1 - f0) * t / DUR_S
phase = 2.0 * np.pi * np.cumsum(inst_freq) / SR
waveform = (0.5 * np.sin(phase)).astype(np.float32)

# ── Feature extractor → log-mel features [n_frames, 128] ─────────────────────
fe = Gemma4AudioFeatureExtractor.from_dict(
    cfg.to_dict().get("feature_extractor", {})
) if False else None
# from_pretrained pulls processor_config.json → feature_extractor block.
from transformers import AutoProcessor
try:
    proc = AutoProcessor.from_pretrained(CKPT)
    fe = proc.feature_extractor
except Exception as e:  # noqa: BLE001
    print(f"WARN: AutoProcessor failed ({e}); building feature extractor from defaults")
    fe = Gemma4AudioFeatureExtractor()

# Pass as a list of 1-D arrays (a batch of one) — the extractor's batch-of-one
# code path; a bare 1-D array trips its internal squeeze(0).
feat = fe([waveform], sampling_rate=SR, return_tensors="pt")
input_features = feat["input_features"].to(dev, dtype)          # [1, n_frames, 128]
input_features_mask = feat["input_features_mask"].to(dev)       # [1, n_frames] bool
n_frames = int(input_features.shape[1])
print(f"waveform: {n_samples} samples ({DUR_S}s @ {SR}Hz) → "
      f"log-mel {tuple(input_features.shape)} mask_valid={int(input_features_mask.sum())}")

# ── Hooks to capture stage outputs ───────────────────────────────────────────
caps = {}


def _mk_hook(name):
    def hook(_m, _i, o):
        out = o[0] if isinstance(o, tuple) else o
        caps[name] = out.detach()
    return hook


# Subsampling output (after SSCP conv stack + input_proj_linear).
audio.subsample_conv_projection.register_forward_hook(_mk_hook("sscp_out"))
# A few Conformer layers.
LAYER_TAPS = sorted(set([0, 5, acfg.num_hidden_layers - 1]))
for li in LAYER_TAPS:
    audio.layers[li].register_forward_hook(_mk_hook(f"layer{li}"))

with torch.no_grad():
    aout = audio(input_features=input_features, attention_mask=input_features_mask)
    enc = aout.last_hidden_state            # [1, n_audio_tok, 1536] (after output_proj)
    enc_2d = enc[0]                         # [n_audio_tok, 1536]
    projected = embed(inputs_embeds=enc_2d)  # [n_audio_tok, 2560]

n_audio_tok = int(enc_2d.shape[0])
print(f"encoder out: {tuple(enc.shape)} → n_audio_tok={n_audio_tok} → "
      f"projected {tuple(projected.shape)}")

caps["encoder_out"] = enc_2d.detach()
caps["projected"] = projected.detach()
caps["input_waveform"] = torch.from_numpy(waveform)
caps["input_features"] = input_features.detach()
caps["input_features_mask"] = input_features_mask.detach()

# ── Dump ─────────────────────────────────────────────────────────────────────
manifest = {
    "checkpoint": CKPT,
    "config": {
        "hidden": acfg.hidden_size, "layers": acfg.num_hidden_layers,
        "heads": acfg.num_attention_heads, "conv_kernel": acfg.conv_kernel_size,
        "sscp_channels": list(acfg.subsampling_conv_channels),
        "chunk_size": acfg.attention_chunk_size,
        "context_left": acfg.attention_context_left,
        "context_right": acfg.attention_context_right,
        "logit_cap": acfg.attention_logit_cap,
        "output_proj_dims": acfg.output_proj_dims,
        "residual_weight": acfg.residual_weight,
        "rms_norm_eps": acfg.rms_norm_eps, "hidden_act": acfg.hidden_act,
        "feature_size": fe.feature_size, "fft_length": fe.fft_length,
        "frame_length": fe.frame_length, "hop_length": fe.hop_length,
        "sampling_rate": fe.sampling_rate,
        "text_hidden": tcfg.hidden_size, "audio_ms_per_token": 40,
        "n_frames": n_frames, "n_audio_tok": n_audio_tok,
        "audio_token_id": cfg.audio_token_id, "boa_token_id": cfg.boa_token_id,
        "eoa_token_id": cfg.eoa_token_id,
    },
    "tensors": {},
}
for name, tns in caps.items():
    arr = tns.float().cpu().numpy() if tns.dtype.is_floating_point else tns.cpu().numpy()
    np.save(os.path.join(OUT, name + ".npy"), arr)
    flat = arr.reshape(-1).astype(np.float64)
    manifest["tensors"][name] = {
        "shape": list(arr.shape), "dtype": str(tns.dtype),
        "mean": float(flat.mean()) if flat.size else 0.0,
        "std": float(flat.std()) if flat.size else 0.0,
        "first8": [float(x) for x in flat[:8]],
    }
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# ── fp32 reference run, for a tight standalone CUDA check + all weights ───────
audio.float()
embed.float()
caps32 = {}
audio.subsample_conv_projection.register_forward_hook(
    lambda _m, _i, o: caps32.__setitem__(
        "sscp_out", (o[0] if isinstance(o, tuple) else o).detach()))
for li in LAYER_TAPS:
    audio.layers[li].register_forward_hook(
        (lambda nm: (lambda _m, _i, o: caps32.__setitem__(
            nm, (o[0] if isinstance(o, tuple) else o).detach())))(f"layer{li}"))
with torch.no_grad():
    aout32 = audio(input_features=input_features.float(),
                   attention_mask=input_features_mask)
    enc32 = aout32.last_hidden_state[0]
    caps32["encoder_out"] = enc32.detach()
    caps32["projected"] = embed(inputs_embeds=enc32).detach()
for name, tns in caps32.items():
    np.save(os.path.join(OUT, name + "_f32.npy"), tns.cpu().numpy())
np.save(os.path.join(OUT, "input_features_f32.npy"),
        input_features.float().cpu().numpy())

wdir = os.path.join(OUT, "weights")
os.makedirs(wdir, exist_ok=True)
allw = {"audio." + k: v for k, v in audio.state_dict().items()}
allw.update({"embed." + k: v for k, v in embed.state_dict().items()})
for name, tns in allw.items():
    np.save(os.path.join(wdir, name + ".npy"), tns.float().cpu().numpy())
print(f"  dumped fp32 refs ({list(caps32)}) + {len(allw)} weights → weights/")

print(f"\nwrote {len(caps)} tensors → {OUT}")
for n, m in manifest["tensors"].items():
    print(f"  {n:22s} {str(m['shape']):20s} mean={m['mean']:+.4f} std={m['std']:.4f}")
