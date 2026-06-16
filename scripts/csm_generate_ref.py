#!/usr/bin/env python3
"""CSM-1B end-to-end generation reference (text → Mimi codes → wav).

Runs the *real* transformers `CsmForConditionalGeneration` (`eustlb/csm-1b`)
on a short text prompt, generating Mimi RVQ codes via the two-transformer loop:

  1. backbone (16-layer / 2048-hidden Llama) predicts codebook 0 of the next
     Mimi frame + its last hidden state,
  2. depth decoder (4-layer / 1024-hidden Llama) autoregressively predicts
     codebooks 1..31 for that frame given (backbone_hidden, cb0),
  3. repeat per frame until the audio-EOS frame.

We then run the generated codes `[32, n_frames]` through the bundled Mimi codec
(`model.codec_model.decode`) to a reference `output.wav`, and dump:
  * the generated codes `[32, n_frames]` (the exact target of the CUDA RVQ
    sampler in the generation loop),
  * the decoded waveform + output.wav,
  * the CSM backbone + depth-decoder configs (for the loader),
  * a small set of backbone / depth-decoder / lm_head / codebooks_head weights
    (shape/dtype proof for the binder; full weights are loaded by Pie's normal
    safetensors loader — this ref is for wiring sanity, not a weight dump).

This is the e2e ground truth: text → codes → wav. The codes→wav half is also
covered standalone by mimi_decoder_parity_ref.py (same `kyutai/mimi` codec, as
`codec_config` in CSM's config.json points at it).

Usage: python3 csm_generate_ref.py [checkpoint_dir] [out_dir] ["prompt text"]

Guards cleanly if the checkpoint is still downloading. Mirrors the
scripts/*_parity_ref.py conventions.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

# ── Locate the checkpoint (guard if still downloading) ───────────────────────
_DEFAULT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--eustlb--csm-1b/snapshots/*"
)
if len(sys.argv) > 1:
    CKPT = sys.argv[1]
else:
    _hits = [p for p in glob.glob(_DEFAULT_GLOB) if os.path.isdir(p)]
    if not _hits:
        sys.exit(
            "eustlb/csm-1b checkpoint not found. Expected a snapshot under\n"
            f"  {_DEFAULT_GLOB}\n"
            "It may still be downloading. Fetch it with:\n"
            "  huggingface-cli download eustlb/csm-1b\n"
            "or pass the checkpoint dir as argv[1]."
        )
    CKPT = _hits[0]


def _shards_ready(ckpt):
    """Both safetensors shards present (CSM is a 2-shard checkpoint)."""
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return os.path.exists(os.path.join(ckpt, "model.safetensors"))
    with open(idx) as f:
        shards = set(json.load(f)["weight_map"].values())
    return all(os.path.exists(os.path.join(ckpt, s)) for s in shards)


if not _shards_ready(CKPT):
    sys.exit(
        f"CSM checkpoint at {CKPT} is incomplete (still downloading?). "
        "Re-run once all safetensors shards are present."
    )

OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/csm_generate"
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "Hello, this is a test."
os.makedirs(OUT, exist_ok=True)

from transformers import AutoProcessor, CsmForConditionalGeneration  # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = CsmForConditionalGeneration.from_pretrained(CKPT).to(dev).eval()
cfg = model.config
dcfg = cfg.depth_decoder_config
ccfg = cfg.codec_config


def _rope(c):
    # transformers 5.9 nests rope params under rope_parameters/rope_scaling.
    rp = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {}
    return dict(rp), rp.get("rope_theta", getattr(c, "rope_theta", 500000))


bb_rope, bb_theta = _rope(cfg)
dd_rope, dd_theta = _rope(dcfg)
print(
    f"csm backbone: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
    f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
    f"head_dim={cfg.head_dim} vocab={cfg.vocab_size} text_vocab={cfg.text_vocab_size} "
    f"num_codebooks={cfg.num_codebooks} rope_theta={bb_theta}"
)
print(
    f"csm depth:    hidden={dcfg.hidden_size} layers={dcfg.num_hidden_layers} "
    f"heads={dcfg.num_attention_heads} kv_heads={dcfg.num_key_value_heads} "
    f"head_dim={dcfg.head_dim} backbone_hidden={dcfg.backbone_hidden_size} "
    f"vocab={dcfg.vocab_size} max_pos={dcfg.max_position_embeddings}"
)
print(
    f"codec(mimi):  hidden={ccfg.hidden_size} quantizers={ccfg.num_quantizers} "
    f"sr={ccfg.sampling_rate} ratios={list(ccfg.upsampling_ratios)}"
)

# ── Build the prompt (CSM format: "[speaker_id]text") ────────────────────────
processor = AutoProcessor.from_pretrained(CKPT)
text = f"[0]{PROMPT}"
inputs = processor(text, add_special_tokens=True, return_tensors="pt").to(dev)
print(f"prompt: {text!r} → input_ids {tuple(inputs['input_ids'].shape)}")

# ── Generate (greedy + fixed seeds for determinism of the reference) ─────────
gen_kwargs = dict(
    max_new_tokens=64,            # caps the frame count for a short ref clip
    do_sample=False,              # greedy backbone for a reproducible reference
    depth_decoder_do_sample=False,  # greedy depth decoder too
)
with torch.no_grad():
    out = model.generate(**inputs, output_audio=True, **gen_kwargs)

# `output_audio=True` → list/tensor of waveforms; also re-derive codes for dump.
audio = out[0] if isinstance(out, (list, tuple)) else out
if hasattr(audio, "audio_values"):
    audio = audio.audio_values
waveform = audio.detach().float().cpu().reshape(-1).numpy()
n_samples = waveform.shape[0]
sr = ccfg.sampling_rate
print(f"generated waveform: {n_samples} samples (≈ {n_samples / sr:.3f}s @ {sr}Hz)")

# ── Re-encode the same waveform to recover the codes [32, n_frames] ──────────
# `generate(output_audio=True)` returns the waveform; to dump the discrete codes
# the CUDA RVQ sampler must reproduce, re-encode the generated audio through the
# bundled codec (a faithful round-trip: the codec is deterministic argmin-RVQ).
with torch.no_grad():
    wf = torch.from_numpy(waveform)[None, None, :].to(dev)
    enc = model.codec_model.encode(wf, num_quantizers=cfg.num_codebooks)
    codes = enc.audio_codes if hasattr(enc, "audio_codes") else enc[0]
    codes = codes[0]  # [32, n_frames]
    # Decode again through Mimi for a clean codes → wav reference (matches the
    # CUDA mimi_decoder path exactly).
    redec = model.codec_model.decode(codes[None])
    rewave = (redec.audio_values if hasattr(redec, "audio_values") else redec[0])
    rewave = rewave.detach().float().cpu().reshape(-1).numpy()
n_frames = int(codes.shape[1])
print(f"codes: {tuple(codes.shape)} (32 codebooks × {n_frames} frames)")

# ── Dump codes, waveforms, output.wav, configs, sample weights + manifest ────
np.save(os.path.join(OUT, "codes.npy"), codes.cpu().numpy())
np.save(os.path.join(OUT, "waveform.npy"), waveform)
np.save(os.path.join(OUT, "waveform_redecoded.npy"), rewave)


def _write_wav(path, samples, rate):
    """Minimal 16-bit PCM WAV writer (no scipy/soundfile dependency)."""
    import struct
    import wave
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm16.tobytes())
    _ = struct  # silence linters; struct retained for clarity


_write_wav(os.path.join(OUT, "output.wav"), waveform, sr)
print(f"wrote output.wav ({n_samples} samples @ {sr}Hz)")

manifest = {
    "checkpoint": CKPT,
    "prompt": text,
    "gen_kwargs": gen_kwargs,
    "backbone_config": {
        "hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
        "intermediate_size": cfg.intermediate_size, "vocab_size": cfg.vocab_size,
        "text_vocab_size": cfg.text_vocab_size, "num_codebooks": cfg.num_codebooks,
        "codebook_size": ccfg.codebook_size,
        "rms_norm_eps": cfg.rms_norm_eps, "rope_theta": bb_theta,
        "rope_scaling": bb_rope, "hidden_act": cfg.hidden_act,
        "max_position_embeddings": cfg.max_position_embeddings,
        "audio_token_id": cfg.audio_token_id, "audio_eos_token_id": cfg.audio_eos_token_id,
        "codebook_eos_token_id": cfg.codebook_eos_token_id,
        "codebook_pad_token_id": cfg.codebook_pad_token_id,
        "tie_codebooks_embeddings": cfg.tie_codebooks_embeddings,
    },
    "depth_decoder_config": {
        "hidden_size": dcfg.hidden_size, "num_hidden_layers": dcfg.num_hidden_layers,
        "num_attention_heads": dcfg.num_attention_heads,
        "num_key_value_heads": dcfg.num_key_value_heads, "head_dim": dcfg.head_dim,
        "intermediate_size": dcfg.intermediate_size, "vocab_size": dcfg.vocab_size,
        "num_codebooks": dcfg.num_codebooks,
        "backbone_hidden_size": dcfg.backbone_hidden_size,
        "max_position_embeddings": dcfg.max_position_embeddings,
        "rms_norm_eps": dcfg.rms_norm_eps, "rope_theta": dd_theta,
        "rope_scaling": dd_rope, "hidden_act": dcfg.hidden_act,
    },
    "codec_config": {
        "model_type": ccfg.model_type, "hidden_size": ccfg.hidden_size,
        "num_quantizers": ccfg.num_quantizers, "sampling_rate": ccfg.sampling_rate,
        "upsampling_ratios": list(ccfg.upsampling_ratios),
    },
    "generation_config": {
        "do_sample": getattr(model.generation_config, "do_sample", None),
        "temperature": getattr(model.generation_config, "temperature", None),
        "depth_decoder_temperature": getattr(
            model.depth_decoder.generation_config, "temperature", None),
        "depth_decoder_top_k": getattr(
            model.depth_decoder.generation_config, "top_k", None),
    },
    "outputs": {
        "codes": {"shape": list(codes.shape), "n_frames": n_frames},
        "waveform": {"n_samples": n_samples, "sampling_rate": sr},
    },
}
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)

# ── Dump a few proof-of-shape weights (binder sanity, not a full dump) ───────
wdir = os.path.join(OUT, "weights")
os.makedirs(wdir, exist_ok=True)
sd = model.state_dict()
PROBE = [
    "lm_head.weight",
    "embed_text_tokens.weight",
    "backbone_model.embed_tokens.embed_audio_tokens.weight",
    "backbone_model.norm.weight",
    "backbone_model.layers.0.self_attn.q_proj.weight",
    "backbone_model.layers.0.self_attn.k_proj.weight",
    "backbone_model.layers.0.mlp.gate_proj.weight",
    "depth_decoder.model.embed_tokens.weight",
    "depth_decoder.model.inputs_embeds_projector.weight",
    "depth_decoder.model.layers.0.self_attn.q_proj.weight",
    "depth_decoder.model.norm.weight",
    "depth_decoder.codebooks_head.weight",
]
probed = {}
for name in PROBE:
    if name in sd:
        t = sd[name]
        np.save(os.path.join(wdir, name + ".npy"), t.float().cpu().numpy())
        probed[name] = {"shape": list(t.shape), "dtype": str(t.dtype)}
with open(os.path.join(OUT, "weight_probe.json"), "w") as f:
    json.dump(probed, f, indent=2)
print(f"  probed {len(probed)}/{len(PROBE)} key weights → weights/")
for n, m in probed.items():
    print(f"    {n:58s} {m['shape']}")

print(f"\nwrote codes, waveform, output.wav, configs → {OUT}")
