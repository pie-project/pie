#!/usr/bin/env python3
"""Gemma-4 vision-tower parity reference.

Runs the *real* transformers `Gemma4VisionModel` + `Gemma4MultimodalEmbedder`
(loaded from the local `google/gemma-4-E4B` checkpoint) on a deterministic
synthetic input, and dumps intermediate activations the CUDA `gemma4_vision`
encoder (MULTIMODAL.md Phase 2.2) must reproduce.

Encoder parity is input-agnostic: we feed identical (pixel_values,
pixel_position_ids) to both torch and the CUDA driver, so a synthetic-but-valid
input is sufficient and avoids a dependency on the image processor (validated
separately in 2.3).

Outputs → OUT_DIR as .npy + manifest.json (shape/dtype/mean/std/first values).
Usage: python3 gemma4_vision_parity_ref.py [checkpoint_dir] [out_dir]
"""
import glob
import json
import os
import struct
import sys

import numpy as np
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else glob.glob(
    os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/*")
)[0]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gemma4_vision_parity"
os.makedirs(OUT, exist_ok=True)

from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4VisionModel,
    Gemma4MultimodalEmbedder,
)
from safetensors import safe_open

torch.manual_seed(0)
dev = "cuda"
dtype = torch.bfloat16

cfg = AutoConfig.from_pretrained(CKPT)
vcfg, tcfg = cfg.vision_config, cfg.text_config
print(f"vision: hidden={vcfg.hidden_size} layers={vcfg.num_hidden_layers} "
      f"patch={vcfg.patch_size} pool_k={vcfg.pooling_kernel_size} "
      f"soft_tokens={cfg.vision_soft_tokens_per_image}")

# ── Build the two vision submodules and load their weights ──────────────────
vision = Gemma4VisionModel(vcfg).to(dev, dtype).eval()
embed = Gemma4MultimodalEmbedder(vcfg, tcfg).to(dev, dtype).eval()

sf = os.path.join(CKPT, "model.safetensors")
vt_sd, ev_sd = {}, {}
with safe_open(sf, framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.startswith("model.vision_tower."):
            vt_sd[k[len("model.vision_tower."):]] = f.get_tensor(k).to(dtype)
        elif k.startswith("model.embed_vision."):
            ev_sd[k[len("model.embed_vision."):]] = f.get_tensor(k).to(dtype)
mv = vision.load_state_dict(vt_sd, strict=False)
me = embed.load_state_dict(ev_sd, strict=False)
print(f"vision load: {len(vt_sd)} tensors, missing={len(mv.missing_keys)} "
      f"unexpected={len(mv.unexpected_keys)}")
print(f"embed  load: {len(ev_sd)} tensors, missing={len(me.missing_keys)} "
      f"unexpected={len(me.unexpected_keys)}")
# Non-buffer missing keys are a real problem; clip-range buffers are fine.
real_missing = [k for k in mv.missing_keys if k.endswith(".weight")]
assert not real_missing, f"missing vision weights: {real_missing[:5]}"

# ── Deterministic synthetic input ───────────────────────────────────────────
# A 60×42 patch grid → 2520 patches → /(3²) → 280 soft tokens. No padding.
GX, GY, K = 60, 42, vcfg.pooling_kernel_size
n_patch = GX * GY
out_len = n_patch // (K * K)
assert out_len == cfg.vision_soft_tokens_per_image, (out_len, cfg.vision_soft_tokens_per_image)
patch_dim = 3 * vcfg.patch_size ** 2

# pixel_values in [0,1] (patch embedder rescales to [-1,1] internally).
pix = torch.linspace(0, 1, n_patch * patch_dim, device=dev).reshape(1, n_patch, patch_dim).to(dtype)
pos = torch.empty(1, n_patch, 2, dtype=torch.long, device=dev)
for i in range(n_patch):
    pos[0, i, 0] = i % GX  # x
    pos[0, i, 1] = i // GX  # y

# ── Hooks to capture stage outputs ──────────────────────────────────────────
caps = {}
vision.patch_embedder.register_forward_hook(
    lambda m, i, o: caps.__setitem__("patch_embed", o.detach()))
vision.encoder.layers[0].register_forward_hook(
    lambda m, i, o: caps.__setitem__("layer0", (o[0] if isinstance(o, tuple) else o).detach()))
vision.encoder.layers[-1].register_forward_hook(
    lambda m, i, o: caps.__setitem__("layer_last", (o[0] if isinstance(o, tuple) else o).detach()))

with torch.no_grad():
    vout = vision(pixel_values=pix, pixel_position_ids=pos)
    last_hidden = vout.last_hidden_state            # [280, 768] (padding stripped)
    projected = embed(inputs_embeds=last_hidden)    # [280, 2560]

caps["pooled_last_hidden"] = last_hidden.detach()
caps["projected"] = projected.detach()
caps["input_pixel_values"] = pix.detach()
caps["input_position_ids"] = pos.detach()

# ── Dump ────────────────────────────────────────────────────────────────────
manifest = {"config": {"hidden": vcfg.hidden_size, "layers": vcfg.num_hidden_layers,
                        "patch": vcfg.patch_size, "pool_k": K, "soft_tokens": out_len,
                        "grid": [GX, GY], "n_patch": n_patch}, "tensors": {}}
for name, t in caps.items():
    arr = t.float().cpu().numpy()
    np.save(os.path.join(OUT, name + ".npy"), arr)
    flat = arr.reshape(-1)
    manifest["tensors"][name] = {
        "shape": list(arr.shape), "dtype": str(t.dtype),
        "mean": float(flat.mean()), "std": float(flat.std()),
        "first8": [float(x) for x in flat[:8]],
    }
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# ── Full fp32 reference run + all weights, for the standalone CUDA forward.
# Re-run the whole vision tower + projector in fp32 so the single-precision
# CUDA forward can be checked to tight tolerance, and dump every weight. ───────
vision.float()
embed.float()
caps32 = {}
vision.patch_embedder.register_forward_hook(
    lambda m, i, o: caps32.__setitem__("patch_embed", o.detach()))
vision.encoder.layers[0].register_forward_hook(
    lambda m, i, o: caps32.__setitem__("layer0", (o[0] if isinstance(o, tuple) else o).detach()))
vision.encoder.layers[-1].register_forward_hook(
    lambda m, i, o: caps32.__setitem__("layer_last", (o[0] if isinstance(o, tuple) else o).detach()))
with torch.no_grad():
    vout32 = vision(pixel_values=pix.float(), pixel_position_ids=pos)
    caps32["pooled_last_hidden"] = vout32.last_hidden_state.detach()
    caps32["projected"] = embed(inputs_embeds=vout32.last_hidden_state).detach()
for name, t in caps32.items():
    np.save(os.path.join(OUT, name + "_f32.npy"), t.cpu().numpy())
np.save(os.path.join(OUT, "input_pixel_values_f32.npy"), pix.float().cpu().numpy())

wdir = os.path.join(OUT, "weights")
os.makedirs(wdir, exist_ok=True)
allw = {"vision." + k: v for k, v in vision.state_dict().items()}
allw.update({"embed." + k: v for k, v in embed.state_dict().items()})
for name, t in allw.items():
    np.save(os.path.join(wdir, name + ".npy"), t.float().cpu().numpy())
print(f"  dumped fp32 refs ({list(caps32)}) + {len(allw)} weights → weights/")

print(f"\nwrote {len(caps)} tensors → {OUT}")
for n, m in manifest["tensors"].items():
    print(f"  {n:22s} {str(m['shape']):18s} mean={m['mean']:+.4f} std={m['std']:.4f}")
