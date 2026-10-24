#!/usr/bin/env python3
"""
ltx2_golden.py -- reference dump for the LTX-2.5 video VAE (M4).

    CUDA_VISIBLE_DEVICES=1 python ltx2_golden.py   # the video VAE decoder alone, fp32

Outputs -> $PIE_IMAGEGEN_GOLDEN/ltx25/ltx2_vae/
    *.f32           a fixed random DiT-space latent clip and the fp32
                    `AutoencoderKLLTX2Video.decode` of it, as raw rows of
                    voxels (see `run_vae`), plus shapes.json

diffusers 0.40's `AutoencoderKLLTX2Video` over the shipped `vae/` folder of
`Lightricks/LTX-2.5-Diffusers` (1.4 GB, in the HuggingFace cache).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golden_common import outdir

MODEL = "ltx25"
REPO = "Lightricks/LTX-2.5-Diffusers"

VAE_LATENT_FRAMES, VAE_LATENT_H, VAE_LATENT_W = 3, 8, 12
VAE_SEED = 7

def run_vae(d: str, device="cuda", shape=(VAE_LATENT_FRAMES, VAE_LATENT_H, VAE_LATENT_W)) -> None:
    """`AutoencoderKLLTX2Video` alone, fp32, over a fixed random latent.

    The DiT works in a NORMALISED latent space and the pipeline undoes that
    before the decode (`_denormalize_latents`: `z * latents_std + latents_mean`,
    `scaling_factor` 1.0); pie's `vae.decode` arm undoes it itself, so what
    is dumped as `latent.f32` is the DiT-space latent — a unit normal, which
    is what a denoised latent in that space looks like — and `denorm.f32` is
    the decoder's own input beside it, for a bisect.

    The decoder is NON-causal (`decoder_causal: False`): `decode` is one call
    over the whole clip, every conv padding its time axis with the clip's own
    first and last frames, and each of the three temporal upsamplers drops
    the first frame after its shuffle, so `F = 8 * (T - 1) + 1`. There is no
    per-frame loop and no cache, so there are no chunk boundaries to dump.
    `timestep_conditioning` is off, so the `temb` argument is not passed.
    """
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        REPO, subfolder="vae", torch_dtype=torch.float32
    ).to(device).eval()
    cfg = vae.config
    assert not cfg.decoder_causal, "this dump states the non-causal decoder"
    assert not cfg.timestep_conditioning, "this dump passes no temb"
    t_lat, h_lat, w_lat = shape

    g = torch.Generator().manual_seed(VAE_SEED)
    z = torch.randn(1, cfg.latent_channels, t_lat, h_lat, w_lat, generator=g).to(
        device=device, dtype=torch.float32
    )
    mean = vae.latents_mean.to(device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = vae.latents_std.to(device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    denorm = z * std / cfg.scaling_factor + mean

    with torch.no_grad():
        x = vae.decode(denorm, return_dict=False)[0]

    frames = int(x.shape[2])
    assert frames == 8 * t_lat - 7, f"{t_lat} latent frames should land {8 * t_lat - 7}, not {frames}"
    assert tuple(x.shape[-2:]) == (32 * h_lat, 32 * w_lat), tuple(x.shape)

    default = shape == (VAE_LATENT_FRAMES, VAE_LATENT_H, VAE_LATENT_W)
    raw = os.path.join(d, "ltx2_vae" if default else f"ltx2_vae_{t_lat}x{h_lat}x{w_lat}")
    os.makedirs(raw, exist_ok=True)
    shapes = {}
    for key, t in (("latent", z[0]), ("denorm", denorm[0]), ("pixels", x[0])):
        cthw = t.detach().float().cpu().numpy()
        rows = np.ascontiguousarray(cthw.transpose(1, 2, 3, 0)).astype("<f4")
        rows.tofile(os.path.join(raw, f"{key}.f32"))
        shapes[key] = {"t": int(cthw.shape[1]), "h": int(cthw.shape[2]),
                       "w": int(cthw.shape[3]), "channels": int(cthw.shape[0])}
    shapes["latents_mean"] = [float(v) for v in vae.latents_mean.float()]
    shapes["latents_std"] = [float(v) for v in vae.latents_std.float()]
    shapes["scaling_factor"] = float(cfg.scaling_factor)
    shapes["decoder_causal"] = bool(cfg.decoder_causal)
    shapes["seed"] = VAE_SEED
    shapes["source"] = f"torch.randn(1, {cfg.latent_channels}, {t_lat}, {h_lat}, {w_lat}) at seed {VAE_SEED}"
    with open(os.path.join(raw, "shapes.json"), "w") as f:
        json.dump(shapes, f, indent=2)

    from PIL import Image
    for k in (0, frames // 2, frames - 1):
        u8 = (np.clip(x[0, :, k].float().cpu().numpy().transpose(1, 2, 0) + 1, 0, 2) * 127.5)
        Image.fromarray(u8.astype("uint8")).save(os.path.join(raw, f"frame{k:03d}.png"))
    print(f"  vae: latent {tuple(z.shape)} -> pixels {tuple(x.shape)} "
          f"[{float(x.min()):.3f}, {float(x.max()):.3f}]")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default=None, help="T,H,W of the latent clip (default 3,8,12)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    shape = tuple(int(v) for v in a.shape.split(",")) if a.shape else (
        VAE_LATENT_FRAMES, VAE_LATENT_H, VAE_LATENT_W)
    run_vae(outdir(MODEL), "cuda" if torch.cuda.is_available() else "cpu", shape)

if __name__ == "__main__":
    main()
