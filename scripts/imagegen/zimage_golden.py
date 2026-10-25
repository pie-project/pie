#!/usr/bin/env python3
"""
zimage_golden.py -- reference dump for Z-Image-Turbo (M1).

Full run (needs ~25 GB VRAM, ~1 min):
    CUDA_VISIBLE_DEVICES=0 python zimage_golden.py --full

Miniature (CPU or GPU, seconds; no weights needed beyond the repo config):
    python zimage_golden.py --mini

VAE only (the FLUX 16-channel AutoencoderKL in fp32, seconds; needs the
zimage_golden.npz of a --full run for its latent, else a seeded one):
    CUDA_VISIBLE_DEVICES=0 python zimage_golden.py --vae

Outputs -> $PIE_IMAGEGEN_GOLDEN/z-image/  (default /root/.cache/pie-imagegen/golden/z-image)
    zimage_golden.npz    prompt embeds (Qwen3 layer -2, unpadded), initial noise,
                         step-0 transformer inputs + velocity, per-step latents, final latent
    zimage_golden.png    decoded 1024x1024 image
    zimage_mini.npz      one random-init forward of a tiny ZImageTransformer2DModel
    zimage_mini.safetensors  the tiny weights (seed 0)
    zimage_vae.npz       --vae: a 64x64 latent (DiT space) and its fp32 decode
                         (512x512, [-1, 1]); that image and its posterior mean
    zimage_vae/*.f32     the same planes as raw little-endian f32 in the
                         row-per-voxel `[h*w, C]` layout pie's voxel axis reads,
                         plus shapes.json — what the Rust parity gate loads

Settings are the official Turbo defaults (Z-Image repo `inference.py`): 8 steps,
guidance 0.0, cfg_normalization False, max_sequence_length 512, shift 3.0 from the
shipped scheduler_config.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from golden_common import (Tap, hook_prepare_latents, hook_scheduler, hook_transformer,
                           manifest, npz_keys, outdir)

REPO = "Tongyi-MAI/Z-Image-Turbo"
MODEL = "z-image"
PROMPT = "a red bicycle leaning on a blue wall"
SEED = 0
STEPS = 8
GUIDANCE = 0.0
SIZE = 1024

MINI_CFG = dict(
    all_patch_size=(2,), all_f_patch_size=(1,),
    in_channels=16, dim=256, n_layers=2, n_refiner_layers=2,
    n_heads=4, n_kv_heads=4, norm_eps=1e-5, qk_norm=True,
    cap_feat_dim=64, siglip_feat_dim=None,
    rope_theta=256.0, t_scale=1000.0,
    axes_dims=[16, 24, 24], axes_lens=[256, 64, 64],
)

def run_full(d: str, dtype=torch.bfloat16):
    from diffusers import ZImagePipeline

    tap = Tap()
    pipe = ZImagePipeline.from_pretrained(REPO, torch_dtype=dtype).to("cuda")
    pipe.set_progress_bar_config(disable=False)

    cfgs = {"transformer": dict(pipe.transformer.config),
            "scheduler": dict(pipe.scheduler.config),
            "vae": dict(pipe.vae.config)}
    with open(os.path.join(d, "zimage_config.json"), "w") as f:
        json.dump(cfgs, f, indent=2, default=str)

    with torch.no_grad():
        embeds = pipe.encode_prompt(PROMPT, device=torch.device("cuda"),
                                    do_classifier_free_guidance=False,
                                    max_sequence_length=512)
    pos = embeds[0] if isinstance(embeds, tuple) else embeds
    tap.put_tree("prompt_embeds", pos)
    lens = [t.shape[0] for t in pos] if isinstance(pos, list) else [pos.shape[-2]]
    tap.put("prompt_embeds.lengths", lens)
    print(f"  prompt embeds: {lens} x {pos[0].shape[-1] if isinstance(pos, list) else pos.shape[-1]}")

    r1 = hook_prepare_latents(pipe, tap)
    r2 = hook_transformer(pipe.transformer, tap, "dit", steps=(0,))
    r3 = hook_scheduler(pipe, tap)
    g = torch.Generator("cpu").manual_seed(SEED)
    out = pipe(prompt=PROMPT, height=SIZE, width=SIZE, num_inference_steps=STEPS,
               guidance_scale=GUIDANCE, generator=g, max_sequence_length=512,
               output_type="pil")
    r3(); r2(); r1()

    tap.put("sigmas", np.asarray(pipe.scheduler.sigmas.float().cpu()))
    tap.put("timesteps", np.asarray(pipe.scheduler.timesteps.float().cpu()))
    final = max((k for k in tap.d if k.startswith("sched.x")), key=lambda k: int(k[7:]))
    tap.d["latent.final"] = tap.d[final]

    img = out.images[0]
    img.save(os.path.join(d, "zimage_golden.png"))
    tap.put("image.rgb", np.asarray(img).astype(np.float32))
    tap.save(os.path.join(d, "zimage_golden.npz"))
    print("  keys:"); npz_keys(tap)

def run_mini(d: str, device="cpu", dtype=torch.float32):
    from diffusers import ZImageTransformer2DModel
    from safetensors.torch import save_file

    torch.manual_seed(0)
    m = ZImageTransformer2DModel(**MINI_CFG).to(device=device, dtype=dtype).eval()
    g = torch.Generator().manual_seed(0)
    for _, p in sorted(m.named_parameters()):
        p.data = (0.02 * torch.randn(p.shape, generator=g)).to(device=device, dtype=dtype)
    sd = {k: v.detach().contiguous().float().cpu() for k, v in m.state_dict().items()}
    save_file(sd, os.path.join(d, "zimage_mini.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(d, "zimage_mini_config.json"), "w") as f:
        json.dump({"config": MINI_CFG,
                   "num_parameters": int(sum(v.numel() for v in sd.values())),
                   "tensors": {k: list(v.shape) for k, v in sorted(sd.items())}}, f, indent=2)

    tap = Tap()
    gi = torch.Generator().manual_seed(1234)
    x = [torch.randn(16, 1, 16, 16, generator=gi).to(device=device, dtype=dtype)]
    cap = [torch.randn(8, MINI_CFG["cap_feat_dim"], generator=gi).to(device=device, dtype=dtype)]
    t = torch.tensor([500.0], device=device, dtype=dtype)
    tap.put_tree("mini.in.x", x); tap.put_tree("mini.in.cap", cap); tap.put("mini.in.t", t)
    with torch.no_grad():
        out = m(x, t, cap, return_dict=False)[0]
    tap.put_tree("mini.out", out)
    tap.save(os.path.join(d, "zimage_mini.npz"))
    print(f"  mini params: {sum(v.numel() for v in sd.values())}")
    npz_keys(tap)

VAE_LATENT = 64

def run_vae(d: str, device="cuda"):
    """The VAE alone, fp32 (`force_upcast`), on one 64x64 latent.

    The latent is the centre crop of `--full`'s final latent (a real one, so
    the decode is a real picture) or, without that file, a seeded normal at
    the DiT's scale. `vae.decode` takes `latent / scaling_factor +
    shift_factor`; `vae.encode(...).latent_dist.mean` is what FLUX and
    Z-Image take (never a sample). Every plane also lands as raw f32 in the
    `[h*w, C]` row-per-voxel layout pie's voxel axis reads.
    """
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    cfg = vae.config
    full = os.path.join(d, "zimage_golden.npz")
    n = VAE_LATENT
    if os.path.exists(full):
        z = np.load(full)["latent.final"]
        z = z.reshape(z.shape[-3:])
        h0 = (z.shape[1] - n) // 2
        w0 = (z.shape[2] - n) // 2
        z = z[:, h0:h0 + n, w0:w0 + n]
        source = "centre crop of zimage_golden.npz latent.final"
    else:
        g = torch.Generator().manual_seed(7)
        z = torch.randn(16, n, n, generator=g).numpy()
        source = "seed 7 normal"
    z = torch.from_numpy(np.ascontiguousarray(z)).to(device=device, dtype=torch.float32)[None]
    with torch.no_grad():
        x = vae.decode(z / cfg.scaling_factor + cfg.shift_factor, return_dict=False)[0]
        mean = vae.encode(x, return_dict=False)[0].mean

    tap = Tap()
    tap.put("vae.latent", z[0])
    tap.put("vae.pixels", x[0])
    tap.put("vae.mean", mean[0])
    tap.put("vae.scaling_factor", float(cfg.scaling_factor))
    tap.put("vae.shift_factor", float(cfg.shift_factor))
    tap.save(os.path.join(d, "zimage_vae.npz"))

    raw = os.path.join(d, "zimage_vae")
    os.makedirs(raw, exist_ok=True)
    shapes = {}
    for key, t in (("latent", z[0]), ("pixels", x[0]), ("mean", mean[0])):
        chw = t.detach().float().cpu().numpy()
        hwc = np.ascontiguousarray(chw.transpose(1, 2, 0)).astype("<f4")
        hwc.tofile(os.path.join(raw, f"{key}.f32"))
        shapes[key] = {"t": 1, "h": int(chw.shape[1]), "w": int(chw.shape[2]), "channels": int(chw.shape[0])}
    shapes["scaling_factor"] = float(cfg.scaling_factor)
    shapes["shift_factor"] = float(cfg.shift_factor)
    shapes["source"] = source
    with open(os.path.join(raw, "shapes.json"), "w") as f:
        json.dump(shapes, f, indent=2)
    back = (mean - (z / cfg.scaling_factor + cfg.shift_factor)).abs().max()
    print(f"  vae: latent {tuple(z.shape)} ({source}) -> pixels {tuple(x.shape)} "
          f"[{float(x.min()):.3f}, {float(x.max()):.3f}] -> mean {tuple(mean.shape)}; "
          f"round trip max |mean - decode input| {float(back):.4f}")

def run_mini_pad(d: str, device="cpu", dtype=torch.float32):
    """A second forward of the `--mini` weights whose rows NEED padding: a 12x16
    latent (6x8 = 48 patches -> 64 rows, 16 image pads) and a 40-row caption
    (-> 64 rows, 24 caption pads), at a pipeline-realistic `t = 0.5` (the
    `--mini` case's `t = 500` is the raw transformer argument, x1000 inside).
    Reads `zimage_mini.safetensors` back rather than re-drawing it, so the
    fixture's weights stay the ones already on disk."""
    from diffusers import ZImageTransformer2DModel
    from safetensors.torch import load_file

    m = ZImageTransformer2DModel(**MINI_CFG).to(device=device, dtype=dtype).eval()
    m.load_state_dict(load_file(os.path.join(d, "zimage_mini.safetensors")))
    tap = Tap()
    gi = torch.Generator().manual_seed(4321)
    x = [torch.randn(16, 1, 12, 16, generator=gi).to(device=device, dtype=dtype)]
    cap = [torch.randn(40, MINI_CFG["cap_feat_dim"], generator=gi).to(device=device, dtype=dtype)]
    t = torch.tensor([0.5], device=device, dtype=dtype)
    tap.put_tree("mini_pad.in.x", x); tap.put_tree("mini_pad.in.cap", cap); tap.put("mini_pad.in.t", t)
    with torch.no_grad():
        out = m(x, t, cap, return_dict=False)[0]
    tap.put_tree("mini_pad.out", out)
    tap.save(os.path.join(d, "zimage_mini_pad.npz"))
    npz_keys(tap)

def run_taps(d: str, dtype=torch.bfloat16):
    """The Turbo transformer ALONE (bf16, CUDA) over the step-0 inputs the full
    golden recorded, with every stage tapped — the bisect a failing
    `zimage_parity.py --turbo` needs — twice: `taps.full.*` on the golden's own
    1024x1024 call and `taps.crop.*` on its top-left 256x256 crop (256 image
    rows), so a long-sequence failure reads as one.  Keys per case:
    `in.x.0 [16,1,H,W]`, `in.cap.0 [L,2560]`, `in.t [1]`, `cap.embed [L32,3840]`,
    `cap.refined [L32,3840]` (after the context refiner), `x.embed [N32,3840]`,
    `x.refined [N32,3840]` (after the noise refiner), `layer{i}.out
    [N32+L32,3840]` (after joint block i), `out.0 [16,1,H,W]`."""
    from diffusers import ZImageTransformer2DModel

    src = np.load(os.path.join(d, "zimage_golden.npz"))
    m = ZImageTransformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=dtype).to("cuda").eval()
    tap = Tap()

    def forward(case: str, x: torch.Tensor, cap: torch.Tensor, t: torch.Tensor):
        p = f"taps.{case}."
        tap.put(p + "in.x.0", x); tap.put(p + "in.cap.0", cap); tap.put(p + "in.t", t)
        handles = []

        def on(module, key, pick=lambda o: o):
            def hook(_m, _i, out):
                tap.put(p + key, pick(out)[0] if pick(out).ndim == 3 else pick(out))
            handles.append(module.register_forward_hook(hook))

        on(m.cap_embedder, "cap.embed")
        on(m.context_refiner[-1], "cap.refined")
        on(m.all_x_embedder["2-1"], "x.embed")
        on(m.noise_refiner[-1], "x.refined")
        for i, layer in enumerate(m.layers):
            if case != "full" or i in (0, len(m.layers) // 2, len(m.layers) - 1):
                on(layer, f"layer{i}.out")
        if case == "tiny":
            b0 = m.noise_refiner[0]
            on(b0.attention_norm1, "b0.norm1")
            handles.append(b0.attention.to_out[0].register_forward_pre_hook(
                lambda _m, args: tap.put(p + "b0.attn", args[0][0])))
            on(b0.attention, "b0.out")
            on(b0.attention_norm2, "b0.norm2")
            on(b0.ffn_norm1, "b0.ffn_norm1")
            on(b0.feed_forward, "b0.ffn")
            on(b0.ffn_norm2, "b0.ffn_norm2")
            on(b0, "b0.res2")
        with torch.no_grad():
            out = m([x], t, [cap], return_dict=False)[0]
        for h in handles:
            h.remove()
        tap.put_tree(p + "out", out)

    x = torch.from_numpy(src["dit.step0.in.arg0.0"]).to("cuda", dtype)
    cap = torch.from_numpy(src["dit.step0.in.arg2.0"]).to("cuda", dtype)
    t = torch.from_numpy(src["dit.step0.in.arg1"]).to("cuda", dtype)
    forward("full", x, cap, t)
    forward("crop", x[:, :, :32, :32].contiguous(), cap, t)
    forward("tiny", x[:, :, :8, :16].contiguous(), cap, t)
    tap.save(os.path.join(d, "zimage_taps.npz"))
    npz_keys(tap)

def main():
    global SIZE, PROMPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--vae", action="store_true")
    ap.add_argument("--mini-pad", action="store_true",
                    help="the padded-rows forward of the --mini weights (zimage_mini_pad.npz)")
    ap.add_argument("--taps", action="store_true",
                    help="the Turbo transformer's stages over the step-0 inputs (zimage_taps.npz; CUDA, ~13 GB)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()
    SIZE, PROMPT = a.size, a.prompt
    if not (a.full or a.mini or a.vae or a.mini_pad or a.taps):
        a.full = a.mini = a.vae = a.mini_pad = True
    d = outdir(MODEL)
    torch.set_grad_enabled(False)
    if a.mini:
        print("== mini =="); run_mini(d, a.device)
    if a.mini_pad:
        print("== mini-pad =="); run_mini_pad(d, a.device)
    if a.taps:
        print("== taps =="); run_taps(d)
    if a.full:
        print("== full =="); run_full(d)
    if a.vae:
        print("== vae =="); run_vae(d, "cuda" if torch.cuda.is_available() else "cpu")
    manifest(d, {"repo": REPO, "prompt": PROMPT, "seed": SEED,
                 "steps": STEPS, "guidance": GUIDANCE, "size": SIZE,
                 "mini_config": MINI_CFG})

if __name__ == "__main__":
    main()
