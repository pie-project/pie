#!/usr/bin/env python3
"""
flux2_golden.py -- reference dump for FLUX.2-klein-4B (M2).

    CUDA_VISIBLE_DEVICES=0 python flux2_golden.py --full     # ~14 GB VRAM
    python flux2_golden.py --mini                            # CPU, seconds
    CUDA_VISIBLE_DEVICES=0 python flux2_golden.py --vae      # seconds

Outputs -> $PIE_IMAGEGEN_GOLDEN/flux2/
    flux2_golden.npz   Qwen3 layer-{9,18,27} concat text embeddings + text_ids,
                       the encoder's input ids + key mask, initial (packed) noise +
                       latent_ids, every step's transformer inputs and velocity,
                       per-step latents, final latent, decoded RGB
    flux2_golden.png
    flux2_mini.npz / flux2_mini.safetensors / flux2_mini_config.json
    flux2_vae.npz      --vae: a 32x32 packed latent (DiT space, 128 channels at
                       /16) and its fp32 decode (512x512, [-1, 1]); that image
                       and the normalised posterior mean it encodes back to
    flux2_vae/*.f32    the same planes as raw little-endian f32 in the
                       row-per-voxel `[h*w, C]` layout pie's voxel axis reads,
                       plus shapes.json -- what the Rust parity gate loads

klein-4B is the distilled deployment: 4 steps, `guidance_embeds: false` in its transformer
config, so no guidance embedding is fed.  Text encoder is Qwen3 (not Mistral) and
`text_encoder_out_layers = (9, 18, 27)` -> joint_attention_dim 7680 = 3 x 2560.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from golden_common import (Tap, hook_prepare_latents, hook_scheduler, hook_transformer,
                           manifest, npz_keys, outdir)

REPO = "black-forest-labs/FLUX.2-klein-4B"
MODEL = "flux2"
PROMPT = "a red bicycle leaning on a blue wall"
SEED = 0
STEPS = 4
SIZE = 1024

MINI_CFG = dict(
    patch_size=1, in_channels=128, out_channels=None,
    num_layers=2, num_single_layers=2,
    attention_head_dim=128, num_attention_heads=2,
    joint_attention_dim=192,
    timestep_guidance_channels=256, mlp_ratio=3.0,
    axes_dims_rope=(32, 32, 32, 32), rope_theta=2000, eps=1e-6,
    guidance_embeds=True,
)
MINI_IMG_HW = (8, 8)
MINI_TXT_LEN = 32
MINI_REFS = 2
MINI_REF_HW = (8, 8)

def run_full(d: str, dtype=torch.bfloat16):
    from diffusers import Flux2KleinPipeline

    tap = Tap()
    pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=dtype).to("cuda")
    with open(os.path.join(d, "flux2_config.json"), "w") as f:
        json.dump({"transformer": dict(pipe.transformer.config),
                   "scheduler": dict(pipe.scheduler.config),
                   "vae": dict(pipe.vae.config)}, f, indent=2, default=str)

    with torch.no_grad():
        pe, tids = pipe.encode_prompt(PROMPT, device=torch.device("cuda"),
                                      max_sequence_length=512,
                                      text_encoder_out_layers=(9, 18, 27))
    tap.put("prompt_embeds", pe); tap.put("text_ids", tids)
    print(f"  prompt embeds {tuple(pe.shape)}  text_ids {tuple(tids.shape)}")
    rendered = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    enc = pipe.tokenizer(rendered, return_tensors="pt", padding="max_length",
                         truncation=True, max_length=512)
    tap.put("text.input_ids", enc["input_ids"][0].numpy().astype(np.int64))
    tap.put("text.attention_mask", enc["attention_mask"][0].numpy().astype(np.int64))
    n_real = int(enc["attention_mask"].sum())
    print(f"  {n_real} real tokens: {enc['input_ids'][0][:n_real].tolist()}")

    r1 = hook_prepare_latents(pipe, tap)
    r2 = hook_transformer(pipe.transformer, tap, "dit", steps=tuple(range(STEPS)))
    r3 = hook_scheduler(pipe, tap)
    g = torch.Generator("cpu").manual_seed(SEED)
    out = pipe(prompt=PROMPT, height=SIZE, width=SIZE, num_inference_steps=STEPS,
               generator=g, max_sequence_length=512, output_type="pil")
    r3(); r2(); r1()

    tap.put("sigmas", pipe.scheduler.sigmas.float().cpu())
    tap.put("timesteps", pipe.scheduler.timesteps.float().cpu())
    final = max((k for k in tap.d if k.startswith("sched.x")), key=lambda k: int(k[7:]))
    tap.d["latent.final"] = tap.d[final]

    img = out.images[0]
    img.save(os.path.join(d, "flux2_golden.png"))
    tap.put("image.rgb", np.asarray(img).astype(np.float32))
    tap.save(os.path.join(d, "flux2_golden.npz"))
    npz_keys(tap)

def run_mini(d: str, device="cpu", dtype=torch.float32):
    from diffusers import Flux2Transformer2DModel
    from safetensors.torch import save_file

    torch.manual_seed(0)
    m = Flux2Transformer2DModel(**MINI_CFG).to(device=device, dtype=dtype).eval()
    g = torch.Generator().manual_seed(0)
    for _, p in sorted(m.named_parameters()):
        p.data = (0.02 * torch.randn(p.shape, generator=g)).to(device=device, dtype=dtype)
    sd = {k: v.detach().contiguous().float().cpu() for k, v in m.state_dict().items()}
    save_file(sd, os.path.join(d, "flux2_mini.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(d, "flux2_mini_config.json"), "w") as f:
        json.dump({"config": {k: list(v) if isinstance(v, tuple) else v
                              for k, v in MINI_CFG.items()},
                   "img_hw": MINI_IMG_HW, "txt_len": MINI_TXT_LEN,
                   "refs": MINI_REFS, "ref_hw": MINI_REF_HW,
                   "num_parameters": int(sum(v.numel() for v in sd.values())),
                   "tensors": {k: list(v.shape) for k, v in sorted(sd.items())}}, f, indent=2)

    h, w = MINI_IMG_HW
    rh, rw = MINI_REF_HW
    si, sr, st = h * w, rh * rw, MINI_TXT_LEN
    gi = torch.Generator().manual_seed(1234)
    B = 1

    def ids(t, hh, ww, ll=1):
        return torch.cartesian_prod(torch.arange(t, t + 1), torch.arange(hh),
                                    torch.arange(ww), torch.arange(ll)).float()

    img_ids = ids(0, h, w)
    ref_ids = torch.cat([ids(10 * (i + 1), rh, rw) for i in range(MINI_REFS)], 0)
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1),
                                   torch.arange(1), torch.arange(st)).float()
    all_img_ids = torch.cat([img_ids, ref_ids], 0)[None].expand(B, -1, -1)
    txt_ids = txt_ids[None].expand(B, -1, -1)

    hs = torch.randn(B, si + MINI_REFS * sr, MINI_CFG["in_channels"], generator=gi)
    ctx = torch.randn(B, st, MINI_CFG["joint_attention_dim"], generator=gi)
    ts = torch.tensor([0.5])
    gd = torch.tensor([4.0])

    kw = dict(hidden_states=hs.to(device, dtype), encoder_hidden_states=ctx.to(device, dtype),
              timestep=ts.to(device, dtype), img_ids=all_img_ids.to(device, dtype),
              txt_ids=txt_ids.to(device, dtype), guidance=gd.to(device, dtype),
              num_ref_tokens=MINI_REFS * sr, return_dict=False)
    tap = Tap()
    for k, v in kw.items():
        if isinstance(v, torch.Tensor):
            tap.put("mini.in." + k, v)
    tap.put("mini.in.num_ref_tokens", MINI_REFS * sr)
    with torch.no_grad():
        out = m(**kw)
    tap.put_tree("mini.out", out)
    tap.save(os.path.join(d, "flux2_mini.npz"))
    print(f"  mini params: {sum(v.numel() for v in sd.values())}")
    npz_keys(tap)

VAE_TOKENS = 32

def run_vae(d: str, device="cuda"):
    """The autoencoder alone, fp32 (`force_upcast`), on one 32x32 token grid.

    `AutoencoderKLFlux2` codes at 32 channels on a /8 grid; the pipeline packs a
    2x2 block of those into one 128-channel cell at /16 and normalises THAT by
    the frozen `bn` (`_patchify_latents`, then
    `(x - running_mean)/sqrt(running_var + eps)`), which is what the transformer
    holds.  So this dump cuts at the 128-wide /16 grid on both sides -- the same
    boundary `models::flux_2::vae` puts its port at:

        decode: latent * bn_std + bn_mean -> _unpatchify_latents -> vae.decode
        encode: vae.encode(...).mean -> _patchify_latents -> (x - bn_mean)/bn_std

    The latent is the centre crop of `--full`'s final packed latent (a real one,
    so the decode is a real picture) or, without that file, a seeded normal.
    Every plane also lands as raw f32 in the `[h*w, C]` row-per-voxel layout
    pie's voxel axis reads.
    """
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    vae = (AutoencoderKLFlux2.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32)
           .to(device).eval())
    cfg = vae.config
    ch = cfg.latent_channels * cfg.patch_size[0] * cfg.patch_size[1]
    n = VAE_TOKENS
    full = os.path.join(d, "flux2_golden.npz")
    if os.path.exists(full):
        z = np.load(full)["latent.final"]
        z = z.reshape(z.shape[-2], z.shape[-1])
        side = int(round(z.shape[0] ** 0.5))
        assert side * side == z.shape[0], f"{z.shape[0]} packed tokens are not a square"
        z = z.reshape(side, side, ch)
        h0 = (side - n) // 2
        z = z[h0:h0 + n, h0:h0 + n]
        z = np.ascontiguousarray(z.transpose(2, 0, 1))
        source = f"centre {n}x{n} crop of flux2_golden.npz latent.final ({side}x{side})"
    else:
        g = torch.Generator().manual_seed(7)
        z = torch.randn(ch, n, n, generator=g).numpy()
        source = "seed 7 normal"
    z = torch.from_numpy(np.ascontiguousarray(z)).to(device=device, dtype=torch.float32)[None]

    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, torch.float32)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + cfg.batch_norm_eps).to(
        device, torch.float32)
    unpatchify = Flux2KleinPipeline._unpatchify_latents
    patchify = Flux2KleinPipeline._patchify_latents

    with torch.no_grad():
        x = vae.decode(unpatchify(z * bn_std + bn_mean), return_dict=False)[0]
        mean = vae.encode(x, return_dict=False)[0].mean
        packed = (patchify(mean) - bn_mean) / bn_std

    tap = Tap()
    tap.put("vae.latent", z[0])
    tap.put("vae.pixels", x[0])
    tap.put("vae.mean", packed[0])
    tap.put("vae.bn_running_mean", vae.bn.running_mean)
    tap.put("vae.bn_running_var", vae.bn.running_var)
    tap.put("vae.batch_norm_eps", float(cfg.batch_norm_eps))
    tap.save(os.path.join(d, "flux2_vae.npz"))

    raw = os.path.join(d, "flux2_vae")
    os.makedirs(raw, exist_ok=True)
    shapes = {}
    for key, t in (("latent", z[0]), ("pixels", x[0]), ("mean", packed[0])):
        chw = t.detach().float().cpu().numpy()
        hwc = np.ascontiguousarray(chw.transpose(1, 2, 0)).astype("<f4")
        hwc.tofile(os.path.join(raw, f"{key}.f32"))
        shapes[key] = {"t": 1, "h": int(chw.shape[1]), "w": int(chw.shape[2]),
                       "channels": int(chw.shape[0])}
    shapes["batch_norm_eps"] = float(cfg.batch_norm_eps)
    shapes["latent_channels"] = int(cfg.latent_channels)
    shapes["patch_size"] = list(cfg.patch_size)
    shapes["source"] = source
    with open(os.path.join(raw, "shapes.json"), "w") as f:
        json.dump(shapes, f, indent=2)
    back = (packed - z).abs().max()
    print(f"  vae: latent {tuple(z.shape)} ({source}) -> pixels {tuple(x.shape)} "
          f"[{float(x.min()):.3f}, {float(x.max()):.3f}] -> mean {tuple(packed.shape)}; "
          f"round trip max |mean - latent| {float(back):.4f}")
    npz_keys(tap)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--vae", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if not (a.full or a.mini or a.vae):
        a.full = a.mini = True
    d = outdir(MODEL)
    torch.set_grad_enabled(False)
    if a.mini:
        print("== mini =="); run_mini(d, a.device)
    if a.full:
        print("== full =="); run_full(d)
    if a.vae:
        print("== vae =="); run_vae(d, "cuda" if torch.cuda.is_available() else "cpu")
    manifest(d, {"repo": REPO, "prompt": PROMPT, "seed": SEED, "steps": STEPS, "size": SIZE})

if __name__ == "__main__":
    main()
