#!/usr/bin/env python3
"""flux/mk_tiny_flux2.py <out.f32> [--bf16 <out.f32>] : flux2-mini's DiT (models/flux_2.poem's `flux2-mini`:
dim 256, 2 heads of 128, 2 double + 2 single blocks, context 192, guidance embeds) as diffusers'
Flux2Transformer2DModel, its weights and inputs drawn by name from the generator engine-cpu's flux test
draws them from, and the velocity of one denoise step: 64 text rows and a 16 x 16 latent (64 image rows).

The weights are drawn under the poem's param names (`dit.double.0.img.attn.qkv`, ...), rounded to bf16, and
laid into diffusers' state dict as the poem's read rules lay a checkpoint into its params, inverted: q / k / v
fused, the adaLN projections' (shift, scale) pairs swapped, a single block's output projection split in two.
The velocity is diffusers' in f32 (bf16 weights and inputs, every activation f32), unpacked and unpatchified as
the pipeline does, written as the poem's `velocity` output reads out: [voxels (h, w), 32] f32, little-endian.
--bf16: the same step with the model and inputs in bf16 (torch's own bf16 roundings), for reference.
--loop <s0,s1,...> <out.f32>: the host's denoise loop from the same latent over sigmas s0, s1, ... then 0 (as the
poem's schedule gives them: engine-cerebras' `Denoise`), an Euler step a velocity, all in f32; the final latent as
the `latents` input lays it ([voxels (h, w), 32]).

Needs diffusers >= 0.36 (Flux2Transformer2DModel) and torch. Threads: 8."""
import math
import sys

import numpy as np
import torch
from diffusers import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline as P

torch.set_num_threads(8)

DIM, HEADS, HD, CTX, IN, VAE = 256, 2, 128, 192, 128, 32
INTER = 3 * DIM
DOUBLE, SINGLE = 2, 2
TXT, H, W = 64, 16, 16
# sigma 0.5 (the poem's timestep 500) and guidance 4: diffusers scales both by 1000 in the model's dtype, and
# 500 and 4000 are exact in bf16 (750 is not: a bf16 run would embed another timestep).
SIGMA, GUIDANCE = 0.5, 4.0
MASK = (1 << 64) - 1


def fnv(name):
    h = 0xCBF29CE484222325
    for b in name.encode():
        h ^= b
        h = (h * 0x100000001B3) & MASK
    return h


def units(name, count):
    """`count` values in [-1/2, 1/2), each exact in f32: engine-cpu's flux test draws the same."""
    seed = fnv(name)
    out = np.empty(count, dtype=np.int64)
    for i in range(count):
        seed = (seed * 6364136223846793005 + 1) & MASK
        out[i] = (seed >> 40) - (1 << 23)
    return out.astype(np.float32) / np.float32(1 << 24)


def bf16(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16).float()


def plane(name, shape):
    """A param of the poem: a norm's gain near one; else uniform with the variance of 1 / fan-in."""
    u = units(name, int(np.prod(shape)))
    if name.endswith("norm"):
        v = np.float32(1) + np.float32(0.5) * u
    else:
        v = u * np.sqrt(np.float32(12) / np.float32(shape[-1]))
    return bf16(v.reshape(shape))


def poem_params():
    p = {}
    p["dit.t_embed.linear_1"] = [DIM, 256]
    p["dit.t_embed.linear_2"] = [DIM, DIM]
    p["dit.g_embed.linear_1"] = [DIM, 256]
    p["dit.g_embed.linear_2"] = [DIM, DIM]
    p["dit.mod_txt"] = [6 * DIM, DIM]
    p["dit.mod_img"] = [6 * DIM, DIM]
    p["dit.mod_single"] = [3 * DIM, DIM]
    p["dit.norm_out"] = [2 * DIM, DIM]
    p["dit.context_embed"] = [DIM, CTX]
    p["dit.x_embed"] = [DIM, IN]
    for i in range(DOUBLE):
        for side in ("txt", "img"):
            s = f"dit.double.{i}.{side}"
            p[f"{s}.attn.qkv"] = [3 * DIM, DIM]
            p[f"{s}.attn.q_norm"] = [HD]
            p[f"{s}.attn.k_norm"] = [HD]
            p[f"{s}.attn.out"] = [DIM, DIM]
            p[f"{s}.ff.linear_in"] = [2 * INTER, DIM]
            p[f"{s}.ff.linear_out"] = [DIM, INTER]
    for i in range(SINGLE):
        s = f"dit.single.{i}"
        p[f"{s}.in_proj"] = [3 * DIM + 2 * INTER, DIM]
        p[f"{s}.q_norm"] = [HD]
        p[f"{s}.k_norm"] = [HD]
        p[f"{s}.out_attn"] = [DIM, DIM]
        p[f"{s}.out_mlp"] = [DIM, INTER]
    p["dit.proj_out"] = [IN, DIM]
    return {n: plane(n, s) for n, s in p.items()}


def adaln(w, order):
    """The poem reads an adaLN projection's slices in `order` (each (shift, scale) as (scale, shift)); the
    order is its own inverse."""
    return torch.cat([w[i * DIM:(i + 1) * DIM] for i in order], 0)


def state_dict(p):
    sd = {
        "time_guidance_embed.timestep_embedder.linear_1.weight": p["dit.t_embed.linear_1"],
        "time_guidance_embed.timestep_embedder.linear_2.weight": p["dit.t_embed.linear_2"],
        "time_guidance_embed.guidance_embedder.linear_1.weight": p["dit.g_embed.linear_1"],
        "time_guidance_embed.guidance_embedder.linear_2.weight": p["dit.g_embed.linear_2"],
        "double_stream_modulation_img.linear.weight": adaln(p["dit.mod_img"], [1, 0, 2, 4, 3, 5]),
        "double_stream_modulation_txt.linear.weight": adaln(p["dit.mod_txt"], [1, 0, 2, 4, 3, 5]),
        "single_stream_modulation.linear.weight": adaln(p["dit.mod_single"], [1, 0, 2]),
        "norm_out.linear.weight": p["dit.norm_out"],
        "context_embedder.weight": p["dit.context_embed"],
        "x_embedder.weight": p["dit.x_embed"],
        "proj_out.weight": p["dit.proj_out"],
    }
    for i in range(DOUBLE):
        b = f"transformer_blocks.{i}"
        for side, qkv, norms, out, ff in (
            ("img", ("to_q", "to_k", "to_v"), ("norm_q", "norm_k"), "to_out.0", "ff"),
            ("txt", ("add_q_proj", "add_k_proj", "add_v_proj"), ("norm_added_q", "norm_added_k"), "to_add_out", "ff_context"),
        ):
            s = f"dit.double.{i}.{side}"
            for j, n in enumerate(qkv):
                sd[f"{b}.attn.{n}.weight"] = p[f"{s}.attn.qkv"][j * DIM:(j + 1) * DIM]
            sd[f"{b}.attn.{norms[0]}.weight"] = p[f"{s}.attn.q_norm"]
            sd[f"{b}.attn.{norms[1]}.weight"] = p[f"{s}.attn.k_norm"]
            sd[f"{b}.attn.{out}.weight"] = p[f"{s}.attn.out"]
            sd[f"{b}.{ff}.linear_in.weight"] = p[f"{s}.ff.linear_in"]
            sd[f"{b}.{ff}.linear_out.weight"] = p[f"{s}.ff.linear_out"]
    for i in range(SINGLE):
        b, s = f"single_transformer_blocks.{i}.attn", f"dit.single.{i}"
        sd[f"{b}.to_qkv_mlp_proj.weight"] = p[f"{s}.in_proj"]
        sd[f"{b}.norm_q.weight"] = p[f"{s}.q_norm"]
        sd[f"{b}.norm_k.weight"] = p[f"{s}.k_norm"]
        sd[f"{b}.to_out.weight"] = torch.cat([p[f"{s}.out_attn"], p[f"{s}.out_mlp"]], 1)
    return sd


def model(sd):
    m = Flux2Transformer2DModel(
        patch_size=1, in_channels=IN, num_layers=DOUBLE, num_single_layers=SINGLE, attention_head_dim=HD,
        num_attention_heads=HEADS, joint_attention_dim=CTX, timestep_guidance_channels=256, mlp_ratio=3.0,
        axes_dims_rope=(32, 32, 32, 32), rope_theta=2000, eps=1e-6, guidance_embeds=True,
    ).float().eval()
    m.load_state_dict(sd, strict=True)
    return m


def inputs():
    """The step's inputs as the poem takes them: the latent's voxels (h, w) x 32 channels and the context's
    rows, drawn as the test draws them (unit variance), rounded to bf16."""
    s = np.sqrt(np.float32(12))
    latents = bf16(units("input.latents", H * W * VAE) * s).reshape(H, W, VAE)
    context = bf16(units("input.context", TXT * CTX) * s).reshape(TXT, CTX)
    return latents, context


def step(m, latents, context, dtype, sigma=SIGMA):
    x = latents.permute(2, 0, 1)[None]                       # (1, 32, H, W)
    x = P._patchify_latents(x)                               # (1, 128, H/2, W/2)
    ids = P._prepare_latent_ids(x)
    tokens = P._pack_latents(x)                              # (1, 64, 128)
    ctx = context[None]
    txt_ids = P._prepare_text_ids(ctx)
    m = m.to(dtype)
    with torch.no_grad():
        v = m(
            hidden_states=tokens.to(dtype), encoder_hidden_states=ctx.to(dtype),
            timestep=torch.tensor([sigma], dtype=dtype), guidance=torch.tensor([GUIDANCE], dtype=dtype),
            img_ids=ids, txt_ids=txt_ids, return_dict=False,
        )[0].float()
    try:
        v = P._unpack_latents_with_ids(v, ids, H // 2, W // 2)  # (1, 128, H/2, W/2)
    except TypeError:                                    # diffusers >= 0.37 reads the grid off the ids
        v = P._unpack_latents_with_ids(v, ids)
    v = P._unpatchify_latents(v)[0]                          # (32, H, W)
    return v.permute(1, 2, 0).reshape(H * W, VAE).contiguous().numpy().astype("<f4")


def main():
    out = sys.argv[1]
    p = poem_params()
    m = model(state_dict(p))
    latents, context = inputs()
    v = step(m, latents, context, torch.float32)
    v.tofile(out)
    print(f"{out}: velocity {v.shape}, rms {float(np.sqrt((v ** 2).mean())):.4f}")
    if "--bf16" in sys.argv:
        b = step(m, latents, context, torch.bfloat16)
        b.tofile(sys.argv[sys.argv.index("--bf16") + 1])
        rel = float(np.linalg.norm(b - v) / np.linalg.norm(v))
        cos = float((b * v).sum() / np.linalg.norm(b) / np.linalg.norm(v))
        print(f"bf16 model vs f32: rel L2 {rel:.3e}, cosine {cos:.6f}")
    if "--loop" in sys.argv:
        at = sys.argv.index("--loop")
        sig = [np.float32(x) for x in sys.argv[at + 1].split(",")] + [np.float32(0)]
        x = latents.clone()
        for i in range(len(sig) - 1):
            v = step(m, x, context, torch.float32, float(sig[i]))
            dt = np.float32(sig[i + 1] - sig[i])
            x = x + torch.from_numpy(v).reshape(H, W, VAE) * float(dt)
        x = x.reshape(H * W, VAE).contiguous().numpy().astype("<f4")
        x.tofile(sys.argv[at + 2])
        print(f"{sys.argv[at + 2]}: latent after {len(sig) - 1} steps, rms {float(np.sqrt((x ** 2).mean())):.4f}")


if __name__ == "__main__":
    main()
