#!/usr/bin/env python3
"""
mini_dit_ref.py -- self-contained PyTorch reference for pie's M0 synthetic `mini-dit` family.

This is NOT a real model.  It is the smallest graph that exercises *exactly* the
substrate patterns milestone M0 of .wiki/imagegen/design.md must land:

  block 0  "single"  single-stream joint block   -- text+image tokens in ONE sequence,
                                                   adaLN-Zero 6-param modulation
  block 1  "double"  MM-DiT double-stream block  -- separate text/image qkv+mlp weights,
                                                   joint attention, per-stream modulation
  block 2  "cross"   Wan-style cross-attn block  -- self-attn + cross-attn (image queries,
                                                   512-wide context) + FFN, 6-param
                                                   modulation = shared temb + per-block table

plus: RMSNorm QK-norm, 3-axis interleaved RoPE (dims [16,24,24], theta 10000),
SwiGLU MLP (ratio 2), final LayerNorm-no-affine + scale/shift + Linear to C*p*p.

Everything is deterministic from seed 0 and runs on CPU in under a second.

Usage
-----
    python mini_dit_ref.py --init            # write mini_dit.safetensors + config.json
    python mini_dit_ref.py --dump            # write mini_dit_dump_fp32.npz + _bf16.npz
    python mini_dit_ref.py --euler           # write mini_dit_euler_fp32.npz + _bf16.npz
    python mini_dit_ref.py --all             # all of the above (default)
    python mini_dit_ref.py --out-dir DIR     # default $PIE_IMAGEGEN_GOLDEN/mini-dit

Artifacts land outside the repo (default /root/.cache/pie-imagegen/golden/mini-dit)
because they are ~23 MB and fully regenerable from seed 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

CONFIG: Dict = {
    "model_type": "mini_dit",
    "version": 1,
    "seed": 0,
    "init_gain": 1.0,
    "init_bias_std": 0.02,
    "init_table_std": 0.5,

    "hidden_size": 256,
    "num_heads": 4,
    "head_dim": 64,
    "block_types": ["single", "double", "cross"],
    "mlp_ratio": 2.0,
    "mlp_hidden": 512,
    "qk_norm": "rms",
    "rms_eps": 1e-6,
    "ln_eps": 1e-6,

    "rope_axes_dims": [16, 24, 24],
    "rope_theta": 10000.0,
    "rope_form": "interleaved",
    "rope_axes": ["t", "h", "w"],

    "timestep_embed_dim": 256,
    "timestep_max_period": 10000.0,
    "timestep_flip_sin_to_cos": False,

    "text_dim": 256,
    "text_len": 8,
    "context_dim": 512,
    "context_len": 16,

    "in_channels": 16,
    "out_channels": 16,
    "patch_size": 2,
    "latent_shape": [16, 16, 16],
    "image_tokens": 64,
    "patch_features": 64,

    "joint_order": ["text", "image"],
    "text_positions": "(i, 0, 0)",
    "image_positions": "(0, h, w)",

    "mod_order_single": ["shift_msa", "scale_msa", "gate_msa",
                         "shift_mlp", "scale_mlp", "gate_mlp"],
    "mod_order_double": ["shift_msa", "scale_msa", "gate_msa",
                         "shift_mlp", "scale_mlp", "gate_mlp"],
    "mod_order_cross": ["shift_msa", "scale_msa", "gate_msa",
                        "shift_ffn", "scale_ffn", "gate_ffn"],
    "mod_order_final": ["shift", "scale"],
    "modulate_form": "x * (1 + scale) + shift",
    "residual_form": "x = x + gate * y",

    "euler_steps": 4,
    "euler_sigmas": [1.0, 0.75, 0.5, 0.25, 0.0],
    "euler_t_scale": 1000.0,
    "euler_update": "x <- x + (sigma[i+1] - sigma[i]) * v",

    "dump_batch": 2,
    "dump_timesteps": [500.0, 250.0],
    "dump_input_seed": 1234,
    "euler_noise_seed": 7,
}

D = CONFIG["hidden_size"]
H = CONFIG["num_heads"]
DH = CONFIG["head_dim"]
FF = CONFIG["mlp_hidden"]
assert H * DH == D
assert sum(CONFIG["rope_axes_dims"]) == DH
assert CONFIG["in_channels"] * CONFIG["patch_size"] ** 2 == CONFIG["patch_features"]

DEFAULT_OUT = os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden")
DEFAULT_OUT = os.path.join(DEFAULT_OUT, "mini-dit")

class Policy:
    """`fp32` is exact fp32 everywhere.  `bf16` rounds to bf16 at every op boundary a
    bf16 serving engine would: weights are bf16, every Linear in/out is bf16, but
    norm/modulation/residual/rope/softmax reductions are done in fp32 and rounded back."""

    def __init__(self, name: str):
        assert name in ("fp32", "bf16")
        self.name = name
        self.dt = torch.float32 if name == "fp32" else torch.bfloat16

    def cast(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.dt)

def sinusoid(t: torch.Tensor, dim: int, max_period: float) -> torch.Tensor:
    """[N] -> [N, dim].  Half sin then half cos, fp32.  (Elementwise::Sinusoid in D3.)"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float64) / half)
    ang = t.to(torch.float64)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).float()

def layernorm_no_affine(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = xf.var(-1, unbiased=False, keepdim=True)
    return (xf - mu) * torch.rsqrt(var + eps)

def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x [B,N,D]; shift/scale [B,D].  fp32."""
    return x.float() * (1.0 + scale.float()[:, None, :]) + shift.float()[:, None, :]

def rope_cos_sin(pos: torch.Tensor, dims: List[int], theta: float):
    """pos [N, len(dims)] float -> (cos, sin) each [N, head_dim//2] fp32.
    Per axis a of width d: freqs = theta ** (-2i/d) for i in [0, d/2)."""
    angs = []
    for a, d in enumerate(dims):
        half = d // 2
        freqs = theta ** (-(torch.arange(half, dtype=torch.float64) * 2.0 / d))
        angs.append(pos[:, a: a + 1].to(torch.float64) * freqs[None, :])
    ang = torch.cat(angs, dim=-1)
    return torch.cos(ang).float(), torch.sin(ang).float()

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B,H,N,DH]; cos/sin [N,DH//2].  Interleaved: pairs (x[2i], x[2i+1]).  fp32 math."""
    b, h, n, d = x.shape
    xr = x.float().reshape(b, h, n, d // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    c, s = cos[None, None], sin[None, None]
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).reshape(b, h, n, d)

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pol: Policy) -> torch.Tensor:
    """[B,H,N,DH] x [B,H,M,DH] x [B,H,M,DH] -> [B,H,N,DH].  scores+softmax in fp32."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(pol.cast(p), pol.cast(v))

def split_heads(x: torch.Tensor) -> torch.Tensor:
    b, n, _ = x.shape
    return x.reshape(b, n, H, DH).permute(0, 2, 1, 3).contiguous()

def merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, n, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, n, h * d).contiguous()

class SelfAttnW(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=True)
        self.norm_q = nn.Parameter(torch.ones(DH))
        self.norm_k = nn.Parameter(torch.ones(DH))
        self.out = nn.Linear(D, D, bias=True)

class CrossAttnW(nn.Module):
    def __init__(self, ctx_dim: int):
        super().__init__()
        self.q = nn.Linear(D, D, bias=True)
        self.kv = nn.Linear(ctx_dim, 2 * D, bias=True)
        self.norm_q = nn.Parameter(torch.ones(DH))
        self.norm_k = nn.Parameter(torch.ones(DH))
        self.out = nn.Linear(D, D, bias=True)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D, FF, bias=True)
        self.up_proj = nn.Linear(D, FF, bias=True)
        self.down_proj = nn.Linear(FF, D, bias=True)

class SingleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.adaLN = nn.Linear(D, 6 * D, bias=True)
        self.attn = SelfAttnW()
        self.mlp = SwiGLU()

class DoubleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_adaLN = nn.Linear(D, 6 * D, bias=True)
        self.txt_adaLN = nn.Linear(D, 6 * D, bias=True)
        self.img_attn = SelfAttnW()
        self.txt_attn = SelfAttnW()
        self.img_mlp = SwiGLU()
        self.txt_mlp = SwiGLU()

class CrossBlock(nn.Module):
    def __init__(self, ctx_dim: int):
        super().__init__()
        self.mod_table = nn.Parameter(torch.zeros(6, D))
        self.adaLN = nn.Linear(D, 6 * D, bias=True)
        self.self_attn = SelfAttnW()
        self.norm_cross = nn.LayerNorm(D, eps=CONFIG["ln_eps"], elementwise_affine=True)
        self.cross_attn = CrossAttnW(ctx_dim)
        self.mlp = SwiGLU()

class MiniDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(CONFIG["patch_features"], D, bias=True)
        self.blocks = nn.ModuleList([SingleBlock(), DoubleBlock(), CrossBlock(CONFIG["context_dim"])])
        self.final_adaLN = nn.Linear(D, 2 * D, bias=True)
        self.final_proj = nn.Linear(D, CONFIG["patch_features"], bias=True)

    def init_deterministic(self):
        gain = CONFIG["init_gain"]
        bstd = CONFIG["init_bias_std"]
        g = torch.Generator().manual_seed(CONFIG["seed"])
        for name, p in sorted(self.named_parameters()):
            r = torch.randn(p.shape, generator=g)
            if name == "blocks.2.mod_table":
                p.data = CONFIG["init_table_std"] * r
            elif name.endswith("norm_q") or name.endswith("norm_k") \
                    or name == "blocks.2.norm_cross.weight":
                p.data = 1.0 + bstd * r
            elif p.ndim == 2:
                p.data = (gain / math.sqrt(p.shape[1])) * r
            else:
                p.data = bstd * r
        return self

def patchify(latent: torch.Tensor) -> torch.Tensor:
    """[B,C,Hs,Ws] -> [B, (Hs/p)*(Ws/p), C*p*p], feature order (c, ph, pw)."""
    p = CONFIG["patch_size"]
    b, c, hs, ws = latent.shape
    x = latent.reshape(b, c, hs // p, p, ws // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (hs // p) * (ws // p), c * p * p).contiguous()

def unpatchify(tokens: torch.Tensor, c: int, hs: int, ws: int) -> torch.Tensor:
    p = CONFIG["patch_size"]
    b = tokens.shape[0]
    x = tokens.reshape(b, hs // p, ws // p, c, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, hs, ws).contiguous()

def image_positions(hs: int, ws: int) -> torch.Tensor:
    p = CONFIG["patch_size"]
    hp, wp = hs // p, ws // p
    hh, ww = torch.meshgrid(torch.arange(hp), torch.arange(wp), indexing="ij")
    z = torch.zeros_like(hh)
    return torch.stack([z, hh, ww], dim=-1).reshape(hp * wp, 3).float()

def text_positions(n: int) -> torch.Tensor:
    i = torch.arange(n).float()
    z = torch.zeros(n)
    return torch.stack([i, z, z], dim=-1)

class Dump(dict):
    def put(self, key: str, t: torch.Tensor):
        self[key] = t.detach().float().cpu().numpy()

def lin(m: nn.Linear, x: torch.Tensor, pol: Policy) -> torch.Tensor:
    return torch.nn.functional.linear(pol.cast(x), pol.cast(m.weight), pol.cast(m.bias))

def swiglu(m: SwiGLU, x: torch.Tensor, pol: Policy) -> torch.Tensor:
    g = lin(m.gate_proj, x, pol)
    u = lin(m.up_proj, x, pol)
    h = pol.cast(torch.nn.functional.silu(g.float()) * u.float())
    return lin(m.down_proj, h, pol)

def forward(model: MiniDiT,
            latent: torch.Tensor,
            text: torch.Tensor,
            context: torch.Tensor,
            timestep: torch.Tensor,
            pol: Policy,
            dump: Dump | None = None,
            prefix: str = "") -> torch.Tensor:
    P = (lambda k, t: dump.put(prefix + k, t)) if dump is not None else (lambda k, t: None)

    b, c, hs, ws = latent.shape
    lt = text.shape[1]
    li = (hs // CONFIG["patch_size"]) * (ws // CONFIG["patch_size"])

    P("in.latent", latent); P("in.text", text); P("in.context", context)
    P("in.timestep", timestep)

    txt_pos = text_positions(lt)
    img_pos = image_positions(hs, ws)
    P("in.txt_pos", txt_pos); P("in.img_pos", img_pos)
    cos_t, sin_t = rope_cos_sin(txt_pos, CONFIG["rope_axes_dims"], CONFIG["rope_theta"])
    cos_i, sin_i = rope_cos_sin(img_pos, CONFIG["rope_axes_dims"], CONFIG["rope_theta"])
    cos_j = torch.cat([cos_t, cos_i], 0); sin_j = torch.cat([sin_t, sin_i], 0)
    P("rope.cos_txt", cos_t); P("rope.sin_txt", sin_t)
    P("rope.cos_img", cos_i); P("rope.sin_img", sin_i)

    temb = sinusoid(timestep, CONFIG["timestep_embed_dim"], CONFIG["timestep_max_period"])
    P("temb", temb)
    temb_act = pol.cast(torch.nn.functional.silu(temb))
    P("temb_silu", temb_act)

    patches = patchify(latent)
    P("patches", patches)
    img = lin(model.x_embedder, patches, pol)
    P("x_embed", img)
    txt = pol.cast(text)

    blk: SingleBlock = model.blocks[0]
    mod = lin(blk.adaLN, temb_act, pol).float()
    P("b0.mod", mod)
    sh_a, sc_a, g_a, sh_m, sc_m, g_m = mod.chunk(6, dim=-1)
    for nm, t in zip(CONFIG["mod_order_single"], (sh_a, sc_a, g_a, sh_m, sc_m, g_m)):
        P(f"b0.mod.{nm}", t)

    x = torch.cat([txt, img], dim=1)
    P("b0.in", x)
    h = pol.cast(modulate(layernorm_no_affine(x, CONFIG["ln_eps"]), sh_a, sc_a))
    P("b0.norm1_out", h)
    qkv = lin(blk.attn.qkv, h, pol)
    q, k, v = [split_heads(t) for t in qkv.chunk(3, dim=-1)]
    P("b0.q_raw", q); P("b0.k_raw", k); P("b0.v", v)
    q = pol.cast(rmsnorm(q, blk.attn.norm_q, CONFIG["rms_eps"]))
    k = pol.cast(rmsnorm(k, blk.attn.norm_k, CONFIG["rms_eps"]))
    P("b0.q_qknorm", q); P("b0.k_qknorm", k)
    q = pol.cast(apply_rope(q, cos_j, sin_j))
    k = pol.cast(apply_rope(k, cos_j, sin_j))
    P("b0.q_rope", q); P("b0.k_rope", k)
    a = attention(q, k, v, pol)
    P("b0.attn_heads", a)
    a = lin(blk.attn.out, merge_heads(a), pol)
    P("b0.attn_out", a)
    x = pol.cast(x.float() + g_a[:, None, :] * a.float())
    P("b0.x_after_attn", x)
    h = pol.cast(modulate(layernorm_no_affine(x, CONFIG["ln_eps"]), sh_m, sc_m))
    P("b0.norm2_out", h)
    m = swiglu(blk.mlp, h, pol)
    P("b0.mlp_out", m)
    x = pol.cast(x.float() + g_m[:, None, :] * m.float())
    P("b0.out", x)
    txt, img = x[:, :lt], x[:, lt:]
    P("b0.out_txt", txt); P("b0.out_img", img)

    dblk: DoubleBlock = model.blocks[1]
    imod = lin(dblk.img_adaLN, temb_act, pol).float()
    tmod = lin(dblk.txt_adaLN, temb_act, pol).float()
    P("b1.img_mod", imod); P("b1.txt_mod", tmod)
    i_sh_a, i_sc_a, i_g_a, i_sh_m, i_sc_m, i_g_m = imod.chunk(6, dim=-1)
    t_sh_a, t_sc_a, t_g_a, t_sh_m, t_sc_m, t_g_m = tmod.chunk(6, dim=-1)
    for nm, t in zip(CONFIG["mod_order_double"], (i_sh_a, i_sc_a, i_g_a, i_sh_m, i_sc_m, i_g_m)):
        P(f"b1.img_mod.{nm}", t)
    for nm, t in zip(CONFIG["mod_order_double"], (t_sh_a, t_sc_a, t_g_a, t_sh_m, t_sc_m, t_g_m)):
        P(f"b1.txt_mod.{nm}", t)

    ih = pol.cast(modulate(layernorm_no_affine(img, CONFIG["ln_eps"]), i_sh_a, i_sc_a))
    th = pol.cast(modulate(layernorm_no_affine(txt, CONFIG["ln_eps"]), t_sh_a, t_sc_a))
    P("b1.img_norm1_out", ih); P("b1.txt_norm1_out", th)

    iq, ik, iv = [split_heads(t) for t in lin(dblk.img_attn.qkv, ih, pol).chunk(3, -1)]
    tq, tk, tv = [split_heads(t) for t in lin(dblk.txt_attn.qkv, th, pol).chunk(3, -1)]
    iq = pol.cast(apply_rope(pol.cast(rmsnorm(iq, dblk.img_attn.norm_q, CONFIG["rms_eps"])), cos_i, sin_i))
    ik = pol.cast(apply_rope(pol.cast(rmsnorm(ik, dblk.img_attn.norm_k, CONFIG["rms_eps"])), cos_i, sin_i))
    tq = pol.cast(apply_rope(pol.cast(rmsnorm(tq, dblk.txt_attn.norm_q, CONFIG["rms_eps"])), cos_t, sin_t))
    tk = pol.cast(apply_rope(pol.cast(rmsnorm(tk, dblk.txt_attn.norm_k, CONFIG["rms_eps"])), cos_t, sin_t))
    P("b1.img_q_rope", iq); P("b1.img_k_rope", ik); P("b1.img_v", iv)
    P("b1.txt_q_rope", tq); P("b1.txt_k_rope", tk); P("b1.txt_v", tv)

    jq = torch.cat([tq, iq], dim=2); jk = torch.cat([tk, ik], dim=2); jv = torch.cat([tv, iv], dim=2)
    ja = attention(jq, jk, jv, pol)
    P("b1.joint_attn_heads", ja)
    ta, ia = ja[:, :, :lt], ja[:, :, lt:]
    ia = lin(dblk.img_attn.out, merge_heads(ia), pol)
    ta = lin(dblk.txt_attn.out, merge_heads(ta), pol)
    P("b1.img_attn_out", ia); P("b1.txt_attn_out", ta)
    img = pol.cast(img.float() + i_g_a[:, None, :] * ia.float())
    txt = pol.cast(txt.float() + t_g_a[:, None, :] * ta.float())
    P("b1.img_after_attn", img); P("b1.txt_after_attn", txt)

    ih = pol.cast(modulate(layernorm_no_affine(img, CONFIG["ln_eps"]), i_sh_m, i_sc_m))
    th = pol.cast(modulate(layernorm_no_affine(txt, CONFIG["ln_eps"]), t_sh_m, t_sc_m))
    P("b1.img_norm2_out", ih); P("b1.txt_norm2_out", th)
    im = swiglu(dblk.img_mlp, ih, pol); tm = swiglu(dblk.txt_mlp, th, pol)
    P("b1.img_mlp_out", im); P("b1.txt_mlp_out", tm)
    img = pol.cast(img.float() + i_g_m[:, None, :] * im.float())
    txt = pol.cast(txt.float() + t_g_m[:, None, :] * tm.float())
    P("b1.out_img", img); P("b1.out_txt", txt)

    cblk: CrossBlock = model.blocks[2]
    proj = lin(cblk.adaLN, temb_act, pol).float().reshape(b, 6, D)
    P("b2.mod_proj", proj)
    mod2 = cblk.mod_table.float()[None] + proj
    P("b2.mod", mod2)
    sh_a, sc_a, g_a, sh_f, sc_f, g_f = [mod2[:, i] for i in range(6)]
    for nm, t in zip(CONFIG["mod_order_cross"], (sh_a, sc_a, g_a, sh_f, sc_f, g_f)):
        P(f"b2.mod.{nm}", t)

    x = img
    P("b2.in", x)
    h = pol.cast(modulate(layernorm_no_affine(x, CONFIG["ln_eps"]), sh_a, sc_a))
    P("b2.norm1_out", h)
    q, k, v = [split_heads(t) for t in lin(cblk.self_attn.qkv, h, pol).chunk(3, -1)]
    q = pol.cast(apply_rope(pol.cast(rmsnorm(q, cblk.self_attn.norm_q, CONFIG["rms_eps"])), cos_i, sin_i))
    k = pol.cast(apply_rope(pol.cast(rmsnorm(k, cblk.self_attn.norm_k, CONFIG["rms_eps"])), cos_i, sin_i))
    P("b2.self_q_rope", q); P("b2.self_k_rope", k); P("b2.self_v", v)
    a = attention(q, k, v, pol)
    P("b2.self_attn_heads", a)
    a = lin(cblk.self_attn.out, merge_heads(a), pol)
    P("b2.self_attn_out", a)
    x = pol.cast(x.float() + g_a[:, None, :] * a.float())
    P("b2.x_after_self", x)

    hc = pol.cast(torch.nn.functional.layer_norm(
        x.float(), (D,), cblk.norm_cross.weight.float(), cblk.norm_cross.bias.float(),
        CONFIG["ln_eps"]))
    P("b2.cross_norm_out", hc)
    cq = split_heads(lin(cblk.cross_attn.q, hc, pol))
    ckv = lin(cblk.cross_attn.kv, context, pol)
    ck, cv = [split_heads(t) for t in ckv.chunk(2, -1)]
    cq = pol.cast(rmsnorm(cq, cblk.cross_attn.norm_q, CONFIG["rms_eps"]))
    ck = pol.cast(rmsnorm(ck, cblk.cross_attn.norm_k, CONFIG["rms_eps"]))
    P("b2.cross_q", cq); P("b2.cross_k", ck); P("b2.cross_v", cv)
    ca = attention(cq, ck, cv, pol)
    P("b2.cross_attn_heads", ca)
    ca = lin(cblk.cross_attn.out, merge_heads(ca), pol)
    P("b2.cross_attn_out", ca)
    x = pol.cast(x.float() + ca.float())
    P("b2.x_after_cross", x)

    h = pol.cast(modulate(layernorm_no_affine(x, CONFIG["ln_eps"]), sh_f, sc_f))
    P("b2.norm3_out", h)
    m = swiglu(cblk.mlp, h, pol)
    P("b2.mlp_out", m)
    x = pol.cast(x.float() + g_f[:, None, :] * m.float())
    P("b2.out", x)

    fmod = lin(model.final_adaLN, temb_act, pol).float()
    f_sh, f_sc = fmod.chunk(2, dim=-1)
    P("final.mod", fmod); P("final.mod.shift", f_sh); P("final.mod.scale", f_sc)
    hf = pol.cast(modulate(layernorm_no_affine(x, CONFIG["ln_eps"]), f_sh, f_sc))
    P("final.norm_out", hf)
    tok = lin(model.final_proj, hf, pol)
    P("final.tokens", tok)
    vel = unpatchify(tok.float(), CONFIG["out_channels"], hs, ws)
    P("velocity", vel)
    return vel

def fixed_inputs():
    g = torch.Generator().manual_seed(CONFIG["dump_input_seed"])
    b = CONFIG["dump_batch"]
    c, hs, ws = CONFIG["latent_shape"]
    latent = torch.randn(b, c, hs, ws, generator=g)
    text = torch.randn(b, CONFIG["text_len"], CONFIG["text_dim"], generator=g)
    context = torch.randn(b, CONFIG["context_len"], CONFIG["context_dim"], generator=g)
    timestep = torch.tensor(CONFIG["dump_timesteps"], dtype=torch.float32)
    return latent, text, context, timestep

def build() -> MiniDiT:
    torch.manual_seed(CONFIG["seed"])
    return MiniDiT().init_deterministic().eval()

def cmd_init(out_dir: str):
    from safetensors.torch import save_file
    m = build()
    sd = {k: v.detach().contiguous().float() for k, v in m.state_dict().items()}
    n = sum(v.numel() for v in sd.values())
    save_file(sd, os.path.join(out_dir, "mini_dit.safetensors"),
              metadata={"format": "pt"})
    cfg = dict(CONFIG)
    cfg["num_parameters"] = int(n)
    cfg["tensors"] = {k: list(v.shape) for k, v in sorted(sd.items())}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[init] {n} params -> {out_dir}/mini_dit.safetensors, config.json")
    for k in sorted(sd):
        print(f"       {k:44s} {tuple(sd[k].shape)}")

@torch.no_grad()
def cmd_dump(out_dir: str):
    m = build()
    latent, text, context, timestep = fixed_inputs()
    for name in ("fp32", "bf16"):
        pol = Policy(name)
        d = Dump()
        forward(m, latent, text, context, timestep, pol, dump=d)
        path = os.path.join(out_dir, f"mini_dit_dump_{name}.npz")
        np.savez(path, **d)
        print(f"[dump/{name}] {len(d)} tensors -> {path}")

@torch.no_grad()
def cmd_euler(out_dir: str):
    m = build()
    _, text, context, _ = fixed_inputs()
    b = CONFIG["dump_batch"]
    c, hs, ws = CONFIG["latent_shape"]
    g = torch.Generator().manual_seed(CONFIG["euler_noise_seed"])
    x0 = torch.randn(b, c, hs, ws, generator=g)
    sig = CONFIG["euler_sigmas"]
    for name in ("fp32", "bf16"):
        pol = Policy(name)
        d = Dump()
        d.put("euler.x_init", x0)
        d["euler.sigmas"] = np.asarray(sig, dtype=np.float32)
        x = x0.clone()
        for i in range(CONFIG["euler_steps"]):
            t = torch.full((b,), sig[i] * CONFIG["euler_t_scale"])
            d.put(f"euler.t{i}", t)
            v = forward(m, x, text, context, t, pol, dump=d, prefix=f"euler.s{i}.")
            d.put(f"euler.v{i}", v)
            x = x + (sig[i + 1] - sig[i]) * v
            d.put(f"euler.x{i + 1}", x)
        d.put("euler.latent", x)
        path = os.path.join(out_dir, f"mini_dit_euler_{name}.npz")
        np.savez(path, **d)
        print(f"[euler/{name}] {len(d)} tensors -> {path}   final |x| = {x.abs().mean():.6f}")

def dpm2m_tables(sig):
    """Per-step scalars of DPM-Solver++ 2M on a flow schedule, the SDK's
    `latent::Dpm2m::of` in numpy: step i goes from sig[i] to sig[i+1] with
    `x0 = x - sig[i] v`, `x <- ratio x + gain D`, `D = (1 + mix) x0 - mix x0_prev`.
    The first step and the step onto sigma 0 are first order (mix 0)."""
    lam = lambda s: math.log(max(1.0 - s, 1e-12) / max(s, 1e-12))
    ratio, gain, mix = [], [], []
    h_prev = None
    for i in range(len(sig) - 1):
        ss, st = sig[i], sig[i + 1]
        a_s, a_t = 1.0 - ss, 1.0 - st
        e = 0.0 if st <= 0.0 else (st * a_s) / (ss * a_t)
        h = math.inf if st <= 0.0 else lam(st) - lam(ss)
        ratio.append(st / ss if ss > 0.0 else 0.0)
        gain.append(-a_t * (e - 1.0))
        mix.append(h / (2.0 * h_prev) if (h_prev is not None and math.isfinite(h)
                                          and math.isfinite(h_prev) and h_prev != 0.0) else 0.0)
        h_prev = h
    return ratio, gain, mix


def cmd_dpm2m(out_dir: str):
    """The Euler golden's start and schedule, integrated by DPM-Solver++ 2M."""
    m = build()
    _, text, context, _ = fixed_inputs()
    b = CONFIG["dump_batch"]
    c, hs, ws = CONFIG["latent_shape"]
    g = torch.Generator().manual_seed(CONFIG["euler_noise_seed"])
    x0 = torch.randn(b, c, hs, ws, generator=g)
    sig = CONFIG["euler_sigmas"]
    ratio, gain, mix = dpm2m_tables(sig)
    for name in ("fp32", "bf16"):
        pol = Policy(name)
        d = Dump()
        d.put("dpm2m.x_init", x0)
        d["dpm2m.sigmas"] = np.asarray(sig, dtype=np.float32)
        d["dpm2m.ratio"] = np.asarray(ratio, dtype=np.float32)
        d["dpm2m.gain"] = np.asarray(gain, dtype=np.float32)
        d["dpm2m.mix"] = np.asarray(mix, dtype=np.float32)
        x = x0.clone()
        prev = torch.zeros_like(x0)
        for i in range(CONFIG["euler_steps"]):
            t = torch.full((b,), sig[i] * CONFIG["euler_t_scale"])
            v = forward(m, x, text, context, t, pol, dump=d, prefix=f"dpm2m.s{i}.")
            d.put(f"dpm2m.v{i}", v)
            pred = x - sig[i] * v
            D = (1.0 + mix[i]) * pred - mix[i] * prev
            x = ratio[i] * x + gain[i] * D
            prev = pred
            d.put(f"dpm2m.x{i + 1}", x)
        d.put("dpm2m.latent", x)
        path = os.path.join(out_dir, f"mini_dit_dpm2m_{name}.npz")
        np.savez(path, **d)
        print(f"[dpm2m/{name}] {len(d)} tensors -> {path}   final |x| = {x.abs().mean():.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--euler", action="store_true")
    ap.add_argument("--dpm2m", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not (a.init or a.dump or a.euler or a.dpm2m):
        a.all = True
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    if a.all or a.init:
        cmd_init(a.out_dir)
    if a.all or a.dump:
        cmd_dump(a.out_dir)
    if a.all or a.euler:
        cmd_euler(a.out_dir)
    if a.all or a.dpm2m:
        cmd_dpm2m(a.out_dir)

if __name__ == "__main__":
    main()
