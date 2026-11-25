"""The recurrence: the torch forwards of the recurrent sequence mixers, Gated
DeltaNet's (Qwen3.5) causal convolution, gates and delta rule.

A training prefill starts every lane with an empty state, so each `*Chunked`
op runs over the whole row from zero: the convolution's history is zero
padding and the delta rule's state starts at zero. The delta rule walks the
tokens one at a time in f32, which autograd differentiates as it is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6
"""The epsilon of the delta rule's q and k L2 norms (the kernels' `GDN_EPS`)."""


def _causal_conv(run, a):
    # A depthwise conv over each channel's last `conv_width` rows, then SiLU.
    x, w = run.get(a["x"]), run.get(a["weight"])
    width, dilation = a["conv_width"], a["dilation"]
    bias = None if a["bias"] is None else run.get(a["bias"]).float()
    xt = F.pad(x.float().transpose(1, 2), ((width - 1) * dilation, 0))
    y = F.conv1d(xt, w.float().reshape(w.shape[0], 1, width), bias, dilation=dilation, groups=w.shape[0])
    run.put(a["y"], F.silu(y).transpose(1, 2).to(x.dtype))


def _gdn_gates(run, a):
    # [b | a] -> [g | beta]: g = -exp(a_log) softplus(a + dt_bias), beta = sigmoid(b).
    ba = run.get(a["ba"]).float()
    heads = ba.shape[-1] // 2
    b, alpha = ba[..., :heads], ba[..., heads:]
    g = -run.get(a["a_log"]).float().exp() * F.softplus(alpha + run.get(a["dt_bias"]).float())
    run.vals[a["gates"]] = torch.cat([g, torch.sigmoid(b)], dim=-1)


def _gated_delta(run, a):
    qkv, gates = run.get(a["qkv"]), run.get(a["gates"]).float()
    hk, hv, dk, dv = a["k_heads"], a["v_heads"], a["k_dim"], a["v_dim"]
    B, T, _ = qkv.shape
    x = qkv.float()
    q = x[..., : hk * dk].reshape(B, T, hk, dk)
    k = x[..., hk * dk : 2 * hk * dk].reshape(B, T, hk, dk)
    v = x[..., 2 * hk * dk :].reshape(B, T, hv, dv)
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + EPS) / dk**0.5
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + EPS)
    # Each key head serves its group of value heads.
    q, k = q.repeat_interleave(hv // hk, dim=2), k.repeat_interleave(hv // hk, dim=2)
    decay, beta = gates[..., :hv].exp(), gates[..., hv:]
    state = x.new_zeros(B, hv, dk, dv)
    out = []
    for t in range(T):
        state = state * decay[:, t, :, None, None]
        delta = (v[:, t] - (state * k[:, t, ..., None]).sum(-2)) * beta[:, t, :, None]
        state = state + k[:, t, ..., None] * delta[..., None, :]
        out.append((state * q[:, t, ..., None]).sum(-2))
    run.put(a["y"], torch.stack(out, dim=1).reshape(B, T, hv * dv).to(qkv.dtype))


FORWARDS = {
    "Ssm.CausalConv1dChunked": _causal_conv,
    "Ssm.GdnGates": _gdn_gates,
    "Ssm.GatedDeltaChunked": _gated_delta,
}
"""Each recurrent op's forward, by `group.name`."""
