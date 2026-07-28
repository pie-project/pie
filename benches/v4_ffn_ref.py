"""Independent numpy/torch reference for one DeepSeek-V4 MoE FFN block.

`act_parity.py` tells you *that* pie and vLLM disagree at `L<nn>_ffn_out`; it
cannot tell you *which* of them is wrong, nor which sub-step (router / MXFP4
dequant / clamped SwiGLU / FP8 shared expert) is responsible. This script
recomputes the block straight from the safetensors, from pie's dumped
`L<nn>_ffn_in`, and reports the cosine of the reference against both engines
plus the intermediate routing tables.

Usage:
    /root/.venv/vllm/bin/python benches/v4_ffn_ref.py \
        --model /root/models/dsv4flash-mini --layer 0 \
        --pie /tmp/act/v4pie --vllm /tmp/act/v4vllm
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def load_tensors(model: str) -> dict:
    out = {}
    for f in sorted(glob.glob(model + "/*.safetensors")):
        with safe_open(f, "pt") as g:
            for k in g.keys():
                out[k] = g.get_tensor(k)
    return out


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`packed` [R, C/2] uint8/int8, `scale` [R, C/32] e8m0 → [R, C] float32."""
    p = packed.view(torch.uint8).to(torch.int64)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    R = p.shape[0]
    vals = torch.stack([lo, hi], dim=-1).reshape(R, -1)
    v = FP4_LUT.to(vals.device)[vals]
    s = scale.view(torch.uint8).to(torch.float32)
    s = torch.pow(2.0, s - 127.0)
    v = v.reshape(R, s.shape[1], 32) * s[:, :, None]
    return v.reshape(R, -1)


def dequant_fp8_block(w: torch.Tensor, scale: torch.Tensor,
                      block: int = 128) -> torch.Tensor:
    """FP8-E4M3 with per-[block, block] scales (stored e8m0 for V4-fp4)."""
    wf = w.to(torch.float32)
    if scale.dtype == torch.float8_e8m0fnu:
        s = torch.pow(2.0, scale.view(torch.uint8).to(torch.float32) - 127.0)
    else:
        s = scale.to(torch.float32)
    R, C = wf.shape
    s = s[: (R + block - 1) // block, : (C + block - 1) // block]
    s = s.repeat_interleave(block, 0).repeat_interleave(block, 1)[:R, :C]
    return wf * s


def clamp_swiglu(gate: torch.Tensor, up: torch.Tensor,
                 limit: float) -> torch.Tensor:
    g = gate.clamp(max=limit)
    u = up.clamp(min=-limit, max=limit)
    return g * torch.sigmoid(g) * u


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--pie", required=True)
    ap.add_argument("--vllm", default=None)
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(Path(args.model) / "config.json"))
    li = args.layer
    pref = f"layers.{li}.ffn."
    t = load_tensors(args.model)

    pie = Path(args.pie)
    x = torch.from_numpy(np.load(pie / f"s{args.step}_L{li:02d}_ffn_in.npy"))
    tok = np.load(pie / f"s{args.step}_tokens.npy").reshape(-1)[: x.shape[0]]
    tok = torch.from_numpy(tok.astype(np.int64))

    E = cfg["n_routed_experts"]
    K = cfg["num_experts_per_tok"]
    limit = float(cfg.get("swiglu_limit") or 0.0)
    rsf = float(cfg.get("routed_scaling_factor", 1.0))

    gate_w = t[pref + "gate.weight"].to(torch.float32)
    logits = x @ gate_w.T
    scores = torch.sqrt(torch.nn.functional.softplus(logits))

    tid2eid = t.get(pref + "gate.tid2eid")
    bias = t.get(pref + "gate.e_score_correction_bias")
    if tid2eid is not None:
        ids = tid2eid[tok].long()
    else:
        sel = scores + (bias.to(torch.float32) if bias is not None else 0.0)
        ids = torch.topk(sel, k=K, dim=-1).indices
    w = scores.gather(1, ids)
    if cfg.get("norm_topk_prob", True):
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-20)
    w = w * rsf

    print(f"layer {li}  hash={tid2eid is not None}  E={E} K={K} "
          f"limit={limit} rsf={rsf}")
    print("ids[0] =", ids[0].tolist())
    print("w[0]   =", [round(v, 5) for v in w[0].tolist()])

    y = torch.zeros_like(x)
    for e in range(E):
        w1 = dequant_mxfp4(t[f"{pref}experts.{e}.w1.weight"],
                           t[f"{pref}experts.{e}.w1.scale"])
        w3 = dequant_mxfp4(t[f"{pref}experts.{e}.w3.weight"],
                           t[f"{pref}experts.{e}.w3.scale"])
        w2 = dequant_mxfp4(t[f"{pref}experts.{e}.w2.weight"],
                           t[f"{pref}experts.{e}.w2.scale"])
        mask = (ids == e)
        rows = mask.any(-1).nonzero().reshape(-1)
        if rows.numel() == 0:
            continue
        xi = x[rows]
        h = clamp_swiglu(xi @ w1.T, xi @ w3.T, limit) if limit > 0 else \
            torch.nn.functional.silu(xi @ w1.T) * (xi @ w3.T)
        oi = h @ w2.T
        wi = (w * mask).sum(-1)[rows]
        y[rows] += oi * wi[:, None]
    routed = y.clone()
    shared = torch.zeros_like(y)

    if pref + "shared_experts.w1.weight" in t:
        s1 = dequant_fp8_block(t[pref + "shared_experts.w1.weight"],
                               t[pref + "shared_experts.w1.scale"])
        s3 = dequant_fp8_block(t[pref + "shared_experts.w3.weight"],
                               t[pref + "shared_experts.w3.scale"])
        s2 = dequant_fp8_block(t[pref + "shared_experts.w2.weight"],
                               t[pref + "shared_experts.w2.scale"])
        h = clamp_swiglu(x @ s1.T, x @ s3.T, limit) if limit > 0 else \
            torch.nn.functional.silu(x @ s1.T) * (x @ s3.T)
        shared = h @ s2.T
        print("shared |.| =", float(shared.norm()), " routed |.| =",
              float(y.norm()))
        y = y + shared

    for tag, refv in (("routed_out", routed), ("shared_out", shared),
                      ("ffn_out", y)):
        f = pie / f"s{args.step}_L{li:02d}_{tag}.npy"
        if not f.exists():
            continue
        print(f"ref vs pie   {tag:<11s} cos={cos(refv.numpy(), np.load(f)):.6f}")
    if args.vllm:
        v = Path(args.vllm) / f"s{args.step}_L{li:02d}_ffn_out.npy"
        if v.exists():
            print(f"ref vs vllm  ffn_out     cos={cos(y.numpy(), np.load(v)):.6f}")


if __name__ == "__main__":
    main()
