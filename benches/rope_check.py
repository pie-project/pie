"""Validate pie's DeepSeek/Kimi YaRN RoPE kernel against a numpy reference.

pie dumps `L<nn>_q_pe_pre` / `L<nn>_q_pe` (and the `k_pe` pair) around the RoPE
call; this script re-applies vLLM's `DeepseekScalingRotaryEmbedding` maths to the
`_pre` tensor and reports how far pie's kernel is from it. Because the reference
is self-contained it isolates the kernel from every other part of the layer.

    python benches/rope_check.py /tmp/act/pie --model /root/models/kimi26-l1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position: int
) -> float:
    return (dim * math.log(max_position / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position: int
) -> tuple[int, int]:
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(low: float, high: float, dim: int) -> np.ndarray:
    if low == high:
        high += 0.001
    ramp = (np.arange(dim, dtype=np.float64) - low) / (high - low)
    return np.clip(ramp, 0, 1)


def yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def deepseek_cos_sin(
    rotary_dim: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    mscale: float,
    mscale_all_dim: float,
    original_max_position: int,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """vLLM `DeepseekScalingRotaryEmbedding._compute_inv_freq` + `_compute_cos_sin`."""
    pos_freqs = base ** (
        np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = yarn_find_correction_range(
        beta_fast, beta_slow, rotary_dim, base, original_max_position
    )
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, rotary_dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    attn_scale = yarn_get_mscale(factor, mscale) / yarn_get_mscale(
        factor, mscale_all_dim
    )
    freqs = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    return np.cos(freqs) * attn_scale, np.sin(freqs) * attn_scale


def rotate_gptj(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """is_neox_style=False: rotate adjacent pairs (x0,x1), (x2,x3), ..."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    out = np.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


def rotate_neox(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """is_neox_style=True: rotate (x[:d/2], x[d/2:]) halves."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(af @ bf / d) if d else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pie_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.pie_dir)
    pre = np.load(d / f"s{args.step}_L{args.layer:02d}_q_pe_pre.npy")
    post = np.load(d / f"s{args.step}_L{args.layer:02d}_q_pe.npy")
    kpre = np.load(d / f"s{args.step}_L{args.layer:02d}_k_pe_pre.npy")
    kpost = np.load(d / f"s{args.step}_L{args.layer:02d}_k_pe.npy")
    positions = np.load(d / f"s{args.step}_positions.npy").ravel().astype(np.int64)

    cfg = json.loads((Path(args.model) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    rs = cfg["rope_scaling"]
    rotary_dim = cfg["qk_rope_head_dim"]
    n_heads = pre.shape[1] // rotary_dim

    cos, sin = deepseek_cos_sin(
        rotary_dim,
        float(cfg["rope_theta"]),
        float(rs["factor"]),
        float(rs.get("beta_fast", 32)),
        float(rs.get("beta_slow", 1)),
        float(rs.get("mscale", 1.0)),
        float(rs.get("mscale_all_dim", 0.0)),
        int(rs.get("original_max_position_embeddings",
                   cfg["max_position_embeddings"])),
        positions,
    )
    print(f"rotary_dim={rotary_dim} heads={n_heads} positions={positions[:6]}...")
    print(f"cos[1,:4]={cos[1,:4]}  attn_scale={cos[0,0]:.6f}")

    q = pre.reshape(-1, n_heads, rotary_dim).astype(np.float64)
    k = kpre.reshape(-1, 1, rotary_dim).astype(np.float64)
    for name, fn in (("gptj(interleaved)", rotate_gptj), ("neox(halves)", rotate_neox)):
        c = cos[:, None, :]
        s = sin[:, None, :]
        rq = fn(q, c, s).reshape(pre.shape)
        rk = fn(k, c, s).reshape(kpre.shape)
        print(f"{name:>18}  q_pe cos={cos_sim(rq, post):.6f}  "
              f"k_pe cos={cos_sim(rk, kpost):.6f}")


if __name__ == "__main__":
    main()
