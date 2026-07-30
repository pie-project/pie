#!/usr/bin/env python3
"""Localize the first diverging (kernel, layer) between the MLX reference taps
and the raw-Metal driver taps.

Both trees are written by the PIE_METAL_GOLDEN_DIR hook: the reference by
tests/mlx/model/qwen3_5.cpp's dump_kernel (or, for gemma4, by
tests/parity/gemma4_mlx_taps.py), the driver by src/batch/golden_tap.cpp (or
tests/gemma4_forward_test.cpp). Identical names, identical shapes, so the
comparison is name-by-name.

Usage: cosine_bisect.py <ref_dir> <metal_dir> [--row N] [--family qwen3_5|gemma4]
"""
import sys
import os
import numpy as np

# The order a tap is produced in, so "first diverging" means first in execution
# order rather than first alphabetically.
GDN_ORDER = ["attn_norm", "gdn_in_qkv", "gdn_in_z", "gdn_in_a", "gdn_in_b",
             "gdn_core", "gdn_out", "attn_resid"]
ATTN_ORDER = ["attn_norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
              "rope_q", "rope_k", "sdpa", "attn_gated", "o_proj", "attn_resid"]
MLP_ORDER = ["ffn_norm", "gate_proj", "up_proj", "swiglu", "down_proj", "layer_out"]

# gemma4: the norm sandwich, the per-layer-embedding residual and the layer
# scalar are all extra, and the whole PLE stream runs once before the stack.
G4_PRE = ["embed", "ple_tok", "ple_proj", "ple_projnorm", "ple"]
G4_LAYER = ["attn_norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
            "v_norm", "rope_q", "rope_k", "sdpa", "o_proj", "attn_postnorm",
            "attn_resid", "ffn_norm", "gate_proj", "up_proj", "geglu",
            "down_proj", "ffn_postnorm", "ffn_resid", "ple_gate", "ple_act",
            "ple_back", "ple_norm", "ple_resid", "layer_out"]
G4_TAIL = ["final_norm", "logits_raw", "logits"]


def taps_qwen3_5(n_layers=24):
    taps = [(-1, "embed")]
    for L in range(n_layers):
        block = ATTN_ORDER if (L % 4) == 3 else GDN_ORDER
        taps += [(L, k) for k in block] + [(L, k) for k in MLP_ORDER]
    return taps + [(-1, "final_norm"), (-1, "logits")]


def taps_gemma4(n_layers=35):
    taps = [(-1, k) for k in G4_PRE]
    for L in range(n_layers):
        taps += [(L, k) for k in G4_LAYER]
    return taps + [(-1, k) for k in G4_TAIL]


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ref_dir, metal_dir = sys.argv[1], sys.argv[2]
    row = None
    if "--row" in sys.argv:
        row = int(sys.argv[sys.argv.index("--row") + 1])
    family = "qwen3_5"
    if "--family" in sys.argv:
        family = sys.argv[sys.argv.index("--family") + 1]

    taps = taps_gemma4() if family == "gemma4" else taps_qwen3_5()

    print(f"{'tap':<22} {'cos':>10} {'rel_l2':>10} {'ref_rms':>10} {'mtl_rms':>10}")
    print("-" * 66)
    first_bad = None
    for layer, name in taps:
        stem = name if layer < 0 else f"{layer}.{name}"
        rp = os.path.join(ref_dir, stem + ".npy")
        mp = os.path.join(metal_dir, stem + ".npy")
        if not (os.path.exists(rp) and os.path.exists(mp)):
            continue
        r, m = np.load(rp), np.load(mp)
        r = r.reshape(r.shape[0], -1) if r.ndim > 1 else r.reshape(1, -1)
        m = m.reshape(m.shape[0], -1) if m.ndim > 1 else m.reshape(1, -1)
        n = min(r.shape[0], m.shape[0])
        if row is not None:
            r, m = r[row:row + 1], m[row:row + 1]
        else:
            r, m = r[:n], m[:n]
        if r.shape[1] != m.shape[1]:
            print(f"{stem:<22} SHAPE {r.shape} vs {m.shape}")
            continue
        c = cosine(r, m)
        rel = float(np.linalg.norm((r - m).astype(np.float64)) /
                    max(np.linalg.norm(r.astype(np.float64)), 1e-30))
        rr = float(np.sqrt(np.mean(r.astype(np.float64) ** 2)))
        mr = float(np.sqrt(np.mean(m.astype(np.float64) ** 2)))
        flag = "" if c > 0.99 else "   <-- DIVERGES"
        if c <= 0.99 and first_bad is None:
            first_bad = stem
        print(f"{stem:<22} {c:>10.6f} {rel:>10.4f} {rr:>10.4f} {mr:>10.4f}{flag}")
    print("-" * 66)
    print("first diverging tap:", first_bad or "none (all cos > 0.99)")


if __name__ == "__main__":
    main()
