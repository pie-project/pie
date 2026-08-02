#!/usr/bin/env python3
"""Localize the first diverging (kernel, layer) between the MLX reference taps
and the raw-Metal driver taps.

Both trees are written by the PIE_METAL_GOLDEN_DIR hook: the reference by
tests/mlx/model/qwen3_5.cpp's dump_kernel (or, for gemma4, by
tests/parity/gemma4_mlx_taps.py), the driver by src/batch/golden_tap.cpp (or
tests/gemma4_forward_test.cpp). Identical names, identical shapes, so the
comparison is name-by-name.

Usage: cosine_bisect.py <ref_dir> <metal_dir> [--row N] [--family qwen3_5|gemma4|gptoss|llama]
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
# `attn_postnorm`/`ffn_postnorm`/`ple_norm`/`ple_resid` are reference-only: the
# driver fuses each closing norm into the residual add that always follows it,
# so those intermediates are no longer tensors anything can observe. Taps only
# one side emits are skipped.
G4_LAYER = ["attn_norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
            "v_norm", "rope_q", "rope_k", "sdpa", "o_proj", "attn_postnorm",
            "attn_resid", "ffn_norm", "gate_proj", "up_proj", "geglu",
            "down_proj", "ffn_postnorm", "ffn_resid", "ple_gate", "ple_act",
            "ple_back", "ple_norm", "ple_resid", "layer_out"]
G4_TAIL = ["final_norm", "logits_raw", "logits"]

# gpt-oss: sparse MoE every layer, attention sinks, no QK-norm.
GO_LAYER = ["attn_norm", "q_proj", "k_proj", "v_proj", "rope_q", "rope_k",
            "sdpa", "o_proj", "attn_resid", "ffn_norm", "router",
            "expert_gate", "expert_up", "expert_act", "expert_out", "moe",
            "layer_out"]


# llama/mistral/qwen2/qwen3/qwen3-moe: one plain decoder block, with two
# independent options. `q_norm`/`k_norm` appear only on qwen3, and the FFN is
# either dense or routed -- so both spellings are listed and whichever the model
# did not emit is skipped, the same way the gemma4 order carries taps only the
# reference produces.
LL_LAYER = ["attn_norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
            "rope_q", "rope_k", "sdpa", "o_proj", "attn_out", "ffn_norm",
            "gate_proj", "up_proj", "ffn_act", "ffn_out",
            "router", "expert_gate", "expert_up", "expert_act", "expert_out",
            "moe", "layer_out"]


def taps_llama(n_layers=48):
    taps = [(-1, "embed")]
    for L in range(n_layers):
        taps += [(L, k) for k in LL_LAYER]
    return taps + [(-1, "final_norm"), (-1, "logits")]


def taps_qwen3_5(n_layers=24):
    taps = [(-1, "embed")]
    for L in range(n_layers):
        block = ATTN_ORDER if (L % 4) == 3 else GDN_ORDER
        taps += [(L, k) for k in block] + [(L, k) for k in MLP_ORDER]
    return taps + [(-1, "final_norm"), (-1, "logits")]


def taps_gptoss(n_layers=24):
    taps = [(-1, "embed")]
    for L in range(n_layers):
        taps += [(L, k) for k in GO_LAYER]
    return taps + [(-1, "final_norm"), (-1, "logits")]


def taps_gemma4(n_layers=35):
    taps = [(-1, k) for k in G4_PRE]
    for L in range(n_layers):
        taps += [(L, k) for k in G4_LAYER]
    return taps + [(-1, k) for k in G4_TAIL]


# The four values a routed layer keeps ONE PER SLOT. Their slot order is the
# router's tie-breaking, and two implementations resolve a near-tie differently
# -- on Qwen3-30B-A3B the top two experts can sit 0.002 apart, inside the
# routers' own disagreement. That reorders the stacks while selecting the SAME
# experts, so `moe` and everything after it are unaffected.
#
# Comparing these in slot order would report a divergence that is entirely the
# ordering, so they are matched slot to slot first. A genuinely different
# SELECTION still shows, because then some slot has no counterpart to match.
SLOT_TAPS = ("expert_gate", "expert_up", "expert_act", "expert_out")


def align_slots(r, m, k):
    """Match `m`'s k slots per row onto `r`'s and drop any that have no
    counterpart, returning how many were dropped.

    A slot either matches its counterpart at ~0.999 or scores ~0.3 against
    everything -- two orders of magnitude apart -- so 0.9 separates them with
    nothing near the line. An unmatched slot means the two routers chose a
    DIFFERENT expert for it, which at the top-k boundary is a near-tie on the
    smallest of the k weights; the caller reports the count rather than hiding
    it, and `moe` is still compared in full.
    """
    rows = min(r.shape[0], m.shape[0])
    rr = r[:rows].reshape(rows, k, -1)
    mm = m[:rows].reshape(rows, k, -1)
    keep_r, keep_m, dropped = [], [], 0
    for i in range(rows):
        a = rr[i] / (np.linalg.norm(rr[i], axis=1, keepdims=True) + 1e-30)
        b = mm[i] / (np.linalg.norm(mm[i], axis=1, keepdims=True) + 1e-30)
        c = a @ b.T
        used_r, used_m = set(), set()
        pairs = sorted(((c[x, y], x, y) for x in range(k) for y in range(k)),
                       key=lambda t: -t[0])
        row_r, row_m = [], []
        for score, x, y in pairs:
            if x in used_r or y in used_m:
                continue
            used_r.add(x)
            used_m.add(y)
            if score > 0.9:
                row_r.append(rr[i][x])
                row_m.append(mm[i][y])
            else:
                dropped += 1
        keep_r.append(np.concatenate(row_r) if row_r else np.zeros(0, dtype=rr.dtype))
        keep_m.append(np.concatenate(row_m) if row_m else np.zeros(0, dtype=mm.dtype))
    width = min(min(len(x) for x in keep_r), min(len(x) for x in keep_m))
    r2 = np.stack([x[:width] for x in keep_r])
    m2 = np.stack([x[:width] for x in keep_m])
    return r2, m2, dropped


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

    if family == "gemma4":
        taps = taps_gemma4()
    elif family == "gptoss":
        taps = taps_gptoss()
    elif family == "llama":
        taps = taps_llama()
    else:
        taps = taps_qwen3_5()

    # How many experts a token routes to, read off the tensors rather than
    # configured: `expert_out` is k stacked hidden-wide rows and `moe` is their
    # weighted sum, so the ratio IS k.
    slots = 0
    for probe in taps:
        eo = os.path.join(metal_dir, f"{probe[0]}.expert_out.npy")
        mo = os.path.join(metal_dir, f"{probe[0]}.moe.npy")
        if os.path.exists(eo) and os.path.exists(mo):
            a, b = np.load(eo), np.load(mo)
            if b.shape[-1] > 0 and a.shape[-1] % b.shape[-1] == 0:
                slots = a.shape[-1] // b.shape[-1]
            break
    if slots > 1:
        print(f"(routed: {slots} slots per token, matched slot-to-slot)")

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
        note = ""
        if name in SLOT_TAPS and slots > 1 and r.shape[1] % slots == 0 and \
                m.shape[1] % slots == 0:
            r, m, dropped = align_slots(r, m, slots)
            if dropped:
                note = f"  [{dropped} slot(s) routed differently]"
        n = min(r.shape[0], m.shape[0])
        if row is not None:
            r, m = r[row:row + 1], m[row:row + 1]
        elif r.shape[0] != m.shape[0]:
            # A teacher-forced decode candidate has ONE row -- the step just
            # run -- against a reference that ran the whole prompt at once. The
            # rows that line up are the last of each; taking `[:n]` instead
            # would compare the reference's FIRST position, which at position 0
            # is the one place RoPE is the identity and so the one place this
            # comparison proves nothing.
            r, m = r[-1:], m[-1:]
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
        print(f"{stem:<22} {c:>10.6f} {rel:>10.4f} {rr:>10.4f} {mr:>10.4f}{flag}{note}")
    print("-" * 66)
    print("first diverging tap:", first_bad or "none (all cos > 0.99)")


if __name__ == "__main__":
    main()
