#!/usr/bin/env python3
"""Regenerate driver/cuda/tests/rope_vllm_cos_sin_golden.hpp.

WHY THIS IS A SCRIPT AND NOT A HAND-EDITED HEADER
=================================================

The golden slice has now been blind twice, the same way both times: it sampled
positions that structurally could not contain the live defect, and passed with
zero tolerance while proving nothing.

  * v1 sampled 13 positions. Not one of its 832 entries lay within even 3 fp32
    ulp of a bf16 rounding midpoint -- and a bf16 table can only disagree near a
    midpoint. It could not fail.
  * v2 added the four positions where the DEVICE-trig implementation differed
    (4498, 7460, 13848, 21467). Moving the table build onto the host fixed all
    four, and moved the residual to entirely different positions. A slice
    holding only the old four was blind again, one iteration later.

The lesson is that the interesting positions are a property of the CURRENT
implementation pair, not a constant. So they are an INPUT here. Re-deriving on
a new reference host is a rerun of this script with new arguments, not an edit
of a generated file.

THE RESIDUAL IS ISA-DEPENDENT -- PASS THE ISA YOU MEASURED ON
=============================================================

100% of the remaining gap is the trig call itself: inv_freq and every angle are
bit-identical between the two sides. torch's fp32 trig is SLEEF, bit-for-bit,
and WHICH SLEEF depends on the host:

  * x86_64 with MKL -> Intel MKL VML, NOT SLEEF. `ATen/cpu/vml.h` specialises
                       vcos/vsin onto MKL whenever AT_MKL_ENABLED() && !__APPLE__,
                       so Vectorized<float>::cos() is never reached. Measured:
                       vmsCos/vmsSin with torch's mode word
                       VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE (0x140102)
                       reproduce torch bit-for-bit, 0 diffs / 960,032.
  * arm64 / Apple   -> SLEEF u10, because those builds have USE_MKL=OFF and
                       __APPLE__.

These are different libraries, not variants, and their residuals sit at
different entries -- a slice derived on the wrong one cannot fail. Record what
you measured with --reference-isa; the test prints it on every run.

AND NOTE WHY THIS CANNOT BE "FIXED" IN THE HOST ARITHMETIC
==========================================================

In principle correct rounding need not be the goal -- MKL VML HA is itself
~0.60 ulp and not correctly rounded, so a similarly imperfect implementation
could land on the reference's side of a bf16 midpoint more often than a perfect
one. Measured against the real MKL reference, it does not:

    fraction of angles whose fp32 value differs from the reference
    double -> round to fp32  (correctly rounded) : 5.205%
    plain cosf/sinf                              : 5.297%

The driver therefore DEFAULTS to the correctly rounded build, which is both the
closer of the two and deterministic across C libraries;
PIE_DEBUG_ROPE_VLLM_TABLE_TRIG=libm selects the other.

A claim that cosf/sinf scores 0 mismatches in the campaign window and 1 overall
is NOT true and should not be re-derived. It came from misreading an artifact
row, "PIE vs libm cosf/sinf -- bf16 differs 1", which measures the distance
between our own two backends rather than glibc against the reference. Against
the reference cosf/sinf misses 18 entries, including 13852, the one in-window
entry. A follow-up explanation that the difference tracks the glibc version is
also false: tested directly in ubuntu:22.04 (glibc 2.35) and debian:13 (glibc
2.41), the per-entry output is BYTE-IDENTICAL.

None of this changes the fixture. The base construction below is the
correctly-rounded one and the overrides are measured against IT, so the fixture
equals the reference regardless of which backend the driver happens to use.

NOTE THE BASE CONSTRUCTION HERE IS DELIBERATELY THE CORRECTLY-ROUNDED ONE, and
is NOT required to match the driver. What matters is that base + overrides
equals the REFERENCE over the surveyed range. The override list must therefore
correspond to whatever this script computes, not to whatever the driver
computes; mixing those up is how the fixture became circular once already.

USAGE
=====

    python3 gen_rope_vllm_golden.py \
        --reference-isa x86_64-avx2-sleef-u35 \
        --survey-max 30000 \
        --mismatch 3269:15:sin:0x3f7d \
        --regression 4498:8:cos --regression 7460:16:sin \
        --regression 13848:7:cos --regression 21467:6:cos \
        --agree 100:1:sin --agree 3411:18:sin \
        --out rope_vllm_cos_sin_golden.hpp

`--override` entries are where the reference and this script's construction are
MEASURED to differ, carrying the REFERENCE's bf16 bits; they are what stops the
slice being decorative, and what makes it a slice of the reference at all. `--regression`
entries are defects already fixed, kept so a fix cannot silently come undone.
`--agree` entries sit equally close to a midpoint but match, so the check
discriminates rather than merely recording known failures.
"""

import argparse
import numpy as np

THETA = 1e7
ROTARY_DIM = 64
ANGLES = ROTARY_DIM // 2

# A spread across every band the campaign measured, independent of any
# particular residual: gsm8k prompt lengths, the mid range, the agentic corpus,
# and past 20000.
BASE_POSITIONS = [0, 7, 137, 199, 1000, 2731, 3999,
                  13000, 16384, 19999, 20000, 24000, 29999]


def parse_entry(spec, with_bits=False):
    """'3269:15:sin' -> (3269, 47).  Entry index is cos=lane, sin=32+lane.

    With `with_bits`, the spec carries the REFERENCE's bf16 bit pattern too:
    '3269:15:sin:0x3f7d' -> (3269, 47, 0x3f7d).
    """
    parts = spec.split(":")
    want = 4 if with_bits else 3
    if len(parts) != want:
        raise SystemExit(
            f"bad entry spec {spec!r}, want POS:LANE:cos|sin"
            + (":REFBITS" if with_bits else ""))
    try:
        pos, lane = int(parts[0]), int(parts[1])
    except ValueError:
        raise SystemExit(f"bad position/lane in {spec!r}")
    kind = parts[2]
    if kind not in ("cos", "sin"):
        raise SystemExit(f"bad kind {kind!r} in {spec!r}, want cos or sin")
    if not 0 <= lane < ANGLES:
        raise SystemExit(f"lane {lane} out of range in {spec!r}")
    entry = lane if kind == "cos" else ANGLES + lane
    if not with_bits:
        return pos, entry
    try:
        bits = int(parts[3], 0)
    except ValueError:
        raise SystemExit(f"bad reference bits {parts[3]!r} in {spec!r}")
    if not 0 <= bits <= 0xFFFF:
        raise SystemExit(f"reference bits out of range in {spec!r}")
    return pos, entry, bits


def inv_freq():
    """vLLM: 1.0 / (base ** (arange(0, rotary_dim, 2).float() / rotary_dim)).

    Exponent formed in fp32; the power taken first and the reciprocal AFTER it.
    This is bit-identical to the reference and is not the source of any
    residual -- verified, 0 differences over the full angle population.
    """
    j = np.arange(0, ROTARY_DIM, 2, dtype=np.float32)
    expo = (j / np.float32(ROTARY_DIM)).astype(np.float32)
    p32 = (np.float64(THETA) ** expo.astype(np.float64)).astype(np.float32)
    return (np.float32(1.0) / p32).astype(np.float32)


def to_bf16(x32):
    b = x32.astype(np.float32).view(np.uint32).astype(np.uint64)
    return (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def midpoint_distance(v):
    """Signed fractional offset of the exact value from the nearest bf16
    rounding midpoint, in fp32 ulp. Measured on the magnitude, because bf16
    rounding is sign-symmetric."""
    av = np.abs(v)
    x32 = av.astype(np.float32)
    bits = x32.view(np.uint32).astype(np.int64)
    low16 = bits & 0xFFFF
    expo_bits = (bits >> 23) & 0xFF
    ulp = np.ldexp(1.0, (expo_bits - 127 - 23).astype(np.int64))
    return (low16 + (av - x32.astype(np.float64)) / ulp) - 0x8000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-isa", required=True,
                    help="ISA the residual was measured on, e.g. x86_64-avx2-sleef-u35")
    ap.add_argument("--survey-max", type=int, required=True,
                    help="highest position the residual survey covered")
    ap.add_argument("--override", action="append", default=[],
                    metavar="POS:LANE:KIND:REFBITS",
                    help="entry MEASURED to differ, with the REFERENCE's bf16 bits")
    ap.add_argument("--regression", action="append", default=[], metavar="POS:LANE:KIND",
                    help="entry that used to differ and must now match")
    ap.add_argument("--agree", action="append", default=[], metavar="POS:LANE:KIND",
                    help="boundary-adjacent entry that matches, as a discriminator")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    overrides = [parse_entry(s, with_bits=True) for s in args.override]
    regressions = [parse_entry(s) for s in args.regression]
    agrees = [parse_entry(s) for s in args.agree]

    positions = sorted(set(BASE_POSITIONS)
                       | {m[0] for m in overrides}
                       | {p for p, _ in regressions}
                       | {p for p, _ in agrees})
    row_of = {p: i for i, p in enumerate(positions)}

    # The table below is built with PIE's construction (trig in double, rounded
    # to fp32), then OVERRIDDEN at every entry measured to differ. That yields
    # the reference's bits exactly -- but only if the survey actually covered
    # every position sampled here. A position past the survey could silently
    # carry Pie's value where the reference differs, which would be a false
    # green of exactly the kind this file keeps producing.
    beyond = [p for p in positions if p > args.survey_max]
    if beyond:
        raise SystemExit(
            f"positions {beyond} lie past the surveyed range "
            f"(--survey-max {args.survey_max}). The generated bits there would "
            f"be this implementation's, not the reference's, and any agreement "
            f"would be circular. Extend the survey or drop those positions.")

    ifr = inv_freq()
    assert ifr[0] == np.float32(1.0), "inv_freq[0] must be exactly 1.0f at this shape"

    rows, near = [], []
    for r, p in enumerate(positions):
        ang = (np.float32(p) * ifr).astype(np.float32)
        a64 = ang.astype(np.float64)
        v = np.concatenate([np.cos(a64), np.sin(a64)])
        dist = midpoint_distance(v)
        bits = to_bf16(v.astype(np.float32))
        # Override with the reference's measured bits wherever the two differ.
        for mp, me, mbits in overrides:
            if mp == p:
                if int(bits[me]) == mbits:
                    raise SystemExit(
                        f"--override {mp}:{me} claims the reference differs, but "
                        f"this construction already produces 0x{mbits:04x} there. "
                        f"Either the measurement is stale or the ISA is wrong.")
                bits[me] = np.uint16(mbits)
        rows.append((p, bits))
        for e in range(ROTARY_DIM):
            if abs(dist[e]) <= 4.0:
                near.append((r, e, float(dist[e]), p))

    within1 = sum(1 for n in near if abs(n[2]) <= 1.0)
    out = []
    w = out.append

    w("#pragma once")
    w("")
    w("// GENERATED by driver/cuda/tests/gen_rope_vllm_golden.py -- do not hand-edit.")
    w("// Re-derive with that script when the reference host or its ISA changes;")
    w("// the interesting positions are a property of the current implementation")
    w("// pair, not a constant. See the script's header for why this slice has")
    w("// been blind twice and what stops it happening again.")
    w("//")
    w("// A slice of vLLM's `cos_sin_cache` for the Qwen3.5 partial-rotary shape:")
    w("//   rotary_dim = 64 (head_dim 256 x partial_rotary_factor 0.25), theta = 1e7.")
    w("//")
    w("// Built by the construction vLLM v0.27.1 performs once at init:")
    w("//   inv_freq = 1.0 / (base ** (arange(0, rotary_dim, 2).float() / rotary_dim))")
    w("//   freqs    = einsum(\"i,j->ij\", arange(max_pos).float(), inv_freq)")
    w("//   cache    = cat((freqs.cos(), freqs.sin()), dim=-1).to(bfloat16)")
    w("//")
    w("// Each row is one position: entries [0,32) are cos, [32,64) are sin, as")
    w("// bf16 bit patterns. THESE BITS ARE THE CONTRACT -- a rounded, lossy table")
    w("// on purpose, because that is what the reference engine indexes.")
    w("//")
    w("// inv_freq[0] is EXACTLY 1.0f here, so lane 0's angle in radians is the")
    w("// token position itself.")
    w("")
    w("#include <cstdint>")
    w("")
    w("namespace pie_rope_golden {")
    w("")
    w(f"constexpr int kRotaryDim = {ROTARY_DIM};")
    w(f"constexpr int kAngles = {ANGLES};")
    w(f"constexpr float kTheta = {THETA:.1f}f;")
    w("")
    w("// The ISA the residual below was measured on. torch's fp32 trig is SLEEF")
    w("// bit-for-bit, and which SLEEF depends on the host: arm64 uses the 1.0 ulp")
    w("// u10 kernels, x86_64 the 3.5 ulp u35 AVX2/AVX512 ones. They do NOT share a")
    w("// residual, so a slice derived on the wrong ISA is a slice that cannot fail.")
    w("// The test prints this on every run.")
    w(f"constexpr char kReferenceIsa[] = \"{args.reference_isa}\";")
    w(f"constexpr int kResidualSurveyMax = {args.survey_max};")
    w("")
    w("struct Row {")
    w("    std::int32_t pos;")
    w(f"    std::uint16_t entry[{ROTARY_DIM}];  // [0,32) cos, [32,64) sin")
    w("};")
    w("")
    w("constexpr Row kCosSinCache[] = {")
    for p, e in rows:
        w(f"    {{{p}, {{")
        for i in range(0, ROTARY_DIM, 8):
            w("        " + ", ".join(f"0x{v:04x}" for v in e[i:i + 8]) + ",")
        w("    }},")
    w("};")
    w("")
    w(f"constexpr int kRows = {len(positions)};")
    w("")
    w("struct Entry { int row; int entry; };")
    w("")
    w("// Entries where this script's construction was MEASURED to differ from the")
    w("// reference, replaced here by the REFERENCE's bits. These are what make this")
    w("// a slice OF THE REFERENCE rather than of ourselves, and what gives it teeth:")
    w("// any implementation that drifts toward correct rounding lights them up. They")
    w("// are NOT an allowance -- the parity check still requires zero mismatches.")
    w("constexpr Entry kReferenceOverrides[] = {")
    if not overrides:
        w("    // none recorded")
    for p, e, bits in sorted(overrides):
        kind, lane = ("cos", e) if e < ANGLES else ("sin", e - ANGLES)
        w(f"    {{{row_of[p]:2d}, {e:2d}}},  // pos={p} {kind}[{lane}]"
          f" reference=0x{bits:04x}")
    w("};")
    w(f"constexpr int kReferenceOverrideCount = {len(overrides)};")
    w("")
    w("// Entries that USED to differ and must now match. Kept so a fix cannot")
    w("// silently come undone.")
    w("constexpr Entry kRegressionGuards[] = {")
    if not regressions:
        w("    // none recorded")
    for p, e in sorted(regressions):
        kind, lane = ("cos", e) if e < ANGLES else ("sin", e - ANGLES)
        w(f"    {{{row_of[p]:2d}, {e:2d}}},  // pos={p} {kind}[{lane}]")
    w("};")
    w(f"constexpr int kRegressionGuardCount = {len(regressions)};")
    w("")
    w("// Entries within 4 fp32 ulp of a bf16 rounding midpoint. `ulp_dist` is the")
    w("// signed fractional offset of the exact value from the midpoint, in fp32")
    w("// ulp, measured on the magnitude.")
    w("struct NearMidpoint { int row; int entry; float ulp_dist; };")
    w("")
    w("constexpr NearMidpoint kNearMidpoints[] = {")
    for r, e, d, p in near:
        kind, lane = ("cos", e) if e < ANGLES else ("sin", e - ANGLES)
        w(f"    {{{r:2d}, {e:2d}, {d:+.4f}f}},  // pos={p} {kind}[{lane}]")
    w("};")
    w(f"constexpr int kNearMidpointCount = {len(near)};")
    w(f"constexpr int kEntriesWithinOneUlp = {within1};")
    w("")
    w("}  // namespace pie_rope_golden")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")

    print(f"wrote {args.out}")
    print(f"  reference ISA        : {args.reference_isa}")
    print(f"  rows / entries       : {len(positions)} / {len(positions) * ROTARY_DIM}")
    print(f"  reference overrides  : {len(overrides)}")
    print(f"  regression guards    : {len(regressions)}")
    print(f"  near-midpoint <=4ulp : {len(near)}  (within 1 ulp: {within1})")
    if not overrides:
        print("  WARNING: no reference overrides -- the slice is our own construction,")
        print("           compared against itself, and cannot fail.")


if __name__ == "__main__":
    main()
