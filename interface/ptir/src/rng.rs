//! Canonical PTIR RNG contract and deterministic backend projections.

use alloc::format;
use alloc::string::String;
use core::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64Round {
    pub xor_shift: u32,
    pub multiplier: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RngFormula {
    pub splitmix64_rounds: [SplitMix64Round; 3],
    pub lane_stride: u64,
    pub lane_index_bias: u64,
    pub uniform_mantissa_shift: u32,
    pub uniform_mantissa_bits: u32,
    pub uniform_midpoint: f32,
    pub ambient_seed_xor: u64,
    pub keyed_word_bits: u32,
}

pub const RNG_FORMULA: RngFormula = RngFormula {
    splitmix64_rounds: [
        SplitMix64Round {
            xor_shift: 27,
            multiplier: Some(0x3C79_AC49_2BA7_B653),
        },
        SplitMix64Round {
            xor_shift: 33,
            multiplier: Some(0x1C69_B3F7_4AC4_AE35),
        },
        SplitMix64Round {
            xor_shift: 27,
            multiplier: None,
        },
    ],
    lane_stride: 0x9E37_79B9_7F4A_7C15,
    lane_index_bias: 1,
    uniform_mantissa_shift: 40,
    uniform_mantissa_bits: 24,
    uniform_midpoint: 0.5,
    ambient_seed_xor: 0xA5A5_A5A5,
    keyed_word_bits: 32,
};

#[inline]
pub fn splitmix64(mut value: u64) -> u64 {
    for round in RNG_FORMULA.splitmix64_rounds {
        value ^= value >> round.xor_shift;
        if let Some(multiplier) = round.multiplier {
            value = value.wrapping_mul(multiplier);
        }
    }
    value
}

#[inline]
pub fn seed_eff(seed: u32) -> u64 {
    seed as u64 ^ RNG_FORMULA.ambient_seed_xor
}

#[inline]
pub fn stream_salt(stream: u32) -> u64 {
    splitmix64((stream as u64).wrapping_mul(RNG_FORMULA.lane_stride))
}

#[inline]
pub fn seed_eff_stream(seed: u32, stream: u32) -> u64 {
    seed_eff(seed) ^ stream_salt(stream)
}

#[inline]
pub fn keyed_seed(key: u32, counter: u32) -> u64 {
    splitmix64(((key as u64) << RNG_FORMULA.keyed_word_bits) | counter as u64)
}

#[inline]
pub fn hash_uniform(seed_eff: u64, index: u32) -> f32 {
    let x = seed_eff.wrapping_add(
        RNG_FORMULA
            .lane_stride
            .wrapping_mul(index as u64 + RNG_FORMULA.lane_index_bias),
    );
    let bits = (splitmix64(x) >> RNG_FORMULA.uniform_mantissa_shift) as u32;
    let denominator = (1u32 << RNG_FORMULA.uniform_mantissa_bits) as f32;
    (bits as f32 + RNG_FORMULA.uniform_midpoint) * (1.0 / denominator)
}

enum CudaProjection {
    Header,
    Source,
}

fn render_cuda_functions(projection: CudaProjection) -> String {
    let mut out = String::new();
    let (inline, u64_ty, u32_ty, u64_suffix) = match projection {
        CudaProjection::Header => ("PTIR_RNG_INLINE", "uint64_t", "uint32_t", "ULL"),
        CudaProjection::Source => (
            "__device__ __forceinline__",
            "unsigned long long",
            "unsigned int",
            "ULL",
        ),
    };
    let denominator = 1u64 << RNG_FORMULA.uniform_mantissa_bits;
    writeln!(out, "{inline} {u64_ty} ptir_rng_splitmix64({u64_ty} x) {{").unwrap();
    for round in RNG_FORMULA.splitmix64_rounds {
        writeln!(out, "  x ^= x >> {};", round.xor_shift).unwrap();
        if let Some(multiplier) = round.multiplier {
            writeln!(out, "  x *= 0x{multiplier:016X}{u64_suffix};").unwrap();
        }
    }
    out.push_str("  return x;\n}\n");
    writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_seed_eff({u32_ty} seed) {{\n  return ({u64_ty})seed ^ 0x{:016X}{u64_suffix};\n}}",
        RNG_FORMULA.ambient_seed_xor
    )
    .unwrap();
    writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_stream_salt({u32_ty} stream) {{\n  return ptir_rng_splitmix64(\n      ({u64_ty})stream * 0x{:016X}{u64_suffix});\n}}",
        RNG_FORMULA.lane_stride
    )
    .unwrap();
    writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_seed_eff_stream(\n    {u32_ty} seed, {u32_ty} stream) {{\n  return ptir_rng_seed_eff(seed) ^ ptir_rng_stream_salt(stream);\n}}"
    )
    .unwrap();
    writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_keyed_seed(\n    {u32_ty} key, {u32_ty} counter) {{\n  return ptir_rng_splitmix64(\n      (({u64_ty})key << {}) | ({u64_ty})counter);\n}}",
        RNG_FORMULA.keyed_word_bits
    )
    .unwrap();
    writeln!(
        out,
        "{inline} float ptir_rng_hash_uniform(\n    {u64_ty} seed_eff, {u32_ty} index) {{\n  const {u64_ty} x = seed_eff +\n      0x{:016X}{u64_suffix} * (({u64_ty})index + {}{u64_suffix});\n  const {u32_ty} bits =\n      ({u32_ty})(ptir_rng_splitmix64(x) >> {});\n  return ((float)bits + {:.1}f) * (1.0f / {denominator}.0f);\n}}\n",
        RNG_FORMULA.lane_stride,
        RNG_FORMULA.lane_index_bias,
        RNG_FORMULA.uniform_mantissa_shift,
        RNG_FORMULA.uniform_midpoint
    )
    .unwrap();
    out
}

pub fn generate_cuda_header() -> String {
    let implementation = render_cuda_functions(CudaProjection::Header);
    let source = render_cuda_functions(CudaProjection::Source);
    format!(
        "// rng_contract.generated.h — GENERATED from interface/ptir/src/rng.rs.\n\
// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-ptir --test rng_contract\n\
#pragma once\n\
#include <stdint.h>\n\
\n\
#if defined(__CUDACC__)\n\
#define PTIR_RNG_INLINE static __host__ __device__ __forceinline__\n\
#else\n\
#define PTIR_RNG_INLINE static inline\n\
#endif\n\
\n\
{implementation}\
#undef PTIR_RNG_INLINE\n\
\n\
#ifdef __cplusplus\n\
inline constexpr char PTIR_RNG_CUDA_PREAMBLE[] = R\"PTIR_RNG_CUDA(\n\
{source}\
)PTIR_RNG_CUDA\";\n\
#endif\n"
    )
}

pub fn generate_msl_preamble() -> String {
    let mut out = String::from(
        "// ptir_rng.generated.metal — GENERATED from interface/ptir/src/rng.rs.\n\
// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-ptir --test rng_contract\n\
#ifndef PIE_PTIR_RNG_GENERATED_METAL\n\
#define PIE_PTIR_RNG_GENERATED_METAL\n\
\n\
inline ulong ptir_rng_splitmix64(ulong x) {\n",
    );
    for round in RNG_FORMULA.splitmix64_rounds {
        writeln!(out, "  x ^= x >> {};", round.xor_shift).unwrap();
        if let Some(multiplier) = round.multiplier {
            writeln!(out, "  x *= 0x{multiplier:016X}ul;").unwrap();
        }
    }
    out.push_str("  return x;\n}\n");
    writeln!(
        out,
        "inline ulong ptir_rng_seed_eff(uint seed) {{\n  return ulong(seed) ^ 0x{:016X}ul;\n}}",
        RNG_FORMULA.ambient_seed_xor
    )
    .unwrap();
    writeln!(
        out,
        "inline ulong ptir_rng_stream_salt(uint stream) {{\n  return ptir_rng_splitmix64(\n      ulong(stream) * 0x{:016X}ul);\n}}",
        RNG_FORMULA.lane_stride
    )
    .unwrap();
    out.push_str(
        "inline ulong ptir_rng_seed_eff_stream(uint seed, uint stream) {\n  return ptir_rng_seed_eff(seed) ^ ptir_rng_stream_salt(stream);\n}\n",
    );
    writeln!(
        out,
        "inline ulong ptir_rng_keyed_seed(uint key, uint counter) {{\n  return ptir_rng_splitmix64(\n      (ulong(key) << {}) | ulong(counter));\n}}",
        RNG_FORMULA.keyed_word_bits
    )
    .unwrap();
    let denominator = 1u64 << RNG_FORMULA.uniform_mantissa_bits;
    writeln!(
        out,
        "inline float ptir_rng_hash_uniform(ulong seed_eff, uint index) {{\n  const ulong x = seed_eff +\n      0x{:016X}ul * (ulong(index) + {}ul);\n  const uint bits = uint(ptir_rng_splitmix64(x) >> {});\n  return (float(bits) + {:.1}f) * (1.0f / {denominator}.0f);\n}}\n",
        RNG_FORMULA.lane_stride,
        RNG_FORMULA.lane_index_bias,
        RNG_FORMULA.uniform_mantissa_shift,
        RNG_FORMULA.uniform_midpoint
    )
    .unwrap();
    out.push_str("#endif\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projections_are_deterministic() {
        assert_eq!(generate_cuda_header(), generate_cuda_header());
        assert_eq!(generate_msl_preamble(), generate_msl_preamble());
    }
}
