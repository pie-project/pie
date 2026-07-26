//! Deterministic backend projections of the canonical PTIR RNG contract.
//!
//! Both projections are printed from [`pie_ir::rng::RNG_FORMULA`], so the device
//! implementations cannot drift from the host one in [`pie_ir::rng`].

use core::fmt::Write;

use pie_ir::rng::RNG_FORMULA;

/// `UNIFORM_MAX` as a decimal literal for the generated backends. The nearest
/// `f32` to this text is exactly `UNIFORM_MAX`, so host and device agree bit
/// for bit.
const UNIFORM_MAX_LITERAL: &str = "0.99999994";
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
    let uniform_max = UNIFORM_MAX_LITERAL;
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
        "{inline} float ptir_rng_hash_uniform(\n    {u64_ty} seed_eff, {u32_ty} index) {{\n  const {u64_ty} x = seed_eff +\n      0x{:016X}{u64_suffix} * (({u64_ty})index + {}{u64_suffix});\n  const {u32_ty} bits =\n      ({u32_ty})(ptir_rng_splitmix64(x) >> {});\n  const float raw = ((float)bits + {:.1}f) * (1.0f / {denominator}.0f);\n  /* clamp off the one draw in 2^24 that rounds to exactly 1.0f, which would\n     make gumbel = -log(-log(u)) evaluate to +inf and hijack every argmax */\n  return raw < {uniform_max}f ? raw : {uniform_max}f;\n}}\n",
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
        "// rng_contract.generated.h — GENERATED from compiler/ir/src/rng.rs.\n\
// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-compiler-tests --test rng_contract\n\
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
        "// ptir_rng.generated.metal — GENERATED from compiler/ir/src/rng.rs.\n\
// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-compiler-tests --test rng_contract\n\
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
    let uniform_max = UNIFORM_MAX_LITERAL;
    writeln!(
        out,
        "inline float ptir_rng_hash_uniform(ulong seed_eff, uint index) {{\n  const ulong x = seed_eff +\n      0x{:016X}ul * (ulong(index) + {}ul);\n  const uint bits = uint(ptir_rng_splitmix64(x) >> {});\n  const float raw = (float(bits) + {:.1}f) * (1.0f / {denominator}.0f);\n  /* clamp off the one draw in 2^24 that rounds to exactly 1.0f */\n  return raw < {uniform_max}f ? raw : {uniform_max}f;\n}}\n",
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
