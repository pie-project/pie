//! Deterministic backend projections of the canonical PTIR RNG contract.
//!
//! Both projections are printed from [`tensor_ir::rng::RNG_FORMULA`], so the device
//! implementations cannot drift from the host one in [`tensor_ir::rng`].

use core::fmt::Write;

use tensor_ir::rng::RNG_FORMULA;

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
    let _ = writeln!(out, "{inline} {u64_ty} ptir_rng_splitmix64({u64_ty} x) {{");
    for round in RNG_FORMULA.splitmix64_rounds {
        let _ = writeln!(out, "  x ^= x >> {};", round.xor_shift);
        if let Some(multiplier) = round.multiplier {
            let _ = writeln!(out, "  x *= 0x{multiplier:016X}{u64_suffix};");
        }
    }
    out.push_str("  return x;\n}\n");
    let _ = writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_seed_eff({u32_ty} seed) {{\n  return ({u64_ty})seed ^ 0x{:016X}{u64_suffix};\n}}",
        RNG_FORMULA.ambient_seed_xor
    );
    let _ = writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_stream_salt({u32_ty} stream) {{\n  return ptir_rng_splitmix64(\n      ({u64_ty})stream * 0x{:016X}{u64_suffix});\n}}",
        RNG_FORMULA.lane_stride
    );
    let _ = writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_seed_eff_stream(\n    {u32_ty} seed, {u32_ty} stream) {{\n  return ptir_rng_seed_eff(seed) ^ ptir_rng_stream_salt(stream);\n}}"
    );
    let _ = writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_keyed_seed(\n    {u32_ty} key, {u32_ty} counter) {{\n  return ptir_rng_splitmix64(\n      (({u64_ty})key << {}) | ({u64_ty})counter);\n}}",
        RNG_FORMULA.keyed_word_bits
    );
    let _ = writeln!(
        out,
        "{inline} float ptir_rng_hash_uniform(\n    {u64_ty} seed_eff, {u32_ty} index) {{\n  const {u64_ty} x = seed_eff +\n      0x{:016X}{u64_suffix} * (({u64_ty})index + {}{u64_suffix});\n  const {u32_ty} bits =\n      ({u32_ty})(ptir_rng_splitmix64(x) >> {});\n  const float raw = ((float)bits + {:.1}f) * (1.0f / {denominator}.0f);\n  /* clamp off the one draw in 2^24 that rounds to exactly 1.0f, which would\n     make gumbel = -log(-log(u)) evaluate to +inf and hijack every argmax */\n  return raw < {uniform_max}f ? raw : {uniform_max}f;\n}}\n",
        RNG_FORMULA.lane_stride,
        RNG_FORMULA.lane_index_bias,
        RNG_FORMULA.uniform_mantissa_shift,
        RNG_FORMULA.uniform_midpoint
    );
    out
}

/// The `__device__` projection, as spliced into emitted CUDA sources.
///
/// [`generate_cuda_header`] embeds the same text in a raw string literal for
/// the C++ side. Both call this; neither recovers the text by generating the
/// header and slicing the literal back out of it. That shortcut makes the
/// emitter depend on the header's punctuation — any whitespace change around
/// the raw-string delimiters turns into a runtime failure in a function whose
/// job has nothing to do with headers.
pub fn cuda_device_functions() -> String {
    render_cuda_functions(CudaProjection::Source)
}

/// Renders the `rng_contract.generated.h` C header — the RNG device functions
/// (guarded for CUDA or plain host inlining) plus the `PTIR_RNG_CUDA_PREAMBLE`
/// constant NVRTC splices into emitted sources.
pub fn generate_cuda_header() -> String {
    let implementation = render_cuda_functions(CudaProjection::Header);
    let source = cuda_device_functions();
    format!(
        "// rng_contract.generated.h — GENERATED from crates/tensor-ir/src/rng.rs.\n\
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

/// Renders the `ptir_rng.generated.metal` preamble — the same RNG contract in
/// MSL, wrapped in an include guard.
pub fn generate_msl_preamble() -> String {
    let mut out = String::from(
        "// ptir_rng.generated.metal — GENERATED from crates/tensor-ir/src/rng.rs.\n\
// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-compiler-tests --test rng_contract\n\
#ifndef PIE_PTIR_RNG_GENERATED_METAL\n\
#define PIE_PTIR_RNG_GENERATED_METAL\n\
\n\
inline ulong ptir_rng_splitmix64(ulong x) {\n",
    );
    for round in RNG_FORMULA.splitmix64_rounds {
        let _ = writeln!(out, "  x ^= x >> {};", round.xor_shift);
        if let Some(multiplier) = round.multiplier {
            let _ = writeln!(out, "  x *= 0x{multiplier:016X}ul;");
        }
    }
    out.push_str("  return x;\n}\n");
    let _ = writeln!(
        out,
        "inline ulong ptir_rng_seed_eff(uint seed) {{\n  return ulong(seed) ^ 0x{:016X}ul;\n}}",
        RNG_FORMULA.ambient_seed_xor
    );
    let _ = writeln!(
        out,
        "inline ulong ptir_rng_stream_salt(uint stream) {{\n  return ptir_rng_splitmix64(\n      ulong(stream) * 0x{:016X}ul);\n}}",
        RNG_FORMULA.lane_stride
    );
    out.push_str(
        "inline ulong ptir_rng_seed_eff_stream(uint seed, uint stream) {\n  return ptir_rng_seed_eff(seed) ^ ptir_rng_stream_salt(stream);\n}\n",
    );
    let _ = writeln!(
        out,
        "inline ulong ptir_rng_keyed_seed(uint key, uint counter) {{\n  return ptir_rng_splitmix64(\n      (ulong(key) << {}) | ulong(counter));\n}}",
        RNG_FORMULA.keyed_word_bits
    );
    let denominator = 1u64 << RNG_FORMULA.uniform_mantissa_bits;
    let uniform_max = UNIFORM_MAX_LITERAL;
    let _ = writeln!(
        out,
        "inline float ptir_rng_hash_uniform(ulong seed_eff, uint index) {{\n  const ulong x = seed_eff +\n      0x{:016X}ul * (ulong(index) + {}ul);\n  const uint bits = uint(ptir_rng_splitmix64(x) >> {});\n  const float raw = (float(bits) + {:.1}f) * (1.0f / {denominator}.0f);\n  /* clamp off the one draw in 2^24 that rounds to exactly 1.0f */\n  return raw < {uniform_max}f ? raw : {uniform_max}f;\n}}\n",
        RNG_FORMULA.lane_stride,
        RNG_FORMULA.lane_index_bias,
        RNG_FORMULA.uniform_mantissa_shift,
        RNG_FORMULA.uniform_midpoint
    );
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
