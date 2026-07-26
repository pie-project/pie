//! The CUDA runtime template every emitted kernel is prefixed with.
//!
//! The C++ built this by concatenating two raw string literals around the
//! generated RNG preamble (`PTIR_RNG_CUDA_PREAMBLE`). The two literals are
//! checked in verbatim under `compiler/codegen/runtime/cuda/`; the RNG half is
//! still generated, by [`crate::rng`], so the device RNG cannot drift from
//! `pie_ir::rng`.

use alloc::string::String;

const PROLOGUE: &str = include_str!("../../runtime/cuda/ptir_m1_runtime_prologue.cuh");
const BODY: &str = include_str!("../../runtime/cuda/ptir_m1_runtime_body.cuh");

/// The `__device__` projection of the RNG contract, as the C++ spliced it in.
fn rng_preamble() -> String {
    let header = crate::rng::generate_cuda_header();
    // The raw-string literal begins immediately after `(`, so its leading
    // newline is part of the constant's value.
    let open = "inline constexpr char PTIR_RNG_CUDA_PREAMBLE[] = R\"PTIR_RNG_CUDA(";
    let close = ")PTIR_RNG_CUDA\";";
    let start = header
        .find(open)
        .expect("the generated RNG header defines PTIR_RNG_CUDA_PREAMBLE")
        + open.len();
    let end = header[start..]
        .find(close)
        .expect("the RNG preamble literal is terminated")
        + start;
    header[start..end].into()
}

/// `singleton_runtime_cuda_source()`.
pub fn singleton_runtime_source() -> String {
    let mut source = String::with_capacity(PROLOGUE.len() + BODY.len() + 4096);
    source.push_str(PROLOGUE);
    source.push_str(&rng_preamble());
    source.push_str(BODY);
    source
}
