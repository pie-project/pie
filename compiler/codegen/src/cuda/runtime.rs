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
///
/// The C++ read this out of `PTIR_RNG_CUDA_PREAMBLE`, whose raw-string literal
/// opens immediately after `(` — so its leading newline is part of the
/// constant's value, and is reproduced here. `rng_contract`'s
/// `cuda_preamble_matches_the_header_literal` holds the two forms together.
fn rng_preamble() -> String {
    let mut preamble = String::from("\n");
    preamble.push_str(&crate::rng::cuda_device_functions());
    preamble
}

/// `singleton_runtime_cuda_source()`.
pub fn singleton_runtime_source() -> String {
    let mut source = String::with_capacity(PROLOGUE.len() + BODY.len() + 4096);
    source.push_str(PROLOGUE);
    source.push_str(&rng_preamble());
    source.push_str(BODY);
    source
}

#[cfg(test)]
mod tests {
    use super::BODY;
    use crate::runtime_scan::assert_execute_covers_the_table;

    #[test]
    fn cuda_execute_covers_the_op_table() {
        assert_execute_covers_the_table(BODY, "ptir_m1_runtime_body.cuh");
    }
}
