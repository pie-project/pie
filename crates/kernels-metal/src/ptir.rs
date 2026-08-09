//! The PTIR substrate's own kernels -- the ones the tensor-compiler's
//! emitted MSL cannot produce because they predate a region.

use kernels::{KernelSig, kernel};

pub static KERNELS: &[KernelSig] = &[
    kernel!(copy_logits_bf16 "copy_logits_bf16"), // ptir_logits_copy.metal
];
