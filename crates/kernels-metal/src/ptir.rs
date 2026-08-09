//! The PTIR substrate's own kernels -- the ones the tensor-compiler's
//! emitted MSL cannot produce because they predate a region.

use kernels::{KernelSig, kernel};

pub static KERNELS: &[KernelSig] = &[
    // NOT filled, and it is not an oversight: this backend's channel-plane
    // interpreter never dispatches it.
    //
    // The kernel stages logits rows GPU-side, and its own header records why
    // it was written -- a sixteen-request fire paid sixteen command-buffer
    // round trips per token to move sixteen vocabulary rows, about 3ms of a
    // 23.5ms step. That problem does not exist here. Apple silicon shares
    // physical memory, so `pipeline::step::PassInputs` takes the read-out as a
    // BORROWED host slice: "copying it per fire would be pure waste."
    //
    // A row is filled when a text names its symbol. No text names this one and
    // none should, so it stays a declaration of what the shader tree contains
    // rather than of what this driver dispatches.
    kernel!(copy_logits_bf16 "copy_logits_bf16"), // ptir/logits_copy.metal
];
