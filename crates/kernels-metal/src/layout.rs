//! Gathers and embeddings -- the kernels that move rows rather than
//! compute over them.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 6 in embed_gather.metal
    kernel!(embed_gather_4bit "embed_gather_4bit", file = Some("layout/embed_gather.metal"), launch = kernels::LaunchRule::Elementwise, axes = &[BF16, GROUP, BITS]),
    // 6 in embed_gather.metal
    kernel!(embed_gather_mb_4bit "embed_gather_mb_4bit", file = Some("layout/embed_gather.metal"), launch = kernels::LaunchRule::ElementwiseRows, axes = &[BF16, GROUP, BITS]),
    // 6 in embed_gather.metal
    kernel!(embed_gather_scaled_4bit "embed_gather_scaled_4bit", axes = &[BF16, GROUP, BITS]),
    // 6 in embed_gather.metal
    kernel!(embed_gather_scaled_mb_4bit "embed_gather_scaled_mb_4bit",
        axes = &[BF16, GROUP, BITS]),
    // 1 in ple_combine.metal
    kernel!(ple_combine "ple_combine", axes = &[BF16]),
    // 1 in row_gather.metal
    kernel!(row_gather "row_gather", axes = &[BF16]),
];
