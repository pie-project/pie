//! Gathers and embeddings -- the kernels that move rows rather than
//! compute over them.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 6 in embed_gather.metal
    kernel!(embed_gather_4bit "embed_gather_4bit", file = Some("layout/embed_gather.metal"), launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            w: Buf <- kernels::Source::Weight(0),
            scales: Buf <- kernels::Source::Weight(1),
            biases: Buf <- kernels::Source::Weight(2),
            // The token IDS: the FIRE's, not the statement's. A text cannot
            // state them — they are this fire's data, not this model's
            // structure — so the row names which table and the driver's
            // resolver answers, the same way `Positions` has always worked.
            id: I32s <- kernels::Source::TokenIds,
            out: BufMut <- kernels::Source::Out(0),
            hidden: I32 <- kernels::Source::Param(0),
        ],
        axes = &[BF16, GROUP, BITS]),
    // 6 in embed_gather.metal
    kernel!(embed_gather_mb_4bit "embed_gather_mb_4bit", file = Some("layout/embed_gather.metal"), launch = kernels::LaunchRule::ElementwiseRows,
        operands = kernels::operands![
            w: Buf <- kernels::Source::Weight(0),
            scales: Buf <- kernels::Source::Weight(1),
            biases: Buf <- kernels::Source::Weight(2),
            // The token IDS: a fire value the text does not state and
            // `Source` has no name for. Stated as a gap rather than omitted —
            // a row is positional, so closing it would shift `out`.
            id: I32s <- kernels::Source::TokenIds,
            out: BufMut <- kernels::Source::Out(0),
            hidden: I32 <- kernels::Source::Param(0),
        ],
        axes = &[BF16, GROUP, BITS]),
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
