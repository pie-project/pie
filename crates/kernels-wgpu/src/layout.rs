//! Gathers and embeddings -- the kernels that move rows rather than
//! compute over them.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 6 in embed_gather.wgsl
    kernel!(embed_gather_4bit "embed_gather_4bit", file = Some("layout/embed_gather.wgsl"), launch = kernels::LaunchRule::Elementwise,
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
    // 6 in embed_gather.wgsl
    kernel!(embed_gather_mb_4bit "embed_gather_mb_4bit", file = Some("layout/embed_gather.wgsl"), launch = kernels::LaunchRule::ElementwiseRows,
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
    // 6 in embed_gather.wgsl
    // `embed_gather_4bit` with the embedding SCALE folded in -- gemma
    // multiplies its embeddings by `sqrt(hidden)`, which is a number the
    // statement carries rather than a kernel that knows the model.
    //
    // Single-row like its unscaled twin: `embed_gather_scaled_mb_4bit` is the
    // M>1 form and the one a text should name.
    kernel!(embed_gather_scaled_4bit "embed_gather_scaled_4bit",
        file = Some("layout/embed_gather.wgsl"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            w: Buf <- kernels::Source::Weight(0),
            scales: Buf <- kernels::Source::Weight(1),
            biases: Buf <- kernels::Source::Weight(2),
            id: I32s <- kernels::Source::TokenIds,
            out: BufMut <- kernels::Source::Out(0),
            hidden: I32 <- kernels::Source::Param(0),
            embed_scale: F32 <- kernels::Source::ParamF32(1),
        ],
        axes = &[BF16, GROUP, BITS]),
    // 6 in embed_gather.wgsl
    // The M>1 form, and the one a text should name: it reduces to the
    // single-row one at N=1, where that one reads `id[0]` whatever grid it is
    // handed.
    kernel!(embed_gather_scaled_mb_4bit "embed_gather_scaled_mb_4bit",
        file = Some("layout/embed_gather.wgsl"),
        launch = kernels::LaunchRule::ElementwiseRows,
        operands = kernels::operands![
            w: Buf <- kernels::Source::Weight(0),
            scales: Buf <- kernels::Source::Weight(1),
            biases: Buf <- kernels::Source::Weight(2),
            id: I32s <- kernels::Source::TokenIds,
            out: BufMut <- kernels::Source::Out(0),
            hidden: I32 <- kernels::Source::Param(0),
            embed_scale: F32 <- kernels::Source::ParamF32(1),
        ],
        axes = &[BF16, GROUP, BITS]),
    // 1 in ple_combine.wgsl
    // gemma's PLE join: `(proj + token) * inv_sqrt2`, over the whole
    // `[n_layers, ple_dim]` block at once. The scale is `1/sqrt(2)` and it is
    // the JOIN's, not a deployment's -- two streams averaged in the
    // root-mean-square sense.
    kernel!(ple_combine "ple_combine", file = Some("layout/ple_combine.wgsl"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            proj: Buf <- kernels::Source::In(0),
            token: Buf <- kernels::Source::In(1),
            out: BufMut <- kernels::Source::Out(0),
            // `PleCombineParams`: inv_sqrt2 then n, packed.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in row_gather.wgsl
    // The readout's gather. A prefill's stream is one row per TOKEN and its
    // readout is one distribution per REQUEST, so the sampled rows have to be
    // picked out before the lm head runs. `Kernel::G4RowGather` is what the
    // retiring driver called it.
    kernel!(row_gather "row_gather", file = Some("layout/row_gather.wgsl"),
        launch = kernels::LaunchRule::ElementwiseRows,
        operands = kernels::operands![
            input: Buf <- kernels::Source::In(0),
            out: BufMut <- kernels::Source::Out(0),
            rows: U32s <- kernels::Source::SamplingIndices,
            // `RowGatherParams` — width then count, PACKED into buffer 3.
            // There is no buffer 4, so the count is not an operand: it is the
            // struct's second FIELD, which `Ty::InPacked` is how a row says.
            //
            // The statement states `[width]` and the driver appends the count,
            // giving `[width, count]` — exactly the struct — because a packed
            // slot's run already covers every scalar after it.
            params: Buf <- kernels::Source::Param(0),
            count: InPacked <- kernels::Source::RequestCount,
        ],
        axes = &[BF16]),
];
