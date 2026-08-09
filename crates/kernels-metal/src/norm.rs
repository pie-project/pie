//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in gated_rms.metal
    kernel!(gated_rms "gated_rms", axes = &[BF16]),
    // 1 in gated_rms.metal
    kernel!(gated_rms_strided "gated_rms_strided", axes = &[BF16]),
    // 1 in layer_scalar.metal
    // gemma's per-layer scale: one number per layer, read from a buffer
    // rather than stated, because which layer is running is the FIRE's.
    kernel!(layer_scalar_mul "layer_scalar_mul", file = Some("norm/layer_scalar.metal"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            scalar: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            // `LayerScalarParams`: the hidden width.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in residual_add.metal
    // Three buffers and no scalars: `out = x + residual`, elementwise, and
    // `out` may alias `x`. Filled because a MIXTURE demands it -- a routed
    // FFN's rows are already down-projected and combined, so all the block
    // owes is the add, where a dense FFN fuses the add into its down
    // projection (`gemm_add`) and never states this symbol.
    kernel!(residual_add "residual_add", file = Some("norm/residual_add.metal"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            residual: Buf <- kernels::Source::In(1),
            out: BufMut <- kernels::Source::Out(0),
        ],
        axes = &[BF16]),
    // 1 in residual_add.metal
    kernel!(residual_add_strided "residual_add_strided", axes = &[BF16]),
    // 1 in rms_norm.metal
    // `rms_single_row` with the block residual folded into its epilogue.
    kernel!(rms_residual "rms_residual", file = Some("norm/rms.metal"),
        launch = kernels::LaunchRule::Rms,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            w: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            params: Buf <- kernels::Source::Param(0),
            r: Buf <- kernels::Source::In(1),
        ],
        axes = &[BF16]),
    // 1 in rms_norm.metal
    // The same, with a per-layer gain beside the residual.
    kernel!(rms_residual_scaled "rms_residual_scaled", file = Some("norm/rms.metal"),
        launch = kernels::LaunchRule::Rms,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            w: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            params: Buf <- kernels::Source::Param(0),
            r: Buf <- kernels::Source::In(1),
            s: Buf <- kernels::Source::In(2),
        ],
        axes = &[BF16]),
    // 1 in rms_norm.metal
    // The first Metal row to state its OPERANDS, and the reason is the
    // finding that made them necessary: the trace states inputs, outputs then
    // weights, and this kernel declares `x, w, out, params`. Binding
    // positionally puts the output where the norm weight belongs. Nothing
    // reported it, because Metal does not validate a binding.
    //
    // `source` is what makes the row a thing a call can be GENERATED from:
    // `<- Source::In(0)` says this buffer takes the statement's first operand,
    // wherever the statement chose to put it.
    kernel!(rms_single_row "rms_single_row", file = Some("norm/rms.metal"), launch = kernels::LaunchRule::Rms,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            w: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in rms_norm.metal
    kernel!(rms_strided_head_row "rms_strided_head_row", axes = &[BF16]),
    // 1 in rms_norm.metal
    kernel!(rms_strided_row "rms_strided_row", axes = &[BF16]),
    // 1 in vnorm.metal
    // A norm with no GAIN: the row divided by its own RMS and nothing else.
    // gemma's value norm, and the absence of a weight is the whole difference
    // from `rms_single_row`.
    kernel!(vnorm_single_row "vnorm_single_row", file = Some("norm/vector.metal"),
        launch = kernels::LaunchRule::Rms,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            out: BufMut <- kernels::Source::Out(0),
            // `VNormParams`: eps then axis_size, packed.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
];
