//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in norm/gated_rms.comp
    kernel!(gated_rms "gated_rms", axes = &[BF16]),
    // 1 in norm/gated_rms.comp
    kernel!(gated_rms_strided "gated_rms_strided", axes = &[BF16]),
    // 1 in norm/layer_scalar.comp
    // gemma's per-layer scale: one number per layer, read from a buffer
    // rather than stated, because which layer is running is the FIRE's.
    kernel!(layer_scalar_mul "layer_scalar_mul", file = Some("norm/layer_scalar.comp"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            scalar: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            // `LayerScalarParams`: the hidden width.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in norm/residual_add.comp
    // Three buffers and no scalars: `out = x + residual`, elementwise, and
    // `out` may alias `x`. Filled because a MIXTURE demands it -- a routed
    // FFN's rows are already down-projected and combined, so all the block
    // owes is the add, where a dense FFN fuses the add into its down
    // projection (`gemm_add`) and never states this symbol.
    kernel!(residual_add "residual_add", file = Some("norm/residual_add.comp"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            residual: Buf <- kernels::Source::In(1),
            out: BufMut <- kernels::Source::Out(0),
        ],
        axes = &[BF16]),
    // 1 in norm/residual_add.comp
    kernel!(residual_add_strided "residual_add_strided", axes = &[BF16]),
    // 1 in norm/rms.comp
    // `rms_single_row` with the block residual folded into its epilogue.
    kernel!(rms_residual "rms_residual", file = Some("norm/rms.comp"),
        launch = kernels::LaunchRule::Rms,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            w: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            params: Buf <- kernels::Source::Param(0),
            r: Buf <- kernels::Source::In(1),
        ],
        axes = &[BF16]),
    // 1 in norm/rms.comp
    // The same, with a per-layer gain beside the residual.
    kernel!(rms_residual_scaled "rms_residual_scaled", file = Some("norm/rms.comp"),
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
    // 1 in norm/rms.comp
    // The first row of the sibling table to state its OPERANDS, and the reason is the
    // finding that made them necessary: the trace states inputs, outputs then
    // weights, and this kernel declares `x, w, out, params`. Binding
    // positionally puts the output where the norm weight belongs. Nothing
    // reported it. Vulkan does not report it either: a descriptor write is
    // typed by the LAYOUT, and every one of these is a storage buffer.
    //
    // `source` is what makes the row a thing a call can be GENERATED from:
    // `<- Source::In(0)` says this buffer takes the statement's first operand,
    // wherever the statement chose to put it.
    kernel!(rms_single_row "rms_single_row", file = Some("norm/rms.comp"), launch = kernels::LaunchRule::Rms,
        // `RmsParams.axis_size`, which is what the kernel strides by.
        grid_param = Some(1),
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            w: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in norm/rms.comp
    kernel!(rms_strided_head_row "rms_strided_head_row", axes = &[BF16]),
    // 1 in norm/rms.comp
    kernel!(rms_strided_row "rms_strided_row", axes = &[BF16]),
    // 1 in norm/vector.comp
    // A norm with no GAIN: the row divided by its own RMS and nothing else.
    // gemma's value norm, and the absence of a weight is the whole difference
    // from `rms_single_row`.
    kernel!(vnorm_single_row "vnorm_single_row", file = Some("norm/vector.comp"),
        launch = kernels::LaunchRule::Rms,
        // `VNormParams.axis_size`, for the reason `rms_single_row` states it:
        // `vector.comp` gives workgroup `gl_WorkGroupID.x` the span
        // `gl_WorkGroupID.x * axis_size`, so the grid needs one workgroup per
        // AXIS. A value norm's axis is the HEAD and its row is every head, so
        // without this the fire's width would be taken for the axis and the
        // whole row reduced as one -- which is not a smaller normalization, it
        // is a different number in every channel.
        grid_param = Some(1),
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            out: BufMut <- kernels::Source::Out(0),
            // `VNormParams`: eps then axis_size, packed.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
];
