#![allow(clippy::too_many_arguments)]
//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in norm/add_bias.wgsl
    // Qwen-2's attention biases. Ported operand for operand from
    // `kernels-metal`'s row, which is `kernels-cuda`'s and
    // `kernels-vulkan`'s, so that the backends read one statement the same
    // way: IN PLACE over the value it biases (`out` from `Out(0)`, the same
    // bytes as `In(0)`), the bias off the statement's named weight, and the
    // row width derived rather than stated -- an `AddBias` carries no scalars,
    // because a bias vector's length is the projection's width and the trace
    // knows it.
    //
    // Stated on this side because coverage is defined as parity: the shared
    // text can only name an op some kernel implements, and until this row none
    // did on Metal, so `qkv_bias` models were served without their biases --
    // which is not a crash and not a NaN, it is fluent, wrong text.
    kernel!(add_bias "add_bias", file = Some("norm/add_bias.wgsl"),
        launch = kernels::LaunchRule::RouteRows,
        in_place = &[(0, 0)],
        operands = kernels::operands![
            out: BufMut <- kernels::Source::Out(0),
            bias: Buf <- kernels::Source::Weight(0),
            width: I32 <- kernels::Source::OutWidth(0),
        ],
        axes = &[BF16]),
    // 1 in gated_rms.wgsl
    kernel!(gated_rms "gated_rms", axes = &[BF16]),
    // 1 in gated_rms.wgsl
    kernel!(gated_rms_strided "gated_rms_strided", axes = &[BF16]),
    // 1 in layer_scalar.wgsl
    // gemma's per-layer scale: one number per layer, read from a buffer
    // rather than stated, because which layer is running is the FIRE's.
    kernel!(layer_scalar_mul "layer_scalar_mul", file = Some("norm/layer_scalar.wgsl"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            scalar: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            // `LayerScalarParams`: the hidden width.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in residual_add.wgsl
    // Three buffers and no scalars: `out = x + residual`, elementwise, and
    // `out` may alias `x`. Filled because a MIXTURE demands it -- a routed
    // FFN's rows are already down-projected and combined, so all the block
    // owes is the add, where a dense FFN fuses the add into its down
    // projection (`gemm_add`) and never states this symbol.
    kernel!(residual_add "residual_add", file = Some("norm/residual_add.wgsl"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            residual: Buf <- kernels::Source::In(1),
            out: BufMut <- kernels::Source::Out(0),
        ],
        axes = &[BF16]),
    // 1 in residual_add.wgsl
    kernel!(residual_add_strided "residual_add_strided", axes = &[BF16]),
    // 1 in rms_norm.wgsl
    // `rms_single_row` with the block residual folded into its epilogue.
    kernel!(rms_residual "rms_residual", file = Some("norm/rms.wgsl"),
        launch = kernels::LaunchRule::Rms,
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            w: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            params: Buf <- kernels::Source::Param(0),
            r: Buf <- kernels::Source::In(1),
        ],
        axes = &[BF16]),
    // 1 in rms_norm.wgsl
    // The same, with a per-layer gain beside the residual.
    kernel!(rms_residual_scaled "rms_residual_scaled", file = Some("norm/rms.wgsl"),
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
    // 1 in rms_norm.wgsl
    // The first row of the sibling tables to state its OPERANDS, and the
    // reason is the finding that made them necessary: the trace states inputs,
    // outputs then weights, and this kernel declares `x, w, out, params`.
    // Binding positionally puts the output where the norm weight belongs.
    // Nothing reported it on Metal, which does not validate a binding, and
    // nothing reports it here either: every one of these is a storage buffer,
    // so a bind group typed by the LAYOUT accepts them in any order.
    //
    // `source` is what makes the row a thing a call can be GENERATED from:
    // `<- Source::In(0)` says this buffer takes the statement's first operand,
    // wherever the statement chose to put it.
    kernel!(rms_single_row "rms_single_row", file = Some("norm/rms.wgsl"), launch = kernels::LaunchRule::Rms,
        // `RmsParams.axis_size`, which is what the kernel strides by.
        grid_param = Some(1),
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            w: Buf <- kernels::Source::Weight(0),
            out: BufMut <- kernels::Source::Out(0),
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in rms_norm.wgsl
    kernel!(rms_strided_head_row "rms_strided_head_row", axes = &[BF16]),
    // 1 in rms_norm.wgsl
    kernel!(rms_strided_row "rms_strided_row", axes = &[BF16]),
    // 1 in vnorm.wgsl
    // A norm with no GAIN: the row divided by its own RMS and nothing else.
    // gemma's value norm, and the absence of a weight is the whole difference
    // from `rms_single_row`.
    kernel!(vnorm_single_row "vnorm_single_row", file = Some("norm/vector.wgsl"),
        launch = kernels::LaunchRule::Rms,
        // `VNormParams.axis_size`, for the reason `rms_single_row` states it:
        // this kernel gives threadgroup `gid` the span `gid * axis_size`, so
        // the grid needs one threadgroup per AXIS. A value norm's axis is the
        // HEAD and its row is every head, so without this the fire's width
        // would be taken for the axis and the whole row reduced as one --
        // which is not a smaller normalization, it is a different number in
        // every channel.
        grid_param = Some(1),
        operands = kernels::operands![
            x: Buf <- kernels::Source::In(0),
            out: BufMut <- kernels::Source::Out(0),
            // `VNormParams`: eps then axis_size, packed.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
];

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine};
use kernels::routine::Refusal;

/// The workgroup width every shader in this family declares.
///
/// `norm/{rms,vector,gated_rms}.wgsl` set `const PIE_LANES = 256u` and
/// `@workgroup_size(PIE_LANES)`; `{residual_add,layer_scalar,add_bias}.wgsl`
/// write `@workgroup_size(256)` directly. A [`Fire`] states LANES and the
/// driver divides by the module's own `@workgroup_size`, so this constant is
/// how many lanes one workgroup's worth is — the same number
/// `kernels-vulkan::norm::GROUP` carries for the same reason.
const GROUP: u32 = 256;

/// One workgroup per AXIS, as lanes.
///
/// **The grid counts axes, not rows, and the two are only the same for a norm
/// that spans its row.** `norm/rms.wgsl` gives a workgroup the span
/// `gid * axis_size`, so a row holds `width / axis` of them. gemma-4
/// normalizes each head of an 8192-wide Q over 256 channels: 32 axes per
/// token where a row-wise grid gives one, which leaves head 0 normalized and
/// the other thirty-one exactly as the projection wrote them — fully written
/// and only partly computed, so nothing downstream can report it.
///
/// `driver-wgpu::geometry`'s `Rule::Rms` arm already counted axes and says so;
/// `kernels-vulkan`'s first crossed routine did not, and that is the defect
/// its `norm` crossing found. This is the same arithmetic, stated once.
///
/// # Errors
///
/// [`Refusal::Empty`] for a zero width, axis or row count; [`Refusal::Wide`]
/// for an axis wider than the row it divides; [`Refusal::Grid`] if the lane
/// count overflows.
fn per_axis(width: i32, axis: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if axis > width {
        return Err(Refusal::Wide {
            what: "axis",
            at: i64::from(axis),
            max: i64::from(width),
        });
    }
    let axes = (width.unsigned_abs() / axis.unsigned_abs()) * rows.unsigned_abs();
    let lanes = axes.checked_mul(GROUP).ok_or(Refusal::Grid {
        what: "axes * the workgroup width",
        at: i64::from(axes) * i64::from(GROUP),
    })?;
    Ok([lanes, 1, 1])
}

/// One workgroup per ROW, for the strided forms whose axis IS the row.
///
/// # Errors
///
/// As `per_axis`: an empty width, axis or row count, an axis wider than its
/// row, or a lane count that overflows.
fn per_row(rows: i32) -> Result<[u32; 3], Refusal> {
    per_axis(1, 1, rows)
}

/// One workgroup per `(head, row)` pair, for the two-level forms.
///
/// A strided head norm's base is two-level — the head is `head * axis` into a
/// row and the next token a uniform `row_pitch` away — and one grid axis
/// cannot carry both terms. y is the head and z is the ROW.
///
/// **z was the defect.** `driver-wgpu::geometry`'s `Rule::GatedRms` passed a
/// literal `1` there while `norm/gated_rms.wgsl` read `wg.z` as the token in
/// both arms it can be built as, so a prefill normalized its first row and
/// left the rest as the projection wrote them. A decode is one row, which is
/// why nothing saw it. Stated here so the routine cannot repeat it.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty head or row count.
fn per_head_row(heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([GROUP, heads.unsigned_abs(), rows.unsigned_abs()])
}

/// `out = w * (x / rms(x))`, one workgroup per axis.
///
/// The order of `x, w, out, params` is the SHADER's and not the trace's. A
/// trace states inputs, outputs, then weights, so binding positionally puts
/// the output where the norm weight belongs — and nothing reports it, because
/// every one of these is a storage buffer and a bind group typed by the
/// LAYOUT accepts them in any order. That is why this was the first row in
/// the tree to state its operands, and here it is the argument order.
///
/// # Errors
///
/// As `per_axis`: an empty width, axis or row count, an axis wider than its
/// row, or a lane count that overflows.
pub fn rms_single_row(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/rms.wgsl",
            entrypoint: "rms_single_row_bfloat16",
            lanes: per_axis(*width, *axis, *rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v()],
    )
}

/// [`rms_single_row`] over rows a `row_pitch` apart rather than an axis apart.
///
/// # Errors
///
/// As `per_row`, which is `per_axis` at one axis per row.
pub fn rms_strided_row(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    row_pitch: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/rms.wgsl",
            entrypoint: "rms_strided_row_bfloat16",
            lanes: per_row(*rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v(), row_pitch.v()],
    )
}

/// [`rms_strided_row`] with the HEAD as its own axis.
///
/// # Errors
///
/// As `per_head_row`: an empty head or row count.
pub fn rms_strided_head_row(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    row_pitch: i32,
    heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/rms.wgsl",
            entrypoint: "rms_strided_head_row_bfloat16",
            lanes: per_head_row(*heads, *rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v(), row_pitch.v()],
    )
}

/// [`rms_single_row`] with the block residual folded into its epilogue.
///
/// # Errors
///
/// As `per_axis`: an empty width, axis or row count, an axis wider than its
/// row, or a lane count that overflows.
pub fn rms_residual(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    r: Buf,
    width: Env<i32>,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/rms.wgsl",
            entrypoint: "rms_residual_bfloat16",
            lanes: per_axis(*width, *axis, *rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v(), r.v()],
    )
}

/// [`rms_residual`] with a per-layer gain beside the residual.
///
/// # Errors
///
/// As `per_axis`: an empty width, axis or row count, an axis wider than its
/// row, or a lane count that overflows.
pub fn rms_residual_scaled(
    ctx: &Ctx<'_>,
    x: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    r: Buf,
    s: Buf,
    width: Env<i32>,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/rms.wgsl",
            entrypoint: "rms_residual_scaled_bfloat16",
            lanes: per_axis(*width, *axis, *rows)?,
        },
        &[x.v(), w.v(), out.v(), params.v(), r.v(), s.v()],
    )
}

/// A norm with no GAIN: the row divided by its own RMS and nothing else.
///
/// gemma's value norm. The absence of a weight is the whole difference from
/// [`rms_single_row`], and the axis is the HEAD — without that the fire's
/// width is taken for the axis and the whole row reduced as one, which is not
/// a smaller normalization but a different number in every channel.
///
/// # Errors
///
/// As `per_axis`: an empty width, axis or row count, an axis wider than its
/// row, or a lane count that overflows.
pub fn vnorm_single_row(
    ctx: &Ctx<'_>,
    x: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    axis: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/vector.wgsl",
            entrypoint: "vnorm_single_row_bfloat16",
            lanes: per_axis(*width, *axis, *rows)?,
        },
        &[x.v(), out.v(), params.v()],
    )
}

/// The gated-delta-net value norm: `out = w * (x / rms(x)) * silu(z)`.
///
/// # Errors
///
/// As `per_head_row`: an empty head or row count.
pub fn gated_rms(
    ctx: &Ctx<'_>,
    x: Buf,
    z: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/gated_rms.wgsl",
            entrypoint: "gated_rms_bfloat16",
            lanes: per_head_row(*heads, *rows)?,
        },
        &[x.v(), z.v(), w.v(), out.v(), params.v()],
    )
}

/// [`gated_rms`] over heads packed inside a wider row.
///
/// # Errors
///
/// As `per_head_row`: an empty head or row count.
pub fn gated_rms_strided(
    ctx: &Ctx<'_>,
    x: Buf,
    z: Buf,
    w: Buf,
    out: BufMut,
    params: Buf,
    row_pitch: i32,
    heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/gated_rms.wgsl",
            entrypoint: "gated_rms_strided_bfloat16",
            lanes: per_head_row(*heads, *rows)?,
        },
        &[x.v(), z.v(), w.v(), out.v(), params.v(), row_pitch.v()],
    )
}

/// gemma's per-layer scale: `out = x * scalar`, the scalar read from a buffer.
///
/// Which layer is running is the FIRE's, so the number is a buffer rather
/// than a stated scalar.
///
/// # Errors
///
/// [`kernels::shader::elementwise_rows`]'s, for a zero width or row count.
pub fn layer_scalar_mul(
    ctx: &Ctx<'_>,
    x: Buf,
    scalar: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/layer_scalar.wgsl",
            entrypoint: "layer_scalar_mul_bfloat16",
            lanes: kernels::shader::elementwise_rows(*width, *rows)?,
        },
        &[x.v(), scalar.v(), out.v(), params.v()],
    )
}

/// `out = x + residual`, elementwise, and `out` may alias `x`.
///
/// A MIXTURE demands it: a routed FFN's rows are already down-projected and
/// combined, so all the block owes is the add, where a dense FFN fuses the
/// add into its down projection and never states this symbol.
///
/// # Errors
///
/// [`kernels::shader::elementwise_rows`]'s.
pub fn residual_add(
    ctx: &Ctx<'_>,
    x: Buf,
    residual: Buf,
    out: BufMut,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/residual_add.wgsl",
            entrypoint: "residual_add_bfloat16",
            lanes: kernels::shader::elementwise_rows(*width, *rows)?,
        },
        &[x.v(), residual.v(), out.v()],
    )
}

/// [`residual_add`] over rows a `row_pitch` apart.
///
/// # Errors
///
/// [`kernels::shader::elementwise_rows`]'s.
pub fn residual_add_strided(
    ctx: &Ctx<'_>,
    x: Buf,
    residual: Buf,
    out: BufMut,
    row_pitch: i32,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/residual_add.wgsl",
            entrypoint: "residual_add_strided_bfloat16",
            lanes: kernels::shader::elementwise_rows(*width, *rows)?,
        },
        &[x.v(), residual.v(), out.v(), row_pitch.v()],
    )
}

/// The Qwen-2 family's attention biases, IN PLACE over the value they bias.
///
/// The trace's `AddBias` states an input and an output and the kernel binds
/// only the output, because they are the same bytes.
///
/// # Errors
///
/// [`kernels::shader::elementwise_rows`]'s.
pub fn add_bias(
    ctx: &Ctx<'_>,
    out: BufMut,
    bias: Buf,
    width: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/add_bias.wgsl",
            entrypoint: "add_bias_bfloat16",
            lanes: kernels::shader::elementwise_rows(width, *rows)?,
        },
        &[out.v(), bias.v(), width.v()],
    )
}

pub static ROUTINES: &[Routine] = &[
    // The one in-place pair in the family, and a REAL one unlike a rotation's:
    // the trace's `AddBias` states an input and an output, and the kernel
    // binds only the output because they are the same bytes.
    crate::routine!(add_bias, in_place = &[(0, 0)]),
    crate::routine!(gated_rms),
    crate::routine!(gated_rms_strided),
    crate::routine!(layer_scalar_mul),
    crate::routine!(residual_add),
    crate::routine!(residual_add_strided),
    crate::routine!(rms_residual),
    crate::routine!(rms_residual_scaled),
    crate::routine!(rms_single_row),
    crate::routine!(rms_strided_head_row),
    crate::routine!(rms_strided_row),
    crate::routine!(vnorm_single_row),
];
