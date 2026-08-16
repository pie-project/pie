#![allow(clippy::too_many_arguments)]
//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

use kernels::KernelSig;

/// EMPTY: this family's rows have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. Twelve kernels, every one of them live —
/// `rms_single_row` is 1084 rectangles of the corpus on its own, one per layer
/// of every model — so the crossing was measured before the rows went:
/// `every_launchs_scalars_land_where_its_module_reads_them` derived 1700
/// rectangles twice, by the row and by the arm, and compared every field.
///
/// It found one on its first run. [`residual_add`] asked for a
/// `[width, rows, 1]` grid and its shader reads `gid.x` alone, so 63 rows of
/// 64 would have gone untouched with the dispatch reporting success. The body
/// had never fired — nothing was armed, so every real `residual_add` went
/// through the row, which said `LaunchRule::Elementwise` and was right.
pub static KERNELS: &[KernelSig] = &[];

/// The entrypoints this family's routines spell, now that its rows are gone.
///
/// See [`crate::sample::ENTRYPOINTS`].
pub static ENTRYPOINTS: &[&str] = &[
    "add_bias_bfloat16",
    "gated_rms_bfloat16",
    "gated_rms_strided_bfloat16",
    "layer_scalar_mul_bfloat16",
    "residual_add_bfloat16",
    "residual_add_strided_bfloat16",
    "rms_residual_bfloat16",
    "rms_residual_scaled_bfloat16",
    "rms_single_row_bfloat16",
    "rms_strided_head_row_bfloat16",
    "rms_strided_row_bfloat16",
    "vnorm_single_row_bfloat16",
];

use crate::routine::{Else, Nth, Over, Reckoned, Say, keys, Ask, Bind, Block, Buf, BufMut, Ctx, Env, Fire, InSlot, OutSlot, Param, ParamOr, Routine, Weight};
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
    x: InSlot<0, Buf>,
    w: Weight<0, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    width: Ask<keys::Width, i32>,
    axis: ParamOr<1, keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    x: InSlot<0, Buf>,
    w: Weight<0, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    row_pitch: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    x: InSlot<0, Buf>,
    w: Weight<0, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    row_pitch: Ask<keys::Width, i32>,
    heads: Reckoned<Over<Say<keys::Width>, Else<Nth<1>, Say<keys::Width>>>, Env<i32>>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/rms.wgsl",
            entrypoint: "rms_strided_head_row_bfloat16",
            lanes: per_head_row(**heads, *rows)?,
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
    x: InSlot<0, Buf>,
    w: Weight<0, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    r: InSlot<1, Buf>,
    width: Ask<keys::Width, i32>,
    axis: ParamOr<1, keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    x: InSlot<0, Buf>,
    w: Weight<0, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    r: InSlot<1, Buf>,
    s: InSlot<2, Buf>,
    width: Ask<keys::Width, i32>,
    axis: ParamOr<1, keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    x: InSlot<0, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    width: Ask<keys::Width, i32>,
    axis: ParamOr<1, keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    x: InSlot<0, Buf>,
    z: InSlot<1, Buf>,
    w: Weight<0, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    heads: Ask<keys::VHeads, i32>,
    rows: Ask<keys::Rows, i32>,
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
    x: InSlot<0, Buf>,
    z: InSlot<1, Buf>,
    w: Weight<0, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    row_pitch: Ask<keys::Width, i32>,
    heads: Ask<keys::VHeads, i32>,
    rows: Ask<keys::Rows, i32>,
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
    x: InSlot<0, Buf>,
    scalar: Weight<0, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    x: InSlot<0, Buf>,
    residual: InSlot<1, Buf>,
    out: OutSlot<0, BufMut>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/residual_add.wgsl",
            entrypoint: "residual_add_bfloat16",
            // FLAT, not `elementwise_rows`. The non-strided variant of
            // `residual_add.wgsl` reads `gid.x` alone; a `[width, rows, 1]`
            // grid would run every y over the SAME first `width` elements and
            // leave the rest of the buffer as the projection wrote it —
            // 63 rows of 64 untouched, with the dispatch reporting success.
            //
            // `kernels-metal` and `kernels-vulkan` both say `elementwise` here
            // and `elementwise_rows` in the strided form below, which is the
            // shape of the shader. This body said `elementwise_rows` in both
            // and had never fired: nothing was armed, so every real
            // `residual_add` went through the row, which states
            // `LaunchRule::Elementwise` and is right.
            // `every_launchs_scalars_land_where_its_module_reads_them` caught it
            // on the first run after `norm` was armed, `[12, 64, 1]` against
            // the row's `[720, 1, 1]`.
            lanes: kernels::shader::elementwise(*width, *rows)?,
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
    x: InSlot<0, Buf>,
    residual: InSlot<1, Buf>,
    out: OutSlot<0, BufMut>,
    row_pitch: Param<0, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
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
    out: OutSlot<0, BufMut>,
    bias: Weight<0, Buf>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "norm/add_bias.wgsl",
            entrypoint: "add_bias_bfloat16",
            lanes: kernels::shader::elementwise_rows(*width, *rows)?,
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
