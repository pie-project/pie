//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

use kernels_macros::routine;
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

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise_rows, keys};
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
/// *z.ptr was the defect.** `driver-wgpu::geometry`'s `Rule::GatedRms` passed a
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
#[routine]
pub fn rms_single_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE STRUCT'S FIRST TWO FIELDS, DECLARED SO THE SECOND ONE'S SLOT EXISTS.
    // The body forwards `ctx.params()` whole, but the AXIS is the norm's own
    // width and it is `params[1]` -- `ParamOr<1, ..>` at HEAD. `eps` holds
    // slot 0 open for it. `x.width` is a row, which is the axis only when a
    // row holds ONE norm; the strided twin below shows why that is not a rule.
    eps: Const<f32>,
    axis: Const<i32>) -> Result<(), Refusal> {
    let _ = eps;
    let params = ctx.params()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/rms.wgsl", "rms_single_row_bfloat16").apply(per_axis(width, *axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), params],
    )
}

/// [`rms_single_row`] over rows a `row_pitch` apart rather than an axis apart.
///
/// # Errors
///
/// As `per_row`, which is `per_axis` at one axis per row.
#[routine]
pub fn rms_strided_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE STRUCT'S FIRST TWO FIELDS, DECLARED SO THE SLOTS EXIST.
    // The body forwards `ctx.params()` whole -- the shader reads a struct --
    // and neither scalar is read here: this plane's group size is the
    // shader's own, so the launch is `per_row(rows)` and the axis never
    // reaches a grid. Metal's twin DOES read it, because there the threadgroup
    // width is the body's to choose (`rms_threads(*axis)`), and that
    // difference is the whole reason both are declared: the statement carries
    // the same two scalars to every plane, so the slots have to line up
    // whether or not a given body has a use for them.
    eps: Const<f32>,
    axis: Const<i32>) -> Result<(), Refusal> {
    let _ = (eps, axis);
    let params = ctx.params()?;
    let row_pitch = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/rms.wgsl", "rms_strided_row_bfloat16").apply(per_row(rows)?),
        &[x.arg(), w.arg(), out.arg(), params, row_pitch.arg()],
    )
}

/// [`rms_strided_row`] with the HEAD as its own axis.
///
/// # Errors
///
/// As `per_head_row`: an empty head or row count.
#[routine]
pub fn rms_strided_head_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE STRUCT'S FIRST TWO FIELDS, DECLARED SO THE SECOND ONE'S SLOT EXISTS.
    // The body forwards `ctx.params()` whole -- the shader reads a struct --
    // but the AXIS is the norm's own width and the body needs it to size the
    // group, and it is NOT `x.width`: a strided row holds several norms and
    // `x.width` is the pitch across all of them. `eps` is declared only to
    // hold slot 0 open for it, and metal's twin declares both the same way.
    eps: Const<f32>,
    axis: Const<i32>) -> Result<(), Refusal> {
    let _ = eps;
    let params = ctx.params()?;
    let row_pitch = x.width;
    // HOW MANY NORMS FIT THE ROW -- the division HEAD spelled `Over<Say<Width>,
    // Else<Nth<1>, ..>>`, which no driver answers as a fact.
    let heads = if *axis > 0 { row_pitch / *axis } else { 0 };
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/rms.wgsl", "rms_strided_head_row_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), w.arg(), out.arg(), params, row_pitch.arg()],
    )
}

/// [`rms_single_row`] with the block residual folded into its epilogue.
///
/// # Errors
///
/// As `per_axis`: an empty width, axis or row count, an axis wider than its
/// row, or a lane count that overflows.
#[routine]
pub fn rms_residual(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    r: In<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let width = x.width;
    let axis = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/rms.wgsl", "rms_residual_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), params, r.arg()],
    )
}

/// [`rms_residual`] with a per-layer gain beside the residual.
///
/// # Errors
///
/// As `per_axis`: an empty width, axis or row count, an axis wider than its
/// row, or a lane count that overflows.
#[routine]
pub fn rms_residual_scaled(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    r: In<Tensor<bf16>>,
    s: In<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let width = x.width;
    let axis = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/rms.wgsl", "rms_residual_scaled_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), params, r.arg(), s.arg()],
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
#[routine]
pub fn vnorm_single_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let width = x.width;
    let axis = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/vector.wgsl", "vnorm_single_row_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), out.arg(), params],
    )
}

/// The gated-delta-net value norm: `out = w * (x / rms(x)) * silu(z)`.
///
/// # Errors
///
/// As `per_head_row`: an empty head or row count.
#[routine]
pub fn gated_rms(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    z: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let heads = ctx.ask::<i32, keys::VHeads>()?;
    let params = ctx.params()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/gated_rms.wgsl", "gated_rms_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), z.arg(), w.arg(), out.arg(), params],
    )
}

/// [`gated_rms`] over heads packed inside a wider row.
///
/// # Errors
///
/// As `per_head_row`: an empty head or row count.
#[routine]
pub fn gated_rms_strided(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    z: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let heads = ctx.ask::<i32, keys::VHeads>()?;
    let params = ctx.params()?;
    let row_pitch = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/gated_rms.wgsl", "gated_rms_strided_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), z.arg(), w.arg(), out.arg(), params, row_pitch.arg()],
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
#[routine]
pub fn layer_scalar_mul(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    scalar: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/layer_scalar.wgsl", "layer_scalar_mul_bfloat16").apply(elementwise_rows(width, rows)?),
        &[x.arg(), scalar.arg(), out.arg(), params],
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
#[routine]
pub fn residual_add(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/residual_add.wgsl", "residual_add_bfloat16").apply(elementwise_rows(width, rows)?),
        &[x.arg(), residual.arg(), out.arg()],
    )
}

/// [`residual_add`] over rows a `row_pitch` apart.
///
/// # Errors
///
/// [`kernels::shader::elementwise_rows`]'s.
#[routine]
pub fn residual_add_strided(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE STATEMENT'S PITCH, WHICH WAS `Param<0, i32>`. A stride is the
    // rectangle the text laid out -- two fires of one deployment stride the
    // same way -- so it fails `ask`'s own test and no driver answers
    // `keys::RowPitch`.
    row_pitch: Const<i32>) -> Result<(), Refusal> {
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/residual_add.wgsl", "residual_add_strided_bfloat16").apply(elementwise_rows(width, rows)?),
        &[x.arg(), residual.arg(), out.arg(), row_pitch.arg()],
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
#[routine]
pub fn add_bias(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = out.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/add_bias.wgsl", "add_bias_bfloat16").apply(elementwise_rows(width, rows)?),
        &[out.arg(), bias.arg(), width.arg()],
    )
}

