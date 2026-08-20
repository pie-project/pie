//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

use kernels_macros::routine;
use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows, keys};
use kernels::KernelSig;
use kernels::routine::Refusal;


/// The entrypoints this family's crossed routines spell, now that their
/// rows are gone. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &[
    "gated_rms_bfloat16",
    "gated_rms_strided_bfloat16",
    "layer_scalar_mul_bfloat16",
    "add_bias_bfloat16",
    "residual_add_bfloat16",
    "residual_add_strided_bfloat16",
    "rms_residual_bfloat16",
    "rms_residual_scaled_bfloat16",
    "rms_single_row_bfloat16",
    "rms_rope_bfloat16",
    "rms_rope_decode_bfloat16",
    "rms_rope_freqs_bfloat16",
    "rms_rope_freqs_decode_bfloat16",
    "rms_rope_prop_bfloat16",
    "rms_rope_prop_decode_bfloat16",
    "rms_strided_head_row_bfloat16",
    "rms_strided_row_bfloat16",
    "vnorm_single_row_bfloat16",
];

/// The workgroup width every reducing kernel in this file is compiled at.
///
/// `PIE_GROUP_X` in `rms.slang` and `gated_rms.slang`, `PIE_GROUP` in
/// `vector.slang`, and 256 in all three. It appears here because these grids
/// are stated in THREADS and a reduction wants one whole workgroup per row:
/// the lane count for `rows` rows is `rows * 256`, and `driver-vulkan`'s
/// `div_ceil` turns it back into `rows` workgroups. A row is reduced by the
/// workgroup that owns it, so this is the shader's number and not a tuning
/// choice -- `256 * N_READS` is also the span the row-walking loop strides by.
const GROUP: u32 = 256;

/// One whole workgroup per row, which is what a row reduction needs.
///
/// For the STRIDED forms only: their base is `group.x * row_pitch`, so a row
/// holds exactly one norm however wide the axis is.
///
/// # Errors
///
/// [`Refusal::Empty`] for no rows, and [`Refusal::Grid`] for a row count whose
/// lane total does not fit a `u32`. The second is not hypothetical arithmetic:
/// multiplying by 256 costs eight bits of headroom, so a row count that fits
/// comfortably can produce a lane count that does not, and a wrapped grid
/// launches a FRACTION of the rows and returns success.
fn per_row(rows: i32) -> Result<[u32; 3], Refusal> {
    per_axis(1, 1, rows)
}

/// One whole workgroup per AXIS, which is not the same as per row.
///
/// `rms.slang`'s packed base is `group.x * p.axis_size`, so the grid counts
/// AXES and a row holds `width / axis` of them. Handing such a fire one
/// workgroup per row normalizes the first axis of each row and leaves the rest
/// as the projection wrote them -- output fully written and only partly
/// normalized, which no read of it can report.
///
/// A hidden-state norm has `axis == width` and this reduces to one workgroup
/// per row, which is what it said before. A per-head q/k norm does not:
/// gemma-4 normalizes each of 32 heads of an 8192-wide Q over 256 channels,
/// and got one workgroup for the 32. Metal's crossing of this family is where
/// that was found; the two backends' shaders index the same way, so it was
/// true here too.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty extent, [`Refusal::Wide`] for an axis wider
/// than the row it sits in, and [`Refusal::Grid`] when the lane total does not
/// fit a `u32`.
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

/// One workgroup per `(head, row)` pair, for the two-level forms.
///
/// A token holds `heads` per-head norms packed `axis_size` apart and the next
/// token is a uniform `row_pitch` away, so the base is two-level and one grid
/// axis cannot carry both terms. The launch gives it two -- y is the head and
/// z is the row -- exactly as Metal's does. x is the workgroup, and it is 256
/// lanes rather than `heads * 256` because the head is its own axis here.
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

/// `out = w * (x / rms(x))`, one workgroup per row.
///
/// The order of `x, w, out, params` is the SHADER's and not the trace's. A
/// trace states inputs, outputs, then weights, so binding positionally puts
/// the output where the norm weight belongs -- and nothing reports it, because
/// a descriptor write is typed by the layout and every one of these is a
/// storage buffer. That mismatch is the reason this was the first row in the
/// tree to state its operands, and here it is the argument order.
///
/// # Errors
///
/// See [`per_axis`].
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
        Fire::at(crate::routine::module_path("rms_single_row_bfloat16", ctx.best()), "rms_single_row_bfloat16").apply(per_axis(width, *axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), params],
    )
}

/// [`rms_single_row`] over rows that are a `row_pitch` apart rather than an
/// axis apart.
///
/// The pitch is a PUSH constant and the axis is a field of the params struct,
/// which is this backend's rule rather than an accident: a scalar rides the
/// push block and a struct stays a buffer. `rms.slang`'s header is where both
/// halves of that rule are written down.
///
/// # Errors
///
/// See [`per_row`].
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
        Fire::at(crate::routine::module_path("rms_strided_row_bfloat16", ctx.best()), "rms_strided_row_bfloat16").apply(per_row(rows)?),
        &[x.arg(), w.arg(), out.arg(), params, row_pitch.arg()],
    )
}

/// The per-head q/k norms: `heads` norms inside each row, `rows` of them.
///
/// Two grid axes and not one. The head form cannot be [`rms_strided_row`] with
/// a different pitch, because the base is `row * row_pitch + head *
/// axis_size` and a single axis cannot carry both terms -- flattening it would
/// need the two strides to be commensurate, and `row_pitch` is the whole
/// token while `axis_size` is one head of it.
///
/// # Errors
///
/// See [`per_head_row`].
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
        Fire::at(crate::routine::module_path("rms_strided_head_row_bfloat16", ctx.best()), "rms_strided_head_row_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), w.arg(), out.arg(), params, row_pitch.arg()],
    )
}

/// One workgroup per `(head, row)`, refusing the rotary widths the fused
/// kernel cannot express.
///
/// The extent is [`per_head_row`]'s exactly -- the fusion changes what a
/// workgroup DOES and not how many there are -- so this only adds the two
/// refusals that come with carrying a rotation.
///
/// # Errors
///
/// [`per_head_row`]'s, plus:
///
/// [`Refusal::Empty`] for an empty rotary extent, which would be a norm
/// spelled as a rotation.
///
/// [`Refusal::Narrow`] for an ODD rotary. The kernel pairs `i` with `i +
/// rotary/2` and its tail loop starts at `rotary`, so an odd extent leaves
/// element `rotary - 1` in neither: it belongs to no pair and is below the
/// tail, and would come out RAW -- neither rotated nor normed nor even
/// multiplied by the gain. `rope.rs`'s `rope_grid` refuses the same shape for
/// the same reason and this is that refusal restated where the fused kernel
/// can see it.
///
/// [`Refusal::Wide`] for a rotary past the head. Rotating beyond `axis_size`
/// would reach into the next head's channels, which is a silent corruption
/// rather than a fault because the buffer is long enough for it.
fn per_head_row_rotating(
    heads: i32,
    rows: i32,
    rotary: i32,
    axis: i32,
) -> Result<[u32; 3], Refusal> {
    if rotary <= 0 {
        return Err(Refusal::Empty { what: "rotary" });
    }
    if rotary % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "rotary",
            at: i64::from(rotary),
        });
    }
    if axis > 0 && rotary > axis {
        return Err(Refusal::Wide {
            what: "rotary",
            at: i64::from(rotary),
            max: i64::from(axis),
        });
    }
    per_head_row(heads, rows)
}

/// [`rms_strided_head_row`] with the NEOX rotation that always follows it
/// folded into its epilogue.
///
/// Two of a decode layer's eleven stages are the q/k norm and then the rope,
/// in that order, over the same tensor and at the same shape. This is that
/// pair as one dispatch. The whole case for it is the barrier between them --
/// a decode step on this backend is 73% ordering -- and the measurement that
/// says the merge does not give it back is `tests/norm_bench.rs`.
///
/// # The operand order is the SHADER's
///
/// `x, w, params, position`, and `x` is bound once as a `Buf` because the
/// rotation is in place, which is `rope.rs`'s convention and not
/// [`rms_strided_head_row`]'s. The norm alone is out-of-place and states an
/// input and an output; the fused form cannot be, because the rotation reads
/// what the norm just wrote.
///
/// # Nine stated params and no push range
///
/// `binding::params` places a launch's scalars in a push range OR in a
/// parameter buffer and never both -- it asks push first, and a module
/// answering with the right push size "is not also hiding a parameter
/// buffer". A norm needs a struct, so this kernel's scalars all ride it, and
/// the statement states nine where every other norm states five.
///
/// The four extra are `row_pitch`, `rotary`, `scale` and the rope base.
/// `rotary` is there because it cannot be recovered from the rectangle:
/// gemma-4 rotates a quarter of each full-attention head and all of each
/// sliding one over the same tensor width. In `neox.slang` it is
/// `gl_NumWorkGroups.x`, which this kernel's grid no longer has room for.
///
/// # Errors
///
/// See [`per_head_row_rotating`].
#[routine]
pub fn rms_rope(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    w: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    // THE STRUCT'S OWN FIELDS, BY INDEX, and that is what this routine has
    // instead of marks. The body forwards `ctx.params()` whole, so the run is
    // `RmsRopeParams` read by FIELD -- and a `Const<i32>` mark is derived onto
    // the run by the order the marks appear, which put the one this routine
    // used to declare on `params[0]`. That word is `eps`, so the rotary width
    // this grid was built from was a float's bit pattern: `Narrow { what:
    // "rotary", at: 897988541 }` at the first q-norm of every fire.
    //
    // `axis` and `row_pitch` were read off the rectangle instead, which is the
    // same mistake in the other direction. They are equal only when a row
    // holds ONE norm, and this kernel exists for the case where it does not:
    // qwen3 norms 16 heads of 128 across a 2048-wide row, and `x.width` for
    // both terms made `heads` exactly 1 -- fifteen heads left unnormed and
    // unrotated, with the dispatch reporting success.
    //
    // `Ctx::param` is the sanctioned way to reach past the marks, and its doc
    // names this exact case: one run serving two readers, the other a struct.
    let axis = ctx.param(1)?;
    let row_pitch = ctx.param(5)?;
    let rotary = ctx.param(6)?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    // The head count is `row_pitch / axis` and BOTH terms are read off the
    // params run rather than one off the run and one off the rectangle. They
    // are the same numbers either way when a text is right, and when it is
    // wrong the grid and the block would disagree silently -- a shader
    // indexing by one and a launch sized by the other.
    let heads = if axis > 0 { row_pitch / axis } else { 0 };
    ctx.fire(
        Fire::at(crate::routine::module_path("rms_rope_bfloat16", ctx.best()), "rms_rope_bfloat16").apply(per_head_row_rotating(heads, rows, rotary, axis)?),
        // Four operands and NO scalars. Everything this kernel takes rides
        // the block, which `driver-vulkan`'s `encode` mints as the
        // statement's whole params run -- so the nine fields of
        // `RmsRopeParams` are nine stated params, in order, and the routine
        // adds nothing after them.
        &[x.arg(), w.arg(), params, position.arg()],
    )
}

/// [`rms_single_row`] with the block residual folded into its epilogue.
///
/// The residual is buffer 4 and arrives AFTER the params struct, which is the
/// order `#if defined(PIE_RESIDUAL)` adds it in -- the conditional bindings
/// come after the unconditional ones, so a fold does not renumber the four
/// every form shares.
///
/// # Errors
///
/// See [`per_axis`].
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
        Fire::at(crate::routine::module_path("rms_residual_bfloat16", ctx.best()), "rms_residual_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), params, r.arg()],
    )
}

/// [`rms_residual`] with a per-layer gain applied AFTER the add.
///
/// The two are separate entrypoints and not one with a gain of one, because
/// the order is the point: the gain multiplies the sum and not the normalised
/// value, and a fused form that scaled before adding would be a different
/// number.
///
/// # Errors
///
/// See [`per_axis`].
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
        Fire::at(crate::routine::module_path("rms_residual_scaled_bfloat16", ctx.best()), "rms_residual_scaled_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), params, r.arg(), s.arg()],
    )
}

/// A norm with no GAIN: the row divided by its own RMS and nothing else.
///
/// gemma's value norm. The absence of a weight is the whole difference from
/// [`rms_single_row`], and it is why `out` is buffer 1 here and buffer 2
/// there -- so an argument list copied between them binds the output where
/// the other reads its input.
///
/// # Errors
///
/// See [`per_axis`].
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
        Fire::at(crate::routine::module_path("vnorm_single_row_bfloat16", ctx.best()), "vnorm_single_row_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), out.arg(), params],
    )
}

/// `out = w * rmsnorm(x) * silu(z)`, per head.
///
/// The gate is buffer 1 and the weight is buffer 2, which is not the order the
/// name suggests: `gated_rms` reads as a norm that happens to be gated, and
/// the shader binds the gate before the gain.
///
/// The grid is `[256, heads, rows]`, and the non-strided form builds its base
/// from `group.z * gl_NumWorkGroups.y + group.y` -- it reads the HEAD COUNT
/// back out of the grid it was launched on. So the y extent is not a
/// scheduling choice here; a y that did not equal `heads` would not run fewer
/// heads, it would address the wrong rows.
///
/// # Errors
///
/// See [`per_head_row`].
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
        Fire::at(crate::routine::module_path("gated_rms_bfloat16", ctx.best()), "gated_rms_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), z.arg(), w.arg(), out.arg(), params],
    )
}

/// [`gated_rms`] over rows a `row_pitch` apart.
///
/// The strided form takes its row base from the pitch rather than from the
/// grid's own y extent, which is the one place the two differ.
///
/// # Errors
///
/// See [`per_head_row`].
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
        Fire::at(crate::routine::module_path("gated_rms_strided_bfloat16", ctx.best()), "gated_rms_strided_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), z.arg(), w.arg(), out.arg(), params, row_pitch.arg()],
    )
}

/// gemma's per-layer scale: one number per layer, read from a buffer.
///
/// Read rather than stated because WHICH layer is running is the fire's, not
/// the statement's -- a scalar operand would have to be re-stated per layer
/// and a buffer is bound per layer for free.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
#[routine]
pub fn layer_scalar_mul(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    scalar: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let _params = ctx.params()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("layer_scalar_mul_bfloat16", ctx.best()), "layer_scalar_mul_bfloat16").apply(elementwise(width, rows)?),
        &[x.arg(), scalar.arg(), out.arg()],
    )
}

/// `out = x + residual`, elementwise, and `out` may alias `x`.
///
/// Filled because a MIXTURE demands it: a routed FFN's rows are already
/// down-projected and combined, so all the block owes is the add, where a
/// dense FFN fuses the add into its down projection and never fires this.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
#[routine]
pub fn residual_add(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("residual_add_bfloat16", ctx.best()), "residual_add_bfloat16").apply(elementwise(width, rows)?),
        &[x.arg(), residual.arg(), out.arg()],
    )
}

/// [`residual_add`] over rows a `row_pitch` apart.
///
/// A RECTANGLE and not a flat run, which is the whole difference: the plain
/// form indexes `gid.x` straight into the buffer and the strided one indexes
/// `gid.y * row_pitch + gid.x`, so the grid has to carry the row on its own
/// axis for the pitch to be applied at all. Firing this on a flat grid adds
/// the first row `width * rows` times.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
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
        Fire::at(crate::routine::module_path("residual_add_strided_bfloat16", ctx.best()), "residual_add_strided_bfloat16").apply(elementwise_rows(width, rows)?),
        &[x.arg(), residual.arg(), out.arg(), row_pitch.arg()],
    )
}

/// The Qwen-2 family's attention biases, IN PLACE over the value they bias.
///
/// One buffer that is both the input and the result -- which is why `out` is
/// the only activation here and why it is `Buf`. The bias is one vector of
/// `width`, broadcast down every row.
///
/// `width` is a real argument and not an extent: the shader reads it from the
/// push block to find both the row base and the guard, so the grid and the
/// addressing come from the same number by construction. That is worth having
/// as one argument rather than two -- a grid width that disagreed with the
/// pushed width would bias the wrong columns of every row after the first.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
#[routine]
pub fn add_bias(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = out.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("add_bias_bfloat16", ctx.best()), "add_bias_bfloat16").apply(elementwise_rows(width, rows)?),
        &[out.arg(), bias.arg(), width.arg()],
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    type Call = (String, [u32; 3], Vec<ArgValue>);

    /// An `Encode` that remembers, and answers the facts this family's bodies
    /// ask for.
    ///
    /// `rows` backs every routine's `ctx.ask::<i32, keys::Rows>()`; the two
    /// families that ask something else -- `rms_strided_head_row`'s
    /// `HeadsPerRow` and `residual_add_strided`'s `RowPitch` -- get a field of
    /// their own. `HeadsPerRow`'s own `Source` is a chain
    /// (`width / (params[1] or width)`, see `keys::HeadsPerRow`), so it is
    /// matched by comparing the whole `Source` rather than by a `Named`
    /// string. `ctx.params()`, which every routine but `residual_add` and
    /// `residual_add_strided` calls for a block none of these tests inspect
    /// the CONTENTS of, is answered generically by `Ty` alone.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        rows: Cell<i32>,
        vheads: Cell<i32>,
        heads_per_row: Cell<i32>,
        row_pitch: Cell<i32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                rows: Cell::new(1),
                vheads: Cell::new(8),
                heads_per_row: Cell::new(1),
                row_pitch: Cell::new(1),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(
            &self,
            ty: kernels::Ty,
            source: kernels::Source,
        ) -> Result<ArgValue, Refusal> {
            // The statement's own scalars, read by index where the params run
            // is the shader's struct -- see `Asks::param`.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                let _ = n;
                return Ok(ArgValue::I32(4096));
            }
            use kernels::keys::Fact;
            // The geometry these bodies read now that their params run is a
            // STRUCT and no slot in it is theirs to take.
            if source == <keys::VHeads as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.vheads.get()));
            }
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            if source == <keys::HeadsPerRow as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.heads_per_row.get()));
            }
            if matches!(ty, kernels::Ty::Buf) {
                return Ok(ArgValue::Buffer { handle: 900, writes: false, rows: 0, width: 0 });
            }
            // Anything else is refused: a probe that invented an answer to a
            // fact it does not know would let a body pass under test while
            // the same fact went unanswered on a real driver.
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    /// A reducing norm gets a WHOLE workgroup per row; an elementwise one gets
    /// a lane per element.
    ///
    /// The two launch shapes in this file look alike written down and are not
    /// the same kind of number. `rms_single_row` reduces a row and the
    /// workgroup that owns it must be whole, so 3 rows is 768 lanes;
    /// `residual_add` has nothing to reduce and 3 rows of 128 is 384 lanes.
    /// Giving the norm the elementwise count launches `ceil(rows * width /
    /// 256)` workgroups -- fewer than `rows` whenever the row is narrower than
    /// 256 -- and every row past the last one is left exactly as it arrived,
    /// unnormalised.
    #[test]
    fn a_reduction_is_launched_by_the_workgroup_and_an_elementwise_op_by_the_lane() {
        let seen = Seen::default();
        seen.rows.set(3);
        rms_single_row(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 3, width: 128 },
            Const::new(Tensor::<bf16>::new(1)),
            Out { ptr: Tensor::<bf16>::new(2), rows: 3, width: 128 },
            // The struct's first two fields: this row holds ONE norm, so the
            // axis is the row's own 128.
            Const::new(1e-5),
            Const::new(128))
        .expect("a launch");
        residual_add(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 3, width: 128 },
            In { ptr: Tensor::<bf16>::new(1), rows: 3, width: 128 },
            Out { ptr: Tensor::<bf16>::new(2), rows: 3, width: 128 },
        )
        .expect("a launch");

        let calls = seen.calls.borrow();
        assert_eq!(
            (calls[0].1, calls[1].1),
            ([768, 1, 1], [384, 1, 1]),
            "three rows is three whole 256-lane workgroups for a reduction and \
             `3 * 128` lanes for an elementwise add"
        );
    }

    /// A two-level base needs two grid axes, and the head count is one of
    /// them.
    ///
    /// `rms_strided_head_row` addresses `row * row_pitch + head * axis_size`,
    /// and the two strides are not commensurate -- `row_pitch` is a whole
    /// token and `axis_size` is one head of it -- so the head cannot be folded
    /// into the row axis at any pitch. `gated_rms` is stronger still: its
    /// non-strided form reads the head count back out of `gl_NumWorkGroups.y`,
    /// so a y extent that is not `heads` does not run fewer heads, it
    /// addresses different rows.
    #[test]
    fn a_two_level_norm_carries_the_head_on_its_own_grid_axis() {
        let seen = Seen::default();
        seen.rows.set(5);
        rms_strided_head_row(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 5, width: 4096 },
            Const::new(Tensor::<bf16>::new(1)),
            Out { ptr: Tensor::<bf16>::new(2), rows: 5, width: 4096 },
            // The struct's first two fields. Eight heads across a 4096-wide
            // row is a 512-wide norm, and the body divides one by the other
            // rather than being told the count a second time.
            Const::new(1e-5),
            Const::new(512),
        )
        .expect("a launch");
        // `gated_rms` takes its head count as a `Const<i32>` parameter, not
        // an ask -- unlike the strided form beside it, it reads the head
        // count back out of the GRID rather than the row, so there is no
        // `x.width` here for it to derive one from.
        gated_rms(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 5, width: 0 },
            In { ptr: Tensor::<bf16>::new(1), rows: 5, width: 0 },
            Const::new(Tensor::<bf16>::new(2)),
            Out { ptr: Tensor::<bf16>::new(3), rows: 5, width: 0 })
        .expect("a launch");

        let calls = seen.calls.borrow();
        assert_eq!(
            (calls[0].1, calls[1].1),
            ([256, 8, 5], [256, 8, 5]),
            "one workgroup per (head, row) pair: x is the workgroup, y is the \
             head and z is the row, and the head is NOT multiplied into x"
        );
    }

    /// A strided elementwise op is a rectangle; a flat one is a run.
    ///
    /// `residual_add_strided` indexes `gid.y * row_pitch + gid.x` and
    /// `residual_add` indexes `gid.x`, so the strided form needs the row on
    /// its own axis for the pitch to be applied at all. Firing it flat gives
    /// every invocation `gid.y == 0` and adds the FIRST row `rows` times over,
    /// which for a one-row test -- the shape most of this tree is tested at --
    /// is indistinguishable from correct.
    #[test]
    fn a_strided_elementwise_op_is_a_rectangle_and_a_flat_one_is_a_run() {
        let seen = Seen::default();
        seen.rows.set(3);
        residual_add(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 3, width: 128 },
            In { ptr: Tensor::<bf16>::new(1), rows: 3, width: 128 },
            Out { ptr: Tensor::<bf16>::new(2), rows: 3, width: 128 },
        )
        .expect("a launch");
        seen.row_pitch.set(4096);
        residual_add_strided(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 3, width: 128 },
            In { ptr: Tensor::<bf16>::new(1), rows: 3, width: 128 },
            Out { ptr: Tensor::<bf16>::new(2), rows: 3, width: 128 },
         Const::new(4096))
        .expect("a launch");

        let calls = seen.calls.borrow();
        assert_eq!(
            (calls[0].1, calls[1].1),
            ([384, 1, 1], [128, 3, 1]),
            "the same 3 rows of 128: flat it is one run of 384 and strided it \
             is a 128-by-3 rectangle"
        );
    }

    /// The width a bias is launched on is the width it is TOLD.
    ///
    /// `add_bias` reads `pc.width` for both the row base and the out-of-range
    /// guard, so the grid and the addressing come from one number. Two
    /// arguments could disagree, and the disagreement is quiet: a grid wider
    /// than the pushed width returns early and biases nothing past the guard,
    /// and a grid narrower biases a prefix of every row and leaves the rest
    /// unbiased. Both decode as fluent text.
    #[test]
    fn a_bias_is_launched_on_the_width_it_is_told() {
        let seen = Seen::default();
        seen.rows.set(7);
        add_bias(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(0), rows: 7, width: 640 },
            Const::new(Tensor::<bf16>::new(1)),
        )
        .expect("a launch");

        let call = &seen.calls.borrow()[0];
        assert_eq!(call.1, [640, 7, 1], "the grid is the pushed width by rows");
        assert_eq!(
            call.2.last(),
            Some(&ArgValue::I32(640)),
            "and the pushed width is that same number, not a second one"
        );
    }

    /// A row count that overflows its lane count is refused, not wrapped.
    ///
    /// Multiplying by the workgroup width costs eight bits of headroom, so a
    /// row count that fits a `u32` comfortably can produce a lane count that
    /// does not. A wrap here is the worst available failure: the grid stays
    /// positive, the dispatch succeeds, and a small FRACTION of the rows is
    /// normalised while the rest pass through unchanged.
    ///
    /// The empty direction matters for the same reason and is quieter still --
    /// a zero grid normalises nothing at all and reports success.
    #[test]
    fn a_row_count_that_does_not_fit_the_lane_count_is_refused() {
        assert!(
            per_row(i32::MAX).is_err(),
            "`i32::MAX` rows is 2^39 lanes and does not fit the grid"
        );
        assert!(
            per_row(0).is_err() && per_row(-1).is_err(),
            "no rows is not a launch"
        );
        assert!(
            per_head_row(0, 4).is_err() && per_head_row(4, 0).is_err(),
            "no heads and no rows are both refused"
        );
        assert_eq!(
            per_row(1).expect("one row"),
            [256, 1, 1],
            "and one row is one whole workgroup"
        );
    }
}
