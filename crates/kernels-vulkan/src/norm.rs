//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

use kernels_macros::routine;
use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows, keys};
use kernels::routine::Refusal;

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
/// The order of `x, w, out` is the SHADER's and not the trace's. A trace
/// states inputs, outputs, then weights, so binding positionally puts the
/// output where the norm weight belongs -- and nothing reports it, because a
/// descriptor write is typed by the layout and every one of these is a
/// storage buffer. That mismatch is the reason this was the first row in the
/// tree to state its operands, and here it is the argument order.
///
/// # All five of the struct's fields are STATED, and the module has no
/// descriptor for them
///
/// `eps`, `axis_size`, `w_stride`, `plus_one` and `gain` were `RmsParams` on
/// `norm/rms.slang`'s binding 3, staged from this statement's run and read by
/// field, while the strided forms pushed `row_pitch` beside it. That pair is
/// the one arrangement `binding::params` cannot serve -- it asks push first,
/// and a module answering with the right push size "is not also hiding a
/// parameter buffer" -- so this family was the reason the rule had to be
/// written down at all.
///
/// All five are marks now: the same five words of the same run, reached by
/// INDEX instead of by struct field, and `Encoder::words` puts them in the
/// push block in the order this body passes them. The descriptor is gone, and
/// with it the binding hole it left -- `norm/rms.slang`'s header carries the
/// renumbering.
///
/// Only two of the five appeared here before, and `eps` only to hold slot 0
/// open so `axis` could be slot 1. The other three were read by the shader and
/// named nowhere in Rust.
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
    // THE STRUCT'S FIVE FIELDS, IN THE STRUCT'S ORDER, because that order is
    // the statement's: the block was staged from words 0..5 of this
    // statement's run and every mark below reads the word its field sat at.
    // `w: Const<Tensor<bf16>>` above does not disturb the count -- a
    // `Const<Tensor<E>>` claims the WEIGHT run and not the params one.
    //
    // The body reads only `axis`, and `x.width` will not do instead: a row is
    // the axis only when it holds ONE norm, which the strided twin below is
    // the counterexample to.
    eps: Const<f32>,
    axis: Const<i32>,
    // THE THREE THE SHADER READ AND NOTHING HERE COULD NAME. `w_stride` is
    // the distance between consecutive CHANNELS of the norm weight,
    // `plus_one` is the gemma convention flag `gain_at` folds as `1 + w`, and
    // `gain` is the scale applied in float before the single bf16 round. All
    // three are passed and none is read here, which is a different thing from
    // the dead marks this signature used to carry: they are the shader's to
    // use, and this body's only job is to put them where it reads them.
    //
    // `u32` and not `i32` for the middle two because no arithmetic here wants
    // a sign, and `Arg::unpack` for a `Const<u32>` takes the unsigned carrier
    // and refuses the signed one -- so the spelling is a claim a caller has to
    // match.
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>) -> Result<(), Refusal> {
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("rms_single_row_bfloat16", ctx.best()), "rms_single_row_bfloat16").apply(per_axis(width, *axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis.arg(), w_stride.arg(), plus_one.arg(), gain.arg()],
    )
}

/// [`rms_single_row`] over rows that are a `row_pitch` apart rather than an
/// axis apart.
///
/// The pitch and the other five ride ONE push block, which is this backend's
/// rule with nothing left beside it: a scalar rides the push range, and the
/// struct that used to hold half of these in a descriptor is gone.
/// `rms.slang`'s header is where that history is written down.
///
/// The pitch goes LAST, after the five the packed form also passes, so that
/// folding a stride in does not renumber the words the two share.
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
    // [`rms_single_row`]'s five, at the same five words of the same run.
    //
    // THIS BODY READS NONE OF THEM, and passes all five. On this plane the
    // group size is the shader's own `[numthreads(PIE_GROUP_X, 1, 1)]`, so the
    // launch is `per_row(rows)` and the axis never reaches a grid; Metal's
    // twin does read it, because there the threadgroup width is the body's to
    // choose (`rms_threads(*axis)`). That difference is about the LAUNCH and
    // not about the argument list: the shader spans `axis_size` on every
    // plane, so every plane hands it over.
    eps: Const<f32>,
    axis: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>) -> Result<(), Refusal> {
    let row_pitch = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("rms_strided_row_bfloat16", ctx.best()), "rms_strided_row_bfloat16").apply(per_row(rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), row_pitch.arg()],
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
    // [`rms_single_row`]'s five, at the same five words of the same run, and
    // the pitch after them for [`rms_strided_row`]'s reason.
    //
    // The AXIS is the one this body reads, and it is NOT `x.width`: a strided
    // row holds several norms and `x.width` is the pitch across all of them,
    // so the head count below is the one divided by the other.
    eps: Const<f32>,
    axis: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>) -> Result<(), Refusal> {
    let row_pitch = x.width;
    // HOW MANY NORMS FIT THE ROW -- the division HEAD spelled `Over<Say<Width>,
    // Else<Nth<1>, ..>>`, which no driver answers as a fact.
    let heads = if *axis > 0 { row_pitch / *axis } else { 0 };
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("rms_strided_head_row_bfloat16", ctx.best()), "rms_strided_head_row_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), row_pitch.arg()],
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
/// The residual is binding 3 and it USED TO BE 4, because the params struct it
/// arrived after is gone. `#if defined(PIE_RESIDUAL)` still adds it after the
/// unconditional bindings, so the fold does not renumber the three every form
/// shares -- but a descriptor set is written from a DENSE list, so the hole
/// the struct left had to close rather than stay.
///
/// # This form stated NO scalar at all, and read five
///
/// It forwarded `ctx.params()` and declared nothing beside its four operands,
/// so the whole of `RmsParams` reached `norm/rms.slang` through a block no
/// signature described. Its three siblings at least held two slots open; this
/// one and [`rms_residual_scaled`] held none.
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
    r: In<Tensor<bf16>>,
    // [`rms_single_row`]'s five, at the same five words of the same run. They
    // are declared after `r` and FIRED before it, because the two orders
    // answer different questions: this list is the STATEMENT's, where an
    // operand precedes a scalar, and the fire array is the SHADER's, where
    // the residual is the conditional binding and therefore last.
    eps: Const<f32>,
    axis_size: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>) -> Result<(), Refusal> {
    let width = x.width;
    // NOT `*axis_size`. A norm sandwich spans its whole row, so the two are
    // the same number here -- but the launch has always taken the row's own
    // width and re-aiming it at the mark is a dispatch that moves, which
    // belongs in a change that can measure the move.
    let axis = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("rms_residual_bfloat16", ctx.best()), "rms_residual_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis_size.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), r.arg()],
    )
}

/// [`rms_residual`] with a per-layer gain applied AFTER the add.
///
/// The two are separate entrypoints and not one with a gain of one, because
/// the order is the point: the gain multiplies the sum and not the normalised
/// value, and a fused form that scaled before adding would be a different
/// number.
///
/// `s` is a one-element BUFFER at binding 4 and `gain` below is a pushed word,
/// and the two are not the same number wearing different carriers: the buffer
/// is the checkpoint's per-layer embedding scale, read as `s[0]` by every
/// invocation after the add, while `gain` is `RmsParams`'s own field, folded
/// into the weight before the norm's multiply. Both were reaching this shader
/// before and only one of them was visible from Rust.
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
    s: In<Tensor<bf16>>,
    // [`rms_residual`]'s five, declared after the operands and fired before
    // them, for the reason stated there.
    eps: Const<f32>,
    axis_size: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>) -> Result<(), Refusal> {
    let width = x.width;
    // NOT `*axis_size`, for [`rms_residual`]'s reason.
    let axis = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("rms_residual_scaled_bfloat16", ctx.best()), "rms_residual_scaled_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis_size.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), r.arg(), s.arg()],
    )
}

/// A norm with no GAIN: the row divided by its own RMS and nothing else.
///
/// gemma's value norm. The absence of a weight is the whole difference from
/// [`rms_single_row`], and it is why `out` is buffer 1 here and buffer 2
/// there -- so an argument list copied between them binds the output where
/// the other reads its input.
///
/// # The two scalars are STATED, and the launch still does not read them
///
/// `eps` and `axis_size` rode a `VNormParams` storage block this body
/// forwarded whole, so this signature could name neither. They are marks now
/// -- the same two words of the same statement run, reached by index instead
/// of by struct field -- and `norm/vector.slang` takes them as the two fields
/// of its push block, which is eight bytes where a descriptor used to be.
///
/// What that does NOT change is where the LAUNCH gets its axis. `per_axis`
/// is still handed `x.width`, so a row holding several heads becomes one
/// workgroup while the shader spans `group.x * axis_size` --
/// `kernels-metal::norm::vnorm_single_row` measures what that costs, its twin
/// having had the identical gap, and closed it by sizing `rms_grid` from the
/// mark. Here the mark now EXISTS to close it with and is deliberately not
/// spent yet: the fix moves the dispatch, and a dispatch that moves belongs
/// in a change that can measure the move rather than in the one that made the
/// number nameable.
///
/// # Errors
///
/// See [`per_axis`].
#[routine]
pub fn vnorm_single_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE STRUCT'S TWO FIELDS, BOTH OF THEM LIVE. `VNormParams { eps,
    // axis_size }` was staged from this statement's run and read as a struct;
    // these are the same two words reached by index, IN THE STRUCT'S ORDER
    // because that order is the statement's. The shader spells `axis_size`
    // `uint` and the mark spells it `i32`, exactly as `RmsParams.axis_size`
    // stands against [`rms_strided_head_row`]'s `axis`: the run is a
    // `Vec<u32>` and the BITS are the value, and what the mark's type decides
    // is what a body may do with the number.
    eps: Const<f32>,
    axis_size: Const<i32>) -> Result<(), Refusal> {
    let width = x.width;
    // NOT `*axis_size`, and the doc above says at length why not: this is the
    // gap `kernels-metal`'s twin closed and this one has not.
    let axis = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("vnorm_single_row_bfloat16", ctx.best()), "vnorm_single_row_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), out.arg(), eps.arg(), axis_size.arg()],
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
/// # One block, where this kernel had two
///
/// `eps` and `vd` were a `GatedRmsParams` storage block staged from the
/// statement's run, while `gated_rms_strided`'s `row_pitch` was a number
/// this body passed and therefore a push range -- a descriptor AND a push
/// range on one module, which is the one arrangement `binding::params` cannot
/// serve from a row, because it asks push first and a module answering with
/// the right push size *"is not also hiding a parameter buffer"*.
///
/// Both scalars are MARKS now, which is the same two words of the same run
/// reached by index instead of by struct field, so the push block is the only
/// place a scalar reaches `norm/gated_rms.slang` and its fields are this
/// argument list in this order.
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
    out: Out<Tensor<bf16>>,
    // THE STRUCT'S TWO FIELDS. They used to be unnameable here -- the body
    // forwarded `ctx.params()` whole, the shader read fields rather than a
    // scalar run, and a `Const` derived onto slot 0 would have read the
    // struct's first field's bits -- which is why the head count beside them
    // is an ASK and not a mark. It stays one: `keys::VHeads` is the FIRE's
    // fact and never was a word of this run.
    eps: Const<f32>,
    vd: Const<i32>) -> Result<(), Refusal> {
    let heads = ctx.ask::<i32, keys::VHeads>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("gated_rms_bfloat16", ctx.best()), "gated_rms_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), z.arg(), w.arg(), out.arg(), eps.arg(), vd.arg()],
    )
}

/// [`gated_rms`] over rows a `row_pitch` apart.
///
/// The strided form takes its row base from the pitch rather than from the
/// grid's own y extent, which is the one place the two differ.
///
/// The pitch goes LAST, after the pair the packed form also passes, so that
/// folding a stride in does not renumber the fields the two variants share --
/// which is why `norm/gated_rms.slang`'s push block is `{ eps, vd, row_pitch }`
/// and not in any other order. `Device::dispatch` refuses a push run whose
/// length is not the pipeline's range, but it cannot see two four-byte words
/// that swapped places.
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
    out: Out<Tensor<bf16>>,
    // [`gated_rms`]'s two, at the same two words of the same run.
    eps: Const<f32>,
    vd: Const<i32>) -> Result<(), Refusal> {
    let heads = ctx.ask::<i32, keys::VHeads>()?;
    let row_pitch = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("gated_rms_strided_bfloat16", ctx.best()), "gated_rms_strided_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), z.arg(), w.arg(), out.arg(), eps.arg(), vd.arg(), row_pitch.arg()],
    )
}

/// gemma's per-layer scale: one number per layer, read from a buffer.
///
/// Read rather than stated because WHICH layer is running is the fire's, not
/// the statement's -- a scalar operand would have to be re-stated per layer
/// and a buffer is bound per layer for free.
///
/// AND NOTHING IS STATED BESIDE IT. This asked for `ctx.params()` and threw
/// the answer away: slangc had deleted `LayerScalarParams`' binding for being
/// unread, so the block reached no argument list and the call survived only as
/// a `let _`. The block is gone from the module too now, which is what a
/// discarded ask was always saying.
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
            // The struct's five fields, in the struct's order: this row holds
            // ONE norm, so the axis is the row's own 128. The stride is one
            // channel, the gemma `+1` is off and the gain is unity -- three
            // words this routine could not name until they became marks, and
            // three this test therefore could not have stated before.
            Const::new(1e-5),
            Const::new(128),
            Const::new(1),
            Const::new(0),
            Const::new(1.0))
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
            // The struct's five fields. Eight heads across a 4096-wide row is
            // a 512-wide norm, and the body divides one by the other rather
            // than being told the count a second time; the last three are the
            // shader's alone and go by untouched.
            Const::new(1e-5),
            Const::new(512),
            Const::new(1),
            Const::new(0),
            Const::new(1.0),
        )
        .expect("a launch");
        // `gated_rms` takes its head count from an ASK -- unlike the strided
        // form beside it, it reads the head count back out of the GRID rather
        // than the row, so there is no `x.width` here for it to derive one
        // from. Its two MARKS are the struct's two fields, and neither reaches
        // the grid: the shader spans `vd` and the launch is `[256, heads,
        // rows]` whatever `vd` says.
        gated_rms(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 5, width: 0 },
            In { ptr: Tensor::<bf16>::new(1), rows: 5, width: 0 },
            Const::new(Tensor::<bf16>::new(2)),
            Out { ptr: Tensor::<bf16>::new(3), rows: 5, width: 0 },
            Const::new(1e-5),
            Const::new(128))
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
