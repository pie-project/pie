//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

use kernels_macros::routine;

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
/// The order of `x, w, out` is the SHADER's and not the trace's. A trace
/// states inputs, outputs, then weights, so binding positionally puts the
/// output where the norm weight belongs — and nothing reports it, because
/// every one of these is a storage buffer and a bind group typed by the
/// LAYOUT accepts them in any order. That is why this was the first row in
/// the tree to state its operands, and here it is the argument order.
///
/// # All five of the struct's fields are STATED, where two were and three
/// were unnameable
///
/// `eps`, `axis_size`, `w_stride`, `plus_one` and `gain` were a `RmsParams`
/// block this body forwarded whole, and only the first two appeared here at
/// all — `eps` for no other reason than to hold slot 0 open so `axis` could
/// be slot 1. The other three were read by `norm/rms.wgsl` out of words 2, 3
/// and 4 and named NOWHERE in Rust, so nothing in this crate could say what a
/// norm's gain was, whether it took gemma's `1 + w`, or how far apart the
/// weight's channels sat; a statement that got one of them wrong produced
/// numbers rather than an error, at every one of the 1084 rectangles this
/// entrypoint is.
///
/// All five are marks now. They are the same five words of the same statement
/// run, reached by INDEX instead of by struct field — nothing about which
/// numbers arrive has changed — and `norm/rms.wgsl` reads them out of the
/// `@group(1)` uniform block `driver-wgpu::lowering::routine::bind` packs
/// from this argument list, in this order.
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
    // THE STRUCT'S FIVE FIELDS, IN THE STRUCT'S ORDER, because that order is
    // the statement's: the block was staged from words 0..5 of this
    // statement's run and every mark below reads the word its field sat at.
    // `w: Const<Tensor<bf16>>` above does not disturb the count -- a
    // `Const<Tensor<E>>` claims the WEIGHT run and not the params one, so
    // `eps` is params slot 0 with a weight declared in front of it.
    //
    // The body reads only `axis`, and `x.width` will not do instead: a row is
    // the axis only when it holds ONE norm, which the strided twin below is
    // the counterexample to.
    eps: Const<f32>,
    axis: Const<i32>,
    // THE THREE THE SHADER READ AND NOTHING HERE COULD NAME. `w_stride` is
    // the distance between consecutive CHANNELS of the norm weight, `plus_one`
    // is the gemma convention flag `gain_at` folds as `1 + w`, and `gain` is
    // the scale applied in float before the single bf16 round. All three are
    // passed and none is read here, which is a different thing from the dead
    // marks this signature used to carry: they are the shader's to use, and
    // this body's only job is to put them where it reads them.
    //
    // `u32` and not `i32` for the middle two because no arithmetic here wants
    // a sign, and `Arg::unpack` for a `Const<u32>` takes the unsigned carrier
    // and refuses the signed one -- so the spelling is a claim a caller has to
    // match. The shader spells all three `u32`/`f32` to match.
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>) -> Result<(), Refusal> {
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/rms.wgsl", "rms_single_row_bfloat16").apply(per_axis(width, *axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis.arg(), w_stride.arg(), plus_one.arg(), gain.arg()],
    )
}

/// [`rms_single_row`] over rows a `row_pitch` apart rather than an axis apart.
///
/// The pitch goes LAST, after the five the packed form also passes, so that
/// folding a stride in does not renumber the fields the two share — the same
/// rule `norm::gated_rms_strided` follows, and the reason `norm/rms.wgsl`
/// declares its strided block as `{ eps, axis_size, w_stride, plus_one, gain,
/// row_pitch }` and not in any other order.
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
    // [`rms_single_row`]'s five, at the same five words of the same run.
    //
    // THIS BODY READS NONE OF THEM, and passes all five. On this plane the
    // group size is the shader's own `@workgroup_size(PIE_LANES)`, so the
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
        Fire::at("norm/rms.wgsl", "rms_strided_row_bfloat16").apply(per_row(rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), row_pitch.arg()],
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
        Fire::at("norm/rms.wgsl", "rms_strided_head_row_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), row_pitch.arg()],
    )
}

/// [`rms_single_row`] with the block residual folded into its epilogue.
///
/// # This form stated NO scalar at all, and read five
///
/// It forwarded `ctx.params()` and declared nothing beside its four operands,
/// so the whole of `RmsParams` reached `norm/rms.wgsl` through a block no
/// signature described. Its four siblings at least held two slots open; this
/// one and [`rms_residual_scaled`] held none, which is why a reader looking
/// for what a gemma-4 norm sandwich actually scales by had to go to the
/// shader to find out. The five marks below are that answer, stated where the
/// rest of the argument list is.
///
/// The residual binds at `@group(0) @binding(3)`, which is where the deleted
/// `RmsParams` block used to sit: the block was NOT the last binding, so its
/// departure closed a hole rather than trimming a tail, and `norm/rms.wgsl`'s
/// header carries the argument for why a hole could not simply be left.
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
    // belongs in a change that can measure the move. `vnorm_single_row`
    // carries the same note for the same reason.
    let axis = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/rms.wgsl", "rms_residual_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis_size.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), r.arg()],
    )
}

/// [`rms_residual`] with a per-layer gain beside the residual.
///
/// `s` is a one-element BUFFER and `gain` below is a stated word, and the two
/// are not the same number wearing different carriers: the buffer is the
/// checkpoint's per-layer embedding scale, read as `s[0]` by every lane after
/// the add, while `gain` is `RmsParams`'s own field, folded into the weight
/// before the norm's multiply. Both were reaching this shader before and only
/// one of them was visible from Rust.
///
/// `s` binds at `@group(0) @binding(4)`, one past the residual, for the reason
/// [`rms_residual`] gives about the binding the deleted block vacated.
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
        Fire::at("norm/rms.wgsl", "rms_residual_scaled_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis_size.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), r.arg(), s.arg()],
    )
}

/// A norm with no GAIN: the row divided by its own RMS and nothing else.
///
/// gemma's value norm. The absence of a weight is the whole difference from
/// [`rms_single_row`], and the axis is the HEAD — without that the fire's
/// width is taken for the axis and the whole row reduced as one, which is not
/// a smaller normalization but a different number in every channel.
///
/// # The two scalars are STATED, and the launch still does not read them
///
/// `eps` and `axis_size` rode a `VNormParams` block this body forwarded whole,
/// so this signature could name neither and the paragraph above described a
/// number nothing here could see. They are marks now — the same two words of
/// the same statement run, reached by index instead of by struct field — and
/// `norm/vector.wgsl` reads them out of the `@group(1)` block
/// `driver-wgpu::lowering::routine::bind` packs from this argument list.
///
/// What that does NOT change is where the LAUNCH gets its axis. `per_axis` is
/// still handed `x.width`, so a row holding several heads becomes one
/// workgroup while the shader spans `wg.x * axis_size` — and
/// `kernels-metal::norm::vnorm_single_row` measures what that costs, because
/// its twin had the identical gap: gemma-4-26b-a4b's V is 8 heads of 256 over
/// 4 rows, so `width / axis == 1` gave 4 threadgroups where 32 were owed and
/// heads 4..7 of every row kept whatever the arena held. Metal closed it by
/// sizing `rms_grid` from the mark. Here the mark now EXISTS to close it with,
/// and it is deliberately not spent yet: the fix moves the dispatch, and a
/// dispatch that moves belongs in a change that can measure the move rather
/// than in the one that made the number nameable.
///
/// # Errors
///
/// As `per_axis`: an empty width, axis or row count, an axis wider than its
/// row, or a lane count that overflows.
#[routine]
pub fn vnorm_single_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE STRUCT'S TWO FIELDS, BOTH OF THEM LIVE. `VNormParams { eps,
    // axis_size }` was staged from this statement's run and read as a struct;
    // these are the same two words reached by index, IN THE STRUCT'S ORDER
    // because that order is the statement's. The shader spells `axis_size`
    // `u32` and the mark spells it `i32`, exactly as `RmsParams.axis_size`
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
        Fire::at("norm/vector.wgsl", "vnorm_single_row_bfloat16").apply(per_axis(width, axis, rows)?),
        &[x.arg(), out.arg(), eps.arg(), axis_size.arg()],
    )
}

/// The gated-delta-net value norm: `out = w * (x / rms(x)) * silu(z)`.
///
/// # One block, where this family had two
///
/// This was the only kernel in `norm/` whose scalars reached its shader by two
/// roads at once: `eps` and `vd` were a `GatedRmsParams` storage block staged
/// from the statement's run, while `gated_rms_strided`'s `row_pitch` was a
/// number this body passed and therefore a `@group(1)` uniform of its own.
/// `driver-wgpu::lowering::routine::bind` carries a branch for exactly that
/// pair — written for `ssm/gdn_prep.wgsl` — which sends the body's scalars to
/// the uniform and the statement's run to the storage pointer.
///
/// Both scalars are MARKS now, which is the same two words of the same run
/// reached by index instead of by struct field, so this fire takes that branch
/// no longer: there is one block, `norm/gated_rms.wgsl` declares it, and its
/// fields are this argument list in this order.
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
        Fire::at("norm/gated_rms.wgsl", "gated_rms_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), z.arg(), w.arg(), out.arg(), eps.arg(), vd.arg()],
    )
}

/// [`gated_rms`] over heads packed inside a wider row.
///
/// The pitch goes LAST, after the pair the packed form also passes, so that
/// folding a stride in does not renumber the fields the two variants share —
/// the same rule the strided norms beside it follow, and the reason
/// `norm/gated_rms.wgsl` declares its block as `{ eps, vd, row_pitch }` and
/// not in any other order.
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
    out: Out<Tensor<bf16>>,
    // [`gated_rms`]'s two, at the same two words of the same run.
    eps: Const<f32>,
    vd: Const<i32>) -> Result<(), Refusal> {
    let heads = ctx.ask::<i32, keys::VHeads>()?;
    let row_pitch = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/gated_rms.wgsl", "gated_rms_strided_bfloat16").apply(per_head_row(heads, rows)?),
        &[x.arg(), z.arg(), w.arg(), out.arg(), eps.arg(), vd.arg(), row_pitch.arg()],
    )
}

/// gemma's per-layer scale: `out = x * scalar`, the scalar read from a buffer.
///
/// Which layer is running is the FIRE's, so the number is a buffer rather
/// than a stated scalar.
///
/// AND THERE IS NO SECOND OPERAND BESIDE IT. This forwarded `ctx.params()` as
/// well, for a `LayerScalarParams` whose one field the shader did not read --
/// the hidden width, which the grid already is. A block bound for the shape of
/// a row rather than for anything in it is a binding every plane has to
/// declare and stage to arrive at nothing, so the block is gone from all three
/// shaders and this signature states no mark in its place.
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
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/layer_scalar.wgsl", "layer_scalar_mul_bfloat16").apply(elementwise_rows(width, rows)?),
        &[x.arg(), scalar.arg(), out.arg()],
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

