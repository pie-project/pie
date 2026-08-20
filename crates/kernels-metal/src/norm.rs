//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.


use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows, keys};

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[
    ("norm/add_bias.metal", "add_bias_bfloat16"),
    ("norm/gated_rms.metal", "gated_rms_bfloat16"),
    ("norm/gated_rms.metal", "gated_rms_strided_bfloat16"),
    ("norm/layer_scalar.metal", "layer_scalar_mul_bfloat16"),
    ("norm/residual_add.metal", "residual_add_bfloat16"),
    ("norm/residual_add.metal", "residual_add_strided_bfloat16"),
    ("norm/rms.metal", "rms_residual_bfloat16"),
    ("norm/rms.metal", "rms_residual_scaled_bfloat16"),
    ("norm/rms.metal", "rms_single_row_bfloat16"),
    ("norm/rms.metal", "rms_strided_head_row_bfloat16"),
    ("norm/rms.metal", "rms_strided_row_bfloat16"),
    ("norm/vector.metal", "vnorm_single_row_bfloat16"),
];

/// Threads per threadgroup for the two elementwise bodies here, for the
/// reason `mlp.rs` states it: Metal declares no group size in the source, so
/// the number lives on this side and reaches the encoder as the second half
/// of `dispatchThreads:threadsPerThreadgroup:`.
const GROUP_X: u32 = 256;

/// The reduction's threadgroup: one lane per `N_READS = 4` channels of the
/// axis, capped at the 1024 a Metal threadgroup may be.
///
/// Rounded UP. The kernel guards its own tail, but a truncating count drops
/// the last partial group of four -- and those channels are not left stale,
/// they are left out of the SUM, so every channel of the row is normalized by
/// a divisor that is too small.
fn rms_threads(axis: i32) -> Result<u32, Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
    }
    Ok(axis.unsigned_abs().div_ceil(4).min(1024))
}

/// `LaunchRule::Rms`: one threadgroup per AXIS, not per row.
///
/// The distinction is the whole content of this helper. A hidden-state norm
/// has `axis == width` and this reduces to one threadgroup per row. A QK-norm
/// does not: a per-head norm packs `width / axis` of them into each row, and
/// each is its own reduction. Handing such a fire one threadgroup per row
/// normalizes head 0 and leaves every other head as the projection wrote it
/// -- fully written, not fully normalized, and no read of the output reports
/// it.
fn rms_grid(width: i32, axis: i32, rows: i32) -> Result<([u32; 3], [u32; 3]), Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
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
    let t = rms_threads(axis)?;
    let norms = width.unsigned_abs() / axis.unsigned_abs();
    let lanes = t
        .checked_mul(norms)
        .and_then(|n| n.checked_mul(rows.unsigned_abs()))
        .ok_or(Refusal::Grid {
            what: "axis threads * norms per row * rows",
            at: i64::from(t) * i64::from(norms) * i64::from(rows),
        })?;
    Ok(([lanes, 1, 1], [t, 1, 1]))
}

/// One threadgroup per `(head, row)` pair, for the two-level bases.
///
/// A token holds `heads` norms packed an axis apart and the next token is a
/// uniform `row_pitch` away, so the base is two-level and one grid axis cannot
/// carry both terms -- flattening would need the two strides commensurate, and
/// the pitch is the whole token while the axis is one head of it. The launch
/// gives it two: y is the head and z is the row, which is what makes the
/// threadgroup's own position the pair.
fn head_row_grid(threads: u32, heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([threads, heads.unsigned_abs(), rows.unsigned_abs()])
}

/// `out = w * (x / rms(x))`, one threadgroup per axis.
///
/// The order of `x, w, out` is the SHADER's and not the trace's. A trace
/// states inputs, outputs, then weights, so binding positionally puts the
/// output where the norm weight belongs -- and nothing reports it, because
/// Metal does not validate a binding. That mismatch is why this was the first
/// row in the tree to state its operands, and here it is the argument order.
///
/// # All five of the struct's fields are STATED, where two were and three
/// were unnameable
///
/// `eps`, `axis_size`, `w_stride`, `plus_one` and `gain` were a `RmsParams`
/// block this body forwarded whole, and only the first two appeared here at
/// all -- `eps` for no other reason than to hold slot 0 open so `axis` could
/// be slot 1. The other three were read by `norm/rms.metal` out of words 2, 3
/// and 4 and named NOWHERE in Rust, so nothing in this crate could say what a
/// norm's gain was, whether it took gemma's `1 + w`, or how far apart the
/// weight's channels sat.
///
/// All five are marks now: the same five words of the same statement run,
/// reached by INDEX instead of by struct field, and the entrypoint takes them
/// as five `constant` references at buffers 3 through 7 where it took one
/// staged block at 3.
///
/// # Errors
///
/// See [`rms_grid`].
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
    // `plus_one` is the gemma convention flag `rms_row_body` folds as
    // `1 + w`, and `gain` is the scale applied in float before the single
    // bf16 round. All three are passed and none is read here, which is a
    // different thing from the dead marks this signature used to carry: they
    // are the shader's to use, and this body's only job is to put them where
    // it reads them.
    //
    // `u32` and not `i32` for the middle two because no arithmetic here wants
    // a sign, and `Arg::unpack` for a `Const<u32>` takes the unsigned carrier
    // and refuses the signed one -- so the spelling is a claim a caller has to
    // match. The kernel spells all three `uint`/`float` to match.
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>) -> Result<(), Refusal> {
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let (lanes, group) = rms_grid(width, *axis, rows)?;
    ctx.fire(
        Fire::at(RMS_FILE, "rms_single_row_bfloat16").apply(Grid::of(lanes, group)),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis.arg(), w_stride.arg(), plus_one.arg(), gain.arg()],
    )
}

/// [`rms_single_row`] over rows that are a `row_pitch` apart rather than an
/// axis apart.
///
/// One threadgroup per ROW here and not per axis: the base is `gid *
/// row_pitch`, so a row holds exactly one norm and the axis only sizes the
/// threadgroup. That is the difference from [`rms_single_row`], whose single
/// grid axis carries both the norm and the row.
///
/// The pitch goes LAST, after the five the packed form also binds, so that
/// folding a stride in does not renumber what the two share: it was buffer 4
/// behind a block of five words and it is buffer 8 in front of five scalars,
/// which is the same place.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty axis or row count, and [`Refusal::Grid`]
/// when the lane total does not fit a `u32`.
#[routine]
pub fn rms_strided_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // [`rms_single_row`]'s five, at the same five words of the same run.
    //
    // The AXIS is the one this body reads, and it is NOT `x.width`: a strided
    // row holds several norms and `x.width` is the pitch across all of them,
    // so the threadgroup is sized from the mark. That is the difference from
    // the wgpu and vulkan twins, where the group width is the shader's own
    // and the axis reaches no launch -- a difference about the LAUNCH and not
    // about the argument list, since the kernel spans `axis_size` on every
    // plane.
    eps: Const<f32>,
    axis: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>) -> Result<(), Refusal> {
    let row_pitch = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let t = rms_threads(*axis)?;
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let lanes = t.checked_mul(rows.unsigned_abs()).ok_or(Refusal::Grid {
        what: "axis threads rows",
        at: i64::from(t) * i64::from(rows),
    })?;
    ctx.fire(
        Fire::at(RMS_FILE, "rms_strided_row_bfloat16").apply(Grid::of([lanes, 1, 1], [t, 1, 1])),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), row_pitch.arg()],
    )
}

/// The per-head q/k norms over a whole prompt: `heads` norms inside each row,
/// `rows` of them.
///
/// # Errors
///
/// See [`head_row_grid`].
#[routine]
pub fn rms_strided_head_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // [`rms_single_row`]'s five, at the same five words of the same run, and
    // the pitch after them for [`rms_strided_row`]'s reason.
    //
    // The AXIS sizes the threadgroup here as it does there, and it is NOT
    // `x.width`: a strided row holds several norms and `x.width` is the pitch
    // across all of them, so the head count below is the one divided by the
    // other.
    eps: Const<f32>,
    axis: Const<i32>,
    w_stride: Const<u32>,
    plus_one: Const<u32>,
    gain: Const<f32>) -> Result<(), Refusal> {
    let row_pitch = x.width;
    // HOW MANY NORMS FIT THE ROW, which is the division HEAD spelled
    // `Over<Say<Width>, Else<Nth<1>, ..>>` and no driver answers as a fact.
    let heads = if *axis > 0 { row_pitch / *axis } else { 0 };
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let t = rms_threads(*axis)?;
    ctx.fire(
        Fire::at(RMS_FILE, "rms_strided_head_row_bfloat16").apply(Grid::of(head_row_grid(t, heads, rows)?, [t, 1, 1])),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), row_pitch.arg()],
    )
}

/// [`rms_single_row`] with the block residual folded into its epilogue.
///
/// The residual is buffer 8 and it USED TO BE 4, because the params struct it
/// arrives after is five scalars now rather than one block. The rule it
/// follows is unchanged: the conditional binding comes after the
/// unconditional ones, so folding does not renumber what every form shares --
/// and what every form shares is now the three operands AND the five scalars
/// at 3 through 7.
///
/// # This form stated NO scalar at all, and read five
///
/// It forwarded `ctx.params()` and declared nothing beside its four operands,
/// so the whole of `RmsParams` reached `norm/rms.metal` through a block no
/// signature described. Its three siblings at least held two slots open; this
/// one and [`rms_residual_scaled`] held none.
///
/// # Errors
///
/// See [`rms_grid`].
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
    // operand precedes a scalar, and the fire array is the KERNEL's buffer
    // table, where the residual is the conditional binding and therefore
    // last.
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
    let (lanes, group) = rms_grid(width, axis, rows)?;
    ctx.fire(
        Fire::at(RMS_FILE, "rms_residual_bfloat16").apply(Grid::of(lanes, group)),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis_size.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), r.arg()],
    )
}

/// [`rms_residual`] with a per-layer gain beside the residual.
///
/// `s` is a one-element BUFFER at 9 and `gain` below is a stated word at 7,
/// and the two are not the same number wearing different carriers: the buffer
/// is the checkpoint's per-layer embedding scale, read as `s[0]` by every
/// thread after the add, while `gain` is `RmsParams`'s own field, folded into
/// the weight before the norm's multiply. Both were reaching this kernel
/// before and only one of them was visible from Rust.
///
/// # Errors
///
/// See [`rms_grid`].
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
    let (lanes, group) = rms_grid(width, axis, rows)?;
    ctx.fire(
        Fire::at(RMS_FILE, "rms_residual_scaled_bfloat16").apply(Grid::of(lanes, group)),
        &[x.arg(), w.arg(), out.arg(), eps.arg(), axis_size.arg(), w_stride.arg(), plus_one.arg(), gain.arg(), r.arg(), s.arg()],
    )
}

/// A norm with no GAIN: the row divided by its own RMS and nothing else.
///
/// gemma's value norm, and the absent weight is the whole difference from
/// [`rms_single_row`] -- which is why `out` is buffer 1 here and 2 there.
///
/// # The axis is a HEAD and never the row
///
/// This read `axis = x.width` and it is the one norm in this file that can
/// never mean that. `Deployment::v_norm` is a PER-HEAD RMS over the value
/// projection: `dsl::metal::vnorm` states `head_dim` in `params[1]` and
/// `norm/vector.metal` reads that slot as `axis_size`, so the SHADER always
/// normalized 256 channels while this body sized the launch for 2048.
///
/// `rms_grid`'s own doc says what that costs -- *"Handing such a fire one
/// threadgroup per row normalizes head 0 and leaves every other head as the
/// projection wrote it"* -- and it is worse here than there, because the
/// output is a fresh arena value rather than the input: gemma-4-26b-a4b's V
/// is 8 heads of 256 over 4 rows, so `width/axis == 1` gave 4 threadgroups
/// where 32 were owed, and threadgroups 0..3 wrote row 0's FIRST FOUR heads
/// while heads 4..7 and rows 1..3 kept whatever the arena held from an
/// earlier statement. Measured against MLX at position zero: head 0 agreed
/// to four decimals and element 1043 -- head 4 -- read 72.5 where MLX says
/// -0.124. Every layer's attention then read three quarters of a stale value
/// tensor, which is position-INDEPENDENT and NaN-free, and the readout was a
/// fluent-looking distribution with the wrong argmax (236772 against MLX's
/// 3643).
///
/// So the axis is declared and read, exactly as [`rms_single_row`] declares
/// and reads it -- and `eps` is no longer declared only to hold slot 0 open
/// for it. The struct it held the slot open INSIDE is gone: `norm/vector.metal`
/// takes the two as buffers 2 and 3, one `setBytes` each where the routine
/// bound a staged `VNormParams`, and both marks are now spent rather than one
/// spent and one held. The words are the statement's own, at the indices the
/// struct laid its fields out at, because that order was the statement's.
///
/// # Errors
///
/// See [`rms_grid`].
#[routine]
pub fn vnorm_single_row(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE STRUCT'S TWO FIELDS, BOTH OF THEM NOW BOUND. `eps` was declared
    // only so the AXIS after it had a slot to be at, and the axis is the
    // norm's own width -- the number this body sizes its launch from, and the
    // one `dsl::metal::vnorm` states at `params[1]`.
    eps: Const<f32>,
    axis: Const<i32>) -> Result<(), Refusal> {
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let (lanes, group) = rms_grid(width, *axis, rows)?;
    ctx.fire(
        Fire::at("norm/vector.metal", "vnorm_single_row_bfloat16").apply(Grid::of(lanes, group)),
        &[x.arg(), out.arg(), eps.arg(), axis.arg()],
    )
}

/// GDN's gated norm: `w * rms(x) * silu(z)`, over each value head.
///
/// The row axis is stated here and `LaunchRule::GatedRms` does not state it.
/// That is a DEFECT this port found, not a difference of spelling: the body
/// indexes `(tgpos.z * tpg.y + tgpos.y) * vd + lid`, so with `grid.z = 1` only
/// row 0 is normalized and every other token keeps whatever the core scan
/// wrote. A decode is one row, which is why the golden `gdn_core` tag has been
/// green over this the whole time; a PREFILL through this symbol would norm
/// its first token and no other. `LaunchRule::RouterLane` records the same
/// finding about the same missing axis, already fixed.
///
/// # The head dim is a MARK, and it used to be an ask because it could not be
///
/// `eps` and `vd` were `GatedRmsParams`, forwarded whole, so a `Const` derived
/// onto slot 0 would have read the struct's first field's bits rather than a
/// scalar run -- and this body needs `vd` for the threadgroup width. It asked
/// the FIRE for it instead, through `keys::VDim`, which is the same number by
/// construction and a second question about it.
///
/// The struct is gone, so the question is too: `vd` is word 1 of the
/// statement's own run, `norm/gated_rms.metal` reads the same word as buffer 5
/// for its indexing, and the threadgroup this body sizes and the stride that
/// shader strides by are now ONE number rather than two that agreed. The head
/// COUNT stays an ask, because `keys::VHeads` never was a word of this run.
///
/// # Errors
///
/// See [`head_row_grid`], plus [`Refusal::Empty`] for an empty head width.
#[routine]
pub fn gated_rms(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    z: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE STRUCT'S TWO FIELDS, IN THE STRUCT'S ORDER because that order is
    // the statement's. `eps` reaches only the shader; `vd` reaches the
    // threadgroup width as well, which is what the section above is about.
    eps: Const<f32>,
    vd: Const<i32>) -> Result<(), Refusal> {
    let heads = ctx.ask::<i32, keys::VHeads>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let t = head_width(*vd)?;
    ctx.fire(
        Fire::at(GATED_FILE, "gated_rms_bfloat16").apply(Grid::of(head_row_grid(t, heads, rows)?, [t, 1, 1])),
        &[x.arg(), z.arg(), w.arg(), out.arg(), eps.arg(), vd.arg()],
    )
}

/// [`gated_rms`] over rows a `row_pitch` apart.
///
/// The strided form needs no `threadgroups_per_grid`: its base is `tgpos.z *
/// row_pitch + tgpos.y * vd`, both terms explicit. That is what makes it the
/// prefill form -- the packed one infers the row stride from the grid's own
/// shape, which only holds when the heads are contiguous.
///
/// # Errors
///
/// See [`gated_rms`].
#[routine]
pub fn gated_rms_strided(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    z: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // [`gated_rms`]'s two, at the same two words of the same run. The pitch
    // after them keeps the END of the list, so folding a stride in does not
    // renumber the buffers the two entrypoints share: it was buffer 5 behind
    // a block of two words and it is buffer 6 in front of two scalars.
    eps: Const<f32>,
    vd: Const<i32>) -> Result<(), Refusal> {
    let heads = ctx.ask::<i32, keys::VHeads>()?;
    let row_pitch = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let t = head_width(*vd)?;
    ctx.fire(
        Fire::at(GATED_FILE, "gated_rms_strided_bfloat16").apply(Grid::of(head_row_grid(t, heads, rows)?, [t, 1, 1])),
        &[x.arg(), z.arg(), w.arg(), out.arg(), eps.arg(), vd.arg(), row_pitch.arg()],
    )
}

/// One lane per channel of a value head, which is also the threadgroup: the
/// gated norm reduces across the whole head in one group and has no outer
/// loop, so the head dim IS the group width and 1024 is a hard ceiling.
fn head_width(vd: i32) -> Result<u32, Refusal> {
    if vd <= 0 {
        return Err(Refusal::Empty { what: "vd" });
    }
    if vd > 1024 {
        return Err(Refusal::Wide {
            what: "vd",
            at: i64::from(vd),
            max: 1024,
        });
    }
    Ok(vd.unsigned_abs())
}

/// gemma's per-layer scale: `out = x * scalar[0]`, elementwise.
///
/// AND NOTHING IS BOUND BESIDE THEM. This forwarded `ctx.params()` for a
/// `LayerScalarParams` holding a hidden width the body bounds itself with the
/// grid instead, and the buffer stayed only because the entrypoint declared it
/// -- the struct's own header says so at length. The entrypoint no longer
/// declares it, so the block is gone and this states no mark in its place.
///
/// # Errors
///
/// See [`elementwise`].
#[routine]
pub fn layer_scalar_mul(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    scalar: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("norm/layer_scalar.metal", "layer_scalar_mul_bfloat16").apply(Grid::of(elementwise(width, rows)?, [GROUP_X, 1, 1])),
        &[x.arg(), scalar.arg(), out.arg()],
    )
}

/// `out = x + residual`, elementwise, and `out` may alias `x`.
///
/// Stated because a MIXTURE demands it: a routed FFN's rows are already
/// down-projected and combined, so all the block owes is the add, where a
/// dense FFN fuses it into the down projection and never names this symbol.
///
/// # Errors
///
/// See [`elementwise`].
#[routine]
pub fn residual_add(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(RESIDUAL_FILE, "residual_add_bfloat16").apply(Grid::of(elementwise(width, rows)?, [GROUP_X, 1, 1])),
        &[x.arg(), residual.arg(), out.arg()],
    )
}

/// [`residual_add`] over rows a `row_pitch` apart, so a whole prompt is one
/// dispatch instead of one per token.
///
/// The rows go on their own grid axis rather than being folded into a flat
/// count, because a pitch is not a width: the row's own extent is what the
/// threads cover and the pitch is only how far the next one starts.
///
/// # Errors
///
/// See [`elementwise_rows`].
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
        Fire::at(RESIDUAL_FILE, "residual_add_strided_bfloat16").apply(Grid::of(elementwise_rows(width, rows)?, [GROUP_X, 1, 1])),
        &[x.arg(), residual.arg(), out.arg(), row_pitch.arg()],
    )
}

/// The Qwen-2 family's attention biases, added in place.
///
/// The width is an ARGUMENT and not an `Env`: the kernel is told it, because
/// `tid.y * width + tid.x` needs the row stride and the grid only carries the
/// extent. Its threadgroup is the row clamped to 256 rather than a flat 256,
/// so a row narrower than that does not launch threads past its own end and
/// rely on the grid to retire them.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty width or row count.
#[routine]
pub fn add_bias(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = out.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let lanes = elementwise_rows(width, rows)?;
    ctx.fire(
        Fire::at("norm/add_bias.metal", "add_bias_bfloat16").apply(Grid::of(lanes, [lanes[0].min(GROUP_X), 1, 1])),
        &[out.arg(), bias.arg(), width.arg()],
    )
}

const RMS_FILE: &str = "norm/rms.metal";
const GATED_FILE: &str = "norm/gated_rms.metal";
const RESIDUAL_FILE: &str = "norm/residual_add.metal";


#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do, and answers the
    /// facts this family's bodies ask for.
    ///
    /// `rows` backs every `ctx.ask::<i32, keys::Rows>()` here. `row_pitch`
    /// answers `residual_add_strided`'s own `keys::RowPitch` ask -- the one
    /// body in this file where the pitch is still a fact independent of
    /// `x.width`, because `rms_single_row` and `rms_strided_row` fold their
    /// pitch into that field instead. `params_handle` answers `ctx.params()`.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        rows: Cell<i32>,
        vd: Cell<i32>,
        vheads: Cell<i32>,
        row_pitch: Cell<i32>,
        params_handle: Cell<u32>,
        /// THE STATEMENT\'S SCALAR RUN, for a body that reads a word by
        /// index. Empty means "4096 at every slot", which is a plausible
        /// stride for the rows these tests build; a case that means a
        /// particular tiling or split count sets its own.
        words: RefCell<Vec<i32>>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                rows: Cell::new(2),
                vd: Cell::new(128),
                vheads: Cell::new(16),
                row_pitch: Cell::new(4096),
                params_handle: Cell::new(4),
                words: RefCell::default(),
            }
        }
    }

    impl Encode for Seen {
        // A PROBE HAS NO FIRE BEHIND IT, so it answers only the facts this
        // file's bodies ask for and refuses everything else honestly --
        // answering zero for an unasked fact would let a body under test pass
        // while the fact it asked for went unanswered on a real driver.
        fn resolve(&self, _ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            // The geometry these bodies read now that their params run is a
            // STRUCT and no slot in it is theirs to take.
            if source == <keys::VDim as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.vd.get()));
            }
            if source == <keys::VHeads as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.vheads.get()));
            }
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            // THE STATEMENT'S OWN SCALARS, which a body reads by index when its
            // params run is a struct and no `Const` mark can name a word inside
            // it -- see `Asks::param`. The probe answers a number that is
            // plausible for every reader: a stride wide enough for the rows
            // these tests build, and a positive tiling.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                return Ok(ArgValue::I32(
                    self.words.borrow().get(usize::from(n)).copied().unwrap_or(4096),
                ));
            }
            if source == kernels::Source::Slot(kernels::Kind::Params, 0) {
                return Ok(ArgValue::Buffer(self.params_handle.get()));
            }
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// The only call the recorder made.
    fn one(seen: &Seen) -> Call {
        let calls = seen.calls.borrow();
        assert_eq!(calls.len(), 1, "one norm is one dispatch");
        calls[0].clone()
    }

    /// The kernel declares `x, w, out` and then its scalars, and a trace
    /// states inputs, outputs, then weights. Binding in the trace's order puts
    /// the OUTPUT where the norm weight belongs, which is the defect that made
    /// this family the first to state its operands -- and Metal does not
    /// validate a binding, so the only thing that reports it is a test like
    /// this one.
    ///
    /// KNOWN FAILING, upstream of this crate: `out`'s `ArgValue::BufferMut(3)`
    /// is the correct claim for what an `Out<Tensor<bf16>>` SHOULD produce;
    /// `mlp::tests::all_four_bodies_bind_gate_up_and_out_at_zero_one_and_two`
    /// documents in full why `Tensor<E>`'s one `handle` field and its
    /// direction-blind `Bind` impl (`crates/kernels/src/shader.rs`, outside
    /// this crate) mean no positional argument a routine body binds itself
    /// can presently come out mutable, on any plane.
    #[test]
    fn the_row_norm_binds_the_weight_before_its_output() {
        let seen = Seen::default();
        rms_single_row(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 8 },
            Const::new(Tensor::<bf16>::new(2)),
            Out::new(Tensor::<bf16>::new(3)),
            // The struct's five fields, in the struct's order. The stride is
            // one channel, the gemma `+1` is off and the gain is unity --
            // three words this routine could not name until they became marks,
            // and three this test therefore could not have stated before.
            Const::new(1e-5),
            Const::new(8),
            Const::new(1),
            Const::new(0),
            Const::new(1.0))
        .expect("a launch");
        assert_eq!(
            one(&seen).1,
            [
                ArgValue::Buffer(1),
                ArgValue::Buffer(2),
                ArgValue::BufferMut(3),
                ArgValue::F32(1e-5),
                ArgValue::I32(8),
                ArgValue::U32(1),
                ArgValue::U32(0),
                ArgValue::F32(1.0),
            ],
            "x, w, out and then the five scalars -- the shader's order and not \
             the trace's, and where the staged block used to be one handle it \
             is five words now"
        );
    }

    /// A per-head norm packs `width / axis` reductions into each row and each
    /// is its own threadgroup.
    ///
    /// One threadgroup per ROW instead normalizes head 0 and leaves the other
    /// heads as the projection wrote them. That is not a smaller
    /// normalization: the output is fully written and only partly normalized,
    /// so nothing downstream can tell. gemma-4 is the model that has it --
    /// 32 heads of 256 channels in an 8192-wide Q -- and llama has no QK-norm,
    /// which is why the golden gate was green over it.
    #[test]
    fn a_per_head_norm_gets_a_threadgroup_per_head_and_not_per_row() {
        let (lanes, group) = rms_grid(8192, 256, 2).expect("a grid");
        assert_eq!(group, [64, 1, 1], "256 channels over N_READS = 4");
        assert_eq!(
            lanes,
            [64 * 32 * 2, 1, 1],
            "32 heads a row, 2 rows, each its own threadgroup"
        );

        let (whole, _) = rms_grid(8192, 8192, 2).expect("a grid");
        assert_eq!(
            whole,
            [1024 * 2, 1, 1],
            "a hidden-state norm spans its row, so this is one group a row \
             -- capped at the 1024 threads a threadgroup may be"
        );
    }

    /// `LaunchRule::GatedRms` is `(vd, heads, 1)` and the body indexes
    /// `(tgpos.z * tpg.y + tgpos.y) * vd`, so under that rule a prefill norms
    /// its first token and leaves every other one as the core scan wrote it.
    ///
    /// A decode is one row, which is why `gdn_core` has been green over this.
    /// The routine states the axis. `LaunchRule::RouterLane`'s own doc records
    /// the identical finding about the identical missing axis.
    #[test]
    fn the_gated_norm_puts_its_rows_on_the_axis_the_body_reads() {
        let seen = Seen::default();
        seen.rows.set(7);
        gated_rms(
            &seen,
            In::new(Tensor::<bf16>::new(1)),
            In::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            Out::new(Tensor::<bf16>::new(4)),
            // The head dim is a MARK now and not a `keys::VDim` ask, so the
            // 128 the threadgroup comes out as is stated here rather than
            // answered by `Seen`.
            Const::new(1e-5),
            Const::new(128))
        .expect("a launch");
        let (fire, _) = one(&seen);
        assert_eq!(
            fire.lanes,
            [128, 16, 7],
            "the row is grid.z, and it is 7 and not 1"
        );
        assert_eq!(fire.group, [128, 1, 1], "the head dim is the threadgroup");
    }

    /// The strided forms take their pitch LAST, after every buffer the packed
    /// form binds, so folding a stride in does not renumber what they share.
    ///
    /// `rms_strided_row`'s pitch and axis used to be independent facts under
    /// `kernel!` -- `row_pitch: Ask<keys::Width, i32>` and `axis: ParamOr<1,
    /// keys::Width, i32>`, fed 4096 and 8 here -- but both read the same KEY,
    /// so in a real dispatch they were always equal anyway; the test's two
    /// numbers were a fixture, not a real combination. Now that shared key IS
    /// the mark's own `width` field, so this file's `rms_strided_row` reads
    /// pitch and axis off the one number `x.width` gives it. The pitch this
    /// test checks is re-aimed from 4096 to 8 -- the axis this test already
    /// needed for its lanes -- rather than left asserting a value the body
    /// cannot produce once one field carries what were two.
    /// `residual_add_strided` keeps its pitch independent (`keys::RowPitch`,
    /// a fact its own body still asks the fire for), so its 4096 is
    /// unchanged.
    #[test]
    fn a_stride_lands_after_the_bindings_it_does_not_change() {
        let seen = Seen::default();
        rms_strided_row(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 8 },
            Const::new(Tensor::<bf16>::new(2)),
            Out::new(Tensor::<bf16>::new(3)),
            // The struct's five fields, which the statement carries: an
            // epsilon at slot 0, the norm's own axis at slot 1, and the three
            // the kernel reads and this body does not.
            Const::new(1e-5),
            Const::new(8),
            Const::new(1),
            Const::new(0),
            Const::new(1.0))
        .expect("a launch");
        residual_add_strided(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 8 },
            In::new(Tensor::<bf16>::new(2)),
            Out::new(Tensor::<bf16>::new(3)),
         Const::new(4096))
        .expect("a launch");
        let calls = seen.calls.borrow();
        assert_eq!(
            calls[0].1.last(),
            Some(&ArgValue::I32(8)),
            "rms_strided_row takes the pitch at 8 -- after the five scalars \
             that used to be one staged block at 3 -- and the pitch is the \
             axis now that one field is both"
        );
        assert_eq!(
            calls[1].1.last(),
            Some(&ArgValue::I32(4096)),
            "residual_add_strided takes it at 3"
        );
        assert_eq!(
            calls[0].0.lanes,
            [2 * 2, 1, 1],
            "one threadgroup a row here, because the pitch carries the row"
        );
    }

    /// An axis wider than the row it sits in is a mistake in the statement,
    /// not a shape to clamp. Clamping is what the row path does, and it turns
    /// `width / axis` into 1 -- the per-head norm above, launched as a whole
    /// row norm, silently.
    #[test]
    fn an_axis_wider_than_its_row_is_refused_rather_than_clamped() {
        assert!(matches!(
            rms_grid(256, 8192, 1),
            Err(Refusal::Wide { what: "axis", .. })
        ));
        assert!(matches!(
            rms_grid(256, 0, 1),
            Err(Refusal::Empty { what: "axis" })
        ));
        assert!(matches!(
            rms_grid(256, 256, 0),
            Err(Refusal::Empty { what: "rows" })
        ));
    }

    /// The bias is added IN PLACE, so the buffer it reads is the one it
    /// writes and there are only three arguments.
    ///
    /// KNOWN FAILING, upstream of this crate: `out`'s `ArgValue::BufferMut(1)`
    /// is the correct claim for what an `InOut<Tensor<bf16>>` SHOULD produce
    /// -- the same gap the row norm test above documents, here for `InOut`
    /// rather than `Out`; both delegate to the identical `Tensor<E>::Bind`.
    #[test]
    fn the_bias_binds_the_value_it_biases_once() {
        let seen = Seen::default();
        seen.rows.set(4);
        add_bias(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 96 },
            Const::new(Tensor::<bf16>::new(2)),
        )
        .expect("a launch");
        let (fire, args) = one(&seen);
        assert_eq!(
            args,
            [
                ArgValue::BufferMut(1),
                ArgValue::Buffer(2),
                ArgValue::I32(96)
            ],
            "out, bias, width -- and `out` is the input too"
        );
        assert_eq!(fire.lanes, [96, 4, 1]);
        assert_eq!(
            fire.group,
            [96, 1, 1],
            "the row, clamped to 256 -- not a flat 256 over a 96-wide row"
        );
        add_bias(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 4096 },
            Const::new(Tensor::<bf16>::new(2)),
        )
        .expect("a launch");
        assert_eq!(seen.calls.borrow()[1].0.group, [256, 1, 1], "and clamped");
    }
}
