//! The CUDA `norm` family: RMS norm and everything fused onto it.
//!
//! RMS norms, elementwise passes sharing their launch shape (residual/bias add,
//! `tanh`, scalar multiply), Gemma-3n's AltUp set, and DeepSeek's `hc_*` pre/post-passes.
//!
//! Every `# Safety` below also requires `ctx`'s stream live across the (async) launch.

use kernels::routine::Asks;
use kernels::{Bind, Fire, keys};
use kernels_macros::routine;
use crate::jit::{Ctx, Launch, aligned16};
use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, f16};
use kernels::Refusal;
use kernels::Region;
use kernels::Stride;
use kernels::routine::{Const, In, InOut, Out};

// `In<N, T>`/`Out<N, T>` name the statement's Nth operand, not the parameter
// position; stating it turns off `alias()`'s correction for that slot.
//
// `Bank<N, T>` (positional weight) and `Weight<N, T>` (named) both take a
// `*const T` and differ by four characters -- a swap at the same index
// compiles and only fails at launch. `Env<keys::X>` names a fact by type.

use core::ffi::c_void;
use core::ptr::NonNull;

/// `rmsnorm.cu`, `dsv4_hc.cu`, `elementwise.cuh` and `add_bias.cuh`'s block width.
const BLOCK: u32 = 256;

/// `rmsnorm.cu`'s `VBLOCK`.
const VBLOCK: u32 = 512;

/// The warp width.
const WARP: u32 = 32;

/// `altup.cu`'s block width.
const ALTUP_BLOCK: u32 = 128;

/// AltUp's own epsilon, not the model's configured one.
pub const ALTUP_EPS: f32 = 1e-5;

/// The width above which the vectorised fused norm prefers a 512-thread block.
pub const RASR_VEC512_ABOVE: i32 = 2560;

/// `dsv4_hc.cuh`'s `MAX_HC_MULT`.
pub const MAX_HC_MULT: i32 = 8;

/// One block per row, [`BLOCK`] wide, nothing shared.
#[must_use]
const fn per_row(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), BLOCK)
}

/// [`per_row`] with the warp-reduction scratch two `altup_aux` kernels share.
#[must_use]
const fn per_row_reducing(rows: i32) -> Launch {
    /// One float per warp of a [`BLOCK`]-wide block.
    const RMS_SMEM: u32 = (BLOCK / WARP) * 4;

    Launch::per_row(rows.unsigned_abs(), BLOCK).smem(RMS_SMEM)
}

/// Flat pointwise over `n` elements, [`BLOCK`] per block, rounded up.
#[must_use]
const fn elementwise(n: i32) -> Launch {
    Launch::flat(n.unsigned_abs(), BLOCK)
}

/// Flat pointwise over `n` elements given as a 64-bit count, saturating.
#[must_use]
fn elementwise_wide(n: i64) -> Launch {
    let blocks = (n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
    Launch::grid([u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1], [BLOCK, 1, 1])
}

/// One block per row, as wide as the row rounded up to a warp and capped at
/// `MAX_BLOCK`. A zero `width` is a caller bug: build a [`Region`] first.
#[must_use]
const fn route_rows(rows: i32, width: i32) -> Launch {
    /// The largest block CUDA will launch.
    const MAX_BLOCK: u32 = 1024;

    let warps = width.unsigned_abs().div_ceil(WARP);
    let warps = if warps == 0 { 1 } else { warps };
    let block = warps.saturating_mul(WARP);
    let block = if block > MAX_BLOCK { MAX_BLOCK } else { block };
    Launch::grid([rows.unsigned_abs(), 1, 1], [block, 1, 1])
}

/// One block per (row, head), [`BLOCK`] wide.
#[must_use]
const fn gated_rms(rows: i32, heads: i32) -> Launch {
    Launch::grid([rows.unsigned_abs(), heads.unsigned_abs(), 1], [BLOCK, 1, 1])
}

/// `dim3(T, K, ceil(H / 128))` at [`ALTUP_BLOCK`] threads.
#[must_use]
const fn altup_streams(rows: i32, streams: i32, hidden: i32) -> Launch {
    Launch::grid(
        [rows.unsigned_abs(), streams.unsigned_abs(), hidden.unsigned_abs().div_ceil(ALTUP_BLOCK)],
        [ALTUP_BLOCK, 1, 1],
    )
}

/// One block per row per head, [`BLOCK`] wide, nothing shared.
///
/// Takes a `&Region` rather than an `i32`, so the zero-width refusal is the caller's `all()`.
fn rows_per_head<P>(dst: &Region<P>, stated_head_dim: i32) -> Result<Launch, Refusal> {
    if stated_head_dim == 0 {
        return Ok(per_row(dst.rows));
    }
    let (w, hd) = (dst.width.unsigned_abs(), stated_head_dim.unsigned_abs());
    if !w.is_multiple_of(hd) {
        return Err(Refusal::Narrow {
            what: "a row that divides by head_dim",
            at: i64::from(dst.width),
        });
    }
    let blocks = dst.rows.unsigned_abs().checked_mul(w / hd).ok_or(Refusal::Narrow {
        what: "a row count that fits a grid",
        at: i64::from(dst.rows),
    })?;
    Ok(Launch::per_row(blocks, BLOCK))
}

/// `rmsnorm_vec8_ok`.
///
/// The two pitches are [`Stride`] and the extent is an `i32`: swapping them is a type error.
#[must_use]
fn vec8_ok(
    x: *const c_void,
    y: *const c_void,
    weight: *const c_void,
    hidden: i32,
    x_row_stride: Stride,
    y_row_stride: Stride,
) -> bool {
    hidden % 8 == 0
        && *x_row_stride % 8 == 0
        && *y_row_stride % 8 == 0
        && aligned16(x)
        && aligned16(y)
        && aligned16(weight)
}

/// How many heads a row of `row.width` elements holds, at `head_dim` each.
fn heads<P>(row: &Region<P>, head_dim: i32) -> Result<i32, Refusal> {
    let width = row.width;
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of heads",
            at: i64::from(width),
        });
    }
    Ok(width / head_dim)
}

/// How many hyper-connection streams a row of `row.width` elements holds, at
/// `hidden_size` each.
///
/// `hidden_size` stays a bare `i32`, not a view: its zero is `Refusal::Empty`, not `Absent`.
fn streams<P>(row: &Region<P>, hidden_size: i32) -> Result<i32, Refusal> {
    let width = row.width;
    if hidden_size <= 0 {
        return Err(Refusal::Empty { what: "the hidden width" });
    }
    if width % hidden_size != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of hyper-connection streams",
            at: i64::from(width),
        });
    }
    let hc_mult = width / hidden_size;
    hc_mult_ok(hc_mult)?;
    Ok(hc_mult)
}

/// The side of a square row: the exact integer square root of `row.width`,
/// refused when the row is not a square.
fn square_side<P>(row: &Region<P>) -> Result<i32, Refusal> {
    let width = row.width;
    // Checks `r - 1, r, r + 1`: `f64::sqrt` can be off by one for a large `i32` square.
    let r = f64::from(width).sqrt() as i32;
    let square = |c: &i32| *c > 0 && i64::from(*c) * i64::from(*c) == i64::from(width);
    let Some(side) = [r - 1, r, r + 1].into_iter().find(square) else {
        return Err(Refusal::Narrow {
            what: "the row is not a square number of coefficients",
            at: i64::from(width),
        });
    };
    Ok(side)
}

/// The other factor of an AltUp `[k * h]` row: `row.width / part`, refused
/// when the row is not a whole number of `part`s.
///
/// `part_what` names the divisor, not the row: the row's own refusal is the caller's `all()`.
fn altup_factor<P>(row: &Region<P>, part: i32, part_what: &'static str) -> Result<i32, Refusal> {
    let width = row.width;
    if part <= 0 {
        return Err(Refusal::Empty { what: part_what });
    }
    if width % part != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of AltUp streams",
            at: i64::from(width),
        });
    }
    Ok(width / part)
}

/// `norm::rmsnorm_strided_bf16`, both arms.
///
/// # Safety
///
/// `x`, `weight` and `y` must address live device memory of the extents the strides describe.
#[routine]
pub fn rmsnorm_strided_bf16(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled it `Env<keys::RmsEps>` and no builder
    // ever began stating it. A `Const` mark PROMISES the statement carries the
    // number at its slot in the params run; where nothing states one the
    // promise breaks at the fire, not at the type. §11.20.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;
    rmsnorm_strided_bf16_at(ctx, x, weight, y, eps)
}

/// [`rmsnorm_strided_bf16`] with the epsilon SUPPLIED.
///
/// `tower::gemma4_vision` calls this off a bare [`Ctx::on`], which answers no
/// facts — it is a host program over its own scratch, not a launch the engine
/// planned — and the tower's epsilon is the tower's, not the text model's.
/// The traced path above asks; see [`rmsnorm_no_scale_at`] for the same split
/// and the same reason.
///
/// # Errors
///
/// Whatever the launch refuses.
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s, unchanged.
pub fn rmsnorm_strided_bf16_at(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    eps: f32,
) -> Result<(), Refusal> {
    // `vec8_ok` accepts a zero pitch (`0 % 8 == 0`); `y`'s width is checked first.
    let dst = y.all("the normalised row's width")?;
    let src = x.all("the source row's pitch")?;
    // A stride is not a width: each pitch comes off its own region, not `dst.width`.
    let vec_ok = vec8_ok(
        x.ptr.cast(),
        y.ptr.cast_const().cast(),
        weight.v.cast(),
        dst.width,
        src.stride,
        dst.stride,
    );
    if vec_ok {
        // SAFETY: every pointer is live for the extent the kernel reads it as.
        return ctx.fire(Fire::at("norm/rmsnorm.cuh", "::pie::norm::rmsnorm_vec8<::pie::i32(512), false, false>").apply(Launch::per_row(dst.rows.unsigned_abs(), VBLOCK)), &[
                    x.arg(),
                    weight.arg(),
                    y.arg(),
                    None::<NonNull<f16>>.arg(),
                    dst.width.arg(),
                    src.stride.arg(),
                    dst.stride.arg(),
                    eps.arg(),
                ]);
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", "::pie::norm::rmsnorm<::pie::bf16, 256>").apply(per_row(dst.rows)), &[
                x.arg(),
                weight.arg(),
                y.arg(),
                dst.width.arg(),
                src.stride.arg(),
                dst.stride.arg(),
                eps.arg(),
            ])
}

/// `norm::rmsnorm_bf16` — one call, nothing else.
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s, unchanged.
#[routine]
pub fn unstrided_bf16(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    // Reads `x` through `y`'s rectangle: both widths here are `y.width`.
    let packed = In { ptr: x.ptr, rows: y.rows, width: y.width };
    rmsnorm_strided_bf16(ctx, packed, weight, y)
}
/// `norm::rmsnorm_bf16_with_fp16`, both arms: the fp16-writing twin of
/// [`unstrided_bf16`], never reachable with a null `y_fp16`.
///
/// # Errors
///
/// [`rmsnorm_strided_bf16`]'s, plus [`Refusal::Wide`] on an fp16 count past `i32::MAX`.
///
/// # Safety
///
/// `x`, `weight`, `y` and `y_fp16` must address `y.rows * y.width` live
/// elements — bf16 for the first three, fp16 for the last.
#[routine]
pub fn rmsnorm_bf16_with_fp16(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    y_fp16: Out<Tensor<f16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let dst = y.all("the normalised row's width")?;
    if !vec8_ok(
        x.ptr.cast(),
        y.ptr.cast_const().cast(),
        weight.v.cast(),
        dst.width,
        dst.stride,
        dst.stride,
    ) {
        // Not `elements()`: that saturates in `i32` and would hide this overflow.
        let n = i64::from(dst.rows) * i64::from(dst.width);
        // `bf16_to_fp16` sizes its launch from a 32-bit count; the ceiling is refused here.
        if n > i64::from(i32::MAX) {
            return Err(Refusal::Wide {
                what: "the fp16 copy's element count, which the cast sizes a \
                       32-bit launch extent from",
                at: n,
                max: i64::from(i32::MAX),
            });
        }
        unstrided_bf16(ctx, x, weight, y)?;
        return crate::quant::bf16_to_fp16(
            ctx,
            In { ptr: y.ptr.cast_const(), rows: y.rows, width: y.width },
            kernels::routine::Out { ptr: y_fp16.ptr, rows: dst.rows, width: dst.width },
        );
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", "::pie::norm::rmsnorm_vec8<::pie::i32(512), false, true>").apply(Launch::per_row(dst.rows.unsigned_abs(), VBLOCK)), &[
                x.arg(),
                weight.arg(),
                y.arg(),
                y_fp16.arg(),
                dst.width.arg(),
                dst.stride.arg(),
                dst.stride.arg(),
                eps.arg(),
            ])
}

/// The `OpKind::Rmsnorm` launcher — `norm::rmsnorm_bf16`.
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s.
#[routine(bf16)]
pub fn rmsnorm<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let per_head_dim = ctx.ask::<i32, keys::PerHeadDim>()?;
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // The one place `dst.stride` would be wrong: pitch between rows here is `hidden`.
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm<{}, 256>", T::CPP))).apply(launch), &[
                x.arg(),
                weight.arg(),
                y.arg(),
                hidden.arg(),
                hidden.arg(),
                hidden.arg(),
                eps.arg(),
            ])
}

/// gemma's `(1 + w)` fold — `norm::rmsnorm_gemma_bf16`.
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s.
#[routine(bf16)]
pub fn rmsnorm_gemma<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    // Named, not banked: gemma's per-head callers use `OpKind::RmsnormPerHead`, no `Arg::Weight`.
    weight: Const<Tensor<T>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let per_head_dim = ctx.ask::<i32, keys::PerHeadDim>()?;
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm_gemma<{}, 256>", T::CPP))).apply(launch), &[
                x.arg(),
                weight.arg(),
                y.arg(),
                hidden.arg(),
                hidden.arg(),
                hidden.arg(),
                eps.arg(),
            ])
}

/// The weightless per-head norm — `norm::rmsnorm_no_scale_bf16`.
///
/// # Safety
///
/// `x` and `y` must address `y.rows * y.width` live bf16 elements.
#[routine(bf16)]
pub fn rmsnorm_no_scale<T>(
    ctx: &Ctx<'_>,
    // A MAY-ALIAS, like [`crate::mlp::geglu_tanh`]'s gate: the row said
    // `in_place = &[(0, 0)]` and `tower::gemma4_vision` calls this directly
    // with a source and a destination that are different buffers, so the
    // kernel does not require the alias and `InOut` would misstate it.
    x: In<Tensor<T>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: both were `Env<keys::_>` before the four marks and
    // no builder ever began stating them. A `Const` mark PROMISES the
    // statement carries the number at its slot in the params run; where
    // nothing states one the promise is broken at the fire, not at the type.
    // See `.wiki/migration.md` §11.20.
    let per_head_dim = ctx.ask::<i32, keys::PerHeadDim>()?;
    let eps = ctx.ask::<f32, keys::RmsEps>()?;
    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm_no_scale<{}, 256>", T::CPP))).apply(launch), &[x.arg(), y.arg(), hidden.arg(), eps.arg()])
}

/// [`rmsnorm_no_scale`] over bf16 with the two numbers SUPPLIED.
///
/// `tower::gemma4_vision` calls the norm directly off a bare [`Ctx::on`],
/// which answers no facts at all — it is a host program over its own scratch,
/// not a launch the engine planned — so an `ask` there would refuse and the
/// tower's epsilon is the TOWER's anyway, not the text model's. The traced
/// path above asks, because there the number is the deployment's and the
/// driver holds it. One kernel, two callers, and only one of them has a
/// context that can answer.
///
/// Not generic, and not a `#[routine]`: no statement names it, the tower is
/// bf16 throughout, and a second registered row for the same symbol would be
/// a second contract for one kernel.
///
/// # Errors
///
/// [`Refusal::Empty`] on an empty grid; whatever the launch refuses.
///
/// # Safety
///
/// [`rmsnorm_no_scale`]'s, unchanged.
pub fn rmsnorm_no_scale_at(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    per_head_dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol("::pie::norm::rmsnorm_no_scale<::pie::bf16, 256>")).apply(launch), &[x.arg(), y.arg(), hidden.arg(), eps.arg()])
}

/// The gated norm — `norm::rmsnorm_gated_bf16`.
///
/// # Safety
///
/// `x`, `gate`, `y` address `y.rows * y.width` live bf16; `weight` `hidden` live floats.
#[routine(bf16)]
pub fn rmsnorm_gated<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    gate: In<Tensor<T>>,
    // Banked, not named: reached only via a launch stating the weight as `Arg::Weight`.
    weight: Const<Tensor<f32>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let per_head_dim = ctx.ask::<i32, keys::PerHeadDim>()?;
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm_gated<{}, 256>", T::CPP))).apply(launch), &[x.arg(), gate.arg(), weight.arg(), y.arg(), hidden.arg(), eps.arg()])
}

/// The gated norm with an fp32 input — `norm::rmsnorm_gated_fp32_in_bf16`.
///
/// # Safety
///
/// `x` must address `y.rows * y.width` live floats, `gate` and `y` the same
/// count of bf16, `weight` `hidden` live floats.
#[routine(bf16)]
pub fn rmsnorm_gated_fp32_in<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<f32>>,
    gate: In<Tensor<T>>,
    // Named, not banked: Env<this op's builder pushes `vec![x, gate]`, keys::Unstated>, no weight operand.
    weight: Const<Tensor<f32>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: `OpKind::RmsnormGated` carries a weight name and
    // nothing else, and only `OpKind::Launch` gets a params run out of
    // `lower::walk` — so a `Const<f32>` here promised an epsilon the
    // statement has nowhere to put, and every qwen3.5 gated norm refused.
    // HEAD spelled it `Env<keys::RmsEps>`.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;
    // THE GATED-DELTA-NET HEAD WIDTH, which is what this row's `Bound` entry
    // said was missing: *"needs `Facts::per_head_dim()`, not a new `Source`"*.
    // It needs neither. `per_head_dim` was `#[unbound]` and read as zero,
    // which this body takes as "the whole row" -- so the one family that
    // states this op would have normalised across every head at once.
    //
    // `keys::GdnVDim` is the answer and the driver already holds it: qwen3.5
    // is the only text that emits `OpKind::RmsnormGated` (nemotron's gated
    // norm is `ssm::zamba_rmsnorm_gated_bf16`, a stated launch of its own),
    // and its core output is `[N, V_h, V_d]` — one head is `v_d` wide.
    let per_head_dim = ctx.ask::<i32, keys::GdnVDim>()?;
    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm_gated_f32_in<{}, 256>", T::CPP))).apply(launch), &[x.arg(), gate.arg(), weight.arg(), y.arg(), hidden.arg(), eps.arg()])
}

/// The residual add and the next block's pre-norm, fused.
///
/// # Safety
///
/// `hidden`, `residual`, `norm_out`: `norm_out.rows * norm_out.width` bf16; `weight`: that width.
#[routine(bf16)]
pub fn residual_add_rmsnorm<
    // MORE THAN THE PLANE'S MINIMUM, and `hidden` is why: the launcher takes
    // its first operand as `T*` while the mark hands a body `T::Read`, so the
    // body needs to know that reading carrier IS `*const T` before it can
    // spell the mutable one. `crate::RoutineElem` alone leaves it opaque.
    T: crate::RoutineElem + kernels::routine::Elem<Read = *const T, Write = *mut T>,
>(
    ctx: &Ctx<'_>,
    // AN INPUT THE LAUNCH WRITES THROUGH, AND NOT A RESULT. The statement
    // places it as operand 0 and declares ONE result (`norm_out`), so `InOut`
    // -- which claims a result slot as well -- would push `norm_out` to
    // `Out(1)`, a slot no statement fills.
    //
    // The mutation is a fact about the KERNEL, not about the statement, so it
    // is said where the kernel is called: `hidden.ptr.cast_mut()` below. That
    // is the whole of what `In<0, *mut T>` used to spell.
    hidden: In<Tensor<T>>,
    residual: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    norm_out: Out<Tensor<T>>,
    eps: Const<f32>) -> Result<(), Refusal> {
    // A zero width reduces over nothing, so this must be a view.
    let dst = norm_out.all("the normalised row's width")?;
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::residual_add_rmsnorm<{}, 256>", T::CPP))).apply(per_row(dst.rows)), &[
                hidden.ptr.cast_mut().arg(),
                residual.arg(),
                weight.arg(),
                norm_out.arg(),
                dst.width.arg(),
                eps.arg(),
            ])
}

/// Norm, then add into the residual stream in place —
///
/// # Safety
///
/// [`residual_add_rmsnorm`]'s.
#[routine(bf16)]
pub fn rmsnorm_residual_add<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    // Banked: `alias()` leaves a `Weight` slot alone; a bare pointer would shift to `In(2)`.
    weight: Const<Tensor<T>>,
    hidden: InOut<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let dst = hidden.all("the normalised row's width")?;
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm_residual_add<{}, 256>", T::CPP))).apply(per_row(dst.rows)), &[x.arg(), weight.arg(), hidden.arg(), dst.width.arg(), eps.arg()])
}

/// `norm::rmsnorm_residual_add_scale_rmsnorm_bf16`: all three arms differ only in
/// instantiation -- vectorised above/below [`RASR_VEC512_ABOVE`], or scalar if unaligned.
///
/// # Safety
///
/// `x`, `weight`, `hidden`, `next_weight`, `norm_out` address live bf16 elements:
/// `hidden.rows * hidden.width` (the two weights, `hidden.width`).
#[routine]
pub fn rmsnorm_residual_add_scale_rmsnorm_bf16(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    // `Weight<0, *const _>`/`Weight<1, *const _>`: stating `x` sets the input counter without consuming a slot.
    weight: Const<Tensor<bf16>>,
    hidden: InOut<Tensor<bf16>>,
    // THE STATEMENT CARRIES IT NOW. It was `#[unbound]` -- a parameter
    // nothing supplies -- and that one entry is what kept this row out of the
    // binder, so gemma-4 traced, lowered and refused here at every layer
    // boundary. There is no fact to ask for either: `Cx::named_scale` reads
    // `DispatchCtx::scales`, and no deployment publishes a `layer_scale`.
    //
    // The number belongs to the CALLER, and gemma-4's own text says which:
    // its last layer lands unfused through `norm::rmsnorm_residual_add_bf16`,
    // which applies no scale at all, so the fused path must apply the
    // identity or the two branches would disagree about the same landing.
    scale: Const<f32>,
    next_weight: Const<Tensor<bf16>>,
    norm_out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    // A zero would pick the vectorised path and normalise nothing (`0 % 8 == 0`).
    let dst = hidden.all("the normalised row's width")?;
    let rows = dst.rows.unsigned_abs();
    let hidden_size = dst.width;
    let vec_ok = hidden_size % 8 == 0
        && aligned16(x.ptr.cast())
        && aligned16(hidden.ptr.cast_const().cast())
        && aligned16(norm_out.ptr.cast_const().cast())
        && aligned16(weight.v.cast())
        && aligned16(next_weight.v.cast());
    let (instantiation, block) = if vec_ok {
        if hidden_size >= RASR_VEC512_ABOVE {
            ("::pie::norm::rmsnorm_rasr_vec8<::pie::i32(512)>", VBLOCK)
        } else {
            ("::pie::norm::rmsnorm_rasr_vec8<::pie::i32(256)>", BLOCK)
        }
    } else {
        ("::pie::norm::rmsnorm_residual_add_scale_rmsnorm<::pie::bf16, 512>", VBLOCK)
    };
    ctx.fire(Fire::at("norm/rmsnorm.cuh", instantiation).apply(Launch::per_row(rows, block)), &[
                x.arg(),
                weight.arg(),
                hidden.arg(),
                scale.arg(),
                next_weight.arg(),
                norm_out.arg(),
                hidden_size.arg(),
                eps.arg(),
            ])
}

/// `out[row][i] += bias[i]` — `norm::add_bias_bf16`.
///
/// # Safety
///
/// `out` must address `out.rows * out.width` live bf16 elements, `bias` `out.width` of them.
#[routine(bf16)]
pub fn add_bias<T>(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<T>>,
    bias: Const<Tensor<T>>) -> Result<(), Refusal> {
    // A zero here reads no bias yet succeeds (`route_rows` clamps up rather than refusing).
    let dst = out.all("the biased row's width")?;
    ctx.fire(Fire::at("norm/add_bias.cuh", crate::jit::symbol(&format!("::pie::norm::add_bias<{}>", T::CPP))).apply(route_rows(dst.rows, dst.width)), &[out.arg(), bias.arg(), dst.width.arg()])
}

/// gemma-3n's altup predict — `norm::altup_predict_bf16`.
///
/// # Safety
///
/// `streams`/`predictions`: `k * predictions.rows * h` bf16; `coefs`: `predictions.rows * k * k`.
#[routine(bf16)]
pub fn altup_predict<T>(
    ctx: &Ctx<'_>,
    streams: In<Tensor<T>>,
    coefs: In<Tensor<f32>>,
    predictions: Out<Tensor<T>>,
    // `k`/`h` computed here, not sourced: `k` is `coefs`'s root, `h` is `streams.width / k`.


) -> Result<(), Refusal> {
    let coef_row = coefs.all("the predict coefficients' row")?;
    let k = square_side(&coef_row)?;
    let stream_row = streams.all("the AltUp stream row's width")?;
    let h = altup_factor(&stream_row, k, "the AltUp stream count")?;
    ctx.fire(Fire::at("norm/altup.cuh", crate::jit::symbol(&format!("::pie::norm::altup_predict<{}>", T::CPP))).apply(altup_streams(predictions.rows, k, h)), &[
                streams.arg(),
                coefs.arg(),
                predictions.arg(),
                k.arg(),
                predictions.rows.arg(),
                h.arg(),
            ])
}

/// gemma-3n's altup correct — `norm::altup_correct_bf16`.
///
/// # Safety
///
/// [`altup_predict`]'s, with `activated` addressing `corrected.rows *
/// activated.width` and `corrected` `k` times that, live bf16 elements.
#[routine(bf16)]
pub fn altup_correct<T>(
    ctx: &Ctx<'_>,
    predictions: In<Tensor<T>>,
    activated: In<Tensor<T>>,
    correction_coefs_plus_one: In<Tensor<f32>>,
    corrected: Out<Tensor<T>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    active_idx: i32) -> Result<(), Refusal> {
    // Both extents are separate views: a zero `k` or `h` would otherwise still launch.
    let coef_row = correction_coefs_plus_one.all("the correction coefficients' width")?;
    let act_row = activated.all("the activated stream's width")?;
    let (k, h) = (coef_row.width, act_row.width);
    ctx.fire(Fire::at("norm/altup.cuh", crate::jit::symbol(&format!("::pie::norm::altup_correct<{}>", T::CPP))).apply(altup_streams(corrected.rows, k, h)), &[
                predictions.arg(),
                activated.arg(),
                correction_coefs_plus_one.arg(),
                corrected.arg(),
                k.arg(),
                corrected.rows.arg(),
                h.arg(),
                active_idx.arg(),
            ])
}

/// The per-row RMS of the reference stream — `norm::compute_rms_bf16`.
///
/// # Safety
///
/// `reference` addresses `reference.rows * reference.width` bf16; `out` `reference.rows` floats.
#[routine(bf16)]
pub fn compute_rms<T>(
    ctx: &Ctx<'_>,
    // Off `reference`, not `out`: the result is one float per row (`out.width` is always 1).
    reference: In<Tensor<T>>,
    out: Out<Tensor<f32>>,
    // `ALTUP_EPS`, not `Env<keys::RmsEps>`: the algorithm's own constant, not the deployment's.


) -> Result<(), Refusal> {
    let src = reference.all("the reduced row's width")?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", crate::jit::symbol(&format!("::pie::norm::compute_rms<{}>", T::CPP))).apply(per_row_reducing(src.rows)), &[reference.arg(), out.arg(), src.width.arg(), ALTUP_EPS.arg()])
}

/// Rescale each row to a stated RMS, in place — `norm::magnitude_rescale_bf16`.
///
/// # Safety
///
/// `x` must address `x.rows * x.width` live bf16 elements, `target_rms` `x.rows` live floats.
#[routine(bf16)]
pub fn magnitude_rescale<T>(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<T>>,
    target_rms: In<Tensor<f32>>) -> Result<(), Refusal> {
    // A zero rescales no element, so the rectangle arrives as a view.
    let dst = x.all("the rescaled row's width")?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", crate::jit::symbol(&format!("::pie::norm::magnitude_rescale<{}>", T::CPP))).apply(per_row_reducing(dst.rows)), &[x.arg(), target_rms.arg(), dst.width.arg(), ALTUP_EPS.arg()])
}

/// The mean over altup's `k` streams — `norm::mean_streams_bf16`.
///
/// # Safety
///
/// `streams` addresses `k * out.rows * out.width` bf16; `out` `out.rows * out.width` of them.
#[routine(bf16)]
pub fn mean_streams<T>(
    ctx: &Ctx<'_>,
    streams: In<Tensor<T>>,
    out: Out<Tensor<T>>,
    // A quotient of the two widths this statement carries: `streams.width / out.width`.



) -> Result<(), Refusal> {
    /// Pointwise with the row on its own grid axis.
    #[must_use]
    const fn elementwise_rows(rows: i32, width: i32) -> Launch {
        Launch::grid([rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1], [BLOCK, 1, 1])
    }

    // A zero here averages nothing, so this must be a view, not a read.
    let dst = out.all("the averaged row's width")?;

    let src = streams.all("the AltUp stream row's width")?;
    let k = altup_factor(&src, dst.width, "the averaged row's width")?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", crate::jit::symbol(&format!("::pie::norm::mean_streams<{}>", T::CPP))).apply(elementwise_rows(dst.rows, dst.width)), &[streams.arg(), out.arg(), k.arg(), dst.rows.arg(), dst.width.arg()])
}

/// bf16 predict coefficients widened to `float` —
///
/// # Safety
///
/// `in_bf16` must address `out.rows * k * k` live bf16 elements, `out` the same count of floats.
#[routine]
pub fn altup_unpack_predict_coefs(
    ctx: &Ctx<'_>,
    in_bf16: In<Tensor<bf16>>,
    out: Out<Tensor<f32>>,
    // `k` is the exact square root of `in_bf16.width`: coefficients are `[t, k * k]`.



) -> Result<(), Refusal> {
    let packed = in_bf16.all("the packed coefficients' width")?;
    let k = square_side(&packed)?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", "::pie::norm::unpack_predict_coefs<::pie::bf16>").apply(route_rows(out.rows, k.saturating_mul(k))), &[in_bf16.arg(), out.arg(), k.arg()])
}

/// bf16 correct coefficients widened to `float` —
///
/// # Safety
///
/// `in_bf16` must address `out.rows * k` live bf16 elements, `out` the same count of floats.
#[routine]
pub fn altup_unpack_correct_coefs(
    ctx: &Ctx<'_>,
    in_bf16: In<Tensor<bf16>>,
    out: Out<Tensor<f32>>,
    // Unlike the predict twin, no square root: `k` is `packed.width` directly.


) -> Result<(), Refusal> {
    // A zero here unpacks nothing, so the count must come off a view.
    let packed = in_bf16.all("the packed coefficients' width")?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", "::pie::norm::unpack_correct_coefs<::pie::bf16>").apply(route_rows(out.rows, packed.width)), &[in_bf16.arg(), out.arg(), packed.width.arg()])
}

/// `tanh` in place over a bf16 slab — `norm::tanh_bf16`.
///
/// # Safety
///
/// `x` must address `x.rows * x.width` live bf16 elements.
#[routine(bf16)]
pub fn tanh<T>(ctx: &Ctx<'_>, x: InOut<Tensor<T>>) -> Result<(), Refusal> {
    // Guards the product through the width: a missing width looks like an ordinary zero.
    let rect = x.all("the rectangle's row width")?;
    let n = rect.elements();
    ctx.fire(Fire::at("norm/altup_aux.cuh", crate::jit::symbol(&format!("::pie::norm::tanh_inplace<{}>", T::CPP))).apply(elementwise(n)), &[x.arg(), n.arg()])
}

/// [`tanh`] over fp16 — `norm::tanh_f16`.
///
/// # Safety
///
/// [`tanh`]'s, with `x` addressing fp16.
#[routine]
pub fn tanh_f16(ctx: &Ctx<'_>, x: InOut<Tensor<f16>>) -> Result<(), Refusal> {
    let rect = x.all("the rectangle's row width")?;
    let n = rect.elements();
    ctx.fire(Fire::at("norm/altup_aux.cuh", "::pie::norm::tanh_inplace<\
                                              ::pie::f16>").apply(elementwise(n)), &[x.arg(), n.arg()])
}

/// `y += x` — `norm::residual_add_bf16`.
///
/// # Safety
///
/// `y` and `x` must address `y.rows * y.width` live bf16 elements.
// `x` is `In<1, *const _>`, not `In<0, *const _>`: one pointer (`y`) serves the in-place buffer.
// `usize` because the CUDA signature takes `usize n`.
#[routine(bf16)]
pub fn residual_add<T>(ctx: &Ctx<'_>, y: InOut<Tensor<T>>, x: In<Tensor<T>>) -> Result<(), Refusal> {
    let rect = y.all("the rectangle's row width")?;
    let Ok(n) = usize::try_from(rect.elements()) else {
        return Err(Refusal::Empty { what: "the residual rectangle's element count" });
    };
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    ctx.fire(Fire::at("norm/elementwise.cuh", crate::jit::symbol(&format!("::pie::norm::residual_add<{}>", T::CPP))).apply(launch), &[y.arg(), x.arg(), n.arg()])
}

/// [`residual_add`] over fp16 — `norm::residual_add_f16`.
///
/// # Safety
///
/// [`residual_add`]'s, with both pointers addressing fp16.
#[routine]
pub fn residual_add_f16(
    ctx: &Ctx<'_>,
    y: InOut<Tensor<f16>>,
    x: In<Tensor<f16>>) -> Result<(), Refusal> {
    let rect = y.all("the rectangle's row width")?;
    let Ok(n) = usize::try_from(rect.elements()) else {
        return Err(Refusal::Empty { what: "the residual rectangle's element count" });
    };
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    ctx.fire(Fire::at("norm/elementwise.cuh", "::pie::norm::residual_add<::pie::f16>").apply(launch), &[y.arg(), x.arg(), n.arg()])
}

/// `x *= s` — `norm::scalar_mul_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements.
#[routine(bf16)]
pub fn scalar_mul<T>(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<T>>,
    // The scale rides the param channel, not an extent: eight sites pass five distinct values.
    s: Const<f32>) -> Result<(), Refusal> {
    let rect = x.all("the rectangle's row width")?;
    let Ok(n) = usize::try_from(rect.elements()) else {
        return Err(Refusal::Empty { what: "the scaled rectangle's element count" });
    };
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    ctx.fire(Fire::at("norm/elementwise.cuh", crate::jit::symbol(&format!("::pie::norm::scalar_mul<{}>", T::CPP))).apply(launch), &[x.arg(), s.arg(), n.arg()])
}

/// `norm::hc_pre_postprocess_bf16`.
///
/// # Safety
///
/// `residual`/`layer_input`: `n * hc_mult * hidden_size`/`n * hidden_size` bf16;
/// `mixes`/`scale`/`base` the layer's slabs; `post_mix`/`comb_mix` scratch floats.
#[routine(bf16)]
pub fn hc_pre_postprocess<T>(
    ctx: &Ctx<'_>,
    mixes: In<Tensor<f32>>,
    // No `Cx` query exists for these float slabs; `mixes` became an operand, these two did not.
    scale: *const f32,
    base: *const f32,
    residual: In<Tensor<T>>,
    post_mix: Out<Tensor<f32>>,
    comb_mix: Out<Tensor<f32>>,
    layer_input: Out<Tensor<T>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    n: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    hc_mult: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    hidden_size: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    hc_eps: f32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    hc_post_alpha: f32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    sinkhorn_iters: i32) -> Result<(), Refusal> {
    hc_mult_ok(hc_mult)?;
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", crate::jit::symbol(&format!("::pie::norm::hc_pre_postprocess<{}, 256>", T::CPP))).apply(per_row(n)), &[
                mixes.arg(),
                scale.arg(),
                base.arg(),
                residual.arg(),
                post_mix.arg(),
                comb_mix.arg(),
                layer_input.arg(),
                hc_mult.arg(),
                hidden_size.arg(),
                hc_eps.arg(),
                hc_post_alpha.arg(),
                sinkhorn_iters.arg(),
            ])
}

/// `norm::hc_post_bf16`.
///
/// `hc_mult` is `out_width / hidden_size`, undivided in the statement; [`streams`] divides it here.
///
/// # Errors
///
/// `Out::all`'s `Absent`, [`streams`]'s `Empty`/`Narrow`, then [`hc_mult_ok`]'s `Wide`.
///
/// # Safety
///
/// [`hc_pre_postprocess`]'s; `out_residual`: `out_residual.rows * out_residual.width` bf16.
#[routine(bf16)]
pub fn hc_post<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    residual: In<Tensor<T>>,
    post_mix: In<Tensor<f32>>,
    comb_mix: In<Tensor<f32>>,
    out_residual: Out<Tensor<T>>) -> Result<(), Refusal> {
    // Only the numerator is a view; `x.width`'s zero is `streams`'s own `Empty`.
    let dst = out_residual.all("the hyper-connection row's width")?;
    let hc_mult = streams(&dst, x.width)?;
    let total = i64::from(dst.rows) * i64::from(x.width);
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", crate::jit::symbol(&format!("::pie::norm::hc_post<{}>", T::CPP))).apply(elementwise_wide(total)), &[
                x.arg(),
                residual.arg(),
                post_mix.arg(),
                comb_mix.arg(),
                out_residual.arg(),
                dst.rows.arg(),
                hc_mult.arg(),
                x.width.arg(),
            ])
}

/// `norm::hc_head_postprocess_bf16`.
///
/// # Safety
///
/// [`hc_pre_postprocess`]'s, with `out` addressing `n * hidden_size` live bf16 elements.
#[routine(bf16)]
pub fn hc_head_postprocess<T>(
    ctx: &Ctx<'_>,
    mixes: In<Tensor<f32>>,
    scale: *const f32,
    base: *const f32,
    residual: In<Tensor<T>>,
    out: Out<Tensor<T>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    n: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    hc_mult: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    hidden_size: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    hc_eps: f32) -> Result<(), Refusal> {
    hc_mult_ok(hc_mult)?;
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", crate::jit::symbol(&format!("::pie::norm::hc_head_postprocess<{}, 256>", T::CPP))).apply(per_row(n)), &[
                mixes.arg(),
                scale.arg(),
                base.arg(),
                residual.arg(),
                out.arg(),
                hc_mult.arg(),
                hidden_size.arg(),
                hc_eps.arg(),
            ])
}

/// `[n, hidden] -> [n, hc_mult, hidden]` — `norm::hc_expand_bf16`.
///
/// `hc_mult` here is the ratio of the statement's two widths, taken where used.
///
/// # Errors
///
/// `Out::all`'s and [`streams`]'s, per [`hc_post`].
///
/// # Safety
///
/// `input` addresses `input.rows * input.width` bf16; `output` `output.rows * output.width`.
#[routine(bf16)]
pub fn hc_expand<T>(
    ctx: &Ctx<'_>,
    input: In<Tensor<T>>,
    output: Out<Tensor<T>>) -> Result<(), Refusal> {
    let dst = output.all("the hyper-connection row's width")?;
    let hc_mult = streams(&dst, input.width)?;
    let total = i64::from(input.rows) * i64::from(input.width);
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", crate::jit::symbol(&format!("::pie::norm::hc_expand<{}>", T::CPP))).apply(elementwise_wide(total)), &[
                input.arg(),
                output.arg(),
                input.rows.arg(),
                hc_mult.arg(),
                input.width.arg(),
            ])
}

/// `norm::hc_rmsnorm_to_f32`.
///
/// # Safety
///
/// `input` addresses `output.rows * output.width` bf16; `output` the same count of floats.
#[routine]
pub fn hc_rmsnorm_to_f32(
    ctx: &Ctx<'_>,
    input: In<Tensor<bf16>>,
    output: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    // The one hyper-connection launcher that doesn't divide, so `streams`'s guard doesn't apply.
    let dst = output.all("the normalised row's width")?;
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", "::pie::norm::hc_rmsnorm_to_f32<::pie::bf16, 256>").apply(per_row(dst.rows)), &[input.arg(), output.arg(), dst.width.arg(), eps.arg()])
}

/// The attention sink's log-sum-exp correction — `norm::attn_sink_correction_bf16`.
///
/// # Errors
///
/// `Out::all`'s `Absent`, then [`heads`]'s `Empty` (bad `head_dim`) or `Narrow`
/// (width not a multiple of heads).
///
/// # Safety
///
/// `out` addresses `out.rows * out.width` live bf16 elements; `lse`/`sink`
/// `out.rows * (out.width / head_dim)`/`out.width / head_dim` live floats.
#[routine(bf16)]
pub fn attn_sink_correction<T>(
    ctx: &Ctx<'_>,
    // [`residual_add`]'s and [`magnitude_rescale`]'s shape again: `lse` is input one, not zero.
    out: InOut<Tensor<T>>,
    lse: In<Tensor<f32>>,
    // Banked: a weight slot consumes no input counter, so `lse` stays at `In(1)`.
    sink: Const<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;

    let dst = out.all("the row whose heads are counted")?;
    let num_heads = heads(&dst, head_dim)?;
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", crate::jit::symbol(&format!("::pie::norm::attn_sink_correction<{}>", T::CPP))).apply(gated_rms(dst.rows, num_heads)), &[out.arg(), lse.arg(), sink.arg(), num_heads.arg(), head_dim.arg()])
}

/// QK-norm in place over a packed head axis — `norm::per_head_rmsnorm_bf16`.
///
/// # Errors
///
/// `Out::all`'s and [`heads`]'s, per [`attn_sink_correction`].
///
/// # Safety
///
/// `q` must address `q.rows * q.width` live bf16 elements.
#[routine(bf16)]
pub fn per_head_rmsnorm<T>(
    ctx: &Ctx<'_>,
    // `Out<0, *mut _>`, no `In<_, _>`: one pointer serves both read and write.
    q: InOut<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let dst = q.all("the row whose heads are counted")?;
    let num_heads = heads(&dst, head_dim)?;
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", crate::jit::symbol(&format!("::pie::norm::per_head_rmsnorm<{}>", T::CPP))).apply(gated_rms(dst.rows, num_heads)), &[q.arg(), head_dim.arg(), eps.arg()])
}

/// `hc_mult <= MAX_HC_MULT`, as a refusal.
fn hc_mult_ok(hc_mult: i32) -> Result<(), Refusal> {
    if hc_mult > MAX_HC_MULT {
        return Err(Refusal::Wide {
            what: "hc_mult, which `hc_post` unrolls into a register array",
            at: i64::from(hc_mult),
            max: i64::from(MAX_HC_MULT),
        });
    }
    Ok(())
}


// `scalar_mul`'s derivation, checked by `cargo check` rather than a test.
//
// `d[0]` IS AN ALIAS AND WAS AN `Out`. The scale is applied in place -- one
// address the statement placed AND declared -- which the old signature spelled
// as an `Out` with an `in_place` pair stated on the row forty lines away. The
// `InOut` mark says it at the parameter, and `Source::Alias(0, 0)` is that
// pair, derived instead of written.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(scalar_mul::<bf16>);
    assert!(d.len() == 2);
    assert!(matches!(d[0], Some(kernels::Source::Alias(0, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
};

// The pitch equals the width (`Layout::packed`); `elements()` is `rows.saturating_mul(width)`.
const _: () = {
    let l = kernels::Layout::packed(7, 4096);
    assert!(l.row_pitch().0 == l.row_width());
    let r = Region {
        ptr: core::ptr::null::<bf16>(),
        rows: 7,
        width: l.row_width(),
        stride: l.row_pitch(),
    };
    assert!(r.stride.0 == r.width);
    assert!(r.elements() == 7i32.saturating_mul(4096));
    // Binds as the `i32` it wraps: `#[repr(transparent)]` makes that a fact, not a hope.
    assert!(core::mem::size_of::<Stride>() == core::mem::size_of::<i32>());
};

// THE HIDDEN STATE IS ONE ADDRESS IN BOTH RUNS, and now the signature says so.
//
// It used to be `In<0, *mut T>` with a comment explaining that a `*mut` at an
// input slot is an operand written through and not a result the statement
// forgot -- direction, not position. `InOut` is that sentence as a mark, and
// `Source::Alias(0, 0)` is what it derives: input slot 0 and result slot 0,
// one buffer. The residual that follows is `In(1)`, and nobody wrote the 1.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(residual_add_rmsnorm::<bf16>);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
};

// Pins [`unstrided_bf16`]'s derivation against a rename to `rmsnorm_bf16` (a different signature).
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(unstrided_bf16);
    assert!(d.len() == 3);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // THE WEIGHT IS A CHAIN, not a bare slot. `Const<Tensor<E>>` inherits what
    // `Weight` derived: the named bank first and the positional one after,
    // because an `OpKind::Launch` places a weight in the operand list where it
    // is positional while a semantic op carries only a NAME.
    assert!(matches!(
        d[1],
        Some(kernels::Source::Or(
            kernels::Source::Named(_),
            kernels::Source::Slot(kernels::Kind::Weight, 0)
        ))
    ));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // THE EPSILON IS THE STATEMENT'S NOW. It was `Source::Named("rms_eps")` --
    // a fact a driver arm answered -- and §3.6 puts it among the checkpoint's
    // constants, so `Const<f32>` carries it in the params run instead.
    assert!(!<unstrided_bf16 as ::kernels::Derivation>::DERIVED[0].nullable && !<unstrided_bf16 as ::kernels::Derivation>::DERIVED[1].nullable && !<unstrided_bf16 as ::kernels::Derivation>::DERIVED[2].nullable);
};

// The `_with_fp16` form still names `Out(1)`; only `nullable` moved to `false`.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(rmsnorm_bf16_with_fp16);
    assert!(d.len() == 4);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(!<rmsnorm_bf16_with_fp16 as ::kernels::Derivation>::DERIVED[3].nullable);
    // Stated, not counted: a bare `*mut f16` would derive the same `Out(1)` today.
};
