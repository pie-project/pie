//! The CUDA `norm` family: RMS norm and everything fused onto it.
//!
//! RMS norms, elementwise passes sharing their launch shape (residual/bias add,
//! `tanh`, scalar multiply), Gemma-3n's AltUp set, and DeepSeek's `hc_*` pre/post-passes.
//!
//! Every `# Safety` below also requires `ctx`'s stream live across the (async) launch.
#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine, aligned16};
use crate::routine;
use crate::jit::Abi;
use crate::jit::abi::Inst;
use crate::jit::abi::{bf16, f16};
use kernels::Refusal;
use kernels::Region;
use kernels::Stride;
use kernels::Bank;
use kernels::Weight;
use kernels::keys;
use kernels::routine::{Env, ParamF32};
use kernels::routine::In;
use kernels::routine::InOut;
use kernels::routine::Out;

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
#[kernels_macros::routine]
pub fn rmsnorm_strided_bf16(
    ctx: &Ctx,
    x: In<0, bf16>,
    weight: Bank<0, bf16>,
    y: Out<0, bf16>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    // `vec8_ok` accepts a zero pitch (`0 % 8 == 0`); `y`'s width is checked first.
    let dst = y.all("the normalised row's width")?;
    let src = x.all("the source row's pitch")?;
    // A stride is not a width: each pitch comes off its own region, not `dst.width`.
    let vec_ok = vec8_ok(
        x.ptr.cast(),
        y.ptr.cast_const().cast(),
        weight.ptr.cast(),
        dst.width,
        src.stride,
        dst.stride,
    );
    if vec_ok {
        // SAFETY: every pointer is live for the extent the kernel reads it as.
        return unsafe {
            ctx.launch(
                "norm/rmsnorm.cuh",
                "::pie::norm::rmsnorm_vec8<::pie::i32(512), false, false>",
                Launch::per_row(dst.rows.unsigned_abs(), VBLOCK),
                &[
                    x.ptr.arg(),
                    weight.ptr.arg(),
                    y.ptr.arg(),
                    None::<NonNull<f16>>.arg(),
                    dst.width.arg(),
                    src.stride.arg(),
                    dst.stride.arg(),
                    eps.arg(),
                ],
            )
        };
    }
    // SAFETY: as above.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            "::pie::norm::rmsnorm<::pie::bf16, 256>",
            per_row(dst.rows),
            &[
                x.ptr.arg(),
                weight.ptr.arg(),
                y.ptr.arg(),
                dst.width.arg(),
                src.stride.arg(),
                dst.stride.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `norm::rmsnorm_bf16` — one call, nothing else.
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s, unchanged.
#[kernels_macros::routine]
pub fn unstrided_bf16(
    ctx: &Ctx,
    x: In<0, bf16>,
    weight: Bank<0, bf16>,
    y: Out<0, bf16>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    // Reads `x` through `y`'s rectangle: both widths here are `y.width`.
    let packed = In { ptr: x.ptr, rows: y.rows, width: y.width };
    rmsnorm_strided_bf16(ctx, packed, weight, y, eps)
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
#[kernels_macros::routine]
pub fn rmsnorm_bf16_with_fp16(
    ctx: &Ctx,
    x: In<0, bf16>,
    weight: Bank<0, bf16>,
    y: Out<0, bf16>,
    y_fp16: Out<1, f16>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    let dst = y.all("the normalised row's width")?;
    if !vec8_ok(
        x.ptr.cast(),
        y.ptr.cast_const().cast(),
        weight.ptr.cast(),
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
        unstrided_bf16(ctx, x, weight, y, eps)?;
        return crate::quant::bf16_to_fp16(
            ctx,
            In { ptr: y.ptr.cast_const(), rows: y.rows, width: y.width },
            kernels::routine::Out { ptr: y_fp16.ptr, rows: dst.rows, width: dst.width },
        );
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            "::pie::norm::rmsnorm_vec8<::pie::i32(512), false, true>",
            Launch::per_row(dst.rows.unsigned_abs(), VBLOCK),
            &[
                x.ptr.arg(),
                weight.ptr.arg(),
                y.ptr.arg(),
                y_fp16.ptr.arg(),
                dst.width.arg(),
                dst.stride.arg(),
                dst.stride.arg(),
                eps.arg(),
            ],
        )
    }
}

/// The `OpKind::Rmsnorm` launcher — `norm::rmsnorm_bf16`.
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s.
#[kernels_macros::routine]
pub fn rmsnorm<T>(
    ctx: &Ctx,
    x: In<0, T>,
    weight: Weight<0, *const T>,
    y: Out<0, T>,
    // Not a width: how many columns of `dst.width` one reduction covers; never refuses.
    per_head_dim: Env<keys::PerHeadDim>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let dst = y.all("the normalised row's width")?;
    let hidden = if **per_head_dim == 0 { dst.width } else { **per_head_dim };
    let launch = rows_per_head(&dst, **per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // The one place `dst.stride` would be wrong: pitch between rows here is `hidden`.
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            &format!("::pie::norm::rmsnorm<{}, 256>", T::CPP),
            launch,
            &[
                x.ptr.arg(),
                weight.ptr.arg(),
                y.ptr.arg(),
                hidden.arg(),
                hidden.arg(),
                hidden.arg(),
                eps.arg(),
            ],
        )
    }
}

/// gemma's `(1 + w)` fold — `norm::rmsnorm_gemma_bf16`.
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s.
#[kernels_macros::routine]
pub fn rmsnorm_gemma<T>(
    ctx: &Ctx,
    x: In<0, T>,
    // Named, not banked: gemma's per-head callers use `OpKind::RmsnormPerHead`, no `Arg::Weight`.
    weight: Weight<0, *const T>,
    y: Out<0, T>,
    per_head_dim: Env<keys::PerHeadDim>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let dst = y.all("the normalised row's width")?;
    let hidden = if **per_head_dim == 0 { dst.width } else { **per_head_dim };
    let launch = rows_per_head(&dst, **per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            &format!("::pie::norm::rmsnorm_gemma<{}, 256>", T::CPP),
            launch,
            &[
                x.ptr.arg(),
                weight.ptr.arg(),
                y.ptr.arg(),
                hidden.arg(),
                hidden.arg(),
                hidden.arg(),
                eps.arg(),
            ],
        )
    }
}

/// The weightless per-head norm — `norm::rmsnorm_no_scale_bf16`.
///
/// # Safety
///
/// `x` and `y` must address `y.rows * y.width` live bf16 elements.
#[kernels_macros::routine]
pub fn rmsnorm_no_scale<T>(
    ctx: &Ctx,
    // `In<0, _>`, not `In<1, _>`: spends two pointers, unlike [`residual_add`]'s one.
    x: In<0, T>,
    y: Out<0, T>,
    per_head_dim: Env<keys::PerHeadDim>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let dst = y.all("the normalised row's width")?;
    let hidden = if **per_head_dim == 0 { dst.width } else { **per_head_dim };
    let launch = rows_per_head(&dst, **per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            &format!("::pie::norm::rmsnorm_no_scale<{}, 256>", T::CPP),
            launch,
            &[x.ptr.arg(), y.ptr.arg(), hidden.arg(), eps.arg()],
        )
    }
}

/// The gated norm — `norm::rmsnorm_gated_bf16`.
///
/// # Safety
///
/// `x`, `gate`, `y` address `y.rows * y.width` live bf16; `weight` `hidden` live floats.
#[kernels_macros::routine]
pub fn rmsnorm_gated<T>(
    ctx: &Ctx,
    x: In<0, T>,
    gate: In<1, T>,
    // Banked, not named: reached only via a launch stating the weight as `Arg::Weight`.
    weight: Bank<0, f32>,
    y: Out<0, T>,
    per_head_dim: Env<keys::PerHeadDim>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let dst = y.all("the normalised row's width")?;
    let hidden = if **per_head_dim == 0 { dst.width } else { **per_head_dim };
    let launch = rows_per_head(&dst, **per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            &format!("::pie::norm::rmsnorm_gated<{}, 256>", T::CPP),
            launch,
            &[x.ptr.arg(), gate.ptr.arg(), weight.ptr.arg(), y.ptr.arg(), hidden.arg(), eps.arg()],
        )
    }
}

/// The gated norm with an fp32 input — `norm::rmsnorm_gated_fp32_in_bf16`.
///
/// # Safety
///
/// `x` must address `y.rows * y.width` live floats, `gate` and `y` the same
/// count of bf16, `weight` `hidden` live floats.
#[kernels_macros::routine]
pub fn rmsnorm_gated_fp32_in<T>(
    ctx: &Ctx,
    x: In<0, f32>,
    gate: In<1, T>,
    // Named, not banked: this op's builder pushes `vec![x, gate]`, no weight operand.
    weight: Weight<0, *const f32>,
    y: Out<0, T>,
    // Unmarked: no arm reaches this symbol; the source was a struct field, not this fire's.
    per_head_dim: i32,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            &format!("::pie::norm::rmsnorm_gated_f32_in<{}, 256>", T::CPP),
            launch,
            &[x.ptr.arg(), gate.ptr.arg(), weight.ptr.arg(), y.ptr.arg(), hidden.arg(), eps.arg()],
        )
    }
}

/// The residual add and the next block's pre-norm, fused.
///
/// # Safety
///
/// `hidden`, `residual`, `norm_out`: `norm_out.rows * norm_out.width` bf16; `weight`: that width.
#[kernels_macros::routine]
pub fn residual_add_rmsnorm<T>(
    ctx: &Ctx,
    // `InOut<0, T>`: listed as an input with no output; `Source` stays `Slot(Kind::In, 0)`.
    hidden: InOut<0, T>,
    residual: In<1, T>,
    weight: Bank<0, T>,
    norm_out: Out<0, T>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // A zero width reduces over nothing, so this must be a view.
    let dst = norm_out.all("the normalised row's width")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            &format!("::pie::norm::residual_add_rmsnorm<{}, 256>", T::CPP),
            per_row(dst.rows),
            &[
                hidden.ptr.arg(),
                residual.ptr.arg(),
                weight.ptr.arg(),
                norm_out.ptr.arg(),
                dst.width.arg(),
                eps.arg(),
            ],
        )
    }
}

/// Norm, then add into the residual stream in place —
///
/// # Safety
///
/// [`residual_add_rmsnorm`]'s.
#[kernels_macros::routine]
pub fn rmsnorm_residual_add<T>(
    ctx: &Ctx,
    x: In<0, T>,
    // Banked: `alias()` leaves a `Weight` slot alone; a bare pointer would shift to `In(2)`.
    weight: Bank<0, T>,
    hidden: Out<0, T>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let dst = hidden.all("the normalised row's width")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            &format!("::pie::norm::rmsnorm_residual_add<{}, 256>", T::CPP),
            per_row(dst.rows),
            &[x.ptr.arg(), weight.ptr.arg(), hidden.ptr.arg(), dst.width.arg(), eps.arg()],
        )
    }
}

/// `norm::rmsnorm_residual_add_scale_rmsnorm_bf16`: all three arms differ only in
/// instantiation -- vectorised above/below [`RASR_VEC512_ABOVE`], or scalar if unaligned.
///
/// # Safety
///
/// `x`, `weight`, `hidden`, `next_weight`, `norm_out` address live bf16 elements:
/// `hidden.rows * hidden.width` (the two weights, `hidden.width`).
#[kernels_macros::routine]
pub fn rmsnorm_residual_add_scale_rmsnorm_bf16(
    ctx: &Ctx,
    x: In<0, bf16>,
    // `Bank<0, _>`/`Bank<1, _>`: stating `x` sets the input counter without consuming a slot.
    weight: Bank<0, bf16>,
    hidden: Out<0, bf16>,
    // Unmarked: the per-layer constant the binder reads has no spelling a statement can carry.
    scale: f32,
    next_weight: Bank<1, bf16>,
    norm_out: Out<1, bf16>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    // A zero would pick the vectorised path and normalise nothing (`0 % 8 == 0`).
    let dst = hidden.all("the normalised row's width")?;
    let rows = dst.rows.unsigned_abs();
    let hidden_size = dst.width;
    let vec_ok = hidden_size % 8 == 0
        && aligned16(x.ptr.cast())
        && aligned16(hidden.ptr.cast_const().cast())
        && aligned16(norm_out.ptr.cast_const().cast())
        && aligned16(weight.ptr.cast())
        && aligned16(next_weight.ptr.cast());
    let (instantiation, block) = if vec_ok {
        if hidden_size >= RASR_VEC512_ABOVE {
            ("::pie::norm::rmsnorm_rasr_vec8<::pie::i32(512)>", VBLOCK)
        } else {
            ("::pie::norm::rmsnorm_rasr_vec8<::pie::i32(256)>", BLOCK)
        }
    } else {
        ("::pie::norm::rmsnorm_residual_add_scale_rmsnorm<::pie::bf16, 512>", VBLOCK)
    };
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/rmsnorm.cuh",
            instantiation,
            Launch::per_row(rows, block),
            &[
                x.ptr.arg(),
                weight.ptr.arg(),
                hidden.ptr.arg(),
                scale.arg(),
                next_weight.ptr.arg(),
                norm_out.ptr.arg(),
                hidden_size.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `out[row][i] += bias[i]` — `norm::add_bias_bf16`.
///
/// # Safety
///
/// `out` must address `out.rows * out.width` live bf16 elements, `bias` `out.width` of them.
#[kernels_macros::routine]
pub fn add_bias<T>(
    ctx: &Ctx,
    out: Out<0, T>,
    bias: Weight<0, *const T>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // A zero here reads no bias yet succeeds (`route_rows` clamps up rather than refusing).
    let dst = out.all("the biased row's width")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/add_bias.cuh",
            &format!("::pie::norm::add_bias<{}>", T::CPP),
            route_rows(dst.rows, dst.width),
            &[out.ptr.arg(), bias.ptr.arg(), dst.width.arg()],
        )
    }
}

/// gemma-3n's altup predict — `norm::altup_predict_bf16`.
///
/// # Safety
///
/// `streams`/`predictions`: `k * predictions.rows * h` bf16; `coefs`: `predictions.rows * k * k`.
#[kernels_macros::routine]
pub fn altup_predict<T>(
    ctx: &Ctx,
    streams: In<0, T>,
    coefs: In<1, f32>,
    predictions: Out<0, T>,
    // `k`/`h` computed here, not sourced: `k` is `coefs`'s root, `h` is `streams.width / k`.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let coef_row = coefs.all("the predict coefficients' row")?;
    let k = square_side(&coef_row)?;
    let stream_row = streams.all("the AltUp stream row's width")?;
    let h = altup_factor(&stream_row, k, "the AltUp stream count")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/altup.cuh",
            &format!("::pie::norm::altup_predict<{}>", T::CPP),
            altup_streams(predictions.rows, k, h),
            &[
                streams.ptr.arg(),
                coefs.ptr.arg(),
                predictions.ptr.arg(),
                k.arg(),
                predictions.rows.arg(),
                h.arg(),
            ],
        )
    }
}

/// gemma-3n's altup correct — `norm::altup_correct_bf16`.
///
/// # Safety
///
/// [`altup_predict`]'s, with `activated` addressing `corrected.rows *
/// activated.width` and `corrected` `k` times that, live bf16 elements.
#[kernels_macros::routine]
pub fn altup_correct<T>(
    ctx: &Ctx,
    predictions: In<0, T>,
    activated: In<1, T>,
    correction_coefs_plus_one: In<2, f32>,
    corrected: Out<0, T>,
    // Unmarked: nothing fills `cfg.altup_active_idx` yet (the driver writes a literal `0`).
    active_idx: i32,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // Both extents are separate views: a zero `k` or `h` would otherwise still launch.
    let coef_row = correction_coefs_plus_one.all("the correction coefficients' width")?;
    let act_row = activated.all("the activated stream's width")?;
    let (k, h) = (coef_row.width, act_row.width);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/altup.cuh",
            &format!("::pie::norm::altup_correct<{}>", T::CPP),
            altup_streams(corrected.rows, k, h),
            &[
                predictions.ptr.arg(),
                activated.ptr.arg(),
                correction_coefs_plus_one.ptr.arg(),
                corrected.ptr.arg(),
                k.arg(),
                corrected.rows.arg(),
                h.arg(),
                active_idx.arg(),
            ],
        )
    }
}

/// The per-row RMS of the reference stream — `norm::compute_rms_bf16`.
///
/// # Safety
///
/// `reference` addresses `reference.rows * reference.width` bf16; `out` `reference.rows` floats.
#[kernels_macros::routine]
pub fn compute_rms<T>(
    ctx: &Ctx,
    // Off `reference`, not `out`: the result is one float per row (`out.width` is always 1).
    reference: In<0, T>,
    out: Out<0, f32>,
    // `ALTUP_EPS`, not `Env<keys::RmsEps>`: the algorithm's own constant, not the deployment's.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
{
    let src = reference.all("the reduced row's width")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/altup_aux.cuh",
            &format!("::pie::norm::compute_rms<{}>", T::CPP),
            per_row_reducing(src.rows),
            &[reference.ptr.arg(), out.ptr.arg(), src.width.arg(), ALTUP_EPS.arg()],
        )
    }
}

/// Rescale each row to a stated RMS, in place — `norm::magnitude_rescale_bf16`.
///
/// # Safety
///
/// `x` must address `x.rows * x.width` live bf16 elements, `target_rms` `x.rows` live floats.
#[kernels_macros::routine]
pub fn magnitude_rescale<T>(
    ctx: &Ctx,
    x: Out<0, T>,
    target_rms: In<1, f32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    // A zero rescales no element, so the rectangle arrives as a view.
    let dst = x.all("the rescaled row's width")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/altup_aux.cuh",
            &format!("::pie::norm::magnitude_rescale<{}>", T::CPP),
            per_row_reducing(dst.rows),
            &[x.ptr.arg(), target_rms.ptr.arg(), dst.width.arg(), ALTUP_EPS.arg()],
        )
    }
}

/// The mean over altup's `k` streams — `norm::mean_streams_bf16`.
///
/// # Safety
///
/// `streams` addresses `k * out.rows * out.width` bf16; `out` `out.rows * out.width` of them.
#[kernels_macros::routine]
pub fn mean_streams<T>(
    ctx: &Ctx,
    streams: In<0, T>,
    out: Out<0, T>,
    // A quotient of the two widths this statement carries: `streams.width / out.width`.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    /// Pointwise with the row on its own grid axis.
    #[must_use]
    const fn elementwise_rows(rows: i32, width: i32) -> Launch {
        Launch::grid([rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1], [BLOCK, 1, 1])
    }

    // A zero here averages nothing, so this must be a view, not a read.
    let dst = out.all("the averaged row's width")?;

    let src = streams.all("the AltUp stream row's width")?;
    let k = altup_factor(&src, dst.width, "the averaged row's width")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/altup_aux.cuh",
            &format!("::pie::norm::mean_streams<{}>", T::CPP),
            elementwise_rows(dst.rows, dst.width),
            // `t_stride` is the row count, not a row pitch: steps one whole stream plane.
            &[streams.ptr.arg(), out.ptr.arg(), k.arg(), dst.rows.arg(), dst.width.arg()],
        )
    }
}

/// bf16 predict coefficients widened to `float` —
///
/// # Safety
///
/// `in_bf16` must address `out.rows * k * k` live bf16 elements, `out` the same count of floats.
#[kernels_macros::routine]
pub fn altup_unpack_predict_coefs(
    ctx: &Ctx,
    in_bf16: In<0, bf16>,
    out: Out<0, f32>,
    // `k` is the exact square root of `in_bf16.width`: coefficients are `[t, k * k]`.
) -> Result<(), Refusal> {
    let packed = in_bf16.all("the packed coefficients' width")?;
    let k = square_side(&packed)?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/altup_aux.cuh",
            "::pie::norm::unpack_predict_coefs<::pie::bf16>",
            route_rows(out.rows, k.saturating_mul(k)),
            &[in_bf16.ptr.arg(), out.ptr.arg(), k.arg()],
        )
    }
}

/// bf16 correct coefficients widened to `float` —
///
/// # Safety
///
/// `in_bf16` must address `out.rows * k` live bf16 elements, `out` the same count of floats.
#[kernels_macros::routine]
pub fn altup_unpack_correct_coefs(
    ctx: &Ctx,
    in_bf16: In<0, bf16>,
    out: Out<0, f32>,
    // Unlike the predict twin, no square root: `k` is `packed.width` directly.
) -> Result<(), Refusal> {
    // A zero here unpacks nothing, so the count must come off a view.
    let packed = in_bf16.all("the packed coefficients' width")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/altup_aux.cuh",
            "::pie::norm::unpack_correct_coefs<::pie::bf16>",
            route_rows(out.rows, packed.width),
            &[in_bf16.ptr.arg(), out.ptr.arg(), packed.width.arg()],
        )
    }
}

/// `tanh` in place over a bf16 slab — `norm::tanh_bf16`.
///
/// # Safety
///
/// `x` must address `x.rows * x.width` live bf16 elements.
#[kernels_macros::routine]
pub fn tanh<T>(ctx: &Ctx, x: Out<0, T>) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    // Guards the product through the width: a missing width looks like an ordinary zero.
    let rect = x.all("the rectangle's row width")?;
    let n = rect.elements();
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/altup_aux.cuh",
            &format!("::pie::norm::tanh_inplace<{}>", T::CPP),
            elementwise(n),
            &[x.ptr.arg(), n.arg()],
        )
    }
}

/// [`tanh`] over fp16 — `norm::tanh_f16`.
///
/// # Safety
///
/// [`tanh`]'s, with `x` addressing fp16.
#[kernels_macros::routine]
pub fn tanh_f16(ctx: &Ctx, x: Out<0, f16>) -> Result<(), Refusal> {
    let rect = x.all("the rectangle's row width")?;
    let n = rect.elements();
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch("norm/altup_aux.cuh", "::pie::norm::tanh_inplace<\
                                              ::pie::f16>", elementwise(n), &[x.ptr.arg(), n.arg()])
    }
}

/// `y += x` — `norm::residual_add_bf16`.
///
/// # Safety
///
/// `y` and `x` must address `y.rows * y.width` live bf16 elements.
// `x` is `In<1, _>`, not `In<0, _>`: one pointer (`y`) serves the in-place buffer.
// `usize` because the CUDA signature takes `usize n`.
#[kernels_macros::routine]
pub fn residual_add<T>(ctx: &Ctx, y: Out<0, T>, x: In<1, T>) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let rect = y.all("the rectangle's row width")?;
    let Ok(n) = usize::try_from(rect.elements()) else {
        return Err(Refusal::Empty { what: "the residual rectangle's element count" });
    };
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/elementwise.cuh",
            &format!("::pie::norm::residual_add<{}>", T::CPP),
            launch,
            &[y.ptr.arg(), x.ptr.arg(), n.arg()],
        )
    }
}

/// [`residual_add`] over fp16 — `norm::residual_add_f16`.
///
/// # Safety
///
/// [`residual_add`]'s, with both pointers addressing fp16.
#[kernels_macros::routine]
pub fn residual_add_f16(
    ctx: &Ctx,
    y: Out<0, f16>,
    x: In<1, f16>,
) -> Result<(), Refusal> {
    let rect = y.all("the rectangle's row width")?;
    let Ok(n) = usize::try_from(rect.elements()) else {
        return Err(Refusal::Empty { what: "the residual rectangle's element count" });
    };
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/elementwise.cuh",
            "::pie::norm::residual_add<::pie::f16>",
            launch,
            &[y.ptr.arg(), x.ptr.arg(), n.arg()],
        )
    }
}

/// `x *= s` — `norm::scalar_mul_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements.
#[kernels_macros::routine]
pub fn scalar_mul<T>(
    ctx: &Ctx,
    x: Out<0, T>,
    // The scale rides the param channel, not an extent: eight sites pass five distinct values.
    s: ParamF32<0>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    let rect = x.all("the rectangle's row width")?;
    let Ok(n) = usize::try_from(rect.elements()) else {
        return Err(Refusal::Empty { what: "the scaled rectangle's element count" });
    };
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/elementwise.cuh",
            &format!("::pie::norm::scalar_mul<{}>", T::CPP),
            launch,
            &[x.ptr.arg(), s.arg(), n.arg()],
        )
    }
}

/// `norm::hc_pre_postprocess_bf16`.
///
/// # Safety
///
/// `residual`/`layer_input`: `n * hc_mult * hidden_size`/`n * hidden_size` bf16;
/// `mixes`/`scale`/`base` the layer's slabs; `post_mix`/`comb_mix` scratch floats.
#[kernels_macros::routine]
pub fn hc_pre_postprocess<T>(
    ctx: &Ctx,
    mixes: In<0, f32>,
    // No `Cx` query exists for these float slabs; `mixes` became an operand, these two did not.
    scale: Env<*const f32>,
    base: Env<*const f32>,
    residual: In<1, T>,
    post_mix: Out<0, f32>,
    comb_mix: Out<1, f32>,
    layer_input: Out<2, T>,
    // Unmarked: a half-bound row would claim bindings nobody checked.
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    hc_eps: f32,
    hc_post_alpha: f32,
    sinkhorn_iters: i32,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    hc_mult_ok(hc_mult)?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/dsv4_hc.cuh",
            &format!("::pie::norm::hc_pre_postprocess<{}, 256>", T::CPP),
            per_row(n),
            &[
                mixes.ptr.arg(),
                scale.arg(),
                base.arg(),
                residual.ptr.arg(),
                post_mix.ptr.arg(),
                comb_mix.ptr.arg(),
                layer_input.ptr.arg(),
                hc_mult.arg(),
                hidden_size.arg(),
                hc_eps.arg(),
                hc_post_alpha.arg(),
                sinkhorn_iters.arg(),
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn hc_post<T>(
    ctx: &Ctx,
    x: In<0, T>,
    residual: In<1, T>,
    post_mix: In<2, f32>,
    comb_mix: In<3, f32>,
    out_residual: Out<0, T>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // Only the numerator is a view; `x.width`'s zero is `streams`'s own `Empty`.
    let dst = out_residual.all("the hyper-connection row's width")?;
    let hc_mult = streams(&dst, x.width)?;
    let total = i64::from(dst.rows) * i64::from(x.width);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/dsv4_hc.cuh",
            &format!("::pie::norm::hc_post<{}>", T::CPP),
            elementwise_wide(total),
            &[
                x.ptr.arg(),
                residual.ptr.arg(),
                post_mix.ptr.arg(),
                comb_mix.ptr.arg(),
                out_residual.ptr.arg(),
                dst.rows.arg(),
                hc_mult.arg(),
                x.width.arg(),
            ],
        )
    }
}

/// `norm::hc_head_postprocess_bf16`.
///
/// # Safety
///
/// [`hc_pre_postprocess`]'s, with `out` addressing `n * hidden_size` live bf16 elements.
#[kernels_macros::routine]
pub fn hc_head_postprocess<T>(
    ctx: &Ctx,
    mixes: In<0, f32>,
    scale: Env<*const f32>,
    base: Env<*const f32>,
    residual: In<1, T>,
    out: Out<0, T>,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    hc_eps: f32,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    hc_mult_ok(hc_mult)?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/dsv4_hc.cuh",
            &format!("::pie::norm::hc_head_postprocess<{}, 256>", T::CPP),
            per_row(n),
            &[
                mixes.ptr.arg(),
                scale.arg(),
                base.arg(),
                residual.ptr.arg(),
                out.ptr.arg(),
                hc_mult.arg(),
                hidden_size.arg(),
                hc_eps.arg(),
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn hc_expand<T>(
    ctx: &Ctx,
    input: In<0, T>,
    output: Out<0, T>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let dst = output.all("the hyper-connection row's width")?;
    let hc_mult = streams(&dst, input.width)?;
    let total = i64::from(input.rows) * i64::from(input.width);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/dsv4_hc.cuh",
            &format!("::pie::norm::hc_expand<{}>", T::CPP),
            elementwise_wide(total),
            &[
                input.ptr.arg(),
                output.ptr.arg(),
                input.rows.arg(),
                hc_mult.arg(),
                input.width.arg(),
            ],
        )
    }
}

/// `norm::hc_rmsnorm_to_f32`.
///
/// # Safety
///
/// `input` addresses `output.rows * output.width` bf16; `output` the same count of floats.
#[kernels_macros::routine]
pub fn hc_rmsnorm_to_f32(
    ctx: &Ctx,
    input: In<0, bf16>,
    output: Out<0, f32>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    // The one hyper-connection launcher that doesn't divide, so `streams`'s guard doesn't apply.
    let dst = output.all("the normalised row's width")?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/dsv4_hc.cuh",
            "::pie::norm::hc_rmsnorm_to_f32<::pie::bf16, 256>",
            per_row(dst.rows),
            &[input.ptr.arg(), output.ptr.arg(), dst.width.arg(), eps.arg()],
        )
    }
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
#[kernels_macros::routine]
pub fn attn_sink_correction<T>(
    ctx: &Ctx,
    // [`residual_add`]'s and [`magnitude_rescale`]'s shape again: `lse` is input one, not zero.
    out: Out<0, T>,
    lse: In<1, f32>,
    // Banked: a weight slot consumes no input counter, so `lse` stays at `In(1)`.
    sink: Bank<0, f32>,
    // `Env<keys::HeadDim>`, not `KvPageSize`: same number on CUDA, a different fact elsewhere.
    head_dim: Env<keys::HeadDim>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    let dst = out.all("the row whose heads are counted")?;
    let num_heads = heads(&dst, **head_dim)?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/dsv4_hc.cuh",
            &format!("::pie::norm::attn_sink_correction<{}>", T::CPP),
            gated_rms(dst.rows, num_heads),
            &[out.ptr.arg(), lse.ptr.arg(), sink.ptr.arg(), num_heads.arg(), head_dim.arg()],
        )
    }
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
#[kernels_macros::routine]
pub fn per_head_rmsnorm<T>(
    ctx: &Ctx,
    // `Out<0, _>`, no `In<_, _>`: one pointer serves both read and write.
    q: Out<0, T>,
    // [`attn_sink_correction`]'s `head_dim`, not `kv_layer().head_dim` (a different fact).
    head_dim: Env<keys::HeadDim>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    let dst = q.all("the row whose heads are counted")?;
    let num_heads = heads(&dst, **head_dim)?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "norm/dsv4_hc.cuh",
            &format!("::pie::norm::per_head_rmsnorm<{}>", T::CPP),
            gated_rms(dst.rows, num_heads),
            &[q.ptr.arg(), head_dim.arg(), eps.arg()],
        )
    }
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

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are derived from the `fn`s above; what's stated here
/// is what no signature carries -- which operands must share an address.
pub static ROUTINES: &[Routine] = &[
    routine!(rmsnorm_strided_bf16, ),
    routine!(unstrided_bf16, ),
    routine!(rmsnorm_bf16_with_fp16, ),
    routine!(rmsnorm_bf16 = rmsnorm::<bf16>, ),
    routine!(rmsnorm_gemma_bf16 = rmsnorm_gemma::<bf16>, ),
    routine!(rmsnorm_no_scale_bf16 = rmsnorm_no_scale::<bf16>, in_place = &[(0, 0)], ),
    routine!(rmsnorm_gated_bf16 = rmsnorm_gated::<bf16>, ),
    routine!(rmsnorm_gated_fp32_in_bf16 = rmsnorm_gated_fp32_in::<bf16>, ),
    // `(1, 0)` is `(output, input)` where output 1 is one past the only
    // declared result: `residual_add_rmsnorm` writes `norm_out` and also
    // `hidden`, a `*mut` sitting in an input slot with no matching output.
    routine!(residual_add_rmsnorm_bf16 = residual_add_rmsnorm::<bf16>, in_place = &[(1, 0)], ),
    routine!(rmsnorm_residual_add_bf16 = rmsnorm_residual_add::<bf16>, in_place = &[(0, 1)], ),
    routine!(
        rmsnorm_residual_add_scale_rmsnorm_bf16,
        in_place = &[(0, 1)]
    ),
    routine!(add_bias_bf16 = add_bias::<bf16>, in_place = &[(0, 0)], ),
    routine!(altup_predict_bf16 = altup_predict::<bf16>, ),
    routine!(altup_correct_bf16 = altup_correct::<bf16>, ),
    routine!(compute_rms_bf16 = compute_rms::<bf16>, ),
    routine!(magnitude_rescale_bf16 = magnitude_rescale::<bf16>, in_place = &[(0, 0)], ),
    routine!(mean_streams_bf16 = mean_streams::<bf16>, ),
    routine!(altup_unpack_predict_coefs, ),
    routine!(altup_unpack_correct_coefs, ),
    routine!(tanh_bf16 = tanh::<bf16>, in_place = &[(0, 0)], ),
    routine!(tanh_f16, in_place = &[(0, 0)], ),
    routine!(residual_add_bf16 = residual_add::<bf16>, in_place = &[(0, 0)], ),
    routine!(residual_add_f16, in_place = &[(0, 0)], ),
    routine!(scalar_mul_bf16 = scalar_mul::<bf16>, in_place = &[(0, 0)], ),
    routine!(hc_pre_postprocess_bf16 = hc_pre_postprocess::<bf16>, ),
    routine!(hc_post_bf16 = hc_post::<bf16>, ),
    routine!(hc_head_postprocess_bf16 = hc_head_postprocess::<bf16>, ),
    routine!(hc_expand_bf16 = hc_expand::<bf16>, ),
    routine!(hc_rmsnorm_to_f32, ),
    routine!(attn_sink_correction_bf16 = attn_sink_correction::<bf16>, in_place = &[(0, 0)], ),
    routine!(per_head_rmsnorm_bf16 = per_head_rmsnorm::<bf16>, in_place = &[(0, 0)], ),
];

/// `norm`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

// `scalar_mul`'s derivation, checked by `cargo check` rather than a test.
const _: () = {
    let d = <scalar_mul as kernels::Derivation>::DERIVED;
    assert!(d.len() == 2);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
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

// `InOut<0, T>` is about direction, not position: still derives `In(0)`, not `Out(0)`.
const _: () = {
    let d = <residual_add_rmsnorm as kernels::Derivation>::DERIVED;
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
};

// Pins [`unstrided_bf16`]'s derivation against a rename to `rmsnorm_bf16` (a different signature).
const _: () = {
    let d = <unstrided_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 4);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `Named(_)`, not the specific key: a `str` isn't matchable in a const.
    assert!(matches!(d[3].source, Some(kernels::Source::Named(_))));
    assert!(!d[0].nullable && !d[1].nullable && !d[2].nullable && !d[3].nullable);
};

// The `_with_fp16` form still names `Out(1)`; only `nullable` moved to `false`.
const _: () = {
    let d = <rmsnorm_bf16_with_fp16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 5);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(d[4].source, Some(kernels::Source::Named(_))));
    assert!(!d[3].nullable);
    // Stated, not counted: a bare `*mut f16` would derive the same `Out(1)` today.
    assert!(d[3].stated);
};
