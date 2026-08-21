
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

use core::ffi::c_void;
use core::ptr::NonNull;

const BLOCK: u32 = 256;

const VBLOCK: u32 = 512;

const WARP: u32 = 32;

const ALTUP_BLOCK: u32 = 128;

pub const ALTUP_EPS: f32 = 1e-5;

pub const RASR_VEC512_ABOVE: i32 = 2560;

pub const MAX_HC_MULT: i32 = 8;

#[must_use]
const fn per_row(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), BLOCK)
}

#[must_use]
const fn per_row_reducing(rows: i32) -> Launch {

    const RMS_SMEM: u32 = (BLOCK / WARP) * 4;

    Launch::per_row(rows.unsigned_abs(), BLOCK).smem(RMS_SMEM)
}

#[must_use]
const fn elementwise(n: i32) -> Launch {
    Launch::flat(n.unsigned_abs(), BLOCK)
}

#[must_use]
fn elementwise_wide(n: i64) -> Launch {
    let blocks = (n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
    Launch::grid([u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1], [BLOCK, 1, 1])
}

#[must_use]
const fn route_rows(rows: i32, width: i32) -> Launch {

    const MAX_BLOCK: u32 = 1024;

    let warps = width.unsigned_abs().div_ceil(WARP);
    let warps = if warps == 0 { 1 } else { warps };
    let block = warps.saturating_mul(WARP);
    let block = if block > MAX_BLOCK { MAX_BLOCK } else { block };
    Launch::grid([rows.unsigned_abs(), 1, 1], [block, 1, 1])
}

#[must_use]
const fn gated_rms(rows: i32, heads: i32) -> Launch {
    Launch::grid([rows.unsigned_abs(), heads.unsigned_abs(), 1], [BLOCK, 1, 1])
}

#[must_use]
const fn altup_streams(rows: i32, streams: i32, hidden: i32) -> Launch {
    Launch::grid(
        [rows.unsigned_abs(), streams.unsigned_abs(), hidden.unsigned_abs().div_ceil(ALTUP_BLOCK)],
        [ALTUP_BLOCK, 1, 1],
    )
}

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

fn square_side<P>(row: &Region<P>) -> Result<i32, Refusal> {
    let width = row.width;

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

#[routine]
pub fn rmsnorm_strided_bf16(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let eps = *eps;
    rmsnorm_strided_bf16_at(ctx, x, weight, y, eps)
}

pub fn rmsnorm_strided_bf16_at(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    eps: f32,
) -> Result<(), Refusal> {

    let dst = y.all("the normalised row's width")?;
    let src = x.all("the source row's pitch")?;

    let vec_ok = vec8_ok(
        x.ptr.cast(),
        y.ptr.cast_const().cast(),
        weight.v.cast(),
        dst.width,
        src.stride,
        dst.stride,
    );
    if vec_ok {

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

#[routine(internal)]
pub fn unstrided_bf16(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    _eps: Const<f32>) -> Result<(), Refusal> {

    let _eps = *_eps;

    let packed = In { ptr: x.ptr, rows: y.rows, width: y.width };
    rmsnorm_strided_bf16(ctx, packed, weight, y)
}

#[routine]
pub fn rmsnorm_bf16_with_fp16(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    y_fp16: Out<Tensor<f16>>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let eps = *eps;

    let dst = y.all("the normalised row's width")?;
    if !vec8_ok(
        x.ptr.cast(),
        y.ptr.cast_const().cast(),
        weight.v.cast(),
        dst.width,
        dst.stride,
        dst.stride,
    ) {

        let n = i64::from(dst.rows) * i64::from(dst.width);

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

#[routine(bf16)]
pub fn rmsnorm<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    y: Out<Tensor<T>>,
    per_head_dim: Const<i32>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let per_head_dim = *per_head_dim;
    let eps = *eps;

    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }

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

#[routine(bf16)]
pub fn rmsnorm_gemma<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    y: Out<Tensor<T>>,
    per_head_dim: Const<i32>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let per_head_dim = *per_head_dim;
    let eps = *eps;

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

#[routine(bf16)]
pub fn rmsnorm_no_scale<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    y: Out<Tensor<T>>,
    per_head_dim: Const<i32>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let per_head_dim = *per_head_dim;
    let eps = *eps;
    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm_no_scale<{}, 256>", T::CPP))).apply(launch), &[x.arg(), y.arg(), hidden.arg(), eps.arg()])
}

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

#[routine(bf16)]
pub fn rmsnorm_gated<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    gate: In<Tensor<T>>,
    weight: Const<Tensor<f32>>,
    y: Out<Tensor<T>>,
    per_head_dim: Const<i32>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let per_head_dim = *per_head_dim;
    let eps = *eps;

    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm_gated<{}, 256>", T::CPP))).apply(launch), &[x.arg(), gate.arg(), weight.arg(), y.arg(), hidden.arg(), eps.arg()])
}

#[routine(bf16)]
pub fn rmsnorm_gated_fp32_in<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<f32>>,
    gate: In<Tensor<T>>,
    weight: Const<Tensor<f32>>,
    y: Out<Tensor<T>>,
    eps: Const<f32>,
    per_head_dim: Const<i32>) -> Result<(), Refusal> {

    let eps = *eps;

    let per_head_dim = *per_head_dim;
    let dst = y.all("the normalised row's width")?;
    let hidden = if per_head_dim == 0 { dst.width } else { per_head_dim };
    let launch = rows_per_head(&dst, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm_gated_f32_in<{}, 256>", T::CPP))).apply(launch), &[x.arg(), gate.arg(), weight.arg(), y.arg(), hidden.arg(), eps.arg()])
}

#[routine(bf16)]
pub fn residual_add_rmsnorm<

    T: crate::RoutineElem + kernels::routine::Elem<Read = *const T, Write = *mut T>,
>(
    ctx: &Ctx<'_>,
    hidden: In<Tensor<T>>,
    residual: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    norm_out: Out<Tensor<T>>,
    eps: Const<f32>) -> Result<(), Refusal> {

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

#[routine(bf16)]
pub fn rmsnorm_residual_add<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    hidden: InOut<Tensor<T>>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let eps = *eps;

    let dst = hidden.all("the normalised row's width")?;
    ctx.fire(Fire::at("norm/rmsnorm.cuh", crate::jit::symbol(&format!("::pie::norm::rmsnorm_residual_add<{}, 256>", T::CPP))).apply(per_row(dst.rows)), &[x.arg(), weight.arg(), hidden.arg(), dst.width.arg(), eps.arg()])
}

#[routine]
pub fn rmsnorm_residual_add_scale_rmsnorm_bf16(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    weight: Const<Tensor<bf16>>,
    hidden: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    next_weight: Const<Tensor<bf16>>,
    norm_out: Out<Tensor<bf16>>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let eps = *eps;

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

#[routine(bf16)]
pub fn add_bias<T>(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<T>>,
    bias: Const<Tensor<T>>) -> Result<(), Refusal> {

    let dst = out.all("the biased row's width")?;
    ctx.fire(Fire::at("norm/add_bias.cuh", crate::jit::symbol(&format!("::pie::norm::add_bias<{}>", T::CPP))).apply(route_rows(dst.rows, dst.width)), &[out.arg(), bias.arg(), dst.width.arg()])
}

#[routine(bf16)]
pub fn altup_predict<T>(
    ctx: &Ctx<'_>,
    streams: In<Tensor<T>>,
    coefs: In<Tensor<f32>>,
    predictions: Out<Tensor<T>>,

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

#[routine(bf16)]
pub fn altup_correct<T>(
    ctx: &Ctx<'_>,
    predictions: In<Tensor<T>>,
    activated: In<Tensor<T>>,
    correction_coefs_plus_one: In<Tensor<f32>>,
    corrected: Out<Tensor<T>>,
    active_idx: Const<i32>) -> Result<(), Refusal> {

    let active_idx = *active_idx;
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

#[routine(bf16)]
pub fn compute_rms<T>(
    ctx: &Ctx<'_>,
    reference: In<Tensor<T>>,
    out: Out<Tensor<f32>>,

) -> Result<(), Refusal> {
    let src = reference.all("the reduced row's width")?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", crate::jit::symbol(&format!("::pie::norm::compute_rms<{}>", T::CPP))).apply(per_row_reducing(src.rows)), &[reference.arg(), out.arg(), src.width.arg(), ALTUP_EPS.arg()])
}

#[routine(bf16)]
pub fn magnitude_rescale<T>(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<T>>,
    target_rms: In<Tensor<f32>>) -> Result<(), Refusal> {

    let dst = x.all("the rescaled row's width")?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", crate::jit::symbol(&format!("::pie::norm::magnitude_rescale<{}>", T::CPP))).apply(per_row_reducing(dst.rows)), &[x.arg(), target_rms.arg(), dst.width.arg(), ALTUP_EPS.arg()])
}

#[routine(bf16)]
pub fn mean_streams<T>(
    ctx: &Ctx<'_>,
    streams: In<Tensor<T>>,
    out: Out<Tensor<T>>,

) -> Result<(), Refusal> {

    #[must_use]
    const fn elementwise_rows(rows: i32, width: i32) -> Launch {
        Launch::grid([rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1], [BLOCK, 1, 1])
    }

    let dst = out.all("the averaged row's width")?;

    let src = streams.all("the AltUp stream row's width")?;
    let k = altup_factor(&src, dst.width, "the averaged row's width")?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", crate::jit::symbol(&format!("::pie::norm::mean_streams<{}>", T::CPP))).apply(elementwise_rows(dst.rows, dst.width)), &[streams.arg(), out.arg(), k.arg(), dst.rows.arg(), dst.width.arg()])
}

#[routine]
pub fn altup_unpack_predict_coefs(
    ctx: &Ctx<'_>,
    in_bf16: In<Tensor<bf16>>,
    out: Out<Tensor<f32>>,

) -> Result<(), Refusal> {
    let packed = in_bf16.all("the packed coefficients' width")?;
    let k = square_side(&packed)?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", "::pie::norm::unpack_predict_coefs<::pie::bf16>").apply(route_rows(out.rows, k.saturating_mul(k))), &[in_bf16.arg(), out.arg(), k.arg()])
}

#[routine]
pub fn altup_unpack_correct_coefs(
    ctx: &Ctx<'_>,
    in_bf16: In<Tensor<bf16>>,
    out: Out<Tensor<f32>>,

) -> Result<(), Refusal> {

    let packed = in_bf16.all("the packed coefficients' width")?;
    ctx.fire(Fire::at("norm/altup_aux.cuh", "::pie::norm::unpack_correct_coefs<::pie::bf16>").apply(route_rows(out.rows, packed.width)), &[in_bf16.arg(), out.arg(), packed.width.arg()])
}

#[routine(bf16)]
pub fn tanh<T>(ctx: &Ctx<'_>, x: InOut<Tensor<T>>) -> Result<(), Refusal> {

    let rect = x.all("the rectangle's row width")?;
    let n = rect.elements();
    ctx.fire(Fire::at("norm/altup_aux.cuh", crate::jit::symbol(&format!("::pie::norm::tanh_inplace<{}>", T::CPP))).apply(elementwise(n)), &[x.arg(), n.arg()])
}

#[routine(internal)]
pub fn tanh_f16(ctx: &Ctx<'_>, x: InOut<Tensor<f16>>) -> Result<(), Refusal> {
    let rect = x.all("the rectangle's row width")?;
    let n = rect.elements();
    ctx.fire(Fire::at("norm/altup_aux.cuh", "::pie::norm::tanh_inplace<\
                                              ::pie::f16>").apply(elementwise(n)), &[x.arg(), n.arg()])
}

#[routine(bf16)]
pub fn residual_add<T>(ctx: &Ctx<'_>, y: InOut<Tensor<T>>, x: In<Tensor<T>>) -> Result<(), Refusal> {
    let rect = y.all("the rectangle's row width")?;
    let Ok(n) = usize::try_from(rect.elements()) else {
        return Err(Refusal::Empty { what: "the residual rectangle's element count" });
    };
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    ctx.fire(Fire::at("norm/elementwise.cuh", crate::jit::symbol(&format!("::pie::norm::residual_add<{}>", T::CPP))).apply(launch), &[y.arg(), x.arg(), n.arg()])
}

#[routine(internal)]
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

#[routine(bf16)]
pub fn scalar_mul<T>(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<T>>,
    s: Const<f32>) -> Result<(), Refusal> {
    let rect = x.all("the rectangle's row width")?;
    let Ok(n) = usize::try_from(rect.elements()) else {
        return Err(Refusal::Empty { what: "the scaled rectangle's element count" });
    };
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    ctx.fire(Fire::at("norm/elementwise.cuh", crate::jit::symbol(&format!("::pie::norm::scalar_mul<{}>", T::CPP))).apply(launch), &[x.arg(), s.arg(), n.arg()])
}

#[routine(bf16)]
pub fn hc_pre_postprocess<T>(
    ctx: &Ctx<'_>,
    mixes: In<Tensor<f32>>,
    scale: Const<Tensor<f32>>,
    base: Const<Tensor<f32>>,
    residual: In<Tensor<T>>,
    post_mix: Out<Tensor<f32>>,
    comb_mix: Out<Tensor<f32>>,
    layer_input: Out<Tensor<T>>,
    hc_eps: Const<f32>,
    hc_post_alpha: Const<f32>,
    sinkhorn_iters: Const<i32>) -> Result<(), Refusal> {
    // THE RECTANGLE ANSWERS THREE OF THE SIX. `n` is the row count and
    // `hidden_size` the collapsed row's width, both on `layer_input`; the
    // stream count is what `residual` is wider by, which is exactly what
    // `streams` computes and what `hc_post` already reads that way. Only the
    // three CONSTANTS are asked for.
    let dst = layer_input.all("the hyper-connection row's width")?;
    let (n, hidden_size) = (dst.rows, dst.width);
    let hc_mult = streams(&residual.all("the residual row's width")?, hidden_size)?;
    let hc_eps = *hc_eps;
    let hc_post_alpha = *hc_post_alpha;
    let sinkhorn_iters = *sinkhorn_iters;
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

#[routine(bf16)]
pub fn hc_post<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    residual: In<Tensor<T>>,
    post_mix: In<Tensor<f32>>,
    comb_mix: In<Tensor<f32>>,
    out_residual: Out<Tensor<T>>) -> Result<(), Refusal> {

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

#[routine(bf16)]
pub fn hc_head_postprocess<T>(
    ctx: &Ctx<'_>,
    mixes: In<Tensor<f32>>,
    scale: Const<Tensor<f32>>,
    base: Const<Tensor<f32>>,
    residual: In<Tensor<T>>,
    out: Out<Tensor<T>>,
    hc_eps: Const<f32>) -> Result<(), Refusal> {
    // [`hc_pre_postprocess`]'s reading, off the one result this form declares.
    let dst = out.all("the collapsed row's width")?;
    let (n, hidden_size) = (dst.rows, dst.width);
    let hc_mult = streams(&residual.all("the residual row's width")?, hidden_size)?;
    let hc_eps = *hc_eps;
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

#[routine]
pub fn hc_rmsnorm_to_f32(
    ctx: &Ctx<'_>,
    input: In<Tensor<bf16>>,
    output: Out<Tensor<f32>>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let eps = *eps;

    let dst = output.all("the normalised row's width")?;
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", "::pie::norm::hc_rmsnorm_to_f32<::pie::bf16, 256>").apply(per_row(dst.rows)), &[input.arg(), output.arg(), dst.width.arg(), eps.arg()])
}

#[routine(bf16)]
pub fn attn_sink_correction<T>(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<T>>,
    lse: In<Tensor<f32>>,
    sink: Const<Tensor<f32>>,
    head_dim: Const<i32>) -> Result<(), Refusal> {

    let head_dim = *head_dim;

    let dst = out.all("the row whose heads are counted")?;
    let num_heads = heads(&dst, head_dim)?;
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", crate::jit::symbol(&format!("::pie::norm::attn_sink_correction<{}>", T::CPP))).apply(gated_rms(dst.rows, num_heads)), &[out.arg(), lse.arg(), sink.arg(), num_heads.arg(), head_dim.arg()])
}

#[routine(bf16)]
pub fn per_head_rmsnorm<T>(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<T>>,
    head_dim: Const<i32>,
    eps: Const<f32>) -> Result<(), Refusal> {

    let head_dim = *head_dim;
    let eps = *eps;

    let dst = q.all("the row whose heads are counted")?;
    let num_heads = heads(&dst, head_dim)?;
    ctx.fire(Fire::at("norm/dsv4_hc.cuh", crate::jit::symbol(&format!("::pie::norm::per_head_rmsnorm<{}>", T::CPP))).apply(gated_rms(dst.rows, num_heads)), &[q.arg(), head_dim.arg(), eps.arg()])
}

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
