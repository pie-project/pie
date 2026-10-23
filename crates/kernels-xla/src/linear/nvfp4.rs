//! NVFP4 projections: e2m1 codes, two per byte (low nibble first), one e4m3
//! scale per 16 codes, one f32 tensor scale, as kernels-cuda's
//! `linear/dequant.cu` decodes them: `y = tensor_scale · Σ x · e2m1(c) ·
//! e4m3(s)`. The codes run through [`quant`](super::quant)'s split planes
//! (group-batched for decode, one dense dot for prefill); a code times its
//! e4m3 scale is exactly a bf16, so neither form rounds the weight.
//!
//! Plane layout the engine binds: `codes` `U8 [n, k / 2]` (or the `Nvfp4`
//! dtype, read as those bytes), `scales` `U8`/`E4m3 [n, k / 16]`.

use dtype::Dtype;

use crate::error::{Error, refuse};
use crate::hlo::{Elem, Func, Val};
use crate::linear::gemm::{extent, head_rows};
use crate::linear::quant::{byte_view, contract_codes, e2m1, e4m3};
use crate::tensor::{Landing, Operand, Stored};
use crate::tuning::Tuning;

const GROUP_CODES: u32 = 16;

/// The planes as they are read for an `[n, k]` projection: the codes as
/// `U8 [n, k / 2]` bytes, the scales as their bytes. The engine reads these
/// views.
pub fn views<T: Stored>(codes: T, scales: T, k: u32) -> (T, T) {
    let codes = if crate::check::elem_of("linear.nvfp4", codes.dtype()).is_err() {
        codes.with_width(k / 2).with_dtype(Dtype::U8)
    } else {
        codes
    };
    (codes, scales.with_dtype(Dtype::U8))
}

/// `y = act · wᵀ` for nvfp4 planes read as [`views`] says. Nothing when `y`
/// has no rows.
#[allow(clippy::too_many_arguments)]
pub fn matmul(
    f: &mut Func,
    act: Operand,
    codes: Operand,
    scales: Operand,
    tensor_scale: f32,
    y: Landing,
    tuning: Tuning,
) -> Result<Option<Val>, Error> {
    fire(
        f,
        "linear.matmul",
        act,
        codes,
        scales,
        tensor_scale,
        y,
        tuning,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn lm_head(
    f: &mut Func,
    act: Operand,
    codes: Operand,
    scales: Operand,
    tensor_scale: f32,
    y: Landing,
    tuning: Tuning,
) -> Result<Option<Val>, Error> {
    fire(
        f,
        "linear.matmul",
        act,
        codes,
        scales,
        tensor_scale,
        y,
        tuning,
    )
}

#[allow(clippy::too_many_arguments)]
fn fire(
    f: &mut Func,
    op: &'static str,
    act: Operand,
    codes: Operand,
    scales: Operand,
    tensor_scale: f32,
    y: Landing,
    tuning: Tuning,
) -> Result<Option<Val>, Error> {
    let (m, n, k) = extent(op, act, y)?;
    if !k.is_multiple_of(GROUP_CODES) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {GROUP_CODES}-code nvfp4 groups"),
        ));
    }
    let codes = byte_view(op, "code", codes, n, u64::from(k) / 2)?;
    if !matches!(scales.dtype, Dtype::U8 | Dtype::E4m3) || scales.width != k / GROUP_CODES {
        return Err(refuse(
            op,
            format!(
                "a {}-wide {:?} scale row is not one e4m3 byte per {GROUP_CODES} codes over a \
                 {k}-wide row",
                scales.width, scales.dtype
            ),
        ));
    }
    if codes.rows != n || scales.rows != n {
        return Err(refuse(
            op,
            format!(
                "the planes have {} code rows and {} scale rows over an {n}-column \
                 projection; both are one row per column",
                codes.rows, scales.rows
            ),
        ));
    }
    if !tensor_scale.is_finite() {
        return Err(refuse(
            op,
            format!("the tensor scale is {tensor_scale}, which every output would carry"),
        ));
    }
    let scales = scales.with_dtype(Dtype::U8);
    if m == 0 {
        return Ok(None);
    }
    let x = act.val;
    let x = head_rows(f, x, m)?;
    let x = if f.elem(x) == Elem::F16 {
        f.convert(x, Elem::F32)
    } else {
        x
    };
    let out = contract(f, x, codes, scales, k, tuning)?;
    let out = f.scale(out, f64::from(tensor_scale))?;
    Ok(Some(out))
}

/// `x · wᵀ` without the tensor scale, f32 `[m, n]`.
fn contract(
    f: &mut Func,
    x: Val,
    codes: Operand,
    scales: Operand,
    k: u32,
    tuning: Tuning,
) -> Result<Val, Error> {
    let g = i64::from(k / GROUP_CODES);
    let raw = codes.val;
    let s = scales.val;
    let s = f.convert(s, Elem::U32);
    let s = e4m3(f, s)?;
    contract_codes(f, x, raw, 4, g, s, None, &|f, w| e2m1(f, w), true, tuning)
}
