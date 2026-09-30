//! NVFP4 projections: e2m1 codes, two per byte (low nibble first), one e4m3
//! scale per 16 codes, one f32 tensor scale, as kernels-wgpu
//! `quant/nvfp4.wgsl` decodes them: `y = tensor_scale · Σ x · e2m1(c) ·
//! e4m3(s)`. The codes run through [`quant`](super::quant)'s split planes
//! (group-batched for decode, one dense dot for prefill); a code times its
//! e4m3 scale is exactly a bf16, so neither form rounds the weight.
//!
//! Plane layout the engine binds: `codes` `U8 [n, k / 2]` (or the `Nvfp4`
//! dtype, read as those bytes), `scales` `U8`/`E4m3 [n, k / 16]`.

use dtype::Dtype;

use crate::cx::{Ctx, Cx};
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Val};
use crate::linear::gemm::{extent, head_rows};
use crate::linear::quant::{byte_view, contract_codes, e2m1, e4m3};
use crate::tensor::Tensor;

const GROUP_CODES: u32 = 16;

pub fn matmul(
    ctx: &Ctx<'_>,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.matmul", act, codes, scales, tensor_scale, y)
}

pub fn lm_head(
    ctx: &Ctx<'_>,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", act, codes, scales, tensor_scale, y)
}

fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: Tensor,
) -> Result<(), Error> {
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
    let scales = Tensor::new(scales.buf, scales.rows, scales.width, Dtype::U8);
    if m == 0 {
        return Ok(());
    }
    ctx.emit(&mut |cx| {
        let x = cx.read(act)?;
        let x = head_rows(cx, x, m)?;
        let x = if cx.elem(x) == Elem::F16 {
            cx.convert(x, Elem::F32)
        } else {
            x
        };
        let out = contract(cx, x, codes, scales, k)?;
        let out = cx.scale(out, f64::from(tensor_scale))?;
        cx.write(y, out)
    })
}

/// `x · wᵀ` without the tensor scale, f32 `[m, n]`.
fn contract(cx: &mut Cx<'_>, x: Val, codes: Tensor, scales: Tensor, k: u32) -> Result<Val, Error> {
    let g = i64::from(k / GROUP_CODES);
    let raw = cx.read(codes)?;
    let s = cx.read(scales)?;
    let s = cx.convert(s, Elem::U32);
    let s = e4m3(cx, s)?;
    contract_codes(cx, x, raw, 4, g, s, None, &|cx, w| e2m1(cx, w), true)
}
