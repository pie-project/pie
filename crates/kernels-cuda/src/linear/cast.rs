//! Dtype casts a weight lands through (`linear/cast.cu`): f32 planes to the
//! engine's dtype, and a scale over rows.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, kernel, nonzero, stated};
use crate::linear::affine::extent;
use crate::tensor::Tensor;

const CAST: &str = "linear/cast.cu";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

pub fn cast_fp32_to(ctx: &Ctx, src: Tensor, dst: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_cast_fp32_to";
    debug_assert_eq!(src.dtype, Dtype::F32, "`{OP}` casts an f32 source");
    let t = dtype_dispatch!(OP, dst.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let n = dst.elements();
    let lanes = nonzero(OP, "the cast's element count", extent(OP, n)?)?;
    ctx.fire(
        OP,
        Fire::at(kernel(CAST, &format!("::pie::linear::cast_f32_to<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[src.arg(), dst.arg(), n.arg()],
    )
}

pub fn scale_rows(ctx: &Ctx, l: Tensor, buf: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_scale_rows";
    let t = dtype_dispatch!(OP, buf.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "rows", buf.rows)?;
    let width = stated(OP, nonzero(OP, "width", buf.width)?)?;
    ctx.fire(
        OP,
        Fire::at(kernel(CAST, &format!("::pie::linear::scale_rows<{t}>")))
            .apply(route_rows(buf.rows, buf.width)),
        &[buf.arg(), l.arg(), width.arg()],
    )
}

/// A norm's gain `g` (`width` long, bf16 or f32) folded into `bank` (`rows x
/// width`, bf16) in place: each weight times its column's `g`, or `1 + g`
/// where `plus_one`, rounded once to bf16 (`kernels_ir::anchor::fold_bank`).
pub fn fold_gain(ctx: &Ctx, bank: &mut Tensor, g: Tensor, plus_one: bool) -> Result<(), Error> {
    const OP: &str = "linear.fold_gain";
    let t = dtype_dispatch!(OP, bank.dtype, { Bf16 => "::pie::bf16" });
    let gt = dtype_dispatch!(OP, g.dtype, { Bf16 => "::pie::bf16", F32 => "float" });
    if g.elements() != u64::from(bank.width) {
        return Err(crate::jit::refuse(
            OP,
            format!("a gain of {} for a bank {} wide", g.elements(), bank.width),
        ));
    }
    let n = bank.elements();
    let lanes = nonzero(OP, "the bank's element count", extent(OP, n)?)?;
    let width = stated(OP, nonzero(OP, "width", bank.width)?)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            CAST,
            &format!("::pie::linear::fold_gain<{t}, {gt}, {plus_one}>"),
        ))
        .apply(Launch::flat(lanes, BLOCK)),
        &[bank.arg(), g.arg(), n.arg(), width.arg()],
    )
}
