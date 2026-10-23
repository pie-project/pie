//! The block-fp8 matvec over bf16 rows (`linear/gemv_fp8.cu`).

use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, stated, symbol};
use crate::tensor::Tensor;

const GEMV_FP8: &str = "linear/gemv_fp8.cu";

/// A block-FP8 projection (`n x k` e4m3 codes at `codes`, their ue8m0
/// exponents laid in atoms a row a `gran` columns, 128 or 32) at a decode's
/// few rows as a GEMV (`linear/fp8_gemv.cuh`): the activations as they are,
/// bf16, the codes streamed once. `Ok(false)` past `kernel.gemm.fp8_gemv_until` rows
/// or for a shape it does not read (`k` a whole number of 128s: a warp's
/// lanes past `k` on its last 512-code pass read nothing).
pub fn fp8_block_gemv(
    ctx: &Ctx,
    act: Tensor,
    codes: u64,
    atoms: u64,
    gran: u32,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "linear.fp8_block_gemv";
    const WARPS: u32 = 8;
    const ROWS: u32 = 4;
    let (m, n, k) = (act.rows, y.width, act.width);
    if m == 0
        || m > ctx.tuning().gemm_fp8_gemv_until
        || act.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || y.rows != m
        || !matches!(gran, 32 | 128)
        || k == 0
        || !k.is_multiple_of(128)
        || n == 0
        || !codes.is_multiple_of(16)
        || !act.ptr.is_multiple_of(16)
    {
        return Ok(false);
    }
    // Warps a row group: each at least two 512-code passes of K.
    let splits = (k / 1024).clamp(1, WARPS).next_power_of_two().min(WARPS);
    let rows_pad: u32 = match m {
        1 => 1,
        2 => 2,
        3 | 4 => 4,
        _ => return Ok(false),
    };
    ctx.fire(
        OP,
        Fire::at(
            GEMV_FP8,
            symbol(&format!(
                "::pie::linear::fp8_block_gemv<{WARPS}, {ROWS}, {rows_pad}{}>",
                if gran == 128 {
                    String::new()
                } else {
                    format!(", {gran}")
                }
            )),
        )
        .apply(
            Launch::grid(
                [n.div_ceil(ROWS * (WARPS / splits)), 1, 1],
                [WARPS * 32, 1, 1],
            )
            .programmatic(ctx.tuning().pdl),
        ),
        &[
            act.arg(),
            stated(OP, act.width)?.arg(),
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            y.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, m)?.arg(),
            stated(OP, n)?.arg(),
            stated(OP, k)?.arg(),
            stated(OP, splits)?.arg(),
        ],
    )?;
    Ok(true)
}
