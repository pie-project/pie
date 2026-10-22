use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const TOPK_FILE: &str = "sampling/topk.hip";

const TOPK_THREADS: u32 = 1024;

/// The dynamic LDS a top-k block may pool candidates in: 64 KiB less the
/// kernel's own few KiB.
const TOPK_POOL_BYTES: u32 = 56 * 1024;

pub fn argmax(ctx: &Ctx, x: Tensor, column: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "sampling.argmax";
    const THREADS: u32 = 1024;
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F32 => "float" });
    debug_assert_eq!(y.dtype, Dtype::I32, "`{OP}` writes i32 column indices");
    let rows = nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "width", x.width)?;
    if column >= y.width {
        return Err(refuse(
            OP,
            format!(
                "column {column} is outside the {}-wide plane it writes",
                y.width
            ),
        ));
    }
    debug_assert_eq!(x.rows, y.rows, "an argmax lands one entry per row");
    ctx.fire(
        OP,
        Fire::at(
            TOPK_FILE,
            symbol(&format!("::pie::layout::argmax_rows<{t}>")),
        )
        .apply(Launch::per_row(rows, THREADS)),
        &[
            x.arg(),
            y.arg(),
            stated(OP, x.width)?.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, column)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn top_k(
    ctx: &Ctx,
    x: Tensor,
    k: u32,
    values: &mut Tensor,
    indices: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "sampling.top_k";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F32 => "float" });
    if k != 8 && k != 16 {
        return Err(refuse(
            OP,
            format!("no point is stamped at k = {k}; the plane stamps bf16 and f32 at 8 and 16"),
        ));
    }
    let rows = nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "width", x.width)?;
    if values.rows != rows || values.width != k || values.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!("the values plane is not [{rows}, {k}] f32"),
        ));
    }
    if indices.rows != rows || indices.width != k || indices.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!("the indices plane is not [{rows}, {k}] i32"),
        ));
    }
    // The kernel pools the columns reaching its floor in LDS: at most k
    // threads' columns, ⌈width / threads⌉ + 8 each, as 8-byte keys.
    let pool = u64::from(k) * (u64::from(x.width.div_ceil(TOPK_THREADS)) + 8) * 8;
    let pool = u32::try_from(pool)
        .ok()
        .filter(|&bytes| bytes <= TOPK_POOL_BYTES)
        .ok_or_else(|| {
            refuse(
                OP,
                format!(
                    "a {}-wide row pools {pool} bytes of top-{k} candidates, past the {TOPK_POOL_BYTES} \
                     bytes of LDS a block holds beside its own",
                    x.width
                ),
            )
        })?;
    ctx.fire(
        OP,
        Fire::at(
            TOPK_FILE,
            symbol(&format!("::pie::layout::topk_rows<{t}, {k}>")),
        )
        .apply(Launch::per_row(rows, TOPK_THREADS).smem(pool)),
        &[
            x.arg(),
            values.arg(),
            indices.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}
