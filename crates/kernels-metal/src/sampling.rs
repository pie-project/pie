use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

pub mod selector_walk;

pub fn argmax(ctx: &Ctx<'_>, x: Tensor, column: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "sampling.argmax";
    debug_assert_eq!(y.dtype, Dtype::I32, "`{OP}` writes i32 column indices");
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "argmax_rows_bfloat16",
        F32 => "argmax_rows_float32",
    });
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
        Fire::at("sampling/argmax_rows.metal", entry)
            .apply(Grid::of([1024, rows, 1], [1024, 1, 1])),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, column)?.arg(),
        ],
    )
}

const TOPK_THREADS: u32 = 128;
/// A row this wide is a vocabulary: one 128-thread group a row left the
/// device a few cores' worth of work for a draft block's eight rows.
const TOPK_WIDE_FROM: u32 = 16_384;
const TOPK_WIDE_THREADS: u32 = 1024;

pub fn topk(
    ctx: &Ctx<'_>,
    x: Tensor,
    k: u32,
    values: Tensor,
    indices: Tensor,
) -> Result<(), Error> {
    const OP: &str = "sampling.top_k";
    let wide = x.width >= TOPK_WIDE_FROM;
    let entry = match (x.dtype, k, wide) {
        (Dtype::Bf16, 8, false) => "topk_rows_bfloat16_k_8",
        (Dtype::Bf16, 16, false) => "topk_rows_bfloat16_k_16",
        (Dtype::F32, 8, false) => "topk_rows_float32_k_8",
        (Dtype::F32, 16, false) => "topk_rows_float32_k_16",
        (Dtype::Bf16, 8, true) => "topk_rows_wide_bfloat16_k_8",
        (Dtype::Bf16, 16, true) => "topk_rows_wide_bfloat16_k_16",
        (Dtype::F32, 8, true) => "topk_rows_wide_float32_k_8",
        (Dtype::F32, 16, true) => "topk_rows_wide_float32_k_16",
        (dtype, k, _) => {
            return Err(refuse(
                OP,
                format!(
                    "no point is stamped for {dtype:?} rows at k = {k}; the plane stamps bf16 and f32 at 8 and 16"
                ),
            ));
        }
    };
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
    let threads = if wide {
        TOPK_WIDE_THREADS
    } else {
        TOPK_THREADS
    };
    ctx.fire(
        Fire::at("sampling/topk.metal", entry).apply(Grid::of([threads, rows, 1], [threads, 1, 1])),
        &[
            x.arg(),
            values.arg_mut(),
            indices.arg_mut(),
            stated(OP, x.width)?.arg(),
        ],
    )
}
