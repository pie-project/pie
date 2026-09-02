//! `Gemm`: dense projections against a transposed weight. gemv vs cuBLAS vs
//! cuBLASLt, algorithm index, and unroll depth per architecture live behind
//! this file's entries, in the sibling `dense`/`gemv` modules.

use crate::error::Error;

use crate::jit::{Ctx, count, dtype_dispatch, stated};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use super::dense;

pub fn matmul(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

/// `y = act x w^T`. A fire with no rows is a silent no-op: the batch extent
/// is a composition fact that may legitimately land nothing, and a refusal
/// here would kill the whole fire under graph capture, where a recorded
/// no-op fire still reaches this entry.
///
/// The two widths are not that fact: they are the weight's, fixed by the
/// checkpoint, so a zero in either is a malformed weight and is refused
/// (checked before the row count, so it answers the same on every fire).
pub fn act_x_wt(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = count(op, "the columns this projection lands", y.width)?;
    let k = count(op, "the contraction this projection walks", act.width)?;
    if y.rows == 0 {
        return Ok(());
    }
    // cuBLASLt's shape->algorithm heuristic swaps kernels and splitK factors
    // on M alone; rounding M up to the fire's lattice point (bucket) freezes
    // that choice per bucket. N and K are the weight's own shape constants
    // and are unaffected. The padded rows `[rows, bucket)` are in bounds (the
    // arena reserves at the ceiling) and harmless (a gemm is row-independent).
    // A padded call is a different cuBLASLt kernel and so a different
    // reduction order: results are numerically but not bit-equal across a
    // bucket boundary.
    let m = ctx.opaque_rows(stated(op, y.rows)?);

    #[cfg(feature = "cuda")]
    {
        dense::act_x_wt(ctx, op, act.ptr, w.ptr, y.ptr, m, n, k)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ctx, w, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}
