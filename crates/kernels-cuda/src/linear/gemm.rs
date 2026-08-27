//! `Gemm`: dense projections against a transposed weight. Everything that
//! used to be a choice upstream — gemv vs cuBLAS vs cuBLASLt, algorithm
//! index, unroll depth per architecture — lives behind this file's entries,
//! in the sibling `dense`/`gemv` modules (decision #13): a dispatch arm
//! never sees it.

use kernels::KernelError;

use crate::jit::{Ctx, dtype_dispatch, stated};
use crate::tensor::Tensor;

#[cfg(feature = "_cuda")]
use super::dense;

pub fn matmul(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), KernelError> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), KernelError> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

/// `y = act x w^T`. An empty projection (any extent zero) is a silent no-op,
/// as before — a conditioned batch may legitimately land nothing, and a
/// refusal here would kill the whole fire under graph capture.
pub fn act_x_wt(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let m = stated(op, y.rows)?;
    let n = stated(op, y.width)?;
    let k = stated(op, act.width)?;

    #[cfg(feature = "_cuda")]
    {
        dense::act_x_wt(ctx, op, act.ptr, w.ptr, y.ptr, m, n, k)
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (ctx, w, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}
