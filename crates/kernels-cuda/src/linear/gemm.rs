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
    // **D4: THE OPAQUE CALLEE SEES THE BUCKET** (`.wiki/palo/cuda-abi.md` §3,
    // refined form). Below this entry is `dense`, and below `dense` is
    // cuBLASLt's shape→algorithm function — a heuristic nobody publishes and
    // which the probe caught swapping kernels and splitK factors on M alone.
    // Rounding M up to the fire's lattice point is the one lever that freezes
    // it without knowing it: the library is handed a per-bucket constant, so
    // its private decisions become per-bucket constants too. Nothing else in
    // this call moves — N and K are the weight's and are already shape
    // constants, and every kernel this tree owns keeps its live extent.
    //
    // The rows `[rows, bucket)` this makes the gemm read and write are in
    // bounds (the arena reserves at the ceiling, P0 refuses a lattice above
    // it) and harmless (a gemm is row-independent, so the garbage that lands
    // there stays there) — and they are NOBODY'S only because the driver
    // declined to arm a windowed region at all. This entry does not and cannot
    // check that: `Ctx::opaque_rows` carries both gates and the argument for
    // each. What the entry owes is the extent of the rectangle it was handed,
    // which is what it passes.
    //
    // **AND ONE THING THIS DOES NOT PRESERVE, STATED WHERE IT IS DONE.** A
    // padded call is a different cuBLASLt kernel, and a different kernel is a
    // different reduction order: the live rows come back numerically equal and
    // not bit-equal to the unpadded call's. Two fires agree bit-for-bit iff
    // they share a bucket. That is the price of freezing the arm, it is what
    // freezing the arm MEANS, and it is measured in
    // `a_padded_fire_is_in_bounds_and_says_something_true`.
    let m = ctx.opaque_rows(stated(op, y.rows)?);
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
