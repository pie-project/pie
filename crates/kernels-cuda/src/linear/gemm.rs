//! `Gemm`: dense projections against a transposed weight. Everything that
//! used to be a choice upstream — gemv vs cuBLAS vs cuBLASLt, algorithm
//! index, unroll depth per architecture — lives behind this file's entries,
//! in the sibling `dense`/`gemv` modules (decision #13): a dispatch arm
//! never sees it.

use crate::error::Error;

use crate::jit::{Ctx, count, dtype_dispatch, stated};
use crate::tensor::Tensor;

#[cfg(feature = "_cuda")]
use super::dense;

pub fn matmul(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

/// `y = act x w^T`. A fire with no ROWS is a silent no-op: the batch extent is
/// a composition fact, a conditioned batch may legitimately land nothing, and
/// a refusal here would kill the whole fire under graph capture.
///
/// **AND CAPTURE IS THE CASE THE NO-OP IS FOR.** `engine::fire::walk`'s rule 1
/// drops a zero-row node before the dispatch, so an eager fire never reaches
/// this entry with nothing to project; a recorded one makes the same decision
/// on the DEVICE instead, which is precisely the instant at which a refusal
/// would take a whole graph down over a node that had nothing to do.
///
/// **THE TWO WIDTHS ARE NOT THAT FACT.** They are the weight's — fixed by the
/// checkpoint, the same in every fire this artifact runs — so a zero in either
/// is a malformed weight row that would land nothing forever rather than a
/// composition that happened to be empty, and it is refused. They are checked
/// BEFORE the row count, so a malformed weight answers the same on every fire
/// and not only on the ones that had rows to project.
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
    // there stays there) — and they are NOBODY'S only because the engine
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
