//! Host-only: the dense products on the vendor's libraries. `cublas` times
//! cuBLAS and cuBLASLt tactics for a bf16 product; `cublas_lt_fp8` is
//! cuBLASLt's fp8 product over `act_quant`'s scales.

use crate::error::Error;

use crate::jit::{Ctx, count, dtype_dispatch, stated};
use crate::linear::skeleton::{Epilogue, In};
use crate::tensor::Tensor;

use dtype::Dtype;

#[cfg(feature = "cuda")]
pub(crate) mod cublas;
#[cfg(feature = "cuda")]
pub mod cublas_lt_fp8;

pub fn matmul(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

/// `act . w^T` over bf16 operands landing f32 (a router's logits).
pub fn matmul_f32(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.matmul_f32";
    dtype_dispatch!(OP, act.dtype, { Bf16 => () });
    dtype_dispatch!(OP, w.dtype, { Bf16 => () });
    dtype_dispatch!(OP, y.dtype, { F32 => () });
    let n = count(OP, "the columns this projection lands", y.width)?;
    let k = count(OP, "the contraction this projection walks", act.width)?;
    if w.rows != y.width || w.width != act.width {
        return Err(crate::jit::refuse(
            OP,
            format!(
                "the weight is {} x {} and this projection contracts {k} into {n}",
                w.rows, w.width
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    let m = ctx.opaque_rows(stated(OP, y.rows)?);
    #[cfg(feature = "cuda")]
    {
        cublas::act_x_wt_f32(ctx, OP, act.ptr, w.ptr, y.ptr, m, n, k)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = m;
        Err(crate::jit::runtimeless(OP))
    }
}

/// `outs = epi(act . w^T)`: an anchored region (kernels-ir's, its epilogue
/// printed for the contraction skeletons, [`Epilogue`]) over a dense bf16 bank
/// of `parts · n` rows, `n` its results' width, `ins` its epilogue's inputs.
/// The GEMM tuner times, per shape and epilogue, each contraction skeleton
/// that covers it fused against the plain product's own tactic followed by
/// the epilogue on the row skeleton, and fires the fastest; a fusion never
/// forces a slower product. A shape no skeleton covers (or one a capture meets
/// untimed) runs that unfused leg. False, landing nothing, only where the
/// row skeleton does not carry the leg either ([`crate::linear::skeleton::serves`]
/// says which do): the caller's unfused ops serve it then.
pub fn anchored(
    ctx: &Ctx,
    op: &'static str,
    epi: &'static Epilogue,
    act: Tensor,
    w: Tensor,
    ins: &[In],
    outs: &[Tensor],
) -> Result<bool, Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    dtype_dispatch!(op, w.dtype, { Bf16 => () });
    if outs.is_empty() {
        return Err(crate::jit::refuse(op, "an anchored region of no result"));
    }
    let tile = w.rows / epi.parts().max(1);
    let n = count(op, "the columns this projection lands", tile)?;
    let k = count(op, "the contraction this projection walks", act.width)?;
    // Each result its window of the tile (every column, or a split's piece).
    let shaped = (0..outs.len()).zip(outs).all(|(at, y)| {
        y.dtype == Dtype::Bf16 && y.width == epi.window(at, tile).width && y.rows == act.rows
    });
    if !shaped || w.width != act.width || w.rows != tile * epi.parts() || !epi.fits(tile) {
        return Err(crate::jit::refuse(
            op,
            format!(
                "the weight is {} x {} and the region contracts {k} into {} part(s) of {n}, \
                 every result bf16, of the activation's rows and its window's columns",
                w.rows,
                w.width,
                epi.parts()
            ),
        ));
    }
    if act.rows == 0 {
        return Ok(true);
    }
    let m = ctx.opaque_rows(stated(op, act.rows)?);
    #[cfg(feature = "cuda")]
    {
        let planes: Vec<u64> = outs.iter().map(|y| y.ptr).collect();
        cublas::anchored(ctx, op, epi, act.ptr, w.ptr, ins, &planes, m, n, k)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ins, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}

/// `out = epi(act . w^T)` on cuBLASLt's own epilogue alone (a bias, a ReLU,
/// or a bias then a ReLU: [`Epilogue::lt`]), at the first
/// heuristic that takes it: one of the candidates [`anchored`] times, as it
/// runs. `ins` is the bias, where the epilogue reads one. False, landing
/// nothing, where no cuBLASLt epilogue lands `epi`'s or no algorithm takes
/// it.
pub fn anchored_on_lt(
    ctx: &Ctx,
    op: &'static str,
    epi: &'static Epilogue,
    act: Tensor,
    w: Tensor,
    ins: &[In],
    out: Tensor,
) -> Result<bool, Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    dtype_dispatch!(op, w.dtype, { Bf16 => () });
    let n = count(op, "the columns this projection lands", out.width)?;
    let k = count(op, "the contraction this projection walks", act.width)?;
    if out.dtype != Dtype::Bf16
        || w.rows != out.width
        || w.width != act.width
        || out.rows != act.rows
    {
        return Err(crate::jit::refuse(
            op,
            "the weight, the activation and the result are not one dense bf16 product",
        ));
    }
    if act.rows == 0 {
        return Ok(true);
    }
    let m = ctx.opaque_rows(stated(op, act.rows)?);
    #[cfg(feature = "cuda")]
    {
        cublas::on_lt(ctx, op, epi, act.ptr, w.ptr, ins, out.ptr, m, n, k)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (epi, ins, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}

pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

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
    let m = ctx.opaque_rows(stated(op, y.rows)?);

    #[cfg(feature = "cuda")]
    {
        cublas::act_x_wt(ctx, op, act.ptr, w.ptr, y.ptr, m, n, k)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ctx, w, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}
