//! Host-only: the dense products on the vendor's libraries. `cublas` runs a
//! bf16 product on a streaming skeleton where one covers its few rows, else
//! on cuBLASLt's first heuristic algorithm for its shape; `cublas_lt_fp8` is
//! cuBLASLt's fp8 product over `act_quant`'s scales.

use crate::error::Error;

use crate::jit::{Ctx, count, dtype_dispatch, stated};
use crate::linear::skeleton::{Epilogue, In, Paging};
use crate::tensor::Tensor;

use dtype::Dtype;

#[cfg(feature = "cuda")]
pub(crate) mod cublas;
#[cfg(feature = "cuda")]
pub use cublas::{Tile, skinny_rows, tile, tile_rows};
#[cfg(feature = "cuda")]
pub mod cublas_lt_fp8;

/// The workgroups (blocks) of the few-rows kernel serving runs a plain bf16
/// product of `m x n x k` on (the rows skeleton, `m` up to its
/// [`crate::linear::skeleton::ROWS_M`]) one multiprocessor of the current
/// device holds at once, at its launch's threads: the driver's occupancy
/// calculator (`cuOccupancyMaxActiveBlocksPerMultiprocessor`) over the
/// kernel's registers and shared memory against the multiprocessor's.
pub fn few_rows_resident_blocks(ctx: &Ctx, m: u32, n: u32, k: u32) -> Result<u32, Error> {
    const OP: &str = "linear.few_rows_occupancy";
    let _ = ctx;
    if !(1..=FEW_ROWS).contains(&m) {
        return Err(crate::jit::refuse(
            OP,
            format!("the few-rows kernel takes 1..={FEW_ROWS} rows, not {m}"),
        ));
    }
    #[cfg(feature = "cuda")]
    {
        crate::linear::gemv::rows_resident_blocks(
            count(OP, "the rows", m)?,
            count(OP, "the columns", n)?,
            count(OP, "the contraction", k)?,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (n, k);
        Err(crate::jit::runtimeless(OP))
    }
}

/// The blocks (each cluster's blocks counted) of the few-rows kernel a
/// dense bf16 product of `m x n x k` runs on, where it does: the rows
/// skeleton's grid from 2 to [`FEW_ROWS`] rows (one row runs the gemv,
/// more cuBLASLt's algorithm). `None` elsewhere.
#[cfg(feature = "cuda")]
#[must_use]
pub fn few_rows_grid(m: u32, n: u32, k: u32) -> Option<u32> {
    let (n, k) = (i32::try_from(n).ok()?, i32::try_from(k).ok()?);
    ((2..=FEW_ROWS).contains(&m) && n > 0 && k > 0 && k % 8 == 0)
        .then(|| crate::linear::gemv::rows_blocks(n, k) * crate::linear::gemv::rows_cluster(n, k))
}

/// The waves cuBLASLt's first heuristic algorithm fills on the current
/// device for a dense bf16 product of `m x n x k` serving runs on it (past
/// the few-rows kernel's rows, and short of `kernel.gemm.bf16_split_from`,
/// where cuBLAS's own algorithm runs): its `wavesCount`, its tiles (and
/// split-K slices) over the blocks of its kernel the device holds at once.
/// Short of one such wave, the waves its tiles fill of the multiprocessors
/// (at most one): a block alone on a multiprocessor runs it near its rate
/// (L40S, width 2048 at 724 rows: 96 tiles of a kernel two a
/// multiprocessor, 0.34 of `wavesCount`'s wave, ran 152 TFLOP/s against
/// 227 at 2048 rows), so the device is short only the multiprocessors it
/// leaves idle. `None` where serving runs another kernel or cuBLASLt states
/// none.
#[cfg(feature = "cuda")]
#[must_use]
pub fn waves(tuning: &crate::tuning::Tuning, m: u32, n: u32, k: u32) -> Option<f64> {
    if m <= FEW_ROWS || (tuning.gemm_bf16_split_from > 0 && m >= tuning.gemm_bf16_split_from) {
        return None;
    }
    let t = tile(tuning, m, n, k)?;
    let stated = f64::from(t.waves);
    if !(stated.is_finite() && stated > 0.0) {
        return None;
    }
    if stated >= 1.0 {
        return Some(stated);
    }
    let tiles =
        f64::from(m.div_ceil(t.rows)) * f64::from(n.div_ceil(t.cols)) * f64::from(t.split_k);
    let sms = f64::from(multiprocessors()?);
    Some((tiles / sms).min(1.0))
}

/// The current device's multiprocessors, where it answers.
#[cfg(feature = "cuda")]
#[must_use]
pub fn multiprocessors() -> Option<u32> {
    crate::jit::device::multiprocessors()
}

/// The most rows CUDA's few-rows kernel (the rows skeleton) takes.
pub const FEW_ROWS: u32 = crate::linear::skeleton::ROWS_M;

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
/// says which do), which the fold never forms.
#[allow(clippy::too_many_arguments)]
pub fn anchored(
    ctx: &Ctx,
    op: &'static str,
    epi: &'static Epilogue,
    act: Tensor,
    w: Tensor,
    ins: &[In],
    outs: &[Tensor],
    paging: Option<Paging>,
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
    // A paged result's plane is its cache's: rows its pages hold, its
    // elements the cache's.
    let shaped = (0..outs.len())
        .zip(outs)
        .all(|(at, y)| match epi.paged(at) {
            None => {
                y.dtype == Dtype::Bf16
                    && y.width == epi.window(at, tile).width
                    && y.rows == act.rows
            }
            Some(paged) => {
                let element = match paged.dtype {
                    crate::linear::skeleton::CacheDtype::Bf16 => Dtype::Bf16,
                    crate::linear::skeleton::CacheDtype::E4m3 { .. } => Dtype::E4m3,
                };
                y.dtype == element && y.width == epi.window(at, tile).width && paging.is_some()
            }
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
        cublas::anchored(ctx, op, epi, act.ptr, w.ptr, ins, &planes, paging, m, n, k)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ins, paging, m, n, k);
        Err(crate::jit::runtimeless(op))
    }
}

/// `outs = epi(y)`: an anchored region's epilogue over a product its own op
/// landed (`y`, bf16, `parts · n` wide), on the row skeleton (the region's
/// unfused leg, every piece of the row), `ins` its inputs and `outs` its
/// results (a paged one's its cache's plane, `paging` where its rows
/// live). False, landing nothing, where the row skeleton does not carry the
/// epilogue.
pub fn epilogue(
    ctx: &Ctx,
    op: &'static str,
    epi: &'static Epilogue,
    y: Tensor,
    ins: &[In],
    outs: &[Tensor],
    paging: Option<Paging>,
) -> Result<bool, Error> {
    dtype_dispatch!(op, y.dtype, { Bf16 => () });
    let parts = epi.parts().max(1);
    let tile = y.width / parts;
    if y.width != tile * parts || !epi.rounds() {
        return Err(crate::jit::refuse(
            op,
            "an epilogue over a landed product reads its bf16 parts as stored",
        ));
    }
    if epi.pages() && paging.is_none() {
        return Err(crate::jit::refuse(
            op,
            "a paged result fired with no paging",
        ));
    }
    if y.rows == 0 {
        return Ok(true);
    }
    #[cfg(feature = "cuda")]
    {
        let Some(leg) = crate::linear::skeleton::RowLeg::of(epi, tile) else {
            return Ok(false);
        };
        let m = ctx.opaque_rows(stated(op, y.rows)?);
        let planes: Vec<u64> = outs.iter().map(|t| t.ptr).collect();
        leg.fire(
            epi,
            ctx,
            y.ptr,
            ins,
            &planes,
            paging,
            m.unsigned_abs(),
            tile,
        )?;
        Ok(true)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ctx, ins, outs, paging);
        Err(crate::jit::runtimeless(op))
    }
}

/// `out = epi(act . w^T)` on cuBLASLt's own epilogue alone (a bias, a ReLU,
/// or a bias then a ReLU: [`Epilogue::lt`]), at cuBLASLt's first heuristic
/// algorithm for the product with it, as [`anchored`] runs a region no
/// streaming skeleton covers. `ins` is the bias, where the epilogue reads
/// one. False, landing nothing, where no cuBLASLt epilogue lands `epi`'s or
/// cuBLASLt offers no algorithm for it.
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
