//! `EmbedWeighted`: the interpolating gather for position embeddings. The
//! table is resampled per image grid, which an import cannot precompute, so
//! it is read at fire time; on the native grid, `layout.embed` is used
//! unchanged.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/embed_weighted.cuh";

const WARP: u32 = 32;

const MAX_BLOCK: u32 = 1024;

/// **`y[r] = Σₜ weights[r, t] · table[ids[r, t]]`.**
///
/// `ids` is `[rows, taps]` `i32` and `weights` is `[rows, taps]` `f32`;
/// `taps` is read off their width (4 for bilinear, 16 for bicubic). `vocab`
/// is the table's row count.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for a table that is not bf16 or f16; a refusal
/// for an `ids` that is not `i32` or a `weights` that is not `f32`, for the
/// two geometry rectangles disagreeing with each other or with `y`, for a
/// zero tap count, and for an empty output.
pub fn embed_weighted(
    ctx: &Ctx,
    ids: Tensor,
    weights: Tensor,
    table: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_weighted";
    let t = dtype_dispatch!(OP, table.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(y.dtype, table.dtype, "`{OP}` gathers into the table's element");

    if ids.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!("the taps this gather is handed are {:?}, and it reads i32 rows", ids.dtype),
        ));
    }
    if weights.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the interpolation weights are {:?}, and this gather reads f32 — they are the \
                 preprocessor's arithmetic and not the activation's",
                weights.dtype
            ),
        ));
    }
    if ids.rows != weights.rows || ids.width != weights.width {
        return Err(refuse(
            OP,
            format!(
                "{} x {} taps and {} x {} weights; every tap is weighted and every weight taps",
                ids.rows, ids.width, weights.rows, weights.width
            ),
        ));
    }
    if ids.rows != y.rows {
        return Err(refuse(
            OP,
            format!(
                "{} rows of taps land {} rows; a gather answers one row per index row",
                ids.rows, y.rows
            ),
        ));
    }
    let taps = stated(OP, nonzero(OP, "the taps per row", ids.width)?)?;
    let vocab = stated(OP, nonzero(OP, "the table's row count", vocab)?)?;
    let hidden = stated(OP, nonzero(OP, "the gathered row's width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    let threads = y.width.div_ceil(WARP).max(1).saturating_mul(WARP).min(MAX_BLOCK);
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::layout::embed_weighted<{t}>")))
            .apply(Launch::per_row(rows, threads)),
        &[
            ids.arg(),
            weights.arg(),
            table.arg(),
            y.arg(),
            hidden.arg(),
            vocab.arg(),
            taps.arg(),
            // staged-geometry seat: live-rows word, or the null seat (`ABSENT`).
            ctx.stage(),
        ],
    )
}
