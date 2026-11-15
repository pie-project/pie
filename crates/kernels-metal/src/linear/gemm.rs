use crate::Tuning;
use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::tensor::Tensor;

pub const TILE_M: u32 = 32;

pub const TILE_N: u32 = 32;

const TILE_ENTRY: &str = "dense_gemm_t_bfloat16_bm_32_bk_32_bn_32";

const TILE_GROUP: [u32; 3] = [32, 2, 2];

const NARROW_M: u32 = 8;

const NARROW_ENTRY: &str = "dense_gemm_t_bfloat16_bm_8_bk_64_bn_32";

const NARROW_GROUP: [u32; 3] = [32, 1, 2];

const VECTOR_ENTRY: &str = "dense_gemv_t_bfloat16";

const KSPLIT_ENTRY: &str = "dense_gemv_t_ksplit_bfloat16";

const KSPLIT_F32_ENTRY: &str = "dense_gemv_t_ksplit_bfloat16_float32";

const VECTOR_GROUP: u32 = 128;

const KSPLIT_MAX_COLUMNS: u32 = 1024;

const LANES_PER_COLUMN: u32 = 32;

pub const VECTOR_MAX_ROWS: u32 = 4;

const FILE: &str = "linear/gemm.metal";

const LANES_PER_TILE: u32 = 32;

const MPP_FILE: &str = "linear/gemm_mpp.metal";

/// The fewest rows the tensor-op tile lands: its smallest row tile.
const MPP_MIN_ROWS: u32 = 8;

/// The contraction the tensor-op tile walks per multiply.
const MPP_K: u32 = 64;

pub fn matmul(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    act_x_wt(ctx, tuning, "linear.matmul", act, w, y)
}

pub fn lm_head(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    act_x_wt(ctx, tuning, "linear.matmul", act, w, y)
}

/// `act . w^T` for bf16 activations and weights. On a GPU with tensor ops,
/// rows from the tensor-op tile's 8 up take it where the shape fits its
/// tiles; every other GPU and shape takes the simdgroup-matrix kernels.
pub fn act_x_wt(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let (rows, columns, contraction) = extent(op, act, y)?;
    if rows == 0 {
        return Ok(());
    }
    let (entry, grid) = if y.dtype == dtype::Dtype::F32 {
        // f32 columns (`matmul_f32`), a threadgroup each.
        (KSPLIT_F32_ENTRY, ksplit_grid(op, rows, columns)?)
    } else if rows < VECTOR_MAX_ROWS && columns <= KSPLIT_MAX_COLUMNS {
        (KSPLIT_ENTRY, ksplit_grid(op, rows, columns)?)
    } else if rows < VECTOR_MAX_ROWS {
        (VECTOR_ENTRY, vector_grid(op, rows, columns)?)
    } else if let Some((bm, bn)) = mpp_fit(tuning, rows, columns, contraction) {
        return mpp(ctx, op, (act, w, y), (rows, columns, contraction), (bm, bn));
    } else if rows < TILE_M {
        (
            NARROW_ENTRY,
            tile_grid(op, rows, columns, NARROW_M, NARROW_GROUP)?,
        )
    } else {
        (
            TILE_ENTRY,
            tile_grid(op, rows, columns, TILE_M, TILE_GROUP)?,
        )
    };
    ctx.fire(
        Fire::at(FILE, entry).apply(grid),
        &[
            act.arg(),
            w.arg(),
            y.arg_mut(),
            stated(op, rows)?.arg(),
            stated(op, columns)?.arg(),
            stated(op, contraction)?.arg(),
        ],
    )
}

/// The tensor-op tile's rows: one row tile for every row count, so the
/// pipelines a shape takes do not follow a fire's live rows between the
/// rungs; the rows past a tail are masked, and below 32 a narrower row tile
/// saves no weight read.
const MPP_M: u32 = 32;

/// Rows up to which the tensor-op tile is 64 columns wide: a decode-sized
/// matmul reads its weight once whatever its rows, and the narrower column
/// tile fills the cores where `columns` is few 128-wide tiles.
///
/// The pipelines a shape takes change only at 4, 8 and here, and a load's
/// rung walk builds a fire's at each rung's rows: it covers every band while
/// the ladder has a rung in `8..=64` and one past 64, as powers of two from
/// 1 do. A ladder that stops at or below 64 rows is covered as well, for no
/// fire's rows pass it; one that skips `8..=64` while live rows land there
/// is not, and would meet a pipeline the walk did not build.
const MPP_NARROW_ROWS: u32 = 64;

/// The tensor-op tile `rows` rows into `columns` columns over `contraction`
/// take, where the GPU has tensor ops and the shape fits it.
fn mpp_fit(tuning: &Tuning, rows: u32, columns: u32, contraction: u32) -> Option<(u32, u32)> {
    let bn = if rows <= MPP_NARROW_ROWS || !columns.is_multiple_of(128) {
        64
    } else {
        128
    };
    (tuning.tensor_ops()
        && rows >= MPP_MIN_ROWS
        && columns.is_multiple_of(bn)
        && contraction.is_multiple_of(MPP_K))
    .then_some((MPP_M, bn))
}

fn mpp(
    ctx: &Ctx<'_>,
    op: &'static str,
    (act, w, y): (Tensor, Tensor, Tensor),
    (rows, columns, contraction): (u32, u32, u32),
    (bm, bn): (u32, u32),
) -> Result<(), Error> {
    let simdgroups = 4;
    let entry = crate::linear::qmm::symbol(&format!("dense_gemm_mpp_m{bm}_n{bn}_s{simdgroups}"));
    let stamp = crate::linear::qmm::symbol(&format!(
        "PIE_DENSE_MPP_POINT(\"{entry}\",{bm},{bn},{simdgroups})"
    ));
    ctx.fire(
        Fire::at(MPP_FILE, entry).stamp(stamp).apply(Grid::of(
            [32 * simdgroups * (columns / bn), rows.div_ceil(bm), 1],
            [32 * simdgroups, 1, 1],
        )),
        &[
            act.arg(),
            w.arg(),
            y.arg_mut(),
            stated(op, rows)?.arg(),
            stated(op, columns)?.arg(),
            stated(op, contraction)?.arg(),
        ],
    )
}

fn extent(op: &'static str, act: Tensor, y: Tensor) -> Result<(u32, u32, u32), Error> {
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    Ok((y.rows, y.width, act.width))
}

fn tile_grid(
    op: &'static str,
    rows: u32,
    columns: u32,
    block: u32,
    group: [u32; 3],
) -> Result<Grid, Error> {
    let tiles = |extent: u32, tile: u32, per: u32, what: &'static str| -> Result<u32, Error> {
        extent
            .div_ceil(tile)
            .checked_mul(per)
            .ok_or_else(|| refuse(op, format!("{what} will not launch at {extent}")))
    };
    Ok(Grid::of(
        [
            tiles(columns, TILE_N, LANES_PER_TILE, "the column tiles")?,
            tiles(rows, block, group[1], "the row tiles")?,
            group[2],
        ],
        group,
    ))
}

fn ksplit_grid(op: &'static str, rows: u32, columns: u32) -> Result<Grid, Error> {
    let lanes = columns.checked_mul(VECTOR_GROUP).ok_or_else(|| {
        refuse(
            op,
            format!("the {columns} columns, one threadgroup each, will not launch"),
        )
    })?;
    Ok(Grid::of([lanes, rows, 1], [VECTOR_GROUP, 1, 1]))
}

fn vector_grid(op: &'static str, rows: u32, columns: u32) -> Result<Grid, Error> {
    let lanes = columns
        .div_ceil(VECTOR_GROUP / LANES_PER_COLUMN)
        .checked_mul(VECTOR_GROUP)
        .ok_or_else(|| {
            refuse(
                op,
                format!("the {columns} columns, one simdgroup each, will not launch"),
            )
        })?;
    Ok(Grid::of([lanes, rows, 1], [VECTOR_GROUP, 1, 1]))
}
