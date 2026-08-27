//! `Gemm`: dense projections against a transposed weight. The gemv-vs-tile
//! choice lives here, inside the entry (decision #13) — a dispatch arm never
//! sees it.

use kernels::KernelError;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::tensor::Tensor;

pub const TILE_M: u32 = 32;

pub const TILE_N: u32 = 32;

const TILE_ENTRY: &str = "dense_gemm_t_bfloat16_bm_32_bn_32";

const VECTOR_ENTRY: &str = "dense_gemv_t_bfloat16";

const FILE: &str = "linear/gemm_dense.metal";

const LANES_PER_TILE: u32 = 32;

const TILE_GROUP: [u32; 3] = [32, 2, 2];

const VECTOR_GROUP: u32 = 128;

const LANES_PER_COLUMN: u32 = 32;

pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), KernelError> {
    act_x_wt(ctx, "gemm.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), KernelError> {
    act_x_wt(ctx, "gemm.lm_head", act, w, y)
}

/// `y = act x w^T`. Skinny fires (fewer than [`TILE_M`] rows — the decode
/// path) take the simdgroup gemv; everything else takes the 32x32 tile.
pub fn act_x_wt(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), KernelError> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let (rows, columns, contraction) = extent(op, act, y)?;
    let (entry, grid) = if rows < TILE_M {
        (VECTOR_ENTRY, vector_grid(op, rows, columns)?)
    } else {
        (TILE_ENTRY, tile_grid(op, rows, columns)?)
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

fn extent(op: &'static str, act: Tensor, y: Tensor) -> Result<(u32, u32, u32), KernelError> {
    if y.rows == 0 {
        return Err(refuse(op, "the rows this projection lands are zero"));
    }
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

fn tile_grid(op: &'static str, rows: u32, columns: u32) -> Result<Grid, KernelError> {
    let tiles = |extent: u32, tile: u32, per: u32, what: &'static str| -> Result<u32, KernelError> {
        extent
            .div_ceil(tile)
            .checked_mul(per)
            .ok_or_else(|| refuse(op, format!("{what} will not launch at {extent}")))
    };
    Ok(Grid::of(
        [
            tiles(columns, TILE_N, LANES_PER_TILE, "the column tiles")?,
            tiles(rows, TILE_M, TILE_GROUP[1], "the row tiles")?,
            TILE_GROUP[2],
        ],
        TILE_GROUP,
    ))
}

fn vector_grid(op: &'static str, rows: u32, columns: u32) -> Result<Grid, KernelError> {
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
