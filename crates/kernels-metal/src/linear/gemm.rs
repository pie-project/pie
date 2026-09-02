//! `Gemm`: dense projections against a transposed weight. Dispatches by fire
//! rows to one of three rungs (vector under [`VECTOR_MAX_ROWS`], an 8-row
//! tile under [`TILE_M`], else the 32x32 tile); the two arms round
//! differently, so the arm picked depends on fire composition and small
//! numerical drift across compositions is accepted (not bit-identity).

use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::tensor::Tensor;

pub const TILE_M: u32 = 32;

pub const TILE_N: u32 = 32;

// The wide rung: `TILE_M` rows of output per threadgroup, over the four
// simdgroups `TILE_GROUP` launches.
const TILE_ENTRY: &str = "dense_gemm_t_bfloat16_bm_32_bk_32_bn_32";

const TILE_GROUP: [u32; 3] = [32, 2, 2];

// The narrow rung: the same kernel at the smallest row block the matrix unit
// has, over one simdgroup of rows rather than two.
const NARROW_M: u32 = 8;

const NARROW_ENTRY: &str = "dense_gemm_t_bfloat16_bm_8_bk_64_bn_32";

const NARROW_GROUP: [u32; 3] = [32, 1, 2];

// The floor: one simdgroup per output column, the contraction split across
// its 32 lanes and folded with `simd_sum` — why this rung's bits differ from
// the tiles'.
const VECTOR_ENTRY: &str = "dense_gemv_t_bfloat16";

// The narrow-column arm of the same rung: a projection landing fewer columns
// than `VECTOR_GROUP` has simdgroups gives `VECTOR_ENTRY` one working
// simdgroup and the rest idle, so the whole threadgroup takes one column and
// splits the contraction across its lanes instead (the shared gate's `[1, K]`
// is the case: forty-eight of them a token).
const KSPLIT_ENTRY: &str = "dense_gemv_t_ksplit_bfloat16";

const VECTOR_GROUP: u32 = 128;

// Up to this many columns the K-split arm is the faster one: one threadgroup a
// column gives a `[512, K]` router 512 threadgroups where the simdgroup-a-column
// arm gives it 128, and each lane walks a quarter of the contraction. Past it
// the column count alone fills the device and the fatter arm wins.
const KSPLIT_MAX_COLUMNS: u32 = 1024;

const LANES_PER_COLUMN: u32 = 32;

// The measured fire width at which the 8-row tile overtakes the vector rung.
// Distinct from `crate::tuning`'s quantized-ladder crossover (a different
// kernel over different operands); not a tuned field.
pub const VECTOR_MAX_ROWS: u32 = 4;

const FILE: &str = "linear/gemm_dense.metal";

const LANES_PER_TILE: u32 = 32;

pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

/// `y = act x w^T`. A fire with no rows lands nothing and encodes nothing.
pub fn act_x_wt(
    ctx: &Ctx<'_>,
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
    let (entry, grid) = if rows < VECTOR_MAX_ROWS && columns <= KSPLIT_MAX_COLUMNS {
        (KSPLIT_ENTRY, ksplit_grid(op, rows, columns)?)
    } else if rows < VECTOR_MAX_ROWS {
        (VECTOR_ENTRY, vector_grid(op, rows, columns)?)
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

// The widths are the weight's (fixed by the checkpoint); a zero there is a
// malformed weight and is refused. Rows are the composition's and may
// legitimately be zero, in which case the caller no-ops.
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

// `block` is the rung's `BM`; `group` is the kernel's instantiated shape,
// whose `y` is how many simdgroups share the block's rows.
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

// One whole threadgroup per (row, column): the kernel reads its column off
// `threadgroup_position_in_grid`, so the grid is exactly `columns x rows`
// threadgroups and nothing is over-launched.
fn ksplit_grid(op: &'static str, rows: u32, columns: u32) -> Result<Grid, Error> {
    let lanes = columns.checked_mul(VECTOR_GROUP).ok_or_else(|| {
        refuse(
            op,
            format!("the {columns} columns, one threadgroup each, will not launch"),
        )
    })?;
    Ok(Grid::of([lanes, rows, 1], [VECTOR_GROUP, 1, 1]))
}

// One simdgroup per output column. The kernel guards `n >= N` and `m >= M`,
// so a column count the threadgroup doesn't divide launches over and discards.
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
