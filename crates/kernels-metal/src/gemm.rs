use kernels::Grid;
use kernels::plane::Refusal;

use crate::plane::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};

pub const TILE_M: i32 = 32;

pub const TILE_N: i32 = 32;

const TILE_ENTRY: &str = "dense_gemm_t_bfloat16_bm_32_bn_32";

const VECTOR_ENTRY: &str = "dense_gemv_t_bfloat16";

const FILE: &str = "gemm/dense.metal";

const LANES_PER_TILE: u32 = 32;

const TILE_GROUP: [u32; 3] = [32, 2, 2];

const VECTOR_GROUP: u32 = 128;

const LANES_PER_COLUMN: u32 = 32;

pub fn act_x_wt(
    ctx: &Ctx<'_>,
    act: In<Tensor<bf16>>,
    w: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
) -> Result<(), Refusal> {
    let (rows, columns, contraction) = extent(act, y)?;
    let (entry, grid) = if rows < TILE_M {
        (VECTOR_ENTRY, vector_grid(rows, columns)?)
    } else {
        (TILE_ENTRY, tile_grid(rows, columns)?)
    };
    ctx.fire(
        Fire::at(FILE, entry).apply(grid),
        &[
            act.arg(),
            w.arg(),
            y.arg(),
            rows.arg(),
            columns.arg(),
            contraction.arg(),
        ],
    )
}

fn extent(act: In<Tensor<bf16>>, y: Out<Tensor<bf16>>) -> Result<(i32, i32, i32), Refusal> {
    if y.rows <= 0 {
        return Err(Refusal::Empty {
            what: "the rows this projection lands",
        });
    }
    if y.width <= 0 {
        return Err(Refusal::Empty {
            what: "the columns this projection lands",
        });
    }
    if act.width <= 0 {
        return Err(Refusal::Empty {
            what: "the contraction this projection walks",
        });
    }
    if act.rows != y.rows {
        return Err(Refusal::Narrow {
            what: "the activation's rows, which are the rows the result lands",
            at: i64::from(act.rows),
        });
    }
    Ok((y.rows, y.width, act.width))
}

fn tile_grid(rows: i32, columns: i32) -> Result<Grid, Refusal> {
    let tiles = |extent: i32, tile: i32, per: u32, what: &'static str| -> Result<u32, Refusal> {
        extent
            .unsigned_abs()
            .div_ceil(tile.unsigned_abs())
            .checked_mul(per)
            .ok_or(Refusal::Grid {
                what,
                at: i64::from(extent),
            })
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

fn vector_grid(rows: i32, columns: i32) -> Result<Grid, Refusal> {
    let lanes = columns
        .unsigned_abs()
        .div_ceil(VECTOR_GROUP / LANES_PER_COLUMN)
        .checked_mul(VECTOR_GROUP)
        .ok_or(Refusal::Grid {
            what: "the columns this projection lands, one simdgroup each",
            at: i64::from(columns),
        })?;
    Ok(Grid::of(
        [lanes, rows.unsigned_abs(), 1],
        [VECTOR_GROUP, 1, 1],
    ))
}
