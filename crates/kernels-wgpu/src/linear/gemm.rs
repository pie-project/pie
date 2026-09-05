#![allow(unused_variables)]
#![allow(clippy::too_many_arguments)]

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

pub const TILE_M: u32 = 64;

pub const TILE_N: u32 = 64;

const NARROW_MAX_ROWS: u32 = 48;

const NARROW_TILE_M: u32 = 16;

const NARROW_TILE_ENTRY: &str = "dense_gemm_t16_bf16";

pub const VECTOR_MAX_ROWS: u32 = 4;

const FILE: &str = "gemm/dense.wgsl";

const TILE_ENTRY: &str = "dense_gemm_t_bf16";

const TILE_GROUP: [u32; 3] = [16, 16, 1];

const VECTOR_ENTRY: &str = "dense_gemv_t_bf16";

const VECTOR_GROUP: [u32; 3] = [32, 8, 1];

pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

pub fn act_x_wt(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let (rows, columns, contraction) = extent(op, act, w, y)?;
    if rows == 0 {
        return Ok(());
    }
    let (entry, grid) = if rows < VECTOR_MAX_ROWS {
        (VECTOR_ENTRY, vector_grid(op, rows, columns)?)
    } else if rows < NARROW_MAX_ROWS {
        (
            NARROW_TILE_ENTRY,
            tile_grid(op, rows, columns, NARROW_TILE_M)?,
        )
    } else {
        (TILE_ENTRY, tile_grid(op, rows, columns, TILE_M)?)
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

fn extent(op: &'static str, act: Tensor, w: Tensor, y: Tensor) -> Result<(u32, u32, u32), Error> {
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    if !y.width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "the projection lands {} columns, odd: every output pair is one bf16 word",
                y.width
            ),
        ));
    }
    if w.width != act.width || w.rows != y.width {
        return Err(refuse(
            op,
            format!(
                "the weight is {} x {} and the projection is {} in, {} out",
                w.rows, w.width, act.width, y.width
            ),
        ));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    Ok((y.rows, y.width, act.width))
}

fn vector_grid(op: &'static str, rows: u32, columns: u32) -> Result<Grid, Error> {
    Ok(Grid::of([VECTOR_GROUP[0], columns, rows], VECTOR_GROUP))
}

fn tile_grid(op: &'static str, rows: u32, columns: u32, tile_m: u32) -> Result<Grid, Error> {
    let lanes = |extent: u32, tile: u32, per: u32, what: &'static str| -> Result<u32, Error> {
        extent
            .div_ceil(tile)
            .checked_mul(per)
            .ok_or_else(|| refuse(op, format!("{what} will not launch at {extent}")))
    };
    Ok(Grid::of(
        [
            lanes(columns, TILE_N, TILE_GROUP[0], "the column tiles")?,
            lanes(rows, tile_m, TILE_GROUP[1], "the row tiles")?,
            TILE_GROUP[2],
        ],
        TILE_GROUP,
    ))
}
