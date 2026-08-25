use kernels::plane::Refusal;
use kernels::points::Scalar;

use crate::plane::{Bind, Const, Ctx, Fire, In, Out};
use crate::points::{Payload, at_bf16};

pub const TILE_M: i32 = 32;

pub const TILE_N: i32 = 32;

const FILE: &str = "gemm/dense.wgsl";

const TILE_ENTRY: &str = "dense_gemm_t_bfloat16";

const VECTOR_ENTRY: &str = "dense_gemv_t_bfloat16";

const TILE_LANES: u32 = 32;

const TILE_ROWS_PER_GROUP: u32 = 2;

const TILE_DEPTH: u32 = 2;

const VECTOR_K_LANES: u32 = 32;

#[kernels_macros::claims]
impl kernels::points::Gemm for Ctx<'_> {
    fn matmul<T: Scalar>(
        &self,
        act: In<Payload<T>>,
        w: Const<Payload<T>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        act_x_wt(
            self,
            act,
            w,
            y,
            "`gemm.matmul`, at an element this plane does not stamp",
        )
    }

    fn lm_head<T: Scalar>(
        &self,
        act: In<Payload<T>>,
        w: Const<Payload<T>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        act_x_wt(
            self,
            act,
            w,
            y,
            "`gemm.lm_head`, at an element this plane does not stamp",
        )
    }

    fn attention_landing<T: Scalar>(
        &self,
        act: In<Payload<T>>,
        w: Const<Payload<T>>,
        layer: u32,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        let _ = layer;
        act_x_wt(
            self,
            act,
            w,
            y,
            "`gemm.attention_landing`, at an element this plane does not stamp",
        )
    }
}

fn act_x_wt<T: Scalar>(
    ctx: &Ctx<'_>,
    act: In<Payload<T>>,
    w: Const<Payload<T>>,
    y: Out<Payload<T>>,
    what: &'static str,
) -> Result<(), Refusal> {
    at_bf16::<T>(what)?;
    let (rows, columns, contraction) = extent(act, y)?;
    let (entry, lanes) = if rows < TILE_M {
        (VECTOR_ENTRY, vector_lanes(rows, columns))
    } else {
        (TILE_ENTRY, tile_lanes(rows, columns)?)
    };
    ctx.fire(
        Fire::at(FILE, entry).apply(lanes),
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

fn extent<T: Scalar>(act: In<Payload<T>>, y: Out<Payload<T>>) -> Result<(i32, i32, i32), Refusal> {
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

fn tile_lanes(rows: i32, columns: i32) -> Result<[u32; 3], Refusal> {
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
    Ok([
        tiles(columns, TILE_N, TILE_LANES, "the column tiles")?,
        tiles(rows, TILE_M, TILE_ROWS_PER_GROUP, "the row tiles")?,
        TILE_DEPTH,
    ])
}

fn vector_lanes(rows: i32, columns: i32) -> [u32; 3] {
    [VECTOR_K_LANES, columns.unsigned_abs(), rows.unsigned_abs()]
}
