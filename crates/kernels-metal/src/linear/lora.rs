use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const GROUP: u32 = 256;

const MAX_RANK: u32 = 128;

/// `y += B·(A·x)` for each row routed to an adapter (`routes`, an id a row;
/// below zero: none), the adapter's planes read through `table_a` and
/// `table_b`: one GPU address an id, of a plane of `rank` rows (A, `rank` x
/// `x.width`) or of `y.width` rows (B, `y.width` x `rank`).
pub fn correct(
    ctx: &Ctx<'_>,
    x: Tensor,
    table_a: Tensor,
    table_b: Tensor,
    routes: Tensor,
    y: Tensor,
    rank: u32,
) -> Result<(), Error> {
    const OP: &str = "linear.lora_correct";

    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "lora_correct" });
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 adapter ids");
    debug_assert_eq!(
        table_a.dtype,
        Dtype::I64,
        "an adapter's planes are read through addresses"
    );
    debug_assert_eq!(
        table_b.dtype,
        Dtype::I64,
        "an adapter's planes are read through addresses"
    );
    debug_assert_eq!(y.rows, x.rows, "a correction lands one row per input row");
    debug_assert_eq!(routes.rows, x.rows, "one adapter id per token row");
    debug_assert_eq!(routes.width, 1, "a correction routes one adapter per row");

    let rows = nonzero(OP, "rows", x.rows)?;
    let in_width = nonzero(OP, "the correction's input width", x.width)?;
    let out_width = nonzero(OP, "the correction's output width", y.width)?;
    let rank = nonzero(OP, "the adapter bank's rank", rank)?;
    if table_a.rows != table_b.rows {
        return Err(refuse(
            OP,
            format!(
                "the bank's two tables hold {} and {} adapter ids",
                table_a.rows, table_b.rows
            ),
        ));
    }
    if rank > MAX_RANK {
        return Err(refuse(
            OP,
            format!(
                "the bank's rank is {rank}, above the {MAX_RANK} `linear/lora.metal` \
                 stages the waist in"
            ),
        ));
    }

    ctx.fire(
        Fire::at("linear/lora.metal", entry).apply(Grid::of([GROUP, rows, 1], [GROUP, 1, 1])),
        &[
            x.arg(),
            table_a.arg(),
            table_b.arg(),
            routes.arg(),
            y.arg_mut(),
            stated(OP, in_width)?.arg(),
            stated(OP, out_width)?.arg(),
            stated(OP, rank)?.arg(),
        ],
    )
}
