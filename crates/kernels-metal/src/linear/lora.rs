use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const GROUP: u32 = 256;

const MAX_RANK: u32 = 128;

pub fn correct(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank_a: Tensor,
    bank_b: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.lora_correct";

    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "lora_correct" });
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 adapter ids");
    debug_assert_eq!(
        bank_a.dtype, x.dtype,
        "the adapter bank rides the activation's dtype"
    );
    debug_assert_eq!(
        bank_b.dtype, x.dtype,
        "the adapter bank rides the activation's dtype"
    );
    debug_assert_eq!(y.rows, x.rows, "a correction lands one row per input row");
    debug_assert_eq!(routes.rows, x.rows, "one adapter id per token row");
    debug_assert_eq!(routes.width, 1, "a correction routes one adapter per row");

    let rows = nonzero(OP, "rows", x.rows)?;
    let in_width = nonzero(OP, "the correction's input width", x.width)?;
    let out_width = nonzero(OP, "the correction's output width", y.width)?;
    if !bank_a.width.is_multiple_of(in_width) {
        return Err(refuse(
            OP,
            format!(
                "the down bank is {} wide over an input of {in_width}, which is not a \
                 whole number of ranks",
                bank_a.width
            ),
        ));
    }
    let rank = nonzero(OP, "the adapter bank's rank", bank_a.width / in_width)?;
    if bank_b.width != out_width.saturating_mul(rank) {
        return Err(refuse(
            OP,
            format!(
                "the up bank is {} wide where {out_width} x {rank} is {}; the two \
                 planes of one bank state two ranks",
                bank_b.width,
                out_width.saturating_mul(rank),
            ),
        ));
    }
    if bank_a.rows != bank_b.rows {
        return Err(refuse(
            OP,
            format!(
                "the bank's two planes seat {} and {} adapters",
                bank_a.rows, bank_b.rows
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
            bank_a.arg(),
            bank_b.arg(),
            routes.arg(),
            y.arg_mut(),
            stated(OP, in_width)?.arg(),
            stated(OP, out_width)?.arg(),
            stated(OP, rank)?.arg(),
        ],
    )
}
