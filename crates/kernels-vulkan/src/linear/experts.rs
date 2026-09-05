use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "moe/expert_gather.slang";
const GROUP: u32 = 256;

pub fn gather(
    ctx: &Ctx<'_>,
    plane: Tensor,
    words: u32,
    routes: Tensor,
    seats: Tensor,
    seat_routes: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.expert_gather";
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 routes");
    debug_assert_eq!(
        seat_routes.dtype,
        Dtype::I32,
        "`{OP}` writes i32 seat routes"
    );
    nonzero(OP, "words per expert", words)?;
    let count = routes
        .rows
        .checked_mul(routes.width)
        .ok_or_else(|| refuse(OP, "the route run will not launch"))?;
    nonzero(OP, "seats", count)?;
    ctx.fire(
        Fire::at(FILE, "expert_gather")
            .groups([words.div_ceil(GROUP), count, 1])
            .group([GROUP, 1, 1]),
        &[
            plane.arg(),
            seats.arg_mut(),
            routes.arg(),
            seat_routes.arg_mut(),
            stated(OP, words)?.arg(),
            stated(OP, count)?.arg(),
        ],
    )
}
