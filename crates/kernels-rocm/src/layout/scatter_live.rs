use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/scatter_live.hip";

/// Lanes per wavefront on CDNA: a block is a whole number of them.
const WAVE: u32 = 64;

const MAX_BLOCK: u32 = 1024;

pub fn scatter_live_rows(
    ctx: &Ctx,
    src: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.scatter_live_rows";
    let unit = dtype_dispatch!(OP, src.dtype, {
        Bf16 => "::pie::bf16",
        F16 => "::pie::f16",
        F32 => "float"
    });
    if routes.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the destination rows this merge is handed are {:?}, and it reads an i32 row map",
                routes.dtype
            ),
        ));
    }
    if routes.rows != src.rows {
        return Err(refuse(
            OP,
            format!(
                "{} rows to place and {} destinations named",
                src.rows, routes.rows
            ),
        ));
    }
    if y.dtype != src.dtype || y.width != src.width {
        return Err(refuse(
            OP,
            format!(
                "the token rectangle is {} x {:?} and the tower's is {} x {:?}; a row copy \
                 does not reshape",
                y.width, y.dtype, src.width, src.dtype
            ),
        ));
    }
    let rows = nonzero(OP, "rows to place", src.rows)?;
    let units = stated(OP, nonzero(OP, "the placed row's width", src.width)?)?;
    let threads = src
        .width
        .div_ceil(WAVE)
        .max(1)
        .saturating_mul(WAVE)
        .min(MAX_BLOCK);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::layout::scatter_live_rows<{unit}>")),
        )
        .apply(Launch::per_row(rows, threads)),
        &[src.arg(), y.arg(), routes.arg(), units.arg(), ctx.stage()],
    )
}
