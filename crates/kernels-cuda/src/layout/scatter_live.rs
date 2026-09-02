//! `ScatterLive`: the embed merge for a rectangle whose tail has no
//! destination. A separate entry rather than a guard inside
//! `layout::scatter_rows`, so that op's existing contract (every route names
//! a row) is not widened underneath its consumers.
//!
//! A compacting fold answers `rows / side²` rows and leaves the rest of the
//! patch rectangle as whatever the arena held; those tail rows have route
//! entries with no legal destination, hence the negative-route sentinel here.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/scatter_live.cuh";

const WARP: u32 = 32;

const MAX_BLOCK: u32 = 1024;

/// The embed merge, with a drop sentinel: row `i` of `src` lands at token row
/// `routes[i]` of `y`; any negative `routes[i]` places it nowhere (`-1` is
/// the value a submission writes, but any negative is dropped).
///
/// The upper bound is still not this kernel's to check: the fire path
/// validates the vector against the token row count before the launch, as
/// for the unguarded twin.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16, f16 and f32; a refusal
/// for a route vector that is not `i32`, one whose length is not `src.rows`,
/// a width or element mismatch between the two rectangles, and an empty
/// source.
pub fn scatter_live_rows(
    ctx: &Ctx,
    src: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_live_rows";
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
        .div_ceil(WARP)
        .max(1)
        .saturating_mul(WARP)
        .min(MAX_BLOCK);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::layout::scatter_live_rows<{unit}>")),
        )
        .apply(Launch::per_row(rows, threads)),
        &[
            src.arg(),
            y.arg(),
            routes.arg(),
            units.arg(),
            // Live-rows word when a body replay armed one, else `ABSENT`.
            ctx.stage(),
        ],
    )
}
