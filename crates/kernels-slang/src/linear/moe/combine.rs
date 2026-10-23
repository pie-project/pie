//! Mixture-of-experts combines (`linear.moe_weighted_sum`, `moe_bias_sum`,
//! `moe_sigmoid_gate_add`) from `kernels/linear/moe/combine.slang`, a pair
//! of elements a lane.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const COMBINE: &str = "linear/moe/combine.slang";

/// The workgroup the combines are built at.
const ROUTE_GROUP: [u32; 3] = [256, 1, 1];

fn route_rows(op: &'static str, width: u32, rows: u32) -> Result<Grid, Error> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    let elements = u64::from(width) * u64::from(rows);
    let words = u32::try_from(elements.div_ceil(2))
        .map_err(|_| refuse(op, format!("a {rows} x {width} plane will not launch")))?;
    Ok(Grid::of([words, 1, 1], ROUTE_GROUP))
}

pub fn weighted_sum(
    ctx: &Ctx<'_>,
    routed: Tensor,
    weights: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_weighted_sum";
    let entry = dtype_dispatch!(OP, routed.dtype, { Bf16 => "linear_moe_weighted_sum_bf16" });
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` reads f32 route weights");
    nonzero(OP, "the token rows this fold lands on", y.rows)?;
    if !routed.rows.is_multiple_of(y.rows) {
        return Err(refuse(
            OP,
            format!(
                "the routed rectangle's {} rows do not fold into the {} token rows",
                routed.rows, y.rows
            ),
        ));
    }
    let top_k = routed.rows / y.rows;
    debug_assert_eq!(
        routed.width, y.width,
        "the routed row's width, which the fold does not change"
    );
    debug_assert!(
        weights.rows == y.rows && weights.width == top_k,
        "the weight plane is one weight per route"
    );
    let grid = route_rows(OP, y.width, y.rows)?;
    ctx.fire(
        Fire::at(COMBINE, entry).apply(grid),
        &[
            routed.arg(),
            weights.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            stated(OP, y.rows)?.arg(),
            stated(OP, top_k)?.arg(),
        ],
    )
}

pub fn bias_sum(
    ctx: &Ctx<'_>,
    x: Tensor,
    bias: Tensor,
    routes: Tensor,
    weights: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_bias_sum";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "linear_moe_bias_sum_bf16" });
    debug_assert_eq!(
        bias.dtype, x.dtype,
        "the expert bias rides the activation's dtype"
    );
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 routes");
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` reads f32 route weights");
    debug_assert!(
        x.rows == y.rows && x.width == y.width,
        "the token rectangle, which adding a bias does not change"
    );
    debug_assert_eq!(
        bias.width, y.width,
        "the bias bank is one row per expert, the activation's width"
    );
    debug_assert!(
        routes.rows == y.rows && weights.rows == y.rows,
        "a routed plane lands one row per token row"
    );
    debug_assert_eq!(
        routes.width, weights.width,
        "the weight plane is one weight per route"
    );
    let top_k = nonzero(OP, "the routed fan-out", routes.width)?;
    let grid = route_rows(OP, y.width, y.rows)?;
    ctx.fire(
        Fire::at(COMBINE, entry).apply(grid),
        &[
            x.arg(),
            bias.arg(),
            routes.arg(),
            weights.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            stated(OP, y.rows)?.arg(),
            stated(OP, top_k)?.arg(),
        ],
    )
}

pub fn sigmoid_gate_add(
    ctx: &Ctx<'_>,
    routed: Tensor,
    shared: Tensor,
    gate: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_sigmoid_gate_add";
    let entry = dtype_dispatch!(OP, routed.dtype, { Bf16 => "linear_moe_sigmoid_gate_add_bf16" });
    debug_assert!(
        shared.rows == routed.rows && shared.width == routed.width,
        "the shared expert's rectangle is the routed one"
    );
    debug_assert_eq!(
        gate.rows, routed.rows,
        "the gate column is one scalar per row"
    );
    debug_assert_eq!(
        gate.width, 1,
        "`shared_expert_combine` reads `gate[row]`, so the gate is a column"
    );
    let grid = route_rows(OP, routed.width, routed.rows)?;
    ctx.fire(
        Fire::at(COMBINE, entry).apply(grid),
        &[
            routed.arg(),
            shared.arg(),
            gate.arg(),
            y.arg_mut(),
            stated(OP, routed.width)?.arg(),
            stated(OP, routed.rows)?.arg(),
        ],
    )
}
