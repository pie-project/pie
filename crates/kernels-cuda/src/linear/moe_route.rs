use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "linear/moe_route.cuh";

const BLOCK: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn hash_route(
    ctx: &Ctx,
    ids: Tensor,
    tid2eid: Tensor,
    logits: Tensor,
    vocab: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_hash_route";
    debug_assert_eq!(logits.dtype, Dtype::Bf16, "`{OP}` reads bf16 router logits");
    debug_assert_eq!(
        logits.rows, routes.rows,
        "the router logits are one row per token row"
    );
    debug_assert!(
        matches!(ids.dtype, Dtype::I32 | Dtype::U32),
        "`{OP}` gathers by a 32-bit token id column"
    );
    debug_assert_eq!(tid2eid.dtype, Dtype::I64, "`{OP}` reads the i64 hash table");
    debug_assert_eq!(
        tid2eid.width, top_k,
        "the hash table names `top_k` experts per token id"
    );
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` lands i32 routes");
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` lands f32 route weights");
    debug_assert!(
        routes.width == top_k && weights.width == top_k,
        "a routed result is the fan-out the statement states"
    );
    debug_assert_eq!(
        routes.rows, weights.rows,
        "a routed result lands one row per token row"
    );
    debug_assert_eq!(
        ids.rows, routes.rows,
        "the token ids handed over are the rows this route lands"
    );

    let rows = nonzero(OP, "rows", routes.rows)?;
    let top_k = nonzero(OP, "the fan-out this router states", top_k)?;
    let vocab = nonzero(OP, "the vocabulary this table spans", vocab)?;
    let experts = nonzero(OP, "the expert count the logits span", logits.width)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::hash_route_gather").apply(Launch::flat(rows, BLOCK)),
        &[
            ids.arg(),
            tid2eid.arg(),
            logits.arg(),
            routes.arg(),
            weights.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, vocab)?.arg(),
            stated(OP, experts)?.arg(),
            stated(OP, top_k)?.arg(),
            i32::from(renormalize).arg(),
            scaling.arg(),
            ctx.stage(),
        ],
    )
}

pub fn group_routes(ctx: &Ctx, groups: u32, routes: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.group_routes";
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` lands i32 routes");
    debug_assert_eq!(routes.width, groups, "the routes are one slot per group");
    let rows = nonzero(OP, "rows", routes.rows)?;
    let groups = nonzero(OP, "the group count", groups)?;
    let lanes = rows.checked_mul(groups).ok_or_else(|| {
        refuse(
            OP,
            format!("the routes will not launch: {rows} tokens x {groups} groups"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::group_routes").apply(Launch::flat(lanes, BLOCK)),
        &[
            routes.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, groups)?.arg(),
            ctx.stage(),
        ],
    )
}
