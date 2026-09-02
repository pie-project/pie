//! `MoeRoute`: the router that reads no logits. One entry,
//! `linear.moe_hash_route`, kept apart from the ranked routers in
//! [`super::moe`] — it shares nothing with them but the shape of the answer.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "linear/moe_route.cuh";

/// One thread per (token row, slot); the gather is two loads wide and wants
/// the lanes, not the block shape.
const BLOCK: u32 = 256;

/// Routes by lookup, not by a gate: `tid2eid` is `[vocab, top_k]` i64,
/// naming the `top_k` experts each token id routes to at uniform weight
/// `1/top_k`. Lands the same `routes` i32 / `weights` f32 pair a ranked gate
/// in [`super::moe`] would.
///
/// The i64 table narrows to i32 routes in the gather, since an expert count
/// never approaches `2^31`. An out-of-range token id falls to table row 0,
/// matching [`crate::layout::embed`]'s gather.
///
/// # Errors
///
/// A refusal for a zero fan-out, a zero vocabulary, an empty row count, and
/// for a `tokens x top_k` lane count that does not fit a 32-bit grid.
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
    // The weights are the gate's sqrt-softplus scores at the table's picks
    // (the official `Gate.forward`), so the router logits are read too.
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
    // Restated here: this file shares no validator with the ranked router.
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
    // One thread per token row: the row's weights normalize together.
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
            // Live-rows word when a body replay armed one, else `ABSENT`.
            ctx.stage(),
        ],
    )
}

/// **THE STATIC ROUTES OF A GROUPED PROJECTION**: `routes[n, g] = g`.
pub fn group_routes(ctx: &Ctx, groups: u32, routes: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.group_routes";
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` lands i32 routes");
    debug_assert_eq!(routes.width, groups, "the routes are one slot per group");
    let rows = nonzero(OP, "rows", routes.rows)?;
    let groups = nonzero(OP, "the group count", groups)?;
    let lanes = rows.checked_mul(groups).ok_or_else(|| {
        refuse(OP, format!("the routes will not launch: {rows} tokens x {groups} groups"))
    })?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::group_routes").apply(Launch::flat(lanes, BLOCK)),
        &[
            routes.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, groups)?.arg(),
            // Live-rows word when a body replay armed one, else `ABSENT`.
            ctx.stage(),
        ],
    )
}
