//! Mixture-of-experts routing (`linear.moe_topk_*`, `moe_predict_route`,
//! `moe_hash_route`, `group_routes`) from `kernels/linear/moe/route.slang`:
//! each token row's experts and their weights.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const ROUTE: &str = "linear/moe/route.slang";

/// The workgroup the routers are built at.
const ROUTE_GROUP: [u32; 3] = [256, 1, 1];

const ROUTER_MAX_EXPERTS: u32 = 1024;

const ROUTER_MAX_TOP_K: u32 = 16;

const SOFTMAX_OVER_SELECTED: u32 = 0;

fn router_grid(op: &'static str, rows: u32) -> Result<Grid, Error> {
    nonzero(op, "rows", rows)?;
    Ok(Grid::of([ROUTE_GROUP[0], rows, 1], ROUTE_GROUP))
}

fn ranked_planes(op: &'static str, logits: Tensor, top_k: u32, routes: Tensor, weights: Tensor) {
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{op}` lands i32 routes");
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{op}` lands f32 route weights");
    debug_assert!(
        routes.rows == logits.rows && weights.rows == logits.rows,
        "a routed result lands one row per token row"
    );
    debug_assert!(
        routes.width == top_k && weights.width == top_k,
        "a routed result is the fan-out the statement states"
    );
}

fn ranked(
    op: &'static str,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<Grid, Error> {
    debug_assert_eq!(
        logits.width, experts,
        "the router's row is the expert count the statement states"
    );
    ranked_planes(op, logits, top_k, routes, weights);
    nonzero(op, "the expert count this router states", experts)?;
    nonzero(op, "the fan-out this router states", top_k)?;
    if experts > ROUTER_MAX_EXPERTS {
        return Err(refuse(
            op,
            format!(
                "the expert count is {experts}, above the {ROUTER_MAX_EXPERTS} this router \
                 stages in shared memory"
            ),
        ));
    }
    if top_k > ROUTER_MAX_TOP_K {
        return Err(refuse(
            op,
            format!(
                "the fan-out is {top_k}, above the {ROUTER_MAX_TOP_K} this router stages in \
                 a shared array"
            ),
        ));
    }
    router_grid(op, logits.rows)
}

pub fn topk_softmax(
    ctx: &Ctx<'_>,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    let entry = dtype_dispatch!(OP, logits.dtype, { Bf16 => "linear_moe_route_topk_softmax_bf16" });
    let grid = ranked(OP, logits, experts, top_k, routes, weights)?;
    ctx.fire(
        Fire::at(ROUTE, entry).apply(grid),
        &[
            logits.arg(),
            routes.arg_mut(),
            weights.arg_mut(),
            experts.arg(),
            top_k.arg(),
            SOFTMAX_OVER_SELECTED.arg(),
            experts.arg(),
        ],
    )
}

pub fn topk_softmax_scaled(
    ctx: &Ctx<'_>,
    logits: Tensor,
    scale: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    let entry =
        dtype_dispatch!(OP, logits.dtype, { Bf16 => "linear_moe_route_topk_softmax_scaled_bf16" });
    debug_assert_eq!(
        scale.dtype, logits.dtype,
        "the gain is read at the router's own width"
    );
    debug_assert_eq!(
        scale.rows * scale.width,
        experts,
        "the gain is indexed by expert, so it holds one entry per expert"
    );
    let grid = ranked(OP, logits, experts, top_k, routes, weights)?;
    ctx.fire(
        Fire::at(ROUTE, entry).apply(grid),
        &[
            logits.arg(),
            routes.arg_mut(),
            weights.arg_mut(),
            scale.arg(),
            experts.arg(),
            top_k.arg(),
            SOFTMAX_OVER_SELECTED.arg(),
            experts.arg(),
        ],
    )
}

pub fn topk_sigmoid(
    ctx: &Ctx<'_>,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    let entry = dtype_dispatch!(OP, logits.dtype, { Bf16 => "linear_moe_route_topk_sigmoid_bf16" });
    let grid = ranked(OP, logits, experts, top_k, routes, weights)?;
    ctx.fire(
        Fire::at(ROUTE, entry).apply(grid),
        &[
            logits.arg(),
            routes.arg_mut(),
            weights.arg_mut(),
            experts.arg(),
            top_k.arg(),
            u32::from(renormalize).arg(),
            scaling.arg(),
        ],
    )
}

fn biased_router(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    debug_assert_eq!(
        bias.dtype,
        Dtype::F32,
        "`{op}` reads an f32 correction bias"
    );
    let grid = ranked(op, logits, experts, top_k, routes, weights)?;
    ctx.fire(
        Fire::at(ROUTE, entry).apply(grid),
        &[
            logits.arg(),
            bias.arg(),
            routes.arg_mut(),
            weights.arg_mut(),
            experts.arg(),
            top_k.arg(),
            u32::from(renormalize).arg(),
            scaling.arg(),
        ],
    )
}

pub fn topk_sigmoid_biased(
    ctx: &Ctx<'_>,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    dtype_dispatch!(OP, logits.dtype, { Bf16 => () });
    biased_router(
        ctx,
        OP,
        "linear_moe_route_topk_sigmoid_biased_bf16",
        logits,
        bias,
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

pub fn topk_sqrt_softplus(
    ctx: &Ctx<'_>,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    dtype_dispatch!(OP, logits.dtype, { Bf16 => () });
    biased_router(
        ctx,
        OP,
        "linear_moe_route_topk_sqrt_softplus_bf16",
        logits,
        bias,
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

pub fn predict_route(
    ctx: &Ctx<'_>,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    dtype_dispatch!(OP, logits.dtype, { Bf16 => () });
    biased_router(
        ctx,
        OP,
        "linear_moe_route_topk_sqrt_softplus_bf16",
        logits,
        bias,
        experts,
        top_k,
        false,
        1.0,
        routes,
        weights,
    )
}

pub fn hash_route(
    ctx: &Ctx<'_>,
    ids: Tensor,
    tid2eid: Tensor,
    logits: Tensor,
    vocab: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    debug_assert!(
        matches!(ids.dtype, Dtype::U32 | Dtype::I32),
        "`{OP}` gathers by a 32-bit token id column"
    );
    dtype_dispatch!(OP, logits.dtype, { Bf16 => () });
    debug_assert_eq!(
        logits.rows, routes.rows,
        "the router logits are one row per token row"
    );
    debug_assert_eq!(tid2eid.dtype, Dtype::I64, "`{OP}` reads the i64 hash table");
    debug_assert_eq!(
        tid2eid.width, top_k,
        "the hash table names `top_k` experts per token id"
    );
    ranked_planes(OP, routes, top_k, routes, weights);
    debug_assert_eq!(
        ids.rows, routes.rows,
        "the token ids handed over are the rows this route lands"
    );
    let top_k = nonzero(OP, "the fan-out this router states", top_k)?;
    nonzero(OP, "the vocabulary this table spans", vocab)?;
    let experts = nonzero(OP, "the expert count the logits span", logits.width)?;
    let rows = nonzero(OP, "rows", routes.rows)?;
    ctx.fire(
        Fire::at(ROUTE, "linear_moe_route_hash_bf16").apply(Grid::of([rows, 1, 1], ROUTE_GROUP)),
        &[
            ids.arg(),
            tid2eid.arg(),
            logits.arg(),
            routes.arg_mut(),
            weights.arg_mut(),
            vocab.arg(),
            experts.arg(),
            top_k.arg(),
            u32::from(renormalize).arg(),
            scaling.arg(),
            rows.arg(),
        ],
    )
}

pub fn group_routes(ctx: &Ctx<'_>, groups: u32, routes: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.group_routes";
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` lands i32 routes");
    debug_assert_eq!(routes.width, groups, "the routes are one slot per group");
    let groups = nonzero(OP, "the group count", groups)?;
    ctx.fire(
        Fire::at(ROUTE, "linear_moe_route_group_routes")
            .apply(Grid::of([groups, routes.rows, 1], ROUTE_GROUP)),
        &[routes.arg_mut(), groups.arg()],
    )
}
