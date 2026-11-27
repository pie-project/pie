//! The routers: each token row's top-k experts and their weights.

#[allow(clippy::wildcard_imports)]
use super::*;

/// A router is one wavefront a token row, each lane landing one pick.
pub(super) const ROUTER_LANES: u32 = 64;

pub(super) const fn router_lane(rows: u32) -> Launch {
    Launch::per_row(rows, ROUTER_LANES)
}

pub(super) fn router_fan(op: &'static str, fan: u32) -> Result<(), Error> {
    if fan > ROUTER_LANES {
        return Err(refuse(
            op,
            format!("the fan-out is {fan}, above the {ROUTER_LANES} picks one wavefront lands"),
        ));
    }
    Ok(())
}

pub(super) fn elementwise_rows(op: &'static str, rows: u32, width: u32) -> Result<Launch, Error> {
    nonzero(op, "rows", rows)?;
    nonzero(op, "width", width)?;
    Ok(Launch::grid(
        [rows, width.div_ceil(BLOCK), 1],
        [BLOCK, 1, 1],
    ))
}

pub(super) fn ranked_planes(
    op: &'static str,
    logits: Tensor,
    top_k: u32,
    routes: &Tensor,
    weights: &Tensor,
) {
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

pub(super) fn router_extents(
    op: &'static str,
    logits: Tensor,
    experts: u32,
    top_k: u32,
) -> Result<(i32, i32), Error> {
    debug_assert_eq!(
        logits.width, experts,
        "the router's row is the expert count the statement states"
    );
    nonzero(op, "rows", logits.rows)?;
    nonzero(op, "the fan-out this router states", top_k)?;
    router_fan(op, top_k)?;
    if experts > MAX_EXPERTS {
        return Err(refuse(
            op,
            format!(
                "the expert count is {experts}, above the {MAX_EXPERTS} scores this router \
                 stages in shared memory"
            ),
        ));
    }
    Ok((stated(op, experts)?, stated(op, top_k)?))
}

pub fn topk_softmax(
    ctx: &Ctx,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ranked_planes(OP, logits, top_k, routes, weights);
    let (e, k) = router_extents(OP, logits, experts, top_k)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::moe_topk_softmax<{t}>"),
        ))
        .apply(router_lane(logits.rows)),
        &[
            logits.arg(),
            ArgValue::ABSENT,
            ArgValue::ABSENT,
            routes.arg(),
            weights.arg(),
            e.arg(),
            k.arg(),
            0_i32.arg(),
            ctx.stage(),
        ],
    )
}

pub fn topk_softmax_scaled(
    ctx: &Ctx,
    logits: Tensor,
    scale: Tensor,
    experts: u32,
    top_k: u32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(
        scale.dtype, logits.dtype,
        "the gain is read at the router's own width"
    );
    debug_assert_eq!(
        scale.rows * scale.width,
        experts,
        "the gain is indexed by expert, so it holds one entry per expert"
    );
    ranked_planes(OP, logits, top_k, routes, weights);
    let (e, k) = router_extents(OP, logits, experts, top_k)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::moe_topk_softmax_scaled<{t}>"),
        ))
        .apply(router_lane(logits.rows)),
        &[
            logits.arg(),
            ArgValue::ABSENT,
            scale.arg(),
            routes.arg(),
            weights.arg(),
            e.arg(),
            k.arg(),
            0_i32.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ranked_router(
    ctx: &Ctx,
    op: &'static str,
    router: &'static Kernel,
    logits: Tensor,
    correction_bias: Option<Tensor>,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    ranked_planes(op, logits, top_k, routes, weights);
    let (e, k) = router_extents(op, logits, experts, top_k)?;
    ctx.fire(
        op,
        Fire::at(router).apply(router_lane(logits.rows)),
        &[
            logits.arg(),
            routes.arg(),
            weights.arg(),
            correction_bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            e.arg(),
            k.arg(),
            renormalize.arg(),
            scaling.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn topk_sigmoid(
    ctx: &Ctx,
    logits: Tensor,
    correction_bias: Option<Tensor>,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    ranked_router(
        ctx,
        OP,
        kernel(FILE, &format!("::pie::linear::moe_topk_sigmoid<{t}>")),
        logits,
        correction_bias,
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn topk_sigmoid_sink(
    ctx: &Ctx,
    logits: Tensor,
    correction_bias: Option<Tensor>,
    global_scale: Option<Tensor>,
    experts: u32,
    top_k: u32,
    sink: u32,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = top_k
        .checked_add(sink)
        .ok_or_else(|| refuse(OP, "the fan-out does not count"))?;
    ranked_planes(OP, logits, fan, routes, weights);
    nonzero(OP, "rows", logits.rows)?;
    nonzero(OP, "the fan-out this router states", top_k)?;
    router_fan(OP, fan)?;
    let width = experts
        .checked_add(sink)
        .ok_or_else(|| refuse(OP, "the expert count does not count"))?;
    if logits.width != width {
        return Err(refuse(
            OP,
            format!(
                "the router's row is {} wide and the statement names {experts} routed + {sink} sink experts",
                logits.width
            ),
        ));
    }
    if width > MAX_EXPERTS {
        return Err(refuse(
            OP,
            format!(
                "the expert count is {width}, above the {MAX_EXPERTS} scores this router \
                 stages in shared memory"
            ),
        ));
    }
    if let Some(bias) = &correction_bias {
        debug_assert_eq!(
            bias.dtype,
            Dtype::F32,
            "`{OP}` reads an f32 correction bias"
        );
    }
    if let Some(scale) = &global_scale {
        debug_assert_eq!(scale.dtype, Dtype::F32, "`{OP}` reads an f32 global scale");
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::linear::moe_topk_sigmoid_sink<{t}>"),
        ))
        .apply(router_lane(logits.rows)),
        &[
            logits.arg(),
            routes.arg(),
            weights.arg(),
            correction_bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            global_scale.map_or(ArgValue::ABSENT, |scale| scale.arg()),
            stated(OP, experts)?.arg(),
            stated(OP, sink)?.arg(),
            stated(OP, top_k)?.arg(),
            scaling.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn topk_sqrt_softplus(
    ctx: &Ctx,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_route";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(
        bias.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 correction bias"
    );
    ranked_router(
        ctx,
        OP,
        kernel(FILE, &format!("::pie::linear::moe_topk_sqrt_softplus<{t}>")),
        logits,
        Some(bias),
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}
