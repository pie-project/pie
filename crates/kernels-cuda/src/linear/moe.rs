//! `Moe`: routers, routed matmuls, and the folds that bring the fan-out
//! back. One entry per IR variant. The one quantized bank form this plane
//! stamps is mxfp4, and its bank arrives as the explicit `(codes, scales)`
//! pair the driver resolved (the metal precedent) — plain pointers, no
//! by-value descriptor.

use kernels::KernelError;
use model_ir::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;

const FILE: &str = "linear/moe.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const GEMV_WARPS: u32 = 4;

/// The router kernels stage every expert score in shared memory.
const MAX_EXPERTS: u32 = 512;

/// The GEMV puts the route run on the grid's y axis.
const MAX_GRID_Y: u32 = 65_535;

/// One block per row, warp-shuffle reduction scratch beside it — the ranked
/// routers' launch.
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

/// One narrow block per token row — the softmax router's launch.
const fn router_lane(rows: u32) -> Launch {
    const ROUTER_BLOCK: u32 = 64;

    Launch::per_row(rows, ROUTER_BLOCK)
}

/// Rows on their own grid axis, the width chunked across blocks.
fn elementwise_rows(op: &'static str, rows: u32, width: u32) -> Result<Launch, KernelError> {
    nonzero(op, "rows", rows)?;
    nonzero(op, "width", width)?;
    Ok(Launch::grid(
        [rows, width.div_ceil(BLOCK), 1],
        [BLOCK, 1, 1],
    ))
}

/// What every router lands: i32 routes and f32 weights, one row per token
/// row, `top_k` wide — the trace-time validator's guarantee, restated.
fn ranked_planes(op: &'static str, logits: Tensor, top_k: u32, routes: &Tensor, weights: &Tensor) {
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

/// The stated extents every router shares, refused or converted once.
fn router_extents(
    op: &'static str,
    logits: Tensor,
    experts: u32,
    top_k: u32,
) -> Result<(i32, i32), KernelError> {
    debug_assert_eq!(
        logits.width, experts,
        "the router's row is the expert count the statement states"
    );
    nonzero(op, "rows", logits.rows)?;
    nonzero(op, "the fan-out this router states", top_k)?;
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
) -> Result<(), KernelError> {
    const OP: &str = "linear.moe_topk_softmax";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ranked_planes(OP, logits, top_k, routes, weights);
    let (e, k) = router_extents(OP, logits, experts, top_k)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::moe_topk_softmax<{t}>")),
        )
        .apply(router_lane(logits.rows)),
        &[
            logits.arg(),
            ArgValue::ABSENT, // the activation seat the fused router form fills
            ArgValue::ABSENT, // the bias seat, likewise
            routes.arg(),
            weights.arg(),
            e.arg(),
            k.arg(),
            0_i32.arg(), // `hidden`, read only by the fused form
        ],
    )
}

/// The ranked routers' shared launch: sigmoid scoring, optionally biased.
#[allow(clippy::too_many_arguments)]
fn ranked_router(
    ctx: &Ctx,
    op: &'static str,
    file: &'static str,
    entrypoint: &'static str,
    logits: Tensor,
    correction_bias: Option<Tensor>,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), KernelError> {
    ranked_planes(op, logits, top_k, routes, weights);
    let (e, k) = router_extents(op, logits, experts, top_k)?;
    ctx.fire(
        op,
        Fire::at(file, entrypoint).apply(rms(logits.rows)),
        &[
            logits.arg(),
            routes.arg(),
            weights.arg(),
            correction_bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            e.arg(),
            k.arg(),
            renormalize.arg(),
            scaling.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn topk_sigmoid(
    ctx: &Ctx,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "linear.moe_topk_sigmoid";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ranked_router(
        ctx,
        OP,
        FILE,
        symbol(&format!("::pie::linear::moe_topk_sigmoid<{t}>")),
        logits,
        None,
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

/// Sigmoid routing with a per-expert correction bias; weights pass through
/// sqrt-softplus.
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
) -> Result<(), KernelError> {
    const OP: &str = "linear.moe_topk_sqrt_softplus";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(
        bias.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 correction bias"
    );
    ranked_router(
        ctx,
        OP,
        FILE,
        symbol(&format!("::pie::linear::moe_topk_sqrt_softplus<{t}>")),
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

/// The routed fan a selected matmul walks: `tokens x top_k` result rows, the
/// activation read either once per token or once per route.
struct Selected {
    /// `tokens x top_k`, the result's rows and the grid's route axis.
    route_count: u32,

    top_k: i32,

    /// Which activation reading the fan implies: `true` when `x` has one row
    /// per token (the up leg), `false` when one per route (the down leg).
    by_token: bool,
}

fn selected(
    op: &'static str,
    x: Tensor,
    routes: Tensor,
    y: &Tensor,
) -> Result<Selected, KernelError> {
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{op}` walks i32 routes");
    let top_k = nonzero(op, "the routed fan-out", routes.width)?;
    let route_count = routes.rows.checked_mul(top_k).ok_or_else(|| {
        refuse(
            op,
            format!(
                "the route run will not launch: {} tokens x {top_k} fan-out",
                routes.rows
            ),
        )
    })?;
    debug_assert_eq!(y.rows, route_count, "the result lands one row per route");
    let by_token = if x.rows == route_count {
        false
    } else if x.rows == routes.rows {
        true
    } else {
        return Err(refuse(
            op,
            format!(
                "the activation's {} rows are neither the fire's tokens nor its routes",
                x.rows
            ),
        ));
    };
    Ok(Selected {
        route_count,
        top_k: stated(op, top_k)?,
        by_token,
    })
}

/// Grouped matmul over a dense bank: each routed row multiplies the expert
/// its route selects — the decode GEMV, one warp-column per output tile.
pub fn matmul_select(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    select_gemv(ctx, "linear.moe_matmul_select", x, bank, routes, y)
}

/// The routed dense GEMV itself, under the caller's own op name.
///
/// **NAMED SEPARATELY BECAUSE A SECOND OP IS THE SAME LAUNCH.** LoRA's
/// projection half — `t[row] = A[routes[row]] · x[row]` — is a routed
/// matmul-select at fan-out one and nothing else: same bank indexing, same
/// per-route zero for a negative id, same float4 ladder. `linear::lora` fires
/// this rather than stamping a twin, and passes its own `op` so that a refusal
/// or a launch failure comes back attributed to the correction the author
/// wrote instead of to an MoE the plan does not contain.
pub(crate) fn select_gemv(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    /// K in whole float4 loads.
    const VEC_WIDTH: u32 = 8;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(bank.dtype, x.dtype, "the bank rides the activation's dtype");
    let fan = selected(op, x, routes, y)?;
    if fan.route_count > MAX_GRID_Y {
        return Err(refuse(
            op,
            format!(
                "the route run is {}, above the {MAX_GRID_Y} rows this GEMV puts on the \
                 grid's y axis; the aligned batched leg is what a wider fire wants",
                fan.route_count
            ),
        ));
    }
    if x.width == 0 || x.width % VEC_WIDTH != 0 {
        return Err(refuse(
            op,
            format!(
                "K is {}, not a whole number of {VEC_WIDTH}-element float4 loads",
                x.width
            ),
        ));
    }
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let form = if fan.by_token { "by_token" } else { "by_route" };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::moe_matmul_select_gemv_{form}<{t}>"
            )),
        )
        .apply(Launch::grid(
            [y.width.div_ceil(GEMV_WARPS), fan.route_count, 1],
            [WARP, GEMV_WARPS, 1],
        )),
        &[
            routes.arg(),
            x.arg(),
            bank.arg(),
            y.arg(),
            fan.top_k.arg(),
            k.arg(),
            n.arg(),
            (i64::from(n) * i64::from(k)).arg(), // the bank's expert stride
        ],
    )
}

/// The mxfp4 selects' shared launch — the one quantized bank form this plane
/// stamps. `codes` are the e2m1 nibbles, `scales` the shared e8m0 exponents,
/// 32 codes to one scale byte. The per-expert bias is what the two entries
/// differ in, so it arrives optional and the kernel's own bias seat carries
/// the absence.
#[allow(clippy::too_many_arguments)]
fn matmul_select_mxfp4(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const MXFP4_BLOCK: u32 = 32;

    const ROWS_PER_WARP: u32 = 4;

    const DECODE_BLOCK: u32 = 128;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "an mxfp4 bank's codes are u8");
    debug_assert_eq!(scales.dtype, Dtype::U8, "an mxfp4 bank's scales are u8");
    debug_assert!(
        bias.is_none_or(|bias| bias.dtype == x.dtype),
        "the expert bias rides the activation's dtype"
    );
    let fan = selected(op, x, routes, y)?;
    if x.width == 0 || x.width % MXFP4_BLOCK != 0 {
        return Err(refuse(
            op,
            format!(
                "K is {}, not a whole number of {MXFP4_BLOCK}-code mxfp4 blocks",
                x.width
            ),
        ));
    }
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    // Each warp decodes `ROWS_PER_WARP` bank rows; a block tiles that many
    // warps' worth of the output width.
    let tile = (DECODE_BLOCK / WARP) * ROWS_PER_WARP;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    ctx.fire(
        op,
        Fire::at(
            "linear/quant.cuh",
            symbol(&format!(
                "::pie::linear::moe_matmul_select_mxfp4<{t}, ::pie::i32({ROWS_PER_WARP})>"
            )),
        )
        .apply(Launch::grid(
            [fan.route_count, y.width.div_ceil(tile), 1],
            [DECODE_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            routes.arg(),
            codes.arg(),
            scales.arg(),
            bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
        ],
    )
}

/// Grouped matmul over an mxfp4 bank, with a per-expert bias — the gate/up
/// leg, whose bank and bias are cut the same way, so the add belongs inside
/// the fold.
#[allow(clippy::too_many_arguments)]
pub fn matmul_select_bias(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "linear.moe_matmul_select_bias";
    matmul_select_mxfp4(ctx, OP, x, codes, scales, Some(bias), routes, y)
}

/// Grouped matmul over an mxfp4 bank with nothing added — the down leg,
/// whose bank is rows-cut, so a replicated bias folded in here would be
/// summed once per rank by the all_reduce that follows. Its routed bias is
/// stated after the reduce instead, by [`bias_sum`].
pub fn matmul_select_quant(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "linear.moe_matmul_select_quant";
    matmul_select_mxfp4(ctx, OP, x, codes, scales, None, routes, y)
}

/// Folds the `top_k` routed rows back to one row per token, weighted.
pub fn weighted_sum(
    ctx: &Ctx,
    routed: Tensor,
    weights: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "linear.moe_weighted_sum";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` reads f32 route weights");
    nonzero(OP, "the token rows this fold lands on", y.rows)?;
    if routed.rows % y.rows != 0 {
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
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::moe_weighted_sum<{t}>")),
        )
        .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            routed.arg(),
            weights.arg(),
            stated(OP, top_k)?.arg(),
            stated(OP, y.width)?.arg(),
        ],
    )
}

/// The routed bias mixture, stated once on an activation that already holds
/// the fold: `y[t] = x[t] + sum_k weights[t, k] * bias[routes[t, k]]`.
///
/// It is its own fire because the expert down-projection is rows-cut under
/// tp: each rank's routed matmul is a *partial* product and the all_reduce
/// sums the ranks, so a replicated bias folded into that matmul would land
/// once per rank. Routing is computed from replicated inputs, so `routes`
/// and `weights` are the same on every rank and the mixture can be stated
/// after the reduce, on the reduced activation. At tp = 1 the value is
/// identical (the weights sum to one), so the statement has one path and no
/// tp branch.
pub fn bias_sum(
    ctx: &Ctx,
    x: Tensor,
    bias: Tensor,
    routes: Tensor,
    weights: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "linear.moe_bias_sum";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 routes");
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` reads f32 route weights");
    debug_assert_eq!(
        bias.dtype, x.dtype,
        "the expert bias rides the activation's dtype"
    );
    debug_assert!(
        x.rows == y.rows && x.width == y.width,
        "the bias lands on the activation's own rectangle"
    );
    debug_assert_eq!(
        bias.width, y.width,
        "an expert's bias row is the width it is added to"
    );
    debug_assert_eq!(
        routes.rows, y.rows,
        "the route plane is one row per token row"
    );
    debug_assert!(
        weights.rows == y.rows && weights.width == routes.width,
        "the weight plane is one weight per route"
    );
    let top_k = nonzero(OP, "the routed fan-out", routes.width)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::moe_bias_sum<{t}>")))
            .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            x.arg(),
            bias.arg(),
            routes.arg(),
            weights.arg(),
            stated(OP, top_k)?.arg(),
            stated(OP, y.width)?.arg(),
        ],
    )
}

/// `y = routed + sigmoid(gate) * shared`, per element; the gate is one
/// scalar per token row, read at its row's head.
pub fn sigmoid_gate_add(
    ctx: &Ctx,
    routed: Tensor,
    shared: Tensor,
    gate: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "linear.moe_sigmoid_gate_add";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        routed.rows == y.rows && routed.width == y.width,
        "the routed row is the combined row"
    );
    debug_assert!(
        shared.rows == y.rows && shared.width == y.width,
        "the shared expert's rectangle is the routed one"
    );
    debug_assert_eq!(gate.rows, y.rows, "the gate column is one scalar per row");
    ctx.fire(
        OP,
        Fire::at(
            "linear/glu.cuh",
            symbol(&format!("::pie::linear::moe_sigmoid_gate_add<{t}>")),
        )
        .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            routed.arg(),
            shared.arg(),
            gate.arg(),
            stated(OP, y.width)?.arg(),
            // The gate's row pitch: handles are dense, so the scalar sits at
            // the head of a `gate.width`-wide row (the old strided column,
            // stated on the handle).
            stated(OP, nonzero(OP, "the gate row's pitch", gate.width)?)?.arg(),
        ],
    )
}
