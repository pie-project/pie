//! `Moe`: routers, routed matmuls, and the folds that bring the fan-out
//! back. One entry per IR variant. The one quantized bank form this plane
//! stamps is mxfp4, spelled in source — the composed affine namespace in
//! [`quant`](crate::linear::quant) has no routed point here.

use kernels::KernelError;
use model_ir::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const ROUTER_MAX_EXPERTS: u32 = 1024;

const ROUTER_MAX_TOP_K: u32 = 16;

const SELECT_GROUP: u32 = 128;

const QMV_GROUP: [u32; 3] = [32, 2, 1];

const MXFP4_BLOCK: u32 = 32;

/// The softmax router's normalization mode: over all experts, not the
/// selected k.
const SOFTMAX_OVER_SELECTED: u32 = 0;

fn router_lanes(op: &'static str, experts: u32) -> Result<u32, KernelError> {
    nonzero(op, "the expert count this router states", experts)?;
    Ok(experts.min(1024).div_ceil(32) * 32)
}

/// One thread per element, one threadgroup row per token row.
fn route_rows(op: &'static str, width: u32, rows: u32) -> Result<Grid, KernelError> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    Ok(Grid::of([width, rows, 1], [width.min(256), 1, 1]))
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

/// The staged routers' shared shape: one lane block per token row, the
/// fan-out held in a threadgroup array — hence the caps.
fn ranked(
    op: &'static str,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<Grid, KernelError> {
    debug_assert_eq!(
        logits.width, experts,
        "the router's row is the expert count the statement states"
    );
    ranked_planes(op, logits, top_k, routes, weights);
    nonzero(op, "the fan-out this router states", top_k)?;
    if experts > ROUTER_MAX_EXPERTS {
        return Err(refuse(
            op,
            format!(
                "the expert count is {experts}, above the {ROUTER_MAX_EXPERTS} lanes this \
                 router gives one apiece"
            ),
        ));
    }
    if top_k > ROUTER_MAX_TOP_K {
        return Err(refuse(
            op,
            format!(
                "the fan-out is {top_k}, above the {ROUTER_MAX_TOP_K} this router stages in \
                 a threadgroup array"
            ),
        ));
    }
    let lanes = router_lanes(op, experts)?;
    nonzero(op, "rows", logits.rows)?;
    Ok(Grid::of([lanes, logits.rows, 1], [lanes, 1, 1]))
}

pub fn topk_softmax(
    ctx: &Ctx<'_>,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "moe.topk_softmax";
    let entry = dtype_dispatch!(OP, logits.dtype, { Bf16 => "router_topk_f32w_bfloat16" });
    debug_assert_eq!(
        logits.width, experts,
        "the router's row is the expert count the statement states"
    );
    ranked_planes(OP, logits, top_k, routes, weights);
    nonzero(OP, "the fan-out this router states", top_k)?;
    let lanes = router_lanes(OP, experts)?;
    nonzero(OP, "rows", logits.rows)?;
    ctx.fire(
        Fire::at("linear/moe_route.metal", entry)
            .apply(Grid::of([lanes, logits.rows, 1], [lanes, 1, 1])),
        &[
            logits.arg(),
            routes.arg_mut(),
            weights.arg_mut(),
            ctx.absent()?, // the bias seat the softplus router fills
            experts.arg(),
            top_k.arg(),
            SOFTMAX_OVER_SELECTED.arg(),
            experts.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn topk_sigmoid(
    ctx: &Ctx<'_>,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "moe.topk_sigmoid";
    let entry = dtype_dispatch!(OP, logits.dtype, { Bf16 => "router_topk_sigmoid" });
    let grid = ranked(OP, logits, experts, top_k, routes, weights)?;
    ctx.fire(
        Fire::at("linear/moe_route.metal", entry).apply(grid),
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

/// Sigmoid routing with a per-expert correction bias; weights pass through
/// sqrt-softplus.
#[allow(clippy::too_many_arguments)]
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
) -> Result<(), KernelError> {
    const OP: &str = "moe.topk_sqrt_softplus";
    let entry = dtype_dispatch!(OP, logits.dtype, { Bf16 => "router_topk_sqrt_softplus" });
    debug_assert_eq!(
        bias.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 correction bias"
    );
    let grid = ranked(OP, logits, experts, top_k, routes, weights)?;
    ctx.fire(
        Fire::at("linear/moe_route.metal", entry).apply(grid),
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

/// The routed fan a selected matmul walks: `tokens x top_k` result rows, the
/// activation read either once per token or once per route.
struct Selected {
    tokens: u32,

    top_k: u32,

    x_row_stride: u32,

    x_slot_stride: u32,
}

fn selected(
    op: &'static str,
    x: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<Selected, KernelError> {
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{op}` walks i32 routes");
    nonzero(op, "the routed fan-out", routes.width)?;
    nonzero(op, "K, the activation's width", x.width)?;
    nonzero(op, "N, the bank's output width", y.width)?;
    let routed = routes.rows.checked_mul(routes.width).ok_or_else(|| {
        refuse(
            op,
            format!(
                "the route run will not launch: {} tokens x {} fan-out",
                routes.rows, routes.width
            ),
        )
    })?;
    debug_assert_eq!(y.rows, routed, "the result lands one row per route");
    let (x_row_stride, x_slot_stride) = if x.rows == routes.rows {
        (x.width, 0)
    } else if x.rows == routed {
        let row = x.width.checked_mul(routes.width).ok_or_else(|| {
            refuse(
                op,
                format!(
                    "the activation's row will not stride: {} wide x {} fan-out",
                    x.width, routes.width
                ),
            )
        })?;
        (row, x.width)
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
        tokens: routes.rows,
        top_k: routes.width,
        x_row_stride,
        x_slot_stride,
    })
}

fn select_gemv_grid(op: &'static str, out_width: u32, routed_rows: u32) -> Result<Grid, KernelError> {
    let lanes = out_width.checked_mul(32).ok_or_else(|| {
        refuse(
            op,
            format!("the {out_width} output columns, one simdgroup each, will not launch"),
        )
    })?;
    Ok(Grid::of([lanes, routed_rows, 1], [SELECT_GROUP, 1, 1]))
}

fn routed_qmv_grid(
    op: &'static str,
    tokens: u32,
    out_width: u32,
    top_k: u32,
) -> Result<[u32; 3], KernelError> {
    let x = tokens.checked_mul(32).ok_or_else(|| {
        refuse(
            op,
            format!("the {tokens} token rows, one simdgroup each, will not launch"),
        )
    })?;
    Ok([x, out_width.div_ceil(4), top_k])
}

/// Grouped matmul over a dense bank: each routed row multiplies the expert
/// its route selects.
pub fn matmul_select(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "moe.matmul_select";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "select_gemv" });
    debug_assert_eq!(bank.dtype, x.dtype, "the bank rides the activation's dtype");
    let fan = selected(OP, x, routes, y)?;
    ctx.fire(
        Fire::at("linear/moe_select.metal", entry).apply(select_gemv_grid(OP, y.width, y.rows)?),
        &[
            x.arg(),
            bank.arg(),
            routes.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, fan.top_k)?.arg(),
            stated(OP, fan.x_row_stride)?.arg(),
            stated(OP, fan.x_slot_stride)?.arg(),
        ],
    )
}

/// Grouped matmul over an mxfp4 bank, with a per-expert bias — the one bank
/// form this plane stamps. The driver resolves the bank weight to its
/// `(codes, scales)` planes before calling.
pub fn matmul_select_bias(
    ctx: &Ctx<'_>,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "moe.matmul_select_bias";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "an mxfp4 bank's codes are u8");
    debug_assert_eq!(scales.dtype, Dtype::U8, "an mxfp4 bank's scales are u8");
    debug_assert_eq!(bias.dtype, x.dtype, "the expert bias rides the activation's dtype");
    let fan = selected(OP, x, routes, y)?;
    if x.width % MXFP4_BLOCK != 0 {
        return Err(refuse(
            OP,
            format!(
                "K is {}, not a whole number of {MXFP4_BLOCK}-code mxfp4 blocks",
                x.width
            ),
        ));
    }
    ctx.fire(
        Fire::at("linear/quant_qmv.metal", entry).apply(Grid::of(
            routed_qmv_grid(OP, fan.tokens, y.width, fan.top_k)?,
            QMV_GROUP,
        )),
        &[
            codes.arg(),
            scales.arg(),
            ctx.absent()?, // the affine biases seat; mxfp4 carries none
            x.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            stated(OP, y.width)?.arg(),
            bias.arg(),
            routes.arg(),
            stated(OP, fan.x_slot_stride)?.arg(),
            stated(OP, fan.x_row_stride)?.arg(),
            stated(OP, fan.top_k)?.arg(),
        ],
    )
}

/// Folds the `top_k` routed rows back to one row per token, weighted.
pub fn weighted_sum(
    ctx: &Ctx<'_>,
    routed: Tensor,
    weights: Tensor,
    y: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "moe.weighted_sum";
    let entry = dtype_dispatch!(OP, routed.dtype, { Bf16 => "expert_combine" });
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
    let grid = route_rows(OP, y.width, y.rows)?;
    ctx.fire(
        Fire::at("linear/moe_route.metal", entry).apply(grid),
        &[
            routed.arg(),
            weights.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            stated(OP, top_k)?.arg(),
        ],
    )
}

/// `y = routed + sigmoid(gate) * shared`, per element.
pub fn sigmoid_gate_add(
    ctx: &Ctx<'_>,
    routed: Tensor,
    shared: Tensor,
    gate: Tensor,
    y: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "moe.sigmoid_gate_add";
    let entry = dtype_dispatch!(OP, routed.dtype, { Bf16 => "shared_expert_combine" });
    debug_assert!(
        shared.rows == routed.rows && shared.width == routed.width,
        "the shared expert's rectangle is the routed one"
    );
    debug_assert!(
        gate.rows == routed.rows && gate.width == routed.width,
        "the gate plane rides the rectangle it gates"
    );
    let grid = route_rows(OP, routed.width, routed.rows)?;
    ctx.fire(
        Fire::at("linear/moe_route.metal", entry).apply(grid),
        &[
            routed.arg(),
            shared.arg(),
            gate.arg(),
            y.arg_mut(),
            stated(OP, routed.width)?.arg(),
        ],
    )
}
