use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;
use dtype::Dtype;

/// The sigmoid router stages twice as many (Kimi-K3 routes over 896).
pub(super) const SIGMOID_MAX_EXPERTS: u32 = 1024;

pub(crate) const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

pub(super) const fn router_lane(rows: u32) -> Launch {
    const ROUTER_BLOCK: u32 = 64;

    Launch::per_row(rows, ROUTER_BLOCK)
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
    router_extents_upto(op, logits, experts, top_k, MAX_EXPERTS)
}

/// [`router_extents`] for a router that stages up to `max` scores.
pub(super) fn router_extents_upto(
    op: &'static str,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    max: u32,
) -> Result<(i32, i32), Error> {
    debug_assert_eq!(
        logits.width, experts,
        "the router's row is the expert count the statement states"
    );
    nonzero(op, "rows", logits.rows)?;
    nonzero(op, "the fan-out this router states", top_k)?;
    if experts > max {
        return Err(refuse(
            op,
            format!(
                "the expert count is {experts}, above the {max} scores this router \
                 stages in shared memory"
            ),
        ));
    }
    Ok((stated(op, experts)?, stated(op, top_k)?))
}

/// A router of at most 512 experts and a top-k of at most 16 a warp a row
/// (`moe_topk_softmax_lane`): `None` past them.
pub(super) fn topk_lane(t: &str, experts: u32, top_k: u32, rows: u32, pdl: bool) -> Option<Fire> {
    if experts == 0 || experts > 512 || top_k == 0 || top_k > 16 {
        return None;
    }
    let per = experts.div_ceil(32).next_power_of_two();
    Some(
        Fire::at(kernel(
            ROUTE,
            &format!("::pie::linear::moe_topk_softmax_lane<{t}, {per}>"),
        ))
        .apply(Launch::grid([rows.max(1), 1, 1], [32, 1, 1]).programmatic(pdl)),
    )
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
    if let Some(fire) = topk_lane(t, experts, top_k, logits.rows, ctx.tuning().pdl) {
        return ctx.fire(
            OP,
            fire,
            &[
                logits.arg(),
                routes.arg(),
                weights.arg(),
                e.arg(),
                k.arg(),
                ctx.stage(),
                ArgValue::ABSENT,
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            ROUTE,
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
    if let Some(fire) = topk_lane(t, experts, top_k, logits.rows, ctx.tuning().pdl) {
        return ctx.fire(
            OP,
            fire,
            &[
                logits.arg(),
                routes.arg(),
                weights.arg(),
                e.arg(),
                k.arg(),
                ctx.stage(),
                scale.arg(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            ROUTE,
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
    softplus: bool,
    logits: Tensor,
    correction_bias: Option<Tensor>,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: &mut Tensor,
    weights: &mut Tensor,
    max: u32,
) -> Result<(), Error> {
    const ROWS: u32 = 4;
    let t = dtype_dispatch!(op, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    ranked_planes(op, logits, top_k, routes, weights);
    let (e, k) = router_extents_upto(op, logits, experts, top_k, max)?;
    let per = experts.div_ceil(WARP).next_power_of_two();
    ctx.fire(
        op,
        Fire::at(kernel(
            ROUTE,
            &format!("::pie::linear::moe_topk_ranked<{t}, {softplus}, {per}>"),
        ))
        .apply(
            Launch::grid([logits.rows.div_ceil(ROWS), 1, 1], [WARP * ROWS, 1, 1])
                .programmatic(ctx.tuning().pdl),
        ),
        &[
            logits.arg(),
            routes.arg(),
            weights.arg(),
            correction_bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            stated(op, logits.rows)?.arg(),
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
    ranked_router(
        ctx,
        OP,
        false,
        logits,
        correction_bias,
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
        SIGMOID_MAX_EXPERTS,
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
            ROUTE,
            &format!("::pie::linear::moe_topk_sigmoid_sink<{t}>"),
        ))
        .apply(rms(logits.rows)),
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
    debug_assert_eq!(
        bias.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 correction bias"
    );
    ranked_router(
        ctx,
        OP,
        true,
        logits,
        Some(bias),
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
        MAX_EXPERTS,
    )
}

#[derive(Clone, Copy)]
pub(crate) struct RouteOrder {
    pub(crate) order: u64,
    pub(crate) offsets: u64,
    pub(crate) work: u64,
    pub(crate) work_cap: u32,
}

pub(crate) fn route_order(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    fan: &Selected,
    experts: u32,
) -> Result<RouteOrder, Error> {
    route_order_sliced(ctx, op, routes, fan, experts, WMMA_ROUTES, false, Share::No)
}

/// Whether a route order is kept for, or taken from, the select beside it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Share {
    No,
    /// A GLU select's order, kept for the down select that follows it.
    Keep,
    /// The down select: the kept order of the same routes, if it is the last
    /// one laid.
    Take,
}

/// What a route order was laid for: the same routes tensor in the same
/// window, stream, bank and slice.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct OrderKey {
    stream: usize,
    routes: u64,
    rows: u32,
    cols: u32,
    stage: u64,
    experts: u32,
    slice: u32,
    count: u32,
}

std::thread_local! {
    /// The last route order laid on this thread, when a GLU select (or one
    /// in [`keeping_order`]) laid it for its down select. Any other order laid clears it: they share the
    /// scratch.
    static KEPT_ORDER: core::cell::Cell<Option<(OrderKey, RouteOrder)>> =
        const { core::cell::Cell::new(None) };
    /// Inside [`keeping_order`]: the select keeps its order, whatever it
    /// lands.
    pub(super) static KEEP_NEXT: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Runs `f` with its select keeping the route order it lays, for the next
/// select of the same routes, which takes the order rather than lay it again.
pub fn keeping_order<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            KEEP_NEXT.with(|k| k.set(self.0));
        }
    }
    let _reset = Reset(KEEP_NEXT.with(|k| k.replace(true)));
    f()
}

/// [`route_order`] with each work item `slice` routes of one expert; with
/// `scan`, its walks over the experts run as block scans
/// (`moe_route_order_scan`) instead of on one thread.
#[allow(clippy::too_many_arguments)]
pub(super) fn route_order_sliced(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    fan: &Selected,
    experts: u32,
    slice: u32,
    scan: bool,
    share: Share,
) -> Result<RouteOrder, Error> {
    const ORDER_BLOCK: u32 = 1024;
    let key = OrderKey {
        stream: ctx.stream().addr(),
        routes: routes.ptr,
        rows: routes.rows,
        cols: routes.width,
        stage: match ctx.stage() {
            ArgValue::Ptr(p) => p,
            _ => u64::MAX,
        },
        experts,
        slice,
        count: fan.route_count,
    };
    let kept = KEPT_ORDER.with(|k| k.take());
    if share == Share::Take
        && let Some((was, order)) = kept
        && was == key
    {
        return Ok(order);
    }
    let offsets_at = (fan.route_count as usize).next_multiple_of(64);
    // An expert's items are its whole slices and at most one part slice: at
    // decode widths the grid is the routes', not the bank's.
    let work_cap = fan.route_count.div_ceil(slice) + experts.min(fan.route_count);
    let work_at = (offsets_at + experts as usize + 3).next_multiple_of(64);
    let order = ctx.scratch(
        op,
        "moe_route_order",
        (work_at + work_cap as usize) * core::mem::size_of::<i32>(),
    )? as usize as u64;
    let offsets = order + (offsets_at * core::mem::size_of::<i32>()) as u64;
    let work = order + (work_at * core::mem::size_of::<i32>()) as u64;
    ctx.fire(
        op,
        if scan {
            Fire::at(kernel(ROUTE, "::pie::linear::moe_route_order_scan"))
        } else {
            Fire::at(kernel(ROUTE, "::pie::linear::moe_route_order"))
        }
        .apply(
            Launch::grid([1, 1, 1], [ORDER_BLOCK, 1, 1])
                .smem(2 * (experts + 1) * 4)
                .programmatic(ctx.tuning().pdl && scan),
        ),
        &[
            routes.arg(),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            stated(op, work_cap)?.arg(),
            slice.arg(),
            fan.top_k.arg(),
            stated(op, fan.route_count)?.arg(),
            stated(op, experts)?.arg(),
            ctx.stage(),
        ],
    )?;
    let laid = RouteOrder {
        order,
        offsets,
        work,
        work_cap,
    };
    if share == Share::Keep {
        KEPT_ORDER.with(|k| k.set(Some((key, laid))));
    }
    Ok(laid)
}

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
    const OP: &str = "linear.moe_route";
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
        Fire::at(kernel(ROUTE, "::pie::linear::hash_route_gather"))
            .apply(Launch::flat(rows, BLOCK)),
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
        Fire::at(kernel(ROUTE, "::pie::linear::group_routes")).apply(Launch::flat(lanes, BLOCK)),
        &[
            routes.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, groups)?.arg(),
            ctx.stage(),
        ],
    )
}

/// How a fused gate ([`gate`]) scores its router's logits: one of the
/// routers above, the same picks and weights.
#[derive(Clone, Copy, Debug)]
pub enum GateScore {
    /// [`topk_softmax`]; with `gain`, [`topk_softmax_scaled`] (a bf16 gain
    /// an expert).
    Softmax { gain: Option<Tensor> },
    /// [`topk_sigmoid`].
    Sigmoid {
        bias: Option<Tensor>,
        renormalize: bool,
        scaling: f32,
    },
    /// [`topk_sqrt_softplus`] (DeepSeek-V4's).
    SqrtSoftplus {
        bias: Tensor,
        renormalize: bool,
        scaling: f32,
    },
}

/// A fused gate: the router's product `x . router^T` landing at `logits`
/// (bf16 for `linear.matmul`, f32 for `linear.matmul_f32`), then
/// `experts` routed to `top_k` a row as `score` says.
#[derive(Clone, Copy, Debug)]
pub struct Gate {
    pub experts: u32,
    pub top_k: u32,
    pub logits: Dtype,
    pub score: GateScore,
}

/// `pie::linear::gate_tc5::Params`, field for field.
#[cfg(feature = "cuda")]
#[repr(C)]
struct GateParams {
    bias: u64,
    gain: u64,
    routes: u64,
    weights: u64,
    win: u64,
    rows: i32,
    k: i32,
    experts: i32,
    top_k: i32,
    renormalize: i32,
    scaling: f32,
}

/// The CTAs a cluster splits the router's K over: B200 seats one cluster of
/// 16 one-block SMs a GPC.
#[cfg(feature = "cuda")]
const GATE_SPLIT: u32 = 16;

/// The token blocks (clusters of [`GATE_SPLIT`]) a fused gate launches at
/// most before it widens its block: past four, B200 seats them in waves.
#[cfg(feature = "cuda")]
const GATE_BLOCKS: u32 = 4;

/// The token blocks the gate takes (an MMA's N), narrowest first.
#[cfg(feature = "cuda")]
const GATE_BN: [u32; 4] = [16, 32, 64, 128];

/// The most experts a gate holds: four 128-lane accumulators.
#[cfg(feature = "cuda")]
const GATE_MAX_EXPERTS: u32 = 512;

/// The most picks a gate lands: a lane each.
#[cfg(feature = "cuda")]
const GATE_MAX_TOP_K: u32 = 32;

/// The ring's bytes a CTA, its partials' too: past them the barriers and
/// the alignment, under the 227 KiB a block opts into.
#[cfg(feature = "cuda")]
const GATE_RING: u32 = 200 * 1024;

/// A gate CTA's threads: a loader warp, an MMA warp, four epilogue warps
/// and two that stage the experts' bias; all eight score and rank.
#[cfg(feature = "cuda")]
const GATE_THREADS: u32 = 256;

/// The head before a gate CTA's ring: its barriers and TMEM slot, then the
/// experts' bias or gain.
#[cfg(feature = "cuda")]
const GATE_HEAD: u32 = 1024 + GATE_MAX_EXPERTS * 4;

/// The partials a gate CTA of a split of `split` ranks lands for the
/// tokens it ranks, then their keys and scores
/// (`pie::linear::gate_tc5::recv_bytes`).
#[cfg(feature = "cuda")]
const fn gate_recv(groups: u32, bn: u32, split: u32) -> u32 {
    (split + 2) * bn.div_ceil(split) * groups * 128 * 4
}

/// The dynamic shared memory a gate CTA takes with its partials beside its
/// ring (`pie::linear::gate_tc5::kRoom`); past it they land over the ring.
#[cfg(feature = "cuda")]
const GATE_ROOM: u32 = 224 * 1024;

/// `pie::linear::gate_tc5::smem_bytes`: the 1024-byte alignment, the head,
/// the ring and the partials beside it where they fit, else over it.
#[cfg(feature = "cuda")]
const fn gate_smem(groups: u32, bn: u32, split: u32, stages: u32) -> u32 {
    let ring = stages * (groups * 128 * 128 + bn * 128);
    let recv = gate_recv(groups, bn, split);
    if 1024 + GATE_HEAD + ring + recv <= GATE_ROOM {
        1024 + GATE_HEAD + ring + recv
    } else {
        1024 + GATE_HEAD + if ring > recv { ring } else { recv }
    }
}

/// The router's product and its top-k in one tcgen05 launch
/// (`linear/moe/route_sm100.cu`): the routes and weights `matmul` (or
/// `matmul_f32`, as `gate.logits` says) and then `gate.score`'s router
/// land, without the logits. `Ok(false)` where it does not take the fire
/// (not a tcgen05 chip, past `kernel.moe.gate_tc5_until` rows, an operand
/// it does not read, more than 512 experts or 32 picks, a softmax of more
/// than 16): the caller runs the two.
pub fn gate(
    ctx: &Ctx,
    x: Tensor,
    router: Tensor,
    gate: Gate,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<bool, Error> {
    #[cfg(feature = "cuda")]
    {
        gate_tc5(ctx, x, router, gate, routes, weights)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ctx, x, router, gate, routes, weights);
        Ok(false)
    }
}

#[cfg(feature = "cuda")]
fn gate_tc5(
    ctx: &Ctx,
    x: Tensor,
    router: Tensor,
    gate: Gate,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<bool, Error> {
    use crate::linear::gemm_sm100::{Map, encode, supported};
    use cudarc::driver::sys as dr;
    const OP: &str = "linear.moe_gate";
    let Gate {
        experts,
        top_k,
        logits,
        score,
    } = gate;
    let rows = x.rows;
    let k = x.width;
    let (code, bias, gain, renormalize, scaling) = match score {
        GateScore::Softmax { gain } => {
            let fits = top_k <= 16
                && logits == Dtype::Bf16
                && gain.is_none_or(|g| g.dtype == Dtype::Bf16 && g.rows * g.width == experts);
            if !fits {
                return Ok(false);
            }
            (0, None, gain, false, 1.0)
        }
        GateScore::Sigmoid {
            bias,
            renormalize,
            scaling,
        } => (1, bias, None, renormalize, scaling),
        GateScore::SqrtSoftplus {
            bias,
            renormalize,
            scaling,
        } => (2, Some(bias), None, renormalize, scaling),
    };
    let takes = supported()
        && rows > 0
        && rows <= ctx.tuning().gate_tc5_until
        && x.dtype == Dtype::Bf16
        && router.dtype == Dtype::Bf16
        && matches!(logits, Dtype::Bf16 | Dtype::F32)
        && router.rows == experts
        && router.width == k
        && (1..=GATE_MAX_EXPERTS).contains(&experts)
        && (1..=GATE_MAX_TOP_K.min(experts)).contains(&top_k)
        && k > 0
        && k.is_multiple_of(8)
        && x.ptr.is_multiple_of(16)
        && router.ptr.is_multiple_of(16)
        && bias.is_none_or(|b| b.dtype == Dtype::F32 && b.rows * b.width == experts)
        && routes.dtype == Dtype::I32
        && weights.dtype == Dtype::F32
        && routes.rows == rows
        && weights.rows == rows
        && routes.width == top_k
        && weights.width == top_k;
    if !takes {
        return Ok(false);
    }
    let groups = experts.div_ceil(128);
    let nb = k.div_ceil(64);
    let split = GATE_SPLIT.min(nb);
    // The narrowest block that keeps the launch to its clusters; else the
    // widest whose partials the ring holds.
    let fit = |bn: u32| gate_recv(groups, bn, split) <= GATE_RING;
    let widest = GATE_BN.into_iter().rfind(|&bn| fit(bn)).unwrap_or(16);
    let bn = GATE_BN
        .into_iter()
        .find(|&bn| fit(bn) && rows.div_ceil(bn) <= GATE_BLOCKS)
        .unwrap_or(widest);
    let blocks = rows.div_ceil(bn);
    let stage = groups * 128 * 128 + bn * 128;
    let stages = nb.div_ceil(split).min(GATE_RING / stage).max(1);
    let map_w = encode(Map {
        ptr: router.ptr,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        dims: [k, experts, 0],
        strides: [u64::from(k) * 2, 0],
        boxes: [64, 128, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    })?;
    let map_x = encode(Map {
        ptr: x.ptr,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        dims: [k, rows, 0],
        strides: [u64::from(k) * 2, 0],
        boxes: [64, bn, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    })?;
    let params = GateParams {
        bias: bias.map_or(0, |b| b.ptr),
        gain: gain.map_or(0, |g| g.ptr),
        routes: routes.ptr,
        weights: weights.ptr,
        win: match ctx.stage() {
            ArgValue::Ptr(p) => p,
            _ => 0,
        },
        rows: stated(OP, rows)?,
        k: stated(OP, k)?,
        experts: stated(OP, experts)?,
        top_k: stated(OP, top_k)?,
        renormalize: i32::from(renormalize),
        scaling,
    };
    let round = logits == Dtype::Bf16;
    ctx.fire(
        OP,
        Fire::at(kernel(
            ROUTE_SM100,
            &format!(
                "::pie::linear::moe_gate_tc5<{groups}, {bn}, {split}, {stages}, {code}, {round}>"
            ),
        ))
        .apply(
            Launch::grid([blocks * split, 1, 1], [GATE_THREADS, 1, 1])
                .smem(gate_smem(groups, bn, split, stages))
                .cluster(split)
                .programmatic(ctx.tuning().pdl),
        ),
        &[
            ArgValue::of(&map_w),
            ArgValue::of(&map_x),
            ArgValue::of(&params),
        ],
    )?;
    Ok(true)
}
