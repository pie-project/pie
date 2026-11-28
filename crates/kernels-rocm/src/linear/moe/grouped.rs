//! The grouped legs: routes ordered by expert, each expert's run one work
//! item, and the strided dense matmuls of a grouped bank.

#[allow(clippy::wildcard_imports)]
use super::*;

/// A grouped projection (`linear.matmul_grouped`): row r's group g,
/// `x[r·groups + g]` (k wide), through the bank's expert `routes[r][g]` (n x k)
/// into `y[r·groups + g]`. `identity` states the routes are `group_routes`' (route g for group
/// g, every row), so the fire is `groups` dense matmuls over strided planes:
/// past the batch legs' reach (an expert there sees a route a row, too many
/// for them, and too few experts for the grouped leg to fill the card) one
/// strided-batched hipBLASLt launch takes it, else the routed select.
pub fn matmul_grouped(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
    identity: bool,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_grouped";
    let groups = routes.width;
    if identity
        && groups > 1
        && x.dtype == Dtype::Bf16
        && bank.dtype == Dtype::Bf16
        && y.dtype == Dtype::Bf16
        // The strided launch reads the contiguous bank only.
        && !experts.tabled()
        && dense_experts(x, bank, y, experts) == Some(groups)
        && routes.rows > ctx.tuning().moe_batch_until()
        && x.rows == routes.rows * groups
        && y.rows == x.rows
        && grouped_lt(
            ctx,
            OP,
            x.ptr,
            bank.ptr,
            y.ptr,
            routes.rows,
            y.width,
            x.width,
            groups,
        )?
    {
        return Ok(());
    }
    matmul_select(ctx, x, bank, routes, y, experts)
}

/// Without a runtime there is no hipBLASLt.
#[cfg(not(feature = "rocm"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn grouped_lt(
    _: &Ctx,
    _: &'static str,
    _: u64,
    _: u64,
    _: u64,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
) -> Result<bool, Error> {
    Ok(false)
}

/// `groups` strided bf16 matmuls in one hipBLASLt launch; `Ok(false)` where
/// hipBLASLt offers no algorithm for the shape.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub(super) fn grouped_lt(
    ctx: &Ctx,
    op: &'static str,
    x: u64,
    w: u64,
    y: u64,
    m: u32,
    n: u32,
    k: u32,
    groups: u32,
) -> Result<bool, Error> {
    use crate::hip::lt;
    let shape = lt::Shape {
        m,
        n,
        k,
        bias: false,
        groups,
        f32_out: false,
    };
    let workspace = ctx.scratch_shared(op, "linear.lt_workspace", lt::WORKSPACE)? as u64;
    // SAFETY: x is [m, groups·k], y [m, groups·n] and w groups blocks of
    // [n, k], device planes the caller sized; the workspace is this stream's.
    let landed = unsafe { lt::matmul(shape, x, w, 0, y, workspace, ctx.stream()) };
    landed.map_err(|why| refuse(op, format!("{why} at {groups} x M={m} N={n} K={k}")))
}

/// The most experts `moe_route_order` buckets in its LDS.
pub(super) const ORDER_EXPERT_CAP: u32 = 4096;

/// The experts of a 16-bit bank the expert-ordered legs can bucket, if it is
/// one and lies in device memory (contiguous, or a located table).
pub(super) fn dense_experts(
    x: Tensor,
    bank: Tensor,
    y: &Tensor,
    experts: ExpertTable,
) -> Option<u32> {
    if experts.streams() {
        return None;
    }
    if !matches!(x.dtype, Dtype::Bf16 | Dtype::F16) || bank.dtype != x.dtype {
        return None;
    }
    let count = bank.rows;
    if count == 0 || count > ORDER_EXPERT_CAP {
        return None;
    }
    if u64::from(bank.width) != u64::from(x.width) * u64::from(y.width) {
        return None;
    }
    Some(count)
}

pub(super) fn grouped_worth(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    y: &Tensor,
    fan: &Selected,
    experts: ExpertTable,
) -> Option<u32> {
    if !ctx.tuning().moe_grouped() {
        return None;
    }
    let count = dense_experts(x, bank, y, experts)?;
    (fan.route_count.div_ceil(count) >= ctx.tuning().moe_group_at()).then_some(count)
}

/// The planes `moe_route_order` lands: routes in expert order, each
/// expert's run, and the work list of `routes`-route items.
pub(super) struct RouteOrder {
    pub(super) order: u64,
    pub(super) offsets: u64,
    pub(super) work: u64,
    pub(super) work_cap: u32,
    pub(super) routes: u32,
}

/// Routes a grouped work item takes: 32, or 128 once the experts average
/// `GROUP_WIDE_AT` routes, where reading each expert's weight a quarter as
/// often outweighs the taller tile's emptier last item.
pub(super) const GROUP_ROUTES: u32 = 32;
pub(super) const GROUP_ROUTES_WIDE: u32 = 128;
pub(super) const GROUP_WIDE_AT: u32 = 64;

pub(super) const fn group_routes(fan: &Selected, experts: u32) -> u32 {
    let per = fan.route_count / experts;
    if per >= GROUP_WIDE_AT {
        GROUP_ROUTES_WIDE
    } else if per >= GROUP_ROUTES {
        64
    } else {
        GROUP_ROUTES
    }
}
/// Weight rows a grouped block takes off the matrix core (`kSimtTileN`),
/// and its threads.
pub(super) const GROUP_TILE_N: u32 = 128;
/// Weight rows a tall grouped block takes (four 16-row blocks a wave, eight
/// waves).
pub(super) const TALL_ROWS: u32 = 512;
pub(super) const TALL_MIN_K: u32 = 2048;
pub(super) const GROUP_BLOCK: u32 = 256;

/// A matrix-core grouped launch over `items` items of `routes` routes: 32
/// weight rows a wave, eight waves at 128 routes (`kGroupWaves`), else four.
pub(super) fn group_launch(items: u32, routes: u32, n: u32) -> Launch {
    let waves = if routes > GROUP_ROUTES { 8 } else { 4 };
    Launch::grid([items, n.div_ceil(32 * waves), 1], [64 * waves, 1, 1])
}

/// The slices of one expert a work item can name: 16 bits of it.
pub(super) const ORDER_SLICES: u32 = 65_535;

pub(super) fn route_order(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    fan: &Selected,
    experts: u32,
    group: u32,
) -> Result<RouteOrder, Error> {
    const ROUTE_ORDER: &str = "moe_route_order";
    const ORDER_BLOCK: u32 = 1024;
    // A work item packs its expert's slice in 16 bits (`e * 65536 + s`,
    // `linear/quant.hip`): one expert may take every route.
    if fan.route_count.div_ceil(group.max(1)) > ORDER_SLICES {
        return Err(refuse(
            op,
            format!(
                "{} routes in items of {group} could give one expert more than the {ORDER_SLICES} \
                 slices a work item names",
                fan.route_count
            ),
        ));
    }
    let offsets_at = (fan.route_count as usize).next_multiple_of(64);
    // An expert's last item is its only short one, and only an expert with a
    // route has one.
    let work_cap = fan.route_count.div_ceil(group) + experts.min(fan.route_count);
    let work_at = (offsets_at + experts as usize + 2).next_multiple_of(64);
    // Past a few spans of routes, the spans bucket their own routes across
    // the card and the one block only prefixes their counts.
    let spans = fan.route_count.div_ceil(ORDER_SPAN);
    let spans = if spans >= ORDER_SPANS_FROM { spans } else { 0 };
    let hist_at = (work_at + work_cap as usize).next_multiple_of(64);
    let order = ctx.scratch_stream(
        op,
        ROUTE_ORDER,
        (hist_at + spans as usize * (experts as usize + 1)) * core::mem::size_of::<i32>(),
    )? as usize as u64;
    let offsets = order + (offsets_at * core::mem::size_of::<i32>()) as u64;
    let work = order + (work_at * core::mem::size_of::<i32>()) as u64;
    let hist = if spans > 0 {
        order + (hist_at * core::mem::size_of::<i32>()) as u64
    } else {
        0
    };
    let span = |entry: &str, order: Option<u64>| -> Result<(), Error> {
        let mut args = vec![routes.arg(), ArgValue::Ptr(hist)];
        args.extend(order.map(ArgValue::Ptr));
        args.extend([
            fan.top_k.arg(),
            stated(op, fan.route_count)?.arg(),
            stated(op, experts)?.arg(),
            ctx.stage(),
        ]);
        ctx.fire(
            op,
            Fire::at(kernel("linear/quant.hip", entry))
                .apply(Launch::grid([spans, 1, 1], [256, 1, 1]).smem((experts + 1) * 4)),
            &args,
        )
    };
    if spans > 0 {
        span("::pie::linear::moe_route_span_count", None)?;
    }
    ctx.fire(
        op,
        Fire::at(kernel("linear/quant.hip", "::pie::linear::moe_route_order"))
            .apply(Launch::grid([1, 1, 1], [ORDER_BLOCK, 1, 1]).smem(2 * (experts + 1) * 4)),
        &[
            routes.arg(),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            stated(op, work_cap)?.arg(),
            group.arg(),
            fan.top_k.arg(),
            stated(op, fan.route_count)?.arg(),
            stated(op, experts)?.arg(),
            ctx.stage(),
            ArgValue::Ptr(hist),
            spans.arg(),
        ],
    )?;
    if spans > 0 {
        span("::pie::linear::moe_route_span_scatter", Some(order))?;
    }
    Ok(RouteOrder {
        order,
        offsets,
        work,
        work_cap,
        routes: group,
    })
}

/// Routes a span of the spread ordering buckets (`kOrderSpan`), and the
/// spans from which it takes over from the one block.
pub(super) const ORDER_SPAN: u32 = 2048;
pub(super) const ORDER_SPANS_FROM: u32 = 8;

/// The grouped dense leg: routes ordered by expert, then 32 or 128 of one expert's
/// routes against 128 of its weight rows a block on the matrix core.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_grouped(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: u32,
    fan: &Selected,
    table: ExpertTable,
) -> Result<(), Error> {
    let ordered = route_order(ctx, op, routes, fan, experts, group_routes(fan, experts))?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    select_grouped_launch(ctx, op, x, bank, y, experts, &ordered, act_div, 0, table)
}

/// The grouped legs' row layouts (`SORTED` of the kernels): the activation's
/// rows already in expert order (route i of the ordering at row i), and the
/// products landed in expert order; otherwise each is by route.
#[cfg(feature = "rocm")]
pub(super) const SORT_ACT: u32 = 1;
#[cfg(feature = "rocm")]
pub(super) const SORT_OUT: u32 = 2;

/// The grouped dense leg over an ordering already made.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_grouped_launch(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    y: &mut Tensor,
    experts: u32,
    ordered: &RouteOrder,
    act_div: i32,
    sorted: u32,
    table: ExpertTable,
) -> Result<(), Error> {
    grouped_launch(
        ctx, op, x, bank, y, experts, ordered, act_div, sorted, None, table,
    )
}

/// [`select_grouped_launch`]; with `glu` (the swiglu's clamp, if any) over a
/// packed [gate | up] bank of 2·`y.width` rows on the tall block (which must
/// take it), landing the swiglu's rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn grouped_launch(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    y: &mut Tensor,
    experts: u32,
    ordered: &RouteOrder,
    act_div: i32,
    sorted: u32,
    glu: Option<Option<f32>>,
    table: ExpertTable,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let k = stated(op, x.width)?;
    let rows =
        nonzero(op, "N, the bank's output width", y.width)? * if glu.is_some() { 2 } else { 1 };
    let n = stated(op, rows)?;
    let tall = takes_tall(ordered, rows, x.width);
    debug_assert!(tall || glu.is_none(), "a GLU launch takes the tall block");
    let (glu_kind, limit) = match glu {
        None => (0, 0.0),
        Some(None) => (1, 0.0),
        Some(Some(limit)) => (2, limit),
    };
    let (entry, launch) = if tall {
        (
            format!(
                "::pie::linear::moe_matmul_select_grouped_tall<{t}, ::pie::i32({}), ::pie::i32(4), ::pie::i32(8), ::pie::i32({sorted}), ::pie::i32({glu_kind})>",
                ordered.routes
            ),
            Launch::grid([ordered.work_cap, rows / TALL_ROWS, 1], [64 * 8, 1, 1]),
        )
    } else {
        (
            format!(
                "::pie::linear::moe_matmul_select_grouped<{t}, ::pie::i32({}), ::pie::i32({sorted})>",
                ordered.routes
            ),
            group_launch(ordered.work_cap, ordered.routes, y.width),
        )
    };
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &entry)).apply(launch),
        &[
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
            ArgValue::Ptr(ordered.work),
            bank.arg(),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(table.table),
            ArgValue::Ptr(table.hits),
        ]
        .into_iter()
        .chain(tall.then(|| limit.arg()))
        .collect::<Vec<_>>(),
    )
}

/// Whether a grouped launch over `rows` bank rows and a `k` contraction
/// takes the tall block: 128-route items over rows in whole 512s (64 weight
/// rows a wave, so each activation piece read from LDS feeds twice the
/// products; narrower banks would idle its rows), and only over long K (a
/// short one ran no faster in the serve).
pub(super) const fn takes_tall(ordered: &RouteOrder, rows: u32, k: u32) -> bool {
    ordered.routes == GROUP_ROUTES_WIDE && rows.is_multiple_of(TALL_ROWS) && k >= TALL_MIN_K
}

/// The routed MLP with its routes kept in expert order between the two
/// matmuls: one route ordering, the gate-up leg reading the tokens' rows by
/// route and landing its products in expert order, the swiglu over those
/// rows, the down leg reading them in that order and landing its products by
/// route for the weighted fold. Against the four steps it skips the gate-up
/// products' and the activation's walks by route (and a second ordering).
/// Taken from `kernel.moe.ordered_from` routes an expert, a device profile
/// decision.
#[cfg(feature = "rocm")]
pub(super) mod ordered {
    use super::{
        FILE, SORT_ACT, SORT_OUT, Selected, dense_experts, group_routes, grouped_launch,
        grouped_worth, route_order, select_grouped_launch, selected, takes_tall, weighted_sum,
    };
    use crate::error::Error;
    use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, kernel, stated};
    use crate::tensor::Tensor;

    /// The routed swiglu MLP in expert order, when both legs are the grouped
    /// dense ones and an expert averages `kernel.moe.ordered_from` routes or
    /// more (0: never). Whether it ran.
    pub(in crate::linear::moe) fn mlp(
        ctx: &Ctx,
        op: &'static str,
        p: &super::RoutedMlp,
        y: &mut Tensor,
    ) -> Result<bool, Error> {
        let (x, i) = (p.x, p.intermediate);
        let fan = selected(op, x, p.routes, &p.packed)?;
        let (Some(experts), Some(down_experts)) = (
            grouped_worth(ctx, x, p.gate_up, &p.packed, &fan, p.gate_up_experts),
            dense_experts(p.h, p.down, &p.routed, p.down_experts),
        ) else {
            return Ok(false);
        };
        let hidden = y.width;
        if experts != down_experts
            || p.packed.width != 2 * i
            || p.routed.width != hidden
            || ![x.width, i, hidden].iter().all(|w| w.is_multiple_of(8))
        {
            return Ok(false);
        }
        let from = ctx.tuning().moe_ordered_from();
        if from == 0 || fan.route_count / experts < from {
            return Ok(false);
        }
        pipeline(ctx, op, p, &fan, experts, y)?;
        Ok(true)
    }

    fn pipeline(
        ctx: &Ctx,
        op: &'static str,
        p: &super::RoutedMlp,
        fan: &Selected,
        experts: u32,
        y: &mut Tensor,
    ) -> Result<(), Error> {
        let (x, i) = (p.x, p.intermediate);
        let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
        let rows = fan.route_count as usize;
        let elem = 2usize;
        let h = ctx.scratch_stream(op, "linear.moe_ordered_h", rows * i as usize * elem)? as usize
            as u64;
        let ordered = route_order(ctx, op, p.routes, fan, experts, group_routes(fan, experts))?;
        let act_div = if fan.by_token { fan.top_k } else { 1 };
        let mut h_t = Tensor::new(h, fan.route_count, i, x.dtype);
        // Gate-up: the tokens' rows by route in, the products in expert
        // order; on the tall block its swiglu lands them, else a pass does.
        if takes_tall(&ordered, 2 * i, x.width) {
            grouped_launch(
                ctx,
                op,
                x,
                p.gate_up,
                &mut h_t,
                experts,
                &ordered,
                act_div,
                SORT_OUT,
                Some(p.limit),
                p.gate_up_experts,
            )?;
        } else {
            let packed = ctx.scratch_stream(
                op,
                "linear.moe_ordered_packed",
                rows * 2 * i as usize * elem,
            )? as usize as u64;
            let mut packed_t = Tensor::new(packed, fan.route_count, 2 * i, x.dtype);
            select_grouped_launch(
                ctx,
                op,
                x,
                p.gate_up,
                &mut packed_t,
                experts,
                &ordered,
                act_div,
                SORT_OUT,
                p.gate_up_experts,
            )?;
            let clamp = if p.limit.is_some() { "true" } else { "false" };
            ctx.fire(
                op,
                Fire::at(kernel(
                    FILE,
                    &format!("::pie::linear::moe_swiglu_ordered<{t}, {clamp}>"),
                ))
                .apply(Launch::grid(
                    [fan.route_count, i.div_ceil(8 * 256), 1],
                    [256, 1, 1],
                )),
                &[
                    ArgValue::Ptr(packed),
                    ArgValue::Ptr(h),
                    ArgValue::Ptr(ordered.offsets),
                    stated(op, experts)?.arg(),
                    stated(op, i)?.arg(),
                    p.limit.unwrap_or(f32::INFINITY).arg(),
                ],
            )?;
        }
        // Down: the ordered rows in, the products by route.
        let mut routed = p.routed;
        select_grouped_launch(
            ctx,
            op,
            h_t,
            p.down,
            &mut routed,
            experts,
            &ordered,
            1,
            SORT_ACT,
            p.down_experts,
        )?;
        weighted_sum(ctx, p.routed, p.weights, y)
    }
}
