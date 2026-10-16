use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;

const FILE: &str = "linear/moe.hip";

const BLOCK: u32 = 256;

const MAX_EXPERTS: u32 = 512;

/// A router is one wavefront a token row, each lane landing one pick.
const ROUTER_LANES: u32 = 64;

const fn router_lane(rows: u32) -> Launch {
    Launch::per_row(rows, ROUTER_LANES)
}

fn router_fan(op: &'static str, fan: u32) -> Result<(), Error> {
    if fan > ROUTER_LANES {
        return Err(refuse(
            op,
            format!("the fan-out is {fan}, above the {ROUTER_LANES} picks one wavefront lands"),
        ));
    }
    Ok(())
}

fn elementwise_rows(op: &'static str, rows: u32, width: u32) -> Result<Launch, Error> {
    nonzero(op, "rows", rows)?;
    nonzero(op, "width", width)?;
    Ok(Launch::grid(
        [rows, width.div_ceil(BLOCK), 1],
        [BLOCK, 1, 1],
    ))
}

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

fn router_extents(
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
    const OP: &str = "linear.moe_topk_softmax_scaled";
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
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::moe_topk_softmax_scaled<{t}>")),
        )
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
) -> Result<(), Error> {
    ranked_planes(op, logits, top_k, routes, weights);
    let (e, k) = router_extents(op, logits, experts, top_k)?;
    ctx.fire(
        op,
        Fire::at(file, entrypoint).apply(router_lane(logits.rows)),
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
    const OP: &str = "linear.moe_topk_sigmoid";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ranked_router(
        ctx,
        OP,
        FILE,
        symbol(&format!("::pie::linear::moe_topk_sigmoid<{t}>")),
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
    const OP: &str = "linear.moe_topk_sigmoid_sink";
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
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::moe_topk_sigmoid_sink<{t}>")),
        )
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

struct Selected {
    route_count: u32,

    top_k: i32,

    by_token: bool,
}

fn selected(op: &'static str, x: Tensor, routes: Tensor, y: &Tensor) -> Result<Selected, Error> {
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertTable {
    pub table: u64,
    pub hits: u64,
}

impl ExpertTable {
    pub const RESIDENT: ExpertTable = ExpertTable { table: 0, hits: 0 };

    #[must_use]
    pub const fn streams(&self) -> bool {
        self.table != 0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupSeat {
    pub cell: u64,
    pub hits: u64,
}

impl GroupSeat {
    pub const RESIDENT: GroupSeat = GroupSeat { cell: 0, hits: 0 };

    #[must_use]
    pub const fn streams(&self) -> bool {
        self.cell != 0
    }
}

pub fn matmul_select(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select";

    let fan = selected(OP, x, routes, y)?;
    // Past 64 routes an expert at a decode batch, a bf16 bank on the matrix
    // core's GEMM tile: items of 128 routes against 128 weight rows.
    if let Some(count) = dense_experts(x, bank, y, experts)
        && x.dtype == Dtype::Bf16
        && x.width.is_multiple_of(64)
        && count <= ORDER_EXPERT_CAP
        && (65..=ctx.tuning().moe_dense_batch_until).contains(&fan.route_count.div_ceil(count))
    {
        return select_gemm(ctx, OP, x, bank, routes, y, &fan, count);
    }
    // A wide decode batch over a resident bank: an item an expert's routes,
    // its weight read once.
    if let Some(count) = dense_experts(x, bank, y, experts)
        && let Some(mw) = wide_items(ctx, &fan, count, x)
    {
        return select_wide(ctx, OP, x, bank, routes, y, &fan, count, mw);
    }
    // A decode batch over a resident bank: each expert read once for all its
    // routes.
    if let Some(count) = dense_experts(x, bank, y, experts)
        && x.width.is_multiple_of(8)
        && let Some(mt) = batch_items(ctx, &fan, count, 64, ctx.tuning().moe_dense_batch_until)
    {
        let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
        let nbw = dense_nbw(mt);
        return select_batch(
            ctx,
            OP,
            FILE,
            &format!(
                "::pie::linear::moe_matmul_select_dense_batch<{t}, ::pie::i32({mt}), ::pie::i32({nbw})>"
            ),
            x,
            &[bank.arg()],
            routes,
            y,
            &fan,
            count,
            mt,
            nbw,
            None,
        );
    }
    if let Some(count) = grouped_worth(ctx, x, bank, y, &fan, experts) {
        if x.width.is_multiple_of(8) {
            #[cfg(feature = "rocm")]
            if select_lt(ctx, OP, x, bank, routes, y, count, &fan)? {
                return Ok(());
            }
            return select_grouped(ctx, OP, x, bank, routes, y, count, &fan);
        }
        // A row the native legs cannot read in 16-byte pieces: rocBLAS's
        // batched GEMM, eagerly. Where both run the native leg is several
        // times the faster.
        #[cfg(feature = "rocm")]
        if ctx.stage() == ArgValue::Ptr(0) && blas_worth(x, y, &fan, count) {
            return select_grouped_blas(ctx, OP, x, bank, routes, y, count);
        }
    }
    select_gemv(ctx, OP, x, bank, routes, y, experts)
}

/// A grouped projection (`linear.matmul_grouped`): row r's group g, x[r·groups
/// + g] (k wide), through the bank's expert routes[r][g] (n x k) into y[r·groups
/// + g]. `identity` states the routes are `group_routes`' (route g for group
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
        && dense_experts(x, bank, y, experts) == Some(groups)
        && routes.rows > ctx.tuning().moe_batch_until
        && x.rows == routes.rows * groups
        && y.rows == x.rows
        && grouped_lt(ctx, OP, x.ptr, bank.ptr, y.ptr, routes.rows, y.width, x.width, groups)?
    {
        return Ok(());
    }
    matmul_select(ctx, x, bank, routes, y, experts)
}

/// `groups` strided bf16 matmuls in one hipBLASLt launch; `Ok(false)` where
/// hipBLASLt is absent or offers no algorithm.
#[allow(clippy::too_many_arguments)]
fn grouped_lt(
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
    };
    let workspace = ctx.scratch_shared(op, "linear.lt_workspace", lt::WORKSPACE)? as u64;
    let start = crate::prof::on()
        .then(|| crate::prof::begin(ctx.stream()))
        .flatten();
    // SAFETY: x is [m, groups·k], y [m, groups·n] and w groups blocks of
    // [n, k], device planes the caller sized; the workspace is this stream's.
    let landed = unsafe { lt::matmul(shape, x, w, 0, y, workspace, ctx.stream()) };
    crate::prof::end(ctx.stream(), start, || {
        format!("{op} hipblaslt groups={groups} m={m} n={n} k={k}")
    });
    landed.map_err(|why| refuse(op, format!("{why} at {groups} x M={m} N={n} K={k}")))
}

/// The most experts `moe_route_order` buckets in its LDS.
const ORDER_EXPERT_CAP: u32 = 4096;

/// The experts of a resident 16-bit bank the expert-ordered legs can
/// bucket, if it is one.
fn dense_experts(x: Tensor, bank: Tensor, y: &Tensor, experts: ExpertTable) -> Option<u32> {
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

fn grouped_worth(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    y: &Tensor,
    fan: &Selected,
    experts: ExpertTable,
) -> Option<u32> {
    if !ctx.tuning().moe_grouped {
        return None;
    }
    let count = dense_experts(x, bank, y, experts)?;
    (fan.route_count.div_ceil(count) >= ctx.tuning().moe_group_at).then_some(count)
}

/// 16-row weight blocks a batch wave takes (the kernels' `NBW`), and so the
/// rows a batch block takes.
const BATCH_NBW: u32 = 1;

/// 16-row weight blocks a dense batch wave takes for `mt`-route items: two
/// at 64, each activation piece read from LDS then feeding twice the products.
const fn dense_nbw(mt: u32) -> u32 {
    if mt >= 64 { DENSE_WIDE_NBW } else { BATCH_NBW }
}
const DENSE_WIDE_NBW: u32 = 2;

/// The routes a batch item takes, if the batch legs take this fire: from
/// `kernel.moe.batch_from` tokens' routes (one token's read each expert once
/// on the stream, which orders nothing), while the experts average (routes over
/// experts, rounded up) at most `until`. An item is the least of 4, 8, 16, 32
/// and (a bank whose kernels take it, `widest`) 64 routes past the average by
/// two of its square roots and one, so the busier experts of a random routing
/// still read their weight once, and an item stages no more empty rows than
/// that takes; past them an expert takes several of the widest.
fn batch_items(ctx: &Ctx, fan: &Selected, experts: u32, widest: u32, until: u32) -> Option<u32> {
    let t = ctx.tuning();
    let top_k = fan.top_k.unsigned_abs();
    if !t.moe_batch || fan.route_count < t.moe_batch_from.saturating_mul(top_k) {
        return None;
    }
    let per = fan.route_count.div_ceil(experts);
    if per > until {
        return None;
    }
    // per + 2·√per + 1 = (√per + 1)², so an item of `mt` covers it while
    // √per + 1 ≤ √mt, i.e. per ≤ (√mt - 1)².
    Some(match per {
        0..=1 => 4,
        2..=3 => 8,
        4..=9 => 16,
        10..=25 => 32,
        _ => widest,
    })
}

/// A batch fire: routes ordered by expert into `mt`-route items, then one
/// item a block row against 64·nbw weight rows a block. `planes` are
/// the bank's arguments, between the work list and the result; a packed
/// bank's seat follows the extents.
#[allow(clippy::too_many_arguments)]
fn select_batch(
    ctx: &Ctx,
    op: &'static str,
    file: &'static str,
    entry: &str,
    x: Tensor,
    planes: &[ArgValue],
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    experts: u32,
    mt: u32,
    nbw: u32,
    seat: Option<GroupSeat>,
) -> Result<(), Error> {
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let ordered = route_order(ctx, op, routes, fan, experts, mt)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let mut args = vec![
        x.arg(),
        ArgValue::Ptr(ordered.order),
        ArgValue::Ptr(ordered.offsets),
        ArgValue::Ptr(ordered.work),
    ];
    args.extend_from_slice(planes);
    args.extend([
        y.arg(),
        act_div.arg(),
        n.arg(),
        k.arg(),
        stated(op, experts)?.arg(),
    ]);
    if let Some(seat) = seat {
        args.extend([ArgValue::Ptr(seat.cell), ArgValue::Ptr(seat.hits)]);
    }
    ctx.fire(
        op,
        Fire::at(file, symbol(entry)).apply(Launch::grid(
            [ordered.work_cap, y.width.div_ceil(64 * nbw), 1],
            [GROUP_BLOCK, 1, 1],
        )),
        &args,
    )
}

/// Weight rows a wide block takes (the kernel's `NT`).
const WIDE_ROWS: u32 = 64;
const WIDE_ROWS_TALL: u32 = 128;
/// Elements a wide stage takes of a row (the kernel's `KC`).
const WIDE_K: u32 = 64;

/// The routes a wide wave takes (the kernel's `MW`; an item is four waves'),
/// if the wide leg takes this fire: from `kernel.moe.dense_wide_from` routes
/// an expert (averaged, rounded up) to `kernel.moe.dense_batch_until`, K
/// whole 64-element stages and the activation under 4 GiB. The item is the
/// least of 64, 128, 192 and 256 routes past the average by two of its
/// square roots and one (as `batch_items`), so each expert's routes are one
/// item.
fn wide_items(ctx: &Ctx, fan: &Selected, experts: u32, x: Tensor) -> Option<u32> {
    let t = ctx.tuning();
    let top_k = fan.top_k.unsigned_abs();
    if !t.moe_batch
        || t.moe_dense_wide_from == 0
        || fan.route_count < t.moe_batch_from.saturating_mul(top_k)
        || !x.width.is_multiple_of(WIDE_K)
        || u64::from(x.rows) * u64::from(x.width) * 2 >= 0xffff_f000
    {
        return None;
    }
    let per = fan.route_count.div_ceil(experts);
    if per < t.moe_dense_wide_from || per > t.moe_dense_batch_until {
        return None;
    }
    // per ≤ (√cap − 1)² for cap = 64, 128, 192, 256.
    Some(match per {
        0..=49 => 16,
        50..=106 => 32,
        107..=165 => 48,
        _ => 64,
    })
}

/// A bf16 bank's select on the GEMM tile (`linear/fp8.hip`'s eight-wave
/// tile at bf16): routes ordered by expert into items of 128, each against
/// 128 weight rows a block.
#[allow(clippy::too_many_arguments)]
fn select_gemm(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    experts: u32,
) -> Result<(), Error> {
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let ordered = route_order(ctx, op, routes, fan, experts, 128)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    // One slice of K: slicing measured no faster (GLM-5.3's mini, 96 to 160
    // routes an expert).
    ctx.fire(
        op,
        Fire::at("linear/fp8.hip", "::pie::linear::moe_bf16_gemm8")
            .apply(Launch::grid([ordered.work_cap, y.width.div_ceil(128), 1], [2 * GROUP_BLOCK, 1, 1])),
        &[
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
            ArgValue::Ptr(ordered.work),
            bank.arg(),
            y.arg(),
            ArgValue::ABSENT,
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            stated(op, y.rows)?.arg(),
            stated(op, 2 * x.width / 128)?.arg(),
        ],
    )
}

/// A wide fire: routes ordered by expert into items of up to 4·`mw`, then
/// one item a block row against 64 or 128 weight rows a block.
#[allow(clippy::too_many_arguments)]
fn select_wide(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    experts: u32,
    mw: u32,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let ordered = route_order(ctx, op, routes, fan, experts, 4 * mw)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    #[allow(clippy::cast_possible_truncation)]
    let act_bytes = (u64::from(x.rows) * u64::from(x.width) * 2) as u32;
    // Items an XCD takes (the kernel's `per_xcd`).
    let per_xcd = ordered.work_cap.div_ceil(8);
    // 128-row tiles once the items are 128 or 192 routes (the activation
    // read half as often), their weight two stages ahead at 192 (so the
    // tile fits the registers); 64 rows otherwise, four stages ahead.
    let nt = if (32..=48).contains(&mw) {
        WIDE_ROWS_TALL
    } else {
        WIDE_ROWS
    };
    let depth = if nt == WIDE_ROWS_TALL && mw >= 48 { 2 } else { 4 };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::moe_matmul_select_dense_wide<{t}, ::pie::i32({mw}), ::pie::i32({nt}), \
                 ::pie::i32({WIDE_K}), ::pie::i32({depth}), ::pie::i32(2)>"
            )),
        )
        .apply(Launch::grid(
            [8 * per_xcd * y.width.div_ceil(nt), 1, 1],
            [GROUP_BLOCK, 1, 1],
        )),
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
            act_bytes.arg(),
            per_xcd.arg(),
        ],
    )
}

/// The planes `moe_route_order` lands: routes in expert order, each
/// expert's run, and the work list of `routes`-route items.
struct RouteOrder {
    order: u64,
    offsets: u64,
    work: u64,
    work_cap: u32,
    routes: u32,
}

/// Routes a grouped work item takes: 32, or 128 once the experts average
/// `GROUP_WIDE_AT` routes, where reading each expert's weight a quarter as
/// often outweighs the taller tile's emptier last item.
const GROUP_ROUTES: u32 = 32;
const GROUP_ROUTES_WIDE: u32 = 128;
const GROUP_WIDE_AT: u32 = 64;

const fn group_routes(fan: &Selected, experts: u32) -> u32 {
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
const GROUP_TILE_N: u32 = 128;
/// Weight rows a tall grouped block takes (four 16-row blocks a wave, eight
/// waves).
const TALL_ROWS: u32 = 512;
const TALL_MIN_K: u32 = 2048;
const GROUP_BLOCK: u32 = 256;

/// A matrix-core grouped launch over `items` items of `routes` routes: 32
/// weight rows a wave, eight waves at 128 routes (`kGroupWaves`), else four.
fn group_launch(items: u32, routes: u32, n: u32) -> Launch {
    let waves = if routes > GROUP_ROUTES { 8 } else { 4 };
    Launch::grid([items, n.div_ceil(32 * waves), 1], [64 * waves, 1, 1])
}

fn route_order(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    fan: &Selected,
    experts: u32,
    group: u32,
) -> Result<RouteOrder, Error> {
    const ROUTE_ORDER: &str = "moe_route_order";
    const ORDER_BLOCK: u32 = 1024;
    let offsets_at = (fan.route_count as usize).next_multiple_of(64);
    // An expert's last item is its only short one, and only an expert with a
    // route has one.
    let work_cap = fan.route_count.div_ceil(group) + experts.min(fan.route_count);
    let work_at = (offsets_at + experts as usize + 2).next_multiple_of(64);
    let order = ctx.scratch(
        op,
        ROUTE_ORDER,
        (work_at + work_cap as usize) * core::mem::size_of::<i32>(),
    )? as usize as u64;
    let offsets = order + (offsets_at * core::mem::size_of::<i32>()) as u64;
    let work = order + (work_at * core::mem::size_of::<i32>()) as u64;
    ctx.fire(
        op,
        Fire::at("linear/quant.hip", symbol("::pie::linear::moe_route_order"))
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
        ],
    )?;
    Ok(RouteOrder {
        order,
        offsets,
        work,
        work_cap,
        routes: group,
    })
}

/// The grouped dense leg: routes ordered by expert, then 32 or 128 of one expert's
/// routes against 128 of its weight rows a block on the matrix core.
#[allow(clippy::too_many_arguments)]
fn select_grouped(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: u32,
    fan: &Selected,
) -> Result<(), Error> {
    let ordered = route_order(ctx, op, routes, fan, experts, group_routes(fan, experts))?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    select_grouped_launch(ctx, op, x, bank, y, experts, &ordered, act_div, 0)
}

/// The grouped legs' row layouts (`SORTED` of the kernels): the activation's
/// rows already in expert order (route i of the ordering at row i), and the
/// products landed in expert order; otherwise each is by route.
const SORT_ACT: u32 = 1;
const SORT_OUT: u32 = 2;

/// The grouped dense leg over an ordering already made.
#[allow(clippy::too_many_arguments)]
fn select_grouped_launch(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    y: &mut Tensor,
    experts: u32,
    ordered: &RouteOrder,
    act_div: i32,
    sorted: u32,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    // 128-route items over a bank whose rows come in whole 512s take the
    // tall block: 64 weight rows a wave, so each activation piece read from
    // LDS feeds twice the products (GLM's 4096-row legs: 461 -> 505
    // TFLOP/s at an 8192-token prefill); narrower banks would idle its rows.
    // Only over long K: Qwen3.8-Flash-Next's 640-deep down leg ran no faster
    // in the serve (its mid-in TPOT 4.8 -> 5.4 ms) for a 10% microbench gain.
    let tall = ordered.routes == GROUP_ROUTES_WIDE
        && y.width.is_multiple_of(TALL_ROWS)
        && x.width >= TALL_MIN_K;
    let (entry, launch) = if tall {
        (
            format!(
                "::pie::linear::moe_matmul_select_grouped_tall<{t}, ::pie::i32({}), ::pie::i32(4), ::pie::i32(8), ::pie::i32({sorted})>",
                ordered.routes
            ),
            Launch::grid([ordered.work_cap, y.width / TALL_ROWS, 1], [64 * 8, 1, 1]),
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
        Fire::at(FILE, symbol(&entry)).apply(launch),
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
        ],
    )
}

/// The routed MLP with its routes kept in expert order between the two
/// matmuls: one route ordering, the gate-up leg reading the tokens' rows by
/// route and landing its products in expert order, the swiglu over those
/// rows, the down leg reading them in that order and landing its products by
/// route for the weighted fold. Against the four steps it skips the gate-up
/// products' and the activation's walks by route (and a second ordering).
/// Taken from `kernel.moe.ordered_from` routes an expert, a device profile
/// decision measured offline (`bench_moe_mlp_paths` in tests/lt_grouped.rs).
#[cfg(feature = "rocm")]
mod ordered {
    use super::{
        FILE, Selected, dense_experts, group_routes, grouped_worth, route_order, select_grouped_launch,
        selected, weighted_sum, SORT_ACT, SORT_OUT,
    };
    use crate::error::Error;
    use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, stated, symbol};
    use crate::tensor::Tensor;

    /// The routed swiglu MLP in expert order, when both legs are the grouped
    /// dense ones and an expert averages `kernel.moe.ordered_from` routes or
    /// more (0: never). Whether it ran.
    pub(super) fn mlp(
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
        let from = ctx.tuning().moe_ordered_from;
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
        let packed = ctx.scratch(op, "linear.moe_ordered_packed", rows * 2 * i as usize * elem)? as usize as u64;
        let h = ctx.scratch(op, "linear.moe_ordered_h", rows * i as usize * elem)? as usize as u64;
        let ordered = route_order(ctx, op, p.routes, fan, experts, group_routes(fan, experts))?;
        let act_div = if fan.by_token { fan.top_k } else { 1 };
        // Gate-up: the tokens' rows by route in, the products in expert order.
        let mut packed_t = Tensor::new(packed, fan.route_count, 2 * i, x.dtype);
        select_grouped_launch(ctx, op, x, p.gate_up, &mut packed_t, experts, &ordered, act_div, SORT_OUT)?;
        // The swiglu over the ordered rows.
        let clamp = if p.limit.is_some() { "true" } else { "false" };
        ctx.fire(
            op,
            Fire::at(FILE, symbol(&format!("::pie::linear::moe_swiglu_ordered<{t}, {clamp}>")))
                .apply(Launch::grid([fan.route_count, i.div_ceil(8 * 256), 1], [256, 1, 1])),
            &[
                ArgValue::Ptr(packed),
                ArgValue::Ptr(h),
                ArgValue::Ptr(ordered.offsets),
                stated(op, experts)?.arg(),
                stated(op, i)?.arg(),
                p.limit.unwrap_or(f32::INFINITY).arg(),
            ],
        )?;
        // Down: the ordered rows in, the products by route.
        let h_t = Tensor::new(h, fan.route_count, i, x.dtype);
        let mut routed = p.routed;
        select_grouped_launch(ctx, op, h_t, p.down, &mut routed, experts, &ordered, 1, SORT_ACT)?;
        weighted_sum(ctx, p.routed, p.weights, y)
    }
}

/// The hipBLASLt grouped leg forced from outside, for tests and benches:
/// `Some(true)` takes it wherever it launches, past the profile's
/// thresholds; `Some(false)` never.
#[cfg(feature = "rocm")]
static LT_FORCED: std::sync::Mutex<Option<bool>> = std::sync::Mutex::new(None);

#[cfg(feature = "rocm")]
pub fn force_lt_grouped(on: Option<bool>) {
    *LT_FORCED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = on;
}

/// Grouped GEMMs the hipBLASLt leg has launched, for tests.
#[cfg(feature = "rocm")]
static LT_FIRES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "rocm")]
pub fn lt_grouped_fires() -> u64 {
    LT_FIRES.load(core::sync::atomic::Ordering::Relaxed)
}

/// The most the hipBLASLt leg stages: its routes' rows in and out.
#[cfg(feature = "rocm")]
const LT_STAGE_CAP: u64 = 3 << 30;

/// Whether the hipBLASLt leg takes `per` routes an expert where the profile
/// starts it at `from` (0: never) and the shape is of the class it takes
/// (`fits`), or as forced.
#[cfg(feature = "rocm")]
fn lt_wanted(ctx: &Ctx, per: u32, from: u32, fits: bool) -> bool {
    match *LT_FORCED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        Some(forced) => forced,
        None => ctx.tuning().moe_lt_grouped && fits && from > 0 && per >= from,
    }
}

/// The library and the solution the profile names for a contraction of `k`
/// (`kernel.moe.lt_tile_deep` from `kernel.moe.lt_deep_k`, else
/// `kernel.moe.lt_tile_shallow`); `None` where either is missing.
#[cfg(feature = "rocm")]
fn lt_solution(
    ctx: &Ctx,
    dtype: Dtype,
    k: u32,
) -> Option<(&'static crate::hip::lt_grouped::Library, usize)> {
    let lib = crate::hip::lt_grouped::library(dtype).ok()?;
    let t = ctx.tuning();
    let tile = if k >= t.moe_lt_deep_k {
        t.moe_lt_tile_deep
    } else {
        t.moe_lt_tile_shallow
    };
    Some((lib, lib.pick(tile)?))
}

/// The hipBLASLt grouped leg (`hip::lt_grouped`) of a routed select: routes
/// ordered by expert, their rows gathered by expert while the same launch
/// writes each expert's device `UserArguments`, one grouped GEMM over a
/// worst-case grid, and the rows scattered back to route order. From
/// `kernel.moe.lt_select_from` routes an expert; `Ok(false)` leaves the fire
/// to the native grouped leg.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
fn select_lt(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: u32,
    fan: &Selected,
) -> Result<bool, Error> {
    let (n, k, rows) = (y.width, x.width, fan.route_count);
    if !matches!(x.dtype, Dtype::Bf16 | Dtype::F16)
        || bank.dtype != x.dtype
        || y.dtype != x.dtype
        || !n.is_multiple_of(8)
        || n < 64
        || u64::from(rows) * (u64::from(n) + u64::from(k)) * 2 > LT_STAGE_CAP
        || !lt_wanted(ctx, rows / experts, ctx.tuning().moe_lt_select_from, true)
    {
        return Ok(false);
    }
    let Some((lib, at)) = lt_solution(ctx, x.dtype, k) else {
        return Ok(false);
    };
    let stage = stage_lt(ctx, op, x, bank, routes, n, experts, fan, 0)?;
    gemm_lt(op, lib, at, &stage, n, rows, experts, ctx.stream())?;
    unscatter_lt(ctx, op, &stage, y, experts, rows)?;
    Ok(true)
}

/// The hipBLASLt leg's planes: routes in expert order, the staged result
/// rows, and the `UserArguments` array (one record an expert and the
/// sentinel).
#[cfg(feature = "rocm")]
struct LtStage {
    ordered: RouteOrder,
    staged_out: u64,
    args: u64,
}

/// Orders the routes, gathers their rows by expert into a stream slab and
/// writes the experts' arguments (`pos`, when given, takes each route's row
/// in the order).
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
fn stage_lt(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    n: u32,
    experts: u32,
    fan: &Selected,
    pos: u64,
) -> Result<LtStage, Error> {
    use crate::hip::lt_grouped::USER_ARGS_BYTES;
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (k, rows) = (x.width, fan.route_count);
    let ordered = route_order(ctx, op, routes, fan, experts, GROUP_ROUTES_WIDE)?;
    let staged_in = ctx.scratch_stream(op, "linear.moe_lt_in", rows as usize * k as usize * 2)?
        as usize as u64;
    let staged_out = ctx.scratch_stream(op, "linear.moe_lt_out", rows as usize * n as usize * 2)?
        as usize as u64;
    let args = ctx.scratch(
        op,
        "linear.moe_lt_args",
        (experts as usize + 1) * USER_ARGS_BYTES,
    )? as usize as u64;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&format!("::pie::linear::moe_lt_scatter<{t}>")))
            .apply(Launch::grid([copy_blocks(rows, k), 1, 1], [256, 1, 1])),
        &[
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
            ArgValue::Ptr(staged_in),
            ArgValue::Ptr(args),
            ArgValue::Ptr(pos),
            bank.ptr.arg(),
            staged_out.arg(),
            act_div.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            stated(op, experts)?.arg(),
        ],
    )?;
    Ok(LtStage {
        ordered,
        staged_out,
        args,
    })
}

/// Blocks a 16-byte grid-stride copy of `rows` rows of `width` takes.
#[cfg(feature = "rocm")]
fn copy_blocks(rows: u32, width: u32) -> u32 {
    let pieces = u64::from(rows) * u64::from(width / 8);
    pieces.div_ceil(256).clamp(1, 304 * 8) as u32
}

/// One grouped GEMM over the arguments at `args`: `experts` groups and the
/// sentinel, a grid for `rows` routes however they fall.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
fn lt_launch(
    op: &'static str,
    lib: &crate::hip::lt_grouped::Library,
    at: usize,
    args: u64,
    n: u32,
    rows: u32,
    experts: u32,
    stream: *mut core::ffi::c_void,
) -> Result<(), Error> {
    let grid = lib.solutions[at].worst_grid(n, rows, experts);
    let start = crate::prof::on().then(|| crate::prof::begin(stream)).flatten();
    // SAFETY: the argument array was written on this stream, its records
    // naming staged planes and the bank, all live through the fire.
    unsafe { lib.launch(at, experts + 1, args, grid, stream) }.map_err(|why| refuse(op, why))?;
    crate::prof::end(stream, start, || {
        format!(
            "{op} hipblaslt grouped {} experts={experts} routes={rows} n={n}",
            lib.solutions[at].label()
        )
    });
    LT_FIRES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    Ok(())
}

#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
fn gemm_lt(
    op: &'static str,
    lib: &crate::hip::lt_grouped::Library,
    at: usize,
    stage: &LtStage,
    n: u32,
    rows: u32,
    experts: u32,
    stream: *mut core::ffi::c_void,
) -> Result<(), Error> {
    lt_launch(op, lib, at, stage.args, n, rows, experts, stream)
}

#[cfg(feature = "rocm")]
fn unscatter_lt(
    ctx: &Ctx,
    op: &'static str,
    stage: &LtStage,
    y: &Tensor,
    experts: u32,
    rows: u32,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&format!("::pie::linear::moe_lt_unscatter<{t}>")))
            .apply(Launch::grid([copy_blocks(rows, y.width), 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(stage.staged_out),
            ArgValue::Ptr(stage.ordered.order),
            ArgValue::Ptr(stage.ordered.offsets),
            y.arg(),
            stated(op, y.width)?.arg(),
            stated(op, experts)?.arg(),
        ],
    )
}

/// The routed swiglu MLP with its rows kept by expert from the gate-up to
/// the fold, over hipBLASLt's grouped GEMM: the routing scatter gathers the
/// token rows by expert once (noting where each route's row went), the
/// swiglu runs over the rows in that order, neither GEMM scatters back, and
/// the fold reads each route's row where it sits. From
/// `kernel.moe.lt_mlp_from` routes an expert, over two deep contractions
/// (hidden and intermediate at least `kernel.moe.lt_deep_k`: on MI300X
/// hipBLASLt's grouped tiles lose to the native legs over a 640-deep down
/// projection, Qwen3.8-Flash-Next's, at every prefill); whether it ran.
#[cfg(feature = "rocm")]
fn mlp_lt(ctx: &Ctx, op: &'static str, p: &RoutedMlp, y: &mut Tensor) -> Result<bool, Error> {
    use crate::hip::lt_grouped::USER_ARGS_BYTES;
    let (x, i) = (p.x, p.intermediate);
    let dtype = x.dtype;
    let (tokens, top_k) = (p.routes.rows, p.routes.width);
    let (k, out) = (x.width, y.width);
    let experts = p.gate_up.rows;
    if !matches!(dtype, Dtype::Bf16 | Dtype::F16)
        || p.gate_up.dtype != dtype
        || p.down.dtype != dtype
        || y.dtype != dtype
        || p.weights.dtype != Dtype::F32
        || p.gate_up_experts.streams()
        || p.down_experts.streams()
        || top_k == 0
        || x.rows != tokens
        || y.rows != tokens
        || experts == 0
        || experts > ORDER_EXPERT_CAP
        || p.down.rows != experts
        || u64::from(p.gate_up.width) != 2 * u64::from(i) * u64::from(k)
        || u64::from(p.down.width) != u64::from(out) * u64::from(i)
        || [k, i, out].iter().any(|w| !w.is_multiple_of(8) || *w < 64)
    {
        return Ok(false);
    }
    let Some(rows) = tokens.checked_mul(top_k) else {
        return Ok(false);
    };
    let t = ctx.tuning();
    if !t.moe_grouped
        || u64::from(rows) * u64::from(k + 3 * i + out) * 2 > LT_STAGE_CAP
        || !lt_wanted(
            ctx,
            rows / experts,
            t.moe_lt_mlp_from,
            k >= t.moe_lt_deep_k && i >= t.moe_lt_deep_k,
        )
    {
        return Ok(false);
    }
    let (Some((lib, up_at)), Some((_, down_at))) =
        (lt_solution(ctx, dtype, k), lt_solution(ctx, dtype, i))
    else {
        return Ok(false);
    };
    let ty = dtype_dispatch!(op, dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = Selected {
        route_count: rows,
        top_k: stated(op, top_k)?,
        by_token: true,
    };
    let stream = ctx.stream();
    let pos = ctx.scratch(op, "linear.moe_lt_pos", rows as usize * 4)? as usize as u64;
    let stage = stage_lt(ctx, op, x, p.gate_up, p.routes, 2 * i, experts, &fan, pos)?;
    gemm_lt(op, lib, up_at, &stage, 2 * i, rows, experts, stream)?;

    // The swiglu over the rows in expert order.
    let hidden = ctx.scratch_stream(op, "linear.moe_lt_hidden", rows as usize * i as usize * 2)?
        as usize as u64;
    let clamp = if p.limit.is_some() { "true" } else { "false" };
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&format!("::pie::linear::moe_swiglu_ordered<{ty}, {clamp}>")))
            .apply(Launch::grid([rows, i.div_ceil(8 * 256), 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(stage.staged_out),
            ArgValue::Ptr(hidden),
            ArgValue::Ptr(stage.ordered.offsets),
            stated(op, experts)?.arg(),
            stated(op, i)?.arg(),
            p.limit.unwrap_or(f32::INFINITY).arg(),
        ],
    )?;

    // The down GEMM over those rows, its products left in expert order.
    let routed = ctx.scratch_stream(op, "linear.moe_lt_down", rows as usize * out as usize * 2)?
        as usize as u64;
    let args = ctx.scratch(
        op,
        "linear.moe_lt_down_args",
        (experts as usize + 1) * USER_ARGS_BYTES,
    )? as usize as u64;
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&format!("::pie::linear::moe_lt_args<{ty}>")))
            .apply(Launch::grid([1, 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(stage.ordered.offsets),
            ArgValue::Ptr(args),
            p.down.ptr.arg(),
            hidden.arg(),
            routed.arg(),
            stated(op, out)?.arg(),
            stated(op, i)?.arg(),
            stated(op, experts)?.arg(),
        ],
    )?;
    lt_launch(op, lib, down_at, args, out, rows, experts, stream)?;

    // The fold, each route's row read where it sits.
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&format!("::pie::linear::moe_lt_fold<{ty}>")))
            .apply(elementwise_rows(op, y.rows, y.width)?),
        &[
            y.arg(),
            routed.arg(),
            ArgValue::Ptr(pos),
            p.routes.arg(),
            p.weights.arg(),
            stated(op, top_k)?.arg(),
            stated(op, out)?.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(true)
}

#[cfg(feature = "rocm")]
const fn block_rows(per_expert: u32) -> u32 {
    let want = per_expert.next_power_of_two();
    if want < 16 {
        16
    } else if want > 128 {
        128
    } else {
        want
    }
}

#[cfg(feature = "rocm")]
const GROUP_WORKSPACE_CAP: u64 = 512 << 20;

/// Whether the rocBLAS leg seats its experts and its staged workspace fits.
#[cfg(feature = "rocm")]
fn blas_worth(x: Tensor, y: &Tensor, fan: &Selected, count: u32) -> bool {
    if count > MAX_EXPERTS {
        return false;
    }
    let block = block_rows(fan.route_count.div_ceil(count));
    let blocks = u64::from(count) + u64::from(fan.route_count.div_ceil(block));
    let staged = (blocks * u64::from(block))
        .saturating_mul(u64::from(x.width) + u64::from(y.width))
        .saturating_mul(x.dtype.bytes_ceil());
    staged <= GROUP_WORKSPACE_CAP
}

/// The rocBLAS leg: routes bucketed into blocks of one expert, gathered,
/// multiplied as one batched GEMM, and scattered back.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
fn select_grouped_blas(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: u32,
) -> Result<(), Error> {
    use crate::hip::blas::{self, Operand};
    use crate::hip::sys;

    let (t, scalar) = match x.dtype {
        Dtype::Bf16 => ("::pie::bf16", sys::BLAS_BF16),
        Dtype::F16 => ("::pie::f16", sys::BLAS_F16),
        other => {
            return Err(refuse(
                op,
                format!("the grouped leg hands rocBLAS a 16-bit bank and this one is {other:?}"),
            ));
        }
    };
    let handle = ctx.blas(op)?;

    let fan = selected(op, x, routes, y)?;
    let per_expert = fan.route_count.div_ceil(experts);
    let block = block_rows(per_expert);
    let blocks = experts + fan.route_count.div_ceil(block);
    let rows = blocks.checked_mul(block).ok_or_else(|| {
        refuse(
            op,
            format!("{blocks} blocks of {block} rows overflow the fire"),
        )
    })?;

    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let route_count = stated(op, fan.route_count)?;
    let aligned_rows = stated(op, rows)?;
    let block_size = stated(op, block)?;
    let max_blocks = stated(op, blocks)?;
    let expert_count = stated(op, experts)?;

    let elem = x.dtype.bytes_ceil();
    let slab = |name: &'static str, bytes: u64| -> Result<u64, Error> {
        usize::try_from(bytes)
            .map_err(|_| {
                refuse(
                    op,
                    format!("{name} wants {bytes} bytes, past this host's usize"),
                )
            })
            .and_then(|bytes| ctx.scratch(op, name, bytes))
            .map(|ptr| ptr as usize as u64)
    };
    let sorted = slab("linear.moe_group_sorted", u64::from(rows) * 4)?;
    let expert_ids = slab("linear.moe_group_experts", u64::from(blocks) * 4)?;
    let staged_in = slab(
        "linear.moe_group_in",
        u64::from(rows) * u64::from(x.width) * elem,
    )?;
    let staged_out = slab(
        "linear.moe_group_out",
        u64::from(rows) * u64::from(y.width) * elem,
    )?;
    let ptrs = slab("linear.moe_group_ptrs", u64::from(blocks) * 3 * 8)?;
    let w_ptrs = ptrs;
    let act_ptrs = ptrs + u64::from(blocks) * 8;
    let out_ptrs = ptrs + u64::from(blocks) * 16;

    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::moe_align_decode<::pie::i32>")
            .apply(Launch::grid([1, 1, 1], [BLOCK, 1, 1]).smem((3 * experts + 34) * 4)),
        &[
            routes.arg(),
            ArgValue::Ptr(sorted),
            ArgValue::Ptr(expert_ids),
            ArgValue::ABSENT,
            route_count.arg(),
            expert_count.arg(),
            block_size.arg(),
            max_blocks.arg(),
            ArgValue::ABSENT,
        ],
    )?;

    let gather_fan = if fan.by_token { fan.top_k } else { 1 };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::gather_moe_aligned_inputs<{t}>")),
        )
        .apply(Launch::grid(
            [rows, x.width.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            ArgValue::Ptr(sorted),
            ArgValue::Ptr(staged_in),
            route_count.arg(),
            aligned_rows.arg(),
            gather_fan.arg(),
            k.arg(),
            (-1i32).arg(),
            stated(op, x.rows)?.arg(),
        ],
    )?;

    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::build_moe_leg_ptrs<{t}>")),
        )
        .apply(Launch::flat(blocks, BLOCK)),
        &[
            ArgValue::Ptr(expert_ids),
            bank.arg(),
            ArgValue::Ptr(staged_in),
            ArgValue::Ptr(staged_out),
            ArgValue::Ptr(w_ptrs),
            ArgValue::Ptr(act_ptrs),
            ArgValue::Ptr(out_ptrs),
            max_blocks.arg(),
            block_size.arg(),
            k.arg(),
            n.arg(),
        ],
    )?;

    // SAFETY: `handle` is this context's live rocBLAS handle. The three
    // pointer arrays hold `blocks` entries each, written by the launch above
    // into slabs this context owns and holds for the fire's life.
    unsafe {
        blas::bind(handle, ctx.stream())?;
        blas::gemm_batched(
            handle,
            sys::BLAS_OP_T,
            sys::BLAS_OP_N,
            n,
            block_size,
            k,
            Operand::new(w_ptrs, scalar, k),
            Operand::new(act_ptrs, scalar, k),
            Operand::new(out_ptrs, scalar, n),
            max_blocks,
        )
    }
    .map_err(|why| {
        refuse(
            op,
            format!("{why} at {max_blocks} blocks of M={block_size} N={n} K={k}"),
        )
    })?;

    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::reorder_moe_aligned_output<{t}>")),
        )
        .apply(Launch::grid(
            [rows, y.width.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            ArgValue::Ptr(staged_out),
            ArgValue::Ptr(sorted),
            y.arg(),
            route_count.arg(),
            aligned_rows.arg(),
            n.arg(),
            (-1i32).arg(),
            0i32.arg(),
            ArgValue::ABSENT,
        ],
    )
}

/// Waves a stream block runs (`kStreamWaves`).
const STREAM_WAVES: u32 = 4;
const STREAM_LANES: u32 = 64;
const STREAM_BLOCK: u32 = STREAM_WAVES * STREAM_LANES;
/// The LDS a block may hold.
const STREAM_LDS: u32 = 64 << 10;

/// Unit-dot slots a stream wave holds in LDS, short of a single row.
const STREAM_DOTS: u32 = 1024;

/// The slots a stream row takes (`stream_slots`): one a unit for a row of
/// fewer units than lanes, else one a lane.
const fn stream_slots(units: u32) -> u32 {
    if units < STREAM_LANES {
        units
    } else {
        STREAM_LANES
    }
}

/// A stream codec as the launch sees it: codes a unit, units a lane
/// fetches a round (`kUnroll`), and whether it stages unit sums.
#[derive(Clone, Copy)]
struct StreamCodec {
    codes: u32,
    unroll: u32,
    xsum: bool,
}

/// The rows a stream wave takes for rows of `units` units: the count (≤ 64,
/// its slots within `STREAM_DOTS`) whose run wastes the least of its last
/// round of fetches, the more rows on a tie.
fn stream_rows(units: u32, unroll: u32) -> u32 {
    let round = STREAM_LANES * unroll;
    let most = (STREAM_DOTS / stream_slots(units).max(1)).clamp(1, STREAM_LANES);
    (1..=most)
        .max_by_key(|&r| {
            let run = r * units;
            // The share of the fetched units the run uses, in 1/1024ths.
            (
                u64::from(run) * 1024 / u64::from(run.div_ceil(round) * round),
                r,
            )
        })
        .unwrap_or(1)
}

/// The decode stream's launch over `routes` routes of `n` rows: each wave
/// takes `stream_rows` rows, and the block stages the activation (and,
/// `xsum`, its unit sums) and its waves' unit dots in LDS. The rows a wave
/// takes ride along as the kernel's `rows_per_wave`.
fn stream_launch(
    op: &'static str,
    routes: u32,
    n: u32,
    k: u32,
    codec: StreamCodec,
) -> Result<(Launch, i32), Error> {
    let units = k / codec.codes;
    let rows = stream_rows(units, codec.unroll);
    let lds = u64::from(2 * k).next_multiple_of(16)
        + if codec.xsum {
            u64::from(units.next_multiple_of(4)) * 4
        } else {
            0
        }
        + u64::from(STREAM_WAVES * rows) * u64::from(stream_slots(units)) * 4;
    if lds > u64::from(STREAM_LDS) {
        return Err(refuse(
            op,
            format!(
                "a {k}-wide row stages {lds} bytes of LDS, past the {STREAM_LDS} a block holds"
            ),
        ));
    }
    let launch = Launch::grid(
        [routes, n.div_ceil(STREAM_WAVES * rows), 1],
        [STREAM_BLOCK, 1, 1],
    )
    .smem(lds as u32);
    Ok((launch, stated(op, rows)?))
}

pub(crate) fn select_gemv(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<(), Error> {
    const VEC_WIDTH: u32 = 8;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(bank.dtype, x.dtype, "the bank rides the activation's dtype");
    let fan = selected(op, x, routes, y)?;
    if x.width == 0 || !x.width.is_multiple_of(VEC_WIDTH) {
        return Err(refuse(
            op,
            format!(
                "K is {}, not a whole number of {VEC_WIDTH}-element 16-byte loads",
                x.width
            ),
        ));
    }
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let (launch, rows) = stream_launch(
        op,
        fan.route_count,
        y.width,
        x.width,
        StreamCodec {
            codes: VEC_WIDTH,
            unroll: 8,
            xsum: false,
        },
    )?;
    let form = if fan.by_token { "by_token" } else { "by_route" };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::moe_matmul_select_gemv_{form}<{t}>"
            )),
        )
        .apply(launch),
        &[
            routes.arg(),
            x.arg(),
            bank.arg(),
            y.arg(),
            fan.top_k.arg(),
            k.arg(),
            n.arg(),
            (i64::from(n) * i64::from(k)).arg(), // the bank's expert stride
            ArgValue::Ptr(experts.table),
            ArgValue::Ptr(experts.hits),
            rows.arg(),
            ctx.stage(),
        ],
    )
}

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
    seat: GroupSeat,
) -> Result<(), Error> {
    const MXFP4_BLOCK: u32 = 32;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "an mxfp4 bank's codes are u8");
    debug_assert_eq!(scales.dtype, Dtype::U8, "an mxfp4 bank's scales are u8");
    debug_assert!(
        bias.is_none_or(|bias| bias.dtype == x.dtype),
        "the expert bias rides the activation's dtype"
    );
    let fan = selected(op, x, routes, y)?;
    if x.width == 0 || !x.width.is_multiple_of(MXFP4_BLOCK) {
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
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    // The grouped kernels read an expert once per batch of its routes, the
    // by-route one once per route but at the card's streaming rate, so grouping
    // pays only once an expert serves more than `kernel.moe.mxfp4_grouped_reuse`
    // routes on average; routes over experts is a floor on that average.
    let experts = codes.rows.clamp(1, ORDER_EXPERT_CAP);
    if x.dtype == Dtype::Bf16
        && codes.rows <= ORDER_EXPERT_CAP
        && let Some(mt) = batch_items(ctx, &fan, experts, 32, ctx.tuning().moe_batch_until)
        // Short of 32 routes a lane reads its two scale bytes as one.
        && (mt == 32 || x.width.is_multiple_of(2 * MXFP4_BLOCK))
    {
        return select_batch(
            ctx,
            op,
            "linear/quant.hip",
            &format!(
                "::pie::linear::moe_matmul_select_mxfp4_batch<::pie::i32({mt}), ::pie::i32({BATCH_NBW})>"
            ),
            x,
            &[
                codes.arg(),
                scales.arg(),
                bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            ],
            routes,
            y,
            &fan,
            experts,
            mt,
            BATCH_NBW,
            Some(seat),
        );
    }
    if fan.route_count > ctx.tuning().mxfp4_grouped_reuse * experts && ctx.tuning().mxfp4_grouped {
        let ordered = route_order(ctx, op, routes, &fan, experts, group_routes(&fan, experts))?;
        let wmma = x.dtype == Dtype::Bf16 && ctx.tuning().mxfp4_wmma;
        let entry = if wmma {
            format!(
                "::pie::linear::moe_matmul_select_mxfp4_wmma<::pie::i32({})>",
                ordered.routes
            )
        } else {
            format!("::pie::linear::moe_matmul_select_mxfp4_grouped<{t}>")
        };
        let mut args = vec![
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
        ];
        if wmma {
            args.push(ArgValue::Ptr(ordered.work));
        }
        args.extend([
            codes.arg(),
            scales.arg(),
            bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
        ]);
        return ctx.fire(
            op,
            Fire::at("linear/quant.hip", symbol(&entry)).apply(if wmma {
                group_launch(ordered.work_cap, ordered.routes, y.width)
            } else {
                Launch::grid(
                    [experts, y.width.div_ceil(GROUP_TILE_N), 1],
                    [GROUP_BLOCK, 1, 1],
                )
            }),
            &args,
        );
    }
    let (launch, rows) = stream_launch(
        op,
        fan.route_count,
        y.width,
        x.width,
        StreamCodec {
            codes: MXFP4_BLOCK,
            unroll: 4,
            xsum: false,
        },
    )?;
    ctx.fire(
        op,
        Fire::at(
            "linear/quant.hip",
            symbol(&format!("::pie::linear::moe_matmul_select_mxfp4<{t}>")),
        )
        .apply(launch),
        &[
            x.arg(),
            routes.arg(),
            codes.arg(),
            scales.arg(),
            bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
            y.arg(),
            fan.top_k.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            rows.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_select_bias(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_bias";
    matmul_select_mxfp4(ctx, OP, x, codes, scales, Some(bias), routes, y, seat)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_select_quant(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    // A byte a code beside four a 128x128 block: an fp8 bank (`e4m3tile128`).
    if biases.is_none() && fp8_tiles(x, codes, scales, y) {
        return Err(refuse(
            OP,
            "an fp8 bank (e4m3tile128) is read against e4m3 activations under f32 scales a 1x128 block \
             (`matmul_select_fp8_mfma`), not as the routes' rows are",
        ));
    }
    match biases {
        None => matmul_select_mxfp4(ctx, OP, x, codes, scales, None, routes, y, seat),
        Some(biases) => matmul_select_mlxu4(ctx, OP, x, codes, scales, biases, routes, y, seat),
    }
}

/// Whether `codes` and `scales` are an fp8 bank's planes for `x` into `y`:
/// a byte an expert's code, an f32 its 128x128 block's factor.
pub fn fp8_tiles(x: Tensor, codes: Tensor, scales: Tensor, y: &Tensor) -> bool {
    let (n, k) = (u64::from(y.width), u64::from(x.width));
    n > 0
        && k > 0
        && u64::from(codes.width) == n * k
        && u64::from(scales.width) == 4 * n.div_ceil(128) * k.div_ceil(128)
}

/// An fp8 bank (`e4m3tile128`, its codes made canonical at load) against
/// the routes' activation rows (bf16) quantized as e4m3 under f32 scales a
/// 1x128 block (`quantized_codes`), on the FP8 matrix core: at decode
/// batches items of the least of 16, 32 and 64 routes holding what an expert
/// averages; past 64 routes an expert the prefill tile, items of 128 against
/// 128 weight rows. An expert past its item takes another.
#[allow(clippy::too_many_arguments)]
pub fn matmul_select_fp8_mfma(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    let (q, act_scales) = crate::linear::act_quant::quantized_codes(ctx, OP, x, 128)?;
    matmul_select_fp8_quantized(ctx, x, q, act_scales, codes, scales, routes, y, seat)
}

/// `matmul_select_fp8_mfma` over rows already quantized (`q`, `act_scales`
/// of `x`'s rows, `x` itself the routes' bf16 rows they came from).
#[allow(clippy::too_many_arguments)]
pub fn matmul_select_fp8_quantized(
    ctx: &Ctx,
    x: Tensor,
    q: Tensor,
    act_scales: Tensor,
    codes: Tensor,
    scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    let op = OP;
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    if q.rows != x.rows || q.width != x.width || act_scales.rows != x.rows || act_scales.width != x.width / 128 {
        return Err(refuse(op, "the quantized rows are not x's codes beside a scale a 128-wide block"));
    }
    let fan = selected(op, x, routes, y)?;
    if !x.width.is_multiple_of(128) || !fp8_tiles(x, codes, scales, y) {
        return Err(refuse(op, "the bank's planes are not e4m3 codes beside an f32 factor a 128x128 block of K"));
    }
    if codes.rows > ORDER_EXPERT_CAP {
        return Err(refuse(
            op,
            format!(
                "the batch legs order at most {ORDER_EXPERT_CAP} experts, and this fp8 bank holds {}",
                codes.rows
            ),
        ));
    }
    let experts = codes.rows.max(1);
    let per = fan.route_count.div_ceil(experts);
    let gemm = per > 64;
    let mt = if gemm { 128 } else { crate::linear::act_quant::fp8_item_rows(per) };
    let kk = stated(op, x.width)?;
    let nn = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let ordered = route_order(ctx, op, routes, &fan, experts, mt)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let mut args = vec![q.arg(), act_scales.arg()];
    let (entry, rows, threads) = if gemm {
        ("::pie::linear::moe_fp8_gemm8".to_string(), 128, 2 * GROUP_BLOCK)
    } else {
        (format!("::pie::linear::moe_fp8_mfma<::pie::i32({mt})>"), 64, GROUP_BLOCK)
    };
    args.extend([
        ArgValue::Ptr(ordered.order),
        ArgValue::Ptr(ordered.offsets),
        ArgValue::Ptr(ordered.work),
        codes.arg(),
        scales.arg(),
        y.arg(),
        act_div.arg(),
        nn.arg(),
        kk.arg(),
        stated(op, experts)?.arg(),
        ArgValue::Ptr(seat.cell),
        ArgValue::Ptr(seat.hits),
    ]);
    ctx.fire(
        op,
        Fire::at("linear/fp8.hip", symbol(&entry))
            .apply(Launch::grid([ordered.work_cap, y.width.div_ceil(rows), 1], [threads, 1, 1])),
        &args,
    )
}

#[allow(clippy::too_many_arguments)]
fn matmul_select_mlxu4(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const UNIT: u32 = 32;

    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(
        codes.dtype,
        Dtype::U8,
        "a packed bank's planes bind as bytes"
    );
    debug_assert_eq!(
        scales.dtype,
        Dtype::U8,
        "a packed bank's planes bind as bytes"
    );
    debug_assert_eq!(
        biases.dtype,
        Dtype::U8,
        "a packed bank's planes bind as bytes"
    );
    let fan = selected(op, x, routes, y)?;
    let k = stated(op, nonzero(op, "K, the bank's contracted width", x.width)?)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let elems = u64::from(x.width) * u64::from(y.width);
    let code_bits = u64::from(codes.width) * 8;
    let bits: u32 = match code_bits {
        b if b == elems * 8 => 8,
        b if b == elems * 4 => 4,
        b if b == elems * 2 => 2,
        _ => {
            return Err(refuse(
                op,
                format!(
                    "a {}-byte expert code row stores {elems} codes at neither two, four nor eight bits",
                    codes.width
                ),
            ));
        }
    };
    let factor_bytes = u64::from(scales.width);
    let groups_per_row = if factor_bytes > 0 && factor_bytes % (2 * u64::from(y.width)) == 0 {
        factor_bytes / (2 * u64::from(y.width))
    } else {
        0
    };
    let group = if groups_per_row > 0 && u64::from(x.width) % groups_per_row == 0 {
        u64::from(x.width) / groups_per_row
    } else {
        0
    };
    let group = u32::try_from(group).unwrap_or(0);
    if !matches!(group, 32 | 64 | 128) {
        return Err(refuse(
            op,
            format!(
                "{groups_per_row} factors over a {}-wide row is not a 32-, 64- or 128-code affine group",
                x.width
            ),
        ));
    }
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    let experts = codes.rows.clamp(1, ORDER_EXPERT_CAP);
    // The batch legs read four-bit codes, and a lane's factors of a stage
    // (an even count of groups) as one read.
    if bits == 4
        && x.dtype == Dtype::Bf16
        && codes.rows <= ORDER_EXPERT_CAP
        && x.width.is_multiple_of(2 * group)
        && let Some(mt) = batch_items(ctx, &fan, experts, 32, ctx.tuning().moe_batch_until)
    {
        return select_batch(
            ctx,
            op,
            "linear/quant.hip",
            &format!(
                "::pie::linear::moe_matmul_select_mlxu4_batch<::pie::i32({bits}), ::pie::i32({group}), \
                 ::pie::i32({mt}), ::pie::i32({BATCH_NBW})>"
            ),
            x,
            &[codes.arg(), scales.arg(), biases.arg()],
            routes,
            y,
            &fan,
            experts,
            mt,
            BATCH_NBW,
            Some(seat),
        );
    }
    let tuning = ctx.tuning();
    if tuning.affine_grouped && fan.route_count >= tuning.affine_grouped_from {
        let ordered = route_order(ctx, op, routes, &fan, experts, group_routes(&fan, experts))?;
        let wmma = x.dtype == Dtype::Bf16 && tuning.affine_wmma;
        let entry = if wmma {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_wmma<::pie::i32({bits}), ::pie::i32({group}), \
                 ::pie::i32({})>",
                ordered.routes
            )
        } else {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_grouped<{t}, ::pie::i32({bits}), \
                 ::pie::i32({group})>"
            )
        };
        let mut args = vec![
            x.arg(),
            ArgValue::Ptr(ordered.order),
            ArgValue::Ptr(ordered.offsets),
        ];
        if wmma {
            args.push(ArgValue::Ptr(ordered.work));
        }
        args.extend([
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
        ]);
        return ctx.fire(
            op,
            Fire::at("linear/quant.hip", symbol(&entry)).apply(if wmma {
                group_launch(ordered.work_cap, ordered.routes, y.width)
            } else {
                Launch::grid(
                    [experts, y.width.div_ceil(GROUP_TILE_N), 1],
                    [GROUP_BLOCK, 1, 1],
                )
            }),
            &args,
        );
    }
    let (launch, rows) = stream_launch(
        op,
        fan.route_count,
        y.width,
        x.width,
        StreamCodec {
            codes: UNIT,
            unroll: if bits == 8 { 2 } else { 4 },
            xsum: true,
        },
    )?;
    ctx.fire(
        op,
        Fire::at(
            "linear/quant.hip",
            symbol(&format!(
                "::pie::linear::moe_matmul_select_mlxu4<{t}, ::pie::i32({bits}), ::pie::i32({group})>"
            )),
        )
        .apply(launch),
        &[
            x.arg(),
            routes.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            fan.top_k.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            rows.arg(),
            ctx.stage(),
        ],
    )
}

/// Each token's routed rows (already weighted) summed in f32, plus its
/// `shared` row, rounded once.
pub fn sum_add(ctx: &Ctx, routed: Tensor, shared: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.moe_sum_add";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "the token rows this fold lands on", y.rows)?;
    if !routed.rows.is_multiple_of(y.rows) || routed.width != y.width {
        return Err(refuse(
            OP,
            format!(
                "the routed rectangle [{}, {}] does not fold into [{}, {}] token rows",
                routed.rows, routed.width, y.rows, y.width
            ),
        ));
    }
    if shared.rows != y.rows || shared.width != y.width || shared.dtype != routed.dtype {
        return Err(refuse(OP, "the shared row is not one token row of the routed dtype"));
    }
    let top_k = routed.rows / y.rows;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::moe_sum_add<{t}>")))
            .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            routed.arg(),
            shared.arg(),
            stated(OP, top_k)?.arg(),
            stated(OP, y.width)?.arg(),
            ctx.stage(),
        ],
    )
}

/// A routed swiglu MLP (`linear.moe_mlp_swiglu`): `packed` = x through the
/// gate_up bank by route, `h` = its swiglu (clamped at `limit` when given),
/// `routed` = h through the down bank, y = their weighted fold. `packed`,
/// `h` and `routed` are the planes the four steps land; an expert-ordered
/// pipeline may leave them untouched.
#[derive(Clone, Copy, Debug)]
pub struct RoutedMlp {
    pub x: Tensor,
    pub routes: Tensor,
    pub weights: Tensor,
    pub gate_up: Tensor,
    pub gate_up_experts: ExpertTable,
    pub down: Tensor,
    pub down_experts: ExpertTable,
    pub intermediate: u32,
    pub limit: Option<f32>,
    /// The swiglu's fan-out over `h`'s rows (the engine's plane fan).
    pub fan: u32,
    pub packed: Tensor,
    pub h: Tensor,
    pub routed: Tensor,
}

/// The routed MLP: on the grouped GEMM with its routes in expert order
/// throughout where that measured faster, else the four routed steps.
pub fn mlp_swiglu(ctx: &Ctx, p: &RoutedMlp, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.moe_mlp_swiglu";
    let steps = || -> Result<(), Error> {
        let (mut packed, mut h, mut routed) = (p.packed, p.h, p.routed);
        matmul_select(ctx, p.x, p.gate_up, p.routes, &mut packed, p.gate_up_experts)?;
        match p.limit {
            Some(limit) => crate::linear::mlp::swiglu_clamp(ctx, packed, p.intermediate, p.fan, limit, &mut h)?,
            None => crate::linear::mlp::swiglu(ctx, packed, p.intermediate, p.fan, &mut h)?,
        }
        matmul_select(ctx, h, p.down, p.routes, &mut routed, p.down_experts)?;
        Ok(())
    };
    #[cfg(feature = "rocm")]
    if mlp_lt(ctx, OP, p, y)? || ordered::mlp(ctx, OP, p, y)? {
        return Ok(());
    }
    let _ = OP;
    steps()?;
    weighted_sum(ctx, p.routed, p.weights, y)
}

pub fn weighted_sum(
    ctx: &Ctx,
    routed: Tensor,
    weights: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_weighted_sum";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
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
            ctx.stage(),
        ],
    )
}

pub fn bias_sum(
    ctx: &Ctx,
    x: Tensor,
    bias: Tensor,
    routes: Tensor,
    weights: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
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
            ctx.stage(),
        ],
    )
}

pub fn sigmoid_gate_add(
    ctx: &Ctx,
    routed: Tensor,
    shared: Tensor,
    gate: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
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
            "linear/glu.hip",
            symbol(&format!("::pie::linear::moe_sigmoid_gate_add<{t}>")),
        )
        .apply(elementwise_rows(OP, y.rows, y.width)?),
        &[
            y.arg(),
            routed.arg(),
            shared.arg(),
            gate.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, nonzero(OP, "the gate row's pitch", gate.width)?)?.arg(),
            ctx.stage(),
        ],
    )
}
