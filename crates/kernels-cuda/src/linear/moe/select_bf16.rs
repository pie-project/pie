use super::*;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, kernel, nonzero, stated};
use crate::tensor::Tensor;
use dtype::Dtype;

/// [`select_bf16_mma`] on Hopper's wgmma (`moe_select_bf16_wgmma`): an
/// expert's slice 16 routes, or 32 past an average of 8.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_bf16_wgmma(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    experts: u32,
) -> Result<(), Error> {
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let slice: u32 = if fan.route_count.div_ceil(experts.max(1)) <= 8 {
        16
    } else {
        32
    };
    let RouteOrder {
        order,
        offsets,
        work,
        work_cap,
    } = route_order_sliced(
        ctx,
        op,
        routes,
        fan,
        experts,
        slice,
        true,
        if KEEP_NEXT.with(core::cell::Cell::get) {
            Share::Keep
        } else {
            Share::Take
        },
    )?;
    // A CTA 64 of the expert's rows, K 128 a stage where it divides (fewer
    // barriers a row: 3.0 TB/s at Gemma 4 26B-A4B's gate-up, every expert
    // routed), else 64: `kWgStages` stages of the rows and the slice's route
    // rows, past the 1024-byte alignment.
    let atoms: u32 = if x.width.is_multiple_of(128) { 2 } else { 1 };
    let smem = 3 * atoms * (64 + slice) * 128 + 1024;
    ctx.fire(
        op,
        Fire::at(kernel(
            SELECT_SM90,
            &format!("::pie::linear::moe_select_bf16_wgmma<{slice}, {atoms}>"),
        ))
        .apply(Launch::grid([work_cap, y.width / 64, 1], [128, 1, 1]).smem(smem)),
        &[
            x.arg(),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            bank.arg(),
            y.arg(),
            act_div.arg(),
            n.arg(),
            stated(op, x.width)?.arg(),
            stated(op, experts)?.arg(),
        ],
    )
}

/// Whether the graph-safe bf16 grouped mma.sync kernel takes a fire of
/// `routes` routes: `kernel.moe.bf16_mma_from ..= kernel.moe.bf16_mma_until`.
pub(super) fn bf16_mma_spans(ctx: &Ctx, routes: u32) -> bool {
    let t = ctx.tuning();
    t.bf16_mma_until > 0 && t.bf16_mma_from <= routes && routes <= t.bf16_mma_until
}

/// `y[route] = x[route or its token] . W[routes[route]]^T` on the grouped
/// bf16 mma.sync kernel: routes laid out by expert on-device (graph-safe),
/// each expert's weight rows read once a slice.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_bf16_mma(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    experts: u32,
) -> Result<(), Error> {
    const THREADS: u32 = 128;
    const COLS: u32 = 128;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let k = stated(op, x.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    // An expert's slice is 8, 16, 32 or 64 routes, by how many the average
    // expert holds: each slice reads its weight rows once for all of them.
    let per = fan.route_count.div_ceil(experts.max(1));
    let nb: u32 = match per {
        0..=8 => 1,
        9..=16 => 2,
        17..=40 => 4,
        _ => 8,
    };
    let RouteOrder {
        order,
        offsets,
        work,
        work_cap,
    } = route_order_sliced(
        ctx,
        op,
        routes,
        fan,
        experts,
        8 * nb,
        true,
        if KEEP_NEXT.with(core::cell::Cell::get) {
            Share::Keep
        } else {
            Share::Take
        },
    )?;
    let grid = [work_cap, y.width.div_ceil(COLS), 1];
    // Three stages of 128 weight rows and 8 nb route rows, each a K step of
    // 64 bf16 padded by 8 (`kBf16MmaStages`, `kBf16MmaBK`).
    let smem = 3 * (128 + 8 * nb) * (64 + 8) * 2;
    ctx.fire(
        op,
        Fire::at(kernel(
            SELECT_SM80,
            &format!("::pie::linear::moe_select_bf16_mma<{nb}>"),
        ))
        .apply(Launch::grid(grid, [THREADS, 1, 1]).smem(smem)),
        &[
            x.arg(),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            bank.arg(),
            ArgValue::ABSENT,
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
        ],
    )
}

/// From this many routes a fire (a prefill's), a W8A16 select is the tiled
/// GEMM that widens a stage's codes once a 64 routes
/// (`moe_select_fp8_gemm`); fewer, the register-held kernel streams the
/// bank as fast.
pub(super) const FP8_GEMM_FROM: u32 = 8192;

/// From this many routes a fire (with K in whole 64s), a bf16 select is the
/// tiled GEMM (`moe_select_bf16_gemm`), a CTA 64 of an expert's routes: a
/// batch's routes crowd onto a few hot experts as often as not, and its
/// tiles stream a hot expert's rows once a 64 routes, near the card's rate
/// with the routes spread.
pub(super) const BF16_GEMM_FROM: u32 = 256;

/// The fewest routes an expert averages that Hopper's warp-specialized
/// wgmma GEMM (`moe_select_bf16_wsgemm`) takes a tiled bf16 select at:
/// a prefill's, in slices of 128.
pub(super) const BF16_WSGEMM_FROM: u32 = 64;

/// A routed select over a bf16 bank (`experts x n x k` at `bank`) on mma.sync
/// (`linear/moe_w8.cuh`): the register-held `moe_select_bf16_held`, each
/// expert's rows streamed once a slice of its routes, or from
/// `BF16_GEMM_FROM` routes the tiled `moe_select_bf16_gemm`; the GLU in its
/// epilogue where `glu` says. `Ok(false)` for a fire it does not take.
#[allow(clippy::too_many_arguments)]
pub fn select_bf16_held(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: u64,
    experts: u32,
    routes: Tensor,
    y: &mut Tensor,
    glu: Option<crate::linear::gemm_sm100::Glu>,
) -> Result<bool, Error> {
    const THREADS: u32 = 128;
    const DEPTH: u32 = 2;
    let fan = selected(op, x, routes, y)?;
    if x.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || !x.width.is_multiple_of(32)
        || !bank.is_multiple_of(16)
    {
        return Ok(false);
    }
    let relu2 = glu == Some(crate::linear::gemm_sm100::Glu::Relu2);
    let glu = glu.filter(|_| !relu2);
    let (code, limit, alpha) = glu_code(op, glu)?;
    let code = if relu2 { 6 } else { code };
    if relu2
        && (crate::linear::gemm_sm100::supported()
            || hopper_wgmma()
            || fan.route_count < BF16_GEMM_FROM
            || !x.width.is_multiple_of(64))
    {
        return Ok(false);
    }
    let bank_rows = if glu.is_some() { 2 * y.width } else { y.width };
    let experts = experts.clamp(1, EXPERT_CAP);
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", bank_rows)?)?;
    let inter = stated(op, y.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    // A prefill's routes on Hopper's warp-specialized wgmma GEMM, an
    // expert's slice 128 of them by 256 weight rows (128 gate and their up
    // rows with a GLU), the GLU in its epilogue.
    if hopper_wgmma()
        && fan.route_count >= BF16_GEMM_FROM
        && fan.route_count.div_ceil(experts) >= BF16_WSGEMM_FROM
        && x.width.is_multiple_of(64)
        && (0..=5).contains(&code)
    {
        const STAGES: u32 = 4;
        let order = route_order_sliced(
            ctx,
            op,
            routes,
            &fan,
            experts,
            128,
            true,
            if glu.is_some() || KEEP_NEXT.with(core::cell::Cell::get) {
                Share::Keep
            } else {
                Share::Take
            },
        )?;
        let cols = if glu.is_some() { 128 } else { 256 };
        ctx.fire(
            op,
            Fire::at(kernel(
                SELECT_SM90,
                &format!("::pie::linear::moe_select_bf16_wsgemm<{STAGES}, {code}>"),
            ))
            .apply(
                Launch::grid([order.work_cap, y.width.div_ceil(cols), 1], [384, 1, 1])
                    .smem(STAGES * 49152 + 1024 + 128),
            ),
            &[
                x.arg(),
                ArgValue::Ptr(order.order),
                ArgValue::Ptr(order.offsets),
                ArgValue::Ptr(order.work),
                ArgValue::Ptr(bank),
                y.arg(),
                act_div.arg(),
                n.arg(),
                k.arg(),
                stated(op, experts)?.arg(),
                inter.arg(),
                limit.arg(),
                alpha.arg(),
            ],
        )?;
        return Ok(true);
    }
    let SelectPlan {
        nb,
        tiled,
        tiles,
        ksplit,
        mut order,
    } = select_plan(
        ctx,
        op,
        routes,
        &fan,
        experts,
        y,
        glu.is_some(),
        fan.route_count >= BF16_GEMM_FROM && x.width.is_multiple_of(64),
        // bf16 codes stream twice an fp8 bank's bytes: deeper K splits keep
        // more of them in flight (Hy4's decode +2%).
        8,
    )?;
    if relu2 && !tiled {
        return Ok(false);
    }
    let fire = if tiled {
        // Three stages of 64 routes' and 128 rows' 64 of K, four warps a
        // CTA each 32 routes by 64 rows: two CTAs an SM.
        const STAGES: u32 = 3;
        const WN: u32 = 2;
        let cols = if glu.is_some() { 64 } else { 128 };
        let bm: u32 = if fan.route_count.div_ceil(experts) >= 128 {
            128
        } else {
            64
        };
        if bm != 64 {
            order = route_order_sliced(ctx, op, routes, &fan, experts, bm, true, Share::No)?;
        }
        if bm == 128 {
            let cols = if glu.is_some() { 128 } else { 256 };
            Fire::at(kernel(
                SELECT_SM80,
                &format!("::pie::linear::moe_select_bf16_gemm_wide<{code}>"),
            ))
            .apply(
                Launch::grid([order.work_cap, y.width.div_ceil(cols), 1], [256, 1, 1])
                    .smem(2 * (128 + 256) * 128),
            )
        } else {
            Fire::at(kernel(
                SELECT_SM80,
                &format!("::pie::linear::moe_select_bf16_gemm<{STAGES}, {code}, {WN}, {bm}>"),
            ))
            .apply(
                Launch::grid([order.work_cap, y.width.div_ceil(cols), 1], [bm * WN, 1, 1])
                    .smem(STAGES * (bm + 128) * 128),
            )
        }
    } else {
        let cols = if glu.is_some() { 64 } else { 64 * tiles };
        Fire::at(kernel(
            SELECT,
            &format!(
                "::pie::linear::moe_select_bf16_held<{nb}, {DEPTH}, {code}, {tiles}, {ksplit}>"
            ),
        ))
        .apply(Launch::grid(
            [order.work_cap, y.width.div_ceil(cols), 1],
            [THREADS * ksplit, 1, 1],
        ))
    };
    ctx.fire(
        op,
        fire,
        &[
            x.arg(),
            ArgValue::Ptr(order.order),
            ArgValue::Ptr(order.offsets),
            ArgValue::Ptr(order.work),
            ArgValue::Ptr(bank),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            inter.arg(),
            limit.arg(),
            alpha.arg(),
        ],
    )?;
    Ok(true)
}
