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
    table: ExpertTable,
) -> Result<(), Error> {
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let slice = wgmma_slice(fan.route_count.div_ceil(experts.max(1)));
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
            ArgValue::Ptr(table.at()),
        ],
    )
}

/// The grouped mma.sync select's `nb` for experts averaging `per` routes: its
/// slices are `8 nb` routes (8, 16, 32 or 64), each reading its weight rows
/// once for all of them.
pub const fn mma_nb(per: u32) -> u32 {
    match per {
        0..=8 => 1,
        9..=16 => 2,
        17..=40 => 4,
        _ => 8,
    }
}

/// Hopper's wgmma select's slice for experts averaging `per` routes: 16
/// routes, or 32 past an average of 8.
pub const fn wgmma_slice(per: u32) -> u32 {
    if per <= 8 { 16 } else { 32 }
}

/// Whether [`Leg::Mma`] runs Hopper's wgmma select, at `n` product columns.
pub fn wgmma_takes(n: u32, experts: ExpertTable) -> bool {
    hopper_wgmma() && n.is_multiple_of(128) && !experts.streams()
}

/// The [`Leg::Mma`] leg's tile for experts averaging `per` routes at an
/// `n x k` plane: the wgmma select's slice by 64 columns on Hopper, else
/// `8 nb` routes by 128 columns in three stages of 64 of K.
pub fn mma_tile(per: u32, n: u32, k: u32, cores: Cores) -> Tile {
    if cores.hopper && n.is_multiple_of(128) && !cores.streams {
        let slice = wgmma_slice(per);
        let atoms: u32 = if k.is_multiple_of(128) { 2 } else { 1 };
        Tile {
            slice,
            cols: 64,
            smem: 3 * atoms * (64 + slice) * 128 + 1024,
        }
    } else {
        let nb = mma_nb(per);
        Tile {
            slice: 8 * nb,
            cols: 128,
            smem: 3 * (128 + 8 * nb) * (64 + 8) * 2,
        }
    }
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
    table: ExpertTable,
) -> Result<(), Error> {
    const THREADS: u32 = 128;
    const COLS: u32 = 128;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    let k = stated(op, x.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let nb = mma_nb(fan.route_count.div_ceil(experts.max(1)));
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
            ArgValue::Ptr(table.at()),
        ],
    )
}

/// From this many routes a fire (a prefill's), a W8A16 select is the tiled
/// GEMM that widens a stage's codes once a 64 routes
/// (`moe_select_fp8_gemm`); fewer, the register-held kernel streams the
/// bank as fast.
pub(super) const FP8_GEMM_FROM: u32 = 8192;

/// The fewest routes an expert averages that Hopper's warp-specialized
/// wgmma GEMM (`moe_select_bf16_wsgemm`) takes a tiled bf16 select at:
/// a prefill's, in slices of 128.
pub(super) const BF16_WSGEMM_FROM: u32 = 64;

/// The tiled leg's tile ([`Leg::Tiled`]) for a plain select whose experts
/// average `per` routes: Hopper's warp-specialized wgmma GEMM's slices of
/// 128 by 256 columns, the wide tile's 128 by 256 where its experts average
/// it (`kernel.moe.bf16_wide_from`, never below 128), else 64 by 128 in
/// `kernel.moe.bf16_gemm_stages` stages.
pub fn tiled_tile(per: u32, k: u32, cores: Cores) -> Tile {
    if cores.hopper && !cores.streams && per >= BF16_WSGEMM_FROM && k.is_multiple_of(64) {
        Tile {
            slice: 128,
            cols: 256,
            smem: WS_STAGES * 49152 + 1024 + 128,
        }
    } else if per >= cores.wide_from.max(128) {
        Tile {
            slice: 128,
            cols: 256,
            smem: 2 * (128 + 256) * 128,
        }
    } else {
        let stages = cores.stages.clamp(2, 4);
        Tile {
            slice: 64,
            cols: 128,
            smem: stages * (64 + 128) * 128,
        }
    }
}

/// The warp-specialized wgmma GEMM's stages.
const WS_STAGES: u32 = 4;

/// A routed select over a bf16 bank (`experts x n x k` at `bank`) on the
/// tiled GEMM ([`Leg::Tiled`]): `moe_select_bf16_gemm`, a CTA 64 of an
/// expert's routes (the wide tile's 128 where its experts average them), on
/// Hopper `moe_select_bf16_wsgemm`; the GLU in its epilogue where `glu`
/// says. `Ok(false)` for a fire it does not take.
#[allow(clippy::too_many_arguments)]
pub fn select_bf16_tiled(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: u64,
    experts: u32,
    routes: Tensor,
    y: &mut Tensor,
    glu: Option<crate::linear::gemm_sm100::Glu>,
    table: ExpertTable,
) -> Result<bool, Error> {
    let fan = selected(op, x, routes, y)?;
    if x.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || !x.width.is_multiple_of(64)
        || !bank.is_multiple_of(16)
    {
        return Ok(false);
    }
    let relu2 = glu == Some(crate::linear::gemm_sm100::Glu::Relu2);
    let glu = glu.filter(|_| !relu2);
    let (code, limit, alpha) = glu_code(op, glu)?;
    let code = if relu2 { 6 } else { code };
    // On Hopper only the warp-specialized GEMM below lands relu^2 in its epilogue.
    if relu2
        && (crate::linear::gemm_sm100::supported()
            || (hopper_wgmma()
                && fan.route_count.div_ceil(experts.clamp(1, EXPERT_CAP)) < BF16_WSGEMM_FROM))
    {
        return Ok(false);
    }
    let bank_rows = if glu.is_some() { 2 * y.width } else { y.width };
    let experts = experts.clamp(1, EXPERT_CAP);
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", bank_rows)?)?;
    let inter = stated(op, y.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let share = if glu.is_some() || KEEP_NEXT.with(core::cell::Cell::get) {
        Share::Keep
    } else {
        Share::Take
    };
    let per = fan.route_count.div_ceil(experts);
    // A prefill's routes on Hopper's warp-specialized wgmma GEMM, an
    // expert's slice 128 of them by 256 weight rows (128 gate and their up
    // rows with a GLU), the GLU in its epilogue. A streamed table stays on
    // the mma.sync kernels below, whose loads go through it.
    if hopper_wgmma() && !table.streams() && per >= BF16_WSGEMM_FROM && (0..=6).contains(&code) {
        let order = route_order_sliced(ctx, op, routes, &fan, experts, 128, true, share)?;
        let cols = if glu.is_some() { 128 } else { 256 };
        ctx.fire(
            op,
            Fire::at(kernel(
                SELECT_SM90,
                &format!("::pie::linear::moe_select_bf16_wsgemm<{WS_STAGES}, {code}>"),
            ))
            .apply(
                Launch::grid([order.work_cap, y.width.div_ceil(cols), 1], [384, 1, 1])
                    .smem(WS_STAGES * 49152 + 1024 + 128),
            ),
            &[
                x.arg(),
                ArgValue::Ptr(order.order),
                ArgValue::Ptr(order.offsets),
                ArgValue::Ptr(order.work),
                ArgValue::Ptr(bank),
                ArgValue::Ptr(0),
                y.arg(),
                act_div.arg(),
                n.arg(),
                k.arg(),
                stated(op, experts)?.arg(),
                inter.arg(),
                limit.arg(),
                alpha.arg(),
                ArgValue::Ptr(table.at()),
            ],
        )?;
        return Ok(true);
    }
    // A prefill's experts averaging the wide tile's routes lay their order in
    // its 128-route slices once, kept for the down select.
    let wide = per >= ctx.tuning().bf16_wide_from().max(128);
    let bm: u32 = if wide { 128 } else { 64 };
    let order = route_order_sliced(ctx, op, routes, &fan, experts, bm, true, share)?;
    let fire = if wide {
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
        // Stages of 64 routes' and 128 rows' 64 of K, four warps a CTA each
        // 32 routes by 64 rows: two CTAs an SM.
        let stages = ctx.tuning().bf16_gemm_stages().clamp(2, 4);
        const WN: u32 = 2;
        let cols = if glu.is_some() { 64 } else { 128 };
        Fire::at(kernel(
            SELECT_SM80,
            &format!("::pie::linear::moe_select_bf16_gemm<{stages}, {code}, {WN}, {bm}>"),
        ))
        .apply(
            Launch::grid([order.work_cap, y.width.div_ceil(cols), 1], [bm * WN, 1, 1])
                .smem(stages * (bm + 128) * 128),
        )
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
            ArgValue::Ptr(table.at()),
        ],
    )?;
    Ok(true)
}

/// A wide fire of a bf16-activation MXFP4 select on Hopper, read from its
/// bank's bf16 copy (`kernel.moe.mxfp4_bf16_from`, each value exactly the
/// code times its block's scale) on the warp-specialized wgmma GEMM, the
/// expert bias and the GLU in its epilogue. `Ok(false)` for a fire it does
/// not take: the caller reads the MXFP4 bank.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn select_bf16_copy(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: u64,
    experts: u32,
    bias: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    glu: Option<crate::linear::gemm_sm100::Glu>,
    table: ExpertTable,
) -> Result<bool, Error> {
    const STAGES: u32 = 4;
    let fan = selected(op, x, routes, y)?;
    if !hopper_wgmma()
        || x.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || !x.width.is_multiple_of(64)
        || !bank.is_multiple_of(16)
        || fan.route_count < ctx.tuning().mxfp4_bf16_from().max(1)
        || bias.is_some_and(|b| b.dtype != Dtype::Bf16)
        || matches!(glu, Some(crate::linear::gemm_sm100::Glu::Relu2))
    {
        return Ok(false);
    }
    let (code, limit, alpha) = glu_code(op, glu)?;
    let bank_rows = if glu.is_some() { 2 * y.width } else { y.width };
    let experts = experts.clamp(1, EXPERT_CAP);
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", bank_rows)?)?;
    let inter = stated(op, y.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
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
            ArgValue::Ptr(bias.map_or(0, |b| b.ptr)),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            inter.arg(),
            limit.arg(),
            alpha.arg(),
            ArgValue::Ptr(table.at()),
        ],
    )?;
    Ok(true)
}
