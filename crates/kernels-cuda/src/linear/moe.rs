use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;

const FILE: &str = "linear/moe.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const GEMV_WARPS: u32 = 4;

const MAX_EXPERTS: u32 = 512;
/// The sigmoid router stages twice as many (Kimi-K3 routes over 896).
const SIGMOID_MAX_EXPERTS: u32 = 1024;

const MAX_GRID_Y: u32 = 65_535;

const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

const fn router_lane(rows: u32) -> Launch {
    const ROUTER_BLOCK: u32 = 64;

    Launch::per_row(rows, ROUTER_BLOCK)
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
    router_extents_upto(op, logits, experts, top_k, MAX_EXPERTS)
}

/// [`router_extents`] for a router that stages up to `max` scores.
fn router_extents_upto(
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
fn topk_lane(t: &str, experts: u32, top_k: u32, rows: u32) -> Option<Fire> {
    if experts == 0 || experts > 512 || top_k == 0 || top_k > 16 {
        return None;
    }
    let per = experts.div_ceil(32).next_power_of_two();
    Some(
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::moe_topk_softmax_lane<{t}, {per}>")),
        )
        .apply(Launch::grid([rows.max(1), 1, 1], [32, 1, 1])),
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
    const OP: &str = "linear.moe_topk_softmax";
    let t = dtype_dispatch!(OP, logits.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ranked_planes(OP, logits, top_k, routes, weights);
    let (e, k) = router_extents(OP, logits, experts, top_k)?;
    if let Some(fire) = topk_lane(t, experts, top_k, logits.rows) {
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
    if let Some(fire) = topk_lane(t, experts, top_k, logits.rows) {
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
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::moe_topk_ranked<{t}, {softplus}, {per}>"
            )),
        )
        .apply(Launch::grid(
            [logits.rows.div_ceil(ROWS), 1, 1],
            [WARP * ROWS, 1, 1],
        )),
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
    const OP: &str = "linear.moe_topk_sigmoid";
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
    const OP: &str = "linear.moe_topk_sigmoid_sink";
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
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::moe_topk_sigmoid_sink<{t}>")),
        )
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
    const OP: &str = "linear.moe_topk_sqrt_softplus";
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

pub(crate) struct Selected {
    pub(crate) route_count: u32,

    pub(crate) top_k: i32,

    pub(crate) by_token: bool,
}

pub(crate) fn selected(
    op: &'static str,
    x: Tensor,
    routes: Tensor,
    y: &Tensor,
) -> Result<Selected, Error> {
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

    #[cfg(feature = "cuda")]
    {
        let fan = selected(OP, x, routes, y)?;
        if tc5_worth(
            ctx,
            x,
            (bank.dtype, bank.rows),
            y,
            experts,
            routes.rows,
            fan.route_count,
        ) {
            // A tcgen05 launch that fails falls back to the arm below, said
            // once a process.
            match select_tc5(
                ctx,
                OP,
                x,
                crate::linear::tc5::Operand::plain(bank.ptr, crate::linear::tc5::Elem::Bf16),
                bank.rows,
                routes,
                y,
                crate::linear::tc5::Epilogue::STORE,
            ) {
                Ok(()) => return Ok(()),
                Err(why) => tc5_fell_back(OP, &why),
            }
        }
        // Where there are no tcgen05 GEMMs, a decode's routes on mma.sync,
        // each expert's rows streamed once a slice rather than once a route.
        if !crate::linear::tc5::supported()
            && bank.dtype == Dtype::Bf16
            && !experts.streams()
            && select_bf16_held(ctx, OP, x, bank.ptr, bank.rows, routes, y, None)?
        {
            return Ok(());
        }
        if let Some(count) = grouped_worth(ctx, x, bank, y, &fan, experts) {
            return select_grouped(ctx, OP, x, bank, routes, y, count);
        }
        // Bf16 experts under a captured graph have no grouped path (tcgen05 is
        // Blackwell's, cuBLAS is not graph-capturable); the per-route GEMV
        // re-reads each expert once a route. The grouped mma.sync kernel reads
        // it once a slice.
        if bf16_mma_spans(ctx, fan.route_count)
            && x.dtype == Dtype::Bf16
            && bank.dtype == Dtype::Bf16
            && !experts.streams()
            && x.width.is_multiple_of(64)
            && y.width.is_multiple_of(8)
            && bank.rows >= 1
            && u64::from(bank.width) == u64::from(x.width) * u64::from(y.width)
        {
            return select_bf16_mma(ctx, OP, x, bank, routes, y, &fan, bank.rows);
        }
    }
    select_gemv(ctx, OP, x, bank, routes, y, experts)
}

/// Whether the graph-safe bf16 grouped mma.sync kernel takes a fire of
/// `routes` routes: `kernel.moe.bf16_mma_from ..= kernel.moe.bf16_mma_until`.
#[cfg(feature = "cuda")]
fn bf16_mma_spans(ctx: &Ctx, routes: u32) -> bool {
    let t = ctx.tuning();
    t.bf16_mma_until > 0 && t.bf16_mma_from <= routes && routes <= t.bf16_mma_until
}

/// `y[route] = x[route or its token] . W[routes[route]]^T` on the grouped
/// bf16 mma.sync kernel: routes laid out by expert on-device (graph-safe),
/// each expert's weight rows read once a slice.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn select_bf16_mma(
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
        Fire::at(
            "linear/moe_w4.cuh",
            symbol(&format!("::pie::linear::moe_select_bf16_mma<{nb}>")),
        )
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

/// Says, once a process, that a select the tcgen05 GEMM was to take fell
/// back to the GEMV, and why: a silent fallback reads as a slow kernel.
#[cfg(feature = "cuda")]
fn tc5_fell_back(op: &'static str, why: &Error) {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("kernels-cuda: `{op}` fell back from the tcgen05 grouped GEMM: {why}");
    }
}

/// Whether a bf16 select runs on the grouped tcgen05 GEMM: a resident bank
/// of `count` experts at widths its tiles take, from `kernel.moe.tc5_from`
/// tokens or two routes an expert (fewer keep the one-launch GEMV, which
/// reads an expert once a route).
#[cfg(feature = "cuda")]
pub(crate) fn tc5_worth(
    ctx: &Ctx,
    x: Tensor,
    (bank, count): (Dtype, u32),
    y: &Tensor,
    experts: ExpertTable,
    tokens: u32,
    routes: u32,
) -> bool {
    ctx.tuning().moe_tc5
        && (tokens >= ctx.tuning().moe_tc5_from || routes >= 2 * count || routes > MAX_GRID_Y)
        && x.dtype == Dtype::Bf16
        && bank == Dtype::Bf16
        && !experts.streams()
        && x.width.is_multiple_of(64)
        && y.width.is_multiple_of(64)
        && crate::linear::tc5::supported()
}

/// `y[route] = x[route or its token] . W[routes[route]]^T` on one grouped
/// tcgen05 GEMM: routes laid out by expert, rows scattered in, the product
/// gathered back to route order.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_tc5(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    w: crate::linear::tc5::Operand,
    experts: u32,
    routes: Tensor,
    y: &mut Tensor,
    epi: crate::linear::tc5::Epilogue,
) -> Result<(), Error> {
    use crate::linear::{act_quant, tc5};
    const GROUPED_X: &str = "linear.moe_tc5_x";
    const GROUPED_Y: &str = "linear.moe_tc5_y";
    let fan = selected(op, x, routes, y)?;
    let count = fan.route_count;
    if count == 0 {
        return Ok(());
    }
    let (n, k) = (y.width, x.width);
    // The routers this select serves name an expert once a token.
    let layout = act_quant::lay_out(ctx, op, routes, experts, tc5::GROUP_ALIGN, true)?;
    let div = if fan.by_token { routes.width } else { 1 };
    let xs = ctx.scratch_stream(op, GROUPED_X, layout.rows as usize * k as usize * 2)? as u64;
    act_quant::scatter_rows(ctx, op, x, div, layout.pos, count, xs)?;
    let ys = ctx.scratch_stream(op, GROUPED_Y, layout.rows as usize * n as usize * 2)? as u64;
    tc5::grouped_gemm_epi(
        ctx,
        tc5::Operand::plain(xs, tc5::Elem::Bf16),
        w,
        ys,
        tc5::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: 0,
        },
        [n, k],
        1.0,
        // An expert's padding rows land nowhere.
        crate::linear::tc5::Epilogue {
            live_rows: layout.live_rows(),
            ..epi
        },
        count,
    )?;
    act_quant::gather_back(ctx, op, ys, layout.pos, routes.width, y)
}

/// A gate-up bank a fused select reads (bf16, mxfp4 with an optional bias,
/// nvfp4, or affine u4): each expert's gate rows, then its up rows.
#[derive(Clone, Copy, Debug)]
pub enum GluBank {
    Bf16(Tensor),
    Mxfp4 {
        codes: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
    },
    Nvfp4 {
        codes: Tensor,
        scales: Tensor,
        tensor_scales: Tensor,
    },
    Affine {
        codes: Tensor,
        scales: Tensor,
        zeros: Tensor,
    },
    /// An NVFP4 bank against NVFP4 activations (W4A4): its scales laid in
    /// atoms, a tensor scale a row, an input scale an expert (the global its
    /// activations quantize under). Only [`experts`] takes it, both legs so.
    Nvfp4A4 {
        codes: Tensor,
        atoms: Tensor,
        tensor_scales: Tensor,
        input_scales: Tensor,
    },
    /// An MXFP4 bank against e4m3 activations in 1x128 blocks (W4A8): its
    /// ue8m0 scales laid in atoms. Only [`experts`] takes it, both legs so.
    Mxfp4A8 {
        codes: Tensor,
        atoms: Tensor,
    },
    /// A 128x128-block FP8 bank against e4m3 activations in 1x128 blocks
    /// (W8A8): e4m3 codes, their ue8m0 exponents laid in atoms a weight row
    /// an atom row. Only [`experts`] takes it, both legs so.
    Fp8A8 {
        codes: Tensor,
        atoms: Tensor,
    },
}

/// The group of an affine 4-bit bank of bf16 factors that the tcgen05
/// GEMM takes (32, 64 or 128 codes), from its planes' widths.
#[cfg(feature = "cuda")]
pub(crate) fn affine_u4_group(x: Tensor, codes: Tensor, scales: Tensor, n: u32) -> Option<u32> {
    let (k, n) = (u64::from(x.width), u64::from(n));
    if x.dtype != Dtype::Bf16 || u64::from(codes.width) * 2 != k * n || k % 64 != 0 {
        return None;
    }
    let groups = u64::from(scales.width).checked_div(2 * n)?;
    let group = k.checked_div(groups)?;
    (groups * group == k && matches!(group, 32 | 64 | 128)).then_some(group as u32)
}

/// A bank's tcgen05 operand, its experts, its bias and weight-scale planes
/// and the dtype `tc5_worth` reads, for a product `n` wide of `x`; `None`
/// for an affine bank whose group the GEMM does not take.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
fn bank_operand(
    x: Tensor,
    bank: GluBank,
    n: u32,
) -> Option<(crate::linear::tc5::Operand, u32, u64, u64, Dtype)> {
    use crate::linear::tc5;
    Some(match bank {
        GluBank::Bf16(b) => (
            tc5::Operand::plain(b.ptr, tc5::Elem::Bf16),
            b.rows,
            0,
            0,
            b.dtype,
        ),
        GluBank::Mxfp4 {
            codes,
            scales,
            bias,
        } => (
            tc5::Operand::scaled(codes.ptr, tc5::Elem::E2m1, scales.ptr, 32),
            codes.rows,
            bias.map_or(0, |b| b.ptr),
            0,
            Dtype::Bf16,
        ),
        GluBank::Nvfp4 {
            codes,
            scales,
            tensor_scales,
        } => (
            tc5::Operand::scaled(codes.ptr, tc5::Elem::E2m1, scales.ptr, 16),
            codes.rows,
            0,
            tensor_scales.ptr,
            Dtype::Bf16,
        ),
        GluBank::Affine {
            codes,
            scales,
            zeros,
        } => {
            let group = affine_u4_group(x, codes, scales, n)?;
            (
                tc5::Operand::affine(codes.ptr, scales.ptr, zeros.ptr, group),
                codes.rows,
                0,
                0,
                Dtype::Bf16,
            )
        }
        // Its activations quantize first: no bf16 operand reads it.
        GluBank::Nvfp4A4 { .. } | GluBank::Mxfp4A8 { .. } | GluBank::Fp8A8 { .. } => {
            return None;
        }
    })
}

/// `y[route] = glu(x[route or its token] . W[routes[route]]^T)`, the GLU in
/// the grouped tcgen05 GEMM's epilogue. `Ok(false)` where the fire does not
/// take it: the caller runs the select and the GLU apart.
#[cfg(feature = "cuda")]
pub fn select_glu(
    ctx: &Ctx,
    x: Tensor,
    bank: GluBank,
    routes: Tensor,
    y: &mut Tensor,
    glu: crate::linear::tc5::Glu,
    experts: ExpertTable,
) -> Result<bool, Error> {
    use crate::linear::tc5;
    const OP: &str = "linear.moe_select_glu";
    let fan = selected(OP, x, routes, y)?;
    // A narrow mxfp4 fire reads its bank faster on the register-dequant
    // kernel, which lands the GLU too.
    if let GluBank::Mxfp4 {
        codes,
        scales,
        bias,
    } = bank
        && mxfp4_mma_spans(ctx, fan.route_count)
        && !experts.streams()
        && x.dtype == Dtype::Bf16
        && x.width > 0
        && x.width.is_multiple_of(32)
    {
        // A launch that fails leaves the select and the GLU to run apart.
        return Ok(select_mxfp4_mma(
            ctx,
            OP,
            x,
            codes,
            scales,
            bias,
            routes,
            y,
            &fan,
            Some(glu),
            None,
            false,
        )
        .is_ok());
    }
    // A narrow NVFP4 (W4A16) fire likewise, where the profile says.
    if let GluBank::Nvfp4 {
        codes,
        scales,
        tensor_scales,
    } = bank
        && nvfp4_mma_worth(ctx, x, y, experts, &fan)
    {
        return Ok(select_mxfp4_mma(
            ctx,
            OP,
            x,
            codes,
            scales,
            None,
            routes,
            y,
            &fan,
            Some(glu),
            Some(tensor_scales),
            false,
        )
        .is_ok());
    }
    let Some((w, rows, bias, weight_scale, dtype)) = bank_operand(x, bank, 2 * y.width) else {
        return Ok(false);
    };
    if !tc5_worth(
        ctx,
        x,
        (dtype, rows),
        y,
        experts,
        routes.rows,
        fan.route_count,
    ) {
        return Ok(false);
    }
    // A tcgen05 launch that fails leaves the select and the GLU to run apart.
    Ok(select_tc5(
        ctx,
        OP,
        x,
        w,
        rows,
        routes,
        y,
        tc5::Epilogue {
            glu: Some(glu),
            bias,
            weight_scale,
            ..tc5::Epilogue::STORE
        },
    )
    .is_ok())
}

/// A routed MLP's operands: the activations, the gate-up and down banks
/// and the GLU between them, the routes and their weights, the GLU's width.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub struct RoutedMlp {
    pub x: Tensor,
    pub up: GluBank,
    pub glu: crate::linear::tc5::Glu,
    pub down: GluBank,
    pub routes: Tensor,
    pub weights: Tensor,
    pub inter: u32,
}

/// A routed MLP on the grouped tcgen05 GEMM, its rows kept by expert from
/// the gate-up select to the down select: routes laid out once, the GLU
/// landing from the gate-up epilogue into the down operand, the weighted
/// sum reading the down product in place. `norm`: each route's down row
/// normed under a scale first (`norm_gate`'s), but on W4A8 banks.
/// `Ok(false)`: run the four apart.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn experts(
    ctx: &Ctx,
    m: RoutedMlp,
    y: &mut Tensor,
    up_experts: ExpertTable,
    down_experts: ExpertTable,
    gate: Option<(Tensor, Tensor)>,
    norm: Option<(Tensor, f32)>,
) -> Result<bool, Error> {
    let RoutedMlp {
        x,
        up,
        glu,
        down,
        routes,
        weights,
        inter,
    } = m;
    use crate::linear::{act_quant, tc5};
    const OP: &str = "linear.moe_experts";
    if let Some((shared, g)) = gate
        && (shared.dtype != Dtype::Bf16
            || g.dtype != Dtype::Bf16
            || shared.rows != y.rows
            || shared.width != y.width
            || g.rows != y.rows
            || g.width == 0)
    {
        return Ok(false);
    }
    const GROUPED_X: &str = "linear.moe_tc5_x";
    const GROUPED_H: &str = "linear.moe_tc5_h";
    const GROUPED_Y: &str = "linear.moe_tc5_y";
    let count = routes.rows * routes.width;
    if let (GluBank::Nvfp4A4 { .. }, GluBank::Nvfp4A4 { .. }) = (up, down) {
        return experts_w4a4(ctx, m, y, gate, norm);
    }
    if let (GluBank::Mxfp4A8 { .. }, GluBank::Mxfp4A8 { .. })
    | (GluBank::Fp8A8 { .. }, GluBank::Fp8A8 { .. }) = (up, down)
    {
        if norm.is_some() {
            return Ok(false);
        }
        return experts_w4a8_as(ctx, m, y, W4a8Sum::Weighted(gate));
    }
    let hidden = Tensor::new(0, count, inter, Dtype::Bf16);
    let routed = Tensor::new(0, count, y.width, Dtype::Bf16);
    // An mxfp4 fire the register-dequant kernel takes stays on it.
    let mma = |bank: GluBank| match bank {
        GluBank::Mxfp4 { .. } => mxfp4_mma_spans(ctx, count),
        GluBank::Nvfp4 { .. } => nvfp4_mma_spans(ctx, count),
        _ => false,
    };
    if count == 0
        || norm.is_some_and(|(s, _)| s.dtype != Dtype::Bf16 || s.rows * s.width != y.width)
        || mma(up)
        || mma(down)
        || weights.dtype != Dtype::F32
        || weights.rows != routes.rows
        || weights.width != routes.width
        || y.rows != routes.rows
        || !y.width.is_multiple_of(8)
    {
        return Ok(false);
    }
    let (Some(u), Some(d)) = (
        bank_operand(x, up, 2 * inter),
        bank_operand(hidden, down, y.width),
    ) else {
        return Ok(false);
    };
    if !tc5_worth(ctx, x, (u.4, u.1), &hidden, up_experts, routes.rows, count)
        || !tc5_worth(
            ctx,
            hidden,
            (d.4, d.1),
            &routed,
            down_experts,
            routes.rows,
            count,
        )
        || u.1 != d.1
    {
        return Ok(false);
    }
    let experts = u.1;
    let mut run = || -> Result<(), Error> {
        let layout = act_quant::lay_out(ctx, OP, routes, experts, tc5::GROUP_ALIGN, true)?;
        let k = x.width;
        let rows = layout.rows as usize;
        let hs = ctx.scratch_stream(OP, GROUPED_H, rows * inter as usize * 2)? as u64;
        let grouped = tc5::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: 0,
        };
        let up_epi = tc5::Epilogue {
            glu: Some(glu),
            bias: u.2,
            weight_scale: u.3,
            live_rows: layout.live_rows(),
            ..tc5::Epilogue::STORE
        };
        let xs = ctx.scratch_stream(OP, GROUPED_X, rows * k as usize * 2)? as u64;
        act_quant::scatter_rows(ctx, OP, x, routes.width, layout.pos, count, xs)?;
        tc5::grouped_gemm_epi(
            ctx,
            tc5::Operand::plain(xs, tc5::Elem::Bf16),
            u.0,
            hs,
            grouped,
            [inter, k],
            1.0,
            up_epi,
            count,
        )?;
        let ys = ctx.scratch_stream(OP, GROUPED_Y, rows * y.width as usize * 2)? as u64;
        tc5::grouped_gemm_epi(
            ctx,
            tc5::Operand::plain(hs, tc5::Elem::Bf16),
            d.0,
            ys,
            grouped,
            [y.width, inter],
            1.0,
            tc5::Epilogue {
                bias: d.2,
                weight_scale: d.3,
                live_rows: layout.live_rows(),
                ..tc5::Epilogue::STORE
            },
            count,
        )?;
        act_quant::weighted_gather(ctx, OP, ys, layout.pos, weights, gate, norm, y)
    };
    // A tcgen05 launch that fails leaves the four to run apart.
    Ok(run().is_ok())
}

/// The most routes an expert on average whose W4A4 legs run swapped
/// (GLM-5.3's experts, measured).
#[cfg(feature = "cuda")]
const SWAPPED_UP_ROUTES: u32 = 64;

/// A W4A8 routed MLP of this many routes an expert or fewer runs its
/// gate-up swapped, in 64-row Q tiles (DeepSeek-V4-Flash, measured).
#[cfg(feature = "cuda")]
const SWAPPED_W4A8_UP_ROUTES: u32 = 32;

/// [`experts`] on two W4A4 banks: the GLU lands as the down GEMM's NVFP4
/// operand (rounded through bf16 first, as apart). Under
/// `kernel.moe.w4a4_fused_from` tokens the four run apart; a narrow fire
/// lays its routes out contiguous all the same.
#[cfg(feature = "cuda")]
fn experts_w4a4(
    ctx: &Ctx,
    m: RoutedMlp,
    y: &mut Tensor,
    gate: Option<(Tensor, Tensor)>,
    norm: Option<(Tensor, f32)>,
) -> Result<bool, Error> {
    let RoutedMlp {
        x,
        up,
        glu,
        down,
        routes,
        weights,
        inter,
    } = m;
    use crate::linear::{act_quant, tc5};
    const OP: &str = "linear.moe_experts";
    const W4A4_X: &str = "linear.moe_tc5_x4";
    const W4A4_XS: &str = "linear.moe_tc5_x4_atoms";
    const W4A4_H: &str = "linear.moe_tc5_h4";
    const W4A4_HS: &str = "linear.moe_tc5_h4_atoms";
    const W4A4_Y: &str = "linear.moe_tc5_y";
    const UP_ALPHA: &str = "linear.moe_tc5_up_alpha";
    const DOWN_ALPHA: &str = "linear.moe_tc5_down_alpha";
    let (
        GluBank::Nvfp4A4 {
            codes: up_codes,
            atoms: up_atoms,
            tensor_scales: up_ts,
            input_scales: up_is,
        },
        GluBank::Nvfp4A4 {
            codes: down_codes,
            atoms: down_atoms,
            tensor_scales: down_ts,
            input_scales: down_is,
        },
    ) = (up, down)
    else {
        return Ok(false);
    };
    let tokens = routes.rows;
    let count = tokens * routes.width;
    let experts = up_codes.rows;
    let (k, n) = (x.width, y.width);
    // Only SwiGLU's and relu2's epilogues land what the four apart would.
    if count == 0
        || norm.is_some_and(|(s, _)| s.dtype != Dtype::Bf16 || s.rows * s.width != y.width)
        || tokens < ctx.tuning().moe_w4a4_fused_from
        || !matches!(glu, tc5::Glu::Swiglu | tc5::Glu::Relu2)
        || !ctx.tuning().moe_tc5
        || !tc5::supported()
        || x.dtype != Dtype::Bf16
        || x.rows != tokens
        || down_codes.rows != experts
        || up_is.rows * up_is.width < experts
        || down_is.rows * down_is.width < experts
        || !k.is_multiple_of(128)
        || !inter.is_multiple_of(128)
        || !n.is_multiple_of(64)
        || weights.dtype != Dtype::F32
        || weights.rows != tokens
        || weights.width != routes.width
        || y.rows != tokens
        || !n.is_multiple_of(8)
    {
        return Ok(false);
    }
    let mut run = || -> Result<(), Error> {
        let e = experts as usize;
        let up_alpha = ctx.scratch_stream(OP, UP_ALPHA, e * 4)? as u64;
        let down_alpha = ctx.scratch_stream(OP, DOWN_ALPHA, e * 4)? as u64;
        let alphas = [
            (
                act_quant::AlphaFrom {
                    input_scales: up_is.ptr,
                    tensor_scales: up_ts.ptr,
                    rows: if glu == tc5::Glu::Relu2 {
                        inter
                    } else {
                        2 * inter
                    },
                },
                up_alpha,
            ),
            (
                act_quant::AlphaFrom {
                    input_scales: down_is.ptr,
                    tensor_scales: down_ts.ptr,
                    rows: n,
                },
                down_alpha,
            ),
        ];
        // An expert's rows start on a 64-row half block: a decode's few rows
        // an expert fill half the tiles a 128-row block would pad; contiguous,
        // as the gate-up's NVFP4 epilogue lands its atoms.
        let align = tc5::GROUP_ALIGN;
        // A narrow fire lays its routes out and quantizes their rows in one
        // launch where the profile says (`kernel.moe.w4a4_fused_layout`).
        let fused = ctx.tuning().moe_w4a4_fused_layout
            && count <= act_quant::FUSED_ROUTES
            && experts <= act_quant::FUSED_EXPERTS;
        let rows = act_quant::contiguous_rows(count, experts, align) as usize;
        let xq = ctx.scratch_stream(OP, W4A4_X, rows * k as usize / 2)? as u64;
        let x_atom_bytes = tc5::atom_bytes(rows, k as usize, 16);
        let xs = act_quant::atoms_of(ctx, OP, W4A4_XS, x_atom_bytes)?;
        let layout = if fused {
            act_quant::lay_out_quant_nvfp4(
                ctx, OP, routes, experts, align, alphas, x, up_is, xq, xs,
            )?
        } else {
            let layout = act_quant::lay_out_with(ctx, OP, routes, experts, align, None, &alphas)?;
            act_quant::quantize(
                ctx,
                OP,
                act_quant::Format::Nvfp4,
                act_quant::Source {
                    x,
                    rows: count,
                    div: routes.width,
                    routes: Some(routes),
                    input_scales: Some(up_is),
                    pos: layout.pos,
                    m_max: 0,
                },
                xq,
                xs,
                x_atom_bytes,
            )?;
            layout
        };
        let rows = layout.rows as usize;
        let hq = ctx.scratch_stream(OP, W4A4_H, rows * inter as usize / 2)? as u64;
        let hs = ctx.scratch_stream(OP, W4A4_HS, tc5::atom_bytes(rows, inter as usize, 16))? as u64;
        let grouped_up = tc5::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: up_alpha,
        };
        // A decode's few rows an expert run the gate-up swapped: a thread
        // a weight row, its epilogue over the expert's rows alone (unswapped,
        // a thread a row, the whole 128-row block's).
        let sms = ctx.multiprocessors().unwrap_or(148);
        // A Q tile of 32 rows where every expert's fit one (the fused
        // layout states its rows the fire's tokens at most), else 64.
        let swapped = tc5::Plan {
            swap: true,
            pair: false,
            bn: if fused && routes.rows <= 32 { 32 } else { 64 },
            stages: 0,
            splits: 1,
        };
        let plan = if count <= SWAPPED_UP_ROUTES * experts {
            swapped
        } else {
            tc5::Plan::grouped(&grouped_up, inter, k, count, sms)
        };
        tc5::grouped_gemm_epi_with(
            ctx,
            tc5::Operand::scaled(xq, tc5::Elem::E2m1, xs, 16),
            tc5::Operand::scaled(up_codes.ptr, tc5::Elem::E2m1, up_atoms.ptr, 16),
            hq,
            grouped_up,
            [inter, k],
            1.0,
            tc5::Epilogue {
                glu: Some(glu),
                out: tc5::Out::Nvfp4 {
                    data: hq,
                    scales: hs,
                    gscale: 1.0,
                },
                live_rows: layout.live_rows(),
                input_scales: down_is.ptr,
                ..tc5::Epilogue::STORE
            },
            plan,
        )?;
        let ys = ctx.scratch_stream(OP, W4A4_Y, rows * n as usize * 2)? as u64;
        let grouped_down = tc5::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: down_alpha,
        };
        let down_plan = if count <= SWAPPED_UP_ROUTES * experts {
            swapped
        } else {
            tc5::Plan::grouped(&grouped_down, n, inter, count, sms)
        };
        tc5::grouped_gemm_epi_with(
            ctx,
            tc5::Operand::scaled(hq, tc5::Elem::E2m1, hs, 16),
            tc5::Operand::scaled(down_codes.ptr, tc5::Elem::E2m1, down_atoms.ptr, 16),
            ys,
            grouped_down,
            [n, inter],
            1.0,
            tc5::Epilogue {
                live_rows: layout.live_rows(),
                weights_ahead: true,
                ..tc5::Epilogue::STORE
            },
            down_plan,
        )?;
        act_quant::weighted_gather(ctx, OP, ys, layout.pos, weights, gate, norm, y)
    };
    // A tcgen05 launch that fails leaves the four to run apart.
    Ok(run().is_ok())
}

/// How a W4A8 routed MLP's routes fold into a token's row.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum W4a8Sum {
    /// Each route under its weight (`moe_weighted_sum`), under a gate plus
    /// the shared row at its sigmoid.
    Weighted(Option<(Tensor, Tensor)>),
    /// Each route under its weight before the down leg (the GLU's rows
    /// scaled before they quantize), then summed plainly and the shared row
    /// added (`moe_sum_add`): DeepSeek's.
    WeightedFirst(Tensor),
}

/// [`experts`]' W4A8 routed MLP the DeepSeek way: route weights scale the
/// GLU's rows before the down leg quantizes them, and the routes sum plainly
/// with the shared row added. `Ok(false)`: run the four apart.
#[cfg(feature = "cuda")]
pub fn experts_weighted_first(
    ctx: &Ctx,
    m: RoutedMlp,
    shared: Tensor,
    y: &mut Tensor,
) -> Result<bool, Error> {
    if shared.dtype != Dtype::Bf16 || shared.rows != y.rows || shared.width != y.width {
        return Ok(false);
    }
    experts_w4a8_as(ctx, m, y, W4a8Sum::WeightedFirst(shared))
}

/// [`experts`] on two W4A8 banks (or two block-FP8 ones) (MXFP4 codes against e4m3 activations in
/// 1x128 blocks): the GLU lands as the down GEMM's e4m3 operand. Under
/// `kernel.moe.w4a8_fused_from` tokens, or masked, the four run apart; but
/// a block-FP8 pair's masked fire (a decode's) runs fused on the masked
/// layout, laid out once: the gate-up lands bf16 and one pass takes its
/// SwiGLU into the down GEMM's e4m3 operand.
#[cfg(feature = "cuda")]
fn experts_w4a8_as(ctx: &Ctx, m: RoutedMlp, y: &mut Tensor, sum: W4a8Sum) -> Result<bool, Error> {
    let RoutedMlp {
        x,
        up,
        glu,
        down,
        routes,
        weights,
        inter,
    } = m;
    use crate::linear::{act_quant, tc5};
    const OP: &str = "linear.moe_experts";
    const W4A8_X: &str = "linear.moe_tc5_x8";
    const W4A8_XS: &str = "linear.moe_tc5_x8_atoms";
    const W4A8_H: &str = "linear.moe_tc5_h8";
    const W4A8_HS: &str = "linear.moe_tc5_h8_atoms";
    const W4A8_Y: &str = "linear.moe_tc5_y";
    const ROW_WEIGHTS: &str = "linear.moe_tc5_row_weights";
    const GRAN: u32 = 128;
    // The weights' element and their scale's span: MXFP4 codes a 32 under
    // ue8m0, or e4m3 a 128 (a block-FP8 bank's rows) under ue8m0.
    let (up_codes, up_atoms, down_codes, down_atoms, w_elem, w_gran) = match (up, down) {
        (
            GluBank::Mxfp4A8 {
                codes: up_codes,
                atoms: up_atoms,
            },
            GluBank::Mxfp4A8 {
                codes: down_codes,
                atoms: down_atoms,
            },
        ) => (
            up_codes,
            up_atoms,
            down_codes,
            down_atoms,
            tc5::Elem::E2m1,
            32,
        ),
        (
            GluBank::Fp8A8 {
                codes: up_codes,
                atoms: up_atoms,
            },
            GluBank::Fp8A8 {
                codes: down_codes,
                atoms: down_atoms,
            },
        ) => (
            up_codes,
            up_atoms,
            down_codes,
            down_atoms,
            tc5::Elem::E4m3,
            128,
        ),
        _ => return Ok(false),
    };
    let tokens = routes.rows;
    let count = tokens * routes.width;
    let experts = up_codes.rows;
    let (k, n) = (x.width, y.width);
    // SwiGLU, its clamp and SiTU land from the epilogue as the apart GLU
    // does.
    let glu_taken = matches!(
        glu,
        tc5::Glu::Swiglu | tc5::Glu::SwigluClamp { .. } | tc5::Glu::Situ { .. }
    );
    let masked = w_gran == 128
        && count <= act_quant::MASKED_FIRE
        && glu == tc5::Glu::Swiglu
        && matches!(sum, W4a8Sum::Weighted(_));
    if (count <= act_quant::MASKED_FIRE || tokens < ctx.tuning().moe_w4a8_fused_from) && !masked
        || !glu_taken
        || !ctx.tuning().moe_tc5
        || !tc5::supported()
        || x.dtype != Dtype::Bf16
        || x.rows != tokens
        || down_codes.rows != experts
        || !k.is_multiple_of(GRAN)
        || !inter.is_multiple_of(GRAN)
        || !n.is_multiple_of(64)
        || weights.dtype != Dtype::F32
        || weights.rows != tokens
        || weights.width != routes.width
        || y.rows != tokens
    {
        return Ok(false);
    }
    let mut run = || -> Result<(), Error> {
        let layout = if masked {
            act_quant::lay_out_with(
                ctx,
                OP,
                routes,
                experts,
                act_quant::BLOCK_ROWS,
                Some(false),
                &[],
            )?
        } else {
            act_quant::lay_out(ctx, OP, routes, experts, act_quant::BLOCK_ROWS, false)?
        };
        let rows = layout.rows as usize;
        let m_max = layout.m_max as usize;
        // A masked layout's scale atoms: each expert's whole 128-row blocks.
        let atoms_for = |cols: u32| {
            if m_max > 0 {
                tc5::masked_atom_bytes(experts as usize, m_max, cols as usize, GRAN as usize)
            } else {
                tc5::atom_bytes(rows, cols as usize, GRAN as usize)
            }
        };
        let fmt = act_quant::Format::E4m3 { gran: GRAN };
        let xq = ctx.scratch_stream(OP, W4A8_X, rows * k as usize)? as u64;
        let x_atom_bytes = atoms_for(k);
        let xs = act_quant::atoms_of(ctx, OP, W4A8_XS, x_atom_bytes)?;
        act_quant::quantize(
            ctx,
            OP,
            fmt,
            act_quant::Source {
                x,
                rows: count,
                div: routes.width,
                routes: None,
                input_scales: None,
                pos: layout.pos,
                m_max: layout.m_max,
            },
            xq,
            xs,
            x_atom_bytes,
        )?;
        let hq = ctx.scratch_stream(OP, W4A8_H, rows * inter as usize)? as u64;
        // A masked layout's padding rows keep their atoms: a zeroed slab.
        let hs = if masked {
            act_quant::atoms_of(ctx, OP, W4A8_HS, atoms_for(inter))?
        } else {
            ctx.scratch_stream(OP, W4A8_HS, atoms_for(inter))? as u64
        };
        // DeepSeek's weights scale the GLU's rows, each at its grouped row.
        let row_scale = match sum {
            W4a8Sum::Weighted(_) => 0,
            W4a8Sum::WeightedFirst(_) => {
                let at = ctx.scratch_stream(OP, ROW_WEIGHTS, rows * 4)? as u64;
                act_quant::scatter_f32(ctx, OP, weights.ptr, layout.pos, count, at)?;
                at
            }
        };
        let grouped = tc5::Grouped {
            experts,
            groups: layout.groups,
            alpha_e: 0,
        };
        if masked {
            const W4A8_U: &str = "linear.moe_tc5_u";
            let us = ctx.scratch_stream(OP, W4A8_U, rows * 2 * inter as usize * 2)? as u64;
            tc5::grouped_gemm_epi(
                ctx,
                tc5::Operand::scaled(xq, tc5::Elem::E4m3, xs, GRAN),
                tc5::Operand::scaled(up_codes.ptr, w_elem, up_atoms.ptr, w_gran),
                us,
                grouped,
                [2 * inter, k],
                1.0,
                tc5::Epilogue::STORE,
                count,
            )?;
            act_quant::swiglu_quant(ctx, OP, us, layout.rows, inter, layout.m_max, hq, hs)?;
        } else {
            // A few rows an expert run the gate-up swapped (bank on P, rows on 64-row
            // Q tiles): the same products at a quarter the MMA work.
            let sms = ctx.multiprocessors().unwrap_or(148);
            let up_plan = if count <= SWAPPED_W4A8_UP_ROUTES * experts {
                tc5::Plan {
                    swap: true,
                    pair: false,
                    bn: 64,
                    stages: 0,
                    splits: 1,
                }
            } else {
                tc5::Plan::grouped(&grouped, inter, k, count, sms)
            };
            tc5::grouped_gemm_epi_with(
                ctx,
                tc5::Operand::scaled(xq, tc5::Elem::E4m3, xs, GRAN),
                tc5::Operand::scaled(up_codes.ptr, w_elem, up_atoms.ptr, w_gran),
                hq,
                grouped,
                [inter, k],
                1.0,
                tc5::Epilogue {
                    glu: Some(glu),
                    out: tc5::Out::Fp8Block {
                        data: hq,
                        scales: hs,
                    },
                    row_scale,
                    live_rows: layout.live_rows(),
                    ..tc5::Epilogue::STORE
                },
                up_plan,
            )?;
        }
        let ys = ctx.scratch_stream(OP, W4A8_Y, rows * n as usize * 2)? as u64;
        tc5::grouped_gemm_epi(
            ctx,
            tc5::Operand::scaled(hq, tc5::Elem::E4m3, hs, GRAN),
            tc5::Operand::scaled(down_codes.ptr, w_elem, down_atoms.ptr, w_gran),
            ys,
            grouped,
            [n, inter],
            1.0,
            tc5::Epilogue {
                live_rows: layout.live_rows(),
                ..tc5::Epilogue::STORE
            },
            count,
        )?;
        match sum {
            W4a8Sum::Weighted(gate) => {
                act_quant::weighted_gather(ctx, OP, ys, layout.pos, weights, gate, None, y)
            }
            W4a8Sum::WeightedFirst(shared) => {
                act_quant::sum_gather(ctx, OP, ys, layout.pos, routes.width, shared, y)
            }
        }
    };
    // A tcgen05 launch that fails leaves the four to run apart.
    Ok(run().is_ok())
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
const GROUP_WORKSPACE_CAP: u64 = 512 << 20;

#[cfg(feature = "cuda")]
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
    if experts.streams() || !matches!(ctx.stage(), ArgValue::Ptr(0)) {
        return None;
    }
    if !matches!(x.dtype, Dtype::Bf16 | Dtype::F16) || bank.dtype != x.dtype {
        return None;
    }
    let count = bank.rows;
    if count == 0 || count > MAX_EXPERTS {
        return None;
    }
    if u64::from(bank.width) != u64::from(x.width) * u64::from(y.width) {
        return None;
    }
    let per_expert = fan.route_count.div_ceil(count);
    if per_expert < ctx.tuning().moe_group_at {
        return None;
    }
    let block = block_rows(per_expert);
    let blocks = u64::from(count) + u64::from(fan.route_count.div_ceil(block));
    let rows = blocks * u64::from(block);
    let staged = rows
        .checked_mul(u64::from(x.width) + u64::from(y.width))?
        .checked_mul(x.dtype.bytes_ceil())?;
    (staged <= GROUP_WORKSPACE_CAP).then_some(count)
}

#[cfg(feature = "cuda")]
fn select_grouped(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: u32,
) -> Result<(), Error> {
    use cudarc::cublas::sys::{
        cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmBatchedEx, cublasHandle_t,
        cublasOperation_t, cublasStatus_t, cudaDataType,
    };

    let (t, scalar) = match x.dtype {
        Dtype::Bf16 => ("::pie::bf16", cudaDataType::CUDA_R_16BF),
        Dtype::F16 => ("::pie::f16", cudaDataType::CUDA_R_16F),
        other => {
            return Err(refuse(
                op,
                format!("the grouped leg hands cuBLAS a 16-bit bank and this one is {other:?}"),
            ));
        }
    };
    let handle: cublasHandle_t = ctx.cublas(op)?.cast::<cublasContext>();

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
            // One slab a stream: the layout lives within this op, so
            // regions captured apart share it rather than holding one each.
            .and_then(|bytes| ctx.scratch_stream(op, name, bytes))
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

    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: `handle` is this context's live cuBLAS handle bound to its
    // stream; the pointer arrays hold `blocks` entries in slabs this context
    // owns for the fire's life.
    let status = unsafe {
        cublasGemmBatchedEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            block_size,
            k,
            std::ptr::from_ref(&alpha).cast(),
            w_ptrs as usize as *const *const std::ffi::c_void,
            scalar,
            k,
            act_ptrs as usize as *const *const std::ffi::c_void,
            scalar,
            k,
            std::ptr::from_ref(&beta).cast(),
            out_ptrs as usize as *const *mut std::ffi::c_void,
            scalar,
            n,
            max_blocks,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(refuse(
            op,
            format!(
                "`cublasGemmBatchedEx` answered {status:?} at {max_blocks} blocks of \
                 M={block_size} N={n} K={k}"
            ),
        ));
    }

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
    if fan.route_count == 0 {
        return Ok(());
    }
    // A run past grid.y's 65535 rows continues on grid.z; every run within
    // it launches as it did.
    let (grid_y, grid_z) = if fan.route_count > MAX_GRID_Y {
        (MAX_GRID_Y, fan.route_count.div_ceil(MAX_GRID_Y))
    } else {
        (fan.route_count, 1)
    };
    if x.width == 0 || !x.width.is_multiple_of(VEC_WIDTH) {
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
    // Four weight rows a warp, their loads in flight together: Qwen3.6's
    // layer at 8 decode tokens (gate-up K 2048, down K 512) 55 + 37 us at one
    // row a warp, 40 + 21 at four (two 44 + 25, eight 45 + 24).
    const ROWS: u32 = 4;
    let entry = format!("::pie::linear::moe_matmul_select_gemv_{form}<{t}, {ROWS}>");
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&entry)).apply(Launch::grid(
            [y.width.div_ceil(GEMV_WARPS * ROWS), grid_y, grid_z],
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
            ArgValue::Ptr(experts.table),
            ArgValue::Ptr(experts.hits),
            stated(op, fan.route_count)?.arg(),
            ctx.stage(),
        ],
    )
}

pub(crate) const EXPERT_CAP: u32 = 4096;
pub(crate) const GROUPED_TILE_N: u32 = 128;
pub(crate) const GROUPED_BLOCK: u32 = 256;
const WMMA_ROUTES: u32 = 32;

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
enum Share {
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
struct OrderKey {
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
    static KEEP_NEXT: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
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
fn route_order_sliced(
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
            Fire::at(
                "linear/moe_w4.cuh",
                symbol("::pie::linear::moe_route_order_scan"),
            )
        } else {
            Fire::at("linear/quant.cuh", symbol("::pie::linear::moe_route_order"))
        }
        .apply(Launch::grid([1, 1, 1], [ORDER_BLOCK, 1, 1]).smem(2 * (experts + 1) * 4)),
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

/// Whether the register-dequant MMA kernel takes a fire of `routes` routes:
/// `kernel.moe.mxfp4_mma_from ..= kernel.moe.mxfp4_mma_until`.
#[cfg(feature = "cuda")]
fn mxfp4_mma_spans(ctx: &Ctx, routes: u32) -> bool {
    let t = ctx.tuning();
    t.mxfp4_mma_from <= routes && routes <= t.mxfp4_mma_until
}

#[cfg(feature = "cuda")]
fn nvfp4_mma_spans(ctx: &Ctx, routes: u32) -> bool {
    let t = ctx.tuning();
    t.nvfp4_mma_from <= routes && routes <= t.nvfp4_mma_until
}

/// Whether a routed NVFP4 select of bf16 activations (W4A16) runs the
/// register-dequant MMA kernel: resident experts, whole 32-code groups, and
/// a width `kernel.moe.nvfp4_mma_from` to `_until` takes.
#[cfg(feature = "cuda")]
pub(crate) fn nvfp4_mma_worth(
    ctx: &Ctx,
    x: Tensor,
    y: &Tensor,
    experts: ExpertTable,
    fan: &Selected,
) -> bool {
    nvfp4_mma_spans(ctx, fan.route_count)
        && !experts.streams()
        && x.dtype == Dtype::Bf16
        && y.dtype == Dtype::Bf16
        && x.width > 0
        && x.width.is_multiple_of(32)
}

/// The routed NVFP4 select of bf16 activations on the register-dequant MMA
/// kernel: `scales` row-major e4m3 a 16 codes, `tensor_scales` an f32 a row.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_nvfp4_mma(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<bool, Error> {
    let fan = selected(op, x, routes, y)?;
    if !nvfp4_mma_worth(ctx, x, y, experts, &fan) {
        return Ok(false);
    }
    select_mxfp4_mma(
        ctx,
        op,
        x,
        codes,
        scales,
        None,
        routes,
        y,
        &fan,
        None,
        Some(tensor_scales),
        false,
    )?;
    Ok(true)
}

/// Whether a routed mxfp4 select runs the register-dequant MMA kernel: bf16
/// activations, resident experts, whole 32-code blocks, and a width
/// [`mxfp4_mma_spans`] takes.
#[cfg(feature = "cuda")]
pub(crate) fn mxfp4_mma_worth(
    ctx: &Ctx,
    x: Tensor,
    y: &Tensor,
    seat: GroupSeat,
    fan: &Selected,
) -> bool {
    mxfp4_mma_spans(ctx, fan.route_count)
        && !seat.streams()
        && x.dtype == Dtype::Bf16
        && y.dtype == Dtype::Bf16
        && x.width > 0
        && x.width.is_multiple_of(32)
}

/// A select's GLU as its kernels' `kGlu` code and the GLU's two
/// constants (`linear/glu.cuh`); the register-dequant selects land GLUs
/// alone, not relu2.
#[cfg(feature = "cuda")]
fn glu_code(
    op: &'static str,
    glu: Option<crate::linear::tc5::Glu>,
) -> Result<(i32, f32, f32), Error> {
    use crate::linear::tc5::Glu;
    Ok(match glu {
        None => (0, 0.0, 1.0),
        Some(Glu::Swiglu) => (1, 0.0, 1.0),
        Some(Glu::SwigluClamp { limit }) => (2, limit, 1.0),
        Some(Glu::SwigluClampAlpha { limit, alpha }) => (3, limit, alpha),
        Some(Glu::GegluTanh) => (4, 0.0, 1.0),
        Some(Glu::Situ { beta, cap }) => (5, beta, cap),
        Some(Glu::Relu2) => {
            return Err(refuse(
                op,
                "the register-dequant select lands GLUs, not relu2",
            ));
        }
    })
}

/// The most routes an expert averages that the register-held mxfp4
/// select takes; past it, the shared-memory-staged one.
const STAGED_FROM: u32 = 32;

/// The 128-code chunks a warp of the register-held mxfp4 select keeps in
/// flight: one, five CTAs an SM; a decode's few CTAs take more warps from
/// the K split instead.
const MMA_DEPTH: u32 = 1;

/// The deepest K split the register-held mxfp4 select's CTA takes.
const MMA_KSPLIT_MAX: u32 = 4;

/// The routed mxfp4 select on mma.sync, each expert's rows dequantized in
/// registers (`linear/moe_w4.cuh`); with `glu`, `y` is the GLU of the bank's
/// gate rows and the up rows after them, half the bank's width.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn select_mxfp4_mma(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    fan: &Selected,
    glu: Option<crate::linear::tc5::Glu>,
    tensor_scales: Option<Tensor>,
    scale_atoms: bool,
) -> Result<(), Error> {
    const THREADS: u32 = 128;
    // The converts read e8m0 scales a 32 codes and no tensor scale, so an
    // NVFP4 bank (e4m3 a 16 codes, a tensor scale a row) keeps the bf16 path.
    let cvt = ctx.tuning().mxfp4_cvt && crate::linear::tc5::supported() && tensor_scales.is_none();
    if scale_atoms && !cvt {
        return Err(refuse(
            op,
            "scales in tcgen05's atoms are read by the e2m1-convert select alone",
        ));
    }
    let (code, limit, alpha) = glu_code(op, glu)?;
    let bank_rows = if glu.is_some() { 2 * y.width } else { y.width };
    let experts = codes.rows.clamp(1, EXPERT_CAP);
    // An expert's slice is 8, 16 or 32 routes: as few 8-route blocks as the
    // average expert fills.
    let per = fan.route_count.div_ceil(experts);
    // Past `STAGED_FROM` routes an expert, its activations staged in shared
    // memory a CTA of 64 routes (`moe_select_mxfp4_mma_staged`), so a wide
    // fire streams each expert's codes once a 64.
    let staged = !cvt && per > STAGED_FROM;
    let nb: u32 = match per {
        _ if staged => 8,
        0..=4 => 1,
        5..=12 => 2,
        _ => 4,
    };
    // A CTA lands 128 rows, or 64 GLU columns (their gate and up rows).
    let cols = if glu.is_some() { 64 } else { 128 };
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", bank_rows)?)?;
    let inter = stated(op, y.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
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
        if glu.is_some() || KEEP_NEXT.with(core::cell::Cell::get) {
            Share::Keep
        } else {
            Share::Take
        },
    )?;
    // A fire of a few routes splits K across a cluster of two CTAs
    // (`kernel.moe.mxfp4_cvt_ksplit_until`), each streaming half the codes.
    let ksplit = if cvt && fan.route_count <= ctx.tuning().mxfp4_cvt_ksplit_until {
        2
    } else {
        1
    };
    let grid = [work_cap * ksplit, y.width.div_ceil(cols), 1];
    // The e2m1 converts exist on the tcgen05 Blackwells alone. Two chunks in
    // flight a warp, a window of 8 chunks of one 8-route block: four CTAs an SM.
    let fire = if cvt {
        const DEPTH: u32 = 2;
        let win = 8 / nb;
        let blocks = match nb {
            1 => 4,
            2 => 3,
            _ => 2,
        };
        Fire::at(
            "linear/moe_w4_cvt.cuh",
            symbol(&format!(
                "::pie::linear::moe_select_mxfp4_cvt<{nb}, {code}, {DEPTH}, {win}, {blocks}, \
                 {ksplit}>"
            )),
        )
        .apply(
            Launch::grid(grid, [THREADS, 1, 1])
                .smem(nb * win * 4096)
                .cluster(ksplit),
        )
    } else if staged {
        const STAGES: u32 = 3;
        let nv = tensor_scales.is_some();
        Fire::at(
            "linear/moe_w4.cuh",
            symbol(&format!(
                "::pie::linear::moe_select_mxfp4_mma_staged<{nb}, {STAGES}, {code}, {nv}>"
            )),
        )
        .apply(Launch::grid(grid, [THREADS, 1, 1]).smem(STAGES * 8 * nb * 16 * 16))
    } else {
        let depth = MMA_DEPTH;
        // A fire of few CTAs (a decode's routes) splits K across groups of
        // the CTA's warps, so the card's memory has warps enough to stream
        // the bank: as many as bring it to four CTAs' worth an SM.
        let sms = ctx.multiprocessors().unwrap_or(108);
        let ctas = (grid[0] * grid[1]).max(1);
        let ksplit = if nb == 1 {
            (4 * sms)
                .div_ceil(ctas)
                .next_power_of_two()
                .clamp(1, MMA_KSPLIT_MAX)
        } else {
            1
        };
        let nv = tensor_scales.is_some();
        Fire::at(
            "linear/moe_w4.cuh",
            // NVFP4 (`nv`): an e4m3 scale a 16 codes, a tensor scale a row.
            symbol(&format!(
                "::pie::linear::moe_select_mxfp4_mma<{nb}, {depth}, {code}, {nv}, {ksplit}>"
            )),
        )
        .apply(Launch::grid(grid, [THREADS * ksplit, 1, 1]))
    };
    let mut args = vec![
        x.arg(),
        ArgValue::Ptr(order),
        ArgValue::Ptr(offsets),
        ArgValue::Ptr(work),
        codes.arg(),
        scales.arg(),
        bias.map_or(ArgValue::ABSENT, |bias| bias.arg()),
        y.arg(),
        act_div.arg(),
        n.arg(),
        k.arg(),
        stated(op, experts)?.arg(),
        inter.arg(),
        limit.arg(),
        alpha.arg(),
    ];
    // The e2m1-convert kernel reads where its scales lie; the bf16 one, an
    // NVFP4 bank's tensor scales.
    if cvt {
        args.push(i32::from(scale_atoms).arg());
    } else {
        args.push(tensor_scales.map_or(ArgValue::ABSENT, |t| t.arg()));
    }
    ctx.fire(op, fire, &args)
}

/// How a W8A16 or bf16 select's fire is cut: an expert's routes in slices
/// of `8 nb` (a tiled GEMM's 64 where `tiled`), and on the register-held
/// kernel its m16 tiles of weight rows a warp and its K split.
#[cfg(feature = "cuda")]
struct SelectPlan {
    nb: u32,
    tiled: bool,
    tiles: u32,
    ksplit: u32,
    order: RouteOrder,
}

/// The [`SelectPlan`] of a select of `fan` over `experts` landing `y`
/// (`glu` where it lands a GLU), on the tiled GEMM where `gemm` and its
/// experts average more than 12 routes.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn select_plan(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    fan: &Selected,
    experts: u32,
    y: &Tensor,
    glu: bool,
    gemm: bool,
) -> Result<SelectPlan, Error> {
    let nb: u32 = match fan.route_count.div_ceil(experts) {
        0..=4 => 1,
        5..=12 => 2,
        _ if !gemm => 4,
        _ => 8,
    };
    let order = route_order_sliced(
        ctx,
        op,
        routes,
        fan,
        experts,
        8 * nb,
        true,
        if glu || KEEP_NEXT.with(core::cell::Cell::get) {
            Share::Keep
        } else {
            Share::Take
        },
    )?;
    // A decode's few CTAs: half the rows a warp (twice the CTAs), then K
    // split across groups of a CTA's warps, toward four CTAs' warps an SM.
    let sms = ctx.multiprocessors().unwrap_or(108);
    let tiles: u32 = if !glu && order.work_cap * y.width.div_ceil(128) < 2 * sms {
        1
    } else {
        2
    };
    let cols = if glu { 64 } else { 64 * tiles };
    let ctas = (order.work_cap * y.width.div_ceil(cols)).max(1);
    let ksplit = if nb == 1 {
        (4 * sms)
            .div_ceil(ctas)
            .next_power_of_two()
            .clamp(1, MMA_KSPLIT_MAX)
    } else {
        1
    };
    Ok(SelectPlan {
        nb,
        tiled: nb == 8,
        tiles,
        ksplit,
        order,
    })
}

/// A routed select over a block-FP8 bank (`experts x n x k` e4m3 codes at
/// `codes`, one ue8m0 exponent a weight row a `gran` columns in atoms at
/// `atoms`) against bf16 activations (W8A16, `linear/moe_w8.cuh`), with the
/// GLU in its epilogue where `glu` says: the select where there are no FP8
/// tensor cores.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn select_fp8_w8a16(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: u64,
    atoms: u64,
    experts: u32,
    gran: u32,
    routes: Tensor,
    y: &mut Tensor,
    glu: Option<crate::linear::tc5::Glu>,
) -> Result<(), Error> {
    const THREADS: u32 = 128;
    const DEPTH: u32 = 2;
    let fan = selected(op, x, routes, y)?;
    if x.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(refuse(op, "a W8A16 select reads and lands bf16 rows"));
    }
    if !matches!(gran, 32 | 128) || !x.width.is_multiple_of(128) {
        return Err(refuse(
            op,
            format!(
                "a W8A16 select walks whole 128s of K in {gran}-wide blocks, not K {}",
                x.width
            ),
        ));
    }
    let (code, limit, alpha) = glu_code(op, glu)?;
    let bank_rows = if glu.is_some() { 2 * y.width } else { y.width };
    let experts = experts.clamp(1, EXPERT_CAP);
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", bank_rows)?)?;
    let inter = stated(op, y.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let SelectPlan {
        nb,
        tiled,
        tiles,
        ksplit,
        order,
    } = select_plan(
        ctx,
        op,
        routes,
        &fan,
        experts,
        y,
        glu.is_some(),
        fan.route_count >= FP8_GEMM_FROM,
    )?;
    let fire = if tiled {
        // Three stages of 64 routes' activations and 128 rows' codes, 64 of
        // K, and the widened tile: two CTAs an SM.
        const STAGES: u32 = 3;
        const WN: u32 = 2;
        let cols = if glu.is_some() { 64 } else { 128 };
        Fire::at(
            "linear/moe_w8.cuh",
            symbol(&format!(
                "::pie::linear::moe_select_fp8_gemm<{STAGES}, {code}, {gran}, {WN}>"
            )),
        )
        .apply(
            Launch::grid([order.work_cap, y.width.div_ceil(cols), 1], [64 * WN, 1, 1])
                .smem(STAGES * (64 * 128 + 128 * 64) + 128 * 128),
        )
    } else {
        let cols = if glu.is_some() { 64 } else { 64 * tiles };
        Fire::at(
            "linear/moe_w8.cuh",
            symbol(&format!(
                "::pie::linear::moe_select_fp8_mma<{nb}, {DEPTH}, {code}, {gran}, {tiles}, {ksplit}>"
            )),
        )
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
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            y.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            stated(op, experts)?.arg(),
            inter.arg(),
            limit.arg(),
            alpha.arg(),
        ],
    )
}

/// From this many routes a fire (a prefill's), a W8A16 select is the tiled
/// GEMM that widens a stage's codes once a 64 routes
/// (`moe_select_fp8_gemm`); fewer, the register-held kernel streams the
/// bank as fast.
const FP8_GEMM_FROM: u32 = 8192;

/// From this many routes a fire (with K in whole 64s), a bf16 select is the
/// tiled GEMM (`moe_select_bf16_gemm`), a CTA 64 of an expert's routes: a
/// batch's routes crowd onto a few hot experts as often as not, and its
/// tiles stream a hot expert's rows once a 64 routes, near the card's rate
/// with the routes spread.
const BF16_GEMM_FROM: u32 = 256;

/// A routed select over a bf16 bank (`experts x n x k` at `bank`) on mma.sync
/// (`linear/moe_w8.cuh`): the register-held `moe_select_bf16_held`, each
/// expert's rows streamed once a slice of its routes, or from
/// `BF16_GEMM_FROM` routes the tiled `moe_select_bf16_gemm`; the GLU in its
/// epilogue where `glu` says. `Ok(false)` for a fire it does not take.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn select_bf16_held(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: u64,
    experts: u32,
    routes: Tensor,
    y: &mut Tensor,
    glu: Option<crate::linear::tc5::Glu>,
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
    let (code, limit, alpha) = glu_code(op, glu)?;
    let bank_rows = if glu.is_some() { 2 * y.width } else { y.width };
    let experts = experts.clamp(1, EXPERT_CAP);
    let k = stated(op, x.width)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", bank_rows)?)?;
    let inter = stated(op, y.width)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let SelectPlan {
        nb,
        tiled,
        tiles,
        ksplit,
        order,
    } = select_plan(
        ctx,
        op,
        routes,
        &fan,
        experts,
        y,
        glu.is_some(),
        fan.route_count >= BF16_GEMM_FROM && x.width.is_multiple_of(64),
    )?;
    let fire = if tiled {
        // Three stages of 64 routes' and 128 rows' 64 of K, four warps a
        // CTA each 32 routes by 64 rows: two CTAs an SM.
        const STAGES: u32 = 3;
        const WN: u32 = 2;
        let cols = if glu.is_some() { 64 } else { 128 };
        Fire::at(
            "linear/moe_w8.cuh",
            symbol(&format!(
                "::pie::linear::moe_select_bf16_gemm<{STAGES}, {code}, {WN}>"
            )),
        )
        .apply(
            Launch::grid([order.work_cap, y.width.div_ceil(cols), 1], [64 * WN, 1, 1])
                .smem(STAGES * (64 + 128) * 128),
        )
    } else {
        let cols = if glu.is_some() { 64 } else { 64 * tiles };
        Fire::at(
            "linear/moe_w8.cuh",
            symbol(&format!(
                "::pie::linear::moe_select_bf16_held<{nb}, {DEPTH}, {code}, {tiles}, {ksplit}>"
            )),
        )
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

/// Whether [`select_mxfp4_atoms`] takes a fire of `routes` over `x`'s
/// width (asked before its activations are read back for it).
#[cfg(feature = "cuda")]
#[must_use]
pub fn select_mxfp4_atoms_takes(ctx: &Ctx, routes: Tensor, x: Tensor) -> bool {
    let t = ctx.tuning();
    let count = routes.rows * routes.width;
    t.mxfp4_cvt
        && crate::linear::tc5::supported()
        && t.mxfp4_a8_from <= count
        && count <= t.mxfp4_a8_until
        && x.width > 0
        && x.width.is_multiple_of(32)
}

/// A routed mxfp4 select with tcgen05-atom scales (a W4A8 bank) on the
/// e2m1-convert register-dequant kernel, against bf16 activations.
/// `Ok(false)` where the kernel does not take the fire.
#[cfg(feature = "cuda")]
pub fn select_mxfp4_atoms(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    atoms: u64,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "linear.moe_select_mxfp4_atoms";
    let fan = selected(OP, x, routes, y)?;
    if !(select_mxfp4_atoms_takes(ctx, routes, x) && y.dtype == Dtype::Bf16) {
        return Ok(false);
    }
    let scales = Tensor::new(atoms, codes.rows, 0, Dtype::U8);
    select_mxfp4_mma(
        ctx, OP, x, codes, scales, None, routes, y, &fan, None, None, true,
    )?;
    Ok(true)
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
    // A launch that fails falls back to the arms below.
    #[cfg(feature = "cuda")]
    if mxfp4_mma_worth(ctx, x, y, seat, &fan)
        && select_mxfp4_mma(
            ctx, op, x, codes, scales, bias, routes, y, &fan, None, None, false,
        )
        .is_ok()
    {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    if !seat.streams()
        && tc5_worth(
            ctx,
            x,
            (Dtype::Bf16, codes.rows),
            y,
            ExpertTable::RESIDENT,
            routes.rows,
            fan.route_count,
        )
    {
        use crate::linear::tc5;
        // A tcgen05 launch that fails falls back to the arm below.
        if select_tc5(
            ctx,
            op,
            x,
            tc5::Operand::scaled(codes.ptr, tc5::Elem::E2m1, scales.ptr, MXFP4_BLOCK),
            codes.rows,
            routes,
            y,
            tc5::Epilogue {
                bias: bias.map_or(0, |b| b.ptr),
                ..tc5::Epilogue::STORE
            },
        )
        .is_ok()
        {
            return Ok(());
        }
    }
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
    let tile = (DECODE_BLOCK / WARP) * ROWS_PER_WARP;
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    let experts = codes.rows.clamp(1, EXPERT_CAP);
    // Grouping pays once an expert serves more than
    // `kernel.moe.mxfp4_grouped_reuse` routes on average.
    if fan.route_count > ctx.tuning().mxfp4_grouped_reuse * experts && ctx.tuning().mxfp4_grouped {
        let RouteOrder {
            order,
            offsets,
            work,
            work_cap,
        } = route_order(ctx, op, routes, &fan, experts)?;
        let wmma = x.dtype == Dtype::Bf16 && ctx.tuning().mxfp4_wmma;
        let entry = if wmma {
            "::pie::linear::moe_matmul_select_mxfp4_wmma".to_string()
        } else {
            format!("::pie::linear::moe_matmul_select_mxfp4_grouped<{t}>")
        };
        let mut args = vec![x.arg(), ArgValue::Ptr(order), ArgValue::Ptr(offsets)];
        if wmma {
            args.push(ArgValue::Ptr(work));
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
            Fire::at("linear/quant.cuh", symbol(&entry)).apply(Launch::grid(
                [
                    if wmma { work_cap } else { experts },
                    y.width.div_ceil(GROUPED_TILE_N),
                    1,
                ],
                [GROUPED_BLOCK, 1, 1],
            )),
            &args,
        );
    }
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
            fan.top_k.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
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
    match biases {
        None => matmul_select_mxfp4(ctx, OP, x, codes, scales, None, routes, y, seat),
        Some(biases) => matmul_select_mlxu4(ctx, OP, x, codes, scales, biases, routes, y, seat),
    }
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
    const ROWS_PER_WARP: u32 = 4;

    const DECODE_BLOCK: u32 = 128;

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
    // A 4-bit bank of bf16 factors on the grouped tcgen05 GEMM.
    #[cfg(feature = "cuda")]
    if let Some(group) = affine_u4_group(x, codes, scales, y.width)
        && !seat.streams()
        && tc5_worth(
            ctx,
            x,
            (Dtype::Bf16, codes.rows),
            y,
            ExpertTable::RESIDENT,
            routes.rows,
            fan.route_count,
        )
    {
        use crate::linear::tc5;
        // A tcgen05 launch that fails falls back to the arm below.
        if select_tc5(
            ctx,
            op,
            x,
            tc5::Operand::affine(codes.ptr, scales.ptr, biases.ptr, group),
            codes.rows,
            routes,
            y,
            tc5::Epilogue::STORE,
        )
        .is_ok()
        {
            return Ok(());
        }
    }
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
    if !matches!(group, 32 | 64 | 128) || !group.is_multiple_of(32 / bits) {
        return Err(refuse(
            op,
            format!(
                "{groups_per_row} factors over a {}-wide row is not a 32-, 64- or 128-code affine group",
                x.width
            ),
        ));
    }
    let tile = (DECODE_BLOCK / WARP) * ROWS_PER_WARP;
    let act_div = if fan.by_token { fan.top_k } else { 1 };

    let experts = codes.rows.clamp(1, EXPERT_CAP);
    let tuning = ctx.tuning();
    if tuning.affine_grouped && fan.route_count >= tuning.affine_grouped_from {
        let RouteOrder {
            order,
            offsets,
            work,
            work_cap,
        } = route_order(ctx, op, routes, &fan, experts)?;
        let wmma = x.dtype == Dtype::Bf16 && tuning.affine_wmma;
        let entry = if wmma {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_wmma<::pie::i32({bits}), ::pie::i32({group})>"
            )
        } else {
            format!(
                "::pie::linear::moe_matmul_select_mlxu4_grouped<{t}, ::pie::i32({bits}), \
                 ::pie::i32({group})>"
            )
        };
        let mut args = vec![x.arg(), ArgValue::Ptr(order), ArgValue::Ptr(offsets)];
        if wmma {
            args.push(ArgValue::Ptr(work));
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
            Fire::at("linear/quant.cuh", symbol(&entry)).apply(Launch::grid(
                [
                    if wmma { work_cap } else { experts },
                    y.width.div_ceil(GROUPED_TILE_N),
                    1,
                ],
                [GROUPED_BLOCK, 1, 1],
            )),
            &args,
        );
    }
    ctx.fire(
        op,
        Fire::at(
            "linear/quant.cuh",
            symbol(&format!(
                "::pie::linear::moe_matmul_select_mlxu4<{t}, ::pie::i32({bits}), ::pie::i32({group}), \
                 ::pie::i32({ROWS_PER_WARP})>"
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
            biases.arg(),
            y.arg(),
            fan.top_k.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            ctx.stage(),
        ],
    )
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
    // Eight columns a thread where the rows and both planes take whole
    // 16-byte runs: the same sums, element for element.
    let vec8 =
        y.width.is_multiple_of(8) && routed.ptr.is_multiple_of(16) && y.ptr.is_multiple_of(16);
    let (entry, threads) = if vec8 {
        ("moe_weighted_sum_vec8", y.width / 8)
    } else {
        ("moe_weighted_sum", y.width)
    };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::{entry}<{t}>")))
            .apply(elementwise_rows(OP, y.rows, threads)?),
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

/// Each token's weighted routed rows summed in f32, plus its `shared` row,
/// rounded once.
pub fn sum_add(ctx: &Ctx, routed: Tensor, shared: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.moe_sum_add";
    let t = dtype_dispatch!(OP, routed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "the token rows this fold lands on", y.rows)?;
    if !routed.rows.is_multiple_of(y.rows)
        || (shared.rows, shared.width) != (y.rows, y.width)
        || routed.width != y.width
    {
        return Err(refuse(
            OP,
            format!(
                "{} x {} routed rows and {} x {} shared ones do not fold into {} x {}",
                routed.rows, routed.width, shared.rows, shared.width, y.rows, y.width
            ),
        ));
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
            stated(OP, nonzero(OP, "the gate row's pitch", gate.width)?)?.arg(),
            ctx.stage(),
        ],
    )
}

/// The most rows [`matmul_grouped_fp8`] takes: the MMA's eight columns.
/// Past them cuBLAS's bf16 leads (mma.sync's fp8 at 16, 32 and 64 rows
/// took 20, 30 and 58 us to its 14 to 15, staged or split as it might be).
pub const GROUPED_FP8_ROWS: u32 = 8;

/// A grouped projection on fp8 planes (DeepSeek-V4's `wo_a`):
/// `y[r, g*n ..] = q(x[r, g*k .. g*k + k]) . w_g^T`, each plane e4m3 under a
/// ue8m0 exponent a 128x128 block, `q` the rows' e4m3 1x128 quantization, in
/// one launch on mma.sync. At most [`GROUPED_FP8_ROWS`] rows; `k` whole 512s,
/// `n` whole 64s.
#[cfg(feature = "cuda")]
pub fn matmul_grouped_fp8(
    ctx: &Ctx,
    x: Tensor,
    codes: u64,
    scales: u64,
    y: &mut Tensor,
    groups: u32,
) -> Result<(), Error> {
    // Four 16-row tiles a CTA, K in quarters a tile, two steps of codes in
    // flight a warp (measured against cuBLAS's bf16 at DeepSeek-V4-Flash).
    matmul_grouped_fp8_with(ctx, x, codes, scales, y, groups, Fp8Shape::GROUPED)
}

/// How [`matmul_grouped_fp8_with`] lays a plane's rows and K over a CTA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8Shape {
    /// 16-row tiles of the plane a CTA.
    pub tiles: u32,
    /// Warps a tile, each a share of K (`k` whole `128 split`s).
    pub split: u32,
    /// 128-wide steps of codes in flight a warp.
    pub depth: u32,
}

impl Fp8Shape {
    /// DeepSeek-V4's grouped output projection's.
    pub const GROUPED: Fp8Shape = Fp8Shape {
        tiles: 4,
        split: 4,
        depth: 2,
    };
}

/// The CTA shape a dense fp8b128 projection (`n x k`) at `rows` decode rows
/// runs [`matmul_grouped_fp8_with`] in where it leads the tcgen05 GEMM and the
/// fp8 GEMV; `None` where one of those does (DeepSeek-V4-Flash, measured).
#[must_use]
pub fn dense_fp8_shape(rows: u32, n: u32, k: u32) -> Option<Fp8Shape> {
    let big = u64::from(n) * u64::from(k) >= 32 << 20;
    let shape = |tiles, split, depth| Fp8Shape {
        tiles,
        split,
        depth,
    };
    let picked = match rows {
        0 | 1 => return None,
        2..=4 if n <= 1024 => return None,
        2..=4 if big && n >= 16384 => shape(2, 4, 2),
        _ if rows > GROUPED_FP8_ROWS => return None,
        5.. if big => return None,
        _ if n >= 8192 => shape(4, 4, 2),
        5.. if n <= 1024 => shape(1, 8, 4),
        _ => shape(2, 8, 2),
    };
    (k.is_multiple_of(128 * picked.split) && n.is_multiple_of(16 * picked.tiles)).then_some(picked)
}

/// [`matmul_grouped_fp8`] in the CTA shape `shape` (`k` whole `128
/// shape.split`s, `n` whole `16 shape.tiles`s).
#[cfg(feature = "cuda")]
pub fn matmul_grouped_fp8_with(
    ctx: &Ctx,
    x: Tensor,
    codes: u64,
    scales: u64,
    y: &mut Tensor,
    groups: u32,
    shape: Fp8Shape,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_grouped";
    let Fp8Shape {
        tiles,
        split,
        depth,
    } = shape;
    if x.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(refuse(OP, "the fp8 grouped leg reads and lands bf16 rows"));
    }
    if groups == 0
        || !x.width.is_multiple_of(groups)
        || !y.width.is_multiple_of(groups)
        || x.rows != y.rows
    {
        return Err(refuse(OP, "the groups do not divide the rows' widths"));
    }
    let (k, n) = (x.width / groups, y.width / groups);
    if !k.is_multiple_of(128 * split) || !n.is_multiple_of(16 * tiles) || y.rows > GROUPED_FP8_ROWS
    {
        return Err(refuse(
            OP,
            format!(
                "the fp8 grouped leg takes K in whole {}s, N in whole {}s and at most \
                 {GROUPED_FP8_ROWS} rows, not {} rows of [{n}, {k}]",
                128 * split,
                16 * tiles,
                y.rows
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    // The rows' codes (padded 16 bytes a row) and exponents, or the
    // tiles' sums of K handed across, whichever is more.
    let smem = (y.rows * (k + 16) + y.rows * (k / 128) * 4).max(tiles * (split - 1) * 4 * 32 * 4);
    ctx.fire(
        OP,
        Fire::at(
            "linear/grouped_fp8.cuh",
            symbol(&format!(
                "::pie::linear::grouped_mma_fp8b128<::pie::i32({tiles}), ::pie::i32({split}), \
                 ::pie::i32({depth})>"
            )),
        )
        .apply(Launch::grid([n / (16 * tiles), groups, 1], [32 * split * tiles, 1, 1]).smem(smem)),
        &[
            ArgValue::Ptr(codes),
            ArgValue::Ptr(scales),
            x.arg(),
            y.arg(),
            stated(OP, y.rows)?.arg(),
            stated(OP, n)?.arg(),
            stated(OP, k)?.arg(),
            stated(OP, groups)?.arg(),
        ],
    )
}

/// A bf16 plane `[rows, k]` as fp8b128: e4m3 codes at `codes`, each 128x128
/// block's ue8m0 exponent spread in tcgen05's atoms at `atoms`. The exponent
/// is the least that holds the block's amax (DeepSeek's rule).
#[cfg(feature = "cuda")]
pub fn quantize_fp8b128(ctx: &Ctx, src: Tensor, codes: u64, atoms: u64) -> Result<(), Error> {
    const OP: &str = "linear.quantize_fp8b128";
    if src.dtype != Dtype::Bf16
        || !src.rows.is_multiple_of(128)
        || !src.width.is_multiple_of(128)
        || src.rows == 0
    {
        return Err(refuse(
            OP,
            format!(
                "quantizes a bf16 plane of whole 128x128 blocks, not {} x {} {:?}",
                src.rows, src.width, src.dtype
            ),
        ));
    }
    ctx.fire(
        OP,
        Fire::at("linear/grouped_fp8.cuh", "::pie::linear::quantize_fp8b128").apply(Launch::grid(
            [src.width / 128, src.rows / 128, 1],
            [256, 1, 1],
        )),
        &[
            src.arg(),
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            stated(OP, src.rows)?.arg(),
            stated(OP, src.width)?.arg(),
        ],
    )
}

/// An fp8b128 plane (`codes` `[rows, k]`, each 128x128 block's exponent
/// spread to its rows in tcgen05's atoms at `atoms`) as bf16 at `out`, each
/// value exactly: the bf16 [`quantize_fp8b128`] read.
#[cfg(feature = "cuda")]
pub fn dequantize_fp8b128(
    ctx: &Ctx,
    codes: u64,
    atoms: u64,
    out: u64,
    rows: u32,
    k: u32,
) -> Result<(), Error> {
    const OP: &str = "linear.dequantize_fp8b128";
    if !k.is_multiple_of(128) {
        return Err(refuse(OP, format!("K {k} is not whole 128-wide blocks")));
    }
    let threads = u32::try_from((u64::from(rows) * u64::from(k)).div_ceil(16))
        .map_err(|_| refuse(OP, "the plane's 16-code runs pass a 32-bit launch"))?;
    ctx.fire(
        OP,
        Fire::at(
            "linear/grouped_fp8.cuh",
            "::pie::linear::dequantize_fp8b128",
        )
        .apply(Launch::flat(threads, 256)),
        &[
            ArgValue::Ptr(codes),
            ArgValue::Ptr(atoms),
            ArgValue::Ptr(out),
            stated(OP, rows)?.arg(),
            stated(OP, k)?.arg(),
        ],
    )
}

/// A grouped projection whose routes send row group `g` to bank `g`: one
/// strided batched bf16 GEMM, `x` `[rows, groups * k]`, `w`
/// `[groups * n, k]`, `y` `[rows, groups * n]`.
#[cfg(feature = "cuda")]
pub fn matmul_grouped_strided(
    ctx: &Ctx,
    x: Tensor,
    w: Tensor,
    y: &mut Tensor,
    groups: u32,
) -> Result<(), Error> {
    use cudarc::cublas::sys::{
        cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmStridedBatchedEx,
        cublasOperation_t, cublasStatus_t, cudaDataType,
    };
    const OP: &str = "linear.matmul_grouped";
    if x.dtype != Dtype::Bf16 || w.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(refuse(OP, "the strided grouped leg reads and lands bf16"));
    }
    if groups == 0
        || !x.width.is_multiple_of(groups)
        || !y.width.is_multiple_of(groups)
        || x.rows != y.rows
    {
        return Err(refuse(OP, "the groups do not divide the rows' widths"));
    }
    let k = x.width / groups;
    let n = y.width / groups;
    if u64::from(w.rows) * u64::from(w.width) != u64::from(groups) * u64::from(n) * u64::from(k) {
        return Err(refuse(
            OP,
            format!(
                "the plane is [{}, {}], not {groups} groups of [{n}, {k}]",
                w.rows, w.width
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    let int =
        |v: u32| i32::try_from(v).map_err(|_| refuse(OP, "an extent past the int cuBLAS takes"));
    let handle = ctx.cublas(OP)?.cast::<cublasContext>();
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // Column-major: Y_g^T = W_g (op T) x X_g^T.
    // SAFETY: `handle` is this context's live cuBLAS handle bound to its stream;
    // the planes hold what the extents say.
    let status = unsafe {
        cublasGemmStridedBatchedEx(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            int(n)?,
            int(y.rows)?,
            int(k)?,
            core::ptr::from_ref(&alpha).cast(),
            w.ptr as usize as *const core::ffi::c_void,
            cudaDataType::CUDA_R_16BF,
            int(k)?,
            i64::from(n) * i64::from(k),
            x.ptr as usize as *const core::ffi::c_void,
            cudaDataType::CUDA_R_16BF,
            int(x.width)?,
            i64::from(k),
            core::ptr::from_ref(&beta).cast(),
            y.ptr as usize as *mut core::ffi::c_void,
            cudaDataType::CUDA_R_16BF,
            int(y.width)?,
            i64::from(n),
            int(groups)?,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    };
    if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(refuse(
            OP,
            format!("`cublasGemmStridedBatchedEx` answered {}", status as i32),
        ))
    }
}
