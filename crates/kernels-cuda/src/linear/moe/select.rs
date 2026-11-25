#![allow(unsafe_code)]
use super::select_fp4::{matmul_select_mlxu4, matmul_select_mxfp4};
use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;
use dtype::Dtype;

pub(super) const GEMV_WARPS: u32 = 4;

/// Weight rows a GEMV warp holds in flight.
pub(super) const GEMV_ROWS: u32 = 4;

pub(super) const MAX_GRID_Y: u32 = 65_535;

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
    const OP: &str = "linear.moe_select";

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
            return select_tc5_with(
                ctx,
                OP,
                x,
                crate::linear::gemm_sm100::Operand::plain(
                    bank.ptr,
                    crate::linear::gemm_sm100::Elem::Bf16,
                ),
                bank.rows,
                routes,
                y,
                crate::linear::gemm_sm100::Epilogue::STORE,
                (experts, Planes::Dense),
            );
        }
        let leg = cheapest(ctx, x, bank, routes, y, experts)?;
        matmul_select_on(ctx, leg, x, bank, routes, y, experts)
    }
    #[cfg(not(feature = "cuda"))]
    select_gemv(ctx, OP, x, bank, routes, y, experts)
}

/// A bf16 or f16 routed select's legs below the grouped tcgen05 GEMM. Each
/// streams an expert's weight rows once a slice of the routes it holds: the
/// GEMV's slice is one route, the tiled GEMMs' their tile's rows.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leg {
    /// The tiled mma.sync GEMM (`moe_select_bf16_gemm`), an expert's routes
    /// in slices of 64 (128 where its experts average the wide tile's), or
    /// on Hopper the warp-specialized wgmma GEMM's 128.
    Tiled,
    /// The grouped mma.sync kernel (`moe_select_bf16_mma`; on Hopper
    /// `moe_select_bf16_wgmma`), slices of 8 to 64 routes by how many the
    /// average expert holds.
    Mma,
    /// One GEMV a route: each route reads its expert's rows.
    Gemv,
}

/// The legs that can take a select of `x` over `bank` into `y`, a tie's
/// winner first: each one's predicate is what its kernel can launch, never
/// what it is worth.
#[cfg(feature = "cuda")]
#[must_use]
pub fn legs(x: Tensor, bank: Tensor, y: &Tensor, experts: ExpertTable) -> Vec<Leg> {
    let bf16 = x.dtype == Dtype::Bf16 && bank.dtype == Dtype::Bf16 && y.dtype == Dtype::Bf16;
    let plane = u64::from(bank.width) == u64::from(x.width) * u64::from(y.width);
    // Where the grouped tcgen05 GEMM reads a resident bank the mma.sync
    // GEMMs do not; a streamed table's mapped host entries go through them.
    let mma_sync = !crate::linear::gemm_sm100::supported() || experts.streams();
    let mut legs = Vec::new();
    if bf16 && mma_sync && x.width.is_multiple_of(64) && bank.ptr.is_multiple_of(16) {
        legs.push(Leg::Tiled);
    }
    if bf16 && x.width.is_multiple_of(64) && y.width.is_multiple_of(8) && bank.rows >= 1 && plane {
        legs.push(Leg::Mma);
    }
    if matches!(x.dtype, Dtype::Bf16 | Dtype::F16)
        && bank.dtype == x.dtype
        && x.width.is_multiple_of(8)
    {
        legs.push(Leg::Gemv);
    }
    legs
}

/// [`matmul_select`] on `leg`, one of [`legs`].
#[cfg(feature = "cuda")]
pub fn matmul_select_on(
    ctx: &Ctx,
    leg: Leg,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";
    let fan = selected(OP, x, routes, y)?;
    match leg {
        Leg::Tiled => {
            if select_bf16_tiled(ctx, OP, x, bank.ptr, bank.rows, routes, y, None, experts)? {
                Ok(())
            } else {
                Err(refuse(OP, "the tiled leg declined a fire `legs` gave it"))
            }
        }
        Leg::Mma if wgmma_takes(y.width, experts) => {
            select_bf16_wgmma(ctx, OP, x, bank, routes, y, &fan, bank.rows, experts)
        }
        Leg::Mma => select_bf16_mma(ctx, OP, x, bank, routes, y, &fan, bank.rows, experts),
        Leg::Gemv => select_gemv(ctx, OP, x, bank, routes, y, experts),
    }
}

/// The leg [`matmul_select`] takes: of [`legs`], the one [`price`] puts
/// lowest on this device's roofline, a tie to the first; the first where the
/// load states no roofline.
#[cfg(feature = "cuda")]
pub fn cheapest(
    ctx: &Ctx,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &Tensor,
    experts: ExpertTable,
) -> Result<Leg, Error> {
    const OP: &str = "linear.moe_select";
    let fan = selected(OP, x, routes, y)?;
    let legs = legs(x, bank, y, experts);
    let none = || {
        refuse(
            OP,
            format!(
                "no leg takes a {:?} select over a {:?} bank at K {}",
                x.dtype, bank.dtype, x.width
            ),
        )
    };
    // No roofline: a load that skips measuring on a card no layer rates.
    let Some(roof) = ctx.tuning().roof.rates() else {
        return legs.first().copied().ok_or_else(none);
    };
    let cores = Cores::of(ctx, experts).ok_or_else(|| {
        refuse(
            OP,
            "the routed select prices its legs by the device's multiprocessors and their \
             shared memory, and the device answers neither",
        )
    })?;
    let shape = Shape {
        routes: fan.route_count,
        experts: bank.rows.max(1),
        n: y.width,
        k: x.width,
    };
    pick(&legs, shape, cores, roof).ok_or_else(none)
}

/// Of `legs`, the one [`price`] puts lowest, a tie to the first; `None` for
/// none.
#[cfg(feature = "cuda")]
#[must_use]
pub fn pick(
    legs: &[Leg],
    shape: Shape,
    cores: Cores,
    roof: &crate::tuning::RoofRates,
) -> Option<Leg> {
    legs.iter()
        .map(|&leg| (leg, price(leg, shape, cores, roof)))
        .reduce(|best, next| if next.1 < best.1 { next } else { best })
        .map(|(leg, _)| leg)
}

/// A routed select as its legs are priced: its routes, the bank's experts,
/// and each expert's `n x k` plane.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub routes: u32,
    pub experts: u32,
    pub n: u32,
    pub k: u32,
}

/// What a leg's tiles fill: the device's multiprocessors and each one's
/// shared memory, the tuning its tiled legs lay their tiles by, and whether
/// they run Hopper's wgmma kernels.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cores {
    pub sms: u32,
    pub smem_per_sm: u32,
    pub hopper: bool,
    /// `kernel.moe.bf16_wide_from`: routes an expert from which the tiled
    /// GEMM lays the wide tile.
    pub wide_from: u32,
    /// `kernel.moe.bf16_gemm_stages`.
    pub stages: u32,
    /// The bank is streamed: Hopper's wgmma kernels stay off its mapped
    /// entries.
    pub streams: bool,
}

#[cfg(feature = "cuda")]
impl Cores {
    /// `ctx`'s device, `None` where it answers neither count.
    #[must_use]
    pub fn of(ctx: &Ctx, experts: ExpertTable) -> Option<Cores> {
        Some(Cores {
            sms: ctx.multiprocessors()?,
            smem_per_sm: ctx.smem_per_sm()?,
            hopper: hopper_wgmma(),
            wide_from: ctx.tuning().bf16_wide_from(),
            stages: ctx.tuning().bf16_gemm_stages(),
            streams: experts.streams(),
        })
    }
}

/// `leg`'s time for a select of `shape` on `cores` under `roof`, in ns: its
/// launches, and the longer of its padded flops at the matmul's rate (its
/// rows as its tiles pad them) and the bytes it streams at the device's
/// bandwidth, stretched by the share of its last wave its tiles leave idle.
/// A leg streams an expert's plane once a slice of the routes the expert
/// holds (the routes spread over the experts they reach), and every route's
/// activation row in and product row out.
#[cfg(feature = "cuda")]
#[must_use]
pub fn price(leg: Leg, shape: Shape, cores: Cores, roof: &crate::tuning::RoofRates) -> f64 {
    let Shape {
        routes,
        experts,
        n,
        k,
    } = shape;
    let (routes_f, n_f, k_f) = (f64::from(routes), f64::from(n), f64::from(k));
    let plane = n_f * k_f * 2.0;
    let rows_moved = routes_f * (n_f + k_f) * 2.0;
    let hit = experts.min(routes).max(1);
    let per = routes.div_ceil(experts.max(1));
    let wave = |tiles: u64, per_sm: u32| -> f64 {
        let slots = u64::from(cores.sms.max(1)) * u64::from(per_sm.max(1));
        let tiles = tiles.max(1);
        tiles as f64 / (tiles.div_ceil(slots) * slots) as f64
    };
    let tile = match leg {
        Leg::Gemv => {
            let tiles = u64::from(n.div_ceil(GEMV_WARPS * GEMV_ROWS)) * u64::from(routes);
            let bytes = routes_f * plane + rows_moved;
            return roof.launch_ns + bytes / roof.bytes_per_ns / wave(tiles, 1);
        }
        Leg::Tiled => tiled_tile(per, k, cores),
        Leg::Mma => mma_tile(per, n, k, cores),
    };
    let slices = u64::from(hit) * u64::from(routes.div_ceil(hit).div_ceil(tile.slice));
    let rows = slices * u64::from(tile.slice);
    let tiles = slices * u64::from(n.div_ceil(tile.cols));
    let fill = wave(tiles, cores.smem_per_sm / tile.smem.max(1));
    let flops = 2.0 * rows as f64 * n_f * k_f;
    let rate = roof.gemm_rate(u32::try_from(rows).unwrap_or(u32::MAX), n, k);
    let bytes = slices as f64 * plane + rows_moved;
    // The route order, then the select.
    2.0 * roof.launch_ns + (flops / rate).max(bytes / roof.bytes_per_ns) / fill
}

/// A tiled leg's tile: the routes of one expert a CTA takes (its slice),
/// the product columns, and the shared memory it holds.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub slice: u32,
    pub cols: u32,
    pub smem: u32,
}

/// Whether this device is Hopper (sm_90), whose arch-specific target has
/// wgmma: the routed W8A8 select's tensor-core GEMM there.
#[cfg(feature = "cuda")]
pub(crate) fn hopper_wgmma() -> bool {
    crate::jit::cache::arch() == Some("sm_90")
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
    crate::linear::gemm_sm100::supported()
        && x.dtype == Dtype::Bf16
        && bank == Dtype::Bf16
        && !experts.streams()
        && x.width.is_multiple_of(64)
        && y.width.is_multiple_of(64)
        && ctx.tuning().moe_tc5()
        && (tokens >= ctx.tuning().moe_tc5_from() || routes >= 2 * count || routes > MAX_GRID_Y)
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
    w: crate::linear::gemm_sm100::Operand,
    experts: u32,
    routes: Tensor,
    y: &mut Tensor,
    epi: crate::linear::gemm_sm100::Epilogue,
) -> Result<(), Error> {
    select_tc5_with(
        ctx,
        op,
        x,
        w,
        experts,
        routes,
        y,
        epi,
        (ExpertTable::RESIDENT, Planes::Dense),
    )
}

/// [`select_tc5`] over the experts `table` places, a bank of `planes`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_tc5_with(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    w: crate::linear::gemm_sm100::Operand,
    experts: u32,
    routes: Tensor,
    y: &mut Tensor,
    epi: crate::linear::gemm_sm100::Epilogue,
    (table, planes): (ExpertTable, Planes),
) -> Result<(), Error> {
    use crate::linear::{act_quant, gemm_sm100};
    const GROUPED_X: &str = "linear.moe_tc5_x";
    const GROUPED_Y: &str = "linear.moe_tc5_y";
    let fan = selected(op, x, routes, y)?;
    let count = fan.route_count;
    if count == 0 {
        return Ok(());
    }
    let (n, k) = (y.width, x.width);
    // The routers this select serves name an expert once a token.
    let layout = act_quant::lay_out(ctx, op, routes, experts, gemm_sm100::GROUP_ALIGN, true)?;
    let div = if fan.by_token { routes.width } else { 1 };
    let xs = ctx.scratch_stream(op, GROUPED_X, layout.rows as usize * k as usize * 2)? as u64;
    act_quant::scatter_rows(ctx, op, x, div, layout.pos, count, xs)?;
    let ys = ctx.scratch_stream(op, GROUPED_Y, layout.rows as usize * n as usize * 2)? as u64;
    gemm_sm100::grouped_gemm_epi(
        ctx,
        gemm_sm100::Operand::plain(xs, gemm_sm100::Elem::Bf16),
        w,
        ys,
        gemm_sm100::Grouped::contiguous(experts, layout.groups, 0).tabled(table, planes),
        [n, k],
        1.0,
        // An expert's padding rows land nowhere.
        crate::linear::gemm_sm100::Epilogue {
            live_rows: layout.live_rows(),
            ..epi
        },
        count,
    )?;
    act_quant::gather_back(ctx, op, ys, layout.pos, routes.width, y)
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
    const ROWS: u32 = GEMV_ROWS;
    let entry = format!("::pie::linear::moe_matmul_select_gemv_{form}<{t}, {ROWS}>");
    ctx.fire(
        op,
        Fire::at(kernel(SELECT, &entry)).apply(
            Launch::grid(
                [y.width.div_ceil(GEMV_WARPS * ROWS), grid_y, grid_z],
                [WARP, GEMV_WARPS, 1],
            )
            .programmatic(ctx.tuning().pdl()),
        ),
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

pub(super) const WMMA_ROUTES: u32 = 32;

/// A select's GLU as its kernels' `kGlu` code and the GLU's two
/// constants (`linear/glu.cuh`); the register-dequant selects land GLUs
/// alone, not relu2.
#[cfg(feature = "cuda")]
pub(super) fn glu_code(
    op: &'static str,
    glu: Option<crate::linear::gemm_sm100::Glu>,
) -> Result<(i32, f32, f32), Error> {
    use crate::linear::gemm_sm100::Glu;
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

/// How a W8A16 or bf16 select's fire is cut: an expert's routes in slices
/// of `8 nb` (a tiled GEMM's 64 where `tiled`), and on the register-held
/// kernel its m16 tiles of weight rows a warp and its K split.
#[cfg(feature = "cuda")]
pub(super) struct SelectPlan {
    pub(super) nb: u32,
    pub(super) tiled: bool,
    pub(super) tiles: u32,
    pub(super) ksplit: u32,
    pub(super) order: RouteOrder,
}

/// The [`SelectPlan`] of a select of `fan` over `experts` landing `y`
/// (`glu` where it lands a GLU), on the tiled GEMM where `gemm` and its
/// experts average more than 12 routes.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn select_plan(
    ctx: &Ctx,
    op: &'static str,
    routes: Tensor,
    fan: &Selected,
    experts: u32,
    y: &Tensor,
    glu: bool,
    gemm: bool,
    fill: u32,
    wide: Option<u32>,
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
        wide.filter(|_| nb == 8).unwrap_or(8 * nb),
        true,
        if glu || KEEP_NEXT.with(core::cell::Cell::get) {
            Share::Keep
        } else {
            Share::Take
        },
    )?;
    // A decode's few CTAs: half the rows a warp (twice the CTAs), then K
    // split across groups of a CTA's warps, toward `fill` CTAs' warps an SM.
    let sms = ctx.multiprocessors().unwrap_or(108);
    let tiles: u32 = if !glu && order.work_cap * y.width.div_ceil(128) < 2 * sms {
        1
    } else {
        2
    };
    let cols = if glu { 64 } else { 64 * tiles };
    let ctas = (order.work_cap * y.width.div_ceil(cols)).max(1);
    let ksplit = if nb == 1 {
        (fill * sms)
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
    experts: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";
    matmul_select_mxfp4(
        ctx,
        OP,
        x,
        codes,
        scales,
        Some(bias),
        routes,
        y,
        seat,
        experts,
    )
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
    experts: ExpertTable,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_select";
    match biases {
        None => matmul_select_mxfp4(ctx, OP, x, codes, scales, None, routes, y, seat, experts),
        Some(biases) => {
            matmul_select_mlxu4(ctx, OP, x, codes, scales, biases, routes, y, seat, experts)
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::tuning::RoofRates;

    /// The RTX Spark's measured rates (its user profile's `[rates]`).
    fn spark() -> RoofRates {
        let flops = |c: &[(u32, f64)]| c.iter().map(|&(m, t)| (m, t * 1e3)).collect();
        #[rustfmt::skip]
        let gemm_by_rows = vec![
            (1024, flops(&[(256, 32.4), (512, 54.3), (1024, 72.3), (2048, 72.7), (4096, 78.7), (8192, 74.3)])),
            (2048, flops(&[(256, 54.3), (512, 80.0), (1024, 85.3), (2048, 84.4), (4096, 81.1), (8192, 83.5)])),
            (4096, flops(&[(256, 27.9), (512, 65.9), (1024, 88.9), (2048, 76.6), (4096, 81.0), (8192, 83.4)])),
        ];
        RoofRates {
            launch_ns: 707.0,
            bytes_per_ns: 264.4,
            gemm_by_rows,
        }
    }

    /// The Spark's 48 multiprocessors of 100 KiB, the registry's tiles.
    const CORES: Cores = Cores {
        sms: 48,
        smem_per_sm: 100 << 10,
        hopper: false,
        wide_from: 128,
        stages: 3,
        streams: false,
    };

    const LEGS: [Leg; 3] = [Leg::Tiled, Leg::Mma, Leg::Gemv];

    /// Qwen3-A3B's gate-up at `per` routes an expert.
    fn gate_up(per: u32) -> Shape {
        Shape {
            routes: 128 * per,
            experts: 128,
            n: 1536,
            k: 2048,
        }
    }

    #[test]
    fn at_one_route_an_expert_every_leg_streams_the_bank_once_and_the_gemv_launches_once() {
        let roof = spark();
        let p = |leg| price(leg, gate_up(1), CORES, &roof);
        assert!(p(Leg::Gemv) < p(Leg::Tiled) && p(Leg::Gemv) < p(Leg::Mma));
        assert_eq!(pick(&LEGS, gate_up(1), CORES, &roof), Some(Leg::Gemv));
    }

    #[test]
    fn past_one_route_an_expert_the_gemv_streams_the_bank_again_a_route() {
        let roof = spark();
        for per in [4, 8, 12, 32] {
            let p = |leg| price(leg, gate_up(per), CORES, &roof);
            let again = p(Leg::Gemv) / p(Leg::Tiled);
            assert!(
                again > f64::from(per) / 2.0,
                "{per} routes an expert: the GEMV priced {again:.1}x the tiled GEMM"
            );
            assert_ne!(pick(&LEGS, gate_up(per), CORES, &roof), Some(Leg::Gemv));
        }
    }

    #[test]
    fn slices_that_each_read_an_expert_once_tie_and_the_tie_is_the_tiled_gemms() {
        let roof = spark();
        for per in [4, 8, 12, 32] {
            let (tiled, mma) = (
                price(Leg::Tiled, gate_up(per), CORES, &roof),
                price(Leg::Mma, gate_up(per), CORES, &roof),
            );
            assert!(
                (tiled - mma).abs() <= 1e-9 * tiled,
                "{per} routes an expert: tiled {tiled} against mma {mma}"
            );
            assert_eq!(pick(&LEGS, gate_up(per), CORES, &roof), Some(Leg::Tiled));
        }
    }

    #[test]
    fn a_wide_fire_takes_the_wide_tile_which_reads_an_expert_half_as_often() {
        let roof = spark();
        let p = |leg| price(leg, gate_up(256), CORES, &roof);
        assert!(
            p(Leg::Tiled) < p(Leg::Mma),
            "{} {}",
            p(Leg::Tiled),
            p(Leg::Mma)
        );
        assert_eq!(pick(&LEGS, gate_up(256), CORES, &roof), Some(Leg::Tiled));
    }

    #[test]
    fn a_fire_short_of_a_wave_is_stretched_by_the_cores_it_leaves_idle() {
        let roof = spark();
        let shape = Shape {
            routes: 32,
            experts: 8,
            n: 128,
            k: 256,
        };
        // Eight experts' one slice by one 128-column tile: 8 tiles on 48
        // multiprocessors.
        let plane = 128.0 * 256.0 * 2.0;
        let bytes = 8.0 * plane + 32.0 * (128.0 + 256.0) * 2.0;
        let full = 2.0 * roof.launch_ns + bytes / roof.bytes_per_ns;
        let tiled = price(Leg::Tiled, shape, CORES, &roof);
        let idle = (tiled - 2.0 * roof.launch_ns) / (full - 2.0 * roof.launch_ns);
        assert!((idle - 6.0).abs() < 1e-9, "stretched {idle}x, not 48 / 8");
    }

    #[test]
    fn no_leg_is_no_pick() {
        assert_eq!(pick(&[], gate_up(1), CORES, &spark()), None);
    }
}
