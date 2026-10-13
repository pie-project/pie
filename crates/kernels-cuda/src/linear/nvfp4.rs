//! NVFP4 projections read at bf16 (W4A16): `codes` are e2m1 nibble pairs,
//! `scales` one e4m3 per 16 codes, `tensor_scales` one f32 a row, each plane
//! bound as bytes, one row of the plane an expert (or the whole projection).
//! A weight is `e2m1(code) · e4m3(scale) · tensor_scale`; the activations
//! stay bf16, so a checkpoint's activation scales are not read here.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::linear::gemm;
use crate::linear::moe::{
    EXPERT_CAP, GROUPED_BLOCK, GROUPED_TILE_N, GroupSeat, RouteOrder, Selected, route_order,
    selected,
};
use crate::tensor::Tensor;

const FILE: &str = "linear/nvfp4.cuh";

const WARP: u32 = 32;

const BLOCK: u32 = 256;

const ROWS_PER_WARP: u32 = 4;

const DECODE_BLOCK: u32 = 128;

/// Codes a 16-byte load holds: the by-route arm's step along K.
const CHUNK: u32 = 32;

/// The grouped arm's K tile.
const TILE_K: u32 = 128;

/// The decoded bf16 tile a dense product reads, as the affine arm keeps it.
const DECODED_WEIGHT: &str = "linear.nvfp4.decoded_weight";

struct Planes {
    n: u32,
    k: u32,
}

/// That the three planes hold `slabs` slabs of `n` rows of `k` codes.
fn planes(
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    n: u32,
    k: u32,
) -> Result<Planes, Error> {
    debug_assert_eq!(codes.dtype, Dtype::Nvfp4, "nvfp4 codes keep their dtype");
    if k == 0 || !k.is_multiple_of(CHUNK) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {CHUNK}-code nvfp4 chunks"),
        ));
    }
    let rows = u64::from(n);
    let wants = [
        ("code", codes, rows * u64::from(k / 2)),
        ("scale", scales, rows * u64::from(k / 16)),
        ("tensor scale", tensor_scales, rows * 4),
    ];
    for (what, plane, bytes) in wants {
        if u64::from(plane.width) != bytes {
            return Err(refuse(
                op,
                format!(
                    "a slab's {what} plane is {} bytes, and {n} rows of {k} codes are {bytes}",
                    plane.width
                ),
            ));
        }
    }
    if scales.rows != codes.rows || tensor_scales.rows != codes.rows {
        return Err(refuse(
            op,
            format!(
                "the planes hold {}, {} and {} slabs; an nvfp4 bank's planes are one \
                 slab a row each",
                codes.rows, scales.rows, tensor_scales.rows
            ),
        ));
    }
    Ok(Planes { n, k })
}

/// The routed product: `y[route] = x[route or its token] · W[routes[route]]ᵀ`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_select(
    ctx: &Ctx,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = selected(OP, x, routes, y)?;
    let Planes { n, k } = planes(OP, codes, scales, tensor_scales, y.width, x.width)?;
    nonzero(OP, "N, the bank's output width", n)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    let experts = codes.rows.clamp(1, EXPERT_CAP);
    // Grouping pays once an expert serves more routes than the mxfp4 bank's
    // knob says, the same trade at the same width of code.
    if x.dtype == Dtype::Bf16
        && k.is_multiple_of(TILE_K)
        && ctx.tuning().mxfp4_grouped
        && fan.route_count > ctx.tuning().mxfp4_grouped_reuse * experts
    {
        return grouped(
            ctx,
            OP,
            x,
            codes,
            scales,
            tensor_scales,
            routes,
            y,
            seat,
            &fan,
            experts,
        );
    }
    by_route(
        ctx,
        OP,
        t,
        x,
        Some(routes),
        codes,
        scales,
        tensor_scales,
        y,
        seat,
        fan.route_count,
        fan.top_k,
        act_div,
    )
}

#[allow(clippy::too_many_arguments)]
fn grouped(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    fan: &Selected,
    experts: u32,
) -> Result<(), Error> {
    let RouteOrder {
        order,
        offsets,
        work,
        work_cap,
    } = route_order(ctx, op, routes, fan, experts)?;
    let act_div = if fan.by_token { fan.top_k } else { 1 };
    ctx.fire(
        op,
        Fire::at(FILE, symbol("::pie::linear::moe_matmul_select_nvfp4_mma")).apply(Launch::grid(
            [work_cap, y.width.div_ceil(GROUPED_TILE_N), 1],
            [GROUPED_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            ArgValue::Ptr(order),
            ArgValue::Ptr(offsets),
            ArgValue::Ptr(work),
            codes.arg(),
            scales.arg(),
            tensor_scales.arg(),
            y.arg(),
            act_div.arg(),
            stated(op, y.width)?.arg(),
            stated(op, x.width)?.arg(),
            stated(op, experts)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn by_route(
    ctx: &Ctx,
    op: &'static str,
    t: &str,
    x: Tensor,
    routes: Option<Tensor>,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    route_count: u32,
    top_k: i32,
    act_div: i32,
) -> Result<(), Error> {
    let tile = (DECODE_BLOCK / WARP) * ROWS_PER_WARP;
    if route_count == 0 {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::moe_matmul_select_nvfp4<{t}, ::pie::i32({ROWS_PER_WARP})>"
            )),
        )
        .apply(Launch::grid(
            [route_count, y.width.div_ceil(tile), 1],
            [DECODE_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            routes.map_or(ArgValue::ABSENT, |routes| routes.arg()),
            codes.arg(),
            scales.arg(),
            tensor_scales.arg(),
            y.arg(),
            top_k.arg(),
            act_div.arg(),
            stated(op, y.width)?.arg(),
            stated(op, x.width)?.arg(),
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            if routes.is_some() {
                ctx.stage()
            } else {
                ArgValue::ABSENT
            },
        ],
    )
}

/// The planes of one `[n, k]` projection decoded into a bf16 tile.
#[allow(clippy::too_many_arguments)]
pub fn decoded_plane(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    n: u32,
    k: u32,
    seat: GroupSeat,
) -> Result<Tensor, Error> {
    if seat.streams() {
        return Err(refuse(
            op,
            "a decoded plane serves resident projections only, and these planes are \
             seated by a streaming tier",
        ));
    }
    let (codes, scales, tensor_scales) = whole(codes, scales, tensor_scales, n)?;
    planes(op, codes, scales, tensor_scales, n, k)?;
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch_stream(op, DECODED_WEIGHT, bytes)? as usize as u64;
    let pairs = u32::try_from(u64::from(n) * u64::from(k / 2)).map_err(|_| {
        refuse(
            op,
            format!("{n} rows of {k} codes do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        op,
        Fire::at(FILE, symbol("::pie::linear::dequant_nvfp4<::pie::bf16>"))
            .apply(Launch::flat(pairs, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            tensor_scales.arg(),
            ArgValue::Ptr(tile),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    Ok(Tensor::new(tile, n, k, Dtype::Bf16))
}

/// A dense projection's planes are bound one row a weight row; read whole,
/// they are one slab.
fn whole(
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    n: u32,
) -> Result<(Tensor, Tensor, Tensor), Error> {
    let fold = |plane: Tensor| -> Tensor {
        if plane.rows == n && n > 1 {
            Tensor::new(plane.ptr, 1, plane.rows * plane.width, plane.dtype)
        } else {
            plane
        }
    };
    Ok((fold(codes), fold(scales), fold(tensor_scales)))
}

/// The dense product `y = act · Wᵀ`: through a decoded bf16 tile and the
/// dense gemm for bf16 activations, the by-route arm over one expert
/// otherwise.
#[allow(clippy::too_many_arguments)]
pub fn matmul(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scales: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (n, k) = (y.width, act.width);
    if act.rows == 0 {
        return Ok(());
    }
    if act.dtype == Dtype::Bf16 && act.rows > 1 && !seat.streams() {
        let tile = decoded_plane(ctx, op, codes, scales, tensor_scales, n, k, seat)?;
        return gemm::act_x_wt(ctx, op, act, tile, None, y);
    }
    let (codes, scales, tensor_scales) = whole(codes, scales, tensor_scales, n)?;
    planes(op, codes, scales, tensor_scales, n, k)?;
    by_route(
        ctx,
        op,
        t,
        act,
        None,
        codes,
        scales,
        tensor_scales,
        y,
        seat,
        act.rows,
        1,
        1,
    )
}
