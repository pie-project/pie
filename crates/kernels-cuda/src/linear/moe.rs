//! `Moe`: routers, routed matmuls, and the folds that bring the fan-out
//! back. One entry per IR variant. The one quantized bank form this plane
//! stamps is mxfp4, and its bank arrives as the explicit `(codes, scales)`
//! pair the engine resolved (the metal precedent) — plain pointers, no
//! by-value descriptor.

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
fn elementwise_rows(op: &'static str, rows: u32, width: u32) -> Result<Launch, Error> {
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
) -> Result<(i32, i32), Error> {
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
            ArgValue::ABSENT, // the activation seat the fused router form fills
            ArgValue::ABSENT, // the bias seat, likewise
            routes.arg(),
            weights.arg(),
            e.arg(),
            k.arg(),
            0_i32.arg(), // `hidden`, read only by the fused form
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// The same router, times the learned per-expert gain.
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
            ArgValue::ABSENT, // the activation seat the fused router form fills
            scale.arg(),
            routes.arg(),
            weights.arg(),
            e.arg(),
            k.arg(),
            0_i32.arg(), // `hidden`, read only by the fused form
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
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
) -> Result<(), Error> {
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
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
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

/// Grouped matmul over a dense bank: each routed row multiplies the expert
/// its route selects — the decode GEMV, one warp-column per output tile.
///
/// A streamed bank hands the select two device addresses; a fully-resident
/// one hands two zeros:
/// * `table` — `expert_id -> base address`, one fixed-address entry per
///   expert. Points into the device slab when resident, or pinned host
///   bytes over UVA otherwise, so a miss costs PCIe bandwidth, never a sync.
/// * `hits` — per-expert usage counters; the select does one `atomicAdd`
///   per routed expert per fire, and the host reads them between fires to
///   promote.
///
/// [`ExpertTable::RESIDENT`] (both zeros) is the degenerate case, not an
/// off switch: with no table the kernel computes the same
/// `bank_base + expert * stride` it always did, and counts nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertTable {
    /// Device address of the `expert_id -> base address` table, or 0.
    pub table: u64,
    /// Device address of the per-expert usage counters, or 0.
    pub hits: u64,
}

impl ExpertTable {
    /// The whole bank is on the device: no table to read, nothing to count.
    pub const RESIDENT: ExpertTable = ExpertTable { table: 0, hits: 0 };

    /// Does this bank stream?
    #[must_use]
    pub const fn streams(&self) -> bool {
        self.table != 0
    }
}

/// [`ExpertTable`]'s twin for the split-plane path, one granularity up: the
/// mxfp4 select computes each plane's expert base itself and dereferences
/// no per-expert table, fixing the unit of residency at the group. A
/// streamed group hands two addresses:
/// * `cell` — one 16-byte, 16-byte-aligned cell holding this group's
///   `(codes, scales)` base pair, read with a single `ld.global.v2.u64`
///   (one extra load per group per launch, an L1 broadcast). Writing it is
///   how a promotion moves a group.
/// * `hits` — this group's usage counter; one `atomicAdd` per routed row
///   per fire, by the block that owns that route's first row tile.
///
/// [`GroupSeat::RESIDENT`] (both zeros) is the degenerate case: the kernel
/// reads the bases it was handed and counts nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupSeat {
    /// Device address of this group's 16-byte `(codes, scales)` base cell,
    /// or 0.
    pub cell: u64,
    /// Device address of this group's `u32` usage counter, or 0.
    pub hits: u64,
}

impl GroupSeat {
    /// The whole group is where the launch says it is: no cell, no counter.
    pub const RESIDENT: GroupSeat = GroupSeat { cell: 0, hits: 0 };

    /// Does this group's tier move?
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
    select_gemv(ctx, "linear.moe_matmul_select", x, bank, routes, y, experts)
}

/// The routed dense GEMV itself, under the caller's own op name. LoRA's
/// projection half is a routed matmul-select at fan-out one and nothing
/// else, so `linear::lora` fires this directly and passes its own `op` so
/// a refusal is attributed to the correction rather than to an MoE the
/// plan does not contain. Its staged-geometry seat rides the same path but
/// is always `ABSENT`, since a grouped region is refused admission to a
/// body; only `linear.moe_matmul_select` ever arms it.
pub(crate) fn select_gemv(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    experts: ExpertTable,
) -> Result<(), Error> {
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
            // The two seats, both `ABSENT` for a resident bank — see [`ExpertTable`].
            ArgValue::Ptr(experts.table),
            ArgValue::Ptr(experts.hits),
            // The staged-geometry seat, read in route space off words
            // written in token space (the grid's route axis is `top_k` of
            // the region's rows). `ABSENT` when no body armed one.
            ctx.stage(),
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
            fan.top_k.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
            // The two seats, both zero for a group the store holds — see [`GroupSeat`].
            ArgValue::Ptr(seat.cell),
            ArgValue::Ptr(seat.hits),
            // The staged-geometry seat, read in route space off words
            // written in token space. `ABSENT` when no body armed one.
            ctx.stage(),
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
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_bias";
    matmul_select_mxfp4(ctx, OP, x, codes, scales, Some(bias), routes, y, seat)
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
    biases: Option<Tensor>,
    routes: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    // The companion shape IS the scheme: mxfp4 centres its own blocks and
    // ships one plane beside the codes, an affine bank ships two. The plane
    // resolution recorded which this bank is when the loader recorded the
    // pairing, so the presence of the zero points is the discriminant and a
    // second statement of the scheme here would be a chance to disagree.
    match biases {
        None => matmul_select_mxfp4(ctx, OP, x, codes, scales, None, routes, y, seat),
        Some(biases) => matmul_select_mlxu4(ctx, OP, x, codes, scales, biases, routes, y, seat),
    }
}

/// The MLX affine-U4 select: 4-bit codes, eight to a `u32` word, sixty-four
/// under one bf16 scale and one bf16 zero point. The dot folds the zero
/// point through the group's activation sum — `Σ (c·s + b)·x` is
/// `s·Σ c·x + b·Σ x` — so the kernel reads each activation once.
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
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed bank's planes bind as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed bank's planes bind as bytes");
    debug_assert_eq!(biases.dtype, Dtype::U8, "a packed bank's planes bind as bytes");
    let fan = selected(op, x, routes, y)?;
    let k = stated(op, nonzero(op, "K, the bank's contracted width", x.width)?)?;
    let n = stated(op, nonzero(op, "N, the bank's output width", y.width)?)?;
    // One expert's planes per row: the code row holds `n * k` codes and the
    // scale row one bf16 per group of them, so the bit width and the group
    // come off the row widths.
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
            // The twin's staged-geometry seat, on the twin's terms.
            ctx.stage(),
        ],
    )
}

/// Folds the `top_k` routed rows back to one row per token, weighted.
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
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// The routed bias mixture, stated once on an activation that already holds
/// the fold: `y[t] = x[t] + sum_k weights[t, k] * bias[routes[t, k]]`. Its
/// own fire because the down-projection is rows-cut under tp — a
/// replicated bias folded into that partial matmul would land once per
/// rank, so this is stated after the all_reduce instead.
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
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
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
            // The gate's row pitch: handles are dense, so the scalar sits at
            // the head of a `gate.width`-wide row, whose pitch the handle
            // states.
            stated(OP, nonzero(OP, "the gate row's pitch", gate.width)?)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
