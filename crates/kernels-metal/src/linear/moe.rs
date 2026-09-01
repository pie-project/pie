//! `Moe`: routers, routed matmuls, and the folds that bring the fan-out
//! back. One entry per IR variant.
//!
//! Both quantized bank forms reach the routed matmul, and the entry picks
//! between them off the [`Bank`] the driver resolved — an mxfp4 bank has no
//! zero points and an affine one does, which is the same discriminator the
//! loader seated the row by. Both live in `quant_qmv.metal`, spelled in
//! source at their own group and width, so neither needs the jit stamp the
//! tiled affine points in [`quant`](crate::linear::quant) carry.

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::{Bank, Tensor};

const ROUTER_MAX_EXPERTS: u32 = 1024;

const ROUTER_MAX_TOP_K: u32 = 16;

const SELECT_GROUP: u32 = 128;

const QMV_GROUP: [u32; 3] = [32, 2, 1];

const MXFP4_BLOCK: u32 = 32;

/// The softmax router's normalization mode: over all experts, not the
/// selected k.
const SOFTMAX_OVER_SELECTED: u32 = 0;

/// The file the sort, the gather and the scatter live in.
const ROUTE_FILE: &str = "linear/moe_route.metal";

/// The file the routed tiled points live in.
const QMM_FILE: &str = "linear/quant_qmm_t.metal";

/// Threadgroup of the tiled point: `WM * WN * SIMD_SIZE` lanes.
const QMM_GROUP: [u32; 3] = [32, 2, 2];

/// The contraction step the tiled points walk.
const QMM_BK: u32 = 32;

/// The three row tiles a routed GEMM is compiled for, narrow first.
const MOE_TILE_ROWS: [u32; 3] = [16, 32, 64];

/// The column tiles it is compiled for, narrow first.
const MOE_TILE_COLS: [u32; 3] = [16, 32, 64];

/// When sorting the rows by expert pays for itself.
///
/// The sort turns `n_pairs` matvecs into `ceil(count_e / tile)` summed over
/// the experts — fewer reads of each expert's weights, but a tile that is
/// only part full does the arithmetic of a whole one. The obvious model says
/// the two meet where an expert's run half fills a tile.
///
/// **THAT MODEL IS WRONG IN THE DIRECTION THAT MATTERS**, and the threshold
/// carries the measurement instead. A 4-bit mixture is bandwidth-bound: what
/// batching buys is reading each expert's slice ONCE instead of once per
/// pair, which is worth far more than the arithmetic a half-empty tile throws
/// away. On the M1 Max the threshold measured its way from 8 down to 1, and
/// the difference on a serving fleet is a step function rather than a margin
/// — gpt-oss-20b at 16 lanes runs 134.1 tok/s at 4 and 310.7 at 1.
///
/// Written against the narrow tile, because that is the cheapest way in: a
/// batch that cannot pay for a 16-row tile cannot pay for a wider one either,
/// and [`tile_rows`] widens only after this has said yes.
#[must_use]
pub fn should_batch(pairs: u32, experts: u32, min_per_expert: u32) -> bool {
    experts > 0 && u64::from(pairs) >= u64::from(experts) * u64::from(min_per_expert)
}

/// Rows each expert's run is padded to, for a batch of `pairs` — 1 when the
/// mixture does not batch at all.
///
/// Priced off ROWS PER EXPERT, because that is what decides how much of a
/// tile a run fills, and measured END TO END rather than modelled: a roofline
/// probe reads ONE expert with a hot cache and a mixture's threadgroups read
/// thirty-two, which is why the probe preferred 64 at 448 rows where the
/// machine wants 32, and preferred a 128-row tile that measures 558.5 → 545.5
/// tok/s slower in a real mixture. The thresholds are a table of measurements
/// and not a curve; they live in `DeviceTuning`, and the reason they must be
/// re-swept whenever the routed GEMM changes lives here.
///
/// What a wider tile does NOT cost is the allocation's worst case:
/// [`sorted_rows`] is deliberately pessimistic and the tiles past the routing
/// decline at `tile_expert < 0`, so a wider tile dispatches more threadgroups
/// that do NOTHING rather than more arithmetic.
#[must_use]
pub fn tile_rows(pairs: u32, experts: u32, tuning: &crate::DeviceTuning) -> u32 {
    if !should_batch(pairs, experts, tuning.moe_batch_min_per_expert) {
        return 1;
    }
    let per = pairs / experts;
    if per >= tuning.moe_tile_wide_per {
        return MOE_TILE_ROWS[2];
    }
    if per >= tuning.moe_tile_mid_per {
        MOE_TILE_ROWS[1]
    } else {
        MOE_TILE_ROWS[0]
    }
}

/// How many sorted rows a batch of `pairs` can produce.
///
/// The WORST case and not the actual: the real count depends on how the
/// router spread the rows, which is a number the GPU has and the host would
/// have to stall to read. Every touched expert can waste `tile - 1` rows and
/// at most `min(pairs, experts)` experts are touched, so this bound is
/// reached and cannot be tightened without the routing itself.
#[must_use]
pub fn sorted_rows(pairs: u32, experts: u32, tuning: &crate::DeviceTuning) -> u32 {
    let tile = tile_rows(pairs, experts, tuning);
    if tile <= 1 {
        return pairs;
    }
    let touched = pairs.min(experts);
    let bound = pairs.saturating_add(touched.saturating_mul(tile - 1));
    bound.div_ceil(tile) * tile
}

/// The widest column tile that divides the output, or `None` when no stamped
/// one does. Wider is strictly fewer dequantizations of each weight tile.
#[must_use]
pub fn tile_cols(out_width: u32) -> Option<u32> {
    MOE_TILE_COLS
        .iter()
        .rev()
        .copied()
        .find(|tile| out_width % tile == 0)
}

fn router_lanes(op: &'static str, experts: u32) -> Result<u32, Error> {
    nonzero(op, "the expert count this router states", experts)?;
    Ok(experts.min(1024).div_ceil(32) * 32)
}

/// One thread per element, one threadgroup row per token row.
fn route_rows(op: &'static str, width: u32, rows: u32) -> Result<Grid, Error> {
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
) -> Result<Grid, Error> {
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
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_softmax";
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

/// The same router, times the learned per-expert gain.
///
/// **NO NEW KERNEL.** `moe_route.metal` instantiates `router_topk` at
/// `SCALED = true` already — the seat at buffer 3 that [`topk_softmax`] binds
/// absent is the gain plane, and the only difference between the two points is
/// which instantiation the entry names.
pub fn topk_softmax_scaled(
    ctx: &Ctx<'_>,
    logits: Tensor,
    scale: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_softmax_scaled";
    let entry = dtype_dispatch!(OP, logits.dtype, { Bf16 => "router_topk_scaled_f32w_bfloat16" });
    debug_assert_eq!(
        scale.dtype, logits.dtype,
        "the gain is read at the router's own width"
    );
    debug_assert_eq!(
        scale.rows * scale.width,
        experts,
        "the gain is indexed by expert, so it holds one entry per expert"
    );
    let grid = ranked(OP, logits, experts, top_k, routes, weights)?;
    ctx.fire(
        Fire::at("linear/moe_route.metal", entry).apply(grid),
        &[
            logits.arg(),
            routes.arg_mut(),
            weights.arg_mut(),
            scale.arg(),
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
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_sigmoid";
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
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_sqrt_softplus";
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

/// The hash router: layers `0..num_hash_layers` route by a per-token LOOKUP,
/// not a learned gate over the router's logits.
///
/// `tid2eid` is `[vocab, top_k]` I64 — for every token id it names `top_k`
/// expert ids, at uniform weight `1/top_k`. This gathers the row the token id
/// selects and lays that pair down in the SAME layout [`topk_softmax`] writes
/// (`routes` I32, `weights` F32, both `[tokens, top_k]` row-major), so its
/// output is drop-in for the same sorted-MoE path — [`matmul_select`],
/// [`weighted_sum`] — with no router logits computed at all.
///
/// **THE TABLE IS I64 AND THE ROUTES ARE I32.** `tid2eid` is a lookup and not
/// a weight-representation dtype the trace can intern; the narrowing is the
/// gather's, in the one place the 64-bit table meets the 32-bit route plane
/// every downstream kernel already reads an expert id as. An expert count
/// never approaches `2^31`.
///
/// **THE TOKEN IDS ARE A 32-BIT COLUMN AND THE SHADER READS `uint`.** The
/// fire's own id stream is i32 (`RuntimeInput::Tokens`) and carries the same
/// bits, so either spelling seats; an out-of-range id falls to row 0 exactly
/// as `embed.metal`'s gather does, so a boundary token reads the first table
/// row rather than off the end.
pub fn hash_route(
    ctx: &Ctx<'_>,
    ids: Tensor,
    tid2eid: Tensor,
    vocab: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_hash_route";
    debug_assert!(
        matches!(ids.dtype, Dtype::U32 | Dtype::I32),
        "`{OP}` gathers by a 32-bit token id column"
    );
    debug_assert_eq!(
        tid2eid.dtype,
        Dtype::I64,
        "`{OP}` reads the i64 hash table"
    );
    debug_assert_eq!(
        tid2eid.width, top_k,
        "the hash table names `top_k` experts per token id"
    );
    // The route/weight planes carry the same shape and element a softmax
    // router lands, so the sorted-MoE path behind them cannot tell the two
    // apart.
    ranked_planes(OP, routes, top_k, routes, weights);
    debug_assert_eq!(
        ids.rows, routes.rows,
        "the token ids handed over are the rows this route lands"
    );
    let top_k = nonzero(OP, "the fan-out this router states", top_k)?;
    nonzero(OP, "the vocabulary this table spans", vocab)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "hash_route_gather").apply(route_rows(OP, top_k, routes.rows)?),
        &[
            ids.arg(),
            tid2eid.arg(),
            routes.arg_mut(),
            weights.arg_mut(),
            vocab.arg(),
            top_k.arg(),
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
) -> Result<Selected, Error> {
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

fn select_gemv_grid(op: &'static str, out_width: u32, routed_rows: u32) -> Result<Grid, Error> {
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
) -> Result<[u32; 3], Error> {
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
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select";
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

/// The routed qmv point one bank arrives at.
///
/// **THE DISCRIMINATOR IS THE THIRD PLANE, NOT THE GROUP SIZE.** Both codecs
/// are four bits and both are read by the same `qmv_routed` template; what
/// separates them is that mxfp4's e8m0 byte IS the whole dequantization while
/// affine's bf16 factor is half of it, and the other half is the bank's zero
/// points. So a bank that carries biases takes the affine instantiation and
/// one that does not takes the mxfp4 one, and the group size is then CHECKED
/// against the point rather than used to pick it — which is what the loader
/// asks for by carrying `(group, bits)` on the row instead of assuming a
/// checkpoint is uniform in them.
fn routed_point(op: &'static str, bank: Bank, biased: bool) -> Result<&'static str, Error> {
    match (bank.affine(), bank.group, bank.bits) {
        (true, 64, 4) if biased => Ok("affine_qmv_routed_bias_bfloat16_gs_64_b_4"),
        (true, 64, 4) => Ok("affine_qmv_routed_bfloat16_gs_64_b_4"),
        // The 2-bit routed decode arm: `quant_qmv.metal` instantiates its
        // group-parametric `AffineU2` codec at all three groups the 2-bit
        // artifacts carry, because a 2-bit checkpoint keeps its expert banks in
        // this same routed path — Qwen3.8-Flash at group 128, DeepSeek-V4-Flash
        // at group 32 (with one layer's gate at 64). The routed impl reads the
        // group off the codec for every scale/bias index, so each group is its
        // own point rather than a 64-code assumption riding a mislabeled name.
        (true, 64, 2) if biased => Ok("affine_qmv_routed_bias_bfloat16_gs_64_b_2"),
        (true, 64, 2) => Ok("affine_qmv_routed_bfloat16_gs_64_b_2"),
        (true, 32, 2) if biased => Ok("affine_qmv_routed_bias_bfloat16_gs_32_b_2"),
        (true, 32, 2) => Ok("affine_qmv_routed_bfloat16_gs_32_b_2"),
        (true, 128, 2) if biased => Ok("affine_qmv_routed_bias_bfloat16_gs_128_b_2"),
        (true, 128, 2) => Ok("affine_qmv_routed_bfloat16_gs_128_b_2"),
        (false, MXFP4_BLOCK, 4) if biased => Ok("mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4"),
        (false, MXFP4_BLOCK, 4) => Ok("mxfp4_qmv_routed_bfloat16_gs_32_b_4"),
        (affine, group, bits) => Err(refuse(
            op,
            format!(
                "the bank is {} at {bits} bits in groups of {group}, and `quant_qmv.metal` \
                 instantiates the routed shapes at affine/{{32,64,128}}/2, affine/64/4 \
                 and mxfp4/32/4 only",
                if affine { "affine" } else { "symmetric" }
            ),
        )),
    }
}

/// What both routed arms share: the planes agree with the codec the point
/// was picked for, and K is a whole number of groups.
fn routed_bank(op: &'static str, x: Tensor, bank: Bank) -> Result<(), Error> {
    if bank.affine() {
        debug_assert_eq!(
            bank.scales.dtype, x.dtype,
            "an affine bank's factors ride the activation's dtype"
        );
    } else {
        debug_assert!(
            matches!(bank.scales.dtype, Dtype::E8m0 | Dtype::U8),
            "an mxfp4 bank's scales are e8m0 exponent bytes"
        );
    }
    if x.width % bank.group != 0 {
        return Err(refuse(
            op,
            format!(
                "K is {}, not a whole number of {}-code groups",
                x.width, bank.group
            ),
        ));
    }
    Ok(())
}

/// Grouped matmul over a quantized bank, with a per-expert bias: each routed
/// row multiplies the expert its route selects. The driver resolves the bank
/// weight to its planes before calling.
pub fn matmul_select_bias(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank: Bank,
    bias: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_bias";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    let entry = routed_point(OP, bank, true)?;
    debug_assert_eq!(bias.dtype, x.dtype, "the expert bias rides the activation's dtype");
    let fan = selected(OP, x, routes, y)?;
    routed_bank(OP, x, bank)?;
    ctx.fire(
        Fire::at("linear/quant_qmv.metal", entry).apply(Grid::of(
            routed_qmv_grid(OP, fan.tokens, y.width, fan.top_k)?,
            QMV_GROUP,
        )),
        &[
            bank.codes.arg(),
            bank.scales.arg(),
            zero_points(ctx, bank)?,
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

/// Grouped matmul over a quantized bank with nothing added — the bias-free
/// twin of [`matmul_select_bias`], for the rows-cut expert down-projection,
/// whose routed bias lands after the reduce through [`bias_sum`] instead.
pub fn matmul_select_quant(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank: Bank,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    let entry = routed_point(OP, bank, false)?;
    let fan = selected(OP, x, routes, y)?;
    routed_bank(OP, x, bank)?;
    ctx.fire(
        Fire::at("linear/quant_qmv.metal", entry).apply(Grid::of(
            routed_qmv_grid(OP, fan.tokens, y.width, fan.top_k)?,
            QMV_GROUP,
        )),
        &[
            bank.codes.arg(),
            bank.scales.arg(),
            zero_points(ctx, bank)?,
            x.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            stated(OP, y.width)?.arg(),
            ctx.absent()?, // the expert bias seat; this entry is the one that adds none
            routes.arg(),
            stated(OP, fan.x_slot_stride)?.arg(),
            stated(OP, fan.x_row_stride)?.arg(),
            stated(OP, fan.top_k)?.arg(),
        ],
    )
}

/// The device planes the sorted arm works in, beside the operands the op
/// named.
///
/// **NOTHING HERE IS AN OUTPUT.** Every field is a working rectangle whose
/// contents are dead the moment [`matmul_select_batched`] returns, and every
/// one is sized off `n_pairs`, `n_experts` and [`sorted_rows`] — quantities
/// the host knows before the fire and the router decides during it. They are
/// a parameter rather than something this plane carves because `kernels-metal`
/// allocates nothing; the shell that owns the arena hands them in.
#[derive(Clone, Copy, Debug)]
pub struct RoutedScratch {
    /// `i32`, [`sorted_rows`] long: the pair index each sorted row came from,
    /// or −1 for a row the padding invented.
    pub perm: Tensor,

    /// `i32`, [`sorted_rows`] long: which expert each sorted row belongs to.
    pub row_expert: Tensor,

    /// `i32`, one per TILE of the sorted stack: the expert that tile serves,
    /// or −1 for a tile past the routing. **This is what makes the padding
    /// free** — a declined tile returns before it reads a weight.
    pub tile_expert: Tensor,

    /// `i32`, `n_pairs` long: where each pair landed. The inverse permutation
    /// comes free from the sort, which is why the scatter costs no second
    /// pass to build one.
    pub inv: Tensor,

    /// The activation gathered into expert-major order: `sorted_rows x K`.
    pub x: Tensor,

    /// The routed product in that same order: `sorted_rows x N`.
    pub y: Tensor,
}

/// The routed tiled point one bank and one tile arrive at, or `None` when the
/// shader stamps none for that combination.
///
/// Four families are stamped and the fifth is absent on purpose:
/// `affine_qmm_t_routed` carries no bias seat, so an affine bank WITH a
/// per-expert bias still takes the matvec arm, and answering it with the
/// point beside it would read an unbound buffer.
///
/// **THE MXFP4 PAIR USED TO BE HALF A FAMILY AND THAT WAS A DEFECT.** Only
/// `_bias` was stamped, on the reading that a mixture which biases its
/// experts biases all of them; gpt-oss biases its gate/up projection and not
/// its down one. So this arm answered `None` once a layer, the driver fell
/// through to the matvec, and the down projection read each expert's slice
/// once per routed row — 9.0 GB of weight traffic a layer against the tiled
/// arm's 0.53. Widening this arm measures 1547.0 -> 784.9 ms at 512 prompt
/// tokens on an M1 Max, 331.0 -> 652.4 tok/s (`.wiki/macos-bench.md` §18).
fn batched_point(
    op: &'static str,
    bank: Bank,
    biased: bool,
    bm: u32,
    bn: u32,
    fp16: bool,
) -> Result<Option<crate::linear::quant::Point>, Error> {
    let (bm, bn) = (stated(op, bm)?, stated(op, bn)?);
    match (bank.affine(), bank.group, bank.bits, biased) {
        // The FP16 staged-weight arm: the loader dequantizes straight to
        // `half` and feeds the instruction pre-Apple9 silicon actually has,
        // which is ~40% of the routed GEMM's arithmetic. Stamped in source,
        // so no jit stamp.
        (true, 64, 4, false) if fp16 => Ok(Some(crate::linear::quant::Point {
            entry: crate::linear::quant::routed_fp16_point(op, bm, bn)?,
            stamp: "",
        })),
        // **AN UNSTAMPED WIDTH IS A DECLINE, NOT A FAULT.** This arm used to
        // walk every affine bank into `qmm_point`, whose `check` against
        // `quant::WIDTHS` returns an ERROR for a width the tiled family is not
        // stamped at — so a bank at such a width faulted the prefill instead
        // of falling through to the matvec arm beside it, which serves every
        // width. Every other refusal on this path answers `Ok(None)` and lets
        // `matmul_select_batched` return `false` (the tile, the column and the
        // `QMM_BK` checks just above it); this one now does too.
        //
        // **TWO IS NO LONGER ONE OF THEM**, and the measurement that put it on
        // `quant::WIDTHS` is written down there: a 2-bit routed bank now takes
        // this tiled arm at prefill and is ~15% faster for it at every prompt
        // length measured. The 2-bit DECODE arm is reached the way it always
        // was, through `routed_point`'s own `(2, {32, 64, 128})` rows — this
        // arm is only asked once `tile_rows` has already said the batch is
        // wide enough to sort.
        (true, group, bits, false) if crate::linear::quant::qmm_stamps_width(bits) => {
            Ok(Some(crate::linear::quant::qmm_point(
                op,
                "_routed",
                "PIE_STAMP_qmm_t_routed",
                stated(op, group)?,
                stated(op, bits)?,
                bm,
                bn,
            )?))
        }
        // Both bias forms, because gpt-oss uses both in the same layer: the
        // gate/up projection carries a per-expert bias and the down one does
        // not, its `down_bias` being folded after the weighted reduce.
        (false, MXFP4_BLOCK, 4, biased) => Ok(Some(crate::linear::quant::Point {
            entry: crate::linear::quant::mxfp4_routed_point(
                op,
                if biased { "_bias" } else { "" },
                bm,
                bn,
            )?,
            stamp: "",
        })),
        _ => Ok(None),
    }
}

/// The sorted, batched routed matmul: one GEMM over runs of equal expert,
/// where the matvec arm runs one simdgroup per (row, four columns).
///
/// **THE SORT IS THE WHOLE OF IT.** `route_sort` histograms the routing
/// decision into per-expert counts, lays each expert's run out on a TILE
/// BOUNDARY, and writes the permutation and its inverse in one threadgroup
/// pass. The gather makes the rows of a run contiguous; the GEMM then reads
/// each expert's weight slice ONCE for the whole run instead of once per
/// routed row, which is what a bandwidth-bound 4-bit mixture is short of.
/// Measured on gpt-oss-20b at 16 lanes, that is 134 → 311 tok/s.
///
/// `tile_rows = 1` collapses the sort to plain grouping, so a decode and a
/// prefill share one dataflow rather than one of them being the special case.
///
/// The permutation is UNDONE rather than folded through, unlike the reference
/// driver's `combine_sorted`: this plane's IR lands `tokens * top_k` routed
/// rows and folds them in a separate statement, so the arm owes its caller
/// that rectangle. See `route_scatter` in `moe_route.metal`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_select_batched(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    bank: Bank,
    bias: Option<Tensor>,
    routes: Tensor,
    experts: u32,
    scratch: RoutedScratch,
    y: Tensor,
    tuning: &crate::DeviceTuning,
) -> Result<bool, Error> {
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    let fan = selected(op, x, routes, y)?;
    routed_bank(op, x, bank)?;
    // The routed rectangle's own rows: `selected` has already agreed them
    // with `tokens x top_k` and refused the multiply that would not fit.
    let pairs = y.rows;
    let tile = tile_rows(pairs, experts, tuning);
    if tile <= 1 {
        return Ok(false);
    }
    let Some(bn) = tile_cols(y.width) else {
        return Ok(false);
    };
    if x.width % QMM_BK != 0 {
        return Ok(false);
    }
    let fp16 = tuning.fp16_gemm_format(bank.bits, bank.group);
    let Some(point) = batched_point(op, bank, bias.is_some(), tile, bn, fp16)? else {
        return Ok(false);
    };
    let padded = sorted_rows(pairs, experts, tuning);
    debug_assert!(
        scratch.x.rows >= padded && scratch.y.rows >= padded,
        "`{op}`'s sorted stack is `sorted_rows` deep"
    );

    // Whether the activation is one row per TOKEN or one per ROUTE is said to
    // the gather as its fan-out: at `1` the pair index IS the row, which is
    // the layout `selected` recognises by a nonzero slot stride.
    let gather_fan = if fan.x_slot_stride == 0 { fan.top_k } else { 1 };
    let sort = [
        routes.arg(),
        scratch.perm.arg_mut(),
        scratch.row_expert.arg_mut(),
        scratch.tile_expert.arg_mut(),
        scratch.inv.arg_mut(),
        pairs.arg(),
        experts.arg(),
        fan.top_k.arg(),
        tile.arg(),
        padded.arg(),
        x.width.arg(),
        0u32.arg(),
    ];
    let lanes = router_lanes(op, experts)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "route_sort").apply(Grid::of([lanes, 1, 1], [lanes, 1, 1])),
        &sort,
    )?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "route_gather").apply(route_rows(op, x.width, padded)?),
        &[
            x.arg(),
            scratch.x.arg_mut(),
            scratch.perm.arg(),
            pairs.arg(),
            experts.arg(),
            gather_fan.arg(),
            tile.arg(),
            padded.arg(),
            x.width.arg(),
            0u32.arg(),
        ],
    )?;

    let mut args = vec![
        bank.codes.arg(),
        bank.scales.arg(),
        zero_points(ctx, bank)?,
        scratch.x.arg(),
        scratch.y.arg_mut(),
        stated(op, x.width)?.arg(),
        stated(op, y.width)?.arg(),
    ];
    // Buffers 7..12 are the seats this family leaves unbound; `tile_expert`
    // is read at 12 and an argument binds at its own index, so the gap is
    // stated rather than closed up.
    args.push(match bias {
        Some(bias) => bias.arg(),
        None => ctx.absent()?,
    });
    for _ in 8..12 {
        args.push(ctx.absent()?);
    }
    args.push(scratch.tile_expert.arg());
    ctx.fire(
        Fire::at(QMM_FILE, point.entry)
            .stamp(point.stamp)
            .apply(Grid::of(
                crate::linear::quant::qmm_grid(
                    op,
                    stated(op, y.width)?,
                    stated(op, bn)?,
                    stated(op, padded)?,
                    stated(op, tile)?,
                    1,
                )?,
                QMM_GROUP,
            )),
        &args,
    )?;

    ctx.fire(
        Fire::at(ROUTE_FILE, "route_scatter").apply(route_rows(op, y.width, pairs)?),
        &[
            scratch.y.arg(),
            y.arg_mut(),
            scratch.inv.arg(),
            pairs.arg(),
            y.width.arg(),
            0u32.arg(),
        ],
    )?;
    Ok(true)
}

/// The zero-point seat, bound or null. Both templates hold it — the mxfp4
/// codec's `dot` never reads it — so an absent plane is a null binding and
/// not a second entry.
fn zero_points(ctx: &Ctx<'_>, bank: Bank) -> Result<crate::encode::ArgValue, Error> {
    match bank.biases {
        Some(biases) => Ok(biases.arg()),
        None => ctx.absent(),
    }
}

/// Folds the `top_k` routed rows back to one row per token, weighted.
pub fn weighted_sum(
    ctx: &Ctx<'_>,
    routed: Tensor,
    weights: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_weighted_sum";
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

/// The routed bias mixture, said once on an already-folded activation:
/// `y[t] = x[t] + Σ_k weights[t, k] · bias[routes[t, k]]`.
///
/// It is its own entry rather than a seat inside the routed matmul because
/// the expert down-projection is rows-cut under tp: each rank's routed
/// matmul is a PARTIAL product, and the all_reduce that follows sums the
/// ranks — a replicated bias folded in there would be summed tp times. The
/// routing is computed from replicated inputs, so `routes` and `weights` are
/// identical on every rank and the mixture can be stated after the reduce,
/// where it lands exactly once. At tp = 1 the weights sum to one, so the
/// value is the same and the model text keeps a single path.
pub fn bias_sum(
    ctx: &Ctx<'_>,
    x: Tensor,
    bias: Tensor,
    routes: Tensor,
    weights: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_bias_sum";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "expert_bias_combine" });
    debug_assert_eq!(
        bias.dtype, x.dtype,
        "the expert bias rides the activation's dtype"
    );
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 routes");
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` reads f32 route weights");
    debug_assert!(
        x.rows == y.rows && x.width == y.width,
        "the token rectangle, which adding a bias does not change"
    );
    debug_assert_eq!(
        bias.width, y.width,
        "the bias bank is one row per expert, the activation's width"
    );
    debug_assert!(
        routes.rows == y.rows && weights.rows == y.rows,
        "a routed plane lands one row per token row"
    );
    debug_assert_eq!(
        routes.width, weights.width,
        "the weight plane is one weight per route"
    );
    let top_k = nonzero(OP, "the routed fan-out", routes.width)?;
    let grid = route_rows(OP, y.width, y.rows)?;
    ctx.fire(
        Fire::at("linear/moe_route.metal", entry).apply(grid),
        &[
            x.arg(),
            bias.arg(),
            routes.arg(),
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
) -> Result<(), Error> {
    const OP: &str = "linear.moe_sigmoid_gate_add";
    let entry = dtype_dispatch!(OP, routed.dtype, { Bf16 => "shared_expert_combine" });
    debug_assert!(
        shared.rows == routed.rows && shared.width == routed.width,
        "the shared expert's rectangle is the routed one"
    );
    debug_assert_eq!(
        gate.rows, routed.rows,
        "the gate column is one scalar per row"
    );
    debug_assert_eq!(
        gate.width, 1,
        "`shared_expert_combine` reads `gate[row]`, so the gate is a column and \
         not a plane the width of what it gates"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceTuning;
    use crate::encode::ArgValue;
    use crate::probe::Probe;

    /// The regression that cost gpt-oss half its prefill: an mxfp4 bank with
    /// NO bias plane has to reach a tiled point, because gpt-oss's expert
    /// down projection is exactly that — `down_bias` is folded after the
    /// weighted reduce. When this arm answered `None`, the driver fell
    /// through to the matvec and read every expert's slice once per routed
    /// row: 1547 -> 785 ms at 512 prompt tokens once it stopped.
    #[test]
    fn an_mxfp4_bank_reaches_a_tiled_point_with_or_without_a_bias() {
        let bank = Bank {
            codes: Tensor::new(0, 2880, 2880, Dtype::U4g64),
            scales: Tensor::new(1, 2880, 2880 / 32, Dtype::E8m0),
            biases: None,
            group: MXFP4_BLOCK,
            bits: 4,
        };
        let biased = batched_point("t", bank, true, 32, 64, false)
            .expect("the biased form answers")
            .expect("and it is stamped");
        assert_eq!(biased.entry, "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_64");
        let plain = batched_point("t", bank, false, 32, 64, false)
            .expect("the unbiased form answers")
            .expect("and it is stamped too — this is the whole fix");
        assert_eq!(plain.entry, "mxfp4_qmm_t_routed_bfloat16_bm_32_bn_64");
    }

    /// The 2-bit routed decode arm: an affine two-bit bank names the
    /// group-parametric `AffineU2` routed points `quant_qmv.metal` instantiates
    /// beside the four-bit one, in both bias forms, at every group the 2-bit
    /// artifacts carry — group 32 (DeepSeek-V4-Flash), group 64 (its one
    /// odd-layer gate), and group 128 (Qwen3.8-Flash). This is the matvec half
    /// of a 2-bit mixture's decode; the tiled batched half is group- and
    /// width-parametric already (`batched_point` JIT-stamps `qmm_t_routed`).
    #[test]
    fn a_two_bit_affine_bank_names_the_routed_arm() {
        let bank = Bank {
            codes: Tensor::new(0, 2048, 2048, Dtype::U4g64),
            scales: Tensor::new(1, 2048, 2048 / 64, Dtype::Bf16),
            biases: Some(Tensor::new(2, 2048, 2048 / 64, Dtype::Bf16)),
            group: 64,
            bits: 2,
        };
        for group in [32, 64, 128] {
            let b = Bank { group, ..bank };
            assert_eq!(
                routed_point("t", b, false).expect("the unbiased 2-bit arm resolves"),
                format!("affine_qmv_routed_bfloat16_gs_{group}_b_2")
            );
            assert_eq!(
                routed_point("t", b, true).expect("the biased 2-bit arm resolves"),
                format!("affine_qmv_routed_bias_bfloat16_gs_{group}_b_2")
            );
        }
        // A group the artifacts do not carry still refuses — the routed codec is
        // instantiated only at {32,64,128}.
        let g16 = Bank { group: 16, ..bank };
        assert!(routed_point("t", g16, false).is_err());
    }

    /// **AND THE TILED HALF TAKES THE SAME BANK, AT EVERY GROUP.**
    ///
    /// Two is on `quant::WIDTHS` — the prefill measurement that put it there
    /// is written at the constant — so a 2-bit routed bank at any of the three
    /// groups the artifacts carry answers `Some` here and takes the sorted
    /// tile family. This is the unit half of that flip: the device half is
    /// `engine-metal`'s `what_a_two_bit_prefill_costs`, and this is what says
    /// the arm is reached at group 32 and 128 as well as at 64, which no SKU
    /// in the catalog exercises all three of.
    ///
    /// **THE DECLINE IS STILL A DECLINE, for a width that is genuinely not
    /// stamped.** `batched_point` walked every affine bank into `qmm_point`,
    /// which checks `bits` against `quant::WIDTHS` and returns an ERROR for an
    /// unstamped one — so such a bank would, at prefill, take the batched door
    /// and come back an Err, a fault on a path with a working arm beside it.
    /// The answer a caller needs is `None`: `matmul_select_batched` reads it
    /// as "not this way" and returns `false`, and the driver falls through to
    /// the matvec. Three bits stands for that case here because
    /// `dequantize`'s own `static_assert` names it as one the templates
    /// refuse.
    #[test]
    fn a_two_bit_bank_takes_the_batched_arm_and_an_unstamped_width_declines_it() {
        let bank = Bank {
            codes: Tensor::new(0, 2048, 2048, Dtype::U4g64),
            // `Some`, because `Bank::affine` IS the biases plane: a bank
            // without one is the mxfp4 family and never reaches the arm under
            // test.
            scales: Tensor::new(1, 2048, 2048 / 64, Dtype::Bf16),
            biases: Some(Tensor::new(2, 2048, 2048 / 64, Dtype::Bf16)),
            group: 64,
            bits: 2,
        };
        for group in [32, 64, 128] {
            let b = Bank { group, ..bank };
            let point = batched_point("t", b, false, 32, 64, false)
                .expect("a stamped width resolves")
                .unwrap_or_else(|| panic!("the 2-bit bank at group {group} takes the tiled arm"));
            assert_eq!(
                point.entry,
                format!("affine_qmm_t_routed_bfloat16_gs_{group}_b_2_bm_32_bn_64")
            );
            assert!(
                point.stamp.starts_with("PIE_STAMP_qmm_t_routed("),
                "the tiled routed point is jit-stamped: {}",
                point.stamp
            );
        }
        // The stamped width still answers, so the decline below is the
        // width's and not every affine bank's.
        let four = Bank { bits: 4, ..bank };
        assert!(
            batched_point("t", four, false, 32, 64, false)
                .expect("four bits resolves")
                .is_some(),
            "the four-bit bank still takes the tiled arm"
        );
        let three = Bank { bits: 3, ..bank };
        assert_eq!(
            batched_point("t", three, false, 32, 64, false)
                .expect("an unstamped width is not a fault"),
            None,
            "a width the templates refuse DECLINES the tiled arm"
        );
    }

    #[test]
    fn a_fleet_batches_where_a_single_decode_does_not() {
        let t = DeviceTuning::default();
        // gpt-oss-20b: 32 experts at top-4. One row is four pairs, which is
        // an eighth of a row an expert.
        assert!(!should_batch(4, 32, 8));
        // At the measured threshold of one, the same fire batches -- which is
        // the 134 -> 311 tok/s step at 16 lanes.
        assert!(should_batch(4 * 16, 32, t.moe_batch_min_per_expert));
        assert_eq!(tile_rows(4 * 16, 32, &t), 16);
    }

    #[test]
    fn the_wide_tile_is_out_of_reach_and_the_mid_one_is_not() {
        let t = DeviceTuning::default();
        // 31 rows an expert still takes the narrow tile, 32 takes the mid --
        // the threshold that moved 12 -> 32 when the routed GEMM stopped
        // emulating a bfloat matrix unit.
        assert_eq!(tile_rows(31 * 32, 32, &t), 16);
        assert_eq!(tile_rows(32 * 32, 32, &t), 32);
        // And nothing reaches 64 on this table.
        assert_eq!(tile_rows(4096 * 32, 32, &t), 32);
    }

    #[test]
    fn the_sorted_bound_is_reachable_and_a_whole_number_of_tiles() {
        let t = DeviceTuning::default();
        let (pairs, experts) = (4 * 16u32, 32u32);
        let tile = tile_rows(pairs, experts, &t);
        let rows = sorted_rows(pairs, experts, &t);
        assert_eq!(rows % tile, 0);
        assert!(rows >= pairs);
        // Every touched expert can waste `tile - 1` rows, and at most
        // `min(pairs, experts)` are touched.
        assert!(rows >= pairs + pairs.min(experts) * (tile - 1));
    }

    #[test]
    fn a_mixture_that_does_not_batch_reports_its_own_rows() {
        let t = DeviceTuning::default();
        assert_eq!(tile_rows(4, 128, &t), 1);
        assert_eq!(sorted_rows(4, 128, &t), 4);
    }

    #[test]
    fn the_column_tile_is_the_widest_that_divides() {
        assert_eq!(tile_cols(2880), Some(64));
        assert_eq!(tile_cols(1024), Some(64));
        assert_eq!(tile_cols(48), Some(16));
        assert_eq!(tile_cols(100), None);
    }

    /// **THE HASH ROUTER GATHERS, IT DOES NOT SCORE.** Its (routes, weights)
    /// are the pair a softmax router lands — `int` routes and `float` weights,
    /// `[tokens, top_k]` row-major — so the fire names the gather point in
    /// `moe_route.metal`, launches one thread per (slot, token row), and
    /// marshals the table, the vocab and the fan-out in the order the shader
    /// reads them. That is what makes it drop-in for the same sorted-MoE path.
    #[test]
    fn the_hash_route_fires_the_gather_at_one_thread_per_slot() {
        let probe = Probe::default();
        // DeepSeek-V4-Flash's hash layers: top-6 over a per-token table.
        let ids = Tensor::new(0, 8, 1, Dtype::U32);
        let tid2eid = Tensor::new(1, 129_280, 6, Dtype::I64);
        let routes = Tensor::new(2, 8, 6, Dtype::I32);
        let weights = Tensor::new(3, 8, 6, Dtype::F32);
        hash_route(&probe, ids, tid2eid, 129_280, 6, routes, weights)
            .expect("the hash route enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.file, "linear/moe_route.metal");
        assert_eq!(f.entrypoint, "hash_route_gather");
        // One thread per (slot, token row), the fan-out its own group.
        assert_eq!(f.lanes, [6, 8, 1]);
        assert_eq!(f.group, [6, 1, 1]);
        assert_eq!(a[0], ArgValue::Buffer(0)); // token ids
        assert_eq!(a[1], ArgValue::Buffer(1)); // the i64 hash table
        assert_eq!(a[2], ArgValue::BufferMut(2)); // routes, drop-in for a router's
        assert_eq!(a[3], ArgValue::BufferMut(3)); // weights, drop-in for a router's
        assert_eq!(a[4], ArgValue::U32(129_280)); // vocab
        assert_eq!(a[5], ArgValue::U32(6)); // top_k
    }

    /// A route plane that is not the i32 the sorted-MoE path reads, or a table
    /// whose width is not the stated fan-out, is a mismatch the entry catches
    /// before any fire — the same plane checks a softmax router's [`ranked`]
    /// makes, on the pair a hash router lands instead of scores.
    ///
    /// **AND IT CATCHES IT WHERE THE GUARD IS COMPILED, WHICH IS WHY THIS
    /// TEST IS TOO.** `ranked_planes` states the marshalling contract with
    /// `debug_assert`, the way all ~170 of this crate's shape and dtype
    /// guards do, so in a release build there is no panic here to expect:
    /// a `should_panic` without this `cfg` was a debug-profile claim asserted
    /// in both, and it failed under `--release` for the one reason release is
    /// different. The split is not an oversight to be promoted away. A route
    /// plane's element is fixed by the IR value the compiler carved the
    /// buffer from — `dispatch::linear` resolves `routes` by `ValueId` and
    /// hands it straight over — so a bf16 one is a bug in the caller, never
    /// something a deployment can ask for. The class this crate spends
    /// `Error` on is the value-dependent one a deployment CAN reach (a zero
    /// fan-out, an expert count over the lane cap), and `refuse` answers that
    /// one in every profile.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "lands i32 routes")]
    fn a_route_plane_that_is_not_i32_trips_the_debug_guard() {
        let probe = Probe::default();
        let ids = Tensor::new(0, 4, 1, Dtype::U32);
        let tid2eid = Tensor::new(1, 32, 6, Dtype::I64);
        // The routes plane is bf16, not the i32 `route_sort` reads.
        let routes = Tensor::new(2, 4, 6, Dtype::Bf16);
        let weights = Tensor::new(3, 4, 6, Dtype::F32);
        let _ = hash_route(&probe, ids, tid2eid, 32, 6, routes, weights);
    }

    /// **THE REFERENCE THE DEVICE POINT ANSWERS TO**, computed on the host so
    /// the numeric contract is pinned where no GPU is: `routes[t, s] =
    /// tid2eid[token_id[t], s]` exactly (i64 → i32, no dedup), every weight
    /// `1/top_k`. The device dispatch and readback live on the engine-metal
    /// device floor, which compares the shader against exactly this gather;
    /// here the semantics and their edges are locked in-crate — a token id at
    /// the vocab boundary reads the LAST row, and a row that names an expert
    /// twice keeps BOTH.
    #[test]
    fn the_host_gather_pins_the_contract_and_its_edges() {
        const VOCAB: usize = 5;
        const K: usize = 6;
        // A synthetic table where row `v` names experts `(v + s) mod 7`, so
        // the last valid id `VOCAB - 1` exercises the boundary read.
        let mut tid2eid = vec![0i64; VOCAB * K];
        for v in 0..VOCAB {
            for s in 0..K {
                tid2eid[v * K + s] = ((v + s) % 7) as i64;
            }
        }
        // A deliberate duplicate in one row: the hash may repeat, and the
        // uniform fold weights every slot alike, so the gather keeps both.
        tid2eid[2 * K + 1] = tid2eid[2 * K];

        // Ids include the boundary (`VOCAB - 1`) and the duplicate row (`2`).
        let token_ids = [0u32, VOCAB as u32 - 1, 2, 0];
        let gather = |t: usize| -> ([i64; K], [f32; K]) {
            let raw = token_ids[t] as usize;
            let v = if raw < VOCAB { raw } else { 0 }; // embed.metal's guard
            let mut e = [0i64; K];
            let mut w = [0f32; K];
            for s in 0..K {
                e[s] = tid2eid[v * K + s];
                w[s] = 1.0 / K as f32;
            }
            (e, w)
        };

        // A boundary id reads the last table row, byte-for-byte, at 1/k.
        let (e1, w1) = gather(1);
        assert_eq!(e1[..], tid2eid[(VOCAB - 1) * K..VOCAB * K]);
        assert!(w1.iter().all(|&x| (x - 1.0 / K as f32).abs() < 1e-9));

        // Row 2's duplicate survives the gather — both slots name it.
        let (e2, _) = gather(2);
        assert_eq!(e2[0], e2[1], "a repeated expert is copied, not deduped");

        // Every in-range id gathers its own row exactly.
        let (e0, _) = gather(0);
        assert_eq!(e0[..], tid2eid[0..K]);
    }
}
