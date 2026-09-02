//! `Moe`: routers, routed matmuls, and the folds that bring the fan-out
//! back. One entry per IR variant. Both quantized bank forms reach the
//! routed matmul; the entry picks between them off the [`Bank`] the driver
//! resolved (an mxfp4 bank has no zero points, an affine one does).

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

/// When sorting the rows by expert pays for itself: trades a partly-full
/// tile's wasted arithmetic for reading each expert's weights once instead
/// of once per pair. `min_per_expert` is a measured threshold, not
/// modeled.
#[must_use]
pub fn should_batch(pairs: u32, experts: u32, min_per_expert: u32) -> bool {
    experts > 0 && u64::from(pairs) >= u64::from(experts) * u64::from(min_per_expert)
}

/// Rows each expert's run is padded to, for a batch of `pairs` — 1 when the
/// mixture does not batch at all. Priced off rows per expert and measured
/// end-to-end (thresholds live in `DeviceTuning`); a wider tile costs
/// nothing extra since [`sorted_rows`] is pessimistic and tiles past the
/// routing decline rather than doing extra arithmetic.
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

/// How many sorted rows a batch of `pairs` can produce. The worst case, not
/// the actual: every touched expert can waste `tile - 1` rows, and at most
/// `min(pairs, experts)` are touched.
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

/// The same router, times the learned per-expert gain. No new kernel: buffer
/// 3, which [`topk_softmax`] binds absent, is the gain plane here.
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

/// **A ROUTE PREDICTION** (`linear.moe_predict_route`): [`topk_sqrt_softplus`]'s
/// ranking under the point the segment cut does not fall after. Unscaled,
/// unnormalized — the weights are nobody's.
pub fn predict_route(
    ctx: &Ctx<'_>,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_predict_route";
    let entry = dtype_dispatch!(OP, logits.dtype, { Bf16 => "router_predict_sqrt_softplus" });
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
            0u32.arg(),
            1.0f32.arg(),
        ],
    )
}

/// The hash router: layers `0..num_hash_layers` route by a per-token
/// lookup, not a learned gate over logits. `tid2eid` is `[vocab, top_k]`
/// i64, naming `top_k` expert ids at uniform weight `1/top_k` per token id;
/// this gathers that row into the same layout [`topk_softmax`] writes, so
/// the output is drop-in for the same sorted-MoE path.
#[allow(clippy::too_many_arguments)]
pub fn hash_route(
    ctx: &Ctx<'_>,
    ids: Tensor,
    tid2eid: Tensor,
    logits: Tensor,
    vocab: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_hash_route";
    debug_assert!(
        matches!(ids.dtype, Dtype::U32 | Dtype::I32),
        "`{OP}` gathers by a 32-bit token id column"
    );
    // **THE WEIGHTS ARE THE GATE'S.** The official `Gate.forward` scores
    // every layer with `sqrt(softplus(x · W))` and, on a hash layer, only
    // replaces the top-k CHOICE with the table's; the chosen experts'
    // weights are still gathered off the scores, renormalized and scaled.
    dtype_dispatch!(OP, logits.dtype, { Bf16 => () });
    debug_assert_eq!(
        logits.rows, routes.rows,
        "the router logits are one row per token row"
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
    // Same shape/element as a softmax router's output.
    ranked_planes(OP, routes, top_k, routes, weights);
    debug_assert_eq!(
        ids.rows, routes.rows,
        "the token ids handed over are the rows this route lands"
    );
    let top_k = nonzero(OP, "the fan-out this router states", top_k)?;
    nonzero(OP, "the vocabulary this table spans", vocab)?;
    let experts = nonzero(OP, "the expert count the logits span", logits.width)?;
    // One thread per token row: the row's `top_k` weights normalize together.
    ctx.fire(
        Fire::at(ROUTE_FILE, "hash_route_gather").apply(Grid::of(
            [routes.rows, 1, 1],
            [routes.rows.min(256), 1, 1],
        )),
        &[
            ids.arg(),
            tid2eid.arg(),
            logits.arg(),
            routes.arg_mut(),
            weights.arg_mut(),
            vocab.arg(),
            experts.arg(),
            top_k.arg(),
            u32::from(renormalize).arg(),
            scaling.arg(),
            routes.rows.arg(),
        ],
    )
}

/// **THE STATIC ROUTES OF A GROUPED PROJECTION**: `routes[n, g] = g`.
pub fn group_routes(ctx: &Ctx<'_>, groups: u32, routes: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.group_routes";
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` lands i32 routes");
    debug_assert_eq!(routes.width, groups, "the routes are one slot per group");
    let groups = nonzero(OP, "the group count", groups)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "group_routes").apply(route_rows(OP, groups, routes.rows)?),
        &[routes.arg_mut(), groups.arg()],
    )
}

/// **THE BLOCK-DIAGONAL PROJECTION** (`linear.matmul_grouped`): `x` is
/// `[tokens, G·K]`, the plane `[G·N, K]`, and `y` `[tokens, G·N]`.
///
/// **NO NEW POINT.** Read with `G` rows per token, `[tokens, G·K]` is
/// `[tokens·G, K]` byte for byte, `[G·N, K]` is a `G`-expert bank of `[N, K]`,
/// and `[tokens, G·N]` is `[tokens·G, N]` — so this is exactly the routed
/// select over the by-route activation (`x.rows == tokens · top_k`, one
/// `K`-wide slice per slot) with [`group_routes`]' `g` in slot `g`, and the
/// three rectangles are restated and handed to it.
pub fn matmul_grouped(
    ctx: &Ctx<'_>,
    x: Tensor,
    plane: GroupedPlane,
    routes: Tensor,
    groups: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_grouped";
    let groups_nz = nonzero(OP, "the group count", groups)?;
    if !x.width.is_multiple_of(groups_nz)
        || !y.width.is_multiple_of(groups_nz)
        || routes.width != groups_nz
    {
        return Err(refuse(
            OP,
            format!(
                "{groups} groups do not divide a {}-wide row into a {}-wide one, or the routes \
                 are {} wide",
                x.width, y.width, routes.width
            ),
        ));
    }
    let rows = x
        .rows
        .checked_mul(groups_nz)
        .ok_or_else(|| refuse(OP, format!("{} rows x {groups} groups will not stride", x.rows)))?;
    let x = Tensor::new(x.buf, rows, x.width / groups_nz, x.dtype);
    let y = Tensor::new(y.buf, rows, y.width / groups_nz, y.dtype);
    match plane {
        GroupedPlane::Bank(bank) => matmul_select_quant(ctx, x, bank, routes, y),
        GroupedPlane::Dense(dense) => matmul_select(ctx, x, dense, routes, y),
    }
}

/// The plane a grouped projection reads: a split-plane quantized bank, or one
/// dense rectangle — resolved by the caller, because a bank's weight never
/// answers as one dense handle.
#[derive(Clone, Copy)]
pub enum GroupedPlane {
    Bank(Bank),
    Dense(Tensor),
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

/// The routed qmv point one bank arrives at. The discriminator is whether the
/// bank carries a biases plane, not the group size: mxfp4's e8m0 scale is the
/// whole dequantization, while affine's bf16 factor needs the zero points
/// too. Group size is then checked against the chosen point rather than used
/// to pick it.
fn routed_point(op: &'static str, bank: Bank, biased: bool) -> Result<&'static str, Error> {
    match (bank.affine(), bank.group, bank.bits) {
        (true, 64, 4) if biased => Ok("affine_qmv_routed_bias_bfloat16_gs_64_b_4"),
        (true, 64, 4) => Ok("affine_qmv_routed_bfloat16_gs_64_b_4"),
        // 2-bit routed decode: `AffineU2` is instantiated at all three groups
        // 2-bit checkpoints carry (group 32, 64, 128).
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
/// named. Every field is scratch (dead once [`matmul_select_batched`]
/// returns), sized off `n_pairs`, `n_experts` and [`sorted_rows`]; the caller
/// allocates it since `kernels-metal` allocates nothing itself.
#[derive(Clone, Copy, Debug)]
pub struct RoutedScratch {
    /// `i32`, [`sorted_rows`] long: the pair index each sorted row came from,
    /// or −1 for a row the padding invented.
    pub perm: Tensor,

    /// `i32`, [`sorted_rows`] long: which expert each sorted row belongs to.
    pub row_expert: Tensor,

    /// `i32`, one per tile of the sorted stack: the expert that tile serves,
    /// or −1 for a tile past the routing (a declined tile returns before
    /// reading a weight, making the padding free).
    pub tile_expert: Tensor,

    /// `i32`, `n_pairs` long: where each pair landed (the inverse
    /// permutation, produced free by the sort).
    pub inv: Tensor,

    /// The activation gathered into expert-major order: `sorted_rows x K`.
    pub x: Tensor,

    /// The routed product in that same order: `sorted_rows x N`.
    pub y: Tensor,
}

/// The routed tiled point one bank and one tile arrive at, or `None` when the
/// shader stamps none for that combination. Four families are stamped; the
/// fifth (`affine_qmm_t_routed` with a per-expert bias) is absent on purpose
/// since that family carries no bias seat, so a biased affine bank falls
/// through to the matvec arm instead. Both mxfp4 bias forms are stamped
/// (biased and unbiased), since a mixture may bias some expert projections
/// and not others.
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
        // `half` for pre-Apple9 silicon. Stamped in source, so no jit stamp.
        (true, 64, 4, false) if fp16 => Ok(Some(crate::linear::quant::Point {
            entry: crate::linear::quant::routed_fp16_point(op, bm, bn)?,
            stamp: "",
        })),
        // An unstamped width declines (`Ok(None)`) rather than faulting, so
        // the caller falls through to the matvec arm, which serves every
        // width.
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
/// vs. the matvec arm's one simdgroup per (row, four columns). `route_sort`
/// lays each expert's run out on a tile boundary and writes the
/// permutation and its inverse; the gather then makes each run's rows
/// contiguous so the GEMM reads each expert's weight slice once per run
/// instead of once per row. `tile_rows = 1` collapses the sort to plain
/// grouping, so decode and prefill share one dataflow.
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

    // Fan-out 1 means the activation is already one row per route (pair
    // index == row); `selected` signals that via a nonzero slot stride.
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
    // Buffers 7..12 are unbound seats in this family; `tile_expert` is
    // read at index 12.
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

/// The zero-point seat, bound or null. Both templates hold the seat (mxfp4's
/// `dot` never reads it), so an absent plane is just a null binding.
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
/// `y[t] = x[t] + Σ_k weights[t, k] · bias[routes[t, k]]`. Its own entry
/// rather than a seat inside the routed matmul because the expert
/// down-projection is rows-cut under tp: each rank's matmul is a partial
/// product, and folding the (replicated) bias in there would sum it tp
/// times. Stating it after the all_reduce lands it exactly once.
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
    
    
    

    /// An mxfp4 bank with no bias plane must still reach a tiled point (not
    /// just the biased form).
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

}
