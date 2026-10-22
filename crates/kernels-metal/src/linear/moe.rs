use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::{Bank, Tensor};

const ROUTER_MAX_EXPERTS: u32 = 1024;

const ROUTER_MAX_TOP_K: u32 = 16;

const SELECT_GROUP: u32 = 128;

const QMV_GROUP: [u32; 3] = [32, 2, 1];

const MXFP4_BLOCK: u32 = 32;

const SOFTMAX_OVER_SELECTED: u32 = 0;

const ROUTE_FILE: &str = "linear/moe_route.metal";

const QMM_FILE: &str = "linear/quant_qmm_t.metal";

const QMM_BK: u32 = 32;

const MOE_TILE_ROWS: [u32; 4] = [8, 16, 32, 64];

const MOE_TILE_COLS: [u32; 3] = [16, 32, 64];

#[must_use]
pub fn tile_rows(pairs: u32, experts: u32, tuning: &crate::Tuning) -> u32 {
    let tile = tuning.moe_tile_rows(pairs, experts);
    debug_assert!(
        tile == 1 || MOE_TILE_ROWS.contains(&tile),
        "`kernel.moe.tile_rows` declares the tiles this crate stamps"
    );
    tile
}

#[must_use]
pub fn sorted_rows(pairs: u32, experts: u32, tuning: &crate::Tuning) -> u32 {
    let tile = tile_rows(pairs, experts, tuning);
    if tile <= 1 {
        return pairs;
    }
    stack_rows(pairs, experts, tile)
}

fn stack_rows(pairs: u32, experts: u32, tile: u32) -> u32 {
    let touched = pairs.min(experts);
    let bound = pairs.saturating_add(touched.saturating_mul(tile - 1));
    bound.div_ceil(tile) * tile
}

#[must_use]
pub fn tile_cols(out_width: u32) -> Option<u32> {
    MOE_TILE_COLS
        .iter()
        .rev()
        .copied()
        .find(|tile| out_width.is_multiple_of(*tile))
}

fn router_lanes(op: &'static str, experts: u32) -> Result<u32, Error> {
    nonzero(op, "the expert count this router states", experts)?;
    Ok(experts.min(1024).div_ceil(32) * 32)
}

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
            ctx.absent()?,
            experts.arg(),
            top_k.arg(),
            SOFTMAX_OVER_SELECTED.arg(),
            experts.arg(),
        ],
    )
}

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

#[allow(clippy::too_many_arguments)]
pub fn topk_sigmoid_biased(
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
    const OP: &str = "linear.moe_topk_sigmoid";
    let entry = dtype_dispatch!(OP, logits.dtype, { Bf16 => "router_topk_sigmoid_biased" });
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
    dtype_dispatch!(OP, logits.dtype, { Bf16 => () });
    debug_assert_eq!(
        logits.rows, routes.rows,
        "the router logits are one row per token row"
    );
    debug_assert_eq!(tid2eid.dtype, Dtype::I64, "`{OP}` reads the i64 hash table");
    debug_assert_eq!(
        tid2eid.width, top_k,
        "the hash table names `top_k` experts per token id"
    );
    ranked_planes(OP, routes, top_k, routes, weights);
    debug_assert_eq!(
        ids.rows, routes.rows,
        "the token ids handed over are the rows this route lands"
    );
    let top_k = nonzero(OP, "the fan-out this router states", top_k)?;
    nonzero(OP, "the vocabulary this table spans", vocab)?;
    let experts = nonzero(OP, "the expert count the logits span", logits.width)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "hash_route_gather")
            .apply(Grid::of([routes.rows, 1, 1], [routes.rows.min(256), 1, 1])),
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
    let rows = x.rows.checked_mul(groups_nz).ok_or_else(|| {
        refuse(
            OP,
            format!("{} rows x {groups} groups will not stride", x.rows),
        )
    })?;
    let x = Tensor::new(x.buf, rows, x.width / groups_nz, x.dtype);
    let y = Tensor::new(y.buf, rows, y.width / groups_nz, y.dtype);
    match plane {
        GroupedPlane::Bank(bank) => matmul_select_quant(ctx, x, bank, routes, y),
        GroupedPlane::Dense(dense) => matmul_select(ctx, x, dense, routes, y),
    }
}

#[derive(Clone, Copy)]
pub enum GroupedPlane {
    Bank(Bank),
    Dense(Tensor),
}

struct Selected {
    tokens: u32,

    top_k: u32,

    x_row_stride: u32,

    x_slot_stride: u32,
}

fn selected(op: &'static str, x: Tensor, routes: Tensor, y: Tensor) -> Result<Selected, Error> {
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

fn routed_point(op: &'static str, bank: Bank, biased: bool) -> Result<&'static str, Error> {
    match (bank.affine(), bank.group, bank.bits) {
        (true, 64, 4) if biased => Ok("affine_qmv_routed_bias_bfloat16_gs_64_b_4"),
        (true, 64, 4) => Ok("affine_qmv_routed_bfloat16_gs_64_b_4"),
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
    if !x.width.is_multiple_of(bank.group) {
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
    debug_assert_eq!(
        bias.dtype, x.dtype,
        "the expert bias rides the activation's dtype"
    );
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
            ctx.absent()?,
            routes.arg(),
            stated(OP, fan.x_slot_stride)?.arg(),
            stated(OP, fan.x_row_stride)?.arg(),
            stated(OP, fan.top_k)?.arg(),
        ],
    )
}

#[derive(Clone, Copy, Debug)]
pub struct RoutedScratch {
    pub perm: Tensor,

    pub row_expert: Tensor,

    pub tile_expert: Tensor,

    pub inv: Tensor,

    pub x: Tensor,

    pub y: Tensor,

    /// One f32 per sorted row and 64-wide activation group: what the MPP matmul folds the
    /// zero points through.
    pub sums: Tensor,
}

/// The MPP point of a routed matmul, as `(row tile, column tile, simdgroups)`.
fn mpp_point(
    x: Tensor,
    bank: Bank,
    biased: bool,
    y: Tensor,
    tile: u32,
    tuning: &crate::Tuning,
) -> Option<(u32, u32, u32)> {
    let m = tile.min(32);
    let (_, n) = crate::tuning::mpp_tile(m);
    let simdgroups = if m == 8 { 2 } else { 4 };
    let fits = tuning.moe_mpp()
        && !biased
        && bank.affine()
        && bank.bits == 4
        && bank.group == 64
        && bank.scales.dtype == Dtype::Bf16
        && x.width.is_multiple_of(64)
        && y.width.is_multiple_of(n);
    fits.then_some((m, n, simdgroups))
}

/// The MPP point of a routed MXFP4 matmul, as `(row tile, column tile, simdgroups)`: the
/// codes are decoded into threadgroup bfloat a 64-wide slice at a time, as mlx-lm's
/// `gather_qmm_rhs_nax` does, so a row tile is the sorted tile, inside one expert's rows.
fn mxfp4_mpp_point(
    x: Tensor,
    bank: Bank,
    y: Tensor,
    tile: u32,
    tuning: &crate::Tuning,
) -> Option<(u32, u32, u32)> {
    // A bank lays an expert's codes out as `y.width` rows of `x.width` or as one row.
    let whole =
        bank.ld == x.width || u64::from(bank.ld) == u64::from(y.width) * u64::from(x.width) / 2;
    let fits = tuning.moe_mpp()
        && !bank.affine()
        && bank.group == MXFP4_BLOCK
        && bank.bits == 4
        && whole
        && tile >= 32
        && x.width.is_multiple_of(64)
        && y.width.is_multiple_of(64);
    fits.then_some((tile, 64, 4))
}

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
        (true, 64, 4, false) if fp16 => Ok(Some(crate::linear::quant::Point {
            entry: crate::linear::quant::routed_fp16_point(op, bm, bn)?,
            stamp: "",
        })),
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

#[allow(clippy::too_many_arguments)]
/// Where a batched routed matmul's rows come from and go: a chain that keeps
/// a routed MLP's rows sorted, as Splash's does, skips the sort and gather of
/// a pass whose input is already the sorted stack and the scatter of one whose
/// output the next pass reads sorted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sorted {
    /// `scratch.x` already holds the input's rows in this routing's order.
    pub input: bool,
    /// Leave the output's rows sorted in `scratch.y`; `y` is not written.
    pub output: bool,
    /// `scratch.sums` already holds the sorted input's group sums at the MPP row tile
    /// [`mpp_sums_tile`] names: the activation that wrote the rows landed them.
    pub sums: bool,
}

/// The row tile whose 64-wide group sums the batched routed matmul of `x` over `bank` into
/// `y` reads from `scratch.sums`, when it takes the MPP kernel: what a pass writing its
/// input rows lands them at to spare the matmul its sums pass.
#[must_use]
pub fn mpp_sums_tile(
    x: Tensor,
    bank: Bank,
    y: Tensor,
    experts: u32,
    sums: Tensor,
    tuning: &crate::Tuning,
) -> Option<u32> {
    let mut tile = tile_rows(y.rows, experts, tuning);
    if tile <= 1 || tile_cols(y.width).is_none() || !x.width.is_multiple_of(QMM_BK) {
        return None;
    }
    let fp16 = tuning.fp16_gemm_format(bank.bits, bank.group);
    if tile == MOE_TILE_ROWS[0] && !(fp16 && bank.affine() && bank.bits == 4 && bank.group == 64) {
        tile = MOE_TILE_ROWS[1];
    }
    let tile = bank_tile(x, bank, y, tile, tuning);
    let padded = stack_rows(y.rows, experts, tile);
    let sums_fit = sums.dtype == Dtype::F32
        && u64::from(sums.rows) * u64::from(sums.width)
            >= u64::from(padded) * u64::from(x.width / 64);
    mpp_point(x, bank, false, y, tile, tuning)
        .filter(|_| sums_fit && x.width.is_multiple_of(64))
        .map(|(m, _, _)| m)
}

/// The rows a batched routed matmul of `x` over `bank` into `y` sorts its routes into, and
/// its tile, when it runs at all.
#[must_use]
pub fn sorted_stack(
    x: Tensor,
    bank: Bank,
    y: Tensor,
    experts: u32,
    tuning: &crate::Tuning,
) -> Option<(u32, u32)> {
    let tile = tile_rows(y.rows, experts, tuning);
    (tile > 1).then(|| {
        let tile = bank_tile(x, bank, y, tile, tuning);
        (stack_rows(y.rows, experts, tile), tile)
    })
}

/// A 64-row tile only where the MXFP4 MPP kernel multiplies it whole; any other kernel
/// multiplies at most 32 rows at a time, and the wider tile would only pad more.
fn bank_tile(x: Tensor, bank: Bank, y: Tensor, tile: u32, tuning: &crate::Tuning) -> u32 {
    if tile == MOE_TILE_ROWS[3] && mxfp4_mpp_point(x, bank, y, tile, tuning).is_none() {
        MOE_TILE_ROWS[2]
    } else {
        tile
    }
}

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
    tuning: &crate::Tuning,
    sorted: Sorted,
) -> Result<bool, Error> {
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    let fan = selected(op, x, routes, y)?;
    routed_bank(op, x, bank)?;
    let pairs = y.rows;
    let mut tile = tile_rows(pairs, experts, tuning);
    if tile <= 1 {
        return unsorted(op, sorted);
    }
    let Some(bn) = tile_cols(y.width) else {
        return unsorted(op, sorted);
    };
    if !x.width.is_multiple_of(QMM_BK) {
        return unsorted(op, sorted);
    }
    let fp16 = tuning.fp16_gemm_format(bank.bits, bank.group);
    if tile == MOE_TILE_ROWS[0] && !(fp16 && bank.affine() && bank.bits == 4 && bank.group == 64) {
        tile = MOE_TILE_ROWS[1];
    }
    let tile = bank_tile(x, bank, y, tile, tuning);
    let padded = stack_rows(pairs, experts, tile);
    let sums = scratch.sums;
    let sums_fit = sums.dtype == Dtype::F32
        && u64::from(sums.rows) * u64::from(sums.width)
            >= u64::from(padded) * u64::from(x.width / 64);
    let mpp = mpp_point(x, bank, bias.is_some(), y, tile, tuning).filter(|_| sums_fit);
    let mxfp4 = mxfp4_mpp_point(x, bank, y, tile, tuning);
    let point = match mpp.or(mxfp4) {
        Some(_) => None,
        None => match batched_point(op, bank, bias.is_some(), tile, bn, fp16)? {
            Some(point) => Some(point),
            None => return Ok(false),
        },
    };
    debug_assert!(
        scratch.x.rows >= padded && scratch.y.rows >= padded,
        "`{op}`'s sorted stack is `sorted_rows` deep"
    );

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
    // The MPP kernel's group sums ride the gather, which has the rows in hand.
    let gathered_sums = !sorted.input && mpp.is_some() && x.width.is_multiple_of(64);
    if !sorted.input {
        ctx.fire(
            Fire::at(ROUTE_FILE, "route_sort").apply(Grid::of([lanes, 1, 1], [lanes, 1, 1])),
            &sort,
        )?;
    }
    if let (true, Some((m, _, _))) = (gathered_sums, mpp) {
        ctx.fire(
            Fire::at(ROUTE_FILE, "route_gather_sums")
                .apply(Grid::of([256, padded, 1], [256, 1, 1])),
            &[
                x.arg(),
                scratch.x.arg_mut(),
                scratch.perm.arg(),
                scratch.sums.arg_mut(),
                gather_fan.arg(),
                x.width.arg(),
                m.arg(),
                0u32.arg(),
            ],
        )?;
    } else if !sorted.input {
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
    }

    if let Some((m, n, simdgroups)) = mxfp4 {
        const FILE: &str = "linear/quant_qmm_mpp.metal";
        let biased = bias.is_some();
        let entry = crate::linear::quant::symbol(&format!(
            "mxfp4_qmm_mpp_routed_m{m}_n{n}_s{simdgroups}_t{tile}_b{}",
            u8::from(biased)
        ));
        let stamp = crate::linear::quant::symbol(&format!(
            "PIE_MPP_MXFP4_POINT(\"{entry}\",{m},{n},{simdgroups},{tile},{biased})"
        ));
        ctx.fire(
            Fire::at(FILE, entry).stamp(stamp).apply(Grid::of(
                [32 * simdgroups * (y.width / n), padded / m, 1],
                [32 * simdgroups, 1, 1],
            )),
            &[
                bank.codes.arg(),
                bank.scales.arg(),
                scratch.x.arg(),
                scratch.y.arg_mut(),
                stated(op, x.width)?.arg(),
                stated(op, y.width)?.arg(),
                match bias {
                    Some(bias) => bias.arg(),
                    None => ctx.absent()?,
                },
                scratch.tile_expert.arg(),
                scratch.perm.arg(),
            ],
        )?;
    } else if let Some((m, n, simdgroups)) = mpp {
        const FILE: &str = "linear/quant_qmm_mpp.metal";
        let k = stated(op, x.width)?;
        debug_assert!(
            !sorted.sums
                || (bias.is_none()
                    && mpp_sums_tile(x, bank, y, experts, scratch.sums, tuning) == Some(m)),
            "`{op}`'s input sums were landed at the tile its MPP kernel reads"
        );
        if !gathered_sums && !sorted.sums {
            ctx.fire(
                Fire::at(FILE, "qmm_mpp_input_sums")
                    .apply(Grid::of([32 * (x.width / 64), padded, 1], [32, 4, 1])),
                &[scratch.x.arg(), scratch.sums.arg_mut(), k.arg(), m.arg()],
            )?;
        }
        let entry = crate::linear::quant::symbol(&format!(
            "affine_qmm_mpp_routed_m{m}_n{n}_s{simdgroups}_t{tile}"
        ));
        let stamp = crate::linear::quant::symbol(&format!(
            "PIE_MPP_ROUTED_POINT(\"{entry}\",{m},{n},{simdgroups},{tile})"
        ));
        ctx.fire(
            Fire::at(FILE, entry).stamp(stamp).apply(Grid::of(
                [32 * simdgroups * (y.width / n), padded / m, 1],
                [32 * simdgroups, 1, 1],
            )),
            &[
                bank.codes.arg(),
                bank.scales.arg(),
                zero_points(ctx, bank)?,
                scratch.x.arg(),
                scratch.y.arg_mut(),
                k.arg(),
                stated(op, y.width)?.arg(),
                scratch.sums.arg(),
                scratch.tile_expert.arg(),
            ],
        )?;
    } else if let Some(point) = point {
        let mut args = vec![
            bank.codes.arg(),
            bank.scales.arg(),
            zero_points(ctx, bank)?,
            scratch.x.arg(),
            scratch.y.arg_mut(),
            stated(op, x.width)?.arg(),
            stated(op, y.width)?.arg(),
        ];
        args.push(match bias {
            Some(bias) => bias.arg(),
            None => ctx.absent()?,
        });
        for _ in 8..12 {
            args.push(ctx.absent()?);
        }
        args.push(scratch.tile_expert.arg());
        let mut grid = crate::linear::quant::qmm_grid(
            op,
            stated(op, y.width)?,
            stated(op, bn)?,
            stated(op, padded)?,
            stated(op, tile)?,
            1,
        )?;
        // The stamped affine kernel is the steel tile's own; the MXFP4 and fp16 ones
        // take the routed threadgroup.
        let group = if point.stamp.is_empty() {
            let group = crate::linear::quant::routed_group(stated(op, tile)?);
            grid[2] = group[2];
            group
        } else {
            crate::linear::quant::qmm_group(stated(op, tile)?)
        };
        ctx.fire(
            Fire::at(QMM_FILE, point.entry)
                .stamp(point.stamp)
                .apply(Grid::of(grid, group)),
            &args,
        )?;
    }

    if sorted.output {
        return Ok(true);
    }
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

/// A pass whose input a chain left sorted cannot fall back to the per-route
/// path, which reads the unsorted rows the chain never wrote.
fn unsorted(op: &'static str, sorted: Sorted) -> Result<bool, Error> {
    if sorted.input {
        return Err(refuse(
            op,
            "a sorted chain's pass did not take the batched path".to_string(),
        ));
    }
    Ok(false)
}

/// The most tokens a routed MLP runs on the register tile: an expert's routes
/// of so many tokens fill one 8-row tile.
pub const REGISTER_TOKENS: u32 = 8;

/// The most experts the register tiles group: a thread each.
pub const REGISTER_EXPERTS: u32 = ROUTER_MAX_EXPERTS;

/// The widest fan-out the register tiles group: `moe_sg_group_gather` stages
/// [`REGISTER_TOKENS`] tokens' routes of it.
pub const REGISTER_FAN_OUT: u32 = 16;

const REGISTER_FILE: &str = "linear/moe_sg.metal";

/// How the register expert passes cut their work: the intermediate's
/// 8-column fragments a simdgroup of the gate-up takes (each with its gate
/// and up columns), the simdgroups a threadgroup of each pass holds, and the
/// tile slots of their grids (0: one for every tile a fire may fill), each
/// slot taking every slots-th live tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterCut {
    pub fragments: u32,
    pub glu_simdgroups: u32,
    pub down_simdgroups: u32,
    pub slots: u32,
}

/// The cut the register expert passes run at: on an M4 Pro over a 35B-A3B
/// layer's eight verify rows, 24 slots take a fifth off the gate-up of 21
/// experts' tiles and add a twentieth to 59's.
pub const REGISTER_CUT: RegisterCut = RegisterCut {
    fragments: 1,
    glu_simdgroups: 4,
    down_simdgroups: 4,
    slots: 24,
};

impl RegisterCut {
    /// The tile slots of a pass over up to `tiles` tiles.
    #[must_use]
    pub fn slots(&self, tiles: u32) -> u32 {
        if self.slots == 0 {
            tiles
        } else {
            self.slots.min(tiles)
        }
    }
}

/// The scratch of a routed MLP on the register tile, as `moe_sg.metal` reads it.
#[derive(Clone, Copy, Debug)]
pub struct RegisterRooms {
    /// The live tile count, then each tile's expert: i32, [`register_tiles`] + 1.
    pub tiles: Tensor,
    /// Each tile row's token, -1 past its routes: i32, 8 a tile.
    pub rows: Tensor,
    /// Each route's tile row: i32, a route each.
    pub inv: Tensor,
    /// Each tile's input table over the gate-up's input, [`register_table_width`] bf16 apiece.
    pub table: Tensor,
    /// Each tile's input table over the down's input, which the gate-up lands.
    pub hidden: Tensor,
    /// Each tile's eight rows of the down's output, of which those that hold
    /// a route are written.
    pub y: Tensor,
    /// What [`register_combine`] reads beside the rows: the routes' weights,
    /// then from [`REGISTER_MIX_GATES`] each token's shared expert gate: f32,
    /// [`REGISTER_MIX`].
    pub mix: Tensor,
}

/// Where the shared expert's gates sit in [`RegisterRooms::mix`], past the
/// most routes a fire groups (`kMixGates`).
pub const REGISTER_MIX_GATES: u32 = 128;

/// The f32 width of [`RegisterRooms::mix`].
pub const REGISTER_MIX: u32 = REGISTER_MIX_GATES + REGISTER_TOKENS;

/// The routes' weights and the shared expert's scalar gate (a bank of one
/// column), which the grouping lands for [`register_combine`].
#[derive(Clone, Copy, Debug)]
pub struct RegisterGate {
    pub weights: Tensor,
    pub gate: Bank,
}

/// Whether the register grouping takes `bank` as the shared expert's scalar
/// gate over `width` inputs: one column, 4- or 8-bit, 64-group affine with
/// bf16 factors.
#[must_use]
pub fn register_gate_bank(bank: Bank, width: u32) -> bool {
    bank.affine()
        && matches!(bank.bits, 4 | 8)
        && bank.group == 64
        && bank.scales.dtype == Dtype::Bf16
        && bank.codes.rows == 1
        && bank.ld == width
        && width.is_multiple_of(256)
}

/// The tiles a fire of `tokens` tokens routing to `top_k` of `experts`
/// experts may fill: an expert's routes take one, and the shared expert, where
/// one rides along, one more.
#[must_use]
pub fn register_tiles(tokens: u32, top_k: u32, experts: u32, shared: bool) -> u32 {
    tokens.saturating_mul(top_k).min(experts) + u32::from(shared)
}

/// The bf16 width of one tile's input table over `width` inputs: its eight
/// rows' inputs and their sums of each 32.
#[must_use]
pub fn register_table_width(width: u32) -> u64 {
    8 * u64::from(width) + u64::from(width / 2)
}

/// Whether a bank multiplies on the register tile: 4-bit, 64-group affine
/// with bf16 factors, each expert `columns` whole 64-column tiles over a
/// whole number of 256-input spans, its rows packed (laid as rows of `width`
/// codes, or an expert a row).
#[must_use]
pub fn register_bank(bank: Bank, width: u32, columns: u32) -> bool {
    bank.affine()
        && bank.bits == 4
        && bank.group == 64
        && bank.scales.dtype == Dtype::Bf16
        && (bank.ld == width || u64::from(bank.ld) == u64::from(columns) * u64::from(width))
        && width.is_multiple_of(256)
        && columns.is_multiple_of(64)
}

/// Whether the register passes cut as `cut` take an intermediate of `inter`
/// columns and a down of `columns`.
#[must_use]
pub fn register_cuts(inter: u32, columns: u32, cut: RegisterCut) -> bool {
    inter.is_multiple_of((8 * cut.fragments * cut.glu_simdgroups).max(64))
        && columns.is_multiple_of(16 * cut.down_simdgroups)
}

/// The routes of `routes` (at most [`REGISTER_TOKENS`] tokens) grouped into
/// the register tile's expert tiles, after the shared expert's where `shared`,
/// and each tile's input table laid from `x`, in one launch; where `gate`,
/// the routes' weights and each token's shared expert gate land in
/// `rooms.mix` too, for [`register_combine`].
pub fn register_group(
    ctx: &Ctx<'_>,
    x: Tensor,
    routes: Tensor,
    experts: u32,
    shared: bool,
    gate: Option<RegisterGate>,
    rooms: RegisterRooms,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 routes");
    let tokens = nonzero(OP, "the routed tokens", routes.rows)?;
    let top_k = nonzero(OP, "the routed fan-out", routes.width)?;
    if tokens > REGISTER_TOKENS
        || top_k > REGISTER_FAN_OUT
        || x.rows != tokens
        || !x.width.is_multiple_of(256)
    {
        return Err(refuse(
            OP,
            format!(
                "the register tile takes at most {REGISTER_TOKENS} tokens' rows of whole \
                 256-input spans and {REGISTER_FAN_OUT} routes a token, not {} x {} over \
                 {tokens} tokens of {top_k}",
                x.rows, x.width
            ),
        ));
    }
    let tiles = register_tiles(tokens, top_k, experts, shared);
    debug_assert!(
        rooms.tiles.width > tiles
            && rooms.rows.width >= 8 * tiles
            && u64::from(rooms.table.width) >= u64::from(tiles) * register_table_width(x.width),
        "`{OP}`'s register rooms hold every tile a fire may fill"
    );
    if experts > REGISTER_EXPERTS {
        return Err(refuse(
            OP,
            format!(
                "{experts} experts are more than the {REGISTER_EXPERTS} the register tiles group"
            ),
        ));
    }
    if let Some(gate) = gate
        && (!shared
            || !register_gate_bank(gate.gate, x.width)
            || gate.weights.rows != tokens
            || gate.weights.width != top_k
            || rooms.mix.width < REGISTER_MIX)
    {
        return Err(refuse(
            OP,
            "the register grouping lands the shared expert's gate of a one-column 4- or 8-bit \
             affine bank, beside the shared tile, and the weights of the routes it groups"
                .to_string(),
        ));
    }
    // A fire without a gate binds the input in its planes' places; nothing reads them.
    let (weights, codes, scales, biases, bits) = match gate {
        Some(RegisterGate { weights, gate }) => (
            weights.arg(),
            gate.codes.arg(),
            gate.scales.arg(),
            zero_points(ctx, gate)?,
            gate.bits,
        ),
        None => (x.arg(), x.arg(), x.arg(), x.arg(), 0),
    };
    ctx.fire(
        Fire::at(REGISTER_FILE, "moe_sg_group_gather")
            .apply(Grid::of([32, x.width / 64, tiles], [32, 4, 1])),
        &[
            x.arg(),
            routes.arg(),
            rooms.table.arg_mut(),
            rooms.tiles.arg_mut(),
            rooms.rows.arg_mut(),
            rooms.inv.arg_mut(),
            x.width.arg(),
            tokens.arg(),
            top_k.arg(),
            experts.arg(),
            u32::from(shared).arg(),
            weights,
            codes,
            scales,
            biases,
            rooms.mix.arg_mut(),
            bits.arg(),
        ],
    )
}

/// Each of `y`'s token rows from the register tiles' down rows: its routes'
/// rows weighed and the shared expert's (tile 0's) gated, by what
/// [`register_group`] landed in `rooms.mix`. `y` is the gated add's output, so
/// the fold, the shared expert's own pass and its gate run nothing.
pub fn register_combine(
    ctx: &Ctx<'_>,
    rooms: RegisterRooms,
    tiles: u32,
    top_k: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_sigmoid_gate_add";
    dtype_dispatch!(OP, y.dtype, { Bf16 => () });
    let tokens = nonzero(OP, "the token rows the combine lands", y.rows)?;
    if tokens > REGISTER_TOKENS
        || top_k > REGISTER_FAN_OUT
        || !y.width.is_multiple_of(4)
        || rooms.mix.width < REGISTER_MIX
        || rooms.y.rows < 8 * tiles
        || rooms.y.width < y.width
    {
        return Err(refuse(
            OP,
            format!(
                "the register combine lands at most {REGISTER_TOKENS} rows of whole 4-column \
                 spans from the rows its rooms hold, not {tokens} x {}",
                y.width
            ),
        ));
    }
    ctx.fire(
        Fire::at(REGISTER_FILE, "moe_sg_combine")
            .apply(Grid::of([y.width / 4, tokens, 1], [128, 1, 1])),
        &[
            rooms.y.arg(),
            rooms.inv.arg(),
            rooms.mix.arg(),
            y.arg_mut(),
            y.width.arg(),
            top_k.arg(),
        ],
    )
}

/// The banks of a register pass: the routed experts' and the shared expert's,
/// which a fire without one binds as the routed bank again (no tile reads it).
fn register_banks(
    ctx: &Ctx<'_>,
    bank: Bank,
    shared: Option<Bank>,
) -> Result<[crate::encode::ArgValue; 6], Error> {
    let own = shared.unwrap_or(bank);
    Ok([
        bank.codes.arg(),
        bank.scales.arg(),
        zero_points(ctx, bank)?,
        own.codes.arg(),
        own.scales.arg(),
        zero_points(ctx, own)?,
    ])
}

/// A routed gate-up (`gate_up`: each expert's `2 * inter` packed gate and up
/// rows over `width` inputs) and its SwiGLU on the register tile, over the
/// tiles [`register_group`] laid: lands each tile's down input table in
/// `rooms.hidden`.
#[allow(clippy::too_many_arguments)]
pub fn register_glu(
    ctx: &Ctx<'_>,
    gate_up: Bank,
    shared: Option<Bank>,
    width: u32,
    inter: u32,
    tiles: u32,
    experts: u32,
    rooms: RegisterRooms,
    cut: RegisterCut,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    let (fragments, simdgroups) = (cut.fragments, cut.glu_simdgroups);
    let entry = match (fragments, simdgroups) {
        (1, 4) => "affine_qmm_sg_glu_routed_h1_s4",
        (1, 8) => "affine_qmm_sg_glu_routed_h1_s8",
        (1, 16) => "affine_qmm_sg_glu_routed_h1_s16",
        (2, 4) => "affine_qmm_sg_glu_routed_h2_s4",
        (2, 8) => "affine_qmm_sg_glu_routed_h2_s8",
        _ => {
            return Err(refuse(
                OP,
                format!(
                    "no register gate-up takes {fragments} fragments a simdgroup in \
                     {simdgroups}-simdgroup threadgroups"
                ),
            ));
        }
    };
    let columns = 8 * fragments * simdgroups;
    if !inter.is_multiple_of(columns.max(64)) || !register_bank(gate_up, width, 2 * inter) {
        return Err(refuse(
            OP,
            format!("the register gate-up does not take this bank over {width} x {inter}"),
        ));
    }
    let [codes, scales, biases, own_codes, own_scales, own_biases] =
        register_banks(ctx, gate_up, shared)?;
    ctx.fire(
        Fire::at(REGISTER_FILE, entry).apply(Grid::of(
            [32 * simdgroups * (inter / columns), cut.slots(tiles), 1],
            [32 * simdgroups, 1, 1],
        )),
        &[
            codes,
            scales,
            biases,
            own_codes,
            own_scales,
            own_biases,
            rooms.table.arg(),
            rooms.hidden.arg_mut(),
            rooms.tiles.arg(),
            stated(OP, width)?.arg(),
            stated(OP, inter)?.arg(),
            stated(OP, experts)?.arg(),
        ],
    )
}

/// A routed down (each expert's `columns` rows over `inter` inputs) on the
/// register tile, over the tables [`register_glu`] landed: each tile's rows
/// that hold a route into `rooms.y`, route `at`'s at row `rooms.inv[at]`.
#[allow(clippy::too_many_arguments)]
pub fn register_down(
    ctx: &Ctx<'_>,
    down: Bank,
    shared: Option<Bank>,
    inter: u32,
    columns: u32,
    tiles: u32,
    experts: u32,
    rooms: RegisterRooms,
    cut: RegisterCut,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    let simdgroups = cut.down_simdgroups;
    let entry = match simdgroups {
        4 => "affine_qmm_sg_routed_s4",
        8 => "affine_qmm_sg_routed_s8",
        16 => "affine_qmm_sg_routed_s16",
        _ => {
            return Err(refuse(
                OP,
                format!("no register down runs {simdgroups}-simdgroup threadgroups"),
            ));
        }
    };
    let span = 16 * simdgroups;
    if !columns.is_multiple_of(span) || !register_bank(down, inter, columns) {
        return Err(refuse(
            OP,
            format!("the register down does not take this bank over {inter} x {columns}"),
        ));
    }
    let [codes, scales, biases, own_codes, own_scales, own_biases] =
        register_banks(ctx, down, shared)?;
    ctx.fire(
        Fire::at(REGISTER_FILE, entry).apply(Grid::of(
            [32 * simdgroups * (columns / span), cut.slots(tiles), 1],
            [32 * simdgroups, 1, 1],
        )),
        &[
            codes,
            scales,
            biases,
            own_codes,
            own_scales,
            own_biases,
            rooms.hidden.arg(),
            rooms.y.arg_mut(),
            rooms.tiles.arg(),
            stated(OP, inter)?.arg(),
            stated(OP, columns)?.arg(),
            stated(OP, experts)?.arg(),
            rooms.rows.arg(),
        ],
    )
}

fn zero_points(ctx: &Ctx<'_>, bank: Bank) -> Result<crate::encode::ArgValue, Error> {
    match bank.biases {
        Some(biases) => Ok(biases.arg()),
        None => ctx.absent(),
    }
}

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

/// [`weighted_sum`] over routed rows a chain left sorted: route `at`'s row
/// is `sorted[inv[at]]`, so every token sums the same rows in the same order.
pub fn weighted_sum_sorted(
    ctx: &Ctx<'_>,
    sorted: Tensor,
    inv: Tensor,
    weights: Tensor,
    top_k: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_weighted_sum";
    dtype_dispatch!(OP, sorted.dtype, { Bf16 => () });
    nonzero(OP, "the token rows this fold lands on", y.rows)?;
    debug_assert_eq!(
        sorted.width, y.width,
        "the sorted rows are the routed width"
    );
    ctx.fire(
        Fire::at("linear/moe_route.metal", "expert_combine_sorted")
            .apply(route_rows(OP, y.width, y.rows)?),
        &[
            sorted.arg(),
            weights.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            stated(OP, top_k)?.arg(),
            inv.arg(),
        ],
    )
}

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

    #[test]
    fn an_mxfp4_bank_reaches_a_tiled_point_with_or_without_a_bias() {
        let bank = Bank {
            mpp_codes: None,
            codes: Tensor::new(0, 2880, 2880, Dtype::U4g64),
            scales: Tensor::new(1, 2880, 2880 / 32, Dtype::E8m0),
            biases: None,
            group: MXFP4_BLOCK,
            bits: 4,
            ld: 2880,
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
