#![allow(unused_variables)]
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

const MOE_TILE_ROWS: [u32; 3] = [16, 32, 64];

const MOE_TILE_COLS: [u32; 3] = [32, 64, 128];

const ROUTER_MAX_EXPERTS: u32 = 1024;

const ROUTER_MAX_TOP_K: u32 = 16;

const MXFP4_BLOCK: u32 = 32;

const SOFTMAX_OVER_SELECTED: u32 = 0;

const ROUTE_FILE: &str = "moe/route.slang";

const QMV_FILE: &str = "moe/qmv_routed.slang";

const ROUTE_GROUP: [u32; 3] = [256, 1, 1];

const QMV_GROUP: [u32; 3] = [32, 8, 1];

#[must_use]
pub fn should_batch(pairs: u32, experts: u32, min_per_expert: u32) -> bool {
    experts > 0 && u64::from(pairs) >= u64::from(experts) * u64::from(min_per_expert)
}

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

#[must_use]
pub fn tile_cols(out_width: u32) -> Option<u32> {
    MOE_TILE_COLS
        .iter()
        .rev()
        .copied()
        .find(|tile| out_width.is_multiple_of(*tile))
}

fn router_grid(op: &'static str, rows: u32) -> Result<Grid, Error> {
    nonzero(op, "rows", rows)?;
    Ok(Grid::of([ROUTE_GROUP[0], rows, 1], ROUTE_GROUP))
}

fn route_rows(op: &'static str, width: u32, rows: u32) -> Result<Grid, Error> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    Ok(Grid::of([width, rows, 1], ROUTE_GROUP))
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
    nonzero(op, "the expert count this router states", experts)?;
    nonzero(op, "the fan-out this router states", top_k)?;
    if experts > ROUTER_MAX_EXPERTS {
        return Err(refuse(
            op,
            format!(
                "the expert count is {experts}, above the {ROUTER_MAX_EXPERTS} this router \
                 stages in shared memory"
            ),
        ));
    }
    if top_k > ROUTER_MAX_TOP_K {
        return Err(refuse(
            op,
            format!(
                "the fan-out is {top_k}, above the {ROUTER_MAX_TOP_K} this router stages in \
                 a shared array"
            ),
        ));
    }
    router_grid(op, logits.rows)
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
    let grid = ranked(OP, logits, experts, top_k, routes, weights)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, entry).apply(grid),
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
        Fire::at(ROUTE_FILE, entry).apply(grid),
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
        Fire::at(ROUTE_FILE, entry).apply(grid),
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

fn biased_router(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    debug_assert_eq!(
        bias.dtype,
        Dtype::F32,
        "`{op}` reads an f32 correction bias"
    );
    let grid = ranked(op, logits, experts, top_k, routes, weights)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, entry).apply(grid),
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
    dtype_dispatch!(OP, logits.dtype, { Bf16 => () });
    biased_router(
        ctx,
        OP,
        "router_topk_sigmoid_biased",
        logits,
        bias,
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

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
    dtype_dispatch!(OP, logits.dtype, { Bf16 => () });
    biased_router(
        ctx,
        OP,
        "router_topk_sqrt_softplus",
        logits,
        bias,
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
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
    dtype_dispatch!(OP, logits.dtype, { Bf16 => () });
    biased_router(
        ctx,
        OP,
        "router_topk_sqrt_softplus",
        logits,
        bias,
        experts,
        top_k,
        false,
        1.0,
        routes,
        weights,
    )
}

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
    let rows = nonzero(OP, "rows", routes.rows)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "hash_route_gather").apply(Grid::of([rows, 1, 1], ROUTE_GROUP)),
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
            rows.arg(),
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
                "{groups} groups do not divide a {}-wide row into a {}-wide one, or the routes are \
                 {} wide",
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

fn routed_qmv_grid(op: &'static str, fan: &Selected, out_width: u32) -> Result<Fire, Error> {
    let blocks = out_width.div_ceil(QMV_GROUP[1]);
    if u64::from(fan.tokens) * u64::from(blocks) * u64::from(fan.top_k) > u64::from(u32::MAX) {
        return Err(refuse(
            op,
            format!(
                "the {} token rows x {blocks} output blocks x {} slots will not launch",
                fan.tokens, fan.top_k
            ),
        ));
    }
    Ok(Fire::at(QMV_FILE, "")
        .groups([fan.tokens, blocks, fan.top_k])
        .group(QMV_GROUP))
}

fn routed_push(
    op: &'static str,
    x: Tensor,
    y: Tensor,
    fan: &Selected,
) -> Result<[ArgValue; 5], Error> {
    Ok([
        stated(op, x.width)?.arg(),
        stated(op, y.width)?.arg(),
        stated(op, fan.x_slot_stride)?.arg(),
        stated(op, fan.x_row_stride)?.arg(),
        stated(op, fan.top_k)?.arg(),
    ])
}

pub fn matmul_select(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "select_gemv_bfloat16" });
    debug_assert_eq!(bank.dtype, x.dtype, "the bank rides the activation's dtype");
    let fan = selected(OP, x, routes, y)?;
    if !x.width.is_multiple_of(4) {
        return Err(refuse(
            OP,
            format!(
                "K is {}, not a whole number of the four codes a lane walks",
                x.width
            ),
        ));
    }
    let fire = routed_qmv_grid(OP, &fan, y.width)?;
    let push = routed_push(OP, x, y, &fan)?;
    ctx.fire(
        Fire {
            entrypoint: entry,
            ..fire
        },
        &[
            bank.arg(),
            x.arg(),
            y.arg_mut(),
            routes.arg(),
            push[0],
            push[1],
            push[2],
            push[3],
            push[4],
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
                "the bank is {} at {bits} bits in groups of {group}, and `moe/qmv_routed.slang` \
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

fn zero_points(ctx: &Ctx<'_>, bank: Bank) -> Result<ArgValue, Error> {
    match bank.biases {
        Some(biases) => Ok(biases.arg()),
        None => ctx.absent(),
    }
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
    let fire = routed_qmv_grid(OP, &fan, y.width)?;
    let push = routed_push(OP, x, y, &fan)?;
    ctx.fire(
        Fire {
            entrypoint: entry,
            ..fire
        },
        &[
            bank.codes.arg(),
            bank.scales.arg(),
            zero_points(ctx, bank)?,
            x.arg(),
            y.arg_mut(),
            bias.arg(),
            routes.arg(),
            push[0],
            push[1],
            push[2],
            push[3],
            push[4],
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
    let fire = routed_qmv_grid(OP, &fan, y.width)?;
    let push = routed_push(OP, x, y, &fan)?;
    ctx.fire(
        Fire {
            entrypoint: entry,
            ..fire
        },
        &[
            bank.codes.arg(),
            bank.scales.arg(),
            zero_points(ctx, bank)?,
            x.arg(),
            y.arg_mut(),
            ctx.absent()?,
            routes.arg(),
            push[0],
            push[1],
            push[2],
            push[3],
            push[4],
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
}

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
    if tile <= 1 || !x.width.is_multiple_of(32) || experts > 1024 {
        return Ok(false);
    }
    let coopmat = crate::tuning::device().coopmat;
    let family = match (bank.affine(), bank.group, bank.bits) {
        (false, 32, 4) => "mxfp4",
        (true, 64, 4) | (true, 64, 2) | (true, 32, 2) | (true, 128, 2) => "affine",
        _ => return Ok(false),
    };
    let Some(entry) = ROUTED_TILE_POINTS
        .iter()
        .copied()
        .find(|(name, f, g, b, biased, bm, cm)| {
            *f == family
                && *g == bank.group
                && *b == bank.bits
                && *biased == bias.is_some()
                && *bm == tile
                && *cm == coopmat
                && !name.is_empty()
        })
    else {
        return Ok(false);
    };
    let padded = sorted_rows(pairs, experts, tuning);
    if scratch.x.rows < padded || scratch.y.rows < padded {
        return Ok(false);
    }

    let (gather_fan, pitch) = if fan.x_slot_stride == 0 {
        (fan.top_k, fan.x_row_stride)
    } else if fan.x_slot_stride == x.width && fan.x_row_stride == x.width * fan.top_k {
        (1, x.width)
    } else {
        return Ok(false);
    };
    ctx.fire(
        Fire::at(SORT_FILE, "route_sort").apply(Grid::of([1024, 1, 1], [1024, 1, 1])),
        &[
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
        ],
    )?;
    ctx.fire(
        Fire::at(SORT_FILE, "route_gather").apply(route_rows(op, x.width, padded)?),
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
            pitch.arg(),
        ],
    )?;
    let bn: u32 = if coopmat { 64 } else { 32 };
    ctx.fire(
        Fire::at(ROUTED_TILE_FILE, entry.0).apply(Grid::of(
            [y.width.div_ceil(bn) * 32, (padded / tile) * 4, 1],
            [32, 4, 1],
        )),
        &[
            bank.codes.arg(),
            bank.scales.arg(),
            zero_points(ctx, bank)?,
            scratch.x.arg(),
            scratch.y.arg_mut(),
            match bias {
                Some(bias) => bias.arg(),
                None => ctx.absent()?,
            },
            scratch.tile_expert.arg(),
            stated(op, x.width)?.arg(),
            stated(op, y.width)?.arg(),
        ],
    )?;
    ctx.fire(
        Fire::at(SORT_FILE, "route_scatter").apply(route_rows(op, y.width, pairs)?),
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

const SORT_FILE: &str = "moe/route_sort.slang";
const ROUTED_TILE_FILE: &str = "moe/qmm_t_routed.slang";

const ROUTED_TILE_POINTS: &[(&str, &str, u32, u32, bool, u32, bool)] = &[
    (
        "mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_16",
        "mxfp4",
        32,
        4,
        false,
        16,
        false,
    ),
    (
        "mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_16_cm",
        "mxfp4",
        32,
        4,
        false,
        16,
        true,
    ),
    (
        "mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_32",
        "mxfp4",
        32,
        4,
        false,
        32,
        false,
    ),
    (
        "mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_32_cm",
        "mxfp4",
        32,
        4,
        false,
        32,
        true,
    ),
    (
        "mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_64",
        "mxfp4",
        32,
        4,
        false,
        64,
        false,
    ),
    (
        "mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_64_cm",
        "mxfp4",
        32,
        4,
        false,
        64,
        true,
    ),
    (
        "mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_16",
        "mxfp4",
        32,
        4,
        true,
        16,
        false,
    ),
    (
        "mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_16_cm",
        "mxfp4",
        32,
        4,
        true,
        16,
        true,
    ),
    (
        "mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_32",
        "mxfp4",
        32,
        4,
        true,
        32,
        false,
    ),
    (
        "mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_32_cm",
        "mxfp4",
        32,
        4,
        true,
        32,
        true,
    ),
    (
        "mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_64",
        "mxfp4",
        32,
        4,
        true,
        64,
        false,
    ),
    (
        "mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_64_cm",
        "mxfp4",
        32,
        4,
        true,
        64,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_4_bm_16",
        "affine",
        64,
        4,
        false,
        16,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_4_bm_16_cm",
        "affine",
        64,
        4,
        false,
        16,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_4_bm_32",
        "affine",
        64,
        4,
        false,
        32,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_4_bm_32_cm",
        "affine",
        64,
        4,
        false,
        32,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_4_bm_64",
        "affine",
        64,
        4,
        false,
        64,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_4_bm_64_cm",
        "affine",
        64,
        4,
        false,
        64,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_16",
        "affine",
        64,
        4,
        true,
        16,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_16_cm",
        "affine",
        64,
        4,
        true,
        16,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_32",
        "affine",
        64,
        4,
        true,
        32,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_32_cm",
        "affine",
        64,
        4,
        true,
        32,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_64",
        "affine",
        64,
        4,
        true,
        64,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_64_cm",
        "affine",
        64,
        4,
        true,
        64,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_2_bm_16",
        "affine",
        64,
        2,
        false,
        16,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_2_bm_16_cm",
        "affine",
        64,
        2,
        false,
        16,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_2_bm_32",
        "affine",
        64,
        2,
        false,
        32,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_2_bm_32_cm",
        "affine",
        64,
        2,
        false,
        32,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_2_bm_64",
        "affine",
        64,
        2,
        false,
        64,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_64_b_2_bm_64_cm",
        "affine",
        64,
        2,
        false,
        64,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_16",
        "affine",
        64,
        2,
        true,
        16,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_16_cm",
        "affine",
        64,
        2,
        true,
        16,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_32",
        "affine",
        64,
        2,
        true,
        32,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_32_cm",
        "affine",
        64,
        2,
        true,
        32,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_64",
        "affine",
        64,
        2,
        true,
        64,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_64_cm",
        "affine",
        64,
        2,
        true,
        64,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_32_b_2_bm_16",
        "affine",
        32,
        2,
        false,
        16,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_32_b_2_bm_16_cm",
        "affine",
        32,
        2,
        false,
        16,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_32_b_2_bm_32",
        "affine",
        32,
        2,
        false,
        32,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_32_b_2_bm_32_cm",
        "affine",
        32,
        2,
        false,
        32,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_32_b_2_bm_64",
        "affine",
        32,
        2,
        false,
        64,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_32_b_2_bm_64_cm",
        "affine",
        32,
        2,
        false,
        64,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_16",
        "affine",
        32,
        2,
        true,
        16,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_16_cm",
        "affine",
        32,
        2,
        true,
        16,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_32",
        "affine",
        32,
        2,
        true,
        32,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_32_cm",
        "affine",
        32,
        2,
        true,
        32,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_64",
        "affine",
        32,
        2,
        true,
        64,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_64_cm",
        "affine",
        32,
        2,
        true,
        64,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_128_b_2_bm_16",
        "affine",
        128,
        2,
        false,
        16,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_128_b_2_bm_16_cm",
        "affine",
        128,
        2,
        false,
        16,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_128_b_2_bm_32",
        "affine",
        128,
        2,
        false,
        32,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_128_b_2_bm_32_cm",
        "affine",
        128,
        2,
        false,
        32,
        true,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_128_b_2_bm_64",
        "affine",
        128,
        2,
        false,
        64,
        false,
    ),
    (
        "affine_qmm_t_routed_bf16_gs_128_b_2_bm_64_cm",
        "affine",
        128,
        2,
        false,
        64,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_16",
        "affine",
        128,
        2,
        true,
        16,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_16_cm",
        "affine",
        128,
        2,
        true,
        16,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_32",
        "affine",
        128,
        2,
        true,
        32,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_32_cm",
        "affine",
        128,
        2,
        true,
        32,
        true,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_64",
        "affine",
        128,
        2,
        true,
        64,
        false,
    ),
    (
        "affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_64_cm",
        "affine",
        128,
        2,
        true,
        64,
        true,
    ),
];

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
        Fire::at(ROUTE_FILE, entry).apply(grid),
        &[
            routed.arg(),
            weights.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            stated(OP, top_k)?.arg(),
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
        Fire::at(ROUTE_FILE, entry).apply(grid),
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
        "`shared_expert_combine` reads `gate[row]`, so the gate is a column"
    );
    let grid = route_rows(OP, routed.width, routed.rows)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, entry).apply(grid),
        &[
            routed.arg(),
            shared.arg(),
            gate.arg(),
            y.arg_mut(),
            stated(OP, routed.width)?.arg(),
        ],
    )
}
