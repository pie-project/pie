//! Routed matmuls a route at a time (`linear.moe_matmul_select`, its bias
//! and quantized forms, and `linear.matmul_grouped` over them) from
//! `kernels/linear/moe/select.slang`: eight output rows of one route a
//! workgroup. At `Tier::Subgroup` (WGSL) a strip folds by shuffles and the
//! route's activations sit in workgroup memory.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{
    Arg, ArgValue, Ctx, Fire, Target, Tier, dtype_dispatch, nonzero, refuse, stated,
};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

const SELECT: &str = "linear/moe/select.slang";

/// Output rows a workgroup lands.
const ROWS: u32 = 8;

/// The MXFP4 block, codes a scale.
const MXFP4_BLOCK: u32 = 32;

/// A routed fire's fan-out: tokens, routes a token, and how the activation
/// strides by token and by route (0: every route of a token reads one row).
pub(crate) struct Fan {
    pub tokens: u32,
    pub top_k: u32,
    pub x_row_stride: u32,
    pub x_slot_stride: u32,
}

pub(crate) fn fan(op: &'static str, x: Tensor, routes: Tensor, y: Tensor) -> Result<Fan, Error> {
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
    Ok(Fan {
        tokens: routes.rows,
        top_k: routes.width,
        x_row_stride,
        x_slot_stride,
    })
}

fn grid(op: &'static str, fan: &Fan, out_width: u32) -> Result<[u32; 3], Error> {
    let blocks = out_width.div_ceil(ROWS);
    if u64::from(fan.tokens) * u64::from(blocks) * u64::from(fan.top_k) > u64::from(u32::MAX) {
        return Err(refuse(
            op,
            format!(
                "the {} token rows x {blocks} output blocks x {} slots will not launch",
                fan.tokens, fan.top_k
            ),
        ));
    }
    Ok([fan.tokens, blocks, fan.top_k])
}

fn push(op: &'static str, x: Tensor, y: Tensor, fan: &Fan) -> Result<[ArgValue; 5], Error> {
    Ok([
        stated(op, x.width)?.arg(),
        stated(op, y.width)?.arg(),
        stated(op, fan.x_slot_stride)?.arg(),
        stated(op, fan.x_row_stride)?.arg(),
        stated(op, fan.top_k)?.arg(),
    ])
}

fn fire_at(entry: &'static str, tier: Tier, groups: [u32; 3]) -> Fire {
    Fire::at(SELECT, entry)
        .tier(tier)
        .groups(groups)
        .group([32, ROWS, 1])
}

/// The tier a select fires at: WGSL's subgroup tier where the engine has it,
/// else the baseline.
fn tier_of(ctx: &Ctx<'_>, tier: Tier) -> Tier {
    if ctx.target() == Target::Wgsl && tier == Tier::Subgroup {
        Tier::Subgroup
    } else {
        Tier::Baseline
    }
}

/// `y[route] = x[token] * bank[expert]^T` over a dense bf16 bank.
pub fn matmul_select(
    ctx: &Ctx<'_>,
    tier: Tier,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "linear_moe_select_dense_bf16" });
    debug_assert_eq!(bank.dtype, x.dtype, "the bank rides the activation's dtype");
    let fan = fan(OP, x, routes, y)?;
    if !x.width.is_multiple_of(4) {
        return Err(refuse(
            OP,
            format!(
                "K is {}, not a whole number of the four codes a lane walks",
                x.width
            ),
        ));
    }
    let push = push(OP, x, y, &fan)?;
    ctx.fire(
        fire_at(entry, tier_of(ctx, tier), grid(OP, &fan, y.width)?),
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

/// The instantiated shape a quantized bank runs at.
pub(crate) fn point(op: &'static str, bank: Bank, biased: bool) -> Result<&'static str, Error> {
    match (bank.biases.is_some(), bank.group, bank.bits) {
        (true, 64, 4) if biased => Ok("linear_moe_select_affine_bias_bf16_gs_64_b_4"),
        (true, 64, 4) => Ok("linear_moe_select_affine_bf16_gs_64_b_4"),
        (true, 64, 2) if biased => Ok("linear_moe_select_affine_bias_bf16_gs_64_b_2"),
        (true, 64, 2) => Ok("linear_moe_select_affine_bf16_gs_64_b_2"),
        (true, 32, 2) if biased => Ok("linear_moe_select_affine_bias_bf16_gs_32_b_2"),
        (true, 32, 2) => Ok("linear_moe_select_affine_bf16_gs_32_b_2"),
        (true, 128, 2) if biased => Ok("linear_moe_select_affine_bias_bf16_gs_128_b_2"),
        (true, 128, 2) => Ok("linear_moe_select_affine_bf16_gs_128_b_2"),
        (false, MXFP4_BLOCK, 4) if biased => Ok("linear_moe_select_mxfp4_bias_bf16_gs_32_b_4"),
        (false, MXFP4_BLOCK, 4) => Ok("linear_moe_select_mxfp4_bf16_gs_32_b_4"),
        (affine, group, bits) => Err(refuse(
            op,
            format!(
                "the bank is {} at {bits} bits in groups of {group}, and `linear/moe/select.slang` \
                 instantiates the routed shapes at affine/{{32,64,128}}/2, affine/64/4 and \
                 mxfp4/32/4 only",
                if affine { "affine" } else { "symmetric" }
            ),
        )),
    }
}

pub(crate) fn check_bank(op: &'static str, x: Tensor, bank: Bank) -> Result<(), Error> {
    if bank.biases.is_some() {
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

pub(crate) fn zero_points(ctx: &Ctx<'_>, bank: Bank) -> Result<ArgValue, Error> {
    match bank.biases {
        Some(biases) => Ok(biases.arg()),
        None => ctx.absent(),
    }
}

fn quantized(
    ctx: &Ctx<'_>,
    op: &'static str,
    tier: Tier,
    x: Tensor,
    bank: Bank,
    bias: Option<Tensor>,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    let entry = point(op, bank, bias.is_some())?;
    let fan = fan(op, x, routes, y)?;
    check_bank(op, x, bank)?;
    let push = push(op, x, y, &fan)?;
    ctx.fire(
        fire_at(entry, tier_of(ctx, tier), grid(op, &fan, y.width)?),
        &[
            bank.codes.arg(),
            bank.scales.arg(),
            zero_points(ctx, bank)?,
            x.arg(),
            y.arg_mut(),
            match bias {
                Some(bias) => bias.arg(),
                None => ctx.absent()?,
            },
            routes.arg(),
            push[0],
            push[1],
            push[2],
            push[3],
            push[4],
        ],
    )
}

/// `y[route] = x[token] * dequant(bank[expert])^T + bias[expert]`.
pub fn matmul_select_bias(
    ctx: &Ctx<'_>,
    tier: Tier,
    x: Tensor,
    bank: Bank,
    bias: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    debug_assert_eq!(
        bias.dtype, x.dtype,
        "the expert bias rides the activation's dtype"
    );
    quantized(
        ctx,
        "linear.moe_matmul_select_bias",
        tier,
        x,
        bank,
        Some(bias),
        routes,
        y,
    )
}

/// `y[route] = x[token] * dequant(bank[expert])^T`.
pub fn matmul_select_quant(
    ctx: &Ctx<'_>,
    tier: Tier,
    x: Tensor,
    bank: Bank,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    quantized(
        ctx,
        "linear.moe_matmul_select_quant",
        tier,
        x,
        bank,
        None,
        routes,
        y,
    )
}

/// A grouped plane: an expert bank, quantized or dense.
#[derive(Clone, Copy)]
pub enum GroupedPlane {
    Bank(Bank),
    Dense(Tensor),
}

/// `linear.matmul_grouped`: each row cut into `groups` slices, slice `g`
/// projected by the plane `routes[row, g]` names.
pub fn matmul_grouped(
    ctx: &Ctx<'_>,
    tier: Tier,
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
        GroupedPlane::Bank(bank) => matmul_select_quant(ctx, tier, x, bank, routes, y),
        GroupedPlane::Dense(dense) => matmul_select(ctx, tier, x, dense, routes, y),
    }
}
