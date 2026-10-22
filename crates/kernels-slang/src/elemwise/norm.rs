//! RMS norms (`elemwise.rmsnorm` and its weight and axis forms), from
//! `kernels/elemwise/norm.slang`: one workgroup a normed run.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const NORM: &str = "elemwise/norm.slang";

/// The workgroup width every rms instantiation is built at.
const GROUP: u32 = 256;

/// A weight read a column (`w_stride` 1).
const DENSE: u32 = 1;

fn grid(op: &'static str, width: u32, axis: u32, rows: u32) -> Result<Grid, Error> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    nonzero(op, "the normed axis", axis)?;
    if axis > width || !width.is_multiple_of(axis) {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {axis}-wide normed axes"),
        ));
    }
    if !axis.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("a {axis}-wide run is odd; rows land two values a word"),
        ));
    }
    let groups = (width / axis)
        .checked_mul(rows)
        .ok_or_else(|| refuse(op, format!("{rows} rows of {width} will not launch")))?;
    let lanes = groups
        .checked_mul(GROUP)
        .ok_or_else(|| refuse(op, format!("{groups} workgroups will not launch")))?;
    Ok(Grid::of([lanes, 1, 1], [GROUP, 1, 1]))
}

#[allow(clippy::too_many_arguments)]
fn rms(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    y: Tensor,
    eps: f32,
    axis: u32,
    w_stride: u32,
    plus_one: bool,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rmsnorm";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_rms_bf16" });
    ctx.fire(
        Fire::at(NORM, entry).apply(grid(OP, x.width, axis, x.rows)?),
        &[
            x.arg(),
            weight.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(OP, axis)?.arg(),
            w_stride.arg(),
            u32::from(plus_one).arg(),
            1.0f32.arg(),
        ],
    )
}

/// `y = x / rms(x) * weight` over the whole row.
pub fn rmsnorm(ctx: &Ctx<'_>, x: Tensor, weight: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    rms(ctx, x, weight, y, eps, x.width, DENSE, false)
}

/// `y = x / rms(x) * (1 + weight)` over the whole row.
pub fn rmsnorm_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms(ctx, x, weight, y, eps, x.width, DENSE, true)
}

/// Each `head_dim`-wide head normed alone, one `head_dim`-wide weight for all.
pub fn rmsnorm_per_head(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms(ctx, x, weight, y, eps, head_dim, DENSE, false)
}

/// As [`rmsnorm_per_head`], times `1 + weight`.
pub fn rmsnorm_per_head_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms(ctx, x, weight, y, eps, head_dim, DENSE, true)
}

/// Each `group`-wide run normed alone, times `1 + weight` from a bank as wide
/// as the row.
pub fn rmsnorm_grouped_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    group: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rmsnorm";
    let axis = nonzero(OP, "the group width", group)?;
    if !x.width.is_multiple_of(axis) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide row is not a whole number of {axis}-wide groups",
                x.width
            ),
        ));
    }
    let groups = x.width / axis;
    let bank = weight.width * weight.rows.max(1);
    if bank != x.width {
        return Err(refuse(
            OP,
            format!(
                "the weight bank is {bank} wide and the {groups} groups of {axis} span {}",
                x.width
            ),
        ));
    }
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_rms_grouped_bf16" });
    ctx.fire(
        Fire::at(NORM, entry).apply(grid(OP, x.width, axis, x.rows)?),
        &[
            x.arg(),
            weight.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(OP, axis)?.arg(),
            DENSE.arg(),
            1u32.arg(),
            1.0f32.arg(),
            stated(OP, groups)?.arg(),
        ],
    )
}

/// Each `head_dim`-wide head normed alone, unscaled.
pub fn rmsnorm_no_scale(
    ctx: &Ctx<'_>,
    x: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rmsnorm";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_rms_unscaled_bf16" });
    ctx.fire(
        Fire::at(NORM, entry).apply(grid(OP, x.width, head_dim, x.rows)?),
        &[x.arg(), y.arg_mut(), eps.arg(), stated(OP, head_dim)?.arg()],
    )
}
