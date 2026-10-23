//! The norms from `kernels/elemwise/norm.slang`, one workgroup a normed
//! run: RMS (`elemwise.rmsnorm` and its weight and axis forms, unscaled),
//! gated RMS (`rmsnorm_gated`, `rmsnorm_gated_by`), layer norm with and
//! without its affine, the residual blend (`res_blend`) and the per-column
//! `standardize`.

use dtype::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, nonzero, refuse, stated,
};
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
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_norm_rms_bf16" });
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
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_norm_rms_grouped_bf16" });
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
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_norm_rms_unscaled_bf16" });
    ctx.fire(
        Fire::at(NORM, entry).apply(grid(OP, x.width, head_dim, x.rows)?),
        &[x.arg(), y.arg_mut(), eps.arg(), stated(OP, head_dim)?.arg()],
    )
}

#[allow(clippy::too_many_arguments)]
fn gated(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    vd: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    debug_assert_eq!(x.dtype, Dtype::F32, "`{op}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{op}` scales by an f32 weight");
    nonzero(op, "rows", x.rows)?;
    if x.width == 0 || !x.width.is_multiple_of(vd) {
        return Err(refuse(
            op,
            format!(
                "the {}-wide row is not a whole number of {vd}-wide heads",
                x.width
            ),
        ));
    }
    if !vd.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("a {vd}-wide head is odd; the landing is two values a word"),
        ));
    }
    let groups = (x.width / vd)
        .checked_mul(x.rows)
        .and_then(|g| g.checked_mul(GROUP))
        .ok_or_else(|| refuse(op, "the grid will not launch"))?;
    ctx.fire(
        Fire::at(NORM, entry).apply(Grid::of([groups, 1, 1], [GROUP, 1, 1])),
        &[
            x.arg(),
            gate.arg(),
            weight.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(op, vd)?.arg(),
        ],
    )
}

/// `y = x / rms(x) * weight * gate`, each `head_dim`-wide head normed alone;
/// the gate is `z * sigmoid(z)`, or `sigmoid(z)` with `sigmoid_gate`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_gated(
    ctx: &Ctx<'_>,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    sigmoid_gate: bool,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rmsnorm_gated";
    let entry = if sigmoid_gate {
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "elemwise_norm_gated_rms_sigmoid_bf16" })
    } else {
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "elemwise_norm_gated_rms_bf16" })
    };
    let vd = nonzero(OP, "the stated value-head width", head_dim)?;
    gated(ctx, OP, entry, x, gate, weight, vd, eps, y)
}

/// The row normed in `heads` equal heads, each gated by `sigmoid(z)`.
pub fn rmsnorm_gated_by(
    ctx: &Ctx<'_>,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    heads: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rmsnorm_gated_by";
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "elemwise_norm_gated_rms_sigmoid_bf16" });
    nonzero(OP, "the stated head count", heads)?;
    if x.width == 0 || !x.width.is_multiple_of(heads) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide normed row does not divide by the stated head count {heads}",
                x.width
            ),
        ));
    }
    gated(ctx, OP, entry, x, gate, weight, x.width / heads, eps, y)
}

fn row_grid(op: &'static str, x: Tensor) -> Result<Grid, Error> {
    nonzero(op, "width", x.width)?;
    let rows = nonzero(op, "rows", x.rows)?;
    if x.dtype == Dtype::Bf16 && !x.width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "a {}-wide bf16 row is odd; rows land two values a word",
                x.width
            ),
        ));
    }
    let lanes = rows
        .checked_mul(GROUP)
        .ok_or_else(|| refuse(op, format!("{rows} rows will not launch")))?;
    Ok(Grid::of([lanes, 1, 1], [GROUP, 1, 1]))
}

/// `y = (x - mean) / sqrt(var + eps) * weight + bias` over each row.
pub fn layernorm(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.layernorm";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_norm_layernorm_bf16" });
    ctx.fire(
        Fire::at(NORM, entry).apply(row_grid(OP, x)?),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(OP, x.width)?.arg(),
        ],
    )
}

/// Each row standardized, with no weight and no bias.
pub fn layernorm_no_scale(ctx: &Ctx<'_>, x: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.layernorm";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_norm_layernorm_unscaled_bf16",
        F32 => "elemwise_norm_layernorm_unscaled_f32",
    });
    ctx.fire(
        Fire::at(NORM, entry).apply(row_grid(OP, x)?),
        &[x.arg(), y.arg_mut(), eps.arg(), stated(OP, x.width)?.arg()],
    )
}

/// The most candidate blocks the shader's softmax scratch holds.
pub const MAX_BLEND_BLOCKS: usize = 32;

/// `y` blended from `prefix` and `blocks`, which the caller lays as stacked
/// planes (`blocks[0]` the first, each `y.rows x y.width` after it).
pub fn res_blend(
    ctx: &Ctx<'_>,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.res_blend";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "elemwise_norm_res_blend_bf16" });
    if blocks.len() > MAX_BLEND_BLOCKS {
        return Err(refuse(
            OP,
            format!(
                "{} candidate blocks exceed the shader's softmax scratch bound of \
                 {MAX_BLEND_BLOCKS}",
                blocks.len()
            ),
        ));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    let hidden = nonzero(OP, "the blended row's width", y.width)?;
    if !hidden.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {hidden}-wide row is odd; rows land two values a word"),
        ));
    }
    let n_blocks = u32::try_from(blocks.len()).unwrap_or(u32::MAX);
    let lanes = rows
        .checked_mul(GROUP)
        .ok_or_else(|| refuse(OP, format!("the grid will not launch: {rows} rows")))?;
    let stack = blocks.first().copied().unwrap_or(prefix);
    ctx.fire(
        Fire::at(NORM, entry).apply(Grid::of([lanes, 1, 1], [GROUP, 1, 1])),
        &[
            prefix.arg(),
            stack.arg(),
            weight.arg(),
            proj.arg(),
            y.arg_mut(),
            stated(OP, n_blocks)?.arg(),
            stated(OP, hidden)?.arg(),
            stated(OP, rows)?.arg(),
            eps.arg(),
        ],
    )
}

pub fn standardize(ctx: &Ctx<'_>, bias: Tensor, scale: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.standardize";
    let entry = dtype_dispatch!(OP, out.dtype, { Bf16 => "elemwise_norm_standardize_bf16" });
    let [width, rows, _] = elementwise_rows(OP, out.width, out.rows)?;
    if !width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {width}-wide bf16 row is odd; rows land two values a word"),
        ));
    }
    for (what, plane) in [("bias", bias), ("scale", scale)] {
        if plane.dtype != out.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is {:?} and the rows it standardizes are {:?}; \
                     both planes ride the activation's element",
                    plane.dtype, out.dtype
                ),
            ));
        }
        if u64::from(plane.rows) * u64::from(plane.width) != u64::from(width) {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is a {} x {} plane, and this row reads one \
                     scalar per column of a {width}-wide rectangle",
                    plane.rows, plane.width
                ),
            ));
        }
    }
    ctx.fire(
        Fire::at(NORM, entry).apply(Grid::of([width.div_ceil(2), rows, 1], [GROUP, 1, 1])),
        &[
            out.arg_mut(),
            bias.arg(),
            scale.arg(),
            stated(OP, width)?.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}
