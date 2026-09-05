#![allow(unused_variables)]
#![allow(clippy::too_many_arguments)]

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, elementwise_rows, nonzero, refuse, stated,
};
use crate::error::Error;
use crate::tensor::Tensor;
use dtype::Dtype;

const DENSE_BANK: u32 = 1;

const ABSOLUTE_BANK: u32 = 0;

const OFFSET_BANK: u32 = 1;

const UNIT_GAIN: f32 = 1.0;

const RMS_GROUP: u32 = 256;

const FLAT_GROUP: u32 = 256;

fn rms_grid(op: &'static str, width: u32, axis: u32, rows: u32) -> Result<Grid, Error> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    nonzero(op, "the normed axis", axis)?;
    if axis > width || !width.is_multiple_of(axis) {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {axis}-wide normed axes"),
        ));
    }
    let norms = width / axis;
    let groups = norms.checked_mul(rows).ok_or_else(|| {
        refuse(
            op,
            format!("the grid will not launch: {norms} norms per row x {rows} rows"),
        )
    })?;
    let lanes = groups
        .checked_mul(RMS_GROUP)
        .ok_or_else(|| refuse(op, format!("the grid will not launch: {groups} workgroups")))?;
    Ok(Grid::of([lanes, 1, 1], [RMS_GROUP, 1, 1]))
}

fn rms_row(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    weight: Tensor,
    y: Tensor,
    eps: f32,
    axis: u32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
) -> Result<(), Error> {
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "rms_single_row_bf16" });
    let grid = rms_grid(op, x.width, axis, x.rows)?;
    ctx.fire(
        Fire::at("norm/rms.slang", entry).apply(grid),
        &[
            x.arg(),
            weight.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(op, axis)?.arg(),
            w_stride.arg(),
            plus_one.arg(),
            gain.arg(),
        ],
    )
}

pub fn rmsnorm(ctx: &Ctx<'_>, x: Tensor, weight: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm",
        x,
        weight,
        y,
        eps,
        x.width,
        DENSE_BANK,
        ABSOLUTE_BANK,
        UNIT_GAIN,
    )
}

pub fn rmsnorm_per_head(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_per_head",
        x,
        weight,
        y,
        eps,
        head_dim,
        DENSE_BANK,
        ABSOLUTE_BANK,
        UNIT_GAIN,
    )
}

pub fn rmsnorm_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_plus_one",
        x,
        weight,
        y,
        eps,
        x.width,
        DENSE_BANK,
        OFFSET_BANK,
        UNIT_GAIN,
    )
}

pub fn rmsnorm_per_head_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_per_head_plus_one",
        x,
        weight,
        y,
        eps,
        head_dim,
        DENSE_BANK,
        OFFSET_BANK,
        UNIT_GAIN,
    )
}

pub fn rmsnorm_grouped_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    group: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_grouped_plus_one";
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
                "the weight bank is {bank} wide and the {groups} groups of {axis} it gains span {}",
                x.width
            ),
        ));
    }
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "rms_grouped_row_bf16" });
    let grid = rms_grid(OP, x.width, axis, x.rows)?;
    ctx.fire(
        Fire::at("norm/rms.slang", entry).apply(grid),
        &[
            x.arg(),
            weight.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(OP, axis)?.arg(),
            DENSE_BANK.arg(),
            OFFSET_BANK.arg(),
            UNIT_GAIN.arg(),
            stated(OP, groups)?.arg(),
        ],
    )
}

pub fn rmsnorm_no_scale(
    ctx: &Ctx<'_>,
    x: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_no_scale";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "vnorm_single_row_bf16" });
    let grid = rms_grid(OP, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at("norm/vector.slang", entry).apply(grid),
        &[x.arg(), y.arg_mut(), eps.arg(), stated(OP, head_dim)?.arg()],
    )
}

fn gated_rms(
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
    debug_assert!(
        gate.rows == x.rows && gate.width == x.width && y.rows == x.rows && y.width == x.width,
        "the gate and the landing ride the normed rectangle"
    );
    let grid = rms_grid(op, x.width, vd, x.rows)?;
    ctx.fire(
        Fire::at("norm/gated_rms.slang", entry).apply(grid),
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
    const OP: &str = "elementwise.rmsnorm_gated";

    let entry = if sigmoid_gate {
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_sigmoid_f32_bf16" })
    } else {
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_f32_bf16" })
    };
    let vd = nonzero(OP, "the stated value-head width", head_dim)?;
    gated_rms(ctx, OP, entry, x, gate, weight, vd, eps, y)
}

pub fn rmsnorm_gated_by(
    ctx: &Ctx<'_>,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    heads: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_gated_by";
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_sigmoid_f32_bf16" });
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
    gated_rms(ctx, OP, entry, x, gate, weight, x.width / heads, eps, y)
}

pub fn residual_add(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "residual_add_bf16" });
    let lanes = elementwise(OP, y.width, y.rows)?;
    ctx.fire(
        Fire::at("norm/residual_add.slang", entry).apply(Grid::of(lanes, [FLAT_GROUP, 1, 1])),
        &[x.arg(), y.arg(), y.arg_mut(), lanes[0].arg()],
    )
}

pub fn add_bias(ctx: &Ctx<'_>, bias: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add_bias";
    let entry = dtype_dispatch!(OP, out.dtype, { Bf16 => "add_bias_bf16" });
    let lanes = elementwise_rows(OP, out.width, out.rows)?;
    ctx.fire(
        Fire::at("norm/add_bias.slang", entry).apply(Grid::of(lanes, [FLAT_GROUP, 1, 1])),
        &[out.arg_mut(), bias.arg(), stated(OP, out.width)?.arg()],
    )
}

pub fn layernorm(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.layernorm";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "layernorm_bf16" });
    let grid = rms_grid(OP, x.width, x.width, x.rows)?;
    ctx.fire(
        Fire::at("norm/layernorm.slang", entry).apply(grid),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg_mut(),
            eps.arg(),
            x.width.arg(),
        ],
    )
}

pub fn standardize(ctx: &Ctx<'_>, bias: Tensor, scale: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.standardize";
    let entry = dtype_dispatch!(OP, out.dtype, { Bf16 => "standardize_bf16" });
    let lanes = elementwise_rows(OP, out.width, out.rows)?;
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
        if u64::from(plane.rows) * u64::from(plane.width) != u64::from(out.width) {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is a {} x {} plane, and this row reads one \
                     scalar per column of a {}-wide rectangle",
                    plane.rows, plane.width, out.width
                ),
            ));
        }
    }
    ctx.fire(
        Fire::at("norm/standardize.slang", entry).apply(Grid::of(lanes, [FLAT_GROUP, 1, 1])),
        &[
            out.arg_mut(),
            bias.arg(),
            scale.arg(),
            stated(OP, out.width)?.arg(),
            stated(OP, out.rows)?.arg(),
        ],
    )
}

pub fn mul_scalar(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.mul_scalar";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "layer_scalar_mul_stated_bf16" });
    let lanes = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at("norm/layer_scalar.slang", entry).apply(Grid::of(lanes, [FLAT_GROUP, 1, 1])),
        &[x.arg(), x.arg_mut(), s.arg(), lanes[0].arg()],
    )
}

pub fn silu_scaled(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.silu_scaled";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "silu_scaled_bf16" });
    let lanes = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at("norm/layer_scalar.slang", entry).apply(Grid::of(lanes, [FLAT_GROUP, 1, 1])),
        &[x.arg(), x.arg_mut(), s.arg(), lanes[0].arg()],
    )
}

pub fn scale(ctx: &Ctx<'_>, s: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.scale";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "layer_scalar_mul_bf16" });
    let lanes = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at("norm/layer_scalar.slang", entry).apply(Grid::of(lanes, [FLAT_GROUP, 1, 1])),
        &[x.arg(), s.arg(), x.arg_mut(), lanes[0].arg()],
    )
}

pub const MAX_BLEND_BLOCKS: usize = 32;

pub fn res_blend(
    ctx: &Ctx<'_>,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.res_blend";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "res_blend_bf16" });
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
    let n_blocks = u32::try_from(blocks.len()).unwrap_or(u32::MAX);

    let lanes = rows
        .checked_mul(RMS_GROUP)
        .ok_or_else(|| refuse(OP, format!("the grid will not launch: {rows} rows")))?;

    let stack = blocks.first().copied().unwrap_or(prefix);
    ctx.fire(
        Fire::at("norm/res_blend.slang", entry).apply(Grid::of([lanes, 1, 1], [RMS_GROUP, 1, 1])),
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
