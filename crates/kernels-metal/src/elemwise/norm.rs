use crate::error::Error;
use dtype::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, elementwise_rows, nonzero, refuse, stated,
};
use crate::tensor::Tensor;

const DENSE_BANK: u32 = 1;

const ABSOLUTE_BANK: u32 = 0;

const OFFSET_BANK: u32 = 1;

const UNIT_GAIN: f32 = 1.0;

fn rms_threads(op: &'static str, axis: u32) -> Result<u32, Error> {
    nonzero(op, "the normed axis", axis)?;
    Ok(axis.div_ceil(4).min(1024))
}

fn rms_grid(op: &'static str, width: u32, axis: u32, rows: u32) -> Result<Grid, Error> {
    nonzero(op, "width", width)?;
    nonzero(op, "rows", rows)?;
    if axis > width {
        return Err(refuse(
            op,
            format!("the normed axis is {axis}, wider than the {width}-wide row"),
        ));
    }
    let t = rms_threads(op, axis)?;
    let norms = width / axis;
    let lanes = t
        .checked_mul(norms)
        .and_then(|n| n.checked_mul(rows))
        .ok_or_else(|| {
            refuse(
                op,
                format!(
                    "the grid will not launch: {t} axis threads x {norms} norms per row x {rows} rows"
                ),
            )
        })?;
    Ok(Grid::of([lanes, 1, 1], [t, 1, 1]))
}

fn head_row_grid(op: &'static str, threads: u32, heads: u32, rows: u32) -> Result<[u32; 3], Error> {
    Ok([
        threads,
        nonzero(op, "heads", heads)?,
        nonzero(op, "rows", rows)?,
    ])
}

fn head_width(op: &'static str, vd: u32) -> Result<u32, Error> {
    nonzero(op, "the value-head width", vd)?;
    if vd > 1024 {
        return Err(refuse(
            op,
            format!("the value-head width is {vd}, above the 1024 threads a group holds"),
        ));
    }
    Ok(vd)
}

#[allow(clippy::too_many_arguments)]
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
    let entry = dtype_dispatch!(op, x.dtype, {
        Bf16 => "rms_single_row_bfloat16",
        F32 => "rms_single_row_float32",
    });
    let grid = rms_grid(op, x.width, axis, x.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_rms.metal", entry).apply(grid),
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

pub fn residual_add_rmsnorm(
    ctx: &Ctx<'_>,
    x: Tensor,
    y: Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    out: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add_rmsnorm";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "residual_add_rms_single_row_bfloat16" });
    debug_assert!(
        x.rows == y.rows && x.width == y.width && out.rows == y.rows && out.width == y.width,
        "`{OP}` folds and norms one rectangle"
    );
    let grid = rms_grid(OP, y.width, y.width, y.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_rms.metal", entry).apply(grid),
        &[
            x.arg(),
            y.arg_mut(),
            weight.arg(),
            out.arg_mut(),
            eps.arg(),
            stated(OP, y.width)?.arg(),
            DENSE_BANK.arg(),
            u32::from(plus_one).arg(),
            UNIT_GAIN.arg(),
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
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "rms_grouped_row_bfloat16" });
    let grid = rms_grid(OP, x.width, axis, x.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_rms.metal", entry).apply(grid),
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
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "vnorm_single_row_bfloat16" });
    let grid = rms_grid(OP, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_vector.metal", entry).apply(grid),
        &[x.arg(), y.arg_mut(), eps.arg(), stated(OP, head_dim)?.arg()],
    )
}

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
    const OP: &str = "elementwise.rmsnorm_gated";
    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    let entry = if sigmoid_gate {
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_sigmoid_f32_bfloat16" })
    } else {
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_f32_bfloat16" })
    };
    let vd = head_width(OP, head_dim)?;
    if x.width == 0 || !x.width.is_multiple_of(vd) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide normed row does not divide by the stated value-head width {vd}",
                x.width
            ),
        ));
    }
    let lanes = head_row_grid(OP, vd, x.width / vd, x.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_gated_rms.metal", entry).apply(Grid::of(lanes, [vd, 1, 1])),
        &[
            x.arg(),
            gate.arg(),
            weight.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(OP, vd)?.arg(),
        ],
    )
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
    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_sigmoid_f32_bfloat16" });
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
    let vd = head_width(OP, x.width / heads)?;
    let lanes = head_row_grid(OP, vd, heads, x.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_gated_rms.metal", entry).apply(Grid::of(lanes, [vd, 1, 1])),
        &[
            x.arg(),
            gate.arg(),
            weight.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(OP, vd)?.arg(),
        ],
    )
}

pub fn residual_add(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add";
    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "residual_add_bfloat16",
        F32 => "residual_add_float32",
    });
    ctx.fire(
        Fire::at("elemwise/norm_residual_add.metal", entry)
            .apply(Grid::of(elementwise(OP, y.width, y.rows)?, [256, 1, 1])),
        &[x.arg(), y.arg(), y.arg_mut()],
    )
}

pub fn add_bias(ctx: &Ctx<'_>, bias: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add_bias";
    let entry = match (out.dtype, bias.dtype) {
        (Dtype::Bf16, Dtype::Bf16) => "add_bias_bfloat16",
        (Dtype::F32, Dtype::F32) => "add_bias_float32",
        (Dtype::F32, Dtype::Bf16) => "add_bias_float32_bf16",
        (row, bank) => {
            return Err(refuse(
                OP,
                format!("no point adds a {bank:?} bias to a {row:?} row"),
            ));
        }
    };
    let lanes = elementwise_rows(OP, out.width, out.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_add_bias.metal", entry)
            .apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
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
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "layernorm_bfloat16" });
    let grid = rms_grid(OP, x.width, x.width, x.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_layernorm.metal", entry).apply(grid),
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

pub fn layernorm_no_scale(ctx: &Ctx<'_>, x: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.layernorm_no_scale";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "layernorm_no_scale_bfloat16",
        F32 => "layernorm_no_scale_float32",
    });
    debug_assert!(
        x.rows == y.rows && x.width == y.width,
        "`{OP}` writes the rectangle it reads"
    );
    let grid = rms_grid(OP, x.width, x.width, x.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_layernorm.metal", entry).apply(grid),
        &[x.arg(), y.arg_mut(), eps.arg(), x.width.arg()],
    )
}

pub fn standardize(ctx: &Ctx<'_>, bias: Tensor, scale: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.standardize";
    let entry = dtype_dispatch!(OP, out.dtype, { Bf16 => "standardize_bfloat16" });
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
        Fire::at("elemwise/norm_standardize.metal", entry)
            .apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
        &[
            out.arg_mut(),
            bias.arg(),
            scale.arg(),
            stated(OP, out.width)?.arg(),
        ],
    )
}

pub fn mul_scalar(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.mul_scalar";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "layer_scalar_mul_stated_bfloat16",
        F32 => "layer_scalar_mul_stated_float32",
    });
    ctx.fire(
        Fire::at("elemwise/norm_layer_scalar.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [256, 1, 1])),
        &[x.arg(), s.arg(), x.arg_mut()],
    )
}

pub fn silu_scaled(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.silu_scaled";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "silu_scaled_bfloat16",
        F32 => "silu_scaled_float32",
    });
    ctx.fire(
        Fire::at("elemwise/norm_layer_scalar.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [256, 1, 1])),
        &[x.arg_mut(), s.arg()],
    )
}

pub fn scale(ctx: &Ctx<'_>, s: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.scale";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "layer_scalar_mul_bfloat16",
        F32 => "layer_scalar_mul_float32",
    });
    ctx.fire(
        Fire::at("elemwise/norm_layer_scalar.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [256, 1, 1])),
        &[x.arg(), s.arg(), x.arg_mut()],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn res_blend(
    ctx: &Ctx<'_>,
    prefix: Tensor,
    blocks: Tensor,
    n_blocks: u32,
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.res_blend";
    const MAX_BLOCKS: usize = 32;

    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "res_blend_bfloat16",
        F32 => "res_blend_float32",
    });
    let rows = nonzero(OP, "rows", y.rows)?;
    let hidden = nonzero(OP, "the blended row's width", y.width)?;
    if n_blocks as usize > MAX_BLOCKS {
        return Err(refuse(
            OP,
            format!(
                "{n_blocks} candidate blocks exceed the kernel's softmax scratch bound of \
                 {MAX_BLOCKS}"
            ),
        ));
    }
    for (what, plane) in [
        ("prefix", prefix),
        ("norm weight", weight),
        ("projection", proj),
    ] {
        if plane.dtype != y.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} is {:?} and the blend lands {:?}; one pass reads them all \
                     in one element",
                    plane.dtype, y.dtype
                ),
            ));
        }
    }
    for (what, plane) in [("norm weight", weight), ("projection", proj)] {
        if plane.rows.saturating_mul(plane.width) != hidden {
            return Err(refuse(
                OP,
                format!(
                    "the {what} is {} x {} and this blend scores rows of {hidden}",
                    plane.rows, plane.width
                ),
            ));
        }
    }
    if u64::from(blocks.rows) != u64::from(rows) * u64::from(n_blocks)
        || blocks.width != hidden
        || blocks.dtype != y.dtype
    {
        return Err(refuse(
            OP,
            format!(
                "the candidates are {} x {} {:?} and this blend takes {n_blocks} stacked \
                 planes of {rows} x {hidden} {:?}",
                blocks.rows, blocks.width, blocks.dtype, y.dtype
            ),
        ));
    }
    let first = blocks;
    let count = n_blocks;
    ctx.fire(
        Fire::at("elemwise/res_blend.metal", entry).apply(Grid::of(
            [
                rows.checked_mul(256).ok_or_else(|| {
                    refuse(OP, format!("the grid will not launch: {rows} rows x 256"))
                })?,
                1,
                1,
            ],
            [256, 1, 1],
        )),
        &[
            prefix.arg(),
            first.arg(),
            weight.arg(),
            proj.arg(),
            y.arg_mut(),
            count.arg(),
            hidden.arg(),
            rows.arg(),
            eps.arg(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::probe::Probe;

    const WIDTH: u32 = 1152;

    fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
        Tensor::new(buf, rows, width, Dtype::Bf16)
    }

    #[test]
    fn norm_every_case() {
        a_plane_that_is_not_one_scalar_per_column_is_refused_by_name();
        a_plane_in_another_element_is_refused_by_name();
        an_element_with_no_instantiation_is_refused_by_dtype();
        an_empty_rectangle_is_refused_by_name();
        a_degenerate_centred_norm_is_refused_by_name();
    }

    fn a_plane_that_is_not_one_scalar_per_column_is_refused_by_name() {
        let probe = Probe::default();
        let out = bf16(1, 5, WIDTH);

        let short = standardize(&probe, bf16(2, 1, WIDTH - 1), bf16(3, 1, WIDTH), out)
            .expect_err("a plane narrower than the rectangle is refused");
        assert!(format!("{short}").contains("scalar per column"), "{short}");

        let long_scale = standardize(&probe, bf16(2, 1, WIDTH), bf16(3, 2, WIDTH), out)
            .expect_err("two scalars per column is not one");
        assert!(
            format!("{long_scale}").contains("scalar per column"),
            "{long_scale}"
        );

        assert!(
            probe.fires().is_empty(),
            "a refused standardization launched"
        );
    }

    fn a_plane_in_another_element_is_refused_by_name() {
        let probe = Probe::default();
        let out = bf16(1, 5, WIDTH);

        let why = standardize(
            &probe,
            bf16(2, 1, WIDTH),
            Tensor::new(3, 1, WIDTH, Dtype::F32),
            out,
        )
        .expect_err("a plane in another element is refused");
        assert!(format!("{why}").contains("activation's element"), "{why}");
        assert!(probe.fires().is_empty());
    }

    fn an_element_with_no_instantiation_is_refused_by_dtype() {
        let probe = Probe::default();
        let why = standardize(
            &probe,
            Tensor::new(2, 1, WIDTH, Dtype::F16),
            Tensor::new(3, 1, WIDTH, Dtype::F16),
            Tensor::new(1, 5, WIDTH, Dtype::F16),
        )
        .expect_err("this plane stamps the standardization for bf16");
        assert!(matches!(why, Error::DtypeUnsupported { .. }), "{why}");
    }

    fn an_empty_rectangle_is_refused_by_name() {
        let probe = Probe::default();
        let why = standardize(
            &probe,
            bf16(2, 1, WIDTH),
            bf16(3, 1, WIDTH),
            bf16(1, 0, WIDTH),
        )
        .expect_err("a rectangle with no rows is refused");
        assert!(format!("{why}").contains("rows"), "{why}");
    }

    fn a_degenerate_centred_norm_is_refused_by_name() {
        let probe = Probe::default();
        let no_width = layernorm(
            &probe,
            bf16(1, 8, 0),
            bf16(2, 1, 0),
            bf16(3, 1, 0),
            1e-6,
            bf16(4, 8, 0),
        )
        .expect_err("a row of no width is refused");
        assert!(format!("{no_width}").contains("width"), "{no_width}");

        let no_rows = layernorm(
            &probe,
            bf16(1, 0, 768),
            bf16(2, 1, 768),
            bf16(3, 1, 768),
            1e-6,
            bf16(4, 0, 768),
        )
        .expect_err("a rectangle of no rows is refused");
        assert!(format!("{no_rows}").contains("rows"), "{no_rows}");
        assert!(probe.fires().is_empty());
    }
}
