//! `Norm`: the rms family, residual folds, and scalar gains — one entry per
//! IR variant. Selection (which shader, which dtype stamp) lives here, so
//! the driver's dispatch arm stays destructure → resolve → call.

use new_kernels::KernelError;
use new_model_ir::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, elementwise_rows, nonzero, refuse, stated,
};
use crate::tensor::Tensor;

/// The single-row rms shader reads its variant from these: a dense weight
/// bank, read absolutely or offset by one (`w + 1`, Gemma-style), at unit
/// output gain.
const DENSE_BANK: u32 = 1;

const ABSOLUTE_BANK: u32 = 0;

const OFFSET_BANK: u32 = 1;

const UNIT_GAIN: f32 = 1.0;

fn rms_threads(op: &'static str, axis: u32) -> Result<u32, KernelError> {
    nonzero(op, "the normed axis", axis)?;
    Ok(axis.div_ceil(4).min(1024))
}

fn rms_grid(op: &'static str, width: u32, axis: u32, rows: u32) -> Result<Grid, KernelError> {
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

fn head_row_grid(
    op: &'static str,
    threads: u32,
    heads: u32,
    rows: u32,
) -> Result<[u32; 3], KernelError> {
    Ok([
        threads,
        nonzero(op, "heads", heads)?,
        nonzero(op, "rows", rows)?,
    ])
}

/// A per-head width that fits one threadgroup.
fn head_width(op: &'static str, vd: u32) -> Result<u32, KernelError> {
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
) -> Result<(), KernelError> {
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "rms_single_row_bfloat16" });
    let grid = rms_grid(op, x.width, axis, x.rows)?;
    ctx.fire(
        Fire::at("norm/rms.metal", entry).apply(grid),
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

pub fn rmsnorm(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), KernelError> {
    rms_row(
        ctx,
        "norm.rmsnorm",
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
) -> Result<(), KernelError> {
    rms_row(
        ctx,
        "norm.rmsnorm_per_head",
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
) -> Result<(), KernelError> {
    rms_row(
        ctx,
        "norm.rmsnorm_plus_one",
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
) -> Result<(), KernelError> {
    rms_row(
        ctx,
        "norm.rmsnorm_per_head_plus_one",
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

pub fn rmsnorm_no_scale(
    ctx: &Ctx<'_>,
    x: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "norm.rmsnorm_no_scale";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "vnorm_single_row_bfloat16" });
    let grid = rms_grid(OP, x.width, head_dim, x.rows)?;
    ctx.fire(
        Fire::at("norm/vector.metal", entry).apply(grid),
        &[
            x.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(OP, head_dim)?.arg(),
        ],
    )
}

/// `x` and `weight` are f32 (the recurrent accumulator); the gate carries
/// the model dtype and the output lands in it.
pub fn rmsnorm_gated(
    ctx: &Ctx<'_>,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "norm.rmsnorm_gated";
    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_f32_bfloat16" });
    let vd = head_width(OP, head_dim)?;
    if x.width == 0 || x.width % vd != 0 {
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
        Fire::at("norm/gated_rms.metal", entry).apply(Grid::of(lanes, [vd, 1, 1])),
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

/// Like [`rmsnorm_gated`], grouped by a stated head count instead of a
/// stated head width.
pub fn rmsnorm_gated_by(
    ctx: &Ctx<'_>,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    heads: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "norm.rmsnorm_gated_by";
    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_by_f32_bfloat16" });
    nonzero(OP, "the stated head count", heads)?;
    if x.width == 0 || x.width % heads != 0 {
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
        Fire::at("norm/gated_rms.metal", entry).apply(Grid::of(lanes, [vd, 1, 1])),
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

/// `y += x`, in place on `y` (the IR aliases `y_out` onto `y`).
pub fn residual_add(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), KernelError> {
    const OP: &str = "norm.residual_add";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "residual_add_bfloat16" });
    ctx.fire(
        Fire::at("norm/residual_add.metal", entry)
            .apply(Grid::of(elementwise(OP, y.width, y.rows)?, [256, 1, 1])),
        &[x.arg(), y.arg(), y.arg_mut()],
    )
}

/// `out += bias` per row, in place on `out`.
pub fn add_bias(ctx: &Ctx<'_>, bias: Tensor, out: Tensor) -> Result<(), KernelError> {
    const OP: &str = "norm.add_bias";
    let entry = dtype_dispatch!(OP, out.dtype, { Bf16 => "add_bias_bfloat16" });
    let lanes = elementwise_rows(OP, out.width, out.rows)?;
    ctx.fire(
        Fire::at("norm/add_bias.metal", entry)
            .apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
        &[
            out.arg_mut(),
            bias.arg(),
            stated(OP, out.width)?.arg(),
        ],
    )
}

/// `x *= s` for a plan-stated scalar, in place on `x`.
pub fn mul_scalar(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), KernelError> {
    const OP: &str = "norm.mul_scalar";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "layer_scalar_mul_stated_bfloat16" });
    ctx.fire(
        Fire::at("norm/layer_scalar.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [256, 1, 1])),
        &[x.arg(), s.arg(), x.arg_mut()],
    )
}

/// `x *= s` for a device-held scalar, in place on `x`.
pub fn scale(ctx: &Ctx<'_>, s: Tensor, x: Tensor) -> Result<(), KernelError> {
    const OP: &str = "norm.scale";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "layer_scalar_mul_bfloat16" });
    ctx.fire(
        Fire::at("norm/layer_scalar.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [256, 1, 1])),
        &[x.arg(), s.arg(), x.arg_mut()],
    )
}

/// The metal plane never claimed this point; the refusal is typed now.
pub fn res_blend(
    _ctx: &Ctx<'_>,
    _prefix: Tensor,
    _blocks: &[Tensor],
    _weight: Tensor,
    _eps: f32,
    _proj: Tensor,
    _y: Tensor,
) -> Result<(), KernelError> {
    Err(KernelError::Unsupported {
        op: "norm.res_blend",
    })
}

/// `Hc`: hyper-connections — residual streams expanded, mixed by learned
/// gates, and collapsed back. The metal plane never claimed any of these
/// points (the old file held an empty claims impl beside the norms), so
/// every entry is a typed refusal and the driver arm stays dumb.
pub mod hc {
    use new_kernels::KernelError;

    use crate::encode::Ctx;
    use crate::tensor::Tensor;

    pub fn expand(
        _ctx: &Ctx<'_>,
        _x: Tensor,
        _streams: u32,
        _y: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: "hc.expand" })
    }

    pub fn rmsnorm_f32(
        _ctx: &Ctx<'_>,
        _streams: Tensor,
        _eps: f32,
        _y: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "hc.rmsnorm_f32",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gates(
        _ctx: &Ctx<'_>,
        _normed: Tensor,
        _streams: Tensor,
        _scale: Tensor,
        _base: Tensor,
        _stream_count: u32,
        _gate_eps: f32,
        _alpha: f32,
        _sinkhorn: u32,
        _x: Tensor,
        _post_mix: Tensor,
        _comb_mix: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: "hc.gates" })
    }

    pub fn fold(
        _ctx: &Ctx<'_>,
        _x: Tensor,
        _streams: Tensor,
        _post_mix: Tensor,
        _comb_mix: Tensor,
        _y: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: "hc.fold" })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn collapse(
        _ctx: &Ctx<'_>,
        _streams: Tensor,
        _head_scale: Tensor,
        _head_base: Tensor,
        _stream_count: u32,
        _gate_eps: f32,
        _y: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: "hc.collapse" })
    }
}
