//! `Norm`: the rms family, residual folds, and scalar gains — one entry per
//! IR variant. Selection (which shader, which dtype stamp) lives here, so
//! the driver's dispatch arm stays destructure → resolve → call.

use crate::error::Error;
use dtype::Dtype;

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

/// A per-head width that fits one threadgroup.
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
    let entry = dtype_dispatch!(op, x.dtype, { Bf16 => "rms_single_row_bfloat16" });
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

/// **THE HYPER-CONNECTION NORM.** `x` is `groups` streams of `group` values
/// side by side in one row; the moments are taken per stream and the gain is
/// `weight + 1` off a bank that is the row's FULL width — one plane per
/// stream, not one shared across them. That last part is the whole difference
/// from [`rmsnorm_per_head_plus_one`], which reads the same plane for every
/// slice of the row.
///
/// qwen4 spends it four times per layer: once on the four-stream residual
/// mixer and three times inside the PLE, whose key, query and convolution
/// norms are each `streams x hidden` wide.
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
    if x.width % axis != 0 {
        return Err(refuse(
            OP,
            format!("the {}-wide row is not a whole number of {axis}-wide groups", x.width),
        ));
    }
    let groups = x.width / axis;
    // The bank is per-group, so it has to span the row. A bank one group wide
    // would read the same plane for every stream and land the right spread
    // around three wrong centres, which is the failure this check exists for.
    let bank = weight.width * weight.rows.max(1);
    if bank != x.width {
        return Err(refuse(
            OP,
            format!("the weight bank is {bank} wide and the {groups} groups of {axis} it gains span {}", x.width),
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
    sigmoid_gate: bool,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_gated";
    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    // The checkpoint's `output_gate_type`, which the CUDA twin takes as the
    // same `bool`: `silu` multiplies the sigmoid by the gate row and `sigmoid`
    // does not. One shader, two instantiations, and the difference is a factor
    // this op would otherwise silently drop.
    let entry = if sigmoid_gate {
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_sigmoid_f32_bfloat16" })
    } else {
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_f32_bfloat16" })
    };
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
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_gated_by";
    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "gated_rms_sigmoid_f32_bfloat16" });
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

/// `y += x`, in place on `y` (the IR aliases `y_out` onto `y`).
pub fn residual_add(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "residual_add_bfloat16" });
    ctx.fire(
        Fire::at("elemwise/norm_residual_add.metal", entry)
            .apply(Grid::of(elementwise(OP, y.width, y.rows)?, [256, 1, 1])),
        &[x.arg(), y.arg(), y.arg_mut()],
    )
}

/// `out += bias` per row, in place on `out`.
pub fn add_bias(ctx: &Ctx<'_>, bias: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add_bias";
    let entry = dtype_dispatch!(OP, out.dtype, { Bf16 => "add_bias_bfloat16" });
    let lanes = elementwise_rows(OP, out.width, out.rows)?;
    ctx.fire(
        Fire::at("elemwise/norm_add_bias.metal", entry)
            .apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
        &[
            out.arg_mut(),
            bias.arg(),
            stated(OP, out.width)?.arg(),
        ],
    )
}

/// **THE WHOLE `nn.LayerNorm`, IN ONE LAUNCH**: `y = (x − mean(x)) ·
/// rsqrt(var(x) + eps) · w + b`, whole rows, both planes `[width]`
/// (`.wiki/alto/multimodal.md` §6.1, §9.1, next.md B5) — this plane's mirror
/// of `kernels_cuda::elemwise::layernorm::layernorm`.
///
/// Every qwen vision block is an `nn.LayerNorm`: the checkpoints publish
/// `blocks.{l}.norm1.bias` beside `.weight`, and an RMSNorm has no bias. The
/// dev tower says it twenty-five times a fire (`norm1`/`norm2` on twelve
/// blocks, plus `merger.norm`), which is why the row is one launch and not
/// the three-op spelling it replaces.
///
/// **THE MOMENTS ARE TWO REDUCTIONS AND NOT ONE.** `var = E[x²] − E[x]²`
/// would halve the barriers and cancels catastrophically on a tower row whose
/// mean is large against its spread — a slightly wrong norm rather than a
/// NaN. The shader reduces the mean, then the centred squares against it,
/// which is `torch.nn.LayerNorm`'s own order.
///
/// **AND THE CENTRED ROW IS NEVER ROUNDED ON THE WAY THROUGH**, which is the
/// one thing the composition cannot claim — see the shader.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16; a refusal for a
/// zero-wide row or a zero-row rectangle, which are the two launches that
/// would leave the destination unwritten rather than normed.
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
    // ONE THREADGROUP PER ROW, over the row's WHOLE width — a `LayerNorm`
    // has no per-head axis, so `rms_grid`'s `norms per row` is one and this
    // is that grid at `axis == width`. Reusing it keeps the two families'
    // launch shapes one function rather than two that could drift about how
    // many threads a wide row gets.
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

/// **THE VISION TOWER'S OUTPUT STANDARDIZATION**: `out = (out - bias) *
/// scale` per COLUMN, in place on `out`, both planes `[width]` of the
/// activation's element — `vision_config.standardize`'s own line
/// (`.wiki/alto/multimodal.md` §21.3), and this plane's mirror of
/// `kernels_cuda::elemwise::norm::standardize`.
///
/// **IT IS [`add_bias`]'S LAUNCH WITH ONE MORE PLANE**, deliberately: same
/// `[width, rows]` grid, same threadgroup, same `tid.x`-is-the-column
/// indexing. What the shader adds is the second read and the multiply.
///
/// **THE DIFFERENCE IS TAKEN IN f32 AND ROUNDED ONCE, AT THE STORE.** The
/// pooler's `sqrt(hidden)` has already expanded the magnitude and this is
/// what brings it back, so where `out` and `bias` nearly cancel the
/// surviving number is many ulps of what a composed spelling — [`add_bias`]
/// with a negated plane, then a per-column multiply — would have rounded away
/// between its two launches. That composition is unavailable anyway
/// ([`scale`] reads ONE device-held scalar), and this note is why it would
/// still be wrong if it were not.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16; a refusal for a plane
/// that is not one scalar per column or not the activation's element, and for
/// an empty rectangle.
pub fn standardize(ctx: &Ctx<'_>, bias: Tensor, scale: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.standardize";
    let entry = dtype_dispatch!(OP, out.dtype, { Bf16 => "standardize_bfloat16" });
    let lanes = elementwise_rows(OP, out.width, out.rows)?;
    // **BOTH PLANES ARE PER-COLUMN**, which is the whole difference from
    // `scale`'s one device-held scalar — so a plane of the wrong width is a
    // silent misread of every column past the first, and gets a refusal.
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

/// `x *= s` for a plan-stated scalar, in place on `x`.
pub fn mul_scalar(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.mul_scalar";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "layer_scalar_mul_stated_bfloat16" });
    ctx.fire(
        Fire::at("elemwise/norm_layer_scalar.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [256, 1, 1])),
        &[x.arg(), s.arg(), x.arg_mut()],
    )
}

/// `silu(s · x)` for a plan-stated scalar, in place on `x` — qwen4's shared
/// gate, whose scale is a trace constant and not a plane.
pub fn silu_scaled(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.silu_scaled";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "silu_scaled_bfloat16" });
    ctx.fire(
        Fire::at("elemwise/norm_layer_scalar.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [256, 1, 1])),
        &[x.arg_mut(), s.arg()],
    )
}

/// `x *= s` for a device-held scalar, in place on `x`.
pub fn scale(ctx: &Ctx<'_>, s: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.scale";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "layer_scalar_mul_bfloat16" });
    ctx.fire(
        Fire::at("elemwise/norm_layer_scalar.metal", entry)
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
) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "elementwise.res_blend",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::ArgValue;
    use crate::probe::Probe;

    /// gemma's wide tower's own hidden, so the launch shape under test is the
    /// one that ships.
    const WIDTH: u32 = 1152;

    fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
        Tensor::new(buf, rows, width, Dtype::Bf16)
    }

    /// **THE SAME LAUNCH AS `add_bias`, ONE PLANE WIDER** — which is the
    /// claim Session I's entry makes, stated where it can be read back.
    #[test]
    fn the_standardization_is_add_bias_launch_with_one_more_plane() {
        let out = bf16(1, 128, WIDTH);

        let biased = Probe::default();
        add_bias(&biased, bf16(2, 1, WIDTH), out).expect("the bias enqueues");
        let (bf, ba) = biased.only();

        let standard = Probe::default();
        standardize(&standard, bf16(2, 1, WIDTH), bf16(3, 1, WIDTH), out)
            .expect("the standardization enqueues");
        let (sf, sa) = standard.only();

        assert_eq!(sf.entrypoint, "standardize_bfloat16");
        assert_eq!(sf.file, "elemwise/norm_standardize.metal");
        // One thread per element, rows on their own axis: the column is the
        // grid's x, which is what the shader indexes both planes by.
        assert_eq!(sf.lanes, [WIDTH, 128, 1]);
        assert_eq!(sf.lanes, bf.lanes);
        assert_eq!(sf.group, bf.group);
        assert_eq!(sf.group, [256, 1, 1]);

        // The rectangle is written in place and the planes are read.
        assert_eq!(sa[0], ArgValue::BufferMut(1));
        assert_eq!(sa[1], ArgValue::Buffer(2));
        assert_eq!(sa[2], ArgValue::Buffer(3));
        assert_eq!(sa[3], ArgValue::I32(WIDTH as i32));
        // The one difference in the argument list: the scale plane sits
        // between the bias and the width.
        assert_eq!(&sa[..2], &ba[..2]);
        assert_eq!(sa[3], ba[2]);
        assert_eq!(sa.len(), ba.len() + 1);
    }

    /// A plane is one scalar per COLUMN, and a rectangle's row count is not
    /// its width — so the two are named apart in the refusal.
    #[test]
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

        assert!(probe.fires().is_empty(), "a refused standardization launched");
    }

    /// **BOTH PLANES RIDE THE ACTIVATION'S ELEMENT.** The checkpoint ships
    /// `vision_tower.std_{bias,scale}` in the tower's own element, and a
    /// plane in another one would be read through the shader's `T`.
    #[test]
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
        assert!(
            format!("{why}").contains("activation's element"),
            "{why}"
        );
        assert!(probe.fires().is_empty());
    }

    /// The element the shader is stamped for is the RECTANGLE's, so an
    /// unstamped activation is a dtype refusal and not a plane refusal —
    /// checked before either plane is looked at.
    #[test]
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

    /// An empty rectangle is refused rather than launched at zero extent —
    /// `elementwise_rows`' own rule, reached through this entry.
    #[test]
    fn an_empty_rectangle_is_refused_by_name() {
        let probe = Probe::default();
        let why = standardize(&probe, bf16(2, 1, WIDTH), bf16(3, 1, WIDTH), bf16(1, 0, WIDTH))
            .expect_err("a rectangle with no rows is refused");
        assert!(format!("{why}").contains("rows"), "{why}");
    }

    /// **ONE THREADGROUP PER ROW, OVER THE ROW'S WHOLE WIDTH.** A
    /// `LayerNorm` has no per-head axis, so the row grid's "norms per row" is
    /// one — which is what makes this `rms_grid` at `axis == width` and not a
    /// second launch shape.
    #[test]
    fn the_centred_norm_launches_one_group_per_row_over_the_whole_row() {
        let probe = Probe::default();
        // qwen35-d0.8b's tower: 768 wide.
        layernorm(
            &probe,
            bf16(1, 96, 768),
            bf16(2, 1, 768),
            bf16(3, 1, 768),
            1e-6,
            bf16(4, 96, 768),
        )
        .expect("the centred norm enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.file, "elemwise/norm_layernorm.metal");
        assert_eq!(f.entrypoint, "layernorm_bfloat16");
        // `rms_threads` is `width.div_ceil(4).min(1024)` = 192, and the grid
        // is one group of those per row.
        assert_eq!(f.group, [192, 1, 1]);
        assert_eq!(f.lanes, [192 * 96, 1, 1]);
        // Both planes are read and the destination is written.
        assert_eq!(a[0], ArgValue::Buffer(1));
        assert_eq!(a[1], ArgValue::Buffer(2));
        assert_eq!(a[2], ArgValue::Buffer(3));
        assert_eq!(a[3], ArgValue::BufferMut(4));
        assert_eq!(a[4], ArgValue::F32(1e-6));
        // The shader reduces over the row's whole width — `eps` sits inside
        // the root beside the variance, and the axis is the row.
        assert_eq!(a[5], ArgValue::U32(768));
    }

    /// A row of no width and a rectangle of no rows are the two launches that
    /// would leave the destination unwritten rather than normed.
    #[test]
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
