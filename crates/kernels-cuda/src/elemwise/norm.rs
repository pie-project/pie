//! `Norm`: the rms family, residual folds, and scalar gains — one entry per IR variant.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/norm.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

/// Elements one thread of a vectorised row kernel moves at once.
const VEC_WIDTH: u32 = 8;

/// The single-row rms unit's two weight readings: the bank read absolutely,
/// or offset by one (`w + 1`).
const ABSOLUTE_BANK: &str = "rmsnorm";

const OFFSET_BANK: &str = "rmsnorm_plus_one";

/// One block per normed span: whole rows when `head_dim` is zero, else one
/// per head.
fn rows_per_head(op: &'static str, rows: u32, width: u32, head_dim: u32) -> Result<Launch, Error> {
    nonzero(op, "rows", rows)?;
    if head_dim == 0 {
        return Ok(Launch::per_row(rows, BLOCK));
    }
    if width == 0 || width % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    let blocks = rows.checked_mul(width / head_dim).ok_or_else(|| {
        refuse(
            op,
            format!(
                "the grid will not launch: {rows} rows x {} heads",
                width / head_dim
            ),
        )
    })?;
    Ok(Launch::per_row(blocks, BLOCK))
}

/// The head count `rows_per_head` flattens into `blockIdx.x`; seated kernels
/// divide by it to recover the token row.
const fn per_head_split(width: u32, head_dim: u32) -> u32 {
    if head_dim == 0 { 1 } else { width / head_dim }
}

/// One thread per element, flattened — refused rather than truncated when
/// the extent outgrows a 32-bit launch, because a clamped grid would leave
/// the tail unwritten.
fn elementwise(op: &'static str, t: Tensor) -> Result<(Launch, u64), Error> {
    let n = t.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(op, "the element count", lanes)?;
    Ok((Launch::flat(lanes, BLOCK), n))
}

/// One block per row, sized to the row in whole warps.
fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

fn rms_row(
    ctx: &Ctx,
    op: &'static str,
    template: &'static str,
    x: Tensor,
    weight: Tensor,
    y: &mut Tensor,
    per_head_dim: u32,
    eps: f32,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(
        op,
        nonzero(
            op,
            "the normed width",
            if per_head_dim == 0 {
                y.width
            } else {
                per_head_dim
            },
        )?,
    )?;
    let launch = rows_per_head(op, y.rows, y.width, per_head_dim)?;
    // How many blocks stand in one token row.
    let heads = stated(op, per_head_split(y.width, per_head_dim))?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::{template}<{t}, 256>")),
        )
        .apply(launch),
        // hidden three times: the normed width and the source/destination
        // row pitches, which coincide on dense handles.
        &[
            x.arg(),
            weight.arg(),
            y.arg(),
            hidden.arg(),
            hidden.arg(),
            hidden.arg(),
            eps.arg(),
            heads.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

pub fn rmsnorm(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm",
        ABSOLUTE_BANK,
        x,
        weight,
        y,
        0,
        eps,
    )
}

pub fn rmsnorm_per_head(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_per_head",
        ABSOLUTE_BANK,
        x,
        weight,
        y,
        head_dim,
        eps,
    )
}

pub fn rmsnorm_plus_one(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_plus_one",
        OFFSET_BANK,
        x,
        weight,
        y,
        0,
        eps,
    )
}

pub fn rmsnorm_per_head_plus_one(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_per_head_plus_one",
        OFFSET_BANK,
        x,
        weight,
        y,
        head_dim,
        eps,
    )
}

pub fn rmsnorm_no_scale(
    ctx: &Ctx,
    x: Tensor,
    head_dim: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_no_scale";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(
        OP,
        nonzero(
            OP,
            "the normed width",
            if head_dim == 0 { y.width } else { head_dim },
        )?,
    )?;
    let launch = rows_per_head(OP, y.rows, y.width, head_dim)?;
    let heads = stated(OP, per_head_split(y.width, head_dim))?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::rmsnorm_no_scale<{t}, 256>")),
        )
        .apply(launch),
        &[
            x.arg(),
            y.arg(),
            hidden.arg(),
            eps.arg(),
            heads.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// `x` and `weight` are f32 (the recurrent accumulator); the gate carries
/// the model dtype and the output lands in it.
pub fn rmsnorm_gated(
    ctx: &Ctx,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    sigmoid_gate: bool,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_gated";
    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    let t = dtype_dispatch!(OP, gate.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(
        OP,
        nonzero(
            OP,
            "the normed width",
            if head_dim == 0 { y.width } else { head_dim },
        )?,
    )?;
    let launch = rows_per_head(OP, y.rows, y.width, head_dim)?;
    let heads = stated(OP, per_head_split(y.width, head_dim))?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::rmsnorm_gated_f32_in<{t}, 256>")),
        )
        .apply(launch),
        &[
            x.arg(),
            gate.arg(),
            weight.arg(),
            y.arg(),
            hidden.arg(),
            eps.arg(),
            i32::from(sigmoid_gate).arg(),
            heads.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// The hyper-connection norm: moments per `group`-wide slice, `weight + 1`
/// over the row's full width. The launch is [`rows_per_head`]'s flattening —
/// one block per (row, group) — and the kernel recovers the group from the
/// block index to read its own slice of the weight.
pub fn rmsnorm_grouped_plus_one(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    group: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_grouped_plus_one";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let group = nonzero(OP, "the group width", group)?;
    if y.width == 0 || y.width % group != 0 {
        return Err(refuse(
            OP,
            format!("the {}-wide row is not a whole number of {group}-wide groups", y.width),
        ));
    }
    debug_assert_eq!(
        weight.width * weight.rows.max(1),
        y.width,
        "the weight spans the row's full width"
    );
    let groups = y.width / group;
    let launch = rows_per_head(OP, y.rows, y.width, group)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::rmsnorm_grouped_plus_one<{t}, 256>")),
        )
        .apply(launch),
        &[
            x.arg(),
            weight.arg(),
            y.arg(),
            stated(OP, group)?.arg(),
            stated(OP, groups)?.arg(),
            eps.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// Like [`rmsnorm_gated`], grouped by a stated head count instead of a
/// stated head width — the kda output norm.
pub fn rmsnorm_gated_by(
    ctx: &Ctx,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    heads: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_gated_by";

    const KDA_BLOCK_MIN: u32 = WARP;

    const KDA_BLOCK_MAX: u32 = 128;

    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    let t = dtype_dispatch!(OP, gate.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "the stated head count", heads)?;
    nonzero(OP, "rows", y.rows)?;
    if x.width == 0 || x.width % heads != 0 {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide normed row does not divide by the stated head count {heads}",
                x.width
            ),
        ));
    }
    let d = x.width / heads;
    ctx.fire(
        OP,
        Fire::at(
            "elemwise/norm.cuh",
            symbol(&format!("::pie::elemwise::rmsnorm_gated_by<{t}>")),
        )
        .apply(Launch::grid(
            [y.rows, heads, 1],
            [d.clamp(KDA_BLOCK_MIN, KDA_BLOCK_MAX), 1, 1],
        )),
        &[
            x.arg(),
            gate.arg(),
            weight.arg(),
            y.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, d)?.arg(),
            eps.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// `y += x`, in place on `y` (the IR aliases `y_out` onto `y`).
pub fn residual_add(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add";
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *y)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::residual_add<{t}>"))).apply(launch),
        &[
            y.arg(),
            x.arg(),
            n.arg(),
            // Element-form seat's width: launch is flat over rows*width, so
            // the kernel needs it to read the staged row count/start as elements.
            stated(OP, y.width)?.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// `y += x` in place, then `out = rmsnorm(y)` (`weight + 1` when
/// `plus_one`): what [`residual_add`] then [`rmsnorm`] land, one launch.
pub fn residual_add_rmsnorm(
    ctx: &Ctx,
    x: Tensor,
    y: &mut Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    out: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add_rmsnorm";
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        x.rows == y.rows && x.width == y.width && out.rows == y.rows && out.width == y.width,
        "the residual, the stream and the normed row share one shape"
    );
    let hidden = stated(OP, nonzero(OP, "the normed width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    let plus = if plus_one { "true" } else { "false" };
    let vectors = y.dtype == Dtype::Bf16
        && y.width % VEC_WIDTH == 0
        && [x.ptr, y.ptr, weight.ptr, out.ptr].iter().all(|&at| aligned16(at));
    let (entrypoint, block) = if vectors {
        let block = (y.width / VEC_WIDTH).clamp(WARP, BLOCK).next_power_of_two();
        (
            format!("::pie::elemwise::residual_add_rmsnorm_vec8<{block}, {plus}>"),
            block,
        )
    } else {
        (
            format!("::pie::elemwise::residual_add_rmsnorm<{t}, 256, {plus}>"),
            BLOCK,
        )
    };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&entrypoint)).apply(Launch::per_row(rows, block)),
        &[
            x.arg(),
            y.arg(),
            weight.arg(),
            out.arg(),
            hidden.arg(),
            eps.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// `out += bias` per row, in place on `out`.
pub fn add_bias(ctx: &Ctx, bias: Tensor, out: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add_bias";
    let t = dtype_dispatch!(OP, out.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "rows", out.rows)?;
    let width = stated(OP, nonzero(OP, "the biased row's width", out.width)?)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::add_bias<{t}>")))
            .apply(route_rows(out.rows, out.width)),
        &[
            out.arg(),
            bias.arg(),
            width.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// `out = (out - bias) * scale` per row, in place on `out`, both planes
/// `[width]`. The difference is taken and scaled in f32 and rounded once at
/// the store, to avoid rounding error where `out` and `bias` nearly cancel.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for a
/// mismatched plane, an empty rectangle, or an extent past a 32-bit launch.
pub fn standardize(
    ctx: &Ctx,
    bias: Tensor,
    scale: Tensor,
    out: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.standardize";
    let t = dtype_dispatch!(OP, out.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "rows", out.rows)?;
    let width = stated(OP, nonzero(OP, "the standardized row's width", out.width)?)?;
    // Both planes are per-column (unlike `scale`'s one device-held scalar).
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
        if plane.elements() != u64::from(out.width) {
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
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::standardize<{t}>")))
            .apply(route_rows(out.rows, out.width)),
        &[out.arg(), bias.arg(), scale.arg(), width.arg()],
    )
}

/// `x *= s` for a plan-stated scalar, in place on `x`.
/// `silu(s * x)`, in place.
pub fn silu_scaled(ctx: &Ctx, s: f32, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.silu_scaled";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *x)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::silu_scaled<{t}>"))).apply(launch),
        &[
            x.arg(),
            s.arg(),
            n.arg(),
            // Element-form seat's width: launch is flat over rows*width, so
            // the kernel needs it to read the staged row count/start as elements.
            stated(OP, x.width)?.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

pub fn mul_scalar(ctx: &Ctx, s: f32, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.mul_scalar";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *x)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::mul_scalar<{t}>"))).apply(launch),
        &[
            x.arg(),
            s.arg(),
            n.arg(),
            // Element-form seat's width: launch is flat over rows*width, so
            // the kernel needs it to read the staged row count/start as elements.
            stated(OP, x.width)?.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// `x *= s` for a device-held scalar, in place on `x`.
pub fn scale(ctx: &Ctx, s: Tensor, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.scale";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *x)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::scale<{t}>"))).apply(launch),
        &[
            x.arg(),
            s.arg(),
            n.arg(),
            // Element-form seat's width: launch is flat over rows*width, so
            // the kernel needs it to read the staged row count/start as elements.
            stated(OP, x.width)?.arg(),
            // Staged-geometry seat: live-rows word if a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}

/// This plane claims the point, but the launch lives with the attention
/// family; this entry is the seam, `crate::attn::res_blend` is the launch.
pub fn res_blend(
    ctx: &Ctx,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    crate::attn::res_blend(ctx, prefix, blocks, weight, eps, proj, y)
}
