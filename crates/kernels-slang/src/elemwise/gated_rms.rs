//! Gated RMS norms (`elemwise.rmsnorm_gated`, `elemwise.rmsnorm_gated_by`)
//! from `kernels/elemwise/gated_rms.slang`: f32 rows and weight, a bf16 gate,
//! bf16 landing, one workgroup a head.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const GATED_RMS: &str = "elemwise/gated_rms.slang";

/// The workgroup width every gated instantiation is built at.
const GROUP: u32 = 256;

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
    debug_assert_eq!(
        x.dtype,
        dtype::Dtype::F32,
        "`{op}` norms an f32 accumulator"
    );
    debug_assert_eq!(
        weight.dtype,
        dtype::Dtype::F32,
        "`{op}` scales by an f32 weight"
    );
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
        Fire::at(GATED_RMS, entry).apply(Grid::of([groups, 1, 1], [GROUP, 1, 1])),
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
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "elemwise_gated_rms_sigmoid_bf16" })
    } else {
        dtype_dispatch!(OP, gate.dtype, { Bf16 => "elemwise_gated_rms_bf16" })
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
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "elemwise_gated_rms_sigmoid_bf16" });
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
