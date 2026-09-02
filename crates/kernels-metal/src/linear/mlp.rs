//! `Mlp`: gated activations over a packed `[gate | up]` row (and one
//! two-tensor form). One entry per IR variant.

use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, elementwise_rows, refuse};
use crate::tensor::Tensor;

const GROUP: u32 = 256;

/// The packed row cut in two, plus the geometry every packed entry shares.
struct Halves {
    packed: Tensor,
    y: Tensor,
    grid: Grid,
    intermediate: u32,
}

fn halves(op: &'static str, packed: Tensor, intermediate: u32, y: Tensor) -> Result<Halves, Error> {
    debug_assert_eq!(
        packed.width,
        intermediate.saturating_mul(2),
        "the packed `[gate | up]` row is twice the intermediate width it states"
    );
    debug_assert_eq!(
        y.width, intermediate,
        "the activation's row is the intermediate width it states"
    );
    debug_assert_eq!(
        y.rows, packed.rows,
        "the activation lands one row per packed row"
    );
    let lanes = elementwise_rows(op, y.width, y.rows)?;
    Ok(Halves {
        packed,
        y,
        grid: Grid::of(lanes, [lanes[0].min(GROUP), 1, 1]),
        intermediate,
    })
}

pub fn swiglu(ctx: &Ctx<'_>, packed: Tensor, intermediate: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_swiglu_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[cut.packed.arg(), cut.y.arg_mut(), cut.intermediate.arg()],
    )
}

pub fn swiglu_clamp(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    limit: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_swiglu_clamp_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            limit.arg(),
        ],
    )
}

pub fn swiglu_clamp_alpha(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    limit: f32,
    alpha: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_alpha";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_gptoss_swiglu_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            limit.arg(),
            alpha.arg(),
        ],
    )
}

/// [`swiglu_clamp`] over an unfused pair: the same clamps and the same silu,
/// with the halves handed over as two tensors rather than as one packed row.
pub fn swiglu_clamp_split(
    ctx: &Ctx<'_>,
    gate: Tensor,
    up: Tensor,
    limit: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_split";
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "swiglu_clamp_bfloat16" });
    debug_assert_eq!(
        (up.rows, up.width),
        (gate.rows, gate.width),
        "the unfused halves are one rectangle each and they are the same rectangle"
    );
    debug_assert_eq!(
        (y.rows, y.width),
        (gate.rows, gate.width),
        "the activation lands on the halves' own rectangle"
    );
    ctx.fire(
        Fire::at("linear/mlp_gated.metal", entry).apply(Grid::of(
            elementwise(OP, gate.width, gate.rows)?,
            [GROUP, 1, 1],
        )),
        &[gate.arg(), up.arg(), y.arg_mut(), limit.arg()],
    )
}

pub fn geglu_tanh(ctx: &Ctx<'_>, gate: Tensor, up: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_geglu_tanh";
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "geglu_tanh_bfloat16" });
    ctx.fire(
        Fire::at("linear/mlp_gated.metal", entry).apply(Grid::of(
            elementwise(OP, gate.width, gate.rows)?,
            [GROUP, 1, 1],
        )),
        &[gate.arg(), up.arg(), y.arg_mut()],
    )
}

/// The ungated map: gelu_tanh over one projection, no
/// `up` multiply — a tower MLP's spelling, not a gated trunk's with ones.
pub fn gelu_tanh(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_gelu_tanh";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "mlp_gelu_tanh_bfloat16" });
    ctx.fire(
        Fire::at("linear/mlp_gated.metal", entry).apply(Grid::of(
            elementwise(OP, x.width, x.rows)?,
            [GROUP, 1, 1],
        )),
        &[x.arg(), y.arg_mut()],
    )
}

pub fn geglu_tanh_packed(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_geglu_tanh_packed";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_geglu_tanh_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[cut.packed.arg(), cut.y.arg_mut(), cut.intermediate.arg()],
    )
}

/// `up_cap: None` means uncapped; the shader reads 0 as "no cap".
pub fn situ(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    beta: f32,
    up_cap: Option<f32>,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_situ";
    if beta == 0.0 {
        return Err(refuse(OP, "beta is zero, and the gate divides by it"));
    }
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_situ_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            beta.arg(),
            up_cap.unwrap_or(0.0).arg(),
        ],
    )
}
