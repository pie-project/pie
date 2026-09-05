#![allow(unused_variables)]
#![allow(clippy::too_many_arguments)]

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, elementwise_rows, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const GROUP: u32 = 256;

const PACKED: &str = "mlp/packed.wgsl";

const GATED: &str = "mlp/gated.wgsl";

struct Halves {
    packed: Tensor,
    y: Tensor,
    grid: Grid,
    intermediate: u32,
}

fn halves(op: &'static str, packed: Tensor, intermediate: u32, y: Tensor) -> Result<Halves, Error> {
    if packed.width != intermediate.saturating_mul(2) || y.width != intermediate {
        return Err(refuse(
            op,
            format!(
                "the packed row is {} wide and lands {} where the intermediate is {intermediate}",
                packed.width, y.width
            ),
        ));
    }
    debug_assert_eq!(
        y.rows, packed.rows,
        "the activation lands one row per packed row"
    );
    if !intermediate.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the intermediate {intermediate} is odd: a bf16 word holds a pair"),
        ));
    }
    let lanes = elementwise_rows(op, y.width / 2, y.rows)?;
    Ok(Halves {
        packed,
        y,
        grid: Grid::of(lanes, [GROUP, 1, 1]),
        intermediate,
    })
}

fn split_grid(
    op: &'static str,
    gate: Tensor,
    up: Tensor,
    y: Tensor,
) -> Result<([u32; 3], u32), Error> {
    if (up.rows, up.width) != (gate.rows, gate.width)
        || (y.rows, y.width) != (gate.rows, gate.width)
    {
        return Err(refuse(
            op,
            "the gate, up and output planes are not one rectangle",
        ));
    }
    let lanes = elementwise(op, gate.width, gate.rows)?;
    Ok(([lanes[0].div_ceil(2), 1, 1], lanes[0]))
}

pub fn swiglu(ctx: &Ctx<'_>, packed: Tensor, intermediate: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_swiglu_bf16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at(PACKED, entry).apply(cut.grid),
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
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_swiglu_clamp_bf16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at(PACKED, entry).apply(cut.grid),
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
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_gptoss_swiglu_bf16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at(PACKED, entry).apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            limit.arg(),
            alpha.arg(),
        ],
    )
}

pub fn swiglu_clamp_split(
    ctx: &Ctx<'_>,
    gate: Tensor,
    up: Tensor,
    limit: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_split";
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "swiglu_clamp_bf16" });
    let (lanes, n) = split_grid(OP, gate, up, y)?;
    ctx.fire(
        Fire::at(GATED, entry).apply(Grid::of(lanes, [GROUP, 1, 1])),
        &[gate.arg(), up.arg(), y.arg_mut(), limit.arg(), n.arg()],
    )
}

pub fn geglu_tanh(ctx: &Ctx<'_>, gate: Tensor, up: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_geglu_tanh";
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "geglu_tanh_bf16" });
    let (lanes, n) = split_grid(OP, gate, up, y)?;
    ctx.fire(
        Fire::at(GATED, entry).apply(Grid::of(lanes, [GROUP, 1, 1])),
        &[gate.arg(), up.arg(), y.arg_mut(), n.arg()],
    )
}

pub fn gelu_tanh(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_gelu_tanh";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "mlp_gelu_tanh_bf16" });
    let lanes = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(GATED, entry).apply(Grid::of([lanes[0].div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg(), y.arg_mut(), lanes[0].arg()],
    )
}

pub fn geglu_tanh_packed(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_geglu_tanh_packed";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_geglu_tanh_bf16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at(PACKED, entry).apply(cut.grid),
        &[cut.packed.arg(), cut.y.arg_mut(), cut.intermediate.arg()],
    )
}

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
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_situ_bf16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at(PACKED, entry).apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            beta.arg(),
            up_cap.unwrap_or(0.0).arg(),
        ],
    )
}
