use crate::error::Error;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, nonzero, refuse, stated,
};
use crate::tensor::Tensor;

pub fn sigmoid_mul_heads(
    ctx: &Ctx<'_>,
    gate: Tensor,
    head_dim: u32,
    scale: f32,
    x: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gate_sigmoid_mul_heads";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "gate_sigmoid_mul_heads_bfloat16" });
    let head_dim = nonzero(OP, "the head width", head_dim)?;
    if !x.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "a {}-wide row is not a whole number of {head_dim}-wide heads",
                x.width
            ),
        ));
    }
    let heads = x.width / head_dim;
    if gate.width != heads || gate.rows < x.rows {
        return Err(refuse(
            OP,
            format!(
                "the gate plane is {} x {} and this fold reads one logit per head of \
                 {heads} for each of {} rows",
                gate.rows, gate.width, x.rows
            ),
        ));
    }
    ctx.fire(
        Fire::at("elemwise/gate.metal", entry).apply(Grid::of(
            elementwise_rows(OP, x.width, x.rows)?,
            [256.min(x.width), 1, 1],
        )),
        &[
            x.arg_mut(),
            gate.arg(),
            heads.arg(),
            head_dim.arg(),
            scale.arg(),
        ],
    )
}

pub fn sigmoid_mul(ctx: &Ctx<'_>, gate: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.gate_sigmoid_mul";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "gate_bfloat16" });
    debug_assert!(
        gate.rows == x.rows && gate.width == x.width && gate.dtype == x.dtype,
        "the gate plane rides the rectangle it gates"
    );
    ctx.fire(
        Fire::at("elemwise/gate.metal", entry).apply(Grid::of(
            elementwise_rows(OP, x.width, x.rows)?,
            [256, 1, 1],
        )),
        &[x.arg_mut(), gate.arg(), stated(OP, x.width)?.arg()],
    )
}
