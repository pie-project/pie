use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const GROUP: u32 = 256;

pub fn sigmoid_mul(ctx: &Ctx<'_>, gate: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.gate_sigmoid_mul";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "gate_sigmoid_mul_bf16" });
    debug_assert!(
        gate.rows == x.rows && gate.width == x.width && gate.dtype == x.dtype,
        "the gate plane rides the rectangle it gates"
    );
    ctx.fire(
        Fire::at("norm/gate.slang", entry).apply(Grid::of(
            elementwise_rows(OP, x.width, x.rows)?,
            [GROUP, 1, 1],
        )),
        &[x.arg_mut(), gate.arg(), stated(OP, x.width)?.arg()],
    )
}
