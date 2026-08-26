//! `Gate`: `x *= sigmoid(gate)`, in place on `x`. One point, and the old
//! plane claimed it.

use kernels::KernelError;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, stated};
use crate::tensor::Tensor;

pub fn sigmoid_mul(ctx: &Ctx<'_>, x: Tensor, gate: Tensor) -> Result<(), KernelError> {
    const OP: &str = "gate.sigmoid_mul";
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
