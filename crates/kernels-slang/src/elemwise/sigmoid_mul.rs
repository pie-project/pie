//! `elemwise.gate_sigmoid_mul` from `kernels/elemwise/sigmoid_mul.slang`:
//! `x *= sigmoid(gate)` in place, the gate a tensor of `x`'s shape.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const SIGMOID_MUL: &str = "elemwise/sigmoid_mul.slang";

/// The workgroup width the instantiation is built at.
const GROUP: u32 = 256;

pub fn sigmoid_mul(ctx: &Ctx<'_>, gate: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.gate_sigmoid_mul";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_sigmoid_mul_bf16" });
    if (gate.rows, gate.width, gate.dtype) != (x.rows, x.width, x.dtype) {
        return Err(refuse(
            OP,
            format!(
                "a {} x {} {:?} gate on a {} x {} {:?} rectangle; the gate rides the \
                 rectangle it gates",
                gate.rows, gate.width, gate.dtype, x.rows, x.width, x.dtype
            ),
        ));
    }
    let [n, _, _] = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(SIGMOID_MUL, entry).apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg_mut(), gate.arg(), n.arg()],
    )
}
