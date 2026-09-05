use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
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

    let elements = u64::from(x.width) * u64::from(x.rows);
    let words = u32::try_from(elements.div_ceil(2)).map_err(|_| {
        refuse(
            OP,
            format!("a {} x {} plane will not launch", x.rows, x.width),
        )
    })?;
    ctx.fire(
        Fire::at("norm/gate.wgsl", entry).apply(Grid::of([words, 1, 1], [GROUP, 1, 1])),
        &[
            x.arg_mut(),
            gate.arg(),
            stated(OP, x.width)?.arg(),
            stated(OP, x.rows)?.arg(),
        ],
    )
}
