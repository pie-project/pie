//! `Gate`: the one-variant family — a sigmoid gate applied in place. It
//! lived beside `Mlp` in the old plane (its unit still does); the IR gives
//! the family its own file, one entry per variant like every other.

use kernels::KernelError;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const BLOCK: u32 = 256;

/// `x *= sigmoid(gate)`, per element, in place on `x` (the IR aliases
/// `x_out` onto `x`).
pub fn sigmoid_mul(ctx: &Ctx, gate: Tensor, x: &mut Tensor) -> Result<(), KernelError> {
    const OP: &str = "elementwise.gate_sigmoid_mul";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    debug_assert_eq!(gate.dtype, x.dtype, "the gate rides the rectangle's dtype");
    debug_assert!(
        gate.rows == x.rows && gate.width == x.width,
        "the gate plane is the rectangle it gates"
    );
    let n = x.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the gated rectangle's element count", lanes)?;
    ctx.fire(
        OP,
        Fire::at(
            "linear/glu.cuh",
            "::pie::linear::gate_sigmoid_mul<::pie::bf16>",
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[x.arg(), gate.arg(), stated(OP, lanes)?.arg()],
    )
}
