use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const BLOCK: u32 = 256;

pub fn sigmoid_mul(ctx: &Ctx, gate: Tensor, fan: u32, x: &mut Tensor) -> Result<(), Error> {
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
        &[
            x.arg(),
            gate.arg(),
            stated(OP, lanes)?.arg(),
            stated(OP, x.width)?.arg(),
            stated(OP, fan)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn sigmoid_mul_heads(
    ctx: &Ctx,
    gate: Tensor,
    head_dim: u32,
    scale: f32,
    x: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gate_sigmoid_mul_heads";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    debug_assert_eq!(gate.dtype, x.dtype, "the gate rides the rectangle's dtype");
    let rows = nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "the head width this gate states", head_dim)?;
    if x.width == 0 || !x.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide row is not a whole number of {head_dim}-wide heads",
                x.width
            ),
        ));
    }
    let heads = x.width / head_dim;
    if gate.width != heads || gate.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "the gate plane is {} x {}, and this gate reads one logit per head ({heads}) \
                 for each of {rows} rows",
                gate.rows, gate.width
            ),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(
            "linear/glu.cuh",
            "::pie::linear::gate_sigmoid_mul_heads<::pie::bf16>",
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            x.arg(),
            gate.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            scale.arg(),
            ctx.stage(),
        ],
    )
}
