use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/pointwise.cuh";

const BLOCK: u32 = 256;

pub fn silu(ctx: &Ctx, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire(ctx, "elementwise.silu", "0", x, o)
}

pub fn tanh(ctx: &Ctx, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire(ctx, "elementwise.tanh", "1", x, o)
}

pub fn gelu_tanh(ctx: &Ctx, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    crate::linear::mlp::gelu_tanh(ctx, x, 1, o)
}

fn fire(ctx: &Ctx, op: &'static str, stamp: &str, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    let t = dtype_dispatch!(op, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    if x.dtype != o.dtype || x.rows < o.rows || x.width != o.width {
        return Err(refuse(
            op,
            format!(
                "the operand is {} x {} {:?} and the answer is {} x {} {:?}; an activation \
                 does not reshape",
                x.rows, x.width, x.dtype, o.rows, o.width, o.dtype
            ),
        ));
    }
    let n = o.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(op, "the element count", lanes)?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::activation<{t}, {stamp}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            o.arg(),
            stated(op, lanes)?.arg(),
            stated(op, o.width)?.arg(),
            ctx.stage(),
        ],
    )
}
