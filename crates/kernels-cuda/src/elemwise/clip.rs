use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/clip.cuh";

const BLOCK: u32 = 256;

pub fn clamp(ctx: &Ctx, lo: f32, hi: f32, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(lo <= hi) {
        return Err(refuse(
            OP,
            format!("the bounds {lo} and {hi} cross, and a clamp between them is the constant {hi}"),
        ));
    }
    let n = x.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the element count", lanes)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::clamp<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            lo.arg(),
            hi.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn clamp_learned(
    ctx: &Ctx,
    lo: Tensor,
    hi: Tensor,
    x: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp_learned";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    for (what, bound) in [("lower", lo), ("upper", hi)] {
        if bound.dtype != x.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is {:?} and the rows it clamps are {:?}; a learned bound \
                     rides the activation's element",
                    bound.dtype, x.dtype
                ),
            ));
        }
        if bound.elements() != 1 {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is a {} x {} plane, and this clamp reads one scalar",
                    bound.rows, bound.width
                ),
            ));
        }
    }
    let n = x.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the element count", lanes)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::clamp_learned<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            lo.arg(),
            hi.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}
