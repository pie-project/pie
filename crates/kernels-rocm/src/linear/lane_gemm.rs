use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/lane_gemm.cuh";

const BLOCK: u32 = 256;

fn element(op: &'static str, what: &str, dtype: Dtype) -> Result<&'static str, Error> {
    match dtype {
        Dtype::F32 => Ok("float"),
        Dtype::Bf16 => Ok("::pie::bf16"),
        Dtype::F16 => Ok("::pie::f16"),
        other => Err(refuse(
            op,
            format!("the {what} is {other:?}, and this projection reads f32, bf16 or f16"),
        )),
    }
}

pub fn act_x_wt(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    let ta = element(op, "activation", act.dtype)?;
    let tw = element(op, "weight", w.dtype)?;
    let ty = element(op, "result", y.dtype)?;
    if y.rows == 0 {
        return Ok(());
    }
    let n = nonzero(op, "the columns this projection lands", y.width)?;
    let k = nonzero(op, "the contraction this projection walks", act.width)?;
    if w.width != k || w.rows != n {
        return Err(refuse(
            op,
            format!(
                "the weight is {} x {} and this projection contracts {k} into {n}",
                w.rows, w.width
            ),
        ));
    }
    if act.rows < y.rows {
        return Err(refuse(
            op,
            format!(
                "the activation has {} rows and the result asks for {}",
                act.rows, y.rows
            ),
        ));
    }
    let warps = BLOCK / 32;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::lane_gemm<{ta}, {tw}, {ty}>")),
        )
        .apply(Launch::grid([n.div_ceil(warps), y.rows, 1], [BLOCK, 1, 1])),
        &[
            act.arg(),
            w.arg(),
            y.arg(),
            stated(op, y.rows)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )
}
