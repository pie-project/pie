use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/rel_bias.cuh";

const BLOCK: u32 = 256;

pub fn rel_bias(
    ctx: &Ctx,
    x: Tensor,
    w: Tensor,
    heads: u32,
    d_rel: u32,
    extent: u32,
    bias: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.rel_bias";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let rows = nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "heads", heads)?;
    nonzero(OP, "the relative feature width", d_rel)?;
    nonzero(OP, "the relative extent", extent)?;
    if x.width != heads * d_rel {
        return Err(refuse(
            OP,
            format!(
                "the relative features are {} wide and the statement names {heads} x {d_rel}",
                x.width
            ),
        ));
    }
    if w.dtype != x.dtype || w.rows != d_rel || w.width != extent {
        return Err(refuse(
            OP,
            format!(
                "the profile bank is [{}, {}] {:?} and the statement names [{d_rel}, {extent}] {:?}",
                w.rows, w.width, w.dtype, x.dtype
            ),
        ));
    }
    if bias.dtype != Dtype::F32 || bias.rows != x.rows || bias.width != heads * extent {
        return Err(refuse(
            OP,
            format!(
                "the bias lands [{}, {}] {:?} and the statement names [{}, {heads} x {extent}] f32",
                bias.rows, bias.width, bias.dtype, x.rows
            ),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::rel_bias<{t}>"))).apply(Launch::grid(
            [extent.div_ceil(BLOCK), heads, rows],
            [BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            w.arg(),
            bias.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, d_rel)?.arg(),
            stated(OP, extent)?.arg(),
            ctx.stage(),
        ],
    )
}
