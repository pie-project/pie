use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::Tensor;

const BLOCK: u32 = 256;

pub fn split_q_b(
    ctx: &Ctx,
    q_b: Tensor,
    heads: u32,
    nope_dim: u32,
    rope_dim: u32,
    q_nope: &mut Tensor,
    q_pe: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_split_q_b";
    let t = dtype_dispatch!(OP, q_b.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = count(OP, "the head count this cut states", heads)?;
    let nope = stated(OP, nope_dim)?;
    let rope = stated(OP, rope_dim)?;
    let width = i64::from(heads) * (i64::from(nope) + i64::from(rope));
    // Eight elements a thread when both widths are whole eights.
    let vec = if nope % 8 == 0 && rope % 8 == 0 { 8 } else { 1 };
    let total = i64::from(q_b.rows) * width / vec;
    let total = i32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} split elements do not fit the kernel's int"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            "attn/mla.hip",
            &format!("::pie::attn::mla_split_q_b<{t}, {vec}>"),
        ))
        .apply(Launch::flat(total.unsigned_abs(), BLOCK)),
        &[
            q_b.arg(),
            q_nope.arg(),
            q_pe.arg(),
            total.arg(),
            heads.arg(),
            nope.arg(),
            rope.arg(),
            ctx.stage(),
        ],
    )
}
