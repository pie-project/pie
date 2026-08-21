use kernels_macros::routine;
use kernels::routine::{Refusal};

use crate::routine::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};

#[routine]
pub fn argmax_logits(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    next_token: Out<Tensor<u32>>,
    params: In<Tensor<bf16>>,
    eos_flag: Out<Tensor<u32>>,
    rows: Const<u32>) -> Result<(), Refusal> {
    let rows = *rows;
    if rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }

    ctx.fire(
        Fire::at("sample/argmax.wgsl", "argmax_logits_bfloat16").apply([256, rows, 1]),
        &[logits.arg(), next_token.arg(), params.arg(), eos_flag.arg()],
    )
}
