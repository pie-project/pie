use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};

#[routine]
pub fn copy_logits_bf16(
    ctx: &Ctx<'_>,
    source: In<Tensor<bf16>>,
    destination: Out<Tensor<bf16>>,
    records: In<Tensor<u32>>,
    rows: Const<u32>) -> Result<(), Refusal> {
    let vocab = source.width.unsigned_abs();
    let rows = *rows;
    if rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if vocab == 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    ctx.fire(
        Fire::at("ptir/logits_copy.metal", "copy_logits_bf16").apply(Grid::of([vocab, rows, 1], [256, 1, 1])),
        &[source.arg(), destination.arg(), records.arg()],
    )
}
