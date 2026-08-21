use crate::routine::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use kernels::routine::Refusal;
use kernels_macros::routine;

#[routine]
pub fn copy_logits_bf16(
    ctx: &Ctx<'_>,
    source: In<Tensor<bf16>>,
    destination: Out<Tensor<bf16>>,
    records: In<Tensor<u32>>,
    rows: Const<u32>,
) -> Result<(), Refusal> {
    let vocab = source.width.unsigned_abs();
    let rows = *rows;
    if rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if vocab == 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    ctx.fire(
        Fire::at(
            crate::routine::module_path("copy_logits_bf16", ctx.best()),
            "copy_logits_bf16",
        )
        .apply([vocab, rows, 1]),
        &[source.arg(), destination.arg(), records.arg()],
    )
}
