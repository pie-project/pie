use kernels::plane::Refusal;

use crate::plane::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};

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
    if !(vocab).is_multiple_of(2) {
        return Err(Refusal::Narrow {
            what: "vocab",
            at: i64::from(vocab),
        });
    }
    ctx.fire(
        Fire::at("ptir/logits_copy.wgsl", "copy_logits_bf16").apply([vocab / 2, rows, 1]),
        &[source.arg(), destination.arg(), records.arg()],
    )
}
