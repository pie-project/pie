//! The PTIR substrate's own kernels -- the ones the tensor-compiler's emitted
//! shader text cannot produce because they predate a region, so they are
//! declared here by hand rather than lowered.

use kernels_macros::routine;
use kernels::routine::{Refusal};

use crate::routine::{Asks, Bind, Ctx, Fire, In, Out, Tensor, bf16, keys};

/// The PTIR logits stage, GPU-side.
///
/// Dark on this backend and crossed for that reason — see
/// [`crate::sample::argmax_logits`]. The VOCABULARY was a fact its row had no
/// column for, and no text names this symbol.
///
/// # The x extent is words, not elements
///
/// `logits_copy.wgsl` packs two bf16 into a `u32` and guards
/// `2u * x + 1u >= p.vocab`, so one lane owns one WORD and the grid is half as
/// wide as the vocabulary. `kernels-metal` states `lanes: [vocab, rows, 1]`
/// for the same kernel and is right about its own shader: MSL has a 16-bit
/// type and WGSL has none. The signatures agree; the grids differ.
///
/// # Errors
///
/// [`Refusal::Empty`] for no rows or an empty vocabulary, and
/// [`Refusal::Narrow`] for an odd one: an odd pitch starts the next row inside
/// the previous row's last word.
#[routine]
pub fn copy_logits_bf16(
    ctx: &Ctx<'_>,
    source: In<Tensor<bf16>>,
    destination: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let vocab = source.width.unsigned_abs();
    let rows = ctx.ask::<u32, keys::Rows>()?;
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
        &[source.arg(), destination.arg(), params],
    )
}
