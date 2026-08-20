//! Sampling. One kernel: the device argmax with its EOS compare.
//!
//! The family table is empty — `driver-wgpu` arms this symbol directly — so
//! the buffer order lives only in the shader's `@binding` numbering.

use kernels_macros::routine;
use kernels::routine::{Refusal};

use crate::routine::{Asks, Bind, Ctx, Fire, In, Out, Tensor, bf16, keys};

/// `sample/argmax.wgsl` — `@compute @workgroup_size(256)`.
///
/// Load-bearing twice: the reduction's stride over the vocabulary AND the x
/// extent of the grid, since one workgroup reduces one whole row.
const GROUP_X: u32 = 256;

/// The device argmax, with its EOS compare.
///
/// **No text names this symbol on this backend**: its row stated no operands
/// and no launch rule, so `driver-wgpu` refuses it as `Unstated`. The four
/// buffers and their order are stated in the SHADER's `@binding(0..=3)`
/// order, which is the only statement of that order anywhere.
///
/// # Errors
///
/// [`Refusal::Empty`] when there are no rows to reduce.
#[routine]
pub fn argmax_logits(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    next_token: Out<Tensor<u32>>,
    params: In<Tensor<bf16>>,
    eos_flag: Out<Tensor<u32>>) -> Result<(), Refusal> {
    let rows = ctx.ask::<u32, keys::Rows>()?;
    if rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    // `wg.y` is the row, and x carries the one workgroup that reduces it.
    ctx.fire(
        Fire::at("sample/argmax.wgsl", "argmax_logits_bfloat16").apply([GROUP_X, rows, 1]),
        &[logits.arg(), next_token.arg(), params.arg(), eos_flag.arg()],
    )
}
