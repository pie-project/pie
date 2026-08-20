//! The PTIR substrate's own kernels -- the ones the tensor-compiler's
//! emitted shader text cannot produce because they predate a region.

use kernels_macros::routine;
use crate::routine::{Asks, Bind, Ctx, Fire, In, Out, Tensor, bf16, keys};
use kernels::routine::Refusal;

// This family's `kernel!` row was NOT filled, and that was never an
// oversight: this backend's channel-plane interpreter does not dispatch
// `copy_logits_bf16`, so nothing was ever there to state.
//
// The kernel stages logits rows GPU-side, and its own header records why it
// was written -- a sixteen-request fire paid sixteen command-buffer round
// trips per token to move sixteen vocabulary rows, about 3 ms of a 23.5 ms
// step. That problem does not exist here. Apple silicon shares physical
// memory, so `pipeline::step::PassInputs` takes the read-out as a BORROWED
// host slice: "copying it per fire would be pure waste."
//
// The routine below is crossed anyway, for the same reason the row existed:
// it states what the shader tree CONTAINS, not what this driver fires.

/// Stage logits rows GPU-side, one dispatch for every row a fire needs.
///
/// `ptir/logits_copy.slang`. The y lane picks the per-row params record, which
/// is why `params` here is an ordinary `StructuredBuffer` and not `PIE_PARAMS`:
/// there is one record per ROW and the row count is the dispatch's, so the
/// driver correctly finds no scalar block. `vocab` and `rows` are the grid and
/// nothing else -- the shader reads the vocabulary back out of its own record
/// and guards the x overhang itself.
///
/// The row above is unfilled and stays unfilled. This backend's interpreter
/// never dispatches this kernel, and crossing it changes that not at all; what
/// it does is put the family's one name in both planes, which is what the
/// countdown in `refactor-bigplan.md` §8 counts.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty vocabulary or an empty row count. A
/// zero-lane dispatch is legal Vulkan that runs nothing and reports success,
/// so the caller would read the destination it passed as a staged row.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("copy_logits_bf16", ctx.best()), "copy_logits_bf16").apply([vocab, rows, 1]),
        &[source.arg(), destination.arg(), params],
    )
}
