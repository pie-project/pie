//! Sampling. One kernel: the device argmax with its EOS compare.
//!
//! The family table is empty — `driver-wgpu` arms this symbol directly — so
//! the buffer order lives only in the shader's `@binding` numbering.

use kernels::KernelSig;
use kernels::routine::{Refusal};

use crate::routine::{keys, Ask, Bind, Buf, BufMut, Ctx, Fire, Routine};
use crate::routine::{InSlot, OutSlot};

/// EMPTY: this family's rows have been RETIRED — `driver-wgpu`'s
/// `lowering::arm::argmax_logits` is this kernel's arm. The static stays
/// because `lib.rs` sums the family tables into `KERNELS`.
pub static KERNELS: &[KernelSig] = &[];

/// The entrypoints this family's routines spell, now that its rows are gone.
///
/// `entrypoints()` is what the sweeps walk, and one compiles every name on a
/// real adapter; a name stated nowhere is one that sweep stops building.
pub static ENTRYPOINTS: &[&str] = &["argmax_logits_bfloat16"];

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
pub fn argmax_logits(
    ctx: &Ctx,
    logits: InSlot<0, Buf>,
    next_token: OutSlot<0, BufMut>,
    params: InSlot<1, Buf>,
    eos_flag: OutSlot<1, BufMut>,
    rows: Ask<keys::Rows, u32>,
) -> Result<(), Refusal> {
    if *rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    // `wg.y` is the row, and x carries the one workgroup that reduces it.
    ctx.dispatch(
        Fire {
            module: "sample/argmax.wgsl",
            entrypoint: "argmax_logits_bfloat16",
            lanes: [GROUP_X, *rows, 1],
        },
        &[logits.v(), next_token.v(), params.v(), eos_flag.v()],
    )
}

/// This family's routines.
pub static ROUTINES: &[Routine] = &[crate::routine!(argmax_logits)];
