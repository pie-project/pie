//! The PTIR substrate's own kernels -- the ones the tensor-compiler's emitted
//! shader text cannot produce because they predate a region, so they are
//! declared here by hand rather than lowered.

use kernels::KernelSig;
use kernels::routine::{Refusal};

use crate::routine::{keys, Ask, Bind, Block, Buf, BufMut, Ctx, Fire, Routine};
use crate::routine::{InSlot, OutSlot};

/// EMPTY: this family's row has been RETIRED.
///
/// The row was never filled — no operands, no launch rule, no axes — because
/// this backend's channel-plane interpreter never dispatches the kernel. It
/// recorded what the shader tree CONTAINS rather than what this driver fires,
/// and `driver-wgpu::lowering::arm::copy_logits_bf16` records that instead.
pub static KERNELS: &[KernelSig] = &[];

/// The entrypoints this family's routines spell, now that its row is gone.
///
/// See [`crate::sample::ENTRYPOINTS`]: `entrypoints()` is what every sweep
/// walks, and one of them compiles each name on a real adapter.
pub static ENTRYPOINTS: &[&str] = &["copy_logits_bf16"];

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
pub fn copy_logits_bf16(
    ctx: &Ctx,
    source: InSlot<0, Buf>,
    destination: OutSlot<0, BufMut>,
    params: Block<Buf>,
    vocab: Ask<keys::Width, u32>,
    rows: Ask<keys::Rows, u32>,
) -> Result<(), Refusal> {
    if *rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if *vocab == 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    if !(*vocab).is_multiple_of(2) {
        return Err(Refusal::Narrow {
            what: "vocab",
            at: i64::from(*vocab),
        });
    }
    ctx.dispatch(
        Fire {
            module: "ptir/logits_copy.wgsl",
            entrypoint: "copy_logits_bf16",
            lanes: [*vocab / 2, *rows, 1],
        },
        &[source.v(), destination.v(), params.v()],
    )
}

/// This family's routines.
pub static ROUTINES: &[Routine] = &[crate::routine!(copy_logits_bf16)];
