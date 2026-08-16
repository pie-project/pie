//! The PTIR substrate's own kernels -- the ones the tensor-compiler's
//! emitted shader text cannot produce because they predate a region.

use kernels::KernelSig;
use kernels::routine::{Env, Refusal};

use crate::routine::{Bind, Buf, BufMut, Ctx, Fire, Routine};

/// EMPTY: this family's row has been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. The row was never filled — no operands,
/// no launch rule, no axes — because this backend's channel-plane interpreter
/// never dispatches the kernel. It recorded what the shader tree CONTAINS
/// rather than what this driver fires, and now
/// `driver-wgpu::lowering::arm::copy_logits_bf16` records that instead.
///
/// The kernel stages logits rows GPU-side, and its own header records why it
/// was written — a sixteen-request fire paid sixteen command-buffer round
/// trips per token to move sixteen vocabulary rows, about 3ms of a 23.5ms
/// step. That problem does not exist here: Apple silicon shares physical
/// memory, so `pipeline::step::PassInputs` takes the read-out as a BORROWED
/// host slice, and "copying it per fire would be pure waste."
pub static KERNELS: &[KernelSig] = &[];

/// The entrypoints this family's routines spell, now that its row is gone.
///
/// See [`crate::sample::ENTRYPOINTS`] for why a retired family has to state
/// these: `entrypoints()` is what every sweep walks, and one of them compiles
/// each name on a real adapter.
pub static ENTRYPOINTS: &[&str] = &["copy_logits_bf16"];

/// The PTIR logits stage, GPU-side.
///
/// Dark on this backend and crossed for that reason — see
/// [`crate::sample::argmax_logits`]. Its row is emptier still: no operands, no
/// launch rule and no axes, so the VOCABULARY was a fact the table had no
/// column for. It would have had to arrive as a `grid_param` pointing into
/// some text's scalars, and no text names this symbol.
///
/// # The x extent is words, not elements
///
/// `logits_copy.wgsl` packs two bf16 into a `u32` and guards
/// `2u * x + 1u >= p.vocab`, so one lane owns one WORD and the grid is half as
/// wide as the vocabulary. `kernels-metal` states `lanes: [vocab, rows, 1]`
/// for the same kernel and is right about its own shader: MSL has a 16-bit
/// type and WGSL has none. The signatures agree; the grids differ, which is
/// exactly the split `refactor-bigplan.md` §2 draws — the argument list is a
/// fact about the kernel and the grid is a fact about the shader.
///
/// # Errors
///
/// [`Refusal::Empty`] for no rows or an empty vocabulary, and
/// [`Refusal::Narrow`] for an odd one: the shader's own header requires
/// `vocab` to be even, because an odd pitch starts the next row inside the
/// previous row's last word.
pub fn copy_logits_bf16(
    ctx: &Ctx,
    source: Buf,
    destination: BufMut,
    params: Buf,
    vocab: Env<u32>,
    rows: Env<u32>,
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
