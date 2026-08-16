//! The PTIR substrate's own kernels -- the ones the tensor-compiler's
//! emitted shader text cannot produce because they predate a region.

use kernels::KernelSig;
use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine};

pub static KERNELS: &[KernelSig] = &[
    // NOT filled, and it is not an oversight: this backend's channel-plane
    // interpreter never dispatches it.
    //
    // The kernel stages logits rows GPU-side, and its own header records why
    // it was written -- a sixteen-request fire paid sixteen command-buffer
    // round trips per token to move sixteen vocabulary rows, about 3ms of a
    // 23.5ms step. That problem does not exist here. Apple silicon shares
    // physical memory, so `pipeline::step::PassInputs` takes the read-out as a
    // BORROWED host slice: "copying it per fire would be pure waste."
    //
    // A row is filled when a text names its symbol. No text names this one and
    // none should, so it stays a declaration of what the shader tree contains
    // rather than of what this driver dispatches.
];

/// The entrypoints this family's routines spell, now that its rows are gone.
///
/// See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &["copy_logits_bf16"];

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
pub fn copy_logits_bf16(
    ctx: &Ctx<'_>,
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
    ctx.dispatch(
        Fire {
            entrypoint: "copy_logits_bf16",
            lanes: [*vocab, *rows, 1],
        },
        &[source.v(), destination.v(), params.v()],
    )
}

/// The crossed rows of this family.
pub static ROUTINES: &[Routine] = &[crate::routine!(copy_logits_bf16)];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    type Call = (String, [u32; 3], Vec<ArgValue>);

    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    /// A lane per vocabulary entry, a row per record, and no scalar pushed.
    ///
    /// The x extent is the FULL vocabulary and not half of it: the wgpu port
    /// of this kernel writes two entries per lane and launches `vocab / 2`,
    /// and the Slang one writes one, so a grid copied across from that crate
    /// would stage the first half of every row and leave the rest holding
    /// whatever the arena held.
    #[test]
    fn a_staging_fires_one_lane_for_each_entry_of_each_row() {
        let seen = Seen::default();
        copy_logits_bf16(&seen, Buf(0), BufMut(1), Buf(2), Env(262_144), Env(16)).unwrap();
        let calls = seen.0.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, [262_144, 16, 1]);
        assert_eq!(calls[0].2.len(), 3, "the row states no scalar block");
    }

    /// An empty grid is refused on either axis, and names which.
    #[test]
    fn an_empty_vocabulary_or_row_count_is_refused() {
        let seen = Seen::default();
        assert!(matches!(
            copy_logits_bf16(&seen, Buf(0), BufMut(1), Buf(2), Env(8), Env(0)),
            Err(Refusal::Empty { what: "rows" })
        ));
        assert!(matches!(
            copy_logits_bf16(&seen, Buf(0), BufMut(1), Buf(2), Env(0), Env(4)),
            Err(Refusal::Empty { what: "vocab" })
        ));
        assert!(seen.0.borrow().is_empty(), "a refused shape dispatched");
    }
}
