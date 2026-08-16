//! The PTIR substrate's own kernels — the ones the tensor-compiler's emitted
//! MSL cannot produce because they predate a region.
//!
//! Crossed second, after [`crate::sample`], for the same two reasons: one
//! kernel, and one this backend never dispatches, so the crossing cannot
//! change what any model computes.

use kernels::routine::Refusal;

use crate::routine::{keys, Ask, Bind, Block, Buf, BufMut, Ctx, Fire, Routine};
use crate::routine::{InSlot, OutSlot};

/// Threads per threadgroup on the vocabulary axis.
///
/// The kernel is a guarded elementwise copy — `if (tid.x >= p.vocab) return;`
/// — so any width is correct and this is the one every other flat elementwise
/// launch on this backend uses (`grid::elementwise_mb`). Stated here because
/// MSL declares no threadgroup and Metal has nothing to reflect it from.
const GROUP_X: u32 = 256;

/// Stage `rows` vocabulary rows, source and destination row per row.
///
/// `ptir/logits_copy.metal`. One dispatch stages every row a fire needs, with
/// `tid.y` selecting which — and that shape is the whole point of the kernel.
/// It used to copy a single row and be submitted as its own command buffer,
/// once per row, so a sixteen-request fire paid sixteen command-buffer round
/// trips per token to move sixteen vocabulary rows: about 3 ms of a 23.5 ms
/// step, scaling linearly with the batch, which is what made the sampler look
/// linear in lanes when the sampler itself is 0.5 ms of GPU.
///
/// `params` is `const device PtirLogitsCopyParams*` — an ARRAY, indexed by
/// `tid.y`, one struct per row. So it is a buffer and not a scalar run, and
/// `rows` has to reach the grid rather than the kernel: nothing in the shader
/// bounds `tid.y`, because the grid is what bounds it.
///
/// # Errors
///
/// [`Refusal::Empty`] when there is no row to stage, or no vocabulary to stage
/// it over. Either would be a dispatch over an empty grid, which runs nothing
/// and reports success — and the caller would then read a destination that
/// still holds the previous token's logits, which is a wrong answer and not a
/// missing one.
pub fn copy_logits_bf16(
    ctx: &Ctx<'_>,
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
    ctx.dispatch(
        Fire {
            entrypoint: "copy_logits_bf16",
            file: "ptir/logits_copy.metal",
            lanes: [*vocab, *rows, 1],
            group: [GROUP_X, 1, 1],
        },
        &[source.v(), destination.v(), params.v()],
    )
}

/// This family's routines.
pub static ROUTINES: &[Routine] = &[crate::routine!(copy_logits_bf16)];

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[("ptir/logits_copy.metal", "copy_logits_bf16")];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    #[derive(Default)]
    struct Seen(RefCell<Vec<(Fire, Vec<ArgValue>)>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// The body states the two-dimensional grid the kernel's `tid.y` needs,
    /// and the three buffers in the shader's binding order.
    ///
    /// The y axis is the interesting half. Nothing in the shader bounds
    /// `tid.y` — it indexes `params[tid.y]` unguarded — so the grid is the
    /// only thing that says how many rows there are. A body that flattened
    /// this to `[vocab * rows, 1, 1]`, which is what `grid::elementwise_mb`
    /// does for every other copy on this backend, would read every row's
    /// parameters out of `params[0]` and stage the same source row `rows`
    /// times.
    #[test]
    fn the_row_axis_is_the_grid_s_and_not_the_kernel_s() {
        let seen = Seen::default();
        copy_logits_bf16(&seen, InSlot::new(Buf(4)), OutSlot::new(BufMut(5)), Block::new(Buf(6)), Ask::new(128_256), Ask::new(16))
            .expect("sixteen rows is a launch");

        let calls = seen.0.borrow();
        assert_eq!(
            calls.len(),
            1,
            "one fire is one dispatch, which is the point"
        );
        let (fire, args) = &calls[0];
        assert_eq!(fire.entrypoint, "copy_logits_bf16");
        assert_eq!(fire.file, "ptir/logits_copy.metal");
        assert_eq!(
            fire.lanes,
            [128_256, 16, 1],
            "the vocabulary on x and the ROWS on y -- flattening the two would \
             index `params[0]` for every row"
        );
        assert_eq!(fire.group, [GROUP_X, 1, 1]);
        assert_eq!(
            args,
            &[
                ArgValue::Buffer(4),
                ArgValue::BufferMut(5),
                ArgValue::Buffer(6)
            ],
            "buffer(0..=2): source, destination, params"
        );
    }

    /// Both extents are refused when zero, and named apart.
    ///
    /// Two refusals rather than one because they are different mistakes: no
    /// rows is an empty fire, and no vocabulary is a model whose head width
    /// never reached the driver. A caller that got `Ok` for either would read
    /// a destination still holding the previous token's logits — a wrong
    /// answer, not a missing one.
    #[test]
    fn an_empty_extent_is_refused_by_name() {
        let seen = Seen::default();
        assert_eq!(
            copy_logits_bf16(&seen, InSlot::new(Buf(0)), OutSlot::new(BufMut(1)), Block::new(Buf(2)), Ask::new(128_256), Ask::new(0)),
            Err(Refusal::Empty { what: "rows" })
        );
        assert_eq!(
            copy_logits_bf16(&seen, InSlot::new(Buf(0)), OutSlot::new(BufMut(1)), Block::new(Buf(2)), Ask::new(0), Ask::new(16)),
            Err(Refusal::Empty { what: "vocab" })
        );
        assert!(seen.0.borrow().is_empty());
    }

    /// The derived row is the signature: two buffers the trace supplies, and
    /// a block and two extents the environment does.
    ///
    /// This said "three buffers the trace supplies" and asserted it, and was
    /// WRONG from the moment `params` became `Env<Buf>` — it could not fail,
    /// because stating the slots on this plane stopped the in-crate tests
    /// compiling and a test that does not build does not disagree with
    /// anything. `params` is `o.params_block()`, which is the environment's
    /// by construction: the statement's scalars, staged by the driver into
    /// one buffer the statement never names.
    ///
    /// `vocab` is the interesting one. The `kernel!` row beside this states no
    /// operands and no launch rule at all, so the vocabulary was a fact the
    /// table could not carry — it would have had to come from a `grid_param`
    /// pointing into the text's scalars. Here it is an argument, and the
    /// signature is where it is written down.
    #[test]
    fn the_derived_row_is_the_signature() {
        let row = &ROUTINES[0];
        assert_eq!(row.name, "copy_logits_bf16");
        assert_eq!(
            row.args,
            &[
                (kernels::Ty::Buf, crate::routine::Supplier::Trace),
                (kernels::Ty::BufMut, crate::routine::Supplier::Trace),
                (kernels::Ty::Buf, crate::routine::Supplier::Env),
                (kernels::Ty::U32, crate::routine::Supplier::Env),
                (kernels::Ty::U32, crate::routine::Supplier::Env),
            ]
        );
    }
}
