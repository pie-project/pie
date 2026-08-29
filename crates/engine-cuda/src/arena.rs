//! The activation arena: one allocation for the model's whole load, and the
//! per-fire table that turns the compiler's static offsets into handles.
//!
//! **THE OFFSETS ARE NOT THIS CRATE'S.** P7 carved them once
//! (`model_compiler::arena`), giving values whose lives do not overlap the
//! same bytes on purpose, and what is left for a shell is one addition:
//! `base + offset`. Only the LENGTH moves with the fire, and it moves by the
//! row expression the carve already wrote down — which is the whole reason a
//! composition never triggers a re-anything.
//!
//! # Why the table is rebuilt every fire, and why that is cheap
//!
//! [`SlotTable`] is `ValueId`-indexed and every row is a `Copy` handle, so
//! rebuilding it is one pass of arithmetic over the plan's values — 855 of
//! them for the smoke's SKU — with no allocation past the `Vec` itself. The
//! alternative, mutating a resident table in place, would save that pass and
//! cost the property that makes it easy to reason about: a `Run` borrows a
//! table that describes exactly one fire.
//!
//! # A row count here is THE FIRE'S, and that is what makes windows work
//!
//! Every rectangle in this table spans the whole fire: a `Dim::Tokens` value
//! gets one column of `rows` rows, indexed by ABSOLUTE fire row. That is not
//! a simplification waiting to be narrowed — it is the substrate design §0's
//! window-split stands on. A merge lowers to "the arms write disjoint row
//! ranges of one buffer", which is a sentence about one column with two
//! writers, and the compiler's carve aliases the arms onto exactly that
//! column ([`arena`](model_compiler::arena)'s union-find through `root`).
//!
//! So the window is applied by the READER, not by the carve:
//! [`Run::tensor`] cuts each operand to the window of the node asking for it
//! ([`window`](crate::window)), which is what lets a decode node take rows
//! `[10,13)` of the same `q` a prefill node takes `[0,10)` of. A per-value
//! span would not answer that, because `Value::split` refines a cond and
//! hands back the SAME `ValueId`.
//!
//! [`Run::tensor`]: crate::run::Run
//! [`SlotTable`]: crate::run::SlotTable

use engine::store::arena::rect;
use kernels_cuda::Tensor;
use model_compiler::ArenaMap;
use model_ir::ValueId;

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::run::SlotTable;

/// The device bytes every arena rectangle is an offset into.
#[derive(Debug)]
pub struct Arena {
    store: Buffer,
}

impl Arena {
    /// Reserve what the carve says the largest admissible fire needs.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the allocation, [`Fault::Runtimeless`] for a
    /// build with no runtime.
    pub fn reserve(map: &ArenaMap) -> Result<Arena> {
        Ok(Arena {
            store: Buffer::zeroed(usize::try_from(map.bytes).unwrap_or(usize::MAX))?,
        })
    }

    /// The base address.
    #[must_use]
    pub fn base(&self) -> u64 {
        self.store.ptr()
    }

    /// How many bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() as u64
    }

    /// The table a fire of `tokens` rows over `lanes` requests resolves
    /// through.
    #[must_use]
    pub fn slots(&self, map: &ArenaMap, tokens: u64, lanes: u64) -> SlotTable {
        carve(self.base(), map, tokens, lanes)
    }

    /// Copy `into.len()` bytes back from an address inside this arena.
    ///
    /// The one read a fire ends with: the `"out"` seam's rows, whose slot the
    /// carve deliberately holds open past the last node so that nothing
    /// shares its bytes with the reader that has not run yet.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for an address outside the arena, [`Fault::Device`]
    /// for the copy.
    pub fn read(&self, at: u64, into: &mut [u8]) -> Result<()> {
        let base = self.base();
        let offset = at.checked_sub(base).ok_or(Fault::Ceiling {
            what: "bytes into the arena",
            need: at,
            have: base,
        })?;
        self.store.read(offset, into)
    }
}

/// One fire's slot table, as pure arithmetic over a base address.
///
/// Separated from [`Arena`] so it can be exercised with no device in the
/// room: the interesting property — that every value resolves to the
/// rectangle its root names, at the row count this fire has — is arithmetic,
/// and a test that needed a GPU to check it would not be run.
///
/// **THE RECTANGLE IS [`engine::store::arena::rect`]'s AND THE ADDITION IS
/// THIS SHELL'S**, which is the whole of what separated this function from
/// its Metal twin. The offsets, the row expressions and the aliasing through
/// `root` were the same arithmetic written twice; what is left here is the
/// one line a CUDA view is — `base + offset`, a `u64` a kernel dereferences,
/// where the Metal plane mints a bounds-checked handle row instead.
#[must_use]
pub fn carve(base: u64, map: &ArenaMap, tokens: u64, lanes: u64) -> SlotTable {
    let mut rows: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);
        rows.push(rect(map, value, tokens, lanes).map(|rect| {
            Tensor::new(base + rect.offset, rect.rows, rect.width, rect.dtype)
        }));
    }
    SlotTable(rows)
}

#[cfg(test)]
mod tests {
    use model_compiler::{Budget, DeviceProfile, compile};
    use model_dsl::Platform;
    use model_ir::{Def, Ty};

    use super::*;

    const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

    fn compiled() -> (model_ir::Trace, model_compiler::CompiledModel) {
        let trace = model::trace_of(SKU).expect("the catalog ships the smoke's SKU");
        let trace = trace(Platform::Cuda);
        let compiled = compile(&trace, &Budget::new(4, 64), &DeviceProfile::default())
            .expect("the smoke's SKU bakes");
        (trace, compiled)
    }

    #[test]
    fn every_op_output_resolves_and_nothing_else_does() {
        let (trace, compiled) = compiled();
        let slots = carve(1 << 20, &compiled.arena, 13, 2);

        assert_eq!(slots.0.len(), trace.values.len(), "one row per trace value");
        for (at, decl) in trace.values.iter().enumerate() {
            let bound = slots.0[at].is_some();
            match &decl.def {
                // A φ resolves like the op output it merges: the compiler
                // aliased every arm onto one column.
                Def::Op(_) | Def::Merge(_) => {
                    let host = matches!(decl.ty, Ty::Struct(_));
                    assert_eq!(
                        bound, !host,
                        "value {at} defines {:?} and the table {} it",
                        decl.ty,
                        if bound { "binds" } else { "leaves" }
                    );
                }
                // Inputs, weights and caches are the engine's, the loader's
                // and the pool's; the arena holds none of them.
                Def::Input(_) | Def::Weight(_) | Def::Cache(_) => {
                    assert!(!bound, "value {at} is not the arena's to bind");
                }
            }
        }
    }

    #[test]
    fn a_rectangle_grows_with_the_fire_and_its_offset_does_not() {
        let (_, compiled) = compiled();
        let small = carve(0, &compiled.arena, 1, 1);
        let large = carve(0, &compiled.arena, 64, 4);

        let mut moved = 0;
        for (a, b) in small.0.iter().zip(&large.0) {
            let (Some(a), Some(b)) = (a, b) else { continue };
            assert_eq!(a.ptr, b.ptr, "an offset is static across every bucket");
            assert_eq!(a.width, b.width, "only the rows move");
            if a.rows != b.rows {
                moved += 1;
            }
        }
        assert!(
            moved > 0,
            "a plan of token-shaped values whose rows never move is a carve \
             that read every dim as a constant"
        );
    }

    #[test]
    fn the_out_seam_lands_the_vocabulary_at_one_row_per_token() {
        let (trace, compiled) = compiled();
        let out = trace
            .seams
            .iter()
            .find(|seam| seam.seam == "out")
            .and_then(|seam| seam.values.first().copied())
            .expect("every traced plan carries an out seam");

        let slots = carve(0, &compiled.arena, 7, 1);
        let logits = slots.0[out.0 as usize].expect("the out seam is an arena rectangle");
        assert_eq!(logits.rows, 7, "one row of logits per token row");
        assert_eq!(logits.width, 248_320, "the SKU's vocabulary");
        assert_eq!(logits.dtype, model_ir::Dtype::Bf16);
    }

    #[test]
    fn the_carve_fits_the_allocation_it_asks_for() {
        let (_, compiled) = compiled();
        let slots = carve(0, &compiled.arena, 64, 4);
        for handle in slots.0.iter().flatten() {
            let end = handle.ptr
                + handle.elements()
                    * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or(0);
            assert!(
                end <= compiled.arena.bytes,
                "a rectangle ending at {end} in an arena of {} bytes",
                compiled.arena.bytes
            );
        }
    }
}
