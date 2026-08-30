//! The activation arena's arithmetic: which rectangle a value resolves to at
//! one fire's shape.
//!
//! **THE OFFSETS ARE NOT THIS CRATE'S.** P7 carved them once
//! (`model_compiler::arena`), giving values whose lives do not overlap the
//! same bytes on purpose, and what is left for a shell is one addition —
//! `base + offset` on CUDA, a bounds-checked `Handles::bind` on Metal. Only
//! the LENGTH moves with the fire, and it moves by the row expression the
//! carve already wrote down, which is the whole reason a composition never
//! triggers a re-anything.
//!
//! **WHAT THE TWO SHELLS SHARED IS EXACTLY [`rect`]**, and what they did not
//! share is one line each: CUDA turns the offset into a `u64` device pointer,
//! Metal turns the `(offset, bytes)` pair into a handle row. Both wrappers
//! stayed in their shells because a view is a device noun on one plane and an
//! address on the other; the slot × bucket → rectangle arithmetic is here.

use model_compiler::{ArenaMap, Placement};
use model_ir::{Dtype, ValueId};

/// One value's rectangle in the arena, before it is a pointer or a handle.
///
/// The arithmetic half of a shell's carve, kept separate because it is the
/// half that is true with no device present — and therefore the half a test
/// can hold to account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Bytes from the arena's base to this rectangle's first element.
    pub offset: u64,
    /// What THIS fire's rows occupy, which is what a handle is
    /// bounds-checked at — never the slot's largest-fire reservation.
    pub bytes: u64,
    /// This fire's row count for the value.
    pub rows: u32,
    /// The row width the carve gave it.
    pub width: u32,
    /// The element it is written in.
    pub dtype: Dtype,
}

/// The rectangle `value` resolves to at this fire's shape, or `None` for a
/// value the arena does not bind.
#[must_use]
pub fn rect(map: &ArenaMap, value: ValueId, rows: model_compiler::FireRows) -> Option<Rect> {
    // An alias IS its root's rectangle — a merge's arms write disjoint row
    // windows of the merged column, and an in-place result is its operand —
    // so the root is followed here rather than at every read.
    let root = map.root(value);
    let Some(Placement::Arena {
        offset,
        bytes,
        rows: expr,
        width,
        dtype,
        ..
    }) = map.placements.get(root.0 as usize)
    else {
        return None;
    };
    // **THE SECOND AXIS'S COUNTS, CARRIED AND NOT DEFAULTED.** `RowExpr::at`
    // takes a whole fire because a door that defaulted the patch pair sizes
    // every tower rectangle at NOTHING and hands back zero bytes — which does
    // not fault, it computes, and the failure arrives a launch later as
    // `the activation's rows are the rows the result lands` from a GEMM whose
    // destination has no rows.
    //
    // This door used to state `FireRows::text_only` itself, on the reading
    // that its callers are the two shells' STORE arithmetic and a tower
    // rectangle would be asked for through `Composition::value_window`
    // instead. It is not: `engine_cuda::run::Run::whole` resolves EVERY arena
    // value through the slot table this builds, tower rectangles included. So
    // the counts come from the caller, who is the only one that knows them,
    // and a text-only shell says so with `FireRows::text_only`.
    let rows = expr.at(rows);
    let element = model_compiler::arena::elem_bytes(*dtype);
    Some(Rect {
        offset: *offset,
        // The fire's own extent when the element is known, and the slot's
        // whole reservation when it is not — a packed storage element with no
        // byte size is a rectangle a shell cannot measure, and claiming less
        // than the carve reserved would refuse a legal bind.
        bytes: element.map_or(*bytes, |element| rows.saturating_mul(*width).saturating_mul(element)),
        rows: u32::try_from(rows).unwrap_or(u32::MAX),
        width: u32::try_from(*width).unwrap_or(u32::MAX),
        dtype: *dtype,
    })
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

    /// The carve, without the minting — every value's rectangle at one shape.
    fn rects(map: &ArenaMap, tokens: u64, lanes: u64) -> Vec<Option<Rect>> {
        (0..map.placements.len())
            .map(|at| {
                rect(
                    map,
                    ValueId(at as u32),
                    model_compiler::FireRows::text_only(tokens, lanes),
                )
            })
            .collect()
    }

    #[test]
    fn every_op_output_resolves_and_nothing_else_does() {
        let (trace, compiled) = compiled();
        let slots = rects(&compiled.arena, 13, 2);

        assert_eq!(slots.len(), trace.values.len(), "one row per trace value");
        for (at, decl) in trace.values.iter().enumerate() {
            let bound = slots[at].is_some();
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
        let small = rects(&compiled.arena, 1, 1);
        let large = rects(&compiled.arena, 64, 4);

        let mut moved = 0;
        for (a, b) in small.iter().zip(&large) {
            let (Some(a), Some(b)) = (a, b) else { continue };
            // A shell compares the two POINTERS or the two handle rows it
            // minted; the claim underneath both is made here, of the offset
            // each is minted at.
            assert_eq!(a.offset, b.offset, "an offset is static across every bucket");
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

        let logits = rect(
            &compiled.arena,
            out,
            model_compiler::FireRows::text_only(7, 1),
        )
        .expect("the out seam is an arena rectangle");
        assert_eq!(logits.rows, 7, "one row of logits per token row");
        assert_eq!(logits.width, 248_320, "the SKU's vocabulary");
        assert_eq!(logits.dtype, model_ir::Dtype::Bf16);
    }

    #[test]
    fn the_carve_fits_the_allocation_it_asks_for() {
        let (_, compiled) = compiled();
        for rect in rects(&compiled.arena, 64, 4).iter().flatten() {
            let end = rect.offset + rect.bytes;
            assert!(
                end <= compiled.arena.bytes,
                "a rectangle ending at {end} in an arena of {} bytes",
                compiled.arena.bytes
            );
        }
    }
}
