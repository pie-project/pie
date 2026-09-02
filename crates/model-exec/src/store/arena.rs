//! The activation arena's arithmetic: which rectangle a value resolves to at one fire's shape.
//! Offsets are carved once by `model_compiler::arena`; only the length moves with the fire, by the row expression the carve already wrote down.
//! Shared between shells as [`rect`]; each shell turns the rectangle into its own pointer or handle.

use model_compiler::{ArenaMap, Placement};
use model_ir::{Dtype, ValueId};

/// One value's rectangle in the arena, before it is a pointer or a handle. True with no device present, so a test can hold it to account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Bytes from the arena's base to this rectangle's first element.
    pub offset: u64,
    /// What this fire's rows occupy; what a handle is bounds-checked at, never the slot's largest-fire reservation.
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
    // an alias is its root's rectangle: a merge's arms write disjoint row windows of the merged column.
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
    // `RowExpr::at` takes a whole fire: a door that defaulted the patch pair would size every tower rectangle at nothing and hand back zero bytes silently.
    let rows = expr.at(rows);
    let element = model_compiler::arena::elem_bytes(*dtype);
    Some(Rect {
        offset: *offset,
        // fire's own extent when the element is known; the slot's whole reservation when it's not (an unmeasurable packed element).
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
        let trace = models::sku(SKU).expect("the catalog ships the smoke's SKU").trace;
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
