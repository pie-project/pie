use model_compiler::{ArenaMap, Placement};
use model_ir::{Dtype, ValueId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub offset: u64,
    pub bytes: u64,
    pub rows: u32,
    pub width: u32,
    pub dtype: Dtype,
}

#[must_use]
pub fn rect(map: &ArenaMap, value: ValueId, rows: model_compiler::FireRows) -> Option<Rect> {
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
    let rows = expr.at(rows);
    let element = model_compiler::arena::elem_bytes(*dtype);
    Some(Rect {
        offset: *offset,
        bytes: element.map_or(*bytes, |element| {
            rows.saturating_mul(*width).saturating_mul(element)
        }),
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
        let trace = models::sku(SKU)
            .expect("the catalog ships the smoke's SKU")
            .trace;
        let trace = trace(Platform::Cuda);
        let compiled = compile(&trace, &Budget::new(4, 64), &DeviceProfile::default())
            .expect("the smoke's SKU bakes");
        (trace, compiled)
    }

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
    fn arena_every_case() {
        every_op_output_resolves_and_nothing_else_does();
        the_carve_fits_the_allocation_it_asks_for();
    }

    fn every_op_output_resolves_and_nothing_else_does() {
        let (trace, compiled) = compiled();
        let slots = rects(&compiled.arena, 13, 2);

        assert_eq!(slots.len(), trace.values.len(), "one row per trace value");
        for (at, decl) in trace.values.iter().enumerate() {
            let bound = slots[at].is_some();
            match &decl.def {
                Def::Op(_) | Def::Merge(_) => {
                    let host = matches!(decl.ty, Ty::Struct(_));
                    assert_eq!(
                        bound,
                        !host,
                        "value {at} defines {:?} and the table {} it",
                        decl.ty,
                        if bound { "binds" } else { "leaves" }
                    );
                }
                Def::Input(_) | Def::Weight(_) | Def::Cache(_) => {
                    assert!(!bound, "value {at} is not the arena's to bind");
                }
            }
        }
    }

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
