//! The capture-unit partition: which regions belong to which exec, derived
//! from the dimension algebra rather than declared. A region's unit is the
//! row axis of the rows it writes; the axis joins the region-coalescing key
//! alongside phase and class mask; a plan whose axes don't form one
//! contiguous run per unit is refused ([`Error::UnitsInterleave`]); and a
//! multi-unit plan can't use the single-graph fold ([`fold_refused`]).

use model_ir::{Operands, RowAxis, Trace, Ty, ValueId};

use crate::compiled::{Phase, Region};
use crate::error::Error;

/// The row axis a node writes, or `None` for a node whose outputs are all
/// fixed blocks or host-owned structs. Outputs, not inputs: the embed-merge
/// node reads patch rows but writes token rows, so it belongs to the token
/// unit — the descriptor row count its launch reads is a token count.
#[must_use]
pub(crate) fn node_axis(
    trace: &Trace,
    at: u32,
    node: &model_ir::Node,
    outs: &mut Vec<ValueId>,
) -> Result<Option<RowAxis>, Error> {
    outs.clear();
    node.op.outputs(outs);
    let mut found = None;
    for value in outs.iter() {
        let Some(decl) = trace.values.get(value.0 as usize) else {
            continue;
        };
        let Ty::Tensor { shape, .. } = &decl.ty else {
            continue;
        };
        let Some(axis) = shape.first().and_then(|dim| dim.axis()) else {
            continue;
        };
        found = Some(match found {
            None => axis,
            Some(held) if held == axis => held,
            Some(_) => return Err(Error::TwoAxes { node: at }),
        });
    }
    Ok(found)
}

/// Every axis this plan states, whether or not any region ended up on it.
/// Read off the values rather than the regions, since a plan may declare a
/// patch-shaped input and compute nothing from it, but a budget with no
/// patch ceiling still has to refuse it.
#[must_use]
pub(crate) fn axes_stated(trace: &Trace) -> Vec<RowAxis> {
    let mut axes = Vec::new();
    for decl in &trace.values {
        let Ty::Tensor { shape, .. } = &decl.ty else {
            continue;
        };
        for dim in shape {
            if let Some(axis) = dim.axis() {
                if !axes.contains(&axis) {
                    axes.push(axis);
                }
            }
        }
    }
    axes.sort_unstable();
    axes
}

/// The axis each region runs on — the first one any of its nodes states, which
/// by the coalescing key is the only one any of them states.

/// The axis of each unit in exec order, and the unit of each region.
///
/// # Errors
///
/// [`Error::UnitsInterleave`] when a unit's capture regions are not one
/// contiguous stretch of the record script.
pub(crate) fn partition(regions: &[Region]) -> Result<(Vec<RowAxis>, Vec<u32>), Error> {
    // Units, in the order their first capture region stands. Prepare
    // regions name no exec (`hoist` puts them all before every capture
    // region) and take unit 0 by construction.
    let mut units: Vec<RowAxis> = Vec::new();
    for region in regions {
        if region.phase != Phase::Capture {
            continue;
        }
        let axis = region.axis.unwrap_or(RowAxis::PRIMARY);
        if !units.contains(&axis) {
            units.push(axis);
        }
    }
    if units.is_empty() {
        units.push(RowAxis::PRIMARY);
    }

    // The primary axis's unit is not unit 0 when the first capture region
    // is a tower's; an axis-less prepare region takes this unit.
    let primary = units
        .iter()
        .position(|held| *held == RowAxis::PRIMARY)
        .unwrap_or(0) as u32;

    // A region with no axis of its own belongs to whichever unit is open
    // where it stands (unit 0 before any capture region has opened one) —
    // except a prepare region, which names no exec, so "open" means
    // nothing for it: prepare-region rows always read per-lane geometry on
    // the primary axis (a plan builder's schedule is keyed there), so an
    // axis-less prepare region takes the primary unit instead. A prepare
    // region that does state an axis still takes its own, unchanged.
    let mut open = 0u32;
    let mut units_of: Vec<u32> = Vec::with_capacity(regions.len());
    for region in regions {
        let unit = match region.axis {
            Some(axis) => units.iter().position(|held| *held == axis).unwrap_or(0) as u32,
            None if region.phase != Phase::Capture => primary,
            None => open,
        };
        if region.phase == Phase::Capture {
            open = unit;
        }
        units_of.push(unit);
    }

    // Each unit's capture regions must be one run of the script: an exec is
    // recorded front to back, so a unit resuming after another one has run
    // would be a second exec of the same unit.
    let mut seen: Vec<u32> = Vec::new();
    let mut previous: Option<u32> = None;
    for (r, region) in regions.iter().enumerate() {
        if region.phase != Phase::Capture {
            continue;
        }
        let unit = units_of[r];
        if previous == Some(unit) {
            continue;
        }
        if seen.contains(&unit) {
            return Err(Error::UnitsInterleave {
                axis: units[unit as usize],
                unit,
                nodes: region.nodes.clone(),
            });
        }
        seen.push(unit);
        previous = Some(unit);
    }

    Ok((units, units_of))
}

/// Does this artifact's fold have to stand down? Yes for every multi-unit
/// plan: the fold plane arms one graph per bucket per key, and a fire
/// launching two execs has two bucket numbers with no single graph to arm.
#[must_use]
pub(crate) fn fold_refused(units: &[RowAxis]) -> bool {
    units.len() > 1
}

#[cfg(test)]
mod tests {
    use crate::budget::{Budget, Budgets, DeviceProfile, PatchLadder};
    
    use crate::fixture::{Build, patch};
    use crate::{Error, compile, compile_axes};
    use model_ir::{Guard, RowAxis};

    fn budget() -> Budget {
        Budget::new(4, 16)
    }

    fn with_patches() -> Budgets {
        Budgets::of(Budget::new(4, 16)).with_patches(PatchLadder {
            max_patches: 32,
            buckets: vec![8, 16, 32],
            max_images: 4,
        })
    }

    /// A plan that states a patch row against a budget that sizes none is a
    /// load that does not happen — not a tower carved at zero rows.
    #[test]
    fn a_patch_row_against_no_patch_ceiling_is_refused_by_name() {
        let mut b = Build::new();
        let pixels = b.input(8);
        let tower = b.shaped(pixels, patch(8), Guard::Always);
        let y = b.op(tower, 8, Guard::Always);
        b.out(y);

        let refusal = compile(&b.trace, &budget(), &DeviceProfile::default())
            .expect_err("no ceiling, no load");
        assert_eq!(
            refusal,
            Error::Unsized {
                axis: RowAxis::Patches
            }
        );
        assert!(refusal.to_string().contains("patches"));
    }

    /// A unit that resumes after another has run is not one exec, and the
    /// refusal says which unit and where.
    #[test]
    fn a_unit_that_resumes_after_another_is_refused_rather_than_recorded_twice() {
        let mut b = Build::new();
        let pixels = b.input(8);
        let tower = b.shaped(pixels, patch(8), Guard::Always);
        let trunk = b.op(tower, 8, Guard::Always);
        // A second stretch of patch rows, after the trunk has already run.
        let again = b.shaped(trunk, patch(8), Guard::Always);
        let y = b.op(again, 8, Guard::Always);
        b.out(y);

        let refusal = compile_axes(&b.trace, &with_patches(), &DeviceProfile::default())
            .expect_err("two stretches of one axis are two execs of one name");
        assert_eq!(
            refusal,
            Error::UnitsInterleave {
                axis: RowAxis::Patches,
                unit: 0,
                nodes: 2..3,
            }
        );
        assert!(refusal.to_string().contains("one contiguous stretch"));
    }

    /// The patch ladder is checked in the token ladder's vocabulary, and it is
    /// its own vector: a rung past the patch ceiling is a refusal even where
    /// it would have been legal against `max_tokens`.
    #[test]
    fn the_patch_ladder_is_its_own_ladder_and_is_refused_on_its_own_terms() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        b.out(y);

        let profile = DeviceProfile::default();
        let ceiling = Budgets::of(Budget::new(4, 16)).with_patches(PatchLadder {
            max_patches: 0,
            buckets: Vec::new(),
            max_images: 4,
        });
        assert!(matches!(
            compile_axes(&b.trace, &ceiling, &profile),
            Err(Error::Budget { .. })
        ));

        let unsorted = Budgets::of(Budget::new(4, 16)).with_patches(PatchLadder {
            max_patches: 32,
            buckets: vec![8, 8],
            max_images: 4,
        });
        assert!(matches!(
            compile_axes(&b.trace, &unsorted, &profile),
            Err(Error::Budget { .. })
        ));

        // 12 is under `max_tokens` and over `max_patches`, which is the whole
        // point of the ladders being two.
        let past = Budgets::of(Budget::new(4, 16)).with_patches(PatchLadder {
            max_patches: 8,
            buckets: vec![4, 12],
            max_images: 4,
        });
        assert!(matches!(
            compile_axes(&b.trace, &past, &profile),
            Err(Error::Budget { .. })
        ));
    }

    /// The carve reserves a patch column at the patch ceiling and a token
    /// column at the token one — two symbols, neither derived from the
    /// other.
    #[test]
    fn a_patch_column_is_reserved_at_the_patch_ceiling() {
        use crate::arena::{Placement, RowExpr};

        let mut b = Build::new();
        let pixels = b.input(8);
        let tower = b.shaped(pixels, patch(8), Guard::Always);
        let y = b.op(tower, 8, Guard::Always);
        b.out(y);

        let compiled =
            compile_axes(&b.trace, &with_patches(), &DeviceProfile::default()).expect("bakes");
        let Placement::Arena { rows, bytes, .. } = &compiled.arena.placements[tower.0 as usize]
        else {
            panic!("the tower's output is a rectangle of the arena")
        };
        assert_eq!(*rows, RowExpr::Patches);
        // 32 patches x 8 elements x 2 bytes.
        assert_eq!(*bytes, 32 * 8 * 2);

        let Placement::Arena { rows, bytes, .. } = &compiled.arena.placements[y.0 as usize] else {
            panic!("the trunk's output is a rectangle of the arena")
        };
        assert_eq!(*rows, RowExpr::Tokens);
        assert_eq!(*bytes, 16 * 8 * 2);

        // And a patch rectangle never shares a column with a token one: the
        // co-tenancy rule demands equal `rows`, and these are two symbols.
        assert!(!compiled.arena.co_tenants(tower, y));
        assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
    }
}
