use model_ir::{Operands, RowAxis, Trace, Ty, ValueId};

use crate::compiled::{Phase, Region};
use crate::error::Error;

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

#[must_use]
pub(crate) fn axes_stated(trace: &Trace) -> Vec<RowAxis> {
    let mut axes = Vec::new();
    for decl in &trace.values {
        let Ty::Tensor { shape, .. } = &decl.ty else {
            continue;
        };
        for dim in shape {
            if let Some(axis) = dim.axis()
                && !axes.contains(&axis)
            {
                axes.push(axis);
            }
        }
    }
    axes.sort_unstable();
    axes
}

pub(crate) fn partition(regions: &[Region]) -> Result<(Vec<RowAxis>, Vec<u32>), Error> {
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

    let primary = units
        .iter()
        .position(|held| *held == RowAxis::PRIMARY)
        .unwrap_or(0) as u32;

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

    #[test]
    fn unit_every_case() {
        a_patch_row_against_no_patch_ceiling_is_refused_by_name();
        a_unit_that_resumes_after_another_is_refused_rather_than_recorded_twice();
        the_patch_ladder_is_its_own_ladder_and_is_refused_on_its_own_terms();
        a_patch_column_is_reserved_at_the_patch_ceiling();
    }

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

    fn a_unit_that_resumes_after_another_is_refused_rather_than_recorded_twice() {
        let mut b = Build::new();
        let pixels = b.input(8);
        let tower = b.shaped(pixels, patch(8), Guard::Always);
        let trunk = b.op(tower, 8, Guard::Always);
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
        assert_eq!(*bytes, 32 * 8 * 2);

        let Placement::Arena { rows, bytes, .. } = &compiled.arena.placements[y.0 as usize] else {
            panic!("the trunk's output is a rectangle of the arena")
        };
        assert_eq!(*rows, RowExpr::Tokens);
        assert_eq!(*bytes, 16 * 8 * 2);

        assert!(!compiled.arena.co_tenants(tower, y));
        assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
    }
}
