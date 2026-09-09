pub mod arena;
pub mod budget;
pub mod compiled;
pub mod error;
pub mod layout;
pub mod lowering;
mod pq;
pub mod prefetch;
mod region;
pub mod stream;
pub mod unit;

#[cfg(test)]
mod fixture;

use model_ir::{ClassTable, Def, Operands, Operation, Trace, Ty, resolve_classes};

pub use arena::{
    ArenaMap, Concurrency, EXPORT_SEAMS, Extent, FLOAT_READOUT_SEAMS, FireRows, Placement, RowExpr,
    Span,
};
pub use budget::{
    Budget, Budgets, DeviceProfile, FamilyCosts, Ladder, PATCH_LATTICE_FLOOR, PatchLadder,
    VoxelLadder,
};
pub use compiled::{
    AxisPlan, ClassOrder, CompiledModel, EventId, Fallback, FallbackRow, FallbackTable, Lowering,
    Phase, Region,
};
pub use error::{Error, Share, Unrectangled};
pub use model_ir::RowAxis;
pub use pq::PqTree;
pub use stream::StreamPlan;

const MAX_FACTS: usize = 20;

pub const MAX_CLASSES: usize = u8::MAX as usize + 1;

pub fn compile(
    trace: &Trace,
    budget: &Budget,
    profile: &DeviceProfile,
) -> Result<CompiledModel, Error> {
    compile_axes(trace, &Budgets::of(budget.clone()), profile)
}

pub fn compile_axes(
    trace: &Trace,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Result<CompiledModel, Error> {
    let budget = &budgets.tokens;
    accept(trace, budgets, profile)?;

    let classes = resolve_classes(trace).map_err(Error::Classes)?;
    if classes.classes.len() > MAX_CLASSES {
        return Err(Error::TooManyClasses {
            classes: classes.classes.len(),
        });
    }
    if classes.node_mask.len() != trace.nodes.len() {
        return Err(Error::MaskLength {
            masks: classes.node_mask.len(),
            nodes: trace.nodes.len(),
        });
    }

    struct_readers_share_one_window(trace, &classes)?;

    let mut regions = region::coalesce(trace, &classes)?;

    region::hoist(trace, &mut regions)?;

    let (units, units_of) = unit::partition(&regions)?;

    lowering::lower(trace, &mut regions, &classes, budget, profile);

    let streams = stream::fork(trace, &mut regions, profile);
    let concurrency =
        Concurrency::with_pairs(&regions, trace.nodes.len(), streams.pairs.iter().copied());

    let (order, fallback) = layout::seriate(
        trace,
        &regions_on(&regions, &units_of, &units, RowAxis::Tokens),
        &classes,
        &budget.buckets,
        budget.max_tokens,
        profile,
    );
    let patches = axis_plan(
        RowAxis::Patches,
        trace,
        &regions,
        &units_of,
        &units,
        &classes,
        budgets,
        profile,
    );
    let voxels = axis_plan(
        RowAxis::Voxels,
        trace,
        &regions,
        &units_of,
        &units,
        &classes,
        budgets,
        profile,
    );

    let arena = arena::carve(trace, budgets, &classes, &concurrency)?;

    Ok(CompiledModel {
        classes,
        regions,
        order,
        fallback,
        arena,
        concurrency,
        streams,
        fold_refused: unit::fold_refused(&units),
        units,
        units_of,
        patches,
        voxels,
    })
}

fn regions_on(
    regions: &[Region],
    units_of: &[u32],
    units: &[RowAxis],
    axis: RowAxis,
) -> Vec<Region> {
    let Some(unit) = units.iter().position(|held| *held == axis) else {
        return Vec::new();
    };
    let unit = unit as u32;
    regions
        .iter()
        .enumerate()
        .filter(|(r, _)| units_of.get(*r).copied().unwrap_or(0) == unit)
        .map(|(_, region)| region.clone())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn axis_plan(
    axis: RowAxis,
    trace: &Trace,
    regions: &[Region],
    units_of: &[u32],
    units: &[RowAxis],
    classes: &ClassTable,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Option<AxisPlan> {
    let ladder = budgets.ladder(axis)?;
    if !units.contains(&axis) {
        return None;
    }
    let on_axis = regions_on(regions, units_of, units, axis);
    let (order, fallback) = layout::seriate(
        trace,
        &on_axis,
        classes,
        ladder.buckets,
        ladder.max_rows,
        profile,
    );
    Some(AxisPlan {
        axis,
        order,
        fallback,
    })
}

fn accept(trace: &Trace, budgets: &Budgets, profile: &DeviceProfile) -> Result<(), Error> {
    let budget = &budgets.tokens;
    let facts = model_ir::fact_width(trace);
    if facts > MAX_FACTS {
        return Err(Error::TooManyFacts { facts });
    }

    for axis in RowAxis::ALL {
        if let Some(ladder) = budgets.ladder(axis) {
            accept_ladder(&ladder, axis)?;
        }
    }

    for axis in unit::axes_stated(trace) {
        if budgets.ladder(axis).is_none() {
            return Err(Error::Unsized { axis });
        }
    }

    if profile.sms == 0 {
        return Err(Error::Profile {
            what: "describes a device with no streaming multiprocessors",
        });
    }

    if budget.max_adapters > 0 {
        let seats = trace
            .params
            .iter()
            .filter(|param| param.source == model_ir::ParamSource::Registered)
            .map(|param| param.shape.first().copied().unwrap_or(0))
            .min();
        match seats {
            None => {
                return Err(Error::AdapterCapacity {
                    asked: budget.max_adapters,
                    seated: 0,
                });
            }
            Some(seated) if seated < u64::from(budget.max_adapters) => {
                return Err(Error::AdapterCapacity {
                    asked: budget.max_adapters,
                    seated,
                });
            }
            Some(_) => {}
        }
    }

    Ok(())
}

fn accept_ladder(ladder: &Ladder<'_>, axis: RowAxis) -> Result<(), Error> {
    let words = LADDER_WORDS[axis];
    if ladder.max_lanes == 0 {
        return Err(Error::Budget {
            what: words.no_lanes,
        });
    }
    if ladder.max_rows == 0 {
        return Err(Error::Budget {
            what: words.no_rows,
        });
    }
    if ladder.max_lanes > ladder.max_rows {
        return Err(Error::Budget {
            what: words.lanes_past_rows,
        });
    }
    let mut previous = 0u32;
    for &bucket in ladder.buckets {
        if bucket <= previous {
            return Err(Error::Budget {
                what: words.unsorted,
            });
        }
        if bucket > ladder.max_rows {
            return Err(Error::Budget {
                what: words.past_ceiling,
            });
        }
        previous = bucket;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct LadderWords {
    no_lanes: &'static str,
    no_rows: &'static str,
    lanes_past_rows: &'static str,
    unsorted: &'static str,
    past_ceiling: &'static str,
}

const LADDER_WORDS: model_ir::PerAxis<LadderWords> = model_ir::PerAxis::new([
    LadderWords {
        no_lanes: "admit no lanes, so no fire can be assembled",
        no_rows: "admit no token rows, so every rectangle is empty",
        lanes_past_rows: "admit more lanes than token rows, and a lane carries at least one row",
        unsorted: "list a bucket lattice that does not strictly ascend",
        past_ceiling: "list a bucket past the token ceiling",
    },
    LadderWords {
        no_lanes: "admit a patch axis with no images, and a patch row is a row of one",
        no_rows: "admit a patch axis with no patch rows, so every tower rectangle is empty",
        lanes_past_rows: "admit more images than patch rows, and an image carries at least one \
                          patch row",
        unsorted: "list a patch lattice that does not strictly ascend",
        past_ceiling: "list a patch bucket past the patch ceiling",
    },
    LadderWords {
        no_lanes: "admit a voxel axis with no clips, and a voxel row is a row of one",
        no_rows: "admit a voxel axis with no voxel rows, so every VAE rectangle is empty",
        lanes_past_rows: "admit more clips than voxel rows, and a clip carries at least one \
                          voxel row",
        unsorted: "list a voxel lattice that does not strictly ascend",
        past_ceiling: "list a voxel bucket past the voxel ceiling",
    },
]);

fn struct_readers_share_one_window(trace: &Trace, classes: &ClassTable) -> Result<(), Error> {
    let structs: Vec<bool> = trace
        .values
        .iter()
        .map(|value| matches!(value.ty, Ty::Struct(_)))
        .collect();

    let mut inputs = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        inputs.clear();
        node.op.inputs(&mut inputs);
        for &read in &inputs {
            if !structs.get(read.0 as usize).copied().unwrap_or(false) {
                continue;
            }
            let Some(Def::Op(built_by)) = trace.values.get(read.0 as usize).map(|v| &v.def) else {
                continue;
            };
            let planned = &classes.node_mask[*built_by as usize];
            let reader = &classes.node_mask[at];
            if planned != reader {
                return Err(Error::Straddled {
                    value: read,
                    node: at as u32,
                    planned: planned.iter().collect(),
                    consumed: reader.iter().collect(),
                });
            }
        }
    }
    Ok(())
}

#[must_use]
pub fn collectives(trace: &Trace) -> Vec<u32> {
    trace
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.op, Operation::Collective(_)))
        .map(|(j, _)| j as u32)
        .collect()
}

#[must_use]
pub fn collectives_are_never_elided(compiled: &CompiledModel) -> bool {
    compiled
        .regions
        .iter()
        .all(|r| !r.collective || r.lowering == Lowering::AlwaysLaunch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, fact};
    use model_ir::Guard;

    fn plan() -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Guard::Always);
        let d = b.op(q, 8, fact(0));
        let p = b.op(q, 8, Guard::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 8);
        let y = b.op(o, 8, Guard::Always);
        b.append(y, Guard::Always);
        b.out(y);
        b
    }

    #[test]
    fn lib_every_case() {
        an_uncovered_merge_refuses_the_load_and_says_which();
        a_budget_that_describes_no_fire_is_refused_before_anything_is_swept();
        a_device_with_no_sms_is_refused();
        the_fact_ceiling_is_a_refusal_and_not_a_panic();
    }

    fn an_uncovered_merge_refuses_the_load_and_says_which() {
        let mut b = Build::new();
        let x = b.input(8);
        let d = b.op(x, 8, fact(0));
        let m = b.op(x, 8, Guard::and(Guard::not(fact(0)), fact(1)));
        let o = b.merge(
            &[(d, fact(0)), (m, Guard::and(Guard::not(fact(0)), fact(1)))],
            8,
        );
        b.out(o);

        let refusal = compile(&b.trace, &Budget::new(4, 16), &DeviceProfile::default())
            .expect_err("a hole is a refusal");
        let Error::Classes(faults) = &refusal else {
            panic!("the class sweep is what refused it: {refusal}")
        };
        assert_eq!(faults.len(), 1);
        assert!(refusal.say(&b.trace).contains("no arm holds there"));
    }

    fn a_budget_that_describes_no_fire_is_refused_before_anything_is_swept() {
        let b = plan();
        let profile = DeviceProfile::default();
        assert!(matches!(
            compile(&b.trace, &Budget::new(0, 16), &profile),
            Err(Error::Budget { .. }),
        ));
        assert!(matches!(
            compile(&b.trace, &Budget::new(4, 0), &profile),
            Err(Error::Budget { .. }),
        ));
        assert!(matches!(
            compile(&b.trace, &Budget::new(32, 16), &profile),
            Err(Error::Budget { .. }),
        ));
        let mut lattice = Budget::new(4, 16);
        lattice.buckets = vec![1, 8, 8];
        assert!(matches!(
            compile(&b.trace, &lattice, &profile),
            Err(Error::Budget { .. }),
        ));
        lattice.buckets = vec![1, 8, 64];
        assert!(matches!(
            compile(&b.trace, &lattice, &profile),
            Err(Error::Budget { .. }),
        ));
    }

    fn a_device_with_no_sms_is_refused() {
        let b = plan();
        let profile = DeviceProfile {
            sms: 0,
            ..DeviceProfile::default()
        };
        assert!(matches!(
            compile(&b.trace, &Budget::new(4, 16), &profile),
            Err(Error::Profile { .. }),
        ));
    }

    fn the_fact_ceiling_is_a_refusal_and_not_a_panic() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, fact(20));
        b.out(y);

        assert_eq!(
            compile(&b.trace, &Budget::new(4, 16), &DeviceProfile::default()),
            Err(Error::TooManyFacts { facts: 21 }),
        );
    }
}
