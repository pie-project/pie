//! Bakes a traced `Trace` once into the `CompiledModel` an engine replays.

pub mod arena;
pub mod compiled;
pub mod budget;
pub mod layout;
pub mod lowering;
/// The dense demand shape's fire-invariant copy order.
pub mod prefetch;
pub mod error;
mod pq;
mod region;
pub mod stream;
pub mod unit;

#[cfg(test)]
mod fixture;

use model_ir::{ClassTable, Def, Operands, Operation, Trace, Ty, resolve_classes};

pub use arena::{ArenaMap, Concurrency, EXPORT_SEAMS, FireRows, RowExpr, Placement, Span, Extent};
pub use compiled::{
    AxisPlan, CompiledModel, EventId, Fallback, FallbackRow, FallbackTable, ClassOrder, Lowering,
    Phase, Region,
};
pub use budget::{
    Budget, Budgets, DeviceProfile, FamilyCosts, Ladder, PATCH_LATTICE_FLOOR, PatchLadder,
};
/// Re-exported under its own name rather than restated, so there is only one
/// answer to "which axis is this".
pub use model_ir::RowAxis;
pub use pq::PqTree;
pub use stream::StreamPlan;
pub use error::{Error, Share, Unrectangled};

/// The fact ceiling: the class sweep is `2^F` and stops being one past this.
/// `F` is read off the plan's own guards by `model_ir::fact_width`.
const MAX_FACTS: usize = 20;

/// Bakes one plan; the only entry point into this crate.
///
/// # Errors
///
/// [`Error`], naming the reason and refusing the load — no silent fallback.
/// Admits only the token row axis; a plan that states patch rows is refused
/// with [`Error::Unsized`]. [`compile_axes`] is the same bake told about more.
/// Ceiling on classes: `ClassOrder::class_order` names a class in a `u8`.
pub const MAX_CLASSES: usize = u8::MAX as usize + 1;

pub fn compile(trace: &Trace, budget: &Budget, profile: &DeviceProfile) -> Result<CompiledModel, Error> {
    compile_axes(trace, &Budgets::of(budget.clone()), profile)
}

/// Bakes one plan against ceilings for more than one row axis. [`compile`]
/// is exactly this at the one axis every plan has. A second axis buys one
/// exec per axis, chained on one stream, each with its own class order,
/// bucket ladder, and fallback rows ([`CompiledModel::units`]).
///
/// # Errors
///
/// [`Error`], as [`compile`] — plus [`Error::Unsized`] for a plan that states
/// a row axis these budgets size no ceiling for, and
/// [`Error::UnitsInterleave`] for one whose axes alternate down the script.
pub fn compile_axes(
    trace: &Trace,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Result<CompiledModel, Error> {
    let budget = &budgets.tokens;
    // Every ceiling downstream code asserts is checked here first and
    // answered as a refusal, so a panic in a load path never happens.
    accept(trace, budgets, profile)?;

    // The accept pass and the class table share this sweep.
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

    // A `Struct` value's readers must share the window it was built in,
    // checked off `node_mask` before regions exist.
    struct_readers_share_one_window(trace, &classes)?;

    // Phase per node, then maximal runs of equal (mask, phase).
    let mut regions = region::coalesce(trace, &classes)?;

    // The prepare half runs first, whole: `coalesce` keeps program order, so
    // without this a host op could stand after the graph reads its slot.
    region::hoist(trace, &mut regions)?;

    // The capture-unit partition, derived rather than declared — a region's
    // unit is the row axis of the rows it writes. A one-axis plan gets
    // `[RowAxis::Tokens]` and `unit: 0` everywhere.
    let (units, units_of) = unit::partition(&regions)?;

    // Marks which regions enter the graph behind a conditional:
    // `stream::forkable` refuses to fork one that does, since a conditional
    // body is a child graph and can't share a fork's event pair with its
    // parent (see [`lowering`]).
    lowering::lower(trace, &mut regions, &classes, budget, profile);

    // The dep DAG over capture regions, cost gate, streams, and event
    // points, stamped into `regions` in place.
    let streams = stream::fork(trace, &mut regions, profile);
    let concurrency =
        Concurrency::with_pairs(&regions, trace.nodes.len(), streams.pairs.iter().copied());

    // Decides the order a fire seriates its rows in. Run per axis rather
    // than combined: a lane's patch count varies independently of the
    // token rectangle, so the patch axis needs its own instance and ladder.
    let (order, fallback) = layout::seriate(
        trace,
        &regions_on(&regions, &units_of, &units, RowAxis::Tokens),
        &classes,
        &budget.buckets,
        budget.max_tokens,
        profile,
    );
    let patches = patch_axis(trace, &regions, &units_of, &units, &classes, budgets, profile);

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
    })
}

/// The regions of one axis, in script order — what `layout` is asked about
/// that axis. A filter, not a rebuild: `Region::nodes` are absolute node
/// indices, so a fallback row this produces still names the right node.
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

/// `layout` on the patch axis, or `None` for a plan that states no patch row.
/// Uses the patch axis's own ladder rather than the token one, since
/// `layout::menu`'s answer is bucket-dependent (copy below the crossover,
/// split above it) and would otherwise key the tower's fallback rows to the
/// trunk's rungs.
fn patch_axis(
    trace: &Trace,
    regions: &[Region],
    units_of: &[u32],
    units: &[RowAxis],
    classes: &ClassTable,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Option<AxisPlan> {
    let ladder = budgets.ladder(RowAxis::Patches)?;
    if !units.contains(&RowAxis::Patches) {
        return None;
    }
    let on_axis = regions_on(regions, units_of, units, RowAxis::Patches);
    // This axis's own ladder; `accept` already refused a non-ascending
    // lattice or a rung past ceiling on this axis.
    let (order, fallback) = layout::seriate(
        trace,
        &on_axis,
        classes,
        ladder.buckets,
        ladder.max_rows,
        profile,
    );
    Some(AxisPlan {
        axis: RowAxis::Patches,
        order,
        fallback,
    })
}

/// Checks about the arguments, not the merges.
///
/// A collective is accepted under any guard and recorded rather than
/// refused: a rank's guard is its own business, but a collective inside a
/// skipped body would deadlock ranks that did not skip, or mispair with a
/// later call. The fact is carried forward as [`Region::collective`].
fn accept(trace: &Trace, budgets: &Budgets, profile: &DeviceProfile) -> Result<(), Error> {
    let budget = &budgets.tokens;
    let facts = model_ir::fact_width(trace);
    if facts > MAX_FACTS {
        return Err(Error::TooManyFacts { facts });
    }

    // One ladder per row axis the deployment states; the patch ladder is
    // checked only when a deployment admits patch rows.
    for axis in RowAxis::ALL {
        if let Some(ladder) = budgets.ladder(axis) {
            accept_ladder(&ladder, axis)?;
        }
    }

    // The plan's side of the same question: a budget may admit an axis no
    // plan states, at no cost, but a plan stating a row axis the budget
    // sizes no ceiling for is refused — its rectangle would have no height.
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

    // Capacity is a shape: the leading axis of every bank marked
    // `ParamSource::Registered`. Asking for more adapters than the model
    // text can seat is refused here, not at registration time.
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

/// The five things a ladder may not say, on any row axis, stated once and
/// called per axis. [`LadderWords`] carries only the per-axis wording of a
/// refusal.
///
/// # Errors
///
/// [`Error::Budget`]: no lanes, no rows, more lanes than rows, a lattice
/// that does not strictly ascend, or a rung past the ceiling.
fn accept_ladder(ladder: &Ladder<'_>, axis: RowAxis) -> Result<(), Error> {
    let words = LADDER_WORDS[axis];
    if ladder.max_lanes == 0 {
        return Err(Error::Budget { what: words.no_lanes });
    }
    if ladder.max_rows == 0 {
        return Err(Error::Budget { what: words.no_rows });
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

/// The five sentences [`accept_ladder`] refuses with, for one row axis.
///
/// Each one completes "this deployment would " — which is what `Error::Budget`
/// prints — so they are written as verb phrases and not as nouns.
#[derive(Debug, Clone, Copy)]
struct LadderWords {
    no_lanes: &'static str,
    no_rows: &'static str,
    lanes_past_rows: &'static str,
    unsorted: &'static str,
    past_ceiling: &'static str,
}

/// One row of [`LadderWords`] per row axis, in [`RowAxis::ALL`]'s order.
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
]);

/// No `Struct` value may be read outside the window it was built in. One
/// pass over the nodes, comparing the defining node's class set against
/// every reader's.
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

/// Every `Collective`-family node in a plan, in program order. The
/// node-granular list, for a diagnostic that has to name which call it
/// means; [`Region::collective`] is the per-region fold the lowering pass
/// actually consults.
#[must_use]
pub fn collectives(trace: &Trace) -> Vec<u32> {
    trace.nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.op, Operation::Collective(_)))
        .map(|(j, _)| j as u32)
        .collect()
}

/// Does this artifact keep the rule that a collective is never elided?
/// Asked of the output, mirroring the gate the lowering pass enforces.
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

    #[test]
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

    #[test]
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

    #[test]
    fn the_fact_ceiling_is_a_refusal_and_not_a_panic() {
        // Bit 20 is the twenty-first, past the ceiling.
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
