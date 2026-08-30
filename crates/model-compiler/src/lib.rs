//! Bakes a traced `Trace` once, into the artifact an engine records and replays
//! forever (palo design §2).
//!
//! > model/ declares a supergraph, model-compiler bakes it once, the engine
//! > records one immutable graph per bucket and replays it forever. Nothing
//! > compiles, allocates, or captures on the fire path.
//!
//! This crate is the middle clause. It runs ONCE PER LOAD and never again: not
//! per fire, not per bucket, not per composition. Everything it produces is
//! either a static table or a symbolic expression the engine evaluates with
//! arithmetic — because the alternative, deciding anything at the fire, is the
//! cost this whole design exists to remove.
//!
//! **Pure.** No device, no I/O, no clock, no environment. `compile` is a
//! function of its three arguments, which is what lets a laptop bake a plan
//! for a machine it has never seen, and lets a test bake all six catalog SKUs
//! with no GPU in the room.
//!
//! # The passes
//!
//! ```text
//! P0 accept      budgets, profile, fact ceiling, collective legality
//! P1 classes     resolve_classes — the check IS the sweep (decision #7)
//! P2 regions     coalesce adjacent nodes with equal (mask, phase, axis)
//! P3 lowering    per region: always-launch (default) | SWITCH | IF
//! M1 units       regions grouped by symbolic row axis -> one exec each
//! P4 layout      C1P over structural consumers -> global class order, per axis
//! P5 phases      Struct-output nodes -> prepare, rest -> capture, and the
//!                prepare half hoisted in front of the capture half (§5)
//! P6 streams     dep DAG over capture regions -> fork/join events
//! P7 arena       liveness carve; offsets fully static
//! P8 emit        CompiledModel
//! ```
//!
//! **Everything but P8 is here.** The seam P8 will fill is TYPED rather than
//! stubbed; see the module docs on [`CompiledModel`] for the four design §2 fields
//! that belong to it and why an empty one would be a claim rather than a gap.
//!
//! P3 ships whole (see [`lowering`]) and chooses ONE region on today's whole
//! catalog: qwen36-27b's MTP head, 26 launches and 576 µs behind the
//! multi-token-prediction fact. Zero-row always-launch is the correctness
//! mechanism (decision #3), so a conditional is worth its evaluation point
//! only where the body is fat AND the launches it skips outweigh the point
//! that skips them — and every other guarded region in every other text is one
//! to seven operators, which is the shape design §4 says never to wrap.
//! `tests/which_skus_get_a_conditional.rs` pins the predicate and prints the
//! margin per SKU.
//!
//! P6 ships whole (see [`stream`]): a dependency DAG over the capture
//! regions, a cost gate against the profile, streams handed out greedily, and
//! the fork/join event points stamped into the region table — with the
//! concurrency relation fed to the carve, which is the hook `Concurrency` was
//! threaded through for. A plan with nothing to overlap, and every plan under
//! a profile whose `side_streams` is 0, bakes stream 0 everywhere with no
//! event point at all, and pays nothing for the pass having run.
//!
//! P4 ships the C1P instance whole (see [`layout`]): a real PQ-tree over the
//! plan's classes, the feasible SET rather than one witness ordering, so that
//! the fire path's stability pick has something to ask. What it does not yet
//! do is ask — [`ClassOrder::class_order`] takes last fire's order and
//! ignores it — and it withdraws the last conflicting constraint rather than
//! the cheapest one, which is where a Tucker certificate goes.
//!

pub mod arena;
pub mod compiled;
pub mod budget;
pub mod layout;
pub mod lowering;
/// The dense demand shape's fire-invariant copy order (alto streaming §2).
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
    Budget, Budgets, DeviceProfile, FamilyCosts, PATCH_LATTICE_FLOOR, PatchLadder,
};
/// The row-axis discriminator, re-exported UNDER ITS OWN NAME from the IR
/// that owns it. A second spelling here would be a second answer to "which
/// axis is this" waiting to disagree with the one a model text wrote.
pub use model_ir::RowAxis;
pub use pq::PqTree;
pub use stream::StreamPlan;
pub use error::{Error, Share, Unrectangled};

/// The fact ceiling. The class sweep is `2^F`, and past 20 it stops being a
/// sweep — the same number `Guard::simplified` and `resolve_classes` state.
///
/// `F` is what `model_ir::fact_width` reads off the plan's own guards, which
/// is the number `resolve_classes` would then assert on. A plan that computes
/// facts it never splits on is not one this ceiling has any opinion about.
const MAX_FACTS: usize = 20;

/// Bake one plan.
///
/// The binding contract of design §2, and the only door into this crate.
///
/// # Errors
///
/// [`Error`], which names the reason and refuses the load. That is the
/// rewrite's doctrine kept whole: **no silent fallback.** A plan whose merges
/// do not cover their classes, whose budgets describe no fire, or whose values
/// have no rectangle is a load that does not happen — not a graph that is
/// quietly missing a window and computes anyway.
///
/// ONE ROW AXIS. This admits the token rectangle and nothing else, which is
/// what every deployment before the second axis existed meant and what every
/// text-only one still means; a plan that states patch rows against it is
/// refused with [`Error::Unsized`]. [`compile_axes`] is the same bake told
/// about more.
pub fn compile(trace: &Trace, budget: &Budget, profile: &DeviceProfile) -> Result<CompiledModel, Error> {
    compile_axes(trace, &Budgets::of(budget.clone()), profile)
}

/// Bake one plan against ceilings for MORE THAN ONE ROW AXIS.
///
/// The same pass and the same artifact — [`compile`] above is exactly this at
/// the axis every plan has, so a deployment that serves no vision tower has
/// nothing to say here and nothing to pay for it. What a second axis buys is
/// stated in [`CompiledModel::units`]: one exec per axis, chained on one
/// stream, with its own class order, its own bucket ladder and its own
/// fallback rows.
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
    // P0. The compiler is the front door, so every ceiling the utilities
    // downstream ASSERT is checked here first and answered as a refusal: a
    // panic in a load path takes the process with it.
    accept(trace, budgets, profile)?;

    // P0 + P1, one walk. The model test suite calls the same function at
    // authoring time for the author's sake; here it is the accept pass and the
    // class table at once, and neither side pays for the other (decision #7).
    let classes = resolve_classes(trace).map_err(Error::Classes)?;

    // P1's acceptance half: a `Struct` value's readers must share the window
    // it was built in. It is asked off `node_mask` — the sweep's own answer,
    // before regions exist — because a region is a maximal run of nodes with
    // EQUAL masks, so the two readings are the same predicate and this one
    // needs no pass to have run. The authoring net in `crates/model/tests`
    // asks it in the same vocabulary against `resolve_classes` alone, which
    // is the earliest and cheapest instant a straddling model text can be
    // told; this is the front door's own restatement, on the load path.
    struct_readers_share_one_window(trace, &classes)?;

    // P5 then P2: phase per node, then maximal runs of equal (mask, phase).
    let mut regions = region::coalesce(trace, &classes);

    // P5's second half: THE PREPARE HALF RUNS FIRST, WHOLE (design §5). A
    // model text may state its plan builds anywhere it likes — qwen3.6 states
    // the multi-token-prediction head's after the trunk, because the head IS
    // after the trunk — and `coalesce` keeps program order, so without this
    // the artifact carries a host op standing after the graph reads its slot.
    // It runs HERE, before P3, so that every pass below reads one region table
    // and no pass has to know that a second order exists.
    region::hoist(trace, &mut regions)?;

    // M1. THE CAPTURE-UNIT PARTITION, and it is derived rather than declared
    // (multimodal §1): a region's unit is the row axis of the rows it writes,
    // read off the dimension algebra a model text already had to state. It
    // runs HERE — after the hoist, so every prepare region is already
    // global-front and `prepare(all) -> capture(tower) -> capture(trunk)` is
    // the script it partitions, and before P3/P6/P4, so every pass below sees
    // a region table whose units are stamped.
    //
    // A ONE-AXIS PLAN GETS `[RowAxis::Tokens]` AND `unit: 0` EVERYWHERE, which
    // is every pre-campaign SKU, and the region table it partitions is the one
    // `coalesce` always built — the G4 invariant, pinned by
    // `tests/the_second_row_axis_costs_the_first_nothing.rs`.
    let axes = unit::axes_of(trace, &regions);
    let (units, units_of) = unit::partition(&regions, &axes)?;

    // P3. Which regions enter the graph behind a conditional node — and P6
    // below reads the answer, because `stream::forkable` refuses to fork one
    // that does. The order is the composition rule and it is a mechanism
    // rather than a preference: a conditional body is a child graph and a
    // fork's event pair is an edge inside one parent graph, so an arm that
    // was both is a dependency that cannot be expressed (see [`lowering`]).
    lowering::lower(trace, &mut regions, &classes, budget, profile);

    // P6. The dep DAG over the capture regions, the cost gate, the streams
    // and the event points — stamped into `regions` in place, because a fork
    // is a property OF a region rather than a second schedule beside it. What
    // comes back is the relation the carve is widened by, and P7 below is the
    // pass that was written waiting for it.
    let streams = stream::fork(trace, &mut regions, profile);
    let concurrency = if streams.pairs.is_empty() {
        Concurrency::sequential(&regions, trace.nodes.len())
    } else {
        Concurrency::with_pairs(&regions, trace.nodes.len(), streams.pairs.iter().copied())
    };

    // P4, ONCE PER AXIS. The C1P instance is read off the region table, so it
    // runs after P2 even though it is numbered before it; and it is read off
    // the MASKS rather than the bytes, so it owes the carve nothing and the
    // carve owes it nothing. What it decides is the order a fire seriates its
    // rows in, which is arithmetic on the descriptor, not an offset.
    //
    // WHY TWICE AND NOT WIDER (multimodal §5.1). `compose`'s merged prefix sum
    // — "rows and lanes break at the same places" — is what makes ONE order
    // answer for the token rectangle and its lane space at once. It does not
    // extend to patches: a lane of a class may carry zero images or three, so
    // the patch axis is its own instance over its own lane space (images),
    // with its own fallback rows indexed into its own ladder. The pass is a
    // pure function of the regions it is handed, so "per axis" is the regions
    // of that axis and nothing else moves.
    let (order, fallback) = layout::seriate(
        trace,
        &regions_on(&regions, &units_of, &units, RowAxis::Tokens),
        &classes,
        budget,
        profile,
    );
    let patches = patch_axis(trace, &regions, &units_of, &units, &classes, budgets, profile);

    // P7. The carve is the pass with something to get wrong, and
    // `ArenaMap::clashes` is what says it did not.
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

/// The regions of one axis, in script order — what P4 is asked about that
/// axis.
///
/// **A FILTER AND NOT A REBUILD.** `Region::nodes` are absolute node indices,
/// so a fallback row this produces names the same node it would have named off
/// the whole table; what dropping the other axis's regions changes is only
/// which windows constrain the order, which is the entire point. A plan with
/// one unit is handed back its own table, element for element.
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

/// P4 on the patch axis, or `None` for a plan that states no patch row.
///
/// THE LADDER IT MEASURES AGAINST IS THE PATCH LADDER. `layout::menu` reads a
/// bucket lattice and a ceiling off the budget it is handed, and the answer it
/// writes is bucket-dependent by construction (copy below the crossover, split
/// above it) — so handing it the TOKEN ladder would key the tower's fallback
/// rows to the trunk's rungs. It is handed the patch axis's own numbers in the
/// budget's own shape instead; nothing else of a `Budget` is read here, and
/// this one never leaves the function.
fn patch_axis(
    trace: &Trace,
    regions: &[Region],
    units_of: &[u32],
    units: &[RowAxis],
    classes: &ClassTable,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Option<AxisPlan> {
    let ladder = budgets.patches.as_ref()?;
    if !units.contains(&RowAxis::Patches) {
        return None;
    }
    let on_axis = regions_on(regions, units_of, units, RowAxis::Patches);
    // THE PATCH AXIS'S OWN LANE COUNT IS IMAGES, and it is stated rather
    // than borrowed from the token axis: a lane may submit three images or
    // none, so `Budget::max_lanes` is not this number and reading it here
    // would size the tower's `images + 1` indptr by an argument about
    // requests. `accept` has already refused a ladder that admits more
    // images than patch rows, so this pair is coherent by the time it is
    // read.
    let ceiling = Budget {
        max_lanes: ladder.max_images,
        max_tokens: ladder.max_patches,
        buckets: ladder.buckets.clone(),
        max_adapters: 0,
    };
    let (order, fallback) = layout::seriate(trace, &on_axis, classes, &ceiling, profile);
    Some(AxisPlan {
        axis: RowAxis::Patches,
        order,
        fallback,
    })
}

/// P0's own checks — the ones that are about the arguments rather than about
/// the merges.
///
/// COLLECTIVES ARE ACCEPTED UNDER ANY GUARD, AND RECORDED (decision #5). A
/// rank's guard is its own business: a text may state `all_reduce` inside a
/// window, and the fire descriptor is replicated across ranks so every rank
/// reaches the same call in the same order. What is NOT its own business is
/// elision — a collective inside a skipped body deadlocks the ranks that did
/// not skip, or silently mispairs with a later collective, because NCCL
/// matches by call order. So the check is not a refusal here; it is a FACT
/// carried forward, [`Region::collective`], and the rule it enforces is P3's:
/// a region carrying one stays always-launch. Stating it as a refusal instead
/// would refuse plans that are perfectly legal today.
fn accept(trace: &Trace, budgets: &Budgets, profile: &DeviceProfile) -> Result<(), Error> {
    let budget = &budgets.tokens;
    let facts = model_ir::fact_width(trace);
    if facts > MAX_FACTS {
        return Err(Error::TooManyFacts { facts });
    }

    if budget.max_lanes == 0 {
        return Err(Error::Budget {
            what: "admit no lanes, so no fire can be assembled",
        });
    }
    if budget.max_tokens == 0 {
        return Err(Error::Budget {
            what: "admit no token rows, so every rectangle is empty",
        });
    }
    if budget.max_lanes > budget.max_tokens {
        return Err(Error::Budget {
            what: "admit more lanes than token rows, and a lane carries at least one row",
        });
    }
    let mut previous = 0u32;
    for &bucket in &budget.buckets {
        if bucket <= previous {
            return Err(Error::Budget {
                what: "list a bucket lattice that does not strictly ascend",
            });
        }
        if bucket > budget.max_tokens {
            return Err(Error::Budget {
                what: "list a bucket past the token ceiling",
            });
        }
        previous = bucket;
    }

    // THE SECOND AXIS'S CEILING, CHECKED IN THE SAME VOCABULARY (multimodal
    // §5.5). A ladder is a ladder whichever axis it climbs, so the same three
    // refusals apply — and they are stated here rather than folded into the
    // loop above so that a reader can see there are two ladders and that
    // neither is derived from the other.
    if let Some(ladder) = &budgets.patches {
        if ladder.max_patches == 0 {
            return Err(Error::Budget {
                what: "admit a patch axis with no patch rows, so every tower                        rectangle is empty",
            });
        }
        if ladder.max_images == 0 {
            return Err(Error::Budget {
                what: "admit a patch axis with no images, and a patch row is a row                        of one",
            });
        }
        if ladder.max_images > ladder.max_patches {
            return Err(Error::Budget {
                what: "admit more images than patch rows, and an image carries at                        least one patch row",
            });
        }
        let mut previous = 0u32;
        for &bucket in &ladder.buckets {
            if bucket <= previous {
                return Err(Error::Budget {
                    what: "list a patch lattice that does not strictly ascend",
                });
            }
            if bucket > ladder.max_patches {
                return Err(Error::Budget {
                    what: "list a patch bucket past the patch ceiling",
                });
            }
            previous = bucket;
        }
    }

    // **AND THE PLAN'S SIDE OF THE SAME QUESTION.** A budget may admit an axis
    // no plan states — a deployment declaring a patch ladder for a text-only
    // model costs that model nothing, and the artifact is bit-identical. The
    // other direction is a refusal: a plan that states a row axis the budget
    // sizes no ceiling for has a rectangle with no height, and a carve at zero
    // rows computes over somebody else's bytes rather than faulting.
    for axis in unit::axes_stated(trace) {
        let sized = match axis {
            RowAxis::Tokens => true,
            RowAxis::Patches => budgets.patches.is_some(),
        };
        if !sized {
            return Err(Error::Unsized { axis });
        }
    }

    if profile.sms == 0 {
        return Err(Error::Profile {
            what: "describes a device with no streaming multiprocessors",
        });
    }

    // **THE ONE PLACE THE DEPLOYMENT'S ADAPTER NUMBER AND THE MODEL'S MEET**
    // (design §8, decision 17). Capacity is a SHAPE — the leading axis of
    // every bank the model text marked `ParamSource::Registered` — because a
    // shape is what a weight table reserves and what the routed op indexes.
    // `Budget::max_adapters` is what the deployment asked to be able to
    // register, and asking for more than the text can seat is a load that
    // does not happen rather than a registration that is refused later:
    // decision 17's budget is not an admission cap, so it has to be true
    // BEFORE the first registration, not discovered at one.
    //
    // A plan with no bank is exempt at `max_adapters == 0` and refused above
    // it, which is the honest reading of "this deployment wants adapters and
    // this model text has no seat for them".
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

/// P1's acceptance check: no `Struct` value may be read outside the window it
/// was built in.
///
/// `Ty::Struct` is exactly the attention SCHEDULES (`StructKind` is closed and
/// every variant is one), and [`Error::Straddled`] argues why a schedule
/// cannot be sliced the way a tensor can. One pass over the nodes: the
/// DEFINING node stands in the window its builder is dispatched in by
/// construction, so what is compared is its class set against every reader's.
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

/// Every `Collective`-family node in a plan, in program order.
///
/// P0'S RECORD, READ BY P3 AND BY THE TESTS. `Region::collective` is the
/// per-region fold of this and is what the lowering pass actually consults;
/// this is the node-granular list, for a diagnostic that has to name which
/// call it means.
#[must_use]
pub fn collectives(trace: &Trace) -> Vec<u32> {
    trace.nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.op, Operation::Collective(_)))
        .map(|(j, _)| j as u32)
        .collect()
}

/// Does this artifact keep the one lowering rule that is not an optimization?
///
/// **A COLLECTIVE IS NEVER ELIDED** (decision #5). P3 enforces it at the gate
/// — `lowering::windowed` refuses a region carrying one before it asks what it
/// costs — and this is the same claim asked of the OUTPUT, which is where an
/// assertion about a pass belongs.
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
    fn a_plan_bakes_into_an_artifact_whose_arena_is_sound() {
        let b = plan();
        let compiled = compile(&b.trace, &Budget::new(4, 16), &DeviceProfile::default())
            .expect("a covering plan bakes");

        assert_eq!(compiled.classes.classes.len(), 2);
        assert!(!compiled.regions.is_empty());
        assert_eq!(compiled.template().len(), compiled.regions.len());
        // P4 seated both windows: the two classes are trivially an interval
        // of any order, so nothing is owed a fallback.
        assert_eq!(compiled.order.tree().expect("P4 seriated it").frontier(), [0, 1]);
        assert!(compiled.fallback.rows.is_empty());
        assert!(compiled.concurrency.pairs().is_empty());
        assert!(compiled.arena.bytes > 0);
        assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
        assert!(collectives_are_never_elided(&compiled));
        // Two two-node arms: fat at no profile and profitable at none, so
        // the one lowering that is correctness and not optimization.
        assert!(
            compiled
                .regions
                .iter()
                .all(|r| r.lowering == Lowering::AlwaysLaunch)
        );
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
        // Bit 20 is the twenty-first, so a guard that reaches it is a sweep of
        // 2^21 — and the ceiling is read off the guard, not off a vocabulary
        // the plan no longer carries.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, fact(20));
        b.out(y);

        assert_eq!(
            compile(&b.trace, &Budget::new(4, 16), &DeviceProfile::default()),
            Err(Error::TooManyFacts { facts: 21 }),
        );
    }

    #[test]
    fn a_collective_is_listed_and_its_region_is_never_elidable() {
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Guard::Always);
        let g = b.all_gather(a, 8, fact(0));
        let o = b.merge(&[(g, fact(0)), (a, Guard::not(fact(0)))], 8);
        b.out(o);

        assert_eq!(collectives(&b.trace), vec![1]);
        let compiled =
            compile(&b.trace, &Budget::new(4, 16), &DeviceProfile::default()).expect("bakes");
        assert!(compiled.regions.iter().any(|r| r.collective));
        assert!(collectives_are_never_elided(&compiled));
    }
}
