//! Bakes a traced `Plan` once, into the artifact a driver records and replays
//! forever (palo design §2).
//!
//! > model/ declares a supergraph, model-compiler bakes it once, the driver
//! > records one immutable graph per bucket and replays it forever. Nothing
//! > compiles, allocates, or captures on the fire path.
//!
//! This crate is the middle clause. It runs ONCE PER LOAD and never again: not
//! per fire, not per bucket, not per composition. Everything it produces is
//! either a static table or a symbolic expression the driver evaluates with
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
//! P2 regions     coalesce adjacent nodes with equal (mask, phase)
//! P3 lowering    per region: always-launch (default) | SWITCH | IF
//! P4 layout      C1P over structural consumers -> global class order
//! P5 phases      Struct-output nodes -> prepare, rest -> capture
//! P6 streams     dep DAG over capture regions -> fork/join events
//! P7 arena       liveness carve; offsets fully static
//! P8 emit        Baked
//! ```
//!
//! **Everything but P8 is here.** The seam P8 will fill is TYPED rather than
//! stubbed; see the module docs on [`Baked`] for the four design §2 fields
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
//! do is ask — [`LayoutOrder::class_order`] takes last fire's order and
//! ignores it — and it withdraws the last conflicting constraint rather than
//! the cheapest one, which is where a Tucker certificate goes.
//!

pub mod arena;
pub mod baked;
pub mod budget;
pub mod layout;
pub mod lowering;
pub mod refusal;
mod region;
pub mod stream;

#[cfg(test)]
mod fixture;

use model_ir::{Classes, Def, Operands, Operation, Plan, Ty, resolve_classes};

pub use arena::{ArenaMap, Concurrency, EXPORT_SEAMS, RowExpr, Slot, Span, Window};
pub use baked::{
    Baked, EventId, Fallback, FallbackRow, FallbackTable, LayoutOrder, Lowering, Phase, Region,
};
pub use budget::{Budgets, DeviceProfile, FamilyCosts};
pub use layout::PqTree;
pub use stream::Forks;
pub use refusal::{Refusal, Share, Unrectangled};

/// The fact ceiling. The class sweep is `2^F`, and past 20 it stops being a
/// sweep — the same number `Cond::simplified` and `resolve_classes` state.
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
/// [`Refusal`], which names the reason and refuses the load. That is the
/// rewrite's doctrine kept whole: **no silent fallback.** A plan whose merges
/// do not cover their classes, whose budgets describe no fire, or whose values
/// have no rectangle is a load that does not happen — not a graph that is
/// quietly missing a window and computes anyway.
pub fn compile(plan: &Plan, budgets: &Budgets, profile: &DeviceProfile) -> Result<Baked, Refusal> {
    // P0. The compiler is the front door, so every ceiling the utilities
    // downstream ASSERT is checked here first and answered as a refusal: a
    // panic in a load path takes the process with it.
    accept(plan, budgets, profile)?;

    // P0 + P1, one walk. The model test suite calls the same function at
    // authoring time for the author's sake; here it is the accept pass and the
    // class table at once, and neither side pays for the other (decision #7).
    let classes = resolve_classes(plan).map_err(Refusal::Classes)?;

    // P1's acceptance half: a `Struct` value's readers must share the window
    // it was built in. It is asked off `node_mask` — the sweep's own answer,
    // before regions exist — because a region is a maximal run of nodes with
    // EQUAL masks, so the two readings are the same predicate and this one
    // needs no pass to have run. The authoring net in `crates/model/tests`
    // asks it in the same vocabulary against `resolve_classes` alone, which
    // is the earliest and cheapest instant a straddling model text can be
    // told; this is the front door's own restatement, on the load path.
    struct_readers_share_one_window(plan, &classes)?;

    // P5 then P2: phase per node, then maximal runs of equal (mask, phase).
    let regions = region::coalesce(plan, &classes);

    // P3. Which regions enter the graph behind a conditional node — and P6
    // below reads the answer, because `stream::forkable` refuses to fork one
    // that does. The order is the composition rule and it is a mechanism
    // rather than a preference: a conditional body is a child graph and a
    // fork's event pair is an edge inside one parent graph, so an arm that
    // was both is a dependency that cannot be expressed (see [`lowering`]).
    let mut regions = regions;
    lowering::lower(plan, &mut regions, &classes, budgets, profile);

    // P6. The dep DAG over the capture regions, the cost gate, the streams
    // and the event points — stamped into `regions` in place, because a fork
    // is a property OF a region rather than a second schedule beside it. What
    // comes back is the relation the carve is widened by, and P7 below is the
    // pass that was written waiting for it.
    let forks = stream::fork(plan, &mut regions, profile);
    let concurrency = if forks.pairs.is_empty() {
        Concurrency::sequential(&regions, plan.nodes.len())
    } else {
        Concurrency::with_pairs(&regions, plan.nodes.len(), forks.pairs.iter().copied())
    };

    // P4. The C1P instance is read off the region table, so it runs after P2
    // even though it is numbered before it; and it is read off the MASKS
    // rather than the bytes, so it owes the carve nothing and the carve owes
    // it nothing. What it decides is the order a fire seriates its rows in,
    // which is arithmetic on the descriptor, not an offset.
    let (order, fallback) = layout::seriate(plan, &regions, &classes, budgets, profile);

    // P7. The carve is the pass with something to get wrong, and
    // `ArenaMap::clashes` is what says it did not.
    let arena = arena::carve(plan, budgets, &classes, &concurrency)?;

    Ok(Baked {
        classes,
        regions,
        order,
        fallback,
        arena,
        concurrency,
        forks,
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
fn accept(plan: &Plan, budgets: &Budgets, profile: &DeviceProfile) -> Result<(), Refusal> {
    let facts = model_ir::fact_width(plan);
    if facts > MAX_FACTS {
        return Err(Refusal::TooManyFacts { facts });
    }

    if budgets.max_lanes == 0 {
        return Err(Refusal::Budget {
            what: "admit no lanes, so no fire can be assembled",
        });
    }
    if budgets.max_tokens == 0 {
        return Err(Refusal::Budget {
            what: "admit no token rows, so every rectangle is empty",
        });
    }
    if budgets.max_lanes > budgets.max_tokens {
        return Err(Refusal::Budget {
            what: "admit more lanes than token rows, and a lane carries at least one row",
        });
    }
    let mut previous = 0u32;
    for &bucket in &budgets.buckets {
        if bucket <= previous {
            return Err(Refusal::Budget {
                what: "list a bucket lattice that does not strictly ascend",
            });
        }
        if bucket > budgets.max_tokens {
            return Err(Refusal::Budget {
                what: "list a bucket past the token ceiling",
            });
        }
        previous = bucket;
    }

    if profile.sms == 0 {
        return Err(Refusal::Profile {
            what: "describes a device with no streaming multiprocessors",
        });
    }

    // **THE ONE PLACE THE DEPLOYMENT'S ADAPTER NUMBER AND THE MODEL'S MEET**
    // (design §8, decision 17). Capacity is a SHAPE — the leading axis of
    // every bank the model text marked `ParamSource::Registered` — because a
    // shape is what a weight table reserves and what the routed op indexes.
    // `Budgets::max_adapters` is what the deployment asked to be able to
    // register, and asking for more than the text can seat is a load that
    // does not happen rather than a registration that is refused later:
    // decision 17's budget is not an admission cap, so it has to be true
    // BEFORE the first registration, not discovered at one.
    //
    // A plan with no bank is exempt at `max_adapters == 0` and refused above
    // it, which is the honest reading of "this deployment wants adapters and
    // this model text has no seat for them".
    if budgets.max_adapters > 0 {
        let seats = plan
            .params
            .iter()
            .filter(|param| param.source == model_ir::ParamSource::Registered)
            .map(|param| param.shape.first().copied().unwrap_or(0))
            .min();
        match seats {
            None => {
                return Err(Refusal::AdapterCapacity {
                    asked: budgets.max_adapters,
                    seated: 0,
                });
            }
            Some(seated) if seated < u64::from(budgets.max_adapters) => {
                return Err(Refusal::AdapterCapacity {
                    asked: budgets.max_adapters,
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
/// every variant is one), and [`Refusal::Straddled`] argues why a schedule
/// cannot be sliced the way a tensor can. One pass over the nodes: the
/// DEFINING node stands in the window its builder is dispatched in by
/// construction, so what is compared is its class set against every reader's.
fn struct_readers_share_one_window(plan: &Plan, classes: &Classes) -> Result<(), Refusal> {
    let structs: Vec<bool> = plan
        .values
        .iter()
        .map(|value| matches!(value.ty, Ty::Struct(_)))
        .collect();

    let mut inputs = Vec::new();
    for (at, node) in plan.nodes.iter().enumerate() {
        inputs.clear();
        node.op.inputs(&mut inputs);
        for &read in &inputs {
            if !structs.get(read.0 as usize).copied().unwrap_or(false) {
                continue;
            }
            let Some(Def::Op(built_by)) = plan.values.get(read.0 as usize).map(|v| &v.def) else {
                continue;
            };
            let planned = &classes.node_mask[*built_by as usize];
            let reader = &classes.node_mask[at];
            if planned != reader {
                return Err(Refusal::Straddled {
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
pub fn collectives(plan: &Plan) -> Vec<u32> {
    plan.nodes
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
pub fn collectives_are_never_elided(baked: &Baked) -> bool {
    baked
        .regions
        .iter()
        .all(|r| !r.collective || r.lowering == Lowering::AlwaysLaunch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, fact};
    use model_ir::Cond;

    fn plan() -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always);
        let d = b.op(q, 8, fact(0));
        let p = b.op(q, 8, Cond::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always);
        b.append(y, Cond::Always);
        b.out(y);
        b
    }

    #[test]
    fn a_plan_bakes_into_an_artifact_whose_arena_is_sound() {
        let b = plan();
        let baked = compile(&b.plan, &Budgets::new(4, 16), &DeviceProfile::default())
            .expect("a covering plan bakes");

        assert_eq!(baked.classes.classes.len(), 2);
        assert!(!baked.regions.is_empty());
        assert_eq!(baked.template().len(), baked.regions.len());
        // P4 seated both windows: the two classes are trivially an interval
        // of any order, so nothing is owed a fallback.
        assert_eq!(baked.order.tree().expect("P4 seriated it").frontier(), [0, 1]);
        assert!(baked.fallback.rows.is_empty());
        assert!(baked.concurrency.pairs().is_empty());
        assert!(baked.arena.bytes > 0);
        assert!(baked.arena.clashes(&baked.concurrency).is_empty());
        assert!(collectives_are_never_elided(&baked));
        // Two two-node arms: fat at no profile and profitable at none, so
        // the one lowering that is correctness and not optimization.
        assert!(
            baked
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
        let m = b.op(x, 8, Cond::and(Cond::not(fact(0)), fact(1)));
        let o = b.merge(
            &[(d, fact(0)), (m, Cond::and(Cond::not(fact(0)), fact(1)))],
            8,
        );
        b.out(o);

        let refusal = compile(&b.plan, &Budgets::new(4, 16), &DeviceProfile::default())
            .expect_err("a hole is a refusal");
        let Refusal::Classes(faults) = &refusal else {
            panic!("the class sweep is what refused it: {refusal}")
        };
        assert_eq!(faults.len(), 1);
        assert!(refusal.say(&b.plan).contains("no arm holds there"));
    }

    #[test]
    fn a_budget_that_describes_no_fire_is_refused_before_anything_is_swept() {
        let b = plan();
        let profile = DeviceProfile::default();
        assert!(matches!(
            compile(&b.plan, &Budgets::new(0, 16), &profile),
            Err(Refusal::Budget { .. }),
        ));
        assert!(matches!(
            compile(&b.plan, &Budgets::new(4, 0), &profile),
            Err(Refusal::Budget { .. }),
        ));
        assert!(matches!(
            compile(&b.plan, &Budgets::new(32, 16), &profile),
            Err(Refusal::Budget { .. }),
        ));
        let mut lattice = Budgets::new(4, 16);
        lattice.buckets = vec![1, 8, 8];
        assert!(matches!(
            compile(&b.plan, &lattice, &profile),
            Err(Refusal::Budget { .. }),
        ));
        lattice.buckets = vec![1, 8, 64];
        assert!(matches!(
            compile(&b.plan, &lattice, &profile),
            Err(Refusal::Budget { .. }),
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
            compile(&b.plan, &Budgets::new(4, 16), &profile),
            Err(Refusal::Profile { .. }),
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
            compile(&b.plan, &Budgets::new(4, 16), &DeviceProfile::default()),
            Err(Refusal::TooManyFacts { facts: 21 }),
        );
    }

    #[test]
    fn a_collective_is_listed_and_its_region_is_never_elidable() {
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Cond::Always);
        let g = b.all_gather(a, 8, fact(0));
        let o = b.merge(&[(g, fact(0)), (a, Cond::not(fact(0)))], 8);
        b.out(o);

        assert_eq!(collectives(&b.plan), vec![1]);
        let baked =
            compile(&b.plan, &Budgets::new(4, 16), &DeviceProfile::default()).expect("bakes");
        assert!(baked.regions.iter().any(|r| r.collective));
        assert!(collectives_are_never_elided(&baked));
    }
}
