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
//! **P0, P1, P2, P4, P5 and P7 are here.** P3, P6 and P8 are not, and the
//! seams they will fill are TYPED rather than stubbed: `Lowering` has all
//! three variants and only `AlwaysLaunch` is ever constructed, and
//! `Concurrency::sequential` is what one stream means. Every one of those is a
//! true statement about a v1 artifact, not a placeholder — a plan baked today
//! runs correctly, just not as fast as it will.
//!
//! P4 ships the C1P instance whole (see [`layout`]): a real PQ-tree over the
//! plan's classes, the feasible SET rather than one witness ordering, so that
//! the fire path's stability pick has something to ask. What it does not yet
//! do is ask — [`LayoutOrder::class_order`] takes last fire's order and
//! ignores it — and it withdraws the last conflicting constraint rather than
//! the cheapest one, which is where a Tucker certificate goes.
//!
//! What is genuinely absent is absent: see the module docs on [`Baked`] for
//! the four design §2 fields that belong to P8 and why an empty one would be a
//! claim rather than a gap.

pub mod arena;
pub mod baked;
pub mod budget;
pub mod layout;
pub mod refusal;
mod region;

#[cfg(test)]
mod fixture;

use model_ir::{Operation, Plan, resolve_classes};

pub use arena::{ArenaMap, Concurrency, RowExpr, Slot, Span, Window};
pub use baked::{
    Baked, Fallback, FallbackRow, FallbackTable, LayoutOrder, Lowering, Phase, Region,
};
pub use budget::{Budgets, DeviceProfile};
pub use layout::PqTree;
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

    // P5 then P2: phase per node, then maximal runs of equal (mask, phase).
    let regions = region::coalesce(plan, &classes);

    // P6's answer, for now: one stream, so nothing runs beside anything.
    let concurrency = Concurrency::sequential(&regions, plan.nodes.len());

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
/// **A COLLECTIVE IS NEVER ELIDED** (decision #5). Trivially true in v1, where
/// every region is [`Lowering::AlwaysLaunch`] — which is exactly why it is
/// written now: the day P3 starts choosing, this is the assertion that has
/// been standing there all along, rather than one somebody has to think to
/// add.
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
        // v1 lowers everything the one way that is correctness and not
        // optimization.
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
