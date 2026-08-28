//! P4's [`FallbackTable`](model_compiler::FallbackTable), read — what a shell
//! does about a window the layout could not make one interval (design §3).
//!
//! **THE TABLE WAS BAKED AND NOBODY READ IT.** `model_compiler`'s P4 solves
//! one C1P instance over the whole plan and, when a consumer's class set
//! cannot be an interval alongside everybody else's, WITHDRAWS that
//! constraint and writes the consumer's nodes into
//! [`FallbackTable`](model_compiler::FallbackTable) with an answer per bucket
//! range. Every driver in this repo ignored that table and treated a
//! fragmented window as a hard fault instead, which is correct only for a
//! catalog that bakes no rows. Today's does. Compiled for CUDA over the
//! fourteen-point bucket lattice at an adapter capacity the texts can seat,
//! the five qwen artifacts owe rows for the `captures_scores` window — C4's
//! axis, which CROSSES `qo_one` where every earlier axis nested inside it, and
//! a crossing family is not laminar and therefore not always C1P:
//!
//! ```text
//! qwen35-d0.8b, d3b    12 rows over  6 attention.prefill_lse nodes
//! qwen35-a3b, a3b-tp2  20 rows over 10 of them
//! qwen36-27b           84 rows over 42 nodes, its MTP head's whole region
//! ```
//!
//! Two rows per node, because the answer is bucket-dependent:
//! [`Fallback::Copy`] below the crossover and [`Fallback::Split`] above it.
//!
//! # What this substrate serves, and what it does not
//!
//! [`Fallback::Split`] — run the kernel once per maximal interval — is the
//! one entry on the menu that needs nothing from a backend, and it is the one
//! `driver::fire::walk` implements: the window becomes a list of spans
//! ([`WindowTable::spans`](crate::fire::WindowTable::spans)) and the region's
//! nodes are dispatched once per span, each over its own pointer and extent.
//! It is ALWAYS CORRECT and it is not always fastest — tart measured a 2-way
//! split at 1.82× the ideal against a copy's 1.07× at 64 rows, converging by
//! 512 (`model_compiler::layout`'s `CROSSOVER_ROWS`).
//!
//! So a [`Fallback::Copy`] row — which is what the table asks for at every
//! bucket below the crossover — is served as a split today, and that is a
//! performance debt rather than a correctness one. What `Copy` needs is a
//! scratch rectangle the fire's arena does not carve and a gather/scatter
//! pair no `kernels` entry publishes; [`Fallback::Grouped`] and
//! [`Fallback::View`] need the same kind of thing (an op that takes an offset
//! list rather than a rectangle), which is exactly why `model_compiler::menu`
//! declines to choose them. This module is where the answer is read, so this
//! module is where a shell that grows those kernels states that it can serve
//! them.

use core::ops::Range;

use model_compiler::{Baked, Fallback, Phase, PqTree, Region};
use model_ir::ClassSet;

/// Every answer P4 wrote for a region's nodes, deduplicated, in the order the
/// bucket lattice states them.
///
/// **EMPTY IS A PROMISE, NOT AN ABSENCE.** A region with no row here is a
/// region P4 seated: its class set IS an interval of the order the artifact
/// ships, so every sub-order a fire filters out of that order keeps it one —
/// and a fire that finds such a window fragmented has an artifact and a class
/// table that were not built from each other. That is the reading
/// `driver_cuda::Fault::Fragmented` keeps, narrowed to it.
///
/// A region with rows is a region P4 could not seat and owes an answer for,
/// and the caller's job is to pay one. The answers are returned rather than
/// resolved against this fire's bucket because the only entry a shell serves
/// today is [`Fallback::Split`] and it serves it at every bucket; the day one
/// serves [`Fallback::Copy`] below the crossover, the bucket is the argument
/// that picks between them and `FallbackRow::buckets` indexes
/// `Budgets::buckets` to say where the cut is.
///
/// The scan is linear over a table with tens of rows and is asked only about
/// a region a fire ALREADY found fragmented, which is the rare case by
/// construction — the whole point of P4 is that it is rare.
#[must_use]
pub fn answers(baked: &Baked, nodes: Range<u32>) -> Vec<Fallback> {
    let mut found: Vec<Fallback> = Vec::new();
    for row in &baked.fallback.rows {
        if !nodes.contains(&row.node) || found.contains(&row.fallback) {
            continue;
        }
        found.push(row.fallback);
    }
    found
}

/// The most launches a mask's window can ever cost: how many runs it breaks
/// into under the class order the artifact SHIPS.
///
/// **AN UPPER BOUND ON EVERY FIRE, WHICH IS WHY IT IS THE NUMBER A LOAD SIZES
/// AGAINST.** A fire orders its classes by that same order with the absent
/// ones dropped
/// ([`LayoutOrder::class_order`](model_compiler::LayoutOrder::class_order)),
/// and dropping a class can only CLOSE a gap — two of the mask's classes that
/// were separated only by absent ones become adjacent. So no composition can
/// find more runs than this, and a fire that does has a class order that did
/// not come from P4's tree.
///
/// It is `1` for every mask P4 seated — that IS the promise — and it agrees
/// with the `r` on a [`Fallback::Split`] row for every mask P4 withdrew,
/// because that `r` is counted the same way on the same frontier. Derived
/// from the order rather than read off the table because the table does not
/// always state it: a bucket lattice sitting entirely below the copy/split
/// crossover gets [`Fallback::Copy`] at every bucket and no `r` at all, and a
/// PREPARE region gets no row whatever the lattice — `model_compiler::layout`
/// offers only capture regions to the C1P instance, so a plan builder's
/// window is neither promised nor answered for.
#[must_use]
pub fn bound(baked: &Baked, mask: &ClassSet) -> u32 {
    let classes = baked.classes.classes.len();
    let order = baked.order.class_order(&ClassSet::of(0..classes), None);
    let mask: Vec<u8> = mask.iter().map(|class| class as u8).collect();
    PqTree::runs(&order, &mask).max(1)
}

/// Is this region's window one P4 PROMISED consecutive?
///
/// **THE QUESTION `Fault::Fragmented` NOW ASKS**, and it has two ways of
/// answering no. A region with a [`FallbackTable`](model_compiler::FallbackTable)
/// row is one P4 withdrew and owes an answer for. A PREPARE region was never
/// offered to the C1P instance at all — `model_compiler::layout::constrains`
/// takes capture regions only, on the reading that prepare is host work
/// outside the graph — so P4 makes it no promise either way, and it is not a
/// bake-integrity failure when a plan builder's window comes back in pieces.
/// It is the same slow path: one schedule carved per interval.
///
/// (That the compiler does not constrain a prepare region whose window a
/// builder demonstrably walks is worth its own look; today's catalog shows it
/// plainly — qwen3.5's `attention.plan_prefill` states the same
/// `captures_scores` mask its six `prefill_lse` readers do, and only the
/// readers get rows. Adding it as a constraint is a P4 change with a
/// blast radius across every baked order, so it is named here rather than
/// made here.)
#[must_use]
pub fn promised(baked: &Baked, region: &Region) -> bool {
    region.phase == Phase::Capture && answers(baked, region.nodes.clone()).is_empty()
}

/// The most launches ANY region of this artifact can cost, over every fire it
/// can ever be handed.
///
/// **A LOAD-TIME NUMBER, WHICH IS THE POINT.** A shell sizes per-run state
/// against it — one attention-schedule grant per run, one plan-payload slot
/// per run — and those are addresses recorded into graphs that are never
/// re-captured, so they cannot be carved per fire. It is a bound and not a
/// measurement: most fires split nothing, and every artifact P4 seated whole
/// answers `1`, which is the shape that state had before the split existed.
#[must_use]
pub fn max_runs(baked: &Baked) -> u32 {
    baked
        .template()
        .iter()
        .map(|region| bound(baked, &region.mask))
        .max()
        .unwrap_or(1)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::fixture::{Build, fact};
    use model_compiler::{Budgets, DeviceProfile, compile};
    use model_ir::Cond;

    /// The smallest plan P4 cannot seat: two facts, and a window on each axis
    /// AND on the diagonal.
    ///
    /// Four classes, and the guards demand `{1,3}`, `{0,2}`, `{2,3}`, `{0,1}`,
    /// `{1,2}` and `{0,3}` all be intervals of one order — every pair of a
    /// four-set, where an order of four has only three adjacent pairs. So the
    /// C1P instance is infeasible however it is solved, P4 withdraws the
    /// constraints it cannot take, and the withdrawn consumers get rows. The
    /// arms are merged rather than left dangling because the demand walk roots
    /// at the seam: an op nothing reads is a dead node and states no window.
    fn crossing() -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let mut v = b.op(x, 4, Cond::Always);
        let xor = Cond::or(
            Cond::and(fact(0), Cond::not(fact(1))),
            Cond::and(Cond::not(fact(0)), fact(1)),
        );
        for axis in [fact(0), fact(1), xor] {
            let taken = b.op(v, 4, axis.clone());
            let other = b.op(v, 4, Cond::not(axis.clone()));
            v = b.merge(&[(taken, axis.clone()), (other, Cond::not(axis))], 4);
        }
        let y = b.op(v, 4, Cond::Always);
        b.out(y);
        b
    }

    #[test]
    fn a_seated_region_is_owed_nothing_and_a_withdrawn_one_names_its_answer() {
        let b = crossing();
        let baked = compile(&b.plan, &Budgets::new(8, 64), &DeviceProfile::default())
            .expect("the fixture bakes");
        assert!(
            !baked.fallback.rows.is_empty(),
            "three crossing windows over four classes are not C1P",
        );

        // The region the table names, and one it does not.
        let owed: Vec<u32> = baked.fallback.rows.iter().map(|row| row.node).collect();
        for region in baked.template() {
            let answers = answers(&baked, region.nodes.clone());
            let named = region.nodes.clone().any(|node| owed.contains(&node));
            assert_eq!(named, !answers.is_empty(), "region {:?}", region.nodes);
        }
    }

    #[test]
    fn the_bound_is_the_run_count_p4_measured_on_the_order_it_shipped() {
        let b = crossing();
        // A lattice that reaches past the crossover, so the menu writes the
        // split entry beside the copy one and states an `r` this can be
        // checked against.
        let wide = Budgets {
            max_lanes: 8,
            max_tokens: 4096,
            buckets: vec![64, 4096],
            max_adapters: 0,
        };
        let baked = compile(&b.plan, &wide, &DeviceProfile::default()).expect("the fixture bakes");

        // THE TWO ANSWERS AGREE, which is the claim `bound`'s doc makes about
        // deriving the number from the order rather than reading it off the
        // table: P4 counts `r` on the frontier it shipped, and so does this.
        let mut checked = 0;
        for region in baked.template() {
            let stated = answers(&baked, region.nodes.clone())
                .into_iter()
                .find_map(|answer| match answer {
                    Fallback::Split { r } => Some(r),
                    _ => None,
                });
            if let Some(stated) = stated {
                assert_eq!(bound(&baked, &region.mask), stated, "{:?}", region.nodes);
                assert!(stated > 1, "a withdrawn consumer costs more than one launch");
                checked += 1;
            }
        }
        assert!(checked > 0, "the fixture withdraws at least one consumer");

        // A seated region is bounded at one launch — the promise, in the same
        // vocabulary.
        for region in baked.template() {
            if promised(&baked, region) {
                assert_eq!(bound(&baked, &region.mask), 1, "{:?}", region.nodes);
            }
        }
    }

    #[test]
    fn a_prepare_region_is_promised_nothing_because_p4_never_constrained_it() {
        let b = crossing();
        let baked = compile(&b.plan, &Budgets::new(8, 64), &DeviceProfile::default())
            .expect("the fixture bakes");
        for region in baked.template() {
            if region.phase == Phase::Prepare {
                assert!(
                    !promised(&baked, region),
                    "a prepare region's window is neither promised nor answered for",
                );
            }
        }
    }
}
