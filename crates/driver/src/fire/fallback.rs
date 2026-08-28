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
//! [`Fallback::Copy`] — gather the window's scattered rows into one
//! contiguous rectangle, run the consumer ONCE over it, scatter the answers
//! back — is the entry the table asks for at every bucket below the
//! crossover, and it is served now by any shell that answers [`Serve`]. It
//! needs two things a split does not: a scratch rectangle the fire's arena
//! deliberately does not carve (`model_compiler::arena`'s carve has no spare)
//! and a row gather/scatter pair (`kernels_cuda::layout::gather_rows`). Both
//! are the SHELL's, which is why the choice is asked of the shell here rather
//! than decided for it: [`Serve::copies`] is a backend saying it has them,
//! and a backend that has not answers `false` and gets the split it always
//! got.
//!
//! [`Fallback::Grouped`] and [`Fallback::View`] still need what they always
//! needed — an op that takes an offset list rather than a rectangle — which
//! is exactly why `model_compiler::layout`'s menu declines to choose them.

use core::ops::Range;

use kernels::KernelError;
use model_compiler::{CompiledModel, Fallback, Phase, PqTree, Region};
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
/// and the caller's job is to pay one. The answers are returned UNRESOLVED —
/// all of them, in lattice order — because the question "what does P4 say
/// about this node" has more than one answer and pretending otherwise is what
/// hid the copy for as long as it was hidden. [`answer_at`] is the resolution,
/// and it takes the bucket because only the caller holds the lattice that
/// `FallbackRow::buckets` indexes.
///
/// The scan is linear over a table with tens of rows and is asked only about
/// a region a fire ALREADY found fragmented, which is the rare case by
/// construction — the whole point of P4 is that it is rare.
#[must_use]
pub fn answers(compiled: &CompiledModel, nodes: Range<u32>) -> Vec<Fallback> {
    let mut found: Vec<Fallback> = Vec::new();
    for row in &compiled.fallback.rows {
        if !nodes.contains(&row.node) || found.contains(&row.fallback) {
            continue;
        }
        found.push(row.fallback);
    }
    found
}

/// P4's answer for one region AT ONE BUCKET, which is the resolution
/// [`answers`] deliberately does not make.
///
/// **THE MENU IS BUCKET-KEYED BECAUSE THE COST MODEL IS**
/// (`model_compiler::layout`'s `CROSSOVER_ROWS`): a two-way split of a 64-row
/// GEMM measured 1.82x the ideal against a copy's 1.07x, and by 2048 rows
/// they converge and the split is free. So a node owed a fallback carries two
/// rows over a bucket lattice that straddles the crossover, and asking "what
/// does P4 say about this node" without saying which fire is asking gets both
/// of them back.
///
/// `bucket` INDEXES [`Budget::buckets`](model_compiler::Budget::buckets) —
/// which is what `FallbackRow::buckets` is a range of — and a deployment that
/// declared no lattice has one implicit bucket at index 0. The caller holds
/// the budgets and this crate does not, which is why the index is an
/// argument: `Composition::bucket` is the bucket's ROW COUNT, and turning
/// that into a position is a lookup in a table only the shell has.
#[must_use]
pub fn answer_at(compiled: &CompiledModel, nodes: Range<u32>, bucket: u32) -> Option<Fallback> {
    compiled
        .fallback
        .rows
        .iter()
        .find(|row| nodes.contains(&row.node) && row.buckets.contains(&bucket))
        .map(|row| row.fallback)
}

/// Is `Fallback::Copy` this artifact's answer, at this bucket, for every
/// region standing over `mask`?
///
/// **THE MASK IS THE KEY AND THE NODES ARE NOT, AND A PREPARE REGION IS
/// WHY.** A copy runs its consumer ONCE over a gathered rectangle, and a
/// consumer of an attention schedule cannot do that unless the schedule was
/// also carved once — over the same union of runs, at the same request count.
/// P4 offers only capture regions to its C1P instance
/// (`model_compiler::layout::constrains`), so the builder that carves the
/// schedule is owed no row at all even though its window fragments in exactly
/// the same fires as its readers'; [`promised`] already says so. If the
/// builder split while its reader copied, the one gathered launch would read
/// a schedule describing the first interval's lanes and index past it for
/// every request after them — wrong logits, no fault.
///
/// So the question is asked of the MASK, and the builder inherits its
/// readers' answer because it shares their window. That the two masks are
/// equal is not a hope: `driver_cuda::window::no_schedule_straddles_its_readers`
/// refuses a bake where they differ, at load, by name.
///
/// A mask no region is owed a row for answers `false` — which is every mask
/// P4 seated, where the question never arises because the window is one
/// interval anyway.
#[must_use]
pub fn copies(compiled: &CompiledModel, mask: &ClassSet, bucket: u32) -> bool {
    compiled
        .template()
        .iter()
        .filter(|region| &region.mask == mask)
        .any(|region| answer_at(compiled, region.nodes.clone(), bucket) == Some(Fallback::Copy))
}

/// **THE SHELL'S HALF OF `Fallback::Copy`**: what a backend must be able to
/// do before [`walk`](crate::fire::walk) will stop splitting.
///
/// A copy is three device steps around the consumer — gather the window's
/// scattered rows into a contiguous rectangle, run the region's nodes once
/// over it, scatter the answers back — and all three are the shell's: the
/// rectangle is scratch the arena does not carve, and the movement is a
/// kernel `kernels` publishes no entry for. The walk knows WHEN a copy is
/// owed (P4's table, read by [`copies`]) and nothing about how to pay one, so
/// the two halves meet here.
///
/// **THE DEFAULT IS `false`, AND THAT IS THE WHOLE COMPATIBILITY STORY.** A
/// backend that says nothing serves every fragmented window as
/// [`Fallback::Split`], which is what every backend did before this trait
/// existed and is always correct — `model_compiler::layout`'s menu is a cost
/// model, not a semantics. So `impl Serve for MyRun {}` is a complete
/// implementation, and a shell opts in one method at a time.
///
/// # The contract, both ways
///
/// - **[`copies`](Serve::copies) is asked once per region per fire**, after
///   the sink has been told which region this is, and before the launch loop
///   turns. A backend that answers `true` is promising that the two calls
///   below will succeed and that its operand resolution already points at the
///   gathered rectangle — the walk dispatches the region's nodes exactly as
///   it would have, and cannot tell the difference.
/// - **the pair brackets the nodes, on the region's own stream.** Gather is
///   enqueued before the first node and scatter after the last, inside
///   whatever conditional and event brackets the region carries, so a copy is
///   ordered against the region's producers and consumers the way the region
///   itself is.
/// - **neither runs in a prepare pass.** They are launches, and a prepare
///   pass is host work on an open stream; the walk filters them by the same
///   [`Phases`](crate::fire::Phases) rule it filters nodes by.
pub trait Serve {
    /// Does this backend serve `region`'s fragmented window as a copy in this
    /// fire?
    ///
    /// Asked ONLY of a region this fire actually found in pieces, so a
    /// backend may answer it by reading state it built for exactly those.
    /// The honest `false` — no gather kernel, no scratch, or a deployment
    /// that turned the path off — costs nothing but the split.
    fn copies(&self, _region: &Region) -> bool {
        false
    }

    /// Lay the window's scattered rows down as one rectangle.
    ///
    /// # Errors
    ///
    /// Whatever the backend's row movement answered. The default is
    /// unreachable — the walk calls it only behind a `true`
    /// [`copies`](Serve::copies) — and refuses rather than panicking, because
    /// a backend that overrides one method and not the other should learn it
    /// from a fire and not from a crash.
    fn gather(&mut self, _region: &Region) -> Result<(), KernelError> {
        Err(unserved("gather"))
    }

    /// Put the answers back where the rows came from.
    ///
    /// # Errors
    ///
    /// As [`gather`](Serve::gather).
    fn scatter(&mut self, _region: &Region) -> Result<(), KernelError> {
        Err(unserved("scatter"))
    }
}

fn unserved(half: &'static str) -> KernelError {
    KernelError::Backend {
        op: "fallback.copy",
        detail: format!("this backend answered `Serve::copies` but publishes no row {half}"),
    }
}

/// Did P4 answer [`Fallback::Grouped`] for this region — ONE launch that walks
/// the intervals itself, rather than one launch per interval?
///
/// **THE ONE QUESTION `driver::fire::walk` ASKS ABOUT A FALLBACK KIND**, and
/// the first one it has ever asked. Every other entry on the menu is served by
/// the same loop at a different cost: `Split { r }` IS the loop, `Copy` is
/// served as a split and owes a copy, `View` is unbuilt. `Grouped` is the one
/// that changes the loop's TRIP COUNT, so the walk cannot serve it by reading
/// the number of spans and must read the table instead — see that function's
/// rule 4 for what the branch cost and why it was still the right shape.
///
/// P4 writes this row for a consumer whose every node the caller declared
/// groupable ([`DeviceProfile::grouped`](model_compiler::DeviceProfile::grouped)),
/// and it writes it at every bucket, so a `true` here is unconditional in the
/// fire's size. A shell that reads it must serve the region's whole node range
/// in one dispatch over the segment list; a shell that cannot must not have
/// named the ops.
///
/// Asked only of a region a fire ALREADY found fragmented — for a window that
/// covers one interval the two answers are the same launch — which keeps the
/// linear scan on the rare path where [`answers`] puts it.
#[must_use]
pub fn grouped(compiled: &CompiledModel, nodes: Range<u32>) -> bool {
    compiled
        .fallback
        .rows
        .iter()
        .any(|row| nodes.contains(&row.node) && row.fallback == Fallback::Grouped)
}


/// The most launches a mask's window can ever cost: how many runs it breaks
/// into under the class order the artifact SHIPS.
///
/// **AN UPPER BOUND ON EVERY FIRE, WHICH IS WHY IT IS THE NUMBER A LOAD SIZES
/// AGAINST.** A fire orders its classes by that same order with the absent
/// ones dropped
/// ([`ClassOrder::class_order`](model_compiler::ClassOrder::class_order)),
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
pub fn bound(compiled: &CompiledModel, mask: &ClassSet) -> u32 {
    let classes = compiled.classes.classes.len();
    let order = compiled.order.class_order(&ClassSet::of(0..classes), None);
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
pub fn promised(compiled: &CompiledModel, region: &Region) -> bool {
    region.phase == Phase::Capture && answers(compiled, region.nodes.clone()).is_empty()
}

/// How many DISTINCT windows this artifact can ever have in pieces.
///
/// **A LOAD-TIME COUNT, AND WHAT A COPY'S STAGING IS RESERVED AGAINST.** A
/// gathered window carries per-lane tables the fire stages beside the
/// boundary vectors, and those addresses are recorded into graphs that are
/// never re-captured — so the room for them is taken once, at the ceiling, in
/// the same breath as everything else in `driver_cuda::inputs`.
///
/// It is the count of distinct MASKS, not of regions: sixty layers stating
/// one window state one window, and the fire's window table deduplicates them
/// for the same reason. Masks P4 seated are excluded, because
/// [`bound`] is `1` for them and a window that is never in pieces is never
/// gathered.
#[must_use]
pub fn fragmentable(compiled: &CompiledModel) -> usize {
    let mut seen: Vec<&ClassSet> = Vec::new();
    for region in compiled.template() {
        if bound(compiled, &region.mask) > 1 && !seen.contains(&&region.mask) {
            seen.push(&region.mask);
        }
    }
    seen.len()
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
pub fn max_runs(compiled: &CompiledModel) -> u32 {
    compiled
        .template()
        .iter()
        .map(|region| bound(compiled, &region.mask))
        .max()
        .unwrap_or(1)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::fixture::{Build, fact};
    use model_compiler::{Budget, DeviceProfile, compile};
    use model_ir::Guard;

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
        let mut v = b.op(x, 4, Guard::Always);
        let xor = Guard::or(
            Guard::and(fact(0), Guard::not(fact(1))),
            Guard::and(Guard::not(fact(0)), fact(1)),
        );
        for axis in [fact(0), fact(1), xor] {
            let taken = b.op(v, 4, axis.clone());
            let other = b.op(v, 4, Guard::not(axis.clone()));
            v = b.merge(&[(taken, axis.clone()), (other, Guard::not(axis))], 4);
        }
        let y = b.op(v, 4, Guard::Always);
        b.out(y);
        b
    }

    #[test]
    fn a_seated_region_is_owed_nothing_and_a_withdrawn_one_names_its_answer() {
        let b = crossing();
        let compiled = compile(&b.trace, &Budget::new(8, 64), &DeviceProfile::default())
            .expect("the fixture bakes");
        assert!(
            !compiled.fallback.rows.is_empty(),
            "three crossing windows over four classes are not C1P",
        );

        // The region the table names, and one it does not.
        let owed: Vec<u32> = compiled.fallback.rows.iter().map(|row| row.node).collect();
        for region in compiled.template() {
            let answers = answers(&compiled, region.nodes.clone());
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
        let wide = Budget {
            max_lanes: 8,
            max_tokens: 4096,
            buckets: vec![64, 4096],
            max_adapters: 0,
        };
        let compiled = compile(&b.trace, &wide, &DeviceProfile::default()).expect("the fixture bakes");

        // THE TWO ANSWERS AGREE, which is the claim `bound`'s doc makes about
        // deriving the number from the order rather than reading it off the
        // table: P4 counts `r` on the frontier it shipped, and so does this.
        let mut checked = 0;
        for region in compiled.template() {
            let stated = answers(&compiled, region.nodes.clone())
                .into_iter()
                .find_map(|answer| match answer {
                    Fallback::Split { r } => Some(r),
                    _ => None,
                });
            if let Some(stated) = stated {
                assert_eq!(bound(&compiled, &region.mask), stated, "{:?}", region.nodes);
                assert!(stated > 1, "a withdrawn consumer costs more than one launch");
                checked += 1;
            }
        }
        assert!(checked > 0, "the fixture withdraws at least one consumer");

        // A seated region is bounded at one launch — the promise, in the same
        // vocabulary.
        for region in compiled.template() {
            if promised(&compiled, region) {
                assert_eq!(bound(&compiled, &region.mask), 1, "{:?}", region.nodes);
            }
        }
    }

    #[test]
    fn a_prepare_region_is_promised_nothing_because_p4_never_constrained_it() {
        let b = crossing();
        let compiled = compile(&b.trace, &Budget::new(8, 64), &DeviceProfile::default())
            .expect("the fixture bakes");
        for region in compiled.template() {
            if region.phase == Phase::Prepare {
                assert!(
                    !promised(&compiled, region),
                    "a prepare region's window is neither promised nor answered for",
                );
            }
        }
    }
}
