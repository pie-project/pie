//! The layout pass's [`FallbackTable`](model_compiler::FallbackTable), read:
//! what a shell does about a window the layout could not make one interval.
//!
//! [`Fallback::Split`] (run once per maximal interval) needs nothing from a
//! backend and is always correct, but not always fastest below the
//! copy/split crossover (`model_compiler::layout`'s `CROSSOVER_ROWS`).
//! [`Fallback::Copy`] (gather the window into one rectangle, run the
//! consumer once, scatter back) is served by any shell that answers
//! [`Serve`]; a backend that doesn't answers `false` and gets the split.
//! [`Fallback::Grouped`] and [`Fallback::View`] need an op that takes an
//! offset list rather than a rectangle, so the menu declines to choose them.
//!
//! A plan has one seriation per row space (token/patch) and therefore one
//! menu per row space, so every function here takes `axis` explicitly rather
//! than assuming the token table.

use core::ops::Range;

use crate::error::KernelError;
use model_compiler::{CompiledModel, Fallback, Phase, PqTree, Region};
use model_ir::{ClassSet, RowAxis};

/// Every answer the layout wrote for a region's nodes, deduplicated, in
/// bucket-lattice order. Empty is a promise, not an absence: a region with no
/// row here is one the layout seated as an interval, and a fire that finds it
/// fragmented anyway has a mismatched artifact/class table. Returned
/// unresolved (all answers, not one) since more than one can apply across the
/// bucket lattice — [`answer_at`] does the resolution.
#[must_use]
pub fn answers(compiled: &CompiledModel, axis: RowAxis, nodes: Range<u32>) -> Vec<Fallback> {
    let Some(table) = compiled.fallback_for(axis) else {
        return Vec::new();
    };
    let mut found: Vec<Fallback> = Vec::new();
    for row in &table.rows {
        if !nodes.contains(&row.node) || found.contains(&row.fallback) {
            continue;
        }
        found.push(row.fallback);
    }
    found
}

/// The layout's answer for one region at one bucket — the resolution
/// [`answers`] deliberately does not make, since the menu is bucket-keyed by
/// the cost model (a fallback carries different rows across the copy/split
/// crossover). `bucket` indexes [`Budget::buckets`](model_compiler::Budget::buckets);
/// a deployment with no lattice has one implicit bucket at index 0. Private:
/// [`copies`] is the one caller holding both the mask and the bucket.
#[must_use]
fn answer_at(
    compiled: &CompiledModel,
    axis: RowAxis,
    nodes: Range<u32>,
    bucket: u32,
) -> Option<Fallback> {
    compiled
        .fallback_for(axis)?
        .rows
        .iter()
        .find(|row| nodes.contains(&row.node) && row.buckets.contains(&bucket))
        .map(|row| row.fallback)
}

/// Is `Fallback::Copy` this artifact's answer, at this bucket, for every
/// region standing over `mask` (within `axis`)?
///
/// Keyed by mask, not by node range: a prepare region that carves an
/// attention schedule gets no row of its own (layout only constrains capture
/// regions), but must copy exactly when its capture readers do, since a
/// schedule carved once must match a rectangle gathered once. `mask` is the
/// key within one axis only — two regions in different row spaces never
/// share a window even with an identical mask.
#[must_use]
pub fn copies(compiled: &CompiledModel, axis: RowAxis, mask: &ClassSet, bucket: u32) -> bool {
    compiled
        .template()
        .iter()
        .enumerate()
        .filter(|(at, region)| &region.mask == mask && compiled.axis_of(*at) == axis)
        .any(|(_, region)| {
            answer_at(compiled, axis, region.nodes.clone(), bucket) == Some(Fallback::Copy)
        })
}

/// The shell's half of `Fallback::Copy`: what a backend must be able to do
/// before [`walk`](crate::fire::walk) will stop splitting. A copy is three
/// device steps around the consumer — gather scattered rows into a
/// contiguous rectangle, run the region's nodes once, scatter the answers
/// back — and all three are the shell's (scratch the arena doesn't carve,
/// and a movement kernel no `kernels-*` library publishes). Default is
/// `false`/unimplemented, so `impl Serve for MyRun {}` is a complete opt-out
/// (every fragmented window served as [`Fallback::Split`]).
///
/// # Contract
///
/// - [`copies`](Serve::copies) is asked once per region per fire, before the
///   launch loop turns; answering `true` promises `gather`/`scatter` will
///   succeed and that operand resolution already points at the gathered
///   rectangle.
/// - the gather/scatter pair brackets the nodes on the region's own stream
///   (gather before the first node, scatter after the last), so a copy is
///   ordered like the region itself.
/// - neither runs in a prepare pass (host work on an open stream).
pub trait Serve {
    /// Does this backend serve `region`'s fragmented window as a copy in this
    /// fire? Asked only of a region this fire actually found in pieces.
    fn copies(&self, _region: &Region) -> bool {
        false
    }

    /// Lay the window's scattered rows down as one rectangle.
    ///
    /// # Errors
    ///
    /// Whatever the backend's row movement answered. Unreachable by default
    /// — the walk calls it only behind a `true` [`copies`](Serve::copies) —
    /// and refuses rather than panics.
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

/// Did the layout answer [`Fallback::Grouped`] for this region: one launch
/// that walks the intervals itself, rather than one per interval? The only
/// fallback kind that changes the walk's trip count, so it must be read off
/// the table rather than inferred from the span count. Written for a consumer
/// whose every node the caller declared groupable
/// ([`DeviceProfile::grouped`](model_compiler::DeviceProfile::grouped)), at
/// every bucket, so `true` here is unconditional in the fire's size.
#[must_use]
pub fn grouped(compiled: &CompiledModel, axis: RowAxis, nodes: Range<u32>) -> bool {
    compiled.fallback_for(axis).is_some_and(|table| {
        table
            .rows
            .iter()
            .any(|row| nodes.contains(&row.node) && row.fallback == Fallback::Grouped)
    })
}

/// The most launches a mask's window can ever cost: how many runs it breaks
/// into under `axis`'s own class order, as the artifact ships it. An upper
/// bound on every fire, since dropping an absent class can only close a gap
/// between two of the mask's own. `1` for every mask the layout seated;
/// derived from the order (rather than read off the table) because the table
/// doesn't always state `r` — a lattice entirely below the copy/split
/// crossover, or a prepare region the layout never constrained.
#[must_use]
pub fn bound(compiled: &CompiledModel, axis: RowAxis, mask: &ClassSet) -> u32 {
    let Some(order) = compiled.order_for(axis) else {
        return 1;
    };
    let classes = compiled.classes.classes.len();
    let order = order.class_order(&ClassSet::of(0..classes));
    let mask: Vec<u8> = mask.iter().map(|class| class as u8).collect();
    PqTree::runs(&order, &mask).max(1)
}

/// Is this region's window one the layout promised consecutive? The question
/// `Fault::Fragmented` asks, with two ways to answer no: a region with a
/// [`FallbackTable`](model_compiler::FallbackTable) row the layout withdrew,
/// or a prepare region never offered to the C1P instance at all (prepare is
/// host work outside the graph) — the latter is not a bake-integrity
/// failure, just the same slow path of one schedule carved per interval.
#[must_use]
pub fn promised(compiled: &CompiledModel, axis: RowAxis, region: &Region) -> bool {
    region.phase == Phase::Capture && answers(compiled, axis, region.nodes.clone()).is_empty()
}

/// How many distinct windows this artifact can ever have in pieces. A
/// load-time count that a copy's staging is reserved against, since gathered
/// addresses are recorded into graphs that are never re-captured. Counts
/// distinct masks (not regions — many layers can share one window), excluding
/// masks the layout seated (never gathered).
#[must_use]
pub fn fragmentable(compiled: &CompiledModel) -> usize {
    let mut seen: Vec<&ClassSet> = Vec::new();
    for (at, region) in compiled.template().iter().enumerate() {
        if bound(compiled, compiled.axis_of(at), &region.mask) > 1
            && !seen.contains(&&region.mask)
        {
            seen.push(&region.mask);
        }
    }
    seen.len()
}

/// The most launches any region of this artifact can cost, over every fire
/// it can ever be handed. A load-time bound a shell sizes per-run state
/// against (addresses recorded into graphs that are never re-captured, so
/// they can't be carved per fire); an artifact seated whole answers `1`.
#[must_use]
pub fn max_runs(compiled: &CompiledModel) -> u32 {
    compiled
        .template()
        .iter()
        .enumerate()
        .map(|(at, region)| bound(compiled, compiled.axis_of(at), &region.mask))
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

    /// The smallest plan the layout cannot seat: four classes whose guards
    /// demand every pair of the four-set be an interval, but an order of four
    /// has only three adjacent pairs, so the C1P instance is infeasible and
    /// the layout withdraws.
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
    fn the_bound_is_the_run_count_p4_measured_on_the_order_it_shipped() {
        let b = crossing();
        // Past the crossover, so the menu states an `r` to check against.
        let wide = Budget {
            max_lanes: 8,
            max_tokens: 4096,
            buckets: vec![64, 4096],
            max_adapters: 0,
        };
        let compiled = compile(&b.trace, &wide, &DeviceProfile::default()).expect("the fixture bakes");

        // The two answers agree: `bound` derives the same `r` the layout counted.
        let mut checked = 0;
        for (at, region) in compiled.template().iter().enumerate() {
            let axis = compiled.axis_of(at);
            let stated = answers(&compiled, axis, region.nodes.clone())
                .into_iter()
                .find_map(|answer| match answer {
                    Fallback::Split { r } => Some(r),
                    _ => None,
                });
            if let Some(stated) = stated {
                assert_eq!(bound(&compiled, axis, &region.mask), stated, "{:?}", region.nodes);
                assert!(stated > 1, "a withdrawn consumer costs more than one launch");
                checked += 1;
            }
        }
        assert!(checked > 0, "the fixture withdraws at least one consumer");

        // A seated region is bounded at one launch.
        for (at, region) in compiled.template().iter().enumerate() {
            let axis = compiled.axis_of(at);
            if promised(&compiled, axis, region) {
                assert_eq!(bound(&compiled, axis, &region.mask), 1, "{:?}", region.nodes);
            }
        }
    }

}
