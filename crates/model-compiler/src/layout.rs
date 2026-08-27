//! P4 — the layout. One global row order, chosen so that as many windowed
//! consumers as possible read a contiguous block (design §3, decision #8).
//!
//! # The problem, stated exactly
//!
//! A fire's rows are seriated by CLASS: every lane of class `c` sits in one
//! block, and the blocks sit in some order `π`. A windowed structural consumer
//! runs over the rows of the classes its guard admits, so it is **one kernel
//! over a pointer and an extent iff its class set is an interval of `π`** —
//! otherwise it is several launches, or a copy, or nothing correct at all.
//!
//! Build the 0/1 matrix with one row per consumer and one column per class.
//! "Can the columns be permuted so that every row's ones are consecutive?" is
//! verbatim the **Consecutive-Ones Property**, decidable — and a witness
//! constructible — in linear time by PQ-trees (Booth & Lueker, JCSS 1976).
//! The rewrite's `solve_scs` pre-sorted every span by a global rank map and
//! then ran greedy maximum-overlap merging, which is the textbook heuristic
//! for Shortest Common Superstring: an NP-hard problem, solved
//! approximately, over a search space the pre-sort had already destroyed, with
//! a `(0, 0)` return when the span was not found (`tart/evidence/
//! layout_planning.md`). This is the same question asked correctly.
//!
//! # One instance, not one per buffer
//!
//! tart's warning is that solving C1P per edge buffer is a trap: if `b1` is
//! written under `π1` and read into a consumer whose other input `b2` is laid
//! out under `π2 ≠ π1`, the mismatch IS the memcpy the design exists to
//! remove. It does not arise here, and the reason is structural rather than
//! lucky: P7 carves ONE arena, every value in it is addressed by the same row
//! index, and a fire has one row order. So there is one C1P instance for the
//! whole plan and the global-consistency problem is solved by construction.
//!
//! # Which consumers are rows of the matrix
//!
//! A capture-phase region whose mask is neither empty nor every class. Empty
//! is a region no class runs (`Classes::dead`); every class is a region that
//! runs over all rows, and "all rows" is an interval of every ordering, so
//! neither constrains anything. Prepare-phase regions are host work outside
//! the graph and read no rows.
//!
//! Correction and weight-varied ops — LoRA's `ΔW·x`, MoE's routed banks — are
//! **excluded** (decision #9): those lower to gather → grouped → scatter,
//! measured at 3.4x the ideal against 33x for a split, so the gather absorbs
//! the discontiguity and paying for it twice in the row order would constrain
//! the layout for nothing. No op family in the IR declares itself one yet;
//! [`gather_absorbs`] is where the answer goes when one does.
//!
//! Identical masks are ONE row of the matrix. Sixty layers of a transformer
//! state the same decode window sixty times, and the constraint is about the
//! set, not about how often it is stated.
//!
//! # When it fails
//!
//! Not a refusal. A consumer whose classes cannot be made consecutive
//! alongside everybody else's is a consumer that pays for a fallback — split,
//! grouped, or copy, bucket-dependent — and the rest of the plan still runs
//! copy-free. So the pass WITHDRAWS the offending constraint, restores the
//! tree it had, and writes the consumer's nodes into the
//! [`FallbackTable`](crate::FallbackTable). See [`menu`] for the cost model
//! and [`insertion_order`] for which constraint gets withdrawn.

mod pq;

use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::{Classes, Plan};

use crate::baked::{Fallback, FallbackRow, FallbackTable, LayoutOrder, Phase, Region};
use crate::budget::{Budgets, DeviceProfile};

pub use pq::{Leaf, PqTree};

/// The most classes this pass will seriate.
///
/// THE FIRE PATH'S CEILING, NOT THIS PASS'S. `class_order` hands the driver a
/// `Vec<u8>` per fire, so a class index is a byte everywhere downstream. A
/// plan with more behaviours than that is not refused HERE —
/// [`LayoutOrder::Identity`] is what it gets, and P4 is an optimization pass
/// with no business turning a load away — but nor can such a plan be fired:
/// the descriptor cannot name its classes either. **The refusal belongs to
/// P8**, beside the `DescriptorAbi` whose byte layout states the ceiling; this
/// pass declining is what keeps it from being stated in two places that could
/// disagree.
const MAX_CLASSES: usize = u8::MAX as usize + 1;

/// The row count at which splitting a GEMM stops losing to copying its rows
/// contiguous, and the device that number was measured on.
///
/// MEASURED, AND A PLACEHOLDER UNTIL IT IS MEASURED AGAIN. RTX 3090, fp16
/// tensor-op GEMM at `K=N=4096`, two-way split against copy-then-dense
/// (`tart/evidence/layout_planning.md`, `files/greenctx/results_layout.txt`):
///
/// | M (rows) | 2-way split | copy + dense |
/// |---:|---:|---:|
/// | 64 | **1.82x** | 1.07x |
/// | 128 | 1.17x | 1.03x |
/// | 256 | 1.19x | 1.03x |
/// | 512 | **0.88x** | 1.02x |
/// | 2048 | 0.94x | 0.94x |
///
/// At decode scale a split is launch- and tile-bound and loses badly; by 512
/// rows it has crossed over and wins. So the answer is
/// **batch-size-dependent**, which is why [`FallbackTable`] is keyed by bucket
/// range and not by node.
const CROSSOVER_ROWS: f32 = 512.0;

/// The SM count of the device [`CROSSOVER_ROWS`] was measured on.
///
/// WHY THE CROSSOVER IS SCALED BY IT. What makes a split lose at small `M` is
/// that neither half has enough tiles to fill the machine, so the crossover
/// sits where one GEMM's tiles start to saturate the SMs — which moves with
/// how many SMs there are. Scaling by the ratio is the cheapest model that
/// gets the direction right on a device nobody has re-measured, and it is
/// exactly the kind of number [`DeviceProfile`] exists to carry: the day a
/// shell measures its own crossover, this constant becomes a profile field and
/// nothing else here changes.
const CROSSOVER_SMS: f32 = 82.0;

/// P4. The class order, and the answers for the consumers it could not seat.
pub(crate) fn seriate(
    plan: &Plan,
    regions: &[Region],
    classes: &Classes,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> (LayoutOrder, FallbackTable) {
    let count = classes.classes.len();
    if count == 0 || count > MAX_CLASSES {
        return (LayoutOrder::Identity, FallbackTable::default());
    }

    // The matrix: one row per DISTINCT mask, remembering every region that
    // stated it, since the fallback is owed to all of them.
    let mut matrix: BTreeMap<Vec<Leaf>, Vec<usize>> = BTreeMap::new();
    for (r, region) in regions.iter().enumerate() {
        if !constrains(plan, region, count) {
            continue;
        }
        let mask: Vec<Leaf> = region.mask.iter().map(|c| c as Leaf).collect();
        matrix.entry(mask).or_default().push(r);
    }

    let mut tree = PqTree::universe(count);
    let mut withdrawn: Vec<Vec<Leaf>> = Vec::new();
    for mask in insertion_order(&matrix) {
        // `reduce` is atomic: a `false` leaves the tree exactly as it was, so
        // withdrawing the constraint is nothing more than not counting it.
        if !tree.reduce(&mask) {
            withdrawn.push(mask);
        }
    }

    let mut rows: Vec<FallbackRow> = Vec::new();
    for mask in &withdrawn {
        // `r` IS MEASURED ON THE ORDER THAT SHIPS, and the ROW is owed by the
        // one that does not: the tree makes this consumer no promise, so it
        // needs an answer whichever frontier the fire path ends up on. If the
        // canonical frontier happens to seat it anyway, `Split { r: 1 }` is
        // one kernel over pointer plus extent — the free case, spelled in the
        // menu's own vocabulary rather than by omitting the row and leaving
        // the day a stability pick chooses differently to fend for itself.
        let answer = menu(PqTree::runs(tree.frontier(), mask), budgets, profile);
        for &r in &matrix[mask] {
            for node in regions[r].nodes.clone() {
                rows.extend(answer.iter().map(|(buckets, fallback)| FallbackRow {
                    node,
                    buckets: buckets.clone(),
                    fallback: *fallback,
                }));
            }
        }
    }
    // Node order, because a reader of the table is looking a node up in it and
    // the withdrawal order is about the tree rather than about the plan.
    rows.sort_by_key(|row| (row.node, row.buckets.start));

    (LayoutOrder::Seriated(tree), FallbackTable { rows })
}

/// Is this region a row of the C1P matrix?
fn constrains(plan: &Plan, region: &Region, classes: usize) -> bool {
    region.phase == Phase::Capture
        && !region.mask.is_empty()
        && region.mask.len() < classes
        && !gather_absorbs(plan, region)
}

/// Decision #9's seat: does this region's work reach its rows through a
/// gather, so that the row order cannot help it and need not try?
///
/// **ALWAYS `false` TODAY, AND THAT IS A TRUE STATEMENT ABOUT THE IR.** The
/// ops the exclusion is about are the weight-varied ones — MoE's routed
/// experts, LoRA's adapter banks — and the corrections that ride with them,
/// and the IR has no seat for a runtime-indexed weight bank yet (design's open
/// items). Until it does, no `Operation` can answer this question differently,
/// and a hand-written list of op names here would be the rewrite's prepare-op
/// list all over again: a table somebody has to remember to add to.
///
/// The arguments are the ones the answer will need — the region names the
/// nodes, the plan names their ops — so that the day the IR grows the seat,
/// this is a function body and not a signature change rippling out to `P4`.
fn gather_absorbs(_plan: &Plan, _region: &Region) -> bool {
    false
}

/// The order constraints are offered to the tree in: MOST CLASSES FIRST, ties
/// broken by the mask itself so that a bake is a function of its inputs.
///
/// WHY DESCENDING, AND WHAT IT REALLY BUYS. The greedy is "insert until one
/// fails, withdraw that one", so the insertion order decides WHICH consumer
/// pays when the family is not C1P — the loser is always a late arrival. Two
/// arguments point the same way:
///
/// - **Cost.** A mask's class count is, near enough, the width of the window
///   the consumer runs over. The fallback is paid in that window's rows —
///   `2 x bytes` for a copy, `r` launches for a split — so withdrawing the
///   SMALLEST set is withdrawing the cheapest one to lose. That is the
///   argument that survives; it is about the objective rather than about the
///   algorithm.
/// - **Freedom.** Design §3's reading is that a bigger set constrains less.
///   It is true at the ends and a heuristic in between: one set of size `k`
///   over `n` classes leaves `(n-k+1)! · k!` orderings, which is `24` at both
///   `k = 1` and `k = n` — no constraint at all — and ties at `12` for `k = 2`
///   and `k = 3` over four classes. So the ordering by size is not monotone in
///   how much freedom it costs, and it is the cost argument above, not this
///   one, that decides the direction.
///
/// A LATER ITEM, DOCUMENTED HERE BECAUSE THIS IS WHERE IT LANDS. Tucker's
/// forbidden-submatrix characterisation (Tucker, JCTB 1972) turns an
/// infeasible instance into a CERTIFICATE — the exact set of mutually
/// conflicting consumers — so the planner can withdraw the one that is
/// cheapest across the whole conflict instead of the one that happened to
/// arrive last. That is a strictly better cut point and it needs the
/// certificate, which needs the null tree this pass deliberately never builds.
fn insertion_order(matrix: &BTreeMap<Vec<Leaf>, Vec<usize>>) -> Vec<Vec<Leaf>> {
    // `BTreeMap` yields the masks in ascending lexicographic order and the
    // sort below is stable, so equal-sized masks keep it.
    let mut masks: Vec<Vec<Leaf>> = matrix.keys().cloned().collect();
    masks.sort_by_key(|mask| std::cmp::Reverse(mask.len()));
    masks
}

/// What one withdrawn consumer does instead, per bucket range.
///
/// **SPLIT AT PREFILL SCALE, COPY AT DECODE SCALE** — see [`CROSSOVER_ROWS`]
/// for the measurement and [`CROSSOVER_SMS`] for how it is carried to a device
/// nobody re-measured. The ranges index [`Budgets::buckets`]; a deployment
/// that declared no bucket lattice has one implicit bucket at
/// [`Budgets::max_tokens`], and gets one row covering it.
///
/// THE OTHER TWO ITEMS ON DESIGN §3'S MENU ARE NOT CHOSEN, AND NOT BECAUSE
/// THEY ARE WORSE. [`Fallback::Grouped`] is the option tart expects to
/// dominate both of these, and it needs a kernel that takes a pointer/offset
/// list — a fact about the backend's kernel table, which this crate has no
/// dependency on and no business inventing. [`Fallback::View`] needs a
/// consumer that takes a stride or an index list for free, which is the same
/// fact. Both variants exist on the enum so that a shell with the answer has
/// somewhere to put it; choosing one here would be this crate claiming to know
/// what a kernel it has never seen supports.
fn menu(runs: u32, budgets: &Budgets, profile: &DeviceProfile) -> Vec<(Range<u32>, Fallback)> {
    let lattice: &[u32] = if budgets.buckets.is_empty() {
        std::slice::from_ref(&budgets.max_tokens)
    } else {
        &budgets.buckets
    };
    let crossover = CROSSOVER_ROWS * profile.sms as f32 / CROSSOVER_SMS;
    let cut = lattice
        .iter()
        .position(|&rows| rows as f32 >= crossover)
        .unwrap_or(lattice.len()) as u32;
    let end = lattice.len() as u32;

    let mut menu = Vec::with_capacity(2);
    if cut > 0 {
        menu.push((0..cut, Fallback::Copy));
    }
    if cut < end {
        menu.push((cut..end, Fallback::Split { r: runs }));
    }
    menu
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, fact};
    use crate::{Baked, LayoutOrder, compile};
    use model_ir::Cond;

    fn bake(b: &Build) -> Baked {
        compile(&b.plan, &Budgets::new(4, 16), &DeviceProfile::default()).expect("the fixtures bake")
    }

    fn frontier(baked: &Baked) -> Vec<Leaf> {
        baked
            .order
            .tree()
            .expect("P4 seriated it")
            .frontier()
            .to_vec()
    }

    /// `f0 XOR f1` — the guard whose class set is the diagonal, and the one
    /// that cannot be an interval alongside both axes.
    fn either_but_not_both() -> Cond {
        Cond::or(
            Cond::and(fact(0), Cond::not(fact(1))),
            Cond::and(Cond::not(fact(0)), fact(1)),
        )
    }

    #[test]
    fn the_two_facts_lexicographic_order_cannot_seat() {
        // The worked example, as a plan: two facts, one windowed consumer per
        // fact. Class i is fact word i, so the windows are {01, 11} = {1, 3}
        // and {10, 11} = {2, 3}. Ascending class order puts class 2 between
        // the qo_one classes; the Gray-coded order does not.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Cond::Always);
        b.append(y, fact(0));
        b.append(y, fact(1));
        b.out(y);

        let baked = bake(&b);
        assert_eq!(baked.classes.classes.len(), 4);
        assert_eq!(frontier(&baked), [0, 1, 3, 2]);
        assert!(
            baked.fallback.rows.is_empty(),
            "both windows are intervals of that order, so nobody pays",
        );

        // Every windowed consumer, read back off the region table.
        for region in &baked.regions {
            if !constrains(&b.plan, region, 4) {
                continue;
            }
            let mask: Vec<Leaf> = region.mask.iter().map(|c| c as Leaf).collect();
            assert!(
                PqTree::is_interval(&frontier(&baked), &mask),
                "{mask:?} is not an interval of the baked order",
            );
        }

        // The window the ascending order splits in two.
        let qo_one = &baked.regions[1].mask;
        assert_eq!(qo_one.iter().collect::<Vec<_>>(), [1, 3]);
        assert_eq!(LayoutOrder::Identity.class_order(qo_one, None), [1, 3]);
        assert!(!PqTree::is_interval(&[0, 1, 2, 3], &[1, 3]));
    }

    #[test]
    fn a_fire_gets_the_present_classes_in_the_baked_order() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Cond::Always);
        b.append(y, fact(0));
        b.append(y, fact(1));
        b.out(y);
        let baked = bake(&b);

        // A fire carrying every class gets the whole frontier; one carrying
        // the masked classes gets the sub-order, which is still an interval.
        let everything = &baked.regions[0].mask;
        assert_eq!(baked.order.class_order(everything, None), [0, 1, 3, 2]);
        let masked = &baked.regions[2].mask;
        assert_eq!(baked.order.class_order(masked, None), [3, 2]);

        // v1 ignores last fire's order, and says so by answering the same.
        assert_eq!(
            baked.order.class_order(everything, Some(&[2, 3, 1, 0])),
            baked.order.class_order(everything, None),
        );
    }

    #[test]
    fn the_one_consumer_that_cannot_be_seated_is_the_one_that_pays() {
        // {A,B}, {B,C}, {C,A} over the first three classes: pairwise
        // overlapping, no common interval order. The greedy seats the first
        // two and withdraws the third; the plan is NOT refused, and the two
        // that were seated keep their promise.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Cond::Always);
        b.append(y, Cond::not(fact(1))); // node 1: classes {0, 1}
        b.append(y, either_but_not_both()); // node 2: classes {1, 2}
        b.append(y, Cond::not(fact(0))); // node 3: classes {0, 2}
        b.out(y);

        let baked = bake(&b);
        assert_eq!(baked.classes.classes.len(), 4);
        assert_eq!(frontier(&baked), [1, 0, 2, 3]);

        // Exactly one consumer fell back, and it is the diagonal one.
        let nodes: Vec<u32> = baked.fallback.rows.iter().map(|row| row.node).collect();
        assert_eq!(nodes, [2]);
        assert!(PqTree::is_interval(&frontier(&baked), &[0, 1]));
        assert!(PqTree::is_interval(&frontier(&baked), &[0, 2]));
        assert!(!PqTree::is_interval(&frontier(&baked), &[1, 2]));

        // No bucket lattice means one implicit bucket at the token ceiling,
        // and 16 rows is decode scale, where a copy beats a split.
        assert_eq!(baked.fallback.rows[0].buckets, 0..1);
        assert_eq!(baked.fallback.rows[0].fallback, Fallback::Copy);
    }

    #[test]
    fn a_plan_no_window_splits_leaves_every_ordering_feasible() {
        // Nothing constrains: one class, and the tree is the free one.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Cond::Always);
        b.append(y, Cond::Always);
        b.out(y);

        let baked = bake(&b);
        assert_eq!(frontier(&baked), [0]);
        assert!(baked.fallback.rows.is_empty());
    }

    #[test]
    fn a_prepare_region_is_not_a_row_of_the_matrix() {
        // The prepare node's window is the same {qo_one} set the decode's is,
        // but it is host work outside the graph and reads no rows, so it is
        // not a consumer. What is left is one constraint, not two, and it is
        // stated by the capture region.
        let mut b = Build::new();
        let x = b.input(4);
        let q = b.op(x, 4, Cond::Always);
        let plan = b.prepare(fact(0));
        let o = b.decode(q, plan, fact(0));
        let merged = b.merge(&[(o, fact(0)), (q, Cond::not(fact(0)))], 4);
        b.out(merged);

        let baked = bake(&b);
        let rows: Vec<Vec<Leaf>> = baked
            .regions
            .iter()
            .filter(|region| constrains(&b.plan, region, baked.classes.classes.len()))
            .map(|region| region.mask.iter().map(|c| c as Leaf).collect())
            .collect();
        assert!(
            rows.iter().all(|mask| mask.len() == 1),
            "only the decode window constrains: {rows:?}",
        );
        assert!(baked.fallback.rows.is_empty());
    }

    #[test]
    fn more_classes_than_a_byte_names_is_not_seriated_and_not_refused() {
        // Nine facts, each with a window of its own: 512 behaviours, which is
        // past what `class_order`'s `u8` can spell. The plan still bakes, and
        // the answer is the identity ordering — correct for every plan, and
        // exactly what a windowed consumer got before P4 existed.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Cond::Always);
        for bit in 0..9 {
            b.append(y, fact(bit));
        }
        b.out(y);

        let baked = bake(&b);
        assert_eq!(baked.classes.classes.len(), 512);
        assert_eq!(baked.order, LayoutOrder::Identity);
        assert!(baked.order.tree().is_none());
        assert!(baked.fallback.rows.is_empty());
    }

    #[test]
    fn the_bake_is_a_function_of_the_plan() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Cond::Always);
        b.append(y, Cond::not(fact(1)));
        b.append(y, either_but_not_both());
        b.append(y, Cond::not(fact(0)));
        b.out(y);

        let once = bake(&b);
        let twice = bake(&b);
        assert_eq!(once.order, twice.order);
        assert_eq!(once.fallback, twice.fallback);
    }

    #[test]
    fn the_fallback_menu_crosses_over_where_the_measurement_did() {
        // The reference device: the crossover is the measured 512 rows, so the
        // buckets below it copy and the buckets at or above it split.
        let reference = DeviceProfile {
            sms: CROSSOVER_SMS as u32,
            ..DeviceProfile::default()
        };
        let budgets = Budgets {
            buckets: vec![1, 64, 512, 1024, 8192],
            ..Budgets::default()
        };
        assert_eq!(
            menu(3, &budgets, &reference),
            vec![(0..2, Fallback::Copy), (2..5, Fallback::Split { r: 3 })],
        );

        // A wider device needs more rows before a split has tiles to fill it.
        let wider = DeviceProfile::default();
        assert_eq!(
            menu(2, &budgets, &wider),
            vec![(0..3, Fallback::Copy), (3..5, Fallback::Split { r: 2 })],
        );

        // An all-prefill lattice never copies, and an all-decode one never
        // splits: the table is bucket-keyed because one answer would be wrong
        // at one end of it.
        let prefill = Budgets {
            buckets: vec![4096, 8192],
            ..Budgets::default()
        };
        assert_eq!(
            menu(2, &prefill, &wider),
            vec![(0..2, Fallback::Split { r: 2 })],
        );
        let decode = Budgets {
            buckets: vec![1, 2, 4],
            ..Budgets::default()
        };
        assert_eq!(menu(2, &decode, &wider), vec![(0..3, Fallback::Copy)]);
    }

    #[test]
    fn the_bigger_sets_are_offered_first() {
        let mut matrix: BTreeMap<Vec<Leaf>, Vec<usize>> = BTreeMap::new();
        for mask in [vec![2, 3], vec![0, 1, 2], vec![0, 1], vec![1, 2, 3]] {
            matrix.insert(mask, vec![0]);
        }
        assert_eq!(
            insertion_order(&matrix),
            vec![vec![0, 1, 2], vec![1, 2, 3], vec![0, 1], vec![2, 3]],
        );
    }
}
