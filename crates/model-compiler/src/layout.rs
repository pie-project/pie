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
//! is a region no class runs (`ClassTable::dead`); every class is a region that
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
//! and [`choose`] for which constraint gets withdrawn.
//!
//! # The menu grew an entry, and it grew a seat rather than an opinion
//!
//! `Fallback::Grouped` — one launch that walks a segment list instead of `r`
//! launches over `r` rectangles — was a typed seam nothing constructed,
//! because choosing it means knowing that some backend has such a kernel and
//! this crate has no dependency that could tell it. [`DeviceProfile::grouped`]
//! is that dependency, inverted: the caller names the ops, by
//! `Operands::name`, exactly as it names the workspace-exclusive ones. A mask
//! every stating region [`composed_of`] such ops gets `Grouped` from [`menu`],
//! at every bucket. An empty list — the default — is the pass as it was.
//!
//! WHAT IT DOES NOT CHANGE IS WHICH CONSUMER LOSES. That is
//! [`choose`], it is decided by COST — and a groupable consumer is nearly
//! free to lose, since `Grouped` is one launch where a split is `r`. So
//! naming an op in [`DeviceProfile::grouped`] is also what makes its window
//! the cheap one to withdraw: the score window keeps its interval, the
//! correction takes a segment list, and neither pays a split.

use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::{ClassTable, Trace};

use crate::compiled::{Fallback, FallbackRow, FallbackTable, ClassOrder, Phase, Region};
use crate::budget::{Budget, DeviceProfile};

use crate::pq::{Leaf, PqTree};

/// The most classes this pass will seriate.
///
/// THE FIRE PATH'S CEILING, NOT THIS PASS'S. `class_order` hands the driver a
/// `Vec<u8>` per fire, so a class index is a byte everywhere downstream. A
/// plan with more behaviours than that is not refused HERE —
/// [`ClassOrder::Identity`] is what it gets, and P4 is an optimization pass
/// with no business turning a load away — but nor can such a plan be fired:
/// the descriptor cannot name its classes either. **The refusal belongs to
/// P8**, beside the `DescriptorAbi` whose byte layout states the ceiling; this
/// pass declining is what keeps it from being stated in two places that could
/// disagree.
const MAX_CLASSES: usize = u8::MAX as usize + 1;

/// The most distinct masks this pass will search exhaustively over.
///
/// `2^k` tree builds, once per load. See [`choose`] for why `k` is small by
/// construction and stays that way: it counts the DISTINCT guard masks a
/// model text states, not the classes those guards name. The catalog's
/// widest text is 7 against 16 classes.
const MAX_SEARCH_MASKS: usize = 12;

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
    trace: &Trace,
    regions: &[Region],
    classes: &ClassTable,
    budget: &Budget,
    profile: &DeviceProfile,
) -> (ClassOrder, FallbackTable) {
    let count = classes.classes.len();
    if count == 0 || count > MAX_CLASSES {
        return (ClassOrder::Identity, FallbackTable::default());
    }

    // The matrix: one row per DISTINCT mask, remembering every region that
    // stated it, since the fallback is owed to all of them.
    let mut matrix: BTreeMap<Vec<Leaf>, Vec<usize>> = BTreeMap::new();
    for (r, region) in regions.iter().enumerate() {
        if !constrains(trace, region, count) {
            continue;
        }
        let mask: Vec<Leaf> = region.mask.iter().map(|c| c as Leaf).collect();
        matrix.entry(mask).or_default().push(r);
    }

    // Asked once here because the answer is wanted in two places and a second
    // walk of the plan is a second answer waiting to disagree: could a shell
    // serve this consumer GROUPED?
    let groupable: BTreeMap<&Vec<Leaf>, bool> = matrix
        .iter()
        .map(|(mask, stated_by)| (mask, composed_of(trace, regions, stated_by, &profile.grouped)))
        .collect();

    let (tree, withdrawn) = choose(trace, regions, &matrix, &groupable, count, profile);

    let mut rows: Vec<FallbackRow> = Vec::new();
    for mask in &withdrawn {
        // `r` IS MEASURED ON THE ORDER THAT SHIPS, and the ROW is owed by the
        // one that does not: the tree makes this consumer no promise, so it
        // needs an answer whichever frontier the fire path ends up on. If the
        // canonical frontier happens to seat it anyway, `Split { r: 1 }` is
        // one kernel over pointer plus extent — the free case, spelled in the
        // menu's own vocabulary rather than by omitting the row and leaving
        // the day a stability pick chooses differently to fend for itself.
        let answer = menu(
            PqTree::runs(tree.frontier(), mask),
            groupable[mask],
            budget,
            profile,
        );
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

    (ClassOrder::Seriated(tree), FallbackTable { rows })
}

/// Is this region a row of the C1P matrix?
fn constrains(trace: &Trace, region: &Region, classes: usize) -> bool {
    region.phase == Phase::Capture
        && !region.mask.is_empty()
        && region.mask.len() < classes
        && !gather_absorbs(trace, region)
}

/// Decision #9's seat: does this region's work reach its rows through a
/// gather, so that the row order cannot help it and need not try?
///
/// **STILL `false`, AND THE SEAT IS NO LONGER EMPTY — IT IS ANSWERED** (palo
/// C2). The IR grew the runtime-indexed bank the old note was waiting for
/// (`linear.lora_correct` over `ParamSource::Registered` planes, indexed by
/// `RuntimeInput::AdapterRoutes`), and with the op in hand the question can be
/// asked properly for the first time. The answer is no, for a reason worth
/// stating because decision #9 reads the other way at first glance.
///
/// **THE GATHER IS OVER WEIGHTS, AND THE CONSTRAINT IS ABOUT ROWS.** What
/// tart measured — gather → grouped → scatter at 3.4× ideal against 33× for a
/// split — is about which WEIGHT a row multiplies. A correction's rows are not
/// gathered at all: `x`, `y` and `routes` are read at `[row_offset, rows)` of
/// their columns, which is a slice, and a slice needs its rows contiguous. So
/// the region is a C1P row like any other windowed region, and excluding it
/// would not make it cheaper — it would make its window FRAGMENTED.
///
/// That is measured rather than argued. On qwen35-d0.8b's four classes
/// (`{}`, `{qo_one}`, `{adapter}`, `{qo_one, adapter}`), offering the
/// correction's `{2,3}` as a constraint beside the two attention windows gives
/// the exact seriation `0 2 3 1` — every one of the three consecutive, no
/// fallback row emitted. Withdrawing it leaves the frontier `[[0 2] [1 3]]`,
/// whose canonical order puts classes 2 and 3 at positions 1 and 3, and a fire
/// carrying an adapted prefill lane beside an adapted decode lane then covers
/// two row intervals rather than one — `Fallback::Split { r: 2 }`, which the
/// shells now serve as two launches (`driver::fire::fallback`). Offering the
/// constraint is still the better answer, because a split costs a launch and
/// a tile that the seated order does not; what it no longer costs is the fire.
/// (It did: `driver_cuda::Fault::Fragmented` used to refuse every such batch,
/// on the premise — false since C4 — that this table is always empty.)
/// `crates/model-compiler/tests/every_sku_carves_an_arena.rs` pins both halves.
///
/// So decision #9's exclusion belongs to the FALLBACK menu — what a withdrawn
/// consumer does instead, where `Fallback::Grouped` is the entry tart expects
/// to dominate — and not to the constraint matrix. What would make this
/// function answer `true` is an op that takes a row-index list rather than a
/// rectangle: an SGMV whose segments are gathered, which is a kernel-table
/// fact this crate still has no dependency on.
///
/// The arguments stay as they are — the region names the nodes, the plan names
/// their ops — so the day such an op lands, this is a function body.
fn gather_absorbs(_trace: &Trace, _region: &Region) -> bool {
    false
}

/// Is every region that stated this mask composed ENTIRELY of ops the caller
/// named?
///
/// **THE ANSWER IS THE CALLER'S AND THE QUESTION IS THIS PASS'S**
/// (decision #24). Two profile lists ask it. [`DeviceProfile::grouped`] asks
/// "could a shell serve this consumer as one launch over a segment list" —
/// `gather_absorbs` above settles that a correction's rows are a SLICE and so
/// belong in the constraint matrix, but says nothing about what a withdrawn
/// one does, and design §3's menu has an entry for exactly this: "gather the
/// rows into one contiguous block, run once", which for an op whose weight
/// side is already runtime-indexed degenerates into "hand the kernel the
/// intervals and let it walk them". Whether a backend has such a kernel is a
/// fact about a kernel table this crate has no dependency on, so it arrives by
/// op name, exactly as [`DeviceProfile::exclusive`] does.
///
/// **EVERY NODE, NOT ANY NODE.** A region is dispatched as a unit — P2
/// coalesced these nodes precisely because they share a mask, and the walk
/// loops over the region's whole node range once per launch — so one launch
/// for the region means one launch for every node in it. A region holding one
/// groupable op beside one that is not is a region that must still be split,
/// and answering `true` for it would be this pass promising a launch count the
/// walk cannot deliver and the second op cannot survive.
///
/// An empty list answers `false` for everything, which is the status quo and
/// the default for both.
fn composed_of(trace: &Trace, regions: &[Region], stated_by: &[usize], names: &[String]) -> bool {
    if names.is_empty() {
        return false;
    }
    stated_by.iter().all(|&r| {
        regions[r].nodes.clone().all(|node| {
            trace.nodes.get(node as usize).is_some_and(|node| {
                names
                    .iter()
                    .any(|named| named == model_ir::Operands::name(&node.op))
            })
        })
    })
}

/// The order constraints are offered to the tree in: MOST CLASSES FIRST, ties
/// broken by the mask itself so that a bake is a function of its inputs.
///
/// **P3 AND P6'S TABLE, READ A THIRD TIME.** `DeviceProfile::family_us` is
/// where a deployment that profiled its own kernels puts the answer, and the
/// conditional gate and the fork gate already ask it "how expensive is this
/// region". A cost model of this pass's own would be a second answer to the
/// same question, free to disagree with theirs.
///
/// WHY NODES AND NOT ROWS, AND WHY THAT IS ENOUGH TO RANK. What a withdrawal
/// actually costs is `node_cost x (fallback_ratio - 1)`, summed over the
/// buckets a deployment fires at and weighted by how often the composition
/// straddles at all — and neither factor is known here (`CROSSOVER_ROWS` is
/// one measured point, and the composition distribution is the engine's).
/// What IS known is that both factors are near enough common across the
/// candidates of one plan for the ranking to survive dropping them: on
/// qwen35-d0.8b the two feasible withdrawals fragment in 158 and 159 of the
/// 255 compositions, so the choice is decided by the nodes alone — 24 linear
/// nodes at 40 us against 6 attention nodes at 60 us, a factor of 2.65. The
/// day the engine reports its own distribution, THIS is the function that
/// grows the weight, and nothing above it moves.
///
/// **AND A GROUPABLE CONSUMER IS NEARLY FREE TO LOSE, WHICH IS WHY THE TWO
/// FEATURES COMPOSE.** `Fallback::Grouped` is one launch over a segment list
/// where a split is `r` — the kernel walks the intervals itself and no bytes
/// move — so withdrawing such a mask costs the fire almost nothing and this
/// returns almost nothing for it. That is what makes the search pick it: on
/// qwen3.5 the adapter window is 24 linear nodes against the score window's 6
/// attention ones and would never lose on cost alone, but a shell that names
/// `linear.lora_correct` makes it the cheap one, the score window keeps its
/// interval, and NEITHER pays a split. The factor is a placeholder — one
/// launch against `r` is not literally free — held above zero so that a
/// groupable mask still loses to withdrawing nothing at all.
fn withdrawal_cost(
    trace: &Trace,
    regions: &[Region],
    rows: &[usize],
    groupable: bool,
    profile: &DeviceProfile,
) -> f32 {
    let discount = if groupable { GROUPED_DISCOUNT } else { 1.0 };
    discount * rows.iter()
        .flat_map(|&r| regions[r].nodes.clone())
        .filter_map(|node| trace.nodes.get(node as usize))
        .map(|node| profile.family_us.of(&node.op))
        .sum::<f32>()
}

/// What a groupable consumer's withdrawal costs, as a fraction of a split's.
///
/// A PLACEHOLDER WITH A DIRECTION, not a measurement. What a grouped launch
/// actually costs over a seated one is the segment loop's own overhead, and
/// nobody has measured it for any op; what is known is that it is one launch
/// where the alternative is `r`, and that the PoC measured 681 recorded nodes
/// against 825 on a four-interval qwen3.5 fire. Small enough that a groupable
/// mask loses to any other candidate, large enough that withdrawing nothing
/// still wins.
const GROUPED_DISCOUNT: f32 = 0.05;

/// The tree that ships, and the masks it makes no promise to.
///
/// **EXACT, NOT GREEDY, AND THE INSTANCE IS WHY.** Choosing a minimum-weight
/// set of rows to delete so that a 0/1 matrix has the consecutive-ones
/// property is NP-hard in general. It is not hard HERE, because `k` — the
/// number of DISTINCT guard masks — is small by construction: a model text
/// states one or two per split site, so `k` grows with the facts a text
/// splits on, roughly `2F`, and NOT with the `2^F` classes those facts name.
/// Today's catalog runs `k` of 2, 4, 6 and 7 against class counts of 2, 3, 8
/// and 16. At `k = 7` the whole power set is 128 tree builds, once per load,
/// and the answer is the optimum rather than an approximation of it.
///
/// WHAT THE OLD ORDER GOT WRONG, AND IT WAS NOT THE ALGORITHM. The greedy
/// this replaces inserted masks by descending class count and withdrew
/// whichever failed — and on qwen35-d0.8b all four size-four masks tie, so
/// the loser was decided by `BTreeMap`'s lexicographic order on CLASS
/// INDICES, which is to say by the bit numbers the model text happened to
/// give its facts. `captures_scores` paid because it is `fact(3)` and
/// `has_adapter` is `fact(1)`. Swapping those two declarations would have
/// moved a 2.65x cost with nothing in the tree to say so, and `Predicate`'s
/// own note — "a bit is a POSITION AND NOTHING ELSE" — would have been false
/// while reading as true. A cost never reads a class index, so the property
/// holds by construction here rather than by care.
///
/// Ties keep the lowest candidate word, so a bake is still a function of its
/// inputs (`tests/every_sku_seriates_its_classes.rs` pins that).
fn choose(
    trace: &Trace,
    regions: &[Region],
    matrix: &BTreeMap<Vec<Leaf>, Vec<usize>>,
    groupable: &BTreeMap<&Vec<Leaf>, bool>,
    count: usize,
    profile: &DeviceProfile,
) -> (PqTree, Vec<Vec<Leaf>>) {
    let masks: Vec<&Vec<Leaf>> = matrix.keys().collect();
    let costs: Vec<f32> = masks
        .iter()
        .map(|mask| withdrawal_cost(trace, regions, &matrix[*mask], groupable[*mask], profile))
        .collect();

    if masks.len() > MAX_SEARCH_MASKS {
        return concede(&masks, &costs, count);
    }

    // `drop` is a bitmask over `masks`, ascending, so `0` — withdraw nothing —
    // is tried first and wins outright when the family is C1P. A strict `<`
    // keeps the earliest candidate on a tie, and a superset of a feasible set
    // is feasible at a higher cost, so nothing needs to be said about
    // minimality: the cheapest feasible set is minimal or it would not be the
    // cheapest.
    let mut best: Option<(f32, u32, PqTree)> = None;
    for drop in 0u32..(1 << masks.len()) {
        let cost: f32 = (0..masks.len())
            .filter(|i| drop >> i & 1 == 1)
            .map(|i| costs[i])
            .sum();
        if best.as_ref().is_some_and(|(held, _, _)| cost >= *held) {
            continue;
        }
        let mut tree = PqTree::universe(count);
        if !(0..masks.len())
            .filter(|i| drop >> i & 1 == 0)
            .all(|i| tree.reduce(masks[i]))
        {
            continue;
        }
        best = Some((cost, drop, tree));
    }

    let (_, drop, tree) = best.expect("withdrawing every mask leaves a universe tree");
    let withdrawn = (0..masks.len())
        .filter(|i| drop >> i & 1 == 1)
        .map(|i| masks[i].clone())
        .collect();
    (tree, withdrawn)
}

/// The search's fallback past [`MAX_SEARCH_MASKS`]: offer the constraints
/// MOST EXPENSIVE FIRST and withdraw whatever will not go in.
///
/// A LATER ITEM, DOCUMENTED HERE BECAUSE THIS IS WHERE IT LANDS. Tucker's
/// forbidden-submatrix characterisation (Tucker, JCTB 1972) turns an
/// infeasible instance into a CERTIFICATE — the exact set of mutually
/// conflicting consumers — which is what the search below stops needing at
/// this scale and would want again past [`MAX_SEARCH_MASKS`], where the
/// exhaustive enumeration gives way to this greedy.
///
/// The same greedy the exhaustive search degenerates to, ordered by the same
/// number it optimises, so the two agree wherever both are affordable and the
/// cheap consumer is the one that loses either way. No catalog text reaches
/// this path today; it is here because a `2^k` with no ceiling is a load-time
/// hang waiting for a text nobody has written yet.
fn concede(masks: &[&Vec<Leaf>], costs: &[f32], count: usize) -> (PqTree, Vec<Vec<Leaf>>) {
    let mut order: Vec<usize> = (0..masks.len()).collect();
    // Descending cost; the mask itself breaks ties, so a bake is a function of
    // its inputs here too.
    order.sort_by(|&a, &b| {
        costs[b]
            .partial_cmp(&costs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| masks[a].cmp(masks[b]))
    });

    let mut tree = PqTree::universe(count);
    let mut withdrawn = Vec::new();
    for i in order {
        // `reduce` is atomic: a `false` leaves the tree exactly as it was, so
        // withdrawing the constraint is nothing more than not counting it.
        if !tree.reduce(masks[i]) {
            withdrawn.push(masks[i].clone());
        }
    }
    (tree, withdrawn)
}

/// What one withdrawn consumer does instead, per bucket range.
///
/// **SPLIT AT PREFILL SCALE, COPY AT DECODE SCALE** — see [`CROSSOVER_ROWS`]
/// for the measurement and [`CROSSOVER_SMS`] for how it is carried to a device
/// nobody re-measured. The ranges index [`Budget::buckets`]; a deployment
/// that declared no bucket lattice has one implicit bucket at
/// [`Budget::max_tokens`], and gets one row covering it.
///
/// **[`Fallback::Grouped`] IS CHOSEN WHEN, AND ONLY WHEN, THE CALLER SAID ITS
/// KERNEL EXISTS** ([`groupable`], [`DeviceProfile::grouped`]). The standing
/// note here was that the entry "needs a kernel that takes a pointer/offset
/// list — a fact about the backend's kernel table, which this crate has no
/// dependency on and no business inventing", and that is still true; what
/// changed is that the fact now has a seat to arrive in, so declining is no
/// longer the only honest answer available.
///
/// It is written at EVERY bucket, and that is a claim rather than a shrug. The
/// split/copy cut above exists because a split's cost is `r` launches and `r`
/// partial tiles, which loses at decode scale and wins at prefill scale, while
/// a copy's cost is `2 x bytes` at every scale — two curves, so a crossover. A
/// grouped launch pays neither: one launch, no copy, and the segments are the
/// same rows the split would have run over. There is no scale at which it
/// loses to a thing that does strictly more work, so there is no cut point to
/// state, and stating one anyway would be inventing a second number nobody
/// measured.
///
/// [`Fallback::View`] is still not chosen, for the reason `Grouped` was not:
/// it needs a consumer that takes a stride or an index list for free, and no
/// profile field says which those are.
fn menu(
    runs: u32,
    groupable: bool,
    budget: &Budget,
    profile: &DeviceProfile,
) -> Vec<(Range<u32>, Fallback)> {
    let lattice: &[u32] = if budget.buckets.is_empty() {
        std::slice::from_ref(&budget.max_tokens)
    } else {
        &budget.buckets
    };
    if groupable {
        return vec![(0..lattice.len() as u32, Fallback::Grouped)];
    }
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
    use crate::{CompiledModel, ClassOrder, compile};
    use model_ir::Guard;

    fn bake(b: &Build) -> CompiledModel {
        compile(&b.trace, &Budget::new(4, 16), &DeviceProfile::default()).expect("the fixtures bake")
    }

    fn frontier(compiled: &CompiledModel) -> Vec<Leaf> {
        compiled
            .order
            .tree()
            .expect("P4 seriated it")
            .frontier()
            .to_vec()
    }

    /// `f0 XOR f1` — the guard whose class set is the diagonal, and the one
    /// that cannot be an interval alongside both axes.
    fn either_but_not_both() -> Guard {
        Guard::or(
            Guard::and(fact(0), Guard::not(fact(1))),
            Guard::and(Guard::not(fact(0)), fact(1)),
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
        let y = b.op(x, 8, Guard::Always);
        b.append(y, fact(0));
        b.append(y, fact(1));
        b.out(y);

        let compiled = bake(&b);
        assert_eq!(compiled.classes.classes.len(), 4);
        assert_eq!(frontier(&compiled), [0, 1, 3, 2]);
        assert!(
            compiled.fallback.rows.is_empty(),
            "both windows are intervals of that order, so nobody pays",
        );

        // Every windowed consumer, read back off the region table.
        for region in &compiled.regions {
            if !constrains(&b.trace, region, 4) {
                continue;
            }
            let mask: Vec<Leaf> = region.mask.iter().map(|c| c as Leaf).collect();
            assert!(
                PqTree::is_interval(&frontier(&compiled), &mask),
                "{mask:?} is not an interval of the baked order",
            );
        }

        // The window the ascending order splits in two.
        let qo_one = &compiled.regions[1].mask;
        assert_eq!(qo_one.iter().collect::<Vec<_>>(), [1, 3]);
        assert_eq!(ClassOrder::Identity.class_order(qo_one, None), [1, 3]);
        assert!(!PqTree::is_interval(&[0, 1, 2, 3], &[1, 3]));
    }

    #[test]
    fn a_fire_gets_the_present_classes_in_the_baked_order() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        b.append(y, fact(0));
        b.append(y, fact(1));
        b.out(y);
        let compiled = bake(&b);

        // A fire carrying every class gets the whole frontier; one carrying
        // the masked classes gets the sub-order, which is still an interval.
        let everything = &compiled.regions[0].mask;
        assert_eq!(compiled.order.class_order(everything, None), [0, 1, 3, 2]);
        let masked = &compiled.regions[2].mask;
        assert_eq!(compiled.order.class_order(masked, None), [3, 2]);

        // v1 ignores last fire's order, and says so by answering the same.
        assert_eq!(
            compiled.order.class_order(everything, Some(&[2, 3, 1, 0])),
            compiled.order.class_order(everything, None),
        );
    }

    #[test]
    fn the_one_consumer_that_cannot_be_seated_is_the_one_that_pays() {
        // {A,B}, {B,C}, {C,A} over the first three classes: pairwise
        // overlapping, no common interval order, so exactly one of the three
        // must pay. The plan is NOT refused, and the two that were seated keep
        // their promise. The diagonal is stated ONCE and the axes twice, so it
        // is also the cheapest to lose and `choose` picks it for that reason
        // rather than for the order it arrived in.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        b.append(y, Guard::not(fact(1))); // node 1: classes {0, 1}
        b.append(y, Guard::not(fact(1))); // node 2: the same window, again
        b.append(y, either_but_not_both()); // node 3: classes {1, 2}
        b.append(y, Guard::not(fact(0))); // node 4: classes {0, 2}
        b.append(y, Guard::not(fact(0))); // node 5: the same window, again
        b.out(y);

        let compiled = bake(&b);
        assert_eq!(compiled.classes.classes.len(), 4);
        assert_eq!(frontier(&compiled), [1, 0, 2, 3]);

        // Exactly one consumer fell back, and it is the diagonal one.
        let nodes: Vec<u32> = compiled.fallback.rows.iter().map(|row| row.node).collect();
        assert_eq!(nodes, [3]);
        assert!(PqTree::is_interval(&frontier(&compiled), &[0, 1]));
        assert!(PqTree::is_interval(&frontier(&compiled), &[0, 2]));
        assert!(!PqTree::is_interval(&frontier(&compiled), &[1, 2]));

        // No bucket lattice means one implicit bucket at the token ceiling,
        // and 16 rows is decode scale, where a copy beats a split.
        assert_eq!(compiled.fallback.rows[0].buckets, 0..1);
        assert_eq!(compiled.fallback.rows[0].fallback, Fallback::Copy);
    }

    #[test]
    fn a_plan_no_window_splits_leaves_every_ordering_feasible() {
        // Nothing constrains: one class, and the tree is the free one.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        b.append(y, Guard::Always);
        b.out(y);

        let compiled = bake(&b);
        assert_eq!(frontier(&compiled), [0]);
        assert!(compiled.fallback.rows.is_empty());
    }

    #[test]
    fn a_prepare_region_is_not_a_row_of_the_matrix() {
        // The prepare node's window is the same {qo_one} set the decode's is,
        // but it is host work outside the graph and reads no rows, so it is
        // not a consumer. What is left is one constraint, not two, and it is
        // stated by the capture region.
        let mut b = Build::new();
        let x = b.input(4);
        let q = b.op(x, 4, Guard::Always);
        let plan = b.prepare(fact(0));
        let o = b.decode(q, plan, fact(0));
        let merged = b.merge(&[(o, fact(0)), (q, Guard::not(fact(0)))], 4);
        b.out(merged);

        let compiled = bake(&b);
        let rows: Vec<Vec<Leaf>> = compiled
            .regions
            .iter()
            .filter(|region| constrains(&b.trace, region, compiled.classes.classes.len()))
            .map(|region| region.mask.iter().map(|c| c as Leaf).collect())
            .collect();
        assert!(
            rows.iter().all(|mask| mask.len() == 1),
            "only the decode window constrains: {rows:?}",
        );
        assert!(compiled.fallback.rows.is_empty());
    }

    #[test]
    fn more_classes_than_a_byte_names_is_not_seriated_and_not_refused() {
        // Nine facts, each with a window of its own: 512 behaviours, which is
        // past what `class_order`'s `u8` can spell. The plan still bakes, and
        // the answer is the identity ordering — correct for every plan, and
        // exactly what a windowed consumer got before P4 existed.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        for bit in 0..9 {
            b.append(y, fact(bit));
        }
        b.out(y);

        let compiled = bake(&b);
        assert_eq!(compiled.classes.classes.len(), 512);
        assert_eq!(compiled.order, ClassOrder::Identity);
        assert!(compiled.order.tree().is_none());
        assert!(compiled.fallback.rows.is_empty());
    }

    #[test]
    fn the_bake_is_a_function_of_the_plan() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        b.append(y, Guard::not(fact(1)));
        b.append(y, either_but_not_both());
        b.append(y, Guard::not(fact(0)));
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
        let budget = Budget {
            buckets: vec![1, 64, 512, 1024, 8192],
            ..Budget::default()
        };
        assert_eq!(
            menu(3, false, &budget, &reference),
            vec![(0..2, Fallback::Copy), (2..5, Fallback::Split { r: 3 })],
        );

        // A wider device needs more rows before a split has tiles to fill it.
        let wider = DeviceProfile::default();
        assert_eq!(
            menu(2, false, &budget, &wider),
            vec![(0..3, Fallback::Copy), (3..5, Fallback::Split { r: 2 })],
        );

        // An all-prefill lattice never copies, and an all-decode one never
        // splits: the table is bucket-keyed because one answer would be wrong
        // at one end of it.
        let prefill = Budget {
            buckets: vec![4096, 8192],
            ..Budget::default()
        };
        assert_eq!(
            menu(2, false, &prefill, &wider),
            vec![(0..2, Fallback::Split { r: 2 })],
        );
        let decode = Budget {
            buckets: vec![1, 2, 4],
            ..Budget::default()
        };
        assert_eq!(menu(2, false, &decode, &wider), vec![(0..3, Fallback::Copy)]);
    }

    #[test]
    fn the_cheapest_conflicting_consumer_pays_whatever_its_class_numbers() {
        // THE REGRESSION THIS PASS EXISTS FOR. The same three-cycle, with the
        // cheap consumer moved onto the LEXICOGRAPHICALLY FIRST mask: `{0,1}`
        // is stated once and the other two twice. The order this replaced
        // offered masks by descending class count and broke the tie — all
        // three are size two — on the class indices themselves, so it seated
        // `{0,1}` and `{0,2}` and withdrew `{1,2}`, paying 120 to save 60
        // because of how the model text numbered its facts. A cost never
        // reads a class index, so the answer here is the other one.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        b.append(y, Guard::not(fact(1))); // node 1: classes {0, 1}, stated once
        b.append(y, either_but_not_both()); // node 2: classes {1, 2}
        b.append(y, either_but_not_both()); // node 3: the same window, again
        b.append(y, Guard::not(fact(0))); // node 4: classes {0, 2}
        b.append(y, Guard::not(fact(0))); // node 5: the same window, again
        b.out(y);

        let compiled = bake(&b);
        assert_eq!(compiled.classes.classes.len(), 4);

        let nodes: Vec<u32> = compiled.fallback.rows.iter().map(|row| row.node).collect();
        assert_eq!(nodes, [1], "the one node that is cheaper to lose than two");
        assert!(PqTree::is_interval(&frontier(&compiled), &[1, 2]));
        assert!(PqTree::is_interval(&frontier(&compiled), &[0, 2]));
        assert!(!PqTree::is_interval(&frontier(&compiled), &[0, 1]));
    }

    #[test]
    fn the_search_and_the_greedy_agree_where_both_are_affordable() {
        // `concede` is the path past `MAX_SEARCH_MASKS`, and nothing in the
        // catalog reaches it — so the only thing that can keep it honest is
        // running it beside the search on an instance small enough for both.
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        b.append(y, Guard::not(fact(1)));
        b.append(y, either_but_not_both());
        b.append(y, either_but_not_both());
        b.append(y, Guard::not(fact(0)));
        b.append(y, Guard::not(fact(0)));
        b.out(y);

        let compiled = bake(&b);
        let profile = DeviceProfile::default();
        let count = compiled.classes.classes.len();
        let mut matrix: BTreeMap<Vec<Leaf>, Vec<usize>> = BTreeMap::new();
        for (r, region) in compiled.regions.iter().enumerate() {
            if constrains(&b.trace, region, count) {
                matrix
                    .entry(region.mask.iter().map(|c| c as Leaf).collect())
                    .or_default()
                    .push(r);
            }
        }
        let masks: Vec<&Vec<Leaf>> = matrix.keys().collect();
        let costs: Vec<f32> = masks
            .iter()
            .map(|mask| withdrawal_cost(&b.trace, &compiled.regions, &matrix[*mask], false, &profile))
            .collect();

        let groupable: BTreeMap<&Vec<Leaf>, bool> =
            matrix.keys().map(|mask| (mask, false)).collect();
        let (_, greedy) = concede(&masks, &costs, count);
        let (_, searched) = choose(&b.trace, &compiled.regions, &matrix, &groupable, count, &profile);
        assert_eq!(greedy, searched, "costs {costs:?} over masks {masks:?}");
        assert_eq!(searched.len(), 1, "the family is a three-cycle; one pays");
    }


    /// **A GROUPABLE CONSUMER IS ANSWERED AT EVERY BUCKET, AND THE COPY/SPLIT
    /// CROSSOVER IS NOT CONSULTED.** One launch over a segment list does
    /// strictly less work than `r` launches over `r` rectangles at every
    /// scale, so there is no cut point — and inventing one would be inventing
    /// a number nobody measured.
    #[test]
    fn a_groupable_consumer_is_answered_grouped_at_every_bucket() {
        let budget = Budget {
            buckets: vec![1, 64, 512, 1024, 8192],
            ..Budget::default()
        };
        let profile = DeviceProfile::default();
        assert_eq!(
            menu(3, true, &budget, &profile),
            vec![(0..5, Fallback::Grouped)],
        );
        // The same lattice, the same runs, the same device — and without the
        // word, the two entries the menu has always written.
        assert_eq!(
            menu(3, false, &budget, &profile),
            vec![(0..3, Fallback::Copy), (3..5, Fallback::Split { r: 3 })],
        );
    }

    /// **A REGION IS GROUPED AS A UNIT OR NOT AT ALL.** The walk dispatches a
    /// region's whole node range once per launch, so one launch for the region
    /// is one launch for every node in it — and a region holding one groupable
    /// op beside one that is not must still be split.
    #[test]
    fn one_ungroupable_node_disqualifies_the_region_that_holds_it() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        let guarded = b.op(y, 8, fact(0));
        b.out(guarded);

        assert_eq!(b.trace.nodes.len(), 2, "one shared op and one guarded one");
        // The op the fixture's `Build::op` emits, and a name no plan states.
        let named = model_ir::Operands::name(&b.trace.nodes[0].op).to_string();
        let region = Region {
            nodes: 0..2,
            mask: model_ir::ClassSet::of([0]),
            phase: Phase::Capture,
            lowering: crate::Lowering::AlwaysLaunch,
            stream: 0,
            wait: Vec::new(),
            open: None,
            close: None,
            sm_hint: None,
            collective: false,
        };
        let regions = [region];

        assert!(
            composed_of(&b.trace, &regions, &[0], std::slice::from_ref(&named)),
            "both nodes are the same op and it is named",
        );
        assert!(!composed_of(
            &b.trace,
            &regions,
            &[0],
            &["linear.matmul".to_string()]
        ));
        assert!(
            !composed_of(&b.trace, &regions, &[0], &[]),
            "the empty list is the status quo and groups nothing",
        );
    }
}
