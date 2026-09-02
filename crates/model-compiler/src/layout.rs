//! Chooses one global row order, via the Consecutive-Ones Property
//! (PQ-trees), so that windowed structural consumers read a contiguous
//! block. A consumer whose classes cannot be seated is not refused: it is
//! withdrawn from the tree and given a fallback (split, grouped, or copy —
//! see [`menu`]/[`choose`]) via the [`FallbackTable`](crate::FallbackTable).

use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::{ClassTable, Trace};

use crate::compiled::{Fallback, FallbackRow, FallbackTable, ClassOrder, Phase, Region};
use crate::budget::DeviceProfile;

use crate::pq::{Leaf, PqTree};

/// Ceiling on distinct guard masks searched exhaustively (`2^k` tree builds,
/// once per load). Counts distinct masks, not classes; the catalog's widest
/// text is 7 masks against 16 classes.
const MAX_SEARCH_MASKS: usize = 12;

/// Row count where splitting a GEMM stops losing to copy-then-dense
/// (measured on an RTX 3090, fp16, K=N=4096). Batch-size-dependent, hence why
/// [`FallbackTable`] is keyed by bucket range and not by node.
const CROSSOVER_ROWS: f32 = 512.0;

/// SM count of the device [`CROSSOVER_ROWS`] was measured on; the crossover
/// scales by this device's SM count over that one, since what decides it is
/// whether a split's tiles saturate the machine.
const CROSSOVER_SMS: f32 = 82.0;

/// The `layout` pass. The class order, and the answers for the consumers it
/// could not seat.
pub(crate) fn seriate(
    trace: &Trace,
    regions: &[Region],
    classes: &ClassTable,
    lattice: &[u32],
    ceiling: u32,
    profile: &DeviceProfile,
) -> (ClassOrder, FallbackTable) {
    let count = classes.classes.len();
    if count == 0 {
        return (ClassOrder::Identity, FallbackTable::default());
    }

    // One row per distinct mask; each mask remembers every region that
    // stated it, since a fallback is owed to all of them.
    let mut matrix: BTreeMap<Vec<Leaf>, Vec<usize>> = BTreeMap::new();
    for (r, region) in regions.iter().enumerate() {
        if !constrains(region, count) {
            continue;
        }
        let mask: Vec<Leaf> = region.mask.iter().map(|c| c as Leaf).collect();
        matrix.entry(mask).or_default().push(r);
    }

    // Computed once: needed both for scoring and for the fallback menu.
    let groupable: BTreeMap<&Vec<Leaf>, bool> = matrix
        .iter()
        .map(|(mask, stated_by)| (mask, composed_of(trace, regions, stated_by, &profile.grouped)))
        .collect();

    let (tree, withdrawn) = choose(trace, regions, &matrix, &groupable, count, profile);

    let mut rows: Vec<FallbackRow> = Vec::new();
    for mask in &withdrawn {
        // The tree makes this consumer no promise, so it needs an answer
        // regardless of frontier; `Split { r: 1 }` is the free case if the
        // shipped order happens to seat it anyway.
        let answer = menu(
            PqTree::runs(tree.frontier(), mask),
            groupable[mask],
            lattice,
            ceiling,
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
    // Sort by node: a reader looks a node up, not the withdrawal order.
    rows.sort_by_key(|row| (row.node, row.buckets.start));

    (ClassOrder::Seriated(tree), FallbackTable { rows })
}

/// Is this region a row of the C1P matrix?
fn constrains(region: &Region, classes: usize) -> bool {
    region.phase == Phase::Capture && !region.mask.is_empty() && region.mask.len() < classes
}

/// Is every region that stated this mask composed entirely of ops the
/// caller named (`DeviceProfile::grouped`)? Must hold for every node in the
/// region (dispatched as a unit); a mixed region is still split.
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

/// Cost of withdrawing this mask's constraint: summed per-node costs from
/// `DeviceProfile::family_us`, discounted by [`GROUPED_DISCOUNT`] when the
/// mask is groupable (nearly free to withdraw).
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

/// Fraction of a split's cost a groupable withdrawal costs. A placeholder,
/// not a measurement: small enough that a groupable mask wins, large enough
/// that withdrawing nothing still wins.
const GROUPED_DISCOUNT: f32 = 0.05;

/// The tree that ships, and the masks it makes no promise to. Exact search
/// (minimum-weight row deletion is NP-hard in general, but `k` distinct
/// guard masks is small, so `2^k` is affordable). Ties keep the lowest
/// candidate word, so a bake never depends on class-index numbering.
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

    // `drop` is a bitmask over `masks`; 0 (withdraw nothing) is tried first.
    // A superset of a feasible set is feasible at higher cost, so the
    // cheapest feasible set found is automatically minimal.
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

/// The search's fallback past [`MAX_SEARCH_MASKS`]: offer constraints
/// most-expensive-first and withdraw whatever doesn't fit. Exists because an
/// unbounded `2^k` search is a load-time hang waiting to happen.
fn concede(masks: &[&Vec<Leaf>], costs: &[f32], count: usize) -> (PqTree, Vec<Vec<Leaf>>) {
    let mut order: Vec<usize> = (0..masks.len()).collect();
    // Descending cost; the mask itself breaks ties, so this stays deterministic.
    order.sort_by(|&a, &b| {
        costs[b]
            .partial_cmp(&costs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| masks[a].cmp(masks[b]))
    });

    let mut tree = PqTree::universe(count);
    let mut withdrawn = Vec::new();
    for i in order {
        // `reduce` is atomic: `false` leaves the tree unchanged, so
        // withdrawing is simply not counting the constraint.
        if !tree.reduce(masks[i]) {
            withdrawn.push(masks[i].clone());
        }
    }
    (tree, withdrawn)
}

/// What one withdrawn consumer does instead, per bucket range. Splits at
/// prefill scale, copies at decode scale (see [`CROSSOVER_ROWS`],
/// [`CROSSOVER_SMS`]); [`Fallback::Grouped`] is used at every bucket when
/// the caller named the op, since it has no crossover to compute.
/// [`Fallback::View`] is never chosen: no profile field says which
/// consumers take a stride or index list for free.
fn menu(
    runs: u32,
    groupable: bool,
    lattice: &[u32],
    ceiling: u32,
    profile: &DeviceProfile,
) -> Vec<(Range<u32>, Fallback)> {
    let lattice: &[u32] = if lattice.is_empty() {
        std::slice::from_ref(&ceiling)
    } else {
        lattice
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
    use crate::{Budget, compile};
    use model_ir::Guard;

    #[test]
    fn more_classes_than_a_byte_names_is_refused() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        for bit in 0..9 {
            b.append(y, fact(bit));
        }
        b.out(y);
        let refused = compile(&b.trace, &Budget::new(4, 16), &DeviceProfile::default());
        assert!(
            matches!(refused, Err(crate::Error::TooManyClasses { classes: 512 })),
            "{refused:?}"
        );
    }

}
