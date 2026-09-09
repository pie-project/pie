use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::{ClassTable, Trace};

use crate::budget::DeviceProfile;
use crate::compiled::{ClassOrder, Fallback, FallbackRow, FallbackTable, Phase, Region};

use crate::pq::{Leaf, PqTree};

const MAX_SEARCH_MASKS: usize = 12;

const CROSSOVER_ROWS: f32 = 512.0;

const CROSSOVER_SMS: f32 = 82.0;

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

    let mut matrix: BTreeMap<Vec<Leaf>, Vec<usize>> = BTreeMap::new();
    for (r, region) in regions.iter().enumerate() {
        if !constrains(region, count) {
            continue;
        }
        let mask: Vec<Leaf> = region.mask.iter().map(|c| c as Leaf).collect();
        matrix.entry(mask).or_default().push(r);
    }

    let groupable: BTreeMap<&Vec<Leaf>, bool> = matrix
        .iter()
        .map(|(mask, stated_by)| {
            (
                mask,
                composed_of(trace, regions, stated_by, &profile.grouped),
            )
        })
        .collect();

    let (tree, withdrawn) = choose(trace, regions, &matrix, &groupable, count, profile);

    let mut rows: Vec<FallbackRow> = Vec::new();
    for mask in &withdrawn {
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
    rows.sort_by_key(|row| (row.node, row.buckets.start));

    (ClassOrder::Seriated(tree), FallbackTable { rows })
}

fn constrains(region: &Region, classes: usize) -> bool {
    region.phase == Phase::Capture && !region.mask.is_empty() && region.mask.len() < classes
}

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

fn withdrawal_cost(
    trace: &Trace,
    regions: &[Region],
    rows: &[usize],
    groupable: bool,
    profile: &DeviceProfile,
) -> f32 {
    let discount = if groupable { GROUPED_DISCOUNT } else { 1.0 };
    discount
        * rows
            .iter()
            .flat_map(|&r| regions[r].nodes.clone())
            .filter_map(|node| trace.nodes.get(node as usize))
            .map(|node| profile.family_us.of(&node.op))
            .sum::<f32>()
}

const GROUPED_DISCOUNT: f32 = 0.05;

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

fn concede(masks: &[&Vec<Leaf>], costs: &[f32], count: usize) -> (PqTree, Vec<Vec<Leaf>>) {
    let mut order: Vec<usize> = (0..masks.len()).collect();
    order.sort_by(|&a, &b| {
        costs[b]
            .partial_cmp(&costs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| masks[a].cmp(masks[b]))
    });

    let mut tree = PqTree::universe(count);
    let mut withdrawn = Vec::new();
    for i in order {
        if !tree.reduce(masks[i]) {
            withdrawn.push(masks[i].clone());
        }
    }
    (tree, withdrawn)
}

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
