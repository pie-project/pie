//! Chooses each region's lowering (`If`, `Switch`, or always-launch): a
//! region is conditionalized only if its window can be empty, its body is
//! fat enough to amortize the evaluation point, and skipping it profits over
//! that point's fixed cost.

use model_ir::{ClassTable, Def, Trace, ValueId};

use crate::compiled::{Lowering, Region};
use crate::budget::{Budget, DeviceProfile};

/// What one region's nodes cost, per the profile's family table; used by
/// both the fork and conditional cost gates so they agree.
pub(crate) fn region_us(trace: &Trace, region: &Region, profile: &DeviceProfile) -> f32 {
    region
        .nodes
        .clone()
        .filter_map(|node| trace.nodes.get(node as usize))
        .map(|node| profile.family_us.of(&node.op))
        .sum()
}

/// How many nodes a region holds — the count the profit gate is about, since
/// always-launch pays one empty launch per node and not per region.
fn nodes(region: &Region) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        region.nodes.len() as f32
    }
}

/// Choose each region's lowering, in place. `regions` comes out stamped with
/// [`Region::lowering`]; everything else is untouched.
///
/// `fat_region_us` at [`f32::INFINITY`] turns this off: no body is ever fat
/// enough, so every region stays [`Lowering::AlwaysLaunch`].
pub(crate) fn lower(
    trace: &Trace,
    regions: &mut [Region],
    classes: &ClassTable,
    budget: &Budget,
    profile: &DeviceProfile,
) {
    let all = classes.classes.len();
    if all == 0 || regions.len() < 2 {
        return;
    }

    // SWITCH first: a group claims its members before they're offered an
    // `IF` of their own.
    let mut claimed = vec![false; regions.len()];
    for group in switch_groups(trace, regions, classes, budget, profile) {
        for (arm, &at) in group.members.iter().enumerate() {
            claimed[at] = true;
            regions[at].lowering = Lowering::Switch {
                merge: group.merge,
                #[allow(clippy::cast_possible_truncation)]
                arm: arm as u8,
                #[allow(clippy::cast_possible_truncation)]
                arms: group.members.len() as u8,
            };
        }
    }

    for (at, region) in regions.iter_mut().enumerate() {
        if claimed[at] {
            continue;
        }
        if !region.windowed(all) {
            continue;
        }
        if !fat(trace, region, profile) {
            continue;
        }
        // One body: `arms` is 1, the per-arm term charged once.
        if !profits(nodes(region), 1, profile) {
            continue;
        }
        region.lowering = Lowering::If;
    }
}

/// Is the body worth an evaluation point that is paid whether it is taken
/// or not?
fn fat(trace: &Trace, region: &Region, profile: &DeviceProfile) -> bool {
    region_us(trace, region, profile) >= profile.fat_region_us
}

/// Are the launches this skips worth more than the evaluation point that
/// skips them? `skipped` counts launches, not regions; strict comparison so
/// a zeroed profile decides nothing.
fn profits(skipped: f32, arms: u8, profile: &DeviceProfile) -> bool {
    let paid = profile.cond_fixed_us + profile.cond_per_arm_us * f32::from(arms);
    skipped * profile.empty_launch_us > paid
}

/// One SWITCH group: the merge it came from and the regions that are its arms,
/// in arm order.
struct Group {
    merge: ValueId,
    members: Vec<usize>,
}

/// Every run of consecutive regions that is exactly one merge's arms, is
/// fat, profits, and provably has at most one live arm per fire.
///
/// Arms must be consecutive and in ascending order; an arm that split into
/// two regions is not a group and stays always-launch.
fn switch_groups(
    trace: &Trace,
    regions: &[Region],
    classes: &ClassTable,
    budget: &Budget,
    profile: &DeviceProfile,
) -> Vec<Group> {
    // Which region defines each value, so arms can be looked up by their
    // defining region.
    let mut region_of = vec![usize::MAX; trace.nodes.len()];
    for (at, region) in regions.iter().enumerate() {
        for node in region.nodes.clone() {
            if let Some(slot) = region_of.get_mut(node as usize) {
                *slot = at;
            }
        }
    }
    let defines: Vec<usize> = trace
        .values
        .iter()
        .map(|decl| match decl.def {
            Def::Op(node) => region_of.get(node as usize).copied().unwrap_or(usize::MAX),
            _ => usize::MAX,
        })
        .collect();

    let mut groups = Vec::new();
    let mut taken = vec![false; regions.len()];
    for (value, decl) in trace.values.iter().enumerate() {
        let Def::Merge(arms) = &decl.def else {
            continue;
        };
        let merge = ValueId(value as u32);
        if arms.len() < 2 || arms.len() > usize::from(u8::MAX) {
            continue;
        }
        if !fire_exclusive(classes, merge, budget) {
            continue;
        }
        let members: Vec<usize> = arms
            .iter()
            .map(|(arm, _)| defines.get(arm.0 as usize).copied().unwrap_or(usize::MAX))
            .collect();
        if members.iter().any(|&at| at == usize::MAX) {
            continue; // an arm no region defines: a weight, an input, a nested merge.
        }
        // `members` must be the ascending run itself, not merely its sorted
        // image: an out-of-order group would bracket the wrong nodes at
        // record time (the walk opens/closes by arm number).
        if members.windows(2).any(|pair| pair[1] != pair[0] + 1) {
            continue;
        }
        if members.iter().any(|&at| taken[at]) {
            continue;
        }
        let all = classes.classes.len();
        if !members
            .iter()
            .all(|&at| regions[at].windowed(all) && fat(trace, &regions[at], profile))
        {
            continue;
        }
        // Exclusivity must hold of the regions, not just the arms:
        // `fire_exclusive` is about a merge, but a region can carry extra
        // nodes and so a wider mask than its arm's own guard. Two members
        // with overlapping masks can both have rows in one fire, and SWITCH
        // runs only one body.
        if !pairwise_disjoint(regions, &members) {
            continue;
        }
        // A SWITCH skips every arm but the fattest one (the pessimistic
        // estimate of "the one taken").
        let launches: f32 = members.iter().map(|&at| nodes(&regions[at])).sum();
        let widest = members
            .iter()
            .map(|&at| nodes(&regions[at]))
            .fold(0.0f32, f32::max);
        #[allow(clippy::cast_possible_truncation)]
        if !profits(launches - widest, members.len() as u8, profile) {
            continue;
        }
        for &at in &members {
            taken[at] = true;
        }
        groups.push(Group { merge, members });
    }
    groups
}

/// Do no two of these regions have rows in the same fire? Two members are
/// simultaneously live exactly when their masks intersect.
fn pairwise_disjoint(regions: &[Region], members: &[usize]) -> bool {
    for (at, &left) in members.iter().enumerate() {
        for &right in &members[at + 1..] {
            if !regions[left].mask.disjoint(&regions[right].mask) {
                return false;
            }
        }
    }
    true
}

/// Can a fire hold two live arms of this merge? True when `max_lanes == 1`
/// (one lane, hence one class, hence one arm per fire) or when every class
/// resolves the merge to the same arm.
fn fire_exclusive(classes: &ClassTable, merge: ValueId, budget: &Budget) -> bool {
    if budget.max_lanes <= 1 {
        return true;
    }
    let Some(arms) = classes.arms_of(merge) else {
        return false;
    };
    let mut seen: Option<u8> = None;
    for arm in arms.iter().flatten() {
        match seen {
            Some(held) if held != *arm => return false,
            _ => seen = Some(*arm),
        }
    }
    true
}

