use model_ir::{ClassTable, Def, Trace, ValueId};

use crate::budget::{Budget, DeviceProfile};
use crate::compiled::{Lowering, Region};

pub(crate) fn region_us(trace: &Trace, region: &Region, profile: &DeviceProfile) -> f32 {
    region
        .nodes
        .clone()
        .filter_map(|node| trace.nodes.get(node as usize))
        .map(|node| profile.family_us.of(&node.op))
        .sum()
}

fn nodes(region: &Region) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        region.nodes.len() as f32
    }
}

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
        if !profits(nodes(region), 1, profile) {
            continue;
        }
        region.lowering = Lowering::If;
    }
}

fn fat(trace: &Trace, region: &Region, profile: &DeviceProfile) -> bool {
    region_us(trace, region, profile) >= profile.fat_region_us
}

fn profits(skipped: f32, arms: u8, profile: &DeviceProfile) -> bool {
    let paid = profile.cond_fixed_us + profile.cond_per_arm_us * f32::from(arms);
    skipped * profile.empty_launch_us > paid
}

struct Group {
    merge: ValueId,
    members: Vec<usize>,
}

fn switch_groups(
    trace: &Trace,
    regions: &[Region],
    classes: &ClassTable,
    budget: &Budget,
    profile: &DeviceProfile,
) -> Vec<Group> {
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
        if members.contains(&usize::MAX) {
            continue;
        }
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
        if !pairwise_disjoint(regions, &members) {
            continue;
        }
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
