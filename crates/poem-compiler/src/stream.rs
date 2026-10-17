use poem_ir::{Def, Operands, Operation, Trace, ValueId};

use crate::budget::DeviceProfile;
use crate::compiled::{EventId, Lowering, Region};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamPlan {
    pub pairs: Vec<(u32, u32)>,
    pub events: u32,
    pub streams: u32,
    /// The widest fire, in rows, a backend runs the side streams for; wider
    /// ones walk every region on the main stream. 0: every fire.
    pub forked_until: u32,
}

/// The regions forked onto side streams, each of `blocks`' branches
/// (`branch::blocks`) first: a branch's regions one chain on one stream, the first branch on the main one, the
/// others on side streams from where the first starts, all joined before
/// the join's region.
pub(crate) fn fork_with(
    trace: &Trace,
    regions: &mut [Region],
    profile: &DeviceProfile,
    blocks: &[crate::branch::Block],
) -> StreamPlan {
    if profile.side_streams == 0 || regions.len() < 3 {
        return StreamPlan {
            pairs: Vec::new(),
            events: 0,
            streams: 1,
            forked_until: 0,
        };
    }

    let touches = Touches::of(trace, regions, profile);
    let ordered = closure(regions, &touches);
    let costs: Vec<f32> = regions
        .iter()
        .map(|region| crate::lowering::region_us(trace, region, profile))
        .collect();

    let mut forks = StreamPlan {
        pairs: Vec::new(),
        events: 0,
        streams: 1,
        forked_until: 0,
    };
    for block in blocks {
        seat_block(regions, &costs, profile, block, &mut forks);
    }
    let mut at = 0usize;
    while at < regions.len() {
        let Some(group) = group_at(regions, &ordered, at) else {
            at += 1;
            continue;
        };
        if group.end < regions.len() {
            seat(regions, &costs, profile, group.clone(), &mut forks);
        }
        at = group.end;
    }

    forks.pairs.sort_unstable();
    forks.pairs.dedup();
    forks
}

/// Every region scheduled onto the main stream and `side_streams` side
/// streams by its dependencies, a list schedule over the regions' priced
/// costs: each launchable region starts on the lane it can start soonest on
/// (the main lane where no other starts it sooner, so a chain comes back to
/// it and its scratch is the main stream's, not a second copy sized to the
/// widest fire that forks), waiting on the close of the last region it
/// follows on each other lane; a region a graph cannot fork (or
/// the last) runs on the main lane after every lane's last region. Regions
/// on different lanes that no lane order or wait orders run side by side.
///
/// With `fixed` (each node's lane), a forkable region takes its first
/// node's lane rather than choosing one.
pub(crate) fn fork_dag(
    trace: &Trace,
    regions: &mut [Region],
    profile: &DeviceProfile,
    fixed: Option<&[u32]>,
) -> StreamPlan {
    let n = regions.len();
    let mut plan = StreamPlan {
        pairs: Vec::new(),
        events: 0,
        streams: 1,
        forked_until: 0,
    };
    if profile.side_streams == 0 || n < 3 {
        return plan;
    }
    plan.forked_until = profile.dag_streams_until;
    let lanes = 1 + profile.side_streams as usize;
    let touches = Touches::of(trace, regions, profile);
    let ordered = closure(regions, &touches);
    let costs: Vec<f32> = regions
        .iter()
        .map(|region| crate::lowering::region_us(trace, region, profile).max(0.5))
        .collect();
    let eligible: Vec<bool> = regions
        .iter()
        .enumerate()
        .map(|(at, region)| forkable(region) && !touches.barrier[at])
        .collect();

    let mut finish = vec![0f32; n];
    let mut free = vec![0f32; lanes];
    let mut lane_of = vec![0usize; n];
    let mut last: Vec<Option<usize>> = vec![None; lanes];
    // `synced[l][t]`: the latest region of lane `t` lane `l` has waited past.
    let mut synced: Vec<Vec<Option<usize>>> = vec![vec![None; lanes]; lanes];
    // Each wait's source region, for the order below.
    let mut sources: Vec<Vec<usize>> = vec![Vec::new(); n];
    for r in 0..n {
        let mut ready = 0f32;
        let mut need: Vec<Option<usize>> = vec![None; lanes];
        for a in 0..r {
            if !ordered.path(a, r) {
                continue;
            }
            ready = ready.max(finish[a]);
            let l = lane_of[a];
            need[l] = Some(need[l].map_or(a, |held| held.max(a)));
        }
        let barrier = !eligible[r] || r + 1 == n;
        let lane = if barrier {
            0
        } else if let Some(fixed) = fixed {
            fixed
                .get(regions[r].nodes.start as usize)
                .map_or(0, |&l| (l as usize).min(lanes - 1))
        } else {
            let start = |l: usize| ready.max(free[l]);
            (0..lanes)
                .min_by(|&x, &y| start(x).total_cmp(&start(y)).then_with(|| x.cmp(&y)))
                .unwrap_or(0)
        };
        if barrier {
            for (t, held) in last.iter().enumerate() {
                if let Some(a) = *held {
                    need[t] = Some(need[t].map_or(a, |b| b.max(a)));
                }
            }
        }
        // A side lane's first region follows the main lane's last at least:
        // its stream joins the fire there.
        if lane != 0
            && last[lane].is_none()
            && (0..lanes).all(|t| t == lane || need[t].is_none())
            && let Some(a) = last[0]
        {
            need[0] = Some(a);
        }
        for t in 0..lanes {
            if t == lane {
                continue;
            }
            let Some(a) = need[t] else { continue };
            if synced[lane][t].is_some_and(|held| held >= a) {
                continue;
            }
            let event = match regions[a].close {
                Some(event) => event,
                None => {
                    let event = EventId(plan.events);
                    plan.events += 1;
                    regions[a].close = Some(event);
                    event
                }
            };
            regions[r].wait.push(event);
            sources[r].push(a);
            synced[lane][t] = Some(a);
        }
        let start = ready.max(free[lane]);
        finish[r] = start + costs[r];
        free[lane] = finish[r];
        lane_of[r] = lane;
        regions[r].stream = lane as u32;
        last[lane] = Some(r);
        plan.streams = plan.streams.max(lane as u32 + 1);
    }

    // Side by side: regions on different lanes neither orders (lane order
    // and waits, transitively).
    let words = n.div_ceil(64);
    let mut before: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    let mut prev: Vec<Option<usize>> = vec![None; lanes];
    for r in 0..n {
        let mut set = vec![0u64; words];
        let add = |set: &mut Vec<u64>, a: usize, before: &Vec<Vec<u64>>| {
            set[a / 64] |= 1 << (a % 64);
            for (word, other) in set.iter_mut().zip(&before[a]) {
                *word |= *other;
            }
        };
        if let Some(p) = prev[lane_of[r]] {
            add(&mut set, p, &before);
        }
        for &a in &sources[r] {
            add(&mut set, a, &before);
        }
        before[r] = set;
        prev[lane_of[r]] = Some(r);
        for a in 0..r {
            if lane_of[a] != lane_of[r] && before[r][a / 64] & (1 << (a % 64)) == 0 {
                plan.pairs.push((a as u32, r as u32));
            }
        }
    }
    plan
}

/// Each node's lane as [`fork_dag`] schedules `regions` (a region an op).
pub(crate) fn dag_lanes(
    trace: &Trace,
    regions: &mut [Region],
    profile: &DeviceProfile,
) -> (Vec<u32>, std::collections::BTreeSet<u32>) {
    fork_dag(trace, regions, profile, None);
    let mut lanes = vec![0u32; trace.nodes.len()];
    // Where a coarser region must still break: where the lane changes,
    // before a node that waits on another lane (a region waits at its start)
    // and after one another lane waits on (a region closes at its end).
    let mut breaks = std::collections::BTreeSet::new();
    for region in regions.iter() {
        for node in region.nodes.clone() {
            if let Some(lane) = lanes.get_mut(node as usize) {
                *lane = region.stream;
            }
        }
        if !region.wait.is_empty() {
            breaks.insert(region.nodes.start);
        }
        if region.close.is_some() {
            breaks.insert(region.nodes.end);
        }
    }
    for j in 1..lanes.len() {
        if lanes[j] != lanes[j - 1] {
            breaks.insert(j as u32);
        }
    }
    (lanes, breaks)
}

fn seat(
    regions: &mut [Region],
    costs: &[f32],
    profile: &DeviceProfile,
    group: core::ops::Range<usize>,
    forks: &mut StreamPlan,
) {
    let main = group.start;
    let mut seated: Vec<(usize, u32)> = Vec::new();
    let mut next = 0u32;
    for member in group.clone().skip(1) {
        if costs[main].min(costs[member]) < profile.fork_floor_us {
            continue;
        }
        let stream = 1 + next % profile.side_streams;
        next += 1;
        seated.push((member, stream));
    }
    if seated.is_empty() {
        return;
    }

    let enter = EventId(forks.events);
    forks.events += 1;
    debug_assert!(regions[main].open.is_none(), "a region opens one group");
    regions[main].open = Some(enter);

    let mut exits: Vec<EventId> = Vec::new();
    for &(member, stream) in &seated {
        let exit = EventId(forks.events);
        forks.events += 1;
        regions[member].stream = stream;
        regions[member].wait.push(enter);
        regions[member].close = Some(exit);
        exits.push(exit);
        forks.streams = forks.streams.max(stream + 1);
    }

    for exit in exits {
        regions[group.end].wait.push(exit);
    }

    let streams: Vec<(usize, u32)> = core::iter::once((main, 0u32))
        .chain(seated.iter().copied())
        .collect();
    for (i, (a, sa)) in streams.iter().enumerate() {
        for (b, sb) in &streams[i + 1..] {
            if sa != sb {
                forks.pairs.push((*a as u32, *b as u32));
            }
        }
    }
}

fn group_at(regions: &[Region], ordered: &Ordered, at: usize) -> Option<core::ops::Range<usize>> {
    if !forkable(&regions[at]) {
        return None;
    }
    let mut end = at + 1;
    while end < regions.len()
        && forkable(&regions[end])
        && (at..end).all(|held| candidates(regions, ordered, held, end))
    {
        end += 1;
    }
    (end - at >= 2).then_some(at..end)
}

fn forkable(region: &Region) -> bool {
    region.launches()
        && region.lowering == Lowering::AlwaysLaunch
        && region.stream == 0
        && region.wait.is_empty()
        && region.open.is_none()
        && region.close.is_none()
}

/// One block's branches onto streams, where every region of it is forkable
/// and each branch its own run of regions.
fn seat_block(
    regions: &mut [Region],
    costs: &[f32],
    profile: &DeviceProfile,
    block: &crate::branch::Block,
    forks: &mut StreamPlan,
) {
    let Some(join) = regions.iter().position(|r| r.nodes.start == block.join) else {
        return;
    };
    let mut clusters: Vec<core::ops::Range<usize>> = Vec::new();
    for branch in &block.branches {
        let Some(first) = regions.iter().position(|r| r.nodes.start == branch.start) else {
            return;
        };
        let Some(last) = regions.iter().position(|r| r.nodes.end == branch.end) else {
            return;
        };
        if last < first || regions[first..=last].iter().any(|r| !forkable(r)) {
            return;
        }
        clusters.push(first..last + 1);
    }
    if clusters.len() < 2 || clusters.last().is_none_or(|c| c.end != join) {
        return;
    }
    let cost = |c: &core::ops::Range<usize>| costs[c.clone()].iter().sum::<f32>();
    let main = clusters[0].clone();
    let mut seated: Vec<(core::ops::Range<usize>, u32)> = Vec::new();
    let mut next = 0u32;
    for cluster in clusters.iter().skip(1) {
        if cost(&main).min(cost(cluster)) < profile.fork_floor_us {
            continue;
        }
        let stream = 1 + next % profile.side_streams;
        next += 1;
        seated.push((cluster.clone(), stream));
    }
    if seated.is_empty() {
        return;
    }
    let enter = EventId(forks.events);
    forks.events += 1;
    regions[main.start].open = Some(enter);
    let mut exits = Vec::new();
    for (cluster, stream) in &seated {
        let exit = EventId(forks.events);
        forks.events += 1;
        for at in cluster.clone() {
            regions[at].stream = *stream;
        }
        regions[cluster.start].wait.push(enter);
        regions[cluster.end - 1].close = Some(exit);
        exits.push(exit);
        forks.streams = forks.streams.max(stream + 1);
    }
    for exit in exits {
        regions[join].wait.push(exit);
    }
    let lanes: Vec<(core::ops::Range<usize>, u32)> = core::iter::once((main, 0u32))
        .chain(seated.iter().cloned())
        .collect();
    for (i, (a, sa)) in lanes.iter().enumerate() {
        for (b, sb) in &lanes[i + 1..] {
            if sa == sb {
                continue;
            }
            for x in a.clone() {
                for y in b.clone() {
                    forks.pairs.push((x.min(y) as u32, x.max(y) as u32));
                }
            }
        }
    }
}

fn candidates(regions: &[Region], ordered: &Ordered, a: usize, b: usize) -> bool {
    !ordered.path(a, b) && !ordered.path(b, a) && regions[a].mask.disjoint(&regions[b].mask)
}

struct Touches {
    reads: Vec<Vec<ValueId>>,
    writes: Vec<Vec<ValueId>>,
    spaces: Vec<Vec<u32>>,
    barrier: Vec<bool>,
    exclusive: Vec<bool>,
}

impl Touches {
    fn of(trace: &Trace, regions: &[Region], profile: &DeviceProfile) -> Touches {
        let spaces_of: Vec<Option<u32>> = trace
            .values
            .iter()
            .map(|value| match value.def {
                Def::Cache(row) => Some(match trace.caches.get(row as usize) {
                    Some(poem_ir::CacheRow::Kv { space, .. }) => *space,
                    _ => u32::MAX - row,
                }),
                _ => None,
            })
            .collect();

        let mut through: Vec<Option<Vec<ValueId>>> = vec![None; trace.values.len()];
        for at in 0..trace.values.len() {
            resolve(trace, &mut through, ValueId(at as u32));
        }

        let mut touches = Touches {
            reads: Vec::with_capacity(regions.len()),
            writes: Vec::with_capacity(regions.len()),
            spaces: Vec::with_capacity(regions.len()),
            barrier: Vec::with_capacity(regions.len()),
            exclusive: Vec::with_capacity(regions.len()),
        };
        let mut scratch = Vec::new();
        for region in regions {
            let (mut reads, mut writes, mut spaces) = (Vec::new(), Vec::new(), Vec::new());
            let mut barrier = false;
            let mut exclusive = false;
            for node in region.nodes.clone() {
                let Some(node) = trace.nodes.get(node as usize) else {
                    continue;
                };
                barrier |= matches!(node.op, Operation::Collective(_));
                exclusive |= profile
                    .exclusive
                    .iter()
                    .any(|named| named == node.op.name());
                scratch.clear();
                node.op.inputs(&mut scratch);
                for &named in &scratch {
                    for &value in arms(&through, named) {
                        match spaces_of.get(value.0 as usize).copied().flatten() {
                            Some(space) => spaces.push(space),
                            None => reads.push(value),
                        }
                    }
                }
                scratch.clear();
                node.op.outputs(&mut scratch);
                for &value in &scratch {
                    match spaces_of.get(value.0 as usize).copied().flatten() {
                        Some(space) => spaces.push(space),
                        None => writes.push(value),
                    }
                }
            }
            reads.sort_unstable();
            reads.dedup();
            writes.sort_unstable();
            writes.dedup();
            spaces.sort_unstable();
            spaces.dedup();
            touches.reads.push(reads);
            touches.writes.push(writes);
            touches.spaces.push(spaces);
            touches.barrier.push(barrier);
            touches.exclusive.push(exclusive);
        }
        touches
    }

    fn edge(&self, regions: &[Region], a: usize, b: usize) -> bool {
        if self.barrier[a] || self.barrier[b] {
            return true;
        }
        if self.exclusive[a] && self.exclusive[b] {
            return true;
        }
        if meets(&self.writes[a], &self.reads[b])
            || meets(&self.writes[a], &self.writes[b])
            || meets(&self.reads[a], &self.writes[b])
        {
            return true;
        }
        meets(&self.spaces[a], &self.spaces[b]) && !regions[a].mask.disjoint(&regions[b].mask)
    }
}

fn resolve(trace: &Trace, through: &mut Vec<Option<Vec<ValueId>>>, value: ValueId) {
    let at = value.0 as usize;
    if through.get(at).is_some_and(Option::is_some) {
        return;
    }
    let Some(decl) = trace.values.get(at) else {
        return;
    };
    through[at] = Some(vec![value]);
    let Def::Merge(arm_list) = &decl.def else {
        return;
    };
    let mut all = Vec::new();
    for (arm, _) in arm_list {
        resolve(trace, through, *arm);
        if let Some(Some(reached)) = through.get(arm.0 as usize) {
            all.extend_from_slice(reached);
        }
    }
    all.sort_unstable();
    all.dedup();
    if !all.is_empty() {
        through[at] = Some(all);
    }
}

fn arms(through: &[Option<Vec<ValueId>>], value: ValueId) -> &[ValueId] {
    match through.get(value.0 as usize) {
        Some(Some(reached)) => reached,
        _ => &[],
    }
}

fn meets<T: Ord>(a: &[T], b: &[T]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => return true,
        }
    }
    false
}

struct Ordered {
    after: Vec<Vec<u64>>,
}

impl Ordered {
    fn path(&self, a: usize, b: usize) -> bool {
        a < b && self.after[a][b / 64] & (1 << (b % 64)) != 0
    }
}

fn closure(regions: &[Region], touches: &Touches) -> Ordered {
    let n = regions.len();
    let words = n.div_ceil(64);
    let mut after: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    for a in (0..n).rev() {
        for b in a + 1..n {
            if !touches.edge(regions, a, b) {
                continue;
            }
            after[a][b / 64] |= 1 << (b % 64);
            let (head, tail) = after.split_at_mut(b);
            for (word, reached) in head[a].iter_mut().zip(&tail[0]) {
                *word |= *reached;
            }
        }
    }
    Ordered { after }
}
