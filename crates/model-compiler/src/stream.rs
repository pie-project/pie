use model_ir::{Def, Operands, Operation, Trace, ValueId};

use crate::budget::DeviceProfile;
use crate::compiled::{EventId, Lowering, Region};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamPlan {
    pub pairs: Vec<(u32, u32)>,
    pub events: u32,
    pub streams: u32,
}

pub(crate) fn fork(trace: &Trace, regions: &mut [Region], profile: &DeviceProfile) -> StreamPlan {
    if profile.side_streams == 0 || regions.len() < 3 {
        return StreamPlan {
            pairs: Vec::new(),
            events: 0,
            streams: 1,
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
    };
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
    region.launches() && region.lowering == Lowering::AlwaysLaunch
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
                    Some(model_ir::CacheRow::Kv { space, .. }) => *space,
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
