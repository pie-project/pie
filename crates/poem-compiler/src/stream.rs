use poem_ir::{Def, Operands, Operation, Trace, ValueId};

use crate::budget::DeviceProfile;
use crate::compiled::{EventId, Lowering, Phase, Region};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamPlan {
    pub pairs: Vec<(u32, u32)>,
    pub events: u32,
    pub streams: u32,
    /// The widest fire, in rows, a backend runs the side streams for; wider
    /// ones walk every region on the main stream. 0: every fire.
    pub forked_until: u32,
}

/// The regions forked onto side streams.
pub(crate) fn fork_with(
    trace: &Trace,
    regions: &mut [Region],
    units_of: &[u32],
    profile: &DeviceProfile,
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
    let mut at = 0usize;
    while at < regions.len() {
        let Some(group) = group_at(regions, units_of, &ordered, at) else {
            at += 1;
            continue;
        };
        if group.end < regions.len()
            && regions[group.end].phase == Phase::Capture
            && units_of.get(group.end) == units_of.get(group.start)
        {
            seat(regions, &costs, profile, group.clone(), &mut forks);
        }
        at = group.end;
    }

    forks.pairs.sort_unstable();
    forks.pairs.dedup();
    forks
}

/// Every region list-scheduled by its dependencies and priced cost onto the
/// main and `side_streams` side streams: a launchable region starts on the
/// lane it can start soonest (the main one on ties, so scratch stays the main
/// stream's), waiting on the last region it follows on each other lane; an
/// unforkable region (and the last) runs on the main lane after every lane.
///
/// With `fixed` (each node's lane), a forkable region takes its first
/// node's lane rather than choosing one.
pub(crate) fn fork_dag(
    trace: &Trace,
    regions: &mut [Region],
    units_of: &[u32],
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

    // Priced per request class: a fire runs one class, so regions of
    // classes apart (a decode arm and a prefill one) never hold a lane
    // against each other.
    let classes = regions
        .iter()
        .filter_map(|r| r.mask.iter().max())
        .max()
        .map_or(1, |c| c + 1);
    let mut finish = vec![vec![0f32; classes]; n];
    let mut free = vec![vec![0f32; classes]; lanes];
    let mut lane_of = vec![0usize; n];
    let mut last: Vec<Option<usize>> = vec![None; lanes];
    // `synced[l][t]`: the latest region of lane `t` lane `l` has waited past.
    let mut synced: Vec<Vec<Option<usize>>> = vec![vec![None; lanes]; lanes];
    // Each wait's source region, for the order below.
    let mut sources: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut unit_start = 0usize;
    for r in 0..n {
        if regions[r].phase != Phase::Capture {
            continue;
        }
        let unit = units_of.get(r).copied().unwrap_or(0);
        let unit_first = r == 0
            || regions[r - 1].phase != Phase::Capture
            || units_of.get(r - 1).copied().unwrap_or(0) != unit;
        if unit_first {
            unit_start = r;
            free.iter_mut().for_each(|lane| lane.fill(0.0));
            last.fill(None);
            for row in &mut synced {
                row.fill(None);
            }
        }
        let unit_last = r + 1 == n
            || regions[r + 1].phase != Phase::Capture
            || units_of.get(r + 1).copied().unwrap_or(0) != unit;

        let mut ready = vec![0f32; classes];
        let mut need: Vec<Option<usize>> = vec![None; lanes];
        for a in unit_start..r {
            if !ordered.path(a, r) {
                continue;
            }
            for (at, held) in ready.iter_mut().zip(&finish[a]) {
                *at = at.max(*held);
            }
            let l = lane_of[a];
            need[l] = Some(need[l].map_or(a, |held| held.max(a)));
        }
        let barrier = !eligible[r] || unit_first || unit_last;
        let lane = if barrier {
            0
        } else if let Some(fixed) = fixed {
            fixed
                .get(regions[r].nodes.start as usize)
                .map_or(0, |&l| (l as usize).min(lanes - 1))
        } else {
            // A side lane only where the region starts sooner there in every
            // class it runs in.
            let start = |l: usize| -> f32 {
                regions[r]
                    .mask
                    .iter()
                    .map(|c| ready[c].max(free[l][c]))
                    .sum()
            };
            let sooner = |l: usize| {
                !regions[r].mask.is_empty()
                    && regions[r]
                        .mask
                        .iter()
                        .all(|c| ready[c].max(free[l][c]) < ready[c].max(free[0][c]))
            };
            (1..lanes)
                .filter(|&l| sooner(l))
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
        for c in regions[r].mask.iter() {
            finish[r][c] = ready[c].max(free[lane][c]) + costs[r];
            free[lane][c] = finish[r][c];
        }
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
        if regions[r].phase != Phase::Capture {
            continue;
        }
        let unit = units_of.get(r).copied().unwrap_or(0);
        if r == 0
            || regions[r - 1].phase != Phase::Capture
            || units_of.get(r - 1).copied().unwrap_or(0) != unit
        {
            prev.fill(None);
        }
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
            if regions[a].phase == Phase::Capture
                && units_of.get(a).copied().unwrap_or(0) == unit
                && lane_of[a] != lane_of[r]
                && before[r][a / 64] & (1 << (a % 64)) == 0
            {
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
    units_of: &[u32],
    profile: &DeviceProfile,
) -> (Vec<u32>, std::collections::BTreeSet<u32>) {
    fork_dag(trace, regions, units_of, profile, None);
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

fn group_at(
    regions: &[Region],
    units_of: &[u32],
    ordered: &Ordered,
    at: usize,
) -> Option<core::ops::Range<usize>> {
    if !forkable(&regions[at]) {
        return None;
    }
    let unit = units_of.get(at);
    let mut end = at + 1;
    while end < regions.len()
        && forkable(&regions[end])
        && units_of.get(end) == unit
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

fn candidates(regions: &[Region], ordered: &Ordered, a: usize, b: usize) -> bool {
    !ordered.path(a, b) && !ordered.path(b, a) && regions[a].mask.disjoint(&regions[b].mask)
}

struct Touches {
    reads: Vec<Vec<ValueId>>,
    writes: Vec<Vec<ValueId>>,
    /// The cache rows (`Def::Cache`) each region reads or writes.
    rows: Vec<Vec<u32>>,
    barrier: Vec<bool>,
    exclusive: Vec<bool>,
}

impl Touches {
    fn of(trace: &Trace, regions: &[Region], profile: &DeviceProfile) -> Touches {
        // A cache or state row is ordered by its own bytes: two regions meet
        // there only where they touch the same row. Its page table and write
        // positions are read-only fire inputs, so rows sharing them never wait.
        let rows_of: Vec<Option<u32>> = trace
            .values
            .iter()
            .map(|value| match value.def {
                Def::Cache(row) => Some(row_key(trace, row)),
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
            rows: Vec::with_capacity(regions.len()),
            barrier: Vec::with_capacity(regions.len()),
            exclusive: Vec::with_capacity(regions.len()),
        };
        let mut scratch = Vec::new();
        for region in regions {
            let (mut reads, mut writes, mut rows) = (Vec::new(), Vec::new(), Vec::new());
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
                        match rows_of.get(value.0 as usize).copied().flatten() {
                            Some(row) => rows.push(row),
                            None => reads.push(value),
                        }
                    }
                }
                scratch.clear();
                node.op.outputs(&mut scratch);
                for &value in &scratch {
                    match rows_of.get(value.0 as usize).copied().flatten() {
                        Some(row) => rows.push(row),
                        None => writes.push(value),
                    }
                }
            }
            reads.sort_unstable();
            reads.dedup();
            writes.sort_unstable();
            writes.dedup();
            rows.sort_unstable();
            rows.dedup();
            touches.reads.push(reads);
            touches.writes.push(writes);
            touches.rows.push(rows);
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
        meets(&self.rows[a], &self.rows[b]) && !regions[a].mask.disjoint(&regions[b].mask)
    }
}

/// The key a cache row is ordered by: the row itself.
#[cfg(not(test))]
fn row_key(_: &Trace, row: u32) -> u32 {
    row
}

#[cfg(test)]
fn row_key(trace: &Trace, row: u32) -> u32 {
    tests::KEY
        .with(core::cell::Cell::get)
        .map_or(row, |key| key(trace, row))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Budget, Budgets, PATCH_LATTICE_FLOOR, PatchLadder, VoxelLadder, compile_axes, region,
    };
    use core::cell::Cell;
    use poem::Platform;
    use poem_ir::{ParamSource, RowAxis, Ty, resolve_classes};

    type Key = fn(&Trace, u32) -> u32;

    thread_local! {
        /// The key [`row_key`] orders a cache row by in place of the row.
        pub(super) static KEY: Cell<Option<Key>> = const { Cell::new(None) };
    }

    /// The order before rows: a kv row by the space it shares with the
    /// space's other rows, any other row by itself. Verification scaffolding:
    /// it and the test below go when kv spaces leave the IR.
    fn by_space(trace: &Trace, row: u32) -> u32 {
        match trace.caches.get(row as usize) {
            Some(poem_ir::CacheRow::Kv { space, .. }) => *space,
            _ => u32::MAX - row,
        }
    }

    fn under<T>(key: Option<Key>, run: impl FnOnce() -> T) -> T {
        KEY.with(|held| held.set(key));
        let out = run();
        KEY.with(|held| held.set(None));
        out
    }

    fn budgets_of(trace: &Trace) -> Budgets {
        let states = |axis: RowAxis| {
            trace.values.iter().any(|decl| {
                matches!(&decl.ty, Ty::Tensor { shape, .. }
                    if shape.first().and_then(|dim| dim.axis()) == Some(axis))
            })
        };
        let seats = trace
            .params
            .iter()
            .filter(|param| param.source == ParamSource::Registered)
            .map(|param| param.shape.first().copied().unwrap_or(0))
            .min()
            .unwrap_or(0);
        let budget = Budget {
            max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
            ..Budget::default()
        };
        let mut budgets = Budgets::of(budget.clone());
        if states(RowAxis::Patches) {
            let max_patches = budget.max_tokens.clamp(PATCH_LATTICE_FLOOR, 4096);
            let mut buckets = Vec::new();
            let mut rung = PATCH_LATTICE_FLOOR;
            while rung < max_patches {
                buckets.push(rung);
                rung = rung.saturating_mul(2);
            }
            buckets.push(max_patches);
            budgets = budgets.with_patches(PatchLadder {
                max_images: (max_patches / PATCH_LATTICE_FLOOR).max(1),
                max_patches,
                buckets,
            });
        }
        if states(RowAxis::Voxels) {
            budgets = budgets.with_voxels(VoxelLadder::new(4096, 4));
        }
        budgets
    }

    /// Over one region list: an edge ordering by row adds that ordering by
    /// space lacked, or one it drops between regions sharing a row, is
    /// wrong. Returns how many it drops.
    fn compare(
        trace: &Trace,
        regions: &[Region],
        profile: &DeviceProfile,
        at: &str,
        wrong: &mut Vec<String>,
    ) -> usize {
        let rows = under(None, || Touches::of(trace, regions, profile));
        let spaces = under(Some(by_space), || Touches::of(trace, regions, profile));
        let mut dropped = 0;
        for a in 0..regions.len() {
            for b in a + 1..regions.len() {
                match (rows.edge(regions, a, b), spaces.edge(regions, a, b)) {
                    (true, false) => wrong.push(format!("{at}: new edge {a}->{b}")),
                    (false, true) => {
                        dropped += 1;
                        if meets(&rows.rows[a], &rows.rows[b]) {
                            wrong.push(format!("{at}: dropped {a}->{b} shares a row"));
                        }
                    }
                    _ => {}
                }
            }
        }
        dropped
    }

    /// Every catalog deployment on every platform, under the default
    /// profile and one scheduled by dependencies: ordering by cache row
    /// keeps a subset of the edges ordering by kv space did, each one it
    /// drops joins regions that share no row, and the plans it changes are
    /// listed with their concurrent pairs before and after.
    ///
    /// `cargo test --release -p pie-poem-compiler --lib rows_order -- --ignored --nocapture`
    #[test]
    #[ignore = "compiles every catalog deployment twice per profile"]
    fn rows_order_a_subset_of_what_spaces_did() {
        let dag = DeviceProfile {
            dag_streams: true,
            ..DeviceProfile::default()
        };
        let profiles = [("default", DeviceProfile::default()), ("dag_streams", dag)];
        let platforms = [
            Platform::Cuda,
            Platform::Metal,
            Platform::Wgpu,
            Platform::Vulkan,
        ];
        let jobs: Vec<(&crate::catalog::Deployment, Platform)> = crate::catalog::deployments()
            .flat_map(|row| platforms.map(|platform| (row, platform)))
            .collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let tally = std::sync::Mutex::new(Tally::default());
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    loop {
                        let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(row, platform)) = jobs.get(at) else {
                            break;
                        };
                        let mut mine = Tally::default();
                        one(row, platform, &profiles, &mut mine);
                        tally.lock().expect("no job panics").add(mine);
                    }
                });
            }
        });
        let tally = tally.into_inner().expect("no job panics");
        let mut lines = tally.lines;
        lines.sort();
        for line in &lines {
            println!("{line}");
        }
        println!(
            "{} plans ({} refused): {} identical, {} differ ({} more concurrent, {} less); \
             {} edges dropped",
            tally.plans,
            tally.refused,
            tally.same,
            tally.plans - tally.same,
            tally.gained,
            tally.lost,
            tally.dropped,
        );
        assert!(
            tally.wrong.is_empty(),
            "{} violations:\n{}",
            tally.wrong.len(),
            tally.wrong.join("\n")
        );
    }

    #[derive(Default)]
    struct Tally {
        plans: usize,
        same: usize,
        refused: usize,
        dropped: usize,
        gained: usize,
        lost: usize,
        lines: Vec<String>,
        wrong: Vec<String>,
    }

    impl Tally {
        fn add(&mut self, other: Tally) {
            self.plans += other.plans;
            self.same += other.same;
            self.refused += other.refused;
            self.dropped += other.dropped;
            self.gained += other.gained;
            self.lost += other.lost;
            self.lines.extend(other.lines);
            self.wrong.extend(other.wrong);
        }
    }

    fn one(
        row: &crate::catalog::Deployment,
        platform: Platform,
        profiles: &[(&str, DeviceProfile)],
        tally: &mut Tally,
    ) {
        let Ok(trace) = row.try_trace(platform) else {
            tally.refused += 1;
            return;
        };
        let budgets = budgets_of(&trace);
        let classes = resolve_classes(&trace).expect("classes resolve");
        let every: std::collections::BTreeSet<u32> = (0..trace.nodes.len() as u32).collect();
        let mut fine =
            region::coalesce_breaking(&trace, &classes, &every).expect("one region an op");
        region::hoist(&trace, &mut fine).expect("hoists");
        for (name, profile) in profiles {
            let at = format!("{} on {platform:?} under {name}", row.name);
            let new = under(None, || compile_axes(&trace, &budgets, profile));
            let old = under(Some(by_space), || compile_axes(&trace, &budgets, profile));
            let (Ok(new), Ok(old)) = (new, old) else {
                tally.refused += 1;
                continue;
            };
            tally.plans += 1;
            for regions in [&fine, &new.regions, &old.regions] {
                tally.dropped += compare(&trace, regions, profile, &at, &mut tally.wrong);
            }
            if new == old {
                tally.same += 1;
                continue;
            }
            let (before, after) = (old.streams.pairs.len(), new.streams.pairs.len());
            if after > before {
                tally.gained += 1;
            } else if after < before {
                tally.lost += 1;
            }
            tally
                .lines
                .push(format!("{at}: {before} -> {after} concurrent pairs"));
        }
    }
}
