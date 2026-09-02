//! The `stream` pass: the dependency DAG over the capture-phase regions, and
//! the fork/join event points it bakes into the region table, so independent
//! windows (masked, decode, prefill) that otherwise ran one after another on
//! one stream — each leaving most of the device idle — can be in flight
//! together. Does not partition SMs; ships [`Region::sm_hint`] as a number
//! nothing reads yet.
//!
//! # The dependency rule
//!
//! Region `B` (later in program order) depends on region `A` (earlier) when
//! any of these holds. `writes(R)`/`reads(R)` are the union of every node's
//! `Operands::outputs`/`inputs` over `R`'s node range.
//!
//! ```text
//! RAW   reads(B)  ∩ writes(A) ≠ ∅        B reads what A wrote
//! WAR   writes(B) ∩ reads(A)  ≠ ∅        B overwrites what A read
//! WAW   writes(B) ∩ writes(A) ≠ ∅        both write the same value
//! CACHE both touch one cache space       an append is an effect no value carries
//! BAR   either carries a collective
//! SLAB  both name a `DeviceProfile::exclusive` op    a shared device workspace
//! ```
//!
//! The cache clause is conservative on purpose: `Attention::KvAppend` names
//! its cache among its inputs and produces no output, so a rule reading only
//! `outputs` would miss the write. Any node naming a `Def::Cache` value is
//! taken to write that space. The one exemption: two regions with disjoint
//! class masks cannot touch the same cache bytes, since a cache's storage is
//! owned per lane (a kv page belongs to one sequence, a recurrent slab to
//! one lane's row) — this exemption does *not* extend to values or to the
//! slab clause (a slab is one buffer for the whole device, not a per-lane
//! row, and `DeviceProfile::exclusive` is a device fact no pure compiler
//! pass can derive on its own).
//!
//! # Concurrency candidates and groups
//!
//! Two capture regions are candidates when the closure of the rule above has
//! no path either way between them AND their class masks are disjoint. A
//! fork group is a maximal run of *consecutive* candidates — consecutive
//! because the walk is a straight line and a stream switches at a region
//! boundary; this compiler has no region-reordering pass.
//!
//! ```text
//! r5  qkv, rope, kv_append   mask {0,2}   stream 0
//! r6  attention.masked       mask {2}     stream 0  open E2 ─────────────────
//! r7  attention.decode       mask {1}     stream 1  wait E2 ····· close E3
//! r8  attention.prefill      mask {0}     stream 2  wait E2 ····· close E4
//! r9  o_proj, mlp, …         mask {0,1,2} stream 0  wait E3, E4 ─────────────
//! ```
//!
//! The first member stays on the main stream and opens the group: it
//! records the entry event on the main stream after its own waits and
//! before its own first launch (at the *top* of the main arm, not the end
//! of the region before it, since two fork groups can sit back to back and
//! an event at the end of a side-stream region says nothing about where the
//! main stream is). Every arm waits on that one event, runs, and records an
//! exit; the region after the group waits on every exit. A group with no
//! region after it is not forked at all — a side stream that never rejoined
//! would end the capture on `cudaErrorStreamCaptureUnjoined`.
//!
//! # The cost gate
//!
//! Forking B out from beside A saves at most `min(cost A, cost B)` and costs
//! one [`DeviceProfile::event_pair_us`](crate::DeviceProfile::event_pair_us)
//! paid on every fire; both sides of the overlap must clear
//! [`DeviceProfile::fork_floor_us`] before a stream is handed out. A plan
//! with no candidate pair worth forking bakes every region on stream 0 with
//! no event point at all.
//!
//! # Determinism
//!
//! The assignment is a pure function of `(plan, classes, profile)`: regions
//! visited in program order, groups found left to right, streams handed out
//! in member order, events numbered in emission order.
//!
//! # The safety argument
//!
//! Two regions this pass puts on different streams cannot race:
//!
//! 1. They write disjoint values — the dependency DAG guarantees it (a
//!    shared write is a WAW edge, disqualifying the pair); a merge's arms
//!    share one rectangle but write disjoint rows of it, since their masks
//!    are disjoint and `Run::cut` slices every windowed write at the
//!    region's own window.
//! 2. They write disjoint arena bytes — the carve guarantees it:
//!    [`Concurrency`](crate::Concurrency) is threaded into `arena::carve`,
//!    so two values in paired regions never share a column.
//! 3. Everything else they share is read-only for the capture phase:
//!    weights landed once at load, the arena/pools/fire-inputs are reserved
//!    at the ceiling and never reallocated, attention schedules were built
//!    in the prepare phase (finished before the first capture region
//!    enqueued), and each plan value has its own workspace seat.
//!
//! `tests/no_concurrent_pair_shares_a_write.rs` checks clauses 1 and 2 over
//! the whole catalog.

use model_ir::{Def, Operands, Operation, Trace, ValueId};

use crate::compiled::{EventId, Lowering, Region};
use crate::budget::DeviceProfile;

/// What `stream` decided, beside the regions it stamped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamPlan {
    /// Region pairs that may be in flight together — what
    /// [`Concurrency::with_pairs`](crate::Concurrency::with_pairs) is built
    /// from and what the carve is widened by.
    pub pairs: Vec<(u32, u32)>,
    /// How many distinct events the template names; the shell creates this
    /// many `cudaEvent_t`s, once, at load.
    pub events: u32,
    /// How many streams the template uses, main included. `1` means nothing
    /// forked.
    pub streams: u32,
}

/// Assign streams and event points over `regions`, in place. The one door
/// into `stream`. `regions` comes out stamped with [`Region::stream`],
/// [`Region::wait`], [`Region::open`], [`Region::close`] and
/// [`Region::sm_hint`]; the relation the carve needs comes back.
pub(crate) fn fork(trace: &Trace, regions: &mut [Region], profile: &DeviceProfile) -> StreamPlan {
    if profile.side_streams == 0 || regions.len() < 3 {
        // Fewer than three regions has no group with a neighbour on both sides.
        return StreamPlan {
            pairs: Vec::new(),
            events: 0,
            streams: 1,
        };
    }

    let touches = Touches::of(trace, regions, profile);
    let ordered = closure(regions, &touches);
    // The same estimator `lowering` gates a conditional with, so the two passes
    // cannot disagree about the same region on the same profile.
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
        // A group needs a region after it to rejoin into, or a side stream
        // never rejoins.
        if group.end < regions.len() {
            seat(regions, &costs, profile, group.clone(), &mut forks);
        }
        at = group.end;
    }

    forks.pairs.sort_unstable();
    forks.pairs.dedup();
    forks
}

/// Hand a group's members their streams and their events. The first member
/// keeps the main stream; every later member that clears the cost gate takes
/// the next side stream, round-robin once the cap is reached (two members
/// sharing a side stream just run one after another, which is fine since
/// they're independent) — the pair table is built over different streams only.
fn seat(
    regions: &mut [Region],
    costs: &[f32],
    profile: &DeviceProfile,
    group: core::ops::Range<usize>,
    forks: &mut StreamPlan,
) {
    let main = group.start;
    // The cost gate, asked once per member against the arm it would overlap.
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

    // The entry event: recorded on the main stream at the top of the group's
    // first region, after its own waits and before its first launch. One
    // event serves every arm since `cudaStreamWaitEvent` does not consume.
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
    // The main arm gets a hint too: the split is between the arms, and one
    // side of a split is not a hint.

    for exit in exits {
        regions[group.end].wait.push(exit);
    }

    // The relation the carve widens on: every pair of members that ended up
    // on different streams.
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

/// The maximal run of consecutive capture regions starting at `at` that are
/// pairwise concurrency candidates, or `None` when that run is one region.
fn group_at(
    regions: &[Region],
    ordered: &Ordered,
    at: usize,
) -> Option<core::ops::Range<usize>> {
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

/// May this region be put on a stream of its own at all? A prepare region
/// may not (host work, runs before capture begins). A collective region may
/// not (NCCL matches calls by order, so a side-stream collective would be a
/// rendezvous out of position). A region no class runs may not (an empty
/// mask is disjoint from everything, making it a spurious candidate with
/// every other region). A conditional body is single-stream: it's a child
/// graph filled by `cudaStreamBeginCaptureToGraph`, while a fork's event
/// pair is an edge between two nodes of one parent graph — a dependency
/// CUDA has no way to express between the two.
fn forkable(region: &Region) -> bool {
    region.launches() && region.lowering == Lowering::AlwaysLaunch
}

/// Are these two regions a concurrency candidate: no path either way, and
/// disjoint class masks?
fn candidates(regions: &[Region], ordered: &Ordered, a: usize, b: usize) -> bool {
    !ordered.path(a, b) && !ordered.path(b, a) && regions[a].mask.disjoint(&regions[b].mask)
}

/// Every region's reads, writes, and cache spaces.
struct Touches {
    reads: Vec<Vec<ValueId>>,
    writes: Vec<Vec<ValueId>>,
    /// Which cache spaces the region's nodes name. `CacheRow::Kv` groups by
    /// its declared `space`; a `CacheRow::State` is its own space, since a
    /// recurrent bank is one slab per lane with no page id to share.
    spaces: Vec<Vec<u32>>,
    /// The region carries a collective — a barrier in both directions.
    barrier: Vec<bool>,
    /// The region names an op that claims a device-wide workspace
    /// (`DeviceProfile::exclusive`). Two such regions are ordered.
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
                    // A state bank shares no page space with anything, so
                    // it's numbered above every kv space it could collide with.
                    _ => u32::MAX - row,
                }),
                _ => None,
            })
            .collect();

        // A `Def::Merge` is data, never dispatched: a reader names the phi
        // and the arms are what wrote it, so a merge read must attribute to
        // its arms or the edge to each arm's producer would be invisible.
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
                            // Any mention of a cache is a write of its space.
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

    /// Is `b` (later) ordered directly after `a` (earlier)?
    fn edge(&self, regions: &[Region], a: usize, b: usize) -> bool {
        if self.barrier[a] || self.barrier[b] {
            return true;
        }
        // The device workspace: one buffer for the whole process, so disjoint
        // lanes buy nothing and the mask exemption below does not apply.
        if self.exclusive[a] && self.exclusive[b] {
            return true;
        }
        if meets(&self.writes[a], &self.reads[b])
            || meets(&self.writes[a], &self.writes[b])
            || meets(&self.reads[a], &self.writes[b])
        {
            return true;
        }
        // The cache clause and its one exemption: disjoint classes are
        // disjoint lanes are disjoint pages.
        meets(&self.spaces[a], &self.spaces[b]) && !regions[a].mask.disjoint(&regions[b].mask)
    }
}

/// The values a name resolves to once every phi in front of it is walked
/// through. A plain value is itself; a `Def::Merge` is the union of its arms,
/// recursively, since merges can nest inside merges.
fn resolve(trace: &Trace, through: &mut Vec<Option<Vec<ValueId>>>, value: ValueId) {
    let at = value.0 as usize;
    if through.get(at).is_some_and(Option::is_some) {
        return;
    }
    let Some(decl) = trace.values.get(at) else {
        return;
    };
    // Seated before the recursion, so a plan whose merges somehow cycle
    // resolves to the phi itself rather than overflowing the stack.
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

/// A value the resolution table has no row for is one the plan does not
/// declare, so it names nothing rather than panicking.
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

/// The transitive closure of the direct edges, as a bitset per region.
///
/// Program order makes this one backward sweep: every edge runs from a lower
/// index to a higher one, so `after[a]` is complete by the time any `b > a`
/// asks about it.
struct Ordered {
    /// `after[a]` holds every region reachable from `a`.
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

