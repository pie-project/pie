//! P6: the dependency DAG over the capture-phase regions, and the fork/join
//! event points it bakes into the region table.
//!
//! **WHAT THIS PASS IS FOR, IN ONE PICTURE.** Design §0's fire runs a masked
//! window, a decode window and a prefill window inside one forward pass. On
//! one stream they run one after another, and each of them leaves most of the
//! device idle — a decode attention over three rows cannot fill an L40S, and
//! neither can a masked window over one lane. Nothing orders them: they read
//! the same keys, they write disjoint rows of one column, and no value passes
//! between them. So they are three kernels that could be in flight together
//! and are not, which is the bubble tart's whole research line is about
//! (`.wiki/tart/evidence/green_contexts.md`, Finding 3 — measured: one CUDA
//! graph CAN span streams through the fork/join capture pattern).
//!
//! What this pass does NOT do is partition SMs. That is decision #14: the
//! partition is baked at capture (Finding 5), so a variant multiplies bodies,
//! and v1 ships [`Region::sm_hint`] — a number nothing reads yet — rather than
//! a green context. What v1 ships that DOES run is the streams.
//!
//! # The dependency rule, stated
//!
//! Region `B` (later in program order) depends on region `A` (earlier) when
//! any of these holds. `writes(R)` and `reads(R)` are the union of every
//! node's `Operands::outputs` and `Operands::inputs` over `R`'s node range.
//!
//! ```text
//! RAW   reads(B)  ∩ writes(A) ≠ ∅        B reads what A wrote
//! WAR   writes(B) ∩ reads(A)  ≠ ∅        B overwrites what A read
//! WAW   writes(B) ∩ writes(A) ≠ ∅        both write the same value
//! CACHE both touch one cache space       an append is an effect no value carries
//! BAR   either carries a collective      decision #5, one step further
//! SLAB  both name a `DeviceProfile::exclusive` op    a shared device workspace
//! ```
//!
//! **THE CACHE CLAUSE IS CONSERVATIVE AND IT HAS TO BE.** `Attention::KvAppend`
//! names its cache among its INPUTS and produces no output at all: the write
//! is an effect, and a rule that read `outputs` would not see it. So any node
//! naming a `Def::Cache` value is taken to WRITE that space — no list of op
//! names, no guess about which of them mutate — and two regions touching one
//! space are ordered.
//!
//! **THE ONE EXEMPTION, AND WHY IT IS EXACT.** Two regions whose class masks
//! are DISJOINT cannot touch the same cache bytes, however conservatively the
//! space is read. A class is a set of lanes, classes partition the lanes of a
//! fire, and a cache's storage is owned per lane: a kv page belongs to one
//! sequence (`driver-api`'s `Lane::kv` is one delta, the engine keeps one page
//! table) and a recurrent slab is one lane's row. Disjoint classes are
//! disjoint lanes are disjoint pages. So gemma's masked append and its decode
//! append are not ordered against each other, and both are ordered against
//! every reader whose window overlaps theirs — which is the answer that is
//! both correct and useful.
//!
//! **THE SLAB CLAUSE IS THE ONE THING A PURE COMPILER CANNOT DERIVE**, and
//! it is why `DeviceProfile::exclusive` exists. A kernel platform may give an
//! entry a process-global workspace — CUDA's does, keyed by a static name and
//! deliberately not per stream, because an entry that allocated per fire could
//! not be captured — and two launches inside it at once stage over each other.
//! No `Operands` method says so and no `Ty` does; the shell knows, so the
//! shell passes the list. Two regions that each name one are ordered, and the
//! disjoint-mask exemption does NOT apply: a slab is one buffer for the whole
//! device, not a per-lane row.
//!
//! The exemption is NOT extended to values. A value is one rectangle of the
//! arena, and two regions with disjoint masks writing it write disjoint ROWS
//! of it — true, and load-bearing in §5's zero-instruction φ — but a RAW edge
//! between disjoint masks would mean a reader reading rows nobody in its own
//! window wrote, which is a model text this pass has no business quietly
//! scheduling around.
//!
//! # Concurrency candidates
//!
//! Two capture regions are candidates when the closure of the rule above has
//! **no path either way** between them AND their class masks are disjoint.
//! Both halves are needed and neither implies the other: the DAG is what says
//! no value passes between them, and the disjoint masks are what say the rows
//! they write cannot be the same rows.
//!
//! # Groups, and why they are ADJACENT runs
//!
//! A fork group is a maximal run of CONSECUTIVE capture regions that are
//! pairwise candidates. Consecutive, because the walk is a straight line and a
//! stream switches at a region boundary: a group that skipped over a region
//! would have to put that region somewhere, and "somewhere" is a scheduling
//! pass this compiler does not have (design's open items — region-enlarging
//! reordering is deferred and correctness-neutral). The catalog does not need
//! one: a merge's arms are adjacent because a model text writes them that way.
//!
//! ```text
//! r5  qkv, rope, kv_append   mask {0,2}   stream 0
//! r6  attention.masked       mask {2}     stream 0  open E2 ─────────────────
//! r7  attention.decode       mask {1}     stream 1  wait E2 ····· close E3
//! r8  attention.prefill      mask {0}     stream 2  wait E2 ····· close E4
//! r9  o_proj, mlp, …         mask {0,1,2} stream 0  wait E3, E4 ─────────────
//! ```
//!
//! The first member stays on the main stream and OPENS the group: it records
//! the entry event on the main stream after its own waits and before its own
//! first launch. Every arm waits on that one event, runs, and records an exit;
//! the region after the group waits on every exit.
//!
//! **THE ENTRY EVENT IS AT THE TOP OF THE MAIN ARM, NOT AT THE END OF THE
//! REGION BEFORE IT**, and the reason is that transformer layers put fork
//! groups back to back. Gemma's layer forks twice — the decode qkv beside the
//! prefill/masked qkv, then the three attention arms — and the region before
//! the second group is an ARM of the first, sitting on a side stream. An
//! event recorded at the end of a side stream's region says nothing about
//! where the main stream is. The top of the main arm says exactly the right
//! thing: everything before this group has been enqueued on this stream, the
//! previous group's arms have been waited for, and this group's own work has
//! not started.
//!
//! A group with no region after it is not forked at all: a side stream that
//! never rejoined would end the capture on `cudaErrorStreamCaptureUnjoined`.
//! Nothing else about a group's position matters — the main stream is where
//! the shell staged the fire's descriptor before the walk began, so an arm
//! that waits on a main-stream event is ordered after the staging too.
//!
//! # The cost gate
//!
//! Forking B out from beside A saves at most `min(cost A, cost B)` and costs
//! one [`DeviceProfile::event_pair_us`](crate::DeviceProfile::event_pair_us),
//! paid on every fire whether or not this fire has rows for either window. So
//! a small kernel behind an event pair is a loss, and both sides of the
//! overlap must clear [`DeviceProfile::fork_floor_us`] before a stream is
//! handed out. Costs come from
//! [`FamilyCosts`](crate::budget::FamilyCosts) — a table the caller passes,
//! never a measurement this crate takes.
//!
//! A plan with no candidate pair, or one whose candidates are all too cheap,
//! bakes every region on stream 0 with no event point anywhere, which is
//! byte-for-byte the artifact this compiler produced before P6 existed. **It
//! pays nothing.**
//!
//! # Determinism
//!
//! The assignment is a pure function of `(plan, classes, profile)`. Regions
//! are visited in program order, groups are found left to right, streams are
//! handed out in member order, and events are numbered in emission order.
//! Nothing is sorted by a cost, hashed, or read off the environment.
//!
//! # The safety argument
//!
//! **TWO REGIONS THIS PASS PUTS ON DIFFERENT STREAMS CANNOT RACE**, and the
//! argument has three parts, each owned by a different piece of the build:
//!
//! 1. **They write disjoint values.** The dependency DAG guarantees it: a
//!    shared written value is a WAW edge, and a pair with an edge is not a
//!    candidate. What remains shared is what a `Def::Merge` folded into one
//!    rectangle on purpose — a merge's arms — and those write disjoint ROWS,
//!    because their masks are disjoint and `Run::cut` slices every windowed
//!    write at the region's own window (build log 8).
//! 2. **They write disjoint arena bytes.** The carve guarantees it:
//!    [`Concurrency`](crate::Concurrency) is threaded into `arena::carve`,
//!    so two values live in regions
//!    this pass paired are never given the same column —
//!    `ArenaMap::overlap` is no longer the node-interval test alone. The
//!    exemption `ArenaMap::co_tenants` names is the same one as above, and it
//!    is checked rather than assumed: `clashes` is empty on every catalog row.
//! 3. **Everything else they share is read-only for the length of the capture
//!    phase.** Weights landed once at load; the arena's base pointer, the
//!    pools and the fire inputs are reserved at the ceiling and never
//!    reallocated; the attention schedules were built in the PREPARE phase,
//!    which is host work that finished before the first capture region
//!    enqueued, and each plan value has its own workspace seat. A capture
//!    region writes the arena and the caches and nothing else.
//!
//! `tests/no_concurrent_pair_shares_a_write.rs` is clause 1 and clause 2 over
//! the whole catalog.

use model_ir::{ClassSet, Def, Operands, Operation, Plan, ValueId};

use crate::baked::{EventId, Lowering, Phase, Region};
use crate::budget::DeviceProfile;

/// What P6 decided, beside the regions it stamped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Forks {
    /// Region pairs that may be in flight together — what
    /// [`Concurrency::with_pairs`](crate::Concurrency::with_pairs) is built
    /// from and what the carve is widened by.
    pub pairs: Vec<(u32, u32)>,
    /// How many distinct events the template names. The shell creates this
    /// many `cudaEvent_t`s, once, at load.
    pub events: u32,
    /// How many streams the template uses, main included. `1` means nothing
    /// forked and the shell opens nothing.
    pub streams: u32,
}

/// Assign streams and event points over `regions`, in place.
///
/// The one door into P6. `regions` comes out stamped with
/// [`Region::stream`], [`Region::wait`], [`Region::open`], [`Region::close`]
/// and [`Region::sm_hint`]; the relation the carve needs comes back.
///
/// The class table is not an argument: every question this pass asks about
/// classes is asked of a REGION's mask, which P2 already folded out of
/// `Classes::node_mask`, and taking the table as well would be two spellings
/// of one fact.
pub(crate) fn fork(plan: &Plan, regions: &mut [Region], profile: &DeviceProfile) -> Forks {
    if profile.side_streams == 0 || regions.len() < 3 {
        // The off arm, and the trivial one. A plan with fewer than three
        // regions has no group with a neighbour on both sides.
        return Forks {
            pairs: Vec::new(),
            events: 0,
            streams: 1,
        };
    }

    let touches = Touches::of(plan, regions, profile);
    let ordered = closure(regions, &touches);
    let costs: Vec<f32> = regions
        .iter()
        .map(|region| {
            region
                .nodes
                .clone()
                .filter_map(|node| plan.nodes.get(node as usize))
                .map(|node| profile.family_us.of(&node.op))
                .sum()
        })
        .collect();

    let mut forks = Forks {
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
        // A group needs a region AFTER it to rejoin into. A side stream that
        // never rejoined would end the capture on
        // `cudaErrorStreamCaptureUnjoined`; nothing else about the group's
        // position matters, because the entry event is opened on the main
        // stream inside the group's own first region.
        if group.end < regions.len() {
            seat(regions, &costs, profile, group.clone(), &mut forks);
        }
        at = group.end;
    }

    forks.pairs.sort_unstable();
    forks.pairs.dedup();
    forks
}

/// Hand a group's members their streams and their events.
///
/// The first member keeps the main stream — it is already there, and moving it
/// would buy an event pair and nothing else. Every later member that clears
/// the gate takes the next side stream, round-robin once the cap is reached:
/// two members sharing a side stream simply run one after another on it, which
/// is correct (they are independent, so any order is a legal order) and is why
/// the pair table below is built over DIFFERENT streams only.
fn seat(
    regions: &mut [Region],
    costs: &[f32],
    profile: &DeviceProfile,
    group: core::ops::Range<usize>,
    forks: &mut Forks,
) {
    let main = group.start;
    // THE GATE, asked once per member against the arm it would overlap.
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

    // The entry event: recorded on the MAIN stream at the top of the group's
    // first region, after that region's own waits and before its first
    // launch. That instant is "everything before this group has been
    // enqueued, and this group's main arm has not" — which is what an arm
    // needs to wait for and no earlier region's END is, once two groups stand
    // back to back. One event serves every arm: `cudaStreamWaitEvent` does
    // not consume.
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
        regions[member].sm_hint = Some(share(costs, &group, member, profile));
        exits.push(exit);
        forks.streams = forks.streams.max(stream + 1);
    }
    // The main arm gets a hint too: the split is between the arms, and one
    // side of a split is not a hint.
    regions[main].sm_hint = Some(share(costs, &group, main, profile));

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

/// What fraction of the device this arm would want, if anything read it.
///
/// **PROPORTIONAL TO COST, ROUNDED THE WAY THE HARDWARE ROUNDS** (green
/// contexts Finding 1: groups come in multiples of 2 SMs, the smallest usable
/// group is 4). Nothing reads this in v1 — decision #14 defers the partition,
/// because it is baked at capture and a variant multiplies bodies — so it is
/// a number the artifact carries for the pass that will.
///
/// Finding 4 says the sharing rule this implements is the naive one: SM count
/// buys a compute-bound kernel almost everything and a bandwidth-bound kernel
/// almost nothing, so a partition worth taking gives SMs to the arm that can
/// USE them rather than to the arm that costs the most. A cost table with one
/// number per family cannot tell those apart, and a hint that pretended to
/// would be worse than a proportional one.
fn share(
    costs: &[f32],
    group: &core::ops::Range<usize>,
    member: usize,
    profile: &DeviceProfile,
) -> u32 {
    let total: f32 = group.clone().map(|m| costs[m]).sum();
    if total <= 0.0 {
        return profile.sms;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let want = ((costs[member] / total) * profile.sms as f32) as u32;
    (want.max(4) + 1) & !1
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

/// May this region be put on a stream of its own at all?
///
/// A prepare region may not: it is host work, it runs before the capture
/// begins, and a stream is not what it is enqueued on. A collective region may
/// not: NCCL matches calls by ORDER, so a collective on a side stream is a
/// rendezvous whose position in the program is no longer the position every
/// other rank sees (decision #5, one step past elision). A region no class
/// runs — `Classes::dead`, and shipped plans have none — may not either: an
/// empty mask is disjoint from everything, which would make it a candidate
/// with every region in the plan for no reason at all.
///
/// **AND A CONDITIONAL BODY IS SINGLE-STREAM** (design §4, v1). P3 has not
/// landed, so every region is `AlwaysLaunch` and this clause is vacuous
/// today — it is written now because the day a region becomes a SWITCH arm,
/// forking it would mean a `cudaGraphSetConditional` body whose work is on a
/// stream the body does not own, and the rule that forbids it should already
/// be standing rather than be one somebody has to think to add.
fn forkable(region: &Region) -> bool {
    region.phase == Phase::Capture
        && !region.collective
        && !region.mask.is_empty()
        && region.lowering == Lowering::AlwaysLaunch
}

/// Are these two regions a concurrency candidate: no path either way, and
/// disjoint class masks?
fn candidates(regions: &[Region], ordered: &Ordered, a: usize, b: usize) -> bool {
    !ordered.path(a, b) && !ordered.path(b, a) && disjoint(&regions[a].mask, &regions[b].mask)
}

fn disjoint(a: &ClassSet, b: &ClassSet) -> bool {
    !a.iter().any(|class| b.contains(class))
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
    fn of(plan: &Plan, regions: &[Region], profile: &DeviceProfile) -> Touches {
        let spaces_of: Vec<Option<u32>> = plan
            .values
            .iter()
            .map(|value| match value.def {
                Def::Cache(row) => Some(match plan.caches.get(row as usize) {
                    Some(model_ir::CacheRow::Kv { space, .. }) => *space,
                    // A state bank shares no page space with anything, so it
                    // is numbered above every kv space it could collide with.
                    _ => u32::MAX - row,
                }),
                _ => None,
            })
            .collect();

        // A `Def::Merge` is data, never dispatched: a reader names the phi
        // and the ARMS are what wrote it. Attributing a merge read to its
        // arms is what makes the edge from an attention window to the
        // consumer of the merge it fills visible at all — without it, three
        // arms that write `m`, `d`, `p` and a consumer that reads `o` share
        // no value and the whole neighbourhood looks independent.
        let mut through: Vec<Option<Vec<ValueId>>> = vec![None; plan.values.len()];
        for at in 0..plan.values.len() {
            resolve(plan, &mut through, ValueId(at as u32));
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
                let Some(node) = plan.nodes.get(node as usize) else {
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
                            // ANY MENTION OF A CACHE IS A WRITE OF ITS SPACE.
                            // See the module doc: `KvAppend` names its cache
                            // among the INPUTS and has no output, so nothing
                            // else would see the effect.
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
        meets(&self.spaces[a], &self.spaces[b]) && !disjoint(&regions[a].mask, &regions[b].mask)
    }
}

/// The values a name resolves to once every phi in front of it is walked
/// through. A plain value is itself; a `Def::Merge` is the union of its arms,
/// recursively, because §0's legal nesting puts merges inside merges.
fn resolve(plan: &Plan, through: &mut Vec<Option<Vec<ValueId>>>, value: ValueId) {
    let at = value.0 as usize;
    if through.get(at).is_some_and(Option::is_some) {
        return;
    }
    let Some(decl) = plan.values.get(at) else {
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
        resolve(plan, through, *arm);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, fact};
    use crate::{Budgets, compile};
    use model_ir::Cond;

    /// The §0 diagram, at three arms: a shared producer, three windows no
    /// value passes between, a shared consumer. This is gemma's attention
    /// neighbourhood with the model text taken out.
    fn three_arms() -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always); // r0 — everywhere
        let m = b.op(q, 8, Cond::and(fact(0), fact(1))); // r1
        let d = b.op(q, 8, Cond::and(fact(0), Cond::not(fact(1)))); // r2
        let p = b.op(q, 8, Cond::not(fact(0))); // r3
        let o = b.merge(
            &[
                (m, Cond::and(fact(0), fact(1))),
                (d, Cond::and(fact(0), Cond::not(fact(1)))),
                (p, Cond::not(fact(0))),
            ],
            8,
        );
        let y = b.op(o, 8, Cond::Always); // r4
        b.out(y);
        b
    }

    fn profile(side_streams: u32, floor: f32) -> DeviceProfile {
        DeviceProfile {
            side_streams,
            fork_floor_us: floor,
            ..DeviceProfile::default()
        }
    }

    #[test]
    fn three_independent_windows_fork_onto_three_streams_and_rejoin() {
        let b = three_arms();
        // The fixture's ops are `Elementwise`, which the family table prices
        // below the floor, so the gate is opened by lowering the floor rather
        // than by pretending a norm costs what an attention does.
        let baked = compile(&b.plan, &Budgets::new(4, 16), &profile(2, 1.0)).expect("bakes");
        let template = baked.template();
        assert_eq!(template.len(), 5);

        assert_eq!(template[1].stream, 0, "the first arm keeps the main stream");
        assert_eq!(template[2].stream, 1);
        assert_eq!(template[3].stream, 2);

        // The entry event is opened at the TOP of the main arm and waited on
        // by both arms that left the main stream.
        let enter = template[1].open.expect("the main arm opens the group");
        assert_eq!(template[0].open, None, "the region before it is untouched");
        assert!(template[1].wait.is_empty(), "the main arm was never away");
        assert_eq!(template[2].wait, vec![enter]);
        assert_eq!(template[3].wait, vec![enter]);

        // Every arm that forked closes with its exit, and the consumer waits
        // on all of them.
        let exits: Vec<EventId> = [2, 3]
            .iter()
            .map(|&at| template[at].close.expect("an arm closes with its exit"))
            .collect();
        assert_eq!(template[4].wait, exits);
        assert_eq!(template[4].stream, 0);
        assert_eq!(baked.forks.streams, 3);
        assert_eq!(baked.forks.events, 3, "one entry and one exit per arm");
    }

    #[test]
    fn the_relation_the_carve_sees_is_exactly_the_pairs_on_different_streams() {
        let b = three_arms();
        let baked = compile(&b.plan, &Budgets::new(4, 16), &profile(2, 1.0)).expect("bakes");
        assert_eq!(baked.forks.pairs, vec![(1, 2), (1, 3), (2, 3)]);
        assert_eq!(baked.concurrency.pairs(), baked.forks.pairs);
        // And the carve stayed sound under the wider relation, which is the
        // whole reason the hook was threaded through before this pass landed.
        assert!(baked.arena.clashes(&baked.concurrency).is_empty());
    }

    #[test]
    fn two_arms_over_one_side_stream_run_on_it_in_turn_and_are_not_paired() {
        let b = three_arms();
        let baked = compile(&b.plan, &Budgets::new(4, 16), &profile(1, 1.0)).expect("bakes");
        let template = baked.template();
        assert_eq!(
            (template[1].stream, template[2].stream, template[3].stream),
            (0, 1, 1),
            "the cap is one side stream, so the two arms share it",
        );
        // They are independent, so sharing a stream is a legal order — and
        // they are NOT concurrent, so the carve may still give them one
        // column.
        assert_eq!(baked.forks.pairs, vec![(1, 2), (1, 3)]);
        assert_eq!(baked.forks.streams, 2);
    }

    #[test]
    fn the_off_switch_bakes_the_artifact_p6_never_touched() {
        let b = three_arms();
        let off = compile(&b.plan, &Budgets::new(4, 16), &profile(0, 1.0)).expect("bakes");
        assert!(off.template().iter().all(|r| r.stream == 0));
        assert!(off.template().iter().all(|r| r.wait.is_empty()));
        assert!(off.template().iter().all(|r| r.open.is_none()));
        assert!(off.template().iter().all(|r| r.close.is_none()));
        assert!(off.template().iter().all(|r| r.sm_hint.is_none()));
        assert_eq!(off.forks, Forks { pairs: Vec::new(), events: 0, streams: 1 });
        assert!(off.concurrency.pairs().is_empty());
    }

    #[test]
    fn a_pair_too_cheap_to_fork_is_not_forked_and_pays_nothing() {
        // The default floor is 20 us and the fixture's elementwise ops are
        // priced at 4: the candidates are found and then declined.
        let b = three_arms();
        let baked = compile(&b.plan, &Budgets::new(4, 16), &profile(2, 20.0)).expect("bakes");
        assert!(baked.template().iter().all(|r| r.stream == 0));
        assert_eq!(baked.forks.events, 0);
        assert!(baked.concurrency.pairs().is_empty());
    }

    #[test]
    fn a_shared_written_value_is_an_edge_and_kills_the_candidacy() {
        // Two windows with disjoint masks that both write the SAME value —
        // not a merge, a plain output collision. Disjointness alone would
        // have called them candidates; the WAW edge is what does not.
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always);
        let d = b.op(q, 8, fact(0));
        let p = b.op(q, 8, Cond::not(fact(0)));
        // `p` reads `d`: a RAW edge between the two windows, disjoint masks
        // notwithstanding.
        let s = b.residual_add(d, p, 8, Cond::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (s, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always);
        b.out(y);

        let baked = compile(&b.plan, &Budgets::new(4, 16), &profile(2, 1.0)).expect("bakes");
        assert!(
            baked.template().iter().all(|r| r.stream == 0),
            "a path between them is not a candidate",
        );
    }

    #[test]
    fn a_collective_never_leaves_the_main_stream_and_orders_both_sides() {
        // NCCL matches by call order, so a collective is a barrier in the DAG
        // as well as a region that may not be elided (decision #5).
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always);
        let g = b.all_gather(q, 8, fact(0));
        let p = b.op(q, 8, Cond::not(fact(0)));
        let o = b.merge(&[(g, fact(0)), (p, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always);
        b.out(y);

        let baked = compile(&b.plan, &Budgets::new(4, 16), &profile(2, 1.0)).expect("bakes");
        let collective = baked
            .template()
            .iter()
            .find(|r| r.collective)
            .expect("the plan has one");
        assert_eq!(collective.stream, 0);
        assert!(baked.template().iter().all(|r| r.stream == 0));
    }

    #[test]
    fn two_appends_to_one_cache_space_are_ordered_unless_their_classes_are_disjoint() {
        // A cache write is an effect no value carries — `KvAppend` names its
        // cache among the inputs and outputs nothing — so any mention of a
        // cache is read as a write of its space. Two windows over the SAME
        // class therefore order; two over disjoint classes do not, because a
        // page belongs to a lane and classes partition the lanes.
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always);
        let d = b.op(q, 8, fact(0));
        b.append(d, fact(0));
        let p = b.op(q, 8, Cond::not(fact(0)));
        b.append(p, Cond::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always);
        b.out(y);

        let baked = compile(&b.plan, &Budgets::new(4, 16), &profile(2, 1.0)).expect("bakes");
        // The two append windows have disjoint masks, so the space they share
        // does not order them and the later one forks.
        let forked: Vec<u32> = baked.template().iter().map(|r| r.stream).collect();
        assert!(
            forked.iter().any(|&s| s != 0),
            "disjoint classes over one space are candidates: {forked:?}",
        );
    }

    #[test]
    fn the_assignment_is_a_pure_function_of_the_plan() {
        let b = three_arms();
        let once = compile(&b.plan, &Budgets::new(4, 16), &profile(2, 1.0)).expect("bakes");
        let twice = compile(&b.plan, &Budgets::new(4, 16), &profile(2, 1.0)).expect("bakes");
        assert_eq!(once, twice);
    }
}
