use super::ProcessId;
use std::collections::HashSet;

use crate::store::rs::RsUnits;

/// The slots of each window list a seated lane past its window holds beyond
/// its prompt's: the straddled slot and the one its next boundary takes
/// before the oldest is let go.
const WINDOW_GROWTH: u32 = 2;

/// The device's chunks by slot shape, each free and in all. In the
/// degenerate mode a chunk is one slot of a pinned shape: a kv page (the list
/// shape), a state slot, a buffer or window-ring page.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChunkStats {
    /// The kv pages', `(free, total)`.
    pub pages: (u32, u32),
    /// The state slots' and buffer (or ring) pages', `(free, total)`.
    pub rs: (RsUnits, RsUnits),
}

/// Physical chunks of a shared pool by what holds them
/// ([`PoolPort::chunk_occupants`]).
/// The runs of adjacent chunks a head's step lays, one entry per run
/// length, the longest first: `(run, pieces)`. Read as one length, a
/// restore whose unpark lays a run of two and fourteen single chunks was
/// asked sixteen single chunks, the map stood them, and the unpark was
/// refused the run of two (the L40S).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunNeed(Vec<(u32, u32)>);

impl RunNeed {
    /// `pieces` runs of `run` chunks.
    #[must_use]
    pub fn one(run: u32, pieces: u32) -> Self {
        let mut need = Self::default();
        need.add(run, pieces);
        need
    }

    /// Each `(run, pieces)` of `entries`, summed by length.
    #[must_use]
    pub fn of(entries: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let mut need = Self::default();
        for (run, pieces) in entries {
            need.add(run, pieces);
        }
        need
    }

    /// `pieces` more runs of `run` chunks.
    pub fn add(&mut self, run: u32, pieces: u32) {
        self.entry(run, |had| had.saturating_add(pieces), pieces);
    }

    /// At least `pieces` runs of `run` chunks.
    pub fn at_least(&mut self, run: u32, pieces: u32) {
        self.entry(run, |had| had.max(pieces), pieces);
    }

    fn entry(&mut self, run: u32, with: impl Fn(u32) -> u32, fresh: u32) {
        if fresh == 0 {
            return;
        }
        let run = run.max(1);
        match self.0.iter_mut().find(|(at, _)| *at == run) {
            Some((_, pieces)) => *pieces = with(*pieces),
            None => {
                self.0.push((run, fresh));
                self.0.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            }
        }
    }

    /// Its entries, the longest run first.
    #[must_use]
    pub fn entries(&self) -> &[(u32, u32)] {
        &self.0
    }

    /// Its longest runs' entry.
    #[must_use]
    pub fn longest(&self) -> Option<(u32, u32)> {
        self.0.first().copied()
    }

    /// The runs of `run` chunks it asks.
    #[must_use]
    pub fn pieces_at(&self, run: u32) -> u32 {
        self.0
            .iter()
            .find(|(at, _)| *at == run.max(1))
            .map_or(0, |&(_, pieces)| pieces)
    }

    /// The chunks it asks in all.
    #[must_use]
    pub fn chunks(&self) -> u64 {
        self.0
            .iter()
            .map(|&(run, pieces)| u64::from(run) * u64::from(pieces))
            .sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Never more chunks than `most` (the pool's state chunks), the longest
    /// runs kept first; `floor`'s entry (what a refusal named) kept whole.
    #[must_use]
    pub fn within(self, most: u32, floor: Option<(u32, u32)>) -> Self {
        let mut left = most;
        let mut need = Self::default();
        for (run, pieces) in self.0 {
            let pieces = pieces.min(left / run);
            left -= pieces * run;
            need.add(run, pieces);
        }
        if let Some((run, pieces)) = floor {
            need.at_least(run, pieces);
        }
        need
    }
}

/// A restore head's whole restore as the store's takes place it
/// ([`PoolPort::restore_placement`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RestorePlacement {
    /// The free physical chunks its own takes open: no room for what is
    /// still to lay ([`Self::short`]).
    pub opened: Vec<u32>,
    /// The budget's headroom its own takes leave; `None`: the map's own.
    pub room: Option<u32>,
    /// What does not stand once its own takes are placed: the pieces of
    /// its own the takes refused (runs of whole chunks as they are, the
    /// rest as the single chunks they open placed alone), then its pending
    /// bind's runs where fewer stand free than it asks. Empty: it stands.
    pub short: RunNeed,
    /// The free runs of the pending bind's length its own takes leave.
    pub bind_free_runs: Option<u32>,
}

impl RestorePlacement {
    /// Whether the whole restore stands now.
    #[must_use]
    pub fn stands(&self) -> bool {
        self.short.is_empty()
    }

    /// Read from `own` placed as `placement` went, `bind` laid in what it
    /// left; `alone` the chunks pieces open placed alone
    /// ([`crate::store::slots::SlotPool::opens_alone`]).
    #[must_use]
    pub fn of(
        own: &[crate::store::slots::OwnPiece],
        placement: &crate::store::slots::Placement,
        bind: &[(u32, u32)],
        alone: impl Fn(&[crate::store::slots::OwnPiece]) -> u32,
    ) -> Self {
        use crate::store::slots::OwnPiece;
        let rest = &own[placement.placed.min(own.len())..];
        let mut short = RunNeed::default();
        for piece in rest {
            if let OwnPiece::Chunks { run, pieces } = *piece {
                short.add(run, pieces);
            }
        }
        let singles: Vec<OwnPiece> = rest
            .iter()
            .filter(|piece| !matches!(piece, OwnPiece::Chunks { .. }))
            .copied()
            .collect();
        short.add(1, alone(&singles));
        // Its longest runs, and every chunk it asks among the free single
        // chunks: the runs of each length stand where both do.
        let bind_free_runs = bind.first().map(|&(run, _)| placement.free_runs(run));
        let bind_chunks: u32 = bind.iter().map(|&(run, pieces)| run.saturating_mul(pieces)).sum();
        let bind_short = bind.first().is_some_and(|&(run, pieces)| placement.free_runs(run) < pieces)
            || (!bind.is_empty() && placement.free_runs(1) < bind_chunks);
        if !bind.is_empty() && (!rest.is_empty() || bind_short) {
            for &(run, pieces) in bind {
                short.add(run, pieces);
            }
        }
        Self {
            opened: placement.opened.iter().copied().collect(),
            room: Some(placement.room),
            short,
            bind_free_runs,
        }
    }

    /// `map` with the chunks its own takes open taken out and the headroom
    /// they leave: the room what is short is laid in.
    pub fn lay(&self, map: &mut ChunkMap) {
        map.free.retain(|chunk| !self.opened.contains(chunk));
        if let Some(room) = self.room {
            map.headroom = Some(map.headroom.map_or(room, |had| had.min(room)));
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkMap {
    /// Each chunk some candidate's pages lie in: the slots it holds in
    /// all, and each candidate's pages there that would leave the device
    /// with it.
    pub held: Vec<ChunkHeld>,
    /// The chunks free now that state may take: open, below the activation
    /// the rung lends and the ceiling.
    pub free: Vec<u32>,
    /// How many chunks state may open now within the budget
    /// ([`crate::store::slots::SlotPool::headroom`]); `None`: no bound.
    /// A free chunk opened takes one; a chunk emptied gives one back.
    pub headroom: Option<u32>,
    /// The chunks a segment holds; a run lies inside one. `0`: no bound.
    pub segment_chunks: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkHeld {
    pub chunk: u32,
    pub used: u32,
    pub by: Vec<(ProcessId, u32)>,
}

impl ChunkHeld {
    /// Whether evicting its candidates empties it: every slot it holds is
    /// theirs to move (no cache entry's, no running set's but theirs).
    #[must_use]
    pub fn emptiable(&self) -> bool {
        self.by.iter().map(|&(_, pages)| pages).sum::<u32>() >= self.used
    }

    fn pages(&self) -> u32 {
        self.by.iter().map(|&(_, pages)| pages).sum()
    }
}

/// The `pieces` disjoint windows of `run` adjacent chunks, each from
/// `windows` (`(cost, first chunk)`), whose costs sum least, as `(cost,
/// first)`; `None` where no `pieces` of them are disjoint. A line of chunks
/// takes them exactly: the least cost of `k` windows within the chunks
/// below each position, built up position by position.
fn cheapest_disjoint(windows: &[(u32, u32)], run: u32, pieces: u32) -> Option<Vec<(u32, u32)>> {
    if pieces == 0 {
        return Some(Vec::new());
    }
    let cost: std::collections::HashMap<u32, u32> =
        windows.iter().map(|&(sum, first)| (first, sum)).collect();
    let end = windows.iter().map(|&(_, first)| first + run).max()? as usize;
    let run = run as usize;
    let pieces = pieces as usize;
    if pieces.saturating_mul(run) > end {
        return None;
    }
    // best[k][i]: the least cost of `k` windows inside chunks `0..i`.
    let mut best = vec![vec![u64::MAX; end + 1]; pieces + 1];
    best[0].fill(0);
    for k in 1..=pieces {
        for i in 1..=end {
            let mut least = best[k][i - 1];
            if i >= run
                && let Some(&sum) = cost.get(&((i - run) as u32))
                && best[k - 1][i - run] != u64::MAX
            {
                least = least.min(best[k - 1][i - run] + u64::from(sum));
            }
            best[k][i] = least;
        }
    }
    if best[pieces][end] == u64::MAX {
        return None;
    }
    let mut taken = Vec::with_capacity(pieces);
    let (mut k, mut i) = (pieces, end);
    while k > 0 {
        if best[k][i] == best[k][i - 1] {
            i -= 1;
            continue;
        }
        let first = (i - run) as u32;
        taken.push((cost[&first], first));
        i -= run;
        k -= 1;
    }
    Some(taken)
}

impl ChunkMap {
    /// The map as read for `pids` alone: each chunk's holders among them,
    /// a chunk none of them holds left out. One map built for every
    /// process serves every candidate list in a generation.
    #[must_use]
    pub fn restricted(&self, pids: &[ProcessId]) -> ChunkMap {
        let wanted: std::collections::HashSet<ProcessId> = pids.iter().copied().collect();
        ChunkMap {
            held: self
                .held
                .iter()
                .filter_map(|chunk| {
                    let by: Vec<(ProcessId, u32)> =
                        chunk.by.iter().copied().filter(|(pid, _)| wanted.contains(pid)).collect();
                    (!by.is_empty()).then(|| ChunkHeld {
                        chunk: chunk.chunk,
                        used: chunk.used,
                        by,
                    })
                })
                .collect(),
            free: self.free.clone(),
            headroom: self.headroom,
            segment_chunks: self.segment_chunks,
        }
    }

    /// `other`'s chunks counted in as well: a chunk both hold has the
    /// occupants of each, a pid in both its slots summed.
    pub fn merge(&mut self, other: ChunkMap) {
        for chunk in other.held {
            match self.held.iter_mut().find(|held| held.chunk == chunk.chunk) {
                Some(held) => {
                    held.used = held.used.max(chunk.used);
                    for (pid, slots) in chunk.by {
                        match held.by.iter_mut().find(|(by, _)| *by == pid) {
                            Some((_, count)) => *count += slots,
                            None => held.by.push((pid, slots)),
                        }
                    }
                }
                None => self.held.push(chunk),
            }
        }
        self.held.sort_unstable_by_key(|held| held.chunk);
        self.free.extend(other.free);
        self.free.sort_unstable();
        self.free.dedup();
        self.headroom = match (self.headroom, other.headroom) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.segment_chunks = self.segment_chunks.max(other.segment_chunks);
    }

    /// Why runs of `run` adjacent chunks do or do not stand in this map,
    /// in one line: free chunks, held ones emptiable by the candidates and
    /// not, the headroom, the segment's length, the longest stretch of
    /// free-or-emptiable chunks inside one segment, and the cheapest
    /// window's cost.
    #[must_use]
    pub fn explain(&self, run: u32) -> String {
        let run = run.max(1);
        let free: std::collections::BTreeSet<u32> = self.free.iter().copied().collect();
        let emptiable: std::collections::BTreeSet<u32> = self
            .held
            .iter()
            .filter(|chunk| chunk.emptiable())
            .map(|chunk| chunk.chunk)
            .collect();
        let stuck = self.held.len() - emptiable.len();
        let candidates: std::collections::BTreeSet<ProcessId> = self
            .held
            .iter()
            .flat_map(|chunk| chunk.by.iter().map(|&(pid, _)| pid))
            .collect();
        let (mut longest, mut stretch, mut last) = (0u32, 0u32, None::<u32>);
        for &chunk in free.union(&emptiable) {
            let same = self.segment_chunks == 0
                || last.is_some_and(|last| last / self.segment_chunks == chunk / self.segment_chunks);
            stretch = if last == Some(chunk.wrapping_sub(1)) && same { stretch + 1 } else { 1 };
            longest = longest.max(stretch);
            last = Some(chunk);
        }
        // Free chunks alone: the longest stretch inside one segment, the
        // disjoint runs of `run` they hold, and how many of those a stretch
        // crossing a segment's edge breaks (free chunks enough, no run).
        let (mut longest_free, mut free_runs, mut cut_at_edge) = (0u32, 0u32, 0u32);
        let (mut stretch, mut last) = (0u32, None::<u32>);
        for &chunk in &free {
            let adjacent = last == Some(chunk.wrapping_sub(1));
            let same = self.segment_chunks == 0
                || last.is_some_and(|last| last / self.segment_chunks == chunk / self.segment_chunks);
            if adjacent && !same {
                cut_at_edge += 1;
            }
            if adjacent && same {
                stretch += 1;
            } else {
                free_runs += stretch / run;
                stretch = 1;
            }
            longest_free = longest_free.max(stretch);
            last = Some(chunk);
        }
        free_runs += stretch / run;
        // Of those, what the state budget's headroom lets be opened: the
        // pool's own take reads runs so (`SlotPool::chunk_runs_for`). Runs
        // standing open past it are no room until chunks elsewhere empty.
        let usable_runs = self
            .headroom
            .map_or(free_runs, |headroom| free_runs.min(headroom / run));
        // What opening those runs takes of the budget's headroom.
        let headroom_short = self
            .headroom
            .is_some_and(|headroom| headroom < run);
        format!(
            "map (scope: chunks a state shape may open, and chunks holding eviction \
             candidates or parked state): run={run} open_for_state={} \
             candidate_or_parked_held={} emptiable={} stuck={stuck} candidates={} \
             headroom={:?} segment_chunks={} longest_usable_stretch={longest} \
             longest_free_stretch={longest_free} free_runs={free_runs} \
             runs_within_headroom={usable_runs} \
             free_stretches_cut_at_segment_edges={cut_at_edge} \
             headroom_short_for_one_run={headroom_short} \
             cheapest_one_run_moves={:?}",
            free.len(),
            self.held.len(),
            emptiable.len(),
            candidates.len(),
            self.headroom,
            self.segment_chunks,
            self.cheapest_runs(run, 1).map(|pids| pids.len()),
        )
    }

    /// The disjoint free runs of `run` adjacent chunks it holds, each inside
    /// one segment, no more than its headroom opens: the map's figure for
    /// the room a refusal read at that run ([`super::ResidencyPlanner::dump_stall`]).
    #[must_use]
    pub fn free_runs(&self, run: u32) -> u32 {
        let run = run.max(1);
        let mut free = self.free.clone();
        free.sort_unstable();
        let (mut runs, mut stretch, mut last) = (0u32, 0u32, None::<u32>);
        for chunk in free {
            let same = self.segment_chunks == 0
                || last.is_some_and(|last| last / self.segment_chunks == chunk / self.segment_chunks);
            stretch = if last == Some(chunk.wrapping_sub(1)) && same { stretch + 1 } else { 1 };
            if stretch == run {
                runs += 1;
                stretch = 0;
            }
            last = Some(chunk);
        }
        self.headroom.map_or(runs, |headroom| runs.min(headroom / run))
    }

    /// The processes whose eviction empties the cheapest `pieces` disjoint
    /// runs of `run` adjacent chunks, each chunk free or emptiable
    /// ([`ChunkHeld::emptiable`]), cost the pages moved; `None` where no
    /// such runs stand.
    #[must_use]
    pub fn cheapest_runs(&self, run: u32, pieces: u32) -> Option<Vec<ProcessId>> {
        self.cheapest_need(&RunNeed::one(run, pieces))
    }

    /// As [`Self::cheapest_runs`], for every entry of `need` at once: its
    /// longest runs laid first, each shorter length in the chunks left
    /// beside them, and the budget's headroom read against every chunk
    /// they open together. Asked one length at a time, each fit the
    /// headroom alone and runs of two were laid where single chunks had
    /// been counted.
    #[must_use]
    pub fn cheapest_need(&self, need: &RunNeed) -> Option<Vec<ProcessId>> {
        let free: std::collections::BTreeSet<u32> = self.free.iter().copied().collect();
        let held: std::collections::BTreeMap<u32, &ChunkHeld> = self
            .held
            .iter()
            .filter(|chunk| chunk.emptiable())
            .map(|chunk| (chunk.chunk, chunk))
            .collect();
        let cost = |chunk: u32| -> Option<u32> {
            if free.contains(&chunk) {
                Some(0)
            } else {
                held.get(&chunk).map(|held| held.pages())
            }
        };
        let starts: std::collections::BTreeSet<u32> =
            free.iter().chain(held.keys()).copied().collect();
        let mut in_runs: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for &(run, pieces) in need.entries() {
            let run = run.max(1);
            let one_segment = |first: u32| {
                self.segment_chunks == 0
                    || first / self.segment_chunks == (first + run - 1) / self.segment_chunks
            };
            // Every window of `run` adjacent chunks each free or emptiable
            // and not laid for a longer run, by its cost; then the cheapest
            // disjoint ones.
            let windows: Vec<(u32, u32)> = starts
                .iter()
                .filter(|&&first| one_segment(first))
                .filter_map(|&first| {
                    (first..first + run)
                        .map(|chunk| if in_runs.contains(&chunk) { None } else { cost(chunk) })
                        .try_fold(0u32, |sum, cost| cost.map(|cost| sum + cost))
                        .map(|sum| (sum, first))
                })
                .collect();
            // The cheapest `pieces` disjoint windows, chosen exactly: taking the
            // cheapest window first could leave no room for the rest where a
            // costlier set of them fits (free runs at 1-3 and 6-8 of eleven
            // chunks, a third run asked: the two free ones taken, nothing fit
            // beside them, and no one was moved, though moving one occupant
            // stood three runs at 0, 3 and 6).
            let taken = cheapest_disjoint(&windows, run, pieces)?;
            in_runs.extend(taken.iter().flat_map(|&(_, first)| first..first + run));
        }
        // The free chunks the runs open take the budget's headroom; past
        // it, each needs a chunk emptied elsewhere (its slots leave, and
        // the budget it held comes back). Counted as costless, a free run
        // the budget could not open was waited on with no one moved.
        let opened = in_runs.iter().filter(|chunk| free.contains(chunk)).count() as u32;
        let short = opened.saturating_sub(self.headroom.unwrap_or(u32::MAX));
        let mut elsewhere: Vec<&ChunkHeld> = held
            .values()
            .filter(|chunk| !in_runs.contains(&chunk.chunk))
            .copied()
            .collect();
        elsewhere.sort_unstable_by_key(|chunk| (chunk.pages(), chunk.chunk));
        if (elsewhere.len() as u32) < short {
            return None;
        }
        let mut pids: Vec<ProcessId> = in_runs
            .iter()
            .filter_map(|chunk| held.get(chunk).copied())
            .chain(elsewhere.into_iter().take(short as usize))
            .flat_map(|held| held.by.iter().map(|&(pid, _)| pid))
            .collect();
        pids.sort_unstable();
        pids.dedup();
        Some(pids)
    }
}

/// Where an activation raise to a rung stands against the pool's room
/// ([`PoolPort::rise_room`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiseRoom {
    /// It fits now (or nothing is left to raise): the raise is taken.
    Fits,
    /// Short of room: `pieces` free chunks are what it takes, those free
    /// now and those to empty, as a bind's runs of one chunk.
    Short { pieces: u32 },
    /// The rung's span and the reserved chunks, `need` chunks, past the
    /// pool's `total`: no move of state ever makes the room.
    Past { need: u32, total: u32 },
}

/// Every tier's room and what is held in it, in one unit, kv pages
/// ([`PoolPort::capacity_census`]). Each item of state counts one figure
/// wherever it lies: a kv page one (the free slots of partly used list
/// chunks are room: a list compaction frees them whole,
/// [`crate::store::kv::KvStore::plan_list_compaction`]), guest state and
/// tables their bytes as pages ([`crate::store::roles::Roles::footprint`]),
/// recurrent state its units as pages. So a move between tiers changes no
/// count of state; only new state raises `held`, and only state given back
/// lowers it. Beside it, `slack`: the room the guest shapes hold open that
/// no state fills (their chunks' unused slots, the disk chunks parked parts
/// take past their bytes), room no other shape takes until it empties, so
/// not free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapacityCensus {
    /// Every tier's chunks for state, as pages: the device's under its
    /// current rung, the host tier's and the disk tier's.
    pub whole: u64,
    /// What every tier holds, each item once.
    pub held: u64,
    /// Room held open that no state fills: not free, and no state.
    pub slack: u64,
    /// Of `held`, what only idle cache holds: free to the gate (the
    /// policy lets it go under pressure).
    pub idle: u64,
    /// Of `held`, the second slots moves under way hold: room taken until
    /// they land, though no state was added.
    pub moving: u64,
}

pub trait PoolPort: Send + Sync + 'static {
    /// The device's chunks by shape.
    fn chunk_stats(&self) -> ChunkStats;
    fn device_stats(&self) -> (u32, u32) {
        self.chunk_stats().pages
    }
    /// Device pages handed out. A target below the pool is room the target
    /// can rise into, so this is not `device_stats`' complement there.
    fn device_out(&self) -> u32 {
        let (free, total) = self.device_stats();
        total.saturating_sub(free)
    }
    fn host_stats(&self) -> (u32, u32);
    fn disk_stats(&self) -> (u32, u32);
    fn rs_stats(&self) -> (RsUnits, RsUnits) {
        self.chunk_stats().rs
    }
    fn rs_host_stats(&self) -> (RsUnits, RsUnits);
    /// Drops idle cache, but only what its retention key lets the runtime
    /// drop unasked (`pressure::may_drop`); room past that is asked of the
    /// policy with tier pressure.
    fn reclaim_idle(&self) -> u32;
    /// Called until it frees nothing, without freeing them.
    fn reclaimable_idle(&self) -> u32;
    /// The host and disk slots only idle cache holds, `(host, disk)`: below
    /// the device what the next demand may take by dropping them, so the
    /// pressure report counts them as room, as [`Self::reclaimable_idle`]
    /// is on the device. Every port states it; a pool with no tier below
    /// the device states `(0, 0)`.
    fn idle_below_device(&self) -> (u32, u32);
    fn suspend_capable(&self) -> bool;
    fn locus(&self) -> (usize, usize);
    /// The rs units a seated lane may still claim as it decodes: a window
    /// ring's page boundary takes a buffer page before the page behind the
    /// window is let go. Fixed state slots never grow.
    fn rs_growth(&self) -> RsUnits {
        RsUnits::default()
    }
    /// Room was freed or a fire retired: what waits on it beside the
    /// planner's own waiters (the rung, the segment walk) looks
    /// again.
    fn freed(&self) {}
    /// Asks for `need` more of each shape than stands free now; true when
    /// memory is on its way: `pages_freed` follows it, or, where the walk
    /// maps nothing after all, the planner's `no_grow`.
    fn grow(&self, need: super::Demand) -> bool {
        let _ = need;
        false
    }
    fn grow_rs(&self, need: RsUnits) -> bool {
        self.grow(super::Demand {
            kv_pages: 0,
            rs: need,
        })
    }
    /// The device pages that, parked whole, make room for `rs` of the state
    /// shapes: a shared pool opens any chunk for any shape, so a state
    /// shortfall is met by moving kv out. 0 where state has pools of its
    /// own, which no kv move frees.
    fn rs_as_pages(&self, rs: RsUnits) -> u32 {
        let _ = rs;
        0
    }
    /// Whether `pid` holds device room the registry's page and state counts
    /// do not see: guest state (its instances' scratch, parked role slots
    /// lying on the device). Moving it frees chunks, so a process holding
    /// only that is still a holder: the starvation endgame's "alone holds
    /// what the pool holds" ([`super::ResidencyPlanner::sole_holder`]) is
    /// false while one does. None where the pool keeps no guest state.
    fn holds_device_room(&self, pid: ProcessId) -> bool {
        let _ = pid;
        false
    }
    /// Whether `pieces` free runs of `run` adjacent chunks stand for a
    /// guest's state now: what a bind short of room waits on
    /// ([`super::wait::WaitKind::BindRoom`]). None where the pool keeps
    /// no guest state.
    fn guest_room_free(&self, run: u32, pieces: u32) -> bool {
        let _ = (run, pieces);
        false
    }
    /// The free runs of `run` adjacent chunks a guest's state could take
    /// now ([`Self::guest_room_free`] counted): what a restore head short of
    /// them watches change.
    fn guest_free_runs(&self, run: u32) -> u32 {
        u32::from(self.guest_room_free(run, 1))
    }
    /// Whether every entry of `need` stands free together once `state` (a
    /// restore's state, taken by its serve before its unpark) is placed, as
    /// the takes lay them: the state's single slots where the store puts
    /// them, then the longest runs first, each shorter length beside them,
    /// the budget's headroom spent by all of them. The store's own
    /// placement ([`crate::store::slots::SlotPool::chunk_runs_beside`]), so
    /// what stands here is what the unpark finds. Where the pool lays
    /// nothing apart, each entry read on its own.
    fn guest_need_free(&self, need: &RunNeed, state: RsUnits) -> bool {
        let _ = state;
        need.entries()
            .iter()
            .all(|&(run, pieces)| self.guest_room_free(run, pieces))
    }
    /// `pid`'s whole restore placed by the store's takes themselves: its
    /// `kv_pages` still to gather and `state` its serve takes, every piece
    /// its unpark takes, in that order, then `bind` (a pending bind's runs)
    /// in what they leave ([`crate::store::slots::SlotPool::place`]). The
    /// one placement a restore is sent on ([`Self::restore_stands`]), its
    /// moves are planned by (the map lays [`RestorePlacement::short`] in
    /// the room its own takes leave) and the stall line reads: a check
    /// that mirrored the takes stood runs the restore's own takes then
    /// broke (Mac Pro at 2 GiB: PendingBindShort, free runs (3, 2) before
    /// and one after, need_moves 0). Where the pool keeps no roles, `need`
    /// (the runs the planner reads its step as) is placed beside `state`.
    fn restore_placement(
        &self,
        pid: ProcessId,
        need: &RunNeed,
        state: RsUnits,
        kv_pages: u32,
        bind: &[(u32, u32)],
    ) -> RestorePlacement {
        let _ = (pid, kv_pages);
        let stands = if bind.is_empty() {
            self.guest_need_free(need, state)
        } else {
            bind.iter().all(|&(run, pieces)| self.room_beside_own(pid, run, pieces))
        };
        RestorePlacement {
            opened: self.serve_opens(state),
            room: None,
            short: if stands {
                RunNeed::default()
            } else if bind.is_empty() {
                need.clone()
            } else {
                RunNeed::of(bind.iter().copied())
            },
            bind_free_runs: None,
        }
    }
    /// Whether `pid`'s whole restore stands now ([`Self::restore_placement`]).
    fn restore_stands(
        &self,
        pid: ProcessId,
        need: &RunNeed,
        state: RsUnits,
        kv_pages: u32,
        bind: &[(u32, u32)],
    ) -> bool {
        self.restore_placement(pid, need, state, kv_pages, bind).stands()
    }
    /// Holds for `pid`'s served restore what its placement chose: the free
    /// chunks its unpark's pieces and then `bind`'s runs open
    /// ([`crate::store::roles::Roles::reserve_restore`]), closed to every
    /// other take until its unpark and bind take them or
    /// [`Self::release_restore`]. Whether it was held (its placement
    /// stands). Where the pool keeps no roles, nothing is held.
    fn reserve_restore(&self, pid: ProcessId, bind: &[(u32, u32)]) -> bool {
        let _ = (pid, bind);
        false
    }
    /// `pid`'s reservation goes back: its restore failed, was queued again,
    /// was withdrawn, landed with no bind pending, or its process is gone.
    fn release_restore(&self, pid: ProcessId) {
        let _ = pid;
    }
    /// The physical device chunks slot refs `refs` of engine shape `shape`
    /// lie in: what a residency copy of them reads or writes, fenced
    /// ([`Self::moving_out`]) until it lands. Empty where the pool keeps
    /// no chunks.
    fn copy_chunks(&self, shape: ::engine::chunk::ShapeId, refs: &[u32]) -> Vec<u32> {
        let _ = (shape, refs);
        Vec::new()
    }
    /// A residency move's copy reads `chunks` (physical) until
    /// [`Self::moved_out`]: no raise takes them back for its activation
    /// meanwhile ([`crate::store::slots::SlotPool::moving_out`]).
    fn moving_out(&self, chunks: &[u32]) {
        let _ = chunks;
    }
    /// The copy [`Self::moving_out`] named has landed, or will not; the
    /// walk looks again (a raise that waited on it).
    fn moved_out(&self, chunks: &[u32]) {
        let _ = chunks;
    }
    /// The free physical chunks a restore's `state` would open as its serve
    /// takes it, lowest first after the shapes' partly filled chunks
    /// ([`crate::store::slots::SlotPool::opens_for`]): chunks the map must
    /// not lay the head's runs in. Empty where the pool keeps no chunks.
    fn serve_opens(&self, state: RsUnits) -> Vec<u32> {
        let _ = state;
        Vec::new()
    }
    /// Where a raise of the activation to `rung` (tokens) stands now: what
    /// a rung raise short of room waits on
    /// ([`super::ResidencyPlanner::fund_rise`]). `Fits` where the pool has
    /// no rungs.
    fn rise_room(&self, rung: u32) -> RiseRoom {
        let _ = rung;
        RiseRoom::Fits
    }
    /// Whether `pieces` free runs of `run` adjacent chunks stand beside
    /// the guest state `pid`'s restore retakes first (its parked dead
    /// parts, [`crate::store::roles::Roles::fits_beside_unpark`]): room for
    /// the bind that suspended it, past its own (the M1 freeze: restored
    /// straight back into the room its park gave, short again, 750 times
    /// a second). Where the pool keeps no guest state, the runs alone.
    fn room_beside_own(&self, pid: ProcessId, run: u32, pieces: u32) -> bool {
        let _ = pid;
        self.guest_room_free(run, pieces)
    }
    /// Whether moving chunks will stand `pieces` free runs of `run`
    /// adjacent chunks where free chunks are enough but scattered (F21): a
    /// compaction planned, the pool's walk asked to carry it out, or one on
    /// its way. Where it is, no resident is moved for the bind; where no
    /// moves could stand the runs, `false`. None where the pool keeps no
    /// guest state.
    fn compaction_comes(&self, run: u32, pieces: u32) -> bool {
        let _ = (run, pieces);
        false
    }
    /// Whether a compaction asked for a head is on its way: its moves
    /// fenced, or planned and still plannable for runs that do not stand
    /// yet ([`crate::store::shared::SharedPool::compaction_under_way`]).
    /// It moves the head's room as an eviction in flight does.
    fn compaction_under_way(&self) -> bool {
        false
    }
    /// The pool's census for a head short of runs of `run` chunks
    /// ([`crate::store::slots::SlotPool::census`]); empty where the pool
    /// keeps no guest state.
    fn chunk_census(&self, run: u32) -> String {
        let _ = run;
        String::new()
    }
    /// What `pid`'s restore would take back and where the pool would put
    /// it ([`crate::store::roles::Roles::unpark_census`]), for `need`;
    /// empty where the pool keeps no roles.
    fn unpark_census(&self, pid: ProcessId, need: Option<(u32, u32)>) -> String {
        let _ = (pid, need);
        String::new()
    }
    /// Where the occupancy map's attribution of `pid` comes from, by
    /// source and physical chunk: its movable kv pages, its guest items
    /// (each item's owner, tier, parts and the chunks they lie in), and
    /// its state's private device slots with each chunk's slots in use.
    /// What a quiet wait's warn sets beside the map's attribution.
    fn occupancy_sources(&self, pid: ProcessId) -> String {
        let _ = pid;
        String::new()
    }
    /// The device pages that, moved out whole, free `chunks` chunks for a
    /// guest's state, as [`Self::rs_as_pages`] for state.
    fn chunks_as_pages(&self, chunks: u32) -> u32 {
        let _ = chunks;
        0
    }
    /// Which of `pids` hold device kv pages in which physical chunk of a
    /// shared pool, and the chunks free now ([`ChunkMap`]): what a bind
    /// short of whole chunks finds emptiable by evicting them. Empty where
    /// the pool keeps no chunks.
    fn chunk_occupants(&self, pids: &[ProcessId]) -> ChunkMap {
        let _ = pids;
        ChunkMap::default()
    }
    /// Which processes' parked state lies on the device in which physical
    /// chunk of a shared pool (role slots a park found no room below for,
    /// [`crate::store::roles::Roles::parked_chunk_slots`]), `except`'s left
    /// out, and the chunks free now: what moving that state down a tier
    /// ([`Self::move_parked_down`]) empties. Empty where the pool keeps no
    /// chunks or nothing parked lies on the device.
    fn parked_occupants(&self, except: ProcessId) -> ChunkMap {
        let _ = except;
        ChunkMap::default()
    }
    /// Moves the parked state `pids` left on the device down a tier, as the
    /// load's tiers lie: to the host where it has room, and where it has
    /// none kv on the host goes on down to disk first to make it (idle
    /// cache, then the parked sets of `order`, the victim order), the disk
    /// taking it only once no kv makes host room; to the disk at once
    /// where the load states no host tier (unified memory). True when a move is on its way, whose landing tells
    /// `planner` ([`ResidencyPlanner::parked_down_done`]); false when
    /// nothing can move.
    ///
    /// [`ResidencyPlanner::parked_down_done`]: super::ResidencyPlanner::parked_down_done
    fn move_parked_down(
        &self,
        planner: &std::sync::Arc<super::ResidencyPlanner>,
        pids: &[ProcessId],
        order: &[ProcessId],
    ) -> bool {
        let _ = (planner, pids, order);
        false
    }
    /// Has `signal` called each time device slots come free, whatever path
    /// gives them back (`SlotPool::set_room_signal`): how the planner hears
    /// of room it did not free itself. None where the pool keeps no slots.
    fn set_room_signal(&self, signal: crate::store::slots::RoomSignal) {
        let _ = signal;
    }
    /// Free chunks of the device pool a state shape may open now: what a
    /// head's runs are weighed against before the planner is woken.
    fn free_chunks(&self) -> u32 {
        self.chunk_stats().pages.0
    }
    /// The runs of adjacent chunks `pid`'s restore lays for its own parked
    /// guest state before a bind's room, one entry per run length
    /// ([`crate::store::roles::Roles::unpark_need`]): what the chunk map
    /// must stand beside the runs of the bind that suspended it. Empty
    /// where the pool keeps no guest state.
    fn unpark_need(&self, pid: ProcessId) -> RunNeed {
        let _ = pid;
        RunNeed::default()
    }
    /// The chunks the device pool holds for kv and state at most (its
    /// chunks past the reserved and the activation's): the most runs of
    /// any length a need can ask. `u32::MAX` where the pool keeps no
    /// chunks.
    fn state_chunks(&self) -> u32 {
        u32::MAX
    }
    /// [`crate::store::shared::SharedPool::state_chunks_most`]: the one
    /// device figure a never-fits check reads. `u32::MAX` where the kv
    /// pages are a pool of their own.
    fn state_chunks_most(&self) -> u32 {
        u32::MAX
    }
    /// Every tier's room and what is held in it, in one unit, kv pages,
    /// read now ([`CapacityCensus`]): what the capacity gate is read from
    /// ([`super::inner::CapacityGate`]). `None` where the pool keeps no
    /// chunks (the tests' own pools).
    fn capacity_census(&self) -> Option<CapacityCensus> {
        None
    }
    /// The device's list chunks a list compaction would empty for
    /// `wanted` whole chunks, the processes `may_fence` names fenced for it
    /// ([`crate::store::kv::KvStore::plan_list_compaction`]); `None` where
    /// none empties, or the pool keeps no chunks.
    fn plan_list_compaction(
        &self,
        wanted: u32,
        may_move: crate::store::kv::MayMove<'_>,
    ) -> Option<crate::store::kv::ListCompaction> {
        let _ = (wanted, may_move);
        None
    }
    /// The guest state and tables `pid`'s binds took, as pages counted as
    /// [`CapacityCensus::held`] counts them; 0 where the pool keeps none.
    fn guest_pages_of(&self, pid: ProcessId) -> u64 {
        let _ = pid;
        0
    }
    /// The kv pages one process may hold at most: the declared context
    /// served ([`crate::store::registry::Stores::context_pages`]), the
    /// capacity gate's end for a process that states none
    /// ([`super::inner::CapacityGate`]). `None`: no gate (a pool of no
    /// tiers' facts, as the tests' own).
    fn context_pages(&self) -> Option<u32> {
        None
    }
    /// The chunks the governor's target holds back from state
    /// ([`crate::store::shared::SharedPool::held_back_chunks`]): room only
    /// the target's rise gives. 0 where none is held back.
    fn held_back(&self) -> u32 {
        0
    }
    /// The device pool's count of room given back (a chunk opened for
    /// state, [`crate::store::slots::SlotPool::room_gives`]): moved since
    /// a read, room has come free since.
    fn room_gives(&self) -> u64 {
        0
    }
    /// The device pool's count of every change to what it holds: room
    /// given back ([`Self::room_gives`]), and each take, give and move of
    /// slots. Unmoved, every placement and map read on it reads the same:
    /// part of the planner's generation ([`super::ResidencyPlanner::generation`]).
    /// Where the pool keeps no chunks, its gives and its free pages.
    fn room_changes(&self) -> u64 {
        self.room_gives()
            .wrapping_add(u64::from(self.device_stats().0) << 32)
    }
    /// How a shared pool's chunks divide for good: what the weights, the
    /// activation and the reserved chunks hold, and what is left for kv and
    /// state. `None` where the kv pages are a pool of their own.
    fn room(&self) -> Option<crate::store::shared::PoolRoom> {
        None
    }
    fn reserve_device(
        &self,
        count: u32,
    ) -> Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>>;
    fn reserve_device_up_to(
        &self,
        count: u32,
    ) -> Vec<crate::store::kv::page_table::PhysicalKvPageId>;
    fn release_device(&self, pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>);
    /// Device pages the planner holds unwritten (gathered for a head),
    /// given back and taken again at once, the fullest partly filled list
    /// chunks first ([`crate::store::kv::KvStore::repack_reserved`]): no
    /// copy, and the chunks only they held stand free whole. As they were
    /// where the pool keeps no chunks.
    fn repack_device(
        &self,
        pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>,
    ) -> Vec<crate::store::kv::page_table::PhysicalKvPageId> {
        pages
    }
    fn reserve_rs(&self, need: RsUnits) -> Option<crate::store::rs::RsGrant>;
    fn release_rs(&self, grant: crate::store::rs::RsGrant);
    /// Drops the checkpoints sharing state with `working_sets` that their
    /// retention keys let go unasked, so the sets' next write lands in
    /// place; returns how many were dropped.
    fn reclaim_sharing(&self, working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize;
    fn prepare_sweep(
        &self,
        keep: &HashSet<crate::store::kv::page_table::WorkingSetId>,
        room: usize,
        order: &[(crate::store::kv::page_table::WorkingSetId, u64)],
        wanted: usize,
    ) -> Option<crate::store::kv::KvResidencyTxn>;
    fn prepare_demote(&self, room: usize) -> Option<crate::store::kv::KvResidencyTxn>;
    /// [`Self::prepare_sweep`] of the tails past the heads `order` names
    /// alone: those heads stay however few pages move.
    fn prepare_tail_sweep(
        &self,
        keep: &HashSet<crate::store::kv::page_table::WorkingSetId>,
        room: usize,
        order: &[(crate::store::kv::page_table::WorkingSetId, u64)],
        wanted: usize,
    ) -> Option<crate::store::kv::KvResidencyTxn> {
        let _ = (keep, room, order, wanted);
        None
    }
    /// Idle cache parked on the host, up to `room` pages, going on down to
    /// disk, lowest key first. `None` where nothing goes.
    fn prepare_spill(&self, room: usize) -> Option<crate::store::kv::KvSpillTxn> {
        let _ = room;
        None
    }
}

pub struct RegistryPool {
    model: usize,
    engine: usize,
    suspend_capable: bool,
    /// The last count of reclaimable idle pages and when it was taken.
    reclaimable: parking_lot::Mutex<Option<(crate::rt::Instant, u32)>>,
    /// A shared pool's segment walker: what waiters are short reaches its
    /// demand, so it commits segments ahead of them.
    shared: Option<crate::store::shared::SharedGrow>,
}

/// How long a count of reclaimable idle pages stands: counting walks every
/// cached entry under the kv lock, and the pressure report asks every
/// 100 ms, so a full cache would hold the lock a fire waits on.
const RECLAIMABLE_FRESH: std::time::Duration = std::time::Duration::from_secs(1);

impl RegistryPool {
    pub fn new(model: usize, engine: usize, suspend_capable: bool) -> Self {
        Self {
            model,
            engine,
            suspend_capable,
            reclaimable: parking_lot::Mutex::new(None),
            shared: None,
        }
    }

    /// Grows a shared pool's segments for the pages waiters are short.
    pub fn with_shared_grow(mut self, grow: crate::store::shared::SharedGrow) -> Self {
        self.shared = Some(grow);
        self
    }

    fn with_kv_tagged<R>(
        &self,
        tag: &'static str,
        operation: impl FnOnce(&mut crate::store::kv::KvStore) -> R,
    ) -> R {
        let stores = crate::store::registry::get(self.model, self.engine);
        crate::store::registry::with_kv_lock(&stores.kv, tag, operation)
    }

    fn with_rs<R>(&self, operation: impl FnOnce(&mut crate::store::rs::RsStore) -> R) -> R {
        let stores = crate::store::registry::get(self.model, self.engine);
        let mut store = stores.rs.lock().unwrap();
        operation(&mut store)
    }
}

impl PoolPort for RegistryPool {
    fn chunk_stats(&self) -> ChunkStats {
        ChunkStats {
            pages: self.device_stats(),
            rs: self.with_rs(|rs| (rs.free(), rs.capacity())),
        }
    }

    /// On a shared pool the kv pages' total is what is left once the
    /// weights' pinned chunks, the activation and the reserved chunks are
    /// set aside: the list shape's own capacity counts every state chunk,
    /// those the weights hold for good included.
    fn device_stats(&self) -> (u32, u32) {
        let (free, capacity) = self.with_kv_tagged("planner-device-stats", |kv| {
            (kv.available_pages() as u32, kv.capacity_pages())
        });
        let total = self
            .room()
            .map_or(capacity, |room| capacity.min(room.list_slots));
        (free, total)
    }

    fn room(&self) -> Option<crate::store::shared::PoolRoom> {
        let stores = crate::store::registry::try_get(self.model, self.engine)?;
        let room = stores.shared.as_ref()?.lock().unwrap().room();
        (room.list_tokens > 0).then_some(room)
    }

    fn device_out(&self) -> u32 {
        self.with_kv_tagged("planner-device-out", |kv| kv.pages_out())
    }

    fn host_stats(&self) -> (u32, u32) {
        self.with_kv_tagged("planner-host-stats", |kv| {
            (kv.host_swap_available() as u32, kv.host_swap_capacity())
        })
    }

    fn disk_stats(&self) -> (u32, u32) {
        self.with_kv_tagged("planner-disk-stats", |kv| {
            (kv.disk_swap_available() as u32, kv.disk_swap_capacity())
        })
    }

    fn rs_stats(&self) -> (RsUnits, RsUnits) {
        self.with_rs(|rs| (rs.free(), rs.capacity()))
    }

    fn rs_growth(&self) -> RsUnits {
        // Each window list's slot boundary takes a slot before the one
        // behind its window is let go.
        let lists = self.with_rs(|rs| rs.window_lists().len());
        let mut windows = [0; crate::store::rs::MAX_WINDOW_LISTS];
        for list in windows.iter_mut().take(lists) {
            *list = WINDOW_GROWTH;
        }
        RsUnits::of_windows(windows)
    }

    fn rs_host_stats(&self) -> (RsUnits, RsUnits) {
        let (free, total, disk) =
            self.with_rs(|rs| (rs.host_available(), rs.host_capacity(), rs.disk_slots()));
        // Where state parks on disk too (no host row free, no host tier on
        // unified memory), the disk's room is room below: read outside the
        // rs store's lock (the slot file is the kv store's).
        let Some(disk) = disk else {
            return (free, total);
        };
        let chunks = u32::try_from(disk.free_chunks()).unwrap_or(u32::MAX);
        let below = self.with_rs(|rs| rs.disk_units(chunks));
        (free.add(below), total.add(below))
    }

    fn guest_room_free(&self, run: u32, pieces: u32) -> bool {
        self.guest_free_runs(run) >= pieces
    }

    fn guest_need_free(&self, need: &RunNeed, state: RsUnits) -> bool {
        use crate::store::slots::OwnPiece;
        let Some((last, laid)) = need.entries().split_last() else {
            return true;
        };
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .is_some_and(|shared| {
                let shared = shared.lock().unwrap();
                let own: Vec<OwnPiece> = shared
                    .state_takes(state)
                    .into_iter()
                    .map(|(shape, slots)| OwnPiece::Take { shape, slots })
                    .chain(laid.iter().map(|&(run, pieces)| OwnPiece::Chunks { run, pieces }))
                    .collect();
                shared.pool().chunk_runs_beside(&own, last.0, last.1)
            })
    }

    fn restore_placement(
        &self,
        pid: ProcessId,
        need: &RunNeed,
        state: RsUnits,
        kv_pages: u32,
        bind: &[(u32, u32)],
    ) -> RestorePlacement {
        use crate::store::slots::OwnPiece;
        let roles_and_shared = crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| Some((stores.shared.clone()?, stores.roles.clone()?)));
        let Some((shared, roles)) = roles_and_shared else {
            let stands = self.guest_need_free(need, state);
            return RestorePlacement {
                opened: self.serve_opens(state),
                room: None,
                short: if stands { RunNeed::default() } else { need.clone() },
                bind_free_runs: None,
            };
        };
        // Its kv pages first, as its grant takes them before its serve's
        // state and its unpark: the takes read under the pool's lock and
        // let go, the roles taking it themselves.
        let takes = {
            let shared = shared.lock().unwrap();
            let mut takes: Vec<_> = shared.kv_takes(kv_pages).into_iter().collect();
            takes.extend(shared.state_takes(state));
            takes
        };
        let (own, placement) = roles.lock().unwrap().restore_placement(pid, &takes);
        let pool_alone = |rest: &[OwnPiece]| {
            shared.lock().unwrap().pool().opens_alone(rest)
        };
        RestorePlacement::of(&own, &placement, bind, pool_alone)
    }

    fn reserve_restore(&self, pid: ProcessId, bind: &[(u32, u32)]) -> bool {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.roles)
            .is_some_and(|roles| roles.lock().unwrap().reserve_restore(pid, bind))
    }

    fn release_restore(&self, pid: ProcessId) {
        if let Some(roles) =
            crate::store::registry::try_get(self.model, self.engine).and_then(|stores| stores.roles)
        {
            roles.lock().unwrap().release_claim(pid);
        }
    }

    fn copy_chunks(&self, shape: ::engine::chunk::ShapeId, refs: &[u32]) -> Vec<u32> {
        let Some(shared) =
            crate::store::registry::try_get(self.model, self.engine).and_then(|stores| stores.shared)
        else {
            return Vec::new();
        };
        let shared = shared.lock().unwrap();
        let Some(pool_shape) = shared.pool_shape(shape) else {
            return Vec::new();
        };
        let mut chunks: Vec<u32> = refs
            .iter()
            .filter_map(|&slot_ref| shared.pool().physical(shared.slot(pool_shape, slot_ref).chunk))
            .collect();
        chunks.sort_unstable();
        chunks.dedup();
        chunks
    }

    fn moving_out(&self, chunks: &[u32]) {
        if let Some(shared) =
            crate::store::registry::try_get(self.model, self.engine).and_then(|stores| stores.shared)
        {
            shared.lock().unwrap().pool_mut().moving_out(chunks);
        }
    }

    fn moved_out(&self, chunks: &[u32]) {
        if let Some(shared) =
            crate::store::registry::try_get(self.model, self.engine).and_then(|stores| stores.shared)
        {
            shared.lock().unwrap().pool_mut().moved_out(chunks);
        }
        crate::store::shared::ask_walk(self.engine);
    }

    fn serve_opens(&self, state: RsUnits) -> Vec<u32> {
        if state.is_zero() {
            return Vec::new();
        }
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .and_then(|shared| {
                let shared = shared.lock().unwrap();
                shared.pool().opens_for(&shared.state_takes(state))
            })
            .unwrap_or_default()
    }

    fn rise_room(&self, rung: u32) -> RiseRoom {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map_or(RiseRoom::Fits, |shared| shared.lock().unwrap().rise_room(rung))
    }

    fn guest_free_runs(&self, run: u32) -> u32 {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map_or(0, |shared| shared.lock().unwrap().pool().free_chunk_runs(run))
    }

    fn holds_device_room(&self, pid: ProcessId) -> bool {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.roles)
            .is_some_and(|roles| {
                let roles = roles.lock().unwrap();
                roles.holds_guest_state(pid)
                    || roles
                        .parked_chunk_slots(ProcessId::nil())
                        .iter()
                        .any(|&(_, holder, _)| holder == pid)
            })
    }

    fn room_beside_own(&self, pid: ProcessId, run: u32, pieces: u32) -> bool {
        match crate::store::registry::try_get(self.model, self.engine).and_then(|s| s.roles) {
            Some(roles) => roles.lock().unwrap().fits_beside_unpark(pid, run, pieces),
            None => self.guest_room_free(run, pieces),
        }
    }

    fn occupancy_sources(&self, pid: ProcessId) -> String {
        let Some(stores) = crate::store::registry::try_get(self.model, self.engine) else {
            return String::new();
        };
        let Some(shared) = stores.shared.clone() else {
            return String::new();
        };
        let kv_sets = crate::inferlet::process::residency::kv_working_sets_for(&[pid], self.model, self.engine);
        let guest = stores
            .roles
            .as_ref()
            .map(|roles| roles.lock().unwrap().guest_items_census(pid))
            .unwrap_or_default();
        let rs_sets = crate::inferlet::process::residency::rs_working_set_ids(pid, self.model, self.engine);
        let rs = if rs_sets.is_empty() {
            Vec::new()
        } else {
            stores.rs.lock().unwrap().private_slot_refs(&rs_sets)
        };
        let kv = crate::store::registry::with_kv_lock(&stores.kv, "planner-sources", |kv| {
            let groups: Vec<_> = kv_sets.into_iter().flatten().collect();
            kv.movable_resident_pages_each(&groups).into_iter().map(|pages| pages.len()).sum::<usize>()
        });
        let pool = shared.lock().unwrap();
        let rs: Vec<String> = rs
            .iter()
            .flat_map(|(engine, refs)| {
                let shape = pool.pool_shape(*engine);
                let pool = &pool;
                refs.iter().filter_map(move |&slot_ref| {
                    let shape = shape?;
                    let slot = pool.slot(shape, slot_ref);
                    let physical = pool.pool().physical(slot.chunk)?;
                    Some(format!("shape {engine:?} chunk {physical} (used {})", pool.pool().chunk_used(slot.chunk)))
                })
            })
            .collect();
        format!("kv_movable_pages={kv} rs_private=[{}] guest=[{guest}]", rs.join(", "))
    }

    fn unpark_census(&self, pid: ProcessId, need: Option<(u32, u32)>) -> String {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.roles)
            .map(|roles| roles.lock().unwrap().unpark_census(pid, need))
            .unwrap_or_default()
    }

    fn chunk_census(&self, run: u32) -> String {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map(|shared| {
                let shared = shared.lock().unwrap();
                format!("{} {}", shared.pool().census(run), shared.pool().budget_line())
            })
            .unwrap_or_default()
    }

    fn compaction_under_way(&self) -> bool {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .is_some_and(|shared| shared.lock().unwrap().compaction_under_way())
    }

    fn compaction_comes(&self, run: u32, pieces: u32) -> bool {
        let comes = crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .is_some_and(|shared| shared.lock().unwrap().compaction_comes(run, pieces));
        if comes {
            crate::store::shared::ask_walk(self.engine);
        }
        comes
    }

    fn free_chunks(&self) -> u32 {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map_or_else(
                || self.device_stats().0,
                |shared| shared.lock().unwrap().pool().free_chunks(),
            )
    }

    fn set_room_signal(&self, signal: crate::store::slots::RoomSignal) {
        let Some(stores) = crate::store::registry::try_get(self.model, self.engine) else {
            return;
        };
        // The device's slots, the host tier's, and the kv store's host and
        // disk slots: every tier room comes back on.
        use crate::store::slots::RoomTier;
        if let Some(shared) = &stores.shared {
            let shared = shared.lock().unwrap();
            let list = shared.list().map(|list| list.pool);
            shared.pool().set_room_signal(signal.clone(), RoomTier::Device, list);
        }
        if let Some(host) = &stores.host {
            let host = host.lock().unwrap();
            let list = host.list().map(|list| list.pool);
            host.pool().set_room_signal(signal.clone(), RoomTier::Swap, list);
        }
        crate::store::registry::with_kv_lock(&stores.kv, "planner-room-signal", |kv| {
            kv.set_room_signal(signal);
        });
    }

    fn unpark_need(&self, pid: ProcessId) -> RunNeed {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.roles)
            .map_or_else(RunNeed::default, |roles| {
                RunNeed::of(roles.lock().unwrap().unpark_need(pid))
            })
    }

    fn state_chunks(&self) -> u32 {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map_or(u32::MAX, |shared| shared.lock().unwrap().state_chunks_now())
    }

    fn state_chunks_most(&self) -> u32 {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map_or(u32::MAX, |shared| shared.lock().unwrap().state_chunks_most())
    }

    fn capacity_census(&self) -> Option<CapacityCensus> {
        let stores = crate::store::registry::try_get(self.model, self.engine)?;
        let (state_chunks, per_chunk, chunk_bytes) = {
            let shared = stores.shared.as_ref()?.lock().unwrap();
            (
                shared.state_chunks_now(),
                shared.list_slots_for_chunks(1),
                shared.pool().chunk_bytes(),
            )
        };
        if per_chunk == 0 {
            return None;
        }
        let host_chunks = stores.host.as_ref().map_or(0, |host| {
            let host = host.lock().unwrap();
            host.pool().total_chunks().saturating_sub(host.pool().reserved_chunks())
        });
        let kv = self.with_kv_tagged("planner-capacity-census", |kv| kv.census());
        let disk_chunks = u64::from(kv.disk_slots) / per_chunk;
        let (footprint, on_disk, slack) = stores.roles.as_ref().map_or((0, 0, 0), |roles| {
            let roles = roles.lock().unwrap();
            (roles.footprint(), roles.footprint_on_disk(), roles.slack())
        });
        let pages = |bytes: u64| (bytes * per_chunk).div_ceil(chunk_bytes.max(1));
        // The disk chunks parked parts take, past their bytes.
        let disk_slack = (kv.guest_disk * chunk_bytes / per_chunk).saturating_sub(on_disk);
        let rs = self.rs_as_pages(self.with_rs(|rs| rs.census_units()));
        Some(CapacityCensus {
            whole: (u64::from(state_chunks) + u64::from(host_chunks) + disk_chunks) * per_chunk,
            held: kv.held + pages(footprint) + u64::from(rs),
            slack: (slack + disk_slack) * per_chunk / chunk_bytes.max(1),
            idle: kv.idle,
            moving: kv.moving,
        })
    }

    fn plan_list_compaction(
        &self,
        wanted: u32,
        may_move: crate::store::kv::MayMove<'_>,
    ) -> Option<crate::store::kv::ListCompaction> {
        crate::store::registry::try_get(self.model, self.engine)?.shared.as_ref()?;
        self.with_kv_tagged("planner-list-compaction", |kv| {
            kv.plan_list_compaction(wanted, may_move)
        })
    }

    fn guest_pages_of(&self, pid: ProcessId) -> u64 {
        let Some(stores) = crate::store::registry::try_get(self.model, self.engine) else {
            return 0;
        };
        let Some(roles) = stores.roles.as_ref() else {
            return 0;
        };
        let bytes = roles.lock().unwrap().footprint_of(pid);
        let chunk_bytes = stores
            .shared
            .as_ref()
            .map_or(1, |shared| shared.lock().unwrap().pool().chunk_bytes().max(1));
        (bytes * u64::from(self.chunks_as_pages(1))).div_ceil(chunk_bytes)
    }

    fn context_pages(&self) -> Option<u32> {
        crate::store::registry::try_get(self.model, self.engine)
            .map(|stores| u32::try_from(stores.context_pages).unwrap_or(u32::MAX))
            .filter(|&pages| pages > 0)
    }

    fn held_back(&self) -> u32 {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map_or(0, |shared| shared.lock().unwrap().held_back_chunks())
    }

    fn room_gives(&self) -> u64 {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map_or(0, |shared| shared.lock().unwrap().pool().room_gives())
    }

    fn room_changes(&self) -> u64 {
        let shared = crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared);
        match shared {
            Some(shared) => {
                let shared = shared.lock().unwrap();
                shared.pool().room_gives().wrapping_add(shared.pool().room_changes())
            }
            None => self
                .room_gives()
                .wrapping_add(u64::from(self.device_stats().0) << 32),
        }
    }

    fn chunk_occupants(&self, pids: &[ProcessId]) -> ChunkMap {
        let Some(stores) = crate::store::registry::try_get(self.model, self.engine) else {
            return ChunkMap::default();
        };
        let Some(shared) = stores.shared.clone() else {
            return ChunkMap::default();
        };
        let sets =
            crate::inferlet::process::residency::kv_working_sets_for(pids, self.model, self.engine);
        // The guest state they hold in Instance chunks (dead between fires,
        // given back at their park), read before the pools' locks: the
        // roles take the device pool's lock themselves.
        let guest = stores
            .roles
            .as_ref()
            .map(|roles| roles.lock().unwrap().guest_chunk_slots(pids))
            .unwrap_or_default();
        // Their state's slots (records, its lists and windows): every slot of
        // every shape a candidate holds empties with it. Read before the
        // pools' locks, as the state store takes the shared pool's itself.
        let state: Vec<(ProcessId, Vec<(::engine::chunk::ShapeId, Vec<u32>)>)> = pids
            .iter()
            .map(|&pid| {
                let sets = crate::inferlet::process::residency::rs_working_set_ids(
                    pid,
                    self.model,
                    self.engine,
                );
                let refs = if sets.is_empty() {
                    Vec::new()
                } else {
                    stores.rs.lock().unwrap().private_slot_refs(&sets)
                };
                (pid, refs)
            })
            .collect();
        // The kv lock first, then the shared pool's: the order kv takes.
        crate::store::registry::with_kv_lock(&stores.kv, "planner-chunks", |kv| {
            let pool = shared.lock().unwrap();
            let Some(list) = pool.list() else {
                return ChunkMap::default();
            };
            let mut by: std::collections::BTreeMap<u32, (u32, Vec<(ProcessId, u32)>)> =
                std::collections::BTreeMap::new();
            // Every process's movable pages read in one walk of the
            // running sets' pages ([`crate::store::kv::KvStore::movable_resident_pages_each`]).
            let (named, groups): (Vec<ProcessId>, Vec<std::collections::HashSet<_>>) = pids
                .iter()
                .zip(sets)
                .filter_map(|(&pid, sets)| sets.map(|sets| (pid, sets)))
                .unzip();
            let movable = kv.movable_resident_pages_each(&groups);
            // Counted by physical chunk in one table the pool's size, its
            // touched chunks reset after each process: no map per page.
            let total = pool.pool().total_chunks() as usize;
            let mut counts = vec![0u32; total];
            let mut used = vec![0u32; total];
            let mut touched: Vec<u32> = Vec::new();
            for (pid, pages) in named.iter().zip(movable) {
                for page in pages {
                    let slot = pool.slot(list.pool, page.0);
                    if let Some(physical) = pool.pool().physical(slot.chunk) {
                        let at = physical as usize;
                        if counts[at] == 0 {
                            touched.push(physical);
                            used[at] = pool.pool().chunk_used(slot.chunk);
                        }
                        counts[at] += 1;
                    }
                }
                for physical in touched.drain(..) {
                    let at = physical as usize;
                    by.entry(physical)
                        .or_insert_with(|| (used[at], Vec::new()))
                        .1
                        .push((*pid, counts[at]));
                    counts[at] = 0;
                }
            }
            for (pid, refs) in &state {
                let mut counts: std::collections::BTreeMap<u32, u32> = Default::default();
                for (engine, refs) in refs {
                    let Some(shape) = pool.pool_shape(*engine) else {
                        continue;
                    };
                    for &slot_ref in refs {
                        let slot = pool.slot(shape, slot_ref);
                        if let Some(physical) = pool.pool().physical(slot.chunk) {
                            *counts.entry(physical).or_default() += 1;
                            by.entry(physical)
                                .or_insert_with(|| (pool.pool().chunk_used(slot.chunk), Vec::new()));
                        }
                    }
                }
                for (physical, slots) in counts {
                    let entry = by.get_mut(&physical).expect("entered");
                    match entry.1.iter_mut().find(|(held, _)| held == pid) {
                        Some((_, count)) => *count += slots,
                        None => entry.1.push((*pid, slots)),
                    }
                }
            }
            for (logical, pid, slots) in guest {
                if let Some(physical) = pool.pool().physical(logical) {
                    let entry = by
                        .entry(physical)
                        .or_insert_with(|| (pool.pool().chunk_used(logical), Vec::new()));
                    match entry.1.iter_mut().find(|(held, _)| *held == pid) {
                        Some((_, count)) => *count += slots,
                        None => entry.1.push((pid, slots)),
                    }
                }
            }
            ChunkMap {
                held: by
                    .into_iter()
                    .map(|(chunk, (used, by))| ChunkHeld { chunk, used, by })
                    .collect(),
                free: pool.pool().state_open_chunks(),
                headroom: Some(pool.pool().headroom()),
                segment_chunks: pool.pool().segments().segment_chunks(),
            }
        })
    }

    fn parked_occupants(&self, except: ProcessId) -> ChunkMap {
        let Some(stores) = crate::store::registry::try_get(self.model, self.engine) else {
            return ChunkMap::default();
        };
        let (Some(shared), Some(roles)) = (stores.shared.clone(), stores.roles.clone()) else {
            return ChunkMap::default();
        };
        // Read before the pool's lock: the roles take it themselves.
        let parked = roles.lock().unwrap().parked_chunk_slots(except);
        let pool = shared.lock().unwrap();
        let mut by: std::collections::BTreeMap<u32, (u32, Vec<(ProcessId, u32)>)> =
            std::collections::BTreeMap::new();
        for (logical, pid, slots) in parked {
            if let Some(physical) = pool.pool().physical(logical) {
                let entry = by
                    .entry(physical)
                    .or_insert_with(|| (pool.pool().chunk_used(logical), Vec::new()));
                match entry.1.iter_mut().find(|(held, _)| *held == pid) {
                    Some((_, count)) => *count += slots,
                    None => entry.1.push((pid, slots)),
                }
            }
        }
        ChunkMap {
            held: by
                .into_iter()
                .map(|(chunk, (used, by))| ChunkHeld { chunk, used, by })
                .collect(),
            free: pool.pool().state_open_chunks(),
            headroom: Some(pool.pool().headroom()),
            segment_chunks: pool.pool().segments().segment_chunks(),
        }
    }

    fn move_parked_down(
        &self,
        planner: &std::sync::Arc<super::ResidencyPlanner>,
        pids: &[ProcessId],
        order: &[ProcessId],
    ) -> bool {
        if !self.suspend_capable {
            return false;
        }
        let Some(stores) = crate::store::registry::try_get(self.model, self.engine) else {
            return false;
        };
        // Their kv left on the device first: a page every reader of which
        // is evicted (a prompt prefix several forks share, held at each
        // eviction by another reader or a key) is theirs together, counted
        // in the map as theirs (`chunk_occupants` of the evicted), and goes
        // down a tier once with their sets, every reader's refs pointing
        // there. Passed over, it held its chunk for good, no process its
        // holder (the L40S at 3 GiB disk-only: chunk 4, one slot, seven
        // evicted forks' refs, a head short of it unserved).
        let kv_sets: Vec<(crate::store::kv::page_table::WorkingSetId, u64)> = pids
            .iter()
            .flat_map(|&pid| {
                crate::inferlet::process::residency::kv_working_set_ids(pid, self.model, self.engine)
            })
            .map(|ws| (ws, 0))
            .collect();
        if !kv_sets.is_empty() {
            // What a restore under way reads stays: it would bring it back.
            let keep: HashSet<crate::store::kv::page_table::WorkingSetId> = planner
                .restoring_pids()
                .into_iter()
                .flat_map(|pid| {
                    crate::inferlet::process::residency::kv_working_set_ids(pid, self.model, self.engine)
                })
                .collect();
            let sweep = self.with_kv_tagged("planner-park-down", |kv| {
                kv.prepare_sweep(&keep, usize::MAX, &kv_sets, usize::MAX)
            });
            match sweep {
                Ok(Some(txn)) => {
                    super::exec::spawn_move_out(
                        planner.clone(),
                        txn,
                        "parked kv move down",
                        super::ResidencyPlanner::parked_down_done,
                    );
                    return true;
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(%error, "planner: parked kv's move down refused");
                }
            }
        }
        // An evicted process's state left on the device at its eviction
        // (records and rs list slots its suspend found no row for, or still
        // settling then) goes down with its parked role slots: left there,
        // queued processes waiting to be restored held chunks no move
        // freed (Mac Pro on 22b0a3e83: 85 chunks of one live slot each, all
        // queued processes').
        let sets: HashSet<crate::store::rs::RsWorkingSetId> = pids
            .iter()
            .flat_map(|&pid| {
                crate::inferlet::process::residency::rs_working_set_ids(pid, self.model, self.engine)
            })
            .collect();
        let rs = if sets.is_empty() {
            None
        } else {
            stores.rs.lock().unwrap().prepare_suspend(&sets)
        };
        let Some(roles) = stores.roles.clone() else {
            if rs.is_some() {
                super::exec::spawn_park_down(planner.clone(), pids.to_vec(), None, rs);
                return true;
            }
            return false;
        };
        let holders: Vec<crate::store::roles::Holder> = pids.to_vec();
        // The host first where the load states one; with none (unified
        // memory) the disk at once.
        let down = roles
            .lock()
            .unwrap()
            .prepare_park_down(&holders, stores.host.is_none());
        if down.txn.is_some() || rs.is_some() {
            super::exec::spawn_park_down(planner.clone(), pids.to_vec(), down.txn, rs);
            return true;
        }
        if let Some(host) = stores.host.as_ref().filter(|_| down.host_chunks > 0) {
            // The host is short: what it holds of kv goes on down to disk
            // to make the room, the cache's first, then the parked sets',
            // lowest first, so the tiers keep the newer above the older.
            let chunks = u32::try_from(down.host_chunks).unwrap_or(u32::MAX);
            let wanted = usize::try_from(host.lock().unwrap().list_slots_for_chunks(chunks))
                .unwrap_or(usize::MAX)
                .max(1);
            let sets: Vec<crate::store::kv::page_table::WorkingSetId> = order
                .iter()
                .flat_map(|&pid| {
                    crate::inferlet::process::residency::kv_working_set_ids(
                        pid,
                        self.model,
                        self.engine,
                    )
                })
                .collect();
            let spill = self.with_kv_tagged("planner-park-down", |kv| {
                kv.prepare_spill(wanted, crate::policy::retention::Installed)
                    .or_else(|| kv.prepare_parked_spill(wanted, &sets))
            });
            if let Some(txn) = spill {
                // The state goes down once the spill has made the room:
                // the next move down takes it.
                if let Some(rs) = rs {
                    stores.rs.lock().unwrap().abort_suspend(rs);
                }
                super::exec::spawn_spill(
                    planner.clone(),
                    txn,
                    super::ResidencyPlanner::parked_down_done,
                );
                return true;
            }
        }
        // No kv on the host to make room (or pages settled on their disk
        // copies made it with nothing to copy): the host where it has room
        // now, else the disk.
        match roles.lock().unwrap().prepare_park_down(&holders, true).txn {
            Some(txn) => {
                super::exec::spawn_park_down(planner.clone(), pids.to_vec(), Some(txn), None);
                true
            }
            None => false,
        }
    }

    fn chunks_as_pages(&self, chunks: u32) -> u32 {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map_or(0, |shared| {
                let slots = shared.lock().unwrap().list_slots_for_chunks(chunks);
                u32::try_from(slots).unwrap_or(u32::MAX)
            })
    }

    fn rs_as_pages(&self, rs: RsUnits) -> u32 {
        if rs.is_zero() {
            return 0;
        }
        let stores = crate::store::registry::get(self.model, self.engine);
        let Some(shared) = stores.shared.as_ref() else {
            return 0;
        };
        let slots = shared.lock().unwrap().list_slots_for(rs);
        u32::try_from(slots).unwrap_or(u32::MAX)
    }

    fn reclaim_idle(&self) -> u32 {
        let pages = self.with_kv_tagged("planner-reclaim-idle", |kv| {
            let epoch = kv.current_epoch();
            kv.reclaim_droppable(epoch, crate::policy::retention::Installed)
        });
        let slots = self.with_rs(|rs| rs.reclaim_droppable(crate::policy::retention::Installed));
        (pages + slots) as u32
    }

    fn reclaimable_idle(&self) -> u32 {
        let mut last = self.reclaimable.lock();
        if let Some((at, pages)) = *last
            && at.elapsed() < RECLAIMABLE_FRESH
        {
            return pages;
        }
        let pages = self.with_kv_tagged("planner-reclaimable-idle", |kv| {
            kv.reclaimable_cache_pages()
        });
        *last = Some((crate::rt::Instant::now(), pages));
        pages
    }

    fn idle_below_device(&self) -> (u32, u32) {
        self.with_kv_tagged("planner-idle-below-device", |kv| {
            kv.below_device_cache_pages()
        })
    }

    fn suspend_capable(&self) -> bool {
        self.suspend_capable
    }

    fn locus(&self) -> (usize, usize) {
        (self.model, self.engine)
    }

    fn reserve_device(
        &self,
        count: u32,
    ) -> Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>> {
        let pages = self.with_kv_tagged("planner-reserve", |kv| {
            let pages = kv.reserve_device_pages(count as usize);
            if self.shared.is_some() {
                if pages.is_none() {
                    kv.note_short(count);
                } else {
                    kv.clear_short();
                }
            }
            pages
        });
        if pages.is_none()
            && let Some(shared) = &self.shared
        {
            shared.wake();
        }
        pages
    }

    fn freed(&self) {
        crate::store::shared::rung_changed(self.engine);
        if let Some(shared) = &self.shared {
            shared.room_freed();
        }
    }

    fn grow(&self, need: super::Demand) -> bool {
        if let Some(shared) = &self.shared {
            if need.kv_pages > 0 {
                self.with_kv_tagged("planner-grow", |kv| kv.note_short(need.kv_pages));
            }
            // State, window and rs-list slots come from the same pool: the
            // walker commits for what their waiters are short too.
            if !need.rs.is_zero() {
                self.with_rs(|rs| rs.note_short_shared(need.rs));
            }
            // Promised only where the walk's next step maps a segment;
            // a walk that ends without one tells the planner no grow.
            return shared.promise_grow();
        }
        false
    }

    fn reserve_device_up_to(
        &self,
        count: u32,
    ) -> Vec<crate::store::kv::page_table::PhysicalKvPageId> {
        self.with_kv_tagged("planner-reserve-up-to", |kv| {
            let take = (kv.free_committed() as u32).min(count);
            if take == 0 {
                return Vec::new();
            }
            kv.reserve_device_pages(take as usize).unwrap_or_default()
        })
    }

    fn release_device(&self, pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>) {
        self.with_kv_tagged("planner-release", |kv| {
            kv.release_device_reservation(pages);
        });
    }

    fn repack_device(
        &self,
        pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>,
    ) -> Vec<crate::store::kv::page_table::PhysicalKvPageId> {
        if crate::store::registry::try_get(self.model, self.engine)
            .is_none_or(|stores| stores.shared.is_none())
        {
            return pages;
        }
        self.with_kv_tagged("planner-repack", |kv| kv.repack_reserved(pages))
    }

    fn reserve_rs(&self, need: RsUnits) -> Option<crate::store::rs::RsGrant> {
        self.with_rs(|rs| {
            let grant = rs.reserve(need);
            // A shared pool's waiter got what it was short: its shortfall
            // no longer asks the segment walker for more.
            if grant.is_some() && self.shared.is_some() {
                rs.note_short_shared(RsUnits::default());
            }
            grant
        })
    }

    fn release_rs(&self, grant: crate::store::rs::RsGrant) {
        self.with_rs(|rs| rs.release_reservation(grant));
    }

    fn reclaim_sharing(&self, working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize {
        self.with_rs(|rs| rs.reclaim_sharing(working_sets, crate::policy::retention::Installed))
    }

    fn prepare_sweep(
        &self,
        keep: &HashSet<crate::store::kv::page_table::WorkingSetId>,
        room: usize,
        order: &[(crate::store::kv::page_table::WorkingSetId, u64)],
        wanted: usize,
    ) -> Option<crate::store::kv::KvResidencyTxn> {
        if !self.suspend_capable {
            return None;
        }
        self.with_kv_tagged("planner-sweep", |kv| {
            kv.prepare_sweep(keep, room, order, wanted)
        })
        .inspect_err(|error| tracing::warn!(%error, "planner: sweep prepare failed"))
        .ok()
        .flatten()
    }

    fn prepare_tail_sweep(
        &self,
        keep: &HashSet<crate::store::kv::page_table::WorkingSetId>,
        room: usize,
        order: &[(crate::store::kv::page_table::WorkingSetId, u64)],
        wanted: usize,
    ) -> Option<crate::store::kv::KvResidencyTxn> {
        if !self.suspend_capable {
            return None;
        }
        self.with_kv_tagged("planner-sweep", |kv| {
            kv.prepare_tail_sweep(keep, room, order, wanted)
        })
        .inspect_err(|error| tracing::warn!(%error, "planner: sweep prepare failed"))
        .ok()
        .flatten()
    }

    fn prepare_demote(&self, room: usize) -> Option<crate::store::kv::KvResidencyTxn> {
        if !self.suspend_capable {
            return None;
        }
        self.with_kv_tagged("planner-demote", |kv| {
            kv.prepare_demote(room, crate::policy::retention::Installed)
        })
        .inspect_err(|error| tracing::warn!(%error, "planner: demote prepare failed"))
        .ok()
        .flatten()
    }

    fn prepare_spill(&self, room: usize) -> Option<crate::store::kv::KvSpillTxn> {
        if !self.suspend_capable {
            return None;
        }
        self.with_kv_tagged("planner-spill", |kv| {
            kv.prepare_spill(room, crate::policy::retention::Installed)
        })
    }
}
