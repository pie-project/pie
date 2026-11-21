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
    /// Whether every entry of `need` stands free together, as a guest's
    /// takes lay them: the longest runs first, each shorter length beside
    /// them, the budget's headroom spent by all of them. Where the pool
    /// lays nothing apart, each entry read on its own.
    fn guest_need_free(&self, need: &RunNeed) -> bool {
        need.entries()
            .iter()
            .all(|&(run, pieces)| self.guest_room_free(run, pieces))
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
    /// The device pool's count of room given back (a chunk opened for
    /// state, [`crate::store::slots::SlotPool::room_gives`]): moved since
    /// a read, room has come free since.
    fn room_gives(&self) -> u64 {
        0
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

    fn guest_need_free(&self, need: &RunNeed) -> bool {
        use crate::store::slots::OwnPiece;
        let Some((last, laid)) = need.entries().split_last() else {
            return true;
        };
        let own: Vec<OwnPiece> = laid
            .iter()
            .map(|&(run, pieces)| OwnPiece::Chunks { run, pieces })
            .collect();
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .is_some_and(|shared| {
                shared
                    .lock()
                    .unwrap()
                    .pool()
                    .chunk_runs_beside(&own, last.0, last.1)
            })
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

    fn room_gives(&self) -> u64 {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .map_or(0, |shared| shared.lock().unwrap().pool().room_gives())
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
            for (&pid, sets) in pids.iter().zip(sets) {
                let Some(sets) = sets else {
                    continue;
                };
                let mut counts: std::collections::BTreeMap<u32, u32> = Default::default();
                for page in kv.movable_resident_pages(&sets) {
                    let slot = pool.slot(list.pool, page.0);
                    if let Some(physical) = pool.pool().physical(slot.chunk) {
                        *counts.entry(physical).or_default() += 1;
                        by.entry(physical)
                            .or_insert_with(|| (pool.pool().chunk_used(slot.chunk), Vec::new()));
                    }
                }
                for (physical, pages) in counts {
                    by.get_mut(&physical).expect("entered").1.push((pid, pages));
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
                super::exec::spawn_park_down(planner.clone(), None, rs);
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
            super::exec::spawn_park_down(planner.clone(), down.txn, rs);
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
                super::exec::spawn_park_down(planner.clone(), Some(txn), None);
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
