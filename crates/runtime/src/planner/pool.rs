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

/// The device pages working sets take, by part ([`PoolPort::working_set_pages`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkingSetPages {
    /// Their kv pages on the device, each shared page once.
    pub resident: u64,
    /// Their kv pages reserved past those mapped: their next step.
    pub reserved: u64,
    /// Their state.
    pub state: u64,
    /// Their instances' slots (tables, rings, guest state).
    pub roles: u64,
}

impl WorkingSetPages {
    pub fn total(&self) -> u64 {
        self.resident
            .saturating_add(self.reserved)
            .saturating_add(self.state)
            .saturating_add(self.roles)
    }
}

/// Physical chunks of a shared pool by what holds them
/// ([`PoolPort::chunk_occupants`]).
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

impl ChunkMap {
    /// The processes whose eviction empties the cheapest `pieces` disjoint
    /// runs of `run` adjacent chunks, each chunk free or emptiable
    /// ([`ChunkHeld::emptiable`]), cost the pages moved; `None` where no
    /// such runs stand.
    #[must_use]
    pub fn cheapest_runs(&self, run: u32, pieces: u32) -> Option<Vec<ProcessId>> {
        let run = run.max(1);
        let one_segment = |first: u32| {
            self.segment_chunks == 0
                || first / self.segment_chunks == (first + run - 1) / self.segment_chunks
        };
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
        // Every window of `run` adjacent chunks each free or emptiable, by
        // its cost; then the cheapest disjoint ones.
        let starts: std::collections::BTreeSet<u32> =
            free.iter().chain(held.keys()).copied().collect();
        let mut windows: Vec<(u32, u32)> = starts
            .iter()
            .filter(|&&first| one_segment(first))
            .filter_map(|&first| {
                (first..first + run)
                    .map(cost)
                    .try_fold(0u32, |sum, cost| cost.map(|cost| sum + cost))
                    .map(|sum| (sum, first))
            })
            .collect();
        windows.sort_unstable();
        let mut taken: Vec<(u32, u32)> = Vec::new();
        for (sum, first) in windows {
            if taken.len() as u32 == pieces {
                break;
            }
            if taken
                .iter()
                .all(|&(_, at)| first + run <= at || at + run <= first)
            {
                taken.push((sum, first));
            }
        }
        if (taken.len() as u32) < pieces {
            return None;
        }
        let in_runs: std::collections::BTreeSet<u32> = taken
            .iter()
            .flat_map(|&(_, first)| first..first + run)
            .collect();
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
    /// Whether `pieces` free runs of `run` adjacent chunks stand for a
    /// guest's state now: what a bind short of room waits on
    /// ([`super::wait::WaitKind::BindRoom`]). None where the pool keeps
    /// no guest state.
    fn guest_room_free(&self, run: u32, pieces: u32) -> bool {
        let _ = (run, pieces);
        false
    }
    /// The device pages that, moved out whole, free `chunks` chunks for a
    /// guest's state, as [`Self::rs_as_pages`] for state.
    fn chunks_as_pages(&self, chunks: u32) -> u32 {
        let _ = chunks;
        0
    }
    /// The device pages the working sets of `pids` take together now, with
    /// their next step: their kv pages resident on the device, each page
    /// they share once, with the pages they reserved past those they map;
    /// their state; their instances' slots (tables, rings, guest state). A
    /// set parked below the device takes none until it comes back. Pages are the room's
    /// unit: what is not kv counts as the pages it keeps out of the pool.
    /// What admission weighs ([`super::ResidencyPlanner::admits`]).
    fn working_set_pages(&self, pids: &[super::ProcessId]) -> WorkingSetPages {
        let _ = pids;
        WorkingSetPages::default()
    }
    /// The device pages dropping or demoting idle cache keyed at most
    /// `up_to` would free: what a demand keyed so may take.
    fn idle_pages_up_to(&self, up_to: i64) -> u64 {
        let _ = up_to;
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
    /// Idle cache keyed above the floor down a tier, lowest key first;
    /// with `bound`, only cache keyed at or below its key, or the asker's
    /// own (what a demand keyed so may take: a higher-keyed set stays).
    fn prepare_demote(
        &self,
        room: usize,
        bound: Option<crate::store::kv::DemoteBound>,
    ) -> Option<crate::store::kv::KvResidencyTxn>;
    /// Idle cache a `retention/tiers` row keeps off the device, to move
    /// down now.
    fn prepare_demote_now(&self) -> Option<crate::store::kv::KvResidencyTxn>;
    /// Idle cache a `retention/tiers` row allows nowhere it is, dropped now.
    fn drop_disallowed(&self) -> usize;
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
        self.with_rs(|rs| (rs.host_available(), rs.host_capacity()))
    }

    fn guest_room_free(&self, run: u32, pieces: u32) -> bool {
        crate::store::registry::try_get(self.model, self.engine)
            .and_then(|stores| stores.shared)
            .is_some_and(|shared| shared.lock().unwrap().pool().free_chunk_runs(run) >= pieces)
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

    fn working_set_pages(&self, pids: &[super::ProcessId]) -> WorkingSetPages {
        let Some(stores) = crate::store::registry::try_get(self.model, self.engine) else {
            return WorkingSetPages::default();
        };
        let kv_sets: HashSet<_> = pids
            .iter()
            .flat_map(|&pid| {
                crate::inferlet::process::residency::kv_working_set_ids(
                    pid,
                    self.model,
                    self.engine,
                )
            })
            .collect();
        let (resident, reserved) =
            crate::store::registry::with_kv_lock(&stores.kv, "planner-working-sets", |kv| {
                kv.working_set_pages(&kv_sets)
            });
        let rs = pids.iter().fold(RsUnits::default(), |sum, &pid| {
            sum.add(super::holdings::held_rs_units(pid, self.model, self.engine))
        });
        let roles = stores.roles.as_ref().map_or(0, |roles| {
            let roles = roles.lock().unwrap();
            pids.iter()
                .map(|&pid| roles.bytes_held_by(pid))
                .sum::<u64>()
        });
        let roles = self.room().map_or(0, |room| {
            let chunks = roles.div_ceil(room.chunk_bytes.max(1));
            self.chunks_as_pages(u32::try_from(chunks).unwrap_or(u32::MAX))
        });
        WorkingSetPages {
            resident: resident as u64,
            reserved: reserved as u64,
            state: u64::from(self.rs_as_pages(rs)),
            roles: u64::from(roles),
        }
    }

    fn idle_pages_up_to(&self, up_to: i64) -> u64 {
        self.with_kv_tagged("planner-idle-up-to", |kv| {
            kv.idle_pages_up_to(crate::policy::retention::Installed, up_to)
        }) as u64
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
            kv.prepare_sweep(
                keep,
                room,
                order,
                wanted,
                crate::policy::retention::Installed,
            )
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
            kv.prepare_tail_sweep(
                keep,
                room,
                order,
                wanted,
                crate::policy::retention::Installed,
            )
        })
        .inspect_err(|error| tracing::warn!(%error, "planner: sweep prepare failed"))
        .ok()
        .flatten()
    }

    fn prepare_demote(
        &self,
        room: usize,
        bound: Option<crate::store::kv::DemoteBound>,
    ) -> Option<crate::store::kv::KvResidencyTxn> {
        if !self.suspend_capable {
            return None;
        }
        self.with_kv_tagged("planner-demote", |kv| {
            kv.prepare_demote(room, crate::policy::retention::Installed, bound)
        })
        .inspect_err(|error| tracing::warn!(%error, "planner: demote prepare failed"))
        .ok()
        .flatten()
    }

    fn prepare_demote_now(&self) -> Option<crate::store::kv::KvResidencyTxn> {
        if !self.suspend_capable {
            return None;
        }
        self.with_kv_tagged("planner-demote", |kv| {
            kv.prepare_demote_now(crate::policy::retention::Installed)
        })
        .inspect_err(|error| tracing::warn!(%error, "planner: demote prepare failed"))
        .ok()
        .flatten()
    }

    fn drop_disallowed(&self) -> usize {
        self.with_kv_tagged("planner-tiers", |kv| {
            Ok::<_, crate::store::kv::KvStoreError>(
                kv.drop_disallowed(crate::policy::retention::Installed),
            )
        })
        .unwrap_or(0)
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
