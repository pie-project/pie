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
    fn suspend_capable(&self) -> bool;
    fn locus(&self) -> (usize, usize);
    /// The rs units a seated lane may still claim as it decodes: a window
    /// ring's page boundary takes a buffer page before the page behind the
    /// window is let go. Fixed state slots never grow.
    fn rs_growth(&self) -> RsUnits {
        RsUnits::default()
    }
    /// Room was freed or a fire retired: what waits on it beside the
    /// planner's own waiters (the rung, the segment walk, arming) looks
    /// again.
    fn freed(&self) {}
    /// Asks for `need` more of each shape than stands free now; true when
    /// memory is on its way, and `pages_freed` follows it.
    fn grow(&self, need: super::Demand) -> bool {
        let _ = need;
        false
    }
    fn grow_pages(&self, count: u32) -> bool {
        self.grow(super::Demand {
            kv_pages: count,
            rs: RsUnits::default(),
        })
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
        self.with_rs(|rs| (rs.host_available(), rs.host_capacity()))
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
        crate::scheduler::arming::room_freed(self.engine);
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
            return shared.can_grow() && {
                shared.wake();
                true
            };
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
