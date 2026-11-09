use std::collections::HashSet;

use crate::store::grow::Grow;
use crate::store::rs::RsUnits;

/// The window-ring pages a seated lane past its window holds beyond its
/// prompt's: the straddled page and the one its next boundary takes before
/// the oldest is let go.
const RING_GROWTH: u32 = 2;

pub trait PoolPort: Send + Sync + 'static {
    fn device_stats(&self) -> (u32, u32);
    /// Device pages handed out. A target below the pool is room the target
    /// can rise into, so this is not `device_stats`' complement there.
    fn device_out(&self) -> u32 {
        let (free, total) = self.device_stats();
        total.saturating_sub(free)
    }
    fn host_stats(&self) -> (u32, u32);
    fn disk_stats(&self) -> (u32, u32);
    fn rs_stats(&self) -> (RsUnits, RsUnits);
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
    /// Asks for `count` pages more than stand free now; true when memory
    /// is on its way, and `pages_freed` follows it.
    fn grow(&self, count: u32) -> bool {
        let _ = count;
        false
    }
    /// Asks for `need` more state than stands free; as [`Self::grow`].
    fn grow_rs(&self, need: RsUnits) -> bool {
        let _ = need;
        false
    }
    /// The working sets reading pages past the pool's limit.
    fn working_sets_past_limit(&self) -> HashSet<crate::store::kv::page_table::WorkingSetId> {
        HashSet::new()
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
}

pub struct RegistryPool {
    model: usize,
    engine: usize,
    suspend_capable: bool,
    /// The last count of reclaimable idle pages and when it was taken.
    reclaimable: parking_lot::Mutex<Option<(crate::rt::Instant, u32)>>,
    grow: Option<Grow>,
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
            grow: None,
        }
    }

    /// Commits the pool's pages on demand rather than holding them whole.
    pub fn with_grow(mut self, grow: Grow) -> Self {
        self.grow = Some(grow);
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
    fn device_stats(&self) -> (u32, u32) {
        self.with_kv_tagged("planner-device-stats", |kv| {
            (kv.available_pages() as u32, kv.capacity_pages())
        })
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
        if self.with_rs(|rs| rs.is_ring()) {
            RsUnits {
                slots: 0,
                pages: RING_GROWTH,
            }
        } else {
            RsUnits::default()
        }
    }

    fn rs_host_stats(&self) -> (RsUnits, RsUnits) {
        self.with_rs(|rs| (rs.host_available(), rs.host_capacity()))
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
        let (pages, due) = self.with_kv_tagged("planner-reserve", |kv| {
            let pages = kv.reserve_device_pages(count as usize);
            let due = self.grow.as_ref().is_some_and(|grow| {
                if pages.is_none() {
                    kv.note_short(count);
                }
                pages.is_none() || kv.commit_due(grow.step())
            });
            (pages, due)
        });
        if due && let Some(grow) = &self.grow {
            grow.wake();
        }
        pages
    }

    fn working_sets_past_limit(&self) -> HashSet<crate::store::kv::page_table::WorkingSetId> {
        self.with_kv_tagged("planner-tail", |kv| kv.working_sets_past_limit())
    }

    fn grow(&self, count: u32) -> bool {
        let Some(grow) = &self.grow else {
            return false;
        };
        let growing = self.with_kv_tagged("planner-grow", |kv| {
            kv.note_short(count);
            kv.can_commit()
        });
        if growing {
            grow.wake();
            return true;
        }
        grow.ask()
    }

    fn grow_rs(&self, need: RsUnits) -> bool {
        let Some(grow) = &self.grow else {
            return false;
        };
        if self.with_rs(|rs| rs.note_short(need)) {
            grow.wake();
            return true;
        }
        grow.state_on_demand() && grow.ask()
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
        let (grant, due) = self.with_rs(|rs| {
            let grant = rs.reserve(need);
            let due = self
                .grow
                .as_ref()
                .is_some_and(|grow| rs.next_extent(grow.state_step()).is_some());
            (grant, due)
        });
        if due && let Some(grow) = &self.grow {
            grow.wake();
        }
        grant
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
}
