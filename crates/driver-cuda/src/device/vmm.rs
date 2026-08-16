//! The elastic virtual-memory arena: hands physical memory back without moving
//! the tensors on top. One reserved virtual range has handles mapped and
//! unmapped under it, so pointers survive a shrink.

use std::sync::Mutex;

use cudarc::driver::sys::{
    CUdeviceptr, CUmemAccess_flags, CUmemAccessDesc, CUmemAllocationGranularity_flags,
    CUmemAllocationProp, CUmemAllocationType, CUmemGenericAllocationHandle, CUmemLocationType,
    cuMemAddressFree, cuMemAddressReserve, cuMemCreate, cuMemGetAllocationGranularity, cuMemMap,
    cuMemRelease, cuMemSetAccess, cuMemUnmap,
};
use cudarc::runtime::sys::cudaDeviceCanAccessPeer;

use crate::error::{Error, Result, check_cu, ignore_in_drop};

/// The logical page the budget is denominated in: 2 MiB. Not CUDA's allocation
/// granularity or map unit — stays meaningful as the map unit varies by device.
pub const LOGICAL_PAGE_BYTES: usize = 2 * 1024 * 1024;

/// Logical pages needed to cover `bytes`, rounding up; zero bytes need zero pages.
#[must_use]
pub const fn pages_for_bytes(bytes: usize, page_bytes: usize) -> usize {
    if bytes == 0 || page_bytes == 0 {
        0
    } else {
        bytes.div_ceil(page_bytes)
    }
}

const fn align_up(value: usize, alignment: usize) -> usize {
    if value == 0 || alignment == 0 {
        0
    } else {
        value.div_ceil(alignment) * alignment
    }
}

/// The shared page budget, with no CUDA in it. Two counters — `held` (reserved
/// before `cuMemCreate`, so a refusal costs no physical touch) and `committed`
/// (mapped) — with a page charged the whole time it moves between them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PoolBudget {
    budget_pages: usize,
    /// High-water ceiling: recalibration won't permanently shrink the pool from a
    /// transient memory dip.
    hard_budget_pages: usize,
    held_pages: usize,
    committed_pages: usize,
    /// Bumped on release or budget change, so a cached "pool full" result knows to retry.
    generation: u64,
}

impl PoolBudget {
    /// A budget covering `budget_bytes`, truncated to whole logical pages.
    #[must_use]
    pub const fn new(budget_bytes: usize) -> Self {
        let pages = budget_bytes / LOGICAL_PAGE_BYTES;
        Self {
            budget_pages: pages,
            hard_budget_pages: pages,
            held_pages: 0,
            committed_pages: 0,
            generation: 1,
        }
    }

    /// Pages charged against the budget: held plus committed.
    #[must_use]
    pub const fn charged_pages(&self) -> usize {
        self.committed_pages + self.held_pages
    }

    /// Pages a caller could still reserve.
    #[must_use]
    pub const fn free_pages(&self) -> usize {
        self.budget_pages.saturating_sub(self.charged_pages())
    }

    /// Current budget ceiling.
    #[must_use]
    pub const fn budget_pages(&self) -> usize {
        self.budget_pages
    }

    /// High-water ceiling that recalibration may restore.
    #[must_use]
    pub const fn hard_budget_pages(&self) -> usize {
        self.hard_budget_pages
    }

    /// Pages promised but not yet mapped.
    #[must_use]
    pub const fn held_pages(&self) -> usize {
        self.held_pages
    }

    /// Pages currently mapped.
    #[must_use]
    pub const fn committed_pages(&self) -> usize {
        self.committed_pages
    }

    /// Changes whenever a previously refused reservation might now succeed.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Claim `pages`; `false` means the budget cannot cover them, unchanged.
    pub const fn try_reserve(&mut self, pages: usize) -> bool {
        if pages > self.free_pages() {
            return false;
        }
        self.held_pages += pages;
        true
    }

    /// Give back pages that were reserved but never mapped. Saturating, not
    /// checked — on the failure/unwind path, over-releasing must not itself fail.
    pub const fn unreserve(&mut self, pages: usize) {
        let released = if pages < self.held_pages {
            pages
        } else {
            self.held_pages
        };
        self.held_pages -= released;
        if released != 0 {
            self.generation += 1;
        }
    }

    /// Move `pages` from held to committed. Errors if they were never reserved
    /// — reserve/commit would have drifted, silently overrunning the budget.
    pub fn mark_committed(&mut self, pages: usize) -> Result<()> {
        if pages > self.held_pages {
            return Err(Error::invalid(
                "PoolBudget::mark_committed",
                format!(
                    "committing {pages} pages but only {} are held",
                    self.held_pages
                ),
            ));
        }
        self.held_pages -= pages;
        self.committed_pages += pages;
        Ok(())
    }

    /// Give back pages that were mapped and have now been unmapped.
    pub const fn mark_uncommitted(&mut self, pages: usize) {
        let released = if pages < self.committed_pages {
            pages
        } else {
            self.committed_pages
        };
        self.committed_pages -= released;
        if released != 0 {
            self.generation += 1;
        }
    }

    /// Re-derive the budget from free device memory: charged plus free beyond
    /// the safety floor, never below charged. `reset_hard_ceiling` also drops the high-water mark.
    pub fn recalibrate(
        &mut self,
        available_bytes: usize,
        safety_floor_bytes: usize,
        reset_hard_ceiling: bool,
    ) {
        let charged = self.charged_pages();
        let usable = available_bytes.saturating_sub(safety_floor_bytes);
        let next_budget = charged
            .saturating_add(usable / LOGICAL_PAGE_BYTES)
            .max(charged);
        let next_hard = if reset_hard_ceiling {
            next_budget
        } else {
            self.hard_budget_pages.max(next_budget)
        };
        if self.budget_pages != next_budget || self.hard_budget_pages != next_hard {
            self.budget_pages = next_budget;
            self.hard_budget_pages = next_hard;
            self.generation += 1;
        }
    }
}

/// A device's supply of physical pages: [`PoolBudget`] plus the CUDA handle lifecycle.
#[derive(Debug)]
pub struct PhysicalPool {
    device_ordinal: i32,
    peer_devices: Vec<i32>,
    allocation_granularity: usize,
    handle_bytes: usize,
    budget: Mutex<PoolBudget>,
}

impl PhysicalPool {
    /// Query the device's allocation granularity and open a budget over
    /// `budget_bytes`; `handle_bytes` is rounded up to a multiple of it, as `cuMemCreate` requires.
    pub fn new(device_ordinal: i32, budget_bytes: usize, handle_bytes: usize) -> Result<Self> {
        let prop = allocation_prop(device_ordinal);
        let mut granularity: usize = 0;
        check_cu(
            unsafe {
                cuMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "cuMemGetAllocationGranularity",
        )?;
        if granularity == 0 {
            return Err(Error::invalid(
                "cuMemGetAllocationGranularity",
                "device reported a zero allocation granularity",
            ));
        }
        Ok(Self {
            device_ordinal,
            peer_devices: Vec::new(),
            allocation_granularity: granularity,
            handle_bytes: align_up(handle_bytes.max(granularity), granularity),
            budget: Mutex::new(PoolBudget::new(budget_bytes)),
        })
    }

    /// The C++ default growth unit, 32 MiB.
    pub const DEFAULT_HANDLE_BYTES: usize = 32 * 1024 * 1024;

    /// The device these pages live on.
    #[must_use]
    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    /// `cuMemCreate`'s minimum allocation size on this device.
    #[must_use]
    pub const fn allocation_granularity(&self) -> usize {
        self.allocation_granularity
    }

    /// The size of one physical handle -- the unit an arena grows by.
    #[must_use]
    pub const fn handle_bytes(&self) -> usize {
        self.handle_bytes
    }

    /// A snapshot of the accounting.
    pub fn budget(&self) -> PoolBudget {
        self.budget
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Run `f` against the live budget under the pool's lock.
    fn with_budget<T>(&self, f: impl FnOnce(&mut PoolBudget) -> T) -> T {
        f(&mut self.budget.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Claim `pages` before mapping them. See [`PoolBudget::try_reserve`].
    pub fn try_reserve(&self, pages: usize) -> bool {
        self.with_budget(|b| b.try_reserve(pages))
    }

    /// Release pages reserved but not mapped.
    pub fn unreserve(&self, pages: usize) {
        self.with_budget(|b| b.unreserve(pages));
    }

    /// Promote reserved pages to committed.
    pub fn mark_committed(&self, pages: usize) -> Result<()> {
        self.with_budget(|b| b.mark_committed(pages))
    }

    /// Release pages that have been unmapped.
    pub fn mark_uncommitted(&self, pages: usize) {
        self.with_budget(|b| b.mark_uncommitted(pages));
    }

    /// Re-derive the budget from free device memory.
    pub fn recalibrate(
        &self,
        available_bytes: usize,
        safety_floor_bytes: usize,
        reset_hard_ceiling: bool,
    ) {
        self.with_budget(|b| {
            b.recalibrate(available_bytes, safety_floor_bytes, reset_hard_ceiling)
        });
    }

    /// Name the peers that may read this pool's pages directly. VMM mappings
    /// are device-private otherwise, so call before any arena grows. Unreachable
    /// peers are dropped with a warning, not refused.
    pub fn set_peer_devices(&mut self, peers: &[i32]) {
        self.peer_devices.clear();
        for &peer in peers {
            if peer == self.device_ordinal {
                continue;
            }
            let mut accessible: i32 = 0;
            let ok = unsafe { cudaDeviceCanAccessPeer(&mut accessible, self.device_ordinal, peer) };
            if ok != cudarc::runtime::sys::cudaError::cudaSuccess || accessible == 0 {
                tracing::warn!(
                    device = self.device_ordinal,
                    peer,
                    "elastic pool: peer access unavailable; arenas stay device-private"
                );
                continue;
            }
            self.peer_devices.push(peer);
        }
    }

    /// The peers named by [`Self::set_peer_devices`] that were reachable.
    #[must_use]
    pub fn peer_devices(&self) -> &[i32] {
        &self.peer_devices
    }

    /// Read/write descriptors for the owning device and every reachable peer.
    fn access_descriptors(&self) -> Vec<CUmemAccessDesc> {
        std::iter::once(self.device_ordinal)
            .chain(self.peer_devices.iter().copied())
            .map(|id| CUmemAccessDesc {
                location: memory_location(id),
                flags: CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            })
            .collect()
    }

    fn acquire_handle(&self, bytes: usize) -> Result<CUmemGenericAllocationHandle> {
        let prop = allocation_prop(self.device_ordinal);
        let mut handle: CUmemGenericAllocationHandle = 0;
        check_cu(
            unsafe { cuMemCreate(&mut handle, bytes, &prop, 0) },
            "cuMemCreate",
        )?;
        Ok(handle)
    }

    fn release_handle(&self, handle: CUmemGenericAllocationHandle) {
        if handle != 0 {
            ignore_in_drop(unsafe { cuMemRelease(handle) });
        }
    }
}

fn allocation_prop(device_ordinal: i32) -> CUmemAllocationProp {
    // Zeroed then filled: a reserved/win32-only tail must stay zero; naming every field
    // would break on a CUDA bump.
    let mut prop: CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location = memory_location(device_ordinal);
    prop
}

fn memory_location(device_ordinal: i32) -> cudarc::driver::sys::CUmemLocation {
    let mut loc: cudarc::driver::sys::CUmemLocation = unsafe { std::mem::zeroed() };
    loc.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    loc.id = device_ordinal;
    loc
}

/// A contiguous virtual range whose physical backing grows and shrinks under a
/// fixed base address, so tensors hold offsets across a shrink/regrow without rebasing.
#[derive(Debug)]
pub struct Arena<'p> {
    pool: &'p PhysicalPool,
    label: String,
    base: CUdeviceptr,
    max_bytes: usize,
    virtual_bytes: usize,
    map_unit_bytes: usize,
    /// Handles currently mapped, in address order: handle `i` backs `[base +
    /// i*map_unit, base + (i+1)*map_unit)`.
    mapped: Vec<CUmemGenericAllocationHandle>,
    /// Handles unmapped by a trim but not released, ready to re-map — trim/regrow
    /// churn is common and `cuMemCreate` is expensive, at the cost of keeping memory charged.
    cached: Vec<CUmemGenericAllocationHandle>,
}

impl<'p> Arena<'p> {
    /// Reserve a virtual range for an arena that will hold at most `max_bytes`,
    /// reserving twice that in *address space* to leave room to extend in place.
    pub fn new(pool: &'p PhysicalPool, max_bytes: usize, label: impl Into<String>) -> Result<Self> {
        let label = label.into();
        if max_bytes == 0 {
            return Err(Error::invalid(
                "Arena::new",
                "an arena needs a non-zero size",
            ));
        }
        let granularity = pool.allocation_granularity();
        let map_unit_bytes = pool
            .handle_bytes()
            .min(align_up(max_bytes, granularity))
            .max(granularity);
        // `checked_mul`, not `saturating_mul`: overflow falls back to the plain size,
        // not `usize::MAX`.
        let virtual_bytes = align_up(
            max_bytes.checked_mul(2).unwrap_or(max_bytes),
            map_unit_bytes,
        );

        let mut base: CUdeviceptr = 0;
        check_cu(
            unsafe { cuMemAddressReserve(&mut base, virtual_bytes, granularity, 0, 0) },
            "cuMemAddressReserve",
        )?;
        Ok(Self {
            pool,
            label,
            base,
            max_bytes,
            virtual_bytes,
            map_unit_bytes,
            mapped: Vec::new(),
            cached: Vec::new(),
        })
    }

    /// The fixed base address of the range.
    #[must_use]
    pub const fn base(&self) -> CUdeviceptr {
        self.base
    }

    /// The largest commit this arena will accept.
    #[must_use]
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// The unit growth happens in.
    #[must_use]
    pub const fn map_unit_bytes(&self) -> usize {
        self.map_unit_bytes
    }

    /// Bytes currently backed by physical memory.
    #[must_use]
    pub const fn committed_bytes(&self) -> usize {
        self.mapped.len() * self.map_unit_bytes
    }

    /// Handles held unmapped against a future regrow.
    #[must_use]
    pub const fn cached_handle_count(&self) -> usize {
        self.cached.len()
    }

    /// `bytes` rounded up to the map unit, or an error if it exceeds capacity.
    pub fn target_committed_bytes(&self, bytes: usize) -> Result<usize> {
        if bytes > self.max_bytes {
            return Err(Error::invalid(
                "Arena::ensure_committed",
                format!(
                    "{}: commit of {bytes} exceeds capacity {}",
                    self.label, self.max_bytes
                ),
            ));
        }
        Ok(align_up(bytes, self.map_unit_bytes))
    }

    /// Logical pages that reaching `bytes` would newly charge to the pool.
    /// Cached handles are already charged and don't count, or oscillation double-charges.
    pub fn physical_growth_pages(&self, bytes: usize) -> Result<usize> {
        let target = self.target_committed_bytes(bytes)?;
        let committed = self.committed_bytes();
        if target <= committed {
            return Ok(0);
        }
        let needed_handles = (target - committed) / self.map_unit_bytes;
        let new_handles = needed_handles.saturating_sub(self.cached.len());
        Ok(pages_for_bytes(
            new_handles * self.map_unit_bytes,
            LOGICAL_PAGE_BYTES,
        ))
    }

    /// Back at least `bytes` with physical memory. Reserves from the pool first
    /// so a refusal costs nothing; a partial failure rolls back fully, never half-grown.
    pub fn ensure_committed(&mut self, bytes: usize) -> Result<()> {
        let before = self.committed_bytes();
        let cached_before = self.cached.len();
        if self.target_committed_bytes(bytes)? <= before {
            return Ok(());
        }
        let pages = self.physical_growth_pages(bytes)?;
        if !self.pool.try_reserve(pages) {
            return Err(Error::invalid(
                "Arena::ensure_committed",
                format!("{}: shared physical pool budget exhausted", self.label),
            ));
        }
        match self
            .grow(bytes)
            .and_then(|()| self.pool.mark_committed(pages))
        {
            Ok(()) => Ok(()),
            Err(e) => {
                self.rollback(before, cached_before);
                self.pool.unreserve(pages);
                Err(e)
            }
        }
    }

    /// Drop physical backing above `bytes`, keeping the addresses reserved.
    pub fn trim_committed(&mut self, bytes: usize) -> Result<()> {
        let target = self.target_committed_bytes(bytes)?;
        self.release_tail(target);
        Ok(())
    }

    /// How many unmapped handles this arena keeps against a regrow: one while
    /// holding at least two (hysteresis), zero otherwise — a cached handle still costs budget.
    const fn cache_goal(target_handles: usize) -> usize {
        if target_handles >= 2 { 1 } else { 0 }
    }

    /// Unmap everything above `target_bytes`, then reconcile cache and pool.
    /// Only handles released to the driver are reported uncommitted — cached ones stay held.
    fn release_tail(&mut self, target_bytes: usize) {
        let target = align_up(target_bytes, self.map_unit_bytes);
        let goal = Self::cache_goal(target / self.map_unit_bytes);
        let mut released_bytes = 0usize;

        while self.committed_bytes() > target && !self.mapped.is_empty() {
            let index = self.mapped.len() - 1;
            let address = self.base + (index * self.map_unit_bytes) as CUdeviceptr;
            ignore_in_drop(unsafe { cuMemUnmap(address, self.map_unit_bytes) });
            let Some(handle) = self.mapped.pop() else {
                break;
            };
            if self.cached.len() < goal {
                self.cached.push(handle);
            } else {
                self.pool.release_handle(handle);
                released_bytes += self.map_unit_bytes;
            }
        }

        // A trim also shrinks a cache left over from a larger arena, so hysteresis
        // can't grow across trims.
        while self.cached.len() > goal {
            let Some(handle) = self.cached.pop() else {
                break;
            };
            self.pool.release_handle(handle);
            released_bytes += self.map_unit_bytes;
        }

        self.pool
            .mark_uncommitted(pages_for_bytes(released_bytes, LOGICAL_PAGE_BYTES));
    }

    fn grow(&mut self, bytes: usize) -> Result<()> {
        let target = self.target_committed_bytes(bytes)?;
        let access = self.pool.access_descriptors();
        while self.committed_bytes() < target {
            let reused = self.cached.pop();
            let handle = match reused {
                Some(h) => h,
                None => self.pool.acquire_handle(self.map_unit_bytes)?,
            };
            let address = self.base + (self.mapped.len() * self.map_unit_bytes) as CUdeviceptr;

            let mapped = check_cu(
                unsafe { cuMemMap(address, self.map_unit_bytes, 0, handle, 0) },
                "cuMemMap",
            );
            let outcome = mapped.and_then(|()| {
                check_cu(
                    unsafe {
                        cuMemSetAccess(address, self.map_unit_bytes, access.as_ptr(), access.len())
                    },
                    "cuMemSetAccess",
                )
                .inspect_err(|_| {
                    // The map landed but the access failed, so this address is
                    // mapped and must be undone before the handle goes back.
                    ignore_in_drop(unsafe { cuMemUnmap(address, self.map_unit_bytes) });
                })
            });

            if let Err(e) = outcome {
                // Put the handle back where it came from: the cache, or released
                // outright. Getting this wrong leaks on every failed grow.
                if reused.is_some() {
                    self.cached.push(handle);
                } else {
                    self.pool.release_handle(handle);
                }
                return Err(e);
            }
            self.mapped.push(handle);
        }
        Ok(())
    }

    /// Undo a failed growth: unmap back to `bytes`, restoring the cache to its
    /// prior size. Nothing here is reported uncommitted — these pages were only
    /// `held`, and `ensure_committed`'s `unreserve` returns them; routing through
    /// `release_tail` would double-credit.
    fn rollback(&mut self, bytes: usize, cached_handle_count: usize) {
        let target_handles = align_up(bytes, self.map_unit_bytes) / self.map_unit_bytes;
        while self.mapped.len() > target_handles {
            let index = self.mapped.len() - 1;
            let address = self.base + (index * self.map_unit_bytes) as CUdeviceptr;
            ignore_in_drop(unsafe { cuMemUnmap(address, self.map_unit_bytes) });
            let Some(handle) = self.mapped.pop() else {
                break;
            };
            if self.cached.len() < cached_handle_count {
                self.cached.push(handle);
            } else {
                self.pool.release_handle(handle);
            }
        }
    }
}

impl Drop for Arena<'_> {
    fn drop(&mut self) {
        // `release_tail(0)` has `cache_goal(0) == 0`, so it unmaps and drains
        // everything back to the pool.
        self.release_tail(0);
        if self.base != 0 {
            ignore_in_drop(unsafe { cuMemAddressFree(self.base, self.virtual_bytes) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `PoolBudget` needs no GPU, so it gets the tests.

    #[test]
    fn a_budget_is_whole_pages_and_partial_pages_are_dropped() {
        let b = PoolBudget::new(5 * LOGICAL_PAGE_BYTES + 1);
        assert_eq!(b.budget_pages(), 5, "a partial page is not a page");
        assert_eq!(b.free_pages(), 5);
        assert_eq!(PoolBudget::new(LOGICAL_PAGE_BYTES - 1).budget_pages(), 0);
    }

    #[test]
    fn pages_for_bytes_rounds_up_but_zero_stays_zero() {
        assert_eq!(pages_for_bytes(0, LOGICAL_PAGE_BYTES), 0);
        assert_eq!(pages_for_bytes(1, LOGICAL_PAGE_BYTES), 1);
        assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES, LOGICAL_PAGE_BYTES), 1);
        assert_eq!(
            pages_for_bytes(LOGICAL_PAGE_BYTES + 1, LOGICAL_PAGE_BYTES),
            2
        );
    }

    #[test]
    fn held_pages_charge_against_the_budget_before_they_are_committed() {
        let mut b = PoolBudget::new(4 * LOGICAL_PAGE_BYTES);
        assert!(b.try_reserve(3));
        assert_eq!(b.free_pages(), 1, "held pages are charged immediately");
        assert!(
            !b.try_reserve(2),
            "must not hand out pages already promised"
        );
        assert!(b.try_reserve(1));
        assert_eq!(b.free_pages(), 0);
    }

    #[test]
    fn committing_moves_pages_from_held_without_changing_the_total_charge() {
        let mut b = PoolBudget::new(4 * LOGICAL_PAGE_BYTES);
        assert!(b.try_reserve(3));
        let charged = b.charged_pages();
        b.mark_committed(2).unwrap();
        assert_eq!(
            b.charged_pages(),
            charged,
            "a commit is a move, not a new charge"
        );
        assert_eq!(b.held_pages(), 1);
        assert_eq!(b.committed_pages(), 2);
    }

    #[test]
    fn committing_more_than_was_reserved_is_an_error_not_an_overrun() {
        let mut b = PoolBudget::new(4 * LOGICAL_PAGE_BYTES);
        assert!(b.try_reserve(1));
        let err = b.mark_committed(2).unwrap_err();
        assert_eq!(err.call(), "PoolBudget::mark_committed");
        assert_eq!(b.committed_pages(), 0, "the failed commit changed nothing");
        assert_eq!(b.held_pages(), 1);
    }

    #[test]
    fn a_failed_growth_returns_every_page_it_reserved() {
        let start = PoolBudget::new(8 * LOGICAL_PAGE_BYTES);
        let mut b = start.clone();
        assert!(b.try_reserve(5));
        b.unreserve(5);
        assert_eq!(b.free_pages(), start.free_pages());
        assert_eq!(b.charged_pages(), 0);
    }

    #[test]
    fn releasing_bumps_the_generation_so_a_refused_caller_knows_to_retry() {
        let mut b = PoolBudget::new(4 * LOGICAL_PAGE_BYTES);
        assert!(b.try_reserve(4));
        let g = b.generation();
        assert!(!b.try_reserve(1), "full");
        assert_eq!(b.generation(), g, "a refusal is not a change");
        b.unreserve(4);
        assert!(
            b.generation() > g,
            "space came back; a cached refusal is now stale"
        );
    }

    #[test]
    fn over_releasing_saturates_rather_than_underflowing() {
        // Runs from unwinding/Drop; a `usize` underflow would turn a small bug into a
        // budget of 2^64 pages.
        let mut b = PoolBudget::new(4 * LOGICAL_PAGE_BYTES);
        assert!(b.try_reserve(1));
        b.unreserve(100);
        assert_eq!(b.held_pages(), 0);
        b.mark_uncommitted(100);
        assert_eq!(b.committed_pages(), 0);
    }

    #[test]
    fn recalibration_never_drops_the_budget_below_what_is_already_charged() {
        // Keeps `free_pages` from wrapping when the device has less memory than the pool is using.
        let mut b = PoolBudget::new(16 * LOGICAL_PAGE_BYTES);
        assert!(b.try_reserve(10));
        b.mark_committed(10).unwrap();
        b.recalibrate(0, 0, false);
        assert_eq!(
            b.budget_pages(),
            10,
            "cannot un-commit pages by recalibrating"
        );
        assert_eq!(b.free_pages(), 0);
    }

    #[test]
    fn recalibration_adds_free_memory_beyond_the_safety_floor() {
        let mut b = PoolBudget::new(0);
        assert_eq!(b.budget_pages(), 0);
        b.recalibrate(10 * LOGICAL_PAGE_BYTES, 4 * LOGICAL_PAGE_BYTES, false);
        assert_eq!(b.budget_pages(), 6, "10 free minus a 4-page floor");
    }

    #[test]
    fn the_safety_floor_can_exceed_available_memory_without_underflowing() {
        let mut b = PoolBudget::new(0);
        b.recalibrate(LOGICAL_PAGE_BYTES, 100 * LOGICAL_PAGE_BYTES, false);
        assert_eq!(b.budget_pages(), 0);
    }

    #[test]
    fn the_hard_ceiling_survives_a_transient_dip_unless_it_is_reset() {
        let mut b = PoolBudget::new(0);
        b.recalibrate(20 * LOGICAL_PAGE_BYTES, 0, false);
        assert_eq!(b.hard_budget_pages(), 20);

        b.recalibrate(5 * LOGICAL_PAGE_BYTES, 0, false);
        assert_eq!(b.budget_pages(), 5);
        assert_eq!(b.hard_budget_pages(), 20, "the high-water mark remembers");

        b.recalibrate(5 * LOGICAL_PAGE_BYTES, 0, true);
        assert_eq!(b.hard_budget_pages(), 5);
    }

    #[test]
    fn align_up_rounds_to_the_unit_and_leaves_zero_alone() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
    }

    #[test]
    fn the_handle_cache_is_capped_at_one_and_empties_completely_on_teardown() {
        assert_eq!(Arena::cache_goal(0), 0, "teardown keeps nothing");
        assert_eq!(
            Arena::cache_goal(1),
            0,
            "an arena down to one unit is not oscillating"
        );
        assert_eq!(Arena::cache_goal(2), 1);
        assert_eq!(Arena::cache_goal(1000), 1, "capped, not proportional");
    }
}
