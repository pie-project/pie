//! A budgeted pool of physical pages backing virtual arenas whose backing
//! grows and shrinks under a fixed address.

use crate::error::{Fault, Result};

/// The accounting unit the budget is counted in.
pub const LOGICAL_PAGE_BYTES: u64 = 2 * 1024 * 1024;

/// The largest quantum an arena grows and trims by.
pub const MAP_UNIT_BYTES: u64 = 32 * 1024 * 1024;

/// Bounds an arena's growth quantum so it stays proportional to the arena's
/// size and never coarser than the map unit.
const HANDLES_PER_ARENA: u64 = 256;

/// Reserved out of free memory so a later driver allocation (cuBLAS
/// workspace, NCCL buffer, module load) still has room.
const SAFETY_FLOOR_BYTES: u64 = 128 * 1024 * 1024;

/// The floor this card holds back, shared by the pool, accounting, and tests.
#[must_use]
pub const fn safety_floor_bytes(total: u64) -> u64 {
    let tenth = total / 10;
    if SAFETY_FLOOR_BYTES < tenth {
        SAFETY_FLOOR_BYTES
    } else {
        tenth
    }
}

/// What the elastic pool may hold.
///
/// ```text
/// budget = total x utilization - (total - free) - floor
/// ```
///
/// A fraction under what is already on the card answers zero rather than
/// wrapping.
#[must_use]
pub fn budget_bytes(free: u64, total: u64, utilization: f64) -> u64 {
    // Clamp again: a NaN must not become a budget.
    let fraction = if utilization.is_finite() {
        utilization.clamp(0.0, 1.0)
    } else {
        1.0
    };
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a byte count of a GPU card is far inside f64's exact integer \
                  range, and the product is floored back into u64 deliberately"
    )]
    let ceiling = (total as f64 * fraction) as u64;
    let already = total.saturating_sub(free);
    ceiling
        .saturating_sub(already)
        .saturating_sub(safety_floor_bytes(total))
}

/// How many logical pages `bytes` occupies.
#[must_use]
pub const fn pages_for_bytes(bytes: u64) -> u64 {
    bytes.div_ceil(LOGICAL_PAGE_BYTES)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if value == 0 || alignment == 0 {
        return 0;
    }
    value.div_ceil(alignment) * alignment
}

/// A budgeted supply of physical pages. `held` is promised-not-yet-mapped
/// pages, `committed` is mapped pages; the budget charges their sum.
#[derive(Debug)]
pub struct PhysicalPool {
    device: i32,
    /// `cuMemGetAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM)`.
    granularity: u64,
    /// One handle's bytes, rounded up to the granularity.
    handle_bytes: u64,
    /// Soft budget, in logical pages (recalibrated when a commit needs
    /// growth).
    budget_pages: u64,
    /// The hard ceiling, in logical pages. Never lowered.
    hard_pages: u64,
    /// Promised, not yet mapped.
    held_pages: u64,
    /// Mapped.
    committed_pages: u64,
    /// High water mark of `committed_pages`.
    high_water_pages: u64,
    /// Operator's memory fraction (`gpu_mem_utilization`); `1.0` is the
    /// whole card.
    utilization: f64,
}

impl PhysicalPool {
    /// Opens the pool for `device` at the operator's memory fraction
    /// `utilization`.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] with no runtime selected, [`Fault::Device`] for
    /// the granularity query or the memory query.
    pub fn open(device: i32, utilization: f64) -> Result<PhysicalPool> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            let granularity = allocation_granularity(device)?;
            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: two live locals; the call only writes them.
            let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
            crate::device::ctx::check("cudaMemGetInfo", asked)?;
            let budget = budget_bytes(free as u64, total as u64, utilization);
            let handle_bytes = align_up(MAP_UNIT_BYTES.max(granularity), granularity);
            let pages = budget / LOGICAL_PAGE_BYTES;
            Ok(PhysicalPool {
                device,
                granularity,
                handle_bytes,
                budget_pages: pages,
                hard_pages: pages,
                held_pages: 0,
                committed_pages: 0,
                high_water_pages: 0,
                utilization,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, utilization);
            Err(Fault::Runtimeless)
        }
    }

    /// A pool with a stated budget and no device — for tests and runtimeless
    /// builds.
    #[must_use]
    pub fn stated(budget_bytes: u64) -> PhysicalPool {
        let pages = budget_bytes / LOGICAL_PAGE_BYTES;
        PhysicalPool {
            device: 0,
            granularity: LOGICAL_PAGE_BYTES,
            handle_bytes: MAP_UNIT_BYTES,
            budget_pages: pages,
            hard_pages: pages,
            held_pages: 0,
            committed_pages: 0,
            high_water_pages: 0,
            // No device behind a stated pool.
            utilization: 1.0,
        }
    }

    /// The operator's fraction of the card this pool was opened at.
    #[must_use]
    pub const fn utilization(&self) -> f64 {
        self.utilization
    }

    /// Bytes one logical page holds.
    #[must_use]
    pub const fn page_bytes(&self) -> u64 {
        LOGICAL_PAGE_BYTES
    }

    /// The soft budget, in logical pages.
    #[must_use]
    pub const fn budget_pages(&self) -> u64 {
        self.budget_pages
    }

    /// The hard ceiling, in logical pages.
    #[must_use]
    pub const fn hard_pages(&self) -> u64 {
        self.hard_pages
    }

    /// Logical pages under a mapping right now.
    #[must_use]
    pub const fn committed_pages(&self) -> u64 {
        self.committed_pages
    }

    /// The most that has ever been.
    #[must_use]
    pub const fn high_water_pages(&self) -> u64 {
        self.high_water_pages
    }

    /// One handle's bytes — the quantum an arena grows and trims by.
    #[must_use]
    pub const fn handle_bytes(&self) -> u64 {
        self.handle_bytes
    }

    /// Promise `pages`, or say no. Charged against committed + held, so a
    /// promise made and not yet mapped still counts. `false` means nothing
    /// was touched.
    pub fn try_reserve(&mut self, pages: u64) -> bool {
        let charged = self.committed_pages + self.held_pages;
        if pages > self.budget_pages.saturating_sub(charged.min(self.budget_pages)) {
            return false;
        }
        self.held_pages += pages;
        true
    }

    /// Give a promise back unused.
    pub fn unreserve(&mut self, pages: u64) {
        self.held_pages -= self.held_pages.min(pages);
    }

    /// A promise became a mapping.
    pub fn mark_committed(&mut self, pages: u64) {
        let promised = self.held_pages.min(pages);
        self.held_pages -= promised;
        self.committed_pages += pages;
        self.high_water_pages = self.high_water_pages.max(self.committed_pages);
    }

    /// A mapping went away.
    pub fn mark_uncommitted(&mut self, pages: u64) {
        self.committed_pages -= self.committed_pages.min(pages);
    }

    /// Re-reads what the card has left, and moves the soft budget. The hard
    /// ceiling only ever rises.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the query; a runtimeless build leaves the budget
    /// where `stated` put it and answers `Ok`.
    pub fn recalibrate(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: two live locals; the call only writes them.
            let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
            crate::device::ctx::check("cudaMemGetInfo", asked)?;
            let available =
                budget_bytes(free as u64, total as u64, self.utilization) / LOGICAL_PAGE_BYTES;
            let charged = self.committed_pages + self.held_pages;
            self.budget_pages = charged.saturating_add(available).max(charged);
            self.hard_pages = self.hard_pages.max(self.budget_pages);
        }
        Ok(())
    }

    /// One physical allocation of `bytes`.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    fn acquire_handle(&self, bytes: u64) -> Result<u64> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys as dr;

            let prop = allocation_prop(self.device);
            let mut handle: dr::CUmemGenericAllocationHandle = 0;
            // SAFETY: `handle` and `prop` are live locals; the handle is
            // released exactly once, by `release_handle`.
            let made = unsafe { dr::cuMemCreate(&raw mut handle, bytes as usize, &raw const prop, 0) };
            said("cuMemCreate", made)?;
            Ok(handle)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = bytes;
            Err(Fault::Runtimeless)
        }
    }

    /// Release one physical allocation. Infallible: used on rollback/trim
    /// paths.
    fn release_handle(&self, handle: u64) {
        #[cfg(feature = "cuda")]
        if handle != 0 {
            use cudarc::driver::sys as dr;

            // SAFETY: the handle came from this pool's own `cuMemCreate` and
            // is released exactly once.
            unsafe {
                let _ = dr::cuMemRelease(handle);
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = handle;
        }
    }
}

/// A fixed virtual range whose backing grows and trims from the tail; `base`
/// never changes after `reserve`.
///
/// Each kv-row plane gets its own arena, so a partial commit can't leave
/// pages mid-row unmapped.
#[derive(Debug)]
pub struct Arena {
    label: &'static str,
    /// `CUdeviceptr`. Fixed from `reserve` to `Drop`.
    base: u64,
    /// The ceiling this arena may be asked to commit to.
    max_bytes: u64,
    /// The address range actually reserved: `max_bytes` rounded up to the map
    /// unit.
    virtual_bytes: u64,
    /// This arena's handle size: the pool's, or the arena's own size if
    /// smaller.
    map_unit: u64,
    /// Mapped handles, in address order. `handles[i]` backs
    /// `base + i * map_unit`.
    handles: Vec<u64>,
    /// Unmapped handles cached for reuse, avoiding a `cuMemCreate` round trip
    /// on the next grow.
    cached: Vec<u64>,
    /// The most `committed_bytes` has ever been.
    high_water: u64,
}

impl Arena {
    /// Reserve `max_bytes` of address space. Nothing is mapped yet; `base`
    /// is valid immediately.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], [`Fault::Device`] for the reservation.
    pub fn reserve(pool: &PhysicalPool, max_bytes: u64, label: &'static str) -> Result<Arena> {
        let ceiling = align_up(max_bytes, pool.granularity).max(pool.granularity);
        let map_unit = align_up(
            (max_bytes / HANDLES_PER_ARENA)
                .clamp(pool.granularity, pool.handle_bytes)
                .min(ceiling),
            pool.granularity,
        )
        .max(pool.granularity);
        let virtual_bytes = align_up(max_bytes, map_unit);
        if virtual_bytes == 0 {
            return Ok(Arena {
                label,
                base: 0,
                max_bytes: 0,
                virtual_bytes: 0,
                map_unit,
                handles: Vec::new(),
                cached: Vec::new(),
                high_water: 0,
            });
        }
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys as dr;

            let mut base: dr::CUdeviceptr = 0;
            // SAFETY: `base` is a live local; the range is this arena's own
            // and is freed exactly once, in `Drop`.
            let reserved = unsafe {
                dr::cuMemAddressReserve(
                    &raw mut base,
                    virtual_bytes as usize,
                    pool.granularity as usize,
                    0,
                    0,
                )
            };
            said("cuMemAddressReserve", reserved)?;
            Ok(Arena {
                label,
                base,
                max_bytes,
                virtual_bytes,
                map_unit,
                handles: Vec::new(),
                cached: Vec::new(),
                high_water: 0,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// The base address, fixed for the arena's life.
    #[must_use]
    pub const fn base(&self) -> u64 {
        self.base
    }

    /// The ceiling this arena's address space was reserved at.
    #[must_use]
    pub const fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// How far past [`Arena::base`] is readable right now.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.handles.len() as u64 * self.map_unit
    }

    /// The most that has ever been.
    #[must_use]
    pub const fn high_water_bytes(&self) -> u64 {
        self.high_water
    }

    /// `bytes`, rounded up to this arena's map unit.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a target past the arena's ceiling.
    pub fn target_bytes(&self, bytes: u64) -> Result<u64> {
        if bytes > self.max_bytes {
            return Err(Fault::Ceiling {
                what: "bytes of an elastic arena",
                need: bytes,
                have: self.max_bytes,
            });
        }
        Ok(align_up(bytes, self.map_unit))
    }

    /// Pages needed from the pool to reach `bytes`; cached handles cost
    /// nothing to re-map.
    fn growth_pages(&self, target: u64) -> u64 {
        let committed = self.committed_bytes();
        if target <= committed {
            return 0;
        }
        let needed = (target - committed) / self.map_unit;
        let fresh = needed.saturating_sub(self.cached.len() as u64);
        pages_for_bytes(fresh * self.map_unit)
    }

    /// Map physical pages under the tail until `target` is readable.
    ///
    /// Caller has already reserved the pages; this only maps, and rolls back
    /// to the start on partial failure.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], [`Fault::Device`] for the map or the access
    /// descriptor.
    fn grow(&mut self, pool: &PhysicalPool, target: u64) -> Result<()> {
        let before = self.committed_bytes();
        let cached_before = self.cached.len();
        if target <= before {
            return Ok(());
        }
        while self.committed_bytes() < target {
            let reused = self.cached.pop();
            let handle = match reused {
                Some(handle) => handle,
                None => match pool.acquire_handle(self.map_unit) {
                    Ok(handle) => handle,
                    Err(fault) => {
                        self.rollback(pool, before, cached_before);
                        return Err(fault);
                    }
                },
            };
            let at = self.base + self.handles.len() as u64 * self.map_unit;
            match self.map(pool, at, handle) {
                Ok(()) => self.handles.push(handle),
                Err(fault) => {
                    if reused.is_some() {
                        self.cached.push(handle);
                    } else {
                        pool.release_handle(handle);
                    }
                    self.rollback(pool, before, cached_before);
                    return Err(fault);
                }
            }
        }
        self.high_water = self.high_water.max(self.committed_bytes());
        Ok(())
    }

    /// One handle, mapped and made readable at `at`.
    fn map(&self, pool: &PhysicalPool, at: u64, handle: u64) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys as dr;

            // SAFETY: `at` is inside this arena's own reserved range and is
            // not currently mapped; `handle` is this pool's own allocation of
            // exactly `map_unit` bytes.
            let mapped = unsafe { dr::cuMemMap(at, self.map_unit as usize, 0, handle, 0) };
            said("cuMemMap", mapped)?;
            let desc = access_desc(pool.device);
            // SAFETY: the range was just mapped; `desc` is a live local.
            let granted =
                unsafe { dr::cuMemSetAccess(at, self.map_unit as usize, &raw const desc, 1) };
            if let Err(fault) = said("cuMemSetAccess", granted) {
                // SAFETY: the range is the one just mapped.
                unsafe {
                    let _ = dr::cuMemUnmap(at, self.map_unit as usize);
                }
                return Err(fault);
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (pool, at, handle);
            Err(Fault::Runtimeless)
        }
    }

    /// Undo a partial grow: unmap back to `bytes`, caching handles up to
    /// `cached_goal` and releasing the rest.
    fn rollback(&mut self, pool: &PhysicalPool, bytes: u64, cached_goal: usize) {
        let target = align_up(bytes, self.map_unit);
        while self.committed_bytes() > target && !self.handles.is_empty() {
            let at = self.base + (self.handles.len() as u64 - 1) * self.map_unit;
            self.unmap(at);
            let handle = self.handles.pop().unwrap_or(0);
            if self.cached.len() < cached_goal {
                self.cached.push(handle);
            } else {
                pool.release_handle(handle);
            }
        }
    }

    /// Unmap the tail down to `bytes` and tell the pool.
    ///
    /// Keeps one handle cached (not released) when 2+ remain, since a trim is
    /// usually followed by a grow. Returns pages actually handed back.
    pub fn release_tail(&mut self, pool: &mut PhysicalPool, bytes: u64) -> u64 {
        let target = align_up(bytes, self.map_unit);
        let target_handles = if self.map_unit == 0 {
            0
        } else {
            target / self.map_unit
        };
        let cache_goal = usize::from(target_handles >= 2);
        let mut released = 0u64;
        while self.committed_bytes() > target && !self.handles.is_empty() {
            let at = self.base + (self.handles.len() as u64 - 1) * self.map_unit;
            self.unmap(at);
            let handle = self.handles.pop().unwrap_or(0);
            if self.cached.len() < cache_goal {
                self.cached.push(handle);
            } else {
                pool.release_handle(handle);
                released += self.map_unit;
            }
        }
        while self.cached.len() > cache_goal {
            if let Some(handle) = self.cached.pop() {
                pool.release_handle(handle);
                released += self.map_unit;
            }
        }
        let pages = pages_for_bytes(released);
        pool.mark_uncommitted(pages);
        pages
    }

    fn unmap(&self, at: u64) {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys as dr;

            // SAFETY: `at` is a range this arena mapped and has not unmapped.
            unsafe {
                let _ = dr::cuMemUnmap(at, self.map_unit as usize);
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = at;
        }
    }

    /// An address `offset` bytes in, checked against committed (not just
    /// reserved) bytes — a fault past the committed edge inside a captured
    /// graph is unattributable.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] naming this arena.
    pub fn span(&self, offset: u64, len: u64) -> Result<u64> {
        let end = offset.saturating_add(len);
        if end > self.committed_bytes() {
            return Err(Fault::Ceiling {
                what: self.label,
                need: end,
                have: self.committed_bytes(),
            });
        }
        Ok(self.base + offset)
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys as dr;

            while let Some(handle) = self.handles.pop() {
                let at = self.base + self.handles.len() as u64 * self.map_unit;
                self.unmap(at);
                // SAFETY: this arena's own handle, released once.
                unsafe {
                    let _ = dr::cuMemRelease(handle);
                }
            }
            for handle in self.cached.drain(..) {
                // SAFETY: as above.
                unsafe {
                    let _ = dr::cuMemRelease(handle);
                }
            }
            if self.base != 0 {
                // SAFETY: the range this arena reserved, freed once.
                unsafe {
                    let _ = dr::cuMemAddressFree(self.base, self.virtual_bytes as usize);
                }
            }
        }
    }
}

/// What one arena is asked to be, for the atomic commit.
#[derive(Debug)]
pub struct Target<'a> {
    pub arena: &'a mut Arena,
    /// Bytes that must be readable afterwards.
    pub bytes: u64,
}

/// The answer to one atomic multi-arena commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Commit {
    /// Every arena is at or past its target. Stream work may proceed.
    Committed,
    /// Not right now; nothing was touched. `required`/`budget` are logical
    /// pages.
    Exhausted {
        /// Logical pages the targets come to, in total.
        required: u64,
        /// Logical pages the soft budget allows.
        budget: u64,
    },
    /// Never, on this device, for this load. Past the hard ceiling.
    Impossible {
        /// Logical pages the targets come to, in total.
        required: u64,
        /// Logical pages the hard ceiling allows.
        ceiling: u64,
    },
}

/// Admission gate: brings every arena to its target, or none of them.
///
/// Priced against the hard ceiling then the soft budget before anything
/// maps, so a refusal touches nothing; `required` is each arena's total, not
/// its growth.
///
/// # Errors
///
/// [`Fault::Ceiling`] for a target past an arena's own ceiling,
/// [`Fault::Device`] for a map that failed after admission.
pub fn commit_atomically(pool: &mut PhysicalPool, targets: &mut [Target<'_>]) -> Result<Commit> {
    let mut required = 0u64;
    let mut growth = 0u64;
    let mut wanted = Vec::with_capacity(targets.len());
    for target in targets.iter() {
        let bytes = target.arena.target_bytes(target.bytes)?;
        required = required.saturating_add(pages_for_bytes(bytes));
        growth = growth.saturating_add(target.arena.growth_pages(bytes));
        wanted.push(bytes);
    }
    if growth == 0 {
        // Already fully mapped: nothing to check or reserve.
        return Ok(Commit::Committed);
    }
    // Recalibrate before both refusals, so freed memory isn't reported
    // Impossible.
    pool.recalibrate()?;
    if required > pool.hard_pages() {
        return Ok(Commit::Impossible {
            required,
            ceiling: pool.hard_pages(),
        });
    }
    if !pool.try_reserve(growth) {
        return Ok(Commit::Exhausted {
            required,
            budget: pool.budget_pages(),
        });
    }
    // Snapshot for rollback on partial failure.
    let was: Vec<(u64, usize)> = targets
        .iter()
        .map(|target| (target.arena.committed_bytes(), target.arena.cached.len()))
        .collect();
    let mut done = 0usize;
    for (target, bytes) in targets.iter_mut().zip(&wanted) {
        if let Err(fault) = target.arena.grow(pool, *bytes) {
            // The failed arena already rolled itself back; unwind the ones
            // before it, tail first.
            for (undone, &(bytes, cached)) in targets[..done].iter_mut().zip(&was[..done]).rev() {
                undone.arena.rollback(pool, bytes, cached);
            }
            pool.unreserve(growth);
            return Err(fault);
        }
        done += 1;
    }
    pool.mark_committed(growth);
    Ok(Commit::Committed)
}

/// Maps a CUDA driver status to a `Fault`.
#[cfg(feature = "cuda")]
fn said(call: &'static str, code: cudarc::driver::sys::CUresult) -> Result<()> {
    if code == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: code as i32,
        })
    }
}

#[cfg(feature = "cuda")]
fn allocation_prop(device: i32) -> cudarc::driver::sys::CUmemAllocationProp {
    use cudarc::driver::sys as dr;

    let mut prop: dr::CUmemAllocationProp = unsafe { core::mem::zeroed() };
    prop.type_ = dr::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop
}

#[cfg(feature = "cuda")]
fn access_desc(device: i32) -> cudarc::driver::sys::CUmemAccessDesc {
    use cudarc::driver::sys as dr;

    let mut desc: dr::CUmemAccessDesc = unsafe { core::mem::zeroed() };
    desc.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = dr::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc
}

#[cfg(feature = "cuda")]
fn allocation_granularity(device: i32) -> Result<u64> {
    use cudarc::driver::sys as dr;

    let prop = allocation_prop(device);
    let mut granularity: usize = 0;
    // SAFETY: two live locals; the call only reads `prop` and writes
    // `granularity`.
    let asked = unsafe {
        dr::cuMemGetAllocationGranularity(
            &raw mut granularity,
            &raw const prop,
            dr::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    };
    said("cuMemGetAllocationGranularity", asked)?;
    Ok(granularity as u64)
}

#[cfg(test)]
mod tests {
    use super::{LOGICAL_PAGE_BYTES, PhysicalPool, pages_for_bytes};

    // A promise not yet mapped still charges the budget.
    #[test]
    fn a_promise_charges_the_budget_before_it_is_a_mapping() {
        let mut pool = PhysicalPool::stated(10 * LOGICAL_PAGE_BYTES);
        assert_eq!(pool.budget_pages(), 10);
        assert!(pool.try_reserve(6));
        assert_eq!(pool.committed_pages(), 0);
        assert!(!pool.try_reserve(5));
        assert!(pool.try_reserve(4));
        pool.mark_committed(10);
        assert_eq!(pool.committed_pages(), 10);
        assert_eq!(pool.high_water_pages(), 10);
        assert!(!pool.try_reserve(1));
    }

    // Bytes round up to whole pages.
    #[test]
    fn a_partial_page_is_a_whole_page() {
        assert_eq!(pages_for_bytes(0), 0);
        assert_eq!(pages_for_bytes(1), 1);
        assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES), 1);
        assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES + 1), 2);
    }

}
