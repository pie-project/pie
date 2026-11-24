#![allow(unsafe_code)]
use engine::SAFETY_FLOOR_BYTES;

use crate::error::{Fault, Result};

pub const LOGICAL_PAGE_BYTES: u64 = 2 * 1024 * 1024;

pub const MAP_UNIT_BYTES: u64 = 32 * 1024 * 1024;

const HANDLES_PER_ARENA: u64 = 256;

/// The handle a flat arena's units carry: they are backed by the arena's
/// one allocation, and own nothing to release.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
const FLAT: u64 = u64::MAX;

/// Whether arenas map pages on demand through the virtual memory API, or
/// each is backed flat by one allocation on its first commit. ROCm backs
/// them flat: on some runtime and kernel driver pairs `hipMemSetAccess`
/// blocks forever rather than failing, which no probe can tell from a slow
/// call. A flat arena gives nothing back until it drops.
#[must_use]
pub const fn mapped_on_demand() -> bool {
    !cfg!(feature = "rocm")
}

/// What the pool holds back from the card for the driver: on a mapped pool,
/// what the driver allocates beside it on its own while serving. A flat
/// range holds back nothing: it is one allocation made once the load has
/// taken what it takes beside it (side streams, library handles), and what
/// arming allocates after is the bodies record's hold, which the fit leaves
/// out of the range by name; once arming ends nothing allocates on the
/// device.
#[must_use]
pub const fn safety_floor_bytes() -> u64 {
    if mapped_on_demand() {
        SAFETY_FLOOR_BYTES
    } else {
        0
    }
}

/// What a load may still take of a card with `free` of `total` bytes free
/// under `allowance`, past the driver's safety floor.
#[must_use]
pub fn budget_bytes(free: u64, total: u64, allowance: engine::Allowance) -> u64 {
    allowance
        .room(total, total.saturating_sub(free))
        .saturating_sub(safety_floor_bytes())
}

/// What a card of `total` bytes may hold in all while a load under
/// `allowance` runs, resolved once against the `held` bytes it had when the
/// load began: the card itself, and no more than a stated budget past what
/// was held then. A later reading's room is this less what the card then
/// holds ([`room_under`]), so a stated budget counts everything the load
/// has taken since, its weights and what the driver holds for its graph
/// bodies among it. No share of the card is left aside for those: the
/// load's last fit reads the card once they are on it.
#[must_use]
pub fn ceiling_of(total: u64, held: u64, allowance: engine::Allowance) -> u64 {
    allowance
        .budget
        .map_or(total, |budget| total.min(held.saturating_add(budget)))
}

/// What may still be taken under `ceiling` of a card with `free` of
/// `total` bytes free, past the driver's safety floor.
#[must_use]
pub fn room_under(ceiling: u64, free: u64, total: u64) -> u64 {
    ceiling
        .min(total)
        .saturating_sub(total.saturating_sub(free))
        .saturating_sub(safety_floor_bytes())
}

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

#[derive(Debug)]
pub struct PhysicalPool {
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    device: i32,
    granularity: u64,
    handle_bytes: u64,
    budget_pages: u64,
    hard_pages: u64,
    pledged_pages: u64,
    held_pages: u64,
    committed_pages: u64,
    /// What the card may hold in all under the load's allowance, resolved
    /// when the load began ([`ceiling_of`]).
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    ceiling: u64,
    /// What the driver holds on the card beside the pool that the device's
    /// free figure does not count: on a flat range, the wave save areas of
    /// this process's compute queues, read when the pool opens (every queue
    /// a load makes is made before: `kernels_rocm::hip::open`, the fan).
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    beside: Beside,
}

/// What the driver holds beside a pool: the device's free figure already
/// leaves it out (engine-cuda's `cudaMemGetInfo` on ROCm); the pool states
/// it, for the load's line and a refusal.
#[derive(Debug, Clone)]
pub struct Beside {
    pub said: String,
}

impl Beside {
    const NONE: Beside = Beside {
        said: String::new(),
    };

    /// On a flat range, the save areas of this process's compute queues;
    /// a mapped pool holds back its safety floor instead.
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    fn read() -> Result<Beside> {
        if mapped_on_demand() {
            return Ok(Beside::NONE);
        }
        #[cfg(feature = "rocm")]
        {
            // A load opens its fan's side streams before its pools, past the
            // hardware queues HIP makes a device: KFD's list counts them all,
            // or HIP's rule, at least as many, where the list is not ours.
            let areas = kernels_cuda::hip::kfd::held().map_err(|why| {
                Fault::Card(format!(
                    "the flat range cannot be sized: the driver's wave save areas for this \
                     process's queues are not readable ({why})"
                ))
            })?;
            Ok(Beside {
                said: format!("the driver holds {areas}"),
            })
        }
        #[cfg(not(feature = "rocm"))]
        Ok(Beside::NONE)
    }
}

impl PhysicalPool {
    /// A pool of what `ceiling` ([`ceiling_of`]) leaves of the card now.
    pub fn open(device: i32, ceiling: u64) -> Result<PhysicalPool> {
        #[cfg(feature = "device")]
        {
            let granularity = allocation_granularity(device)?;
            let (free, total) = crate::device::card::memory()?;
            let beside = Beside::read()?;
            if !beside.said.is_empty() {
                tracing::info!("outside the pool, {}", beside.said);
            }
            let budget = room_under(ceiling, free, total);
            let handle_bytes = align_up(MAP_UNIT_BYTES.max(granularity), granularity);
            let pages = budget / LOGICAL_PAGE_BYTES;
            Ok(PhysicalPool {
                device,
                granularity,
                handle_bytes,
                budget_pages: pages,
                hard_pages: pages,
                pledged_pages: 0,
                held_pages: 0,
                committed_pages: 0,
                ceiling,
                beside,
            })
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (device, ceiling);
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub fn stated(budget_bytes: u64) -> PhysicalPool {
        let pages = budget_bytes / LOGICAL_PAGE_BYTES;
        PhysicalPool {
            device: 0,
            granularity: LOGICAL_PAGE_BYTES,
            handle_bytes: MAP_UNIT_BYTES,
            budget_pages: pages,
            hard_pages: pages,
            pledged_pages: 0,
            held_pages: 0,
            committed_pages: 0,
            ceiling: u64::MAX,
            beside: Beside::NONE,
        }
    }

    #[must_use]
    pub const fn page_bytes(&self) -> u64 {
        LOGICAL_PAGE_BYTES
    }

    #[must_use]
    pub const fn budget_pages(&self) -> u64 {
        self.budget_pages
    }

    #[must_use]
    pub const fn hard_pages(&self) -> u64 {
        self.hard_pages
    }

    #[must_use]
    pub const fn map_unit_bytes(&self) -> u64 {
        self.handle_bytes
    }

    #[must_use]
    pub const fn granularity(&self) -> u64 {
        self.granularity
    }

    pub fn spare_bytes(&self, reserve: u64) -> Result<u64> {
        #[cfg(feature = "device")]
        {
            let (free, total) = crate::device::card::memory()?;
            Ok(room_under(self.ceiling, free, total).saturating_sub(reserve))
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = reserve;
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub const fn committed_pages(&self) -> u64 {
        self.committed_pages
    }

    /// What the pool can still take: the budget past what is committed and
    /// what is promised.
    #[must_use]
    pub const fn spare_pages(&self) -> u64 {
        self.budget_pages
            .saturating_sub(self.committed_pages + self.held_pages)
    }

    pub fn try_reserve(&mut self, pages: u64) -> bool {
        if pages > self.spare_pages() {
            return false;
        }
        self.held_pages += pages;
        true
    }

    pub fn unreserve(&mut self, pages: u64) {
        self.held_pages -= self.held_pages.min(pages);
    }

    pub fn mark_committed(&mut self, pages: u64) {
        let promised = self.held_pages.min(pages);
        self.held_pages -= promised;
        self.committed_pages += pages;
    }

    /// `pages` committed earlier went back to the device.
    pub fn mark_released(&mut self, pages: u64) {
        self.committed_pages = self.committed_pages.saturating_sub(pages);
    }

    /// Pledges the pool `pages` whatever the device reports free from here
    /// on. A fit declares a watermark from what the card has left, and the
    /// runtime admits frames against that watermark; what a fire allocates
    /// beside the pool afterwards (a program's scratch, a kernel's slab)
    /// comes out of the safety floor the budget held back, not out of the
    /// pages the fit already promised, so a recalibration that finds the
    /// room taken does not refuse a frame inside the pledge. Past it the
    /// device's word stands.
    pub fn pledge(&mut self, pages: u64) {
        self.pledged_pages = pages;
        self.budget_pages = self.budget_pages.max(pages);
        self.hard_pages = self.hard_pages.max(self.budget_pages);
    }

    pub fn recalibrate(&mut self) -> Result<()> {
        #[cfg(feature = "device")]
        {
            let (free, total) = crate::device::card::memory()?;
            let available = room_under(self.ceiling, free, total) / LOGICAL_PAGE_BYTES;
            self.recalibrate_to(available);
        }
        Ok(())
    }

    /// The budget as the device states it: what is charged plus the
    /// `available` pages under the ceiling, and never under the pledge.
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    fn recalibrate_to(&mut self, available: u64) {
        let charged = self.committed_pages + self.held_pages;
        self.budget_pages = charged.saturating_add(available).max(self.pledged_pages);
        self.hard_pages = self.hard_pages.max(self.budget_pages);
    }

    fn acquire_handle(&self, bytes: u64) -> Result<u64> {
        #[cfg(feature = "device")]
        {
            use crate::device::dr;

            let prop = allocation_prop(self.device);
            let mut handle: dr::CUmemGenericAllocationHandle = 0;
            // SAFETY: `handle` and `prop` are live locals; the handle is
            // released exactly once, by `release_handle`.
            let create = |handle: &mut dr::CUmemGenericAllocationHandle| unsafe {
                dr::cuMemCreate(handle, bytes as usize, &raw const prop, 0)
            };
            let mut made = create(&mut handle);
            if made != dr::CUresult::CUDA_SUCCESS && crate::device::held::release_idle() > 0 {
                made = create(&mut handle);
            }
            said("cuMemCreate", made)?;
            Ok(handle)
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = bytes;
            Err(Fault::Runtimeless)
        }
    }

    fn release_handle(&self, handle: u64) {
        #[cfg(feature = "device")]
        if handle != 0 {
            use crate::device::dr;

            // SAFETY: the handle came from this pool's own `cuMemCreate` and
            // is released exactly once.
            unsafe {
                let _ = dr::cuMemRelease(handle);
            }
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = handle;
        }
    }
}

#[derive(Debug)]
pub struct Arena {
    label: &'static str,
    base: u64,
    max_bytes: u64,
    virtual_bytes: u64,
    map_unit: u64,
    units: Vec<Option<u64>>,
    mapped: u64,
    cached: Vec<u64>,
    high_water: u64,
    /// Backed flat (one allocation of `virtual_bytes`, made on the first
    /// commit) rather than mapped unit by unit.
    flat: bool,
}

#[derive(Debug, Clone)]
pub enum Want {
    Prefix(u64),
    Ranges(Vec<(u64, u64)>),
}

impl Want {
    fn reach(&self) -> u64 {
        match self {
            Want::Prefix(bytes) => *bytes,
            Want::Ranges(ranges) => ranges
                .iter()
                .map(|&(offset, len)| offset.saturating_add(len))
                .max()
                .unwrap_or(0),
        }
    }
}

/// The unit an arena of `max_bytes` maps in: a 256th of it, held between the
/// driver's granularity and the pool's handle size, so a small arena is
/// backed in small pieces and a large one in few handles.
#[must_use]
pub fn map_unit_for(max_bytes: u64, granularity: u64, handle_bytes: u64) -> u64 {
    let ceiling = align_up(max_bytes, granularity).max(granularity);
    align_up(
        (max_bytes / HANDLES_PER_ARENA)
            .clamp(granularity, handle_bytes)
            .min(ceiling),
        granularity,
    )
    .max(granularity)
}

impl Arena {
    pub fn reserve(pool: &PhysicalPool, max_bytes: u64, label: &'static str) -> Result<Arena> {
        // A flat arena maps nothing, so its unit is only the pool's page: the
        // whole range is allocated at once, and a coarser unit would round
        // every arena up past what the fit priced it at.
        let map_unit = if mapped_on_demand() {
            map_unit_for(max_bytes, pool.granularity, pool.handle_bytes)
        } else {
            LOGICAL_PAGE_BYTES
        };
        let virtual_bytes = align_up(max_bytes, map_unit);
        if virtual_bytes == 0 {
            return Ok(Arena {
                label,
                base: 0,
                max_bytes: 0,
                virtual_bytes: 0,
                map_unit,
                units: Vec::new(),
                mapped: 0,
                cached: Vec::new(),
                high_water: 0,
                flat: !mapped_on_demand(),
            });
        }
        if !mapped_on_demand() {
            return Ok(Arena {
                label,
                base: 0,
                max_bytes,
                virtual_bytes,
                map_unit,
                units: vec![None; (virtual_bytes / map_unit) as usize],
                mapped: 0,
                cached: Vec::new(),
                high_water: 0,
                flat: true,
            });
        }
        #[cfg(feature = "device")]
        {
            use crate::device::dr;

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
                units: vec![None; (virtual_bytes / map_unit) as usize],
                mapped: 0,
                cached: Vec::new(),
                high_water: 0,
                flat: false,
            })
        }
        #[cfg(not(feature = "device"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub const fn base(&self) -> u64 {
        self.base
    }

    #[must_use]
    pub const fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.mapped * self.map_unit
    }

    fn units_of(&self, want: &Want) -> Vec<usize> {
        if self.map_unit == 0 {
            return Vec::new();
        }
        let mut units: Vec<usize> = match want {
            Want::Prefix(bytes) => (0..align_up(*bytes, self.map_unit) / self.map_unit)
                .map(|u| u as usize)
                .collect(),
            Want::Ranges(ranges) => ranges
                .iter()
                .filter(|&&(_, len)| len > 0)
                .flat_map(|&(offset, len)| {
                    let first = offset / self.map_unit;
                    let last = (offset + len - 1) / self.map_unit;
                    (first..=last).map(|u| u as usize)
                })
                .collect(),
        };
        units.sort_unstable();
        units.dedup();
        units
    }

    #[must_use]
    pub const fn high_water_bytes(&self) -> u64 {
        self.high_water
    }

    pub fn target_units(&self, want: &Want) -> Result<Vec<usize>> {
        let reach = want.reach();
        if reach > self.max_bytes {
            return Err(Fault::Ceiling {
                what: "bytes of an elastic arena",
                need: reach,
                have: self.max_bytes,
            });
        }
        Ok(self.units_of(want))
    }
    fn unbacked(&self, u: usize) -> bool {
        self.units.get(u).is_none_or(Option::is_none)
    }
    /// The units committing `units` backs anew: the unbacked ones, or for a
    /// flat arena not yet allocated, every unit it has (one allocation backs
    /// them all) and none after.
    fn fresh_count(&self, units: &[usize]) -> u64 {
        if self.flat {
            let wants = units.iter().any(|&u| self.unbacked(u));
            return if wants && self.base == 0 {
                self.units.len() as u64
            } else {
                0
            };
        }
        units.iter().filter(|&&u| self.unbacked(u)).count() as u64
    }
    fn target_bytes_of(&self, units: &[usize]) -> u64 {
        self.committed_bytes() + self.fresh_count(units) * self.map_unit
    }
    fn growth_pages(&self, units: &[usize]) -> u64 {
        let fresh = self
            .fresh_count(units)
            .saturating_sub(self.cached.len() as u64);
        pages_for_bytes(fresh * self.map_unit)
    }

    /// A flat arena's one allocation, backing every unit.
    fn grow_flat(&mut self, pool: &PhysicalPool) -> Result<()> {
        if self.base != 0 {
            return Ok(());
        }
        // Read only where a device allocates (its refusal names the pool).
        #[cfg(not(feature = "device"))]
        let _ = pool;
        #[cfg(feature = "device")]
        {
            use crate::device::rt;

            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: a live out-parameter; freed exactly once, in `Drop`.
            let allocate = |base: &mut *mut core::ffi::c_void| unsafe {
                rt::cudaMalloc(base, self.virtual_bytes as usize)
            };
            let mut made = allocate(&mut base);
            if made != rt::cudaError::cudaSuccess && crate::device::held::release_idle() > 0 {
                made = allocate(&mut base);
            }
            crate::device::ctx::check("cudaMalloc", made).map_err(|why| {
                Fault::Card(format!(
                    "the flat range of {} bytes was sized to what the card had free less what \
                     the driver holds beside it ({}), and the device refused it: {why}",
                    self.virtual_bytes,
                    if pool.beside.said.is_empty() { "nothing" } else { &pool.beside.said },
                ))
            })?;
            self.base = base as u64;
            // `unwritten-nan`: nothing has written any of it yet.
            if crate::serve::diag::on().unwritten_nan {
                crate::device::fill_span(self.base, self.virtual_bytes as usize, 0xFF)?;
            }
            self.units.fill(Some(FLAT));
            self.mapped = self.units.len() as u64;
            self.high_water = self.high_water.max(self.committed_bytes());
            Ok(())
        }
        #[cfg(not(feature = "device"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    fn grow(&mut self, pool: &PhysicalPool, units: &[usize]) -> Result<()> {
        if self.flat {
            return if units.iter().any(|&u| self.unbacked(u)) {
                self.grow_flat(pool)
            } else {
                Ok(())
            };
        }
        let cached_before = self.cached.len();
        let mut fresh: Vec<usize> = Vec::new();
        for &unit in units {
            if !self.unbacked(unit) || unit >= self.units.len() {
                continue;
            }
            let reused = self.cached.pop();
            let handle = match reused {
                Some(handle) => handle,
                None => match pool.acquire_handle(self.map_unit) {
                    Ok(handle) => handle,
                    Err(fault) => {
                        self.rollback(pool, &fresh, cached_before);
                        return Err(fault);
                    }
                },
            };
            let at = self.base + unit as u64 * self.map_unit;
            match self.map(pool, at, handle) {
                Ok(()) => {
                    self.units[unit] = Some(handle);
                    self.mapped += 1;
                    fresh.push(unit);
                }
                Err(fault) => {
                    if reused.is_some() {
                        self.cached.push(handle);
                    } else {
                        pool.release_handle(handle);
                    }
                    self.rollback(pool, &fresh, cached_before);
                    return Err(fault);
                }
            }
        }
        self.high_water = self.high_water.max(self.committed_bytes());
        Ok(())
    }
    fn map(&self, pool: &PhysicalPool, at: u64, handle: u64) -> Result<()> {
        #[cfg(feature = "device")]
        {
            use crate::device::dr;

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
        #[cfg(not(feature = "device"))]
        {
            let _ = (pool, at, handle);
            Err(Fault::Runtimeless)
        }
    }

    fn rollback(&mut self, pool: &PhysicalPool, fresh: &[usize], cached_goal: usize) {
        if self.flat {
            // `fresh` is non-empty only when this commit made the allocation.
            if !fresh.is_empty() && self.base != 0 {
                #[cfg(feature = "device")]
                // SAFETY: the allocation `grow_flat` just made, freed once.
                unsafe {
                    let _ = crate::device::rt::cudaFree(self.base as *mut core::ffi::c_void);
                }
                self.base = 0;
                self.units.iter_mut().for_each(|unit| *unit = None);
                self.mapped = 0;
            }
            let _ = (pool, cached_goal);
            return;
        }
        for &unit in fresh.iter().rev() {
            let Some(handle) = self.units.get_mut(unit).and_then(Option::take) else {
                continue;
            };
            self.mapped = self.mapped.saturating_sub(1);
            self.unmap(self.base + unit as u64 * self.map_unit);
            if self.cached.len() < cached_goal {
                self.cached.push(handle);
            } else {
                pool.release_handle(handle);
            }
        }
    }
    fn unmap(&self, at: u64) {
        #[cfg(feature = "device")]
        {
            use crate::device::dr;

            // SAFETY: `at` is a range this arena mapped and has not unmapped.
            unsafe {
                let _ = dr::cuMemUnmap(at, self.map_unit as usize);
            }
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = at;
        }
    }

    /// Unmaps every backed unit not in `keep` (ascending) and returns its
    /// memory to the device; a flat arena keeps its one allocation.
    fn shrink_to(&mut self, pool: &mut PhysicalPool, keep: &[usize]) {
        if self.flat {
            // One allocation backs every unit: it goes only when none is
            // kept, as a reshape gives its range back before allocating
            // the next.
            if keep.is_empty() && self.base != 0 {
                #[cfg(feature = "device")]
                // SAFETY: the flat arena's one allocation, freed once; the
                // base is cleared so `Drop` frees nothing.
                unsafe {
                    let _ = crate::device::rt::cudaFree(self.base as *mut core::ffi::c_void);
                }
                pool.mark_released(pages_for_bytes(self.mapped * self.map_unit));
                self.base = 0;
                self.units.fill(None);
                self.mapped = 0;
            }
            return;
        }
        let mut dropped = 0u64;
        for unit in 0..self.units.len() {
            if keep.binary_search(&unit).is_ok() {
                continue;
            }
            let Some(handle) = self.units[unit].take() else {
                continue;
            };
            self.unmap(self.base + unit as u64 * self.map_unit);
            pool.release_handle(handle);
            self.mapped = self.mapped.saturating_sub(1);
            dropped += 1;
        }
        pool.mark_released(pages_for_bytes(dropped * self.map_unit));
    }

    pub fn span(&self, offset: u64, len: u64) -> Result<u64> {
        let end = offset.saturating_add(len);
        let backed = self.map_unit != 0
            && end <= self.virtual_bytes
            && (len == 0
                || ((offset / self.map_unit)..=((end - 1) / self.map_unit))
                    .all(|u| !self.unbacked(u as usize)));
        if !backed {
            // A size-only walk names unbacked spans by a stand-in base:
            // nothing reads or writes through it (`crate::sizing`).
            if crate::sizing::active() && end <= self.virtual_bytes {
                return Ok(crate::sizing::STAND_IN + offset);
            }
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
        #[cfg(feature = "device")]
        {
            use crate::device::dr;

            if self.flat {
                if self.base != 0 {
                    // SAFETY: the flat arena's one allocation, freed once.
                    unsafe {
                        let _ = crate::device::rt::cudaFree(self.base as *mut core::ffi::c_void);
                    }
                }
                return;
            }
            for unit in 0..self.units.len() {
                let Some(handle) = self.units[unit].take() else {
                    continue;
                };
                self.unmap(self.base + unit as u64 * self.map_unit);
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

#[derive(Debug)]
pub struct Target<'a> {
    pub arena: &'a mut Arena,
    pub want: Want,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Commit {
    Committed,
    /// The growth this commit would map, and the pages the pool had left
    /// for it: a refusal is the first past the second, never the footprint
    /// against the whole budget.
    Exhausted {
        growth: u64,
        spare: u64,
    },
    Impossible {
        required: u64,
        ceiling: u64,
    },
}

pub fn commit_atomically(pool: &mut PhysicalPool, targets: &mut [Target<'_>]) -> Result<Commit> {
    let mut required = 0u64;
    let mut growth = 0u64;
    let mut wanted = Vec::with_capacity(targets.len());
    for target in targets.iter() {
        let units = target.arena.target_units(&target.want)?;
        required = required.saturating_add(pages_for_bytes(target.arena.target_bytes_of(&units)));
        growth = growth.saturating_add(target.arena.growth_pages(&units));
        wanted.push(units);
    }
    if growth == 0 {
        return Ok(Commit::Committed);
    }
    pool.recalibrate()?;
    if required > pool.hard_pages() {
        return Ok(Commit::Impossible {
            required,
            ceiling: pool.hard_pages(),
        });
    }
    if !pool.try_reserve(growth) {
        return Ok(Commit::Exhausted {
            growth,
            spare: pool.spare_pages(),
        });
    }
    let was: Vec<(Vec<usize>, usize)> = targets
        .iter()
        .zip(&wanted)
        .map(|(target, units)| {
            let fresh: Vec<usize> = units
                .iter()
                .copied()
                .filter(|&u| target.arena.unbacked(u))
                .collect();
            (fresh, target.arena.cached.len())
        })
        .collect();
    let failed = targets
        .iter_mut()
        .zip(&wanted)
        .enumerate()
        .find_map(|(done, (target, units))| {
            target.arena.grow(pool, units).err().map(|f| (done, f))
        });
    if let Some((done, fault)) = failed {
        for (undone, (fresh, cached)) in targets[..done].iter_mut().zip(&was[..done]).rev() {
            undone.arena.rollback(pool, fresh, *cached);
        }
        pool.unreserve(growth);
        return Err(fault);
    }
    pool.mark_committed(growth);
    Ok(Commit::Committed)
}

/// Backs `want` of `arena` beside the pool's budget rather than out of it,
/// as a kernel's slab is: what the device has free pays for it, and the
/// budget recalibrates around it. What a fire grows past the fit takes none
/// of the pages the fit pledged the cache rows.
pub fn commit_beside(pool: &PhysicalPool, arena: &mut Arena, want: &Want) -> Result<()> {
    let units = arena.target_units(want)?;
    arena.grow(pool, &units)
}

/// One reserved address range cut into equal segments, backed from the
/// bottom a segment at a time, with spans of it held backed apart: the
/// state pool's chunks grow up from offset 0 and a fire's activation sits
/// under the range's end, so neither end's base ever moves.
///
/// A segment need not be a whole number of map units: the units a segment's
/// bytes touch are backed, and a unit two segments share stays backed while
/// either is committed.
#[derive(Debug)]
pub struct Range {
    arena: Arena,
    segment_bytes: u64,
    committed: u32,
    /// Spans `(from, bytes)` that stay backed whatever is committed.
    held: Vec<(u64, u64)>,
    /// The bytes from the range's start a committed segment backs at most:
    /// past them a segment backs nothing ([`Range::limit`]).
    limit: u64,
    /// The most bytes the committed segments have backed, the held spans
    /// apart.
    segment_high: u64,
    /// Where the segments part, and by how many bytes ([`Range::part`]):
    /// those from the first lie that many bytes further up the range.
    parted: (u64, u64),
    /// A flat range before its fit: its holds are noted, not backed, so the
    /// one allocation is made at the size the fit gives it
    /// ([`Range::undefer`]), not at the estimate it was reserved with.
    deferred: bool,
}

/// The spans the first `bottom` bytes of a range's segments lie in, parted
/// at `parted.0` by `parted.1` bytes.
fn laid(bottom: u64, parted: (u64, u64)) -> [(u64, u64); 2] {
    let (from, by) = parted;
    if bottom <= from {
        return [(0, bottom), (0, 0)];
    }
    [(0, from), (from + by, bottom - from)]
}

impl Range {
    /// Reserves `bytes` of addresses in segments of `segment_bytes`; nothing
    /// is backed yet.
    pub fn reserve(
        pool: &PhysicalPool,
        bytes: u64,
        segment_bytes: u64,
        label: &'static str,
    ) -> Result<Range> {
        Ok(Range {
            arena: Arena::reserve(pool, bytes, label)?,
            segment_bytes: segment_bytes.max(1),
            committed: 0,
            held: Vec::new(),
            limit: u64::MAX,
            segment_high: 0,
            parted: (u64::MAX, 0),
            deferred: !mapped_on_demand(),
        })
    }

    /// Ends a flat range's deferral: its holds are backed, so it is
    /// allocated at its first.
    pub fn undefer(&mut self) {
        self.deferred = false;
    }

    /// Where the range lies. A flat range is allocated at its first
    /// commit, so before the fit it lies nowhere; a size-only walk over it
    /// then sees it at the stand-in base, so every address it derives from
    /// the range is a stand-in and nothing the walk writes reaches memory
    /// (`crate::sizing`).
    #[must_use]
    pub fn base(&self) -> u64 {
        if self.arena.base == 0 && crate::sizing::active() {
            return crate::sizing::STAND_IN;
        }
        self.arena.base
    }

    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.arena.max_bytes
    }

    #[must_use]
    pub const fn segment_bytes(&self) -> u64 {
        self.segment_bytes
    }

    /// The unit the range is backed in: what a commit rounds up to.
    #[must_use]
    pub const fn map_unit(&self) -> u64 {
        self.arena.map_unit
    }

    /// The segments the range holds, the last one short where the range is
    /// not a whole number of them.
    #[must_use]
    pub fn segments(&self) -> u32 {
        u32::try_from(self.arena.max_bytes.div_ceil(self.segment_bytes)).unwrap_or(u32::MAX)
    }

    #[must_use]
    pub const fn committed_segments(&self) -> u32 {
        self.committed
    }

    /// The bytes held backed from `from`, apart from the segments.
    #[must_use]
    pub fn held(&self, from: u64) -> u64 {
        self.held
            .iter()
            .find(|span| span.0 == from)
            .map_or(0, |span| span.1)
    }

    /// The bytes the spans held backed that start at or past `from` cover:
    /// a byte two of them hold is counted once.
    #[must_use]
    pub fn held_past(&self, from: u64) -> u64 {
        let mut spans: Vec<(u64, u64)> = self
            .held
            .iter()
            .filter(|span| span.0 >= from)
            .map(|span| (span.0, span.0.saturating_add(span.1)))
            .collect();
        spans.sort_unstable();
        let (mut covered, mut reached) = (0, from);
        for (start, end) in spans {
            covered += end.saturating_sub(start.max(reached));
            reached = reached.max(end);
        }
        covered
    }

    /// Every byte backed: the committed segments' and the held span's.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.arena.committed_bytes()
    }

    /// The bytes the committed segments back, the held span apart.
    #[must_use]
    pub fn segment_bytes_committed(&self) -> u64 {
        let bottom = (u64::from(self.committed) * self.segment_bytes)
            .min(self.arena.max_bytes)
            .min(self.limit);
        self.arena
            .units_of(&Want::Ranges(laid(bottom, self.parted).to_vec()))
            .len() as u64
            * self.arena.map_unit
    }

    #[must_use]
    pub const fn high_water_bytes(&self) -> u64 {
        self.arena.high_water
    }

    /// The most bytes the committed segments have backed: the high water
    /// of what the segments' prefix ever held, the held spans apart.
    #[must_use]
    pub const fn segment_high_water_bytes(&self) -> u64 {
        self.segment_high
    }

    /// The address of `offset`, where `len` bytes from it are backed.
    pub fn span(&self, offset: u64, len: u64) -> Result<u64> {
        self.arena.span(offset, len)
    }

    /// What the range backs with `segments` committed and `held` held.
    fn wanted(&self, segments: u32, held: &[(u64, u64)]) -> Want {
        let bytes = self.arena.max_bytes.min(self.limit);
        let bottom = (u64::from(segments) * self.segment_bytes).min(bytes);
        Want::Ranges(
            laid(bottom, self.parted)
                .into_iter()
                .chain(held.iter().copied())
                .collect(),
        )
    }

    /// The pages backing the first `segments` segments under `limit` bytes
    /// would take beyond what is backed now, were the segments parted at
    /// `parted.0` by `parted.1` bytes ([`Range::part`]), the held spans
    /// standing: what a commit of them would ask the pool for. Counted,
    /// nothing backed.
    pub fn growth_of(&self, segments: u32, limit: u64, parted: (u64, u64)) -> Result<u64> {
        let bytes = self.arena.max_bytes.min(limit);
        let bottom = (u64::from(segments.min(self.segments())) * self.segment_bytes).min(bytes);
        let want = Want::Ranges(
            laid(bottom, parted)
                .into_iter()
                .chain(self.held.iter().copied())
                .collect(),
        );
        let units = self.arena.target_units(&want)?;
        Ok(self.arena.growth_pages(&units))
    }

    /// Where the segments part ([`Range::part`]).
    #[must_use]
    pub const fn parted(&self) -> (u64, u64) {
        self.parted
    }

    /// Parts the segments at `from` bytes, a whole number of segments none
    /// past which is committed: those from there on lie `by` bytes further
    /// up the range, and the `by` bytes between are never backed. A range
    /// reserved for more than it comes to hold keeps both its ends where
    /// they lie this way: its first segments at its base, its last ones
    /// under what lies above them.
    pub fn part(&mut self, from: u64, by: u64) -> Result<()> {
        if !from.is_multiple_of(self.segment_bytes) || u64::from(self.committed) * self.segment_bytes > from
        {
            return Err(Fault::program(
                "device::elastic",
                "a range parted inside a segment, or under committed ones",
            ));
        }
        self.parted = (from, by);
        Ok(())
    }

    /// Holds the segments to the first `bytes` of the range: a segment that
    /// reaches past them backs only its part under them. For a range
    /// reserved for more than its pool may come to take; set before a
    /// segment that reaches past it is committed.
    pub fn limit(&mut self, bytes: u64) {
        self.limit = bytes;
    }

    /// Backs `bytes` from `from`, and keeps them backed whatever segments
    /// are committed or released; a span held from `from` before grows to
    /// this one, the spans held elsewhere stand. `beside` backs what the
    /// pool's budget refuses beside it ([`commit_beside`]).
    pub fn hold(
        &mut self,
        pool: &mut PhysicalPool,
        from: u64,
        bytes: u64,
        beside: bool,
    ) -> Result<Commit> {
        // A size-only walk backs nothing and holds nothing (`crate::sizing`).
        if crate::sizing::active() {
            return Ok(Commit::Committed);
        }
        // A span held from `from` already covering `bytes` leaves what is
        // held as it is, and what is held is backed: nothing to commit. A
        // bind holds the same span again on every program it lays, and the
        // walk below spells out every unit of the range to find that out.
        if self.held(from) >= bytes {
            return Ok(Commit::Committed);
        }
        let mut held = self.held.clone();
        match held.iter_mut().find(|span| span.0 == from) {
            Some(span) => span.1 = span.1.max(bytes),
            None => held.push((from, bytes)),
        }
        if self.deferred && !self.flat_allocated() {
            self.held = held;
            return Ok(Commit::Committed);
        }
        let want = self.wanted(self.committed, &held);
        let mut targets = [Target {
            arena: &mut self.arena,
            want: want.clone(),
        }];
        let mut outcome = commit_atomically(pool, &mut targets)?;
        if outcome != Commit::Committed && beside {
            commit_beside(pool, &mut self.arena, &want)?;
            outcome = Commit::Committed;
        }
        if outcome == Commit::Committed {
            self.held = held;
        }
        Ok(outcome)
    }

    /// Backs the first `segments` segments; those already backed stay.
    pub fn commit_segments(&mut self, pool: &mut PhysicalPool, segments: u32) -> Result<Commit> {
        // A size-only walk backs nothing (`crate::sizing`).
        if crate::sizing::active() {
            return Ok(Commit::Committed);
        }
        let segments = segments.min(self.segments());
        if segments <= self.committed {
            return Ok(Commit::Committed);
        }
        let want = self.wanted(segments, &self.held);
        let mut targets = [Target {
            arena: &mut self.arena,
            want,
        }];
        let outcome = commit_atomically(pool, &mut targets)?;
        if outcome == Commit::Committed {
            let bottom = |segments: u32| {
                (u64::from(segments) * self.segment_bytes)
                    .min(self.arena.max_bytes)
                    .min(self.limit)
            };
            let before = bottom(self.committed);
            let after = bottom(segments);
            self.committed = segments;
            self.segment_high = self.segment_high.max(self.segment_bytes_committed());
            // `unwritten-nan`: what the segments back anew holds NaN until a
            // kernel writes it.
            // (A flat range's one allocation is filled whole as it is made.)
            if crate::serve::diag::on().unwritten_nan && mapped_on_demand() {
                let [low, high] = laid(after, self.parted);
                let [was_low, was_high] = laid(before, self.parted);
                // Each piece grows at its end, from what it held.
                for ((at, len), (_, was)) in [(low, was_low), (high, was_high)] {
                    if len > was {
                        crate::device::fill_span(
                            self.span(at + was, len - was)?,
                            usize::try_from(len - was).unwrap_or(0),
                            0xFF,
                        )?;
                    }
                }
            }
        }
        Ok(outcome)
    }

    /// Gives back every segment past the first `segments`: the units only
    /// they touch are unmapped and their memory returned to the device. The
    /// caller has moved everything out of them and waited for whatever read
    /// or wrote them. A flat range gives nothing back.
    pub fn release_segments(&mut self, pool: &mut PhysicalPool, segments: u32) -> Result<()> {
        if segments >= self.committed {
            return Ok(());
        }
        let keep = self
            .arena
            .target_units(&self.wanted(segments, &self.held))?;
        self.arena.shrink_to(pool, &keep);
        self.committed = segments;
        Ok(())
    }

    /// Gives back everything, the held span too: before the range is
    /// dropped for another, so the pool's count of what is backed stands.
    pub fn release_all(&mut self, pool: &mut PhysicalPool) {
        self.arena.shrink_to(pool, &[]);
        self.committed = 0;
        self.held.clear();
    }

    /// Whether this is a flat range already allocated: its one allocation
    /// stands for the range's life ([`Range::unhold`]).
    #[must_use]
    pub fn flat_allocated(&self) -> bool {
        self.arena.flat && self.arena.base != 0
    }

    /// Forgets what is committed and held on a flat range already
    /// allocated, keeping the allocation: the range is laid out anew inside
    /// it. Allocating a flat range costs the driver clearing it (about five
    /// seconds for 150 GiB on gfx942), so a load allocates it once.
    pub fn unhold(&mut self) {
        debug_assert!(self.flat_allocated());
        self.committed = 0;
        self.held.clear();
    }
}

/// `bytes` on `device` that every device in `readers` may load and store,
/// without turning on peer access for the whole device: that maps every
/// later allocation into the peers, and freeing one then waits on a peer
/// that may be spinning in a collective (#745). Held for the process's life.
#[cfg(feature = "device")]
pub fn shared(device: i32, readers: &[i32], bytes: u64) -> Result<u64> {
    use crate::device::dr;

    let granularity = allocation_granularity(device)?;
    let bytes = align_up(bytes, granularity) as usize;
    let prop = allocation_prop(device);
    let mut handle: dr::CUmemGenericAllocationHandle = 0;
    let mut base: dr::CUdeviceptr = 0;
    let mut readers = readers.to_vec();
    readers.sort_unstable();
    readers.dedup();
    let descs: Vec<dr::CUmemAccessDesc> = readers.iter().map(|&d| access_desc(d)).collect();
    // SAFETY: live locals as out-parameters; the range is mapped to the
    // handle just made, and neither is released.
    unsafe {
        said(
            "cuMemCreate",
            dr::cuMemCreate(&raw mut handle, bytes, &raw const prop, 0),
        )?;
        said(
            "cuMemAddressReserve",
            dr::cuMemAddressReserve(&raw mut base, bytes, granularity as usize, 0, 0),
        )?;
        said("cuMemMap", dr::cuMemMap(base, bytes, 0, handle, 0))?;
        said(
            "cuMemSetAccess",
            dr::cuMemSetAccess(base, bytes, descs.as_ptr(), descs.len()),
        )?;
    }
    Ok(base)
}

#[cfg(feature = "device")]
fn said(call: &'static str, code: crate::device::dr::CUresult) -> Result<()> {
    if code == crate::device::dr::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: code as i32,
        })
    }
}

#[cfg(feature = "device")]
fn allocation_prop(device: i32) -> crate::device::dr::CUmemAllocationProp {
    use crate::device::dr;

    // SAFETY: a plain C struct of ints, enums whose zero is a variant, and a
    // null-able pointer; all zero bytes is the driver's documented default.
    let mut prop: dr::CUmemAllocationProp = unsafe { core::mem::zeroed() };
    prop.type_ = dr::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop
}

#[cfg(feature = "device")]
fn access_desc(device: i32) -> crate::device::dr::CUmemAccessDesc {
    use crate::device::dr;

    // SAFETY: a plain C struct of ints and enums whose zero is a variant;
    // every field is then set.
    let mut desc: dr::CUmemAccessDesc = unsafe { core::mem::zeroed() };
    desc.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = dr::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc
}

#[cfg(feature = "device")]
fn allocation_granularity(device: i32) -> Result<u64> {
    use crate::device::dr;

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
    use super::{Arena, LOGICAL_PAGE_BYTES, PhysicalPool, Range, pages_for_bytes};

    #[test]
    fn elastic_every_case() {
        a_parted_range_backs_the_chunks_it_states();
        a_promise_charges_the_budget_before_it_is_a_mapping();
        a_pledge_outlives_what_the_device_reports_free();
        a_partial_page_is_a_whole_page();
    }

    fn a_pledge_outlives_what_the_device_reports_free() {
        // Issue #672: the fit declared the pool from what the card had left
        // and the runtime admitted a c64 frame against it, then a fire took
        // room beside the pool and the recalibration under the frame's
        // commit budgeted only what was free, refusing the growth as
        // retryable past static admission.
        let mut pool = PhysicalPool::stated(100 * LOGICAL_PAGE_BYTES);
        pool.pledge(90);
        assert_eq!(
            pool.budget_pages(),
            100,
            "a pledge under the budget changes nothing"
        );
        assert!(pool.try_reserve(40));
        pool.mark_committed(40);

        pool.recalibrate_to(20);
        assert_eq!(
            pool.budget_pages(),
            90,
            "the device says 60; the pledge stands"
        );
        assert_eq!(pool.spare_pages(), 50);
        assert!(
            pool.try_reserve(50),
            "the frame inside the pledge is admitted"
        );
        pool.mark_committed(50);
        assert_eq!(pool.spare_pages(), 0);
        assert!(!pool.try_reserve(1), "and the pledge is where it ends");

        pool.recalibrate_to(30);
        assert_eq!(
            pool.budget_pages(),
            120,
            "past the pledge the device's word stands"
        );
        assert_eq!(pool.spare_pages(), 30);

        let mut unpledged = PhysicalPool::stated(100 * LOGICAL_PAGE_BYTES);
        assert!(unpledged.try_reserve(40));
        unpledged.mark_committed(40);
        unpledged.recalibrate_to(20);
        assert_eq!(unpledged.budget_pages(), 60);
        assert!(!unpledged.try_reserve(50), "as it was without one");
    }

    fn a_promise_charges_the_budget_before_it_is_a_mapping() {
        let mut pool = PhysicalPool::stated(10 * LOGICAL_PAGE_BYTES);
        assert_eq!(pool.budget_pages(), 10);
        assert!(pool.try_reserve(6));
        assert_eq!(pool.committed_pages(), 0);
        assert!(!pool.try_reserve(5));
        assert!(pool.try_reserve(4));
        pool.mark_committed(10);
        assert_eq!(pool.committed_pages(), 10);
        assert!(!pool.try_reserve(1));
    }

    /// A range of `bytes` in segments of `segment`, mapped in `unit`s, as
    /// reserved: nothing backed, no device asked.
    fn reserved(bytes: u64, segment: u64, unit: u64) -> Range {
        let virtual_bytes = bytes.div_ceil(unit) * unit;
        Range {
            arena: Arena {
                label: "a test's range",
                base: 0,
                max_bytes: bytes,
                virtual_bytes,
                map_unit: unit,
                units: vec![None; (virtual_bytes / unit) as usize],
                mapped: 0,
                cached: Vec::new(),
                high_water: 0,
                flat: false,
            },
            segment_bytes: segment,
            committed: 0,
            held: Vec::new(),
            limit: u64::MAX,
            segment_high: 0,
            parted: (u64::MAX, 0),
            deferred: false,
        }
    }

    /// The range stated is the range backed: a range reserved for more
    /// chunks than the card comes to back, stated as the chunks it does
    /// (parted, limited to them), backs with every segment committed those
    /// chunks and no byte of the ones skipped, its first chunks at its base
    /// and its last ending where the span above them starts, whatever the
    /// fit came to.
    fn a_parted_range_backs_the_chunks_it_states() {
        // A chunk that is no whole number of map units, as a served one.
        const CHUNK: u64 = 19 << 20;
        const UNIT: u64 = 1 << 20;
        const SEGMENT: u64 = 8;
        const RESERVED: u64 = 100;
        const ABOVE: u64 = 30;
        let units = |from: u64, bytes: u64| from / UNIT..(from + bytes).div_ceil(UNIT);
        for fitted in [8, 9, 16, 37, 64, 99] {
            for parting in [8, 16] {
                // A pool that fits under where it would part is not parted.
                if parting >= fitted {
                    continue;
                }
                let mut range = reserved((RESERVED + ABOVE) * CHUNK, SEGMENT * CHUNK, UNIT);
                let skipped = RESERVED - fitted;
                range.part(parting * CHUNK, skipped * CHUNK).unwrap();
                range.limit(fitted * CHUNK);
                let want = range.wanted(range.segments(), &[]);
                let backed = range.arena.target_units(&want).unwrap();
                let stated: Vec<usize> = units(0, parting * CHUNK)
                    .chain(units(
                        (parting + skipped) * CHUNK,
                        (fitted - parting) * CHUNK,
                    ))
                    .map(|unit| unit as usize)
                    .collect();
                assert_eq!(
                    backed, stated,
                    "{fitted} chunks stated of {RESERVED}, parted at {parting}"
                );
                assert_eq!(backed.len() as u64 * UNIT, fitted * CHUNK);
                assert_eq!(
                    backed.last().map(|&unit| (unit as u64 + 1) * UNIT),
                    Some(RESERVED * CHUNK),
                    "the last chunk stated ends where the span above starts"
                );
                // What a commit of them all would ask is those chunks.
                assert_eq!(
                    range
                        .growth_of(range.segments(), fitted * CHUNK, range.parted())
                        .unwrap(),
                    pages_for_bytes(fitted * CHUNK)
                );
            }
        }
        // Spans held over one another are counted once.
        let mut range = reserved((RESERVED + ABOVE) * CHUNK, SEGMENT * CHUNK, UNIT);
        let above = RESERVED * CHUNK;
        range.held = vec![
            (above, 4 * CHUNK),
            (above + CHUNK, CHUNK),
            (above + 3 * CHUNK, 2 * CHUNK),
            (above + 9 * CHUNK, CHUNK),
            (CHUNK, CHUNK),
        ];
        assert_eq!(range.held_past(above), 6 * CHUNK);
        assert_eq!(range.held(above), 4 * CHUNK);
        // Parted under a committed segment, or inside one, it is refused:
        // what is backed never moves.
        let mut range = reserved((RESERVED + ABOVE) * CHUNK, SEGMENT * CHUNK, UNIT);
        assert!(range.part(3 * CHUNK, CHUNK).is_err());
        range.committed = 2;
        assert!(range.part(SEGMENT * CHUNK, CHUNK).is_err());
        assert!(range.part(2 * SEGMENT * CHUNK, CHUNK).is_ok());
    }

    fn a_partial_page_is_a_whole_page() {
        assert_eq!(pages_for_bytes(0), 0);
        assert_eq!(pages_for_bytes(1), 1);
        assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES), 1);
        assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES + 1), 2);
    }
}
