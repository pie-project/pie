#![allow(unsafe_code)]
use crate::error::{Fault, Result};

pub const LOGICAL_PAGE_BYTES: u64 = 2 * 1024 * 1024;

pub const MAP_UNIT_BYTES: u64 = 32 * 1024 * 1024;

const HANDLES_PER_ARENA: u64 = 256;

const SAFETY_FLOOR_BYTES: u64 = 128 * 1024 * 1024;

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

#[must_use]
pub const fn safety_floor_bytes(total: u64) -> u64 {
    let tenth = total / 10;
    if SAFETY_FLOOR_BYTES < tenth {
        SAFETY_FLOOR_BYTES
    } else {
        tenth
    }
}

#[must_use]
pub fn budget_bytes(free: u64, total: u64, utilization: f64) -> u64 {
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
    high_water_pages: u64,
    utilization: f64,
}

impl PhysicalPool {
    pub fn open(device: i32, utilization: f64) -> Result<PhysicalPool> {
        #[cfg(feature = "device")]
        {
            use crate::device::rt;

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
                pledged_pages: 0,
                held_pages: 0,
                committed_pages: 0,
                high_water_pages: 0,
                utilization,
            })
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (device, utilization);
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
            high_water_pages: 0,
            utilization: 1.0,
        }
    }

    #[must_use]
    pub const fn utilization(&self) -> f64 {
        self.utilization
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
            use crate::device::rt;

            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: two live locals; the call only writes them.
            let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
            crate::device::ctx::check("cudaMemGetInfo", asked)?;
            Ok(budget_bytes(free as u64, total as u64, self.utilization).saturating_sub(reserve))
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

    #[must_use]
    pub const fn high_water_pages(&self) -> u64 {
        self.high_water_pages
    }

    #[must_use]
    pub const fn handle_bytes(&self) -> u64 {
        self.handle_bytes
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
        self.high_water_pages = self.high_water_pages.max(self.committed_pages);
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
            use crate::device::rt;

            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: two live locals; the call only writes them.
            let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
            crate::device::ctx::check("cudaMemGetInfo", asked)?;
            let available =
                budget_bytes(free as u64, total as u64, self.utilization) / LOGICAL_PAGE_BYTES;
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
            let made =
                unsafe { dr::cuMemCreate(&raw mut handle, bytes as usize, &raw const prop, 0) };
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
    pub const fn map_unit(&self) -> u64 {
        self.map_unit
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
    fn grow_flat(&mut self) -> Result<()> {
        if self.base != 0 {
            return Ok(());
        }
        #[cfg(feature = "device")]
        {
            use crate::device::rt;

            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: a live out-parameter; freed exactly once, in `Drop`.
            crate::device::ctx::check("cudaMalloc", unsafe {
                rt::cudaMalloc(&raw mut base, self.virtual_bytes as usize)
            })?;
            self.base = base as u64;
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
                self.grow_flat()
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

    pub fn span(&self, offset: u64, len: u64) -> Result<u64> {
        let end = offset.saturating_add(len);
        let backed = self.map_unit != 0
            && end <= self.virtual_bytes
            && (len == 0
                || ((offset / self.map_unit)..=((end - 1) / self.map_unit))
                    .all(|u| !self.unbacked(u as usize)));
        if !backed {
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
    use super::{LOGICAL_PAGE_BYTES, PhysicalPool, pages_for_bytes};

    #[test]
    fn elastic_every_case() {
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
        assert_eq!(pool.high_water_pages(), 10);
        assert!(!pool.try_reserve(1));
    }

    fn a_partial_page_is_a_whole_page() {
        assert_eq!(pages_for_bytes(0), 0);
        assert_eq!(pages_for_bytes(1), 1);
        assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES), 1);
        assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES + 1), 2);
    }
}
