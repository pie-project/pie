use crate::error::{Fault, Result};

pub const LOGICAL_PAGE_BYTES: u64 = 2 * 1024 * 1024;

pub const MAP_UNIT_BYTES: u64 = 32 * 1024 * 1024;

const HANDLES_PER_ARENA: u64 = 256;

const SAFETY_FLOOR_BYTES: u64 = 128 * 1024 * 1024;

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
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    device: i32,
    granularity: u64,
    handle_bytes: u64,
    budget_pages: u64,
    hard_pages: u64,
    held_pages: u64,
    committed_pages: u64,
    high_water_pages: u64,
    utilization: f64,
}

impl PhysicalPool {
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

    pub fn spare_bytes(&self, reserve: u64) -> Result<u64> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: two live locals; the call only writes them.
            let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
            crate::device::ctx::check("cudaMemGetInfo", asked)?;
            Ok(budget_bytes(free as u64, total as u64, self.utilization).saturating_sub(reserve))
        }
        #[cfg(not(feature = "cuda"))]
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

    pub fn try_reserve(&mut self, pages: u64) -> bool {
        let charged = self.committed_pages + self.held_pages;
        if pages > self.budget_pages.saturating_sub(charged.min(self.budget_pages)) {
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

    pub fn mark_uncommitted(&mut self, pages: u64) {
        self.committed_pages -= self.committed_pages.min(pages);
    }

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

impl Arena {
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
                units: Vec::new(),
                mapped: 0,
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
                units: vec![None; (virtual_bytes / map_unit) as usize],
                mapped: 0,
                cached: Vec::new(),
                high_water: 0,
            })
        }
        #[cfg(not(feature = "cuda"))]
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
    fn target_bytes_of(&self, units: &[usize]) -> u64 {
        let fresh = units.iter().filter(|&&u| self.unbacked(u)).count() as u64;
        self.committed_bytes() + fresh * self.map_unit
    }
    fn growth_pages(&self, units: &[usize]) -> u64 {
        let fresh = units.iter().filter(|&&u| self.unbacked(u)).count() as u64;
        let fresh = fresh.saturating_sub(self.cached.len() as u64);
        pages_for_bytes(fresh * self.map_unit)
    }

    fn grow(&mut self, pool: &PhysicalPool, units: &[usize]) -> Result<()> {
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

    fn rollback(&mut self, pool: &PhysicalPool, fresh: &[usize], cached_goal: usize) {
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
    pub fn release_outside(&mut self, pool: &mut PhysicalPool, want: &Want) -> u64 {
        let keep = self.units_of(want);
        let cache_goal = usize::from(keep.len() >= 2);
        let mut released = 0u64;
        for unit in 0..self.units.len() {
            if keep.binary_search(&unit).is_ok() {
                continue;
            }
            let Some(handle) = self.units[unit].take() else {
                continue;
            };
            self.mapped = self.mapped.saturating_sub(1);
            self.unmap(self.base + unit as u64 * self.map_unit);
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
    pub fn release_tail(&mut self, pool: &mut PhysicalPool, bytes: u64) -> u64 {
        self.release_outside(pool, &Want::Prefix(bytes))
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
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys as dr;

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
    Exhausted {
        required: u64,
        budget: u64,
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
            required,
            budget: pool.budget_pages(),
        });
    }
    let was: Vec<(Vec<usize>, usize)> = targets
        .iter()
        .zip(&wanted)
        .map(|(target, units)| {
            let fresh: Vec<usize> =
                units.iter().copied().filter(|&u| target.arena.unbacked(u)).collect();
            (fresh, target.arena.cached.len())
        })
        .collect();
    let mut done = 0usize;
    for (target, units) in targets.iter_mut().zip(&wanted) {
        if let Err(fault) = target.arena.grow(pool, units) {
            for (undone, (fresh, cached)) in targets[..done].iter_mut().zip(&was[..done]).rev() {
                undone.arena.rollback(pool, fresh, *cached);
            }
            pool.unreserve(growth);
            return Err(fault);
        }
        done += 1;
    }
    pool.mark_committed(growth);
    Ok(Commit::Committed)
}

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

    #[test]
    fn elastic_every_case() {
        a_promise_charges_the_budget_before_it_is_a_mapping();
        a_partial_page_is_a_whole_page();
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
