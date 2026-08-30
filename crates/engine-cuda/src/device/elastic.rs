//! **The elastic supply**: a budgeted pool of physical pages, and virtual
//! ranges whose backing grows and shrinks underneath a fixed address (alto
//! design §8; articles 4, 7 and 8).
//!
//! Ported from the C++ driver's `store/elastic.{hpp,cpp}` on `origin/dev`,
//! shape for shape, with the two things Rust changes: the pool is borrowed
//! rather than reference-counted (one owner, one thread — the engine loop),
//! and a failed grow rolls back through an explicit `rollback` rather than
//! through an exception's unwind.
//!
//! # Why a virtual range and not a bigger `cudaMalloc`
//!
//! **ADDRESSES ARE BAKE-TIME; BYTES ARE FIRE-TIME** (article 7). Every
//! address captured work reads is fixed at load: a `cudaGraphExec_t` records
//! the pool pointer its kv append writes through, and a pool that moved would
//! have to be re-recorded. `cuMemAddressReserve` gives the whole ceiling's
//! worth of address space at load — address space costs nothing, it is not
//! memory — and `cuMemCreate` + `cuMemMap` put physical pages under the front
//! of it as demand arrives. The base never moves; only how far past it is
//! readable changes. That is the one shape under which "grow the kv pool"
//! and "never re-record a graph" are both true.
//!
//! # The three numbers
//!
//! ```text
//! logical page   2 MiB   the accounting unit the budget is counted in
//! map unit       32 MiB  one `cuMemCreate` handle; the growth quantum
//! budget         total x utilization - what is already on the card
//!                        - safety floor, at load
//! ```
//!
//! The logical page is dev's `kLogicalPageBytes` (elastic.hpp:24) and is
//! deliberately NOT the kv page: a kv page is a number of tokens the model
//! declares, this is a number of bytes the allocator counts in. The map unit
//! is dev's `cuda_vmm_handle_bytes()` default (context.cpp:132-145) — 32 MiB,
//! there an env var and here a constant, because article 9 forbids a shell to
//! read the environment. The budget is dev's context.cpp:1015-1020 with the
//! operator's fraction put back into it: a safety floor of
//! `min(128 MiB, total/10)` is held back so that a driver allocation made
//! after ours still has somewhere to land, and what is left is bounded by
//! `[engine] gpu_mem_utilization` of the WHOLE CARD rather than by everything
//! the card happens to have free.
//!
//! # The fraction, and why it is of the card and not of what is free
//!
//! **`[engine] gpu_mem_utilization` IS THE OPERATOR'S CEILING OVER PIE'S
//! WHOLE FOOTPRINT** — weights included, which is what the key's own doc in
//! `worker::config` says (*"Fraction of each GPU's memory pie may use, weights
//! included"*). So the fraction multiplies `total`, and what is already on the
//! card at the moment the pool opens — this load's weight store, the context,
//! and anything another process holds — is SUBTRACTED from that ceiling rather
//! than ignored:
//!
//! ```text
//! budget = total x utilization - (total - free) - floor
//! ```
//!
//! At `utilization = 1.0` the two middle terms collapse to `free` and this is
//! byte for byte what the pool took before the fraction reached it, which is
//! how the arm is checked. On the L40S this workspace serves from — a
//! 48,305,799,168-byte card under a 13,761,281,792-byte gpt-oss weight store —
//! it is the difference between the 34.4 GB the pool used to take and the
//! 29.6 GB an operator who wrote `0.90` asked for (alto streaming §3 item 5,
//! `next.md` B1).
//!
//! # Soft budget and hard ceiling
//!
//! Two numbers, because article 4 asks for two refusals. A frame whose target
//! is past the HARD ceiling can never fit and is `Impossible` — nothing
//! anybody frees helps. A frame whose growth the SOFT budget will not cover
//! right now is `Exhausted` — the identical frame is worth re-submitting once
//! something else gives pages back. Dev keeps the pair for the same reason
//! (elastic.cpp:598-608) and recalibrates the soft one against a fresh
//! `cudaMemGetInfo` whenever a commit needs growth, which is
//! [`PhysicalPool::recalibrate`].

use crate::error::{Fault, Result};

/// The accounting unit the budget is counted in — dev's `kLogicalPageBytes`
/// (elastic.hpp:24).
pub const LOGICAL_PAGE_BYTES: u64 = 2 * 1024 * 1024;

/// The LARGEST quantum an arena grows and trims by — dev's
/// `cuda_vmm_handle_bytes()` default (context.cpp:132-145).
///
/// A constant rather than dev's `PIE_CUDA_VMM_HANDLE_MB`, because article 9
/// says a shell reads no environment.
pub const MAP_UNIT_BYTES: u64 = 32 * 1024 * 1024;

/// **How many handles one arena is willing to hold**, and therefore how fine
/// its growth quantum gets.
///
/// The one place this port does not simply take dev's number, and the reason
/// is the arena COUNT. Dev had four allocators holding a handful of very
/// large arenas, so a flat 32 MiB quantum was a rounding error. Here a kv
/// row's every plane is an arena of its own — dozens of them for a
/// forty-layer plan — and a flat 32 MiB would put a 32 MiB floor under each,
/// so a fire touching one page would commit a gigabyte and the whole claim of
/// this wave ("committed is demand, not ceiling") would be false by
/// construction. Dividing the ceiling instead keeps the quantum proportional
/// and the handle count bounded: an arena is at most this many driver objects
/// however big it is, and never coarser than 32 MiB.
const HANDLES_PER_ARENA: u64 = 256;

/// Reserved out of what the card says is free, so that a driver allocation
/// made after ours — a cuBLAS workspace, an NCCL buffer, a module load — has
/// somewhere to land. Dev's `min(128 MiB, total/10)` (context.cpp:1016-1017).
const SAFETY_FLOOR_BYTES: u64 = 128 * 1024 * 1024;

/// **The floor this card holds back** — dev's `min(128 MiB, total/10)`, in one
/// place so that the pool, the ahead-of-time accounting in
/// [`crate::store::Accounting`] and every test that spells the sentence read
/// one number.
#[must_use]
pub const fn safety_floor_bytes(total: u64) -> u64 {
    let tenth = total / 10;
    if SAFETY_FLOOR_BYTES < tenth {
        SAFETY_FLOOR_BYTES
    } else {
        tenth
    }
}

/// **What the elastic pool may hold, from three numbers and a fraction** —
/// the whole of `[engine] gpu_mem_utilization`'s effect, written where a test
/// can reach it without a device.
///
/// ```text
/// budget = total x utilization - (total - free) - floor
/// ```
///
/// `total - free` is everything already on the card when the pool opens: this
/// load's weight store (allocated first, by order), the context, and whatever
/// another process holds. Subtracting it is what makes the fraction a ceiling
/// over pie's WHOLE footprint rather than over the kv pool alone.
///
/// **`utilization = 1.0` IS THE OLD ARITHMETIC EXACTLY.** The two middle terms
/// collapse to `free`, so the pre-fraction pool is not a special case, it is
/// the top of the range — which is what makes the A/B a number and not a
/// branch. A fraction that lands under what is already on the card answers
/// zero rather than wrapping.
#[must_use]
pub fn budget_bytes(free: u64, total: u64, utilization: f64) -> u64 {
    // The fraction is validated in (0.0, 1.0] at boot, by the key's name, in
    // `crate::boot`. Clamped again here because this is a `pub fn` a caller
    // could reach with anything, and a NaN must not become a budget.
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

/// How many logical pages `bytes` occupies — dev's `pages_for_bytes`
/// (elastic.hpp:26-31).
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

/// **A budgeted supply of physical pages** — dev's `CudaPhysicalPool`
/// (elastic.hpp:36-45, elastic.cpp:39-227).
///
/// Reservation is the whole point: a caller that cannot get pages is TOLD so,
/// before anything is mapped, which is what makes the multi-arena commit
/// atomic. `held` is pages promised to a commit that has not finished
/// mapping; `committed` is pages actually under a mapping. Their sum is what
/// the budget is charged against, so two commits in flight cannot both be
/// told yes for the same page.
#[derive(Debug)]
pub struct PhysicalPool {
    device: i32,
    /// `cuMemGetAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM)`.
    granularity: u64,
    /// One handle's bytes, rounded up to the granularity.
    handle_bytes: u64,
    /// The soft budget, in logical pages. Recalibrated against a fresh
    /// `cudaMemGetInfo` when a commit needs growth.
    budget_pages: u64,
    /// The hard ceiling, in logical pages. Never lowered.
    hard_pages: u64,
    /// Promised, not yet mapped.
    held_pages: u64,
    /// Mapped.
    committed_pages: u64,
    /// The most `committed_pages` has ever been — the number article 8 asks
    /// the ENGINE to own and report, rather than have the runtime re-derive
    /// it from a free-list scan.
    high_water_pages: u64,
    /// **The operator's fraction of the card**, `[engine]
    /// gpu_mem_utilization`, kept because [`PhysicalPool::recalibrate`] re-asks
    /// the card and must re-apply the same ceiling. `1.0` is the whole card,
    /// which is what every pre-fraction caller got.
    utilization: f64,
}

impl PhysicalPool {
    /// **Open the pool against this device, at the operator's fraction of it**
    /// (dev context.cpp:1013-1025; alto streaming §3 item 5, `next.md` B1).
    ///
    /// `cudaMemGetInfo` at load, bounded by [`budget_bytes`], is the budget. It
    /// is read once here and re-read only by [`PhysicalPool::recalibrate`]: a
    /// `cudaMemGetInfo` on the fire path is a driver round trip, and the
    /// admission gate calls it only when a frame actually needs to grow.
    ///
    /// **`utilization` IS `[engine] gpu_mem_utilization`, AND THIS IS THE ONE
    /// READER.** It was declared, defaulted to `0.90`, validated and schema'd
    /// in the worker's config since before the palo rewrite and reached no
    /// shell at all: the pool took everything the card had free, which on the
    /// L40S this workspace serves from is 34.4 GB where an operator who wrote
    /// `0.90` asked for 29.6. The fraction is of the WHOLE CARD and pie's
    /// weight store is already on it by the time this runs, so the store's
    /// bytes are charged against the fraction rather than left outside it —
    /// which is the sentence [`crate::store::Accounting`] refuses ahead of.
    ///
    /// `1.0` is the arithmetic this constructor had before the fraction
    /// arrived, byte for byte.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] with no runtime selected, [`Fault::Device`] for
    /// the granularity query or the memory query.
    pub fn open(device: i32, utilization: f64) -> Result<PhysicalPool> {
        #[cfg(feature = "_cuda")]
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
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (device, utilization);
            Err(Fault::Runtimeless)
        }
    }

    /// A pool with a stated budget and no device behind it — what the
    /// accounting tests want, and the one constructor a runtimeless build can
    /// answer.
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
            // A stated budget IS the answer, so there is no card for a
            // fraction to be a fraction OF. `recalibrate` on a runtimeless
            // build leaves the budget where this put it either way.
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

    /// The most that has ever been (article 8: the engine owns this number).
    #[must_use]
    pub const fn high_water_pages(&self) -> u64 {
        self.high_water_pages
    }

    /// One handle's bytes — the quantum an arena grows and trims by.
    #[must_use]
    pub const fn handle_bytes(&self) -> u64 {
        self.handle_bytes
    }

    /// **Promise `pages`, or say no** — dev's `try_reserve`
    /// (elastic.cpp:97-106).
    ///
    /// Charged against committed + held, so a promise made and not yet mapped
    /// still counts. `false` is the whole of `Exhausted`: nothing was
    /// touched.
    pub fn try_reserve(&mut self, pages: u64) -> bool {
        let charged = self.committed_pages + self.held_pages;
        if pages > self.budget_pages.saturating_sub(charged.min(self.budget_pages)) {
            return false;
        }
        self.held_pages += pages;
        true
    }

    /// Give a promise back unused — dev's `unreserve` (elastic.cpp:108-113).
    pub fn unreserve(&mut self, pages: u64) {
        self.held_pages -= self.held_pages.min(pages);
    }

    /// A promise became a mapping — dev's `mark_committed`
    /// (elastic.cpp:127-135).
    pub fn mark_committed(&mut self, pages: u64) {
        let promised = self.held_pages.min(pages);
        self.held_pages -= promised;
        self.committed_pages += pages;
        self.high_water_pages = self.high_water_pages.max(self.committed_pages);
    }

    /// A mapping went away — dev's `mark_uncommitted` (elastic.cpp:137-142).
    pub fn mark_uncommitted(&mut self, pages: u64) {
        self.committed_pages -= self.committed_pages.min(pages);
    }

    /// **Re-read what the card has left, and move the soft budget** — dev's
    /// `recalibrate_budget` (elastic.cpp:166-186) driven from
    /// `recalibrate_elastic_budget` (context.cpp:2075-2085).
    ///
    /// The budget is what is charged plus what [`budget_bytes`] says is left
    /// under the operator's fraction, because the free figure already excludes
    /// our own mappings and those are already in `charged`. The hard ceiling
    /// only ever rises: it is the "this can never fit" line, and a transient
    /// shortage must not turn a frame that fits into `Impossible`.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the query; a runtimeless build leaves the budget
    /// where `stated` put it and answers `Ok`.
    pub fn recalibrate(&mut self) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: two live locals; the call only writes them.
            let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
            crate::device::ctx::check("cudaMemGetInfo", asked)?;
            // **THE SAME CEILING, RE-ASKED** — the fraction is a property of
            // the deployment and not of the instant, so a recalibration that
            // dropped it would hand back at the first grow exactly what the
            // operator declined to give at load.
            let available =
                budget_bytes(free as u64, total as u64, self.utilization) / LOGICAL_PAGE_BYTES;
            let charged = self.committed_pages + self.held_pages;
            self.budget_pages = charged.saturating_add(available).max(charged);
            self.hard_pages = self.hard_pages.max(self.budget_pages);
        }
        Ok(())
    }

    /// One physical allocation of `bytes` — dev's `acquire_handle`
    /// (elastic.cpp:186-201).
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    fn acquire_handle(&self, bytes: u64) -> Result<u64> {
        #[cfg(feature = "_cuda")]
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
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = bytes;
            Err(Fault::Runtimeless)
        }
    }

    /// Release one physical allocation — dev's `release_handle`
    /// (elastic.cpp:203-208). Infallible on purpose: it runs on the rollback
    /// and the trim paths, where there is nothing left to report a failure
    /// to.
    fn release_handle(&self, handle: u64) {
        #[cfg(feature = "_cuda")]
        if handle != 0 {
            use cudarc::driver::sys as dr;

            // SAFETY: the handle came from this pool's own `cuMemCreate` and
            // is released exactly once.
            unsafe {
                let _ = dr::cuMemRelease(handle);
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = handle;
        }
    }
}

/// **A fixed virtual range whose physical backing grows and trims from the
/// tail** — dev's `CudaArena` (elastic.hpp:128-163, elastic.cpp:229-424).
///
/// [`Arena::base`] is answered before one byte is mapped and never changes
/// again, which is article 7 in one method. What changes is
/// [`Arena::committed_bytes`]: how far past the base is readable.
///
/// **THE HOT BYTES MUST BE AT THE FRONT.** An arena grows and trims at its
/// TAIL, so an arena is the right shape for exactly one thing: a table whose
/// entry `n` lives at `n * stride`, addressed by a watermark. That is why a
/// kv row's PLANES get one arena each rather than one arena per row — page
/// `p` of the value plane sits halfway down a row-wide allocation, and
/// committing a prefix of that would leave every value page unmapped.
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
    /// One handle's bytes for THIS arena — the pool's, or the arena's own
    /// size where that is smaller, so a small arena is not quantized to
    /// nothing (dev elastic.cpp:243-250).
    map_unit: u64,
    /// Mapped handles, in address order. `handles[i]` backs
    /// `base + i * map_unit`.
    handles: Vec<u64>,
    /// Unmapped handles kept for the next grow — dev's `cached_handles_`
    /// (elastic.cpp:398-410). One `cuMemCreate` costs a driver round trip and
    /// a trim that released everything would pay it again on the next frame.
    cached: Vec<u64>,
    /// The most `committed_bytes` has ever been.
    high_water: u64,
}

impl Arena {
    /// **Reserve `max_bytes` of address space** — dev's constructor
    /// (elastic.cpp:236-262).
    ///
    /// Nothing is mapped. `base` is answerable immediately, which is what
    /// lets a load hand every kernel its pointer before a single page exists
    /// behind it.
    ///
    /// Dev reserves twice the ceiling here; this does not. The doubling
    /// served an allocator that could hand out an arena bigger than the one
    /// it asked for — [`Arena::grow`] refuses a target past `max_bytes`
    /// either way, so the second half was address space nothing could reach.
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
        #[cfg(feature = "_cuda")]
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
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// The base address, fixed for the arena's life (article 7).
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

    /// `bytes`, rounded up to this arena's map unit — dev's
    /// `target_committed_bytes` (elastic.cpp:311-318).
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

    /// Logical pages this arena would have to take from the pool to reach
    /// `bytes` — dev's `physical_growth_pages` (elastic.cpp:325-338).
    ///
    /// Cached handles are already charged, so they cost nothing to re-map.
    fn growth_pages(&self, target: u64) -> u64 {
        let committed = self.committed_bytes();
        if target <= committed {
            return 0;
        }
        let needed = (target - committed) / self.map_unit;
        let fresh = needed.saturating_sub(self.cached.len() as u64);
        pages_for_bytes(fresh * self.map_unit)
    }

    /// **Map physical pages under the tail until `target` is readable** —
    /// dev's `grow_reserved` (elastic.cpp:340-395).
    ///
    /// The caller has already reserved [`Arena::growth_pages`] from the pool;
    /// this only maps. A failure part-way rolls itself back to where it
    /// started before it returns, so the arena is never left with a partial
    /// growth (article 4's zero side effects, one level down).
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
        #[cfg(feature = "_cuda")]
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
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (pool, at, handle);
            Err(Fault::Runtimeless)
        }
    }

    /// Undo a partial grow — dev's `rollback_reserved` (elastic.cpp:397-410).
    ///
    /// Unmaps back to `bytes`, returning handles to the cache until it is the
    /// size it was and releasing the rest. Infallible: the caller is already
    /// on a failure path.
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

    /// **Unmap the tail down to `bytes`, and tell the pool** — dev's
    /// `release_tail` (elastic.cpp:412-440).
    ///
    /// ONE HANDLE IS KEPT MAPPED-OUT rather than released, when the arena is
    /// still holding two or more: a trim is nearly always followed by a grow,
    /// and a cached handle turns that grow into a `cuMemMap` alone. That
    /// single retained allocation is dev's `cache_goal` and it is why a
    /// grow → trim → grow cycle costs one `cuMemCreate` and not three.
    ///
    /// Returns the logical pages actually handed back.
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
        #[cfg(feature = "_cuda")]
        {
            use cudarc::driver::sys as dr;

            // SAFETY: `at` is a range this arena mapped and has not unmapped.
            unsafe {
                let _ = dr::cuMemUnmap(at, self.map_unit as usize);
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = at;
        }
    }

    /// **An address `offset` bytes in, checked against what is COMMITTED.**
    ///
    /// The bounds check [`Buffer::at`](crate::device::Buffer) makes, against
    /// the number that moves. Past the committed edge is reserved address
    /// space with nothing behind it: a write there faults, and a fault inside
    /// a captured graph is unattributable — so the door is here, at the one
    /// place that knows.
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
        #[cfg(feature = "_cuda")]
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
    /// The arena.
    pub arena: &'a mut Arena,
    /// Bytes of it that must be readable afterwards.
    pub bytes: u64,
}

/// **The answer to one atomic multi-arena commit** — dev's
/// `CudaCommitResult` (elastic.hpp:165-176).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Commit {
    /// Every arena is at or past its target. Stream work may proceed.
    Committed,
    /// Not right now. `required` is what the whole set of targets comes to,
    /// `budget` is what the pool would allow — both in logical pages, both
    /// worth putting in front of a human. **Nothing was mapped or unmapped.**
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

/// **The frame admission gate**: bring every arena to its target, or bring
/// none of them — dev's `commit_cuda_arena_targets_atomically`
/// (elastic.cpp:535-627).
///
/// The order is the whole of article 4. Every target is PRICED first, against
/// the hard ceiling and then against the soft budget, and only once the pool
/// has promised the pages does anything map. A refusal at either gate has
/// touched nothing — no handle created, no page mapped, no counter moved —
/// which is what makes the identical frame worth re-submitting. A failure
/// DURING the mapping rolls every arena back to where it started and hands
/// the promise back, and is a device fault rather than a refusal.
///
/// `required` is the TOTAL each arena would hold, not the growth: the ceiling
/// question is "does this load fit", and a load that already mapped most of
/// what it wants is not thereby entitled to more.
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
        // Everything asked for is already mapped, so it fits by having
        // fitted: no query, no ceiling test, no reservation. This is the
        // steady state and it costs one pass of arithmetic.
        return Ok(Commit::Committed);
    }
    // Only a commit that must GROW pays for a fresh `cudaMemGetInfo`: dev
    // guards the recalibration the same way (context.cpp:2321-2329), because
    // a driver round trip per frame is exactly the host work article 2 spends
    // its whole budget avoiding. It happens BEFORE both refusals, so a card
    // that has had memory handed back to it since load is not told
    // `Impossible` about a frame that now fits.
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
    // Where each arena stood before this commit touched it, so a failure
    // half-way can put every one of them back. Dev keeps the same pair per
    // arena in its `Growth` record (elastic.cpp:539-546).
    let was: Vec<(u64, usize)> = targets
        .iter()
        .map(|target| (target.arena.committed_bytes(), target.arena.cached.len()))
        .collect();
    let mut done = 0usize;
    for (target, bytes) in targets.iter_mut().zip(&wanted) {
        if let Err(fault) = target.arena.grow(pool, *bytes) {
            // The arena that failed rolled ITSELF back (`grow`'s contract);
            // the ones before it are wound back here, tail first.
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

/// One CUDA driver status, as a shell fault — [`nodes::said`]'s twin, kept
/// here so this module compiles with no runtime selected.
#[cfg(feature = "_cuda")]
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

#[cfg(feature = "_cuda")]
fn allocation_prop(device: i32) -> cudarc::driver::sys::CUmemAllocationProp {
    use cudarc::driver::sys as dr;

    let mut prop: dr::CUmemAllocationProp = unsafe { core::mem::zeroed() };
    prop.type_ = dr::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop
}

#[cfg(feature = "_cuda")]
fn access_desc(device: i32) -> cudarc::driver::sys::CUmemAccessDesc {
    use cudarc::driver::sys as dr;

    let mut desc: dr::CUmemAccessDesc = unsafe { core::mem::zeroed() };
    desc.location.type_ = dr::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = dr::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc
}

#[cfg(feature = "_cuda")]
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
    use super::{Commit, LOGICAL_PAGE_BYTES, PhysicalPool, pages_for_bytes};

    /// The accounting is the refusal. A promise that has not become a mapping
    /// still charges the budget, which is what stops two commits in flight
    /// from both being told yes for the same page.
    #[test]
    fn a_promise_charges_the_budget_before_it_is_a_mapping() {
        let mut pool = PhysicalPool::stated(10 * LOGICAL_PAGE_BYTES);
        assert_eq!(pool.budget_pages(), 10);
        assert!(pool.try_reserve(6));
        // Six are promised and none are mapped, and the budget knows it.
        assert_eq!(pool.committed_pages(), 0);
        assert!(!pool.try_reserve(5));
        assert!(pool.try_reserve(4));
        pool.mark_committed(10);
        assert_eq!(pool.committed_pages(), 10);
        assert_eq!(pool.high_water_pages(), 10);
        assert!(!pool.try_reserve(1));
    }

    /// A trim gives pages back to the budget and leaves the high water where
    /// it was — the engine owns that number and it is a record, not a level.
    #[test]
    fn a_trim_returns_pages_and_the_high_water_remembers() {
        let mut pool = PhysicalPool::stated(8 * LOGICAL_PAGE_BYTES);
        assert!(pool.try_reserve(8));
        pool.mark_committed(8);
        pool.mark_uncommitted(5);
        assert_eq!(pool.committed_pages(), 3);
        assert_eq!(pool.high_water_pages(), 8);
        assert!(pool.try_reserve(5));
    }

    /// Bytes round UP to logical pages: a page half used is a page spent.
    #[test]
    fn a_partial_page_is_a_whole_page() {
        assert_eq!(pages_for_bytes(0), 0);
        assert_eq!(pages_for_bytes(1), 1);
        assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES), 1);
        assert_eq!(pages_for_bytes(LOGICAL_PAGE_BYTES + 1), 2);
    }

    /// The two refusals are different sentences: `Exhausted` invites the
    /// identical frame back, `Impossible` does not.
    #[test]
    fn the_two_refusals_carry_different_numbers() {
        let exhausted = Commit::Exhausted {
            required: 40,
            budget: 12,
        };
        let impossible = Commit::Impossible {
            required: 40,
            ceiling: 20,
        };
        assert_ne!(exhausted, impossible);
        assert_ne!(exhausted, Commit::Committed);
    }
}
