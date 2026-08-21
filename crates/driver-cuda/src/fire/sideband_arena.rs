//! The grow-only device arena behind the attention hooks' sidebands: three
//! single-slot regions, growth a free-then-realloc keeping the address stable
//! while capacity suffices. `generation()` is offered but nothing reads it —
//! the hook-graph fingerprint is strictly stronger and used instead.

use std::ffi::c_void;

/// Which sideband a slot belongs to. The discriminants index the slot array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Region {
    /// One slot, re-acquired every layer: decode and prefill captures are
    /// mutually exclusive and layers run in sequence, so at most one is live.
    Score = 0,
    /// One slot, acquired once per fire and held for the whole layer loop.
    Mask = 1,
    /// One slot, acquired transiently by the hook-graph prepare pass: the
    /// folded-offset device CSR plus one padded row per score-reading (layer, lane).
    ScoreRows = 2,
}

impl Region {
    /// Every region, in slot order.
    pub const ALL: [Region; 3] = [Region::Score, Region::Mask, Region::ScoreRows];

    /// The spelling the trace output carries.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Region::Score => "score",
            Region::Mask => "mask",
            Region::ScoreRows => "score_rows",
        }
    }
}

/// Why an [`SidebandArena::acquire`] handed back nothing — named separately
/// because the four variants leave the slot in different states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The region is already held; the second caller is refused rather than
    /// handed the same bytes.
    Busy,
    /// Zero bytes were asked for; the slot is left alone, not marked busy.
    ZeroBytes,
    /// The pre-growth stream sync failed; nothing was freed, so the slot
    /// keeps its old block.
    SyncFailed,
    /// The replacement alloc failed after the old block was freed, so the
    /// region is empty; the generation does not move — the fingerprint notices.
    AllocFailed,
}

/// The device-memory operations the arena needs. A trait, not a direct CUDA
/// call, so the free-before-realloc ordering is testable against a
/// fail-on-cue allocator.
pub trait DeviceMemory {
    /// Allocate `bytes` of device memory, or `None` on failure.
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void>;
    /// Release a block previously handed out by [`Self::alloc`].
    fn free(&mut self, ptr: *mut c_void);
    /// Wait for the stream to drain, returning `false` on failure.
    fn synchronize(&mut self) -> bool;
}

/// The live [`DeviceMemory`]: raw `cudaMalloc`/`cudaFree` plus a
/// `cudaStreamSynchronize` before a growth frees the in-flight block. Raw
/// rather than [`crate::device::Allocator`], whose capture-safe deferral
/// would reorder the free-then-realloc dance.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy)]
pub struct LiveDeviceMemory<'a> {
    stream: crate::device::StreamRef<'a>,
}

#[cfg(feature = "_cuda")]
impl<'a> LiveDeviceMemory<'a> {
    /// Ops ordered on `stream` — the one the fire's launches run on.
    #[must_use]
    pub const fn new(stream: crate::device::StreamRef<'a>) -> Self {
        Self { stream }
    }
}

#[cfg(feature = "_cuda")]
impl DeviceMemory for LiveDeviceMemory<'_> {
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
        use cudarc::runtime::sys::{cudaError, cudaMalloc};
        let mut raw: *mut c_void = std::ptr::null_mut();
        let ok = unsafe { cudaMalloc(&mut raw, bytes) } == cudaError::cudaSuccess;
        (ok && !raw.is_null()).then_some(raw)
    }

    // Safe by design: passes back only pointers this impl's `alloc` handed out.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn free(&mut self, ptr: *mut c_void) {
        // A failed free on a dying arena has nowhere to report to.
        let _ = unsafe { cudarc::runtime::sys::cudaFree(ptr) };
    }

    fn synchronize(&mut self) -> bool {
        use cudarc::runtime::sys::{cudaError, cudaStreamSynchronize};
        let code = unsafe { cudaStreamSynchronize(self.stream.as_raw()) };
        code == cudaError::cudaSuccess
    }
}

/// One region's backing block.
#[derive(Debug, Clone, Copy)]
struct Slot {
    base: *mut c_void,
    capacity: usize,
    busy: bool,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            base: std::ptr::null_mut(),
            capacity: 0,
            busy: false,
        }
    }
}

/// The growth ladder: powers of two from 64 KiB, logarithmic in the largest fire seen.
#[must_use]
pub fn round_capacity(bytes: usize) -> usize {
    let mut cap: usize = 64 * 1024;
    while cap < bytes {
        cap *= 2;
    }
    cap
}

/// Grow-only device arena for the hook sidebands. Confined to the single lane
/// thread that runs fires; nothing here is thread-safe, hence not `Sync`.
#[derive(Debug)]
pub struct SidebandArena {
    slots: [Slot; 3],
    generation: u64,
    fire_index: u64,
    fire_acquires: u32,
    fire_grows: u32,
    total_acquires: u64,
    total_grows: u64,
}

impl Default for SidebandArena {
    fn default() -> Self {
        Self::new()
    }
}

impl SidebandArena {
    /// An arena holding nothing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: [Slot::default(); 3],
            generation: 0,
            fire_index: 0,
            fire_acquires: 0,
            fire_grows: 0,
            total_acquires: 0,
            total_grows: 0,
        }
    }

    /// The region's base pointer with at least `bytes` of capacity. The check
    /// order is load-bearing: busy before size, so an overlapping acquire is
    /// refused without triggering an unused growth.
    pub fn acquire<M: DeviceMemory>(
        &mut self,
        mem: &mut M,
        region: Region,
        bytes: usize,
    ) -> Result<*mut c_void, Refusal> {
        let index = region as usize;
        if self.slots[index].busy {
            return Err(Refusal::Busy);
        }
        if bytes == 0 {
            return Err(Refusal::ZeroBytes);
        }
        if bytes > self.slots[index].capacity {
            // Growth path: retire in-flight readers of the old block, then free+realloc.
            let new_capacity = round_capacity(bytes);
            if !mem.synchronize() {
                return Err(Refusal::SyncFailed);
            }
            if !self.slots[index].base.is_null() {
                mem.free(self.slots[index].base);
                self.slots[index].base = std::ptr::null_mut();
                self.slots[index].capacity = 0;
            }
            let Some(fresh) = mem.alloc(new_capacity) else {
                return Err(Refusal::AllocFailed);
            };
            self.slots[index].base = fresh;
            self.slots[index].capacity = new_capacity;
            self.generation += 1;
            self.fire_grows += 1;
            self.total_grows += 1;
        }
        self.slots[index].busy = true;
        self.fire_acquires += 1;
        self.total_acquires += 1;
        Ok(self.slots[index].base)
    }

    /// Release the region's slot; the backing is reused by the next acquire, not freed.
    pub fn release(&mut self, region: Region) {
        self.slots[region as usize].busy = false;
    }

    /// Bumped on every successful growth. A failed growth moves the region
    /// without moving this — the hook-graph fingerprint is what invalidates.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether a region is currently held.
    #[must_use]
    pub const fn is_held(&self, region: Region) -> bool {
        self.slots[region as usize].busy
    }

    /// A region's current capacity in bytes.
    #[must_use]
    pub const fn capacity(&self, region: Region) -> usize {
        self.slots[region as usize].capacity
    }

    /// Fire boundary for the trace counters; returns the finished fire's
    /// counts, if any. The acquires-to-grows ratio is evidence the arena works.
    pub fn begin_fire(&mut self) -> Option<FireCounts> {
        let finished = (self.fire_index > 0).then_some(FireCounts {
            fire: self.fire_index,
            acquires: self.fire_acquires,
            grows: self.fire_grows,
            total_acquires: self.total_acquires,
            total_grows: self.total_grows,
            generation: self.generation,
        });
        self.fire_index += 1;
        self.fire_acquires = 0;
        self.fire_grows = 0;
        finished
    }

    /// Free every block, leaving the arena empty. Not a `Drop`, since `Drop`
    /// has no access to the allocator — callers must call this themselves.
    pub fn destroy<M: DeviceMemory>(&mut self, mem: &mut M) {
        for slot in &mut self.slots {
            if !slot.base.is_null() {
                mem.free(slot.base);
            }
            *slot = Slot::default();
        }
    }
}

/// What one fire drew from the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireCounts {
    /// The fire's index, counting from one.
    pub fire: u64,
    /// Acquires during the fire.
    pub acquires: u32,
    /// Growths during the fire — the real device allocations.
    pub grows: u32,
    /// Acquires since construction.
    pub total_acquires: u64,
    /// Growths since construction.
    pub total_grows: u64,
    /// The generation at the fire boundary.
    pub generation: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A counting allocator that can be told to fail.
    #[derive(Default)]
    struct FakeMemory {
        next: usize,
        handed_out: usize,
        fail_allocs: usize,
        fail_syncs: usize,
        freed: Vec<*mut c_void>,
    }

    impl FakeMemory {
        fn new() -> Self {
            Self {
                next: 0x1000,
                ..Self::default()
            }
        }
    }

    impl DeviceMemory for FakeMemory {
        fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
            if self.fail_allocs > 0 {
                self.fail_allocs -= 1;
                return None;
            }
            let _ = bytes;
            self.handed_out += 1;
            let p = self.next as *mut c_void;
            self.next += 1 << 24;
            Some(p)
        }
        fn free(&mut self, ptr: *mut c_void) {
            self.freed.push(ptr);
        }
        fn synchronize(&mut self) -> bool {
            if self.fail_syncs > 0 {
                self.fail_syncs -= 1;
                return false;
            }
            true
        }
    }

    #[test]
    fn the_ladder_doubles_from_64_kib_and_an_exact_fit_does_not_grow() {
        assert_eq!(round_capacity(1), 64 * 1024);
        assert_eq!(round_capacity(64 * 1024), 64 * 1024);
        assert_eq!(round_capacity(64 * 1024 + 1), 128 * 1024);
        assert_eq!(round_capacity(1 << 20), 1 << 20);
    }

    #[test]
    fn an_acquire_within_capacity_returns_the_same_address() {
        let mut mem = FakeMemory::new();
        let mut arena = SidebandArena::new();
        let first = arena.acquire(&mut mem, Region::Score, 1024).unwrap();
        arena.release(Region::Score);
        let second = arena.acquire(&mut mem, Region::Score, 4096).unwrap();
        assert_eq!(first, second, "the address must be stable within capacity");
        assert_eq!(arena.generation(), 1, "no growth, no generation bump");
        assert_eq!(mem.handed_out, 1);
    }

    #[test]
    fn an_overlapping_acquire_is_refused_before_it_can_trigger_a_growth() {
        let mut mem = FakeMemory::new();
        let mut arena = SidebandArena::new();
        arena.acquire(&mut mem, Region::Score, 1024).unwrap();
        assert_eq!(
            arena.acquire(&mut mem, Region::Score, 1 << 30),
            Err(Refusal::Busy)
        );
        assert_eq!(mem.handed_out, 1, "the refused caller paid for no growth");
        assert_eq!(arena.capacity(Region::Score), 64 * 1024);
    }

    #[test]
    fn a_zero_byte_request_is_refused_without_marking_the_slot_busy() {
        let mut mem = FakeMemory::new();
        let mut arena = SidebandArena::new();
        assert_eq!(
            arena.acquire(&mut mem, Region::Score, 0),
            Err(Refusal::ZeroBytes)
        );
        assert!(!arena.is_held(Region::Score));
        assert!(arena.acquire(&mut mem, Region::Score, 1024).is_ok());
    }

    #[test]
    fn a_failed_sync_leaves_the_old_block_alone() {
        let mut mem = FakeMemory::new();
        let mut arena = SidebandArena::new();
        let first = arena.acquire(&mut mem, Region::Score, 1024).unwrap();
        arena.release(Region::Score);
        mem.fail_syncs = 1;
        assert_eq!(
            arena.acquire(&mut mem, Region::Score, 1 << 20),
            Err(Refusal::SyncFailed)
        );
        assert!(mem.freed.is_empty(), "nothing may be freed before the sync");
        arena.release(Region::Score);
        assert_eq!(arena.acquire(&mut mem, Region::Score, 1024).unwrap(), first);
    }

    /// The case the module docs single out: the region moves, the counter does not.
    #[test]
    fn a_failed_alloc_empties_the_region_without_bumping_the_generation() {
        let mut mem = FakeMemory::new();
        let mut arena = SidebandArena::new();
        arena.acquire(&mut mem, Region::Score, 1024).unwrap();
        arena.release(Region::Score);
        let before = arena.generation();
        mem.fail_allocs = 1;
        assert_eq!(
            arena.acquire(&mut mem, Region::Score, 1 << 20),
            Err(Refusal::AllocFailed)
        );
        assert_eq!(mem.freed.len(), 1, "the old block was freed first");
        assert_eq!(arena.capacity(Region::Score), 0);
        assert_eq!(
            arena.generation(),
            before,
            "the address moved and the counter did not -- see the module docs"
        );
    }

    #[test]
    fn the_regions_are_independent_slots_over_one_generation_counter() {
        let mut mem = FakeMemory::new();
        let mut arena = SidebandArena::new();
        for region in Region::ALL {
            arena.acquire(&mut mem, region, 1024).unwrap();
        }
        assert_eq!(arena.generation(), 3);
        assert!(Region::ALL.iter().all(|&r| arena.is_held(r)));
        arena.release(Region::Score);
        assert!(!arena.is_held(Region::Score));
        assert!(arena.is_held(Region::Mask));
    }

    #[test]
    fn the_first_fire_boundary_reports_nothing_because_no_fire_has_finished() {
        let mut mem = FakeMemory::new();
        let mut arena = SidebandArena::new();
        assert_eq!(arena.begin_fire(), None);
        arena.acquire(&mut mem, Region::Score, 1024).unwrap();
        arena.release(Region::Score);
        let counts = arena.begin_fire().expect("one fire finished");
        assert_eq!(counts.acquires, 1);
        assert_eq!(counts.grows, 1);
        assert_eq!(counts.fire, 1);
    }
}
