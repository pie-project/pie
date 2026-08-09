//! The grow-only device arena behind the attention hooks' sidebands.
//!
//! Three independent single-slot regions — the per-layer score capture, the
//! per-fire page mask, and the hook-graph prepare pass's score rows. Before it
//! existed those were `cudaMallocAsync`/`cudaFreeAsync` churn on the hot path,
//! ~90 alloc/free pairs on a 28-layer hook fire.
//!
//! "Allocation" is a capacity check: the caller carves its own sub-buffers as
//! offsets into the returned block. A fire needing more than the region has
//! grows it — stream-synced free, then realloc — which is rare after warmup
//! because the ladder doubles from 64 KiB.
//!
//! # The address-stability precondition
//!
//! While a region's capacity suffices, the address it hands out is **stable**.
//! That is what lets a captured hook fire replay against the same sideband
//! pointers, and it is the property [`SidebandArena`] exists to provide.
//!
//! The C++ header offers `generation()` as the invalidation signal for it, and
//! **nothing reads it** — that was checked across the whole repository before
//! this port kept it. It is not a hole: `pipeline/dispatch.cu`'s hook-graph
//! fingerprint mixes every arena address it baked (`mask_plan.keep`,
//! `out_indices`, `out_indptr`, `out_last_lens`, the stride, `score_rows_base`)
//! and recaptures when any of them moves, which is a strictly stronger check
//! than a counter — it notices a region that moved *without* the counter
//! moving, which the out-of-memory path in [`SidebandArena::acquire`] can
//! produce. The counter is kept because it is cheap, is the documented
//! contract, and is what the trace output reports; it should not be relied on
//! alone.

use std::ffi::c_void;

/// Which sideband a slot belongs to.
///
/// Discriminants match the C++ `HookSidebandArena::Region`, which indexes its
/// slot array with them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Region {
    /// One slot, re-acquired every layer. Decode and prefill score captures
    /// are mutually exclusive within a layer and layers run in sequence, so at
    /// most one capture is live at a time.
    Score = 0,
    /// One slot, acquired once per fire by the page mask and held for the
    /// whole layer loop.
    Mask = 1,
    /// One slot, acquired transiently by the hook-graph prepare pass: the
    /// folded-offset device CSR plus one padded row per score-reading (layer,
    /// lane).
    ScoreRows = 2,
}

impl Region {
    /// Every region, in slot order.
    pub const ALL: [Region; 3] = [Region::Score, Region::Mask, Region::ScoreRows];

    /// The spelling the C++ `region_name` uses, which the trace output and the
    /// parity transcript both carry.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Region::Score => "score",
            Region::Mask => "mask",
            Region::ScoreRows => "score_rows",
        }
    }
}

/// Why an [`SidebandArena::acquire`] handed back nothing.
///
/// The C++ returns a bare `nullptr` for all four, leaving the caller's refusal
/// path to treat "you asked for zero bytes" and "the device is out of memory"
/// as the same event. They are named here because they are not the same event:
/// one is a caller bug, one is an upstream bug, and two are resource failures
/// that leave the slot in *different* states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The region is already held. Overlapping acquisition is a bug upstream;
    /// the second caller is refused rather than handed the same bytes.
    Busy,
    /// Zero bytes were asked for. The slot is left alone — in particular it is
    /// **not** marked busy, so the next real acquire still succeeds.
    ZeroBytes,
    /// The pre-growth stream sync failed. Nothing was freed; the region keeps
    /// whatever block and capacity it had.
    SyncFailed,
    /// The replacement allocation failed. The old block was **already freed**
    /// by the time this was discovered, so the region is now empty and its
    /// previous address is dangling. See the module docs: the generation does
    /// not move across this, and the fingerprint is what notices.
    AllocFailed,
}

/// The device-memory operations the arena needs.
///
/// A trait rather than a direct call into [`crate::cuda`] because the growth
/// path frees the old block before it learns whether a replacement exists, and
/// that ordering is only testable against an allocator that can be told to
/// fail on cue.
pub trait DeviceMemory {
    /// Allocate `bytes` of device memory, or `None` on failure.
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void>;
    /// Release a block previously handed out by [`Self::alloc`].
    fn free(&mut self, ptr: *mut c_void);
    /// Wait for the stream to drain, returning `false` on failure.
    fn synchronize(&mut self) -> bool;
}

/// The live [`DeviceMemory`] (retirement plan phase B): the same three CUDA
/// calls the C++ arena makes — raw `cudaMalloc`/`cudaFree`, and
/// `cudaStreamSynchronize` before a growth frees the block in flight
/// (`hook_sideband_arena.cpp`). The stream lives HERE because the seam
/// folded the C++ carve's per-call stream parameter into the ops value: the
/// executor points one of these at the fire's stream before acquiring.
///
/// GATED: the only thing in this file that calls CUDA. `ArenaOps` above
/// it is a trait and `SidebandArena` is arithmetic over it, which is what
/// lets the arena's growth rule be proved without a card.
///
/// Raw `cudaMalloc`, deliberately NOT [`crate::gpu::device::Allocator`]: the C++
/// arena frees unconditionally after its own explicit synchronize, and the
/// capture-safe deferral would reorder exactly the free-then-realloc dance
/// the oracle pinned.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy)]
pub struct LiveDeviceMemory<'a> {
    stream: crate::gpu::device::StreamRef<'a>,
}

#[cfg(feature = "_cuda")]
impl<'a> LiveDeviceMemory<'a> {
    /// Ops ordered on `stream` — the one the fire's launches run on.
    #[must_use]
    pub const fn new(stream: crate::gpu::device::StreamRef<'a>) -> Self {
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

    // The seam's method is safe by design — the arena passes back only
    // pointers this impl's `alloc` handed out, and the recorders that share
    // the trait never touch memory at all. Marking one impl `unsafe` would
    // have to change the trait every oracle drives.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn free(&mut self, ptr: *mut c_void) {
        // The C++ ignores the status here, on both the growth and teardown
        // paths; a failed free on a dying arena has nowhere to report to.
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

/// The growth ladder: powers of two from 64 KiB.
///
/// Keeps growths logarithmic in the largest fire ever seen, so the "rare after
/// warmup" claim is structural rather than hopeful.
#[must_use]
pub fn round_capacity(bytes: usize) -> usize {
    let mut cap: usize = 64 * 1024;
    while cap < bytes {
        cap *= 2;
    }
    cap
}

/// Grow-only device arena for the hook sidebands.
///
/// Confined to the single lane thread that runs fires; nothing here is
/// thread-safe, which is why it is not `Sync`.
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

    /// The region's base pointer with at least `bytes` of capacity, growing
    /// the backing allocation when this fire needs more.
    ///
    /// Note the order of the checks, which is load-bearing: the busy test
    /// comes **before** the size test, so an overlapping acquisition is
    /// refused without first triggering a growth that the refused caller would
    /// have paid for and not used.
    ///
    /// # Errors
    ///
    /// See [`Refusal`]. Each variant leaves the slot in a different state.
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
            // Growth path: retire everything in flight that may still read the
            // old block, then free+realloc.
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

    /// Release the region's slot.
    ///
    /// Frees nothing — the backing allocation is reused by the next acquire.
    pub fn release(&mut self, region: Region) {
        self.slots[region as usize].busy = false;
    }

    /// Bumped on every successful growth.
    ///
    /// See the module docs before depending on this: a failed growth moves the
    /// region without moving the counter, and the hook-graph fingerprint — not
    /// this — is what the engine actually invalidates against.
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

    /// Fire boundary for the trace counters.
    ///
    /// Returns the finished fire's counts, if there was one — each acquire is
    /// a `cudaMallocAsync` the pre-arena code would have issued, and growths
    /// are the only real device allocations left, so the ratio is the evidence
    /// the arena is doing its job. The C++ prints this behind
    /// `PIE_SIDEBAND_TRACE`; returning it instead lets the caller decide, and
    /// lets a test read it without capturing stderr.
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

    /// Free every block, leaving the arena empty.
    ///
    /// The C++ does this in `~HookSidebandArena`. Rust cannot: `Drop` has no
    /// access to the allocator, and inventing a global one to make `Drop` work
    /// would reintroduce exactly the ambient state the trait removed. Callers
    /// own the arena's lifetime and must call this; the alternative — an owned
    /// allocator handle — was rejected because the arena is constructed beside
    /// `Workspace` at engine scope and outlives no allocator it could hold.
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

    /// The case the module docs single out: the region moves and the counter
    /// does not.
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
