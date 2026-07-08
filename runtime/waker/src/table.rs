//! The waker slot table (B9/B10): generation-tagged SPSC slots, epoch-filtered
//! wakes, and the two-fixed-slots-per-channel [`ChannelWakers`]. The
//! register-then-recheck race closure is documented on [`WakerTable::register`].

use std::task::Waker;

#[cfg(not(loom))]
use crate::r#loom::OnceLock;
use crate::r#loom::{AtomicU32, AtomicU64, Mutex, Ordering, RwLock, fence};

/// Opaque slot id: `generation:u32 << 32 | index:u32`. `0` is never valid
/// (generations start at 1), so a zeroed field on the C++ side is inert.
pub type WakerSlotId = u64;

/// Sentinel epoch meaning "no waiter registered".
const EPOCH_NONE: u64 = u64::MAX;

fn slot_id(generation: u32, index: u32) -> WakerSlotId {
    ((generation as u64) << 32) | index as u64
}
fn split_id(id: WakerSlotId) -> (u32, u32) {
    ((id >> 32) as u32, id as u32)
}

/// One waiter slot. `epoch` is the ring index the current waiter observed
/// when it registered (`EPOCH_NONE` = empty); `generation` invalidates stale
/// ids after `free`.
struct Slot {
    generation: AtomicU32,
    epoch: AtomicU64,
    /// The parked waker. A mutex (never held across a wake call's `wake()`
    /// itself — the waker is *taken* under the lock, woken outside it) keeps
    /// the table panic- and deadlock-free; contention is at most the one
    /// waiter vs the one committer of an SPSC endpoint.
    waker: Mutex<Option<Waker>>,
}

impl Slot {
    fn new(generation: u32) -> Slot {
        Slot {
            generation: AtomicU32::new(generation),
            epoch: AtomicU64::new(EPOCH_NONE),
            waker: Mutex::new(None),
        }
    }

    fn take_waker(&self) -> Option<Waker> {
        self.waker.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    fn put_waker(&self, w: Waker) {
        *self.waker.lock().unwrap_or_else(|e| e.into_inner()) = Some(w);
    }
}

/// What a wake attempt did — the probe surface distinguishes useful wakes
/// from filtered/stale/empty ones (`wakes-per-fire`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WakeOutcome {
    /// A parked waker was woken.
    Woken,
    /// Valid slot, nobody parked (already woken, or the waiter completed).
    Empty,
    /// Epoch filter: the ring index has not passed the registered epoch.
    Filtered,
    /// Stale generation or out-of-range index — a dead channel's id (B10).
    Stale,
}

/// Monotonic counters for the X0 probes (`wake-to-poll latency` is measured
/// by the harness around [`pie_wake`]; these give `wakes per fire`).
#[derive(Debug, Default)]
pub struct WakerMetrics {
    pub woken: AtomicU64,
    pub empty: AtomicU64,
    pub filtered: AtomicU64,
    pub stale: AtomicU64,
    pub swept: AtomicU64,
}

/// Snapshot of [`WakerMetrics`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetricsSnapshot {
    pub woken: u64,
    pub empty: u64,
    pub filtered: u64,
    pub stale: u64,
    pub swept: u64,
}

/// The Rust-owned waker slot table (B9/B10). One per process — the FFI
/// reaches it through [`WakerTable::global`] — but independently
/// constructible for tests and loom models.
pub struct WakerTable {
    slots: RwLock<Vec<&'static Slot>>,
    free: Mutex<Vec<u32>>,
    metrics: WakerMetrics,
}

impl WakerTable {
    pub fn new() -> WakerTable {
        WakerTable {
            slots: RwLock::new(Vec::new()),
            free: Mutex::new(Vec::new()),
            metrics: WakerMetrics::default(),
        }
    }

    /// The process-global table [`pie_wake`] dispatches into.
    #[cfg(not(loom))]
    pub fn global() -> &'static WakerTable {
        static GLOBAL: OnceLock<WakerTable> = OnceLock::new();
        GLOBAL.get_or_init(WakerTable::new)
    }

    fn slot(&self, index: u32) -> Option<&'static Slot> {
        self.slots
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .copied()
    }

    /// Allocate a slot; the returned id is what crosses the boundary.
    pub fn alloc(&self) -> WakerSlotId {
        let reused = self.free.lock().unwrap_or_else(|e| e.into_inner()).pop();
        if let Some(index) = reused {
            let slot = self.slot(index).expect("freelist index in range");
            // Generation was already bumped by `free`; the slot is quiescent.
            return slot_id(slot.generation.load(Ordering::Acquire), index);
        }
        let mut slots = self.slots.write().unwrap_or_else(|e| e.into_inner());
        let index = slots.len() as u32;
        // Slots are leaked on purpose: the table lives for the process, and
        // a stable `&'static` lets `wake` run outside any table lock. Freed
        // slots are recycled through the freelist, so the set stays bounded
        // by the high-water channel count.
        let slot: &'static Slot = Box::leak(Box::new(Slot::new(1)));
        slots.push(slot);
        slot_id(1, index)
    }

    /// Release a slot: bumps the generation (stale ids become no-ops), wakes
    /// any residual waiter (belt-and-braces — `sweep` should already have),
    /// and recycles the index.
    pub fn free(&self, id: WakerSlotId) {
        let (generation, index) = split_id(id);
        let Some(slot) = self.slot(index) else { return };
        if slot
            .generation
            .compare_exchange(generation, generation.wrapping_add(1), Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return; // stale free — someone else's generation
        }
        slot.epoch.store(EPOCH_NONE, Ordering::Release);
        if let Some(w) = slot.take_waker() {
            w.wake();
        }
        self.free.lock().unwrap_or_else(|e| e.into_inner()).push(index);
    }

    /// Park a waker on the slot, tagged with the ring index the waiter
    /// observed (B9). Returns `false` for a stale id (channel died).
    ///
    /// ## The register-then-recheck protocol (race closure)
    ///
    /// The caller MUST re-check its ready condition **after** this returns
    /// and proceed if it now holds (deregistering is optional — a later wake
    /// is spurious and harmless). Publication order inside: waker first,
    /// then epoch (Release). A concurrent committer either (a) reads the
    /// published epoch and wakes, or (b) read the old epoch — but then its
    /// index bump happened-before the caller's re-check read, which
    /// therefore observes the commit. No interleaving loses the wake.
    pub fn register(&self, id: WakerSlotId, waker: &Waker, observed_epoch: u64) -> bool {
        debug_assert_ne!(observed_epoch, EPOCH_NONE, "EPOCH_NONE is reserved");
        let (generation, index) = split_id(id);
        let Some(slot) = self.slot(index) else { return false };
        if slot.generation.load(Ordering::Acquire) != generation {
            return false;
        }
        slot.put_waker(waker.clone());
        slot.epoch.store(observed_epoch, Ordering::SeqCst);
        // Dekker/store-buffering closure (loom-found): the waiter's
        // publication (epoch store) and its subsequent condition re-check
        // (ring-index load) pair against the committer's index store and its
        // epoch load in `wake_impl`. Without SeqCst fences BOTH sides can
        // read stale values (Release/Acquire alone synchronizes only when
        // the released value is actually read) — a lost wake. The fence
        // here orders publish-before-recheck; the fence in `wake_impl`
        // orders index-store-before-epoch-load; the SC total order on the
        // two fences guarantees at least one side observes the other.
        fence(Ordering::SeqCst);
        // Re-validate the generation: a concurrent `free` may have swept
        // between the check above and our publication; if so, undo — the
        // channel is dead and the caller's re-check will see its poison.
        if slot.generation.load(Ordering::Acquire) != generation {
            slot.epoch.store(EPOCH_NONE, Ordering::Release);
            slot.take_waker();
            return false;
        }
        true
    }

    /// Clear the parked waker (future completed or dropped). Keeps
    /// `wakes-per-fire` honest; never required for correctness.
    pub fn deregister(&self, id: WakerSlotId) {
        let (generation, index) = split_id(id);
        let Some(slot) = self.slot(index) else { return };
        if slot.generation.load(Ordering::Acquire) != generation {
            return;
        }
        slot.epoch.store(EPOCH_NONE, Ordering::Release);
        slot.take_waker();
    }

    /// Unconditional wake — the exported [`pie_wake`] body.
    pub fn wake(&self, id: WakerSlotId) -> WakeOutcome {
        self.wake_impl(id, None)
    }

    /// Epoch-filtered wake (B9): wakes only if the committed `ring_index`
    /// has *passed* the waiter's observed epoch (`ring_index > epoch`).
    /// The committer calls this with the post-commit index.
    pub fn wake_past(&self, id: WakerSlotId, ring_index: u64) -> WakeOutcome {
        self.wake_impl(id, Some(ring_index))
    }

    fn wake_impl(&self, id: WakerSlotId, ring_index: Option<u64>) -> WakeOutcome {
        let (generation, index) = split_id(id);
        let outcome = 'o: {
            let Some(slot) = self.slot(index) else { break 'o WakeOutcome::Stale };
            if slot.generation.load(Ordering::Acquire) != generation {
                break 'o WakeOutcome::Stale;
            }
            // Pairs with `register`'s fence — see the comment there. The
            // committer MUST have published its ring index (SeqCst store,
            // or any store followed by a SeqCst fence) before calling in.
            fence(Ordering::SeqCst);
            let epoch = slot.epoch.load(Ordering::SeqCst);
            if epoch == EPOCH_NONE {
                break 'o WakeOutcome::Empty;
            }
            if let Some(k) = ring_index {
                if k <= epoch {
                    break 'o WakeOutcome::Filtered;
                }
            }
            slot.epoch.store(EPOCH_NONE, Ordering::Release);
            match slot.take_waker() {
                Some(w) => {
                    w.wake(); // outside every lock
                    WakeOutcome::Woken
                }
                None => WakeOutcome::Empty,
            }
        };
        let ctr = match outcome {
            WakeOutcome::Woken => &self.metrics.woken,
            WakeOutcome::Empty => &self.metrics.empty,
            WakeOutcome::Filtered => &self.metrics.filtered,
            WakeOutcome::Stale => &self.metrics.stale,
        };
        ctr.fetch_add(1, Ordering::Relaxed);
        outcome
    }

    /// B12: unconditional wake of every given slot (poison/close/abort of
    /// the touched channels) — ignores epochs, so blocked waiters re-poll
    /// and observe the failure instead of hanging.
    pub fn sweep(&self, ids: &[WakerSlotId]) {
        for &id in ids {
            self.wake(id);
            self.metrics.swept.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn metrics(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            woken: self.metrics.woken.load(Ordering::Relaxed),
            empty: self.metrics.empty.load(Ordering::Relaxed),
            filtered: self.metrics.filtered.load(Ordering::Relaxed),
            stale: self.metrics.stale.load(Ordering::Relaxed),
            swept: self.metrics.swept.load(Ordering::Relaxed),
        }
    }
}

impl Default for WakerTable {
    fn default() -> Self {
        Self::new()
    }
}

/// The two fixed waiter slots of one host-visible SPSC channel (B10): the
/// host `take/read` future parks on `reader`, the host `put` (back-pressure)
/// future parks on `writer`. The driver holds the two u64s next to the
/// channel's ring words and calls `pie_wake_past(reader, head)` after a net
/// put commits / `pie_wake_past(writer, tail)` after a net take commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelWakers {
    pub reader: WakerSlotId,
    pub writer: WakerSlotId,
}

impl ChannelWakers {
    pub fn alloc(table: &WakerTable) -> ChannelWakers {
        ChannelWakers { reader: table.alloc(), writer: table.alloc() }
    }
    /// B12: wake both endpoints (poison/close/abort).
    pub fn sweep(&self, table: &WakerTable) {
        table.sweep(&[self.reader, self.writer]);
    }
    pub fn free(&self, table: &WakerTable) {
        table.free(self.reader);
        table.free(self.writer);
    }
}
