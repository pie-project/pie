//! The waker slot table (B9/B10): generation-tagged SPSC slots, epoch-filtered
//! wakes, and the two-fixed-slots-per-channel [`ChannelWakers`].
//!
//! Every generation-sensitive field is protected by one per-slot mutex. An old
//! id therefore cannot validate one generation and later touch a recycled one.

use std::task::Waker;

#[cfg(not(loom))]
use crate::r#loom::OnceLock;
use crate::r#loom::{AtomicU64, Mutex, Ordering, RwLock};

/// Opaque slot id: `generation:u32 << 32 | index:u32`. `0` is never valid.
pub type WakerSlotId = u64;

/// The first valid epoch for a payload-free completion callback.
pub const FIRST_COMPLETION_EPOCH: u64 = 1;

/// Reserved waiter sentinel. It is rejected by public epoch-taking methods.
const EPOCH_NONE: u64 = u64::MAX;

fn slot_id(generation: u32, index: u32) -> WakerSlotId {
    debug_assert_ne!(generation, 0);
    ((generation as u64) << 32) | index as u64
}

fn split_id(id: WakerSlotId) -> (u32, u32) {
    ((id >> 32) as u32, id as u32)
}

struct Slot {
    state: Mutex<SlotState>,
}

struct SlotState {
    generation: u32,
    waiter_epoch: Option<u64>,
    published_epoch: u64,
    waker: Option<Waker>,
    retired: bool,
}

impl Slot {
    fn new() -> Slot {
        Slot {
            state: Mutex::new(SlotState {
                generation: 1,
                waiter_epoch: None,
                published_epoch: 0,
                waker: None,
                retired: false,
            }),
        }
    }
}

/// What a wake attempt did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WakeOutcome {
    /// A parked waker was woken.
    Woken,
    /// Valid slot, nobody parked.
    Empty,
    /// The supplied epoch has not passed the waiter's observation.
    Filtered,
    /// Stale generation, retired slot, or out-of-range index.
    Stale,
    /// The caller supplied a reserved completion epoch.
    InvalidEpoch,
}

/// Monotonic counters for the X0 probes.
#[derive(Debug, Default)]
pub struct WakerMetrics {
    pub woken: AtomicU64,
    pub empty: AtomicU64,
    pub filtered: AtomicU64,
    pub stale: AtomicU64,
    pub invalid_epoch: AtomicU64,
    pub swept: AtomicU64,
}

/// Snapshot of [`WakerMetrics`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetricsSnapshot {
    pub woken: u64,
    pub empty: u64,
    pub filtered: u64,
    pub stale: u64,
    pub invalid_epoch: u64,
    pub swept: u64,
}

/// The Rust-owned waker slot table. Slot storage is leaked for process-stable
/// addresses; freed indices are recycled until their generation would wrap.
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

    /// The process-global table used by the C ABI callbacks.
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

    fn is_live(state: &SlotState, generation: u32) -> bool {
        generation != 0 && !state.retired && state.generation != 0 && state.generation == generation
    }

    fn record(&self, outcome: WakeOutcome) {
        let counter = match outcome {
            WakeOutcome::Woken => &self.metrics.woken,
            WakeOutcome::Empty => &self.metrics.empty,
            WakeOutcome::Filtered => &self.metrics.filtered,
            WakeOutcome::Stale => &self.metrics.stale,
            WakeOutcome::InvalidEpoch => &self.metrics.invalid_epoch,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn invalid_epoch(&self) -> WakeOutcome {
        let outcome = WakeOutcome::InvalidEpoch;
        self.record(outcome);
        outcome
    }

    /// Allocate a slot. A recycled index is visible here only after [`free`]
    /// completed its generation bump and full state reset.
    pub fn alloc(&self) -> WakerSlotId {
        loop {
            let reused = self.free.lock().unwrap_or_else(|e| e.into_inner()).pop();
            if let Some(index) = reused {
                let slot = self.slot(index).expect("freelist index in range");
                let state = slot.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.retired || state.generation == 0 {
                    continue;
                }
                debug_assert!(state.waiter_epoch.is_none());
                debug_assert_eq!(state.published_epoch, 0);
                debug_assert!(state.waker.is_none());
                return slot_id(state.generation, index);
            }

            let mut slots = self.slots.write().unwrap_or_else(|e| e.into_inner());
            let index = u32::try_from(slots.len()).expect("waker slot index space exhausted");
            let slot: &'static Slot = Box::leak(Box::new(Slot::new()));
            slots.push(slot);
            return slot_id(1, index);
        }
    }

    /// Invalidate and reset a slot atomically. The residual waker is invoked only
    /// after the slot guard is released; the index reaches the freelist last.
    ///
    /// A slot at generation `u32::MAX` is retired instead of wrapping to zero or
    /// reusing an earlier generation.
    pub fn free(&self, id: WakerSlotId) {
        let (generation, index) = split_id(id);
        let Some(slot) = self.slot(index) else {
            return;
        };

        let (waker, recycle) = {
            let mut state = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            if !Self::is_live(&state, generation) {
                return;
            }

            let waker = state.waker.take();
            state.waiter_epoch = None;
            state.published_epoch = 0;

            let recycle = if state.generation == u32::MAX {
                state.generation = 0;
                state.retired = true;
                false
            } else {
                state.generation += 1;
                true
            };
            (waker, recycle)
        };

        if let Some(waker) = waker {
            waker.wake();
        }
        if recycle {
            self.free
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(index);
        }
    }

    /// Park a waker tagged with the epoch the waiter observed.
    ///
    /// The caller must re-check its ready condition after this returns. The
    /// per-slot mutex linearizes registration against wake/publication; if the
    /// committer won the mutex first, its external condition publication
    /// happens-before this lock acquisition and therefore the mandatory re-check.
    pub fn register(&self, id: WakerSlotId, waker: &Waker, observed_epoch: u64) -> bool {
        if observed_epoch == EPOCH_NONE {
            return false;
        }
        let (generation, index) = split_id(id);
        let Some(slot) = self.slot(index) else {
            return false;
        };
        let waker = waker.clone();

        let replaced = {
            let mut state = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            if !Self::is_live(&state, generation) {
                return false;
            }
            state.waiter_epoch = Some(observed_epoch);
            state.waker.replace(waker)
        };
        drop(replaced);
        true
    }

    /// Clear the parked waker for this exact generation.
    pub fn deregister(&self, id: WakerSlotId) {
        let (generation, index) = split_id(id);
        let Some(slot) = self.slot(index) else {
            return;
        };
        let removed = {
            let mut state = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            if !Self::is_live(&state, generation) {
                return;
            }
            state.waiter_epoch = None;
            state.waker.take()
        };
        drop(removed);
    }

    /// Unconditionally wake the waiter parked on this exact generation.
    pub fn wake(&self, id: WakerSlotId) -> WakeOutcome {
        self.wake_impl(id, None)
    }

    /// Wake iff `ring_index` passed the registered observation.
    pub fn wake_past(&self, id: WakerSlotId, ring_index: u64) -> WakeOutcome {
        if ring_index == EPOCH_NONE {
            return self.invalid_epoch();
        }
        self.wake_impl(id, Some(ring_index))
    }

    /// Publish a completion epoch monotonically and wake a passed waiter.
    ///
    /// Epoch zero means "not completed" and `u64::MAX` is reserved, so both are
    /// rejected in release builds.
    pub fn publish(&self, id: WakerSlotId, epoch: u64) -> WakeOutcome {
        if !(FIRST_COMPLETION_EPOCH..EPOCH_NONE).contains(&epoch) {
            return self.invalid_epoch();
        }

        let (generation, index) = split_id(id);
        let Some(slot) = self.slot(index) else {
            let outcome = WakeOutcome::Stale;
            self.record(outcome);
            return outcome;
        };

        let (outcome, waker) = {
            let mut state = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            if !Self::is_live(&state, generation) {
                (WakeOutcome::Stale, None)
            } else {
                state.published_epoch = state.published_epoch.max(epoch);
                match state.waiter_epoch {
                    None => (WakeOutcome::Empty, None),
                    Some(observed) if state.published_epoch <= observed => {
                        (WakeOutcome::Filtered, None)
                    }
                    Some(_) => {
                        state.waiter_epoch = None;
                        match state.waker.take() {
                            Some(waker) => (WakeOutcome::Woken, Some(waker)),
                            None => (WakeOutcome::Empty, None),
                        }
                    }
                }
            }
        };

        if let Some(waker) = waker {
            waker.wake();
        }
        self.record(outcome);
        outcome
    }

    /// Read the latest completion epoch for this exact generation.
    pub fn published(&self, id: WakerSlotId) -> Option<u64> {
        let (generation, index) = split_id(id);
        let slot = self.slot(index)?;
        let state = slot.state.lock().unwrap_or_else(|e| e.into_inner());
        Self::is_live(&state, generation).then_some(state.published_epoch)
    }

    fn wake_impl(&self, id: WakerSlotId, ring_index: Option<u64>) -> WakeOutcome {
        let (generation, index) = split_id(id);
        let Some(slot) = self.slot(index) else {
            let outcome = WakeOutcome::Stale;
            self.record(outcome);
            return outcome;
        };

        let (outcome, waker) = {
            let mut state = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            if !Self::is_live(&state, generation) {
                (WakeOutcome::Stale, None)
            } else {
                match state.waiter_epoch {
                    None => (WakeOutcome::Empty, None),
                    Some(observed) if ring_index.is_some_and(|committed| committed <= observed) => {
                        (WakeOutcome::Filtered, None)
                    }
                    Some(_) => {
                        state.waiter_epoch = None;
                        match state.waker.take() {
                            Some(waker) => (WakeOutcome::Woken, Some(waker)),
                            None => (WakeOutcome::Empty, None),
                        }
                    }
                }
            }
        };

        if let Some(waker) = waker {
            waker.wake();
        }
        self.record(outcome);
        outcome
    }

    /// Wake every listed endpoint after poison/close/abort.
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
            invalid_epoch: self.metrics.invalid_epoch.load(Ordering::Relaxed),
            swept: self.metrics.swept.load(Ordering::Relaxed),
        }
    }

    #[cfg(all(test, not(loom)))]
    pub(crate) fn force_generation_for_test(
        &self,
        id: WakerSlotId,
        generation: u32,
    ) -> Option<WakerSlotId> {
        if generation == 0 {
            return None;
        }
        let (current, index) = split_id(id);
        let slot = self.slot(index)?;
        let mut state = slot.state.lock().unwrap_or_else(|e| e.into_inner());
        if !Self::is_live(&state, current) {
            return None;
        }
        state.generation = generation;
        Some(slot_id(generation, index))
    }
}

impl Default for WakerTable {
    fn default() -> Self {
        Self::new()
    }
}

/// The two fixed waiter slots of one host-visible SPSC channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelWakers {
    pub reader: WakerSlotId,
    pub writer: WakerSlotId,
}

impl ChannelWakers {
    pub fn alloc(table: &WakerTable) -> ChannelWakers {
        ChannelWakers {
            reader: table.alloc(),
            writer: table.alloc(),
        }
    }

    pub fn sweep(&self, table: &WakerTable) {
        table.sweep(&[self.reader, self.writer]);
    }

    pub fn free(&self, table: &WakerTable) {
        table.free(self.reader);
        table.free(self.writer);
    }
}
