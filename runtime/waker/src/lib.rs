//! # X0 — the tensor-waker substrate (Runtime–Driver Boundary, B9–B12)
//!
//! The host-side dual of contract **C2**: the driver waits on words at device
//! cut points; the *host* parks futures on the same ring indices and is woken
//! by the driver (or the mock) through this table. This module is the whole
//! foundation X1–X4 build on: it owns every [`Waker`], hands the other side
//! of the boundary nothing but opaque `u64` slot ids, and closes the
//! register/commit race without a lock shared across the boundary.
//!
//! ## Locked decisions realized here
//!
//! - **B9 — epoch-tagged registration.** A waiter reads the channel's ring
//!   index (head or tail — whichever its condition watches), and registers
//!   `(waker, observed_epoch)`. The committer wakes when the ring index
//!   *passes* the registered epoch ([`WakerTable::wake_past`]). The race
//!   (commit lands between the waiter's observation and its registration) is
//!   closed by the **register-then-recheck protocol**: `register` publishes
//!   the waker *first*, then the caller re-checks its condition — either the
//!   committer sees the published waker, or the re-check sees the committed
//!   index. [`WaitFuture`] encodes the protocol so callers cannot get it
//!   wrong; hand-rolled pollers MUST follow it (documented on
//!   [`WakerTable::register`]).
//! - **B10 — C++ never holds a `Waker`.** The FFI surface is
//!   [`pie_wake`]/[`pie_wake_past`]: opaque `u64` in, `0/1` out, callable
//!   from any thread, never unwinds. All waker memory lives in this table.
//!   Slots are **generation-tagged** (id = `generation << 32 | index`), so a
//!   stale id held by C++ after a channel died is a harmless no-op.
//! - **SPSC ⇒ two fixed slots per host-visible channel** (one reader-waiter,
//!   one writer-waiter — [`ChannelWakers`]): no waiter lists, no thundering
//!   herd, O(1) memory per channel.
//! - **B12 — sweep on poison/close/abort.** [`WakerTable::sweep`] /
//!   [`ChannelWakers::sweep`] wake every registered slot of the touched
//!   channels unconditionally (ignoring epochs), so a blocked
//!   `take().await?` re-polls, observes the poison, and resolves to `Err` —
//!   it never hangs.
//!
//! Spurious wakes are permitted everywhere (the futures contract); the epoch
//! filter exists to keep them *rare* (the `wakes-per-fire` probe), never to
//! guarantee their absence.
//!
//! Mock-first: nothing here touches CUDA. The mock driver calls the same
//! `pie_wake*` exports the C++ driver will.

#[cfg(not(loom))]
use std::sync::OnceLock;
use std::task::Waker;

// Loom-swappable sync primitives: under `--cfg loom` the model checker
// explores every interleaving of these; in normal builds they are std.
#[cfg(loom)]
use loom::sync::{
    Mutex, RwLock,
    atomic::{AtomicU32, AtomicU64, Ordering, fence},
};
#[cfg(not(loom))]
use std::sync::{
    Mutex, RwLock,
    atomic::{AtomicU32, AtomicU64, Ordering, fence},
};

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

// ===========================================================================
// The waiting future — the protocol, encoded
// ===========================================================================

/// One observation of the waiter's condition.
pub enum Readiness<T> {
    Ready(T),
    /// Not ready; `observed_epoch` is the ring index the check read (what
    /// the eventual commit must pass).
    Pending { observed_epoch: u64 },
}

/// A future that parks on `slot` until `check` returns [`Readiness::Ready`].
/// Encodes register-then-recheck, tolerates spurious wakes, and resolves
/// (via `check` observing poison and returning `Ready(Err(..))`-shaped
/// values) after a B12 sweep.
pub struct WaitFuture<'t, F> {
    table: &'t WakerTable,
    slot: WakerSlotId,
    check: F,
}

impl<'t, F, T> WaitFuture<'t, F>
where
    F: FnMut() -> Readiness<T> + Unpin,
{
    pub fn new(table: &'t WakerTable, slot: WakerSlotId, check: F) -> Self {
        WaitFuture { table, slot, check }
    }
}

impl<'t, F, T> std::future::Future for WaitFuture<'t, F>
where
    F: FnMut() -> Readiness<T> + Unpin,
{
    type Output = T;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<T> {
        let this = self.get_mut();
        // Fast path.
        let observed = match (this.check)() {
            Readiness::Ready(v) => return std::task::Poll::Ready(v),
            Readiness::Pending { observed_epoch } => observed_epoch,
        };
        // Publish the waker, then MANDATORY re-check (see `register` docs).
        if !this.table.register(this.slot, cx.waker(), observed) {
            // Stale slot: the channel died between checks — one more check
            // must surface the failure; poll again immediately.
            cx.waker().wake_by_ref();
            return std::task::Poll::Pending;
        }
        match (this.check)() {
            Readiness::Ready(v) => {
                this.table.deregister(this.slot);
                std::task::Poll::Ready(v)
            }
            Readiness::Pending { .. } => std::task::Poll::Pending,
        }
    }
}

// ===========================================================================
// FFI — the only surface the other side of the boundary sees (B10)
// ===========================================================================

/// Wake the waiter parked on `slot_id`, unconditionally. Returns `1` if a
/// waker was woken, `0` otherwise (stale id / nobody parked). Callable from
/// any thread; never unwinds.
#[cfg(not(loom))]
#[unsafe(no_mangle)]
pub extern "C" fn pie_wake(slot_id: u64) -> u8 {
    let r = std::panic::catch_unwind(|| WakerTable::global().wake(slot_id));
    matches!(r, Ok(WakeOutcome::Woken)) as u8
}

/// Epoch-filtered wake (B9): wake the waiter parked on `slot_id` iff the
/// committed `ring_index` has passed its registered observation. Callable
/// from any thread; never unwinds.
#[cfg(not(loom))]
#[unsafe(no_mangle)]
pub extern "C" fn pie_wake_past(slot_id: u64, ring_index: u64) -> u8 {
    let r = std::panic::catch_unwind(|| WakerTable::global().wake_past(slot_id, ring_index));
    matches!(r, Ok(WakeOutcome::Woken)) as u8
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64 as StdAtomicU64};
    use std::task::{Context, Poll, Wake};

    /// A test waker that records wakes and can unpark a spinning poller.
    struct Flag(AtomicBool);
    impl Wake for Flag {
        fn wake(self: Arc<Self>) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    fn flag_waker() -> (Arc<Flag>, Waker) {
        let f = Arc::new(Flag(AtomicBool::new(false)));
        (f.clone(), f.into())
    }

    #[test]
    fn alloc_register_wake_roundtrip() {
        let t = WakerTable::new();
        let id = t.alloc();
        let (f, w) = flag_waker();
        assert!(t.register(id, &w, 0));
        assert_eq!(t.wake(id), WakeOutcome::Woken);
        assert!(f.0.load(std::sync::atomic::Ordering::SeqCst));
        // One-shot: the waker was taken.
        assert_eq!(t.wake(id), WakeOutcome::Empty);
    }

    #[test]
    fn epoch_filter_wakes_only_when_index_passes() {
        let t = WakerTable::new();
        let id = t.alloc();
        let (f, w) = flag_waker();
        assert!(t.register(id, &w, 5));
        assert_eq!(t.wake_past(id, 4), WakeOutcome::Filtered);
        assert_eq!(t.wake_past(id, 5), WakeOutcome::Filtered);
        assert!(!f.0.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(t.wake_past(id, 6), WakeOutcome::Woken);
        assert!(f.0.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn stale_generation_is_noop_b10() {
        let t = WakerTable::new();
        let id = t.alloc();
        let (_, w) = flag_waker();
        assert!(t.register(id, &w, 0));
        t.free(id);
        // The dead channel's id, still held by "C++": every op is inert.
        assert_eq!(t.wake(id), WakeOutcome::Stale);
        assert_eq!(t.wake_past(id, 99), WakeOutcome::Stale);
        assert!(!t.register(id, &w, 0));
        // The recycled slot gets a NEW generation: old id still stale.
        let id2 = t.alloc();
        assert_eq!(id & 0xFFFF_FFFF, id2 & 0xFFFF_FFFF, "index recycled");
        assert_ne!(id, id2, "generation bumped");
        assert!(t.register(id2, &w, 0));
        assert_eq!(t.wake(id), WakeOutcome::Stale);
        assert_eq!(t.wake(id2), WakeOutcome::Woken);
    }

    #[test]
    fn spurious_wakes_are_harmless() {
        let t = WakerTable::new();
        let id = t.alloc();
        // Nobody parked: empty, not an error, no panic.
        assert_eq!(t.wake(id), WakeOutcome::Empty);
        assert_eq!(t.wake_past(id, 1), WakeOutcome::Empty);
        // Double-wake after a single register: second is empty.
        let (_, w) = flag_waker();
        assert!(t.register(id, &w, 0));
        assert_eq!(t.wake(id), WakeOutcome::Woken);
        assert_eq!(t.wake(id), WakeOutcome::Empty);
        let m = t.metrics();
        assert_eq!(m.woken, 1);
        assert_eq!(m.empty, 3);
    }

    #[test]
    fn sweep_on_abort_resolves_blocked_take_to_err_b12() {
        // A take().await? parked on a channel that will NEVER become full:
        // poison + sweep (from a foreign thread, as the driver would) must
        // resolve it to Err — never hang.
        let t = Arc::new(WakerTable::new());
        let ch = ChannelWakers::alloc(&t);
        let poisoned = Arc::new(AtomicBool::new(false));
        let head = Arc::new(StdAtomicU64::new(0)); // ring index: never bumps

        let sweeper = {
            let (t, poisoned) = (t.clone(), poisoned.clone());
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                poisoned.store(true, std::sync::atomic::Ordering::SeqCst);
                ch.sweep(&t); // B12: wake both endpoints, epochs ignored
            })
        };

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let res = rt.block_on(async {
            let (t, poisoned, head) = (t.clone(), poisoned.clone(), head.clone());
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                WaitFuture::new(&*t, ch.reader, move || {
                    if poisoned.load(std::sync::atomic::Ordering::SeqCst) {
                        return Readiness::Ready(Err::<u64, &str>("poisoned"));
                    }
                    let h = head.load(std::sync::atomic::Ordering::SeqCst);
                    if h > 0 {
                        Readiness::Ready(Ok(h))
                    } else {
                        Readiness::Pending { observed_epoch: h }
                    }
                }),
            )
            .await
            .expect("sweep lost: blocked take hung")
        });
        sweeper.join().unwrap();
        assert_eq!(res, Err("poisoned"));
        let m = t.metrics();
        assert_eq!(m.swept, 2, "both endpoints swept");
    }

    #[test]
    fn ffi_wake_from_foreign_thread_completes_future() {
        // The exported symbol path: a raw std thread (standing in for the
        // C++ commit tail) calls pie_wake on the GLOBAL table.
        let t = WakerTable::global();
        let id = t.alloc();
        let head = Arc::new(StdAtomicU64::new(0));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let res = rt.block_on(async {
            let head2 = head.clone();
            let waiter = WaitFuture::new(t, id, move || {
                let h = head2.load(std::sync::atomic::Ordering::SeqCst);
                if h > 0 { Readiness::Ready(h) } else { Readiness::Pending { observed_epoch: h } }
            });
            let committer = {
                let head = head.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    head.store(1, std::sync::atomic::Ordering::SeqCst);
                    super::pie_wake_past(id, 1)
                })
            };
            let v = tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
                .await
                .expect("no lost wake");
            let woke = committer.join().unwrap();
            (v, woke)
        });
        assert_eq!(res.0, 1);
        // Either the FFI wake delivered it (1) or the fast path won the race
        // and the wake found the slot empty/deregistered (0) — both legal.
        assert!(res.1 <= 1);
        t.free(id);
    }

    #[test]
    fn register_commit_race_stress_no_lost_wake() {
        // Non-loom stress version of the race: many iterations of
        // observe(e) ∥ commit(e+1)+wake_past — a lost wake deadlocks (caught
        // by the timeout).
        let t = WakerTable::global();
        for i in 0..2000 {
            let id = t.alloc();
            let head = Arc::new(StdAtomicU64::new(0));
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .unwrap();
            let head2 = head.clone();
            let committer = std::thread::spawn(move || {
                if i % 2 == 0 {
                    std::thread::yield_now();
                }
                head2.store(1, std::sync::atomic::Ordering::SeqCst);
                super::pie_wake_past(id, 1);
            });
            let v = rt.block_on(async {
                let head = head.clone();
                tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    WaitFuture::new(t, id, move || {
                        let h = head.load(std::sync::atomic::Ordering::SeqCst);
                        if h > 0 { Readiness::Ready(h) } else { Readiness::Pending { observed_epoch: h } }
                    }),
                )
                .await
                .expect("lost wake (register/commit race)")
            });
            assert_eq!(v, 1);
            committer.join().unwrap();
            t.free(id);
        }
    }

    #[test]
    fn wake_to_poll_latency_probe() {
        // X0 probe: time from pie_wake to the woken future's poll observing
        // readiness. Prints; asserts only a sanity bound (CI-safe).
        let t = WakerTable::global();
        let id = t.alloc();
        let head = Arc::new(StdAtomicU64::new(0));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let woke_at = Arc::new(StdAtomicU64::new(0));
        let base = std::time::Instant::now();
        let head2 = head.clone();
        let woke_at2 = woke_at.clone();
        let committer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            head2.store(1, std::sync::atomic::Ordering::SeqCst);
            let now = base.elapsed().as_nanos() as u64;
            woke_at2.store(now, std::sync::atomic::Ordering::SeqCst);
            super::pie_wake(id);
        });
        let polled_at = rt.block_on(async {
            let head = head.clone();
            WaitFuture::new(t, id, move || {
                let h = head.load(std::sync::atomic::Ordering::SeqCst);
                if h > 0 { Readiness::Ready(base.elapsed().as_nanos() as u64) } else { Readiness::Pending { observed_epoch: h } }
            })
            .await
        });
        committer.join().unwrap();
        let woke = woke_at.load(std::sync::atomic::Ordering::SeqCst);
        let latency_ns = polled_at.saturating_sub(woke);
        println!("wake-to-poll latency: {latency_ns} ns");
        assert!(latency_ns < 100_000_000, "sanity bound: < 100ms");
        t.free(id);
    }

    #[test]
    fn wakes_per_fire_metrics() {
        let t = WakerTable::new();
        let ch = ChannelWakers::alloc(&t);
        let (_, w) = flag_waker();
        // Reader parked at epoch 3; a fire commits index 4 on this channel
        // and blind-fires both endpoints (writer idle): exactly one useful
        // wake, one empty — the probe distinguishes them.
        assert!(t.register(ch.reader, &w, 3));
        assert_eq!(t.wake_past(ch.reader, 4), WakeOutcome::Woken);
        assert_eq!(t.wake_past(ch.writer, 4), WakeOutcome::Empty);
        let m = t.metrics();
        assert_eq!((m.woken, m.empty), (1, 1));
    }
}

// Loom model: the register/commit race explored exhaustively. Run with
//   RUSTFLAGS="--cfg loom" cargo test -p pie --features ptir --lib \
//     driver::waker::loom_tests --release
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicBool, AtomicU64 as LAtomicU64, Ordering as O};
    use std::task::{RawWaker, RawWakerVTable, Waker};

    /// A loom-friendly waker: sets a flag the waiter thread spins on
    /// (loom's explicit yield makes the spin explorable).
    fn flag_waker(flag: Arc<AtomicBool>) -> Waker {
        fn clone(p: *const ()) -> RawWaker {
            let a = unsafe { Arc::from_raw(p as *const AtomicBool) };
            let b = a.clone();
            std::mem::forget(a);
            RawWaker::new(Arc::into_raw(b) as *const (), &VT)
        }
        fn wake(p: *const ()) {
            let a = unsafe { Arc::from_raw(p as *const AtomicBool) };
            a.store(true, O::SeqCst);
        }
        fn wake_by_ref(p: *const ()) {
            let a = unsafe { Arc::from_raw(p as *const AtomicBool) };
            a.store(true, O::SeqCst);
            std::mem::forget(a);
        }
        fn drop_raw(p: *const ()) {
            unsafe { drop(Arc::from_raw(p as *const AtomicBool)) };
        }
        static VT: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop_raw);
        unsafe { Waker::from_raw(RawWaker::new(Arc::into_raw(flag) as *const (), &VT)) }
    }

    /// B9 exhaustive race: waiter observes e=0 and registers; committer
    /// concurrently bumps the index to 1 and wake_past(1). Assert: the
    /// waiter, following register-then-recheck, ALWAYS either sees the
    /// commit on its re-check or gets the wake — never both missed.
    #[test]
    fn register_commit_race_no_lost_wake() {
        loom::model(|| {
            let table = Arc::new(WakerTable::new());
            let id = table.alloc();
            let head = Arc::new(LAtomicU64::new(0));
            let woken = Arc::new(AtomicBool::new(false));

            let committer = {
                let (table, head) = (table.clone(), head.clone());
                loom::thread::spawn(move || {
                    head.store(1, O::SeqCst);
                    table.wake_past(id, 1);
                })
            };

            // Waiter: observe → register → MANDATORY re-check.
            let e = head.load(O::SeqCst);
            if e == 0 {
                let w = flag_waker(woken.clone());
                assert!(table.register(id, &w, e));
                let ready_now = head.load(O::SeqCst) > e;
                committer.join().unwrap();
                // After the committer finished: if the re-check hadn't seen
                // it, the wake MUST have been delivered.
                if !ready_now {
                    assert!(
                        woken.load(O::SeqCst),
                        "lost wake: neither re-check nor waker fired"
                    );
                }
            } else {
                committer.join().unwrap();
            }
            table.free(id); // drop any parked waker (loom tracks its Arc)
        });
    }

    /// B12 exhaustive: sweep concurrent with registration — the waiter is
    /// always either woken by the sweep or observes the poison flag on its
    /// re-check.
    #[test]
    fn sweep_vs_register_no_hang() {
        loom::model(|| {
            let table = Arc::new(WakerTable::new());
            let id = table.alloc();
            let poisoned = Arc::new(AtomicBool::new(false));
            let woken = Arc::new(AtomicBool::new(false));

            let sweeper = {
                let (table, poisoned) = (table.clone(), poisoned.clone());
                loom::thread::spawn(move || {
                    poisoned.store(true, O::SeqCst);
                    table.sweep(&[id]);
                })
            };

            if !poisoned.load(O::SeqCst) {
                let w = flag_waker(woken.clone());
                assert!(table.register(id, &w, 0));
                let saw_poison = poisoned.load(O::SeqCst);
                sweeper.join().unwrap();
                if !saw_poison {
                    assert!(woken.load(O::SeqCst), "sweep lost: waiter would hang");
                }
            } else {
                sweeper.join().unwrap();
            }
            table.free(id); // drop any parked waker (loom tracks its Arc)
        });
    }
}
