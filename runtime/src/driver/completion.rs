//! # X3 — the completion-wake consumer (Runtime–Driver Boundary, decisions B9–B11)
//!
//! The **runtime side** of X3: turn a *driver completion* into X0
//! [`pie_waker`](crate::driver::waker) wakes. When a fire commits, the driver
//! publishes the new committed ring indices into the instance's pinned word
//! region (B9) and signals the runtime that the instance made progress; this
//! consumer **scans that instance's host-visible channels and wakes per
//! channel** — the endgame the [X1 mock](crate::driver::control) previews with
//! its single-batch [`complete_next`](crate::driver::MockControlPlane) and the
//! X0 [`ChannelWakers`](crate::driver::waker::ChannelWakers) doc describes:
//!
//! > the driver ... calls `pie_wake_past(reader, head)` after a net put commits
//! > / `pie_wake_past(writer, tail)` after a net take commits.
//!
//! For embedded drivers the *runtime* owns that scan: it reads the committed
//! head/tail straight from the pinned words (B8/B13 — reads are pure loads,
//! never through the driver) and issues the epoch-filtered
//! [`wake_past`](crate::driver::waker::WakerTable::wake_past). The value path
//! never travels through the driver; only the *signal* ("instance N
//! committed") does.
//!
//! ## What this lands, and what it does not
//!
//! This is the **consumer mechanism**, built on X0's waker table and X1's
//! `enqueue → completion` shape. It is deliberately decoupled from the
//! device-side completion *signal* (built in parallel):
//!
//! - [`CommittedIndex`] abstracts *reading* a driver-published ring index. The
//!   mock backs it with an `Arc<AtomicU64>`; the CUDA driver backs it with an
//!   acquire load of the pinned word at `word_base + offset` (B9). The consumer
//!   only ever reads.
//! - [`CompletionSource`] is the driver→runtime **signal seam**: the CUDA side
//!   implements the pull side that names the instances whose fire committed.
//!   How the consumer learns there *is* work (the doorbell / event-driven fire
//!   rule, X4) is the wake protocol reconciled separately — this module offers
//!   both a push entry ([`CompletionConsumer::scan_instance`]) and a pull drain
//!   ([`CompletionConsumer::drain`]) so either wiring composes.
//!
//! Correctness rides X0 unchanged: the driver publishes the committed index
//! **before** it signals (B11 publish-before-wake), and each per-channel
//! [`wake_past`](crate::driver::waker::WakerTable::wake_past) is race-free by
//! the register-then-recheck protocol — a commit that lands before the host
//! future parks is caught by the future's mandatory re-check; one that lands
//! after is caught by this wake. A completion naming an already-closed instance
//! is a benign race (close raced the last fire), counted, never a trap.
//!
//! ## Composition with X4 (enqueue-ahead) — the pinned seam
//!
//! X3 and X4 (`EnqueueAhead`) are **parallel consumers of one driver
//! completion**, not a caller/callee pair:
//!
//! - **X4 — control-flow / pacing.** `EnqueueAhead::submit`/`pump` awaits the
//!   per-batch X1 [`Completion`](crate::driver::Completion) returned by
//!   [`ControlPlane::enqueue`](crate::driver::ControlPlane::enqueue). That
//!   completion is resolved by the driver advancing its *per-batch word* + slot
//!   (mock: `complete_next`; CUDA: `cuda_carry_done`). This retires the oldest
//!   fire so the next enqueues — the bubble-hiding window.
//! - **X3 — data-flow / output drain.** This [`CompletionConsumer`] wakes the
//!   *per-channel* [`ChannelWakers`] so a host future reading the fire's
//!   **device-resident output** (X4's "producer link drained on demand")
//!   resolves. Its [`CommittedIndex`] reads the same pinned `word_base` region
//!   the X2 carrier publishes — a distinct word from X4's per-batch completion.
//!
//! **The contract (so X4's scheduler wire-in is a clean flip):** each driver
//! completion for instance `B` drives *both* — resolve `B`'s `Completion` (X4
//! retire) **and** scan `B`'s channels (X3 output wakes) — off one signal, so a
//! fire retiring ⟺ its output channels are scanned. guru **ratified the inline
//! callback calling [`scan_instance`](CompletionConsumer::scan_instance)
//! directly**: its registry lock is a brief snapshot (the `Arc` clone in
//! [`channels_for`](CompletionConsumer::channels_for)) that is **dropped before
//! any wake**, so it never holds a lock across the foreign waker — within the
//! `cudaLaunchHostFunc` no-locks-across-the-boundary rule. So the CUDA carrier's
//! `done` host-func fires both: delta's per-batch `wake_past`, then
//! `CompletionConsumer::global().scan_instance(instance)`. [`scan_channels`] is
//! the **optional strictly-lock-free** alternative (zero registry touch) for a
//! caller that captures the [`Arc<[ChannelScan]>`] at enqueue and prefers no lock
//! at all; both are correct. Neither type imports the other; the shared surface
//! is the [`ControlPlane`](crate::driver::ControlPlane) + the driver signal.
//!
//! [`scan_channels`]: CompletionConsumer::scan_channels

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::driver::control::InstanceId;
use crate::driver::waker::{ChannelWakers, WakeOutcome, WakerTable};

/// Reads the driver-published **committed ring index** for one channel
/// endpoint (B9). The committer publishes this word *before* signalling the
/// completion (B11), so an acquire load here never races ahead of the value
/// the woken host future will read.
///
/// The mock backs it with a shared [`Arc<AtomicU64>`]; the CUDA driver backs
/// it with an acquire load of the pinned word at `word_base + offset` (the
/// device DMAs the committed index into the instance's pinned mirror). The
/// consumer only ever *reads* — this is the host-side dual of C2, never a call
/// back through the driver.
pub trait CommittedIndex: Send + Sync {
    /// Acquire-load the current committed ring index.
    fn load(&self) -> u64;
}

impl CommittedIndex for Arc<AtomicU64> {
    fn load(&self) -> u64 {
        self.as_ref().load(Ordering::Acquire)
    }
}

/// A [`CommittedIndex`] backed by the **pinned ring word the CUDA driver
/// publishes** — the X2 device path (bravo's `CudaControlPlane`).
/// [`ControlPlane::bind_instance`](crate::driver::ControlPlane::bind_instance)
/// returns the instance's [`FrameAddresses`](crate::driver::FrameAddresses); the
/// carrier stores `word[word_index] = committed_index` into the pinned word
/// region at `word_base` *before* it signals (B9/B11), so reading it is a plain
/// acquire load of that pinned host word — never a call back through the driver.
///
/// **PROVISIONAL — flip-ready for X2.** The exact per-channel `word_index` is
/// guru's frame-layout reconcile (the X2 carrier's `word_index`, word 0 today);
/// [`from_word_base`](Self::from_word_base) computes `word_base + word_index *
/// WORD_BYTES` against that placeholder layout. When bravo's X2 frames land the
/// [`CompletionConsumer`] scans this in place of the mock's [`Arc<AtomicU64>`] —
/// no other consumer change. Soundness rests on B6: the frame address never
/// moves for the instance's lifetime, so the pinned word stays valid and fixed.
pub struct PinnedRingWord {
    /// Points into the instance's pinned word region (`word_base + offset`).
    /// The driver only ever advances it; the consumer only ever reads it.
    word: *const AtomicU64,
}

// SAFETY: `word` addresses pinned host memory the driver keeps live and fixed
// for the instance's lifetime (B6). The consumer only acquire-loads it (never
// writes, never frees) and the driver publishes with a release store before
// signalling (B11), so sharing/sending the read handle across runtime threads
// is sound.
unsafe impl Send for PinnedRingWord {}
unsafe impl Sync for PinnedRingWord {}

impl PinnedRingWord {
    /// Size of one pinned ring word (the driver's `u64` word).
    pub const WORD_BYTES: usize = std::mem::size_of::<u64>();

    /// Wrap a raw pointer to a live pinned ring word.
    ///
    /// # Safety
    /// `word` must be a valid, aligned, live pinned-word address that stays
    /// valid for as long as this `PinnedRingWord` is read (the instance's
    /// lifetime, B6).
    pub unsafe fn from_raw(word: *const AtomicU64) -> Self {
        PinnedRingWord { word }
    }

    /// Derive the pinned-word address from the X2 `word_base` and a per-channel
    /// `word_index` (`word_base + word_index * WORD_BYTES`) — the provisional
    /// layout mapping guru reconciles.
    ///
    /// **Proposed channel-word layout (for bravo's `pie_frame_carry` publish
    /// indices to match these reader indices):** the `word_base` region is a flat
    /// array of ring words dedicated to X3's per-channel commit indices — for
    /// host-visible channel `c`, the committed *head* (put index, wakes the
    /// reader) at `word_index = 2*c` and the committed *tail* (take index, wakes
    /// the writer) at `2*c + 1`. delta's per-batch `Completion` word is NOT in
    /// this region (it stays the carrier's Rust `Arc<AtomicU64>`), so the two
    /// never alias. The carrier publishes these words stream-ordered before it
    /// fires `done` (B11). guru finalizes the exact assignment; bravo + X3 agree
    /// on this contiguous head-then-tail shape so the indices line up.
    ///
    /// # Safety
    /// `word_base` must be the instance's real pinned word-region base (from
    /// [`bind_instance`](crate::driver::ControlPlane::bind_instance)) and
    /// `word_index` within that region's word count.
    pub unsafe fn from_word_base(word_base: u64, word_index: usize) -> Self {
        let addr = word_base + (word_index * Self::WORD_BYTES) as u64;
        // SAFETY: forwarded to the caller's `from_word_base` contract above.
        unsafe { Self::from_raw(addr as *const AtomicU64) }
    }
}

impl CommittedIndex for PinnedRingWord {
    fn load(&self) -> u64 {
        // SAFETY: `word` is a live, aligned pinned cell per the constructor
        // contract; this acquire load pairs with the driver's release publish.
        unsafe { (*self.word).load(Ordering::Acquire) }
    }
}

/// One host-visible SPSC channel's completion-scan descriptor: its two X0
/// waiter slots plus how to read the committed head/tail (B10).
///
/// On a fire commit the consumer wakes:
///   - `wakers.reader` past the committed **head** — a net *put* committed, so
///     a host `take().await` can make progress.
///   - `wakers.writer` past the committed **tail** — a net *take* committed, so
///     a host `put().await` blocked on back-pressure has room.
///
/// Both endpoints are scanned on every completion; [`wake_past`] is
/// epoch-filtered, so an endpoint whose index did not pass its waiter's
/// observation is a cheap [`WakeOutcome::Filtered`] no-op, not a spurious wake.
///
/// [`wake_past`]: crate::driver::waker::WakerTable::wake_past
pub struct ChannelScan {
    /// The channel's two X0 waiter slots (reader + writer).
    pub wakers: ChannelWakers,
    /// Reads the committed put index (the head the reader waits to pass).
    pub head: Box<dyn CommittedIndex>,
    /// Reads the committed take index (the tail the writer waits to pass).
    pub tail: Box<dyn CommittedIndex>,
}

impl ChannelScan {
    /// Convenience constructor for host-backed (mock) channels, where the
    /// committed indices are shared atomics.
    pub fn host_backed(wakers: ChannelWakers, head: Arc<AtomicU64>, tail: Arc<AtomicU64>) -> Self {
        ChannelScan {
            wakers,
            head: Box::new(head),
            tail: Box::new(tail),
        }
    }
}

/// The driver→runtime **completion signal seam** (the pull side of X3). The
/// CUDA completion-signal side (built in parallel) implements this to name the
/// instances whose fire committed since the last drain; the mock implements it
/// over its in-flight queue.
///
/// Non-blocking by contract: [`drain_completed`](Self::drain_completed) returns
/// immediately with whatever has committed (empty if nothing yet) and never
/// blocks the caller across a device wait. *How* the consumer is told there is
/// work to drain (a doorbell / event loop, X4) is the wake protocol reconciled
/// separately; this trait is only the pull.
pub trait CompletionSource: Send + Sync {
    /// Append the instance ids whose fire committed since the last call to
    /// `out`. Must not block.
    fn drain_completed(&self, out: &mut Vec<InstanceId>);
}

/// Tally of one scan pass — the probe surface (`wakes-per-fire`) and the drift
/// detector: `instances_unknown > 0` means a completion named an instance the
/// consumer has no registration for (signal/registry disagreement or a
/// close-vs-fire race).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanReport {
    /// Registered instances whose channels were scanned.
    pub instances_scanned: usize,
    /// Completions naming an instance with no registration (benign race or
    /// drift — never a trap).
    pub instances_unknown: usize,
    /// Total host-visible channels visited across the scanned instances.
    pub channels_scanned: usize,
    /// Reader endpoints actually woken ([`WakeOutcome::Woken`]).
    pub reader_wakes: usize,
    /// Writer endpoints actually woken ([`WakeOutcome::Woken`]).
    pub writer_wakes: usize,
}

impl ScanReport {
    fn merge(&mut self, other: ScanReport) {
        self.instances_scanned += other.instances_scanned;
        self.instances_unknown += other.instances_unknown;
        self.channels_scanned += other.channels_scanned;
        self.reader_wakes += other.reader_wakes;
        self.writer_wakes += other.writer_wakes;
    }
}

/// The X3 runtime completion consumer. Holds the registry of live instances ×
/// their host-visible channels and turns a driver completion into per-channel
/// X0 wakes. Direct-call and lock-light: the registry mutex is never held
/// across a device wait (the scan only reads pinned words + wakes), matching
/// the X1 direct-control-plane discipline (B1).
///
/// It generalizes [`MockControlPlane::complete_next`] — a single per-batch slot
/// wake — to the multi-channel scan a real driver's completion handler needs,
/// using the same [`wake_past`](WakerTable::wake_past) primitive the X1
/// [`Completion`](crate::driver::Completion) future parks against.
///
/// [`MockControlPlane::complete_next`]: crate::driver::MockControlPlane::complete_next
pub struct CompletionConsumer {
    table: &'static WakerTable,
    registry: Mutex<HashMap<InstanceId, Arc<[ChannelScan]>>>,
}

impl CompletionConsumer {
    /// Construct a consumer over an explicit waker table (tests / loom).
    pub fn new(table: &'static WakerTable) -> CompletionConsumer {
        CompletionConsumer {
            table,
            registry: Mutex::new(HashMap::new()),
        }
    }

    /// Construct a consumer over the process-global X0 table — the same table
    /// the `pie_wake*` FFI and the [`MockControlPlane`](crate::driver::MockControlPlane)
    /// dispatch into.
    pub fn with_global() -> CompletionConsumer {
        CompletionConsumer::new(WakerTable::global())
    }

    /// The **process-global consumer** over the global X0 table — the single
    /// registry the embedded driver's one completion trigger reaches. This is
    /// what lets a free `extern "C"` completion callback (X2's `cuda_carry_done`,
    /// which cannot capture a handle) drive the per-channel scan. guru **ratified
    /// the inline callback calling [`scan_instance`](Self::scan_instance)
    /// directly** — its registry lock is a brief snapshot dropped before the
    /// wakes (never held across the foreign waker), within the
    /// `cudaLaunchHostFunc` rule:
    ///
    /// ```ignore
    /// // inside cuda_carry_done (host-func), after delta's per-batch wake:
    /// CompletionConsumer::global().scan_instance(instance);
    /// ```
    ///
    /// [`scan_channels`](Self::scan_channels) is the optional strictly-lock-free
    /// alternative: capture the `Arc<[ChannelScan]>` at enqueue via
    /// [`channels_for`](Self::channels_for) and scan it with zero registry touch.
    /// [`drain`](Self::drain) is the pull-loop entry. All three are correct.
    ///
    /// PROVISIONAL: a global singleton fits the single embedded driver; if guru's
    /// reconcile picks the pull model (a runtime loop draining a
    /// [`CompletionSource`]) or multiple drivers, the consumer moves behind that
    /// owner and this singleton retires. Mirrors [`WakerTable::global`].
    pub fn global() -> &'static CompletionConsumer {
        static GLOBAL: OnceLock<CompletionConsumer> = OnceLock::new();
        GLOBAL.get_or_init(CompletionConsumer::with_global)
    }

    /// The X0 [`WakerTable`] this consumer wakes into. **The channel wakers a
    /// caller registers (`ChannelWakers::alloc`) and the host reader parks on
    /// MUST come from this same table** — otherwise a scan's `wake_past` targets
    /// a different table and the reader never wakes. Exposed so the CUDA carrier
    /// / bind site allocate on the guaranteed-correct table:
    /// `ChannelWakers::alloc(CompletionConsumer::global().table())`.
    pub fn table(&self) -> &'static WakerTable {
        self.table
    }

    /// Register (or replace) an instance's host-visible channels. Called at
    /// bind time, after [`ControlPlane::bind_instance`](crate::driver::ControlPlane::bind_instance)
    /// hands back the instance's fixed [`FrameAddresses`](crate::driver::FrameAddresses):
    /// the channel scans carry the pinned word readers derived from
    /// `word_base` and the [`ChannelWakers`] allocated for each channel.
    ///
    /// Returns the shared [`Arc<[ChannelScan]>`] so the caller can capture it
    /// directly into the completion context (the CUDA `CarryWake`) — the inline
    /// host-func then calls [`scan_channels`](Self::scan_channels) on it with
    /// **zero registry lookup and zero lock**, satisfying guru's no-lock rule for
    /// the `cudaLaunchHostFunc` thread. [`channels_for`](Self::channels_for) is
    /// the equivalent lookup for a call site (e.g. `enqueue`) that only has the
    /// instance id.
    pub fn register_instance(&self, id: InstanceId, channels: Vec<ChannelScan>) -> Arc<[ChannelScan]> {
        let arc: Arc<[ChannelScan]> = Arc::from(channels);
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, Arc::clone(&arc));
        arc
    }

    /// Look up an instance's shared channel handle (a cheap `Arc` refcount
    /// bump). Runtime-thread call (takes the registry lock) — use it at
    /// `enqueue` to capture the [`Arc<[ChannelScan]>`] into the per-batch
    /// completion context, so the **host-func** later calls
    /// [`scan_channels`](Self::scan_channels) lock-free. Returns `None` for an
    /// unknown/closed instance.
    pub fn channels_for(&self, id: InstanceId) -> Option<Arc<[ChannelScan]>> {
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&id)
            .map(Arc::clone)
    }

    /// Retire an instance (B6/B12). Sweeps every channel's waiter slots so a
    /// host future blocked on this instance re-polls, observes the closure, and
    /// resolves instead of hanging; then frees the slots and drops the
    /// registration. Returns `true` if the instance was registered.
    ///
    /// If an in-flight completion still holds an [`Arc<[ChannelScan]>`] for this
    /// instance (a close racing a fire's host-func), the `ChannelScan` structs
    /// stay alive via that Arc, but their waker slots are freed here — so a
    /// racing [`scan_channels`](Self::scan_channels)'s `wake_past` hits a bumped
    /// generation and is a harmless X0 no-op (B10). The pinned words' *memory*
    /// validity is the driver's B6 grace period (do not free the frame until
    /// in-flight passes retire); this consumer only guarantees the Rust side.
    pub fn close_instance(&self, id: InstanceId) -> bool {
        let channels = self
            .registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
        match channels {
            Some(channels) => {
                for ch in channels.iter() {
                    // B12: wake both endpoints unconditionally so blocked
                    // waiters re-poll and see closure, then recycle the slots.
                    ch.wakers.sweep(self.table);
                    ch.wakers.free(self.table);
                }
                true
            }
            None => false,
        }
    }

    /// Number of currently registered instances.
    pub fn registered(&self) -> usize {
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// **The lock-free hot path (host-func safe).** Scan an explicit channel
    /// slice — read each channel's committed head/tail and wake its
    /// reader/writer past those indices (B9/B11). Takes **no registry lock** and
    /// allocates nothing: it only issues `CommittedIndex::load` (plain atomic
    /// acquire loads) + [`wake_past`](WakerTable::wake_past) — exactly the X0
    /// primitive delta's pacing wake already calls in the same `cudaLaunchHostFunc`
    /// thread, so it adds no lock guru's inline-callback ruling forbids.
    ///
    /// The caller (the CUDA `CarryWake`) holds the instance's
    /// [`Arc<[ChannelScan]>`] captured at bind/enqueue, so the host-func resolves
    /// id→channels with zero lookup. Runtime-thread callers use
    /// [`scan_instance`](Self::scan_instance), which looks the Arc up first.
    pub fn scan_channels(&self, channels: &[ChannelScan]) -> ScanReport {
        let mut report = ScanReport {
            channels_scanned: channels.len(),
            ..ScanReport::default()
        };
        for ch in channels {
            if self.table.wake_past(ch.wakers.reader, ch.head.load()) == WakeOutcome::Woken {
                report.reader_wakes += 1;
            }
            if self.table.wake_past(ch.wakers.writer, ch.tail.load()) == WakeOutcome::Woken {
                report.writer_wakes += 1;
            }
        }
        report
    }

    /// **The X3 scan (runtime-thread path).** For one committed instance, look
    /// up its channels (a cheap `Arc` clone under the registry lock) and hand
    /// them to the lock-free [`scan_channels`](Self::scan_channels). A completion
    /// for an unknown/closed instance is a benign race: counted in
    /// [`ScanReport::instances_unknown`], never a trap.
    ///
    /// The lock is held only for the `Arc` refcount bump and dropped **before**
    /// any `wake` runs — `wake` invokes a foreign [`Waker`](std::task::Waker), so
    /// it must never run under a lock spanning the boundary (the X0 table's "wake
    /// outside every lock" discipline). Because that snapshot lock is never held
    /// across a wake, **guru ratified calling this directly from the inline
    /// `cudaLaunchHostFunc` callback** (see [`global`](Self::global)).
    /// [`scan_channels`](Self::scan_channels) is the optional strictly-lock-free
    /// alternative for a caller that has already captured the channels' `Arc`.
    pub fn scan_instance(&self, id: InstanceId) -> ScanReport {
        let channels = self.channels_for(id);
        let Some(channels) = channels else {
            return ScanReport {
                instances_unknown: 1,
                ..ScanReport::default()
            };
        };
        let mut report = self.scan_channels(&channels);
        report.instances_scanned = 1;
        report
    }

    /// Scan a batch of committed instances (the driver's completion handler
    /// hands the runtime the set that committed in one fire). Sums the
    /// per-instance [`ScanReport`]s.
    pub fn scan(&self, committed: &[InstanceId]) -> ScanReport {
        let mut report = ScanReport::default();
        for &id in committed {
            report.merge(self.scan_instance(id));
        }
        report
    }

    /// Pull side: drain a [`CompletionSource`] and scan everything it reports.
    /// One turn of the consumer loop — the driving loop (when to call this) is
    /// the reconciled wake protocol (X4), not this module.
    pub fn drain(&self, source: &dyn CompletionSource) -> ScanReport {
        let mut committed = Vec::new();
        source.drain_completed(&mut committed);
        self.scan(&committed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;
    use std::task::{Wake, Waker};

    /// A waker that records whether it was woken — lets a scan prove a wake
    /// reached a genuinely parked future through the X0 table.
    struct Flag(AtomicU32);
    impl Wake for Flag {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn flag() -> (Arc<Flag>, Waker) {
        let f = Arc::new(Flag(AtomicU32::new(0)));
        (f.clone(), Waker::from(f))
    }

    /// Leak a fresh table so it satisfies the `&'static` the consumer holds
    /// (the process-global table is shared across tests; a per-test table
    /// keeps the metrics/slot assertions independent).
    fn table() -> &'static WakerTable {
        Box::leak(Box::new(WakerTable::new()))
    }

    fn word(v: u64) -> Arc<AtomicU64> {
        Arc::new(AtomicU64::new(v))
    }

    #[test]
    fn scan_wakes_reader_when_head_advances() {
        let t = table();
        let consumer = CompletionConsumer::new(t);

        let wakers = ChannelWakers::alloc(t);
        let head = word(0);
        let tail = word(0);
        consumer.register_instance(
            7,
            vec![ChannelScan::host_backed(wakers, head.clone(), tail.clone())],
        );

        // A host `take` parks on the reader, having observed head == 0.
        let (f, w) = flag();
        assert!(t.register(wakers.reader, &w, 0));

        // The driver commits a put: publish the head (B11), then signal.
        head.store(1, Ordering::Release);
        let report = consumer.scan_instance(7);

        assert_eq!(f.0.load(Ordering::SeqCst), 1, "reader woken through X0");
        assert_eq!(report.instances_scanned, 1);
        assert_eq!(report.channels_scanned, 1);
        assert_eq!(report.reader_wakes, 1);
        assert_eq!(report.writer_wakes, 0);
        assert_eq!(report.instances_unknown, 0);
    }

    #[test]
    fn scan_is_epoch_filtered_when_index_unchanged() {
        let t = table();
        let consumer = CompletionConsumer::new(t);
        let wakers = ChannelWakers::alloc(t);
        let head = word(3);
        consumer.register_instance(
            1,
            vec![ChannelScan::host_backed(wakers, head.clone(), word(0))],
        );

        // Reader observed head == 3 and parks; the fire committed nothing new
        // on this channel (head still 3) → wake_past(reader, 3) is Filtered.
        let (f, w) = flag();
        assert!(t.register(wakers.reader, &w, 3));
        let report = consumer.scan_instance(1);

        assert_eq!(f.0.load(Ordering::SeqCst), 0, "no spurious wake");
        assert_eq!(report.reader_wakes, 0);
        assert_eq!(report.channels_scanned, 1);
    }

    #[test]
    fn scan_wakes_writer_on_committed_take() {
        let t = table();
        let consumer = CompletionConsumer::new(t);
        let wakers = ChannelWakers::alloc(t);
        let tail = word(0);
        consumer.register_instance(
            2,
            vec![ChannelScan::host_backed(wakers, word(0), tail.clone())],
        );

        // A host `put` blocked on back-pressure parks on the writer at tail 0.
        let (f, w) = flag();
        assert!(t.register(wakers.writer, &w, 0));
        tail.store(1, Ordering::Release); // driver committed a take
        let report = consumer.scan_instance(2);

        assert_eq!(f.0.load(Ordering::SeqCst), 1);
        assert_eq!(report.writer_wakes, 1);
        assert_eq!(report.reader_wakes, 0);
    }

    #[test]
    fn completion_for_unknown_instance_is_benign() {
        let t = table();
        let consumer = CompletionConsumer::new(t);
        let report = consumer.scan_instance(999);
        assert_eq!(report.instances_unknown, 1);
        assert_eq!(report.instances_scanned, 0);
        assert_eq!(report.channels_scanned, 0);
    }

    #[test]
    fn close_instance_sweeps_blocked_waiter_and_deregisters() {
        let t = table();
        let consumer = CompletionConsumer::new(t);
        let wakers = ChannelWakers::alloc(t);
        consumer.register_instance(5, vec![ChannelScan::host_backed(wakers, word(0), word(0))]);
        assert_eq!(consumer.registered(), 1);

        // A waiter parked on a channel that will never advance; close must
        // wake it (B12) so its future re-polls and sees closure, not a hang.
        let (f, w) = flag();
        assert!(t.register(wakers.reader, &w, 0));

        assert!(consumer.close_instance(5), "was registered");
        assert_eq!(f.0.load(Ordering::SeqCst), 1, "swept on close");
        assert_eq!(consumer.registered(), 0);
        // A late completion for the closed instance is a benign race.
        assert_eq!(consumer.scan_instance(5).instances_unknown, 1);
        // Double close is a benign false, never a trap.
        assert!(!consumer.close_instance(5));
    }

    /// A mock pull source, as the CUDA completion-signal side will implement.
    struct MockSource(Mutex<VecDeque<InstanceId>>);
    impl CompletionSource for MockSource {
        fn drain_completed(&self, out: &mut Vec<InstanceId>) {
            let mut q = self.0.lock().unwrap();
            out.extend(q.drain(..));
        }
    }

    #[test]
    fn drain_pulls_from_source_and_scans_each() {
        let t = table();
        let consumer = CompletionConsumer::new(t);

        let (wa, ha) = (ChannelWakers::alloc(t), word(0));
        let (wb, hb) = (ChannelWakers::alloc(t), word(0));
        consumer.register_instance(10, vec![ChannelScan::host_backed(wa, ha.clone(), word(0))]);
        consumer.register_instance(11, vec![ChannelScan::host_backed(wb, hb.clone(), word(0))]);

        let (fa, wla) = flag();
        let (fb, wlb) = flag();
        assert!(t.register(wa.reader, &wla, 0));
        assert!(t.register(wb.reader, &wlb, 0));
        ha.store(1, Ordering::Release);
        hb.store(1, Ordering::Release);

        let source = MockSource(Mutex::new(VecDeque::from([10, 11])));
        let report = consumer.drain(&source);

        assert_eq!(report.instances_scanned, 2);
        assert_eq!(report.reader_wakes, 2);
        assert_eq!(fa.0.load(Ordering::SeqCst), 1);
        assert_eq!(fb.0.load(Ordering::SeqCst), 1);

        // Source is drained: a second turn scans nothing.
        assert_eq!(consumer.drain(&source), ScanReport::default());
    }

    #[test]
    fn scan_channels_is_lock_free_hot_path_via_captured_arc() {
        // The inline host-func path: capture the Arc at "enqueue" (channels_for),
        // then scan it directly — no registry lookup, matching guru's no-lock
        // rule for the cudaLaunchHostFunc thread.
        let t = table();
        let consumer = CompletionConsumer::new(t);
        let wakers = ChannelWakers::alloc(t);
        let head = word(0);
        consumer.register_instance(
            7,
            vec![ChannelScan::host_backed(wakers, head.clone(), word(0))],
        );

        // Capture the shared channel handle (what the CarryWake holds).
        let channels = consumer.channels_for(7).expect("registered");

        let (f, w) = flag();
        assert!(t.register(wakers.reader, &w, 0));
        head.store(1, Ordering::Release); // driver commit

        // Host-func: scan the captured Arc directly (no id, no registry lock).
        let report = consumer.scan_channels(&channels);
        assert_eq!(f.0.load(Ordering::SeqCst), 1, "reader woken lock-free");
        assert_eq!(report.reader_wakes, 1);
        assert_eq!(report.channels_scanned, 1);
        // scan_channels reports no instance accounting (it has no id).
        assert_eq!(report.instances_scanned, 0);
        assert_eq!(report.instances_unknown, 0);
    }

    #[test]
    fn register_returns_arc_and_channels_for_shares_it() {
        let t = table();
        let consumer = CompletionConsumer::new(t);
        let wakers = ChannelWakers::alloc(t);
        let from_register =
            consumer.register_instance(3, vec![ChannelScan::host_backed(wakers, word(0), word(0))]);
        let from_lookup = consumer.channels_for(3).expect("registered");
        // Both handles are the SAME allocation (shared Arc, no ChannelScan clone).
        assert!(Arc::ptr_eq(&from_register, &from_lookup));
        assert_eq!(from_register.len(), 1);
        // Unknown instance yields None (benign).
        assert!(consumer.channels_for(999).is_none());
    }

    #[test]
    fn captured_arc_outlives_close_and_scan_is_a_safe_noop() {
        // A close racing an in-flight host-func: the captured Arc keeps the
        // ChannelScans alive, and close frees the waker slots so the racing
        // scan_channels' wake_past is a bumped-generation X0 no-op (B10) — no
        // wake, no trap, no use-after-free of the Rust structs.
        let t = table();
        let consumer = CompletionConsumer::new(t);
        let wakers = ChannelWakers::alloc(t);
        let head = word(0);
        let channels = consumer
            .register_instance(4, vec![ChannelScan::host_backed(wakers, head.clone(), word(0))]);

        let (f, w) = flag();
        assert!(t.register(wakers.reader, &w, 0));

        assert!(consumer.close_instance(4), "was registered");
        // The sweep already woke the parked waiter once (B12).
        assert_eq!(f.0.load(Ordering::SeqCst), 1);

        // In-flight host-func fires AFTER close, still holding the Arc:
        head.store(1, Ordering::Release);
        let report = consumer.scan_channels(&channels);
        // The freed slot's generation was bumped → wake_past is Stale, no wake.
        assert_eq!(report.reader_wakes, 0, "freed slot wake is a no-op");
        assert_eq!(f.0.load(Ordering::SeqCst), 1, "no second wake");
        assert_eq!(report.channels_scanned, 1);
    }

    #[test]
    fn pinned_ring_word_reads_driver_published_index() {
        // A leaked cell stands in for the driver's pinned ring word (a stable
        // address the "driver" publishes into); proves the unsafe read is sound.
        let cell: &'static AtomicU64 = Box::leak(Box::new(AtomicU64::new(0)));
        let pw = unsafe { PinnedRingWord::from_raw(cell as *const AtomicU64) };
        assert_eq!(pw.load(), 0);
        cell.store(42, Ordering::Release); // driver publishes a commit
        assert_eq!(pw.load(), 42, "reads the driver-published index");
        // The trait object path the consumer actually uses.
        let boxed: Box<dyn CommittedIndex> = Box::new(pw);
        assert_eq!(boxed.load(), 42);
    }

    #[test]
    fn pinned_ring_word_from_word_base_offsets_by_word_index() {
        // Two contiguous pinned words; from_word_base(base, 1) reads the second.
        let cells: &'static [AtomicU64; 2] =
            Box::leak(Box::new([AtomicU64::new(10), AtomicU64::new(20)]));
        let base = &cells[0] as *const AtomicU64 as u64;
        let w0 = unsafe { PinnedRingWord::from_word_base(base, 0) };
        let w1 = unsafe { PinnedRingWord::from_word_base(base, 1) };
        assert_eq!(w0.load(), 10);
        assert_eq!(w1.load(), 20, "word_index 1 lands on the second word");
    }

    #[test]
    fn global_consumer_is_a_stable_singleton() {
        // The single registry the embedded driver's one completion trigger
        // reaches; must be idempotent (same address) so the FFI callback and the
        // bind site see one registry.
        let a = CompletionConsumer::global() as *const CompletionConsumer;
        let b = CompletionConsumer::global() as *const CompletionConsumer;
        assert_eq!(a, b, "global() returns the same singleton");
    }

    /// **Multi-fire regression — pins the producer invariant.** A MONOTONIC
    /// committed head (the producer's `committed_head += n_this_fire`, charlie's
    /// executor) wakes the output reader on EVERY fire: each fire the reader
    /// re-parks observing the last head, and the next monotonic advance passes
    /// that epoch. This is the invariant the CUDA producer must satisfy; the
    /// consumer side (this scan) is correct across fires.
    #[test]
    fn monotonic_head_wakes_reader_every_fire() {
        let t = table();
        let consumer = CompletionConsumer::new(t);
        let wakers = ChannelWakers::alloc(t);
        let head = word(0);
        consumer.register_instance(
            1,
            vec![ChannelScan::host_backed(wakers, head.clone(), word(0))],
        );

        // Three successive fires; the producer advances the head monotonically.
        let mut observed = 0u64;
        for fire in 1..=3u64 {
            let (f, w) = flag();
            assert!(t.register(wakers.reader, &w, observed));
            head.store(fire, Ordering::Release); // committed_head += 1 (monotonic)
            let report = consumer.scan_instance(1);
            assert_eq!(report.reader_wakes, 1, "fire {fire}: monotonic head wakes reader");
            assert_eq!(f.0.load(Ordering::SeqCst), 1, "fire {fire}: reader woken");
            observed = fire; // reader re-parks having consumed up to `fire`
        }
    }

    /// **Multi-fire regression — demonstrates the bug (constant head stalls).**
    /// A CONSTANT head (the mock/carrier `target=1` every fire) wakes fire 1 but
    /// STALLS fire 2+: the reader re-parks observing 1, the producer writes 1
    /// again (no advance), so `wake_past(reader, 1)` with epoch 1 is Filtered and
    /// the reader never wakes — the latent multi-fire decode stall single-fire
    /// tests miss. Locks in why the producer head MUST be monotonic.
    #[test]
    fn constant_head_stalls_second_fire_the_bug() {
        let t = table();
        let consumer = CompletionConsumer::new(t);
        let wakers = ChannelWakers::alloc(t);
        let head = word(0);
        consumer.register_instance(
            1,
            vec![ChannelScan::host_backed(wakers, head.clone(), word(0))],
        );

        // Fire 1: head 0 → 1, reader (observed 0) wakes.
        let (f1, w1) = flag();
        assert!(t.register(wakers.reader, &w1, 0));
        head.store(1, Ordering::Release);
        assert_eq!(consumer.scan_instance(1).reader_wakes, 1);
        assert_eq!(f1.0.load(Ordering::SeqCst), 1);

        // Fire 2: reader re-parks observing 1; constant producer writes 1 AGAIN.
        let (f2, w2) = flag();
        assert!(t.register(wakers.reader, &w2, 1));
        head.store(1, Ordering::Release); // BUG: not monotonic
        let report = consumer.scan_instance(1);
        assert_eq!(report.reader_wakes, 0, "constant head → fire 2 Filtered (stall)");
        assert_eq!(f2.0.load(Ordering::SeqCst), 0, "reader NOT woken → decode stall");
    }
}
