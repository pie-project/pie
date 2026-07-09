//! # X1 — direct control plane (Runtime–Driver Boundary, decisions B1–B7, B14)
//!
//! The hot control plane — program registration, instance bind/close, and
//! batch enqueue — as **direct, bounded, non-blocking in-proc calls** (B1).
//! No submission queue, no polling channel, no response frames for control.
//!
//! ## Why this is off the `DriverChannel` trait (B2)
//!
//! [`DriverChannel`](crate::driver::DriverChannel) keeps `submit`/`notify` for
//! the *observation-shaped* out-of-proc contract (the subprocess path lowers
//! the same semantics to shmem rings + a doorbell). Control verbs get
//! direct-call **fast paths**, not new trait obligations: an embedded driver
//! implements [`ControlPlane`] and the runtime calls it directly, bypassing
//! the request/response transport entirely.
//!
//! ## What X1 lands (mock-first, house rule)
//!
//! This is the **mock** increment: [`MockControlPlane`] stands in for the
//! embedded CUDA/Metal driver so the whole shape — `register_program →
//! bind_instance → enqueue → completion` — proves out with **zero queue hops**
//! before any device code exists. Bind returns B5's address triple
//! ([`FrameAddresses`]); the mock backs it with plain host allocations standing
//! in for the device frame and the pinned mirror/word regions. Completion is an
//! **edge-triggered [`pie_waker`](crate::driver::waker) park** (X0): the
//! blocked host future is woken when the mock commits the batch — the first
//! real consumer of the X0 substrate.
//!
//! CUDA frames + mirrors (X2), the completion-handler channel scan (X3), and
//! the event-driven fire rule (X4) build on this shape; none of them are here.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use anyhow::{anyhow, Result};

use crate::driver::waker::{ChannelWakers, WakerSlotId, WakerTable};

/// Registration-time program handle (B4). Returned by
/// [`ControlPlane::register_program`]; names the trace whose frame layout +
/// per-stage kernels the driver computed once.
pub type ProgramId = u64;

/// Bind-time instance handle (B4/B6). Names one live instance whose frame
/// address is fixed for its lifetime.
pub type InstanceId = u64;

/// B5 — **the bind-time address contract**: everything the per-step data plane
/// needs is derived from these three bases plus the trace-known frame layout,
/// so after bind the data plane exchanges no layout information. On a real
/// driver `frame_base` is device memory and `mirror_base`/`word_base` are
/// pinned host memory; the [`MockControlPlane`] backs all three with ordinary
/// host allocations (the addresses are real and distinct, just not device).
///
/// This is the host-side dual of C2 proposed for the masterplan as **C5 — the
/// boundary is addresses plus wakes**.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameAddresses {
    /// Base of the instance's device frame (cells live at
    /// `frame_base + channel offset + ring index`). Never moves for the
    /// instance's lifetime (B6).
    pub frame_base: u64,
    /// Base of the pinned host mirror the host reads committed cells from
    /// (B8/B13 — reads are pure loads from here, never through the driver).
    pub mirror_base: u64,
    /// Base of the pinned ring-index words the host waits on (B9 — a waiter
    /// registers the index it observed; the driver wakes when the word passes).
    pub word_base: u64,
}

/// The result of [`ControlPlane::bind_instance`]: the instance handle plus its
/// fixed [`FrameAddresses`] and the bind-time [`WakeSlots`] (B15: all waker
/// slots are bind-time fixed and freed only at close; after bind, no layout or
/// wake-target information is ever exchanged again).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundInstance {
    pub id: InstanceId,
    pub addresses: FrameAddresses,
    /// The X0 waker slots the driver wakes DIRECTLY from its instance table
    /// (no per-fire callback context): the pacing slot (woken when `word[0]`,
    /// the committed fire count, advances) plus one reader/writer pair per
    /// host-visible channel.
    pub wakes: WakeSlots,
}

/// The bind-time-fixed X0 waker slots for one instance (B10/B15). The driver
/// holds these next to the instance's pinned words and calls `pie_wake_past`
/// directly — no `CarryWake`/`user_ptr` trampoline. Allocated at bind, freed at
/// close; steady state allocates nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WakeSlots {
    /// The pacing slot, woken when `word[0]` (committed fire count) advances —
    /// the scheduler's run-ahead pacing wait ([`Completion`] parks here).
    pub pacing: WakerSlotId,
    /// Per host-visible channel (dense order): the reader slot (woken on a
    /// committed head/net-put) + writer slot (woken on a committed tail/net-take).
    pub channels: Vec<ChannelWakers>,
}

/// The fixed pinned-word layout (boundary.md §), shared by host + driver: after
/// bind neither side exchanges layout info again (B15). `word[0]` is the
/// instance's pacing counter (committed fire count); host of channel `c` is at
/// [`head`](WordLayout::head), its tail at [`tail`](WordLayout::tail).
pub struct WordLayout;

impl WordLayout {
    /// `word[0]`: the pacing counter (committed fire count, monotonic).
    pub const PACING: usize = 0;
    /// `word[1 + 2c]`: channel `c`'s committed head (net puts).
    pub fn head(channel: usize) -> usize {
        1 + 2 * channel
    }
    /// `word[2 + 2c]`: channel `c`'s committed tail (net takes).
    pub fn tail(channel: usize) -> usize {
        2 + 2 * channel
    }
    /// Number of `u64` ring words for `n_channels` (pacing + head/tail each).
    pub fn words(n_channels: usize) -> usize {
        1 + 2 * n_channels
    }
}

/// A batch handed to [`ControlPlane::enqueue`] (B7 — the launch descriptor).
/// Mock-minimal: the real per-fire descriptor rebuilds `bases[lane]` (+ row
/// maps) so glue kernels address cells as `bases[lane] + offset + index`; here
/// it just names the instance to complete and carries opaque descriptor bytes.
#[derive(Debug, Clone)]
pub struct EnqueueBatch {
    /// The bound instance this fire drives.
    pub instance: InstanceId,
    /// Opaque launch-descriptor bytes (stand-in for `bases[lane]` + row maps).
    pub descriptor: Vec<u8>,
}

/// The completion signal of one enqueued batch (B9/B11) — an **epoch-tagged
/// waker park**, not a response frame. `await`ing it resolves when the driver
/// commits the batch: it advances the completion word and wakes the slot
/// through the X0 table. The value path never travels through the driver.
///
/// Correctness is race-free by the register-then-recheck protocol (X0 B9): if
/// the commit lands before the future parks, the recheck sees the advanced
/// word; otherwise the wake fires. Dropping a still-pending `Completion`
/// cancels the wait; the driver's later wake becomes a generation no-op.
pub struct Completion {
    table: &'static WakerTable,
    slot: WakerSlotId,
    word: Arc<AtomicU64>,
    target: u64,
    /// Whether this completion OWNS its waker slot (a legacy per-batch slot,
    /// freed on drop) or BORROWS a bind-fixed slot (the instance's pacing slot,
    /// freed only at close — B15). Re-backing on `word[0]` + the fixed pacing
    /// slot makes steady-state fires allocate nothing.
    owns_slot: bool,
}

impl Completion {
    /// The waker slot the driver wakes to resolve this completion.
    pub fn slot(&self) -> WakerSlotId {
        self.slot
    }

    /// X2 — construct a completion parked on `word`/`slot` for an **external**
    /// driver (the CUDA carrier) to resolve. The driver advances `word` to
    /// `target` and wakes the slot through the X0 table; the future resolves on
    /// the same register-then-recheck protocol [`enqueue`](MockControlPlane::enqueue)
    /// builds for the mock. Kept crate-visible so `control_cuda` reuses the exact
    /// completion shape rather than duplicating the park. This variant OWNS its
    /// slot (freed on drop); use [`parked_pacing`](Self::parked_pacing) to park
    /// on a bind-fixed pacing slot.
    #[cfg_attr(not(feature = "driver-cuda"), allow(dead_code))]
    pub(crate) fn parked(
        table: &'static WakerTable,
        slot: WakerSlotId,
        word: Arc<AtomicU64>,
        target: u64,
    ) -> Completion {
        Completion { table, slot, word, target, owns_slot: true }
    }

    /// Park on the instance's FIXED pacing slot + `word[0]` (the committed fire
    /// count), targeting this fire's sequence number (B11/B15). The slot is
    /// bind-fixed — dropping the completion does NOT free it (it is freed at
    /// close). This is the run-ahead pacing wait re-backing (boundary.md
    /// Phase 1): steady-state fires allocate no waker slot.
    pub(crate) fn parked_pacing(
        table: &'static WakerTable,
        pacing: WakerSlotId,
        word: Arc<AtomicU64>,
        target: u64,
    ) -> Completion {
        Completion { table, slot: pacing, word, target, owns_slot: false }
    }
}

impl Future for Completion {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        // Fast path: the word already passed the target.
        if this.word.load(Ordering::Acquire) >= this.target {
            this.table.deregister(this.slot);
            return Poll::Ready(());
        }
        // Publish the waker, then MANDATORY re-check (X0 register-then-recheck).
        let observed = this.word.load(Ordering::Acquire);
        if !this.table.register(this.slot, cx.waker(), observed) {
            // Stale slot (channel died between checks): re-poll to surface it.
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        if this.word.load(Ordering::Acquire) >= this.target {
            this.table.deregister(this.slot);
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        // Recycle the per-batch waker slot ONLY if this completion owns it. A
        // completion re-backed on a bind-fixed pacing slot leaves it alone (the
        // slot is freed at close, B15). Any residual driver wake targeting a
        // freed id is a harmless generation no-op (X0 B10).
        if self.owns_slot {
            self.table.free(self.slot);
        }
    }
}

/// **The direct control plane (B1/B4/B14).** Runtime threads call these
/// directly on an embedded driver — each call is append-to-structure work that
/// never takes a lock spanning a GPU wait and never calls a synchronizing
/// device API. Kept off [`DriverChannel`](crate::driver::DriverChannel) (B2):
/// the request/response trait serves the out-of-proc path; these verbs are
/// the in-proc fast path.
pub trait ControlPlane: Send + Sync {
    /// B4 — register a trace: compute its frame layout + per-stage kernels +
    /// the host-visible channel list, once, and return a stable handle.
    /// Nothing per-step re-sends state a word already carries.
    fn register_program(&self, trace: &[u8]) -> Result<ProgramId>;

    /// B4/B5 — bind an instance of a registered program and return its fixed
    /// [`FrameAddresses`]. The frame address never moves for the instance's
    /// lifetime (B6) — the precondition for wakers and direct reads.
    fn bind_instance(&self, program: ProgramId, bindings: &[u8]) -> Result<BoundInstance>;

    /// B6 — release an instance. The frame returns to its slab after every
    /// in-flight pass retires (the §5.2 grace-period discipline).
    fn close_instance(&self, id: InstanceId) -> Result<()>;

    /// B14 — enqueue a batch (the one per-step runtime→driver call). Returns a
    /// [`Completion`] the caller parks on; the driver wakes it on commit.
    fn enqueue(&self, batch: EnqueueBatch) -> Result<Completion>;
}

// ===========================================================================
// Mock driver — the X1 stand-in (house rule: prove the shape before CUDA)
// ===========================================================================

/// Trace-known frame layout the mock computes at registration (B4). Real
/// drivers derive these from the trace's channel list + ring sizes; the mock
/// derives trivial sizes from the trace bytes so bind has something to allocate.
#[derive(Debug, Clone, Copy)]
struct MockProgram {
    frame_size: usize,
    mirror_size: usize,
    word_size: usize,
    /// Host-visible channel count (mock-derived from the trace) — bind allocates
    /// one reader/writer wake pair per channel + the ring words for them.
    n_channels: usize,
}

/// One bound instance's host-backed frame regions (B6 — stable addresses) plus
/// its bind-fixed X0 wake slots + pacing word (B15).
/// Boxed slices never reallocate, so the addresses handed out at bind stay
/// valid for the instance's lifetime.
struct MockInstance {
    #[allow(dead_code)]
    program: ProgramId,
    frame: Box<[u8]>,
    mirror: Box<[u8]>,
    word: Box<[u8]>,
    /// The instance's bind-fixed wake slots (pacing + per-channel), freed at close.
    wakes: WakeSlots,
    /// `word[0]` model: the committed fire count the pacing slot tracks. Fires
    /// re-back their [`Completion`] on THIS shared word (monotonic), not a
    /// per-batch word — the run-ahead pacing wait.
    pacing_word: Arc<AtomicU64>,
    /// Monotonic fire-sequence counter (each enqueue targets `++fire_seq`).
    fire_seq: u64,
}

impl MockInstance {
    fn addresses(&self) -> FrameAddresses {
        FrameAddresses {
            frame_base: self.frame.as_ptr() as u64,
            mirror_base: self.mirror.as_ptr() as u64,
            word_base: self.word.as_ptr() as u64,
        }
    }
}

/// One in-flight batch the mock has accepted but not yet committed. The mock
/// plays the driver's completion handler (X3 preview) via
/// [`MockControlPlane::complete_next`], which advances the instance's pacing
/// `word` and wakes its bind-fixed pacing `slot`.
struct PendingBatch {
    #[allow(dead_code)]
    instance: InstanceId,
    /// The instance's bind-fixed pacing slot (shared across its fires).
    slot: WakerSlotId,
    /// The instance's shared pacing word (`word[0]`).
    word: Arc<AtomicU64>,
    /// This fire's sequence number — the pacing word target.
    target: u64,
}

/// The X1 mock control plane. Direct calls only — it holds no channel, no
/// inbox, no response slot; completion rides the X0 waker table.
///
/// > Alternative endgame (not built, B-note): a real driver's completion
/// > handler scans committed instances × their host-visible channels and wakes
/// > per channel (X3). The mock's `complete_next` is the single-batch preview.
pub struct MockControlPlane {
    table: &'static WakerTable,
    inner: Mutex<MockState>,
}

struct MockState {
    programs: Vec<MockProgram>,
    instances: Vec<Option<MockInstance>>,
    pending: Vec<PendingBatch>,
    next_program: u64,
    next_instance: u64,
}

impl Default for MockControlPlane {
    fn default() -> Self {
        Self::new()
    }
}

impl MockControlPlane {
    pub fn new() -> MockControlPlane {
        MockControlPlane {
            table: WakerTable::global(),
            inner: Mutex::new(MockState {
                programs: Vec::new(),
                instances: Vec::new(),
                pending: Vec::new(),
                next_program: 1,
                next_instance: 1,
            }),
        }
    }

    /// Mock driver-side **completion handler** (X3 preview): commit the oldest
    /// in-flight batch — advance its completion word, then wake the parked host
    /// through the X0 table (publish-before-wake, B11). Returns `false` when
    /// there is nothing in flight. The value publish (word store) is ordered
    /// **before** the wake.
    pub fn complete_next(&self) -> bool {
        let batch = {
            let mut st = self.inner.lock().unwrap();
            if st.pending.is_empty() {
                return false;
            }
            st.pending.remove(0)
        };
        // Publish first (B11): the committed word is visible before the wake.
        batch.word.store(batch.target, Ordering::Release);
        self.table.wake_past(batch.slot, batch.target);
        true
    }

    /// How many batches are enqueued but not yet committed.
    pub fn in_flight(&self) -> usize {
        self.inner.lock().unwrap().pending.len()
    }
}

impl ControlPlane for MockControlPlane {
    fn register_program(&self, trace: &[u8]) -> Result<ProgramId> {
        // B4: compute a (mock) frame layout from the trace. Real drivers read
        // the trace's channel list + ring sizes; the mock sizes the frame from
        // the trace bytes so bind has concrete regions to allocate.
        let program = MockProgram {
            frame_size: trace.len().max(64),
            mirror_size: 64,
            word_size: 64,
            // Mock channel count: derive a small stable count from the trace so
            // bind allocates a plausible per-channel wake set. Real drivers read
            // the host-visible channel list from the trace's decls.
            n_channels: (trace.len() / 64).clamp(1, 8),
        };
        let mut st = self.inner.lock().unwrap();
        let id = st.next_program;
        st.next_program += 1;
        st.programs.push(program);
        Ok(id)
    }

    fn bind_instance(&self, program: ProgramId, _bindings: &[u8]) -> Result<BoundInstance> {
        let mut st = self.inner.lock().unwrap();
        let layout = *st
            .programs
            .get((program.wrapping_sub(1)) as usize)
            .ok_or_else(|| anyhow!("bind_instance: unknown program {program}"))?;
        // Size the pinned word region for the shared layout (pacing + head/tail
        // per channel); allocate the bind-fixed wake slots (B15).
        let word_bytes = layout.word_size.max(WordLayout::words(layout.n_channels) * 8);
        let wakes = WakeSlots {
            pacing: self.table.alloc(),
            channels: (0..layout.n_channels).map(|_| ChannelWakers::alloc(self.table)).collect(),
        };
        let instance = MockInstance {
            program,
            frame: vec![0u8; layout.frame_size].into_boxed_slice(),
            mirror: vec![0u8; layout.mirror_size].into_boxed_slice(),
            word: vec![0u8; word_bytes].into_boxed_slice(),
            wakes: wakes.clone(),
            pacing_word: Arc::new(AtomicU64::new(0)),
            fire_seq: 0,
        };
        let addresses = instance.addresses();
        let id = st.next_instance;
        st.next_instance += 1;
        st.instances.push(Some(instance));
        Ok(BoundInstance { id, addresses, wakes })
    }

    fn close_instance(&self, id: InstanceId) -> Result<()> {
        let mut st = self.inner.lock().unwrap();
        let slot = (id.wrapping_sub(1)) as usize;
        match st.instances.get_mut(slot) {
            Some(entry @ Some(_)) => {
                // Free the bind-fixed wake slots (B15: freed only at close).
                let inst = entry.take().expect("just matched Some");
                self.table.free(inst.wakes.pacing);
                for ch in &inst.wakes.channels {
                    ch.free(self.table);
                }
                Ok(())
            }
            _ => Err(anyhow!("close_instance: unknown or already-closed instance {id}")),
        }
    }

    fn enqueue(&self, batch: EnqueueBatch) -> Result<Completion> {
        let mut st = self.inner.lock().unwrap();
        let slot_idx = (batch.instance.wrapping_sub(1)) as usize;
        let Some(Some(inst)) = st.instances.get_mut(slot_idx) else {
            return Err(anyhow!("enqueue: unknown instance {}", batch.instance));
        };
        // Re-back on the instance's FIXED pacing slot + shared pacing word[0]
        // (boundary.md Phase 1): each fire targets the next monotonic sequence
        // number; the pacing word advances 1,2,3,… as fires commit. Steady state
        // allocates NO waker slot (the pacing slot is bind-fixed).
        inst.fire_seq += 1;
        let target = inst.fire_seq;
        let pacing = inst.wakes.pacing;
        let word = Arc::clone(&inst.pacing_word);
        st.pending.push(PendingBatch { instance: batch.instance, slot: pacing, word: Arc::clone(&word), target });
        Ok(Completion::parked_pacing(self.table, pacing, word, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::task::Wake;

    /// A waker that counts how many times it was woken — lets the round-trip
    /// test prove completion came through a wake (X0), not a queue hop.
    struct CountWaker(AtomicU32);
    impl Wake for CountWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn bind_returns_distinct_nonzero_frame_addresses() {
        let cp = MockControlPlane::new();
        let prog = cp.register_program(b"mock-trace-bytes").unwrap();
        let bound = cp.bind_instance(prog, b"seeds").unwrap();
        let a = bound.addresses;
        assert_ne!(a.frame_base, 0);
        assert_ne!(a.mirror_base, 0);
        assert_ne!(a.word_base, 0);
        // Three distinct regions (B5).
        assert_ne!(a.frame_base, a.mirror_base);
        assert_ne!(a.mirror_base, a.word_base);
        assert_ne!(a.frame_base, a.word_base);
        cp.close_instance(bound.id).unwrap();
        // Double close is a loud error, never a trap.
        assert!(cp.close_instance(bound.id).is_err());
    }

    #[test]
    fn unknown_program_and_instance_fail_loud() {
        let cp = MockControlPlane::new();
        assert!(cp.bind_instance(999, b"").is_err());
        assert!(cp
            .enqueue(EnqueueBatch { instance: 999, descriptor: vec![] })
            .is_err());
    }

    /// The X1 exit: register → bind → enqueue → completion round-trips on the
    /// mock with **zero queue hops** — the completion resolves through an X0
    /// wake of a genuinely parked future, never through `submit`/a response
    /// frame. Fully deterministic: park first, then the mock driver commits.
    #[test]
    fn register_bind_enqueue_completion_round_trip_no_queue() {
        let cp = MockControlPlane::new();

        let prog = cp.register_program(b"trace").unwrap();
        let bound = cp.bind_instance(prog, b"seeds").unwrap();
        let completion = cp
            .enqueue(EnqueueBatch { instance: bound.id, descriptor: vec![1, 2, 3] })
            .unwrap();
        assert_eq!(cp.in_flight(), 1);

        let cw = Arc::new(CountWaker(AtomicU32::new(0)));
        let waker = std::task::Waker::from(cw.clone());
        let mut cx = Context::from_waker(&waker);
        let mut fut = Box::pin(completion);

        // Not yet committed — the future genuinely parks.
        assert!(matches!(fut.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(cw.0.load(Ordering::SeqCst), 0);

        let woken_before = cp.table.metrics().woken;

        // The mock driver commits the batch: publish word, then wake (B11).
        assert!(cp.complete_next());
        assert_eq!(cp.in_flight(), 0);

        // The wake reached our parked waker (not a queue).
        assert_eq!(cw.0.load(Ordering::SeqCst), 1);
        assert_eq!(cp.table.metrics().woken, woken_before + 1);

        // Re-poll resolves.
        assert!(matches!(fut.as_mut().poll(&mut cx), Poll::Ready(())));
    }

    /// The register-then-recheck protocol also resolves when the commit races
    /// ahead of the first poll (no wake needed) — proves the fast path.
    #[test]
    fn completion_resolves_when_commit_precedes_poll() {
        let cp = MockControlPlane::new();
        let prog = cp.register_program(b"t").unwrap();
        let bound = cp.bind_instance(prog, b"").unwrap();
        let completion = cp
            .enqueue(EnqueueBatch { instance: bound.id, descriptor: vec![] })
            .unwrap();

        // Commit BEFORE the future is ever polled.
        assert!(cp.complete_next());

        let cw = Arc::new(CountWaker(AtomicU32::new(0)));
        let waker = std::task::Waker::from(cw);
        let mut cx = Context::from_waker(&waker);
        let mut fut = Box::pin(completion);
        // First poll sees the advanced word on the fast path.
        assert!(matches!(fut.as_mut().poll(&mut cx), Poll::Ready(())));
    }

    #[test]
    fn completions_are_ordered_fifo_across_instances() {
        let cp = MockControlPlane::new();
        let prog = cp.register_program(b"t").unwrap();
        let a = cp.bind_instance(prog, b"").unwrap();
        let b = cp.bind_instance(prog, b"").unwrap();
        let _ca = cp.enqueue(EnqueueBatch { instance: a.id, descriptor: vec![] }).unwrap();
        let _cb = cp.enqueue(EnqueueBatch { instance: b.id, descriptor: vec![] }).unwrap();
        assert_eq!(cp.in_flight(), 2);
        assert!(cp.complete_next());
        assert!(cp.complete_next());
        assert!(!cp.complete_next());
    }

    // ── Ported regression invariants (C1a): the "monotonic head" + "close-
    // during-in-flight" tests, moved off the deleted completion.rs
    // CompletionConsumer / carry_bridge InFlightTracker onto the carrier layer
    // (the re-backed Completion + the mock's shared pacing word). ──

    /// **Monotonic pacing (ported from completion.rs::monotonic_head_wakes_
    /// reader_every_fire).** Three successive fires on ONE instance re-back on
    /// the shared bind-fixed pacing slot + `word[0]`; each re-parks observing the
    /// prior committed count, and the NEXT monotonic advance (1→2→3) wakes it. The
    /// pacing slot is bind-fixed (not freed between fires), so run-ahead allocates
    /// nothing per fire.
    #[test]
    fn monotonic_pacing_word_wakes_each_fire() {
        let cp = MockControlPlane::new();
        let prog = cp.register_program(b"trace").unwrap();
        let bound = cp.bind_instance(prog, b"seeds").unwrap();
        for fire in 1..=3u64 {
            let completion = cp
                .enqueue(EnqueueBatch { instance: bound.id, descriptor: vec![] })
                .unwrap();
            let cw = Arc::new(CountWaker(AtomicU32::new(0)));
            let waker = std::task::Waker::from(cw.clone());
            let mut cx = Context::from_waker(&waker);
            let mut fut = Box::pin(completion);
            assert!(matches!(fut.as_mut().poll(&mut cx), Poll::Pending), "fire {fire} parks");
            assert!(cp.complete_next(), "fire {fire} commits");
            assert_eq!(cw.0.load(Ordering::SeqCst), 1, "fire {fire}: monotonic advance wakes");
            assert!(matches!(fut.as_mut().poll(&mut cx), Poll::Ready(())), "fire {fire} resolves");
        }
        cp.close_instance(bound.id).unwrap();
    }

    /// **Non-monotonic pacing stalls (ported from completion.rs::constant_head_
    /// stalls_second_fire).** Locks in WHY the committed word must advance: a
    /// waiter re-parked observing epoch 1 is NOT woken by a store of 1 again
    /// (`wake_past` epoch-Filters it) — the latent multi-fire stall. Only a
    /// monotonic advance to 2 wakes it.
    #[test]
    fn non_monotonic_pacing_filters_the_second_fire() {
        let t = crate::driver::waker::WakerTable::global();
        let slot = t.alloc();
        let word = Arc::new(AtomicU64::new(0));
        // Fire 1: waiter observed 0; word 0→1 wakes it.
        let (f1, w1) = {
            let cw = Arc::new(CountWaker(AtomicU32::new(0)));
            (cw.clone(), std::task::Waker::from(cw))
        };
        assert!(t.register(slot, &w1, 0));
        word.store(1, Ordering::Release);
        t.wake_past(slot, word.load(Ordering::Acquire));
        assert_eq!(f1.0.load(Ordering::SeqCst), 1, "fire 1 wakes");
        // Fire 2: waiter re-parks observing 1; a CONSTANT store of 1 is Filtered.
        let (f2, w2) = {
            let cw = Arc::new(CountWaker(AtomicU32::new(0)));
            (cw.clone(), std::task::Waker::from(cw))
        };
        assert!(t.register(slot, &w2, 1));
        word.store(1, Ordering::Release); // BUG: not monotonic
        t.wake_past(slot, word.load(Ordering::Acquire));
        assert_eq!(f2.0.load(Ordering::SeqCst), 0, "constant word → fire 2 stalls");
        // A monotonic advance to 2 finally wakes it.
        word.store(2, Ordering::Release);
        t.wake_past(slot, word.load(Ordering::Acquire));
        assert_eq!(f2.0.load(Ordering::SeqCst), 1, "monotonic advance wakes fire 2");
        t.free(slot);
    }

    /// **Close-during-in-flight is safe (ported from carry_bridge
    /// InFlightTracker close-gate).** Dropping the host future while its fire is
    /// still in flight cancels the wait (the bind-fixed pacing slot is NOT
    /// freed — owns_slot=false); the driver's later commit is a harmless no-op
    /// wake (nothing parked), and close then reclaims the instance + frees its
    /// bind-fixed wake slots without a trap.
    #[test]
    fn close_during_in_flight_is_safe() {
        let cp = MockControlPlane::new();
        let prog = cp.register_program(b"trace").unwrap();
        let bound = cp.bind_instance(prog, b"seeds").unwrap();
        let completion = cp
            .enqueue(EnqueueBatch { instance: bound.id, descriptor: vec![] })
            .unwrap();
        assert_eq!(cp.in_flight(), 1);
        // Cancel the wait mid-flight.
        drop(completion);
        // The driver still commits the now-unawaited batch — a no-op wake.
        assert!(cp.complete_next());
        assert_eq!(cp.in_flight(), 0);
        // Close reclaims the instance + frees the bind-fixed wake slots safely.
        cp.close_instance(bound.id).unwrap();
        assert!(cp.close_instance(bound.id).is_err(), "double close is a loud error");
    }

}
