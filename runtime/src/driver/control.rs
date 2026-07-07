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

use crate::driver::waker::{WakerSlotId, WakerTable};

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
/// fixed [`FrameAddresses`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundInstance {
    pub id: InstanceId,
    pub addresses: FrameAddresses,
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
    /// completion shape rather than duplicating the park.
    #[cfg_attr(not(feature = "driver-cuda"), allow(dead_code))]
    pub(crate) fn parked(
        table: &'static WakerTable,
        slot: WakerSlotId,
        word: Arc<AtomicU64>,
        target: u64,
    ) -> Completion {
        Completion { table, slot, word, target }
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
        // Recycle the per-batch waker slot. Any residual driver wake targeting
        // the old id is a harmless generation no-op (X0 B10).
        self.table.free(self.slot);
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
}

/// One bound instance's host-backed frame regions (B6 — stable addresses).
/// Boxed slices never reallocate, so the addresses handed out at bind stay
/// valid for the instance's lifetime.
struct MockInstance {
    #[allow(dead_code)]
    program: ProgramId,
    frame: Box<[u8]>,
    mirror: Box<[u8]>,
    word: Box<[u8]>,
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
/// [`MockControlPlane::complete_next`], which advances `word` and wakes `slot`.
struct PendingBatch {
    #[allow(dead_code)]
    instance: InstanceId,
    slot: WakerSlotId,
    word: Arc<AtomicU64>,
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
        let instance = MockInstance {
            program,
            frame: vec![0u8; layout.frame_size].into_boxed_slice(),
            mirror: vec![0u8; layout.mirror_size].into_boxed_slice(),
            word: vec![0u8; layout.word_size].into_boxed_slice(),
        };
        let addresses = instance.addresses();
        let id = st.next_instance;
        st.next_instance += 1;
        st.instances.push(Some(instance));
        Ok(BoundInstance { id, addresses })
    }

    fn close_instance(&self, id: InstanceId) -> Result<()> {
        let mut st = self.inner.lock().unwrap();
        let slot = (id.wrapping_sub(1)) as usize;
        match st.instances.get_mut(slot) {
            Some(entry @ Some(_)) => {
                *entry = None; // frees the host-backed frame regions
                Ok(())
            }
            _ => Err(anyhow!("close_instance: unknown or already-closed instance {id}")),
        }
    }

    fn enqueue(&self, batch: EnqueueBatch) -> Result<Completion> {
        let mut st = self.inner.lock().unwrap();
        let slot_idx = (batch.instance.wrapping_sub(1)) as usize;
        if !matches!(st.instances.get(slot_idx), Some(Some(_))) {
            return Err(anyhow!("enqueue: unknown instance {}", batch.instance));
        }
        // A fresh epoch-tagged waker slot per batch. The word starts at 0 and
        // the driver advances it to `target` (1) on commit; the host resolves
        // when it passes the target.
        let slot = self.table.alloc();
        let word = Arc::new(AtomicU64::new(0));
        let target = 1;
        st.pending.push(PendingBatch {
            instance: batch.instance,
            slot,
            word: Arc::clone(&word),
            target,
        });
        Ok(Completion { table: self.table, slot, word, target })
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
}
