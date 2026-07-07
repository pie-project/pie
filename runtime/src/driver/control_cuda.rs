//! # X2 — CUDA control plane (Runtime–Driver Boundary, the real-device dual of X1)
//!
//! X1 proved the `register_program → bind_instance → enqueue → completion` shape
//! on [`MockControlPlane`](super::control::MockControlPlane) with host allocations
//! standing in for the device frame and the pinned mirror/word regions. X2 backs
//! the same [`ControlPlane`] interface with the real thing — a device frame
//! (`cudaMalloc`), a pinned host mirror, and pinned ring-index words — plus the
//! **carrier**: the direct driver↔inferlet frame transport that D2H-mirrors
//! committed cells and publishes the ring word on a dedicated copy stream, then
//! resolves the parked host future through the X0 waker.
//!
//! ## Shape (builds on X1, no new trait)
//!
//! The device work lives behind the `pie_frame_*` `extern "C"` surface in the CUDA
//! driver ([`frame_carrier.cpp`]), declared here exactly like the `pie_pinned_*` /
//! `pie_device_*` tensor-I/O fast path — a direct call into `pie_driver_cuda_lib`
//! that bypasses the IPC/driver channel (B1/B2). [`bind_instance`] returns B5's
//! [`FrameAddresses`] straight from the device/pinned allocations; [`enqueue`]
//! hands the carrier an X0 waker slot + word and the [`cuda_carry_done`]
//! trampoline, and returns the X1 [`Completion`] parked on them. When the carrier's
//! copy-stream host callback fires (mirror landed, ring word published), the
//! trampoline advances the word and wakes the slot — the X3-shaped completion
//! handler, previewed here.
//!
//! [`frame_carrier.cpp`]: driver/cuda/src/sampling_ir/frame_carrier.cpp
//!
//! ## PROVISIONAL (velocity shift)
//!
//! The exact per-channel frame layout — which cells commit, at what offsets, which
//! ring word a fire advances — is guru's to reconcile as he steers the boundary
//! rework. This module is built to the interface: [`enqueue`] mirrors the whole
//! committed frame (`n_bytes = 0`) and advances ring word 0, the layout-agnostic
//! default the carrier resolves against the instance's real sizes. The layout swap
//! lands WITHOUT touching this control plane. CUDA validation is deferred to the
//! later batch (per directive); this lands the mechanism.
//!
//! Gated on `all(ptir, driver-cuda)`: off either flag the `pie_frame_*` symbols
//! are never referenced (no undefined-symbol link error) and only the mock exists.

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Result};
use pie_driver_abi::ForwardRequest;

use super::carry_bridge::{push_carry_request, CloseAction, InFlightTracker};
use super::completion::{ChannelScan, CompletionConsumer, PinnedRingWord};
use super::control::{
    BoundInstance, Completion, ControlPlane, EnqueueBatch, FrameAddresses, InstanceId, ProgramId,
};
use crate::driver::waker::{ChannelWakers, WakerSlotId, WakerTable};

unsafe extern "C" {
    /// B4 — register a trace with the CUDA driver; returns a 1-based program
    /// handle (`0` = rejected). Implemented in `frame_carrier.cpp`.
    fn pie_frame_register(trace: *const u8, trace_len: usize) -> u64;
    /// B4/B5 — bind an instance; writes the device frame + pinned mirror + pinned
    /// word bases. Returns the 1-based instance id (`0` = unknown program).
    fn pie_frame_bind(
        program: u64,
        out_frame_base: *mut u64,
        out_mirror_base: *mut u64,
        out_word_base: *mut u64,
    ) -> u64;
    /// B6 — release an instance's frame/mirror/word regions (fail-loud on unknown).
    fn pie_frame_close(instance: u64);
    /// The carrier — commit a fire: D2H the committed frame cells into the pinned
    /// mirror, publish `word[word_index] = target`, then run `done(user_data)` on a
    /// driver-internal thread once it lands (the X0 wake). `n_bytes == 0` mirrors
    /// the whole committed frame. The callback must not call CUDA APIs.
    fn pie_frame_carry(
        instance: u64,
        frame_offset: usize,
        mirror_offset: usize,
        n_bytes: usize,
        word_index: usize,
        target: u64,
        forward_evt: *mut c_void,
        done: extern "C" fn(*mut c_void),
        user_data: *mut c_void,
    );
    /// Register the stable completion callback ONCE (guru's register-once
    /// refinement). After this, an executor `pie_frame_carry` with a null `done`
    /// fires the registered callback — so the (a) BRIDGE carry-descriptor threads
    /// only `{user_data, word_index}` per request, never the fn-ptr.
    fn pie_frame_set_carry_done(done: extern "C" fn(*mut c_void));
}

/// Register `cuda_carry_done` with the carrier exactly once (idempotent). Called
/// lazily on the first control-plane op so the executor's null-`done` carry calls
/// resolve to it (the (a) BRIDGE path); the pre-bridge runtime also passes it
/// directly, so this is belt-and-suspenders for the executor cut.
fn register_carry_done_once() {
    use std::sync::Once;
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `cuda_carry_done` is a stable static extern fn; registered once.
        unsafe { pie_frame_set_carry_done(cuda_carry_done) };
    });
}

/// The completion context handed across the FFI to the carrier's copy-stream host
/// callback. Boxed and leaked into `pie_frame_carry`; reclaimed by
/// [`cuda_carry_done`] when the carrier fires exactly once per batch.
struct CarryWake {
    table: &'static WakerTable,
    slot: WakerSlotId,
    word: Arc<AtomicU64>,
    target: u64,
    /// The instance this batch drives — carried so the completion callback can
    /// retire it on the in-flight close-gate ([`InFlightTracker::on_complete`]).
    instance: InstanceId,
    /// The instance's host-visible channels, **captured at enqueue** (runtime
    /// thread, where taking the registry lock is fine). Held as the shared
    /// [`Arc<[ChannelScan]>`] so the host-func scans them **lock-free** — the
    /// registry `Mutex` is NEVER taken inside the `cudaLaunchHostFunc` thread
    /// (guru's inline ruling / the host-func no-lock rule). `None` if the instance
    /// had no registration (unknown/closed) — the scan is then skipped.
    channels: Option<Arc<[ChannelScan]>>,
}

/// The single completion trigger — the "parallel consumers, one signal" contract
/// (alpha's X3 ↔ delta's X4). The carrier calls this from `cudaLaunchHostFunc` once
/// the D2H mirror + the pinned channel-word publishes have landed (stream-ordered,
/// Release, B11), and it fans out to BOTH consumers off that one wake, **taking no
/// lock and making no CUDA call** (the host-func discipline):
///   1. **delta's pacing** — advance the Rust-side per-batch word the
///      [`Completion`] polls, then wake the batch slot (the X0 park `EnqueueAhead`
///      awaits). This word is `CarryWake.word` (an `Arc`), *distinct* from the
///      pinned `word_base` channel words, so the two never alias.
///   2. **alpha's X3 output-drain** — [`CompletionConsumer::scan_channels`] over the
///      pre-captured [`Arc<[ChannelScan]>`] reads this instance's committed
///      head/tail from the pinned words (published Release before this callback)
///      and issues the epoch-filtered per-channel wakes. Lock-free: the channels
///      were captured at enqueue, so no registry `Mutex` is touched here.
///
/// Reconstitutes and drops the boxed [`CarryWake`], so the per-batch context frees
/// exactly once.
extern "C" fn cuda_carry_done(user: *mut c_void) {
    // SAFETY: `user` is the `Box<CarryWake>` leaked by `enqueue`, handed back
    // verbatim by the carrier and fired exactly once per batch.
    let ctx = unsafe { Box::from_raw(user as *mut CarryWake) };
    // (1) delta's per-batch pacing completion (publish-before-wake, B11).
    ctx.word.store(ctx.target, Ordering::Release);
    ctx.table.wake_past(ctx.slot, ctx.target);
    // (2) alpha's X3 per-channel output-drain — LOCK-FREE over the captured Arc.
    // The pinned channel words were published (Release, stream-ordered) before this
    // callback, so the scan's acquire loads observe the committed head/tail.
    if let Some(channels) = &ctx.channels {
        CompletionConsumer::global().scan_channels(channels);
    }
    // (3) retire this in-flight carry for the close-gate (B6/§5.2). CALLBACK-SAFE:
    // this only decrements + QUEUES any owed deferred frame-free — it never calls a
    // CUDA API here (no `pie_frame_close` in the host-func thread). The queued free
    // is performed off-thread by `reap_deferred_frees` at the next control-plane call.
    InFlightTracker::global().on_complete(ctx.instance);
}

/// Perform any frame region-frees deferred by the close-gate — off the host-func
/// thread (a runtime thread that may call CUDA). Drained at every control-plane
/// entry point so a close-during-in-flight's `pie_frame_close` lands promptly once
/// the last in-flight carry retires.
fn reap_deferred_frees() {
    // SAFETY: `pie_frame_close` runs on this runtime thread (not the CUDA host-func
    // callback); each id was a live instance whose carries have all retired.
    InFlightTracker::global().reap(|id| unsafe { pie_frame_close(id) });
}

/// **The CUDA control plane (X2).** The embedded-driver [`ControlPlane`] backed by
/// real device frames + pinned mirrors/words + the copy-stream carrier. Direct
/// calls only — like [`MockControlPlane`](super::control::MockControlPlane) it
/// holds no channel and no response slot; completion rides the X0 waker.
pub struct CudaControlPlane {
    table: &'static WakerTable,
    /// Per-instance **monotonic committed-head counter** — the source of the X3
    /// channel-head value published into the pinned word (`word_index = 0`) each
    /// commit. It MUST strictly increase per fire: alpha's X3 read side wakes the
    /// reader with the epoch-filtered `wake_past(reader, head)`, which fires only
    /// when `head` passes the reader's last-observed epoch. Publishing a per-batch
    /// constant (e.g. the pacing `target = 1`) would leave the head at 1 on every
    /// fire → after fire 1 the wake Filters → multi-fire decode output-drain stalls.
    /// PROVISIONAL: for the single provisional channel the committed head == the
    /// commit count; at the multi-channel reconcile the executor supplies the real
    /// per-channel net-put committed index instead.
    commit_heads: Mutex<HashMap<InstanceId, u64>>,
}

impl Default for CudaControlPlane {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaControlPlane {
    pub fn new() -> CudaControlPlane {
        CudaControlPlane {
            table: WakerTable::global(),
            commit_heads: Mutex::new(HashMap::new()),
        }
    }

    /// The process-wide control plane. The carry *populate* ([`populate_carry`]) is
    /// called from the fire-build in `inference.rs`, which holds no `ControlPlane`
    /// handle (the legacy `submit_async` path predates X1); a singleton gives it the
    /// carry infrastructure (the shared committed-head counter + the global waker /
    /// completion / in-flight singletons) without threading a plane through the loop.
    ///
    /// [`populate_carry`]: CudaControlPlane::populate_carry
    pub fn global() -> &'static CudaControlPlane {
        static GLOBAL: OnceLock<CudaControlPlane> = OnceLock::new();
        GLOBAL.get_or_init(CudaControlPlane::new)
    }

    /// Next monotonic committed-head value for `instance` (1, 2, 3, …). A brief
    /// lock (append-to-structure, never spanning a GPU wait — the B1 discipline);
    /// lazily seeds the counter so an enqueue that races bind is still monotonic.
    fn next_commit_head(&self, instance: InstanceId) -> u64 {
        let mut heads = self.commit_heads.lock().unwrap_or_else(|e| e.into_inner());
        let head = heads.entry(instance).or_insert(0);
        *head += 1;
        *head
    }

    /// Build the per-fire completion context + its parked [`Completion`], shared by
    /// the standalone [`enqueue`](ControlPlane::enqueue) path (which then calls the
    /// carrier directly) and the (a) BRIDGE [`populate_carry`](Self::populate_carry)
    /// path (which stashes the boxed context on the request wire for the executor to
    /// carry). Returns the leaked `CarryWake` as a raw `user_data` pointer (reclaimed
    /// exactly once by [`cuda_carry_done`] when the carry fires) + the `Completion`
    /// parked on the fresh per-batch pacing slot/word. Registers the carry as
    /// in-flight for the close-gate (B6/§5.2) — the frame region can't be freed until
    /// it retires.
    fn build_carry_wake(&self, instance: InstanceId) -> (*mut c_void, Completion) {
        // delta's per-batch PACING completion: a fresh epoch-tagged waker slot +
        // `Arc` word per batch (the X1 shape), resolving once at `pacing_target = 1`.
        let slot = self.table.alloc();
        let word = Arc::new(AtomicU64::new(0));
        let pacing_target = 1;
        // Capture the instance's channel scans NOW (runtime thread, lock OK) so the
        // host-func can scan them LOCK-FREE — the registry `Mutex` is never taken in
        // the `cudaLaunchHostFunc` thread (guru's no-lock host-func rule).
        let channels = CompletionConsumer::global().channels_for(instance);
        let ctx = Box::new(CarryWake {
            table: self.table,
            slot,
            word: Arc::clone(&word),
            target: pacing_target,
            instance,
            channels,
        });
        // Count this carry as in-flight for the close-gate (B6/§5.2).
        InFlightTracker::global().on_enqueue(instance);
        let user_data = Box::into_raw(ctx) as *mut c_void;
        (user_data, Completion::parked(self.table, slot, word, pacing_target))
    }

    /// (a) BRIDGE **populate** — the runtime→`ForwardRequest` carry stash, the fire-
    /// build dual of [`enqueue`](ControlPlane::enqueue). For each per-request bound
    /// `instance` in `instances` (a2 batches R requests, each its own instance), it
    /// builds the completion context via [`build_carry_wake`](Self::build_carry_wake)
    /// and stashes `{user_data, word_index, instance}` into the request's parallel
    /// carry SoA cols (+ the once-set ABI-version guard) via
    /// [`push_carry_request`]. The CUDA executor reads those cols at a2 fire-commit
    /// and calls `pie_frame_carry(instance, word_index, committed_head, sample_done,
    /// /*done=*/ null → the once-registered `cuda_carry_done`, user_data)` per
    /// request — the runtime no longer calls the carrier itself (guru's (a) ruling).
    ///
    /// Returns one parked [`Completion`] per request (R-aligned with the pushed
    /// cols). Empty `instances` ⇒ no cols pushed ⇒ the behavior-neutral dormant
    /// state (charlie's a2 loop skips on the empty guard). The committed-head *value*
    /// is the executor's (device net-put index), not the runtime counter, under the
    /// bridge — so this path does NOT consume [`next_commit_head`](Self::next_commit_head).
    ///
    /// PROVISIONAL single channel: `word_index = 2*c = 0` (the channel-0 head).
    pub fn populate_carry(
        &self,
        req: &mut ForwardRequest,
        instances: &[InstanceId],
    ) -> Vec<Completion> {
        reap_deferred_frees(); // drain any close-during-in-flight frees owed
        register_carry_done_once(); // ensure the null-done executor carries resolve
        let mut completions = Vec::with_capacity(instances.len());
        for &instance in instances {
            let (user_data, completion) = self.build_carry_wake(instance);
            // Provisional single channel: publish the committed head into ring word 0.
            push_carry_request(req, user_data as u64, /*word_index=*/ 0, instance);
            completions.push(completion);
        }
        completions
    }
}

impl ControlPlane for CudaControlPlane {
    fn register_program(&self, trace: &[u8]) -> Result<ProgramId> {
        // SAFETY: `trace` is a valid slice for the duration of the call; the driver
        // only reads it to compute the frame layout.
        let id = unsafe { pie_frame_register(trace.as_ptr(), trace.len()) };
        if id == 0 {
            return Err(anyhow!("register_program: CUDA driver rejected the trace"));
        }
        Ok(id)
    }

    fn bind_instance(&self, program: ProgramId, _bindings: &[u8]) -> Result<BoundInstance> {
        reap_deferred_frees(); // drain any close-during-in-flight frees owed
        register_carry_done_once(); // register cuda_carry_done for null-done executor carries
        let (mut frame_base, mut mirror_base, mut word_base) = (0u64, 0u64, 0u64);
        // SAFETY: the three out-params are valid, distinct locals; the driver writes
        // each base once and returns 0 (leaving them untouched) on unknown program.
        let id = unsafe {
            pie_frame_bind(program, &mut frame_base, &mut mirror_base, &mut word_base)
        };
        if id == 0 {
            return Err(anyhow!("bind_instance: unknown program {program}"));
        }
        // X3 registration (alpha's seam): register this instance's host-visible
        // channels so the completion trigger can wake per channel. PROVISIONAL —
        // one channel today (guru's frame-layout reconcile fixes the channel count
        // + which word_index is each channel's head/tail). The agreed layout: for
        // channel `c`, committed head at `word_index = 2*c` (wakes the reader),
        // tail at `2*c+1` (wakes the writer). The wakers MUST come from the
        // consumer's own table (else a scan's `wake_past` targets the wrong table).
        let consumer = CompletionConsumer::global();
        let channel = ChannelScan {
            wakers: ChannelWakers::alloc(consumer.table()),
            // SAFETY: `word_base` is the instance's live pinned word-region base
            // (fixed for its lifetime, B6); indices 0/1 are within the region.
            head: Box::new(unsafe { PinnedRingWord::from_word_base(word_base, 0) }),
            tail: Box::new(unsafe { PinnedRingWord::from_word_base(word_base, 1) }),
        };
        consumer.register_instance(id, vec![channel]);
        // Seed the per-instance monotonic committed-head counter (X3 head source).
        self.commit_heads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, 0);
        // Register the instance with the in-flight close-gate (B6/§5.2 grace).
        InFlightTracker::global().on_bind(id);
        Ok(BoundInstance {
            id,
            addresses: FrameAddresses { frame_base, mirror_base, word_base },
        })
    }

    fn close_instance(&self, id: InstanceId) -> Result<()> {
        reap_deferred_frees(); // drain any previously-owed frees first
        // X3 registration sweep first (B6/B12): re-poll + free the channel waiter
        // slots so a host future blocked on this instance resolves rather than
        // hangs, before the device frame regions go. (Any in-flight completion still
        // holds its own `Arc<[ChannelScan]>`, so a late host-func scan stays valid.)
        CompletionConsumer::global().close_instance(id);
        // Drop the per-instance committed-head counter (no unbounded growth).
        self.commit_heads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
        // Close-GATE (B6/§5.2 grace): NEVER free the frame region while a carry is
        // in flight (a pending carry still writes the frame/mirror/word). Free NOW
        // only if nothing is in flight; otherwise the free is deferred and lands
        // when the last in-flight carry retires (via `reap_deferred_frees`).
        match InFlightTracker::global().request_close(id) {
            CloseAction::FreeNow => {
                // SAFETY: `id` names an instance with no carry in flight; the driver
                // fails loud on unknown/closed.
                unsafe { pie_frame_close(id) };
            }
            CloseAction::Deferred => { /* freed on the completing carry's reap */ }
        }
        Ok(())
    }

    fn enqueue(&self, batch: EnqueueBatch) -> Result<Completion> {
        reap_deferred_frees(); // drain any close-during-in-flight frees owed
        // Build the per-batch completion context (fresh pacing slot/word + captured
        // channel scans) + its parked `Completion`, registering the carry in-flight
        // for the close-gate. Shared with the (a) BRIDGE `populate_carry` path — the
        // one difference is that this STANDALONE path calls the carrier directly
        // below (there is no executor to stash the descriptor for), whereas the
        // bridge path stashes `user_data` on the request wire for the executor.
        let (user_data, completion) = self.build_carry_wake(batch.instance);
        // alpha's X3 channel-head value: the PER-INSTANCE MONOTONIC committed index
        // (1, 2, 3, …), NOT the per-batch pacing target. The carrier release-stores
        // this into the pinned head word (`word_index = 0`) stream-ordered before the
        // wake; alpha's scan epoch-filters `wake_past(reader, head)` on it, so a
        // per-batch constant would stall every fire after the first. (Provisional
        // single channel: committed head == commit count; the multi-channel reconcile
        // swaps this for the executor's real net-put committed index.) Under the (a)
        // BRIDGE this VALUE is the executor's device net-put index instead — see
        // [`populate_carry`](CudaControlPlane::populate_carry).
        let head_value = self.next_commit_head(batch.instance);
        // (a) BRIDGE stash point: under the executor cut, the carry is stashed on the
        // request wire by `populate_carry` (parallel SoA cols) instead of called here;
        // the executor reads it at a2 fire-commit and calls `pie_frame_carry` with it
        // + the device `fire_done` + real `committed_head`. Pre-bridge (no executor),
        // the runtime calls the carrier directly below with the SAME `user_data`.
        // PROVISIONAL carrier args (guru reconciles offsets/word-index from the real
        // frame layout): mirror the whole committed frame (n_bytes=0) at offset 0 and
        // publish the head into ring word 0. `descriptor` is the opaque launch bytes
        // X1 carries — unused by the mock-shape carrier, kept for the reconcile.
        let _ = &batch.descriptor;
        // SAFETY: `user_data` is the leaked `CarryWake` handed back verbatim to
        // `cuda_carry_done`, which reclaims it exactly once on commit.
        unsafe {
            pie_frame_carry(
                batch.instance,
                /*frame_offset=*/ 0,
                /*mirror_offset=*/ 0,
                /*n_bytes=*/ 0,
                /*word_index=*/ 0,
                /*value=*/ head_value,
                // forward_evt: null until the executor cut records the forward-done
                // event to serialize the D2H behind (charlie's device-side seam).
                core::ptr::null_mut(),
                cuda_carry_done,
                user_data,
            );
        }
        Ok(completion)
    }
}
