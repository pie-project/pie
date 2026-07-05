//! Per-driver batch scheduler.
//!
//! Each `BatchScheduler` owns its own RPC client, scheduling policy,
//! and tokio task. It accepts pre-translated forward pass requests,
//! accumulates them into batches, and fires them greedily (one policy
//! under FCFS).
//!
//! ## Terminology (thrust-2 / PTIR, frozen in phase S0)
//!
//! The pipelined-execution vocabulary is fixed here so the scheduler,
//! driver, and SDK use one set of names (overview §3, §7.2):
//!
//! - **launch** — the non-blocking form of `execute()`: enqueue a forward
//!   pass and return an async handle, never blocking on the device. The
//!   old "response-synchronous fire" is what launch replaces.
//! - **late channel** — the direct host→driver path that carries a
//!   logit-side late input (a `tensor.write`) to the executor's cut point,
//!   bypassing the scheduler. Its readiness is a word the executor waits on
//!   (C2), never a scheduler round trip.
//! - **`output()` / `outputs()`** — the async tensor handle(s) a launched
//!   forward yields; a handle may feed a later forward on-device (a
//!   producer link) with no host round trip.
//! - **producer link** — the device-resident buffer a sampled token lives
//!   in until every consumer forward drains it; its lifetime owner is the
//!   scheduler (`next_input_free_links` accounting).
//! - **quorum** — the fire rule (F1–F6): fire a dense batch the moment every
//!   *counted* pipeline's next pass is structurally ready, one deep behind
//!   the batch in flight. Idle-escape and cold-hold are its other two
//!   clauses. There is **no** completion estimation, lead-time EWMA, or
//!   "parity phase" in the decision path (F6) — those RA terms are dropped;
//!   the run-ahead depth-1 enqueue is what a measured lead time only
//!   approximated.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::oneshot;

use crate::arena::PhysicalPageId;
use crate::driver::{self, DriverId, SchedulerLimits};
use crate::process::ProcessId;

use super::ForwardOutput;
use super::stats::{self, SchedulerStats, BatchExecutionTiming};
use super::batch::{
    self, BatchAccumulator, RequestCapacityUsage, prepare_pending_for_batch,
    prepare_pending_with_usage,
};
use super::response;

mod chunked;

use chunked::ChunkContinuation;

/// Which clause of the fire rule fired this round (or `Hold`) — the label the
/// wait-for-all quorum rule (`WaitAllPolicy`) stamps for the fire-domain probes
/// (overview §7.2 / thrust-2 F1–F6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FireClause {
    /// F1: every active pipeline is ready — enqueue the dense wave behind the
    /// in-flight batch. The steady-state trigger.
    Quorum,
    /// Depth-N submit-ahead (G3 bubble): fire the ready subset eagerly behind
    /// the in-flight batch to keep the driver ring fed (bubble → 0).
    SubmitAhead,
    /// F2: device idle with an empty queue — fire the ready subset now.
    IdleEscape,
    /// F3: nothing in flight and the cold-hold window elapsed — fire partial.
    ColdHold,
    /// Do not fire this round.
    Hold,
}

impl FireClause {
    #[inline]
    pub fn fires(self) -> bool {
        !matches!(self, FireClause::Hold)
    }
}

fn scheduler_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PIE_SCHED_TRACE").is_some())
}

/// Emit one `[pie-sched-trace] …` line. When `PIE_SCHED_TRACE_FILE=<path>` is set
/// the line is written to that FILE via direct file I/O; otherwise it goes to
/// stderr (`eprintln!`). The file sink exists so a captured-output test harness
/// (cargo's default `libtest` capture) can read the scheduler's trace reliably:
/// the scheduler runs on its own OS thread, and Rust's libtest inherits its
/// per-test output-capture sink into engine-spawned threads, so a plain
/// `eprintln!` here is swallowed into cargo's buffer (not the test's fd-2 dup2
/// capture) in the battery form — the grammar10 G3-capture heisenbug. A real file
/// write bypasses that entirely, so the trace is parseable in BOTH `--nocapture`
/// and the default captured form. Append-mode with per-line flush so a reader can
/// slice by byte offset per phase. Additive + env-gated: zero effect when unset.
fn sched_trace_write(line: &str) {
    use std::io::Write;
    static SINK: OnceLock<Option<std::sync::Mutex<std::fs::File>>> = OnceLock::new();
    let sink = SINK.get_or_init(|| {
        std::env::var_os("PIE_SCHED_TRACE_FILE").and_then(|p| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .ok()
                .map(std::sync::Mutex::new)
        })
    });
    match sink {
        Some(m) => {
            if let Ok(mut f) = m.lock() {
                let _ = writeln!(f, "{line}");
                let _ = f.flush();
            }
        }
        None => eprintln!("{line}"),
    }
}

/// Test-only deterministic batch-accumulation hold (µs). When set, after the
/// first request the scheduler blocks up to this long for more requests to
/// arrive before firing, so concurrent requests reliably co-batch into one fire
/// (a deterministic `forward_R >= 2` for the merged-path verify). Default unset
/// → today's fire-on-arrival, zero production impact. This is the test-lever
/// ancestor of #10's production accumulation-window admission policy.
fn scheduler_accum_hold_us() -> Option<u64> {
    static HOLD: OnceLock<Option<u64>> = OnceLock::new();
    *HOLD.get_or_init(|| {
        std::env::var("PIE_SCHED_ACCUM_HOLD_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&us| us > 0)
    })
}

fn sched_epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// The cap-respecting force-fire gate (guru #2 / bravo's depth-k carrier
/// invariant). A stashed `next_pending` (a prebuilt solo or a capacity split)
/// force-fires the current batch to make room for it — but ONLY while depth is
/// below the cap, so a force-fire can NEVER push more than `max_in_flight`
/// fires in flight. bravo's device-resident carrier runs at cap=k and the
/// driver's `pi.sampled` WAR-event ring is sized D−1; a force-fire past the cap
/// would race the shared sample buffer (silent corruption). At/over the cap the
/// caller falls through to `policy.decide` (which returns `Wait`); the Wait
/// branch is `next_pending`-guarded (drains `latency_rx` only, never
/// `recv(req_rx)`) so the stash survives until a completion frees a slot.
#[inline]
fn force_fire_ready(next_pending_present: bool, in_flight_count: usize, cap: usize) -> bool {
    next_pending_present && in_flight_count < cap
}

/// Apply one [`FireCompletion`] to the run loop's in-flight accounting + policy
/// (the single drain shared by all four `latency_rx` sites). Always pops the
/// in-flight FIFO (`in_flight_count`), so a completion — real or an
/// `accounting_only` [`FireCompletionGuard`] fallback — can never leave the
/// policy Waiting at a phantom cap (D1 async-fire liveness). Only a real
/// completion feeds the timing EWMAs (`on_submitted` / `on_complete`).
#[inline]
fn drain_completion(
    policy: &mut FirePolicy,
    c: &FireCompletion,
    in_flight_count: &mut usize,
    device_idle_since: &mut Option<u64>,
    in_flight_fires: &mut std::collections::HashMap<u64, Vec<ProcessId>>,
    released_fires: &mut std::collections::HashSet<u64>,
) {
    // Stop tracking this fire either way.
    in_flight_fires.remove(&c.fire_id);
    // BAR-2 per-lane in-flight release: a fire whose cap slot was RELEASED early
    // (stuck fire of a demoted lane, see `release_lane_in_flight`) already had
    // its `policy.in_flight` + `in_flight_count` decremented at release time.
    // Its (eventual) completion must be a NO-OP on those counters — otherwise
    // the cap under-reads → over-fires → D−1 WAR-ring overrun.
    if released_fires.remove(&c.fire_id) {
        return;
    }
    if !c.accounting_only {
        policy.on_submitted(c.submission_latency);
        policy.on_complete(c.forward_latency);
    }
    *in_flight_count = in_flight_count.saturating_sub(1);
    if *in_flight_count == 0 {
        *device_idle_since = Some(now_micros());
    }
}

/// BAR-2 cap-saturation fix: release the cap slots held by a demoted lane's
/// in-flight fires. A demoted lane is unresponsive (≥5 missed waves ≈ 50ms of
/// not submitting) and stuck in `drain_own_fires` — its in-flight fires are
/// genuinely stuck (never retiring), pinning `policy.in_flight` at the cap so
/// `decide_wave_at` never fires ANY wave (the whole fleet stalls, incl. this
/// lane's re-promotion). Free those slots so the cap drops below max and the
/// wave re-fires. The released fires are recorded in `released_fires` so their
/// later `FireCompletion` (if the driver ever un-sticks) is a no-op on the
/// counters — no double-decrement, cap invariant preserved (a genuinely-stuck
/// fire never produced a `pi.sampled` D2H, so it holds no WAR-ring slot).
fn release_lane_in_flight(
    pid: ProcessId,
    policy: &mut FirePolicy,
    in_flight_count: &mut usize,
    in_flight_fires: &std::collections::HashMap<u64, Vec<ProcessId>>,
    released_fires: &mut std::collections::HashSet<u64>,
) -> usize {
    let to_release: Vec<u64> = in_flight_fires
        .iter()
        .filter(|(fid, lanes)| lanes.contains(&pid) && !released_fires.contains(fid))
        .map(|(&fid, _)| fid)
        .collect();
    for fid in &to_release {
        // Release one cap slot per stuck fire (mirror `on_complete`'s in_flight--
        // without touching the timing EWMAs — a stuck fire has no real latency).
        policy.on_complete(Duration::ZERO);
        *in_flight_count = in_flight_count.saturating_sub(1);
        released_fires.insert(*fid);
    }
    to_release.len()
}


/// Where a request went when the shared admission cascade [`try_admit`] handled
/// it — the caller maps this to its own loop control flow (`continue`/`break`).
enum AdmitOutcome {
    /// Folded into the forming batch.
    Pushed,
    /// Held off-batch: the chain gate swallowed it, or it was parked in
    /// `dep_stash` for a later wave. The caller keeps draining.
    Consumed,
    /// Stashed in `next_pending` to force the current batch out first (a prebuilt
    /// solo fire or a capacity/run-ahead split).
    Deferred,
}

/// The shared per-request admission cascade run by every drain loop in [`run`]:
/// (1) chain-continuation gate, (2) prebuilt-solo defer, (3) run-ahead dep-stash
/// or defer, (4) capacity-exceed defer, else push. `usage` selects the
/// capacity-tracked push (`Some`, the accumulate path) over the plain push;
/// `allow_prebuilt_solo` gates the prebuilt check (off for the Wait-branch, which
/// historically skipped it). Consumes `pending`, routing it into the batch,
/// `dep_stash`, or `next_pending`.
fn try_admit(
    pending: PendingRequest,
    usage: Option<RequestCapacityUsage>,
    allow_prebuilt_solo: bool,
    batch: &mut BatchAccumulator,
    next_pending: &mut Option<PendingRequest>,
    policy: &mut FirePolicy,
    tombstones: &Tombstones,
    driver_idx: usize,
) -> AdmitOutcome {
    // G2 prebuilt-passthrough: a prebuilt beam fire is a complete pre-assembled
    // multi-lane batch — it never co-batches. Stash it to force a solo fire.
    if allow_prebuilt_solo && (pending.prebuilt || batch.has_prebuilt()) {
        *next_pending = Some(pending);
        return AdmitOutcome::Deferred;
    }
    let exceeds = match &usage {
        Some(u) => batch.would_exceed_with(u),
        None => batch.would_exceed(&pending),
    };
    if exceeds {
        if scheduler_trace_enabled() {
            let reason = batch
                .would_exceed_reason(&pending)
                .unwrap_or_else(|| "unknown".to_string());
            sched_trace_write(&format!(
                "[pie-sched-trace] driver={} stash current_requests={} current_tokens={} reason={}",
                driver_idx,
                batch.len(),
                batch.total_tokens(),
                reason,
            ));
        }
        *next_pending = Some(pending);
        return AdmitOutcome::Deferred;
    }
    policy.on_arrival(
        &pending.program_identity_hashes,
        tombstones.filter(pending.pipeline_id),
        Instant::now(),
    );
    match usage {
        Some(u) => batch.push_with(pending, u),
        None => batch.push(pending),
    }
    AdmitOutcome::Pushed
}

/// Apply one pipeline [`LifecycleEvent`] to the run loop's wait-set / tombstones
/// / stash. Shared by the top-of-loop drain AND the idle-select `lifecycle_rx`
/// arm (BAR 1: a `Join` must wake an idle-frozen loop — the select otherwise
/// watches only `req_rx`/`latency_rx`, so a resume while the fleet is frozen
/// would never be seen). `Terminate` drops + errors the stash; `Suspend` parks
/// it; `Join` decays the tombstone + un-parks it.
fn apply_lifecycle_event(
    ev: LifecycleEvent,
    policy: &mut FirePolicy,
    tombstones: &mut Tombstones,
    suspended: &mut std::collections::HashSet<ProcessId>,
) {
    match ev {
        LifecycleEvent::Leave(pid, LeaveKind::Terminate) => {
            policy.on_pipeline_leave(pid);
            tombstones.insert(pid);
            suspended.remove(&pid);
        }
        LifecycleEvent::Leave(pid, LeaveKind::Suspend) => {
            policy.on_pipeline_leave(pid);
            tombstones.insert(pid);
            suspended.insert(pid);
        }
        LifecycleEvent::Join(pid) => {
            tombstones.remove(pid);
            suspended.remove(&pid);
        }
    }
}

/// How a pipeline left the fleet (Phase-2 carrier × contention). Determines the
/// fate of its stashed run-ahead chain.
#[derive(Clone, Copy, Debug)]
pub(crate) enum LeaveKind {
    /// Dead pipeline (user cancel / exit / wait-for-all miss-limit terminate) —
    /// its passes will NEVER resume, so its `dep_stash` is dropped + errored.
    Terminate,
    /// Contention-preempt / demote: the pipeline is ALIVE and RESUMES its exact
    /// in-flight + stashed passes on restore. Its stash is SKIP-PROMOTED (parked
    /// in chain order), never errored — erroring would break resume transparency
    /// (C4). It re-promotes on the matching [`LifecycleEvent::Join`].
    Suspend,
}

/// A pipeline lifecycle event the scheduler run loop consumes on `lifecycle_rx`
/// (M-A1 Stage 2, extended for Phase-2). One enum closes both the Phase-1
/// tombstone-decay (via `Join`) and the Phase-2 stash fate (via `Leave` kind).
#[derive(Clone, Copy, Debug)]
pub(crate) enum LifecycleEvent {
    /// Left the wait-set — drop it from membership + tombstone it; the stash
    /// fate follows the [`LeaveKind`].
    Leave(ProcessId, LeaveKind),
    /// Rejoined (contention restore / post-suspend release) — DECAY the tombstone
    /// so its queued/stashed requests can re-join the wave, and un-park its stash.
    /// THIS is the Phase-1 sustained-hang fix: a stale tombstone otherwise kept a
    /// restored pipeline permanently untracked (`filter → None`), so it never
    /// held a wave and the fleet stalled.
    Join(ProcessId),
}

/// Registry of per-driver wait-for-all scheduler `lifecycle_tx` senders (M-A1
/// Stage 2). A pipeline lifecycle event ([`LifecycleEvent`]) broadcasts to every
/// waitall run loop; each loop applies it to its wait-set / tombstones / stash.
/// Only waitall run loops register — legacy/quorum schedulers never touch this.
static LIFECYCLE_SENDERS: OnceLock<
    std::sync::Mutex<Vec<crossbeam::channel::Sender<LifecycleEvent>>>,
> = OnceLock::new();

fn lifecycle_senders(
) -> &'static std::sync::Mutex<Vec<crossbeam::channel::Sender<LifecycleEvent>>> {
    LIFECYCLE_SENDERS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

fn broadcast_lifecycle(ev: LifecycleEvent) {
    if let Some(lock) = LIFECYCLE_SENDERS.get() {
        if let Ok(mut senders) = lock.lock() {
            senders.retain(|tx| tx.send(ev).is_ok());
        }
    }
}

/// Broadcast a pipeline `Leave` to every waitall run loop. `Terminate` (dead) →
/// drop + error its stash; `Suspend` (contention-preempt, resumes) → park its
/// stash. Called by `process::terminate` (Terminate) + `Orchestrator::
/// report_suspended` (Suspend). No-op when no waitall scheduler is registered.
pub(crate) fn notify_pipeline_leave(pid: ProcessId, kind: LeaveKind) {
    broadcast_lifecycle(LifecycleEvent::Leave(pid, kind));
}

/// Broadcast a pipeline `Join` (contention restore / suspend release) — the run
/// loop decays its tombstone + un-parks its stash. Called by the orchestrator's
/// restore/release points (guru's Phase-2 half wires the emissions).
#[allow(dead_code)] // wired by the orchestrator restore path (guru, Phase-2 compose)
pub(crate) fn notify_pipeline_join(pid: ProcessId) {
    broadcast_lifecycle(LifecycleEvent::Join(pid));
}

/// Bounded set of recently-departed pipeline ids (M-A1 Stage 2). Prevents a
/// STALE queued request (a pipeline submitted just before it terminated) from
/// implicitly re-joining the wait-set via `on_pipeline_request` — a terminated
/// pid would otherwise become a phantom that holds every wave until the
/// miss-counter re-terminates it, on a loop. FIFO-evicted at `cap` (a departed
/// pid's stale requests all drain well within this window).
struct Tombstones {
    set: std::collections::HashSet<ProcessId>,
    order: std::collections::VecDeque<ProcessId>,
    cap: usize,
}

impl Tombstones {
    fn new(cap: usize) -> Self {
        Self {
            set: std::collections::HashSet::new(),
            order: std::collections::VecDeque::new(),
            cap,
        }
    }

    fn insert(&mut self, pid: ProcessId) {
        if self.set.insert(pid) {
            self.order.push_back(pid);
            if self.order.len() > self.cap {
                if let Some(old) = self.order.pop_front() {
                    self.set.remove(&old);
                }
            }
        }
    }

    /// Map a request's pipeline id through the tombstone: a departed pid rejoins
    /// as UNTRACKED (`None`) — the request still rides the wave, but can't
    /// resurrect the dead pipeline into the wait-set.
    fn filter(&self, pid: Option<ProcessId>) -> Option<ProcessId> {
        match pid {
            Some(p) if self.set.contains(&p) => None,
            other => other,
        }
    }

    /// Decay a tombstone on `Join` (contention restore): the pipeline is back, so
    /// its requests must be trackable again (`filter` stops mapping it to
    /// `None`). The Phase-1 sustained-hang fix — a stale tombstone otherwise kept
    /// a restored pipeline permanently untracked, so it never held a wave.
    fn remove(&mut self, pid: ProcessId) {
        if self.set.remove(&pid) {
            self.order.retain(|&p| p != pid);
        }
    }
}

pub(crate) fn now_micros() -> u64 {
    sched_epoch().elapsed().as_micros() as u64
}

// =============================================================================
// FirePolicy — the run loop's single fire rule (WaitAllPolicy)
// =============================================================================

/// The run loop's fire outcome. Generalizes [`Decision`] with the wait-for-all
/// `missing` list — the active pipelines that missed the wave's straggler
/// deadline. Informational at the fire site (the wave fires the ready batch
/// either way); the miss accounting + demotion already happened in
/// `WaitAllPolicy::decide_wave_at`.
enum FireOutcome {
    Fire { missing: Vec<ProcessId> },
    Wait(Duration),
}

/// The run loop's fire interface — the single [`WaitAllPolicy`] wave rule. Kept
/// as a thin wrapper so the run loop's call sites stay stable; a follow-up can
/// inline it now that only one rule remains.
enum FirePolicy {
    WaitAll(super::policy::WaitAllPolicy),
}

impl FirePolicy {
    fn on_submitted(&mut self, latency: Duration) {
        match self {
            FirePolicy::WaitAll(w) => w.on_submitted(latency),
        }
    }

    fn on_complete(&mut self, latency: Duration) {
        match self {
            FirePolicy::WaitAll(w) => w.on_complete(latency),
        }
    }

    fn on_fired(&mut self, fired_size: usize) {
        match self {
            FirePolicy::WaitAll(w) => w.on_fired(fired_size),
        }
    }

    /// A request entered the current wave. WaitAll marks the pipeline ready
    /// (wave membership) and arms the straggler clock — `None` rides as
    /// untracked (prebuilt/beam). Identity hashes are unused (single rule).
    fn on_arrival(
        &mut self,
        _program_identity_hashes: &[u64],
        pipeline_id: Option<ProcessId>,
        now: Instant,
    ) {
        match self {
            FirePolicy::WaitAll(w) => w.on_pipeline_request(pipeline_id, now),
        }
    }

    fn decide(&mut self, current_batch_size: usize, now: Instant) -> FireOutcome {
        match self {
            FirePolicy::WaitAll(w) => match w.decide_wave_at(current_batch_size, now) {
                super::policy::WaveDecision::Fire { missing } => {
                    FireOutcome::Fire { missing }
                }
                super::policy::WaveDecision::Wait(d) => FireOutcome::Wait(d),
            },
        }
    }

    fn distinct_program_count(&self) -> usize {
        match self {
            FirePolicy::WaitAll(w) => w.active_pipelines(),
        }
    }

    /// Pipelines demoted at the consecutive-miss limit — the run loop
    /// `process::terminate`s them (M-A1 liveness).
    fn take_terminate_candidates(&mut self) -> Vec<ProcessId> {
        match self {
            FirePolicy::WaitAll(w) => w.take_terminate_candidates(),
        }
    }

    /// A pipeline left the fleet — drop it from the wait-set.
    #[allow(dead_code)]
    fn on_pipeline_leave(&mut self, pipeline_id: ProcessId) {
        if let FirePolicy::WaitAll(w) = self {
            w.on_pipeline_leave(pipeline_id);
        }
    }
}



/// Completion feedback the spawned fire task sends back to the scheduler loop
/// so the run-ahead policy can update its timing EWMAs. `forward_latency` (the
/// off-thread GPU/driver wait) feeds `on_complete` and pops the in-flight FIFO;
/// `submission_latency` (the host batch-build/enqueue done on the scheduler
/// thread) feeds `on_submitted` (the lead time the fire is brought forward by).
///
/// `accounting_only` marks a fallback completion emitted by [`FireCompletionGuard`]
/// when the fire task unwound before finishing (panic in `handle.wait()` /
/// `dispatch_fired_batch`, or an early exit). The run loop still pops the
/// in-flight FIFO (so `in_flight_count` drains and the policy never Waits at a
/// phantom cap — the D1 async-fire liveness guarantee) but SKIPS the EWMA update
/// so a zero-latency fallback can't pollute the policy's timing model.
struct FireCompletion {
    forward_latency: Duration,
    submission_latency: Duration,
    accounting_only: bool,
    /// The unique id of the fire this completion belongs to (BAR-2 per-lane
    /// in-flight release). The run loop matches it against `released_fires` — a
    /// fire whose cap slot was RELEASED early (a stuck fire of a demoted lane)
    /// must NOT double-decrement `policy.in_flight`/`in_flight_count` here (else
    /// the cap under-reads → over-fires → D−1 WAR-ring overrun).
    fire_id: u64,
}

/// RAII guard that guarantees the run loop's in-flight accounting drains even if
/// the spawned fire task unwinds. Under async fire (D1) a fire task that panics
/// in `handle.wait()` / `dispatch_fired_batch`, or exits early, would otherwise
/// never send its [`FireCompletion`] → `in_flight_count` leaks → the policy Waits
/// at a phantom cap FOREVER (the cap≥4 hang class). Drop runs on the normal path
/// AND on panic unwind, so the completion is sent exactly once from a single site
/// (Drop). The success path calls [`complete`](Self::complete) to swap in the
/// measured `forward_latency`; a guard dropped without `complete` sends an
/// `accounting_only` fallback (pops the FIFO, no EWMA pollution).
struct FireCompletionGuard {
    tx: crossbeam::channel::Sender<FireCompletion>,
    submission_latency: Duration,
    forward_latency: Option<Duration>,
    fire_id: u64,
}

impl FireCompletionGuard {
    fn new(
        tx: crossbeam::channel::Sender<FireCompletion>,
        submission_latency: Duration,
        fire_id: u64,
    ) -> Self {
        Self {
            tx,
            submission_latency,
            forward_latency: None,
            fire_id,
        }
    }

    /// Record the measured GPU/driver latency for the success path. Drop then
    /// sends the real (non-`accounting_only`) completion.
    fn complete(&mut self, forward_latency: Duration) {
        self.forward_latency = Some(forward_latency);
    }
}

impl Drop for FireCompletionGuard {
    fn drop(&mut self) {
        let (forward_latency, accounting_only) = match self.forward_latency {
            Some(f) => (f, false),
            None => (Duration::ZERO, true),
        };
        let _ = self.tx.send(FireCompletion {
            forward_latency,
            submission_latency: self.submission_latency,
            accounting_only,
            fire_id: self.fire_id,
        });
    }
}

// =============================================================================
// PendingRequest
// =============================================================================

/// A forward pass request bundled with its response channel and physical pages.
pub(crate) struct PendingRequest {
    pub(crate) request: pie_driver_abi::ForwardRequest,
    pub(crate) completion: Completion,
    pub(crate) physical_page_ids: Vec<PhysicalPageId>,
    pub(crate) last_page_len: u32,
    /// #10: per-program `program_identity_hash` for this request (one per program
    /// in its sampler pass; empty for plain decode). Computed once at attach
    /// (host-side, before carrier encoding) and threaded to the policy's
    /// distinct-program set via `on_arrival` — runtime-side only, never on the
    /// wire `ForwardRequest`.
    pub(crate) program_identity_hashes: Vec<u64>,
    /// M-A1 (wait-for-all rebuild): the submitting pipeline's `ProcessId` — the
    /// wave-barrier membership key (fire when every active pipeline has submitted
    /// its N+1). `None` for solo/prebuilt fires (beam passthrough), which bypass
    /// the wave. Threaded from `execute_impl(state.id())` at submit; consumed by
    /// the QuorumModel policy core (M-A2, guru). Runtime-side only, never on the
    /// wire `ForwardRequest`.
    #[allow(dead_code)]
    pub(crate) pipeline_id: Option<ProcessId>,
    /// R-decomposition probe (charlie): the `now_micros()` stamp taken at
    /// `submit_async` (the guest's resubmit instant), in the scheduler epoch so
    /// it's comparable to `last_dispatch_end_micros`. Splits the round-trip R
    /// into `guest_roundtrip_us` (dispatch→submit = guest wake + wasm + rebuild)
    /// and `service_queue_us` (submit→recv = the SERVICE actor hop). `0` for
    /// non-`submit_async` paths (prebuilt/beam, plain submit, chunked/dispatch
    /// re-submits) — skipped in the accumulation.
    pub(crate) submitted_at_us: u64,
    /// G2 prebuilt-passthrough: this request carries a COMPLETE, wire-final
    /// multi-lane `ForwardRequest` (a PTIR beam fire = B forward lanes, one
    /// program/epilogue) that must fire VERBATIM. The scheduler then (1) fires it
    /// SOLO (never co-batched), (2) skips `build_batch_request` /
    /// `append_request_with_options` (which would re-fold the B-lane geometry
    /// from a single `physical_page_ids`), and (3) routes the WHOLE rich response
    /// to the single completion (bypassing the per-row `num_requests ==
    /// requests.len()` split). `physical_page_ids` is then the union of all
    /// lanes' pages, carried for KV-txn / ref tracking only.
    pub(crate) prebuilt: bool,
}

pub(crate) enum Completion {
    Direct(oneshot::Sender<Result<ForwardOutput>>),
    Chunk {
        continuation: ChunkContinuation,
        sampler_slots: Vec<usize>,
    },
}

impl PendingRequest {
    fn direct(
        request: pie_driver_abi::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
        pipeline_id: Option<ProcessId>,
        submitted_at_us: u64,
    ) -> Self {
        Self {
            request,
            completion: Completion::Direct(response_tx),
            physical_page_ids,
            last_page_len,
            program_identity_hashes,
            pipeline_id,
            submitted_at_us,
            prebuilt: false,
        }
    }

    /// G2 prebuilt-passthrough constructor — the request fires VERBATIM + SOLO;
    /// see [`PendingRequest::prebuilt`].
    fn direct_prebuilt(
        request: pie_driver_abi::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
    ) -> Self {
        Self {
            request,
            completion: Completion::Direct(response_tx),
            physical_page_ids,
            last_page_len,
            program_identity_hashes,
            pipeline_id: None,
            submitted_at_us: 0,
            prebuilt: true,
        }
    }
}

// =============================================================================
// SchedulerHandle
// =============================================================================

/// Cloneable submit handle.
///
/// Backed by a sync crossbeam_channel rather than tokio mpsc so the
/// receiving main loop (sync OS thread) can recv with futex-level
/// wake latency (~5-15 µs) instead of tokio's task-wake roundtrip
/// (~100-200 µs).
#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    tx: crossbeam::channel::Sender<PendingRequest>,
}

impl SchedulerHandle {
    pub fn submit(
        &self,
        request: pie_driver_abi::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Result<()> {
        self.submit_with_identity(
            request,
            response_tx,
            physical_page_ids,
            last_page_len,
            Vec::new(),
            None,
            0,
        )
    }

    /// Submit carrying the request's per-program `program_identity_hash`es (the
    /// #10 distinct-count key, computed host-side at attach). Empty ⇒ plain
    /// decode. The hashes ride on `PendingRequest` (runtime-side only) and reach
    /// the policy via `on_arrival`; they are never placed on the wire request.
    pub fn submit_with_identity(
        &self,
        request: pie_driver_abi::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
        pipeline_id: Option<ProcessId>,
        submitted_at_us: u64,
    ) -> Result<()> {
        self.tx
            .send(PendingRequest::direct(
                request,
                response_tx,
                physical_page_ids,
                last_page_len,
                program_identity_hashes,
                pipeline_id,
                submitted_at_us,
            ))
            .map_err(|_| anyhow::anyhow!("scheduler channel closed"))?;
        Ok(())
    }

    /// G2 prebuilt-passthrough submit — the request is a COMPLETE, wire-final
    /// multi-lane `ForwardRequest` (a PTIR beam fire) that fires VERBATIM + SOLO
    /// (bypassing the per-request re-fold). `physical_page_ids` is the union of
    /// all lanes' pages, for KV-txn / ref tracking only. See
    /// [`PendingRequest::prebuilt`].
    pub fn submit_prebuilt(
        &self,
        request: pie_driver_abi::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
    ) -> Result<()> {
        self.tx
            .send(PendingRequest::direct_prebuilt(
                request,
                response_tx,
                physical_page_ids,
                last_page_len,
                program_identity_hashes,
            ))
            .map_err(|_| anyhow::anyhow!("scheduler channel closed"))?;
        Ok(())
    }
}

/// Per-driver mutable state for the [`BatchScheduler::run`] loop, grouped so the
/// loop's phases (drain / accumulate / fire / wait) can share it by `&mut self`
/// instead of threading a dozen locals. One instance per `run` call, never
/// shared across threads (the fire task captures clones of what it needs).
struct LoopState {
    /// The forming batch: `PendingRequest`s folded under the `SchedulerLimits`
    /// caps until the policy fires it.
    batch: BatchAccumulator,
    /// The single wait-for-all quorum fire rule (WaitAllPolicy, overview §7.2 /
    /// thrust-2 F1-F6): waits until every active pipeline's next pass is ready,
    /// then enqueues the dense wave behind the in-flight batch; stragglers fire
    /// on the wave deadline and demote at the miss limit.
    policy: FirePolicy,
    /// A request held back for the NEXT batch (a prebuilt-solo fire or a
    /// capacity/run-ahead split) - forces the current batch out first.
    next_pending: Option<PendingRequest>,
    /// Pipelines removed from the wait-set (terminate/cancel/exit/preempt); a
    /// stale queued request from a tombstoned pid must not re-join the wave.
    tombstones: Tombstones,
    /// Pipelines SUSPENDED by contention (`LeaveKind::Suspend`) whose dep-stash
    /// chain is PARKED - skip-promoted (not fired, not errored) until they rejoin
    /// (`LifecycleEvent::Join`), then resumed in chain order.
    suspended: std::collections::HashSet<ProcessId>,
    /// Batches currently outstanding on the driver (bounds the run-ahead depth).
    in_flight_count: usize,
    /// Host-side device-idle bubble proxy: the instant the device *appeared*
    /// idle (a completion observed with nothing queued behind it). The next
    /// fire's enqueue records `inter_batch_bubble_us = launch - idle_since`.
    device_idle_since: Option<u64>,
    /// Monotonic id stamped on each fire for BAR-2 per-lane in-flight release.
    next_fire_id: u64,
    /// BAR-2: each in-flight fire's id -> its tracked lane pids, so a demoted
    /// lane's stuck fires can be found + their cap slots released.
    in_flight_fires: std::collections::HashMap<u64, Vec<ProcessId>>,
    /// Fires whose cap slot was RELEASED early (a demoted lane's stuck fire); a
    /// released fire's later `FireCompletion` is a no-op on the cap counters (no
    /// double-decrement -> no WAR over-fire).
    released_fires: std::collections::HashSet<u64>,
}

impl LoopState {
    fn new(limits: SchedulerLimits, page_size: u32, stats: &Arc<SchedulerStats>) -> Self {
        Self {
            batch: BatchAccumulator::new(limits, page_size),
            policy: FirePolicy::WaitAll(super::policy::WaitAllPolicy::new(
                limits.max_forward_requests,
                Some(stats.clone()),
            )),
            next_pending: None,
            tombstones: Tombstones::new(4096),
            suspended: std::collections::HashSet::new(),
            in_flight_count: 0,
            device_idle_since: None,
            next_fire_id: 0,
            in_flight_fires: std::collections::HashMap::new(),
            released_fires: std::collections::HashSet::new(),
        }
    }

    /// Fire the assembled batch: final coalescing drain, per-identity + wave
    /// telemetry, then build + enqueue + spawn the off-thread dispatch. In-flight
    /// bookkeeping (cap slot, fire id, bubble proxy) is updated on the scheduler
    /// thread before the fire task is spawned.
    fn fire_batch(
        &mut self,
        missing: Vec<ProcessId>,
        req_rx: &crossbeam::channel::Receiver<PendingRequest>,
        submit_tx: &crossbeam::channel::Sender<PendingRequest>,
        latency_tx: &crossbeam::channel::Sender<FireCompletion>,
        rt_handle: &tokio::runtime::Handle,
        stats: &Arc<SchedulerStats>,
        driver_id: DriverId,
        driver_idx: usize,
        page_size: u32,
        waitall_active: bool,
    ) {
        // No in-flight gate to acquire: the scheduler runs
        // execute_batch synchronously, so we can only reach
        // here when the previous fire has fully completed.

        // Do one last non-blocking drain so requests that
        // arrived between the recv loop and here are
        // coalesced into this batch instead of being
        // stranded behind it.
        let fire_prepare_start = Instant::now();
        while self.next_pending.is_none() && !self.batch.is_full() {
            let Ok(pending) = req_rx.try_recv() else {
                break;
            };
            if let Some(msg) = self.batch.single_request_limit_error(&pending) {
                pending.send_error(msg);
                continue;
            }
            match try_admit(
                pending,
                None,
                true,
                &mut self.batch,
                &mut self.next_pending,
                &mut self.policy,
                &self.tombstones,
                driver_idx,
            ) {
                AdmitOutcome::Consumed => continue,
                AdmitOutcome::Deferred => break,
                AdmitOutcome::Pushed => {}
            }
        }

        let total_tokens = self.batch.total_tokens();
        // Wait-for-all wave gauge (M-AB): sample the wait-set size +
        // stragglers at each WaitAll fire so `avg_active`/`avg_missing`
        // (get_stats) discriminate a persistent wait-set converging to
        // fleet width (dense waves) from a transient one stuck ≈1
        // (singleton waves), and a deadline hold (missing>0) from an
        // all-ready fire. Legacy/quorum never reach here as a WaitAll.
        if waitall_active {
            let active = self.policy.distinct_program_count() as u64;
            stats.fire.quorum.wave_active_sum.fetch_add(active, Relaxed);
            stats
                .fire
                .quorum
                .wave_missing_sum
                .fetch_add(missing.len() as u64, Relaxed);
            stats.fire.quorum.wave_fires.fetch_add(1, Relaxed);
        }
        if scheduler_trace_enabled() {
            sched_trace_write(&format!(
                "[pie-sched-trace] driver={} fire requests={} tokens={} prefill_like={} stashed={} distinct_programs={} wave_missing={}",
                driver_idx,
                self.batch.len(),
                total_tokens,
                self.batch.should_prefill_coalesce(),
                self.next_pending.is_some(),
                self.policy.distinct_program_count(),
                missing.len(),
            ));
        }
        let requests_to_fire = self.batch.take();
        let batch_size = requests_to_fire.len() as u64;

        // Per-identity (C3) co-batch accounting for THIS fire: for each
        // distinct `program_identity_hash` present, +1 fire and +rows =
        // the number of co-batched requests carrying it. Done here on the
        // scheduler thread (requests_to_fire is moved off-thread below).
        {
            let mut per_fire: std::collections::HashMap<u64, u64> =
                std::collections::HashMap::new();
            for req in &requests_to_fire {
                let mut seen: std::collections::HashSet<u64> =
                    std::collections::HashSet::new();
                for &h in &req.program_identity_hashes {
                    if h != 0 && seen.insert(h) {
                        *per_fire.entry(h).or_insert(0) += 1;
                    }
                }
            }
            for (hash, rows) in per_fire {
                stats.record_identity_fire(hash, rows);
            }
        }

        crate::probe_fire_record!(
            stats.fire.pre_dispatch.fire_prepare_us,
            fire_prepare_start.elapsed()
        );

        // Inter-fire instrumentation: time between consecutive fires,
        // and the post-dispatch-to-next-fire gap (rendezvous window).
        // The timestamps themselves (last_fire_spawn_micros,
        // last_dispatch_end_micros) are always-on — cheap atomic
        // swap/load. The accumulators are probe-gated.
        let now_us = now_micros();
        let last_spawn = stats.fire.last_fire_spawn_micros.swap(now_us, Relaxed);
        if last_spawn != 0 {
            crate::probe_fire_record!(
                stats.fire.inter_fire_us,
                std::time::Duration::from_micros(now_us.saturating_sub(last_spawn))
            );
        }
        let last_dispatch_end = stats.fire.last_dispatch_end_micros.load(Relaxed);
        if last_dispatch_end != 0 {
            crate::probe_fire_record!(
                stats.fire.post_dispatch_to_fire_us,
                std::time::Duration::from_micros(
                    now_us.saturating_sub(last_dispatch_end)
                )
            );
        }

        // Build the batched request on the scheduler thread (this
        // overlaps the GPU of any in-flight batch), ENQUEUE it in
        // fire-order here (fixing driver-inbox order == fire order,
        // so a forward `t+1` never reaches the worker before its
        // token-carryover source `t`), and AWAIT the response
        // off-thread so this thread is freed to collect/build the
        // next batch.
        let build_start = Instant::now();
        // G2 prebuilt-passthrough: a solo prebuilt beam fire is already
        // wire-final (B lanes folded in ptir_host) — fire it VERBATIM.
        // Folding it through `append_request_with_options` would
        // re-derive one page-run from a single `physical_page_ids` and
        // collapse the B-lane geometry (per-lane page-run/klen/kvm).
        let batch_req = if requests_to_fire.len() == 1
            && requests_to_fire[0].prebuilt
        {
            requests_to_fire[0].request.clone()
        } else {
            batch::build_batch_request(&requests_to_fire, page_size, &stats)
        };
        let submission_latency = build_start.elapsed();

        match driver::fire_batch_deferred(driver_idx, batch_req) {
            Ok(handle) => {
                // The batch is enqueued (its order fixed) — record it
                // as in-flight so the policy paces the next fire.
                self.policy.on_fired(batch_size as usize);

                // M-A1 liveness: pipelines demoted at the wave
                // miss-limit (WaitAll only) — terminate them so they
                // stop holding the fleet + reclaim their KV (alpha's
                // WS drop → arena free). Empty for legacy policies.
                //
                // A PROGRESSING carrier no longer reaches here: it
                // LEAVES the wave via the drain-gate `Leave{Suspend}`
                // (`drain_own_fires`) while gate-blocked, so it never
                // accrues misses and is never demoted. That retired the
                // old carrier-aware terminate-SKIP husk; a lane demoted
                // here is genuinely unresponsive → terminate it.
                for pid in self.policy.take_terminate_candidates() {
                    // BAR-2 cap-saturation defense (DECOUPLED from the
                    // retired skip — runs for every demoted lane): the
                    // lane's in-flight fires may be genuinely STUCK
                    // (e.g. an undelivered D1 completion — a live
                    // #17-tail candidate), pinning `policy.in_flight`
                    // at the cap so `decide_wave_at` never fires ANY
                    // wave (the whole fleet stalls). Free those cap
                    // slots BEFORE terminating; each released fire's
                    // later `FireCompletion` is a no-op (`released_fires`)
                    // — no double-decrement, WAR-ring invariant held (a
                    // genuinely-stuck fire never D2H'd a `pi.sampled`,
                    // so it holds no ring slot).
                    release_lane_in_flight(
                        pid,
                        &mut self.policy,
                        &mut self.in_flight_count,
                        &self.in_flight_fires,
                        &mut self.released_fires,
                    );
                    // Tombstone BEFORE terminating: a stale queued
                    // request from this pid must not re-join the
                    // wait-set (it was already removed in
                    // `decide_wave_at`). `process::terminate` also
                    // broadcasts a Leave, but tombstoning here closes
                    // the window until that drains.
                    self.tombstones.insert(pid);
                    // Demote = terminate (dead): drop + error its
                    // stash NOW so THIS fire's promotion below doesn't
                    // fire the dead pid's chain untracked (the async
                    // Leave from `process::terminate` only arrives a
                    // loop-iteration later).
                    self.suspended.remove(&pid);
                    crate::process::terminate(
                        pid,
                        Err("scheduler: wait-for-all miss-limit (unresponsive pipeline)"
                            .to_string()),
                    );
                }


                // Inter-batch bubble (host proxy): if the device had
                // no batch outstanding when this fire launched, the
                // gap since it went idle is a bubble. Depth-1
                // enqueue-ahead keeps a batch queued in steady state,
                // so `in_flight_count > 0` here and no bubble is
                // charged (F1: bubble → 0). The cold-start fire (no
                // prior idle timestamp) is not charged.
                //
                // The histogram records EVERY fire's inter-batch gap —
                // 0 when the device was busy (enqueue-ahead covered) or
                // on the cold-start fire, else the idle gap — so
                // `bubble_us_hist` yields a true p50/p99 across all fires
                // (always-on; the probe accumulator above stays gated).
                let bubble_us = if self.in_flight_count == 0 {
                    self.device_idle_since
                        .take()
                        .map(|idle| now_micros().saturating_sub(idle))
                        .unwrap_or(0)
                } else {
                    0
                };
                stats.record_bubble_us(bubble_us);
                if bubble_us > 0 {
                    crate::probe_fire_record!(
                        stats.fire.quorum.inter_batch_bubble_us,
                        std::time::Duration::from_micros(bubble_us)
                    );
                }
                // BAR-2 per-lane in-flight release: assign this fire a
                // unique id + record its tracked lane pids, so a
                // demoted lane's stuck fires can be found + their cap
                // slots released (`release_lane_in_flight`). Untracked
                // (`None` pid) requests are simply absent from the set.
                let fire_id = self.next_fire_id;
                self.next_fire_id = self.next_fire_id.wrapping_add(1);
                let fire_lanes: Vec<ProcessId> = requests_to_fire
                    .iter()
                    .filter_map(|r| r.pipeline_id)
                    .collect();
                self.in_flight_fires.insert(fire_id, fire_lanes);
                self.in_flight_count += 1;
                let stats_spawn = Arc::clone(&stats);
                let rt_handle_spawn = rt_handle.clone();
                let submit_tx_spawn = submit_tx.clone();
                let latency_tx_spawn = latency_tx.clone();
                rt_handle.spawn_blocking(move || {
                    // D1 async-fire liveness: the Drop-guard sends the
                    // `FireCompletion` from its Drop (normal path AND
                    // panic unwind), so a panic in `handle.wait()` /
                    // `dispatch_fired_batch` can't leak `in_flight`
                    // (the phantom-cap hang class). `complete()` below
                    // arms it with the real latency for the success
                    // path; an unwind before that sends an
                    // accounting-only fallback (pops the FIFO, no EWMA).
                    let mut fire_guard =
                        FireCompletionGuard::new(latency_tx_spawn, submission_latency, fire_id);
                    // Phase: driver_fire — block off-thread for the
                    // GPU response. (The ipc_submit probe was set on
                    // the scheduler thread during enqueue; under
                    // `profile-driver-cuda` it reads 0 here — the
                    // gpu_wait probe set in this task is accurate.)
                    let fire_start = Instant::now();
                    let fire_result = crate::probe_fire!(
                        stats_spawn.fire.execute.driver_fire_us,
                        {
                            let r = handle.wait();
                            let ipc_submit_us =
                                crate::probe::driver_cuda::take_ipc_submit_us();
                            let gpu_wait_us =
                                crate::probe::driver_cuda::take_gpu_wait_us();
                            let ipc_recv_us =
                                crate::probe::driver_cuda::take_ipc_recv_us();
                            if ipc_submit_us > 0 {
                                stats_spawn
                                    .driver_cuda
                                    .ipc_submit_us
                                    .fetch_add(ipc_submit_us, Relaxed);
                            }
                            if gpu_wait_us > 0 {
                                stats_spawn
                                    .driver_cuda
                                    .gpu_wait_us
                                    .fetch_add(gpu_wait_us, Relaxed);
                            }
                            if ipc_recv_us > 0 {
                                stats_spawn
                                    .driver_cuda
                                    .ipc_recv_us
                                    .fetch_add(ipc_recv_us, Relaxed);
                            }
                            r
                        }
                    );
                    let forward_latency = fire_start.elapsed();
                    // Arm the guard with the measured GPU/driver wait
                    // (Drop sends the real completion at scope end); a
                    // panic in dispatch below now still feeds the EWMA
                    // with this real sample, not the fallback.
                    fire_guard.complete(forward_latency);
                    let timing = response::dispatch_fired_batch(
                        fire_result,
                        requests_to_fire,
                        driver_id,
                        page_size,
                        &rt_handle_spawn,
                        Some(submit_tx_spawn),
                        &stats_spawn,
                    );
                    stats::record_fire_stats(
                        &stats_spawn,
                        &timing,
                        forward_latency,
                        batch_size,
                        total_tokens,
                    );
                    // `fire_guard` drops here → sends the real
                    // `FireCompletion` from a single site (its Drop).
                });
            }
            Err(e) => {
                // Enqueue failed (channel closed/aborted) — fail the
                // batch's requests; nothing went in flight.
                let msg =
                    format!("fire_batch_deferred failed for driver {driver_id}: {e:#}");
                for req in requests_to_fire {
                    req.send_result::<ForwardOutput>(
                        Err(anyhow::anyhow!(msg.clone())),
                        None,
                        page_size,
                    );
                }
            }
        }
    }

    /// Idle / backpressure wait when the policy declines to fire: drain a
    /// completion or poll-timeout. Returns `Break` when `req_rx` closed (the run
    /// loop should exit), else `Continue`. `next_pending`-guarded so a stashed
    /// request is never overwritten (guru #2).
    fn handle_wait(
        &mut self,
        wait_duration: Duration,
        req_rx: &crossbeam::channel::Receiver<PendingRequest>,
        latency_rx: &crossbeam::channel::Receiver<FireCompletion>,
        driver_idx: usize,
    ) -> std::ops::ControlFlow<()> {
        // `next_pending`-GUARD (guru #2): if a request is already
        // stashed (a prebuilt/capacity split the cap gate above held
        // back because `in_flight_count` is at `max_in_flight()`), do
        // NOT `recv(req_rx)` — that would overwrite `next_pending` and
        // silently DROP the stashed request (→ that pipeline hangs).
        // Only drain `latency_rx` (free a slot) or poll-timeout; the
        // next loop iteration re-hits the cap gate and force-fires the
        // batch the instant a completion drops in_flight below the cap.
        // A request arriving now stays buffered in `req_rx` and is
        // drained on the next uncontended iteration (nothing is lost).
        if self.next_pending.is_some() {
            crossbeam::channel::select! {
                recv(latency_rx) -> completion => {
                    if let Ok(c) = completion {
                        drain_completion(&mut self.policy, &c, &mut self.in_flight_count, &mut self.device_idle_since, &mut self.in_flight_fires, &mut self.released_fires);
                    }
                }
                default(wait_duration) => {}
            }
            return std::ops::ControlFlow::Continue(());
        }
        crossbeam::channel::select! {
            recv(req_rx) -> maybe_req => {
                match maybe_req {
                    Ok(pending) => {
                        let Some(pending) = prepare_pending_for_batch(&self.batch, pending)
                        else {
                            return std::ops::ControlFlow::Continue(());
                        };
                        match try_admit(
                            pending,
                            None,
                            false,
                            &mut self.batch,
                            &mut self.next_pending,
                            &mut self.policy,
                            &self.tombstones,
                            driver_idx,
                        ) {
                            AdmitOutcome::Consumed
                            | AdmitOutcome::Deferred => return std::ops::ControlFlow::Continue(()),
                            AdmitOutcome::Pushed => {}
                        }
                    }
                    Err(_) => return std::ops::ControlFlow::Break(()), // channel closed
                }
            }
            recv(latency_rx) -> completion => {
                if let Ok(c) = completion {
                    drain_completion(&mut self.policy, &c, &mut self.in_flight_count, &mut self.device_idle_since, &mut self.in_flight_fires, &mut self.released_fires);
                }
            }
            default(wait_duration) => {}
        }
        std::ops::ControlFlow::Continue(())
    }
}


// =============================================================================
// BatchScheduler
// =============================================================================

/// Per-driver batch scheduler.
///
/// Owns an RPC client, a scheduling policy, and a tokio task that
/// runs the batch accumulation and firing loop.
pub(crate) struct BatchScheduler {
    tx: crossbeam::channel::Sender<PendingRequest>,
    stats: Arc<SchedulerStats>,
}

impl BatchScheduler {
    /// Spawn a new batch scheduler for a single driver.
    ///
    /// The RPC connection is owned by the driver service; the scheduler
    /// only stores the driver index for routing calls.
    pub fn new(
        driver_id: DriverId,
        driver_idx: usize,
        page_size: u32,
        limits: SchedulerLimits,
        request_timeout_secs: u64,
    ) -> Self {
        let (tx, rx) = crossbeam::channel::unbounded::<PendingRequest>();
        let submit_tx = tx.clone();
        let stats = Arc::new(SchedulerStats::default());

        // Run the main scheduling loop on a dedicated OS thread with
        // crossbeam channels. Why: tokio's mpsc/select! wake-pickup
        // path takes ~100-200 µs because the receiver's waker has to be
        // scheduled onto a runtime worker. crossbeam's recv/select uses
        // futex parking directly — wake latency drops to ~5-15 µs.
        // execute_batch tasks still spawn on the shared tokio runtime
        // (captured via Handle) so they keep multi-worker parallelism
        // for the GPU/IPC and response dispatch.
        let rt_handle = tokio::runtime::Handle::current();
        let stats_for_loop = stats.clone();
        std::thread::Builder::new()
            .name(format!("pie-sched-{driver_idx}"))
            .spawn(move || {
                Self::run(
                    driver_id,
                    driver_idx,
                    rx,
                    submit_tx,
                    page_size,
                    limits,
                    request_timeout_secs,
                    stats_for_loop,
                    rt_handle,
                );
            })
            .expect("spawn pie-sched thread");

        Self { tx, stats }
    }

    /// Get a handle to the cumulative scheduler stats (lock-free).
    pub fn stats(&self) -> &Arc<SchedulerStats> {
        &self.stats
    }

    /// Submit a pre-translated forward pass request.
    pub fn submit(
        &self,
        request: pie_driver_abi::ForwardRequest,
        response_tx: oneshot::Sender<Result<ForwardOutput>>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Result<()> {
        self.tx
            .send(PendingRequest::direct(
                request,
                response_tx,
                physical_page_ids,
                last_page_len,
                Vec::new(),
                None,
                0,
            ))
            .map_err(|_| anyhow::anyhow!("scheduler channel closed"))?;
        Ok(())
    }

    /// Cloneable handle for tasks that need to submit outside the
    /// scheduler's `run` loop.
    pub(crate) fn handle(&self) -> SchedulerHandle {
        SchedulerHandle {
            tx: self.tx.clone(),
        }
    }

    // =========================================================================
    // Internal: Scheduling Loop
    // =========================================================================

    /// Main scheduling loop for a single driver. Sync OS thread —
    /// recv/select use futex parking (no tokio waker overhead).
    fn run(
        driver_id: DriverId,
        driver_idx: usize,
        req_rx: crossbeam::channel::Receiver<PendingRequest>,
        submit_tx: crossbeam::channel::Sender<PendingRequest>,
        page_size: u32,
        limits: SchedulerLimits,
        request_timeout_secs: u64,
        stats: Arc<SchedulerStats>,
        rt_handle: tokio::runtime::Handle,
    ) {
        // The per-request timeout is currently unused by the run-ahead fire
        // path (the driver wait is bounded by the channel). Kept consuming the
        // param so the scheduler signature is stable for a future per-request
        // deadline.
        let _request_timeout = Duration::from_secs(request_timeout_secs);

        // Channel for batch-completion feedback to the policy: the off-thread
        // forward (GPU) latency + the on-thread submission latency.
        let (latency_tx, latency_rx) = crossbeam::channel::unbounded::<FireCompletion>();

        // M-A1 Stage 2: wait-for-all pipeline-`Leave` channel. The single wave
        // rule always registers its sender (so `notify_pipeline_leave` reaches
        // it) and acts on Leaves.
        let waitall_active = true;
        let (lifecycle_tx, lifecycle_rx) = crossbeam::channel::unbounded::<LifecycleEvent>();
        if waitall_active {
            if let Ok(mut senders) = lifecycle_senders().lock() {
                senders.push(lifecycle_tx);
            }
        }

        // All per-driver mutable loop state (batch accumulator, fire policy,
        // in-flight bookkeeping, dep-stash, tombstones); see `LoopState`.
        let mut st = LoopState::new(limits, page_size, &stats);

        // One-time boot diagnostic (no per-fire perturbation): resolve the
        // accum-hold OnceLock ONCE so a trace run reveals whether the test-only
        // co-batch hold actually engaged (`Some(us)`) or is fire-on-arrival
        // (`None`). Emitting this in the hot accum loop would itself add stderr
        // latency and mask the very arrival-timing it measures (the grammar10
        // heisenbug), so it is stamped here at boot, outside the loop.
        if scheduler_trace_enabled() {
            sched_trace_write(&format!(
                "[pie-sched-trace] driver={} boot accum_hold_us={:?}",
                driver_idx,
                scheduler_accum_hold_us(),
            ));
        }

        'run_loop: loop {
            // Drain completed batch feedback (non-blocking): GPU latency →
            // on_complete (+ FIFO pop), submission latency → on_submitted.
            while let Ok(c) = latency_rx.try_recv() {
                drain_completion(&mut st.policy, &c, &mut st.in_flight_count, &mut st.device_idle_since, &mut st.in_flight_fires, &mut st.released_fires);
            }

            // M-A1 Stage 2: drain pipeline `Leave`s (terminate / cancel / exit /
            // contention-preempt) — drop each from the wait-set so it no longer
            // holds the wave, and tombstone it so a stale queued request can't
            // implicitly re-join it. No-op on the legacy path (channel empty).
            while let Ok(ev) = lifecycle_rx.try_recv() {
                apply_lifecycle_event(
                    ev,
                    &mut st.policy,
                    &mut st.tombstones,
                    &mut st.suspended,
                );
            }

            // Wait for first request if batch is empty. crossbeam's
            // recv() parks via futex — far lower wake latency than
            // tokio's mpsc waker path. Time the wait (once warm) as the
            // steady-state scheduler idle-wait — the round-trip residual of R
            // (dispatch→inferlet resubmit→SERVICE→recv), excluded from the
            // scheduler's own build/decide cost.
            let recv_block_start = Instant::now();
            let was_empty = st.batch.is_empty();
            let mut first_submitted_at_us = 0u64;
            while st.batch.is_empty() {
                let pending = if let Some(pending) = st.next_pending.take() {
                    pending
                } else {
                    // Wait for the first request, BUT keep draining completion
                    // feedback while we wait. Otherwise, at the in-flight cap with
                    // an empty batch, a completed fire's `FireCompletion` sits
                    // undrained (`in_flight` never decrements) until a resubmit
                    // happens to wake `req_rx` — and that resubmit can be gated
                    // behind the very slot we're failing to free (the cap≥4
                    // lost-wakeup, M-AB). Draining here keeps `in_flight` accurate
                    // so the next fire isn't blocked by a phantom cap.
                    //
                    crossbeam::channel::select! {
                        recv(req_rx) -> msg => match msg {
                            Ok(p) => p,
                            Err(_) => break 'run_loop,
                        },
                        recv(latency_rx) -> completion => {
                            if let Ok(c) = completion {
                                drain_completion(&mut st.policy, &c, &mut st.in_flight_count, &mut st.device_idle_since, &mut st.in_flight_fires, &mut st.released_fires);
                            }
                            continue;
                        }
                        // BAR 1: a `Join` (or any lifecycle event) must wake the
                        // idle-blocked loop — the select otherwise watches only
                        // req_rx/latency_rx, so a resume while the fleet is frozen
                        // (in-flight 0, all stash parked) is never seen and the
                        // un-park never happens. Apply it, then loop → the pump
                        // above re-drives the now-un-parked stash.
                        recv(lifecycle_rx) -> ev => {
                            if let Ok(ev) = ev {
                                apply_lifecycle_event(
                                    ev,
                                    &mut st.policy,
                                    &mut st.tombstones,
                                    &mut st.suspended,
                                );
                            }
                            continue;
                        }
                    }
                };
                let Some(pending) = prepare_pending_for_batch(&st.batch, pending) else {
                    continue;
                };
                first_submitted_at_us = pending.submitted_at_us;
                st.policy.on_arrival(
                    &pending.program_identity_hashes,
                    st.tombstones.filter(pending.pipeline_id),
                    Instant::now(),
                );
                st.batch.push(pending);
            }
            // Record only when warm (a fire has spawned) so the cold-start wait
            // for the first-ever request doesn't skew the steady-state metric.
            if was_empty && stats.fire.last_fire_spawn_micros.load(Relaxed) != 0 {
                crate::probe_fire_record!(
                    stats.fire.recv_block_wait_us,
                    recv_block_start.elapsed()
                );
                // R-decomposition (charlie): split the round-trip into the guest
                // wake+rebuild (dispatch→submit) and the SERVICE hop (submit→recv).
                // Only the `submit_async` decode path stamps `submitted_at_us`
                // (!= 0); solo/prebuilt/chunked re-submits skip it.
                let last_dispatch_end = stats.fire.last_dispatch_end_micros.load(Relaxed);
                if last_dispatch_end != 0 && first_submitted_at_us != 0 {
                    crate::probe_fire_record!(
                        stats.fire.guest_roundtrip_us,
                        Duration::from_micros(
                            first_submitted_at_us.saturating_sub(last_dispatch_end)
                        )
                    );
                    crate::probe_fire_record!(
                        stats.fire.service_queue_us,
                        Duration::from_micros(
                            now_micros().saturating_sub(first_submitted_at_us)
                        )
                    );
                }
            }

            // Accumulate more requests (non-blocking). If a request is
            // already stashed for the next batch, fire the current batch
            // before reading more; overwriting the stash would drop that
            // request's response channel.
            let accum_start = Instant::now();
            // Test-only deterministic co-batch hold (`PIE_SCHED_ACCUM_HOLD_US`):
            // block up to the deadline after the first request so concurrent
            // requests land in the same drain window (deterministic
            // `forward_R >= 2` for the merged-path verify). Unset → `None` →
            // today's fire-on-arrival, unchanged.
            let accum_deadline =
                scheduler_accum_hold_us().map(|us| accum_start + Duration::from_micros(us));
            while st.next_pending.is_none() {
                let pending = match accum_deadline
                    .and_then(|d| d.checked_duration_since(Instant::now()))
                {
                    Some(remaining) => match req_rx.recv_timeout(remaining) {
                        Ok(p) => p,
                        Err(crossbeam::channel::RecvTimeoutError::Timeout) => break,
                        Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                            break 'run_loop;
                        }
                    },
                    None => match req_rx.try_recv() {
                        Ok(p) => p,
                        Err(crossbeam::channel::TryRecvError::Empty) => break,
                        Err(crossbeam::channel::TryRecvError::Disconnected) => {
                            break 'run_loop;
                        }
                    },
                };
                let Some((pending, usage)) = prepare_pending_with_usage(&st.batch, pending)
                else {
                    continue;
                };
                match try_admit(
                    pending,
                    Some(usage),
                    true,
                    &mut st.batch,
                    &mut st.next_pending,
                    &mut st.policy,
                    &st.tombstones,
                    driver_idx,
                ) {
                    AdmitOutcome::Consumed => continue,
                    AdmitOutcome::Deferred => break,
                    AdmitOutcome::Pushed => {
                        if st.batch.is_full() {
                            break;
                        }
                    }
                }
            }
            crate::probe_fire_record!(
                stats.fire.accumulate.accum_loop_us,
                accum_start.elapsed()
            );

            // Ask the policy what to do. A stashed `next_pending` (prebuilt solo /
            // capacity split) forces the current batch out to make room. Under
            // WaitAll the ①-bug run-ahead `would_depend` requests no longer reach
            // here (they go to `dep_stash`, above) — only prebuilt/capacity do,
            // which SHOULD force-fire.
            //
            // guru #2 (cap invariant): the force-fire is gated on
            // `in_flight_count < max_in_flight()` so it can NEVER push depth past
            // the cap (bravo's carrier runs at cap=k; the driver WAR-event ring is
            // sized D−1 — a force-fire past the cap silently corrupts the shared
            // sample buffer). At the cap we fall through to `policy.decide`, which
            // returns `Wait`; the Wait branch below is `next_pending`-GUARDED (it
            // only drains `latency_rx`, never `recv(req_rx)`) so it cannot overwrite
            // the stashed request. When a completion frees a slot, the next
            // iteration re-hits this gate with `in_flight_count < max` and fires.
            let decision = if force_fire_ready(
                st.next_pending.is_some(),
                st.in_flight_count,
                super::policy::max_in_flight(),
            ) {
                FireOutcome::Fire {
                    missing: Vec::new(),
                }
            } else {
                st.policy.decide(st.batch.len(), Instant::now())
            };
            match decision {
                FireOutcome::Fire { missing } => {
                    st.fire_batch(missing, &req_rx, &submit_tx, &latency_tx, &rt_handle, &stats, driver_id, driver_idx, page_size, waitall_active);
                }
                FireOutcome::Wait(wait_duration) => {
                    if st
                        .handle_wait(wait_duration, &req_rx, &latency_rx, driver_idx)
                        .is_break()
                    {
                        break 'run_loop;
                    }
                }
            }
        }

        // Shutdown: fire the remaining batch synchronously so any
        // inferlets still awaiting responses get them before we exit.
        // ~10 ms of additional shutdown latency in the worst case.
        if !st.batch.is_empty() {
            let requests = st.batch.take();
            let _ = Self::execute_batch_blocking(
                driver_idx,
                requests,
                driver_id,
                page_size,
                &rt_handle,
                None,
                &stats,
            );
        }
    }



    /// Build + enqueue + await + dispatch a batch synchronously on the caller's
    /// thread. Used for the shutdown drain (no overlap needed); the hot path
    /// instead splits these phases across the scheduler thread and a spawned
    /// task so the GPU wait overlaps the next batch's build.
    fn execute_batch_blocking(
        driver_idx: usize,
        requests: Vec<PendingRequest>,
        driver_id: DriverId,
        page_size: u32,
        rt_handle: &tokio::runtime::Handle,
        submit_tx: Option<crossbeam::channel::Sender<PendingRequest>>,
        stats: &SchedulerStats,
    ) -> BatchExecutionTiming {
        let batch_req = batch::build_batch_request(&requests, page_size, stats);
        let fire_result = match driver::fire_batch_deferred(driver_idx, batch_req) {
            Ok(handle) => handle.wait(),
            Err(e) => Err(e),
        };
        response::dispatch_fired_batch(
            fire_result,
            requests,
            driver_id,
            page_size,
            rt_handle,
            submit_tx,
            stats,
        )
    }

}
