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
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::oneshot;

use crate::arena::PhysicalPageId;
use crate::driver::{self, DriverId, SchedulerLimits};
use crate::process::ProcessId;

use super::adaptive_policy::RunAheadPolicy;
use super::{ForwardOutput, request};

mod chunked;
pub(super) mod quorum;

use chunked::ChunkContinuation;

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

/// Whether the PTIR quorum fire rule (thrust-2 §3, F1–F6) drives the live
/// scheduler in place of the legacy run-ahead timing formula. Gated behind the
/// `run-ahead` feature AND `PIE_SCHED_POLICY=quorum` so the default build keeps
/// the legacy `RunAheadPolicy` (no-regression, masterplan §4). Off ⇒ the quorum
/// core is compiled but never selected; the legacy path is byte-for-byte.
fn quorum_policy_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        cfg!(feature = "run-ahead")
            && std::env::var("PIE_SCHED_POLICY")
                .map(|v| v.eq_ignore_ascii_case("quorum"))
                .unwrap_or(false)
    })
}

/// Whether the wait-for-all-active-pipelines wave rule (`WaitAllPolicy`, M-A1)
/// drives the scheduler. Gated behind the `run-ahead` feature AND
/// `PIE_SCHED_POLICY=waitall`; off ⇒ the legacy quorum/run-ahead path is
/// byte-for-byte. Takes precedence over `quorum` when both would match.
fn waitall_policy_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        cfg!(feature = "run-ahead")
            && std::env::var("PIE_SCHED_POLICY")
                .map(|v| v.eq_ignore_ascii_case("waitall"))
                .unwrap_or(false)
    })
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

/// Max per-pipeline `dep_stash` depth. bravo's device-resident carrier submits a
/// k-deep chain UP FRONT (`decode_pipelined_deep`); the stash holds the ≤k links
/// awaiting promotion. This is a generous fail-loud ceiling against a runaway
/// pre-submission loop (unbounded stash growth = a memory leak), NOT a
/// functional limit — a well-behaved carrier stays well under it.
const MAX_DEP_STASH_PER_PIPELINE: usize = 64;

/// N+k deep-pre-submission chain ordering (bravo's device-resident carrier).
/// The carrier submits a k-deep chain (N+1…N+k) UP FRONT, each link consuming
/// the prior fire's sample via next-input links. `would_depend_on_batch` only
/// inspects the CURRENT batch, so a link whose producer is itself STASHED (not
/// in the batch) reads as independent — it would co-batch with the head and
/// read a not-yet-sampled token. Guard: once a pipeline has ANY stashed request,
/// every subsequent request from it is the next chain link → stash it in FIFO
/// order (promoted one-per-fire into successive waves at cap=k). Returns `None`
/// if stashed (caller `continue`s); `Some(pending)` to batch it normally. Off
/// WaitAll or untracked (`None` pid) → passthrough (legacy force-fire is correct
/// there). A runaway chain past [`MAX_DEP_STASH_PER_PIPELINE`] fails loud.
fn stash_chain_continuation(
    dep_stash: &mut std::collections::HashMap<ProcessId, std::collections::VecDeque<PendingRequest>>,
    waitall_active: bool,
    pending: PendingRequest,
) -> Option<PendingRequest> {
    if !waitall_active {
        return Some(pending);
    }
    let Some(pid) = pending.pipeline_id else {
        return Some(pending);
    };
    match dep_stash.get(&pid) {
        Some(q) if !q.is_empty() => {
            if q.len() >= MAX_DEP_STASH_PER_PIPELINE {
                pending.send_error(format!(
                    "scheduler: pipeline pre-submission chain exceeded {MAX_DEP_STASH_PER_PIPELINE} deep (runaway carrier)"
                ));
                return None;
            }
            dep_stash.entry(pid).or_default().push_back(pending);
            None
        }
        _ => Some(pending),
    }
}

/// Halt a TERMINATED pipeline's stashed run-ahead chain (Phase-2). Drops the
/// pid's `dep_stash` entry and errors each stashed forward so its staged
/// txn/pins release. Applied only on [`LeaveKind::Terminate`] (dead: cancel /
/// exit / miss-limit) — the pipeline's passes will never resume, so erroring is
/// correct + reclaims. (A SUSPENDED victim is instead skip-promoted, never
/// errored — it RESUMES its passes on restore.) Without this, the post-fire
/// promotion keeps firing the dead pid's N+k links UNTRACKED — wasted device
/// work + delivery to a closed channel. No-op if nothing was stashed.
fn clear_left_pipeline_stash(
    dep_stash: &mut std::collections::HashMap<ProcessId, std::collections::VecDeque<PendingRequest>>,
    pid: ProcessId,
) {
    if let Some(stashed) = dep_stash.remove(&pid) {
        for req in stashed {
            req.send_error(
                "scheduler: pipeline left the fleet (contention preempt / terminate) — \
                 stashed run-ahead chain aborted"
                    .to_string(),
            );
        }
    }
}

/// Promote one link per NON-suspended pipeline from `dep_stash` into `batch`
/// (Piece 4 + BAR 1). Each pipeline's stashed run-ahead chain advances one fire:
/// pop its front link, mark it ready (`on_arrival`, tombstone-filtered), push it.
/// SUSPENDED pipelines are skipped (parked for transparent resume on `Join`).
/// Called post-fire AND at the BAR-1 ignition pump — the latter is the ONLY
/// re-drive when NO fire is in flight (promotion otherwise runs only post-fire,
/// so a zero-in-flight idle with a promotable stash would freeze permanently:
/// no fire ⇒ no promotion ⇒ no submission ⇒ no fire).
fn promote_dep_stash(
    dep_stash: &mut std::collections::HashMap<ProcessId, std::collections::VecDeque<PendingRequest>>,
    suspended: &std::collections::HashSet<ProcessId>,
    policy: &mut FirePolicy,
    tombstones: &Tombstones,
    batch: &mut BatchAccumulator,
) {
    if dep_stash.is_empty() {
        return;
    }
    let now = Instant::now();
    let pids: Vec<ProcessId> = dep_stash.keys().copied().collect();
    for pid in pids {
        if suspended.contains(&pid) {
            continue;
        }
        let req = dep_stash.get_mut(&pid).and_then(|q| q.pop_front());
        if dep_stash.get(&pid).is_some_and(|q| q.is_empty()) {
            dep_stash.remove(&pid);
        }
        if let Some(req) = req {
            policy.on_arrival(&req.program_identity_hashes, tombstones.filter(Some(pid)), now);
            batch.push(req);
        }
    }
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
    dep_stash: &mut std::collections::HashMap<ProcessId, std::collections::VecDeque<PendingRequest>>,
) {
    match ev {
        LifecycleEvent::Leave(pid, LeaveKind::Terminate) => {
            policy.on_pipeline_leave(pid);
            tombstones.insert(pid);
            suspended.remove(&pid);
            clear_left_pipeline_stash(dep_stash, pid);
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
// Scheduling Policy Trait
// =============================================================================

/// Pluggable scheduling policy.
///
/// A policy receives event callbacks (`on_arrival`, `on_complete`,
/// `on_fired`) and returns a [`Decision`] when asked whether to fire
/// the current batch.
pub(super) trait SchedulingPolicy: Send {
    /// A request was added to the accumulator. `program_identity_hashes` are the
    /// per-program `program_identity_hash`es carried by this request (one per
    /// program in its sampler pass; empty for plain decode) — run-ahead policies
    /// union them into the window's distinct-program set to drive the #10
    /// dedup-aware accumulation. Other policies ignore them.
    fn on_arrival(&mut self, program_identity_hashes: &[u64]);

    /// A batch finished executing. `latency` is the wall-clock time
    /// the forward pass took on the driver.
    fn on_complete(&mut self, latency: Duration);

    /// The current batch was fired. `fired_size` is the number of
    /// requests in the batch — policies use it to learn the steady-
    /// state cohort size and avoid firing partial batches in the next
    /// cycle.
    fn on_fired(&mut self, fired_size: usize);

    /// The current batch was submitted (enqueued). `submission_latency` is the
    /// host-side enqueue duration; run-ahead policies EWMA it into the
    /// `lead_time` (how far ahead of an in-flight batch's completion to fire so
    /// the next enqueue lands just-in-time). Default no-op for non-run-ahead
    /// policies, whose fire is synchronous (no separate submission phase).
    fn on_submitted(&mut self, _submission_latency: Duration) {}

    /// Decide whether to fire or wait, given the current batch size.
    /// `&mut self` so policies can update internal state on every poll.
    fn decide(&mut self, current_batch_size: usize) -> Decision;

    /// The number of DISTINCT programs (`program_identity_hash`) accumulated in
    /// the current not-yet-fired window — the #10 distinct-count witness (read at
    /// the fire trace so the verify can assert dedup: N-same-grammar ⇒ 1, and the
    /// distinct-burst cap: N-distinct ⇒ N). Default `0` for policies that don't
    /// track it; the run-ahead policy returns its live set size.
    fn distinct_program_count(&self) -> usize {
        0
    }
}

// =============================================================================
// Scheduling Decision
// =============================================================================

/// The outcome of a scheduling policy decision.
pub(super) enum Decision {
    /// Fire the current batch immediately.
    Fire,
    /// Wait for more requests, up to the given duration. Greedy-only under
    /// FCFS never constructs this; collapsing the policy abstraction is a
    /// deferred follow-up.
    #[allow(dead_code)]
    Wait(Duration),
}

// =============================================================================
// FirePolicy — M-A1 run-loop policy dispatch (wait-for-all vs legacy)
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

/// Dispatches the run loop between the legacy trait policies (RunAhead / Quorum,
/// via `Box<dyn SchedulingPolicy>`) and the wait-for-all [`WaitAllPolicy`] — a
/// concrete type whose wave methods (pipeline membership, the `missing` list,
/// terminate candidates) the `SchedulingPolicy` trait surface can't carry.
/// Selected by `PIE_SCHED_POLICY` (`waitall` ⇒ WaitAll, else Legacy).
enum FirePolicy {
    Legacy(Box<dyn SchedulingPolicy>),
    WaitAll(super::adaptive_policy::WaitAllPolicy),
}

impl FirePolicy {
    fn on_submitted(&mut self, latency: Duration) {
        match self {
            FirePolicy::Legacy(p) => p.on_submitted(latency),
            FirePolicy::WaitAll(w) => w.on_submitted(latency),
        }
    }

    fn on_complete(&mut self, latency: Duration) {
        match self {
            FirePolicy::Legacy(p) => p.on_complete(latency),
            FirePolicy::WaitAll(w) => w.on_complete(latency),
        }
    }

    fn on_fired(&mut self, fired_size: usize) {
        match self {
            FirePolicy::Legacy(p) => p.on_fired(fired_size),
            FirePolicy::WaitAll(w) => w.on_fired(fired_size),
        }
    }

    /// A request entered the current wave. Legacy policies consume the identity
    /// hashes (#10 dedup); WaitAll marks the pipeline ready (wave membership) and
    /// arms the straggler clock — `None` rides as untracked (prebuilt/beam).
    fn on_arrival(
        &mut self,
        program_identity_hashes: &[u64],
        pipeline_id: Option<ProcessId>,
        now: Instant,
    ) {
        match self {
            FirePolicy::Legacy(p) => p.on_arrival(program_identity_hashes),
            FirePolicy::WaitAll(w) => w.on_pipeline_request(pipeline_id, now),
        }
    }

    fn decide(&mut self, current_batch_size: usize, now: Instant) -> FireOutcome {
        match self {
            FirePolicy::Legacy(p) => match p.decide(current_batch_size) {
                Decision::Fire => FireOutcome::Fire {
                    missing: Vec::new(),
                },
                Decision::Wait(d) => FireOutcome::Wait(d),
            },
            FirePolicy::WaitAll(w) => match w.decide_wave_at(current_batch_size, now) {
                super::adaptive_policy::WaveDecision::Fire { missing } => {
                    FireOutcome::Fire { missing }
                }
                super::adaptive_policy::WaveDecision::Wait(d) => FireOutcome::Wait(d),
            },
        }
    }

    fn distinct_program_count(&self) -> usize {
        match self {
            FirePolicy::Legacy(p) => p.distinct_program_count(),
            FirePolicy::WaitAll(w) => w.active_pipelines(),
        }
    }

    /// Pipelines demoted at the consecutive-miss limit (WaitAll only) — the run
    /// loop `process::terminate`s them (M-A1 liveness). Empty for legacy.
    fn take_terminate_candidates(&mut self) -> Vec<ProcessId> {
        match self {
            FirePolicy::WaitAll(w) => w.take_terminate_candidates(),
            FirePolicy::Legacy(_) => Vec::new(),
        }
    }

    /// A pipeline left the fleet — drop it from the wait-set (WaitAll only).
    /// Stage 2: driven by the process-lifecycle `lifecycle_rx` hook. Until then
    /// an exited pipeline is removed by the miss-counter backstop.
    #[allow(dead_code)]
    fn on_pipeline_leave(&mut self, pipeline_id: ProcessId) {
        if let FirePolicy::WaitAll(w) = self {
            w.on_pipeline_leave(pipeline_id);
        }
    }
}

// =============================================================================
// SchedulerStats (lock-free snapshot for monitoring)
// =============================================================================

pub const SYSTEM_SPEC_DRAFT_POS_BUCKETS: usize = 32;

/// Bounded per-identity (C3) co-batch table width. Fleets carry ~10 distinct
/// program identities; a first-seen table of this many slots covers them with
/// headroom. A fleet exceeding it reads `identities_dropped > 0` (fail-loud
/// "saturated" — never silently missing an identity).
pub const PER_IDENTITY_BATCH_CAP: usize = 32;

/// Inter-batch bubble histogram: exclusive upper bounds (µs) per bucket. A
/// boundary is pinned at 100 (the masterplan p50 gate) so "p50 < 100 µs" reads as
/// "the p50 bucket's upper bound ≤ 100". `u64::MAX` is the overflow catch-all.
pub const BUBBLE_HIST_UPPER_US: [u64; 16] = [
    1, 2, 4, 8, 16, 32, 64, 100, 150, 250, 500, 1_000, 2_000, 8_000, 32_000, u64::MAX,
];

/// Cumulative stats exposed for monitoring. Updated atomically after each batch.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    // ── Always-on counters (no Instant::now needed). ────────────────────────
    pub total_batches: AtomicU64,
    pub total_tokens_processed: AtomicU64,
    /// Total request count across all batches (sum of batch sizes).
    /// Divide by `total_batches` for mean batch size in requests.
    pub total_requests_processed: AtomicU64,
    /// Largest forward request count ever fired by this scheduler.
    pub max_forward_requests_observed: AtomicU64,
    /// Coarse histogram of batch sizes. Buckets:
    /// [0]=1, [1]=2-3, [2]=4-7, [3]=8-15, [4]=16-31,
    /// [5]=32-63, [6]=64-127, [7]=128+.
    pub batch_size_hist: [AtomicU64; 8],
    pub last_batch_latency_us: AtomicU64,
    pub cumulative_latency_us: AtomicU64,
    pub system_spec_draft_tokens_proposed: AtomicU64,
    pub system_spec_draft_tokens_accepted: AtomicU64,
    pub system_spec_draft_tokens_proposed_per_pos:
        [AtomicU64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
    pub system_spec_draft_tokens_accepted_per_pos:
        [AtomicU64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],

    // ── Per-identity (C3) co-batch density (pentathlon per-identity-batch-density
    //    probe = rows / fires per program identity). First-seen bounded table keyed
    //    on `program_identity_hash` (bytecode ⊕ manifest — the SAME key the quorum
    //    scheduler batches / dedups / compiles on, so the telemetry can't disagree
    //    with the batching identity it measures). Written by the single per-driver
    //    scheduler thread; read (Relaxed) by `aggregate_stats`.
    /// Slot key = `program_identity_hash` (0 = empty slot).
    pub per_identity_hash: [AtomicU64; PER_IDENTITY_BATCH_CAP],
    /// Fires that included ≥1 request of this identity.
    pub per_identity_fires: [AtomicU64; PER_IDENTITY_BATCH_CAP],
    /// Total rows (co-batched requests) fired for this identity.
    pub per_identity_rows: [AtomicU64; PER_IDENTITY_BATCH_CAP],
    /// Fire-records dropped because the table was saturated (drop-new). > 0 ⇒ the
    /// fleet has more than `PER_IDENTITY_BATCH_CAP` distinct identities; the
    /// per-identity table is a bounded sample, NOT silently missing.
    pub identities_dropped: AtomicU64,

    /// Inter-batch bubble histogram (always-on, HOST PROXY) — one count per fire,
    /// bucketed by [`BUBBLE_HIST_UPPER_US`]. Yields a true p50/p99 (the masterplan
    /// gate) across ALL fires (0 when enqueue-ahead covered the gap), unlike the
    /// probe-gated `fire.quorum.inter_batch_bubble_us` accumulator (average,
    /// non-zero only). The host proxy stamps device-idle at the RUST enqueue point,
    /// so it OVER-counts by the host submit/handshake delay — see the driver
    /// histogram below for the accurate device-idle p50.
    pub bubble_us_hist: [AtomicU64; BUBBLE_HIST_UPPER_US.len()],
    /// Inter-batch bubble histogram (DRIVER STAMP) — fed by the CUDA driver's
    /// `probe_device_idle_us` (`t0_entry − t5_prev_retire`, the true on-device idle
    /// between a batch retiring and the next entering). Populated only when the
    /// driver stamps it (profile-driver-cuda on the driver side); empty otherwise,
    /// in which case readers fall back to the host-proxy histogram. This is the
    /// accurate G3 bubble-p50 measurement (no host-side over-count).
    pub bubble_us_hist_driver: [AtomicU64; BUBBLE_HIST_UPPER_US.len()],


    // ── Fire-domain probes (gated behind `profile-fire` feature). ───────────
    //
    // Hierarchy + invariants documented in `crate::probe::fire`. Writers
    // use the `probe_fire!` macro from that module so the fetch_add
    // disappears when the feature is off. The struct itself is always
    // defined so callers and readers compile uniformly.
    pub fire: crate::probe::fire::FireProbes,

    // ── Driver-fire phase breakdown (gated behind `profile-driver-cuda`). ──
    //
    // Decomposes the `fire.execute.driver_fire_us` bucket into Rust
    // (ipc_submit / gpu_wait / ipc_recv) and C++ host phases (wire_parse
    // / plan / h2d / kernel_launch / sync / response_build). See
    // `crate::probe::driver_cuda` for the plumbing.
    pub driver_cuda: crate::probe::driver_cuda::DriverCudaProbes,
}

impl SchedulerStats {
    /// Record one fire's contribution for `hash`: +1 fire, +`rows` co-batched
    /// requests. First-seen linear insertion into the bounded per-identity table;
    /// drop-new + bump `identities_dropped` on saturation (fail-loud). Called only
    /// from the single per-driver scheduler thread, so plain load/store is race-free.
    pub fn record_identity_fire(&self, hash: u64, rows: u64) {
        if hash == 0 {
            return; // 0 is the empty-slot sentinel; a hashless plain forward isn't C3-keyed.
        }
        for slot in 0..PER_IDENTITY_BATCH_CAP {
            let cur = self.per_identity_hash[slot].load(Relaxed);
            if cur == hash {
                self.per_identity_fires[slot].fetch_add(1, Relaxed);
                self.per_identity_rows[slot].fetch_add(rows, Relaxed);
                return;
            }
            if cur == 0 {
                self.per_identity_hash[slot].store(hash, Relaxed);
                self.per_identity_fires[slot].fetch_add(1, Relaxed);
                self.per_identity_rows[slot].fetch_add(rows, Relaxed);
                return;
            }
        }
        // Table saturated → drop this record, count it (fail-loud).
        self.identities_dropped.fetch_add(1, Relaxed);
    }

    /// Record one fire's inter-batch bubble (µs) into the HOST-PROXY histogram.
    /// Called only from the single per-driver scheduler thread (race-free plain
    /// fetch_add).
    pub fn record_bubble_us(&self, us: u64) {
        self.bubble_us_hist[Self::bubble_bucket(us)].fetch_add(1, Relaxed);
    }

    /// Record one fire's inter-batch bubble (µs) into the DRIVER-STAMP histogram
    /// (the accurate `probe_device_idle_us`). Called from the same single
    /// scheduler thread at response-processing time.
    pub fn record_bubble_us_driver(&self, us: u64) {
        self.bubble_us_hist_driver[Self::bubble_bucket(us)].fetch_add(1, Relaxed);
    }

    #[inline]
    fn bubble_bucket(us: u64) -> usize {
        BUBBLE_HIST_UPPER_US
            .iter()
            .position(|&upper| us < upper)
            .unwrap_or(BUBBLE_HIST_UPPER_US.len() - 1)
    }
}


/// Out-of-band data execute_batch reports back to the run loop. Per-fire
/// *timing* probes are no longer in this struct — they're recorded
/// directly into `stats.fire.*` via `probe_fire!`. What's left here is
/// genuine fire-output data (spec-decoding draft counters) that the run
/// loop then folds into the spec-domain atomics.
#[derive(Debug, Default, Clone, Copy)]
struct BatchExecutionTiming {
    system_spec_draft_tokens_proposed: u64,
    system_spec_draft_tokens_accepted: u64,
    system_spec_draft_tokens_proposed_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
    system_spec_draft_tokens_accepted_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
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
struct PendingRequest {
    request: pie_driver_abi::ForwardRequest,
    completion: Completion,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
    /// #10: per-program `program_identity_hash` for this request (one per program
    /// in its sampler pass; empty for plain decode). Computed once at attach
    /// (host-side, before carrier encoding) and threaded to the policy's
    /// distinct-program set via `on_arrival` — runtime-side only, never on the
    /// wire `ForwardRequest`.
    program_identity_hashes: Vec<u64>,
    /// M-A1 (wait-for-all rebuild): the submitting pipeline's `ProcessId` — the
    /// wave-barrier membership key (fire when every active pipeline has submitted
    /// its N+1). `None` for solo/prebuilt fires (beam passthrough), which bypass
    /// the wave. Threaded from `execute_impl(state.id())` at submit; consumed by
    /// the QuorumModel policy core (M-A2, guru). Runtime-side only, never on the
    /// wire `ForwardRequest`.
    #[allow(dead_code)]
    pipeline_id: Option<ProcessId>,
    /// R-decomposition probe (charlie): the `now_micros()` stamp taken at
    /// `submit_async` (the guest's resubmit instant), in the scheduler epoch so
    /// it's comparable to `last_dispatch_end_micros`. Splits the round-trip R
    /// into `guest_roundtrip_us` (dispatch→submit = guest wake + wasm + rebuild)
    /// and `service_queue_us` (submit→recv = the SERVICE actor hop). `0` for
    /// non-`submit_async` paths (prebuilt/beam, plain submit, chunked/dispatch
    /// re-submits) — skipped in the accumulation.
    submitted_at_us: u64,
    /// G2 prebuilt-passthrough: this request carries a COMPLETE, wire-final
    /// multi-lane `ForwardRequest` (a PTIR beam fire = B forward lanes, one
    /// program/epilogue) that must fire VERBATIM. The scheduler then (1) fires it
    /// SOLO (never co-batched), (2) skips `build_batch_request` /
    /// `append_request_with_options` (which would re-fold the B-lane geometry
    /// from a single `physical_page_ids`), and (3) routes the WHOLE rich response
    /// to the single completion (bypassing the per-row `num_requests ==
    /// requests.len()` split). `physical_page_ids` is then the union of all
    /// lanes' pages, carried for KV-txn / ref tracking only.
    prebuilt: bool,
}

enum Completion {
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

#[derive(Debug, Clone, Copy, Default)]
struct RequestCapacityUsage {
    forward_tokens: usize,
    page_refs: usize,
    logit_rows: usize,
    prob_rows: usize,
    sampler_rows: usize,
    logprob_labels: usize,
    user_custom_mask_bytes: usize,
    spec_custom_mask_bytes: usize,
    has_spec_drafts: bool,
    has_dense_logit_requirement: bool,
    has_prob_sampling: bool,
    is_single_token_decode: bool,
    all_samplers_token: bool,
}

fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    let input_tokens = req.request.token_ids.len();
    let spec_tokens = req.request.spec_token_ids.len();
    let forward_tokens = input_tokens.saturating_add(spec_tokens);
    let mut sampler_rows = req.request.sampling_indices.len();
    if spec_tokens > 0 {
        sampler_rows = sampler_rows.saturating_add(spec_tokens.saturating_add(1));
    }
    let mut all_samplers_token = true;
    let mut has_prob_sampling = false;
    for i in 0..req.request.n_samplers() {
        let sampler = req.request.sampler_at(i).expect("slot in range");
        if !is_token_sampler(&sampler) {
            all_samplers_token = false;
        }
        if sampler_needs_prob_rows(&sampler) {
            has_prob_sampling = true;
        }
    }
    let has_dense_logit_requirement = req.request.has_user_mask
        || !req.request.logit_masks.is_empty()
        || spec_tokens > 0
        || !all_samplers_token;
    let is_single_token_decode = input_tokens == 1
        && spec_tokens == 0
        && req.request.single_token_mode
        && !req.request.has_user_mask;
    let page_refs = req.physical_page_ids.len();
    let spec_custom_mask_bytes =
        packed_mask_bytes(forward_tokens, page_refs, req.last_page_len, page_size);
    let user_custom_mask_bytes = if req.request.has_user_mask && input_tokens > 1 {
        packed_mask_bytes(input_tokens, page_refs, req.last_page_len, page_size)
    } else {
        0
    };

    RequestCapacityUsage {
        forward_tokens,
        page_refs,
        logit_rows: 0,
        prob_rows: 0,
        sampler_rows,
        logprob_labels: request_logprob_labels(&req.request),
        user_custom_mask_bytes,
        spec_custom_mask_bytes,
        has_spec_drafts: spec_tokens > 0,
        has_dense_logit_requirement,
        has_prob_sampling,
        is_single_token_decode,
        all_samplers_token,
    }
}

fn is_token_sampler(sampler: &pie_driver_abi::Sampler) -> bool {
    matches!(
        sampler,
        pie_driver_abi::Sampler::Multinomial { .. }
            | pie_driver_abi::Sampler::TopK { .. }
            | pie_driver_abi::Sampler::TopP { .. }
            | pie_driver_abi::Sampler::MinP { .. }
            | pie_driver_abi::Sampler::TopKTopP { .. }
    )
}

fn sampler_needs_prob_rows(sampler: &pie_driver_abi::Sampler) -> bool {
    match sampler {
        pie_driver_abi::Sampler::TopK { temperature, k } => *temperature > 0.0 && *k > 0,
        pie_driver_abi::Sampler::TopP { temperature, p } => *temperature > 0.0 && *p < 1.0,
        pie_driver_abi::Sampler::TopKTopP { temperature, k, p } => {
            *temperature > 0.0 && (*k > 0 || *p < 1.0)
        }
        _ => false,
    }
}

fn packed_mask_bytes(
    query_tokens: usize,
    page_refs: usize,
    last_page_len: u32,
    page_size: u32,
) -> usize {
    if query_tokens == 0 || page_refs == 0 || page_size == 0 {
        return 0;
    }
    let kv_len = page_refs
        .saturating_sub(1)
        .saturating_mul(page_size as usize)
        .saturating_add(last_page_len as usize);
    query_tokens.saturating_mul(kv_len).saturating_add(7) / 8
}

fn request_logprob_labels(req: &pie_driver_abi::ForwardRequest) -> usize {
    // Logprob → 1 label, Logprobs → its token_ids count (from the CSR), else 0.
    (0..req.n_samplers())
        .map(|s| match req.sampler_kinds[s] {
            pie_driver_abi::PIE_SAMPLER_LOGPROB => 1,
            pie_driver_abi::PIE_SAMPLER_LOGPROBS => (req.sampler_token_ids_indptr[s + 1]
                - req.sampler_token_ids_indptr[s])
                as usize,
            _ => 0,
        })
        .sum()
}

// =============================================================================
// BatchAccumulator
// =============================================================================

/// Accumulates pending requests into a batch.
///
/// Pure synchronous struct — no async, no channels. Can be tested
/// independently from the scheduling loop.
struct BatchAccumulator {
    requests: Vec<PendingRequest>,
    total_tokens: usize,
    total_pages: usize,
    total_logit_rows: usize,
    total_prob_rows: usize,
    total_sampler_rows: usize,
    total_logprob_labels: usize,
    total_user_custom_mask_bytes: usize,
    total_spec_custom_mask_bytes: usize,
    has_spec_drafts: bool,
    has_dense_logit_requirement: bool,
    has_prob_sampling: bool,
    all_single_token_decode: bool,
    all_samplers_token: bool,
    page_size: u32,
    limits: SchedulerLimits,
}

impl BatchAccumulator {
    fn new(limits: SchedulerLimits, page_size: u32) -> Self {
        Self {
            requests: Vec::new(),
            total_tokens: 0,
            total_pages: 0,
            total_logit_rows: 0,
            total_prob_rows: 0,
            total_sampler_rows: 0,
            total_logprob_labels: 0,
            total_user_custom_mask_bytes: 0,
            total_spec_custom_mask_bytes: 0,
            has_spec_drafts: false,
            has_dense_logit_requirement: false,
            has_prob_sampling: false,
            all_single_token_decode: true,
            all_samplers_token: true,
            page_size,
            limits,
        }
    }

    fn projected_rows(
        &self,
        extra: Option<&RequestCapacityUsage>,
    ) -> (usize, usize, bool, bool, bool) {
        let total_tokens = self
            .total_tokens
            .saturating_add(extra.map(|usage| usage.forward_tokens).unwrap_or(0));
        let total_sampler_rows = self
            .total_sampler_rows
            .saturating_add(extra.map(|usage| usage.sampler_rows).unwrap_or(0));
        let has_dense_logit_requirement = self.has_dense_logit_requirement
            || extra
                .map(|usage| usage.has_dense_logit_requirement)
                .unwrap_or(false);
        let has_prob_sampling =
            self.has_prob_sampling || extra.map(|usage| usage.has_prob_sampling).unwrap_or(false);
        let all_samplers_token =
            self.all_samplers_token && extra.map(|usage| usage.all_samplers_token).unwrap_or(true);
        let all_single_token_decode = self.all_single_token_decode
            && extra
                .map(|usage| usage.is_single_token_decode)
                .unwrap_or(true);
        let compact_logit_rows = !all_single_token_decode
            && !has_dense_logit_requirement
            && !has_prob_sampling
            && all_samplers_token
            && total_sampler_rows > 0
            && total_sampler_rows < total_tokens;
        let logit_rows = if total_sampler_rows == 0 {
            // Pure KV-fill / encode fire (e.g. a multimodal image-token
            // prefill): no token is sampled, so the driver computes no
            // logits. Without this, the fire would project `total_tokens`
            // logit rows and trip `max_logit_rows` for large image spans.
            0
        } else if compact_logit_rows {
            total_sampler_rows
        } else {
            total_tokens
        };
        let prob_rows = if has_prob_sampling { total_tokens } else { 0 };
        (
            logit_rows,
            prob_rows,
            has_dense_logit_requirement,
            has_prob_sampling,
            all_single_token_decode,
        )
    }

    fn push(&mut self, req: PendingRequest) {
        let usage = request_capacity_usage(&req, self.page_size);
        self.push_with(req, usage);
    }

    fn push_with(&mut self, req: PendingRequest, mut usage: RequestCapacityUsage) {
        let (logit_rows, prob_rows, _, _, _) = self.projected_rows(Some(&usage));
        usage.logit_rows = logit_rows;
        usage.prob_rows = prob_rows;
        self.total_tokens = self.total_tokens.saturating_add(usage.forward_tokens);
        self.total_pages = self.total_pages.saturating_add(usage.page_refs);
        self.total_logit_rows = usage.logit_rows;
        self.total_prob_rows = usage.prob_rows;
        self.total_sampler_rows = self.total_sampler_rows.saturating_add(usage.sampler_rows);
        self.total_logprob_labels = self
            .total_logprob_labels
            .saturating_add(usage.logprob_labels);
        self.total_user_custom_mask_bytes = self
            .total_user_custom_mask_bytes
            .saturating_add(usage.user_custom_mask_bytes);
        self.total_spec_custom_mask_bytes = self
            .total_spec_custom_mask_bytes
            .saturating_add(usage.spec_custom_mask_bytes);
        self.has_spec_drafts |= usage.has_spec_drafts;
        self.has_dense_logit_requirement |= usage.has_dense_logit_requirement;
        self.has_prob_sampling |= usage.has_prob_sampling;
        self.all_single_token_decode &= usage.is_single_token_decode;
        self.all_samplers_token &= usage.all_samplers_token;
        self.requests.push(req);
    }

    fn single_request_limit_error(&self, req: &PendingRequest) -> Option<String> {
        let usage = request_capacity_usage(req, self.page_size);
        let (logit_rows, prob_rows, _, _, _) =
            BatchAccumulator::new(self.limits, self.page_size).projected_rows(Some(&usage));
        if usage.forward_tokens > self.limits.max_forward_tokens {
            return Some(format!(
                "forward request has {} forward tokens, exceeding driver limit {}",
                usage.forward_tokens, self.limits.max_forward_tokens
            ));
        }

        if usage.page_refs > self.limits.max_page_refs {
            return Some(format!(
                "forward request has {} page refs, exceeding driver limit {}",
                usage.page_refs, self.limits.max_page_refs
            ));
        }

        if logit_rows > self.limits.max_logit_rows {
            return Some(format!(
                "forward request needs {} logit rows, exceeding driver limit {}",
                logit_rows, self.limits.max_logit_rows
            ));
        }

        if prob_rows > self.limits.max_prob_rows {
            return Some(format!(
                "forward request needs {} probability rows, exceeding driver limit {}",
                prob_rows, self.limits.max_prob_rows
            ));
        }

        if usage.sampler_rows > self.limits.max_sampler_rows {
            return Some(format!(
                "forward request has {} sampler rows, exceeding driver limit {}",
                usage.sampler_rows, self.limits.max_sampler_rows
            ));
        }

        if usage.logprob_labels > self.limits.max_logprob_labels {
            return Some(format!(
                "forward request has {} logprob labels, exceeding driver limit {}",
                usage.logprob_labels, self.limits.max_logprob_labels
            ));
        }

        let custom_mask_bytes = if usage.has_spec_drafts {
            usage.spec_custom_mask_bytes
        } else {
            usage.user_custom_mask_bytes
        };
        if custom_mask_bytes > self.limits.max_custom_mask_bytes {
            return Some(format!(
                "forward request needs {custom_mask_bytes} custom mask bytes, exceeding driver limit {}",
                self.limits.max_custom_mask_bytes
            ));
        }

        if self.limits.max_forward_requests == 0 {
            return Some("driver max forward requests is zero".to_string());
        }

        None
    }

    fn would_exceed(&self, req: &PendingRequest) -> bool {
        if self.requests.is_empty() {
            return false;
        }
        let usage = request_capacity_usage(req, self.page_size);
        self.would_exceed_with(&usage)
    }

    /// Run-ahead token-carryover separation (one-step #6, R10): a candidate
    /// forward `t+1` whose `next_input_producer_links` references the
    /// `pipeline_source_link` of a forward `t` ALREADY in this batch must NOT
    /// co-batch with it. `t+1`'s pre-forward `next-inputs` inject reads `t`'s
    /// sampled token (a *prior* fire's retained buffer); `t` samples
    /// post-forward, so co-batching would read a not-yet-sampled token → `t`
    /// and `t+1` must fire in separate, ordered batches (`t` first). This is
    /// purely the token-carryover data-dependency — not a KV/working-set
    /// concern. Batch-local set-membership on the link ids (`0` = not a source;
    /// links are 1-based). Depth ≤1 makes this a simple membership check; the
    /// deeper-run-ahead path (queue depth > 1) is deferred (thrust-2 open Q5),
    /// and the superseded RA parity-phase formula is not carried forward (F6).
    fn would_depend_on_batch(&self, req: &PendingRequest) -> bool {
        let deps = &req.request.next_input_producer_links;
        if deps.is_empty() || self.requests.is_empty() {
            return false;
        }
        self.requests.iter().any(|in_batch| {
            let source_link = in_batch.request.pipeline_source_link;
            source_link != 0 && deps.contains(&source_link)
        })
    }

    fn would_exceed_with(&self, usage: &RequestCapacityUsage) -> bool {
        if self.requests.is_empty() {
            return false;
        }
        // rs_cache spec-decode (MTP for hybrid GDN models) no longer needs a
        // per-batch cap: the driver runs a frozen verify (committed slot stays
        // at its pre-verify value) and a single batched repair forward over
        // [input | accepted] to advance state. There is no per-request snapshot
        // buffer, so rs-spec batches grow to the normal forward limits below,
        // exactly like non-spec batches.
        let next_has_spec = self.has_spec_drafts || usage.has_spec_drafts;
        let next_custom_mask_bytes = if next_has_spec {
            self.total_spec_custom_mask_bytes
                .saturating_add(usage.spec_custom_mask_bytes)
        } else {
            self.total_user_custom_mask_bytes
                .saturating_add(usage.user_custom_mask_bytes)
        };
        let (next_logit_rows, next_prob_rows, _, _, _) = self.projected_rows(Some(usage));
        self.requests.len() + 1 > self.limits.max_forward_requests
            || self.total_tokens.saturating_add(usage.forward_tokens)
                > self.limits.max_forward_tokens
            || self.total_pages.saturating_add(usage.page_refs) > self.limits.max_page_refs
            || next_logit_rows > self.limits.max_logit_rows
            || next_prob_rows > self.limits.max_prob_rows
            || self.total_sampler_rows.saturating_add(usage.sampler_rows)
                > self.limits.max_sampler_rows
            || self
                .total_logprob_labels
                .saturating_add(usage.logprob_labels)
                > self.limits.max_logprob_labels
            || next_custom_mask_bytes > self.limits.max_custom_mask_bytes
    }

    fn would_exceed_reason(&self, req: &PendingRequest) -> Option<String> {
        if self.requests.is_empty() {
            return None;
        }
        let usage = request_capacity_usage(req, self.page_size);
        let next_has_spec = self.has_spec_drafts || usage.has_spec_drafts;
        let next_custom_mask_bytes = if next_has_spec {
            self.total_spec_custom_mask_bytes
                .saturating_add(usage.spec_custom_mask_bytes)
        } else {
            self.total_user_custom_mask_bytes
                .saturating_add(usage.user_custom_mask_bytes)
        };
        let (next_logit_rows, next_prob_rows, _, _, _) = self.projected_rows(Some(&usage));
        let checks = [
            (
                "requests",
                self.requests.len().saturating_add(1),
                self.limits.max_forward_requests,
            ),
            (
                "tokens",
                self.total_tokens.saturating_add(usage.forward_tokens),
                self.limits.max_forward_tokens,
            ),
            (
                "pages",
                self.total_pages.saturating_add(usage.page_refs),
                self.limits.max_page_refs,
            ),
            ("logit_rows", next_logit_rows, self.limits.max_logit_rows),
            ("prob_rows", next_prob_rows, self.limits.max_prob_rows),
            (
                "sampler_rows",
                self.total_sampler_rows.saturating_add(usage.sampler_rows),
                self.limits.max_sampler_rows,
            ),
            (
                "logprob_labels",
                self.total_logprob_labels
                    .saturating_add(usage.logprob_labels),
                self.limits.max_logprob_labels,
            ),
            (
                "custom_mask_bytes",
                next_custom_mask_bytes,
                self.limits.max_custom_mask_bytes,
            ),
        ];
        checks
            .into_iter()
            .find(|(_, have, limit)| have > limit)
            .map(|(name, have, limit)| {
                format!(
                    "{name} {have}>{limit} pending_tokens={} pending_pages={} pending_sampler_rows={} pending_has_spec={} pending_dense={} pending_prob={}",
                    usage.forward_tokens,
                    usage.page_refs,
                    usage.sampler_rows,
                    usage.has_spec_drafts,
                    usage.has_dense_logit_requirement,
                    usage.has_prob_sampling,
                )
            })
    }

    fn is_full(&self) -> bool {
        let active_custom_mask_bytes = if self.has_spec_drafts {
            self.total_spec_custom_mask_bytes
        } else {
            self.total_user_custom_mask_bytes
        };
        self.requests.len() >= self.limits.max_forward_requests
            // rs-spec batches (frozen verify + batched repair) carry no
            // per-request buffer, so they fill to the normal forward limits
            // like any other batch — no rs-spec-specific early fire.
            || self.total_tokens >= self.limits.max_forward_tokens
            || self.total_pages >= self.limits.max_page_refs
            || self.total_logit_rows >= self.limits.max_logit_rows
            || self.total_prob_rows >= self.limits.max_prob_rows
            || self.total_sampler_rows >= self.limits.max_sampler_rows
            || self.total_logprob_labels >= self.limits.max_logprob_labels
            || active_custom_mask_bytes >= self.limits.max_custom_mask_bytes
    }

    fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// True if any accumulated request is a G2 prebuilt-passthrough fire (a
    /// complete pre-assembled multi-lane batch that must fire solo). Forces a
    /// batch boundary so a prebuilt fire is never co-batched.
    fn has_prebuilt(&self) -> bool {
        self.requests.iter().any(|r| r.prebuilt)
    }

    fn len(&self) -> usize {
        self.requests.len()
    }

    fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    fn should_prefill_coalesce(&self) -> bool {
        !self.has_spec_drafts && self.total_tokens > self.requests.len()
    }

    fn take(&mut self) -> Vec<PendingRequest> {
        self.total_tokens = 0;
        self.total_pages = 0;
        self.total_logit_rows = 0;
        self.total_prob_rows = 0;
        self.total_sampler_rows = 0;
        self.total_logprob_labels = 0;
        self.total_user_custom_mask_bytes = 0;
        self.total_spec_custom_mask_bytes = 0;
        self.has_spec_drafts = false;
        self.has_dense_logit_requirement = false;
        self.has_prob_sampling = false;
        self.all_single_token_decode = true;
        self.all_samplers_token = true;
        std::mem::take(&mut self.requests)
    }
}

fn prepare_pending_for_batch(
    batch: &BatchAccumulator,
    pending: PendingRequest,
) -> Option<PendingRequest> {
    prepare_pending_with_usage(batch, pending).map(|(p, _)| p)
}

/// Same as `prepare_pending_for_batch` but also returns the computed
/// `RequestCapacityUsage`, avoiding a recompute when the caller will
/// immediately consult `would_exceed_with` + `push_with`.
///
/// The pure-decode fast path skips `maybe_start_chunking` and
/// `single_request_limit_error`: for `single_token_mode` requests with
/// 1 token, no spec drafts, no user mask, and no logit masks, both are
/// no-ops (chunk_size is never reached; per-request limits hold trivially
/// given the BatchAccumulator's `would_exceed_with` check will still gate
/// page_refs / sampler_rows etc. when batching).
fn prepare_pending_with_usage(
    batch: &BatchAccumulator,
    pending: PendingRequest,
) -> Option<(PendingRequest, RequestCapacityUsage)> {
    if is_pure_decode_pending(&pending) {
        let usage = request_capacity_usage(&pending, batch.page_size);
        let limits = batch.limits;
        // Fields that COULD still trip the single-request limit for decode:
        // page_refs (long-context decode) and logprob_labels. Token/sampler/
        // mask limits hold trivially for 1-token single_token_mode requests.
        if usage.page_refs <= limits.max_page_refs
            && usage.logprob_labels <= limits.max_logprob_labels
            && limits.max_forward_requests > 0
        {
            return Some((pending, usage));
        }
        // Fall through to the slow path so the proper error message is
        // surfaced via `single_request_limit_error`.
    }
    let pending = match pending.maybe_start_chunking(batch.limits, batch.page_size) {
        Ok(pending) => pending,
        Err((pending, msg)) => {
            pending.send_error(msg);
            return None;
        }
    };
    if let Some(msg) = batch.single_request_limit_error(&pending) {
        pending.send_error(msg);
        return None;
    }
    let usage = request_capacity_usage(&pending, batch.page_size);
    Some((pending, usage))
}

#[inline]
fn is_pure_decode_pending(p: &PendingRequest) -> bool {
    matches!(&p.completion, Completion::Direct(_)) && p.request.token_ids.len() == 1
        && p.request.spec_token_ids.is_empty()
        && p.request.single_token_mode
        && !p.request.has_user_mask
        && p.request.logit_masks.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max_requests: usize, max_tokens: usize, max_pages: usize) -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: max_requests,
            max_forward_tokens: max_tokens,
            max_page_refs: max_pages,
            max_logit_rows: usize::MAX,
            max_prob_rows: usize::MAX,
            max_sampler_rows: usize::MAX,
            max_custom_mask_bytes: usize::MAX,
            max_logprob_labels: usize::MAX,
        }
    }

    fn pending(tokens: usize, page_refs: usize) -> PendingRequest {
        let (tx, _rx) = oneshot::channel();
        PendingRequest::direct(
            pie_driver_abi::ForwardRequest {
                token_ids: vec![0; tokens],
                ..Default::default()
            },
            tx,
            vec![0; page_refs],
            1,
            Vec::new(),
            None,
            0,
        )
    }

    /// The FirePolicy wrapper routes membership + the wave decision to
    /// WaitAllPolicy (concrete), and delegates the legacy path to the trait.
    #[test]
    fn firepolicy_waitall_routes_membership_and_decision() {
        use crate::inference::adaptive_policy::WaitAllPolicy;
        let now = Instant::now();
        let p1 = crate::process::ProcessId::from_u128(1);
        let p2 = crate::process::ProcessId::from_u128(2);

        let mut fp = FirePolicy::WaitAll(WaitAllPolicy::new(64, None));
        // Arrivals route to `on_pipeline_request` → join the wait-set.
        fp.on_arrival(&[], Some(p1), now);
        fp.on_arrival(&[], Some(p2), now);
        assert_eq!(
            fp.distinct_program_count(),
            2,
            "wrapper surfaces active_pipelines() for WaitAll"
        );
        // The decision is a wave outcome; no pipeline has missed yet.
        assert!(matches!(
            fp.decide(2, now),
            FireOutcome::Fire { .. } | FireOutcome::Wait(_)
        ));
        assert!(fp.take_terminate_candidates().is_empty());

        // Legacy delegates to the trait — no pipeline tracking, no candidates.
        let mut legacy = FirePolicy::Legacy(Box::new(RunAheadPolicy::new(64)));
        legacy.on_arrival(&[], Some(p1), now);
        assert!(legacy.take_terminate_candidates().is_empty());
    }

    /// The bounded tombstone maps a departed pid's stale request to untracked
    /// (`None`) and FIFO-evicts past capacity so a long-lived scheduler doesn't
    /// grow the set unboundedly.
    #[test]
    fn tombstones_filter_and_bounded_eviction() {
        let p = |n| crate::process::ProcessId::from_u128(n);
        let mut t = Tombstones::new(2);
        t.insert(p(1));
        assert_eq!(t.filter(Some(p(1))), None, "tombstoned → untracked");
        assert_eq!(t.filter(Some(p(9))), Some(p(9)), "live pid passes through");
        assert_eq!(t.filter(None), None);
        // Insert past cap (2) → the oldest (p1) is evicted.
        t.insert(p(2));
        t.insert(p(3));
        assert_eq!(t.filter(Some(p(1))), Some(p(1)), "p1 evicted → no longer tombstoned");
        assert_eq!(t.filter(Some(p(2))), None);
        assert_eq!(t.filter(Some(p(3))), None);
    }

    /// Phase-1 sustained-hang fix: `Join` decays a tombstone so a restored
    /// pipeline is trackable again (`filter` stops mapping it to `None`), and it
    /// frees a FIFO slot (a decayed pid no longer counts toward the cap).
    #[test]
    fn tombstones_remove_decays_on_join() {
        let p = |n| crate::process::ProcessId::from_u128(n);
        let mut t = Tombstones::new(4);
        t.insert(p(1));
        t.insert(p(2));
        assert_eq!(t.filter(Some(p(1))), None, "tombstoned");
        // Join p1 → decayed → trackable again.
        t.remove(p(1));
        assert_eq!(t.filter(Some(p(1))), Some(p(1)), "decayed → trackable");
        assert_eq!(t.filter(Some(p(2))), None, "p2 still tombstoned");
        // Removing a non-tombstoned pid is a no-op.
        t.remove(p(9));
        assert_eq!(t.filter(Some(p(2))), None);
        // The decayed slot is freed: re-inserting p1 then evicting past cap
        // behaves normally (no phantom slot leak).
        t.insert(p(1));
        assert_eq!(t.filter(Some(p(1))), None, "re-tombstoned after re-insert");
    }

    /// guru #2 / bravo's depth-k carrier invariant: the force-fire gate fires a
    /// stashed `next_pending` (prebuilt/capacity split) ONLY below the cap, so a
    /// force-fire can never push depth past `max_in_flight`. At/over the cap it
    /// yields to the policy (whose `Wait` is `next_pending`-guarded), so the stash
    /// survives until a completion frees a slot. Scales with the configured cap
    /// (the depth-k carrier raises it via `PIE_SCHED_MAX_IN_FLIGHT`).
    #[test]
    fn force_fire_respects_in_flight_cap() {
        // Default cap = 2 (compute + one prefetched).
        assert!(force_fire_ready(true, 0, 2), "stash + idle → force-fire");
        assert!(
            force_fire_ready(true, 1, 2),
            "stash + 1 in flight (< cap) → force-fire"
        );
        assert!(
            !force_fire_ready(true, 2, 2),
            "stash AT cap → yield to policy (Wait), never over-fire"
        );
        assert!(
            !force_fire_ready(true, 3, 2),
            "stash OVER cap → yield to policy"
        );
        assert!(
            !force_fire_ready(false, 0, 2),
            "no stash → policy decides (gate inert)"
        );
        // Depth-k carrier: the gate scales with the raised cap.
        assert!(
            force_fire_ready(true, 7, 8),
            "stash below raised cap=8 → force-fire"
        );
        assert!(
            !force_fire_ready(true, 8, 8),
            "stash at raised cap=8 → yield"
        );
    }

    /// D1 async-fire liveness (guru surface-b): a `FireCompletionGuard` dropped
    /// WITHOUT `complete()` — the panic / early-exit path — still sends an
    /// `accounting_only` fallback so the run loop pops the in-flight FIFO and the
    /// policy never Waits at a phantom cap.
    #[test]
    fn fire_completion_guard_fallback_on_bare_drop() {
        let (tx, rx) = crossbeam::channel::unbounded::<FireCompletion>();
        {
            let _g = FireCompletionGuard::new(tx, Duration::from_micros(5), 0);
        }
        let c = rx.try_recv().expect("guard must send on drop");
        assert!(c.accounting_only, "bare-drop fallback is accounting_only");
        assert_eq!(c.forward_latency, Duration::ZERO);
        assert_eq!(c.submission_latency, Duration::from_micros(5));
        assert!(rx.try_recv().is_err(), "exactly one send (Drop is the sole site)");
    }

    /// The success path: `complete()` arms the measured latency; Drop sends a
    /// real (non-`accounting_only`) completion that feeds the timing EWMAs.
    #[test]
    fn fire_completion_guard_real_on_complete() {
        let (tx, rx) = crossbeam::channel::unbounded::<FireCompletion>();
        {
            let mut g = FireCompletionGuard::new(tx, Duration::from_micros(3), 0);
            g.complete(Duration::from_micros(42));
        }
        let c = rx.try_recv().expect("guard must send on drop");
        assert!(!c.accounting_only);
        assert_eq!(c.forward_latency, Duration::from_micros(42));
        assert_eq!(c.submission_latency, Duration::from_micros(3));
    }

    /// Drop runs on panic unwind: a panic BEFORE `complete()` sends the fallback;
    /// a panic AFTER `complete()` still sends the REAL sample (the fire retired
    /// on-device — only dispatch failed — so the EWMA sample is valid).
    #[test]
    fn fire_completion_guard_sends_on_panic_unwind() {
        let (tx, rx) = crossbeam::channel::unbounded::<FireCompletion>();
        let pre = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = FireCompletionGuard::new(tx.clone(), Duration::from_micros(1), 0);
            panic!("handle.wait() blew up before complete()");
        }));
        assert!(pre.is_err());
        assert!(
            rx.try_recv().expect("Drop runs on unwind").accounting_only,
            "panic before complete → accounting_only fallback"
        );

        let post = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g = FireCompletionGuard::new(tx, Duration::from_micros(1), 0);
            g.complete(Duration::from_micros(7));
            panic!("dispatch blew up after the fire retired");
        }));
        assert!(post.is_err());
        let c = rx.try_recv().expect("Drop runs on unwind");
        assert!(!c.accounting_only, "post-complete panic keeps the real sample");
        assert_eq!(c.forward_latency, Duration::from_micros(7));
    }

    /// The shared drain always pops the in-flight FIFO (both flavors) but only a
    /// real completion feeds the policy EWMAs; hitting 0 marks device-idle.
    #[test]
    fn drain_completion_pops_fifo_and_marks_idle() {
        use std::collections::{HashMap, HashSet};
        let mut policy = FirePolicy::Legacy(Box::new(RunAheadPolicy::new(64)));
        let mut in_flight = 2usize;
        let mut idle: Option<u64> = None;
        let mut iff: HashMap<u64, Vec<crate::process::ProcessId>> = HashMap::new();
        let mut rel: HashSet<u64> = HashSet::new();
        drain_completion(
            &mut policy,
            &FireCompletion {
                forward_latency: Duration::ZERO,
                submission_latency: Duration::ZERO,
                accounting_only: true,
                fire_id: 0,
            },
            &mut in_flight,
            &mut idle,
            &mut iff,
            &mut rel,
        );
        assert_eq!(in_flight, 1, "accounting_only still pops the FIFO");
        assert!(idle.is_none(), "device not idle while a fire is in flight");
        drain_completion(
            &mut policy,
            &FireCompletion {
                forward_latency: Duration::from_micros(10),
                submission_latency: Duration::from_micros(2),
                accounting_only: false,
                fire_id: 1,
            },
            &mut in_flight,
            &mut idle,
            &mut iff,
            &mut rel,
        );
        assert_eq!(in_flight, 0);
        assert!(idle.is_some(), "device-idle stamped when in_flight hits 0");
    }

    /// BAR-2 per-lane in-flight release: releasing a demoted lane's stuck fire
    /// frees ONE cap slot per fire (drops `in_flight`), and the released fire's
    /// LATER completion is a NO-OP on the counters — no double-decrement (which
    /// would under-read the cap → over-fire → D−1 WAR-ring overrun).
    #[test]
    fn release_lane_in_flight_then_completion_is_no_op() {
        use std::collections::{HashMap, HashSet};
        let p1 = crate::process::ProcessId::from_u128(1);
        let p2 = crate::process::ProcessId::from_u128(2);
        // WaitAll policy at cap: two in-flight fires (in_flight=2).
        let mut policy = FirePolicy::WaitAll(
            crate::inference::adaptive_policy::WaitAllPolicy::new(64, None),
        );
        policy.on_fired(1);
        policy.on_fired(1);
        let mut in_flight = 2usize;
        // fire 10 = a batch with p1 (+ untracked p2 co-batched); fire 11 = p2 only.
        let mut iff: HashMap<u64, Vec<crate::process::ProcessId>> = HashMap::new();
        iff.insert(10, vec![p1, p2]);
        iff.insert(11, vec![p2]);
        let mut rel: HashSet<u64> = HashSet::new();

        // Release p1's stuck fire (10) → one cap slot freed, recorded.
        let n = release_lane_in_flight(p1, &mut policy, &mut in_flight, &iff, &mut rel);
        assert_eq!(n, 1, "one fire contained p1");
        assert_eq!(in_flight, 1, "cap slot freed at release");
        assert!(rel.contains(&10));

        // The released fire (10) eventually completes → NO-OP: in_flight unchanged.
        let mut idle: Option<u64> = None;
        drain_completion(
            &mut policy,
            &FireCompletion {
                forward_latency: Duration::from_micros(9),
                submission_latency: Duration::ZERO,
                accounting_only: false,
                fire_id: 10,
            },
            &mut in_flight,
            &mut idle,
            &mut iff,
            &mut rel,
        );
        assert_eq!(in_flight, 1, "released fire's completion is a no-op — no double-decrement");
        assert!(!rel.contains(&10), "released marker consumed");
        assert!(!iff.contains_key(&10), "tracking cleared");

        // A NON-released fire (11) completes normally → decrements.
        drain_completion(
            &mut policy,
            &FireCompletion {
                forward_latency: Duration::from_micros(9),
                submission_latency: Duration::ZERO,
                accounting_only: false,
                fire_id: 11,
            },
            &mut in_flight,
            &mut idle,
            &mut iff,
            &mut rel,
        );
        assert_eq!(in_flight, 0, "normal completion decrements");
    }

    /// #14 / BAR-2 CAP-SATURATION LIVENESS GATE — the deterministic in-repo
    /// regression guard for the multi-lane cap-saturation hang that
    /// `release_lane_in_flight` defends against (release-before-terminate for
    /// every demoted lane). Empirical confirm stays charlie's FLEET=8×b500.
    ///
    /// The hang, at the decision level: a demoted lane holds a GENUINELY-STUCK
    /// fire (an undelivered completion — e.g. the D1-delivery mechanism under
    /// the live #17-tail investigation), so its `FireCompletion` never arrives →
    /// `on_complete` never runs → `in_flight` stays pinned at `max_in_flight()`.
    /// `decide_wave_at`'s first clause (`in_flight >= cap ⇒ Wait`) then holds
    /// FOREVER — independent of `now`, a true deadlock, not a straggler wait. No
    /// wave can fire ⇒ the whole fleet stalls (charlie's 1609-active /
    /// requests=1 / no-240s-convergence signature).
    ///
    /// The defense: when the miss-limit demotes the stuck lane, the run loop
    /// calls `release_lane_in_flight` for it BEFORE terminating, freeing the
    /// stuck fire's cap slot ⇒ the very next `decide` Fires a waiting fleet lane.
    /// WITHOUT the release the cap can never drop (nothing decrements a stuck
    /// fire) — the pre-release poll loop below Waits at every `now`. The stuck
    /// fire's LATER completion is a no-op (WAR-safe: it never D2H'd a
    /// `pi.sampled`, so it holds no ring slot — cap invariant preserved).
    #[test]
    fn cap_saturation_stuck_fire_hangs_without_release_greens_with() {
        use std::collections::{HashMap, HashSet};
        let cap = crate::inference::adaptive_policy::max_in_flight();
        assert!(cap >= 1, "cap must be positive");

        // `stuck` = the demoted lane whose in-flight fire never retires; `ready`
        // = a fleet lane with a dense, ready pass that the pinned cap blocks.
        let stuck = crate::process::ProcessId::from_u128(1);
        let ready = crate::process::ProcessId::from_u128(2);

        let mut policy = FirePolicy::WaitAll(
            crate::inference::adaptive_policy::WaitAllPolicy::new(64, None),
        );
        let mut in_flight = 0usize;
        let mut iff: HashMap<u64, Vec<crate::process::ProcessId>> = HashMap::new();
        let mut rel: HashSet<u64> = HashSet::new();

        // Saturate the cap with `cap` stuck fires — none will ever complete.
        // Fire 100 belongs to the `stuck` lane; 101.. are other stuck lanes.
        for i in 0..cap {
            policy.on_fired(1);
            in_flight += 1;
            let fire_id = 100 + i as u64;
            let lane = if i == 0 { stuck } else { crate::process::ProcessId::from_u128(200 + i as u128) };
            iff.insert(fire_id, vec![lane]);
        }
        assert_eq!(in_flight, cap, "in-flight pinned at the cap by stuck fires");

        // A ready fleet pass enters the wave: dense + ready, yet the cap gate
        // must still block it while the stuck fires pin `in_flight`.
        let t0 = Instant::now();
        policy.on_arrival(&[], Some(ready), t0);

        // WOULD-HANG: the cap gate returns Wait at EVERY `now` — advancing the
        // clock arbitrarily far never lets the wave fire (proving it is the cap
        // deadlock, not a bounded straggler deadline). This is exactly the
        // pre-fix behavior — nothing can drop the cap, so this Wait is forever.
        for secs in [0u64, 1, 10, 100, 3600] {
            let now = t0 + Duration::from_secs(secs);
            assert!(
                matches!(policy.decide(1, now), FireOutcome::Wait(_)),
                "cap-saturated: a ready batch must NOT fire while the cap is pinned (now=+{secs}s)"
            );
        }

        // THE DEFENSE (release-before-terminate): the demote handler releases
        // the stuck lane's cap slot, freeing exactly its one stuck fire.
        let freed = release_lane_in_flight(stuck, &mut policy, &mut in_flight, &iff, &mut rel);
        assert_eq!(freed, 1, "exactly the stuck lane's one fire was released");
        assert_eq!(in_flight, cap - 1, "the freed slot drops in-flight below the cap");
        assert!(rel.contains(&100), "the released fire is recorded for its no-op completion");

        // GREENS: with the cap unpinned, the ready fleet pass fires immediately —
        // the fleet re-drives, `drain_own_fires` unblocks.
        assert!(
            matches!(policy.decide(1, t0), FireOutcome::Fire { .. }),
            "post-release: a waiting fleet pass fires (liveness restored)"
        );

        // WAR-SAFETY: the stuck fire eventually completes (driver un-sticks) →
        // NO-OP on the counters, no double-decrement that would under-read the
        // cap and over-fire (the D−1 `pi.sampled` ring invariant).
        let mut idle: Option<u64> = None;
        drain_completion(
            &mut policy,
            &FireCompletion {
                forward_latency: Duration::from_micros(9),
                submission_latency: Duration::ZERO,
                accounting_only: false,
                fire_id: 100,
            },
            &mut in_flight,
            &mut idle,
            &mut iff,
            &mut rel,
        );
        assert_eq!(in_flight, cap - 1, "late completion of a released fire is a no-op");
        assert!(!rel.contains(&100), "released marker consumed exactly once");
    }

    fn with_spec(mut req: PendingRequest, spec_tokens: usize) -> PendingRequest {
        req.request.spec_token_ids = vec![1; spec_tokens];
        req.request.spec_position_ids = vec![1; spec_tokens];
        req.request.spec_indptr = vec![0, spec_tokens as u32];
        req
    }

    fn with_sampler_rows(mut req: PendingRequest, sampler_rows: usize) -> PendingRequest {
        req.request.sampling_indices = vec![0; sampler_rows];
        req
    }

    fn with_samplers(
        mut req: PendingRequest,
        indices: Vec<u32>,
        samplers: Vec<pie_driver_abi::Sampler>,
    ) -> PendingRequest {
        req.request.sampling_indices = indices;
        req.request.set_samplers(&samplers);
        req
    }

    fn with_pipeline_source(mut req: PendingRequest, link: u32) -> PendingRequest {
        req.request.pipeline_source_link = link;
        req
    }

    fn with_next_input(mut req: PendingRequest, producer_link: u32) -> PendingRequest {
        req.request.next_input_producer_links = vec![producer_link];
        req
    }

    fn with_pid(mut req: PendingRequest, pid: crate::process::ProcessId) -> PendingRequest {
        req.pipeline_id = Some(pid);
        req
    }

    /// N+k deep pre-submission: once a pipeline has a stashed predecessor, its
    /// next request is the next chain link (its producer is stashed, not in the
    /// batch — `would_depend_on_batch` can't see it) → stash it in FIFO order so
    /// it never co-batches with the head. Off WaitAll / untracked → passthrough.
    #[test]
    fn stash_chain_continuation_orders_the_pipeline_chain() {
        use std::collections::{HashMap, VecDeque};
        let p = crate::process::ProcessId::from_u128(1);
        let p2 = crate::process::ProcessId::from_u128(2);
        let mut stash: HashMap<crate::process::ProcessId, VecDeque<PendingRequest>> =
            HashMap::new();

        // Off WaitAll → passthrough even with a stash present.
        stash.entry(p).or_default().push_back(with_pid(pending(1, 1), p));
        assert!(stash_chain_continuation(&mut stash, false, with_pid(pending(1, 1), p)).is_some());
        stash.clear();

        // Untracked (None pid) → passthrough (legacy force-fire owns those).
        assert!(stash_chain_continuation(&mut stash, true, pending(1, 1)).is_some());

        // Empty stash for p → passthrough (the head joins the wave; the first
        // dependent is stashed by the would_depend path, not this guard).
        assert!(stash_chain_continuation(&mut stash, true, with_pid(pending(1, 1), p)).is_some());
        assert!(stash.get(&p).is_none());

        // Seed N+1 (the would_depend stash), then N+2/N+3 chain-order behind it.
        stash.entry(p).or_default().push_back(with_pid(pending(1, 1), p));
        assert!(stash_chain_continuation(&mut stash, true, with_pid(pending(1, 1), p)).is_none());
        assert!(stash_chain_continuation(&mut stash, true, with_pid(pending(1, 1), p)).is_none());
        assert_eq!(stash.get(&p).map(|q| q.len()), Some(3), "N+1..N+3 FIFO");

        // A different pipeline with no stash is independent → passthrough.
        assert!(stash_chain_continuation(&mut stash, true, with_pid(pending(1, 1), p2)).is_some());
    }

    /// A runaway pre-submission chain fails loud at the ceiling rather than
    /// growing the stash unbounded (memory leak).
    #[test]
    fn stash_chain_continuation_fails_loud_past_ceiling() {
        use std::collections::{HashMap, VecDeque};
        let p = crate::process::ProcessId::from_u128(1);
        let mut stash: HashMap<crate::process::ProcessId, VecDeque<PendingRequest>> =
            HashMap::new();
        {
            let q = stash.entry(p).or_default();
            for _ in 0..MAX_DEP_STASH_PER_PIPELINE {
                q.push_back(with_pid(pending(1, 1), p));
            }
        }
        assert!(stash_chain_continuation(&mut stash, true, with_pid(pending(1, 1), p)).is_none());
        assert_eq!(
            stash.get(&p).map(|q| q.len()),
            Some(MAX_DEP_STASH_PER_PIPELINE),
            "no growth past the ceiling",
        );
    }

    /// Phase-2 (carrier × contention): a departed pipeline's stashed run-ahead
    /// chain is DROPPED + each stashed forward ERRORED (releasing its staged
    /// txn/pins so contention can reclaim), while other pipelines are untouched.
    /// This halts the perpetual-pin promotion tail (a preempted victim would
    /// otherwise keep firing its stash untracked until it drains).
    #[test]
    fn clear_left_pipeline_stash_drops_and_errors() {
        use std::collections::{HashMap, VecDeque};
        let p = crate::process::ProcessId::from_u128(1);
        let p2 = crate::process::ProcessId::from_u128(2);
        let mut stash: HashMap<crate::process::ProcessId, VecDeque<PendingRequest>> =
            HashMap::new();

        let make = |pid| {
            let (tx, rx) = oneshot::channel();
            let req = PendingRequest::direct(
                pie_driver_abi::ForwardRequest {
                    token_ids: vec![0; 1],
                    ..Default::default()
                },
                tx,
                vec![0; 1],
                1,
                Vec::new(),
                Some(pid),
                0,
            );
            (req, rx)
        };

        let mut rxs = Vec::new();
        for _ in 0..3 {
            let (req, rx) = make(p);
            stash.entry(p).or_default().push_back(req);
            rxs.push(rx);
        }
        let (req2, mut rx2) = make(p2);
        stash.entry(p2).or_default().push_back(req2);

        clear_left_pipeline_stash(&mut stash, p);

        assert!(stash.get(&p).is_none(), "left pid's stash dropped");
        assert_eq!(
            stash.get(&p2).map(|q| q.len()),
            Some(1),
            "an unrelated pipeline's stash is untouched"
        );
        for mut rx in rxs {
            assert!(
                matches!(rx.try_recv(), Ok(Err(_))),
                "each stashed forward received an abort error (txn/pins release)"
            );
        }
        assert!(
            rx2.try_recv().is_err(),
            "the untouched pipeline's forward is still pending (not resolved)"
        );
    }

    /// BAR 1 / Piece 4: `promote_dep_stash` advances each NON-suspended
    /// pipeline's chain by one FIFO link into the batch and skips SUSPENDED
    /// (parked) pipelines — the shared engine of both the post-fire promotion and
    /// the ignition pump.
    #[test]
    fn promote_dep_stash_advances_nonsuspended_fifo() {
        use crate::inference::adaptive_policy::WaitAllPolicy;
        use std::collections::{HashMap, HashSet, VecDeque};
        let p1 = crate::process::ProcessId::from_u128(1);
        let p2 = crate::process::ProcessId::from_u128(2);
        let mut stash: HashMap<crate::process::ProcessId, VecDeque<PendingRequest>> =
            HashMap::new();
        // p1: a 2-deep chain; p2: 1-deep but SUSPENDED (parked).
        stash.entry(p1).or_default().push_back(with_pid(pending(1, 1), p1));
        stash.entry(p1).or_default().push_back(with_pid(pending(1, 1), p1));
        stash.entry(p2).or_default().push_back(with_pid(pending(1, 1), p2));
        let mut suspended: HashSet<crate::process::ProcessId> = HashSet::new();
        suspended.insert(p2);
        let mut policy = FirePolicy::WaitAll(WaitAllPolicy::new(64, None));
        let tombstones = Tombstones::new(16);
        let mut batch = BatchAccumulator::new(limits(8, 100, 100), 16);

        promote_dep_stash(&mut stash, &suspended, &mut policy, &tombstones, &mut batch);
        assert_eq!(batch.len(), 1, "one non-suspended pipeline promoted one link");
        assert_eq!(stash.get(&p1).map(|q| q.len()), Some(1), "p1 chain advanced by one");
        assert_eq!(stash.get(&p2).map(|q| q.len()), Some(1), "p2 parked — not promoted");

        // Un-park p2 (the Join) → the next promotion advances BOTH remaining links.
        suspended.remove(&p2);
        promote_dep_stash(&mut stash, &suspended, &mut policy, &tombstones, &mut batch);
        assert_eq!(batch.len(), 3, "p1's last link + p2's link promoted");
        assert!(stash.is_empty(), "both chains fully drained");
    }

    #[test]
    fn accumulator_separates_run_ahead_carryover_dependency() {
        // One-step run-ahead (R10): a forward `t` samples its token under link
        // L=1; a forward `t+1` that injects L=1 via `next-inputs` MUST NOT
        // co-batch with it (it would read `t`'s not-yet-sampled token). `t+1` is
        // stashed for the next fire; an unrelated/plain request is unaffected.
        let mut batch = BatchAccumulator::new(limits(8, 100, 100), 16);
        batch.push(with_pipeline_source(pending(1, 1), 1));
        assert!(batch.would_depend_on_batch(&with_next_input(pending(1, 1), 1)));
        assert!(!batch.would_depend_on_batch(&with_next_input(pending(1, 1), 2)));
        assert!(!batch.would_depend_on_batch(&pending(1, 1)));
    }

    #[test]
    fn accumulator_splits_by_forward_tokens() {
        let mut batch = BatchAccumulator::new(limits(8, 6, 100), 16);
        batch.push(pending(4, 1));
        assert!(!batch.would_exceed(&pending(2, 1)));
        assert!(batch.would_exceed(&pending(3, 1)));
    }

    #[test]
    fn accumulator_splits_by_forward_requests() {
        let mut batch = BatchAccumulator::new(limits(2, 100, 100), 16);
        batch.push(pending(1, 1));
        assert!(!batch.would_exceed(&pending(1, 1)));
        batch.push(pending(1, 1));
        assert!(batch.is_full());
        assert!(batch.would_exceed(&pending(1, 1)));
    }

    #[test]
    fn accumulator_splits_by_page_refs() {
        let mut batch = BatchAccumulator::new(limits(8, 100, 5), 16);
        batch.push(pending(1, 3));
        assert!(!batch.would_exceed(&pending(1, 2)));
        assert!(batch.would_exceed(&pending(1, 3)));
    }

    #[test]
    fn accumulator_rejects_single_request_over_limit() {
        let batch = BatchAccumulator::new(limits(8, 6, 5), 16);
        assert!(batch.single_request_limit_error(&pending(7, 1)).is_some());
        assert!(batch.single_request_limit_error(&pending(1, 6)).is_some());
        assert!(batch.single_request_limit_error(&pending(6, 5)).is_none());
    }

    #[test]
    fn accumulator_counts_speculative_tokens() {
        let mut batch = BatchAccumulator::new(limits(8, 6, 100), 16);
        batch.push(with_spec(pending(4, 1), 2));
        assert!(batch.is_full());
        assert!(batch.would_exceed(&pending(1, 1)));

        let batch = BatchAccumulator::new(limits(8, 6, 100), 16);
        assert!(
            batch
                .single_request_limit_error(&with_spec(pending(5, 1), 2))
                .is_some()
        );
    }

    #[test]
    fn accumulator_splits_by_sampler_rows() {
        let mut capped = limits(8, 100, 100);
        capped.max_sampler_rows = 3;
        let mut batch = BatchAccumulator::new(capped, 16);
        batch.push(with_sampler_rows(pending(1, 1), 2));
        assert!(!batch.would_exceed(&with_sampler_rows(pending(1, 1), 1)));
        assert!(batch.would_exceed(&with_sampler_rows(pending(1, 1), 2)));
        assert!(
            batch
                .single_request_limit_error(&with_sampler_rows(pending(1, 1), 4))
                .is_some()
        );
    }

    #[test]
    fn accumulator_counts_spec_verification_sampler_rows() {
        let mut capped = limits(8, 100, 100);
        capped.max_sampler_rows = 3;
        let batch = BatchAccumulator::new(capped, 16);
        let req = with_spec(with_sampler_rows(pending(1, 1), 1), 2);
        assert!(batch.single_request_limit_error(&req).is_some());
    }

    #[test]
    fn accumulator_splits_by_custom_mask_bytes() {
        let mut capped = limits(8, 100, 100);
        capped.max_custom_mask_bytes = 31;
        let mut batch = BatchAccumulator::new(capped, 16);

        // 2 query rows x 64 KV positions = 128 bits = 16 bytes.
        let mut user_mask = pending(2, 4);
        user_mask.last_page_len = 16;
        user_mask.request.has_user_mask = true;
        batch.push(user_mask);

        let mut next = pending(2, 4);
        next.last_page_len = 16;
        next.request.has_user_mask = true;
        assert!(batch.would_exceed(&next));
    }

    #[test]
    fn adding_spec_request_counts_existing_requests_for_spec_mask_path() {
        let mut capped = limits(8, 100, 100);
        capped.max_custom_mask_bytes = 31;
        let mut batch = BatchAccumulator::new(capped, 16);

        let mut existing = pending(2, 4);
        existing.last_page_len = 16;
        batch.push(existing);

        let mut spec = with_spec(pending(1, 4), 1);
        spec.last_page_len = 16;
        assert!(batch.would_exceed(&spec));
    }

    #[test]
    fn accumulator_rejects_logprob_label_over_limit() {
        let mut capped = limits(8, 100, 100);
        capped.max_logprob_labels = 2;
        let batch = BatchAccumulator::new(capped, 16);
        let mut req = pending(1, 1);
        req.request.set_samplers(&[pie_driver_abi::Sampler::Logprobs {
            token_ids: vec![1, 2, 3],
        }]);
        assert!(batch.single_request_limit_error(&req).is_some());
    }

    #[test]
    fn accumulator_allows_compact_prefill_logit_rows() {
        let mut capped = limits(8, 8, 100);
        capped.max_logit_rows = 2;
        let batch = BatchAccumulator::new(capped, 16);
        let req = with_samplers(
            pending(4, 1),
            vec![3],
            vec![pie_driver_abi::Sampler::TopP {
                temperature: 0.0,
                p: 1.0,
            }],
        );
        assert!(batch.single_request_limit_error(&req).is_none());
    }

    #[test]
    fn accumulator_splits_by_compact_logit_rows() {
        let mut capped = limits(8, 100, 100);
        capped.max_logit_rows = 2;
        let mut batch = BatchAccumulator::new(capped, 16);
        let req = || {
            with_samplers(
                pending(4, 1),
                vec![3],
                vec![pie_driver_abi::Sampler::TopP {
                    temperature: 0.0,
                    p: 1.0,
                }],
            )
        };
        batch.push(req());
        assert!(!batch.would_exceed(&req()));
        batch.push(req());
        assert!(batch.would_exceed(&req()));
    }

    #[test]
    fn accumulator_rejects_dense_logit_over_limit() {
        let mut capped = limits(8, 100, 100);
        capped.max_logit_rows = 3;
        let batch = BatchAccumulator::new(capped, 16);
        let req = with_samplers(pending(4, 1), vec![3], vec![pie_driver_abi::Sampler::RawLogits]);
        assert!(batch.single_request_limit_error(&req).is_some());
    }

    #[test]
    fn accumulator_rejects_probability_rows_over_limit() {
        let mut capped = limits(8, 100, 100);
        capped.max_logit_rows = 8;
        capped.max_prob_rows = 3;
        let batch = BatchAccumulator::new(capped, 16);
        let req = with_samplers(
            pending(4, 1),
            vec![3],
            vec![pie_driver_abi::Sampler::TopP {
                temperature: 1.0,
                p: 0.9,
            }],
        );
        assert!(batch.single_request_limit_error(&req).is_some());
    }

    #[test]
    fn taking_batch_does_not_drop_stashed_request_shape() {
        let mut batch = BatchAccumulator::new(limits(8, 6, 5), 16);
        batch.push(pending(4, 2));
        let stashed = pending(4, 4);
        assert!(batch.would_exceed(&stashed));
        let fired = batch.take();
        assert_eq!(fired.len(), 1);
        assert!(batch.is_empty());
        assert!(!batch.would_exceed(&stashed));
    }

    /// Capacity limits split the ready set *before* any policy decision
    /// (overview §7.2: the splitter runs before the fire clause). A
    /// homogeneous ready fleet larger than `max_forward_requests` caps the
    /// batch at the limit regardless of what a policy would decide — the
    /// `would_exceed`/`is_full` gate is the splitter, not the policy.
    #[test]
    fn capacity_splits_ready_fleet_before_policy() {
        let mut batch = BatchAccumulator::new(limits(4, 1000, 1000), 16);
        // Admit ready requests until the request-count cap; the 5th must be
        // stashed by the capacity gate, not by any policy.
        for _ in 0..4 {
            let req = pending(1, 1);
            assert!(!batch.would_exceed(&req), "under cap admits");
            batch.push(req);
        }
        assert!(batch.is_full(), "capacity cap reached at max_forward_requests");
        let fifth = pending(1, 1);
        assert!(
            batch.would_exceed(&fifth),
            "over-cap request splits into the next fire before policy.decide()"
        );
        // The over-budget set becomes two fires: [4] then [1].
        let first = batch.take();
        assert_eq!(first.len(), 4);
        assert!(!batch.would_exceed(&fifth), "remainder fits the next batch");
        batch.push(fifth);
        assert_eq!(batch.take().len(), 1);
    }

    /// Dense rebatch drops a run-ahead laggard and lets it rejoin the next
    /// fire. A forward `t+1` whose token-carryover source `t` is in the current
    /// batch is dropped from it (would read `t`'s not-yet-sampled token); after
    /// the batch fires, `t` has sampled, so `t+1` co-batches next round — a
    /// stateless rejoin (R5), no holes left behind.
    #[test]
    fn dense_rebatch_drops_and_rejoins_run_ahead_laggard() {
        let mut batch = BatchAccumulator::new(limits(8, 100, 100), 16);
        batch.push(with_pipeline_source(pending(1, 1), 1)); // forward t (source L=1)
        let t_plus_1 = with_next_input(pending(1, 1), 1); // consumes L=1
        assert!(
            batch.would_depend_on_batch(&t_plus_1),
            "t+1 laggard dropped from t's batch"
        );
        // Fire t's batch; t samples its token.
        let fired = batch.take();
        assert_eq!(fired.len(), 1);
        // t+1 now rejoins: its producer is no longer in-batch.
        assert!(
            !batch.would_depend_on_batch(&t_plus_1),
            "laggard rejoins the next fire after its producer retired"
        );
    }

    /// S2 exit: a re-formed batch may contain consumers whose producers came
    /// from DIFFERENT prior batches. Producer links are global ids retained
    /// across fires; `would_depend_on_batch` only splits a consumer off a
    /// producer present in the SAME batch. Two consumers injecting from
    /// prior-fire links (neither in this batch) co-batch freely with each other.
    #[test]
    fn consumers_from_different_prior_batches_cobatch() {
        let mut batch = BatchAccumulator::new(limits(8, 100, 100), 16);
        // A plain decode (no source link) opens the batch.
        batch.push(pending(1, 1));
        // Consumer A injects from prior-batch producer link 5 (retained from an
        // earlier fire, not present here) → no in-batch dependency.
        let cons_a = with_next_input(pending(1, 1), 5);
        assert!(!batch.would_depend_on_batch(&cons_a));
        batch.push(cons_a);
        // Consumer B injects from a DIFFERENT prior-batch producer link 9.
        let cons_b = with_next_input(pending(1, 1), 9);
        assert!(!batch.would_depend_on_batch(&cons_b));
        batch.push(cons_b);
        assert_eq!(batch.len(), 3, "both cross-prior-batch consumers co-batch");
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

        // Per-driver state
        let mut batch = BatchAccumulator::new(limits, page_size);
        // Fire policy. The legacy run-ahead just-in-time policy (#6) is the
        // default; the PTIR quorum rule (thrust-2 §3, F1–F6) is selected behind
        // the `run-ahead` feature + `PIE_SCHED_POLICY=quorum`. Both fire the next
        // batch without blocking on the in-flight one (depth-1, R10); the quorum
        // rule drops the `fire_at` completion/lead-time estimate (F6) for a pure
        // membership decision. Degrades to greedy when nothing is in flight.
        let mut policy: FirePolicy = if waitall_policy_enabled() {
            FirePolicy::WaitAll(super::adaptive_policy::WaitAllPolicy::new(
                limits.max_forward_requests,
                Some(stats.clone()),
            ))
        } else if quorum_policy_enabled() {
            FirePolicy::Legacy(Box::new(super::adaptive_policy::QuorumPolicy::new(
                limits.max_forward_requests,
                Some(stats.clone()),
            )))
        } else {
            FirePolicy::Legacy(Box::new(RunAheadPolicy::new(limits.max_forward_requests)))
        };
        // The fire is non-blocking: `execute_batch` is split into an ordered
        // enqueue on this thread (fixing driver-inbox order == fire order) plus
        // an off-thread wait, so building/enqueuing the next batch overlaps the
        // in-flight GPU. The in-flight depth is bounded by the policy's FIFO
        // cap (one-step run-ahead, R10).

        // Channel for batch completion feedback to the policy. Carries the
        // off-thread forward (GPU) latency + the on-thread submission latency.
        let (latency_tx, latency_rx) = crossbeam::channel::unbounded::<FireCompletion>();
        let mut next_pending: Option<PendingRequest> = None;
        // Host-side device-idle bubble proxy (thrust-2 S5 probe scaffolding). The
        // loop tracks how many batches are outstanding on the driver and, when
        // that drops to zero, the instant the device *appears* idle (a completion
        // was observed with nothing queued behind it). The next fire's enqueue
        // then records `inter_batch_bubble_us = launch − idle_since` — the depth-1
        // steady state keeps a batch queued so this stays ~0 (F1 enqueue-ahead).
        //
        // NOTE (GPU window): this is a HOST proxy — `device_idle_since` is stamped
        // when the loop *receives* the completion, which lags the true device
        // retirement by the IPC/response-propagation delay, so it over-counts. The
        // precise bubble-p50 gate needs a driver-emitted device-idle timestamp
        // (a `driver_cuda` probe field carried on the response); this host proxy
        // is the interim signal until that hook lands.
        let mut in_flight_count: usize = 0;
        let mut device_idle_since: Option<u64> = None;

        // M-A1 Stage 2: wait-for-all pipeline-`Leave` channel + tombstone set.
        // Only a waitall run loop registers its sender (so `notify_pipeline_leave`
        // reaches it) and acts on Leaves; the legacy path leaves both inert.
        let waitall_active = matches!(policy, FirePolicy::WaitAll(_));
        let (lifecycle_tx, lifecycle_rx) = crossbeam::channel::unbounded::<LifecycleEvent>();
        if waitall_active {
            if let Ok(mut senders) = lifecycle_senders().lock() {
                senders.push(lifecycle_tx);
            }
        }
        let mut tombstones = Tombstones::new(4096);
        // Piece 4 (① co-batch): per-pipeline dep-stash. Under WaitAll, a
        // `would_depend_on_batch` request (a pipeline's `t+1` whose token source
        // `t` is in the GATHERING wave) is held here per-pipeline instead of
        // force-firing the wave via `next_pending` (the sync-era bypass that
        // collapsed every wave to a singleton — mean_batch=1). The wave keeps
        // gathering; each stash promotes into the NEXT wave post-fire. At
        // depth-2 the stash is ≤1 deep; under bravo's device-resident carrier
        // deep pre-submission it holds the k-deep chain (N+1…N+k), each link
        // gated in FIFO order by `stash_chain_continuation` (once a pipeline has
        // a stashed predecessor, its next request is the next chain link — it
        // must not co-batch with the head). Promotes one link per fire, so at
        // cap=k the chain fires back-to-back. Legacy keeps `next_pending` — its
        // force-fire is correct there.
        let mut dep_stash: std::collections::HashMap<
            ProcessId,
            std::collections::VecDeque<PendingRequest>,
        > = std::collections::HashMap::new();
        // Phase-2 (carrier × contention): pipelines SUSPENDED by contention
        // (`LeaveKind::Suspend`) whose `dep_stash` chain is PARKED — skip-promoted
        // (not fired, not errored) until they rejoin (`LifecycleEvent::Join`),
        // then resumed in chain order. A `Suspend` victim RESUMES its exact
        // passes on restore (transparent, C4), so erroring the stash is wrong.
        let mut suspended: std::collections::HashSet<ProcessId> =
            std::collections::HashSet::new();
        // Host-side device-idle bubble proxy (thrust-2 S5 probe scaffolding). The
        // loop tracks how many batches are outstanding on the driver and, when
        // that drops to zero, the instant the device *appears* idle (a completion
        // was observed with nothing queued behind it). The next fire's enqueue
        // then records `inter_batch_bubble_us = launch − idle_since` — the depth-1
        // steady state keeps a batch queued so this stays ~0 (F1 enqueue-ahead).
        //
        // NOTE (GPU window): this is a HOST proxy — `device_idle_since` is stamped
        // when the loop *receives* the completion, which lags the true device
        // retirement by the IPC/response-propagation delay, so it over-counts. The
        // precise bubble-p50 gate needs a driver-emitted device-idle timestamp
        // (a `driver_cuda` probe field carried on the response); this host proxy
        // is the interim signal until that hook lands.
        let mut in_flight_count: usize = 0;
        let mut device_idle_since: Option<u64> = None;
        // BAR-2 per-lane in-flight release. `in_flight_fires` maps each in-flight
        // fire's unique id → the tracked lane pids in its batch, so a demoted
        // lane's stuck fires can be found + their cap slots RELEASED (via
        // `release_lane_in_flight`) to break a cap-saturation stall. A released
        // fire's id goes into `released_fires` so its later `FireCompletion` is a
        // no-op on the cap counters (no double-decrement → no WAR over-fire).
        let mut next_fire_id: u64 = 0;
        let mut in_flight_fires: std::collections::HashMap<u64, Vec<ProcessId>> =
            std::collections::HashMap::new();
        let mut released_fires: std::collections::HashSet<u64> =
            std::collections::HashSet::new();

        // One-time boot diagnostic (no per-fire perturbation): resolve the
        // accum-hold OnceLock ONCE so a trace run reveals whether the test-only
        // co-batch hold actually engaged (`Some(us)`) or is fire-on-arrival
        // (`None`). Emitting this in the hot accum loop would itself add stderr
        // latency and mask the very arrival-timing it measures (the grammar10
        // heisenbug), so it is stamped here at boot, outside the loop.
        if scheduler_trace_enabled() {
            sched_trace_write(&format!(
                "[pie-sched-trace] driver={} boot accum_hold_us={:?} quorum_policy={}",
                driver_idx,
                scheduler_accum_hold_us(),
                quorum_policy_enabled(),
            ));
        }

        'run_loop: loop {
            // Drain completed batch feedback (non-blocking): GPU latency →
            // on_complete (+ FIFO pop), submission latency → on_submitted.
            while let Ok(c) = latency_rx.try_recv() {
                drain_completion(&mut policy, &c, &mut in_flight_count, &mut device_idle_since, &mut in_flight_fires, &mut released_fires);
            }

            // M-A1 Stage 2: drain pipeline `Leave`s (terminate / cancel / exit /
            // contention-preempt) — drop each from the wait-set so it no longer
            // holds the wave, and tombstone it so a stale queued request can't
            // implicitly re-join it. No-op on the legacy path (channel empty).
            while let Ok(ev) = lifecycle_rx.try_recv() {
                apply_lifecycle_event(
                    ev,
                    &mut policy,
                    &mut tombstones,
                    &mut suspended,
                    &mut dep_stash,
                );
            }

            // Wait for first request if batch is empty. crossbeam's
            // recv() parks via futex — far lower wake latency than
            // tokio's mpsc waker path. Time the wait (once warm) as the
            // steady-state scheduler idle-wait — the round-trip residual of R
            // (dispatch→inferlet resubmit→SERVICE→recv), excluded from the
            // scheduler's own build/decide cost.
            let recv_block_start = Instant::now();
            let was_empty = batch.is_empty();
            let mut first_submitted_at_us = 0u64;
            while batch.is_empty() {
                // BAR 1 ignition: promotion runs only post-fire, so a zero-in-
                // flight idle with the fleet's next fires PARKED in `dep_stash`
                // (a burst-then-idle, or a `Join` that just un-parked them) would
                // freeze forever — the resumed guests RESUME their parked passes,
                // they don't re-submit, so `req_rx` never wakes. Pump a promotion
                // whenever in-flight is idle + a promotable stash exists → the
                // stash re-drives the fleet from the zero-in-flight instant.
                if in_flight_count == 0 {
                    promote_dep_stash(
                        &mut dep_stash,
                        &suspended,
                        &mut policy,
                        &tombstones,
                        &mut batch,
                    );
                    if !batch.is_empty() {
                        break;
                    }
                }
                let pending = if let Some(pending) = next_pending.take() {
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
                    // BAR 1 diagnostic: about to idle-BLOCK. If a non-empty
                    // `dep_stash` exists here, the pump above did NOT re-drive —
                    // log the freeze signature so a trace discriminates charlie's
                    // stuck-fire hypothesis (`in_flight > 0` ⇒ a fire never
                    // completed → the pump can't engage; driver-scope) from a pump
                    // reach/logic bug (`in_flight == 0` + `promotable > 0` ⇒ the
                    // pump SHOULD have fired → my scope) from a legit all-parked
                    // wait (`promotable == 0` ⇒ waiting on a `Join`).
                    if scheduler_trace_enabled() && !dep_stash.is_empty() {
                        let promotable =
                            dep_stash.keys().filter(|p| !suspended.contains(p)).count();
                        sched_trace_write(&format!(
                            "[pie-sched-trace] driver={} idle-block in_flight={} dep_stash={} promotable={} suspended_marked={}",
                            driver_idx,
                            in_flight_count,
                            dep_stash.len(),
                            promotable,
                            suspended.len(),
                        ));
                    }
                    crossbeam::channel::select! {
                        recv(req_rx) -> msg => match msg {
                            Ok(p) => p,
                            Err(_) => break 'run_loop,
                        },
                        recv(latency_rx) -> completion => {
                            if let Ok(c) = completion {
                                drain_completion(&mut policy, &c, &mut in_flight_count, &mut device_idle_since, &mut in_flight_fires, &mut released_fires);
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
                                    &mut policy,
                                    &mut tombstones,
                                    &mut suspended,
                                    &mut dep_stash,
                                );
                            }
                            continue;
                        }
                    }
                };
                let Some(pending) = prepare_pending_for_batch(&batch, pending) else {
                    continue;
                };
                let Some(pending) =
                    stash_chain_continuation(&mut dep_stash, waitall_active, pending)
                else {
                    continue;
                };
                first_submitted_at_us = pending.submitted_at_us;
                policy.on_arrival(
                    &pending.program_identity_hashes,
                    tombstones.filter(pending.pipeline_id),
                    Instant::now(),
                );
                batch.push(pending);
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
            while next_pending.is_none() {
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
                let Some((pending, usage)) = prepare_pending_with_usage(&batch, pending)
                else {
                    continue;
                };
                // N+k chain ordering (bravo deep pre-submission): a stashed
                // pipeline's next request is the next chain link — stash it in
                // order before any batch classification (see the helper).
                let Some(pending) =
                    stash_chain_continuation(&mut dep_stash, waitall_active, pending)
                else {
                    continue;
                };
                // G2 prebuilt-passthrough: a prebuilt beam fire is a complete
                // pre-assembled multi-lane batch — it never co-batches. Stash it
                // to force a solo fire (either the incoming is prebuilt → fire the
                // current batch first, or the current batch already holds one).
                if pending.prebuilt || batch.has_prebuilt() {
                    next_pending = Some(pending);
                    break;
                }
                // Run-ahead one-step separation (R10): a forward `t+1` whose
                // token-carryover source `t` is in this batch fires in the NEXT
                // batch (after `t` samples its token) — co-batching would read a
                // not-yet-sampled token.
                if batch.would_depend_on_batch(&pending) {
                    // Piece 4: under WaitAll, hold `t+1` per-pipeline and KEEP
                    // gathering the wave (do NOT force-fire via `next_pending`).
                    // Promoted into the next wave post-fire. Untracked (`None`)
                    // requests + legacy fall through to the correct force-fire.
                    if waitall_active {
                        if let Some(pid) = pending.pipeline_id {
                            dep_stash.entry(pid).or_default().push_back(pending);
                            continue;
                        }
                    }
                    next_pending = Some(pending);
                    break;
                }
                if batch.would_exceed_with(&usage) {
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
                    next_pending = Some(pending);
                    break;
                }
                policy.on_arrival(
                    &pending.program_identity_hashes,
                    tombstones.filter(pending.pipeline_id),
                    Instant::now(),
                );
                batch.push_with(pending, usage);
                if batch.is_full() {
                    break;
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
                next_pending.is_some(),
                in_flight_count,
                super::adaptive_policy::max_in_flight(),
            ) {
                FireOutcome::Fire {
                    missing: Vec::new(),
                }
            } else {
                policy.decide(batch.len(), Instant::now())
            };
            match decision {
                FireOutcome::Fire { missing } => {
                    // No in-flight gate to acquire: the scheduler runs
                    // execute_batch synchronously, so we can only reach
                    // here when the previous fire has fully completed.

                    // Do one last non-blocking drain so requests that
                    // arrived between the recv loop and here are
                    // coalesced into this batch instead of being
                    // stranded behind it.
                    let fire_prepare_start = Instant::now();
                    while next_pending.is_none() && !batch.is_full() {
                        let Ok(pending) = req_rx.try_recv() else {
                            break;
                        };
                        if let Some(msg) = batch.single_request_limit_error(&pending) {
                            pending.send_error(msg);
                            continue;
                        }
                        // N+k chain ordering (bravo deep pre-submission): stash a
                        // chained pipeline's next link in order (see the helper).
                        let Some(pending) =
                            stash_chain_continuation(&mut dep_stash, waitall_active, pending)
                        else {
                            continue;
                        };
                        // G2 prebuilt-passthrough: never co-batch a prebuilt fire.
                        if pending.prebuilt || batch.has_prebuilt() {
                            next_pending = Some(pending);
                            break;
                        }
                        if batch.would_depend_on_batch(&pending) {
                            if waitall_active {
                                if let Some(pid) = pending.pipeline_id {
                                    dep_stash.entry(pid).or_default().push_back(pending);
                                    continue;
                                }
                            }
                            next_pending = Some(pending);
                            break;
                        }
                        if batch.would_exceed(&pending) {
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
                            next_pending = Some(pending);
                            break;
                        }
                        policy.on_arrival(
                    &pending.program_identity_hashes,
                    tombstones.filter(pending.pipeline_id),
                    Instant::now(),
                );
                        batch.push(pending);
                    }

                    let total_tokens = batch.total_tokens();
                    // Wait-for-all wave gauge (M-AB): sample the wait-set size +
                    // stragglers at each WaitAll fire so `avg_active`/`avg_missing`
                    // (get_stats) discriminate a persistent wait-set converging to
                    // fleet width (dense waves) from a transient one stuck ≈1
                    // (singleton waves), and a deadline hold (missing>0) from an
                    // all-ready fire. Legacy/quorum never reach here as a WaitAll.
                    if waitall_active {
                        let active = policy.distinct_program_count() as u64;
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
                            batch.len(),
                            total_tokens,
                            batch.should_prefill_coalesce(),
                            next_pending.is_some(),
                            policy.distinct_program_count(),
                            missing.len(),
                        ));
                    }
                    let requests_to_fire = batch.take();
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
                        Self::build_batch_request(&requests_to_fire, page_size, &stats)
                    };
                    let submission_latency = build_start.elapsed();

                    match driver::fire_batch_deferred(driver_idx, batch_req) {
                        Ok(handle) => {
                            // The batch is enqueued (its order fixed) — record it
                            // as in-flight so the policy paces the next fire.
                            policy.on_fired(batch_size as usize);

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
                            for pid in policy.take_terminate_candidates() {
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
                                    &mut policy,
                                    &mut in_flight_count,
                                    &in_flight_fires,
                                    &mut released_fires,
                                );
                                // Tombstone BEFORE terminating: a stale queued
                                // request from this pid must not re-join the
                                // wait-set (it was already removed in
                                // `decide_wave_at`). `process::terminate` also
                                // broadcasts a Leave, but tombstoning here closes
                                // the window until that drains.
                                tombstones.insert(pid);
                                // Demote = terminate (dead): drop + error its
                                // stash NOW so THIS fire's promotion below doesn't
                                // fire the dead pid's chain untracked (the async
                                // Leave from `process::terminate` only arrives a
                                // loop-iteration later).
                                suspended.remove(&pid);
                                clear_left_pipeline_stash(&mut dep_stash, pid);
                                crate::process::terminate(
                                    pid,
                                    Err("scheduler: wait-for-all miss-limit (unresponsive pipeline)"
                                        .to_string()),
                                );
                            }

                            // Piece 4: the fired wave cleared these pipelines' `t`
                            // (no longer in the now-empty batch), so their stashed
                            // `t+1` can join the forming NEXT wave (SUSPENDED
                            // pipelines skipped — parked for resume). Same promotion
                            // the BAR-1 ignition pump uses.
                            if waitall_active {
                                promote_dep_stash(
                                    &mut dep_stash,
                                    &suspended,
                                    &mut policy,
                                    &tombstones,
                                    &mut batch,
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
                            let bubble_us = if in_flight_count == 0 {
                                device_idle_since
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
                            let fire_id = next_fire_id;
                            next_fire_id = next_fire_id.wrapping_add(1);
                            let fire_lanes: Vec<ProcessId> = requests_to_fire
                                .iter()
                                .filter_map(|r| r.pipeline_id)
                                .collect();
                            in_flight_fires.insert(fire_id, fire_lanes);
                            in_flight_count += 1;
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
                                let timing = Self::dispatch_fired_batch(
                                    fire_result,
                                    requests_to_fire,
                                    driver_id,
                                    page_size,
                                    &rt_handle_spawn,
                                    Some(submit_tx_spawn),
                                    &stats_spawn,
                                );
                                Self::record_fire_stats(
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
                FireOutcome::Wait(wait_duration) => {
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
                    if next_pending.is_some() {
                        crossbeam::channel::select! {
                            recv(latency_rx) -> completion => {
                                if let Ok(c) = completion {
                                    drain_completion(&mut policy, &c, &mut in_flight_count, &mut device_idle_since, &mut in_flight_fires, &mut released_fires);
                                }
                            }
                            default(wait_duration) => {}
                        }
                        continue;
                    }
                    crossbeam::channel::select! {
                        recv(req_rx) -> maybe_req => {
                            match maybe_req {
                                Ok(pending) => {
                                    let Some(pending) = prepare_pending_for_batch(&batch, pending)
                                    else {
                                        continue;
                                    };
                                    // N+k chain ordering (bravo deep pre-submission):
                                    // stash a chained pipeline's next link in order.
                                    let Some(pending) = stash_chain_continuation(
                                        &mut dep_stash,
                                        waitall_active,
                                        pending,
                                    ) else {
                                        continue;
                                    };
                                    if batch.would_depend_on_batch(&pending) {
                                        if waitall_active {
                                            if let Some(pid) = pending.pipeline_id {
                                                dep_stash
                                                    .entry(pid)
                                                    .or_default()
                                                    .push_back(pending);
                                                continue;
                                            }
                                        }
                                        next_pending = Some(pending);
                                        continue;
                                    }
                                    if batch.would_exceed(&pending) {
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
                                        next_pending = Some(pending);
                                        continue;
                                    }
                                    policy.on_arrival(
                    &pending.program_identity_hashes,
                    tombstones.filter(pending.pipeline_id),
                    Instant::now(),
                );
                                    batch.push(pending);
                                }
                                Err(_) => break 'run_loop, // channel closed
                            }
                        }
                        recv(latency_rx) -> completion => {
                            if let Ok(c) = completion {
                                drain_completion(&mut policy, &c, &mut in_flight_count, &mut device_idle_since, &mut in_flight_fires, &mut released_fires);
                            }
                        }
                        default(wait_duration) => {}
                    }
                }
            }
        }

        // Shutdown: fire the remaining batch synchronously so any
        // inferlets still awaiting responses get them before we exit.
        // ~10 ms of additional shutdown latency in the worst case.
        if !batch.is_empty() {
            let requests = batch.take();
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

    /// Build the batched `pie_driver_abi::ForwardRequest` by folding each
    /// per-request shape into one batch. Runs on the scheduler thread (so it
    /// overlaps the GPU of any in-flight batch); the caller then enqueues it in
    /// fire-order via [`driver::fire_batch_deferred`].
    fn build_batch_request(
        requests: &[PendingRequest],
        page_size: u32,
        stats: &SchedulerStats,
    ) -> pie_driver_abi::ForwardRequest {
        // Build batched request — a single `pie_driver_abi::ForwardRequest`
        // populated by folding each per-request shape into the batch.
        let elide_decode_masks = requests.iter().all(|req| {
            req.request.single_token_mode
                && !req.request.has_user_mask
                && req.request.token_ids.len() <= 1
                && req.request.spec_token_ids.is_empty()
        });
        crate::probe_fire!(stats.fire.execute.batch_build_us, {
            let mut batch_req =
                request::new_batched_forward_request_with_capacity(requests.len());
            for req in requests {
                request::append_request_with_options(
                    &mut batch_req,
                    &req.request,
                    &req.physical_page_ids,
                    req.last_page_len,
                    page_size,
                    elide_decode_masks,
                );
            }
            batch_req
        })
    }

    /// Fold a completed batch's always-on counters + spec-domain accumulators
    /// into the shared stats. `latency` is the off-thread forward (GPU) wait —
    /// the dominant component of the batch's wall time under the overlapped
    /// fire (the host build/enqueue overlaps the prior in-flight batch).
    fn record_fire_stats(
        stats: &SchedulerStats,
        timing: &BatchExecutionTiming,
        latency: Duration,
        batch_size: u64,
        total_tokens: usize,
    ) {
        crate::probe_fire!(stats.fire.post_dispatch.stats_update_us, {
            stats.total_batches.fetch_add(1, Relaxed);
            stats
                .total_tokens_processed
                .fetch_add(total_tokens as u64, Relaxed);
            stats
                .total_requests_processed
                .fetch_add(batch_size, Relaxed);
            stats
                .max_forward_requests_observed
                .fetch_max(batch_size, Relaxed);
            let bucket = match batch_size {
                0 | 1 => 0,
                2..=3 => 1,
                4..=7 => 2,
                8..=15 => 3,
                16..=31 => 4,
                32..=63 => 5,
                64..=127 => 6,
                _ => 7,
            };
            stats.batch_size_hist[bucket].fetch_add(1, Relaxed);
            stats
                .last_batch_latency_us
                .store(latency.as_micros() as u64, Relaxed);
            stats
                .cumulative_latency_us
                .fetch_add(latency.as_micros() as u64, Relaxed);
            stats
                .system_spec_draft_tokens_proposed
                .fetch_add(timing.system_spec_draft_tokens_proposed, Relaxed);
            stats
                .system_spec_draft_tokens_accepted
                .fetch_add(timing.system_spec_draft_tokens_accepted, Relaxed);
            for (counter, value) in stats
                .system_spec_draft_tokens_proposed_per_pos
                .iter()
                .zip(timing.system_spec_draft_tokens_proposed_per_pos)
            {
                if value != 0 {
                    counter.fetch_add(value, Relaxed);
                }
            }
            for (counter, value) in stats
                .system_spec_draft_tokens_accepted_per_pos
                .iter()
                .zip(timing.system_spec_draft_tokens_accepted_per_pos)
            {
                if value != 0 {
                    counter.fetch_add(value, Relaxed);
                }
            }
        });
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
        let batch_req = Self::build_batch_request(&requests, page_size, stats);
        let fire_result = match driver::fire_batch_deferred(driver_idx, batch_req) {
            Ok(handle) => handle.wait(),
            Err(e) => Err(e),
        };
        Self::dispatch_fired_batch(
            fire_result,
            requests,
            driver_id,
            page_size,
            rt_handle,
            submit_tx,
            stats,
        )
    }

    /// Dispatch a fired batch's response to the per-request oneshots and
    /// accumulate spec-decode draft counters. `fire_result` is the awaited
    /// forward response (the GPU wait already happened off-thread). The
    /// `deferred_drop` punt still routes request-husk dealloc to the blocking
    /// pool so it does not compete with response dispatch.
    fn dispatch_fired_batch(
        fire_result: Result<pie_driver_abi::ForwardResponse>,
        requests: Vec<PendingRequest>,
        driver_id: DriverId,
        page_size: u32,
        rt_handle: &tokio::runtime::Handle,
        submit_tx: Option<crossbeam::channel::Sender<PendingRequest>>,
        stats: &SchedulerStats,
    ) -> BatchExecutionTiming {
        // Detect if ANY request carries system spec drafts. The
        // common case (256-conc decode) has none, so we skip the
        // per-request Vec build + position-histogram loop.
        let any_spec = requests.iter().any(|req| !req.request.spec_token_ids.is_empty());
        let system_spec_proposed_per_req: Vec<usize> = if any_spec {
            requests
                .iter()
                .map(|req| req.request.spec_token_ids.len())
                .collect()
        } else {
            Vec::new()
        };
        let system_spec_draft_tokens_proposed =
            system_spec_proposed_per_req.iter().sum::<usize>() as u64;
        let mut system_spec_draft_tokens_accepted = 0u64;
        let mut system_spec_draft_tokens_proposed_per_pos =
            [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];
        let mut system_spec_draft_tokens_accepted_per_pos =
            [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];
        if any_spec {
            for proposed in &system_spec_proposed_per_req {
                for pos in 0..(*proposed).min(SYSTEM_SPEC_DRAFT_POS_BUCKETS) {
                    system_spec_draft_tokens_proposed_per_pos[pos] += 1;
                }
            }
        }

        // Response dispatch: per-request oneshot fires and queueing the
        // deferred_drop Vec. The GPU wait already happened off-thread (the
        // caller awaited `FireHandle::wait` before handing us `fire_result`).
        //
        // Per-completion-type counts are accumulated into these locals
        // inside the match arms and fetch_add'd once after the loop, so
        // we don't pay a per-request atomic op on the hot path.
        let response_dispatch_start = Instant::now();
        let mut direct_count: u64 = 0;
        let mut chunk_count: u64 = 0;
        match fire_result {
            Ok(batch_resp) => {
                let wp = batch_resp.probe_wire_parse_us as u64;
                let pl = batch_resp.probe_plan_us as u64;
                let hd = batch_resp.probe_h2d_us as u64;
                let kl = batch_resp.probe_kernel_launch_us as u64;
                let sy = batch_resp.probe_sync_us as u64;
                let rb = batch_resp.probe_response_build_us as u64;
                let di = batch_resp.probe_device_idle_us as u64;
                if wp | pl | hd | kl | sy | rb | di != 0 {
                    stats.driver_cuda.wire_parse_us.fetch_add(wp, Relaxed);
                    stats.driver_cuda.plan_us.fetch_add(pl, Relaxed);
                    stats.driver_cuda.h2d_us.fetch_add(hd, Relaxed);
                    stats.driver_cuda.kernel_launch_us.fetch_add(kl, Relaxed);
                    stats.driver_cuda.sync_us.fetch_add(sy, Relaxed);
                    stats.driver_cuda.response_build_us.fetch_add(rb, Relaxed);
                    stats.driver_cuda.device_idle_us.fetch_add(di, Relaxed);
                    // Feed the ACCURATE device-idle into the driver bubble
                    // histogram (the true G3 p50). We're inside the non-zero
                    // guard, so the CUDA driver IS profiling — `di == 0` here
                    // legitimately means "device was busy, no bubble" (recorded
                    // as bucket 0, parallel to the host proxy's 0s). When the
                    // driver doesn't profile, this block is skipped entirely, so
                    // the driver histogram stays empty and readers fall back to
                    // the host-proxy histogram (`InferenceStats::bubble_p50`).
                    stats.record_bubble_us_driver(di);
                }
                let n_results = batch_resp.num_requests as usize;
                if requests.len() == 1 && requests[0].prebuilt {
                    // === G2 prebuilt-passthrough (beam) response ===
                    // B driver-lanes but ONE program (program 0) + ONE
                    // PendingRequest. Hand the WHOLE rich response verbatim to the
                    // single completion — ptir_host reads `resp.ptir_output_at(0)`
                    // (the [B] out/out_par/out_scr program tensors). Bypasses the
                    // per-row `num_requests == requests.len()` split below
                    // (num_requests = B ≠ the 1 PendingRequest).
                    let req = requests
                        .into_iter()
                        .next()
                        .expect("prebuilt fire has exactly one request");
                    // `..` drops `request` + `physical_page_ids` inline (one
                    // request — negligible; the KV txn is finalized by ptir_host).
                    let PendingRequest { completion, .. } = req;
                    match completion {
                        Completion::Direct(tx) => {
                            direct_count += 1;
                            tx.send(Ok(ForwardOutput::Response(batch_resp))).ok();
                        }
                        Completion::Chunk { .. } => {
                            // Unreachable: a prebuilt beam fire is always a single
                            // Direct completion, never a chunked continuation.
                            tracing::error!(
                                "prebuilt beam fire had an unexpected Chunk completion"
                            );
                        }
                    }
                    stats.fire.last_dispatch_end_micros.store(now_micros(), Relaxed);
                } else if n_results != requests.len() {
                    let msg = format!(
                        "batch response count mismatch from driver {driver_id}: \
                         expected {}, got {n_results}",
                        requests.len()
                    );
                    tracing::error!(
                        driver = driver_id,
                        expected = requests.len(),
                        got = n_results,
                        "Batch response count mismatch",
                    );
                    for req in requests {
                        req.send_result::<ForwardOutput>(
                            Err(anyhow::anyhow!(msg.clone())),
                            None,
                            page_size,
                        );
                    }
                } else {
                    let has_chunked = requests
                        .iter()
                        .any(|req| matches!(req.completion, Completion::Chunk { .. }));
                    let token_payload_only = !has_chunked
                        && batch_resp.dists_ids.is_empty()
                        && batch_resp.dists_probs.is_empty()
                        && batch_resp.logits_bytes.is_empty()
                        && batch_resp.logprobs_values.is_empty()
                        && batch_resp.entropies.is_empty()
                        && batch_resp.spec_tokens.is_empty()
                        // Key on DECLARED program-token output SLOTS, not the flat
                        // `program_tokens` emptiness: a program that declares a
                        // `[k]`-Token output but produces an EMPTY accept-prefix
                        // (all `-1` → truncated) has empty flat `program_tokens` AND
                        // empty dense `tokens`, so the old `program_tokens.is_empty()`
                        // mis-routed it to the dense-token shortcut → its declared
                        // output was lost ("no output tensor"; the §6.1 mtpverify /
                        // grammar_inferlet_constrains_output root). `program_tokens_
                        // req_indptr.last()` = Σ declared slots; 0 ⇒ no program-token
                        // output ⇒ dense-token-only. Non-zero ⇒ take the rich
                        // Response path (which reconstructs the empty `[k]` tensor).
                        && batch_resp
                            .program_tokens_req_indptr
                            .last()
                            .map_or(true, |&n| n == 0)
                        && batch_resp.tokens_indptr.len() >= requests.len() + 1;

                    // Send oneshot replies first, defer drop of the
                    // request husks. Each PendingRequest's drop is
                    // ~3-4 µs (22-Vec ForwardRequest), and doing it
                    // inline adds avoidable tail latency to response
                    // dispatch for large batches.
                    let mut deferred_drop: Vec<(
                        pie_driver_abi::ForwardRequest,
                        Vec<PhysicalPageId>,
                    )> = Vec::with_capacity(n_results);
                    // #27 cut #1 eager-D2H fast-path (a2-mode): a request that set
                    // up the `sampling_output_*` dst table had its sampled token
                    // copied DIRECTLY to the pinned output Tensor (D2H), so the
                    // driver response carries NO token (`tokens[]` empty). Resolve
                    // each oneshot with success WITHOUT extracting `tokens[..]` —
                    // the inferlet's `output()` reads the filled pinned buffer
                    // (gated on `forward_result.is_some()`); an `Err`/drop here
                    // would hit the abort path (txn drop + "no output tensor").
                    // Keyed per-request on `sampling_output_*`, which
                    // `populate_output_fastpath` sets iff it also stashed the
                    // pinned outputs (1:1 with the host pinned-read gate, so no
                    // skew). One-ahead MVP batches are all-or-nothing fast-path.
                    let all_fast_path = !requests.is_empty()
                        && requests
                            .iter()
                            .all(|req| !req.request.sampling_output_dst_ptrs.is_empty());
                    if all_fast_path {
                        for req in requests.into_iter() {
                            // Empty `Tokens` is `Some` → the host gate passes and
                            // reads the pinned buffer; the payload is ignored.
                            let output = ForwardOutput::Tokens(Vec::new());
                            let PendingRequest {
                                request,
                                completion,
                                physical_page_ids,
                                program_identity_hashes,
                                pipeline_id,
                                ..
                            } = req;
                            match completion {
                                Completion::Direct(tx) => {
                                    direct_count += 1;
                                    tx.send(Ok(output)).ok();
                                    deferred_drop.push((request, physical_page_ids));
                                }
                                Completion::Chunk { .. } => {
                                    chunk_count += 1;
                                    let req = PendingRequest {
                                        request,
                                        completion,
                                        physical_page_ids,
                                        last_page_len: 0,
                                        program_identity_hashes,
                                        pipeline_id,
                                        submitted_at_us: 0,
                                        prebuilt: false,
                                    };
                                    req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                                }
                            }
                        }
                    } else if token_payload_only {
                        for (r, req) in requests.into_iter().enumerate() {
                            let lo = batch_resp.tokens_indptr[r] as usize;
                            let hi = batch_resp.tokens_indptr[r + 1] as usize;
                            if system_spec_proposed_per_req
                                .get(r)
                                .copied()
                                .unwrap_or_default()
                                > 0
                            {
                                let accepted = hi.saturating_sub(lo).saturating_sub(1);
                                system_spec_draft_tokens_accepted += accepted as u64;
                                for pos in 0..accepted.min(SYSTEM_SPEC_DRAFT_POS_BUCKETS) {
                                    system_spec_draft_tokens_accepted_per_pos[pos] += 1;
                                }
                            }
                            let output = if hi == lo + 1 {
                                ForwardOutput::Token(batch_resp.tokens[lo])
                            } else {
                                ForwardOutput::Tokens(batch_resp.tokens[lo..hi].to_vec())
                            };
                            let PendingRequest {
                                request,
                                completion,
                                physical_page_ids,
                                program_identity_hashes,
                                pipeline_id,
                                ..
                            } = req;
                            match completion {
                                Completion::Direct(tx) => {
                                    direct_count += 1;
                                    tx.send(Ok(output)).ok();
                                    deferred_drop.push((request, physical_page_ids));
                                }
                                Completion::Chunk { .. } => {
                                    chunk_count += 1;
                                    let req = PendingRequest {
                                        request,
                                        completion,
                                        physical_page_ids,
                                        last_page_len: 0,
                                        program_identity_hashes,
                                        pipeline_id,
                                        submitted_at_us: 0,
                                        prebuilt: false,
                                    };
                                    req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                                }
                            }
                        }
                    } else {
                        for (r, req) in requests.into_iter().enumerate() {
                            let per_req = request::extract_per_request(&batch_resp, r);
                            if system_spec_proposed_per_req
                                .get(r)
                                .copied()
                                .unwrap_or_default()
                                > 0
                            {
                                let accepted = per_req.tokens.len().saturating_sub(1);
                                system_spec_draft_tokens_accepted += accepted as u64;
                                for pos in 0..accepted.min(SYSTEM_SPEC_DRAFT_POS_BUCKETS) {
                                    system_spec_draft_tokens_accepted_per_pos[pos] += 1;
                                }
                            }
                            let output = ForwardOutput::Response(per_req);
                            let PendingRequest {
                                request,
                                completion,
                                physical_page_ids,
                                program_identity_hashes,
                                pipeline_id,
                                ..
                            } = req;
                            match completion {
                                Completion::Direct(tx) => {
                                    direct_count += 1;
                                    tx.send(Ok(output)).ok();
                                    deferred_drop.push((request, physical_page_ids));
                                }
                                Completion::Chunk { .. } => {
                                    chunk_count += 1;
                                    let req = PendingRequest {
                                        request,
                                        completion,
                                        physical_page_ids,
                                        last_page_len: 0,
                                        program_identity_hashes,
                                        pipeline_id,
                                        submitted_at_us: 0,
                                        prebuilt: false,
                                    };
                                    req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                                }
                            }
                        }
                    }
                    stats.fire.last_dispatch_end_micros.store(now_micros(), Relaxed);
                    if !deferred_drop.is_empty() {
                        // Dedicated blocking pool so this dealloc task
                        // does not compete with response dispatch. Use the captured
                        // `rt_handle` because we're now on the scheduler
                        // OS thread, not a tokio task — `tokio::task::
                        // spawn_blocking` would panic without an ambient
                        // runtime context.
                        rt_handle.spawn_blocking(move || drop(deferred_drop));
                    }
                }
            }
            Err(e) => {
                tracing::error!("fire_batch failed for driver {}: {:?}", driver_id, e);
                for req in requests {
                    req.send_result::<ForwardOutput>(
                        Err(anyhow::anyhow!(
                            "fire_batch failed for driver {driver_id}: {e:#}"
                        )),
                        None,
                        page_size,
                    );
                }
            }
        }
        crate::probe_fire_record!(
            stats.fire.execute.response_dispatch.total_us,
            response_dispatch_start.elapsed()
        );
        // Task-B (carrier ⋈ contention): every response sent above is now
        // drainable by its owner's gate loop — wake lanes waiting to finalize
        // their own retired fires (releases their pins/grace refs so
        // `classify_for_suspend` can yield). No-op outside preempt mode.
        crate::contention::notify_fire_retired();
        // Per-completion-type counts. Counters, not durations — three
        // atomic ops per fire regardless of batch size, so always-on
        // (no feature gate).
        if direct_count > 0 {
            stats
                .fire
                .execute
                .response_dispatch
                .direct_count
                .fetch_add(direct_count, Relaxed);
        }
        if chunk_count > 0 {
            stats
                .fire
                .execute
                .response_dispatch
                .chunk_count
                .fetch_add(chunk_count, Relaxed);
        }
        BatchExecutionTiming {
            system_spec_draft_tokens_proposed,
            system_spec_draft_tokens_accepted,
            system_spec_draft_tokens_proposed_per_pos,
            system_spec_draft_tokens_accepted_per_pos,
        }
    }
}
