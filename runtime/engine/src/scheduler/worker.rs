//! Per-driver direct batch scheduler.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::driver::{
    BoundInstance, ChannelRegistrationPlan, DriverBackend, DriverId, InstanceBindingPlan,
    LaunchLease, LaunchPrepareOutcome, PoolResizePlan, ProgramRegistration, RegisteredChannel,
    SchedulerLimits, StateCopyPlan, SubmissionCompletion, WorkItemAttemptOutcome,
    WorkItemCompletion,
};
use crate::scheduler::ProcessId;
use anyhow::{Result, anyhow};

use super::batch::{self, BatchAccumulator};
use super::quorum;
use super::stats::{self, SchedulerStats};
use super::{ControlCompletion, RetryClassifier};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FireClause {
    Quorum,
    ColdHold,
    /// The wave window expired: fired narrow, demoting the absentees.
    Straggler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaveKind {
    Terminate,
    Suspend,
    /// A pipeline closed or dropped. Its wait-set row is released immediately,
    /// while every request already accepted by the scheduler continues
    /// untracked to settlement. Later guest submissions are rejected by the
    /// pipeline resource before they reach the scheduler.
    Close,
}

/// A pipeline left the fleet (cancel / kill / exit / TASK-A terminate /
/// TASK-B preempt). Broadcasts to EVERY registered driver's scheduler
/// thread (a pipeline's requests may have landed on any of them) so each
/// thread's local [`quorum::WaitAllPolicy`] drops `pid` from its wave
/// wait-set. Fire-and-forget: a shutting-down/closed scheduler channel is
/// silently skipped (nothing left there to notify).
pub(crate) fn notify_pipeline_leave(pid: ProcessId, _kind: LeaveKind) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::PipelineLeave(pid, _kind));
    }
}

/// No-op: quorum rejoin is implicit on the pipeline's next scheduler
/// submission, so a join event has
/// nothing to do here (`store::reclaim` owns the join hook for its own
/// suspend/restore callers, which is equally inert for the same reason).
#[allow(dead_code)] // no live caller — see doc.
pub(crate) fn notify_pipeline_join(_pid: ProcessId) {}

/// Wake-class counter (plan §16.2): completions that the 250 ms hang backstop
/// discovered already settled — a lost nudge. Steady state stays at zero; any
/// increment is a wake-path regression worth a warning.
pub(crate) static BACKSTOP_RETIREMENTS: AtomicU64 = AtomicU64::new(0);
static NEXT_LOGICAL_FIRE_ID: AtomicU64 = AtomicU64::new(1);

/// Total backstop-path retirements since process start (test observability).
#[cfg(test)]
pub(crate) fn backstop_retirements() -> u64 {
    BACKSTOP_RETIREMENTS.load(Ordering::Relaxed)
}

fn retry_warn_at() -> u32 {
    static WARN_AT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *WARN_AT.get_or_init(|| {
        std::env::var("PIE_FIRE_RETRY_WARN")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(32)
    })
}

fn max_fire_retries() -> u32 {
    static MAX_RETRIES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MAX_RETRIES.get_or_init(|| {
        std::env::var("PIE_FIRE_RETRY_MAX")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1024)
            .max(1)
    })
}

pub(crate) struct PendingRequest {
    pub(crate) logical_fire_id: u64,
    pub(crate) request: crate::driver::LaunchPlan,
    pub(crate) instance_id: u64,
    pub(crate) completion: WorkItemCompletion,
    pub(crate) last_page_len: u32,
    /// The owning process. Process-wide suspend/terminate acts on every
    /// request with this identity.
    pub(crate) process_id: Option<ProcessId>,
    /// The submitting pipeline resource's stable scope identity, or `None`
    /// for an untracked/prebuilt fire. This is the quorum wait-set key.
    pub(crate) pipeline_id: Option<ProcessId>,
    pub(crate) prebuilt: bool,
    /// Stable logical-fire retry state. The request payload and completion are
    /// retained across attempts; only the native terminal cell is reset.
    pub(crate) retry_count: u32,
    /// Earliest instant the next attempt may dispatch. Peek and dispatch skip
    /// a not-yet-due retry, so retry pacing is the backoff itself — never the
    /// peers' cadence (a single-pipeline fleet must not spin hot — RV-20).
    pub(crate) retry_after: Option<Instant>,
    pub(crate) prelaunch_copy: Option<crate::driver::KvCopyPlan>,
    pub(crate) prelaunch_state_copy: Option<StateCopyPlan>,
    pub(crate) retry_classifier: Option<RetryClassifier>,
    /// Whether this request currently holds an unconsumed readiness credit
    /// in the quorum wait-set. Published at admission (or, for a fire behind
    /// a pre-launch copy, when the copy retires), consumed by the wave
    /// dispatch, re-published on a RETRY re-arm. Drop paths give the credit
    /// back through `on_request_dropped` ONLY when this is set — a fire
    /// cancelled before its copy retires never had one (RV-20).
    pub(crate) credit_published: bool,
    /// The quorum identity generation of `pipeline_id` at worker receipt
    /// (`WaitAllPolicy::generation_of`). Every quorum accounting call for
    /// this request routes through [`Scheduler::quorum_pid`], which
    /// compares this stamp against the CURRENT generation: a mismatch means
    /// the pipeline left (Close/Terminate/Suspend bump the generation)
    /// after this request was admitted, so its credit rides — and is
    /// consumed — untracked. Without the stamp, a fire finishing async
    /// preparation after a Close or allocation-wait leave re-created the
    /// departed scope's wait-set row, and a straddling wave could consume a
    /// later incarnation's credit:
    /// +1 permanent `untracked_ready` per close, wave_started resets
    /// starved, instant narrow fires after any >window lull (W1, 2026-07-17).
    pub(crate) quorum_generation: u64,
    pub(super) timing: Option<FireTimingState>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FireTimingState {
    submitted_us: u64,
    enqueued_us: Option<u64>,
    ready_us: Option<u64>,
}

impl FireTimingState {
    fn new() -> Self {
        Self {
            submitted_us: super::fire_timing_now_us(),
            enqueued_us: None,
            ready_us: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WaveTimingState {
    wave_id: u64,
    membership_hash: u64,
    dispatch_started_us: u64,
    batch_built_us: u64,
    driver_started_us: u64,
    launch_returned_us: u64,
    decision_us: u64,
    active_pipelines: usize,
    missing_pipelines: usize,
    candidate_count: usize,
    deferred_pipelines: usize,
    depth_capped_pipelines: usize,
}

#[derive(Clone, Copy, Debug)]
struct FireTimingSnapshot {
    outcome_index: usize,
    logical_fire_id: u64,
    instance_id: u64,
    process_id: Option<ProcessId>,
    sampled_rows: usize,
    retry_count: u32,
    timing: FireTimingState,
}

impl PendingRequest {
    fn direct(
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        last_page_len: u32,
        process_id: Option<ProcessId>,
        pipeline_id: Option<ProcessId>,
        prebuilt: bool,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
        retry_classifier: Option<RetryClassifier>,
        timing_enabled: bool,
    ) -> Self {
        let logical_fire_id = NEXT_LOGICAL_FIRE_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            logical_fire_id,
            request,
            instance_id,
            completion,
            last_page_len,
            process_id,
            pipeline_id,
            prebuilt,
            retry_count: 0,
            retry_after: None,
            prelaunch_copy,
            prelaunch_state_copy,
            retry_classifier,
            credit_published: false,
            // Stamped with the live generation at worker RECEIPT
            // (SchedulerItem::Launch arm) — constructed requests outside the
            // worker cannot see the policy; 0 matches a never-left pipeline.
            quorum_generation: 0,
            timing: timing_enabled.then(FireTimingState::new),
        }
    }

    fn retry_eligible(&self) -> bool {
        self.request.rs_slot_ids.is_empty()
            && self.request.rs_buffer_slot_ids.is_empty()
            && self.request.rs_fold_lens.is_empty()
    }

    fn clone_for_batch(&self) -> Self {
        Self {
            logical_fire_id: self.logical_fire_id,
            request: self.request.clone(),
            instance_id: self.instance_id,
            completion: self.completion.clone(),
            last_page_len: self.last_page_len,
            pipeline_id: self.pipeline_id,
            prebuilt: self.prebuilt,
            retry_count: self.retry_count,
            retry_after: self.retry_after,
            prelaunch_copy: self.prelaunch_copy.clone(),
            prelaunch_state_copy: self.prelaunch_state_copy.clone(),
            retry_classifier: None,
            credit_published: self.credit_published,
            quorum_generation: self.quorum_generation,
            timing: self.timing,
        }
    }

    pub(crate) fn wire_row_count(&self) -> usize {
        self.request.qo_indptr.len().saturating_sub(1)
    }

    pub(crate) fn preserves_inner_rows(&self) -> bool {
        self.wire_row_count() > 1
    }

    fn requires_solo_submission(&self) -> bool {
        (self.prebuilt && self.pipeline_id.is_none())
            || (self.preserves_inner_rows() && self.request.qo_indptr.last().copied() == Some(0))
    }
}

fn fire_membership_hash<'a>(logical_fire_ids: impl IntoIterator<Item = &'a u64>) -> u64 {
    logical_fire_ids
        .into_iter()
        .fold(14_695_981_039_346_656_037u64, |hash, logical_fire_id| {
            (hash ^ logical_fire_id).wrapping_mul(1_099_511_628_211)
        })
}

#[derive(Default)]
struct LaunchGrouping {
    instances: HashSet<u64>,
    /// Tracked pipelines already contributing to this wave. ONE wave-member
    /// per pipeline per wave: fires of one pipeline are ORDERED (B3), and
    /// the driver's compose-time geometry/containment validation reads the
    /// device state as of wave entry — a decode fire composed into the same
    /// wave as the prefill it depends on (R4-4 device-carried handoff
    /// submits it run-ahead) validates against KV the prefill has not
    /// committed yet and FAIL-STOPs the lane. Same-instance dedup already
    /// enforced this within an instance; the single-pipeline stream makes
    /// the cross-instance case reachable. Also aligns the composer with the
    /// quorum's one-credit-per-pipeline-per-wave model.
    pipelines: HashSet<ProcessId>,
    count: usize,
    forward_tokens: usize,
    page_refs: usize,
    has_solo_submission: bool,
    has_user_mask: bool,
    has_device_geometry: bool,
}

fn has_dense_device_mask(request: &crate::driver::LaunchPlan) -> bool {
    request.has_user_mask && request.masks.is_empty()
}

impl LaunchGrouping {
    fn accepts(&self, request: &PendingRequest, limits: SchedulerLimits, page_size: u32) -> bool {
        if self.instances.contains(&request.instance_id) {
            return false;
        }
        if request
            .pipeline_id
            .is_some_and(|pid| self.pipelines.contains(&pid))
        {
            return false;
        }
        if self.count != 0 && (request.requires_solo_submission() || self.has_solo_submission) {
            return false;
        }
        // Custom wire masks co-batch freely with other wire-geometry fires —
        // the wire layer emits a mask row per request (synthesized causal for
        // the unmasked ones) and the driver predicates per row. They cannot
        // ride a composed device-geometry batch: wire masks index the wire
        // request layout, which composition replaces (driver fails loud).
        // A DENSE-masked device-resolved fire is stricter still: unlike a
        // host-derived channel mask, it has no wire BRLE rows and the composed
        // path cannot merge it with another program.
        let masked_device_geometry = has_dense_device_mask(&request.request);
        let wire_mask_on_device_geometry = request.request.has_user_mask
            && !request.request.masks.is_empty()
            && request.request.device_resolved_geometry;
        if self.count != 0
            && (masked_device_geometry
                || wire_mask_on_device_geometry
                || (self.has_user_mask && self.has_device_geometry)
                || (request.request.has_user_mask && self.has_device_geometry)
                || (request.request.device_resolved_geometry && self.has_user_mask))
        {
            return false;
        }
        if self.count == 0 {
            return true;
        }
        let usage = batch::request_capacity_usage(request, page_size);
        self.count.saturating_add(usage.forward_requests) <= limits.max_forward_requests
            && self.forward_tokens.saturating_add(usage.forward_tokens) <= limits.max_forward_tokens
            && self.page_refs.saturating_add(usage.page_refs) <= limits.max_page_refs
    }

    fn push(&mut self, request: &PendingRequest, limits: SchedulerLimits, page_size: u32) -> bool {
        let usage = batch::request_capacity_usage(request, page_size);
        self.instances.insert(request.instance_id);
        if let Some(pid) = request.pipeline_id {
            self.pipelines.insert(pid);
        }
        self.count = self.count.saturating_add(usage.forward_requests);
        self.forward_tokens = self.forward_tokens.saturating_add(usage.forward_tokens);
        self.page_refs = self.page_refs.saturating_add(usage.page_refs);
        self.has_solo_submission |= request.requires_solo_submission();
        self.has_user_mask |= request.request.has_user_mask;
        self.has_device_geometry |= request.request.device_resolved_geometry;
        request.requires_solo_submission()
            || has_dense_device_mask(&request.request)
            || self.count >= limits.max_forward_requests
            || self.forward_tokens >= limits.max_forward_tokens
            || self.page_refs >= limits.max_page_refs
    }
}

struct LaunchBatchPreview {
    count: usize,
    logical_fire_ids: Vec<u64>,
    pipelines: HashSet<ProcessId>,
    /// Pipelines whose queued fires the COMPOSER deferred to the next wave
    /// (driver capacity or same-instance dedup). They
    /// are scheduled, not late — the quorum treats them as at-depth, never
    /// missing (zero-straggler gate, V6 iteration 35).
    deferred: HashSet<ProcessId>,
    /// SUBMISSION = PRESENCE (operator model, V6 iteration 38): every
    /// pipeline with ANY work in the engine — queued launch, preparation,
    /// pre-launch copy. The gathering wave may still wait for them (that
    /// wait keeps waves dense), but they can never be demoted or counted
    /// as stragglers: an inferlet that submitted is on time by definition;
    /// engine-side prepare latency must never be punished as straggling.
    submitted: HashSet<ProcessId>,
    structurally_full: bool,
}

enum SchedulerItem {
    Launch {
        pending: PendingRequest,
    },
    RegisterProgram {
        plan: ProgramRegistration,
        response: tokio::sync::oneshot::Sender<Result<u64>>,
    },
    RegisterChannel {
        plan: ChannelRegistrationPlan,
        response: tokio::sync::oneshot::Sender<Result<RegisteredChannel>>,
    },
    RegisterChannels {
        plans: Vec<ChannelRegistrationPlan>,
        response: tokio::sync::oneshot::Sender<Result<Vec<RegisteredChannel>>>,
    },
    BindInstance {
        pipeline_id: Option<ProcessId>,
        plan: InstanceBindingPlan,
        response: tokio::sync::oneshot::Sender<Result<BoundInstance>>,
    },
    /// One dispatch registering an instance's channels AND binding it —
    /// the two per-join controls always run back-to-back with only an
    /// ordering dependency, and dispatching them separately doubled the
    /// turnover control convoy (V6 iteration 25 attribution).
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistrationPlan>,
        /// Some on the program cache's first sight (the driver requires the
        /// instance's channels registered BEFORE the program — status -5
        /// otherwise — so registration must ride between channels and bind
        /// inside the one dispatch); None when the hash is already
        /// registered, with `bind.program_id` carrying the cached id.
        program: Option<ProgramRegistration>,
        bind: InstanceBindingPlan,
        response:
            tokio::sync::oneshot::Sender<Result<(Vec<RegisteredChannel>, u64, BoundInstance)>>,
    },
    CopyKv {
        plan: crate::driver::KvCopyPlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CopyKvTracked {
        plan: crate::driver::KvCopyPlan,
        completion: ControlCompletion,
    },
    // Only reached via `SchedulerHandle::copy_state`/`resize_pool`, which
    // the mock-driver fire path doesn't call yet (`scheduler::resize_pool`
    // is exercised by this module's unit tests) — see `scheduler::dispatch`'s
    // module doc for the full driver-ABI-completeness rationale.
    #[allow(dead_code)]
    CopyState {
        plan: StateCopyPlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    #[allow(dead_code)]
    ResizePool {
        plan: PoolResizePlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CloseInstance {
        id: u64,
        pacing_wait_id: u64,
    },
    CloseChannel {
        id: u64,
    },
    FreezePipeline {
        pid: ProcessId,
        response: tokio::sync::oneshot::Sender<()>,
    },
    ResumePipeline(ProcessId),
    /// Event-driven retirement wake: sent by [`NudgeWaker`] when an in-flight
    /// driver submission completion publishes. Carries no work; it only unblocks the
    /// scheduler's wait so the retire pass runs immediately.
    Nudge,
    /// A pipeline left the fleet ([`notify_pipeline_leave`]'s broadcast).
    /// Handled immediately on dequeue (like [`SchedulerItem::Nudge`]): it
    /// only mutates the run-loop's local `WaitAllPolicy`, never touching
    /// `pending`, so it can't reorder control ops or launches.
    PipelineLeave(ProcessId, LeaveKind),
    /// Snapshot the run loop's state as a human-readable dump (queue
    /// composition, in-flight work, barrier membership). Answered inline on
    /// dequeue — a held wave must be inspectable from outside the thread.
    DebugDump {
        response: tokio::sync::oneshot::Sender<String>,
    },
    /// A driver-lane reply (launch accepted/rejected, control commit).
    /// Handled immediately on dequeue, like `Nudge` — it mutates only
    /// in-flight bookkeeping, never queue order.
    Lane(LaneReply),
    Stop,
}

/// Wakes the scheduler thread through its own queue when a registered driver
/// submission completion publishes, so batch/control retirement is event-driven instead
/// of timeout-polled (plan §5.1).
struct NudgeWaker {
    tx: crossbeam::channel::Sender<SchedulerItem>,
}

impl std::task::Wake for NudgeWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.tx.send(SchedulerItem::Nudge);
    }
}

/// Register the nudge waker on a pending completion's wait slot with
/// register-then-recheck. Returns false when the completion has already
/// settled (or its slot is gone) and the caller should retire immediately.
fn arm_completion_nudge(completion: &SubmissionCompletion, waker: &std::task::Waker) -> bool {
    if completion.is_settled() {
        return false;
    }
    let table = pie_waker::WakerTable::global();
    let slot = completion.wait_id();
    let observed = table.published(slot).unwrap_or_default();
    if !table.register(slot, waker, observed) {
        return false;
    }
    if completion.is_settled() {
        table.deregister(slot);
        return false;
    }
    true
}

#[derive(Clone)]
enum PreLaunchCopy {
    Kv(crate::driver::KvCopyPlan),
    State(StateCopyPlan),
}

impl PreLaunchCopy {
    fn label(&self) -> &'static str {
        match self {
            Self::Kv(_) => "KV copy",
            Self::State(_) => "recurrent-state copy",
        }
    }
}

// =============================================================================
// Driver lane (V6 iteration 48)
// =============================================================================
//
// A dedicated thread owns the `DriverBackend` and executes EVERY driver call
// in FIFO post order, so the driver keeps the exact single-threaded
// serialization it has always had — no driver-side concurrency is introduced.
// What changes is who blocks: launch submit (1.2–2.5 ms) and lifecycle
// controls (0.16 ms p50 with 2–10 ms allocator tails) leave the scheduler
// worker's critical path, which must otherwise fit inside the run-ahead
// pipelining window (the measured 283–437 ms/run of
// sched-lag-after-quorum — the fast/slow boot-mode spread — and the
// 66–88 ms/run of control-occupancy wave gaps).
//
// Division of state:
// - The lane owns the driver and the `channels` registry set (only control
//   execution ever touched it).
// - The worker keeps ALL policy and admission state: the quorum, `pending`,
//   `instances` (read by launch admission and gather on every pass). Control
//   arms that used to mutate `instances` inline are split: the driver half
//   runs on the lane, and the map mutation + response happen back on the
//   worker when the lane's reply arrives (`apply_lane_reply`) — keeping the
//   invariant that a bind's response is sent only AFTER the instance is
//   admissible, on the same thread that admits launches.
// - Replies ride the scheduler's own channel (`SchedulerItem::Lane`), so a
//   reply wakes the worker exactly like any other event.

/// A [`LaunchSubmission`] in transit to the driver lane.
///
/// SAFETY: the submission is `!Send` only through its
/// `Vec<*mut PieTerminalCell>` — raw pointers into the driver's pinned
/// terminal-cell slots, which are process-stable allocations with no thread
/// affinity (the driver itself reads them from its own threads today). The
/// submission is built complete on the worker, moved to the lane, and
/// consumed exactly once by `driver.launch` — the same single-consumer
/// discipline as the worker-inline call this replaces, with the backing
/// requests kept alive in `in_flight_launches` until the wave retires
/// (retire happens strictly after the lane's reply).
struct LaneLaunch(crate::driver::LaunchSubmission);
unsafe impl Send for LaneLaunch {}

/// Worker → lane requests, executed strictly in FIFO order.
enum LaneRequest {
    Launch {
        token: u64,
        submission: LaneLaunch,
        lease: Option<LaunchLease>,
    },
    Prepare {
        submission: LaneLaunch,
        response: crossbeam::channel::Sender<std::result::Result<LaunchPrepareOutcome, String>>,
    },
    Release {
        lease: LaunchLease,
        response: crossbeam::channel::Sender<std::result::Result<(), String>>,
    },
    /// A control `QueuedItem` (never `Launch`/`Prepare`): the lane runs the
    /// driver half of the old `dispatch_ordered_item` arm.
    Control { token: u64, item: QueuedItem },
    /// Drain marker: the lane replies with the driver and its channel set so
    /// the worker can run shutdown teardown with everything already quiesced.
    Shutdown {
        response: crossbeam::channel::Sender<(Option<DriverBackend>, HashSet<u64>)>,
    },
}

/// Lane → worker replies (via `SchedulerItem::Lane`).
enum LaneReply {
    LaunchDone {
        token: u64,
        result: std::result::Result<SubmissionCompletion, String>,
        driver_started_us: Option<u64>,
        launch_returned_us: Option<u64>,
    },
    ControlDone {
        token: u64,
        commit: LaneCommit,
    },
}

/// The worker-side half of a control that the lane finished executing.
enum LaneCommit {
    /// Nothing to commit — the lane already sent the response (pure driver
    /// ops that touch no worker state: program/channel registers, channel
    /// closes, failed binds after lane-side rollback).
    None,
    /// A successful bind: insert the instance, THEN respond (launch admission
    /// reads `instances` on the worker thread, so respond-after-insert is the
    /// ordering that makes the guest's first fire admissible).
    BindInstance {
        pipeline_id: Option<ProcessId>,
        bound: BoundInstance,
        respond: BindRespond,
    },
    /// A bind control completed without creating an instance.
    BindFinished { pipeline_id: Option<ProcessId> },
    /// A successful driver-side instance close: remove + close wait slots.
    CloseInstance { id: u64 },
    /// An async-completing control (copies / pool resizes): install the
    /// driver's completion into the pending control slot, or clear the slot
    /// on a synchronous driver rejection.
    AsyncControl {
        result: std::result::Result<SubmissionCompletion, String>,
    },
}

/// Which response shape a successful bind commits to.
enum BindRespond {
    Bind(tokio::sync::oneshot::Sender<Result<BoundInstance>>),
    ChannelsBind {
        registered: Vec<RegisteredChannel>,
        program_id: u64,
        program_registered: bool,
        response:
            tokio::sync::oneshot::Sender<Result<(Vec<RegisteredChannel>, u64, BoundInstance)>>,
    },
}

struct DriverLane {
    /// Launch fast path: served before any queued control. A queued launch
    /// and a queued control are ALWAYS mutually independent — a close only
    /// posts once its instance is quiesced (in-flight counts from post), and
    /// a fire can only exist after its bind COMMITTED on the worker — so
    /// preferring launches never reorders a dependent pair. Without the
    /// split, control bursts (a prefix cohort turnover posts hundreds of
    /// closes + binds faster than the lane drains them) head-of-line block
    /// the wave train: measured +7 % prefix regression on the single-FIFO
    /// variant, gaps doubling while sched-lag stayed low.
    launch_tx: crossbeam::channel::Sender<LaneRequest>,
    control_tx: crossbeam::channel::Sender<LaneRequest>,
    thread: Option<std::thread::JoinHandle<()>>,
    admission_supported: bool,
}

impl DriverLane {
    fn spawn(
        driver_idx: usize,
        driver: Option<DriverBackend>,
        reply_tx: crossbeam::channel::Sender<SchedulerItem>,
        stats: Arc<SchedulerStats>,
    ) -> Self {
        let admission_supported = driver
            .as_ref()
            .is_some_and(DriverBackend::supports_elastic_admission);
        let (launch_tx, launch_rx) = crossbeam::channel::unbounded::<LaneRequest>();
        let (control_tx, control_rx) = crossbeam::channel::unbounded::<LaneRequest>();
        let thread = std::thread::Builder::new()
            .name(format!("pie-driver-{driver_idx}"))
            .spawn(move || Self::run(driver, launch_rx, control_rx, reply_tx, stats))
            .expect("spawn pie-driver lane thread");
        Self {
            launch_tx,
            control_tx,
            thread: Some(thread),
            admission_supported,
        }
    }

    fn post(&self, request: LaneRequest) {
        // The lane outlives every poster (shutdown joins it last); a send
        // failure means the lane thread panicked, which the join reports.
        let _ = match &request {
            LaneRequest::Launch { .. }
            | LaneRequest::Prepare { .. }
            | LaneRequest::Release { .. } => self.launch_tx.send(request),
            LaneRequest::Control { .. } | LaneRequest::Shutdown { .. } => {
                self.control_tx.send(request)
            }
        };
    }

    fn prepare(
        &self,
        submission: crate::driver::LaunchSubmission,
    ) -> std::result::Result<LaunchPrepareOutcome, String> {
        let (response, receiver) = crossbeam::channel::bounded(1);
        self.post(LaneRequest::Prepare {
            submission: LaneLaunch(submission),
            response,
        });
        receiver
            .recv()
            .unwrap_or_else(|_| Err("driver lane closed during launch preparation".to_string()))
    }

    fn release(&self, lease: LaunchLease) -> std::result::Result<(), String> {
        let (response, receiver) = crossbeam::channel::bounded(1);
        self.post(LaneRequest::Release { lease, response });
        receiver
            .recv()
            .unwrap_or_else(|_| Err("driver lane closed during lease release".to_string()))
    }

    /// Drain both queues and take the driver + channel set back for
    /// teardown. The worker only calls this with `lane_inflight == 0`, so
    /// both queues are empty and the Shutdown marker is the sole item.
    fn shutdown(&mut self) -> (Option<DriverBackend>, HashSet<u64>) {
        let (response_tx, response_rx) = crossbeam::channel::bounded(1);
        let _ = self.control_tx.send(LaneRequest::Shutdown {
            response: response_tx,
        });
        let state = response_rx
            .recv()
            .unwrap_or_else(|_| (None, HashSet::new()));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        state
    }

    /// Receive the next request, launches first. Blocks on both queues when
    /// idle; between waves (a cadence of idle gaps) queued controls drain,
    /// so control progress rides the wave rhythm instead of competing with
    /// it.
    fn next_request(
        launch_rx: &crossbeam::channel::Receiver<LaneRequest>,
        control_rx: &crossbeam::channel::Receiver<LaneRequest>,
    ) -> std::result::Result<LaneRequest, ()> {
        use crossbeam::channel::TryRecvError;
        // Stay HOT for one wave window after going empty before parking.
        // The lane hop sits on the enqueue-ahead path: a parked lane pays a
        // thread wake (µs–ms under the box's wake-burst contention) on
        // every submit, which measurably broke run-ahead pipelining
        // (81 % → ~30 % of transitions enqueued ahead on the parked
        // variant). At wave cadence the spin window always covers the next
        // post, so the submit hop costs a cache-hot poll instead; the lane
        // parks only after a full window of true idleness (between
        // requests / at shutdown). The bound reuses the scheduler's own
        // derived wave-window constant (PIE_SCHED_WAVE_WINDOW_US).
        let mut spin_until = Instant::now() + quorum::wave_window();
        loop {
            match launch_rx.try_recv() {
                Ok(request) => return Ok(request),
                // Both senders live in the worker's `DriverLane` handle and
                // drop together (the graceful path is the Shutdown marker):
                // drain what remains on the other queue, then stop.
                Err(TryRecvError::Disconnected) => {
                    return control_rx.try_recv().map_err(|_| ());
                }
                Err(TryRecvError::Empty) => {}
            }
            match control_rx.try_recv() {
                Ok(request) => return Ok(request),
                Err(TryRecvError::Disconnected) => {
                    return launch_rx.try_recv().map_err(|_| ());
                }
                Err(TryRecvError::Empty) => {}
            }
            if Instant::now() < spin_until {
                std::hint::spin_loop();
                continue;
            }
            let mut select = crossbeam::channel::Select::new();
            select.recv(launch_rx);
            select.recv(control_rx);
            // Only wait; the loop re-runs the launch-first try_recv order
            // (and the disconnect handling above) once something is ready,
            // with a fresh spin window.
            select.ready();
            spin_until = Instant::now() + quorum::wave_window();
        }
    }

    fn run(
        mut driver: Option<DriverBackend>,
        launch_rx: crossbeam::channel::Receiver<LaneRequest>,
        control_rx: crossbeam::channel::Receiver<LaneRequest>,
        reply_tx: crossbeam::channel::Sender<SchedulerItem>,
        stats: Arc<SchedulerStats>,
    ) {
        let mut channels: HashSet<u64> = HashSet::new();
        while let Ok(request) = Self::next_request(&launch_rx, &control_rx) {
            match request {
                LaneRequest::Launch {
                    token,
                    submission,
                    lease,
                } => {
                    let LaneLaunch(submission) = submission;
                    let timing_enabled = super::fire_timing_enabled();
                    let driver_started_us = timing_enabled.then(super::fire_timing_now_us);
                    let result = match driver.as_mut() {
                        Some(driver) => crate::probe_fire!(
                            stats.fire.execute.driver_fire_us,
                            match lease {
                                Some(lease) => {
                                    driver.launch_prepared(&submission, lease)
                                }
                                None => driver.launch(&submission),
                            }
                        )
                        .map_err(|err| format!("{err:#}")),
                        None => Err("driver has no backend installed".to_string()),
                    };
                    let launch_returned_us = timing_enabled.then(super::fire_timing_now_us);
                    let _ = reply_tx.send(SchedulerItem::Lane(LaneReply::LaunchDone {
                        token,
                        result,
                        driver_started_us,
                        launch_returned_us,
                    }));
                }
                LaneRequest::Prepare {
                    submission,
                    response,
                } => {
                    let LaneLaunch(submission) = submission;
                    let result = driver
                        .as_mut()
                        .ok_or_else(|| "driver has no backend installed".to_string())
                        .and_then(|driver| {
                            driver
                                .prepare_launch(&submission)
                                .map_err(|error| format!("{error:#}"))
                        });
                    let _ = response.send(result);
                }
                LaneRequest::Release { lease, response } => {
                    let result = driver
                        .as_mut()
                        .ok_or_else(|| "driver has no backend installed".to_string())
                        .and_then(|driver| {
                            driver
                                .release_launch(lease)
                                .map_err(|error| format!("{error:#}"))
                        });
                    let _ = response.send(result);
                }
                LaneRequest::Control { token, item } => {
                    let control_timing = super::fire_timing_full().then(|| {
                        (
                            BatchScheduler::item_kind(&item),
                            super::fire_timing_now_us(),
                        )
                    });
                    let commit = Self::execute_control(&mut driver, &mut channels, item);
                    if let Some((kind, started_us)) = control_timing {
                        let finished_us = super::fire_timing_now_us();
                        super::fire_timing_write(&serde_json::json!({
                            "schema": 1,
                            "source": "scheduler",
                            "event": "control_dispatched",
                            "kind": kind,
                            "started_us": started_us,
                            "occupancy_us": finished_us.saturating_sub(started_us),
                        }));
                    }
                    let _ = reply_tx.send(SchedulerItem::Lane(LaneReply::ControlDone {
                        token,
                        commit,
                    }));
                }
                LaneRequest::Shutdown { response } => {
                    let _ = response.send((driver.take(), std::mem::take(&mut channels)));
                    return;
                }
            }
        }
        // Worker dropped its sender without a shutdown handshake (panic
        // path): release the driver here.
        drop(driver.take());
    }

    /// The driver half of the old `dispatch_ordered_item`: everything a
    /// control does against the driver and the lane-owned `channels` set,
    /// with worker-map effects returned as a [`LaneCommit`]. Failures respond
    /// directly from here (after lane-side rollback) — only effects that
    /// must be ordered with worker state travel back.
    fn execute_control(
        driver: &mut Option<DriverBackend>,
        channels: &mut HashSet<u64>,
        item: QueuedItem,
    ) -> LaneCommit {
        match item {
            QueuedItem::Launch(_) => unreachable!(),
            QueuedItem::PreLaunchCopy {
                plan: _,
                logical_completion,
                ..
            } if logical_completion.is_settled() => LaneCommit::AsyncControl {
                result: Err("pre-launch copy already settled".to_string()),
            },
            QueuedItem::PreLaunchCopy {
                plan: _,
                logical_completion,
                ..
            } if logical_completion.cancel_requested() => {
                logical_completion
                    .reject_unsubmitted("logical fire cancelled before pre-launch copy");
                LaneCommit::AsyncControl {
                    result: Err("logical fire cancelled before pre-launch copy".to_string()),
                }
            }
            QueuedItem::PreLaunchCopy {
                plan,
                logical_completion,
                ..
            } => {
                let operation = plan.label();
                match driver.as_mut() {
                    Some(driver) => {
                        let submitted = match plan {
                            PreLaunchCopy::Kv(plan) => driver.copy_kv(&plan),
                            PreLaunchCopy::State(plan) => driver.copy_state(&plan),
                        };
                        match submitted {
                            Ok(completion) => LaneCommit::AsyncControl {
                                result: Ok(completion),
                            },
                            Err(error) => {
                                let message = format!("pre-launch {operation} rejected: {error:#}");
                                logical_completion.reject_unsubmitted(message.clone());
                                LaneCommit::AsyncControl {
                                    result: Err(message),
                                }
                            }
                        }
                    }
                    None => {
                        logical_completion.reject_unsubmitted("driver has no backend installed");
                        LaneCommit::AsyncControl {
                            result: Err("driver has no backend installed".to_string()),
                        }
                    }
                }
            }
            QueuedItem::RegisterProgram { plan, response } => {
                if response.is_closed() {
                    tracing::warn!(
                        operation = "register_program",
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::None;
                }
                let result = match driver.as_mut() {
                    Some(driver) => driver.register_program(&plan),
                    None => Err(anyhow!("driver has no backend installed")),
                };
                match result {
                    Ok(program_id) => {
                        if response.send(Ok(program_id)).is_err() {
                            tracing::warn!(
                                operation = "register_program",
                                program_hash = format_args!("0x{:016x}", plan.program_hash),
                                "scheduler RPC cancelled after program registration; retaining driver-lifetime program"
                            );
                        }
                    }
                    Err(error) => {
                        let _ = response.send(Err(error));
                    }
                }
                LaneCommit::None
            }
            QueuedItem::RegisterChannel { plan, response } => {
                if response.is_closed() {
                    Self::release_channel_plan_wait_slots(std::slice::from_ref(&plan));
                    tracing::warn!(
                        operation = "register_channel",
                        channel_id = plan.channel_id,
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::None;
                }
                let result = if channels.contains(&plan.channel_id) {
                    Err(anyhow!("channel {} is already registered", plan.channel_id))
                } else {
                    match driver.as_mut() {
                        Some(driver) => driver.register_channel(&plan).map(|channel| {
                            channels.insert(plan.channel_id);
                            channel
                        }),
                        None => Err(anyhow!("driver has no backend installed")),
                    }
                };
                match result {
                    Ok(channel) => {
                        if let Err(Ok(channel)) = response.send(Ok(channel)) {
                            if let Some(driver) = driver.as_mut() {
                                Self::rollback_channel_set(
                                    driver,
                                    channels,
                                    std::slice::from_ref(&channel),
                                    "register_channel",
                                    true,
                                );
                            }
                            Self::release_registered_channel_wait_slots(std::slice::from_ref(
                                &channel,
                            ));
                        }
                    }
                    Err(error) => {
                        if response.send(Err(error)).is_err() {
                            Self::release_channel_plan_wait_slots(std::slice::from_ref(&plan));
                        }
                    }
                }
                LaneCommit::None
            }
            QueuedItem::RegisterChannels { plans, response } => {
                if response.is_closed() {
                    Self::release_channel_plan_wait_slots(&plans);
                    tracing::warn!(
                        operation = "register_channels",
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::None;
                }
                let result = match driver.as_mut() {
                    Some(driver) => Self::register_channel_set(driver, channels, &plans),
                    None => Err(anyhow!("driver has no backend installed")),
                };
                match result {
                    Ok(registered) => {
                        if let Err(Ok(registered)) = response.send(Ok(registered)) {
                            if let Some(driver) = driver.as_mut() {
                                Self::rollback_channel_set(
                                    driver,
                                    channels,
                                    &registered,
                                    "register_channels",
                                    true,
                                );
                            }
                            Self::release_registered_channel_wait_slots(&registered);
                        }
                    }
                    Err(error) => {
                        if response.send(Err(error)).is_err() {
                            Self::release_channel_plan_wait_slots(&plans);
                        }
                    }
                }
                LaneCommit::None
            }
            QueuedItem::BindInstance {
                pipeline_id,
                plan,
                response,
            } => {
                if response.is_closed() {
                    DriverLane::release_wait_slots([plan.pacing_wait_id]);
                    tracing::warn!(
                        operation = "bind_instance",
                        requested_instance_id = plan.requested_instance_id,
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::BindFinished { pipeline_id };
                }
                match driver.as_mut() {
                    Some(driver) => match driver.bind_instance(&plan) {
                        Ok(bound) => LaneCommit::BindInstance {
                            pipeline_id,
                            bound,
                            respond: BindRespond::Bind(response),
                        },
                        Err(error) => {
                            if response.send(Err(error)).is_err() {
                                DriverLane::release_wait_slots([plan.pacing_wait_id]);
                            }
                            LaneCommit::BindFinished { pipeline_id }
                        }
                    },
                    None => {
                        if response
                            .send(Err(anyhow!("driver has no backend installed")))
                            .is_err()
                        {
                            Self::release_wait_slots([plan.pacing_wait_id]);
                        }
                        LaneCommit::BindFinished { pipeline_id }
                    }
                }
            }
            QueuedItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                program,
                mut bind,
                response,
            } => {
                if response.is_closed() {
                    DriverLane::release_channel_plan_wait_slots(&plans);
                    DriverLane::release_wait_slots([bind.pacing_wait_id]);
                    tracing::warn!(
                        operation = "register_channels_bind",
                        requested_instance_id = bind.requested_instance_id,
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::BindFinished { pipeline_id };
                }
                let Some(driver) = driver.as_mut() else {
                    if response
                        .send(Err(anyhow!("driver has no backend installed")))
                        .is_err()
                    {
                        DriverLane::release_channel_plan_wait_slots(&plans);
                        DriverLane::release_wait_slots([bind.pacing_wait_id]);
                    }
                    return LaneCommit::BindFinished { pipeline_id };
                };
                let registered = match Self::register_channel_set(driver, channels, &plans) {
                    Ok(registered) => registered,
                    Err(error) => {
                        if response.send(Err(error)).is_err() {
                            Self::release_channel_plan_wait_slots(&plans);
                            Self::release_wait_slots([bind.pacing_wait_id]);
                        }
                        return LaneCommit::BindFinished { pipeline_id };
                    }
                };
                if response.is_closed() {
                    Self::rollback_channel_set(
                        driver,
                        channels,
                        &registered,
                        "register_channels_bind",
                        true,
                    );
                    DriverLane::release_registered_channel_wait_slots(&registered);
                    Self::release_wait_slots([bind.pacing_wait_id]);
                    return LaneCommit::BindFinished { pipeline_id };
                }
                let program_registered = program.is_some();
                if let Some(plan) = &program {
                    match driver.register_program(plan) {
                        Ok(program_id) => bind.program_id = program_id,
                        Err(error) => {
                            Self::rollback_channel_set(
                                driver,
                                channels,
                                &registered,
                                "register_channels_bind",
                                false,
                            );
                            if response.send(Err(error)).is_err() {
                                DriverLane::release_registered_channel_wait_slots(&registered);
                                Self::release_wait_slots([bind.pacing_wait_id]);
                            }
                            return LaneCommit::BindFinished { pipeline_id };
                        }
                    }
                }
                if response.is_closed() {
                    Self::rollback_channel_set(
                        driver,
                        channels,
                        &registered,
                        "register_channels_bind",
                        true,
                    );
                    Self::release_registered_channel_wait_slots(&registered);
                    Self::release_wait_slots([bind.pacing_wait_id]);
                    if program_registered {
                        tracing::warn!(
                            operation = "register_channels_bind",
                            program_id = bind.program_id,
                            "scheduler RPC cancelled after program registration; retaining driver-lifetime program"
                        );
                    }
                    return LaneCommit::BindFinished { pipeline_id };
                }
                match driver.bind_instance(&bind) {
                    Ok(bound) => LaneCommit::BindInstance {
                        pipeline_id,
                        bound,
                        respond: BindRespond::ChannelsBind {
                            registered,
                            program_id: bind.program_id,
                            program_registered,
                            response,
                        },
                    },
                    Err(error) => {
                        Self::rollback_channel_set(
                            driver,
                            channels,
                            &registered,
                            "register_channels_bind",
                            false,
                        );
                        if response.send(Err(error)).is_err() {
                            Self::release_registered_channel_wait_slots(&registered);
                            Self::release_wait_slots([bind.pacing_wait_id]);
                        }
                        LaneCommit::BindFinished { pipeline_id }
                    }
                }
            }
            QueuedItem::CopyKv { plan, response } => match driver.as_mut() {
                Some(driver) => match driver.copy_kv(&plan) {
                    Ok(completion) => {
                        let _ = response.send(Ok(completion.clone()));
                        LaneCommit::AsyncControl {
                            result: Ok(completion),
                        }
                    }
                    Err(err) => {
                        let message = format!("{err:#}");
                        let _ = response.send(Err(err));
                        LaneCommit::AsyncControl {
                            result: Err(message),
                        }
                    }
                },
                None => {
                    let _ = response.send(Err(anyhow!("driver has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("driver has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::CopyKvTracked { plan, completion } => match driver.as_mut() {
                Some(driver) => match driver.copy_kv(&plan) {
                    Ok(native_completion) => LaneCommit::AsyncControl {
                        result: Ok(native_completion),
                    },
                    Err(error) => {
                        let message = format!("{error:#}");
                        completion.resolve(&Err(error));
                        LaneCommit::AsyncControl {
                            result: Err(message),
                        }
                    }
                },
                None => {
                    completion.resolve(&Err(anyhow!("driver has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("driver has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::CopyState { plan, response } => match driver.as_mut() {
                Some(driver) => match driver.copy_state(&plan) {
                    Ok(completion) => {
                        let _ = response.send(Ok(completion.clone()));
                        LaneCommit::AsyncControl {
                            result: Ok(completion),
                        }
                    }
                    Err(err) => {
                        let message = format!("{err:#}");
                        let _ = response.send(Err(err));
                        LaneCommit::AsyncControl {
                            result: Err(message),
                        }
                    }
                },
                None => {
                    let _ = response.send(Err(anyhow!("driver has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("driver has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::ResizePool { plan, response } => match driver.as_mut() {
                Some(driver) => match driver.resize_pool(&plan) {
                    Ok(completion) => {
                        let _ = response.send(Ok(completion.clone()));
                        LaneCommit::AsyncControl {
                            result: Ok(completion),
                        }
                    }
                    Err(err) => {
                        let message = format!("{err:#}");
                        let _ = response.send(Err(err));
                        LaneCommit::AsyncControl {
                            result: Err(message),
                        }
                    }
                },
                None => {
                    let _ = response.send(Err(anyhow!("driver has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("driver has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::CloseInstance { id, .. } => match driver.as_mut() {
                // The worker already gated existence/pacing/quiescence before
                // posting; the map removal happens at commit.
                Some(driver) => match driver.close_instance(id) {
                    Ok(()) => LaneCommit::CloseInstance { id },
                    Err(err) => {
                        tracing::warn!(instance_id = id, ?err, "scheduler close_instance failed");
                        LaneCommit::None
                    }
                },
                None => {
                    tracing::warn!(instance_id = id, "scheduler has no backend installed");
                    LaneCommit::None
                }
            },
            QueuedItem::CloseChannel { id } => {
                let result = if !channels.contains(&id) {
                    Err(anyhow!("channel {id} is unknown or stale"))
                } else {
                    match driver.as_mut() {
                        Some(driver) => driver.close_channel(id).map(|()| {
                            channels.remove(&id);
                        }),
                        None => Err(anyhow!("driver has no backend installed")),
                    }
                };
                if let Err(err) = result {
                    tracing::warn!(channel_id = id, ?err, "scheduler close_channel failed");
                }
                LaneCommit::None
            }
        }
    }

    /// Register a set of channels with all-or-nothing rollback (the shared
    /// body of `RegisterChannels` and `RegisterChannelsBind`).
    fn register_channel_set(
        driver: &mut DriverBackend,
        channels: &mut HashSet<u64>,
        plans: &[ChannelRegistrationPlan],
    ) -> Result<Vec<RegisteredChannel>> {
        let mut registered = Vec::with_capacity(plans.len());
        let mut registered_ids = Vec::with_capacity(plans.len());
        for plan in plans {
            if channels.contains(&plan.channel_id) {
                for channel_id in registered_ids.iter().rev() {
                    let _ = driver.close_channel(*channel_id);
                    channels.remove(channel_id);
                }
                return Err(anyhow!("channel {} is already registered", plan.channel_id));
            }
            match driver.register_channel(plan) {
                Ok(channel) => {
                    channels.insert(plan.channel_id);
                    registered_ids.push(plan.channel_id);
                    registered.push(channel);
                }
                Err(cause) => {
                    for channel_id in registered_ids.iter().rev() {
                        let _ = driver.close_channel(*channel_id);
                        channels.remove(channel_id);
                    }
                    return Err(cause);
                }
            }
        }
        Ok(registered)
    }

    fn rollback_channel_set(
        driver: &mut DriverBackend,
        channels: &mut HashSet<u64>,
        registered: &[RegisteredChannel],
        operation: &'static str,
        cancellation: bool,
    ) {
        for channel in registered.iter().rev() {
            let channel_id = channel.binding.channel_id;
            match driver.close_channel(channel_id) {
                Ok(()) => {
                    channels.remove(&channel_id);
                }
                Err(error) => {
                    tracing::error!(
                        operation,
                        cancellation,
                        channel_id,
                        ?error,
                        "scheduler cancellation rollback close_channel failed"
                    );
                }
            }
        }
        tracing::warn!(
            operation,
            cancellation,
            channel_count = registered.len(),
            "scheduler registration rollback closed registered channels"
        );
    }

    fn release_channel_plan_wait_slots(plans: &[ChannelRegistrationPlan]) {
        Self::release_wait_slots(
            plans
                .iter()
                .flat_map(|plan| [plan.reader_wait_id, plan.writer_wait_id]),
        );
    }

    fn release_registered_channel_wait_slots(registered: &[RegisteredChannel]) {
        Self::release_wait_slots(
            registered
                .iter()
                .flat_map(|channel| [channel.reader_wait_id, channel.writer_wait_id]),
        );
    }

    fn release_wait_slots(wait_ids: impl IntoIterator<Item = u64>) {
        let wait_ids: Vec<u64> = wait_ids.into_iter().collect();
        let table = pie_waker::WakerTable::global();
        table.sweep(&wait_ids);
        for wait_id in wait_ids {
            table.deregister(wait_id);
            table.free(wait_id);
        }
    }
}

enum QueuedItem {
    Launch(PendingRequest),
    PreLaunchCopy {
        plan: PreLaunchCopy,
        logical_completion: WorkItemCompletion,
        process_id: Option<ProcessId>,
        pipeline_id: Option<ProcessId>,
        credit_ready: bool,
        /// The coupled launch's quorum identity stamp (W1): the credit this
        /// copy publishes at retire belongs to that launch, so it must
        /// route through the same generation check.
        quorum_generation: u64,
    },
    RegisterProgram {
        plan: ProgramRegistration,
        response: tokio::sync::oneshot::Sender<Result<u64>>,
    },
    RegisterChannel {
        plan: ChannelRegistrationPlan,
        response: tokio::sync::oneshot::Sender<Result<RegisteredChannel>>,
    },
    RegisterChannels {
        plans: Vec<ChannelRegistrationPlan>,
        response: tokio::sync::oneshot::Sender<Result<Vec<RegisteredChannel>>>,
    },
    BindInstance {
        pipeline_id: Option<ProcessId>,
        plan: InstanceBindingPlan,
        response: tokio::sync::oneshot::Sender<Result<BoundInstance>>,
    },
    /// One dispatch registering an instance's channels AND binding it —
    /// the two per-join controls always run back-to-back with only an
    /// ordering dependency, and dispatching them separately doubled the
    /// turnover control convoy (V6 iteration 25 attribution).
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistrationPlan>,
        /// Some on the program cache's first sight (the driver requires the
        /// instance's channels registered BEFORE the program — status -5
        /// otherwise — so registration must ride between channels and bind
        /// inside the one dispatch); None when the hash is already
        /// registered, with `bind.program_id` carrying the cached id.
        program: Option<ProgramRegistration>,
        bind: InstanceBindingPlan,
        response:
            tokio::sync::oneshot::Sender<Result<(Vec<RegisteredChannel>, u64, BoundInstance)>>,
    },
    CopyKv {
        plan: crate::driver::KvCopyPlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CopyKvTracked {
        plan: crate::driver::KvCopyPlan,
        completion: ControlCompletion,
    },
    CopyState {
        plan: StateCopyPlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    ResizePool {
        plan: PoolResizePlan,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CloseInstance {
        id: u64,
        pacing_wait_id: u64,
    },
    CloseChannel {
        id: u64,
    },
}

#[derive(Clone, Copy)]
enum QueueEnd {
    Front,
    Back,
}

/// A posted launch's lane lifecycle: the batch enters `in_flight_launches`
/// (and the run-ahead depth) at POST; the driver's verdict arrives as a
/// `LaneReply::LaunchDone` and upgrades the state. Retirement only ever
/// consumes `Accepted` (settled) or `Failed` heads — a `Posted` head is
/// simply not ready yet.
enum LaunchState {
    Posted { token: u64 },
    Accepted(SubmissionCompletion),
    Failed(String),
}

struct PendingLaunchBatch {
    state: LaunchState,
    requests: Vec<PendingRequest>,
    pipeline_epochs: Vec<Option<quorum::PipelineEpoch>>,
    started: Instant,
    batch_size: u64,
    total_tokens: usize,
    timing: Option<WaveTimingState>,
}

/// The control slot's lane lifecycle (async-completing controls only —
/// copies and pool resizes; lifecycle controls never occupy the slot).
enum ControlSlotState {
    Posted { token: u64 },
    Ready(SubmissionCompletion),
}

struct PendingControl {
    state: ControlSlotState,
    logical_completion: Option<WorkItemCompletion>,
    process_id: Option<ProcessId>,
    pipeline_id: Option<ProcessId>,
    tracked_completion: Option<ControlCompletion>,
    operation: &'static str,
    credit_ready: bool,
    /// See `QueuedItem::PreLaunchCopy::quorum_generation` (W1).
    quorum_generation: u64,
}

struct SchedulerControl {
    tx: crossbeam::channel::Sender<SchedulerItem>,
    active_senders: AtomicUsize,
    shutdown_wait: Condvar,
    shutdown_gate: Mutex<()>,
    program_ids: Mutex<HashMap<u64, (u64, Vec<u8>, Vec<u8>)>>,
    accepting: AtomicBool,
    stats: Arc<SchedulerStats>,
}

#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    inner: Arc<SchedulerControl>,
}

impl SchedulerHandle {
    fn send(&self, item: SchedulerItem) -> Result<()> {
        if !self.inner.accepting.load(Ordering::SeqCst) {
            return Err(anyhow!("scheduler shutting down"));
        }
        self.inner.active_senders.fetch_add(1, Ordering::SeqCst);
        if !self.inner.accepting.load(Ordering::SeqCst) {
            self.finish_send();
            return Err(anyhow!("scheduler shutting down"));
        }
        let result = self
            .inner
            .tx
            .send(item)
            .map_err(|_| anyhow!("scheduler channel closed"));
        self.finish_send();
        result
    }

    fn finish_send(&self) {
        if self.inner.active_senders.fetch_sub(1, Ordering::SeqCst) == 1
            && !self.inner.accepting.load(Ordering::SeqCst)
        {
            let _guard = self.inner.shutdown_gate.lock().unwrap();
            self.inner.shutdown_wait.notify_all();
        }
    }

    fn begin_shutdown(&self) {
        if !self.inner.accepting.swap(false, Ordering::SeqCst) {
            return;
        }
        let mut guard = self.inner.shutdown_gate.lock().unwrap();
        while self.inner.active_senders.load(Ordering::SeqCst) != 0 {
            guard = self.inner.shutdown_wait.wait(guard).unwrap();
        }
        let _ = self.inner.tx.send(SchedulerItem::Stop);
    }

    async fn request<T>(
        &self,
        make: impl FnOnce(tokio::sync::oneshot::Sender<T>) -> SchedulerItem,
    ) -> Result<T> {
        let (response, receiver) = tokio::sync::oneshot::channel();
        self.send(make(response))?;
        receiver
            .await
            .map_err(|_| anyhow!("scheduler channel closed"))
    }

    /// This driver's lock-free stats snapshot (read by
    /// `scheduler::get_stats`'s cross-driver aggregation).
    pub(crate) fn stats(&self) -> Arc<SchedulerStats> {
        Arc::clone(&self.inner.stats)
    }

    pub fn submit_with_identity_and_copy(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        last_page_len: u32,
        pipeline_id: Option<ProcessId>,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
        timing_enabled: bool,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                last_page_len,
                pipeline_id,
                pipeline_id,
                false,
                prelaunch_copy,
                prelaunch_state_copy,
                None,
                timing_enabled,
            ),
        })
    }

    pub fn submit_prebuilt_with_copy(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        last_page_len: u32,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                last_page_len,
                None,
                None,
                true,
                prelaunch_copy,
                prelaunch_state_copy,
                None,
                super::fire_timing_full(),
            ),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn submit_prebuilt_tracked_with_copy(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        last_page_len: u32,
        process_id: ProcessId,
        pipeline_id: ProcessId,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
        retry_classifier: Option<RetryClassifier>,
        timing_enabled: bool,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                last_page_len,
                Some(process_id),
                Some(pipeline_id),
                true,
                prelaunch_copy,
                prelaunch_state_copy,
                retry_classifier,
                timing_enabled,
            ),
        })
    }

    pub(crate) fn nudge(&self) -> Result<()> {
        self.send(SchedulerItem::Nudge)
    }

    pub(crate) async fn freeze_pipeline(&self, pid: ProcessId) -> Result<()> {
        self.request(|response| SchedulerItem::FreezePipeline { pid, response })
            .await
    }

    pub(crate) fn resume_pipeline(&self, pid: ProcessId) -> Result<()> {
        self.send(SchedulerItem::ResumePipeline(pid))
    }

    pub async fn register_program(&self, plan: ProgramRegistration) -> Result<u64> {
        let program_hash = plan.program_hash;
        {
            let program_ids = self.inner.program_ids.lock().unwrap();
            if let Some((program_id, canonical, sidecar)) = program_ids.get(&program_hash) {
                if canonical != &plan.canonical_bytes || sidecar != &plan.sidecar_bytes {
                    return Err(anyhow!("program hash collision for 0x{program_hash:016x}"));
                }
                return Ok(*program_id);
            }
        }
        let canonical = plan.canonical_bytes.clone();
        let sidecar = plan.sidecar_bytes.clone();
        let program_id = self
            .request(|response| SchedulerItem::RegisterProgram { plan, response })
            .await??;
        self.inner
            .program_ids
            .lock()
            .unwrap()
            .insert(program_hash, (program_id, canonical, sidecar));
        Ok(program_id)
    }

    pub async fn register_channel(
        &self,
        plan: ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        self.request(|response| SchedulerItem::RegisterChannel { plan, response })
            .await?
    }

    pub async fn register_channels(
        &self,
        plans: Vec<ChannelRegistrationPlan>,
    ) -> Result<Vec<RegisteredChannel>> {
        self.request(|response| SchedulerItem::RegisterChannels { plans, response })
            .await?
    }

    pub async fn bind_instance(
        &self,
        pipeline_id: Option<ProcessId>,
        plan: InstanceBindingPlan,
    ) -> Result<BoundInstance> {
        self.request(|response| SchedulerItem::BindInstance {
            pipeline_id,
            plan,
            response,
        })
        .await?
    }

    pub async fn register_channels_bind(
        &self,
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistrationPlan>,
        program: ProgramRegistration,
        mut bind: InstanceBindingPlan,
    ) -> Result<(Vec<RegisteredChannel>, BoundInstance)> {
        let program_hash = program.program_hash;
        let cached = {
            let program_ids = self.inner.program_ids.lock().unwrap();
            match program_ids.get(&program_hash) {
                Some((program_id, canonical, sidecar)) => {
                    if canonical != &program.canonical_bytes || sidecar != &program.sidecar_bytes {
                        return Err(anyhow!("program hash collision for 0x{program_hash:016x}"));
                    }
                    Some(*program_id)
                }
                None => None,
            }
        };
        let (program_field, cache_fill) = match cached {
            Some(program_id) => {
                bind.program_id = program_id;
                (None, None)
            }
            None => (
                Some(program.clone()),
                Some((program.canonical_bytes, program.sidecar_bytes)),
            ),
        };
        let (registered, program_id, bound) = self
            .request(|response| SchedulerItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                program: program_field,
                bind,
                response,
            })
            .await??;
        if let Some((canonical, sidecar)) = cache_fill {
            self.inner
                .program_ids
                .lock()
                .unwrap()
                .insert(program_hash, (program_id, canonical, sidecar));
        }
        Ok((registered, bound))
    }

    pub async fn copy_kv(&self, plan: crate::driver::KvCopyPlan) -> Result<SubmissionCompletion> {
        self.request(|response| SchedulerItem::CopyKv { plan, response })
            .await?
    }

    pub(crate) fn copy_kv_tracked(
        &self,
        plan: crate::driver::KvCopyPlan,
    ) -> Result<ControlCompletion> {
        let completion = ControlCompletion::new();
        self.send(SchedulerItem::CopyKvTracked {
            plan,
            completion: completion.clone(),
        })?;
        Ok(completion)
    }

    /// Human-readable snapshot of the run loop's state (see
    /// [`SchedulerItem::DebugDump`]).
    pub(crate) async fn debug_dump(&self) -> Result<String> {
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            self.request(|response| SchedulerItem::DebugDump { response }),
        )
        .await
        .map_err(|_| anyhow!("scheduler did not answer the debug dump"))?
    }

    // Only called from `scheduler::dispatch::copy_rs_d2d`/`resize_pool`
    // (not yet issued by the mock-driver fire path) and this module's own
    // unit tests — see `scheduler::dispatch`'s module doc.
    #[allow(dead_code)]
    pub async fn copy_state(&self, plan: StateCopyPlan) -> Result<SubmissionCompletion> {
        self.request(|response| SchedulerItem::CopyState { plan, response })
            .await?
    }

    #[allow(dead_code)]
    pub async fn resize_pool(&self, plan: PoolResizePlan) -> Result<SubmissionCompletion> {
        self.request(|response| SchedulerItem::ResizePool { plan, response })
            .await?
    }

    pub fn close_instance(&self, id: u64, pacing_wait_id: u64) -> Result<()> {
        self.send(SchedulerItem::CloseInstance { id, pacing_wait_id })
    }

    pub fn close_channel(&self, id: u64) -> Result<()> {
        self.send(SchedulerItem::CloseChannel { id })
    }
}

pub struct BatchScheduler {
    driver_id: DriverId,
    handle: SchedulerHandle,
    thread: Option<std::thread::JoinHandle<()>>,
    stats: Arc<SchedulerStats>,
}

impl BatchScheduler {
    pub fn new(
        driver_id: DriverId,
        driver_idx: usize,
        page_size: u32,
        limits: SchedulerLimits,
        request_timeout_secs: u64,
    ) -> Self {
        let (tx, rx) = crossbeam::channel::unbounded::<SchedulerItem>();
        let stats = Arc::new(SchedulerStats::default());
        let handle = SchedulerHandle {
            inner: Arc::new(SchedulerControl {
                tx,
                active_senders: AtomicUsize::new(0),
                shutdown_wait: Condvar::new(),
                shutdown_gate: Mutex::new(()),
                program_ids: Mutex::new(HashMap::new()),
                accepting: AtomicBool::new(true),
                stats: Arc::clone(&stats),
            }),
        };
        crate::scheduler::install_scheduler_handle(driver_id, handle.clone());
        let stats_for_loop = Arc::clone(&stats);
        let nudge_tx = handle.inner.tx.clone();
        let thread = std::thread::Builder::new()
            .name(format!("pie-sched-{driver_idx}"))
            .spawn(move || {
                let _request_timeout = Duration::from_secs(request_timeout_secs);
                Self::run(driver_id, rx, nudge_tx, page_size, limits, stats_for_loop);
            })
            .expect("spawn pie-sched thread");
        Self {
            driver_id,
            handle,
            thread: Some(thread),
            stats,
        }
    }

    pub fn stats(&self) -> &Arc<SchedulerStats> {
        &self.stats
    }

    fn shutdown(&mut self) {
        self.handle.begin_shutdown();
        crate::scheduler::clear_scheduler_handle(self.driver_id);
        if let Some(thread) = self.thread.take() {
            if let Err(err) = thread.join() {
                tracing::error!(
                    driver_id = self.driver_id,
                    ?err,
                    "scheduler thread panicked"
                );
            }
        }
    }

    fn run(
        driver_id: DriverId,
        rx: crossbeam::channel::Receiver<SchedulerItem>,
        nudge_tx: crossbeam::channel::Sender<SchedulerItem>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: Arc<SchedulerStats>,
    ) {
        let lane_reply_tx = nudge_tx.clone();
        let nudge_waker = std::task::Waker::from(Arc::new(NudgeWaker {
            tx: nudge_tx.clone(),
        }));
        let driver = crate::driver::take_driver_backend(driver_id).ok();
        let mut lane = DriverLane::spawn(driver_id, driver, lane_reply_tx, Arc::clone(&stats));
        // Worker→lane requests not yet replied to (launch posts + control
        // posts). Shutdown may only tear down once this drains — every lane
        // request produces exactly one reply.
        let mut lane_inflight: u64 = 0;
        let mut lane_token: u64 = 0;
        let mut instances = HashMap::new();
        let mut pending = VecDeque::new();
        let mut frozen_pipelines = HashSet::new();
        // Pipelines that LEFT the fleet (terminate / preempt) with work still
        // in flight. A protected straggler resolving later must ride the wave
        // untracked instead of re-arming the barrier on a ghost (RV-20); the
        // pipeline's own next request is its implicit rejoin.
        let mut departed_pipelines: HashSet<ProcessId> = HashSet::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = None;
        let mut admission_retry_at = None;
        let mut stopping = false;
        // The wait-for-all-active-pipelines fire rule (overview §7.2): one
        // instance per driver thread, mirroring `instances`/`channels` above.
        // Every backend schedules the same way; density comes from the wave,
        // throughput from run-ahead depth within it.
        let mut policy =
            quorum::WaitAllPolicy::new(limits.max_forward_requests, Some(Arc::clone(&stats)));
        // Stall self-diagnosis: a scheduler that spins on the backstop with
        // queued or in-flight work and zero progress is deadlocked from the
        // caller's point of view, and every wait in this loop is silent. After
        // 10s of that, print the full state dump so the wedge names itself
        // (then re-print every 60s while it persists).
        let mut stall_since: Option<std::time::Instant> = None;
        let mut stall_dumps: u32 = 0;

        loop {
            let mut progress = false;
            // Drain the scheduler channel FIRST: lane replies upgrade posted
            // waves to Accepted and posted controls to Ready, and readiness
            // credits land in `pending` — retiring and dispatching against a
            // fresh view saves one full pass of latency per wave, which at
            // decode cadence is the difference between an enqueued-ahead
            // launch and a GPU gap (the retire path breaks on a Posted head).
            while let Ok(item) = rx.try_recv() {
                progress = true;
                if let SchedulerItem::DebugDump { response } = item {
                    let _ = response.send(Self::render_debug_dump(
                        &pending,
                        &frozen_pipelines,
                        &departed_pipelines,
                        &in_flight_launches,
                        &in_flight_control,
                        &instances,
                        &policy,
                    ));
                    continue;
                }
                if let SchedulerItem::Lane(reply) = item {
                    Self::apply_lane_reply(
                        reply,
                        &mut lane_inflight,
                        &mut in_flight_launches,
                        &mut in_flight_control,
                        &mut instances,
                        &mut policy,
                        &nudge_tx,
                    );
                    continue;
                }
                Self::enqueue_item(
                    &mut pending,
                    &mut frozen_pipelines,
                    &mut departed_pipelines,
                    &mut in_flight_control,
                    &instances,
                    limits,
                    page_size,
                    &mut stopping,
                    &mut policy,
                    item,
                );
            }
            progress |= Self::retire_ready_launches(
                &mut in_flight_launches,
                &mut instances,
                &mut pending,
                &stats,
                &mut policy,
                stopping,
            );
            progress |=
                Self::retire_ready_control(&mut in_flight_control, &mut pending, &mut policy);
            let (dispatched, wait_hint) = Self::dispatch_ready_items(
                &lane,
                &mut lane_inflight,
                &mut lane_token,
                &mut instances,
                &mut pending,
                &mut in_flight_launches,
                &mut in_flight_control,
                &mut admission_retry_at,
                page_size,
                limits,
                &stats,
                &mut policy,
                stopping,
            );
            progress |= dispatched;
            if stopping
                && pending.is_empty()
                && in_flight_launches.is_empty()
                && in_flight_control.is_none()
                && lane_inflight == 0
            {
                break;
            }

            if progress {
                stall_since = None;
                stall_dumps = 0;
                continue;
            }

            let item = if pending.is_empty()
                && in_flight_launches.is_empty()
                && in_flight_control.is_none()
                && !stopping
            {
                match rx.recv() {
                    Ok(item) => Some(item),
                    Err(_) => {
                        stopping = true;
                        None
                    }
                }
            } else {
                // Event-driven retirement: park the nudge waker on the oldest
                // in-flight completions so the driver callback wakes this
                // thread the moment one publishes. The timeout is only a hang
                // backstop, never the steady-state wake path.
                let mut armed = true;
                if let Some(front) = in_flight_launches.front() {
                    match &front.state {
                        // A posted launch's reply arrives on the scheduler
                        // channel itself — recv() IS the wake path.
                        LaunchState::Posted { .. } => {}
                        LaunchState::Accepted(completion) => {
                            armed &= arm_completion_nudge(completion, &nudge_waker);
                        }
                        // A failed launch is retire-ready right now.
                        LaunchState::Failed(_) => armed = false,
                    }
                }
                if let Some(control) = in_flight_control.as_ref() {
                    match &control.state {
                        ControlSlotState::Posted { .. } => {}
                        ControlSlotState::Ready(completion) => {
                            armed &= arm_completion_nudge(completion, &nudge_waker);
                        }
                    }
                }
                if !armed {
                    // Something already settled; retire it on the next pass.
                    continue;
                }
                // A pending quorum hold (cold gather / wait-all barrier /
                // depth-cap poll) re-arms the backstop at its own
                // cadence — never longer than the 250ms hang backstop, so a
                // held wave still fires on time even with no new arrival or
                // completion nudge in between.
                let backstop = Duration::from_millis(250);
                let recv_wait = wait_hint.map(|hold| hold.min(backstop)).unwrap_or(backstop);
                match rx.recv_timeout(recv_wait) {
                    Ok(item) => Some(item),
                    Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                        // A settled completion discovered by the backstop
                        // means a wake was lost somewhere — the steady-state
                        // count stays zero (plan §16.2). Shutdown races are
                        // excluded: teardown may legitimately cross a tick.
                        // A quorum-hold timeout is NOT a lost wake (it is
                        // the wait's own cadence), so it never counts here.
                        let missed = in_flight_launches.front().is_some_and(|front| {
                            matches!(&front.state, LaunchState::Accepted(c) if c.is_settled())
                        }) || in_flight_control.as_ref().is_some_and(|control| {
                            matches!(&control.state, ControlSlotState::Ready(c) if c.is_settled())
                        });
                        if missed && !stopping && wait_hint.is_none() {
                            let total = BACKSTOP_RETIREMENTS.fetch_add(1, Ordering::Relaxed) + 1;
                            tracing::warn!(
                                driver_id,
                                total,
                                "completion retired by the backstop poll, not the nudge"
                            );
                        }
                        let stalled_for = stall_since
                            .get_or_insert_with(std::time::Instant::now)
                            .elapsed();
                        if stalled_for
                            >= Duration::from_secs(10)
                                .saturating_add(Duration::from_secs(60) * stall_dumps)
                        {
                            stall_dumps += 1;
                            eprintln!(
                                "[pie-sched] driver {driver_id} stalled for {stalled_for:?} \
                                 (no progress, work queued or in flight); state:\n{}",
                                Self::render_debug_dump(
                                    &pending,
                                    &frozen_pipelines,
                                    &departed_pipelines,
                                    &in_flight_launches,
                                    &in_flight_control,
                                    &instances,
                                    &policy,
                                ),
                            );
                        }
                        None
                    }
                    Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                        stopping = true;
                        None
                    }
                }
            };

            if let Some(item) = item {
                if let SchedulerItem::DebugDump { response } = item {
                    let _ = response.send(Self::render_debug_dump(
                        &pending,
                        &frozen_pipelines,
                        &departed_pipelines,
                        &in_flight_launches,
                        &in_flight_control,
                        &instances,
                        &policy,
                    ));
                    continue;
                }
                if let SchedulerItem::Lane(reply) = item {
                    Self::apply_lane_reply(
                        reply,
                        &mut lane_inflight,
                        &mut in_flight_launches,
                        &mut in_flight_control,
                        &mut instances,
                        &mut policy,
                        &nudge_tx,
                    );
                    continue;
                }
                Self::enqueue_item(
                    &mut pending,
                    &mut frozen_pipelines,
                    &mut departed_pipelines,
                    &mut in_flight_control,
                    &instances,
                    limits,
                    page_size,
                    &mut stopping,
                    &mut policy,
                    item,
                );
            }
        }

        // The lane has no pending requests here (`lane_inflight == 0` gates
        // the loop exit), so shutdown returns the quiesced driver and the
        // channel registry for teardown.
        let (mut driver, mut channels) = lane.shutdown();
        Self::shutdown_instances(&mut driver, &mut instances);
        Self::shutdown_channels(&mut driver, &mut channels);
        drop(driver.take());
    }

    #[allow(clippy::too_many_arguments)]
    fn render_debug_dump(
        pending: &VecDeque<QueuedItem>,
        frozen_pipelines: &HashSet<ProcessId>,
        departed_pipelines: &HashSet<ProcessId>,
        in_flight_launches: &VecDeque<PendingLaunchBatch>,
        in_flight_control: &Option<PendingControl>,
        instances: &HashMap<u64, TrackedInstance>,
        policy: &quorum::WaitAllPolicy,
    ) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let describe = |request: &PendingRequest| {
            format!(
                "fire {} instance {} pipeline {:?} retries {} tracked={} settled={} \
                 cancelled={} retry_wait={}",
                request.logical_fire_id,
                request.instance_id,
                request.pipeline_id,
                request.retry_count,
                instances.contains_key(&request.instance_id),
                request.completion.is_settled(),
                request.completion.cancel_requested(),
                request.retry_after.is_some_and(|due| due > Instant::now()),
            )
        };
        let _ = writeln!(out, "pending ({}):", pending.len());
        for item in pending {
            let line = match item {
                QueuedItem::Launch(request) => format!("Launch: {}", describe(request)),
                QueuedItem::PreLaunchCopy {
                    plan, pipeline_id, ..
                } => format!("PreLaunchCopy({}) pipeline {pipeline_id:?}", plan.label()),
                QueuedItem::RegisterProgram { .. } => "RegisterProgram".to_string(),
                QueuedItem::RegisterChannel { .. } => "RegisterChannel".to_string(),
                QueuedItem::RegisterChannels { plans, .. } => {
                    format!("RegisterChannels({})", plans.len())
                }
                QueuedItem::BindInstance { .. } => "BindInstance".to_string(),
                QueuedItem::RegisterChannelsBind { .. } => "RegisterChannelsBind".to_string(),
                QueuedItem::CopyKv { .. } => "CopyKv".to_string(),
                QueuedItem::CopyKvTracked { .. } => "CopyKvTracked".to_string(),
                QueuedItem::CopyState { .. } => "CopyState".to_string(),
                QueuedItem::ResizePool { .. } => "ResizePool".to_string(),
                QueuedItem::CloseInstance { id, .. } => format!("CloseInstance {id}"),
                QueuedItem::CloseChannel { id, .. } => format!("CloseChannel {id}"),
            };
            let _ = writeln!(out, "  {line}");
        }
        let _ = writeln!(out, "in_flight_launches ({}):", in_flight_launches.len());
        for batch in in_flight_launches {
            let state = match &batch.state {
                LaunchState::Posted { token } => format!("posted(token={token})"),
                LaunchState::Accepted(c) => format!("settled={}", c.is_settled()),
                LaunchState::Failed(msg) => format!("failed({msg})"),
            };
            let _ = writeln!(
                out,
                "  batch of {} ({state}, age={:?})",
                batch.requests.len(),
                batch.started.elapsed(),
            );
        }
        match in_flight_control {
            Some(control) => {
                let state = match &control.state {
                    ControlSlotState::Posted { token } => format!("posted(token={token})"),
                    ControlSlotState::Ready(c) => format!("settled={}", c.is_settled()),
                };
                let _ = writeln!(
                    out,
                    "in_flight_control: {} pipeline {:?} {state}",
                    control.operation, control.pipeline_id,
                );
            }
            None => {
                let _ = writeln!(out, "in_flight_control: none");
            }
        }
        let _ = writeln!(
            out,
            "frozen: {:?} departed: {:?}",
            frozen_pipelines, departed_pipelines,
        );
        let _ = write!(out, "quorum: {}", policy.debug_summary());
        out
    }

    fn enqueue_item(
        pending: &mut VecDeque<QueuedItem>,
        frozen_pipelines: &mut HashSet<ProcessId>,
        departed_pipelines: &mut HashSet<ProcessId>,
        in_flight_control: &mut Option<PendingControl>,
        instances: &HashMap<u64, TrackedInstance>,
        limits: SchedulerLimits,
        page_size: u32,
        stopping: &mut bool,
        policy: &mut quorum::WaitAllPolicy,
        item: SchedulerItem,
    ) {
        match item {
            SchedulerItem::Stop => {
                *stopping = true;
            }
            // Answered inline at both dequeue sites in `run` — it never
            // reaches this queue-mutating path.
            SchedulerItem::DebugDump { .. } => {
                unreachable!("DebugDump is intercepted before enqueue_item")
            }
            // A nudge only unblocks the wait; the retire pass at the top of
            // the loop does the work.
            SchedulerItem::Nudge => {}
            SchedulerItem::FreezePipeline { pid, response } => {
                frozen_pipelines.insert(pid);
                let _ = response.send(());
            }
            SchedulerItem::ResumePipeline(pid) => {
                frozen_pipelines.remove(&pid);
            }
            // Immediate, not queued. Termination rejects queued work; graceful
            // pipeline close instead releases the wait-set and lets every
            // already-admitted request drain untracked.
            SchedulerItem::PipelineLeave(pid, kind) => {
                if kind == LeaveKind::Terminate {
                    let protected = in_flight_control
                        .as_ref()
                        .filter(|control| control.process_id == Some(pid))
                        .and_then(|control| control.logical_completion.clone());
                    if let Some(completion) = &protected {
                        completion.request_cancel();
                    }
                    Self::reject_pipeline_queued(pending, policy, pid, protected.as_ref());
                    frozen_pipelines.remove(&pid);
                }
                if kind != LeaveKind::Close {
                    departed_pipelines.insert(pid);
                    policy.on_process_leave(pid);
                } else {
                    policy.on_pipeline_leave(pid);
                }
            }
            SchedulerItem::Launch {
                pending: mut launch,
            } => {
                if let Some(timing) = launch.timing.as_mut() {
                    timing.enqueued_us = Some(super::fire_timing_now_us());
                }
                // Quorum identity stamp: the request belongs to its
                // pipeline's CURRENT incarnation. Guest submissions strictly
                // precede the pipeline's Close on the same FIFO, so a
                // pre-close fire always stamps the pre-close generation and
                // routes untracked after the leave (see `quorum_pid`).
                if let Some(pid) = launch.pipeline_id {
                    launch.quorum_generation = policy.generation_of(pid);
                }
                let validation = BatchAccumulator::new(limits, page_size);
                let rejection = if launch.completion.cancel_requested() {
                    Some("logical fire cancelled before scheduler admission".to_string())
                } else if !instances.contains_key(&launch.instance_id) {
                    Some(format!(
                        "instance {} is unknown or stale",
                        launch.instance_id
                    ))
                } else if let Some(message) = validation.single_request_limit_error(&launch) {
                    Some(message)
                } else if *stopping {
                    Some("scheduler shutting down".to_string())
                } else {
                    None
                };
                if let Some(message) = rejection {
                    launch.completion.reject_unsubmitted(message);
                } else {
                    // The wave gather starts at acceptance, not dispatch:
                    // this request now counts toward `decide_wave_at`'s
                    // wait-set/untracked-ready even while it sits in
                    // `pending` behind an in-flight-depth or quorum hold.
                    if let Some(process_id) = launch.process_id {
                        departed_pipelines.remove(&process_id);
                    }
                    // No fire-admission early join: the readiness credit below
                    // creates the ordinary wait-set entry. Bind controls use
                    // separate assembly membership, which cannot age into
                    // demotion while the control remains pending; preparation
                    // alone still does not create a new member.
                    if !Self::request_needs_prelaunch(&launch) {
                        let qpid =
                            Self::quorum_pid(policy, launch.pipeline_id, launch.quorum_generation);
                        policy.on_pipeline_request_owned(qpid, launch.process_id, Instant::now());
                        launch.credit_published = true;
                        if let Some(timing) = launch.timing.as_mut() {
                            timing.ready_us = Some(super::fire_timing_now_us());
                        }
                    }
                    Self::queue_attempt(pending, launch, QueueEnd::Back);
                }
            }

            SchedulerItem::RegisterProgram { plan, response } => {
                pending.push_back(QueuedItem::RegisterProgram { plan, response });
            }
            SchedulerItem::RegisterChannel { plan, response } => {
                pending.push_back(QueuedItem::RegisterChannel { plan, response });
            }
            SchedulerItem::RegisterChannels { plans, response } => {
                pending.push_back(QueuedItem::RegisterChannels { plans, response });
            }
            SchedulerItem::BindInstance {
                pipeline_id,
                plan,
                response,
            } => {
                policy.on_bind_enqueued(pipeline_id);
                pending.push_back(QueuedItem::BindInstance {
                    pipeline_id,
                    plan,
                    response,
                });
            }
            SchedulerItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                program,
                bind,
                response,
            } => {
                policy.on_bind_enqueued(pipeline_id);
                pending.push_back(QueuedItem::RegisterChannelsBind {
                    pipeline_id,
                    plans,
                    program,
                    bind,
                    response,
                });
            }
            SchedulerItem::CopyKv { plan, response } => {
                pending.push_back(QueuedItem::CopyKv { plan, response });
            }
            SchedulerItem::CopyKvTracked { plan, completion } => {
                pending.push_back(QueuedItem::CopyKvTracked { plan, completion });
            }
            SchedulerItem::CopyState { plan, response } => {
                pending.push_back(QueuedItem::CopyState { plan, response });
            }
            SchedulerItem::ResizePool { plan, response } => {
                pending.push_back(QueuedItem::ResizePool { plan, response });
            }
            SchedulerItem::CloseInstance { id, pacing_wait_id } => {
                pending.push_back(QueuedItem::CloseInstance { id, pacing_wait_id });
            }
            SchedulerItem::CloseChannel { id } => {
                pending.push_back(QueuedItem::CloseChannel { id });
            }
            // Handled on dequeue in the run loop (like DebugDump) before
            // enqueue_item is reached.
            SchedulerItem::Lane(_) => unreachable!(),
        }
    }

    /// Exponential retry pacing, 20us doubling to a 1ms cap: enough to stop
    /// a lone pipeline from hammering a transient device condition, far below
    /// wave cadence, and small enough that the full retry budget
    /// (`PIE_FIRE_RETRY_MAX`, default 1024) exhausts in about a second.
    fn retry_backoff(retry_count: u32) -> Duration {
        Duration::from_micros((20u64 << retry_count.min(6)).min(1_000))
    }

    fn reject_pipeline_queued(
        pending: &mut VecDeque<QueuedItem>,
        policy: &mut quorum::WaitAllPolicy,
        pid: ProcessId,
        protected: Option<&WorkItemCompletion>,
    ) {
        let mut kept = VecDeque::with_capacity(pending.len());
        while let Some(item) = pending.pop_front() {
            let reject = match &item {
                QueuedItem::Launch(request) => {
                    request.process_id == Some(pid)
                        && protected
                            .is_none_or(|completion| !request.completion.same_request(completion))
                }
                QueuedItem::PreLaunchCopy {
                    process_id,
                    logical_completion,
                    ..
                } => {
                    *process_id == Some(pid)
                        && protected
                            .is_none_or(|completion| !logical_completion.same_request(completion))
                }
                _ => false,
            };
            if reject {
                match item {
                    QueuedItem::Launch(request) => {
                        Self::drop_request_credit(policy, &request);
                        request
                            .completion
                            .reject_unsubmitted("pipeline left while queued");
                    }
                    // A pre-launch copy is order-coupled to its consumer
                    // launch (one fire, one book entry — the Launch arm
                    // resolves it).
                    QueuedItem::PreLaunchCopy {
                        logical_completion, ..
                    } => logical_completion
                        .reject_unsubmitted("pipeline left before pre-launch copy"),
                    _ => unreachable!("rejected item kind checked above"),
                }
            } else {
                kept.push_back(item);
            }
        }
        *pending = kept;
    }

    fn queue_attempt(pending: &mut VecDeque<QueuedItem>, request: PendingRequest, end: QueueEnd) {
        let mut copies = Vec::with_capacity(2);
        if let Some(plan) = request.prelaunch_copy.clone() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::Kv(plan),
                logical_completion: request.completion.clone(),
                process_id: request.process_id,
                pipeline_id: request.pipeline_id,
                credit_ready: false,
                quorum_generation: request.quorum_generation,
            });
        }
        if let Some(plan) = request.prelaunch_state_copy.clone() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::State(plan),
                logical_completion: request.completion.clone(),
                process_id: request.process_id,
                pipeline_id: request.pipeline_id,
                credit_ready: false,
                quorum_generation: request.quorum_generation,
            });
        }
        if let Some(QueuedItem::PreLaunchCopy { credit_ready, .. }) = copies.last_mut() {
            *credit_ready = true;
        }
        match end {
            QueueEnd::Front => {
                pending.push_front(QueuedItem::Launch(request));
                for copy in copies.into_iter().rev() {
                    pending.push_front(copy);
                }
            }

            QueueEnd::Back => {
                for copy in copies {
                    pending.push_back(copy);
                }
                pending.push_back(QueuedItem::Launch(request));
            }
        }
    }

    fn request_needs_prelaunch(request: &PendingRequest) -> bool {
        request.prelaunch_copy.is_some() || request.prelaunch_state_copy.is_some()
    }

    /// Whether any queued fire still targets `instance_id` (a queued
    /// `PreLaunchCopy` is covered by its consumer launch queued behind it).
    /// Together with `TrackedInstance::in_flight` this is the close gate:
    /// an instance with neither queued nor in-flight work is quiesced.
    fn instance_has_queued_work(pending: &VecDeque<QueuedItem>, instance_id: u64) -> bool {
        pending.iter().any(|item| match item {
            QueuedItem::Launch(request) => request.instance_id == instance_id,
            _ => false,
        })
    }

    /// The pid this request's quorum accounting must use NOW: its
    /// `pipeline_id` while the pipeline's identity generation still matches
    /// the request's admission stamp, `None` (untracked) once the pipeline
    /// has left. Every publication, consumption, and drop site routes
    /// through this — the leak class it closes is a post-leave publication
    /// re-creating the scope's wait-set row and the straddling wave then
    /// consuming a later incarnation's credit.
    fn quorum_pid(
        policy: &quorum::WaitAllPolicy,
        pipeline_id: Option<ProcessId>,
        stamp: u64,
    ) -> Option<ProcessId> {
        pipeline_id.filter(|pid| policy.generation_of(*pid) == stamp)
    }

    /// Returns a dropped request's readiness credit to the barrier, if it
    /// holds one. A fire that never published (still awaiting its pre-launch
    /// copy) or whose credit a dispatched wave already consumed has nothing
    /// to give back — decrementing anyway corrupts the wave accounting by
    /// eating a sibling's credit (RV-20).
    fn drop_request_credit(policy: &mut quorum::WaitAllPolicy, request: &PendingRequest) {
        if request.credit_published {
            let qpid = Self::quorum_pid(policy, request.pipeline_id, request.quorum_generation);
            policy.on_request_dropped(qpid);
        }
    }

    /// A standalone KV/state copy: suspend D2H, restore H2D, graft/CAS
    /// copies. These touch pages no queued fire references (suspend takes
    /// only unpinned drained pages; restore writes freshly reserved ones),
    /// so a held wave must NEVER starve them — the preemption ladder is
    /// what unsticks a held wave in the first place. `PreLaunchCopy` is
    /// NOT in this class: it is order-coupled to its own launch (queued
    /// directly in front of it) and must keep queue order.
    const fn standalone_copy(item: &QueuedItem) -> bool {
        matches!(
            item,
            QueuedItem::CopyKv { .. }
                | QueuedItem::CopyKvTracked { .. }
                | QueuedItem::CopyState { .. }
        )
    }

    /// Controls that dispatch without draining in-flight launches. The
    /// registrations are synchronous and create entities nothing in flight
    /// can reference yet — with one caveat: a channel registration that
    /// grows the driver's shared slot table would reallocate arrays whose
    /// pointers in-flight kernels hold, so the CUDA registry quiesces the
    /// device inside `grow()` (RV-27; capacity is driver knowledge, so the
    /// drain lives there, not here). The copies (standalone and pre-launch)
    /// address only committed or quiesced extents, which in-flight launches
    /// never rewrite (append-only ledger) — a copy's coupled consumer launch
    /// still holds behind `in_flight_control` until the copy settles.
    /// A channel close only ever follows its instance closes (the guest
    /// awaits each control's response) and the driver rejects a close with
    /// live attachments, so no in-flight kernel can reference the closing
    /// channel — it needs no drain. `CloseInstance` has its own per-instance
    /// quiescence gate in `dispatch_ready_items`. Only pool resizes keep the
    /// empty-pipe requirement: drain IS their ordering mechanism.
    const fn pipe_concurrent_control(item: &QueuedItem) -> bool {
        Self::standalone_copy(item)
            || matches!(
                item,
                QueuedItem::PreLaunchCopy { .. }
                    | QueuedItem::RegisterProgram { .. }
                    | QueuedItem::RegisterChannel { .. }
                    | QueuedItem::RegisterChannels { .. }
                    | QueuedItem::BindInstance { .. }
                    | QueuedItem::RegisterChannelsBind { .. }
                    | QueuedItem::CloseChannel { .. }
            )
    }

    /// Move held launches behind work that can make the current quorum
    /// denser. The ENTIRE contiguous launch prefix rotates to the back in one
    /// call: per-instance launch order is a dispatch invariant
    /// (`launch_has_earlier_instance_member` defers an out-of-order head, and
    /// a head whose earlier sibling sits beyond a non-launch item is
    /// unreachable — a permanent stall), so a run-ahead sibling group must
    /// never be split by a partial rotation.
    ///
    /// A `PreLaunchCopy` is valid rotate-target work under `allow_controls`:
    /// it occupies the free control slot exactly like a lifecycle control.
    /// Rotating front launches past it cannot break copy→consumer coupling —
    /// a consumer launch is enqueued behind its copy and stays behind it (a
    /// Lifecycle controls (registers, binds, closes) never order against a
    /// launch already in the queue: a fire can only be submitted after its
    /// own bind returned to the guest, so every queued launch's lifecycle
    /// dependencies have already dispatched. Only `PreLaunchCopy` (channel
    /// data feeding a later queued launch of the same pipeline) and pool
    /// resizes (pipe drains) order against queued launches.
    const fn lifecycle_control(item: &QueuedItem) -> bool {
        matches!(
            item,
            QueuedItem::RegisterProgram { .. }
                | QueuedItem::RegisterChannel { .. }
                | QueuedItem::RegisterChannels { .. }
                | QueuedItem::BindInstance { .. }
                | QueuedItem::RegisterChannelsBind { .. }
                | QueuedItem::CloseInstance { .. }
                | QueuedItem::CloseChannel { .. }
        )
    }

    /// launch that reached the queue front has no queued copy left).
    fn rotate_launch_for_wave_work(
        pending: &mut VecDeque<QueuedItem>,
        allow_controls: bool,
    ) -> bool {
        if !matches!(pending.front(), Some(QueuedItem::Launch(_))) {
            return false;
        }
        let Some(work) = pending
            .iter()
            .skip(1)
            .find(|item| !matches!(item, QueuedItem::Launch(_)))
        else {
            return false;
        };
        if !(Self::standalone_copy(work)
            || (allow_controls
                && (Self::lifecycle_control(work)
                    || matches!(work, QueuedItem::PreLaunchCopy { .. }))))
        {
            return false;
        }
        while matches!(pending.front(), Some(QueuedItem::Launch(_))) {
            let launch = pending.pop_front().expect("launch front");
            pending.push_back(launch);
        }
        true
    }

    fn rotate_launch_for_admission_work(pending: &mut VecDeque<QueuedItem>) -> bool {
        if Self::rotate_launch_for_wave_work(pending, true) {
            return true;
        }
        if !matches!(pending.front(), Some(QueuedItem::Launch(_))) {
            return false;
        }
        if !matches!(
            pending
                .iter()
                .skip(1)
                .find(|item| { !matches!(item, QueuedItem::Launch(_)) }),
            Some(QueuedItem::ResizePool { .. })
        ) {
            return false;
        }
        while matches!(pending.front(), Some(QueuedItem::Launch(_))) {
            let launch = pending.pop_front().expect("launch front");
            pending.push_back(launch);
        }
        true
    }

    fn launch_has_earlier_instance_member(
        pending: &VecDeque<QueuedItem>,
        request: &PendingRequest,
    ) -> bool {
        pending.iter().any(|item| {
            let QueuedItem::Launch(earlier) = item else {
                return false;
            };
            earlier.instance_id == request.instance_id
                && earlier.logical_fire_id < request.logical_fire_id
        })
    }

    /// B3 FIFO invariant, CROSS-INSTANCE: the smallest queued/preparing
    /// logical fire id per pipeline (logical fire ids are minted at submit,
    /// so per-pipeline id order IS submission order). A candidate Launch
    /// whose id is above its pipeline's floor has an earlier undispatched
    /// sibling and must defer. Same-instance order is carried by
    /// preparation lanes and `launch_has_earlier_instance_member`; the
    /// cross-instance case is the R4-4 single-pipeline prefill→decode
    /// handoff, where the prefill's requeue-at-back after async preparation
    /// let the already-credited decode fires OVERTAKE it — the decode then
    /// executed against KV the prefill had not committed and the driver's
    /// compose kernel FAIL-STOPPED the lane. An ACTIVE preparation gates
    /// its whole pipeline (floor 0): its fire is earlier than anything
    /// queued by construction and is in neither queue. ONE O(queue) sweep
    /// per composing pass, O(1) per candidate — a per-candidate rescan
    /// measured −5% whole-run (peek runs at poll cadence). Computed at
    /// pass start, the floor also covers items the pass later parks in its
    /// deferred side-queue.
    fn pipeline_order_floor(pending: &VecDeque<QueuedItem>) -> HashMap<ProcessId, u64> {
        let mut floor: HashMap<ProcessId, u64> = HashMap::new();
        let mut note = |pid: Option<ProcessId>, id: u64| {
            if let Some(pid) = pid {
                floor
                    .entry(pid)
                    .and_modify(|lowest| *lowest = (*lowest).min(id))
                    .or_insert(id);
            }
        };
        for item in pending {
            if let QueuedItem::Launch(request) = item {
                note(request.pipeline_id, request.logical_fire_id);
            }
        }
        floor
    }

    /// Peeks how many requests at `pending`'s front would land in the NEXT
    /// launch batch — same grouping rules `dispatch_launch_batch` applies
    /// (same-instance dedup, mask-solo, structural capacity) — without
    /// mutating the queue or the driver. Feeds `WaitAllPolicy::
    /// decide_wave_at`'s `current_batch_size` so the quorum decision sees
    /// the exact geometry the dispatcher is about to build (a stale item —
    /// its instance closed after enqueue — is skipped here exactly like
    /// the real dispatch skips/rejects it, so it never inflates the count).
    fn peek_launch_batch(
        pending: &VecDeque<QueuedItem>,
        instances: &HashMap<u64, TrackedInstance>,
        limits: SchedulerLimits,
        page_size: u32,
    ) -> LaunchBatchPreview {
        let mut grouping = LaunchGrouping::default();
        let mut blocked_pipelines = HashSet::new();
        let mut pipelines = HashSet::new();
        let mut deferred: HashSet<ProcessId> = HashSet::new();
        let mut structurally_full = false;
        let mut logical_fire_ids = Vec::new();
        let mut barrier_pipelines: HashSet<ProcessId> = HashSet::new();
        let order_floor = Self::pipeline_order_floor(pending);
        for item in pending.iter() {
            let next = match item {
                QueuedItem::Launch(next) => next,
                // Mirror dispatch_launch_batch exactly: skip lifecycle
                // controls and pre-launch copies while barriering their
                // pipeline; stop at everything else (pool resizes, state
                // copies).
                item if Self::lifecycle_control(item) => continue,
                QueuedItem::PreLaunchCopy { pipeline_id, .. } => {
                    if let Some(pid) = pipeline_id {
                        barrier_pipelines.insert(*pid);
                    }
                    continue;
                }
                _ => break,
            };
            if next
                .pipeline_id
                .is_some_and(|pid| barrier_pipelines.contains(&pid))
            {
                // NOT deferred: a fire behind its own preparation is
                // imminent (~1 ms) — the barrier's short hold for it is the
                // useful gathering (excluding these fired eager narrow
                // waves and cost 20 % tput, measured V6 iteration 35).
                continue;
            }
            if !instances.contains_key(&next.instance_id) {
                continue;
            }
            // Mirror dispatch's drops exactly: a settled or cancelled request
            // never reaches the batch, so counting it here would let the
            // quorum decision see a bigger candidate wave than dispatch can
            // build (transient barrier violation — RV-20).
            if next.completion.is_settled() || next.completion.cancel_requested() {
                continue;
            }
            if next.retry_after.is_some_and(|due| due > Instant::now()) {
                continue;
            }
            if Self::launch_has_earlier_instance_member(pending, next) {
                continue;
            }
            if next.pipeline_id.is_some_and(|pid| {
                order_floor
                    .get(&pid)
                    .is_some_and(|&lowest| lowest < next.logical_fire_id)
            }) {
                // Composer-ordered behind its pipeline's earlier fire —
                // scheduled for a later wave, never missing.
                if let Some(pid) = next.pipeline_id {
                    deferred.insert(pid);
                }
                continue;
            }
            if next
                .pipeline_id
                .is_some_and(|pid| blocked_pipelines.contains(&pid))
            {
                continue;
            }
            if !grouping.accepts(next, limits, page_size) {
                if (grouping.instances.contains(&next.instance_id)
                    || next
                        .pipeline_id
                        .is_some_and(|pid| grouping.pipelines.contains(&pid)))
                    && let Some(pid) = next.pipeline_id
                {
                    blocked_pipelines.insert(pid);
                    deferred.insert(pid);
                    continue;
                }
                if let Some(pid) = next.pipeline_id {
                    deferred.insert(pid);
                }
                structurally_full = grouping.count != 0;
                break;
            }
            let stop = grouping.push(next, limits, page_size);
            logical_fire_ids.push(next.logical_fire_id);
            if let Some(pid) = next.pipeline_id {
                pipelines.insert(pid);
            }
            if stop {
                structurally_full = true;
                break;
            }
        }
        // Classify every remaining queued launch: the composing pass stops
        // at the batch-full / drain break, but queued fires beyond it are
        // SCHEDULED (composer/ordering-deferred), not missing — without
        // this sweep, budget-queued prefills past the break counted missing
        // and demoted en masse on the prefix shape (V6 iteration 35's open
        // bug). Preparation-barriered lanes stay awaited: their fires are
        // imminent and waiting ~1 ms for them keeps waves dense.
        let mut submitted: HashSet<ProcessId> = HashSet::new();
        for item in pending.iter() {
            match item {
                QueuedItem::Launch(next) => {
                    if let Some(pid) = next.pipeline_id {
                        submitted.insert(pid);
                        if !pipelines.contains(&pid) && !barrier_pipelines.contains(&pid) {
                            deferred.insert(pid);
                        }
                    }
                }
                QueuedItem::PreLaunchCopy { pipeline_id, .. } => {
                    if let Some(pid) = pipeline_id {
                        submitted.insert(*pid);
                    }
                }
                _ => {}
            }
        }
        submitted.extend(pipelines.iter().copied());
        LaunchBatchPreview {
            count: grouping.count,
            logical_fire_ids,
            pipelines,
            deferred,
            submitted,
            structurally_full,
        }
    }

    fn candidate_submission(
        pending: &VecDeque<QueuedItem>,
        logical_fire_ids: &[u64],
        page_size: u32,
    ) -> Option<crate::driver::LaunchSubmission> {
        let mut requests = Vec::with_capacity(logical_fire_ids.len());
        let mut next = 0usize;
        for item in pending {
            let QueuedItem::Launch(request) = item else {
                continue;
            };
            if logical_fire_ids.get(next) == Some(&request.logical_fire_id) {
                requests.push(request.clone_for_batch());
                next += 1;
                if next == logical_fire_ids.len() {
                    break;
                }
            }
        }
        if requests.is_empty() || next != logical_fire_ids.len() {
            return None;
        }
        let proposal_stats = SchedulerStats::default();
        Some(batch::build_batch_request(
            &requests,
            page_size,
            &proposal_stats,
        ))
    }

    fn reject_launch_candidates(
        pending: &mut VecDeque<QueuedItem>,
        logical_fire_ids: &[u64],
        policy: &mut quorum::WaitAllPolicy,
        message: &str,
    ) -> usize {
        let ids: HashSet<u64> = logical_fire_ids.iter().copied().collect();
        let mut kept = VecDeque::with_capacity(pending.len());
        let mut rejected = 0usize;
        while let Some(item) = pending.pop_front() {
            match item {
                QueuedItem::Launch(request) if ids.contains(&request.logical_fire_id) => {
                    Self::drop_request_credit(policy, &request);
                    Self::resolve_stream_fire(policy, &request);
                    request.completion.reject_unsubmitted(message.to_string());
                    rejected += 1;
                }
                item => kept.push_back(item),
            }
        }
        *pending = kept;
        rejected
    }

    fn dispatch_ready_items(
        driver_lane: &DriverLane,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut VecDeque<QueuedItem>,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &mut Option<PendingControl>,
        admission_retry_at: &mut Option<Instant>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        policy: &mut quorum::WaitAllPolicy,
        stopping: bool,
    ) -> (bool, Option<Duration>) {
        let mut progress = false;
        let mut wait_hint: Option<Duration> = None;
        // Busy-close rotations this pass: bounded so a queue of nothing but
        // busy closes breaks out instead of spinning.
        let mut close_rotations = 0usize;
        loop {
            let Some(item) = pending.front() else {
                break;
            };
            match item {
                QueuedItem::Launch(_) => {
                    if in_flight_control.is_some() {
                        // A settling copy/resize holds launches — the very
                        // next launch may be the copy's coupled consumer —
                        // but preparation work still advances behind it.
                        if Self::rotate_launch_for_wave_work(pending, false) {
                            progress = true;
                            continue;
                        }
                        break;
                    }
                    if in_flight_launches.len() >= quorum::configured_max_in_flight() {
                        // Depth-capped: launches cannot dispatch anyway, so
                        // held wave work rotates back to reach ANY
                        // dispatchable control — including lifecycle
                        // controls with the pipe full. They are
                        // pipe-concurrent and ~100 µs, and making a
                        // transitioning pipeline's register→bind round
                        // trips wait a full wave each was the cohort-swap
                        // join-latency tail (V6 iteration 6).
                        if Self::rotate_launch_for_wave_work(pending, true) {
                            progress = true;
                            continue;
                        }
                        break;
                    }
                    let candidate = Self::peek_launch_batch(pending, instances, limits, page_size);
                    let mut decision_us = 0u64;
                    let mut decision_missing = 0usize;
                    let decision_candidate_count = candidate.count;
                    let decision_deferred = candidate.deferred.len();
                    let decision_depth_capped = policy.depth_capped_pipelines();
                    let decision_now = Instant::now();
                    let mut prepared_lease = None;
                    if driver_lane.admission_supported
                        && admission_retry_at.is_some_and(|due| due > decision_now)
                    {
                        let hold = admission_retry_at
                            .expect("admission retry deadline is present")
                            .saturating_duration_since(decision_now);
                        wait_hint = Some(wait_hint.map_or(hold, |old| old.min(hold)));
                        if Self::rotate_launch_for_admission_work(pending) {
                            progress = true;
                            continue;
                        }
                        break;
                    }
                    if !stopping {
                        let decision_started_us =
                            super::ledger_timing_enabled().then(super::fire_timing_now_us);
                        let decision = if driver_lane.admission_supported {
                            policy.preview_candidate_wave_at(
                                candidate.count,
                                &candidate.pipelines,
                                &candidate.deferred,
                                &candidate.submitted,
                                candidate.structurally_full,
                                decision_now,
                            )
                        } else {
                            policy.decide_candidate_wave_at(
                                candidate.count,
                                &candidate.pipelines,
                                &candidate.deferred,
                                &candidate.submitted,
                                candidate.structurally_full,
                                decision_now,
                            )
                        };
                        if let Some(started_us) = decision_started_us {
                            decision_us = super::fire_timing_now_us().saturating_sub(started_us);
                        }
                        match decision {
                            quorum::WaveDecision::Wait(hold) => {
                                let hold = if driver_lane.admission_supported {
                                    match policy.decide_candidate_wave_at(
                                        candidate.count,
                                        &candidate.pipelines,
                                        &candidate.deferred,
                                        &candidate.submitted,
                                        candidate.structurally_full,
                                        decision_now,
                                    ) {
                                        quorum::WaveDecision::Wait(committed) => committed,
                                        quorum::WaveDecision::Fire { .. } => {
                                            continue;
                                        }
                                    }
                                } else {
                                    hold
                                };
                                wait_hint = Some(hold);
                                // A holding wave yields the thread to any
                                // dispatchable control regardless of pipe
                                // depth (see the depth-cap branch): the
                                // hold is exactly when a transitioning
                                // pipeline's bind/register round trips must
                                // not wait a wave each.
                                if Self::rotate_launch_for_wave_work(pending, true) {
                                    progress = true;
                                    continue;
                                }
                                break;
                            }
                            quorum::WaveDecision::Fire { missing } => {
                                decision_missing = missing.len();
                            }
                        }
                    }
                    if driver_lane.admission_supported && !candidate.logical_fire_ids.is_empty() {
                        let Some(submission) = Self::candidate_submission(
                            pending,
                            &candidate.logical_fire_ids,
                            page_size,
                        ) else {
                            progress |= Self::reject_launch_candidates(
                                pending,
                                &candidate.logical_fire_ids,
                                policy,
                                "launch admission proposal became inconsistent",
                            ) != 0;
                            continue;
                        };
                        match driver_lane.prepare(submission) {
                            Ok(LaunchPrepareOutcome::Prepared(lease)) => {
                                prepared_lease = Some(lease);
                                *admission_retry_at = None;
                                if !stopping {
                                    match policy.decide_candidate_wave_at(
                                        candidate.count,
                                        &candidate.pipelines,
                                        &candidate.deferred,
                                        &candidate.submitted,
                                        candidate.structurally_full,
                                        decision_now,
                                    ) {
                                        quorum::WaveDecision::Fire { missing } => {
                                            decision_missing = missing.len();
                                        }
                                        quorum::WaveDecision::Wait(hold) => {
                                            let _ = driver_lane.release(lease);
                                            wait_hint = Some(hold);
                                            break;
                                        }
                                    }
                                }
                            }
                            Ok(LaunchPrepareOutcome::Exhausted {
                                budget_generation,
                                required_pages,
                                budget_pages,
                            }) => {
                                if stopping {
                                    progress |= Self::reject_launch_candidates(
                                        pending,
                                        &candidate.logical_fire_ids,
                                        policy,
                                        "scheduler shutdown interrupted launch admission",
                                    ) != 0;
                                    continue;
                                }
                                let _ = (budget_generation, required_pages, budget_pages);
                                const RETRY: Duration = Duration::from_millis(1);
                                *admission_retry_at = Some(Instant::now() + RETRY);
                                wait_hint = Some(RETRY);
                                if Self::rotate_launch_for_admission_work(pending) {
                                    progress = true;
                                    continue;
                                }
                                break;
                            }
                            Ok(LaunchPrepareOutcome::Impossible {
                                required_pages,
                                budget_pages,
                            }) => {
                                *admission_retry_at = None;
                                let message = format!(
                                    "launch requires {required_pages} physical pages, \
                                     exceeding driver ceiling {budget_pages}"
                                );
                                progress |= Self::reject_launch_candidates(
                                    pending,
                                    &candidate.logical_fire_ids,
                                    policy,
                                    &message,
                                ) != 0;
                                continue;
                            }
                            Ok(LaunchPrepareOutcome::Unsupported) => {
                                if !stopping {
                                    match policy.decide_candidate_wave_at(
                                        candidate.count,
                                        &candidate.pipelines,
                                        &candidate.deferred,
                                        &candidate.submitted,
                                        candidate.structurally_full,
                                        decision_now,
                                    ) {
                                        quorum::WaveDecision::Fire { missing } => {
                                            decision_missing = missing.len();
                                        }
                                        quorum::WaveDecision::Wait(hold) => {
                                            wait_hint = Some(hold);
                                            break;
                                        }
                                    }
                                }
                            }
                            Err(error) => {
                                *admission_retry_at = None;
                                progress |= Self::reject_launch_candidates(
                                    pending,
                                    &candidate.logical_fire_ids,
                                    policy,
                                    &format!("launch preparation failed: {error}"),
                                ) != 0;
                                continue;
                            }
                        }
                    }
                    let decision_active = policy.active_pipelines();
                    let prepared_fire_ids =
                        prepared_lease.map(|_| candidate.logical_fire_ids.clone());
                    let before = in_flight_launches.len();
                    let dispatched = crate::probe_fire!(
                        stats.fire.execute.total_us,
                        Self::dispatch_launch_batch(
                            driver_lane,
                            lane_inflight,
                            lane_token,
                            instances,
                            pending,
                            in_flight_launches,
                            page_size,
                            limits,
                            stats,
                            policy,
                            decision_us,
                            decision_active,
                            decision_missing,
                            decision_candidate_count,
                            decision_deferred,
                            decision_depth_capped,
                            prepared_lease,
                            prepared_fire_ids,
                        )
                    );
                    // Only a batch that actually reached `in_flight_launches`
                    // (posted to the driver lane) increments the policy's
                    // depth counter — a synchronous stale-instance-reject
                    // never occupies a run-ahead slot. A lane-side
                    // `driver.launch` rejection unwinds through the retire
                    // path (the batch retires as failed, releasing depth and
                    // instance accounting the same way a completed wave
                    // does).
                    if in_flight_launches.len() > before {
                        let accepted = in_flight_launches
                            .back()
                            .expect("accepted batch is present");
                        // Consumption mirrors publication: a request whose
                        // pipeline left after admission published untracked,
                        // so it must consume untracked too — the row lookup
                        // by bare pid would eat a successor pipeline's
                        // credit (W1).
                        let participants = accepted
                            .requests
                            .iter()
                            .map(|request| {
                                Self::quorum_pid(
                                    policy,
                                    request.pipeline_id,
                                    request.quorum_generation,
                                )
                            })
                            .collect::<Vec<_>>();
                        let accepted_len = accepted.requests.len();
                        let epochs = policy.on_wave_dispatched(&participants, Instant::now());
                        let accepted = in_flight_launches
                            .back_mut()
                            .expect("accepted batch is present");
                        accepted.pipeline_epochs = epochs;
                        // The wave consumed each row's readiness credit; a
                        // RETRY re-arm publishes (and re-marks) a fresh one.
                        for request in accepted.requests.iter_mut() {
                            request.credit_published = false;
                        }
                        if super::sched_trace_enabled() {
                            let (candidate_present, candidate_at_depth, candidate_missing) =
                                policy.candidate_state_counts(&candidate.pipelines);
                            let mut queued_launches = 0usize;
                            let mut queued_controls = 0usize;
                            let mut queued_pipelines = HashSet::new();
                            for item in pending.iter() {
                                match item {
                                    QueuedItem::Launch(request) => {
                                        queued_launches += 1;
                                        if let Some(pid) = request.pipeline_id {
                                            queued_pipelines.insert(pid);
                                        }
                                    }
                                    _ => queued_controls += 1,
                                }
                            }
                            super::sched_trace_write(format_args!(
                                concat!(
                                    "wave candidate={} dispatched={} active={} ",
                                    "pending={} launches={} ",
                                    "controls={} ready_pipelines={} ",
                                    "candidate_pipelines={} present={} at_depth={} missing={}"
                                ),
                                candidate.count,
                                accepted_len,
                                policy.active_pipelines(),
                                pending.len(),
                                queued_launches,
                                queued_controls,
                                queued_pipelines.len(),
                                candidate.pipelines.len(),
                                candidate_present,
                                candidate_at_depth,
                                candidate_missing,
                            ));
                        }
                    }
                    progress |= dispatched;
                    if !dispatched {
                        break;
                    }
                }
                QueuedItem::CloseInstance { id, .. } => {
                    // A close needs only ITS OWN instance quiesced — never a
                    // global pipe drain. Inferlets submit passes upfront, so
                    // an idle instance's close is always safe to overlap
                    // with other instances' in-flight launches; the old
                    // whole-pipe drain stalled every launch queued behind a
                    // front close during cohort swaps, made freshly-bound
                    // pipelines' credits ragged against the wave window,
                    // and turned straggler demotion into a storm (V6
                    // iteration 3: demotions track lost throughput 1:1, and
                    // healthy inferlets should see none).
                    if in_flight_control.is_some() {
                        break;
                    }
                    let id = *id;
                    let busy = instances
                        .get(&id)
                        .is_some_and(|tracked| tracked.in_flight != 0)
                        || Self::instance_has_queued_work(pending, id);
                    if !busy {
                        let item = pending.pop_front().expect("close front");
                        Self::post_control(
                            driver_lane,
                            lane_inflight,
                            lane_token,
                            instances,
                            in_flight_control,
                            policy,
                            item,
                        );
                        progress = true;
                        // One control post per pass: the outer loop drains
                        // the scheduler channel between controls, so a
                        // cohort-swap burst cannot let readiness credits
                        // pile up unread while the wave window ages — the
                        // post-burst decide then sees a dense fleet instead
                        // of mass-demoting pipelines whose credits sat in
                        // the mailbox (V6 iteration 5).
                        break;
                    }
                    // Busy: rotate the close behind the queue so the fires
                    // that will quiesce it (and everything unrelated) keep
                    // flowing; its own retirement re-checks it. A close can
                    // only move BACKWARD, so it never overtakes its own
                    // instance's queued work.
                    if close_rotations >= pending.len()
                        || !pending
                            .iter()
                            .skip(1)
                            .any(|item| !matches!(item, QueuedItem::CloseInstance { .. }))
                    {
                        break;
                    }
                    close_rotations += 1;
                    let item = pending.pop_front().expect("close front");
                    pending.push_back(item);
                    progress = true;
                }
                // Single control slot: a settling copy/resize blocks the
                // next control (the slot only ever holds async-completing
                // controls now — lifecycle controls execute on the lane
                // without occupying it, their driver order guaranteed by the
                // lane FIFO).
                _ if in_flight_control.is_some() => break,
                _ if !in_flight_launches.is_empty() && !Self::pipe_concurrent_control(item) => {
                    break;
                }
                _ => {
                    let item = pending.pop_front().expect("front item present");
                    Self::post_control(
                        driver_lane,
                        lane_inflight,
                        lane_token,
                        instances,
                        in_flight_control,
                        policy,
                        item,
                    );
                    progress = true;
                    // Keep readiness publication interleaved with control
                    // posting; flooding the lane did not improve first-full.
                    break;
                }
            }
        }
        (progress, wait_hint)
    }

    /// Static item-kind label for the control-occupancy fire-timing probe.
    const fn item_kind(item: &QueuedItem) -> &'static str {
        match item {
            QueuedItem::Launch(_) => "launch",
            QueuedItem::PreLaunchCopy { .. } => "pre_launch_copy",
            QueuedItem::RegisterProgram { .. } => "register_program",
            QueuedItem::RegisterChannel { .. } => "register_channel",
            QueuedItem::RegisterChannels { .. } => "register_channels",
            QueuedItem::BindInstance { .. } => "bind_instance",
            QueuedItem::RegisterChannelsBind { .. } => "register_channels_bind",
            QueuedItem::CopyKv { .. } => "copy_kv",
            QueuedItem::CopyKvTracked { .. } => "copy_kv_tracked",
            QueuedItem::CopyState { .. } => "copy_state",
            QueuedItem::ResizePool { .. } => "resize_pool",
            QueuedItem::CloseInstance { .. } => "close_instance",
            QueuedItem::CloseChannel { .. } => "close_channel",
        }
    }

    /// Post a control to the driver lane after the worker-side pre-checks
    /// that read scheduler state. The driver half runs on the lane in FIFO
    /// order; worker-map effects come back as a [`LaneCommit`]. Async
    /// controls (copies / pool resizes) occupy the single control slot from
    /// the moment they post.
    fn post_control(
        driver_lane: &DriverLane,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        in_flight_control: &mut Option<PendingControl>,
        policy: &mut quorum::WaitAllPolicy,
        item: QueuedItem,
    ) {
        match &item {
            QueuedItem::Launch(_) => unreachable!(),
            QueuedItem::PreLaunchCopy {
                logical_completion, ..
            } if logical_completion.is_settled() => return,
            QueuedItem::PreLaunchCopy {
                logical_completion, ..
            } if logical_completion.cancel_requested() => {
                logical_completion
                    .reject_unsubmitted("logical fire cancelled before pre-launch copy");
                return;
            }
            QueuedItem::CloseInstance {
                id, pacing_wait_id, ..
            } => {
                let error = match instances.get(id) {
                    Some(instance) if instance.pacing_wait_id == *pacing_wait_id => {
                        (instance.in_flight != 0).then(|| format!("instance {id} is busy"))
                    }
                    _ => Some(format!("instance {id} is unknown or stale")),
                };
                if let Some(message) = error {
                    tracing::warn!(
                        instance_id = id,
                        error = %message,
                        "scheduler close_instance skipped"
                    );
                    return;
                }
            }
            QueuedItem::BindInstance { plan, .. }
                if plan.requested_instance_id != 0
                    && instances.contains_key(&plan.requested_instance_id) =>
            {
                let QueuedItem::BindInstance {
                    pipeline_id,
                    plan,
                    response,
                } = item
                else {
                    unreachable!();
                };
                if response
                    .send(Err(anyhow!(
                        "instance {} is already bound",
                        plan.requested_instance_id
                    )))
                    .is_err()
                {
                    DriverLane::release_wait_slots([plan.pacing_wait_id]);
                }
                policy.on_bind_completed(pipeline_id, Instant::now());
                return;
            }
            QueuedItem::RegisterChannelsBind { bind, .. }
                if bind.requested_instance_id != 0
                    && instances.contains_key(&bind.requested_instance_id) =>
            {
                let QueuedItem::RegisterChannelsBind {
                    pipeline_id,
                    plans,
                    bind,
                    response,
                    ..
                } = item
                else {
                    unreachable!();
                };
                if response
                    .send(Err(anyhow!(
                        "instance {} is already bound",
                        bind.requested_instance_id
                    )))
                    .is_err()
                {
                    DriverLane::release_channel_plan_wait_slots(&plans);
                    DriverLane::release_wait_slots([bind.pacing_wait_id]);
                }
                policy.on_bind_completed(pipeline_id, Instant::now());
                return;
            }
            _ => {}
        }
        *lane_token += 1;
        let token = *lane_token;
        // Async-completing controls hold the single control slot from POST:
        // the copy's coupled consumer launch (and any later control) must
        // not pass it, exactly as before the lane existed.
        match &item {
            QueuedItem::PreLaunchCopy {
                plan,
                logical_completion,
                process_id,
                pipeline_id,
                credit_ready,
                quorum_generation,
            } => {
                *in_flight_control = Some(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: Some(logical_completion.clone()),
                    process_id: *process_id,
                    pipeline_id: *pipeline_id,
                    tracked_completion: None,
                    operation: plan.label(),
                    credit_ready: *credit_ready,
                    quorum_generation: *quorum_generation,
                });
            }
            QueuedItem::CopyKv { .. } => {
                *in_flight_control = Some(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: None,
                    operation: "KV copy",
                    credit_ready: false,
                    quorum_generation: 0,
                });
            }
            QueuedItem::CopyKvTracked { completion, .. } => {
                *in_flight_control = Some(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: Some(completion.clone()),
                    operation: "tracked KV copy",
                    credit_ready: false,
                    quorum_generation: 0,
                });
            }
            QueuedItem::CopyState { .. } => {
                *in_flight_control = Some(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: None,
                    operation: "state copy",
                    credit_ready: false,
                    quorum_generation: 0,
                });
            }
            QueuedItem::ResizePool { .. } => {
                *in_flight_control = Some(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: None,
                    operation: "pool resize",
                    credit_ready: false,
                    quorum_generation: 0,
                });
            }
            _ => {}
        }
        *lane_inflight += 1;
        driver_lane.post(LaneRequest::Control { token, item });
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_launch_batch(
        driver_lane: &DriverLane,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut VecDeque<QueuedItem>,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        policy: &mut quorum::WaitAllPolicy,
        decision_us: u64,
        active_pipelines: usize,
        missing_pipelines: usize,
        candidate_count: usize,
        deferred_pipelines: usize,
        depth_capped_pipelines: usize,
        prepared_lease: Option<LaunchLease>,
        prepared_fire_ids: Option<Vec<u64>>,
    ) -> bool {
        let mut batch = BatchAccumulator::new(limits, page_size);
        let mut grouping = LaunchGrouping::default();
        let mut deferred = VecDeque::new();
        let mut blocked_pipelines = HashSet::new();
        let mut rejected_stale = false;
        let mut barrier_pipelines: HashSet<ProcessId> = HashSet::new();
        let mut selected_count = 0usize;
        let order_floor = Self::pipeline_order_floor(pending);
        loop {
            // Gather past items that do not order against queued launches:
            // lifecycle controls never do (see `lifecycle_control`);
            // pre-launch copies order only against LATER fires of their
            // pipeline, so they are skipped with a barrier that defers
            // exactly those.
            // Deferred items return to the queue front in original order.
            match pending.front() {
                Some(QueuedItem::Launch(_)) => {}
                Some(item) if Self::lifecycle_control(item) => {
                    deferred.push_back(pending.pop_front().expect("control front"));
                    continue;
                }
                Some(QueuedItem::PreLaunchCopy { pipeline_id, .. }) => {
                    if let Some(pid) = pipeline_id {
                        barrier_pipelines.insert(*pid);
                    }
                    deferred.push_back(pending.pop_front().expect("copy front"));
                    continue;
                }
                _ => break,
            }
            let Some(QueuedItem::Launch(next)) = pending.front() else {
                unreachable!();
            };
            if prepared_fire_ids
                .as_ref()
                .is_some_and(|expected| selected_count >= expected.len())
            {
                break;
            }
            let prepared_member = prepared_fire_ids
                .as_ref()
                .is_some_and(|expected| expected.contains(&next.logical_fire_id));
            if next
                .pipeline_id
                .is_some_and(|pid| barrier_pipelines.contains(&pid))
            {
                deferred.push_back(pending.pop_front().expect("barriered launch front"));
                continue;
            }
            if !prepared_member && next.completion.is_settled() {
                let Some(QueuedItem::Launch(dropped)) = pending.pop_front() else {
                    unreachable!();
                };
                Self::drop_request_credit(policy, &dropped);
                rejected_stale = true;
                continue;
            }
            if !prepared_member && next.completion.cancel_requested() {
                let Some(QueuedItem::Launch(cancelled)) = pending.pop_front() else {
                    unreachable!();
                };
                Self::drop_request_credit(policy, &cancelled);
                cancelled
                    .completion
                    .reject_unsubmitted("logical fire cancelled before native launch");
                rejected_stale = true;
                continue;
            }
            if next.retry_after.is_some_and(|due| due > Instant::now()) {
                deferred.push_back(pending.pop_front().expect("backoff launch front"));
                continue;
            }
            if instances.get(&next.instance_id).is_none() {
                // A launch whose instance closed between enqueue validation
                // and dispatch must be rejected here, not left at the queue
                // front where it would head-of-line block the driver forever.
                let Some(QueuedItem::Launch(stale)) = pending.pop_front() else {
                    unreachable!();
                };
                Self::drop_request_credit(policy, &stale);
                stale.completion.reject_unsubmitted(format!(
                    "instance {} is unknown or stale",
                    stale.instance_id
                ));
                rejected_stale = true;
                continue;
            }
            if Self::launch_has_earlier_instance_member(pending, next) {
                deferred.push_back(pending.pop_front().expect("out-of-order launch front"));
                continue;
            }
            if next.pipeline_id.is_some_and(|pid| {
                order_floor
                    .get(&pid)
                    .is_some_and(|&lowest| lowest < next.logical_fire_id)
            }) {
                deferred.push_back(pending.pop_front().expect("pipeline-order launch front"));
                continue;
            }
            if next
                .pipeline_id
                .is_some_and(|pid| blocked_pipelines.contains(&pid))
            {
                deferred.push_back(pending.pop_front().expect("blocked launch front"));
                continue;
            }
            if !grouping.accepts(next, limits, page_size) {
                if (grouping.instances.contains(&next.instance_id)
                    || next
                        .pipeline_id
                        .is_some_and(|pid| grouping.pipelines.contains(&pid)))
                    && let Some(pid) = next.pipeline_id
                {
                    blocked_pipelines.insert(pid);
                    deferred.push_back(pending.pop_front().expect("runahead launch front"));
                    continue;
                }
                break;
            }
            if let Some(expected) = prepared_fire_ids.as_ref()
                && expected.get(selected_count) != Some(&next.logical_fire_id)
            {
                break;
            }
            let QueuedItem::Launch(next) = pending.pop_front().expect("launch front") else {
                unreachable!();
            };
            let stop = grouping.push(&next, limits, page_size);
            batch.push(next);
            selected_count += 1;
            if stop {
                break;
            }
        }
        while let Some(item) = deferred.pop_back() {
            pending.push_front(item);
        }
        let mut requests = batch.take();
        if requests.is_empty() {
            if let Some(lease) = prepared_lease {
                let _ = driver_lane.release(lease);
            }
            return rejected_stale;
        }
        if let Some(expected) = prepared_fire_ids.as_ref() {
            let actual = requests
                .iter()
                .map(|request| request.logical_fire_id)
                .collect::<Vec<_>>();
            if &actual != expected {
                for request in requests.into_iter().rev() {
                    pending.push_front(QueuedItem::Launch(request));
                }
                if let Some(lease) = prepared_lease {
                    let _ = driver_lane.release(lease);
                }
                return true;
            }
        }
        let timing_enabled = super::fire_timing_enabled();
        let dispatch_started_us = timing_enabled.then(super::fire_timing_now_us);
        if let Some(now) = dispatch_started_us {
            for request in &mut requests {
                if let Some(timing) = request.timing.as_mut()
                    && timing.ready_us.is_none()
                {
                    timing.ready_us = Some(now);
                }
            }
        }
        let batch_size = requests.len() as u64;
        // Token stats must be read before build: a prebuilt single-request
        // batch moves its plan into the submission.
        let total_tokens = requests
            .iter()
            .map(|req| req.request.token_ids.len())
            .sum::<usize>();
        let submission = batch::build_batch_request(&mut requests, page_size, stats);
        let batch_built_us = timing_enabled.then(super::fire_timing_now_us);
        for request in &requests {
            if instances.get(&request.instance_id).is_none() {
                for request in &mut requests {
                    Self::drop_request_credit(policy, request);
                    let message = format!("instance {} is unknown or stale", request.instance_id);
                    request.completion.reject_unsubmitted(message.clone());
                }
                if let Some(lease) = prepared_lease {
                    let _ = driver_lane.release(lease);
                }
                return true;
            }
        }
        // Post to the driver lane and enter the run-ahead pipe immediately:
        // the batch occupies a depth slot and its instances' in-flight
        // counts (close gating) from POST, and the driver's verdict arrives
        // as a `LaneLaunchDone` reply. Target epochs commit at ACCEPT, not
        // here: an instance's epoch ledger must match the driver's launch
        // acceptance order exactly (the completion settles when the
        // instance slot's published count reaches its target, so a gap from
        // a rejected launch would hang every later fire on the instance) —
        // and lane replies arrive in post order, which IS the driver's
        // acceptance order.
        let membership_hash = if timing_enabled {
            fire_membership_hash(requests.iter().map(|request| &request.logical_fire_id))
        } else {
            0
        };
        let wave_timing = dispatch_started_us.map(|dispatch_started_us| WaveTimingState {
            // Filled in when the lane's reply carries the completion.
            wave_id: 0,
            membership_hash,
            dispatch_started_us,
            batch_built_us: batch_built_us.unwrap_or(dispatch_started_us),
            driver_started_us: dispatch_started_us,
            launch_returned_us: dispatch_started_us,
            decision_us,
            active_pipelines,
            missing_pipelines,
            candidate_count,
            deferred_pipelines,
            depth_capped_pipelines,
        });
        for request in &requests {
            if let Some(instance) = instances.get_mut(&request.instance_id) {
                instance.in_flight += 1;
            }
        }
        *lane_token += 1;
        let token = *lane_token;
        in_flight_launches.push_back(PendingLaunchBatch {
            state: LaunchState::Posted { token },
            requests,
            pipeline_epochs: Vec::new(),
            started: Instant::now(),
            batch_size,
            total_tokens,
            timing: wave_timing,
        });
        *lane_inflight += 1;
        driver_lane.post(LaneRequest::Launch {
            token,
            submission: LaneLaunch(submission),
            lease: prepared_lease,
        });
        true
    }

    fn retire_ready_launches(
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut VecDeque<QueuedItem>,
        stats: &Arc<SchedulerStats>,
        policy: &mut quorum::WaitAllPolicy,
        stopping: bool,
    ) -> bool {
        let mut progress = false;
        while let Some(front) = in_flight_launches.front() {
            // A lane-rejected launch retires like a wave: it entered the
            // pipe (depth, credits, instance accounting) at post, so the
            // common unwind below applies; only its requests' settlement
            // differs (rejected, never submitted).
            let launch_failure = match &front.state {
                LaunchState::Posted { .. } => break,
                LaunchState::Failed(message) => Some(message.clone()),
                LaunchState::Accepted(_) => None,
            };
            let result = match &front.state {
                LaunchState::Accepted(completion) => {
                    let Some(result) = completion.check() else {
                        break;
                    };
                    Some(result)
                }
                _ => None,
            };
            let retired = in_flight_launches.pop_front().expect("front batch exists");
            let native_complete_us = retired.timing.as_ref().map(|_| super::fire_timing_now_us());
            let timing_snapshots = retired
                .timing
                .as_ref()
                .map(|_| Self::fire_timing_snapshots(&retired.requests));
            policy.on_wave_retired(&retired.pipeline_epochs);
            for request in &retired.requests {
                if let Some(instance) = instances.get_mut(&request.instance_id) {
                    instance.in_flight = instance.in_flight.saturating_sub(1);
                }
            }
            if let Some(message) = launch_failure {
                if let (Some(timing), Some(native_complete_us), Some(snapshots)) =
                    (retired.timing, native_complete_us, timing_snapshots)
                {
                    let settled_us = super::fire_timing_now_us();
                    Self::emit_fire_timing(
                        &snapshots,
                        timing,
                        false,
                        native_complete_us,
                        settled_us,
                        &vec!["launch_error"; snapshots.len()],
                        retired.batch_size,
                        retired.total_tokens,
                        policy.untracked_ready_count(),
                        &[],
                    );
                }
                let message = format!("direct launch rejected: {message}");
                for request in &retired.requests {
                    request.completion.reject_unsubmitted(message.clone());
                }
                progress = true;
                continue;
            }
            let result = result.expect("accepted batch carries a settled result");
            match result {
                Ok(()) => {
                    for request in &retired.requests {
                        request.completion.mark_native_retired();
                    }
                    let mut retries = Vec::new();
                    let mut outcomes = Vec::with_capacity(retired.requests.len());
                    let mut token_instance_ids = Vec::new();
                    for mut request in retired.requests {
                        match request.completion.resolve_from_terminal() {
                            Ok(WorkItemAttemptOutcome::Committed) => {
                                outcomes.push("committed");
                                if !request.request.sampling_indices.is_empty() {
                                    token_instance_ids.push(request.instance_id);
                                }
                            }
                            Ok(WorkItemAttemptOutcome::Failed) => {
                                outcomes.push("failed");
                            }
                            Ok(WorkItemAttemptOutcome::Retry) => {
                                outcomes.push("retry");
                                request.retry_count += 1;
                                if let Some(timing) = request.timing.as_mut() {
                                    timing.ready_us = None;
                                }
                                if request.completion.cancel_requested() {
                                    request
                                        .completion
                                        .reject("logical fire cancelled after native attempt");
                                } else if stopping {
                                    request
                                        .completion
                                        .reject("scheduler shutdown interrupted a retrying fire");
                                } else if !request.retry_eligible() {
                                    request.completion.reject(
                                        "driver requested RETRY for an RS-carrying, retry-ineligible fire",
                                    );
                                } else if let Some(reason) = request
                                    .retry_classifier
                                    .as_ref()
                                    .and_then(|classify| classify())
                                {
                                    request.completion.reject(format!(
                                        "driver requested RETRY for a permanent channel condition: {reason}"
                                    ));
                                } else if request.retry_count > max_fire_retries() {
                                    request.completion.reject(format!(
                                        "fire exceeded retry limit {}",
                                        max_fire_retries()
                                    ));
                                } else {
                                    if request.retry_count == retry_warn_at() {
                                        tracing::warn!(
                                            instance_id = request.instance_id,
                                            retries = request.retry_count,
                                            "logical fire remains uncommitted and will retry"
                                        );
                                    }
                                    // Preserve the admission generation. A
                                    // fire retrying after graceful close stays
                                    // stale and therefore untracked; close must
                                    // not re-create the wait-set row it released.
                                    if !Self::request_needs_prelaunch(&request) {
                                        let qpid = Self::quorum_pid(
                                            policy,
                                            request.pipeline_id,
                                            request.quorum_generation,
                                        );
                                        policy.on_pipeline_join_owned(qpid, request.process_id);
                                        policy.on_pipeline_request_owned(
                                            qpid,
                                            request.process_id,
                                            Instant::now(),
                                        );
                                        request.credit_published = true;
                                        if let Some(timing) = request.timing.as_mut() {
                                            timing.ready_us = Some(super::fire_timing_now_us());
                                        }
                                    }
                                    request.retry_after = Some(
                                        Instant::now() + Self::retry_backoff(request.retry_count),
                                    );
                                    retries.push(request);
                                }
                            }
                            Err(err) => {
                                outcomes.push("settlement_error");
                                tracing::warn!(
                                    instance_id = request.instance_id,
                                    ?err,
                                    "direct launch terminal settlement failed"
                                );
                            }
                        }
                    }
                    for request in retries.into_iter().rev() {
                        Self::queue_attempt(pending, request, QueueEnd::Front);
                    }
                    if let (Some(timing), Some(native_complete_us), Some(snapshots)) =
                        (retired.timing, native_complete_us, timing_snapshots)
                    {
                        let settled_us = super::fire_timing_now_us();
                        Self::emit_fire_timing(
                            &snapshots,
                            timing,
                            true,
                            native_complete_us,
                            settled_us,
                            &outcomes,
                            retired.batch_size,
                            retired.total_tokens,
                            policy.untracked_ready_count(),
                            &token_instance_ids,
                        );
                    }
                    stats::record_fire_stats(
                        stats,
                        retired.started.elapsed(),
                        retired.batch_size,
                        retired.total_tokens,
                    )
                }

                Err(err) => {
                    if let (Some(timing), Some(native_complete_us), Some(snapshots)) =
                        (retired.timing, native_complete_us, timing_snapshots)
                    {
                        let settled_us = super::fire_timing_now_us();
                        Self::emit_fire_timing(
                            &snapshots,
                            timing,
                            true,
                            native_complete_us,
                            settled_us,
                            &vec!["completion_error"; snapshots.len()],
                            retired.batch_size,
                            retired.total_tokens,
                            policy.untracked_ready_count(),
                            &[],
                        );
                    }
                    tracing::warn!(?err, "direct launch completion closed before callback");
                    for request in &retired.requests {
                        request.completion.reject(format!(
                            "direct launch batch callback closed before terminal settlement: {err:#}"
                        ));
                        if let Some(instance) = instances.get(&request.instance_id) {
                            instance.wait_slots.close();
                        }
                    }
                }
            }
            progress = true;
        }
        progress
    }

    fn fire_timing_snapshots(requests: &[PendingRequest]) -> Vec<FireTimingSnapshot> {
        requests
            .iter()
            .enumerate()
            .filter_map(|(outcome_index, request)| {
                request.timing.map(|timing| FireTimingSnapshot {
                    outcome_index,
                    logical_fire_id: request.logical_fire_id,
                    instance_id: request.instance_id,
                    process_id: request.process_id,
                    sampled_rows: request.request.sampling_indices.len(),
                    retry_count: request.retry_count,
                    timing,
                })
            })
            .collect()
    }

    fn emit_fire_timing(
        requests: &[FireTimingSnapshot],
        timing: WaveTimingState,
        cuda_submitted: bool,
        native_complete_us: u64,
        settled_us: u64,
        outcomes: &[&str],
        batch_size: u64,
        total_tokens: usize,
        untracked_ready: usize,
        token_instance_ids: &[u64],
    ) {
        let committed = outcomes
            .iter()
            .filter(|&&outcome| outcome == "committed")
            .count();
        let retried = outcomes
            .iter()
            .filter(|&&outcome| outcome == "retry")
            .count();
        let failed = outcomes.len().saturating_sub(committed + retried);
        let mut record = serde_json::json!({
            "schema": 1,
            "source": "scheduler",
            "event": "scheduler_wave",
            "wave_id": timing.wave_id,
            "membership_hash": timing.membership_hash,
            "cuda_submitted": cuda_submitted,
            "fire_count": batch_size,
            "batch_size": batch_size,
            "tokens": total_tokens,
            "committed": committed,
            "retried": retried,
            "failed": failed,
            "dispatch_started_us": timing.dispatch_started_us,
            "batch_built_us": timing.batch_built_us,
            "driver_started_us": timing.driver_started_us,
            "launch_returned_us": timing.launch_returned_us,
            "native_complete_us": native_complete_us,
            "settled_us": settled_us,
            "batch_build_us": timing
                .batch_built_us
                .saturating_sub(timing.dispatch_started_us),
            "driver_submit_us": timing
                .launch_returned_us
                .saturating_sub(timing.driver_started_us),
            "native_inflight_us": native_complete_us
                .saturating_sub(timing.launch_returned_us),
            "retire_settle_us": settled_us.saturating_sub(native_complete_us),
            "decision_us": timing.decision_us,
            "active_pipelines": timing.active_pipelines,
            "missing_pipelines": timing.missing_pipelines,
            "candidate_count": timing.candidate_count,
            "deferred_pipelines": timing.deferred_pipelines,
            "depth_capped_pipelines": timing.depth_capped_pipelines,
            // W1 leak gate: must be 0 whenever no untracked fires are in
            // the gather (a monotonic climb here is the close-leave leak).
            "untracked_ready": untracked_ready,
        });
        if super::ledger_timing_enabled() {
            record["token_instance_ids"] = serde_json::json!(token_instance_ids);
        }
        super::fire_timing_write(&record);
        for request in requests {
            let outcome = outcomes
                .get(request.outcome_index)
                .copied()
                .unwrap_or("unknown");
            let fire = request.timing;
            super::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "scheduler",
                "event": "scheduler_fire",
                "wave_id": timing.wave_id,
                "logical_fire_id": request.logical_fire_id,
                "instance_id": request.instance_id,
                "process_id": request.process_id,
                "sampled_rows": request.sampled_rows,
                "attempt": request.retry_count,
                "preparation_retries": 0,
                "outcome": outcome,
                "submitted_us": fire.submitted_us,
                "enqueued_us": fire.enqueued_us,
                "prepare_started_us": null,
                "prepared_us": null,
                "ready_us": fire.ready_us,
                "native_complete_us": native_complete_us,
                "settled_us": settled_us,
                "submit_to_enqueue_us": fire
                    .enqueued_us
                    .map(|value| value.saturating_sub(fire.submitted_us)),
                "prepare_us": 0,
                "ready_to_dispatch_us": fire.ready_us
                    .map(|ready| timing.dispatch_started_us.saturating_sub(ready)),
            }));
        }
    }

    fn retire_ready_control(
        in_flight_control: &mut Option<PendingControl>,
        pending: &mut VecDeque<QueuedItem>,
        policy: &mut quorum::WaitAllPolicy,
    ) -> bool {
        let operation = in_flight_control
            .as_ref()
            .map(|pending| pending.operation)
            .unwrap_or("control operation");
        let Some(result) = in_flight_control
            .as_ref()
            .and_then(|pending| match &pending.state {
                // Still waiting for the lane's reply to install the driver
                // completion (or clear the slot on rejection).
                ControlSlotState::Posted { .. } => None,
                ControlSlotState::Ready(completion) => completion.check(),
            })
        else {
            return false;
        };
        if let Some(tracked) = in_flight_control
            .as_ref()
            .and_then(|pending| pending.tracked_completion.as_ref())
        {
            tracked.resolve(&result);
        }
        if result.is_ok()
            && in_flight_control
                .as_ref()
                .is_some_and(|pending| pending.credit_ready)
        {
            // A protected control retiring after its pipeline left must not
            // re-arm the barrier on a ghost; the generation route publishes
            // its coupled launch's credit untracked (W1; covers Close,
            // which the departed-set guard never did).
            let qpid = in_flight_control.as_ref().and_then(|pending| {
                Self::quorum_pid(policy, pending.pipeline_id, pending.quorum_generation)
            });
            let owner = in_flight_control
                .as_ref()
                .and_then(|pending| pending.process_id);
            policy.on_pipeline_join_owned(qpid, owner);
            policy.on_pipeline_request_owned(qpid, owner, Instant::now());
            if let Some(completion) = in_flight_control
                .as_ref()
                .and_then(|control| control.logical_completion.as_ref())
            {
                // The credit published above belongs to this control's
                // coupled consumer launch, still queued behind the control
                // slot: mark it as the credit holder so a later drop of
                // that launch gives the credit back (RV-20).
                let ready_us = super::fire_timing_enabled().then(super::fire_timing_now_us);
                for item in pending.iter_mut() {
                    let QueuedItem::Launch(request) = item else {
                        continue;
                    };
                    if request.completion.same_request(completion) {
                        request.credit_published = true;
                        if let (Some(timing), Some(ready_us)) = (request.timing.as_mut(), ready_us)
                        {
                            timing.ready_us = Some(ready_us);
                        }
                        break;
                    }
                }
            }
        }
        if let Err(ref err) = result {
            tracing::warn!(
                ?err,
                operation,
                "direct control completion closed before callback"
            );
            if let Some(logical) = in_flight_control
                .as_ref()
                .and_then(|pending| pending.logical_completion.as_ref())
            {
                logical.reject_unsubmitted(format!("pre-launch {operation} failed: {err:#}"));
            }
        }
        *in_flight_control = None;
        true
    }

    /// Apply a driver-lane reply on the worker thread: fill in a posted
    /// launch's verdict, commit a control's worker-map effects, or install an
    /// async control's driver completion. Replies arrive in lane FIFO order.
    fn apply_lane_reply(
        reply: LaneReply,
        lane_inflight: &mut u64,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &mut Option<PendingControl>,
        instances: &mut HashMap<u64, TrackedInstance>,
        policy: &mut quorum::WaitAllPolicy,
        rollback_tx: &crossbeam::channel::Sender<SchedulerItem>,
    ) {
        *lane_inflight = lane_inflight.saturating_sub(1);
        match reply {
            LaneReply::LaunchDone {
                token,
                result,
                driver_started_us,
                launch_returned_us,
            } => {
                let Some(batch) = in_flight_launches.iter_mut().find(
                    |batch| matches!(batch.state, LaunchState::Posted { token: t } if t == token),
                ) else {
                    // The batch can only leave the deque by retiring, and a
                    // Posted batch never retires — a missing token is a bug.
                    tracing::error!(token, "lane launch reply for an unknown batch");
                    return;
                };
                match result {
                    Ok(completion) => {
                        // Commit target epochs AT ACCEPT: lane replies arrive
                        // in post order — the driver's launch acceptance
                        // order — so the per-instance ledger stays gapless
                        // (a rejected launch commits nothing) and each
                        // completion's target matches the ordinal the
                        // instance slot will publish.
                        for request in &batch.requests {
                            if let Some(instance) = instances.get_mut(&request.instance_id) {
                                let epoch = instance.next_target_epoch;
                                request.completion.commit_target_epoch(epoch);
                                instance.next_target_epoch = epoch + 1;
                            }
                        }
                        if let Some(timing) = batch.timing.as_mut() {
                            timing.wave_id = completion.wait_id();
                            if let Some(at) = driver_started_us {
                                timing.driver_started_us = at;
                            }
                            if let Some(at) = launch_returned_us {
                                timing.launch_returned_us = at;
                            }
                        }
                        batch.state = LaunchState::Accepted(completion);
                    }
                    Err(message) => {
                        if let (Some(timing), Some(at)) =
                            (batch.timing.as_mut(), launch_returned_us)
                        {
                            timing.launch_returned_us = at;
                            if let Some(started) = driver_started_us {
                                timing.driver_started_us = started;
                            }
                        }
                        batch.state = LaunchState::Failed(message);
                    }
                }
            }
            LaneReply::ControlDone { token, commit } => match commit {
                LaneCommit::None => {}
                LaneCommit::BindFinished { pipeline_id } => {
                    policy.on_bind_completed(pipeline_id, Instant::now());
                }
                LaneCommit::BindInstance {
                    pipeline_id,
                    bound,
                    respond,
                } => {
                    policy.on_bind_completed(pipeline_id, Instant::now());
                    if instances.contains_key(&bound.instance_id) {
                        // Practically unreachable: driver-assigned ids are
                        // unique and requested ids are pre-checked at post
                        // (a guest awaits its bind response before it could
                        // reuse an id). Refuse loudly; the legit instance in
                        // the map stays untouched.
                        tracing::error!(
                            instance_id = bound.instance_id,
                            "bind committed an already-bound instance id"
                        );
                        let error = anyhow!("instance {} is already bound", bound.instance_id);
                        match respond {
                            BindRespond::Bind(response) => {
                                let _ = response.send(Err(error));
                            }
                            BindRespond::ChannelsBind { response, .. } => {
                                let _ = response.send(Err(error));
                            }
                        }
                        return;
                    }
                    let instance_id = bound.instance_id;
                    instances.insert(instance_id, TrackedInstance::from_bound(&bound));
                    // Respond AFTER the insert: launch admission reads
                    // `instances` on this thread, so the guest's first fire
                    // (sent only after this response) is always admissible.
                    match respond {
                        BindRespond::Bind(response) => {
                            if let Err(Ok(bound)) = response.send(Ok(bound)) {
                                tracing::warn!(
                                    operation = "bind_instance",
                                    instance_id = bound.instance_id,
                                    "scheduler cancellation rollback enqueued bound instance"
                                );
                                if rollback_tx
                                    .send(SchedulerItem::CloseInstance {
                                        id: bound.instance_id,
                                        pacing_wait_id: bound.pacing_wait_id,
                                    })
                                    .is_err()
                                {
                                    tracing::error!(
                                        operation = "bind_instance",
                                        instance_id = bound.instance_id,
                                        "scheduler cancellation rollback enqueue failed"
                                    );
                                }
                            }
                        }
                        BindRespond::ChannelsBind {
                            registered,
                            program_id,
                            program_registered,
                            response,
                        } => {
                            if let Err(Ok((registered, _, bound))) =
                                response.send(Ok((registered, program_id, bound)))
                            {
                                tracing::warn!(
                                    operation = "register_channels_bind",
                                    instance_id = bound.instance_id,
                                    channel_count = registered.len(),
                                    "scheduler cancellation rollback enqueued bound instance and channels"
                                );
                                if program_registered {
                                    tracing::warn!(
                                        operation = "register_channels_bind",
                                        program_id,
                                        "scheduler RPC cancelled after program registration; retaining driver-lifetime program"
                                    );
                                }
                                DriverLane::release_registered_channel_wait_slots(&registered);
                                let instance_id = bound.instance_id;
                                if rollback_tx
                                    .send(SchedulerItem::CloseInstance {
                                        id: instance_id,
                                        pacing_wait_id: bound.pacing_wait_id,
                                    })
                                    .is_err()
                                {
                                    tracing::error!(
                                        operation = "register_channels_bind",
                                        instance_id,
                                        "scheduler cancellation rollback close_instance enqueue failed"
                                    );
                                }
                                for channel in registered {
                                    let channel_id = channel.binding.channel_id;
                                    if rollback_tx
                                        .send(SchedulerItem::CloseChannel { id: channel_id })
                                        .is_err()
                                    {
                                        tracing::error!(
                                            operation = "register_channels_bind",
                                            channel_id,
                                            "scheduler cancellation rollback close_channel enqueue failed"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                LaneCommit::CloseInstance { id } => {
                    if let Some(instance) = instances.remove(&id) {
                        instance.close_wait_slots();
                    }
                }
                LaneCommit::AsyncControl { result } => {
                    let holds_token = in_flight_control.as_ref().is_some_and(
                        |control| matches!(control.state, ControlSlotState::Posted { token: t } if t == token),
                    );
                    if !holds_token {
                        tracing::error!(
                            token,
                            "lane async-control reply without a matching control slot"
                        );
                        return;
                    }
                    match result {
                        Ok(completion) => {
                            if let Some(control) = in_flight_control.as_mut() {
                                control.state = ControlSlotState::Ready(completion);
                            }
                        }
                        // The lane already rejected/resolved the control's
                        // completions; the slot just frees.
                        Err(_) => *in_flight_control = None,
                    }
                }
            },
        }
    }

    fn shutdown_instances(
        driver: &mut Option<DriverBackend>,
        instances: &mut HashMap<u64, TrackedInstance>,
    ) {
        let outstanding = std::mem::take(instances);
        for (instance_id, instance) in outstanding {
            if let Some(driver) = driver.as_mut() {
                if let Err(err) = driver.close_instance(instance_id) {
                    tracing::warn!(
                        instance_id,
                        ?err,
                        "scheduler shutdown close_instance failed"
                    );
                }
            }
            instance.close_wait_slots();
        }
    }

    fn shutdown_channels(driver: &mut Option<DriverBackend>, channels: &mut HashSet<u64>) {
        let outstanding = std::mem::take(channels);
        for channel_id in outstanding {
            if let Some(driver) = driver.as_mut()
                && let Err(err) = driver.close_channel(channel_id)
            {
                tracing::warn!(channel_id, ?err, "scheduler shutdown close_channel failed");
            }
        }
    }
}

impl Drop for BatchScheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

struct TrackedInstance {
    pacing_wait_id: u64,
    wait_slots: Arc<crate::driver::instance::BoundWaitSlots>,
    in_flight: usize,
    next_target_epoch: u64,
}

impl TrackedInstance {
    fn from_bound(bound: &BoundInstance) -> Self {
        Self {
            pacing_wait_id: bound.pacing_wait_id,
            wait_slots: bound.wait_slots(),
            in_flight: 0,
            next_target_epoch: pie_waker::FIRST_COMPLETION_EPOCH,
        }
    }

    fn close_wait_slots(self) {
        self.wait_slots.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{
        self, ChannelValue, DriverSpec, LaunchPlan, ProgramRegistration, SchedulerLimits,
    };
    use pie_driver_abi::{PieInstanceBinding, PieKvMoveCell, PiePoolRange};
    use pie_driver_dummy_lib::DummyDriverOptions;
    use pie_ptir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use pie_ptir::op::Op;
    use pie_ptir::registry::Stage;
    use pie_ptir::types::{DType, Literal, Shape};
    use tokio::time::{Duration, timeout};

    async fn setup_scheduler(
        operation_log: Arc<Mutex<Vec<String>>>,
    ) -> anyhow::Result<(
        usize,
        BatchScheduler,
        crate::driver::BoundInstance,
        Vec<Arc<crate::driver::ChannelEndpoint>>,
    )> {
        setup_scheduler_with_options(DummyDriverOptions {
            operation_log: Some(operation_log),
            ..DummyDriverOptions::default()
        })
        .await
    }

    fn dummy_launch() -> LaunchPlan {
        LaunchPlan {
            token_ids: vec![1],
            position_ids: vec![0],
            kv_page_indptr: vec![0, 0],
            kv_last_page_lens: vec![0],
            qo_indptr: vec![0, 1],
            sampling_indices: vec![0],
            sampling_indptr: vec![0, 1],
            mask_indptr: vec![0, 0],
            single_token_mode: true,
            ..LaunchPlan::default()
        }
    }

    fn dummy_prefill(tokens: usize) -> LaunchPlan {
        let mut launch = dummy_launch();
        launch.token_ids = vec![1; tokens];
        launch.position_ids = (0..tokens as u32).collect();
        launch.qo_indptr = vec![0, tokens as u32];
        launch.sampling_indices = vec![tokens.saturating_sub(1) as u32];
        launch.single_token_mode = false;
        launch
    }

    /// Test lane over a driverless backend plus the reply stream the worker
    /// loop would normally drain.
    fn test_lane(
        driver: Option<DriverBackend>,
    ) -> (DriverLane, crossbeam::channel::Receiver<SchedulerItem>) {
        let (reply_tx, reply_rx) = crossbeam::channel::unbounded();
        let lane = DriverLane::spawn(
            usize::MAX,
            driver,
            reply_tx,
            Arc::new(SchedulerStats::default()),
        );
        (lane, reply_rx)
    }

    async fn wait_for_operation_count(
        operation_log: &Arc<Mutex<Vec<String>>>,
        operation: &str,
        count: usize,
    ) {
        timeout(Duration::from_secs(5), async {
            loop {
                if operation_log
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|entry| entry.as_str() == operation)
                    .count()
                    >= count
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("scheduler operation must complete");
    }

    fn chan(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 2,
            host_role: role,
            seeded,
        }
    }

    fn dummy_program() -> ProgramRegistration {
        let bytes = TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::U32, HostRole::None, true),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::Const(Literal::U32(1)),
                    Op::Add(0, 1),
                    Op::ChanPut { chan: 0, value: 2 },
                    Op::ChanPut { chan: 1, value: 2 },
                ],
            }],
        }
        .encode();
        ProgramRegistration {
            program_hash: pie_ptir::container_hash(&bytes),
            canonical_bytes: bytes,
            sidecar_bytes: Vec::new(),
        }
    }

    async fn register_test_channels(
        driver_id: usize,
        channel_ids: [u64; 2],
    ) -> anyhow::Result<Vec<Arc<crate::driver::ChannelEndpoint>>> {
        let mut endpoints = Vec::new();
        for (channel_id, host_role, seeded) in [
            (channel_ids[0], HostRole::None, true),
            (channel_ids[1], HostRole::Reader, false),
        ] {
            endpoints.push(
                crate::scheduler::register_channel(
                    driver_id,
                    ChannelRegistrationPlan {
                        driver_id,
                        channel_id,
                        shape: vec![1],
                        dtype: pie_driver_abi::PIE_CHANNEL_DTYPE_U32,
                        host_role: host_role as u8,
                        seeded,
                        extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_NONE,
                        capacity: 2,
                        reader_wait_id: 0,
                        writer_wait_id: 0,
                        extern_name: Vec::new(),
                    },
                )
                .await?,
            );
        }
        Ok(endpoints)
    }

    async fn setup_scheduler_with_options(
        options: DummyDriverOptions,
    ) -> anyhow::Result<(
        usize,
        BatchScheduler,
        crate::driver::BoundInstance,
        Vec<Arc<crate::driver::ChannelEndpoint>>,
    )> {
        setup_scheduler_with_limits(
            options,
            SchedulerLimits {
                max_forward_requests: 1,
                max_forward_tokens: 64,
                max_page_refs: 64,
            },
        )
        .await
    }

    /// Like [`setup_scheduler_with_options`], but with a caller-chosen
    /// `SchedulerLimits` — the quorum wait-all rule's structural cap
    /// (`max_forward_requests`) short-circuits any cold-hold/wait-all
    /// delay once a batch saturates it (see `quorum::tests::
    /// structural_cap_fires_immediately_even_cold`), so every other test in
    /// this module runs at cap 1 and never observes the quorum hold. Tests
    /// that need to actually exercise the hold (coalescing/leave)
    /// use this with a cap > 1 instead.
    async fn setup_scheduler_with_limits(
        options: DummyDriverOptions,
        limits: SchedulerLimits,
    ) -> anyhow::Result<(
        usize,
        BatchScheduler,
        crate::driver::BoundInstance,
        Vec<Arc<crate::driver::ChannelEndpoint>>,
    )> {
        let driver_id = driver::register_driver_backend(
            DriverSpec {
                num_kv_pages: 16,
                limits,
                device_geometry_port_mask: 0,
            },
            DriverBackend::Dummy(crate::driver::DummyDriver::new(options)),
        );
        let scheduler = BatchScheduler::new(driver_id, driver_id, 16, limits, 1);
        let program_id = crate::scheduler::register_program(driver_id, dummy_program()).await?;
        let endpoints = register_test_channels(driver_id, [7, 8]).await?;
        let bound = crate::scheduler::bind_instance(
            driver_id,
            None,
            program_id,
            41,
            vec![7, 8],
            vec![ChannelValue {
                channel: 7,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;
        Ok((driver_id, scheduler, bound, endpoints))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn typed_copy_paths_dispatch_to_distinct_driver_methods() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;

        let copy_kv = crate::scheduler::copy_kv_cells(
            driver_id,
            vec![PieKvMoveCell {
                dst_page_id: 1,
                dst_token_offset: 0,
                src_page_id: 2,
                src_token_offset: 0,
            }],
        )
        .await?;
        timeout(Duration::from_secs(5), copy_kv).await??;
        let copy_state = crate::scheduler::copy_rs_d2d(driver_id, &[3], &[4]).await?;
        timeout(Duration::from_secs(5), copy_state).await??;
        crate::scheduler::close_instance(&bound)?;

        let log = operation_log.lock().unwrap().clone();
        let copy_kv_idx = log
            .iter()
            .position(|entry| entry == "copy_kv")
            .expect("copy_kv logged");
        let copy_state_idx = log
            .iter()
            .position(|entry| entry == "copy_state")
            .expect("copy_state logged");
        assert!(
            copy_kv_idx < copy_state_idx,
            "copy_kv should precede copy_state: {log:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn resize_ops_run_before_queued_launches() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;

        let resize = crate::scheduler::resize_pool(
            driver_id,
            7,
            32,
            vec![PiePoolRange {
                page_index: 0,
                page_count: 4,
            }],
            Vec::new(),
        )
        .await?;
        let launch = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            launch.clone(),
        )?;

        timeout(Duration::from_secs(5), resize).await??;
        timeout(Duration::from_secs(5), launch).await??;
        crate::scheduler::close_instance(&bound)?;

        let log = operation_log.lock().unwrap().clone();
        let resize_idx = log
            .iter()
            .position(|entry| entry == "resize_pool")
            .expect("resize_pool logged");
        let launch_idx = log
            .iter()
            .position(|entry| entry == "launch")
            .expect("launch logged");
        assert!(
            resize_idx < launch_idx,
            "resize should precede launch: {log:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_instance_retires_bound_wait_slots() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, _scheduler, bound, _endpoints) = setup_scheduler(operation_log).await?;
        let pacing_wait_id = bound.pacing_wait_id;
        crate::scheduler::close_instance(&bound)?;

        timeout(Duration::from_secs(5), async {
            while pie_waker::WakerTable::global()
                .published(pacing_wait_id)
                .is_some()
            {
                tokio::task::yield_now().await;
            }
        })
        .await?;
        Ok(())
    }

    #[test]
    fn cancelled_register_channel_releases_wait_slots_before_creation() {
        let table = pie_waker::WakerTable::global();
        let reader_wait_id = table.alloc();
        let writer_wait_id = table.alloc();
        let (response, receiver) = tokio::sync::oneshot::channel();
        drop(receiver);
        let mut driver = None;
        let mut channels = HashSet::new();

        let commit = DriverLane::execute_control(
            &mut driver,
            &mut channels,
            QueuedItem::RegisterChannel {
                plan: ChannelRegistrationPlan {
                    driver_id: 0,
                    channel_id: 91,
                    shape: vec![1],
                    dtype: pie_driver_abi::PIE_CHANNEL_DTYPE_U32,
                    host_role: HostRole::None as u8,
                    seeded: false,
                    extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_NONE,
                    capacity: 1,
                    reader_wait_id,
                    writer_wait_id,
                    extern_name: Vec::new(),
                },
                response,
            },
        );

        assert!(matches!(commit, LaneCommit::None));
        assert!(table.published(reader_wait_id).is_none());
        assert!(table.published(writer_wait_id).is_none());
        assert!(channels.is_empty());
    }

    #[test]
    fn cancelled_bind_response_enqueues_instance_rollback() {
        let pacing_wait_id = pie_waker::WakerTable::global().alloc();
        let bound = BoundInstance::new(
            7,
            11,
            PieInstanceBinding {
                instance_id: 41,
                geometry_class: pie_driver_abi::GeometryClass::Host as u32,
                reserved0: 0,
            },
            pacing_wait_id,
        );
        let (response, receiver) = tokio::sync::oneshot::channel();
        drop(receiver);
        let (rollback_tx, rollback_rx) = crossbeam::channel::unbounded();
        let mut lane_inflight = 1;
        let mut launches = VecDeque::new();
        let mut control = None;
        let mut instances = HashMap::new();
        let mut policy = quorum::WaitAllPolicy::new(1, None);

        BatchScheduler::apply_lane_reply(
            LaneReply::ControlDone {
                token: 1,
                commit: LaneCommit::BindInstance {
                    pipeline_id: None,
                    bound,
                    respond: BindRespond::Bind(response),
                },
            },
            &mut lane_inflight,
            &mut launches,
            &mut control,
            &mut instances,
            &mut policy,
            &rollback_tx,
        );

        assert!(matches!(
            rollback_rx.try_recv(),
            Ok(SchedulerItem::CloseInstance {
                id: 41,
                pacing_wait_id: wait_id,
            }) if wait_id == pacing_wait_id
        ));
        instances
            .remove(&41)
            .expect("cancelled bind remains tracked until ordered rollback")
            .close_wait_slots();
        assert!(
            pie_waker::WakerTable::global()
                .published(pacing_wait_id)
                .is_none()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn duplicate_bind_preserves_original_instance() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;

        let error = crate::scheduler::bind_instance(
            driver_id,
            None,
            bound.program_id,
            bound.instance_id,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await
        .expect_err("duplicate requested instance id must be rejected");
        assert!(error.to_string().contains("already bound"));

        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion).await??;
        crate::scheduler::close_instance(&bound)?;

        let log = operation_log.lock().unwrap();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "bind_instance")
                .count(),
            1,
            "duplicate bind must be rejected before entering the backend"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_defers_slot_retirement_until_outstanding_completion_drops() -> anyhow::Result<()>
    {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, _scheduler, bound, _endpoints) = setup_scheduler(operation_log).await?;
        let pacing_wait_id = bound.pacing_wait_id;
        let outstanding = bound.reserve_completion();

        let close_bound = std::thread::spawn({
            let bound = bound;
            move || crate::scheduler::close_instance(&bound)
        });

        std::thread::sleep(Duration::from_millis(10));
        assert!(
            close_bound.is_finished(),
            "close must not block the scheduler on an externally held completion"
        );
        close_bound.join().unwrap()?;
        assert!(
            !matches!(
                pie_waker::WakerTable::global().publish(pacing_wait_id, 1),
                pie_waker::WakeOutcome::Stale
            ),
            "bound wait slots remain leased until the completion drops"
        );
        drop(outstanding);

        assert!(matches!(
            pie_waker::WakerTable::global().publish(pacing_wait_id, 2),
            pie_waker::WakeOutcome::Stale
        ));
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn published_completion_survives_nonblocking_close_before_late_poll() -> anyhow::Result<()>
    {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) = setup_scheduler(operation_log).await?;
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;

        timeout(Duration::from_secs(5), async {
            loop {
                if pie_waker::WakerTable::global()
                    .published(completion.wait_id())
                    .is_some_and(|epoch| epoch >= completion.target_epoch())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await?;

        let close_bound = std::thread::spawn({
            let bound = bound;
            move || crate::scheduler::close_instance(&bound)
        });
        std::thread::sleep(Duration::from_millis(10));
        assert!(
            close_bound.is_finished(),
            "published terminal cells do not require close to wait for a late poll"
        );
        close_bound.join().unwrap()?;

        timeout(Duration::from_secs(5), completion).await??;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn one_instance_multi_row_rs_launch_reaches_dummy_intact() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) = setup_scheduler_with_limits(
            DummyDriverOptions {
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            },
            SchedulerLimits {
                max_forward_requests: 2,
                max_forward_tokens: 64,
                max_page_refs: 64,
            },
        )
        .await?;
        let mut launch = dummy_launch();
        launch.token_ids = vec![1, 2];
        launch.position_ids = vec![0, 0];
        launch.qo_indptr = vec![0, 1, 2];
        launch.kv_page_indptr = vec![0, 0, 0];
        launch.kv_last_page_lens = vec![0, 0];
        launch.sampling_indices = vec![0, 1];
        launch.sampling_indptr = vec![0, 1, 2];
        launch.mask_indptr = vec![0, 0, 0];
        launch.rs_slot_ids = vec![7, 9];
        launch.rs_slot_flags = vec![crate::driver::RS_FLAG_RESET, 0];

        let completion = bound.reserve_completion();
        crate::scheduler::submit_async(
            launch,
            driver_id,
            bound.instance_id,
            0,
            None,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion).await??;

        assert!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .any(|entry| entry.starts_with("launch-shape tokens=2 programs=1"))
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn synchronous_launch_rejection_has_no_callback_or_epoch_gap() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                reject_launches_remaining: 1,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;

        let rejected = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            rejected.clone(),
        )?;
        let err = timeout(Duration::from_secs(5), rejected.clone())
            .await?
            .expect_err("rejected launch must fail");
        assert!(err.to_string().contains("direct launch rejected"));
        assert_eq!(
            rejected.target_epoch(),
            0,
            "rejected launch must not commit an epoch"
        );

        let accepted = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            accepted.clone(),
        )?;
        timeout(Duration::from_secs(5), accepted.clone()).await??;
        assert_eq!(
            accepted.target_epoch(),
            pie_waker::FIRST_COMPLETION_EPOCH,
            "the first accepted launch must still claim the first completion epoch"
        );

        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "callback")
                .count(),
            1,
            "only the accepted launch may emit a callback: {log:?}"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn retry_requeues_the_same_logical_fire_without_publishing_effects() -> anyhow::Result<()>
    {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                retry_launches_remaining: 1,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;

        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async_with_kv_copy(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
            vec![1],
            vec![2],
        )?;
        timeout(Duration::from_secs(5), completion.clone()).await??;

        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            2
        );
        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "copy_kv")
                .count(),
            2,
            "every attempt replays the pre-launch snapshot copy"
        );
        assert_eq!(
            completion.target_epoch(),
            pie_waker::FIRST_COMPLETION_EPOCH + 1
        );
        let binding = endpoints[1].registered().binding;
        let tail = unsafe {
            (&*((binding.word_base as *const std::sync::atomic::AtomicU64)
                .add(binding.tail_word_index as usize)))
                .load(Ordering::Acquire)
        };
        let value = unsafe { std::ptr::read_unaligned(binding.mirror_base as *const u32) };
        assert_eq!(tail, 1, "the RETRY attempt publishes no reader cell");
        assert_eq!(value, 2, "the retried fire executes exactly once logically");
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exhausted_admission_preserves_wave_books_and_wakes_later() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                elastic_admission: true,
                prepare_exhaustions_remaining: 1,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let stats = Arc::clone(scheduler.stats());
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion.clone()).await??;

        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "prepare_launch-exhausted")
                .count(),
            1,
            "{log:?}"
        );
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "prepare_launch-ready")
                .count(),
            1,
            "{log:?}"
        );
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "launch_prepared")
                .count(),
            1,
            "{log:?}"
        );
        assert_eq!(stats.total_batches.load(Ordering::Relaxed), 1);
        assert_eq!(
            stats.fire.quorum.wave_fires.load(Ordering::Relaxed),
            1,
            "the denied proposal must not count as a wave"
        );
        assert_eq!(
            completion.target_epoch(),
            pie_waker::FIRST_COMPLETION_EPOCH,
            "physical exhaustion is not an ordinary driver RETRY"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn impossible_admission_fails_without_parking() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                elastic_admission: true,
                prepare_impossible_above_kv_pages: 1,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let stats = Arc::clone(scheduler.stats());
        let mut launch = dummy_launch();
        launch.required_kv_pages = 2;
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            launch,
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        let error = timeout(Duration::from_secs(1), completion)
            .await?
            .expect_err("impossible demand must fail explicitly");
        assert!(error.to_string().contains("exceeding driver ceiling"));
        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "prepare_launch-impossible")
                .count(),
            1,
            "{log:?}"
        );
        assert!(!log.iter().any(|entry| entry == "launch_prepared"));
        assert_eq!(stats.total_batches.load(Ordering::Relaxed), 0);
        assert_eq!(
            stats.fire.quorum.wave_fires.load(Ordering::Relaxed),
            0,
            "an impossible proposal never commits wave statistics"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rs_carrying_fire_is_retry_ineligible() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                retry_launches_remaining: 1,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let mut launch = dummy_launch();
        launch.rs_slot_ids = vec![0];
        launch.rs_slot_flags = vec![crate::driver::RS_FLAG_RESET];
        launch.rs_fold_lens = vec![0];

        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            launch,
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        let error = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("RS non-commit must fail rather than retry");
        assert!(
            error.to_string().contains("retry-ineligible"),
            "unexpected error: {error:#}"
        );
        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            1
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ticket_mismatch_retries_later_fire_instead_of_stealing_predecessor_state()
    -> anyhow::Result<()> {
        let (driver_id, _scheduler, bound, endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                retry_launches_remaining: 1,
                callback_delay_ms: 20,
                ..DummyDriverOptions::default()
            })
            .await?;
        let mut first_plan = dummy_launch();
        first_plan.channel_expected_head = vec![0, crate::driver::command::CHANNEL_TICKET_NONE];
        first_plan.channel_expected_tail = vec![1, 0];
        let mut second_plan = dummy_launch();
        second_plan.channel_expected_head = vec![1, crate::driver::command::CHANNEL_TICKET_NONE];
        second_plan.channel_expected_tail = vec![2, 1];

        let first = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            first_plan,
            driver_id,
            bound.instance_id,
            0,
            first.clone(),
        )?;
        let second = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            second_plan,
            driver_id,
            bound.instance_id,
            0,
            second.clone(),
        )?;
        timeout(Duration::from_secs(5), first).await??;
        timeout(Duration::from_secs(5), second).await??;

        let binding = endpoints[1].registered().binding;
        let first_value = unsafe { std::ptr::read_unaligned(binding.mirror_base as *const u32) };
        let second_value =
            unsafe { std::ptr::read_unaligned((binding.mirror_base as *const u32).add(1)) };
        assert_eq!((first_value, second_value), (2, 3));
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn permanent_retry_escalates_to_failure() -> anyhow::Result<()> {
        let retries = max_fire_retries() + 1;
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                retry_launches_remaining: retries,
                ..DummyDriverOptions::default()
            })
            .await?;
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        let error = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("permanent retry must eventually fail");
        assert!(error.to_string().contains("exceeded retry limit"));
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancellation_stops_retry_after_the_live_attempt_retires() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                retry_launches_remaining: max_fire_retries(),
                callback_delay_ms: 20,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        tokio::time::sleep(Duration::from_millis(5)).await;
        completion.request_cancel();

        let error = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("cancelled retry must reject");
        assert!(error.to_string().contains("cancelled"));
        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            1,
            "cancellation waits for the live attempt but never launches another"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn permanent_retry_classifier_fails_without_burning_the_retry_budget()
    -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                retry_launches_remaining: max_fire_retries(),
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let classifier: crate::scheduler::RetryClassifier =
            Box::new(|| Some("writer endpoint closed".to_string()));
        let completion = bound.reserve_completion();
        scheduler.handle.submit_prebuilt_tracked_with_copy(
            dummy_launch(),
            bound.instance_id,
            completion.clone(),
            0,
            ProcessId::new_v4(),
            ProcessId::new_v4(),
            None,
            None,
            Some(classifier),
            false,
        )?;

        let error = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("permanent retry cause must reject");
        assert!(error.to_string().contains("writer endpoint closed"));
        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            1
        );
        Ok(())
    }

    #[test]
    fn termination_preserves_launch_until_its_inflight_prelaunch_copy_retires() {
        let pid = ProcessId::new_v4();
        let completion = WorkItemCompletion::deferred_with_guard(None);
        let request = PendingRequest::direct(
            dummy_launch(),
            1,
            completion.clone(),
            0,
            Some(pid),
            Some(pid),
            false,
            None,
            None,
            None,
            false,
        );
        let mut pending = VecDeque::from([QueuedItem::Launch(request)]);
        let mut policy = quorum::WaitAllPolicy::new(1, None);
        completion.request_cancel();
        BatchScheduler::reject_pipeline_queued(&mut pending, &mut policy, pid, Some(&completion));
        assert_eq!(pending.len(), 1);
        assert!(completion.cancel_requested());
        assert!(!completion.is_settled());
    }

    #[tokio::test]
    async fn tracked_control_completion_wakes_multiple_waiters() {
        let completion = ControlCompletion::new();
        let first = completion.clone();
        let second = completion.clone();
        let first = tokio::spawn(async move { first.wait().await });
        let second = tokio::spawn(async move { second.wait().await });
        tokio::task::yield_now().await;
        completion.resolve(&Ok(()));
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    }

    #[test]
    fn aggregated_rs_copy_is_queued_before_its_launch() {
        let completion = WorkItemCompletion::deferred_with_guard(None);
        let state_copy = StateCopyPlan {
            slot_ranges: vec![
                pie_driver_abi::PieStateCopyRange {
                    src_slot_id: 3,
                    dst_slot_id: 5,
                    src_token_offset: 0,
                    dst_token_offset: 0,
                    token_count: 0,
                },
                pie_driver_abi::PieStateCopyRange {
                    src_slot_id: 3,
                    dst_slot_id: 6,
                    src_token_offset: 0,
                    dst_token_offset: 0,
                    token_count: 0,
                },
            ],
        };
        let request = PendingRequest::direct(
            dummy_launch(),
            1,
            completion,
            0,
            None,
            None,
            false,
            None,
            Some(state_copy),
            None,
            false,
        );
        let mut pending = VecDeque::new();
        BatchScheduler::queue_attempt(&mut pending, request, QueueEnd::Back);

        let QueuedItem::PreLaunchCopy {
            plan: PreLaunchCopy::State(plan),
            ..
        } = pending.pop_front().unwrap()
        else {
            panic!("aggregated state copy must precede the launch");
        };
        assert_eq!(plan.slot_ranges.len(), 2);
        assert_eq!(plan.slot_ranges[0].src_slot_id, 3);
        assert_eq!(plan.slot_ranges[1].dst_slot_id, 6);
        assert!(matches!(pending.pop_front(), Some(QueuedItem::Launch(_))));
        assert!(pending.is_empty());
    }

    #[test]
    fn unresolved_multi_row_prebuilt_request_remains_solo() {
        let mut launch = dummy_launch();
        launch.qo_indptr = vec![0, 0, 0];
        let pid = ProcessId::new_v4();
        let request = PendingRequest::direct(
            launch,
            1,
            WorkItemCompletion::deferred_with_guard(None),
            0,
            Some(pid),
            Some(pid),
            true,
            None,
            None,
            None,
            false,
        );
        assert!(request.preserves_inner_rows());
        assert!(request.requires_solo_submission());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_terminal_outcome_rejects_launch_completion() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                fail_launches_after_accept: true,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;

        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        let err = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("failed launch terminal outcome must fail");
        assert!(err.to_string().contains("Failed terminal outcome"));

        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "callback")
                .count(),
            1,
            "failed accepted launches still publish exactly one callback: {log:?}"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn launches_can_overlap_before_prior_callback_when_fifo_allows() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 50,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18]).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            bound_a.program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;

        let first = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            first.clone(),
        )?;

        let second = bound_b.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            0,
            second.clone(),
        )?;

        let overlapping_launches = timeout(Duration::from_secs(5), async {
            loop {
                let launches = operation_log
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|entry| entry.as_str() == "launch")
                    .count();
                if launches >= 2 {
                    return launches;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?;
        assert_eq!(
            overlapping_launches, 2,
            "launch 2 should submit before callback 1 when overlap is allowed"
        );

        timeout(Duration::from_secs(5), first).await??;
        timeout(Duration::from_secs(5), second).await??;
        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn same_instance_launches_can_run_ahead_across_batches() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 50,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;

        let first = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            first.clone(),
        )?;

        let second = bound.reserve_completion();
        let second_for_submit = second.clone();
        let instance_id = bound.instance_id;
        let second_submit = std::thread::spawn(move || {
            crate::scheduler::submit_prebuilt_async(
                dummy_launch(),
                driver_id,
                instance_id,
                0,
                second_for_submit,
            )
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            2,
            "same-instance launch 2 should be accepted before launch 1 callback"
        );
        assert!(
            second_submit.is_finished(),
            "same-instance acceptance should not wait for launch 1 callback"
        );

        second_submit.join().unwrap()?;
        timeout(Duration::from_secs(5), first).await??;
        timeout(Duration::from_secs(5), second).await??;
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn launch_then_control_then_launch_preserves_fifo_order() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 50,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18]).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            bound_a.program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;

        let first = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            first.clone(),
        )?;

        let resize_join = tokio::spawn(async move {
            crate::scheduler::resize_pool(
                driver_id,
                7,
                32,
                vec![PiePoolRange {
                    page_index: 0,
                    page_count: 4,
                }],
                Vec::new(),
            )
            .await
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        let second = bound_b.reserve_completion();
        let second_for_submit = second.clone();
        let second_driver_id = bound_b.driver_id;
        let second_instance_id = bound_b.instance_id;
        let second_submit = std::thread::spawn(move || {
            crate::scheduler::submit_prebuilt_async(
                dummy_launch(),
                second_driver_id,
                second_instance_id,
                0,
                second_for_submit,
            )
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            1,
            "queued control should prevent later launches from bypassing the FIFO"
        );
        assert!(
            second_submit.is_finished(),
            "queue acceptance must not wait for the earlier native callback"
        );

        timeout(Duration::from_secs(5), first).await??;
        let resize = resize_join.await??;
        tokio::time::sleep(Duration::from_millis(10)).await;
        let during_control = operation_log.lock().unwrap().clone();
        assert_eq!(
            during_control
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            1,
            "second launch must wait until the control callback retires"
        );
        assert!(
            during_control.iter().any(|entry| entry == "resize_pool"),
            "resize should dispatch after launch 1 retires"
        );
        timeout(Duration::from_secs(5), resize).await??;
        timeout(Duration::from_secs(5), async {
            loop {
                if operation_log
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|entry| entry.as_str() == "launch")
                    .count()
                    >= 2
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await?;
        let log = operation_log.lock().unwrap().clone();
        let resize_idx = log
            .iter()
            .position(|entry| entry == "resize_pool")
            .expect("resize should dispatch after launch 1 retires");
        let second_launch_idx = log
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.as_str() == "launch")
            .nth(1)
            .map(|(index, _)| index)
            .expect("second launch should dispatch");
        assert!(
            resize_idx < second_launch_idx,
            "second launch must remain behind the queued control effect: {log:?}"
        );

        second_submit.join().unwrap()?;
        timeout(Duration::from_secs(5), second).await??;
        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_enqueues_before_accepted_launch_retires() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 75,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;

        let launch = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            bound.driver_id,
            bound.instance_id,
            0,
            launch.clone(),
        )?;

        let started = std::time::Instant::now();
        crate::scheduler::close_instance(&bound)?;
        assert!(
            started.elapsed() < Duration::from_millis(10),
            "fire-and-forget close must return after enqueue"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(
            !operation_log
                .lock()
                .unwrap()
                .iter()
                .any(|entry| entry == "close_instance"),
            "native close still waits for the accepted launch to retire"
        );
        timeout(Duration::from_secs(5), launch).await??;
        wait_for_operation_count(&operation_log, "close_instance", 1).await;

        let log = operation_log.lock().unwrap().clone();
        let launch_idx = log.iter().position(|entry| entry == "launch").unwrap();
        let close_idx = log
            .iter()
            .position(|entry| entry == "close_instance")
            .unwrap();
        assert!(
            launch_idx < close_idx,
            "close should happen after launch retires: {log:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stale_instance_close_is_fire_and_forget() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, scheduler, bound, endpoints) =
            setup_scheduler(operation_log.clone()).await?;
        crate::scheduler::close_instance(&bound)?;
        crate::scheduler::close_instance(&bound)?;
        drop(endpoints);
        drop(scheduler);
        assert_eq!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "close_instance")
                .count(),
            2,
            "both fire-and-forget requests are attempted; stale-close diagnostics are scheduler-owned"
        );
        Ok(())
    }

    /// A close needs only ITS OWN instance quiesced: instance B's close
    /// completes while instance A's launch is still in flight. The old
    /// behavior held every close hostage to a global pipe drain, which at
    /// cohort swaps stalled all queued launches behind a front close and
    /// turned the wave barrier's straggler demotion into a storm (V6
    /// iteration 3).
    #[tokio::test(flavor = "current_thread")]
    async fn close_of_idle_instance_overlaps_in_flight_launches() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 200,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let program_id = crate::scheduler::register_program(driver_id, dummy_program()).await?;
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18]).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;

        let launch = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            bound_a.driver_id,
            bound_a.instance_id,
            0,
            launch.clone(),
        )?;

        // Give the worker a moment to dispatch A's launch into flight.
        std::thread::sleep(Duration::from_millis(20));
        let started = std::time::Instant::now();
        crate::scheduler::close_instance(&bound_b)?;
        assert!(
            started.elapsed() < Duration::from_millis(120),
            "idle-instance close must not wait for the pipe to drain"
        );
        wait_for_operation_count(&operation_log, "close_instance", 1).await;

        let log = operation_log.lock().unwrap().clone();
        let launch_idx = log.iter().position(|entry| entry == "launch");
        let close_idx = log.iter().position(|entry| entry == "close_instance");
        assert!(
            launch_idx.is_some() && close_idx.is_some() && launch_idx < close_idx,
            "B's close must overlap A's in-flight launch: {log:?}"
        );
        timeout(Duration::from_secs(5), launch).await??;
        Ok(())
    }

    /// Synchronous controls interleave with scheduler mailbox draining.
    #[tokio::test(flavor = "current_thread")]
    async fn one_synchronous_control_dispatches_per_pass() {
        let (tx_a, mut rx_a) = tokio::sync::oneshot::channel();
        let (tx_b, mut rx_b) = tokio::sync::oneshot::channel();
        let mut pending = VecDeque::from([
            QueuedItem::RegisterProgram {
                plan: dummy_program(),
                response: tx_a,
            },
            QueuedItem::RegisterProgram {
                plan: dummy_program(),
                response: tx_b,
            },
        ]);
        let (lane, _lane_rx) = test_lane(None);
        let mut lane_inflight = 0u64;
        let mut lane_token = 0u64;
        let mut instances = HashMap::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = None;
        let mut admission_retry_at = None;
        let limits = SchedulerLimits {
            max_forward_requests: 64,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let stats = Arc::new(SchedulerStats::default());
        let mut policy = quorum::WaitAllPolicy::new(limits.max_forward_requests, None);

        let (progress, _) = BatchScheduler::dispatch_ready_items(
            &lane,
            &mut lane_inflight,
            &mut lane_token,
            &mut instances,
            &mut pending,
            &mut in_flight_launches,
            &mut in_flight_control,
            &mut admission_retry_at,
            16,
            limits,
            &stats,
            &mut policy,
            false,
        );
        assert!(progress);
        assert!(
            timeout(Duration::from_secs(5), &mut rx_a).await.is_ok(),
            "the first control dispatches this pass"
        );
        assert!(
            matches!(
                rx_b.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "the second control waits for the next pass"
        );
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn instance_queued_work_gate_sees_launches() {
        let pid = ProcessId::new_v4();
        let pending = VecDeque::from([QueuedItem::Launch(dummy_launch_request(pid, 7))]);
        assert!(BatchScheduler::instance_has_queued_work(&pending, 7));
        assert!(!BatchScheduler::instance_has_queued_work(&pending, 8));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scheduler_shutdown_drains_instances_and_destroys_once() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, scheduler, bound_a, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 40,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let program_id = crate::scheduler::register_program(driver_id, dummy_program()).await?;
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18]).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;

        let resize =
            crate::scheduler::resize_pool(driver_id, 9, 16, Vec::new(), Vec::new()).await?;
        let a = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            a,
        )?;
        let b = bound_b.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            0,
            b,
        )?;
        drop(resize);
        drop(scheduler);

        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            2
        );
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "close_instance")
                .count(),
            2
        );
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "destroy")
                .count(),
            1
        );
        let destroy_idx = log.iter().position(|entry| entry == "destroy").unwrap();
        let last_callback_idx = log
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| (entry == "callback").then_some(idx))
            .max()
            .unwrap();
        assert!(
            last_callback_idx < destroy_idx,
            "destroy must be last after callbacks: {log:?}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completion_retirement_is_event_driven() -> anyhow::Result<()> {
        // Plan §14 gate 6: the driver callback's nudge retires the batch, not
        // the backstop poll. A retirement that misses the nudge waits out the
        // 250 ms backstop and trips the bound below.
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 30,
                operation_log: Some(operation_log),
                ..DummyDriverOptions::default()
            })
            .await?;
        let backstops_before = backstop_retirements();
        let completion = bound.reserve_completion();
        let started = Instant::now();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion).await??;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(200),
            "retirement must ride the completion nudge, not the backstop poll (took {elapsed:?})"
        );
        assert_eq!(
            backstop_retirements(),
            backstops_before,
            "steady state retires with zero backstop-path wakeups (plan §16.2)"
        );
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn parked_reader_wakes_straight_from_the_driver_callback() -> anyhow::Result<()> {
        // Plan §14 gates 2/3: a task that never submitted (and drains no
        // pipeline FIFO) parks on the channel's reader wait slot and wakes
        // straight from the driver's per-channel notify, with the published
        // tail word already visible.
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, endpoints) = setup_scheduler(operation_log).await?;
        let waiter = tokio::spawn({
            let endpoint = Arc::clone(&endpoints[1]);
            async move { endpoint.wait_for_reader_change(0).await }
        });
        tokio::task::yield_now().await;

        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), waiter)
            .await??
            .expect("reader wake surfaces the new tail, not an error");
        let binding = endpoints[1].registered().binding;
        let tail = unsafe {
            (&*((binding.word_base as *const std::sync::atomic::AtomicU64)
                .add(binding.tail_word_index as usize)))
                .load(Ordering::Acquire)
        };
        assert_eq!(tail, 1, "the tail word is published before the wake");
        timeout(Duration::from_secs(5), completion).await??;
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn parked_reader_wakes_into_poisoned_not_empty() -> anyhow::Result<()> {
        // Plan §14 gate 7: a failed fire release-stores the poison word BEFORE
        // the channel notify, so a parked reader wakes into Poisoned — never
        // into a spurious Empty retry.
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                fail_launches_after_accept: true,
                operation_log: Some(operation_log),
                ..DummyDriverOptions::default()
            })
            .await?;
        let waiter = tokio::spawn({
            let endpoint = Arc::clone(&endpoints[1]);
            async move { endpoint.wait_for_reader_change(0).await }
        });
        tokio::task::yield_now().await;
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        let woke = timeout(Duration::from_secs(5), waiter).await??;
        assert!(
            matches!(
                woke,
                Err(crate::driver::channel::ChannelWaitError::Poisoned(_))
            ),
            "a parked take classifies the failed fire as Poisoned, got {woke:?}"
        );
        let _ = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("the failed fire's terminal outcome is surfaced");
        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn extern_export_flows_into_importing_instance() -> anyhow::Result<()> {
        // Plan §14 gate 3: instance A's fire fills a shared extern channel;
        // instance B's fire consumes it and publishes to its host reader —
        // cross-instance dataflow over one global channel registration.
        use pie_ptir::container::{ExternDecl, ExternDir};
        let driver_id = driver::register_driver_backend(
            DriverSpec {
                num_kv_pages: 16,
                limits: SchedulerLimits {
                    max_forward_requests: 1,
                    max_forward_tokens: 64,
                    max_page_refs: 64,
                },
                device_geometry_port_mask: 0,
            },
            DriverBackend::Dummy(crate::driver::DummyDriver::new(
                DummyDriverOptions::default(),
            )),
        );
        let _scheduler = BatchScheduler::new(
            driver_id,
            driver_id,
            16,
            SchedulerLimits {
                max_forward_requests: 1,
                max_forward_tokens: 64,
                max_page_refs: 64,
            },
            1,
        );
        let exporter_bytes = TraceContainer {
            names: vec!["shared".to_string()],
            externs: vec![ExternDecl {
                name: 0,
                dir: ExternDir::Export,
                chan: 0,
            }],
            channels: vec![chan(Shape::vector(1), DType::U32, HostRole::None, false)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::Const(Literal::U32(7)),
                    Op::Broadcast {
                        value: 0,
                        shape: Shape::vector(1),
                    },
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            }],
        }
        .encode();
        let importer_bytes = TraceContainer {
            names: vec!["shared".to_string()],
            externs: vec![ExternDecl {
                name: 0,
                dir: ExternDir::Import,
                chan: 0,
            }],
            channels: vec![
                chan(Shape::vector(1), DType::U32, HostRole::None, false),
                chan(Shape::vector(1), DType::U32, HostRole::Reader, false),
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::ChanTake(0), Op::ChanPut { chan: 1, value: 0 }],
            }],
        }
        .encode();
        let exporter_program = crate::scheduler::register_program(
            driver_id,
            ProgramRegistration {
                program_hash: pie_ptir::container_hash(&exporter_bytes),
                canonical_bytes: exporter_bytes,
                sidecar_bytes: Vec::new(),
            },
        )
        .await?;
        let importer_program = crate::scheduler::register_program(
            driver_id,
            ProgramRegistration {
                program_hash: pie_ptir::container_hash(&importer_bytes),
                canonical_bytes: importer_bytes,
                sidecar_bytes: Vec::new(),
            },
        )
        .await?;
        let shared = crate::scheduler::register_channel(
            driver_id,
            ChannelRegistrationPlan {
                driver_id,
                channel_id: 91,
                shape: vec![1],
                dtype: pie_driver_abi::PIE_CHANNEL_DTYPE_U32,
                host_role: HostRole::None as u8,
                seeded: false,
                extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_EXPORT,
                capacity: 2,
                reader_wait_id: 0,
                writer_wait_id: 0,
                extern_name: b"shared".to_vec(),
            },
        )
        .await?;
        let reader = crate::scheduler::register_channel(
            driver_id,
            ChannelRegistrationPlan {
                driver_id,
                channel_id: 92,
                shape: vec![1],
                dtype: pie_driver_abi::PIE_CHANNEL_DTYPE_U32,
                host_role: HostRole::Reader as u8,
                seeded: false,
                extern_dir: pie_driver_abi::PIE_CHANNEL_EXTERN_NONE,
                capacity: 2,
                reader_wait_id: 0,
                writer_wait_id: 0,
                extern_name: Vec::new(),
            },
        )
        .await?;
        let _ = shared;
        let exporter = crate::scheduler::bind_instance(
            driver_id,
            None,
            exporter_program,
            61,
            vec![91],
            Vec::new(),
        )
        .await?;
        let importer = crate::scheduler::bind_instance(
            driver_id,
            None,
            importer_program,
            62,
            vec![91, 92],
            Vec::new(),
        )
        .await?;

        // A parked take on the importer's reader — a task that never
        // submitted anything — observes the cross-instance flow end to end.
        let waiter = tokio::spawn({
            let endpoint = Arc::clone(&reader);
            async move { endpoint.wait_for_reader_change(0).await }
        });
        tokio::task::yield_now().await;
        let export_fire = exporter.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            exporter.instance_id,
            0,
            export_fire.clone(),
        )?;
        let import_fire = importer.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            importer.instance_id,
            0,
            import_fire.clone(),
        )?;
        timeout(Duration::from_secs(5), export_fire).await??;
        timeout(Duration::from_secs(5), import_fire).await??;
        timeout(Duration::from_secs(5), waiter)
            .await??
            .expect("the importer's publish wakes the parked reader");
        let binding = reader.registered().binding;
        let value = unsafe { std::ptr::read_unaligned(binding.mirror_base as *const u32) };
        assert_eq!(value, 7, "the exported value crossed instances");
        crate::scheduler::close_instance(&exporter)?;
        crate::scheduler::close_instance(&importer)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn timeout_bounded_shutdown_stress() -> anyhow::Result<()> {
        timeout(Duration::from_secs(5), async {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (driver_id, scheduler, bound, _endpoints) =
                setup_scheduler_with_options(DummyDriverOptions {
                    callback_delay_ms: 5,
                    operation_log: Some(operation_log),
                    ..DummyDriverOptions::default()
                })
                .await?;
            for _ in 0..16 {
                let completion = bound.reserve_completion();
                crate::scheduler::submit_prebuilt_async(
                    dummy_launch(),
                    driver_id,
                    bound.instance_id,
                    0,
                    completion,
                )?;
            }
            drop(scheduler);
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(())
    }

    /// Every quorum-hold test below needs a structural cap big enough that
    /// a single request never trivially saturates it (else the wait-all
    /// rule short-circuits straight to a fire — see `quorum::tests::
    /// structural_cap_fires_immediately_even_cold`).
    fn coalescing_limits() -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: 4,
            max_forward_tokens: 64,
            max_page_refs: 64,
        }
    }

    /// Binds a second instance on the same program/driver as `bound_a`, for
    /// tests that need two independent pipelines' fires in flight at once.
    async fn bind_second_instance(
        driver_id: usize,
        bound_a: &crate::driver::BoundInstance,
        channel_ids: [u64; 2],
        requested_instance_id: u64,
    ) -> anyhow::Result<(
        crate::driver::BoundInstance,
        Vec<Arc<crate::driver::ChannelEndpoint>>,
    )> {
        let endpoints = register_test_channels(driver_id, channel_ids).await?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            None,
            bound_a.program_id,
            requested_instance_id,
            channel_ids.to_vec(),
            vec![ChannelValue {
                channel: channel_ids[0],
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .await?;
        Ok((bound_b, endpoints))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn two_pipelines_coalesce_into_one_wave_after_cold_hold() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a, _endpoints) = setup_scheduler_with_limits(
            DummyDriverOptions {
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            },
            coalescing_limits(),
        )
        .await?;
        let (bound_b, _secondary_endpoints) =
            bind_second_instance(driver_id, &bound_a, [27, 28], 52).await?;

        let pid_a = ProcessId::new_v4();
        let pid_b = ProcessId::new_v4();

        // Submitted back-to-back, no await in between: both land in the
        // scheduler's queue before it next drains, so both `on_pipeline_
        // request` calls land in the SAME wave-gather — no timing race
        // with the 500us cold-hold window.
        let first = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            Some(pid_a),
            first.clone(),
        )?;
        let second = bound_b.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            0,
            Some(pid_b),
            second.clone(),
        )?;

        // The wait-all quorum's bootstrap cold-hold gathers both pipelines'
        // first requests into ONE dense wave (`requests=2`) instead of two
        // solo fires — the dummy driver's launch-shape trace names the
        // program count directly.
        let coalesced = timeout(Duration::from_secs(5), async {
            loop {
                let hit = operation_log
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|entry| entry.starts_with("launch-shape tokens=2 programs=2"));
                if hit {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await?;
        assert!(
            coalesced,
            "both pipelines' first requests should coalesce into one programs=2 wave: {:?}",
            operation_log.lock().unwrap()
        );

        timeout(Duration::from_secs(5), first).await??;
        timeout(Duration::from_secs(5), second).await??;
        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn token_capacity_partitions_wait_all_wave_without_deadlock() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let limits = SchedulerLimits {
            max_forward_requests: 4,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let (driver_id, _scheduler, bound_a, _endpoints) = setup_scheduler_with_limits(
            DummyDriverOptions {
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            },
            limits,
        )
        .await?;
        let (bound_b, _secondary_endpoints) =
            bind_second_instance(driver_id, &bound_a, [29, 30], 53).await?;
        let pid_a = ProcessId::new_v4();
        let pid_b = ProcessId::new_v4();

        for _ in 0..2 {
            let first = bound_a.reserve_completion();
            crate::scheduler::submit_async(
                dummy_prefill(40),
                driver_id,
                bound_a.instance_id,
                0,
                Some(pid_a),
                first.clone(),
            )?;
            let second = bound_b.reserve_completion();
            crate::scheduler::submit_async(
                dummy_prefill(40),
                driver_id,
                bound_b.instance_id,
                0,
                Some(pid_b),
                second.clone(),
            )?;

            timeout(Duration::from_secs(5), first).await??;
            timeout(Duration::from_secs(5), second).await??;
        }

        let launches = operation_log
            .lock()
            .unwrap()
            .iter()
            .filter(|entry| entry.starts_with("launch-shape tokens=40 programs=1"))
            .count();
        assert_eq!(
            launches,
            4,
            "each logical wave should split into two capacity-limited launches: {:?}",
            operation_log.lock().unwrap()
        );

        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn leave_unblocks_a_wave_holding_for_a_missing_member() -> anyhow::Result<()> {
        let (driver_id, _scheduler, bound_a, _endpoints) =
            setup_scheduler_with_limits(DummyDriverOptions::default(), coalescing_limits()).await?;
        let (bound_b, _secondary_endpoints) =
            bind_second_instance(driver_id, &bound_a, [27, 28], 54).await?;

        let pid_a = ProcessId::new_v4();
        let pid_b = ProcessId::new_v4();

        // Wave 1: both pipelines seen, both in the wait-set.
        let first_a = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            Some(pid_a),
            first_a.clone(),
        )?;
        let first_b = bound_b.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            0,
            Some(pid_b),
            first_b.clone(),
        )?;
        timeout(Duration::from_secs(5), first_a).await??;
        timeout(Duration::from_secs(5), first_b).await??;

        // Wave 2: only `a` resubmits; `b` instead leaves the fleet. The
        // quorum drops it from the wait-set and releases `a`.
        let started = Instant::now();
        let second_a = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            0,
            Some(pid_a),
            second_a.clone(),
        )?;
        notify_pipeline_leave(pid_b, LeaveKind::Terminate);
        timeout(Duration::from_secs(5), second_a).await??;
        assert!(
            started.elapsed() < Duration::from_millis(8),
            "leave should unblock the wait-all hold promptly, took {:?}",
            started.elapsed()
        );

        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scoped_leave_does_not_remove_a_sibling_pipeline_of_the_same_process()
    -> anyhow::Result<()> {
        let (driver_id, scheduler, bound_a, _endpoints) =
            setup_scheduler_with_limits(DummyDriverOptions::default(), coalescing_limits()).await?;
        let (bound_b, _secondary_endpoints) =
            bind_second_instance(driver_id, &bound_a, [31, 32], 55).await?;
        let process_id = ProcessId::new_v4();
        let pipeline_a = ProcessId::new_v4();
        let pipeline_b = ProcessId::new_v4();

        for (bound, pipeline_id) in [(&bound_a, pipeline_a), (&bound_b, pipeline_b)] {
            let completion = bound.reserve_completion();
            scheduler.handle.submit_prebuilt_tracked_with_copy(
                dummy_launch(),
                bound.instance_id,
                completion.clone(),
                0,
                process_id,
                pipeline_id,
                None,
                None,
                None,
                false,
            )?;
            if pipeline_id == pipeline_b {
                timeout(Duration::from_secs(5), completion).await??;
            }
        }

        // Wait for the first wave's other completion before starting wave 2.
        // Both scopes now belong to the same process but have independent
        // quorum membership.
        tokio::time::sleep(Duration::from_millis(10)).await;
        let sibling = bound_b.reserve_completion();
        scheduler.handle.submit_prebuilt_tracked_with_copy(
            dummy_launch(),
            bound_b.instance_id,
            sibling.clone(),
            0,
            process_id,
            pipeline_b,
            None,
            None,
            None,
            false,
        )?;
        notify_pipeline_leave(pipeline_a, LeaveKind::Close);
        timeout(Duration::from_secs(5), sibling).await??;

        let dump = scheduler.handle.debug_dump().await?;
        assert!(
            dump.contains(&pipeline_b.to_string()),
            "sibling pipeline must remain in the quorum:\n{dump}"
        );
        assert!(
            !dump.contains(&format!("pipeline {pipeline_a}")),
            "only the departed scope should be removed:\n{dump}"
        );

        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pipeline_close_drains_the_already_submitted_run_ahead_tail() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 25,
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let pid = ProcessId::new_v4();
        let mut completions = Vec::new();
        for _ in 0..3 {
            let completion = bound.reserve_completion();
            crate::scheduler::submit_async(
                dummy_launch(),
                driver_id,
                bound.instance_id,
                0,
                Some(pid),
                completion.clone(),
            )?;
            completions.push(completion);
        }

        // FIFO receipt puts this after all three launches. At least one launch
        // remains queued behind the scheduler's run-ahead depth while close
        // releases the wait-set; none may be cancelled.
        notify_pipeline_leave(pid, LeaveKind::Close);
        timeout(Duration::from_secs(5), completions.remove(0)).await??;
        timeout(Duration::from_secs(5), completions.remove(0)).await??;

        // The first two outputs remain committed after close. Consume them as
        // a post-close `take` would, releasing capacity for the queued third
        // fire; close did not poison or discard either value.
        let binding = endpoints[1].registered().binding;
        let words = binding.word_base as *const std::sync::atomic::AtomicU64;
        let tail =
            unsafe { (&*words.add(binding.tail_word_index as usize)).load(Ordering::Acquire) };
        assert_eq!(tail, 2, "settled outputs remain visible after close");
        unsafe {
            (&*words.add(binding.head_word_index as usize)).store(2, Ordering::Release);
        }
        crate::scheduler::nudge(driver_id);
        timeout(Duration::from_secs(5), completions.remove(0)).await??;

        assert!(
            operation_log
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.as_str() == "launch")
                .count()
                >= 3,
            "close must preserve queued, preparing, and dispatched fires"
        );

        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn untracked_prebuilt_fire_never_blocks_on_the_quorum() -> anyhow::Result<()> {
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_limits(DummyDriverOptions::default(), coalescing_limits()).await?;

        // `submit_prebuilt_async` always carries `pipeline_id: None` — it
        // never joins the wait-set, so it must fire promptly even though
        // nothing else is active to gather with it (bootstrap cold-hold at
        // most).
        let started = Instant::now();
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            0,
            completion.clone(),
        )?;
        timeout(Duration::from_secs(5), completion).await??;
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "an untracked prebuilt fire must never hold for the quorum, took {:?}",
            started.elapsed()
        );

        crate::scheduler::close_instance(&bound)?;
        Ok(())
    }

    fn dummy_launch_request(pipeline_id: ProcessId, instance_id: u64) -> PendingRequest {
        PendingRequest::direct(
            dummy_launch(),
            instance_id,
            WorkItemCompletion::deferred_with_guard(None),
            0,
            Some(pipeline_id),
            Some(pipeline_id),
            false,
            None,
            None,
            None,
            false,
        )
    }

    #[test]
    fn launch_grouping_uses_driver_token_capacity() {
        let limits = SchedulerLimits {
            max_forward_requests: 4,
            max_forward_tokens: 4096,
            max_page_refs: 4096,
        };
        let mut first = dummy_launch_request(ProcessId::new_v4(), 1);
        first.request = dummy_prefill(1536);
        let mut second = dummy_launch_request(ProcessId::new_v4(), 2);
        second.request = dummy_prefill(1536);

        let mut grouping = LaunchGrouping::default();
        assert!(grouping.accepts(&first, limits, 16));
        grouping.push(&first, limits, 16);
        assert!(
            grouping.accepts(&second, limits, 16),
            "the scheduler must not impose a token cap below the driver limit"
        );
    }

    #[test]
    fn launch_grouping_only_solos_device_derived_masks() {
        let limits = SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let mut host_mask = dummy_launch_request(ProcessId::new_v4(), 1);
        host_mask.request.has_user_mask = true;
        host_mask.request.masks = vec![crate::driver::command::EncodedMask::new(vec![0, 1], 1)];
        host_mask.request.mask_indptr = vec![0, 1];
        let causal = dummy_launch_request(ProcessId::new_v4(), 2);

        let mut grouping = LaunchGrouping::default();
        assert!(grouping.accepts(&host_mask, limits, 16));
        assert!(
            !grouping.push(&host_mask, limits, 16),
            "a host-derived wire mask must not close the batch"
        );
        assert!(
            grouping.accepts(&causal, limits, 16),
            "host-derived custom and causal fires should co-batch"
        );

        let mut dense = dummy_launch_request(ProcessId::new_v4(), 3);
        dense.request.has_user_mask = true;
        dense.request.device_resolved_geometry = true;
        let mut grouping = LaunchGrouping::default();
        assert!(grouping.accepts(&dense, limits, 16));
        assert!(
            grouping.push(&dense, limits, 16),
            "a device-derived dense mask remains a solo batch"
        );

        let mut host_on_device = dummy_launch_request(ProcessId::new_v4(), 4);
        host_on_device.request.has_user_mask = true;
        host_on_device.request.device_resolved_geometry = true;
        host_on_device.request.masks =
            vec![crate::driver::command::EncodedMask::new(vec![0, 1], 1)];
        host_on_device.request.mask_indptr = vec![0, 1];
        let mut grouping = LaunchGrouping::default();
        assert!(
            !grouping.push(&host_on_device, limits, 16),
            "wire rows distinguish a host-derived mask from dense device lowering"
        );
        let mut ordinary_group = LaunchGrouping::default();
        ordinary_group.push(&dummy_launch_request(ProcessId::new_v4(), 5), limits, 16);
        assert!(
            !ordinary_group.accepts(&host_on_device, limits, 16),
            "resolved-geometry host masks remain incompatible with reordered wire rows"
        );
    }

    /// Rotating held launches behind wave work must move the WHOLE contiguous
    /// launch prefix in one call. A partial rotation reorders a pipeline's
    /// run-ahead siblings; dispatch then defers the out-of-order head
    /// (`launch_has_earlier_instance_member`) and, with the earlier sibling
    /// sitting beyond a non-launch item, can never reach it — a permanent
    /// scheduler stall (the V5 benchmark deadlock, 2026-07-15).
    #[test]
    fn launch_rotation_preserves_per_instance_order() {
        let pipeline_a = ProcessId::new_v4();
        let pipeline_b = ProcessId::new_v4();
        let mut pending = VecDeque::new();
        pending.push_back(QueuedItem::Launch(dummy_launch_request(pipeline_a, 1)));
        pending.push_back(QueuedItem::Launch(dummy_launch_request(pipeline_a, 1)));
        pending.push_back(QueuedItem::Launch(dummy_launch_request(pipeline_b, 2)));
        pending.push_back(QueuedItem::CloseInstance {
            id: 9,
            pacing_wait_id: 0,
        });

        assert!(BatchScheduler::rotate_launch_for_wave_work(
            &mut pending,
            true
        ));

        assert!(
            matches!(pending.front(), Some(QueuedItem::CloseInstance { .. })),
            "the rotate-target work must reach the queue front"
        );
        let launches: Vec<(u64, u64)> = pending
            .iter()
            .filter_map(|item| match item {
                QueuedItem::Launch(request) => Some((request.instance_id, request.logical_fire_id)),
                _ => None,
            })
            .collect();
        assert_eq!(
            launches
                .iter()
                .map(|(instance, _)| *instance)
                .collect::<Vec<_>>(),
            vec![1, 1, 2],
            "rotation must not interleave the launch prefix"
        );
        assert!(
            launches[0].1 < launches[1].1,
            "same-instance run-ahead fires must stay FIFO across rotation"
        );
    }

    /// A `PreLaunchCopy` queued behind held launches is dispatchable control
    /// work: rotation must treat it as a valid target when controls are
    /// allowed (it occupies the free control slot exactly like a lifecycle
    /// control), or a held front launch starves the copy — and the copy's
    /// consumer launch — forever.
    #[test]
    fn launch_rotation_reaches_a_pre_launch_copy() {
        let make_pending = || {
            let mut pending = VecDeque::new();
            pending.push_back(QueuedItem::Launch(dummy_launch_request(
                ProcessId::new_v4(),
                1,
            )));
            pending.push_back(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::Kv(crate::driver::KvCopyPlan::default()),
                logical_completion: WorkItemCompletion::deferred_with_guard(None),
                process_id: None,
                pipeline_id: Some(ProcessId::new_v4()),
                credit_ready: false,
                quorum_generation: 0,
            });
            pending
        };

        let mut pending = make_pending();
        assert!(
            !BatchScheduler::rotate_launch_for_wave_work(&mut pending, false),
            "a settling control slot (controls disallowed) must keep launch order"
        );

        let mut pending = make_pending();
        assert!(BatchScheduler::rotate_launch_for_wave_work(
            &mut pending,
            true
        ));
        assert!(
            matches!(pending.front(), Some(QueuedItem::PreLaunchCopy { .. })),
            "the copy must reach the front so it can occupy the control slot"
        );
    }

    /// A fire behind a pre-launch copy joins the barrier immediately but
    /// holds no readiness credit until its copy retires. Cancelling it in
    /// that window must drop it WITHOUT giving a credit back: the unguarded
    /// drop used to panic debug builds ("dropped pipeline request has no
    /// readiness credit") and eat a sibling's credit in release (RV-20).
    #[test]
    fn cancelled_creditless_fire_drops_without_wave_accounting() {
        let pid = ProcessId::new_v4();
        let completion = WorkItemCompletion::deferred_with_guard(None);
        let request = PendingRequest::direct(
            dummy_launch(),
            7,
            completion.clone(),
            0,
            Some(pid),
            Some(pid),
            false,
            None,
            None,
            None,
            false,
        );
        assert!(
            !request.credit_published,
            "a fresh request holds no readiness credit"
        );
        completion.request_cancel();

        let mut policy = quorum::WaitAllPolicy::new(64, None);
        policy.on_pipeline_join(Some(pid));

        let mut pending = VecDeque::from([QueuedItem::Launch(request)]);
        let (lane, _lane_rx) = test_lane(None);
        let mut lane_inflight = 0u64;
        let mut lane_token = 0u64;
        let mut instances = HashMap::new();
        let mut in_flight_launches = VecDeque::new();
        let limits = SchedulerLimits {
            max_forward_requests: 64,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let stats = Arc::new(SchedulerStats::default());

        assert!(BatchScheduler::dispatch_launch_batch(
            &lane,
            &mut lane_inflight,
            &mut lane_token,
            &mut instances,
            &mut pending,
            &mut in_flight_launches,
            16,
            limits,
            &stats,
            &mut policy,
            0,
            1,
            0,
            1,
            0,
            0,
            None,
            None,
        ));

        assert!(pending.is_empty());
        assert!(completion.is_settled(), "the cancelled fire must reject");
        assert_eq!(
            policy.active_pipelines(),
            1,
            "the pipeline stays awaited; only a held credit may be returned"
        );
    }
}
