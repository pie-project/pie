//! Per-engine direct batch scheduler.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ::engine::{ChannelRegistration, ProgramRegistration, StateCopy};

use crate::engine::{
    BoundInstance, ChannelJoin, EngineBox, EngineId, InstanceBindingPlan,
    RegisteredChannel, SchedulerLimits, SubmissionCompletion, WorkItemAttemptOutcome,
    WorkItemCompletion,
};
use crate::scheduler::ProcessId;
use anyhow::{Result, anyhow};

use super::ControlCompletion;
use super::batch::{self, AdmissionLimits};
use super::frame::{self, FramePlan, FramePolicy, FrameStamp};
use super::stats::{self, SchedulerStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaveKind {
    Terminate,
    /// The planner is evicting the process: its lanes stop being awaited
    /// (boundaries seal without it) while already-submitted frames stay
    /// sealable — the tail drains untracked and its fire leases release,
    /// which the eviction's quiescence wait depends on. No purge.
    Suspend,
    /// A pipeline closed or dropped. Its wait-set row is released immediately,
    /// while every request already accepted by the scheduler continues
    /// untracked to settlement. Later guest submissions are rejected by the
    /// pipeline resource before they reach the scheduler.
    Close,
}

/// Post one pipeline-leave to EVERY registered engine's scheduler thread (a
/// pipeline's requests may have landed on any of them) so each thread's
/// local [`FramePolicy`] drops the leaver from its wait-set. Fire-and-forget:
/// a shutting-down/closed scheduler channel is silently skipped.
///
/// The `id` KEY SPACE DEPENDS ON `kind` — Close is lane-keyed (pipeline
/// scope id), Suspend/Terminate are process-keyed. Callers use the typed
/// wrappers below so the key space is fixed by the function name instead of
/// per-call-site convention (the §15.2 seal-wedge bug class).
fn post_pipeline_leave(id: ProcessId, owner: Option<ProcessId>, kind: LeaveKind) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::PipelineLeave(id, owner, kind, None));
    }
}

/// One LANE (pipeline scope) leaves the wait-all quorum gracefully; its
/// accepted fires drain to settlement untracked. `owner` is the owning
/// process when known — the KV allocation-wait park posts it, because the
/// process may block before its first fire, and without the owner the
/// policy cannot drop it from the process-keyed `staged` /
/// `joins_in_flight` (the frame seal would wait forever for a join that
/// cannot come).
pub(crate) fn notify_lane_close(scope: ProcessId, owner: Option<ProcessId>) {
    post_pipeline_leave(scope, owner, LeaveKind::Close);
}

/// EVERY lane `pid` owns leaves the wait-all quorum (planner suspend); the
/// submitted tail stays sealable and drains untracked. Process-keyed.
pub(crate) fn notify_process_suspend(pid: ProcessId) {
    post_pipeline_leave(pid, Some(pid), LeaveKind::Suspend);
}

/// `pid` is runnable again after a suspend (restore committed, or the
/// eviction rolled back). Undoes the wait-set consequences of
/// [`notify_process_suspend`]. Process-keyed, fire-and-forget: a missed
/// resume is fail-safe (the fleet just stops waiting for the process).
pub(crate) fn notify_process_resume(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ProcessResume(pid));
    }
}

/// Terminate `pid`'s lanes, fire-and-forget (queued fires are rejected).
/// Process-keyed. The waited sibling is [`notify_process_terminate`].
pub(crate) fn post_process_terminate(pid: ProcessId) {
    post_pipeline_leave(pid, None, LeaveKind::Terminate);
}

/// Post `pid`'s Terminate leave to every engine and hand back the fences that
/// resolve once each scheduler has PROCESSED it — the posting half of
/// [`notify_process_terminate`], split from its await.
///
/// The split is what lets a retiring process release its execution seat
/// SYNCHRONOUSLY, in `ProcessCtx::drop`, instead of carrying the permit into
/// the spawned teardown task: the leave is already queued ahead of the
/// release broadcast on the same producer, so every engine still observes
/// leave-then-release, while the deferred teardown keeps the awaited fence it
/// needs before recycling pooled resources. Measured at conc 512, holding the
/// permit until the teardown task ran cost 27.8 ms p50 per retiree — 16.7 ms
/// of it purely waiting for the spawn to be scheduled behind 511 siblings.
pub(crate) fn post_process_terminate_fenced(pid: ProcessId) -> Vec<TerminateFence> {
    let handles = super::handle_registry().read().unwrap();
    handles
        .iter()
        .flatten()
        .filter_map(|handle| {
            let (response, received) = tokio::sync::oneshot::channel();
            handle
                .send(SchedulerItem::PipelineLeave(
                    pid,
                    None,
                    LeaveKind::Terminate,
                    Some(response),
                ))
                .ok()
                .map(|_| received)
        })
        .collect()
}

/// One engine's acknowledgement that it has processed a posted Terminate
/// leave (see [`post_process_terminate_fenced`]).
pub(crate) type TerminateFence = tokio::sync::oneshot::Receiver<()>;

/// Await fences from [`post_process_terminate_fenced`]. Equivalent to the
/// tail of [`notify_process_terminate`]: once this resolves, every engine's
/// scheduler has purged the pid's queued work and cancelled its protected
/// in-flight control, so pooled resources can be recycled.
pub(crate) async fn await_terminate_fences(fences: Vec<TerminateFence>) {
    for fence in fences {
        let _ = fence.await;
    }
}

async fn notify_pipeline_leave_and_wait(pid: ProcessId, kind: LeaveKind) {
    let responses = {
        let handles = super::handle_registry().read().unwrap();
        handles
            .iter()
            .flatten()
            .filter_map(|handle| {
                let (response, received) = tokio::sync::oneshot::channel();
                handle
                    .send(SchedulerItem::PipelineLeave(
                        pid,
                        None,
                        kind,
                        Some(response),
                    ))
                    .ok()
                    .map(|_| received)
            })
            .collect::<Vec<_>>()
    };
    for response in responses {
        let _ = response.await;
    }
}

pub(crate) async fn notify_pipeline_close(pid: ProcessId) {
    notify_pipeline_leave_and_wait(pid, LeaveKind::Close).await;
}

/// `forward.park()`: the lane is leaving the frame wait-set until it fires
/// again. Broadcast to every engine's scheduler and fire-and-forget — a
/// policy that has never seen the lane fire ignores it, and the exit is
/// ordered by `seq` against that lane's own submits rather than against this
/// call. Deliberately NOT routed through the control path: park exists to
/// release a gather, and the control slot is depth 1, so a park queued behind
/// the dispatch the gather is holding could never arrive.
pub(crate) fn notify_lane_park(pid: ProcessId, seq: u64) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::LanePark { lane: pid, seq });
    }
}

/// A retiring process released its capped execution permit (capped
/// deployments only). Broadcast to every engine's scheduler (mirrors
/// [`notify_pipeline_leave`]): a policy with no staged successor ignores it,
/// the policy holding the successor's staged bind earmarks the join.
/// Carries the retiree's identity so the policy resolves exactly that
/// holder's departure. Fire-and-forget: the caller posted the holder's
/// Terminate leave first, on this same producer, so every engine sees
/// leave-then-release.
pub(crate) fn notify_execution_slot_released(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ExecutionSlotReleased(pid));
    }
}

/// The deferred teardown finished: every event the process can ever
/// produce is already in each engine's mailbox ahead of this one (the
/// teardown task is the process's last producer and sends this after its
/// final drop). The worker retires the pid's terminate tombstone on
/// receipt, bounding `terminated_processes` by live-plus-draining
/// processes instead of every process that ever ran.
pub(crate) fn notify_process_quiesced(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ProcessQuiesced(pid));
    }
}

/// A parked process acquired its execution permit: its first fire is
/// imminent, so it is a named join in flight and the cohort-boundary window
/// stays open until it lands. Sent BEFORE the
/// process's first fire enters the mailbox (same producer), so the policy
/// sees consume-then-fire; a reordered arrival is harmless (the policy's
/// staged guard skips a lane that already fired).
pub(crate) fn notify_execution_slot_consumed(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ExecutionSlotConsumed(pid));
    }
}

/// A process joined the execution-admission FIFO. Announced before the
/// permit wait so the frame policy can earmark it by identity.
pub(crate) fn notify_admission_queued(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::AdmissionQueued(pid));
    }
}

/// A process left the FIFO -- it took its permit, or it was cancelled.
pub(crate) fn notify_admission_dequeued(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::AdmissionDequeued(pid));
    }
}

/// No-op: wait-set rejoin is implicit on the pipeline's next scheduler
/// submission, so a join event has nothing to do here (the planner's
/// restore path relies on the same implicit rejoin).
#[allow(dead_code)] // no live caller — see doc.
pub(crate) fn notify_pipeline_join(_pid: ProcessId) {}

/// Wake-class counter (plan §16.2): completions that the 250 ms hang backstop
/// discovered already settled — a lost nudge. Steady state stays at zero; any
/// increment is a wake-path regression worth a warning.
pub(crate) static BACKSTOP_RETIREMENTS: AtomicU64 = AtomicU64::new(0);
static NEXT_LOGICAL_FIRE_ID: AtomicU64 = AtomicU64::new(1);

// `backstop_retirements()` -- a `#[cfg(test)]` reader for the counter above
// -- was deleted rather than allowed: nothing called it, and a test-only
// accessor nobody accesses is an observability claim with no observer. The
// counter itself stays; it is written on the live path and read through the
// stats surface. `BACKSTOP_RETIREMENTS.load(Ordering::Relaxed)` is the whole
// body if a test ever wants it back.

pub(crate) struct PendingRequest {
    pub(crate) logical_fire_id: u64,
    pub(crate) request: crate::engine::FireRequest,
    pub(crate) instance_id: u64,
    pub(crate) completion: WorkItemCompletion,
    /// The owning process. Process-wide suspend/terminate acts on every
    /// request with this identity.
    pub(crate) process_id: Option<ProcessId>,
    /// The submitting pipeline resource's stable scope identity, or `None`
    /// for an untracked/prebuilt fire. This is the wait-set key: the frame
    /// lane (and at k = 1 the synthesized single-slot stamp's lane).
    pub(crate) pipeline_id: Option<ProcessId>,
    pub(crate) prelaunch_copy: Option<::engine::KvCopy>,
    pub(crate) prelaunch_state_copy: Option<StateCopy>,
    /// Vesuvius frame identity: which lane/frame/slot this fire belongs to.
    /// At k = 1 the worker synthesizes a single-slot stamp at admission for
    /// every tracked fire (`lane` = `pipeline_id`, `seq` = the fire id).
    /// `None` = an untracked/prebuilt rider — dispatched outside the
    /// sealed-wave order, never awaited.
    pub(crate) frame: Option<FrameStamp>,
    /// tart (0.3 re-port step 1): whether this fire's program carries
    /// attention-stage hooks (OnAttnProj/OnAttn). Stamped at the pipeline
    /// submit from the bound container; fire planning keeps the hook rows
    /// in the full-depth prefix under the Act-2 order.
    pub(crate) hook_program: bool,
    /// The pass-wide adapter sink (the region table's LORA bit reads it).
    pub(crate) lora_program: bool,
}

impl PendingRequest {
    #[allow(
        clippy::too_many_arguments,
        reason = "a launch's whole descriptor, and this IS the struct constructor — \
                  every argument is a field of the `PendingRequest` being built, so \
                  \"factor it into a struct\" would produce a second struct with the \
                  same twelve fields"
    )]
    fn direct(
        request: crate::engine::FireRequest,
        instance_id: u64,
        completion: WorkItemCompletion,
        process_id: Option<ProcessId>,
        pipeline_id: Option<ProcessId>,
        prelaunch_copy: Option<::engine::KvCopy>,
        prelaunch_state_copy: Option<StateCopy>,
        frame: Option<FrameStamp>,
        hook_program: bool,
        lora_program: bool,
    ) -> Self {
        let logical_fire_id = NEXT_LOGICAL_FIRE_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            logical_fire_id,
            request,
            instance_id,
            completion,
            process_id,
            pipeline_id,
            prelaunch_copy,
            prelaunch_state_copy,
            frame,
            hook_program,
            lora_program,
        }
    }

    pub(crate) fn wire_row_count(&self) -> usize {
        self.request.lanes.len()
    }
}

/// `PIE_WAVE_TRACE` — wave/enqueue observability, resolved once (this sits
/// on the per-fire enqueue path).
pub(crate) fn wave_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_WAVE_TRACE").is_some())
}

/// Whether this request lowered its mask to rows the host can see.
///
/// Was `!request.masks.is_empty()` on a flat mask vector; a mask lives on its
/// LANE now (`Lane::mask`), so the question is whether any lane carries one.
fn has_wire_masks(request: &crate::engine::FireRequest) -> bool {
    request.lanes.iter().any(|lane| lane.mask.is_some())
}

#[allow(
    clippy::large_enum_variant,
    reason = "measured: the enum is 1408 bytes and `Launch { pending: PendingRequest }` \
              IS those 1408 (next largest is `RegisterProgram` at 232). But `Launch` \
              is the hot variant — one per forward step — and every other variant is \
              a rare control. Boxing it would put an allocation on the launch path to \
              shrink messages that are sent orders of magnitude less often, which is \
              backwards. The cold-variant case is handled the other way: see \
              `LaneRequest::Control`, which IS boxed"
)]
enum SchedulerItem {
    Launch {
        pending: PendingRequest,
    },
    RegisterProgram {
        plan: ProgramRegistration,
        response: tokio::sync::oneshot::Sender<Result<u64>>,
    },
    RegisterChannel {
        plan: ChannelRegistration,
        response: tokio::sync::oneshot::Sender<Result<RegisteredChannel>>,
    },
    RegisterChannels {
        plans: Vec<ChannelRegistration>,
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
        plans: Vec<ChannelRegistration>,
        /// Some on the program cache's first sight (the engine requires the
        /// instance's channels registered BEFORE the program — status -5
        /// otherwise — so registration must ride between channels and bind
        /// inside the one dispatch); None when the hash is already
        /// registered, with `bind.program_id()` carrying the cached id.
        program: Option<ProgramRegistration>,
        bind: InstanceBindingPlan,
        response:
            tokio::sync::oneshot::Sender<Result<(Vec<RegisteredChannel>, u64, BoundInstance)>>,
    },
    CopyKv {
        plan: ::engine::KvCopy,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CopyKvTracked {
        plan: ::engine::KvCopy,
        completion: ControlCompletion,
    },
    // Only reached via `SchedulerHandle::copy_state`, which the mock-engine
    // fire path doesn't call yet — see `scheduler::dispatch`'s module doc for
    // the full engine-ABI-completeness rationale. `ResizePool` stood beside
    // it and is gone with the verb (alto design §8, wave C).
    #[allow(dead_code)]
    CopyState {
        plan: StateCopy,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CloseInstance {
        id: u64,
        pacing_wait_id: u64,
    },
    CloseChannel {
        id: u64,
    },
    /// A whole cohort of channel closes in one mailbox item — posted by
    /// process teardown, which owns every id it retires. One item per
    /// departing process instead of one per channel keeps a teardown herd
    /// from inflating the epoch a worker pass has to drain.
    CloseChannels {
        ids: Vec<u64>,
    },
    /// Event-driven retirement wake: sent by [`NudgeWaker`] when an in-flight
    /// engine submission completion publishes. Carries no work; it only unblocks the
    /// scheduler's wait so the retire pass runs immediately.
    Nudge,
    /// A pipeline left the fleet ([`notify_pipeline_leave`]'s broadcast).
    /// Handled immediately on dequeue (like [`SchedulerItem::Nudge`]): it
    /// only mutates the run-loop's local [`FramePolicy`] (and, for
    /// Terminate, rejects the pid's queued work), so it can't reorder
    /// control ops or launches.
    /// `.0` is the leaving lane's PIPELINE SCOPE id; `.1` names the owning
    /// PROCESS when the caller knows it. The two are different key spaces
    /// (`FramePolicy::lanes` vs `staged`/`joins_in_flight`/`pending_binds`),
    /// and a leaver that has not fired yet has no lane to recover the owner
    /// from — so the owner must travel with the notification.
    PipelineLeave(
        ProcessId,
        Option<ProcessId>,
        LeaveKind,
        Option<tokio::sync::oneshot::Sender<()>>,
    ),
    /// A capped execution slot was released: the named retiree's deferred
    /// teardown dropped its execution permit ([`notify_execution_slot_released`]'s
    /// broadcast). While the freed slot has a staged taker the frame seal
    /// waits, so a cohort turnover gathers the incoming herd instead of
    /// sealing narrow epochs. Uncapped deployments never send this.
    ExecutionSlotReleased(ProcessId),
    /// The named process's deferred teardown finished; no event from it can
    /// follow. Retires its terminate tombstone.
    ProcessQuiesced(ProcessId),
    /// A parked process acquired its execution permit
    /// ([`notify_execution_slot_consumed`]'s broadcast): the frame seal
    /// waits for this exact process's first fire (identity-paired with the
    /// release above — the two race through the mailbox in either order).
    ExecutionSlotConsumed(ProcessId),
    /// A process is queued for an execution permit; it is the identified
    /// taker of the next slot to free (the semaphore is FIFO-fair).
    AdmissionQueued(ProcessId),
    /// It took the permit, or went away before it could.
    AdmissionDequeued(ProcessId),
    /// The planner concluded a suspended process is runnable again (restore
    /// committed, or the eviction rolled back): its lanes may rejoin the
    /// wait-set and batch full frames again. Process-keyed.
    ProcessResume(ProcessId),
    /// A frame submit failed mid-way host-side: only `submitted` of the
    /// declared fires exist. The frame policy adjusts the lane frame's
    /// expected count so it can still seal (frame mode only; a no-op
    /// otherwise).
    FrameTruncate {
        lane: ProcessId,
        seq: u64,
        submitted: u32,
    },
    /// `forward.park()`: the guest is leaving the seal's wait-set until it
    /// fires again. Ordered against that lane's submits by `seq` — the exit
    /// lands once every frame submitted before it has sealed, so a guest may
    /// park with fires still outstanding (frame mode only; a no-op
    /// otherwise).
    LanePark {
        lane: ProcessId,
        seq: u64,
    },
    /// Snapshot the run loop's state as a human-readable dump (queue
    /// composition, in-flight work, barrier membership). Answered inline on
    /// dequeue — a held wave must be inspectable from outside the thread.
    DebugDump {
        response: tokio::sync::oneshot::Sender<String>,
    },
    /// An engine-lane reply (launch accepted/rejected, control commit).
    /// Handled immediately on dequeue, like `Nudge` — it mutates only
    /// in-flight bookkeeping, never queue order.
    Lane(LaneReply),
    Stop,
}

/// Wakes the scheduler thread through its own queue when a registered engine
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
    let table = waker::WakerTable::global();
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
    Kv(::engine::KvCopy),
    State(StateCopy),
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
// Engine lane (V6 iteration 48)
// =============================================================================
//
// A dedicated thread owns the `EngineBox` and executes EVERY engine call
// in FIFO post order, so the engine keeps the exact single-threaded
// serialization it has always had — no engine-side concurrency is introduced.
// What changes is who blocks: launch submit (1.2–2.5 ms) and lifecycle
// controls (0.16 ms p50 with 2–10 ms allocator tails) leave the scheduler
// worker's critical path, which must otherwise fit inside the run-ahead
// pipelining window (the measured 283–437 ms/run of
// sched-lag-after-wait-all — the fast/slow boot-mode spread — and the
// 66–88 ms/run of control-occupancy wave gaps).
//
// Division of state:
// - The lane owns the engine and the `channels` registry set (only control
//   execution ever touched it).
// - The worker keeps ALL policy and admission state: the frame policy, `pending`,
//   `instances` (read by launch admission and gather on every pass). Control
//   arms that used to mutate `instances` inline are split: the engine half
//   runs on the lane, and the map mutation + response happen back on the
//   worker when the lane's reply arrives (`apply_lane_reply`) — keeping the
//   invariant that a bind's response is sent only AFTER the instance is
//   admissible, on the same thread that admits launches.
// - Replies ride the scheduler's own channel (`SchedulerItem::Lane`), so a
//   reply wakes the worker exactly like any other event.

/// A [`FrameSubmission`] in transit to the engine lane.
///
/// SAFETY: the submission is `!Send` only through its
/// `Vec<*mut TerminalCell>` — raw pointers into the engine's pinned
/// terminal-cell slots, which are process-stable allocations with no thread
/// affinity (the engine itself reads them from its own threads today). The
/// submission is built complete on the worker, moved to the lane, and
/// consumed exactly once by `engine.launch` — the same single-consumer
/// discipline as the worker-inline call this replaces, with the backing
/// requests kept alive in `in_flight_launches` until the frame retires
/// (retire happens strictly after the lane's reply).
struct LaneLaunch(crate::engine::FrameFire);
unsafe impl Send for LaneLaunch {}

/// Worker → lane requests, executed strictly in FIFO order.
/// Charges one lane request's wall time to `lane_launch_us` or
/// `lane_control_us` on drop, so an early `return`/`continue` inside the
/// arm still accounts. Diagnostic only.
struct LaneCharge<'a> {
    stats: &'a SchedulerStats,
    began: Instant,
    control: bool,
    charge: bool,
    prefill: bool,
}

impl Drop for LaneCharge<'_> {
    fn drop(&mut self) {
        if !self.charge {
            return;
        }
        use std::sync::atomic::Ordering::Relaxed;
        let us = self.began.elapsed().as_micros() as u64;
        let q = &self.stats.fire.quorum;
        if self.control {
            q.lane_control_us.fetch_add(us, Relaxed);
            q.lane_control_n.fetch_add(1, Relaxed);
            q.lane_control_max_us.fetch_max(us, Relaxed);
        } else if self.prefill {
            q.lane_prefill_us.fetch_add(us, Relaxed);
            q.lane_prefill_n.fetch_add(1, Relaxed);
        } else {
            q.lane_launch_us.fetch_add(us, Relaxed);
            q.lane_launch_n.fetch_add(1, Relaxed);
        }
    }
}

enum LaneRequest {
    Launch {
        token: u64,
        submission: LaneLaunch,
        /// Does this wave carry a prefill? (`tokens > rows`, i.e. some lane
        /// contributed more than one token.) Diagnostic only: it splits the
        /// lane's launch time so the cost of enqueuing a prefill-carrying
        /// wave can be read against a pure-decode one of the same width.
        prefill: bool,
    },
    /// A control `QueuedItem` (never `Launch`): the lane runs the engine
    /// half of the old `dispatch_ordered_item` arm.
    ///
    /// Boxed deliberately. Measured: `QueuedItem` is 376 bytes and the whole
    /// `LaneRequest` was 384 because of it, while the hot `Launch` variant
    /// needs only ~120. Controls are the COLD traffic on this lane (binds,
    /// registers, closes, copies, pool resizes) and launches are the per-
    /// forward-step traffic, so paying one allocation per control to keep
    /// every queued launch a third of the size is the right way round.
    Control { token: u64, item: Box<QueuedItem> },
    /// Drain marker: the lane replies with the engine and its channel set so
    /// the worker can run shutdown teardown with everything already quiesced.
    Shutdown {
        response: crossbeam::channel::Sender<(Option<EngineBox>, ChannelJoin)>,
    },
}

/// Lane → worker replies (via `SchedulerItem::Lane`).
enum LaneReply {
    LaunchDone {
        token: u64,
        result: std::result::Result<SubmissionCompletion, String>,
    },
    ControlDone {
        token: u64,
        commit: LaneCommit,
    },
}

/// The worker-side half of a control that the lane finished executing.
enum LaneCommit {
    /// Nothing to commit — the lane already sent the response (pure engine
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
    /// A successful engine-side instance close: remove + close wait slots.
    CloseInstance { id: u64 },
    /// An async-completing control (copies / pool resizes): install the
    /// engine's completion into the pending control slot, or clear the slot
    /// on a synchronous engine rejection.
    AsyncControl {
        result: std::result::Result<SubmissionCompletion, String>,
    },
}

/// What a lane request still owes its caller: one reply, on one token.
///
/// Read off the request BEFORE it is served, because the reason it exists is
/// the case where serving it does not return -- a panic on the lane thread.
/// The scheduler's launch and control slots are keyed by token and waited on
/// with no timeout, so a token that is never answered is a request that never
/// ends.
enum Owed {
    Launch(u64),
    Control(u64),
    /// Shutdown answers on its own channel, and a lane that panicked answers it
    /// in [`EngineLoop::drain_poisoned`] instead.
    Nothing,
}

impl Owed {
    fn of(request: &LaneRequest) -> Owed {
        match request {
            LaneRequest::Launch { token, .. } => Owed::Launch(*token),
            LaneRequest::Control { token, .. } => Owed::Control(*token),
            LaneRequest::Shutdown { .. } => Owed::Nothing,
        }
    }

    fn answer(self, reply_tx: &crossbeam::channel::Sender<SchedulerItem>, why: &str) {
        let reply = match self {
            Owed::Launch(token) => LaneReply::LaunchDone {
                token,
                result: Err(why.to_string()),
            },
            Owed::Control(token) => LaneReply::ControlDone {
                token,
                commit: LaneCommit::AsyncControl {
                    result: Err(why.to_string()),
                },
            },
            Owed::Nothing => return,
        };
        let _ = reply_tx.send(SchedulerItem::Lane(reply));
    }
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

struct EngineLoop {
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
}

/// Whose turn the engine lane is serving — the starvation bound on
/// [`EngineLoop::next_request`]'s launch-first preference.
///
/// Launches are preferred because they are the device's work and a control
/// is not; the counters only stop that preference from becoming an absolute
/// priority, which is what it silently became once alto's run-ahead made the
/// launch queue permanently non-empty (see `next_request`).
#[derive(Default)]
struct LaneTurn {
    /// Launches served since the last control turn ended.
    launch_run: u32,
    /// Controls served in the turn now running.
    control_run: u32,
}

impl LaneTurn {
    /// Launches the lane serves before it offers the controls a turn. Two
    /// rather than one: a control turn that finds nothing costs a `try_recv`,
    /// and at the run-ahead depth the lane is meant to be serving launches
    /// back to back. Two also bounds a control's wait at roughly one frame,
    /// which is the granularity anything downstream can act on anyway.
    const LAUNCH_RUN_BEFORE_CONTROL: u32 = 2;

    /// Controls one turn may serve before the launches get the lane back.
    /// A cohort turnover posts its whole successor generation's binds at
    /// once, and draining them one per launch run would spread a
    /// sub-millisecond burst of engine work across the whole generation —
    /// which is the bring-up starvation this bound exists to end. Capped so
    /// a pathological control flood still cannot hold the device off.
    const CONTROL_RUN_MAX: u32 = 32;

    /// Is a control turn owed, and does it have budget left?
    const fn control_due(&self) -> bool {
        self.launch_run >= Self::LAUNCH_RUN_BEFORE_CONTROL
            && self.control_run < Self::CONTROL_RUN_MAX
    }

    fn took_launch(&mut self) {
        self.launch_run = self.launch_run.saturating_add(1);
    }

    fn took_control(&mut self) {
        self.control_run = self.control_run.saturating_add(1);
        if self.control_run >= Self::CONTROL_RUN_MAX {
            self.end_control_turn();
        }
    }

    /// The turn is over — the queue ran dry or the budget did. Launches own
    /// the lane again until they have had another run.
    fn end_control_turn(&mut self) {
        self.launch_run = 0;
        self.control_run = 0;
    }
}

impl EngineLoop {
    fn spawn(
        engine_idx: usize,
        engine: Option<EngineBox>,
        reply_tx: crossbeam::channel::Sender<SchedulerItem>,
        stats: Arc<SchedulerStats>,
    ) -> Self {
        let (launch_tx, launch_rx) = crossbeam::channel::unbounded::<LaneRequest>();
        let (control_tx, control_rx) = crossbeam::channel::unbounded::<LaneRequest>();
        let thread = std::thread::Builder::new()
            .name(format!("pie-engine-{engine_idx}"))
            .spawn(move || Self::run(engine_idx, engine, launch_rx, control_rx, reply_tx, stats))
            .expect("spawn pie-engine lane thread");
        Self {
            launch_tx,
            control_tx,
            thread: Some(thread),
        }
    }

    fn post(&self, request: LaneRequest) {
        // The lane outlives every poster (shutdown joins it last); a send
        // failure means the lane thread panicked, which the join reports.
        let _ = match &request {
            LaneRequest::Launch { .. } => self.launch_tx.send(request),
            LaneRequest::Control { .. } | LaneRequest::Shutdown { .. } => {
                self.control_tx.send(request)
            }
        };
    }

    /// Drain both queues and take the engine + channel set back for
    /// teardown. The worker only calls this with `lane_inflight == 0`, so
    /// both queues are empty and the Shutdown marker is the sole item.
    fn shutdown(&mut self) -> (Option<EngineBox>, ChannelJoin) {
        let (response_tx, response_rx) = crossbeam::channel::bounded(1);
        let _ = self.control_tx.send(LaneRequest::Shutdown {
            response: response_tx,
        });
        let state = response_rx
            .recv()
            .unwrap_or_else(|_| (None, ChannelJoin::new()));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        state
    }

    /// Receive the next request, launches first — but with a BOUND on how
    /// long a control may be starved by them.
    ///
    /// The unbounded form of this ("try `launch_rx`; only if it is empty,
    /// try `control_rx`") is what the per-wave quorum could afford. It read
    /// `control_rx` exactly when the launch queue ran dry, and under the
    /// wait-all-per-wave rule it ran dry every wave: the gather for wave
    /// n+1 could not seal until every lane had resubmitted, so the lane
    /// thread saw a real gap at each boundary and the queued controls drained
    /// into it. The doc comment said so — "between waves (a cadence of idle
    /// gaps) queued controls drain".
    ///
    /// **alto CLOSED THAT GAP ON PURPOSE.** Frames seal EARLY and overlap
    /// on-stream (`scheduler::frame`: the next frame seals while the current
    /// one executes), the run-ahead posts to `frame::configured_dispatch_depth`
    /// frames deep, and `submit` answers with the device still running (F2b),
    /// so a frame retires from its completion callback while the lane is
    /// already serving the next one. Saturation is the whole point of that
    /// work and it succeeds: measured on the c=16 throughput cell, `launch_rx`
    /// held exactly one launch at EVERY poll for the whole run. Launch-first
    /// with no bound then never reaches `control_rx` at all.
    ///
    /// What starved was BRING-UP. A guest's first fire cannot exist until its
    /// bind control has committed on this lane, so a starved control queue is
    /// a fleet that cannot seat its lanes: measured at c=16 (16 live
    /// processes), 13-14 binds stood queued on `control_rx` while the lane
    /// served an unbroken launch train, only 1-4 pipelines ever reached the
    /// frame policy's wait-set, and the wait-all gate — correctly — sealed
    /// 2-to-3-wide frames because two or three lanes were all there were.
    /// The engine's own histogram: `batch size hist [0,548,750,0,0,0,0,0]`
    /// against `max_lanes` 256.
    ///
    /// So the preference stays and the starvation does not. After
    /// [`LaneTurn::LAUNCH_RUN_BEFORE_CONTROL`] consecutive launches the lane
    /// takes ONE control turn: it drains `control_rx` until the queue is
    /// empty or [`LaneTurn::CONTROL_RUN_MAX`] controls have gone, then hands
    /// the lane back to the launches. Both bounds are cheap in the units that
    /// matter — a bind is ~59 us of engine time against a ~8 ms launch, so a
    /// full turn costs well under a percent of one frame, and the turn
    /// resolves to a single failed `try_recv` whenever nothing is queued.
    ///
    /// Ordering is unaffected: a queued launch and a queued control are
    /// mutually independent by construction (see [`EngineLoop`] — a fire
    /// exists only after its own bind committed, a close only posts once its
    /// instance is quiesced, and `pipe_concurrent_control` is what decides
    /// which controls the worker may post while launches are in flight), and
    /// independence is symmetric. Serving a control ahead of a queued launch
    /// reorders no dependent pair, exactly as the single-FIFO predecessor
    /// this queue split replaced did not.
    ///
    /// Blocks on both queues when idle.
    fn next_request(
        launch_rx: &crossbeam::channel::Receiver<LaneRequest>,
        control_rx: &crossbeam::channel::Receiver<LaneRequest>,
        turn: &mut LaneTurn,
    ) -> std::result::Result<LaneRequest, ()> {
        use crossbeam::channel::TryRecvError;
        // Stay hot briefly after going empty before parking.
        // The lane hop sits on the enqueue-ahead path: a parked lane pays a
        // thread wake (µs–ms under the box's wake-burst contention) on
        // every submit, which measurably broke run-ahead pipelining
        // (81 % → ~30 % of transitions enqueued ahead on the parked
        // variant). At wave cadence the spin window always covers the next
        // post, so the submit hop costs a cache-hot poll instead; the lane
        // parks only after true idleness (between requests / at shutdown).
        // This is a lane wake optimization, independent of the fire policy.
        const ENGINE_LANE_HOT_US: u64 = 1_000_000;
        let hot_window = Duration::from_micros(ENGINE_LANE_HOT_US);
        let mut spin_until = Instant::now() + hot_window;
        loop {
            // The control turn: launches have had their run, so the queued
            // controls get theirs before the next one. `Empty` ends the turn
            // rather than merely skipping it, so a lane with nothing queued
            // pays one failed `try_recv` per run and never re-enters.
            if turn.control_due() {
                match control_rx.try_recv() {
                    Ok(request) => {
                        turn.took_control();
                        return Ok(request);
                    }
                    Err(TryRecvError::Disconnected) => {
                        return launch_rx.try_recv().map_err(|_| ());
                    }
                    Err(TryRecvError::Empty) => turn.end_control_turn(),
                }
            }
            match launch_rx.try_recv() {
                Ok(request) => {
                    turn.took_launch();
                    return Ok(request);
                }
                // Both senders live in the worker's `EngineLoop` handle and
                // drop together (the graceful path is the Shutdown marker):
                // drain what remains on the other queue, then stop.
                Err(TryRecvError::Disconnected) => {
                    return control_rx.try_recv().map_err(|_| ());
                }
                Err(TryRecvError::Empty) => {}
            }
            match control_rx.try_recv() {
                Ok(request) => {
                    turn.took_control();
                    return Ok(request);
                }
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
            spin_until = Instant::now() + hot_window;
        }
    }

    fn run(
        engine_idx: usize,
        mut engine: Option<EngineBox>,
        launch_rx: crossbeam::channel::Receiver<LaneRequest>,
        control_rx: crossbeam::channel::Receiver<LaneRequest>,
        reply_tx: crossbeam::channel::Sender<SchedulerItem>,
        stats: Arc<SchedulerStats>,
    ) {
        let mut channels = ChannelJoin::new();
        // **THE BROKER, FINALLY EARNING ITS KEEP** (survey debt 7, design §9).
        //
        // 901 lines of run-ahead bookkeeping — a waker table, recycling
        // terminal cells, per-work-item leases — sat idle for the whole of F1
        // because every shell settled inside `submit` and the lane could write
        // the outcomes itself. F2b is the wave it was built for: the CUDA
        // shell answers `submit` with the device still running, and what
        // resolves a frame is its own completion callback, arriving on the
        // driver's host-function thread and speaking through these two.
        let broker = crate::engine::CompletionBroker::new();
        let settlements = crate::engine::completion::FrameSettlements::new();
        // **THIS THREAD IS THE ONE THAT DRIVES THE DEVICE.** The engine was
        // opened and LOADED on the worker's boot thread and moved here; a
        // shell with per-thread device state (`engine-cuda`'s `cudaSetDevice`
        // — see `Engine::bind_thread`) has to hear about the hand-off, and
        // this is the one instant at which it has happened and no verb has
        // run yet. Every verb below runs on this thread and nowhere else.
        //
        // A refusal is not fatal HERE: it is the same refusal the first verb
        // will answer with, and answering it there names the verb.
        if let Some(engine) = engine.as_mut()
            && let Err(error) = engine.bind_thread()
        {
            tracing::error!(engine_idx, %error, "engine lane could not bind its thread");
        }
        // **AND THE SINK, INSTALLED ONCE**, on the thread that owns the
        // engine and before the first `submit`. A synchronous engine is not
        // asked: there is no instant at which it would call this that is not
        // simply "inside `submit`", and `settles_asynchronously` is the fact
        // the lane branches on rather than an inference from empty readouts.
        //
        // **WHAT RUNS IN HERE RUNS ON THE DRIVER'S CALLBACK THREAD.** It may
        // make no CUDA call and must not block for long: what it does is take
        // one uncontended mutex, publish a handful of release stores into
        // atomic terminal cells, and wake. Every one of those was designed to
        // be safe from an engine's own completion thread — the lane's own
        // comments have said so since F1 ("the physical pool frees resolve on
        // the engine's own completion threads, never on this lane") — and this
        // is the wave that takes them up on it.
        if let Some(engine) = engine.as_mut()
            && engine.settles_asynchronously()
        {
            let book = std::sync::Arc::clone(&settlements);
            let published = broker.clone();
            engine.on_complete(std::sync::Arc::new(move |at: engine::StepDone, outcome| {
                book.settled(at.frame, &outcome, &published);
            }));
        }
        // A launch this loop has already RECEIVED but not yet served: the
        // next-fire lookahead inside `fire_frame` takes a queued launch out
        // of `launch_rx` early so the engine can be told its composition
        // (`Engine::expect_fire`) before the fire ahead of it runs. FIFO is
        // kept by construction — the stash is always the oldest launch not
        // yet served, and the loop serves it before it receives anything
        // else — and the panic arm below answers its owed frame like any
        // other, because a stashed launch nobody answers is the same
        // forever-in-flight frame the `owed` machinery exists for.
        let mut stash: Option<LaneRequest> = None;
        // Whose turn the lane is serving; see `next_request`.
        let mut lane_turn = LaneTurn::default();
        loop {
            let request = match stash.take() {
                Some(stashed) => {
                    // The lookahead took this one out of `launch_rx` itself,
                    // so `next_request` never saw it. Count it anyway, or a
                    // stash chain would be a second way for the launches to
                    // hold the lane without ever owing the controls a turn.
                    lane_turn.took_launch();
                    stashed
                }
                None => match Self::next_request(&launch_rx, &control_rx, &mut lane_turn) {
                    Ok(request) => request,
                    Err(()) => break,
                },
            };
            let lane_began = Instant::now();
            let lane_was_control = matches!(request, LaneRequest::Control { .. });
            let lane_was_prefill = matches!(request, LaneRequest::Launch { prefill: true, .. });
            let lane_was_work = lane_was_control || matches!(request, LaneRequest::Launch { .. });
            let _lane_charge = LaneCharge {
                stats: &stats,
                began: lane_began,
                control: lane_was_control,
                charge: lane_was_work,
                prefill: lane_was_prefill,
            };
            // The token this request must be answered with, taken before the
            // work so a panic can still answer it. Every arm below replies
            // exactly once; a panic that skips its reply does not fail the
            // request, it leaves the frame in flight forever -- measured, on a
            // panic in the shared program interpreter, as
            // `engine 0 stalled for 7030.132606596s (no progress, work queued
            // or in flight)` until the process was killed by hand.
            let owed = Owed::of(&request);
            // Answering the owed frame is only possible if the panic unwinds
            // to here. Under `panic = "abort"` the stall this replaced is
            // traded for killing every other lane and session too, silently.
            #[cfg(panic = "abort")]
            compile_error!(
                "the engine lane answers its owed frame from the panic path, \
                 which requires unwinding; under `panic = \"abort\"` a panic \
                 in one lane takes down every session the runtime is serving"
            );
            let handled =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    match request {
                        LaneRequest::Launch {
                            token, submission, ..
                        } => {
                            // MUTABLE, because `fire_frame` MOVES each
                            // step's submission into the one
                            // `FrameSubmission` the engine is handed. The
                            // frame keeps everything the runtime still needs
                            // after the device has it — the terminal cells,
                            // the per-lane instances — and nothing copies a
                            // token vector to cross the boundary.
                            let LaneLaunch(mut frame) = submission;
                            // Folded admission, and it does not retry (alto
                            // E, article 4). Both refusals are ERROR VARIANTS
                            // — `Error::{Exhausted, Impossible}` — and both
                            // are terminal for this frame: everything a
                            // device gate could refuse was proved impossible
                            // at `submit_frame`, so a refusal here is a
                            // contract violation rather than back-pressure.
                            // Keeping them as errors rather than a third `Ok`
                            // shape is what makes a caller that ignores the
                            // distinction fail loudly instead of silently
                            // dropping a submission.
                            let result = match engine.as_mut() {
                                Some(engine) => {
                                    crate::probe_fire!(stats.fire.execute.engine_fire_us, {
                                        Self::fire_frame(
                                            engine,
                                            &channels,
                                            &mut frame,
                                            &launch_rx,
                                            &mut stash,
                                            &broker,
                                            &settlements,
                                        )
                                    })
                                }
                                None => Err("engine has no backend installed".to_string()),
                            };
                            if let Err(reason) = &result {
                                // EVERY CELL THE FRAME OWNS, FAILED. The engine used
                                // to publish these itself across the ABI; it answers
                                // a `Result` now, so the runtime writes them — and a
                                // frame that never reached the device must still
                                // settle, or its work items park forever (the
                                // 850-second hang `settle_control` was written for).
                                let _ = reason;
                                let cells: Vec<_> = frame.terminal_cells().collect();
                                crate::engine::completion::settle(
                                    &cells,
                                    crate::engine::completion::TERMINAL_OUTCOME_FAILED,
                                );
                            }
                            let _ = reply_tx
                                .send(SchedulerItem::Lane(LaneReply::LaunchDone { token, result }));
                        }
                        LaneRequest::Control { token, item } => {
                            let commit =
                                Self::execute_control(engine_idx, &mut engine, &mut channels, *item);
                            let _ = reply_tx.send(SchedulerItem::Lane(LaneReply::ControlDone {
                                token,
                                commit,
                            }));
                        }
                        LaneRequest::Shutdown { response } => {
                            let _ = response.send((engine.take(), std::mem::take(&mut channels)));
                            return true;
                        }
                    }
                    false
                }));
            match handled {
                Ok(true) => return,
                Ok(false) => {}
                Err(_) => {
                    // The panic hook has already printed it. What is left is
                    // the frame nobody will complete, and the queue behind it.
                    tracing::error!(
                        "engine lane panicked mid-request; failing it and every request \
                         behind it rather than leaving them in flight"
                    );
                    // **AND EVERY FRAME THE DEVICE STILL OWES** (alto F2b).
                    // A panic leaves frames registered whose completion
                    // callbacks may never arrive — the shell they were
                    // enqueued on is in an unknown state — and a frame nobody
                    // completes is the forever-in-flight stall the `owed`
                    // machinery exists for, one layer down. Failing them here
                    // is the same verdict `owed` gives the request.
                    settlements.close_all(&broker);
                    owed.answer(&reply_tx, "the engine lane panicked serving this request");
                    // A launch the lookahead stashed is queued work the
                    // channels no longer hold — fail it with the queue, or
                    // its frame is the in-flight-forever stall the `owed`
                    // machinery answers everywhere else.
                    if let Some(stashed) = stash.take() {
                        Owed::of(&stashed).answer(&reply_tx, "the engine lane is down after a panic");
                    }
                    Self::drain_poisoned(&launch_rx, &control_rx, &reply_tx, engine, channels);
                    return;
                }
            }
        }
        // Worker dropped its sender without a shutdown handshake (panic
        // path): release the engine here.
        drop(engine.take());
    }

    /// Answer every request still queued, and every one that arrives, with the
    /// same failure -- the lane is gone and its engine is not touched again.
    ///
    /// The engine is LEAKED rather than dropped. A panic mid-fire tears state
    /// the destructor would then run over (a command buffer recorded and never
    /// submitted, a pool half resized), and a second panic in a `Drop` during
    /// unwinding aborts the process. Leaking an engine in the seconds before an
    /// operator restarts the server is the cheaper of the two.
    fn drain_poisoned(
        launch_rx: &crossbeam::channel::Receiver<LaneRequest>,
        control_rx: &crossbeam::channel::Receiver<LaneRequest>,
        reply_tx: &crossbeam::channel::Sender<SchedulerItem>,
        engine: Option<EngineBox>,
        channels: ChannelJoin,
    ) {
        std::mem::forget(engine);
        drop(channels);
        // A fresh turn: the drain answers everything on both queues, so the
        // order it takes them in decides nothing.
        let mut turn = LaneTurn::default();
        while let Ok(request) = Self::next_request(launch_rx, control_rx, &mut turn) {
            if let LaneRequest::Shutdown { response } = request {
                let _ = response.send((None, ChannelJoin::new()));
                return;
            }
            Owed::of(&request).answer(reply_tx, "the engine lane is down after a panic");
        }
    }

    /// The engine half of the old `dispatch_ordered_item`: everything a
    /// control does against the engine and the lane-owned `channels` set,
    /// with worker-map effects returned as a [`LaneCommit`]. Failures respond
    /// directly from here (after lane-side rollback) — only effects that
    /// must be ordered with worker state travel back.
    /// **Submit one frame, and settle its steps' cells** (alto design §2;
    /// articles 1, 2 and 4).
    ///
    /// # What changed, and what deliberately did not
    ///
    /// This loop used to call `Engine::fire` once per step, with a retry
    /// around each call. That shape could not be made to obey article 4: a
    /// frame whose third step answered `Exhausted` had already written the
    /// first two steps' KV, so "retry the frame" was not a thing the runtime
    /// could offer and "retry the step" left a partial frame on the device
    /// while it waited. `submit` takes the whole frame, admits it or refuses
    /// it with zero side effects, and the retry is therefore around the
    /// FRAME — which is also what makes `Error::is_retryable` a legal thing
    /// to act on.
    ///
    /// **THE SETTLEMENT TAKES WHICHEVER SHAPE THE ENGINE HAS** (F2b).
    /// `Engine::settles_asynchronously` is the fact this branches on, and it
    /// is a fact rather than an inference from empty readouts — a frame every
    /// lane of which asked for `Readout::None` has empty readouts and settled
    /// synchronously, and a caller that conflated the two would park forever
    /// on the second.
    ///
    /// ```text
    /// synchronous (metal)   submit returns -> the device is done -> settle
    ///                       the terminal cells here, wake here, retire here
    /// asynchronous (cuda)   submit returns -> the device is RUNNING -> register
    ///                       the frame with the broker and return a completion
    ///                       nobody has settled; the engine's own callback
    ///                       publishes the cells, wakes the guests and retires
    ///                       the batch
    /// ```
    ///
    /// **AND THIS THREAD DOES NOT WAIT FOR THE SECOND ONE**, which is the
    /// whole of article 1 on the runtime's side: the lane returns to
    /// `next_request` with the frame still on the device, takes the next
    /// launch off the queue and enqueues it behind. What bounds the run-ahead
    /// is `frame_dispatch_depth` — the in-flight accounting the frame policy
    /// already keeps, now measuring something real — rather than the length of
    /// a device wait.
    ///
    /// **AND IT DOES NOT RETRY** (alto E, design §1 article 4). A bounded
    /// sleep-and-resubmit loop stood around `submit`, answering
    /// `Error::Exhausted` by parking the lane for 200 us and offering the
    /// identical frame again, up to ~5 s. Article 4 has no such outcome:
    /// everything a device gate could refuse is proved impossible at submit —
    /// `pipeline::fire::validate_frame` walks the frame's steps in slot order
    /// and proves device-only ring occupancy, host-writer staging and reader
    /// pressure against the channels' declared capacities — so a refusal that
    /// reaches this line is a contract violation and is reported as one. The
    /// lane fails the frame, settles its cells FAILED, and moves on; nothing
    /// sleeps, and no later frame is held behind a frame that will never fit.
    fn fire_frame(
        engine: &mut EngineBox,
        channels: &ChannelJoin,
        frame: &mut crate::engine::FrameFire,
        launch_rx: &crossbeam::channel::Receiver<LaneRequest>,
        stash: &mut Option<LaneRequest>,
        broker: &crate::engine::CompletionBroker,
        settlements: &std::sync::Arc<crate::engine::completion::FrameSettlements>,
    ) -> std::result::Result<SubmissionCompletion, String> {
        use crate::engine::completion;

        // ── THE NEXT LAUNCH, STATED (`Engine::expect_fire`, advisory).
        //    A prebind wants a SUCCESSOR's composition on the table before a
        //    step fires, because the hidden window it uses is inside the fire
        //    itself — after the launch, before the sync. The CUDA fold was
        //    where that was measured and it is deleted (the tier-2 campaign:
        //    a body pays nothing per fire, so there is nothing to prebind);
        //    `engine_metal` is the consumer now, and the seam is unchanged.
        //    Two places know a successor, and the frame verb splits them:
        //
        //    * the frame's own next step is the ENGINE's to see now. A frame
        //      crosses whole, so `submit` states each step's successor from
        //      the steps it is holding; the loop that used to do it here
        //      would have had to state it before handing the frame over,
        //      which is one hint too early for every step but the first.
        //    * a whole launch already queued behind this one — the
        //      run-ahead's depth-2 shape — is only visible from this thread.
        //      `launch_rx` has no peek, so the lookahead RECEIVES it into
        //      `stash`; the run loop serves it next, ahead of everything,
        //      which is the order the channel held.
        //
        //    Stated once per frame rather than once per step, and outside the
        //    retry loop: an admission refusal re-submits the same frame, and
        //    a hint consumed by a retried submission costs one wasted prebind
        //    — the advisory contract's price for being wrong.
        if stash.is_none()
            && let Ok(queued) = launch_rx.try_recv()
        {
            if let LaneRequest::Launch { submission, .. } = &queued
                && let Some(first) = submission.0.steps.first()
            {
                engine.expect_fire(&first.submission);
            }
            *stash = Some(queued);
        }

        // ── THE FRAME, AS ONE SUBMISSION. Moved, not copied: a step's
        //    submission has done its runtime-side job by now and everything
        //    the runtime still needs after the device has the frame — the
        //    terminal cells, the per-lane instances — lives beside it rather
        //    than inside it. So a frame reaches the engine without a single
        //    token vector being cloned, and the retry below re-submits the
        //    same value rather than rebuilding it.
        let submitted = ::engine::FrameSubmission {
            steps: frame
                .steps
                .iter_mut()
                .map(|step| std::mem::take(&mut step.submission))
                .collect(),
        };

        // ── THE CHANNEL JOIN, IN (`palo B2`). Every cell the guest has put
        //    into a host ring since the last fire crosses into the device
        //    ring the attached instance's pass reads.
        //
        //    **IT SITS AROUND THE FRAME WHERE IT SAT AROUND THE FIRE, AND
        //    PASS-ATOMICITY IS UNCHANGED.** The guarantee the pump makes is
        //    that a guest's `take` sees the cells of a pass that FINISHED and
        //    never of one that did not; a frame is admitted whole or not at
        //    all, and every attached pass in it committed before `submit`
        //    answered `Ok`. What changed for a multi-step frame is the
        //    interleaving: all the steps' cells go in before the frame does
        //    and all come out after it — which is the ordering article 2
        //    requires anyway, since a host pump between two steps IS a host
        //    touch between two waves.
        //
        //    **AND FOR A DEVICE-COMMIT ENGINE IT IS ALREADY ALMOST NOTHING**
        //    (alto F2a). Every channel whose host ring the engine published
        //    is ADOPTED: the guest wrote its cell into the very bytes
        //    `channel::pull_validate` reads, so `pump_in` moves nothing and
        //    only wakes. **Wave E: this call site and its partner below are
        //    what settlement-through-the-broker retires** — the wake is the
        //    last thing either does on the CUDA path, and a broker that woke
        //    the waiter on the completion callback would leave nothing here
        //    at all.
        //
        //    It runs ONCE. It used to sit inside the retry loop, because a
        //    frame that answered `Exhausted` consumed nothing and the guest
        //    might publish what would unblock the next attempt. There is no
        //    next attempt (article 4), so there is no second pump.
        for step in &submitted.steps {
            for attachment in &step.attachments {
                if let Err(error) = channels.pump_in(engine.as_mut(), attachment.instance) {
                    return Err(format!("channel publish: {error}"));
                }
            }
        }
        // ── ADMISSION IS THE RUNTIME'S, AND IT ALREADY HAPPENED. `submit`
        //    admits the frame or refuses it with zero side effects, and both
        //    refusals are terminal here: `Exhausted` means a resource the
        //    static admission at `submit_frame` was supposed to have proved
        //    available is not (a contract violation, named), and `Impossible`
        //    is a ceiling this deployment was baked against.
        let ticket = match engine.submit(&submitted) {
            Ok(ticket) => ticket,
            Err(error) if error.is_retryable() => {
                return Err(format!(
                    "frame admission answered a retryable refusal past static \
                     admission, which article 4 forbids: {error}"
                ));
            }
            Err(error) => return Err(format!("{error}")),
        };

        // ── AND OUT. `Engine::submit` answered `Ok`, so every attached pass
        //    COMMITTED — a blocked or declined one is an error, and a frame
        //    that never committed has nothing to take.
        //
        //    For an adopted channel this moves nothing: the committed cell
        //    reached the guest's mirror through `channel::scatter_publish` and
        //    the engine advances the tail word at settle. All that is left is
        //    the wake — and against an asynchronous engine the wake is
        //    DEFERRED, because waking here would tell the guest to read a cell
        //    the device is still computing. The ids ride into the frame's
        //    settlement instead.
        let asynchronous = engine.settles_asynchronously();
        let mut wakes: Vec<u64> = Vec::new();
        for step in &submitted.steps {
            for attachment in &step.attachments {
                let deferred = asynchronous.then(|| &mut wakes);
                if let Err(error) =
                    channels.pump_out_with(engine.as_mut(), attachment.instance, deferred)
                {
                    return Err(format!("channel take: {error}"));
                }
            }
        }

        if !asynchronous {
            // THE SYNCHRONOUS SHAPE, UNCHANGED: the walk completed before
            // `submit` returned, so every step's work items are committed the
            // moment this line is reached.
            for step in &frame.steps {
                completion::settle(&step.terminal_cells, completion::TERMINAL_OUTCOME_SUCCESS);
            }
            return Ok(SubmissionCompletion::ready());
        }

        // ── THE ASYNCHRONOUS SHAPE. The device is running; the receipt is a
        //    correlation id. Register the frame's cells and its deferred wakes
        //    against `FrameTicket::id`, hand back a completion nobody has
        //    settled, and go take the next launch off the queue.
        //
        //    The completion is what the run-ahead depth accounting parks on,
        //    so a batch retires when the DEVICE is done with it rather than
        //    when the host finished enqueueing it — which is what makes
        //    `frame_dispatch_depth` a real bound rather than a formality.
        let completion = broker.submission_completion(waker::FIRST_COMPLETION_EPOCH);
        let cells: Vec<_> = frame.terminal_cells().collect();
        settlements.expect(
            ticket.id,
            ticket.steps.len(),
            cells,
            wakes,
            &completion,
            broker,
        );
        Ok(completion)
    }

    fn execute_control(
        engine_idx: usize,
        engine: &mut Option<EngineBox>,
        channels: &mut ChannelJoin,
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
                match engine.as_mut() {
                    Some(engine) => {
                        let submitted = crate::engine::verbs::settled(match plan {
                            PreLaunchCopy::Kv(plan) => engine.copy_kv(&plan),
                            PreLaunchCopy::State(plan) => engine.copy_state(&plan),
                        });
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
                        logical_completion.reject_unsubmitted("engine has no backend installed");
                        LaneCommit::AsyncControl {
                            result: Err("engine has no backend installed".to_string()),
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
                let result = match engine.as_mut() {
                    // The host-codegen splice happens HERE, on the layer that
                    // holds the engine handle and can therefore ask which
                    // backend the plan is bound for. It used to happen inside
                    // the engine layer, which had to reach into
                    // `crate::pipeline` to do it -- against its own header.
                    Some(engine) => {
                        let backend = crate::engine::verbs::codegen_backend(engine);
                        let plan = crate::pipeline::program::with_host_codegen(&plan, backend);
                        engine.register_program(&plan).map_err(anyhow::Error::from)
                    }
                    None => Err(anyhow!("engine has no backend installed")),
                };
                match result {
                    Ok(program_id) => {
                        if response.send(Ok(program_id)).is_err() {
                            tracing::warn!(
                                operation = "register_program",
                                program_hash = format_args!("0x{:016x}", plan.program_hash),
                                "scheduler RPC cancelled after program registration; retaining engine-lifetime program"
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
                        channel_id = plan.id,
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::None;
                }
                let result = if channels.contains(plan.id) {
                    Err(anyhow!("channel {} is already registered", plan.id))
                } else {
                    match engine.as_mut() {
                        Some(engine) => crate::engine::verbs::register_channel(
                            engine, engine_idx, &plan,
                        )
                        .inspect(|channel| {
                            channels.insert(channel.clone(), plan.host_role);
                        }),
                        None => Err(anyhow!("engine has no backend installed")),
                    }
                };
                match result {
                    Ok(channel) => {
                        if let Err(Ok(channel)) = response.send(Ok(channel)) {
                            if let Some(engine) = engine.as_mut() {
                                Self::rollback_channel_set(
                                    engine,
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
                let result = match engine.as_mut() {
                    Some(engine) => Self::register_channel_set(engine, engine_idx, channels, &plans),
                    None => Err(anyhow!("engine has no backend installed")),
                };
                match result {
                    Ok(registered) => {
                        if let Err(Ok(registered)) = response.send(Ok(registered)) {
                            if let Some(engine) = engine.as_mut() {
                                Self::rollback_channel_set(
                                    engine,
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
                    EngineLoop::release_wait_slots([plan.pacing_wait_id]);
                    tracing::warn!(
                        operation = "bind_instance",
                        program_id = plan.program_id(),
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::BindFinished { pipeline_id };
                }
                match engine.as_mut() {
                    Some(engine) => match engine.bind_instance(&plan.binding).map(|bound| crate::engine::BoundInstance::new(plan.engine_id, &bound, plan.pacing_wait_id)).map_err(anyhow::Error::from) {
                        Ok(bound) => {
                            // WHICH CHANNEL IS WHICH DENSE SLOT, kept for the
                            // pump. `InstanceBinding::channels` is the
                            // package's declaration order, which is the
                            // numbering `publish_channel`/`take_channel`
                            // address, and the lane is the only party that
                            // sees both it and the engine.
                            channels.bind(bound.instance_id, plan.binding.channels.clone());
                            LaneCommit::BindInstance {
                                pipeline_id,
                                bound,
                                respond: BindRespond::Bind(response),
                            }
                        }
                        Err(error) => {
                            if response.send(Err(error)).is_err() {
                                EngineLoop::release_wait_slots([plan.pacing_wait_id]);
                            }
                            LaneCommit::BindFinished { pipeline_id }
                        }
                    },
                    None => {
                        if response
                            .send(Err(anyhow!("engine has no backend installed")))
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
                    EngineLoop::release_channel_plan_wait_slots(&plans);
                    EngineLoop::release_wait_slots([bind.pacing_wait_id]);
                    tracing::warn!(
                        operation = "register_channels_bind",
                        program_id = bind.program_id(),
                        "scheduler RPC cancelled before resource creation"
                    );
                    return LaneCommit::BindFinished { pipeline_id };
                }
                let Some(engine) = engine.as_mut() else {
                    if response
                        .send(Err(anyhow!("engine has no backend installed")))
                        .is_err()
                    {
                        EngineLoop::release_channel_plan_wait_slots(&plans);
                        EngineLoop::release_wait_slots([bind.pacing_wait_id]);
                    }
                    return LaneCommit::BindFinished { pipeline_id };
                };
                let registered = match Self::register_channel_set(engine, engine_idx, channels, &plans) {
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
                        engine,
                        channels,
                        &registered,
                        "register_channels_bind",
                        true,
                    );
                    EngineLoop::release_registered_channel_wait_slots(&registered);
                    Self::release_wait_slots([bind.pacing_wait_id]);
                    return LaneCommit::BindFinished { pipeline_id };
                }
                let program_registered = program.is_some();
                if let Some(plan) = &program {
                    let backend = crate::engine::verbs::codegen_backend(engine);
                    let plan = &crate::pipeline::program::with_host_codegen(plan, backend);
                    match engine.register_program(plan).map_err(anyhow::Error::from) {
                        Ok(program_id) => bind.binding.program = program_id,
                        Err(error) => {
                            Self::rollback_channel_set(
                                engine,
                                channels,
                                &registered,
                                "register_channels_bind",
                                false,
                            );
                            if response.send(Err(error)).is_err() {
                                EngineLoop::release_registered_channel_wait_slots(&registered);
                                Self::release_wait_slots([bind.pacing_wait_id]);
                            }
                            return LaneCommit::BindFinished { pipeline_id };
                        }
                    }
                }
                if response.is_closed() {
                    Self::rollback_channel_set(
                        engine,
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
                            program_id = bind.program_id(),
                            "scheduler RPC cancelled after program registration; retaining engine-lifetime program"
                        );
                    }
                    return LaneCommit::BindFinished { pipeline_id };
                }
                match engine.bind_instance(&bind.binding).map(|bound| crate::engine::BoundInstance::new(bind.engine_id, &bound, bind.pacing_wait_id)).map_err(anyhow::Error::from) {
                    Ok(bound) => {
                        // As the `BindInstance` arm above: the dense channel
                        // order the pump addresses by.
                        channels.bind(bound.instance_id, bind.binding.channels.clone());
                        LaneCommit::BindInstance {
                            pipeline_id,
                            bound,
                            respond: BindRespond::ChannelsBind {
                                registered,
                                program_id: bind.program_id(),
                                program_registered,
                                response,
                            },
                        }
                    }
                    Err(error) => {
                        Self::rollback_channel_set(
                            engine,
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
            QueuedItem::CopyKv { plan, response } => match engine.as_mut() {
                Some(engine) => match crate::engine::verbs::settled(engine.copy_kv(&plan)) {
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
                    let _ = response.send(Err(anyhow!("engine has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("engine has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::CopyKvTracked { plan, completion } => match engine.as_mut() {
                Some(engine) => match crate::engine::verbs::settled(engine.copy_kv(&plan)) {
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
                    completion.resolve(&Err(anyhow!("engine has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("engine has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::CopyState { plan, response } => match engine.as_mut() {
                Some(engine) => match crate::engine::verbs::settled(engine.copy_state(&plan)) {
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
                    let _ = response.send(Err(anyhow!("engine has no backend installed")));
                    LaneCommit::AsyncControl {
                        result: Err("engine has no backend installed".to_string()),
                    }
                }
            },
            QueuedItem::CloseInstance { id, .. } => match engine.as_mut() {
                // The worker already gated existence/pacing/quiescence before
                // posting; the map removal happens at commit.
                Some(engine) => match engine.close_instance(id).map_err(anyhow::Error::from) {
                    Ok(()) => {
                        channels.unbind(id);
                        LaneCommit::CloseInstance { id }
                    }
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
            QueuedItem::CloseChannels { ids } => {
                for id in ids {
                    let result = if !channels.contains(id) {
                        Err(anyhow!("channel {id} is unknown or stale"))
                    } else {
                        match engine.as_mut() {
                            Some(engine) => Self::close_channel(engine, id).map(|()| {
                                channels.remove(id);
                            }),
                            None => Err(anyhow!("engine has no backend installed")),
                        }
                    };
                    if let Err(err) = result {
                        tracing::warn!(channel_id = id, ?err, "scheduler close_channel failed");
                    }
                }
                LaneCommit::None
            }
        }
    }

    /// Close one channel on the engine, tolerating the shell that has no
    /// standalone channel to close.
    ///
    /// **BINDING IS REGISTRATION FOR A SHELL WHOSE RINGS ARE ITS INSTANCES'**
    /// (`Engine::register_channel`'s own doc, and `verbs::register_channel`
    /// tolerates the same refusal on the way in). Such a shell allocated
    /// nothing for a bare channel id and frees nothing for it, so
    /// `Unsupported` here means the close SUCCEEDED as far as anything the
    /// engine owns is concerned — and treating it as a failure would leave
    /// the runtime's own ring in the lane's table forever, held alive by an
    /// error it re-logs on every teardown.
    fn close_channel(engine: &mut EngineBox, id: u64) -> Result<()> {
        match engine.close_channel(id) {
            Ok(()) | Err(engine::Error::Unsupported { .. }) => Ok(()),
            Err(error) => Err(anyhow::Error::from(error)),
        }
    }

    /// Register a set of channels with all-or-nothing rollback (the shared
    /// body of `RegisterChannels` and `RegisterChannelsBind`).
    ///
    fn register_channel_set(
        engine: &mut EngineBox,
        engine_idx: usize,
        channels: &mut ChannelJoin,
        plans: &[ChannelRegistration],
    ) -> Result<Vec<RegisteredChannel>> {
        let mut registered = Vec::with_capacity(plans.len());
        let mut registered_ids = Vec::with_capacity(plans.len());
        for plan in plans {
            if channels.contains(plan.id) {
                for channel_id in registered_ids.iter().rev() {
                    let _ = engine.close_channel(*channel_id);
                    channels.remove(*channel_id);
                }
                return Err(anyhow!("channel {} is already registered", plan.id));
            }
            match crate::engine::verbs::register_channel(engine, engine_idx, plan) {
                Ok(channel) => {
                    // THE RING, THE ROLE AND THE ID, all three kept: the pump
                    // that joins this host ring to the device half needs to
                    // know where the cells are and which way they travel, and
                    // both facts are in hand exactly here.
                    channels.insert(channel.clone(), plan.host_role);
                    registered_ids.push(plan.id);
                    registered.push(channel);
                }
                Err(cause) => {
                    for channel_id in registered_ids.iter().rev() {
                        let _ = engine.close_channel(*channel_id);
                        channels.remove(*channel_id);
                    }
                    return Err(cause);
                }
            }
        }
        Ok(registered)
    }

    fn rollback_channel_set(
        engine: &mut EngineBox,
        channels: &mut ChannelJoin,
        registered: &[RegisteredChannel],
        operation: &'static str,
        cancellation: bool,
    ) {
        for channel in registered.iter().rev() {
            let channel_id = channel.binding.channel_id;
            match Self::close_channel(engine, channel_id) {
                Ok(()) => {
                    channels.remove(channel_id);
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

    /// **NOTHING TO RELEASE ANY MORE, AND THAT IS THE POINT.** A
    /// registration used to CARRY its two wait slots — the runtime allocated
    /// them and handed them across so a C engine could publish into them — so
    /// a cancelled registration had to give them back. The contract's
    /// `RegisteredChannel` ANSWERS them, which means an unregistered channel
    /// never allocated any. Kept as a named no-op because the cancellation
    /// paths read as a pair with `release_registered_channel_wait_slots`,
    /// which still has real work.
    fn release_channel_plan_wait_slots(plans: &[ChannelRegistration]) {
        let _ = plans;
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
        let table = waker::WakerTable::global();
        table.sweep(&wait_ids);
        for wait_id in wait_ids {
            table.deregister(wait_id);
            table.free(wait_id);
        }
    }
}

enum QueuedItem {
    /// Boxed: the queue is rotated and compacted item-by-item on every
    /// dispatcher pass, and an inline `PendingRequest` made `QueuedItem`
    /// 1280 bytes — a cohort boundary moved ~800 MB through `VecDeque`
    /// rotations alone. The indirection makes every queue move a pointer
    /// move; the payload itself never moves.
    Launch(QueuedLaunch),
    PreLaunchCopy {
        plan: PreLaunchCopy,
        logical_completion: WorkItemCompletion,
        process_id: Option<ProcessId>,
        pipeline_id: Option<ProcessId>,
    },
    RegisterProgram {
        plan: ProgramRegistration,
        response: tokio::sync::oneshot::Sender<Result<u64>>,
    },
    RegisterChannel {
        plan: ChannelRegistration,
        response: tokio::sync::oneshot::Sender<Result<RegisteredChannel>>,
    },
    RegisterChannels {
        plans: Vec<ChannelRegistration>,
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
        plans: Vec<ChannelRegistration>,
        /// Some on the program cache's first sight (the engine requires the
        /// instance's channels registered BEFORE the program — status -5
        /// otherwise — so registration must ride between channels and bind
        /// inside the one dispatch); None when the hash is already
        /// registered, with `bind.program_id()` carrying the cached id.
        program: Option<ProgramRegistration>,
        bind: InstanceBindingPlan,
        response:
            tokio::sync::oneshot::Sender<Result<(Vec<RegisteredChannel>, u64, BoundInstance)>>,
    },
    CopyKv {
        plan: ::engine::KvCopy,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CopyKvTracked {
        plan: ::engine::KvCopy,
        completion: ControlCompletion,
    },
    CopyState {
        plan: StateCopy,
        response: tokio::sync::oneshot::Sender<Result<SubmissionCompletion>>,
    },
    CloseInstance {
        id: u64,
        pacing_wait_id: u64,
    },
    /// A coalesced run of channel closes: one lane round trip retires the
    /// whole batch (per-id engine calls inside), so a 512-lane cohort's
    /// ~9.2k teardown closes cost ~512 control posts instead of ~9.2k.
    CloseChannels {
        ids: Vec<u64>,
    },
}

/// A queued launch, plus the only two fields the dispatcher's queue scan
/// reads, mirrored inline next to the box.
///
/// [`BatchScheduler::scan_queue`] walks the whole queue and previously read
/// both fields *through* the box, which costs one cache miss per queued item.
/// The scan runs a fixed ~250 times per 1000 tokens no matter how many
/// processes are admitted, so that per-item miss made the scan's cost linear
/// in queue depth and therefore the host's scheduling cost linear in
/// concurrency while the work stayed constant: measured 13.9 us/scan at 256
/// admitted processes against 24.1 us at 512 (mixed-phase shape, same token
/// count both sides), 3.9 s vs 6.5 s of loop time for identical work.
///
/// The mirror cannot go stale, structurally: `QueuedLaunch` hands out only
/// `&PendingRequest` (there is deliberately no `DerefMut`), so neither field
/// can be reassigned while the item is queued. Both are already final by the
/// time they are mirrored — `logical_fire_id` is assigned at construction and
/// the frame stamp is synthesized at ACCEPT, before `queue_attempt` hands the
/// item to the queue.
struct QueuedLaunch {
    fire_id: u64,
    framed: bool,
    request: Box<PendingRequest>,
}

impl QueuedLaunch {
    fn new(request: Box<PendingRequest>) -> Self {
        Self {
            fire_id: request.logical_fire_id,
            framed: request.frame.is_some(),
            request,
        }
    }

    fn into_request(self) -> Box<PendingRequest> {
        self.request
    }
}

impl std::ops::Deref for QueuedLaunch {
    type Target = PendingRequest;
    fn deref(&self) -> &Self::Target {
        &self.request
    }
}

/// A posted launch's lane lifecycle: the batch enters `in_flight_launches`
/// (and the run-ahead depth) at POST; the engine's verdict arrives as a
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
    #[allow(
        clippy::vec_box,
        reason = "measured: `PendingRequest` is 1408 bytes, and this vec is handed \
                  straight to `batch::build_frame_submission`, which shuffles its \
                  elements between wave/step-group/deferred vecs; the box keeps each \
                  of those moves 8 bytes. Matches that function's signature"
    )]
    requests: Vec<Box<PendingRequest>>,
    started: Instant,
    batch_size: u64,
    total_tokens: usize,
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
    /// Whether launches must wait for this control to settle. True for a
    /// `PreLaunchCopy` (its consumer fire is queued right behind it) and a
    /// pool resize (its pipe drain must not admit new frames under it);
    /// false for standalone copies — their pages are grant-pinned and no
    /// queued fire references them, so frames keep posting while
    /// suspend/restore traffic settles.
    ///
    /// It is also the exclusivity test: a control that holds launches needs
    /// the whole in-flight set empty and blocks every other control while it
    /// settles, exactly as the original single slot did.
    holds_launches: bool,
}

/// How often to re-check a control op that is holding launches while the
/// device sits idle. Matches the frame policy's own gather poll: the same
/// "something will settle shortly, do not sleep on the hang backstop" case.
const CONTROL_SETTLE_POLL_US: u64 = 500;

/// The async-completing controls the worker is waiting on — copies and pool
/// resizes; lifecycle controls execute on the lane without ever entering
/// here.
///
/// Two classes share this set. An **exclusive** control (a `PreLaunchCopy`,
/// whose consumer fire is queued directly behind it, and a pool resize, whose
/// pipe drain IS its ordering mechanism) keeps the original rule: it needs
/// the set empty and, once posted, nothing else may join it.
///
/// **Standalone copies** — the residency planner's suspend/restore traffic —
/// instead settle concurrently with one another. Nothing queued orders
/// against them: their pages are grant-pinned and no queued fire can name
/// one, which `pipe_concurrent_control` already relies on. So a single slot
/// bought no safety, only a queue, and the queue was on the planner's
/// critical path. Measured at 512-way KV contention: up to 7 restores wanted
/// the slot at once (`restoring` p90 = 4, max = 7) and each H2D copy took
/// 22.8 ms end to end against ~3.3 ms of transfer — 1.528 ms/page, versus
/// 0.227 ms/page on the D2H side, which the planner itself issues strictly
/// one at a time and which therefore never queued. The 6.7x asymmetry was
/// the wait for this slot, not the device.
///
/// Nothing here needs a concurrency ceiling: the pending queue is the bound.
/// Only copies the planner has already enqueued can be in flight, and the
/// planner enqueues at most one per suspending or restoring process.
#[derive(Default)]
struct InFlightControls {
    settling: Vec<PendingControl>,
}

impl InFlightControls {
    fn is_empty(&self) -> bool {
        self.settling.is_empty()
    }

    /// Whether anything is still settling.
    fn is_settling(&self) -> bool {
        !self.settling.is_empty()
    }

    fn iter(&self) -> std::slice::Iter<'_, PendingControl> {
        self.settling.iter()
    }

    /// Whether a standalone copy may be posted now: only an exclusive
    /// control can refuse one.
    fn admits_copy(&self) -> bool {
        self.settling.iter().all(|control| !control.holds_launches)
    }

    /// Whether `item` may be posted into this set now. A standalone copy and
    /// a lifecycle control are each refused only by an EXCLUSIVE control: the
    /// copy addresses grant-pinned pages nothing queued can name, and a
    /// lifecycle control never enters this set at all — it executes on the
    /// lane, its engine order guaranteed by the lane FIFO. Blocking lifecycle
    /// controls on a settling copy wedged the fleet: the planner's copies are
    /// in flight almost continuously under churn, so every bind waited out the
    /// whole strict-watchdog window (measured on `churn`: 270 binds at 1.0-2.4 s
    /// end to end against a 59 us engine bind).
    fn admits(&self, item: &QueuedItem) -> bool {
        if BatchScheduler::standalone_copy(item) || BatchScheduler::lifecycle_control(item) {
            !self.holds_launches()
        } else {
            self.is_empty()
        }
    }

    /// Whether any settling control makes queued launches wait.
    fn holds_launches(&self) -> bool {
        self.settling.iter().any(|control| control.holds_launches)
    }

    fn push(&mut self, control: PendingControl) {
        self.settling.push(control);
    }

    fn position_posted(&self, token: u64) -> Option<usize> {
        self.settling.iter().position(
            |control| matches!(control.state, ControlSlotState::Posted { token: t } if t == token),
        )
    }
}

/// What one pass over the pending queue tells the frame dispatcher — see
/// [`BatchScheduler::scan_queue`].
#[derive(Default)]
struct QueueScan {
    /// Stamped fire ids still in the queue (the frame policy resolves
    /// sealed ids that vanished against this set).
    queued_ids: frame::QueuedFireIds,
    /// Lanes a frame post must hold for: only lanes with a queued
    /// `PreLaunchCopy` (order-coupled to its consumer fire).
    blocked_lanes: HashSet<ProcessId>,
    /// The oldest unstamped rider, dispatched as its own batch.
    untracked: Option<u64>,
    /// Every queued fire id in queue order, for the shutdown drain. Only
    /// filled while `stopping` — the steady-state scan never allocates it.
    drain_eligible: Vec<u64>,
}

impl QueueScan {
    /// Reset for reuse, keeping the allocations.
    fn clear(&mut self) {
        self.queued_ids.clear();
        self.blocked_lanes.clear();
        self.untracked = None;
        self.drain_eligible.clear();
    }
}

/// The worker's pending queue, plus an epoch that changes on every mutation.
///
/// The epoch exists so [`BatchScheduler::scan_queue`] can skip a pass whose
/// answer cannot have changed. `DerefMut` bumps it, which is what makes the
/// invalidation total: every `&mut` reach into the queue counts, including
/// rotations that leave the length alone, in-place edits through `iter_mut`,
/// and the rebuild in `post_frame`. A length or endpoint fingerprint would
/// have missed all three. Over-invalidation (a `&mut` taken but not used) is
/// merely a wasted scan.
#[derive(Default)]
struct PendingQueue {
    items: VecDeque<QueuedItem>,
    epoch: u64,
    /// `(epoch, index of the first non-`Launch` item)`.
    first_other: Option<(u64, Option<usize>)>,
    /// `(epoch, index of the first queued close)`.
    first_close: Option<(u64, usize)>,
}

impl PendingQueue {
    fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Replace the contents wholesale, preserving the epoch counter.
    fn replace(&mut self, items: VecDeque<QueuedItem>) {
        self.items = items;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Offset of the first item that is not a `Launch`, or `None` when the
    /// queue is all launches.
    ///
    /// Both this and [`Self::first_close`] answer "where does the launch run
    /// end", which every worker pass asks at least once. At a cohort boundary
    /// the queue is a run of thousands of launches held back by an unsealed
    /// frame, so the linear search — run once per pass and once per queued
    /// control — was measured as the loop's single largest per-pass cost
    /// (~18 us/pass against ~3 us elsewhere). The scan itself is unavoidable;
    /// repeating it while the queue is unchanged is not, so both are cached
    /// against the same epoch that guards [`ScanCache`].
    fn first_other(&mut self) -> Option<usize> {
        if let Some((epoch, idx)) = self.first_other
            && epoch == self.epoch
        {
            return idx;
        }
        let idx = self
            .items
            .iter()
            .position(|item| !matches!(item, QueuedItem::Launch(_)));
        self.first_other = Some((self.epoch, idx));
        idx
    }

    /// Offset of the first queued close, or the length when there is none.
    fn first_close(&mut self) -> usize {
        if let Some((epoch, idx)) = self.first_close
            && epoch == self.epoch
        {
            return idx;
        }
        let idx = self
            .items
            .iter()
            .position(|item| {
                matches!(
                    item,
                    QueuedItem::CloseInstance { .. } | QueuedItem::CloseChannels { .. }
                )
            })
            .unwrap_or(self.items.len());
        self.first_close = Some((self.epoch, idx));
        idx
    }

    /// Move the leading run of launches behind the rest of the queue.
    ///
    /// Equivalent to popping each leading launch off the front and pushing it
    /// back, but as one rotation rather than thousands of individual moves.
    fn rotate_launch_run_to_back(&mut self, run_len: usize) {
        self.items.rotate_left(run_len);
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Insert a bring-up control ahead of the trailing close run.
    fn insert_before_closes(&mut self, item: QueuedItem) {
        let index = self.first_close();
        self.items.insert(index, item);
        self.epoch = self.epoch.wrapping_add(1);
        // The insert shifted the close run one to the right and put a
        // non-close at `index`, so the next control lands after this one and
        // bind-vs-bind order is preserved without rescanning.
        self.first_close = Some((self.epoch, index + 1));
        self.first_other = None;
    }
}

impl std::ops::Deref for PendingQueue {
    type Target = VecDeque<QueuedItem>;
    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

impl std::ops::DerefMut for PendingQueue {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.epoch = self.epoch.wrapping_add(1);
        &mut self.items
    }
}

impl From<VecDeque<QueuedItem>> for PendingQueue {
    fn from(items: VecDeque<QueuedItem>) -> Self {
        Self {
            items,
            epoch: 0,
            first_other: None,
            first_close: None,
        }
    }
}

impl FromIterator<QueuedItem> for PendingQueue {
    fn from_iter<T: IntoIterator<Item = QueuedItem>>(iter: T) -> Self {
        Self {
            items: iter.into_iter().collect(),
            epoch: 0,
            first_other: None,
            first_close: None,
        }
    }
}

/// A [`QueueScan`] plus the queue epoch it was taken at.
#[derive(Default)]
struct ScanCache {
    scan: QueueScan,
    /// `None` until the first scan; otherwise the (epoch, stopping) the
    /// cached scan is valid for.
    taken_at: Option<(u64, bool)>,
}

/// Reused across frames: `post_frame` places each picked fire into its
/// sealed slot, and a fresh `Vec` per frame would be a ~650 KB allocation on
/// the loop's critical path. Compaction `take()`s every slot, so the buffer
/// comes back empty and only ever grows.
type SlotBuffer = Vec<Vec<Option<Box<PendingRequest>>>>;

struct SchedulerControl {
    tx: crossbeam::channel::Sender<SchedulerItem>,
    active_senders: AtomicUsize,
    shutdown_wait: Condvar,
    shutdown_gate: Mutex<()>,
    program_ids: Mutex<HashMap<u64, (u64, ::eta_compiler::codegen::launch::LaunchPackage)>>,
    accepting: AtomicBool,
    stats: Arc<SchedulerStats>,
    /// Which memory this engine's KV pages live in.
    ///
    /// Carried on the handle because the two `*_on` submit paths are handed a
    /// handle and no engine id, and a `KvCopyPlan` they build has to name the
    /// right memory. See `scheduler::device_domain` for what naming the wrong
    /// one cost.
    device_domain: ::engine::MemoryDomain,
}

#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    inner: Arc<SchedulerControl>,
}

impl SchedulerHandle {
    /// The memory this scheduler's engine keeps its KV pages in.
    pub(crate) fn device_domain(&self) -> ::engine::MemoryDomain {
        self.inner.device_domain
    }

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

    /// This engine's lock-free stats snapshot (read by
    /// `scheduler::get_stats`'s cross-engine aggregation).
    pub(crate) fn stats(&self) -> Arc<SchedulerStats> {
        Arc::clone(&self.inner.stats)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "a pass-through to `PendingRequest::direct`: every argument is \
                  forwarded unchanged into that constructor, so the width is that \
                  struct's field list arriving one call earlier"
    )]
    pub fn submit_with_identity_and_copy(
        &self,
        request: crate::engine::FireRequest,
        instance_id: u64,
        completion: WorkItemCompletion,
        pipeline_id: Option<ProcessId>,
        prelaunch_copy: Option<::engine::KvCopy>,
        prelaunch_state_copy: Option<StateCopy>,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                pipeline_id,
                pipeline_id,
                prelaunch_copy,
                prelaunch_state_copy,
                None,
                /*hook_program=*/ false,
                /*lora_program=*/ false,
            ),
        })
    }

    pub fn submit_prebuilt_with_copy(
        &self,
        request: crate::engine::FireRequest,
        instance_id: u64,
        completion: WorkItemCompletion,
        prelaunch_copy: Option<::engine::KvCopy>,
        prelaunch_state_copy: Option<StateCopy>,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                None,
                None,
                prelaunch_copy,
                prelaunch_state_copy,
                None,
                /*hook_program=*/ false,
                /*lora_program=*/ false,
            ),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn submit_prebuilt_tracked_with_copy(
        &self,
        request: crate::engine::FireRequest,
        instance_id: u64,
        completion: WorkItemCompletion,
        process_id: ProcessId,
        pipeline_id: ProcessId,
        prelaunch_copy: Option<::engine::KvCopy>,
        prelaunch_state_copy: Option<StateCopy>,
        frame: Option<FrameStamp>,
        hook_program: bool,
        lora_program: bool,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                Some(process_id),
                Some(pipeline_id),
                prelaunch_copy,
                prelaunch_state_copy,
                frame,
                hook_program,
                lora_program,
            ),
        })
    }

    /// Fire-and-forget frame truncation notice (frame mode): a host frame
    /// submit failed mid-way, so only `submitted` fires of (lane, seq) exist.
    pub fn frame_truncate(&self, lane: ProcessId, seq: u64, submitted: u32) -> Result<()> {
        self.send(SchedulerItem::FrameTruncate {
            lane,
            seq,
            submitted,
        })
    }

    pub(crate) fn nudge(&self) -> Result<()> {
        self.send(SchedulerItem::Nudge)
    }
    pub async fn register_program(&self, plan: ProgramRegistration) -> Result<u64> {
        let program_hash = plan.program_hash;
        {
            let program_ids = self.inner.program_ids.lock().unwrap();
            if let Some((program_id, launch)) = program_ids.get(&program_hash) {
                if launch != &plan.launch {
                    return Err(anyhow!("program hash collision for 0x{program_hash:016x}"));
                }
                return Ok(*program_id);
            }
        }
        let launch = plan.launch.clone();
        let program_id = self
            .request(|response| SchedulerItem::RegisterProgram { plan, response })
            .await??;
        self.inner
            .program_ids
            .lock()
            .unwrap()
            .insert(program_hash, (program_id, launch));
        Ok(program_id)
    }

    /// `engine_id` is an ARGUMENT NOW, because the ring is the runtime's.
    ///
    /// A registration used to carry it (`ChannelRegistrationPlan::engine_id`)
    /// so the engine could stamp it back onto the answer; the contract's
    /// `ChannelRegistration` states only what an engine needs, and the field
    /// that says which registry slot a channel's endpoint belongs to is the
    /// runtime's own.
    pub async fn register_channel(
        &self,
        engine_id: crate::engine::EngineId,
        plan: ChannelRegistration,
    ) -> Result<RegisteredChannel> {
        let _ = engine_id;
        self.request(|response| SchedulerItem::RegisterChannel { plan, response })
            .await?
    }

    /// As [`SchedulerHandle::register_channel`], for a set.
    pub async fn register_channels(
        &self,
        engine_id: crate::engine::EngineId,
        plans: Vec<ChannelRegistration>,
    ) -> Result<Vec<RegisteredChannel>> {
        let _ = engine_id;
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
        engine_id: crate::engine::EngineId,
        plans: Vec<ChannelRegistration>,
        program: ProgramRegistration,
        mut bind: InstanceBindingPlan,
    ) -> Result<(Vec<RegisteredChannel>, BoundInstance)> {
        let _ = engine_id;
        let program_hash = program.program_hash;
        let cached = {
            let program_ids = self.inner.program_ids.lock().unwrap();
            match program_ids.get(&program_hash) {
                Some((program_id, launch)) => {
                    if launch != &program.launch {
                        return Err(anyhow!("program hash collision for 0x{program_hash:016x}"));
                    }
                    Some(*program_id)
                }
                None => None,
            }
        };
        let (program_field, cache_fill) = match cached {
            Some(program_id) => {
                bind.binding.program = program_id;
                (None, None)
            }
            None => (Some(program.clone()), Some(program.launch)),
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
        if let Some(launch) = cache_fill {
            self.inner
                .program_ids
                .lock()
                .unwrap()
                .insert(program_hash, (program_id, launch));
        }
        Ok((registered, bound))
    }

    pub async fn copy_kv(&self, plan: ::engine::KvCopy) -> Result<SubmissionCompletion> {
        self.request(|response| SchedulerItem::CopyKv { plan, response })
            .await?
    }

    pub(crate) fn copy_kv_tracked(
        &self,
        plan: ::engine::KvCopy,
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

    // Only called from `scheduler::dispatch::copy_rs_d2d` (not yet issued by
    // the mock-engine fire path) — see `scheduler::dispatch`'s module doc.
    #[allow(dead_code)]
    pub async fn copy_state(&self, plan: StateCopy) -> Result<SubmissionCompletion> {
        self.request(|response| SchedulerItem::CopyState { plan, response })
            .await?
    }

    pub fn close_instance(&self, id: u64, pacing_wait_id: u64) -> Result<()> {
        self.send(SchedulerItem::CloseInstance { id, pacing_wait_id })
    }

    pub fn close_channel(&self, id: u64) -> Result<()> {
        self.send(SchedulerItem::CloseChannel { id })
    }

    /// Batched form of [`Self::close_channel`] for callers that retire a
    /// whole cohort of channels at once (process teardown).
    pub fn close_channels(&self, ids: Vec<u64>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        self.send(SchedulerItem::CloseChannels { ids })
    }
}

pub struct BatchScheduler {
    engine_id: EngineId,
    handle: SchedulerHandle,
    thread: Option<std::thread::JoinHandle<()>>,
    stats: Arc<SchedulerStats>,
}

impl BatchScheduler {
    pub fn new(
        engine_id: EngineId,
        engine_idx: usize,
        page_size: u32,
        limits: SchedulerLimits,
        request_timeout_secs: u64,
        frame_size: usize,
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
                device_domain: crate::scheduler::device_domain(engine_idx),
            }),
        };
        crate::scheduler::install_scheduler_handle(engine_id, handle.clone());
        let stats_for_loop = Arc::clone(&stats);
        let nudge_tx = handle.inner.tx.clone();
        let thread = std::thread::Builder::new()
            .name(format!("pie-sched-{engine_idx}"))
            .spawn(move || {
                let _request_timeout = Duration::from_secs(request_timeout_secs);
                Self::run(
                    engine_id,
                    rx,
                    nudge_tx,
                    page_size,
                    limits,
                    stats_for_loop,
                    frame_size,
                );
            })
            .expect("spawn pie-sched thread");
        Self {
            engine_id,
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
        crate::scheduler::clear_scheduler_handle(self.engine_id);
        if let Some(thread) = self.thread.take()
            && let Err(err) = thread.join()
        {
            tracing::error!(
                engine_id = self.engine_id,
                ?err,
                "scheduler thread panicked"
            );
        }
    }

    fn run(
        engine_id: EngineId,
        rx: crossbeam::channel::Receiver<SchedulerItem>,
        nudge_tx: crossbeam::channel::Sender<SchedulerItem>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: Arc<SchedulerStats>,
        frame_size: usize,
    ) {
        let lane_reply_tx = nudge_tx.clone();
        let nudge_waker = std::task::Waker::from(Arc::new(NudgeWaker {
            tx: nudge_tx.clone(),
        }));
        let engine = crate::engine::take_engine_backend(engine_id).ok();
        let mut lane = EngineLoop::spawn(engine_id, engine, lane_reply_tx, Arc::clone(&stats));
        // Worker→lane requests not yet replied to (launch posts + control
        // posts). Shutdown may only tear down once this drains — every lane
        // request produces exactly one reply.
        let mut lane_inflight: u64 = 0;
        let mut lane_token: u64 = 0;
        let mut instances = HashMap::new();
        let mut pending = PendingQueue::default();
        let mut scan_cache = ScanCache::default();
        let mut slot_buffer = SlotBuffer::new();
        let mut terminated_processes: HashSet<ProcessId> = HashSet::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = InFlightControls::default();
        let mut stopping = false;
        // THE fire rule: the wait-for-all-active-lanes frame policy, one
        // instance per engine thread, mirroring `instances`/`channels` above.
        // Every backend and every k schedules the same way — at the default
        // k=1 a frame is one wave (each tracked fire admits as
        // a synthesized single-slot frame); density comes from the sealed
        // epoch, throughput from run-ahead depth within it.
        let mut frame_policy = FramePolicy::new(
            frame_size,
            limits.max_forward_requests,
            limits.max_forward_tokens,
            Some(Arc::clone(&stats)),
        );
        frame_policy
            .preload_free_slots(crate::inferlet::process::execution_slot_capacity().unwrap_or(0));
        // Stall self-diagnosis: a scheduler that spins on the backstop with
        // queued or in-flight work and zero progress is deadlocked from the
        // caller's point of view, and every wait in this loop is silent. After
        // 10s of that, print the full state dump so the wedge names itself
        // (then re-print every 60s while it persists).
        let mut stall_since: Option<std::time::Instant> = None;
        let mut stall_dumps: u32 = 0;

        loop {
            let mut progress = false;
            // Epoch drain: a pass consumes only what was queued when it began.
            // A sustained producer flood (the next cohort's bring-up at a
            // generation boundary) otherwise keeps `try_recv` non-empty for
            // tens of milliseconds and holds retire/dispatch hostage behind
            // the live stream — the seal-opening events (leaves, slot
            // releases, first fires) then reach the policy tail-late and the
            // boundary wave dispatches when the mailbox finally runs dry
            // instead of the pass after the last join lands.
            let mailbox_epoch = rx.len();
            for _ in 0..mailbox_epoch {
                let Ok(item) = rx.try_recv() else { break };
                progress = true;
                match item {
                    SchedulerItem::DebugDump { response } => {
                        let _ = response.send(Self::render_debug_dump(
                            &pending,
                            &in_flight_launches,
                            &in_flight_control,
                            &instances,
                            &frame_policy,
                        ));
                    }
                    SchedulerItem::Lane(reply) => {
                        Self::apply_lane_reply(
                            reply,
                            &mut lane_inflight,
                            &mut in_flight_launches,
                            &mut in_flight_control,
                            &mut instances,
                            &mut frame_policy,
                            &nudge_tx,
                        );
                    }
                    item => {
                        Self::enqueue_item(
                            &mut pending,
                            &mut terminated_processes,
                            &mut in_flight_control,
                            &instances,
                            limits,
                            page_size,
                            &mut stopping,
                            &mut frame_policy,
                            item,
                        );
                    }
                }
            }
            progress |= Self::retire_ready_launches(
                &mut in_flight_launches,
                &mut instances,
                &stats,
                &mut frame_policy,
            );
            progress |= Self::retire_ready_control(&mut in_flight_control);
            let (dispatched, wait_hint) = Self::dispatch_ready_items(
                &lane,
                &mut lane_inflight,
                &mut lane_token,
                &mut instances,
                &mut pending,
                &mut in_flight_launches,
                &mut in_flight_control,
                page_size,
                limits,
                &stats,
                &mut frame_policy,
                &mut scan_cache,
                &mut slot_buffer,
                stopping,
            );
            progress |= dispatched;
            if stopping
                && pending.is_empty()
                && in_flight_launches.is_empty()
                && in_flight_control.is_empty()
                && lane_inflight == 0
            {
                break;
            }

            // Cohort-boundary bind deferral: while a successor's arrival is
            // imminent, hold back the bind permits that retiring
            // processes return, so the staged cohort's working-set
            // declaration and prefill construction do not compete with the
            // boundary's own bring-up. Cleared the moment this pass has
            // nothing left to do, which is what makes the hold incapable of
            // being the last thing standing.
            crate::inferlet::process::set_bind_release_hold(
                !stopping
                    && frame_policy.is_joining()
                    && (progress
                        || !in_flight_launches.is_empty()
                        || in_flight_control.is_settling()),
            );

            if progress {
                stall_since = None;
                stall_dumps = 0;
                continue;
            }

            let item = if pending.is_empty()
                && in_flight_launches.is_empty()
                && in_flight_control.is_empty()
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
                // in-flight completions so the engine callback wakes this
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
                for control in in_flight_control.iter() {
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
                // A pending wait-all hold (cold gather / seal barrier /
                // depth-cap poll) re-arms the backstop at its own
                // cadence — never longer than the 250ms hang backstop, so a
                // held wave still fires on time even with no new arrival or
                // completion nudge in between.
                let backstop = Duration::from_millis(250);
                let recv_wait = wait_hint.map(|hold| hold.min(backstop)).unwrap_or(backstop);
                // Attribute the park: sleeping with the device IDLE and an
                // in-flight control op holding launches is the state that
                // produces this cell's ~300 ms stalls, because a `Posted`
                // control slot arms no nudge (see the match above) and only
                // the 250 ms backstop can end the wait.
                let idle_park = in_flight_launches.is_empty();
                let control_park = idle_park && in_flight_control.holds_launches();
                let park_began = Instant::now();
                let parked = rx.recv_timeout(recv_wait);
                if idle_park {
                    let slept = park_began.elapsed().as_micros() as u64;
                    use std::sync::atomic::Ordering::Relaxed;
                    if control_park {
                        stats
                            .fire
                            .quorum
                            .idle_park_control_us
                            .fetch_add(slept, Relaxed);
                        // Name the operation that held the device idle. Same
                        // env switch as the device-idle census; only the long
                        // ones are worth a line.
                        if slept >= frame::idle_dump_threshold_us() {
                            let who: Vec<String> = in_flight_control
                                .iter()
                                .filter(|c| c.holds_launches)
                                .map(|c| {
                                    format!(
                                        "{}({})",
                                        c.operation,
                                        match &c.state {
                                            ControlSlotState::Posted { .. } => "posted",
                                            ControlSlotState::Ready(comp) =>
                                                if comp.is_settled() {
                                                    "ready-settled"
                                                } else {
                                                    "ready-unsettled"
                                                },
                                        }
                                    )
                                })
                                .collect();
                            println!(
                                "[idle-park] {slept}us woke={} holders=[{}]",
                                match &parked {
                                    Ok(_) => "channel",
                                    Err(_) => "backstop",
                                },
                                who.join(",")
                            );
                        }
                    } else {
                        stats
                            .fire
                            .quorum
                            .idle_park_other_us
                            .fetch_add(slept, Relaxed);
                    }
                }
                match parked {
                    Ok(item) => Some(item),
                    Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                        // A settled completion discovered by the backstop
                        // means a wake was lost somewhere — the steady-state
                        // count stays zero (plan §16.2). Shutdown races are
                        // excluded: teardown may legitimately cross a tick.
                        // A wait-all-hold timeout is NOT a lost wake (it is
                        // the wait's own cadence), so it never counts here.
                        let missed = in_flight_launches.front().is_some_and(|front| {
                            matches!(&front.state, LaunchState::Accepted(c) if c.is_settled())
                        }) || in_flight_control.iter().any(|control| {
                            matches!(&control.state, ControlSlotState::Ready(c) if c.is_settled())
                        });
                        if missed && !stopping && wait_hint.is_none() {
                            let total = BACKSTOP_RETIREMENTS.fetch_add(1, Ordering::Relaxed) + 1;
                            tracing::warn!(
                                engine_id,
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
                                "[pie-sched] engine {engine_id} stalled for {stalled_for:?} \
                                 (no progress, work queued or in flight); state:\n{}",
                                Self::render_debug_dump(
                                    &pending,
                                    &in_flight_launches,
                                    &in_flight_control,
                                    &instances,
                                    &frame_policy,
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
                        &in_flight_launches,
                        &in_flight_control,
                        &instances,
                        &frame_policy,
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
                        &mut frame_policy,
                        &nudge_tx,
                    );
                    continue;
                }
                Self::enqueue_item(
                    &mut pending,
                    &mut terminated_processes,
                    &mut in_flight_control,
                    &instances,
                    limits,
                    page_size,
                    &mut stopping,
                    &mut frame_policy,
                    item,
                );
            }
        }

        // The lane has no pending requests here (`lane_inflight == 0` gates
        // the loop exit), so shutdown returns the quiesced engine and the
        // channel registry for teardown.
        let (mut engine, mut channels) = lane.shutdown();
        Self::shutdown_instances(&mut engine, &mut instances);
        Self::shutdown_channels(&mut engine, &mut channels);
        drop(engine.take());
    }

    #[allow(clippy::too_many_arguments)]
    fn render_debug_dump(
        pending: &VecDeque<QueuedItem>,
        in_flight_launches: &VecDeque<PendingLaunchBatch>,
        in_flight_control: &InFlightControls,
        instances: &HashMap<u64, TrackedInstance>,
        frame_policy: &FramePolicy,
    ) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let describe = |request: &PendingRequest| {
            format!(
                "fire {} instance {} pipeline {:?} tracked={} settled={} cancelled={}",
                request.logical_fire_id,
                request.instance_id,
                request.pipeline_id,
                instances.contains_key(&request.instance_id),
                request.completion.is_settled(),
                request.completion.cancel_requested(),
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
                QueuedItem::CloseInstance { id, .. } => format!("CloseInstance {id}"),
                QueuedItem::CloseChannels { ids } => format!("CloseChannels x{}", ids.len()),
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
        if in_flight_control.is_empty() {
            let _ = writeln!(out, "in_flight_control: none");
        }
        for control in in_flight_control.iter() {
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
        let _ = write!(out, "{}", frame_policy.debug_summary());
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_item(
        pending: &mut PendingQueue,
        terminated_processes: &mut HashSet<ProcessId>,
        in_flight_control: &mut InFlightControls,
        instances: &HashMap<u64, TrackedInstance>,
        limits: SchedulerLimits,
        page_size: u32,
        stopping: &mut bool,
        frame_policy: &mut FramePolicy,
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
            // Immediate, not queued. Termination rejects queued work; graceful
            // pipeline close instead releases the wait-set and lets every
            // already-admitted request drain untracked.
            SchedulerItem::ExecutionSlotReleased(pid) => {
                frame_policy.on_execution_slot_released(pid);
            }
            SchedulerItem::ProcessQuiesced(pid) => {
                terminated_processes.remove(&pid);
            }
            SchedulerItem::ExecutionSlotConsumed(pid) => {
                frame_policy.on_execution_slot_consumed(pid);
            }
            SchedulerItem::AdmissionQueued(pid) => {
                frame_policy.on_admission_queued(pid);
            }
            SchedulerItem::AdmissionDequeued(pid) => {
                frame_policy.on_admission_dequeued(pid);
            }
            SchedulerItem::PipelineLeave(pid, owner, kind, response) => {
                if kind == LeaveKind::Terminate {
                    if !terminated_processes.insert(pid) {
                        // Duplicate Terminate (the exit funnel notifies from
                        // the terminate entry point and again from deferred
                        // teardown): the first leave did all the work and
                        // every step below is a no-op for it — skip straight
                        // to the ack a waiting sender may hold.
                        if let Some(response) = response {
                            let _ = response.send(());
                        }
                        return;
                    }
                    // A departing slot holder's release broadcast is now in
                    // flight; the seal keeps gathering its successor (the
                    // ragged-boundary guard — see on_slotted_terminate).
                    frame_policy.on_slotted_terminate(pid);
                    let protected = in_flight_control
                        .iter()
                        .find(|control| control.process_id == Some(pid))
                        .and_then(|control| control.logical_completion.clone());
                    if let Some(completion) = &protected {
                        completion.request_cancel();
                    }
                    Self::reject_pipeline_queued(pending, pid, protected.as_ref());
                }
                match kind {
                    LeaveKind::Close => {
                        // Graceful close keeps queued frames: their accepted
                        // fires drain to settlement like any submitted work.
                        frame_policy.on_lane_leave(pid, owner, false);
                    }
                    LeaveKind::Suspend => {
                        // Process-wide graceful leave (see the variant doc);
                        // `pid` names the process here.
                        frame_policy.on_process_suspend(pid);
                    }
                    LeaveKind::Terminate => {
                        // Terminate rejected the lane's queued fires. Both
                        // ids are the process here, so the owner is `pid`.
                        frame_policy.on_lane_leave(pid, owner.or(Some(pid)), true);
                        frame_policy.on_process_leave(pid);
                    }
                }
                if let Some(response) = response {
                    let _ = response.send(());
                }
            }
            SchedulerItem::Launch {
                pending: mut launch,
            } => {
                let validation = AdmissionLimits::new(limits, page_size);
                let rejection = if launch.completion.cancel_requested() {
                    Some("logical fire cancelled before scheduler admission".to_string())
                } else if launch
                    .process_id
                    .is_some_and(|pid| terminated_processes.contains(&pid))
                {
                    Some("process terminated before scheduler admission".to_string())
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
                    // A rejected mid-frame fire still counts toward its
                    // frame's arrival completeness (the surviving fires
                    // execute; the guest observed the rejection) — UNLESS
                    // its process already terminated: the terminate purge
                    // removed the lane and rejected its queued siblings, so
                    // recording this straggler would resurrect a ghost lane
                    // no future event releases.
                    if let Some(stamp) = launch.frame
                        && !launch
                            .process_id
                            .is_some_and(|pid| terminated_processes.contains(&pid))
                    {
                        frame_policy.on_fire_rejected_at_admission(stamp, launch.process_id);
                    }
                    launch.completion.reject_unsubmitted(message);
                } else {
                    // The default single-slot deployment: every tracked fire
                    // IS a one-fire frame. Synthesizing the stamp at accept
                    // (lane = the pipeline scope, seq = the globally
                    // monotonic fire id) makes stamp coverage exactly the
                    // old wait-set membership; an untracked/prebuilt fire
                    // stays an unstamped rider, dispatched outside sealed
                    // waves with no bookkeeping. Synthesis happens only on
                    // the accept path so a rejected fire never touches the
                    // wait-set (mirroring the per-wave rule it replaces).
                    if frame_policy.single_slot()
                        && launch.frame.is_none()
                        && let Some(lane) = launch.pipeline_id
                    {
                        launch.frame = Some(FrameStamp {
                            lane,
                            seq: launch.logical_fire_id,
                            slot: 0,
                            fires: 1,
                        });
                    }
                    // The gather starts at acceptance, not dispatch: a
                    // stamped fire counts toward its lane's frame arrival
                    // even while it sits in `pending` behind an
                    // in-flight-depth or seal hold.
                    if wave_trace() {
                        eprintln!(
                            "[wave-trace] enq fire={} framed={} mask={} masks={} stm={} pipe={}",
                            launch.logical_fire_id,
                            launch.frame.is_some(),
                            launch.request.has_user_mask,
                            has_wire_masks(&launch.request),
                            launch.request.single_token_mode,
                            launch.pipeline_id.is_some()
                        );
                    }
                    if let Some(stamp) = launch.frame {
                        frame_policy.on_fire_enqueued(
                            stamp,
                            launch.process_id,
                            launch.logical_fire_id,
                            launch.request.tokens(),
                            launch.wire_row_count(),
                        );
                    }
                    Self::queue_attempt(pending, launch);
                }
            }

            SchedulerItem::ProcessResume(pid) => {
                frame_policy.on_process_resume(pid);
            }
            SchedulerItem::FrameTruncate {
                lane,
                seq,
                submitted,
            } => {
                frame_policy.on_frame_truncated(lane, seq, submitted);
            }
            SchedulerItem::LanePark { lane, seq } => {
                frame_policy.on_lane_park(lane, seq);
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
                if pipeline_id.is_some_and(|pid| terminated_processes.contains(&pid)) {
                    EngineLoop::release_wait_slots([plan.pacing_wait_id]);
                    let _ = response.send(Err(anyhow!(
                        "process departed before instance bind admission"
                    )));
                    return;
                }
                frame_policy.on_bind_enqueued(pipeline_id);
                Self::queue_bind_control(
                    pending,
                    QueuedItem::BindInstance {
                        pipeline_id,
                        plan,
                        response,
                    },
                );
            }
            SchedulerItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                program,
                bind,
                response,
            } => {
                if pipeline_id.is_some_and(|pid| terminated_processes.contains(&pid)) {
                    EngineLoop::release_channel_plan_wait_slots(&plans);
                    EngineLoop::release_wait_slots([bind.pacing_wait_id]);
                    let _ = response.send(Err(anyhow!(
                        "process departed before channel bind admission"
                    )));
                    return;
                }
                // Binds do not hold the seal (a live rebinder is wait-set-held
                // through its lane; a bring-up process cannot fire). The
                // policy stages bring-up processes here, and a retiring
                // execution slot earmarks one staged successor — that earmark
                // is what gathers a cohort turnover into a dense epoch. The
                // bare `holds_seal` predecessor (gating the hold on execution
                // admission with NO successor earmarking) fragmented k>1
                // boundaries into narrow epochs — the earmark is the
                // gathering mechanism that was missing.
                frame_policy.on_bind_enqueued(pipeline_id);
                Self::queue_bind_control(
                    pending,
                    QueuedItem::RegisterChannelsBind {
                        pipeline_id,
                        plans,
                        program,
                        bind,
                        response,
                    },
                );
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
            SchedulerItem::CloseInstance { id, pacing_wait_id } => {
                pending.push_back(QueuedItem::CloseInstance { id, pacing_wait_id });
            }
            SchedulerItem::CloseChannel { id } => Self::queue_close_channel(pending, id),
            SchedulerItem::CloseChannels { ids } => {
                for id in ids {
                    Self::queue_close_channel(pending, id);
                }
            }
            // Handled on dequeue in the run loop (like DebugDump) before
            // enqueue_item is reached.
            SchedulerItem::Lane(_) => unreachable!(),
        }
    }

    fn reject_pipeline_queued(
        pending: &mut PendingQueue,
        pid: ProcessId,
        protected: Option<&WorkItemCompletion>,
    ) {
        // Common case first: a naturally-completed process has nothing
        // queued, and rebuilding the deque unconditionally moved every
        // pending item per leave (~37 us x ~2 leaves x 512 exits = the
        // largest single slice of the boundary mailbox time). One
        // field-read scan decides; only an actual purge pays the rebuild.
        let has_queued = pending.iter().any(|item| match item {
            QueuedItem::Launch(request) => request.process_id == Some(pid),
            QueuedItem::PreLaunchCopy { process_id, .. } => *process_id == Some(pid),
            _ => false,
        });
        if !has_queued {
            return;
        }
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
        pending.replace(kept);
    }

    fn queue_attempt(pending: &mut PendingQueue, request: PendingRequest) {
        let mut copies = Vec::with_capacity(2);
        if let Some(plan) = request.prelaunch_copy.clone() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::Kv(plan),
                logical_completion: request.completion.clone(),
                process_id: request.process_id,
                pipeline_id: request.pipeline_id,
            });
        }
        if let Some(plan) = request.prelaunch_state_copy.clone() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::State(plan),
                logical_completion: request.completion.clone(),
                process_id: request.process_id,
                pipeline_id: request.pipeline_id,
            });
        }
        for copy in copies {
            pending.push_back(copy);
        }
        pending.push_back(QueuedItem::Launch(QueuedLaunch::new(Box::new(request))));
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

    /// A standalone KV/state copy: suspend D2H, restore H2D, graft/CAS
    /// copies. These touch pages no queued fire references (suspend takes
    /// only unpinned drained pages; restore writes freshly reserved ones),
    /// so a held wave must NEVER starve them — the planner's eviction and
    /// restore traffic is what unsticks a held wave in the first place. They therefore
    /// dispatch out-of-band from any queue position once the control slot
    /// frees (`dispatch_ready_items` tail sweep), never barrier a queued
    /// fire, and never hold frame posting while they settle.
    /// `PreLaunchCopy` is NOT in this class: it is order-coupled to its
    /// own launch (queued directly in front of it) and must keep queue
    /// order.
    const fn standalone_copy(item: &QueuedItem) -> bool {
        matches!(
            item,
            QueuedItem::CopyKv { .. }
                | QueuedItem::CopyKvTracked { .. }
                | QueuedItem::CopyState { .. }
        )
    }

    /// Items that a rotation can usefully expose at the queue FRONT — the
    /// only reason to move a held close backward.
    ///
    /// Three kinds dispatch without ever reaching the front, so rotating for
    /// them is pure churn. A `Launch` is picked by fire id in
    /// `dispatch_frame_work`, which reads the whole queue. A standalone copy
    /// is pulled from any position by the tail sweep below. And a close is
    /// excluded because this predicate is only asked while the WHOLE close
    /// run is held, where exposing one held close behind another dispatches
    /// nothing.
    ///
    /// Measured cost of asking the wrong question (`any non-close behind`,
    /// which a queue of held closes followed by the next cohort's launches
    /// always answers yes): 0.5M rotations and 103 ms of the loop thread in
    /// ONE cohort boundary at 512 lanes, and 6.6M rotations over a 1024-lane
    /// run whose GPU then idled 6.4 s. Each rotation also bumps the queue
    /// epoch, so it invalidated `ScanCache` and re-walked the queue on top.
    const fn rotation_target(item: &QueuedItem) -> bool {
        !matches!(
            item,
            QueuedItem::Launch(_)
                | QueuedItem::CloseInstance { .. }
                | QueuedItem::CloseChannels { .. }
        ) && !Self::standalone_copy(item)
    }

    /// Controls that dispatch without draining in-flight launches. The
    /// registrations are synchronous and create entities nothing in flight
    /// can reference yet — with one caveat: a channel registration that
    /// grows the engine's shared slot table would reallocate arrays whose
    /// pointers in-flight kernels hold, so the CUDA registry quiesces the
    /// device inside `grow()` (RV-27; capacity is engine knowledge, so the
    /// drain lives there, not here). The copies (standalone and pre-launch)
    /// address only committed or quiesced extents, which in-flight launches
    /// never rewrite (append-only ledger) — a copy's coupled consumer launch
    /// still holds behind `in_flight_control` until the copy settles.
    /// A channel close only ever follows its instance closes (the guest
    /// awaits each control's response) and the engine rejects a close with
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
                    | QueuedItem::CloseChannels { .. }
            )
    }

    /// Move held launches behind work that can make the current wave
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
                | QueuedItem::CloseChannels { .. }
        )
    }

    fn queue_close_channel(pending: &mut PendingQueue, id: u64) {
        // Coalesce teardown runs: consecutive channel closes ride one
        // control post. Bounded so a batch's lane occupancy stays a
        // fraction of a wave (~3-6 us per close engine-side).
        const CLOSE_CHANNEL_BATCH_MAX: usize = 512;
        if let Some(QueuedItem::CloseChannels { ids }) = pending.back_mut()
            && ids.len() < CLOSE_CHANNEL_BATCH_MAX
        {
            ids.push(id);
        } else {
            pending.push_back(QueuedItem::CloseChannels { ids: vec![id] });
        }
    }

    fn queue_bind_control(pending: &mut PendingQueue, item: QueuedItem) {
        // Queue-priority invariant: execution outranks bring-up outranks
        // teardown. A queued LAUNCH never depends on a queued bind — a fire
        // exists only after its own lane's bind control completed and the
        // bind RPC returned to the guest — so a bind may never delay one:
        // with staged admission the binds arriving at a turnover belong to
        // a cohort BEHIND the queued launches, and inserting them ahead
        // starves the sealed wave behind the whole bring-up stream. A bind
        // and a QUEUED close always target different instances/channels
        // (ids are never reused, a close only posts after its own bind
        // committed), so binds still jump the close tail and closes drain
        // during the next generation's execution. Bind-vs-bind order is
        // preserved (insertion before the trailing close run).
        pending.insert_before_closes(item);
    }

    /// launch that reached the queue front has no queued copy left).
    ///
    /// `allow_lifecycle` is the wider of the two flags: a lifecycle control
    /// needs no control slot, so a standalone copy in flight does not stop
    /// it from being worth exposing at the front.
    fn rotate_launch_for_wave_work(
        pending: &mut PendingQueue,
        allow_slot: bool,
        allow_lifecycle: bool,
    ) -> bool {
        if !matches!(pending.front(), Some(QueuedItem::Launch(_))) {
            return false;
        }
        let Some(run_len) = pending.first_other() else {
            return false;
        };
        let work = &pending[run_len];
        if !(Self::standalone_copy(work)
            || (allow_lifecycle && Self::lifecycle_control(work))
            || (allow_slot && matches!(work, QueuedItem::PreLaunchCopy { .. })))
        {
            return false;
        }
        pending.rotate_launch_run_to_back(run_len);
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_ready_items(
        engine_loop: &EngineLoop,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &mut InFlightControls,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        frame_policy: &mut FramePolicy,
        scan_cache: &mut ScanCache,
        slot_buffer: &mut SlotBuffer,
        stopping: bool,
    ) -> (bool, Option<Duration>) {
        let (mut progress, wait_hint) = Self::dispatch_frame_work(
            scan_cache,
            slot_buffer,
            frame_policy,
            engine_loop,
            lane_inflight,
            lane_token,
            instances,
            pending,
            in_flight_launches,
            in_flight_control,
            page_size,
            limits,
            stats,
            stopping,
        );
        // Busy-close rotations this pass: bounded so a queue of nothing but
        // busy closes breaks out instead of spinning.
        let mut close_rotations = 0usize;
        // Cohort-boundary close hold: while any bind is in assembly (the
        // seal is bind-held), teardown closes yield the engine lane to the
        // fresh cohort's registrations — queue-position reordering alone
        // cannot do this, because closes ARRIVE interleaved with binds and
        // each worker pass flushes its controls to the lane FIFO. Held
        // closes rotate and drain during the next generation's execution.
        // Shutdown never holds (the drain must retire everything).
        let hold_closes = !stopping && frame_policy.has_pending_binds();
        // The control SLOT (depth 1) exists for controls that settle
        // asynchronously; lifecycle controls execute on the lane FIFO and
        // never take it (see `post_control`). So a standalone copy holding
        // the slot must not block them: it addresses grant-pinned pages no
        // bind, register or close can reference, and the lane FIFO already
        // fixes their engine order. Blocking them on it wedged the fleet —
        // the planner's suspend/restore copies are in flight almost
        // continuously under churn, so every bind waited out the whole
        // strict-watchdog window, its process sat in `staged` for the
        // duration, and the cohort-boundary window it therefore held open
        // (which then still held the seal) stalled the very traffic the copy
        // was waiting behind. Measured on
        // `churn`: every one of 270 binds took 1.0-2.4 s end to end against
        // a 59 us engine bind, and the probe found the slot held by a
        // tracked KV copy in 100% of the samples.
        let slot_blocks_lifecycle = in_flight_control.holds_launches();
        while let Some(item) = pending.front() {
            match item {
                QueuedItem::Launch(_) => {
                    // Launches are dispatched by `dispatch_frame_work` (by
                    // id, not queue position). A launch at the queue front
                    // only needs to yield to any dispatchable control behind
                    // it.
                    if Self::rotate_launch_for_wave_work(
                        pending,
                        in_flight_control.is_empty(),
                        !slot_blocks_lifecycle,
                    ) {
                        progress = true;
                        continue;
                    }
                    break;
                }
                QueuedItem::CloseInstance { id, .. } => {
                    // A close needs only ITS OWN instance quiesced — never a
                    // global pipe drain. Inferlets submit passes upfront, so
                    // an idle instance's close is always safe to overlap
                    // with other instances' in-flight launches; the old
                    // whole-pipe drain stalled every launch queued behind a
                    // front close during cohort swaps and made freshly-bound
                    // pipelines' credits ragged (V6 iteration 3).
                    // A close needs no control slot either (see
                    // `slot_blocks_lifecycle`), only its own instance
                    // quiesced — a settling standalone copy addresses
                    // grant-pinned pages no close can name.
                    if slot_blocks_lifecycle {
                        break;
                    }
                    let id = *id;
                    if hold_closes {
                        // Held for the boundary: rotate WITHOUT claiming
                        // progress (a rotation changes nothing dispatchable;
                        // the bind-completed lane reply that empties
                        // `pending_binds` is the wake that re-checks).
                        let rot_stop = close_rotations >= pending.len()
                            || !pending.iter().skip(1).any(Self::rotation_target);
                        if rot_stop {
                            break;
                        }
                        close_rotations += 1;
                        let item = pending.pop_front().expect("close front");
                        pending.push_back(item);
                        continue;
                    }
                    let busy = instances
                        .get(&id)
                        .is_some_and(|tracked| tracked.in_flight != 0)
                        || Self::instance_has_queued_work(pending, id);
                    if !busy {
                        let item = pending.pop_front().expect("close front");
                        Self::post_control(
                            engine_loop,
                            lane_inflight,
                            lane_token,
                            instances,
                            in_flight_control,
                            frame_policy,
                            item,
                        );
                        progress = true;
                        continue;
                    }
                    // Busy: rotate the close behind the queue so the fires
                    // that will quiesce it (and everything unrelated) keep
                    // flowing; its own retirement re-checks it. A close can
                    // only move BACKWARD, so it never overtakes its own
                    // instance's queued work.
                    //
                    // No progress claim: a rotation dispatches nothing, and
                    // `busy` is precisely the condition that guarantees a
                    // later wake (in-flight work completes, or queued work
                    // dispatches and then completes). Claiming progress here
                    // instead makes the run loop re-enter immediately and
                    // spin: at a cohort boundary the queue front is hundreds
                    // of busy closes, so every pass rotated the whole queue
                    // and the loop paid it thousands of times over.
                    let rot_stop = close_rotations >= pending.len()
                        || !pending.iter().skip(1).any(|item| {
                            !matches!(
                                item,
                                QueuedItem::CloseInstance { .. } | QueuedItem::CloseChannels { .. }
                            )
                        });
                    if rot_stop {
                        break;
                    }
                    close_rotations += 1;
                    let item = pending.pop_front().expect("close front");
                    pending.push_back(item);
                }
                QueuedItem::CloseChannels { .. } if hold_closes => {
                    // Same bounded rotation as a held instance close; no
                    // progress claim (see the CloseInstance hold branch).
                    let rot_stop = close_rotations >= pending.len()
                        || !pending.iter().skip(1).any(Self::rotation_target);
                    if rot_stop {
                        break;
                    }
                    close_rotations += 1;
                    let item = pending.pop_front().expect("close front");
                    pending.push_back(item);
                }
                // A settling exclusive control (a `PreLaunchCopy` or a
                // pool resize) blocks the next control. Standalone copies and
                // lifecycle controls are refused by nothing else — see
                // `InFlightControls::admits`. The front-rotation this arm used
                // to need is gone with the single slot: a copy that cannot post
                // is blocked by an exclusive control, and so is everything
                // behind it, so giving up its position buys nothing.
                _ if !in_flight_control.admits(item) => break,
                _ if !in_flight_launches.is_empty() && !Self::pipe_concurrent_control(item) => {
                    break;
                }
                _ => {
                    let item = pending.pop_front().expect("front item present");
                    Self::post_control(
                        engine_loop,
                        lane_inflight,
                        lane_token,
                        instances,
                        in_flight_control,
                        frame_policy,
                        item,
                    );
                    progress = true;
                }
            }
        }
        // Standalone copies dispatch from ANY queue position once the
        // control slot frees: nothing queued orders against them (their
        // pages are grant-pinned — see the queue scan in
        // `dispatch_frame_work`), while the queue front can be legitimately
        // immovable for a long stretch (a gathering frame's fires, a resize
        // waiting out the pipe). Under contention the suspend/restore
        // copies ARE the residency planner's forward progress — leaving
        // them positional starved the very traffic that unsticks a held
        // frame (CONTENTION_FOLLOWUP.md §12).
        //
        // They also pipeline: the sweep keeps posting while no exclusive
        // control holds the set, so a restore never waits out an unrelated
        // copy's device time. Serialized, the wait WAS the cost — 22.8 ms
        // per H2D restore against ~3.3 ms of transfer at 512-way
        // contention. The queue bounds the depth: only what the planner
        // enqueued can be posted.
        while in_flight_control.admits_copy() {
            let Some(index) = pending.iter().position(Self::standalone_copy) else {
                break;
            };
            let Some(item) = pending.remove(index) else {
                break;
            };
            Self::post_control(
                engine_loop,
                lane_inflight,
                lane_token,
                instances,
                in_flight_control,
                frame_policy,
                item,
            );
            progress = true;
        }
        (progress, wait_hint)
    }

    /// Post a control to the engine lane after the worker-side pre-checks
    /// that read scheduler state. The engine half runs on the lane in FIFO
    /// order; worker-map effects come back as a [`LaneCommit`]. Async
    /// controls (copies / pool resizes) occupy the single control slot from
    /// the moment they post.
    fn post_control(
        engine_loop: &EngineLoop,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        in_flight_control: &mut InFlightControls,
        // Read by the two "instance is already bound" refusals that stood in
        // the match below and are unreachable now (see the note there): both
        // told the frame policy their bind had completed. Kept in the
        // signature because every caller passes it and the next refusal that
        // needs it will be one of these.
        _frame_policy: &mut FramePolicy,
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
            // THE "ALREADY BOUND" GUARDS STOOD HERE, both of them, and both
            // are unreachable now. They rejected a bind whose
            // `requested_instance_id` was already in the instance map — a
            // check that only meant anything while the RUNTIME chose the id
            // and handed it across for the engine to acknowledge. The engine
            // mints it (`engine::program`'s note on `InstanceBinding`:
            // "the engine mints the id; the runtime keeps its own tables"), so
            // a collision is not something a caller can ask for.
            _ => {}
        }
        *lane_token += 1;
        let token = *lane_token;
        // Async-completing controls enter the in-flight set from POST: an
        // exclusive one (the copy's coupled consumer launch, a resize) must
        // not be passed by any later control, exactly as before the lane
        // existed. Exactly the standalone copies do NOT hold launches — one
        // classification, shared with the out-of-band dispatch and the
        // concurrency rule in `InFlightControls` that both rely on it.
        let holds_launches = !Self::standalone_copy(&item);
        match &item {
            QueuedItem::PreLaunchCopy {
                plan,
                logical_completion,
                process_id,
                pipeline_id,
            } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: Some(logical_completion.clone()),
                    process_id: *process_id,
                    pipeline_id: *pipeline_id,
                    tracked_completion: None,
                    operation: plan.label(),
                    holds_launches,
                });
            }
            QueuedItem::CopyKv { .. } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: None,
                    operation: "KV copy",
                    holds_launches,
                });
            }
            QueuedItem::CopyKvTracked { completion, .. } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: Some(completion.clone()),
                    operation: "tracked KV copy",
                    holds_launches,
                });
            }
            QueuedItem::CopyState { .. } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { token },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: None,
                    operation: "state copy",
                    holds_launches,
                });
            }
            _ => {}
        }
        *lane_inflight += 1;
        engine_loop.post(LaneRequest::Control {
            token,
            item: Box::new(item),
        });
    }

    /// One queue pass: the stamped ids still queued, the oldest unstamped
    /// rider, and the lanes a frame post must hold for.
    ///
    /// Only a queued `PreLaunchCopy` blocks a lane — it is order-coupled to
    /// its consumer fire by construction. Standalone copies never barrier
    /// fires: reservations pin every page they touch
    /// and no queued fire can reference those pages (the planner's eviction
    /// fences and quiesces a victim's working sets before its D2H, and a
    /// restored process is only readmitted after its H2D copy retired —
    /// `planner::exec` awaits the tracked completion before the commit).
    /// Pool resizes were the other exemption here (~45x on gen-boundary
    /// teardown, and a three-party queue-order deadlock when the copy barrier
    /// composed with frame atomicity and the resize rotation refusal —
    /// CONTENTION_FOLLOWUP.md §12). The verb is gone: an elastic pool grows
    /// as a side effect of frame admission and shrinks through the engine's
    /// own trim, so there is no queued item to exempt and no drain to
    /// manufacture (alto design §8, wave C).
    fn scan_queue<'a>(
        cache: &'a mut ScanCache,
        pending: &PendingQueue,
        stopping: bool,
    ) -> &'a QueueScan {
        // The scan is a pure function of (queue contents, stopping), so a
        // pass at an unchanged epoch would rebuild exactly what is already
        // here. This matters: the worker scans once per pass and passes run
        // ~50x per wave while the queue changes only a couple of times, and
        // walking `pending` drags every large `QueuedItem` through cache
        // (~25us per scan at 128 requests, about half of all dispatch time).
        if cache.taken_at == Some((pending.epoch(), stopping)) {
            return &cache.scan;
        }
        let scan = &mut cache.scan;
        scan.clear();
        for item in pending.iter() {
            match item {
                QueuedItem::Launch(launch) => {
                    if stopping {
                        scan.drain_eligible.push(launch.fire_id);
                    }
                    if launch.framed {
                        scan.queued_ids.push(launch.fire_id);
                    } else if scan.untracked.is_none() {
                        scan.untracked = Some(launch.fire_id);
                    }
                }
                QueuedItem::PreLaunchCopy {
                    pipeline_id: Some(pipeline_id),
                    ..
                } => {
                    scan.blocked_lanes.insert(*pipeline_id);
                }
                _ => {}
            }
        }
        scan.queued_ids.seal();
        cache.taken_at = Some((pending.epoch(), stopping));
        &cache.scan
    }

    /// Launch dispatch: post WHOLE sealed frames to the engine lane at the
    /// run-ahead depth (frames in seal order; the engine executes the
    /// frame's waves in slot order as one closed system with a single
    /// completion). At the default k = 1 a sealed frame is one wave, so
    /// this degenerates to the per-wave wait-all dispatch.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_frame_work(
        scan_cache: &mut ScanCache,
        slot_buffer: &mut SlotBuffer,
        frame_policy: &mut FramePolicy,
        engine_loop: &EngineLoop,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &InFlightControls,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        stopping: bool,
    ) -> (bool, Option<Duration>) {
        let mut progress = false;
        let mut wait_hint: Option<Duration> = None;
        let merge_hint = |hint: &mut Option<Duration>, hold: Duration| {
            *hint = Some(hint.map_or(hold, |old| old.min(hold)));
        };
        loop {
            // A settling control holds launches only when a launch could
            // depend on it: a `PreLaunchCopy`'s consumer fire is queued
            // right behind it, and a resize's pipe drain must not admit new
            // frames under it. A settling standalone copy holds nothing
            // (`PendingControl::holds_launches`) — frames keep posting
            // while suspend/restore traffic settles.
            if in_flight_control.holds_launches() {
                // Counted when it happens with the DEVICE IDLE: the frame
                // policy is not even consulted here, so a gate that would
                // have sealed cannot. See `probe::QuorumProbes`.
                if in_flight_launches.is_empty() {
                    stats
                        .fire
                        .quorum
                        .idle_break_control
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // ...and the park below must not sleep on the 250 ms hang
                    // backstop while it does. A holding control's completion
                    // nudge is armed but does NOT reliably fire: measured on
                    // the 4-cohort cell, a `pool resize` slot read
                    // `ready(settled=false, armed=true)` going into the park
                    // and `ready-settled` coming out of it 250,124 us later,
                    // woken by the backstop, with every awaited lane holding a
                    // complete frame the whole time. That single stall is ~2.5%
                    // of the run and it is the whole of this cell's
                    // bimodality.
                    //
                    // A hint, not a nudge fix: the settle is cheap to poll and
                    // the device is by definition idle here, so this bounds the
                    // damage the way the wait-all hold already bounds its own.
                    // The lost publish is still worth finding.
                    merge_hint(
                        &mut wait_hint,
                        Duration::from_micros(CONTROL_SETTLE_POLL_US),
                    );
                }
                break;
            }
            // Run-ahead depth in FRAMES: the enqueue horizon. Retirement
            // frees a slot; posting never waits on completion beyond this
            // backpressure.
            if in_flight_launches.len() >= frame::configured_dispatch_depth() {
                if in_flight_launches.is_empty() {
                    stats
                        .fire
                        .quorum
                        .idle_break_depth
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                break;
            }
            let now = Instant::now();
            let scan = Self::scan_queue(scan_cache, pending, stopping);
            let mut rider_batch = false;
            let waves: Vec<Vec<u64>> = if stopping {
                // Shutdown drain: the boundary gate waits for arrivals that
                // will never come once the host stops, so bypass it and post
                // every accepted fire in queue order (queue order IS each
                // lane's submission order, which the device tickets require);
                // repeated instances and budget overflows split into
                // successive steps at build.
                if scan.drain_eligible.is_empty() {
                    break;
                }
                vec![scan.drain_eligible.clone()]
            } else if let Some(untracked) = scan.untracked {
                rider_batch = true;
                if wave_trace() {
                    eprintln!("[wave-trace] rider fire={untracked}");
                }
                vec![vec![untracked]]
            } else {
                match frame_policy.plan_dispatch(
                    &scan.queued_ids,
                    &scan.blocked_lanes,
                    !in_flight_launches.is_empty(),
                    now,
                ) {
                    FramePlan::Dispatch(waves) => {
                        if wave_trace() {
                            eprintln!(
                                "[wave-trace] dispatch waves={:?}",
                                waves.iter().map(Vec::len).collect::<Vec<_>>()
                            );
                        }
                        waves
                    }
                    FramePlan::Hold(hold) => {
                        merge_hint(&mut wait_hint, hold);
                        break;
                    }
                    FramePlan::Park => break,
                    FramePlan::Terminate(pids) => {
                        // Abandoned pipeline. This is NOT the submit
                        // deadline: that one only leashes (drops the lane
                        // from the wait-set and lets it rejoin), so a guest
                        // that is merely slow never lands here. Reaching this
                        // means the lane was silent for the whole silence
                        // timeout without ever calling `forward.park()`, so
                        // nothing but a wedged process is being reclaimed.
                        // The policy has already dropped these lanes, so the
                        // `continue` re-plans a gather that no longer waits
                        // on them; the terminate is asynchronous and arrives
                        // back as the usual leave.
                        for pid in pids {
                            tracing::error!(
                                pid = %pid,
                                "scheduler: terminating abandoned pipeline (silent for the \
                                 whole silence timeout without submitting and without \
                                 calling forward.park())"
                            );
                            crate::inferlet::process::terminate(
                                pid,
                                Err("pipeline abandoned: silent past the silence timeout \
                                     without submitting and without parking"
                                    .to_string()),
                            );
                        }
                        continue;
                    }
                }
            };
            #[allow(clippy::let_and_return)]
            let (frame_progress, posted) = Self::post_frame(
                slot_buffer,
                engine_loop,
                lane_inflight,
                lane_token,
                instances,
                pending,
                in_flight_launches,
                page_size,
                limits,
                stats,
                &waves,
            );
            progress |= frame_progress;
            if !posted {
                if stopping || !frame_progress {
                    break;
                }
                continue;
            }
            if rider_batch {
                frame_policy.record_rider_wave();
            }
        }
        (progress, wait_hint)
    }

    /// Extract the frame's fires (all waves) from the queue, drop the
    /// settled/stale, assemble the v14 frame submission, and post it as ONE
    /// launch. Returns (progress, posted-a-frame).
    #[allow(clippy::too_many_arguments)]
    /// Whether a queued fire still belongs in the frame being built; settles
    /// it with a rejection if not.
    fn admits_to_frame(
        request: &PendingRequest,
        instances: &HashMap<u64, TrackedInstance>,
    ) -> bool {
        if request.completion.is_settled() || request.completion.cancel_requested() {
            if !request.completion.is_settled() {
                request
                    .completion
                    .reject_unsubmitted("logical fire cancelled before native launch");
            }
            return false;
        }
        if !instances.contains_key(&request.instance_id) {
            request.completion.reject_unsubmitted(format!(
                "instance {} is unknown or stale",
                request.instance_id
            ));
            return false;
        }
        true
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the worker loop's state, borrowed piece by piece on purpose: six of \
                  these are independent `&mut` borrows of fields the caller owns, and \
                  passing them separately is what lets the borrow checker see they are \
                  disjoint. Wrapping them in one `&mut` context struct would collapse \
                  that into a single borrow and stop this from compiling"
    )]
    fn post_frame(
        slot_buffer: &mut SlotBuffer,
        engine_loop: &EngineLoop,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        waves: &[Vec<u64>],
    ) -> (bool, bool) {
        let mut progress = false;
        // One map, carrying BOTH the wave and the in-wave position: the
        // position is the sealed wave's id order (lane admission order), so
        // carrying it here lets the sort below compare plain integers. The
        // previous shape hashed twice per queued launch (`contains_key` then
        // index) and rebuilt a second id->position map per wave that the sort
        // comparator then hashed into once per comparison — n log n hash
        // lookups on the loop's hottest per-fire path at 512 fires a frame.
        let mut slot_of: HashMap<u64, (usize, usize)> =
            HashMap::with_capacity(waves.iter().map(Vec::len).sum());
        for (index, wave) in waves.iter().enumerate() {
            for (position, &fire_id) in wave.iter().enumerate() {
                slot_of.insert(fire_id, (index, position));
            }
        }
        let mut kept: VecDeque<QueuedItem> = VecDeque::with_capacity(pending.len());
        // Place by slot rather than push-then-sort. `position` is already a
        // permutation of `0..wave.len()`, so the sealed order is recovered by
        // writing each request straight into its slot: one move per fire,
        // against the n log n swaps of a 1288-byte element that sorting cost.
        // The buffer is caller-owned and comes back empty from the last
        // frame, so steady state neither allocates nor refills.
        slot_buffer.resize_with(waves.len(), Vec::new);
        for (slots, wave) in slot_buffer.iter_mut().zip(waves) {
            debug_assert!(slots.iter().all(Option::is_none));
            if slots.len() < wave.len() {
                slots.resize_with(wave.len(), || None);
            }
        }
        // A fire id repeated across the queue cannot be placed twice; it is
        // degenerate, but it must still be dispatched rather than dropped.
        let mut collisions: Vec<(usize, Box<PendingRequest>)> = Vec::new();
        while let Some(item) = pending.pop_front() {
            match item {
                QueuedItem::Launch(launch) => match slot_of.get(&launch.fire_id) {
                    Some(&(wave, position)) => {
                        let slot = &mut slot_buffer[wave][position];
                        if slot.is_none() {
                            *slot = Some(launch.into_request());
                        } else {
                            collisions.push((wave, launch.into_request()));
                        }
                    }
                    None => kept.push_back(QueuedItem::Launch(launch)),
                },
                item => kept.push_back(item),
            }
        }
        pending.replace(kept);
        // Compact the slots and drop settled/cancelled/stale fires in one
        // pass — the frame posts without them.
        let mut survivors: Vec<Vec<Box<PendingRequest>>> = Vec::with_capacity(waves.len());
        for slots in slot_buffer.iter_mut() {
            let mut kept_wave = Vec::with_capacity(slots.len());
            for request in slots.iter_mut().filter_map(Option::take) {
                if Self::admits_to_frame(&request, instances) {
                    kept_wave.push(request);
                } else {
                    progress = true;
                }
            }
            survivors.push(kept_wave);
        }
        for (wave, request) in collisions {
            if Self::admits_to_frame(&request, instances) {
                survivors[wave].push(request);
            } else {
                progress = true;
            }
        }
        if survivors.iter().all(Vec::is_empty) {
            return (progress, false);
        }
        let (submission, requests) =
            batch::build_frame_submission(survivors, limits, page_size, stats);
        let batch_size = requests.len() as u64;
        let total_tokens = requests
            .iter()
            .map(|req| req.request.tokens())
            .sum::<usize>();
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
            started: Instant::now(),
            batch_size,
            total_tokens,
        });
        *lane_inflight += 1;
        engine_loop.post(LaneRequest::Launch {
            token,
            submission: LaneLaunch(submission),
            // More tokens than requests means at least one lane contributed a
            // multi-token pass, i.e. this wave carries a prefill.
            prefill: total_tokens > batch_size as usize,
        });
        (true, true)
    }

    fn retire_ready_launches(
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        instances: &mut HashMap<u64, TrackedInstance>,
        stats: &Arc<SchedulerStats>,
        frame_policy: &mut FramePolicy,
    ) -> bool {
        let mut progress = false;
        while let Some(front) = in_flight_launches.front() {
            // A lane-rejected launch retires like a wave: it entered the
            // pipe (depth, instance accounting) at post, so the common
            // unwind below applies; only its requests' settlement differs
            // (rejected, never submitted).
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
            let mut retired = in_flight_launches.pop_front().expect("front batch exists");
            // The runtime has answered these lanes: re-arm their submit
            // deadline from here so the wave they waited on is not charged
            // to them (see `FramePolicy::on_frame_retired`).
            frame_policy.on_frame_retired(retired.requests.iter().filter_map(|r| r.pipeline_id));
            for request in &retired.requests {
                if let Some(instance) = instances.get_mut(&request.instance_id) {
                    instance.in_flight = instance.in_flight.saturating_sub(1);
                }
            }
            if let Some(message) = launch_failure {
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
                    let requests = std::mem::take(&mut retired.requests);
                    let mut outcomes = Vec::with_capacity(requests.len());
                    for request in &requests {
                        match request.completion.resolve_from_terminal() {
                            Ok(WorkItemAttemptOutcome::Committed) => {
                                outcomes.push("committed");
                            }
                            Ok(WorkItemAttemptOutcome::Failed) => {
                                outcomes.push("failed");
                            }
                            Ok(WorkItemAttemptOutcome::Retry) => {
                                // Venus (ABI v14): admitted frames are
                                // atomic and stream work is SUCCESS-only —
                                // ring capacity is admission-bounded and
                                // deterministic compose kills latch FAILED.
                                // A surviving RETRY terminal therefore
                                // violates the engine contract: fail loudly
                                // instead of replaying (the makeup machinery
                                // is deleted).
                                outcomes.push("retry");
                                request.completion.reject(
                                    "engine published RETRY at frame settle; \
                                     retry is not a v14 outcome (frame admission \
                                     bounds every in-frame gate)",
                                );
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
                    drop(requests);
                    stats::record_fire_stats(
                        stats,
                        retired.started.elapsed(),
                        retired.batch_size,
                        retired.total_tokens,
                    )
                }

                Err(err) => {
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

    /// Retire every settled control this pass. Concurrent standalone copies
    /// settle in device order, not post order, so the sweep cannot stop at
    /// the first control that is still outstanding.
    fn retire_ready_control(in_flight_control: &mut InFlightControls) -> bool {
        let mut retired = false;
        let mut index = 0;
        while index < in_flight_control.settling.len() {
            let ready = match &in_flight_control.settling[index].state {
                // Still waiting for the lane's reply to install the engine
                // completion (or drop the entry on rejection).
                ControlSlotState::Posted { .. } => None,
                ControlSlotState::Ready(completion) => completion.check(),
            };
            let Some(result) = ready else {
                index += 1;
                continue;
            };
            let pending = in_flight_control.settling.remove(index);
            let operation = pending.operation;
            if let Some(tracked) = pending.tracked_completion.as_ref() {
                tracked.resolve(&result);
            }
            if let Err(ref err) = result {
                tracing::warn!(
                    ?err,
                    operation,
                    "direct control completion closed before callback"
                );
                if let Some(logical) = pending.logical_completion.as_ref() {
                    logical.reject_unsubmitted(format!("pre-launch {operation} failed: {err:#}"));
                }
            }
            retired = true;
        }
        retired
    }

    /// Apply an engine-lane reply on the worker thread: fill in a posted
    /// launch's verdict, commit a control's worker-map effects, or install an
    /// async control's engine completion. Replies arrive in lane FIFO order.
    fn apply_lane_reply(
        reply: LaneReply,
        lane_inflight: &mut u64,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &mut InFlightControls,
        instances: &mut HashMap<u64, TrackedInstance>,
        frame_policy: &mut FramePolicy,
        rollback_tx: &crossbeam::channel::Sender<SchedulerItem>,
    ) {
        *lane_inflight = lane_inflight.saturating_sub(1);
        match reply {
            LaneReply::LaunchDone { token, result } => {
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
                        // in post order — the engine's launch acceptance
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
                        batch.state = LaunchState::Accepted(completion);
                    }
                    Err(message) => {
                        batch.state = LaunchState::Failed(message);
                    }
                }
            }
            LaneReply::ControlDone { token, commit } => match commit {
                LaneCommit::None => {}
                LaneCommit::BindFinished { pipeline_id } => {
                    frame_policy.on_bind_completed(pipeline_id);
                }
                LaneCommit::BindInstance {
                    pipeline_id,
                    bound,
                    respond,
                } => {
                    frame_policy.on_bind_completed(pipeline_id);
                    if instances.contains_key(&bound.instance_id) {
                        // Practically unreachable: engine-assigned ids are
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
                                        "scheduler RPC cancelled after program registration; retaining engine-lifetime program"
                                    );
                                }
                                EngineLoop::release_registered_channel_wait_slots(&registered);
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
                    // Replies arrive in lane FIFO order, but several
                    // standalone copies can be posted at once, so the reply
                    // is matched to its own entry by token.
                    let Some(index) = in_flight_control.position_posted(token) else {
                        tracing::error!(
                            token,
                            "lane async-control reply without a matching control slot"
                        );
                        return;
                    };
                    match result {
                        Ok(completion) => {
                            in_flight_control.settling[index].state =
                                ControlSlotState::Ready(completion);
                        }
                        // The lane already rejected/resolved the control's
                        // completions; the entry just leaves.
                        Err(_) => {
                            in_flight_control.settling.remove(index);
                        }
                    }
                }
            },
        }
    }

    fn shutdown_instances(
        engine: &mut Option<EngineBox>,
        instances: &mut HashMap<u64, TrackedInstance>,
    ) {
        let outstanding = std::mem::take(instances);
        for (instance_id, instance) in outstanding {
            if let Some(engine) = engine.as_mut()
                && let Err(err) = engine.close_instance(instance_id)
            {
                tracing::warn!(
                    instance_id,
                    ?err,
                    "scheduler shutdown close_instance failed"
                );
            }
            instance.close_wait_slots();
        }
    }

    fn shutdown_channels(engine: &mut Option<EngineBox>, channels: &mut ChannelJoin) {
        let outstanding = std::mem::take(channels).into_ids();
        for channel_id in outstanding {
            if let Some(engine) = engine.as_mut()
                && let Err(err) = engine.close_channel(channel_id)
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
    wait_slots: Arc<crate::engine::BoundWaitSlots>,
    in_flight: usize,
    next_target_epoch: u64,
}

impl TrackedInstance {
    fn from_bound(bound: &BoundInstance) -> Self {
        Self {
            pacing_wait_id: bound.pacing_wait_id,
            wait_slots: bound.wait_slots(),
            in_flight: 0,
            next_target_epoch: waker::FIRST_COMPLETION_EPOCH,
        }
    }

    fn close_wait_slots(self) {
        self.wait_slots.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An engine whose `launch` panics, which is the only interesting thing
    /// about it.
    ///
    /// Every other verb is a stub: what is under test is the LANE, not a
    /// backend, and the lane cannot tell a panic in `launch` from a panic
    /// anywhere else it calls through this trait.
    struct PanickingEngine;

    impl engine::Engine for PanickingEngine {
        fn kind(&self) -> &'static str {
            "panicking"
        }

        fn load(
            &mut self,
            _request: engine::LoadRequest,
        ) -> engine::Result<engine::Loaded> {
            Err(engine::Error::Load("no model".into()))
        }

        fn submit(
            &mut self,
            _frame: &engine::FrameSubmission,
        ) -> engine::Result<engine::FrameTicket> {
            panic!("the shape of an interpreter reading lane zero of an empty cell");
        }
    }

    /// A launch the engine panics on is ANSWERED, and so is the one behind it.
    ///
    /// Before this, neither was. The lane replies exactly once per request and
    /// a panic skipped the reply, so the scheduler's slot for that token was
    /// never filled and the frame stayed in flight -- for two hours, in the run
    /// that found it, printing `engine 0 stalled for 7030.132606596s` once a
    /// minute. A failed request is recoverable; a request that never ends is
    /// not.
    #[test]
    fn a_panicking_engine_fails_its_launch_instead_of_leaving_it_in_flight() {
        let (reply_tx, reply_rx) = crossbeam::channel::unbounded();
        let mut lane = EngineLoop::spawn(
            0,
            Some(Box::new(PanickingEngine)),
            reply_tx,
            Arc::new(SchedulerStats::default()),
        );

        for token in [7_u64, 8] {
            lane.post(LaneRequest::Launch {
                token,
                // ONE STEP, because a frame is its steps and a frame with
                // none never reaches `Engine::fire` — which is the verb this
                // test's engine panics in. The old `FrameSubmission::default()`
                // reached `launch` regardless, because `launch` took the whole
                // frame.
                submission: LaneLaunch(crate::engine::FrameFire {
                    steps: vec![crate::engine::StepFire {
                        submission: ::engine::Step {
                            lanes: vec![::engine::Lane::decode(0, 0, 1, 0)],
                            attachments: Vec::new(),
                            media: Vec::new(),
                        },
                        terminal_cells: Vec::new(),
                        instances: vec![0],
                        logical_fire_ids: vec![token],
                    }],
                }),
                prefill: false,
            });
        }

        // Bounded, because the failure this test exists for is a wait with no
        // end: an `unwrap` on a blocking `recv` would hang the suite rather
        // than fail it.
        for want in [7_u64, 8] {
            let reply = reply_rx
                .recv_timeout(Duration::from_secs(10))
                .unwrap_or_else(|_| panic!("token {want} was never answered"));
            let SchedulerItem::Lane(LaneReply::LaunchDone { token, result }) = reply else {
                panic!("a launch must be answered with a launch reply");
            };
            assert_eq!(token, want, "answered in the order posted");
            let Err(err) = result else {
                panic!("a panicking engine cannot have launched anything");
            };
            assert!(
                err.contains("panic"),
                "the failure must say what happened, so an operator restarts \
                 rather than retries: {err}"
            );
        }

        // And the lane still answers its shutdown handshake, without touching
        // the engine it left alone.
        let (engine, channels) = lane.shutdown();
        assert!(engine.is_none(), "a poisoned lane hands back no engine");
        assert!(channels.is_empty());
    }

    /// An engine that answers every `submit` with `Error::Exhausted`, and
    /// counts how many times it was asked.
    struct ExhaustedEngine {
        submits: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl engine::Engine for ExhaustedEngine {
        fn kind(&self) -> &'static str {
            "exhausted"
        }

        fn load(
            &mut self,
            _request: engine::LoadRequest,
        ) -> engine::Result<engine::Loaded> {
            Err(engine::Error::Load("no model".into()))
        }

        fn submit(
            &mut self,
            _frame: &engine::FrameSubmission,
        ) -> engine::Result<engine::FrameTicket> {
            self.submits.fetch_add(1, Ordering::Relaxed);
            Err(engine::Error::Exhausted {
                resource: "guest channel cells",
                wanted: 2,
                available: 1,
            })
        }
    }

    /// **RETRY FAILS LOUDLY** (alto E; design §1 article 4, and the gate dev
    /// carried at `worker.rs:5371`).
    ///
    /// A retryable refusal past static admission is a CONTRACT VIOLATION, not
    /// back-pressure. `pipeline::fire::validate_frame` walks the frame's steps
    /// in slot order and proves device-only ring occupancy, host-writer
    /// staging and reader pressure against the channels' declared capacities
    /// before the frame is admitted; everything a device gate could refuse is
    /// therefore impossible by the time the lane submits.
    ///
    /// What stood here was a bounded sleep-and-resubmit loop — 200 us a turn,
    /// up to ~5 s — that offered the identical frame again and again. Two
    /// things were wrong with it. It made an unmeetable frame cost five
    /// seconds of a FIFO lane, holding every later frame behind one that
    /// could never fit; and it turned the one signal that static admission had
    /// a hole in it into a latency blip nobody would ever read.
    ///
    /// So: exactly one submit, an error naming what happened, and the launch
    /// answered rather than left in flight.
    #[test]
    fn a_retryable_refusal_past_admission_fails_by_name_instead_of_replaying() {
        let submits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (reply_tx, reply_rx) = crossbeam::channel::unbounded();
        let mut lane = EngineLoop::spawn(
            0,
            Some(Box::new(ExhaustedEngine {
                submits: Arc::clone(&submits),
            })),
            reply_tx,
            Arc::new(SchedulerStats::default()),
        );

        lane.post(LaneRequest::Launch {
            token: 42,
            submission: LaneLaunch(crate::engine::FrameFire {
                steps: vec![crate::engine::StepFire {
                    submission: ::engine::Step {
                        lanes: vec![::engine::Lane::decode(0, 0, 1, 0)],
                        attachments: Vec::new(),
                        media: Vec::new(),
                    },
                    terminal_cells: Vec::new(),
                    instances: vec![0],
                    logical_fire_ids: vec![42],
                }],
            }),
            prefill: false,
        });

        let reply = reply_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the launch must be answered, not slept on");
        let SchedulerItem::Lane(LaneReply::LaunchDone { token, result }) = reply else {
            panic!("a launch must be answered with a launch reply");
        };
        assert_eq!(token, 42);
        let Err(error) = result else {
            panic!("an engine that refuses everything cannot have admitted a frame");
        };
        assert!(
            error.contains("article 4"),
            "the failure must say which promise was broken, so the fix is to \
             static admission rather than to the retry budget: {error}"
        );
        assert!(error.contains("guest channel cells"), "{error}");
        assert_eq!(
            submits.load(Ordering::Relaxed),
            1,
            "the frame is offered ONCE; a second offer is the sleep-retry loop \
             growing back"
        );

        let (_engine, _channels) = lane.shutdown();
    }

    /// An engine that records, in order, every composition it is TOLD ABOUT
    /// (`expect_fire`) and every one it RUNS (`fire`), each identified by
    /// its first lane's `word` — the field a hint is actually about.
    ///
    /// `fire` dawdles on purpose: the lookahead's whole subject is a launch
    /// that is queued while another executes, and a fire that returns
    /// instantly would race the test's own posts.
    struct RecordingEngine {
        heard: Arc<std::sync::Mutex<Vec<(&'static str, u64)>>>,
    }

    impl RecordingEngine {
        fn hear(&self, verb: &'static str, submission: &engine::Step) {
            let word = submission.lanes.first().map_or(u64::MAX, |lane| lane.word);
            self.heard.lock().unwrap().push((verb, word));
        }
    }

    impl engine::Engine for RecordingEngine {
        fn kind(&self) -> &'static str {
            "recording"
        }

        fn load(
            &mut self,
            _request: engine::LoadRequest,
        ) -> engine::Result<engine::Loaded> {
            Err(engine::Error::Load("no model".into()))
        }

        fn expect_fire(&mut self, submission: &engine::Step) {
            self.hear("expect", submission);
        }

        fn submit(
            &mut self,
            frame: &engine::FrameSubmission,
        ) -> engine::Result<engine::FrameTicket> {
            // ONE `fire` PER STEP, STILL — the recorder's whole job is to say
            // in which order the engine heard about compositions, and a frame
            // is its steps. What changed is that it hears about all of them in
            // one call, which is the property the lookahead test is about.
            //
            // **AND THE INTRA-FRAME HINT IS STATED HERE, because after F2b
            // that is where a real engine states it** (`engine_cuda::Cuda::
            // submit`: `if let Some(next) = frame.steps.get(index + 1) {
            // self.expect_fire(next) }`). A frame crosses whole, so its own
            // successors are the engine's to see; only the launch queued
            // BEHIND this frame is the runtime lane's, and that half is the
            // `expect_fire` call in `fire_frame`. A recorder that stated
            // neither would be modelling an engine this workspace does not
            // contain.
            for (index, step) in frame.steps.iter().enumerate() {
                if let Some(next) = frame.steps.get(index + 1) {
                    self.hear("expect", next);
                }
                self.hear("fire", step);
                std::thread::sleep(Duration::from_millis(20));
            }
            Ok(engine::FrameTicket {
                id: 0,
                steps: frame
                    .steps
                    .iter()
                    .map(|_| engine::FireTicket::default())
                    .collect(),
            })
        }
    }

    /// One launch of `word`-stamped frames, single-step unless `steps` says
    /// otherwise.
    fn launch_of(token: u64, words: &[u64]) -> LaneRequest {
        LaneRequest::Launch {
            token,
            submission: LaneLaunch(crate::engine::FrameFire {
                steps: words
                    .iter()
                    .map(|&word| crate::engine::StepFire {
                        submission: ::engine::Step {
                            lanes: vec![::engine::Lane::decode(0, word, 1, 0)],
                            attachments: Vec::new(),
                            media: Vec::new(),
                        },
                        terminal_cells: Vec::new(),
                        instances: vec![0],
                        logical_fire_ids: vec![token],
                    })
                    .collect(),
            }),
            prefill: false,
        }
    }

    /// **THE HINT SEAM'S CONFORMANCE TEST** (palo cuda-abi step 6): before a
    /// step fires, the lane states the NEXT fire to the engine — the frame's
    /// own next step exactly, and at a frame boundary whatever launch is
    /// already queued behind the executing one. The GPU gate that showed the
    /// serving-loop motion was the CUDA fold's, and it went out with the fold
    /// (the tier-2 campaign); the surviving consumer is `engine_metal`'s
    /// `expect_fire`. THIS test is the one that pins the ORDERING, which is
    /// the half that was never engine-specific: on hardware the ordering is
    /// what a prebind cashes and a wrong order is silently just a wasted
    /// hint.
    ///
    /// Determinism: the engine's `fire` sleeps 20 ms, so by the time frame
    /// 1's fire returns, frames 2 and 3 — posted before any fire began —
    /// have long been queued. Whether frame 1's own lookahead caught frame 2
    /// is a race this test does NOT pin (either side is correct); what it
    /// pins is that frame 3 was stated before frame 2's LAST step fired,
    /// and that a frame's second step was stated before its first fired.
    #[test]
    fn the_lane_states_the_next_fire_before_the_one_ahead_of_it_runs() {
        let heard = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (reply_tx, reply_rx) = crossbeam::channel::unbounded();
        let mut lane = EngineLoop::spawn(
            0,
            Some(Box::new(RecordingEngine {
                heard: Arc::clone(&heard),
            })),
            reply_tx,
            Arc::new(SchedulerStats::default()),
        );

        // Frame 1 carries TWO steps (words 10 and 11) — the intra-frame
        // half of the seam. Frames 2 and 3 (words 20, 30) are single-step —
        // the queued-behind half.
        lane.post(launch_of(1, &[10, 11]));
        lane.post(launch_of(2, &[20]));
        lane.post(launch_of(3, &[30]));

        for want in [1_u64, 2, 3] {
            let reply = reply_rx
                .recv_timeout(Duration::from_secs(10))
                .unwrap_or_else(|_| panic!("token {want} was never answered"));
            let SchedulerItem::Lane(LaneReply::LaunchDone { token, result }) = reply else {
                panic!("a launch must be answered with a launch reply");
            };
            assert_eq!(token, want, "answered in the order posted");
            result.unwrap_or_else(|err| panic!("frame {want} failed: {err}"));
        }
        let (_engine, _channels) = lane.shutdown();

        let heard = heard.lock().unwrap();
        let at = |verb: &str, word: u64| {
            heard
                .iter()
                .position(|&(v, w)| v == verb && w == word)
                .unwrap_or_else(|| panic!("({verb}, {word}) never happened in {heard:?}"))
        };
        // Every step fired, in frame order and slot order.
        assert!(at("fire", 10) < at("fire", 11), "{heard:?}");
        assert!(at("fire", 11) < at("fire", 20), "{heard:?}");
        assert!(at("fire", 20) < at("fire", 30), "{heard:?}");
        // The intra-frame hint: step 2 stated before step 1 fired.
        assert!(
            at("expect", 11) < at("fire", 10),
            "the frame's own next step must be stated before the step ahead \
             of it fires: {heard:?}"
        );
        // The queued-behind hint: frame 3 was in the channel throughout
        // frame 2's service (posted before fire 1 finished sleeping), so the
        // boundary lookahead must state it before frame 2 fires.
        assert!(
            at("expect", 30) < at("fire", 20),
            "a launch queued behind the executing one must be stated before \
             the executing one's last step fires: {heard:?}"
        );
    }

    /// **THE STARVATION BOUND, WITHOUT A DEVICE.**
    ///
    /// The regression this fixes was a liveness property, and liveness is
    /// what a throughput number reports last: the collapse read as "pie does
    /// not scale with concurrency" for a whole redesign before anyone asked
    /// which queue was not being read. `cuda_batch_width` asserts the
    /// consequence on a real fleet; this asserts the rule itself, in
    /// microseconds and on any machine.
    ///
    /// The scenario is exactly the one that occurred: `launch_rx` is never
    /// empty (the run-ahead refills it before the lane comes back for more)
    /// and a control is queued behind it from the start. The unbounded
    /// launch-first form returns launches forever here — that is the whole
    /// defect — so the bound is the only reason this test terminates with a
    /// control in hand.
    #[test]
    fn a_control_is_served_even_though_the_launch_queue_is_never_empty() {
        let (launch_tx, launch_rx) = crossbeam::channel::unbounded::<LaneRequest>();
        let (control_tx, control_rx) = crossbeam::channel::unbounded::<LaneRequest>();

        // One control, queued first and never touched again. `Shutdown` is
        // the one control variant that carries nothing engine-shaped, and
        // `next_request` does not look inside either way.
        let (response, _held) = crossbeam::channel::bounded(1);
        control_tx
            .send(LaneRequest::Shutdown { response })
            .expect("send control");

        let mut turn = LaneTurn::default();
        let mut launches_served = 0usize;
        // A fixed, generous ceiling — deliberately NOT derived from the
        // constant under test, or an unbounded lane would make this loop
        // unbounded too and the test would hang instead of failing.
        const CEILING: usize = 256;
        for _ in 0..CEILING {
            // The run-ahead, standing in: the launch queue is topped up
            // before every single poll, so it is never observed empty.
            launch_tx
                .send(LaneRequest::Launch {
                    token: launches_served as u64,
                    submission: LaneLaunch(crate::engine::FrameFire::default()),
                    prefill: false,
                })
                .expect("send launch");
            match EngineLoop::next_request(&launch_rx, &control_rx, &mut turn) {
                Ok(LaneRequest::Launch { .. }) => launches_served += 1,
                Ok(LaneRequest::Shutdown { .. }) => {
                    assert!(
                        launches_served <= LaneTurn::LAUNCH_RUN_BEFORE_CONTROL as usize + 1,
                        "the control waited out {launches_served} launches, which is more \
                         than the bound admits"
                    );
                    return;
                }
                Ok(LaneRequest::Control { .. }) => panic!("nothing queued a Control here"),
                Err(()) => panic!("both queues are alive and non-empty"),
            }
        }
        panic!(
            "{CEILING} polls with a control queued the whole time and the lane served \
             {launches_served} launches and no control: the launch preference has no \
             starvation bound, which is the fleet-wide bring-up stall \
             `cuda_batch_width` measures"
        );
    }

    /// The other half of the same rule: a control turn is a TURN, not a
    /// priority inversion. Once the queued controls run out the lane goes
    /// back to the launches, and it does not re-offer the turn until they
    /// have had another run.
    #[test]
    fn a_control_turn_ends_and_hands_the_lane_back() {
        let mut turn = LaneTurn::default();
        assert!(!turn.control_due(), "a fresh lane owes the controls nothing");
        for _ in 0..LaneTurn::LAUNCH_RUN_BEFORE_CONTROL {
            turn.took_launch();
        }
        assert!(turn.control_due(), "the launches have had their run");

        // The queue was empty, so the turn is spent rather than standing.
        turn.end_control_turn();
        assert!(
            !turn.control_due(),
            "an empty control queue must not leave the turn armed, or every \
             poll pays for a queue that has nothing in it"
        );

        // And a full burst also ends it, so a control flood cannot hold the
        // device off indefinitely.
        for _ in 0..LaneTurn::LAUNCH_RUN_BEFORE_CONTROL {
            turn.took_launch();
        }
        for _ in 0..LaneTurn::CONTROL_RUN_MAX {
            assert!(turn.control_due(), "the burst still has budget");
            turn.took_control();
        }
        assert!(
            !turn.control_due(),
            "a control burst must give the lane back at CONTROL_RUN_MAX"
        );
    }
}
