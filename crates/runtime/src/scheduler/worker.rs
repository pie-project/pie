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
    /// The planner is evicting the process: its lanes stop being awaited, but
    /// already-submitted frames stay sealable and drain untracked. No purge.
    Suspend,
    /// A pipeline closed or dropped: its wait-set row releases immediately,
    /// while already-accepted requests continue untracked to settlement.
    Close,
}

/// Posts one pipeline-leave to every engine's scheduler thread so each
/// [`FramePolicy`] drops the leaver from its wait-set (fire-and-forget).
/// `id` is lane-keyed for Close, process-keyed for Suspend/Terminate.
fn post_pipeline_leave(id: ProcessId, owner: Option<ProcessId>, kind: LeaveKind) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::PipelineLeave(id, owner, kind, None));
    }
}

/// One lane leaves the wait-all quorum gracefully; accepted fires drain
/// untracked. `owner`, when known, also drops it from process-keyed
/// `staged`/`joins_in_flight` (needed if the process may block before its first fire).
pub(crate) fn notify_lane_close(scope: ProcessId, owner: Option<ProcessId>) {
    post_pipeline_leave(scope, owner, LeaveKind::Close);
}

/// Every lane `pid` owns leaves the wait-all quorum (planner suspend); the
/// submitted tail stays sealable and drains untracked. Process-keyed.
pub(crate) fn notify_process_suspend(pid: ProcessId) {
    post_pipeline_leave(pid, Some(pid), LeaveKind::Suspend);
}

/// `pid` is runnable again; undoes [`notify_process_suspend`].
/// Fire-and-forget: a missed resume just means the fleet stops waiting for it.
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

/// Posts `pid`'s Terminate leave to every engine and returns fences that
/// resolve once each scheduler processes it — split from
/// [`notify_process_terminate`]'s await so a retiring process can release
/// its execution seat synchronously while teardown awaits the fence.
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

/// Awaits fences from [`post_process_terminate_fenced`]: once resolved,
/// every engine has purged the pid's queued work, so pooled resources can be recycled.
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

/// `forward.park()`: the lane leaves the frame wait-set until it fires
/// again. Broadcast fire-and-forget, ordered by `seq`. Not routed through
/// the control path (depth 1): park releases a gather, and a park queued
/// behind that dispatch could never arrive.
pub(crate) fn notify_lane_park(pid: ProcessId, seq: u64) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::LanePark { lane: pid, seq });
    }
}

/// A retiring process released its capped execution permit (capped
/// deployments only). Broadcasts the retiree's identity so a staged
/// successor's bind can resolve the departure; caller posts Terminate first
/// so every engine sees leave-then-release.
pub(crate) fn notify_execution_slot_released(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ExecutionSlotReleased(pid));
    }
}

/// Deferred teardown finished: every event the process can produce is
/// already in each engine's mailbox. Retires the terminate tombstone,
/// bounding `terminated_processes` by live-plus-draining processes.
pub(crate) fn notify_process_quiesced(pid: ProcessId) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::ProcessQuiesced(pid));
    }
}

/// A parked process acquired its execution permit; its first fire is a
/// named join in flight, keeping the cohort-boundary window open until it
/// lands. Sent before the fire enters the mailbox so the policy sees
/// consume-then-fire (a reordered arrival is harmless).
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
/// submission.
#[allow(dead_code)] // no live caller — see doc.
pub(crate) fn notify_pipeline_join(_pid: ProcessId) {}

/// Completions the 250ms hang backstop found already settled (a lost
/// nudge). Should stay zero; any increment is a wake-path regression.
pub(crate) static BACKSTOP_RETIREMENTS: AtomicU64 = AtomicU64::new(0);
static NEXT_LOGICAL_FIRE_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct PendingRequest {
    pub(crate) logical_fire_id: u64,
    pub(crate) request: crate::engine::FireRequest,
    pub(crate) instance_id: u64,
    pub(crate) completion: WorkItemCompletion,
    /// The owning process. Process-wide suspend/terminate acts on every
    /// request with this identity.
    pub(crate) process_id: Option<ProcessId>,
    /// The submitting pipeline resource's stable scope identity, or `None`
    /// for an untracked/prebuilt fire; this is the wait-set key (frame lane,
    /// or at k=1 the synthesized single-slot stamp's lane).
    pub(crate) pipeline_id: Option<ProcessId>,
    pub(crate) prelaunch_copy: Option<::engine::KvCopy>,
    pub(crate) prelaunch_state_copy: Option<StateCopy>,
    /// Frame identity: lane/frame/slot this fire belongs to. At k=1 the
    /// worker synthesizes a single-slot stamp at admission (`lane` =
    /// `pipeline_id`, `seq` = the fire id); `None` = untracked/prebuilt,
    /// dispatched outside sealed-wave order.
    pub(crate) frame: Option<FrameStamp>,
    /// Whether this fire's program carries attention-stage hooks
    /// (OnAttnProj/OnAttn); fire planning keeps hook rows in the full-depth prefix.
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

/// Whether this request lowered its mask to rows the host can see. A mask
/// lives on its lane (`Lane::mask`), so this asks whether any lane carries one.
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
    /// One dispatch registering an instance's channels and binding it: the
    /// two per-join controls run back-to-back, and dispatching them
    /// separately doubled the turnover control convoy.
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistration>,
        /// Some on the program cache's first sight (registration rides
        /// between channels and bind in one dispatch); None when already
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
    // Only reached via `SchedulerHandle::copy_state`, not yet called by the
    // mock-engine fire path.
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
    /// A whole cohort of channel closes in one mailbox item, posted by
    /// process teardown; one item per departing process (not per channel)
    /// bounds the epoch a worker pass drains.
    CloseChannels {
        ids: Vec<u64>,
    },
    /// Event-driven retirement wake sent by [`NudgeWaker`] when an in-flight
    /// engine submission completion publishes. Carries no work.
    Nudge,
    /// A pipeline left the fleet ([`notify_pipeline_leave`]'s broadcast),
    /// handled immediately on dequeue like `Nudge`. `.0` is the leaving
    /// lane's scope id, `.1` the owning process when known — a different key
    /// space, and a leaver with no fires yet has no lane to recover it from.
    PipelineLeave(
        ProcessId,
        Option<ProcessId>,
        LeaveKind,
        Option<tokio::sync::oneshot::Sender<()>>,
    ),
    /// A capped execution slot was released ([`notify_execution_slot_released`]'s
    /// broadcast); the frame seal waits while the freed slot has a staged
    /// taker. Uncapped deployments never send this.
    ExecutionSlotReleased(ProcessId),
    /// The named process's deferred teardown finished; no event from it can
    /// follow. Retires its terminate tombstone.
    ProcessQuiesced(ProcessId),
    /// A parked process acquired its execution permit
    /// ([`notify_execution_slot_consumed`]'s broadcast): the frame seal
    /// waits for this process's first fire (can arrive before or after the release above).
    ExecutionSlotConsumed(ProcessId),
    /// A process is queued for an execution permit; it is the identified
    /// taker of the next slot to free (the semaphore is FIFO-fair).
    AdmissionQueued(ProcessId),
    /// It took the permit, or went away before it could.
    AdmissionDequeued(ProcessId),
    /// The planner concluded a suspended process is runnable again; its
    /// lanes may rejoin the wait-set and batch full frames again. Process-keyed.
    ProcessResume(ProcessId),
    /// A frame submit failed mid-way host-side: only `submitted` of the
    /// declared fires exist. The frame policy adjusts the expected count so
    /// it can still seal (frame mode only).
    FrameTruncate {
        lane: ProcessId,
        seq: u64,
        submitted: u32,
    },
    /// `forward.park()`: the guest leaves the seal's wait-set until it fires
    /// again, ordered by `seq` against that lane's submits; a guest may park
    /// with fires still outstanding (frame mode only).
    LanePark {
        lane: ProcessId,
        seq: u64,
    },
    /// Snapshots the run loop's state (queue composition, in-flight work,
    /// barrier membership); answered inline on dequeue so a held wave is inspectable.
    DebugDump {
        response: tokio::sync::oneshot::Sender<String>,
    },
    /// An engine-lane reply (launch accepted/rejected, control commit),
    /// handled immediately on dequeue like `Nudge` — mutates only in-flight
    /// bookkeeping, never queue order.
    Lane(LaneReply),
    Stop,
}

/// Wakes the scheduler thread through its own queue on a submission
/// completion, so batch/control retirement is event-driven, not timeout-polled.
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

// Engine lane: a dedicated thread owns the `EngineBox` and executes every
// engine call in FIFO order, keeping the engine's single-threaded
// serialization, off the scheduler worker's critical path.
//
// The lane owns the engine and the `channels` registry; the worker keeps
// all policy/admission state. Control arms split their engine half onto the
// lane and their map mutation + response onto the worker via
// `apply_lane_reply`, so a bind's response sends only after the instance is
// admissible. Replies ride `SchedulerItem::Lane`.

/// A [`FrameSubmission`] in transit to the engine lane.
///
/// SAFETY: `!Send` only through raw pointers into the engine's pinned,
/// thread-independent terminal-cell slots. Built complete on the worker,
/// moved to the lane, consumed exactly once by `engine.launch`; backing
/// requests stay alive in `in_flight_launches` until the frame retires
/// (strictly after the lane's reply).
struct LaneLaunch(crate::engine::FrameFire);
unsafe impl Send for LaneLaunch {}

/// Charges one lane request's wall time to `lane_launch_us`/`lane_control_us`
/// on drop, so an early return/continue still accounts. Diagnostic only.
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
        /// Whether this wave carries a prefill (`tokens > rows`); diagnostic
        /// only, splitting launch time to compare prefill vs. decode cost.
        prefill: bool,
    },
    /// A control `QueuedItem` (never `Launch`); the lane runs the engine
    /// half. Boxed since `QueuedItem` is large and controls are cold traffic,
    /// keeping every queued launch small.
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
    /// ops touching no worker state: registers, channel closes, failed binds).
    None,
    /// A successful bind: insert the instance, then respond — launch
    /// admission reads `instances` on the worker thread, so
    /// respond-after-insert makes the guest's first fire admissible.
    BindInstance {
        pipeline_id: Option<ProcessId>,
        bound: BoundInstance,
        respond: BindRespond,
    },
    /// A bind control completed without creating an instance.
    BindFinished { pipeline_id: Option<ProcessId> },
    /// A successful engine-side instance close: remove + close wait slots.
    CloseInstance { id: u64 },
    /// An async-completing control (copies / pool resizes): installs the
    /// engine's completion into the pending slot, or clears it on a
    /// synchronous rejection.
    AsyncControl {
        result: std::result::Result<SubmissionCompletion, String>,
    },
}

/// What a lane request still owes its caller: one reply, on one token.
/// Read off before serving, to cover a panic mid-serve: launch/control
/// slots are keyed by token with no timeout, so an unanswered token never resolves.
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
    /// Launch fast path: served before any queued control. A launch and a
    /// control are always mutually independent (a close posts only after its
    /// instance quiesces, a fire only after its bind commits), so preferring
    /// launches never reorders a dependent pair — and avoids control bursts
    /// head-of-line blocking the wave train.
    launch_tx: crossbeam::channel::Sender<LaneRequest>,
    control_tx: crossbeam::channel::Sender<LaneRequest>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Whose turn the engine lane is serving — the starvation bound on
/// [`EngineLoop::next_request`]'s launch-first preference (launches are the
/// device's own work; the counters cap that preference short of absolute priority).
#[derive(Default)]
struct LaneTurn {
    /// Launches served since the last control turn ended.
    launch_run: u32,
    /// Controls served in the turn now running.
    control_run: u32,
}

impl LaneTurn {
    /// Launches served before offering controls a turn. Two (not one)
    /// bounds a control's wait at roughly one frame while still letting
    /// launches run back to back.
    const LAUNCH_RUN_BEFORE_CONTROL: u32 = 2;

    /// Controls one turn may serve before launches get the lane back. High
    /// enough to drain a cohort turnover's whole bind generation in one
    /// turn; capped against a control flood holding the device off.
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

    /// Drains both queues and takes the engine + channel set back for
    /// teardown; the worker only calls this with `lane_inflight == 0`.
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

    /// Receives the next request, launches first but bounded so controls
    /// can't starve indefinitely (a bind control must commit here before its
    /// guest's first fire can exist, so a starved queue stalls bring-up).
    /// After [`LaneTurn::LAUNCH_RUN_BEFORE_CONTROL`] launches, drains
    /// `control_rx` up to [`LaneTurn::CONTROL_RUN_MAX`] controls, then
    /// returns to launches. Blocks on both queues when idle.
    fn next_request(
        launch_rx: &crossbeam::channel::Receiver<LaneRequest>,
        control_rx: &crossbeam::channel::Receiver<LaneRequest>,
        turn: &mut LaneTurn,
    ) -> std::result::Result<LaneRequest, ()> {
        use crossbeam::channel::TryRecvError;
        // Stay hot briefly after going empty before parking: a parked lane
        // pays a thread wake on every submit, which measurably hurt run-ahead pipelining.
        const ENGINE_LANE_HOT_US: u64 = 1_000_000;
        let hot_window = Duration::from_micros(ENGINE_LANE_HOT_US);
        let mut spin_until = Instant::now() + hot_window;
        loop {
            // Empty ends the turn (not just skips it), so a lane with
            // nothing queued pays one failed try_recv per run.
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
                // Both senders drop together (the graceful path is the
                // Shutdown marker); drain what remains, then stop.
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
            // Only wait; the loop re-runs launch-first try_recv once
            // something is ready, with a fresh spin window.
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
        // Async-completion bookkeeping: a shell may answer `submit` while
        // the device still runs, resolving a frame later from its own
        // completion callback on the driver's host-function thread.
        let broker = crate::engine::CompletionBroker::new();
        let settlements = crate::engine::completion::FrameSettlements::new();
        // The engine was opened on the worker's boot thread and moved here,
        // so per-thread device state (e.g. `cudaSetDevice`) must bind here
        // before any verb runs; a bind failure surfaces from the first verb.
        if let Some(engine) = engine.as_mut()
            && let Err(error) = engine.bind_thread()
        {
            tracing::error!(engine_idx, %error, "engine lane could not bind its thread");
        }
        // Completion sink installed once, before the first `submit`. This
        // callback runs on the driver's own completion thread and must not
        // block: one uncontended mutex, atomic release stores, then wake.
        if let Some(engine) = engine.as_mut()
            && engine.settles_asynchronously()
        {
            let book = std::sync::Arc::clone(&settlements);
            let published = broker.clone();
            engine.on_complete(std::sync::Arc::new(move |at: engine::StepDone, outcome| {
                book.settled(at.frame, &outcome, &published);
            }));
        }
        // A launch already received but not yet served: the next-fire
        // lookahead in `fire_frame` takes a queued launch out of `launch_rx`
        // early so the engine can be told its composition before the fire
        // ahead of it runs. FIFO holds since the stash is always the oldest unserved launch.
        let mut stash: Option<LaneRequest> = None;
        // Whose turn the lane is serving; see `next_request`.
        let mut lane_turn = LaneTurn::default();
        loop {
            let request = match stash.take() {
                Some(stashed) => {
                    // The lookahead took this out of `launch_rx` itself, so
                    // `next_request` never saw it; count it anyway, or a
                    // stash chain could hold the lane without owing controls a turn.
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
            // exactly once; a panic that skips its reply leaves the frame in
            // flight forever instead of failing the request.
            let owed = Owed::of(&request);
            // Answering the owed frame is only possible if the panic unwinds
            // here; under `panic = "abort"` this trades the old stall for
            // silently killing every other lane and session.
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
                            // Mutable: `fire_frame` moves each step's
                            // submission into the engine's `FrameSubmission`
                            // without copying a token vector.
                            let LaneLaunch(mut frame) = submission;
                            // Does not retry: both refusals are terminal —
                            // everything a device gate could refuse was
                            // already proved impossible at `submit_frame`,
                            // so a refusal here is a contract violation, not back-pressure.
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
                                // A frame that never reached the device must
                                // still settle, or its work items park forever.
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
                    // Every frame the device still owes: a panic leaves
                    // frames registered whose completion callbacks may never
                    // arrive, so fail them with the same verdict as `owed`.
                    settlements.close_all(&broker);
                    owed.answer(&reply_tx, "the engine lane panicked serving this request");
                    // A launch the lookahead stashed is queued work the
                    // channels no longer hold — fail it with the queue.
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

    /// Answers every request still queued, and every one that arrives, with
    /// the same failure. The engine is leaked rather than dropped: a panic
    /// mid-fire tears state (a half-recorded buffer, a half-resized pool)
    /// the destructor would run over, and a second panic during unwind aborts the process.
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

    /// Submits one frame and settles its steps. `submit` is all-or-nothing,
    /// so retry is around the whole frame, not per-step (a per-step retry
    /// would be unsound once earlier steps' KV is written).
    ///
    /// Settlement branches on `Engine::settles_asynchronously`: a
    /// synchronous engine (Metal) settles/wakes/retires inline; an
    /// asynchronous one (CUDA) returns an unsettled completion, and the
    /// device's own callback later publishes cells and retires the batch —
    /// this thread does not wait, returning to `next_request` with the
    /// frame still on the device (run-ahead bounded by `frame_dispatch_depth`).
    ///
    /// Never retries: a refusal here means a device gate refused something
    /// `validate_frame` already proved admissible, which is a contract
    /// violation — the lane fails the frame, settles it FAILED, and moves on.
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

        // The next launch, stated (`Engine::expect_fire`, advisory): a
        // prebind wants a successor's composition known before a step
        // fires. `launch_rx` has no peek, so the lookahead receives a
        // queued launch into `stash` and the run loop serves it next.
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

        // The frame as one submission, moved not copied: everything still
        // needed after the device has it lives beside it, so no token
        // vector is cloned to reach the engine.
        let submitted = ::engine::FrameSubmission {
            steps: frame
                .steps
                .iter_mut()
                .map(|step| std::mem::take(&mut step.submission))
                .collect(),
        };

        // Channel join, in: cells the guest put into a host ring since the
        // last fire cross into the device ring here. A frame is admitted
        // whole or not at all, so every attached pass committed before
        // `submit` answers `Ok`. An adopted channel moves nothing and only wakes.
        for step in &submitted.steps {
            for attachment in &step.attachments {
                if let Err(error) = channels.pump_in(engine.as_mut(), attachment.instance) {
                    return Err(format!("channel publish: {error}"));
                }
            }
        }
        // `submit` admits or refuses with zero side effects; both refusals
        // are terminal here (`Exhausted` is a contract violation past
        // static admission, `Impossible` is a baked-in ceiling).
        let ticket = match engine.submit(&submitted) {
            Ok(ticket) => ticket,
            Err(error) if error.is_retryable() => {
                return Err(format!(
                    "frame admission answered a retryable refusal past static \
                     admission, which the frame contract forbids: {error}"
                ));
            }
            Err(error) => return Err(format!("{error}")),
        };

        // `submit` answered `Ok`, so every attached pass committed. An
        // async wake is deferred (waking now would race the still-computing
        // device), so the ids ride into settlement instead.
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
            for step in &frame.steps {
                completion::settle(&step.terminal_cells, completion::TERMINAL_OUTCOME_SUCCESS);
            }
            return Ok(SubmissionCompletion::ready());
        }

        // Device running; the receipt is a correlation id. Registers cells
        // and deferred wakes against `FrameTicket::id`; run-ahead depth
        // accounting parks on this completion, not on enqueue.
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
                    // Host-codegen splice happens here, on the layer
                    // holding the engine handle that knows the backend.
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
                            // Dense slot order: `InstanceBinding::channels`'
                            // declaration order is what publish/take_channel address by.
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

    /// Closes one channel, tolerating a shell with no standalone channel to
    /// close: binding is the registration there, so `Unsupported` means the close already succeeded.
    fn close_channel(engine: &mut EngineBox, id: u64) -> Result<()> {
        match engine.close_channel(id) {
            Ok(()) | Err(engine::Error::Unsupported { .. }) => Ok(()),
            Err(error) => Err(anyhow::Error::from(error)),
        }
    }

    /// Register a set of channels with all-or-nothing rollback (the shared
    /// body of `RegisterChannels` and `RegisterChannelsBind`).
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
                    // The pump joining this host ring to the device half
                    // needs the ring, role, and id to know where cells are
                    // and which way they travel.
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

    /// No-op: an unregistered channel never allocated wait slots
    /// (`RegisteredChannel` is what answers them). Named so the
    /// cancellation paths pair with `release_registered_channel_wait_slots`, which still has real work.
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
    /// Boxed: the queue is rotated/compacted item-by-item every dispatcher
    /// pass, so an inline `PendingRequest` would move its whole payload;
    /// boxing makes a queue move a pointer move.
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
    /// One dispatch registering an instance's channels and binding it: the
    /// two per-join controls run back-to-back, and dispatching them
    /// separately doubled the turnover control convoy.
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistration>,
        /// Some on the program cache's first sight (registration rides
        /// between channels and bind in one dispatch); None when already
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
    /// whole batch instead of one control post per channel.
    CloseChannels {
        ids: Vec<u64>,
    },
}

/// A queued launch, plus the two fields the dispatcher's queue scan reads,
/// mirrored inline so [`BatchScheduler::scan_queue`] avoids a cache miss per
/// item read through the box. Cannot go stale: `QueuedLaunch` hands out
/// only `&PendingRequest` (no `DerefMut`), so neither field can be reassigned while queued.
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

/// A posted launch's lane lifecycle: enters `in_flight_launches` at POST;
/// `LaneReply::LaunchDone` upgrades the state. Retirement only ever
/// consumes `Accepted`/`Failed` heads — `Posted` isn't ready yet.
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
    /// Whether launches must wait for this control to settle: true for a
    /// `PreLaunchCopy` (its consumer fire is queued right behind it) and a
    /// pool resize (its drain must not admit new frames); false for
    /// standalone copies, whose grant-pinned pages no queued fire
    /// references. Also the exclusivity test — a holding control needs the
    /// in-flight set empty and blocks every other control while it settles.
    holds_launches: bool,
}

/// How often to re-check a control op holding launches while the device
/// sits idle; matches the frame policy's own gather poll.
const CONTROL_SETTLE_POLL_US: u64 = 500;

/// The async-completing controls the worker is waiting on — copies and
/// pool resizes; lifecycle controls execute on the lane and never enter
/// here. An exclusive control (`PreLaunchCopy`, pool resize) needs the set
/// empty and blocks anything else once posted. Standalone copies (the
/// residency planner's suspend/restore traffic) settle concurrently:
/// grant-pinned pages that no queued fire can name, so a single slot bought
/// no safety. No concurrency ceiling needed — the pending queue bounds it
/// (the planner enqueues at most one copy per suspending/restoring process).
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

    /// Whether `item` may post into this set now. A standalone copy and a
    /// lifecycle control are each refused only by an exclusive control; a
    /// lifecycle control never enters this set at all (lane FIFO guarantees its order).
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

/// The worker's pending queue, plus an epoch that changes on every
/// mutation, so [`BatchScheduler::scan_queue`] can skip a pass whose answer
/// hasn't changed. `DerefMut` bumps it, making invalidation total — every
/// `&mut` reach counts (rotations, in-place edits, rebuilds); over-invalidation just wastes a scan.
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

    /// Offset of the first item that is not a `Launch`, or `None` if the
    /// queue is all launches. Cached against the epoch, like
    /// [`Self::first_close`]: both answer "where does the launch run end",
    /// asked at least once per worker pass.
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

    /// Moves the leading run of launches behind the rest of the queue:
    /// equivalent to popping each off the front and pushing it back, but as one rotation.
    fn rotate_launch_run_to_back(&mut self, run_len: usize) {
        self.items.rotate_left(run_len);
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Insert a bring-up control ahead of the trailing close run.
    fn insert_before_closes(&mut self, item: QueuedItem) {
        let index = self.first_close();
        self.items.insert(index, item);
        self.epoch = self.epoch.wrapping_add(1);
        // The insert shifted the close run right and put a non-close at
        // `index`, so the next control lands after this one without rescanning.
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
/// the critical path. Compaction `take()`s every slot, so it only ever grows.
type SlotBuffer = Vec<Vec<Option<Box<PendingRequest>>>>;

struct SchedulerControl {
    tx: crossbeam::channel::Sender<SchedulerItem>,
    active_senders: AtomicUsize,
    shutdown_wait: Condvar,
    shutdown_gate: Mutex<()>,
    program_ids: Mutex<HashMap<u64, (u64, ::eta_compiler::codegen::launch::LaunchPackage)>>,
    accepting: AtomicBool,
    stats: Arc<SchedulerStats>,
    /// Which memory this engine's KV pages live in; carried on the handle
    /// since the `*_on` submit paths get a handle but no engine id.
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

    /// `engine_id` is an argument here because the ring is the runtime's:
    /// `ChannelRegistration` states only what the engine needs, not which
    /// registry slot a channel's endpoint belongs to.
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
        // Wait-for-all-active-lanes frame policy, one instance per engine
        // thread. At the default k=1 a frame is one wave; density comes
        // from the sealed epoch, throughput from run-ahead depth within it.
        let mut frame_policy = FramePolicy::new(
            frame_size,
            limits.max_forward_requests,
            limits.max_forward_tokens,
            Some(Arc::clone(&stats)),
        );
        frame_policy
            .preload_free_slots(crate::inferlet::process::execution_slot_capacity().unwrap_or(0));
        // Stall self-diagnosis: after 10s of zero progress with queued or
        // in-flight work, print the full state dump so the wedge names
        // itself (then re-print every 60s while it persists).
        let mut stall_since: Option<std::time::Instant> = None;
        let mut stall_dumps: u32 = 0;

        loop {
            let mut progress = false;
            // Epoch drain: a pass consumes only what was queued when it began,
            // so a sustained producer flood cannot keep `try_recv` non-empty
            // and hold retire/dispatch hostage behind the live stream.
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
            // imminent, hold back the bind permits retiring processes return,
            // so the staged cohort's bring-up does not compete with it.
            // Cleared once this pass has nothing left to do.
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
                // A pending wait-all hold re-arms the backstop at its own
                // cadence, never longer than the 250ms hang backstop, so a
                // held wave still fires on time with no new arrival.
                let backstop = Duration::from_millis(250);
                let recv_wait = wait_hint.map(|hold| hold.min(backstop)).unwrap_or(backstop);
                // With the device idle and a control holding launches, a
                // `Posted` control slot arms no nudge, so only the backstop ends the wait.
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
                        // means a wake was lost; steady-state count stays
                        // zero. Shutdown races and wait-all-hold timeouts
                        // (the wait's own cadence) never count here.
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
                        // Duplicate Terminate (exit funnel notifies from the
                        // terminate entry point and again from deferred
                        // teardown): the first leave did the work; skip
                        // straight to the ack a waiting sender may hold.
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
                    // frame's arrival completeness, unless its process already
                    // terminated (recording it would resurrect a ghost lane).
                    if let Some(stamp) = launch.frame
                        && !launch
                            .process_id
                            .is_some_and(|pid| terminated_processes.contains(&pid))
                    {
                        frame_policy.on_fire_rejected_at_admission(stamp, launch.process_id);
                    }
                    launch.completion.reject_unsubmitted(message);
                } else {
                    // Default single-slot deployment: every tracked fire is
                    // a one-fire frame, stamped at accept so a rejected fire
                    // never touches the wait-set; an untracked/prebuilt fire
                    // stays an unstamped rider.
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
                // Binds do not hold the seal. The policy stages bring-up
                // processes here, and a retiring execution slot earmarks one
                // staged successor — that earmark is what gathers a cohort
                // turnover into a dense epoch.
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
        // queued. A cheap scan decides; only an actual purge pays the
        // rebuild below.
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

    /// A standalone KV/state copy (suspend D2H, restore H2D, graft/CAS):
    /// touches pages no queued fire references, so it dispatches out-of-band
    /// once its slot frees, never barriering a queued fire. `PreLaunchCopy`
    /// is not in this class — it's order-coupled to its own launch.
    const fn standalone_copy(item: &QueuedItem) -> bool {
        matches!(
            item,
            QueuedItem::CopyKv { .. }
                | QueuedItem::CopyKvTracked { .. }
                | QueuedItem::CopyState { .. }
        )
    }

    /// Items a rotation can usefully expose at the queue front. A `Launch`
    /// is picked by fire id (reads the whole queue), a standalone copy is
    /// pulled from any position by the tail sweep, and a close is excluded
    /// (asked only while the whole close run is held) — none benefit from rotation.
    const fn rotation_target(item: &QueuedItem) -> bool {
        !matches!(
            item,
            QueuedItem::Launch(_)
                | QueuedItem::CloseInstance { .. }
                | QueuedItem::CloseChannels { .. }
        ) && !Self::standalone_copy(item)
    }

    /// Controls that dispatch without draining in-flight launches:
    /// registrations create entities nothing in flight can reference yet,
    /// copies touch only committed/quiesced extents, and closes follow
    /// their instance closes (the engine rejects closes with live
    /// attachments). Only pool resizes keep the empty-pipe requirement —
    /// drain is their ordering mechanism.
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

    /// Moves held launches behind work that can densify the current wave.
    /// The whole contiguous launch prefix rotates at once, since
    /// per-instance launch order is a dispatch invariant a partial rotation
    /// could split. Never breaks a `PreLaunchCopy`'s copy->consumer coupling
    /// (the consumer stays behind its copy). Lifecycle controls (registers,
    /// binds, closes) never order against a queued launch — only
    /// `PreLaunchCopy` and pool resizes do.
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
        // teardown. A bind never delays a queued launch, and jumps the
        // close tail (bind-vs-bind order preserved) since a bind and a
        // queued close always target different instances/channels.
        pending.insert_before_closes(item);
    }

    /// Rotates a front launch out only when doing so exposes dispatchable
    /// work behind it. `allow_lifecycle` is the wider flag: a lifecycle
    /// control needs no control slot, so a standalone copy in flight does
    /// not stop it from being worth exposing at the front.
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
        // Cohort-boundary close hold: while any bind is in assembly, teardown
        // closes yield the engine lane to the fresh cohort's registrations.
        // Held closes rotate and drain during the next generation's
        // execution. Shutdown never holds (the drain must retire everything).
        let hold_closes = !stopping && frame_policy.has_pending_binds();
        // The control slot exists for controls that settle asynchronously;
        // lifecycle controls run on the lane FIFO and never take it. A
        // standalone copy holding the slot doesn't block them: it addresses
        // grant-pinned pages no bind/register/close can reference.
        let slot_blocks_lifecycle = in_flight_control.holds_launches();
        while let Some(item) = pending.front() {
            match item {
                QueuedItem::Launch(_) => {
                    // Launches dispatch by id (`dispatch_frame_work`), not
                    // queue position; a launch at the front only needs to
                    // yield to a dispatchable control behind it.
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
                    // A close needs only its own instance quiesced, never a
                    // global pipe drain or control slot (a settling
                    // standalone copy addresses pages no close can name).
                    if slot_blocks_lifecycle {
                        break;
                    }
                    let id = *id;
                    if hold_closes {
                        // Held for the boundary: rotate without claiming
                        // progress (the bind-completed lane reply that empties
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
                    // quiescing it keep flowing; retirement re-checks it. A
                    // close only moves backward, so it never overtakes its
                    // own instance's work. No progress claim: `busy` guarantees a later wake.
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
                // A settling exclusive control (a `PreLaunchCopy` or a pool
                // resize) blocks the next control; standalone copies and
                // lifecycle controls are refused by nothing else — see
                // `InFlightControls::admits`.
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
        // Standalone copies dispatch from any queue position once the
        // control slot frees: their pages are grant-pinned so nothing
        // queued orders against them, and leaving them positional would
        // starve the planner's suspend/restore progress on a held frame.
        // They also pipeline (the sweep keeps posting while no exclusive
        // control holds the set); the queue bounds depth to what the planner enqueued.
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

    /// Posts a control to the engine lane after worker-side pre-checks. The
    /// engine half runs on the lane in FIFO order; worker-map effects come
    /// back as a [`LaneCommit`]. Async controls occupy the single control slot from post.
    fn post_control(
        engine_loop: &EngineLoop,
        lane_inflight: &mut u64,
        lane_token: &mut u64,
        instances: &mut HashMap<u64, TrackedInstance>,
        in_flight_control: &mut InFlightControls,
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
            _ => {}
        }
        *lane_token += 1;
        let token = *lane_token;
        // Async-completing controls enter the in-flight set from post: an
        // exclusive one must not be passed by any later control. Only
        // standalone copies do not hold launches — the classification shared
        // with the out-of-band dispatch and `InFlightControls`'s concurrency
        // rule.
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
    /// rider, and the lanes a frame post must hold for. Only a queued
    /// `PreLaunchCopy` blocks a lane; standalone copies never barrier fires
    /// (their pinned pages are never referenced by a queued fire).
    fn scan_queue<'a>(
        cache: &'a mut ScanCache,
        pending: &PendingQueue,
        stopping: bool,
    ) -> &'a QueueScan {
        // A pure function of (queue contents, stopping): a pass at an
        // unchanged epoch would rebuild what's already here, which matters
        // since passes run many times per wave while the queue rarely changes.
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

    /// Launch dispatch: posts whole sealed frames to the engine lane at the
    /// run-ahead depth (the engine executes a frame's waves in slot order
    /// as one closed system). At k=1 a sealed frame is one wave, so this
    /// degenerates to per-wave wait-all dispatch.
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
            // depend on it (a `PreLaunchCopy`'s consumer fire queued behind
            // it, a resize's pipe drain); a settling standalone copy holds
            // nothing, so frames keep posting while it settles.
            if in_flight_control.holds_launches() {
                // Counted only when the device is idle: the frame policy
                // isn't even consulted here. See `probe::QuorumProbes`.
                if in_flight_launches.is_empty() {
                    stats
                        .fire
                        .quorum
                        .idle_break_control
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // The completion nudge is armed but doesn't reliably
                    // fire, so the park must not sleep the full backstop; a
                    // hint, not a nudge fix, since the settle is cheap to poll.
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
                        // Abandoned pipeline: silent past the timeout
                        // without calling `forward.park()`. The policy
                        // already dropped these lanes, so `continue` re-plans
                        // a gather that no longer waits on them.
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
    /// settled/stale, assemble the frame submission, and post it as one
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
        // One map, carrying both the wave and the in-wave position (the
        // sealed wave's id order / lane admission order), so requests below
        // are placed by slot rather than sorted.
        let mut slot_of: HashMap<u64, (usize, usize)> =
            HashMap::with_capacity(waves.iter().map(Vec::len).sum());
        for (index, wave) in waves.iter().enumerate() {
            for (position, &fire_id) in wave.iter().enumerate() {
                slot_of.insert(fire_id, (index, position));
            }
        }
        let mut kept: VecDeque<QueuedItem> = VecDeque::with_capacity(pending.len());
        // Place by slot rather than push-then-sort: `position` is already a
        // permutation of `0..wave.len()`, so the sealed order is recovered by
        // writing each request straight into its slot. The buffer is
        // caller-owned and comes back empty from the last frame.
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
            // A lane-rejected launch retires like a wave (it entered the
            // pipe at post, so the common unwind applies); only its
            // requests' settlement differs (rejected, never submitted).
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
                                // Admitted frames are atomic and stream
                                // work is success-only, so a surviving RETRY
                                // violates the engine contract: fail loudly instead of replaying.
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
                    // A Posted batch never retires; a missing token is a bug.
                    tracing::error!(token, "lane launch reply for an unknown batch");
                    return;
                };
                match result {
                    Ok(completion) => {
                        // Commit target epochs at accept: lane replies
                        // arrive in post (acceptance) order, so the
                        // per-instance ledger stays gapless and each
                        // completion's target matches the ordinal the instance slot will publish.
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
                        // unique and requested ids are pre-checked at post.
                        // Refuse loudly; the legit instance stays untouched.
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
                    // Respond after the insert: launch admission reads
                    // `instances` here, so the guest's first fire (sent
                    // only after this response) is always admissible.
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

    /// An engine whose `launch` panics; every other verb is a stub, since
    /// what is under test is the lane, not a backend.
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

    /// A launch the engine panics on is answered, and so is the one behind
    /// it: the lane replies exactly once per request, so a panic must not
    /// skip the reply and leave the frame in flight forever.
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
                // One step: a frame with none never reaches `Engine::fire`,
                // the verb this test's engine panics in.
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

        // Bounded: the failure this test exists for is a wait with no end,
        // and an unwrap on a blocking recv would hang the suite rather than fail it.
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

    /// A retryable refusal past static admission is a contract violation,
    /// not back-pressure — `validate_frame` proves no device gate can
    /// refuse before admission, so a refusal here fails loudly instead of retrying.
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
            error.contains("frame contract forbids"),
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

}
