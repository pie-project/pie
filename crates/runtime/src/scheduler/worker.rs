use crate::rt::Instant;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use ::engine::{ChannelRegistration, ProgramRegistration, StateCopy};

use crate::engine::{
    BoundInstance, ChannelJoin, EngineBox, EngineId, InstanceBindingPlan, RegisteredChannel,
    SchedulerLimits, SubmissionCompletion, WorkItemCompletion,
};
use crate::scheduler::ProcessId;
use anyhow::{Result, anyhow};

use super::ControlCompletion;
use super::batch::{self, AdmissionLimits};
use super::frame::{self, FramePlan, FramePolicy, FrameStamp};
use super::stats::{self, SchedulerStats};

use super::lane::{ControlReplyRx, Lane, LaneCommit, LaunchReplyRx, LaunchResult};
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaveKind {
    Terminate,
    Suspend,
    Close,
}

fn broadcast(make: impl Fn() -> SchedulerItem) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(make());
    }
}

fn broadcast_leave_fenced(pid: ProcessId, kind: LeaveKind) -> Vec<TerminateFence> {
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
        .collect()
}

pub(crate) fn notify_lane_close(scope: ProcessId, owner: Option<ProcessId>) {
    broadcast(|| SchedulerItem::PipelineLeave(scope, owner, LeaveKind::Close, None));
}

pub(crate) fn notify_process_suspend(pid: ProcessId) {
    broadcast(|| SchedulerItem::PipelineLeave(pid, Some(pid), LeaveKind::Suspend, None));
}

/// `pid` is blocked on the planner or a store (a grant, its restore, room
/// for a bind, a segment, a fire span, a rung): until it is through it
/// submits nothing, so no frame on any engine waits on its lanes, and its
/// complete frames go at once ([`super::frame::FramePolicy::on_owner_waits`]).
/// Its next submit brings its lanes back.
pub(crate) fn notify_owner_blocked(pid: ProcessId) {
    #[cfg(test)]
    OWNERS_BLOCKED.lock().unwrap().push(pid);
    broadcast(|| SchedulerItem::OwnerWaits(pid));
}

/// `pid`'s wait on the planner or a store ended (told before control goes
/// back to its guest): its fires are awaited at boundaries again.
pub(crate) fn notify_owner_unblocked(pid: ProcessId) {
    broadcast(|| SchedulerItem::OwnerUnblocked(pid));
}

/// The owners told blocked, in order; for a test.
#[cfg(test)]
pub(crate) static OWNERS_BLOCKED: std::sync::Mutex<Vec<ProcessId>> = std::sync::Mutex::new(Vec::new());

/// `pid`'s guest waits on input from outside the runtime (a session
/// receive, a timer, an HTTP response): no frame on any engine waits on its
/// lanes until each submits again.
pub(crate) fn notify_owner_away(pid: ProcessId) {
    broadcast(|| SchedulerItem::OwnerAway(pid));
}

pub(crate) fn notify_process_resume(pid: ProcessId) {
    broadcast(|| SchedulerItem::ProcessResume(pid));
}

pub(crate) fn post_process_terminate(pid: ProcessId) {
    broadcast(|| SchedulerItem::PipelineLeave(pid, None, LeaveKind::Terminate, None));
}

pub(crate) fn post_process_terminate_fenced(pid: ProcessId) -> Vec<TerminateFence> {
    broadcast_leave_fenced(pid, LeaveKind::Terminate)
}

pub(crate) type TerminateFence = tokio::sync::oneshot::Receiver<()>;

pub(crate) async fn await_terminate_fences(fences: Vec<TerminateFence>) {
    for fence in fences {
        let _ = fence.await;
    }
}

pub(crate) async fn notify_pipeline_close(pid: ProcessId) {
    await_terminate_fences(broadcast_leave_fenced(pid, LeaveKind::Close)).await;
}

pub(crate) fn notify_lane_open(lane: ProcessId, owner: ProcessId) {
    broadcast(|| SchedulerItem::LaneOpen { lane, owner });
}

pub(crate) fn notify_lane_park(pid: ProcessId, seq: u64) {
    broadcast(|| SchedulerItem::LanePark { lane: pid, seq });
}

pub(crate) fn notify_execution_slot_released(pid: ProcessId) {
    broadcast(|| SchedulerItem::ExecutionSlotReleased(pid));
}

pub(crate) fn notify_process_quiesced(pid: ProcessId) {
    broadcast(|| SchedulerItem::ProcessQuiesced(pid));
}

pub(crate) fn notify_execution_slot_consumed(pid: ProcessId) {
    broadcast(|| SchedulerItem::ExecutionSlotConsumed(pid));
}

pub(crate) fn notify_schedule_changed() {
    broadcast(|| SchedulerItem::Nudge);
}

pub(crate) fn notify_admission_queued(pid: ProcessId) {
    broadcast(|| SchedulerItem::AdmissionQueued(pid));
}

pub(crate) fn notify_admission_dequeued(pid: ProcessId) {
    broadcast(|| SchedulerItem::AdmissionDequeued(pid));
}

pub(crate) static BACKSTOP_RETIREMENTS: AtomicU64 = AtomicU64::new(0);
static NEXT_LOGICAL_FIRE_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct PendingRequest {
    pub(crate) logical_fire_id: u64,
    pub(crate) request: crate::engine::FireRequest,
    pub(crate) instance_id: ::engine::InstanceId,
    pub(crate) completion: WorkItemCompletion,
    pub(crate) process_id: Option<ProcessId>,
    pub(crate) pipeline_id: Option<ProcessId>,
    pub(crate) prelaunch_copy: Option<::engine::KvCopy>,
    pub(crate) prelaunch_state_copy: Option<StateCopy>,
    pub(crate) frame: Option<FrameStamp>,
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
        instance_id: ::engine::InstanceId,
        completion: WorkItemCompletion,
        process_id: Option<ProcessId>,
        pipeline_id: Option<ProcessId>,
        prelaunch_copy: Option<::engine::KvCopy>,
        prelaunch_state_copy: Option<StateCopy>,
        frame: Option<FrameStamp>,
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
            lora_program,
        }
    }

    pub(crate) fn wire_row_count(&self) -> usize {
        self.request.lanes.len()
    }
}

pub(crate) fn wave_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_WAVE_TRACE").is_some())
}

pub(crate) fn wave_trace_emit(line: String) {
    static BUFFERED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static LINES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let buffered = *BUFFERED.get_or_init(|| {
        #[cfg(target_arch = "wasm32")]
        let on = false;
        #[cfg(not(target_arch = "wasm32"))]
        let on = std::env::var_os("PIE_WAVE_TRACE").is_some_and(|v| v == "buffer");
        #[cfg(not(target_arch = "wasm32"))]
        if on {
            std::thread::spawn(|| {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    let drained: Vec<String> =
                        std::mem::take(&mut *LINES.lock().expect("lines lock poisoned"));
                    for line in drained {
                        eprintln!("{line}");
                    }
                }
            });
        }
        on
    });
    if buffered {
        LINES.lock().expect("lines lock poisoned").push(line);
    } else {
        eprintln!("{line}");
    }
}

pub(crate) fn wave_trace_us() -> u128 {
    static START: std::sync::OnceLock<crate::rt::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(crate::rt::Instant::now)
        .elapsed()
        .as_micros()
}

fn has_wire_masks(request: &crate::engine::FireRequest) -> bool {
    request.lanes.iter().any(|lane| lane.mask.is_some())
}

#[allow(
    clippy::large_enum_variant,
    reason = "`Launch` is the hot variant: boxing it would allocate per step"
)]
enum SchedulerItem {
    Launch {
        pending: PendingRequest,
    },
    PrepareProgram {
        program: Box<ProgramRegistration>,
        response: tokio::sync::oneshot::Sender<Result<::engine::ProgramCompile>>,
    },
    FinishProgram {
        prepared: ::engine::Prepared,
        response: tokio::sync::oneshot::Sender<Result<::engine::ProgramId>>,
    },
    FireFacts {
        key: crate::store::shared::programs::FireKey,
        response: tokio::sync::oneshot::Sender<Result<()>>,
    },
    /// Asks the engine what its standing guest bytes would be once a
    /// binding is bound, and answers what that adds to the pool's booked
    /// total (`store::shared::programs`).
    StandingAfter {
        binding: Box<::engine::InstanceBinding>,
        response: tokio::sync::oneshot::Sender<Result<u64>>,
    },
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistration>,
        bind: InstanceBindingPlan,
        response: tokio::sync::oneshot::Sender<
            Result<(Vec<RegisteredChannel>, ::engine::ProgramId, BoundInstance)>,
        >,
    },
    CopyTracked {
        plan: PreLaunchCopy,
        completion: ControlCompletion,
    },
    CloseInstance {
        id: ::engine::InstanceId,
        pacing_wait_id: u64,
    },
    CloseChannel {
        id: ::engine::ChannelId,
    },
    CloseChannels {
        ids: Vec<::engine::ChannelId>,
    },
    Nudge,
    PipelineLeave(
        ProcessId,
        Option<ProcessId>,
        LeaveKind,
        Option<tokio::sync::oneshot::Sender<()>>,
    ),
    ExecutionSlotReleased(ProcessId),
    ProcessQuiesced(ProcessId),
    ExecutionSlotConsumed(ProcessId),
    AdmissionQueued(ProcessId),
    AdmissionDequeued(ProcessId),
    ProcessResume(ProcessId),
    /// The process waits on something the frame's firing would free (a
    /// raise of the activation rung): its lanes go out of the frame
    /// ([`super::frame::FramePolicy::on_owner_waits`]).
    OwnerWaits(ProcessId),
    /// The owner's wait on the planner or a store ended.
    OwnerUnblocked(ProcessId),
    /// The process's guest waits on outside input: its lanes go out of the
    /// frame until each submits again
    /// ([`super::frame::FramePolicy::on_owner_away`]).
    OwnerAway(ProcessId),
    FrameTruncate {
        lane: ProcessId,
        seq: u64,
        submitted: u32,
    },
    LaneOpen {
        lane: ProcessId,
        owner: ProcessId,
    },
    LanePark {
        lane: ProcessId,
        seq: u64,
    },
    HostKv {
        op: HostKvOp,
        response: tokio::sync::oneshot::Sender<Result<Vec<u8>>>,
    },
    Pool {
        op: PoolOp,
        response: tokio::sync::oneshot::Sender<Result<()>>,
    },
    Stop,
}

/// The rows a frame may seal at: the fire cap, or, until the scheduler has
/// granted the walk that cap (`granted`), the rung the engine still holds
/// (`held`) where frames queued wider (`queued`) were cut under it. Frames
/// cut before the cap came down go; the grant, and the rung's lower after
/// it, wait for them.
pub(super) fn seal_rows(cap: u32, held: u32, granted: Option<u32>, queued: usize) -> usize {
    let cap = cap as usize;
    if granted == Some(cap as u32) || queued <= cap {
        return cap;
    }
    queued.min(held as usize).max(cap)
}

/// A change to how much of the pool has memory under it.
#[derive(Debug, Clone)]
pub(crate) enum PoolOp {
    /// A shared pool backs segments `0..n` (`Engine::commit_segments`).
    CommitSegments(u32),
    /// A shared pool gives back segments `n..` (`Engine::release_segments`).
    ReleaseSegments(u32),
    /// The activation rung moves (`Engine::set_rung`), where the span is
    /// backed on demand: a raise before any fire of the wider rung, a lower
    /// as a notice after its retire fence.
    Rung { from: u32, to: u32 },
    /// A shared pool whose span is backed on demand backs its top `n` span
    /// segments (`Engine::commit_span`).
    CommitSpan(u32),
    /// A shared pool whose span is backed on demand keeps its top `n` span
    /// segments and gives back the rest (`Engine::release_span`).
    ReleaseSpan(u32),
    /// A shared pool's chunks move, ahead of every fire queued after it
    /// (`Engine::evacuate`).
    Evacuate(::engine::chunk::Evacuation),
    /// An instance's tables or a ring were unparked into these slots
    /// (`Engine::rebind_slots`), ahead of every fire queued after it.
    Rebind(::engine::Rebind),
    /// Every copy submitted so far has landed (`Engine::settle_copies`):
    /// what an aborted move waits for before it gives back slots a copy may
    /// still be writing.
    SettleCopies,
}

impl PoolOp {
    /// Its name, as a stall dump says the call a lane is in.
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::CommitSegments(_) => "commit segments",
            Self::ReleaseSegments(_) => "release segments",
            Self::Rung { .. } => "set rung",
            Self::CommitSpan(_) => "commit span",
            Self::ReleaseSpan(_) => "release span",
            Self::Evacuate(_) => "evacuate",
            Self::Rebind(_) => "rebind",
            Self::SettleCopies => "settle copies (waits every copy landed)",
        }
    }
}

/// A read or write of host kv slots' bytes, behind every copy queued before
/// it.
#[derive(Debug)]
pub(crate) enum HostKvOp {
    Write(Vec<u32>, Vec<u8>),
    /// The engine's host-tier layout, as JSON; empty when it cannot say.
    Layout,
    /// A copy of device pages into host slots, then a read of those slots:
    /// one item on the lane, with no completion to settle between.
    CopyRead(::engine::KvCopy, Vec<u32>),
    /// The bytes of host kv slots.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    Read(Vec<u32>),
    /// The bytes of host rs rows: state slots, then buffer pages.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    ReadState(Vec<u32>, Vec<u32>),
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    WriteState(Vec<u32>, Vec<u32>, Vec<u8>),
    /// The bytes of host tier slots, shape by shape (`Engine::read_tier_slots`).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    ReadTier(Vec<(::engine::chunk::ShapeId, Vec<u32>)>),
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    WriteTier(Vec<(::engine::chunk::ShapeId, Vec<u32>)>, Vec<u8>),
    /// The engine's tap groups (`Engine::tap_groups`), as JSON.
    TapGroups,
    /// The weight cache's resident set (`Engine::weight_residency`),
    /// answered with the bytes the engine holds back once it took it, ahead
    /// of every fire queued after it.
    WeightResidency(::engine::chunk::WeightResidency),
    /// The engine's weight use record since it last answered
    /// (`Engine::weight_uses`), as JSON.
    WeightUses,
}

type SchedTx = tokio::sync::mpsc::UnboundedSender<SchedulerItem>;
type SchedRx = tokio::sync::mpsc::UnboundedReceiver<SchedulerItem>;
#[derive(Default)]
struct LaneReplies {
    next_control_id: u64,
    waits: FuturesUnordered<ReplyWait>,
}

impl LaneReplies {
    fn is_empty(&self) -> bool {
        self.waits.is_empty()
    }

    fn post_launch(&mut self, reply: LaunchReplyRx) {
        self.waits.push(ReplyWait::Launch(reply));
    }

    fn next_control_id(&mut self) -> u64 {
        self.next_control_id += 1;
        self.next_control_id
    }

    fn post_control(&mut self, id: u64, reply: ControlReplyRx) {
        self.waits.push(ReplyWait::Control { id, reply });
    }

    fn try_next(&mut self) -> Option<LaneReply> {
        self.waits.next().now_or_never().flatten()
    }
}

enum ReplyWait {
    Launch(LaunchReplyRx),
    Control { id: u64, reply: ControlReplyRx },
}

enum LaneReply {
    Launch(LaunchResult),
    Control { id: u64, commit: LaneCommit },
}

impl std::future::Future for ReplyWait {
    type Output = LaneReply;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<LaneReply> {
        match self.get_mut() {
            ReplyWait::Launch(reply) => std::pin::Pin::new(reply).poll(cx).map(|answer| {
                LaneReply::Launch(answer.unwrap_or_else(|_| {
                    Err("the engine lane dropped this launch unanswered".to_string().into())
                }))
            }),
            ReplyWait::Control { id, reply } => {
                std::pin::Pin::new(reply)
                    .poll(cx)
                    .map(|answer| LaneReply::Control {
                        id: *id,
                        commit: answer.unwrap_or_else(|_| LaneCommit::AsyncControl {
                            result: Err(
                                "the engine lane dropped this control unanswered".to_string()
                            ),
                        }),
                    })
            }
        }
    }
}

#[derive(Clone)]
pub(crate) enum PreLaunchCopy {
    Kv(::engine::KvCopy),
    State(StateCopy),
    /// Slots of one shape between the device and a tier below it
    /// (CONTRACT-BC 22).
    Slots(::engine::chunk::SlotCopy),
}

impl PreLaunchCopy {
    /// What the copy moves, between which tiers: what a stall dump names
    /// an unsettled copy by.
    pub(super) fn describe(&self) -> String {
        match self {
            Self::Kv(copy) => format!(
                "kv {:?}->{:?} {} pages",
                copy.src,
                copy.dst,
                copy.src_page_ids.len().max(copy.moves.len())
            ),
            Self::State(_) => "recurrent state".to_string(),
            Self::Slots(copy) => format!(
                "shape {} {:?}->{:?} {} slots {} runs",
                copy.shape.0,
                copy.src,
                copy.dst,
                copy.slots.len(),
                copy.runs.len()
            ),
        }
    }

    pub(super) fn label(&self) -> &'static str {
        match self {
            Self::Kv(_) => "KV copy",
            Self::State(_) => "recurrent-state copy",
            Self::Slots(_) => "slot copy",
        }
    }
}

pub(super) struct BindRespond {
    pub(super) registered: Vec<RegisteredChannel>,
    pub(super) program_id: ::engine::ProgramId,
    pub(super) response: tokio::sync::oneshot::Sender<
        Result<(Vec<RegisteredChannel>, ::engine::ProgramId, BoundInstance)>,
    >,
}

pub(super) enum QueuedItem {
    Launch(QueuedLaunch),
    PreLaunchCopy {
        plan: PreLaunchCopy,
        logical_completion: WorkItemCompletion,
        process_id: Option<ProcessId>,
        pipeline_id: Option<ProcessId>,
    },
    /// A program's registration, first step: the engine checks it and
    /// hands back its compile (`Engine::prepare_program`), which runs off
    /// the lane while the lane fires on.
    PrepareProgram {
        program: Box<ProgramRegistration>,
        response: tokio::sync::oneshot::Sender<Result<::engine::ProgramCompile>>,
    },
    /// A program's registration, last step: the engine takes what its
    /// compile made (`Engine::finish_program`), and the lane learns what an
    /// instance of it holds in the pool.
    FinishProgram {
        prepared: ::engine::Prepared,
        response: tokio::sync::oneshot::Sender<Result<::engine::ProgramId>>,
    },
    /// Asks the engine what fires of a program bound at given extents take
    /// beside the model's activation, for the pool's rung arithmetic.
    FireFacts {
        key: crate::store::shared::programs::FireKey,
        response: tokio::sync::oneshot::Sender<Result<()>>,
    },
    /// Asks the engine what its standing guest bytes would be once a
    /// binding is bound, and answers what that adds to the pool's booked
    /// total (`store::shared::programs`).
    StandingAfter {
        binding: Box<::engine::InstanceBinding>,
        response: tokio::sync::oneshot::Sender<Result<u64>>,
    },
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistration>,
        bind: InstanceBindingPlan,
        response: tokio::sync::oneshot::Sender<
            Result<(Vec<RegisteredChannel>, ::engine::ProgramId, BoundInstance)>,
        >,
    },
    CopyTracked {
        plan: PreLaunchCopy,
        completion: ControlCompletion,
    },
    HostKv {
        op: HostKvOp,
        response: tokio::sync::oneshot::Sender<Result<Vec<u8>>>,
    },
    Pool {
        op: PoolOp,
        response: tokio::sync::oneshot::Sender<Result<()>>,
    },
    CloseInstance {
        id: ::engine::InstanceId,
        pacing_wait_id: u64,
    },
    CloseChannels {
        ids: Vec<::engine::ChannelId>,
    },
}

/// The rows the scheduler's queued and in-flight launches ask a fire to
/// carry, held as a rung want: the most they asked since the scheduler
/// last slept with nothing to do, so a batch that breathes between its
/// steps does not move the rung back and forth each step.
#[derive(Default)]
struct BatchWant(Option<(usize, crate::store::shared::RungWant)>);

impl BatchWant {
    fn ask(&mut self, engine: usize, rows: usize) {
        if rows == 0 || self.0.as_ref().is_some_and(|(held, _)| *held >= rows) {
            return;
        }
        let rows32 = u32::try_from(rows).unwrap_or(u32::MAX);
        let want = crate::store::shared::want_rows(engine, rows32);
        self.0 = Some((rows, want));
        // As a pass's ask does: the rung is raised at once where nothing
        // moves, and a cap the walker lowered behind a rung still up is
        // lifted back, so the batch is not cut while the walk catches up.
        if let Some(shared) = crate::store::registry::try_get(0, engine).and_then(|s| s.shared) {
            crate::store::shared::raise_now(&shared, engine, rows32);
        }
    }

    /// Asks the rows one fire of `fires` (the queued and in-flight
    /// launches) takes, no more than a step carries.
    fn ask_for<'a>(
        &mut self,
        engine: usize,
        count: &mut RowCount,
        fires: impl IntoIterator<Item = &'a crate::engine::FireRequest>,
        max_forward_tokens: usize,
    ) {
        self.ask(engine, count.count(fires).min(max_forward_tokens));
    }

    fn release(&mut self) {
        self.0 = None;
    }
}

/// The rows one fire of a set of launches takes: each lane's rows (a
/// prompt's tokens, a verify's drafted rows, one a decode), summed over the
/// lanes. A lane rides its seat in every launch it is queued or in flight
/// in, and a fire carries one of them, so a lane counts the widest of its
/// pipelined launches once; a seatless lane counts in the launch whose
/// seatless lanes take the most rows. The rows a launch carries past its
/// own ([`crate::engine::FireRequest::rung_rows`]) count as the widest such
/// launch's.
#[derive(Default)]
struct RowCount {
    seats: Vec<u32>,
}

impl RowCount {
    fn count<'a>(
        &mut self,
        fires: impl IntoIterator<Item = &'a crate::engine::FireRequest>,
    ) -> usize {
        self.seats.fill(0);
        let mut seatless = 0usize;
        let mut carried = 0usize;
        for fire in fires {
            let mut here = 0usize;
            for lane in &fire.lanes {
                let rows = lane.tokens.len().max(1);
                if lane.kv_less {
                    here += rows;
                    continue;
                }
                let seat = lane.slot as usize;
                if seat >= self.seats.len() {
                    self.seats.resize(seat + 1, 0);
                }
                let rows32 = u32::try_from(rows).unwrap_or(u32::MAX);
                self.seats[seat] = self.seats[seat].max(rows32);
            }
            seatless = seatless.max(here);
            carried = carried.max(fire.rung_rows().saturating_sub(fire.tokens()));
        }
        let seated: usize = self.seats.iter().map(|&rows| rows as usize).sum();
        seated + seatless + carried
    }
}

pub(super) struct QueuedLaunch {
    fire_id: u64,
    framed: bool,
    /// Fired in a batch of its own ([`HeldForPlacement::solo`]).
    solo: bool,
    request: Box<PendingRequest>,
}

impl QueuedLaunch {
    fn new(request: Box<PendingRequest>) -> Self {
        Self {
            fire_id: request.logical_fire_id,
            framed: request.frame.is_some(),
            solo: false,
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

enum LaunchState {
    Posted,
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
    /// What the batch takes of its rung: its tokens, or the recurrent rows
    /// its requests carry where they are more.
    rung_rows: usize,
    /// Where the engine refused it for where state lies
    /// (`engine::Error::Placement`), the logical chunks it named.
    placement: Vec<u32>,
    /// Refused so, its steps as the engine was handed them: the lanes its
    /// requests go again by (`batch::restore_lanes`).
    unfired: Vec<crate::engine::StepFire>,
}

/// Fires of a batch the engine refused for where state lies, held until
/// the pool has moved that state down ([`crate::store::shared::evacuate_refused`]),
/// then fired again; none fails for it.
struct HeldForPlacement {
    chunks: Vec<u32>,
    requests: Vec<Box<PendingRequest>>,
    /// Fired again one by one: refused again after the move landed, so
    /// only the fires the engine still refuses alone fail.
    solo: bool,
    /// The planner was asked for the free chunks its move lacks.
    funding: bool,
}

enum ControlSlotState {
    Posted { id: u64 },
    Ready(SubmissionCompletion),
}

struct PendingControl {
    state: ControlSlotState,
    logical_completion: Option<WorkItemCompletion>,
    process_id: Option<ProcessId>,
    pipeline_id: Option<ProcessId>,
    tracked_completion: Option<ControlCompletion>,
    operation: &'static str,
    holds_launches: bool,
    prelaunch: bool,
    /// What it moves, said by a stall dump (a copy's tiers and slots).
    what: String,
    /// When it was posted to the lane: how long a dump finds it unsettled.
    since: crate::rt::Instant,
}

#[derive(Default)]
struct InFlightControls {
    settling: Vec<PendingControl>,
}

impl InFlightControls {
    fn is_empty(&self) -> bool {
        self.settling.is_empty()
    }

    fn is_settling(&self) -> bool {
        !self.settling.is_empty()
    }

    fn iter(&self) -> std::slice::Iter<'_, PendingControl> {
        self.settling.iter()
    }

    fn admits_copy(&self) -> bool {
        self.settling.iter().all(|control| !control.holds_launches)
    }

    fn admits(&self, item: &QueuedItem) -> bool {
        if BatchScheduler::standalone_copy(item) || BatchScheduler::lifecycle_control(item) {
            !self.holds_launches()
        } else if let QueuedItem::PreLaunchCopy {
            pipeline_id: Some(pipeline),
            ..
        } = item
        {
            // Other pipelines' pre-launch copies touch other state: they post
            // together, so the launches behind them ride one wave.
            self.settling
                .iter()
                .all(|control| control.prelaunch && control.pipeline_id != Some(*pipeline))
        } else {
            self.is_empty()
        }
    }

    fn copies_only(&self) -> bool {
        self.settling.iter().all(|control| control.prelaunch)
    }

    fn holds_launches(&self) -> bool {
        self.settling.iter().any(|control| control.holds_launches)
    }

    fn push(&mut self, control: PendingControl) {
        self.settling.push(control);
    }

    fn position_posted(&self, id: u64) -> Option<usize> {
        self.settling.iter().position(
            |control| matches!(control.state, ControlSlotState::Posted { id: t } if t == id),
        )
    }
}

#[derive(Default)]
struct QueueScan {
    queued_ids: frame::QueuedFireIds,
    blocked_lanes: HashSet<ProcessId>,
    /// The unframed fires that ride together: each pipeline's first, none
    /// behind its copy.
    untracked: Vec<u64>,
    riding: HashSet<Option<ProcessId>>,
    /// The pipelines with a framed launch queued: an unframed one behind it
    /// does not ride ahead of it.
    framed_ahead: HashSet<ProcessId>,
    drain_eligible: Vec<u64>,
    /// The rider batch is one fire fired alone ([`QueuedLaunch::solo`]).
    solo: bool,
}

impl QueueScan {
    fn clear(&mut self) {
        self.queued_ids.clear();
        self.blocked_lanes.clear();
        self.untracked.clear();
        self.riding.clear();
        self.framed_ahead.clear();
        self.solo = false;
        self.drain_eligible.clear();
    }
}

#[derive(Default)]
struct PendingQueue {
    items: VecDeque<QueuedItem>,
    epoch: u64,
    first_other: Option<(u64, Option<usize>)>,
    first_close: Option<(u64, usize)>,
}

impl PendingQueue {
    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn replace(&mut self, items: VecDeque<QueuedItem>) {
        self.items = items;
        self.epoch = self.epoch.wrapping_add(1);
    }

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

    fn rotate_launch_run_to_back(&mut self, run_len: usize) {
        self.items.rotate_left(run_len);
        self.epoch = self.epoch.wrapping_add(1);
    }

    fn insert_before_closes(&mut self, item: QueuedItem) {
        let index = self.first_close();
        self.items.insert(index, item);
        self.epoch = self.epoch.wrapping_add(1);
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

#[derive(Default)]
struct ScanCache {
    scan: QueueScan,
    taken_at: Option<(u64, bool)>,
}

type SlotBuffer = Vec<Vec<Option<Box<PendingRequest>>>>;

struct SchedulerControl {
    tx: SchedTx,
    active_senders: AtomicUsize,
    shutdown_wait: Condvar,
    shutdown_gate: Mutex<()>,
    /// Each program registered or registering, by its content hash, with
    /// its launch package (a hash collision is refused by name): one
    /// registration a program, however many binds ask for it at once.
    program_ids: Mutex<HashMap<u64, (Known, ::eta_compiler::codegen::launch::LaunchPackage)>>,
    accepting: AtomicBool,
    stats: Arc<SchedulerStats>,
    device_domain: ::engine::MemoryDomain,
    /// The backend a registration's host codegen emits for, off the
    /// engine's thread (`EngineSpec::codegen_backend`).
    codegen_backend: Option<String>,
}

/// What a bind finds of a program by its content hash.
enum Known {
    /// Registered: its id.
    Ready(::engine::ProgramId),
    /// Its registration under way: the answer every bind of it waits for.
    Registering(tokio::sync::watch::Receiver<Option<Answer>>),
}

/// A registration's answer, as each bind waiting on it takes it.
type Answer = std::result::Result<::engine::ProgramId, String>;

/// Answers a registration's waiters when it is dropped unanswered (its
/// task cancelled or panicking): the in-flight entry goes, and each waiter
/// is refused by the program's hash, so a later bind registers it afresh.
struct Unanswered {
    handle: SchedulerHandle,
    hash: u64,
    answer: Option<tokio::sync::watch::Sender<Option<Answer>>>,
}

impl Unanswered {
    fn answer(mut self, answer: Answer) {
        {
            let mut ids = self.handle.inner.program_ids.lock().unwrap();
            match answer {
                Ok(id) => {
                    if let Some(entry) = ids.get_mut(&self.hash) {
                        entry.0 = Known::Ready(id);
                    }
                }
                Err(_) => {
                    ids.remove(&self.hash);
                }
            }
        }
        if let Some(sender) = self.answer.take() {
            let _ = sender.send(Some(answer));
        }
    }
}

impl Drop for Unanswered {
    fn drop(&mut self) {
        if let Some(sender) = self.answer.take() {
            self.handle
                .inner
                .program_ids
                .lock()
                .unwrap()
                .remove(&self.hash);
            let _ = sender.send(Some(Err(format!(
                "program 0x{:016x}: its registration ended unanswered",
                self.hash
            ))));
        }
    }
}

#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    inner: Arc<SchedulerControl>,
}

impl SchedulerHandle {
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

    /// Registers `program`: its host codegen on a blocking thread, then
    /// the lane hands back its compile, which runs on a blocking thread
    /// while the lane fires on, and the lane then makes it the program. Its
    /// instances bind once this answers. Every failure names the program's
    /// hash; an engine's prepare holds nothing its finish must undo, so a
    /// compile that panics leaves nothing behind.
    async fn register(&self, program: ProgramRegistration) -> Result<::engine::ProgramId> {
        let hash = program.program_hash;
        let program = match self.inner.codegen_backend.clone() {
            Some(backend) => crate::rt::spawn_blocking(move || {
                let coded = match crate::pipeline::program::with_host_codegen(
                    &program,
                    Some(backend.as_str()),
                ) {
                    std::borrow::Cow::Owned(coded) => Some(coded),
                    std::borrow::Cow::Borrowed(_) => None,
                };
                coded.unwrap_or(program)
            })
            .await
            .map_err(|e| anyhow!("program 0x{hash:016x}: its host codegen did not finish: {e}"))?,
            None => program,
        };
        let compile = self
            .request(|response| SchedulerItem::PrepareProgram {
                program: Box::new(program),
                response,
            })
            .await?
            .map_err(|e| anyhow!("program 0x{hash:016x}: {e:#}"))?;
        let prepared = crate::rt::spawn_blocking(compile)
            .await
            .map_err(|e| anyhow!("program 0x{hash:016x}: its compile did not finish: {e}"))?;
        self.request(|response| SchedulerItem::FinishProgram { prepared, response })
            .await?
            .map_err(|e| anyhow!("program 0x{hash:016x}: {e:#}"))
    }

    /// The id of `program`, registered once however many binds ask at
    /// once: the first registers it in a task of its own, which a bind's
    /// cancellation does not stop (its finish is always reached, so no
    /// compile is lost and no prepared program is left behind), and every
    /// bind waits on its answer.
    async fn program_id(
        &self,
        pid: ProcessId,
        program: ProgramRegistration,
    ) -> Result<::engine::ProgramId> {
        self.program_id_with(pid, program, |this, program| async move {
            this.register(program).await
        })
        .await
    }

    /// [`SchedulerHandle::program_id`], registering by `register`.
    async fn program_id_with<F, Fut>(
        &self,
        pid: ProcessId,
        program: ProgramRegistration,
        register: F,
    ) -> Result<::engine::ProgramId>
    where
        F: FnOnce(SchedulerHandle, ProgramRegistration) -> Fut,
        Fut: std::future::Future<Output = Result<::engine::ProgramId>> + Send + 'static,
    {
        let hash = program.program_hash;
        let mut answer = {
            let mut ids = self.inner.program_ids.lock().unwrap();
            match ids.get(&hash) {
                Some((_, launch)) if launch != &program.launch => {
                    return Err(anyhow!("program hash collision for 0x{hash:016x}"));
                }
                Some((Known::Ready(id), _)) => return Ok(*id),
                Some((Known::Registering(answer), _)) => answer.clone(),
                None => {
                    let (sender, answer) = tokio::sync::watch::channel(None);
                    ids.insert(
                        hash,
                        (Known::Registering(answer.clone()), program.launch.clone()),
                    );
                    let this = self.clone();
                    let unanswered = Unanswered {
                        handle: self.clone(),
                        hash,
                        answer: Some(sender),
                    };
                    let registering = register(this, program);
                    crate::rt::spawn(async move {
                        let registered = registering.await;
                        unanswered.answer(registered.map_err(|e| format!("{e:#}")));
                    });
                    answer
                }
            }
        };
        let answered = crate::inferlet::process::said(
            "bind: register the program",
            pid,
            answer.wait_for(Option::is_some),
        )
        .await
        .map_err(|_| anyhow!("program 0x{hash:016x}: its registration ended unanswered"))?;
        match &*answered {
            Some(Ok(id)) => Ok(*id),
            Some(Err(why)) => Err(anyhow!("{why}")),
            None => Err(anyhow!(
                "program 0x{hash:016x}: its registration ended unanswered"
            )),
        }
    }

    pub(crate) fn stats(&self) -> Arc<SchedulerStats> {
        Arc::clone(&self.inner.stats)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn submit_prebuilt_tracked_with_copy(
        &self,
        request: crate::engine::FireRequest,
        instance_id: ::engine::InstanceId,
        completion: WorkItemCompletion,
        process_id: ProcessId,
        pipeline_id: ProcessId,
        prelaunch_copy: Option<::engine::KvCopy>,
        prelaunch_state_copy: Option<StateCopy>,
        frame: Option<FrameStamp>,
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
                lora_program,
            ),
        })
    }

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

    pub async fn register_channels_bind(
        &self,
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistration>,
        program: ProgramRegistration,
        mut bind: InstanceBindingPlan,
    ) -> Result<(Vec<RegisteredChannel>, BoundInstance)> {
        let program_id = self
            .program_id(pipeline_id.unwrap_or_default(), program)
            .await?;
        bind.binding.program = program_id;
        // The fire term of the program at these extents counts while the
        // bind is under way, and the activation the fires may take holds it
        // before the bind is sent (`store::shared::programs`). The engine is
        // asked the pair's facts once, at its first bind.
        let _open = match crate::store::registry::try_get(0, bind.engine_id)
            .and_then(|stores| stores.shared)
        {
            Some(shared) => {
                let key = crate::store::shared::programs::FireKey {
                    program: program_id,
                    extents: bind.binding.extents,
                };
                if !shared.lock().unwrap().knows_fire(&key) {
                    self.request(|response| SchedulerItem::FireFacts { key, response })
                        .await??;
                }
                // What the bind adds to the engine's standing guest bytes,
                // asked on the lane before anything is bound: held with the
                // term, refused with it by name, and backed with it below
                // the carve before the bind is sent.
                let standing = self
                    .request(|response| SchedulerItem::StandingAfter {
                        binding: Box::new(bind.binding.clone()),
                        response,
                    })
                    .await??;
                let open = if bind.float {
                    crate::store::shared::programs::ProgramOpen::float(&shared, key, standing)
                } else {
                    crate::store::shared::programs::ProgramOpen::new(&shared, key, standing)
                }
                .map_err(|why| anyhow!("pipeline: {why}"))?;
                let engine = bind.engine_id;
                let pid = pipeline_id.unwrap_or_default();
                crate::inferlet::process::said(
                    "bind: the program's fire span",
                    pid,
                    crate::store::shared::programs::fund_span(
                        &shared,
                        || crate::store::shared::fire_cap(engine),
                        &crate::store::shared::programs::Planner { pid, quorum: pid },
                        &crate::store::shared::programs::LaneMover {
                            engine,
                            epoch_now: || crate::store::shared::programs::epoch_of(engine),
                        },
                    ),
                )
                .await
                .map_err(|why| {
                    anyhow!("pipeline: a program's fires need a wider activation span: {why}")
                })?;
                Some(open)
            }
            None => None,
        };
        let pid = pipeline_id.unwrap_or_default();
        let lane_bind = || {
            let (plans, bind) = (plans.clone(), bind.clone());
            async move {
                crate::inferlet::process::said(
                    "bind: the lane's bind",
                    pid,
                    self.request(|response| SchedulerItem::RegisterChannelsBind {
                        pipeline_id,
                        plans,
                        bind,
                        response,
                    }),
                )
                .await?
            }
        };
        // Suspended meanwhile, its channels parked with it: the bind waits
        // for it to be resident again, as a fire does, never refused.
        let resident = || async move {
            match crate::planner::planner_of(pid) {
                Some(planner) => {
                    crate::inferlet::process::said(
                        "bind: resident again",
                        pid,
                        planner.wait_resident(pid),
                    )
                    .await
                }
                None => Err(crate::planner::PlannerError::Cancelled),
            }
        };
        // Short of a free run for its guest state, it waits on the walk to
        // commit a segment, as a kv admission does, never refused while a
        // segment is left to commit within the budget.
        let engine = bind.engine_id;
        let grown = |prefix: u32, run: u32| async move {
            let shared = crate::store::registry::try_get(0, engine)
                .and_then(|stores| stores.shared)
                .ok_or_else(|| format!("engine {engine} has no shared pool to grow"))?;
            crate::inferlet::process::said(
                "bind: a segment for its guest state",
                pid,
                crate::store::roles::grow_for_guest(&shared, engine, prefix, run),
            )
            .await
        };
        // Short of room the budget holds, with no segment left to commit,
        // it waits for room given back (a process completing, or parked),
        // as an admission waits on pages, never refused for it.
        let room = || {
            crate::store::registry::try_get(0, engine)
                .and_then(|stores| stores.roles)
                .map(|roles| roles.lock().unwrap().room_watch())
        };
        // Whether `pieces` runs of `run` chunks stand free for its state,
        // and the capacity gate lets them in.
        let fits = |run: u32, pieces: u32| {
            crate::store::registry::try_get(0, engine)
                .and_then(|stores| stores.roles)
                .is_some_and(|roles| roles.lock().unwrap().free_runs(run) >= pieces)
                && crate::planner::planner_for(0, engine).is_none_or(|planner| {
                    !planner.capacity_would_hold(pid, run.max(1).saturating_mul(pieces))
                })
        };
        // Holding guest state of its own, it is suspended instead: waiting,
        // it would hold room its own bind lacks.
        let suspend = |run: u32, pieces: u32| {
            let holds = crate::store::registry::try_get(0, engine)
                .and_then(|stores| stores.roles)
                .is_some_and(|roles| roles.lock().unwrap().holds_guest_state(pid));
            holds
                && crate::planner::planner_of(pid)
                    .is_some_and(|planner| planner.suspend_for_room(pid, run, pieces))
        };
        // Waiting, it is a waiter in the planner's queue: idle cache goes
        // first, then a resident is parked for it, as for a restore short
        // of room for its roles.
        let planned = |run: u32, pieces: u32| async move {
            let planner = crate::planner::planner_of(pid)
                .ok_or_else(|| "its process has no planner".to_string())?;
            crate::inferlet::process::said(
                "bind: room for its guest state",
                pid,
                planner.wait_bind_room(pid, run, pieces),
            )
            .await
            .map_err(|why| why.to_string())
        };
        // However the bind ends, the pool no longer counts its need.
        let leave = move || {
            if let Some(roles) =
                crate::store::registry::try_get(0, engine).and_then(|stores| stores.roles)
            {
                roles.lock().unwrap().leave_guest_wait(pid);
            }
        };
        // Waiting at the bind for room, the process is a victim the
        // planner may park for an older head, as an idle one is.
        let waiting = move |waits: bool| {
            if let Some(planner) = crate::planner::planner_of(pid) {
                planner.set_bind_waiting(pid, waits);
            }
        };
        // Before it waits, what its process has in flight lands and gives
        // its leases back: nothing else finalizes it while its guest is in
        // this bind.
        let settle = move || crate::pipeline::fire::settle_process_fires(pid);
        let (registered, _, bound) = crate::store::roles::bind_when_resident(
            pid, lane_bind, resident, grown, room, fits, suspend, planned, leave, waiting, settle,
        )
        .await?;
        Ok((registered, bound))
    }

    pub(crate) fn copy_tracked(&self, plan: PreLaunchCopy) -> Result<ControlCompletion> {
        let completion = ControlCompletion::new();
        self.send(SchedulerItem::CopyTracked {
            plan,
            completion: completion.clone(),
        })?;
        Ok(completion)
    }

    pub(crate) async fn host_kv(&self, op: HostKvOp) -> Result<Vec<u8>> {
        self.request(|response| SchedulerItem::HostKv { op, response })
            .await?
    }

    /// Queues `op` on the lane without waiting on it: what the lane answers
    /// arrives on the receiver.
    pub(crate) fn host_kv_queued(
        &self,
        op: HostKvOp,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<Vec<u8>>>> {
        let (response, receiver) = tokio::sync::oneshot::channel();
        self.send(SchedulerItem::HostKv { op, response })?;
        Ok(receiver)
    }

    /// Between frames, on the lane that owns the engine.
    pub(crate) async fn pool(&self, op: PoolOp) -> Result<()> {
        self.request(|response| SchedulerItem::Pool { op, response })
            .await?
    }

    pub fn close_instance(&self, id: ::engine::InstanceId, pacing_wait_id: u64) -> Result<()> {
        self.send(SchedulerItem::CloseInstance { id, pacing_wait_id })
    }

    pub fn close_channel(&self, id: ::engine::ChannelId) -> Result<()> {
        self.send(SchedulerItem::CloseChannel { id })
    }

    pub fn close_channels(&self, ids: Vec<::engine::ChannelId>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        self.send(SchedulerItem::CloseChannels { ids })
    }
}

pub struct BatchScheduler {
    engine_id: EngineId,
    handle: SchedulerHandle,
    thread: Option<crate::rt::thread::JoinHandle<()>>,
}

impl BatchScheduler {
    pub fn new(
        engine_id: EngineId,
        engine_idx: usize,
        page_size: u32,
        limits: SchedulerLimits,
        frame_size: usize,
    ) -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SchedulerItem>();
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
                codegen_backend: crate::engine::get_spec(engine_idx)
                    .ok()
                    .and_then(|spec| spec.codegen_backend),
            }),
        };
        crate::scheduler::install_scheduler_handle(engine_id, handle.clone());
        let stats_for_loop = Arc::clone(&stats);
        let nudge_tx = handle.inner.tx.clone();
        let thread = crate::rt::thread::Builder::new()
            .name(format!("pie-sched-{engine_idx}"))
            .spawn(move || {
                crate::rt::block_on(Self::run(
                    engine_id,
                    rx,
                    nudge_tx,
                    page_size,
                    limits,
                    stats_for_loop,
                    frame_size,
                ));
            })
            .expect("spawn pie-sched thread");
        Self {
            engine_id,
            handle,
            thread: Some(thread),
        }
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

    async fn run(
        engine_id: EngineId,
        mut rx: SchedRx,
        nudge_tx: SchedTx,
        page_size: u32,
        limits: SchedulerLimits,
        stats: Arc<SchedulerStats>,
        frame_size: usize,
    ) {
        let engine = crate::engine::take_engine_backend(engine_id).ok();
        let mut lane = Lane::spawn(engine_id, engine, Arc::clone(&stats));
        let mut replies = LaneReplies::default();
        let mut instances = HashMap::new();
        let mut pending = PendingQueue::default();
        let mut scan_cache = ScanCache::default();
        let mut slot_buffer = SlotBuffer::new();
        let mut terminated_processes: HashSet<ProcessId> = HashSet::new();
        let mut in_flight_launches = VecDeque::new();
        let mut held_for_placement: Vec<HeldForPlacement> = Vec::new();
        let mut refused_once: HashSet<u32> = HashSet::new();
        let mut batch_want = BatchWant::default();
        let mut row_count = RowCount::default();
        let mut in_flight_control = InFlightControls::default();
        let mut stopping = false;
        let mut frame_policy = FramePolicy::new(
            frame_size,
            limits.max_forward_requests,
            limits.max_forward_tokens,
            Some(Arc::clone(&stats)),
        )
        .with_max_wave_readouts(limits.max_readouts);
        frame_policy
            .preload_free_slots(crate::inferlet::process::execution_slot_capacity().unwrap_or(0));
        let mut stall_since: Option<crate::rt::Instant> = None;
        let mut stall_dumps: u32 = 0;
        // The cap the scheduler last told the walk it holds nothing wider
        // than ([`crate::store::shared::grant_rung`]), and what was said of
        // frames held under the cap.
        let mut granted_cap: Option<u32> = None;
        let mut held_cap_said: Option<u32> = None;
        let mut owed_said = 0usize;

        loop {
            let mut progress = false;
            while let Some(reply) = replies.try_next() {
                Self::apply_lane_reply(
                    reply,
                    &mut in_flight_launches,
                    &mut instances,
                    &mut in_flight_control,
                    &mut frame_policy,
                    &nudge_tx,
                    engine_id,
                );
                progress = true;
            }
            loop {
                let Ok(item) = rx.try_recv() else { break };
                progress = true;
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
            progress |= Self::retire_ready_launches(
                &mut in_flight_launches,
                &mut instances,
                &stats,
                &mut frame_policy,
                &mut held_for_placement,
                &mut refused_once,
                engine_id,
            );
            progress |= Self::release_held(&mut held_for_placement, &mut pending, engine_id);
            progress |= Self::retire_ready_control(&mut in_flight_control);
            // A step's tokens stay within the pool's activation rung (SPEC
            // decision 6): smaller while the system is critical.
            // What is queued and in flight asks the rung of the rows one
            // fire of it takes (REDESIGN 2.5): a decode batch's lane count,
            // a prompt wave's tokens summed over its lanes, so a frame is
            // never cut to a narrower rung for want of asking. A pass asks
            // for its own rows besides ([`crate::store::shared::want_rows`]).
            let queued = pending.items.iter().filter_map(|item| match item {
                QueuedItem::Launch(launch) => Some(&launch.request.request),
                _ => None,
            });
            let flying = in_flight_launches
                .iter()
                .flat_map(|batch: &PendingLaunchBatch| batch.requests.iter())
                .map(|request| &request.request);
            batch_want.ask_for(
                engine_id,
                &mut row_count,
                queued.chain(flying),
                limits.max_forward_tokens,
            );
            let cap = crate::store::shared::fire_cap(engine_id);
            // A grant is of one cap: moved since, it is granted again.
            if granted_cap.is_some_and(|granted| granted != cap) {
                granted_cap = None;
            }
            let mut stepped = limits;
            // Frames queued before the cap came down were cut at the rung
            // the engine still holds: they seal under it, and the rung
            // comes down once they have gone (the grant below waits on
            // them). Capped there, a complete frame sat unsealed on a raise
            // the pressure that lowered the cap could not fund (Mac's beam
            // mix: the rung raised to 64, the cap lowered to 16 under frames
            // cut at 64, nothing submitted for 25 minutes). Past the grant
            // (none wider queued then), nothing wider seals: the rung may
            // be coming down.
            let seal = seal_rows(
                cap,
                crate::store::shared::engine_rows(engine_id),
                granted_cap,
                frame_policy.queued_widest(),
            );
            if seal > cap as usize && held_cap_said != Some(cap) {
                held_cap_said = Some(cap);
                tracing::info!(
                    engine_id,
                    cap,
                    queued = frame_policy.queued_widest(),
                    "the activation rung's cap came down under frames already cut wider: they \
                     seal under the rung the engine holds, and the rung comes down once they \
                     have gone"
                );
            }
            stepped.max_forward_tokens = stepped.max_forward_tokens.min(seal).max(1);
            // No frame past the cap is sealed from here, and no unframed
            // launch past it posted (`post_frame` holds it): once none in
            // flight is either, nor a frame sealed before still to go, the
            // pool may lower its rung.
            frame_policy.cap_fires(stepped.max_forward_tokens);
            // Nor while a complete frame cut wider than the cap waits
            // unsealed (capped): lowered under it, the frame went only once
            // raised again, a raise its owner's eviction could not fund
            // while that eviction waited on the frame (Mac Pro at 2 GiB on
            // 4cef0c7d3 and 6c21ebca3: capped frames, evictions waiting on
            // them, the raise never served). Kept, the walk lifts the cap
            // to the frame with no raise at all.
            if in_flight_launches
                .iter()
                .all(|batch| batch.rung_rows <= cap as usize)
                && frame_policy.sealed_widest() <= cap as usize
                && frame_policy.queued_widest() <= cap as usize
            {
                crate::store::shared::grant_rung(engine_id, cap);
                granted_cap = Some(cap);
            }
            let (dispatched, wait_hint) = Self::dispatch_ready_items(
                &lane,
                &mut replies,
                &mut instances,
                &mut pending,
                &mut in_flight_launches,
                &mut in_flight_control,
                page_size,
                stepped,
                &stats,
                &mut frame_policy,
                &mut scan_cache,
                &mut slot_buffer,
                stopping,
            );
            // Frames held out under the cap are owed a raise: the cap's
            // owner is told, its walk raising the rung to them (a hold with
            // nothing in flight waited on a raise nothing asked for).
            let (owed, owed_by) = frame_policy.cut_owed_by();
            // A frame queued wider after the lower was granted waits on a
            // raise to it: said once a width, never silent.
            if owed > 0 && owed_said != owed {
                tracing::info!(
                    engine_id,
                    owed,
                    cap,
                    owner = ?owed_by,
                    "a frame cut wider than the activation rung's cap waits unsealed for a raise \
                     to it, asked of the cap's owner"
                );
            }
            owed_said = owed;
            crate::store::shared::owe_cut_rows(
                engine_id,
                u32::try_from(owed).unwrap_or(u32::MAX),
                owed_by,
            );
            // The cap this turn sealed under is read again after the hold
            // is told: a cap the walk moved meanwhile (raised back before
            // the report it would have nudged on) is the next turn's at
            // once, never a cached value the scheduler parks on.
            progress |= crate::store::shared::cap_moved(engine_id, cap);
            progress |= dispatched;
            if stopping
                && pending.is_empty()
                && in_flight_launches.is_empty()
                && in_flight_control.is_empty()
                && replies.is_empty()
            {
                break;
            }

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
                && replies.is_empty()
                && !stopping
            {
                // Nothing queued or in flight: the batch's ask goes while
                // the scheduler sleeps.
                batch_want.release();
                lane.parks().parking();
                match rx.recv().await {
                    Some(item) => Some(item),
                    None => {
                        stopping = true;
                        None
                    }
                }
            } else {
                let front_launch =
                    in_flight_launches
                        .front()
                        .and_then(|front| match &front.state {
                            LaunchState::Accepted(completion) => Some(completion.clone()),
                            _ => None,
                        });
                let mut control_settles: Vec<SubmissionCompletion> = in_flight_control
                    .iter()
                    .filter_map(|control| match &control.state {
                        ControlSlotState::Ready(completion) => Some(completion.clone()),
                        _ => None,
                    })
                    .collect();
                let front_failed = in_flight_launches
                    .front()
                    .is_some_and(|front| matches!(front.state, LaunchState::Failed(_)));
                if front_failed
                    || front_launch
                        .as_ref()
                        .is_some_and(SubmissionCompletion::is_settled)
                    || control_settles.iter().any(SubmissionCompletion::is_settled)
                {
                    continue;
                }
                let backstop = Duration::from_millis(250);
                let recv_wait = wait_hint.map(|hold| hold.min(backstop)).unwrap_or(backstop);
                let idle_park = in_flight_launches.is_empty();
                let control_park = idle_park && in_flight_control.holds_launches();
                let park_began = Instant::now();
                let have_replies = !replies.is_empty();
                let have_settle = front_launch.is_some();
                let have_control_settles = !control_settles.is_empty();
                let mut reply_answer = None;
                // The lane stays awake for this turn's launches only until
                // the scheduler parks (`Lane::next_request`).
                lane.parks().parking();
                let parked = {
                    let settle_wait = async {
                        match front_launch {
                            Some(completion) => completion.await,
                            None => std::future::pending().await,
                        }
                    };
                    let control_settle_wait = async {
                        if control_settles.is_empty() {
                            std::future::pending().await
                        } else {
                            let (result, _, _) =
                                futures::future::select_all(control_settles.iter_mut()).await;
                            result
                        }
                    };
                    tokio::select! {
                        mailbox = crate::rt::time::timeout(recv_wait, rx.recv()) => mailbox,
                        reply = replies.waits.next(), if have_replies => {
                            reply_answer = reply;
                            Ok(Some(SchedulerItem::Nudge))
                        }
                        settled = settle_wait, if have_settle => {
                            let _ = settled;
                            Ok(Some(SchedulerItem::Nudge))
                        }
                        settled = control_settle_wait, if have_control_settles => {
                            let _ = settled;
                            Ok(Some(SchedulerItem::Nudge))
                        }
                    }
                };
                if let Some(reply) = reply_answer {
                    Self::apply_lane_reply(
                        reply,
                        &mut in_flight_launches,
                        &mut instances,
                        &mut in_flight_control,
                        &mut frame_policy,
                        &nudge_tx,
                        engine_id,
                    );
                }
                tracing::debug!(
                    ?recv_wait,
                    timed_out = parked.is_err(),
                    slept_us = park_began.elapsed().as_micros() as u64,
                    ?wait_hint,
                    "scheduler parked"
                );
                if parked.is_err() && crate::planner::trace_enabled() {
                    static LAST: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
                    let mut last = LAST.lock().unwrap();
                    let due = last.is_none_or(|at| at.elapsed() >= Duration::from_secs(3));
                    let owed = frame_policy.has_queued_frames()
                        || !pending.is_empty()
                        || !in_flight_launches.is_empty();
                    if due && owed {
                        *last = Some(Instant::now());
                        println!(
                            "[sched-stall t_us={}] pending={} in_flight={} {}",
                            crate::scheduler::fire_timing_now_us(),
                            pending.len(),
                            in_flight_launches.len(),
                            frame_policy.debug_summary()
                        );
                    }
                }
                if idle_park {
                    let slept = park_began.elapsed().as_micros() as u64;
                    use std::sync::atomic::Ordering::Relaxed;
                    if control_park {
                        stats
                            .fire
                            .quorum
                            .idle_park_control_us
                            .fetch_add(slept, Relaxed);
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
                    Ok(Some(item)) => Some(item),
                    Ok(None) => {
                        stopping = true;
                        None
                    }
                    Err(_) => {
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
                            .get_or_insert_with(crate::rt::Instant::now)
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
                                    engine_id,
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
                }
            };

            if let Some(item) = item {
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

        let (mut engine, mut channels) = lane.shutdown();
        Self::shutdown_instances(&mut engine, &mut instances);
        Self::shutdown_channels(&mut engine, &mut channels);
        drop(engine.take());
    }

    #[allow(clippy::too_many_arguments)]
    fn render_debug_dump(
        engine_id: usize,
        pending: &VecDeque<QueuedItem>,
        in_flight_launches: &VecDeque<PendingLaunchBatch>,
        in_flight_control: &InFlightControls,
        instances: &HashMap<::engine::InstanceId, TrackedInstance>,
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
                QueuedItem::PrepareProgram { .. } => "PrepareProgram".to_string(),
                QueuedItem::FinishProgram { .. } => "FinishProgram".to_string(),
                QueuedItem::FireFacts { key, .. } => format!("FireFacts({key})"),
                QueuedItem::StandingAfter { binding, .. } => {
                    format!("StandingAfter(program {})", binding.program)
                }
                QueuedItem::RegisterChannelsBind { .. } => "RegisterChannelsBind".to_string(),
                QueuedItem::CopyTracked { plan, .. } => format!("CopyTracked({})", plan.label()),
                QueuedItem::HostKv { op, .. } => format!("HostKv({op:?})"),
                QueuedItem::Pool { op, .. } => format!("Pool({op:?})"),
                QueuedItem::CloseInstance { id, .. } => format!("CloseInstance {id}"),
                QueuedItem::CloseChannels { ids } => format!("CloseChannels x{}", ids.len()),
            };
            let _ = writeln!(out, "  {line}");
        }
        let _ = writeln!(out, "in_flight_launches ({}):", in_flight_launches.len());
        for batch in in_flight_launches {
            let state = match &batch.state {
                LaunchState::Posted => "posted".to_string(),
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
        // Where each control stands, by whose move it waits on: posted and
        // unanswered, the lane has not taken it (it is in another call,
        // said below); accepted and unsettled, the engine owes its landing
        // (its copy pipeline: a staging ring, a tier write, an event).
        for control in in_flight_control.iter() {
            let age = control.since.elapsed();
            let state = match &control.state {
                ControlSlotState::Posted { id } => {
                    format!("posted(id={id}) {age:?} ago, unanswered: the lane has not taken it")
                }
                ControlSlotState::Ready(c) if c.is_settled() => format!("settled {age:?} after posting"),
                ControlSlotState::Ready(_) => {
                    format!("accepted, unsettled {age:?} after posting: the engine owes its landing")
                }
            };
            let _ = writeln!(
                out,
                "in_flight_control: {} ({}) pipeline {:?} {state}",
                control.operation, control.what, control.pipeline_id,
            );
        }
        let _ = writeln!(out, "{}", super::lane::lane_now(engine_id));
        let _ = write!(out, "{}", frame_policy.debug_summary());
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_item(
        pending: &mut PendingQueue,
        terminated_processes: &mut HashSet<ProcessId>,
        in_flight_control: &mut InFlightControls,
        instances: &HashMap<::engine::InstanceId, TrackedInstance>,
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
            SchedulerItem::Nudge => {}
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
                        if let Some(response) = response {
                            let _ = response.send(());
                        }
                        return;
                    }
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
                        frame_policy.on_lane_leave(pid, owner, false);
                    }
                    LeaveKind::Suspend => {
                        frame_policy.on_process_suspend(pid);
                    }
                    LeaveKind::Terminate => {
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
                    if let Some(stamp) = launch.frame
                        && !launch
                            .process_id
                            .is_some_and(|pid| terminated_processes.contains(&pid))
                    {
                        frame_policy.on_fire_rejected_at_admission(stamp, launch.process_id);
                    }
                    launch.completion.reject_unsubmitted(message);
                } else {
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
                    if wave_trace() {
                        wave_trace_emit(format!(
                            "[wave-trace] t={}us enq fire={} framed={} mask={} masks={} stm={} pipe={}",
                            wave_trace_us(),
                            launch.logical_fire_id,
                            launch.frame.is_some(),
                            launch.request.has_user_mask,
                            has_wire_masks(&launch.request),
                            launch.request.single_token_mode,
                            launch.pipeline_id.is_some()
                        ));
                    }
                    if let Some(stamp) = launch.frame {
                        // Read only for the seal trace: off, no lookup.
                        let program = if frame::seal_trace() {
                            instances
                                .get(&launch.instance_id)
                                .map_or(0, |instance| instance.program_id.0)
                        } else {
                            0
                        };
                        frame_policy.on_fire_reading(
                            stamp,
                            launch.process_id,
                            launch.logical_fire_id,
                            launch.request.tokens(),
                            launch.wire_row_count(),
                            launch.request.sequence_readouts(),
                            launch.request.rung_rows(),
                            program,
                        );
                    }
                    Self::queue_attempt(pending, launch);
                }
            }

            SchedulerItem::ProcessResume(pid) => {
                frame_policy.on_process_resume(pid);
            }
            SchedulerItem::OwnerWaits(pid) => {
                frame_policy.on_owner_waits(pid);
            }
            SchedulerItem::OwnerUnblocked(pid) => {
                frame_policy.on_owner_unblocked(pid);
            }
            SchedulerItem::OwnerAway(pid) => {
                frame_policy.on_owner_away(pid);
            }
            SchedulerItem::FrameTruncate {
                lane,
                seq,
                submitted,
            } => {
                frame_policy.on_frame_truncated(lane, seq, submitted);
            }
            SchedulerItem::LaneOpen { lane, owner } => {
                if !terminated_processes.contains(&owner) && !terminated_processes.contains(&lane) {
                    frame_policy.on_lane_open(lane, owner);
                }
            }
            SchedulerItem::LanePark { lane, seq } => {
                frame_policy.on_lane_park(lane, seq);
            }
            SchedulerItem::PrepareProgram { program, response } => {
                pending.push_back(QueuedItem::PrepareProgram { program, response });
            }
            SchedulerItem::FinishProgram { prepared, response } => {
                pending.push_back(QueuedItem::FinishProgram { prepared, response });
            }
            SchedulerItem::FireFacts { key, response } => {
                pending.push_back(QueuedItem::FireFacts { key, response });
            }
            SchedulerItem::StandingAfter { binding, response } => {
                pending.push_back(QueuedItem::StandingAfter { binding, response });
            }
            SchedulerItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                bind,
                response,
            } => {
                if pipeline_id.is_some_and(|pid| terminated_processes.contains(&pid)) {
                    Lane::release_wait_slots([bind.pacing_wait_id]);
                    let _ = response.send(Err(anyhow!(
                        "process departed before channel bind admission"
                    )));
                    return;
                }
                frame_policy.on_bind_enqueued(pipeline_id);
                pending.insert_before_closes(QueuedItem::RegisterChannelsBind {
                    pipeline_id,
                    plans,
                    bind,
                    response,
                });
            }
            SchedulerItem::CopyTracked { plan, completion } => {
                pending.push_back(QueuedItem::CopyTracked { plan, completion });
            }
            SchedulerItem::Pool { op, response } => {
                pending.push_back(QueuedItem::Pool { op, response });
            }
            SchedulerItem::HostKv { op, response } => {
                pending.push_back(QueuedItem::HostKv { op, response });
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
        }
    }

    fn reject_pipeline_queued(
        pending: &mut PendingQueue,
        pid: ProcessId,
        protected: Option<&WorkItemCompletion>,
    ) {
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

    fn queue_attempt(pending: &mut PendingQueue, mut request: PendingRequest) {
        let mut copies = Vec::with_capacity(2);
        if let Some(plan) = request.prelaunch_copy.take() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::Kv(plan),
                logical_completion: request.completion.clone(),
                process_id: request.process_id,
                pipeline_id: request.pipeline_id,
            });
        }
        if let Some(plan) = request.prelaunch_state_copy.take() {
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

    fn instance_has_queued_work(
        pending: &VecDeque<QueuedItem>,
        instance_id: ::engine::InstanceId,
    ) -> bool {
        pending.iter().any(|item| match item {
            QueuedItem::Launch(request) => request.instance_id == instance_id,
            _ => false,
        })
    }

    const fn standalone_copy(item: &QueuedItem) -> bool {
        matches!(
            item,
            QueuedItem::CopyTracked { .. }
                | QueuedItem::HostKv { .. }
                | QueuedItem::Pool { .. }
                | QueuedItem::PrepareProgram { .. }
                | QueuedItem::FinishProgram { .. }
                | QueuedItem::FireFacts { .. }
                | QueuedItem::StandingAfter { .. }
        )
    }

    const fn rotation_target(item: &QueuedItem) -> bool {
        !matches!(
            item,
            QueuedItem::Launch(_)
                | QueuedItem::CloseInstance { .. }
                | QueuedItem::CloseChannels { .. }
        ) && !Self::standalone_copy(item)
    }

    const fn pipe_concurrent_control(item: &QueuedItem) -> bool {
        Self::standalone_copy(item)
            || matches!(
                item,
                QueuedItem::PreLaunchCopy { .. }
                    | QueuedItem::RegisterChannelsBind { .. }
                    | QueuedItem::CloseChannels { .. }
            )
    }

    const fn lifecycle_control(item: &QueuedItem) -> bool {
        matches!(
            item,
            QueuedItem::RegisterChannelsBind { .. }
                | QueuedItem::CloseInstance { .. }
                | QueuedItem::CloseChannels { .. }
        )
    }

    fn queue_close_channel(pending: &mut PendingQueue, id: ::engine::ChannelId) {
        const CLOSE_CHANNEL_BATCH_MAX: usize = 512;
        if let Some(QueuedItem::CloseChannels { ids }) = pending.back_mut()
            && ids.len() < CLOSE_CHANNEL_BATCH_MAX
        {
            ids.push(id);
        } else {
            pending.push_back(QueuedItem::CloseChannels { ids: vec![id] });
        }
    }

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
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
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
            replies,
            instances,
            pending,
            in_flight_launches,
            in_flight_control,
            page_size,
            limits,
            stats,
            stopping,
        );
        let mut close_rotations = 0usize;
        let mut rotated = false;
        let hold_closes = !stopping && frame_policy.has_pending_binds();
        let slot_blocks_lifecycle = in_flight_control.holds_launches();
        while let Some(item) = pending.front() {
            match item {
                QueuedItem::Launch(_) => {
                    if Self::rotate_launch_for_wave_work(
                        pending,
                        in_flight_control.copies_only(),
                        !slot_blocks_lifecycle,
                    ) {
                        rotated = true;
                        continue;
                    }
                    break;
                }
                QueuedItem::CloseInstance { id, .. } => {
                    if slot_blocks_lifecycle {
                        break;
                    }
                    let id = *id;
                    if hold_closes {
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
                            replies,
                            instances,
                            in_flight_control,
                            item,
                        );
                        progress = true;
                        continue;
                    }
                    // Launches and this close would swap places until its batch
                    // lands, and a browser lands it only after the worker parks.
                    rotated = false;
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
                    let rot_stop = close_rotations >= pending.len()
                        || !pending.iter().skip(1).any(Self::rotation_target);
                    if rot_stop {
                        break;
                    }
                    close_rotations += 1;
                    let item = pending.pop_front().expect("close front");
                    pending.push_back(item);
                }
                _ if !in_flight_control.admits(item) => break,
                _ if !in_flight_launches.is_empty() && !Self::pipe_concurrent_control(item) => {
                    break;
                }
                _ => {
                    let item = pending.pop_front().expect("front item present");
                    Self::post_control(engine_loop, replies, instances, in_flight_control, item);
                    progress = true;
                }
            }
        }
        while in_flight_control.admits_copy() {
            let Some(index) = pending.iter().position(Self::standalone_copy) else {
                break;
            };
            let Some(item) = pending.remove(index) else {
                break;
            };
            Self::post_control(engine_loop, replies, instances, in_flight_control, item);
            progress = true;
        }
        (progress || rotated, wait_hint)
    }

    fn post_control(
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        in_flight_control: &mut InFlightControls,
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
                        instance_id = id.0,
                        error = %message,
                        "scheduler close_instance skipped"
                    );
                    return;
                }
            }
            _ => {}
        }
        let id = replies.next_control_id();
        let holds_launches = !Self::standalone_copy(&item);
        match &item {
            QueuedItem::PreLaunchCopy {
                plan,
                logical_completion,
                process_id,
                pipeline_id,
            } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { id },
                    logical_completion: Some(logical_completion.clone()),
                    process_id: *process_id,
                    pipeline_id: *pipeline_id,
                    tracked_completion: None,
                    operation: plan.label(),
                    holds_launches,
                    prelaunch: true,
                    what: plan.describe(),
                    since: crate::rt::Instant::now(),
                });
            }
            QueuedItem::CopyTracked { completion, plan } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { id },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: Some(completion.clone()),
                    operation: "tracked KV copy",
                    holds_launches,
                    prelaunch: false,
                    what: plan.describe(),
                    since: crate::rt::Instant::now(),
                });
            }
            _ => {}
        }
        replies.post_control(id, engine_loop.control(item));
    }

    fn scan_queue<'a>(
        cache: &'a mut ScanCache,
        pending: &PendingQueue,
        stopping: bool,
    ) -> &'a QueueScan {
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
                    // A launch never overtakes a copy its pipeline queued
                    // ahead of it: the copy fills what the launch reads.
                    // Nor a framed launch its pipeline queued ahead of it:
                    // what that fire writes (its kv, its rings), this one
                    // reads. It rides once that one's frame has gone.
                    let behind_copy = launch.pipeline_id.is_some_and(|pipeline| {
                        scan.blocked_lanes.contains(&pipeline)
                            || scan.framed_ahead.contains(&pipeline)
                    });
                    if launch.framed {
                        scan.queued_ids.push(launch.fire_id);
                        if let Some(pipeline) = launch.pipeline_id {
                            scan.framed_ahead.insert(pipeline);
                        }
                    } else if launch.solo {
                        // Fired alone: the first such, and with no rider.
                        if !behind_copy && scan.untracked.is_empty() {
                            scan.untracked.push(launch.fire_id);
                            scan.riding.insert(None);
                            scan.solo = true;
                        }
                    } else if scan.solo {
                        // Behind a solo fire: the next batch.
                    } else if !behind_copy && scan.riding.insert(launch.pipeline_id) {
                        // Pieces of different prompts' passes (cut at their
                        // checkpoints) share a fire rather than taking one each.
                        scan.untracked.push(launch.fire_id);
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

    #[allow(clippy::too_many_arguments)]
    fn dispatch_frame_work(
        scan_cache: &mut ScanCache,
        slot_buffer: &mut SlotBuffer,
        frame_policy: &mut FramePolicy,
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
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
            if in_flight_control.holds_launches() {
                if wave_trace() {
                    wave_trace_emit(format!(
                        "[wave-trace] t={}us hold control in_flight={}",
                        wave_trace_us(),
                        in_flight_launches.len()
                    ));
                }
                if in_flight_launches.is_empty() {
                    stats
                        .fire
                        .quorum
                        .idle_break_control
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                break;
            }
            if in_flight_launches.len() >= frame::configured_dispatch_depth() {
                if wave_trace() {
                    wave_trace_emit(format!(
                        "[wave-trace] t={}us hold depth in_flight={} depth={}",
                        wave_trace_us(),
                        in_flight_launches.len(),
                        frame::configured_dispatch_depth()
                    ));
                }
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
            if wave_trace() {
                wave_trace_emit(format!(
                    "[wave-trace] t={}us plan in_flight={}",
                    wave_trace_us(),
                    in_flight_launches.len()
                ));
            }
            let mut rider_batch = false;
            let waves: Vec<Vec<u64>> = if stopping {
                if scan.drain_eligible.is_empty() {
                    break;
                }
                vec![scan.drain_eligible.clone()]
            } else if !scan.untracked.is_empty() {
                rider_batch = true;
                if wave_trace() {
                    wave_trace_emit(format!(
                        "[wave-trace] t={}us rider fires={:?}",
                        wave_trace_us(),
                        scan.untracked
                    ));
                }
                vec![scan.untracked.clone()]
            } else {
                match frame_policy.plan_dispatch(
                    &scan.queued_ids,
                    &scan.blocked_lanes,
                    !in_flight_launches.is_empty(),
                    now,
                ) {
                    FramePlan::Dispatch(waves) => {
                        if wave_trace() {
                            wave_trace_emit(format!(
                                "[wave-trace] t={}us dispatch waves={:?}",
                                wave_trace_us(),
                                waves.iter().map(Vec::len).collect::<Vec<_>>()
                            ));
                        }
                        waves
                    }
                    FramePlan::Hold(hold) => {
                        merge_hint(&mut wait_hint, hold);
                        break;
                    }
                    FramePlan::Park => break,
                }
            };
            let (frame_progress, posted) = Self::post_frame(
                slot_buffer,
                frame_policy,
                engine_loop,
                replies,
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
            // What the frame posted owes its output from here, and only that.
            if let Some(batch) = in_flight_launches.back() {
                frame_policy.on_frame_posted(
                    batch
                        .requests
                        .iter()
                        .filter(|request| request.frame.is_some())
                        .filter_map(|request| request.pipeline_id),
                    now,
                );
            }
            if rider_batch {
                frame_policy.record_rider_wave();
            }
        }
        (progress, wait_hint)
    }

    fn admits_to_frame(
        request: &PendingRequest,
        instances: &HashMap<::engine::InstanceId, TrackedInstance>,
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
        frame_policy: &mut FramePolicy,
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        waves: &[Vec<u64>],
    ) -> (bool, bool) {
        let mut progress = false;
        let mut slot_of: HashMap<u64, (usize, usize)> =
            HashMap::with_capacity(waves.iter().map(Vec::len).sum());
        for (index, wave) in waves.iter().enumerate() {
            for (position, &fire_id) in wave.iter().enumerate() {
                slot_of.insert(fire_id, (index, position));
            }
        }
        let mut kept: VecDeque<QueuedItem> = VecDeque::with_capacity(pending.len());
        slot_buffer.resize_with(waves.len(), Vec::new);
        for (slots, wave) in slot_buffer.iter_mut().zip(waves) {
            debug_assert!(slots.iter().all(Option::is_none));
            if slots.len() < wave.len() {
                slots.resize_with(wave.len(), || None);
            }
        }
        let mut collisions: Vec<(usize, Box<PendingRequest>)> = Vec::new();
        while let Some(item) = pending.pop_front() {
            match item {
                // An unframed launch past the pool's activation rung (SPEC
                // decision 6) waits until the rung is back. A framed one was
                // sealed under the cap, and its frame goes whole: the rung is
                // not lowered under a sealed frame (`FramePolicy::cap_fires`).
                QueuedItem::Launch(launch)
                    if !launch.framed
                        && launch.request.request.rung_rows() > limits.max_forward_tokens =>
                {
                    kept.push_back(QueuedItem::Launch(launch));
                }
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
        let mut survivors: Vec<Vec<Box<PendingRequest>>> = Vec::with_capacity(waves.len());
        for slots in slot_buffer.iter_mut() {
            let mut kept_wave = Vec::with_capacity(slots.len());
            for request in slots.iter_mut().filter_map(Option::take) {
                if request.request.lanes.is_empty() {
                    Self::refuse_laneless(&request);
                    progress = true;
                } else if Self::admits_to_frame(&request, instances) {
                    kept_wave.push(request);
                } else {
                    progress = true;
                }
            }
            survivors.push(kept_wave);
        }
        for (wave, request) in collisions {
            if request.request.lanes.is_empty() {
                Self::refuse_laneless(&request);
                progress = true;
            } else if Self::admits_to_frame(&request, instances) {
                survivors[wave].push(request);
            } else {
                progress = true;
            }
        }
        // The device runs the waves in order, and a pipeline's fires in a
        // wave in the order they stand: each pipeline's fires post in the
        // order it submitted them, or the one out of order is refused.
        for wave in &mut survivors {
            wave.retain(|request| {
                let Some(pipeline) = request.pipeline_id else {
                    return true;
                };
                match frame_policy.admit_post(pipeline, request.logical_fire_id) {
                    Ok(()) => true,
                    Err(later) => {
                        let reason = super::frame::out_of_order(pipeline, request.logical_fire_id, later);
                        tracing::error!("{reason}");
                        request.completion.reject_unsubmitted(reason);
                        progress = true;
                        false
                    }
                }
            });
        }
        if survivors.iter().all(Vec::is_empty) {
            return (progress, false);
        }
        // Counted before the build moves the lanes into the fire.
        let total_tokens = survivors
            .iter()
            .flatten()
            .map(|req| req.request.tokens())
            .sum::<usize>();
        let carried = survivors
            .iter()
            .flatten()
            .map(|req| req.request.carried as usize)
            .sum::<usize>();
        let (submission, requests) =
            batch::build_frame_submission(survivors, limits, page_size, stats);
        debug_assert!(
            !submission.steps.is_empty()
                && submission.steps.iter().all(|step| !step.submission.lanes.is_empty()),
            "a fire posted to the engine carries at least one lane in every step"
        );
        let batch_size = requests.len() as u64;
        for request in &requests {
            if let Some(instance) = instances.get_mut(&request.instance_id) {
                instance.in_flight += 1;
            }
        }
        replies.post_launch(engine_loop.fire(submission, total_tokens > batch_size as usize));
        in_flight_launches.push_back(PendingLaunchBatch {
            state: LaunchState::Posted,
            requests,
            started: Instant::now(),
            batch_size,
            total_tokens,
            rung_rows: total_tokens.max(carried),
            placement: Vec::new(),
            unfired: Vec::new(),
        });
        (true, true)
    }

    /// A fire with no lanes is never posted (the engine refuses it whole,
    /// failing every fire beside it): it fails alone, by name. None is
    /// formed so; one that reaches here is a runtime bug.
    fn refuse_laneless(request: &PendingRequest) {
        let reason = format!(
            "scheduler: fire {} of pipeline {:?} carries no lanes (a runtime bug): it fails alone, \
             never posted",
            request.logical_fire_id, request.pipeline_id
        );
        tracing::error!("{reason}");
        request.completion.reject_unsubmitted(reason);
    }

    /// Fires held for a refused chunk's move whose state has moved go
    /// again, ahead of what is queued, as fires of their own (no frame):
    /// their frame went when they were first sent.
    fn release_held(
        held: &mut Vec<HeldForPlacement>,
        pending: &mut PendingQueue,
        engine_id: EngineId,
    ) -> bool {
        let (ready, mut waiting): (Vec<_>, Vec<_>) = std::mem::take(held)
            .into_iter()
            .partition(|batch| !crate::store::shared::refusals_pending(engine_id, &batch.chunks));
        // A move that found no free chunk below for some of its chunks is a
        // need like any other: the planner is asked for that many free
        // chunks, funded as a bind's room (idle cache, parked state down,
        // residents moved), never a silent wait.
        for batch in waiting.iter_mut().filter(|batch| !batch.funding) {
            let stuck = crate::store::shared::refusals_stuck(engine_id, &batch.chunks);
            let owner = batch.requests.iter().find_map(|request| request.process_id);
            if stuck == 0 {
                continue;
            }
            let Some(owner) = owner else {
                continue;
            };
            let Some(planner) = crate::planner::planner_for(0, engine_id) else {
                continue;
            };
            batch.funding = true;
            tracing::debug!(
                engine_id,
                %owner,
                stuck,
                "scheduler: a refused chunk's move lacks free chunks below: the planner makes them"
            );
            crate::rt::spawn(async move {
                if let Err(error) = planner.wait_bind_room(owner, 1, stuck).await {
                    tracing::warn!(%owner, %error, "scheduler: the room for a refused chunk's move was refused");
                }
                crate::store::shared::ask_walk(engine_id);
            });
        }
        *held = waiting;
        if ready.is_empty() {
            return false;
        }
        let mut items = std::mem::take(&mut pending.items);
        for batch in ready.into_iter().rev() {
            for mut request in batch.requests.into_iter().rev() {
                request.frame = None;
                let mut launch = QueuedLaunch::new(request);
                launch.solo = batch.solo;
                items.push_front(QueuedItem::Launch(launch));
            }
        }
        pending.replace(items);
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn retire_ready_launches(
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        stats: &Arc<SchedulerStats>,
        frame_policy: &mut FramePolicy,
        held: &mut Vec<HeldForPlacement>,
        refused_once: &mut HashSet<u32>,
        engine_id: EngineId,
    ) -> bool {
        let mut progress = false;
        while let Some(front) = in_flight_launches.front() {
            let launch_failure = match &front.state {
                LaunchState::Posted => break,
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
            tracing::debug!(
                target: "pie::launch",
                lanes = retired.requests.len(),
                tokens = retired.total_tokens,
                fire_us = retired.started.elapsed().as_micros() as u64,
                "fire retired"
            );
            if wave_trace() {
                wave_trace_emit(format!(
                    "[wave-trace] t={}us retire lanes={}",
                    wave_trace_us(),
                    retired.requests.len()
                ));
            }
            frame_policy.on_frame_retired(retired.requests.iter().filter_map(|r| r.pipeline_id));
            for request in &retired.requests {
                if let Some(instance) = instances.get_mut(&request.instance_id) {
                    instance.in_flight = instance.in_flight.saturating_sub(1);
                }
            }
            if let Some(message) = launch_failure {
                // Refused for where state lies: nothing was read or
                // written. Its state moves down and its fires go again;
                // none fails. Refused again for chunks already moved once,
                // a runtime bug: said loudly, its fires go again one by
                // one, and only one the engine still refuses alone fails.
                if !retired.placement.is_empty() {
                    // Its fires go again by the lanes the refused steps hand
                    // back: the build moved them out of the requests, and a
                    // request fired again without them is a fire with no
                    // lanes. One a refused lane took out of the frame failed
                    // alone already and goes nowhere again.
                    let restored = batch::restore_lanes(
                        &mut retired.requests,
                        std::mem::take(&mut retired.unfired),
                    );
                    let mut again = Vec::with_capacity(retired.requests.len());
                    for (request, back) in retired.requests.into_iter().zip(restored) {
                        if back {
                            again.push(request);
                        } else if !request.completion.is_settled() {
                            debug_assert!(false, "a fire refused for placement came back with no lanes");
                            tracing::error!(
                                engine_id,
                                "scheduler: a fire refused for placement came back with no \
                                 lanes to fire again by (a runtime bug): it fails"
                            );
                            request.completion.reject_unsubmitted(format!(
                                "direct launch rejected: {message} (its lanes were not handed back)"
                            ));
                        }
                    }
                    retired.requests = again;
                    if retired.requests.is_empty() {
                        progress = true;
                        continue;
                    }
                    let repeat = retired.placement.iter().all(|chunk| refused_once.contains(chunk));
                    if !repeat || retired.requests.len() > 1 {
                        if repeat {
                            tracing::error!(
                                engine_id,
                                chunks = ?retired.placement,
                                %message,
                                lanes = retired.requests.len(),
                                "scheduler: the engine refused chunks moved down for it already \
                                 (a runtime bug): its fires go again one by one"
                            );
                        }
                        refused_once.extend(retired.placement.iter().copied());
                        crate::store::shared::evacuate_refused(engine_id, &retired.placement);
                        held.push(HeldForPlacement {
                            chunks: retired.placement,
                            requests: retired.requests,
                            solo: repeat,
                            funding: false,
                        });
                        progress = true;
                        continue;
                    }
                    tracing::error!(
                        engine_id,
                        chunks = ?retired.placement,
                        %message,
                        "scheduler: a fire refused alone for chunks moved down for it already \
                         fails (a runtime bug)"
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
                    // A fire its step resolved as it landed is resolved.
                    crate::engine::completion::settle_batched(|| {
                        for request in retired
                            .requests
                            .iter()
                            .filter(|r| !r.completion.is_settled())
                        {
                            request.completion.resolve_landed("frame settle");
                        }
                    });
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

    fn retire_ready_control(in_flight_control: &mut InFlightControls) -> bool {
        let mut retired = false;
        let mut index = 0;
        while index < in_flight_control.settling.len() {
            let ready = match &in_flight_control.settling[index].state {
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

    fn accept_launch_reply(
        batch: &mut PendingLaunchBatch,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        result: LaunchResult,
    ) {
        match result {
            Ok(completion) => {
                for request in &batch.requests {
                    if let Some(instance) = instances.get_mut(&request.instance_id) {
                        let epoch = instance.next_target_epoch;
                        request.completion.commit_target_epoch(epoch);
                        instance.next_target_epoch = epoch + 1;
                    }
                }
                batch.state = LaunchState::Accepted(completion);
            }
            Err(refusal) => {
                batch.state = LaunchState::Failed(refusal.message);
                batch.placement = refusal.placement;
                batch.unfired = refusal.unfired;
            }
        }
    }

    fn apply_lane_reply(
        reply: LaneReply,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        in_flight_control: &mut InFlightControls,
        frame_policy: &mut FramePolicy,
        rollback_tx: &SchedTx,
        engine_id: EngineId,
    ) {
        match reply {
            LaneReply::Launch(result) => {
                let batch = in_flight_launches
                    .iter_mut()
                    .find(|batch| matches!(batch.state, LaunchState::Posted))
                    .expect("a launch answer arrives only while its batch is posted");
                Self::accept_launch_reply(batch, instances, result);
            }
            LaneReply::Control { id, commit } => Self::apply_control_commit(
                id,
                commit,
                in_flight_control,
                instances,
                frame_policy,
                rollback_tx,
                engine_id,
            ),
        }
    }

    fn apply_control_commit(
        id: u64,
        commit: LaneCommit,
        in_flight_control: &mut InFlightControls,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        frame_policy: &mut FramePolicy,
        rollback_tx: &SchedTx,
        engine_id: EngineId,
    ) {
        match commit {
            LaneCommit::None => {}
            LaneCommit::BindFinished { pipeline_id, waits } => {
                frame_policy.on_bind_completed(pipeline_id);
                // Waiting for room, its process submits nothing until the
                // bind lands: its lanes go out of the frame, as a
                // suspension's do, and come back with its next submit.
                if waits && let Some(pid) = pipeline_id {
                    frame_policy.on_owner_waits(pid);
                }
            }
            LaneCommit::BindInstance {
                pipeline_id,
                bound,
                respond:
                    BindRespond {
                        registered,
                        program_id,
                        response,
                    },
            } => {
                frame_policy.on_bind_completed(pipeline_id);
                // Bound: whatever wait its refusals marked it blocked for
                // has ended.
                if let Some(pid) = pipeline_id {
                    frame_policy.on_owner_unblocked(pid);
                }
                if instances.contains_key(&bound.instance_id) {
                    tracing::error!(
                        instance_id = bound.instance_id.0,
                        "bind committed an already-bound instance id"
                    );
                    let error = anyhow!("instance {} is already bound", bound.instance_id);
                    let _ = response.send(Err(error));
                    return;
                }
                let instance_id = bound.instance_id;
                instances.insert(instance_id, TrackedInstance::from_bound(&bound));
                if let Err(Ok((registered, _, bound))) =
                    response.send(Ok((registered, program_id, bound)))
                {
                    tracing::warn!(
                        operation = "register_channels_bind",
                        instance_id = bound.instance_id.0,
                        channel_count = registered.len(),
                        "scheduler cancellation rollback enqueued bound instance and channels"
                    );
                    Lane::release_registered_channel_wait_slots(&registered);
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
                            instance_id = instance_id.0,
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
                                channel_id = channel_id.0,
                                "scheduler cancellation rollback close_channel enqueue failed"
                            );
                        }
                    }
                }
            }
            LaneCommit::CloseInstance { id } => {
                if let Some(instance) = instances.remove(&id) {
                    instance.close_wait_slots();
                }
                // The lane dropped the instance's count; the guest term is
                // built here, off the lane and outside the pool's lock.
                if let Some(shared) =
                    crate::store::registry::try_get(0, engine_id).and_then(|stores| stores.shared)
                {
                    crate::store::shared::programs::fit_fire(&shared);
                }
            }
            LaneCommit::AsyncControl { result } => {
                let Some(index) = in_flight_control.position_posted(id) else {
                    tracing::error!(
                        id,
                        "lane async-control reply without a matching control slot"
                    );
                    return;
                };
                match result {
                    Ok(completion) => {
                        in_flight_control.settling[index].state =
                            ControlSlotState::Ready(completion);
                    }
                    // Refused on the lane (or the lane dropped it): what
                    // waits on the copy is told now, never left waiting on
                    // a completion that cannot come (a residency move held
                    // its items moving for good).
                    Err(error) => {
                        let pending = in_flight_control.settling.remove(index);
                        if let Some(tracked) = pending.tracked_completion.as_ref() {
                            tracked.resolve(&Err(anyhow!("{error}")));
                        }
                        if let Some(logical) = pending.logical_completion.as_ref() {
                            logical.reject_unsubmitted(format!(
                                "pre-launch {} failed: {error}",
                                pending.operation
                            ));
                        }
                    }
                }
            }
        }
    }

    fn shutdown_instances(
        engine: &mut Option<EngineBox>,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
    ) {
        let outstanding = std::mem::take(instances);
        for (instance_id, instance) in outstanding {
            if let Some(engine) = engine.as_mut()
                && let Err(err) = engine.close_instance(instance_id)
            {
                tracing::warn!(
                    instance_id = instance_id.0,
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
                tracing::warn!(
                    channel_id = channel_id.0,
                    ?err,
                    "scheduler shutdown close_channel failed"
                );
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
    /// The program it runs, as the seal trace names a fire's.
    program_id: ::engine::ProgramId,
    pacing_wait_id: u64,
    wait_slots: Arc<crate::engine::BoundWaitSlots>,
    in_flight: usize,
    next_target_epoch: u64,
}

impl TrackedInstance {
    fn from_bound(bound: &BoundInstance) -> Self {
        Self {
            program_id: bound.program_id,
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

    struct PanickingEngine;

    impl engine::Engine for PanickingEngine {
        /// A test engine's copies land inside the call: nothing to wait for.
        fn settle_copies(&mut self) -> ::engine::Result<()> {
            Ok(())
        }

        /// A test engine's copies complete inside the call: their sources are
        /// read by the time it returns.
        fn copy_slots_landing(
            &mut self,
            copy: &::engine::chunk::SlotCopy,
            _token: u64,
        ) -> ::engine::Result<bool> {
            self.copy_slots(copy)?;
            Ok(true)
        }

        fn kind(&self) -> &'static str {
            "panicking"
        }

        fn prepare_program(
            &mut self,
            _registration: &::engine::program::ProgramRegistration,
        ) -> ::engine::Result<::engine::ProgramCompile> {
            Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
        }

        fn finish_program(
            &mut self,
            _prepared: ::engine::Prepared,
        ) -> ::engine::Result<::engine::ProgramId> {
            Err(::engine::Error::unsupported(self.kind(), "finish_program"))
        }

        fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
            Err(::engine::Error::unsupported(self.kind(), "close_program"))
        }

        fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
            Err(engine::Error::Load("no model".into()))
        }

        fn submit(
            &mut self,
            _frame: &engine::FrameSubmission,
        ) -> engine::Result<engine::FrameTicket> {
            panic!("the shape of an interpreter reading lane zero of an empty cell");
        }
    }

    /// A batch of decodes asks the rung of its lane count while it is
    /// queued or in flight, and asks nothing once none is.
    #[test]
    fn a_batch_asks_the_rung_of_its_rows_while_it_lasts() {
        const ENGINE: usize = 9201;
        let mut want = BatchWant::default();
        want.ask(ENGINE, 16);
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 16);
        want.ask(ENGINE, 64);
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 64);
        // A batch that breathes keeps what it asked until the scheduler
        // sleeps with nothing to do.
        want.ask(ENGINE, 32);
        want.ask(ENGINE, 0);
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 64);
        want.release();
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
    }

    fn fire_of(seats: std::ops::Range<u32>, rows: usize) -> crate::engine::FireRequest {
        crate::engine::FireRequest {
            lanes: seats
                .map(|slot| {
                    ::engine::Lane {
                        slot,
                        tokens: vec![7; rows],
                        ..Default::default()
                    }
                    .into()
                })
                .collect(),
            ..Default::default()
        }
    }

    /// A batch asks the rows its fire carries: a lane queued in several
    /// pipelined launches, and in flight in another, counts once.
    #[test]
    fn a_lane_s_pipelined_launches_ask_its_one_row() {
        let fire = |seats: std::ops::Range<u32>| fire_of(seats, 1);
        let launches: Vec<_> = (0..16).map(|_| fire(0..64)).collect();
        let mut count = RowCount::default();
        assert_eq!(count.count(&launches), 64);
        // Two batches of 32 lanes each, each pipelined: 64 rows.
        let halves: Vec<_> = (0..8).flat_map(|_| [fire(0..32), fire(32..64)]).collect();
        assert_eq!(count.count(&halves), 64);
        // The count starts fresh each turn.
        assert_eq!(count.count(&[fire(0..8)]), 8);
        // Seatless lanes count in the launch carrying the most of them.
        let mut seatless = fire(0..4);
        for lane in &mut seatless.lanes {
            lane.kv_less = true;
        }
        assert_eq!(count.count(&[seatless.clone(), seatless, fire(0..8)]), 12);
        // A decode whose token the device resolves carries none and still
        // takes its row.
        let mut resolved = fire(0..4);
        for lane in &mut resolved.lanes {
            lane.tokens.clear();
        }
        assert_eq!(count.count(&[resolved]), 4);
    }

    /// A wave of prompts asks the rung for the rows one fire of it takes,
    /// its tokens summed over its lanes, not its lane count: 40 prompts of
    /// 125 rows ask 5000, no more than a step carries.
    #[test]
    fn a_prompt_wave_asks_the_rows_of_its_lanes_not_their_count() {
        const ENGINE: usize = 9202;
        let wave: Vec<_> = (0..40).map(|seat| fire_of(seat..seat + 1, 125)).collect();
        let mut count = RowCount::default();
        assert_eq!(count.count(&wave), 5000);
        for max_forward_tokens in [1024, 8192] {
            let mut want = BatchWant::default();
            want.ask_for(ENGINE, &mut count, &wave, max_forward_tokens);
            assert_eq!(
                crate::store::shared::wanted_rows(ENGINE) as usize,
                5000usize.min(max_forward_tokens),
            );
            // Nothing queued or in flight, the scheduler sleeps: the ask goes.
            want.release();
            assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
        }
        // A lane's prompt and the decodes pipelined behind it take the
        // prompt's rows in a fire, once; a verify its drafted rows.
        let lane = [fire_of(0..1, 125), fire_of(0..1, 1), fire_of(0..1, 1)];
        assert_eq!(count.count(&lane), 125);
        let verify = [fire_of(0..8, 4), fire_of(0..8, 1)];
        assert_eq!(count.count(&verify), 32);
        // Rows a launch carries past its own count as the widest such.
        let mut carrying = fire_of(0..2, 1);
        carrying.carried = 40;
        assert_eq!(count.count(&[carrying, fire_of(2..6, 1)]), 6 + 38);
    }

    /// Decode-only lanes still ask their lane count.
    #[test]
    fn a_decode_batch_asks_its_lane_count() {
        const ENGINE: usize = 9203;
        let decodes: Vec<_> = (0..4).map(|_| fire_of(0..48, 1)).collect();
        let mut count = RowCount::default();
        let mut want = BatchWant::default();
        want.ask_for(ENGINE, &mut count, &decodes, 1024);
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 48);
        // An empty queue asks nothing new; the scheduler's sleep releases.
        want.ask_for(ENGINE, &mut count, &[], 1024);
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 48);
        want.release();
        assert_eq!(crate::store::shared::wanted_rows(ENGINE), 0);
    }

    /// A control the lane refuses or drops (a panic on the copy, the
    /// reply lost) tells what waits on it: a residency move's copy waited
    /// on a completion that never came, and the move held its items moving
    /// for good (Mac Pro's M5: every restore of a pair's other holders
    /// waited on it).
    #[test]
    fn a_refused_control_tells_its_tracked_completion() {
        let tracked = crate::scheduler::ControlCompletion::new();
        let mut controls = InFlightControls::default();
        controls.settling.push(PendingControl {
            state: ControlSlotState::Posted { id: 41 },
            logical_completion: None,
            process_id: None,
            pipeline_id: None,
            tracked_completion: Some(tracked.clone()),
            operation: "tracked roles copy",
            holds_launches: false,
            prelaunch: false,
            what: String::new(),
            since: crate::rt::Instant::now(),
        });
        let (rollback, _rx) = tokio::sync::mpsc::unbounded_channel();
        BatchScheduler::apply_control_commit(
            41,
            LaneCommit::AsyncControl {
                result: Err("the engine lane dropped this control unanswered".to_string()),
            },
            &mut controls,
            &mut HashMap::new(),
            &mut FramePolicy::new(1, 1, 1, None),
            &rollback,
            0,
        );
        assert!(controls.settling.is_empty());
        let told = crate::rt::block_on(async {
            crate::rt::time::timeout(Duration::from_secs(5), tracked.wait()).await
        })
        .expect("told, never left waiting");
        let error = told.expect_err("told it failed");
        assert!(error.to_string().contains("unanswered"), "{error}");
    }

    /// Fire `fire`'s request: one decode lane in slot `fire`, a page of its own.
    fn refused_request(fire: u64) -> Box<PendingRequest> {
        let slot = u32::try_from(fire).expect("a test fire id");
        Box::new(PendingRequest {
            lora_program: false,
            logical_fire_id: fire,
            request: crate::engine::FireRequest::one(crate::engine::fire::lane_of(
                slot,
                vec![100 + slot],
                3,
                vec![slot],
            )),
            instance_id: ::engine::InstanceId(1),
            completion: crate::engine::WorkItemCompletion::new(1, 0),
            process_id: None,
            pipeline_id: Some(ProcessId::from_u128(u128::from(fire))),
            prelaunch_copy: None,
            prelaunch_state_copy: None,
            frame: Some(FrameStamp {
                lane: ProcessId::from_u128(u128::from(fire)),
                seq: 0,
                slot: 0,
                fires: 1,
            }),
        })
    }

    fn refused_limits() -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 64,
            max_page_refs: 64,
            max_context: 0,
            prefill_rows: 0,
            prefill_lanes: 0,
            max_readouts: 0,
            readout_store: 0,
        }
    }

    /// `requests` built into a frame the engine refused for where `chunks`
    /// lie before any step ran: its steps come back with the refusal, as
    /// the lane hands them (`LaunchRefusal::unfired`).
    fn refused_frame(requests: Vec<Box<PendingRequest>>, chunks: &[u32]) -> PendingLaunchBatch {
        let fires = requests.len();
        let (frame, requests) = batch::build_frame_submission(
            vec![requests],
            refused_limits(),
            16,
            &SchedulerStats::default(),
        );
        PendingLaunchBatch {
            state: LaunchState::Failed(
                "impossible submission: state chunk 2057 lies inside the carve".into(),
            ),
            requests,
            started: Instant::now(),
            batch_size: fires as u64,
            total_tokens: fires,
            rung_rows: fires,
            placement: chunks.to_vec(),
            unfired: frame.steps,
        }
    }

    fn refused_batch(fires: &[u64], chunks: &[u32]) -> PendingLaunchBatch {
        refused_frame(fires.iter().map(|&fire| refused_request(fire)).collect(), chunks)
    }

    /// beamx40 on M1 (6ea0b14ef): a frame refused for where state lies went
    /// again from requests the build had moved its lanes out of, and the
    /// engine refused it whole, "a fire carries no lanes", poisoning a
    /// rescore pipeline's channel. A fire refused for placement goes again
    /// with every lane, page and token it was built with; one a refused
    /// lane took out of the frame (failed alone) does not go again; and a
    /// fire with no lanes is never posted, failing alone by name.
    fn a_frame_refused_for_placement_goes_again_with_every_lane_it_was_built_with() {
        let engine = 9_998;
        let stats = Arc::new(SchedulerStats::default());
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        let (mut held, mut refused_once) = (Vec::new(), HashSet::new());
        let built: Vec<_> = [7, 8, 9].map(refused_request).into_iter().collect();
        // What the build stamps (the fires after it in its frame) aside.
        let lanes_of = |request: &PendingRequest| {
            let mut lanes = request.request.lanes.clone();
            for lane in &mut lanes {
                lane.fires_after = None;
            }
            lanes
        };
        let before: Vec<_> = built.iter().map(|request| lanes_of(request)).collect();
        // Fire 9 rode the frame, but a refused lane took it out (failed
        // alone) before the placement refusal: it is in no step.
        let gone = refused_request(9);
        gone.completion.reject_unsubmitted("engine: lane 2 refused".to_string());
        let mut batch = refused_frame(built.into_iter().take(2).collect(), &[2057]);
        assert!(
            batch.requests.iter().all(|request| request.request.lanes.iter().all(|lane| lane.tokens.is_empty())),
            "the build moved the lanes into the steps"
        );
        batch.requests.push(gone);
        let mut flight = VecDeque::from([batch]);
        assert!(BatchScheduler::retire_ready_launches(
            &mut flight,
            &mut HashMap::new(),
            &stats,
            &mut policy,
            &mut held,
            &mut refused_once,
            engine,
        ));
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].requests.len(), 2, "the fire failed alone goes nowhere again");
        let mut pending = PendingQueue::default();
        assert!(BatchScheduler::release_held(&mut held, &mut pending, engine));
        let again: Vec<Box<PendingRequest>> = std::mem::take(&mut pending.items)
            .into_iter()
            .map(|item| match item {
                QueuedItem::Launch(launch) => launch.into_request(),
                _ => panic!("only launches go again"),
            })
            .collect();
        for (request, lanes) in again.iter().zip(&before) {
            assert!(!request.completion.is_settled(), "none failed");
            assert_eq!(&lanes_of(request), lanes, "every lane, page and token it was built with");
        }
        let (frame, _) =
            batch::build_frame_submission(vec![again], refused_limits(), 16, &SchedulerStats::default());
        assert!(!frame.steps.is_empty());
        assert!(
            frame.steps.iter().all(|step| step.submission.validate().is_ok()),
            "the engine takes the fire it goes again as"
        );

        // A request with no lanes reaching the post fails alone, by name,
        // and what rides beside it is posted.
        let mut laneless = refused_request(11);
        laneless.request.lanes.clear();
        laneless.frame = None;
        let laneless_completion = laneless.completion.clone();
        let mut beside = refused_request(12);
        beside.frame = None;
        let instance = ::engine::InstanceId(1);
        let bound = crate::engine::BoundInstance::new(
            &::engine::BoundInstance {
                id: instance,
                program: ::engine::ProgramId::default(),
                geometry: eta_ir::registry::GeometryClass::default(),
                standing_guest_bytes: 0,
            },
            0,
        );
        let mut instances = HashMap::from([(instance, TrackedInstance::from_bound(&bound))]);
        let mut pending = PendingQueue::default();
        pending.push_back(QueuedItem::Launch(QueuedLaunch::new(laneless)));
        pending.push_back(QueuedItem::Launch(QueuedLaunch::new(beside)));
        let mut lane = Lane::spawn(9_997, None, Arc::clone(&stats));
        let mut replies = LaneReplies::default();
        let mut posted = VecDeque::new();
        let (progress, fired) = BatchScheduler::post_frame(
            &mut SlotBuffer::new(),
            &mut FramePolicy::new(1, 64, 4096, None),
            &lane,
            &mut replies,
            &mut instances,
            &mut pending,
            &mut posted,
            16,
            refused_limits(),
            &stats,
            &[vec![11, 12]],
        );
        assert!(progress && fired);
        assert!(laneless_completion.is_settled(), "the fire with no lanes fails alone");
        assert_eq!(posted.len(), 1);
        assert_eq!(posted[0].requests.len(), 1, "only the fire beside it is posted");
        lane.shutdown();
    }

    /// Mac Pro at 5 GiB on 6c21ebca3: one launch refused "impossible
    /// submission: state chunk 2057 ... inside the activation span's backed
    /// carve" failed all 42 lanes of its frame and poisoned their channels.
    /// A batch the engine refuses for where state lies is held, nothing
    /// failed, while the pool moves that state down, then fired again as
    /// fires of their own; refused again for a chunk already moved, its
    /// fires go again one by one, and only one the engine still refuses
    /// alone fails.
    fn a_launch_refused_for_placement_is_fired_again_after_the_move_failing_none() {
        let engine = 9_999;
        let stats = Arc::new(SchedulerStats::default());
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        let (mut held, mut refused_once) = (Vec::new(), HashSet::new());
        let mut pending = PendingQueue::default();
        let mut flight = VecDeque::from([refused_batch(&[1, 2, 3], &[2057])]);
        let mut retire = |flight: &mut VecDeque<PendingLaunchBatch>,
                          held: &mut Vec<HeldForPlacement>,
                          refused_once: &mut HashSet<u32>| {
            BatchScheduler::retire_ready_launches(
                flight,
                &mut HashMap::new(),
                &stats,
                &mut policy,
                held,
                refused_once,
                engine,
            )
        };
        assert!(retire(&mut flight, &mut held, &mut refused_once));
        assert_eq!(held.len(), 1, "held for its state's move");
        assert!(!held[0].solo);
        let unsettled = |pending: &PendingQueue| {
            pending.items.iter().all(|item| match item {
                QueuedItem::Launch(launch) => !launch.request.completion.is_settled(),
                _ => true,
            })
        };
        // No pool here: nothing is pending its move, so they go again.
        assert!(BatchScheduler::release_held(&mut held, &mut pending, engine));
        assert_eq!(pending.len(), 3, "fired again");
        assert!(unsettled(&pending), "none failed");
        assert!(
            pending.items.iter().all(|item| matches!(
                item,
                QueuedItem::Launch(launch) if !launch.framed && !launch.solo
            )),
            "as fires of their own, riding together"
        );
        // Refused again for the chunk already moved: one by one.
        let mut flight = VecDeque::from([refused_batch(&[4, 5], &[2057])]);
        assert!(retire(&mut flight, &mut held, &mut refused_once));
        assert!(held[0].solo, "a runtime bug: its fires go again alone");
        let mut pending = PendingQueue::default();
        BatchScheduler::release_held(&mut held, &mut pending, engine);
        assert!(pending.items.iter().all(|item| matches!(item, QueuedItem::Launch(launch) if launch.solo)));
        assert!(unsettled(&pending));
        // One refused alone again fails, alone.
        let alone = refused_batch(&[4], &[2057]);
        let completion = alone.requests[0].completion.clone();
        let mut flight = VecDeque::from([alone]);
        retire(&mut flight, &mut held, &mut refused_once);
        assert!(held.is_empty());
        assert!(completion.is_settled(), "the one the engine still refuses alone fails");
    }

    #[test]
    fn worker_every_case() {
        a_launch_refused_for_placement_is_fired_again_after_the_move_failing_none();
        a_frame_refused_for_placement_goes_again_with_every_lane_it_was_built_with();
        a_panicking_engine_fails_its_launch_instead_of_leaving_it_in_flight();
        a_retryable_refusal_past_admission_fails_by_name_instead_of_replaying();
        a_launch_waits_for_the_copy_queued_ahead_of_it();
        an_unframed_piece_waits_for_its_pipelines_framed_fire_queued_ahead();
        a_pass_that_only_reorders_lets_the_worker_park();
        what_a_frame_settled_inside_submit_wrote_is_kept_when_it_is_answered();
        a_step_wider_than_the_rung_the_engine_holds_never_reaches_it();
    }

    /// The lane refuses by name a step whose token rows (or lanes) pass the
    /// rung the engine holds, before the engine sees it (this engine panics
    /// on any submit); one within it goes on to the engine.
    fn a_step_wider_than_the_rung_the_engine_holds_never_reaches_it() {
        const ENGINE: usize = 9207;
        crate::store::shared::set_engine_rows_for_test(ENGINE, 2);
        let mut lane = Lane::spawn(
            ENGINE,
            Some(Box::new(PanickingEngine)),
            Arc::new(SchedulerStats::default()),
        );
        let frame = |lanes: u32| crate::engine::FrameFire {
            steps: vec![crate::engine::StepFire {
                submission: ::engine::Step {
                    lanes: (0..lanes)
                        .map(|lane| ::engine::Lane::decode(lane, 0, 1, 0))
                        .collect(),
                    attachments: Vec::new(),
                    media: Vec::new(),
                    voxels: Vec::new(),
                },
                terminal_cells: Vec::new(),
                completions: Vec::new(),
                member_lanes: Vec::new(),
                instances: vec![::engine::InstanceId(0); lanes as usize],
            }],
        };
        let answer = |lanes: u32| {
            let reply = lane.fire(frame(lanes), false);
            crate::rt::block_on(async {
                crate::rt::time::timeout(Duration::from_secs(10), reply).await
            })
            .expect("answered")
            .expect("answered")
        };
        let Err(wide) = answer(3) else {
            panic!("past the rung: refused");
        };
        assert!(
            wide.message.contains("3 token rows")
                && wide.message.contains("the rung the engine was told"),
            "refused by name: {}",
            wide.message
        );
        let Err(narrow) = answer(2) else {
            panic!("the engine panics on what reaches it");
        };
        assert!(narrow.message.contains("panic"), "it reached the engine: {}", narrow.message);
        let _ = lane.shutdown();
    }

    /// A browser's single thread lands the batch a close waits on only after
    /// the worker parks, so a pass that reports progress here never ends.
    fn a_pass_that_only_reorders_lets_the_worker_park() {
        let instance = ::engine::InstanceId(1);
        let mut pending = PendingQueue::default();
        pending.push_back(QueuedItem::CloseInstance {
            id: instance,
            pacing_wait_id: 0,
        });
        pending.push_back(QueuedItem::Launch(QueuedLaunch::new(Box::new(
            PendingRequest {
                lora_program: false,
                logical_fire_id: 9,
                request: crate::engine::FireRequest::default(),
                instance_id: instance,
                completion: crate::engine::WorkItemCompletion::new(1, 0),
                process_id: None,
                pipeline_id: None,
                prelaunch_copy: None,
                prelaunch_state_copy: None,
                frame: Some(FrameStamp {
                    lane: ProcessId::from_u128(5),
                    seq: 0,
                    slot: 0,
                    fires: 1,
                }),
            },
        ))));
        let stats = Arc::new(SchedulerStats::default());
        let mut lane = Lane::spawn(0, Some(Box::new(PanickingEngine)), Arc::clone(&stats));
        let (progress, _) = BatchScheduler::dispatch_ready_items(
            &lane,
            &mut LaneReplies::default(),
            &mut HashMap::new(),
            &mut pending,
            &mut VecDeque::new(),
            &mut InFlightControls::default(),
            16,
            SchedulerLimits {
                max_forward_requests: 8,
                max_forward_tokens: 64,
                max_page_refs: 64,
                max_context: 0,
                prefill_rows: 0,
                prefill_lanes: 0,
                max_readouts: 0,
                readout_store: 0,
            },
            &stats,
            &mut FramePolicy::new(1, 64, 4096, None),
            &mut ScanCache::default(),
            &mut SlotBuffer::new(),
            false,
        );
        assert!(!progress, "nothing left the queue");
        assert_eq!(pending.len(), 2);
        lane.shutdown();
    }

    fn a_launch_waits_for_the_copy_queued_ahead_of_it() {
        let pipeline = ProcessId::from_u128(5);
        let request = PendingRequest {
            lora_program: false,
            logical_fire_id: 9,
            request: crate::engine::FireRequest::default(),
            instance_id: ::engine::InstanceId(1),
            completion: crate::engine::WorkItemCompletion::new(1, 0),
            process_id: None,
            pipeline_id: Some(pipeline),
            prelaunch_copy: None,
            prelaunch_state_copy: Some(StateCopy::default()),
            frame: None,
        };
        let mut pending = PendingQueue::default();
        BatchScheduler::queue_attempt(&mut pending, request);
        let mut cache = ScanCache::default();
        let scan = BatchScheduler::scan_queue(&mut cache, &pending, false);
        assert!(scan.untracked.is_empty(), "behind its copy");
        // The copy is dispatched: the launch rides.
        let copy = pending.pop_front();
        assert!(matches!(copy, Some(QueuedItem::PreLaunchCopy { .. })));
        let mut cache = ScanCache::default();
        let scan = BatchScheduler::scan_queue(&mut cache, &pending, false);
        assert_eq!(scan.untracked, [9]);
    }

    /// A cut pass in slot 1 sends its leading pieces unframed: they never
    /// ride ahead of slot 0 of the same pipeline, queued framed before them
    /// and waiting for its frame. Another pipeline's piece still rides; once
    /// slot 0's frame has gone, the piece rides too.
    fn an_unframed_piece_waits_for_its_pipelines_framed_fire_queued_ahead() {
        let (pipeline, other) = (ProcessId::from_u128(5), ProcessId::from_u128(6));
        let launch = |fire: u64, pipeline: ProcessId, framed: bool| PendingRequest {
            lora_program: false,
            logical_fire_id: fire,
            request: crate::engine::FireRequest::default(),
            instance_id: ::engine::InstanceId(fire),
            completion: crate::engine::WorkItemCompletion::new(fire, 0),
            process_id: None,
            pipeline_id: Some(pipeline),
            prelaunch_copy: None,
            prelaunch_state_copy: None,
            frame: framed.then_some(FrameStamp {
                lane: pipeline,
                seq: 0,
                slot: 0,
                fires: 2,
            }),
        };
        let mut pending = PendingQueue::default();
        BatchScheduler::queue_attempt(&mut pending, launch(1, pipeline, true));
        BatchScheduler::queue_attempt(&mut pending, launch(2, pipeline, false));
        BatchScheduler::queue_attempt(&mut pending, launch(3, other, false));
        let mut cache = ScanCache::default();
        let scan = BatchScheduler::scan_queue(&mut cache, &pending, false);
        assert_eq!(scan.untracked, [3], "slot 1's piece waits behind slot 0");
        assert_eq!(scan.queued_ids, [1].into_iter().collect());
        // Slot 0's frame is posted: the piece rides.
        assert!(matches!(pending.pop_front(), Some(QueuedItem::Launch(_))));
        let mut cache = ScanCache::default();
        let scan = BatchScheduler::scan_queue(&mut cache, &pending, false);
        assert_eq!(scan.untracked, [2, 3]);
    }

    fn a_panicking_engine_fails_its_launch_instead_of_leaving_it_in_flight() {
        let mut lane = Lane::spawn(
            0,
            Some(Box::new(PanickingEngine)),
            Arc::new(SchedulerStats::default()),
        );

        let mut answers = Vec::new();
        for token in [7_u64, 8] {
            let answer_rx = lane.fire(
                crate::engine::FrameFire {
                    steps: vec![crate::engine::StepFire {
                        submission: ::engine::Step {
                            lanes: vec![::engine::Lane::decode(0, 0, 1, 0)],
                            attachments: Vec::new(),
                            media: Vec::new(),
                            voxels: Vec::new(),
                        },
                        terminal_cells: Vec::new(),
                        completions: Vec::new(),
                        member_lanes: Vec::new(),
                        instances: vec![::engine::InstanceId(0)],
                    }],
                },
                false,
            );
            answers.push((token, answer_rx));
        }

        for (want, answer_rx) in answers {
            let result = crate::rt::block_on(async {
                crate::rt::time::timeout(Duration::from_secs(10), answer_rx).await
            })
            .unwrap_or_else(|_| panic!("token {want} was never answered"))
            .unwrap_or_else(|_| panic!("token {want} was never answered"));
            let Err(err) = result else {
                panic!("a panicking engine cannot have launched anything");
            };
            assert!(
                err.contains("panic"),
                "the failure must say what happened, so an operator restarts \
                 rather than retries: {err}"
            );
        }

        let (engine, channels) = lane.shutdown();
        assert!(engine.is_none(), "a poisoned lane hands back no engine");
        assert!(channels.is_empty());
    }

    struct ExhaustedEngine {
        submits: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl engine::Engine for ExhaustedEngine {
        /// A test engine's copies land inside the call: nothing to wait for.
        fn settle_copies(&mut self) -> ::engine::Result<()> {
            Ok(())
        }

        /// A test engine's copies complete inside the call: their sources are
        /// read by the time it returns.
        fn copy_slots_landing(
            &mut self,
            copy: &::engine::chunk::SlotCopy,
            _token: u64,
        ) -> ::engine::Result<bool> {
            self.copy_slots(copy)?;
            Ok(true)
        }

        fn kind(&self) -> &'static str {
            "exhausted"
        }

        fn prepare_program(
            &mut self,
            _registration: &::engine::program::ProgramRegistration,
        ) -> ::engine::Result<::engine::ProgramCompile> {
            Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
        }

        fn finish_program(
            &mut self,
            _prepared: ::engine::Prepared,
        ) -> ::engine::Result<::engine::ProgramId> {
            Err(::engine::Error::unsupported(self.kind(), "finish_program"))
        }

        fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
            Err(::engine::Error::unsupported(self.kind(), "close_program"))
        }

        fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
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

    fn a_retryable_refusal_past_admission_fails_by_name_instead_of_replaying() {
        let submits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut lane = Lane::spawn(
            0,
            Some(Box::new(ExhaustedEngine {
                submits: Arc::clone(&submits),
            })),
            Arc::new(SchedulerStats::default()),
        );

        let answer_rx = lane.fire(
            crate::engine::FrameFire {
                steps: vec![crate::engine::StepFire {
                    submission: ::engine::Step {
                        lanes: vec![::engine::Lane::decode(0, 0, 1, 0)],
                        attachments: Vec::new(),
                        media: Vec::new(),
                        voxels: Vec::new(),
                    },
                    terminal_cells: Vec::new(),
                    completions: Vec::new(),
                    member_lanes: Vec::new(),
                    instances: vec![::engine::InstanceId(0)],
                }],
            },
            false,
        );

        let result = crate::rt::block_on(async {
            crate::rt::time::timeout(Duration::from_secs(10), answer_rx).await
        })
        .expect("the launch must be answered, not slept on")
        .expect("the launch must be answered, not slept on");
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

    /// An engine that settles each frame inside `submit`: it tells the sink
    /// it was given what each reporting lane wrote, before it returns.
    struct InlineEngine {
        sink: Option<::engine::CompletionSink>,
    }

    impl engine::Engine for InlineEngine {
        /// A test engine's copies land inside the call: nothing to wait for.
        fn settle_copies(&mut self) -> ::engine::Result<()> {
            Ok(())
        }

        /// A test engine's copies complete inside the call: their sources are
        /// read by the time it returns.
        fn copy_slots_landing(
            &mut self,
            copy: &::engine::chunk::SlotCopy,
            _token: u64,
        ) -> ::engine::Result<bool> {
            self.copy_slots(copy)?;
            Ok(true)
        }

        fn kind(&self) -> &'static str {
            "inline"
        }

        fn prepare_program(
            &mut self,
            _registration: &::engine::program::ProgramRegistration,
        ) -> ::engine::Result<::engine::ProgramCompile> {
            Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
        }

        fn finish_program(
            &mut self,
            _prepared: ::engine::Prepared,
        ) -> ::engine::Result<::engine::ProgramId> {
            Err(::engine::Error::unsupported(self.kind(), "finish_program"))
        }

        fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
            Err(::engine::Error::unsupported(self.kind(), "close_program"))
        }

        fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
            Err(engine::Error::Load("no model".into()))
        }

        fn on_complete(&mut self, sink: ::engine::CompletionSink) {
            self.sink = Some(sink);
        }

        fn submit(
            &mut self,
            frame: &engine::FrameSubmission,
        ) -> engine::Result<engine::FrameTicket> {
            let id = ::engine::FrameId(0);
            let sink = self.sink.as_ref().ok_or_else(|| {
                engine::Error::Invalid("no sink was installed before the first submit".into())
            })?;
            for (step, fired) in frame.steps.iter().enumerate() {
                let wrote = fired
                    .lanes
                    .iter()
                    .filter(|lane| lane.reports != 0)
                    .map(|lane| ::engine::KvWrites {
                        tag: lane.reports,
                        fold: Some(4),
                        ..Default::default()
                    })
                    .collect();
                sink(
                    ::engine::StepDone {
                        frame: id,
                        step: step as u32,
                    },
                    ::engine::StepOutcome::wrote(wrote),
                );
            }
            Ok(engine::FrameTicket {
                id,
                steps: frame.steps.iter().map(|_| Default::default()).collect(),
                ..Default::default()
            })
        }
    }

    /// A synchronous engine has no landing to tell a report with: the lane
    /// gives it a sink all the same, and what its lanes wrote is kept by
    /// the time the fire is answered.
    fn what_a_frame_settled_inside_submit_wrote_is_kept_when_it_is_answered() {
        const TAG: u64 = u64::MAX - 0x5e77;
        let mut lane = Lane::spawn(
            0,
            Some(Box::new(InlineEngine { sink: None })),
            Arc::new(SchedulerStats::default()),
        );

        let answer_rx = lane.fire(
            crate::engine::FrameFire {
                steps: vec![crate::engine::StepFire {
                    submission: ::engine::Step {
                        lanes: vec![::engine::Lane {
                            reports: TAG,
                            ..::engine::Lane::decode(0, 0, 1, 0)
                        }],
                        attachments: Vec::new(),
                        media: Vec::new(),
                        voxels: Vec::new(),
                    },
                    terminal_cells: Vec::new(),
                    completions: Vec::new(),
                    member_lanes: Vec::new(),
                    instances: Vec::new(),
                }],
            },
            false,
        );

        let result = crate::rt::block_on(async {
            crate::rt::time::timeout(Duration::from_secs(10), answer_rx).await
        })
        .expect("the launch is answered")
        .expect("the launch is answered");
        if let Err(why) = &result {
            panic!("the frame fires: {why}");
        }
        let kept = crate::pipeline::fire::kept_kv_writes(TAG)
            .expect("what the lane wrote is kept before its fire is answered");
        assert_eq!(kept.fold, Some(4), "the fold the engine told");

        let (_engine, _channels) = lane.shutdown();
    }

    /// Why [`LaneFaultEngine`]'s faulting lane faulted.
    const GUARD: &str = "guest epilogue faulted: stage 0: guard";

    /// An engine whose guest pass of one instance faults: it tells the step
    /// with that instance's lanes in `faulted` and commits the rest, settling off the
    /// submit (`asynchronous`) or inside it. With `lost` it refuses the
    /// whole frame instead, before any lane runs.
    struct LaneFaultEngine {
        sink: Option<::engine::CompletionSink>,
        asynchronous: bool,
        faulting: Option<::engine::InstanceId>,
        lost: bool,
        next: u64,
    }

    impl engine::Engine for LaneFaultEngine {
        /// A test engine's copies land inside the call: nothing to wait for.
        fn settle_copies(&mut self) -> ::engine::Result<()> {
            Ok(())
        }

        /// A test engine's copies complete inside the call: their sources are
        /// read by the time it returns.
        fn copy_slots_landing(
            &mut self,
            copy: &::engine::chunk::SlotCopy,
            _token: u64,
        ) -> ::engine::Result<bool> {
            self.copy_slots(copy)?;
            Ok(true)
        }

        fn kind(&self) -> &'static str {
            "lane-fault"
        }

        fn prepare_program(
            &mut self,
            _registration: &::engine::program::ProgramRegistration,
        ) -> ::engine::Result<::engine::ProgramCompile> {
            Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
        }

        fn finish_program(
            &mut self,
            _prepared: ::engine::Prepared,
        ) -> ::engine::Result<::engine::ProgramId> {
            Err(::engine::Error::unsupported(self.kind(), "finish_program"))
        }

        fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
            Err(::engine::Error::unsupported(self.kind(), "close_program"))
        }

        fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
            Err(engine::Error::Load("no model".into()))
        }

        fn settles_asynchronously(&self) -> bool {
            self.asynchronous
        }

        fn on_complete(&mut self, sink: ::engine::CompletionSink) {
            self.sink = Some(sink);
        }

        fn submit(
            &mut self,
            frame: &engine::FrameSubmission,
        ) -> engine::Result<engine::FrameTicket> {
            if self.lost {
                return Err(engine::Error::Device("the device is lost".into()));
            }
            let id = ::engine::FrameId(self.next);
            self.next += 1;
            let sink = self.sink.clone().expect("a sink before the first submit");
            for (step, fired) in frame.steps.iter().enumerate() {
                let mut outcome = ::engine::StepOutcome::committed();
                for attachment in fired
                    .attachments
                    .iter()
                    .filter(|a| Some(a.instance) == self.faulting)
                {
                    outcome.fault_lanes(
                        attachment.lane..attachment.lane + attachment.lanes,
                        Some(attachment.instance),
                        GUARD,
                    );
                }
                sink(
                    ::engine::StepDone {
                        frame: id,
                        step: step as u32,
                    },
                    outcome,
                );
            }
            Ok(engine::FrameTicket {
                id,
                steps: frame.steps.iter().map(|_| Default::default()).collect(),
                ..Default::default()
            })
        }
    }

    /// One step of fires, a lane each, each the pass of its own instance
    /// of `instances`; and the fires' completions.
    fn step_of(
        instances: &[u64],
    ) -> (
        crate::engine::FrameFire,
        Vec<crate::engine::WorkItemCompletion>,
    ) {
        let lanes = u32::try_from(instances.len()).expect("a few lanes");
        let completions: Vec<_> = (0..lanes)
            .map(|_| crate::engine::WorkItemCompletion::deferred_with_guard(None))
            .collect();
        let instance = |lane: u32| ::engine::InstanceId(instances[lane as usize]);
        let fire = crate::engine::FrameFire {
            steps: vec![crate::engine::StepFire {
                submission: ::engine::Step {
                    lanes: (0..lanes)
                        .map(|lane| ::engine::Lane::decode(lane, 0, 1, 0))
                        .collect(),
                    attachments: (0..lanes)
                        .map(|lane| ::engine::Attachment {
                            lane,
                            lanes: 1,
                            instance: instance(lane),
                            adapter_only: false,
                            scratch: ::engine::ScratchPlace::None,
                        })
                        .collect(),
                    media: Vec::new(),
                    voxels: Vec::new(),
                },
                terminal_cells: completions.iter().map(|c| c.terminal_cell_ptr()).collect(),
                completions: completions.clone(),
                member_lanes: (0..=lanes).collect(),
                instances: (0..lanes).map(instance).collect(),
            }],
        };
        (fire, completions)
    }

    /// Fires `fire` on `lane` and waits out its frame: its answer, and
    /// each fire's result once it settled.
    fn fire_and_settle(
        lane: &Lane,
        fire: crate::engine::FrameFire,
        completions: &[crate::engine::WorkItemCompletion],
    ) -> (
        std::result::Result<(), String>,
        Vec<std::result::Result<(), String>>,
    ) {
        let answer = within("the launch is answered", lane.fire(fire, false))
            .expect("the launch is answered");
        let answer = match answer {
            Ok(frame) => within("the frame settles", frame).map_err(|error| format!("{error:#}")),
            Err(error) => Err(error.message),
        };
        let fires = completions
            .iter()
            .map(|completion| {
                if !completion.is_settled() {
                    // A frame settled inside `submit` resolves its fires as
                    // the worker retires it.
                    completion.resolve_landed("test retire");
                }
                within("the fire settles", completion.clone()).map_err(|error| format!("{error:#}"))
            })
            .collect();
        (answer, fires)
    }

    /// A guest epilogue faulting on one lane of four fails that lane's fire
    /// alone, by the engine's reason; the other three commit, and fire on.
    fn one_faulted_lane_fails_its_own_fire_alone(asynchronous: bool) {
        let mut lane = Lane::spawn(
            0,
            Some(Box::new(LaneFaultEngine {
                sink: None,
                asynchronous,
                faulting: Some(::engine::InstanceId(102)),
                lost: false,
                next: 0,
            })),
            Arc::new(SchedulerStats::default()),
        );
        let (fire, completions) = step_of(&[100, 101, 102, 103]);
        let (frame, fires) = fire_and_settle(&lane, fire, &completions);
        assert_eq!(frame, Ok(()), "the frame is not failed by one lane's fault");
        for (at, fired) in fires.iter().enumerate() {
            if at == 2 {
                let why = fired.as_ref().expect_err("the faulted lane's fire fails");
                assert!(why.contains(GUARD), "by the engine's reason: {why}");
                assert!(why.contains("lane 2"), "naming its lane: {why}");
                assert!(why.contains("instance 102"), "naming its instance: {why}");
            } else {
                assert_eq!(fired, &Ok(()), "lane {at} commits");
            }
        }

        // The three go on: their next fires commit, and nothing of theirs
        // was poisoned by the fourth's fault.
        let (fire, completions) = step_of(&[100, 101, 103]);
        let (frame, fires) = fire_and_settle(&lane, fire, &completions);
        assert_eq!(frame, Ok(()));
        assert!(
            fires.iter().all(Result::is_ok),
            "the three fire on: {fires:?}"
        );
        let _ = lane.shutdown();
    }

    #[test]
    fn a_faulted_lane_fails_its_own_fire_alone_off_the_submit() {
        one_faulted_lane_fails_its_own_fire_alone(true);
    }

    #[test]
    fn a_faulted_lane_fails_its_own_fire_alone_inside_the_submit() {
        one_faulted_lane_fails_its_own_fire_alone(false);
    }

    /// A submit refused before any lane ran fails every fire of its frame.
    #[test]
    fn a_refused_submit_fails_every_fire_of_its_frame() {
        let mut lane = Lane::spawn(
            0,
            Some(Box::new(LaneFaultEngine {
                sink: None,
                asynchronous: true,
                faulting: None,
                lost: true,
                next: 0,
            })),
            Arc::new(SchedulerStats::default()),
        );
        let (fire, completions) = step_of(&[100, 101, 102, 103]);
        let answer = within("the launch is answered", lane.fire(fire, false))
            .expect("the launch is answered");
        let Err(why) = answer else {
            panic!("a lost device refuses the frame");
        };
        assert!(why.contains("the device is lost"), "{why}");
        for completion in &completions {
            assert_eq!(
                completion.resolve_from_terminal().ok(),
                Some(crate::engine::completion::WorkItemAttemptOutcome::Failed),
                "every fire of the refused frame fails"
            );
        }
        let _ = lane.shutdown();
    }

    /// The word bit [`LaneRefusalEngine`] refuses a lane for: a fact its
    /// load serves in no arm (a mask word no arm reads, say).
    const UNSERVED: u64 = 1 << 7;

    /// An engine that refuses, for its own sake, every lane whose word
    /// states [`UNSERVED`], staging a frame step by step as the engines do:
    /// at its first step the frame is refused naming those lanes
    /// ([`engine::Error::Lanes`]); at a later one the frame stops there,
    /// its ticket naming them. Each step it fires lands what each lane fed
    /// (`fired`, as `(instance, token)`, step by step), each lane read
    /// through its attachment, so a lane renumbered under the wrong
    /// instance shows.
    struct LaneRefusalEngine {
        sink: Option<::engine::CompletionSink>,
        asynchronous: bool,
        fired: Arc<std::sync::Mutex<Vec<Vec<(u64, u32)>>>>,
        next: u64,
        /// Lands a frame stopped short after the frame submitted next,
        /// out of the order they were submitted in.
        reorder: bool,
        held: Vec<::engine::StepDone>,
    }

    impl engine::Engine for LaneRefusalEngine {
        /// A test engine's copies land inside the call: nothing to wait for.
        fn settle_copies(&mut self) -> ::engine::Result<()> {
            Ok(())
        }

        /// A test engine's copies complete inside the call: their sources are
        /// read by the time it returns.
        fn copy_slots_landing(
            &mut self,
            copy: &::engine::chunk::SlotCopy,
            _token: u64,
        ) -> ::engine::Result<bool> {
            self.copy_slots(copy)?;
            Ok(true)
        }

        fn kind(&self) -> &'static str {
            "lane-refusal"
        }

        fn prepare_program(
            &mut self,
            _registration: &::engine::program::ProgramRegistration,
        ) -> ::engine::Result<::engine::ProgramCompile> {
            Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
        }

        fn finish_program(
            &mut self,
            _prepared: ::engine::Prepared,
        ) -> ::engine::Result<::engine::ProgramId> {
            Err(::engine::Error::unsupported(self.kind(), "finish_program"))
        }

        fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
            Err(::engine::Error::unsupported(self.kind(), "close_program"))
        }

        fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
            Err(engine::Error::Load("no model".into()))
        }

        fn settles_asynchronously(&self) -> bool {
            self.asynchronous
        }

        fn on_complete(&mut self, sink: ::engine::CompletionSink) {
            self.sink = Some(sink);
        }

        fn submit(
            &mut self,
            frame: &engine::FrameSubmission,
        ) -> engine::Result<engine::FrameTicket> {
            let id = ::engine::FrameId(self.next);
            self.next += 1;
            let sink = self.sink.clone().expect("a sink before the first submit");
            let mut steps = Vec::new();
            let mut refused = None;
            for (index, step) in frame.steps.iter().enumerate() {
                // A lane whose program places its rows lands them at the
                // slots its tokens name, as a guest's `slots` port would.
                let lanes: Vec<::engine::LaneFault> = (step.lanes.iter().enumerate())
                    .filter_map(|(at, lane)| {
                        let reason = if lane.word & UNSERVED != 0 {
                            Some("its word reads a mask in no arm this load applies one in".into())
                        } else if lane.list == ::engine::ListKind::Translation {
                            ::engine::fire::misplaced_write(lane.kv.writable(), &lane.tokens)
                        } else {
                            None
                        };
                        reason.map(|reason| ::engine::LaneFault {
                            lane: u32::try_from(at).expect("a few lanes"),
                            instance: None,
                            reason,
                        })
                    })
                    .collect();
                if !lanes.is_empty() {
                    let error = engine::Error::Lanes(lanes);
                    if !error.stops_frame_at(index) {
                        return Err(error);
                    }
                    refused = Some(error);
                    break;
                }
                let fed = step
                    .attachments
                    .iter()
                    .map(|attachment| {
                        let lane = &step.lanes[attachment.lane as usize];
                        assert_eq!(
                            u64::from(lane.slot),
                            attachment.instance.0,
                            "an attachment reads its own instance's lane"
                        );
                        (attachment.instance.0, lane.tokens[0])
                    })
                    .collect();
                self.fired.lock().unwrap().push(fed);
                steps.push(::engine::FireTicket::default());
            }
            let landings: Vec<::engine::StepDone> = (0..steps.len())
                .map(|step| ::engine::StepDone {
                    frame: id,
                    step: u32::try_from(step).expect("a few steps"),
                })
                .collect();
            if self.reorder && refused.is_some() {
                self.held.extend(landings);
            } else {
                for at in landings.into_iter().chain(std::mem::take(&mut self.held)) {
                    sink(at, ::engine::StepOutcome::committed());
                }
            }
            Ok(engine::FrameTicket::of(id, steps, refused))
        }
    }

    /// A frame of steps of fires, a lane each, each the pass of its own
    /// instance: step `s` is `steps[s]`, `(instance, unserved)` a fire. A
    /// lane's slot is its instance and its token `1000 * step + instance`.
    /// The fires' completions, step by step.
    fn frame_of(
        steps: &[&[(u64, bool)]],
    ) -> (
        crate::engine::FrameFire,
        Vec<Vec<crate::engine::WorkItemCompletion>>,
    ) {
        let lanes: Vec<Vec<(u64, ::engine::Lane)>> = steps
            .iter()
            .enumerate()
            .map(|(at, fires)| {
                fires
                    .iter()
                    .map(|&(instance, unserved)| {
                        let word = if unserved { UNSERVED | 1 } else { 1 };
                        let token = u32::try_from(1000 * at as u64 + instance).expect("small");
                        let slot = u32::try_from(instance).expect("small");
                        (instance, ::engine::Lane::decode(slot, word, token, 0))
                    })
                    .collect()
            })
            .collect();
        frame_of_lanes(&lanes)
    }

    /// A frame of steps of fires, a lane each, each the pass of its own
    /// instance: step `s` is `steps[s]`, `(instance, lane)` a fire.
    fn frame_of_lanes(
        steps: &[Vec<(u64, ::engine::Lane)>],
    ) -> (
        crate::engine::FrameFire,
        Vec<Vec<crate::engine::WorkItemCompletion>>,
    ) {
        let mut fire = crate::engine::FrameFire::default();
        let mut all = Vec::new();
        for fires in steps {
            let lanes = u32::try_from(fires.len()).expect("a few lanes");
            let completions: Vec<_> = (0..lanes)
                .map(|_| crate::engine::WorkItemCompletion::deferred_with_guard(None))
                .collect();
            fire.steps.push(crate::engine::StepFire {
                submission: ::engine::Step {
                    lanes: fires.iter().map(|(_, lane)| lane.clone()).collect(),
                    attachments: (0..lanes)
                        .map(|lane| ::engine::Attachment {
                            lane,
                            lanes: 1,
                            instance: ::engine::InstanceId(fires[lane as usize].0),
                            adapter_only: false,
                            scratch: ::engine::ScratchPlace::None,
                        })
                        .collect(),
                    media: Vec::new(),
                    voxels: Vec::new(),
                },
                terminal_cells: completions.iter().map(|c| c.terminal_cell_ptr()).collect(),
                completions: completions.clone(),
                member_lanes: (0..=lanes).collect(),
                instances: fires.iter().map(|&(id, _)| ::engine::InstanceId(id)).collect(),
            });
            all.push(completions);
        }
        (fire, all)
    }

    /// Fires `steps` on a [`LaneRefusalEngine`]: the frame's answer, each
    /// fire's result step by step, and what each step the engine fired fed.
    #[allow(clippy::type_complexity, reason = "a test's three answers")]
    fn fire_refusing(
        asynchronous: bool,
        steps: &[&[(u64, bool)]],
    ) -> (
        std::result::Result<(), String>,
        Vec<Vec<std::result::Result<(), String>>>,
        Vec<Vec<(u64, u32)>>,
    ) {
        fire_refusing_landing(asynchronous, false, steps)
    }

    /// [`fire_refusing`], with `reorder` landing a frame stopped short
    /// after the one submitted next.
    #[allow(clippy::type_complexity, reason = "a test's three answers")]
    fn fire_refusing_landing(
        asynchronous: bool,
        reorder: bool,
        steps: &[&[(u64, bool)]],
    ) -> (
        std::result::Result<(), String>,
        Vec<Vec<std::result::Result<(), String>>>,
        Vec<Vec<(u64, u32)>>,
    ) {
        fire_refusing_frame(asynchronous, reorder, frame_of(steps))
    }

    /// Fires `frame` on a [`LaneRefusalEngine`], as [`fire_refusing`] does.
    #[allow(clippy::type_complexity, reason = "a test's three answers")]
    fn fire_refusing_frame(
        asynchronous: bool,
        reorder: bool,
        (fire, completions): (
            crate::engine::FrameFire,
            Vec<Vec<crate::engine::WorkItemCompletion>>,
        ),
    ) -> (
        std::result::Result<(), String>,
        Vec<Vec<std::result::Result<(), String>>>,
        Vec<Vec<(u64, u32)>>,
    ) {
        let fired = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut lane = Lane::spawn(
            0,
            Some(Box::new(LaneRefusalEngine {
                sink: None,
                asynchronous,
                fired: Arc::clone(&fired),
                next: 0,
                reorder,
                held: Vec::new(),
            })),
            Arc::new(SchedulerStats::default()),
        );
        let flat: Vec<_> = completions.iter().flatten().cloned().collect();
        let (frame, results) = if reorder {
            // Every fire of a launch landing out of order is failed by the
            // lane as it lands, whenever after the frame that does.
            let answer = within("the launch is answered", lane.fire(fire, false))
                .expect("the launch is answered");
            let answer = match answer {
                Ok(frame) => {
                    within("the frame settles", frame).map_err(|error| format!("{error:#}"))
                }
                Err(error) => Err(error.message),
            };
            let results = flat
                .iter()
                .map(|completion| {
                    within("the fire settles", completion.clone())
                        .map_err(|error| format!("{error:#}"))
                })
                .collect();
            (answer, results)
        } else {
            fire_and_settle(&lane, fire, &flat)
        };
        let mut results = results.into_iter();
        let by_step = completions
            .iter()
            .map(|step| results.by_ref().take(step.len()).collect())
            .collect();
        let _ = lane.shutdown();
        let fed = fired.lock().unwrap().clone();
        (frame, by_step, fed)
    }

    fn refused(result: &std::result::Result<(), String>) -> bool {
        result
            .as_ref()
            .is_err_and(|why| why.contains("lane") && why.contains("refused"))
    }

    /// One lane of four refused at a fire's staging fails its own fire
    /// alone, by the engine's reason; the other three fire once each, in
    /// the same submission, with what they fed.
    fn one_refused_lane_fails_its_own_fire_alone(asynchronous: bool) {
        let (frame, fires, fed) = fire_refusing(
            asynchronous,
            &[&[(100, false), (101, false), (102, true), (103, false)]],
        );
        assert_eq!(frame, Ok(()), "the frame is not failed by one lane's refusal");
        let why = fires[0][2].as_ref().expect_err("the refused lane's fire fails");
        assert!(why.contains("reads a mask in no arm"), "by the engine's reason: {why}");
        assert!(why.contains("lane 2"), "naming its lane: {why}");
        for at in [0, 1, 3] {
            assert_eq!(fires[0][at], Ok(()), "fire {at} commits");
        }
        assert_eq!(
            fed,
            vec![vec![(100, 100), (101, 101), (103, 103)]],
            "the three fire once, each with its own token"
        );
    }

    #[test]
    fn a_refused_lane_fails_its_own_fire_alone_off_the_submit() {
        one_refused_lane_fails_its_own_fire_alone(true);
    }

    #[test]
    fn a_refused_lane_fails_its_own_fire_alone_inside_the_submit() {
        one_refused_lane_fails_its_own_fire_alone(false);
    }

    /// A lane refused at a frame's second step: the first step fired whole
    /// (its fires commit, once), and the second fires again without the
    /// refused fire, whose earlier fire stands.
    fn a_lane_refused_past_the_first_step_stops_the_frame_there(asynchronous: bool) {
        let (frame, fires, fed) = fire_refusing(
            asynchronous,
            &[
                &[(100, false), (101, false), (102, false)],
                &[(100, false), (101, true), (102, false)],
            ],
        );
        assert_eq!(frame, Ok(()));
        assert!(fires[0].iter().all(Result::is_ok), "the first step commits: {fires:?}");
        assert!(refused(&fires[1][1]), "the refused fire fails: {:?}", fires[1][1]);
        assert_eq!(fires[1][0], Ok(()));
        assert_eq!(fires[1][2], Ok(()));
        assert_eq!(
            fed,
            vec![
                vec![(100, 100), (101, 101), (102, 102)],
                vec![(100, 1100), (102, 1102)],
            ],
            "no step fires twice, and none of the others' tokens is lost"
        );
    }

    #[test]
    fn a_lane_refused_past_the_first_step_stops_the_frame_there_off_the_submit() {
        a_lane_refused_past_the_first_step_stops_the_frame_there(true);
    }

    #[test]
    fn a_lane_refused_past_the_first_step_stops_the_frame_there_inside_the_submit() {
        a_lane_refused_past_the_first_step_stops_the_frame_there(false);
    }

    /// A launch split into two engine frames whose second lands before its
    /// first is refused by name: every fire of the launch fails, none on
    /// another frame's tokens.
    #[test]
    fn a_split_launch_landing_out_of_order_is_refused_by_name() {
        let (_frame, fires, fed) = fire_refusing_landing(
            true,
            true,
            &[
                &[(100, false), (101, false)],
                &[(100, false), (101, true)],
            ],
        );
        assert_eq!(fed.len(), 2, "the engine fired both frames: {fed:?}");
        assert!(refused(&fires[1][1]), "{:?}", fires[1][1]);
        for fired in [&fires[0][0], &fires[0][1], &fires[1][0]] {
            let why = fired.as_ref().expect_err("no fire takes an out-of-order frame's tokens");
            assert!(
                why.contains("settled its engine frame 1 of 2 while frame 0 was expected"),
                "named: {why}"
            );
        }
    }

    /// A lane refused at a frame's first step takes its instance's later
    /// fires of the frame with it (they read what it would have left); the
    /// other instances' fires of every step fire once, and a step left
    /// with nothing goes.
    #[test]
    fn a_refused_lane_takes_its_instances_later_fires_of_the_frame() {
        let (frame, fires, fed) = fire_refusing(
            true,
            &[
                &[(100, false), (101, true)],
                &[(101, false)],
                &[(100, false), (101, false)],
            ],
        );
        assert_eq!(frame, Ok(()));
        assert!(refused(&fires[0][1]), "{:?}", fires[0][1]);
        for later in [&fires[1][0], &fires[2][1]] {
            let why = later.as_ref().expect_err("the instance's later fires fail");
            assert!(why.contains("an earlier fire of this instance"), "{why}");
        }
        assert_eq!(fires[0][0], Ok(()));
        assert_eq!(fires[2][0], Ok(()));
        assert_eq!(fed, vec![vec![(100, 100)], vec![(100, 2100)]]);
    }

    /// A lane whose program places its rows at a slot below its set's
    /// frozen prefix, or past what it reserved, is refused by name, the
    /// write never made; the lanes beside it, writing their own slots,
    /// fire once each in the same submission.
    #[test]
    fn a_row_placed_outside_its_writable_slots_fails_its_own_fire_alone() {
        // Each lane's set froze slots 0..8 and reserved through 16; its
        // program places its row at the slot its token names.
        let placed = |instance: u64, slot: u32| {
            let mut lane = ::engine::Lane::decode(u32::try_from(instance).unwrap(), 1, slot, 0);
            lane.list = ::engine::ListKind::Translation;
            lane.kv.folded = 8;
            lane.kv.write_end = 16;
            (instance, lane)
        };
        for asynchronous in [true, false] {
            let steps = vec![vec![
                placed(100, 8),
                placed(101, 3),
                placed(102, 15),
                placed(103, 16),
            ]];
            let (frame, fires, fed) =
                fire_refusing_frame(asynchronous, false, frame_of_lanes(&steps));
            assert_eq!(frame, Ok(()), "the frame is not failed by one lane's write");
            let below = fires[0][1]
                .as_ref()
                .expect_err("the lane writing below fails");
            assert!(
                below.contains("lane 1") && below.contains("kv slot 3") && below.contains("frozen"),
                "named, by the slot it placed: {below}"
            );
            let past = fires[0][3]
                .as_ref()
                .expect_err("the lane writing past fails");
            assert!(
                past.contains("lane 3") && past.contains("kv slot 16"),
                "{past}"
            );
            assert_eq!(
                fires[0][0],
                Ok(()),
                "a row at the frozen prefix's end lands"
            );
            assert_eq!(fires[0][2], Ok(()), "a row inside what it reserved lands");
            assert_eq!(
                fed,
                vec![vec![(100, 8), (102, 15)]],
                "the lanes writing their own slots fire once; the refused never fire"
            );
        }
    }

    /// What [`HoldingEngine`] answers it holds back once it took a resident
    /// set.
    const HELD: u64 = 96 << 20;

    struct HoldingEngine;

    impl engine::Engine for HoldingEngine {
        /// A test engine's copies land inside the call: nothing to wait for.
        fn settle_copies(&mut self) -> ::engine::Result<()> {
            Ok(())
        }

        /// A test engine's copies complete inside the call: their sources are
        /// read by the time it returns.
        fn copy_slots_landing(
            &mut self,
            copy: &::engine::chunk::SlotCopy,
            _token: u64,
        ) -> ::engine::Result<bool> {
            self.copy_slots(copy)?;
            Ok(true)
        }

        fn kind(&self) -> &'static str {
            "holding"
        }

        fn prepare_program(
            &mut self,
            _registration: &::engine::program::ProgramRegistration,
        ) -> ::engine::Result<::engine::ProgramCompile> {
            Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
        }

        fn finish_program(
            &mut self,
            _prepared: ::engine::Prepared,
        ) -> ::engine::Result<::engine::ProgramId> {
            Err(::engine::Error::unsupported(self.kind(), "finish_program"))
        }

        fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
            Err(::engine::Error::unsupported(self.kind(), "close_program"))
        }

        fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
            Err(engine::Error::Load("no model".into()))
        }

        fn submit(
            &mut self,
            _frame: &engine::FrameSubmission,
        ) -> engine::Result<engine::FrameTicket> {
            Err(engine::Error::Load("no model".into()))
        }

        fn weight_residency(
            &mut self,
            _residency: &::engine::chunk::WeightResidency,
        ) -> engine::Result<u64> {
            Ok(HELD)
        }
    }

    fn within<T>(what: &str, wait: impl std::future::Future<Output = T>) -> T {
        crate::rt::block_on(async { crate::rt::time::timeout(Duration::from_secs(10), wait).await })
            .unwrap_or_else(|_| panic!("timed out: {what}"))
    }

    /// A resident set queued on the lane is answered with what the engine
    /// holds back once it took it, and the weight port reads that answer back
    /// as the bytes; an engine that does not stream answers the port nothing.
    #[test]
    fn a_resident_set_is_answered_with_what_the_engine_holds_back() {
        use crate::store::weights::port::EnginePort;
        const ENGINE: usize = 9309;
        let mut lane = Lane::spawn(
            ENGINE,
            Some(Box::new(HoldingEngine)),
            Arc::new(SchedulerStats::default()),
        );
        let (response, answer) = tokio::sync::oneshot::channel();
        let done = lane.control(QueuedItem::HostKv {
            op: HostKvOp::WeightResidency(::engine::chunk::WeightResidency::default()),
            response,
        });
        within("the lane takes the resident set", done).expect("the lane answers");
        let answer = within("the resident set is answered", answer).expect("answered, not dropped");
        let mut port = EnginePort::new(ENGINE, false);
        assert_eq!(port.answered(answer).expect("taken"), Some(HELD));
        assert!(!port.idle());
        let unsupported = ::engine::Error::unsupported("holding", "weight_residency");
        assert_eq!(
            port.answered(Err(unsupported.into())).expect("no error"),
            None
        );
        assert!(port.idle(), "asked nothing more");
        let (_engine, _channels) = lane.shutdown();
    }

    /// A handle on no scheduler: what a program's registration by content
    /// hash keeps, driven by a registration of the test's own.
    fn bare_handle() -> SchedulerHandle {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        SchedulerHandle {
            inner: Arc::new(SchedulerControl {
                tx,
                active_senders: AtomicUsize::new(0),
                shutdown_wait: Condvar::new(),
                shutdown_gate: Mutex::new(()),
                program_ids: Mutex::new(HashMap::new()),
                accepting: AtomicBool::new(true),
                stats: Arc::new(SchedulerStats::default()),
                device_domain: ::engine::MemoryDomain::HostPinned,
                codegen_backend: None,
            }),
        }
    }

    async fn never_registers() -> Result<::engine::ProgramId> {
        panic!("a program found by its hash registers again")
    }

    async fn panics_registering() -> Result<::engine::ProgramId> {
        tokio::time::sleep(Duration::from_millis(20)).await;
        panic!("the compile panics")
    }

    fn registration(hash: u64) -> ProgramRegistration {
        ProgramRegistration {
            program_hash: hash,
            ..ProgramRegistration::default()
        }
    }

    /// Binds asking at once for one program they find unregistered wait on
    /// one registration: one compile, one id, and a later bind finds it.
    #[test]
    fn concurrent_binds_of_one_new_program_register_it_once() {
        let handle = bare_handle();
        let registered = Arc::new(AtomicUsize::new(0));
        let ids = crate::rt::block_on(async {
            let binds = (0..8).map(|_| {
                let registered = Arc::clone(&registered);
                handle.program_id_with(
                    ProcessId::default(),
                    registration(0xd0d0),
                    |_, _| async move {
                        registered.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        Ok(::engine::ProgramId(7))
                    },
                )
            });
            futures::future::join_all(binds).await
        });
        for id in ids {
            assert_eq!(id.expect("each bind gets the id"), ::engine::ProgramId(7));
        }
        assert_eq!(registered.load(Ordering::SeqCst), 1, "one registration");
        let again = crate::rt::block_on(handle.program_id_with(
            ProcessId::default(),
            registration(0xd0d0),
            |_, _| never_registers(),
        ));
        assert_eq!(again.expect("found"), ::engine::ProgramId(7));
        let mut colliding = registration(0xd0d0);
        colliding.launch.plans.push(Default::default());
        let collided =
            crate::rt::block_on(
                handle.program_id_with(ProcessId::default(), colliding, |_, _| never_registers()),
            );
        assert!(
            collided
                .expect_err("refused")
                .to_string()
                .contains("collision"),
            "a hash collision is refused by name"
        );
    }

    /// A bind dropped while its program compiles does not stop the
    /// registration: it reaches its end, and the next bind finds the id.
    #[test]
    fn a_registration_outlives_the_bind_that_started_it() {
        let handle = bare_handle();
        let finished = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&finished);
        crate::rt::block_on(async {
            let bind = handle.program_id_with(
                ProcessId::default(),
                registration(0xc0c0),
                |_, _| async move {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    done.store(true, Ordering::SeqCst);
                    Ok(::engine::ProgramId(9))
                },
            );
            assert!(
                crate::rt::time::timeout(Duration::from_millis(10), bind)
                    .await
                    .is_err(),
                "the bind is dropped mid-compile"
            );
            tokio::time::sleep(Duration::from_millis(300)).await;
        });
        assert!(finished.load(Ordering::SeqCst), "the registration finished");
        let found = crate::rt::block_on(handle.program_id_with(
            ProcessId::default(),
            registration(0xc0c0),
            |_, _| never_registers(),
        ));
        assert_eq!(found.expect("found"), ::engine::ProgramId(9));
    }

    /// A registration that panics refuses each bind waiting on it by the
    /// program's hash, and leaves no entry: the next bind registers afresh.
    #[test]
    fn a_registration_that_panics_is_refused_by_name_and_forgotten() {
        let handle = bare_handle();
        let refused = crate::rt::block_on(async {
            let binds = (0..3).map(|_| {
                handle.program_id_with(ProcessId::default(), registration(0xbad1), |_, _| {
                    panics_registering()
                })
            });
            futures::future::join_all(binds).await
        });
        for bind in refused {
            let why = bind.expect_err("refused").to_string();
            assert!(why.contains("0x000000000000bad1"), "by name: {why}");
        }
        assert!(
            !handle
                .inner
                .program_ids
                .lock()
                .unwrap()
                .contains_key(&0xbad1),
            "no entry is left"
        );
        let retried = crate::rt::block_on(handle.program_id_with(
            ProcessId::default(),
            registration(0xbad1),
            |_, _| async { Ok(::engine::ProgramId(3)) },
        ));
        assert_eq!(retried.expect("registers afresh"), ::engine::ProgramId(3));
    }
}

/// A test's own lane for an engine: the copies its planner and stores
/// submit (`copy_tier_tracked`) are handed to the test, which lands each
/// one (or refuses it) when it chooses; the pool requests a move waits on
/// (`settle_copies`, a rebind) are answered at once. Installed for this
/// thread alone ([`crate::scheduler::set_test_handle`]).
#[cfg(test)]
pub(crate) mod test_lane {
    use super::*;

    pub(crate) struct TestLane {
        engine: usize,
        rx: tokio::sync::mpsc::UnboundedReceiver<SchedulerItem>,
    }

    /// One copy handed to the lane, unlanded.
    pub(crate) struct LaneCopy {
        /// Down to the disk tier.
        pub(crate) to_disk: bool,
        /// Up from the disk tier.
        pub(crate) from_disk: bool,
        /// The slots it copies, shape by shape (a shared pool's every copy).
        pub(crate) slots: Option<::engine::chunk::SlotCopy>,
        completion: ControlCompletion,
    }

    impl LaneCopy {
        /// The copy lands: its completion resolves.
        pub(crate) fn land(self) {
            self.completion.resolve(&Ok(()));
        }

        /// The copy did not land (`engine::Landing::Failed`): its
        /// destination is not to be trusted, its sources hold what they
        /// held.
        pub(crate) fn fail_landing(self, why: &str) {
            self.completion.resolve(&Err(anyhow::Error::new(
                crate::engine::completion::CopyLandingFailed(why.to_string()),
            )));
        }

        /// The tier it copies down to is full (`engine::Error::TierFull`):
        /// nothing on the device changed.
        pub(crate) fn refuse_tier_full(self) {
            self.completion
                .resolve(&Err(anyhow::Error::new(::engine::Error::TierFull(
                    "the disk tier's slot file is full (test lane)".into(),
                ))));
        }
    }

    impl TestLane {
        pub(crate) fn install(engine: usize) -> Self {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SchedulerItem>();
            let handle = SchedulerHandle {
                inner: Arc::new(SchedulerControl {
                    tx,
                    active_senders: AtomicUsize::new(0),
                    shutdown_wait: Condvar::new(),
                    shutdown_gate: Mutex::new(()),
                    program_ids: Mutex::new(HashMap::new()),
                    accepting: AtomicBool::new(true),
                    stats: Arc::new(SchedulerStats::default()),
                    device_domain: ::engine::MemoryDomain::CudaDevice(0),
                    codegen_backend: None,
                }),
            };
            crate::scheduler::set_test_handle(engine, Some(handle));
            Self { engine, rx }
        }

        /// The copies submitted since the last take, in order; every pool
        /// request is answered, a rebind by `rebind` as the lane runs it
        /// (`store::roles::rebind`), every other item dropped.
        pub(crate) fn take(
            &mut self,
            rebind: &mut dyn FnMut(::engine::Rebind) -> Result<()>,
        ) -> Vec<LaneCopy> {
            let mut copies = Vec::new();
            while let Ok(item) = self.rx.try_recv() {
                match item {
                    SchedulerItem::CopyTracked { plan, completion } => {
                        let disk = ::engine::MemoryDomain::LocalDisk;
                        let (to_disk, from_disk) = match &plan {
                            PreLaunchCopy::Slots(copy) => (copy.dst == disk, copy.src == disk),
                            PreLaunchCopy::Kv(copy) => (copy.dst == disk, copy.src == disk),
                            PreLaunchCopy::State(_) => (false, false),
                        };
                        let slots = match &plan {
                            PreLaunchCopy::Slots(copy) => Some(copy.clone()),
                            _ => None,
                        };
                        copies.push(LaneCopy {
                            to_disk,
                            from_disk,
                            slots,
                            completion,
                        });
                    }
                    SchedulerItem::Pool {
                        op: PoolOp::Rebind(change),
                        response,
                    } => {
                        let _ = response.send(rebind(change));
                    }
                    SchedulerItem::Pool { response, .. } => {
                        let _ = response.send(Ok(()));
                    }
                    SchedulerItem::HostKv { response, .. } => {
                        let _ = response.send(Ok(Vec::new()));
                    }
                    _ => {}
                }
            }
            copies
        }
    }

    impl Drop for TestLane {
        fn drop(&mut self) {
            crate::scheduler::set_test_handle(self.engine, None);
        }
    }
}
