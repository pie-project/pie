use crate::rt::Instant;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use ::engine::{ChannelRegistration, Engine, ProgramRegistration, StateCopy};

use crate::engine::verbs::serve;
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

mod dispatch;
mod dump;
mod enqueue;
mod handle;
mod retire;

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
pub(crate) static OWNERS_BLOCKED: std::sync::Mutex<Vec<ProcessId>> =
    std::sync::Mutex::new(Vec::new());

/// `pid`'s guest waits on input from outside the runtime (a session
/// receive, a timer, an HTTP response): no frame on any engine waits on its
/// lanes until each submits again.
pub(crate) fn notify_owner_away(pid: ProcessId) {
    broadcast(|| SchedulerItem::OwnerAway(pid));
}

/// `lane`'s process on `engine` waits on the output of the fire
/// `completion` settles, having submitted nothing since: no frame on
/// `engine` waits on the lane until it submits again
/// ([`super::frame::FramePolicy::on_lane_waits`]).
pub(crate) fn notify_lane_waits(engine: usize, lane: ProcessId, completion: WorkItemCompletion) {
    if completion.is_settled() {
        return;
    }
    if let Ok(handle) = crate::scheduler::scheduler_handle(engine) {
        let _ = handle.send(SchedulerItem::LaneWaits { lane, completion });
    }
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

#[cfg_attr(
    target_arch = "wasm32",
    allow(dead_code, reason = "no policy runs in the browser")
)]
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
    /// When it was handed to the scheduler: a request's lanes name the
    /// slots its stores held then (a debug check at the engine says how
    /// long ago).
    pub(crate) queued: crate::rt::wall::Instant,
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
            queued: crate::rt::wall::Instant::now(),
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
    /// The lane's process waits on the output of the fire the completion
    /// settles ([`super::frame::FramePolicy::on_lane_waits`]).
    LaneWaits {
        lane: ProcessId,
        completion: WorkItemCompletion,
    },
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
    /// A shared pool backs segments `0..n` (`Pool::commit`).
    CommitSegments(u32),
    /// A shared pool gives back segments `n..` (`Pool::release`).
    ReleaseSegments(u32),
    /// The activation rung moves (`Pool::set_rung`), where the span is
    /// backed on demand: a raise before any fire of the wider rung, a lower
    /// as a notice after its retire fence.
    Rung { from: u32, to: u32 },
    /// A shared pool whose span is backed on demand backs its top `n` span
    /// segments (`Pool::commit_span`).
    CommitSpan(u32),
    /// A shared pool whose span is backed on demand keeps its top `n` span
    /// segments and gives back the rest (`Pool::release_span`).
    ReleaseSpan(u32),
    /// A shared pool's chunks move, ahead of every fire queued after it
    /// (`Pool::evacuate`).
    Evacuate(::engine::chunk::Evacuation),
    /// An instance's tables or a ring were unparked into these slots
    /// (`Programs::rebind`), ahead of every fire queued after it.
    Rebind(::engine::Rebind),
    /// Every copy submitted so far has landed (`Tiers::settle`):
    /// what an aborted move waits for before it gives back slots a copy may
    /// still be writing.
    SettleCopies,
    /// The engine's chunk table (`Pool::chunk_table`) checked against the
    /// runtime's logical to physical map, by logical chunk: refused naming
    /// the first chunk they place apart. What a refused evacuation is
    /// checked by, the engine's word that nothing moved not taken on trust.
    CheckTable(Vec<u32>),
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
            Self::CheckTable(_) => "check chunk table",
        }
    }
}

/// A read or write of host kv slots' bytes, behind every copy queued before
/// it.
#[derive(Debug)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) enum HostKvOp {
    Write(Vec<u32>, Vec<u8>),
    /// The engine's host-tier layout, as JSON; empty when it cannot say.
    Layout,
    /// A copy of device pages into host slots, then a read of those slots:
    /// one item on the lane, with no completion to settle between.
    CopyRead(::engine::KvCopy, Vec<u32>),
    /// The bytes of host kv slots.
    #[cfg_attr(
        target_arch = "wasm32",
        allow(dead_code, reason = "the browser has no host or disk tier")
    )]
    Read(Vec<u32>),
    /// The bytes of host tier slots, shape by shape (`Tiers::read`).
    #[cfg_attr(
        target_arch = "wasm32",
        allow(dead_code, reason = "the browser has no host or disk tier")
    )]
    ReadTier(Vec<(::engine::chunk::ShapeId, Vec<u32>)>),
    #[cfg_attr(
        target_arch = "wasm32",
        allow(dead_code, reason = "the browser has no host or disk tier")
    )]
    WriteTier(Vec<(::engine::chunk::ShapeId, Vec<u32>)>, Vec<u8>),
    /// The engine's tap groups (`Taps::groups`), as JSON.
    TapGroups,
    /// The weight cache's resident set (`WeightCache::set_residency`),
    /// answered with the bytes the engine holds back once it took it, ahead
    /// of every fire queued after it.
    WeightResidency(::engine::chunk::WeightResidency),
    /// The engine's weight use record since it last answered
    /// (`WeightCache::uses`), as JSON.
    WeightUses,
    /// A beam group's beam's page list (`Programs::beam_list`), little-endian
    /// u32s: behind every fire queued before it.
    BeamList(::engine::InstanceId, u32),
    /// Overwrites the model's weights in place from a checkpoint
    /// (`WeightCache::reload`), answering what it wrote as JSON.
    ReloadWeights(std::path::PathBuf),
}

#[cfg(not(target_arch = "wasm32"))]
type SchedTx = tokio::sync::mpsc::UnboundedSender<SchedulerItem>;
#[cfg(not(target_arch = "wasm32"))]
type SchedRx = tokio::sync::mpsc::UnboundedReceiver<SchedulerItem>;

/// The tab's scheduler mailbox: web_std's channel, as the lane's control
/// queue is, so a send while the scheduler sleeps on it wakes it through the
/// page's executor (a tokio mailbox's send there left it asleep).
#[cfg(target_arch = "wasm32")]
type SchedTx = crate::rt::channel::Sender<SchedulerItem>;
#[cfg(target_arch = "wasm32")]
struct SchedRx(crate::rt::channel::Receiver<SchedulerItem>);

#[cfg(target_arch = "wasm32")]
impl SchedRx {
    async fn recv(&mut self) -> Option<SchedulerItem> {
        self.0.recv_async().await.ok()
    }

    fn try_recv(&mut self) -> Result<SchedulerItem, crate::rt::channel::TryRecvError> {
        self.0.try_recv()
    }
}

fn sched_channel() -> (SchedTx, SchedRx) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        tokio::sync::mpsc::unbounded_channel::<SchedulerItem>()
    }
    #[cfg(target_arch = "wasm32")]
    {
        let (tx, rx) = crate::rt::channel::unbounded::<SchedulerItem>();
        (tx, SchedRx(rx))
    }
}
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
                    Err("the engine lane dropped this launch unanswered"
                        .to_string()
                        .into())
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
    /// hands back its compile (`Programs::prepare`), which runs off
    /// the lane while the lane fires on.
    PrepareProgram {
        program: Box<ProgramRegistration>,
        response: tokio::sync::oneshot::Sender<Result<::engine::ProgramCompile>>,
    },
    /// A program's registration, last step: the engine takes what its
    /// compile made (`Programs::finish`), and the lane learns what an
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
    #[allow(
        clippy::vec_box,
        reason = "as `PendingLaunchBatch::requests`: the box keeps each move of a 1408-byte request 8 bytes"
    )]
    requests: Vec<Box<PendingRequest>>,
    /// Fired again one by one: refused again after the move landed, so
    /// only the fires the engine still refuses alone fail.
    solo: bool,
    /// The planner was asked for the free chunks its move lacks.
    funding: bool,
    /// What its fires' pipelines queued behind them, in order: each reads
    /// what they write, so it goes behind them ([`BatchScheduler::hold_behind`]).
    later: Vec<QueuedItem>,
}

impl HeldForPlacement {
    fn holds(&self, pipeline: ProcessId) -> bool {
        self.requests
            .iter()
            .any(|request| request.pipeline_id == Some(pipeline))
    }
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

    /// The item at `at` comes to the front; every other keeps its order. A
    /// launch never moves behind a later one: a cut pass's pieces, each
    /// reading what the one before it writes, reach the device in the order
    /// they were queued.
    fn bring_to_front(&mut self, at: usize) {
        if let Some(item) = self.items.remove(at) {
            self.items.push_front(item);
        }
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// A launch queued behind a later one of its pipeline, `(pipeline, the
    /// later fire ahead, the fire behind it)`: a mover of the queue broke
    /// their order.
    fn out_of_order(&self) -> Option<(ProcessId, u64, u64)> {
        let mut last: HashMap<ProcessId, u64> = HashMap::new();
        self.items.iter().find_map(|item| {
            let QueuedItem::Launch(launch) = item else {
                return None;
            };
            let pipeline = launch.pipeline_id?;
            let fire = launch.request.logical_fire_id;
            match last.insert(pipeline, fire) {
                Some(ahead) if ahead > fire => Some((pipeline, ahead, fire)),
                _ => None,
            }
        })
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
        let (tx, rx) = sched_channel();
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
            Self::hold_behind(&mut held_for_placement, &mut pending);
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
            // Frames queued before the cap came down were cut at the rung the
            // engine still holds: they seal under it, and the rung comes down
            // once they have gone (the grant below waits on them). Capped
            // there, a complete frame would sit unsealed on a raise the
            // pressure that lowered the cap cannot fund. Past the grant (none
            // wider queued then), nothing wider seals: the rung may be coming
            // down.
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
            // unsealed (capped): lowered under it, the frame goes only once
            // raised again, a raise its owner's eviction cannot fund while
            // that eviction waits on the frame. Kept, the walk lifts the cap
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
            // A raise to held frames refused by name (it never fits beside
            // its owner's own state): the frames held for it leave, their
            // fires failed by that name, never held for good.
            if let Some((owner, why)) = crate::store::shared::take_refused_owed(engine_id) {
                let fires = frame_policy.refuse_held(owner);
                Self::reject_fires_queued(&mut pending, &fires, &why);
                progress |= !fires.is_empty();
            }
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
                crate::rt::tab_step("sleeps on its mailbox", "the scheduler");
                let received = rx.recv().await;
                crate::rt::tab_step("woke on its mailbox", "the scheduler");
                match received {
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
                crate::rt::tab_step("parks on its mailbox, settles and replies", "the scheduler");
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
                crate::rt::tab_step(
                    match &parked {
                        Err(_) => "woke on its backstop",
                        Ok(_) if reply_answer.is_some() => "woke on a lane reply",
                        Ok(_) => "woke on its mailbox or a settle",
                    },
                    "the scheduler",
                );
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
                if parked.is_err() && tracing::enabled!(tracing::Level::DEBUG) {
                    static LAST: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
                    let mut last = LAST.lock().unwrap();
                    let due = last.is_none_or(|at| at.elapsed() >= Duration::from_secs(3));
                    let owed = frame_policy.has_queued_frames()
                        || !pending.is_empty()
                        || !in_flight_launches.is_empty();
                    if due && owed {
                        *last = Some(Instant::now());
                        tracing::debug!(
                            pending = pending.len(),
                            in_flight = in_flight_launches.len(),
                            "scheduler stalled: {}",
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
                            tracing::error!(
                                engine_id,
                                "stalled for {stalled_for:?} (no progress, work queued or \
                                 in flight); state:\n{}",
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

    fn shutdown_instances(
        engine: &mut Option<EngineBox>,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
    ) {
        let outstanding = std::mem::take(instances);
        for (instance_id, instance) in outstanding {
            if let Some(engine) = engine.as_mut()
                && let Err(err) = serve(engine, Engine::programs, "close_instance")
                    .and_then(|programs| programs.close(instance_id))
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
                && let Err(err) = serve(engine, Engine::channels, "close_channel")
                    .and_then(|channels| channels.close(channel_id))
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
mod tests;

/// A test's own lane for an engine: the copies its planner and stores
/// submit (`copy_tier_tracked`) are handed to the test, which lands each
/// one (or refuses it) when it chooses; the pool requests a move waits on
/// (`settle_copies`, a rebind) are answered at once. Installed for this
/// thread alone ([`crate::scheduler::set_test_handle`]).
#[cfg(test)]
pub(crate) mod test_lane;
