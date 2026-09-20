use crate::rt::Instant;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use ::engine::{ChannelRegistration, ProgramRegistration, StateCopy};

use crate::engine::{
    BoundInstance, ChannelJoin, EngineBox, EngineId, InstanceBindingPlan, RegisteredChannel,
    SchedulerLimits, SubmissionCompletion, WorkItemAttemptOutcome, WorkItemCompletion,
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
    pub(crate) instance_id: u64,
    pub(crate) completion: WorkItemCompletion,
    pub(crate) process_id: Option<ProcessId>,
    pub(crate) pipeline_id: Option<ProcessId>,
    pub(crate) prelaunch_copy: Option<::engine::KvCopy>,
    pub(crate) prelaunch_state_copy: Option<StateCopy>,
    pub(crate) frame: Option<FrameStamp>,
    pub(crate) hook_program: bool,
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
                        std::mem::take(&mut *LINES.lock().unwrap_or_else(|e| e.into_inner()));
                    for line in drained {
                        eprintln!("{line}");
                    }
                }
            });
        }
        on
    });
    if buffered {
        LINES.lock().unwrap_or_else(|e| e.into_inner()).push(line);
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
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistration>,
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
    CloseChannels {
        ids: Vec<u64>,
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
    FrameTruncate {
        lane: ProcessId,
        seq: u64,
        submitted: u32,
    },
    LanePark {
        lane: ProcessId,
        seq: u64,
    },
    DebugDump {
        response: tokio::sync::oneshot::Sender<String>,
    },
    Stop,
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
                    Err("the engine lane dropped this launch unanswered".to_string())
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
pub(super) enum PreLaunchCopy {
    Kv(::engine::KvCopy),
    State(StateCopy),
}

impl PreLaunchCopy {
    pub(super) fn label(&self) -> &'static str {
        match self {
            Self::Kv(_) => "KV copy",
            Self::State(_) => "recurrent-state copy",
        }
    }
}

pub(super) enum BindRespond {
    Bind(tokio::sync::oneshot::Sender<Result<BoundInstance>>),
    ChannelsBind {
        registered: Vec<RegisteredChannel>,
        program_id: u64,
        program_registered: bool,
        response:
            tokio::sync::oneshot::Sender<Result<(Vec<RegisteredChannel>, u64, BoundInstance)>>,
    },
}

pub(super) enum QueuedItem {
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
    RegisterChannelsBind {
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistration>,
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
    CloseChannels {
        ids: Vec<u64>,
    },
}

pub(super) struct QueuedLaunch {
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
        } else {
            self.is_empty()
        }
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
    untracked: Option<u64>,
    drain_eligible: Vec<u64>,
}

impl QueueScan {
    fn clear(&mut self) {
        self.queued_ids.clear();
        self.blocked_lanes.clear();
        self.untracked = None;
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
    program_ids: Mutex<HashMap<u64, (u64, ::eta_compiler::codegen::launch::LaunchPackage)>>,
    accepting: AtomicBool,
    stats: Arc<SchedulerStats>,
    device_domain: ::engine::MemoryDomain,
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

    pub async fn register_channel(&self, plan: ChannelRegistration) -> Result<RegisteredChannel> {
        self.request(|response| SchedulerItem::RegisterChannel { plan, response })
            .await?
    }

    pub async fn register_channels(
        &self,
        plans: Vec<ChannelRegistration>,
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
        plans: Vec<ChannelRegistration>,
        program: ProgramRegistration,
        mut bind: InstanceBindingPlan,
    ) -> Result<(Vec<RegisteredChannel>, BoundInstance)> {
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

    pub(crate) fn copy_kv_tracked(&self, plan: ::engine::KvCopy) -> Result<ControlCompletion> {
        let completion = ControlCompletion::new();
        self.send(SchedulerItem::CopyKvTracked {
            plan,
            completion: completion.clone(),
        })?;
        Ok(completion)
    }

    pub(crate) async fn debug_dump(&self) -> Result<String> {
        crate::rt::time::timeout(
            std::time::Duration::from_secs(2),
            self.request(|response| SchedulerItem::DebugDump { response }),
        )
        .await
        .map_err(|_| anyhow!("scheduler did not answer the debug dump"))?
    }

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
    thread: Option<crate::rt::thread::JoinHandle<()>>,
    stats: Arc<SchedulerStats>,
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
        let mut in_flight_control = InFlightControls::default();
        let mut stopping = false;
        let mut frame_policy = FramePolicy::new(
            frame_size,
            limits.max_forward_requests,
            limits.max_forward_tokens,
            Some(Arc::clone(&stats)),
        );
        frame_policy
            .preload_free_slots(crate::inferlet::process::execution_slot_capacity().unwrap_or(0));
        let mut stall_since: Option<crate::rt::Instant> = None;
        let mut stall_dumps: u32 = 0;

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
                );
                progress = true;
            }
            loop {
                let Ok(item) = rx.try_recv() else { break };
                progress = true;
                Self::enqueue_item(
                    &mut pending,
                    &mut terminated_processes,
                    &in_flight_launches,
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
            );
            progress |= Self::retire_ready_control(&mut in_flight_control);
            let (dispatched, wait_hint) = Self::dispatch_ready_items(
                &lane,
                &mut replies,
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
                    &in_flight_launches,
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
        for control in in_flight_control.iter() {
            let state = match &control.state {
                ControlSlotState::Posted { id } => format!("posted(id={id})"),
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
        in_flight_launches: &VecDeque<PendingLaunchBatch>,
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
            SchedulerItem::DebugDump { response } => {
                let _ = response.send(Self::render_debug_dump(
                    pending,
                    in_flight_launches,
                    in_flight_control,
                    instances,
                    frame_policy,
                ));
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
                        let cohort = launch
                            .request
                            .lanes
                            .first()
                            .and_then(|lane| {
                                crate::pipeline::instance::cohort_key(lane.group, lane.peer)
                            })
                            .zip(launch.request.cohort);
                        frame_policy.on_fire_enqueued(
                            stamp,
                            launch.process_id,
                            launch.logical_fire_id,
                            launch.request.tokens(),
                            launch.wire_row_count(),
                            cohort,
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
                    Lane::release_wait_slots([plan.pacing_wait_id]);
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
                    Lane::release_wait_slots([bind.pacing_wait_id]);
                    let _ = response.send(Err(anyhow!(
                        "process departed before channel bind admission"
                    )));
                    return;
                }
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

    fn instance_has_queued_work(pending: &VecDeque<QueuedItem>, instance_id: u64) -> bool {
        pending.iter().any(|item| match item {
            QueuedItem::Launch(request) => request.instance_id == instance_id,
            _ => false,
        })
    }

    const fn standalone_copy(item: &QueuedItem) -> bool {
        matches!(
            item,
            QueuedItem::CopyKv { .. }
                | QueuedItem::CopyKvTracked { .. }
                | QueuedItem::CopyState { .. }
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
                    | QueuedItem::RegisterProgram { .. }
                    | QueuedItem::RegisterChannel { .. }
                    | QueuedItem::RegisterChannels { .. }
                    | QueuedItem::BindInstance { .. }
                    | QueuedItem::RegisterChannelsBind { .. }
                    | QueuedItem::CloseChannels { .. }
            )
    }

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
        pending.insert_before_closes(item);
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
        let hold_closes = !stopping && frame_policy.has_pending_binds();
        let slot_blocks_lifecycle = in_flight_control.holds_launches();
        while let Some(item) = pending.front() {
            match item {
                QueuedItem::Launch(_) => {
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
        (progress, wait_hint)
    }

    fn post_control(
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<u64, TrackedInstance>,
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
                        instance_id = id,
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
                });
            }
            QueuedItem::CopyKv { .. } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { id },
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
                    state: ControlSlotState::Posted { id },
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
                    state: ControlSlotState::Posted { id },
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

    #[allow(clippy::too_many_arguments)]
    fn dispatch_frame_work(
        scan_cache: &mut ScanCache,
        slot_buffer: &mut SlotBuffer,
        frame_policy: &mut FramePolicy,
        engine_loop: &Lane,
        replies: &mut LaneReplies,
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
            } else if let Some(untracked) = scan.untracked {
                rider_batch = true;
                if wave_trace() {
                    wave_trace_emit(format!(
                        "[wave-trace] t={}us rider fire={untracked}",
                        wave_trace_us()
                    ));
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
                    FramePlan::Terminate(doomed) => {
                        for (pid, doom) in doomed {
                            let reason = doom.to_string();
                            tracing::error!(pid = %pid, "scheduler: terminating pipeline: {reason}");
                            crate::inferlet::process::terminate(pid, Err(reason));
                        }
                        continue;
                    }
                }
            };
            let (frame_progress, posted) = Self::post_frame(
                slot_buffer,
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
            if rider_batch {
                frame_policy.record_rider_wave();
            }
        }
        (progress, wait_hint)
    }

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
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<u64, TrackedInstance>,
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
        replies.post_launch(engine_loop.fire(submission, total_tokens > batch_size as usize));
        in_flight_launches.push_back(PendingLaunchBatch {
            state: LaunchState::Posted,
            requests,
            started: Instant::now(),
            batch_size,
            total_tokens,
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
        instances: &mut HashMap<u64, TrackedInstance>,
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
            Err(message) => {
                batch.state = LaunchState::Failed(message);
            }
        }
    }

    fn apply_lane_reply(
        reply: LaneReply,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        instances: &mut HashMap<u64, TrackedInstance>,
        in_flight_control: &mut InFlightControls,
        frame_policy: &mut FramePolicy,
        rollback_tx: &SchedTx,
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
            ),
        }
    }

    fn apply_control_commit(
        id: u64,
        commit: LaneCommit,
        in_flight_control: &mut InFlightControls,
        instances: &mut HashMap<u64, TrackedInstance>,
        frame_policy: &mut FramePolicy,
        rollback_tx: &SchedTx,
    ) {
        match commit {
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
                    Err(_) => {
                        in_flight_control.settling.remove(index);
                    }
                }
            }
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

    struct PanickingEngine;

    impl engine::Engine for PanickingEngine {
        fn kind(&self) -> &'static str {
            "panicking"
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

    #[test]
    fn worker_every_case() {
        a_panicking_engine_fails_its_launch_instead_of_leaving_it_in_flight();
        a_retryable_refusal_past_admission_fails_by_name_instead_of_replaying();
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
                        instances: vec![0],
                        logical_fire_ids: vec![token],
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
        fn kind(&self) -> &'static str {
            "exhausted"
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
                    instances: vec![0],
                    logical_fire_ids: vec![42],
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
}
