//! Per-driver direct batch scheduler.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::driver::{
    BoundInstance, ChannelRegistrationPlan, DriverBackend, DriverId, InstanceBindingPlan,
    PoolResizePlan, ProgramRegistration, RegisteredChannel, SchedulerLimits, StateCopyPlan,
    SubmissionCompletion, WorkItemAttemptOutcome, WorkItemCompletion,
};
use crate::scheduler::ProcessId;
use crate::store::kv::project::PhysicalPageId;
use anyhow::{Result, anyhow};

use super::batch::{self, BatchAccumulator};
use super::quorum;
use super::stats::{self, SchedulerStats};
use super::{ControlCompletion, LaunchPreparation, LaunchPreparationError, RetryClassifier};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FireClause {
    Quorum,
    IdleEscape,
    ColdHold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaveKind {
    Terminate,
    Suspend,
}

/// A pipeline left the fleet (cancel / kill / exit / TASK-A terminate /
/// TASK-B preempt). Broadcasts to EVERY registered driver's scheduler
/// thread (a pipeline's requests may have landed on any of them) so each
/// thread's local [`quorum::WaitAllPolicy`] drops `pid` from its wave
/// wait-set. Fire-and-forget: a shutting-down/closed scheduler channel is
/// silently skipped (nothing left there to notify). `_kind` is unused —
/// the quorum drops a pipeline the same way whether it terminated or was
/// merely suspended (see [`quorum::WaitAllPolicy::on_pipeline_leave`]'s
/// doc); kept on the signature so call sites document which case fired.
pub(crate) fn notify_pipeline_leave(pid: ProcessId, _kind: LeaveKind) {
    let handles = super::handle_registry().read().unwrap();
    for handle in handles.iter().flatten() {
        let _ = handle.send(SchedulerItem::PipelineLeave(pid, _kind));
    }
}
/// No-op: quorum rejoin is implicit on the pipeline's next
/// [`quorum::WaitAllPolicy::on_pipeline_request`] call, so a join event has
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
    pub(crate) physical_page_ids: Vec<PhysicalPageId>,
    pub(crate) last_page_len: u32,
    /// The submitting pipeline's identity (`ProcessCtx::process_id()`), or
    /// `None` for an untracked/prebuilt fire (device-geometry, beam replay,
    /// this module's own unit tests) — the quorum wait-set key
    /// ([`quorum::WaitAllPolicy::on_pipeline_request`]).
    pub(crate) pipeline_id: Option<ProcessId>,
    /// Submit-order lane for deferred preparation. Requests in the same
    /// `(pipeline_id, key)` lane may not prepare out of logical-fire order,
    /// while unrelated lanes may pass a retrying preparation.
    pub(crate) preparation_order_key: Option<u64>,
    pub(crate) prebuilt: bool,
    /// Stable logical-fire retry state. The request payload and completion are
    /// retained across attempts; only the native terminal cell is reset.
    pub(crate) retry_count: u32,
    pub(crate) prelaunch_copy: Option<crate::driver::KvCopyPlan>,
    pub(crate) prelaunch_state_copy: Option<StateCopyPlan>,
    pub(crate) preparation: Option<LaunchPreparation>,
    pub(crate) retry_classifier: Option<RetryClassifier>,
    pub(crate) preparation_retries: u32,
}

impl PendingRequest {
    fn direct(
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        pipeline_id: Option<ProcessId>,
        preparation_order_key: Option<u64>,
        prebuilt: bool,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
        preparation: Option<LaunchPreparation>,
        retry_classifier: Option<RetryClassifier>,
    ) -> Self {
        Self {
            logical_fire_id: NEXT_LOGICAL_FIRE_ID.fetch_add(1, Ordering::Relaxed),
            request,
            instance_id,
            completion,
            physical_page_ids,
            last_page_len,
            pipeline_id,
            preparation_order_key,
            prebuilt,
            retry_count: 0,
            prelaunch_copy,
            prelaunch_state_copy,
            preparation,
            retry_classifier,
            preparation_retries: 0,
        }
    }

    fn retry_eligible(&self) -> bool {
        self.request.rs_slot_ids.is_empty()
            && self.request.rs_buffer_slot_ids.is_empty()
            && self.request.rs_fold_lens.is_empty()
    }

    pub(crate) fn wire_row_count(&self) -> usize {
        self.request.qo_indptr.len().saturating_sub(1)
    }

    pub(crate) fn preserves_inner_rows(&self) -> bool {
        self.wire_row_count() > 1
    }

    fn requires_solo_submission(&self) -> bool {
        (self.prebuilt && self.pipeline_id.is_none()) || self.preserves_inner_rows()
    }
}

#[derive(Default)]
struct LaunchGrouping {
    instances: HashSet<u64>,
    count: usize,
    forward_tokens: usize,
    page_refs: usize,
    has_solo_submission: bool,
    has_user_mask: bool,
}

impl LaunchGrouping {
    fn accepts(&self, request: &PendingRequest, limits: SchedulerLimits, page_size: u32) -> bool {
        if self.instances.contains(&request.instance_id) {
            return false;
        }
        if self.count != 0
            && (request.requires_solo_submission()
                || self.has_solo_submission
                || request.request.has_user_mask
                || self.has_user_mask)
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
        self.count = self.count.saturating_add(usage.forward_requests);
        self.forward_tokens = self.forward_tokens.saturating_add(usage.forward_tokens);
        self.page_refs = self.page_refs.saturating_add(usage.page_refs);
        self.has_solo_submission |= request.requires_solo_submission();
        self.has_user_mask |= request.request.has_user_mask;
        request.requires_solo_submission()
            || request.request.has_user_mask
            || self.count >= limits.max_forward_requests
            || self.forward_tokens >= limits.max_forward_tokens
            || self.page_refs >= limits.max_page_refs
    }
}

enum SchedulerItem {
    Launch {
        pending: PendingRequest,
    },
    RegisterProgram {
        plan: ProgramRegistration,
        response: crossbeam::channel::Sender<Result<u64>>,
    },
    RegisterChannel {
        plan: ChannelRegistrationPlan,
        response: crossbeam::channel::Sender<Result<RegisteredChannel>>,
    },
    BindInstance {
        plan: InstanceBindingPlan,
        response: crossbeam::channel::Sender<Result<BoundInstance>>,
    },
    CopyKv {
        plan: crate::driver::KvCopyPlan,
        response: crossbeam::channel::Sender<Result<SubmissionCompletion>>,
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
        response: crossbeam::channel::Sender<Result<SubmissionCompletion>>,
    },
    #[allow(dead_code)]
    ResizePool {
        plan: PoolResizePlan,
        response: crossbeam::channel::Sender<Result<SubmissionCompletion>>,
    },
    CloseInstance {
        id: u64,
        pacing_wait_id: u64,
        response: crossbeam::channel::Sender<Result<()>>,
    },
    CloseChannel {
        id: u64,
        response: crossbeam::channel::Sender<Result<()>>,
    },
    FreezePipeline {
        pid: ProcessId,
        response: crossbeam::channel::Sender<()>,
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

enum QueuedItem {
    Launch(PendingRequest),
    Prepare(PendingRequest),
    PreLaunchCopy {
        plan: PreLaunchCopy,
        logical_completion: WorkItemCompletion,
        pipeline_id: Option<ProcessId>,
    },
    RegisterProgram {
        plan: ProgramRegistration,
        response: crossbeam::channel::Sender<Result<u64>>,
    },
    RegisterChannel {
        plan: ChannelRegistrationPlan,
        response: crossbeam::channel::Sender<Result<RegisteredChannel>>,
    },
    BindInstance {
        plan: InstanceBindingPlan,
        response: crossbeam::channel::Sender<Result<BoundInstance>>,
    },
    CopyKv {
        plan: crate::driver::KvCopyPlan,
        response: crossbeam::channel::Sender<Result<SubmissionCompletion>>,
    },
    CopyKvTracked {
        plan: crate::driver::KvCopyPlan,
        completion: ControlCompletion,
    },
    CopyState {
        plan: StateCopyPlan,
        response: crossbeam::channel::Sender<Result<SubmissionCompletion>>,
    },
    ResizePool {
        plan: PoolResizePlan,
        response: crossbeam::channel::Sender<Result<SubmissionCompletion>>,
    },
    CloseInstance {
        id: u64,
        pacing_wait_id: u64,
        response: crossbeam::channel::Sender<Result<()>>,
    },
    CloseChannel {
        id: u64,
        response: crossbeam::channel::Sender<Result<()>>,
    },
}

#[derive(Clone, Copy)]
enum QueueEnd {
    Front,
    Back,
}

struct PendingLaunchBatch {
    completion: SubmissionCompletion,
    requests: Vec<PendingRequest>,
    started: Instant,
    batch_size: u64,
    total_tokens: usize,
}

struct PendingControl {
    completion: SubmissionCompletion,
    logical_completion: Option<WorkItemCompletion>,
    pipeline_id: Option<ProcessId>,
    tracked_completion: Option<ControlCompletion>,
    operation: &'static str,
}

struct SchedulerControl {
    tx: crossbeam::channel::Sender<SchedulerItem>,
    gate: Mutex<()>,
    accepting: AtomicBool,
    stats: Arc<SchedulerStats>,
}

#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    inner: Arc<SchedulerControl>,
}

impl SchedulerHandle {
    fn send(&self, item: SchedulerItem) -> Result<()> {
        let _guard = self.inner.gate.lock().unwrap();
        if !self.inner.accepting.load(Ordering::Acquire) {
            return Err(anyhow!("scheduler shutting down"));
        }
        self.inner
            .tx
            .send(item)
            .map_err(|_| anyhow!("scheduler channel closed"))
    }

    fn begin_shutdown(&self) {
        let _guard = self.inner.gate.lock().unwrap();
        if !self.inner.accepting.swap(false, Ordering::AcqRel) {
            return;
        }
        let _ = self.inner.tx.send(SchedulerItem::Stop);
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
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        pipeline_id: Option<ProcessId>,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                physical_page_ids,
                last_page_len,
                pipeline_id,
                None,
                false,
                prelaunch_copy,
                prelaunch_state_copy,
                None,
                None,
            ),
        })
    }

    pub fn submit_prebuilt_with_copy(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                physical_page_ids,
                last_page_len,
                None,
                None,
                true,
                prelaunch_copy,
                prelaunch_state_copy,
                None,
                None,
            ),
        })
    }

    pub fn submit_prebuilt_tracked_with_copy(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        pipeline_id: ProcessId,
        prelaunch_copy: Option<crate::driver::KvCopyPlan>,
        prelaunch_state_copy: Option<StateCopyPlan>,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                physical_page_ids,
                last_page_len,
                Some(pipeline_id),
                None,
                true,
                prelaunch_copy,
                prelaunch_state_copy,
                None,
                None,
            ),
        })
    }

    pub fn submit_deferred(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: WorkItemCompletion,
        pipeline_id: Option<ProcessId>,
        preparation_order_key: Option<u64>,
        prelaunch_state_copy: Option<StateCopyPlan>,
        preparation: LaunchPreparation,
        retry_classifier: Option<RetryClassifier>,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                Vec::new(),
                0,
                pipeline_id,
                preparation_order_key,
                false,
                None,
                prelaunch_state_copy,
                Some(preparation),
                retry_classifier,
            ),
        })
    }

    pub(crate) fn nudge(&self) -> Result<()> {
        self.send(SchedulerItem::Nudge)
    }

    pub(crate) fn freeze_pipeline(&self, pid: ProcessId) -> Result<()> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::FreezePipeline { pid, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))
    }

    pub(crate) fn resume_pipeline(&self, pid: ProcessId) -> Result<()> {
        self.send(SchedulerItem::ResumePipeline(pid))
    }

    pub fn register_program(&self, plan: ProgramRegistration) -> Result<u64> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::RegisterProgram { plan, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    pub fn register_channel(&self, plan: ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::RegisterChannel { plan, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    pub fn bind_instance(&self, plan: InstanceBindingPlan) -> Result<BoundInstance> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::BindInstance { plan, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    pub fn copy_kv(&self, plan: crate::driver::KvCopyPlan) -> Result<SubmissionCompletion> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::CopyKv { plan, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
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

    // Only called from `scheduler::dispatch::copy_rs_d2d`/`resize_pool`
    // (not yet issued by the mock-driver fire path) and this module's own
    // unit tests — see `scheduler::dispatch`'s module doc.
    #[allow(dead_code)]
    pub fn copy_state(&self, plan: StateCopyPlan) -> Result<SubmissionCompletion> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::CopyState { plan, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    #[allow(dead_code)]
    pub fn resize_pool(&self, plan: PoolResizePlan) -> Result<SubmissionCompletion> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::ResizePool { plan, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    pub fn close_instance(&self, id: u64, pacing_wait_id: u64) -> Result<()> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::CloseInstance {
            id,
            pacing_wait_id,
            response: tx,
        })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    pub fn close_channel(&self, id: u64) -> Result<()> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::CloseChannel { id, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
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
                gate: Mutex::new(()),
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
        let nudge_waker = std::task::Waker::from(Arc::new(NudgeWaker { tx: nudge_tx }));
        let mut driver = crate::driver::take_driver_backend(driver_id).ok();
        let mut instances = HashMap::new();
        let mut channels = HashSet::new();
        let mut pending = VecDeque::new();
        let mut blocked_preparations = VecDeque::new();
        let mut frozen_pipelines = HashSet::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = None;
        let mut stopping = false;
        // The wait-for-all-active-pipelines fire rule (overview §7.2): one
        // instance per driver thread, mirroring `instances`/`channels` above.
        let mut policy =
            quorum::WaitAllPolicy::new(limits.max_forward_requests, Some(Arc::clone(&stats)));

        loop {
            let mut progress = false;
            progress |= Self::retire_ready_launches(
                &mut in_flight_launches,
                &mut instances,
                &mut pending,
                &stats,
                &mut policy,
                stopping,
            );
            progress |= Self::retire_ready_control(&mut in_flight_control);
            let (dispatched, wait_hint) = Self::dispatch_ready_items(
                &mut driver,
                &mut instances,
                &mut channels,
                &mut pending,
                &mut blocked_preparations,
                &frozen_pipelines,
                &mut in_flight_launches,
                &mut in_flight_control,
                page_size,
                limits,
                &stats,
                &mut policy,
                stopping,
            );
            progress |= dispatched;
            // M-A2: reap any pipeline the quorum just demoted for missing
            // too many consecutive wave deadlines. Drained every pass (not
            // only on a demoting fire) so a demotion is never left stranded.
            for pid in policy.take_terminate_candidates() {
                crate::scheduler::terminate_demoted_pipeline(pid);
            }

            if stopping
                && pending.is_empty()
                && blocked_preparations.is_empty()
                && in_flight_launches.is_empty()
                && in_flight_control.is_none()
            {
                break;
            }

            while let Ok(item) = rx.try_recv() {
                progress = true;
                Self::enqueue_item(
                    &mut pending,
                    &mut blocked_preparations,
                    &mut frozen_pipelines,
                    &mut in_flight_control,
                    &instances,
                    limits,
                    page_size,
                    &mut stopping,
                    &mut policy,
                    item,
                );
            }

            if progress {
                continue;
            }

            let item = if pending.is_empty()
                && blocked_preparations.is_empty()
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
                    armed &= arm_completion_nudge(&front.completion, &nudge_waker);
                }
                if let Some(control) = in_flight_control.as_ref() {
                    armed &= arm_completion_nudge(&control.completion, &nudge_waker);
                }
                if !armed {
                    // Something already settled; retire it on the next pass.
                    continue;
                }
                // A pending quorum hold (cold-hold gather / straggler
                // deadline / depth-cap poll) re-arms the backstop at its own
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
                        let missed = in_flight_launches
                            .front()
                            .is_some_and(|front| front.completion.is_settled())
                            || in_flight_control
                                .as_ref()
                                .is_some_and(|control| control.completion.is_settled());
                        if missed && !stopping && wait_hint.is_none() {
                            let total = BACKSTOP_RETIREMENTS.fetch_add(1, Ordering::Relaxed) + 1;
                            tracing::warn!(
                                driver_id,
                                total,
                                "completion retired by the backstop poll, not the nudge"
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
                Self::enqueue_item(
                    &mut pending,
                    &mut blocked_preparations,
                    &mut frozen_pipelines,
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

        Self::shutdown_instances(&mut driver, &mut instances);
        Self::shutdown_channels(&mut driver, &mut channels);
        drop(driver.take());
    }

    fn enqueue_item(
        pending: &mut VecDeque<QueuedItem>,
        blocked_preparations: &mut VecDeque<PendingRequest>,
        frozen_pipelines: &mut HashSet<ProcessId>,
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
                pending.extend(blocked_preparations.drain(..).map(QueuedItem::Prepare));
            }
            // A nudge only unblocks the wait; the retire pass at the top of
            // the loop does the work.
            SchedulerItem::Nudge => {
                pending.extend(blocked_preparations.drain(..).map(QueuedItem::Prepare));
            }
            SchedulerItem::FreezePipeline { pid, response } => {
                frozen_pipelines.insert(pid);
                let _ = response.send(());
            }
            SchedulerItem::ResumePipeline(pid) => {
                frozen_pipelines.remove(&pid);
                pending.extend(blocked_preparations.drain(..).map(QueuedItem::Prepare));
            }
            // Immediate, not queued: this only mutates the local quorum
            // policy (never `pending`/`instances`), so it can't reorder
            // control ops or launches.
            SchedulerItem::PipelineLeave(pid, kind) => {
                policy.on_pipeline_leave(pid);
                if kind == LeaveKind::Terminate {
                    frozen_pipelines.remove(&pid);
                    let protected = in_flight_control
                        .as_ref()
                        .filter(|control| control.pipeline_id == Some(pid))
                        .and_then(|control| control.logical_completion.clone());
                    if let Some(completion) = &protected {
                        completion.request_cancel();
                    }
                    Self::reject_pipeline_queued(
                        pending,
                        blocked_preparations,
                        policy,
                        pid,
                        protected.as_ref(),
                    );
                }
            }
            SchedulerItem::Launch { pending: launch } => {
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
                } else if launch.preparation.is_some()
                    && launch
                        .pipeline_id
                        .is_some_and(|pid| frozen_pipelines.contains(&pid))
                {
                    blocked_preparations.push_back(launch);
                } else if launch.preparation.is_some() {
                    policy.on_retry_participation(launch.pipeline_id);
                    pending.push_back(QueuedItem::Prepare(launch));
                } else {
                    // The wave gather starts at acceptance, not dispatch:
                    // this request now counts toward `decide_wave_at`'s
                    // wait-set/untracked-ready even while it sits in
                    // `pending` behind an in-flight-depth or quorum hold.
                    policy.on_pipeline_request(launch.pipeline_id, Instant::now());
                    Self::queue_attempt(pending, launch, QueueEnd::Back);
                }
            }

            SchedulerItem::RegisterProgram { plan, response } => {
                pending.push_back(QueuedItem::RegisterProgram { plan, response });
            }
            SchedulerItem::RegisterChannel { plan, response } => {
                pending.push_back(QueuedItem::RegisterChannel { plan, response });
            }
            SchedulerItem::BindInstance { plan, response } => {
                pending.push_back(QueuedItem::BindInstance { plan, response });
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
            SchedulerItem::CloseInstance {
                id,
                pacing_wait_id,
                response,
            } => {
                pending.push_back(QueuedItem::CloseInstance {
                    id,
                    pacing_wait_id,
                    response,
                });
            }
            SchedulerItem::CloseChannel { id, response } => {
                pending.push_back(QueuedItem::CloseChannel { id, response });
            }
        }
    }

    fn reject_pipeline_queued(
        pending: &mut VecDeque<QueuedItem>,
        blocked_preparations: &mut VecDeque<PendingRequest>,
        policy: &mut quorum::WaitAllPolicy,
        pid: ProcessId,
        protected: Option<&WorkItemCompletion>,
    ) {
        let mut kept = VecDeque::with_capacity(pending.len());
        while let Some(item) = pending.pop_front() {
            let reject = match &item {
                QueuedItem::Prepare(request) | QueuedItem::Launch(request) => {
                    request.pipeline_id == Some(pid)
                        && protected
                            .is_none_or(|completion| !request.completion.same_request(completion))
                }
                QueuedItem::PreLaunchCopy {
                    pipeline_id,
                    logical_completion,
                    ..
                } => {
                    *pipeline_id == Some(pid)
                        && protected
                            .is_none_or(|completion| !logical_completion.same_request(completion))
                }
                _ => false,
            };
            if reject {
                match item {
                    QueuedItem::Prepare(request) => request
                        .completion
                        .reject_unsubmitted("pipeline terminated while queued"),
                    QueuedItem::Launch(request) => {
                        policy.on_request_dropped(request.pipeline_id);
                        request
                            .completion
                            .reject_unsubmitted("pipeline terminated while queued");
                    }
                    QueuedItem::PreLaunchCopy {
                        logical_completion, ..
                    } => logical_completion
                        .reject_unsubmitted("pipeline terminated before pre-launch copy"),
                    _ => unreachable!("rejected item kind checked above"),
                }
            } else {
                kept.push_back(item);
            }
        }
        *pending = kept;

        let mut kept = VecDeque::with_capacity(blocked_preparations.len());
        while let Some(request) = blocked_preparations.pop_front() {
            if request.pipeline_id == Some(pid)
                && protected.is_none_or(|completion| !request.completion.same_request(completion))
            {
                request
                    .completion
                    .reject_unsubmitted("pipeline terminated while preparation was blocked");
            } else {
                kept.push_back(request);
            }
        }
        *blocked_preparations = kept;
    }

    fn queue_attempt(pending: &mut VecDeque<QueuedItem>, request: PendingRequest, end: QueueEnd) {
        let mut copies = Vec::with_capacity(2);
        if let Some(plan) = request.prelaunch_copy.clone() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::Kv(plan),
                logical_completion: request.completion.clone(),
                pipeline_id: request.pipeline_id,
            });
        }
        if let Some(plan) = request.prelaunch_state_copy.clone() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::State(plan),
                logical_completion: request.completion.clone(),
                pipeline_id: request.pipeline_id,
            });
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

    fn preparation_has_earlier_lane_member(
        pending: &VecDeque<QueuedItem>,
        request: &PendingRequest,
    ) -> bool {
        let Some(key) = request.preparation_order_key else {
            return false;
        };
        pending.iter().skip(1).any(|item| {
            let QueuedItem::Prepare(earlier) = item else {
                return false;
            };
            earlier.pipeline_id == request.pipeline_id
                && earlier.preparation_order_key == Some(key)
                && earlier.logical_fire_id < request.logical_fire_id
        })
    }

    fn preparation_has_blocked_lane_head(
        blocked: &VecDeque<PendingRequest>,
        request: &PendingRequest,
    ) -> bool {
        let Some(key) = request.preparation_order_key else {
            return false;
        };
        blocked.iter().any(|earlier| {
            earlier.pipeline_id == request.pipeline_id
                && earlier.preparation_order_key == Some(key)
                && earlier.logical_fire_id < request.logical_fire_id
        })
    }

    /// Peeks how many requests at `pending`'s front would land in the NEXT
    /// launch batch — same grouping rules `dispatch_launch_batch` applies
    /// (same-instance dedup, mask-solo, structural capacity) — without
    /// mutating the queue or the driver. Feeds `WaitAllPolicy::
    /// decide_wave_at`'s `current_batch_size` so the quorum decision sees
    /// the exact geometry the dispatcher is about to build (a stale item —
    /// its instance closed after enqueue — is skipped here exactly like
    /// the real dispatch skips/rejects it, so it never inflates the count).
    fn peek_launch_batch_size(
        pending: &VecDeque<QueuedItem>,
        instances: &HashMap<u64, TrackedInstance>,
        limits: SchedulerLimits,
        page_size: u32,
    ) -> usize {
        let mut grouping = LaunchGrouping::default();
        let mut blocked_pipelines = HashSet::new();
        for item in pending.iter() {
            let QueuedItem::Launch(next) = item else {
                break;
            };
            if !instances.contains_key(&next.instance_id) {
                continue;
            }
            if next
                .pipeline_id
                .is_some_and(|pid| blocked_pipelines.contains(&pid))
            {
                continue;
            }
            if !grouping.accepts(next, limits, page_size) {
                if grouping.instances.contains(&next.instance_id)
                    && let Some(pid) = next.pipeline_id
                {
                    blocked_pipelines.insert(pid);
                    continue;
                }
                break;
            }
            if grouping.push(next, limits, page_size) {
                break;
            }
        }
        grouping.count
    }

    fn dispatch_ready_items(
        driver: &mut Option<DriverBackend>,
        instances: &mut HashMap<u64, TrackedInstance>,
        channels: &mut HashSet<u64>,
        pending: &mut VecDeque<QueuedItem>,
        blocked_preparations: &mut VecDeque<PendingRequest>,
        frozen_pipelines: &HashSet<ProcessId>,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &mut Option<PendingControl>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        policy: &mut quorum::WaitAllPolicy,
        stopping: bool,
    ) -> (bool, Option<Duration>) {
        let mut progress = false;
        let mut wait_hint = None;
        loop {
            if in_flight_control.is_some() {
                break;
            }
            let Some(item) = pending.front() else {
                break;
            };
            match item {
                QueuedItem::Prepare(_) => {
                    if pending.front().is_some_and(|item| {
                        matches!(
                            item,
                            QueuedItem::Prepare(request)
                                if request
                                    .pipeline_id
                                    .is_some_and(|pid| frozen_pipelines.contains(&pid))
                        )
                    }) {
                        let Some(QueuedItem::Prepare(request)) = pending.pop_front() else {
                            unreachable!();
                        };
                        blocked_preparations.push_back(request);
                        progress = true;
                        continue;
                    }
                    if pending.front().is_some_and(|item| {
                        matches!(
                            item,
                            QueuedItem::Prepare(request)
                                if request.completion.cancel_requested()
                        )
                    }) {
                        let Some(QueuedItem::Prepare(request)) = pending.pop_front() else {
                            unreachable!();
                        };
                        request.completion.reject_unsubmitted(
                            "logical fire cancelled during dispatch preparation",
                        );
                        progress = true;
                        continue;
                    }
                    if pending.front().is_some_and(|item| {
                        matches!(
                            item,
                            QueuedItem::Prepare(request)
                                if Self::preparation_has_blocked_lane_head(
                                    blocked_preparations,
                                    request,
                                )
                        )
                    }) {
                        let Some(QueuedItem::Prepare(request)) = pending.pop_front() else {
                            unreachable!();
                        };
                        blocked_preparations.push_back(request);
                        progress = true;
                        continue;
                    }
                    let should_defer = match pending.front() {
                        Some(QueuedItem::Prepare(request)) => {
                            Self::preparation_has_earlier_lane_member(pending, request)
                        }
                        _ => false,
                    };
                    if should_defer {
                        let request = pending.pop_front().expect("prepare front");
                        pending.push_back(request);
                        progress = true;
                        continue;
                    }
                    let Some(QueuedItem::Prepare(mut request)) = pending.pop_front() else {
                        unreachable!();
                    };
                    if stopping {
                        request.completion.reject_unsubmitted(
                            "scheduler shutdown interrupted dispatch preparation",
                        );
                        progress = true;
                        continue;
                    }
                    let result = crate::probe_fire!(
                        stats.fire.pre_dispatch.fire_prepare_us,
                        request
                            .preparation
                            .as_mut()
                            .expect("prepare item carries a preparation")(
                            &mut request.request
                        )
                    );
                    match result {
                        Ok(prepared) => {
                            request.physical_page_ids = prepared.page_refs;
                            request.last_page_len = prepared.last_page_len;
                            request.request.kv_translation_version =
                                prepared.kv_translation_version;
                            request.prelaunch_copy = (!prepared.copy_src.is_empty()).then_some(
                                crate::driver::KvCopyPlan {
                                    src_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
                                    src_device_ordinal: 0,
                                    dst_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
                                    dst_device_ordinal: 0,
                                    src_page_ids: prepared.copy_src,
                                    dst_page_ids: prepared.copy_dst,
                                    cells: Vec::new(),
                                },
                            );
                            request.preparation = None;
                            let validation = BatchAccumulator::new(limits, page_size);
                            if request.completion.cancel_requested() {
                                request.completion.reject_unsubmitted(
                                    "logical fire cancelled during dispatch preparation",
                                );
                            } else if let Some(message) =
                                validation.single_request_limit_error(&request)
                            {
                                request.completion.reject_unsubmitted(message);
                            } else {
                                policy.on_pipeline_request(request.pipeline_id, Instant::now());
                                Self::queue_attempt(pending, request, QueueEnd::Back);
                            }
                        }
                        Err(LaunchPreparationError::Retry(reason)) => {
                            request.preparation_retries += 1;
                            if request.completion.cancel_requested() {
                                request.completion.reject_unsubmitted(
                                    "logical fire cancelled during dispatch preparation",
                                );
                            } else if request.preparation_retries > max_fire_retries() {
                                request.completion.reject_unsubmitted(format!(
                                    "dispatch preparation exceeded retry limit: {reason}"
                                ));
                            } else {
                                policy.on_retry_participation(request.pipeline_id);
                                pending.push_back(QueuedItem::Prepare(request));
                                wait_hint =
                                    Some(Duration::from_micros(super::quorum::QUORUM_POLL_US));
                            }
                            progress = true;
                            break;
                        }
                        Err(LaunchPreparationError::Blocked(_reason)) => {
                            blocked_preparations.push_back(request);
                        }
                        Err(LaunchPreparationError::Failed(reason)) => {
                            request.completion.reject_unsubmitted(reason);
                        }
                    }
                    progress = true;
                }
                QueuedItem::Launch(_) => {
                    if in_flight_launches.len() >= quorum::configured_max_in_flight() {
                        break;
                    }
                    let candidate_size =
                        Self::peek_launch_batch_size(pending, instances, limits, page_size);
                    if !stopping {
                        match policy.decide_wave_at(candidate_size, Instant::now()) {
                            quorum::WaveDecision::Wait(hold) => {
                                wait_hint = Some(hold);
                                break;
                            }
                            quorum::WaveDecision::Fire { .. } => {}
                        }
                    }
                    let before = in_flight_launches.len();
                    let dispatched = crate::probe_fire!(
                        stats.fire.execute.total_us,
                        Self::dispatch_launch_batch(
                            driver,
                            instances,
                            pending,
                            in_flight_launches,
                            page_size,
                            limits,
                            stats,
                            policy,
                        )
                    );
                    // Only a batch that actually reached `in_flight_launches`
                    // (the driver accepted the launch) increments the
                    // policy's depth counter — a synchronous
                    // stale-instance-reject or `driver.launch` failure never
                    // occupies a run-ahead slot, so nothing would ever
                    // decrement it back.
                    if in_flight_launches.len() > before {
                        let accepted = in_flight_launches
                            .back()
                            .expect("accepted batch is present");
                        let participants = accepted
                            .requests
                            .iter()
                            .map(|request| request.pipeline_id)
                            .collect::<Vec<_>>();
                        policy.on_wave_dispatched(&participants, Instant::now());
                        if super::sched_trace_enabled() {
                            let mut queued_launches = 0usize;
                            let mut queued_preparations = 0usize;
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
                                    QueuedItem::Prepare(_) => {
                                        queued_preparations += 1;
                                    }
                                    _ => queued_controls += 1,
                                }
                            }
                            super::sched_trace_write(format_args!(
                                concat!(
                                    "wave candidate={} dispatched={} active={} ",
                                    "pending={} launches={} preparations={} ",
                                    "controls={} ready_pipelines={}"
                                ),
                                candidate_size,
                                accepted.requests.len(),
                                policy.active_pipelines(),
                                pending.len(),
                                queued_launches,
                                queued_preparations,
                                queued_controls,
                                queued_pipelines.len(),
                            ));
                        }
                    }
                    progress |= dispatched;
                    if !dispatched {
                        break;
                    }
                }
                _ if !in_flight_launches.is_empty() => break,
                _ => {
                    let item = pending.pop_front().expect("front item present");
                    Self::dispatch_ordered_item(
                        driver,
                        instances,
                        channels,
                        in_flight_control,
                        item,
                    );
                    progress = true;
                }
            }
        }
        (progress, wait_hint)
    }

    fn dispatch_ordered_item(
        driver: &mut Option<DriverBackend>,
        instances: &mut HashMap<u64, TrackedInstance>,
        channels: &mut HashSet<u64>,
        in_flight_control: &mut Option<PendingControl>,
        item: QueuedItem,
    ) {
        match item {
            QueuedItem::Launch(_) | QueuedItem::Prepare(_) => unreachable!(),
            QueuedItem::PreLaunchCopy {
                plan: _,
                logical_completion,
                ..
            } if logical_completion.is_settled() => {}
            QueuedItem::PreLaunchCopy {
                plan: _,
                logical_completion,
                ..
            } if logical_completion.cancel_requested() => {
                logical_completion
                    .reject_unsubmitted("logical fire cancelled before pre-launch copy");
            }
            QueuedItem::PreLaunchCopy {
                plan,
                logical_completion,
                pipeline_id,
            } => {
                let operation = plan.label();
                match driver.as_mut() {
                    Some(driver) => {
                        let submitted = match plan {
                            PreLaunchCopy::Kv(plan) => driver.copy_kv(&plan),
                            PreLaunchCopy::State(plan) => driver.copy_state(&plan),
                        };
                        match submitted {
                            Ok(completion) => {
                                *in_flight_control = Some(PendingControl {
                                    completion,
                                    logical_completion: Some(logical_completion),
                                    pipeline_id,
                                    tracked_completion: None,
                                    operation,
                                });
                            }
                            Err(error) => logical_completion.reject_unsubmitted(format!(
                                "pre-launch {operation} rejected: {error:#}"
                            )),
                        }
                    }
                    None => {
                        logical_completion.reject_unsubmitted("driver has no backend installed")
                    }
                }
            }
            QueuedItem::RegisterProgram { plan, response } => {
                let _ = response.send(match driver.as_mut() {
                    Some(driver) => driver.register_program(&plan),
                    None => Err(anyhow!("driver has no backend installed")),
                });
            }
            QueuedItem::RegisterChannel { plan, response } => {
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
                let _ = response.send(result);
            }
            QueuedItem::BindInstance { plan, response } => {
                let result = if plan.requested_instance_id != 0
                    && instances.contains_key(&plan.requested_instance_id)
                {
                    Err(anyhow!(
                        "instance {} is already bound",
                        plan.requested_instance_id
                    ))
                } else {
                    match driver.as_mut() {
                        Some(driver) => driver.bind_instance(&plan).and_then(|bound| {
                            if instances.contains_key(&bound.instance_id) {
                                let _ = driver.close_instance(bound.instance_id);
                                return Err(anyhow!(
                                    "instance {} is already bound",
                                    bound.instance_id
                                ));
                            }
                            instances
                                .insert(bound.instance_id, TrackedInstance::from_bound(&bound));
                            Ok(bound)
                        }),
                        None => Err(anyhow!("driver has no backend installed")),
                    }
                };
                let _ = response.send(result);
            }
            QueuedItem::CopyKv { plan, response } => {
                let _ = response.send(match driver.as_mut() {
                    Some(driver) => match driver.copy_kv(&plan) {
                        Ok(completion) => {
                            *in_flight_control = Some(PendingControl {
                                completion: completion.clone(),
                                logical_completion: None,
                                pipeline_id: None,
                                tracked_completion: None,
                                operation: "KV copy",
                            });
                            Ok(completion)
                        }
                        Err(err) => Err(err),
                    },
                    None => Err(anyhow!("driver has no backend installed")),
                });
            }
            QueuedItem::CopyKvTracked { plan, completion } => match driver.as_mut() {
                Some(driver) => match driver.copy_kv(&plan) {
                    Ok(native_completion) => {
                        *in_flight_control = Some(PendingControl {
                            completion: native_completion,
                            logical_completion: None,
                            pipeline_id: None,
                            tracked_completion: Some(completion),
                            operation: "tracked KV copy",
                        });
                    }
                    Err(error) => completion.resolve(&Err(error)),
                },
                None => completion.resolve(&Err(anyhow!("driver has no backend installed"))),
            },
            QueuedItem::CopyState { plan, response } => {
                let _ = response.send(match driver.as_mut() {
                    Some(driver) => match driver.copy_state(&plan) {
                        Ok(completion) => {
                            *in_flight_control = Some(PendingControl {
                                completion: completion.clone(),
                                logical_completion: None,
                                pipeline_id: None,
                                tracked_completion: None,
                                operation: "state copy",
                            });
                            Ok(completion)
                        }
                        Err(err) => Err(err),
                    },
                    None => Err(anyhow!("driver has no backend installed")),
                });
            }
            QueuedItem::ResizePool { plan, response } => {
                let _ = response.send(match driver.as_mut() {
                    Some(driver) => match driver.resize_pool(&plan) {
                        Ok(completion) => {
                            *in_flight_control = Some(PendingControl {
                                completion: completion.clone(),
                                logical_completion: None,
                                pipeline_id: None,
                                tracked_completion: None,
                                operation: "pool resize",
                            });
                            Ok(completion)
                        }
                        Err(err) => Err(err),
                    },
                    None => Err(anyhow!("driver has no backend installed")),
                });
            }
            QueuedItem::CloseInstance {
                id,
                pacing_wait_id,
                response,
            } => {
                let result = match instances.get(&id) {
                    Some(instance) if instance.pacing_wait_id == pacing_wait_id => {
                        if instance.in_flight != 0 {
                            Err(anyhow!("instance {id} is busy"))
                        } else {
                            match driver.as_mut() {
                                Some(driver) => match driver.close_instance(id) {
                                    Ok(()) => {
                                        if let Some(instance) = instances.remove(&id) {
                                            instance.close_wait_slots();
                                        }
                                        Ok(())
                                    }
                                    Err(err) => Err(err),
                                },
                                None => Err(anyhow!("driver has no backend installed")),
                            }
                        }
                    }
                    _ => Err(anyhow!("instance {id} is unknown or stale")),
                };
                let _ = response.send(result);
            }
            QueuedItem::CloseChannel { id, response } => {
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
                let _ = response.send(result);
            }
        }
    }

    fn dispatch_launch_batch(
        driver: &mut Option<DriverBackend>,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut VecDeque<QueuedItem>,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        policy: &mut quorum::WaitAllPolicy,
    ) -> bool {
        let mut batch = BatchAccumulator::new(limits, page_size);
        let mut grouping = LaunchGrouping::default();
        let mut deferred = VecDeque::new();
        let mut blocked_pipelines = HashSet::new();
        let mut rejected_stale = false;
        while let Some(QueuedItem::Launch(next)) = pending.front() {
            if next.completion.is_settled() {
                let Some(QueuedItem::Launch(dropped)) = pending.pop_front() else {
                    unreachable!();
                };
                policy.on_request_dropped(dropped.pipeline_id);
                rejected_stale = true;
                continue;
            }
            if next.completion.cancel_requested() {
                let Some(QueuedItem::Launch(cancelled)) = pending.pop_front() else {
                    unreachable!();
                };
                policy.on_request_dropped(cancelled.pipeline_id);
                cancelled
                    .completion
                    .reject_unsubmitted("logical fire cancelled before native launch");
                rejected_stale = true;
                continue;
            }
            if instances.get(&next.instance_id).is_none() {
                // A launch whose instance closed between enqueue validation
                // and dispatch must be rejected here, not left at the queue
                // front where it would head-of-line block the driver forever.
                let Some(QueuedItem::Launch(stale)) = pending.pop_front() else {
                    unreachable!();
                };
                policy.on_request_dropped(stale.pipeline_id);
                stale.completion.reject_unsubmitted(format!(
                    "instance {} is unknown or stale",
                    stale.instance_id
                ));
                rejected_stale = true;
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
                if grouping.instances.contains(&next.instance_id)
                    && let Some(pid) = next.pipeline_id
                {
                    blocked_pipelines.insert(pid);
                    deferred.push_back(pending.pop_front().expect("runahead launch front"));
                    continue;
                }
                break;
            }
            let QueuedItem::Launch(next) = pending.pop_front().expect("launch front") else {
                unreachable!();
            };
            let stop = grouping.push(&next, limits, page_size);
            batch.push(next);
            if stop {
                break;
            }
        }
        while let Some(item) = deferred.pop_back() {
            pending.push_front(item);
        }
        let mut requests = batch.take();
        if requests.is_empty() {
            return rejected_stale;
        }
        let batch_size = requests.len() as u64;
        // Token stats must be read before build: a prebuilt single-request
        // batch moves its plan into the submission.
        let total_tokens = requests
            .iter()
            .map(|req| req.request.token_ids.len())
            .sum::<usize>();
        let submission = batch::build_batch_request(&mut requests, page_size, stats);
        let mut candidate_epochs = Vec::with_capacity(requests.len());
        for request in &requests {
            let Some(instance) = instances.get(&request.instance_id) else {
                for request in &mut requests {
                    policy.on_request_dropped(request.pipeline_id);
                    let message = format!("instance {} is unknown or stale", request.instance_id);
                    request.completion.reject_unsubmitted(message.clone());
                }
                return true;
            };
            candidate_epochs.push(instance.next_target_epoch);
        }
        match driver.as_mut() {
            Some(driver) => match crate::probe_fire!(
                stats.fire.execute.driver_fire_us,
                driver.launch(&submission)
            ) {
                Ok(completion) => {
                    for (request, &epoch) in requests.iter().zip(candidate_epochs.iter()) {
                        request.completion.commit_target_epoch(epoch);
                    }
                    for (request, &epoch) in requests.iter().zip(candidate_epochs.iter()) {
                        if let Some(instance) = instances.get_mut(&request.instance_id) {
                            instance.in_flight += 1;
                            instance.next_target_epoch = epoch + 1;
                        }
                    }
                    in_flight_launches.push_back(PendingLaunchBatch {
                        completion,
                        requests,
                        started: Instant::now(),
                        batch_size,
                        total_tokens,
                    });
                }
                Err(err) => {
                    let message = format!("direct launch rejected: {err:#}");
                    for request in &mut requests {
                        policy.on_request_dropped(request.pipeline_id);
                        request.completion.reject_unsubmitted(message.clone());
                    }
                }
            },
            None => {
                for request in &mut requests {
                    policy.on_request_dropped(request.pipeline_id);
                    request
                        .completion
                        .reject_unsubmitted("driver has no backend installed");
                }
            }
        }
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
            let Some(result) = front.completion.check() else {
                break;
            };
            let retired = in_flight_launches.pop_front().expect("front batch exists");
            let participants = retired
                .requests
                .iter()
                .map(|request| request.pipeline_id)
                .collect::<Vec<_>>();
            policy.on_wave_retired(&participants);
            for request in &retired.requests {
                if let Some(instance) = instances.get_mut(&request.instance_id) {
                    instance.in_flight = instance.in_flight.saturating_sub(1);
                }
            }
            match result {
                Ok(()) => {
                    for request in &retired.requests {
                        request.completion.mark_native_retired();
                    }
                    let mut retries = Vec::new();
                    for mut request in retired.requests {
                        match request.completion.resolve_from_terminal() {
                            Ok(
                                WorkItemAttemptOutcome::Committed | WorkItemAttemptOutcome::Failed,
                            ) => {}
                            Ok(WorkItemAttemptOutcome::Retry) => {
                                request.retry_count += 1;
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
                                    policy.on_pipeline_request(request.pipeline_id, Instant::now());
                                    retries.push(request);
                                }
                            }
                            Err(err) => {
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

    fn retire_ready_control(in_flight_control: &mut Option<PendingControl>) -> bool {
        let operation = in_flight_control
            .as_ref()
            .map(|pending| pending.operation)
            .unwrap_or("control operation");
        let Some(result) = in_flight_control
            .as_ref()
            .and_then(|pending| pending.completion.check())
        else {
            return false;
        };
        if let Some(tracked) = in_flight_control
            .as_ref()
            .and_then(|pending| pending.tracked_completion.as_ref())
        {
            tracked.resolve(&result);
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
    /// This instance's compiled program identity — the fire trace
    /// (`PIE_SCHED_TRACE`) derives `distinct_programs` from these per
    /// dispatched batch (no per-identity hash/stats table: this is just the
    /// bind-time `BoundInstance::program_id` already tracked here).
    program_id: crate::driver::ProgramId,
}

impl TrackedInstance {
    fn from_bound(bound: &BoundInstance) -> Self {
        Self {
            pacing_wait_id: bound.pacing_wait_id,
            wait_slots: bound.wait_slots(),
            in_flight: 0,
            next_target_epoch: pie_waker::FIRST_COMPLETION_EPOCH,
            program_id: bound.program_id,
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
    use pie_driver_abi::{PieKvMoveCell, PiePoolRange};
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

    fn register_test_channels(
        driver_id: usize,
        channel_ids: [u64; 2],
    ) -> anyhow::Result<Vec<Arc<crate::driver::ChannelEndpoint>>> {
        [
            (channel_ids[0], HostRole::None, true),
            (channel_ids[1], HostRole::Reader, false),
        ]
        .into_iter()
        .map(|(channel_id, host_role, seeded)| {
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
        })
        .collect()
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
    /// (`max_forward_requests`) short-circuits any cold-hold/deadline
    /// delay once a batch saturates it (see `quorum::tests::
    /// structural_cap_fires_immediately_even_cold`), so every other test in
    /// this module runs at cap 1 and never observes the quorum hold. Tests
    /// that need to actually exercise the hold (coalescing/deadline/leave)
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
            },
            DriverBackend::Dummy(crate::driver::DummyDriver::new(options)),
        );
        let scheduler = BatchScheduler::new(driver_id, driver_id, 16, limits, 1);
        let program_id = crate::scheduler::register_program(driver_id, dummy_program())?;
        let endpoints = register_test_channels(driver_id, [7, 8])?;
        let bound = crate::scheduler::bind_instance(
            driver_id,
            program_id,
            41,
            vec![7, 8],
            vec![ChannelValue {
                channel: 7,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )?;
        Ok((driver_id, scheduler, bound, endpoints))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn typed_copy_paths_dispatch_to_distinct_driver_methods() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;

        timeout(
            Duration::from_secs(5),
            crate::scheduler::copy_kv_cells(
                driver_id,
                vec![PieKvMoveCell {
                    dst_page_id: 1,
                    dst_token_offset: 0,
                    src_page_id: 2,
                    src_token_offset: 0,
                }],
            )?,
        )
        .await??;
        timeout(
            Duration::from_secs(5),
            crate::scheduler::copy_rs_d2d(driver_id, &[3], &[4])?,
        )
        .await??;
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
        )?;
        let launch = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Vec::new(),
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

        assert!(matches!(
            pie_waker::WakerTable::global().publish(pacing_wait_id, 1),
            pie_waker::WakeOutcome::Stale
        ));
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn duplicate_bind_preserves_original_instance() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;

        let error = crate::scheduler::bind_instance(
            driver_id,
            bound.program_id,
            bound.instance_id,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )
        .expect_err("duplicate requested instance id must be rejected");
        assert!(error.to_string().contains("already bound"));

        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
                .any(|entry| entry == "launch-shape tokens=2 programs=1")
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
            0,
            first.clone(),
        )?;
        let second = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            second_plan,
            driver_id,
            bound.instance_id,
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_options(DummyDriverOptions {
                retry_launches_remaining: max_fire_retries(),
                operation_log: Some(operation_log.clone()),
                ..DummyDriverOptions::default()
            })
            .await?;
        let preparation: crate::scheduler::LaunchPreparation = Box::new(|_| {
            Ok(crate::scheduler::PreparedLaunch {
                page_refs: Vec::new(),
                last_page_len: 0,
                kv_translation_version: 0,
                copy_src: Vec::new(),
                copy_dst: Vec::new(),
            })
        });
        let classifier: crate::scheduler::RetryClassifier =
            Box::new(|| Some("writer endpoint closed".to_string()));
        let completion = bound.reserve_completion();
        crate::scheduler::submit_async_deferred(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Some(ProcessId::new_v4()),
            None,
            completion.clone(),
            preparation,
            Some(classifier),
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

    #[tokio::test(flavor = "current_thread")]
    async fn dispatch_preparation_retries_before_the_single_driver_attempt() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let attempts_in_prepare = attempts.clone();
        let preparation: crate::scheduler::LaunchPreparation = Box::new(move |request| {
            let attempt = attempts_in_prepare.fetch_add(1, Ordering::AcqRel);
            if attempt == 0 {
                return Err(crate::scheduler::LaunchPreparationError::Retry(
                    "predecessor translation not ready".to_string(),
                ));
            }
            request.kv_translation = Vec::new();
            Ok(crate::scheduler::PreparedLaunch {
                page_refs: Vec::new(),
                last_page_len: 0,
                kv_translation_version: 0,
                copy_src: Vec::new(),
                copy_dst: Vec::new(),
            })
        });
        let completion = bound.reserve_completion();
        crate::scheduler::submit_async_deferred(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Some(ProcessId::new_v4()),
            None,
            completion.clone(),
            preparation,
            None,
        )?;
        timeout(Duration::from_secs(5), completion).await??;
        assert_eq!(attempts.load(Ordering::Acquire), 2);
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
    async fn blocked_preparation_runs_only_after_an_event_nudge() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler(operation_log.clone()).await?;
        let ready = Arc::new(AtomicBool::new(false));
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prepare_ready = ready.clone();
        let prepare_attempts = attempts.clone();
        let preparation: crate::scheduler::LaunchPreparation = Box::new(move |_| {
            prepare_attempts.fetch_add(1, Ordering::AcqRel);
            if !prepare_ready.load(Ordering::Acquire) {
                return Err(crate::scheduler::LaunchPreparationError::Blocked(
                    "waiting for contention grant".to_string(),
                ));
            }
            Ok(crate::scheduler::PreparedLaunch {
                page_refs: Vec::new(),
                last_page_len: 0,
                kv_translation_version: 0,
                copy_src: Vec::new(),
                copy_dst: Vec::new(),
            })
        });
        let completion = bound.reserve_completion();
        crate::scheduler::submit_async_deferred(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Some(ProcessId::new_v4()),
            None,
            completion.clone(),
            preparation,
            None,
        )?;
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert_eq!(attempts.load(Ordering::Acquire), 1);
        assert!(!completion.is_settled());

        ready.store(true, Ordering::Release);
        crate::scheduler::nudge(driver_id);
        timeout(Duration::from_secs(5), completion).await??;
        assert_eq!(attempts.load(Ordering::Acquire), 2);
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
    async fn pipeline_freeze_blocks_preparation_until_resume() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, scheduler, bound, _endpoints) = setup_scheduler(operation_log).await?;
        let pid = ProcessId::new_v4();
        scheduler.handle.freeze_pipeline(pid)?;
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prepare_attempts = attempts.clone();
        let preparation: crate::scheduler::LaunchPreparation = Box::new(move |_| {
            prepare_attempts.fetch_add(1, Ordering::AcqRel);
            Ok(crate::scheduler::PreparedLaunch {
                page_refs: Vec::new(),
                last_page_len: 0,
                kv_translation_version: 0,
                copy_src: Vec::new(),
                copy_dst: Vec::new(),
            })
        });
        let completion = bound.reserve_completion();
        crate::scheduler::submit_async_deferred(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Some(pid),
            None,
            completion.clone(),
            preparation,
            None,
        )?;
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert_eq!(attempts.load(Ordering::Acquire), 0);
        assert!(!completion.is_settled());

        scheduler.handle.resume_pipeline(pid)?;
        timeout(Duration::from_secs(5), completion).await??;
        assert_eq!(attempts.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn termination_rejects_frozen_preparation() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, scheduler, bound, _endpoints) = setup_scheduler(operation_log).await?;
        let pid = ProcessId::new_v4();
        scheduler.handle.freeze_pipeline(pid)?;
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prepare_attempts = attempts.clone();
        let preparation: crate::scheduler::LaunchPreparation = Box::new(move |_| {
            prepare_attempts.fetch_add(1, Ordering::AcqRel);
            Err(crate::scheduler::LaunchPreparationError::Failed(
                "must not run".to_string(),
            ))
        });
        let completion = bound.reserve_completion();
        crate::scheduler::submit_async_deferred(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Some(pid),
            None,
            completion.clone(),
            preparation,
            None,
        )?;
        notify_pipeline_leave(pid, LeaveKind::Terminate);
        let error = timeout(Duration::from_secs(5), completion)
            .await?
            .expect_err("termination must reject blocked preparation");
        assert!(error.to_string().contains("terminated"));
        assert_eq!(attempts.load(Ordering::Acquire), 0);
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
            Vec::new(),
            0,
            Some(pid),
            None,
            false,
            None,
            None,
            None,
            None,
        );
        let mut pending = VecDeque::from([QueuedItem::Launch(request)]);
        let mut blocked = VecDeque::new();
        let mut policy = quorum::WaitAllPolicy::new(1, None);
        completion.request_cancel();
        BatchScheduler::reject_pipeline_queued(
            &mut pending,
            &mut blocked,
            &mut policy,
            pid,
            Some(&completion),
        );
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
            Vec::new(),
            0,
            None,
            None,
            false,
            None,
            Some(state_copy),
            None,
            None,
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
    fn tracked_multi_row_prebuilt_request_remains_solo() {
        let mut launch = dummy_launch();
        launch.qo_indptr = vec![0, 0, 0];
        let request = PendingRequest::direct(
            launch,
            1,
            WorkItemCompletion::deferred_with_guard(None),
            Vec::new(),
            0,
            Some(ProcessId::new_v4()),
            None,
            true,
            None,
            None,
            None,
            None,
        );
        assert!(request.preserves_inner_rows());
        assert!(request.requires_solo_submission());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn retrying_preparation_preserves_lane_order_without_global_blocking()
    -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound, _endpoints) = setup_scheduler(operation_log).await?;
        let pid = ProcessId::new_v4();
        let allow_first = Arc::new(AtomicBool::new(false));
        let order = Arc::new(Mutex::new(Vec::new()));

        let first_allow = allow_first.clone();
        let first_order = order.clone();
        let first_preparation: crate::scheduler::LaunchPreparation = Box::new(move |_| {
            if !first_allow.load(Ordering::Acquire) {
                return Err(crate::scheduler::LaunchPreparationError::Retry(
                    "first lane member is waiting".to_string(),
                ));
            }
            first_order.lock().unwrap().push("first");
            Ok(crate::scheduler::PreparedLaunch {
                page_refs: Vec::new(),
                last_page_len: 0,
                kv_translation_version: 0,
                copy_src: Vec::new(),
                copy_dst: Vec::new(),
            })
        });
        let second_allow = allow_first.clone();
        let second_order = order.clone();
        let second_preparation: crate::scheduler::LaunchPreparation = Box::new(move |_| {
            assert!(
                second_allow.load(Ordering::Acquire),
                "a later same-lane preparation overtook its predecessor"
            );
            second_order.lock().unwrap().push("second");
            Ok(crate::scheduler::PreparedLaunch {
                page_refs: Vec::new(),
                last_page_len: 0,
                kv_translation_version: 0,
                copy_src: Vec::new(),
                copy_dst: Vec::new(),
            })
        });
        let unblock_order = order.clone();
        let unblock_preparation: crate::scheduler::LaunchPreparation = Box::new(move |_| {
            unblock_order.lock().unwrap().push("unrelated");
            allow_first.store(true, Ordering::Release);
            Err(crate::scheduler::LaunchPreparationError::Failed(
                "test-only unrelated preparation stop".to_string(),
            ))
        });

        let first = bound.reserve_completion();
        crate::scheduler::submit_async_deferred(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Some(pid),
            Some(7),
            first.clone(),
            first_preparation,
            None,
        )?;
        let second = bound.reserve_completion();
        crate::scheduler::submit_async_deferred(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Some(pid),
            Some(7),
            second.clone(),
            second_preparation,
            None,
        )?;
        let unrelated = bound.reserve_completion();
        crate::scheduler::submit_async_deferred(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Some(pid),
            Some(8),
            unrelated.clone(),
            unblock_preparation,
            None,
        )?;

        timeout(Duration::from_secs(5), async {
            first.await?;
            second.await?;
            let _ = unrelated.await.expect_err("unrelated preparation stops");
            Ok::<(), anyhow::Error>(())
        })
        .await??;
        assert_eq!(
            order.lock().unwrap().as_slice(),
            ["unrelated", "first", "second"]
        );
        Ok(())
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
            Vec::new(),
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
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18])?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            bound_a.program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )?;

        let first = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            0,
            first.clone(),
        )?;

        let second = bound_b.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            Vec::new(),
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
            Vec::new(),
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
                Vec::new(),
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
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18])?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            bound_a.program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )?;

        let first = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            0,
            first.clone(),
        )?;

        let resize_join = std::thread::spawn(move || {
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
                Vec::new(),
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
        let resize = resize_join.join().unwrap()?;
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
    async fn close_waits_for_accepted_launch_then_succeeds() -> anyhow::Result<()> {
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
            Vec::new(),
            0,
            launch.clone(),
        )?;

        let close_bound = std::thread::spawn({
            let bound = bound;
            move || crate::scheduler::close_instance(&bound)
        });

        std::thread::sleep(Duration::from_millis(10));
        assert!(
            !close_bound.is_finished(),
            "close should block until accepted launch retires"
        );
        timeout(Duration::from_secs(5), launch).await??;
        close_bound.join().unwrap()?;

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

    #[test]
    fn stale_instance_handles_are_rejected() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let (_driver_id, _scheduler, bound, _endpoints) =
                setup_scheduler(operation_log).await?;
            crate::scheduler::close_instance(&bound)?;
            let err = crate::scheduler::close_instance(&bound).unwrap_err();
            assert!(err.to_string().contains("stale"));
            Ok::<_, anyhow::Error>(())
        })?;
        Ok(())
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
        let program_id = crate::scheduler::register_program(driver_id, dummy_program())?;
        let _secondary_endpoints = register_test_channels(driver_id, [17, 18])?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            program_id,
            42,
            vec![17, 18],
            vec![ChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )?;

        let resize = crate::scheduler::resize_pool(driver_id, 9, 16, Vec::new(), Vec::new())?;
        let a = bound_a.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            0,
            a,
        )?;
        let b = bound_b.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
        )?;
        let importer_program = crate::scheduler::register_program(
            driver_id,
            ProgramRegistration {
                program_hash: pie_ptir::container_hash(&importer_bytes),
                canonical_bytes: importer_bytes,
                sidecar_bytes: Vec::new(),
            },
        )?;
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
        )?;
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
        )?;
        let _ = shared;
        let exporter =
            crate::scheduler::bind_instance(driver_id, exporter_program, 61, vec![91], Vec::new())?;
        let importer = crate::scheduler::bind_instance(
            driver_id,
            importer_program,
            62,
            vec![91, 92],
            Vec::new(),
        )?;

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
            Vec::new(),
            0,
            export_fire.clone(),
        )?;
        let import_fire = importer.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            importer.instance_id,
            Vec::new(),
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
                    Vec::new(),
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
    fn bind_second_instance(
        driver_id: usize,
        bound_a: &crate::driver::BoundInstance,
        channel_ids: [u64; 2],
        requested_instance_id: u64,
    ) -> anyhow::Result<(
        crate::driver::BoundInstance,
        Vec<Arc<crate::driver::ChannelEndpoint>>,
    )> {
        let endpoints = register_test_channels(driver_id, channel_ids)?;
        let bound_b = crate::scheduler::bind_instance(
            driver_id,
            bound_a.program_id,
            requested_instance_id,
            channel_ids.to_vec(),
            vec![ChannelValue {
                channel: channel_ids[0],
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )?;
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
            bind_second_instance(driver_id, &bound_a, [27, 28], 52)?;

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
            Vec::new(),
            0,
            Some(pid_a),
            first.clone(),
        )?;
        let second = bound_b.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            Vec::new(),
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
                    .any(|entry| entry == "launch-shape tokens=2 programs=2");
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
    async fn a_missing_members_wave_deadline_fires_and_counts_a_miss() -> anyhow::Result<()> {
        let (driver_id, _scheduler, bound_a, _endpoints) =
            setup_scheduler_with_limits(DummyDriverOptions::default(), coalescing_limits()).await?;
        let (bound_b, _secondary_endpoints) =
            bind_second_instance(driver_id, &bound_a, [27, 28], 53)?;

        let pid_a = ProcessId::new_v4();
        let pid_b = ProcessId::new_v4();

        // Wave 1: both pipelines are seen — establishes them in the
        // wait-set (dense-fires quickly via the cold-hold gather).
        let first_a = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            0,
            Some(pid_a),
            first_a.clone(),
        )?;
        let first_b = bound_b.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            Vec::new(),
            0,
            Some(pid_b),
            first_b.clone(),
        )?;
        timeout(Duration::from_secs(5), first_a).await??;
        timeout(Duration::from_secs(5), first_b).await??;

        // Wave 2: only `a` resubmits (its decode loop's next token); `b`
        // never comes back. The quorum holds `a` for `b` up to the default
        // 500us
        // wave deadline, then fires solo — a straggler deadline-fire, not
        // an immediate dense fire.
        let started = Instant::now();
        let second_a = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            0,
            Some(pid_a),
            second_a.clone(),
        )?;
        timeout(Duration::from_secs(5), second_a).await??;
        assert!(
            started.elapsed() >= Duration::from_micros(300),
            "a solo wave missing `b` should hold the wave \
             deadline before firing, took {:?}",
            started.elapsed()
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
            bind_second_instance(driver_id, &bound_a, [27, 28], 54)?;

        let pid_a = ProcessId::new_v4();
        let pid_b = ProcessId::new_v4();

        // Wave 1: both pipelines seen, both in the wait-set.
        let first_a = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            0,
            Some(pid_a),
            first_a.clone(),
        )?;
        let first_b = bound_b.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            Vec::new(),
            0,
            Some(pid_b),
            first_b.clone(),
        )?;
        timeout(Duration::from_secs(5), first_a).await??;
        timeout(Duration::from_secs(5), first_b).await??;

        // Wave 2: only `a` resubmits; `b` instead LEAVES the fleet (as the
        // process facade's terminate path would call on exit) before the
        // wave deadline — the quorum should drop it from the wait-set and
        // fire `a`'s wave immediately, not after the ~10ms deadline.
        let started = Instant::now();
        let second_a = bound_a.reserve_completion();
        crate::scheduler::submit_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            0,
            Some(pid_a),
            second_a.clone(),
        )?;
        notify_pipeline_leave(pid_b, LeaveKind::Terminate);
        timeout(Duration::from_secs(5), second_a).await??;
        assert!(
            started.elapsed() < Duration::from_millis(8),
            "leave should unblock the hold well before the ~10ms wave \
             deadline, took {:?}",
            started.elapsed()
        );

        crate::scheduler::close_instance(&bound_a)?;
        crate::scheduler::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn untracked_prebuilt_fire_never_blocks_on_the_quorum() -> anyhow::Result<()> {
        let (driver_id, _scheduler, bound, _endpoints) =
            setup_scheduler_with_limits(DummyDriverOptions::default(), coalescing_limits()).await?;

        // `submit_prebuilt_async` always carries `pipeline_id: None` — it
        // never joins the wait-set, so it must fire promptly even though
        // nothing else is active to gather with it (bootstrap cold-hold at
        // most, never the ~10ms straggler deadline).
        let started = Instant::now();
        let completion = bound.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Vec::new(),
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
}
