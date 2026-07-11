//! Per-driver direct batch scheduler.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::driver::{
    BoundInstance, ChannelRegistrationPlan, Completion, DriverId, InstanceBindingPlan,
    InstanceCompletion, LocalDriver, NativeDriver, PoolResizePlan, ProgramRegistration,
    RegisteredChannel, SchedulerLimits, StateCopyPlan,
};
use crate::scheduler::ProcessId;
use crate::store::kv::project::PhysicalPageId;
use anyhow::{Result, anyhow};

use super::batch::{self, BatchAccumulator};
use super::quorum;
use super::stats::{self, SchedulerStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FireClause {
    Quorum,
    SubmitAhead,
    IdleEscape,
    ColdHold,
    Hold,
}

impl FireClause {
    #[inline]
    pub fn fires(self) -> bool {
        !matches!(self, FireClause::Hold)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaveKind {
    Terminate,
    #[allow(dead_code)] // Suspend is now handled by store::reclaim's own local twin.
    Suspend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LifecycleEvent {
    #[allow(dead_code)]
    pid: ProcessId,
    #[allow(dead_code)]
    kind: Option<LeaveKind>,
}

pub(crate) fn notify_pipeline_leave(_pid: ProcessId, _kind: LeaveKind) {}
#[allow(dead_code)] // store::reclaim now owns the join hook for its own callers.
pub(crate) fn notify_pipeline_join(_pid: ProcessId) {}

/// Wake-class counter (plan §16.2): completions that the 250 ms hang backstop
/// discovered already settled — a lost nudge. Steady state stays at zero; any
/// increment is a wake-path regression worth a warning.
pub(crate) static BACKSTOP_RETIREMENTS: AtomicU64 = AtomicU64::new(0);

/// Total backstop-path retirements since process start (test observability).
pub(crate) fn backstop_retirements() -> u64 {
    BACKSTOP_RETIREMENTS.load(Ordering::Relaxed)
}

pub(crate) fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or_default()
}

pub(crate) struct PendingRequest {
    pub(crate) request: crate::driver::LaunchPlan,
    pub(crate) instance_id: u64,
    pub(crate) completion: InstanceCompletion,
    pub(crate) physical_page_ids: Vec<PhysicalPageId>,
    pub(crate) last_page_len: u32,
    pub(crate) program_identity_hashes: Vec<u64>,
    #[allow(dead_code)]
    pub(crate) pipeline_id: Option<ProcessId>,
    pub(crate) submitted_at_us: u64,
    pub(crate) prebuilt: bool,
}

impl PendingRequest {
    fn direct(
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: InstanceCompletion,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
        pipeline_id: Option<ProcessId>,
        submitted_at_us: u64,
        prebuilt: bool,
    ) -> Self {
        Self {
            request,
            instance_id,
            completion,
            physical_page_ids,
            last_page_len,
            program_identity_hashes,
            pipeline_id,
            submitted_at_us,
            prebuilt,
        }
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
        response: crossbeam::channel::Sender<Result<Completion>>,
    },
    // Only reached via `SchedulerHandle::copy_state`/`resize_pool`, which
    // the mock-driver fire path doesn't call yet (`scheduler::resize_pool`
    // is exercised by this module's unit tests) — see `scheduler::dispatch`'s
    // module doc for the full driver-ABI-completeness rationale.
    #[allow(dead_code)]
    CopyState {
        plan: StateCopyPlan,
        response: crossbeam::channel::Sender<Result<Completion>>,
    },
    #[allow(dead_code)]
    ResizePool {
        plan: PoolResizePlan,
        response: crossbeam::channel::Sender<Result<Completion>>,
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
    /// Event-driven retirement wake: sent by [`NudgeWaker`] when an in-flight
    /// driver completion publishes. Carries no work; it only unblocks the
    /// scheduler's wait so the retire pass runs immediately.
    Nudge,
    Stop,
}

/// Wakes the scheduler thread through its own queue when a registered driver
/// completion publishes, so batch/control retirement is event-driven instead
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
fn arm_completion_nudge(completion: &Completion, waker: &std::task::Waker) -> bool {
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

enum QueuedItem {
    Launch(PendingRequest),
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
        response: crossbeam::channel::Sender<Result<Completion>>,
    },
    CopyState {
        plan: StateCopyPlan,
        response: crossbeam::channel::Sender<Result<Completion>>,
    },
    ResizePool {
        plan: PoolResizePlan,
        response: crossbeam::channel::Sender<Result<Completion>>,
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

struct PendingLaunchBatch {
    completion: Completion,
    requests: Vec<PendingRequest>,
    started: Instant,
    batch_size: u64,
    total_tokens: usize,
}

struct PendingControl {
    completion: Completion,
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

    pub fn submit_with_identity(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: InstanceCompletion,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
        pipeline_id: Option<ProcessId>,
        submitted_at_us: u64,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                physical_page_ids,
                last_page_len,
                program_identity_hashes,
                pipeline_id,
                submitted_at_us,
                false,
            ),
        })
    }

    pub fn submit_prebuilt(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        completion: InstanceCompletion,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                physical_page_ids,
                last_page_len,
                program_identity_hashes,
                None,
                0,
                true,
            ),
        })
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

    pub fn copy_kv(&self, plan: crate::driver::KvCopyPlan) -> Result<Completion> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::CopyKv { plan, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    // Only called from `scheduler::dispatch::copy_rs_d2d`/`resize_pool`
    // (not yet issued by the mock-driver fire path) and this module's own
    // unit tests — see `scheduler::dispatch`'s module doc.
    #[allow(dead_code)]
    pub fn copy_state(&self, plan: StateCopyPlan) -> Result<Completion> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::CopyState { plan, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    #[allow(dead_code)]
    pub fn resize_pool(&self, plan: PoolResizePlan) -> Result<Completion> {
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

    pub(crate) fn handle(&self) -> SchedulerHandle {
        self.handle.clone()
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
        let mut driver = crate::driver::take_native_driver(driver_id).ok();
        let mut instances = HashMap::new();
        let mut channels = HashSet::new();
        let mut pending = VecDeque::new();
        let mut in_flight_launches = VecDeque::new();
        let mut in_flight_control = None;
        let mut stopping = false;

        loop {
            let mut progress = false;
            progress |=
                Self::retire_ready_launches(&mut in_flight_launches, &mut instances, &stats);
            progress |= Self::retire_ready_control(&mut in_flight_control);
            progress |= Self::dispatch_ready_items(
                &mut driver,
                &mut instances,
                &mut channels,
                &mut pending,
                &mut in_flight_launches,
                &mut in_flight_control,
                page_size,
                limits,
                &stats,
            );

            if stopping
                && pending.is_empty()
                && in_flight_launches.is_empty()
                && in_flight_control.is_none()
            {
                break;
            }

            while let Ok(item) = rx.try_recv() {
                progress = true;
                Self::enqueue_item(
                    &mut pending,
                    &instances,
                    limits,
                    page_size,
                    &mut stopping,
                    item,
                );
            }

            if progress {
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
                    armed &= arm_completion_nudge(&front.completion, &nudge_waker);
                }
                if let Some(control) = in_flight_control.as_ref() {
                    armed &= arm_completion_nudge(&control.completion, &nudge_waker);
                }
                if !armed {
                    // Something already settled; retire it on the next pass.
                    continue;
                }
                match rx.recv_timeout(Duration::from_millis(250)) {
                    Ok(item) => Some(item),
                    Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                        // A settled completion discovered by the backstop
                        // means a wake was lost somewhere — the steady-state
                        // count stays zero (plan §16.2). Shutdown races are
                        // excluded: teardown may legitimately cross a tick.
                        let missed = in_flight_launches
                            .front()
                            .is_some_and(|front| front.completion.is_settled())
                            || in_flight_control
                                .as_ref()
                                .is_some_and(|control| control.completion.is_settled());
                        if missed && !stopping {
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
                    &instances,
                    limits,
                    page_size,
                    &mut stopping,
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
        instances: &HashMap<u64, TrackedInstance>,
        limits: SchedulerLimits,
        page_size: u32,
        stopping: &mut bool,
        item: SchedulerItem,
    ) {
        match item {
            SchedulerItem::Stop => *stopping = true,
            // A nudge only unblocks the wait; the retire pass at the top of
            // the loop does the work.
            SchedulerItem::Nudge => {}
            SchedulerItem::Launch { pending: launch } => {
                let validation = BatchAccumulator::new(limits, page_size);
                let rejection = if !instances.contains_key(&launch.instance_id) {
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
                    launch.completion.reject(message);
                } else {
                    pending.push_back(QueuedItem::Launch(launch));
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

    fn dispatch_ready_items(
        driver: &mut Option<NativeDriver>,
        instances: &mut HashMap<u64, TrackedInstance>,
        channels: &mut HashSet<u64>,
        pending: &mut VecDeque<QueuedItem>,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &mut Option<PendingControl>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
    ) -> bool {
        let mut progress = false;
        loop {
            if in_flight_control.is_some() {
                break;
            }
            let Some(item) = pending.front() else {
                break;
            };
            match item {
                QueuedItem::Launch(_) => {
                    if in_flight_launches.len() >= quorum::max_in_flight() {
                        break;
                    }
                    let dispatched = Self::dispatch_launch_batch(
                        driver,
                        instances,
                        pending,
                        in_flight_launches,
                        page_size,
                        limits,
                        stats,
                    );
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
        progress
    }

    fn dispatch_ordered_item(
        driver: &mut Option<NativeDriver>,
        instances: &mut HashMap<u64, TrackedInstance>,
        channels: &mut HashSet<u64>,
        in_flight_control: &mut Option<PendingControl>,
        item: QueuedItem,
    ) {
        match item {
            QueuedItem::Launch(_) => unreachable!(),
            QueuedItem::RegisterProgram { plan, response } => {
                let _ = response.send(match driver.as_mut() {
                    Some(driver) => driver.register_program(&plan),
                    None => Err(anyhow!("driver has no native backend installed")),
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
                        None => Err(anyhow!("driver has no native backend installed")),
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
                        None => Err(anyhow!("driver has no native backend installed")),
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
                            });
                            Ok(completion)
                        }
                        Err(err) => Err(err),
                    },
                    None => Err(anyhow!("driver has no native backend installed")),
                });
            }
            QueuedItem::CopyState { plan, response } => {
                let _ = response.send(match driver.as_mut() {
                    Some(driver) => match driver.copy_state(&plan) {
                        Ok(completion) => {
                            *in_flight_control = Some(PendingControl {
                                completion: completion.clone(),
                            });
                            Ok(completion)
                        }
                        Err(err) => Err(err),
                    },
                    None => Err(anyhow!("driver has no native backend installed")),
                });
            }
            QueuedItem::ResizePool { plan, response } => {
                let _ = response.send(match driver.as_mut() {
                    Some(driver) => match driver.resize_pool(&plan) {
                        Ok(completion) => {
                            *in_flight_control = Some(PendingControl {
                                completion: completion.clone(),
                            });
                            Ok(completion)
                        }
                        Err(err) => Err(err),
                    },
                    None => Err(anyhow!("driver has no native backend installed")),
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
                                None => Err(anyhow!("driver has no native backend installed")),
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
                        None => Err(anyhow!("driver has no native backend installed")),
                    }
                };
                let _ = response.send(result);
            }
        }
    }

    fn dispatch_launch_batch(
        driver: &mut Option<NativeDriver>,
        instances: &mut HashMap<u64, TrackedInstance>,
        pending: &mut VecDeque<QueuedItem>,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
    ) -> bool {
        let mut batch = BatchAccumulator::new(limits, page_size);
        let mut batch_instances = std::collections::HashSet::new();
        let mut batch_has_prebuilt = false;
        let mut batch_has_user_mask = false;
        let mut rejected_stale = false;
        while let Some(QueuedItem::Launch(next)) = pending.front() {
            if instances.get(&next.instance_id).is_none() {
                // A launch whose instance closed between enqueue validation
                // and dispatch must be rejected here, not left at the queue
                // front where it would head-of-line block the driver forever.
                let Some(QueuedItem::Launch(stale)) = pending.pop_front() else {
                    unreachable!();
                };
                stale.completion.reject(format!(
                    "instance {} is unknown or stale",
                    stale.instance_id
                ));
                rejected_stale = true;
                continue;
            }
            if !batch_instances.insert(next.instance_id) {
                break;
            }
            // Mask co-batch policy (v1 of the composed multi-program batch):
            // the driver merges channel-resolved and wire geometry but not
            // attention MASKS across programs. A mask-carrying fire (dense
            // device mask or guest BRLE mask) runs SOLO.
            let next_prebuilt = next.prebuilt;
            let next_masked = next.request.has_user_mask;
            if !batch.is_empty() && (next_masked || batch_has_user_mask) {
                break;
            }
            if batch.would_exceed(next) {
                break;
            }
            let QueuedItem::Launch(next) = pending.pop_front().expect("launch front") else {
                unreachable!();
            };
            batch.push(next);
            batch_has_prebuilt |= next_prebuilt;
            batch_has_user_mask |= next_masked;
            let _ = batch_has_prebuilt;
            if next_masked {
                break; // solo mask-carrying fire
            }
            if batch.is_full() {
                break;
            }
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
                    let message = format!("instance {} is unknown or stale", request.instance_id);
                    request.completion.reject(message.clone());
                }
                return true;
            };
            candidate_epochs.push(instance.next_target_epoch);
        }
        match driver.as_mut() {
            Some(driver) => match driver.launch(&submission) {
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
                        request.completion.reject(message.clone());
                    }
                }
            },
            None => {
                for request in &mut requests {
                    request
                        .completion
                        .reject("driver has no native backend installed");
                }
            }
        }
        true
    }

    fn retire_ready_launches(
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        instances: &mut HashMap<u64, TrackedInstance>,
        stats: &Arc<SchedulerStats>,
    ) -> bool {
        let mut progress = false;
        while let Some(front) = in_flight_launches.front() {
            let Some(result) = front.completion.check() else {
                break;
            };
            let retired = in_flight_launches.pop_front().expect("front batch exists");
            for request in &retired.requests {
                if let Some(instance) = instances.get_mut(&request.instance_id) {
                    instance.in_flight = instance.in_flight.saturating_sub(1);
                }
            }
            match result {
                Ok(()) => {
                    for request in &retired.requests {
                        if let Err(err) = request.completion.resolve_from_terminal() {
                            tracing::warn!(
                                instance_id = request.instance_id,
                                ?err,
                                "direct launch terminal settlement failed"
                            );
                        }
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
        let Some(result) = in_flight_control
            .as_ref()
            .and_then(|pending| pending.completion.check())
        else {
            return false;
        };
        if let Err(err) = result {
            tracing::warn!(?err, "direct control completion closed before callback");
        }
        *in_flight_control = None;
        true
    }

    fn shutdown_instances(
        driver: &mut Option<NativeDriver>,
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

    fn shutdown_channels(driver: &mut Option<NativeDriver>, channels: &mut HashSet<u64>) {
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
    wait_slots: Arc<crate::driver::frame::BoundWaitSlots>,
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
        let driver_id = driver::register_native_driver(
            DriverSpec {
                num_kv_pages: 16,
                limits: SchedulerLimits {
                    max_forward_requests: 1,
                    max_forward_tokens: 64,
                    max_page_refs: 64,
                },
            },
            NativeDriver::Dummy(crate::driver::DummyLocalDriver::new(options)),
        );
        let scheduler = BatchScheduler::new(
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
            first.clone(),
        )?;

        let second = bound_b.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            Vec::new(),
            0,
            Vec::new(),
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
            Vec::new(),
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
                Vec::new(),
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
            Vec::new(),
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
                Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
            a,
        )?;
        let b = bound_b.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            Vec::new(),
            0,
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
            completion.clone(),
        )?;
        let woke = timeout(Duration::from_secs(5), waiter).await??;
        assert!(
            matches!(
                woke,
                Err(crate::driver::frame::ChannelWaitError::Poisoned(_))
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
        let driver_id = driver::register_native_driver(
            DriverSpec {
                num_kv_pages: 16,
                limits: SchedulerLimits {
                    max_forward_requests: 1,
                    max_forward_tokens: 64,
                    max_page_refs: 64,
                },
            },
            NativeDriver::Dummy(crate::driver::DummyLocalDriver::new(
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
            Vec::new(),
            export_fire.clone(),
        )?;
        let import_fire = importer.reserve_completion();
        crate::scheduler::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            importer.instance_id,
            Vec::new(),
            0,
            Vec::new(),
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
                    Vec::new(),
                    completion,
                )?;
            }
            drop(scheduler);
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(())
    }
}
