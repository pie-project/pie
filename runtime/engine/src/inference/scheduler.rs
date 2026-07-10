//! Per-driver direct batch scheduler.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::arena::PhysicalPageId;
use crate::driver::{
    BoundInstance, Completion, DriverId, InstanceBindingPlan, InstanceCompletion, LocalDriver,
    NativeDriver, PoolResizePlan, ProgramRegistration, SchedulerLimits, StateCopyPlan,
};
use crate::process::ProcessId;
use crate::ptir::PtirChannelValue;
use anyhow::{Result, anyhow};

use super::batch::{self, BatchAccumulator};
use super::policy;
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
pub(crate) fn notify_pipeline_join(_pid: ProcessId) {}

pub(crate) fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or_default()
}

pub(crate) struct PendingRequest {
    pub(crate) request: crate::driver::LaunchPlan,
    pub(crate) instance_id: u64,
    pub(crate) host_puts: Vec<PtirChannelValue>,
    pub(crate) completion: InstanceCompletion,
    pub(crate) physical_page_ids: Vec<PhysicalPageId>,
    pub(crate) last_page_len: u32,
    pub(crate) program_identity_hashes: Vec<u64>,
    #[allow(dead_code)]
    pub(crate) pipeline_id: Option<ProcessId>,
    pub(crate) submitted_at_us: u64,
    pub(crate) prebuilt: bool,
    acceptance: Option<crossbeam::channel::Sender<Result<()>>>,
}

impl PendingRequest {
    fn direct(
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        host_puts: Vec<PtirChannelValue>,
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
            host_puts,
            completion,
            physical_page_ids,
            last_page_len,
            program_identity_hashes,
            pipeline_id,
            submitted_at_us,
            prebuilt,
            acceptance: None,
        }
    }
}

enum SchedulerItem {
    Launch {
        pending: PendingRequest,
        response: crossbeam::channel::Sender<Result<()>>,
    },
    RegisterProgram {
        plan: ProgramRegistration,
        response: crossbeam::channel::Sender<Result<u64>>,
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
    Stop,
}

enum QueuedItem {
    Launch(PendingRequest),
    RegisterProgram {
        plan: ProgramRegistration,
        response: crossbeam::channel::Sender<Result<u64>>,
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

    pub fn submit_with_identity(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        host_puts: Vec<PtirChannelValue>,
        completion: InstanceCompletion,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
        pipeline_id: Option<ProcessId>,
        submitted_at_us: u64,
    ) -> Result<()> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                host_puts,
                completion,
                physical_page_ids,
                last_page_len,
                program_identity_hashes,
                pipeline_id,
                submitted_at_us,
                false,
            ),
            response: tx,
        })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    pub fn submit_prebuilt(
        &self,
        request: crate::driver::LaunchPlan,
        instance_id: u64,
        host_puts: Vec<PtirChannelValue>,
        completion: InstanceCompletion,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        program_identity_hashes: Vec<u64>,
    ) -> Result<()> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                host_puts,
                completion,
                physical_page_ids,
                last_page_len,
                program_identity_hashes,
                None,
                0,
                true,
            ),
            response: tx,
        })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

    pub fn register_program(&self, plan: ProgramRegistration) -> Result<u64> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::RegisterProgram { plan, response: tx })?;
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

    pub fn copy_state(&self, plan: StateCopyPlan) -> Result<Completion> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.send(SchedulerItem::CopyState { plan, response: tx })?;
        rx.recv().map_err(|_| anyhow!("scheduler channel closed"))?
    }

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
            }),
        };
        let _ = crate::driver::registry::install_scheduler_handle(driver_id, handle.clone());
        let stats_for_loop = Arc::clone(&stats);
        let thread = std::thread::Builder::new()
            .name(format!("pie-sched-{driver_idx}"))
            .spawn(move || {
                let _request_timeout = Duration::from_secs(request_timeout_secs);
                Self::run(driver_id, rx, page_size, limits, stats_for_loop);
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
        let _ = crate::driver::registry::clear_scheduler_handle(self.driver_id);
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
        page_size: u32,
        limits: SchedulerLimits,
        stats: Arc<SchedulerStats>,
    ) {
        let mut driver = crate::driver::take_native_driver(driver_id).ok();
        let mut instances = HashMap::new();
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
                match rx.recv_timeout(Duration::from_millis(5)) {
                    Ok(item) => Some(item),
                    Err(crossbeam::channel::RecvTimeoutError::Timeout) => None,
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
            SchedulerItem::Launch {
                pending: mut launch,
                response,
            } => {
                let validation = BatchAccumulator::new(limits, page_size);
                let result = if !instances.contains_key(&launch.instance_id) {
                    Err(anyhow!(
                        "instance {} is unknown or stale",
                        launch.instance_id
                    ))
                } else if let Some(message) = validation.single_request_limit_error(&launch) {
                    Err(anyhow!(message))
                } else if *stopping {
                    Err(anyhow!("scheduler shutting down"))
                } else {
                    launch.acceptance = Some(response.clone());
                    pending.push_back(QueuedItem::Launch(launch));
                    Ok(())
                };
                if result.is_err() {
                    let _ = response.send(result);
                }
            }
            SchedulerItem::RegisterProgram { plan, response } => {
                pending.push_back(QueuedItem::RegisterProgram { plan, response });
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
        }
    }

    fn dispatch_ready_items(
        driver: &mut Option<NativeDriver>,
        instances: &mut HashMap<u64, TrackedInstance>,
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
                    if in_flight_launches.len() >= policy::max_in_flight() {
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
                    Self::dispatch_ordered_item(driver, instances, in_flight_control, item);
                    progress = true;
                }
            }
        }
        progress
    }

    fn dispatch_ordered_item(
        driver: &mut Option<NativeDriver>,
        instances: &mut HashMap<u64, TrackedInstance>,
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
            QueuedItem::BindInstance { plan, response } => {
                let result = match driver.as_mut() {
                    Some(driver) => driver.bind_instance(&plan).and_then(|bound| {
                        if instances.contains_key(&bound.instance_id) {
                            let _ = driver.close_instance(bound.instance_id);
                            return Err(anyhow!("instance {} is already bound", bound.instance_id));
                        }
                        instances.insert(bound.instance_id, TrackedInstance::from_bound(&bound));
                        Ok(bound)
                    }),
                    None => Err(anyhow!("driver has no native backend installed")),
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
                                            instance.close_wait_slots_and_wait();
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
        while let Some(QueuedItem::Launch(next)) = pending.front() {
            if instances
                .get(&next.instance_id)
                .is_some_and(|instance| instance.in_flight != 0)
                || !batch_instances.insert(next.instance_id)
            {
                break;
            }
            if batch.would_exceed(next) {
                break;
            }
            let QueuedItem::Launch(next) = pending.pop_front().expect("launch front") else {
                unreachable!();
            };
            batch.push(next);
            if batch.is_full() {
                break;
            }
        }
        let mut requests = batch.take();
        if requests.is_empty() {
            return false;
        }
        let submission = batch::build_batch_request(&requests, page_size, stats);
        let batch_size = requests.len() as u64;
        let total_tokens = requests
            .iter()
            .map(|req| req.request.token_ids.len())
            .sum::<usize>();
        match driver.as_mut() {
            Some(driver) => match driver.launch(&submission) {
                Ok(completion) => {
                    for request in &mut requests {
                        if let Some(acceptance) = request.acceptance.take() {
                            let _ = acceptance.send(Ok(()));
                        }
                    }
                    for request in &requests {
                        if let Some(instance) = instances.get_mut(&request.instance_id) {
                            instance.in_flight += 1;
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
                        if let Some(acceptance) = request.acceptance.take() {
                            let _ = acceptance.send(Err(anyhow!(message.clone())));
                        }
                    }
                }
            },
            None => {
                for request in &mut requests {
                    if let Some(acceptance) = request.acceptance.take() {
                        let _ =
                            acceptance.send(Err(anyhow!("driver has no native backend installed")));
                    }
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
                Ok(()) => stats::record_fire_stats(
                    stats,
                    retired.started.elapsed(),
                    retired.batch_size,
                    retired.total_tokens,
                ),
                Err(err) => {
                    tracing::warn!(?err, "direct launch completion closed before callback");
                    for request in &retired.requests {
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
            instance.close_wait_slots_and_wait();
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
}

impl TrackedInstance {
    fn from_bound(bound: &BoundInstance) -> Self {
        Self {
            pacing_wait_id: bound.pacing_wait_id,
            wait_slots: bound.wait_slots(),
            in_flight: 0,
        }
    }

    fn close_wait_slots_and_wait(self) {
        self.wait_slots.close_and_wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{self, DriverSpec, LaunchPlan, ProgramRegistration, SchedulerLimits};
    use crate::ptir::PtirChannelValue;
    use pie_driver_abi::{PieKvMoveCell, PiePoolRange};
    use pie_driver_dummy_lib::DummyDriverOptions;
    use pie_ptir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use pie_ptir::op::Op;
    use pie_ptir::registry::Stage;
    use pie_ptir::types::{DType, Literal, Shape};
    use tokio::time::{Duration, timeout};

    async fn setup_scheduler(
        operation_log: Arc<Mutex<Vec<String>>>,
    ) -> anyhow::Result<(usize, BatchScheduler, crate::driver::BoundInstance)> {
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
            capacity: 1,
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

    async fn setup_scheduler_with_options(
        options: DummyDriverOptions,
    ) -> anyhow::Result<(usize, BatchScheduler, crate::driver::BoundInstance)> {
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
        let program_id = driver::register_program(driver_id, dummy_program())?;
        let bound = driver::bind_instance(
            driver_id,
            program_id,
            41,
            vec![7, 8],
            vec![PtirChannelValue {
                channel: 7,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )?;
        Ok((driver_id, scheduler, bound))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn typed_copy_paths_dispatch_to_distinct_driver_methods() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound) = setup_scheduler(operation_log.clone()).await?;

        timeout(
            Duration::from_secs(5),
            driver::copy_kv_cells(
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
            driver::copy_rs_d2d(driver_id, &[3], &[4])?,
        )
        .await??;
        driver::close_instance(&bound)?;

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
        let (driver_id, _scheduler, bound) = setup_scheduler(operation_log.clone()).await?;

        let resize = driver::resize_pool(
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
        crate::inference::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Vec::new(),
            Vec::new(),
            0,
            Vec::new(),
            launch.clone(),
        )?;

        timeout(Duration::from_secs(5), resize).await??;
        timeout(Duration::from_secs(5), launch).await??;
        driver::close_instance(&bound)?;

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
        let (_driver_id, _scheduler, bound) = setup_scheduler(operation_log).await?;
        let pacing_wait_id = bound.pacing_wait_id;
        let channel_waits = bound.channel_waits.clone();

        driver::close_instance(&bound)?;

        assert!(matches!(
            pie_waker::WakerTable::global().publish(pacing_wait_id, 1),
            pie_waker::WakeOutcome::Stale
        ));
        for waits in channel_waits {
            assert!(matches!(
                pie_waker::WakerTable::global().publish(waits.reader_wait_id, 1),
                pie_waker::WakeOutcome::Stale
            ));
            assert!(matches!(
                pie_waker::WakerTable::global().publish(waits.writer_wait_id, 1),
                pie_waker::WakeOutcome::Stale
            ));
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_waits_for_outstanding_completion_lease_before_retiring_slots()
    -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, _scheduler, bound) = setup_scheduler(operation_log).await?;
        let pacing_wait_id = bound.pacing_wait_id;
        let channel_waits = bound.channel_waits.clone();
        let outstanding = bound.reserve_completion();

        let close_bound = std::thread::spawn({
            let bound = bound;
            move || driver::close_instance(&bound)
        });

        std::thread::sleep(Duration::from_millis(10));
        assert!(
            !close_bound.is_finished(),
            "close should wait for outstanding completion leases"
        );
        drop(outstanding);
        close_bound.join().unwrap()?;

        assert!(matches!(
            pie_waker::WakerTable::global().publish(pacing_wait_id, 1),
            pie_waker::WakeOutcome::Stale
        ));
        for waits in channel_waits {
            assert!(matches!(
                pie_waker::WakerTable::global().publish(waits.reader_wait_id, 1),
                pie_waker::WakeOutcome::Stale
            ));
            assert!(matches!(
                pie_waker::WakerTable::global().publish(waits.writer_wait_id, 1),
                pie_waker::WakeOutcome::Stale
            ));
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn published_completion_survives_close_before_late_poll() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound) = setup_scheduler(operation_log).await?;
        let completion = bound.reserve_completion();
        crate::inference::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Vec::new(),
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
            move || driver::close_instance(&bound)
        });
        std::thread::sleep(Duration::from_millis(10));
        assert!(
            !close_bound.is_finished(),
            "close should retain published slots until the late poll drops its lease"
        );

        timeout(Duration::from_secs(5), completion).await??;
        close_bound.join().unwrap()?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn launches_can_overlap_before_prior_callback_when_fifo_allows() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a) = setup_scheduler_with_options(DummyDriverOptions {
            callback_delay_ms: 50,
            operation_log: Some(operation_log.clone()),
            ..DummyDriverOptions::default()
        })
        .await?;
        let bound_b = driver::bind_instance(
            driver_id,
            bound_a.program_id,
            42,
            vec![17, 18],
            vec![PtirChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )?;

        let first = bound_a.reserve_completion();
        crate::inference::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            Vec::new(),
            0,
            Vec::new(),
            first.clone(),
        )?;

        let second = bound_b.reserve_completion();
        crate::inference::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            Vec::new(),
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
        driver::close_instance(&bound_a)?;
        driver::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn same_instance_launch_waits_for_prior_callback() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound) = setup_scheduler_with_options(DummyDriverOptions {
            callback_delay_ms: 50,
            operation_log: Some(operation_log.clone()),
            ..DummyDriverOptions::default()
        })
        .await?;

        let first = bound.reserve_completion();
        crate::inference::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound.instance_id,
            Vec::new(),
            Vec::new(),
            0,
            Vec::new(),
            first.clone(),
        )?;

        let second = bound.reserve_completion();
        let second_for_submit = second.clone();
        let instance_id = bound.instance_id;
        let second_submit = std::thread::spawn(move || {
            crate::inference::submit_prebuilt_async(
                dummy_launch(),
                driver_id,
                instance_id,
                Vec::new(),
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
            "same-instance launch 2 must not race launch 1 host bookkeeping"
        );
        assert!(
            !second_submit.is_finished(),
            "same-instance acceptance must wait for launch 1 callback"
        );

        timeout(Duration::from_secs(5), first).await??;
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
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await?;
        second_submit.join().unwrap()?;
        timeout(Duration::from_secs(5), second).await??;
        driver::close_instance(&bound)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn launch_then_control_then_launch_preserves_fifo_order() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, _scheduler, bound_a) = setup_scheduler_with_options(DummyDriverOptions {
            callback_delay_ms: 50,
            operation_log: Some(operation_log.clone()),
            ..DummyDriverOptions::default()
        })
        .await?;
        let bound_b = driver::bind_instance(
            driver_id,
            bound_a.program_id,
            42,
            vec![17, 18],
            vec![PtirChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )?;

        let first = bound_a.reserve_completion();
        crate::inference::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            Vec::new(),
            0,
            Vec::new(),
            first.clone(),
        )?;

        let resize_join = std::thread::spawn(move || {
            driver::resize_pool(
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
            crate::inference::submit_prebuilt_async(
                dummy_launch(),
                second_driver_id,
                second_instance_id,
                Vec::new(),
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
            !second_submit.is_finished(),
            "launch acceptance should wait until the queued control retires"
        );

        timeout(Duration::from_secs(5), first).await??;
        let resize = resize_join.join().unwrap()?;
        tokio::time::sleep(Duration::from_millis(10)).await;
        let log = operation_log.lock().unwrap().clone();
        assert_eq!(
            log.iter()
                .filter(|entry| entry.as_str() == "launch")
                .count(),
            1,
            "second launch must still wait behind the queued control completion"
        );
        assert!(
            log.iter().any(|entry| entry == "resize_pool"),
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
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?;
        second_submit.join().unwrap()?;
        timeout(Duration::from_secs(5), second).await??;
        driver::close_instance(&bound_a)?;
        driver::close_instance(&bound_b)?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_waits_for_accepted_launch_then_succeeds() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (_driver_id, _scheduler, bound) = setup_scheduler_with_options(DummyDriverOptions {
            callback_delay_ms: 75,
            operation_log: Some(operation_log.clone()),
            ..DummyDriverOptions::default()
        })
        .await?;

        let launch = bound.reserve_completion();
        crate::inference::submit_prebuilt_async(
            dummy_launch(),
            bound.driver_id,
            bound.instance_id,
            Vec::new(),
            Vec::new(),
            0,
            Vec::new(),
            launch.clone(),
        )?;

        let close_bound = std::thread::spawn({
            let bound = bound;
            move || driver::close_instance(&bound)
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
            let (_driver_id, _scheduler, bound) = setup_scheduler(operation_log).await?;
            driver::close_instance(&bound)?;
            let err = driver::close_instance(&bound).unwrap_err();
            assert!(err.to_string().contains("stale"));
            Ok::<_, anyhow::Error>(())
        })?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scheduler_shutdown_drains_instances_and_destroys_once() -> anyhow::Result<()> {
        let operation_log = Arc::new(Mutex::new(Vec::new()));
        let (driver_id, scheduler, bound_a) = setup_scheduler_with_options(DummyDriverOptions {
            callback_delay_ms: 40,
            operation_log: Some(operation_log.clone()),
            ..DummyDriverOptions::default()
        })
        .await?;
        let program_id = driver::register_program(driver_id, dummy_program())?;
        let bound_b = driver::bind_instance(
            driver_id,
            program_id,
            42,
            vec![17, 18],
            vec![PtirChannelValue {
                channel: 17,
                bytes: 1u32.to_le_bytes().to_vec(),
            }],
        )?;

        let resize = driver::resize_pool(driver_id, 9, 16, Vec::new(), Vec::new())?;
        let a = bound_a.reserve_completion();
        crate::inference::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_a.instance_id,
            Vec::new(),
            Vec::new(),
            0,
            Vec::new(),
            a,
        )?;
        let b = bound_b.reserve_completion();
        crate::inference::submit_prebuilt_async(
            dummy_launch(),
            driver_id,
            bound_b.instance_id,
            Vec::new(),
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
    async fn timeout_bounded_shutdown_stress() -> anyhow::Result<()> {
        timeout(Duration::from_secs(5), async {
            let operation_log = Arc::new(Mutex::new(Vec::new()));
            let (driver_id, scheduler, bound) = setup_scheduler_with_options(DummyDriverOptions {
                callback_delay_ms: 5,
                operation_log: Some(operation_log),
                ..DummyDriverOptions::default()
            })
            .await?;
            for _ in 0..16 {
                let completion = bound.reserve_completion();
                crate::inference::submit_prebuilt_async(
                    dummy_launch(),
                    driver_id,
                    bound.instance_id,
                    Vec::new(),
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
