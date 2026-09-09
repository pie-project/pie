pub(crate) mod batch;
pub(crate) mod dispatch;
pub(crate) mod fire_plan;
pub(crate) mod frame;
pub(crate) mod probe;
pub(crate) mod stats;
pub mod worker;

pub use frame::FrameStamp;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

use anyhow::{Result, anyhow};

#[allow(unused_imports)]
pub(crate) use dispatch::{
    bind_instance, bind_instance_classified, close_channels, close_instance, copy_d2d, copy_d2h,
    copy_d2h_tracked, copy_h2d, copy_h2d_tracked, copy_h2h, copy_kv_cells, copy_rs_d2d,
    register_channel, register_channels, register_channels_bind_classified, register_program,
};
pub use stats::{AggregateStats, HostSubmitStats};
pub use worker::BatchScheduler;
use worker::SchedulerHandle;

use crate::engine::EngineId;

pub type ProcessId = uuid::Uuid;

#[derive(Clone)]
pub(crate) struct ControlCompletion {
    inner: Arc<ControlCompletionState>,
}

struct ControlCompletionState {
    result: Mutex<Option<std::result::Result<(), String>>>,
    notify: tokio::sync::Notify,
}

impl ControlCompletion {
    fn new() -> Self {
        Self {
            inner: Arc::new(ControlCompletionState {
                result: Mutex::new(None),
                notify: tokio::sync::Notify::new(),
            }),
        }
    }

    pub(crate) async fn wait(&self) -> Result<()> {
        loop {
            if let Some(result) = self.inner.result.lock().unwrap().clone() {
                return result.map_err(anyhow::Error::msg);
            }
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = self.inner.result.lock().unwrap().clone() {
                return result.map_err(anyhow::Error::msg);
            }
            notified.await;
        }
    }

    fn resolve(&self, result: &Result<()>) {
        let result = result
            .as_ref()
            .map(|_| ())
            .map_err(|error| format!("{error:#}"));
        *self.inner.result.lock().unwrap() = Some(result);
        self.inner.notify.notify_waiters();
    }
}

pub(crate) fn fire_timing_now_us() -> u64 {
    ledger_monotonic_ns() / 1_000
}

pub(crate) fn ledger_monotonic_ns() -> u64 {
    let mut value = std::mem::MaybeUninit::<libc::timespec>::uninit();
    let status = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, value.as_mut_ptr()) };
    assert_eq!(status, 0, "CLOCK_MONOTONIC is unavailable");
    let value = unsafe { value.assume_init() };
    (value.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(value.tv_nsec as u64)
}

fn handle_registry() -> &'static RwLock<Vec<Option<SchedulerHandle>>> {
    static REGISTRY: std::sync::OnceLock<RwLock<Vec<Option<SchedulerHandle>>>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

pub(crate) fn install_scheduler_handle(engine_id: usize, scheduler: SchedulerHandle) {
    let mut handles = handle_registry().write().unwrap();
    if handles.len() <= engine_id {
        handles.resize_with(engine_id + 1, || None);
    }
    handles[engine_id] = Some(scheduler);
}

pub(crate) fn clear_scheduler_handle(engine_id: usize) {
    let mut handles = handle_registry().write().unwrap();
    if let Some(slot) = handles.get_mut(engine_id) {
        *slot = None;
    }
}

pub(crate) fn scheduler_handle(engine_id: usize) -> Result<SchedulerHandle> {
    handle_registry()
        .read()
        .unwrap()
        .get(engine_id)
        .and_then(|slot| slot.clone())
        .ok_or_else(|| anyhow!("engine {engine_id} has no scheduler"))
}

pub async fn debug_dump(engine_id: usize) -> Result<String> {
    scheduler_handle(engine_id)?.debug_dump().await
}

pub fn configured_submit_deadline() -> Duration {
    *SUBMIT_DEADLINE.get_or_init(|| Duration::from_micros(50_000))
}

pub fn set_submit_deadline(deadline: Duration) {
    let _ = SUBMIT_DEADLINE.set(deadline);
}

static SUBMIT_DEADLINE: OnceLock<Duration> = OnceLock::new();

pub fn configured_silence_timeout() -> Duration {
    *SILENCE_TIMEOUT.get_or_init(|| Duration::from_secs(30))
}

pub fn set_silence_timeout(timeout: Duration) {
    let _ = SILENCE_TIMEOUT.set(timeout);
}

static SILENCE_TIMEOUT: OnceLock<Duration> = OnceLock::new();

pub fn configured_frame_size() -> usize {
    match FRAME_SIZE.load(Ordering::Relaxed) {
        0 => DEFAULT_FRAME_SIZE,
        k => k,
    }
}

pub fn set_frame_size(frame_size: usize) {
    FRAME_SIZE.store(frame_size, Ordering::Relaxed);
}

const DEFAULT_FRAME_SIZE: usize = 2;
static FRAME_SIZE: AtomicUsize = AtomicUsize::new(0);

pub fn configured_submit_depth() -> usize {
    ::engine::runahead::Runahead::of(u8::try_from(configured_dispatch_depth()).unwrap_or(u8::MAX))
        .submit_depth()
}

pub fn set_seal_default_ready(ready: bool) {
    frame::set_seal_default_ready(ready);
}

pub fn set_seal_coalesce_default(window: std::time::Duration) {
    frame::set_seal_coalesce_default(window);
}

pub fn set_dispatch_depth(depth: usize) {
    frame::set_dispatch_depth(depth);
}

pub fn configured_dispatch_depth() -> usize {
    frame::configured_dispatch_depth()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconfigureRefused {
    Busy(usize),
}

impl std::fmt::Display for ReconfigureRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy(n) => write!(
                f,
                "{n} process(es) still live; these knobs can only change while \
                 the runtime is idle"
            ),
        }
    }
}

impl std::error::Error for ReconfigureRefused {}

pub fn reconfigure(frame_size: usize, dispatch_depth: usize) -> Result<(), ReconfigureRefused> {
    let live = crate::inferlet::process::live_count();
    if live > 0 {
        return Err(ReconfigureRefused::Busy(live));
    }
    set_frame_size(frame_size);
    set_dispatch_depth(dispatch_depth);
    Ok(())
}

pub fn channel_capacity() -> usize {
    ::engine::runahead::Runahead::of(u8::try_from(configured_dispatch_depth()).unwrap_or(u8::MAX))
        .channel_capacity(configured_frame_size())
}

pub fn run_ahead_window() -> usize {
    ::engine::runahead::Runahead::of(u8::try_from(configured_dispatch_depth()).unwrap_or(u8::MAX))
        .submit_depth()
        * configured_frame_size()
}

pub struct SchedulerShutdownHandle {
    schedulers: Vec<BatchScheduler>,
}

fn dynamic_schedulers() -> &'static Mutex<HashMap<EngineId, BatchScheduler>> {
    static SCHEDULERS: OnceLock<Mutex<HashMap<EngineId, BatchScheduler>>> = OnceLock::new();
    SCHEDULERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn build_engine_scheduler(
    engine_id: EngineId,
    page_size: u32,
    request_timeout_secs: u64,
) -> Result<BatchScheduler> {
    let limits = crate::engine::get_spec(engine_id)?.scheduler_limits();
    Ok(BatchScheduler::new(
        engine_id,
        engine_id,
        page_size,
        limits,
        request_timeout_secs,
        configured_frame_size(),
    ))
}

pub fn spawn_engine(engine_id: EngineId, page_size: u32, request_timeout_secs: u64) -> Result<()> {
    let mut schedulers = dynamic_schedulers().lock().unwrap();
    if schedulers.contains_key(&engine_id) {
        return Err(anyhow!(
            "engine {engine_id} already has a dynamic scheduler"
        ));
    }
    let scheduler = build_engine_scheduler(engine_id, page_size, request_timeout_secs)?;
    schedulers.insert(engine_id, scheduler);
    Ok(())
}

pub fn stop_engine(engine_id: EngineId) -> Result<()> {
    let scheduler = dynamic_schedulers()
        .lock()
        .unwrap()
        .remove(&engine_id)
        .ok_or_else(|| anyhow!("engine {engine_id} has no dynamic scheduler"))?;
    drop(scheduler);
    Ok(())
}

impl SchedulerShutdownHandle {
    pub async fn shutdown(self) -> Result<()> {
        drop(self.schedulers);
        Ok(())
    }
}

pub async fn spawn(
    engine_indices: &[usize],
    page_size: u32,
    request_timeout_secs: u64,
) -> Result<SchedulerShutdownHandle> {
    let schedulers: Vec<BatchScheduler> = engine_indices
        .iter()
        .map(|&engine_id| build_engine_scheduler(engine_id, page_size, request_timeout_secs))
        .collect::<Result<_>>()?;

    Ok(SchedulerShutdownHandle { schedulers })
}

fn rs_state_copy_plan(
    src_slots: Vec<u32>,
    dst_slots: Vec<u32>,
) -> Result<Option<::engine::StateCopy>> {
    if src_slots.len() != dst_slots.len() {
        return Err(anyhow!(
            "recurrent-state copy source/destination lengths differ: {} != {}",
            src_slots.len(),
            dst_slots.len()
        ));
    }
    if src_slots.is_empty() {
        return Ok(None);
    }
    let slot_ranges = src_slots
        .into_iter()
        .zip(dst_slots)
        .map(|(src_slot_id, dst_slot_id)| ::engine::StateMove {
            src_slot_id,
            dst_slot_id,
            src_token_offset: 0,
            dst_token_offset: 0,
            token_count: 0,
        })
        .collect();
    Ok(Some(::engine::StateCopy { moves: slot_ranges }))
}

pub fn submit_async(
    request: crate::engine::FireRequest,
    engine_idx: usize,
    instance_id: u64,
    pipeline_id: Option<ProcessId>,
    completion: crate::engine::WorkItemCompletion,
) -> Result<()> {
    submit_async_with_kv_copy(
        request,
        engine_idx,
        instance_id,
        pipeline_id,
        completion,
        Vec::new(),
        Vec::new(),
    )
}

pub(crate) fn device_domain(engine_idx: usize) -> ::engine::MemoryDomain {
    crate::engine::get_spec(engine_idx)
        .map_or(::engine::MemoryDomain::HostPinned, |s| s.device_domain)
}

pub(crate) fn nudge(engine_idx: usize) {
    if let Ok(handle) = scheduler_handle(engine_idx) {
        let _ = handle.nudge();
    }
}

#[allow(clippy::too_many_arguments)]
pub fn submit_async_with_kv_copy(
    request: crate::engine::FireRequest,
    engine_idx: usize,
    instance_id: u64,
    pipeline_id: Option<ProcessId>,
    completion: crate::engine::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
) -> Result<()> {
    let prelaunch_copy = (!copy_src.is_empty()).then_some(::engine::KvCopy {
        src: device_domain(engine_idx),
        dst: device_domain(engine_idx),
        src_page_ids: copy_src,
        dst_page_ids: copy_dst,
        moves: Vec::new(),
    });
    scheduler_handle(engine_idx)?.submit_with_identity_and_copy(
        request,
        instance_id,
        completion,
        pipeline_id,
        prelaunch_copy,
        None,
    )
}

pub fn submit_prebuilt_async(
    request: crate::engine::FireRequest,
    engine_idx: usize,
    instance_id: u64,
    completion: crate::engine::WorkItemCompletion,
) -> Result<()> {
    submit_prebuilt_async_with_kv_copy(
        request,
        engine_idx,
        instance_id,
        completion,
        Vec::new(),
        Vec::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn submit_prebuilt_async_with_kv_copy(
    request: crate::engine::FireRequest,
    engine_idx: usize,
    instance_id: u64,
    completion: crate::engine::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
) -> Result<()> {
    let prelaunch_copy = (!copy_src.is_empty()).then_some(::engine::KvCopy {
        src: device_domain(engine_idx),
        dst: device_domain(engine_idx),
        src_page_ids: copy_src,
        dst_page_ids: copy_dst,
        moves: Vec::new(),
    });
    scheduler_handle(engine_idx)?.submit_prebuilt_with_copy(
        request,
        instance_id,
        completion,
        prelaunch_copy,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn submit_prebuilt_async_with_kv_and_rs_copy(
    request: crate::engine::FireRequest,
    engine_idx: usize,
    instance_id: u64,
    completion: crate::engine::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
    rs_copy_src: Vec<u32>,
    rs_copy_dst: Vec<u32>,
) -> Result<()> {
    let prelaunch_copy = (!copy_src.is_empty()).then_some(::engine::KvCopy {
        src: device_domain(engine_idx),
        dst: device_domain(engine_idx),
        src_page_ids: copy_src,
        dst_page_ids: copy_dst,
        moves: Vec::new(),
    });
    scheduler_handle(engine_idx)?.submit_prebuilt_with_copy(
        request,
        instance_id,
        completion,
        prelaunch_copy,
        rs_state_copy_plan(rs_copy_src, rs_copy_dst)?,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn submit_prebuilt_tracked_async_with_kv_and_rs_copy(
    request: crate::engine::FireRequest,
    engine_idx: usize,
    instance_id: u64,
    pipeline_id: ProcessId,
    completion: crate::engine::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
    rs_copy_src: Vec<u32>,
    rs_copy_dst: Vec<u32>,
) -> Result<()> {
    submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
        &scheduler_handle(engine_idx)?,
        request,
        instance_id,
        pipeline_id,
        pipeline_id,
        completion,
        copy_src,
        copy_dst,
        rs_copy_src,
        rs_copy_dst,
        None,
        /*hook_program=*/ false,
        /*lora_program=*/ false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
    handle: &worker::SchedulerHandle,
    request: crate::engine::FireRequest,
    instance_id: u64,
    process_id: ProcessId,
    pipeline_id: ProcessId,
    completion: crate::engine::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
    rs_copy_src: Vec<u32>,
    rs_copy_dst: Vec<u32>,
    frame: Option<FrameStamp>,
    hook_program: bool,
    lora_program: bool,
) -> Result<()> {
    let prelaunch_copy = (!copy_src.is_empty()).then_some(::engine::KvCopy {
        src: handle.device_domain(),
        dst: handle.device_domain(),
        src_page_ids: copy_src,
        dst_page_ids: copy_dst,
        moves: Vec::new(),
    });
    handle.submit_prebuilt_tracked_with_copy(
        request,
        instance_id,
        completion,
        process_id,
        pipeline_id,
        prelaunch_copy,
        rs_state_copy_plan(rs_copy_src, rs_copy_dst)?,
        frame,
        hook_program,
        lora_program,
    )
}

pub async fn get_stats() -> AggregateStats {
    let scheduler_stats: Vec<Arc<stats::SchedulerStats>> = handle_registry()
        .read()
        .unwrap()
        .iter()
        .filter_map(|slot| slot.as_ref().map(|handle| handle.stats()))
        .collect();
    stats::aggregate(&scheduler_stats)
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_unregistered_engine_names_no_devices_memory() {
        let domain = super::device_domain(usize::MAX);
        assert_eq!(domain, ::engine::MemoryDomain::HostPinned);
        assert_ne!(domain, ::engine::MemoryDomain::CudaDevice(0));
        assert_ne!(domain, ::engine::MemoryDomain::VulkanDevice(0));
        assert_ne!(domain, ::engine::MemoryDomain::WgpuDevice(0));
        assert_ne!(domain, ::engine::MemoryDomain::MetalPrivate);
    }
}
