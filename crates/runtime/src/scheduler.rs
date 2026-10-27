pub(crate) mod batch;
pub(crate) mod dispatch;
pub(crate) mod fire_plan;
pub(crate) mod frame;
pub mod lane;
pub(crate) mod probe;
pub(crate) mod stats;
pub mod worker;

pub use frame::FrameStamp;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

use anyhow::{Result, anyhow};

pub(crate) use dispatch::{
    TierKind, TierMove, close_channels, copy_tier_tracked, register_channels_bind_classified,
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
    // `Instant` is the monotonic clock on every platform this ships to. The
    // epoch is the first call rather than boot, which the ledger does not
    // care about: every consumer reads these as deltas or as trace ordering,
    // never against a wall clock or a device timestamp.
    static START: std::sync::OnceLock<crate::rt::Instant> = std::sync::OnceLock::new();
    let start = *START.get_or_init(crate::rt::Instant::now);
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
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

/// Device pages `device` copied into host slots `slots`, and those slots'
/// bytes.
pub(crate) async fn read_device_kv(
    engine_id: usize,
    device: Vec<u32>,
    slots: Vec<u32>,
) -> Result<Vec<u8>> {
    let copy = ::engine::KvCopy {
        src: device_domain(engine_id),
        dst: ::engine::MemoryDomain::HostPinned,
        src_page_ids: device,
        dst_page_ids: slots.clone(),
        moves: Vec::new(),
        windowed: Vec::new(),
    };
    scheduler_handle(engine_id)?
        .host_kv(worker::HostKvOp::CopyRead(copy, slots))
        .await
}

pub(crate) async fn write_host_kv(engine_id: usize, slots: Vec<u32>, bytes: Vec<u8>) -> Result<()> {
    scheduler_handle(engine_id)?
        .host_kv(worker::HostKvOp::Write(slots, bytes))
        .await
        .map(drop)
}

/// The engine's host-tier layout, once asked.
pub(crate) async fn host_kv_layout(engine_id: usize) -> Option<state::PageLayout> {
    static LAYOUTS: std::sync::Mutex<Vec<Option<Option<state::PageLayout>>>> =
        std::sync::Mutex::new(Vec::new());
    if let Some(Some(known)) = LAYOUTS.lock().unwrap().get(engine_id) {
        return known.clone();
    }
    let bytes = scheduler_handle(engine_id)
        .ok()?
        .host_kv(worker::HostKvOp::Layout)
        .await
        .ok()?;
    let layout: Option<state::PageLayout> = serde_json::from_slice(&bytes).ok();
    let mut layouts = LAYOUTS.lock().unwrap();
    if layouts.len() <= engine_id {
        layouts.resize(engine_id + 1, None);
    }
    layouts[engine_id] = Some(layout.clone());
    layout
}

/// The groups the engine reads a tapped page's planes out in, as spans of a
/// host slot; `None` where it taps nothing. Once asked, known.
pub(crate) async fn tap_groups(engine_id: usize) -> Option<::engine::TapGroups> {
    type Known = Option<Option<::engine::TapGroups>>;
    static GROUPS: std::sync::Mutex<Vec<Known>> = std::sync::Mutex::new(Vec::new());
    if let Some(Some(known)) = GROUPS.lock().unwrap().get(engine_id) {
        return known.clone();
    }
    let bytes = scheduler_handle(engine_id)
        .ok()?
        .host_kv(worker::HostKvOp::TapGroups)
        .await
        .ok()?;
    let groups: Option<::engine::TapGroups> = serde_json::from_slice(&bytes).ok().flatten();
    let mut known = GROUPS.lock().unwrap();
    if known.len() <= engine_id {
        known.resize(engine_id + 1, None);
    }
    known[engine_id] = Some(groups.clone());
    groups
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

fn build_engine_scheduler(engine_id: EngineId, page_size: u32) -> Result<BatchScheduler> {
    let limits = crate::engine::get_spec(engine_id)?.scheduler_limits();
    Ok(BatchScheduler::new(
        engine_id,
        engine_id,
        page_size,
        limits,
        configured_frame_size(),
    ))
}

impl SchedulerShutdownHandle {
    pub async fn shutdown(self) -> Result<()> {
        drop(self.schedulers);
        Ok(())
    }
}

pub async fn spawn(engine_indices: &[usize], page_size: u32) -> Result<SchedulerShutdownHandle> {
    let schedulers: Vec<BatchScheduler> = engine_indices
        .iter()
        .map(|&engine_id| build_engine_scheduler(engine_id, page_size))
        .collect::<Result<_>>()?;

    Ok(SchedulerShutdownHandle { schedulers })
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
pub(crate) fn submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
    handle: &worker::SchedulerHandle,
    request: crate::engine::FireRequest,
    instance_id: ::engine::InstanceId,
    process_id: ProcessId,
    pipeline_id: ProcessId,
    completion: crate::engine::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
    rs_copy: Option<::engine::StateCopy>,
    frame: Option<FrameStamp>,
    lora_program: bool,
) -> Result<()> {
    let prelaunch_copy = (!copy_src.is_empty()).then_some(::engine::KvCopy {
        src: handle.device_domain(),
        dst: handle.device_domain(),
        src_page_ids: copy_src,
        dst_page_ids: copy_dst,
        moves: Vec::new(),
        windowed: Vec::new(),
    });
    handle.submit_prebuilt_tracked_with_copy(
        request,
        instance_id,
        completion,
        process_id,
        pipeline_id,
        prelaunch_copy,
        rs_copy,
        frame,
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

#[cfg_attr(
    target_arch = "wasm32",
    allow(dead_code, reason = "no policy runs in the browser")
)]
pub(crate) fn last_batch_us() -> Option<u64> {
    handle_registry()
        .read()
        .unwrap()
        .iter()
        .flatten()
        .map(|handle| {
            handle
                .stats()
                .last_batch_latency_us
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .filter(|&us| us > 0)
        .max()
}

/// The processes in each engine's last fire, summed over engines.
#[cfg_attr(
    target_arch = "wasm32",
    allow(dead_code, reason = "no policy runs in the browser")
)]
pub(crate) fn last_batch_size() -> u64 {
    handle_registry()
        .read()
        .unwrap()
        .iter()
        .flatten()
        .map(|handle| {
            handle
                .stats()
                .last_batch_size
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .sum()
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
