//! Per-driver batching: when accumulated fires launch.
//!
//! - [`worker`]: `BatchScheduler` — the per-driver run loop (accumulate,
//!   decide, dispatch, retire). The only public submodule (external crates
//!   construct `worker::BatchScheduler` directly for host-driver test
//!   harnesses); `batch`/`dispatch`/`frame`/`probe`/`stats`/`wire` are
//!   internal.
//! - `batch`: capacity accounting + the dense-batch accumulator.
//! - `dispatch`: the driver ABI's per-`driver_id` verbs (`register_program`,
//!   `bind_instance`, the `copy_*` family, ...) — re-exported at this
//!   module's root since they call [`scheduler_handle`], which is
//!   scheduler-owned state.
//! - `wire`: owned `LaunchPlan`s -> the batched wire request, page-trim.
//! - `frame`: the wait-all-active-lanes frame fire rule (every k,
//!   including the default single-slot k = 1).
//! - `stats`: `SchedulerStats` (per-driver, lock-free) + [`AggregateStats`]
//!   (cross-driver, this module's `get_stats`).
//! - `probe`: per-fire lifecycle probes (`profile-fire` feature).
//!
//! This module also owns the driver-id -> `SchedulerHandle` registry: the
//! `dispatch` trampolines and this module's own `submit_async`/
//! `submit_prebuilt_async` look a handle up here to reach the scheduler
//! that owns a given `driver_id`. `driver/` (L0) never imports this module.

pub(crate) mod batch;
// tart (V2): the fire planner — seriation (deepest-first bands, the
// gray sentinel) and the per-site lowerings. Orphaned by the dev merge
// (upstream's assembly does not call it yet); re-declared so the module
// and its pinned tests stay live while the 0.3 regraft lands
// (playbook: "0.3 re-port step 1").
pub(crate) mod fire_plan;
pub(crate) mod dispatch;
pub(crate) mod frame;
pub(crate) mod probe;
pub(crate) mod stats;
pub(crate) mod wire;
pub mod worker;

pub use frame::FrameStamp;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

use anyhow::{Result, anyhow};

// `copy_d2h`/`copy_h2d`/`copy_h2h`/`copy_rs_d2d`/`resize_pool` round out the
// driver ABI verb surface (see `dispatch`'s module doc for which are wired
// into the current mock-driver fire path vs. reserved/unit-test-only).
#[allow(unused_imports)]
pub(crate) use dispatch::{
    bind_instance, bind_instance_classified, close_channels, close_instance, copy_d2d, copy_d2h,
    copy_d2h_tracked, copy_h2d, copy_h2d_tracked, copy_h2h, copy_kv_cells, copy_rs_d2d,
    register_channel, register_channels, register_channels_bind_classified, register_program,
    resize_pool,
};
pub use stats::AggregateStats;
pub use worker::BatchScheduler;
use worker::SchedulerHandle;

use crate::driver::DriverId;

/// Process identity the scheduler and wait-all fire rule track (co-batch
/// membership, wait-set keys). Kept as the leaf `uuid::Uuid` representation
/// so the scheduler stays below the guest runtime in the layering.
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

// =============================================================================
/// Host `CLOCK_MONOTONIC` timestamp used by scheduler timing records and the
/// opt-in guest/client ledger clock.
/// Callers guard this with [`fire_timing_enabled`] so disabled builds do not
/// execute an `Instant::now()` on the hot path.
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

// Scheduler handle registry (moved out of `driver/registry.rs`)
// =============================================================================

fn handle_registry() -> &'static RwLock<Vec<Option<SchedulerHandle>>> {
    static REGISTRY: std::sync::OnceLock<RwLock<Vec<Option<SchedulerHandle>>>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

/// Install the scheduler handle for `driver_id` (called once, from
/// [`BatchScheduler::new`]).
pub(crate) fn install_scheduler_handle(driver_id: usize, scheduler: SchedulerHandle) {
    let mut handles = handle_registry().write().unwrap();
    if handles.len() <= driver_id {
        handles.resize_with(driver_id + 1, || None);
    }
    handles[driver_id] = Some(scheduler);
}

/// Clear the scheduler handle for `driver_id` (called once, from
/// [`BatchScheduler`]'s shutdown).
pub(crate) fn clear_scheduler_handle(driver_id: usize) {
    let mut handles = handle_registry().write().unwrap();
    if let Some(slot) = handles.get_mut(driver_id) {
        *slot = None;
    }
}

/// The installed scheduler handle for `driver_id`, or an error if none is
/// installed (the `dispatch` trampolines call this).
pub(crate) fn scheduler_handle(driver_id: usize) -> Result<SchedulerHandle> {
    handle_registry()
        .read()
        .unwrap()
        .get(driver_id)
        .and_then(|slot| slot.clone())
        .ok_or_else(|| anyhow!("driver {driver_id} has no scheduler"))
}

/// Human-readable snapshot of driver `driver_id`'s run-loop state (queue
/// composition, in-flight work, wave barrier membership). For diagnostics on
/// a stalled fleet — a held wave must be inspectable from outside the thread.
pub async fn debug_dump(driver_id: usize) -> Result<String> {
    scheduler_handle(driver_id)?.debug_dump().await
}

// =============================================================================
// Frame size (`[model.scheduler] frame_size`) — the Vesuvius constant k
// =============================================================================

/// How long a lane that is hard-blocking a frame's seal may go without
/// submitting before the engine stops waiting for it
/// (`[model.scheduler] submit_deadline_us`, default 50ms). Guests read it as
/// `model.submit-deadline-us()`.
///
/// Small enough to be a real bound on fleet exposure because it measures a
/// much narrower interval than its size suggests: the clock runs only while
/// the lane is an awaited member with nothing submitted, and is stopped by
/// run-ahead, by an unretired dispatch (the engine owes it a result), by a
/// bind in flight, and by `forward.park()`. The host round trip has its own
/// headroom in [`configured_submit_depth`].
///
/// This number can no longer kill: at the deadline the lane is dropped from
/// the wait-set (an involuntary `forward.park()`), its queued frames still
/// dispatch, and its next fire rejoins. It is a density bound — how long the
/// fleet waits for a straggler — so a value that is too small costs a little
/// epoch density and never a request. Termination is a separate, far longer
/// verdict; see [`configured_silence_timeout`].
pub fn configured_submit_deadline() -> Duration {
    *SUBMIT_DEADLINE.get_or_init(|| Duration::from_micros(50_000))
}

/// Install the configured deadline at bootstrap. First writer wins; later
/// calls are ignored so the value a guest has already read cannot change.
pub fn set_submit_deadline(deadline: Duration) {
    let _ = SUBMIT_DEADLINE.set(deadline);
}

static SUBMIT_DEADLINE: OnceLock<Duration> = OnceLock::new();

/// How long a lane may stay silent in total before its process is
/// terminated. Unlike the leash above this IS a verdict, so it is generous:
/// the leash already keeps a straggler from holding the fleet, which means
/// nothing but an abandoned pipeline ever reaches this. A guest that means to
/// go quiet calls `forward.park()`, which ends the silence and is never
/// killed — that is exactly the contract this enforces.
///
/// Configured by `[model.scheduler] silence_timeout_secs` (default 30s).
pub fn configured_silence_timeout() -> Duration {
    *SILENCE_TIMEOUT.get_or_init(|| Duration::from_secs(30))
}

/// Install the configured silence timeout at bootstrap. First writer wins.
pub fn set_silence_timeout(timeout: Duration) {
    let _ = SILENCE_TIMEOUT.set(timeout);
}

static SILENCE_TIMEOUT: OnceLock<Duration> = OnceLock::new();

/// Waves per frame (k): a static deployment constant, fixed at engine start
/// exactly like the KV page size — never renegotiated per frame and never
/// adapted from runtime timing. Guests query it via `model.frame-size()` and
/// size their frames/channels to it.
///
/// The default is 2. At k = 1 the wait-all quorum runs once per token, and
/// above ~64 concurrent processes the fleet stops overlapping batches
/// entirely — measured duty (forward batches in flight) collapses from 1.7
/// to 1.0 and becomes bimodal, costing 29% throughput and 28% latency at
/// concurrency 256. k = 2 halves the number of quorum boundaries and holds
/// duty at 1.6 with no regression at any lower concurrency. k = 3 and k = 4
/// measure the same as k = 2 while costing more driver staging depth, so 2
/// is the setting (CONTENTION_FOLLOWUP §20.8). Set `[model.scheduler]
/// frame_size = 1` to restore the per-wave path.
pub fn configured_frame_size() -> usize {
    match FRAME_SIZE.load(Ordering::Relaxed) {
        0 => DEFAULT_FRAME_SIZE,
        k => k,
    }
}

/// Install the configured frame size at bootstrap.
pub fn set_frame_size(frame_size: usize) {
    FRAME_SIZE.store(frame_size, Ordering::Relaxed);
}

const DEFAULT_FRAME_SIZE: usize = 2;
/// `0` = never installed, so [`configured_frame_size`] answers the default.
/// Not a `OnceLock` any more; see [`reconfigure`].
static FRAME_SIZE: AtomicUsize = AtomicUsize::new(0);

// =============================================================================
// Guest run-ahead sizing
// =============================================================================

/// Frames a lane keeps outstanding: one running, plus the rest queued behind
/// it so the device always has work while the guest is back on the host
/// deciding what to submit next. Too few and the pipeline collapses to
/// lockstep — submit, wait, submit, wait — because the host round trip lands
/// squarely in the critical path.
///
/// The default is 3, sized for the default k = 2, where it covers the round
/// trip with one frame to spare. It is stated as a flat frame count rather
/// than derived, so this and [`configured_frame_size`] are NOT independent: a
/// frame is k waves of device time, so the same three frames cover
/// proportionally less real time as k shrinks, and a deployment that moves k
/// must re-measure this. Undersizing this window is what collapsed k = 1
/// throughput (CONTENTION_FOLLOWUP §20.11).
///
/// Guests never read this directly — they get `model.channel-capacity()`, so
/// it can move without touching the guest contract.
pub fn configured_submit_depth() -> usize {
    match SUBMIT_DEPTH.load(Ordering::Relaxed) {
        0 => DEFAULT_SUBMIT_DEPTH,
        frames => frames,
    }
}

/// Install the configured window at bootstrap.
pub fn set_submit_depth(frames: usize) {
    SUBMIT_DEPTH.store(frames, Ordering::Relaxed);
}

/// Install the configured dispatch depth at bootstrap.
pub fn set_dispatch_depth(depth: usize) {
    frame::set_dispatch_depth(depth);
}

const DEFAULT_SUBMIT_DEPTH: usize = 3;
/// `0` = never installed; see [`reconfigure`].
static SUBMIT_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Why a [`reconfigure`] was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconfigureRefused {
    /// Guests are running. The count is what was seen.
    Busy(usize),
}

impl std::fmt::Display for ReconfigureRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy(n) => write!(
                f,
                "{n} process(es) still live; these knobs can only change while \
                 the engine is idle"
            ),
        }
    }
}

impl std::error::Error for ReconfigureRefused {}

/// Change the batching knobs on a running engine, between rounds of a
/// measurement sweep.
///
/// **This is a quiesce gate, not a drain barrier, and the difference is the
/// point.** A drain barrier would let in-flight frames retire and then swap.
/// That is not sufficient here, because [`configured_frame_size`] is not
/// engine-internal state: it is handed to guests through `model.frame-size()`
/// (`crate::inferlet::host::model`), and the SDK caches it for the life of the
/// program (`ptir.rs`, a per-thread `OnceLock`). A value already read cannot be
/// recalled by anything the engine does afterwards, so a guest that survives
/// the swap keeps building frames to the old k while the engine expects the
/// new one. No barrier fixes that; only the absence of guests does.
///
/// Which costs nothing, because the sweep restarts the load between rounds
/// anyway — restarting a guest is milliseconds and restarting the model is
/// minutes, and that asymmetry is the whole reason these knobs became
/// swappable rather than staying boot-fixed.
///
/// The joint bound against the driver's staging pool
/// (`frame_dispatch_depth * frame_size < kUploadStagingDepth`) is NOT checked
/// here. `SchedulerConfig::validate` owns it, is where both factors are visible
/// at once, and is the one place that knows the driver's constant; duplicating
/// it would create a second spelling that can disagree. Callers pass values
/// that came through that validation.
pub fn reconfigure(
    frame_size: usize,
    submit_depth: usize,
    dispatch_depth: usize,
) -> Result<(), ReconfigureRefused> {
    let live = crate::inferlet::process::live_count();
    if live > 0 {
        return Err(ReconfigureRefused::Busy(live));
    }
    set_frame_size(frame_size);
    set_submit_depth(submit_depth);
    set_dispatch_depth(dispatch_depth);
    Ok(())
}

/// Host-reader channel capacity, in cells, that lets a lane sustain
/// [`configured_submit_depth`] without the ring becoming the bottleneck.
///
/// The trailing `+ 1` is structural, not empirical. A ring sized to exactly the
/// peak occupancy requires the consumer's take to be visible to the producer at
/// the moment it publishes — and that visibility delay IS the host round trip
/// run-ahead exists to hide. Zero margin therefore re-imports the round trip
/// into the critical path: a 3k-1 ring measured 28.0k vs 34.3k tok/s against
/// the same guest at 3k (text-completion-bench).
///
/// At k = 1 an undersized ring is silent: `fire::submit_pass_stamped`
/// short-circuits before `validate_frame`, so every capacity check is k >= 2
/// only and a k = 1 guest serialises with no diagnostic at all.
///
/// Sized for a fully live frame (r = k). A lane the engine forces to one live
/// slot per frame — a recurrent-state model — needs strictly less, so this is
/// a safe bound for every guest.
pub fn channel_capacity() -> usize {
    configured_submit_depth() * configured_frame_size() + 1
}

// =============================================================================
// Public API: spawn/get_stats/shutdown plain scheduler surfaces (no actor)
// =============================================================================

/// Handle returned by [`spawn`]; dropping/`shutdown`ing it stops every
/// per-driver `BatchScheduler` it spawned.
pub struct SchedulerShutdownHandle {
    schedulers: Vec<BatchScheduler>,
}

fn dynamic_schedulers() -> &'static Mutex<HashMap<DriverId, BatchScheduler>> {
    static SCHEDULERS: OnceLock<Mutex<HashMap<DriverId, BatchScheduler>>> = OnceLock::new();
    SCHEDULERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn build_driver_scheduler(
    driver_id: DriverId,
    page_size: u32,
    request_timeout_secs: u64,
) -> Result<BatchScheduler> {
    let limits = crate::driver::get_spec(driver_id)?.scheduler_limits();
    Ok(BatchScheduler::new(
        driver_id,
        driver_id,
        page_size,
        limits,
        request_timeout_secs,
        configured_frame_size(),
    ))
}

pub fn spawn_driver(driver_id: DriverId, page_size: u32, request_timeout_secs: u64) -> Result<()> {
    let mut schedulers = dynamic_schedulers().lock().unwrap();
    if schedulers.contains_key(&driver_id) {
        return Err(anyhow!(
            "driver {driver_id} already has a dynamic scheduler"
        ));
    }
    let scheduler = build_driver_scheduler(driver_id, page_size, request_timeout_secs)?;
    schedulers.insert(driver_id, scheduler);
    Ok(())
}

pub fn stop_driver(driver_id: DriverId) -> Result<()> {
    let scheduler = dynamic_schedulers()
        .lock()
        .unwrap()
        .remove(&driver_id)
        .ok_or_else(|| anyhow!("driver {driver_id} has no dynamic scheduler"))?;
    drop(scheduler);
    Ok(())
}

impl SchedulerShutdownHandle {
    pub async fn shutdown(self) -> Result<()> {
        // `BatchScheduler::drop` joins the worker thread and clears the
        // handle registry; dropping the Vec here shuts every driver down.
        drop(self.schedulers);
        Ok(())
    }
}

/// Spawns one per-driver [`BatchScheduler`] for each of `driver_indices`.
/// Replaces the former `InferenceService` actor: schedulers are plain
/// worker threads registered directly in this module's handle registry, so
/// there is no actor round-trip on the hot submit path.
pub async fn spawn(
    driver_indices: &[usize],
    page_size: u32,
    request_timeout_secs: u64,
) -> Result<SchedulerShutdownHandle> {
    let schedulers: Vec<BatchScheduler> = driver_indices
        .iter()
        .map(|&driver_id| build_driver_scheduler(driver_id, page_size, request_timeout_secs))
        .collect::<Result<_>>()?;

    Ok(SchedulerShutdownHandle { schedulers })
}

fn rs_state_copy_plan(
    src_slots: Vec<u32>,
    dst_slots: Vec<u32>,
) -> Result<Option<crate::driver::StateCopyPlan>> {
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
        .map(
            |(src_slot_id, dst_slot_id)| pie_driver_abi::PieStateCopyRange {
                src_slot_id,
                dst_slot_id,
                src_token_offset: 0,
                dst_token_offset: 0,
                token_count: 0,
            },
        )
        .collect();
    Ok(Some(crate::driver::StateCopyPlan { slot_ranges }))
}

pub fn submit_async(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    last_page_len: u32,
    pipeline_id: Option<ProcessId>,
    completion: crate::driver::WorkItemCompletion,
) -> Result<()> {
    submit_async_with_kv_copy(
        request,
        driver_idx,
        instance_id,
        last_page_len,
        pipeline_id,
        completion,
        Vec::new(),
        Vec::new(),
    )
}

pub(crate) fn nudge(driver_idx: usize) {
    if let Ok(handle) = scheduler_handle(driver_idx) {
        let _ = handle.nudge();
    }
}

#[allow(clippy::too_many_arguments)]
pub fn submit_async_with_kv_copy(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    last_page_len: u32,
    pipeline_id: Option<ProcessId>,
    completion: crate::driver::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
) -> Result<()> {
    let prelaunch_copy = (!copy_src.is_empty()).then_some(crate::driver::KvCopyPlan {
        src_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        src_device_ordinal: 0,
        dst_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        dst_device_ordinal: 0,
        src_page_ids: copy_src,
        dst_page_ids: copy_dst,
        cells: Vec::new(),
    });
    scheduler_handle(driver_idx)?.submit_with_identity_and_copy(
        request,
        instance_id,
        completion,
        last_page_len,
        pipeline_id,
        prelaunch_copy,
        None,
    )
}

pub fn submit_prebuilt_async(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    last_page_len: u32,
    completion: crate::driver::WorkItemCompletion,
) -> Result<()> {
    submit_prebuilt_async_with_kv_copy(
        request,
        driver_idx,
        instance_id,
        last_page_len,
        completion,
        Vec::new(),
        Vec::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn submit_prebuilt_async_with_kv_copy(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    last_page_len: u32,
    completion: crate::driver::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
) -> Result<()> {
    let prelaunch_copy = (!copy_src.is_empty()).then_some(crate::driver::KvCopyPlan {
        src_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        src_device_ordinal: 0,
        dst_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        dst_device_ordinal: 0,
        src_page_ids: copy_src,
        dst_page_ids: copy_dst,
        cells: Vec::new(),
    });
    scheduler_handle(driver_idx)?.submit_prebuilt_with_copy(
        request,
        instance_id,
        completion,
        last_page_len,
        prelaunch_copy,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn submit_prebuilt_async_with_kv_and_rs_copy(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    last_page_len: u32,
    completion: crate::driver::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
    rs_copy_src: Vec<u32>,
    rs_copy_dst: Vec<u32>,
) -> Result<()> {
    let prelaunch_copy = (!copy_src.is_empty()).then_some(crate::driver::KvCopyPlan {
        src_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        src_device_ordinal: 0,
        dst_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        dst_device_ordinal: 0,
        src_page_ids: copy_src,
        dst_page_ids: copy_dst,
        cells: Vec::new(),
    });
    scheduler_handle(driver_idx)?.submit_prebuilt_with_copy(
        request,
        instance_id,
        completion,
        last_page_len,
        prelaunch_copy,
        rs_state_copy_plan(rs_copy_src, rs_copy_dst)?,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn submit_prebuilt_tracked_async_with_kv_and_rs_copy(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    pipeline_id: ProcessId,
    last_page_len: u32,
    completion: crate::driver::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
    rs_copy_src: Vec<u32>,
    rs_copy_dst: Vec<u32>,
) -> Result<()> {
    submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
        &scheduler_handle(driver_idx)?,
        request,
        instance_id,
        pipeline_id,
        pipeline_id,
        last_page_len,
        completion,
        copy_src,
        copy_dst,
        rs_copy_src,
        rs_copy_dst,
        None,
        /*hook_program=*/false,
        /*lora_program=*/false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
    handle: &worker::SchedulerHandle,
    request: crate::driver::LaunchPlan,
    instance_id: u64,
    process_id: ProcessId,
    pipeline_id: ProcessId,
    last_page_len: u32,
    completion: crate::driver::WorkItemCompletion,
    copy_src: Vec<u32>,
    copy_dst: Vec<u32>,
    rs_copy_src: Vec<u32>,
    rs_copy_dst: Vec<u32>,
    frame: Option<FrameStamp>,
    hook_program: bool,
    lora_program: bool,
) -> Result<()> {
    let prelaunch_copy = (!copy_src.is_empty()).then_some(crate::driver::KvCopyPlan {
        src_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        src_device_ordinal: 0,
        dst_domain: pie_driver_abi::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        dst_device_ordinal: 0,
        src_page_ids: copy_src,
        dst_page_ids: copy_dst,
        cells: Vec::new(),
    });
    handle.submit_prebuilt_tracked_with_copy(
        request,
        instance_id,
        completion,
        last_page_len,
        process_id,
        pipeline_id,
        prelaunch_copy,
        rs_state_copy_plan(rs_copy_src, rs_copy_dst)?,
        frame,
        hook_program,
        lora_program,
    )
}

/// Returns aggregated scheduler stats across every registered driver
/// (lock-free, non-blocking — the per-driver `SchedulerStats` are plain
/// atomics, so this needs no actor round-trip).
pub async fn get_stats() -> AggregateStats {
    let scheduler_stats: Vec<Arc<stats::SchedulerStats>> = handle_registry()
        .read()
        .unwrap()
        .iter()
        .filter_map(|slot| slot.as_ref().map(|handle| handle.stats()))
        .collect();
    stats::aggregate(&scheduler_stats)
}

