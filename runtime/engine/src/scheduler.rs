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
pub(crate) mod dispatch;
pub(crate) mod frame;
pub(crate) mod probe;
pub(crate) mod stats;
pub(crate) mod wire;
pub mod worker;

pub use frame::FrameStamp;

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

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
// Frame size (`PIE_FRAME_SIZE`) — the Vesuvius deployment constant k
// =============================================================================

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
/// is the setting (CONTENTION_FOLLOWUP §20.8). Set `PIE_FRAME_SIZE=1` to
/// restore the per-wave path.
pub fn configured_frame_size() -> usize {
    static CONFIGURED: OnceLock<usize> = OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        std::env::var("PIE_FRAME_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 64)
    })
}

// =============================================================================
// Guest run-ahead sizing (`PIE_TURNAROUND_WAVES`)
// =============================================================================

/// How many waves of work a lane must keep submitted to cover one host
/// resubmit turnaround: the guest takes a result, does its host-side work, and
/// submits again, and the device must have something to run for that whole
/// interval or the pipeline collapses to lockstep.
///
/// This is a property of the HOST round trip, so it is counted in waves and is
/// independent of k. Sizing it in frames instead is the unit error that
/// collapsed k = 1 throughput (CONTENTION_FOLLOWUP §20.11): a frame-counted
/// window shrinks in real work as k shrinks, exactly when each frame covers
/// less time.
///
/// Fixed at 3 for now. The value is a candidate for adaptation from observed
/// turnaround, which is why guests read it through `model.channel-capacity()`
/// rather than baking a constant — unlike `frame-size`, this MAY change.
const HOST_TURNAROUND_WAVES: usize = 3;

fn turnaround_waves() -> usize {
    static CONFIGURED: OnceLock<usize> = OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        std::env::var("PIE_TURNAROUND_WAVES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(HOST_TURNAROUND_WAVES)
            .clamp(1, 64)
    })
}

/// Frames a lane should keep outstanding: one for the frame currently running,
/// plus enough to cover the host turnaround. `ceil` because a partial frame
/// still costs a whole frame of submission.
pub fn frames_in_flight() -> usize {
    let k = configured_frame_size();
    1 + turnaround_waves().div_ceil(k)
}

/// Host-reader channel capacity, in cells, that lets a lane sustain
/// `frames_in_flight()` without the ring becoming the bottleneck.
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
    frames_in_flight() * configured_frame_size() + 1
}

// =============================================================================
// Fire trace (`PIE_SCHED_TRACE` / `PIE_SCHED_TRACE_FILE`)
// =============================================================================

/// Whether the scheduler fire trace is enabled. Read once (cached, like
/// `frame::configured_max_in_flight`'s env lever) — MUST be set before the first fire
/// (before boot), since later env mutations are never re-observed. `worker`
/// checks this before doing any per-fire trace bookkeeping (e.g. the
/// distinct-program count), so tracing off costs nothing on the hot path.
pub(crate) fn sched_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("PIE_SCHED_TRACE").is_ok_and(|v| v != "0" && !v.is_empty()))
}

/// The optional trace sink (`PIE_SCHED_TRACE_FILE`), opened once in append
/// mode. A real file — unlike `eprintln!`'s fd 2 — survives libtest's
/// stdout/stderr capture-sink for a background scheduler thread (see
/// `cuda_grammar10.rs`'s `StderrCapture` doc for why the file form exists
/// alongside the fd-2 form `cuda_grammar_r2.rs` captures via `dup2`).
fn sched_trace_file() -> Option<&'static Mutex<std::fs::File>> {
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    FILE.get_or_init(|| {
        let path = std::env::var_os("PIE_SCHED_TRACE_FILE")?;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
            .map(Mutex::new)
    })
    .as_ref()
}

/// Appends one `[pie-sched-trace] …` fire line: to stderr (fd 2 — the
/// `cuda_grammar_r2` capture) always when [`sched_trace_enabled`], and ALSO
/// to `PIE_SCHED_TRACE_FILE` when set (the `cuda_grammar10` capture),
/// flushed immediately so a polling reader observes it append-only and
/// promptly. Callers should guard any per-fire bookkeeping this line needs
/// (e.g. a distinct-program count) behind [`sched_trace_enabled`] first, so
/// tracing-off costs nothing beyond that one flag read.
pub(crate) fn sched_trace_write(args: std::fmt::Arguments) {
    if !sched_trace_enabled() {
        return;
    }
    eprintln!("[pie-sched-trace] {args}");
    if let Some(file) = sched_trace_file() {
        use std::io::Write;
        let mut file = file.lock().unwrap();
        let _ = writeln!(file, "[pie-sched-trace] {args}");
        let _ = file.flush();
    }
}

// =============================================================================
// Structured fire timing (`PIE_FIRE_TIMING`)
// =============================================================================

/// Whether correlated per-wave timing is enabled. Unlike the cumulative
/// `profile-fire` feature, this is a diagnostic stream intended for short,
/// attribution-focused benchmark captures.
pub(crate) fn fire_timing_full() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PIE_FIRE_TIMING").is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

/// `PIE_FIRE_TIMING=waves` keeps the per-wave and per-pass records but drops
/// the per-fire ones. The per-fire stream emits one JSON line per request from
/// inside `retire_ready_launches`, which at 128-request waves costs a
/// millisecond or two of the very pass it is measuring — enough to make the
/// scheduler look like the bottleneck it is being used to find.
pub(crate) fn fire_timing_per_fire() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !std::env::var("PIE_FIRE_TIMING").is_ok_and(|value| value == "waves"))
}

/// Worker-loop phase accumulators, in nanoseconds, summed across every pass
/// since the last wave dispatch and drained into that wave's record. The
/// per-wave `settled -> next dispatch` gap is the throughput lever at high
/// concurrency, and these say whether it is the scheduler thread burning CPU
/// (mailbox/retire/dispatch) or parked waiting on the guest lanes.
pub(crate) struct LoopPhaseAcc {
    pub mailbox_ns: AtomicU64,
    pub retire_ns: AtomicU64,
    pub dispatch_ns: AtomicU64,
    pub park_ns: AtomicU64,
    pub passes: AtomicU64,
    pub mailbox_items: AtomicU64,
    pub scan_ns: AtomicU64,
    pub plan_ns: AtomicU64,
    pub post_ns: AtomicU64,
    pub scans: AtomicU64,
    pub lag_ns: AtomicU64,
    pub lag_max_ns: AtomicU64,
    pub lag_n: AtomicU64,
    pub pass_max_ns: AtomicU64,
}

/// Guest-side turnaround probe: how long an inferlet lane takes between
/// being woken with its sampled token and submitting the successor fire.
/// `wake` = driver wake -> `take` returned a value; `work` = that return ->
/// `forward.submit`. Aggregated (not per-fire) to stay off the critical path.
pub(crate) struct GuestPhaseAcc {
    pub wake_woken: AtomicU64,
    pub wake_empty: AtomicU64,
    pub resume_ns: AtomicU64,
    pub resume_max_ns: AtomicU64,
    pub resume_n: AtomicU64,
    pub wake_ns: AtomicU64,
    pub work_ns: AtomicU64,
    pub work_max_ns: AtomicU64,
    pub n: AtomicU64,
}

/// `fire_timing_now_us` of the most recent scheduler retire resolve, used to
/// measure how long a woken lane takes to actually resume on the runtime.
pub(crate) static LAST_RESOLVE_US: AtomicU64 = AtomicU64::new(0);

pub(crate) static GUEST_PHASES: GuestPhaseAcc = GuestPhaseAcc {
    wake_woken: AtomicU64::new(0),
    wake_empty: AtomicU64::new(0),
    resume_ns: AtomicU64::new(0),
    resume_max_ns: AtomicU64::new(0),
    resume_n: AtomicU64::new(0),
    wake_ns: AtomicU64::new(0),
    work_ns: AtomicU64::new(0),
    work_max_ns: AtomicU64::new(0),
    n: AtomicU64::new(0),
};

pub(crate) static LOOP_PHASES: LoopPhaseAcc = LoopPhaseAcc {
    mailbox_ns: AtomicU64::new(0),
    retire_ns: AtomicU64::new(0),
    dispatch_ns: AtomicU64::new(0),
    park_ns: AtomicU64::new(0),
    passes: AtomicU64::new(0),
    mailbox_items: AtomicU64::new(0),
    scan_ns: AtomicU64::new(0),
    plan_ns: AtomicU64::new(0),
    post_ns: AtomicU64::new(0),
    scans: AtomicU64::new(0),
    lag_ns: AtomicU64::new(0),
    lag_max_ns: AtomicU64::new(0),
    lag_n: AtomicU64::new(0),
    pass_max_ns: AtomicU64::new(0),
};

pub(crate) fn ledger_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PIE_LEDGER_TIMING").is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

pub(crate) fn fire_timing_enabled() -> bool {
    fire_timing_full() || ledger_timing_enabled()
}

/// Compatibility claim for submission APIs that do not carry a process
/// context. The production per-token path passes a process-local claim through
/// the `_on` APIs and never touches this set.
pub(crate) fn fire_timing_request_enabled(
    pipeline_id: Option<crate::inferlet::process::ProcessId>,
) -> bool {
    if fire_timing_full() {
        return true;
    }
    if !ledger_timing_enabled() {
        return false;
    }
    static CLAIMED: OnceLock<Mutex<std::collections::HashSet<uuid::Uuid>>> = OnceLock::new();
    pipeline_id.is_some_and(|pipeline_id| {
        CLAIMED
            .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
            .lock()
            .unwrap()
            .insert(pipeline_id)
    })
}

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

pub(crate) fn fire_timing_unix_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX)
}

/// Emit one NDJSON-compatible timing record. CUDA emits the same prefix, so a
/// benchmark log can be split and correlated without a second transport.
pub(crate) fn fire_timing_write(record: &serde_json::Value) {
    if !fire_timing_enabled() {
        return;
    }
    use std::io::Write;
    let line = format!("[pie-fire-timing] {record}\n");
    let _ = std::io::stderr().lock().write(line.as_bytes());
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
        fire_timing_request_enabled(pipeline_id),
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
        fire_timing_request_enabled(Some(pipeline_id)),
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
    timing_enabled: bool,
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
        timing_enabled,
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
