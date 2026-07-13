//! Per-driver batching: when accumulated fires launch.
//!
//! - [`worker`]: `BatchScheduler` — the per-driver run loop (accumulate,
//!   decide, dispatch, retire). The only public submodule (external crates
//!   construct `worker::BatchScheduler` directly for host-driver test
//!   harnesses); `batch`/`dispatch`/`probe`/`quorum`/`stats`/`wire` are
//!   internal.
//! - `batch`: capacity accounting + the dense-batch accumulator.
//! - `dispatch`: the driver ABI's per-`driver_id` verbs (`register_program`,
//!   `bind_instance`, the `copy_*` family, ...) — re-exported at this
//!   module's root since they call [`scheduler_handle`], which is
//!   scheduler-owned state.
//! - `wire`: owned `LaunchPlan`s -> the batched wire request, page-trim.
//! - `quorum`: the wait-all-active-pipelines fire rule.
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
pub(crate) mod probe;
pub(crate) mod quorum;
pub(crate) mod stats;
pub(crate) mod wire;
pub mod worker;

use std::sync::{Arc, Mutex, OnceLock, RwLock};

use anyhow::{Result, anyhow};

// `copy_d2h`/`copy_h2d`/`copy_h2h`/`copy_rs_d2d`/`resize_pool` round out the
// driver ABI verb surface (see `dispatch`'s module doc for which are wired
// into the current mock-driver fire path vs. reserved/unit-test-only).
#[allow(unused_imports)]
pub(crate) use dispatch::{
    bind_instance, close_instance, copy_d2d, copy_d2h, copy_d2h_tracked, copy_h2d,
    copy_h2d_tracked, copy_h2h, copy_kv_cells, copy_rs_d2d, register_channel, register_program,
    resize_pool,
};
pub use stats::AggregateStats;
pub use worker::BatchScheduler;
use worker::SchedulerHandle;

use crate::driver::DriverId;

/// Process identity the scheduler and quorum rule track (co-batch
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

// =============================================================================
// Fire trace (`PIE_SCHED_TRACE` / `PIE_SCHED_TRACE_FILE`)
// =============================================================================

/// Whether the scheduler fire trace is enabled. Read once (cached, like
/// `quorum::max_in_flight`'s env lever) — MUST be set before the first fire
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
// Demoted-pipeline reaper hook (M-A2)
// =============================================================================

/// `WaitAllPolicy::take_terminate_candidates` reaps a pipeline through the
/// process facade (`inferlet::process::terminate`), which is L4 — above
/// `scheduler` (L2) in the layering, so this module may not import it
/// directly. A plain closure seam instead: `bootstrap` installs it once, at
/// startup, wired to the real facade; a unit-test scheduler with no
/// installed hook just leaves demoted candidates undrained (no reaper, no
/// panic).
type TerminateHook = Box<dyn Fn(ProcessId) + Send + Sync>;
static TERMINATE_HOOK: std::sync::OnceLock<TerminateHook> = std::sync::OnceLock::new();

/// Installs the demoted-pipeline reaper (called once, from `bootstrap`,
/// after [`spawn`]).
pub(crate) fn install_terminate_hook(hook: impl Fn(ProcessId) + Send + Sync + 'static) {
    let _ = TERMINATE_HOOK.set(Box::new(hook));
}

/// Reaps a pipeline `WaitAllPolicy` demoted for missing too many consecutive
/// wave deadlines. No-op until [`install_terminate_hook`] runs.
pub(crate) fn terminate_demoted_pipeline(pid: ProcessId) {
    if let Some(hook) = TERMINATE_HOOK.get() {
        hook(pid);
    }
}

// =============================================================================
// Public API: spawn/get_stats/shutdown plain scheduler surfaces (no actor)
// =============================================================================

/// Handle returned by [`spawn`]; dropping/`shutdown`ing it stops every
/// per-driver `BatchScheduler` it spawned.
pub struct SchedulerShutdownHandle {
    schedulers: Vec<BatchScheduler>,
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
    let driver_ids: Vec<DriverId> = driver_indices.to_vec();
    let mut driver_batch_limits = Vec::with_capacity(driver_indices.len());
    for &driver_idx in driver_indices {
        let info = crate::driver::get_spec(driver_idx)
            .unwrap_or_else(|e| panic!("Failed to get driver info for index {driver_idx}: {e}"));
        driver_batch_limits.push(info.scheduler_limits());
    }

    let schedulers: Vec<BatchScheduler> = driver_ids
        .iter()
        .enumerate()
        .map(|(driver_idx, &driver_id)| {
            let limits = driver_batch_limits[driver_idx];
            BatchScheduler::new(
                driver_id,
                driver_idx,
                page_size,
                limits,
                request_timeout_secs,
            )
        })
        .collect();

    Ok(SchedulerShutdownHandle { schedulers })
}

pub struct PreparedLaunch {
    pub page_refs: Vec<crate::store::kv::project::PhysicalPageId>,
    pub last_page_len: u32,
    pub kv_translation_version: u64,
    pub copy_src: Vec<u32>,
    pub copy_dst: Vec<u32>,
}

pub enum LaunchPreparationError {
    Blocked(String),
    Retry(String),
    Failed(String),
}

pub type LaunchPreparation = Box<
    dyn FnMut(
            &mut crate::driver::LaunchPlan,
        ) -> std::result::Result<PreparedLaunch, LaunchPreparationError>
        + Send,
>;

pub type RetryClassifier = Box<dyn Fn() -> Option<String> + Send + Sync>;

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
    physical_page_ids: Vec<crate::store::kv::project::PhysicalPageId>,
    last_page_len: u32,
    pipeline_id: Option<ProcessId>,
    completion: crate::driver::WorkItemCompletion,
) -> Result<()> {
    submit_async_with_kv_copy(
        request,
        driver_idx,
        instance_id,
        physical_page_ids,
        last_page_len,
        pipeline_id,
        completion,
        Vec::new(),
        Vec::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn submit_async_deferred(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    pipeline_id: Option<ProcessId>,
    preparation_order_key: Option<u64>,
    completion: crate::driver::WorkItemCompletion,
    preparation: LaunchPreparation,
    retry_classifier: Option<RetryClassifier>,
) -> Result<()> {
    scheduler_handle(driver_idx)?.submit_deferred(
        request,
        instance_id,
        completion,
        pipeline_id,
        preparation_order_key,
        None,
        preparation,
        retry_classifier,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn submit_async_deferred_with_rs_copy(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    pipeline_id: Option<ProcessId>,
    preparation_order_key: Option<u64>,
    completion: crate::driver::WorkItemCompletion,
    preparation: LaunchPreparation,
    retry_classifier: Option<RetryClassifier>,
    rs_copy_src: Vec<u32>,
    rs_copy_dst: Vec<u32>,
) -> Result<()> {
    scheduler_handle(driver_idx)?.submit_deferred(
        request,
        instance_id,
        completion,
        pipeline_id,
        preparation_order_key,
        rs_state_copy_plan(rs_copy_src, rs_copy_dst)?,
        preparation,
        retry_classifier,
    )
}

pub(crate) fn nudge(driver_idx: usize) {
    if let Ok(handle) = scheduler_handle(driver_idx) {
        let _ = handle.nudge();
    }
}

pub(crate) fn freeze_pipeline(pid: ProcessId) -> Result<()> {
    let handles: Vec<_> = handle_registry()
        .read()
        .unwrap()
        .iter()
        .flatten()
        .cloned()
        .collect();
    for handle in handles {
        handle.freeze_pipeline(pid)?;
    }
    Ok(())
}

pub(crate) fn resume_pipeline(pid: ProcessId) {
    let handles: Vec<_> = handle_registry()
        .read()
        .unwrap()
        .iter()
        .flatten()
        .cloned()
        .collect();
    for handle in handles {
        let _ = handle.resume_pipeline(pid);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn submit_async_with_kv_copy(
    request: crate::driver::LaunchPlan,
    driver_idx: usize,
    instance_id: u64,
    physical_page_ids: Vec<crate::store::kv::project::PhysicalPageId>,
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
        physical_page_ids,
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
    physical_page_ids: Vec<crate::store::kv::project::PhysicalPageId>,
    last_page_len: u32,
    completion: crate::driver::WorkItemCompletion,
) -> Result<()> {
    submit_prebuilt_async_with_kv_copy(
        request,
        driver_idx,
        instance_id,
        physical_page_ids,
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
    physical_page_ids: Vec<crate::store::kv::project::PhysicalPageId>,
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
        physical_page_ids,
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
    physical_page_ids: Vec<crate::store::kv::project::PhysicalPageId>,
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
        physical_page_ids,
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
    physical_page_ids: Vec<crate::store::kv::project::PhysicalPageId>,
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
    scheduler_handle(driver_idx)?.submit_prebuilt_tracked_with_copy(
        request,
        instance_id,
        completion,
        physical_page_ids,
        last_page_len,
        pipeline_id,
        prelaunch_copy,
        rs_state_copy_plan(rs_copy_src, rs_copy_dst)?,
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
