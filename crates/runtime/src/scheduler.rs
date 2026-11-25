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
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Result, anyhow};

pub(crate) use dispatch::{
    TierKind, TierMove, close_channels, copy_tier_tracked, register_channels_bind_classified,
};
pub use stats::{AggregateStats, HostSubmitStats};
pub use worker::BatchScheduler;
pub(crate) use lane::open_behind;
use worker::SchedulerHandle;

use crate::engine::EngineId;

pub type ProcessId = uuid::Uuid;

#[derive(Clone)]
pub(crate) struct ControlCompletion {
    inner: Arc<ControlCompletionState>,
}

struct ControlCompletionState {
    result: Mutex<Option<std::result::Result<(), (String, Refused)>>>,
    notify: tokio::sync::Notify,
}

/// An engine refused a control (a copy) before reading or writing any of
/// it: its submission was impossible as stated (`engine::Error::Impossible`).
/// What it would have moved lies where it lay.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct RefusedUntouched(pub String);

/// A copy down a tier failed writing to that tier because it is full
/// (`engine::Error::TierFull`): nothing on the device changed, so it is
/// refused untouched too, and the waiter takes the move back by name.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct TierBelowFull(pub String);

/// A copy the engine answered as still landing did not land
/// ([`::engine::Landing::Failed`]): its destination is not to be trusted
/// and its sources hold what they held, so the move is taken back from
/// them, never committed.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct LandingFailed(pub String);

/// How an engine refused a control, read from its error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Refused {
    /// It failed: what it moved may be partly written.
    Failed,
    /// Refused before reading or writing any of it.
    Untouched,
    /// The tier it copied down to is full; nothing on the device changed.
    TierFull,
    /// It did not land: its sources hold what they held.
    LandingFailed,
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
                return result.map_err(Self::error);
            }
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = self.inner.result.lock().unwrap().clone() {
                return result.map_err(Self::error);
            }
            notified.await;
        }
    }

    fn error((message, refused): (String, Refused)) -> anyhow::Error {
        match refused {
            Refused::Untouched => anyhow::Error::new(RefusedUntouched(message)),
            Refused::TierFull => anyhow::Error::new(TierBelowFull(message)),
            Refused::LandingFailed => anyhow::Error::new(LandingFailed(message)),
            Refused::Failed => anyhow::Error::msg(message),
        }
    }

    fn resolve(&self, result: &Result<()>) {
        let result = result.as_ref().map(|_| ()).map_err(|error| {
            let refused = error
                .chain()
                .find_map(|cause| {
                    if cause
                        .downcast_ref::<crate::engine::completion::CopyLandingFailed>()
                        .is_some()
                    {
                        return Some(Refused::LandingFailed);
                    }
                    match cause.downcast_ref::<::engine::Error>() {
                        Some(::engine::Error::TierFull(_)) => Some(Refused::TierFull),
                        Some(
                            ::engine::Error::Impossible(_) | ::engine::Error::Placement { .. },
                        ) => Some(Refused::Untouched),
                        _ => None,
                    }
                })
                .unwrap_or(Refused::Failed);
            (format!("{error:#}"), refused)
        });
        *self.inner.result.lock().unwrap() = Some(result);
        self.inner.notify.notify_waiters();
    }
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

// A test's own scheduler for an engine, on its thread alone
// ([`worker::test_lane::TestLane`]): the copies and pool requests its
// planner and stores submit reach the test, never a lane of another test.
#[cfg(test)]
thread_local! {
    static TEST_HANDLES: std::cell::RefCell<Vec<(usize, SchedulerHandle)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Puts `handle` (or, `None`, takes it off) as `engine_id`'s scheduler on
/// this thread only.
#[cfg(test)]
pub(crate) fn set_test_handle(engine_id: usize, handle: Option<SchedulerHandle>) {
    TEST_HANDLES.with(|handles| {
        let mut handles = handles.borrow_mut();
        handles.retain(|(engine, _)| *engine != engine_id);
        if let Some(handle) = handle {
            handles.push((engine_id, handle));
        }
    });
}

pub(crate) fn scheduler_handle(engine_id: usize) -> Result<SchedulerHandle> {
    #[cfg(test)]
    if let Some(handle) = TEST_HANDLES.with(|handles| {
        handles
            .borrow()
            .iter()
            .find(|(engine, _)| *engine == engine_id)
            .map(|(_, handle)| handle.clone())
    }) {
        return Ok(handle);
    }
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

/// Pages on `tier` copied into host slots `slots`, and those slots' bytes.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) async fn read_tier_kv(
    engine_id: usize,
    tier: ::engine::MemoryDomain,
    pages: Vec<u32>,
    slots: Vec<u32>,
) -> Result<Vec<u8>> {
    let copy = ::engine::KvCopy {
        src: tier,
        dst: ::engine::MemoryDomain::HostPinned,
        src_page_ids: pages,
        dst_page_ids: slots.clone(),
        moves: Vec::new(),
        windowed: Vec::new(),
    };
    scheduler_handle(engine_id)?
        .host_kv(worker::HostKvOp::CopyRead(copy, slots))
        .await
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) async fn read_host_kv(engine_id: usize, slots: Vec<u32>) -> Result<Vec<u8>> {
    scheduler_handle(engine_id)?
        .host_kv(worker::HostKvOp::Read(slots))
        .await
}

/// The bytes of host tier slots, shape by shape, in order (CONTRACT-BC 22).
#[cfg_attr(target_arch = "wasm32", allow(dead_code, reason = "the browser has no host or disk tier"))]
pub(crate) async fn read_tier_slots(
    engine_id: usize,
    slots: Vec<(::engine::chunk::ShapeId, Vec<u32>)>,
) -> Result<Vec<u8>> {
    scheduler_handle(engine_id)?
        .host_kv(worker::HostKvOp::ReadTier(slots))
        .await
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code, reason = "the browser has no host or disk tier"))]
pub(crate) async fn write_tier_slots(
    engine_id: usize,
    slots: Vec<(::engine::chunk::ShapeId, Vec<u32>)>,
    bytes: Vec<u8>,
) -> Result<()> {
    scheduler_handle(engine_id)?
        .host_kv(worker::HostKvOp::WriteTier(slots, bytes))
        .await
        .map(drop)
}

pub(crate) async fn write_host_kv(engine_id: usize, slots: Vec<u32>, bytes: Vec<u8>) -> Result<()> {
    scheduler_handle(engine_id)?
        .host_kv(worker::HostKvOp::Write(slots, bytes))
        .await
        .map(drop)
}

pub(crate) fn schedulers() -> Vec<SchedulerHandle> {
    handle_registry()
        .read()
        .unwrap()
        .iter()
        .flatten()
        .cloned()
        .collect()
}

pub(crate) async fn reload_weights(
    scheduler: &SchedulerHandle,
    path: &std::path::Path,
) -> Result<::engine::Reloaded> {
    let answer = scheduler
        .host_kv(worker::HostKvOp::ReloadWeights(path.to_path_buf()))
        .await?;
    Ok(serde_json::from_slice(&answer)?)
}

pub(crate) async fn pool(engine_id: usize, op: worker::PoolOp) -> Result<()> {
    scheduler_handle(engine_id)?.pool(op).await
}

/// Waits until every copy engine `engine_id` was handed has landed: what an
/// aborted move waits for before it gives back slots a copy may still write.
pub(crate) async fn settle_copies(engine_id: usize) -> Result<()> {
    pool(engine_id, worker::PoolOp::SettleCopies).await
}

/// Gives back an aborted move's slots once every copy it handed the engine
/// has landed: its copies' completions (which may settle before a copy has
/// written its destinations), then `settled`, every copy landed. A copy that
/// did not settle keeps the slots out (`false`): reused under a late write,
/// they would hold another owner's bytes.
pub(crate) async fn abort_once_landed(
    in_flight: Vec<ControlCompletion>,
    settled: impl std::future::Future<Output = anyhow::Result<()>>,
    abort: impl FnOnce(),
) -> std::result::Result<(), String> {
    for completion in in_flight {
        let _ = completion.wait().await;
    }
    settled.await.map_err(|error| format!("{error:#}"))?;
    abort();
    Ok(())
}

/// Tells engine `engine_id` where moved pool slots lie now, on its lane
/// ahead of every fire queued after it.
pub(crate) async fn rebind_slots(engine_id: usize, rebind: ::engine::Rebind) -> Result<()> {
    pool(engine_id, worker::PoolOp::Rebind(rebind)).await
}

static LAYOUTS: std::sync::Mutex<Vec<Option<Option<state::PageLayout>>>> =
    std::sync::Mutex::new(Vec::new());

/// The engine's host-tier layout, once asked.
pub(crate) async fn host_kv_layout(engine_id: usize) -> Option<state::PageLayout> {
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

type Known = Option<Option<::engine::TapGroups>>;
static GROUPS: std::sync::Mutex<Vec<Known>> = std::sync::Mutex::new(Vec::new());

/// The groups the engine reads a tapped page's planes out in, as spans of a
/// host slot; `None` where it taps nothing. Once asked, known.
pub(crate) async fn tap_groups(engine_id: usize) -> Result<Option<::engine::TapGroups>> {
    if let Some(Some(known)) = GROUPS.lock().unwrap().get(engine_id) {
        return Ok(known.clone());
    }
    let bytes = scheduler_handle(engine_id)?
        .host_kv(worker::HostKvOp::TapGroups)
        .await
        .map_err(|error| anyhow!("engine {engine_id}'s tap groups: {error}"))?;
    let groups: Option<::engine::TapGroups> = serde_json::from_slice(&bytes)
        .map_err(|error| anyhow!("engine {engine_id}'s tap groups did not parse: {error}"))?;
    let mut known = GROUPS.lock().unwrap();
    if known.len() <= engine_id {
        known.resize(engine_id + 1, None);
    }
    known[engine_id] = Some(groups.clone());
    Ok(groups)
}

/// Forgets the boot's schedulers and what their engines answered, for
/// [`crate::bootstrap::reset`].
pub(crate) fn reset() {
    handle_registry().write().unwrap().clear();
    LAYOUTS.lock().unwrap().clear();
    GROUPS.lock().unwrap().clear();
    FRAME_SIZE.store(0, Ordering::Relaxed);
    HOST_COVER.store(0, Ordering::Relaxed);
    FRAME_CEILING.store(0, Ordering::Relaxed);
    frame::reset();
    probe::host_submit().reset();
    lane::reset();
    worker::BACKSTOP_RETIREMENTS.store(0, Ordering::Relaxed);
}

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
    ::engine::runahead::Runahead::unstaged(
        u8::try_from(configured_dispatch_depth()).unwrap_or(u8::MAX),
    )
    .submit_depth(host_cover())
}

/// The frames a lane's guest turnaround hides behind past the ones in
/// flight, as measured at load: the turnaround (a frame's landing to the
/// lane's next submit) over one frame's device time, rounded up, on a warm
/// run of the standard sampler before ready (`bootstrap`). The frame seal
/// waits for every lane, so too little cover idles the device: at one frame
/// of cover, 256 requests of 1024 in / 256 out on Gemma-4-31B held 378 of
/// 525 seals with the device idle; at three, 17. 0 until measured (the
/// prewarm's registrations compile the same code at any capacity, and the
/// warm run itself runs at depth 1).
pub fn host_cover() -> usize {
    HOST_COVER.load(Ordering::Relaxed)
}

/// Sets the measured cover (bootstrap, before ready).
pub fn set_host_cover(cover: usize) {
    HOST_COVER.store(cover, Ordering::Relaxed);
}

static HOST_COVER: AtomicUsize = AtomicUsize::new(0);

pub fn set_dispatch_depth(depth: usize) {
    frame::set_dispatch_depth(depth);
}

pub fn configured_dispatch_depth() -> usize {
    frame::configured_dispatch_depth()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconfigureRefused {
    Busy(usize),
    /// A frame size past the load's (its run-ahead window was laid for it).
    PastCeiling {
        asked: usize,
        ceiling: usize,
    },
}

impl std::fmt::Display for ReconfigureRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy(n) => write!(
                f,
                "{n} process(es) still live; these knobs can only change while \
                 the runtime is idle"
            ),
            Self::PastCeiling { asked, ceiling } => write!(
                f,
                "a frame size of {asked} is past the load's {ceiling}, which its \
                 run-ahead window was laid for; load at the larger frame size"
            ),
        }
    }
}

impl std::error::Error for ReconfigureRefused {}

/// Sets the frame size while the runtime is idle. The dispatch depth is
/// the engines' stated frames in flight
/// ([`::engine::Capabilities::frames_in_flight`]) and changes only with a
/// load: a sweep over it reloads the engine at each depth.
pub fn reconfigure(frame_size: usize) -> Result<(), ReconfigureRefused> {
    let live = crate::inferlet::process::live_count();
    if live > 0 {
        return Err(ReconfigureRefused::Busy(live));
    }
    let ceiling = frame_ceiling();
    if frame_size > ceiling {
        return Err(ReconfigureRefused::PastCeiling {
            asked: frame_size,
            ceiling,
        });
    }
    set_frame_size(frame_size);
    Ok(())
}

/// The load's largest frame size: the frame size it boots at, which its
/// run-ahead window is laid for; a reconfigure stays at or under it.
pub fn frame_ceiling() -> usize {
    match FRAME_CEILING.load(Ordering::Relaxed) {
        0 => configured_frame_size(),
        k => k,
    }
}

/// Sets the load's largest frame size (bootstrap, before any program).
pub fn set_frame_ceiling(frame_size: usize) {
    FRAME_CEILING.store(frame_size, Ordering::Relaxed);
}

static FRAME_CEILING: AtomicUsize = AtomicUsize::new(0);

pub fn channel_capacity() -> usize {
    ::engine::runahead::Runahead::unstaged(
        u8::try_from(configured_dispatch_depth()).unwrap_or(u8::MAX),
    )
    .channel_capacity(host_cover(), configured_frame_size())
}

pub fn run_ahead_window() -> usize {
    ::engine::runahead::Runahead::unstaged(
        u8::try_from(configured_dispatch_depth()).unwrap_or(u8::MAX),
    )
    .submit_depth(host_cover())
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
    // A test's own lane names its device ([`worker::test_lane::TestLane`]).
    #[cfg(test)]
    if let Some(domain) = TEST_HANDLES.with(|handles| {
        handles
            .borrow()
            .iter()
            .find(|(engine, _)| *engine == engine_idx)
            .map(|(_, handle)| handle.device_domain())
    }) {
        return domain;
    }
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

/// The times the engine lanes slept on their queues (nothing queued, nothing
/// of their own owed) and stayed awake for work owed them (a landing, or the
/// launch their scheduler posts from fires they settled), every lane summed.
pub fn lane_waits() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (lane::LANE_SLEEPS.load(Relaxed), lane::LANE_AWAITS.load(Relaxed))
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

#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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
        assert!(!matches!(domain, ::engine::MemoryDomain::Device { .. }));
    }
}

#[cfg(test)]
mod abort_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// An aborted move's copy has settled (its completion resolved) but not
    /// landed: its slots go back only once the engine says every copy has
    /// landed, never before, and never at all if that fails.
    #[tokio::test(flavor = "current_thread")]
    async fn an_aborted_move_gives_back_its_slots_only_once_its_copies_land() {
        let copy = ControlCompletion::new();
        copy.resolve(&Ok(()));
        let (landed, landing) = tokio::sync::oneshot::channel::<anyhow::Result<()>>();
        let freed = Arc::new(AtomicBool::new(false));
        let gave = Arc::clone(&freed);
        let abort = tokio::spawn(abort_once_landed(
            vec![copy],
            async move { landing.await.expect("the test settles it") },
            move || gave.store(true, Ordering::SeqCst),
        ));
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert!(
            !freed.load(Ordering::SeqCst),
            "a copy that settled but has not landed keeps its slots out"
        );
        landed.send(Ok(())).expect("the abort waits on it");
        abort.await.expect("the abort runs").expect("it settled");
        assert!(freed.load(Ordering::SeqCst), "landed, the slots go back");

        let copy = ControlCompletion::new();
        copy.resolve(&Ok(()));
        let freed = Arc::new(AtomicBool::new(false));
        let gave = Arc::clone(&freed);
        let refused = abort_once_landed(
            vec![copy],
            async { Err(anyhow::anyhow!("the engine could not settle")) },
            move || gave.store(true, Ordering::SeqCst),
        )
        .await;
        assert!(refused.is_err());
        assert!(
            !freed.load(Ordering::SeqCst),
            "copies that did not settle keep their slots out"
        );
    }
}

#[cfg(test)]
mod refused_untouched_tests {
    use super::*;

    /// An engine refusing a copy as impossible (before reading or writing
    /// any of it) is told apart from a copy that failed: the waiter can
    /// take the move back rather than fail the process.
    #[tokio::test(flavor = "current_thread")]
    async fn a_copy_refused_as_impossible_is_told_untouched() {
        let refused = ControlCompletion::new();
        refused.resolve(&Err(anyhow::Error::new(::engine::Error::Impossible(
            "state chunk 1936 lies inside the carve".into(),
        ))
        .context("kv to LocalDisk copy")));
        let error = refused.wait().await.expect_err("refused");
        assert!(error.downcast_ref::<RefusedUntouched>().is_some(), "{error:#}");
        let failed = ControlCompletion::new();
        failed.resolve(&Err(anyhow::anyhow!("device lost")));
        let error = failed.wait().await.expect_err("failed");
        assert!(error.downcast_ref::<RefusedUntouched>().is_none(), "{error:#}");
    }

    /// A copy down to a full disk tier (the slot file's write failing
    /// with `StorageFull`) is told as the tier below full, nothing on the
    /// device changed: never a failed copy, never a string.
    #[tokio::test(flavor = "current_thread")]
    async fn a_copy_to_a_full_tier_is_told_as_the_tier_below_full() {
        let full = ControlCompletion::new();
        full.resolve(&Err(anyhow::Error::new(::engine::Error::TierFull(
            "the disk tier's slot file is full: No space left on device".into(),
        ))
        .context("kv to LocalDisk copy")));
        let error = full.wait().await.expect_err("refused");
        assert!(error.downcast_ref::<TierBelowFull>().is_some(), "{error:#}");
        assert!(error.downcast_ref::<RefusedUntouched>().is_none(), "{error:#}");
    }
}
