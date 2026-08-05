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
use std::sync::atomic::{AtomicU64, Ordering};
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

/// How long a lane that is hard-blocking a frame's seal may go without
/// submitting before the engine stops waiting for it
/// (`[model.scheduler] submit_deadline_us`, default 50ms). Guests read it as
/// `model.submit-deadline-us()`.
///
/// Small enough to be a real bound on fleet exposure because it measures a
/// much narrower interval than its size suggests: the clock runs only while
/// the lane is an awaited member with nothing submitted, and is stopped by
/// run-ahead, by an unretired dispatch (the engine owes it a result), by a
/// bind in flight, and by `forward.park()`. Host turnaround has its own
/// headroom in `HOST_TURNAROUND_WAVES`.
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
// Frame dispatch trace (`PIE_SCHED_TRACE` / `PIE_SCHED_TRACE_FILE`)
// =============================================================================

/// Whether the scheduler dispatch trace is enabled. Read once (cached, like
/// `frame::configured_max_in_flight`'s env lever) — MUST be set before the first fire
/// (before boot), since later env mutations are never re-observed. `worker`
/// checks this before doing any per-dispatch trace bookkeeping, so tracing
/// off costs nothing on the hot path.
pub(crate) fn sched_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("PIE_SCHED_TRACE").is_ok_and(|v| v != "0" && !v.is_empty()))
}

/// The optional trace sink (`PIE_SCHED_TRACE_FILE`), opened once in append
/// mode. A real file — unlike `eprintln!`'s fd 2 — survives libtest's
/// stdout/stderr capture-sink for a background scheduler thread, which is
/// why the file form exists alongside the fd-2 form.
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

/// Appends one `[pie-sched-trace] …` line: to stderr (fd 2) always when
/// [`sched_trace_enabled`], and ALSO to `PIE_SCHED_TRACE_FILE` when set,
/// flushed immediately so a polling reader observes it append-only and
/// promptly. Callers should guard any per-dispatch bookkeeping this line
/// needs behind [`sched_trace_enabled`] first, so tracing-off costs nothing
/// beyond that one flag read.
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
    pub retire_instances_ns: AtomicU64,
    pub retire_mark_ns: AtomicU64,
    pub retire_resolve_ns: AtomicU64,
    pub retire_emit_ns: AtomicU64,
    pub retire_drop_ns: AtomicU64,
    pub retire_n: AtomicU64,
    pub post_map_ns: AtomicU64,
    pub post_drain_ns: AtomicU64,
    pub post_filter_ns: AtomicU64,
    pub post_tail_ns: AtomicU64,
    pub post_drain_n: AtomicU64,
    pub disp_frame_ns: AtomicU64,
    pub disp_rot_ns: AtomicU64,
    pub disp_rot_n: AtomicU64,
    pub disp_busy_ns: AtomicU64,
    pub disp_busy_n: AtomicU64,
    pub disp_copy_ns: AtomicU64,
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
    retire_instances_ns: AtomicU64::new(0),
    retire_mark_ns: AtomicU64::new(0),
    retire_resolve_ns: AtomicU64::new(0),
    retire_emit_ns: AtomicU64::new(0),
    retire_drop_ns: AtomicU64::new(0),
    retire_n: AtomicU64::new(0),
    post_map_ns: AtomicU64::new(0),
    post_drain_ns: AtomicU64::new(0),
    post_filter_ns: AtomicU64::new(0),
    post_tail_ns: AtomicU64::new(0),
    post_drain_n: AtomicU64::new(0),
    disp_frame_ns: AtomicU64::new(0),
    disp_rot_ns: AtomicU64::new(0),
    disp_rot_n: AtomicU64::new(0),
    disp_busy_ns: AtomicU64::new(0),
    disp_busy_n: AtomicU64::new(0),
    disp_copy_ns: AtomicU64::new(0),
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
/// Messages to the fire-timing writer thread.
enum FireTimingMsg {
    Line(String),
    Flush(std::sync::mpsc::Sender<()>),
}

/// Formatted records go to a dedicated writer thread instead of straight to
/// `stderr`.
///
/// Writing inline took the process-wide `stderr` lock and issued one `write`
/// syscall PER RECORD. That is harmless at a few hundred records, but the
/// per-process lifecycle stream emits ~28 records per process, so a 512-lane
/// cohort boundary pushed ~14k locked syscalls through a single mutex — tens
/// of milliseconds of serialisation landing exactly on the boundary the
/// stream exists to measure, and paid by the guest, scheduler and driver-lane
/// threads alike. The producer now pays a channel send; ordering is preserved
/// because a single consumer writes them.
fn fire_timing_sink() -> &'static crossbeam::channel::Sender<FireTimingMsg> {
    static SINK: OnceLock<crossbeam::channel::Sender<FireTimingMsg>> = OnceLock::new();
    SINK.get_or_init(|| {
        let (tx, rx) = crossbeam::channel::unbounded::<FireTimingMsg>();
        std::thread::Builder::new()
            .name("pie-fire-timing".into())
            .spawn(move || {
                use std::io::Write;
                let mut out = std::io::BufWriter::with_capacity(1 << 20, std::io::stderr());
                while let Ok(msg) = rx.recv() {
                    match msg {
                        FireTimingMsg::Line(line) => {
                            let _ = out.write_all(line.as_bytes());
                            // Flush only when the producers have gone quiet:
                            // a burst costs one syscall, and nothing is left
                            // sitting in the buffer once the burst ends.
                            if rx.is_empty() {
                                let _ = out.flush();
                            }
                        }
                        FireTimingMsg::Flush(ack) => {
                            let _ = out.flush();
                            let _ = ack.send(());
                        }
                    }
                }
                let _ = out.flush();
            })
            .expect("spawning the fire-timing writer thread");
        tx
    })
}

/// Block until every record queued so far has reached `stderr`. Called on
/// scheduler shutdown so a benchmark that reads the stream after the process
/// exits sees the tail.
pub(crate) fn fire_timing_flush() {
    if !fire_timing_enabled() {
        return;
    }
    let (ack_tx, ack_rx) = std::sync::mpsc::channel();
    if fire_timing_sink()
        .send(FireTimingMsg::Flush(ack_tx))
        .is_ok()
    {
        let _ = ack_rx.recv_timeout(std::time::Duration::from_secs(5));
    }
}

pub(crate) fn fire_timing_write(record: &serde_json::Value) {
    if !fire_timing_enabled() {
        return;
    }
    let _ = fire_timing_sink().send(FireTimingMsg::Line(format!("[pie-fire-timing] {record}\n")));
}

// =============================================================================
// Wall-bucketed CPU census
// =============================================================================
//
// A cohort boundary is a CPU BUDGET problem (CONTENTION_FOLLOWUP §20.26): the
// boundary window demands more cores than the cgroup quota grants, and most of
// the demand was uninstrumented because it is spent INSIDE a guest task, where
// no host timer sees it. Per-event wall timings cannot find it — a task that is
// descheduled looks identical to one that is running.
//
// So census CPU, not wall: each task class accumulates
// `CLOCK_THREAD_CPUTIME_ID` deltas across its own polls into a 10 ms bucket of
// wall time. Summing a boundary's buckets gives the core-ms that landed in it,
// split by class, with no sampling error.

const CPU_CENSUS_BUCKET_US: u64 = 10_000;
const CPU_CENSUS_BUCKETS: usize = 8192;

/// Task classes the census separates. Kept tiny: every poll indexes it.
#[derive(Clone, Copy)]
pub(crate) enum CpuClass {
    /// The guest `main` future: guest WASM plus the host functions it calls.
    Guest = 0,
    Teardown = 1,
    /// The whole per-process task, `Guest` included — the difference is the
    /// bring-up and retirement the guest future does not cover.
    Process = 2,
}
const CPU_CLASSES: usize = 3;

static CPU_CENSUS: [[AtomicU64; CPU_CENSUS_BUCKETS]; CPU_CLASSES] =
    [const { [const { AtomicU64::new(0) }; CPU_CENSUS_BUCKETS] }; CPU_CLASSES];
static CPU_CENSUS_EPOCH_US: AtomicU64 = AtomicU64::new(0);
/// CPU that arrived after the census window closed. The window is finite
/// (`CPU_CENSUS_BUCKETS * CPU_CENSUS_BUCKET_US`, currently 81.9 s) and runs
/// longer than that do exist — the contention `soak` scenario is 124 s. This
/// is reported with the dump so a truncated census is never read as a whole
/// one.
static CPU_CENSUS_DROPPED_NS: [AtomicU64; CPU_CLASSES] = [const { AtomicU64::new(0) }; CPU_CLASSES];

/// `HostShadow::advance` accounting, written only when fire timing is on.
pub(crate) static SHADOW_ADVANCE_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SHADOW_ADVANCE_FOLDS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SHADOW_ADVANCE_NS: AtomicU64 = AtomicU64::new(0);

/// Per-thread CPU time. Unlike `CLOCK_MONOTONIC` this stops while the thread
/// is off-core, so a poll that waits for a core costs the census nothing.
pub(crate) fn thread_cpu_ns() -> u64 {
    let mut value = std::mem::MaybeUninit::<libc::timespec>::uninit();
    let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, value.as_mut_ptr()) };
    if status != 0 {
        return 0;
    }
    let value = unsafe { value.assume_init() };
    (value.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(value.tv_nsec as u64)
}

pub(crate) fn record_task_cpu(class: CpuClass, at_us: u64, cpu_ns: u64) {
    if cpu_ns == 0 {
        return;
    }
    let mut epoch = CPU_CENSUS_EPOCH_US.load(Ordering::Relaxed);
    if epoch == 0 {
        epoch = match CPU_CENSUS_EPOCH_US.compare_exchange(
            0,
            at_us,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => at_us,
            Err(existing) => existing,
        };
    }
    let bucket = (at_us.saturating_sub(epoch) / CPU_CENSUS_BUCKET_US) as usize;
    if bucket < CPU_CENSUS_BUCKETS {
        CPU_CENSUS[class as usize][bucket].fetch_add(cpu_ns, Ordering::Relaxed);
    } else {
        CPU_CENSUS_DROPPED_NS[class as usize].fetch_add(cpu_ns, Ordering::Relaxed);
    }
}

/// A region metered as CPU: it contains no `.await`, so the task holds its
/// thread throughout and the elapsed wall is its CPU there — up to preemption,
/// which inflates every phase together on a contended box (measured: uniformly
/// +10-20% at `/proc/loadavg` 18 against a 13.6-core quota). Read as a share of
/// [`CPU_CENSUS`], not as an absolute.
#[derive(Clone, Copy)]
pub(crate) enum CpuPhase {
    /// The whole host-geometry submit, `Preamble` excluded.
    SubmitTotal = 0,
    /// `drain_settled` + `wire_channels_to_pipeline`, ahead of the submit.
    Preamble,
    BindGeometry,
    Declare,
    Grant,
    /// Handing the built fire to the scheduler.
    Launch,
    /// The device-geometry submit, whole. Zero on host-geometry workloads.
    DeviceGeometrySubmit,
    /// One non-blocking channel poll in `materialize_channel`.
    ChannelPoll,
}

impl CpuPhase {
    const COUNT: usize = 8;
    const NAMES: [&'static str; Self::COUNT] = [
        "submit_total",
        "submit_preamble",
        "submit_bind_geometry",
        "submit_declare",
        "submit_grant",
        "submit_launch",
        "devgeo_submit_total",
        "chan_poll",
    ];
}

/// A region metered as WALL time: it spans `.await` points, so most of what it
/// measures is the task NOT running. Deliberately a separate type and a
/// separate record from [`CpuPhase`]: the two live in different units and a
/// single table invited exactly the mistake of adding them together.
#[derive(Clone, Copy)]
pub(crate) enum WaitPhase {
    /// A whole `materialize_channel`, park included.
    ChannelTake = 0,
    /// The park itself, entered only when the channel is not already ready.
    ChannelProgress,
}

impl WaitPhase {
    const COUNT: usize = 2;
    const NAMES: [&'static str; Self::COUNT] = ["chan_take", "chan_await_progress"];
}

/// Counters behind one phase enum. Both tables share this so the two units
/// cannot drift apart in how they accumulate.
struct PhaseTable<const N: usize> {
    ns: [AtomicU64; N],
    calls: [AtomicU64; N],
}

impl<const N: usize> PhaseTable<N> {
    const fn new() -> Self {
        Self {
            ns: [const { AtomicU64::new(0) }; N],
            calls: [const { AtomicU64::new(0) }; N],
        }
    }

    #[inline]
    fn add(&self, index: usize, started: Option<std::time::Instant>) {
        if let Some(started) = started {
            self.ns[index].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            self.calls[index].fetch_add(1, Ordering::Relaxed);
        }
    }
}

static CPU_PHASES: PhaseTable<{ CpuPhase::COUNT }> = PhaseTable::new();
static WAIT_PHASES: PhaseTable<{ WaitPhase::COUNT }> = PhaseTable::new();

/// Accumulate one CPU-phase sample. `started` is `None` when fire timing is
/// off, which makes every call site a branch and nothing else.
#[inline]
pub(crate) fn cpu_phase_add(phase: CpuPhase, started: Option<std::time::Instant>) {
    CPU_PHASES.add(phase as usize, started);
}

/// Accumulate one wall-phase sample. See [`cpu_phase_add`] for `started`.
#[inline]
pub(crate) fn wait_phase_add(phase: WaitPhase, started: Option<std::time::Instant>) {
    WAIT_PHASES.add(phase as usize, started);
}

/// `Some(Instant::now())` exactly when fire timing is on.
#[inline]
pub(crate) fn phase_start() -> Option<std::time::Instant> {
    fire_timing_enabled().then(std::time::Instant::now)
}

fn dump_phase_table<const N: usize>(event: &str, names: [&'static str; N], table: &PhaseTable<N>) {
    for (index, name) in names.iter().enumerate() {
        // Emitted even at zero calls: an absent row cannot be told apart from
        // a path that was never reached.
        fire_timing_write(&serde_json::json!({
            "schema": 1,
            "source": "runtime",
            "event": event,
            "phase": name,
            "calls": table.calls[index].load(Ordering::Relaxed),
            "ns": table.ns[index].load(Ordering::Relaxed),
        }));
    }
}

/// Emit the census as one record per class. Called on scheduler shutdown, so
/// the arrays are read once and never on a hot path.
pub(crate) fn fire_timing_dump_cpu_census() {
    if !fire_timing_enabled() {
        return;
    }
    fire_timing_write(&serde_json::json!({
        "schema": 1,
        "source": "runtime",
        "event": "shadow_advance",
        "calls": SHADOW_ADVANCE_CALLS.load(Ordering::Relaxed),
        "folds": SHADOW_ADVANCE_FOLDS.load(Ordering::Relaxed),
        "ns": SHADOW_ADVANCE_NS.load(Ordering::Relaxed),
    }));
    dump_phase_table("submit_phase", CpuPhase::NAMES, &CPU_PHASES);
    dump_phase_table("submit_wait", WaitPhase::NAMES, &WAIT_PHASES);
    let epoch = CPU_CENSUS_EPOCH_US.load(Ordering::Relaxed);
    if epoch == 0 {
        return;
    }
    for (index, name) in [
        (CpuClass::Guest as usize, "guest"),
        (CpuClass::Teardown as usize, "teardown"),
        (CpuClass::Process as usize, "process"),
    ] {
        let buckets: Vec<u64> = CPU_CENSUS[index]
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed) / 1_000)
            .collect();
        let dropped_us = CPU_CENSUS_DROPPED_NS[index].load(Ordering::Relaxed) / 1_000;
        let last = buckets.iter().rposition(|&value| value != 0);
        if last.is_none() && dropped_us == 0 {
            continue;
        }
        fire_timing_write(&serde_json::json!({
            "schema": 1,
            "source": "runtime",
            "event": "cpu_census",
            "class": name,
            "epoch_us": epoch,
            "bucket_us": CPU_CENSUS_BUCKET_US,
            "window_us": CPU_CENSUS_BUCKET_US * CPU_CENSUS_BUCKETS as u64,
            // Nonzero means the run outlived the window and this record is a
            // prefix of the truth, not the whole of it.
            "dropped_us": dropped_us,
            "cpu_us": &buckets[..last.map_or(0, |last| last + 1)],
        }));
    }
}

/// Accumulates the CPU its inner future burns, per poll, into the census.
///
/// A guest task's poll runs guest WASM *and* the host functions it calls, so
/// this is exactly "CPU attributable to this process" — the term §20.26 could
/// not see.
pub(crate) struct CpuMetered<F> {
    inner: F,
    class: CpuClass,
}

impl<F> CpuMetered<F> {
    pub(crate) fn new(class: CpuClass, inner: F) -> Self {
        Self { inner, class }
    }
}

impl<F: Future> Future for CpuMetered<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // Structural pinning: `inner` is never moved out, and `Self` is only
        // ever polled through this projection.
        let this = unsafe { self.get_unchecked_mut() };
        let class = this.class;
        let inner = unsafe { std::pin::Pin::new_unchecked(&mut this.inner) };
        let started = thread_cpu_ns();
        let outcome = inner.poll(cx);
        record_task_cpu(
            class,
            fire_timing_now_us(),
            thread_cpu_ns().saturating_sub(started),
        );
        outcome
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
        fire_timing_dump_cpu_census();
        fire_timing_flush();
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
        /*hook_program=*/false,
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
    hook_program: bool,
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
        hook_program,
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

#[cfg(test)]
mod phase_counter_tests {
    use super::{CPU_PHASES, CpuPhase, Ordering, WAIT_PHASES, WaitPhase};

    #[test]
    fn a_phase_writes_the_slot_its_name_is_dumped_from() {
        // The dump pairs `NAMES[i]` with slot `i`, so a variant whose
        // discriminant drifts from its name would silently mislabel a column.
        let index = CpuPhase::ChannelPoll as usize;
        assert_eq!(CpuPhase::NAMES[index], "chan_poll");
        CPU_PHASES.calls[index].store(0, Ordering::Relaxed);
        CPU_PHASES.add(index, Some(std::time::Instant::now()));
        assert_eq!(CPU_PHASES.calls[index].load(Ordering::Relaxed), 1);

        let index = WaitPhase::ChannelProgress as usize;
        assert_eq!(WaitPhase::NAMES[index], "chan_await_progress");
        WAIT_PHASES.calls[index].store(0, Ordering::Relaxed);
        WAIT_PHASES.add(index, Some(std::time::Instant::now()));
        assert_eq!(WAIT_PHASES.calls[index].load(Ordering::Relaxed), 1);
    }

    #[test]
    fn the_last_variant_of_each_table_is_in_range() {
        assert!((CpuPhase::ChannelPoll as usize) < CpuPhase::COUNT);
        assert!((WaitPhase::ChannelProgress as usize) < WaitPhase::COUNT);
    }
}
