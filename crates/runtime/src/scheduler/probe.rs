//! Per-fire scheduler lifecycle probes.
//!
//! Gated by the `profile-fire` Cargo feature: with it off, `probe_fire!`
//! expands to a body-only no-op, while the holder structs and atomics stay
//! defined so callers still compile. The two timestamp fields are always-on.
//!
//! **Invariant**: `execute.total_us` equals `batch_build_us + engine_fire_us`
//! within probe overhead. `inter_fire_us` and `post_dispatch_to_fire_us` are
//! SIBLINGS measuring gaps *between* fires, so summing them with children of
//! `execute` double-counts.

use std::sync::atomic::AtomicU64;

/// Every probe group for one fire, plus the two always-on timestamps.
#[derive(Debug, Default)]
pub struct FireProbes {
    /// Time between consecutive fire starts, via `swap` on
    /// `last_fire_spawn_micros`. **Sibling** of `execute.*`: includes
    /// `execute.total_us` plus the gap before the next fire.
    pub inter_fire_us: AtomicU64,

    /// Retirement of fire N → start of fire N+1: the rendezvous gap
    /// (chain-extender wake, main-loop drain, cohort fill). **Sibling** of
    /// `execute.*`.
    pub post_dispatch_to_fire_us: AtomicU64,

    /// Steady-state scheduler idle-wait: time blocked in the `while
    /// batch.is_empty()` `recv()`, recorded only once warm. It spans
    /// dispatch→wake→sample→resubmit→SERVICE→recv, everything OUTSIDE the
    /// scheduler's own build/decide.
    pub recv_block_wait_us: AtomicU64,

    /// Micros from `sched_epoch` of the most recent fire start; `swap`ped to
    /// derive `inter_fire_us`. Cheap, so always-on.
    pub last_fire_spawn_micros: AtomicU64,

    /// Micros from `sched_epoch` of the most recent retirement; drives
    /// `post_dispatch_to_fire_us`. Cheap, so always-on.
    pub last_dispatch_end_micros: AtomicU64,

    pub accumulate: AccumulateProbes,
    pub pre_dispatch: PreDispatchProbes,
    pub execute: ExecuteProbes,
    pub post_dispatch: PostDispatchProbes,
    pub quorum: QuorumProbes,
}

/// The quorum fire rule's own counters (F1): device-side bubble, quorum
/// latency, and the two reasons a fire went out narrow.
#[derive(Debug, Default)]
pub struct QuorumProbes {
    /// DEVICE-side idle between one batch retiring and the next launching —
    /// the bubble the quorum rule drives to zero (F1), bounded at p50 < 100 µs
    /// by the M1/M3 gate. Distinct from host-side `inter_fire_us`.
    pub inter_batch_bubble_us: AtomicU64,

    /// Last counted pipeline structurally ready → dense batch enqueued (F1).
    /// In steady state this completes mid-flight, so the value is the slack
    /// before the in-flight batch retires.
    pub quorum_latency_us: AtomicU64,

    /// Idle-escape fires (F2): device idle with an empty queue, so the ready
    /// subset fired at once. Dominant on agentic fleets, near-zero on
    /// saturated decode fleets.
    pub escape_fires: AtomicU64,

    /// Depth-2 submit-ahead fires (G3 bubble): a batch in flight, below the
    /// cap, partial cohort, so the ready subset fired eagerly behind it. The
    /// steady-state decode-fleet bubble-filler; zero under pure quorum.
    pub submit_ahead_fires: AtomicU64,

    /// Legacy counter retained in telemetry; strict wait-all never fires narrow.
    pub straggler_fires: AtomicU64,
    /// Legacy counter retained in telemetry; strict wait-all never demotes.
    pub straggler_demotions: AtomicU64,

    /// Readiness misses: a pass launched structurally ready (F5) whose late
    /// host edge (grammar mask) missed its consuming stage's device cut point,
    /// so the sample dummy-ran and the stage resubmits. M3 gate: rate < 1%.
    pub readiness_miss: AtomicU64,

    /// Wait-for-all wave diagnostics, sampled at each WaitAll fire.
    /// `wave_active_sum / wave_fires` discriminates a PERSISTENT wait-set
    /// (converges to fleet width ⇒ dense waves) from a TRANSIENT one (≈1).
    pub wave_active_sum: AtomicU64,
    pub wave_missing_sum: AtomicU64,
    pub wave_fires: AtomicU64,

    /// Chain engagement: sealed partitions, and the subset sealed while a
    /// frame was still on the device. Their ratio says whether the fleet is
    /// PIPELINED — at 1.0 the next boundary is assembled behind the current
    /// launch, at 0.0 every boundary starts from a standing start.
    pub seal_events: AtomicU64,
    pub seal_while_executing: AtomicU64,

    /// Times `plan_dispatch` held the ENTIRE sealed queue because the front
    /// frame had a queued pre-launch copy. The queue is FIFO and only its
    /// front is examined, so one arriving lane's copy barrier stalls every
    /// frame behind it. Pure decode makes almost no such copies.
    pub dispatch_blocked_holds: AtomicU64,

    /// DEVICE STARVATION measured where it happens: a frame posted while
    /// nothing was executing means the device sat idle since the previous
    /// retirement. Chain engagement does NOT substitute — a seal landing while
    /// the device is busy counts as chained whether it beat retirement by
    /// 30 ms or 30 µs, so engagement can read ~97% on an empty pipeline.
    pub device_idle_us: AtomicU64,
    pub device_idle_gaps: AtomicU64,

    /// Passes that left the dispatch loop WITHOUT consulting the frame policy
    /// while the device was idle. Of the two `break`s preceding
    /// `plan_dispatch`, the control-op one can starve the device invisibly.
    pub idle_break_control: AtomicU64,
    pub idle_break_depth: AtomicU64,

    /// Micros parked with the device idle, split by whether an in-flight
    /// control op was holding launches. A `Posted` control slot arms no
    /// completion nudge, so that park can only end on the 250 ms backstop.
    pub idle_park_control_us: AtomicU64,
    pub idle_park_other_us: AtomicU64,

    /// The scheduler thread's SERIAL ingest cost: micros inside
    /// `on_fire_enqueued`, and the call count. A 256-wide boundary ingests ~256
    /// of these on ONE thread, so per-call micros become a scheduler tail.
    pub accept_us: AtomicU64,
    pub accept_calls: AtomicU64,

    /// Guest turnaround, sampled per lane per seal. The boundary period is the
    /// MAX across the fleet, so `turnaround_max / (sum / n)` separates a
    /// uniformly slow fleet from a fast one with a tail.
    pub turnaround_sum_us: AtomicU64,
    pub turnaround_max_us: AtomicU64,
    pub turnaround_n: AtomicU64,

    /// The engine lane is ONE FIFO thread that prefers launches but cannot
    /// preempt: a launch arriving mid control op waits it out. These split the
    /// lane's busy time so "should control leave the launch lane?" is measured.
    pub lane_launch_us: AtomicU64,
    pub lane_launch_n: AtomicU64,
    /// The same for prefill-carrying waves: `lane_prefill_us / lane_prefill_n`
    /// against the launch average is what an arrival costs the one thread that
    /// posts every frame.
    pub lane_prefill_us: AtomicU64,
    pub lane_prefill_n: AtomicU64,
    pub lane_control_us: AtomicU64,
    pub lane_control_n: AtomicU64,
    pub lane_control_max_us: AtomicU64,
}

/// Probes that fire *during* the non-blocking accumulator pass — i.e.
/// while the main loop is draining the request channel between fires.
#[derive(Debug, Default)]
pub struct AccumulateProbes {
    /// Wall time of the per-iter `try_recv + prepare + would_exceed + push`
    /// loop, until `try_recv` returns Empty, the batch fills, or a request is
    /// stashed for the next batch.
    pub accum_loop_us: AtomicU64,
}

/// Probes between the policy's "fire" decision and the actual execute call.
#[derive(Debug, Default)]
pub struct PreDispatchProbes {
    /// Post-decision drain (requests arriving between the accum loop and here)
    /// plus batch_ctx_ids collection.
    pub fire_prepare_us: AtomicU64,
}

/// The fire's hot path. Children sum to `total_us`.
#[derive(Debug, Default)]
pub struct ExecuteProbes {
    /// Total wall time of `BatchScheduler::execute_batch`.
    pub total_us: AtomicU64,

    /// Time spent folding per-request `LaunchPlan`s into one
    /// `BatchedForwardRequest` via `append_request_with_options`.
    pub batch_build_us: AtomicU64,

    /// Direct launch submission plus payload-free completion wait.
    pub engine_fire_us: AtomicU64,
}

/// Probes after execute returns, while the scheduler thread is doing
/// per-fire bookkeeping before looping back to accumulate.
#[derive(Debug, Default)]
pub struct PostDispatchProbes {
    /// Inert probe slot — the post-dispatch hook it timed was removed under
    /// FCFS; kept for stats-key stability.
    pub context_tick_us: AtomicU64,
    /// Cumulative-counter `fetch_add` block at the end of the fire
    /// (latency and batch-size counters).
    pub stats_update_us: AtomicU64,
}

// `probe_fire!(target, body)` runs `body`, accumulates elapsed micros into
// `target` and returns the body's value. With `profile-fire` off it expands
// to `{ let _ = &target; body }`, which keeps type-checking `target:
// &AtomicU64` at every call site while emitting no timing code.
// `probe_fire_record!` is the form for sites that already hold a `Duration`.

#[cfg(feature = "profile-fire")]
#[macro_export]
macro_rules! probe_fire {
    ($target:expr, $body:expr) => {{
        let __probe_start = ::std::time::Instant::now();
        let __probe_result = $body;
        $target.fetch_add(
            __probe_start.elapsed().as_micros() as u64,
            ::std::sync::atomic::Ordering::Relaxed,
        );
        __probe_result
    }};
}

#[cfg(not(feature = "profile-fire"))]
#[macro_export]
macro_rules! probe_fire {
    ($target:expr, $body:expr) => {{
        let _ = &$target;
        $body
    }};
}

#[cfg(feature = "profile-fire")]
#[macro_export]
macro_rules! probe_fire_record {
    ($target:expr, $duration:expr) => {{
        $target.fetch_add(
            $duration.as_micros() as u64,
            ::std::sync::atomic::Ordering::Relaxed,
        );
    }};
}

#[cfg(not(feature = "profile-fire"))]
#[macro_export]
macro_rules! probe_fire_record {
    ($target:expr, $duration:expr) => {{
        let _ = (&$target, &$duration);
    }};
}

// =============================================================================
// The HOST-SUBMIT probes (palo D0)
// =============================================================================
//
// Everything above measures the SCHEDULER thread. Nothing measured the guest
// thread's own submit — `pipeline::fire::submit_pass_stamped`, which is where
// build log 19's ruling put the frame path's cost — so the ruling could not be
// checked. These probes are process-global rather than per-engine because the
// submit runs on the guest's task, which has no scheduler handle in hand until
// the very last statement of the function.
//
// They are gated by the same `profile-fire` feature. `probe_fire_record!` is
// not enough on its own here: every site is a REGION spanning `?` and early
// returns, so it cannot be wrapped in `probe_fire!`, and by the time the macro
// sees a `Duration` the site has already paid for the timestamp. `ProbeClock`
// below is the missing half — it holds an `Instant` with the feature on and
// nothing at all with it off, so both the clock read and the `fetch_add`
// disappear together. What survives in a feature-off build is one relaxed
// `OnceLock` load per submit, against a 3.5 ms fire.

/// One phase of a guest-thread submit, and how many submits were seen.
#[derive(Debug, Default)]
pub struct HostSubmitProbes {
    /// Submits counted (one per non-no-op frame slot).
    pub submits: AtomicU64,
    /// Whole `submit_pass_stamped`, from entry to the pending-fire push.
    pub total_us: AtomicU64,
    /// The non-blocking settlement drain at the top of the submit.
    pub drain_settled_us: AtomicU64,
    /// Evaluated fire geometry: the host shadow's fold, the port map and the
    /// attention-mask lowering.
    pub geometry_us: AtomicU64,
    /// KV declaration resolve, demand, grant acquire and reserved prepare —
    /// everything that takes the store lock or awaits the planner.
    pub kv_prepare_us: AtomicU64,
    /// The working set's page translation, copied into the request.
    pub translation_us: AtomicU64,
    /// The handoff to the scheduler thread.
    pub scheduler_submit_us: AtomicU64,
    /// `HostShadow::advance` — the fold that moves the shadow one fire on.
    pub shadow_advance_us: AtomicU64,
    /// `validate_frame`, which only a k > 1 frame with a live slot reaches.
    pub validate_frame_us: AtomicU64,
    pub validate_frame_calls: AtomicU64,
}

/// The process-global host-submit probe set.
pub fn host_submit() -> &'static HostSubmitProbes {
    static PROBES: std::sync::OnceLock<HostSubmitProbes> = std::sync::OnceLock::new();
    PROBES.get_or_init(HostSubmitProbes::default)
}

/// A probe's clock. With `profile-fire` on it is an `Instant`; with the
/// feature off it holds nothing and answers `Duration::ZERO`, so the
/// `Instant::now()` syscall disappears along with the `fetch_add` — which
/// `probe_fire_record!` alone cannot do, because the site has already paid for
/// the timestamp by the time the macro sees the duration.
#[derive(Clone, Copy, Debug)]
pub struct ProbeClock {
    #[cfg(feature = "profile-fire")]
    began: std::time::Instant,
}

impl ProbeClock {
    #[must_use]
    pub fn start() -> Self {
        Self {
            #[cfg(feature = "profile-fire")]
            began: std::time::Instant::now(),
        }
    }

    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        #[cfg(feature = "profile-fire")]
        {
            self.began.elapsed()
        }
        #[cfg(not(feature = "profile-fire"))]
        {
            std::time::Duration::ZERO
        }
    }
}

/// `probe_fire_count!(target)` — a probe-gated counter bump.
#[cfg(feature = "profile-fire")]
#[macro_export]
macro_rules! probe_fire_count {
    ($target:expr) => {{
        $target.fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    }};
}

#[cfg(not(feature = "profile-fire"))]
#[macro_export]
macro_rules! probe_fire_count {
    ($target:expr) => {{
        let _ = &$target;
    }};
}
