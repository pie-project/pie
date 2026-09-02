//! Per-fire scheduler lifecycle probes.
//!
//! Gated by the `profile-fire` Cargo feature: with it off, `probe_fire!`
//! expands to a body-only no-op, while the holder structs and atomics stay
//! defined so callers still compile. The two timestamp fields are always-on.
//!
//! Invariant: `execute.total_us` equals `batch_build_us + engine_fire_us`
//! within probe overhead. `inter_fire_us` and `post_dispatch_to_fire_us`
//! measure gaps *between* fires, so summing them with children of `execute`
//! double-counts.

use std::sync::atomic::AtomicU64;

/// Every probe group for one fire, plus the two always-on timestamps.
#[derive(Debug, Default)]
pub struct FireProbes {
    /// Time between consecutive fire starts, via `swap` on
    /// `last_fire_spawn_micros`. Sibling of `execute.*`: includes
    /// `execute.total_us` plus the gap before the next fire.
    pub inter_fire_us: AtomicU64,

    /// Retirement of fire N to start of fire N+1: the rendezvous gap
    /// (chain-extender wake, main-loop drain, cohort fill). Sibling of
    /// `execute.*`.
    pub post_dispatch_to_fire_us: AtomicU64,

    /// Steady-state scheduler idle-wait: time blocked in the `while
    /// batch.is_empty()` `recv()`, recorded only once warm.
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

/// The quorum fire rule's own counters: device-side bubble, quorum latency,
/// and the reasons a fire went out narrow.
#[derive(Debug, Default)]
pub struct QuorumProbes {
    /// Device-side idle between one batch retiring and the next launching —
    /// the bubble the quorum rule drives to zero. Distinct from host-side
    /// `inter_fire_us`.
    pub inter_batch_bubble_us: AtomicU64,

    /// Last counted pipeline structurally ready to dense batch enqueued. In
    /// steady state this completes mid-flight, so the value is the slack
    /// before the in-flight batch retires.
    pub quorum_latency_us: AtomicU64,

    /// Idle-escape fires: device idle with an empty queue, so the ready
    /// subset fired at once. Dominant on agentic fleets, near-zero on
    /// saturated decode fleets.
    pub escape_fires: AtomicU64,

    /// Depth-2 submit-ahead fires: a batch in flight, below the cap, partial
    /// cohort, so the ready subset fired eagerly behind it.
    pub submit_ahead_fires: AtomicU64,

    /// Legacy counter retained in telemetry; strict wait-all never fires narrow.
    pub straggler_fires: AtomicU64,
    /// Legacy counter retained in telemetry; strict wait-all never demotes.
    pub straggler_demotions: AtomicU64,

    /// Readiness misses: a pass launched structurally ready whose late host
    /// edge (grammar mask) missed its consuming stage's device cut point, so
    /// the sample dummy-ran and the stage resubmits.
    pub readiness_miss: AtomicU64,

    /// Wait-for-all wave diagnostics, sampled at each WaitAll fire.
    /// `wave_active_sum / wave_fires` discriminates a persistent wait-set
    /// (converges to fleet width, dense waves) from a transient one (~1).
    pub wave_active_sum: AtomicU64,
    pub wave_missing_sum: AtomicU64,
    pub wave_fires: AtomicU64,

    /// Chain engagement: sealed partitions, and the subset sealed while a
    /// frame was still on the device. Their ratio says whether the fleet is
    /// pipelined (1.0: next boundary assembled behind the current launch;
    /// 0.0: every boundary starts cold).
    pub seal_events: AtomicU64,
    pub seal_while_executing: AtomicU64,

    /// Times `plan_dispatch` held the entire sealed queue because the front
    /// frame had a queued pre-launch copy (the queue is FIFO, front-only).
    pub dispatch_blocked_holds: AtomicU64,

    /// Device starvation measured where it happens: a frame posted while
    /// nothing was executing means the device sat idle since the previous
    /// retirement. Chain engagement does not substitute for this.
    pub device_idle_us: AtomicU64,
    pub device_idle_gaps: AtomicU64,

    /// Passes that left the dispatch loop without consulting the frame
    /// policy while the device was idle.
    pub idle_break_control: AtomicU64,
    pub idle_break_depth: AtomicU64,

    /// Micros parked with the device idle, split by whether an in-flight
    /// control op was holding launches. A `Posted` control slot arms no
    /// completion nudge, so that park can only end on the 250 ms backstop.
    pub idle_park_control_us: AtomicU64,
    pub idle_park_other_us: AtomicU64,

    /// The scheduler thread's serial ingest cost: micros inside
    /// `on_fire_enqueued`, and the call count.
    pub accept_us: AtomicU64,
    pub accept_calls: AtomicU64,

    /// Guest turnaround, sampled per lane per seal. The boundary period is
    /// the max across the fleet, so `turnaround_max / (sum / n)` separates a
    /// uniformly slow fleet from a fast one with a tail.
    pub turnaround_sum_us: AtomicU64,
    pub turnaround_max_us: AtomicU64,
    pub turnaround_n: AtomicU64,

    /// The engine lane is one FIFO thread that prefers launches but cannot
    /// preempt: a launch arriving mid control op waits it out. These split
    /// the lane's busy time to measure whether control should leave it.
    pub lane_launch_us: AtomicU64,
    pub lane_launch_n: AtomicU64,
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
    /// Inert probe slot — FCFS has no post-dispatch hook to time; kept for
    /// stats-key stability.
    pub context_tick_us: AtomicU64,
    /// Cumulative-counter `fetch_add` block at the end of the fire
    /// (latency and batch-size counters).
    pub stats_update_us: AtomicU64,
}

// `probe_fire!(target, body)` runs `body`, accumulates elapsed micros into
// `target`, and returns the body's value. With `profile-fire` off it expands
// to a no-op that still type-checks. `probe_fire_record!` is the form for
// sites that already hold a `Duration`.

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

// The host-submit probes.
//
// Everything above measures the scheduler thread. These measure the guest
// thread's own submit (`pipeline::fire::submit_pass_stamped`) instead, and
// are process-global because that submit runs on the guest's task, which has
// no scheduler handle until the function's last statement.
//
// Gated by the same `profile-fire` feature. `probe_fire_record!` alone isn't
// enough: every site is a region spanning `?` and early returns, so it can't
// be wrapped in `probe_fire!`. `ProbeClock` below holds an `Instant` with the
// feature on and nothing with it off, so the clock read and the `fetch_add`
// disappear together.

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
/// feature off it holds nothing and answers `Duration::ZERO`.
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
