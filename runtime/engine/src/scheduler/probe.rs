//! Per-fire scheduler lifecycle probes.
//!
//! Gated by the `profile-fire` Cargo feature. With the feature off,
//! the `probe_fire!` macro expands to a body-only no-op (no
//! `Instant::now()`, no `fetch_add`) — the holder structs and atomics
//! are still defined so callers and readers compile, but probe sites
//! produce no code.
//!
//! ## Hierarchy
//!
//! ```text
//! FireProbes
//! ├── inter_fire_us            sibling — gap between consecutive fire starts
//! ├── post_dispatch_to_fire_us sibling — gap from dispatch end to next fire
//! ├── last_fire_spawn_micros   timestamp (always-on, cheap)
//! ├── last_dispatch_end_micros timestamp (always-on, cheap)
//! ├── accumulate.*             before-fire host work
//! ├── pre_dispatch.*           between fire-decision and execute
//! ├── execute.*                the hot path; children sum to total_us
//! └── post_dispatch.*          after-execute host work
//! ```
//!
//! **Invariant**: `execute.total_us` should equal `batch_build_us +
//! driver_fire_us` within probe overhead.
//!
//! **Sibling vs nested**: `inter_fire_us` and `post_dispatch_to_fire_us`
//! are NOT contained in any `execute.*` probe — they measure gaps
//! *between* fires, not work done during a fire. Don't sum them with
//! children of `execute`.
//!
//! ## Quorum-rule probes (`quorum.*`)
//!
//! The quorum fire rule (overview §7.2, thrust-2 §3 F1–F6) needs a
//! dedicated probe family. These
//! land as scaffolding in phase S0 and are populated by the quorum core
//! in phase S5; until then they read zero. Each maps to a clause or a
//! health signal of the rule:
//!
//! ```text
//! FireProbes.quorum
//! ├── inter_batch_bubble_us  device idle between one batch retiring and the next launching (F1 target: →0)
//! ├── quorum_latency_us      last-pipeline-ready → dense-batch enqueue (F1 quorum completion)
//! ├── escape_fires           count of F2 idle-escape fires (ready subset fired on device-idle+empty-queue)
//! ├── straggler_fires        legacy field; always zero under strict wait-all
//! └── readiness_miss         count of dummy-runs: a pass launched structurally-ready whose late edge missed (M3 gate: rate < 1%)
//! ```

use std::sync::atomic::AtomicU64;

#[derive(Debug, Default)]
pub struct FireProbes {
    /// Time between consecutive fire starts (start of fire N → start of
    /// fire N+1). Computed by `swap`ping the previous timestamp in
    /// `last_fire_spawn_micros`. **Sibling** of the `execute.*` group:
    /// includes `execute.total_us` plus the gap before the next fire
    /// (accumulation, policy decision, dispatch tail).
    pub inter_fire_us: AtomicU64,

    /// Time from end of completion retirement (fire N) to start of fire
    /// N+1. The "rendezvous gap" — chain-extender wake propagation +
    /// main-loop drain + cohort fill. **Sibling** of the `execute.*`
    /// group.
    pub post_dispatch_to_fire_us: AtomicU64,

    /// Steady-state scheduler idle-wait: time the run loop spent blocked in the
    /// `while batch.is_empty()` `recv()` waiting for the NEXT batch's first
    /// request (only recorded once warm — a fire has spawned — so the cold-start
    /// wait for the first-ever request is excluded). This is the dominant chunk
    /// of the round-trip R when the fleet is decode-bound: it measures
    /// dispatch→inferlet-wake→sample→resubmit→SERVICE-hop→scheduler-recv, i.e.
    /// everything OUTSIDE the scheduler's own build/decide (`accum_loop` +
    /// `fire_prepare` + `batch_build`). A large value ⇒ R lives in the resubmit
    /// round-trip (inferlet/SERVICE), not scheduler processing or the driver.
    pub recv_block_wait_us: AtomicU64,

    /// Timestamp (micros from `sched_epoch`) of the most recent fire
    /// start. Used to compute `inter_fire_us` via `swap`. Cheap — kept
    /// always-on regardless of `profile-fire`.
    pub last_fire_spawn_micros: AtomicU64,

    /// Timestamp (micros from `sched_epoch`) of the most recent completion
    /// retirement. Used to compute `post_dispatch_to_fire_us`.
    /// Cheap — kept always-on.
    pub last_dispatch_end_micros: AtomicU64,

    pub accumulate: AccumulateProbes,
    pub pre_dispatch: PreDispatchProbes,
    pub execute: ExecuteProbes,
    pub post_dispatch: PostDispatchProbes,
    pub quorum: QuorumProbes,
}

/// Probes for the quorum fire rule (overview §7.2; thrust-2 §3 F1–F6).
///
/// Scaffolding lands in S0; the quorum core (S5) writes these. Duration
/// fields (`*_us`) accumulate micros via `probe_fire!` / `probe_fire_record!`;
/// the `*_fires` / `*_miss` fields are counters incremented per event. All
/// are `AtomicU64` and read via `load(Relaxed)`; readers derive rates by
/// dividing against `total_batches` (see `crate::scheduler`).
#[derive(Debug, Default)]
pub struct QuorumProbes {
    /// Device idle between one batch retiring and the next launching — the
    /// inter-batch bubble the quorum rule drives to zero in steady state
    /// (F1). Distinct from `inter_fire_us` (host-side gap between fire
    /// *starts*): this is the *device*'s idle window, the bubble the M1/M3
    /// gate bounds at p50 < 100 µs.
    pub inter_batch_bubble_us: AtomicU64,

    /// Quorum latency: from the moment the last counted pipeline becomes
    /// structurally ready to the dense batch's enqueue (F1). Steady state
    /// this completes mid-flight, so the value is the slack before the
    /// in-flight batch retires.
    pub quorum_latency_us: AtomicU64,

    /// Count of idle-escape fires (F2): device went idle with the queue
    /// empty and the ready subset fired immediately. Divide by
    /// `total_batches` for the escape rate — dominant on agentic fleets,
    /// near-zero on saturated decode fleets.
    pub escape_fires: AtomicU64,

    /// Count of depth-2 submit-ahead fires (G3 bubble): a batch was in flight
    /// and below the cap with a partial cohort, so the ready subset fired
    /// eagerly behind it rather than holding for quorum. Divide by
    /// `total_batches` for the submit-ahead rate — the steady-state
    /// decode-fleet bubble-filler. Zero when the cohort always completes
    /// before the in-flight batch retires (pure quorum).
    pub submit_ahead_fires: AtomicU64,

    /// Legacy counter retained in telemetry; strict wait-all never fires narrow.
    pub straggler_fires: AtomicU64,
    /// Legacy counter retained in telemetry; strict wait-all never demotes.
    pub straggler_demotions: AtomicU64,

    /// Dummy-run / readiness-miss count: a pass launched as structurally
    /// ready (F5) whose genuinely-late host edge (grammar mask) had not
    /// landed when its consuming stage reached the device cut point, so the
    /// sample dummy-ran and the stage resubmits. The M3 gate holds this
    /// rate < 1% on the steady-state decode fleet.
    pub readiness_miss: AtomicU64,

    /// Wait-for-all wave diagnostics (M-AB, delta). Sampled at each WaitAll
    /// fire: `wave_active_sum` = Σ active_pipelines (the wait-set size) and
    /// `wave_fires` = the denominator. `avg_active = wave_active_sum /
    /// wave_fires` discriminates a PERSISTENT wait-set (converges to fleet
    /// width ⇒ waves should be dense) from a TRANSIENT one (stuck ≈1 ⇒
    /// singleton waves). `wave_missing_sum` counts absentees at fire time:
    /// Strict wait-all records zero absentees at every fire.
    pub wave_active_sum: AtomicU64,
    pub wave_missing_sum: AtomicU64,
    pub wave_fires: AtomicU64,

    /// CHAIN ENGAGEMENT, restored (the campaign's `seal_exec`). `seal_events`
    /// counts sealed partitions; `seal_while_executing` counts the subset
    /// sealed while a frame was still on the device. Their ratio is the one
    /// number that says whether the fleet is PIPELINED: at 1.0 the next
    /// boundary is assembled behind the current launch and the host chain is
    /// hidden, at 0.0 every boundary is gathered from a standing start with
    /// the device idle. §10.22 built its whole per-decile account on this and
    /// the counter was deleted afterwards, which is why every section since
    /// has had to reason about chaining without being able to see it.
    pub seal_events: AtomicU64,
    pub seal_while_executing: AtomicU64,

    /// Times `plan_dispatch` held the ENTIRE sealed queue because the front
    /// frame had a member with a queued pre-launch copy. The queue is FIFO
    /// and only its front is examined, so one arriving lane's copy barrier
    /// stalls every frame behind it. Pure decode generates almost no
    /// pre-launch copies, so this counter is expected to be ~0 without
    /// turnover — which is exactly what makes it a discriminator.
    pub dispatch_blocked_holds: AtomicU64,

    /// DEVICE STARVATION, measured where it happens: when a frame is posted
    /// and nothing was executing, the device sat idle from the previous
    /// frame's retirement until now. `device_idle_us` sums those gaps and
    /// `device_idle_gaps` counts them.
    ///
    /// This is the quantity §3.3 identified as the whole recoverable prize
    /// (8.03 s of 38.2 s contended against 0.42 s of 28.1 s uncontended) and
    /// it has had no instrument since. Chain engagement does NOT substitute
    /// for it: a seal that lands while the device is busy counts as chained
    /// whether it beat the retirement by 30 ms or by 30 us, so engagement can
    /// read ~97% while the pipeline is in fact running on empty.
    pub device_idle_us: AtomicU64,
    pub device_idle_gaps: AtomicU64,

    /// Passes that left the dispatch loop WITHOUT consulting the frame
    /// policy at all, while the device was idle. Two `break`s sit in front of
    /// `plan_dispatch`: an in-flight control op that holds launches, and the
    /// run-ahead depth cap. The first can starve the device while every lane
    /// has a complete frame, and no gate counter can see it because the gate
    /// never runs.
    pub idle_break_control: AtomicU64,
    pub idle_break_depth: AtomicU64,

    /// Microseconds the scheduler thread spent PARKED with the device idle,
    /// split by whether an in-flight control op was holding launches at the
    /// moment it parked. A `Posted` control slot arms no completion nudge, so
    /// that park can only end on the 250 ms backstop — which is exactly the
    /// length of the stalls that decide the turnover cell's mode.
    pub idle_park_control_us: AtomicU64,
    pub idle_park_other_us: AtomicU64,

    /// The scheduler thread's own SERIAL cost of ingesting arrivals:
    /// microseconds inside `on_fire_enqueued` and the number of calls. A
    /// boundary at 256-wide ingests ~256 of these on ONE thread, so if the
    /// per-call cost is microseconds the gate cannot open until milliseconds
    /// after the last guest actually submitted -- and the tail would be the
    /// scheduler, not the guests.
    pub accept_us: AtomicU64,
    pub accept_calls: AtomicU64,

    /// Guest turnaround, sampled for every lane at every seal: the span from
    /// the retirement that handed it a result to the moment its next frame
    /// read complete. The boundary period is the MAX of these across the
    /// fleet, so `turnaround_max / (turnaround_sum / turnaround_n)` says
    /// whether the fleet is uniformly slow (a serial host cost everyone pays)
    /// or fast with a tail (a few lanes late, which is what the device-idle
    /// census sees).
    pub turnaround_sum_us: AtomicU64,
    pub turnaround_max_us: AtomicU64,
    pub turnaround_n: AtomicU64,

    /// The driver lane is ONE thread running every driver call in FIFO order,
    /// and it already prefers launches (`next_request` tries `launch_rx`
    /// first). What it cannot do is preempt: once a control op starts, a
    /// launch that arrives waits it out. These split the lane's own busy time
    /// so "should control leave the launch lane?" is a measured question.
    pub lane_launch_us: AtomicU64,
    pub lane_launch_n: AtomicU64,
    /// The same, for waves that carry a prefill. `lane_prefill_us /
    /// lane_prefill_n` against `lane_launch_us / lane_launch_n` is the cost
    /// of an arrival to the one thread that posts every frame.
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
    /// Wall time of the per-iter `try_recv + prepare + would_exceed +
    /// push` loop, until the first `try_recv` returns Empty (or the
    /// batch is full / a request was stashed for next batch).
    pub accum_loop_us: AtomicU64,
}

/// Probes between the policy's "fire" decision and the actual execute call.
#[derive(Debug, Default)]
pub struct PreDispatchProbes {
    /// Time spent on the post-decision drain (catches requests that
    /// arrived between the accum loop and here) plus batch_ctx_ids
    /// collection.
    pub fire_prepare_us: AtomicU64,
}

/// The fire's hot path. Children sum to `total_us`.
#[derive(Debug, Default)]
pub struct ExecuteProbes {
    /// Total wall time of `BatchScheduler::execute_batch`. Should
    /// equal `batch_build_us + driver_fire_us` within probe overhead.
    pub total_us: AtomicU64,

    /// Time spent folding per-request `LaunchPlan`s into one
    /// `BatchedForwardRequest` via `append_request_with_options`.
    pub batch_build_us: AtomicU64,

    /// Direct launch submission plus payload-free completion wait.
    pub driver_fire_us: AtomicU64,
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

// =============================================================================
// Macros
// =============================================================================
//
// `probe_fire!(target, body)` runs `body`, accumulates the elapsed
// micros into `target`, and returns the body's value. With
// `profile-fire` off the macro expands to `{ let _ = &target; body }`
// — body runs unchanged, no `Instant::now()` call, no `fetch_add`.
// The `let _ = &target;` keeps the macro accepting the same call sites
// (so we still type-check that `target: &AtomicU64`).
//
// `probe_fire_record!(target, duration)` is the lower-level form for
// sites that already have a `Duration` in hand (e.g. derived from
// existing `Instant::elapsed()` outside the macro).

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
