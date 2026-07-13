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
//! ├── cold_hold_us           time spent in the F3 cold-hold window (nothing in flight)
//! ├── cold_hold_fires        count of fires that went through the F3 cold-hold path
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

    /// Time spent holding in the F3 cold-hold window (nothing in flight at
    /// all): the sub-millisecond wait for arrivals before firing partial.
    pub cold_hold_us: AtomicU64,

    /// Count of fires that went through the cold-hold path (F3), the
    /// denominator for cold-hold occupancy (`cold_hold_us / cold_hold_fires`).
    pub cold_hold_fires: AtomicU64,

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
    /// singleton waves). `wave_missing_sum` is retained for status-schema
    /// compatibility and remains zero under pure wait-all.
    pub wave_active_sum: AtomicU64,
    pub wave_missing_sum: AtomicU64,
    pub wave_fires: AtomicU64,
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
