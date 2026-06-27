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
//! **Invariant**: `execute.total_us` should equal the sum of its
//! children (`batch_build_us + driver_fire_us + response_dispatch_us`)
//! within ~5 µs of probe overhead. If they diverge meaningfully, there
//! is unaccounted work *inside* `execute` that needs its own probe.
//!
//! **Sibling vs nested**: `inter_fire_us` and `post_dispatch_to_fire_us`
//! are NOT contained in any `execute.*` probe — they measure gaps
//! *between* fires, not work done during a fire. Don't sum them with
//! children of `execute`.

use std::sync::atomic::AtomicU64;

#[derive(Debug, Default)]
pub struct FireProbes {
    /// Time between consecutive fire starts (start of fire N → start of
    /// fire N+1). Computed by `swap`ping the previous timestamp in
    /// `last_fire_spawn_micros`. **Sibling** of the `execute.*` group:
    /// includes `execute.total_us` plus the gap before the next fire
    /// (accumulation, policy decision, dispatch tail).
    pub inter_fire_us: AtomicU64,

    /// Time from end of response dispatch (fire N) to start of fire
    /// N+1. The "rendezvous gap" — chain-extender wake propagation +
    /// main-loop drain + cohort fill. **Sibling** of the `execute.*`
    /// group.
    pub post_dispatch_to_fire_us: AtomicU64,

    /// Timestamp (micros from `sched_epoch`) of the most recent fire
    /// start. Used to compute `inter_fire_us` via `swap`. Cheap — kept
    /// always-on regardless of `profile-fire`.
    pub last_fire_spawn_micros: AtomicU64,

    /// Timestamp (micros from `sched_epoch`) of the most recent
    /// response-dispatch end. Used to compute `post_dispatch_to_fire_us`.
    /// Cheap — kept always-on.
    pub last_dispatch_end_micros: AtomicU64,

    pub accumulate: AccumulateProbes,
    pub pre_dispatch: PreDispatchProbes,
    pub execute: ExecuteProbes,
    pub post_dispatch: PostDispatchProbes,
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
    /// equal `batch_build_us + driver_fire_us + response_dispatch.total_us`
    /// within ~µs.
    pub total_us: AtomicU64,

    /// Time spent folding per-request `ForwardRequest`s into one
    /// `BatchedForwardRequest` via `append_request_with_options`.
    pub batch_build_us: AtomicU64,

    /// IPC submit + GPU compute + response sync. The bulk of every
    /// fire (~10 ms at conc=256). Further decomposed by the
    /// `driver_cuda` probe domain into `ipc_submit / gpu_wait /
    /// ipc_recv` (Rust-side) and `wire_parse / plan / h2d /
    /// kernel_launch / sync / response_build` (C++-side, filled via
    /// the response payload).
    pub driver_fire_us: AtomicU64,

    /// Per-request response handling — oneshot fires (`Direct`),
    /// chain-extender submits (`Chain`), chunked-retry routes
    /// (`Chunk`), and queueing the `deferred_drop` Vec for the
    /// blocking pool. Sub-structured by completion type.
    pub response_dispatch: ResponseDispatchProbes,
}

/// Response-dispatch sub-probes. `total_us` is the wall time of the
/// response-dispatch loop; the `*_count` fields are counters (not
/// durations) recording the workload mix per fire.
///
/// We deliberately don't time each completion-type arm separately —
/// each `oneshot::Sender::send` and `pool.submit` call is ~50-100 ns,
/// and probe overhead at that granularity would dwarf the work. The
/// counts let us reason about workload shape (`direct_count / fire =
/// concurrency` at steady state) without per-call probe cost.
#[derive(Debug, Default)]
pub struct ResponseDispatchProbes {
    /// Wall time of the entire response-dispatch loop.
    pub total_us: AtomicU64,
    /// Number of `Completion::Direct` arms taken per fire (sum across
    /// all fires; divide by `total_batches` for per-fire mean).
    pub direct_count: AtomicU64,
    /// Number of `Completion::Chunk` arms taken (chunked-retry
    /// continuations). Rare in the hot path.
    pub chunk_count: AtomicU64,
}

/// Probes after execute returns, while the scheduler thread is doing
/// per-fire bookkeeping before looping back to accumulate.
#[derive(Debug, Default)]
pub struct PostDispatchProbes {
    /// Inert probe slot — the post-dispatch hook it timed was removed under
    /// FCFS; kept for stats-key stability.
    pub context_tick_us: AtomicU64,
    /// Cumulative-counter `fetch_add` block at the end of the fire
    /// (latency, batch_size_hist, system_spec_*).
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
