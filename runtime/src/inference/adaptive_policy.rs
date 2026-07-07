//! Batch scheduling policies.
//!
//! Three policies live here:
//!
//!   - `GreedyPolicy` — fire immediately, zero waiting.
//!   - `EagerPolicy` — wait for the peer cohort to gather (using a
//!     `pinned_count`-driven cap and a `last_latency` safety bound).
//!   - `AdaptivePolicy` (default) — fire on a structural cap or on
//!     `fired_high_water`, with `last_latency` as a deadlock
//!     watchdog. One heuristic (`fired_high_water`), explicitly
//!     scoped.
//!
//! Selection via the per-model `[model.scheduler].policy` field in
//! the TOML config (default = `adaptive`).
//!
//! ## Pie batching model
//!
//! Pie performs **iteration-level batching**: each in-flight context
//! re-submits a forward-pass request after every token. The scheduler
//! accumulates these into a batch and the policy decides when to fire.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::scheduler::quorum::FireClause;
use super::scheduler::{Decision, SchedulerStats, SchedulingPolicy};
use crate::process::ProcessId;

// =============================================================================
// AdaptivePolicy — fire on structural cap or historical cohort size.
// =============================================================================
//
// Firing rule (in evaluation order):
//
//   1. `B >= max_forward_requests`  — structural driver limit.
//   2. `fired_high_water == 0`      — no historical fire yet; fire
//                                     whatever we have (cold start).
//   3. `B >= fired_high_water`      — we've matched the largest cohort
//                                     ever fired; firing more won't
//                                     produce a bigger batch from
//                                     existing peers.
//   4. `elapsed >= last_latency`    — deadlock watchdog: the cohort
//                                     never grew back to
//                                     `fired_high_water` (active
//                                     concurrency dropped — e.g.,
//                                     inferlets finished). Fire what
//                                     we have so the system makes
//                                     progress. `last_latency` is a
//                                     natural, self-recalibrating
//                                     bound (don't wait longer than
//                                     one fire takes); not a tuning
//                                     parameter.
//   otherwise: `Wait(last_latency - elapsed)`.
//
// `fired_high_water` is the only firing-rule concept; `last_latency`
// is a watchdog to prevent indefinite parking, not a steady-state
// trigger.

pub(super) struct AdaptivePolicy {
    max_forward_requests: usize,
    /// Largest batch size that has fired. Monotonic; only ratchets
    /// upward. The single firing-rule concept.
    fired_high_water: usize,
    /// Previous fire's compute time, in seconds. Used solely as the
    /// deadlock watchdog (rule 4). Zero until the second fire
    /// completes (first fire's latency is unrepresentative).
    last_latency: f64,
    /// Total fires the policy has observed completing. Used to
    /// suppress the first `on_complete` from setting `last_latency`.
    fires_completed: usize,
    /// When the current batch started accumulating. Reset on fire.
    batch_start_time: Option<Instant>,
}

impl AdaptivePolicy {
    pub fn new(max_forward_requests: usize, _driver_idx: usize) -> Self {
        Self {
            max_forward_requests,
            fired_high_water: 0,
            last_latency: 0.0,
            fires_completed: 0,
            batch_start_time: None,
        }
    }
}

impl SchedulingPolicy for AdaptivePolicy {
    fn on_arrival(&mut self, _program_identity_hashes: &[u64]) {
        if self.batch_start_time.is_none() {
            self.batch_start_time = Some(Instant::now());
        }
    }

    fn on_complete(&mut self, latency: Duration) {
        self.fires_completed += 1;
        if self.fires_completed > 1 {
            self.last_latency = latency.as_secs_f64();
        }
    }

    fn on_fired(&mut self, fired_size: usize) {
        self.batch_start_time = None;
        if fired_size > self.fired_high_water {
            self.fired_high_water = fired_size;
        }
    }

    fn decide(&mut self, current_forward_requests: usize) -> Decision {
        if current_forward_requests >= self.max_forward_requests {
            return Decision::Fire;
        }
        if self.fired_high_water == 0 || current_forward_requests >= self.fired_high_water {
            return Decision::Fire;
        }
        let Some(start) = self.batch_start_time else {
            // Defensive: `batch_start_time` should be `Some` whenever
            // `decide` runs (set on first `on_arrival`).
            return Decision::Fire;
        };
        // Cold start: no fire has completed yet, no bound to apply.
        if self.last_latency == 0.0 {
            return Decision::Fire;
        }
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= self.last_latency {
            return Decision::Fire;
        }
        Decision::Wait(Duration::from_secs_f64(self.last_latency - elapsed))
    }
}

// =============================================================================
// EagerPolicy — pinned_count cohort cap + last_latency safety bound.
// =============================================================================
//
// Fires when the live `pinned_count(driver_idx)` cohort has fully
// assembled in the batch, or when one `last_latency` has elapsed.
// Tracks `cohort_high_water` per-batch (monotonic over the
// accumulation window) so transient `pinned_count` dips between
// pin/unpin don't fire a partial batch.

pub(super) struct EagerPolicy {
    max_forward_requests: usize,
    driver_idx: usize,
    batch_start_time: Option<Instant>,
    last_latency: f64,
    batches_completed: usize,
    cohort_high_water: usize,
}

impl EagerPolicy {
    pub fn new(max_forward_requests: usize, driver_idx: usize) -> Self {
        Self {
            max_forward_requests,
            driver_idx,
            batch_start_time: None,
            last_latency: 0.0,
            batches_completed: 0,
            cohort_high_water: 0,
        }
    }
}

impl SchedulingPolicy for EagerPolicy {
    fn on_arrival(&mut self, _program_identity_hashes: &[u64]) {
        if self.batch_start_time.is_none() {
            self.batch_start_time = Some(Instant::now());
        }
    }

    fn on_complete(&mut self, latency: Duration) {
        self.batches_completed += 1;
        if self.batches_completed > 1 {
            self.last_latency = latency.as_secs_f64();
        }
    }

    fn on_fired(&mut self, _fired_size: usize) {
        self.batch_start_time = None;
        self.cohort_high_water = 0;
    }

    fn decide(&mut self, current_forward_requests: usize) -> Decision {
        if current_forward_requests >= self.max_forward_requests {
            return Decision::Fire;
        }
        // Per-driver pinned-context count is gone with the context actor
        // (Phase 5); the cohort-high-water path degrades to the latency-based
        // decision below.
        let active = 0;
        if active > self.cohort_high_water {
            self.cohort_high_water = active;
        }
        let target = self.cohort_high_water;
        if target > 0 && current_forward_requests >= target {
            return Decision::Fire;
        }
        if self.last_latency == 0.0 {
            return Decision::Fire;
        }
        let Some(start) = self.batch_start_time else {
            return Decision::Fire;
        };
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= self.last_latency {
            return Decision::Fire;
        }
        Decision::Wait(Duration::from_secs_f64(self.last_latency - elapsed))
    }
}

// =============================================================================
// GreedyPolicy — fire immediately. Zero state.
// =============================================================================
//
// Retained as a reference baseline. Useful when:
//   - Debugging the scheduler itself: any non-trivial policy bug
//     looks like a scheduler bug. Swap in greedy and see if the
//     symptom persists; if not, the policy is the cause.
//   - Workloads where you genuinely want zero coalescing (RPS=1,
//     latency above all else, or a single interactive session).
//
// At any RPS > 1, greedy produces small B=1–2 batches and pays
// per-batch fixed overhead on every token — measured p50 stays at
// ~1100–1250 ms regardless of load (vs 350–800 ms for adaptive
// across the same range). Throughput at saturation suffers
// correspondingly (~6% below adaptive at RPS=64 in our matrix).

pub(super) struct GreedyPolicy;

impl GreedyPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl SchedulingPolicy for GreedyPolicy {
    fn on_arrival(&mut self, _program_identity_hashes: &[u64]) {}
    fn on_complete(&mut self, _latency: Duration) {}
    fn on_fired(&mut self, _fired_size: usize) {}

    fn decide(&mut self, _current_forward_requests: usize) -> Decision {
        // The scheduler only calls `decide` when the batch is
        // non-empty, and the BatchAccumulator already enforces
        // `max_forward_requests` upstream of the policy. So: just fire,
        // every time.
        Decision::Fire
    }
}

// =============================================================================
// RunAheadPolicy — #6 run-ahead just-in-time firing.
// =============================================================================
//
// The run-ahead scheduler keeps the GPU pipe full by firing the next decode
// batch *before* the in-flight batch finishes, so the enqueue lands just as the
// GPU goes idle. The single firing rule (Oracle §4):
//
//   fire_at = max(in_flight_completion - lead_time, collection_complete)
//
//   * `in_flight_completion` — when the in-flight batch is estimated to finish
//     on the GPU: `fire_time + EWMA(forward latency)`. `None` when nothing is in
//     flight, so the term drops out and we fire once collected.
//   * `lead_time` — `EWMA(submission/enqueue latency)`: how far ahead to fire so
//     the next batch is enqueued just-in-time (fed via `on_submitted`).
//   * `collection_complete` — when the CPU has the next batch ready. MVP: a
//     non-empty batch is "collected" (dense rebatch is the loop's barrier job),
//     so this is `now` whenever `decide` runs on a non-empty batch.
//
// GPU-bound -> the in-flight term dominates (fire one lead-time early);
// CPU-bound -> the collection term dominates (fire as soon as ready; the GPU
// takes a bubble, correct for that workload). MVP = one-step run-ahead (R10);
// the run loop caps the in-flight depth.
//
// Degrades cleanly under the current synchronous fire path: with no batch in
// flight at `decide` time, `in_flight_completion` is `None` -> fire-when-
// collected (greedy). The timing rule only bites once the fire path is
// non-blocking (the scheduler-pipelining rewire).

/// EWMA smoothing factor (recent-weighted; the TCP-RTT-style 1/4).
const RUN_AHEAD_EWMA_ALPHA: f64 = 0.25;

/// One-step run-ahead (R10): the DEFAULT run-ahead depth — at most one batch
/// computing + one prefetched. Override via `PIE_SCHED_MAX_IN_FLIGHT`
/// (see [`max_in_flight`]).
const DEFAULT_MAX_IN_FLIGHT: usize = 2;

/// The configured run-ahead depth cap — the `in_flight` FIFO bound both policies
/// gate on. Reads `PIE_SCHED_MAX_IN_FLIGHT` once (default
/// `DEFAULT_MAX_IN_FLIGHT` = 2, floored at 1).
///
/// This is the depth-2 → depth-N plumbing for the G3 cross-pipeline-concurrency
/// lever: at D=2 only two fires overlap, covering ~2×(µs compute) of the ~700µs
/// host round-trip → ~500-600µs device idle/step on the tiny-model fleet. Raising
/// D toward the fleet width keeps ~D independent pipelines' fires in flight so the
/// GPU stays fed through any one pipeline's round-trip.
///
/// WAR safety at D>2: the driver's single scalar `last_eager_d2h_done` guard
/// (executor.cpp) is depth-k-correct WITHOUT a per-depth event ring. All eager
/// D2H copies are enqueued on ONE FIFO copy stream, so waiting the newest done
/// event transitively waits every earlier copy (stream order) — one event
/// covers all un-drained prior fires. Ratified empirically at cap=4 (DEEP4,
/// byte-identical vs sync); the once-planned WAR ring was dropped as a no-op.
/// Default 2 ⇒ byte-for-byte no behavior change; raising D is a perf/policy
/// experiment (device residency vs round-trip), not a correctness gamble.
pub(super) fn max_in_flight() -> usize {
    static DEPTH: OnceLock<usize> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("PIE_SCHED_MAX_IN_FLIGHT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&d| d >= 1)
            .unwrap_or(DEFAULT_MAX_IN_FLIGHT)
    })
}

/// Floor (µs) for the cap-wait poll so an overdue estimate doesn't spin; the
/// completion channel preempts the wait the instant a batch actually finishes.
const MIN_POLL_US: u64 = 50;

/// #10 cost-model coefficients, calibrated from delta's #11 load-test on
/// `50809f5d` (RTX 4090). Load-bearing finding: **compile ≫ fire by ~10³–10⁴** —
/// the per-DISTINCT-program compile wall dominates; per-fire launch is a
/// negligible correction. The accumulation decision keys on the distinct-program
/// compile cost, not batch size.
///
/// **Per-distinct finalize residual** (delta C2): the on-context
/// `cuModuleLoadData` PTX→SASS that **prefetch CANNOT hide** (the ~11ms NVRTC,
/// C1−C2, IS prefetch-hidden). Bimodal ~0.3–13ms (driver-SASS-cache-hit vs cold).
/// Conservative midpoint; **delta C1 is a LOWER BOUND** (trivial add-kernel) —
/// real grammar/spec-verify programs compile slower, so #10-verify must use REAL
/// grammars and the true wall is ≥ this.
const PER_DISTINCT_FINALIZE_S: f64 = 0.005;
/// Per-distinct-program M-batched fire cost (s). echo's occupancy bench
/// (`5a7e6ec9`; 4090, vocab=151936, real argmax `OneBlockPerRow`) MEASURED the
/// single-row argmax at ~30µs of real COMPUTE (1 block reducing 151936 / ~1/128
/// SM occupancy) — NOT delta's trivial-kernel launch-only C3 (1.4µs). The
/// M-batch (one `num_rows=N` kernel per distinct program) collapses N sequential
/// fires into ~30-58µs total for the identical-collapsed case. So the fire, like
/// the compile, is **per-DISTINCT-program** (the M-batch groups identical rows);
/// 58µs ≪ the ~5ms finalize residual ⇒ the policy stays compile-dominated
/// (decision unchanged, coefficient corrected).
const PER_MBATCH_FIRE_S: f64 = 0.000058;

/// #10 merged-batch cost model, delta-#11-calibrated. Cost is DOMINATED by the
/// per-DISTINCT-program compile wall; **#11 dedup (landed `50809f5d`, delta C4
/// exact: K identical → 1 compile) is the scaling win** — it collapses the ~11ms
/// compile for identical programs, so a batch's compile cost is
/// `distinct_programs`, not `total_requests`. echo's identical-bytecode M-batch
/// (the FIRE-axis collapse) is a ~0.01% secondary correction, DECOUPLED from the
/// policy (measure-first, land-vs-defer). So the policy accumulates IDENTICAL
/// programs near-free (dedup) and CAPS DISTINCT programs (each a compile wall) —
/// driven by `distinct_program_count` (the dedup key), not the M-batch.
#[derive(Clone, Copy, Debug)]
struct MergedCost {
    /// Per-distinct-program finalize residual (s) — the prefetch-unhideable floor.
    per_distinct_finalize_s: f64,
    /// Per-distinct-program M-batched fire cost (s) — echo `5a7e6ec9`; a
    /// negligible correction vs the per-distinct finalize residual.
    per_mbatch_fire_s: f64,
    /// #11 dedup collapses identical programs → one compile (landed `50809f5d`),
    /// so the compile cost is per-DISTINCT. `true` reflects the landed dedup; the
    /// distinct-`program_identity_hash` count is now plumbed through `on_arrival`
    /// (the policy's `distinct_programs` set), so `estimate_s` prices the actual
    /// distinct count — identical batches accumulate near-free, distinct bursts
    /// hit the cap.
    dedup_collapses_identical: bool,
}

impl MergedCost {
    /// Calibrated from delta's #11 load-test; dedup landed on `50809f5d`.
    const fn calibrated() -> Self {
        Self {
            per_distinct_finalize_s: PER_DISTINCT_FINALIZE_S,
            per_mbatch_fire_s: PER_MBATCH_FIRE_S,
            dedup_collapses_identical: true,
        }
    }

    /// Estimated at-fire **sampling-program** cost (s) of a batch of
    /// `total_requests` spanning `distinct_programs` distinct programs. Under the
    /// landed optimizations BOTH axes collapse per-DISTINCT: #11 dedup
    /// (`50809f5d`) → one compile per distinct; echo's M-batch (`5a7e6ec9`) → one
    /// `num_rows=N` fire kernel per distinct. So the cost is
    /// `distinct × (finalize_residual + M_batched_fire)`. Plain-decode requests
    /// carry no program ⇒ `distinct_programs = 0` ⇒ zero sampling cost (they
    /// batch freely; their forward-pass cost is bounded by capacity, not here).
    /// The ~5ms finalize dominates the ~58µs fire ⇒ compile-driven policy.
    fn estimate_s(&self, total_requests: usize, distinct_programs: usize) -> f64 {
        let distinct = if self.dedup_collapses_identical {
            distinct_programs
        } else {
            // Counterfactual (dedup off): every request is its own compile+fire.
            total_requests
        };
        distinct as f64 * (self.per_distinct_finalize_s + self.per_mbatch_fire_s)
    }
}

/// The #10 distinct-program compile-cost ceiling (s), read once from
/// `PIE_SCHED_MAX_ACCUM_COST_US`. DECOUPLED from the latency window (echo's note):
/// `accum_window` caps how long we WAIT; this caps how much distinct-program
/// COMPILE cost a batch absorbs before firing (the burst-of-distinct wall).
/// Default = the budget below (~40ms ≈ 8 distinct programs at the finalize
/// residual). dedup makes identical programs free, so this only bites genuine
/// distinct-program bursts.
fn max_accum_cost_from_env() -> Duration {
    use std::sync::OnceLock;
    static MAX: OnceLock<Duration> = OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("PIE_SCHED_MAX_ACCUM_COST_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_micros)
            .unwrap_or(Duration::from_millis(40))
    })
}

/// The #10 adaptive accumulation window (the hard p99 latency cap on how long the
/// cold/idle path waits to co-batch concurrent requests), read once from
/// `PIE_SCHED_ACCUM_WINDOW_US`. Default unset → `ZERO` → today's fire-on-arrival
/// (no production behavior change until enabled + #10-verify proves no
/// common-path regression). Distinct from the FIXED test env-hold
/// `PIE_SCHED_ACCUM_HOLD_US` (run-loop, blind wait) — this one is ADAPTIVE: it
/// fires immediately under low arrival rate.
fn accum_window_from_env() -> Duration {
    use std::sync::OnceLock;
    static WINDOW: OnceLock<Duration> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        std::env::var("PIE_SCHED_ACCUM_WINDOW_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_micros)
            .unwrap_or(Duration::ZERO)
    })
}

#[allow(dead_code)] // wired into `run()` when the fire path goes non-blocking.
pub(super) struct RunAheadPolicy {
    /// Structural cap — a full batch always fires immediately.
    max_forward_requests: usize,
    /// EWMA of forward (GPU) latency, in seconds. Estimates in-flight completion.
    forward_latency: f64,
    /// EWMA of submission (enqueue) latency, in seconds = the lead time.
    lead_time: f64,
    /// Estimated GPU-completion instants of in-flight batches, fire-order FIFO
    /// (front = earliest to complete; drives `fire_at`). Bounded by
    /// `MAX_IN_FLIGHT` (one-step run-ahead, R10).
    in_flight: VecDeque<Instant>,
    /// Forwards observed completing — the first latency is unrepresentative
    /// (cold start) and is skipped from the EWMA.
    forwards_completed: usize,
    /// #10 adaptive accumulation window (p99 latency cap). `ZERO` ⇒ fire-on-arrival.
    accum_window: Duration,
    /// Last request arrival instant — feeds the inter-arrival EWMA.
    last_arrival: Option<Instant>,
    /// EWMA of inter-arrival time (seconds). Short (high arrival rate) ⇒ co-batch;
    /// long/sparse ⇒ fire immediately (the low-concurrency no-regression path).
    inter_arrival_ewma: f64,
    /// When the current (not-yet-fired) batch began accumulating — bounds the
    /// accumulation wait to `accum_window`. Reset on fire.
    accumulating_since: Option<Instant>,
    /// #10 merged-batch cost model (delta-calibrated, dedup-driven).
    merged_cost: MergedCost,
    /// #10 distinct-program compile-cost ceiling — decoupled from `accum_window`
    /// (the latency cap) per echo's note.
    max_accum_cost: Duration,
    /// #10 distinct-program identities (`program_identity_hash`) accumulated in
    /// the current (not-yet-fired) window, unioned across all programs in all
    /// accumulated requests. `len()` = the distinct compile+fire walls the batch
    /// will pay (identical programs self-dedup → one entry). Reset on fire.
    distinct_programs: HashSet<u64>,
}

#[allow(dead_code)]
impl RunAheadPolicy {
    pub fn new(max_forward_requests: usize) -> Self {
        Self {
            max_forward_requests,
            forward_latency: 0.0,
            lead_time: 0.0,
            in_flight: VecDeque::new(),
            forwards_completed: 0,
            accum_window: accum_window_from_env(),
            last_arrival: None,
            inter_arrival_ewma: 0.0,
            accumulating_since: None,
            merged_cost: MergedCost::calibrated(),
            max_accum_cost: max_accum_cost_from_env(),
            distinct_programs: HashSet::new(),
        }
    }

    /// Recent-weighted EWMA; the first non-zero sample seeds the average.
    fn ewma(prev: f64, sample: f64) -> f64 {
        if prev == 0.0 {
            sample
        } else {
            RUN_AHEAD_EWMA_ALPHA * sample + (1.0 - RUN_AHEAD_EWMA_ALPHA) * prev
        }
    }

    /// #10 adaptive accumulation window (the cold/idle path, where no in-flight
    /// batch provides a natural overlap window). Returns `Wait` ONLY when the EWMA
    /// arrival rate predicts another request within the remaining latency budget
    /// AND the merged-fire cost still justifies a larger batch; otherwise `Fire`.
    /// Low concurrency / sparse arrivals fire immediately — the common-path
    /// no-latency-regression guarantee (the #10-verify bar).
    fn accumulation_decision(&self, current_forward_requests: usize) -> Decision {
        // Disabled (default, env unset) ⇒ today's fire-on-arrival, byte-for-byte.
        if self.accum_window.is_zero() {
            return Decision::Fire;
        }
        // Capacity cap — a full batch fires now.
        if current_forward_requests >= self.max_forward_requests {
            return Decision::Fire;
        }
        // Cost cap (DECOUPLED from the latency window per echo's note): fire once
        // the batch's estimated distinct-program compile+fire cost reaches
        // `max_accum_cost`. delta's #11: compile ≫ fire, and #11 dedup (landed)
        // collapses identical → one compile, echo's M-batch (`5a7e6ec9`) one
        // fire-kernel — so identical batches accumulate near-free while genuine
        // distinct-program bursts hit this wall. The distinct count is the live
        // `distinct_programs` set (per-program `program_identity_hash` unioned
        // across the window via `on_arrival`); an empty set (all plain-decode, or
        // hashes not supplied) ⇒ zero sampling cost ⇒ no cost-cap fire.
        let cost = Duration::from_secs_f64(
            self.merged_cost
                .estimate_s(current_forward_requests, self.distinct_programs.len()),
        );
        if cost >= self.max_accum_cost {
            return Decision::Fire;
        }
        let now = Instant::now();
        let elapsed = self
            .accumulating_since
            .map(|s| now.saturating_duration_since(s))
            .unwrap_or_default();
        // p99 latency cap reached ⇒ fire.
        if elapsed >= self.accum_window {
            return Decision::Fire;
        }
        // No arrival-rate estimate yet (cold / first request) ⇒ fire (no regression).
        if self.inter_arrival_ewma <= 0.0 {
            return Decision::Fire;
        }
        let remaining = self.accum_window - elapsed;
        let next_eta = Duration::from_secs_f64(self.inter_arrival_ewma);
        // Sparse: the next request won't arrive within the budget ⇒ no batching
        // benefit ⇒ fire now (the low-concurrency common path).
        if next_eta >= remaining {
            return Decision::Fire;
        }
        // Benefit: another request likely arrives within the window ⇒ accumulate,
        // waking at the predicted arrival (bounded by the remaining window).
        Decision::Wait(next_eta.min(remaining).max(Duration::from_micros(MIN_POLL_US)))
    }
}

impl SchedulingPolicy for RunAheadPolicy {
    fn on_arrival(&mut self, program_identity_hashes: &[u64]) {
        let now = Instant::now();
        if let Some(prev) = self.last_arrival {
            self.inter_arrival_ewma = Self::ewma(
                self.inter_arrival_ewma,
                now.saturating_duration_since(prev).as_secs_f64(),
            );
        }
        self.last_arrival = Some(now);
        // First arrival of a fresh batch opens the accumulation window.
        self.accumulating_since.get_or_insert(now);
        // Union this request's per-program identities into the window's distinct
        // set (identical programs self-dedup). Empty ⇒ plain decode (no program).
        self.distinct_programs.extend(program_identity_hashes.iter().copied());
    }

    fn on_complete(&mut self, latency: Duration) {
        self.forwards_completed += 1;
        // The first fire's latency carries cold-start costs; skip it.
        if self.forwards_completed > 1 {
            self.forward_latency = Self::ewma(self.forward_latency, latency.as_secs_f64());
        }
        // The earliest in-flight batch finished (the GPU serializes, so
        // completions arrive in fire-order); drop it from the FIFO front.
        self.in_flight.pop_front();
    }

    fn on_submitted(&mut self, submission_latency: Duration) {
        self.lead_time = Self::ewma(self.lead_time, submission_latency.as_secs_f64());
    }

    fn on_fired(&mut self, _fired_size: usize) {
        // A batch went in flight. It queues behind any in-flight batches (the GPU
        // serializes), so it completes ~forward_latency after the latest current
        // estimate (or now if none). Cold (no latency sample) -> ~now, so the
        // next batch fires promptly while we gather timing.
        let base = self
            .in_flight
            .back()
            .copied()
            .unwrap_or_else(Instant::now)
            .max(Instant::now());
        let est = if self.forward_latency > 0.0 {
            base + Duration::from_secs_f64(self.forward_latency)
        } else {
            base
        };
        self.in_flight.push_back(est);
        // The accumulated batch fired; the next batch opens a fresh window.
        self.accumulating_since = None;
        // The window's distinct programs went out with the fired batch; the next
        // window re-accumulates from empty.
        self.distinct_programs.clear();
    }

    fn distinct_program_count(&self) -> usize {
        self.distinct_programs.len()
    }

    fn decide(&mut self, current_forward_requests: usize) -> Decision {
        // One-step run-ahead cap (R10): at most `MAX_IN_FLIGHT` batches in flight
        // (one computing + one prefetched). At the cap the GPU pipe is full —
        // wait for the earliest to complete before firing more. The run loop also
        // wakes on the actual completion (the completion channel), so the estimate
        // is just a fallback poll bound.
        if self.in_flight.len() >= max_in_flight() {
            let now = Instant::now();
            let wait = self
                .in_flight
                .front()
                .map(|&f| f.saturating_duration_since(now))
                .unwrap_or_default();
            return Decision::Wait(wait.max(Duration::from_micros(MIN_POLL_US)));
        }
        // A full batch fires immediately (capacity permitting, checked above).
        if current_forward_requests >= self.max_forward_requests {
            return Decision::Fire;
        }
        // `collection_complete == now` for a non-empty batch (MVP). A batch is in
        // flight iff `in_flight` is non-empty.
        if self.in_flight.front().is_none() {
            // Nothing in flight: the run-ahead overlap term is absent. Rather than
            // fire-on-arrival (which splits concurrent requests into R=1), apply the
            // #10 adaptive accumulation window (cold/idle path — UNCHANGED).
            return self.accumulation_decision(current_forward_requests);
        }
        // EAGER SUBMIT-AHEAD (G3 bubble-p50 gate-closer, bravo's depth-2 design §3):
        // a batch is in flight AND we're below the cap → fire N+1 NOW to keep the
        // driver ring fed (N+1 device-queued before N retires; charlie's PART-2
        // back-to-backs N+1 behind N on retire). The prior JIT `completion −
        // lead_time` hold was tuned for the response-synchronous driver (firing
        // early didn't help serial processing) and left the retire→submit
        // round-trip idle (~600µs) exposed; eager fill is the latency-hide. Only
        // the STEADY-STATE decode fleet reaches here (in_flight non-empty); the
        // cold path above keeps the #10 accumulation window. WAR-safe at
        // MAX_IN_FLIGHT=2 — the single `last_eager_d2h_done` guard covers the one
        // prior un-drained fire (do NOT raise the cap without a WAR-event ring, §5).
        Decision::Fire
    }
}

// =============================================================================
// QuorumPolicy — the PTIR thrust-2 quorum fire rule (overview §7.2; F1–F6).
// =============================================================================
//
// Replaces the RA `fire_at = max(completion − lead_time, collection)` formula
// (superseded, F6) with the three-clause quorum rule. No completion estimation,
// no lead-time EWMA in the decision path — the decision is pure membership
// (`current_batch_size` vs the learned quorum denominator) plus the in-flight
// depth (R10). Gated behind the `run-ahead` feature + `PIE_SCHED_POLICY=quorum`
// (see `scheduler::quorum_policy_enabled`); the legacy policies stay selectable
// during rollout (masterplan §4, no-regression while off).
//
// Denominator (F4). The scheduler is FCFS request-oriented, so it does not track
// pipelines individually; the quorum denominator is *learned* as the recent
// steady-state cohort width from fired batch sizes (`on_fired`). This is a
// membership count, NOT a timing estimate (F6): it snaps up to a larger fleet
// immediately and decays down slowly, so a transient one-step stall never
// collapses the target (anti-bifurcation) while a genuinely smaller fleet — a
// host-blocked pipeline that stays absent — lowers it (F4: absent, not awaited).

/// Sub-millisecond cold-hold window (F3): on a true cold start, hold this long
/// for arrivals before firing a partial batch.
const COLD_HOLD_US: u64 = 500;

/// Bounded poll for the quorum-hold wait. The completion channel and new
/// arrivals both preempt the `Decision::Wait` select the instant they fire, so
/// this only bounds the worst-case re-evaluation cadence (a hang backstop).
const QUORUM_POLL_US: u64 = 200;

/// Downward decay of the learned cohort per fire (F4). Upward is instantaneous.
const COHORT_DECAY: f64 = 0.125;

pub(super) struct QuorumPolicy {
    /// Structural cap — a full batch always fires immediately.
    max_forward_requests: usize,
    /// Batches enqueued but not yet retired (one computing + one queued at the
    /// depth-1 cap, R10). Incremented on fire, decremented on completion.
    in_flight: usize,
    /// The learned quorum denominator (F4): the steady-state cohort width,
    /// upward-instant / downward-decayed from fired batch sizes.
    cohort_ewma: f64,
    /// Whether any batch has ever fired (distinguishes F3 cold start from F2
    /// mid-stream idle escape).
    ever_fired: bool,
    /// The F3 cold-hold deadline, armed on the first cold-start `decide`.
    cold_hold_deadline: Option<Instant>,
    /// Probe sink for the quorum clause counters (`escape_fires`,
    /// `cold_hold_fires`); `None` in unit tests. Written under `profile-fire`.
    stats: Option<Arc<SchedulerStats>>,
}

impl QuorumPolicy {
    pub fn new(max_forward_requests: usize, stats: Option<Arc<SchedulerStats>>) -> Self {
        Self {
            max_forward_requests,
            in_flight: 0,
            cohort_ewma: 0.0,
            ever_fired: false,
            cold_hold_deadline: None,
            stats,
        }
    }

    /// The learned quorum denominator (F4), at least 1.
    fn cohort(&self) -> usize {
        (self.cohort_ewma.round() as usize).max(1)
    }

    /// Fold a fired batch size into the cohort estimate: snap up to a larger
    /// fleet at once, decay down slowly (COHORT_DECAY/fire). Membership only.
    fn learn_cohort(&mut self, fired: usize) {
        let f = fired as f64;
        self.cohort_ewma = if f > self.cohort_ewma {
            f
        } else {
            (1.0 - COHORT_DECAY) * self.cohort_ewma + COHORT_DECAY * f
        };
    }

    fn record_clause(&self, clause: FireClause) {
        #[cfg(feature = "profile-fire")]
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            match clause {
                FireClause::IdleEscape => {
                    stats.fire.quorum.escape_fires.fetch_add(1, Relaxed);
                }
                FireClause::SubmitAhead => {
                    stats.fire.quorum.submit_ahead_fires.fetch_add(1, Relaxed);
                }
                FireClause::ColdHold => {
                    stats.fire.quorum.cold_hold_fires.fetch_add(1, Relaxed);
                }
                _ => {}
            }
        }
        let _ = (clause, &self.stats);
    }

    /// The pure decision core (F1–F3), separated so tests drive it with an
    /// explicit `now` (no wall-clock flakiness on the cold-hold window).
    fn decide_at(&mut self, current_forward_requests: usize, now: Instant) -> Decision {
        let n = current_forward_requests;
        // R10 depth-1: the pipe is full (one computing + one queued). Hold; the
        // completion channel preempts the wait the instant a batch retires.
        if self.in_flight >= max_in_flight() {
            return Decision::Wait(Duration::from_micros(QUORUM_POLL_US));
        }
        // Structural capacity cap — a full batch fires immediately.
        if n >= self.max_forward_requests {
            self.record_clause(FireClause::Quorum);
            return Decision::Fire;
        }

        if self.in_flight == 0 {
            // Device idle: firing now can't overlap an in-flight batch, but not
            // firing bubbles the device.
            let c = self.cohort();
            if n >= c {
                // Quorum already met (or exceeded) even with nothing in flight —
                // fire at once (no cold-hold wait).
                let clause = if self.ever_fired {
                    FireClause::IdleEscape
                } else {
                    FireClause::Quorum
                };
                self.record_clause(clause);
                return Decision::Fire;
            }
            if self.ever_fired {
                // F2 idle escape: mid-stream idle with a partial cohort — fire
                // the ready subset now rather than bubble; the missing pipelines
                // rejoin a later fire.
                self.record_clause(FireClause::IdleEscape);
                return Decision::Fire;
            }
            // F3 cold hold: a true cold start with a partial cohort — hold
            // sub-ms for arrivals, then fire partial.
            match self.cold_hold_deadline {
                None => {
                    self.cold_hold_deadline = Some(now + Duration::from_micros(COLD_HOLD_US));
                    Decision::Wait(Duration::from_micros(COLD_HOLD_US))
                }
                Some(deadline) if now < deadline => Decision::Wait(deadline - now),
                _ => {
                    self.record_clause(FireClause::ColdHold);
                    Decision::Fire
                }
            }
        } else {
            // A batch is in flight with room in the depth-1 queue.
            // F1 quorum: the full cohort is ready — enqueue the dense batch
            // behind the in-flight one so quorum completes mid-flight (bubble
            // zero).
            if n >= self.cohort() {
                self.record_clause(FireClause::Quorum);
                Decision::Fire
            } else {
                // Depth-2 submit-ahead (G3 bubble-p50 gate-closer, bravo's
                // design ported to the quorum path): a batch is in flight AND
                // we're below MAX_IN_FLIGHT, but the cohort is only partially
                // ready. The prior quorum-hold (`Wait` for the rest of the
                // cohort) exposed the fire-N-1-retire → response → schedule →
                // submit-N host round-trip idle (~600µs). Fire the ready subset
                // EAGERLY behind the in-flight batch instead — the run-ahead
                // carrier stream-orders N+1 behind N on the driver, so N+1
                // device-queues before N retires (bubble → 0). Only the
                // STEADY-STATE decode fleet reaches here (in_flight >= 1); the
                // cold path (in_flight == 0) above keeps the F2/F3 window
                // UNTOUCHED. WAR-safe at MAX_IN_FLIGHT=2 (one prior un-drained
                // fire; do NOT raise the cap without a WAR-event ring).
                self.record_clause(FireClause::SubmitAhead);
                Decision::Fire
            }
        }
    }
}

impl SchedulingPolicy for QuorumPolicy {
    fn on_arrival(&mut self, _program_identity_hashes: &[u64]) {}

    fn on_complete(&mut self, _latency: Duration) {
        // A batch retired — F6: latency is a probe only, never a decision input.
        self.in_flight = self.in_flight.saturating_sub(1);
    }

    fn on_fired(&mut self, fired_size: usize) {
        self.in_flight += 1;
        self.ever_fired = true;
        self.cold_hold_deadline = None;
        self.learn_cohort(fired_size);
    }

    fn decide(&mut self, current_forward_requests: usize) -> Decision {
        self.decide_at(current_forward_requests, Instant::now())
    }
}

// =============================================================================
// WaitAllPolicy — wait-for-all-active-pipelines dense-wave fire rule
// (thrust-2 rework; replaces the learned-cohort quorum).
// =============================================================================
//
// The scheduler TRACKS the active pipelines (pipeline ≡ inferlet process,
// keyed by `ProcessId`) instead of inferring a cohort width from fired batch
// sizes. A WAVE fires when every active pipeline has submitted its next
// request (its N+1); requests a pipeline ran ahead with (N+k) are stashed by
// the run loop and re-enter on the next wave. The result is a dense batch of
// the whole fleet — no singleton fires, no EWMA estimation.
//
// Liveness is explicit, not economic (three tiers, In Gim's directive):
//   1. A cancelled/killed/exited pipeline LEAVES the wait-set
//      (`on_pipeline_leave`, fed by the run loop's lifecycle events). A
//      TASK-B-preempted (suspended) pipeline leaves the same way — a frozen
//      process must never be awaited — and rejoins implicitly on its first
//      request after restore.
//   2. A straggler is bounded by the per-wave DEADLINE
//      (`PIE_SCHED_WAITALL_DEADLINE_US`, default 10ms, armed when the wave
//      starts gathering): at expiry the wave fires without it (`missing`
//      reported to the run loop — M-A2 dummy-fills those slots to keep the
//      CUDA-graph batch geometry stable).
//   3. A pipeline missing `PIE_SCHED_WAITALL_MISS_LIMIT` consecutive
//      deadline-fires (default 5) is DEMOTED from the wait-set and queued in
//      `take_terminate_candidates()` for the run loop to terminate — it can
//      no longer hold the fleet.
//
// Membership: JOIN is implicit on the first request from an unseen
// `ProcessId` (a process that never infers can never block the fleet; a
// restored/demoted process rejoins by submitting). Requests with NO pipeline
// identity (prebuilt/beam, replay) are untracked: they ride the current wave
// but are never awaited.
//
// Bootstrap: before the first fire, "all active pipelines ready" is trivially
// true when the first request arrives (the set has one member). The policy
// holds a short cold window (`COLD_HOLD_US`) so the co-launched fleet's first
// requests gather into one dense wave — the cohort=1 singleton trap cannot
// occur because membership, not fired size, defines the fleet from then on.
//
// NOTE (wiring contract with the run-loop graft): the `SchedulingPolicy`
// trait methods carry no pipeline identity, so `on_arrival` only arms the
// wave clock. The grafted run loop MUST additionally call
// `on_pipeline_request(pid)` per request entering the wave, forward lifecycle
// Leaves to `on_pipeline_leave`, and use `decide_wave_at` (the trait `decide`
// discards the `missing` list). Without the graft the policy degenerates to
// fire-on-arrival-after-cold-hold (safe, never stalls).

/// Per-wave straggler deadline default (µs): how long a wave holds for the
/// remaining active pipelines' N+1 before firing without them.
const WAITALL_DEADLINE_US_DEFAULT: u64 = 10_000;

/// Consecutive deadline-misses before a pipeline is demoted from the wait-set
/// and queued for termination.
const WAITALL_MISS_LIMIT_DEFAULT: u32 = 5;

/// The per-wave straggler deadline, read once from
/// `PIE_SCHED_WAITALL_DEADLINE_US` (default 10ms, floored at 1µs).
fn waitall_deadline() -> Duration {
    static DEADLINE: OnceLock<Duration> = OnceLock::new();
    *DEADLINE.get_or_init(|| {
        std::env::var("PIE_SCHED_WAITALL_DEADLINE_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&us| us >= 1)
            .map(Duration::from_micros)
            .unwrap_or(Duration::from_micros(WAITALL_DEADLINE_US_DEFAULT))
    })
}

/// The consecutive-miss demotion limit, read once from
/// `PIE_SCHED_WAITALL_MISS_LIMIT` (default 5, floored at 1).
fn waitall_miss_limit() -> u32 {
    static LIMIT: OnceLock<u32> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("PIE_SCHED_WAITALL_MISS_LIMIT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(WAITALL_MISS_LIMIT_DEFAULT)
    })
}

/// The wave-level fire decision. `Fire::missing` is the pipelines that missed
/// the wave (deadline fire; pre-demotion) — the run loop dummy-fills their
/// slots (M-A2) so batch geometry stays stable. Empty on a dense fire.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum WaveDecision {
    Fire { missing: Vec<ProcessId> },
    Wait(Duration),
}

/// Per-pipeline wave participation.
#[derive(Debug, Default)]
struct PipelineWaveState {
    /// Requests this pipeline has contributed to the CURRENT wave.
    /// `> 0` ⇒ ready (its N+1 is in).
    wave_ready: usize,
    /// Consecutive deadline-fires this pipeline missed. Reset on
    /// participation; demotes at `waitall_miss_limit()`.
    consecutive_misses: u32,
}

#[allow(dead_code)] // wired by the run-loop graft (delta's plumbing, M-A1).
pub(super) struct WaitAllPolicy {
    /// Structural cap — a full batch always fires immediately.
    max_forward_requests: usize,
    /// Batches enqueued but not yet retired; bounded by `max_in_flight()`.
    in_flight: usize,
    /// THE wait-set: every active pipeline and its wave participation.
    /// BTreeMap for deterministic `missing` ordering.
    active: BTreeMap<ProcessId, PipelineWaveState>,
    /// Untracked (pipeline-less) requests in the current wave: counted so an
    /// untracked-only batch still fires, never awaited.
    untracked_ready: usize,
    /// When the current wave started gathering (first request after the
    /// previous fire) — arms the straggler deadline.
    wave_started: Option<Instant>,
    /// Whether any wave has fired (bootstrap discriminator).
    ever_fired: bool,
    /// The bootstrap gather window deadline (armed on the first cold decide).
    cold_hold_deadline: Option<Instant>,
    /// Pipelines demoted at the miss limit, awaiting the run loop's
    /// `process::terminate` (M-A2). Drained by `take_terminate_candidates`.
    terminate_candidates: Vec<ProcessId>,
    /// Probe sink (`profile-fire`); `None` in unit tests.
    stats: Option<Arc<SchedulerStats>>,
}

#[allow(dead_code)] // wired by the run-loop graft (delta's plumbing, M-A1).
impl WaitAllPolicy {
    pub fn new(max_forward_requests: usize, stats: Option<Arc<SchedulerStats>>) -> Self {
        Self {
            max_forward_requests,
            in_flight: 0,
            active: BTreeMap::new(),
            untracked_ready: 0,
            wave_started: None,
            ever_fired: false,
            cold_hold_deadline: None,
            terminate_candidates: Vec::new(),
            stats,
        }
    }

    /// A request entered the current wave. `Some(pid)` joins the wait-set
    /// implicitly on first sight and marks the pipeline ready; `None` is
    /// untracked (prebuilt/beam/replay — rides the wave, never awaited).
    pub fn on_pipeline_request(&mut self, pid: Option<ProcessId>, now: Instant) {
        if self.wave_started.is_none() {
            self.wave_started = Some(now);
        }
        match pid {
            Some(pid) => {
                self.active.entry(pid).or_default().wave_ready += 1;
            }
            None => self.untracked_ready += 1,
        }
    }

    /// A pipeline left the fleet (cancel / kill / exit / TASK-A terminate /
    /// TASK-B preempt). Its requests already in the wave ride along as
    /// untracked; it is no longer awaited. Rejoin is implicit on its next
    /// request (post-restore, or a demoted pipeline resubmitting).
    pub fn on_pipeline_leave(&mut self, pid: ProcessId) {
        if let Some(state) = self.active.remove(&pid) {
            self.untracked_ready += state.wave_ready;
        }
    }

    /// Pipelines demoted at the miss limit since the last call. The run loop
    /// terminates these (M-A2); already removed from the wait-set here.
    pub fn take_terminate_candidates(&mut self) -> Vec<ProcessId> {
        std::mem::take(&mut self.terminate_candidates)
    }

    /// The wait-set size (probe/test accessor).
    pub fn active_pipelines(&self) -> usize {
        self.active.len()
    }

    /// Whether `pid` is currently awaited (probe/test accessor).
    pub fn is_active(&self, pid: ProcessId) -> bool {
        self.active.contains_key(&pid)
    }

    /// `pid`'s consecutive deadline-miss count (probe/test accessor).
    pub fn misses(&self, pid: ProcessId) -> Option<u32> {
        self.active.get(&pid).map(|s| s.consecutive_misses)
    }

    /// The pure wave decision, driven with an explicit `now` (test-stable).
    ///
    /// Order: depth cap → capacity fire → empty guard → the wave barrier
    /// (all-ready dense fire / bootstrap gather / straggler deadline).
    pub fn decide_wave_at(&mut self, current_batch_size: usize, now: Instant) -> WaveDecision {
        // Run-ahead depth cap (R10): the pipe is full — hold; the completion
        // channel preempts the wait the instant a batch retires. Misses are
        // NOT counted while capped: a straggler can't be blamed for a wave
        // that couldn't fire anyway.
        if self.in_flight >= max_in_flight() {
            return WaveDecision::Wait(Duration::from_micros(QUORUM_POLL_US));
        }
        // Structural capacity cap — a full batch fires immediately, no miss
        // penalty (the wave didn't run out of patience; it ran out of room).
        if current_batch_size >= self.max_forward_requests {
            self.record_clause(FireClause::Quorum);
            return WaveDecision::Fire {
                missing: Vec::new(),
            };
        }
        // Nothing to fire (defensive: the run loop decides on non-empty
        // batches). No miss accrual on an empty wave — if the whole fleet is
        // between requests, nobody is holding anybody.
        if current_batch_size == 0 {
            return WaveDecision::Wait(Duration::from_micros(QUORUM_POLL_US));
        }

        let missing: Vec<ProcessId> = self
            .active
            .iter()
            .filter(|(_, s)| s.wave_ready == 0)
            .map(|(&pid, _)| pid)
            .collect();

        if missing.is_empty() {
            // Every active pipeline is in (or the batch is untracked-only).
            if self.ever_fired {
                // Dense wave — the steady-state fire.
                self.record_clause(FireClause::Quorum);
                return WaveDecision::Fire {
                    missing: Vec::new(),
                };
            }
            // Bootstrap: membership is still forming (the wait-set has only
            // the pipelines that already submitted), so "all ready" is
            // trivially true. Hold the cold window for the co-launched
            // fleet's first requests to gather into one dense wave.
            match self.cold_hold_deadline {
                None => {
                    let window = Duration::from_micros(COLD_HOLD_US);
                    self.cold_hold_deadline = Some(now + window);
                    return WaveDecision::Wait(window);
                }
                Some(deadline) if now < deadline => {
                    return WaveDecision::Wait(deadline - now);
                }
                _ => {
                    self.record_clause(FireClause::ColdHold);
                    return WaveDecision::Fire {
                        missing: Vec::new(),
                    };
                }
            }
        }

        // Stragglers outstanding: hold for them up to the wave deadline.
        let deadline = self
            .wave_started
            .map(|t| t + waitall_deadline())
            .unwrap_or(now);
        if now < deadline {
            return WaveDecision::Wait(deadline - now);
        }

        // Deadline expired — fire without the stragglers. Count the miss and
        // demote any pipeline that has now missed too many consecutive waves
        // (it stops holding the fleet; the run loop terminates it, M-A2).
        let limit = waitall_miss_limit();
        for pid in &missing {
            let demote = match self.active.get_mut(pid) {
                Some(state) => {
                    state.consecutive_misses += 1;
                    state.consecutive_misses >= limit
                }
                None => false,
            };
            if demote {
                self.active.remove(pid);
                self.terminate_candidates.push(*pid);
            }
        }
        self.record_clause(FireClause::IdleEscape);
        WaveDecision::Fire { missing }
    }

    fn record_clause(&self, clause: FireClause) {
        #[cfg(feature = "profile-fire")]
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            match clause {
                // Dense/capacity waves count as quorum fires; deadline fires
                // as escapes; bootstrap gathers as cold holds — reusing the
                // quorum probe counters so existing dumps keep reading.
                FireClause::IdleEscape => {
                    stats.fire.quorum.escape_fires.fetch_add(1, Relaxed);
                }
                FireClause::ColdHold => {
                    stats.fire.quorum.cold_hold_fires.fetch_add(1, Relaxed);
                }
                _ => {}
            }
        }
        let _ = (clause, &self.stats);
    }
}

impl SchedulingPolicy for WaitAllPolicy {
    fn on_arrival(&mut self, _program_identity_hashes: &[u64]) {
        // Identity-less trait hook: arm the wave clock only. The grafted run
        // loop supplies identity via `on_pipeline_request` (see module note).
        if self.wave_started.is_none() {
            self.wave_started = Some(Instant::now());
        }
    }

    fn on_complete(&mut self, _latency: Duration) {
        self.in_flight = self.in_flight.saturating_sub(1);
    }

    fn on_fired(&mut self, _fired_size: usize) {
        self.in_flight += 1;
        self.ever_fired = true;
        self.cold_hold_deadline = None;
        self.wave_started = None;
        self.untracked_ready = 0;
        // Participation resets the straggler count; absentees keep theirs
        // (already incremented on the deadline path that fired this wave).
        for state in self.active.values_mut() {
            if state.wave_ready > 0 {
                state.consecutive_misses = 0;
                state.wave_ready = 0;
            }
        }
    }

    fn decide(&mut self, current_batch_size: usize) -> Decision {
        match self.decide_wave_at(current_batch_size, Instant::now()) {
            // The trait surface can't carry `missing`; the grafted run loop
            // calls `decide_wave_at` directly for the dummy-fill list.
            WaveDecision::Fire { .. } => Decision::Fire,
            WaveDecision::Wait(d) => Decision::Wait(d),
        }
    }
}

#[cfg(test)]
mod quorum_policy_tests {
    use super::*;

    fn policy(max: usize) -> QuorumPolicy {
        QuorumPolicy::new(max, None)
    }

    /// Cold start with the whole cohort already collected fires immediately —
    /// no cold-hold wait (quorum is met even with nothing in flight).
    #[test]
    fn cold_start_full_cohort_fires_immediately() {
        let mut p = policy(64);
        // Cohort defaults to 1; a lone ready request meets it.
        assert!(matches!(p.decide(1), Decision::Fire));
    }

    /// Cold start with a partial cohort holds sub-ms, then fires partial.
    #[test]
    fn cold_start_partial_holds_then_fires() {
        let mut p = policy(64);
        p.cohort_ewma = 8.0; // pretend a cohort of 8 is expected
        let t0 = Instant::now();
        // First cold decide arms the hold window.
        assert!(matches!(p.decide_at(3, t0), Decision::Wait(_)));
        // Still within the window → keep holding.
        assert!(matches!(
            p.decide_at(3, t0 + Duration::from_micros(100)),
            Decision::Wait(_)
        ));
        // Past the window → fire the partial batch (F3).
        assert!(matches!(
            p.decide_at(3, t0 + Duration::from_micros(COLD_HOLD_US + 1)),
            Decision::Fire
        ));
    }

    /// Depth-2 submit-ahead (G3 bubble): with a batch in flight and room below
    /// the cap, a PARTIAL cohort fires EAGERLY behind the in-flight batch rather
    /// than holding for the full cohort (the prior JIT/quorum hold exposed the
    /// retire→submit round-trip idle); a full cohort fires via quorum. Both
    /// enqueue-ahead behind the in-flight batch (the carrier orders N+1 → N).
    #[test]
    fn in_flight_below_cap_fires_eagerly() {
        let mut p = policy(64);
        p.on_fired(8); // learns cohort 8, in_flight = 1 (below MAX_IN_FLIGHT=2)
        assert!(
            matches!(p.decide(7), Decision::Fire),
            "7 < 8 but a batch is in flight below the cap → eager submit-ahead"
        );
        assert!(
            matches!(p.decide(8), Decision::Fire),
            "8 == cohort: quorum fire"
        );
    }

    /// Depth-1 cap (R10): at MAX_IN_FLIGHT the pipe is full — hold regardless of
    /// how ready the batch is.
    #[test]
    fn depth_one_cap_holds() {
        let mut p = policy(64);
        p.on_fired(8); // in_flight = 1
        p.on_fired(8); // in_flight = 2 == MAX_IN_FLIGHT
        assert!(matches!(p.decide(8), Decision::Wait(_)));
        p.on_complete(Duration::from_millis(1)); // in_flight = 1
        assert!(matches!(p.decide(8), Decision::Fire), "room again → quorum");
    }

    /// Mid-stream idle escape (F2): the device drained (nothing in flight) with
    /// a partial cohort — fire the ready subset now rather than bubble.
    #[test]
    fn idle_escape_fires_partial_when_device_drained() {
        let mut p = policy(64);
        p.on_fired(8); // cohort 8, in_flight = 1
        p.on_complete(Duration::from_millis(1)); // in_flight = 0, ever_fired
        // Only 5 ready, but the device is idle → escape rather than hold.
        assert!(matches!(p.decide(5), Decision::Fire));
    }

    /// Convoy anti-bifurcation: a one-step stall dents the batch by one, but the
    /// learned cohort does NOT collapse — the fleet re-converges to full-batch
    /// fires next round instead of settling into alternating half-batches.
    #[test]
    fn convoy_anti_bifurcation_cohort_does_not_collapse() {
        let mut p = policy(64);
        // Steady state: several full fires of 8 → cohort 8.
        for _ in 0..4 {
            p.on_fired(8);
            p.on_complete(Duration::from_millis(1));
        }
        assert_eq!(p.cohort(), 8);
        // One-step stall: a single fire of 7 (a pipeline briefly host-blocked).
        p.on_fired(7);
        p.on_complete(Duration::from_millis(1));
        assert_eq!(p.cohort(), 8, "one dip must not collapse the quorum target");
        // The stalled pipeline returns → the full cohort is ready again → quorum
        // fires 8 (not a bifurcated half-batch).
        assert!(matches!(p.decide(8), Decision::Fire));
    }

    /// F4 denominator excludes a persistently host-blocked pipeline: if the
    /// fleet stays at 7, the learned cohort decays to 7 so quorum stops waiting
    /// for the absent pipeline.
    #[test]
    fn denominator_adapts_to_persistently_smaller_fleet() {
        let mut p = policy(64);
        p.on_fired(8); // cohort 8
        p.on_complete(Duration::from_millis(1)); // retire it → in_flight = 0
        // The 8th pipeline goes host-blocked and stays absent; the fleet fires 7.
        for _ in 0..40 {
            p.on_fired(7);
            p.on_complete(Duration::from_millis(1));
        }
        assert_eq!(p.cohort(), 7, "cohort decays to the real fleet width (F4)");
        // With a batch in flight, 7 now meets quorum directly — no escape needed.
        p.on_fired(7);
        assert!(matches!(p.decide(7), Decision::Fire));
    }

    /// Laggard rejoin: after firing a dented batch, the laggard's next submit
    /// brings the cohort back to full and the next fire is full-width.
    #[test]
    fn laggard_rejoins_next_fire() {
        let mut p = policy(64);
        p.on_fired(8);
        p.on_complete(Duration::from_millis(1));
        // Dented fire (laggard absent) via idle escape.
        p.on_fired(7);
        // Laggard rejoined: 8 ready again with a batch in flight → quorum fires.
        assert!(matches!(p.decide(8), Decision::Fire));
        assert_eq!(p.cohort(), 8);
    }
}

#[cfg(test)]
mod waitall_policy_tests {
    use super::*;

    fn wp() -> WaitAllPolicy {
        WaitAllPolicy::new(64, None)
    }

    fn pid() -> ProcessId {
        ProcessId::new_v4()
    }

    const MS: Duration = Duration::from_millis(1);

    /// Warm a policy through its bootstrap wave: `pids` all submit, the cold
    /// window elapses, the dense wave fires and retires. Returns the instant
    /// after the fire.
    fn warm(p: &mut WaitAllPolicy, pids: &[ProcessId], t0: Instant) -> Instant {
        for &id in pids {
            p.on_pipeline_request(Some(id), t0);
        }
        assert!(matches!(
            p.decide_wave_at(pids.len(), t0),
            WaveDecision::Wait(_)
        ));
        let t1 = t0 + Duration::from_micros(COLD_HOLD_US + 1);
        assert_eq!(
            p.decide_wave_at(pids.len(), t1),
            WaveDecision::Fire { missing: vec![] }
        );
        p.on_fired(pids.len());
        p.on_complete(MS);
        t1
    }

    /// Bootstrap: the first arrival does NOT fire a singleton wave — the cold
    /// window gathers the co-launched fleet into one dense fire.
    #[test]
    fn bootstrap_gathers_fleet_then_fires_dense() {
        let mut p = wp();
        let t0 = Instant::now();
        let a = pid();
        p.on_pipeline_request(Some(a), t0);
        // First arrival holds (membership still forming — no singleton fire).
        assert!(matches!(p.decide_wave_at(1, t0), WaveDecision::Wait(_)));
        // Seven more pipelines land during the window.
        let fleet: Vec<ProcessId> = (0..7).map(|_| pid()).collect();
        for &id in &fleet {
            p.on_pipeline_request(Some(id), t0 + Duration::from_micros(100));
        }
        // Window elapses → ONE dense wave of all 8.
        assert_eq!(
            p.decide_wave_at(8, t0 + Duration::from_micros(COLD_HOLD_US + 1)),
            WaveDecision::Fire { missing: vec![] }
        );
        p.on_fired(8);
        assert_eq!(p.active_pipelines(), 8);
    }

    /// The core wait-for-all barrier: a warm wave HOLDS while any active
    /// pipeline's N+1 is outstanding, and fires the instant the last one
    /// lands — no cohort estimation, no partial fire before the deadline.
    #[test]
    fn wave_waits_for_all_active_pipelines() {
        let mut p = wp();
        let t0 = Instant::now();
        let (a, b, c) = (pid(), pid(), pid());
        let t1 = warm(&mut p, &[a, b, c], t0);

        // Wave 2: a and b resubmit; c is still mid-round-trip.
        p.on_pipeline_request(Some(a), t1);
        p.on_pipeline_request(Some(b), t1);
        assert!(
            matches!(p.decide_wave_at(2, t1), WaveDecision::Wait(_)),
            "hold: c's N+1 outstanding"
        );
        // c lands → dense fire immediately (no residual hold).
        p.on_pipeline_request(Some(c), t1 + MS);
        assert_eq!(
            p.decide_wave_at(3, t1 + MS),
            WaveDecision::Fire { missing: vec![] }
        );
    }

    /// A wave fires behind an in-flight batch (depth-2 ride-along preserved):
    /// the barrier is orthogonal to the in-flight depth below the cap.
    #[test]
    fn dense_wave_fires_behind_in_flight_batch() {
        let mut p = wp();
        let t0 = Instant::now();
        let (a, b) = (pid(), pid());
        let t1 = warm(&mut p, &[a, b], t0);
        p.on_fired(2); // wave 1 re-fired: in_flight = 1 (below the cap of 2)
        p.on_pipeline_request(Some(a), t1);
        p.on_pipeline_request(Some(b), t1);
        assert_eq!(
            p.decide_wave_at(2, t1),
            WaveDecision::Fire { missing: vec![] },
            "dense wave enqueues behind the in-flight batch"
        );
    }

    /// Depth cap (R10): at MAX_IN_FLIGHT the pipe is full — hold regardless of
    /// readiness, and accrue NO misses while capped.
    #[test]
    fn depth_cap_holds_without_miss_accrual() {
        let mut p = wp();
        let t0 = Instant::now();
        let (a, b) = (pid(), pid());
        let t1 = warm(&mut p, &[a, b], t0);
        p.on_fired(2);
        p.on_fired(2); // in_flight = 2 == MAX_IN_FLIGHT
        p.on_pipeline_request(Some(a), t1);
        // Way past the deadline, but the pipe is full → Wait, no miss for b.
        let late = t1 + Duration::from_millis(100);
        assert!(matches!(p.decide_wave_at(1, late), WaveDecision::Wait(_)));
        assert_eq!(p.misses(b), Some(0), "no blame while the device is capped");
        p.on_complete(MS); // room again → the deadline path may now fire
        assert_eq!(
            p.decide_wave_at(1, late),
            WaveDecision::Fire { missing: vec![b] }
        );
        assert_eq!(p.misses(b), Some(1), "one wave missed ⇒ one miss");
    }

    /// Straggler deadline: past `waitall_deadline()` the wave fires without
    /// the straggler, reporting it in `missing` (M-A2 dummy-fills that slot),
    /// and its consecutive-miss count increments.
    #[test]
    fn deadline_fires_partial_and_counts_miss() {
        let mut p = wp();
        let t0 = Instant::now();
        let (a, b) = (pid(), pid());
        let t1 = warm(&mut p, &[a, b], t0);

        p.on_pipeline_request(Some(a), t1);
        // Inside the window: hold for b.
        assert!(matches!(
            p.decide_wave_at(1, t1 + waitall_deadline() / 2),
            WaveDecision::Wait(_)
        ));
        // Past the window: fire without b.
        let late = t1 + waitall_deadline() + Duration::from_micros(1);
        assert_eq!(
            p.decide_wave_at(1, late),
            WaveDecision::Fire { missing: vec![b] }
        );
        assert_eq!(p.misses(b), Some(1));
        assert!(p.is_active(b), "one miss does not demote");
    }

    /// Capacity fire: a full batch fires immediately with no miss penalty —
    /// the wave ran out of room, not patience.
    #[test]
    fn capacity_fire_no_miss_penalty() {
        let mut p = WaitAllPolicy::new(4, None);
        let t0 = Instant::now();
        let (a, b) = (pid(), pid());
        let t1 = warm(&mut p, &[a, b], t0);
        for _ in 0..4 {
            p.on_pipeline_request(Some(a), t1); // a runs the batch to the cap
        }
        assert_eq!(
            p.decide_wave_at(4, t1),
            WaveDecision::Fire { missing: vec![] }
        );
        assert_eq!(p.misses(b), Some(0), "capacity fire never blames stragglers");
    }

    /// Miss limit: `WAITALL_MISS_LIMIT` consecutive deadline-misses demote the
    /// pipeline from the wait-set and queue it for termination; the fleet
    /// stops waiting on it from the next wave.
    #[test]
    fn miss_limit_demotes_and_queues_terminate() {
        let mut p = wp();
        let t0 = Instant::now();
        let (a, b) = (pid(), pid());
        let mut t = warm(&mut p, &[a, b], t0);

        let limit = waitall_miss_limit();
        for i in 0..limit {
            p.on_pipeline_request(Some(a), t);
            let late = t + waitall_deadline() + Duration::from_micros(1);
            assert_eq!(
                p.decide_wave_at(1, late),
                WaveDecision::Fire { missing: vec![b] },
                "deadline fire {i}"
            );
            p.on_fired(1);
            p.on_complete(MS);
            t = late + MS;
        }
        assert!(!p.is_active(b), "demoted at the limit");
        assert_eq!(p.take_terminate_candidates(), vec![b]);
        assert!(p.take_terminate_candidates().is_empty(), "drained once");

        // The fleet no longer waits on b: a alone is a dense wave.
        p.on_pipeline_request(Some(a), t);
        assert_eq!(
            p.decide_wave_at(1, t),
            WaveDecision::Fire { missing: vec![] }
        );
    }

    /// Participation resets the consecutive-miss count — only an unbroken run
    /// of missed waves demotes.
    #[test]
    fn participation_resets_misses() {
        let mut p = wp();
        let t0 = Instant::now();
        let (a, b) = (pid(), pid());
        let t1 = warm(&mut p, &[a, b], t0);

        // b misses one wave...
        p.on_pipeline_request(Some(a), t1);
        let late = t1 + waitall_deadline() + Duration::from_micros(1);
        assert_eq!(
            p.decide_wave_at(1, late),
            WaveDecision::Fire { missing: vec![b] }
        );
        p.on_fired(1);
        p.on_complete(MS);
        assert_eq!(p.misses(b), Some(1));

        // ...then makes the next wave → the count resets on fire.
        let t2 = late + MS;
        p.on_pipeline_request(Some(a), t2);
        p.on_pipeline_request(Some(b), t2);
        assert_eq!(
            p.decide_wave_at(2, t2),
            WaveDecision::Fire { missing: vec![] }
        );
        p.on_fired(2);
        assert_eq!(p.misses(b), Some(0), "participation clears the streak");
    }

    /// Leave (cancel / kill / exit / TASK-B preempt): the pipeline stops being
    /// awaited immediately; requests it already contributed ride the wave as
    /// untracked. Rejoin is implicit on its next request (post-restore).
    #[test]
    fn leave_removes_from_wait_set_and_rejoin_is_implicit() {
        let mut p = wp();
        let t0 = Instant::now();
        let (a, b) = (pid(), pid());
        let t1 = warm(&mut p, &[a, b], t0);

        // Wave 2: a submits, b is preempted/cancelled before resubmitting.
        p.on_pipeline_request(Some(a), t1);
        assert!(matches!(p.decide_wave_at(1, t1), WaveDecision::Wait(_)));
        p.on_pipeline_leave(b);
        assert_eq!(
            p.decide_wave_at(1, t1),
            WaveDecision::Fire { missing: vec![] },
            "a frozen pipeline must not hold the fleet"
        );
        p.on_fired(1);
        p.on_complete(MS);
        assert!(!p.is_active(b));

        // b restored → its first request rejoins the wait-set.
        let t2 = t1 + MS;
        p.on_pipeline_request(Some(b), t2);
        assert!(p.is_active(b));
        assert_eq!(p.misses(b), Some(0), "fresh state on rejoin");
        // And the fleet now waits for a again.
        assert!(matches!(p.decide_wave_at(1, t2), WaveDecision::Wait(_)));
    }

    /// A leaver's already-contributed requests still fire (moved to the
    /// untracked count — the batch content is untouched).
    #[test]
    fn leaver_requests_ride_as_untracked() {
        let mut p = wp();
        let t0 = Instant::now();
        let (a, b) = (pid(), pid());
        let t1 = warm(&mut p, &[a, b], t0);

        p.on_pipeline_request(Some(a), t1);
        p.on_pipeline_request(Some(b), t1);
        p.on_pipeline_leave(b); // cancel lands after its submit
        assert_eq!(
            p.decide_wave_at(2, t1),
            WaveDecision::Fire { missing: vec![] },
            "b's request fires with the wave; b is just not awaited"
        );
    }

    /// The cuda_bubble M-AB ① timeline (charlie's 4090 repro shape): 8
    /// pipelines with a ~5.4ms round-trip, first submissions staggered inside
    /// one round-trip, each resubmitting ~5.4ms after its wave fires. The
    /// barrier must CONVERGE: the bootstrap may fire the lone early pipeline,
    /// but wave 2 gathers the whole fleet (the 10ms deadline absorbs the
    /// stagger) and every steady wave is dense 8-wide. If this passes while
    /// the device shows mean_batch=1.00, the defect is in the run-loop feed
    /// or a fire-forcing bypass (e.g. the `next_pending` gate), NOT the
    /// policy — this is the exoneration regression for that diagnosis.
    #[test]
    fn staggered_fleet_converges_to_dense_waves() {
        let mut p = wp();
        let t0 = Instant::now();
        let round_trip = Duration::from_micros(5400);
        let stagger = Duration::from_micros(600); // 8 spread inside one round-trip
        let fleet: Vec<ProcessId> = (0..8).map(|_| pid()).collect();

        // Pipeline 0 launches first; the 500µs bootstrap window elapses
        // before anyone else arrives → a singleton bootstrap fire.
        p.on_pipeline_request(Some(fleet[0]), t0);
        assert!(matches!(p.decide_wave_at(1, t0), WaveDecision::Wait(_)));
        let t_boot = t0 + Duration::from_micros(COLD_HOLD_US + 1);
        assert_eq!(
            p.decide_wave_at(1, t_boot),
            WaveDecision::Fire { missing: vec![] }
        );
        p.on_fired(1);
        p.on_complete(MS);

        // Wave 2 — the GATHER: pipelines 1..7 join staggered; pipeline 0
        // resubmits a round-trip after its fire. Every intermediate decide
        // must HOLD (pipeline 0 is active but mid-flight).
        let mut queued = 0usize;
        for (i, &id) in fleet.iter().enumerate().skip(1) {
            let at = t0 + stagger * i as u32;
            p.on_pipeline_request(Some(id), at);
            queued += 1;
            assert!(
                matches!(p.decide_wave_at(queued, at), WaveDecision::Wait(_)),
                "wave 2 must hold for pipeline 0's resubmit (i={i})"
            );
        }
        let t_resub = t_boot + round_trip;
        p.on_pipeline_request(Some(fleet[0]), t_resub);
        queued += 1;
        assert_eq!(
            p.decide_wave_at(queued, t_resub),
            WaveDecision::Fire { missing: vec![] },
            "wave 2 gathers the WHOLE fleet within the deadline"
        );
        assert_eq!(queued, 8, "dense 8-wide wave");
        p.on_fired(8);
        p.on_complete(MS);

        // Steady state: three more rounds — all 8 resubmit staggered inside
        // one round-trip; each wave holds until the last arrival, then fires
        // dense. No deadline (partial) fire ever happens.
        let mut wave_start = t_resub;
        for round in 0..3 {
            let mut queued = 0usize;
            for (i, &id) in fleet.iter().enumerate() {
                let at = wave_start + round_trip + stagger * i as u32;
                p.on_pipeline_request(Some(id), at);
                queued += 1;
                let d = p.decide_wave_at(queued, at);
                if i < 7 {
                    assert!(
                        matches!(d, WaveDecision::Wait(_)),
                        "round {round}: hold at {queued}/8"
                    );
                } else {
                    assert_eq!(
                        d,
                        WaveDecision::Fire { missing: vec![] },
                        "round {round}: dense fire at 8/8"
                    );
                }
            }
            p.on_fired(8);
            p.on_complete(MS);
            wave_start = wave_start + round_trip + stagger * 7;
            for s in fleet.iter().map(|&id| p.misses(id)) {
                assert_eq!(s, Some(0), "no misses in steady dense waves");
            }
        }
        assert_eq!(p.active_pipelines(), 8);
    }

    /// Untracked-only traffic (prebuilt/beam/replay, or pre-graft trait-only
    /// wiring) never stalls: with no active pipelines the wave fires as soon
    /// as it's warm.
    #[test]
    fn untracked_only_batch_fires() {
        let mut p = wp();
        let t0 = Instant::now();
        // Bootstrap with one untracked request: cold-hold then fire.
        p.on_pipeline_request(None, t0);
        assert!(matches!(p.decide_wave_at(1, t0), WaveDecision::Wait(_)));
        let t1 = t0 + Duration::from_micros(COLD_HOLD_US + 1);
        assert_eq!(
            p.decide_wave_at(1, t1),
            WaveDecision::Fire { missing: vec![] }
        );
        p.on_fired(1);
        p.on_complete(MS);
        // Warm untracked requests fire immediately.
        p.on_pipeline_request(None, t1);
        assert_eq!(
            p.decide_wave_at(1, t1),
            WaveDecision::Fire { missing: vec![] }
        );
        assert_eq!(p.active_pipelines(), 0);
    }
}

#[cfg(test)]
mod run_ahead_tests {
    use super::*;

    #[test]
    fn run_ahead_cold_start_fires() {
        // No completion observed yet -> no in-flight estimate -> fire to make
        // progress (degrades to greedy under the synchronous fire path).
        let mut policy = RunAheadPolicy::new(512);
        policy.on_fired(1); // cold: forward_latency==0 -> in_flight_completion None
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn run_ahead_fires_at_structural_cap() {
        let mut policy = RunAheadPolicy::new(512);
        policy.on_complete(Duration::from_millis(500)); // skipped (cold)
        policy.on_complete(Duration::from_millis(50));
        policy.on_fired(64);
        // Full batch fires unconditionally, ignoring the timing rule.
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    #[test]
    fn run_ahead_fires_eagerly_when_in_flight_far() {
        // Depth-2 submit-ahead (G3 bubble gate-closer): once a batch is in flight
        // AND we're below MAX_IN_FLIGHT, fire N+1 EAGERLY to keep the driver ring
        // fed — do NOT JIT-hold for `completion − lead_time`, even when the
        // in-flight batch is ~1s out. The prior JIT hold exposed the
        // retire→submit round-trip idle; eager fill hides it.
        let mut policy = RunAheadPolicy::new(512);
        policy.on_complete(Duration::from_millis(10)); // skipped
        policy.on_complete(Duration::from_secs(1)); // forward_latency = 1s
        policy.on_submitted(Duration::from_millis(5)); // lead_time = 5ms
        policy.on_fired(1); // in_flight_completion ~= now + 1s
        assert!(
            matches!(policy.decide(1), Decision::Fire),
            "eager submit-ahead fires N+1 while N runs (below the cap), no JIT hold"
        );
    }

    #[test]
    fn run_ahead_fires_when_lead_exceeds_remaining() {
        // Lead time >= the in-flight batch's remaining time -> fire now to keep
        // the pipe full (enqueue takes longer than what's left).
        let mut policy = RunAheadPolicy::new(512);
        policy.on_complete(Duration::from_millis(10)); // skipped
        policy.on_complete(Duration::from_millis(10)); // forward_latency = 10ms
        policy.on_submitted(Duration::from_secs(1)); // lead_time = 1s (>> 10ms)
        policy.on_fired(1); // in_flight_completion ~= now + 10ms
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn run_ahead_fires_when_nothing_in_flight() {
        // After a completion clears the in-flight term, the next decide fires
        // immediately (collection dominates) — the synchronous-path behaviour.
        let mut policy = RunAheadPolicy::new(512);
        policy.on_complete(Duration::from_millis(10)); // skipped
        policy.on_complete(Duration::from_secs(1)); // forward_latency = 1s
        policy.on_fired(1); // in flight
        policy.on_complete(Duration::from_secs(1)); // completes -> clears in_flight
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn run_ahead_caps_at_one_step() {
        // Two batches in flight = the one-step cap (R10): the GPU pipe is full,
        // so even a structurally-full batch waits until the earliest completes.
        let mut policy = RunAheadPolicy::new(512);
        policy.on_complete(Duration::from_millis(10)); // skipped
        policy.on_complete(Duration::from_secs(1)); // forward_latency = 1s
        policy.on_fired(1); // in_flight = [~now+1s]
        policy.on_fired(1); // in_flight = [~now+1s, ~now+2s] -> at cap
        assert!(matches!(policy.decide(512), Decision::Wait(_)));
        policy.on_complete(Duration::from_secs(1)); // earliest completes -> pops front
        // Capacity freed -> a full batch fires.
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    // ── #10 adaptive accumulation window ────────────────────────────────────

    #[test]
    fn accum_window_disabled_fires_on_arrival() {
        // Default (window ZERO) ⇒ no accumulation ⇒ today's fire-on-arrival.
        let mut p = RunAheadPolicy::new(512);
        p.accum_window = Duration::ZERO;
        p.on_arrival(&[]);
        assert!(matches!(p.decide(1), Decision::Fire));
    }

    #[test]
    fn accum_window_low_concurrency_fires_immediately() {
        // The common-path no-regression guarantee: cold (no estimate) and sparse
        // (inter-arrival >> window) both fire immediately, no added latency.
        let mut p = RunAheadPolicy::new(512);
        p.accum_window = Duration::from_millis(5);
        p.accumulating_since = Some(Instant::now());
        // cold: no arrival-rate estimate yet ⇒ fire
        assert!(matches!(p.decide(1), Decision::Fire));
        // sparse: 1s inter-arrival ⇒ no request within the 5ms budget ⇒ fire
        p.inter_arrival_ewma = 1.0;
        assert!(matches!(p.decide(1), Decision::Fire));
    }

    #[test]
    fn accum_window_high_concurrency_accumulates() {
        // Dense arrivals (inter-arrival << window) + window open ⇒ Wait to co-batch.
        let mut p = RunAheadPolicy::new(512);
        p.accum_window = Duration::from_millis(50);
        p.inter_arrival_ewma = 0.0005; // 0.5ms << 50ms window
        p.accumulating_since = Some(Instant::now());
        assert!(matches!(p.decide(2), Decision::Wait(_)));
    }

    #[test]
    fn accum_window_capacity_cap_fires() {
        // A full batch fires immediately despite dense arrivals.
        let mut p = RunAheadPolicy::new(4);
        p.accum_window = Duration::from_millis(50);
        p.inter_arrival_ewma = 0.0005;
        p.accumulating_since = Some(Instant::now());
        assert!(matches!(p.decide(4), Decision::Fire));
    }

    #[test]
    fn accum_window_cost_cap_fires() {
        // Distinct-program compile-cost ceiling (DECOUPLED from the latency
        // window): each DISTINCT program pays the finalize residual (~5ms) + the
        // M-batched fire (~58µs); fire once the batch's estimated distinct cost
        // reaches `max_accum_cost`. dedup (landed) collapses identical → 1
        // compile, so the cost keys on the distinct `program_identity_hash` count
        // (the `distinct_programs` set), now plumbed through `on_arrival`.
        let mut p = RunAheadPolicy::new(512);
        p.accum_window = Duration::from_millis(50); // ample latency budget
        p.max_accum_cost = Duration::from_millis(40); // / ~5ms-per-distinct ⇒ 8
        p.inter_arrival_ewma = 0.0005;
        p.accumulating_since = Some(Instant::now());
        // 8 DISTINCT programs ⇒ 8×~5.06ms ≈ 40.5ms ≥ cap ⇒ compile wall ⇒ fire.
        p.on_arrival(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(matches!(p.decide(8), Decision::Fire));

        // Same request COUNT but all IDENTICAL (1 distinct) ⇒ dedup → 1 compile
        // ⇒ ~5ms ≪ cap ⇒ accumulate aggressively (the dedup win the policy turns).
        let mut q = RunAheadPolicy::new(512);
        q.accum_window = Duration::from_millis(50);
        q.max_accum_cost = Duration::from_millis(40);
        q.inter_arrival_ewma = 0.0005;
        q.accumulating_since = Some(Instant::now());
        q.on_arrival(&[42]);
        q.on_arrival(&[42]);
        q.on_arrival(&[42]);
        assert_eq!(q.distinct_programs.len(), 1);
        assert!(matches!(q.decide(8), Decision::Wait(_)));
    }

    #[test]
    fn accum_window_p99_cap_fires() {
        // Window (p99 latency) budget exhausted ⇒ fire even under dense arrivals.
        let mut p = RunAheadPolicy::new(512);
        p.accum_window = Duration::from_millis(5);
        p.inter_arrival_ewma = 0.0005;
        p.accumulating_since = Instant::now().checked_sub(Duration::from_millis(20));
        assert!(matches!(p.decide(2), Decision::Fire));
    }

    #[test]
    fn merged_cost_compile_dominated_and_dedup_collapses() {
        // delta #11 calibration: compile ≫ fire, and dedup (landed) collapses
        // identical programs to per-DISTINCT compile cost.
        let c = MergedCost::calibrated();
        // 100 requests, 1 distinct program (all identical) ⇒ ~1 compile (dedup),
        // far cheaper than 100 distinct programs ⇒ 100 compiles.
        let identical = c.estimate_s(100, 1);
        let all_distinct = c.estimate_s(100, 100);
        assert!(all_distinct > identical);
        // compile dominates: the per-distinct finalize residual dwarfs the
        // (M-batched, echo `5a7e6ec9`) per-distinct fire by >10×.
        assert!(c.per_distinct_finalize_s > 10.0 * c.per_mbatch_fire_s);
        // plain decode (no program) ⇒ zero sampling cost.
        assert_eq!(c.estimate_s(100, 0), 0.0);
    }

    #[test]
    fn distinct_programs_union_dedup_and_reset_on_fire() {
        // The distinct-count plumbing: `on_arrival` unions each request's
        // per-program `program_identity_hash`es into the window set (identical
        // self-dedup), and `on_fired` resets it for the next window. The public
        // `distinct_program_count()` (the #10 fire-trace witness) tracks it.
        let mut p = RunAheadPolicy::new(512);
        // Two requests, same grammar (one hash each) + one request with two
        // distinct programs ⇒ 3 distinct identities total.
        p.on_arrival(&[7]);
        p.on_arrival(&[7]); // identical ⇒ dedups
        p.on_arrival(&[8, 9]); // a 2-program pass
        assert_eq!(p.distinct_programs.len(), 3);
        assert_eq!(p.distinct_program_count(), 3); // the witness accessor
        // Plain-decode arrival (empty) adds nothing.
        p.on_arrival(&[]);
        assert_eq!(p.distinct_program_count(), 3);
        // Firing the batch drains the window.
        p.on_fired(4);
        assert!(p.distinct_programs.is_empty());
        assert_eq!(p.distinct_program_count(), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AdaptivePolicy -----------------------------------------------------

    #[test]
    fn adaptive_cold_start_fires() {
        // No fire has happened yet (fired_high_water == 0); fire to
        // make progress.
        let mut policy = AdaptivePolicy::new(512, 0);
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn adaptive_fires_at_structural_cap() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(64);
        // Structural cap fires unconditionally.
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    #[test]
    fn adaptive_fires_at_fired_high_water() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(64);
        policy.on_arrival(&[]);
        // Matched the historical cohort size; fire.
        assert!(matches!(policy.decide(64), Decision::Fire));
    }

    #[test]
    fn adaptive_waits_below_fired_high_water() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(64);
        policy.on_arrival(&[]);
        // Below the historical cohort — wait for stragglers.
        assert!(matches!(policy.decide(32), Decision::Wait(_)));
    }

    #[test]
    fn adaptive_watchdog_fires_after_last_latency() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_micros(1)); // tiny → watchdog elapses fast
        policy.on_fired(64);
        policy.on_arrival(&[]);
        std::thread::sleep(Duration::from_millis(2));
        // last_latency was 1 µs; 2 ms has elapsed → watchdog fires
        // even though batch is below fired_high_water.
        assert!(matches!(policy.decide(32), Decision::Fire));
    }

    #[test]
    fn adaptive_cold_start_fires_even_without_batch_start_time() {
        // No arrival recorded yet (batch_start_time is None). The
        // defensive branch fires immediately.
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(64);
        assert!(matches!(policy.decide(32), Decision::Fire));
    }

    #[test]
    fn adaptive_skips_first_batch_in_latency_update() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        assert_eq!(
            policy.last_latency, 0.0,
            "first batch is skipped (CUDA/warmup overhead inflates it)"
        );
    }

    #[test]
    fn adaptive_last_latency_tracks_most_recent_completed_batch() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(30));
        assert!((policy.last_latency - 0.030).abs() < 1e-6);
        policy.on_complete(Duration::from_millis(5));
        assert!((policy.last_latency - 0.005).abs() < 1e-6);
    }

    #[test]
    fn adaptive_fired_high_water_ratchets_only_upward() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_fired(40);
        policy.on_fired(80);
        policy.on_fired(60); // smaller — must not pull the watermark down
        assert_eq!(policy.fired_high_water, 80);
    }

    // ---- EagerPolicy --------------------------------------------------------

    #[test]
    fn eager_cold_start_fires() {
        let mut policy = EagerPolicy::new(512, 0);
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn eager_fires_at_max_forward_requests() {
        let mut policy = EagerPolicy::new(512, 0);
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    #[test]
    fn eager_skips_first_batch_in_latency_update() {
        let mut policy = EagerPolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        assert_eq!(policy.last_latency, 0.0);
    }

    // ---- GreedyPolicy -------------------------------------------------------

    #[test]
    fn greedy_always_fires() {
        let mut policy = GreedyPolicy::new();
        assert!(matches!(policy.decide(1), Decision::Fire));
        assert!(matches!(policy.decide(100), Decision::Fire));
        assert!(matches!(policy.decide(512), Decision::Fire));
    }
}
