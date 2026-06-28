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

use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

use super::scheduler::{Decision, SchedulingPolicy};

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

/// One-step run-ahead (R10): at most one batch computing + one prefetched.
const MAX_IN_FLIGHT: usize = 2;

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
        if self.in_flight.len() >= MAX_IN_FLIGHT {
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
        // `collection_complete == now` for a non-empty batch (MVP), so
        //   fire_at = max(earliest_in_flight_completion - lead_time, now).
        let Some(&completion) = self.in_flight.front() else {
            // Nothing in flight: the run-ahead overlap term is absent. Rather than
            // fire-on-arrival (which splits concurrent requests into R=1), apply the
            // #10 adaptive accumulation window (cold/idle path).
            return self.accumulation_decision(current_forward_requests);
        };
        let now = Instant::now();
        let lead = Duration::from_secs_f64(self.lead_time);
        // `lead >= remaining` (or underflow) -> fire now; else wait until one
        // lead-time before the earliest in-flight batch completes.
        let fire_at = completion.checked_sub(lead).unwrap_or(now);
        if now >= fire_at {
            Decision::Fire
        } else {
            Decision::Wait(fire_at - now)
        }
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
    fn run_ahead_waits_when_in_flight_far() {
        // Large forward latency + small lead -> fire_at is ~1s out -> Wait.
        let mut policy = RunAheadPolicy::new(512);
        policy.on_complete(Duration::from_millis(10)); // skipped
        policy.on_complete(Duration::from_secs(1)); // forward_latency = 1s
        policy.on_submitted(Duration::from_millis(5)); // lead_time = 5ms
        policy.on_fired(1); // in_flight_completion ~= now + 1s
        match policy.decide(1) {
            Decision::Wait(d) => assert!(d > Duration::from_millis(800)),
            Decision::Fire => panic!("expected Wait while the in-flight batch is ~1s out"),
        }
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
