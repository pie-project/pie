//! The batch scheduler's fire policy: the **wait-for-all-active-pipelines**
//! quorum rule ([`WaitAllPolicy`], overview §7.2 / thrust-2 F1–F6) — the single
//! scheduling algorithm. Wait until every active pipeline's next pass is ready,
//! then enqueue the dense wave behind the in-flight batch (depth-`max_in_flight`
//! multi-inflight run-ahead, zero bubble); stragglers fire on the wave deadline
//! and demote at the miss limit.
//!
//! ## Pie batching model
//!
//! Pie performs **iteration-level batching**: each in-flight context
//! re-submits a forward-pass request after every token. The scheduler
//! accumulates these into a wave and the rule decides when to fire.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::stats::SchedulerStats;
use super::worker::FireClause;
use crate::scheduler::ProcessId;

/// One-step run-ahead (R10): the DEFAULT run-ahead depth — at most one batch
/// computing + one prefetched. Override via `PIE_SCHED_MAX_IN_FLIGHT`
/// (see [`max_in_flight`]).
const DEFAULT_MAX_IN_FLIGHT: usize = 2;

const COLD_HOLD_US: u64 = 500;

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
    DEFAULT_MAX_IN_FLIGHT
}

/// Bounded poll for the quorum-hold wait. The completion channel and new
/// arrivals both preempt the `Decision::Wait` select the instant they fire, so
/// this only bounds the worst-case re-evaluation cadence (a hang backstop).
const QUORUM_POLL_US: u64 = 200;

const WAITALL_DEADLINE_US_DEFAULT: u64 = 10_000;

/// Consecutive deadline-misses before a pipeline is demoted from the wait-set
/// and queued for termination.
const WAITALL_MISS_LIMIT_DEFAULT: u32 = 5;

/// The per-wave straggler deadline (10ms).
fn waitall_deadline() -> Duration {
    Duration::from_micros(WAITALL_DEADLINE_US_DEFAULT)
}

/// The consecutive-miss demotion limit (5).
fn waitall_miss_limit() -> u32 {
    WAITALL_MISS_LIMIT_DEFAULT
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

    /// A wave was dispatched: bump the run-ahead depth `decide_wave_at`
    /// gates on, and reset this wave's bookkeeping so the next arrivals
    /// start a fresh gather. Participation resets the straggler count for
    /// whoever rode this wave; absentees keep theirs (already bumped on the
    /// deadline path that fired this wave, if it was a deadline fire) —
    /// this is the live dispatch-side half of the removed `on_fired`,
    /// without its EWMA/decay baggage.
    pub fn on_wave_dispatched(&mut self) {
        self.in_flight += 1;
        self.ever_fired = true;
        self.cold_hold_deadline = None;
        self.wave_started = None;
        self.untracked_ready = 0;
        for state in self.active.values_mut() {
            if state.wave_ready > 0 {
                state.consecutive_misses = 0;
                state.wave_ready = 0;
            }
        }
    }

    /// An in-flight batch retired: frees one run-ahead slot so
    /// `decide_wave_at`'s depth cap admits the next wave.
    pub fn on_wave_retired(&mut self) {
        self.in_flight = self.in_flight.saturating_sub(1);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pid() -> ProcessId {
        ProcessId::new_v4()
    }

    /// Drives a fresh `policy` through its bootstrap cold-hold (arm at `t0`,
    /// then fire past the window) so the tests below can start from an
    /// already-`ever_fired` policy without re-deriving the two-call
    /// cold-hold sequence every time.
    fn bootstrap_fire(
        policy: &mut WaitAllPolicy,
        current_batch_size: usize,
        t0: Instant,
    ) -> Instant {
        assert_eq!(
            policy.decide_wave_at(current_batch_size, t0),
            WaveDecision::Wait(Duration::from_micros(COLD_HOLD_US))
        );
        let past_cold_hold = t0 + Duration::from_micros(COLD_HOLD_US + 100);
        assert_eq!(
            policy.decide_wave_at(current_batch_size, past_cold_hold),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
        policy.on_wave_dispatched();
        past_cold_hold
    }

    #[test]
    fn structural_cap_fires_immediately_even_cold() {
        // max_forward_requests=1: a single request already saturates the
        // structural cap, so it fires with no cold-hold/quorum delay —
        // the reason every pre-existing worker.rs test (which all run at
        // this cap) is unaffected by wiring the quorum rule in.
        let mut policy = WaitAllPolicy::new(1, None);
        let now = Instant::now();
        policy.on_pipeline_request(None, now);
        assert_eq!(
            policy.decide_wave_at(1, now),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
    }

    #[test]
    fn empty_batch_waits_without_miss_accrual() {
        let mut policy = WaitAllPolicy::new(4, None);
        let now = Instant::now();
        assert_eq!(
            policy.decide_wave_at(0, now),
            WaveDecision::Wait(Duration::from_micros(QUORUM_POLL_US))
        );
    }

    #[test]
    fn cold_hold_gathers_two_pipelines_then_fires_dense() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        // Both pipelines' first requests are in; bootstrap membership is
        // still forming, so this holds the cold-hold gather window rather
        // than firing solo.
        assert_eq!(
            policy.decide_wave_at(2, t0),
            WaveDecision::Wait(Duration::from_micros(COLD_HOLD_US))
        );
        // Past the window: fires dense, both pipelines in, no stragglers.
        let after = t0 + Duration::from_micros(COLD_HOLD_US + 100);
        assert_eq!(
            policy.decide_wave_at(2, after),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
    }

    #[test]
    fn straggler_deadline_fires_missing_and_counts_a_miss() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let past_cold_hold = bootstrap_fire(&mut policy, 2, t0);

        // Wave 2: only `a` resubmits.
        policy.on_pipeline_request(Some(a), past_cold_hold);
        // Still inside the 10ms straggler deadline: holds for `b`.
        match policy.decide_wave_at(1, past_cold_hold + Duration::from_millis(1)) {
            WaveDecision::Wait(_) => {}
            other => panic!("expected a straggler wait, got {other:?}"),
        }
        // Deadline passed: fires with `b` missing and counts it a miss.
        let deadline_passed = past_cold_hold + Duration::from_millis(11);
        assert_eq!(
            policy.decide_wave_at(1, deadline_passed),
            WaveDecision::Fire { missing: vec![b] }
        );
        assert_eq!(policy.misses(b), Some(1));
        assert!(
            policy.is_active(b),
            "one miss is well under the demote limit"
        );
    }

    #[test]
    fn leave_drops_a_pipeline_from_the_wait_set_immediately() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let past_cold_hold = bootstrap_fire(&mut policy, 2, t0);

        policy.on_pipeline_request(Some(a), past_cold_hold);
        // `b` leaves (terminate/preempt) instead of resubmitting — dropped
        // from the wait-set immediately, well before its deadline.
        policy.on_pipeline_leave(b);
        assert!(!policy.is_active(b));
        assert_eq!(policy.active_pipelines(), 1, "only `a` remains awaited");
        assert_eq!(
            policy.decide_wave_at(1, past_cold_hold + Duration::from_micros(1)),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
    }

    #[test]
    fn demotes_after_the_consecutive_miss_limit() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let mut t = bootstrap_fire(&mut policy, 2, t0);

        for _ in 0..5 {
            policy.on_pipeline_request(Some(a), t);
            t += Duration::from_millis(11);
            assert_eq!(
                policy.decide_wave_at(1, t),
                WaveDecision::Fire { missing: vec![b] }
            );
            policy.on_wave_dispatched();
            // Retire the wave before the next one gathers — otherwise the
            // run-ahead depth cap itself would hold (unrelated to this
            // test's miss-limit demotion).
            policy.on_wave_retired();
        }
        assert!(
            !policy.is_active(b),
            "b should be demoted at the miss limit"
        );
    }
}
