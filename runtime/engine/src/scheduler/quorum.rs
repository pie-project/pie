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
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::stats::SchedulerStats;
use super::worker::FireClause;
use crate::scheduler::ProcessId;

/// Default run-ahead depth: one batch computing plus one prefetched.
/// Override via `PIE_SCHED_MAX_IN_FLIGHT`
/// (see [`configured_max_in_flight`]).
const DEFAULT_MAX_IN_FLIGHT: usize = 2;

const COLD_HOLD_US: u64 = 2_000;

fn cold_hold() -> Duration {
    static HOLD: OnceLock<Duration> = OnceLock::new();
    *HOLD.get_or_init(|| {
        Duration::from_micros(
            std::env::var("PIE_SCHED_COLD_HOLD_US")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(COLD_HOLD_US)
                .max(1),
        )
    })
}

/// Reads the requested run-ahead depth once. Dispatch-time preparation is the
/// allocation-credit gate: physical pool allocation is atomic, and an
/// exhausted request remains a retrying preparation rather than overcommitting.
fn parse_max_in_flight(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_IN_FLIGHT)
        .max(1)
}

pub(super) fn configured_max_in_flight() -> usize {
    static CONFIGURED: OnceLock<usize> = OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        parse_max_in_flight(std::env::var("PIE_SCHED_MAX_IN_FLIGHT").ok().as_deref())
    })
}

/// Bounded poll for the quorum-hold wait. The completion channel and new
/// arrivals both preempt the `Decision::Wait` select the instant they fire, so
/// this only bounds the worst-case re-evaluation cadence (a hang backstop).
pub(super) const QUORUM_POLL_US: u64 = 200;

const WAITALL_DEADLINE_US_DEFAULT: u64 = 500;

/// Consecutive deadline-misses before a pipeline is demoted from the wait-set
/// and queued for termination.
const WAITALL_MISS_LIMIT_DEFAULT: u32 = 10_000;

/// The per-wave straggler deadline (500 us by default).
fn waitall_deadline() -> Duration {
    static DEADLINE: OnceLock<Duration> = OnceLock::new();
    *DEADLINE.get_or_init(|| {
        Duration::from_micros(
            std::env::var("PIE_SCHED_WAITALL_DEADLINE_US")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(WAITALL_DEADLINE_US_DEFAULT)
                .max(1),
        )
    })
}

/// The consecutive-miss demotion limit. Ten thousand default-length misses
/// leave five seconds for a healthy pipeline's host round-trip.
fn waitall_miss_limit() -> u32 {
    static LIMIT: OnceLock<u32> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("PIE_SCHED_WAITALL_MISS_LIMIT")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(WAITALL_MISS_LIMIT_DEFAULT)
            .max(1)
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
    /// An unready dispatch preparation participated in this wave but has no
    /// launch row yet. It neither blocks the narrower wave nor accrues a miss.
    retry_participating: bool,
    in_flight: usize,
}

pub(super) struct WaitAllPolicy {
    greedy: bool,
    /// Structural cap — a full batch always fires immediately.
    max_forward_requests: usize,
    /// Batches enqueued but not yet retired; bounded by the configured depth.
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
        Self::with_greedy(max_forward_requests, stats, false)
    }

    pub fn with_greedy(
        max_forward_requests: usize,
        stats: Option<Arc<SchedulerStats>>,
        greedy: bool,
    ) -> Self {
        Self {
            greedy,
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
                let state = self.active.entry(pid).or_default();
                state.retry_participating = false;
                state.wave_ready += 1;
            }
            None => self.untracked_ready += 1,
        }
    }

    pub fn on_retry_participation(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid {
            let state = self.active.entry(pid).or_default();
            state.consecutive_misses = 0;
            state.retry_participating = true;
        }
    }

    /// A queued request that had already contributed a readiness credit will
    /// never dispatch (cancellation, stale instance, or synchronous rejection).
    pub fn on_request_dropped(&mut self, pid: Option<ProcessId>) {
        match pid {
            Some(pid) => {
                if let Some(state) = self.active.get_mut(&pid) {
                    debug_assert!(
                        state.wave_ready > 0,
                        "dropped pipeline request has no readiness credit"
                    );
                    state.wave_ready = state.wave_ready.saturating_sub(1);
                } else {
                    self.untracked_ready = self.untracked_ready.saturating_sub(1);
                }
            }
            None => {
                self.untracked_ready = self.untracked_ready.saturating_sub(1);
            }
        }
        if self.untracked_ready == 0 && self.active.values().all(|state| state.wave_ready == 0) {
            self.wave_started = None;
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
        if self.active.is_empty() {
            self.ever_fired = false;
            self.cold_hold_deadline = None;
            self.wave_started = None;
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
        if self.in_flight >= configured_max_in_flight() {
            return WaveDecision::Wait(Duration::from_micros(QUORUM_POLL_US));
        }
        // Structural capacity cap — a full batch fires immediately, no miss
        // penalty (the wave didn't run out of patience; it ran out of room).
        if current_batch_size >= self.max_forward_requests {
            self.record_clause(FireClause::Quorum);
            self.record_wave(0);
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
        if self.greedy {
            self.record_clause(FireClause::Quorum);
            self.record_wave(0);
            return WaveDecision::Fire {
                missing: Vec::new(),
            };
        }
        let missing: Vec<ProcessId> = self
            .active
            .iter()
            .filter(|(_, s)| {
                s.wave_ready == 0
                    && !s.retry_participating
                    && s.in_flight < configured_max_in_flight()
            })
            .map(|(&pid, _)| pid)
            .collect();

        if missing.is_empty() {
            // Every active pipeline is in (or the batch is untracked-only).
            if self.ever_fired {
                // Dense wave — the steady-state fire.
                self.record_clause(FireClause::Quorum);
                self.record_wave(0);
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
                    let window = cold_hold();
                    self.cold_hold_deadline = Some(now + window);
                    return WaveDecision::Wait(window);
                }
                Some(deadline) if now < deadline => {
                    return WaveDecision::Wait(deadline - now);
                }
                _ => {
                    self.record_clause(FireClause::ColdHold);
                    self.record_wave(0);
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
        self.record_wave(missing.len());
        WaveDecision::Fire { missing }
    }

    /// A wave was dispatched: consume exactly one readiness credit for each
    /// submitted row and retain credits for requests already queued behind
    /// this wave. Participation resets the straggler count for whoever rode
    /// this wave; absentees keep theirs.
    pub fn on_wave_dispatched(&mut self, participants: &[Option<ProcessId>], now: Instant) {
        self.in_flight += 1;
        self.ever_fired = true;
        self.cold_hold_deadline = None;
        for participant in participants {
            match participant {
                Some(pid) => {
                    if let Some(state) = self.active.get_mut(pid) {
                        debug_assert!(
                            state.wave_ready > 0,
                            "dispatched pipeline row has no readiness credit"
                        );
                        state.wave_ready = state.wave_ready.saturating_sub(1);
                        state.in_flight += 1;
                        state.consecutive_misses = 0;
                    } else {
                        self.untracked_ready = self.untracked_ready.saturating_sub(1);
                    }
                }
                None => {
                    self.untracked_ready = self.untracked_ready.saturating_sub(1);
                }
            }
        }
        for state in self.active.values_mut() {
            state.retry_participating = false;
        }
        if self.untracked_ready == 0 && self.active.values().all(|state| state.wave_ready == 0) {
            self.wave_started = None;
        } else {
            self.wave_started = Some(now);
        }
    }

    /// An in-flight batch retired: frees one run-ahead slot so
    /// `decide_wave_at`'s depth cap admits the next wave.
    pub fn on_wave_retired(&mut self, participants: &[Option<ProcessId>]) {
        self.in_flight = self.in_flight.saturating_sub(1);
        for pid in participants.iter().flatten() {
            if let Some(state) = self.active.get_mut(pid) {
                state.in_flight = state.in_flight.saturating_sub(1);
            }
        }
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

    fn record_wave(&self, missing: usize) {
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            stats
                .fire
                .quorum
                .wave_active_sum
                .fetch_add(self.active.len() as u64, Relaxed);
            stats
                .fire
                .quorum
                .wave_missing_sum
                .fetch_add(missing as u64, Relaxed);
            stats.fire.quorum.wave_fires.fetch_add(1, Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid() -> ProcessId {
        ProcessId::new_v4()
    }

    #[test]
    fn max_in_flight_configuration_is_truthful_and_safely_capped() {
        assert_eq!(parse_max_in_flight(None), DEFAULT_MAX_IN_FLIGHT);
        assert_eq!(parse_max_in_flight(Some("0")), 1);
        assert_eq!(parse_max_in_flight(Some("4")), 4);
        assert_eq!(parse_max_in_flight(Some("invalid")), DEFAULT_MAX_IN_FLIGHT);
        assert!(configured_max_in_flight() >= 1);
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
        let participants = policy
            .active
            .iter()
            .filter(|(_, state)| state.wave_ready > 0)
            .map(|(&pid, _)| Some(pid))
            .collect::<Vec<_>>();
        policy.on_wave_dispatched(&participants, past_cold_hold);
        policy.on_wave_retired(&participants);
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
        // Still inside the straggler deadline: holds for `b`.
        match policy.decide_wave_at(1, past_cold_hold + Duration::from_micros(250)) {
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
    fn one_in_flight_request_does_not_exempt_a_pipeline_from_runahead_wave() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let past_cold_hold = t0 + Duration::from_micros(COLD_HOLD_US + 100);
        let _ = policy.decide_wave_at(2, t0);
        assert_eq!(
            policy.decide_wave_at(2, past_cold_hold),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
        policy.on_wave_dispatched(&[Some(a), Some(b)], past_cold_hold);

        policy.on_pipeline_request(Some(a), past_cold_hold);
        assert!(
            matches!(
                policy.decide_wave_at(1, past_cold_hold + Duration::from_micros(250)),
                WaveDecision::Wait(_)
            ),
            "b has only one in-flight request and must still fill depth two"
        );
    }

    #[test]
    fn dispatch_consumes_only_participating_readiness_credits() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);

        let _ = policy.decide_wave_at(2, t0);
        let after = t0 + Duration::from_micros(COLD_HOLD_US + 100);
        assert_eq!(
            policy.decide_wave_at(2, after),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
        policy.on_wave_dispatched(&[Some(a), Some(b)], after);

        assert_eq!(policy.active[&a].wave_ready, 1);
        assert_eq!(policy.active[&b].wave_ready, 0);
        match policy.decide_wave_at(1, after + Duration::from_micros(250)) {
            WaveDecision::Wait(_) => {}
            other => panic!("queued a credit must leave only b missing, got {other:?}"),
        }
    }

    #[test]
    fn dropping_a_queued_request_removes_only_its_credit() {
        let mut policy = WaitAllPolicy::new(4, None);
        let a = pid();
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(a), t0);

        policy.on_request_dropped(Some(a));

        assert_eq!(policy.active[&a].wave_ready, 1);
        assert_eq!(policy.wave_started, Some(t0));
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
    fn empty_wait_set_rearms_bootstrap_gather_for_the_next_fleet() {
        let mut policy = WaitAllPolicy::new(4, None);
        let first = pid();
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(first), t0);
        let fired = bootstrap_fire(&mut policy, 1, t0);
        policy.on_pipeline_leave(first);

        let next = pid();
        policy.on_pipeline_request(Some(next), fired);
        assert_eq!(
            policy.decide_wave_at(1, fired),
            WaveDecision::Wait(Duration::from_micros(COLD_HOLD_US))
        );
    }

    #[test]
    fn retry_participation_neither_blocks_nor_demotes_the_pipeline() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let t = bootstrap_fire(&mut policy, 2, t0);

        policy.on_pipeline_request(Some(a), t);
        policy.on_retry_participation(Some(b));
        assert_eq!(
            policy.decide_wave_at(1, t + Duration::from_micros(1)),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
        assert_eq!(policy.misses(b), Some(0));
    }

    #[test]
    fn demotes_after_the_consecutive_miss_limit() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let mut t = bootstrap_fire(&mut policy, 2, t0);

        for _ in 0..waitall_miss_limit() {
            policy.on_pipeline_request(Some(a), t);
            t += Duration::from_millis(11);
            assert_eq!(
                policy.decide_wave_at(1, t),
                WaveDecision::Fire { missing: vec![b] }
            );
            policy.on_wave_dispatched(&[Some(a)], t);
            // Retire the wave before the next one gathers — otherwise the
            // run-ahead depth cap itself would hold (unrelated to this
            // test's miss-limit demotion).
            policy.on_wave_retired(&[Some(a)]);
        }
        assert!(
            !policy.is_active(b),
            "b should be demoted at the miss limit"
        );
    }
}
