//! The batch scheduler's fire policy: the **wait-for-all-active-pipelines**
//! quorum rule ([`WaitAllPolicy`], overview §7.2 / thrust-2 F1–F6) — the single
//! scheduling algorithm. Wait until every active pipeline's next pass is ready,
//! then enqueue the dense wave behind the in-flight batch (depth-`max_in_flight`
//! multi-inflight run-ahead, zero bubble). Steady-state waves never fire narrow:
//! membership changes only through explicit bind/close/leave events. A single
//! watchdog reports a non-responsive member but does not alter scheduling.
//!
//! ## Pie batching model
//!
//! Pie performs **iteration-level batching**: each in-flight context
//! re-submits a forward-pass request after every token. The scheduler
//! accumulates these into a wave and the rule decides when to fire.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::stats::SchedulerStats;
use super::worker::FireClause;
use crate::scheduler::ProcessId;

/// Default run-ahead depth: one batch computing plus one prefetched.
/// The N9 depth dose-response superseded the older depth-3 default:
/// depth 2 reduced missing/deferred enough to win steady throughput in
/// all three paired production-shape runs while retaining pre-enqueue.
/// `PIE_SCHED_MAX_IN_FLIGHT` may reduce this; depth above three is
/// intentionally capped because the CUDA driver sizes its pinned staging
/// pools from this value (`kSchedulerMaxInFlight` in
/// driver/cuda/src/runahead.hpp — staging depth must EXCEED run-ahead,
/// so raising this without raising that re-serializes every submit).
const DEFAULT_MAX_IN_FLIGHT: usize = 2;
const MAX_IN_FLIGHT: usize = 3;

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

const STRICT_WATCHDOG_US: u64 = 1_000_000;

/// Reads the requested run-ahead depth once. Dispatch-time preparation is the
/// allocation-credit gate: physical pool allocation is atomic, and an
/// exhausted request remains a retrying preparation rather than overcommitting.
fn parse_max_in_flight(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_IN_FLIGHT)
        .clamp(1, MAX_IN_FLIGHT)
}

pub(super) fn configured_max_in_flight() -> usize {
    static CONFIGURED: OnceLock<usize> = OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        parse_max_in_flight(std::env::var("PIE_SCHED_MAX_IN_FLIGHT").ok().as_deref())
    })
}

/// Defensive poll used outside a steady barrier (depth/admission backstops).
pub(super) const QUORUM_POLL_US: u64 = 200;

/// The wave-level fire decision. Strict quorum fires always carry an empty
/// `missing` set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum WaveDecision {
    Fire { missing: Vec<ProcessId> },
    Wait(Duration),
}

/// Per-pipeline wave participation.
#[derive(Debug, Default, Clone)]
struct PipelineWaveState {
    /// Process owning this pipeline scope. Process suspend/terminate removes
    /// every scope with the same owner; close removes only the keyed scope.
    owner: ProcessId,
    /// Temporary process-keyed membership created by instance binding before
    /// the guest submits on a concrete pipeline resource.
    bind_placeholder: bool,
    /// Requests this pipeline has contributed to the CURRENT wave.
    /// `> 0` ⇒ ready (its N+1 is in).
    wave_ready: usize,
    in_flight: usize,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PipelineEpoch {
    pid: ProcessId,
    generation: u64,
}

#[derive(Clone)]
pub(super) struct WaitAllPolicy {
    /// Structural cap — a full batch always fires immediately.
    max_forward_requests: usize,
    /// Batches enqueued but not yet retired; bounded by the configured depth.
    in_flight: usize,
    /// THE wait-set: every active pipeline and its wave participation.
    /// BTreeMap for deterministic `missing` ordering.
    active: BTreeMap<ProcessId, PipelineWaveState>,
    /// Pipelines already dispatched in a structurally partitioned logical
    /// round. They are not awaited again until every active member has
    /// dispatched once and the round closes.
    round_served: HashSet<ProcessId>,
    /// Bind controls accepted by the scheduler but not yet completed. These
    /// processes are active missing members even before their first fire, and
    /// a wave holds unconditionally while any of them is absent.
    pending_binds: BTreeMap<ProcessId, usize>,
    generations: BTreeMap<ProcessId, u64>,
    /// Untracked (pipeline-less) requests in the current wave: counted so an
    /// untracked-only batch still fires, never awaited.
    untracked_ready: usize,
    /// Liveness-only deadline for the current strict wait episode.
    strict_watchdog_deadline: Option<Instant>,
    /// Whether any wave has fired (bootstrap discriminator).
    ever_fired: bool,
    /// The bootstrap gather window deadline (armed on the first cold decide).
    cold_hold_deadline: Option<Instant>,
    /// Probe sink (`profile-fire`); `None` in unit tests.
    stats: Option<Arc<SchedulerStats>>,
}

impl WaitAllPolicy {
    pub fn new(max_forward_requests: usize, stats: Option<Arc<SchedulerStats>>) -> Self {
        Self {
            max_forward_requests,
            in_flight: 0,
            active: BTreeMap::new(),
            round_served: HashSet::new(),
            pending_binds: BTreeMap::new(),
            generations: BTreeMap::new(),
            untracked_ready: 0,
            strict_watchdog_deadline: None,
            ever_fired: false,
            cold_hold_deadline: None,
            stats,
        }
    }

    /// One-line-per-pipeline snapshot of the barrier state, for the
    /// scheduler's debug dump (a held wave must be inspectable).
    pub fn debug_summary(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = write!(
            out,
            "in_flight={} pending_binds={} round_served={} untracked_ready={} ever_fired={} watchdog={:?} cold_hold={:?}",
            self.in_flight,
            self.pending_binds.values().sum::<usize>(),
            self.round_served.len(),
            self.untracked_ready,
            self.ever_fired,
            self.strict_watchdog_deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now())),
            self.cold_hold_deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now())),
        );
        for (pid, state) in &self.active {
            let _ = write!(
                out,
                "\n  pipeline {pid}: wave_ready={} in_flight={} generation={}",
                state.wave_ready, state.in_flight, state.generation,
            );
        }
        out
    }

    /// A request entered the current wave. `Some(pid)` joins the wait-set
    /// implicitly on first sight and marks the pipeline ready; `None` is
    /// untracked (prebuilt/beam/replay — rides the wave, never awaited).
    #[cfg(test)]
    pub fn on_pipeline_request(&mut self, pid: Option<ProcessId>, now: Instant) {
        self.on_pipeline_request_owned(pid, pid, now);
    }

    pub fn on_pipeline_request_owned(
        &mut self,
        pid: Option<ProcessId>,
        owner: Option<ProcessId>,
        _now: Instant,
    ) {
        match (pid, owner) {
            (Some(pid), Some(owner)) => {
                self.retire_bind_placeholder(owner, pid);
                let state = self.active_state(pid, owner, false);
                state.wave_ready += 1;
            }
            _ => self.untracked_ready += 1,
        }
    }

    /// A tracked pipeline has submitted work that is not launch-ready yet.
    /// It joins the barrier immediately and remains missing until preparation
    /// publishes a readiness credit.
    #[cfg(test)]
    pub fn on_pipeline_join(&mut self, pid: Option<ProcessId>) {
        self.on_pipeline_join_owned(pid, pid);
    }

    pub fn on_pipeline_join_owned(&mut self, pid: Option<ProcessId>, owner: Option<ProcessId>) {
        if let (Some(pid), Some(owner)) = (pid, owner) {
            self.retire_bind_placeholder(owner, pid);
            self.active_state(pid, owner, false);
        }
    }

    /// A bind control entered the scheduler. Assembly membership starts here,
    /// before native bind completion or the process's first fire.
    pub fn on_bind_enqueued(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid {
            self.active_state(pid, pid, true);
            *self.pending_binds.entry(pid).or_default() += 1;
        }
    }

    /// A bind control completed, whether successfully or with an error. Bind
    /// membership has no concrete pipeline scope yet, so its process-keyed
    /// placeholder ends with the last bind; the first fire joins under the
    /// actual pipeline resource identity.
    pub fn on_bind_completed(&mut self, pid: Option<ProcessId>, _now: Instant) {
        let Some(pid) = pid else {
            return;
        };
        let Some(count) = self.pending_binds.get_mut(&pid) else {
            return;
        };
        *count = count.saturating_sub(1);
        let completed = *count == 0;
        if completed {
            self.pending_binds.remove(&pid);
            if self
                .active
                .get(&pid)
                .is_some_and(|state| state.bind_placeholder)
            {
                self.on_pipeline_leave(pid);
            }
        }
    }

    fn active_state(
        &mut self,
        pid: ProcessId,
        owner: ProcessId,
        bind_placeholder: bool,
    ) -> &mut PipelineWaveState {
        let generation = *self.generations.entry(pid).or_default();
        self.active
            .entry(pid)
            .and_modify(|state| {
                debug_assert_eq!(state.owner, owner);
                state.bind_placeholder &= bind_placeholder;
            })
            .or_insert(PipelineWaveState {
                owner,
                bind_placeholder,
                generation,
                ..PipelineWaveState::default()
            })
    }

    fn retire_bind_placeholder(&mut self, owner: ProcessId, pipeline_id: ProcessId) {
        if owner == pipeline_id || self.pending_binds.contains_key(&owner) {
            return;
        }
        if self
            .active
            .get(&owner)
            .is_some_and(|state| state.bind_placeholder)
        {
            let placeholder = self.active.remove(&owner).unwrap();
            self.untracked_ready += placeholder.wave_ready;
        }
    }

    /// A queued request that had already contributed a readiness credit will
    /// never dispatch (cancellation, stale instance, or synchronous
    /// rejection). CONTRACT: callers invoke this only for a request that
    /// holds an unconsumed credit (`PendingRequest::credit_published` in the
    /// worker) — a fire cancelled before its pre-launch copy retires never
    /// published one, and giving back a credit it never held would eat a
    /// sibling's.
    pub fn on_request_dropped(&mut self, pid: Option<ProcessId>) {
        match pid {
            Some(pid) => {
                if let Some(state) = self.active.get_mut(&pid) {
                    debug_assert!(
                        state.wave_ready > 0,
                        "dropped pipeline request has no readiness credit \
                         (caller must gate on credit_published)"
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
    }

    /// Release a pipeline from the wait-set. Readiness credits already
    /// published by its queued fires transfer to the untracked bucket; fires
    /// still preparing publish there after observing the bumped generation.
    /// This is what lets graceful close stop being awaited immediately while
    /// every accepted run-ahead fire continues to settlement.
    pub fn on_pipeline_leave(&mut self, pid: ProcessId) {
        if let Some(state) = self.active.remove(&pid) {
            self.untracked_ready += state.wave_ready;
        }
        self.round_served.remove(&pid);
        self.close_round_if_complete();
        self.pending_binds.remove(&pid);
        *self.generations.entry(pid).or_default() += 1;
        if self.active.is_empty() {
            self.ever_fired = false;
            self.cold_hold_deadline = None;
            self.strict_watchdog_deadline = None;
            self.round_served.clear();
        }
    }

    /// Release every quorum scope owned by one process. Suspend/terminate are
    /// process lifecycle events; unlike pipeline close/allocation-wait they
    /// must remove sibling pipeline scopes together.
    pub fn on_process_leave(&mut self, owner: ProcessId) {
        let pipelines: Vec<_> = self
            .active
            .iter()
            .filter_map(|(pid, state)| (state.owner == owner).then_some(*pid))
            .collect();
        for pipeline_id in pipelines {
            self.on_pipeline_leave(pipeline_id);
        }
        self.pending_binds.remove(&owner);
    }

    /// The wait-set size (probe/test accessor).
    pub fn active_pipelines(&self) -> usize {
        self.active.len()
    }

    pub fn depth_capped_pipelines(&self) -> usize {
        self.active
            .values()
            .filter(|state| state.in_flight >= configured_max_in_flight())
            .count()
    }

    /// The pipeline's current identity generation. Bumped by every
    /// [`Self::on_pipeline_leave`]; a request stamped with an older
    /// generation belongs to a pipeline that has since left, and its
    /// quorum accounting must route untracked (`quorum_pid` in the
    /// worker) — the pid alone cannot distinguish a closed pipeline's
    /// straggler from a successor pipeline of the same process.
    pub fn generation_of(&self, pid: ProcessId) -> u64 {
        self.generations.get(&pid).copied().unwrap_or(0)
    }

    /// Untracked readiness credits currently in the gather (probe/test
    /// accessor; the W1 leak gate — must drain to 0 with the fleet).
    pub fn untracked_ready_count(&self) -> usize {
        self.untracked_ready
    }

    pub fn candidate_state_counts(
        &self,
        candidate_pipelines: &HashSet<ProcessId>,
    ) -> (usize, usize, usize) {
        let mut present = 0;
        let mut at_depth = 0;
        let mut missing = 0;
        for (pid, state) in &self.active {
            // `at_depth` counts everyone the wave is not waiting on because
            // it already participated in this round or exhausted run-ahead.
            if self.pending_binds.contains_key(pid) {
                missing += 1;
            } else if self.round_served.contains(pid)
                || state.in_flight >= configured_max_in_flight()
            {
                at_depth += 1;
            } else if candidate_pipelines.contains(pid) {
                present += 1;
            } else {
                missing += 1;
            }
        }
        (present, at_depth, missing)
    }

    /// Whether `pid` is currently awaited (test accessor).
    #[cfg(test)]
    pub fn is_active(&self, pid: ProcessId) -> bool {
        self.active.contains_key(&pid)
    }

    /// The pure wave decision, driven with an explicit `now` (test-stable).
    ///
    /// Order: depth cap → capacity fire → empty guard → the wave barrier
    /// (all-ready dense fire / bootstrap gather).
    #[cfg(test)]
    pub fn decide_wave_at(&mut self, current_batch_size: usize, now: Instant) -> WaveDecision {
        static EMPTY: OnceLock<HashSet<ProcessId>> = OnceLock::new();
        let empty = EMPTY.get_or_init(HashSet::new);
        self.decide_wave_inner(current_batch_size, None, empty, false, now)
    }

    /// Production decision using the pipelines present in the batch that the
    /// dispatcher can actually build now. Readiness credits behind a control or
    /// preparation barrier cannot make a narrower candidate satisfy quorum.
    pub fn decide_candidate_wave_at(
        &mut self,
        current_batch_size: usize,
        candidate_pipelines: &HashSet<ProcessId>,
        deferred_pipelines: &HashSet<ProcessId>,
        structurally_full: bool,
        now: Instant,
    ) -> WaveDecision {
        self.decide_wave_inner(
            current_batch_size,
            Some(candidate_pipelines),
            deferred_pipelines,
            structurally_full,
            now,
        )
    }

    /// Side-effect-free candidate decision used before driver admission.
    ///
    /// A denied physical preparation must not advance barrier state or wave
    /// counters, so the scheduler evaluates against a private copy and
    /// invokes the mutating decision only after preparation succeeds.
    pub fn preview_candidate_wave_at(
        &self,
        current_batch_size: usize,
        candidate_pipelines: &HashSet<ProcessId>,
        deferred_pipelines: &HashSet<ProcessId>,
        structurally_full: bool,
        now: Instant,
    ) -> WaveDecision {
        let mut preview = self.clone();
        preview.stats = None;
        preview.decide_wave_inner(
            current_batch_size,
            Some(candidate_pipelines),
            deferred_pipelines,
            structurally_full,
            now,
        )
    }

    fn decide_wave_inner(
        &mut self,
        current_batch_size: usize,
        candidate_pipelines: Option<&HashSet<ProcessId>>,
        deferred_pipelines: &HashSet<ProcessId>,
        structurally_full: bool,
        now: Instant,
    ) -> WaveDecision {
        // Run-ahead depth cap: the completion channel preempts this wait the
        // instant a batch retires.
        if self.in_flight >= configured_max_in_flight() {
            return WaveDecision::Wait(Duration::from_micros(QUORUM_POLL_US));
        }
        // Structural capacity cap — a full batch fires immediately, no miss
        // penalty (the wave didn't run out of patience; it ran out of room).
        if self.pending_binds.is_empty()
            && (structurally_full || current_batch_size >= self.max_forward_requests)
        {
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
        let missing: Vec<ProcessId> = self
            .active
            .iter()
            .filter(|(pid, state)| {
                self.pending_binds.contains_key(*pid)
                    || (
                        // Deferred BY THE COMPOSER (capacity / wave token
                        // budget / same-instance dedup): scheduled for the next
                        // wave, not late — never awaited, never a straggler.
                        !deferred_pipelines.contains(*pid)
                            && !self.round_served.contains(*pid)
                            && state.in_flight < configured_max_in_flight()
                            && candidate_pipelines
                                .map(|pipelines| !pipelines.contains(pid))
                                .unwrap_or(state.wave_ready == 0)
                    )
            })
            .map(|(&pid, _)| pid)
            .collect();

        if missing.is_empty() {
            self.strict_watchdog_deadline = None;
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

        let deadline = self
            .strict_watchdog_deadline
            .get_or_insert(now + Duration::from_micros(STRICT_WATCHDOG_US));
        if now >= *deadline {
            *deadline = now + Duration::from_micros(STRICT_WATCHDOG_US);
            crate::scheduler::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "scheduler",
                "event": "strict_wait_watchdog",
                "at_us": crate::scheduler::fire_timing_now_us(),
                "batch_size": current_batch_size,
                "missing_count": missing.len(),
                "active_pipelines": self.active.len(),
            }));
        }
        WaveDecision::Wait(deadline.saturating_duration_since(now))
    }

    /// A wave was dispatched: consume exactly one readiness credit for each
    /// submitted row and retain credits for requests already queued behind
    /// this wave.
    pub fn on_wave_dispatched(
        &mut self,
        participants: &[Option<ProcessId>],
        _now: Instant,
    ) -> Vec<Option<PipelineEpoch>> {
        self.in_flight += 1;
        self.ever_fired = true;
        self.cold_hold_deadline = None;
        let mut epochs = Vec::with_capacity(participants.len());
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
                        self.round_served.insert(*pid);
                        epochs.push(Some(PipelineEpoch {
                            pid: *pid,
                            generation: state.generation,
                        }));
                    } else {
                        self.untracked_ready = self.untracked_ready.saturating_sub(1);
                        epochs.push(None);
                    }
                }
                None => {
                    self.untracked_ready = self.untracked_ready.saturating_sub(1);
                    epochs.push(None);
                }
            }
        }
        self.strict_watchdog_deadline = None;
        self.close_round_if_complete();
        epochs
    }

    /// An in-flight batch retired: frees one run-ahead slot so
    /// `decide_wave_at`'s depth cap admits the next wave.
    pub fn on_wave_retired(&mut self, participants: &[Option<PipelineEpoch>]) {
        self.in_flight = self.in_flight.saturating_sub(1);
        for epoch in participants.iter().flatten() {
            if let Some(state) = self.active.get_mut(&epoch.pid)
                && state.generation == epoch.generation
            {
                state.in_flight = state.in_flight.saturating_sub(1);
            }
        }
    }

    fn record_clause(&self, clause: FireClause) {
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            match clause {
                // Bootstrap gathers are recorded separately; dense/capacity
                // waves are the ordinary quorum path.
                FireClause::ColdHold => {
                    stats.fire.quorum.cold_hold_fires.fetch_add(1, Relaxed);
                }
                _ => {}
            }
        }
    }

    fn close_round_if_complete(&mut self) {
        if !self.active.is_empty()
            && self
                .active
                .keys()
                .all(|pid| self.round_served.contains(pid))
        {
            self.round_served.clear();
        }
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
        assert_eq!(parse_max_in_flight(Some("4")), MAX_IN_FLIGHT);
        assert_eq!(parse_max_in_flight(Some("invalid")), DEFAULT_MAX_IN_FLIGHT);
        assert!(configured_max_in_flight() >= 1);
    }

    /// Graceful close releases the wait-set immediately without dropping the
    /// run-ahead tail: an already-credited queued fire transfers to untracked,
    /// and a still-preparing fire publishes untracked after the generation
    /// bump. Both credits drain with their fires.
    #[test]
    fn close_releases_straddling_fires_to_untracked_drain() {
        let mut policy = WaitAllPolicy::new(64, None);
        let process = pid();
        let now = Instant::now();
        let admitted_generation = policy.generation_of(process);

        // F_0 is credited+queued; F_1 is accepted but still preparing.
        policy.on_pipeline_request(Some(process), now);
        policy.on_pipeline_leave(process);
        assert_eq!(
            policy.untracked_ready_count(),
            1,
            "the queued fire's published credit must transfer"
        );
        assert_eq!(policy.active_pipelines(), 0, "close releases the wait-set");
        assert_ne!(
            policy.generation_of(process),
            admitted_generation,
            "a leave must bump the identity generation"
        );

        // F_1 finishes preparing after close and observes the stale stamp.
        policy.on_pipeline_request(None, now);
        assert_eq!(policy.untracked_ready_count(), 2);
        let epochs = policy.on_wave_dispatched(&[None, None], now);
        assert_eq!(policy.untracked_ready_count(), 0);
        assert_eq!(policy.active_pipelines(), 0);
        policy.on_wave_retired(&epochs);
    }

    /// A fire that RETRYs after close remains untracked. Retrying must not
    /// resurrect the released wait-set row.
    #[test]
    fn retry_after_close_remains_untracked() {
        let mut policy = WaitAllPolicy::new(64, None);
        let process = pid();
        let now = Instant::now();
        policy.on_pipeline_request(Some(process), now);
        policy.on_pipeline_leave(process);
        let epochs = policy.on_wave_dispatched(&[None], now);
        assert_eq!(policy.active_pipelines(), 0);
        assert_eq!(policy.untracked_ready_count(), 0);

        policy.on_pipeline_request(None, now);
        assert_eq!(policy.active_pipelines(), 0);
        assert_eq!(policy.untracked_ready_count(), 1);
        let epochs_retry = policy.on_wave_dispatched(&[None], now);
        assert_eq!(policy.untracked_ready_count(), 0);
        policy.on_wave_retired(&epochs);
        policy.on_wave_retired(&epochs_retry);
    }

    /// W1 consequence regression: once the books drain, a fresh fleet must
    /// enter a fresh cold assembly episode rather than inherit old readiness.
    #[test]
    fn drained_books_rearm_the_gather_clock() {
        let mut policy = WaitAllPolicy::new(64, None);
        let closer = pid();
        let start = Instant::now();
        // A graceful close transfers the queued fire's credit; its drain
        // consumes that untracked credit before the next fleet arrives.
        policy.on_pipeline_request(Some(closer), start);
        policy.on_pipeline_leave(closer);
        let closed_epochs = policy.on_wave_dispatched(&[None], start);
        policy.on_wave_retired(&closed_epochs);
        assert_eq!(policy.untracked_ready_count(), 0);
        // Long after the prior drain, a fresh fleet gathers: B is ready,
        // C has joined but is not launch-ready yet.
        let later = start + Duration::from_millis(50);
        let ready = pid();
        let joining = pid();
        policy.on_pipeline_request(Some(ready), later);
        policy.on_pipeline_join(Some(joining));
        let candidates: HashSet<ProcessId> = [ready].into_iter().collect();
        let empty: HashSet<ProcessId> = HashSet::new();
        match policy.decide_candidate_wave_at(
            1,
            &candidates,
            &empty,
            false,
            later + Duration::from_micros(200),
        ) {
            WaveDecision::Wait(_) => {}
            WaveDecision::Fire { missing } => panic!(
                "the gather clock must start at the fresh fleet's arrival, \
                 not the drained wave: fired narrow with missing={missing:?}"
            ),
        }
    }

    #[test]
    fn depth_cap_and_strict_barrier_both_wait() {
        let mut policy = WaitAllPolicy::new(64, None);
        let ready = pid();
        let absent = pid();
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(ready), t0);
        policy.on_pipeline_request(Some(absent), t0);
        let now = bootstrap_fire(&mut policy, 2, t0);

        // Fill the run-ahead depth with dense waves from both pipelines.
        let mut epochs = Vec::new();
        for _ in 0..configured_max_in_flight() {
            policy.on_pipeline_request(Some(ready), now);
            policy.on_pipeline_request(Some(absent), now);
            assert_eq!(
                policy.decide_wave_at(2, now),
                WaveDecision::Fire {
                    missing: Vec::new()
                }
            );
            epochs.push(policy.on_wave_dispatched(&[Some(ready), Some(absent)], now));
        }

        // A new wave gathers with only `ready` while the pipe is full.
        policy.on_pipeline_request(Some(ready), now);
        let capped_until = now + Duration::from_secs(10);
        assert_eq!(
            policy.decide_wave_at(1, capped_until),
            WaveDecision::Wait(Duration::from_micros(QUORUM_POLL_US))
        );

        // Releasing depth does not weaken strict membership.
        policy.on_wave_retired(&epochs[0]);
        let shortly_after = capped_until + Duration::from_micros(QUORUM_POLL_US);
        assert!(matches!(
            policy.decide_wave_at(1, shortly_after),
            WaveDecision::Wait(_)
        ));
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
        let epochs = policy.on_wave_dispatched(&participants, past_cold_hold);
        policy.on_wave_retired(&epochs);
        past_cold_hold
    }

    #[test]
    fn denied_admission_preview_changes_no_quorum_state_or_stats() {
        let stats = Arc::new(SchedulerStats::default());
        let mut policy = WaitAllPolicy::new(64, Some(Arc::clone(&stats)));
        let (ready, absent) = (pid(), pid());
        let start = Instant::now();
        policy.on_pipeline_request(Some(ready), start);
        policy.on_pipeline_request(Some(absent), start);
        let fired = bootstrap_fire(&mut policy, 2, start);
        policy.on_pipeline_request(Some(ready), fired);
        let candidates = HashSet::from([ready]);
        let before = policy.debug_summary();
        let waves_before = stats
            .fire
            .quorum
            .wave_fires
            .load(std::sync::atomic::Ordering::Relaxed);
        let decision = policy.preview_candidate_wave_at(
            1,
            &candidates,
            &HashSet::new(),
            false,
            fired + Duration::from_secs(60),
        );
        assert!(matches!(decision, WaveDecision::Wait(_)));
        assert_eq!(policy.debug_summary(), before);
        assert_eq!(
            stats
                .fire
                .quorum
                .wave_fires
                .load(std::sync::atomic::Ordering::Relaxed,),
            waves_before,
        );
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
    fn missing_pipeline_waits_until_it_submits_or_leaves() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let past_cold_hold = bootstrap_fire(&mut policy, 2, t0);

        policy.on_pipeline_request(Some(a), past_cold_hold);
        assert!(matches!(
            policy.decide_wave_at(1, past_cold_hold + Duration::from_secs(60)),
            WaveDecision::Wait(_)
        ));

        policy.on_pipeline_request(Some(b), past_cold_hold);
        assert_eq!(
            policy.decide_wave_at(2, past_cold_hold),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
    }

    #[test]
    fn credits_behind_queue_barriers_do_not_satisfy_the_candidate_wave() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let t = bootstrap_fire(&mut policy, 2, t0);

        policy.on_pipeline_request(Some(a), t);
        policy.on_pipeline_request(Some(b), t);
        assert!(matches!(
            policy.decide_candidate_wave_at(1, &HashSet::from([a]), &HashSet::new(), false, t),
            WaveDecision::Wait(_)
        ));
        assert_eq!(
            policy.decide_candidate_wave_at(2, &HashSet::from([a, b]), &HashSet::new(), false, t),
            WaveDecision::Fire {
                missing: Vec::new()
            }
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
        let _ = policy.on_wave_dispatched(&[Some(a), Some(b)], past_cold_hold);

        policy.on_pipeline_request(Some(a), past_cold_hold);
        assert!(
            matches!(
                policy.decide_wave_at(1, past_cold_hold + Duration::from_micros(250)),
                WaveDecision::Wait(_)
            ),
            "b has spare run-ahead depth and must still be awaited"
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
        let _ = policy.on_wave_dispatched(&[Some(a), Some(b)], after);

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
        // `b` leaves instead of resubmitting and is dropped from the wait-set
        // immediately.
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
    fn closing_one_pipeline_scope_keeps_its_process_sibling_active() {
        let mut policy = WaitAllPolicy::new(4, None);
        let owner = pid();
        let (closed, sibling) = (pid(), pid());
        let now = Instant::now();
        policy.on_pipeline_request_owned(Some(closed), Some(owner), now);
        policy.on_pipeline_request_owned(Some(sibling), Some(owner), now);

        policy.on_pipeline_leave(closed);

        assert!(!policy.is_active(closed));
        assert!(policy.is_active(sibling));
        assert_eq!(policy.active_pipelines(), 1);
    }

    #[test]
    fn allocation_wait_leave_keeps_sibling_pipeline_in_the_quorum() {
        let mut policy = WaitAllPolicy::new(4, None);
        let owner = pid();
        let (waiting, sibling) = (pid(), pid());
        let now = Instant::now();
        policy.on_pipeline_request_owned(Some(waiting), Some(owner), now);
        policy.on_pipeline_request_owned(Some(sibling), Some(owner), now);
        let fired = bootstrap_fire(&mut policy, 2, now);
        policy.on_pipeline_request_owned(Some(sibling), Some(owner), fired);

        // Allocation-wait uses the same scope-leave operation as close, but
        // the process and its independently runnable sibling remain live.
        policy.on_pipeline_leave(waiting);

        assert_eq!(
            policy.decide_wave_at(1, fired + Duration::from_micros(1)),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
        assert!(policy.is_active(sibling));
    }

    #[test]
    fn process_suspend_removes_all_owned_pipeline_scopes() {
        let mut policy = WaitAllPolicy::new(4, None);
        let owner = pid();
        let other_owner = pid();
        let (first, second, unrelated) = (pid(), pid(), pid());
        let now = Instant::now();
        policy.on_pipeline_request_owned(Some(first), Some(owner), now);
        policy.on_pipeline_request_owned(Some(second), Some(owner), now);
        policy.on_pipeline_request_owned(Some(unrelated), Some(other_owner), now);

        policy.on_process_leave(owner);

        assert!(!policy.is_active(first));
        assert!(!policy.is_active(second));
        assert!(policy.is_active(unrelated));
    }

    #[test]
    fn completed_bind_placeholder_does_not_outlive_assembly() {
        let mut policy = WaitAllPolicy::new(4, None);
        let owner = pid();
        let pipeline = pid();
        let now = Instant::now();
        policy.on_bind_enqueued(Some(owner));
        policy.on_bind_completed(Some(owner), now);
        assert!(!policy.is_active(owner));

        policy.on_pipeline_request_owned(Some(pipeline), Some(owner), now);

        assert!(!policy.is_active(owner));
        assert!(policy.is_active(pipeline));
        assert_eq!(policy.active_pipelines(), 1);
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
    fn preparation_blocks_until_its_pipeline_is_ready() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let t = bootstrap_fire(&mut policy, 2, t0);

        policy.on_pipeline_request(Some(a), t);
        policy.on_pipeline_join(Some(b));
        assert!(matches!(
            policy.decide_wave_at(1, t + Duration::from_micros(1)),
            WaveDecision::Wait(_)
        ));
        policy.on_pipeline_request(Some(b), t);
        assert_eq!(
            policy.decide_wave_at(2, t + Duration::from_micros(1)),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
    }

    #[test]
    fn pending_binds_hold_the_wave_until_assembly_completes() {
        let mut policy = WaitAllPolicy::new(8, None);
        let lanes = [pid(), pid(), pid(), pid()];
        let t0 = Instant::now();
        for lane in lanes {
            policy.on_pipeline_request(Some(lane), t0);
        }
        let t = bootstrap_fire(&mut policy, lanes.len(), t0);

        policy.on_pipeline_request(Some(lanes[0]), t);
        policy.on_pipeline_request(Some(lanes[1]), t);
        policy.on_bind_enqueued(Some(lanes[2]));
        policy.on_bind_enqueued(Some(lanes[3]));
        let candidates = HashSet::from([lanes[0], lanes[1]]);
        let empty = HashSet::new();

        assert_eq!(policy.candidate_state_counts(&candidates), (2, 0, 2));
        assert!(matches!(
            policy.decide_candidate_wave_at(
                2,
                &candidates,
                &empty,
                true,
                t + Duration::from_secs(60),
            ),
            WaveDecision::Wait(_)
        ));

        for lane in &lanes[2..] {
            policy.on_bind_completed(Some(*lane), t);
            policy.on_pipeline_request(Some(*lane), t);
        }
        let all_candidates = lanes.into_iter().collect();
        assert_eq!(policy.candidate_state_counts(&all_candidates), (4, 0, 0));
        assert_eq!(
            policy.decide_candidate_wave_at(4, &all_candidates, &empty, true, t,),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
    }

    #[test]
    fn missing_pipeline_is_never_removed_by_time() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let t = bootstrap_fire(&mut policy, 2, t0);
        policy.on_pipeline_request(Some(a), t);
        assert!(matches!(
            policy.decide_wave_at(1, t + Duration::from_secs(60)),
            WaveDecision::Wait(_)
        ));
        assert!(policy.is_active(b));
    }

    #[test]
    fn retirement_from_an_old_generation_does_not_free_the_new_one() {
        let mut policy = WaitAllPolicy::new(4, None);
        let pipeline = pid();
        let now = Instant::now();
        policy.on_pipeline_request(Some(pipeline), now);
        policy.ever_fired = true;
        let old = policy.on_wave_dispatched(&[Some(pipeline)], now);

        policy.on_pipeline_leave(pipeline);
        policy.on_pipeline_request(Some(pipeline), now);
        let new = policy.on_wave_dispatched(&[Some(pipeline)], now);
        assert_eq!(policy.active[&pipeline].in_flight, 1);

        policy.on_wave_retired(&old);
        assert_eq!(policy.active[&pipeline].in_flight, 1);
        policy.on_wave_retired(&new);
        assert_eq!(policy.active[&pipeline].in_flight, 0);
    }
}
