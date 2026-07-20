//! The batch scheduler's fire policy: the **wait-for-all-active-pipelines**
//! quorum rule ([`WaitAllPolicy`], overview §7.2 / thrust-2 F1–F6) — the single
//! scheduling algorithm. Wait until every active pipeline's next pass is ready,
//! then enqueue the dense wave behind the in-flight batch (depth-`max_in_flight`
//! multi-inflight run-ahead, zero bubble). A gathering wave holds the barrier
//! for at most the wave window: a pipeline that misses a full window demotes —
//! the wave fires narrower without it and the barrier stops awaiting it until
//! it next participates (straggler demotion, runahead plan rev 2). Pipelines
//! still leave the wait-set only explicitly; demotion never removes membership,
//! it only stops a straggler from holding everyone else's barrier.
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

/// Steady-state straggler window: how long a gathering wave may hold the
/// barrier for awaited absentees before demoting them and firing narrow.
///
/// The window must exceed one dense-wave cadence (dispatch-to-dispatch,
/// ~3ms on the reference RTX 4090 decode workload), or the barrier can
/// never re-merge phase-shifted lane groups: each group's window expires
/// just before the other group's fires arrive, demotion locks the split
/// in, and the fleet decoheres into parallel narrow-wave trains
/// (measured: 915 waves instead of ~520 and −35% throughput at 2ms;
/// dense-throughout at 6ms; over-holding regresses again by 12ms).
const WAVE_WINDOW_US: u64 = 6_000;

/// Resolves the wave window from the raw env value, flooring at the
/// cold-hold gather window. A window below the floor would make the
/// steady-state barrier less patient than bootstrap — and at ≈0 every
/// wave fires with whatever happened to arrive, which is arrival-order
/// scheduling reintroduced through a knob. Wait-all is the design, not
/// a config option, so the knob tunes patience above the floor and
/// cannot disable the barrier. Returns the window and whether the
/// requested value was clamped (the caller warns loudly: a clamped
/// config is a misconfiguration, not a tuning).
fn resolve_wave_window(raw: Option<&str>, floor: Duration) -> (Duration, bool) {
    let requested = Duration::from_micros(
        raw.and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(WAVE_WINDOW_US),
    );
    if requested < floor {
        (floor, true)
    } else {
        (requested, false)
    }
}

pub(super) fn wave_window() -> Duration {
    static WINDOW: OnceLock<Duration> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        let raw = std::env::var("PIE_SCHED_WAVE_WINDOW_US").ok();
        let floor = cold_hold();
        let (window, clamped) = resolve_wave_window(raw.as_deref(), floor);
        if clamped {
            eprintln!(
                "[pie-sched] PIE_SCHED_WAVE_WINDOW_US={} is below the cold-hold floor \
                 ({floor:?}); clamping to the floor — a near-zero wave window would \
                 disable the wait-all barrier (arrival-order scheduling)",
                raw.as_deref().unwrap_or("<unset>"),
            );
        }
        window
    })
}

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

/// Bounded poll for the quorum-hold wait. The completion channel and new
/// arrivals both preempt the `Decision::Wait` select the instant they fire, so
/// this only bounds the worst-case re-evaluation cadence (a hang backstop).
pub(super) const QUORUM_POLL_US: u64 = 200;

/// The wave-level fire decision. Dense quorum fires carry an empty `missing`
/// set; a straggler demotion fires narrow and names the absentees.
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
    /// The barrier stopped awaiting this pipeline: it missed a full wave
    /// window while the fleet was ready. It still rides any wave it shows
    /// up for, and participation re-promotes it immediately.
    demoted: bool,
    /// This pipeline was absent at one window expiry and got a grace
    /// window instead of demotion. Demotion requires a SECOND consecutive
    /// expiry with the pipeline still absent-and-idle (the anti-stall
    /// fallback survives, one window later); any sign of life —
    /// participation or a fresh submission — clears the flag. Universal
    /// (not just credited backlog): the demotion forensics (V6 iteration
    /// 42) showed absentees arriving in synchronized cohorts of 28–45 of
    /// 64 — a host-global pause, not individual straggling — and a global
    /// pause shorter than two windows must demote nobody, because every
    /// healthy pipeline resumes during the grace window.
    straggler_grace: bool,
    /// The pipeline's last observable liveness: a readiness credit arrived
    /// or one of its fires retired. Demotion requires a FULL wave window of
    /// idleness measured from HERE — not from the wave's gather start. With
    /// run-ahead pipelining a wave gathers across ~2–3 cadences, so a
    /// pipeline whose fire settles late in the gather met an already-expired
    /// window at its natural replenish moment and was demoted 0 ms after its
    /// own settlement (V6 iteration 29: all 135 mid-run demotions in the
    /// census had gap-since-last-settle = 0.0 ms with resubmission
    /// 0.08–15 ms later).
    last_activity: Option<Instant>,
    in_flight: usize,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PipelineEpoch {
    pid: ProcessId,
    generation: u64,
}

pub(super) struct WaitAllPolicy {
    /// Rolling estimate (EWMA, α=1/8) of the wave retire-to-retire cadence
    /// in µs. The straggler idleness threshold is `wave_window + cadence`:
    /// the window constant's own rationale ("must exceed one dense-wave
    /// cadence") calibrated it for the ~3 ms standard-shape cadence, and at
    /// prefix-heavy cadence (~4.5 ms) the fixed 6 ms left ~1.5 ms of margin
    /// — ordinary settle→resubmit cycles looked idle and demoted en masse
    /// (V6 iteration 31: 200–220 demotions/run on the prefix shape).
    cadence_ewma_us: u64,
    last_retire_at: Option<Instant>,
    /// Structural cap — a full batch always fires immediately.
    max_forward_requests: usize,
    /// Batches enqueued but not yet retired; bounded by the configured depth.
    in_flight: usize,
    /// THE wait-set: every active pipeline and its wave participation.
    /// BTreeMap for deterministic `missing` ordering.
    active: BTreeMap<ProcessId, PipelineWaveState>,
    /// Bind controls accepted by the scheduler but not yet completed. These
    /// processes are active missing members even before their first fire, and
    /// a wave holds unconditionally while any of them is absent.
    pending_binds: BTreeMap<ProcessId, usize>,
    generations: BTreeMap<ProcessId, u64>,
    /// Untracked (pipeline-less) requests in the current wave: counted so an
    /// untracked-only batch still fires, never awaited.
    untracked_ready: usize,
    /// When the current wave started gathering.
    wave_started: Option<Instant>,
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
            cadence_ewma_us: 0,
            last_retire_at: None,
            max_forward_requests,
            in_flight: 0,
            active: BTreeMap::new(),
            pending_binds: BTreeMap::new(),
            generations: BTreeMap::new(),
            untracked_ready: 0,
            wave_started: None,
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
            "in_flight={} pending_binds={} untracked_ready={} ever_fired={} wave_age={:?} cold_hold={:?}",
            self.in_flight,
            self.pending_binds.values().sum::<usize>(),
            self.untracked_ready,
            self.ever_fired,
            self.wave_started.map(|since| since.elapsed()),
            self.cold_hold_deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now())),
        );
        for (pid, state) in &self.active {
            let _ = write!(
                out,
                "\n  pipeline {pid}: wave_ready={} demoted={} in_flight={} generation={}",
                state.wave_ready, state.demoted, state.in_flight, state.generation,
            );
        }
        out
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
                let state = self.active_state(pid);
                state.wave_ready += 1;
                state.last_activity = Some(now);
                // A fresh submission ends any grace episode: the pipeline
                // is alive, so its next absence starts a new first-expiry
                // grace instead of inheriting a stale second-strike.
                state.straggler_grace = false;
            }
            None => self.untracked_ready += 1,
        }
    }

    /// A tracked pipeline has submitted work that is not launch-ready yet.
    /// It joins the barrier immediately and remains missing until preparation
    /// publishes a readiness credit.
    pub fn on_pipeline_join(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid {
            self.active_state(pid);
        }
    }

    /// A bind control entered the scheduler. Assembly membership starts here,
    /// before native bind completion or the process's first fire.
    pub fn on_bind_enqueued(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid {
            self.active_state(pid);
            *self.pending_binds.entry(pid).or_default() += 1;
        }
    }

    /// A bind control completed, whether successfully or with an error. The
    /// process remains an active missing member until its first fire arrives
    /// or the existing leave/demotion machinery handles it.
    pub fn on_bind_completed(&mut self, pid: Option<ProcessId>, now: Instant) {
        let Some(pid) = pid else {
            return;
        };
        let Some(count) = self.pending_binds.get_mut(&pid) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.pending_binds.remove(&pid);
        }
        if self.pending_binds.is_empty() && self.wave_started.is_some() {
            self.wave_started = Some(now);
        }
    }

    fn active_state(&mut self, pid: ProcessId) -> &mut PipelineWaveState {
        let generation = *self.generations.entry(pid).or_default();
        self.active.entry(pid).or_insert(PipelineWaveState {
            generation,
            ..PipelineWaveState::default()
        })
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
        if self.untracked_ready == 0 && self.active.values().all(|state| state.wave_ready == 0) {
            self.wave_started = None;
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
        self.pending_binds.remove(&pid);
        *self.generations.entry(pid).or_default() += 1;
        if self.active.is_empty() {
            self.ever_fired = false;
            self.cold_hold_deadline = None;
            self.wave_started = None;
        }
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
            // `at_depth` counts everyone the wave is not waiting on
            // (depth-capped / demoted stragglers).
            if self.pending_binds.contains_key(pid) {
                missing += 1;
            } else if state.demoted || state.in_flight >= configured_max_in_flight() {
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
        self.decide_wave_inner(current_batch_size, None, empty, empty, false, now)
    }

    /// Production decision using the pipelines present in the batch that the
    /// dispatcher can actually build now. Readiness credits behind a control or
    /// preparation barrier cannot make a narrower candidate satisfy quorum.
    pub fn decide_candidate_wave_at(
        &mut self,
        current_batch_size: usize,
        candidate_pipelines: &HashSet<ProcessId>,
        deferred_pipelines: &HashSet<ProcessId>,
        submitted_pipelines: &HashSet<ProcessId>,
        structurally_full: bool,
        now: Instant,
    ) -> WaveDecision {
        self.decide_wave_inner(
            current_batch_size,
            Some(candidate_pipelines),
            deferred_pipelines,
            submitted_pipelines,
            structurally_full,
            now,
        )
    }

    fn decide_wave_inner(
        &mut self,
        current_batch_size: usize,
        candidate_pipelines: Option<&HashSet<ProcessId>>,
        deferred_pipelines: &HashSet<ProcessId>,
        submitted_pipelines: &HashSet<ProcessId>,
        structurally_full: bool,
        now: Instant,
    ) -> WaveDecision {
        // Run-ahead depth cap (R10): the pipe is full — hold; the completion
        // channel preempts the wait the instant a batch retires. Misses are
        // NOT counted while capped: a straggler can't be blamed for a wave
        // that couldn't fire anyway. The wave clock pauses for the same
        // reason — restarting it on every capped decide means the straggler
        // window measures only time the wave could actually fire, so the
        // first post-cap wave can't demote absentees for a hold no
        // participation could have ended.
        if self.in_flight >= configured_max_in_flight() {
            if self.wave_started.is_some() {
                self.wave_started = Some(now);
            }
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
                    || (!state.demoted
                    // Deferred BY THE COMPOSER (capacity / wave token
                    // budget / same-instance dedup): scheduled for the next
                    // wave, not late — never awaited, never a straggler.
                    && !deferred_pipelines.contains(*pid)
                    && state.in_flight < configured_max_in_flight()
                    && candidate_pipelines
                        .map(|pipelines| !pipelines.contains(pid))
                        .unwrap_or(state.wave_ready == 0))
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

        // Assembly hold: a native bind already accepted into the scheduler is
        // submitted work, not a straggler. Keep the wave open so the existing
        // hold-path control drain can finish the bind cohort. Once each bind
        // completes, ordinary readiness and demotion semantics resume.
        if missing
            .iter()
            .any(|pid| self.pending_binds.contains_key(pid))
        {
            return WaveDecision::Wait(Duration::from_micros(QUORUM_POLL_US));
        }

        // Straggler demotion (runahead plan rev 2): the barrier holds for the
        // wave window — new arrivals and completion notifications preempt the
        // bounded poll — then fires the narrower wave that actually gathered
        // and stops awaiting the absentees until they participate again.
        let age = self
            .wave_started
            .map(|started| now.saturating_duration_since(started))
            .unwrap_or_default();
        let window = wave_window();
        if age < window {
            let remaining = window - age;
            return WaveDecision::Wait(remaining.min(Duration::from_micros(QUORUM_POLL_US)));
        }
        let mut any_demoted = false;
        let mut demoted_count = 0usize;
        for pid in &missing {
            if submitted_pipelines.contains(pid) {
                // Submission = presence (north-star item 1): this pipeline
                // has work IN THE ENGINE (queued behind a barrier, preparing,
                // or pre-launch copying) — the barrier may hold for it
                // (density), but it can never be punished. An inferlet that
                // submitted is on time by definition; only silent guests age
                // toward demotion. (The b61297af commit ratified this but
                // only plumbed the parameter; this is the missing consumer.)
                continue;
            }
            if let Some(state) = self.active.get_mut(pid) {
                if !state.straggler_grace {
                    // First expiry absent: one universal grace window — a
                    // credited backlog's fire has arrived (composition
                    // excluded it), and an uncredited absentee is more often
                    // a victim of a host-global pause than a straggler
                    // (iteration 42 forensics: demotions land in
                    // synchronized cohorts of 28–45/64). Whoever is back
                    // before the next expiry was never a straggler.
                    state.straggler_grace = true;
                } else if state.last_activity.is_some_and(|at| {
                    now.saturating_duration_since(at)
                        < window + Duration::from_micros(self.cadence_ewma_us)
                }) {
                    // Active within the window (a fire of its just retired,
                    // or a credit just arrived and was consumed): its
                    // natural replenish moment is NOW — the wave fires
                    // without it, but a pipeline is a straggler only after
                    // a FULL window of idleness measured from its own last
                    // activity, not from the wave's gather start.
                } else {
                    state.demoted = true;
                    state.straggler_grace = false;
                    any_demoted = true;
                    demoted_count += 1;
                    if let Some(stats) = &self.stats {
                        use std::sync::atomic::Ordering::Relaxed;
                        stats.fire.quorum.straggler_demotions.fetch_add(1, Relaxed);
                    }
                }
            }
        }
        if crate::scheduler::fire_timing_full() {
            // Named absentees with their per-pid OUTCOME: "demoted" means
            // punished (excluded until next participation); false means
            // graced (credited backlog) or recently-active (idle-window
            // rule) — still awaited, merely absent from THIS narrow wave.
            let detail: Vec<serde_json::Value> = missing
                .iter()
                .map(|pid| {
                    let state = self.active.get(pid);
                    serde_json::json!({
                        "pid": pid.to_string(),
                        "in_flight": state.map(|s| s.in_flight),
                        "wave_ready": state.map(|s| s.wave_ready),
                        "demoted": state.map(|s| s.demoted),
                        "grace": state.map(|s| s.straggler_grace),
                    })
                })
                .collect();
            crate::scheduler::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "scheduler",
                "event": "straggler_demotion",
                "at_us": crate::scheduler::fire_timing_now_us(),
                "wave_age_us": age.as_micros() as u64,
                "batch_size": current_batch_size,
                "missing": detail,
            }));
        } else if crate::scheduler::ledger_timing_enabled() {
            crate::scheduler::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "scheduler",
                "event": "straggler_demotion",
                "at_us": crate::scheduler::fire_timing_now_us(),
                "wave_age_us": age.as_micros() as u64,
                "batch_size": current_batch_size,
                "missing_count": missing.len(),
                "demoted_count": demoted_count,
            }));
        }
        // A patience-expired fire that punished NOBODY (every absentee is
        // within its own idle grace or credited backlog) is a normal quorum
        // outcome — the barrier waited its window and moved on with everyone
        // unharmed. Only a fire that actually demoted someone is a
        // straggler event (zero-straggler gate: this counter must be 0 for
        // healthy workloads because no pipeline IS a straggler).
        self.record_clause(if any_demoted {
            FireClause::Straggler
        } else {
            FireClause::Quorum
        });
        self.record_wave(missing.len());
        WaveDecision::Fire { missing }
    }

    /// A wave was dispatched: consume exactly one readiness credit for each
    /// submitted row and retain credits for requests already queued behind
    /// this wave. Participation re-promotes a demoted straggler; absentees
    /// stay demoted.
    pub fn on_wave_dispatched(
        &mut self,
        participants: &[Option<ProcessId>],
        now: Instant,
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
                        // Participation re-promotes a demoted straggler and
                        // ends any grace episode (the rotation that once
                        // also lived here is DELETED — operator ruling R2,
                        // 2026-07-17: the one-pass-per-pipeline-per-round
                        // lockstep was a byproduct of budget splitting, and
                        // after the W1 accounting fix it made the fleet
                        // OSCILLATE — a thin wave barred its riders from
                        // the next wave, alternating large/small waves
                        // forever instead of re-gathering dense).
                        state.demoted = false;
                        state.straggler_grace = false;
                        state.in_flight += 1;
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
        if self.untracked_ready == 0 && self.active.values().all(|state| state.wave_ready == 0) {
            self.wave_started = None;
        } else {
            self.wave_started = Some(now);
        }
        epochs
    }

    /// An in-flight batch retired: frees one run-ahead slot so
    /// `decide_wave_at`'s depth cap admits the next wave.
    pub fn on_wave_retired(&mut self, participants: &[Option<PipelineEpoch>]) {
        self.in_flight = self.in_flight.saturating_sub(1);
        let now = Instant::now();
        if let Some(prev) = self.last_retire_at {
            // Clamp samples to 100 ms: end-of-run silences must not poison
            // the estimate.
            let sample = (now.saturating_duration_since(prev).as_micros() as u64).min(100_000);
            self.cadence_ewma_us = if self.cadence_ewma_us == 0 {
                sample
            } else {
                (self.cadence_ewma_us * 7 + sample) / 8
            };
        }
        self.last_retire_at = Some(now);
        for epoch in participants.iter().flatten() {
            if let Some(state) = self.active.get_mut(&epoch.pid)
                && state.generation == epoch.generation
            {
                state.in_flight = state.in_flight.saturating_sub(1);
                state.last_activity = Some(now);
            }
        }
    }

    fn record_clause(&self, clause: FireClause) {
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            match clause {
                // Bootstrap gathers and straggler demotions are recorded
                // separately; dense/capacity waves are the ordinary quorum
                // path. Unconditional (like `record_wave`): stats export
                // these counters in every build, and a default build that
                // reports zero while actively demoting is exactly the
                // silently-wrong stat this scheduler bans.
                FireClause::ColdHold => {
                    stats.fire.quorum.cold_hold_fires.fetch_add(1, Relaxed);
                }
                FireClause::Straggler => {
                    stats.fire.quorum.straggler_fires.fetch_add(1, Relaxed);
                }
                _ => {}
            }
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

    #[test]
    fn wave_window_floors_at_cold_hold() {
        let floor = Duration::from_micros(COLD_HOLD_US);
        // The greedy backdoor: a near-zero request clamps to the floor
        // (and the production caller warns loudly).
        assert_eq!(resolve_wave_window(Some("0"), floor), (floor, true));
        assert_eq!(resolve_wave_window(Some("1"), floor), (floor, true));
        assert_eq!(resolve_wave_window(Some("1999"), floor), (floor, true));
        // At or above the floor the request passes through untouched.
        assert_eq!(resolve_wave_window(Some("2000"), floor), (floor, false));
        assert_eq!(
            resolve_wave_window(Some("12000"), floor),
            (Duration::from_micros(12_000), false)
        );
        // Unset or unparseable values fall back to the default, which
        // clears any sane floor.
        assert_eq!(
            resolve_wave_window(None, floor),
            (Duration::from_micros(WAVE_WINDOW_US), false)
        );
        assert_eq!(
            resolve_wave_window(Some("not-a-number"), floor),
            (Duration::from_micros(WAVE_WINDOW_US), false)
        );
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

    /// W1 consequence regression: once the books drain, the gather clock
    /// re-arms. A fresh fleet starting long after the drained wave must
    /// get a full window before any narrow fire — with the leak
    /// (`untracked_ready` stuck above zero) the `wave_started` reset never
    /// fired and the first decide after any >window lull fired narrow
    /// instantly, marking healthy pipelines stragglers.
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
        // Long after (>> wave window), a fresh fleet gathers: B is ready,
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

    /// While the run-ahead depth cap holds, no participation can end the
    /// wave — so no absentee may age toward demotion (the no-blame rule).
    /// The wave clock restarts on every capped decide; the straggler
    /// window then measures only time the wave could actually fire.
    #[test]
    fn depth_cap_pauses_the_straggler_clock() {
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
        // Far past the wave window, the capped decide still holds — and
        // restarts the wave clock.
        policy.on_pipeline_request(Some(ready), now);
        let capped_until = now + Duration::from_micros(10 * WAVE_WINDOW_US);
        assert_eq!(
            policy.decide_wave_at(1, capped_until),
            WaveDecision::Wait(Duration::from_micros(QUORUM_POLL_US))
        );

        // Release the cap: the wave is young again, so the barrier keeps
        // waiting for `absent` instead of demoting it for time it spent
        // depth-capped.
        policy.on_wave_retired(&epochs[0]);
        let shortly_after = capped_until + Duration::from_micros(QUORUM_POLL_US);
        assert!(matches!(
            policy.decide_wave_at(1, shortly_after),
            WaveDecision::Wait(_)
        ));

        // Demotion still works, counted from the release.
        let past_fresh_window = capped_until + Duration::from_micros(WAVE_WINDOW_US + 100);
        assert_eq!(
            policy.decide_wave_at(1, past_fresh_window),
            WaveDecision::Fire {
                missing: vec![absent]
            }
        );
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
    fn credited_backlog_gets_one_grace_window_before_demotion() {
        // A fires; B is credited (its pass arrived) but batch composition
        // excludes it from the candidate. First window expiry must NOT
        // demote B (backlog, not a straggler); a second consecutive expiry
        // while still credited-and-unbuilt must (anti-stall preserved).
        let mut policy = WaitAllPolicy::new(64, None);
        let (a, b) = (pid(), pid());
        let start = Instant::now();
        policy.on_pipeline_request(Some(a), start);
        policy.on_pipeline_request(Some(b), start);
        let t0 = bootstrap_fire(&mut policy, 2, start);
        // Next round: both credited, only A is in the candidate.
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let candidates: HashSet<ProcessId> = [a].into_iter().collect();
        let expiry1 = t0 + Duration::from_micros(wave_window().as_micros() as u64 + 100);
        assert_eq!(
            policy.decide_candidate_wave_at(
                1,
                &candidates,
                &HashSet::new(),
                &HashSet::new(),
                false,
                expiry1
            ),
            WaveDecision::Fire { missing: vec![b] }
        );
        // B kept its credit and was NOT demoted: it still holds the barrier.
        let epochs = policy.on_wave_dispatched(&[Some(a)], expiry1);
        policy.on_wave_retired(&epochs);
        policy.on_pipeline_request(Some(a), expiry1);
        let expiry2 = expiry1 + Duration::from_micros(wave_window().as_micros() as u64 + 100);
        assert_eq!(
            policy.decide_candidate_wave_at(
                1,
                &candidates,
                &HashSet::new(),
                &HashSet::new(),
                false,
                expiry2
            ),
            WaveDecision::Fire { missing: vec![b] }
        );
        // Second expiry demoted B: it no longer blocks the wave decision.
        let epochs = policy.on_wave_dispatched(&[Some(a)], expiry2);
        policy.on_wave_retired(&epochs);
        policy.on_pipeline_request(Some(a), expiry2);
        assert_eq!(
            policy.decide_candidate_wave_at(
                1,
                &candidates,
                &HashSet::new(),
                &HashSet::new(),
                false,
                expiry2
            ),
            WaveDecision::Fire {
                missing: Vec::new()
            }
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
    fn straggler_demotes_to_a_narrower_wave_after_the_window() {
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let past_cold_hold = bootstrap_fire(&mut policy, 2, t0);

        // Wave 2: only `a` resubmits.
        policy.on_pipeline_request(Some(a), past_cold_hold);
        // Within the wave window the barrier holds for `b`.
        match policy.decide_wave_at(1, past_cold_hold + Duration::from_micros(250)) {
            WaveDecision::Wait(_) => {}
            other => panic!("expected a straggler hold, got {other:?}"),
        }
        // Past the window the wave fires narrow, naming `b` — but the FIRST
        // expiry only grants the universal grace window: `b` is still
        // awaited afterwards.
        let expiry1 = past_cold_hold + wave_window() + Duration::from_micros(100);
        match policy.decide_wave_at(1, expiry1) {
            WaveDecision::Fire { missing } => assert_eq!(missing, vec![b]),
            other => panic!("expected a narrow straggler fire, got {other:?}"),
        }
        let epochs = policy.on_wave_dispatched(&[Some(a)], expiry1);
        policy.on_wave_retired(&epochs);

        // Wave 3: the graced `b` still holds the barrier within the window.
        policy.on_pipeline_request(Some(a), expiry1);
        assert!(matches!(
            policy.decide_wave_at(1, expiry1 + Duration::from_micros(250)),
            WaveDecision::Wait(_)
        ));
        // A second consecutive expiry with `b` still absent-and-idle demotes.
        let expiry2 = expiry1 + wave_window() + Duration::from_micros(100);
        match policy.decide_wave_at(1, expiry2) {
            WaveDecision::Fire { missing } => assert_eq!(missing, vec![b]),
            other => panic!("expected the demoting straggler fire, got {other:?}"),
        }
        let epochs = policy.on_wave_dispatched(&[Some(a)], expiry2);
        policy.on_wave_retired(&epochs);

        // Wave 4: the demoted `b` no longer holds the barrier at all.
        policy.on_pipeline_request(Some(a), expiry2);
        match policy.decide_wave_at(1, expiry2) {
            WaveDecision::Fire { missing } => {
                assert!(missing.is_empty(), "demoted straggler must not be awaited")
            }
            other => panic!("expected an immediate dense fire, got {other:?}"),
        }
        let epochs = policy.on_wave_dispatched(&[Some(a)], expiry2);
        policy.on_wave_retired(&epochs);

        // Wave 5: `b` shows up and rides — participation re-promotes it.
        policy.on_pipeline_request(Some(a), expiry2);
        policy.on_pipeline_request(Some(b), expiry2);
        assert_eq!(
            policy.decide_wave_at(2, expiry2),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
        let epochs = policy.on_wave_dispatched(&[Some(a), Some(b)], expiry2);
        policy.on_wave_retired(&epochs);

        // Wave 6: `a` alone again — the re-promoted `b` holds the barrier
        // within the window, exactly like before its demotion.
        policy.on_pipeline_request(Some(a), expiry2);
        assert!(matches!(
            policy.decide_wave_at(1, expiry2 + Duration::from_micros(250)),
            WaveDecision::Wait(_)
        ));
    }

    #[test]
    fn submitted_pipeline_is_awaited_but_never_demoted() {
        // Submission = presence: a pipeline whose work sits in the engine
        // (preparing / queued behind a barrier / pre-launch copying) may be
        // held for, but never punished — across ANY number of expiries.
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let past_cold_hold = bootstrap_fire(&mut policy, 2, t0);

        // `b`'s next pass is submitted but stuck in preparation: it is in
        // the SUBMITTED set, absent from the candidate, for three expiries.
        policy.on_pipeline_request(Some(a), past_cold_hold);
        let candidates: HashSet<ProcessId> = [a].into_iter().collect();
        let submitted: HashSet<ProcessId> = [b].into_iter().collect();
        let mut expiry = past_cold_hold;
        for _ in 0..3 {
            expiry += wave_window() + Duration::from_micros(100);
            match policy.decide_candidate_wave_at(
                1,
                &candidates,
                &HashSet::new(),
                &submitted,
                false,
                expiry,
            ) {
                WaveDecision::Fire { missing } => assert_eq!(missing, vec![b]),
                other => panic!("expected a narrow fire awaiting b, got {other:?}"),
            }
            let epochs = policy.on_wave_dispatched(&[Some(a)], expiry);
            policy.on_wave_retired(&epochs);
            policy.on_pipeline_request(Some(a), expiry);
        }

        // Still awaited after all of it: the barrier holds for `b` within
        // the window — a demoted pipeline would not hold it.
        assert!(matches!(
            policy.decide_candidate_wave_at(
                1,
                &candidates,
                &HashSet::new(),
                &submitted,
                false,
                expiry + Duration::from_micros(250),
            ),
            WaveDecision::Wait(_)
        ));
    }

    #[test]
    fn pause_survivor_is_regraced_not_demoted() {
        // A pipeline that misses one expiry (host-global pause), resumes,
        // and later misses another expiry gets a FRESH grace window each
        // episode: a new submission ends the episode, so no stale
        // second-strike ever demotes a pipeline that keeps coming back.
        let mut policy = WaitAllPolicy::new(4, None);
        let (a, b) = (pid(), pid());
        let t0 = Instant::now();
        policy.on_pipeline_request(Some(a), t0);
        policy.on_pipeline_request(Some(b), t0);
        let past_cold_hold = bootstrap_fire(&mut policy, 2, t0);

        // Episode 1: `b` misses an expiry — graced, fired around.
        policy.on_pipeline_request(Some(a), past_cold_hold);
        let expiry1 = past_cold_hold + wave_window() + Duration::from_micros(100);
        assert_eq!(
            policy.decide_wave_at(1, expiry1),
            WaveDecision::Fire { missing: vec![b] }
        );
        let epochs = policy.on_wave_dispatched(&[Some(a)], expiry1);
        policy.on_wave_retired(&epochs);

        // `b` resumes and rides a dense wave — the grace episode ends.
        policy.on_pipeline_request(Some(a), expiry1);
        policy.on_pipeline_request(Some(b), expiry1);
        assert_eq!(
            policy.decide_wave_at(2, expiry1),
            WaveDecision::Fire {
                missing: Vec::new()
            }
        );
        let epochs = policy.on_wave_dispatched(&[Some(a), Some(b)], expiry1);
        policy.on_wave_retired(&epochs);

        // Episode 2: `b` misses another expiry — a fresh grace, not the
        // second strike of a demotion.
        policy.on_pipeline_request(Some(a), expiry1);
        let expiry2 = expiry1 + wave_window() + Duration::from_micros(100);
        assert_eq!(
            policy.decide_wave_at(1, expiry2),
            WaveDecision::Fire { missing: vec![b] }
        );
        let epochs = policy.on_wave_dispatched(&[Some(a)], expiry2);
        policy.on_wave_retired(&epochs);

        // Still awaited: the barrier holds for `b` within the next window —
        // a demoted pipeline would not hold it.
        policy.on_pipeline_request(Some(a), expiry2);
        assert!(matches!(
            policy.decide_wave_at(1, expiry2 + Duration::from_micros(250)),
            WaveDecision::Wait(_)
        ));
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
            policy.decide_candidate_wave_at(
                1,
                &HashSet::from([a]),
                &HashSet::new(),
                &HashSet::new(),
                false,
                t
            ),
            WaveDecision::Wait(_)
        ));
        assert_eq!(
            policy.decide_candidate_wave_at(
                2,
                &HashSet::from([a, b]),
                &HashSet::new(),
                &HashSet::new(),
                false,
                t
            ),
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
        assert_eq!(
            policy.decide_candidate_wave_at(
                2,
                &candidates,
                &empty,
                &empty,
                true,
                t + Duration::from_secs(60),
            ),
            WaveDecision::Wait(Duration::from_micros(QUORUM_POLL_US))
        );

        for lane in &lanes[2..] {
            policy.on_bind_completed(Some(*lane), t);
            policy.on_pipeline_request(Some(*lane), t);
        }
        let all_candidates = lanes.into_iter().collect();
        assert_eq!(policy.candidate_state_counts(&all_candidates), (4, 0, 0));
        assert_eq!(
            policy.decide_candidate_wave_at(4, &all_candidates, &empty, &empty, true, t,),
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
        // Way past the window the wave demotes `b` and fires narrow — but
        // only an explicit leave removes membership; time never does.
        match policy.decide_wave_at(1, t + Duration::from_secs(60)) {
            WaveDecision::Fire { missing } => assert_eq!(missing, vec![b]),
            other => panic!("expected a narrow straggler fire, got {other:?}"),
        }
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
