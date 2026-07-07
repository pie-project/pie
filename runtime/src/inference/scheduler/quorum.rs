//! Pure model of the quorum fire rule (overview §7.2; thrust-2 §3 F1–F6).
//!
//! No driver, no async, no timing in the decision path — a synchronous
//! struct that decides *when* a dense batch fires from per-pipeline
//! readiness alone. This is the phase-S0 scaffolding the quorum core
//! (phase S5) wires into the live scheduler: S5 owns per-pipeline next-pass
//! state, one in-flight batch, one queued batch (depth 1), and drives this
//! model each fire; the capacity splitter (`BatchAccumulator`, chunked
//! prefill) runs *before* the clause here (overview §7.2).
//!
//! The rule (one rule, three clauses):
//!
//! - **F1 Quorum.** The moment every *counted* pipeline's next pass is
//!   structurally ready, enqueue the dense batch behind the batch in flight
//!   (depth-1). Steady state: quorum completes mid-flight, bubble zero.
//! - **F2 Idle escape.** Device idle + queue empty → fire the ready subset
//!   now; missing instances are absent (no holes) and rejoin a later fire.
//! - **F3 Cold hold.** Nothing in flight at all → hold sub-millisecond for
//!   arrivals, then fire partial.
//!
//! - **F4 Denominator.** Counts pipelines that *can* be ready this round: a
//!   submitted next pass, or an in-flight pass that has been running ahead.
//!   Host-blocked pipelines (tool call, drained compact) are absent, not
//!   awaited.
//! - **F5 Structural readiness.** Submitted, and every input dependency
//!   satisfied or produced by a pass ahead of it in flight. Genuinely-late
//!   host edges (grammar masks) never gate the batch — they park the
//!   consuming stage at the device cut point (C2), not the fire.
//! - **F6 No estimation.** No lead-time EWMA, no completion prediction here.
//!   Membership is recomputed each fire (stateless, R5) — there is no
//!   hysteresis state to corrupt, which is what keeps the fleet from
//!   bifurcating into alternating half-batches.
#![allow(dead_code)] // wired live by phase S5; exercised by the S0 tests below.

use std::collections::BTreeMap;

/// Opaque pipeline identity. The scheduler keys pipelines on this; the model
/// never interprets it.
pub(crate) type PipelineId = u64;

/// The readiness of a pipeline's *next* pass this round (F4/F5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NextPass {
    /// No submitted next pass this round, or the pipeline is blocked on host
    /// work at a value-dependent boundary (a tool call, §6.2's drained
    /// compact). Absent, not awaited (F4). An in-flight pass that is *not*
    /// running ahead is also `Absent` for fire purposes: it has no next pass
    /// to enqueue yet.
    Absent,
    /// A next pass is submitted, but a structural dependency is not yet
    /// satisfied — e.g. a producer link whose producer is not ahead of it in
    /// flight. NOT the state of a genuinely-late host edge: those park the
    /// stage and leave the pass structurally ready (F5).
    SubmittedWaiting,
    /// Submitted and structurally ready (F5): every input dependency is
    /// satisfied or produced by a pass ahead of it in flight. Includes an
    /// in-flight pipeline whose run-ahead next pass is ready.
    Ready,
}

impl NextPass {
    /// F4: does this pipeline count toward the quorum denominator this round?
    #[inline]
    fn is_counted(self) -> bool {
        matches!(self, NextPass::SubmittedWaiting | NextPass::Ready)
    }

    #[inline]
    fn is_ready(self) -> bool {
        matches!(self, NextPass::Ready)
    }
}

/// A pipeline's participation in the current fire round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Membership {
    /// Readiness of this pipeline's next pass (F4/F5).
    pub next: NextPass,
    /// Opaque batch identity (C3). Pipelines sharing an identity co-batch;
    /// thrust 3 supplies the stage-tuple hash, until then it is the legacy
    /// sampler/shape identity. The model groups by it but never interprets it.
    pub identity: u64,
}

impl Membership {
    pub fn new(next: NextPass, identity: u64) -> Self {
        Self { next, identity }
    }
}

/// Device/queue conditions at the fire-decision point. Booleans only — no
/// durations feed the decision (F6). The single timer the rule keeps
/// (`cold_hold_elapsed`) is a sub-millisecond arrivals hold, not a completion
/// estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Conditions {
    /// The device has no batch executing *right now* (the current in-flight
    /// batch just retired, or none was launched).
    pub device_idle: bool,
    /// Depth of the driver queue behind the in-flight batch. Locked to ≤1
    /// (R10): quorum only enqueues when this is 0.
    pub queue_depth: usize,
    /// Whether the stream is *warm* — a batch is executing, or the fleet has
    /// been actively firing (recently drained). The F2-vs-F3 discriminator: a
    /// warm device that idles takes the idle-escape (F2); a cold device (never
    /// warmed, or long-quiescent) takes the cold-hold (F3).
    pub anything_in_flight: bool,
    /// The sub-millisecond cold-hold window (F3) has elapsed. Only consulted
    /// on a cold start (`!anything_in_flight`).
    pub cold_hold_elapsed: bool,
}

/// Which clause of the fire rule fires this round (or `Hold`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FireClause {
    /// F1: every counted pipeline is ready — enqueue the dense batch behind
    /// the in-flight one (depth-1). The steady-state trigger.
    Quorum,
    /// Depth-2 submit-ahead (G3 bubble): a batch is in flight and below the
    /// cap, but the cohort is only partially ready — fire the ready subset
    /// EAGERLY behind the in-flight batch (via the run-ahead carrier) rather
    /// than JIT-holding for the rest of the cohort. Keeps the driver ring fed
    /// so N+1 device-queues before N retires (bubble → 0). WAR-safe at cap=2.
    SubmitAhead,
    /// F2: device idle with an empty queue — fire the ready subset now.
    IdleEscape,
    /// F3: nothing in flight and the cold-hold window elapsed — fire partial.
    ColdHold,
    /// Do not fire this round.
    Hold,
}

impl FireClause {
    #[inline]
    pub fn fires(self) -> bool {
        !matches!(self, FireClause::Hold)
    }
}

/// The quorum decision model. Holds per-pipeline membership for one round and
/// answers the fire question. Recomputed each fire (R5) — cheap to rebuild,
/// no cross-round state.
#[derive(Clone, Debug, Default)]
pub(crate) struct QuorumModel {
    pipelines: BTreeMap<PipelineId, Membership>,
}

impl QuorumModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record (or overwrite) a pipeline's membership for this round.
    pub fn set(&mut self, id: PipelineId, membership: Membership) {
        self.pipelines.insert(id, membership);
    }

    /// Drop a pipeline entirely (retired / closed).
    pub fn remove(&mut self, id: PipelineId) {
        self.pipelines.remove(&id);
    }

    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// F4: the quorum denominator — pipelines that *can* be ready this round.
    pub fn counted(&self) -> usize {
        self.pipelines
            .values()
            .filter(|m| m.next.is_counted())
            .count()
    }

    /// Pipelines whose next pass is structurally ready (F5). The dense fire
    /// membership — no holes, no padding rows (F2).
    pub fn ready_subset(&self) -> Vec<PipelineId> {
        self.pipelines
            .iter()
            .filter(|(_, m)| m.next.is_ready())
            .map(|(&id, _)| id)
            .collect()
    }

    /// F1: is every counted pipeline ready? (Vacuously false when the
    /// denominator is empty — nothing to fire.)
    pub fn quorum_met(&self) -> bool {
        let counted = self.counted();
        counted > 0 && counted == self.ready_subset().len()
    }

    /// The distinct co-batch identities among the ready subset (C3). Thrust 3
    /// keys graph/batch formation on this; the model only reports it.
    pub fn ready_identities(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .pipelines
            .values()
            .filter(|m| m.next.is_ready())
            .map(|m| m.identity)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// The fire decision (F1–F3). Clause order is normative: quorum first (the
    /// primary trigger), then the escape/cold paths only when the device would
    /// otherwise idle. No timing feeds this beyond the cold-hold boolean (F6).
    pub fn decide(&self, conds: Conditions) -> FireClause {
        // F1 — quorum: every counted pipeline ready, and room in the depth-1
        // queue to enqueue behind the in-flight batch. This is the only clause
        // that fires while a batch is still executing (enqueue-ahead → zero
        // bubble). It requires a *full* quorum: a single laggard that is still
        // counted (submitted-but-waiting) holds the quorum, not the batch —
        // but a host-blocked laggard is absent (F4), so it never holds it.
        if conds.queue_depth == 0 && self.quorum_met() {
            return FireClause::Quorum;
        }

        // Below the quorum, only fire to keep the device from idling, and only
        // if there is a non-empty ready subset to fire (dense — laggards
        // dropped, they rejoin later; §1's resubmission is exactly this
        // membership).
        if self.ready_subset().is_empty() {
            return FireClause::Hold;
        }

        // F3 — cold hold: nothing in flight at all. Hold sub-ms for arrivals,
        // then fire partial. Checked before idle-escape because a cold start
        // is device-idle too, but wants the arrivals hold.
        if !conds.anything_in_flight {
            return if conds.cold_hold_elapsed {
                FireClause::ColdHold
            } else {
                FireClause::Hold
            };
        }

        // F2 — idle escape: device went idle mid-stream with an empty queue.
        // Fire the ready subset immediately; missing instances rejoin later.
        if conds.device_idle && conds.queue_depth == 0 {
            return FireClause::IdleEscape;
        }

        FireClause::Hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready(identity: u64) -> Membership {
        Membership::new(NextPass::Ready, identity)
    }
    fn waiting(identity: u64) -> Membership {
        Membership::new(NextPass::SubmittedWaiting, identity)
    }
    fn absent(identity: u64) -> Membership {
        Membership::new(NextPass::Absent, identity)
    }

    /// Steady state: all counted pipelines ready + room in the depth-1 queue →
    /// quorum fires while the previous batch is still in flight (bubble zero).
    fn steady() -> Conditions {
        Conditions {
            device_idle: false,
            queue_depth: 0,
            anything_in_flight: true,
            cold_hold_elapsed: false,
        }
    }

    #[test]
    fn quorum_fires_when_all_counted_ready() {
        let mut q = QuorumModel::new();
        for id in 0..8 {
            q.set(id, ready(1));
        }
        assert_eq!(q.counted(), 8);
        assert!(q.quorum_met());
        assert_eq!(q.decide(steady()), FireClause::Quorum);
    }

    #[test]
    fn quorum_held_by_a_counted_laggard() {
        // A pipeline that is *submitted but waiting* (structural dep unmet) is
        // counted — it holds the quorum (F1), not the batch. No fire while it
        // waits and the device is busy.
        let mut q = QuorumModel::new();
        for id in 0..7 {
            q.set(id, ready(1));
        }
        q.set(7, waiting(1));
        assert_eq!(q.counted(), 8);
        assert!(!q.quorum_met());
        assert_eq!(q.decide(steady()), FireClause::Hold);
    }

    /// F4 + dense rebatch: a host-blocked laggard is *absent* (not counted), so
    /// the quorum completes over the rest and the dense batch drops it. It
    /// rejoins on its next submit.
    #[test]
    fn dense_rebatch_drops_host_blocked_laggard() {
        let mut q = QuorumModel::new();
        for id in 0..7 {
            q.set(id, ready(1));
        }
        q.set(7, absent(1)); // blocked on a tool call — absent, not awaited.

        assert_eq!(q.counted(), 7, "absent laggard excluded from denominator");
        assert!(q.quorum_met(), "quorum met over the 7 counted");
        assert_eq!(
            q.ready_subset().len(),
            7,
            "dense fire membership drops the laggard — no holes"
        );
        assert_eq!(q.decide(steady()), FireClause::Quorum);

        // Laggard's host work finishes → it submits and becomes ready → it
        // re-enters the denominator on the next fire (stateless rejoin, R5).
        q.set(7, ready(1));
        assert_eq!(q.counted(), 8);
        assert_eq!(q.ready_subset().len(), 8);
    }

    /// Cold start: a lone ready pipeline with nothing in flight fires
    /// immediately via quorum (a single-pipeline quorum is met at once — no
    /// wait for the cold-hold timer).
    #[test]
    fn cold_start_fires_immediately_on_quorum() {
        let mut q = QuorumModel::new();
        q.set(0, ready(1));
        let cold = Conditions {
            device_idle: true,
            queue_depth: 0,
            anything_in_flight: false,
            cold_hold_elapsed: false, // timer NOT elapsed — quorum still fires
        };
        assert_eq!(q.decide(cold), FireClause::Quorum);
    }

    /// Cold start with a still-waiting member (no quorum): hold sub-ms, then
    /// fire the ready subset partial (F3).
    #[test]
    fn cold_hold_then_fires_partial() {
        let mut q = QuorumModel::new();
        q.set(0, ready(1));
        q.set(1, waiting(1)); // counted, so no quorum
        let mut cold = Conditions {
            device_idle: true,
            queue_depth: 0,
            anything_in_flight: false,
            cold_hold_elapsed: false,
        };
        assert_eq!(q.decide(cold), FireClause::Hold, "hold during the window");
        cold.cold_hold_elapsed = true;
        assert_eq!(q.decide(cold), FireClause::ColdHold, "then fire partial");
    }

    /// F2 idle escape: mid-stream the device goes idle with an empty queue and
    /// no full quorum — fire the ready subset now rather than bubble.
    #[test]
    fn idle_escape_fires_ready_subset() {
        let mut q = QuorumModel::new();
        q.set(0, ready(1));
        q.set(1, waiting(1)); // counted → no quorum
        let idle = Conditions {
            device_idle: true,
            queue_depth: 0,
            anything_in_flight: true, // mid-stream, not cold
            cold_hold_elapsed: false,
        };
        assert_eq!(q.decide(idle), FireClause::IdleEscape);
    }

    #[test]
    fn no_fire_when_nothing_ready() {
        let mut q = QuorumModel::new();
        q.set(0, absent(1));
        q.set(1, absent(1));
        assert_eq!(q.counted(), 0);
        assert!(!q.quorum_met());
        let idle = Conditions {
            device_idle: true,
            queue_depth: 0,
            anything_in_flight: false,
            cold_hold_elapsed: true,
        };
        assert_eq!(q.decide(idle), FireClause::Hold);
    }

    /// Quorum does not enqueue when the depth-1 queue is already occupied
    /// (R10): it waits for the queued batch to promote first.
    #[test]
    fn quorum_respects_depth_one_queue() {
        let mut q = QuorumModel::new();
        for id in 0..4 {
            q.set(id, ready(1));
        }
        let mut conds = steady();
        conds.queue_depth = 1; // a batch already queued behind the in-flight one
        assert!(q.quorum_met());
        assert_eq!(q.decide(conds), FireClause::Hold);
    }

    /// Convoy anti-bifurcation. Perturb one pipeline of an 8-wide homogeneous
    /// decode fleet with a one-step host stall and assert the fleet
    /// re-converges to full-batch fires within one round — never settling into
    /// alternating half-batches. The mechanism: membership is recomputed each
    /// fire with no hysteresis (R5/F6), and quorum (not fire-on-idle) is the
    /// primary trigger, so a transient absence dents the batch for exactly the
    /// round it is absent.
    #[test]
    fn convoy_anti_bifurcation() {
        const FLEET: u64 = 8;
        const ROUNDS: usize = 12;
        const STALL_ROUND: usize = 5;
        const STALLED: PipelineId = 3;

        let mut fire_sizes = Vec::new();
        for round in 0..ROUNDS {
            // Rebuild membership from scratch each round (R5: stateless).
            let mut q = QuorumModel::new();
            for id in 0..FLEET {
                // The stalled pipeline is host-blocked (absent) for exactly one
                // round; everyone else is ready.
                let blocked = round == STALL_ROUND && id == STALLED;
                q.set(id, if blocked { absent(1) } else { ready(1) });
            }
            // Steady state: previous batch in flight, quorum enqueues behind it.
            assert_eq!(q.decide(steady()), FireClause::Quorum, "round {round}");
            fire_sizes.push(q.ready_subset().len());
        }

        // Exactly one dented round (the stall), size 7; every other round full.
        let dented: Vec<usize> = fire_sizes
            .iter()
            .copied()
            .filter(|&s| s != FLEET as usize)
            .collect();
        assert_eq!(dented, vec![7], "one dip of size 7, no others: {fire_sizes:?}");
        assert_eq!(fire_sizes[STALL_ROUND], 7);
        assert_eq!(fire_sizes[STALL_ROUND + 1], 8, "re-converged next round");
        // No round ever bifurcates into a half-batch.
        assert!(
            fire_sizes.iter().all(|&s| s >= 7),
            "no half-batch bifurcation: {fire_sizes:?}"
        );
    }

    // ── Discrete-event fleet simulation (S5 mock timing harness) ─────────────
    //
    // A deterministic tick-driven driver of the quorum model over a fleet — the
    // "mock timing harness" the S5 exit calls for, but discrete (no wall clock)
    // so it is flake-free. A batch occupies the device for `COMPUTE_TICKS`; the
    // depth-1 queue holds one enqueued batch behind the in-flight one. Each tick
    // rebuilds membership (R5) from a per-pipeline readiness closure, runs the
    // quorum decision, and advances the device — tracking device-idle "bubble"
    // ticks and the per-fire clause/size.

    struct FleetSim {
        fleet: u64,
        compute_ticks: u32,
        in_flight_remaining: Option<u32>,
        queued: Option<usize>,
        started: bool,
        idle_ticks_after_warmup: u32,
        fires: Vec<(FireClause, usize)>,
    }

    impl FleetSim {
        fn new(fleet: u64, compute_ticks: u32) -> Self {
            Self {
                fleet,
                compute_ticks,
                in_flight_remaining: None,
                queued: None,
                started: false,
                idle_ticks_after_warmup: 0,
                fires: Vec::new(),
            }
        }

        /// Run `ticks` steps; `state(id, tick)` gives each pipeline's next-pass
        /// state. `warmup` ticks are excluded from the bubble count.
        fn run(&mut self, ticks: u32, warmup: u32, state: impl Fn(u64, u32) -> NextPass) {
            for tick in 0..ticks {
                // 1. Advance the device: retire the in-flight batch when its
                //    compute elapses, promoting the depth-1 queued batch.
                if let Some(rem) = self.in_flight_remaining {
                    if rem <= 1 {
                        self.in_flight_remaining =
                            self.queued.take().map(|_| self.compute_ticks);
                    } else {
                        self.in_flight_remaining = Some(rem - 1);
                    }
                }

                // 2. Membership (R5: rebuilt each tick).
                let mut q = QuorumModel::new();
                for id in 0..self.fleet {
                    q.set(id, Membership::new(state(id, tick), 1));
                }

                // 3. Conditions. `anything_in_flight` is the F2-vs-F3
                //    discriminator ("stream warm"): a batch is executing OR the
                //    fleet has been actively firing — so a mid-stream drain takes
                //    the idle-escape path (F2), not the cold-start hold (F3).
                let conds = Conditions {
                    device_idle: self.in_flight_remaining.is_none(),
                    queue_depth: self.queued.is_some() as usize,
                    anything_in_flight: self.in_flight_remaining.is_some() || self.started,
                    cold_hold_elapsed: true, // steady-state sim skips the cold window
                };

                // 4. Decide + act.
                let clause = q.decide(conds);
                let ready_n = q.ready_subset().len();
                match clause {
                    FireClause::Quorum | FireClause::SubmitAhead => {
                        if self.in_flight_remaining.is_none() {
                            self.in_flight_remaining = Some(self.compute_ticks);
                        } else {
                            self.queued = Some(ready_n);
                        }
                        self.started = true;
                        self.fires.push((clause, ready_n));
                    }
                    FireClause::IdleEscape | FireClause::ColdHold => {
                        self.in_flight_remaining = Some(self.compute_ticks);
                        self.started = true;
                        self.fires.push((clause, ready_n));
                    }
                    FireClause::Hold => {}
                }

                // 5. Bubble accounting: a device-idle tick with ready work held.
                if tick >= warmup
                    && self.started
                    && self.in_flight_remaining.is_none()
                    && ready_n > 0
                {
                    self.idle_ticks_after_warmup += 1;
                }
            }
        }
    }

    /// A homogeneous decode fleet (every pipeline ready every tick) fires full
    /// batches via quorum and NEVER bubbles: the depth-1 queue is always primed
    /// before the in-flight batch retires (F1 enqueue-ahead → zero bubble).
    #[test]
    fn homogeneous_fleet_zero_bubble() {
        let mut sim = FleetSim::new(8, 3);
        sim.run(60, 6, |_id, _tick| NextPass::Ready);
        assert_eq!(
            sim.idle_ticks_after_warmup, 0,
            "steady-state homogeneous fleet must not bubble the device"
        );
        // Every steady-state fire is a full-cohort quorum enqueue.
        let steady_fires: Vec<usize> = sim
            .fires
            .iter()
            .filter(|(c, _)| *c == FireClause::Quorum)
            .map(|(_, n)| *n)
            .collect();
        assert!(!steady_fires.is_empty());
        assert!(
            steady_fires.iter().all(|&n| n == 8),
            "all quorum fires full-width: {steady_fires:?}"
        );
    }

    /// An agentic fleet (pipelines cycling ready / submitted-waiting / host-
    /// blocked) keeps the device fed via the escape path when quorum can't
    /// complete, and no pipeline starves. A `SubmittedWaiting` pipeline holds the
    /// quorum (counted, not ready); when the in-flight batch then retires with an
    /// empty queue, the escape fires the ready subset (agentic fleets live in the
    /// escape, F-note) — dense, exactly its ready set.
    #[test]
    fn agentic_fleet_escape_dominates_no_starvation() {
        let mut sim = FleetSim::new(8, 2);
        // Deterministic pseudo-random per-pipeline state: a cheap hash buckets
        // each (id, tick) into ready / waiting / blocked.
        let state = |id: u64, tick: u32| -> NextPass {
            let h = (id.wrapping_mul(2654435761).wrapping_add(tick as u64 * 40503)) & 0xF;
            match h {
                0..=1 => NextPass::Absent,          // ~12% host-blocked
                2..=4 => NextPass::SubmittedWaiting, // ~19% counted-but-waiting
                _ => NextPass::Ready,                // ~69% ready
            }
        };
        sim.run(200, 10, state);

        // The escape path is exercised (a waiting pipeline holds quorum, the
        // device drains, escape fires the ready subset).
        let escapes = sim
            .fires
            .iter()
            .filter(|(c, _)| *c == FireClause::IdleEscape)
            .count();
        assert!(escapes > 0, "agentic fleet must use the idle-escape path");
        // Every fire is dense (only ready pipelines, no padding) within bounds.
        assert!(
            sim.fires.iter().all(|&(_, n)| n >= 1 && n <= 8),
            "dense fires within fleet bounds: {:?}",
            sim.fires
        );
        // Liveness: sustained progress, no deadlock/starvation.
        let total: usize = sim.fires.iter().map(|(_, n)| *n).sum();
        assert!(
            total > 8 * 40,
            "fleet must make sustained progress, got {total} tokens over 200 ticks"
        );
    }

    /// A persistently host-blocked pipeline never holds the batch: the quorum
    /// completes over the remaining ready pipelines every round (F4 — the absent
    /// pipeline is not in the denominator), so the fleet fires full-minus-one
    /// steadily with zero bubble.
    #[test]
    fn persistently_host_blocked_excluded_from_denominator() {
        let mut sim = FleetSim::new(8, 3);
        // Pipeline 7 is host-blocked for the entire run (a long tool call).
        sim.run(60, 6, |id, _tick| {
            if id == 7 { NextPass::Absent } else { NextPass::Ready }
        });
        assert_eq!(sim.idle_ticks_after_warmup, 0, "absent pipeline must not bubble the device");
        let steady: Vec<usize> = sim
            .fires
            .iter()
            .filter(|(c, _)| *c == FireClause::Quorum)
            .map(|(_, n)| *n)
            .collect();
        assert!(
            steady.iter().all(|&n| n == 7),
            "quorum completes over the 7 counted, not awaiting the absent 8th: {steady:?}"
        );
    }
}
