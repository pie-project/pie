//! Vesuvius frame scheduling — THE scheduler policy: the
//! **wait-for-all-active-lanes** quorum rule at frame granularity. Wait
//! until every awaited lane's next FRAME is fully submitted, then seal the
//! dense epoch and dispatch its k waves in slot order.
//!
//! Every deployment routes here, including the default `PIE_FRAME_SIZE=1`:
//! a 1-slot frame IS a wave, so k = 1 reproduces the per-wave wait-all
//! barrier (each tracked fire arrives as its own single-fire frame; a seal
//! boundary is a wave boundary). The former per-wave `WaitAllPolicy`
//! (scheduler/quorum.rs) was this policy specialized to k = 1 and is folded
//! in — its run-ahead depth lever, cold hold, and watchdog constants live
//! here now.
//!
//! A *frame* is k consecutive waves submitted as one unit per lane: the guest
//! supplies exactly k ordered slots (`forward.submit`), slot i executes in
//! wave i, and a `none` slot is a no-op for that wave. The scheduler:
//!
//! - collects per-lane frame submissions (each fire carries a
//!   [`FrameStamp`]);
//! - HOLDS the next seal until every awaited lane's oldest queued frame is
//!   arrival-complete — the infinite wait-all rule: membership changes only
//!   through explicit close/leave/first-fire events, and the 1 s watchdog
//!   only REPORTS a non-responsive lane, never evicts it;
//! - holds while a JOIN is in flight: a process in bring-up (bind
//!   accepted, no fire yet) is staged; once it acquires a contended
//!   execution permit it is a join-in-flight the seal waits for by
//!   identity, and while a freed slot has a staged taker the seal waits
//!   for that handoff — so a cohort turnover gathers the incoming herd
//!   instead of sealing narrow epochs. A bind alone holds nothing: a
//!   live rebinder is already wait-set-held through its lane, and an
//!   unadmitted process cannot fire;
//! - seals from every ready lane (deterministic first-fit in lane-id order
//!   against the per-wave token/row budgets — pure arithmetic over declared
//!   demand, never timing). A lane deferred by capacity is served in the
//!   same structurally partitioned round without re-awaiting the lanes
//!   already served ([`FramePolicy::round_served`], the quorum's round rule);
//! - seals EARLY and overlaps frames on-stream: the next frame seals the
//!   moment the wait-all gate holds — normally while the current frame
//!   executes — and its waves post behind the executing frame's tail at
//!   the run-ahead depth. There is no launch-time barrier: the driver's
//!   device-side readiness gate (`pass_commit` channel tickets) orders
//!   dependent fires by stream order, and a frame-boundary dependency is
//!   structurally identical to an intra-frame one. Posting is globally
//!   ordered (seal order across frames, slot order within one), a RETRY
//!   replays through the globally wave-ordered makeup set, and settlement
//!   only gates resource reclamation, never a launch;
//! - releases a gracefully closed lane from the wait-set immediately while
//!   its accepted frames drain to settlement.
//!
//! The policy is a pure state machine: the worker owns the queue and the
//! driver lane, and drives this through the `on_*` bookkeeping calls plus
//! [`FramePolicy::plan_dispatch`]. Every fire id that enters a sealed
//! frame leaves it through exactly one of `on_fires_posted` →
//! `on_fire_retired` (possibly cycling through the makeup set on RETRY) or
//! `on_fire_dropped` (rejected while queued).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::stats::SchedulerStats;
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

/// The frame identity one fire carries from `forward.submit`: which lane
/// (pipeline scope), which frame of that lane, which wave slot, and how many
/// fires the whole frame holds (so arrival completeness is decidable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStamp {
    pub lane: ProcessId,
    pub seq: u64,
    pub slot: u32,
    pub fires: u32,
}

/// Bootstrap gather window before the FIRST seal of an assembly episode, so
/// a co-launched fleet's first frames land in one sealed epoch instead of a
/// narrow head frame.
const COLD_HOLD_US: u64 = 2_000;

/// Deploy lever for M2 frame-group settlement deferral (`settle_defer` on
/// non-tail waves). DEFAULT OFF: with per-wave publication kept (spec §6.2),
/// the deferrable bookkeeping is microseconds while frame-granular
/// completion resolution couples the posting window to frame-sized
/// retirement — measured a net k>1 loss on the CUDA driver (see
/// vesuvius-phase2.md "M2+M3 measured outcome"). The driver machinery is
/// live either way; this only gates whether the engine marks waves.
pub(super) fn settle_defer_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("PIE_FRAME_SETTLE_DEFER").is_ok_and(|value| value == "1")
    })
}

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

/// Liveness watchdog for a blocked gather. Report-only, exactly as in the
/// per-wave quorum: it never removes a member and never fires a narrow
/// epoch — an unresponsive lane leaves only through close/terminate.
const STRICT_WATCHDOG_US: u64 = 1_000_000;


struct ArrivedFire {
    slot: u32,
    /// `None` when the fire was rejected at scheduler admission — it counts
    /// toward arrival completeness but never dispatches.
    fire_id: Option<u64>,
    tokens: usize,
    rows: usize,
}

struct PendingFrame {
    seq: u64,
    /// Fires this frame is declared to hold (host-adjusted down on a
    /// mid-frame submit failure via [`FramePolicy::on_frame_truncated`]).
    expected: u32,
    fires: Vec<ArrivedFire>,
}

impl PendingFrame {
    fn is_complete(&self) -> bool {
        self.fires.len() >= self.expected as usize
    }
}

struct LaneState {
    owner: Option<ProcessId>,
    /// Wait-set membership: joined on the lane's first stamped fire, held
    /// through every later frame (an idle lane between frames is a missing
    /// member the seal waits on), released only by close/terminate.
    awaited: bool,
    frames: VecDeque<PendingFrame>,
}

/// One sealed (immutable) frame: per-wave fire-id lists in admission order.
struct SealedFrame {
    waves: Vec<Vec<u64>>,
    fire_waves: HashMap<u64, usize>,
    /// Next wave index to open (waves below it are fully posted/resolved).
    next_wave: usize,
    /// Fire ids of the OPEN wave not yet posted or dropped.
    open_remaining: HashSet<u64>,
    /// Fires that came back RETRY after posting: they re-dispatch before
    /// anything else opens, GLOBALLY wave-ordered — the oldest frame's
    /// oldest makeup wave first, and only once every earlier wave (of this
    /// frame and every frame sealed before it) has drained. Per-lane slot
    /// order is enforced by channel tickets; the makeup gate keeps the pipe
    /// from racing ahead of a known-uncommitted step.
    makeup: HashSet<u64>,
    /// Posted fires not yet retired.
    outstanding: usize,
    /// Posted-unretired fires per wave index — the chaining depth gate and
    /// the makeup ordering guard both read this.
    wave_outstanding: Vec<usize>,
}

impl SealedFrame {
    fn is_complete(&self) -> bool {
        self.next_wave >= self.waves.len()
            && self.open_remaining.is_empty()
            && self.outstanding == 0
            && self.makeup.is_empty()
    }
}

/// What the worker should do next for frame-managed launches.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum FramePlan {
    /// Dispatch (a subset of) these fire ids now, in the given order.
    Dispatch(Vec<u64>),
    /// Nothing dispatchable yet; re-decide after the bootstrap hold or at
    /// the blocked-gather watchdog deadline.
    Hold(Duration),
    /// No sealed work and no seal candidates: park until an arrival.
    Park,
}

pub(super) struct FramePolicy {
    k: usize,
    /// Whether the wave returned by the LAST `plan_dispatch` call belongs to
    /// a sealed frame with a later still-queued wave — i.e. its launches may
    /// carry `settle_defer = 1` (M2 frame-group settlement). Makeup replays,
    /// 1-slot frames, and riders never defer.
    last_plan_defers: bool,
    max_wave_tokens: usize,
    max_wave_rows: usize,
    /// THE wait-set plus each lane's queued frames. BTreeMap for
    /// deterministic admission order.
    lanes: BTreeMap<ProcessId, LaneState>,
    sealed: VecDeque<SealedFrame>,
    /// Bind controls accepted by the scheduler but not yet completed.
    /// Feeds [`FramePolicy::has_pending_binds`] (the worker defers teardown
    /// closes while bring-up owns the driver lane); binds do NOT hold the
    /// seal — a lane's own wait-set membership covers a live rebinder, and
    /// bring-up lanes are gathered through `staged`/`pending_joins`.
    pending_binds: BTreeMap<ProcessId, usize>,
    /// Successor pool: processes in bring-up (first bind control accepted)
    /// whose lane has not yet submitted its first stamped fire. While a
    /// free execution slot exists (`pending_slots > 0`), one of these is
    /// about to take it — the seal waits so the join lands in this
    /// boundary instead of a narrow epoch.
    staged: BTreeSet<ProcessId>,
    /// Released execution slots not yet re-consumed by an admission:
    /// +1 on `on_execution_slot_released`, saturating -1 on EVERY
    /// `on_execution_slot_consumed` (uncontended admissions notify too —
    /// the semaphore launders released permits into the free pool, so
    /// only drain-on-every-admission keeps the balance honest; the
    /// saturation absorbs initial-pool consumptions, and a release is
    /// always mailed before its consumer can acquire, so a release-paired
    /// drain is never lost to the clamp). A positive balance with a
    /// non-empty `staged` pool means a successor's admission is imminent:
    /// the seal waits. Multi-driver note: consume/release broadcasts reach
    /// every driver's policy, and it is the GLOBAL admission semaphore
    /// (one pool across drivers) that bounds outstanding consumes to
    /// capacity — that is what keeps each policy's balance from going
    /// negative under foreign-pid traffic.
    pending_slots: u64,
    /// Identity-paired in-flight joins: a parked process that ACQUIRED its
    /// execution permit but whose first stamped fire has not arrived yet.
    /// The seal waits for exactly these lanes (removed by first fire and
    /// by every leave path, so a joiner that dies cannot wedge the seal).
    joins_in_flight: BTreeSet<ProcessId>,
    /// Processes that consumed an execution slot and still hold it. A
    /// Terminate leave of a member moves it to `departing`: its slot is
    /// now certain to resolve (the permit's only exit is the capped
    /// teardown, which broadcasts the release — or, on the no-runtime
    /// error path, an explicit forfeit — after its terminate leave).
    slotted: BTreeSet<ProcessId>,
    /// Slot holders between their Terminate leave and their teardown's
    /// release (or forfeit) broadcast, identity-paired. The leave is
    /// delivered pass-granular while the release waits on the teardown
    /// task, so a seal check in that window sees a zero `pending_slots`
    /// balance with departures already recorded — without this set the
    /// seal would close on the partial cohort and split the fleet, and
    /// the split persists for the rest of the run (run-ahead lead is
    /// hysteretic). Every entry resolves: the same teardown that owns the
    /// permit sends the disarm after its leave (same-producer FIFO), and
    /// the leaked-permit error path disarms via forfeit.
    departing: BTreeSet<ProcessId>,
    /// Liveness-only deadline for the current blocked-gather episode.
    strict_watchdog_deadline: Option<Instant>,
    cold_hold_deadline: Option<Instant>,
    ever_sealed: bool,
    /// Probe sink (`profile-fire` wave counters); `None` in unit tests.
    stats: Option<Arc<SchedulerStats>>,
}

impl FramePolicy {
    pub fn new(
        k: usize,
        max_wave_rows: usize,
        max_wave_tokens: usize,
        stats: Option<Arc<SchedulerStats>>,
    ) -> Self {
        Self {
            k,
            max_wave_tokens,
            max_wave_rows,
            last_plan_defers: false,
            lanes: BTreeMap::new(),
            sealed: VecDeque::new(),
            pending_binds: BTreeMap::new(),
            staged: BTreeSet::new(),
            pending_slots: 0,
            joins_in_flight: BTreeSet::new(),
            slotted: BTreeSet::new(),
            departing: BTreeSet::new(),
            strict_watchdog_deadline: None,
            cold_hold_deadline: None,
            ever_sealed: false,
            stats,
        }
    }

    /// Whether this deployment runs 1-slot frames (`PIE_FRAME_SIZE=1`, the
    /// default): a frame is a wave, and the worker synthesizes a per-fire
    /// stamp at admission instead of the guest submitting frames.
    /// See [`FramePolicy::last_plan_defers`]: whether the most recent
    /// `plan_dispatch` wave may post with `settle_defer = 1`.
    pub fn last_plan_defers_settlement(&self) -> bool {
        self.last_plan_defers
    }

    pub fn single_slot(&self) -> bool {
        self.k == 1
    }

    /// A stamped fire was accepted into the scheduler queue.
    pub fn on_fire_enqueued(
        &mut self,
        stamp: FrameStamp,
        owner: Option<ProcessId>,
        fire_id: u64,
        tokens: usize,
        rows: usize,
    ) {
        self.record_arrival(
            stamp,
            owner,
            ArrivedFire {
                slot: stamp.slot,
                fire_id: Some(fire_id),
                tokens,
                rows,
            },
        );
    }

    /// A stamped fire was rejected at scheduler admission: it still counts
    /// toward its frame's arrival completeness so the frame can seal (its
    /// surviving fires execute; the guest observed the rejection).
    pub fn on_fire_rejected_at_admission(&mut self, stamp: FrameStamp, owner: Option<ProcessId>) {
        self.record_arrival(
            stamp,
            owner,
            ArrivedFire {
                slot: stamp.slot,
                fire_id: None,
                tokens: 0,
                rows: 0,
            },
        );
    }

    fn record_arrival(&mut self, stamp: FrameStamp, owner: Option<ProcessId>, fire: ArrivedFire) {
        // Staged -> live promotion: the owner's fire arrived, so its
        // wait-set membership takes over from the join-in-flight hold.
        // Keyed by OWNER (the process id): `staged`/`joins_in_flight`
        // are process-scoped (bind and admission events carry process
        // ids) while `stamp.lane` is the pipeline scope id.
        if let Some(owner) = owner {
            self.staged.remove(&owner);
            self.joins_in_flight.remove(&owner);
        }
        let lane = self.lanes.entry(stamp.lane).or_insert_with(|| LaneState {
            owner,
            awaited: true,
            frames: VecDeque::new(),
        });
        if lane.owner.is_none() {
            lane.owner = owner;
        }
        let frame = match lane.frames.iter_mut().find(|frame| frame.seq == stamp.seq) {
            Some(frame) => frame,
            None => {
                lane.frames.push_back(PendingFrame {
                    seq: stamp.seq,
                    expected: stamp.fires,
                    fires: Vec::with_capacity(stamp.fires as usize),
                });
                lane.frames.back_mut().expect("frame just pushed")
            }
        };
        frame.expected = frame.expected.max(stamp.fires);
        frame.fires.push(fire);
    }

    /// The host failed a frame mid-submit: only `submitted` fires exist.
    pub fn on_frame_truncated(&mut self, lane: ProcessId, seq: u64, submitted: u32) {
        if let Some(lane) = self.lanes.get_mut(&lane)
            && let Some(frame) = lane.frames.iter_mut().find(|frame| frame.seq == seq)
        {
            frame.expected = submitted;
        }
    }

    /// A bind control entered the scheduler. A bind does not hold the seal:
    /// a live rebinder is already wait-set-held through its lane, and a
    /// bring-up process (no lane yet) enters the `staged` successor pool —
    /// the seal waits for it only once a slot opens for it
    /// ([`FramePolicy::on_execution_slot_released`]) or it acquires one
    /// ([`FramePolicy::on_execution_slot_consumed`]).
    pub fn on_bind_enqueued(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid {
            *self.pending_binds.entry(pid).or_default() += 1;
            // Process-scoped (pid is the owning process). A live rebinder
            // gets a transient entry too — cleared by its next fire — which
            // can at most extend a gather by the rebind window.
            self.staged.insert(pid);
        }
    }

    /// Bootstrap: seed the slot balance with the execution pool's initial
    /// free capacity, so the "free slot with a staged taker" hold covers
    /// the COLD START by the same rule as a turnover — the first seal
    /// waits for the whole co-launched fleet's admissions and first fires.
    /// (A ragged first epoch otherwise starts lead-less lanes, and lead is
    /// hysteretic: they pace every seal of the first generation at the
    /// full commit roundtrip.) Thereafter the balance stays exact: -1 per
    /// admission, +1 per release.
    pub fn preload_free_slots(&mut self, slots: usize) {
        self.pending_slots = slots as u64;
    }

    /// A retiring process's deferred teardown dropped its execution permit
    /// (capped deployments only). While the freed slot stays unconsumed and
    /// a successor is staged, the seal holds — the successor's admission and
    /// first fire are imminent. Resolves the holder's departure by identity
    /// (its terminate leave always precedes this broadcast: both are sent
    /// by the teardown task, in that order).
    pub fn on_execution_slot_released(&mut self, pid: ProcessId) {
        self.departing.remove(&pid);
        self.pending_slots += 1;
    }

    /// The no-runtime teardown error path leaked the holder's permit
    /// (`std::mem::forget`): the slot is destroyed, not freed. Resolves the
    /// departure WITHOUT crediting `pending_slots` — the semaphore capacity
    /// shrank by one and the balance must agree, and a departure entry that
    /// never resolves would hold every seal with a staged successor.
    pub fn on_execution_slot_forfeited(&mut self, pid: ProcessId) {
        self.departing.remove(&pid);
    }

    /// A process acquired its execution permit (every capped admission
    /// notifies, uncontended ones included). Its first fire is imminent:
    /// the seal now waits for `pid` ITSELF, identity-paired, so no event
    /// interleaving can make the policy wait for the wrong process (the
    /// anonymous-counting predecessor deadlocked exactly that way).
    /// Guarded on `staged`: only a process that bound on THIS driver can
    /// fire here.
    pub fn on_execution_slot_consumed(&mut self, pid: ProcessId) {
        self.pending_slots = self.pending_slots.saturating_sub(1);
        self.slotted.insert(pid);
        if self.staged.contains(&pid) {
            self.joins_in_flight.insert(pid);
        }
    }

    /// A slot holder's Terminate leave arrived: its release (or forfeit)
    /// broadcast is now in flight (the permit's only exit is the capped
    /// teardown, which leaves first and resolves second). The seal treats
    /// the imminent slot like a freed one — without this, a seal check
    /// between the leave and the resolution sees `pending_slots == 0` and
    /// closes on a partial cohort. Guarded on `slotted` so only an actual
    /// holder's first Terminate arms; the exit funnel emits more than one
    /// leave per process, and the worker's tombstone dedup normally stops
    /// duplicates before they reach here (this guard is defense in depth).
    pub fn on_slotted_terminate(&mut self, pid: ProcessId) {
        if self.slotted.remove(&pid) {
            self.departing.insert(pid);
        }
    }

    /// A bind control completed, whether successfully or with an error. The
    /// lane itself joins the wait-set at its first stamped fire.
    /// Whether any lane's bind is still in assembly (the seal is
    /// bind-held): the cohort-boundary window in which the worker defers
    /// teardown closes so fresh-lane bring-up owns the driver lane.
    pub fn has_pending_binds(&self) -> bool {
        !self.pending_binds.is_empty()
    }


    pub fn on_bind_completed(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid
            && let Some(count) = self.pending_binds.get_mut(&pid)
        {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.pending_binds.remove(&pid);
            }
        }
    }

    /// A pipeline scope left. `purge_queued` for Terminate/Suspend (its
    /// queued fires were rejected); graceful Close releases the lane from
    /// the wait-set immediately but keeps queued frames — the
    /// already-accepted fires drain to settlement.
    pub fn on_lane_leave(&mut self, lane: ProcessId, purge_queued: bool) {
        if purge_queued {
            self.lanes.remove(&lane);
        } else if let Some(state) = self.lanes.get_mut(&lane) {
            state.awaited = false;
            if state.frames.is_empty() {
                self.lanes.remove(&lane);
            }
        }
        self.pending_binds.remove(&lane);
        self.forget_staged(lane);
        self.maybe_reset_episode();
    }

    /// Every scope owned by `owner` left (process terminate/suspend).
    pub fn on_process_leave(&mut self, owner: ProcessId) {
        self.lanes.retain(|_, lane| lane.owner != Some(owner));
        self.pending_binds.remove(&owner);
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

    /// A staged or joining successor departed before its first fire: the
    /// seal must never wait for a lane that cannot arrive.
    fn forget_staged(&mut self, pid: ProcessId) {
        self.staged.remove(&pid);
        self.joins_in_flight.remove(&pid);
    }

    /// Mirror of the quorum's empty-wait-set re-arm: when the last awaited
    /// lane leaves, the next fleet enters a fresh bootstrap gather.
    fn maybe_reset_episode(&mut self) {
        if self.lanes.values().any(|lane| lane.awaited) {
            return;
        }
        self.ever_sealed = false;
        self.cold_hold_deadline = None;
        self.strict_watchdog_deadline = None;
    }

    /// A queued fire was dropped without dispatch (cancelled / stale /
    /// rejected). Resolves it out of whichever sealed wave holds it.
    pub fn on_fire_dropped(&mut self, fire_id: u64) {
        for frame in &mut self.sealed {
            if frame.fire_waves.contains_key(&fire_id) {
                frame.open_remaining.remove(&fire_id);
                frame.makeup.remove(&fire_id);
                break;
            }
        }
        self.pop_complete();
    }

    /// Fires were posted to the driver lane in a launch batch.
    pub fn on_fires_posted(&mut self, fire_ids: &[u64]) {
        for fire_id in fire_ids {
            for frame in &mut self.sealed {
                if let Some(&wave) = frame.fire_waves.get(fire_id) {
                    let in_open = frame.open_remaining.remove(fire_id);
                    let in_makeup = frame.makeup.remove(fire_id);
                    if in_open || in_makeup {
                        frame.outstanding += 1;
                        frame.wave_outstanding[wave] += 1;
                    }
                    break;
                }
            }
        }
    }

    /// A posted fire's launch retired. `will_retry` iff the worker requeued
    /// it (RETRY outcome within budget) — it becomes a makeup fire.
    pub fn on_fire_retired(&mut self, fire_id: u64, will_retry: bool) {
        for frame in &mut self.sealed {
            if let Some(&wave) = frame.fire_waves.get(&fire_id) {
                frame.outstanding = frame.outstanding.saturating_sub(1);
                let count = &mut frame.wave_outstanding[wave];
                *count = count.saturating_sub(1);
                if will_retry {
                    frame.makeup.insert(fire_id);
                }
                break;
            }
        }
        self.pop_complete();
    }

    fn pop_complete(&mut self) {
        while self
            .sealed
            .front()
            .is_some_and(SealedFrame::is_complete)
        {
            self.sealed.pop_front();
        }
    }

    fn have_seal_candidate(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete))
    }

    /// Seal EVERY ready lane's front frame — the whole boundary at once,
    /// first-fit in lane-id order against the per-wave row/token budgets,
    /// partitioned into as many coexisting frames as the budgets require
    /// (partitions post in seal order and pipeline on-stream). Exactly one
    /// frame per lane per boundary keeps the fleet on one frame sequence.
    /// Called only once the wait-all gate holds (no missing awaited lane,
    /// no earmarked successor assembling); deterministic — no timing input beyond the
    /// bootstrap cold hold.
    fn seal(&mut self, now: Instant) -> Option<FramePlan> {
        if !self.have_seal_candidate() {
            self.cold_hold_deadline = None;
            return None;
        }
        let mut cold_hold_fired = false;
        if !self.ever_sealed && !self.structurally_full() {
            // Bootstrap gather: membership is still forming (the wait-set
            // has only the lanes that already submitted), so "all ready" is
            // trivially true. Hold the first seal briefly so a co-launched
            // fleet lands in one epoch. A structurally full wave fires
            // immediately even cold — it didn't run out of patience, it ran
            // out of room.
            match self.cold_hold_deadline {
                None => {
                    let hold = cold_hold();
                    self.cold_hold_deadline = Some(now + hold);
                    return Some(FramePlan::Hold(hold));
                }
                Some(deadline) if now < deadline => {
                    return Some(FramePlan::Hold(deadline - now));
                }
                Some(_) => {
                    cold_hold_fired = true;
                }
            }
        }
        self.cold_hold_deadline = None;

        // One frame per lane per boundary: a lane whose SECOND frame is
        // also already complete (back-to-back prefill chains) contributes
        // only its front — the rest waits for the next boundary's gate.
        let mut served: HashSet<ProcessId> = HashSet::new();
        let mut sealed_any = false;
        loop {
            let mut waves: Vec<Vec<u64>> = vec![Vec::new(); self.k];
            let mut fire_waves = HashMap::new();
            let mut wave_tokens = vec![0usize; self.k];
            let mut wave_rows = vec![0usize; self.k];
            let mut members: HashSet<ProcessId> = HashSet::new();
            for (lane_id, lane) in self.lanes.iter_mut() {
                if served.contains(lane_id) {
                    continue;
                }
                let Some(front) = lane.frames.front() else {
                    continue;
                };
                if !front.is_complete() {
                    continue;
                }
                let live: Vec<&ArrivedFire> = front
                    .fires
                    .iter()
                    .filter(|fire| fire.fire_id.is_some())
                    .collect();
                if live.is_empty() {
                    lane.frames.pop_front();
                    continue;
                }
                let fits = live.iter().all(|fire| {
                    let wave = (fire.slot as usize).min(self.k - 1);
                    wave_rows[wave] + fire.rows.max(1) <= self.max_wave_rows
                        && wave_tokens[wave] + fire.tokens <= self.max_wave_tokens
                });
                if !fits {
                    // Over budget: the lane seals into this boundary's next
                    // partition (the loop's next pass).
                    continue;
                }
                for fire in live {
                    let wave = (fire.slot as usize).min(self.k - 1);
                    wave_rows[wave] += fire.rows.max(1);
                    wave_tokens[wave] += fire.tokens;
                    let fire_id = fire.fire_id.expect("live fire has an id");
                    waves[wave].push(fire_id);
                    fire_waves.insert(fire_id, wave);
                }
                members.insert(*lane_id);
                lane.frames.pop_front();
            }
            if fire_waves.is_empty() {
                break;
            }
            sealed_any = true;
            self.ever_sealed = true;
            served.extend(members.iter().copied());
            self.record_sealed_waves(
                waves.iter().filter(|wave| !wave.is_empty()).count(),
                cold_hold_fired,
            );
            cold_hold_fired = false;
            let wave_count = waves.len();
            self.sealed.push_back(SealedFrame {
                waves,
                fire_waves,
                next_wave: 0,
                open_remaining: HashSet::new(),
                makeup: HashSet::new(),
                outstanding: 0,
                wave_outstanding: vec![0; wave_count],
            });
        }
        self.lanes
            .retain(|_, lane| lane.awaited || !lane.frames.is_empty());
        if sealed_any {
            self.last_plan_defers = false;
        }
        sealed_any.then(|| FramePlan::Dispatch(Vec::new()))
    }

    /// Structural capacity: a wave of the ready front frames already
    /// saturates a per-wave budget, so gathering longer cannot widen it —
    /// the bootstrap cold hold is bypassed (the wave didn't run out of
    /// patience; it ran out of room).
    fn structurally_full(&self) -> bool {
        let mut wave_rows = vec![0usize; self.k];
        let mut wave_tokens = vec![0usize; self.k];
        for lane in self.lanes.values() {
            let Some(front) = lane.frames.front() else {
                continue;
            };
            if !front.is_complete() {
                continue;
            }
            for fire in front.fires.iter().filter(|fire| fire.fire_id.is_some()) {
                let wave = (fire.slot as usize).min(self.k - 1);
                wave_rows[wave] += fire.rows.max(1);
                wave_tokens[wave] += fire.tokens;
                if wave_rows[wave] >= self.max_wave_rows
                    || wave_tokens[wave] >= self.max_wave_tokens
                {
                    return true;
                }
            }
        }
        false
    }

    /// An unstamped rider batch posted outside the sealed waves: it is
    /// still one wave fire for the density counters (the per-wave quorum
    /// counted untracked-only batches the same way).
    pub fn record_rider_wave(&self) {
        self.record_sealed_waves(1, false);
    }

    /// Wave-density probe counters (the former quorum `record_wave`/
    /// `record_clause`): `avg_active = wave_active_sum / wave_fires`
    /// discriminates a persistent wait-set from one that empties between
    /// fires. A seal never fires with a missing awaited lane (the wait-all
    /// gate held), so `wave_missing_sum` stays 0 by construction.
    fn record_sealed_waves(&self, wave_count: usize, cold_hold_fired: bool) {
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            let awaited = self.lanes.values().filter(|lane| lane.awaited).count() as u64;
            let waves = wave_count as u64;
            stats.fire.quorum.wave_fires.fetch_add(waves, Relaxed);
            stats
                .fire
                .quorum
                .wave_active_sum
                .fetch_add(awaited * waves, Relaxed);
            if cold_hold_fired {
                stats.fire.quorum.cold_hold_fires.fetch_add(1, Relaxed);
            }
        }
    }

    /// Advance one sealed frame's open wave past empty/fully-resolved waves.
    /// A fire that left the queue before its wave opened (cancelled /
    /// rejected) resolves here: only ids in `still_queued` enter the open set.
    fn advance_open_wave(&mut self, index: usize, still_queued: Option<&HashSet<u64>>) {
        let Some(frame) = self.sealed.get_mut(index) else {
            return;
        };
        while frame.open_remaining.is_empty() && frame.next_wave < frame.waves.len() {
            let wave = frame.next_wave;
            frame.open_remaining = frame.waves[wave]
                .iter()
                .filter(|fire_id| still_queued.is_none_or(|queued| queued.contains(fire_id)))
                .copied()
                .collect();
            frame.next_wave += 1;
            if !frame.open_remaining.is_empty() {
                break;
            }
        }
    }

    /// The fire ids the worker should try to dispatch now, in order: makeup
    /// fires of the oldest frame first, then the open wave.
    /// `still_queued` tells the policy which ids remain in the worker queue;
    /// open-wave ids that vanished (rejected/cancelled) resolve here.
    pub fn plan_dispatch(&mut self, still_queued: &HashSet<u64>, now: Instant) -> FramePlan {
        // Resolve open-wave ids that left the queue without posting.
        for frame in &mut self.sealed {
            frame
                .open_remaining
                .retain(|fire_id| still_queued.contains(fire_id));
            frame
                .makeup
                .retain(|fire_id| still_queued.contains(fire_id));
        }
        self.pop_complete();

        // Frames OVERLAP on-stream — there is no launch-time barrier. All
        // dispatch is globally ordered: frames in seal order, waves in slot
        // order within a frame, every wave posted at the worker's run-ahead
        // depth without waiting for retirement. The driver's device-side
        // readiness gate (`pass_commit` channel tickets) orders dependent
        // fires by stream order, and a frame-boundary dependency (f's last
        // wave → f+1's wave 0 of the same lane) is structurally identical
        // to an intra-frame one. A fire whose predecessor did not commit
        // comes back RETRY into the globally wave-ordered makeup set;
        // settlement of a fully posted frame proceeds asynchronously and
        // never gates the next frame's launch.
        loop {
            let mut advanced = false;
            for index in 0..self.sealed.len() {
                let frame = &self.sealed[index];
                if !frame.makeup.is_empty() {
                    // Replay the oldest makeup wave first, and only once
                    // every earlier wave — across ALL earlier frames — has
                    // drained: the rings must be at the state the makeup's
                    // channel tickets expect. Nothing later posts while a
                    // known-uncommitted step replays.
                    if self
                        .sealed
                        .iter()
                        .take(index)
                        .any(|earlier| earlier.outstanding > 0)
                    {
                        return FramePlan::Park;
                    }
                    let min_wave = frame
                        .makeup
                        .iter()
                        .filter_map(|fire_id| frame.fire_waves.get(fire_id))
                        .copied()
                        .min()
                        .unwrap_or(0);
                    let earlier_draining = frame.wave_outstanding[..min_wave]
                        .iter()
                        .any(|count| *count > 0);
                    if earlier_draining {
                        return FramePlan::Park;
                    }
                    let mut makeups: Vec<u64> = frame
                        .makeup
                        .iter()
                        .filter(|fire_id| frame.fire_waves.get(*fire_id) == Some(&min_wave))
                        .copied()
                        .collect();
                    makeups.sort_unstable();
                    self.last_plan_defers = false;
                    return FramePlan::Dispatch(makeups);
                }
                if !frame.open_remaining.is_empty() {
                    let wave = frame.next_wave - 1;
                    let ordered: Vec<u64> = frame.waves[wave]
                        .iter()
                        .filter(|fire_id| frame.open_remaining.contains(fire_id))
                        .copied()
                        .collect();
                    // The wave defers settlement only while a later wave of
                    // this frame still has queued fires to become the tail.
                    let defers = frame.waves[wave + 1..]
                        .iter()
                        .flatten()
                        .any(|fire_id| still_queued.contains(fire_id));
                    self.last_plan_defers = defers;
                    return FramePlan::Dispatch(ordered);
                }
                if frame.next_wave < frame.waves.len() {
                    self.advance_open_wave(index, Some(still_queued));
                    self.pop_complete();
                    advanced = true;
                    break;
                }
                // Fully posted: it retires asynchronously while the next
                // frame posts behind it.
            }
            if advanced {
                continue;
            }
            // Boundary: the wait-all frame quorum. Seal only once every
            // awaited lane's next frame is fully submitted (an idle lane
            // between frames is a missing member) and no join is in
            // flight (an admitted-but-unfired successor, or a freed slot
            // with a staged taker — either way the incoming lane's first
            // frame lands in this boundary instead of a narrow epoch).
            // The wait is INFINITE by principle — the watchdog below
            // reports a stalled gather but never evicts a member and
            // never fires a narrow epoch; membership changes only through
            // close/leave/first-fire events.
            if !self.lanes.values().any(|lane| !lane.frames.is_empty()) {
                // Nothing queued anywhere: no gather episode is running.
                self.strict_watchdog_deadline = None;
                return FramePlan::Park;
            }
            let missing = self
                .lanes
                .values()
                .filter(|lane| {
                    lane.awaited
                        && !lane.frames.front().is_some_and(PendingFrame::is_complete)
                })
                .count();
            let joining = !self.joins_in_flight.is_empty()
                || ((self.pending_slots > 0 || !self.departing.is_empty())
                    && !self.staged.is_empty());
            if joining || missing > 0 {
                if !self.sealed.is_empty() {
                    // An epoch is executing: its retirements re-decide and
                    // the gather continues in the background.
                    return FramePlan::Park;
                }
                let deadline = self
                    .strict_watchdog_deadline
                    .get_or_insert(now + Duration::from_micros(STRICT_WATCHDOG_US));
                if now >= *deadline {
                    *deadline = now + Duration::from_micros(STRICT_WATCHDOG_US);
                    crate::scheduler::fire_timing_write(&serde_json::json!({
                        "schema": 1,
                        "source": "scheduler",
                        "event": "frame_wait_watchdog",
                        "at_us": crate::scheduler::fire_timing_now_us(),
                        "missing_count": missing,
                        "pending_binds": self.pending_binds.values().sum::<usize>(),
                        "pending_slots": self.pending_slots,
                        "departing": self.departing.len(),
                        "joins_in_flight": self.joins_in_flight.len(),
                        "staged": self.staged.len(),
                        "slotted": self.slotted.len(),
                        "awaited_lanes":
                            self.lanes.values().filter(|lane| lane.awaited).count(),
                    }));
                }
                return FramePlan::Hold(deadline.saturating_duration_since(now));
            }
            self.strict_watchdog_deadline = None;
            // EARLY seal: the gate held (every awaited lane's next frame is
            // fully submitted, no earmarked successor assembling), so seal NOW — normally
            // while the previous frame still executes. Sealing early is
            // what re-merges stragglers into one dense fleet epoch without
            // any drain barrier: a seal never excludes a busy lane, because
            // it waits for every lane's submission instead.
            match self.seal(now) {
                Some(FramePlan::Dispatch(_)) => continue,
                Some(plan) => return plan,
                // Ready lanes exist but none can seal (all busy in an
                // executing round partition): retirements re-decide.
                None => return FramePlan::Park,
            }
        }
    }

    /// Probe/diagnostic summary line.
    pub fn debug_summary(&self) -> String {
        use std::fmt::Write as _;
        let mut out = format!(
            "frame k={} lanes={} awaited={} sealed={} pending_binds={} ever_sealed={} watchdog={:?}",
            self.k,
            self.lanes.len(),
            self.lanes.values().filter(|lane| lane.awaited).count(),
            self.sealed.len(),
            self.pending_binds.values().sum::<usize>(),
            self.ever_sealed,
            self.strict_watchdog_deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now())),
        );
        for (pid, lane) in &self.lanes {
            let front_complete = lane
                .frames
                .front()
                .is_some_and(PendingFrame::is_complete);
            let _ = write!(
                out,
                "\n  lane {pid}: awaited={} queued_frames={} front_complete={front_complete}",
                lane.awaited,
                lane.frames.len(),
            );
        }
        for (index, frame) in self.sealed.iter().enumerate() {
            let _ = write!(
                out,
                "\n  sealed[{index}]: waves={} next_wave={} open={} makeup={} outstanding={}",
                frame.waves.len(),
                frame.next_wave,
                frame.open_remaining.len(),
                frame.makeup.len(),
                frame.outstanding,
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid() -> ProcessId {
        ProcessId::new_v4()
    }

    fn stamp(lane: ProcessId, seq: u64, slot: u32, fires: u32) -> FrameStamp {
        FrameStamp {
            lane,
            seq,
            slot,
            fires,
        }
    }

    fn drive_past_cold_hold(policy: &mut FramePolicy, queued: &HashSet<u64>) -> FramePlan {
        let now = Instant::now();
        match policy.plan_dispatch(queued, now) {
            FramePlan::Hold(hold) => policy.plan_dispatch(queued, now + hold + Duration::from_micros(1)),
            plan => plan,
        }
    }

    #[test]
    fn max_in_flight_configuration_is_truthful_and_safely_capped() {
        assert_eq!(parse_max_in_flight(None), DEFAULT_MAX_IN_FLIGHT);
        assert_eq!(parse_max_in_flight(Some("0")), 1);
        assert_eq!(parse_max_in_flight(Some("4")), MAX_IN_FLIGHT);
        assert_eq!(parse_max_in_flight(Some("invalid")), DEFAULT_MAX_IN_FLIGHT);
        assert!(configured_max_in_flight() >= 1);
    }

    /// A structurally full wave bypasses the bootstrap cold hold — the
    /// reason every pre-existing worker.rs unit test (which all run at row
    /// budget 1) observes no gather delay on the unified k = 1 path.
    #[test]
    fn structural_cap_seals_immediately_even_cold() {
        let mut policy = FramePolicy::new(1, 1, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 7, 1, 1);
        let queued: HashSet<u64> = [7].into_iter().collect();
        assert_eq!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Dispatch(vec![7]),
            "a full wave must seal with no cold-hold delay"
        );
    }

    /// The bootstrap gather at k = 1: two lanes' first single-slot frames
    /// hold through the cold window, then seal as ONE dense wave (the former
    /// per-wave quorum's `cold_hold_gathers_two_pipelines_then_fires_dense`).
    #[test]
    fn bootstrap_cold_hold_gathers_single_slot_lanes_then_seals_dense() {
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        let (a, b) = (pid(), pid());
        // Single-slot stamps as the worker synthesizes them at k = 1:
        // seq = the fire id, slot 0, one fire per frame.
        policy.on_fire_enqueued(stamp(a, 10, 0, 1), Some(a), 10, 1, 1);
        policy.on_fire_enqueued(stamp(b, 11, 0, 1), Some(b), 11, 1, 1);
        let queued: HashSet<u64> = [10, 11].into_iter().collect();
        let t0 = Instant::now();
        let FramePlan::Hold(hold) = policy.plan_dispatch(&queued, t0) else {
            panic!("bootstrap membership is forming: the cold hold must arm");
        };
        let FramePlan::Dispatch(wave) =
            policy.plan_dispatch(&queued, t0 + hold + Duration::from_micros(1))
        else {
            panic!("past the window the epoch must seal");
        };
        assert_eq!(wave.len(), 2, "dense: both lanes' fires in one wave");
        policy.on_fires_posted(&wave);
        policy.on_fire_retired(10, false);
        policy.on_fire_retired(11, false);

        // Steady state: `a` resubmits, `b` does not — the wave holds.
        policy.on_fire_enqueued(stamp(a, 12, 0, 1), Some(a), 12, 1, 1);
        let queued: HashSet<u64> = [12].into_iter().collect();
        match policy.plan_dispatch(&queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("wait-all must hold for the idle lane, got {plan:?}"),
        }
        // `b`'s next fire arrives: the wave seals dense, no cold hold.
        policy.on_fire_enqueued(stamp(b, 13, 0, 1), Some(b), 13, 1, 1);
        let queued: HashSet<u64> = [12, 13].into_iter().collect();
        let FramePlan::Dispatch(next) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("all lanes ready: the steady-state wave must seal");
        };
        assert_eq!(next.len(), 2);
    }

    #[test]
    fn seals_complete_lanes_and_orders_waves_by_slot() {
        let mut policy = FramePolicy::new(4, 64, 4096, None);
        let (a, b) = (pid(), pid());
        // Lane a: full decode frame (4 fires). Lane b: chunk in slot 0 only.
        for slot in 0..4 {
            policy.on_fire_enqueued(stamp(a, 0, slot, 4), Some(a), 100 + slot as u64, 1, 1);
        }
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 200, 37, 1);

        let queued: HashSet<u64> = [100, 101, 102, 103, 200].into_iter().collect();
        let plan = drive_past_cold_hold(&mut policy, &queued);
        // Wave 0 = both lanes' slot-0 fires, lane-id order preserved.
        let FramePlan::Dispatch(wave0) = plan else {
            panic!("expected wave-0 dispatch, got {plan:?}");
        };
        assert_eq!(wave0.len(), 2);
        assert!(wave0.contains(&100) && wave0.contains(&200));

        policy.on_fires_posted(&wave0);
        // Overlap: wave 1 posts immediately behind the in-flight wave 0 —
        // there is no retirement barrier.
        let FramePlan::Dispatch(wave1) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("expected wave-1 dispatch");
        };
        assert_eq!(wave1, vec![101]);
    }

    /// THE wait-all regression test: an incomplete lane BLOCKS the seal.
    /// The watchdog reports (Hold at its cadence) but never evicts, and no
    /// narrow epoch fires. When the straggler completes, the epoch seals
    /// DENSE with every lane in.
    #[test]
    fn incomplete_lane_holds_the_seal_until_it_completes() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (fast, slow) = (pid(), pid());
        policy.on_fire_enqueued(stamp(fast, 0, 0, 2), Some(fast), 1, 1, 1);
        policy.on_fire_enqueued(stamp(fast, 0, 1, 2), Some(fast), 2, 1, 1);
        // `slow` declared 2 fires but only one arrived: a missing member.
        policy.on_fire_enqueued(stamp(slow, 0, 0, 2), Some(slow), 3, 1, 1);

        let queued: HashSet<u64> = [1, 2, 3].into_iter().collect();
        let t0 = Instant::now();
        match policy.plan_dispatch(&queued, t0) {
            FramePlan::Hold(hold) => {
                assert_eq!(hold, Duration::from_micros(STRICT_WATCHDOG_US));
            }
            plan => panic!("wait-all must hold for the incomplete lane, got {plan:?}"),
        }
        // Long past the watchdog: it re-arms and reports; it never fires.
        match policy.plan_dispatch(&queued, t0 + Duration::from_secs(60)) {
            FramePlan::Hold(_) => {}
            plan => panic!("the watchdog reports, it must not fire: got {plan:?}"),
        }

        // The straggler completes: one dense epoch, both lanes' slot-0.
        policy.on_fire_enqueued(stamp(slow, 0, 1, 2), Some(slow), 4, 1, 1);
        let queued: HashSet<u64> = [1, 2, 3, 4].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("all lanes ready: the epoch must seal");
        };
        assert_eq!(wave0.len(), 2, "dense wave 0 holds BOTH lanes");
        assert!(wave0.contains(&1) && wave0.contains(&3));
    }

    #[test]
    fn retry_becomes_makeup_and_gates_wave_advance() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 10, 1, 1);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 2), Some(lane), 11, 1, 1);
        let queued: HashSet<u64> = [10, 11].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("expected wave 0");
        };
        policy.on_fires_posted(&wave0);
        // Wave 0's fire comes back RETRY and is requeued.
        policy.on_fire_retired(10, true);
        let plan = policy.plan_dispatch(&queued, Instant::now());
        assert_eq!(
            plan,
            FramePlan::Dispatch(vec![10]),
            "makeup fire re-dispatches before wave 1 opens"
        );
        policy.on_fires_posted(&[10]);
        policy.on_fire_retired(10, false);
        let FramePlan::Dispatch(wave1) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("expected wave 1 after makeup commits");
        };
        assert_eq!(wave1, vec![11]);
    }

    /// Overlap: the next frame seals the moment the wait-all gate holds —
    /// during the current frame's execution — and its waves post behind
    /// the in-flight tail with zero retirements. A late joiner rides the
    /// same early seal, and a cross-frame RETRY replays globally
    /// wave-ordered.
    #[test]
    fn frames_overlap_across_the_boundary() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 50, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 51, 1, 1);
        let queued: HashSet<u64> = [50, 51].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("expected lane a's wave 0");
        };
        assert_eq!(wave0, vec![50]);
        policy.on_fires_posted(&wave0);
        // Mid-execution, a straggler submits its first frame and lane a its
        // next: the wait-all gate holds, so f+1 seals NOW.
        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 60, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 61, 1, 1);
        policy.on_fire_enqueued(stamp(a, 1, 0, 2), Some(a), 52, 1, 1);
        policy.on_fire_enqueued(stamp(a, 1, 1, 2), Some(a), 53, 1, 1);
        let queued: HashSet<u64> = [50, 51, 52, 53, 60, 61].into_iter().collect();
        // Global order: f's remaining wave posts first...
        let FramePlan::Dispatch(wave1) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("expected lane a's wave 1");
        };
        assert_eq!(wave1, vec![51]);
        policy.on_fires_posted(&wave1);
        // ...then f+1's dense wave 0 — both lanes, zero retirements so far.
        let FramePlan::Dispatch(merged) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("the overlapped next frame must seal and post");
        };
        assert_eq!(merged.len(), 2, "wave 0 must hold BOTH lanes: {merged:?}");
        assert!(merged.contains(&52) && merged.contains(&60));
        policy.on_fires_posted(&merged);
        // Cross-frame RETRY replay: f's wave-1 fire bounces; its makeup
        // holds until wave 0 drains, and nothing later posts meanwhile.
        policy.on_fire_retired(51, true);
        assert_eq!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Park
        );
        policy.on_fire_retired(50, false);
        assert_eq!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Dispatch(vec![51])
        );
    }

    #[test]
    fn frame_waves_post_back_to_back_without_retirement() {
        let mut policy = FramePolicy::new(3, 64, 4096, None);
        let lane = pid();
        for slot in 0..3 {
            policy.on_fire_enqueued(stamp(lane, 0, slot, 3), Some(lane), 30 + slot as u64, 1, 1);
        }
        let queued: HashSet<u64> = [30, 31, 32].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("expected wave 0");
        };
        assert_eq!(wave0, vec![30]);
        policy.on_fires_posted(&wave0);
        let FramePlan::Dispatch(wave1) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("wave 1 must post behind the in-flight wave 0");
        };
        assert_eq!(wave1, vec![31]);
        policy.on_fires_posted(&wave1);
        let FramePlan::Dispatch(wave2) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("wave 2 must post behind waves 0-1");
        };
        assert_eq!(wave2, vec![32]);
        policy.on_fires_posted(&wave2);
        assert_eq!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Park,
            "fully posted: retirement is asynchronous"
        );
    }

    #[test]
    fn retry_replays_wave_ordered_makeups() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 40, 1, 1);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 2), Some(lane), 41, 1, 1);
        let queued: HashSet<u64> = [40, 41].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("expected wave 0");
        };
        policy.on_fires_posted(&wave0);
        let FramePlan::Dispatch(wave1) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("expected chained wave 1");
        };
        policy.on_fires_posted(&wave1);
        // The predecessor fails to commit and requeues; its chained
        // successor is still in flight and will bounce on the stale rings.
        policy.on_fire_retired(40, true);
        assert_eq!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Dispatch(vec![40]),
            "the oldest wave's makeup replays first"
        );
        policy.on_fires_posted(&[40]);
        policy.on_fire_retired(41, true);
        assert_eq!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Park,
            "a later wave's makeup holds while the earlier wave is in flight"
        );
        policy.on_fire_retired(40, false);
        assert_eq!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Dispatch(vec![41]),
            "the successor's makeup replays once the predecessor drained"
        );
        policy.on_fires_posted(&[41]);
        policy.on_fire_retired(41, false);
        assert_eq!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Park
        );
    }

    #[test]
    fn dropped_fires_resolve_and_leave_rearms_the_gather() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 20, 1, 1);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 2), Some(lane), 21, 1, 1);
        let queued: HashSet<u64> = [20, 21].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("expected wave 0");
        };
        policy.on_fires_posted(&wave0);
        policy.on_fire_retired(20, false);
        // Fire 21 is cancelled while queued: the wave resolves without it.
        let queued: HashSet<u64> = HashSet::new();
        assert_eq!(policy.plan_dispatch(&queued, Instant::now()), FramePlan::Park);
        assert_eq!(policy.sealed.len(), 0, "frame completed and popped");
        assert!(
            policy.ever_sealed,
            "the wait-set persists while the lane is awaited — drained \
             books alone do not re-arm the gather"
        );
        // Only the lane's LEAVE empties the wait-set and re-arms bootstrap.
        policy.on_lane_leave(lane, false);
        assert!(!policy.ever_sealed, "an emptied wait-set re-arms the gather");
    }

    #[test]
    fn truncated_frame_seals_with_submitted_fires_only() {
        let mut policy = FramePolicy::new(4, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 4), Some(lane), 30, 1, 1);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 4), Some(lane), 31, 1, 1);
        // Host submit failed at slot 2: only 2 fires exist.
        policy.on_frame_truncated(lane, 0, 2);
        let queued: HashSet<u64> = [30, 31].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("truncated frame must still seal");
        };
        assert_eq!(wave0, vec![30]);
    }

    /// Capacity partitioning is the quorum's round rule: the deferred lane
    /// seals in the SAME round (lane-disjoint, pipelining behind the
    /// executing partition) without re-awaiting the lanes already served.
    #[test]
    fn over_budget_lane_is_served_in_the_same_round() {
        // Wave budget of 40 tokens: lane a's 37-token chunk fits, lane b's
        // additional 37 does not.
        let mut policy = FramePolicy::new(2, 64, 40, None);
        let (a, b) = {
            let (x, y) = (pid(), pid());
            if x < y { (x, y) } else { (y, x) }
        };
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 40, 37, 1);
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 41, 37, 1);
        let queued: HashSet<u64> = [40, 41].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("expected a seal");
        };
        assert_eq!(wave0, vec![40], "first-fit admits lane a only");
        policy.on_fires_posted(&[40]);
        // Lane a is served (round rule) — b seals NOW, while a's partition
        // is still in flight, because their lane sets are disjoint.
        let FramePlan::Dispatch(partition) = policy.plan_dispatch(&queued, Instant::now())
        else {
            panic!("the deferred lane must seal within the round");
        };
        assert_eq!(partition, vec![41]);
        policy.on_fires_posted(&[41]);
        policy.on_fire_retired(40, false);
        policy.on_fire_retired(41, false);
        // Round closed at b's seal: the next epoch awaits BOTH lanes again.
        let queued: HashSet<u64> = HashSet::new();
        assert_eq!(policy.plan_dispatch(&queued, Instant::now()), FramePlan::Park);
    }

    /// After an epoch, every awaited lane must resubmit before the next
    /// epoch seals — a newly arrived lane cannot start a narrow epoch while
    /// an executed lane is still thinking.
    #[test]
    fn next_epoch_waits_for_every_awaited_lane() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 60, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 61, 1, 1);
        let queued: HashSet<u64> = [60, 61].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("expected lane a's wave 0");
        };
        assert_eq!(wave0, vec![60]);
        policy.on_fires_posted(&[60]);
        policy.on_fire_retired(60, false);
        let FramePlan::Dispatch(wave1) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("expected lane a's wave 1");
        };
        assert_eq!(wave1, vec![61]);
        policy.on_fires_posted(&[61]);
        policy.on_fire_retired(61, false);

        // b arrives with a complete frame; a has NOT resubmitted. Wait-all:
        // no seal until a's next frame is in (or a leaves).
        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 70, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 71, 1, 1);
        let queued: HashSet<u64> = [70, 71].into_iter().collect();
        match policy.plan_dispatch(&queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("the epoch must wait for lane a to resubmit, got {plan:?}"),
        }

        // a resubmits: the epoch seals DENSE with both lanes.
        policy.on_fire_enqueued(stamp(a, 1, 0, 2), Some(a), 80, 1, 1);
        policy.on_fire_enqueued(stamp(a, 1, 1, 2), Some(a), 81, 1, 1);
        let queued: HashSet<u64> = [70, 71, 80, 81].into_iter().collect();
        let FramePlan::Dispatch(dense) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("all lanes ready: the epoch must seal");
        };
        assert_eq!(dense.len(), 2, "dense wave 0 holds both lanes");
        assert!(dense.contains(&70) && dense.contains(&80));
    }

    /// Graceful close is the ONLY way a straggler stops being awaited: the
    /// lane leaves the wait-set immediately and the fleet seals without it.
    #[test]
    fn graceful_close_releases_the_wait() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 90, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 91, 1, 1);
        let queued: HashSet<u64> = [90, 91].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("expected the bootstrap epoch");
        };
        assert_eq!(wave0.len(), 2);
        policy.on_fires_posted(&wave0);
        policy.on_fire_retired(90, false);
        policy.on_fire_retired(91, false);

        // b resubmits; a does not — the gather blocks on a.
        policy.on_fire_enqueued(stamp(b, 1, 0, 1), Some(b), 92, 1, 1);
        let queued: HashSet<u64> = [92].into_iter().collect();
        match policy.plan_dispatch(&queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("the gather must block on lane a, got {plan:?}"),
        }
        // a closes gracefully: released from the wait-set, b seals.
        policy.on_lane_leave(a, false);
        let FramePlan::Dispatch(next) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("the close must release the wait");
        };
        assert_eq!(next, vec![92]);
    }

    /// A bind alone holds nothing: an unadmitted bring-up process cannot
    /// fire, so its bind must not gate an executing fleet's seal (staging
    /// the next cohort's binds behind the current generation is the point).
    #[test]
    fn unearmarked_staged_bind_does_not_hold_the_seal() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        let binder = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 95, 1, 1);
        policy.on_bind_enqueued(Some(binder));
        let queued: HashSet<u64> = [95].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("a staged (unearmarked) bind must not hold the seal");
        };
        assert_eq!(wave0, vec![95]);
    }

    /// A freed slot with a staged taker holds the seal; the successor's
    /// admission converts the hold to an identity-paired join-in-flight,
    /// and its first fire releases it — the incoming lane lands in the
    /// same epoch as the fleet.
    #[test]
    fn freed_slot_with_staged_taker_gathers_the_join() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        let successor = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 95, 1, 1);
        policy.on_bind_enqueued(Some(successor));
        policy.on_bind_completed(Some(successor));
        policy.on_execution_slot_released(pid());
        let queued: HashSet<u64> = [95].into_iter().collect();
        match drive_past_cold_hold(&mut policy, &queued) {
            FramePlan::Hold(_) => {}
            plan => panic!("a freed slot with a staged taker must hold, got {plan:?}"),
        }
        policy.on_execution_slot_consumed(successor);
        match drive_past_cold_hold(&mut policy, &queued) {
            FramePlan::Hold(_) => {}
            plan => panic!("an admitted-but-unfired join must hold, got {plan:?}"),
        }
        policy.on_fire_enqueued(stamp(successor, 0, 0, 1), Some(successor), 96, 1, 1);
        let queued: HashSet<u64> = [95, 96].into_iter().collect();
        let FramePlan::Dispatch(mut wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("successor arrived: the seal must proceed dense");
        };
        wave0.sort_unstable();
        assert_eq!(wave0, vec![95, 96], "both lanes gathered into one epoch");
    }

    /// Regression: between a slot holder's Terminate leave and its
    /// teardown's release broadcast the balance reads zero, and a seal
    /// check in that window closed on the partial cohort — splitting the
    /// fleet into two sub-cohorts that never re-merge (run-ahead lead is
    /// hysteretic). The departure itself must hold: a leaving holder's
    /// release is in flight, so its staged successor is gathered.
    #[test]
    fn departed_slot_holder_holds_the_seal_until_release_lands() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        let predecessor = pid();
        let successor = pid();
        // Predecessor consumed its slot from the initial pool and ran.
        policy.on_execution_slot_consumed(predecessor);
        // The survivor's next fire is queued; the successor is staged.
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 95, 1, 1);
        policy.on_bind_enqueued(Some(successor));
        policy.on_bind_completed(Some(successor));
        // Terminate leave lands pass-granular, release still in flight.
        policy.on_slotted_terminate(predecessor);
        policy.on_lane_leave(predecessor, true);
        policy.on_process_leave(predecessor);
        let queued: HashSet<u64> = [95].into_iter().collect();
        match drive_past_cold_hold(&mut policy, &queued) {
            FramePlan::Hold(_) => {}
            plan => panic!("a departed slot holder's in-flight release must hold, got {plan:?}"),
        }
        // The release lands (paired by the holder's identity): the hold
        // converts to the freed-slot form, then to the identity-paired
        // join, then the fire seals dense.
        policy.on_execution_slot_released(predecessor);
        match drive_past_cold_hold(&mut policy, &queued) {
            FramePlan::Hold(_) => {}
            plan => panic!("freed slot with staged taker must keep holding, got {plan:?}"),
        }
        policy.on_execution_slot_consumed(successor);
        policy.on_fire_enqueued(stamp(successor, 0, 0, 1), Some(successor), 96, 1, 1);
        let queued: HashSet<u64> = [95, 96].into_iter().collect();
        let FramePlan::Dispatch(mut wave) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("successor arrived: the seal must proceed dense");
        };
        wave.sort_unstable();
        assert_eq!(wave, vec![95, 96], "cohort gathered across the departure");
    }

    /// The departure hold arms only for actual slot holders, exactly once:
    /// a Terminate for a never-admitted pid (or a duplicate leave from the
    /// exit funnel's two notification paths) leaves no phantom hold, and a
    /// staged successor whose predecessor's release was already consumed
    /// elsewhere does not re-hold the seal.
    #[test]
    fn terminate_arms_only_live_slot_holders() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        let holder = pid();
        let bystander = pid();
        policy.on_execution_slot_consumed(holder);
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 95, 1, 1);
        policy.on_bind_enqueued(Some(bystander));
        // Never-admitted pid and a duplicate leave: neither may arm.
        policy.on_slotted_terminate(bystander);
        policy.on_slotted_terminate(holder);
        policy.on_slotted_terminate(holder);
        policy.on_execution_slot_released(holder);
        policy.on_execution_slot_consumed(bystander);
        policy.on_fire_enqueued(stamp(bystander, 0, 0, 1), Some(bystander), 96, 1, 1);
        let queued: HashSet<u64> = [95, 96].into_iter().collect();
        assert!(
            matches!(
                drive_past_cold_hold(&mut policy, &queued),
                FramePlan::Dispatch(_)
            ),
            "a retired departure must leave no phantom hold"
        );
    }

    /// A forfeited slot (the leaked-permit teardown error path) resolves
    /// its holder's departure WITHOUT crediting the balance: the seal
    /// stops waiting, and no phantom free slot earmarks a staged
    /// successor that can never admit.
    #[test]
    fn forfeit_resolves_departure_without_freeing_a_slot() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        let holder = pid();
        let successor = pid();
        policy.on_execution_slot_consumed(holder);
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 95, 1, 1);
        policy.on_bind_enqueued(Some(successor));
        policy.on_bind_completed(Some(successor));
        policy.on_slotted_terminate(holder);
        let queued: HashSet<u64> = [95].into_iter().collect();
        match drive_past_cold_hold(&mut policy, &queued) {
            FramePlan::Hold(_) => {}
            plan => panic!("an unresolved departure must hold, got {plan:?}"),
        }
        policy.on_execution_slot_forfeited(holder);
        assert!(
            matches!(
                drive_past_cold_hold(&mut policy, &queued),
                FramePlan::Dispatch(_)
            ),
            "a forfeited slot must neither hold nor earmark"
        );
    }

    /// Preloaded free capacity gathers the initial fleet: while a free
    /// slot has a staged taker, the first seal waits — the co-launched
    /// herd lands in one aligned epoch instead of a ragged ramp.
    #[test]
    fn preloaded_free_slots_gather_the_initial_fleet() {
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        policy.preload_free_slots(2);
        let (a, b) = (pid(), pid());
        policy.on_bind_enqueued(Some(a));
        policy.on_bind_enqueued(Some(b));
        policy.on_execution_slot_consumed(a);
        policy.on_fire_enqueued(stamp(a, 10, 0, 1), Some(a), 10, 1, 1);
        let queued: HashSet<u64> = [10].into_iter().collect();
        match drive_past_cold_hold(&mut policy, &queued) {
            FramePlan::Hold(_) => {}
            plan => panic!("free slot with staged taker must gather, got {plan:?}"),
        }
        policy.on_execution_slot_consumed(b);
        policy.on_fire_enqueued(stamp(b, 11, 0, 1), Some(b), 11, 1, 1);
        let queued: HashSet<u64> = [10, 11].into_iter().collect();
        let FramePlan::Dispatch(mut wave) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("fleet complete: seal must fire dense");
        };
        wave.sort_unstable();
        assert_eq!(wave, vec![10, 11]);
    }

    /// Regression (fleet-wide stall): released permits launder into
    /// the semaphore's free pool, so a consumer may admit UNCONTENDED — if
    /// only parked admissions notified, the balance stayed positive forever
    /// and a staged bystander held every seal of the first generation.
    /// Every admission notifies now: a release consumed by anyone drains
    /// the balance, initial-pool admissions saturate at zero, and a later
    /// bystander inherits no phantom hold.
    #[test]
    fn consumed_release_leaves_no_phantom_hold_for_bystander() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let executing = pid();
        let bystander = pid();
        // Initial-pool admission before any release: saturates, no residue.
        policy.on_bind_enqueued(Some(executing));
        policy.on_bind_completed(Some(executing));
        policy.on_execution_slot_consumed(executing);
        policy.on_fire_enqueued(stamp(executing, 0, 0, 1), Some(executing), 95, 1, 1);
        // A retirement's release is then consumed by an uncontended
        // admission elsewhere (its notify still arrives), while a later
        // process binds and stages.
        policy.on_execution_slot_released(pid());
        policy.on_execution_slot_consumed(executing);
        policy.on_bind_enqueued(Some(bystander));
        let queued: HashSet<u64> = [95].into_iter().collect();
        assert!(
            matches!(
                drive_past_cold_hold(&mut policy, &queued),
                FramePlan::Dispatch(_)
            ),
            "a drained release must not hold for a staged bystander"
        );
    }

    /// Holds never outlive the successors they wait for: a joiner that
    /// dies before firing (and a slot release with nobody staged) can
    /// never wedge the seal.
    #[test]
    fn holds_never_outlive_departed_successors() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        let successor = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 95, 1, 1);
        let queued: HashSet<u64> = [95].into_iter().collect();

        // Slot released with an empty staged pool: no earmark, no hold.
        policy.on_execution_slot_released(pid());
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("a slot release with nobody staged must not hold");
        };
        assert_eq!(wave0, vec![95]);
        policy.on_fires_posted(&wave0);
        policy.on_fire_retired(95, false);

        // Stage a successor (freed slot pending from above), admit it,
        // then let it die before firing: the leave releases the hold.
        policy.on_fire_enqueued(stamp(lane, 1, 0, 1), Some(lane), 96, 1, 1);
        policy.on_bind_enqueued(Some(successor));
        let queued: HashSet<u64> = [96].into_iter().collect();
        match policy.plan_dispatch(&queued, Instant::now()) {
            FramePlan::Hold(_) | FramePlan::Park => {}
            plan => panic!("freed slot with staged taker must hold, got {plan:?}"),
        }
        policy.on_execution_slot_consumed(successor);
        match policy.plan_dispatch(&queued, Instant::now()) {
            FramePlan::Hold(_) | FramePlan::Park => {}
            plan => panic!("join in flight must hold, got {plan:?}"),
        }
        policy.on_process_leave(successor);
        assert!(matches!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Dispatch(_)
        ));
    }

    #[test]
    fn terminate_purges_queued_frames_close_keeps_them() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (closed, terminated) = (pid(), pid());
        let owner = pid();
        policy.on_fire_enqueued(stamp(closed, 0, 0, 1), Some(owner), 50, 1, 1);
        policy.on_fire_enqueued(stamp(terminated, 0, 0, 1), Some(owner), 51, 1, 1);

        policy.on_lane_leave(closed, false);
        policy.on_lane_leave(terminated, true);
        let queued: HashSet<u64> = [50].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("closed lane's accepted frame still seals");
        };
        assert_eq!(wave0, vec![50]);
    }
}
