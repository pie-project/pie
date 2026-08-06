//! Vesuvius frame scheduling — THE scheduler policy: the
//! **wait-for-all-active-lanes** quorum rule at frame granularity. Wait
//! until every awaited lane's next FRAME is fully submitted, then seal the
//! dense epoch and dispatch its k waves in slot order.
//!
//! Every deployment routes here, including k = 1:
//! a 1-slot frame IS a wave, so k = 1 reproduces the per-wave wait-all
//! barrier (each tracked fire arrives as its own single-fire frame; a seal
//! boundary is a wave boundary). The former per-wave `WaitAllPolicy`
//! (scheduler/quorum.rs) was this policy specialized to k = 1 and is folded
//! in — its run-ahead depth lever and watchdog constants live here now.
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
//! - tracks the COHORT-BOUNDARY WINDOW: a process in bring-up (bind
//!   accepted, no fire yet) is staged; once it acquires a contended
//!   execution permit it is a join-in-flight, and a freed slot is matched
//!   by position against the FIFO-fair admission queue to name its taker.
//!   A joiner is NOT a wait-set member and never holds the seal — sealing
//!   without it excludes nobody, it only starts the next epoch one process
//!   short. The window instead defers the bind permits a retiring process
//!   returns, so bring-up does not compete with the boundary it is joining;
//! - seals ONE partition per call from the ready lanes (deterministic
//!   first-fit against the per-wave token/row budgets — pure arithmetic over
//!   declared demand, never timing). The gate guards the OPENING of a
//!   boundary, not each partition of it: a lane deferred by capacity is
//!   served in a later partition of the same boundary without re-awaiting
//!   the lanes already fired (the quorum's round rule), and — because the
//!   partitions are sealed one at a time rather than all at once — a lane
//!   whose earlier partition has RETIRED rejoins a later one with its next
//!   frame. That is what makes a prefill-heavy boundary produce mixed
//!   prefill+decode launches instead of prefill-only ones
//!   (CONTENTION_FOLLOWUP §20.49). Every partition must advance the
//!   boundary — a continuation-only partition is refused and control
//!   returns to the gate — so a boundary closes in at most `lanes`
//!   partitions and can never fire a narrow epoch;
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::stats::SchedulerStats;
use crate::scheduler::ProcessId;

/// Default dispatch depth (`[model.scheduler] frame_dispatch_depth`): one
/// batch computing plus one prefetched. The N9 depth dose-response superseded
/// the older depth-3 default: depth 2 reduced missing/deferred enough to win
/// steady throughput in all three paired production-shape runs while retaining
/// pre-enqueue.
///
/// Dispatch-time preparation is the allocation-credit gate: physical pool
/// allocation is atomic, and an exhausted request remains a retrying
/// preparation rather than overcommitting. Depth therefore costs committed
/// pages, which is half of why deeper is not simply better.
///
/// The joint bound against the driver's staging pool
/// (`frame_dispatch_depth * frame_size < kUploadStagingDepth`) is enforced by
/// `SchedulerConfig::validate()`, where both factors are visible at once.
const DEFAULT_DISPATCH_DEPTH: usize = 2;

pub(super) fn configured_dispatch_depth() -> usize {
    match DISPATCH_DEPTH.load(Ordering::Relaxed) {
        0 => DEFAULT_DISPATCH_DEPTH,
        depth => depth,
    }
}

/// Install the configured dispatch depth at bootstrap.
pub(crate) fn set_dispatch_depth(depth: usize) {
    DISPATCH_DEPTH.store(depth, Ordering::Relaxed);
}

/// `0` = never installed; see `crate::scheduler::reconfigure`.
static DISPATCH_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// `PIE_SEAL_MODE=ready`: when the device is idle and at least one lane's
/// next frame is arrival-complete, OPEN the boundary with that subset
/// instead of holding for every awaited lane. Late lanes stream into the
/// same boundary as fresh partitions when they arrive (the §20.49
/// partition machinery), so nothing is excluded — the boundary just opens
/// before they get there. The wait-all rule keeps guarding what it must:
/// dense gathering while an epoch executes (`executing` still parks).
/// Default: strict wait-all.
fn seal_mode_ready() -> bool {
    static CONFIGURED: OnceLock<bool> = OnceLock::new();
    *CONFIGURED.get_or_init(|| std::env::var("PIE_SEAL_MODE").is_ok_and(|value| value == "ready"))
}

/// Default ON; `PIE_GATE_CONTRIBUTED=0` restores the strict wait-all rule.
///
/// In the wait-all gate, do not count a lane as missing when the engine still
/// OWES it a result (an unretired dispatch, or a bind in flight) and it has
/// nothing queued behind it. Such a lane's next frame CANNOT exist yet — its
/// guest is credit-bound on a result the device has not produced — so waiting
/// for it makes chaining structurally impossible once the fleet is one frame
/// deep (analysis.md 10.24-10.25). Applied only while a frame is executing:
/// with the device idle, dense gathering is simply right. Membership, frame
/// atomicity and FCFS are untouched; the lanes not waited for land as later
/// partitions of the SAME boundary.
///
/// Measured on D/S/X (analysis.md 10.26): +5.7% / +6.8% / +3.8%, uncontended
/// exactly neutral, chain engagement 29-56% (bistable) -> 93-94%, batch width
/// unchanged; re-measured at default-on, D 15,524 against 14,392 forced off.
///
/// The rule the code evaluates is [`LaneState::gate_verdict`] and nothing else:
/// `awaited && frames.is_empty() && owes`. An earlier `=1` variant skipped
/// every contributed-and-empty lane whether or not anything was owed to it; it
/// fragmented the batch and is gone. Any value other than `0` selects the
/// owed-only rule, so a stale `PIE_GATE_CONTRIBUTED=1` in a harness reads ON.
fn gate_contributed() -> bool {
    static CONFIGURED: OnceLock<bool> = OnceLock::new();
    *CONFIGURED
        .get_or_init(|| !std::env::var("PIE_GATE_CONTRIBUTED").is_ok_and(|value| value == "0"))
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

/// Liveness watchdog for a blocked gather. Report-only, exactly as in the
/// per-wave quorum: it never removes a member and never fires a narrow
/// epoch — an unresponsive lane leaves only through close/terminate.
const STRICT_WATCHDOG_US: u64 = 1_000_000;

/// How long a blocked gather waits before looking again. This is NOT the
/// watchdog above: that one is a *reporting* cadence, and using it as the hold
/// as well meant a boundary that was one lane short went back to sleep for as
/// long as the worker's 250ms backstop allowed, with the GPU idle the whole
/// time, even though the missing lane submitted microseconds later. Separating
/// the two is worth 2x on a sixteen-request fleet (206 -> 422 tok/s).
const GATHER_POLL_US: u64 = 500;

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
    /// Cut short at what had ARRIVED, so it is sealable as-is. Set by the
    /// host's mid-submit failure and by every path that takes the lane out
    /// of the wait-set mid-frame; once set, later arrivals for this frame
    /// keep it self-completing instead of re-raising `expected`.
    truncated: bool,
    /// `forward.park()`: not work, a wait-set exit ORDERED like a submit.
    /// It carries no fires and is complete on arrival, so a gather blocked on
    /// this lane is satisfied the moment the guest parks — the guest never has
    /// to drain its outstanding fires first (which it could only do by
    /// awaiting results, i.e. while still on the submit deadline's clock).
    /// The exit itself lands when the marker reaches the head of the lane's
    /// queue, which is exactly when every frame the guest submitted before it
    /// has sealed.
    park: bool,
    fires: Vec<ArrivedFire>,
}

impl PendingFrame {
    fn is_complete(&self) -> bool {
        self.park || self.fires.len() >= self.expected as usize
    }
}

struct LaneState {
    owner: Option<ProcessId>,
    /// Wait-set membership: joined on the lane's first stamped fire, held
    /// through every later frame (an idle lane between frames is a missing
    /// member the seal waits on), released by close/terminate, by the guest
    /// itself through `forward.park()`, or by the leash below.
    awaited: bool,
    /// Left the wait-set through `forward.park()` rather than close/terminate.
    /// Distinguished from a cleared `awaited` so that rejoin can be implicit
    /// (the next accepted fire) without disturbing the suspend path, which
    /// clears membership for a reason the guest cannot revoke.
    parked: bool,
    /// Parked by the LEASH rather than by the guest: it went silent while
    /// holding the boundary, so the seal stopped waiting for it. Rejoin is
    /// identical to a voluntary park (the next accepted fire), but the
    /// silence clock keeps running, because a lane that never comes back has
    /// broken the contract `forward.park()` exists to honour.
    leashed: bool,
    /// Start of this lane's silence: the instant it was first seen blocking
    /// the boundary with nothing owed to it. Per lane and absolute on purpose
    /// — a global "is anything executing?" clock is reset by OTHER lanes'
    /// traffic, which is exactly the case a silent member has to be found in.
    /// Cleared by any accepted fire and by any debt the engine owes the lane;
    /// deliberately NOT restarted by the leash, so a lane cannot buy silence
    /// budget by being dropped from the wait-set. `None` means the lane is
    /// not being timed.
    clock_from: Option<Instant>,
    frames: VecDeque<PendingFrame>,
    /// Whether this lane still owes the boundary currently open. Cleared for
    /// every lane when the wait-all gate opens a boundary, set when the lane is
    /// picked into a partition — and ALSO set for every lane, fired or not, by
    /// [`FramePolicy::close_boundary`], which writes off a boundary no owing
    /// lane can advance. So "true" means "has fired into this boundary, or has
    /// been written off from it", not "has fired": a census that reads this as
    /// evidence of contribution will find it true of every lane at the gate
    /// (which is exactly what the §10.24 dump saw).
    ///
    /// A boundary is OPEN while some awaited lane has not fired into it. While
    /// it is open the gate is not re-evaluated: those lanes were all ready when
    /// the gate held, so no partition can be narrow. A lane that joins the
    /// wait-set mid-boundary starts fired so it waits for the next gate —
    /// which is where join gathering already happens.
    ///
    /// Deliberately NOT part of [`LaneState::gate_verdict`]: because the gate
    /// only ever runs against a CLOSED boundary, this is true for every
    /// blocking lane there, so testing it said nothing.
    fired_this_boundary: bool,
}

/// The wait-all gate's verdict for ONE lane — [`LaneState::gate_verdict`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateVerdict {
    /// Not a member, or its next frame is fully submitted: the gate is free
    /// to seal as far as this lane is concerned.
    Satisfied,
    /// A member whose next frame is not fully submitted. The boundary holds.
    Blocking,
    /// A member with nothing queued that the engine still OWES a result, so
    /// its next frame cannot exist until the device answers (see
    /// [`gate_contributed`]). Not counted missing.
    SkippedOwed,
}

impl LaneState {
    /// Does this lane deny the wait-all gate?
    ///
    /// THE gate rule, in one place: a lane blocks while it is a member
    /// (`awaited`) whose next frame is not arrival-complete. `relax` is
    /// [`gate_contributed`] AND the device being busy — with the device idle
    /// there is no chain to protect and dense gathering is simply right.
    /// `owes` is the engine's debt to this lane (an unretired dispatch or a
    /// bind in flight), the same quantity that stops its deadline clock.
    ///
    /// Total function of `(awaited, front-complete, frames.is_empty(), owes)`
    /// and the two flags — no timing input, no interior mutation — so the
    /// policy's central decision is unit-testable without driving a boundary.
    fn gate_verdict(&self, relax: bool, owes: bool) -> GateVerdict {
        if !self.awaited || self.frames.front().is_some_and(PendingFrame::is_complete) {
            return GateVerdict::Satisfied;
        }
        // Contributed-and-owed relaxation: this lane is already inside the
        // frame on the device and has nothing queued, so waiting for it is
        // waiting for the device. `frames.is_empty()` rather than
        // "front incomplete" on purpose — a half-arrived frame IS work the
        // gate must still wait for, and only an empty queue proves the guest
        // has nothing it could have submitted.
        if relax && owes && self.frames.is_empty() {
            return GateVerdict::SkippedOwed;
        }
        GateVerdict::Blocking
    }
}

/// One sealed (immutable) frame, dispatched WHOLE: per-wave fire-id lists in
/// slot order. Venus: the frame is the launch unit — it leaves this queue the
/// moment the worker posts it, and the policy tracks nothing after dispatch
/// (settlement is the worker's frame retirement).
struct SealedFrame {
    waves: Vec<Vec<u64>>,
    /// Member lanes — read by the dispatch gate to hold a frame whose lane
    /// has a pre-launch copy barrier queued.
    members: BTreeSet<ProcessId>,
}

/// What the worker should do next for frame-managed launches.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum FramePlan {
    /// Post this WHOLE sealed frame now: per-wave fire ids in slot order.
    /// The frame has left the policy — the worker owns it from here.
    Dispatch(Vec<Vec<u64>>),
    /// Nothing dispatchable yet; re-decide after the bootstrap hold or at
    /// the blocked-gather watchdog deadline.
    Hold(Duration),
    /// No sealed work and no seal candidates: park until an arrival.
    Park,
    /// These processes were silent past the silence timeout without ever
    /// parking: abandoned, not merely slow (a lane that only misses the
    /// submit deadline is leashed out of the wait-set and rejoins on its own).
    /// The worker terminates them; the policy has already dropped their
    /// lanes, so re-planning proceeds without them.
    Terminate(Vec<ProcessId>),
}

/// The stamped fire ids still in the worker queue, sorted.
///
/// This is rebuilt on EVERY scheduler pass (~46 per wave at 128 requests),
/// so its build cost is on the hot path: as a `HashSet<u64>` the rebuild
/// measured 35us per pass / 1.6ms per wave — the single largest scheduler
/// item, against a 3ms GPU wave. Sorted-push plus binary search costs no
/// hashing and no table growth, and the queue is already in ascending id
/// order (ids come from one monotonic counter), so the sort is a no-op scan.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct QueuedFireIds {
    ids: Vec<u64>,
}

impl QueuedFireIds {
    pub fn clear(&mut self) {
        self.ids.clear();
    }

    pub fn push(&mut self, fire_id: u64) {
        self.ids.push(fire_id);
    }

    /// Restore the sorted invariant `contains` relies on. Retries and
    /// re-admissions can put a lower id behind a higher one, so this cannot
    /// assume queue order.
    pub fn seal(&mut self) {
        if !self.ids.windows(2).all(|pair| pair[0] <= pair[1]) {
            self.ids.sort_unstable();
        }
    }

    pub fn contains(&self, fire_id: &u64) -> bool {
        self.ids.binary_search(fire_id).is_ok()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl FromIterator<u64> for QueuedFireIds {
    fn from_iter<T: IntoIterator<Item = u64>>(iter: T) -> Self {
        let mut out = Self {
            ids: iter.into_iter().collect(),
        };
        out.seal();
        out
    }
}

pub(super) struct FramePolicy {
    k: usize,
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
    /// a bring-up lane joins on its own first fire.
    pending_binds: BTreeMap<ProcessId, usize>,
    /// Successor pool: processes in bring-up (first bind control accepted)
    /// whose lane has not yet submitted its first stamped fire. While a
    /// free execution slot exists (`pending_slots > 0`), one of these is
    /// about to take it, which is what opens the cohort-boundary window.
    staged: BTreeSet<ProcessId>,
    /// Released execution slots not yet re-consumed by an admission:
    /// +1 on `on_execution_slot_released`, saturating -1 on EVERY
    /// `on_execution_slot_consumed` (uncontended admissions notify too —
    /// the semaphore launders released permits into the free pool, so
    /// only drain-on-every-admission keeps the balance honest; the
    /// saturation absorbs initial-pool consumptions, and a release is
    /// always mailed before its consumer can acquire, so a release-paired
    /// drain is never lost to the clamp). A positive balance with a
    /// non-empty `staged` pool means a successor's admission is imminent,
    /// which opens the cohort-boundary window. Multi-driver note: consume/release broadcasts reach
    /// every driver's policy, and it is the GLOBAL admission semaphore
    /// (one pool across drivers) that bounds outstanding consumes to
    /// capacity — that is what keeps each policy's balance from going
    /// negative under foreign-pid traffic.
    pending_slots: u64,
    /// Identity-paired in-flight joins: a parked process that ACQUIRED its
    /// execution permit but whose first stamped fire has not arrived yet.
    /// Cleared by the first fire and by every leave path, so a joiner that
    /// dies cannot pin the window open.
    joins_in_flight: BTreeSet<ProcessId>,
    /// Processes blocked on an execution permit, in the order the FIFO-fair
    /// semaphore will hand them one. A slot that is free or about to be free
    /// belongs to the head of this queue, which is what makes the successor
    /// earmark a named process.
    admission_queue: VecDeque<ProcessId>,
    /// Processes that consumed an execution slot and still hold it. A
    /// Terminate leave of a member moves it to `departing`: its slot is
    /// now certain to resolve (the permit's only exit is `ProcessCtx::drop`,
    /// which broadcasts the release immediately after posting the leave).
    slotted: BTreeSet<ProcessId>,
    /// Slot holders between their Terminate leave and their release
    /// broadcast, identity-paired. Both are posted back-to-back by
    /// `ProcessCtx::drop`, but the worker processes them pass-granular, so a
    /// seal check landing between the two sees a zero `pending_slots`
    /// balance with the departure already recorded — without this set the
    /// seal would close on the partial cohort and split the fleet, and
    /// the split persists for the rest of the run (run-ahead lead is
    /// hysteretic). Every entry resolves: the retiring process posts the
    /// disarm right after its leave, on the same producer (FIFO).
    departing: BTreeSet<ProcessId>,
    /// Processes the planner has suspended (evicted) and not yet observed
    /// running again. While a process is marked here its lanes never
    /// (re-)enter the wait-set, so an arrival that races the suspend cannot
    /// resurrect the fleet's obligation to wait for a process that is being
    /// swapped out. Self-healing: the mark clears the moment one of its
    /// lanes presents a COMPLETE frame, which only a running process can do
    /// (CONTENTION_FOLLOWUP §20.8).
    suspended: BTreeSet<ProcessId>,
    /// Per-lane: the frame seq cut short when the lane last left the
    /// wait-set mid-frame. The slots that truncation cut off may still
    /// arrive (the guest resumes inside its submit loop) and must not
    /// re-form an unsatisfiable frame under a seq whose earlier slots have
    /// already sealed. Kept OUTSIDE `lanes` because a truncated lane is
    /// normally dropped the moment its frames drain, well before the late
    /// slot lands.
    truncated_seqs: BTreeMap<ProcessId, u64>,
    /// Liveness-only deadline for the current blocked-gather episode.
    strict_watchdog_deadline: Option<Instant>,
    /// Monotonic count of accepted frame-fire arrivals — the ready-mode
    /// quiescence signal: the gate opens early only after one full hold
    /// cycle in which this did not move (the arrival burst has ended).
    arrival_seq: u64,
    /// `arrival_seq` observed at the last held gate evaluation that had a
    /// seal candidate. `Some(seq) == arrival_seq` at the next evaluation
    /// means no new arrival landed in between: quiesced.
    quiesce_mark: Option<u64>,
    /// Lanes with fires the engine has dispatched but not yet retired: the
    /// engine owes them a result, so the submit deadline's clock must not run
    /// against them. This is the debt a lockstep guest (no run-ahead, awaits
    /// each result before submitting again) is in for an entire GPU wave —
    /// without it an aggressive deadline would kill exactly the guests that
    /// are behaving, since they cannot submit until the engine answers.
    in_flight_lanes: BTreeSet<ProcessId>,
    /// When each in-flight lane's debt was incurred. `in_flight_lanes` stops
    /// the silence clock outright, which is right for a wave that will retire
    /// and wrong for one that never will: a command buffer that came back out
    /// of memory is never retired, `on_frame_retired` is never called, and the
    /// lane's clock is reset on every tick forever. The engine then owes a
    /// result it cannot produce and the scheduler waits for it without limit
    /// -- a hang with no message, which is how the Qwen residency overrun
    /// presented before it was diagnosed. The debt itself gets a deadline.
    in_flight_since: BTreeMap<ProcessId, Instant>,
    /// How long a silent member may hold the boundary before the LEASH drops
    /// it from the wait-set. Not a verdict and not a failure: the lane parks,
    /// its queued frames still dispatch, and its next fire rejoins. Purely a
    /// density bound — how long the fleet waits for a straggler. See
    /// [`crate::scheduler::configured_submit_deadline`].
    submit_deadline: Duration,
    /// How long a lane may stay silent in total — through the leash and
    /// beyond — before the process is terminated. This IS a verdict: a
    /// pipeline that intends to stop must call `forward.park()`, which ends
    /// the silence and is never killed. Generous on purpose; the leash
    /// already protects the fleet, so nothing but a genuinely abandoned
    /// process reaches this.
    silence_timeout: Duration,
    /// [`gate_contributed`], read once so the lever is fixed for the policy's
    /// life (and overridable in tests without touching the process
    /// environment). `false` is the strict wait-all rule.
    gate_contributed: bool,
    /// Probe sink (`profile-fire` wave counters); `None` in unit tests.
    stats: Option<Arc<SchedulerStats>>,
}

impl FramePolicy {
    /// Whether any lane of `owner` is currently held by the leash. Tests
    /// only: the leash is deliberately invisible to the guest, so this is the
    /// only way to assert on it.
    #[cfg(test)]
    fn leashed(&self, owner: ProcessId) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.owner == Some(owner) && lane.leashed)
    }

    /// Whether the engine still records an unretired dispatch for `lane`.
    /// Tests only: `in_flight_lanes` membership IS `owes`, which is the whole
    /// of the contributed relaxation and also what stops the silence clock, so
    /// a leaked entry is a permanent wait-all exemption with no other symptom.
    #[cfg(test)]
    fn owes_lane(&self, lane: ProcessId) -> bool {
        self.in_flight_lanes.contains(&lane)
    }

    /// Override the silence timeout. Tests only.
    #[cfg(test)]
    fn with_silence_timeout(mut self, timeout: Duration) -> Self {
        self.silence_timeout = timeout;
        self
    }

    /// Override the submit deadline. Tests only — in a live engine the value
    /// comes from config once and must not move afterwards.
    #[cfg(test)]
    fn with_submit_deadline(mut self, deadline: Duration) -> Self {
        self.submit_deadline = deadline;
        // Out of the way unless a test asks for it: these cases exercise the
        // leash, and a default 30s kill would fire on the same driven clock.
        self.silence_timeout = Duration::from_secs(86_400);
        self
    }

    /// Pin the contributed-and-owed gate relaxation for one policy. Tests
    /// only, and it pins BOTH directions on purpose: the live lever is the
    /// process-wide `PIE_GATE_CONTRIBUTED`, so a test that read the
    /// environment would assert on whatever the suite was run with, and one
    /// that SET it would decide the answer for every other test in the binary.
    #[cfg(test)]
    fn with_gate_contributed(mut self, on: bool) -> Self {
        self.gate_contributed = on;
        self
    }

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
            lanes: BTreeMap::new(),
            sealed: VecDeque::new(),
            pending_binds: BTreeMap::new(),
            staged: BTreeSet::new(),
            pending_slots: 0,
            joins_in_flight: BTreeSet::new(),
            admission_queue: VecDeque::new(),
            slotted: BTreeSet::new(),
            departing: BTreeSet::new(),
            suspended: BTreeSet::new(),
            truncated_seqs: BTreeMap::new(),
            strict_watchdog_deadline: None,
            arrival_seq: 0,
            quiesce_mark: None,
            in_flight_lanes: BTreeSet::new(),
            in_flight_since: BTreeMap::new(),
            submit_deadline: crate::scheduler::configured_submit_deadline(),
            silence_timeout: crate::scheduler::configured_silence_timeout(),
            gate_contributed: gate_contributed(),
            stats,
        }
    }

    /// Whether this deployment runs 1-slot frames (k = 1):
    /// a frame is a wave, and the worker synthesizes a per-fire stamp at
    /// admission instead of the guest submitting frames.
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
        // A fire can overtake the planner's suspend broadcast (both are
        // queued, and the eviction fence stops LEASES, not an arrival
        // already past it). At k = 1 that is harmless — a 1-fire frame is
        // complete on arrival, so it seals and drains. At k > 1 the first
        // slot alone would resurrect the wait-set membership the suspend
        // just dropped, and the remaining slots sit behind the eviction:
        // the half-arrived frame then holds the whole fleet's seal forever
        // (CONTENTION_FOLLOWUP §20.8). So while the owner is suspended its
        // lane records fires without joining the wait-set.
        let lane_owner = owner.or_else(|| self.lanes.get(&stamp.lane).and_then(|lane| lane.owner));
        let suspended = lane_owner.is_some_and(|owner| self.suspended.contains(&owner));
        // A slot arriving under a seq this lane already truncated is the
        // tail of a frame whose earlier slots have sealed; it stands alone.
        // A newer seq means the guest moved on, so the cut is spent.
        let late = match self.truncated_seqs.get(&stamp.lane).copied() {
            Some(seq) if seq == stamp.seq => true,
            Some(seq) if seq < stamp.seq => {
                self.truncated_seqs.remove(&stamp.lane);
                false
            }
            _ => false,
        };
        let lane = self.lanes.entry(stamp.lane).or_insert_with(|| LaneState {
            owner,
            awaited: !suspended,
            // A lane born under its owner's suspension is not a member, but
            // it must become one on its first post-restore fire — the same
            // implicit-rejoin latch every park-shaped exit carries.
            parked: suspended,
            leashed: false,
            clock_from: None,
            frames: VecDeque::new(),
            // `open_boundary` is the only thing that makes a lane owe: a lane
            // arriving mid-boundary is not a member of it, and one arriving
            // between boundaries is picked up by the next gate.
            fired_this_boundary: true,
        });
        if lane.owner.is_none() {
            lane.owner = owner;
        }
        // Implicit rejoin: a parked lane returns to the quorum on its next
        // accepted fire. Atomic with the fire on purpose — an explicit join
        // would reopen the window this whole contract closes (a member that
        // has joined but not yet submitted). A fire racing the suspend
        // broadcast must NOT consume the latch: the lane is leaving, and
        // spending `parked` here would strand it un-awaited after restore.
        if lane.parked && !suspended {
            lane.parked = false;
            lane.awaited = true;
        }
        // Any accepted arrival is proof of life: the deadline measures
        // consecutive silence while blocking a seal, not elapsed lifetime.
        lane.leashed = false;
        lane.clock_from = None;
        let frame = match lane.frames.iter_mut().find(|frame| frame.seq == stamp.seq) {
            Some(frame) => frame,
            None => {
                lane.frames.push_back(PendingFrame {
                    seq: stamp.seq,
                    expected: stamp.fires,
                    truncated: false,
                    park: false,
                    fires: Vec::with_capacity(stamp.fires as usize),
                });
                lane.frames.back_mut().expect("frame just pushed")
            }
        };
        frame.truncated |= late || suspended;
        frame.fires.push(fire);
        self.arrival_seq = self.arrival_seq.wrapping_add(1);
        frame.expected = if frame.truncated {
            frame.fires.len() as u32
        } else {
            frame.expected.max(stamp.fires)
        };
        // Cut below what the guest declared: the slots still to come belong
        // to a frame that has already sealed, so remember the seq.
        let cut = frame.truncated && frame.expected < stamp.fires;
        if cut {
            self.truncated_seqs.insert(stamp.lane, stamp.seq);
        }
    }

    /// The host failed a frame mid-submit: only `submitted` fires exist.
    pub fn on_frame_truncated(&mut self, lane: ProcessId, seq: u64, submitted: u32) {
        if let Some(lane) = self.lanes.get_mut(&lane)
            && let Some(frame) = lane.frames.iter_mut().find(|frame| frame.seq == seq)
        {
            frame.expected = submitted;
            frame.truncated = true;
        }
    }

    /// `forward.park()`: the guest declares it will not submit again until it
    /// fires, so the seal must stop waiting for it.
    ///
    /// This is a SUBMIT, not a control operation, and it must stay one. Routed
    /// through the control slot it would queue behind the very dispatch the
    /// gather is holding, which is the `bind -> dispatch -> seal -> bind`
    /// cycle this contract exists to remove: the request to leave would be
    /// trapped behind the bus it is trying not to hold up.
    ///
    /// Ordered like any other submit, so outstanding fires are fine — the exit
    /// lands when the marker reaches the head of the lane's queue. A lane with
    /// nothing queued parks immediately.
    pub fn on_lane_park(&mut self, lane: ProcessId, seq: u64) {
        let Some(state) = self.lanes.get_mut(&lane) else {
            // Never fired: no wait-set membership to release.
            return;
        };
        if state.frames.iter().any(|frame| frame.park) {
            // Already parked or park already queued; parking twice with no
            // fire in between is a no-op, not an error.
            return;
        }
        // Ordered by seq, not appended: the mailbox drains by item priority,
        // so a park can be dequeued after a fire the guest submitted later.
        // The guest's own submission order is what the contract is written
        // against, and `seq` is what carries it.
        let at = state
            .frames
            .iter()
            .position(|frame| frame.seq > seq)
            .unwrap_or(state.frames.len());
        state.frames.insert(
            at,
            PendingFrame {
                seq,
                expected: 0,
                truncated: true,
                park: true,
                fires: Vec::new(),
            },
        );
    }

    /// Retire any park marker that has reached the head of its lane's queue:
    /// every frame the guest submitted before parking has sealed, so the lane
    /// leaves the wait-set now. Runs before the gather so a parked lane is
    /// never counted missing.
    /// A dispatched batch settled: the engine has paid these lanes back, so
    /// their submit deadline starts running again — from NOW, with the full
    /// budget, because the clock must never charge the guest for the wave it
    /// was waiting on.
    ///
    /// Clearing rather than decrementing is deliberate. A frame can retire in
    /// more than one batch, and a count that drifted either way would be
    /// worse than useless: too high and a dead lane becomes immortal, too low
    /// and a live one is charged for engine latency. Re-arming from each
    /// retirement is monotonically safe under both.
    pub fn on_frame_retired(&mut self, lanes: impl IntoIterator<Item = ProcessId>) {
        for lane in lanes {
            self.in_flight_lanes.remove(&lane);
            self.in_flight_since.remove(&lane);
            if let Some(state) = self.lanes.get_mut(&lane) {
                state.clock_from = None;
            }
        }
    }

    fn retire_parks(&mut self) {
        for state in self.lanes.values_mut() {
            while state.frames.front().is_some_and(|frame| frame.park) {
                state.frames.pop_front();
                // Only an exit with nothing behind it removes the lane. A
                // frame queued after the park is a submit the guest made
                // after asking to leave, and a submit rejoins — the exit is
                // ordered by `seq`, so it must lose to anything later even
                // when the two are processed in the same drain.
                if state.frames.is_empty() {
                    state.awaited = false;
                    state.parked = true;
                    state.clock_from = None;
                }
            }
        }
    }

    /// A bind control entered the scheduler. A bind on its own means
    /// nothing to the boundary: a live rebinder is already wait-set-held
    /// through its lane, and a bring-up process (no lane yet) only enters
    /// the `staged` successor pool, which counts once a slot opens for it
    /// ([`FramePolicy::on_execution_slot_released`]) or it acquires one
    /// ([`FramePolicy::on_execution_slot_consumed`]).
    pub fn on_bind_enqueued(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid {
            *self.pending_binds.entry(pid).or_default() += 1;
            // Only a process with NO lane yet is a successor the seal
            // should wait for. One that has already fired is in the
            // wait-set through its lane, and staging it again makes the
            // seal wait for a fire it cannot issue: its next fire may be
            // ordered behind the settlement of the frame the seal is
            // holding — a guest that binds a second program after its
            // first fire (prefill then decode, the ordinary shape)
            // deadlocked the whole gather whenever the execution pool was
            // capped.
            if !self.lanes.values().any(|lane| lane.owner == Some(pid)) {
                self.staged.insert(pid);
            }
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
    /// a successor is staged, the cohort-boundary window is open — the
    /// successor's admission and first fire are imminent. Resolves the holder's departure by identity
    /// (its terminate leave always precedes this broadcast: both are sent
    /// by the teardown task, in that order).
    pub fn on_execution_slot_released(&mut self, pid: ProcessId) {
        self.departing.remove(&pid);
        self.pending_slots += 1;
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
        self.admission_queue.retain(|queued| *queued != pid);
        self.slotted.insert(pid);
        if self.staged.contains(&pid) {
            self.joins_in_flight.insert(pid);
        }
    }

    /// A process began waiting for an execution permit.
    pub fn on_admission_queued(&mut self, pid: ProcessId) {
        if !self.admission_queue.contains(&pid) {
            self.admission_queue.push_back(pid);
        }
    }

    /// It took the permit, or was cancelled before it could.
    pub fn on_admission_dequeued(&mut self, pid: ProcessId) {
        self.admission_queue.retain(|queued| *queued != pid);
    }

    /// A slot holder's Terminate leave arrived: its release broadcast is
    /// now in flight (the permit's only exit is `ProcessCtx::drop`, which
    /// leaves first and resolves second). The seal treats
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

    /// The cohort-boundary window: a swap is earmarked (a slot is free or
    /// about to be, and the process that will take it is identified) or an
    /// admitted successor's first fire is still in flight.
    ///
    /// This never held the seal and no longer pretends to. A joiner is not a
    /// wait-set member, so sealing without it excludes nobody. What the
    /// window is for is that the engine is visibly mid-handover, which is the
    /// moment to hold back the bind permits a retiring process returns
    /// (`set_bind_release_hold`) so the incoming cohort's bring-up does not
    /// compete with the boundary it is joining. That is now its only use: it
    /// also used to stop every lane's submit-deadline clock, which measured
    /// as worth nothing.
    ///
    /// The earmark used to be `(pending_slots > 0 || !departing.is_empty())
    /// && !staged.is_empty()`: a cross product, not a matching. It never
    /// claimed that *that* staged process would take *that* slot, because it
    /// could not -- admission is a race for a counting permit. Once the pool
    /// was oversubscribed both sides were permanently non-empty, so the
    /// predicate was permanently true and only a grace timer stopped a
    /// fully-submitted boundary from idling the GPU for the whole
    /// strict-watchdog window.
    ///
    /// `admission_queue` closes that gap. `tokio::Semaphore` is FIFO-fair,
    /// so the processes blocked on a permit ARE the takers of the next slots
    /// to free, in order. Matching the two by position earmarks named
    /// processes, and the window now lasts only from a release until its
    /// identified taker wakes -- microseconds -- instead of standing
    /// permanently true.
    pub fn earmarked(&self) -> impl Iterator<Item = ProcessId> + '_ {
        let slots = self.pending_slots as usize + self.departing.len();
        self.admission_queue
            .iter()
            .filter(|pid| self.staged.contains(pid))
            .take(slots)
            .copied()
    }

    pub fn is_joining(&self) -> bool {
        !self.joins_in_flight.is_empty() || self.earmarked().next().is_some()
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
    ///
    /// TWO KEY SPACES: `lane` is a PIPELINE SCOPE id (see `record_arrival`),
    /// while `staged` / `joins_in_flight` / `pending_binds` are PROCESS
    /// scoped. `owner` therefore has to be supplied by the caller whenever
    /// the leaver has no lane yet — a process that parks in KV acquire
    /// BEFORE its first fire has nothing in `lanes` to recover an owner
    /// from, so cleaning the process-keyed maps with `lane` silently matched
    /// nothing and left the process in `joins_in_flight` forever, wedging
    /// the seal gate against a join that could never arrive.
    pub fn on_lane_leave(&mut self, lane: ProcessId, owner: Option<ProcessId>, purge_queued: bool) {
        // Recover the owner from the lane while it still exists; an explicit
        // `owner` wins, since a laneless leaver can only be identified that way.
        let owner = owner.or_else(|| self.lanes.get(&lane).and_then(|state| state.owner));
        if purge_queued {
            self.lanes.remove(&lane);
            self.truncated_seqs.remove(&lane);
            // A terminated lane's fires may be cancelled rather than retired,
            // so `on_frame_retired` is not guaranteed to clear this. Left
            // behind, the entry makes `owes` true forever for that lane id —
            // and `owes` is the whole of the contributed relaxation, so a lane
            // that came back under the same id would be permanently skipped by
            // wait-all AND permanently exempt from the leash/silence ladder
            // (both are downstream of the same `owes`). Purged with the lane
            // that made it, like every other lane-keyed map here.
            self.in_flight_lanes.remove(&lane);
        } else if let Some(state) = self.lanes.get_mut(&lane) {
            state.awaited = false;
            // The leave is a PARK, not a verdict: both callers (the planner's
            // allocation park and a graceful pipeline close) promise "rejoin
            // is implicit on the next accepted fire". A drained lane keeps
            // that promise through removal + re-creation (recreated awaited),
            // but a lane that stays for its queued frames used to keep
            // `parked` clear — so `record_arrival`'s rejoin branch never ran
            // and the lane FREE-RODE outside the wait-set for the rest of its
            // life (measured: ra_rejoins == 0 across whole contended runs,
            // seal-time awaited count decaying 108 -> ~55 while ~90 lanes
            // stayed resident). Latching `parked` restores the contract; a
            // closed pipeline never fires again, so for it the flag is inert.
            state.parked = true;
            self.truncate_incomplete(lane);
            let drained = self
                .lanes
                .get(&lane)
                .is_some_and(|state| state.frames.is_empty());
            if drained {
                self.lanes.remove(&lane);
            }
        }
        if let Some(owner) = owner {
            self.pending_binds.remove(&owner);
            self.forget_staged(owner);
        }
        self.maybe_reset_episode();
    }

    /// Every scope owned by `owner` left (process terminate/suspend).
    pub fn on_process_leave(&mut self, owner: ProcessId) {
        let owned: Vec<ProcessId> = self
            .lanes
            .iter()
            .filter(|(_, lane)| lane.owner == Some(owner))
            .map(|(id, _)| *id)
            .collect();
        for id in owned {
            self.truncated_seqs.remove(&id);
            // The process is gone, so no retirement is owed to these lanes and
            // none may be waited for. See the purge in `on_lane_leave`: a
            // surviving `in_flight_lanes` entry is a permanent `owes`, which is
            // a permanent wait-all exemption plus a permanently stopped
            // silence clock. Deliberately NOT done in `on_process_suspend` —
            // there the debt is real, the fires drain and retire normally, and
            // clearing it would start the silence clock against an evicted
            // process the engine genuinely still owes.
            self.in_flight_lanes.remove(&id);
        }
        self.lanes.retain(|_, lane| lane.owner != Some(owner));
        self.pending_binds.remove(&owner);
        self.suspended.remove(&owner);
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

    /// The planner is evicting `owner`: every lane it owns stops being
    /// awaited so boundaries seal without it, but already-submitted frames
    /// stay sealable — the tail drains untracked (exactly the graceful
    /// pipeline-close shape, applied process-wide), releasing the fire
    /// leases the eviction's quiescence wait needs. Purging here would
    /// instead orphan the queued fires WITH their leases and wedge the
    /// eviction. Rejoin is implicit: post-restore, the lane's next arrival
    /// recreates it awaited.
    pub fn on_process_suspend(&mut self, owner: ProcessId) {
        self.suspended.insert(owner);
        let owned: Vec<ProcessId> = self
            .lanes
            .iter()
            .filter(|(_, lane)| lane.owner == Some(owner))
            .map(|(id, _)| *id)
            .collect();
        for lane_id in owned {
            if let Some(lane) = self.lanes.get_mut(&lane_id) {
                lane.awaited = false;
                // Same rejoin contract as the park-shaped leave: a lane kept
                // for its queued frames must re-enter the wait-set on its
                // first post-restore fire, and only the `parked` latch makes
                // `record_arrival` do that. The suspend mark below keeps the
                // latch from being consumed by a fire racing the broadcast.
                lane.parked = true;
            }
            self.truncate_incomplete(lane_id);
            let drained = self
                .lanes
                .get(&lane_id)
                .is_some_and(|lane| lane.frames.is_empty());
            if drained {
                self.lanes.remove(&lane_id);
            }
        }
        self.pending_binds.remove(&owner);
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

    /// The planner concluded `owner` is runnable again — a restore
    /// committed, or an eviction attempt rolled back. Clearing the suspend
    /// mark lets its lanes rejoin the wait-set (naturally: the drained lane
    /// is dropped and its next arrival recreates it awaited) and restores
    /// full-frame batching for it. A missed resume is fail-safe — the fleet
    /// simply stops waiting for that process — so this is posted from the
    /// planner's runnable-again chokepoints rather than trusted as the only
    /// path back.
    pub fn on_process_resume(&mut self, owner: ProcessId) {
        self.suspended.remove(&owner);
    }

    /// Make every unfinished frame on `lane` sealable at what has ARRIVED.
    ///
    /// A lane that leaves the wait-set mid-frame (KV allocation park,
    /// graceful close, planner suspend) cannot finish that frame: the guest
    /// is blocked on exactly the progress the frame is holding up. Its
    /// submitted slots must still seal and drain, because their fire leases
    /// are what an eviction's quiescence wait blocks on — leave them
    /// stranded and the eviction never completes, the planner head never
    /// advances, and the whole fleet starves behind it
    /// (CONTENTION_FOLLOWUP §20.8). k = 1 can never reach this: a 1-slot
    /// frame is complete the moment it arrives.
    fn truncate_incomplete(&mut self, lane_id: ProcessId) {
        let Some(lane) = self.lanes.get_mut(&lane_id) else {
            return;
        };
        let mut cut = None;
        for frame in &mut lane.frames {
            if frame.is_complete() {
                continue;
            }
            frame.expected = frame.fires.len() as u32;
            frame.truncated = true;
            // Only the newest frame can be unfinished — a guest submits a
            // frame's slots in order — so one seq covers the cut.
            cut = Some(frame.seq);
        }
        if let Some(seq) = cut {
            self.truncated_seqs.insert(lane_id, seq);
        }
    }

    /// A staged or joining successor departed before its first fire: the
    /// seal must never wait for a lane that cannot arrive.
    fn forget_staged(&mut self, pid: ProcessId) {
        self.staged.remove(&pid);
        self.joins_in_flight.remove(&pid);
        // A lane that departs owing a result owes it to nobody. Dropped here
        // as well as at retirement so the debt ledger cannot outlive the
        // process it is keyed by.
        self.in_flight_since.remove(&pid);
    }

    /// Mirror of the quorum's empty-wait-set re-arm: when the last awaited
    /// lane leaves, the next fleet starts a fresh episode.
    fn maybe_reset_episode(&mut self) {
        if self.lanes.values().any(|lane| lane.awaited) {
            return;
        }
        self.strict_watchdog_deadline = None;
    }

    fn have_seal_candidate(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete))
    }

    /// Whether a boundary is currently open: some awaited lane has not yet
    /// fired into it. While open, the wait-all gate is not re-evaluated —
    /// every lane that still owes was ready when the gate held.
    fn boundary_open(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.awaited && !lane.fired_this_boundary)
    }

    /// Open a fresh boundary: every awaited lane owes it one frame.
    fn open_boundary(&mut self) {
        for lane in self.lanes.values_mut() {
            lane.fired_this_boundary = false;
        }
    }

    /// Close the open boundary without serving the rest of it: no owing lane
    /// can seal (they are all mid-flight, idle or escaped), so control returns
    /// to the wait-all gate rather than to a narrow continuation partition.
    fn close_boundary(&mut self) {
        for lane in self.lanes.values_mut() {
            lane.fired_this_boundary = true;
        }
    }

    /// Seal ONE partition of the open boundary — first-fit against the
    /// per-wave row/token budgets, in this candidate order:
    ///
    /// 1. the lowest-id lane that still owes this boundary a frame (progress
    ///    guarantee: every partition retires at least one, so a boundary
    ///    closes in at most `lanes` partitions);
    /// 2. lanes already served this boundary that have RESUBMITTED — their
    ///    earlier partition retired and the guest answered. This is
    ///    continuing work: on a decode step it costs one token, so it never
    ///    crowds out (3);
    /// 3. the remaining lanes that still owe the boundary a frame.
    ///
    /// Sealing one partition per call is what lets (2) exist at all. Sealing
    /// the WHOLE boundary at once — as this did before — fixed every
    /// partition's content at an instant when no fire had executed, so a
    /// prefill-heavy boundary could only ever produce prefill-only launches
    /// (CONTENTION_FOLLOWUP §20.49). The worker posts at most
    /// `configured_dispatch_depth()` frames, so partitions beyond that were
    /// dispatched long after their content froze.
    ///
    /// Called only once the wait-all gate holds for a CLOSED boundary (no
    /// missing awaited lane, no earmarked successor assembling). Fully
    /// deterministic — no timing input at all.
    fn seal(&mut self) -> Option<FramePlan> {
        if !self.have_seal_candidate() {
            return None;
        }
        // `plan_dispatch` calls in mid-boundary only while one IS open, so
        // otherwise this call OPENS a boundary — but only if it actually
        // seals, so a call that finds nothing sealable leaves the policy
        // untouched and control at the gate.
        let mid_boundary = self.boundary_open();

        // ONE partition per call. Candidate order: the lowest-id lane that has
        // not fired into this boundary first (progress guarantee: every
        // partition retires at least one, so a boundary closes in at most
        // `lanes` partitions), then lanes that already fired into it and have
        // RESUBMITTED — continuing work, one token per decode row, so it never
        // crowds out the rest — then the remaining lanes that have not fired.
        // At a freshly opened boundary nothing has fired, so this is plain
        // lane-id order.
        let k = self.k;
        let max_wave_rows = self.max_wave_rows;
        let max_wave_tokens = self.max_wave_tokens;
        loop {
            let mut fresh: Vec<ProcessId> = Vec::new();
            let mut continuing: Vec<ProcessId> = Vec::new();
            for (lane_id, lane) in self.lanes.iter() {
                if !lane.frames.front().is_some_and(PendingFrame::is_complete) {
                    continue;
                }
                if mid_boundary && lane.fired_this_boundary {
                    continuing.push(*lane_id);
                } else {
                    fresh.push(*lane_id);
                }
            }
            if mid_boundary && fresh.is_empty() {
                // Nothing left that advances this boundary. Sealing the
                // continuation lanes alone would be exactly the narrow epoch
                // the wait-all rule exists to prevent: hand control back so
                // the gate re-gathers.
                return None;
            }
            let mut order: Vec<ProcessId> = Vec::with_capacity(fresh.len() + continuing.len());
            let mut rest = fresh.as_slice();
            if let Some((first, tail)) = fresh.split_first() {
                order.push(*first);
                rest = tail;
            }
            order.extend_from_slice(&continuing);
            order.extend_from_slice(rest);

            let mut waves: Vec<Vec<u64>> = vec![Vec::new(); k];
            let mut fire_waves = HashMap::new();
            let mut wave_tokens = vec![0usize; k];
            let mut wave_rows = vec![0usize; k];
            let mut members: HashSet<ProcessId> = HashSet::new();
            let mut dropped_empty = false;
            for lane_id in order {
                let Some(lane) = self.lanes.get_mut(&lane_id) else {
                    continue;
                };
                let Some(front) = lane.frames.front() else {
                    continue;
                };
                let live: Vec<&ArrivedFire> = front
                    .fires
                    .iter()
                    .filter(|fire| fire.fire_id.is_some())
                    .collect();
                if live.is_empty() {
                    lane.frames.pop_front();
                    dropped_empty = true;
                    continue;
                }
                let fits = live.iter().all(|fire| {
                    let wave = (fire.slot as usize).min(k - 1);
                    wave_rows[wave] + fire.rows.max(1) <= max_wave_rows
                        && wave_tokens[wave] + fire.tokens <= max_wave_tokens
                });
                if !fits {
                    // Over budget: the lane seals into a later partition.
                    continue;
                }
                for fire in live {
                    let wave = (fire.slot as usize).min(k - 1);
                    wave_rows[wave] += fire.rows.max(1);
                    wave_tokens[wave] += fire.tokens;
                    let fire_id = fire.fire_id.expect("live fire has an id");
                    waves[wave].push(fire_id);
                    fire_waves.insert(fire_id, wave);
                }
                members.insert(lane_id);
                lane.frames.pop_front();
            }
            if fire_waves.is_empty() {
                // Nothing sealable. Retry only if a frame that turned out to
                // hold no live fire was dropped — the lane's NEXT front may
                // be sealable, and the old whole-boundary loop caught that.
                if dropped_empty {
                    continue;
                }
                self.lanes
                    .retain(|_, lane| lane.awaited || lane.leashed || !lane.frames.is_empty());
                return None;
            }
            if !mid_boundary {
                self.open_boundary();
            }
            for member in &members {
                if let Some(lane) = self.lanes.get_mut(member) {
                    lane.fired_this_boundary = true;
                }
            }
            self.record_sealed_waves(waves.iter().filter(|wave| !wave.is_empty()).count());
            let _ = &fire_waves;
            self.sealed.push_back(SealedFrame {
                waves,
                members: members.iter().copied().collect(),
            });
            // A leashed lane is kept even with nothing queued: its silence
            // clock is the only thing that can still terminate an abandoned
            // pipeline, and dropping the entry would lose it.
            self.lanes
                .retain(|_, lane| lane.awaited || lane.leashed || !lane.frames.is_empty());
            return Some(FramePlan::Dispatch(Vec::new()));
        }
    }

    /// An unstamped rider batch posted outside the sealed waves: it is
    /// still one wave fire for the density counters (the per-wave quorum
    /// counted untracked-only batches the same way).
    pub fn record_rider_wave(&self) {
        self.record_sealed_waves(1);
    }

    /// Wave-density probe counters (the former quorum `record_wave`/
    /// `record_clause`): `avg_active = wave_active_sum / wave_fires`
    /// discriminates a persistent wait-set from one that empties between
    /// fires. A seal never fires with a missing awaited lane (the wait-all
    /// gate held), so `wave_missing_sum` stays 0 by construction.
    fn record_sealed_waves(&self, wave_count: usize) {
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
        }
    }

    /// The next sealed frame the worker should POST WHOLE, if any.
    ///
    /// `still_queued` tells the policy which ids remain in the worker queue —
    /// sealed ids that vanished (rejected/cancelled) resolve here.
    /// `blocked_lanes` holds lanes with a queued pre-launch copy barrier: a
    /// frame containing such a lane's fire holds until the copy retires
    /// (frames are atomic — nothing partial posts).
    /// `executing` is the worker's in-flight signal (frames posted, not yet
    /// retired) — while true, a blocked gather parks instead of holding.
    ///
    /// Frames OVERLAP on-stream — there is no launch-time barrier. Posting
    /// is globally ordered (frames in seal order; waves in slot order inside
    /// the driver), and the device-side readiness gate (`pass_commit`
    /// channel tickets) orders dependent fires by stream order — a
    /// frame-boundary dependency (f's last wave → f+1's wave 0 of the same
    /// lane) is structurally identical to an intra-frame one. Settlement of
    /// a posted frame proceeds asynchronously and never gates the next
    /// frame's post.
    pub fn plan_dispatch(
        &mut self,
        still_queued: &QueuedFireIds,
        blocked_lanes: &HashSet<ProcessId>,
        executing: bool,
        now: Instant,
    ) -> FramePlan {
        loop {
            // A guest that parked leaves the wait-set before anything is
            // counted missing, so the gather below never sees it.
            self.retire_parks();
            // Resolve sealed ids that left the queue without posting.
            for frame in &mut self.sealed {
                for wave in &mut frame.waves {
                    wave.retain(|fire_id| still_queued.contains(fire_id));
                }
            }
            while self
                .sealed
                .front()
                .is_some_and(|frame| frame.waves.iter().all(Vec::is_empty))
            {
                self.sealed.pop_front();
            }
            if let Some(front) = self.sealed.front() {
                if front
                    .members
                    .iter()
                    .any(|member| blocked_lanes.contains(member))
                {
                    // The copy's retirement re-decides through the scheduler
                    // channel; the hold is only a liveness backstop.
                    return FramePlan::Hold(Duration::from_micros(500));
                }
                let frame = self.sealed.pop_front().expect("front frame exists");
                // The engine now owes every member a result. Their deadline
                // clocks stop here and only restart at `on_frame_retired`,
                // so the wave's own latency is never charged to the guest.
                self.in_flight_lanes.extend(frame.members.iter().copied());
                // Stamped once per lane, not re-stamped: a lane already in
                // debt from an earlier wave is measured from when the debt
                // began, or a steady stream of new waves would refresh the
                // backstop it exists to enforce.
                // `now`, not `Instant::now()`: the scan below measures the debt
                // against the same clock the caller passed, and two clocks
                // that differ by a scheduling delay make the backstop fire a
                // tick late for reasons nothing can see.
                for member in frame.members.iter().copied() {
                    self.in_flight_since.entry(member).or_insert(now);
                }
                return FramePlan::Dispatch(frame.waves);
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
            //
            // The gate guards the OPENING of a boundary, not each of its
            // partitions. While a boundary is open, every lane that still
            // owes it a frame was ready when the gate held, so the next
            // partition cannot be narrow — and re-gating there is exactly
            // what used to force each boundary's whole content to be fixed
            // before any of it executed (CONTENTION_FOLLOWUP §20.49).
            if self.boundary_open() {
                if let Some(plan) = self.seal() {
                    match plan {
                        FramePlan::Dispatch(_) => continue,
                        plan => return plan,
                    }
                }
                // Every remaining owing lane is unsealable (blocked in an
                // executing partition): close the boundary and re-gate.
                self.close_boundary();
            }
            if !self.lanes.values().any(|lane| !lane.frames.is_empty()) {
                // Nothing queued anywhere: no gather episode is running.
                self.strict_watchdog_deadline = None;
                return FramePlan::Park;
            }
            // `missing` counts awaited lanes whose next frame is not fully
            // submitted. There is deliberately NO escape hatch for an EMPTY
            // one: an idle member is waited for like any other.
            //
            // The `bind -> dispatch -> seal -> bind` cycle this used to fear
            // is broken a layer down instead. A bind never orders against a
            // queued launch (`Scheduler::lifecycle_control`), launches are
            // dispatched by id rather than queue position, and
            // `rotate_launch_for_wave_work` rotates a front launch run to the
            // back so a queued bind reaches the driver while this boundary is
            // still unsealed — so a rebinder's next fire is not gated on this
            // seal at all. Measured on `churn` (2048 requests, 512-way): with
            // the escape removed the shape it fired on never persisted 3ms,
            // `wave_missing_sum` stayed 0 across 1886 fires, and wall time was
            // unchanged.
            let mut missing = 0usize;
            // The submit deadline's clock is armed HERE, per lane, because
            // this loop is the exact definition of the interval the guest is
            // answerable for: the lane is a member, it is not complete, and
            // therefore it alone is what the seal is waiting on. A lane that
            // is not missing owes nothing and its clock is disarmed, so the
            // deadline measures consecutive blocking, never lifetime.
            //
            // The clock stops only for debts owed to THIS lane: an unretired
            // dispatch and a bind in flight. Both are cases where the guest
            // physically cannot submit because it is waiting on us.
            //
            // A cohort boundary elsewhere in the engine used to stop every
            // lane's clock too, on the theory that a fleet mid-handover owes
            // its members some slack. Measured on `churn_extreme` (8 runs an
            // arm, same binary, alternating order, quiet host) and `soak`, it
            // moves nothing: 12.13s vs 12.20s wall, 6719 vs 6612 tok/s, zero
            // failures either way. It was a global stop for a per-lane
            // question, and the deadline is honest without it.
            //
            // Deliberately NOT gated on the global `executing` flag either —
            // a silent member has to be found precisely when other lanes are
            // busy, and a global clock would be reset by their traffic
            // forever.
            let mut expired: Vec<ProcessId> = Vec::new();
            let leash = self.submit_deadline;
            let silence = self.silence_timeout;
            // The relaxation applies only while a frame is EXECUTING: with the
            // device idle there is no chain to protect and dense gathering is
            // simply right.
            let contributed_relax = self.gate_contributed && executing;
            // The gate is reached only for a CLOSED boundary — the block above
            // either sealed and restarted the loop, or ran `close_boundary`.
            // That is what makes `fired_this_boundary` uninformative here (it
            // is true of every lane), so `gate_verdict` does not test it.
            debug_assert!(
                !self.boundary_open(),
                "the wait-all gate is evaluated only against a closed boundary"
            );
            for (lane_id, lane) in self.lanes.iter_mut() {
                let owes = self.in_flight_lanes.contains(lane_id)
                    || lane
                        .owner
                        .is_some_and(|owner| self.pending_binds.contains_key(&owner));
                // The debt's own deadline. Generous on purpose -- twice the
                // silence timeout -- because this is a backstop against an
                // engine that will never answer, not a bound on how long a
                // legitimate wave may take. A lane past it is treated as
                // silent again, so the ordinary expiry below reports it
                // instead of the scheduler waiting forever without a word.
                //
                // Hoisted ABOVE the gate verdict because the two compose: the
                // relaxation skips a lane precisely because the engine owes it
                // a result, which is the same state this backstop exists to
                // notice when the result never comes. Skipping such a lane
                // forever would swallow exactly the hang the backstop reports,
                // so a debt past its deadline suppresses the relaxation and
                // the lane is judged by the ordinary ladder below. No measured
                // run has ever carried a debt this old, so the rule the
                // campaign measured is unchanged.
                let debt_since = if owes { self.in_flight_since.get(lane_id).copied() } else { None };
                let owes_forever = debt_since
                    .is_some_and(|from| now.saturating_duration_since(from) >= silence * 2);
                // THE gate rule lives in `gate_verdict`. Leashed: already
                // dropped by the leash below and still silent — it blocks
                // nobody, but the clock keeps running because only a submit or
                // a park ends the silence.
                let blocking = match lane.gate_verdict(contributed_relax && !owes_forever, owes) {
                    GateVerdict::Blocking => true,
                    GateVerdict::SkippedOwed | GateVerdict::Satisfied => false,
                };
                if !blocking && !lane.leashed {
                    lane.clock_from = None;
                    continue;
                }
                if owes && !owes_forever {
                    lane.clock_from = None;
                } else {
                    // Seed the silence clock from when the DEBT began, not
                    // from now: the lane has been unanswered for twice the
                    // timeout already, and starting a fresh clock here would
                    // make the backstop take three timeouts to fire.
                    if owes_forever {
                        lane.clock_from = debt_since;
                    }
                    match lane.clock_from {
                        Some(from) if now.saturating_duration_since(from) >= silence => {
                            // The guest has neither submitted nor parked for
                            // the whole silence timeout with nothing owed to
                            // it. `forward.park()` is how a pipeline says it
                            // is going away; skipping it while holding a
                            // process's resources is the contract breach.
                            lane.awaited = false;
                            lane.leashed = false;
                            lane.clock_from = None;
                            expired.push(lane.owner.unwrap_or(*lane_id));
                            continue;
                        }
                        Some(from) if blocking && now.saturating_duration_since(from) >= leash => {
                            // LEASH, not a verdict. The lane stops being
                            // waited on so the boundary seals immediately —
                            // frames it already submitted still dispatch, and
                            // its next fire rejoins the wait-set through the
                            // ordinary parked path. A guest that simply lost
                            // a race for the CPU is delayed, never failed.
                            lane.awaited = false;
                            lane.parked = true;
                            lane.leashed = true;
                            continue;
                        }
                        Some(_) => {}
                        None => lane.clock_from = Some(now),
                    }
                }
                if blocking {
                    missing += 1;
                }
            }
            if !expired.is_empty() {
                // One process can own several lanes, so the same owner can
                // breach on more than one of them in a single pass.
                expired.sort_unstable();
                expired.dedup();
                return FramePlan::Terminate(expired);
            }
            // A joiner never holds the seal. It is by definition NOT a
            // member yet — no lane in the wait-set — so sealing without it
            // excludes nobody; it only starts the next epoch one process
            // short. A/B on the same binary, with the hold isolated from the
            // deadline clock above (the two are disjoint: the hold needed
            // `missing == 0`, the clock only runs for missing lanes) and the
            // host quiet: NO difference in wall time, tok/s or submit
            // deadline breaches on any scenario. It bought +6% epoch density
            // on `churn_extreme` alone, which never reached throughput — not
            // worth a 20ms timer and an admission earmark to guess at.
            if missing > 0 {
                if executing {
                    // An epoch is executing: its retirements re-decide and
                    // the gather continues in the background.
                    return FramePlan::Park;
                }
                if seal_mode_ready() && self.have_seal_candidate() {
                    // Ready mode, arrival-quiescence rule: open the
                    // boundary from the arrival-complete subset only after
                    // one full hold cycle in which NO new fire arrived —
                    // the reactive burst has ended, so the still-missing
                    // lanes are genuinely slow (parked, evicted, mid-grant)
                    // and holding for them is pure device idle. Opening on
                    // the first idle evaluation instead (no quiescence)
                    // fragmented the waves and lost 13% (see the analysis
                    // addendum): the burst is still landing at that point.
                    // Pure event arithmetic — no timers, no thresholds.
                    if self.quiesce_mark == Some(self.arrival_seq) {
                        self.quiesce_mark = None;
                        self.strict_watchdog_deadline = None;
                        match self.seal() {
                            Some(FramePlan::Dispatch(_)) => continue,
                            Some(plan) => return plan,
                            // Candidates exist but none sealed (e.g. every
                            // ready lane is capacity-deferred): fall through
                            // to the ordinary hold.
                            None => {}
                        }
                    } else {
                        self.quiesce_mark = Some(self.arrival_seq);
                    }
                }
                let deadline = self
                    .strict_watchdog_deadline
                    .get_or_insert(now + Duration::from_micros(STRICT_WATCHDOG_US));
                if now >= *deadline {
                    *deadline = now + Duration::from_micros(STRICT_WATCHDOG_US);
                }
                let plan = FramePlan::Hold(
                    deadline
                        .saturating_duration_since(now)
                        .min(Duration::from_micros(GATHER_POLL_US)),
                );
                return plan;
            }
            self.strict_watchdog_deadline = None;
            // EARLY seal: the gate held (every awaited lane's next frame is
            // fully submitted, no earmarked successor assembling), so seal NOW — normally
            // while the previous frame still executes. Sealing early is
            // what re-merges stragglers into one dense fleet epoch without
            // any drain barrier: a seal never excludes a busy lane, because
            // it waits for every lane's submission instead.
            match self.seal() {
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
            "frame k={} lanes={} awaited={} sealed={} \
pending_binds={} staged={} joins_in_flight={} departing={} suspended={} \
pending_slots={} watchdog={:?}",
            self.k,
            self.lanes.len(),
            self.lanes.values().filter(|lane| lane.awaited).count(),
            self.sealed.len(),
            self.pending_binds.values().sum::<usize>(),
            self.staged.len(),
            self.joins_in_flight.len(),
            self.departing.len(),
            self.suspended.len(),
            self.pending_slots,
            self.strict_watchdog_deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now())),
        );
        let _ = write!(
            out,
            "\n  pending_bind_pids=[{}]",
            self.pending_binds
                .keys()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        for (pid, lane) in &self.lanes {
            let front_complete = lane.frames.front().is_some_and(PendingFrame::is_complete);
            let _ = write!(
                out,
                "\n  lane {pid}: owner={:?} awaited={} queued_frames={} front_complete={front_complete}",
                lane.owner,
                lane.awaited,
                lane.frames.len(),
            );
        }
        for (index, frame) in self.sealed.iter().enumerate() {
            let _ = write!(
                out,
                "\n  sealed[{index}]: waves={} fires={} members={}",
                frame.waves.len(),
                frame.waves.iter().map(Vec::len).sum::<usize>(),
                frame.members.len(),
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

    fn plan(policy: &mut FramePolicy, queued: &QueuedFireIds, now: Instant) -> FramePlan {
        policy.plan_dispatch(queued, &HashSet::new(), false, now)
    }

    /// Seal, then settle: the worker retires every dispatched batch, which is
    /// what pays back the lanes' submit-deadline debt. A test that dispatches
    /// without retiring leaves the engine permanently owing, so the deadline
    /// can never run — mirror the real lifecycle instead.
    fn retire(policy: &mut FramePolicy, lanes: impl IntoIterator<Item = ProcessId>) {
        policy.on_frame_retired(lanes);
    }

    /// Flatten a whole-frame dispatch for order-insensitive membership
    /// asserts.
    fn fires(plan: &FramePlan) -> Vec<u64> {
        match plan {
            FramePlan::Dispatch(waves) => waves.iter().flatten().copied().collect(),
            plan => panic!("expected a frame dispatch, got {plan:?}"),
        }
    }

    #[test]
    fn dispatch_depth_defaults_when_bootstrap_installs_nothing() {
        // Nothing in this binary installs a depth, so the accessor must still
        // hand back a usable one: a driver harness that never boots the worker
        // config path still dispatches.
        assert_eq!(configured_dispatch_depth(), DEFAULT_DISPATCH_DEPTH);
    }

    /// Density at k = 1 comes from the wait-all gate, not from a timer: two
    /// lanes whose first single-slot frames are both ready seal as ONE dense
    /// wave, and a wave missing a member holds until that member submits.
    #[test]
    fn ready_single_slot_lanes_seal_as_one_dense_wave() {
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        let (a, b) = (pid(), pid());
        // Single-slot stamps as the worker synthesizes them at k = 1:
        // seq = the fire id, slot 0, one fire per frame.
        policy.on_fire_enqueued(stamp(a, 10, 0, 1), Some(a), 10, 1, 1);
        policy.on_fire_enqueued(stamp(b, 11, 0, 1), Some(b), 11, 1, 1);
        let queued: QueuedFireIds = [10, 11].into_iter().collect();
        let sealed = plan(&mut policy, &queued, Instant::now());
        assert_eq!(
            fires(&sealed).len(),
            2,
            "dense: both lanes' fires in one wave"
        );

        // Steady state: `a` resubmits, `b` does not — the wave holds.
        policy.on_fire_enqueued(stamp(a, 12, 0, 1), Some(a), 12, 1, 1);
        let queued: QueuedFireIds = [12].into_iter().collect();
        match plan(&mut policy, &queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("wait-all must hold for the idle lane, got {plan:?}"),
        }
        // `b`'s next fire arrives: the wave seals dense.
        policy.on_fire_enqueued(stamp(b, 13, 0, 1), Some(b), 13, 1, 1);
        let queued: QueuedFireIds = [12, 13].into_iter().collect();
        let next = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&next).len(), 2);
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

        let queued: QueuedFireIds = [100, 101, 102, 103, 200].into_iter().collect();
        let sealed = plan(&mut policy, &queued, Instant::now());
        // One whole frame: wave 0 = both lanes' slot-0 fires (lane-id order
        // preserved), later slots in slot order.
        let FramePlan::Dispatch(waves) = sealed else {
            panic!("expected a whole-frame dispatch");
        };
        assert_eq!(waves.len(), 4);
        assert_eq!(waves[0].len(), 2);
        assert!(waves[0].contains(&100) && waves[0].contains(&200));
        assert_eq!(waves[1], vec![101]);
        assert_eq!(waves[2], vec![102]);
        assert_eq!(waves[3], vec![103]);
    }

    /// THE wait-all regression test: an incomplete lane BLOCKS the seal.
    /// The watchdog reports (Hold at its cadence) but never evicts, and no
    /// narrow epoch fires. When the straggler completes, the epoch seals
    /// DENSE with every lane in.
    ///
    /// The submit deadline is parked out past the horizon here on purpose:
    /// it is the *other* answer to a silent lane, and it removes one by
    /// killing it. This test is about what the gather may not do on its own
    /// authority — quietly seal a narrow epoch and leave a live member
    /// behind — which stays forbidden however long the wait runs.
    #[test]
    fn incomplete_lane_holds_the_seal_until_it_completes() {
        let mut policy =
            FramePolicy::new(2, 64, 4096, None).with_submit_deadline(Duration::from_secs(86_400));
        let (fast, slow) = (pid(), pid());
        policy.on_fire_enqueued(stamp(fast, 0, 0, 2), Some(fast), 1, 1, 1);
        policy.on_fire_enqueued(stamp(fast, 0, 1, 2), Some(fast), 2, 1, 1);
        // `slow` declared 2 fires but only one arrived: a missing member.
        policy.on_fire_enqueued(stamp(slow, 0, 0, 2), Some(slow), 3, 1, 1);

        let queued: QueuedFireIds = [1, 2, 3].into_iter().collect();
        let t0 = Instant::now();
        match plan(&mut policy, &queued, t0) {
            // A blocked gather re-looks at the GATHER POLL cadence, not at
            // the watchdog's. The watchdog is report-only and its deadline
            // stays an order of magnitude further out; sleeping the whole
            // way there with the GPU idle was the 2x regression that split
            // the two constants apart.
            FramePlan::Hold(hold) => {
                assert_eq!(hold, Duration::from_micros(GATHER_POLL_US));
                assert!(hold < Duration::from_micros(STRICT_WATCHDOG_US));
            }
            plan => panic!("wait-all must hold for the incomplete lane, got {plan:?}"),
        }
        // Long past the watchdog: it re-arms and reports; it never fires.
        match plan(&mut policy, &queued, t0 + Duration::from_secs(60)) {
            FramePlan::Hold(_) => {}
            plan => panic!("the watchdog reports, it must not fire: got {plan:?}"),
        }

        // The straggler completes: one dense epoch, both lanes' slot-0.
        policy.on_fire_enqueued(stamp(slow, 0, 1, 2), Some(slow), 4, 1, 1);
        let queued: QueuedFireIds = [1, 2, 3, 4].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("all lanes ready: the epoch must seal");
        };
        assert_eq!(waves[0].len(), 2, "dense wave 0 holds BOTH lanes");
        assert!(waves[0].contains(&1) && waves[0].contains(&3));
    }

    /// Venus: a sealed frame dispatches WHOLE — every wave in slot order in
    /// one plan — and the policy tracks nothing afterwards (the worker owns
    /// posting and retirement; overlap is the worker's run-ahead depth).
    #[test]
    fn sealed_frame_dispatches_whole_and_frames_overlap() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 50, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 51, 1, 1);
        let queued: QueuedFireIds = [50, 51].into_iter().collect();
        let FramePlan::Dispatch(frame0) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected lane a's whole frame");
        };
        assert_eq!(frame0, vec![vec![50], vec![51]]);
        // Mid-execution (worker in flight), a straggler submits its first
        // frame and lane a its next: the wait-all gate holds, so f+1 seals
        // NOW and dispatches whole behind the executing frame.
        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 60, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 61, 1, 1);
        policy.on_fire_enqueued(stamp(a, 1, 0, 2), Some(a), 52, 1, 1);
        policy.on_fire_enqueued(stamp(a, 1, 1, 2), Some(a), 53, 1, 1);
        let queued: QueuedFireIds = [52, 53, 60, 61].into_iter().collect();
        let FramePlan::Dispatch(merged) =
            policy.plan_dispatch(&queued, &HashSet::new(), true, Instant::now())
        else {
            panic!("the overlapped next frame must seal and dispatch whole");
        };
        assert_eq!(merged[0].len(), 2, "wave 0 must hold BOTH lanes");
        assert!(merged[0].contains(&52) && merged[0].contains(&60));
        assert_eq!(merged[1].len(), 2);
        assert!(merged[1].contains(&53) && merged[1].contains(&61));
    }

    /// A frame whose lane has a queued pre-launch copy barrier holds WHOLE
    /// (frames are atomic); it dispatches once the barrier clears.
    #[test]
    fn blocked_lane_holds_the_whole_frame() {
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 70, 1, 1);
        let queued: QueuedFireIds = [70].into_iter().collect();
        let blocked: HashSet<ProcessId> = [lane].into_iter().collect();
        // Seal happens; dispatch holds on the blocked lane.
        let now = Instant::now();
        let held = match policy.plan_dispatch(&queued, &blocked, false, now) {
            FramePlan::Hold(hold) => policy.plan_dispatch(
                &queued,
                &blocked,
                false,
                now + hold + Duration::from_micros(1),
            ),
            plan => plan,
        };
        assert!(
            matches!(held, FramePlan::Hold(_)),
            "a blocked member must hold the whole frame, got {held:?}"
        );
        let FramePlan::Dispatch(waves) =
            policy.plan_dispatch(&queued, &HashSet::new(), false, Instant::now())
        else {
            panic!("the frame must dispatch once the barrier clears");
        };
        assert_eq!(waves, vec![vec![70]]);
    }

    #[test]
    fn dropped_fires_resolve_and_leave_rearms_the_episode() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 20, 1, 1);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 2), Some(lane), 21, 1, 1);
        let queued: QueuedFireIds = [20, 21].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected the whole frame");
        };
        assert_eq!(waves, vec![vec![20], vec![21]]);
        let queued = QueuedFireIds::default();
        assert_eq!(plan(&mut policy, &queued, Instant::now()), FramePlan::Park);
        assert_eq!(policy.sealed.len(), 0, "frame popped at dispatch");
        // The lane's LEAVE empties the wait-set; the episode's liveness
        // watchdog re-arms with it.
        policy.on_lane_leave(lane, None, false);
        assert!(
            policy.strict_watchdog_deadline.is_none(),
            "an emptied wait-set re-arms the episode"
        );
    }

    /// CONTENTION_FOLLOWUP §20.8, half A: a fire can overtake the planner's
    /// suspend broadcast. At k > 1 the first slot alone used to resurrect the
    /// lane's wait-set membership while the rest of the frame sat behind the
    /// eviction, so the half-arrived frame held the fleet's seal forever.
    /// Half B: that slot must still SEAL, or its fire lease is never
    /// released and the eviction's quiescence wait wedges instead.
    #[test]
    fn a_fire_racing_the_suspend_seals_alone_without_rejoining_the_wait_set() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (victim, healthy) = {
            let (x, y) = (pid(), pid());
            if x < y { (x, y) } else { (y, x) }
        };
        // Steady state: both lanes complete a frame, so both are awaited.
        policy.on_fire_enqueued(stamp(victim, 0, 0, 2), Some(victim), 100, 1, 1);
        policy.on_fire_enqueued(stamp(victim, 0, 1, 2), Some(victim), 101, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 0, 0, 2), Some(healthy), 102, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 0, 1, 2), Some(healthy), 103, 1, 1);
        let queued: QueuedFireIds = [100, 101, 102, 103].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, Instant::now()),
            FramePlan::Dispatch(_)
        ));

        // The planner evicts the victim, then its slot 0 for the NEXT frame
        // lands (already past the eviction fence). Slot 1 cannot follow: the
        // guest is parked behind the very eviction that is waiting on this
        // fire's lease.
        policy.on_process_suspend(victim);
        policy.on_fire_enqueued(stamp(victim, 1, 0, 2), Some(victim), 200, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 1, 0, 2), Some(healthy), 300, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 1, 1, 2), Some(healthy), 301, 1, 1);
        assert!(
            !policy.lanes[&victim].awaited,
            "a suspended owner's arrival must not rejoin the wait-set"
        );

        let queued: QueuedFireIds = [200, 300, 301].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("the boundary must seal without waiting for the victim");
        };
        assert_eq!(
            waves[0],
            vec![200, 300],
            "the victim's stranded slot seals too — that lease has to drain"
        );
        assert_eq!(waves[1], vec![301]);

        // Post-restore the guest resumes inside its submit loop and sends the
        // slot the truncation cut off. It stands alone rather than re-forming
        // an unsatisfiable 2-slot frame under a seq that already sealed.
        policy.on_process_resume(victim);
        policy.on_fire_enqueued(stamp(victim, 1, 1, 2), Some(victim), 201, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 2, 0, 2), Some(healthy), 302, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 2, 1, 2), Some(healthy), 303, 1, 1);
        let queued: QueuedFireIds = [201, 302, 303].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("the late slot must seal");
        };
        assert_eq!(waves[1], vec![201, 303], "the late slot keeps its wave");

        // Fully rejoined: the next boundary waits for the victim again.
        policy.on_fire_enqueued(stamp(victim, 2, 0, 2), Some(victim), 400, 1, 1);
        policy.on_fire_enqueued(stamp(victim, 2, 1, 2), Some(victim), 401, 1, 1);
        assert!(
            policy.lanes[&victim].awaited,
            "a resumed process rejoins on its next frame"
        );
    }

    /// The same stranding reachable WITHOUT the planner: a KV allocation park
    /// posts a lane close mid-frame, and the guest cannot finish that frame
    /// until it is served — which needs the fleet to advance. The submitted
    /// slot must seal so the circle never forms.
    #[test]
    fn a_lane_parked_mid_frame_seals_what_it_submitted() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 10, 1, 1);
        // Parked before slot 1: the allocation wait posts a lane close.
        policy.on_lane_leave(lane, Some(lane), false);
        let queued: QueuedFireIds = [10].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("a parked lane's submitted slot must still seal");
        };
        assert_eq!(waves[0], vec![10]);
        assert!(waves[1].is_empty());
    }

    #[test]
    fn truncated_frame_seals_with_submitted_fires_only() {
        let mut policy = FramePolicy::new(4, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 4), Some(lane), 30, 1, 1);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 4), Some(lane), 31, 1, 1);
        // Host submit failed at slot 2: only 2 fires exist.
        policy.on_frame_truncated(lane, 0, 2);
        let queued: QueuedFireIds = [30, 31].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("truncated frame must still seal");
        };
        assert_eq!(waves[0], vec![30]);
        assert_eq!(waves[1], vec![31]);
        assert!(waves[2].is_empty() && waves[3].is_empty());
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
        let queued: QueuedFireIds = [40, 41].into_iter().collect();
        let FramePlan::Dispatch(frame_a) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected a seal");
        };
        assert_eq!(frame_a[0], vec![40], "first-fit admits lane a only");
        // Lane a is served (round rule) — b seals NOW, while a's partition
        // is still in flight, because their lane sets are disjoint.
        let FramePlan::Dispatch(frame_b) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("the deferred lane must seal within the round");
        };
        assert_eq!(frame_b[0], vec![41]);
        // Round closed at b's seal: the next epoch awaits BOTH lanes again.
        let queued = QueuedFireIds::default();
        assert_eq!(plan(&mut policy, &queued, Instant::now()), FramePlan::Park);
    }

    /// A boundary is sealed one partition at a time, so a lane whose earlier
    /// partition already RETIRED rejoins a later partition of the SAME
    /// boundary. This is what lets a prefill-heavy boundary produce mixed
    /// prefill+decode launches: sealing the whole boundary at once fixed
    /// every partition's content before any of it had executed, so the
    /// trailing partitions could only ever be prefill-only
    /// (CONTENTION_FOLLOWUP §20.49).
    #[test]
    fn a_retired_lane_rejoins_a_later_partition_of_the_same_boundary() {
        // 40-token wave budget: one 37-token prefill per partition.
        let mut policy = FramePolicy::new(1, 64, 40, None);
        let (a, b, c) = {
            let mut ids = [pid(), pid(), pid()];
            ids.sort_unstable();
            (ids[0], ids[1], ids[2])
        };
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 40, 37, 1);
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 41, 37, 1);
        policy.on_fire_enqueued(stamp(c, 0, 0, 1), Some(c), 42, 37, 1);
        let queued: QueuedFireIds = [40, 41, 42].into_iter().collect();
        let FramePlan::Dispatch(p1) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected the first partition");
        };
        assert_eq!(p1[0], vec![40], "first-fit admits lane a only");
        let FramePlan::Dispatch(p2) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected the second partition");
        };
        assert_eq!(p2[0], vec![41]);

        // Lane a's partition retires and the guest answers with a decode
        // row. The boundary is still open (c has not fired into it), so a's
        // decode rides along with c's prefill instead of paying for its own
        // near-empty launch.
        policy.on_fire_enqueued(stamp(a, 1, 0, 1), Some(a), 43, 1, 1);
        let queued: QueuedFireIds = [42, 43].into_iter().collect();
        let FramePlan::Dispatch(p3) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected the third partition");
        };
        assert_eq!(
            p3[0],
            vec![42, 43],
            "the continuing decode joins the fresh prefill"
        );

        // Boundary closed at c's seal: the gate re-gathers everyone.
        let queued: QueuedFireIds = QueuedFireIds::default();
        assert_eq!(plan(&mut policy, &queued, Instant::now()), FramePlan::Park);
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
        let queued: QueuedFireIds = [60, 61].into_iter().collect();
        let FramePlan::Dispatch(frame_a) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected lane a's whole frame");
        };
        assert_eq!(frame_a, vec![vec![60], vec![61]]);

        // b arrives with a complete frame; a has NOT resubmitted. Wait-all:
        // no seal until a's next frame is in (or a leaves).
        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 70, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 71, 1, 1);
        let queued: QueuedFireIds = [70, 71].into_iter().collect();
        match plan(&mut policy, &queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("the epoch must wait for lane a to resubmit, got {plan:?}"),
        }

        // a resubmits: the epoch seals DENSE with both lanes.
        policy.on_fire_enqueued(stamp(a, 1, 0, 2), Some(a), 80, 1, 1);
        policy.on_fire_enqueued(stamp(a, 1, 1, 2), Some(a), 81, 1, 1);
        let queued: QueuedFireIds = [70, 71, 80, 81].into_iter().collect();
        let FramePlan::Dispatch(dense) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("all lanes ready: the epoch must seal");
        };
        assert_eq!(dense[0].len(), 2, "dense wave 0 holds both lanes");
        assert!(dense[0].contains(&70) && dense[0].contains(&80));
    }

    /// Graceful close is the ONLY way a straggler stops being awaited: the
    /// lane leaves the wait-set immediately and the fleet seals without it.
    #[test]
    fn graceful_close_releases_the_wait() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 90, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 91, 1, 1);
        let queued: QueuedFireIds = [90, 91].into_iter().collect();
        let bootstrap = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&bootstrap).len(), 2);

        // b resubmits; a does not — the gather blocks on a.
        policy.on_fire_enqueued(stamp(b, 1, 0, 1), Some(b), 92, 1, 1);
        let queued: QueuedFireIds = [92].into_iter().collect();
        match plan(&mut policy, &queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("the gather must block on lane a, got {plan:?}"),
        }
        // a closes gracefully: released from the wait-set, b seals.
        policy.on_lane_leave(a, None, false);
        let next = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&next), vec![92]);
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
        let queued: QueuedFireIds = [95].into_iter().collect();
        let sealed = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&sealed), vec![95]);
    }

    /// The owner is still recovered from the lane when the caller does not
    /// supply one, so pipeline-scoped closes keep working unchanged.
    #[test]
    fn lane_leave_without_an_owner_recovers_it_from_the_lane() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        let owner = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(owner), 95, 1, 1);
        policy.on_bind_enqueued(Some(owner)); // stages the OWNER, not the lane
        policy.on_lane_leave(lane, None, true);
        assert!(
            !policy.staged.contains(&owner),
            "the lane's recorded owner must clear the process-keyed staging"
        );
    }

    /// The departure hold arms only for actual slot holders, exactly once:
    /// a Terminate for a never-admitted pid (or a duplicate leave from the
    /// exit funnel's two notification paths) leaves no phantom hold, and a
    /// staged successor whose predecessor's release was already consumed
    /// elsewhere does not re-open the window.
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
        let queued: QueuedFireIds = [95, 96].into_iter().collect();
        assert!(
            matches!(
                plan(&mut policy, &queued, Instant::now()),
                FramePlan::Dispatch(_)
            ),
            "a retired departure must leave no phantom hold"
        );
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
        let queued: QueuedFireIds = [95].into_iter().collect();
        assert!(
            matches!(
                plan(&mut policy, &queued, Instant::now()),
                FramePlan::Dispatch(_)
            ),
            "a drained release must not hold for a staged bystander"
        );
    }

    #[test]
    fn terminate_purges_queued_frames_close_keeps_them() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (closed, terminated) = (pid(), pid());
        let owner = pid();
        policy.on_fire_enqueued(stamp(closed, 0, 0, 1), Some(owner), 50, 1, 1);
        policy.on_fire_enqueued(stamp(terminated, 0, 0, 1), Some(owner), 51, 1, 1);

        policy.on_lane_leave(closed, None, false);
        policy.on_lane_leave(terminated, None, true);
        let queued: QueuedFireIds = [50].into_iter().collect();
        let sealed = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&sealed), vec![50]);
    }

    /// The whole point of `park`: a lane that stops submitting releases the
    /// wait-all gate instead of holding the fleet until an escape fires.
    /// Same setup as `next_epoch_waits_for_every_awaited_lane`, which asserts
    /// the block; this asserts the exit.
    #[test]
    fn park_releases_the_wait_all_gate() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 60, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 61, 1, 1);
        let queued: QueuedFireIds = [60, 61].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![60, 61]
        );

        // b has a complete frame; a is awaited but silent, so nothing seals.
        policy.on_fire_enqueued(stamp(b, 1, 0, 2), Some(b), 70, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 1, 2), Some(b), 71, 1, 1);
        let queued: QueuedFireIds = [70, 71].into_iter().collect();
        assert!(
            !matches!(
                plan(&mut policy, &queued, Instant::now()),
                FramePlan::Dispatch(_)
            ),
            "wait-all must hold while a is still a member"
        );

        // a leaves the wait-set: b's frame is now the whole quorum.
        policy.on_lane_park(a, 1);
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![70, 71]
        );
    }

    /// Park is ordered against the lane's OWN submits, not against the call:
    /// a frame already in the queue still seals with the parking lane in it.
    /// This is what makes it legal to park with fires outstanding.
    #[test]
    fn park_takes_effect_only_after_queued_frames_seal() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 80, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 81, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 90, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 91, 1, 1);
        // a parks immediately after submitting — the exit sits behind seq 0.
        policy.on_lane_park(a, 1);

        let queued: QueuedFireIds = [80, 81, 90, 91].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, Instant::now()));
        sealed.sort_unstable();
        assert_eq!(
            sealed,
            vec![80, 81, 90, 91],
            "a's queued frame still counts"
        );

        // Now the park is at the head: b seals alone.
        policy.on_fire_enqueued(stamp(b, 1, 0, 2), Some(b), 92, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 1, 2), Some(b), 93, 1, 1);
        let queued: QueuedFireIds = [92, 93].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![92, 93]
        );
    }

    /// There is no `join`: the next submit rejoins atomically with the slot
    /// it contributes, so the wait-set never holds a member that has joined
    /// but not yet submitted.
    #[test]
    fn submit_after_park_rejoins_the_wait_set() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 100, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 101, 1, 1);
        let queued: QueuedFireIds = [100, 101].into_iter().collect();
        plan(&mut policy, &queued, Instant::now());
        policy.on_lane_park(a, 1);

        // a rejoins with a PARTIAL frame while b is complete: wait-all must
        // hold again, which it only can if the submit restored membership.
        policy.on_fire_enqueued(stamp(a, 2, 0, 2), Some(a), 110, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 0, 2), Some(b), 120, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 1, 2), Some(b), 121, 1, 1);
        let queued: QueuedFireIds = [110, 120, 121].into_iter().collect();
        assert!(
            !matches!(
                plan(&mut policy, &queued, Instant::now()),
                FramePlan::Dispatch(_)
            ),
            "the rejoined lane must be awaited again"
        );

        policy.on_fire_enqueued(stamp(a, 2, 1, 2), Some(a), 111, 1, 1);
        let queued: QueuedFireIds = [110, 111, 120, 121].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, Instant::now()));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![110, 111, 120, 121]);
    }

    /// Parking twice with no submit in between is a no-op, not a second exit
    /// that could survive the rejoining submit.
    #[test]
    fn double_park_is_a_no_op() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 130, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 131, 1, 1);
        let queued: QueuedFireIds = [130, 131].into_iter().collect();
        plan(&mut policy, &queued, Instant::now());
        policy.on_lane_park(a, 1);
        policy.on_lane_park(a, 2);

        // One submit must be enough to rejoin; a leftover second park marker
        // would swallow it and let b seal without a.
        policy.on_fire_enqueued(stamp(a, 3, 0, 2), Some(a), 140, 1, 1);
        policy.on_fire_enqueued(stamp(a, 3, 1, 2), Some(a), 141, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 0, 2), Some(b), 150, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 1, 2), Some(b), 151, 1, 1);
        let queued: QueuedFireIds = [140, 141, 150, 151].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, Instant::now()));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![140, 141, 150, 151]);
    }

    /// Parking a lane the policy has never seen fire is a no-op: there is no
    /// membership to leave, and inventing one would make the lane a member.
    #[test]
    fn park_of_an_unknown_lane_is_ignored() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_lane_park(b, 0);
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 160, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 161, 1, 1);
        let queued: QueuedFireIds = [160, 161].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![160, 161]
        );
    }

    /// A member that stops submitting is dropped from the wait-set once the
    /// deadline elapses, so the boundary seals without it — but it is NOT
    /// failed. The leash is an involuntary `forward.park()`: the lane's next
    /// fire rejoins it through the ordinary parked path. A guest that merely
    /// lost a race for the CPU costs the fleet one epoch, not its request.
    #[test]
    fn silent_member_is_leashed_at_the_submit_deadline() {
        let deadline = Duration::from_millis(50);
        let mut policy = FramePolicy::new(2, 64, 4096, None).with_submit_deadline(deadline);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 200, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 201, 1, 1);
        let queued: QueuedFireIds = [200, 201].into_iter().collect();
        plan(&mut policy, &queued, Instant::now());
        retire(&mut policy, [a]);

        // b is complete, a is silent: the seal blocks on a.
        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 210, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 211, 1, 1);
        let queued: QueuedFireIds = [210, 211].into_iter().collect();
        let now = Instant::now();
        assert!(
            !matches!(plan(&mut policy, &queued, now), FramePlan::Dispatch(_)),
            "the clock arms on this pass; nothing is due yet"
        );
        assert_eq!(
            fires(&plan(&mut policy, &queued, now + deadline)),
            vec![210, 211],
            "a leaves the wait-set at the deadline and b seals in the same pass"
        );
        assert!(policy.leashed(a), "a left involuntarily, not by parking");

        // Nothing was killed: a's next fire puts it back in the quorum.
        policy.on_fire_enqueued(stamp(a, 1, 0, 2), Some(a), 220, 1, 1);
        assert!(!policy.leashed(a), "the submit ended the silence");
        policy.on_fire_enqueued(stamp(a, 1, 1, 2), Some(a), 221, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 0, 2), Some(b), 230, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 1, 2), Some(b), 231, 1, 1);
        retire(&mut policy, [a, b]);
        let queued: QueuedFireIds = [220, 221, 230, 231].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, Instant::now()));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![220, 221, 230, 231], "a is a full member again");
    }

    /// The leash spares a slow guest; it does not spare an abandoned one. A
    /// pipeline that means to stop calls `forward.park()`, which ends the
    /// silence and is never killed. Staying silent through the leash and on
    /// past the silence timeout is the breach the terminate answers.
    #[test]
    fn an_abandoned_lane_is_terminated_at_the_silence_timeout() {
        let deadline = Duration::from_millis(50);
        let silence = Duration::from_secs(30);
        let mut policy = FramePolicy::new(2, 64, 4096, None)
            .with_submit_deadline(deadline)
            .with_silence_timeout(silence);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 240, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 241, 1, 1);
        let queued: QueuedFireIds = [240, 241].into_iter().collect();
        plan(&mut policy, &queued, Instant::now());
        retire(&mut policy, [a]);

        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 250, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 251, 1, 1);
        let queued: QueuedFireIds = [250, 251].into_iter().collect();
        let now = Instant::now();
        plan(&mut policy, &queued, now);
        assert_eq!(
            fires(&plan(&mut policy, &queued, now + deadline)),
            vec![250, 251]
        );
        assert!(policy.leashed(a));
        retire(&mut policy, [b]);

        // The scan rides the gather, so the fleet has to still be working for
        // an abandoned lane to be noticed: b keeps submitting while a stays
        // silent. (An engine with nothing at all to do parks and reaps
        // nobody — there is no fleet to protect at that point.)
        let tick = |policy: &mut FramePolicy, seq: u64, at: Instant| -> FramePlan {
            let base = 260u64 + seq * 2;
            policy.on_fire_enqueued(stamp(b, seq, 0, 2), Some(b), base, 1, 1);
            policy.on_fire_enqueued(stamp(b, seq, 1, 2), Some(b), base + 1, 1, 1);
            let queued: QueuedFireIds = [base, base + 1].into_iter().collect();
            let out = plan(policy, &queued, at);
            retire(policy, [b]);
            out
        };

        // The clock does NOT restart at the leash: silence is measured from
        // when it began, so a lane cannot buy time by being dropped.
        assert!(
            !matches!(
                tick(&mut policy, 1, now + silence - deadline),
                FramePlan::Terminate(_)
            ),
            "still inside the budget"
        );
        assert_eq!(
            tick(&mut policy, 2, now + silence),
            FramePlan::Terminate(vec![a]),
            "silent through the leash and past the timeout, without parking"
        );
    }

    /// A lockstep lane — one that awaits each result before submitting the
    /// next fire — must survive its own turnaround. Between dispatch and
    /// retirement the lane *cannot* submit: it is waiting on us. Charging it
    /// for that interval would make the deadline lethal to correct guests at
    /// any value below a GPU wave, which at 50ms is every one of them. The
    /// debt is only settled by retirement.
    #[test]
    fn a_lane_awaiting_its_own_result_is_never_charged_for_the_wait() {
        let deadline = Duration::from_millis(50);
        let mut policy = FramePolicy::new(2, 64, 4096, None).with_submit_deadline(deadline);
        let (a, b) = (pid(), pid());
        for (owner, base) in [(a, 300), (b, 302)] {
            policy.on_fire_enqueued(stamp(owner, 0, 0, 2), Some(owner), base, 1, 1);
            policy.on_fire_enqueued(stamp(owner, 0, 1, 2), Some(owner), base + 1, 1, 1);
        }
        let queued: QueuedFireIds = [300, 301, 302, 303].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, Instant::now()),
            FramePlan::Dispatch(_)
        ));

        // b's result came back and it is part-way through its next frame; a's
        // has not, so a is still owed while b drives the gather.
        policy.on_frame_retired([b]);
        policy.on_fire_enqueued(stamp(b, 1, 0, 2), Some(b), 310, 1, 1);
        let queued: QueuedFireIds = [310].into_iter().collect();
        let now = Instant::now();
        for elapsed in [Duration::ZERO, deadline, deadline * 100] {
            plan(&mut policy, &queued, now + elapsed);
            assert!(
                !policy.leashed(a),
                "a is waiting on the engine, not the other way round"
            );
        }

        // The result lands. Only now does a's silence become its own, and it
        // is charged from that moment — not retroactively for the wait.
        policy.on_frame_retired([a]);
        let armed = now + deadline * 100;
        plan(&mut policy, &queued, armed);
        assert!(!policy.leashed(a));
        plan(&mut policy, &queued, armed + deadline);
        assert!(
            policy.leashed(a),
            "the clock starts at retirement and runs its full budget from there"
        );
    }

    /// ...but the debt itself has a deadline. A wave the driver never retires
    /// -- a command buffer returned out of memory, say -- leaves the lane in
    /// `in_flight_lanes` forever, and the branch above resets its clock on
    /// every tick. The scheduler then waits for a result that cannot arrive,
    /// with no message and no bound. Past twice the silence timeout the debt
    /// stops excusing the silence and the ordinary expiry reports it.
    #[test]
    fn a_result_that_never_arrives_does_not_excuse_the_silence_forever() {
        let deadline = Duration::from_millis(50);
        let silence = Duration::from_secs(30);
        let mut policy = FramePolicy::new(2, 64, 4096, None)
            .with_submit_deadline(deadline)
            .with_silence_timeout(silence);
        let (a, b) = (pid(), pid());
        for (owner, base) in [(a, 400), (b, 402)] {
            policy.on_fire_enqueued(stamp(owner, 0, 0, 2), Some(owner), base, 1, 1);
            policy.on_fire_enqueued(stamp(owner, 0, 1, 2), Some(owner), base + 1, 1, 1);
        }
        let queued: QueuedFireIds = [400, 401, 402, 403].into_iter().collect();
        let now = Instant::now();
        assert!(matches!(plan(&mut policy, &queued, now), FramePlan::Dispatch(_)));

        // b's wave retires and b keeps the fleet working; a's never does.
        let tick = |policy: &mut FramePolicy, seq: u64, at: Instant| -> FramePlan {
            let base = 410u64 + seq * 2;
            policy.on_fire_enqueued(stamp(b, seq, 0, 2), Some(b), base, 1, 1);
            policy.on_fire_enqueued(stamp(b, seq, 1, 2), Some(b), base + 1, 1, 1);
            let queued: QueuedFireIds = [base, base + 1].into_iter().collect();
            let out = plan(policy, &queued, at);
            policy.on_frame_retired([b]);
            out
        };
        policy.on_frame_retired([b]);

        // Inside the debt's budget a is excused, however long the fleet runs.
        assert!(
            !matches!(tick(&mut policy, 1, now + silence), FramePlan::Terminate(_)),
            "a wave may legitimately take a long time"
        );
        assert_eq!(
            tick(&mut policy, 2, now + silence * 2),
            FramePlan::Terminate(vec![a]),
            "a result that never came back cannot excuse the silence forever"
        );
    }

    /// The clock measures consecutive blocking, not lifetime: a lane that
    /// keeps submitting is never at risk however long it lives.
    #[test]
    fn submitting_lane_never_trips_the_deadline() {
        let deadline = Duration::from_millis(50);
        let mut policy = FramePolicy::new(2, 64, 4096, None).with_submit_deadline(deadline);
        let (a, b) = (pid(), pid());
        let mut now = Instant::now();
        for round in 0..8u64 {
            policy.on_fire_enqueued(stamp(a, round, 0, 2), Some(a), 300 + round * 4, 1, 1);
            policy.on_fire_enqueued(stamp(a, round, 1, 2), Some(a), 301 + round * 4, 1, 1);
            policy.on_fire_enqueued(stamp(b, round, 0, 2), Some(b), 302 + round * 4, 1, 1);
            policy.on_fire_enqueued(stamp(b, round, 1, 2), Some(b), 303 + round * 4, 1, 1);
            let queued: QueuedFireIds = (300 + round * 4..304 + round * 4).collect();
            // Each round costs more than the deadline in wall clock.
            now += deadline * 2;
            let plan = match plan(&mut policy, &queued, now) {
                FramePlan::Hold(hold) => super::FramePolicy::plan_dispatch(
                    &mut policy,
                    &queued,
                    &HashSet::new(),
                    false,
                    now + hold + Duration::from_micros(1),
                ),
                other => other,
            };
            assert!(
                matches!(plan, FramePlan::Dispatch(_)),
                "round {round}: both lanes submitted, so neither is blocking"
            );
        }
    }

    /// `park` is the sanctioned way to stop, so it must also stop the clock —
    /// otherwise the deadline would kill exactly the guests that complied.
    #[test]
    fn parked_lane_is_not_subject_to_the_deadline() {
        let deadline = Duration::from_millis(50);
        let mut policy = FramePolicy::new(2, 64, 4096, None).with_submit_deadline(deadline);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 400, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 401, 1, 1);
        let queued: QueuedFireIds = [400, 401].into_iter().collect();
        plan(&mut policy, &queued, Instant::now());
        retire(&mut policy, [a]);
        policy.on_lane_park(a, 1);

        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 410, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 411, 1, 1);
        let queued: QueuedFireIds = [410, 411].into_iter().collect();
        let now = Instant::now();
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![410, 411]
        );
        // Long past the deadline, the parked lane is still alive and can
        // rejoin — it left the wait-set, so it was never blocking anything.
        let queued = QueuedFireIds::default();
        assert!(
            !matches!(
                plan(&mut policy, &queued, now + deadline * 100),
                FramePlan::Terminate(_)
            ),
            "a parked lane owes nothing and must not be killed"
        );
        policy.on_fire_enqueued(stamp(a, 2, 0, 2), Some(a), 420, 1, 1);
        policy.on_fire_enqueued(stamp(a, 2, 1, 2), Some(a), 421, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 0, 2), Some(b), 430, 1, 1);
        policy.on_fire_enqueued(stamp(b, 1, 1, 2), Some(b), 431, 1, 1);
        let queued: QueuedFireIds = [420, 421, 430, 431].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, Instant::now()));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![420, 421, 430, 431]);
    }

    /// A bring-up lane's first frame is a join in flight, and the
    /// cohort-boundary window exists so it lands in THIS boundary rather than
    /// a narrow epoch. A half-arrived frame is work the gate must still wait
    /// for, so the contributed relaxation must not touch it: only an EMPTY
    /// queue proves the guest had nothing it could have submitted.
    #[test]
    fn a_fresh_lane_with_a_half_arrived_frame_still_holds_the_gate() {
        let mut policy = FramePolicy::new(2, 64, 4096, None).with_gate_contributed(true);
        let (a, fresh) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 10, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 11, 1, 1);
        let queued: QueuedFireIds = [10, 11].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())).len(), 2);
        retire(&mut policy, [a]);

        // A newly admitted process submits slot 0 of its first frame while a
        // frame executes, and the incumbent's next frame is complete.
        policy.on_bind_enqueued(Some(fresh));
        policy.on_fire_enqueued(stamp(a, 1, 0, 2), Some(a), 12, 1, 1);
        policy.on_fire_enqueued(stamp(a, 1, 1, 2), Some(a), 13, 1, 1);
        policy.on_fire_enqueued(stamp(fresh, 0, 0, 2), Some(fresh), 20, 1, 1);
        let queued: QueuedFireIds = [12, 13, 20].into_iter().collect();
        assert!(
            matches!(
                policy.plan_dispatch(&queued, &HashSet::new(), true, Instant::now()),
                FramePlan::Park
            ),
            "a fresh join keeps the narrow-epoch protection"
        );
    }

    /// The engine's own debts hold the clock: a lane whose rebind the engine
    /// has not finished cannot submit, so the deadline must not run against
    /// it. Without this the leash would drop guests for engine latency, and
    /// the silence timeout would eventually kill them for it.
    #[test]
    fn engine_debt_suspends_the_deadline_clock() {
        let deadline = Duration::from_millis(50);
        let mut policy = FramePolicy::new(2, 64, 4096, None).with_submit_deadline(deadline);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 500, 1, 1);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 501, 1, 1);
        let queued: QueuedFireIds = [500, 501].into_iter().collect();
        plan(&mut policy, &queued, Instant::now());
        retire(&mut policy, [a]);

        // a is mid-rebind: the engine owes it, so its silence is not a breach.
        policy.on_bind_enqueued(Some(a));
        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 510, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 511, 1, 1);
        let queued: QueuedFireIds = [510, 511].into_iter().collect();
        let now = Instant::now();
        assert!(
            !matches!(
                plan(&mut policy, &queued, now + deadline * 10),
                FramePlan::Terminate(_)
            ),
            "the engine, not the guest, is why a has not submitted"
        );

        // Bind done: now the clock runs, and only from here.
        policy.on_bind_completed(Some(a));
        let armed = now + deadline * 10;
        plan(&mut policy, &queued, armed);
        assert!(!policy.leashed(a));
        plan(&mut policy, &queued, armed + deadline);
        assert!(policy.leashed(a));
    }

    // =====================================================================
    // The contributed-and-owed gate relaxation (`PIE_GATE_CONTRIBUTED`,
    // default ON — analysis.md 10.24-10.26). These pin `gate_verdict`'s rule:
    // `awaited && frames.is_empty() && owes`. All four drive the SAME shape
    // and vary exactly one input, so a regression names its own cause.
    // =====================================================================

    /// One dense boundary, then `b` submits again and `a` does not — because
    /// `a` cannot: the engine owes it the result of the frame still on the
    /// device, and its guest is credit-bound on exactly that. Waiting for `a`
    /// would make chaining structurally impossible once the fleet is one frame
    /// deep, which is the 0% chain-engagement regime of §10.22.
    fn contributed_scenario(relax: bool) -> (FramePolicy, ProcessId, ProcessId) {
        let mut policy = FramePolicy::new(1, 64, 4096, None).with_gate_contributed(relax);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 10, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 20, 1, 1);
        let queued: QueuedFireIds = [10, 20].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, Instant::now()));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![10, 20], "the fleet starts as one dense boundary");
        // Deliberately NOT retired: both lanes are now owed a result, which is
        // the state the relaxation is about. `b` answers anyway (its guest had
        // run-ahead); `a` has nothing queued and cannot get any.
        policy.on_fire_enqueued(stamp(b, 1, 0, 1), Some(b), 21, 1, 1);
        assert!(policy.owes_lane(a), "a's dispatched frame is unretired");
        (policy, a, b)
    }

    #[test]
    fn an_owed_lane_with_nothing_queued_does_not_deny_the_chain() {
        let (mut policy, _a, _b) = contributed_scenario(true);
        let queued: QueuedFireIds = [21].into_iter().collect();
        assert_eq!(
            fires(&policy.plan_dispatch(&queued, &HashSet::new(), true, Instant::now())),
            vec![21],
            "the gatherable fleet seals while the device still owes `a` a result"
        );
    }

    /// A debt the engine will never answer must not hide behind the
    /// exemption. The relaxation skips a lane BECAUSE the engine owes it a
    /// result, which is the same state the never-answered backstop exists to
    /// notice; composed naively the skip returns before the backstop is even
    /// consulted, and a lane whose command buffer died would be waited on in
    /// silence forever. Past the debt deadline the relaxation stands down and
    /// the lane is judged by the ordinary ladder.
    #[test]
    fn a_debt_past_its_deadline_stops_buying_the_exemption() {
        let (mut policy, a, b) = contributed_scenario(true);
        let queued: QueuedFireIds = [21].into_iter().collect();
        // Fresh debt: `a` is skipped and the rest of the fleet seals.
        assert_eq!(
            fires(&policy.plan_dispatch(&queued, &HashSet::new(), true, Instant::now())),
            vec![21],
            "a fresh debt buys the exemption"
        );
        policy.on_fire_enqueued(stamp(b, 2, 0, 1), Some(b), 22, 1, 1);
        // Same state, but the debt is now older than twice the silence
        // timeout: `a` is counted missing again, so the gate holds instead of
        // sealing without it.
        let far = Instant::now() + policy.silence_timeout * 2 + Duration::from_millis(1);
        let queued: QueuedFireIds = [22].into_iter().collect();
        assert!(
            !matches!(
                policy.plan_dispatch(&queued, &HashSet::new(), true, far),
                FramePlan::Dispatch(_)
            ),
            "a debt past its deadline no longer exempts the lane, so the fleet \
             does not seal without it -- the backstop judges it instead"
        );
        assert!(policy.owes_lane(a), "the debt is still outstanding");
    }

    /// The debt is what buys the exemption, so paying it takes the exemption
    /// back: once `on_frame_retired` clears the unretired dispatch, `a` is an
    /// ordinary awaited member and strict wait-all holds for it again. This is
    /// what bounds the relaxation — no lane can be skipped twice without the
    /// engine having answered it in between.
    #[test]
    fn retiring_the_debt_restores_wait_all_for_the_same_lane() {
        let (mut policy, a, b) = contributed_scenario(true);
        let queued: QueuedFireIds = [21].into_iter().collect();
        assert_eq!(
            fires(&policy.plan_dispatch(&queued, &HashSet::new(), true, Instant::now())),
            vec![21]
        );
        // `b` runs ahead once more so the gate is actually reached (a policy
        // with nothing queued anywhere parks before the census).
        policy.on_fire_enqueued(stamp(b, 2, 0, 1), Some(b), 22, 1, 1);
        retire(&mut policy, [a]);
        assert!(!policy.owes_lane(a), "the retirement paid a back");
        let queued: QueuedFireIds = [22].into_iter().collect();
        assert!(
            matches!(
                policy.plan_dispatch(&queued, &HashSet::new(), true, Instant::now()),
                FramePlan::Park
            ),
            "with nothing owed to it, `a` denies the boundary like any member"
        );
    }

    /// The relaxation is a CHAINING device, not a gathering rule: it applies
    /// only while a frame is executing. With the device idle there is no chain
    /// to protect and dense gathering is simply right, so the same owed-and-
    /// empty lane holds the gate.
    #[test]
    fn an_idle_device_still_gathers_densely() {
        let (mut policy, _a, _b) = contributed_scenario(true);
        let queued: QueuedFireIds = [21].into_iter().collect();
        assert!(
            matches!(
                policy.plan_dispatch(&queued, &HashSet::new(), false, Instant::now()),
                FramePlan::Hold(_)
            ),
            "with the device idle the gate waits for every awaited lane"
        );
    }

    /// `PIE_GATE_CONTRIBUTED=0` restores the strict rule exactly: same shape,
    /// same executing device, and the whole fleet parks on the owed lane.
    #[test]
    fn the_strict_rule_holds_the_fleet_for_an_owed_lane() {
        let (mut policy, _a, _b) = contributed_scenario(false);
        let queued: QueuedFireIds = [21].into_iter().collect();
        assert!(
            matches!(
                policy.plan_dispatch(&queued, &HashSet::new(), true, Instant::now()),
                FramePlan::Park
            ),
            "strict wait-all holds the whole fleet for the owed lane"
        );
    }

    /// A terminated lane's fires may be cancelled rather than retired, so
    /// `on_frame_retired` cannot be relied on to clear its debt. Left behind,
    /// the entry makes `owes` true forever for that lane id — and since `owes`
    /// is the whole of the relaxation AND what stops the silence clock, the
    /// lane would be permanently exempt from wait-all and permanently
    /// unleashable. The terminate path purges it with the lane.
    #[test]
    fn terminate_purges_the_engines_debt_to_the_lane() {
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        let a = pid();
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 10, 1, 1);
        let queued: QueuedFireIds = [10].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())), vec![10]);
        assert!(policy.owes_lane(a), "the dispatched frame is an unretired debt");

        // Exactly what the worker posts for a Terminate leave.
        policy.on_lane_leave(a, Some(a), true);
        policy.on_process_leave(a);
        assert!(
            !policy.owes_lane(a),
            "a terminated lane's debt cannot outlive the lane that incurred it"
        );
    }

    /// The mirror image: a SUSPEND is not a termination. An evicted process's
    /// already-dispatched frames still drain and retire — that is what releases
    /// the fire leases the eviction's quiescence wait blocks on — so the debt
    /// is real and must survive. Clearing it here would start the silence clock
    /// against a process the engine genuinely still owes.
    #[test]
    fn suspend_keeps_the_engines_debt_to_the_lane() {
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        let a = pid();
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 10, 1, 1);
        let queued: QueuedFireIds = [10].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())), vec![10]);

        policy.on_process_suspend(a);
        assert!(
            policy.owes_lane(a),
            "an evicted process's dispatched frame still retires; the debt stands"
        );
        retire(&mut policy, [a]);
        assert!(!policy.owes_lane(a), "and the retirement clears it normally");
    }
}
