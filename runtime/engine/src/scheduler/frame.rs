//! Vesuvius frame scheduling — THE scheduler policy: the
//! **wait-for-all-active-lanes** quorum rule at frame granularity. Wait
//! until every awaited lane's next FRAME is fully submitted, then seal the
//! dense epoch and dispatch its k waves in slot order.
//!
//! Every deployment routes here, including `PIE_FRAME_SIZE=1`:
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

/// Grace period before a gather blocked EXCLUSIVELY on empty lanes whose
/// owners hold an in-flight bind releases them (see the `missing` predicate
/// in [`FramePolicy::plan_dispatch`]). Only consulted in escape mode 2.
const REBIND_ESCAPE_US: u64 = 2_000;

/// A/B toggle for the rebind escape (CONTENTION_FOLLOWUP §20.3):
/// `0` = never escape (pre-fix; deadlocks), `1` = escape unconditionally
/// (correct, but costs epoch density — measured -16% on a roomy
/// decode-heavy fleet), `2` = escape only after [`REBIND_ESCAPE_US`] with
/// nothing executing and every missing lane an empty rebinder. `2` is the
/// default: it keeps the dense wait-all path for healthy gathers (which
/// resolve in microseconds) and only releases the deadlock shape itself.
fn rebind_escape_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("PIE_FRAME_REBIND_ESCAPE")
            .ok()
            .and_then(|value| value.parse::<u8>().ok())
            .unwrap_or(2)
    })
}

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
    /// Deadline for the escape-mode-2 rebind grace period.
    rebind_escape_deadline: Option<Instant>,
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
            lanes: BTreeMap::new(),
            sealed: VecDeque::new(),
            pending_binds: BTreeMap::new(),
            staged: BTreeSet::new(),
            pending_slots: 0,
            joins_in_flight: BTreeSet::new(),
            slotted: BTreeSet::new(),
            departing: BTreeSet::new(),
            suspended: BTreeSet::new(),
            truncated_seqs: BTreeMap::new(),
            strict_watchdog_deadline: None,
            rebind_escape_deadline: None,
            cold_hold_deadline: None,
            ever_sealed: false,
            stats,
        }
    }

    /// Whether this deployment runs 1-slot frames (`PIE_FRAME_SIZE=1`):
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
                    truncated: false,
                    fires: Vec::with_capacity(stamp.fires as usize),
                });
                lane.frames.back_mut().expect("frame just pushed")
            }
        };
        frame.truncated |= late || suspended;
        frame.fires.push(fire);
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

    /// A bind control entered the scheduler. A bind does not hold the seal:
    /// a live rebinder is already wait-set-held through its lane, and a
    /// bring-up process (no lane yet) enters the `staged` successor pool —
    /// the seal waits for it only once a slot opens for it
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
            // holding. A recurrent-state pass is exactly that — it
            // serializes behind every earlier fire — so a guest that
            // binds a second program after its first fire (prefill then
            // decode, the ordinary shape) deadlocked the whole gather
            // whenever the execution pool was capped.
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
    /// a successor is staged, the seal holds — the successor's admission and
    /// first fire are imminent. Resolves the holder's departure by identity
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
        self.slotted.insert(pid);
        if self.staged.contains(&pid) {
            self.joins_in_flight.insert(pid);
        }
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
        } else if let Some(state) = self.lanes.get_mut(&lane) {
            state.awaited = false;
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
        self.rebind_escape_deadline = None;
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
            let _ = &fire_waves;
            self.sealed.push_back(SealedFrame {
                waves,
                members: members.iter().copied().collect(),
            });
        }
        self.lanes
            .retain(|_, lane| lane.awaited || !lane.frames.is_empty());
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
            if !self.lanes.values().any(|lane| !lane.frames.is_empty()) {
                // Nothing queued anywhere: no gather episode is running.
                self.strict_watchdog_deadline = None;
                self.rebind_escape_deadline = None;
                return FramePlan::Park;
            }
            // `missing` counts awaited lanes whose next frame is not fully
            // submitted; `missing_rebind` is the subset that is EMPTY and
            // whose owner holds an in-flight bind.
            //
            // Such a lane is not a member that is about to submit: its next
            // fire is ordered behind the bind, and a bind completes through
            // the driver lane's control slot — behind the dispatch this
            // boundary is holding. Waiting for it closes the cycle
            // `bind -> dispatch -> seal -> bind`, which wedges the whole
            // fleet permanently once the KV pool is oversubscribed enough
            // that the boundary has nothing else executing to re-decide it
            // (CONTENTION_FOLLOWUP §20.3: `awaited=24` with
            // `pending_binds=8` and eight zero-frame lanes,
            // `joins_in_flight=0`).
            //
            // This is the same hazard `on_bind_enqueued` avoids for
            // successors it declines to re-stage; reached here through the
            // lane's own wait-set membership instead. Restricting it to
            // EMPTY lanes is what keeps the epoch dense: a lane with queued
            // frames is genuinely mid-submission and is still waited for.
            // Rejoin is implicit — the lane's next accepted fire restores it
            // to the quorum.
            let mut missing = 0usize;
            let mut missing_rebind = 0usize;
            let mut missing_idle = 0usize;
            for lane in self.lanes.values() {
                if !lane.awaited {
                    continue;
                }
                if lane.frames.front().is_some_and(PendingFrame::is_complete) {
                    continue;
                }
                missing += 1;
                if lane.frames.is_empty() {
                    missing_idle += 1;
                    if lane
                        .owner
                        .is_some_and(|owner| self.pending_binds.contains_key(&owner))
                    {
                        missing_rebind += 1;
                    }
                }
            }
            let joining = !self.joins_in_flight.is_empty()
                || ((self.pending_slots > 0 || !self.departing.is_empty())
                    && !self.staged.is_empty());
            let mode = rebind_escape_mode();
            // Mode 2 escapes only from the deadlock shape itself: every missing
            // lane is EMPTY and nothing is executing, so nothing the engine
            // controls will ever make one of them submit -- their next fire is
            // ordered behind a result only this seal can produce, which is the
            // cycle `seal -> result -> submit -> seal`. An empty rebinder is one
            // way to land there (its fire is ordered behind a bind that needs
            // the control slot this boundary holds); a lane simply idle between
            // frames is another, and it is the shape two concurrent request
            // streams hit as soon as one of them drains its run-ahead window
            // while the other still has work. Escaping cannot reorder anything
            // that was going to resolve, because nothing is executing. The grace
            // period keeps a healthy gather (which resolves in microseconds) on
            // the dense wait-all path.
            let escaping = missing_idle > 0
                && match mode {
                    0 => false,
                    1 => missing_rebind > 0,
                    _ => {
                        !joining && !executing && missing == missing_idle && {
                            let deadline = *self
                                .rebind_escape_deadline
                                .get_or_insert(now + Duration::from_micros(REBIND_ESCAPE_US));
                            now >= deadline
                        }
                    }
                };
            let escaped = if !escaping {
                0
            } else if mode == 1 {
                missing_rebind
            } else {
                missing_idle
            };
            if !escaping && mode == 2 && (missing != missing_idle || executing || joining) {
                self.rebind_escape_deadline = None;
            }
            let missing = missing - escaped;
            if joining || missing > 0 {
                let mut stalled = false;
                if executing {
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
                    stalled = true;
                }
                let plan = FramePlan::Hold(deadline.saturating_duration_since(now));
                if stalled && crate::planner::trace_enabled() {
                    println!("[frame-stall] {}", self.debug_summary());
                }
                return plan;
            }
            self.strict_watchdog_deadline = None;
            self.rebind_escape_deadline = None;
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
            "frame k={} lanes={} awaited={} sealed={} pending_binds={} \
staged={} joins_in_flight={} departing={} suspended={} pending_slots={} \
ever_sealed={} watchdog={:?}",
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
            self.ever_sealed,
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

    fn drive_past_cold_hold(policy: &mut FramePolicy, queued: &QueuedFireIds) -> FramePlan {
        let now = Instant::now();
        match plan(policy, queued, now) {
            FramePlan::Hold(hold) => plan(policy, queued, now + hold + Duration::from_micros(1)),
            plan => plan,
        }
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
        let queued: QueuedFireIds = [7].into_iter().collect();
        assert_eq!(
            plan(&mut policy, &queued, Instant::now()),
            FramePlan::Dispatch(vec![vec![7]]),
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
        let queued: QueuedFireIds = [10, 11].into_iter().collect();
        let t0 = Instant::now();
        let FramePlan::Hold(hold) = plan(&mut policy, &queued, t0) else {
            panic!("bootstrap membership is forming: the cold hold must arm");
        };
        let sealed = plan(&mut policy, &queued, t0 + hold + Duration::from_micros(1));
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
        // `b`'s next fire arrives: the wave seals dense, no cold hold.
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
        let sealed = drive_past_cold_hold(&mut policy, &queued);
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
    #[test]
    fn incomplete_lane_holds_the_seal_until_it_completes() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (fast, slow) = (pid(), pid());
        policy.on_fire_enqueued(stamp(fast, 0, 0, 2), Some(fast), 1, 1, 1);
        policy.on_fire_enqueued(stamp(fast, 0, 1, 2), Some(fast), 2, 1, 1);
        // `slow` declared 2 fires but only one arrived: a missing member.
        policy.on_fire_enqueued(stamp(slow, 0, 0, 2), Some(slow), 3, 1, 1);

        let queued: QueuedFireIds = [1, 2, 3].into_iter().collect();
        let t0 = Instant::now();
        match plan(&mut policy, &queued, t0) {
            FramePlan::Hold(hold) => {
                assert_eq!(hold, Duration::from_micros(STRICT_WATCHDOG_US));
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
        let FramePlan::Dispatch(waves) = drive_past_cold_hold(&mut policy, &queued) else {
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
        let FramePlan::Dispatch(frame0) = drive_past_cold_hold(&mut policy, &queued) else {
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
    fn dropped_fires_resolve_and_leave_rearms_the_gather() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 20, 1, 1);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 2), Some(lane), 21, 1, 1);
        let queued: QueuedFireIds = [20, 21].into_iter().collect();
        let FramePlan::Dispatch(waves) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("expected the whole frame");
        };
        assert_eq!(waves, vec![vec![20], vec![21]]);
        let queued = QueuedFireIds::default();
        assert_eq!(plan(&mut policy, &queued, Instant::now()), FramePlan::Park);
        assert_eq!(policy.sealed.len(), 0, "frame popped at dispatch");
        assert!(
            policy.ever_sealed,
            "the wait-set persists while the lane is awaited — drained \
             books alone do not re-arm the gather"
        );
        // Only the lane's LEAVE empties the wait-set and re-arms bootstrap.
        policy.on_lane_leave(lane, None, false);
        assert!(
            !policy.ever_sealed,
            "an emptied wait-set re-arms the gather"
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
        let queued: HashSet<u64> = [100, 101, 102, 103].into_iter().collect();
        assert!(matches!(
            drive_past_cold_hold(&mut policy, &queued),
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

        let queued: HashSet<u64> = [200, 300, 301].into_iter().collect();
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
        let queued: HashSet<u64> = [201, 302, 303].into_iter().collect();
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
        let queued: HashSet<u64> = [10].into_iter().collect();
        let FramePlan::Dispatch(waves) = drive_past_cold_hold(&mut policy, &queued) else {
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
        let FramePlan::Dispatch(waves) = drive_past_cold_hold(&mut policy, &queued) else {
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
        let FramePlan::Dispatch(frame_a) = drive_past_cold_hold(&mut policy, &queued) else {
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
        let FramePlan::Dispatch(frame_a) = drive_past_cold_hold(&mut policy, &queued) else {
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
        let bootstrap = drive_past_cold_hold(&mut policy, &queued);
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
        let sealed = drive_past_cold_hold(&mut policy, &queued);
        assert_eq!(fires(&sealed), vec![95]);
    }

    /// REGRESSION (the rebind-seal wedge, CONTENTION_FOLLOWUP.md §20).
    /// A process whose lane has drained to empty and that has a bind in
    /// flight must not hold the boundary: its next fire is ordered behind
    /// the bind, and the bind completes through the driver lane's control
    /// slot — behind the very dispatch the seal is holding. Waiting for it
    /// closes `bind -> dispatch -> seal -> bind` and wedges the fleet
    /// permanently (observed at 32x KV oversubscription: `awaited=24`,
    /// `pending_binds=8`, eight zero-frame lanes, `joins_in_flight=0`,
    /// nothing executing, ~10% of runs).
    #[test]
    fn an_empty_lane_awaiting_its_own_rebind_does_not_hold_the_seal() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let runner = pid();
        let rebinder = pid();

        // Both lanes fire once and the epoch dispatches.
        policy.on_fire_enqueued(stamp(runner, 0, 0, 1), Some(runner), 10, 1, 1);
        policy.on_fire_enqueued(stamp(rebinder, 0, 0, 1), Some(rebinder), 11, 1, 1);
        let queued: HashSet<u64> = [10, 11].into_iter().collect();
        let mut wave0 = fires(&drive_past_cold_hold(&mut policy, &queued));
        wave0.sort_unstable();
        assert_eq!(wave0, vec![10, 11]);

        // The runner submits its next frame. The rebinder's lane drained
        // empty and it is waiting on a bind (prefill -> decode, the
        // ordinary shape), so it cannot submit until the bind commits.
        policy.on_fire_enqueued(stamp(runner, 1, 0, 1), Some(runner), 12, 1, 1);
        policy.on_bind_enqueued(Some(rebinder));

        let queued: HashSet<u64> = [12].into_iter().collect();
        let sealed = drive_past_cold_hold(&mut policy, &queued);
        assert_eq!(
            fires(&sealed),
            vec![12],
            "the boundary must seal without the rebinder, whose fire cannot \
             arrive until this dispatch releases the control slot"
        );

        // The exclusion is scoped to the bind, not permanent: once it
        // commits the lane is a full member again and the boundary waits
        // for it exactly as before.
        policy.on_bind_completed(Some(rebinder));
        policy.on_fire_enqueued(stamp(runner, 2, 0, 1), Some(runner), 14, 1, 1);
        let queued: HashSet<u64> = [14].into_iter().collect();
        match drive_past_cold_hold(&mut policy, &queued) {
            FramePlan::Hold(_) => {}
            plan => panic!("the committed rebinder must hold the seal, got {plan:?}"),
        }
        policy.on_fire_enqueued(stamp(rebinder, 1, 0, 1), Some(rebinder), 13, 1, 1);
        let queued: HashSet<u64> = [13, 14].into_iter().collect();
        let mut wave = fires(&drive_past_cold_hold(&mut policy, &queued));
        wave.sort_unstable();
        assert_eq!(wave, vec![13, 14], "the rebinder gathered back in");
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
        let queued: QueuedFireIds = [95].into_iter().collect();
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
        let queued: QueuedFireIds = [95, 96].into_iter().collect();
        let sealed = drive_past_cold_hold(&mut policy, &queued);
        let mut wave0 = fires(&sealed);
        wave0.sort_unstable();
        assert_eq!(wave0, vec![95, 96], "both lanes gathered into one epoch");
    }

    /// REGRESSION (the frame-seal wedge). A joined-but-unfired successor that
    /// parks in KV allocation must release the seal — and it has NO lane yet,
    /// so the only thing that identifies it is the owner travelling with the
    /// leave. `lanes` is keyed by pipeline scope while `staged` /
    /// `joins_in_flight` are keyed by process; cleaning the latter with the
    /// lane id silently matched nothing, so the gather waited forever for a
    /// join that could not arrive while the fleet held every KV page.
    #[test]
    fn a_parked_join_releases_the_seal_even_though_it_has_no_lane() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        let successor = pid();
        let successor_scope = pid(); // distinct id space from the process
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 95, 1, 1);
        policy.on_bind_enqueued(Some(successor));
        policy.on_bind_completed(Some(successor));
        policy.on_execution_slot_released(pid());
        policy.on_execution_slot_consumed(successor);
        let queued: QueuedFireIds = [95].into_iter().collect();
        match drive_past_cold_hold(&mut policy, &queued) {
            FramePlan::Hold(_) => {}
            plan => panic!("an admitted-but-unfired join must hold, got {plan:?}"),
        }

        // It blocks on KV before ever firing: leave carries the scope id it
        // waits under AND the owning process.
        policy.on_lane_leave(successor_scope, Some(successor), false);
        let sealed = drive_past_cold_hold(&mut policy, &queued);
        assert_eq!(
            fires(&sealed),
            vec![95],
            "the fleet must seal without the parked join"
        );
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
        policy.on_lane_leave(predecessor, None, true);
        policy.on_process_leave(predecessor);
        let queued: QueuedFireIds = [95].into_iter().collect();
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
        let queued: QueuedFireIds = [95, 96].into_iter().collect();
        let sealed = drive_past_cold_hold(&mut policy, &queued);
        let mut wave = fires(&sealed);
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
        let queued: QueuedFireIds = [95, 96].into_iter().collect();
        assert!(
            matches!(
                drive_past_cold_hold(&mut policy, &queued),
                FramePlan::Dispatch(_)
            ),
            "a retired departure must leave no phantom hold"
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
        let queued: QueuedFireIds = [10].into_iter().collect();
        match drive_past_cold_hold(&mut policy, &queued) {
            FramePlan::Hold(_) => {}
            plan => panic!("free slot with staged taker must gather, got {plan:?}"),
        }
        policy.on_execution_slot_consumed(b);
        policy.on_fire_enqueued(stamp(b, 11, 0, 1), Some(b), 11, 1, 1);
        let queued: QueuedFireIds = [10, 11].into_iter().collect();
        let sealed = drive_past_cold_hold(&mut policy, &queued);
        let mut wave = fires(&sealed);
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
        let queued: QueuedFireIds = [95].into_iter().collect();
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
        let queued: QueuedFireIds = [95].into_iter().collect();

        // Slot released with an empty staged pool: no earmark, no hold.
        policy.on_execution_slot_released(pid());
        let sealed = drive_past_cold_hold(&mut policy, &queued);
        assert_eq!(fires(&sealed), vec![95]);

        // Stage a successor (freed slot pending from above), admit it,
        // then let it die before firing: the leave releases the hold.
        policy.on_fire_enqueued(stamp(lane, 1, 0, 1), Some(lane), 96, 1, 1);
        policy.on_bind_enqueued(Some(successor));
        let queued: QueuedFireIds = [96].into_iter().collect();
        match plan(&mut policy, &queued, Instant::now()) {
            FramePlan::Hold(_) | FramePlan::Park => {}
            plan => panic!("freed slot with staged taker must hold, got {plan:?}"),
        }
        policy.on_execution_slot_consumed(successor);
        match plan(&mut policy, &queued, Instant::now()) {
            FramePlan::Hold(_) | FramePlan::Park => {}
            plan => panic!("join in flight must hold, got {plan:?}"),
        }
        policy.on_process_leave(successor);
        assert!(matches!(
            plan(&mut policy, &queued, Instant::now()),
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

        policy.on_lane_leave(closed, None, false);
        policy.on_lane_leave(terminated, None, true);
        let queued: QueuedFireIds = [50].into_iter().collect();
        let sealed = drive_past_cold_hold(&mut policy, &queued);
        assert_eq!(fires(&sealed), vec![50]);
    }
}
