use crate::rt::Instant;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::stats::SchedulerStats;
use crate::engine::WorkItemCompletion;
use crate::scheduler::ProcessId;

const DEFAULT_DISPATCH_DEPTH: usize = 2;

pub(super) fn configured_dispatch_depth() -> usize {
    match DISPATCH_DEPTH.load(Ordering::Relaxed) {
        0 => DEFAULT_DISPATCH_DEPTH,
        depth => depth,
    }
}

pub(crate) fn set_dispatch_depth(depth: usize) {
    DISPATCH_DEPTH.store(depth, Ordering::Relaxed);
}

static DISPATCH_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// The next boot states its own depth.
pub(crate) fn reset() {
    DISPATCH_DEPTH.store(0, Ordering::Relaxed);
}

/// How long a frame held for lanes that have not submitted waits before it
/// says which, and again after each such span.
const HOLD_SAYS_AFTER: Duration = Duration::from_secs(5);

/// Whether a seal says what it sealed: the scheduler's debug level. Off,
/// nothing on the hot path reads a fire's program.
pub(crate) fn seal_trace() -> bool {
    tracing::enabled!(tracing::Level::DEBUG)
}

/// Says, on the seal trace, why `lane` stops being awaited: which
/// mechanism took it out of the gate.
fn trace_unawait(cause: &str, lane: ProcessId, owner: Option<ProcessId>) {
    if !seal_trace() {
        return;
    }
    let line = format!("unawait cause={cause} lane={lane} owner={owner:?}");
    tracing::debug!("{line}");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStamp {
    pub lane: ProcessId,
    pub seq: u64,
    pub slot: u32,
    pub fires: u32,
}

const STRICT_WATCHDOG_US: u64 = 1_000_000;

const GATHER_POLL_US: u64 = 500;
const JOIN_HOLD_US: u64 = 2_000;

struct ArrivedFire {
    slot: u32,
    fire_id: Option<u64>,
    tokens: usize,
    rows: usize,
    readouts: usize,
    /// What the fire takes of its activation rung
    /// ([`crate::engine::FireRequest::rung_rows`]).
    rung_rows: usize,
    /// The program its instance runs, as the seal trace names it; 0 where
    /// the trace is off ([`seal_trace`]).
    program: u64,
}

struct PlacedFire {
    wave: usize,
    fire_id: u64,
    rows: usize,
    tokens: usize,
    readouts: usize,
}

struct PendingFrame {
    seq: u64,
    expected: u32,
    truncated: bool,
    park: bool,
    fires: Vec<ArrivedFire>,
    complete_at: Option<Instant>,
}

impl PendingFrame {
    fn is_complete(&self) -> bool {
        self.park || self.fires.len() >= self.expected as usize
    }
}

struct LaneState {
    owner: Option<ProcessId>,
    awaited: bool,
    served: bool,
    retired_at: Option<Instant>,
    parked: bool,
    frames: VecDeque<PendingFrame>,
    fired_this_boundary: bool,
    /// Its last sealed frame carried a prompt: its next pass is built on the
    /// host from what that prompt answers, so only decodes wait for it.
    prompting: bool,
    /// It comes back only by a submit of its own: its pipeline closed
    /// ([`FramePolicy::on_lane_leave`]), its guest parked it
    /// (`forward.park()`, [`FramePolicy::on_lane_park`]), or its guest waits
    /// on outside input ([`FramePolicy::on_owner_away`]). Kept only while frames it submitted
    /// wait to go, and never expected again by its owner's unblock or
    /// resume. Kept parked as any other, an unblock would expect it again,
    /// and a frame would wait for good on a lane that never submits (a
    /// verified beam's rescoring pipelines, closed mid-run).
    own_submit: bool,
    /// The fire whose output its process waits on, having submitted
    /// nothing since ([`FramePolicy::on_lane_waits`]): until that fire
    /// settles the lane submits nothing, so no frame waits on it.
    waits_on: Option<WorkItemCompletion>,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateVerdict {
    Satisfied,
    Blocking,
    /// It waits on its own output from a fire not yet settled: it can
    /// submit nothing before that frame lands, so the frame goes without
    /// it and it joins the next one it submits for.
    Waits,
}

impl LaneState {
    /// Settlement is read here, at each plan: a lane whose awaited fire
    /// settled holds the frame again until it submits.
    fn gate_verdict(&self) -> GateVerdict {
        if !self.awaited || self.frames.front().is_some_and(PendingFrame::is_complete) {
            return GateVerdict::Satisfied;
        }
        let waits = self
            .waits_on
            .as_ref()
            .is_some_and(|completion| !completion.is_settled());
        if waits && self.frames.is_empty() {
            return GateVerdict::Waits;
        }
        GateVerdict::Blocking
    }
}

struct SealedFrame {
    waves: Vec<Vec<u64>>,
    members: BTreeSet<ProcessId>,
    /// The widest of its fires' rung rows: a frame sealed under the cap
    /// goes as sealed, and the rung is not lowered under it until it has.
    widest: usize,
    /// Its members whose fires in it are a prompt's.
    prompts: BTreeSet<ProcessId>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FramePlan {
    Dispatch(Vec<Vec<u64>>),
    Hold(Duration),
    Park,
}

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

    pub fn seal(&mut self) {
        if !self.ids.windows(2).all(|pair| pair[0] <= pair[1]) {
            self.ids.sort_unstable();
        }
    }

    pub fn contains(&self, fire_id: &u64) -> bool {
        self.ids.binary_search(fire_id).is_ok()
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

/// Ready lanes a seal passed over, by why ([`FramePolicy::seal`]).
#[derive(Debug, Clone, Copy, Default)]
struct SealSkips {
    /// A fire cut under a cap since lowered.
    capped: usize,
    /// Past the wave's rows, tokens or readouts.
    unfit: usize,
    /// A prompt behind another in flight.
    prompt_held: usize,
}

pub(crate) struct FramePolicy {
    k: usize,
    max_wave_tokens: usize,
    max_wave_rows: usize,
    /// The rows a wave may read out, the engine's `max_readouts`; 0 bounds
    /// nothing.
    max_wave_readouts: usize,
    lanes: BTreeMap<ProcessId, LaneState>,
    sealed: VecDeque<SealedFrame>,
    executing_now: bool,
    last_retire_at: Option<Instant>,
    pending_binds: BTreeMap<ProcessId, usize>,
    staged: BTreeSet<ProcessId>,
    /// The seal under way runs while lanes are coming off a prompt: no
    /// prompt waits for them, and the decodes ready beside them ride along.
    prompts_only: bool,
    pending_slots: u64,
    joins_in_flight: BTreeSet<ProcessId>,
    admission_queue: VecDeque<ProcessId>,
    slotted: BTreeSet<ProcessId>,
    departing: BTreeSet<ProcessId>,
    suspended: BTreeSet<ProcessId>,
    /// Lanes whose pipeline closed, by lane, with their owner: kept until
    /// the owner leaves, so a fire that raced the close rides the next seal
    /// as a straggler and never makes the lane expected again (a closed
    /// lane re-expected by such a fire would hold every later frame for
    /// good).
    closed: BTreeMap<ProcessId, Option<ProcessId>>,
    /// Lanes a frame sealed past while their guests computed and the device
    /// would otherwise have gone idle: each rejoins with its next submit.
    /// Named in the stall dump; cleared by that submit.
    sealed_past: BTreeSet<ProcessId>,
    /// Owners blocked on the planner or a store ([`Self::on_owner_waits`])
    /// until their wait ends ([`Self::on_owner_unblocked`]).
    blocked: BTreeSet<ProcessId>,
    /// Why the last seal that sealed nothing passed over each ready lane.
    seal_skips: SealSkips,
    /// When an unsealed complete frame with every expected lane in was last
    /// named.
    unsealed_said: Option<Instant>,
    truncated_seqs: BTreeMap<ProcessId, u64>,
    strict_watchdog_deadline: Option<Instant>,
    /// When a held frame last said which lanes it waits on.
    hold_said: Option<Instant>,
    /// The frame to seal next has waited on a lane that had not submitted
    /// (counted once a frame: [`super::probe::FireProbes`]'s
    /// `frames_parked_on_lane`).
    parked_on_lane: bool,
    join_hold: Option<Instant>,
    gather_seq: u64,
    quiesce_mark: Option<u64>,
    /// The lanes with a fire in a posted frame, exactly
    /// ([`Self::on_frame_posted`], [`Self::on_frame_retired`]).
    in_flight_lanes: BTreeSet<ProcessId>,
    /// Lanes whose prompt is executing on the engine now: posted
    /// ([`Self::on_frame_posted`]) from a frame that sealed it as a prompt,
    /// until it lands or fails ([`Self::on_frame_retired`]), its lane
    /// leaves, or its owner is suspended or leaves. Only these hold a
    /// prompt behind them: a prompt recorded in flight that no landing ever
    /// clears held every later prompt for good.
    prompts_executing: BTreeSet<ProcessId>,
    /// The prompts of the frame last handed out to post
    /// ([`FramePlan::Dispatch`]): executing once posted.
    dispatched_prompts: BTreeSet<ProcessId>,
    in_flight_since: BTreeMap<ProcessId, Instant>,
    /// The rung rows a fire sealed now may take: the activation rung's cap
    /// ([`Self::cap_fires`]).
    fire_cap: usize,
    stats: Option<Arc<SchedulerStats>>,
    /// Each pipeline's last fire posted, by its submit order (the logical
    /// fire id its pass took at submit): a fire posted behind a later one of
    /// its pipeline is refused by name ([`Self::admit_post`]).
    posted_through: HashMap<ProcessId, u64>,
}

/// Why a fire posted behind a later one of its pipeline is refused.
pub(crate) fn out_of_order(pipeline: ProcessId, fire: u64, later: u64) -> String {
    format!(
        "scheduler: fire {fire} of pipeline {pipeline} would post after its later fire {later}: \
         a pipeline's fires reach the device in the order it submitted them"
    )
}

impl FramePolicy {
    /// Admits `fire` of `pipeline` to the device if no later fire of the
    /// pipeline has posted before it; else names that later fire.
    pub fn admit_post(&mut self, pipeline: ProcessId, fire: u64) -> Result<(), u64> {
        match self.posted_through.get(&pipeline) {
            Some(&later) if later > fire => Err(later),
            _ => {
                self.posted_through.insert(pipeline, fire);
                Ok(())
            }
        }
    }


    /// Waves read out no more than `readouts` rows (the engine refuses a
    /// fire past them), a fire that alone reads more still taking a wave of
    /// its own.
    pub fn with_max_wave_readouts(mut self, readouts: usize) -> Self {
        self.max_wave_readouts = readouts;
        self
    }

    /// The worker policy's bound on a batch of passes, as it stands now;
    /// none leaves a wave the engine's one fire's tokens.
    /// Fires are cut to `rows` of the activation rung now: a lane whose
    /// next frame holds a wider fire (cut under a cap since lowered) is
    /// left out of the frames sealed from here, its frame kept whole until
    /// the cap is back; every fire of a frame sealed goes.
    pub fn cap_fires(&mut self, rows: usize) {
        self.fire_cap = rows;
    }

    /// The widest rung rows of a ready frame held out under the cap: fires
    /// cut under a wider cap before it came down, kept whole, which only a
    /// raise lets go. What the cap's owner is told
    /// ([`crate::store::shared::owe_cut_rows`]) so a hold is never on a
    /// raise nothing asks for. 0 where none is held.
    #[cfg(test)]
    #[must_use]
    pub fn cut_owed(&self) -> usize {
        self.cut_owed_by().0
    }

    /// [`Self::cut_owed`], and the process whose frame holds that widest
    /// fire: the lane's owner, whom a raise to it is funded for (the lane
    /// where it names none). Named by its lane, the raise would ask the
    /// planner for a process it does not know, be told "planner request
    /// cancelled" every pass, and the capped frames would sit for good.
    #[must_use]
    pub fn cut_owed_by(&self) -> (usize, Option<ProcessId>) {
        self.lanes
            .iter()
            .filter_map(|(&lane_id, lane)| {
                let front = lane.frames.front().filter(|front| front.is_complete())?;
                let widest = front
                    .fires
                    .iter()
                    .map(|fire| fire.rung_rows)
                    .filter(|&rows| rows > self.fire_cap)
                    .max()?;
                Some((widest, lane.owner.unwrap_or(lane_id)))
            })
            .max_by_key(|&(rows, _)| rows)
            .map_or((0, None), |(rows, pid)| (rows, Some(pid)))
    }

    /// A raise to the frames held out under the cap for `owner` refused by
    /// name (it never fits beside the owner's own state,
    /// [`crate::store::shared::refuse_owed`]): its lanes' next frames that
    /// hold a fire wider than the cap leave, and their fires are returned,
    /// for the scheduler to fail by that name rather than hold for good.
    pub fn refuse_held(&mut self, owner: ProcessId) -> Vec<u64> {
        let cap = self.fire_cap;
        let mut fires = Vec::new();
        for (&lane_id, lane) in &mut self.lanes {
            if lane.owner.unwrap_or(lane_id) != owner {
                continue;
            }
            let held = lane.frames.front().is_some_and(|front| {
                front.is_complete() && front.fires.iter().any(|fire| fire.rung_rows > cap)
            });
            if held && let Some(frame) = lane.frames.pop_front() {
                fires.extend(frame.fires.iter().filter_map(|fire| fire.fire_id));
            }
        }
        fires
    }

    /// The widest rung rows any fire queued in a lane's frames takes,
    /// complete or still gathering: the rung is not lowered under them
    /// (a frame still gathering when the lower is granted can complete
    /// wider than the cap, capped, and wait on a raise).
    #[must_use]
    pub fn queued_widest(&self) -> usize {
        self.lanes
            .values()
            .flat_map(|lane| lane.frames.iter())
            .flat_map(|frame| frame.fires.iter())
            .map(|fire| fire.rung_rows)
            .max()
            .unwrap_or(0)
    }

    /// The widest rung rows a sealed frame still to go takes; 0 where none
    /// waits. The rung is not lowered under it until it has gone.
    #[must_use]
    pub fn sealed_widest(&self) -> usize {
        self.sealed
            .iter()
            .map(|frame| frame.widest)
            .max()
            .unwrap_or(0)
    }

    /// A frame was posted to the engine with fires of `lanes`: they owe its
    /// output until it retires ([`Self::on_frame_retired`]).
    pub fn on_frame_posted(&mut self, lanes: impl IntoIterator<Item = ProcessId>, now: Instant) {
        let prompts = std::mem::take(&mut self.dispatched_prompts);
        for lane in lanes {
            self.in_flight_lanes.insert(lane);
            self.in_flight_since.entry(lane).or_insert(now);
            if prompts.contains(&lane) {
                self.prompts_executing.insert(lane);
            }
        }
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
            max_wave_readouts: 0,
            lanes: BTreeMap::new(),
            sealed: VecDeque::new(),
            executing_now: false,
            last_retire_at: None,
            pending_binds: BTreeMap::new(),
            staged: BTreeSet::new(),
            prompts_only: false,
            pending_slots: 0,
            joins_in_flight: BTreeSet::new(),
            admission_queue: VecDeque::new(),
            slotted: BTreeSet::new(),
            departing: BTreeSet::new(),
            suspended: BTreeSet::new(),
            closed: BTreeMap::new(),
            sealed_past: BTreeSet::new(),
            blocked: BTreeSet::new(),
            seal_skips: SealSkips::default(),
            unsealed_said: None,
            truncated_seqs: BTreeMap::new(),
            strict_watchdog_deadline: None,
            hold_said: None,
            parked_on_lane: false,
            join_hold: None,
            gather_seq: 0,
            quiesce_mark: None,
            in_flight_lanes: BTreeSet::new(),
            prompts_executing: BTreeSet::new(),
            dispatched_prompts: BTreeSet::new(),
            fire_cap: usize::MAX,
            in_flight_since: BTreeMap::new(),
            stats,
            posted_through: HashMap::new(),
        }
    }

    pub fn single_slot(&self) -> bool {
        self.k == 1
    }

    /// [`Self::on_fire_reading`] for a fire that reads out one row a lane.
    #[cfg(test)]
    pub fn on_fire_enqueued(
        &mut self,
        stamp: FrameStamp,
        owner: Option<ProcessId>,
        fire_id: u64,
        tokens: usize,
        rows: usize,
    ) {
        self.on_fire_reading(stamp, owner, fire_id, tokens, rows, rows, tokens, 0);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn on_fire_reading(
        &mut self,
        stamp: FrameStamp,
        owner: Option<ProcessId>,
        fire_id: u64,
        tokens: usize,
        rows: usize,
        readouts: usize,
        rung_rows: usize,
        program: u64,
    ) {
        let accept_began = self.stats.is_some().then(Instant::now);
        self.record_arrival(
            stamp,
            owner,
            ArrivedFire {
                slot: stamp.slot,
                fire_id: Some(fire_id),
                tokens,
                rows,
                readouts,
                rung_rows,
                program,
            },
        );
        if let (Some(began), Some(stats)) = (accept_began, &self.stats) {
            use std::sync::atomic::Ordering::Relaxed;
            stats
                .fire
                .quorum
                .accept_us
                .fetch_add(began.elapsed().as_micros() as u64, Relaxed);
            stats.fire.quorum.accept_calls.fetch_add(1, Relaxed);
        }
    }

    pub fn on_fire_rejected_at_admission(&mut self, stamp: FrameStamp, owner: Option<ProcessId>) {
        self.record_arrival(
            stamp,
            owner,
            ArrivedFire {
                slot: stamp.slot,
                fire_id: None,
                tokens: 0,
                rows: 0,
                readouts: 0,
                rung_rows: 0,
                program: 0,
            },
        );
    }

    fn record_arrival(&mut self, stamp: FrameStamp, owner: Option<ProcessId>, fire: ArrivedFire) {
        if let Some(owner) = owner {
            self.staged.remove(&owner);
            self.joins_in_flight.remove(&owner);
        }
        let lane_owner = owner.or_else(|| self.lanes.get(&stamp.lane).and_then(|lane| lane.owner));
        // An owner suspended or blocked is not waited for, nor a lane whose
        // pipeline closed: a fire of theirs that raced the park or the close
        // rides the next seal, the lane not expected.
        let closed = self.closed.contains_key(&stamp.lane);
        let suspended = closed
            || lane_owner.is_some_and(|owner| {
                self.suspended.contains(&owner) || self.blocked.contains(&owner)
            });
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
            served: false,
            retired_at: None,
            parked: suspended,
            frames: VecDeque::new(),
            fired_this_boundary: true,
            prompting: false,
            own_submit: closed,
            waits_on: None,
        });
        if lane.owner.is_none() {
            lane.owner = owner;
        }
        if lane.parked && !suspended {
            lane.parked = false;
            lane.awaited = true;
            lane.own_submit = false;
        }
        self.sealed_past.remove(&stamp.lane);
        lane.waits_on = None;
        let for_front_frame = lane
            .frames
            .front()
            .is_none_or(|front| front.seq == stamp.seq);
        let frame = match lane.frames.iter_mut().find(|frame| frame.seq == stamp.seq) {
            Some(frame) => frame,
            None => {
                lane.frames.push_back(PendingFrame {
                    seq: stamp.seq,
                    expected: stamp.fires,
                    truncated: false,
                    park: false,
                    fires: Vec::with_capacity(stamp.fires as usize),
                    complete_at: None,
                });
                lane.frames.back_mut().expect("frame just pushed")
            }
        };
        frame.truncated |= late || suspended;
        frame.fires.push(fire);
        if for_front_frame {
            self.gather_seq = self.gather_seq.wrapping_add(1);
        }
        frame.expected = if frame.truncated {
            frame.fires.len() as u32
        } else {
            frame.expected.max(stamp.fires)
        };
        if frame.complete_at.is_none() && frame.is_complete() {
            frame.complete_at = Some(Instant::now());
        }
        let cut = frame.truncated && frame.expected < stamp.fires;
        if cut {
            self.truncated_seqs.insert(stamp.lane, stamp.seq);
        }
    }

    pub fn on_frame_truncated(&mut self, lane: ProcessId, seq: u64, submitted: u32) {
        if let Some(lane) = self.lanes.get_mut(&lane)
            && let Some(frame) = lane.frames.iter_mut().find(|frame| frame.seq == seq)
        {
            frame.expected = submitted;
            frame.truncated = true;
        }
    }

    pub fn on_lane_open(&mut self, lane: ProcessId, owner: ProcessId) {
        let suspended = self.suspended.contains(&owner);
        self.lanes.entry(lane).or_insert_with(|| LaneState {
            owner: Some(owner),
            awaited: !suspended,
            served: false,
            retired_at: None,
            parked: suspended,
            frames: VecDeque::new(),
            fired_this_boundary: true,
            prompting: false,
            own_submit: false,
            waits_on: None,
        });
    }

    pub fn on_lane_park(&mut self, lane: ProcessId, seq: u64) {
        let Some(state) = self.lanes.get_mut(&lane) else {
            return;
        };
        if state.frames.iter().any(|frame| frame.park) {
            return;
        }
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
                complete_at: Some(Instant::now()),
            },
        );
    }

    pub fn on_frame_retired(&mut self, lanes: impl IntoIterator<Item = ProcessId>) {
        let now = Instant::now();
        self.last_retire_at = Some(now);
        for lane in lanes {
            self.in_flight_lanes.remove(&lane);
            self.in_flight_since.remove(&lane);
            self.prompts_executing.remove(&lane);
            if let Some(state) = self.lanes.get_mut(&lane) {
                state.retired_at = Some(now);
            }
        }
    }

    fn retire_parks(&mut self) {
        for (lane_id, state) in self.lanes.iter_mut() {
            while state.frames.front().is_some_and(|frame| frame.park) {
                state.frames.pop_front();
                if state.frames.is_empty() {
                    if state.awaited {
                        trace_unawait("park", *lane_id, state.owner);
                    }
                    state.awaited = false;
                    state.parked = true;
                    // Parked by its own guest: it comes back by its own
                    // next submit, never by its owner's unblock or resume.
                    state.own_submit = true;
                }
            }
        }
    }

    pub fn on_bind_enqueued(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid {
            *self.pending_binds.entry(pid).or_default() += 1;
            if !self
                .lanes
                .values()
                .any(|lane| lane.owner == Some(pid) && !lane.frames.is_empty())
            {
                self.staged.insert(pid);
                if self.slotted.contains(&pid) {
                    self.joins_in_flight.insert(pid);
                }
            }
        }
    }

    pub fn preload_free_slots(&mut self, slots: usize) {
        self.pending_slots = slots as u64;
    }

    pub fn on_execution_slot_released(&mut self, pid: ProcessId) {
        self.departing.remove(&pid);
        self.pending_slots += 1;
    }

    pub fn on_execution_slot_consumed(&mut self, pid: ProcessId) {
        self.pending_slots = self.pending_slots.saturating_sub(1);
        self.admission_queue.retain(|queued| *queued != pid);
        self.slotted.insert(pid);
        if self.staged.contains(&pid) {
            self.joins_in_flight.insert(pid);
        }
    }

    pub fn on_admission_queued(&mut self, pid: ProcessId) {
        if !self.admission_queue.contains(&pid) {
            self.admission_queue.push_back(pid);
        }
    }

    pub fn on_admission_dequeued(&mut self, pid: ProcessId) {
        self.admission_queue.retain(|queued| *queued != pid);
    }

    pub fn on_slotted_terminate(&mut self, pid: ProcessId) {
        if self.slotted.remove(&pid) {
            self.departing.insert(pid);
        }
    }

    pub fn has_pending_binds(&self) -> bool {
        !self.pending_binds.is_empty()
    }

    pub fn earmarked(&self) -> impl Iterator<Item = ProcessId> + '_ {
        let slots = self.pending_slots as usize + self.departing.len();
        // The queue the execution gate grants from, in the order it grants:
        // arrival, unless a schedule ranks it.
        let mut queue: Vec<ProcessId> = self.admission_queue.iter().copied().collect();
        if let Some(schedule) = crate::policy::schedule::current() {
            queue.sort_by_key(|pid| std::cmp::Reverse(schedule.key(*pid)));
        }
        queue
            .into_iter()
            .filter(|pid| self.staged.contains(pid))
            .take(slots)
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

    pub fn on_lane_leave(&mut self, lane: ProcessId, owner: Option<ProcessId>, purge_queued: bool) {
        let owner = owner.or_else(|| self.lanes.get(&lane).and_then(|state| state.owner));
        // Its prompt holds no other behind it from here, landed or not.
        self.prompts_executing.remove(&lane);
        self.posted_through.remove(&lane);
        if purge_queued {
            self.lanes.remove(&lane);
            self.truncated_seqs.remove(&lane);
            self.in_flight_lanes.remove(&lane);
        } else {
            self.closed.insert(lane, owner);
            if let Some(state) = self.lanes.get_mut(&lane) {
                if state.awaited {
                    trace_unawait("close", lane, owner);
                }
                state.awaited = false;
                state.parked = true;
                state.own_submit = true;
                self.truncate_incomplete(lane);
                let drained = self
                    .lanes
                    .get(&lane)
                    .is_some_and(|state| state.frames.is_empty());
                if drained {
                    self.lanes.remove(&lane);
                }
            }
        }
        if let Some(owner) = owner {
            self.pending_binds.remove(&owner);
            self.forget_staged(owner);
        }
        self.maybe_reset_episode();
    }

    pub fn on_process_leave(&mut self, owner: ProcessId) {
        let owned: Vec<ProcessId> = self
            .lanes
            .iter()
            .filter(|(_, lane)| lane.owner == Some(owner))
            .map(|(id, _)| *id)
            .collect();
        for id in owned {
            self.truncated_seqs.remove(&id);
            self.in_flight_lanes.remove(&id);
            self.prompts_executing.remove(&id);
            self.posted_through.remove(&id);
        }
        self.posted_through.remove(&owner);
        self.lanes.retain(|_, lane| lane.owner != Some(owner));
        self.pending_binds.remove(&owner);
        self.suspended.remove(&owner);
        self.blocked.remove(&owner);
        self.closed.retain(|_, by| *by != Some(owner));
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

    pub fn on_process_suspend(&mut self, owner: ProcessId) {
        self.suspended.insert(owner);
        // A suspended owner's prompt holds no other prompt: its fires are
        // drained or cancelled by the suspension, and whether a landing
        // for them ever reaches here is the suspension's, not the frame's.
        let owned: Vec<ProcessId> = self
            .lanes
            .iter()
            .filter(|(_, lane)| lane.owner == Some(owner))
            .map(|(id, _)| *id)
            .collect();
        for lane in owned {
            self.prompts_executing.remove(&lane);
        }
        self.prompts_executing.remove(&owner);
        self.pending_binds.remove(&owner);
        self.park_owner(owner, "suspend");
    }

    /// `owner` waits on something only the frame's firing frees: its bind
    /// waits for room for its guest state, or its pass waits for a raise of
    /// the activation rung (the room a raise is funded with is freed by the
    /// planner, never by the frame it waits behind). Until it is through
    /// the process submits nothing, so a frame does not wait on its lanes:
    /// they are parked as a suspension's are, and come back with its next
    /// submit ([`Self::record_arrival`]).
    /// For a test of the waits that park an owner: `owner`'s lane fired at
    /// a boundary another lane holds open, its next frame complete; told
    /// it waits, no frame waits on it: its frame rides the seal the other,
    /// still expected, lane completes.
    #[cfg(test)]
    pub(crate) fn parked_owner_frame_goes_at_once(owner: ProcessId) -> bool {
        Self::parked_owner_frame_goes_at_once_by(owner, false)
    }

    /// As [`Self::parked_owner_frame_goes_at_once`], the owner parked by
    /// its suspension (`suspend`: an eviction, its guest between waits)
    /// rather than by a wait of its own.
    #[cfg(test)]
    pub(crate) fn parked_owner_frame_goes_at_once_by(owner: ProcessId, suspend: bool) -> bool {
        let mut policy =
            Self::new(1, 64, 4096, None);
        let other = ProcessId::new_v4();
        let stamp = |lane, seq| FrameStamp {
            lane,
            seq,
            slot: 0,
            fires: 1,
        };
        policy.on_fire_enqueued(stamp(owner, 0), Some(owner), 1, 1, 1);
        policy.on_fire_enqueued(stamp(other, 0), Some(other), 2, 1, 1);
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        let now = Instant::now();
        let _ = policy.plan_dispatch(&queued, &HashSet::new(), false, now);
        policy.on_frame_posted([owner, other], now);
        policy.on_frame_retired([owner, other]);
        policy.on_fire_enqueued(stamp(owner, 1), Some(owner), 3, 1, 1);
        let queued: QueuedFireIds = [3].into_iter().collect();
        // The control, with a frame in flight: the other lane's guest
        // computes and is held for (only an idle device seals past it).
        let held = matches!(
            policy.plan_dispatch(&queued, &HashSet::new(), true, now),
            FramePlan::Dispatch(waves) if waves.iter().flatten().any(|&id| id == 3)
        );
        if suspend {
            policy.on_process_suspend(owner);
        } else {
            policy.on_owner_waits(owner);
        }
        // The other lane, still expected, submits: the parked owner's frame
        // rides the seal it completes, one frame of both.
        policy.on_fire_enqueued(stamp(other, 1), Some(other), 4, 1, 1);
        let queued: QueuedFireIds = [3, 4].into_iter().collect();
        let rides = matches!(
            policy.plan_dispatch(&queued, &HashSet::new(), false, now),
            FramePlan::Dispatch(waves)
                if waves.iter().flatten().any(|&id| id == 3)
                    && waves.iter().flatten().any(|&id| id == 4)
        );
        !held && rides
    }

    pub fn on_owner_waits(&mut self, owner: ProcessId) {
        self.blocked.insert(owner);
        self.park_owner(owner, "owner-waits");
    }

    /// `owner`'s guest waits on input from outside the runtime: a session
    /// receive, a timer, an HTTP response. Nothing the frame does brings
    /// that input, so no frame waits on its lanes: they are parked, and
    /// each comes back with its own next submit ([`Self::record_arrival`]).
    /// A guest that computes, or waits on its own fires and channels, is
    /// not away: its lanes stay awaited.
    pub fn on_owner_away(&mut self, owner: ProcessId) {
        self.park_owner(owner, "outside-wait");
        for lane in self.lanes.values_mut() {
            if lane.owner == Some(owner) {
                lane.own_submit = true;
            }
        }
        self.lanes
            .retain(|_, lane| lane.awaited || (lane.parked && !lane.own_submit) || !lane.frames.is_empty());
    }

    /// Parks `owner`'s lanes: none is awaited, what arrived of an
    /// incomplete frame goes as it stands.
    fn park_owner(&mut self, owner: ProcessId, cause: &str) {
        let owned: Vec<ProcessId> = self
            .lanes
            .iter()
            .filter(|(_, lane)| lane.owner == Some(owner))
            .map(|(id, _)| *id)
            .collect();
        for lane_id in owned {
            if let Some(lane) = self.lanes.get_mut(&lane_id) {
                if lane.awaited {
                    trace_unawait(cause, lane_id, Some(owner));
                }
                lane.awaited = false;
                lane.parked = true;
            }
            // Kept, drained or not: its owner's return makes it expected
            // again ([`Self::expect_again`]), before its next submit.
            self.truncate_incomplete(lane_id);
        }
        // Not joining while it waits: it holds no frame back as one.
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

    /// `owner`'s wait on the planner or a store ended, before control went
    /// back to its guest: its next fires are awaited as before.
    pub fn on_owner_unblocked(&mut self, owner: ProcessId) {
        self.blocked.remove(&owner);
        self.expect_again(owner);
    }

    /// `lane`'s process waits on the output of the fire `completion`
    /// settles, having submitted nothing since: it submits nothing before
    /// that fire's frame lands, so the frames sealed until then go without
    /// it. Its next submit ends the wait ([`Self::record_arrival`]), as the
    /// fire settling does.
    pub fn on_lane_waits(&mut self, lane: ProcessId, completion: WorkItemCompletion) {
        if completion.is_settled() {
            return;
        }
        if let Some(state) = self.lanes.get_mut(&lane) {
            state.waits_on = Some(completion);
        }
    }

    pub fn on_process_resume(&mut self, owner: ProcessId) {
        self.suspended.remove(&owner);
        self.expect_again(owner);
    }

    /// `owner` neither suspended nor blocked now: its parked lanes are
    /// expected again, its next submit coming. Forgotten until that submit,
    /// a frame sealed at another's park went without them, and batches
    /// split at every block.
    fn expect_again(&mut self, owner: ProcessId) {
        if self.suspended.contains(&owner) || self.blocked.contains(&owner) {
            return;
        }
        for lane in self.lanes.values_mut() {
            if lane.owner == Some(owner) && lane.parked && !lane.own_submit {
                lane.parked = false;
                lane.awaited = true;
            }
        }
    }

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
            cut = Some(frame.seq);
        }
        if let Some(seq) = cut {
            self.truncated_seqs.insert(lane_id, seq);
        }
    }

    fn forget_staged(&mut self, pid: ProcessId) {
        self.staged.remove(&pid);
        self.joins_in_flight.remove(&pid);
        self.in_flight_since.remove(&pid);
    }

    fn maybe_reset_episode(&mut self) {
        if self.lanes.values().any(|lane| lane.awaited) {
            return;
        }
        self.strict_watchdog_deadline = None;
    }

    pub fn has_queued_frames(&self) -> bool {
        self.lanes.values().any(|lane| !lane.frames.is_empty())
    }

    pub fn debug_summary(&self) -> String {
        use std::fmt::Write as _;
        let mut out = format!(
            "frame k={} lanes={} awaited={} sealed={} \
pending_binds={} staged={} joins_in_flight={} departing={} suspended={} \
blocked={} pending_slots={} watchdog={:?}",
            self.k,
            self.lanes.len(),
            self.lanes.values().filter(|lane| lane.awaited).count(),
            self.sealed.len(),
            self.pending_binds.values().sum::<usize>(),
            self.staged.len(),
            self.joins_in_flight.len(),
            self.departing.len(),
            self.suspended.len(),
            self.blocked.len(),
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
            let owes = self.in_flight_lanes.contains(pid)
                || lane
                    .owner
                    .is_some_and(|owner| self.pending_binds.contains_key(&owner));
            // What the lane waits on, named: its own fire or bind, its guest
            // computing (held while a frame covers the device), or sealed
            // past while its guest computed (it rejoins with its next submit).
            let waits = if self.sealed_past.contains(pid) {
                "sealed past: its guest computed while the device would have idled; rejoins with its next submit"
            } else if !lane.awaited || front_complete {
                "nothing"
            } else if self.in_flight_lanes.contains(pid) {
                "its fire in flight"
            } else if owes {
                "its bind"
            } else if !lane.frames.is_empty() {
                "the rest of its frame's fires"
            } else {
                "its guest computing: held while a frame is in flight"
            };
            let _ = write!(
                out,
                "\n  lane {pid}: owner={:?} awaited={} queued_frames={} front_complete={front_complete} \
                 owner_suspended={} owner_blocked={} expected={} waits={waits}",
                lane.owner,
                lane.awaited,
                lane.frames.len(),
                lane.owner.is_some_and(|owner| self.suspended.contains(&owner)),
                lane.owner.is_some_and(|owner| self.blocked.contains(&owner)),
                lane.awaited && !lane.parked,
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

mod seal;

#[cfg(test)]
mod tests;
