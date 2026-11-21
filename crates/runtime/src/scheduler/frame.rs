use crate::rt::Instant;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use super::stats::SchedulerStats;
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

/// How long a frame held for lanes that have not submitted waits before it
/// says which, and again after each such span.
const HOLD_SAYS_AFTER: Duration = Duration::from_secs(5);

fn idle_dump_min_seats() -> usize {
    static SEATS: OnceLock<usize> = OnceLock::new();
    *SEATS.get_or_init(|| {
        std::env::var("PIE_IDLE_DUMP_MIN_SEATS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Whether a seal says what it sealed: the wave trace, or the scheduler's
/// debug level. Off, nothing on the hot path reads a fire's program.
pub(crate) fn seal_trace() -> bool {
    super::worker::wave_trace() || tracing::enabled!(tracing::Level::DEBUG)
}

/// Says, on the seal trace, why `lane` stops being awaited: which
/// mechanism took it out of the gate.
fn trace_unawait(cause: &str, lane: ProcessId, owner: Option<ProcessId>) {
    if !seal_trace() {
        return;
    }
    let line = format!("unawait cause={cause} lane={lane} owner={owner:?}");
    if super::worker::wave_trace() {
        super::worker::wave_trace_emit(format!(
            "[wave-trace] t={}us {line}",
            super::worker::wave_trace_us()
        ));
    }
    tracing::debug!("{line}");
}

pub(super) fn idle_dump_threshold_us() -> u64 {
    static THRESHOLD: OnceLock<u64> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("PIE_IDLE_DUMP_US")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(u64::MAX)
    })
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
    clock_from: Option<Instant>,
    frames: VecDeque<PendingFrame>,
    fired_this_boundary: bool,
    /// Its last sealed frame carried a prompt: its next pass is built on the
    /// host from what that prompt answers, so only decodes wait for it.
    prompting: bool,
}

pub(super) const ABANDONED: &str = "pipeline abandoned: silent past the silence timeout without \
     submitting and without calling forward.park()";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateVerdict {
    Satisfied,
    Blocking,
    SkippedOwed,
}

impl LaneState {
    fn gate_verdict(&self, relax: bool, owes: bool) -> GateVerdict {
        if !self.awaited || self.frames.front().is_some_and(PendingFrame::is_complete) {
            return GateVerdict::Satisfied;
        }
        if relax && owes && self.frames.is_empty() {
            return GateVerdict::SkippedOwed;
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
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FramePlan {
    Dispatch(Vec<Vec<u64>>),
    Hold(Duration),
    Park,
    Terminate(Vec<ProcessId>),
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

pub(crate) struct FramePolicy {
    k: usize,
    max_wave_tokens: usize,
    /// The tokens a wave takes where the worker policy stated them
    /// (`install-batch`), in place of the engine's one fire's: prompts past
    /// one fire then share a frame, and the worker cuts the wave into fires.
    batch_tokens: Option<usize>,
    max_wave_rows: usize,
    /// The rows a wave may read out, the engine's `max_readouts`; 0 bounds
    /// nothing.
    max_wave_readouts: usize,
    lanes: BTreeMap<ProcessId, LaneState>,
    sealed: VecDeque<SealedFrame>,
    executing_now: bool,
    last_retire_at: Option<Instant>,
    idle_dumped: bool,
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
    truncated_seqs: BTreeMap<ProcessId, u64>,
    strict_watchdog_deadline: Option<Instant>,
    /// When a held frame last said which lanes it waits on.
    hold_said: Option<Instant>,
    join_hold: Option<Instant>,
    gather_seq: u64,
    quiesce_mark: Option<u64>,
    /// The lanes with a fire in a posted frame, exactly
    /// ([`Self::on_frame_posted`], [`Self::on_frame_retired`]).
    in_flight_lanes: BTreeSet<ProcessId>,
    in_flight_since: BTreeMap<ProcessId, Instant>,
    /// The rung rows a fire sealed now may take: the activation rung's cap
    /// ([`Self::cap_fires`]).
    fire_cap: usize,
    silence_timeout: Duration,
    stats: Option<Arc<SchedulerStats>>,
}

impl FramePolicy {
    #[cfg(test)]
    fn with_silence_timeout_off(mut self) -> Self {
        self.silence_timeout = Duration::from_secs(86_400);
        self
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
    pub fn bound_batch(&mut self, tokens: Option<usize>) {
        self.batch_tokens = tokens;
    }

    /// Fires are cut to `rows` of the activation rung now: a lane whose
    /// next frame holds a wider fire (cut under a cap since lowered) is
    /// left out of the frames sealed from here, its frame kept whole until
    /// the cap is back; every fire of a frame sealed goes.
    pub fn cap_fires(&mut self, rows: usize) {
        self.fire_cap = rows;
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
        for lane in lanes {
            self.in_flight_lanes.insert(lane);
            self.in_flight_since.entry(lane).or_insert(now);
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
            batch_tokens: None,
            max_wave_rows,
            max_wave_readouts: 0,
            lanes: BTreeMap::new(),
            sealed: VecDeque::new(),
            executing_now: false,
            last_retire_at: None,
            idle_dumped: false,
            pending_binds: BTreeMap::new(),
            staged: BTreeSet::new(),
            prompts_only: false,
            pending_slots: 0,
            joins_in_flight: BTreeSet::new(),
            admission_queue: VecDeque::new(),
            slotted: BTreeSet::new(),
            departing: BTreeSet::new(),
            suspended: BTreeSet::new(),
            truncated_seqs: BTreeMap::new(),
            strict_watchdog_deadline: None,
            hold_said: None,
            join_hold: None,
            gather_seq: 0,
            quiesce_mark: None,
            in_flight_lanes: BTreeSet::new(),
            fire_cap: usize::MAX,
            in_flight_since: BTreeMap::new(),
            silence_timeout: crate::scheduler::configured_silence_timeout(),
            stats,
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
        let suspended = lane_owner.is_some_and(|owner| self.suspended.contains(&owner));
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
            clock_from: None,
            frames: VecDeque::new(),
            fired_this_boundary: true,
            prompting: false,
        });
        if lane.owner.is_none() {
            lane.owner = owner;
        }
        if lane.parked && !suspended {
            lane.parked = false;
            lane.awaited = true;
        }
        lane.clock_from = None;
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
            clock_from: None,
            frames: VecDeque::new(),
            fired_this_boundary: true,
            prompting: false,
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
            if let Some(state) = self.lanes.get_mut(&lane) {
                state.clock_from = None;
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
                    state.clock_from = None;
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
        if purge_queued {
            self.lanes.remove(&lane);
            self.truncated_seqs.remove(&lane);
            self.in_flight_lanes.remove(&lane);
        } else if let Some(state) = self.lanes.get_mut(&lane) {
            if state.awaited {
                trace_unawait("close", lane, owner);
            }
            state.awaited = false;
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
        }
        self.lanes.retain(|_, lane| lane.owner != Some(owner));
        self.pending_binds.remove(&owner);
        self.suspended.remove(&owner);
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

    pub fn on_process_suspend(&mut self, owner: ProcessId) {
        self.suspended.insert(owner);
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
    pub fn on_owner_waits(&mut self, owner: ProcessId) {
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
            self.truncate_incomplete(lane_id);
            let drained = self
                .lanes
                .get(&lane_id)
                .is_some_and(|lane| lane.frames.is_empty());
            if drained {
                self.lanes.remove(&lane_id);
            }
        }
        // Not joining while it waits: it holds no frame back as one.
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

    pub fn on_process_resume(&mut self, owner: ProcessId) {
        self.suspended.remove(&owner);
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
        self.idle_dumped = false;
    }

    fn have_seal_candidate(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete))
    }

    fn boundary_open(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.awaited && !lane.fired_this_boundary)
    }

    fn open_boundary(&mut self) {
        for lane in self.lanes.values_mut() {
            lane.fired_this_boundary = false;
        }
    }

    fn close_boundary(&mut self) {
        for lane in self.lanes.values_mut() {
            lane.fired_this_boundary = true;
        }
    }

    fn seal(&mut self) -> Option<FramePlan> {
        if !self.have_seal_candidate() {
            return None;
        }
        let mid_boundary = self.boundary_open();
        let trace = seal_trace();

        let k = self.k;
        let max_wave_rows = self.max_wave_rows;
        let max_wave_tokens = self.batch_tokens.unwrap_or(self.max_wave_tokens);
        let max_wave_readouts = self.max_wave_readouts;
        // One frame of prompts on the card at a time: prompts arriving behind
        // it bind together into the next, while decodes keep running ahead.
        // A lane whose own prompt is in flight continues it in the next frame
        // (its next piece reads what that one writes, in order on the
        // device): there is no prompt behind it to bind.
        let prompting_in_flight: BTreeSet<ProcessId> = self
            .in_flight_lanes
            .iter()
            .copied()
            .filter(|lane| self.lanes.get(lane).is_some_and(|state| state.prompting))
            .collect();
        loop {
            let mut fresh: Vec<ProcessId> = Vec::new();
            let mut continuing: Vec<ProcessId> = Vec::new();
            for (lane_id, lane) in self.lanes.iter() {
                if !lane.frames.front().is_some_and(PendingFrame::is_complete) {
                    continue;
                }
                if mid_boundary && lane.fired_this_boundary {
                    if !lane.prompting {
                        continuing.push(*lane_id);
                    }
                } else {
                    fresh.push(*lane_id);
                }
            }
            if mid_boundary && fresh.is_empty() {
                return None;
            }
            // The policy's schedule ranks lanes by their process's key; a
            // stable sort keeps lane order among equals, so with no schedule
            // nothing moves.
            if let Some(schedule) = crate::policy::schedule::current() {
                let key = |lane: &ProcessId| {
                    std::cmp::Reverse(
                        self.lanes
                            .get(lane)
                            .and_then(|l| l.owner)
                            .map_or(0, |owner| schedule.key(owner)),
                    )
                };
                fresh.sort_by_key(key);
                continuing.sort_by_key(key);
            }
            let mut order: Vec<ProcessId> = Vec::with_capacity(fresh.len() + continuing.len());
            let mut rest = fresh.as_slice();
            if let Some((first, tail)) = fresh.split_first() {
                order.push(*first);
                rest = tail;
            }
            order.extend_from_slice(&continuing);
            order.extend_from_slice(rest);
            // Decodes ride a seal for the lanes coming off a prompt only
            // beside a prompt: alone they would split the wave those lanes
            // rejoin.
            if self.prompts_only
                && !order.iter().any(|lane| {
                    self.lanes.get(lane).is_some_and(|state| {
                        state.frames.front().is_some_and(|front| {
                            front.fires.iter().any(|fire| {
                                fire.fire_id.is_some() && fire.tokens > fire.rows.max(1)
                            })
                        })
                    })
                })
            {
                return None;
            }
            // Riding decodes take each of the frame's steps: the prompts
            // beside them spread over the steps too, so none runs decodes alone.
            let spread = self.prompts_only
                && k > 1
                && order.iter().any(|lane| {
                    self.lanes.get(lane).is_some_and(|state| {
                        state.frames.front().is_some_and(|front| {
                            front
                                .fires
                                .iter()
                                .any(|fire| fire.fire_id.is_some() && fire.slot > 0)
                        })
                    })
                });

            let mut waves: Vec<Vec<u64>> = vec![Vec::new(); k];
            let mut fire_waves = HashMap::new();
            let mut wave_tokens = vec![0usize; k];
            let mut wave_rows = vec![0usize; k];
            let mut wave_readouts = vec![0usize; k];
            let mut members: HashSet<ProcessId> = HashSet::new();
            let mut sealed_widest = 0usize;
            let mut dropped_empty = false;
            // The lanes of each program the frame carries (the trace's).
            let mut by_program: BTreeMap<u64, usize> = BTreeMap::new();
            for lane_id in &order {
                let Some(lane) = self.lanes.get_mut(lane_id) else {
                    continue;
                };
                let Some(front) = lane.frames.front() else {
                    continue;
                };
                // Cut under a cap since lowered: kept whole, out of this
                // frame, until the cap is back (the queued rows ask it).
                if front
                    .fires
                    .iter()
                    .any(|fire| fire.rung_rows > self.fire_cap)
                {
                    continue;
                }
                let mut live: Vec<PlacedFire> = front
                    .fires
                    .iter()
                    .filter_map(|fire| {
                        fire.fire_id.map(|fire_id| PlacedFire {
                            wave: (fire.slot as usize).min(k - 1),
                            fire_id,
                            rows: fire.rows.max(1),
                            tokens: fire.tokens,
                            readouts: fire.readouts,
                        })
                    })
                    .collect();
                if live.is_empty() {
                    lane.frames.pop_front();
                    dropped_empty = true;
                    continue;
                }
                if spread
                    && let [fire] = live.as_mut_slice()
                    && fire.tokens > fire.rows
                {
                    fire.wave = (0..k).min_by_key(|&w| wave_tokens[w]).unwrap_or(0);
                }
                let fits = live.iter().all(|fire| {
                    let reading = wave_readouts[fire.wave];
                    // A lane wider than the wave still takes one of its own.
                    wave_rows[fire.wave] == 0
                        || wave_rows[fire.wave] + fire.rows <= max_wave_rows
                            && wave_tokens[fire.wave] + fire.tokens <= max_wave_tokens
                            && (max_wave_readouts == 0
                                || reading == 0
                                || reading + fire.readouts <= max_wave_readouts)
                });
                if !fits {
                    continue;
                }
                let prompting = live.iter().any(|fire| fire.tokens > fire.rows);
                if prompting
                    && !mid_boundary
                    && !self.prompts_only
                    && prompting_in_flight.iter().any(|lane| lane != lane_id)
                {
                    continue;
                }
                for fire in &live {
                    wave_rows[fire.wave] += fire.rows;
                    wave_tokens[fire.wave] += fire.tokens;
                    wave_readouts[fire.wave] += fire.readouts;
                    waves[fire.wave].push(fire.fire_id);
                    fire_waves.insert(fire.fire_id, fire.wave);
                }
                members.insert(*lane_id);
                let lane = self.lanes.get_mut(lane_id).expect("measured lane exists");
                sealed_widest = lane
                    .frames
                    .front()
                    .into_iter()
                    .flat_map(|front| front.fires.iter().map(|fire| fire.rung_rows))
                    .fold(sealed_widest, usize::max);
                let frame_complete_at = lane.frames.front().and_then(|front| front.complete_at);
                if let (Some(done), Some(from)) = (frame_complete_at, lane.retired_at) {
                    let us = done.saturating_duration_since(from).as_micros() as u64;
                    if let Some(stats) = &self.stats {
                        use std::sync::atomic::Ordering::Relaxed;
                        stats.fire.quorum.turnaround_sum_us.fetch_add(us, Relaxed);
                        stats.fire.quorum.turnaround_n.fetch_add(1, Relaxed);
                        stats.fire.quorum.turnaround_max_us.fetch_max(us, Relaxed);
                    }
                }
                if trace {
                    let mut programs: Vec<u64> = lane
                        .frames
                        .front()
                        .into_iter()
                        .flat_map(|front| front.fires.iter())
                        .filter(|fire| fire.fire_id.is_some())
                        .map(|fire| fire.program)
                        .collect();
                    programs.sort_unstable();
                    programs.dedup();
                    for program in programs {
                        *by_program.entry(program).or_default() += 1;
                    }
                }
                lane.served = true;
                lane.prompting = prompting;
                lane.frames.pop_front();
            }
            if fire_waves.is_empty() {
                if dropped_empty {
                    continue;
                }
                self.lanes
                    .retain(|_, lane| lane.awaited || !lane.frames.is_empty());
                return None;
            }
            if trace {
                let complete = self
                    .lanes
                    .values()
                    .filter(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete))
                    .count();
                let awaited = self.lanes.values().filter(|lane| lane.awaited).count();
                let parked = self.lanes.values().filter(|lane| lane.parked).count();
                let line = format!(
                    "seal partition={} fresh={} continuing={} complete_left={} awaited={} parked={parked} lanes={} in_flight={} mid_boundary={mid_boundary} prompts_only={} executing={} programs={} lanes_by_program={by_program:?} rows={wave_rows:?} tokens={wave_tokens:?}",
                    members.len(),
                    fresh.len(),
                    continuing.len(),
                    complete,
                    awaited,
                    self.lanes.len(),
                    self.in_flight_lanes.len(),
                    self.prompts_only,
                    self.executing_now,
                    by_program.len(),
                );
                if super::worker::wave_trace() {
                    super::worker::wave_trace_emit(format!(
                        "[wave-trace] t={}us {line}",
                        super::worker::wave_trace_us()
                    ));
                }
                tracing::debug!("{line}");
            }
            if !mid_boundary && !self.prompts_only {
                self.open_boundary();
            }
            for member in &members {
                if let Some(lane) = self.lanes.get_mut(member) {
                    lane.fired_this_boundary = true;
                }
            }
            self.record_sealed_waves(waves.iter().filter(|wave| !wave.is_empty()).count());
            self.record_seal_engagement();
            let _ = &fire_waves;
            self.sealed.push_back(SealedFrame {
                waves,
                members: members.iter().copied().collect(),
                widest: sealed_widest,
            });
            self.lanes
                .retain(|_, lane| lane.awaited || !lane.frames.is_empty());
            return Some(FramePlan::Dispatch(Vec::new()));
        }
    }

    pub fn record_rider_wave(&self) {
        self.record_sealed_waves(1);
    }

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

    fn record_seal_engagement(&self) {
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            stats.fire.quorum.seal_events.fetch_add(1, Relaxed);
            if self.executing_now {
                stats.fire.quorum.seal_while_executing.fetch_add(1, Relaxed);
            }
        }
    }

    pub fn plan_dispatch(
        &mut self,
        still_queued: &QueuedFireIds,
        blocked_lanes: &HashSet<ProcessId>,
        executing: bool,
        now: Instant,
    ) -> FramePlan {
        self.executing_now = executing;
        let threshold = idle_dump_threshold_us();
        if threshold != u64::MAX
            && !executing
            && let Some(since) = self.last_retire_at
        {
            let idle = now.saturating_duration_since(since).as_micros() as u64;
            let seated = self.lanes.values().filter(|lane| lane.awaited).count();
            if idle >= threshold && !self.idle_dumped && seated >= idle_dump_min_seats() {
                self.idle_dumped = true;
                let (mut ready, mut empty_owed, mut empty_unowed, mut partial) = (0, 0, 0, 0);
                let (mut unowed_fresh, mut unowed_between) = (0, 0);
                let (mut turn_max, mut turn_sum, mut turn_n) = (0u64, 0u64, 0u64);
                let mut oldest_ready_us = 0u64;
                let mut newest_ready_us = u64::MAX;
                for (lane_id, lane) in &self.lanes {
                    if !lane.awaited {
                        continue;
                    }
                    if lane.frames.front().is_some_and(PendingFrame::is_complete) {
                        ready += 1;
                        if let Some(at) = lane.frames.front().and_then(|f| f.complete_at) {
                            let age = now.saturating_duration_since(at).as_micros() as u64;
                            oldest_ready_us = oldest_ready_us.max(age);
                            newest_ready_us = newest_ready_us.min(age);
                        }
                    } else if lane.frames.is_empty() {
                        let owes = self.in_flight_lanes.contains(lane_id)
                            || lane
                                .owner
                                .is_some_and(|owner| self.pending_binds.contains_key(&owner));
                        if owes {
                            empty_owed += 1;
                        } else {
                            empty_unowed += 1;
                            if lane.served {
                                unowed_between += 1;
                                if let Some(at) = lane.retired_at {
                                    let age = now.saturating_duration_since(at).as_micros() as u64;
                                    turn_max = turn_max.max(age);
                                    turn_sum += age;
                                    turn_n += 1;
                                }
                            } else {
                                unowed_fresh += 1;
                            }
                        }
                    } else {
                        partial += 1;
                    }
                }
                let open = self.boundary_open();
                let (mut fresh, mut continuing) = (0, 0);
                for lane in self.lanes.values() {
                    if lane.frames.front().is_some_and(PendingFrame::is_complete) {
                        if open && lane.fired_this_boundary {
                            continuing += 1;
                        } else {
                            fresh += 1;
                        }
                    }
                }
                println!(
                    "[device-idle] {idle}us awaited={} ready={ready} empty+owed={empty_owed} \
empty+unowed={empty_unowed}(fresh={unowed_fresh},between={unowed_between}) \
partial_front={partial} sealed={} staged={} \
pending_slots={} joins={} binds={} boundary_open={open} fresh={fresh} \
continuing={continuing} turnaround_max={turn_max}us \
turnaround_mean={}us ready_age_oldest={oldest_ready_us}us \
ready_age_newest={}us",
                    ready + empty_owed + empty_unowed + partial,
                    self.sealed.len(),
                    self.staged.len(),
                    self.pending_slots,
                    self.joins_in_flight.len(),
                    self.pending_binds.values().sum::<usize>(),
                    turn_sum.checked_div(turn_n).unwrap_or(0),
                    if newest_ready_us == u64::MAX {
                        0
                    } else {
                        newest_ready_us
                    },
                );
            }
        }
        loop {
            self.retire_parks();
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
                    if let Some(stats) = &self.stats {
                        stats
                            .fire
                            .quorum
                            .dispatch_blocked_holds
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    return FramePlan::Hold(Duration::from_micros(500));
                }
                let frame = self.sealed.pop_front().expect("front frame exists");
                let mut starved_us = 0u64;
                if !self.executing_now
                    && let Some(since) = self.last_retire_at
                    && let Some(stats) = &self.stats
                {
                    use std::sync::atomic::Ordering::Relaxed;
                    let idle = now.saturating_duration_since(since).as_micros() as u64;
                    stats.fire.quorum.device_idle_us.fetch_add(idle, Relaxed);
                    stats.fire.quorum.device_idle_gaps.fetch_add(1, Relaxed);
                    stats.record_bubble_us(idle);
                    starved_us = idle;
                }
                if starved_us >= idle_dump_threshold_us() {
                    println!("[idle-gap] {starved_us}us  {}", self.debug_summary());
                }
                return FramePlan::Dispatch(frame.waves);
            }
            if self.boundary_open() {
                if let Some(plan) = self.seal() {
                    match plan {
                        FramePlan::Dispatch(_) => continue,
                        plan => return plan,
                    }
                }
                self.close_boundary();
            }
            if !self.lanes.values().any(|lane| !lane.frames.is_empty()) {
                self.strict_watchdog_deadline = None;
                self.idle_dumped = false;
                return FramePlan::Park;
            }
            let mut missing_decoders = 0usize;
            let mut missing = 0;
            let mut blocking_lanes: Vec<(ProcessId, bool, bool)> = Vec::new();
            let mut expired: Vec<ProcessId> = Vec::new();
            let silence = self.silence_timeout;
            debug_assert!(
                !self.boundary_open(),
                "the wait-all gate is evaluated only against a closed boundary"
            );
            for (lane_id, lane) in self.lanes.iter_mut() {
                let owes = self.in_flight_lanes.contains(lane_id)
                    || lane
                        .owner
                        .is_some_and(|owner| self.pending_binds.contains_key(&owner));
                let debt_since = if owes {
                    self.in_flight_since.get(lane_id).copied()
                } else {
                    None
                };
                let owes_forever = debt_since
                    .is_some_and(|from| now.saturating_duration_since(from) >= silence * 2);
                let blocking = match lane.gate_verdict(false, owes) {
                    GateVerdict::Blocking => true,
                    GateVerdict::SkippedOwed | GateVerdict::Satisfied => false,
                };
                if !blocking {
                    lane.clock_from = None;
                    continue;
                }
                if (owes || executing) && !owes_forever {
                    lane.clock_from = None;
                } else {
                    if owes_forever {
                        lane.clock_from = debt_since;
                    }
                    match lane.clock_from {
                        Some(from) if now.saturating_duration_since(from) >= silence => {
                            trace_unawait("silence", *lane_id, lane.owner);
                            lane.awaited = false;
                            lane.clock_from = None;
                            expired.push(lane.owner.unwrap_or(*lane_id));
                            continue;
                        }
                        Some(_) => {}
                        None => lane.clock_from = Some(now),
                    }
                }
                if blocking {
                    missing += 1;
                    missing_decoders += usize::from(!lane.prompting);
                    if blocking_lanes.len() < 16 {
                        blocking_lanes.push((lane.owner.unwrap_or(*lane_id), lane.prompting, owes));
                    }
                }
            }
            if !expired.is_empty() {
                expired.sort_unstable();
                expired.dedup();
                return FramePlan::Terminate(expired);
            }
            if super::worker::wave_trace() {
                super::worker::wave_trace_emit(format!(
                    "[wave-trace] t={}us gate missing={missing} executing={executing} candidate={} in_flight={} awaited={} lanes={}",
                    super::worker::wave_trace_us(),
                    self.have_seal_candidate(),
                    self.in_flight_lanes.len(),
                    self.lanes.values().filter(|lane| lane.awaited).count(),
                    self.lanes.len()
                ));
            }
            if missing > 0
                && missing_decoders == 0
                && !self.has_pending_binds()
                && !self.is_joining()
            {
                self.prompts_only = true;
                let sealed = self.seal();
                self.prompts_only = false;
                if let Some(FramePlan::Dispatch(_)) = sealed {
                    continue;
                }
            }
            if missing > 0
                && self
                    .hold_said
                    .is_none_or(|said| now.saturating_duration_since(said) >= HOLD_SAYS_AFTER)
            {
                if self.hold_said.is_some() {
                    tracing::warn!(
                        missing,
                        executing,
                        pending_binds = self.pending_binds.values().sum::<usize>(),
                        joining = self.is_joining(),
                        lanes = self.lanes.len(),
                        waits_on = ?blocking_lanes,
                        "a frame is held for lanes that have not submitted (owner, prompting, owes)"
                    );
                }
                self.hold_said = Some(now);
            }
            if missing == 0 {
                self.hold_said = None;
            }
            if missing > 0 {
                if executing {
                    self.quiesce_mark = None;
                    return FramePlan::Park;
                }
                let deadline = self
                    .strict_watchdog_deadline
                    .get_or_insert(now + Duration::from_micros(STRICT_WATCHDOG_US));
                if now >= *deadline {
                    *deadline = now + Duration::from_micros(STRICT_WATCHDOG_US);
                }
                let hold = deadline
                    .saturating_duration_since(now)
                    .min(Duration::from_micros(GATHER_POLL_US));
                return FramePlan::Hold(hold);
            }
            self.strict_watchdog_deadline = None;
            self.idle_dumped = false;
            self.quiesce_mark = None;
            if !executing && self.is_joining() && self.have_seal_candidate() {
                let until = *self
                    .join_hold
                    .get_or_insert(now + Duration::from_micros(JOIN_HOLD_US));
                if now < until {
                    return FramePlan::Hold(
                        until
                            .saturating_duration_since(now)
                            .min(Duration::from_micros(GATHER_POLL_US)),
                    );
                }
            }
            self.join_hold = None;
            match self.seal() {
                Some(FramePlan::Dispatch(_)) => continue,
                Some(plan) => return plan,
                None => return FramePlan::Park,
            }
        }
    }

    pub fn has_queued_frames(&self) -> bool {
        self.lanes.values().any(|lane| !lane.frames.is_empty())
    }

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

    fn fires(plan: &FramePlan) -> Vec<u64> {
        match plan {
            FramePlan::Dispatch(waves) => waves.iter().flatten().copied().collect(),
            plan => panic!("expected a frame dispatch, got {plan:?}"),
        }
    }

    /// Lanes of three kinds share frames of two steps under a dispatch
    /// depth of two, each kind its own program (the epilogue lanes two, by
    /// turns, as a beam step's).
    ///
    /// A constant lane's next frame needs nothing from its guest, so it
    /// runs a frame ahead of the device. A channel lane's next frame reads
    /// an input its guest writes each step into a ring of `capacity` cells,
    /// and a put into a full ring waits until the device takes a cell: its
    /// next submit waits on the retire of the frame that takes what it
    /// wrote. An epilogue lane harvests its fire's result, then its guest
    /// computes (wasm between fires, no channel wait, no park) across
    /// several of the scheduler's passes, some while a frame executes and
    /// some while none does, before it submits again.
    ///
    /// Nothing tells the scheduler a guest waits or computes: each lane is
    /// expected this step like any other, so every frame seals with every
    /// lane of every kind.
    #[test]
    fn a_lane_whose_guest_writes_its_input_or_computes_between_fires_is_awaited() {
        const K: usize = 2;
        const DEPTH: usize = 2;
        const EACH: usize = 3;
        /// The scheduler passes an epilogue spans after its harvest.
        const EPILOGUE: usize = 3;
        #[derive(Clone, Copy)]
        enum Kind {
            Constant,
            /// The ring's tail (cells written) and head (cells taken).
            Channel(usize, usize),
            /// Passes left in its epilogue.
            Epilogue(usize),
        }
        struct Lane {
            id: ProcessId,
            kind: Kind,
            /// Frames submitted and not retired.
            outstanding: usize,
            seq: u64,
        }
        impl Lane {
            fn index(&self) -> usize {
                match self.kind {
                    Kind::Constant => 0,
                    Kind::Channel(..) => 1,
                    Kind::Epilogue(_) => 2,
                }
            }
        }
        for capacity in [K, 2 * K] {
            let mut policy = FramePolicy::new(K, 64, 4096, None);
            let mut lanes: Vec<Lane> = (0..3 * EACH)
                .map(|at| Lane {
                    id: pid(),
                    kind: match at / EACH {
                        0 => Kind::Constant,
                        1 => Kind::Channel(0, 0),
                        _ => Kind::Epilogue(0),
                    },
                    outstanding: 0,
                    seq: 0,
                })
                .collect();
            let mut next_fire = 0u64;
            let mut owner_of: HashMap<u64, ProcessId> = HashMap::new();
            let mut queued: Vec<u64> = Vec::new();
            let mut in_flight: VecDeque<BTreeSet<ProcessId>> = VecDeque::new();
            let mut sealed: Vec<[usize; 3]> = Vec::new();

            let submit = |lanes: &mut Vec<Lane>,
                          policy: &mut FramePolicy,
                          queued: &mut Vec<u64>,
                          owner_of: &mut HashMap<u64, ProcessId>,
                          next_fire: &mut u64| {
                for lane in lanes.iter_mut() {
                    loop {
                        let room = match lane.kind {
                            Kind::Constant => lane.outstanding < DEPTH,
                            // The guest writes the frame's K cells first;
                            // a full ring parks the put.
                            Kind::Channel(tail, head) => tail - head + K <= capacity,
                            // It submits once its harvest is through and
                            // its guest has computed what comes next.
                            Kind::Epilogue(left) => lane.outstanding == 0 && left == 0,
                        };
                        if !room {
                            break;
                        }
                        if let Kind::Channel(tail, _) = &mut lane.kind {
                            *tail += K;
                        }
                        let program = match lane.kind {
                            Kind::Constant => 1,
                            Kind::Channel(..) => 2,
                            Kind::Epilogue(_) => 12 + 3 * (lane.seq % 2),
                        };
                        for slot in 0..K as u32 {
                            let fire = *next_fire;
                            *next_fire += 1;
                            policy.on_fire_reading(
                                stamp(lane.id, lane.seq, slot, K as u32),
                                Some(lane.id),
                                fire,
                                1,
                                1,
                                1,
                                1,
                                program,
                            );
                            owner_of.insert(fire, lane.id);
                            queued.push(fire);
                        }
                        lane.seq += 1;
                        lane.outstanding += 1;
                    }
                }
            };
            let schedule = |policy: &mut FramePolicy,
                            queued: &mut Vec<u64>,
                            owner_of: &HashMap<u64, ProcessId>,
                            in_flight: &mut VecDeque<BTreeSet<ProcessId>>,
                            sealed: &mut Vec<[usize; 3]>,
                            lanes: &Vec<Lane>| {
                while in_flight.len() < DEPTH {
                    let ids: QueuedFireIds = queued.iter().copied().collect();
                    let plan = policy.plan_dispatch(
                        &ids,
                        &HashSet::new(),
                        !in_flight.is_empty(),
                        Instant::now(),
                    );
                    let FramePlan::Dispatch(waves) = plan else {
                        break;
                    };
                    let fires: Vec<u64> = waves.into_iter().flatten().collect();
                    queued.retain(|fire| !fires.contains(fire));
                    let members: BTreeSet<ProcessId> =
                        fires.iter().map(|fire| owner_of[fire]).collect();
                    let mut kinds = [0usize; 3];
                    for lane in lanes.iter().filter(|lane| members.contains(&lane.id)) {
                        kinds[lane.index()] += 1;
                    }
                    sealed.push(kinds);
                    policy.on_frame_posted(members.iter().copied(), Instant::now());
                    in_flight.push_back(members);
                }
            };

            for _ in 0..16 {
                submit(
                    &mut lanes,
                    &mut policy,
                    &mut queued,
                    &mut owner_of,
                    &mut next_fire,
                );
                schedule(
                    &mut policy,
                    &mut queued,
                    &owner_of,
                    &mut in_flight,
                    &mut sealed,
                    &lanes,
                );
                // The device runs the oldest frame: its takes free the
                // channel lanes' cells, and the epilogue lanes harvest it.
                let members = in_flight
                    .pop_front()
                    .expect("a frame is in flight: the gate never stalls every kind");
                policy.on_frame_retired(members.iter().copied());
                for lane in lanes.iter_mut().filter(|lane| members.contains(&lane.id)) {
                    lane.outstanding -= 1;
                    match &mut lane.kind {
                        Kind::Channel(_, head) => *head += K,
                        Kind::Epilogue(left) => *left = EPILOGUE,
                        Kind::Constant => {}
                    }
                }
                // The scheduler plans while the epilogues compute, the
                // other kinds' next submits landing meanwhile.
                for pass in 0..EPILOGUE {
                    schedule(
                        &mut policy,
                        &mut queued,
                        &owner_of,
                        &mut in_flight,
                        &mut sealed,
                        &lanes,
                    );
                    for lane in lanes.iter_mut() {
                        if let Kind::Epilogue(left) = &mut lane.kind {
                            *left = left.saturating_sub(1);
                        }
                    }
                    if pass == 0 {
                        submit(
                            &mut lanes,
                            &mut policy,
                            &mut queued,
                            &mut owner_of,
                            &mut next_fire,
                        );
                    }
                }
            }
            assert!(sealed.len() >= 8, "capacity {capacity}: frames {sealed:?}");
            assert!(
                sealed.iter().all(|&kinds| kinds == [EACH; 3]),
                "capacity {capacity}: every frame seals with every lane of every kind \
                 (constant, channel, epilogue): {sealed:?}"
            );
        }
    }

    #[test]
    fn frame_every_case() {
        seals_complete_lanes_and_orders_waves_by_slot();
        incomplete_lane_holds_the_seal_until_it_completes();
        sealed_frame_dispatches_whole_and_frames_overlap();
        a_fire_racing_the_suspend_seals_alone_without_rejoining_the_wait_set();
        a_lane_parked_mid_frame_seals_what_it_submitted();
        truncated_frame_seals_with_submitted_fires_only();
        graceful_close_releases_the_wait();
        no_prompt_waits_for_a_lane_off_a_prompt();
        consumed_release_leaves_no_phantom_hold_for_bystander();
        a_burst_of_cut_prefills_seals_its_pieces_step_by_step();
        a_wave_reads_out_no_more_than_the_engine_carves();
        a_policys_bound_packs_prompts_into_a_frame_up_to_it_and_no_further();
        a_frame_goes_without_a_lane_whose_process_waits();
        a_lane_whose_guest_waits_on_the_outside_holds_no_frame_and_rejoins_on_its_submit();
        a_prompt_piece_cut_under_a_cap_since_lowered_waits_whole_and_owes_nothing();
    }

    /// A prompt split across a lower: its last piece, cut under the cap
    /// before it came down, is kept whole out of the frames sealed under
    /// the lowered cap (the frame goes with the other lanes), and its lane
    /// owes nothing meanwhile; the cap back, the piece goes and completes.
    /// A frame sealed under the old cap goes whole, and the rung is not
    /// lowered under it until it has.
    fn a_prompt_piece_cut_under_a_cap_since_lowered_waits_whole_and_owes_nothing() {
        let mut policy =
            FramePolicy::new(1, 64, 4096, None).with_silence_timeout_off();
        let (prompt, decode) = (pid(), pid());
        policy.cap_fires(128);
        // The prompt's first piece (64 rows) and a decode, one frame.
        policy.on_fire_enqueued(stamp(prompt, 0, 0, 1), Some(prompt), 1, 64, 1);
        policy.on_fire_enqueued(stamp(decode, 0, 0, 1), Some(decode), 2, 1, 1);
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        let mut first = fires(&plan(&mut policy, &queued, Instant::now()));
        first.sort_unstable();
        assert_eq!(first, vec![1, 2]);
        policy.on_frame_posted([prompt, decode], Instant::now());
        policy.on_frame_retired([prompt, decode]);

        // Its last piece (33 rows), cut under 128, arrives; the cap comes
        // down to 32 before it is sealed.
        policy.cap_fires(32);
        policy.on_fire_enqueued(stamp(prompt, 1, 0, 1), Some(prompt), 3, 33, 1);
        policy.on_fire_enqueued(stamp(decode, 1, 0, 1), Some(decode), 4, 1, 1);
        let queued: QueuedFireIds = [3, 4].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![4],
            "the frame goes without the piece past the cap"
        );
        assert!(!policy.in_flight_lanes.contains(&prompt), "it owes nothing");
        assert_eq!(policy.sealed_widest(), 0);
        policy.on_frame_posted([decode], Instant::now());
        policy.on_frame_retired([decode]);

        // The cap back (the queued rows ask it): the piece goes.
        policy.cap_fires(128);
        let queued: QueuedFireIds = [3].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())), vec![3]);
    }

    /// A guest blocked on input from outside (a session receive, a timer,
    /// an HTTP response) is parked by the host: the frames go without its
    /// lane, a frame it submitted before it blocked still goes, and its
    /// next submit brings the lane back, awaited like any other.
    fn a_lane_whose_guest_waits_on_the_outside_holds_no_frame_and_rejoins_on_its_submit() {
        let mut policy = FramePolicy::new(1, 64, 4096, None).with_silence_timeout_off();
        let (runner, receiver) = (pid(), pid());
        policy.on_fire_enqueued(stamp(runner, 0, 0, 1), Some(runner), 1, 1, 1);
        policy.on_fire_enqueued(stamp(receiver, 0, 0, 1), Some(receiver), 2, 1, 1);
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())).len(), 2);
        policy.on_frame_retired([runner, receiver]);

        // The receiver runs a frame ahead, then its guest blocks on a
        // session receive; the frame it submitted still goes.
        policy.on_fire_enqueued(stamp(receiver, 1, 0, 1), Some(receiver), 3, 1, 1);
        policy.on_owner_away(receiver);
        policy.on_fire_enqueued(stamp(runner, 1, 0, 1), Some(runner), 4, 1, 1);
        let queued: QueuedFireIds = [3, 4].into_iter().collect();
        let mut both = fires(&plan(&mut policy, &queued, Instant::now()));
        both.sort_unstable();
        assert_eq!(both, vec![3, 4], "what it submitted before it blocked goes");
        policy.on_frame_retired([runner, receiver]);

        // The runner submits: the frame goes without the waiting receiver.
        policy.on_fire_enqueued(stamp(runner, 2, 0, 1), Some(runner), 5, 1, 1);
        let queued: QueuedFireIds = [5].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![5],
            "the frame goes without the lane whose guest waits on the outside"
        );
        policy.on_frame_retired([runner]);
        // It stays out while it waits: the runner's frames go alone.
        policy.on_fire_enqueued(stamp(runner, 3, 0, 1), Some(runner), 6, 1, 1);
        let queued: QueuedFireIds = [6].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())), vec![6]);
        policy.on_frame_retired([runner]);

        // The message comes; its guest submits: awaited again.
        policy.on_fire_enqueued(stamp(receiver, 2, 0, 1), Some(receiver), 7, 1, 1);
        let queued: QueuedFireIds = [7].into_iter().collect();
        assert!(
            matches!(plan(&mut policy, &queued, Instant::now()), FramePlan::Hold(_)),
            "back, it waits for the runner, which is awaited"
        );
        policy.on_fire_enqueued(stamp(runner, 4, 0, 1), Some(runner), 8, 1, 1);
        let queued: QueuedFireIds = [7, 8].into_iter().collect();
        let mut both = fires(&plan(&mut policy, &queued, Instant::now()));
        both.sort_unstable();
        assert_eq!(both, vec![7, 8], "it rides the next frame");
        policy.on_frame_retired([runner, receiver]);

        // Computing again, it is awaited again.
        policy.on_fire_enqueued(stamp(runner, 5, 0, 1), Some(runner), 9, 1, 1);
        let queued: QueuedFireIds = [9].into_iter().collect();
        assert!(
            matches!(plan(&mut policy, &queued, Instant::now()), FramePlan::Hold(_)),
            "its lane is awaited while its guest computes"
        );
    }

    /// A process whose bind found no room for its guest state (or whose
    /// pass waits for a raise of the activation rung) submits nothing
    /// until it is through: the frame its lane was awaited in
    /// goes without it (the soak's 56-lane frame held 6 minutes on one
    /// process stuck at bind). Its next submit brings the lane back.
    fn a_frame_goes_without_a_lane_whose_process_waits() {
        let mut policy =
            FramePolicy::new(1, 64, 4096, None).with_silence_timeout_off();
        let (runner, binder) = (pid(), pid());
        policy.on_fire_enqueued(stamp(runner, 0, 0, 1), Some(runner), 1, 1, 1);
        policy.on_fire_enqueued(stamp(binder, 0, 0, 1), Some(binder), 2, 1, 1);
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())).len(), 2);
        policy.on_frame_retired([runner, binder]);

        // The binder's process binds another instance; the runner submits.
        policy.on_fire_enqueued(stamp(runner, 1, 0, 1), Some(runner), 3, 1, 1);
        let queued: QueuedFireIds = [3].into_iter().collect();
        policy.on_bind_enqueued(Some(binder));
        policy.on_bind_completed(Some(binder));
        let t0 = Instant::now();
        assert!(
            matches!(plan(&mut policy, &queued, t0), FramePlan::Hold(_)),
            "control: the binder's lane is awaited"
        );
        // The bind is short of guest-state room and waits for it.
        policy.on_owner_waits(binder);
        assert_eq!(
            fires(&plan(&mut policy, &queued, t0)),
            vec![3],
            "the frame goes without the waiting binder"
        );
        policy.on_frame_retired([runner]);

        // Bound at last, it submits: its lane is awaited again.
        policy.on_fire_enqueued(stamp(binder, 1, 0, 1), Some(binder), 4, 1, 1);
        policy.on_fire_enqueued(stamp(runner, 2, 0, 1), Some(runner), 5, 1, 1);
        let queued: QueuedFireIds = [4, 5].into_iter().collect();
        let mut both = fires(&plan(&mut policy, &queued, Instant::now()));
        both.sort_unstable();
        assert_eq!(both, vec![4, 5], "it rides the next frame");
        policy.on_frame_retired([runner, binder]);

        // Mid-prompt, its pass waits for a raise of the activation rung:
        // the frame goes with the runner alone, as for the bind.
        policy.on_fire_enqueued(stamp(runner, 3, 0, 1), Some(runner), 6, 1, 1);
        let queued: QueuedFireIds = [6].into_iter().collect();
        assert!(
            matches!(plan(&mut policy, &queued, t0), FramePlan::Hold(_)),
            "control: the prompting lane is awaited"
        );
        policy.on_owner_waits(binder);
        assert_eq!(fires(&plan(&mut policy, &queued, t0)), vec![6]);
    }

    /// Prompts past half a fire's tokens take a frame each, the lane's
    /// first decode row riding its prompt's, while no policy bounds a batch.
    /// With the worker policy's bound standing they share a frame up to it
    /// and no further, on either side of the engine's fire; a prompt past
    /// the bound alone still takes a frame; and with the bound withdrawn a
    /// frame is one fire's again.
    fn a_policys_bound_packs_prompts_into_a_frame_up_to_it_and_no_further() {
        const LANES: usize = 64;
        const FIRE: usize = 5120;
        // The lanes each sealed frame took, as the frames retire in turn.
        fn frames(policy: &mut FramePolicy, lanes: usize, prompt: usize) -> Vec<usize> {
            let ids: Vec<ProcessId> = (0..lanes).map(|_| pid()).collect();
            for (at, &lane) in ids.iter().enumerate() {
                let at = at as u64;
                policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 1000 + at, prompt, 1);
                policy.on_fire_enqueued(stamp(lane, 0, 1, 2), Some(lane), 2000 + at, 1, 1);
            }
            let every: Vec<u64> = (0..lanes as u64)
                .flat_map(|at| [1000 + at, 2000 + at])
                .collect();
            let mut sealed: Vec<u64> = Vec::new();
            let mut sizes = Vec::new();
            while sealed.len() < every.len() {
                let queued: QueuedFireIds = every
                    .iter()
                    .copied()
                    .filter(|id| !sealed.contains(id))
                    .collect();
                let FramePlan::Dispatch(waves) = plan(policy, &queued, Instant::now()) else {
                    panic!("prompts of {prompt} tokens: expected a whole-frame dispatch");
                };
                assert_eq!(waves.len(), 2);
                let mut prompts: Vec<u64> = waves[0].iter().map(|id| id - 1000).collect();
                let mut decodes: Vec<u64> = waves[1].iter().map(|id| id - 2000).collect();
                prompts.sort_unstable();
                decodes.sort_unstable();
                assert_eq!(
                    decodes, prompts,
                    "a lane's first decode row rides the wave after its prompt"
                );
                sizes.push(prompts.len());
                sealed.extend(waves.iter().flatten().copied());
                policy.on_frame_retired(prompts.iter().map(|&at| ids[at as usize]));
            }
            sizes
        }
        let long = FIRE / 2 + 1;

        let mut policy = FramePolicy::new(2, LANES, FIRE, None);
        assert_eq!(frames(&mut policy, 6, long), vec![1; 6], "a frame a prompt");

        // Four prompts' tokens: four a frame, the fifth waits for the next.
        policy.bound_batch(Some(4 * long));
        assert_eq!(frames(&mut policy, 10, long), vec![4, 4, 2]);
        // One token short of four holds three.
        policy.bound_batch(Some(4 * long - 1));
        assert_eq!(frames(&mut policy, 7, long), vec![3, 3, 1]);
        // The engine's lanes bound a frame whatever the tokens stated.
        policy.bound_batch(Some(u32::MAX as usize));
        assert_eq!(frames(&mut policy, LANES + 6, long), vec![LANES, 6]);
        // A bound under one fire holds short prompts to it too.
        policy.bound_batch(Some(64));
        assert_eq!(frames(&mut policy, 5, 32), vec![2, 2, 1]);
        // A prompt past the bound alone takes a frame of its own.
        assert_eq!(frames(&mut policy, 3, long), vec![1; 3]);
        policy.bound_batch(Some(0));
        assert_eq!(frames(&mut policy, 3, 32), vec![1; 3]);

        // Withdrawn, a frame is one fire's tokens again.
        policy.bound_batch(None);
        assert_eq!(frames(&mut policy, 4, long), vec![1; 4]);
        assert_eq!(frames(&mut policy, 6, FIRE / 4), vec![4, 2]);
    }

    /// Lanes that read out several rows each (a verified draft block) share
    /// a wave only within the engine's readout ceiling; the rest wait a
    /// frame instead of the fire being refused, and a fire past the ceiling
    /// alone still takes a wave of its own.
    fn a_wave_reads_out_no_more_than_the_engine_carves() {
        let mut policy = FramePolicy::new(1, 64, 4096, None).with_max_wave_readouts(8);
        let lanes: Vec<ProcessId> = (0..3).map(|_| pid()).collect();
        for (at, &lane) in lanes.iter().enumerate() {
            policy.on_fire_reading(stamp(lane, 0, 0, 1), Some(lane), 10 + at as u64, 5, 1, 5, 5, 0);
        }
        let mut sealed_ids = Vec::new();
        for _ in 0..3 {
            let queued: QueuedFireIds = (10..13).filter(|id| !sealed_ids.contains(id)).collect();
            let sealed = fires(&plan(&mut policy, &queued, Instant::now()));
            assert_eq!(
                sealed.len(),
                1,
                "two lanes of five readouts each pass a ceiling of eight"
            );
            sealed_ids.extend(sealed);
            policy.on_frame_retired(lanes.iter().copied());
        }
        sealed_ids.sort_unstable();
        assert_eq!(sealed_ids, vec![10, 11, 12], "every lane seals in its turn");

        let lone = pid();
        policy.on_fire_reading(stamp(lone, 0, 0, 1), Some(lone), 20, 12, 1, 12, 12, 0);
        let queued: QueuedFireIds = [20].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![20],
            "a fire past the ceiling alone takes a wave of its own"
        );
    }

    /// Concurrent prompts whose prefills are cut at checkpoints, each lane's
    /// pieces submitted together: a lane's pieces fill the steps of its
    /// frames in order (`piece_frames`), so the pieces of every lane seal
    /// into one frame, each piece in its own step, not one unframed launch
    /// at a time.
    fn a_burst_of_cut_prefills_seals_its_pieces_step_by_step() {
        const LANES: usize = 8;
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lanes: Vec<ProcessId> = (0..LANES).map(|_| pid()).collect();
        let mut ids = Vec::new();
        for (at, &lane) in lanes.iter().enumerate() {
            let mut seq = 0u64;
            let framed =
                crate::pipeline::fire::piece_frames(Some(stamp(lane, 0, 0, 1)), 2, 2, || {
                    seq += 1;
                    seq
                });
            assert_eq!(
                framed,
                vec![Some(stamp(lane, 0, 0, 2)), Some(stamp(lane, 0, 1, 2))],
                "two pieces are one frame's two steps"
            );
            for (piece, frame) in framed.into_iter().enumerate() {
                let id = 100 + 2 * at as u64 + piece as u64;
                policy.on_fire_enqueued(frame.expect("framed"), Some(lane), id, 128, 1);
                ids.push(id);
            }
        }
        let queued: QueuedFireIds = ids.iter().copied().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected a whole-frame dispatch");
        };
        for (step, wave) in waves.iter().enumerate() {
            let mut wave = wave.clone();
            wave.sort_unstable();
            let want: Vec<u64> = (0..LANES as u64)
                .map(|at| 100 + 2 * at + step as u64)
                .collect();
            assert_eq!(wave, want, "step {step} holds piece {step} of every lane");
        }

        let mut seq = 0u64;
        let three =
            crate::pipeline::fire::piece_frames(Some(stamp(lanes[0], 4, 0, 1)), 3, 2, || {
                seq += 1;
                seq
            });
        assert_eq!(
            three,
            vec![
                Some(stamp(lanes[0], 4, 0, 2)),
                Some(stamp(lanes[0], 4, 1, 2)),
                Some(stamp(lanes[0], 1, 0, 1)),
            ],
            "a third piece opens the next frame, alone"
        );

        let shared =
            crate::pipeline::fire::piece_frames(Some(stamp(lanes[0], 9, 1, 2)), 3, 2, || 0);
        assert_eq!(
            shared,
            vec![None, None, Some(stamp(lanes[0], 9, 1, 2))],
            "a shared frame keeps its last piece"
        );
    }

    fn seals_complete_lanes_and_orders_waves_by_slot() {
        let mut policy = FramePolicy::new(4, 64, 4096, None);
        let (a, b) = (pid(), pid());
        for slot in 0..4 {
            policy.on_fire_enqueued(stamp(a, 0, slot, 4), Some(a), 100 + slot as u64, 1, 1);
        }
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 200, 37, 1);

        let queued: QueuedFireIds = [100, 101, 102, 103, 200].into_iter().collect();
        let sealed = plan(&mut policy, &queued, Instant::now());
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

    fn incomplete_lane_holds_the_seal_until_it_completes() {
        let mut policy =
            FramePolicy::new(2, 64, 4096, None).with_silence_timeout_off();
        let (fast, slow) = (pid(), pid());
        policy.on_fire_enqueued(stamp(fast, 0, 0, 2), Some(fast), 1, 1, 1);
        policy.on_fire_enqueued(stamp(fast, 0, 1, 2), Some(fast), 2, 1, 1);
        policy.on_fire_enqueued(stamp(slow, 0, 0, 2), Some(slow), 3, 1, 1);

        let queued: QueuedFireIds = [1, 2, 3].into_iter().collect();
        let t0 = Instant::now();
        match plan(&mut policy, &queued, t0) {
            FramePlan::Hold(hold) => {
                assert_eq!(hold, Duration::from_micros(GATHER_POLL_US));
                assert!(hold < Duration::from_micros(STRICT_WATCHDOG_US));
            }
            plan => panic!("wait-all must hold for the incomplete lane, got {plan:?}"),
        }
        match plan(&mut policy, &queued, t0 + Duration::from_secs(60)) {
            FramePlan::Hold(_) => {}
            plan => panic!("the watchdog reports, it must not fire: got {plan:?}"),
        }

        policy.on_fire_enqueued(stamp(slow, 0, 1, 2), Some(slow), 4, 1, 1);
        let queued: QueuedFireIds = [1, 2, 3, 4].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("all lanes ready: the epoch must seal");
        };
        assert_eq!(waves[0].len(), 2, "dense wave 0 holds BOTH lanes");
        assert!(waves[0].contains(&1) && waves[0].contains(&3));
    }

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

    fn a_fire_racing_the_suspend_seals_alone_without_rejoining_the_wait_set() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (victim, healthy) = {
            let (x, y) = (pid(), pid());
            if x < y { (x, y) } else { (y, x) }
        };
        policy.on_fire_enqueued(stamp(victim, 0, 0, 2), Some(victim), 100, 1, 1);
        policy.on_fire_enqueued(stamp(victim, 0, 1, 2), Some(victim), 101, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 0, 0, 2), Some(healthy), 102, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 0, 1, 2), Some(healthy), 103, 1, 1);
        let queued: QueuedFireIds = [100, 101, 102, 103].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, Instant::now()),
            FramePlan::Dispatch(_)
        ));

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

        policy.on_process_resume(victim);
        policy.on_fire_enqueued(stamp(victim, 1, 1, 2), Some(victim), 201, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 2, 0, 2), Some(healthy), 302, 1, 1);
        policy.on_fire_enqueued(stamp(healthy, 2, 1, 2), Some(healthy), 303, 1, 1);
        let queued: QueuedFireIds = [201, 302, 303].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("the late slot must seal");
        };
        assert_eq!(waves[1], vec![201, 303], "the late slot keeps its wave");

        policy.on_fire_enqueued(stamp(victim, 2, 0, 2), Some(victim), 400, 1, 1);
        policy.on_fire_enqueued(stamp(victim, 2, 1, 2), Some(victim), 401, 1, 1);
        assert!(
            policy.lanes[&victim].awaited,
            "a resumed process rejoins on its next frame"
        );
    }

    fn a_lane_parked_mid_frame_seals_what_it_submitted() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 10, 1, 1);
        policy.on_lane_leave(lane, Some(lane), false);
        let queued: QueuedFireIds = [10].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("a parked lane's submitted slot must still seal");
        };
        assert_eq!(waves[0], vec![10]);
        assert!(waves[1].is_empty());
    }

    fn truncated_frame_seals_with_submitted_fires_only() {
        let mut policy = FramePolicy::new(4, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 4), Some(lane), 30, 1, 1);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 4), Some(lane), 31, 1, 1);
        policy.on_frame_truncated(lane, 0, 2);
        let queued: QueuedFireIds = [30, 31].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("truncated frame must still seal");
        };
        assert_eq!(waves[0], vec![30]);
        assert_eq!(waves[1], vec![31]);
        assert!(waves[2].is_empty() && waves[3].is_empty());
    }

    fn graceful_close_releases_the_wait() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 90, 1, 1);
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 91, 1, 1);
        let queued: QueuedFireIds = [90, 91].into_iter().collect();
        let bootstrap = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&bootstrap).len(), 2);

        policy.on_fire_enqueued(stamp(b, 1, 0, 1), Some(b), 92, 1, 1);
        let queued: QueuedFireIds = [92].into_iter().collect();
        match plan(&mut policy, &queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("the gather must block on lane a, got {plan:?}"),
        }
        policy.on_lane_leave(a, None, false);
        let next = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&next), vec![92]);
    }

    fn no_prompt_waits_for_a_lane_off_a_prompt() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b, c) = (pid(), pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 90, 512, 1);
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 91, 1, 1);
        let queued: QueuedFireIds = [90, 91].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())).len(), 2);

        policy.on_fire_enqueued(stamp(b, 1, 0, 1), Some(b), 92, 1, 1);
        policy.on_fire_enqueued(stamp(c, 0, 0, 1), Some(c), 93, 256, 1);
        let queued: QueuedFireIds = [92, 93].into_iter().collect();
        let mut sealed =
            fires(&policy.plan_dispatch(&queued, &HashSet::new(), true, Instant::now()));
        sealed.sort_unstable();
        assert_eq!(
            sealed,
            vec![92, 93],
            "c's prompt seals behind a's on the card, b's decode rides beside it"
        );
        policy.on_frame_retired([a, b]);
        policy.on_fire_enqueued(stamp(b, 2, 0, 1), Some(b), 94, 1, 1);
        let queued: QueuedFireIds = [94].into_iter().collect();
        assert!(
            matches!(
                plan(&mut policy, &queued, Instant::now()),
                FramePlan::Hold(_)
            ),
            "with no prompt to ride, b's decode waits for a's next pass"
        );
    }

    fn consumed_release_leaves_no_phantom_hold_for_bystander() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let executing = pid();
        let bystander = pid();
        policy.on_bind_enqueued(Some(executing));
        policy.on_bind_completed(Some(executing));
        policy.on_execution_slot_consumed(executing);
        policy.on_fire_enqueued(stamp(executing, 0, 0, 1), Some(executing), 95, 1, 1);
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
}
