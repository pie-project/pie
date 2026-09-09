use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

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

fn idle_dump_min_seats() -> usize {
    static SEATS: OnceLock<usize> = OnceLock::new();
    *SEATS.get_or_init(|| {
        std::env::var("PIE_IDLE_DUMP_MIN_SEATS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
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

static SEAL_DEFAULT_READY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_seal_default_ready(ready: bool) {
    SEAL_DEFAULT_READY.store(ready, Ordering::Relaxed);
}

static SEAL_COALESCE_DEFAULT_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(crate) fn set_seal_coalesce_default(window: Duration) {
    SEAL_COALESCE_DEFAULT_US.store(window.as_micros() as u64, Ordering::Relaxed);
}

fn seal_coalesce() -> Duration {
    static CONFIGURED: OnceLock<Duration> = OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        std::env::var("PIE_SEAL_COALESCE_US")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .map_or_else(
                || Duration::from_micros(SEAL_COALESCE_DEFAULT_US.load(Ordering::Relaxed)),
                Duration::from_micros,
            )
    })
}

fn seal_mode_ready() -> bool {
    static CONFIGURED: OnceLock<bool> = OnceLock::new();
    *CONFIGURED.get_or_init(|| match std::env::var("PIE_SEAL_MODE") {
        Ok(value) => match value.trim() {
            "strict" => false,
            "ready" => true,
            other => {
                tracing::warn!(
                    value = other,
                    "PIE_SEAL_MODE must be \"ready\" or \"strict\"; \
                     ignoring and keeping the default (ready)"
                );
                true
            }
        },
        Err(_) => SEAL_DEFAULT_READY.load(Ordering::Relaxed),
    })
}

fn gate_contributed() -> bool {
    static CONFIGURED: OnceLock<bool> = OnceLock::new();
    *CONFIGURED
        .get_or_init(|| std::env::var("PIE_GATE_CONTRIBUTED").is_ok_and(|value| value == "1"))
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

struct ArrivedFire {
    slot: u32,
    fire_id: Option<u64>,
    tokens: usize,
    rows: usize,
}

struct PlacedFire {
    wave: usize,
    fire_id: u64,
    rows: usize,
    tokens: usize,
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
    leashed: bool,
    clock_from: Option<Instant>,
    frames: VecDeque<PendingFrame>,
    group: Option<u32>,
    fired_this_boundary: bool,
}

#[derive(Debug)]
struct Cohort {
    expected: u32,
    arrived: BTreeSet<ProcessId>,
    since: Instant,
    condemned: bool,
}

impl Cohort {
    fn complete(&self) -> bool {
        self.arrived.len() as u64 >= u64::from(self.expected)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Doom {
    Abandoned,
    CohortNeverComplete {
        group: u32,
        expected: u32,
        arrived: u32,
    },
}

impl std::fmt::Display for Doom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Abandoned => f.write_str(
                "pipeline abandoned: silent past the silence timeout without submitting \
                 and without calling forward.park()",
            ),
            Self::CohortNeverComplete {
                group,
                expected,
                arrived,
            } => write!(
                f,
                "attention group {group} never composed: {expected} live forward passes name \
                 the group and only {arrived} of them reached the runtime on a lane of its \
                 own before the silence timeout. A group attends inside ONE fire, so the \
                 runtime will not fire it short — every forward pass that names a group must \
                 submit into the group's next frame, each down a pipeline of its own (a \
                 pipeline is serial, so two of a group's passes submitted to ONE pipeline \
                 arrive as one lane and count once; a pass that means to sit out must not \
                 name the group, and dropping it is enough to leave)"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateVerdict {
    Satisfied,
    Blocking,
    SkippedOwed,
}

impl LaneState {
    fn group_key(&self) -> Option<(ProcessId, u32)> {
        self.owner.zip(self.group)
    }

    fn owes_its_group(&self) -> bool {
        (self.awaited || self.leashed)
            && !self.frames.front().is_some_and(PendingFrame::is_complete)
    }

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
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum FramePlan {
    Dispatch(Vec<Vec<u64>>),
    Hold(Duration),
    Park,
    Terminate(Vec<(ProcessId, Doom)>),
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

pub(super) struct FramePolicy {
    k: usize,
    max_wave_tokens: usize,
    max_wave_rows: usize,
    lanes: BTreeMap<ProcessId, LaneState>,
    sealed: VecDeque<SealedFrame>,
    executing_now: bool,
    last_retire_at: Option<Instant>,
    idle_dumped: bool,
    pending_binds: BTreeMap<ProcessId, usize>,
    cohorts: BTreeMap<(ProcessId, u32), Cohort>,
    staged: BTreeSet<ProcessId>,
    pending_slots: u64,
    joins_in_flight: BTreeSet<ProcessId>,
    admission_queue: VecDeque<ProcessId>,
    slotted: BTreeSet<ProcessId>,
    departing: BTreeSet<ProcessId>,
    suspended: BTreeSet<ProcessId>,
    truncated_seqs: BTreeMap<ProcessId, u64>,
    strict_watchdog_deadline: Option<Instant>,
    gather_seq: u64,
    quiesce_mark: Option<u64>,
    in_flight_lanes: BTreeSet<ProcessId>,
    in_flight_since: BTreeMap<ProcessId, Instant>,
    submit_deadline: Duration,
    silence_timeout: Duration,
    gate_contributed: bool,
    seal_mode_ready: bool,
    seal_coalesce: Duration,
    coalesce_since: Option<Instant>,
    stats: Option<Arc<SchedulerStats>>,
}

impl FramePolicy {
    #[cfg(test)]
    fn with_submit_deadline(mut self, deadline: Duration) -> Self {
        self.submit_deadline = deadline;
        self.silence_timeout = Duration::from_secs(86_400);
        self
    }

    #[cfg(test)]
    fn with_silence_timeout(mut self, timeout: Duration) -> Self {
        self.silence_timeout = timeout;
        self
    }

    #[cfg(test)]
    fn with_seal_mode_ready(mut self, on: bool) -> Self {
        self.seal_mode_ready = on;
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
            executing_now: false,
            last_retire_at: None,
            idle_dumped: false,
            pending_binds: BTreeMap::new(),
            cohorts: BTreeMap::new(),
            staged: BTreeSet::new(),
            pending_slots: 0,
            joins_in_flight: BTreeSet::new(),
            admission_queue: VecDeque::new(),
            slotted: BTreeSet::new(),
            departing: BTreeSet::new(),
            suspended: BTreeSet::new(),
            truncated_seqs: BTreeMap::new(),
            strict_watchdog_deadline: None,
            gather_seq: 0,
            quiesce_mark: None,
            in_flight_lanes: BTreeSet::new(),
            in_flight_since: BTreeMap::new(),
            submit_deadline: crate::scheduler::configured_submit_deadline(),
            silence_timeout: crate::scheduler::configured_silence_timeout(),
            gate_contributed: gate_contributed(),
            seal_mode_ready: seal_mode_ready(),
            seal_coalesce: seal_coalesce(),
            coalesce_since: None,
            stats,
        }
    }

    pub fn single_slot(&self) -> bool {
        self.k == 1
    }

    pub fn on_fire_enqueued(
        &mut self,
        stamp: FrameStamp,
        owner: Option<ProcessId>,
        fire_id: u64,
        tokens: usize,
        rows: usize,
        cohort: Option<(u32, u32)>,
    ) {
        let accept_began = self.stats.is_some().then(Instant::now);
        self.gather_cohort(stamp.lane, owner, cohort);
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
        if let Some((group, _)) = cohort
            && let Some(lane) = self.lanes.get_mut(&stamp.lane)
        {
            lane.group = Some(group);
        }
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
            },
        );
    }

    fn gather_cohort(
        &mut self,
        lane: ProcessId,
        owner: Option<ProcessId>,
        cohort: Option<(u32, u32)>,
    ) {
        let (Some(owner), Some((group, expected))) = (owner, cohort) else {
            return;
        };
        let entry = self
            .cohorts
            .entry((owner, group))
            .or_insert_with(|| Cohort {
                expected,
                arrived: BTreeSet::new(),
                since: Instant::now(),
                condemned: false,
            });
        entry.expected = entry.expected.max(expected);
        entry.arrived.insert(lane);
        if entry.complete() {
            self.cohorts.remove(&(owner, group));
        }
    }

    fn cohort_verdict(&mut self, now: Instant) -> Vec<(ProcessId, Doom)> {
        if self.cohorts.is_empty() {
            return Vec::new();
        }
        let timeout = self.silence_timeout;
        let mut doomed = Vec::new();
        for ((owner, group), cohort) in self.cohorts.iter_mut() {
            if cohort.condemned || now.saturating_duration_since(cohort.since) < timeout {
                continue;
            }
            cohort.condemned = true;
            doomed.push((
                *owner,
                Doom::CohortNeverComplete {
                    group: *group,
                    expected: cohort.expected,
                    arrived: cohort.arrived.len() as u32,
                },
            ));
        }
        doomed
    }

    fn group_assembling(&self) -> bool {
        !self.cohorts.is_empty()
            || self
                .lanes
                .values()
                .any(|lane| !lane.frames.is_empty() && self.group_short(lane))
    }

    fn group_short(&self, lane: &LaneState) -> bool {
        let Some(key) = lane.group_key() else {
            return false;
        };
        if self.cohorts.contains_key(&key) {
            return true;
        }
        self.lanes
            .values()
            .any(|other| other.group_key() == Some(key) && other.owes_its_group())
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
            leashed: false,
            clock_from: None,
            frames: VecDeque::new(),
            group: None,
            fired_this_boundary: true,
        });
        if lane.owner.is_none() {
            lane.owner = owner;
        }
        if lane.parked && !suspended {
            lane.parked = false;
            lane.awaited = true;
        }
        lane.leashed = false;
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
        for state in self.lanes.values_mut() {
            while state.frames.front().is_some_and(|frame| frame.park) {
                state.frames.pop_front();
                if state.frames.is_empty() {
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
            if !self.lanes.values().any(|lane| lane.owner == Some(pid)) {
                self.staged.insert(pid);
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

    pub fn on_lane_leave(&mut self, lane: ProcessId, owner: Option<ProcessId>, purge_queued: bool) {
        let owner = owner.or_else(|| self.lanes.get(&lane).and_then(|state| state.owner));
        if purge_queued {
            self.lanes.remove(&lane);
            self.truncated_seqs.remove(&lane);
            self.in_flight_lanes.remove(&lane);
        } else if let Some(state) = self.lanes.get_mut(&lane) {
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
            self.cohorts.retain(|(who, _), _| *who != owner);
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
        self.cohorts.retain(|(who, _), _| *who != owner);
        self.suspended.remove(&owner);
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

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
        self.cohorts.retain(|(who, _), _| *who != owner);
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
        self.lanes.values().any(|lane| {
            lane.frames.front().is_some_and(PendingFrame::is_complete) && !self.group_short(lane)
        })
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
                if self.group_short(lane) {
                    continue;
                }
                if mid_boundary && lane.fired_this_boundary {
                    continuing.push(*lane_id);
                } else {
                    fresh.push(*lane_id);
                }
            }
            if mid_boundary && fresh.is_empty() {
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
            let mut units: Vec<Vec<ProcessId>> = Vec::with_capacity(order.len());
            let mut placed: BTreeSet<ProcessId> = BTreeSet::new();
            for lane_id in &order {
                if !placed.insert(*lane_id) {
                    continue;
                }
                match self.lanes.get(lane_id).and_then(LaneState::group_key) {
                    None => units.push(vec![*lane_id]),
                    Some(key) => {
                        let unit: Vec<ProcessId> = order
                            .iter()
                            .copied()
                            .filter(|id| {
                                self.lanes.get(id).and_then(LaneState::group_key) == Some(key)
                            })
                            .collect();
                        placed.extend(unit.iter().copied());
                        units.push(unit);
                    }
                }
            }
            for unit in units {
                let mut unit_fires: Vec<(ProcessId, Vec<PlacedFire>)> =
                    Vec::with_capacity(unit.len());
                let mut unit_rows = vec![0usize; k];
                let mut unit_tokens = vec![0usize; k];
                let mut short = false;
                for lane_id in &unit {
                    let Some(lane) = self.lanes.get_mut(lane_id) else {
                        short = true;
                        continue;
                    };
                    let Some(front) = lane.frames.front() else {
                        short = true;
                        continue;
                    };
                    let live: Vec<PlacedFire> = front
                        .fires
                        .iter()
                        .filter_map(|fire| {
                            fire.fire_id.map(|fire_id| PlacedFire {
                                wave: (fire.slot as usize).min(k - 1),
                                fire_id,
                                rows: fire.rows.max(1),
                                tokens: fire.tokens,
                            })
                        })
                        .collect();
                    if live.is_empty() {
                        lane.frames.pop_front();
                        dropped_empty = true;
                        short = true;
                        continue;
                    }
                    let fits = live.iter().all(|fire| {
                        wave_rows[fire.wave] + unit_rows[fire.wave] + fire.rows <= max_wave_rows
                            && wave_tokens[fire.wave] + unit_tokens[fire.wave] + fire.tokens
                                <= max_wave_tokens
                    });
                    if !fits {
                        short = true;
                        continue;
                    }
                    for fire in &live {
                        unit_rows[fire.wave] += fire.rows;
                        unit_tokens[fire.wave] += fire.tokens;
                    }
                    unit_fires.push((*lane_id, live));
                }
                if short {
                    continue;
                }
                for (lane_id, live) in unit_fires {
                    for fire in &live {
                        wave_rows[fire.wave] += fire.rows;
                        wave_tokens[fire.wave] += fire.tokens;
                        waves[fire.wave].push(fire.fire_id);
                        fire_waves.insert(fire.fire_id, fire.wave);
                    }
                    members.insert(lane_id);
                    let lane = self.lanes.get_mut(&lane_id).expect("measured lane exists");
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
                    lane.served = true;
                    lane.frames.pop_front();
                }
            }
            if fire_waves.is_empty() {
                if dropped_empty {
                    continue;
                }
                self.lanes
                    .retain(|_, lane| lane.awaited || lane.leashed || !lane.frames.is_empty());
                return None;
            }
            if super::worker::wave_trace() {
                let complete = self
                    .lanes
                    .values()
                    .filter(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete))
                    .count();
                let awaited = self.lanes.values().filter(|lane| lane.awaited).count();
                super::worker::wave_trace_emit(format!(
                    "[wave-trace] t={}us seal partition={} fresh={} continuing={} complete_left={} awaited={} lanes={} in_flight={} mid_boundary={mid_boundary} executing={}",
                    super::worker::wave_trace_us(),
                    members.len(),
                    fresh.len(),
                    continuing.len(),
                    complete,
                    awaited,
                    self.lanes.len(),
                    self.in_flight_lanes.len(),
                    self.executing_now
                ));
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
            self.record_seal_engagement();
            let _ = &fire_waves;
            self.sealed.push_back(SealedFrame {
                waves,
                members: members.iter().copied().collect(),
            });
            self.lanes
                .retain(|_, lane| lane.awaited || lane.leashed || !lane.frames.is_empty());
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
                self.in_flight_lanes.extend(frame.members.iter().copied());
                for member in frame.members.iter().copied() {
                    self.in_flight_since.entry(member).or_insert(now);
                }
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
            let doomed = self.cohort_verdict(now);
            if !doomed.is_empty() {
                return FramePlan::Terminate(doomed);
            }
            let mut missing = 0;
            let mut expired: Vec<ProcessId> = Vec::new();
            let leash = self.submit_deadline;
            let silence = self.silence_timeout;
            let contributed_relax = self.gate_contributed && executing;
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
                    if owes_forever {
                        lane.clock_from = debt_since;
                    }
                    match lane.clock_from {
                        Some(from) if now.saturating_duration_since(from) >= silence => {
                            lane.awaited = false;
                            lane.leashed = false;
                            lane.clock_from = None;
                            expired.push(lane.owner.unwrap_or(*lane_id));
                            continue;
                        }
                        Some(from) if blocking && now.saturating_duration_since(from) >= leash => {
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
                expired.sort_unstable();
                expired.dedup();
                return FramePlan::Terminate(
                    expired
                        .into_iter()
                        .map(|pid| (pid, Doom::Abandoned))
                        .collect(),
                );
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
            if missing > 0 {
                if executing {
                    self.quiesce_mark = None;
                    self.coalesce_since = None;
                    return FramePlan::Park;
                }
                let mut window_left: Option<Duration> = None;
                if self.seal_mode_ready && self.have_seal_candidate() {
                    let since = *self.coalesce_since.get_or_insert(now);
                    let open_for = now.saturating_duration_since(since);
                    let quiesced = self.quiesce_mark == Some(self.gather_seq);
                    if quiesced && open_for >= self.seal_coalesce {
                        self.quiesce_mark = None;
                        self.coalesce_since = None;
                        self.strict_watchdog_deadline = None;
                        self.idle_dumped = false;
                        match self.seal() {
                            Some(FramePlan::Dispatch(_)) => continue,
                            Some(plan) => return plan,
                            None => {}
                        }
                    } else {
                        if !quiesced {
                            self.quiesce_mark = Some(self.gather_seq);
                        }
                        window_left = Some(self.seal_coalesce.saturating_sub(open_for));
                    }
                }
                let deadline = self
                    .strict_watchdog_deadline
                    .get_or_insert(now + Duration::from_micros(STRICT_WATCHDOG_US));
                if now >= *deadline {
                    *deadline = now + Duration::from_micros(STRICT_WATCHDOG_US);
                }
                let mut hold = deadline
                    .saturating_duration_since(now)
                    .min(Duration::from_micros(GATHER_POLL_US));
                if let Some(left) = window_left
                    && left > Duration::ZERO
                {
                    hold = hold.min(left);
                }
                return FramePlan::Hold(hold);
            }
            self.strict_watchdog_deadline = None;
            self.idle_dumped = false;
            self.quiesce_mark = None;
            self.coalesce_since = None;
            match self.seal() {
                Some(FramePlan::Dispatch(_)) => continue,
                Some(plan) => return plan,
                None if self.group_assembling() => {
                    return FramePlan::Hold(Duration::from_micros(GATHER_POLL_US));
                }
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

    #[test]
    fn frame_every_case() {
        seals_complete_lanes_and_orders_waves_by_slot();
        a_grouped_lanes_first_frame_waits_for_its_cohort();
        a_group_composes_on_a_pipeline_that_already_fired_ungrouped();
        a_stated_cohort_is_never_sealed_short();
        a_cohort_that_never_completes_kills_its_request_by_name();
        a_group_never_fires_short_when_the_leash_drops_a_member();
        a_gathering_cohort_holds_its_own_lanes_and_not_the_fleet();
        a_group_is_never_split_across_two_partitions();
        incomplete_lane_holds_the_seal_until_it_completes();
        sealed_frame_dispatches_whole_and_frames_overlap();
        a_fire_racing_the_suspend_seals_alone_without_rejoining_the_wait_set();
        a_lane_parked_mid_frame_seals_what_it_submitted();
        truncated_frame_seals_with_submitted_fires_only();
        graceful_close_releases_the_wait();
        consumed_release_leaves_no_phantom_hold_for_bystander();
    }

    fn seals_complete_lanes_and_orders_waves_by_slot() {
        let mut policy = FramePolicy::new(4, 64, 4096, None);
        let (a, b) = (pid(), pid());
        for slot in 0..4 {
            policy.on_fire_enqueued(stamp(a, 0, slot, 4), Some(a), 100 + slot as u64, 1, 1, None);
        }
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 200, 37, 1, None);

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

    fn a_grouped_lanes_first_frame_waits_for_its_cohort() {
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_secs(86_400));
        let owner = pid();
        let (caption, context, image) = (pid(), pid(), pid());
        let now = Instant::now();

        policy.on_fire_enqueued(stamp(caption, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 3)));
        let queued: QueuedFireIds = [1].into_iter().collect();
        assert!(
            matches!(plan(&mut policy, &queued, now), FramePlan::Hold(_)),
            "the first lane of a cohort of three must not seal alone"
        );
        policy.on_fire_enqueued(stamp(context, 0, 0, 1), Some(owner), 2, 16, 1, Some((7, 3)));
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        assert!(
            matches!(plan(&mut policy, &queued, now), FramePlan::Hold(_)),
            "two of three still hold"
        );
        policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 3, 64, 1, Some((7, 3)));
        let queued: QueuedFireIds = [1, 2, 3].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, now));
        sealed.sort_unstable();
        assert_eq!(
            sealed,
            vec![1, 2, 3],
            "the whole cohort seals into one frame"
        );
        assert!(policy.cohorts.is_empty(), "a complete cohort is forgotten");

        policy.on_fire_enqueued(stamp(caption, 1, 0, 1), Some(owner), 4, 8, 1, Some((7, 3)));
        policy.on_fire_enqueued(stamp(image, 1, 0, 1), Some(owner), 5, 64, 1, Some((7, 3)));
        let queued: QueuedFireIds = [4, 5].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, now),
            FramePlan::Hold(_)
        ));
        policy.on_fire_enqueued(stamp(context, 1, 0, 1), Some(owner), 6, 16, 1, Some((7, 3)));
        assert!(policy.cohorts.is_empty(), "a complete cohort is forgotten");
        let queued: QueuedFireIds = [4, 5, 6].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, now));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![4, 5, 6]);
    }

    fn a_group_composes_on_a_pipeline_that_already_fired_ungrouped() {
        for reuse_first in [false, true] {
            let mut policy = FramePolicy::new(1, 64, 4096, None)
                .with_seal_mode_ready(false)
                .with_submit_deadline(Duration::from_millis(50));
            let owner = pid();
            let (image, context) = (pid(), pid());
            let now = Instant::now();

            policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 1, 8, 1, None);
            let queued: QueuedFireIds = [1].into_iter().collect();
            assert_eq!(fires(&plan(&mut policy, &queued, now)), vec![1]);
            policy.on_frame_retired([image]);

            let (first, second) = if reuse_first {
                ((image, 1, 2u64), (context, 0, 3u64))
            } else {
                ((context, 0, 2u64), (image, 1, 3u64))
            };
            policy.on_fire_enqueued(
                stamp(first.0, first.1, 0, 1),
                Some(owner),
                first.2,
                8,
                1,
                Some((0, 2)),
            );
            let queued: QueuedFireIds = [first.2].into_iter().collect();
            assert!(
                matches!(plan(&mut policy, &queued, now), FramePlan::Hold(_)),
                "one lane of a cohort of two must not seal alone (reuse_first={reuse_first})"
            );
            policy.on_fire_enqueued(
                stamp(second.0, second.1, 0, 1),
                Some(owner),
                second.2,
                16,
                1,
                Some((0, 2)),
            );
            let queued: QueuedFireIds = [first.2, second.2].into_iter().collect();
            let mut sealed = fires(&plan(&mut policy, &queued, now));
            sealed.sort_unstable();
            assert_eq!(
                sealed,
                vec![2, 3],
                "the reused pipeline's fire completes the cohort (reuse_first={reuse_first})"
            );
            assert!(policy.cohorts.is_empty(), "a complete cohort is forgotten");
        }
    }

    fn a_stated_cohort_is_never_sealed_short() {
        let leash = Duration::from_millis(50);
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(leash);
        let owner = pid();
        let (image, text) = (pid(), pid());
        let now = Instant::now();
        policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 2)));
        let queued: QueuedFireIds = [1].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, now),
            FramePlan::Hold(_)
        ));
        for late in [leash + Duration::from_millis(1), leash * 20] {
            match plan(&mut policy, &queued, now + late) {
                FramePlan::Hold(_) => {}
                plan => panic!("the image lane must not fire without its group, got {plan:?}"),
            }
        }
        let landed = now + leash * 20;
        policy.on_fire_enqueued(stamp(text, 0, 0, 1), Some(owner), 2, 16, 1, Some((7, 2)));
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, landed));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![1, 2], "the late cohort still seals whole");
    }

    fn a_cohort_that_never_completes_kills_its_request_by_name() {
        let silence = Duration::from_secs(30);
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_millis(50))
            .with_silence_timeout(silence);
        let owner = pid();
        let lone = pid();
        let now = Instant::now();
        policy.on_fire_enqueued(stamp(lone, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 2)));
        let queued: QueuedFireIds = [1].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, now),
            FramePlan::Hold(_)
        ));

        let verdict = plan(&mut policy, &queued, now + silence + Duration::from_secs(1));
        let FramePlan::Terminate(doomed) = &verdict else {
            panic!("a cohort short past the silence timeout ends the request, got {verdict:?}");
        };
        assert_eq!(
            doomed,
            &vec![(
                owner,
                Doom::CohortNeverComplete {
                    group: 7,
                    expected: 2,
                    arrived: 1,
                },
            )]
        );
        let said = doomed[0].1.to_string();
        assert!(
            said.contains("attention group 7")
                && said.contains("2 live forward passes")
                && said.contains("only 1")
                && said.contains("must submit into the group's next frame"),
            "the error names the group, the tally, and the guest's mistake: {said}"
        );
        match plan(&mut policy, &queued, now + silence * 3) {
            FramePlan::Hold(_) => {}
            plan => panic!("a condemned cohort neither repeats nor fires, got {plan:?}"),
        }
    }

    fn a_group_never_fires_short_when_the_leash_drops_a_member() {
        let leash = Duration::from_millis(50);
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(leash);
        let owner = pid();
        let (image, text) = (pid(), pid());
        let now = Instant::now();
        policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 2)));
        policy.on_fire_enqueued(stamp(text, 0, 0, 1), Some(owner), 2, 16, 1, Some((7, 2)));
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, now)).len(), 2);
        policy.on_frame_retired([image, text]);
        assert!(policy.cohorts.is_empty(), "steady state keeps no cohort");

        policy.on_fire_enqueued(stamp(image, 1, 0, 1), Some(owner), 3, 8, 1, Some((7, 2)));
        let queued: QueuedFireIds = [3].into_iter().collect();
        for late in [Duration::ZERO, leash + Duration::from_millis(1), leash * 10] {
            match plan(&mut policy, &queued, now + late) {
                FramePlan::Hold(_) => {}
                plan => panic!("half a group must never fire, got {plan:?}"),
            }
        }
        policy.on_fire_enqueued(stamp(text, 1, 0, 1), Some(owner), 4, 16, 1, Some((7, 2)));
        let queued: QueuedFireIds = [3, 4].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, now + leash * 10));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![3, 4]);
    }

    fn a_gathering_cohort_holds_its_own_lanes_and_not_the_fleet() {
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_millis(50));
        let owner = pid();
        let grouped = pid();
        let stranger = pid();
        let now = Instant::now();
        policy.on_fire_enqueued(stamp(grouped, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 2)));
        policy.on_fire_enqueued(stamp(stranger, 0, 0, 1), Some(stranger), 2, 8, 1, None);
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, now)),
            vec![2],
            "the ungrouped lane fires; the grouped one waits for its group"
        );
    }

    fn a_group_is_never_split_across_two_partitions() {
        let mut policy = FramePolicy::new(1, 8, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_millis(50));
        let owner = ProcessId::from_u128(9);
        let stranger = ProcessId::from_u128(1);
        let (image, text) = (ProcessId::from_u128(2), ProcessId::from_u128(3));
        let now = Instant::now();
        policy.on_fire_enqueued(stamp(stranger, 0, 0, 1), Some(stranger), 1, 2, 2, None);
        policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 2, 4, 4, Some((7, 2)));
        policy.on_fire_enqueued(stamp(text, 0, 0, 1), Some(owner), 3, 4, 4, Some((7, 2)));
        let queued: QueuedFireIds = [1, 2, 3].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, now)),
            vec![1],
            "the group does not fit beside the stranger, so it takes the next partition whole"
        );
        let mut group_frame = fires(&plan(&mut policy, &queued, now));
        group_frame.sort_unstable();
        assert_eq!(group_frame, vec![2, 3], "the group's lanes ride one frame");
    }

    fn incomplete_lane_holds_the_seal_until_it_completes() {
        let mut policy = FramePolicy::new(2, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_secs(86_400));
        let (fast, slow) = (pid(), pid());
        policy.on_fire_enqueued(stamp(fast, 0, 0, 2), Some(fast), 1, 1, 1, None);
        policy.on_fire_enqueued(stamp(fast, 0, 1, 2), Some(fast), 2, 1, 1, None);
        policy.on_fire_enqueued(stamp(slow, 0, 0, 2), Some(slow), 3, 1, 1, None);

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

        policy.on_fire_enqueued(stamp(slow, 0, 1, 2), Some(slow), 4, 1, 1, None);
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
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 50, 1, 1, None);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 51, 1, 1, None);
        let queued: QueuedFireIds = [50, 51].into_iter().collect();
        let FramePlan::Dispatch(frame0) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected lane a's whole frame");
        };
        assert_eq!(frame0, vec![vec![50], vec![51]]);
        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 60, 1, 1, None);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 61, 1, 1, None);
        policy.on_fire_enqueued(stamp(a, 1, 0, 2), Some(a), 52, 1, 1, None);
        policy.on_fire_enqueued(stamp(a, 1, 1, 2), Some(a), 53, 1, 1, None);
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
        policy.on_fire_enqueued(stamp(victim, 0, 0, 2), Some(victim), 100, 1, 1, None);
        policy.on_fire_enqueued(stamp(victim, 0, 1, 2), Some(victim), 101, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 0, 0, 2), Some(healthy), 102, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 0, 1, 2), Some(healthy), 103, 1, 1, None);
        let queued: QueuedFireIds = [100, 101, 102, 103].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, Instant::now()),
            FramePlan::Dispatch(_)
        ));

        policy.on_process_suspend(victim);
        policy.on_fire_enqueued(stamp(victim, 1, 0, 2), Some(victim), 200, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 1, 0, 2), Some(healthy), 300, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 1, 1, 2), Some(healthy), 301, 1, 1, None);
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
        policy.on_fire_enqueued(stamp(victim, 1, 1, 2), Some(victim), 201, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 2, 0, 2), Some(healthy), 302, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 2, 1, 2), Some(healthy), 303, 1, 1, None);
        let queued: QueuedFireIds = [201, 302, 303].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("the late slot must seal");
        };
        assert_eq!(waves[1], vec![201, 303], "the late slot keeps its wave");

        policy.on_fire_enqueued(stamp(victim, 2, 0, 2), Some(victim), 400, 1, 1, None);
        policy.on_fire_enqueued(stamp(victim, 2, 1, 2), Some(victim), 401, 1, 1, None);
        assert!(
            policy.lanes[&victim].awaited,
            "a resumed process rejoins on its next frame"
        );
    }

    fn a_lane_parked_mid_frame_seals_what_it_submitted() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 10, 1, 1, None);
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
        policy.on_fire_enqueued(stamp(lane, 0, 0, 4), Some(lane), 30, 1, 1, None);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 4), Some(lane), 31, 1, 1, None);
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
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 90, 1, 1, None);
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 91, 1, 1, None);
        let queued: QueuedFireIds = [90, 91].into_iter().collect();
        let bootstrap = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&bootstrap).len(), 2);

        policy.on_fire_enqueued(stamp(b, 1, 0, 1), Some(b), 92, 1, 1, None);
        let queued: QueuedFireIds = [92].into_iter().collect();
        match plan(&mut policy, &queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("the gather must block on lane a, got {plan:?}"),
        }
        policy.on_lane_leave(a, None, false);
        let next = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&next), vec![92]);
    }

    fn consumed_release_leaves_no_phantom_hold_for_bystander() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let executing = pid();
        let bystander = pid();
        policy.on_bind_enqueued(Some(executing));
        policy.on_bind_completed(Some(executing));
        policy.on_execution_slot_consumed(executing);
        policy.on_fire_enqueued(stamp(executing, 0, 0, 1), Some(executing), 95, 1, 1, None);
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
