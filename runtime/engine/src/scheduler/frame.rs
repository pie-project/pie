//! Vesuvius frame scheduling (k > 1): the **wait-for-all-active-lanes**
//! quorum rule ([`super::quorum`]) lifted to frame granularity. Wait until
//! every awaited lane's next FRAME is fully submitted, then seal the dense
//! epoch and dispatch its k waves in slot order.
//!
//! A *frame* is k consecutive waves submitted as one unit per lane: the guest
//! supplies exactly k ordered slots (`forward.submit`), slot i executes in
//! wave i, and a `none` slot is a no-op for that wave. The scheduler:
//!
//! - collects per-lane frame submissions (each fire carries a
//!   [`FrameStamp`]);
//! - HOLDS the next seal until every awaited lane's oldest queued frame is
//!   arrival-complete — the infinite wait-all rule: membership changes only
//!   through explicit bind/close/leave events, and the 1 s watchdog only
//!   REPORTS a non-responsive lane, never evicts it;
//! - holds unconditionally while a bind control is in assembly (a binding
//!   process is a missing member before its first fire, exactly as in the
//!   per-wave quorum);
//! - seals from every ready lane (deterministic first-fit in lane-id order
//!   against the per-wave token/row budgets — pure arithmetic over declared
//!   demand, never timing). A lane deferred by capacity is served in the
//!   same structurally partitioned round without re-awaiting the lanes
//!   already served ([`FramePolicy::round_served`], the quorum's round rule);
//! - dispatches the sealed frame's waves in slot order, each wave gated on
//!   its own posted fires' retirement (the per-wave driver path cannot
//!   overlap data-dependent waves);
//! - releases a gracefully closed lane from the wait-set immediately while
//!   its accepted frames drain to settlement.
//!
//! The policy is a pure state machine: the worker owns the queue and the
//! driver lane, and drives this through the `on_*` bookkeeping calls plus
//! [`FramePolicy::plan_dispatch`]. Every fire id that enters a sealed
//! frame leaves it through exactly one of `on_fires_posted` →
//! `on_fire_retired` (possibly cycling through the makeup set on RETRY) or
//! `on_fire_dropped` (rejected while queued).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::scheduler::ProcessId;

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
/// narrow head frame. Same constant/lever as the quorum cold hold.
const COLD_HOLD_US: u64 = 2_000;

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
    /// The member lanes. Coexisting sealed frames have DISJOINT lane sets
    /// (a busy lane may not re-seal), so a capacity-partitioned round's
    /// frames have no cross-frame data dependencies and may pipeline.
    lanes: HashSet<ProcessId>,
    waves: Vec<Vec<u64>>,
    fire_waves: HashMap<u64, usize>,
    /// Next wave index to open (waves below it are fully posted/resolved).
    next_wave: usize,
    /// Fire ids of the OPEN wave not yet posted or dropped.
    open_remaining: HashSet<u64>,
    /// Fires that came back RETRY after posting: they re-dispatch before any
    /// further wave of any frame opens (per-lane slot order is enforced by
    /// channel tickets; the makeup gate keeps the pipe from racing ahead of
    /// a known-uncommitted step).
    makeup: HashSet<u64>,
    /// Posted fires not yet retired.
    outstanding: usize,
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
    max_wave_tokens: usize,
    max_wave_rows: usize,
    /// THE wait-set plus each lane's queued frames. BTreeMap for
    /// deterministic admission order.
    lanes: BTreeMap<ProcessId, LaneState>,
    sealed: VecDeque<SealedFrame>,
    /// Bind controls accepted by the scheduler but not yet completed. These
    /// processes are missing members even before their first fire, and a
    /// seal holds unconditionally while any of them is absent.
    pending_binds: BTreeMap<ProcessId, usize>,
    /// Lanes already sealed in a structurally partitioned logical round.
    /// They are not awaited again until every awaited lane has sealed once
    /// and the round closes.
    round_served: HashSet<ProcessId>,
    /// Liveness-only deadline for the current blocked-gather episode.
    strict_watchdog_deadline: Option<Instant>,
    cold_hold_deadline: Option<Instant>,
    ever_sealed: bool,
}

impl FramePolicy {
    pub fn new(k: usize, max_wave_rows: usize, max_wave_tokens: usize) -> Self {
        Self {
            k,
            max_wave_tokens,
            max_wave_rows,
            lanes: BTreeMap::new(),
            sealed: VecDeque::new(),
            pending_binds: BTreeMap::new(),
            round_served: HashSet::new(),
            strict_watchdog_deadline: None,
            cold_hold_deadline: None,
            ever_sealed: false,
        }
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

    /// A bind control entered the scheduler. The binding process is a
    /// missing member from this moment: no seal fires while it assembles.
    pub fn on_bind_enqueued(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid {
            *self.pending_binds.entry(pid).or_default() += 1;
        }
    }

    /// A bind control completed, whether successfully or with an error. The
    /// lane itself joins the wait-set at its first stamped fire.
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
        self.round_served.remove(&lane);
        self.close_round_if_complete();
        self.maybe_reset_episode();
    }

    /// Every scope owned by `owner` left (process terminate/suspend).
    pub fn on_process_leave(&mut self, owner: ProcessId) {
        self.lanes.retain(|_, lane| lane.owner != Some(owner));
        self.pending_binds.remove(&owner);
        self.round_served
            .retain(|pid| self.lanes.contains_key(pid));
        self.close_round_if_complete();
        self.maybe_reset_episode();
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
        self.round_served.clear();
    }

    fn close_round_if_complete(&mut self) {
        let mut any_awaited = false;
        let all_served = self.lanes.iter().all(|(pid, lane)| {
            if lane.awaited {
                any_awaited = true;
                self.round_served.contains(pid)
            } else {
                true
            }
        });
        if any_awaited && all_served {
            self.round_served.clear();
        }
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
                if frame.fire_waves.contains_key(fire_id) {
                    let in_open = frame.open_remaining.remove(fire_id);
                    let in_makeup = frame.makeup.remove(fire_id);
                    if in_open || in_makeup {
                        frame.outstanding += 1;
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
            if frame.fire_waves.contains_key(&fire_id) {
                frame.outstanding = frame.outstanding.saturating_sub(1);
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

    /// Seal the next frame from every ready lane, first-fit in lane-id
    /// order against the per-wave row/token budgets. Called only once the
    /// wait-all gate holds (no missing awaited lane, no bind in assembly);
    /// deterministic — no timing input beyond the bootstrap cold hold.
    fn seal(&mut self, now: Instant) -> Option<FramePlan> {
        if !self.have_seal_candidate() {
            self.cold_hold_deadline = None;
            return None;
        }
        if !self.ever_sealed {
            // Bootstrap gather: membership is still forming (the wait-set
            // has only the lanes that already submitted), so "all ready" is
            // trivially true. Hold the first seal briefly so a co-launched
            // fleet lands in one epoch.
            match self.cold_hold_deadline {
                None => {
                    let hold = Duration::from_micros(COLD_HOLD_US);
                    self.cold_hold_deadline = Some(now + hold);
                    return Some(FramePlan::Hold(hold));
                }
                Some(deadline) if now < deadline => {
                    return Some(FramePlan::Hold(deadline - now));
                }
                Some(_) => {}
            }
        }
        self.cold_hold_deadline = None;

        let mut waves: Vec<Vec<u64>> = vec![Vec::new(); self.k];
        let mut fire_waves = HashMap::new();
        let mut wave_tokens = vec![0usize; self.k];
        let mut wave_rows = vec![0usize; self.k];
        let mut members: HashSet<ProcessId> = HashSet::new();
        // A lane still holding fires in an executing frame may not join a
        // new one (its next frame depends on the executing frame's waves).
        let busy: HashSet<ProcessId> = self
            .sealed
            .iter()
            .flat_map(|frame| frame.lanes.iter().copied())
            .collect();
        for (lane_id, lane) in self.lanes.iter_mut() {
            if busy.contains(lane_id) {
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
                // Over budget this epoch: the lane stays queued (ready, not
                // missing) and seals in this round's next partition.
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
        self.lanes
            .retain(|_, lane| lane.awaited || !lane.frames.is_empty());
        if fire_waves.is_empty() {
            return None;
        }
        self.ever_sealed = true;
        self.round_served.extend(members.iter().copied());
        self.close_round_if_complete();
        self.sealed.push_back(SealedFrame {
            lanes: members,
            waves,
            fire_waves,
            next_wave: 0,
            open_remaining: HashSet::new(),
            makeup: HashSet::new(),
            outstanding: 0,
        });
        // Open the first non-empty wave immediately.
        let newest = self.sealed.len() - 1;
        self.advance_open_wave(newest, None);
        Some(FramePlan::Dispatch(Vec::new()))
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

        // Consecutive waves OF ONE FRAME are data-dependent (device-advanced
        // channels): the per-wave driver path validates predecessor
        // publication at launch staging, so a frame's next wave opens only
        // once its own posted fires have retired. Coexisting sealed frames
        // (a capacity-partitioned round) have DISJOINT lane sets, so their
        // waves are independent and pipeline freely at the launch depth.
        loop {
            let mut advanced = false;
            for index in 0..self.sealed.len() {
                let frame = &self.sealed[index];
                if !frame.makeup.is_empty() {
                    let mut makeups: Vec<u64> = frame.makeup.iter().copied().collect();
                    makeups.sort_unstable();
                    return FramePlan::Dispatch(makeups);
                }
                if !frame.open_remaining.is_empty() {
                    let wave = frame.next_wave - 1;
                    let ordered: Vec<u64> = frame.waves[wave]
                        .iter()
                        .filter(|fire_id| frame.open_remaining.contains(fire_id))
                        .copied()
                        .collect();
                    return FramePlan::Dispatch(ordered);
                }
                if frame.next_wave < frame.waves.len() && frame.outstanding == 0 {
                    self.advance_open_wave(index, Some(still_queued));
                    self.pop_complete();
                    advanced = true;
                    break;
                }
            }
            if advanced {
                continue;
            }
            // Boundary: the wait-all frame quorum. Seal only once every
            // awaited lane's next frame is fully submitted (an idle lane
            // between frames is a missing member) and no bind is in
            // assembly. The wait is INFINITE by principle — the watchdog
            // below reports a stalled gather but never evicts a member and
            // never fires a narrow epoch; membership changes only through
            // bind/close/leave events.
            if !self.lanes.values().any(|lane| !lane.frames.is_empty()) {
                // Nothing queued anywhere: no gather episode is running.
                self.strict_watchdog_deadline = None;
                return FramePlan::Park;
            }
            let missing = self
                .lanes
                .iter()
                .filter(|(pid, lane)| {
                    lane.awaited
                        && !self.round_served.contains(*pid)
                        && !lane.frames.front().is_some_and(PendingFrame::is_complete)
                })
                .count();
            if !self.pending_binds.is_empty() || missing > 0 {
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
                        "awaited_lanes":
                            self.lanes.values().filter(|lane| lane.awaited).count(),
                    }));
                }
                return FramePlan::Hold(deadline.saturating_duration_since(now));
            }
            self.strict_watchdog_deadline = None;
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
            "frame k={} lanes={} awaited={} sealed={} pending_binds={} round_served={} ever_sealed={} watchdog={:?}",
            self.k,
            self.lanes.len(),
            self.lanes.values().filter(|lane| lane.awaited).count(),
            self.sealed.len(),
            self.pending_binds.values().sum::<usize>(),
            self.round_served.len(),
            self.ever_sealed,
            self.strict_watchdog_deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now())),
        );
        for (pid, lane) in &self.lanes {
            if lane.frames.is_empty() {
                continue;
            }
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
    fn seals_complete_lanes_and_orders_waves_by_slot() {
        let mut policy = FramePolicy::new(4, 64, 4096);
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
        assert_eq!(
            policy.plan_dispatch(&queued, Instant::now()),
            FramePlan::Park,
            "a dependent wave holds until the posted wave retires"
        );
        for fire_id in &wave0 {
            policy.on_fire_retired(*fire_id, false);
        }
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
        let mut policy = FramePolicy::new(2, 64, 4096);
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
        let mut policy = FramePolicy::new(2, 64, 4096);
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

    #[test]
    fn dropped_fires_resolve_and_leave_rearms_the_gather() {
        let mut policy = FramePolicy::new(2, 64, 4096);
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
        let mut policy = FramePolicy::new(4, 64, 4096);
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
        let mut policy = FramePolicy::new(2, 64, 40);
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
        let mut policy = FramePolicy::new(2, 64, 4096);
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
        let mut policy = FramePolicy::new(2, 64, 4096);
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

    /// A bind in assembly holds the seal unconditionally — the binding
    /// process is a missing member before its first fire, exactly as in the
    /// per-wave quorum.
    #[test]
    fn pending_binds_hold_the_seal_until_assembly_completes() {
        let mut policy = FramePolicy::new(2, 64, 4096);
        let lane = pid();
        let binder = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 95, 1, 1);
        policy.on_bind_enqueued(Some(binder));
        let queued: HashSet<u64> = [95].into_iter().collect();
        match policy.plan_dispatch(&queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("a pending bind must hold the seal, got {plan:?}"),
        }
        policy.on_bind_completed(Some(binder));
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("assembly complete: the seal must proceed");
        };
        assert_eq!(wave0, vec![95]);
    }

    #[test]
    fn terminate_purges_queued_frames_close_keeps_them() {
        let mut policy = FramePolicy::new(2, 64, 4096);
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
