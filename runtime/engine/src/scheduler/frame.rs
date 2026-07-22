//! Vesuvius frame scheduling (k > 1): sealed membership epochs over k-wave
//! frames, replacing the per-wave wait-all barrier ([`super::quorum`]).
//!
//! A *frame* is k consecutive waves submitted as one unit per lane: the guest
//! supplies exactly k ordered slots (`forward.submit`), slot i executes in
//! wave i, and a `none` slot is a no-op for that wave. The scheduler:
//!
//! - collects per-lane frame submissions (each fire carries a
//!   [`FrameStamp`]);
//! - at each frame boundary SEALS the next frame's member set from the lanes
//!   whose oldest queued frame is complete — a lane that misses a boundary is
//!   auto-Idled and misses one whole frame, but nothing blocks on it (the
//!   wait-all barrier's fleet-stall failure mode is deleted);
//! - dispatches the sealed frame's waves in slot order at the existing
//!   in-flight depth (Phase 1 executes over the per-wave driver path);
//! - admits members deterministically (arrival-complete lanes, first-fit in
//!   lane-id order against the per-wave token/row budgets) — pure arithmetic
//!   over declared demand, never timing.
//!
//! The policy is a pure state machine: the worker owns the queue and the
//! driver lane, and drives this through the `on_*` bookkeeping calls plus
//! [`FramePolicy::dispatch_candidates`]. Every fire id that enters a sealed
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
    /// Fires that came back RETRY after posting: they re-dispatch before any
    /// further wave of any frame opens (per-lane slot order is enforced by
    /// channel tickets; the makeup gate keeps the pipe from racing ahead of
    /// a known-uncommitted step).
    makeup: HashSet<u64>,
    /// Posted fires not yet retired.
    outstanding: usize,
}

impl SealedFrame {
    fn waves_all_posted(&self) -> bool {
        self.next_wave >= self.waves.len() && self.open_remaining.is_empty()
    }

    fn is_complete(&self) -> bool {
        self.waves_all_posted() && self.outstanding == 0 && self.makeup.is_empty()
    }
}

/// What the worker should do next for frame-managed launches.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum FramePlan {
    /// Dispatch (a subset of) these fire ids now, in the given order.
    Dispatch(Vec<u64>),
    /// Nothing dispatchable yet; re-decide after `0` = immediately-parked
    /// (event-driven) or after the bootstrap hold.
    Hold(Duration),
    /// No sealed work and no seal candidates: park until an arrival.
    Park,
}

pub(super) struct FramePolicy {
    k: usize,
    max_wave_tokens: usize,
    max_wave_rows: usize,
    lanes: BTreeMap<ProcessId, LaneState>,
    sealed: VecDeque<SealedFrame>,
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
            cold_hold_deadline: None,
            ever_sealed: false,
        }
    }

    pub fn k(&self) -> usize {
        self.k
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

    /// A pipeline scope left. `purge_queued` for Terminate/Suspend (its
    /// queued fires were rejected); graceful Close keeps queued frames — the
    /// already-accepted fires drain to settlement.
    pub fn on_lane_leave(&mut self, lane: ProcessId, purge_queued: bool) {
        if purge_queued {
            self.lanes.remove(&lane);
        }
    }

    /// Every scope owned by `owner` left (process terminate/suspend).
    pub fn on_process_leave(&mut self, owner: ProcessId) {
        self.lanes
            .retain(|_, lane| lane.owner != Some(owner) && !lane.frames.is_empty());
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
        if self.sealed.is_empty() && self.lanes.values().all(|lane| lane.frames.is_empty()) {
            // Books drained: the next fleet re-enters a bootstrap gather.
            self.ever_sealed = false;
            self.cold_hold_deadline = None;
            self.lanes.retain(|_, lane| !lane.frames.is_empty());
        }
    }

    /// Whether a new frame may seal now: every already-sealed frame has all
    /// waves posted (waves are globally ordered — frame f's waves all precede
    /// f+1's), and no makeup fire is outstanding anywhere.
    fn can_seal(&self) -> bool {
        self.sealed
            .iter()
            .all(|frame| frame.waves_all_posted() && frame.makeup.is_empty())
    }

    fn have_seal_candidate(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete))
    }

    /// Seal the next frame from the arrival-complete lanes, first-fit in
    /// lane-id order against the per-wave row/token budgets. Deterministic:
    /// no timing input beyond the bootstrap cold hold.
    fn seal(&mut self, now: Instant) -> Option<FramePlan> {
        if !self.have_seal_candidate() {
            self.cold_hold_deadline = None;
            return None;
        }
        if !self.ever_sealed {
            // Bootstrap gather: hold the first seal briefly so a co-launched
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
        for lane in self.lanes.values_mut() {
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
                // Over budget this epoch: the lane stays queued and is
                // admitted at a later boundary (first-fit, no relocation).
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
            lane.frames.pop_front();
        }
        self.lanes.retain(|_, lane| !lane.frames.is_empty());
        if fire_waves.is_empty() {
            return None;
        }
        self.ever_sealed = true;
        self.sealed.push_back(SealedFrame {
            waves,
            fire_waves,
            next_wave: 0,
            open_remaining: HashSet::new(),
            makeup: HashSet::new(),
            outstanding: 0,
        });
        // Open the first non-empty wave immediately.
        self.advance_open_wave(None);
        Some(FramePlan::Dispatch(Vec::new()))
    }

    /// Advance the newest sealed frame's open wave past empty/fully-resolved
    /// waves. A fire that left the queue before its wave opened (cancelled /
    /// rejected) resolves here: only ids in `still_queued` enter the open set.
    fn advance_open_wave(&mut self, still_queued: Option<&HashSet<u64>>) {
        let Some(frame) = self.sealed.back_mut() else {
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
    /// fires of the oldest frame first, then the newest frame's open wave.
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

        // Makeup fires gate everything: a known-uncommitted step re-runs
        // before any later wave opens.
        for frame in &self.sealed {
            if !frame.makeup.is_empty() {
                let mut makeups: Vec<u64> = frame.makeup.iter().copied().collect();
                makeups.sort_unstable();
                return FramePlan::Dispatch(makeups);
            }
        }

        loop {
            // Keep the open wave of the newest sealed frame fed.
            if let Some(frame) = self.sealed.back() {
                if !frame.open_remaining.is_empty() {
                    let wave = frame.next_wave - 1;
                    let ordered: Vec<u64> = frame.waves[wave]
                        .iter()
                        .filter(|fire_id| frame.open_remaining.contains(fire_id))
                        .copied()
                        .collect();
                    return FramePlan::Dispatch(ordered);
                }
                if frame.next_wave < frame.waves.len() {
                    self.advance_open_wave(Some(still_queued));
                    self.pop_complete();
                    continue;
                }
            }
            // Every sealed wave is posted: boundary. Seal the next frame if a
            // candidate is ready; otherwise park (event-driven).
            if self.can_seal() {
                match self.seal(now) {
                    Some(FramePlan::Dispatch(_)) => continue,
                    Some(plan) => return plan,
                    None => return FramePlan::Park,
                }
            }
            return FramePlan::Park;
        }
    }

    /// Probe/diagnostic summary line.
    pub fn debug_summary(&self) -> String {
        use std::fmt::Write as _;
        let mut out = format!(
            "frame k={} lanes={} sealed={} ever_sealed={}",
            self.k,
            self.lanes.len(),
            self.sealed.len(),
            self.ever_sealed,
        );
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
        let FramePlan::Dispatch(wave1) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("expected wave-1 dispatch");
        };
        assert_eq!(wave1, vec![101]);
    }

    #[test]
    fn incomplete_lane_is_auto_idled_not_awaited() {
        let mut policy = FramePolicy::new(2, 64, 4096);
        let (fast, slow) = (pid(), pid());
        policy.on_fire_enqueued(stamp(fast, 0, 0, 2), Some(fast), 1, 1, 1);
        policy.on_fire_enqueued(stamp(fast, 0, 1, 2), Some(fast), 2, 1, 1);
        // `slow` declared 2 fires but only one arrived: not a candidate.
        policy.on_fire_enqueued(stamp(slow, 0, 0, 2), Some(slow), 3, 1, 1);

        let queued: HashSet<u64> = [1, 2, 3].into_iter().collect();
        let FramePlan::Dispatch(wave0) = drive_past_cold_hold(&mut policy, &queued) else {
            panic!("fast lane must seal without waiting for slow");
        };
        assert_eq!(wave0, vec![1], "only the complete lane seals");

        // Slow completes: it joins the NEXT boundary, after fast's frame
        // fully posts.
        policy.on_fire_enqueued(stamp(slow, 0, 1, 2), Some(slow), 4, 1, 1);
        policy.on_fires_posted(&[1]);
        let FramePlan::Dispatch(wave1) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("expected fast wave 1");
        };
        assert_eq!(wave1, vec![2]);
        policy.on_fires_posted(&[2]);
        // Boundary: slow's complete frame seals even while fast's fires are
        // still outstanding (waves-all-posted is the seal gate).
        let queued: HashSet<u64> = [3, 4].into_iter().collect();
        let FramePlan::Dispatch(slow0) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("expected slow's frame to seal at the boundary");
        };
        assert_eq!(slow0, vec![3]);
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
    fn dropped_fires_resolve_and_frames_complete() {
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
        assert!(!policy.ever_sealed, "drained books re-arm the gather");
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

    #[test]
    fn over_budget_lane_defers_to_a_later_boundary() {
        // Wave budget of 40 tokens: lane a's 37-token chunk fits, lane b's
        // additional 37 does not — b waits for the next boundary.
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
        let FramePlan::Dispatch(next) = policy.plan_dispatch(&queued, Instant::now()) else {
            panic!("lane b must seal at the next boundary");
        };
        assert_eq!(next, vec![41]);
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
