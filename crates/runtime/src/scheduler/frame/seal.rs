//! Sealing: a boundary opened and closed, the lanes ready sealed into a
//! frame's waves, and the sealed frames planned for dispatch.

use super::*;

impl FramePolicy {
    pub(super) fn have_seal_candidate(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete))
    }

    pub(super) fn boundary_open(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.awaited && !lane.fired_this_boundary)
    }

    pub(super) fn open_boundary(&mut self) {
        for lane in self.lanes.values_mut() {
            lane.fired_this_boundary = false;
        }
    }

    pub(super) fn close_boundary(&mut self) {
        for lane in self.lanes.values_mut() {
            lane.fired_this_boundary = true;
        }
    }

    pub(super) fn seal(&mut self) -> Option<FramePlan> {
        if !self.have_seal_candidate() {
            return None;
        }
        let mid_boundary = self.boundary_open();
        let trace = seal_trace();

        let k = self.k;
        let max_wave_rows = self.max_wave_rows;
        let max_wave_tokens = self.max_wave_tokens;
        let max_wave_readouts = self.max_wave_readouts;
        // One frame of prompts on the card at a time: prompts arriving behind
        // it bind together into the next, while decodes keep running ahead.
        // A lane whose own prompt is in flight continues it in the next frame
        // (its next piece reads what that one writes, in order on the
        // device): there is no prompt behind it to bind.
        let prompting_in_flight: BTreeSet<ProcessId> = self.prompts_executing.clone();
        debug_assert!(
            self.prompts_executing.is_subset(&self.in_flight_lanes),
            "a prompt executing on no lane in flight: {:?} of {:?}",
            self.prompts_executing,
            self.in_flight_lanes
        );
        loop {
            // Lanes that have not fired this boundary go ahead of those that
            // have: a first-come, first-served turn per boundary, so no lane
            // fires twice while another waits for its first. It ranks no
            // tenant: the policy's schedule key, where one is installed,
            // orders each group below, and lane order breaks its ties.
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
            let mut prompts: BTreeSet<ProcessId> = BTreeSet::new();
            let mut sealed_widest = 0usize;
            let mut dropped_empty = false;
            let mut skipped = SealSkips::default();
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
                    skipped.capped += 1;
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
                    skipped.unfit += 1;
                    continue;
                }
                let prompting = live.iter().any(|fire| fire.tokens > fire.rows);
                if prompting
                    && !mid_boundary
                    && !self.prompts_only
                    && prompting_in_flight.iter().any(|lane| lane != lane_id)
                {
                    skipped.prompt_held += 1;
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
                if prompting {
                    prompts.insert(*lane_id);
                }
                lane.frames.pop_front();
            }
            if fire_waves.is_empty() {
                if dropped_empty {
                    continue;
                }
                debug_assert!(
                    skipped.prompt_held == 0 || !self.prompts_executing.is_empty(),
                    "a prompt held behind none executing"
                );
                self.seal_skips = skipped;
                self.lanes.retain(|_, lane| {
                    lane.awaited || (lane.parked && !lane.own_submit) || !lane.frames.is_empty()
                });
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
            self.parked_on_lane = false;
            if let Some(stats) = &self.stats
                && self.lanes.iter().any(|(lane_id, lane)| {
                    !members.contains(lane_id)
                        && lane.awaited
                        && lane.frames.is_empty()
                        && lane.waits_on.as_ref().is_some_and(|c| !c.is_settled())
                })
            {
                stats
                    .fire
                    .quorum
                    .frames_past_waiting_lane
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let _ = &fire_waves;
            self.sealed.push_back(SealedFrame {
                waves,
                members: members.iter().copied().collect(),
                widest: sealed_widest,
                prompts,
            });
            self.lanes.retain(|_, lane| {
                lane.awaited || (lane.parked && !lane.own_submit) || !lane.frames.is_empty()
            });
            return Some(FramePlan::Dispatch(Vec::new()));
        }
    }

    pub fn record_rider_wave(&self) {
        self.record_sealed_waves(1);
    }

    pub(super) fn record_sealed_waves(&self, wave_count: usize) {
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

    pub(super) fn record_seal_engagement(&self) {
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            stats.fire.quorum.seal_events.fetch_add(1, Relaxed);
            if self.executing_now {
                stats.fire.quorum.seal_while_executing.fetch_add(1, Relaxed);
            }
        }
    }

    /// What to do with the queued frames now. A park is a wait for an
    /// event: with nothing executing (no landing to wake it) and frames
    /// queued, one is always scheduled ([`Self::wake_scheduled`]), never a
    /// silent scheduler; asserted in debug builds.
    pub fn plan_dispatch(
        &mut self,
        still_queued: &QueuedFireIds,
        blocked_lanes: &HashSet<ProcessId>,
        executing: bool,
        now: Instant,
    ) -> FramePlan {
        // A frame handed out and never posted makes no prompt executing.
        self.dispatched_prompts.clear();
        let plan = self.plan_frames(still_queued, blocked_lanes, executing, now);
        debug_assert!(
            plan != FramePlan::Park
                || executing
                || !self.has_queued_frames()
                || !blocked_lanes.is_empty()
                || self.wake_scheduled(),
            "a park with nothing executing, frames queued and no wake scheduled: {}",
            self.debug_summary()
        );
        plan
    }

    /// Whether an event is coming that plans the queued frames again, with
    /// nothing executing: a lane still expected to submit (awaited with no
    /// frame, or a frame begun and not complete: its owner sends the rest), a bind or a join
    /// on its way, an owner suspended or blocked (its resume or its wait's
    /// end), or a ready frame cut wider than the cap (the cap's rise).
    #[must_use]
    pub fn wake_scheduled(&self) -> bool {
        self.has_pending_binds()
            || self.is_joining()
            || !self.suspended.is_empty()
            || !self.blocked.is_empty()
            || self.lanes.values().any(|lane| {
                (lane.awaited && lane.frames.is_empty())
                    || lane
                        .frames
                        .front()
                        .is_some_and(|front| !front.is_complete())
                    || lane.frames.front().is_some_and(|front| {
                        front
                            .fires
                            .iter()
                            .any(|fire| fire.rung_rows > self.fire_cap)
                    })
            })
    }

    pub(super) fn plan_frames(
        &mut self,
        still_queued: &QueuedFireIds,
        blocked_lanes: &HashSet<ProcessId>,
        executing: bool,
        now: Instant,
    ) -> FramePlan {
        self.executing_now = executing;
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
                if !self.executing_now
                    && let Some(since) = self.last_retire_at
                    && let Some(stats) = &self.stats
                {
                    use std::sync::atomic::Ordering::Relaxed;
                    let idle = now.saturating_duration_since(since).as_micros() as u64;
                    stats.fire.quorum.device_idle_us.fetch_add(idle, Relaxed);
                    stats.fire.quorum.device_idle_gaps.fetch_add(1, Relaxed);
                    stats.record_bubble_us(idle);
                }
                self.dispatched_prompts = frame.prompts;
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
                return FramePlan::Park;
            }
            let mut missing_decoders = 0usize;
            let mut missing = 0;
            let mut blocking_lanes: Vec<(ProcessId, bool, bool)> = Vec::new();
            // The blocking lanes whose guests are computing: no fire of theirs
            // in flight, no bind pending, no frame begun, and not parked in a
            // named wait of their process. A frame holds for them only while
            // the device has a frame in flight to cover their compute.
            let mut computing: Vec<ProcessId> = Vec::new();
            debug_assert!(
                !self.boundary_open(),
                "the wait-all gate is evaluated only against a closed boundary"
            );
            for (lane_id, lane) in self.lanes.iter() {
                let owes = self.in_flight_lanes.contains(lane_id)
                    || lane
                        .owner
                        .is_some_and(|owner| self.pending_binds.contains_key(&owner));
                let blocking = match lane.gate_verdict() {
                    GateVerdict::Blocking => true,
                    GateVerdict::Waits | GateVerdict::Satisfied => false,
                };
                if !blocking {
                    continue;
                }
                if !owes && lane.frames.is_empty() && !lane.parked {
                    computing.push(*lane_id);
                }
                missing += 1;
                missing_decoders += usize::from(!lane.prompting);
                if blocking_lanes.len() < 16 {
                    blocking_lanes.push((lane.owner.unwrap_or(*lane_id), lane.prompting, owes));
                }
            }
            // Holding for computing lanes would leave the device idle: nothing
            // in flight, and every other member has submitted since the last
            // retire. The frame seals past them; each rejoins with its next
            // submit, as a parked lane does. An event, not a clock: a guest
            // that computes past the frame in flight fires in the next one,
            // and one that never submits holds no other lane.
            // Per lane: a computing lane is held while a frame is in flight,
            // or until every other member has submitted since the last
            // retire. Only a lane computing alone, every other member in,
            // is sealed past.
            if !executing && missing == 1 && computing.len() == 1 && self.have_seal_candidate() {
                for lane_id in &computing {
                    if let Some(lane) = self.lanes.get_mut(lane_id) {
                        trace_unawait("computing", *lane_id, lane.owner);
                        lane.awaited = false;
                        lane.parked = true;
                        lane.own_submit = false;
                        self.sealed_past.insert(*lane_id);
                    }
                }
                continue;
            }
            tracing::debug!(
                "gate missing={missing} executing={executing} candidate={} in_flight={} awaited={} lanes={}",
                self.have_seal_candidate(),
                self.in_flight_lanes.len(),
                self.lanes.values().filter(|lane| lane.awaited).count(),
                self.lanes.len(),
            );
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
            if missing > 0 && !self.parked_on_lane {
                self.parked_on_lane = true;
                if let Some(stats) = &self.stats {
                    stats
                        .fire
                        .quorum
                        .frames_parked_on_lane
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
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
                None => {
                    // Every expected lane in and a complete frame unsealed:
                    // say what each ready lane was passed over for, once a
                    // while, so a held frame names its cause.
                    if self.have_seal_candidate()
                        && self.unsealed_said.is_none_or(|said| {
                            now.saturating_duration_since(said) >= HOLD_SAYS_AFTER
                        })
                    {
                        self.unsealed_said = Some(now);
                        tracing::warn!(
                            skips = ?self.seal_skips,
                            fire_cap = self.fire_cap,
                            summary = %self.debug_summary(),
                            "every expected lane is in and a complete frame went unsealed"
                        );
                    }
                    return FramePlan::Park;
                }
            }
        }
    }
}
