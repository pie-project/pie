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

/// As [`plan`], with a frame of the policy's in flight on the device:
/// a lane whose guest computes is held for (only an idle device seals
/// past a lone computing lane).
fn plan_in_flight(policy: &mut FramePolicy, queued: &QueuedFireIds, now: Instant) -> FramePlan {
    policy.plan_dispatch(queued, &HashSet::new(), true, now)
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
    a_lone_computing_lane_is_sealed_past_only_with_the_device_idle();
    no_prompt_waits_for_a_lane_off_a_prompt();
    consumed_release_leaves_no_phantom_hold_for_bystander();
    a_burst_of_cut_prefills_seals_its_pieces_step_by_step();
    a_wave_reads_out_no_more_than_the_engine_carves();
    a_policys_bound_packs_prompts_into_a_frame_up_to_it_and_no_further();
    a_frame_goes_without_a_lane_whose_process_waits();
    a_pipelines_fire_posted_behind_a_later_one_is_refused_by_name();
    a_closed_lane_is_never_expected_again_by_its_owner_s_unblock();
    a_lane_its_guest_parked_is_never_expected_again_by_its_owner_s_unblock();
    a_lane_whose_guest_waits_on_the_outside_holds_no_frame_and_rejoins_on_its_submit();
    a_prompt_piece_cut_under_a_cap_since_lowered_waits_whole_and_owes_nothing();
    a_parked_owner_s_complete_frame_goes_without_waiting_on_the_boundary();
    processes_over_the_execution_permits_joining_together_all_finish();
    joiners_waiting_on_a_claimant_s_prefix_hold_no_frame();
    a_lane_closing_after_its_owner_s_wait_has_its_queued_frame_sealed();
    a_lane_waiting_on_its_output_holds_no_one_elses_next_frame();
}

/// A verified beam of eight: a verify beam's lane is
/// unawaited as its owner waits, rides a seal, submits a frame of two
/// fires and closes, a fire of its racing the close; the other lanes
/// close or leave. Its queued frame is complete and nothing else is
/// expected: it seals, and the closed lane is never expected again.
fn a_lane_closing_after_its_owner_s_wait_has_its_queued_frame_sealed() {
    for late_after_close in [false, true] {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (owner, other) = (pid(), pid());
        let (lane, beside) = (pid(), pid());
        let mut next = 1u64;
        let mut submit = |policy: &mut FramePolicy, at: ProcessId, by: ProcessId, seq: u64, slots: std::ops::Range<u32>| {
            let mut ids = Vec::new();
            for slot in slots {
                policy.on_fire_enqueued(stamp(at, seq, slot, 2), Some(by), next, 1, 1);
                ids.push(next);
                next += 1;
            }
            ids
        };
        let mut queued = submit(&mut policy, lane, owner, 0, 0..2);
        queued.extend(submit(&mut policy, beside, other, 0, 0..2));
        let ids: QueuedFireIds = queued.iter().copied().collect();
        assert_eq!(fires(&plan(&mut policy, &ids, Instant::now())).len(), 4);
        policy.on_frame_posted([lane, beside], Instant::now());
        policy.on_frame_retired([lane, beside]);
        // Its owner waits (a rescore's bind): unawaited; the other lane's
        // frame goes.
        policy.on_owner_waits(owner);
        let queued = submit(&mut policy, beside, other, 1, 0..2);
        let ids: QueuedFireIds = queued.iter().copied().collect();
        assert_eq!(fires(&plan(&mut policy, &ids, Instant::now())).len(), 2);
        policy.on_frame_posted([beside], Instant::now());
        policy.on_frame_retired([beside]);
        // The other process ends.
        policy.on_lane_leave(beside, Some(other), false);
        policy.on_process_leave(other);
        // Its wait ends; back, its lane submits a frame of two and
        // closes; one fire races the close.
        policy.on_owner_unblocked(owner);
        let mut queued = submit(&mut policy, lane, owner, 1, 0..1);
        if late_after_close {
            policy.on_lane_leave(lane, Some(owner), false);
            queued.extend(submit(&mut policy, lane, owner, 1, 1..2));
        } else {
            queued.extend(submit(&mut policy, lane, owner, 1, 1..2));
            policy.on_lane_leave(lane, Some(owner), false);
        }
        let ids: QueuedFireIds = queued.iter().copied().collect();
        let mut sent = Vec::new();
        for _ in 0..4 {
            match plan(&mut policy, &ids, Instant::now()) {
                FramePlan::Dispatch(waves) => sent.extend(waves.into_iter().flatten()),
                _ => break,
            }
        }
        sent.sort_unstable();
        assert_eq!(
            sent, queued,
            "late_after_close={late_after_close}: the closed lane's queued frame goes: {}",
            policy.debug_summary()
        );
        assert!(
            policy.lanes.values().all(|lane| !lane.awaited),
            "late_after_close={late_after_close}: a closed lane is not expected again: {}",
            policy.debug_summary()
        );
    }
}

/// More processes than execution permits arrive together (lanes would
/// be missing from the first frame, their processes waiting at the
/// execution gate with no permit left). A lane is
/// expected from its opening, so it opens only once its process is
/// admitted past the bind and execution gates (`Pipeline::new` admits
/// first). Opened while its process waited at a gate, the frame waited
/// on lanes that could not submit, and the admitted processes, their
/// frames queued in it, never finished to give their permits back.
/// Modelled as the host drives the frame: admitted processes open their
/// lane and submit a step at a time, gated ones only queue, and one
/// finishing hands its permit to the next. Every process finishes; with
/// lanes opened at the gate instead, the first frame is held for good.
fn processes_over_the_execution_permits_joining_together_all_finish() {
    const PROCESSES: usize = 8;
    const PERMITS: usize = 2;
    const STEPS: u64 = 3;
    for open_at_the_gate in [false, true] {
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        policy.preload_free_slots(PERMITS);
        let owners: Vec<ProcessId> = (0..PROCESSES).map(|_| pid()).collect();
        let lanes: Vec<ProcessId> = (0..PROCESSES).map(|_| pid()).collect();
        let mut admitted = vec![false; PROCESSES];
        let mut opened = vec![false; PROCESSES];
        let mut done = vec![false; PROCESSES];
        let mut submitted = vec![0u64; PROCESSES];
        let mut retired = vec![0u64; PROCESSES];
        let mut fire_of: HashMap<u64, usize> = HashMap::new();
        let mut queued: Vec<u64> = Vec::new();
        let mut next_id = 1u64;
        let mut free = PERMITS;
        let mut now = Instant::now();
        // All arrive together and queue at the gate.
        for at in 0..PROCESSES {
            policy.on_admission_queued(owners[at]);
            if open_at_the_gate {
                // Its guest opens its pipeline before its first submit
                // meets the gate: expected from here.
                policy.on_lane_open(lanes[at], owners[at]);
                opened[at] = true;
            }
        }
        let mut stalled = 0;
        for _round in 0..PROCESSES * STEPS as usize * 8 {
            if done.iter().all(|d| *d) || stalled > 8 {
                break;
            }
            // The gate grants in arrival order while permits are free.
            for at in 0..PROCESSES {
                if free > 0 && !admitted[at] && !done[at] {
                    free -= 1;
                    admitted[at] = true;
                    policy.on_execution_slot_consumed(owners[at]);
                    if !opened[at] {
                        policy.on_lane_open(lanes[at], owners[at]);
                        opened[at] = true;
                    }
                }
            }
            // Each admitted process submits its next step once its last
            // landed (its guest reads the token first).
            for at in 0..PROCESSES {
                if admitted[at]
                    && !done[at]
                    && submitted[at] == retired[at]
                    && submitted[at] < STEPS
                {
                    policy.on_fire_enqueued(
                        stamp(lanes[at], submitted[at], 0, 1),
                        Some(owners[at]),
                        next_id,
                        1,
                        1,
                    );
                    fire_of.insert(next_id, at);
                    queued.push(next_id);
                    next_id += 1;
                    submitted[at] += 1;
                }
            }
            now += Duration::from_millis(5);
            let ids: QueuedFireIds = queued.iter().copied().collect();
            let mut progressed = false;
            if let FramePlan::Dispatch(waves) = plan(&mut policy, &ids, now) {
                let posted: Vec<u64> = waves.iter().flatten().copied().collect();
                queued.retain(|id| !posted.contains(id));
                let members: Vec<ProcessId> =
                    posted.iter().map(|id| lanes[fire_of[id]]).collect();
                policy.on_frame_posted(members.clone(), now);
                policy.on_frame_retired(members);
                for id in &posted {
                    retired[fire_of[id]] += 1;
                }
                progressed = !posted.is_empty();
            }
            // A process whose last step landed ends: its lane closes and
            // its permit goes to the next at the gate.
            for at in 0..PROCESSES {
                if admitted[at] && !done[at] && retired[at] == STEPS {
                    done[at] = true;
                    policy.on_lane_leave(lanes[at], Some(owners[at]), false);
                    policy.on_slotted_terminate(owners[at]);
                    policy.on_process_leave(owners[at]);
                    policy.on_execution_slot_released(owners[at]);
                    free += 1;
                    progressed = true;
                }
            }
            stalled = if progressed { 0 } else { stalled + 1 };
        }
        let finished = done.iter().filter(|d| **d).count();
        if open_at_the_gate {
            assert_eq!(
                finished, 0,
                "control: lanes expected while their processes wait at the gate hold the first frame"
            );
            let ids: QueuedFireIds = queued.iter().copied().collect();
            assert!(
                !matches!(plan_in_flight(&mut policy, &ids, now), FramePlan::Dispatch(_)),
                "control: held for the gated lanes: {}",
                policy.debug_summary()
            );
        } else {
            assert_eq!(
                finished,
                PROCESSES,
                "every process over the permits finishes: {}",
                policy.debug_summary()
            );
        }
    }
}

/// Greedy decodes of one prompt arriving together, the prefix cache
/// cold (16 on Vulkan: 15 of 16 lanes missing from the first frame,
/// for good): the first pass claims the prefix pages, and each other
/// process's first pass waits for them to seal
/// (`coalesce::before_pass`) before it submits, its lane open.
/// Expected, the frame held the claimant's prefill, whose pages they
/// wait on. Waiting on another pass's fire, a process is told waiting
/// (parked: no frame waits on its lanes): the claimant's frame goes
/// alone, and each waiter, its pages sealed, is unblocked and rides the
/// next frame with the rest.
fn joiners_waiting_on_a_claimant_s_prefix_hold_no_frame() {
    // Four greedy decodes and sixteen.
    for count in [3usize, 15] {
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        let claimant = pid();
        let joiners: Vec<ProcessId> = (0..count).map(|_| pid()).collect();
        for &owner in std::iter::once(&claimant).chain(&joiners) {
            policy.on_lane_open(owner, owner);
        }
        policy.on_fire_enqueued(stamp(claimant, 0, 0, 1), Some(claimant), 1, 16, 1);
        let queued: QueuedFireIds = [1].into_iter().collect();
        assert!(
            !matches!(plan_in_flight(&mut policy, &queued, Instant::now()), FramePlan::Dispatch(_)),
            "control: the joiners' open lanes are expected"
        );
        // Each joiner's pass finds the prefix claimed and waits.
        for &owner in &joiners {
            policy.on_owner_waits(owner);
        }
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![1],
            "the claimant's prefill goes without the waiting joiners"
        );
        policy.on_frame_posted([claimant], Instant::now());
        policy.on_frame_retired([claimant]);
        // Its pages seal: each joiner's wait ends (unblocked before its
        // guest goes on) and it submits its first pass, the claimant its
        // first decode; they ride one frame.
        for &owner in &joiners {
            policy.on_owner_unblocked(owner);
        }
        let mut queued: Vec<u64> = vec![100];
        policy.on_fire_enqueued(stamp(claimant, 1, 0, 1), Some(claimant), 100, 1, 1);
        for (at, &owner) in joiners.iter().enumerate() {
            let id = 2 + at as u64;
            policy.on_fire_enqueued(stamp(owner, 0, 0, 1), Some(owner), id, 4, 1);
            queued.push(id);
        }
        // The joiners' prompts go together in the next frame (the claimant's
        // decode beside them, or with the joiners' first decodes after, as
        // the policy packs prompts and decodes): no frame waits on a lane
        // that does not submit.
        let ids: QueuedFireIds = queued.iter().copied().collect();
        let sent = fires(&plan(&mut policy, &ids, Instant::now()));
        assert!(
            (2..2 + count as u64).all(|id| sent.contains(&id)),
            "every joiner rides the next frame: {sent:?}"
        );
    }
}

/// The disk-only soaks' self-cycle: a process whose guest went into a
/// bind had a complete front frame queued, a continuing lane mid
/// boundary, and the boundary stayed open on lanes whose owners waited
/// on the planner's grants. Its frame was never sealed, so its fire
/// never ran, its lease never went back, and its eviction waited on it
/// for good. Parked (suspended, or waiting at its bind), an owner is
/// waited on by no frame; with the other owner blocked on its grant too,
/// no lane is expected and the frame goes.
fn a_parked_owner_s_complete_frame_goes_without_waiting_on_the_boundary() {
    for suspend in [true, false] {
        let mut policy = FramePolicy::new(1, 64, 4096, None);
        let (binder, waiting) = (pid(), pid());
        // A frame of both: the boundary opens with both fired.
        policy.on_fire_enqueued(stamp(binder, 0, 0, 1), Some(binder), 1, 1, 1);
        policy.on_fire_enqueued(stamp(waiting, 0, 0, 1), Some(waiting), 2, 1, 1);
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        let mut first = fires(&plan(&mut policy, &queued, Instant::now()));
        first.sort_unstable();
        assert_eq!(first, vec![1, 2]);
        policy.on_frame_posted([binder, waiting], Instant::now());
        policy.on_frame_retired([binder, waiting]);
        // The binder's next frame, complete; the other lane's owner
        // waits on a grant and submits nothing.
        policy.on_fire_enqueued(stamp(binder, 1, 0, 1), Some(binder), 3, 1, 1);
        let queued: QueuedFireIds = [3].into_iter().collect();
        let held = plan_in_flight(&mut policy, &queued, Instant::now());
        assert!(
            !matches!(&held, FramePlan::Dispatch(waves) if waves.iter().flatten().any(|&id| id == 3)),
            "control: mid boundary, it waits for the other lane: {held:?}"
        );
        // Its guest goes into a bind that waits, or it is evicted: no
        // frame waits on it.
        if suspend {
            policy.on_process_suspend(binder);
        } else {
            policy.on_owner_waits(binder);
        }
        // The other owner, blocked on its grant, is waited on by no one
        // either: no lane is expected, and the binder's frame goes.
        policy.on_owner_waits(waiting);
        assert_eq!(
            fires(&plan(&mut policy, &queued, Instant::now())),
            vec![3],
            "no lane left expected: its complete frame goes"
        );
    }
}

/// Lanes' complete frames wider than the cap: a raise asked of the
/// planner for the lane's id, which no process has, is told "planner
/// request cancelled" every pass. The raise is funded for the lane's
/// owner.
#[test]
fn a_capped_frame_s_raise_is_owed_by_its_lane_s_owner() {
    let mut policy =
        FramePolicy::new(1, 64, 4096, None);
    let (lane, owner) = (pid(), pid());
    policy.cap_fires(128);
    policy.on_fire_reading(stamp(lane, 0, 0, 1), Some(owner), 1, 1, 1, 1, 256, 0);
    assert_eq!(policy.cut_owed_by(), (256, Some(owner)), "the owner, not the lane");
}

/// A frame still gathering when the lower is granted completes 112 rows
/// wide under a cap of 64. The
/// widest queued fire counts whether its frame is complete or not.
#[test]
fn a_frame_still_gathering_counts_toward_the_widest_queued() {
    let mut policy =
        FramePolicy::new(1, 64, 4096, None);
    let (lane, owner) = (pid(), pid());
    policy.cap_fires(64);
    // One fire of two in: the frame is not complete.
    policy.on_fire_reading(stamp(lane, 0, 0, 2), Some(owner), 1, 1, 1, 1, 112, 0);
    assert_eq!(policy.cut_owed_by().0, 0, "not complete: owes no raise yet");
    assert_eq!(policy.queued_widest(), 112, "but the rung is not lowered under it");
}

/// The frame / eviction cycle (a process waited at admission behind a
/// preempted one, then its eviction waited on its set's fires, which
/// sat in an unsealed frame): 42 lanes, 41 of them in with complete
/// frames, the 42nd mid-frame (one of its two fires in). Its owner's
/// eviction starts: its lane leaves the expected set, its front cut to
/// what it submitted, and the frame seals at once, all 42 lanes' fires
/// in one seal and one dispatch, none of them sent apart; the evicted
/// owner's fire is with the engine, so the eviction's wait on it ends.
#[test]
fn an_owner_s_eviction_seals_the_frame_it_holds_whole() {
    let mut policy =
        FramePolicy::new(1, 64, 4096, None);
    let lanes: Vec<ProcessId> = (0..42).map(|_| pid()).collect();
    // A first frame of all 42: every lane is expected after it.
    for (at, &lane) in lanes.iter().enumerate() {
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), at as u64 + 1, 1, 1);
    }
    let queued: QueuedFireIds = (1..=42).collect();
    assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())).len(), 42);
    policy.on_frame_posted(lanes.clone(), Instant::now());
    policy.on_frame_retired(lanes.clone());
    // The next frame: 41 lanes in, the 42nd one fire of two.
    let mut queued: Vec<u64> = Vec::new();
    for (at, &lane) in lanes.iter().enumerate().take(41) {
        policy.on_fire_enqueued(stamp(lane, 1, 0, 1), Some(lane), 100 + at as u64, 1, 1);
        queued.push(100 + at as u64);
    }
    let evicted = lanes[41];
    policy.on_fire_enqueued(stamp(evicted, 1, 0, 2), Some(evicted), 200, 1, 1);
    queued.push(200);
    let ids: QueuedFireIds = queued.iter().copied().collect();
    assert!(
        !matches!(plan(&mut policy, &ids, Instant::now()), FramePlan::Dispatch(_)),
        "control: the 42nd lane is expected and mid-frame: the frame waits"
    );
    // Its eviction starts.
    policy.on_process_suspend(evicted);
    let FramePlan::Dispatch(waves) = plan(&mut policy, &ids, Instant::now()) else {
        panic!("its eviction seals the frame at once");
    };
    let sent: Vec<u64> = waves.iter().flatten().copied().collect();
    assert_eq!(sent.len(), 42, "all 42 lanes' fires in the one seal: {waves:?}");
    assert!(sent.contains(&200), "the evicted owner's fire goes with the engine");
    assert_eq!(waves.iter().filter(|wave| !wave.is_empty()).count(), 1, "one dispatch");
    let rest: QueuedFireIds = std::iter::empty().collect();
    assert!(
        !matches!(plan(&mut policy, &rest, Instant::now()), FramePlan::Dispatch(_)),
        "nothing sent apart"
    );
}

/// od-1's census, held for good ("frame k=2 lanes=44 awaited=10
/// sealed=0 suspended=11 blocked=40", gate "missing=0 candidate=true
/// in_flight=0"): 10 lanes expected and complete; 13 blocked owners'
/// lanes with a complete frame; 10 blocked owners' lanes with none; 10
/// suspended and blocked owners' lanes with none; 1 suspended owner's
/// lane with a complete frame. Every expected lane is in, so the frame
/// seals at once, the 14 parked owners' complete frames riding it: 24
/// lanes' fires in one seal, nothing left behind. Run with the parked
/// owners' frames submitted before their park and after it (a straggler),
/// and at k=2 with each frame two slots.
#[test]
fn od_1_census_seals_with_every_expected_lane_in() {
    for straggle in [false, true] {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lanes: Vec<ProcessId> = (0..44).map(|_| pid()).collect();
        let mut next = 1u64;
        let mut submit = |policy: &mut FramePolicy, lane: ProcessId, seq: u64| -> Vec<u64> {
            let ids = vec![next, next + 1];
            for (slot, &id) in ids.iter().enumerate() {
                policy.on_fire_enqueued(stamp(lane, seq, slot as u32, 2), Some(lane), id, 1, 1);
            }
            next += 2;
            ids
        };
        // A first frame of all 44: every lane expected after it.
        let mut queued: Vec<u64> = Vec::new();
        for &lane in &lanes {
            queued.extend(submit(&mut policy, lane, 0));
        }
        let ids: QueuedFireIds = queued.iter().copied().collect();
        assert_eq!(fires(&plan(&mut policy, &ids, Instant::now())).len(), 88);
        policy.on_frame_posted(lanes.clone(), Instant::now());
        policy.on_frame_retired(lanes.clone());
        let mut queued: Vec<u64> = Vec::new();
        let (expected, rest) = lanes.split_at(10);
        let (blocked_full, rest) = rest.split_at(13);
        let (blocked_empty, rest) = rest.split_at(10);
        let (both_empty, suspended_full) = rest.split_at(10);
        for &lane in expected {
            queued.extend(submit(&mut policy, lane, 1));
        }
        for &lane in blocked_full {
            if !straggle {
                queued.extend(submit(&mut policy, lane, 1));
            }
            policy.on_owner_waits(lane);
            if straggle {
                queued.extend(submit(&mut policy, lane, 1));
            }
        }
        for &lane in blocked_empty {
            policy.on_owner_waits(lane);
        }
        for &lane in both_empty {
            policy.on_process_suspend(lane);
            policy.on_owner_waits(lane);
        }
        for &lane in suspended_full {
            if !straggle {
                queued.extend(submit(&mut policy, lane, 1));
            }
            policy.on_process_suspend(lane);
            if straggle {
                queued.extend(submit(&mut policy, lane, 1));
            }
        }
        assert_eq!(policy.lanes.values().filter(|lane| lane.awaited).count(), 10);
        let ids: QueuedFireIds = queued.iter().copied().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &ids, Instant::now()) else {
            panic!("straggle={straggle}: every expected lane is in: the frame seals at once\n{}", policy.debug_summary());
        };
        let mut sent: Vec<u64> = waves.iter().flatten().copied().collect();
        sent.sort_unstable();
        queued.sort_unstable();
        assert_eq!(sent, queued, "straggle={straggle}: all 24 lanes' fires in the one seal");
    }
}

/// od-1 with the expected lanes two frames deep (qf=2, k=2): with every
/// expected lane in, the front frame seals at once with the parked
/// owners' complete frames riding it, and the next frame seals in the
/// same pass of plans, before anything lands (two frames in flight per
/// lane): no candidate with nothing missing is left for a later wake.
#[test]
fn od_1_census_seals_front_and_next_frames_with_every_expected_lane_in() {
    let mut policy =
        FramePolicy::new(2, 64, 4096, None);
    let lanes: Vec<ProcessId> = (0..44).map(|_| pid()).collect();
    let mut next = 1u64;
    let mut submit = |policy: &mut FramePolicy, lane: ProcessId, seq: u64| -> Vec<u64> {
        let ids = vec![next, next + 1];
        for (slot, &id) in ids.iter().enumerate() {
            policy.on_fire_enqueued(stamp(lane, seq, slot as u32, 2), Some(lane), id, 1, 1);
        }
        next += 2;
        ids
    };
    let mut queued: Vec<u64> = Vec::new();
    for &lane in &lanes {
        queued.extend(submit(&mut policy, lane, 0));
    }
    let ids: QueuedFireIds = queued.iter().copied().collect();
    assert_eq!(fires(&plan(&mut policy, &ids, Instant::now())).len(), 88);
    policy.on_frame_posted(lanes.clone(), Instant::now());
    policy.on_frame_retired(lanes.clone());
    let mut queued: Vec<u64> = Vec::new();
    let (expected, rest) = lanes.split_at(10);
    let (blocked_full, rest) = rest.split_at(13);
    let (blocked_empty, rest) = rest.split_at(10);
    let (both_empty, suspended_full) = rest.split_at(10);
    for &lane in expected {
        queued.extend(submit(&mut policy, lane, 1));
        queued.extend(submit(&mut policy, lane, 2));
    }
    for &lane in blocked_full {
        queued.extend(submit(&mut policy, lane, 1));
        policy.on_owner_waits(lane);
    }
    for &lane in blocked_empty {
        policy.on_owner_waits(lane);
    }
    for &lane in both_empty {
        policy.on_process_suspend(lane);
        policy.on_owner_waits(lane);
    }
    for &lane in suspended_full {
        queued.extend(submit(&mut policy, lane, 1));
        policy.on_process_suspend(lane);
    }
    // Plan as the scheduler does: each dispatch posted, nothing landing.
    let mut frames = Vec::new();
    for _ in 0..4 {
        let ids: QueuedFireIds = queued.iter().copied().collect();
        match plan(&mut policy, &ids, Instant::now()) {
            FramePlan::Dispatch(waves) => {
                let sent: Vec<u64> = waves.iter().flatten().copied().collect();
                queued.retain(|id| !sent.contains(id));
                let members: Vec<ProcessId> =
                    lanes.iter().copied().filter(|_| true).collect();
                policy.on_frame_posted(members, Instant::now());
                frames.push(sent.len());
            }
            _ => break,
        }
    }
    assert_eq!(frames, vec![48, 20], "front frame of 24 lanes, then the next of 10, at once\n{}", policy.debug_summary());
    assert!(queued.is_empty());
}

/// Frames under pressure: 32 lanes, one frame each a round (one fire
/// at k=1, two slots at k=2), a quarter of the owners blocking on the
/// planner each round and coming back the next. A blocked owner is never
/// waited for: its lanes leave the expected set, and what it submitted
/// rides the frame the remaining expected lanes complete. So under every
/// pattern (no one blocking; blocking once the round is in; blocking
/// right after each own submit) a round is one seal, each seal k
/// dispatches (one per slot, none out of frame), each dispatch a row per
/// lane: as wide as with no one blocking. Prints seals, dispatches per
/// seal and rows per dispatch for each.
#[test]
fn owners_blocking_under_pressure_keep_frames_whole() {
    const LANES: usize = 32;
    const ROUNDS: usize = 20;
    // 0: no one blocks; 1: a quarter block once the round is in; 2: a
    // quarter block each right after its own submit.
    let run = |mode: u8, k: usize| -> (usize, usize, usize) {
        let mut policy = FramePolicy::new(k, 64, 4096, None);
        let lanes: Vec<ProcessId> = (0..LANES).map(|_| pid()).collect();
        let mut blocked: Vec<ProcessId> = Vec::new();
        let mut lane_of: HashMap<u64, ProcessId> = HashMap::new();
        let mut queued: Vec<u64> = Vec::new();
        let (mut seals, mut dispatches, mut rows) = (0usize, 0usize, 0usize);
        let mut next_id = 1u64;
        // A posted frame stays in flight on the device until the next is
        // posted: the guests compute their next pass under it, as a
        // pipelined device serves them, and only the last round's drain
        // lets the device finish (an idle device sealing past a lone
        // computing lane is its own case,
        // `a_lone_computing_lane_is_sealed_past_only_with_the_device_idle`).
        let mut in_flight: Option<Vec<ProcessId>> = None;
        let drain = |policy: &mut FramePolicy,
                     queued: &mut Vec<u64>,
                     lane_of: &HashMap<u64, ProcessId>,
                     in_flight: &mut Option<Vec<ProcessId>>,
                     finish: bool|
         -> (usize, usize, usize) {
            let (mut seals, mut dispatches, mut rows) = (0, 0, 0);
            loop {
                let ids: QueuedFireIds = queued.iter().copied().collect();
                match policy.plan_dispatch(&ids, &HashSet::new(), in_flight.is_some(), Instant::now()) {
                    FramePlan::Dispatch(waves) => {
                        seals += 1;
                        for wave in waves.iter().filter(|wave| !wave.is_empty()) {
                            dispatches += 1;
                            rows += wave.len();
                        }
                        let posted: Vec<u64> = waves.iter().flatten().copied().collect();
                        queued.retain(|id| !posted.contains(id));
                        let mut members: Vec<ProcessId> =
                            posted.iter().map(|id| lane_of[id]).collect();
                        members.sort_unstable();
                        members.dedup();
                        if let Some(done) = in_flight.take() {
                            policy.on_frame_retired(done);
                        }
                        policy.on_frame_posted(members.clone(), Instant::now());
                        *in_flight = Some(members);
                    }
                    _ => match in_flight.take() {
                        Some(done) if finish => policy.on_frame_retired(done),
                        still => {
                            *in_flight = still;
                            return (seals, dispatches, rows);
                        }
                    },
                }
            }
        };
        for round in 0..ROUNDS {
            for owner in blocked.drain(..) {
                policy.on_owner_unblocked(owner);
            }
            // Round 0 is the lanes' arrival: every one known before any
            // blocks, as a serving fleet's lanes are.
            let blocks = |at: usize| mode > 0 && round > 0 && (at + round) % 4 == 0;
            for (at, &lane) in lanes.iter().enumerate() {
                for slot in 0..k {
                    policy.on_fire_enqueued(
                        stamp(lane, round as u64, slot as u32, k as u32),
                        Some(lane),
                        next_id,
                        1,
                        1,
                    );
                    lane_of.insert(next_id, lane);
                    queued.push(next_id);
                    next_id += 1;
                }
                if mode == 2 && blocks(at) {
                    policy.on_owner_waits(lane);
                    blocked.push(lane);
                    let (s, d, r) = drain(&mut policy, &mut queued, &lane_of, &mut in_flight, false);
                    seals += s;
                    dispatches += d;
                    rows += r;
                }
            }
            if mode == 1 {
                for (at, &lane) in lanes.iter().enumerate() {
                    if blocks(at) {
                        policy.on_owner_waits(lane);
                        blocked.push(lane);
                    }
                }
            }
            let (s, d, r) =
                drain(&mut policy, &mut queued, &lane_of, &mut in_flight, round + 1 == ROUNDS);
            seals += s;
            dispatches += d;
            rows += r;
            assert!(queued.is_empty(), "mode {mode} k={k} round {round}: {} fires held", queued.len());
        }
        (seals, dispatches, rows)
    };
    for k in [1, 2] {
        for (mode, what) in [(0, "no one blocking"), (1, "blocking once the round is in"), (2, "blocking right after each own submit")] {
            let (seals, dispatches, rows) = run(mode, k);
            println!(
                "k={k} {what}: {seals} seals, {:.2} dispatches per seal, {:.1} rows per dispatch",
                dispatches as f64 / seals.max(1) as f64,
                rows as f64 / dispatches.max(1) as f64,
            );
            assert_eq!(rows, LANES * ROUNDS * k, "k={k} {what}: every fire dispatched");
            assert_eq!(seals, ROUNDS, "k={k} {what}: one seal a round");
            assert_eq!(dispatches, ROUNDS * k, "k={k} {what}: k dispatches a seal, none out of frame");
        }
    }
}

/// A prompt's owner suspended while its prompt is in flight, its landing
/// never reaching the frame (its fires cancelled with the suspension):
/// the next prompt is sealed and dispatched at once, held behind no
/// prompt that is not executing; so too once a prompt's lane leaves
/// (its set released) before it lands.
#[test]
fn a_prompt_whose_owner_is_suspended_mid_flight_holds_no_later_prompt() {
    let mut policy = FramePolicy::new(1, 64, 4096, None);
    let (a, b, c) = (pid(), pid(), pid());
    policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 1, 8, 1);
    let queued: QueuedFireIds = [1].into_iter().collect();
    assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())), vec![1]);
    policy.on_frame_posted([a], Instant::now());
    assert!(policy.prompts_executing.contains(&a), "a's prompt executes");
    policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 2, 8, 1);
    let queued: QueuedFireIds = [2].into_iter().collect();
    // a's owner is suspended; its landing never comes.
    policy.on_process_suspend(a);
    assert!(policy.prompts_executing.is_empty());
    let planned = plan(&mut policy, &queued, Instant::now());
    assert_eq!(fires(&planned), vec![2], "b's prompt goes");
    policy.on_frame_posted([b], Instant::now());
    // b's lane leaves (its set released) before it lands: c's prompt
    // is held by nothing.
    policy.on_lane_leave(b, Some(b), false);
    policy.on_fire_enqueued(stamp(c, 0, 0, 1), Some(c), 3, 8, 1);
    let queued: QueuedFireIds = [3].into_iter().collect();
    assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())), vec![3]);
}

/// The soak's held frame: k=2, 75 lanes, 18 of them expected, the cap
/// lowered to 128 under prompts cut at 512 before it came down, nothing
/// executing. The decodes go; the prompts are held out whole and named
/// to the cap's owner as owed (`cut_owed`: 512), which is the wake the
/// hold waits on; once the walk raises the cap to it, they seal. Never
/// a hold on a raise nothing asked for.
#[test]
fn frames_held_under_a_lowered_cap_are_owed_a_raise_and_seal_once_it_comes() {
    // Planned past any join hold: a frame goes, or what holds it.
    fn settled(policy: &mut FramePolicy, ids: &QueuedFireIds) -> FramePlan {
        let mut now = Instant::now();
        for _ in 0..100 {
            match plan(policy, ids, now) {
                FramePlan::Hold(wait) => now += wait.max(Duration::from_millis(1)),
                plan => return plan,
            }
        }
        panic!("held for good: {}", policy.debug_summary());
    }
    let mut policy = FramePolicy::new(2, 4096, 1 << 20, None);
    let lanes: Vec<ProcessId> = (0..75).map(|_| pid()).collect();
    // Every lane once, then all but 18 wait on the planner (parked).
    let mut id = 1u64;
    let mut queued = Vec::new();
    for &lane in &lanes {
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), id, 1, 1);
        queued.push(id);
        id += 1;
    }
    let ids: QueuedFireIds = queued.iter().copied().collect();
    let first = fires(&settled(&mut policy, &ids));
    assert_eq!(first.len(), 75);
    policy.on_frame_posted(lanes.clone(), Instant::now());
    policy.on_frame_retired(lanes.clone());
    for &lane in &lanes[18..] {
        policy.on_owner_waits(lane);
    }
    // The cap comes down to 128; six of the 18 had prompts cut at 512.
    policy.cap_fires(128);
    let mut queued = Vec::new();
    let mut prompts = Vec::new();
    for (at, &lane) in lanes[..18].iter().enumerate() {
        let tokens = if at < 6 { 512 } else { 1 };
        policy.on_fire_enqueued(stamp(lane, 1, 0, 1), Some(lane), id, tokens, 1);
        queued.push(id);
        if tokens > 1 {
            prompts.push(id);
        }
        id += 1;
    }
    let ids: QueuedFireIds = queued.iter().copied().collect();
    let decodes = fires(&settled(&mut policy, &ids));
    assert!(decodes.iter().all(|id| !prompts.contains(id)), "the prompts are held out");
    assert_eq!(decodes.len(), 12, "the decodes go");
    policy.on_frame_posted(lanes[6..18].to_vec(), Instant::now());
    policy.on_frame_retired(lanes[6..18].to_vec());
    // The decode lanes submit again, nothing executing: the frame is
    // every expected lane in, the prompts held out under the cap.
    let mut queued = prompts.clone();
    for &lane in &lanes[6..18] {
        policy.on_fire_enqueued(stamp(lane, 2, 0, 1), Some(lane), id, 1, 1);
        queued.push(id);
        id += 1;
    }
    let ids: QueuedFireIds = queued.iter().copied().collect();
    let decodes = fires(&settled(&mut policy, &ids));
    assert!(decodes.iter().all(|id| !prompts.contains(id)), "still held out");
    policy.on_frame_posted(lanes[6..18].to_vec(), Instant::now());
    policy.on_frame_retired(lanes[6..18].to_vec());
    assert_eq!(policy.cut_owed(), 512, "named to the cap's owner");
    assert!(policy.wake_scheduled(), "the raise is the wake it waits on");
    // The walk raises the cap to what is owed: the prompts seal.
    policy.cap_fires(policy.cut_owed());
    let ids: QueuedFireIds = prompts.iter().copied().collect();
    let mut sent = fires(&settled(&mut policy, &ids));
    sent.sort_unstable();
    assert_eq!(sent, prompts);
    assert_eq!(policy.cut_owed(), 0);
}

/// The rung lowered to 8, one lane's process waits for a raise
/// ("enqueue: the activation rung") while three lanes' fronts stand
/// complete, and the frame is held for the fourth.
/// Waiting on the raise, its owner is parked (`parks_here`, as
/// owner-waits): the frame seals with the three, and once the raise
/// comes the fourth submits and serves.
#[test]
fn a_lane_whose_process_waits_for_a_raise_holds_no_frame() {
    fn settled(policy: &mut FramePolicy, ids: &QueuedFireIds) -> FramePlan {
        let mut now = Instant::now();
        for _ in 0..100 {
            match plan(policy, ids, now) {
                FramePlan::Hold(wait) => now += wait.max(Duration::from_millis(1)),
                plan => return plan,
            }
        }
        panic!("held for good: {}", policy.debug_summary());
    }
    let mut policy = FramePolicy::new(1, 4096, 1 << 20, None);
    let lanes: Vec<ProcessId> = (0..4).map(|_| pid()).collect();
    let mut id = 1u64;
    let mut first = Vec::new();
    for &lane in &lanes {
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), id, 1, 1);
        first.push(id);
        id += 1;
    }
    let ids: QueuedFireIds = first.iter().copied().collect();
    assert_eq!(fires(&settled(&mut policy, &ids)).len(), 4);
    policy.on_frame_posted(lanes.clone(), Instant::now());
    policy.on_frame_retired(lanes.clone());
    // Three submit; the fourth's process waits for a raise, parked.
    let mut three = Vec::new();
    for &lane in &lanes[..3] {
        policy.on_fire_enqueued(stamp(lane, 1, 0, 1), Some(lane), id, 1, 1);
        three.push(id);
        id += 1;
    }
    policy.on_owner_waits(lanes[3]);
    let ids: QueuedFireIds = three.iter().copied().collect();
    let mut sent = fires(&settled(&mut policy, &ids));
    sent.sort_unstable();
    assert_eq!(sent, three, "the frame seals with what arrived");
    policy.on_frame_posted(lanes[..3].to_vec(), Instant::now());
    policy.on_frame_retired(lanes[..3].to_vec());
    // The raise comes: its process submits, and it serves.
    policy.on_owner_unblocked(lanes[3]);
    policy.on_fire_enqueued(stamp(lanes[3], 1, 0, 1), Some(lanes[3]), id, 8, 1);
    let ids: QueuedFireIds = [id].into_iter().collect();
    assert_eq!(fires(&settled(&mut policy, &ids)), vec![id], "the fourth serves");
}

/// Mac's verified beam mix under pressure: frames cut at a rung of 64
/// queued, then the cap lowered to 16 while the engine still held 64.
/// Capped at 16 they sat unsealed on a raise the pressure could not
/// fund, nothing submitted for 25 minutes. They seal under the rung the
/// engine holds; once they are gone the cap is granted, and from then
/// nothing wider seals.
#[test]
fn frames_queued_at_a_rung_the_cap_came_down_under_seal_under_the_rung_held() {
    use crate::scheduler::worker::seal_rows;
    fn settled(policy: &mut FramePolicy, ids: &QueuedFireIds) -> FramePlan {
        let mut now = Instant::now();
        for _ in 0..100 {
            match plan(policy, ids, now) {
                FramePlan::Hold(wait) => now += wait.max(Duration::from_millis(1)),
                plan => return plan,
            }
        }
        panic!("held for good: {}", policy.debug_summary());
    }
    let mut policy = FramePolicy::new(1, 4096, 1 << 20, None);
    let lanes: Vec<ProcessId> = (0..4).map(|_| pid()).collect();
    policy.cap_fires(64);
    let mut ids = Vec::new();
    for (at, &lane) in lanes.iter().enumerate() {
        let id = at as u64 + 1;
        policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), id, 64, 1);
        ids.push(id);
    }
    // The cap comes down to 16; the engine holds 64, nothing granted.
    let seal = seal_rows(16, 64, None, policy.queued_widest());
    assert_eq!(seal, 64, "they seal under the rung held");
    policy.cap_fires(seal);
    let queued: QueuedFireIds = ids.iter().copied().collect();
    let mut sent = fires(&settled(&mut policy, &queued));
    sent.sort_unstable();
    assert_eq!(sent, ids, "every lane serves");
    assert_eq!(policy.cut_owed(), 0, "no raise owed");
    policy.on_frame_posted(lanes.clone(), Instant::now());
    policy.on_frame_retired(lanes.clone());
    // Gone, the cap is the seal, and once granted nothing wider seals.
    assert_eq!(seal_rows(16, 64, None, policy.queued_widest()), 16);
    assert_eq!(seal_rows(16, 64, Some(16), 64), 16, "past the grant");
}

/// A frame held out under the cap whose raise is refused by name (it
/// never fits beside its owner's own state) leaves with its fires, for
/// the scheduler to fail: nothing is held for it after, and another
/// owner's held frame is kept.
#[test]
fn a_held_frame_whose_raise_is_refused_leaves_with_its_fires() {
    let mut policy = FramePolicy::new(2, 4096, 1 << 20, None);
    let (lane, other) = (pid(), pid());
    policy.cap_fires(128);
    policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 7, 512, 1);
    policy.on_fire_enqueued(stamp(other, 0, 0, 1), Some(other), 8, 256, 1);
    assert_eq!(policy.cut_owed(), 512, "held, owed");
    assert_eq!(policy.refuse_held(lane), vec![7], "its fires, to fail");
    assert_eq!(policy.cut_owed(), 256, "the other's still held");
    assert!(policy.refuse_held(lane).is_empty(), "nothing held for it after");
}

/// The soak's stall as the scheduler's turns meet the walk: the cap
/// lowered 512 -> 256 -> 128 fast, a prompt cut at 512 queued; a turn
/// reads 128 and holds the prompt out, and before its report of the
/// hold lands (`owe_cut_rows`) the walk raises the rung back and the
/// cap with it, so the raise had no hold to nudge for and the report
/// found the cap already up. The turn reads the cap again after its
/// report (`cap_moved`): the next turn seals the prompt under the
/// current cap at once, nothing in flight, never parked on the 128 it
/// cached.
#[test]
fn a_cap_raised_back_before_the_hold_is_told_seals_on_the_next_turn() {
    const ENGINE: usize = 9401;
    fn settled(policy: &mut FramePolicy, ids: &QueuedFireIds) -> FramePlan {
        let mut now = Instant::now();
        for _ in 0..100 {
            match plan(policy, ids, now) {
                FramePlan::Hold(wait) => now += wait.max(Duration::from_millis(1)),
                plan => return plan,
            }
        }
        panic!("held for good: {}", policy.debug_summary());
    }
    // One scheduler turn: the cap read, the frames planned under it,
    // and the hold told; `between` runs where the walk may move the
    // cap after the read. Whether the turn goes again at once.
    fn turn(
        policy: &mut FramePolicy,
        ids: &QueuedFireIds,
        between: impl FnOnce(),
    ) -> (FramePlan, bool) {
        let cap = crate::store::shared::fire_cap(ENGINE);
        policy.cap_fires(cap as usize);
        let planned = settled(policy, ids);
        between();
        let (owed, by) = policy.cut_owed_by();
        crate::store::shared::owe_cut_rows(ENGINE, owed as u32, by);
        (planned, crate::store::shared::cap_moved(ENGINE, cap))
    }
    let mut policy = FramePolicy::new(2, 4096, 1 << 20, None);
    let lane = pid();
    policy.on_fire_enqueued(stamp(lane, 0, 0, 1), Some(lane), 1, 1, 1);
    let first: QueuedFireIds = [1].into_iter().collect();
    crate::store::shared::set_fire_cap_for_test(ENGINE, 512);
    assert_eq!(fires(&turn(&mut policy, &first, || {}).0), vec![1]);
    policy.on_frame_posted([lane], Instant::now());
    policy.on_frame_retired([lane]);
    // The walk lowers the cap twice, fast; a prompt cut at 512 queues.
    crate::store::shared::set_fire_cap_for_test(ENGINE, 256);
    crate::store::shared::set_fire_cap_for_test(ENGINE, 128);
    policy.on_fire_enqueued(stamp(lane, 1, 0, 1), Some(lane), 2, 512, 1);
    let prompt: QueuedFireIds = [2].into_iter().collect();
    // The turn reads 128 and holds the prompt; the raise lands before
    // the hold is told.
    let (held, again) = turn(&mut policy, &prompt, || {
        crate::store::shared::set_fire_cap_for_test(ENGINE, 512);
    });
    assert!(
        matches!(held, FramePlan::Park),
        "held out under the 128 it read: {}",
        policy.debug_summary()
    );
    assert_eq!(policy.cut_owed(), 512, "owed, told");
    assert!(again, "the cap moved since the read: the scheduler goes again");
    let (sealed, again) = turn(&mut policy, &prompt, || {});
    assert_eq!(fires(&sealed), vec![2], "sealed under the current cap at once");
    assert!(!again);
    crate::store::shared::owe_cut_rows(ENGINE, 0, None);
}

/// Shape (2) as a seeded schedule: lanes submit frames in rounds (one or
/// two frames in flight per lane, landings out of order), some
/// owners then pause briefly in a wait parked nowhere (their lanes stay
/// expected, holding the frame until they submit), and some owners are
/// parked (suspended, or blocked on the planner) with a complete frame
/// queued, some coming back, some sending stragglers. After every plan,
/// with every expected lane in, no complete frame is left unsealed and
/// no parked owner's fire undispatched: what waits on its fires (an
/// eviction's wait on their leases, a bind's settle) waits only on lanes
/// still expected, never on a parked one. Failing seeds print their
/// schedules.
#[test]
fn no_parked_owner_s_complete_frame_waits_on_a_boundary_over_seeded_schedules() {
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            if n == 0 { 0 } else { self.next() % n }
        }
    }
    let mut failures = Vec::new();
    for seed in 0..1000u64 {
        let mut rng = Rng(seed);
        // One frame or two in flight per lane (the dumps' k=2).
        let k = 1 + rng.below(2) as usize;
        // At k=2, frames of one slot or two (a blocked owner's complete
        // next frame two fires, as od-1's were).
        let slots = 1 + rng.below(k as u64) as u32;
        let mut policy = FramePolicy::new(k, 64, 4096, None);
        let lanes: Vec<ProcessId> = (0..2 + rng.below(4)).map(|_| pid()).collect();
        let mut seq = vec![0u64; lanes.len()];
        let mut last = vec![0u64; lanes.len()];
        let mut next_id = 1u64;
        let mut queued: Vec<u64> = Vec::new();
        // 0 submitting, 1 waiting on the planner (silent, awaited), 2 parked.
        let mut state = vec![0u8; lanes.len()];
        let mut parked_fires: Vec<u64> = Vec::new();
        let mut landing_late: Vec<ProcessId> = Vec::new();
        let mut lost: HashSet<ProcessId> = HashSet::new();
        let mut schedule = vec![format!("k={k} slots={slots} lanes={}", lanes.len())];
        let mut failed = None;
        for round in 0..10 {
            for (at, &lane) in lanes.iter().enumerate() {
                // A parked owner's straggler: a fire that raced its
                // park rides the next seal, its lane not expected.
                if state[at] == 2 && rng.below(4) == 0 {
                    for slot in 0..slots {
                        policy.on_fire_enqueued(stamp(lane, seq[at], slot, slots), Some(lane), next_id, 1, 1);
                        queued.push(next_id);
                        parked_fires.push(next_id);
                        last[at] = next_id;
                        next_id += 1;
                    }
                    seq[at] += 1;
                    schedule.push(format!("r{round}: parked lane {at} submits"));
                    continue;
                }
                if state[at] != 0 {
                    continue;
                }
                // Some frames are a prompt's (tokens past its row).
                let tokens = if rng.below(4) == 0 { 8 } else { 1 };
                for slot in 0..slots {
                    policy.on_fire_enqueued(stamp(lane, seq[at], slot, slots), Some(lane), next_id, tokens, 1);
                    queued.push(next_id);
                    last[at] = next_id;
                    next_id += 1;
                }
                seq[at] += 1;
            }
            // A brief wait parked nowhere (a lock, a turn) ends: its
            // owner submits again next round.
            for at in 0..lanes.len() {
                if state[at] == 1 && rng.below(2) == 0 {
                    state[at] = 0;
                    schedule.push(format!("r{round}: lane {at}'s brief wait ends"));
                }
            }
            // Some parked owners come back (their wait ended, or they
            // were restored) and submit again next round.
            for at in 0..lanes.len() {
                if state[at] == 2 && rng.below(4) == 0 {
                    state[at] = 0;
                    policy.on_owner_unblocked(lanes[at]);
                    policy.on_process_resume(lanes[at]);
                    schedule.push(format!("r{round}: lane {at} unparks"));
                }
            }
            // Some owners change state after submitting this round.
            for at in 0..lanes.len() {
                if state[at] == 0 && rng.below(5) == 0 {
                    state[at] = 1;
                    schedule.push(format!("r{round}: lane {at} waits briefly, parked nowhere"));
                } else if state[at] == 0 && rng.below(6) == 0 {
                    state[at] = 2;
                    // Its last-submitted fire, if still queued, is complete.
                    if queued.contains(&last[at]) {
                        parked_fires.push(last[at]);
                    }
                    if rng.below(2) == 0 {
                        policy.on_process_suspend(lanes[at]);
                        // Suspended mid-prompt, its landing may never
                        // reach the frame (its fires cancelled).
                        if rng.below(2) == 0 {
                            landing_late.retain(|lane| *lane != lanes[at]);
                            lost.insert(lanes[at]);
                            schedule.push(format!("r{round}: lane {at}'s landing is lost"));
                        }
                        schedule.push(format!("r{round}: lane {at} suspended"));
                    } else {
                        policy.on_owner_waits(lanes[at]);
                        schedule.push(format!("r{round}: lane {at} waits at a bind"));
                    }
                }
            }
            // Lanes whose landing came late land now.
            policy.on_frame_retired(std::mem::take(&mut landing_late));
            // Pressure lowers the cap under the prompts already cut
            // (they are 8 rows wide).
            if rng.below(4) == 0 {
                policy.cap_fires(4);
                schedule.push(format!("r{round}: the cap comes down to 4"));
            }
            // Plan until it holds; posted fires land (some late).
            for _ in 0..12 {
                let ids: QueuedFireIds = queued.iter().copied().collect();
                match plan(&mut policy, &ids, Instant::now()) {
                    FramePlan::Dispatch(waves) => {
                        let posted: Vec<u64> = waves.iter().flatten().copied().collect();
                        if posted.is_empty() {
                            continue;
                        }
                        queued.retain(|id| !posted.contains(id));
                        parked_fires.retain(|id| !posted.contains(id));
                        let members: Vec<ProcessId> = lanes.clone();
                        policy.on_frame_posted(members.clone(), Instant::now());
                        lost.clear();
                        // Landings out of order: some lanes retire now,
                        // the rest with a later frame.
                        let (now_lanes, later): (Vec<ProcessId>, Vec<ProcessId>) =
                            members.into_iter().partition(|_| rng.below(3) != 0);
                        policy.on_frame_retired(now_lanes);
                        landing_late.extend(later);
                    }
                    // A park with nothing executing waits on an event
                    // that comes: frames queued and none scheduled is
                    // a silent scheduler.
                    FramePlan::Park
                        if policy.has_queued_frames() && !policy.wake_scheduled() =>
                    {
                        failed = Some(format!(
                            "r{round}: parked with frames queued and no wake scheduled: {}",
                            policy.debug_summary()
                        ));
                        break;
                    }
                    // Frames held out under the cap are owed a raise,
                    // told to the cap's owner, whose walk takes it: the
                    // cap rises to what the scheduler says it holds,
                    // and only that (a hold it did not name stays held,
                    // and the invariants below fail).
                    _ if policy.cut_owed() > 0 => {
                        let owed = policy.cut_owed();
                        policy.cap_fires(owed);
                        schedule.push(format!("r{round}: the walk raises the cap to {owed}"));
                    }
                    _ => break,
                }
            }
            if failed.is_some() {
                break;
            }
            // (A seal never holds a prompt behind none executing: asserted
            // where it skips one, in this debug build.)
            // A prompt executing holds the next prompt until it lands:
            // a held frame then waits on that landing, not on nothing.
            let prompt_lands = !policy.prompts_executing.is_empty();
            // A prompt holds others only while it executes: none with
            // no lane in flight, and none whose landing was lost with
            // its owner's suspension (not posted since).
            if !policy.prompts_executing.is_subset(&policy.in_flight_lanes)
                || policy.prompts_executing.iter().any(|lane| lost.contains(lane))
            {
                failed = Some(format!(
                    "r{round}: prompts held behind {:?} with {:?} in flight, {:?} lost",
                    policy.prompts_executing, policy.in_flight_lanes, lost
                ));
                break;
            }
            // The invariant: a frame whose expected lanes have all
            // submitted is sealed (nothing ready waits on nothing).
            let ready: Vec<&LaneState> = policy
                .lanes
                .values()
                .filter(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete))
                .collect();
            let all_in = policy
                .lanes
                .values()
                .filter(|lane| lane.awaited)
                .all(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete));
            // A parked or blocked owner's complete frame waits only on
            // lanes still expected: with all of them in, it is sealed.
            let held_for_nobody = all_in && !prompt_lands && policy.lanes.values().any(|lane| {
                !lane.awaited
                    && lane.owner.is_some_and(|owner| {
                        policy.suspended.contains(&owner) || policy.blocked.contains(&owner)
                    })
                    && lane.frames.front().is_some_and(PendingFrame::is_complete)
            });
            if held_for_nobody {
                failed = Some(format!(
                    "r{round}: a complete front frame of a parked owner is held"
                ));
                break;
            }
            if !ready.is_empty() && all_in && !queued.is_empty() && !prompt_lands {
                failed = Some(format!(
                    "r{round}: every expected lane submitted and {} complete frames unsealed",
                    ready.len()
                ));
                break;
            }
            if let Some(id) = parked_fires.iter().find(|id| all_in && !prompt_lands && queued.contains(id)) {
                failed = Some(format!("r{round}: a parked owner's complete fire {id} not dispatched"));
                break;
            }
        }
        if let Some(why) = failed {
            failures.push(format!("seed {seed}: {why}\n  schedule: {}", schedule.join(" -> ")));
            if failures.len() > 10 {
                break;
            }
        }
    }
    assert!(failures.is_empty(), "{} failing seeds:\n{}", failures.len(), failures.join("\n"));
}

/// A prompt split across a lower: its last piece, cut under the cap
/// before it came down, is kept whole out of the frames sealed under
/// the lowered cap (the frame goes with the other lanes), and its lane
/// owes nothing meanwhile; the cap back, the piece goes and completes.
/// A frame sealed under the old cap goes whole, and the rung is not
/// lowered under it until it has.
fn a_prompt_piece_cut_under_a_cap_since_lowered_waits_whole_and_owes_nothing() {
    let mut policy =
        FramePolicy::new(1, 64, 4096, None);
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

/// Each pipeline's fires post in the order it submitted them: one that
/// would post behind a later fire of its pipeline is refused, naming
/// both; other pipelines keep their own order; a pipeline that leaves
/// is forgotten.
fn a_pipelines_fire_posted_behind_a_later_one_is_refused_by_name() {
    let mut policy = FramePolicy::new(2, 64, 4096, None);
    let (a, b) = (pid(), pid());
    assert_eq!(policy.admit_post(a, 5), Ok(()));
    assert_eq!(policy.admit_post(a, 7), Ok(()));
    assert_eq!(policy.admit_post(b, 6), Ok(()), "another pipeline's order is its own");
    assert_eq!(policy.admit_post(a, 6), Err(7), "6 would post after 7");
    assert_eq!(policy.admit_post(a, 8), Ok(()), "the refused fire moved nothing");
    let reason = out_of_order(a, 6, 7);
    assert!(reason.contains("fire 6") && reason.contains("later fire 7"), "{reason}");
    policy.on_lane_leave(a, None, true);
    assert_eq!(policy.admit_post(a, 1), Ok(()), "a pipeline that left is forgotten");
}

/// A joining frame with lanes in a mixed state (a beam of eight): an
/// owner's lane closes with a frame
/// still queued, so it is kept until that frame goes; the owner blocks
/// on the planner and is unblocked again. Its closed lane is never
/// expected again: the owner's live lane's next frame goes, never held
/// for a lane that will not submit.
fn a_closed_lane_is_never_expected_again_by_its_owner_s_unblock() {
    let mut policy = FramePolicy::new(1, 64, 4096, None);
    let (owner, closing, live) = (pid(), pid(), pid());
    // Both lanes submit; the closing one's frame waits behind a held
    // device (nothing planned yet) as its pipeline closes.
    policy.on_fire_enqueued(stamp(closing, 0, 0, 1), Some(owner), 1, 1, 1);
    policy.on_fire_enqueued(stamp(live, 0, 0, 1), Some(owner), 2, 1, 1);
    policy.on_lane_leave(closing, Some(owner), false);
    let queued: QueuedFireIds = [1, 2].into_iter().collect();
    let mut both = fires(&plan(&mut policy, &queued, Instant::now()));
    both.sort_unstable();
    assert_eq!(both, vec![1, 2], "the closed lane's last frame still goes");
    policy.on_frame_retired([closing, live]);
    // The owner blocks on the planner and is unblocked.
    policy.on_owner_waits(owner);
    policy.on_owner_unblocked(owner);
    policy.on_fire_enqueued(stamp(live, 1, 0, 1), Some(owner), 3, 1, 1);
    let queued: QueuedFireIds = [3].into_iter().collect();
    assert_eq!(
        fires(&plan(&mut policy, &queued, Instant::now())),
        vec![3],
        "the live lane's frame goes: the closed lane is not waited for"
    );
}

/// A lane whose guest parked it (`forward.park()`) comes back only by
/// its own submit: its owner blocking on the planner and unblocked
/// again does not make it expected, and the owner's other lane's next
/// frame goes without it.
fn a_lane_its_guest_parked_is_never_expected_again_by_its_owner_s_unblock() {
    let mut policy = FramePolicy::new(1, 64, 4096, None);
    let (owner, parking, live) = (pid(), pid(), pid());
    policy.on_fire_enqueued(stamp(parking, 0, 0, 1), Some(owner), 1, 1, 1);
    policy.on_fire_enqueued(stamp(live, 0, 0, 1), Some(owner), 2, 1, 1);
    let queued: QueuedFireIds = [1, 2].into_iter().collect();
    assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())).len(), 2);
    policy.on_frame_retired([parking, live]);
    // Its guest parks the lane after its frame 0.
    policy.on_lane_park(parking, 0);
    policy.on_fire_enqueued(stamp(live, 1, 0, 1), Some(owner), 3, 1, 1);
    let queued: QueuedFireIds = [3].into_iter().collect();
    assert_eq!(fires(&plan(&mut policy, &queued, Instant::now())), vec![3]);
    policy.on_frame_retired([live]);
    // The owner blocks on the planner and is unblocked.
    policy.on_owner_waits(owner);
    policy.on_owner_unblocked(owner);
    policy.on_fire_enqueued(stamp(live, 2, 0, 1), Some(owner), 4, 1, 1);
    let queued: QueuedFireIds = [4].into_iter().collect();
    assert_eq!(
        fires(&plan(&mut policy, &queued, Instant::now())),
        vec![4],
        "the live lane's frame goes: the parked lane is not waited for"
    );
    policy.on_frame_retired([live]);
    // Its own submit brings it back, expected like any other.
    policy.on_fire_enqueued(stamp(parking, 1, 0, 1), Some(owner), 5, 1, 1);
    let queued: QueuedFireIds = [5].into_iter().collect();
    assert!(
        !matches!(plan_in_flight(&mut policy, &queued, Instant::now()), FramePlan::Dispatch(_)),
        "back, it waits for the live lane, which is expected"
    );
}

/// A guest blocked on input from outside (a session receive, a timer,
/// an HTTP response) is parked by the host: the frames go without its
/// lane, a frame it submitted before it blocked still goes, and its
/// next submit brings the lane back, awaited like any other.
fn a_lane_whose_guest_waits_on_the_outside_holds_no_frame_and_rejoins_on_its_submit() {
    let mut policy = FramePolicy::new(1, 64, 4096, None);
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
        !matches!(plan_in_flight(&mut policy, &queued, Instant::now()), FramePlan::Dispatch(_)),
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
        !matches!(plan_in_flight(&mut policy, &queued, Instant::now()), FramePlan::Dispatch(_)),
        "its lane is awaited while its guest computes"
    );
}

/// A process whose bind found no room for its guest state (or whose
/// pass waits for a raise of the activation rung) submits nothing
/// until it is through: the frame its lane was awaited in
/// goes without it (else a wide frame is held on one process stuck at
/// bind). Its next submit brings the lane back.
fn a_frame_goes_without_a_lane_whose_process_waits() {
    let mut policy =
        FramePolicy::new(1, 64, 4096, None);
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
        !matches!(plan_in_flight(&mut policy, &queued, t0), FramePlan::Dispatch(_)),
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

    // Bound at last (its wait ends before its guest goes on), it
    // submits: its lane is awaited again.
    policy.on_owner_unblocked(binder);
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
        !matches!(plan_in_flight(&mut policy, &queued, t0), FramePlan::Dispatch(_)),
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

    // A frame is one fire's tokens.
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
        FramePolicy::new(2, 64, 4096, None);
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
    match plan_in_flight(&mut policy, &queued, t0 + Duration::from_secs(60)) {
        FramePlan::Hold(_) | FramePlan::Park => {}
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
        policy.lanes.get(&victim).is_none_or(|lane| !lane.awaited),
        "a suspended owner's arrival must not rejoin the wait-set"
    );

    // The victim's stranded slot rides the healthy lane's frame (its
    // lease has to drain), which seals without waiting for the victim.
    let queued: QueuedFireIds = [200, 300, 301].into_iter().collect();
    let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
        panic!("the boundary must seal without waiting for the victim");
    };
    let mut sent = fires(&FramePlan::Dispatch(waves));
    sent.sort_unstable();
    assert_eq!(sent, vec![200, 300, 301], "one frame: the straggler rides it");

    policy.on_process_resume(victim);
    policy.on_fire_enqueued(stamp(victim, 1, 1, 2), Some(victim), 201, 1, 1);
    policy.on_fire_enqueued(stamp(healthy, 2, 0, 2), Some(healthy), 302, 1, 1);
    policy.on_fire_enqueued(stamp(healthy, 2, 1, 2), Some(healthy), 303, 1, 1);
    let queued: QueuedFireIds = [201, 302, 303].into_iter().collect();
    let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
        panic!("the late slot must seal");
    };
    assert!(
        waves.iter().flatten().any(|&id| id == 302) && waves.iter().flatten().any(|&id| id == 303),
        "the healthy lane's frame goes: {waves:?}"
    );

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

    // b is in, a computes alone and nothing is in flight: holding for a
    // would idle the device, so the frame seals past it (no timer, no
    // kill). a's close then leaves nothing to wait on.
    policy.on_fire_enqueued(stamp(b, 1, 0, 1), Some(b), 92, 1, 1);
    let queued: QueuedFireIds = [92].into_iter().collect();
    let next = plan(&mut policy, &queued, Instant::now());
    assert_eq!(fires(&next), vec![92], "the frame seals past computing lane a");
    assert!(
        policy.debug_summary().contains("sealed past"),
        "the stall dump names lane a sealed past: {}",
        policy.debug_summary()
    );
    policy.on_lane_leave(a, None, false);
    let empty = QueuedFireIds::default();
    assert!(
        !matches!(plan(&mut policy, &empty, Instant::now()), FramePlan::Dispatch(_)),
        "nothing is left to seal once a closes"
    );
}

/// A guest that computes past its frame-mates with nothing in flight is
/// sealed past, never held for good nor terminated: its frame-mates fire,
/// and its next submit rejoins the frame after. While a frame is in
/// flight, the same lane is held for.
fn a_lone_computing_lane_is_sealed_past_only_with_the_device_idle() {
    let mut policy = FramePolicy::new(1, 64, 4096, None);
    let (a, b) = (pid(), pid());
    policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 10, 1, 1);
    policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 11, 1, 1);
    let both: QueuedFireIds = [10, 11].into_iter().collect();
    assert_eq!(fires(&plan(&mut policy, &both, Instant::now())).len(), 2);
    policy.on_frame_posted(vec![a, b], Instant::now());
    policy.on_frame_retired(vec![a, b]);

    // b submits, a computes. With a frame in flight, a is held for.
    policy.on_fire_enqueued(stamp(b, 1, 0, 1), Some(b), 12, 1, 1);
    let only_b: QueuedFireIds = [12].into_iter().collect();
    match policy.plan_dispatch(&only_b, &HashSet::new(), true, Instant::now()) {
        FramePlan::Dispatch(waves) => panic!("held for a while a frame is in flight, got {waves:?}"),
        _ => {}
    }
    // Nothing in flight: the frame seals past a.
    assert_eq!(fires(&plan(&mut policy, &only_b, Instant::now())), vec![12]);
    assert!(policy.debug_summary().contains("sealed past"));
    policy.on_frame_posted(vec![b], Instant::now());
    policy.on_frame_retired(vec![b]);

    // a's next submit rejoins: it is awaited again and fires with b.
    policy.on_fire_enqueued(stamp(a, 1, 0, 1), Some(a), 13, 1, 1);
    policy.on_fire_enqueued(stamp(b, 2, 0, 1), Some(b), 14, 1, 1);
    let rejoined: QueuedFireIds = [13, 14].into_iter().collect();
    let mut got = fires(&plan(&mut policy, &rejoined, Instant::now()));
    got.sort_unstable();
    assert_eq!(got, vec![13, 14], "a rejoins with its next submit");
    assert!(!policy.debug_summary().contains("sealed past"), "its submit clears the mark");
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
        !matches!(plan_in_flight(&mut policy, &queued, Instant::now()), FramePlan::Dispatch(_)),
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

/// Eight lanes that run ahead and one that reads each output before
/// its next pass (a beam on host geometry): while a frame runs, the
/// next goes with the eight, and the slow lane joins the next frame it
/// submits for, its passes in order.
fn a_lane_waiting_on_its_output_holds_no_one_elses_next_frame() {
    use std::sync::atomic::Ordering::Relaxed;
    let stats = Arc::new(SchedulerStats::default());
    let mut policy = FramePolicy::new(1, 64, 4096, Some(Arc::clone(&stats)));
    let fast: Vec<ProcessId> = (0..8).map(|_| pid()).collect();
    let slow = pid();
    let mut queued: Vec<u64> = Vec::new();
    let next_id = std::cell::Cell::new(1u64);
    let submit = |policy: &mut FramePolicy, queued: &mut Vec<u64>, lane: ProcessId, seq: u64| {
        let id = next_id.replace(next_id.get() + 1);
        policy.on_fire_enqueued(stamp(lane, seq, 0, 1), Some(lane), id, 1, 1);
        queued.push(id);
        id
    };
    let queue = |queued: &Vec<u64>| queued.iter().copied().collect::<QueuedFireIds>();

    // Frame 0: every lane's first pass.
    for &lane in fast.iter().chain([&slow]) {
        submit(&mut policy, &mut queued, lane, 0);
    }
    let frame0 = fires(&plan(&mut policy, &queue(&queued), Instant::now()));
    assert_eq!(frame0.len(), 9);
    queued.retain(|id| !frame0.contains(id));
    policy.on_frame_posted(fast.iter().copied().chain([slow]), Instant::now());

    // Frame 0 runs: the fast lanes run ahead; the slow one waits on its
    // pass's output and submits nothing.
    let slow_out0 = WorkItemCompletion::deferred_with_guard(None);
    policy.on_lane_waits(slow, slow_out0.clone());
    for &lane in &fast {
        submit(&mut policy, &mut queued, lane, 1);
    }
    let FramePlan::Dispatch(frame1) =
        policy.plan_dispatch(&queue(&queued), &HashSet::new(), true, Instant::now())
    else {
        panic!("a lane waiting on its output parked the others' next frame");
    };
    let frame1: Vec<u64> = frame1.into_iter().flatten().collect();
    assert_eq!(frame1.len(), 8, "the next frame is the eight that submitted");
    queued.retain(|id| !frame1.contains(id));
    assert_eq!(stats.fire.quorum.frames_past_waiting_lane.load(Relaxed), 1);
    assert_eq!(stats.fire.quorum.frames_parked_on_lane.load(Relaxed), 0);

    // Frame 0 lands: the slow lane's wait ends and it submits its second
    // pass; the fast lanes their third, behind frame 1.
    slow_out0.reject("landed");
    policy.on_frame_retired(fast.iter().copied().chain([slow]));
    policy.on_frame_posted(fast.iter().copied(), Instant::now());
    let slow1 = submit(&mut policy, &mut queued, slow, 1);
    for &lane in &fast {
        submit(&mut policy, &mut queued, lane, 2);
    }
    let frame2 = fires(&policy.plan_dispatch(
        &queue(&queued),
        &HashSet::new(),
        true,
        Instant::now(),
    ));
    assert_eq!(frame2.len(), 9, "the slow lane joins the next frame it is ready for");
    assert!(frame2.contains(&slow1), "its second pass, after its first");
    queued.retain(|id| !frame2.contains(id));

    // A lane that has not submitted and waits on nothing still holds the
    // frame: it is on its way.
    policy.on_frame_retired(fast.iter().copied());
    policy.on_frame_posted(fast.iter().copied().chain([slow]), Instant::now());
    for &lane in &fast[1..] {
        submit(&mut policy, &mut queued, lane, 3);
    }
    assert_eq!(
        policy.plan_dispatch(&queue(&queued), &HashSet::new(), true, Instant::now()),
        FramePlan::Park
    );
    assert_eq!(stats.fire.quorum.frames_parked_on_lane.load(Relaxed), 1);
}
