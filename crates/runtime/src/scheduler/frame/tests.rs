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
    a_pipelines_fire_posted_behind_a_later_one_is_refused_by_name();
    processes_over_the_execution_permits_joining_together_all_finish();
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
