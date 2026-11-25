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
    graceful_close_releases_the_wait();
    a_lone_computing_lane_is_sealed_past_only_with_the_device_idle();
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
