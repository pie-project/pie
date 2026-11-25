//! What the scheduler sends the engine: ready items posted, the queue
//! scanned, a frame's waves dispatched, and a control item posted.

use super::*;

impl BatchScheduler {
    pub(super) fn rotate_launch_for_wave_work(
        pending: &mut PendingQueue,
        allow_slot: bool,
        allow_lifecycle: bool,
    ) -> bool {
        if !matches!(pending.front(), Some(QueuedItem::Launch(_))) {
            return false;
        }
        let Some(run_len) = pending.first_other() else {
            return false;
        };
        let work = &pending[run_len];
        if !(Self::standalone_copy(work)
            || (allow_lifecycle && Self::lifecycle_control(work))
            || (allow_slot && matches!(work, QueuedItem::PreLaunchCopy { .. })))
        {
            return false;
        }
        pending.bring_to_front(run_len);
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_ready_items(
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &mut InFlightControls,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        frame_policy: &mut FramePolicy,
        scan_cache: &mut ScanCache,
        slot_buffer: &mut SlotBuffer,
        stopping: bool,
    ) -> (bool, Option<Duration>) {
        let (mut progress, wait_hint) = Self::dispatch_frame_work(
            scan_cache,
            slot_buffer,
            frame_policy,
            engine_loop,
            replies,
            instances,
            pending,
            in_flight_launches,
            in_flight_control,
            page_size,
            limits,
            stats,
            stopping,
        );
        let mut close_rotations = 0usize;
        let mut rotated = false;
        let hold_closes = !stopping && frame_policy.has_pending_binds();
        let slot_blocks_lifecycle = in_flight_control.holds_launches();
        while let Some(item) = pending.front() {
            match item {
                QueuedItem::Launch(_) => {
                    if Self::rotate_launch_for_wave_work(
                        pending,
                        in_flight_control.copies_only(),
                        !slot_blocks_lifecycle,
                    ) {
                        rotated = true;
                        continue;
                    }
                    break;
                }
                QueuedItem::CloseInstance { id, .. } => {
                    if slot_blocks_lifecycle {
                        break;
                    }
                    let id = *id;
                    if hold_closes {
                        let rot_stop = close_rotations >= pending.len()
                            || !pending.iter().skip(1).any(Self::rotation_target);
                        if rot_stop {
                            break;
                        }
                        close_rotations += 1;
                        let item = pending.pop_front().expect("close front");
                        pending.push_back(item);
                        continue;
                    }
                    let busy = instances
                        .get(&id)
                        .is_some_and(|tracked| tracked.in_flight != 0)
                        || Self::instance_has_queued_work(pending, id);
                    if !busy {
                        let item = pending.pop_front().expect("close front");
                        Self::post_control(
                            engine_loop,
                            replies,
                            instances,
                            in_flight_control,
                            item,
                        );
                        progress = true;
                        continue;
                    }
                    // Launches and this close would swap places until its batch
                    // lands, and a browser lands it only after the worker parks.
                    rotated = false;
                    let rot_stop = close_rotations >= pending.len()
                        || !pending.iter().skip(1).any(|item| {
                            !matches!(
                                item,
                                QueuedItem::CloseInstance { .. } | QueuedItem::CloseChannels { .. }
                            )
                        });
                    if rot_stop {
                        break;
                    }
                    close_rotations += 1;
                    let item = pending.pop_front().expect("close front");
                    pending.push_back(item);
                }
                QueuedItem::CloseChannels { .. } if hold_closes => {
                    let rot_stop = close_rotations >= pending.len()
                        || !pending.iter().skip(1).any(Self::rotation_target);
                    if rot_stop {
                        break;
                    }
                    close_rotations += 1;
                    let item = pending.pop_front().expect("close front");
                    pending.push_back(item);
                }
                _ if !in_flight_control.admits(item) => break,
                _ if !in_flight_launches.is_empty() && !Self::pipe_concurrent_control(item) => {
                    break;
                }
                _ => {
                    let item = pending.pop_front().expect("front item present");
                    Self::post_control(engine_loop, replies, instances, in_flight_control, item);
                    progress = true;
                }
            }
        }
        while in_flight_control.admits_copy() {
            let Some(index) = pending.iter().position(Self::standalone_copy) else {
                break;
            };
            let Some(item) = pending.remove(index) else {
                break;
            };
            Self::post_control(engine_loop, replies, instances, in_flight_control, item);
            progress = true;
        }
        (progress || rotated, wait_hint)
    }

    pub(super) fn post_control(
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        in_flight_control: &mut InFlightControls,
        item: QueuedItem,
    ) {
        match &item {
            QueuedItem::Launch(_) => unreachable!(),
            QueuedItem::PreLaunchCopy {
                logical_completion, ..
            } if logical_completion.is_settled() => return,
            QueuedItem::PreLaunchCopy {
                logical_completion, ..
            } if logical_completion.cancel_requested() => {
                logical_completion
                    .reject_unsubmitted("logical fire cancelled before pre-launch copy");
                return;
            }
            QueuedItem::CloseInstance {
                id, pacing_wait_id, ..
            } => {
                let error = match instances.get(id) {
                    Some(instance) if instance.pacing_wait_id == *pacing_wait_id => {
                        (instance.in_flight != 0).then(|| format!("instance {id} is busy"))
                    }
                    _ => Some(format!("instance {id} is unknown or stale")),
                };
                if let Some(message) = error {
                    tracing::warn!(
                        instance_id = id.0,
                        error = %message,
                        "scheduler close_instance skipped"
                    );
                    return;
                }
            }
            _ => {}
        }
        let id = replies.next_control_id();
        let holds_launches = !Self::standalone_copy(&item);
        match &item {
            QueuedItem::PreLaunchCopy {
                plan,
                logical_completion,
                process_id,
                pipeline_id,
            } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { id },
                    logical_completion: Some(logical_completion.clone()),
                    process_id: *process_id,
                    pipeline_id: *pipeline_id,
                    tracked_completion: None,
                    operation: plan.label(),
                    holds_launches,
                    prelaunch: true,
                    what: plan.describe(),
                    since: crate::rt::Instant::now(),
                });
            }
            QueuedItem::CopyTracked { completion, plan } => {
                in_flight_control.push(PendingControl {
                    state: ControlSlotState::Posted { id },
                    logical_completion: None,
                    process_id: None,
                    pipeline_id: None,
                    tracked_completion: Some(completion.clone()),
                    operation: "tracked KV copy",
                    holds_launches,
                    prelaunch: false,
                    what: plan.describe(),
                    since: crate::rt::Instant::now(),
                });
            }
            _ => {}
        }
        replies.post_control(id, engine_loop.control(item));
    }

    pub(super) fn scan_queue<'a>(
        cache: &'a mut ScanCache,
        pending: &PendingQueue,
        stopping: bool,
    ) -> &'a QueueScan {
        if cache.taken_at == Some((pending.epoch(), stopping)) {
            return &cache.scan;
        }
        debug_assert_eq!(
            pending.out_of_order(),
            None,
            "a launch queued behind a later one of its pipeline"
        );
        let scan = &mut cache.scan;
        scan.clear();
        for item in pending.iter() {
            match item {
                QueuedItem::Launch(launch) => {
                    if stopping {
                        scan.drain_eligible.push(launch.fire_id);
                    }
                    // A launch never overtakes a copy its pipeline queued
                    // ahead of it: the copy fills what the launch reads.
                    // Nor a framed launch its pipeline queued ahead of it:
                    // what that fire writes (its kv, its rings), this one
                    // reads. It rides once that one's frame has gone.
                    let behind_copy = launch.pipeline_id.is_some_and(|pipeline| {
                        scan.blocked_lanes.contains(&pipeline)
                            || scan.framed_ahead.contains(&pipeline)
                    });
                    if launch.framed {
                        scan.queued_ids.push(launch.fire_id);
                        if let Some(pipeline) = launch.pipeline_id {
                            scan.framed_ahead.insert(pipeline);
                        }
                    } else if launch.solo {
                        // Fired alone: the first such, and with no rider.
                        if !behind_copy && scan.untracked.is_empty() {
                            scan.untracked.push(launch.fire_id);
                            scan.riding.insert(None);
                            scan.solo = true;
                        }
                    } else if scan.solo {
                        // Behind a solo fire: the next batch.
                    } else if !behind_copy && scan.riding.insert(launch.pipeline_id) {
                        // Pieces of different prompts' passes (cut at their
                        // checkpoints) share a fire rather than taking one each.
                        scan.untracked.push(launch.fire_id);
                    }
                }
                QueuedItem::PreLaunchCopy {
                    pipeline_id: Some(pipeline_id),
                    ..
                } => {
                    scan.blocked_lanes.insert(*pipeline_id);
                }
                _ => {}
            }
        }
        scan.queued_ids.seal();
        cache.taken_at = Some((pending.epoch(), stopping));
        &cache.scan
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_frame_work(
        scan_cache: &mut ScanCache,
        slot_buffer: &mut SlotBuffer,
        frame_policy: &mut FramePolicy,
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        in_flight_control: &InFlightControls,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        stopping: bool,
    ) -> (bool, Option<Duration>) {
        let mut progress = false;
        let mut wait_hint: Option<Duration> = None;
        let merge_hint = |hint: &mut Option<Duration>, hold: Duration| {
            *hint = Some(hint.map_or(hold, |old| old.min(hold)));
        };
        loop {
            if in_flight_control.holds_launches() {
                tracing::debug!("hold control in_flight={}", in_flight_launches.len(),);
                if in_flight_launches.is_empty() {
                    stats
                        .fire
                        .quorum
                        .idle_break_control
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                break;
            }
            if in_flight_launches.len() >= frame::configured_dispatch_depth() {
                tracing::debug!(
                    "hold depth in_flight={} depth={}",
                    in_flight_launches.len(),
                    frame::configured_dispatch_depth(),
                );
                if in_flight_launches.is_empty() {
                    stats
                        .fire
                        .quorum
                        .idle_break_depth
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                break;
            }
            let now = Instant::now();
            let scan = Self::scan_queue(scan_cache, pending, stopping);
            tracing::debug!("plan in_flight={}", in_flight_launches.len(),);
            let mut rider_batch = false;
            let waves: Vec<Vec<u64>> = if stopping {
                if scan.drain_eligible.is_empty() {
                    break;
                }
                vec![scan.drain_eligible.clone()]
            } else if !scan.untracked.is_empty() {
                rider_batch = true;
                tracing::debug!("rider fires={:?}", scan.untracked,);
                vec![scan.untracked.clone()]
            } else {
                match frame_policy.plan_dispatch(
                    &scan.queued_ids,
                    &scan.blocked_lanes,
                    !in_flight_launches.is_empty(),
                    now,
                ) {
                    FramePlan::Dispatch(waves) => {
                        tracing::debug!(
                            "dispatch waves={:?}",
                            waves.iter().map(Vec::len).collect::<Vec<_>>(),
                        );
                        waves
                    }
                    FramePlan::Hold(hold) => {
                        merge_hint(&mut wait_hint, hold);
                        break;
                    }
                    FramePlan::Park => break,
                }
            };
            let (frame_progress, posted) = Self::post_frame(
                slot_buffer,
                frame_policy,
                engine_loop,
                replies,
                instances,
                pending,
                in_flight_launches,
                page_size,
                limits,
                stats,
                &waves,
            );
            progress |= frame_progress;
            if !posted {
                if stopping || !frame_progress {
                    break;
                }
                continue;
            }
            // What the frame posted owes its output from here, and only that.
            if let Some(batch) = in_flight_launches.back() {
                frame_policy.on_frame_posted(
                    batch
                        .requests
                        .iter()
                        .filter(|request| request.frame.is_some())
                        .filter_map(|request| request.pipeline_id),
                    now,
                );
            }
            if rider_batch {
                frame_policy.record_rider_wave();
            }
        }
        (progress, wait_hint)
    }

    pub(super) fn admits_to_frame(
        request: &PendingRequest,
        instances: &HashMap<::engine::InstanceId, TrackedInstance>,
    ) -> bool {
        if request.completion.is_settled() || request.completion.cancel_requested() {
            if !request.completion.is_settled() {
                request
                    .completion
                    .reject_unsubmitted("logical fire cancelled before native launch");
            }
            return false;
        }
        if !instances.contains_key(&request.instance_id) {
            request.completion.reject_unsubmitted(format!(
                "instance {} is unknown or stale",
                request.instance_id
            ));
            return false;
        }
        true
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the worker loop's state, borrowed piece by piece on purpose: six of \
                  these are independent `&mut` borrows of fields the caller owns, and \
                  passing them separately is what lets the borrow checker see they are \
                  disjoint. Wrapping them in one `&mut` context struct would collapse \
                  that into a single borrow and stop this from compiling"
    )]
    pub(super) fn post_frame(
        slot_buffer: &mut SlotBuffer,
        frame_policy: &mut FramePolicy,
        engine_loop: &Lane,
        replies: &mut LaneReplies,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        pending: &mut PendingQueue,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        page_size: u32,
        limits: SchedulerLimits,
        stats: &Arc<SchedulerStats>,
        waves: &[Vec<u64>],
    ) -> (bool, bool) {
        let mut progress = false;
        let mut slot_of: HashMap<u64, (usize, usize)> =
            HashMap::with_capacity(waves.iter().map(Vec::len).sum());
        for (index, wave) in waves.iter().enumerate() {
            for (position, &fire_id) in wave.iter().enumerate() {
                slot_of.insert(fire_id, (index, position));
            }
        }
        let mut kept: VecDeque<QueuedItem> = VecDeque::with_capacity(pending.len());
        slot_buffer.resize_with(waves.len(), Vec::new);
        for (slots, wave) in slot_buffer.iter_mut().zip(waves) {
            debug_assert!(slots.iter().all(Option::is_none));
            if slots.len() < wave.len() {
                slots.resize_with(wave.len(), || None);
            }
        }
        let mut collisions: Vec<(usize, Box<PendingRequest>)> = Vec::new();
        while let Some(item) = pending.pop_front() {
            match item {
                // An unframed launch past the pool's activation rung (SPEC
                // decision 6) waits until the rung is back. A framed one was
                // sealed under the cap, and its frame goes whole: the rung is
                // not lowered under a sealed frame (`FramePolicy::cap_fires`).
                QueuedItem::Launch(launch)
                    if !launch.framed
                        && launch.request.request.rung_rows() > limits.max_forward_tokens =>
                {
                    kept.push_back(QueuedItem::Launch(launch));
                }
                QueuedItem::Launch(launch) => match slot_of.get(&launch.fire_id) {
                    Some(&(wave, position)) => {
                        let slot = &mut slot_buffer[wave][position];
                        if slot.is_none() {
                            *slot = Some(launch.into_request());
                        } else {
                            collisions.push((wave, launch.into_request()));
                        }
                    }
                    None => kept.push_back(QueuedItem::Launch(launch)),
                },
                item => kept.push_back(item),
            }
        }
        pending.replace(kept);
        let mut survivors: Vec<Vec<Box<PendingRequest>>> = Vec::with_capacity(waves.len());
        for slots in slot_buffer.iter_mut() {
            let mut kept_wave = Vec::with_capacity(slots.len());
            for request in slots.iter_mut().filter_map(Option::take) {
                if request.request.lanes.is_empty() {
                    Self::refuse_laneless(&request);
                    progress = true;
                } else if Self::admits_to_frame(&request, instances) {
                    kept_wave.push(request);
                } else {
                    progress = true;
                }
            }
            survivors.push(kept_wave);
        }
        for (wave, request) in collisions {
            if request.request.lanes.is_empty() {
                Self::refuse_laneless(&request);
                progress = true;
            } else if Self::admits_to_frame(&request, instances) {
                survivors[wave].push(request);
            } else {
                progress = true;
            }
        }
        // The device runs the waves in order, and a pipeline's fires in a
        // wave in the order they stand: each pipeline's fires post in the
        // order it submitted them, or the one out of order is refused.
        for wave in &mut survivors {
            wave.retain(|request| {
                let Some(pipeline) = request.pipeline_id else {
                    return true;
                };
                match frame_policy.admit_post(pipeline, request.logical_fire_id) {
                    Ok(()) => true,
                    Err(later) => {
                        let reason =
                            super::frame::out_of_order(pipeline, request.logical_fire_id, later);
                        tracing::error!("{reason}");
                        request.completion.reject_unsubmitted(reason);
                        progress = true;
                        false
                    }
                }
            });
        }
        if survivors.iter().all(Vec::is_empty) {
            return (progress, false);
        }
        // Counted before the build moves the lanes into the fire.
        let total_tokens = survivors
            .iter()
            .flatten()
            .map(|req| req.request.tokens())
            .sum::<usize>();
        let carried = survivors
            .iter()
            .flatten()
            .map(|req| req.request.carried as usize)
            .sum::<usize>();
        let (submission, requests) =
            batch::build_frame_submission(survivors, limits, page_size, stats);
        // Every step refused for a shape outside the stated set (a runtime
        // bug, its fires failed by name): nothing to post.
        if submission.steps.is_empty() {
            return (true, false);
        }
        debug_assert!(
            !submission.steps.is_empty()
                && submission
                    .steps
                    .iter()
                    .all(|step| !step.submission.lanes.is_empty()),
            "a fire posted to the engine carries at least one lane in every step"
        );
        // A buffered step takes the stated buffered rung it pads to.
        let padded = submission
            .steps
            .iter()
            .filter_map(|step| {
                let shape = step.submission.shape?;
                let stated = step.shapes?.shapes.get(shape.id as usize)?;
                (stated.treatment == ::engine::Treatment::Buffers).then_some(stated.rung as usize)
            })
            .max()
            .unwrap_or(0);
        let batch_size = requests.len() as u64;
        for request in &requests {
            if let Some(instance) = instances.get_mut(&request.instance_id) {
                instance.in_flight += 1;
            }
        }
        replies.post_launch(engine_loop.fire(submission, total_tokens > batch_size as usize));
        in_flight_launches.push_back(PendingLaunchBatch {
            state: LaunchState::Posted,
            requests,
            started: Instant::now(),
            batch_size,
            total_tokens,
            rung_rows: total_tokens.max(carried).max(padded),
            placement: Vec::new(),
            unfired: Vec::new(),
        });
        (true, true)
    }

    /// A fire with no lanes is never posted (the engine refuses it whole,
    /// failing every fire beside it): it fails alone, by name. None is
    /// formed so; one that reaches here is a runtime bug.
    pub(super) fn refuse_laneless(request: &PendingRequest) {
        let reason = format!(
            "scheduler: fire {} of pipeline {:?} carries no lanes (a runtime bug): it fails alone, \
             never posted",
            request.logical_fire_id, request.pipeline_id
        );
        tracing::error!("{reason}");
        request.completion.reject_unsubmitted(reason);
    }
}
