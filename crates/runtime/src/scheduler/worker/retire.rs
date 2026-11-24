//! What the scheduler takes back: held launches released, launches and
//! control items retired as their completions settle, and replies applied.

use super::*;

impl BatchScheduler {
    /// Fires held for a refused chunk's move whose state has moved go
    /// again, ahead of what is queued, as fires of their own (no frame):
    /// their frame went when they were first sent.
    pub(super) fn release_held(
        held: &mut Vec<HeldForPlacement>,
        pending: &mut PendingQueue,
        engine_id: EngineId,
    ) -> bool {
        let (ready, mut waiting): (Vec<_>, Vec<_>) = std::mem::take(held)
            .into_iter()
            .partition(|batch| !crate::store::shared::refusals_pending(engine_id, &batch.chunks));
        // A move that found no free chunk below for some of its chunks is a
        // need like any other: the planner is asked for that many free
        // chunks, funded as a bind's room (idle cache, parked state down,
        // residents moved), never a silent wait.
        for batch in waiting.iter_mut().filter(|batch| !batch.funding) {
            let stuck = crate::store::shared::refusals_stuck(engine_id, &batch.chunks);
            let owner = batch.requests.iter().find_map(|request| request.process_id);
            if stuck == 0 {
                continue;
            }
            let Some(owner) = owner else {
                continue;
            };
            let Some(planner) = crate::planner::planner_for(0, engine_id) else {
                continue;
            };
            batch.funding = true;
            tracing::debug!(
                engine_id,
                %owner,
                stuck,
                "scheduler: a refused chunk's move lacks free chunks below: the planner makes them"
            );
            crate::rt::spawn(async move {
                if let Err(error) = planner.wait_bind_room(owner, 1, stuck, 0).await {
                    tracing::warn!(%owner, %error, "scheduler: the room for a refused chunk's move was refused");
                }
                crate::store::shared::ask_walk(engine_id);
            });
        }
        *held = waiting;
        if ready.is_empty() {
            return false;
        }
        let mut items = std::mem::take(&mut pending.items);
        for batch in ready.into_iter().rev() {
            for mut request in batch.requests.into_iter().rev() {
                request.frame = None;
                let mut launch = QueuedLaunch::new(request);
                launch.solo = batch.solo;
                items.push_front(QueuedItem::Launch(launch));
            }
        }
        pending.replace(items);
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn retire_ready_launches(
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        stats: &Arc<SchedulerStats>,
        frame_policy: &mut FramePolicy,
        held: &mut Vec<HeldForPlacement>,
        refused_once: &mut HashSet<u32>,
        engine_id: EngineId,
    ) -> bool {
        let mut progress = false;
        while let Some(front) = in_flight_launches.front() {
            let launch_failure = match &front.state {
                LaunchState::Posted => break,
                LaunchState::Failed(message) => Some(message.clone()),
                LaunchState::Accepted(_) => None,
            };
            let result = match &front.state {
                LaunchState::Accepted(completion) => {
                    let Some(result) = completion.check() else {
                        break;
                    };
                    Some(result)
                }
                _ => None,
            };
            let mut retired = in_flight_launches.pop_front().expect("front batch exists");
            tracing::debug!(
                target: "pie::launch",
                lanes = retired.requests.len(),
                tokens = retired.total_tokens,
                fire_us = retired.started.elapsed().as_micros() as u64,
                "fire retired"
            );
            tracing::debug!("retire lanes={}", retired.requests.len(),);
            frame_policy.on_frame_retired(retired.requests.iter().filter_map(|r| r.pipeline_id));
            for request in &retired.requests {
                if let Some(instance) = instances.get_mut(&request.instance_id) {
                    instance.in_flight = instance.in_flight.saturating_sub(1);
                }
            }
            if let Some(message) = launch_failure {
                // Refused for where state lies: nothing was read or
                // written. Its state moves down and its fires go again;
                // none fails. Refused again for chunks already moved once,
                // a runtime bug: said loudly, its fires go again one by
                // one, and only one the engine still refuses alone fails.
                if !retired.placement.is_empty() {
                    // Its fires go again by the lanes the refused steps hand
                    // back: the build moved them out of the requests, and a
                    // request fired again without them is a fire with no
                    // lanes. One a refused lane took out of the frame failed
                    // alone already and goes nowhere again.
                    let restored = batch::restore_lanes(
                        &mut retired.requests,
                        std::mem::take(&mut retired.unfired),
                    );
                    let mut again = Vec::with_capacity(retired.requests.len());
                    for (request, back) in retired.requests.into_iter().zip(restored) {
                        if back {
                            again.push(request);
                        } else if !request.completion.is_settled() {
                            debug_assert!(
                                false,
                                "a fire refused for placement came back with no lanes"
                            );
                            tracing::error!(
                                engine_id,
                                "scheduler: a fire refused for placement came back with no \
                                 lanes to fire again by (a runtime bug): it fails"
                            );
                            request.completion.reject_unsubmitted(format!(
                                "direct launch rejected: {message} (its lanes were not handed back)"
                            ));
                        }
                    }
                    retired.requests = again;
                    if retired.requests.is_empty() {
                        progress = true;
                        continue;
                    }
                    let repeat = retired
                        .placement
                        .iter()
                        .all(|chunk| refused_once.contains(chunk));
                    if !repeat || retired.requests.len() > 1 {
                        if repeat {
                            tracing::error!(
                                engine_id,
                                chunks = ?retired.placement,
                                %message,
                                lanes = retired.requests.len(),
                                "scheduler: the engine refused chunks moved down for it already \
                                 (a runtime bug): its fires go again one by one"
                            );
                        }
                        refused_once.extend(retired.placement.iter().copied());
                        crate::store::shared::evacuate_refused(engine_id, &retired.placement);
                        held.push(HeldForPlacement {
                            chunks: retired.placement,
                            requests: retired.requests,
                            solo: repeat,
                            funding: false,
                        });
                        progress = true;
                        continue;
                    }
                    tracing::error!(
                        engine_id,
                        chunks = ?retired.placement,
                        %message,
                        "scheduler: a fire refused alone for chunks moved down for it already \
                         fails (a runtime bug)"
                    );
                }
                let message = format!("direct launch rejected: {message}");
                for request in &retired.requests {
                    request.completion.reject_unsubmitted(message.clone());
                }
                progress = true;
                continue;
            }
            let result = result.expect("accepted batch carries a settled result");
            match result {
                Ok(()) => {
                    for request in &retired.requests {
                        request.completion.mark_native_retired();
                    }
                    // A fire its step resolved as it landed is resolved.
                    crate::engine::completion::settle_batched(|| {
                        for request in retired
                            .requests
                            .iter()
                            .filter(|r| !r.completion.is_settled())
                        {
                            request.completion.resolve_landed("frame settle");
                        }
                    });
                    stats::record_fire_stats(
                        stats,
                        retired.started.elapsed(),
                        retired.batch_size,
                        retired.total_tokens,
                    )
                }

                Err(err) => {
                    tracing::warn!(?err, "direct launch completion closed before callback");
                    for request in &retired.requests {
                        request.completion.reject(format!(
                            "direct launch batch callback closed before terminal settlement: {err:#}"
                        ));
                        if let Some(instance) = instances.get(&request.instance_id) {
                            instance.wait_slots.close();
                        }
                    }
                }
            }
            progress = true;
        }
        progress
    }

    pub(super) fn retire_ready_control(in_flight_control: &mut InFlightControls) -> bool {
        let mut retired = false;
        let mut index = 0;
        while index < in_flight_control.settling.len() {
            let ready = match &in_flight_control.settling[index].state {
                ControlSlotState::Posted { .. } => None,
                ControlSlotState::Ready(completion) => completion.check(),
            };
            let Some(result) = ready else {
                index += 1;
                continue;
            };
            let pending = in_flight_control.settling.remove(index);
            let operation = pending.operation;
            if let Some(tracked) = pending.tracked_completion.as_ref() {
                tracked.resolve(&result);
            }
            if let Err(ref err) = result {
                tracing::warn!(
                    ?err,
                    operation,
                    "direct control completion closed before callback"
                );
                if let Some(logical) = pending.logical_completion.as_ref() {
                    logical.reject_unsubmitted(format!("pre-launch {operation} failed: {err:#}"));
                }
            }
            retired = true;
        }
        retired
    }

    pub(super) fn accept_launch_reply(
        batch: &mut PendingLaunchBatch,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        result: LaunchResult,
    ) {
        match result {
            Ok(completion) => {
                for request in &batch.requests {
                    if let Some(instance) = instances.get_mut(&request.instance_id) {
                        let epoch = instance.next_target_epoch;
                        request.completion.commit_target_epoch(epoch);
                        instance.next_target_epoch = epoch + 1;
                    }
                }
                batch.state = LaunchState::Accepted(completion);
            }
            Err(refusal) => {
                batch.state = LaunchState::Failed(refusal.message);
                batch.placement = refusal.placement;
                batch.unfired = refusal.unfired;
            }
        }
    }

    pub(super) fn apply_lane_reply(
        reply: LaneReply,
        in_flight_launches: &mut VecDeque<PendingLaunchBatch>,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        in_flight_control: &mut InFlightControls,
        frame_policy: &mut FramePolicy,
        rollback_tx: &SchedTx,
        engine_id: EngineId,
    ) {
        match reply {
            LaneReply::Launch(result) => {
                let batch = in_flight_launches
                    .iter_mut()
                    .find(|batch| matches!(batch.state, LaunchState::Posted))
                    .expect("a launch answer arrives only while its batch is posted");
                Self::accept_launch_reply(batch, instances, result);
            }
            LaneReply::Control { id, commit } => Self::apply_control_commit(
                id,
                commit,
                in_flight_control,
                instances,
                frame_policy,
                rollback_tx,
                engine_id,
            ),
        }
    }

    pub(super) fn apply_control_commit(
        id: u64,
        commit: LaneCommit,
        in_flight_control: &mut InFlightControls,
        instances: &mut HashMap<::engine::InstanceId, TrackedInstance>,
        frame_policy: &mut FramePolicy,
        rollback_tx: &SchedTx,
        engine_id: EngineId,
    ) {
        match commit {
            LaneCommit::None => {}
            LaneCommit::BindFinished { pipeline_id, waits } => {
                frame_policy.on_bind_completed(pipeline_id);
                // Waiting for room, its process submits nothing until the
                // bind lands: its lanes go out of the frame, as a
                // suspension's do, and come back with its next submit.
                if waits && let Some(pid) = pipeline_id {
                    frame_policy.on_owner_waits(pid);
                }
            }
            LaneCommit::BindInstance {
                pipeline_id,
                bound,
                respond:
                    BindRespond {
                        registered,
                        program_id,
                        response,
                    },
            } => {
                frame_policy.on_bind_completed(pipeline_id);
                // Bound: whatever wait its refusals marked it blocked for
                // has ended.
                if let Some(pid) = pipeline_id {
                    frame_policy.on_owner_unblocked(pid);
                }
                if instances.contains_key(&bound.instance_id) {
                    tracing::error!(
                        instance_id = bound.instance_id.0,
                        "bind committed an already-bound instance id"
                    );
                    let error = anyhow!("instance {} is already bound", bound.instance_id);
                    let _ = response.send(Err(error));
                    return;
                }
                let instance_id = bound.instance_id;
                instances.insert(instance_id, TrackedInstance::from_bound(&bound));
                if let Err(Ok((registered, _, bound))) =
                    response.send(Ok((registered, program_id, bound)))
                {
                    tracing::warn!(
                        operation = "register_channels_bind",
                        instance_id = bound.instance_id.0,
                        channel_count = registered.len(),
                        "scheduler cancellation rollback enqueued bound instance and channels"
                    );
                    Lane::release_registered_channel_wait_slots(&registered);
                    let instance_id = bound.instance_id;
                    if rollback_tx
                        .send(SchedulerItem::CloseInstance {
                            id: instance_id,
                            pacing_wait_id: bound.pacing_wait_id,
                        })
                        .is_err()
                    {
                        tracing::error!(
                            operation = "register_channels_bind",
                            instance_id = instance_id.0,
                            "scheduler cancellation rollback close_instance enqueue failed"
                        );
                    }
                    for channel in registered {
                        let channel_id = channel.binding.channel_id;
                        if rollback_tx
                            .send(SchedulerItem::CloseChannel { id: channel_id })
                            .is_err()
                        {
                            tracing::error!(
                                operation = "register_channels_bind",
                                channel_id = channel_id.0,
                                "scheduler cancellation rollback close_channel enqueue failed"
                            );
                        }
                    }
                }
            }
            LaneCommit::CloseInstance { id } => {
                if let Some(instance) = instances.remove(&id) {
                    instance.close_wait_slots();
                }
                // The lane dropped the instance's count; the guest term is
                // built here, off the lane and outside the pool's lock.
                if let Some(shared) =
                    crate::store::registry::try_get(0, engine_id).and_then(|stores| stores.shared)
                {
                    crate::store::shared::programs::fit_fire(&shared);
                }
            }
            LaneCommit::AsyncControl { result } => {
                let Some(index) = in_flight_control.position_posted(id) else {
                    tracing::error!(
                        id,
                        "lane async-control reply without a matching control slot"
                    );
                    return;
                };
                match result {
                    Ok(completion) => {
                        in_flight_control.settling[index].state =
                            ControlSlotState::Ready(completion);
                    }
                    // Refused on the lane (or the lane dropped it): what
                    // waits on the copy is told now, never left waiting on
                    // a completion that cannot come (a residency move held
                    // its items moving for good).
                    Err(error) => {
                        let pending = in_flight_control.settling.remove(index);
                        if let Some(tracked) = pending.tracked_completion.as_ref() {
                            tracked.resolve(&Err(anyhow!("{error}")));
                        }
                        if let Some(logical) = pending.logical_completion.as_ref() {
                            logical.reject_unsubmitted(format!(
                                "pre-launch {} failed: {error}",
                                pending.operation
                            ));
                        }
                    }
                }
            }
        }
    }
}
