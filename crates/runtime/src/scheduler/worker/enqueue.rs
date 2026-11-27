//! What the scheduler queues: an item taken into the pending queue, a
//! pipeline's fires refused, and what an instance still has queued.

use super::*;

impl BatchScheduler {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_item(
        pending: &mut PendingQueue,
        terminated_processes: &mut HashSet<ProcessId>,
        in_flight_control: &mut InFlightControls,
        instances: &HashMap<::engine::InstanceId, TrackedInstance>,
        limits: SchedulerLimits,
        page_size: u32,
        stopping: &mut bool,
        frame_policy: &mut FramePolicy,
        item: SchedulerItem,
    ) {
        match item {
            SchedulerItem::Stop => {
                *stopping = true;
            }
            SchedulerItem::Nudge => {}
            SchedulerItem::ExecutionSlotReleased(pid) => {
                frame_policy.on_execution_slot_released(pid);
            }
            SchedulerItem::ProcessQuiesced(pid) => {
                terminated_processes.remove(&pid);
            }
            SchedulerItem::ExecutionSlotConsumed(pid) => {
                frame_policy.on_execution_slot_consumed(pid);
            }
            SchedulerItem::AdmissionQueued(pid) => {
                frame_policy.on_admission_queued(pid);
            }
            SchedulerItem::AdmissionDequeued(pid) => {
                frame_policy.on_admission_dequeued(pid);
            }
            SchedulerItem::PipelineLeave(pid, owner, kind, response) => {
                if kind == LeaveKind::Terminate {
                    if !terminated_processes.insert(pid) {
                        if let Some(response) = response {
                            let _ = response.send(());
                        }
                        return;
                    }
                    frame_policy.on_slotted_terminate(pid);
                    let protected = in_flight_control
                        .iter()
                        .find(|control| control.process_id == Some(pid))
                        .and_then(|control| control.logical_completion.clone());
                    if let Some(completion) = &protected {
                        completion.request_cancel();
                    }
                    Self::reject_pipeline_queued(pending, pid, protected.as_ref());
                }
                match kind {
                    LeaveKind::Close => {
                        frame_policy.on_lane_leave(pid, owner, false);
                    }
                    LeaveKind::Suspend => {
                        frame_policy.on_process_suspend(pid);
                    }
                    LeaveKind::Terminate => {
                        frame_policy.on_lane_leave(pid, owner.or(Some(pid)), true);
                        frame_policy.on_process_leave(pid);
                    }
                }
                if let Some(response) = response {
                    let _ = response.send(());
                }
            }
            SchedulerItem::Launch {
                pending: mut launch,
            } => {
                let validation = AdmissionLimits::new(limits, page_size);
                let rejection = if launch.completion.cancel_requested() {
                    Some("logical fire cancelled before scheduler admission".to_string())
                } else if launch
                    .process_id
                    .is_some_and(|pid| terminated_processes.contains(&pid))
                {
                    Some("process terminated before scheduler admission".to_string())
                } else if !instances.contains_key(&launch.instance_id) {
                    Some(format!(
                        "instance {} is unknown or stale",
                        launch.instance_id
                    ))
                } else if let Some(message) = validation.single_request_limit_error(&launch) {
                    Some(message)
                } else if *stopping {
                    Some("scheduler shutting down".to_string())
                } else {
                    None
                };
                if let Some(message) = rejection {
                    if let Some(stamp) = launch.frame
                        && !launch
                            .process_id
                            .is_some_and(|pid| terminated_processes.contains(&pid))
                    {
                        frame_policy.on_fire_rejected_at_admission(stamp, launch.process_id);
                    }
                    launch.completion.reject_unsubmitted(message);
                } else {
                    launch.request.pads_to = batch::pads_to(&launch, limits.shapes);
                    if frame_policy.single_slot()
                        && launch.frame.is_none()
                        && let Some(lane) = launch.pipeline_id
                    {
                        launch.frame = Some(FrameStamp {
                            lane,
                            seq: launch.logical_fire_id,
                            slot: 0,
                            fires: 1,
                        });
                    }
                    tracing::debug!(
                        "enq fire={} framed={} mask={} masks={} stm={} pipe={}",
                        launch.logical_fire_id,
                        launch.frame.is_some(),
                        launch.request.has_user_mask,
                        has_wire_masks(&launch.request),
                        launch.request.single_token_mode,
                        launch.pipeline_id.is_some(),
                    );
                    if let Some(stamp) = launch.frame {
                        // Read only for the seal trace: off, no lookup.
                        let program = if frame::seal_trace() {
                            instances
                                .get(&launch.instance_id)
                                .map_or(0, |instance| instance.program_id.0)
                        } else {
                            0
                        };
                        frame_policy.on_fire_reading(
                            stamp,
                            launch.process_id,
                            launch.logical_fire_id,
                            launch.request.tokens(),
                            launch.wire_row_count(),
                            launch.request.sequence_readouts(),
                            launch.request.rung_rows(),
                            program,
                        );
                    }
                    Self::queue_attempt(pending, launch);
                }
            }

            SchedulerItem::ProcessResume(pid) => {
                frame_policy.on_process_resume(pid);
            }
            SchedulerItem::OwnerWaits(pid) => {
                frame_policy.on_owner_waits(pid);
            }
            SchedulerItem::OwnerUnblocked(pid) => {
                frame_policy.on_owner_unblocked(pid);
            }
            SchedulerItem::OwnerAway(pid) => {
                frame_policy.on_owner_away(pid);
            }
            SchedulerItem::LaneWaits { lane, completion } => {
                frame_policy.on_lane_waits(lane, completion);
            }
            SchedulerItem::FrameTruncate {
                lane,
                seq,
                submitted,
            } => {
                frame_policy.on_frame_truncated(lane, seq, submitted);
            }
            SchedulerItem::LaneOpen { lane, owner } => {
                if !terminated_processes.contains(&owner) && !terminated_processes.contains(&lane) {
                    frame_policy.on_lane_open(lane, owner);
                }
            }
            SchedulerItem::LanePark { lane, seq } => {
                frame_policy.on_lane_park(lane, seq);
            }
            SchedulerItem::PrepareProgram { program, response } => {
                crate::rt::tab_step("queues PrepareProgram", "the scheduler");
                pending.push_back(QueuedItem::PrepareProgram { program, response });
            }
            SchedulerItem::FinishProgram { prepared, response } => {
                pending.push_back(QueuedItem::FinishProgram { prepared, response });
            }
            SchedulerItem::FireFacts { key, response } => {
                pending.push_back(QueuedItem::FireFacts { key, response });
            }
            SchedulerItem::StandingAfter { binding, response } => {
                pending.push_back(QueuedItem::StandingAfter { binding, response });
            }
            SchedulerItem::RegisterChannelsBind {
                pipeline_id,
                plans,
                bind,
                response,
            } => {
                if pipeline_id.is_some_and(|pid| terminated_processes.contains(&pid)) {
                    Lane::release_wait_slots([bind.pacing_wait_id]);
                    let _ = response.send(Err(anyhow!(
                        "process departed before channel bind admission"
                    )));
                    return;
                }
                frame_policy.on_bind_enqueued(pipeline_id);
                pending.insert_before_closes(QueuedItem::RegisterChannelsBind {
                    pipeline_id,
                    plans,
                    bind,
                    response,
                });
            }
            SchedulerItem::CopyTracked { plan, completion } => {
                pending.push_back(QueuedItem::CopyTracked { plan, completion });
            }
            SchedulerItem::Pool { op, response } => {
                pending.push_back(QueuedItem::Pool { op, response });
            }
            SchedulerItem::HostKv { op, response } => {
                pending.push_back(QueuedItem::HostKv { op, response });
            }
            SchedulerItem::CloseInstance { id, pacing_wait_id } => {
                pending.push_back(QueuedItem::CloseInstance { id, pacing_wait_id });
            }
            SchedulerItem::CloseChannel { id } => Self::queue_close_channel(pending, id),
            SchedulerItem::CloseChannels { ids } => {
                for id in ids {
                    Self::queue_close_channel(pending, id);
                }
            }
        }
    }

    /// Fails the queued launches of `fires` by `why`, unsubmitted: a raise
    /// to the frames they were held in was refused by name.
    pub(super) fn reject_fires_queued(pending: &mut PendingQueue, fires: &[u64], why: &str) {
        if fires.is_empty() {
            return;
        }
        let mut kept = VecDeque::with_capacity(pending.len());
        while let Some(item) = pending.pop_front() {
            match item {
                QueuedItem::Launch(request) if fires.contains(&request.fire_id) => {
                    request
                        .completion
                        .reject_unsubmitted(format!("scheduler: {why}"));
                }
                item => kept.push_back(item),
            }
        }
        pending.replace(kept);
    }

    pub(super) fn reject_pipeline_queued(
        pending: &mut PendingQueue,
        pid: ProcessId,
        protected: Option<&WorkItemCompletion>,
    ) {
        let has_queued = pending.iter().any(|item| match item {
            QueuedItem::Launch(request) => request.process_id == Some(pid),
            QueuedItem::PreLaunchCopy { process_id, .. } => *process_id == Some(pid),
            _ => false,
        });
        if !has_queued {
            return;
        }
        let mut kept = VecDeque::with_capacity(pending.len());
        while let Some(item) = pending.pop_front() {
            let reject = match &item {
                QueuedItem::Launch(request) => {
                    request.process_id == Some(pid)
                        && protected
                            .is_none_or(|completion| !request.completion.same_request(completion))
                }
                QueuedItem::PreLaunchCopy {
                    process_id,
                    logical_completion,
                    ..
                } => {
                    *process_id == Some(pid)
                        && protected
                            .is_none_or(|completion| !logical_completion.same_request(completion))
                }
                _ => false,
            };
            if reject {
                match item {
                    QueuedItem::Launch(request) => {
                        request
                            .completion
                            .reject_unsubmitted("pipeline left while queued");
                    }
                    QueuedItem::PreLaunchCopy {
                        logical_completion, ..
                    } => logical_completion
                        .reject_unsubmitted("pipeline left before pre-launch copy"),
                    _ => unreachable!("rejected item kind checked above"),
                }
            } else {
                kept.push_back(item);
            }
        }
        pending.replace(kept);
    }

    pub(super) fn queue_attempt(pending: &mut PendingQueue, mut request: PendingRequest) {
        let mut copies = Vec::with_capacity(2);
        if let Some(plan) = request.prelaunch_copy.take() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::Kv(plan),
                logical_completion: request.completion.clone(),
                process_id: request.process_id,
                pipeline_id: request.pipeline_id,
            });
        }
        if let Some(plan) = request.prelaunch_state_copy.take() {
            copies.push(QueuedItem::PreLaunchCopy {
                plan: PreLaunchCopy::State(plan),
                logical_completion: request.completion.clone(),
                process_id: request.process_id,
                pipeline_id: request.pipeline_id,
            });
        }
        for copy in copies {
            pending.push_back(copy);
        }
        pending.push_back(QueuedItem::Launch(QueuedLaunch::new(Box::new(request))));
    }

    pub(super) fn instance_has_queued_work(
        pending: &VecDeque<QueuedItem>,
        instance_id: ::engine::InstanceId,
    ) -> bool {
        pending.iter().any(|item| match item {
            QueuedItem::Launch(request) => request.instance_id == instance_id,
            _ => false,
        })
    }

    pub(super) fn queue_close_channel(pending: &mut PendingQueue, id: ::engine::ChannelId) {
        const CLOSE_CHANNEL_BATCH_MAX: usize = 512;
        if let Some(QueuedItem::CloseChannels { ids }) = pending.back_mut()
            && ids.len() < CLOSE_CHANNEL_BATCH_MAX
        {
            ids.push(id);
        } else {
            pending.push_back(QueuedItem::CloseChannels { ids: vec![id] });
        }
    }
}
