//! The scheduler's state, rendered for the stall dump.

use super::*;

impl BatchScheduler {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_debug_dump(
        engine_id: usize,
        pending: &VecDeque<QueuedItem>,
        in_flight_launches: &VecDeque<PendingLaunchBatch>,
        in_flight_control: &InFlightControls,
        instances: &HashMap<::engine::InstanceId, TrackedInstance>,
        frame_policy: &FramePolicy,
    ) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let describe = |request: &PendingRequest| {
            format!(
                "fire {} instance {} pipeline {:?} tracked={} settled={} cancelled={}",
                request.logical_fire_id,
                request.instance_id,
                request.pipeline_id,
                instances.contains_key(&request.instance_id),
                request.completion.is_settled(),
                request.completion.cancel_requested(),
            )
        };
        let _ = writeln!(out, "pending ({}):", pending.len());
        for item in pending {
            let line = match item {
                QueuedItem::Launch(request) => format!("Launch: {}", describe(request)),
                QueuedItem::PreLaunchCopy {
                    plan, pipeline_id, ..
                } => format!("PreLaunchCopy({}) pipeline {pipeline_id:?}", plan.label()),
                QueuedItem::PrepareProgram { .. } => "PrepareProgram".to_string(),
                QueuedItem::FinishProgram { .. } => "FinishProgram".to_string(),
                QueuedItem::FireFacts { key, .. } => format!("FireFacts({key})"),
                QueuedItem::StandingAfter { binding, .. } => {
                    format!("StandingAfter(program {})", binding.program)
                }
                QueuedItem::RegisterChannelsBind { .. } => "RegisterChannelsBind".to_string(),
                QueuedItem::CopyTracked { plan, .. } => format!("CopyTracked({})", plan.label()),
                QueuedItem::HostKv { op, .. } => format!("HostKv({op:?})"),
                QueuedItem::Pool { op, .. } => format!("Pool({op:?})"),
                QueuedItem::CloseInstance { id, .. } => format!("CloseInstance {id}"),
                QueuedItem::CloseChannels { ids } => format!("CloseChannels x{}", ids.len()),
            };
            let _ = writeln!(out, "  {line}");
        }
        let _ = writeln!(out, "in_flight_launches ({}):", in_flight_launches.len());
        for batch in in_flight_launches {
            let state = match &batch.state {
                LaunchState::Posted => "posted".to_string(),
                LaunchState::Accepted(c) => format!("settled={}", c.is_settled()),
                LaunchState::Failed(msg) => format!("failed({msg})"),
            };
            let _ = writeln!(
                out,
                "  batch of {} ({state}, age={:?})",
                batch.requests.len(),
                batch.started.elapsed(),
            );
        }
        if in_flight_control.is_empty() {
            let _ = writeln!(out, "in_flight_control: none");
        }
        // Where each control stands, by whose move it waits on: posted and
        // unanswered, the lane has not taken it (it is in another call,
        // said below); accepted and unsettled, the engine owes its landing
        // (its copy pipeline: a staging ring, a tier write, an event).
        for control in in_flight_control.iter() {
            let age = control.since.elapsed();
            let state = match &control.state {
                ControlSlotState::Posted { id } => {
                    format!("posted(id={id}) {age:?} ago, unanswered: the lane has not taken it")
                }
                ControlSlotState::Ready(c) if c.is_settled() => {
                    format!("settled {age:?} after posting")
                }
                ControlSlotState::Ready(_) => {
                    format!(
                        "accepted, unsettled {age:?} after posting: the engine owes its landing"
                    )
                }
            };
            let _ = writeln!(
                out,
                "in_flight_control: {} ({}) pipeline {:?} {state}",
                control.operation, control.what, control.pipeline_id,
            );
        }
        let _ = writeln!(out, "{}", super::super::lane::lane_now(engine_id));
        let _ = write!(out, "{}", frame_policy.debug_summary());
        out
    }
}
