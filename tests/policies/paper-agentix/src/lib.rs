//! Agentix/Autellix PLAS-style group service scheduling.

use plex::serde_json::json;
use plex::{
    FeedbackContext, FeedbackSubject, Host, Policy, ScheduleContext, SchedulePlan,
    ScheduleSelection, State,
};

struct Agentix;

impl Policy for Agentix {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let candidate = &ctx.runnable[index];
            let service = candidate
                .request
                .group_id
                .as_ref()
                .and_then(|group_id| state.group(group_id.as_str()).ok())
                .and_then(|group| group.scratch["observed_service_us"].as_u64())
                .unwrap_or(0);
            let waiting_us = candidate.facts["waiting_ms"]
                .as_u64()
                .unwrap_or(0)
                .saturating_mul(1000);
            let starved = waiting_us >= service.max(1).saturating_mul(4);
            (!starved, service, index)
        });
        let mut remaining_selections = ctx.capacity.max_selections;
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        let mut selections = Vec::new();
        for index in order {
            if remaining_selections == 0 || remaining_requests == 0 || remaining_tokens == 0 {
                break;
            }
            let candidate = &ctx.runnable[index];
            let budget = u64::from(candidate.max_token_budget).min(remaining_tokens) as u32;
            selections.push(ScheduleSelection {
                requests: vec![index as u32],
                token_budgets: vec![budget],
            });
            remaining_selections -= 1;
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }
        Ok(SchedulePlan { selections })
    }

    fn feedback(ctx: &FeedbackContext, state: &mut State, _host: &Host) -> plex::Result<()> {
        for record in &ctx.records {
            let service = record.facts["service_us"].as_u64().unwrap_or(0);
            match &record.subject {
                FeedbackSubject::WorkGroup(group_id) => {
                    let group = state.group_mut(group_id.as_str())?;
                    group.scratch["observed_service_us"] =
                        json!(group.scratch["observed_service_us"].as_u64().unwrap_or(0) + service);
                }
                FeedbackSubject::Request(request_id) => {
                    let request = state.request_mut(request_id.as_str())?;
                    request.scratch["observed_service_us"] = json!(
                        request.scratch["observed_service_us"].as_u64().unwrap_or(0) + service
                    );
                }
                _ => {}
            }
        }
        Ok(())
    }
}

plex::export_policy!(Agentix);
