use plex::serde_json::json;
use plex::{Host, Policy, ScheduleContext, SchedulePlan, ScheduleSelection, State};

struct AttainedService;

impl Policy for AttainedService {
    fn schedule(
        ctx: &ScheduleContext,
        state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        state.shared["working_set_size"] = json!(state.request_ids().count());
        let mut order = (0..ctx.runnable.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| {
            let request_id = ctx.runnable[index].request.request_id.as_str();
            state
                .request(request_id)
                .ok()
                .and_then(|request| request.facts()["attained_service"].as_u64())
                .unwrap_or(0)
        });
        let mut remaining_requests = ctx.capacity.max_requests;
        let mut remaining_tokens = ctx.capacity.max_total_tokens;
        let mut selections = Vec::new();
        for index in order {
            if remaining_requests == 0 || remaining_tokens == 0 {
                break;
            }
            let candidate = &ctx.runnable[index];
            let budget = u64::from(candidate.max_token_budget).min(remaining_tokens) as u32;
            state
                .request_mut(candidate.request.request_id.as_str())?
                .scratch["schedule_calls"] = json!(
                state
                    .request(candidate.request.request_id.as_str())?
                    .scratch["schedule_calls"]
                    .as_u64()
                    .unwrap_or(0)
                    + 1
            );
            selections.push(ScheduleSelection {
                requests: vec![index as u32],
                token_budgets: vec![budget],
            });
            remaining_requests -= 1;
            remaining_tokens -= u64::from(budget);
        }
        Ok(SchedulePlan { selections })
    }
}

plex::export_policy!(AttainedService);
