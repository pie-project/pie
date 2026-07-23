use plex::serde_json::json;
use plex::{Host, Policy, RouteContext, RouteDecision, RoutePlan, State, policy_error};

struct StageAction;

impl Policy for StageAction {
    fn route(ctx: &RouteContext, state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let edge = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .find(|(_, edge)| edge.request_index as usize == request_index);
            if let Some((index, edge)) = edge {
                let target = &ctx.targets[edge.target_index as usize];
                state
                    .request_mut(request.request.request_id.as_str())?
                    .scratch["action_attempts"] = json!(1);
                host.rebalance_request(
                    request.request.request_id.as_str(),
                    target.target_id.as_str(),
                    &format!("stage-{request_index}"),
                )?;
                decisions.push(RouteDecision::Assign(index as u32));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        match ctx
            .requests
            .first()
            .and_then(|request| request.facts["mode"].as_str())
        {
            Some("fallback") => return Err(policy_error("fallback-required", "injected")),
            Some("trap") => panic!("trap after staging actions"),
            _ => {}
        }
        Ok(RoutePlan { decisions })
    }
}

plex::export_policy!(StageAction);
