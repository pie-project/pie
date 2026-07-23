use plex::{Host, Policy, RouteContext, RouteDecision, RoutePlan, State};

struct ActionBadResult;

impl Policy for ActionBadResult {
    fn route(ctx: &RouteContext, state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        if let Some(request) = ctx.requests.first() {
            state.request_mut(request.request.request_id.as_str())?.scratch
                ["should_not_commit"] = plex::serde_json::json!(true);
            if let Some(target) = ctx.targets.first() {
                host.rebalance_request(
                    request.request.request_id.as_str(),
                    target.target_id.as_str(),
                    "bad-result",
                )?;
            }
        }
        Ok(RoutePlan {
            decisions: vec![RouteDecision::Defer; ctx.requests.len().saturating_sub(1)],
        })
    }
}

plex::export_policy!(ActionBadResult);
