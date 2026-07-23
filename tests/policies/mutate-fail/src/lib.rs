use plex::serde_json::json;
use plex::{Host, Policy, RouteContext, RoutePlan, State, policy_error};

struct MutateFail;

impl Policy for MutateFail {
    fn route(ctx: &RouteContext, state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        if let Some(request) = ctx.requests.first() {
            state
                .request_mut(request.request.request_id.as_str())?
                .scratch["should_not_commit"] = json!(true);
        }
        state.shared["should_not_commit"] = json!(true);
        Err(policy_error("fallback-required", "injected fallback"))
    }
}

plex::export_policy!(MutateFail);
