use plex::serde_json::json;
use plex::{Host, Policy, RouteContext, RoutePlan, State};

struct Trap;

impl Policy for Trap {
    fn route(ctx: &RouteContext, state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        if let Some(request) = ctx.requests.first() {
            state
                .request_mut(request.request.request_id.as_str())?
                .scratch["should_not_commit"] = json!(true);
        }
        state.shared["should_not_commit"] = json!(true);
        panic!("injected typed PLEX trap")
    }
}

plex::export_policy!(Trap);
