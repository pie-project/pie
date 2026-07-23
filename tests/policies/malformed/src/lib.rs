use plex::{Host, Policy, RouteContext, RoutePlan, State};

struct Malformed;

impl Policy for Malformed {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        Ok(RoutePlan {
            decisions: Vec::with_capacity(ctx.requests.len()),
        })
    }
}

plex::export_policy!(Malformed);
