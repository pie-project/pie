use plex::{Host, Policy, RouteContext, RouteDecision, RoutePlan, State};

struct InvalidIndex;

impl Policy for InvalidIndex {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        Ok(RoutePlan {
            decisions: vec![RouteDecision::Assign(u32::MAX); ctx.requests.len()],
        })
    }
}

plex::export_policy!(InvalidIndex);
