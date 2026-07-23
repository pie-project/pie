use plex::serde_json::json;
use plex::{Host, Policy, RouteContext, RouteDecision, RoutePlan, State};

struct LeastLoaded;

impl Policy for LeastLoaded {
    fn route(ctx: &RouteContext, state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        state.shared["route_owner_calls"] =
            json!(state.shared["route_owner_calls"].as_u64().unwrap_or(0) + 1);
        let decisions = ctx
            .requests
            .iter()
            .enumerate()
            .map(|(request_index, _)| {
                ctx.feasible_edges
                    .iter()
                    .enumerate()
                    .filter(|(_, edge)| edge.request_index as usize == request_index)
                    .min_by_key(|(_, edge)| edge.facts["queue_depth"].as_u64().unwrap_or(u64::MAX))
                    .map_or(RouteDecision::Defer, |(index, _)| {
                        RouteDecision::Assign(index as u32)
                    })
            })
            .collect();
        Ok(RoutePlan { decisions })
    }
}

plex::export_policy!(LeastLoaded);
