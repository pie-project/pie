use plex::serde_json::json;
use plex::{Host, Policy, RouteContext, RouteDecision, RoutePlan, State};

struct QueryAssisted;

impl Policy for QueryAssisted {
    fn route(ctx: &RouteContext, _state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        let observation = host.query_raw(
            "pie.cluster.capacity@1",
            &json!({"model": "example-model"}),
        )?;
        let bias = observation["route_bias"].as_i64().unwrap_or(0);
        let decisions = ctx
            .requests
            .iter()
            .enumerate()
            .map(|(request_index, _)| {
                ctx.feasible_edges
                    .iter()
                    .enumerate()
                    .filter(|(_, edge)| edge.request_index as usize == request_index)
                    .min_by_key(|(_, edge)| {
                        (edge.facts["queue_depth"].as_i64().unwrap_or(i64::MAX) - bias).max(0)
                    })
                    .map_or(RouteDecision::Defer, |(index, _)| {
                        RouteDecision::Assign(index as u32)
                    })
            })
            .collect();
        Ok(RoutePlan { decisions })
    }
}

plex::export_policy!(QueryAssisted);
