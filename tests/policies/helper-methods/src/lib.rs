use plex::serde_json::json;
use plex::{Host, Policy, RouteContext, RouteDecision, RoutePlan, State};

struct HelperMethods;

impl Policy for HelperMethods {
    fn route(ctx: &RouteContext, state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        let observation = host.query_raw(
            "pie.cluster.capacity@1",
            &json!({"model": "example-model"}),
        )?;
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let edge = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .find(|(_, edge)| edge.request_index as usize == request_index);
            decisions.push(edge.map_or(RouteDecision::Defer, |(index, _)| {
                RouteDecision::Assign(index as u32)
            }));
            if let Some(target) = edge.map(|(_, edge)| &ctx.targets[edge.target_index as usize]) {
                let action = host.rebalance_request(
                    request.request.request_id.as_str(),
                    target.target_id.as_str(),
                    &format!("helper-{request_index}"),
                )?;
                state.shared["helper_action"] = json!(action.0);
            }
        }
        state.shared["helper_query"] = observation;
        Ok(RoutePlan { decisions })
    }
}

plex::export_policy!(HelperMethods);
