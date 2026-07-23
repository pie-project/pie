use plex::serde_json::json;
use plex::{Host, Policy, RouteContext, RouteDecision, RoutePlan, State};

struct RawHelpers;

impl Policy for RawHelpers {
    fn route(ctx: &RouteContext, state: &mut State, host: &Host) -> plex::Result<RoutePlan> {
        let query = host.query_raw("engine.custom-query@1", &json!({"value": 7}))?;
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let edge = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .find(|(_, edge)| edge.request_index as usize == request_index);
            if let Some((index, edge)) = edge {
                let target = &ctx.targets[edge.target_index as usize];
                let action_id = host.action_raw(
                    "pie.request.rebalance@1",
                    &json!({
                        "request_id": request.request.request_id.as_str(),
                        "target_id": target.target_id.as_str(),
                        "idempotency_key": format!("raw-{request_index}")
                    }),
                )?;
                state.shared["raw"] = json!({"query": query, "action_id": action_id.0});
                decisions.push(RouteDecision::Assign(index as u32));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        Ok(RoutePlan { decisions })
    }
}

plex::export_policy!(RawHelpers);
