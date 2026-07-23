//! Preble E2 exploit/explore routing over a sparse feasible graph.

use plex::{Host, Policy, RouteContext, RouteDecision, RoutePlan, State};

struct Preble;

impl Policy for Preble {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let decisions = ctx
            .requests
            .iter()
            .enumerate()
            .map(|(request_index, request)| {
                let edges = ctx
                    .feasible_edges
                    .iter()
                    .enumerate()
                    .filter(|(_, edge)| edge.request_index as usize == request_index)
                    .collect::<Vec<_>>();
                let remaining = request.facts["uncached_tokens"].as_u64().unwrap_or(0);
                let longest = edges
                    .iter()
                    .map(|(_, edge)| edge.facts["cached_tokens"].as_u64().unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                let exploit = longest > remaining;
                edges
                    .into_iter()
                    .max_by_key(|(_, edge)| {
                        if exploit {
                            edge.facts["cached_tokens"].as_i64().unwrap_or(0)
                        } else {
                            -((edge.facts["load_cost"].as_i64().unwrap_or(0)
                                + edge.facts["eviction_cost"].as_i64().unwrap_or(0)
                                + remaining as i64)
                                .max(0))
                        }
                    })
                    .map_or(RouteDecision::Defer, |(index, _)| {
                        RouteDecision::Assign(index as u32)
                    })
            })
            .collect();
        Ok(RoutePlan { decisions })
    }
}

plex::export_policy!(Preble);
