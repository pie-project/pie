//! Preble E2 exploit/explore routing over a sparse feasible graph.

use plex::{Host, Policy, RouteContext, RouteDecision, RoutePlan, State};

struct Preble;

impl Policy for Preble {
    fn route(ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        let mut target_counts = vec![0u32; ctx.targets.len()];
        let mut decisions = Vec::with_capacity(ctx.requests.len());
        for (request_index, request) in ctx.requests.iter().enumerate() {
            let edges = ctx
                .feasible_edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    edge.request_index as usize == request_index
                        && target_counts[edge.target_index as usize]
                            < ctx.targets[edge.target_index as usize].max_assignments
                })
                .collect::<Vec<_>>();
            let remaining = request.facts["uncached_tokens"].as_u64().unwrap_or(0);
            let longest = edges
                .iter()
                .map(|(_, edge)| edge.facts["cached_tokens"].as_u64().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let exploit = longest > remaining;
            let selected = edges.into_iter().max_by_key(|(_, edge)| {
                if exploit {
                    (
                        edge.facts["cached_tokens"].as_i64().unwrap_or(0),
                        -edge.facts["load_cost"].as_i64().unwrap_or(0),
                    )
                } else {
                    (
                        -((edge.facts["load_cost"].as_i64().unwrap_or(0)
                            + edge.facts["eviction_cost"].as_i64().unwrap_or(0)
                            + edge.facts["miss_prefill_cost"]
                                .as_i64()
                                .unwrap_or(remaining as i64))
                        .max(0)),
                        0,
                    )
                }
            });
            if let Some((index, edge)) = selected {
                target_counts[edge.target_index as usize] += 1;
                decisions.push(RouteDecision::Assign(index as u32));
            } else {
                decisions.push(RouteDecision::Defer);
            }
        }
        Ok(RoutePlan { decisions })
    }
}

plex::export_policy!(Preble);
