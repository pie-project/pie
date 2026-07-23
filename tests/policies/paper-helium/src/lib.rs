//! Helium cache-aware critical-path scheduling with forced progress.

use plex::{Host, Policy, ScheduleContext, SchedulePlan, ScheduleSelection, State};

struct Helium;

impl Policy for Helium {
    fn schedule(
        ctx: &ScheduleContext,
        _state: &mut State,
        _host: &Host,
    ) -> plex::Result<SchedulePlan> {
        let any_ready = ctx
            .runnable
            .iter()
            .any(|candidate| candidate.facts["ready"].as_bool().unwrap_or(false));
        let forced = ctx
            .runnable
            .iter()
            .enumerate()
            .min_by_key(|(_, candidate)| {
                candidate.facts["earliest_start"]
                    .as_u64()
                    .unwrap_or(u64::MAX)
            })
            .map(|(index, _)| index);
        let selected =
            (ctx.capacity.max_selections > 0
                && ctx.capacity.max_requests > 0
                && ctx.capacity.max_total_tokens > 0)
                .then(|| {
                    ctx.runnable
                        .iter()
                        .enumerate()
                        .filter(|(index, candidate)| {
                            if any_ready {
                                candidate.facts["ready"].as_bool().unwrap_or(false)
                            } else {
                                Some(*index) == forced
                            }
                        })
                        .max_by_key(|(_, candidate)| {
                            (
                                candidate.facts["dependency_depth"].as_u64().unwrap_or(0),
                                candidate.facts["prefix_reuse_tokens"].as_u64().unwrap_or(0),
                                std::cmp::Reverse(
                                    candidate.facts["profiled_token_cost"]
                                        .as_u64()
                                        .unwrap_or(u64::MAX),
                                ),
                            )
                        })
                        .map(|(index, candidate)| ScheduleSelection {
                            requests: vec![index as u32],
                            token_budgets: vec![candidate.max_token_budget.min(
                                u32::try_from(ctx.capacity.max_total_tokens).unwrap_or(u32::MAX),
                            )],
                        })
                })
                .flatten();
        Ok(SchedulePlan {
            selections: selected.into_iter().collect(),
        })
    }
}

plex::export_policy!(Helium);
